"""
predict.py

This script implements the core OilSpillPredictor class, which encapsulates the AI pipeline.
It also serves as a command-line interface (CLI) to perform standalone predictions on images.
"""

import os
import sys
import json
import uuid
import numpy as np
import tensorflow as tf
from PIL import Image
import logging

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project utilities
from utils.preprocess import preprocess_image_for_model, estimate_spill_percentage, map_severity
from utils.gradcam import generate_gradcam_heatmap

class OilSpillPredictor:
    """
    Main predictor class that orchestrates image preprocessing, neural network inference,
    spill percentage calculation, severity mapping, and Grad-CAM generation.
    """
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.model_path = self.config["model_path"]
        self.model = None
        self.load_predictor_model()

    def load_predictor_model(self):
        """Loads the compiled keras model from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at: {self.model_path}. You may need to train it first.")
            return False
        
        try:
            logger.info("Loading TensorFlow model...")
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("Model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, image_path_or_pil, user_id=None, save_outputs=True):
        """
        Runs the full classification and explainability pipeline.
        
        Args:
            image_path_or_pil: Path to file or PIL Image object.
            user_id: Optional ID of the user triggering the prediction.
            save_outputs: If True, saves original and heatmap images to the output folders.
            
        Returns:
            result_dict: Metadata and scoring outputs.
        """
        if self.model is None:
            # Try reloading in case model was trained since startup
            if not self.load_predictor_model():
                raise RuntimeError("Predictor model is not loaded. Please run train.py first.")

        # 1. Preprocess image for the deep learning model input
        w, h = self.config["img_width"], self.config["img_height"]
        preprocessed_rgb = preprocess_image_for_model(image_path_or_pil, w, h)
        
        # Prepare batch input: shape (1, W, H, 3) and scale to [-1, 1] for MobileNetV2
        # Preprocessing of MobileNetV2 is handled inside preprocess_image_for_model which performs CLAHE etc,
        # but we need to normalize to [-1, 1] for the neural network.
        img_batch = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.expand_dims(preprocessed_rgb.astype(np.float32), axis=0)
        )

        # 2. Run inference
        prediction_prob = float(self.model.predict(img_batch, verbose=0)[0][0])
        
        # Check classification label
        threshold = self.config["classification_threshold"]
        is_spill = prediction_prob >= threshold
        label = "Oil Spill" if is_spill else "No Spill"
        confidence = (prediction_prob if is_spill else (1.0 - prediction_prob)) * 100.0

        # Convert input to PIL image for statistics and saving
        if isinstance(image_path_or_pil, str):
            image_name = os.path.basename(image_path_or_pil)
            pil_image = Image.open(image_path_or_pil)
        else:
            image_name = f"uploaded_{uuid.uuid4().hex[:8]}.png"
            pil_image = image_path_or_pil

        # 3. Estimate metrics (Spill Area & Severity)
        if is_spill:
            spill_pct, _ = estimate_spill_percentage(pil_image)
            severity = map_severity(spill_pct)
        else:
            spill_pct = 0.0
            severity = "None"

        # 4. Generate Explainability Heatmap using Grad-CAM
        heatmap_rgb, _ = generate_gradcam_heatmap(img_batch, self.model)
        heatmap_pil = Image.fromarray(heatmap_rgb)

        # 5. Save output images if requested
        original_saved_path = ""
        heatmap_saved_path = ""
        
        if save_outputs:
            # Create subdirs under assets / outputs
            outputs_dir = os.path.join(self.config["base_dir"], "streamlit_app", "assets", "predictions")
            os.makedirs(outputs_dir, exist_ok=True)
            
            unique_id = uuid.uuid4().hex[:12]
            
            # Paths to save
            original_saved_path = os.path.join(outputs_dir, f"{unique_id}_orig.png")
            heatmap_saved_path = os.path.join(outputs_dir, f"{unique_id}_heatmap.png")
            
            # Save original image resized to 500x500 to keep it consistent
            pil_image.resize((500, 500)).save(original_saved_path)
            # Save heatmap resized to 500x500
            heatmap_pil.resize((500, 500)).save(heatmap_saved_path)

        result = {
            "image_name": image_name,
            "prediction_label": label,
            "confidence": confidence,
            "severity": severity,
            "spill_percentage": spill_pct,
            "original_image_path": original_saved_path,
            "heatmap_path": heatmap_saved_path
        }
        
        # Save to database if db module is active and user_id is supplied
        if user_id is not None:
            try:
                from utils.database import save_prediction as db_save
                db_save(
                    user_id=user_id,
                    image_name=image_name,
                    label=label,
                    confidence=confidence,
                    severity=severity,
                    spill_percentage=spill_pct,
                    original_path=original_saved_path,
                    heatmap_path=heatmap_saved_path
                )
                logger.info("Prediction saved to database.")
            except Exception as db_err:
                logger.error(f"Could not save prediction to database: {db_err}")

        return result

if __name__ == "__main__":
    # Standalone CLI execution
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    test_image_path = sys.argv[1]
    if not os.path.exists(test_image_path):
        print(f"Error: File not found at {test_image_path}")
        sys.exit(1)

    predictor = OilSpillPredictor()
    if predictor.model is None:
        print("Error: Predictor model failed to load. Run train.py first to build the model.")
        sys.exit(1)

    try:
        res = predictor.predict(test_image_path, save_outputs=True)
        print("\n=== PREDICTION RESULTS ===")
        print(f"Image File:   {res['image_name']}")
        print(f"Class Label:  {res['prediction_label']}")
        print(f"Confidence:   {res['confidence']:.2f}%")
        print(f"Severity:     {res['severity']}")
        print(f"Spill Area:   {res['spill_percentage']:.2f}%")
        print(f"Saved Original: {res['original_image_path']}")
        print(f"Saved Heatmap:  {res['heatmap_path']}")
        print("==========================\n")
    except Exception as err:
        print(f"Prediction failed: {err}")
        sys.exit(1)
