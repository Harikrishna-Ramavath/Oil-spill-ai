"""
utils/preprocess.py

This module contains image preprocessing functions and metrics extraction helpers,
including noise reduction, CLAHE contrast enhancement, normalization, and spill area
percentage estimation using adaptive thresholding.
"""

import cv2
import numpy as np
from PIL import Image

def preprocess_image_for_model(image_path_or_pil, target_width, target_height):
    """
    Reads an image (from path or PIL), applies noise reduction, CLAHE contrast enhancement,
    resizes to target dimensions, and returns a preprocessed numpy array ready for prediction.
    """
    # 1. Load image as numpy array
    if isinstance(image_path_or_pil, str):
        img_bgr = cv2.imread(image_path_or_pil)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path_or_pil}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Assuming PIL Image
        img_rgb = np.array(image_path_or_pil.convert('RGB'))

    # 2. Resize to target dimension
    img_resized = cv2.resize(img_rgb, (target_width, target_height))

    # 3. Noise Reduction (Gaussian blur)
    img_blur = cv2.GaussianBlur(img_resized, (3, 3), 0)

    # 4. Contrast Enhancement (CLAHE on L channel of L*a*b* space)
    # This prevents color distortion while enhancing spatial contrast.
    lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    img_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return img_enhanced

def estimate_spill_percentage(pil_image):
    """
    Estimates the percentage of the image area affected by an oil spill.
    Calculated using color thresholding on Value channel (darker patches)
    and contour detection.
    """
    # Convert PIL to BGR OpenCV format
    img_rgb = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Convert to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, _, v_channel = cv2.split(hsv)

    # Apply Gaussian blur to smooth out noise
    v_blur = cv2.GaussianBlur(v_channel, (5, 5), 0)

    # Adaptive dark region segmentation
    # Since oil spills are typically dark, we can use Otsu's thresholding to segment dark clusters,
    # or threshold relative to the image median intensity.
    median_val = np.median(v_blur)
    
    # We threshold values that are significantly darker than the median sea level
    # e.g., value < 0.7 * median_val
    threshold_val = max(10, int(median_val * 0.70))
    _, thresh = cv2.threshold(v_blur, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # Calculate fraction of pixels flagged as spill
    spill_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    percentage = (spill_pixels / total_pixels) * 100.0

    return round(percentage, 2), thresh

def map_severity(spill_percentage):
    """
    Classifies the severity based on the affected area percentage.
    """
    if spill_percentage == 0:
        return "None"
    elif spill_percentage < 10.0:
        return "Low"
    elif spill_percentage < 25.0:
        return "Medium"
    elif spill_percentage < 50.0:
        return "High"
    else:
        return "Very High"
