"""
utils/gradcam.py

This module implements the Grad-CAM (Gradient-weighted Class Activation Mapping)
algorithm to provide model explainability, highlighting the regions of the image
that influenced the AI's classification decision.
"""

import numpy as np
import cv2
import tensorflow as tf

def find_target_layer(model):
    """
    Dynamically finds the last convolutional layer or activation layer in the network.
    """
    # If the first layer of the model is another model (e.g., MobileNetV2 base)
    if hasattr(model, 'layers') and len(model.layers) > 0 and isinstance(model.layers[0], tf.keras.Model):
        base_model = model.layers[0]
    else:
        base_model = model

    # MobileNetV2's final conv activation is named 'out_relu'
    for layer in reversed(base_model.layers):
        if layer.name == 'out_relu':
            return layer.name, base_model
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name, base_model

    # Fallback to the last layer in base model
    return base_model.layers[-1].name, base_model

def generate_gradcam_heatmap(img_array, model, intensity=0.4, colormap=cv2.COLORMAP_JET):
    """
    Generates a Grad-CAM heatmap superimposed on the original image.
    
    Args:
        img_array: Preprocessed numpy image of shape (1, H, W, 3).
        model: Trained Keras model.
        intensity: Transparency factor of the heatmap overlay.
        colormap: OpenCV colormap to apply.
        
    Returns:
        superimposed_img: Superimposed heatmap image.
        heatmap: Grayscale activation mapping matrix.
    """
    # Find last convolutional/activation layer name and base model
    layer_name, base_model = find_target_layer(model)
    
    # 1. Create a sub-model of the base model (always functional, thus stable inputs/outputs)
    base_grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(layer_name).output, base_model.output]
    )

    # 2. Record operations under GradientTape to compute gradients
    with tf.GradientTape() as tape:
        # Run base model forward pass
        last_conv_layer_output, base_output = base_grad_model(img_array)
        # Watch the intermediate convolution output
        tape.watch(last_conv_layer_output)
        
        # Propagate base_output through the head layers of the sequential model manually
        x = base_output
        for layer in model.layers[1:]:
            if isinstance(layer, tf.keras.layers.Dropout):
                continue
            x = layer(x)
            
        # Class score (binary classifier output is prediction probability)
        class_channel = x[:, 0]

    # 3. Compute gradients of the prediction class with respect to target layer feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Global Average Pool the gradients to get channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply each channel in the feature map by its importance weight, then sum
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU to keep positive activations and normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # If the heatmap is NaN (e.g. gradients were 0), return a blank heatmap
    if np.isnan(heatmap).any():
        heatmap = np.zeros(heatmap.shape)

    # 7. Convert original preprocessed image back to uint8 format
    # In MobileNetV2, image is preprocessed to [-1, 1]. Let's deprocess it back to [0, 255]
    img = img_array[0]
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 8. Resize and colorize heatmap
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # OpenCV color map is in BGR, convert to RGB
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    # 9. Superimpose the heatmap on the original image
    superimposed_img = color_heatmap * intensity + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img, heatmap_resized
