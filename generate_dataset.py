"""
generate_dataset.py

This script programmatically generates a synthetic dataset representing oil spills
and clean ocean surfaces. It is designed to create a visual representation of satellite/aerial
imagery to train a binary classifier (oil spill vs. no spill) using TensorFlow.
"""

import os
import json
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw

def load_config():
    """Load configuration from config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def generate_ocean_texture(width, height):
    """
    Generates a realistic ocean water background texture using noise and gradients.
    """
    # Base ocean color (deep blue-grey to green-blue)
    base_color = np.array([random.randint(10, 40), random.randint(35, 75), random.randint(55, 95)], dtype=np.uint8)
    img = np.ones((height, width, 3), dtype=np.uint8) * base_color

    # Add micro-textures (perlin-like noise using OpenCV)
    noise = np.zeros((height, width), dtype=np.uint8)
    cv2.randu(noise, 0, 40)
    noise = cv2.GaussianBlur(noise, (5, 5), 0)
    
    # Apply noise as wave texture
    img[:, :, 0] = np.clip(img[:, :, 0] + noise * 0.1 - 2, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1] + noise * 0.12 - 2, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(img[:, :, 2] + noise * 0.15 - 2, 0, 255).astype(np.uint8)

    # Draw sea wave streaks
    for _ in range(random.randint(5, 12)):
        pts = np.array([
            [random.randint(-50, width), random.randint(0, height)],
            [random.randint(0, width + 50), random.randint(0, height)],
            [random.randint(0, width + 50), random.randint(0, height)]
        ], np.int32)
        color_tuple = (int(base_color[0]) + 5, int(base_color[1]) + 8, int(base_color[2]) + 10)
        cv2.polylines(img, [pts], isClosed=False, color=color_tuple, thickness=random.randint(1, 3))

    return img

def add_spill_slick(img):
    """
    Overlays a dark, low-contrast oil slick onto the ocean image.
    Uses irregular polygons and blends them to look organic.
    """
    height, width, _ = img.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Determine spill scale (number of blobs)
    num_blobs = random.randint(1, 4)
    for _ in range(num_blobs):
        # Generate irregular polygon for the slick
        center_x = random.randint(int(width * 0.25), int(width * 0.75))
        center_y = random.randint(int(height * 0.25), int(height * 0.75))
        radius = random.randint(15, 60)
        
        num_vertices = random.randint(8, 16)
        vertices = []
        for i in range(num_vertices):
            angle = (i / num_vertices) * 2 * np.pi
            r = radius * random.uniform(0.5, 1.5)
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            vertices.append([x, y])

        vertices = np.array(vertices, dtype=np.int32)
        cv2.fillPoly(mask, [vertices], 255)

    # Blur the mask to create organic edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # Blend dark oil color using the mask
    # Oil spills are dark grey/metallic black
    oil_color = np.array([random.randint(5, 20), random.randint(10, 25), random.randint(15, 30)], dtype=np.uint8)
    
    # Overlay logic: Pixel = Image * (1 - mask) + Oil * mask
    normalized_mask = mask.astype(float) / 255.0
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * (1.0 - normalized_mask) + oil_color[c] * normalized_mask).astype(np.uint8)

    # Add a metallic sheen border (optional, simulate thin film diffraction)
    # We can create a thin border with slightly altered colors
    sheen_mask = cv2.dilate(mask, np.ones((5,5), np.uint8)) - mask
    sheen_mask = cv2.GaussianBlur(sheen_mask, (5, 5), 0)
    sheen_normalized = sheen_mask.astype(float) / 255.0
    sheen_color = np.array([random.randint(20, 40), random.randint(45, 65), random.randint(40, 60)], dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * (1.0 - sheen_normalized) + sheen_color[c] * sheen_normalized).astype(np.uint8)

    return img

def add_no_spill_features(img):
    """
    Adds non-spill features like clouds or wind gusts to create varied negative samples.
    """
    height, width, _ = img.shape
    
    # Randomly add a cloud or foam patch
    if random.random() > 0.5:
        cloud_mask = np.zeros((height, width), dtype=np.uint8)
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        radius = random.randint(30, 80)
        
        # Simple blurry blob
        cv2.circle(cloud_mask, (center_x, center_y), radius, 255, -1)
        cloud_mask = cv2.GaussianBlur(cloud_mask, (31, 31), 0)
        
        normalized_mask = cloud_mask.astype(float) / 255.0 * 0.4  # Max 40% transparency
        cloud_color = np.array([random.randint(200, 240), random.randint(200, 240), random.randint(210, 250)], dtype=np.uint8)
        
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * (1.0 - normalized_mask) + cloud_color[c] * normalized_mask).astype(np.uint8)
            
    return img

def build_dataset():
    """Build the synthetic training and testing directories."""
    config = load_config()
    dataset_dir = config["dataset_dir"]
    width = config["img_width"]
    height = config["img_height"]

    splits = ["train", "test"]
    categories = ["oil_spill", "no_spill"]
    
    counts = {
        "train": {"oil_spill": 150, "no_spill": 150},
        "test": {"oil_spill": 40, "no_spill": 40}
    }

    print("Generating synthetic dataset...")

    for split in splits:
        for cat in categories:
            dir_path = os.path.join(dataset_dir, split, cat)
            os.makedirs(dir_path, exist_ok=True)
            
            num_images = counts[split][cat]
            print(f"Generating {num_images} images for {split}/{cat}...")
            
            for i in range(num_images):
                # Generate clean ocean texture
                img = generate_ocean_texture(width, height)
                
                # Apply features based on category
                if cat == "oil_spill":
                    img = add_spill_slick(img)
                else:
                    img = add_no_spill_features(img)
                
                # Save the image
                file_path = os.path.join(dir_path, f"img_{i:04d}.png")
                # cv2 uses BGR, but we save as standard image using Pillow
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                pil_img.save(file_path)

    print("Dataset generation completed successfully!")

if __name__ == "__main__":
    build_dataset()
