"""
Standalone script to calculate FID between generated images and real images.
"""

import os
import argparse
import torch as th
from PIL import Image
import numpy as np

try:
    from pytorch_fid import fid_score
except ImportError:
    raise ImportError("Please install pytorch-fid: pip install pytorch-fid")

def center_crop(image, target_h=288, target_w=140):
    """Center crop image to target size."""
    h, w = image.shape[:2]
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return image[top:top+target_h, left:left+target_w]

def preprocess_images(image_dir, output_dir, target_size=(288, 140)):
    """
    Preprocess images by center cropping and saving to new directory.
    Returns path to processed directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(image_dir, img_name)
        try:
            img = np.array(Image.open(img_path))
            cropped_img = center_crop(img, *target_size)
            Image.fromarray(cropped_img).save(os.path.join(output_dir, img_name))
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    return output_dir

def calculate_fid(real_dir, gen_dir, batch_size=50, device=None):
    """Calculate FID between real and generated images."""
    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Create temp dirs for processed images
    import tempfile
    with tempfile.TemporaryDirectory() as real_temp, tempfile.TemporaryDirectory() as gen_temp:
        # Preprocess images to ensure consistent size
        print("Preprocessing real images...")
        processed_real = preprocess_images(real_dir, real_temp)
        
        print("Preprocessing generated images...")
        processed_gen = preprocess_images(gen_dir, gen_temp)
        
        print("Calculating FID...")
        fid_value = fid_score.calculate_fid_given_paths(
            [processed_real, processed_gen],
            batch_size=batch_size,
            device=device,
            dims=2048
        )
    
    return fid_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True,
                       help="Directory containing real images")
    parser.add_argument("--gen_dir", type=str, required=True,
                       help="Directory containing generated images")
    parser.add_argument("--batch_size", type=int, default=50,
                       help="Batch size for FID calculation")
    args = parser.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fid_value = calculate_fid(args.real_dir, args.gen_dir, args.batch_size, device)
    print(f"FID score: {fid_value:.2f}")

if __name__ == "__main__":
    main()