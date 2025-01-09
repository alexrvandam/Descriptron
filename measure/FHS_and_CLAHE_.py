import os
import numpy as np
from skimage import io, color
from skimage.segmentation import felzenszwalb
import cv2
import argparse

# Function to perform color segmentation
def fhs_segmentation(image_path, mask_path, save_path):
    # Read the image and the binary mask
    image = io.imread(image_path)
    mask = io.imread(mask_path)
    
    # Apply the mask to the image
    masked_image = np.zeros_like(image)
    masked_image[mask == 255] = image[mask == 255]
    
    # Perform Felzenszwalb segmentation
    segments = felzenszwalb(masked_image, scale=100, sigma=0.5, min_size=50)
    
    # Segment the image based on segmentation
    segmented_img = color.label2rgb(segments, image, kind='avg', bg_label=0)
    
    # Save the segmented image
    io.imsave(save_path, segmented_img)

# Function to normalize the color of an image
def normalize_color(image_path, save_path):
    # Read and normalize the image
    img = cv2.imread(image_path)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img_norm = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    
    # Save the normalized image
    cv2.imwrite(save_path, img_norm)

def parse_args():
    parser = argparse.ArgumentParser(description='Color segmentation and hue normalization on foreground masks.')
    parser.add_argument('--input_dir', required=True, help='Directory containing input mask images.')
    parser.add_argument('--output_dir', required=True, help='Directory to save output images.')
    parser.add_argument('--run_segmentation', action='store_true', help='Flag to run Felzenszwalb segmentation.')
    parser.add_argument('--run_normalization', action='store_true', help='Flag to run hue normalization.')
    parser.add_argument('--visualize', action='store_true', help='Flag to enable visualization of results.')
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    run_segmentation = args.run_segmentation
    run_normalization = args.run_normalization
    visualize = args.visualize

    # If neither flag is set, run both operations by default
    if not run_segmentation and not run_normalization:
        run_segmentation = True
        run_normalization = True

    os.makedirs(output_dir, exist_ok=True)

    # Get all files in the input directory
    all_files = os.listdir(input_dir)

    # Filter binary mask and foreground mask files
    binary_mask_files = [f for f in all_files if f.startswith('binary_mask_')]
    foreground_mask_files = [f for f in all_files if f.startswith('foreground_mask_')]

    # Create a dictionary to map base names to their binary mask and foreground mask files
    file_pairs = {}
    for bm_file in binary_mask_files:
        base_name = bm_file[len('binary_mask_'):]
        file_pairs[base_name] = {'binary_mask': bm_file}

    for fm_file in foreground_mask_files:
        base_name = fm_file[len('foreground_mask_'):]
        if base_name in file_pairs:
            file_pairs[base_name]['foreground_mask'] = fm_file

    # Process each pair of image and mask
    for base_name, files in file_pairs.items():
        if 'binary_mask' in files and 'foreground_mask' in files:
            image_path = os.path.join(input_dir, files['foreground_mask'])
            mask_path = os.path.join(input_dir, files['binary_mask'])
            output_image_name = files['foreground_mask']

            # Initialize the path for the output image
            current_image_path = image_path

            # Perform segmentation if selected
            if run_segmentation:
                fhs_path = os.path.join(output_dir, f"fhs_{output_image_name}")
                fhs_segmentation(current_image_path, mask_path, fhs_path)
                current_image_path = fhs_path  # Update current image path for next operation

            # Perform normalization if selected
            if run_normalization:
                normalized_prefix = 'normalized_'
                if run_segmentation:
                    normalized_prefix = 'normalized_fhs_'
                normalized_path = os.path.join(output_dir, f"{normalized_prefix}{output_image_name}")
                normalize_color(current_image_path, normalized_path)
                current_image_path = normalized_path  # Update current image path for next operation

            if visualize:
                # Placeholder for visualization code
                pass
        else:
            print(f"Warning: Matching files for base name '{base_name}' not found.")

if __name__ == "__main__":
    main()
