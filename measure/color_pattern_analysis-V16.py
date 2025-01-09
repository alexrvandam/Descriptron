#!/usr/bin/env python3
"""
Color Pattern Analysis Script Using TPS Transformation

This script performs the following operations:
1. Extracts contours from binary masks.
2. Resamples contours to ensure uniform point count.
3. Performs Procrustes alignment on the resampled contours.
4. Computes the mean shape from aligned contours.
5. Warps color masks into the mean shape using TPS transformation.
6. Extracts interior pixel data based on a common mean shape mask.
7. Performs PCA on the extracted color data.
8. Generates PCA visualizations.

Usage:
    python color_pattern_analysis_tps.py \
        --binary_masks_dir path/to/binary_masks/ \
        --color_masks_dir path/to/color_transformed_masks/ \
        --mask_types fhs normalized fhs_normalized \
        --output_dir path/to/output/ \
        --n_components 2 \
        --visualize \
        --num_points 500
"""
import os
import sys
import argparse
import numpy as np
import cv2
import csv
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import traceback

# Import for TPS transformation
from skimage.transform import PiecewiseAffineTransform, warp

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Color Pattern Analysis with TPS Transformation and PCA')
    parser.add_argument('--binary_masks_dir', required=True, help='Directory containing binary mask images with prefix "binary_mask_".')
    parser.add_argument('--color_masks_dir', required=True, help='Directory containing color-transformed mask images corresponding to binary masks.')
    parser.add_argument('--mask_types', nargs='+', required=True, choices=['fhs', 'normalized', 'fhs_normalized'], help='List of mask types to analyze.')
    parser.add_argument('--output_dir', required=True, help='Directory to save all outputs.')
    parser.add_argument('--n_components', type=int, default=2, help='Number of PCA components (default: 2).')
    parser.add_argument('--visualize', action='store_true', help='Flag to enable visualization of PCA results.')
    parser.add_argument('--num_points', type=int, default=500, help='Number of points to resample each contour to (default: 500).')
    return parser.parse_args()

def find_corresponding_color_mask(base_name, color_masks_dir, mask_type):
    """
    Find the corresponding color-transformed mask for a given base name and mask type.
    """
    color_mask_files = [f for f in os.listdir(color_masks_dir) if os.path.isfile(os.path.join(color_masks_dir, f))]
    possible_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    color_mask_files = [f for f in color_mask_files if any(f.lower().endswith(ext) for ext in possible_extensions)]

    # Define prefix based on mask type
    prefix_map = {
        'fhs': 'fhs_foreground_mask_',
        'normalized': 'normalized_foreground_mask_',
        'fhs_normalized': 'normalized_fhs_foreground_mask_'
    }

    prefix = prefix_map.get(mask_type)
    if not prefix:
        print(f"Error: Unknown mask type '{mask_type}'.")
        return None

    # Match based on prefix and base_name
    matches = [f for f in color_mask_files if f.startswith(prefix) and base_name == f[len(prefix):]]

    if len(matches) == 1:
        print(f"[DEBUG] Using color mask: {matches[0]} for sample: {base_name}")
        return os.path.join(color_masks_dir, matches[0])
    elif len(matches) > 1:
        print(f'Error: Multiple color mask files found for base name "{base_name}" and mask type "{mask_type}": {matches}')
        return None
    else:
        print(f'Error: No color mask file found for base name "{base_name}" and mask type "{mask_type}".')
        return None

def extract_contour(binary_mask_path, contour_save_path):
    """
    Extract the largest contour from a binary mask and save its coordinates to a CSV file.
    """
    print(f"Extracting contour from: {binary_mask_path}")
    # Read the binary mask
    mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Unable to read binary mask at {binary_mask_path}.")
        return False

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Warning: No contours found in binary mask {binary_mask_path}.")
        return False

    # Largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour = largest_contour.reshape(-1, 2)  # Shape: (N, 2)

    # Save contour
    try:
        with open(contour_save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y'])  # Header
            writer.writerows(largest_contour)
        print(f"Saved contour coordinates to: {contour_save_path}")
        return True
    except Exception as e:
        print(f"Error saving contour to {contour_save_path}: {e}")
        traceback.print_exc()
        return False

def resample_contour(contour, num_points=500):
    """
    Resample a contour to have a fixed number of points.
    """
    # Calculate cumulative distance
    distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
    total_dist = cumulative_dist[-1]

    if total_dist == 0:
        print("Warning: Contour has zero total distance. Returning original contour.")
        return contour

    # Equally spaced distances
    equal_dist = np.linspace(0, total_dist, num_points)

    # Interpolate
    resampled_x = np.interp(equal_dist, cumulative_dist, contour[:, 0])
    resampled_y = np.interp(equal_dist, cumulative_dist, contour[:, 1])

    resampled_contour = np.vstack((resampled_x, resampled_y)).T
    return resampled_contour

def generalized_procrustes_analysis(contours, num_points=500):
    """
    Perform Generalized Procrustes Analysis on a list of contours after resampling.
    """
    print("Performing Generalized Procrustes Analysis (GPA)...")
    resampled_contours = [resample_contour(contour, num_points=num_points) for contour in contours]

    mean_shape = resampled_contours[0]
    aligned_contours = [mean_shape]

    for iteration in range(100):
        temp_aligned = []
        for contour in resampled_contours:
            mtx1, mtx2, disparity = procrustes(mean_shape, contour)
            temp_aligned.append(mtx2)
        new_mean_shape = np.mean(temp_aligned, axis=0)
        diff = np.linalg.norm(new_mean_shape - mean_shape)
        print(f"Iteration {iteration+1}: Mean shape change = {diff}")
        if diff < 1e-5:
            print(f"GPA converged after {iteration+1} iterations.")
            aligned_contours = temp_aligned
            break
        mean_shape = new_mean_shape
    else:
        print("GPA did not converge within the maximum number of iterations.")
        aligned_contours = temp_aligned

    return aligned_contours, mean_shape

def compute_tps_transform(src_points, dst_points):
    """
    Compute TPS transformation using corresponding points.
    """
    # Normalize points to [0, 1] range
    src_points_norm = (src_points - src_points.min(axis=0)) / np.ptp(src_points, axis=0)
    dst_points_norm = (dst_points - dst_points.min(axis=0)) / np.ptp(dst_points, axis=0)

    # Scale to image size
    src_points_img = src_points_norm * 499
    dst_points_img = dst_points_norm * 499

    tform = PiecewiseAffineTransform()
    tform.estimate(src_points_img, dst_points_img)
    return tform

def warp_image_tps(image, tform, output_shape=(500, 500)):
    """
    Warp image using TPS transformation.
    """
    warped_image = warp(image, tform, output_shape=output_shape)
    warped_image = (warped_image * 255).astype(np.uint8)
    return warped_image

def extract_interior_pixels(image, mask):
    """
    Extract interior pixel values from an image using a binary mask.
    """
    pixels = image[mask > 0]
    return pixels

def perform_pca_analysis(data_matrix, n_components=2):
    """
    Perform PCA on the standardized data matrix.
    """
    print("Standardizing data...")
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data_matrix)

    print("Performing PCA...")
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_std)
    print(f"PCA completed. Explained variance ratios: {pca.explained_variance_ratio_}")
    return principal_components, pca

def plot_pca(principal_components, sample_names, output_path, visualize=False):
    """
    Plot PCA results with sample names annotated.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.7)

    for i, name in enumerate(sample_names):
        plt.annotate(name, (principal_components[i, 0], principal_components[i, 1]),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.title('PCA of Color Patterns')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"PCA plot saved to: {output_path}")
    if visualize:
        plt.show()
    plt.close()

def main():
    args = parse_args()
    binary_masks_dir = args.binary_masks_dir
    color_masks_dir = args.color_masks_dir
    mask_types = args.mask_types
    output_dir = args.output_dir
    n_components = args.n_components
    visualize_flag = args.visualize
    num_points = args.num_points

    # Validate mask types
    valid_mask_types = ['fhs', 'normalized', 'fhs_normalized']
    for mask_type in mask_types:
        if mask_type not in valid_mask_types:
            print(f"Error: Invalid mask type '{mask_type}'. Valid types are: {valid_mask_types}")
            sys.exit(1)

    # Create output directories for each mask type
    mask_output_dirs = {}
    for mask_type in mask_types:
        mask_output_dirs[mask_type] = os.path.join(output_dir, mask_type)
        os.makedirs(mask_output_dirs[mask_type], exist_ok=True)
        # Create subdirectories
        os.makedirs(os.path.join(mask_output_dirs[mask_type], 'contours'), exist_ok=True)
        os.makedirs(os.path.join(mask_output_dirs[mask_type], 'aligned_masks'), exist_ok=True)

    # Iterate over each mask type
    for mask_type in mask_types:
        print(f"\n=== Processing Mask Type: {mask_type} ===")

        # Define specific output directories
        contours_dir = os.path.join(mask_output_dirs[mask_type], 'contours')
        aligned_masks_dir = os.path.join(mask_output_dirs[mask_type], 'aligned_masks')

        # Step 1: Extract contours from binary masks
        print("Step 1: Extracting contours from binary masks...")
        binary_mask_files = [f for f in os.listdir(binary_masks_dir) if f.startswith('binary_mask_')]
        binary_mask_files = [f for f in binary_mask_files if os.path.isfile(os.path.join(binary_masks_dir, f))]

        if not binary_mask_files:
            print(f"Error: No binary mask files found in {binary_masks_dir}.")
            continue

        contours = []
        sample_names = []
        for bm_file in binary_mask_files:
            base_name = bm_file[len('binary_mask_'):]
            binary_mask_path = os.path.join(binary_masks_dir, bm_file)
            contour_save_path = os.path.join(contours_dir, f'contour_{base_name}.csv')
            success = extract_contour(binary_mask_path, contour_save_path)
            if success:
                try:
                    contour = np.loadtxt(contour_save_path, delimiter=',', skiprows=1)
                    contours.append(contour)
                    sample_names.append(base_name)
                except Exception as e:
                    print(f"Error loading contour from {contour_save_path}: {e}")
                    traceback.print_exc()

        if not contours:
            print(f"Error: No contours were extracted for mask type '{mask_type}'.")
            continue

        # Step 2: Perform Generalized Procrustes Analysis
        print("Step 2: Performing Generalized Procrustes Analysis (GPA)...")
        aligned_contours, mean_shape = generalized_procrustes_analysis(contours, num_points=num_points)

        # Save mean shape
        mean_shape_path = os.path.join(mask_output_dirs[mask_type], 'mean_shape.csv')
        np.savetxt(mean_shape_path, mean_shape, delimiter=',', header='X,Y', comments='')
        print(f"Mean shape saved to: {mean_shape_path}")

        # Create a reference image from the mean shape
        mean_shape_image = np.zeros((500, 500, 3), dtype=np.uint8)
        mean_shape_scaled = mean_shape.copy()
        mean_shape_scaled[:, 0] = (mean_shape_scaled[:, 0] - mean_shape_scaled[:, 0].min()) / np.ptp(mean_shape_scaled[:, 0]) * 499
        mean_shape_scaled[:, 1] = (mean_shape_scaled[:, 1] - mean_shape_scaled[:, 1].min()) / np.ptp(mean_shape_scaled[:, 1]) * 499
        cv2.fillPoly(mean_shape_image, [mean_shape_scaled.astype(np.int32)], (255, 255, 255))

        # Save mean shape image
        mean_shape_image_path = os.path.join(mask_output_dirs[mask_type], 'mean_shape_image.png')
        cv2.imwrite(mean_shape_image_path, mean_shape_image)

        # Create binary mask from mean shape image
        mean_shape_gray = cv2.cvtColor(mean_shape_image, cv2.COLOR_BGR2GRAY)
        _, mean_shape_mask = cv2.threshold(mean_shape_gray, 1, 255, cv2.THRESH_BINARY)

        # Step 3: Warp color masks and extract interior pixels
        print("Step 3: Warping color masks and extracting interior pixels...")
        data_matrix = []
        valid_sample_names = []

        for idx, base_name in enumerate(sample_names):
            color_mask_path = find_corresponding_color_mask(base_name, color_masks_dir, mask_type)
            if not color_mask_path:
                print(f"Skipping sample '{base_name}' due to missing color mask.")
                continue

            # Load color mask image
            color_mask = cv2.imread(color_mask_path)
            if color_mask is None:
                print(f"Error: Unable to read color mask at {color_mask_path}.")
                continue

            # Warp color mask using TPS
            print(f"Warping color mask for sample '{base_name}' using TPS...")
            # Load the aligned contour for this sample
            aligned_contour = aligned_contours[idx]
            # Original contour
            original_contour = contours[idx]

            # Apply scaling to match image coordinates
            src_points = original_contour.copy()
            dst_points = aligned_contour.copy()
            src_points[:, 0] = (src_points[:, 0] - src_points[:, 0].min()) / np.ptp(src_points[:, 0]) * 499
            src_points[:, 1] = (src_points[:, 1] - src_points[:, 1].min()) / np.ptp(src_points[:, 1]) * 499
            dst_points[:, 0] = (dst_points[:, 0] - dst_points[:, 0].min()) / np.ptp(dst_points[:, 0]) * 499
            dst_points[:, 1] = (dst_points[:, 1] - dst_points[:, 1].min()) / np.ptp(dst_points[:, 1]) * 499

            # Compute TPS transform
            tps_tform = compute_tps_transform(src_points, dst_points)

            # Warp image using TPS
            warped_mask = warp_image_tps(color_mask, tps_tform, output_shape=(500,500))

            # Save warped mask
            warped_mask_filename = f'warped_{base_name}'
            warped_mask_path = os.path.join(aligned_masks_dir, warped_mask_filename)
            try:
                cv2.imwrite(warped_mask_path, warped_mask)
                print(f"Warped mask saved to: {warped_mask_path}")
            except Exception as e:
                print(f"Error saving warped mask to {warped_mask_path}: {e}")
                traceback.print_exc()
                continue

            # Extract interior pixels using mean shape mask
            pixels = extract_interior_pixels(warped_mask, mean_shape_mask)
            if pixels.size == 0:
                print(f"Warning: No interior pixels found for sample '{base_name}'.")
                continue
            data_matrix.append(pixels.flatten())
            valid_sample_names.append(base_name)

        if not data_matrix:
            print(f"Error: No data collected for PCA for mask type '{mask_type}'.")
            continue

        # Convert data_matrix to NumPy array
        data_matrix = np.array(data_matrix)
        print(f"Collected data matrix shape for PCA: {data_matrix.shape}")

        # Step 4: Perform PCA
        print("Step 4: Performing PCA on extracted color data...")
        principal_components, pca_model = perform_pca_analysis(data_matrix, n_components=n_components)

        # Save PCA results
        pca_results_path = os.path.join(mask_output_dirs[mask_type], 'pca_results.csv')
        header = ['Sample'] + [f'PC{i+1}' for i in range(n_components)]
        pca_data = np.column_stack((valid_sample_names, principal_components))
        try:
            with open(pca_results_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(pca_data)
            print(f"PCA results saved to: {pca_results_path}")
        except Exception as e:
            print(f"Error saving PCA results to {pca_results_path}: {e}")
            traceback.print_exc()

        # Step 5: Visualization
        if visualize_flag:
            pca_plot_path = os.path.join(mask_output_dirs[mask_type], 'pca_plot.png')
            plot_pca(principal_components, valid_sample_names, pca_plot_path, visualize=True)

    print("\nColor Pattern Analysis completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
