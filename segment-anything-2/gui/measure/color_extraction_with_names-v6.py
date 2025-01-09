"""
Color Histogram Analysis Script with Enhanced Visualization and Grouping Functionality

This script performs the following operations:
1. Loads binary masks and corresponding color masks.
2. Extracts interior pixels using binary masks.
3. Computes color histograms from the interior pixels in LAB color space.
4. Performs PCA on the histogram data.
5. Generates PCA visualizations with thumbnails of the color masks.
6. Plots 3D color histograms for each sample.
7. Visualizes the region used to calculate the histogram by drawing a green contour on the color mask.
8. Handles grouping of samples (e.g., species) and computes per-group statistics (mean, SD, range).
9. Appends human-readable color summaries by species using color names.

Usage:
    python color_pattern_analysis-Hist-V6.py \
        --binary_masks_dir path/to/binary_masks/ \
        --color_masks_dir path/to/color_transformed_masks/ \
        --mask_types fhs normalized fhs_normalized \
        --output_dir path/to/output/ \
        --n_components 2 \
        --visualize \
        --grouping_file path/to/grouping_file.csv \
        --append_human_readable
"""

import os
import sys
import argparse
import numpy as np
import cv2
import csv
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import traceback
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

# ======================= #
# === Added Section === #
# ======================= #

# Color Lookup Dictionary (from color_extraction_with_names-v4.py)
color_lookup = {
    (128, 0, 0): 'maroon',
    (139, 0, 0): 'dark red',
    (165, 42, 42): 'brown',
    (178, 34, 34): 'firebrick',
    (220, 20, 60): 'crimson',
    (255, 0, 0): 'red',
    (255, 99, 71): 'tomato',
    (255, 127, 80): 'coral',
    (205, 92, 92): 'indian red',
    (240, 128, 128): 'light coral',
    (233, 150, 122): 'dark salmon',
    (250, 128, 114): 'salmon',
    (255, 160, 122): 'light salmon',
    (255, 69, 0): 'orange red',
    (255, 140, 0): 'dark orange',
    (255, 165, 0): 'orange',
    (255, 215, 0): 'gold',
    (184, 134, 11): 'dark golden rod',
    (218, 165, 32): 'golden rod',
    (238, 232, 170): 'pale golden rod',
    (189, 183, 107): 'dark khaki',
    (240, 230, 140): 'khaki',
    (128, 128, 0): 'olive',
    (255, 255, 0): 'yellow',
    (154, 205, 50): 'yellow green',
    (85, 107, 47): 'dark olive green',
    (107, 142, 35): 'olive drab',
    (124, 252, 0): 'lawn green',
    (127, 255, 0): 'chartreuse',
    (173, 255, 47): 'green yellow',
    (0, 100, 0): 'dark green',
    (0, 128, 0): 'green',
    (34, 139, 34): 'forest green',
    (0, 255, 0): 'lime',
    (50, 205, 50): 'lime green',
    (144, 238, 144): 'light green',
    (152, 251, 152): 'pale green',
    (143, 188, 143): 'dark sea green',
    (0, 250, 154): 'medium spring green',
    (0, 255, 127): 'spring green',
    (46, 139, 87): 'sea green',
    (102, 205, 170): 'medium aqua marine',
    (60, 179, 113): 'medium sea green',
    (32, 178, 170): 'light sea green',
    (47, 79, 79): 'dark slate gray',
    (0, 128, 128): 'teal',
    (0, 139, 139): 'dark cyan',
    (0, 255, 255): 'aqua',
    (224, 255, 255): 'light cyan',
    (0, 206, 209): 'dark turquoise',
    (64, 224, 208): 'turquoise',
    (72, 209, 204): 'medium turquoise',
    (175, 238, 238): 'pale turquoise',
    (127, 255, 212): 'aqua marine',
    (176, 224, 230): 'powder blue',
    (95, 158, 160): 'cadet blue',
    (70, 130, 180): 'steel blue',
    (100, 149, 237): 'corn flower blue',
    (0, 191, 255): 'deep sky blue',
    (30, 144, 255): 'dodger blue',
    (173, 216, 230): 'light blue',
    (135, 206, 235): 'sky blue',
    (135, 206, 250): 'light sky blue',
    (25, 25, 112): 'midnight blue',
    (0, 0, 128): 'navy',
    (0, 0, 139): 'dark blue',
    (0, 0, 205): 'medium blue',
    (0, 0, 255): 'blue',
    (65, 105, 225): 'royal blue',
    (138, 43, 226): 'blue violet',
    (75, 0, 130): 'indigo',
    (72, 61, 139): 'dark slate blue',
    (106, 90, 205): 'slate blue',
    (123, 104, 238): 'medium slate blue',
    (147, 112, 219): 'medium purple',
    (139, 0, 139): 'dark magenta',
    (148, 0, 211): 'dark violet',
    (153, 50, 204): 'dark orchid',
    (186, 85, 211): 'medium orchid',
    (128, 0, 128): 'purple',
    (216, 191, 216): 'thistle',
    (221, 160, 221): 'plum',
    (238, 130, 238): 'violet',
    (255, 0, 255): 'magenta / fuchsia',
    (218, 112, 214): 'orchid',
    (199, 21, 133): 'medium violet red',
    (219, 112, 147): 'pale violet red',
    (255, 20, 147): 'deep pink',
    (255, 105, 180): 'hot pink',
    (255, 182, 193): 'light pink',
    (255, 192, 203): 'pink',
    (250, 235, 215): 'antique white',
    (245, 245, 220): 'beige',
    (255, 228, 196): 'bisque',
    (255, 235, 205): 'blanched almond',
    (245, 222, 179): 'wheat',
    (255, 248, 220): 'corn silk',
    (255, 250, 205): 'lemon chiffon',
    (250, 250, 210): 'light golden rod yellow',
    (255, 255, 224): 'light yellow',
    (139, 69, 19): 'saddle brown',
    (160, 82, 45): 'sienna',
    (210, 105, 30): 'chocolate',
    (205, 133, 63): 'peru',
    (244, 164, 96): 'sandy brown',
    (222, 184, 135): 'burly wood',
    (210, 180, 140): 'tan',
    (188, 143, 143): 'rosy brown',
    (255, 228, 181): 'moccasin',
    (255, 222, 173): 'navajo white',
    (255, 218, 185): 'peach puff',
    (255, 228, 225): 'misty rose',
    (255, 240, 245): 'lavender blush',
    (250, 240, 230): 'linen',
    (253, 245, 230): 'old lace',
    (255, 239, 213): 'papaya whip',
    (255, 245, 238): 'sea shell',
    (245, 255, 250): 'mint cream',
    (112, 128, 144): 'slate gray',
    (119, 136, 153): 'light slate gray',
    (176, 196, 222): 'light steel blue',
    (230, 230, 250): 'lavender',
    (255, 250, 240): 'floral white',
    (240, 248, 255): 'alice blue',
    (248, 248, 255): 'ghost white',
    (240, 255, 240): 'honeydew',
    (255, 255, 240): 'ivory',
    (240, 255, 255): 'azure',
    (255, 250, 250): 'snow',
    (0, 0, 0): 'black',
    (105, 105, 105): 'dim gray / dim grey',
    (128, 128, 128): 'gray / grey',
    (169, 169, 169): 'dark gray / dark grey',
    (192, 192, 192): 'silver',
    (211, 211, 211): 'light gray / light grey',
    (220, 220, 220): 'gainsboro',
    (245, 245, 245): 'white smoke',
    (255, 255, 255): 'white',
    # Added more shades of dark brown
    (101, 67, 33): 'dark brown',
    (102, 51, 0): 'wood brown',
    (92, 64, 51): 'taupe brown',
    (139, 69, 19): 'saddle brown',
    (128, 0, 0): 'maroon',
    (128, 0, 0): 'burgundy',
    (80, 55, 35): 'coffee brown',
    (120, 80, 50): 'deep brown',
    (100, 65, 45): 'chestnut brown',
}

# Prepare color data for KDTree
colors = np.array(list(color_lookup.keys()))
color_names = list(color_lookup.values())

# Convert RGB to LAB color space for all colors in the lookup
colors_lab = np.array([cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0][0] for color in colors])

# Build KDTree for fast nearest neighbor search in LAB color space
tree = KDTree(colors_lab)

# Function to find the closest color name in LAB color space
def closest_color_name(rgb):
    """
    Finds the closest human-readable color name for a given RGB color.
    
    Parameters:
        rgb (array-like): RGB color as a list or array [R, G, B].
    
    Returns:
        str: Closest color name.
    """
    rgb = np.round(rgb).astype(np.uint8)
    lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0]
    _, index = tree.query(lab)
    return color_names[index]

# ======================= #
# === End of Added === #
# ======================= #

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Color Histogram Analysis with Enhanced Visualization and Grouping')
    parser.add_argument('--binary_masks_dir', required=True, help='Directory containing binary mask images.')
    parser.add_argument('--color_masks_dir', required=True, help='Directory containing color-transformed mask images.')
    parser.add_argument('--mask_types', nargs='+', required=True, choices=['fhs', 'normalized', 'fhs_normalized'], help='List of mask types to analyze.')
    parser.add_argument('--output_dir', required=True, help='Directory to save all outputs.')
    parser.add_argument('--n_components', type=int, default=2, help='Number of PCA components (default: 2).')
    parser.add_argument('--visualize', action='store_true', help='Flag to enable visualization of PCA results.')
    parser.add_argument('--grouping_file', required=False, help='Path to the grouping file (CSV, TSV, JSON) mapping specimens to groups.')
    parser.add_argument('--append_human_readable', action='store_true', help='Append human-readable color summaries by group.')
    return parser.parse_args()

def parse_grouping_file(grouping_file_path):
    """
    Parse the grouping file and return a dictionary mapping specimen names to group names.
    
    Supports CSV, TSV, and JSON formats.
    Expected formats:
        - CSV/TSV: Columns 'specimen_name', 'group_name'
        - JSON: { "specimen_name1": "group_name1", "specimen_name2": "group_name2", ... }
    """
    grouping = {}
    if not os.path.isfile(grouping_file_path):
        print(f"Error: Grouping file '{grouping_file_path}' does not exist.")
        sys.exit(1)
    
    _, ext = os.path.splitext(grouping_file_path)
    try:
        if ext.lower() == '.json':
            with open(grouping_file_path, 'r') as f:
                grouping = json.load(f)
        elif ext.lower() in ['.csv', '.tsv']:
            delimiter = ',' if ext.lower() == '.csv' else '\t'
            with open(grouping_file_path, 'r', newline='') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    specimen = row.get('specimen_name') or row.get('Specimen_Name')
                    group = row.get('group_name') or row.get('Group_Name')
                    if specimen and group:
                        grouping[specimen] = group
        else:
            print(f"Error: Unsupported file extension '{ext}' for grouping file.")
            sys.exit(1)
    except Exception as e:
        print(f"Error parsing grouping file '{grouping_file_path}': {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"[DEBUG] Grouping loaded: {len(grouping)} specimens mapped to groups.")
    return grouping

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

def extract_interior_pixels(image, mask):
    """
    Extract interior pixel values from an image using a binary mask.
    """
    pixels = image[mask > 0]
    return pixels

def compute_color_histogram(pixels, bins=(8, 8, 8)):
    """
    Compute a color histogram for the given pixels in LAB color space.

    Parameters:
        pixels (numpy.ndarray): The input pixels (Nx3 array).
        bins (tuple): Number of bins for each channel.

    Returns:
        numpy.ndarray: Flattened histogram vector.
    """
    # Convert pixels to LAB color space
    pixels_lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    pixels_lab = pixels_lab.reshape(-1, 3)

    # Compute histogram
    hist, _ = np.histogramdd(pixels_lab, bins=bins, range=[(0, 256), (0, 256), (0, 256)])
    # Normalize histogram
    hist = hist / np.sum(hist)
    hist = hist.flatten()
    return hist

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

def create_thumbnail(image, mask, thumbnail_size=(80, 80)):
    """
    Creates a thumbnail image from an image and a binary mask with a transparent background.
    Resizes the image while maintaining aspect ratio and adds padding to fit the thumbnail size.

    Parameters:
        image (numpy.ndarray): Original image (height, width, channels).
        mask (numpy.ndarray): Binary mask (height, width), where foreground pixels are > 0.
        thumbnail_size (tuple): Size to resize the thumbnail (width, height).

    Returns:
        numpy.ndarray: RGBA thumbnail image with transparent background.
    """
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)

    # Create foreground by masking the image
    foreground = np.zeros_like(image)
    foreground[binary_mask == 1] = image[binary_mask == 1]

    # Convert BGR to RGB
    foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)

    # Create alpha channel based on the mask
    alpha_channel = (binary_mask * 255).astype(np.uint8)

    # Combine RGB and alpha channel
    rgba_image = np.dstack((foreground_rgb, alpha_channel))

    # Convert to PIL Image for resizing
    pil_img = Image.fromarray(rgba_image)

    # Resize while maintaining aspect ratio
    pil_img.thumbnail(thumbnail_size, Image.LANCZOS)

    # Create a new image with transparent background
    new_img = Image.new('RGBA', thumbnail_size, (255, 255, 255, 0))
    # Calculate position to center the thumbnail
    offset = ((thumbnail_size[0] - pil_img.size[0]) // 2,
              (thumbnail_size[1] - pil_img.size[1]) // 2)
    new_img.paste(pil_img, offset)

    # Convert back to NumPy array
    thumbnail_rgba = np.array(new_img)

    return thumbnail_rgba

def plot_pca_with_thumbnails(principal_components, sample_names, color_masks, binary_masks, output_path, visualize=False):
    """
    Plot PCA results with thumbnails of the foreground masks placed at their PCA coordinates.

    Parameters:
        principal_components (numpy.ndarray): PCA-transformed coordinates (N_samples, 2).
        sample_names (list): List of sample names.
        color_masks (list): List of color mask images (as numpy arrays).
        binary_masks (list): List of binary mask images (as numpy arrays).
        output_path (str): Path to save the PCA plot.
        visualize (bool): Whether to display the plot.
    """
    print("Plotting PCA with Thumbnails...")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title('PCA of Color Histograms with Thumbnails')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    # Plot invisible scatter points to set the axis limits
    ax.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.0)

    # Set axis limits based on PCA coordinates
    x_min, x_max = principal_components[:, 0].min(), principal_components[:, 0].max()
    y_min, y_max = principal_components[:, 1].min(), principal_components[:, 1].max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    for i, (pc, color_mask, binary_mask) in enumerate(zip(principal_components, color_masks, binary_masks)):
        print(f"Processing thumbnail {i+1}/{len(color_masks)} at PCA coordinates: {pc}")

        # Create thumbnail from color mask and binary mask
        thumbnail = create_thumbnail(color_mask, binary_mask)

        # Create an OffsetImage with RGBA (transparent background)
        imagebox = OffsetImage(thumbnail, zoom=0.75)
        ab = AnnotationBbox(imagebox, (pc[0], pc[1]),
                            frameon=False, pad=0.0)
        ax.add_artist(ab)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"PCA plot with thumbnails saved to: {output_path}")
    if visualize:
        plt.show()
    plt.close()

def plot_3d_color_histogram(hist, bins=(8, 8, 8), sample_name='Sample', output_path=None, visualize=False):
    """
    Plot a 3D color histogram.

    Parameters:
        hist (numpy.ndarray): Flattened histogram vector.
        bins (tuple): Number of bins for each channel.
        sample_name (str): Name of the sample.
        output_path (str): Path to save the plot.
        visualize (bool): Whether to display the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'3D Color Histogram for {sample_name}')
    ax.set_xlabel('L Channel')
    ax.set_ylabel('A Channel')
    ax.set_zlabel('B Channel')

    # Prepare data for plotting
    hist_reshaped = hist.reshape(bins)
    x_bins = np.arange(bins[0])
    y_bins = np.arange(bins[1])
    z_bins = np.arange(bins[2])

    x, y, z = np.meshgrid(x_bins, y_bins, z_bins, indexing='ij')
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = hist_reshaped.flatten()

    # Only plot bins with non-zero values
    nonzero = values > 0
    x = x[nonzero]
    y = y[nonzero]
    z = z[nonzero]
    values = values[nonzero]

    # Normalize the values for color mapping
    values_norm = values / values.max()

    # Create a scatter plot
    img = ax.scatter(x, y, z, c=values_norm, cmap='viridis', marker='o', s=50, alpha=0.6)
    fig.colorbar(img, ax=ax, label='Normalized Frequency')

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"3D color histogram saved to: {output_path}")
    if visualize:
        plt.show()
    plt.close()

def draw_contour_on_color_mask(color_mask, binary_mask, output_path):
    """
    Draws a green contour around the area defined by the binary mask on the color mask.

    Parameters:
        color_mask (numpy.ndarray): Color mask image.
        binary_mask (numpy.ndarray): Binary mask image.
        output_path (str): Path to save the output image.
    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the color mask
    color_mask_with_contour = color_mask.copy()
    cv2.drawContours(color_mask_with_contour, contours, -1, (0, 255, 0), 2)  # Green color

    # Save the image
    cv2.imwrite(output_path, color_mask_with_contour)
    print(f"Contour drawn on color mask saved to: {output_path}")

def create_thumbnail_key(sample_names, color_masks, binary_masks, output_path, thumbnail_size=(100, 100), images_per_row=5):
    """
    Creates a key image with thumbnails and corresponding filenames.

    Parameters:
        sample_names (list): List of sample names.
        color_masks (list): List of color mask images (as numpy arrays).
        binary_masks (list): List of binary mask images (as numpy arrays).
        output_path (str): Path to save the key image.
        thumbnail_size (tuple): Size of each thumbnail (width, height).
        images_per_row (int): Number of thumbnails per row.
    """
    from PIL import Image, ImageDraw, ImageFont

    # Ensure PIL can find a default font
    try:
        font = ImageFont.load_default()
    except:
        print("Warning: Unable to load default font. Filenames may not be displayed.")
        font = None

    num_samples = len(sample_names)
    num_rows = (num_samples + images_per_row - 1) // images_per_row

    # Calculate extra space needed for rotated labels
    extra_space = int(thumbnail_size[1] * 0.5)  # Adjust as needed

    # Create a blank image to hold all thumbnails and filenames
    grid_width = images_per_row * thumbnail_size[0]
    grid_height = num_rows * (thumbnail_size[1] + extra_space) + 50  # Extra padding
    key_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))

    draw = ImageDraw.Draw(key_image)

    for idx, (name, color_mask, binary_mask) in enumerate(zip(sample_names, color_masks, binary_masks)):
        # Create thumbnail
        thumbnail = create_thumbnail(color_mask, binary_mask, thumbnail_size)

        # Convert thumbnail to PIL image
        thumbnail_pil = Image.fromarray(thumbnail)

        # Calculate position
        row = idx // images_per_row
        col = idx % images_per_row
        x = col * thumbnail_size[0]
        y = row * (thumbnail_size[1] + extra_space)

        # Paste thumbnail onto key image
        key_image.paste(thumbnail_pil, (x, y), thumbnail_pil)

        # Draw filename below thumbnail
        if font:
            # Create an image for the text
            text_img = Image.new('RGBA', (thumbnail_size[0], extra_space), (255, 255, 255, 0))
            text_draw = ImageDraw.Draw(text_img)

            # Get text bounding box
            bbox = font.getbbox(name)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position text at the center of the image
            text_x = (thumbnail_size[0] - text_width) // 2
            text_y = 0  # Start at the top of the extra space

            # Draw text onto the text image
            text_draw.text((text_x, text_y), name, fill='black', font=font)

            # Rotate the text image
            rotated_text_img = text_img.rotate(-30, expand=True)
            rotated_bbox = rotated_text_img.getbbox()
            vertical_offset = rotated_bbox[1]
            # Calculate position to paste the rotated text
            text_img_width, text_img_height = rotated_text_img.size
            text_paste_x = x + (thumbnail_size[0] - text_img_width) // 2
            text_paste_y = y + thumbnail_size[1] - vertical_offset

            # Paste the rotated text onto the key image
            key_image.paste(rotated_text_img, (text_paste_x, text_paste_y), rotated_text_img)
        else:
            # If font is not available, skip drawing text
            pass

    # Save the key image
    key_image.save(output_path)
    print(f"Thumbnail key image saved to: {output_path}")

def compute_group_statistics(all_metrics_df, grouping_dict):
    """
    Compute mean, standard deviation, and range of color histograms per group.

    Parameters:
        all_metrics_df (pandas.DataFrame): DataFrame containing histogram data and sample names.
        grouping_dict (dict): Dictionary mapping sample names to group names.

    Returns:
        pandas.DataFrame: DataFrame containing statistics per group.
    """
    # Map sample names to groups
    all_metrics_df['group'] = all_metrics_df['Sample'].map(grouping_dict)

    # Check for unmapped samples
    unmapped = all_metrics_df[all_metrics_df['group'].isna()]
    if not unmapped.empty:
        print(f"Warning: {len(unmapped)} samples could not be mapped to any group. They will be excluded from group statistics.")
        all_metrics_df = all_metrics_df.dropna(subset=['group'])

    # Compute statistics per group
    group_stats = all_metrics_df.groupby('group').agg(['mean', 'std', 'min', 'max']).reset_index()

    # Flatten MultiIndex columns
    group_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in group_stats.columns.values]

    print("[DEBUG] Group statistics computed.")
    return group_stats

def save_group_statistics(group_stats_df, output_dir, group_type='group'):
    """
    Save group statistics to a CSV file.

    Parameters:
        group_stats_df (pandas.DataFrame): DataFrame containing statistics per group.
        output_dir (str): Directory to save the statistics file.
        group_type (str): Type of grouping (default: 'group').
    """
    stats_output_path = os.path.join(output_dir, f'{group_type}_statistics.csv')
    group_stats_df.to_csv(stats_output_path, index=False)
    print(f"Group statistics saved to: {stats_output_path}")

# ======================= #
# === End of Added === #
# ======================= #

def main():
    args = parse_args()
    binary_masks_dir = args.binary_masks_dir
    color_masks_dir = args.color_masks_dir
    mask_types = args.mask_types
    output_dir = args.output_dir
    n_components = args.n_components
    visualize_flag = args.visualize
    grouping_file = args.grouping_file
    append_human_readable = args.append_human_readable

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
        os.makedirs(os.path.join(mask_output_dirs[mask_type], 'histograms'), exist_ok=True)

    # Handle grouping if grouping_file is provided
    grouping_dict = {}
    if grouping_file:
        grouping_dict = parse_grouping_file(grouping_file)
        if not grouping_dict:
            print("Error: Grouping dictionary is empty. Please check the grouping file.")
            sys.exit(1)

    # Iterate over each mask type
    for mask_type in mask_types:
        print(f"\n=== Processing Mask Type: {mask_type} ===")

        # Define specific output directories
        histograms_dir = os.path.join(mask_output_dirs[mask_type], 'histograms')

        # Step 1: Load binary masks and corresponding color masks
        print("Step 1: Loading binary masks and corresponding color masks...")
        binary_mask_files = [f for f in os.listdir(binary_masks_dir) if f.startswith('binary_mask_')]
        binary_mask_files = [f for f in binary_mask_files if os.path.isfile(os.path.join(binary_masks_dir, f))]

        if not binary_mask_files:
            print(f"Error: No binary mask files found in {binary_masks_dir}.")
            continue

        data_matrix = []
        sample_names = []
        color_masks_list = []
        binary_masks_list = []

        for bm_file in binary_mask_files:
            base_name = bm_file[len('binary_mask_'):]
            color_mask_path = find_corresponding_color_mask(base_name, color_masks_dir, mask_type)
            if not color_mask_path:
                print(f"Skipping sample '{base_name}' due to missing color mask.")
                continue

            # Load binary mask
            binary_mask = cv2.imread(os.path.join(binary_masks_dir, bm_file), cv2.IMREAD_GRAYSCALE)
            if binary_mask is None:
                print(f"Error: Unable to read binary mask at {os.path.join(binary_masks_dir, bm_file)}.")
                continue

            # Load color mask image
            color_mask = cv2.imread(color_mask_path)
            if color_mask is None:
                print(f"Error: Unable to read color mask at {color_mask_path}.")
                continue

            # Save color masks and binary masks for later use
            color_masks_list.append(color_mask)
            binary_masks_list.append(binary_mask)

            # Visualize the region used to calculate the histogram
            contour_output_path = os.path.join(mask_output_dirs[mask_type], f'contour_on_color_mask_{base_name}.png')
            draw_contour_on_color_mask(color_mask, binary_mask, contour_output_path)

            # Step 2: Extract interior pixels
            pixels = extract_interior_pixels(color_mask, binary_mask)
            if pixels.size == 0:
                print(f"Warning: No interior pixels found for sample '{base_name}'.")
                continue

            # Step 3: Compute color histogram
            hist = compute_color_histogram(pixels, bins=(8, 8, 8))  # Adjust bins as needed
            data_matrix.append(hist)
            sample_names.append(base_name)

            # Save histogram
            histogram_path = os.path.join(histograms_dir, f'histogram_{base_name}.npy')
            np.save(histogram_path, hist)
            print(f"Histogram saved to: {histogram_path}")

            # Plot 3D histogram
            histogram_plot_path = os.path.join(histograms_dir, f'histogram_plot_{base_name}.png')
            plot_3d_color_histogram(hist, bins=(8, 8, 8), sample_name=base_name, output_path=histogram_plot_path, visualize=False)

        if not data_matrix:
            print(f"Error: No data collected for PCA for mask type '{mask_type}'.")
            continue

        # Convert data_matrix to NumPy array
        data_matrix = np.array(data_matrix)
        print(f"Collected data matrix shape for PCA: {data_matrix.shape}")

        # Step 4: Perform PCA
        print("Step 4: Performing PCA on color histograms...")
        principal_components, pca_model = perform_pca_analysis(data_matrix, n_components=n_components)

        # Save PCA results
        pca_results_path = os.path.join(mask_output_dirs[mask_type], 'pca_results.csv')
        header = ['Sample'] + [f'PC{i+1}' for i in range(n_components)]
        pca_data = np.column_stack((sample_names, principal_components))
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
            # Plot PCA with thumbnails
            pca_plot_path = os.path.join(mask_output_dirs[mask_type], 'pca_plot_with_thumbnails.png')
            plot_pca_with_thumbnails(principal_components, sample_names, color_masks_list, binary_masks_list, pca_plot_path, visualize=True)

            # Create thumbnail key image
            key_image_path = os.path.join(mask_output_dirs[mask_type], 'thumbnail_key.png')
            create_thumbnail_key(sample_names, color_masks_list, binary_masks_list, key_image_path)

        # Step 6: Grouping and Statistics (if grouping_file is provided)
        if grouping_file:
            print("Step 6: Computing group-based statistics...")
            # Create a DataFrame for all metrics
            all_metrics_df = pd.DataFrame(data_matrix, columns=[f'bin_{i}' for i in range(data_matrix.shape[1])])
            all_metrics_df.insert(0, 'Sample', sample_names)

            # Compute group statistics
            group_stats_df = compute_group_statistics(all_metrics_df, grouping_dict)

            # Save group statistics
            group_stats_output_dir = os.path.join(mask_output_dirs[mask_type], 'group_statistics')
            os.makedirs(group_stats_output_dir, exist_ok=True)
            save_group_statistics(group_stats_df, group_stats_output_dir, group_type='group')

    print("\nColor Histogram Analysis with Grouping completed successfully.")

# ======================= #
# === Added Section === #
# ======================= #

def append_human_readable_colors_by_group(args):
    """
    Reads the JSONL output file and appends a human-readable summary of colors per group
    to the end of the text results file.

    - Uses grouping_dict to map each specimen (base_name) to a group.
    - Aggregates top_k_color_names from JSONL for each group.
    - Appends a group-based human-readable summary to the results text file.
    """
    jsonl_file = os.path.join(args.output_dir_images, 'color_analysis_resultsV3.jsonl')
    results_file = os.path.join(args.output_dir_images, 'color_analysis_resultsV3.txt')

    if not os.path.isfile(jsonl_file):
        print(f"No JSONL file found at {jsonl_file}, cannot append human readable colors by group.")
        return

    grouping_dict = parse_grouping_file(args.grouping_file)
    if not grouping_dict:
        print("No valid grouping dictionary, skipping human-readable grouping summary.")
        return

    # group_top_colors: dict[group -> list of color sets]
    group_top_colors = {}

    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            base_name = entry.get('base_name', 'unknown')
            group = grouping_dict.get(base_name, 'Unassigned')

            # Collect top colors from each section if available
            # We'll combine all available sections (foreground_mask, normalized, etc.)
            # into one combined color list for simplicity.
            species_colors = []
            for section in ['foreground_mask', 'normalized', 'normalized_only']:
                if section in entry and 'top_k_color_names' in entry[section]:
                    species_colors.extend(entry[section]['top_k_color_names'])

            # Add to group list
            if species_colors:
                if group not in group_top_colors:
                    group_top_colors[group] = []
                group_top_colors[group].append((base_name, species_colors))

    if not group_top_colors:
        print("No human-readable colors to append by group.")
        return

    # Append to results file
    with open(results_file, 'a') as txt_file:
        txt_file.write("\n\n=== Human Readable Color Summaries by Group ===\n")
        for group, items in group_top_colors.items():
            txt_file.write(f"\nGroup: {group}\n")
            for (base_name, colors_list) in items:
                txt_file.write(f"  {base_name}: {', '.join(colors_list)}\n")

    print("Appended human-readable color summaries by group to the results file.")

# ======================= #
# === End of Added === #
# ======================= #

if __name__ == "__main__":
    try:
        args = parse_args()
        main()
        # After main completes, optionally append human-readable data if requested
        if args.append_human_readable and args.grouping_file:
            append_human_readable_colors_by_group(args)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
