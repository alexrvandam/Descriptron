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

Additionally (new in this version):
- Optionally filter which masks to process by a given --category substring.
- Compute average color, dominant color, and top K colors for each sample.
- Produce two JSONL outputs per mask type: human-readable colors & raw RGB data.
- Provide a --top_k argument for controlling how many top colors to extract.
- Provide a --images_json argument for flexible file name matching via a JSON list of image names.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import csv
import json
import re
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

# Color Lookup Dictionary
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
    # Additional dark browns, etc.
    (101, 67, 33): 'dark brown',
    (102, 51, 0): 'wood brown',
    (92, 64, 51): 'taupe brown',
    (139, 69, 19): 'saddle brown',
    (128, 0, 0): 'maroon',
    (128, 0, 0): 'burgundy',  # note: duplicates maroon but left as-is
    (80, 55, 35): 'coffee brown',
    (120, 80, 50): 'deep brown',
    (100, 65, 45): 'chestnut brown',
}

# Prepare color data for KDTree
colors = np.array(list(color_lookup.keys()))
color_names = list(color_lookup.values())

# Convert RGB to LAB color space for all colors in the lookup
colors_lab = np.array([cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2LAB)[0][0] for c in colors])
tree = KDTree(colors_lab)

def closest_color_name(rgb):
    """
    Finds the closest human-readable color name for a given RGB color,
    using LAB distance in a KDTree.
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
    parser.add_argument('--mask_types', nargs='+', required=True, 
                        choices=['fhs', 'normalized', 'fhs_normalized'], 
                        help='List of mask types to analyze.')
    parser.add_argument('--output_dir', required=True, help='Directory to save all outputs.')
    parser.add_argument('--n_components', type=int, default=2, help='Number of PCA components (default: 2).')
    parser.add_argument('--visualize', action='store_true', help='Flag to enable visualization of PCA results.')
    parser.add_argument('--grouping_file', required=False, 
                        help='Path to the grouping file (CSV, TSV, JSON) mapping specimens to groups.')
    parser.add_argument('--append_human_readable', action='store_true', 
                        help='Append human-readable color summaries by group.')
    parser.add_argument('--category', required=False, default=None,
                        help='If specified, only process masks whose filenames contain this substring (e.g. "entire_forewing").')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top colors to output per sample (default=5).')

    # Optional: A JSON file containing official image names to help match base names
    parser.add_argument('--images_json', required=False,
                        help='Optional JSON file listing official image names for flexible base-name matching.')
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


def parse_images_json(images_json_path):
    """
    If provided, parse the JSON for a key "images" -> list of official image names.
    Return a dictionary or list of canonical forms for matching.
    Example JSON:
    {
      "images": [
        "acuminata_1.jpg",
        "acuminata_2.jpg",
        ...
      ]
    }
    """
    if not os.path.isfile(images_json_path):
        print(f"Error: images_json file '{images_json_path}' does not exist.")
        sys.exit(1)
    with open(images_json_path, 'r') as f:
        data = json.load(f)
    # Return a list of official names in canonical form
#    official_list = data.get("images", [])
    official_list = data.get("images", [])

    # If the first item is a dict (like in COCO JSON),
    # try to pull out the "file_name" field. Adjust if your field is different!
    if official_list and isinstance(official_list[0], dict):
        # Convert each dict to its 'file_name' string
        official_list = [
            str(item["file_name"]) for item in official_list 
            if "file_name" in item
        ]

    official_canonical = [canonicalize(img) for img in official_list]


    #official_canonical = [canonicalize(img) for img in official_list]
    return official_list, official_canonical


def canonicalize(name):
    """
    Return a canonical version of a file name string by:
    1. Removing extension
    2. Removing non-alphanumeric (except underscore)
    3. Lowercasing
    e.g. "Acuminata_1.jpg" -> "acuminata_1"
         "acuminata_1jpg" -> "acuminata_1jpg"
    """
    base, _ = os.path.splitext(name)
    # keep letters, digits, underscores
    canon = re.sub(r'[^a-zA-Z0-9_]+', '', base)
    return canon.lower()


def find_corresponding_color_mask(base_name, color_masks_dir, mask_type, images_canonical=None):
    """
    Find the corresponding color-transformed mask for a given base name and mask type.
    
    - Supports multiple possible prefixes for each mask type.
    - If images_canonical is provided, uses flexible 'canonical' matching approach.
    - Otherwise, relies on prefix-based logic.
    """
    # If you changed prefix_map to a list for each mask type, do that above, e.g.:
    # prefix_map = {
    #     'fhs': ['fhs_foreground_mask_'],
    #     'normalized': ['normalized_foreground_mask_'],
    #     'fhs_normalized': [
    #         'normalized_fhs_foreground_mask_',
    #         'fhs_normalized_'  # your additional prefix
    #     ]
    # }
    prefix_map = {
        'fhs': ['fhs_foreground_mask_'],
        'normalized': ['normalized_foreground_mask_'],
        'fhs_normalized': [
            'normalized_fhs_foreground_mask_',
            'fhs_normalized_'
        ]
    }
    
    # Gather color mask files
    color_mask_files = [
        f for f in os.listdir(color_masks_dir)
        if os.path.isfile(os.path.join(color_masks_dir, f))
    ]
    possible_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    color_mask_files = [
        f for f in color_mask_files
        if any(f.lower().endswith(ext) for ext in possible_extensions)
    ]
    
    # Get the list of prefixes for this mask_type
    possible_prefixes = prefix_map.get(mask_type)
    if not possible_prefixes:
        print(f"Error: Unknown mask type '{mask_type}'. No prefixes defined.")
        return None
    
    # images_canonical logic
    if images_canonical:
        # We do flexible matching: canonicalize base_name and compare
        from re import sub as re_sub  # or import at top
        def canonicalize(name):
            import os
            name_base, _ = os.path.splitext(name)
            return re_sub(r'[^a-zA-Z0-9_]+', '', name_base).lower()
        
        base_canon = canonicalize(base_name)
        candidates = []
        for f in color_mask_files:
            # For each prefix in possible_prefixes, see if f starts with that prefix
            for pfx in possible_prefixes:
                if f.startswith(pfx):
                    remainder = f[len(pfx):]
                    remainder_canon = canonicalize(remainder)
                    if remainder_canon == base_canon or (base_canon in remainder_canon) or (remainder_canon in base_canon):
                        candidates.append(f)
        # Decide how many we found
        if len(candidates) == 1:
            print(f"[DEBUG] Using color mask: {candidates[0]} for sample: {base_name}")
            return os.path.join(color_masks_dir, candidates[0])
        elif len(candidates) > 1:
            print(f"Error: Multiple color mask files match '{base_name}' + mask type '{mask_type}': {candidates}")
            return None
        else:
            print(f"Error: No color mask file found for base '{base_name}' + '{mask_type}'.")
            return None
    
    else:
        # Original prefix-based logic
        matches = []
        for pfx in possible_prefixes:
            for f in color_mask_files:
                if f.startswith(pfx) and (base_name == f[len(pfx):]):
                    matches.append(f)
        
        if len(matches) == 1:
            print(f"[DEBUG] Using color mask: {matches[0]} for sample: {base_name}")
            return os.path.join(color_masks_dir, matches[0])
        elif len(matches) > 1:
            print(f"Error: Multiple color mask files found for base '{base_name}' + '{mask_type}': {matches}")
            return None
        else:
            print(f"Error: No color mask file found for base '{base_name}' + '{mask_type}'.")
            return None

def extract_interior_pixels(image, mask):
    """
    Extract interior pixel values from an image using a binary mask.
    """
    return image[mask > 0]


def compute_color_histogram(pixels, bins=(8, 8, 8)):
    """
    Compute a color histogram for the given pixels in LAB color space.
    Returns a flattened histogram vector (bins[0]*bins[1]*bins[2]).
    """
    pixels_lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    pixels_lab = pixels_lab.reshape(-1, 3)
    hist, _ = np.histogramdd(
        pixels_lab, bins=bins, range=[(0, 256), (0, 256), (0, 256)]
    )
    hist = hist / np.sum(hist)  # normalize
    return hist.flatten()


def perform_pca_analysis(data_matrix, n_components=2):
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
    Creates a thumbnail image (RGBA) from an image and a binary mask.
    Foreground is masked out; transparent background elsewhere.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    foreground = np.zeros_like(image)
    foreground[binary_mask == 1] = image[binary_mask == 1]
    foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    alpha_channel = (binary_mask * 255).astype(np.uint8)
    rgba_image = np.dstack((foreground_rgb, alpha_channel))

    pil_img = Image.fromarray(rgba_image)
    pil_img.thumbnail(thumbnail_size, Image.LANCZOS)
    new_img = Image.new('RGBA', thumbnail_size, (255, 255, 255, 0))
    offset = (
        (thumbnail_size[0] - pil_img.size[0]) // 2,
        (thumbnail_size[1] - pil_img.size[1]) // 2
    )
    new_img.paste(pil_img, offset)
    return np.array(new_img)


def plot_pca_with_thumbnails(principal_components, sample_names, color_masks, binary_masks, output_path, visualize=False):
    print("Plotting PCA with Thumbnails...")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title('PCA of Color Histograms with Thumbnails')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    ax.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.0)
    x_min, x_max = principal_components[:, 0].min(), principal_components[:, 0].max()
    y_min, y_max = principal_components[:, 1].min(), principal_components[:, 1].max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    for i, (pc, color_mask, binary_mask) in enumerate(zip(principal_components, color_masks, binary_masks)):
        print(f"Processing thumbnail {i+1}/{len(color_masks)} at PCA coords: {pc}")
        thumbnail = create_thumbnail(color_mask, binary_mask)
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
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'3D Color Histogram for {sample_name}')
    ax.set_xlabel('L Channel')
    ax.set_ylabel('A Channel')
    ax.set_zlabel('B Channel')

    hist_reshaped = hist.reshape(bins)
    x_bins = np.arange(bins[0])
    y_bins = np.arange(bins[1])
    z_bins = np.arange(bins[2])
    x, y, z = np.meshgrid(x_bins, y_bins, z_bins, indexing='ij')
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = hist_reshaped.flatten()

    nonzero = values > 0
    x = x[nonzero]
    y = y[nonzero]
    z = z[nonzero]
    values = values[nonzero]
    values_norm = values / values.max()

    img = ax.scatter(x, y, z, c=values_norm, cmap='viridis', marker='o', s=50, alpha=0.6)
    fig.colorbar(img, ax=ax, label='Normalized Frequency')

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"3D color histogram saved to: {output_path}")
    if visualize:
        plt.show()
    plt.close()


def draw_contour_on_color_mask(color_mask, binary_mask, output_path):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_mask_with_contour = color_mask.copy()
    cv2.drawContours(color_mask_with_contour, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(output_path, color_mask_with_contour)
    print(f"Contour drawn on color mask saved to: {output_path}")


def create_thumbnail_key(sample_names, color_masks, binary_masks, output_path, thumbnail_size=(100, 100), images_per_row=5):
    try:
        font = ImageFont.load_default()
    except:
        print("Warning: Unable to load default font. Filenames may not be displayed.")
        font = None

    num_samples = len(sample_names)
    num_rows = (num_samples + images_per_row - 1) // images_per_row
    extra_space = int(thumbnail_size[1] * 0.5)
    grid_width = images_per_row * thumbnail_size[0]
    grid_height = num_rows * (thumbnail_size[1] + extra_space) + 50
    key_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(key_image)

    for idx, (name, color_mask, binary_mask) in enumerate(zip(sample_names, color_masks, binary_masks)):
        thumbnail = create_thumbnail(color_mask, binary_mask, thumbnail_size)
        thumbnail_pil = Image.fromarray(thumbnail)
        row = idx // images_per_row
        col = idx % images_per_row
        x = col * thumbnail_size[0]
        y = row * (thumbnail_size[1] + extra_space)
        key_image.paste(thumbnail_pil, (x, y), thumbnail_pil)

        if font:
            text_img = Image.new('RGBA', (thumbnail_size[0], extra_space), (255, 255, 255, 0))
            text_draw = ImageDraw.Draw(text_img)
            bbox = font.getbbox(name)
            text_width = bbox[2] - bbox[0]
            text_x = (thumbnail_size[0] - text_width) // 2
            text_y = 0
            text_draw.text((text_x, text_y), name, fill='black', font=font)

            rotated_text_img = text_img.rotate(-30, expand=True)
            rotated_bbox = rotated_text_img.getbbox()
            vertical_offset = rotated_bbox[1]
            text_img_width, text_img_height = rotated_text_img.size
            text_paste_x = x + (thumbnail_size[0] - text_img_width) // 2
            text_paste_y = y + thumbnail_size[1] - vertical_offset
            key_image.paste(rotated_text_img, (text_paste_x, text_paste_y), rotated_text_img)

    key_image.save(output_path)
    print(f"Thumbnail key image saved to: {output_path}")


def compute_group_statistics(all_metrics_df, grouping_dict):
    all_metrics_df['group'] = all_metrics_df['Sample'].map(grouping_dict)
    unmapped = all_metrics_df[all_metrics_df['group'].isna()]
    if not unmapped.empty:
        print(f"Warning: {len(unmapped)} samples could not be mapped to any group. They will be excluded from group statistics.")
        all_metrics_df = all_metrics_df.dropna(subset=['group'])

    group_stats = all_metrics_df.groupby('group').agg(['mean', 'std', 'min', 'max']).reset_index()
    group_stats.columns = [
        '_'.join(col).strip() if col[1] else col[0] 
        for col in group_stats.columns.values
    ]
    print("[DEBUG] Group statistics computed.")
    return group_stats


def save_group_statistics(group_stats_df, output_dir, group_type='group'):
    stats_output_path = os.path.join(output_dir, f'{group_type}_statistics.csv')
    group_stats_df.to_csv(stats_output_path, index=False)
    print(f"Group statistics saved to: {stats_output_path}")


# Helpers for color details
def get_average_bgr(pixels):
    """
    Return the average BGR color of the Nx3 array of BGR pixels.
    """
    avg = np.mean(pixels, axis=0)  # shape (3,)
    return avg


def lab_bin_index_to_center(idx, bins=(8,8,8)):
    """
    Convert a flat histogram bin index to approximate LAB center.
    """
    b2 = bins[1]*bins[2]
    l_bin = idx // b2
    rem = idx % b2
    a_bin = rem // bins[2]
    b_bin = rem % bins[2]

    def center_value(bin_i, nb):
        return (bin_i + 0.5)*(256/nb)
    L_center = center_value(l_bin, bins[0])
    A_center = center_value(a_bin, bins[1])
    B_center = center_value(b_bin, bins[2])
    return [L_center, A_center, B_center]


def lab_to_bgr(lab):
    """
    Convert a single LAB color [L, A, B] to BGR (float -> uint8).
    """
    lab_reshaped = np.uint8([[lab]])
    bgr = cv2.cvtColor(lab_reshaped, cv2.COLOR_Lab2BGR)[0][0]
    return bgr


def get_top_k_colors(hist, k=5, bins=(8,8,8)):
    """
    Find top-k bins by frequency, return a list of dicts with:
    - bin_index
    - freq
    - approx_lab_center
    - approx_bgr_center
    - approx_color_name
    """
    indices = np.argsort(hist)[::-1]
    top_indices = indices[:k]
    results = []
    for i in top_indices:
        freq = float(hist[i])
        lab_center = lab_bin_index_to_center(i, bins=bins)
        bgr_center = lab_to_bgr(lab_center)
        name_approx = closest_color_name(bgr_center[::-1])  # bgr->rgb
        results.append({
            "bin_index": int(i),
            "freq": freq,
            "approx_lab_center": [float(x) for x in lab_center],
            "approx_bgr_center": [int(x) for x in bgr_center],
            "approx_color_name": name_approx
        })
    return results


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
    category_substring = args.category
    top_k_val = args.top_k
    images_json_path = args.images_json

    # Possibly load "official" images from images_json
    official_images = None
    official_canonical = None
    if images_json_path:
        official_images, official_canonical = parse_images_json(images_json_path)

    valid_mask_types = ['fhs', 'normalized', 'fhs_normalized']
    for mask_type in mask_types:
        if mask_type not in valid_mask_types:
            print(f"Error: Invalid mask type '{mask_type}'. Valid types are: {valid_mask_types}")
            sys.exit(1)

    # Create output directories
    mask_output_dirs = {}
    for mask_type in mask_types:
        mask_output_dirs[mask_type] = os.path.join(output_dir, mask_type)
        os.makedirs(mask_output_dirs[mask_type], exist_ok=True)
        os.makedirs(os.path.join(mask_output_dirs[mask_type], 'histograms'), exist_ok=True)

    # Handle grouping
    grouping_dict = {}
    if grouping_file:
        grouping_dict = parse_grouping_file(grouping_file)
        if not grouping_dict:
            print("Error: Grouping dictionary is empty. Please check the grouping file.")
            sys.exit(1)

    # Process each mask type
    for mask_type in mask_types:
        print(f"\n=== Processing Mask Type: {mask_type} ===")
        histograms_dir = os.path.join(mask_output_dirs[mask_type], 'histograms')

        # Gather all binary masks
        print("Step 1: Loading binary masks and corresponding color masks...")
        # If no images_json is provided, use the original 'binary_mask_' prefix
        if not official_canonical:
            binary_mask_files = [
                f for f in os.listdir(binary_masks_dir)
                if f.startswith('binary_mask_') and os.path.isfile(os.path.join(binary_masks_dir, f))
            ]
        else:
            # If we do have images_json, let's match more flexibly
            all_files = [f for f in os.listdir(binary_masks_dir)
                         if os.path.isfile(os.path.join(binary_masks_dir, f))]
            # We'll pick any file that contains 'binary_mask_' or something similar
            # and also partially matches one of the official_canonical names
            binary_mask_files = []
            for f in all_files:
                if 'binary_mask_' in f:
                    # see if canonical form intersects with an official name
                    fc = canonicalize(f.replace('binary_mask_', ''))  # remove prefix
                    # check if fc is in official_canonical or partial match
                    for offc in official_canonical:
                        if (fc == offc) or (fc in offc) or (offc in fc):
                            binary_mask_files.append(f)
                            break

        if not binary_mask_files:
            print(f"Error: No binary mask files found in {binary_masks_dir} for mask_type={mask_type}.")
            continue

        data_matrix = []
        sample_names = []
        color_masks_list = []
        binary_masks_list = []

        hr_json_lines = []   # human-readable
        rgb_json_lines = []  # raw RGB

        for bm_file in binary_mask_files:
            # Optional: skip if it doesn't match the category
            if category_substring and (category_substring not in bm_file):
                continue

            # Original approach to get base_name if no images_json
            if not official_canonical:
                base_name = bm_file[len('binary_mask_'):]
            else:
                # If we have images_json, let's figure out which official name matches
                # so we can produce a stable "base_name"
                candidate_canon = canonicalize(bm_file.replace('binary_mask_', ''))
                matched_official = None
                for offc in official_canonical:
                    if (candidate_canon == offc) or (candidate_canon in offc) or (offc in candidate_canon):
                        # find original official name if needed
                        idx = official_canonical.index(offc)
                        matched_official = official_images[idx]
                        break
                if matched_official:
                    base_name = matched_official
                else:
                    base_name = bm_file[len('binary_mask_'):]

            # Find color mask
            color_mask_path = find_corresponding_color_mask(
                base_name, color_masks_dir, mask_type,
                images_canonical=official_canonical if images_json_path else None
            )
            if not color_mask_path:
                print(f"Skipping sample '{base_name}' due to missing color mask.")
                continue

            # Load the binary mask
            bin_path = os.path.join(binary_masks_dir, bm_file)
            binary_mask = cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE)
            if binary_mask is None:
                print(f"Error: Unable to read binary mask at {bin_path}.")
                continue

            # Load the color mask
            color_mask = cv2.imread(color_mask_path)
            if color_mask is None:
                print(f"Error: Unable to read color mask at {color_mask_path}.")
                continue

            color_masks_list.append(color_mask)
            binary_masks_list.append(binary_mask)

            # Draw contour for visual check
            contour_output_path = os.path.join(
                mask_output_dirs[mask_type],
                f'contour_on_color_mask_{base_name}.png'
            )
            draw_contour_on_color_mask(color_mask, binary_mask, contour_output_path)

            # Step 2: Extract interior pixels
            pixels = extract_interior_pixels(color_mask, binary_mask)
            if pixels.size == 0:
                print(f"Warning: No interior pixels found for sample '{base_name}'.")
                continue

            # Step 3: Compute color histogram
            hist = compute_color_histogram(pixels, bins=(8, 8, 8))
            data_matrix.append(hist)
            sample_names.append(base_name)

            histogram_path = os.path.join(histograms_dir, f'histogram_{base_name}.npy')
            np.save(histogram_path, hist)
            print(f"Histogram saved to: {histogram_path}")

            # 3D plot
            histogram_plot_path = os.path.join(histograms_dir, f'histogram_plot_{base_name}.png')
            plot_3d_color_histogram(hist, bins=(8, 8, 8),
                                    sample_name=base_name,
                                    output_path=histogram_plot_path,
                                    visualize=False)

            # Gather average/dominant/top-k info
            avg_bgr = get_average_bgr(pixels)
            avg_bgr_uint8 = np.round(avg_bgr).astype(np.uint8)
            avg_rgb_uint8 = avg_bgr_uint8[::-1]  # swap BGR->RGB
            avg_color_name = closest_color_name(avg_rgb_uint8)

            # Top K bins
            top_k_list = get_top_k_colors(hist, k=top_k_val, bins=(8,8,8))
            dominant_info = top_k_list[0] if top_k_list else None
            group_name = grouping_dict.get(base_name, "Unassigned") if grouping_dict else None

            record_common = {
                "sample_name": base_name,
                "mask_type": mask_type
            }
            if group_name:
                record_common["group"] = group_name

            # numeric raw data
            record_rgb = {
                **record_common,
                "average_bgr": [float(x) for x in avg_bgr],
                "average_rgb": [int(x) for x in avg_rgb_uint8],
                "top_k_bins": top_k_list
            }

            # human readable
            record_hr = {
                **record_common,
                "average_color_name": avg_color_name,
                "top_k_color_names": [d["approx_color_name"] for d in top_k_list]
            }
            if dominant_info:
                record_hr["dominant_color_name"] = dominant_info["approx_color_name"]

            hr_json_lines.append(json.dumps(record_hr))
            rgb_json_lines.append(json.dumps(record_rgb))

        # End loop over binary_mask_files

        if not data_matrix:
            print(f"Error: No data collected for PCA for mask type '{mask_type}'.")
            continue

        # Save the JSONL files
        hr_json_path = os.path.join(
            mask_output_dirs[mask_type],
            f"human_readable_color_info_{mask_type}.jsonl"
        )
        raw_json_path = os.path.join(
            mask_output_dirs[mask_type],
            f"raw_rgb_color_info_{mask_type}.jsonl"
        )
        with open(hr_json_path, 'w') as fhr:
            for line in hr_json_lines:
                fhr.write(line + "\n")
        with open(raw_json_path, 'w') as frgb:
            for line in rgb_json_lines:
                frgb.write(line + "\n")

        print(f"[NEW] Wrote JSONL files:\n  {hr_json_path}\n  {raw_json_path}")

        # PCA analysis
        data_matrix = np.array(data_matrix)
        print(f"Collected data matrix shape for PCA: {data_matrix.shape}")
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

        # Visualization
        if visualize_flag:
            pca_plot_path = os.path.join(mask_output_dirs[mask_type], 'pca_plot_with_thumbnails.png')
            plot_pca_with_thumbnails(principal_components, sample_names, color_masks_list, binary_masks_list,
                                     pca_plot_path, visualize=True)

            key_image_path = os.path.join(mask_output_dirs[mask_type], 'thumbnail_key.png')
            create_thumbnail_key(sample_names, color_masks_list, binary_masks_list, key_image_path)

        # Group stats
        if grouping_file:
            print("Step 6: Computing group-based statistics...")
            all_metrics_df = pd.DataFrame(data_matrix, columns=[f'bin_{i}' for i in range(data_matrix.shape[1])])
            all_metrics_df.insert(0, 'Sample', sample_names)

            group_stats_df = compute_group_statistics(all_metrics_df, grouping_dict)
            group_stats_output_dir = os.path.join(mask_output_dirs[mask_type], 'group_statistics')
            os.makedirs(group_stats_output_dir, exist_ok=True)
            save_group_statistics(group_stats_df, group_stats_output_dir, group_type='group')

    print("\nColor Histogram Analysis with Grouping completed successfully.")


# ======================= #
# === Added Section === #
# ======================= #

def append_human_readable_colors_by_group(args):
    """
    Reads the JSONL output file (from older approach) and appends a human-readable summary 
    of colors per group to a text results file. 
    (Kept for backward compatibility with the original approach.)
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

    group_top_colors = {}
    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            base_name = entry.get('base_name', 'unknown')
            group = grouping_dict.get(base_name, 'Unassigned')

            # Collect top colors from each section if available
            species_colors = []
            for section in ['foreground_mask', 'normalized', 'normalized_only']:
                if section in entry and 'top_k_color_names' in entry[section]:
                    species_colors.extend(entry[section]['top_k_color_names'])

            if species_colors:
                if group not in group_top_colors:
                    group_top_colors[group] = []
                group_top_colors[group].append((base_name, species_colors))

    if not group_top_colors:
        print("No human-readable colors to append by group.")
        return

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
        if args.append_human_readable and args.grouping_file:
            append_human_readable_colors_by_group(args)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
