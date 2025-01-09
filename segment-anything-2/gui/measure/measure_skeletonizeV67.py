import argparse
import json
import cv2
import numpy as np
import os
import csv
import traceback
from pycocotools import mask as maskUtils
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from skimage.morphology import skeletonize
from scipy.cluster.hierarchy import dendrogram, linkage
import easyocr
import umap
import seaborn as sns
import pandas as pd
import sys
import re
from collections import defaultdict

# === New Imports for Grouping and Statistics ===
from scipy.spatial.distance import pdist, squareform
import numpy as np
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Measure length and width of objects in images.')
    parser.add_argument('--json', required=True, help='Path to the COCO JSON file')
    parser.add_argument('--image_dir', required=True, help='Directory containing the images')
    parser.add_argument('--output_dir', required=False, help='Directory to save outputs', default='./outputs')
    parser.add_argument('--image_id', type=int, required=False, help='ID of the image to process (optional)')
    parser.add_argument('--category_name', required=False, help='Name of the category to process', default=None)
    parser.add_argument('--method', choices=['skeleton', 'pca'], default='pca',
                        help='Method to measure length and width: "skeleton" or "pca"')
    parser.add_argument('--skeleton_method', choices=['opencv', 'skimage'], default='opencv',
                        help='Skeletonization method to use: "opencv" or "skimage"')
    parser.add_argument('--trim_branches', action='store_true', help='Enable branch trimming in skeletonization')
    parser.add_argument('--save_results', action='store_true', help='Save measurement results to individual files')
    parser.add_argument('--output_file', type=str, required=False, help='Path to a single output CSV file for measurements')
    parser.add_argument('--jsonl_output', type=str, required=False, help='Path to the output JSONL file for measurements', default=None)
    parser.add_argument('--min_aspect_ratio', type=float, default=5.0, help='Minimum aspect ratio for scale bar detection')
    parser.add_argument('--max_aspect_ratio', type=float, default=200.0, help='Maximum aspect ratio for scale bar detection')
    parser.add_argument('--min_width', type=int, default=10, help='Minimum width in pixels for scale bar detection')
    parser.add_argument('--mm_scale', type=float, required=False, help='Default pixels per mm ratio to use for images without a detected scale bar', default=None)
    
    # === New Argument for Grouping File ===
    parser.add_argument('--grouping_file', type=str, required=False, help='Path to the grouping file (CSV, TSV, or JSON) mapping specimen to groups', default=None)
    
    return parser.parse_args()


def load_annotations(json_path, image_id=None, category_name=None):
    with open(json_path, 'r') as f:
        coco = json.load(f)
    
    # Create mappings
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    images = {img['id']: img for img in coco['images']}
    
    # Determine which image IDs to process
    if image_id is not None:
        image_ids = [image_id]
    else:
        image_ids = list(images.keys())
    
    # Collect annotations for the selected image IDs
    annotations_per_image = {}
    for img_id in image_ids:
        img = images.get(img_id)
        if img is None:
            continue
        
        image_filename = img['file_name']
        
        # **Skip images in subdirectories**
        if '/' in image_filename or '\\' in image_filename:
            print(f"Skipping image '{image_filename}' as it is in a subdirectory.")
            continue
        
        anns = [ann for ann in coco['annotations'] if ann['image_id'] == img_id]
        if category_name:
            category_ids = [cat_id for cat_id, name in categories.items() if name == category_name]
            anns = [ann for ann in anns if ann['category_id'] in category_ids]
        if anns:
            annotations_per_image[img_id] = anns
    
    return annotations_per_image, images, categories


def create_mask_from_annotation(annotation, height, width):
    # COCO annotations can be in segmentation or RLE format
    if 'segmentation' in annotation:
        # Handle both polygon and RLE segmentations
        if isinstance(annotation['segmentation'], list):
            # Polygon format
            rles = maskUtils.frPyObjects(annotation['segmentation'], height, width)
            rle = maskUtils.merge(rles)
        else:
            # RLE format
            rle = annotation['segmentation']
        mask = maskUtils.decode(rle)
    else:
        # Handle other types of annotations if necessary
        mask = np.zeros((height, width), dtype=np.uint8)
    return mask


def ensure_easyocr_models_present(languages=['en'], cache_dir=None):
    """
    Ensures that EasyOCR models for the specified languages are present.
    Downloads them if not.
    
    Parameters:
        languages (list): List of languages for OCR. Default is ['en'].
        cache_dir (str): Directory to cache the models. Defaults to ~/.EasyOCR.
    
    Returns:
        reader (easyocr.Reader): Initialized EasyOCR Reader.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.EasyOCR")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    models_needed = False
    for lang in languages:
        # Define expected model directories
        detector_dir = os.path.join(cache_dir, lang, 'detector')
        recognizer_dir = os.path.join(cache_dir, lang, 'recognizer')
        
        # Check if detector and recognizer directories exist
        if not os.path.isdir(detector_dir) or not os.path.isdir(recognizer_dir):
            models_needed = True
            break
    
    if models_needed:
        print("EasyOCR models not found. Downloading models, please wait...")
        reader = easyocr.Reader(languages, model_storage_directory=cache_dir)
        print("EasyOCR models downloaded successfully.")
    else:
        print("EasyOCR models already present. Skipping download.")
        reader = easyocr.Reader(languages, model_storage_directory=cache_dir)
    
    return reader


def find_scale_bar_length_in_pixels(image, args):
    """
    Detects the scale bar in the image and returns its length in pixels along with its bounding rectangle and contour.
    
    Returns:
        w (int or None): Width of the scale bar in pixels.
        bbox (tuple or None): Bounding box of the scale bar (x, y, w, h).
        scale_bar_contour (numpy.ndarray or None): Contour of the scale bar.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ranges = [
        {"lower": np.array([0, 0, 200]), "upper": np.array([180, 25, 255])},  # White on Gray
        {"lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 50])},    # Black on Gray or White
        {"lower": np.array([0, 0, 200]), "upper": np.array([180, 25, 255])}   # White on Black
    ]
    
    for color_range in ranges:
        mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            scale_bar_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(scale_bar_contour)
            if h < w:
                # Verify that the scale bar is sufficiently long and thin
                aspect_ratio = w / h if h != 0 else 0
                if w >= args.min_width and (args.min_aspect_ratio <= aspect_ratio <= args.max_aspect_ratio):
                    return w, (x, y, w, h), scale_bar_contour
    return None, None, None


def find_scale_bar_length_and_label(image, scale_bar_bbox, reader, args):
    """
    Given the image and bounding box of the scale bar, extracts the label using OCR.
    
    Parameters:
        image (numpy.ndarray): The original image.
        scale_bar_bbox (tuple): Bounding box (x, y, w, h) of the scale bar.
        reader (easyocr.Reader): Initialized EasyOCR Reader.
        args: Command-line arguments containing scale detection thresholds.
    
    Returns:
        scale_length_pixels (int or None): Length of the scale bar in pixels.
        pixels_per_unit (float or None): Number of pixels per unit of measurement.
    """
    if scale_bar_bbox is None:
        print("No scale bar bounding box provided.")
        return None, None

    x, y, w, h = scale_bar_bbox

    # Define ROI for the scale bar label (assuming label is above the scale bar)
    label_roi_y_start = max(y - h - 50, 0)  # 50 pixels above the scale bar
    label_roi_y_end = y
    label_roi_x_start = max(x - 50, 0)      # 50 pixels to the left
    label_roi_x_end = x + w + 50            # 50 pixels to the right
    label_roi = image[label_roi_y_start:label_roi_y_end, label_roi_x_start:label_roi_x_end]

    # Perform OCR on the label ROI
    text = extract_scale_text(label_roi, reader)
    scale_length, scale_unit = parse_scale_text(text)

    if scale_length and scale_unit:
        pixels_per_unit = calculate_pixels_per_mm(w, scale_length, scale_unit)
        return w, pixels_per_unit
    else:
        print("Failed to parse scale bar label.")
        return w, None


def extract_scale_text(roi_image, reader):
    """
    Extracts text from the ROI image using EasyOCR.
    Returns the concatenated text and the highest confidence score.
    """
    # Convert ROI to RGB as EasyOCR expects RGB images
    roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

    # Perform OCR
    results = reader.readtext(roi_rgb)

    # Process OCR results
    text = ""
    highest_confidence = 0.0
    for res in results:
        confidence = res[2]
        detected_text = res[1]
        if confidence > highest_confidence:
            highest_confidence = confidence
            text = detected_text

    # Print extracted text and confidence for debugging
    print(f"DEBUG: Extracted OCR Text: '{text}' with confidence {highest_confidence:.2f}")

    return text


def parse_scale_text(text):
    """
    Parses the extracted text to find scale length and unit.
    """
    # Replace comma with dot for European decimal separator
    text = text.replace(',', '.')

    # Regular expression to find patterns like '1 mm', '100 µm', etc.
    match = re.search(r'(\d+(\.\d+)?)\s*(mm|µm|um)', text.lower())
    if match:
        scale_length = float(match.group(1))
        scale_unit = match.group(3)
        return scale_length, scale_unit
    return None, None


def calculate_pixels_per_mm(scale_bar_length_pixels, scale_length, scale_unit):
    """
    Calculates the number of pixels per millimeter based on the scale bar.
    """
    if scale_unit == 'mm':
        return scale_bar_length_pixels / scale_length
    elif scale_unit in ['µm', 'um']:
        # Convert micrometers to millimeters
        return scale_bar_length_pixels / (scale_length / 1000.0)
    else:
        return None


def clean_filename(filename):
    """
    Cleans the filename by removing or replacing problematic characters.
    
    Parameters:
        filename (str): Original filename.
    
    Returns:
        cleaned (str): Cleaned filename.
    """
    # Ensure the filename is a string
    if not isinstance(filename, (str, bytes)):
        filename = str(filename)
    
    # Replace spaces with underscores and remove other problematic characters
    cleaned = re.sub(r'[^\w\-_.]', '_', filename)
    return cleaned


def measure_metrics_pca(contour, mask, pixels_per_mm=None, img_id=None, ann_idx=None, category_name=None, image_basename=None):
    """
    Measures various metrics of the object using PCA on its contour.
    Length is the maximum distance along the principal component within the contour.
    Height is the maximum internal width perpendicular to the principal component within the contour.
    """
    # Flatten the contour array
    contour = contour.reshape(-1, 2)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(contour)
    
    # Get the principal components
    pc1 = pca.components_[0]  # Principal component (length direction)
    pc2 = pca.components_[1]  # Perpendicular component (height direction)
    
    # Project the contour points onto the principal components
    projections_pc1 = contour @ pc1
    projections_pc2 = contour @ pc2
    
    # Identify points corresponding to maximum and minimum projections along pc1 (length)
    idx_max_pc1 = np.argmax(projections_pc1)
    idx_min_pc1 = np.argmin(projections_pc1)
    point_max_pc1 = contour[idx_max_pc1]
    point_min_pc1 = contour[idx_min_pc1]
    
    # Length is the distance between these two points
    length = np.linalg.norm(point_max_pc1 - point_min_pc1)
    
    # Rotate the contour points to align pc1 with the x-axis
    theta = -np.arctan2(pc1[1], pc1[0])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    rotated_contour = contour @ rotation_matrix.T
    
    # Now, for each unique x in rotated_contour, find the y_min and y_max
    x_values = rotated_contour[:, 0]
    y_values = rotated_contour[:, 1]
    
    # Create a list of unique x-values (we may bin them to avoid issues with discrete pixels)
    x_bins = np.linspace(x_values.min(), x_values.max(), num=100)
    digitized = np.digitize(x_values, x_bins)
    
    max_width = 0
    x_at_max_width = None
    y_top = None
    y_bottom = None
    
    for bin_number in np.unique(digitized):
        y_in_bin = y_values[digitized == bin_number]
        if len(y_in_bin) > 0:
            y_min = y_in_bin.min()
            y_max = y_in_bin.max()
            width = y_max - y_min
            if width > max_width:
                max_width = width
                x_at_max_width = x_bins[bin_number - 1]  # bin_number - 1 because np.digitize bins start from 1
                y_top = y_max
                y_bottom = y_min

    # Height is the maximum width found
    height = max_width
    
    # Coordinates of the height line in the rotated coordinate system
    point_top_rotated = np.array([x_at_max_width, y_top])
    point_bottom_rotated = np.array([x_at_max_width, y_bottom])
    
    # Rotate the points back to the original coordinate system
    rotation_matrix_inv = np.array([[np.cos(-theta), -np.sin(-theta)],
                                    [np.sin(-theta), np.cos(-theta)]])
    point_top = point_top_rotated @ rotation_matrix_inv.T
    point_bottom = point_bottom_rotated @ rotation_matrix_inv.T
    
    # Length and height in mm
    length_mm = length / pixels_per_mm if pixels_per_mm else None
    height_mm = height / pixels_per_mm if pixels_per_mm else None
    
    # Length to height ratio
    length_to_height_ratio = length / height if height != 0 else None
    
    # Calculate area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Aspect ratio
    aspect_ratio = length / height if height != 0 else None
    
    # Bounding rectangle area for extent
    rect_area = length * height if (length and height) else None
    extent = area / rect_area if (rect_area and rect_area != 0) else None
    
    # Convex hull for solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if (hull_area and hull_area != 0) else None
    
    # Equivalent diameter
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    
    # Orientation (angle of the first principal component)
    angle = np.degrees(np.arctan2(pc1[1], pc1[0]))
    
    # Major and minor axis lengths from PCA (standard deviations times 2)
    MA = 2 * np.sqrt(pca.explained_variance_[0]) * np.sqrt(len(contour))
    ma = 2 * np.sqrt(pca.explained_variance_[1]) * np.sqrt(len(contour))
    
    # Convert measurements to real units if pixels_per_mm is provided
    if pixels_per_mm:
        area_mm2 = area / (pixels_per_mm ** 2)
        perimeter_mm = perimeter / pixels_per_mm
        equivalent_diameter_mm = equivalent_diameter / pixels_per_mm
        MA_mm = MA / pixels_per_mm
        ma_mm = ma / pixels_per_mm
    else:
        area_mm2 = perimeter_mm = equivalent_diameter_mm = MA_mm = ma_mm = None
    
    # Prepare metrics dictionary
    metrics = {
        "image_id": img_id,
        "annotation_index": ann_idx + 1,
        "category_name": category_name,
        "method": "PCA",
        "length_pixels": length,
        "height_pixels": height,
        "length_to_height_ratio": length_to_height_ratio,
        "area_pixels": area,
        "perimeter_pixels": perimeter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "equivalent_diameter_pixels": equivalent_diameter,
        "major_axis_length_pixels": MA,
        "minor_axis_length_pixels": ma,
        "orientation_degrees": angle,
        "length_mm": length_mm,
        "height_mm": height_mm,
        "area_mm2": area_mm2,
        "perimeter_mm": perimeter_mm,
        "equivalent_diameter_mm": equivalent_diameter_mm,
        "major_axis_length_mm": MA_mm,
        "minor_axis_length_mm": ma_mm,
        "image_filename": image_basename,
        "length_line_start": point_min_pc1.tolist(),
        "length_line_end": point_max_pc1.tolist(),
        "height_line_start": point_bottom.tolist(),
        "height_line_end": point_top.tolist()
    }
    
    return metrics


def preprocess_mask_for_skeleton(mask):
    """
    Preprocess the binary mask to improve skeletonization results.
    This includes noise removal and gap filling.
    """
    # Remove small objects (noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Close small gaps within the object
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def skeletonize_skimage(mask):
    """
    Perform skeletonization using scikit-image's skeletonize function.
    """
    # Convert mask to boolean
    mask_bool = mask.astype(bool)
    skeleton = skeletonize(mask_bool)
    # Convert boolean skeleton back to uint8
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    return skeleton_uint8


def measure_length_skeleton(skeleton):
    """
    Measures the length of the skeleton by counting the number of skeleton pixels.
    """
    # Compute length by counting skeleton pixels
    length = np.count_nonzero(skeleton)
    return length


def measure_width_skeleton(mask, skeleton):
    """
    Measures the average width of the object based on the skeleton.
    """
    # Compute the distance transform
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Get the radius at skeleton points
    radius = distance * (skeleton > 0)
    # Width is twice the radius
    width_map = radius * 2
    # Avoid division by zero
    non_zero_count = np.count_nonzero(width_map)
    if non_zero_count == 0:
        average_width = 0
    else:
        average_width = np.sum(width_map) / non_zero_count
    return average_width


def remove_branch_points(skeleton):
    """
    Removes branch points from the skeleton to clean up extraneous branches.
    """
    # Define 8-connected neighborhood
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]], dtype=np.uint8)
    
    while True:
        # Count neighbors
        neighbor_count = cv2.filter2D(skeleton, -1, kernel)
        # Branch points have value > 12 (more than two neighbors)
        branch_points = np.where(neighbor_count > 12, 255, 0).astype(np.uint8)
        
        # If no branch points found, exit loop
        if not np.any(branch_points):
            break
        
        # Remove branch points
        skeleton = cv2.subtract(skeleton, branch_points)
    
    return skeleton


def visualize_contours(image, contour, output_dir, img_id, ann_idx, category_name, image_basename):
    """
    Saves a visualization of the contour with labeled points.
    """
    plt.figure(figsize=(10, 10))
    # Plot the original image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Ensure the contour is a 2D array with two columns (x, y)
    if contour.ndim != 2 or contour.shape[1] != 2:
        print(f"Contour for Image {image_basename} {img_id}, Annotation {ann_idx+1} is not valid. Skipping visualization.")
        plt.close()
        return

    # Plot the contour as blue lines
    plt.plot(np.append(contour[:, 0], contour[0, 0]), np.append(contour[:, 1], contour[0, 1]), '-o', color='blue', linewidth=2)

    # Add a bright red dot on point 1 (index 0 in zero-indexing)
    plt.scatter(contour[0, 0], contour[0, 1], color='red', s=100, zorder=5)

    # Calculate the number of points to label based on dividing by 12
    num_points = len(contour)
    label_step = max(1, num_points // 12)  # Avoid step of 0

    # Label points every `label_step` interval
    for j in range(0, num_points, label_step):
        plt.text(contour[j, 0], contour[j, 1], str(j + 1), fontsize=12, color='blue', zorder=10)

    plt.title(f"Contour for Image {image_basename} {img_id}, Annotation {ann_idx+1}, Category: {category_name}")
    plt.axis('off')

    # Save the image with contour
    img_output_dir = os.path.join(output_dir, f"image_{image_basename}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    save_path = os.path.join(img_output_dir, f"ann_{image_basename}_{ann_idx+1}_{category_name}_contour.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Contour visualization saved to {save_path}")


def visualize_length_height(image, contour, length, height, length_mm, height_mm, orientation_degrees, output_dir, img_id, ann_idx, category_name, cleaned_image_basename, length_line_start, length_line_end, height_line_start, height_line_end):
    """
    Saves a visualization showing the length and height measurements on the image.
    """
    # Ensure contour is a 2D array of shape (N, 2)
    if contour.ndim != 2 or contour.shape[1] != 2:
        contour = contour.reshape(-1, 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw the contour
    plt.plot(np.append(contour[:, 0], contour[0, 0]), np.append(contour[:, 1], contour[0, 1]), 'b-', linewidth=2)

    # Plot the length line in red
    plt.plot([length_line_start[0], length_line_end[0]], [length_line_start[1], length_line_end[1]], 'r-', linewidth=2)

    # Plot the height line in green
    plt.plot([height_line_start[0], height_line_end[0]], [height_line_start[1], height_line_end[1]], 'g-', linewidth=2)

    # Optionally, mark the start and end points
    plt.scatter([length_line_start[0], length_line_end[0]], [length_line_start[1], length_line_end[1]], color='red', s=50, zorder=5)
    plt.scatter([height_line_start[0], height_line_end[0]], [height_line_start[1], height_line_end[1]], color='green', s=50, zorder=5)

    # Prepare title with length and height in mm if available
    title = f"Length: {length:.2f} px"
    if length_mm is not None:
        title += f" ({length_mm:.2f} mm)"
    title += f", Height: {height:.2f} px"
    if height_mm is not None:
        title += f" ({height_mm:.2f} mm)"

    plt.title(title)
    plt.axis('off')

    # Save the image with length and height
    img_output_dir = os.path.join(output_dir, f"image_{cleaned_image_basename}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    save_path = os.path.join(img_output_dir, f"ann_{cleaned_image_basename}_{ann_idx+1}_{category_name}_length_height.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Length and Height visualization saved to {save_path}")


def visualize_skeleton_visual(image, skeleton, output_dir, img_id, ann_idx, category_name, image_basename):
    """
    Saves a visualization of the skeleton overlaid on the original image.
    NOTE: Renamed to avoid conflict with another visualize_skeleton function.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(skeleton, cmap='gray', alpha=0.5)
    plt.title(f"Skeleton for Image {image_basename} {img_id}, Annotation {ann_idx+1}, Category: {category_name}")
    plt.axis('off')
    # Create a subdirectory for each image
    img_output_dir = os.path.join(output_dir, f"image_{image_basename}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    save_path = os.path.join(img_output_dir, f"ann_{image_basename}_{ann_idx+1}_{category_name}_skeleton.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Skeleton visualization saved to {save_path}")


def visualize_scale_bar(image, scale_bar_contour, scale_text, output_dir, img_id, image_basename):
    """
    Saves a visualization of the detected scale bar with its extracted text.
    """
    if scale_bar_contour is not None and len(scale_bar_contour) >= 1:
        x, y, w, h = cv2.boundingRect(scale_bar_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
        
        # Put extracted text near the scale bar
        cv2.putText(image, scale_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2)
        
        # Create a subdirectory for each image
        img_output_dir = os.path.join(output_dir, f"image_{image_basename}_{img_id}")
        os.makedirs(img_output_dir, exist_ok=True)
        # Save the visualization
        save_path = os.path.join(img_output_dir, f"scale_bar_visualization.png")
        cv2.imwrite(save_path, image)
        print(f"Scale bar visualization saved to {save_path}")
    else:
        print("Invalid scale_bar_contour provided. Skipping scale bar visualization.")


def save_metrics_to_file(output_dir, filename, metrics, img_id):
    """
    Saves the measurement metrics to a text file.
    """
    img_output_dir = os.path.join(output_dir, f"image_{metrics['image_filename']}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    file_path = os.path.join(img_output_dir, filename)
    with open(file_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics saved to {file_path}")


def save_all_metrics_to_csv(output_file, all_metrics):
    """
    Saves all collected metrics to a single CSV file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Define CSV headers based on the first metric entry
    if not all_metrics:
        print("No metrics to save.")
        return
    
    headers = all_metrics[0].keys()
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for metrics in all_metrics:
            # Convert numpy data types to native Python types
            metrics = {k: v.item() if isinstance(v, np.generic) else v for k, v in metrics.items()}
            writer.writerow(metrics)
    
    print(f"All metrics saved to {output_file}")


def save_metrics_to_jsonl(jsonl_output_path, all_metrics):
    """
    Saves each metric dictionary as a separate JSON object per line in a JSONL file.

    Parameters:
        jsonl_output_path (str): Path to the output JSONL file.
        all_metrics (list): List of metric dictionaries.
    """
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return [convert_types(v) for v in obj]  # Convert tuple to list
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(jsonl_output_path, 'w') as jsonl_file:
        for metric in all_metrics:
            # Convert all values in the metric dict to native Python types
            serializable_metric = convert_types(metric)
            jsonl_file.write(json.dumps(serializable_metric) + '\n')
    print(f"All metrics saved to JSONL file at {jsonl_output_path}")


# === New Functions for Parsing Grouping File and Counting Instances ===

def parse_grouping_file(grouping_file_path):
    """
    Parses the grouping file and returns a dictionary mapping specimen names to groups.

    Parameters:
        grouping_file_path (str): Path to the grouping file (CSV, TSV, or JSON).

    Returns:
        dict: Mapping of specimen_name (str) to group_name (str).
    """
    if not os.path.exists(grouping_file_path):
        print(f"Grouping file not found at {grouping_file_path}. Proceeding without grouping.")
        return {}
    
    _, file_extension = os.path.splitext(grouping_file_path)
    group_mapping = {}
    
    try:
        if file_extension.lower() in ['.csv', '.tsv', '.txt']:
            # Attempt to read with headers
            try:
                if file_extension.lower() == '.csv':
                    df = pd.read_csv(grouping_file_path)
                else:
                    df = pd.read_csv(grouping_file_path, delimiter='\t')
                
                if list(df.columns)[:2] == ['specimen_name', 'group']:
                    group_mapping = pd.Series(df.group.values, index=df.specimen_name).to_dict()
                else:
                    # Assume first two columns are specimen and group
                    group_mapping = pd.Series(df.iloc[:,1].values, index=df.iloc[:,0]).to_dict()
            except pd.errors.ParserError:
                # If headers are missing, read without headers
                if file_extension.lower() == '.csv':
                    df = pd.read_csv(grouping_file_path, header=None)
                else:
                    df = pd.read_csv(grouping_file_path, delimiter='\t', header=None)
                group_mapping = pd.Series(df.iloc[:,1].values, index=df.iloc[:,0]).to_dict()
        
        elif file_extension.lower() == '.json':
            with open(grouping_file_path, 'r') as f:
                data = json.load(f)
                # Expecting a list of dictionaries or a single dictionary
                if isinstance(data, list):
                    for entry in data:
                        specimen = entry.get('specimen_name') or entry.get('specimen') or entry.get('name')
                        group = entry.get('group') or entry.get('species') or entry.get('cluster')
                        if specimen and group:
                            group_mapping[specimen] = group
                elif isinstance(data, dict):
                    # Assuming specimen names are keys and groups are values
                    group_mapping = data
        else:
            print(f"Unsupported file format: {file_extension}. Proceeding without grouping.")
            return {}
        
        print(f"DEBUG: Parsed grouping file with {len(group_mapping)} entries.")
    except Exception as e:
        print(f"Error parsing grouping file: {e}. Proceeding without grouping.")
        return {}
    
    return group_mapping


def count_instances(all_metrics, group_mapping, category_name):
    """
    Counts the number of instances per specimen and per species/group.

    Parameters:
        all_metrics (list of dict): List containing measurement metrics for each annotation.
        group_mapping (dict): Mapping of specimen_name to group_name.
        category_name (str): The category to count instances for.

    Returns:
        dict: Counts per specimen.
        dict: Counts per species/group.
    """
    counts_per_specimen = defaultdict(int)
    counts_per_group = defaultdict(list)
    
    for metric in all_metrics:
        specimen_name = os.path.splitext(metric['image_filename'])[0]
        if metric['category_name'] != category_name:
            continue  # Only count the specified category
        
        counts_per_specimen[specimen_name] += 1
        
        group_name = group_mapping.get(specimen_name, 'Unassigned')
        counts_per_group[group_name].append(1)  # Each annotation counts as 1
    
    # For species/groups without any instances, ensure they are represented with 0
    if group_mapping:
        unique_groups = set(group_mapping.values())
        for group in unique_groups:
            if group not in counts_per_group:
                counts_per_group[group] = [0]
    
    # DEBUG: Print counts per specimen and per group
    print("DEBUG: Counts per specimen:")
    for specimen, count in counts_per_specimen.items():
        print(f"  {specimen}: {count}")
    
    print("\nDEBUG: Counts per group:")
    for group, counts in counts_per_group.items():
        print(f"  {group}: {counts}")
    
    return counts_per_specimen, counts_per_group


def compute_statistics(counts_per_group):
    """
    Computes mean, standard deviation, min, and max of counts per group.

    Parameters:
        counts_per_group (dict): Mapping of group_name (str) to list of counts (list of int).

    Returns:
        dict: Nested dictionary with group names as keys and their statistics as values.
              Example:
              {
                  'group1': {'mean': ..., 'std': ..., 'min': ..., 'max': ...},
                  'group2': {'mean': ..., 'std': ..., 'min': ..., 'max': ...},
                  ...
              }
    """
    stats_per_group = {}
    
    for group, counts in counts_per_group.items():
        counts_array = np.array(counts)
        mean = float(np.mean(counts_array))
        std = float(np.std(counts_array))
        range_ = (int(np.min(counts_array)), int(np.max(counts_array)))
        specimens_examined = int(len(counts))
        stats_per_group[group] = {
            'mean': mean,
            'std': std,
            'min': range_[0],
            'max': range_[1],
            'specimens_examined': specimens_examined
        }
    
    # DEBUG: Print statistics per group
    print("\nDEBUG: Statistics per group:")
    for group, stats in stats_per_group.items():
        print(f"  {group}: {stats}")
    
    return stats_per_group


def save_group_statistics(stats_per_group, output_dir, category_name):
    """
    Saves the group statistics to separate CSV and JSON files.

    Parameters:
        stats_per_group (dict): Nested dictionary with group names as keys and their statistics as values.
        output_dir (str): Directory to save the statistics files.
        category_name (str): The category name for labeling the files.

    Returns:
        None
    """
    # Prepare CSV
    csv_path = os.path.join(output_dir, f"{category_name}_group_counts_statistics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['group_name', 'mean', 'std', 'min', 'max']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for group, stats in stats_per_group.items():
            writer.writerow({
                'group_name': group,
                'mean': f"{stats['mean']:.2f}" if stats['mean'] is not None else 'NaN',
                'std': f"{stats['std']:.2f}" if stats['std'] is not None else 'NaN',
                'min': f"{stats['min']:.2f}" if stats['min'] is not None else 'NaN',
                'max': f"{stats['max']:.2f}" if stats['max'] is not None else 'NaN'
            })
    
    print(f"DEBUG: Group counts statistics saved to CSV: {csv_path}")

    # Prepare JSON
    json_path = os.path.join(output_dir, f"{category_name}_group_counts_statistics.json")
    with open(json_path, 'w') as jsonfile:
        json.dump(stats_per_group, jsonfile, indent=4)
    
    print(f"DEBUG: Group counts statistics saved to JSON: {json_path}")


def compute_measurement_statistics(all_metrics, group_mapping, category_name):
    """
    Computes mean, standard deviation, min, and max for each measurement per group.

    Parameters:
        all_metrics (list of dict): List of measurement metrics.
        group_mapping (dict): Mapping of specimen_name to group_name.
        category_name (str): The category name to filter metrics.

    Returns:
        dict: Nested dictionary with group names as keys and measurement statistics as values.
    """
    # Initialize a dictionary to hold measurements per group
    measurements_per_group = defaultdict(lambda: defaultdict(list))
    
    # Define the measurement fields to compute statistics on
    measurement_fields = ['length_mm', 'height_mm', 'area_mm2', 'perimeter_mm',
                         'equivalent_diameter_mm', 'major_axis_length_mm', 'minor_axis_length_mm']
    
    for metric in all_metrics:
        specimen_name = os.path.splitext(metric['image_filename'])[0]
        if metric['category_name'] != category_name:
            continue  # Only consider the specified category
        
        group_name = group_mapping.get(specimen_name, 'Unassigned')
        
        for field in measurement_fields:
            value = metric.get(field)
            if value is not None:
                measurements_per_group[group_name][field].append(value)
    
    # Compute statistics per group
    stats_per_group = {}
    for group, measurements in measurements_per_group.items():
        stats_per_group[group] = {}
        for field, values in measurements.items():
            if values:
                mean = np.mean(values)
                std = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                stats_per_group[group][field] = {
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val
                }
            else:
                stats_per_group[group][field] = {
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None
                }
    
    # DEBUG: Print statistics per group
    print("\nDEBUG: Measurement Statistics per group:")
    for group, measurements in stats_per_group.items():
        print(f"Group: {group}")
        for field, stats in measurements.items():
            if stats['mean'] is not None:
                print(f"  {field}: Mean={stats['mean']:.2f}, SD={stats['std']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}")
            else:
                print(f"  {field}: Mean=NaN, SD=NaN, Min=NaN, Max=NaN")
        print()
    
    return stats_per_group


def save_group_measurement_statistics(stats_per_group, output_dir, category_name):
    """
    Saves the measurement statistics per group to separate CSV and JSON files.

    Parameters:
        stats_per_group (dict): Nested dictionary with group names as keys and their statistics as values.
        output_dir (str): Directory to save the statistics files.
        category_name (str): The category name for labeling.

    Returns:
        None
    """
    # Prepare CSV
    csv_path = os.path.join(output_dir, f"{category_name}_group_measurement_statistics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        # Define CSV headers
        fieldnames = ['group_name']
        measurement_fields = ['length_mm', 'height_mm', 'area_mm2', 'perimeter_mm',
                             'equivalent_diameter_mm', 'major_axis_length_mm', 'minor_axis_length_mm']
        for field in measurement_fields:
            fieldnames.extend([f"{field}_mean", f"{field}_std", f"{field}_min", f"{field}_max"])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for group, measurements in stats_per_group.items():
            row = {'group_name': group}
            for field, stats in measurements.items():
                row[f"{field}_mean"] = f"{stats['mean']:.2f}" if stats['mean'] is not None else 'NaN'
                row[f"{field}_std"] = f"{stats['std']:.2f}" if stats['std'] is not None else 'NaN'
                row[f"{field}_min"] = f"{stats['min']:.2f}" if stats['min'] is not None else 'NaN'
                row[f"{field}_max"] = f"{stats['max']:.2f}" if stats['max'] is not None else 'NaN'
            writer.writerow(row)

    print(f"DEBUG: Group measurement statistics saved to CSV: {csv_path}")

    # Prepare JSON
    json_path = os.path.join(output_dir, f"{category_name}_group_measurement_statistics.json")
    with open(json_path, 'w') as jsonfile:
        json.dump(stats_per_group, jsonfile, indent=4)
    
    print(f"DEBUG: Group measurement statistics saved to JSON: {json_path}")


def create_thumbnail(image, mask, thumbnail_size=(80, 80)):
    """
    Creates a thumbnail image from an image and a binary mask with a transparent background.
    Resizes the image while maintaining aspect ratio and adds padding to fit the thumbnail size.

    Parameters:
        image (numpy.ndarray): Original image (height, width, channels).
        mask (numpy.ndarray): Binary mask (height, width), where foreground pixels are 1.
        thumbnail_size (tuple): Size to resize the thumbnail (width, height).

    Returns:
        foreground_rgba (numpy.ndarray): RGBA thumbnail image with transparent background.
    """
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Create foreground by masking the image
    foreground = np.zeros_like(image)
    foreground[binary_mask == 1] = image[binary_mask == 1]
    
    # Convert BGR to RGB if image has 3 channels
    if foreground.shape[2] == 3:
        foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    else:
        foreground_rgb = foreground  # Assume already RGB
    
    # Convert to PIL Image
    pil_img = Image.fromarray(foreground_rgb)
    
    # Create alpha channel based on the mask
    pil_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))
    
    # Combine image and mask to create RGBA image
    pil_img.putalpha(pil_mask)
    
    # Resize while maintaining aspect ratio
    pil_img.thumbnail(thumbnail_size, Image.LANCZOS)  # Resizes in place
    
    # Create a new image with the desired thumbnail size and transparent background
    new_img = Image.new('RGBA', thumbnail_size, (255, 255, 255, 0))
    # Calculate position to center the thumbnail
    offset = ((thumbnail_size[0] - pil_img.size[0]) // 2,
              (thumbnail_size[1] - pil_img.size[1]) // 2)
    new_img.paste(pil_img, offset)
    
    # Convert back to NumPy array
    foreground_rgba = np.array(new_img)
    
    return foreground_rgba


def plot_pca_umap(principal_components, images, masks, output_dir, category_name, method='PCA', thumbnail_size=(80, 80)):
    """
    Plots PCA or UMAP analysis with thumbnails of the foreground masks placed at their reduced coordinates.

    Parameters:
        principal_components (numpy.ndarray): PCA or UMAP-transformed coordinates (N_samples, 2).
        images (list of numpy.ndarray): List of original images corresponding to masks.
        masks (list of numpy.ndarray): List of binary masks.
        output_dir (str): Directory to save the PCA/UMAP plot.
        category_name (str): Name of the category for labeling.
        method (str): 'PCA' or 'UMAP'.
        thumbnail_size (tuple): Size to resize thumbnails for plotting.

    Returns:
        None
    """
    print(f"DEBUG: Plotting {method} with Thumbnails for category '{category_name}'...")
    print("DEBUG: Principal Components Shape:", principal_components.shape)
    print("DEBUG: First 5 Principal Components:\n", principal_components[:5])

    # Check if PCA data is valid
    if principal_components.size == 0:
        print(f"Error: {method} data is empty.")
        return

    # Verify that the number of data points matches
    if len(principal_components) != len(images) or len(images) != len(masks):
        print("Error: The lengths of principal_components, images, and masks do not match.")
        print(f"principal_components: {principal_components.shape}, images: {len(images)}, masks: {len(masks)}")
        return

    fig, ax = plt.subplots(figsize=(16, 12))  # Increased figure size to accommodate thumbnails
    ax.set_title(f'{method} of {category_name} Measurement Metrics with Thumbnails')
    ax.set_xlabel(f'{method} Component 1')
    ax.set_ylabel(f'{method} Component 2')

    # Plot invisible scatter points to set the axis limits
    ax.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.0)

    # Set axis limits based on PCA coordinates
    x_min, x_max = principal_components[:, 0].min(), principal_components[:, 0].max()
    y_min, y_max = principal_components[:, 1].min(), principal_components[:, 1].max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    for i, (pc, mask, img) in enumerate(zip(principal_components, masks, images)):
        print(f"DEBUG: Processing thumbnail {i+1}/{len(images)} at {method} coordinates: {pc}")

        # Create thumbnail from image and mask
        thumbnail = create_thumbnail(img, mask, thumbnail_size)
        if thumbnail is None:
            print(f"Skipping thumbnail {i+1} due to creation failure.")
            continue

        # Create an OffsetImage with RGBA (transparent background)
        imagebox = OffsetImage(thumbnail, zoom=0.75)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (pc[0], pc[1]),
                            frameon=False, pad=0.0)
        ax.add_artist(ab)

    plt.grid(True)
    cleaned_category_name = clean_filename(category_name)
    plot_filename = f'{method.lower()}_{cleaned_category_name}_with_thumbnails.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"{method} plot with thumbnails saved to: {plot_path}")


def plot_hierarchical_dendrogram(X, output_dir, category_name):
    """
    Plots a hierarchical clustering dendrogram.

    Parameters:
        X (numpy.ndarray): Scaled feature data.
        output_dir (str): Directory to save the dendrogram.
        category_name (str): Name of the category for labeling.

    Returns:
        None
    """
    linked = linkage(X, 'ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=False)
    plt.title(f'Hierarchical Clustering Dendrogram for {category_name}')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    cleaned_category_name = clean_filename(category_name)
    plot_filename = f'hierarchical_dendrogram_{cleaned_category_name}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Hierarchical dendrogram saved to: {plot_path}")


def perform_dbscan(X, output_dir, category_name, principal_components, images, masks):
    """
    Performs DBSCAN clustering and plots the results with thumbnails.

    Parameters:
        X (numpy.ndarray): Scaled feature data.
        output_dir (str): Directory to save the DBSCAN plot.
        category_name (str): Name of the category for labeling.
        principal_components (numpy.ndarray): PCA-transformed coordinates for plotting.
        images (list of numpy.ndarray): List of original images corresponding to masks.
        masks (list of numpy.ndarray): List of binary masks.

    Returns:
        dbscan_labels (numpy.ndarray): Cluster labels from DBSCAN.
    """
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN found {num_clusters} clusters")
    
    # Add cluster labels to principal components for plotting
    plt.figure(figsize=(16, 12))
    unique_labels = set(dbscan_labels)
    colors = sns.color_palette(None, len(unique_labels))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Black used for noise.
            color = (0, 0, 0, 1)
            label_name = 'Noise'
        else:
            label_name = f'Cluster {label+1}'
        class_member_mask = (dbscan_labels == label)
        xy = principal_components[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[color], label=label_name, alpha=0.6, edgecolors='w', linewidths=0.5)
    
    plt.title(f'DBSCAN Clustering of {category_name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    
    # Optionally, add thumbnails
    for i, (pc, img, mask, label) in enumerate(zip(principal_components, images, masks, dbscan_labels)):
        if label == -1:
            continue  # Skip noise points
        thumbnail = create_thumbnail(img, mask)
        if thumbnail is None:
            continue
        imagebox = OffsetImage(thumbnail, zoom=0.5)
        ab = AnnotationBbox(imagebox, (pc[0], pc[1]), frameon=False, pad=0.0)
        plt.gca().add_artist(ab)
    
    cleaned_category_name = clean_filename(category_name)
    plot_filename = f'dbscan_{cleaned_category_name}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"DBSCAN plot saved to: {plot_path}")

    return dbscan_labels


def perform_umap(X, n_components=2, random_state=42):
    """
    Performs UMAP dimensionality reduction.

    Parameters:
        X (numpy.ndarray): Scaled feature data.
        n_components (int): Number of dimensions for UMAP.
        random_state (int): Random state for reproducibility.

    Returns:
        umap_results (numpy.ndarray): UMAP-transformed coordinates.
        umap_model (umap.UMAP): Trained UMAP model.
    """
    umap_model = umap.UMAP(n_components=n_components, random_state=random_state)
    umap_results = umap_model.fit_transform(X)
    print(f"UMAP completed with {n_components} components.")
    return umap_results, umap_model


def flatten_output_directory(output_dir):
    """
    Moves all files from subdirectories into the top-level output_dir.
    Renames files by prefixing with their subdirectory name to prevent collisions.
    Removes empty subdirectories after moving files.
    
    Parameters:
        output_dir (str): Path to the top-level output directory.
        
    Returns:
        None
    """
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for file in files:
            # Construct full file path
            file_path = os.path.join(root, file)
            
            # Skip moving files already in the top-level directory
            if root == output_dir:
                continue
            
            # Extract the subdirectory name
            subdir_name = os.path.basename(root)
            
            # Define the new filename with subdir prefix
            new_filename = f"{subdir_name}_{file}"
            
            # Define the destination path
            destination = os.path.join(output_dir, new_filename)
            
            # Move and rename the file
            shutil.move(file_path, destination)
            print(f"Moved '{file_path}' to '{destination}'")
        
        # After moving files, attempt to remove empty subdirectories
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")
            except OSError:
                print(f"Directory not empty, could not remove: {dir_path}")
    
    print("All files have been moved to the top-level output directory.")



def main():
    global args  # To make args accessible in functions
    args = parse_args()
    print("DEBUG: Arguments parsed successfully.")
    
    # === Ensure Output Directory Exists ===
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"DEBUG: Output directory is set to '{args.output_dir}'")
    # === End of Added Section ===
    
    annotations_dict, images_info, categories = load_annotations(args.json, args.image_id, args.category_name)
    print(f"DEBUG: Loaded annotations for {len(annotations_dict)} images.")
    
    if not annotations_dict:
        if args.image_id:
            print(f"No annotations found for image ID {args.image_id}.")
        else:
            print("No annotations found in the JSON file.")
        return

    # === New Section: Parsing Grouping File ===
    if args.grouping_file:
        group_mapping = parse_grouping_file(args.grouping_file)
        print(f"DEBUG: Group mapping contains {len(group_mapping)} entries.")
    else:
        group_mapping = {}
        print("DEBUG: No grouping file provided. Proceeding without grouping information.")
    # === End of New Section ===

    # Ensure EasyOCR models are present and initialize the reader
    reader = ensure_easyocr_models_present(languages=['en'])
    print("DEBUG: EasyOCR reader initialized.")

    # Collect all metrics and corresponding images and masks if plotting is needed
    all_metrics = []
    images_list = []  # To store images for plotting
    masks_list = []   # To store masks for plotting
    
    for img_id, annotations in annotations_dict.items():
        image_info = images_info.get(img_id)
        if image_info is None:
            print(f"No image found with ID {img_id} in the JSON annotations.")
            continue
        
        # Get the image filename from the metadata
        image_filename = image_info['file_name']
        image_basename = os.path.basename(os.path.normpath(image_filename))
        
        # Clean the basename to remove or replace problematic characters
        cleaned_image_basename = clean_filename(image_basename)

        image_path = os.path.join(args.image_dir, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found: {image_path}")
            continue
        height, width = image.shape[:2]
        
        if not annotations:
            print(f"Image ID ('{image_filename}') {img_id}: No annotations found.")
            continue

        # === Handling Skeleton Method ===
        if args.method == 'skeleton':
            for ann_idx, ann in enumerate(annotations):
                category_name = categories.get(ann['category_id'], 'Unknown')
                mask = create_mask_from_annotation(ann, height, width)
                
                # Convert mask to binary
                binary_mask = (mask > 0).astype(np.uint8) * 255

                # Preprocess the mask
                preprocessed_mask = preprocess_mask_for_skeleton(binary_mask)

                if args.skeleton_method == 'opencv':
                    # Apply thinning using Zhang-Suen and Guo-Hall algorithms
                    try:
                        skeleton_zhang = cv2.ximgproc.thinning(preprocessed_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
                    except AttributeError:
                        print("cv2.ximgproc.thinning (Zhang-Suen) is not available. Ensure that opencv-contrib-python is installed.")
                        continue

                    try:
                        skeleton_guo = cv2.ximgproc.thinning(preprocessed_mask, thinningType=cv2.ximgproc.THINNING_GUOHALL)
                    except AttributeError:
                        print("cv2.ximgproc.thinning (Guo-Hall) is not available. Ensure that opencv-contrib-python is installed.")
                        continue

                    # Choose the skeleton with more connected pixels or based on your preference
                    num_pixels_zhang = np.count_nonzero(skeleton_zhang)
                    num_pixels_guo = np.count_nonzero(skeleton_guo)

                    if num_pixels_zhang > num_pixels_guo:
                        skeleton = skeleton_zhang
                        chosen_algorithm = "Zhang-Suen"
                    else:
                        skeleton = skeleton_guo
                        chosen_algorithm = "Guo-Hall"

                    print(f"DEBUG: Chosen Thinning Algorithm: {chosen_algorithm}")
                
                elif args.skeleton_method == 'skimage':
                    # Apply skeletonization using scikit-image
                    skeleton = skeletonize_skimage(preprocessed_mask)
                    chosen_algorithm = "skimage.skeletonize"
                    num_pixels = np.count_nonzero(skeleton)
                    print(f"DEBUG: Number of skeleton pixels after skeletonization: {num_pixels}")

                # Debugging: Print the number of skeleton pixels after skeletonization
                num_skeleton_pixels = np.count_nonzero(skeleton)
                print(f"DEBUG: Number of skeleton pixels after skeletonization: {num_skeleton_pixels}")
                
                if num_skeleton_pixels == 0:
                    print(f"Image ID {img_id}, Annotation {ann_idx+1}: Skeletonization resulted in no skeleton pixels.")
                    length = 0
                    average_width = 0
                    orientation_degrees = 0
                    height_line_start = (None, None)
                    height_line_end = (None, None)
                else:
                    # **Apply Branch Trimming if Enabled**
                    if args.trim_branches:
                        skeleton = remove_branch_points(skeleton)
                        print("DEBUG: Branch trimming applied to skeleton.")
                        # Recalculate skeleton pixels after trimming
                        num_skeleton_pixels = np.count_nonzero(skeleton)
                        print(f"DEBUG: Number of skeleton pixels after branch trimming: {num_skeleton_pixels}")

                    # Measure length and width using skeletonization
                    length = measure_length_skeleton(skeleton)
                    average_width = measure_width_skeleton(binary_mask, skeleton)

                    # Estimate orientation (dummy value; adjust as needed)
                    orientation_degrees = 0  # Placeholder, adjust based on actual orientation if needed

                    # For skeleton method, height plotting isn't implemented in this context
                    height_line_start = (None, None)
                    height_line_end = (None, None)

                # Prepare metrics dictionary
                metrics = {
                    "image_id": img_id,
                    "annotation_index": ann_idx + 1,
                    "category_name": category_name,
                    "method": f"Skeletonization ({chosen_algorithm})",
                    "length_pixels": length,
                    "average_width_pixels": average_width,
                    "image_filename": image_basename
                }
                
                # === New Section: Counting Instances and Grouping ===
                # **Find Scale Bar Length and Pixels per Unit**
                scale_length_pixels, scale_bar_bbox, scale_bar_contour = find_scale_bar_length_in_pixels(image, args)
                scale_length_pixels, pixels_per_unit = find_scale_bar_length_and_label(image, scale_bar_bbox, reader, args)
                
                # **Assign Default pixels_per_mm if No Scale Bar Detected and --mm_scale is Provided**
                if pixels_per_unit is None and args.mm_scale is not None:
                    pixels_per_unit = args.mm_scale
                    metrics["pixels_per_mm"] = pixels_per_unit
                    print(f"DEBUG: Assigned default pixels_per_mm={pixels_per_unit} for Image '{image_basename}', Annotation {ann_idx+1} as no scale bar was detected.")
                else:
                    metrics["pixels_per_mm"] = pixels_per_unit  # Can be None
                
                # Convert to real units if pixels_per_mm is available
                if pixels_per_unit:
                    metrics["length_mm"] = length / pixels_per_unit
                    metrics["average_width_mm"] = average_width / pixels_per_unit
                else:
                    metrics["length_mm"] = None
                    metrics["average_width_mm"] = None
                # === End of New Section ===

                print(f"DEBUG: Image ID {img_id}, Annotation {ann_idx+1}:")
                print(f"  Category: {category_name}")
                print(f"  Method: Skeletonization ({chosen_algorithm})")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                print()
                
                # Save skeleton visualization
                visualize_skeleton_visual(image, skeleton, args.output_dir, img_id, ann_idx, category_name, image_basename)
                
                # Save the binary mask as an image for visual inspection
                visual_output_dir = os.path.join(args.output_dir, f"image_{cleaned_image_basename}_{img_id}")
                os.makedirs(visual_output_dir, exist_ok=True)
                mask_filename = os.path.join(visual_output_dir, f"ann_{cleaned_image_basename}_{ann_idx+1}_{category_name}_mask.png")
                cv2.imwrite(mask_filename, binary_mask)

                # === New Section: Save scale bar visualization if scale bar was detected ===
                if scale_length_pixels and pixels_per_unit and scale_bar_contour is not None:
                    visualize_scale_bar(
                        image, 
                        scale_bar_contour, 
                        f"{scale_length_pixels} px / {pixels_per_unit:.2f} px/mm", 
                        args.output_dir, 
                        img_id, 
                        image_basename
                    )
                else:
                    print(f"DEBUG: Image ID {img_id}, Annotation {ann_idx+1}: No valid scale bar detected for visualization.")
                # === End of New Section ===

                # Save metrics to individual file if required
                if args.save_results:
                    filename = f"ann_{cleaned_image_basename}_{ann_idx+1}_{category_name}_skeleton_metrics.txt"
                    save_metrics_to_file(args.output_dir, filename, metrics, img_id)
                
                # Collect metrics and corresponding images and masks for clustering and visualization
                if args.output_file or args.jsonl_output:
                    all_metrics.append(metrics)
                    images_list.append(image)
                    masks_list.append(binary_mask)
        
        # === Handling PCA Method ===
        if args.method == 'pca':
            for ann_idx, ann in enumerate(annotations):
                category_name = categories.get(ann['category_id'], 'Unknown')
                mask = create_mask_from_annotation(ann, height, width)
                
                # Convert mask to binary
                binary_mask = (mask > 0).astype(np.uint8) * 255

                # Fill holes in the mask to ensure the contour is fully connected
                binary_mask_filled = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

                # Extract the object's contour from the mask
                contours, _ = cv2.findContours(binary_mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    print(f"Image ID {img_id}, Annotation {ann_idx+1}: No contours found.")
                    continue

                object_contour = max(contours, key=cv2.contourArea)
                
                # === New Section: Counting Instances and Grouping ===
                # **Find Scale Bar Length and Pixels per Unit**
                scale_length_pixels, scale_bar_bbox, scale_bar_contour = find_scale_bar_length_in_pixels(image, args)
                scale_length_pixels, pixels_per_unit = find_scale_bar_length_and_label(image, scale_bar_bbox, reader, args)
                
                # **Assign Default pixels_per_mm if No Scale Bar Detected and --mm_scale is Provided**
                if pixels_per_unit is None and args.mm_scale is not None:
                    pixels_per_unit = args.mm_scale
                    print(f"DEBUG: Assigned default pixels_per_mm={pixels_per_unit} for Image '{image_basename}', Annotation {ann_idx+1} as no scale bar was detected.")
                
                # **Handle pixels_per_mm and Pass to measure_metrics_pca**
                if pixels_per_unit:
                    pixels_per_mm = pixels_per_unit
                else:
                    pixels_per_mm = None

                # Perform PCA measurements with the updated function
                metrics = measure_metrics_pca(
                    contour=object_contour,
                    mask=binary_mask_filled,  # Pass the filled binary mask
                    pixels_per_mm=pixels_per_mm,  # Pass the determined pixels_per_mm
                    img_id=img_id,
                    ann_idx=ann_idx,
                    category_name=category_name,
                    image_basename=image_basename
                )
                
                # === End of New Section ===

                print(f"DEBUG: Image ID {img_id}, Annotation {ann_idx+1}:")
                print(f"  Category: {category_name}")
                print(f"  Method: PCA")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                print()
                
                # Save contour visualization
                visualize_contours(image, object_contour, args.output_dir, img_id, ann_idx, category_name, image_basename)
                
                # === New Section: Visualize Length and Height with Lines ===
                # **Pass the orientation_degrees and height line coordinates to visualize_length_height**
                visualize_length_height(
                    image, 
                    object_contour, 
                    metrics["length_pixels"], 
                    metrics["height_pixels"], 
                    metrics["length_mm"], 
                    metrics["height_mm"],
                    metrics["orientation_degrees"],
                    args.output_dir, 
                    img_id, 
                    ann_idx, 
                    category_name, 
                    cleaned_image_basename,  # Pass cleaned_image_basename instead of image_basename
                    metrics.get("length_line_start"),
                    metrics.get("length_line_end"),
                    metrics.get("height_line_start"),
                    metrics.get("height_line_end")
                )
                # === End of New Section ===

                # === New Section: Save scale bar visualization if scale bar was detected ===
                if scale_length_pixels and pixels_per_unit and scale_bar_contour is not None:
                    visualize_scale_bar(
                        image, 
                        scale_bar_contour, 
                        f"{scale_length_pixels} px / {pixels_per_unit:.2f} px/mm", 
                        args.output_dir, 
                        img_id, 
                        image_basename
                    )
                else:
                    print(f"DEBUG: Image ID {img_id}, Annotation {ann_idx+1}: No valid scale bar detected for visualization.")
                # === End of New Section ===

                # Save metrics to individual file if required
                if args.save_results:
                    filename = f"ann_{cleaned_image_basename}_{ann_idx+1}_{category_name}_pca_metrics.txt"
                    save_metrics_to_file(args.output_dir, filename, metrics, img_id)
                
                # Collect metrics and corresponding images and masks for clustering and visualization
                if args.output_file or args.jsonl_output:
                    all_metrics.append(metrics)
                    images_list.append(image)
                    masks_list.append(binary_mask_filled)

    # === After Processing All Images and Annotations ===
    # Save all metrics to CSV if output_file is specified
    if args.output_file:
        save_all_metrics_to_csv(args.output_file, all_metrics)
        print(f"DEBUG: Metrics saved to CSV file: {args.output_file}")
    
    # Save all metrics to JSONL if jsonl_output is specified
    if args.jsonl_output:
        save_metrics_to_jsonl(args.jsonl_output, all_metrics)
        print(f"DEBUG: Metrics saved to JSONL file: {args.jsonl_output}")

    # === New Section: Counting Instances and Calculating Statistics ===
    if args.output_file or args.jsonl_output:
        counts_per_specimen, counts_per_group = count_instances(all_metrics, group_mapping, args.category_name)
        stats_per_group_counts = compute_statistics(counts_per_group)
        save_group_statistics(stats_per_group_counts, args.output_dir, args.category_name)

        # === New Section: Compute and Save Measurement Statistics per Group ===
        measurement_stats_per_group = compute_measurement_statistics(all_metrics, group_mapping, args.category_name)
        save_group_measurement_statistics(measurement_stats_per_group, args.output_dir, args.category_name)
        # === End of New Section ===

        # === New Section: Clustering and Visualization per Category ===
        # After collecting all metrics, perform clustering and visualize
        if args.method == 'pca' and (args.output_file or args.jsonl_output):
            # Load data from CSV or JSONL
            if args.output_file:
                df = pd.read_csv(args.output_file)
                print(f"DEBUG: Loaded data from CSV: {args.output_file} with {len(df)} entries.")
            elif args.jsonl_output:
                try:
                    df = pd.read_json(args.jsonl_output, lines=True)
                    print(f"DEBUG: Loaded data from JSONL: {args.jsonl_output} with {len(df)} entries.")
                except ValueError as e:
                    print(f"Error loading JSONL file: {e}")
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame(all_metrics)
            
            # Select numerical features for clustering
            features = ['length_mm', 'height_mm', 'area_mm2', 'perimeter_mm',
                        'equivalent_diameter_mm', 'major_axis_length_mm', 'minor_axis_length_mm']
            
            # Drop rows with missing values
            df_clean = df.dropna(subset=features + ['category_name'])
            print(f"DEBUG: Data after dropping missing values: {len(df_clean)} entries.")
            
            # Check if df_clean is empty
            if df_clean.empty:
                print("No valid data available for clustering after removing rows with missing values.")
                return
            
            # Group by category
            grouped = df_clean.groupby('category_name')
            
            for category, group in grouped:
                print(f"DEBUG: Processing category: {category}")
                
                # Extract features
                X = group[features].values
                
                # Scale the features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                print("DEBUG: Feature scaling completed.")
                
                # Perform PCA
                pca_model = PCA(n_components=2)
                principal_components = pca_model.fit_transform(X_scaled)
                group = group.copy()
                group['PC1'] = principal_components[:, 0]
                group['PC2'] = principal_components[:, 1]
                print("DEBUG: PCA transformation completed.")
                
                # Perform UMAP
                umap_results, umap_model = perform_umap(X_scaled, n_components=2)
                group['UMAP1'] = umap_results[:, 0]
                group['UMAP2'] = umap_results[:, 1]
                print("DEBUG: UMAP transformation completed.")
                
                # Collect corresponding images and masks
                # Use the index to match all_metrics
                indices = group.index.tolist()
                clustered_images = [images_list[i] for i in indices]
                clustered_masks = [masks_list[i] for i in indices]
                print(f"DEBUG: Collected {len(clustered_images)} images and masks for category '{category}'.")

                # Plot PCA with thumbnails
                plot_pca_umap(principal_components, clustered_images, clustered_masks, args.output_dir, category, method='PCA')
                
                # Plot UMAP with thumbnails
                plot_pca_umap(umap_results, clustered_images, clustered_masks, args.output_dir, category, method='UMAP')
                
                # Perform DBSCAN
                dbscan_labels = perform_dbscan(X_scaled, args.output_dir, category, principal_components, clustered_images, clustered_masks)
                group = group.copy()
                group['DBSCAN_cluster'] = dbscan_labels
                print(f"DEBUG: DBSCAN labels assigned for category '{category}'.")

                # Perform Hierarchical Clustering
                hierarchical = AgglomerativeClustering(n_clusters=3)
                hierarchical_labels = hierarchical.fit_predict(X_scaled)
                group['Hierarchical_cluster'] = hierarchical_labels
                print(f"DEBUG: Hierarchical Clustering labels assigned for category '{category}'.")
                
                # Visualize Hierarchical Clustering Dendrogram
                plot_hierarchical_dendrogram(X_scaled, args.output_dir, category)
                
                # === NEW SECTION: Save Cluster Labels to Separate Files ===
                # Save DBSCAN labels
                dbscan_csv = os.path.join(args.output_dir, f"dbscan_labels_{clean_filename(category)}.csv")
                dbscan_df = group[['DBSCAN_cluster']]
                dbscan_df.to_csv(dbscan_csv, index=True, header=True)
                print(f"DEBUG: DBSCAN cluster labels saved to CSV: {dbscan_csv}")
                
                # Save Hierarchical Clustering labels
                hierarchical_csv = os.path.join(args.output_dir, f"hierarchical_clustering_labels_{clean_filename(category)}.csv")
                hierarchical_df = group[['Hierarchical_cluster']]
                hierarchical_df.to_csv(hierarchical_csv, index=True, header=True)
                print(f"DEBUG: Hierarchical clustering labels saved to CSV: {hierarchical_csv}")
                # === END OF NEW SECTION ===

                # === NEW SECTION: Do NOT append cluster labels to the main CSV ===
                # If you wish to save cluster labels separately, you can save them here without modifying the main CSV.
                # The previous code saves them as separate CSV files.
                # Hence, no need to append to the main CSV.
                # === END OF NEW SECTION ===

    # === End of New Section ===

    # === Add the Flattening Step Here ===
    flatten_output_directory(args.output_dir)
    # === End of Flattening Step ===
    
    print("Processing completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
