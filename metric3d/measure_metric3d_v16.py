import argparse
import json
import cv2
import numpy as np
import os
import csv
import traceback
from pycocotools import mask as maskUtils
from sklearn.decomposition import PCA
from PIL import Image
from PIL.ExifTags import TAGS
import sys
import re
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import easyocr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import umap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.morphology import skeletonize_3d
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import shutil
import glob

# === New ===

# Predefined pixel size (in micrometers) and sensor size (in mm) for common cameras
CAMERA_SPECS = {
    "Canon EOS 5D Mark IV": {"sensor_size": (36.0, 24.0), "pixel_size": 6.36},  # micrometers
    "Nikon D850": {"sensor_size": (35.9, 23.9), "pixel_size": 4.35},
    "Sony Alpha A7R IV": {"sensor_size": (35.7, 23.8), "pixel_size": 3.76},
    "Generic Full Frame": {"sensor_size": (36.0, 24.0), "pixel_size": 5.9},
    # Add more as needed
}

def extract_exif_and_intrinsics(image_path):
    """
    Extracts EXIF data and calculates camera intrinsics from image files.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: A dictionary with focal length, camera make, model, pixel size, 
              sensor size, and camera intrinsics (fx, fy, cx, cy).
    """
    try:
        # Open image and extract EXIF data
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            raise ValueError("No EXIF data found in the image.")

        # Parse EXIF tags
        exif = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
        camera_make = exif.get("Make", "Unknown").strip()
        camera_model = exif.get("Model", "Unknown").strip()
        focal_length = exif.get("FocalLength", None)  # Typically in mm as a tuple (numerator, denominator)
        image_width = exif.get("ExifImageWidth", img.size[0])
        image_height = exif.get("ExifImageHeight", img.size[1])

        # Handle focal length as a tuple
        if isinstance(focal_length, tuple):
            focal_length_mm = float(focal_length[0]) / float(focal_length[1])
        elif isinstance(focal_length, (int, float)):
            focal_length_mm = float(focal_length)
        else:
            focal_length_mm = None

        # Lookup sensor and pixel size
        camera_key = f"{camera_make} {camera_model}".strip()
        specs = CAMERA_SPECS.get(camera_key, None)
        if not specs:
            print(f"Camera '{camera_key}' not found in CAMERA_SPECS. Skipping EXIF-based intrinsics.")
            return {}

        sensor_size = specs["sensor_size"]  # (Width in mm, Height in mm)
        pixel_size = specs["pixel_size"] / 1000.0  # Convert to mm

        # Calculate focal length in pixels
        if focal_length_mm and sensor_size:
            fx = (focal_length_mm * image_width) / sensor_size[0]
            fy = (focal_length_mm * image_height) / sensor_size[1]
        else:
            fx = fy = None

        # Principal point (center of image)
        cx, cy = image_width / 2, image_height / 2

        return {
            "Camera Make": camera_make,
            "Camera Model": camera_model,
            "Focal Length (mm)": focal_length_mm,
            "Sensor Size (mm)": sensor_size,
            "Pixel Size (mm)": pixel_size,
            "Image Dimensions": (image_width, image_height),
            "Camera Intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        }

    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
        return {}

# === End of New ===

# Determine the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='Measure length and width of objects in images using 3D depth estimation.')
    parser.add_argument('--json', required=True, help='Path to the COCO JSON file')
    parser.add_argument('--image_dir', required=True, help='Directory containing the images')
    parser.add_argument('--output_dir', required=False, help='Directory to save outputs', default='./test_images')
    parser.add_argument('--image_id', type=int, required=False, help='ID of the image to process (optional)')
    parser.add_argument('--category_name', required=False, help='Name of the category to process', default=None)
    parser.add_argument('--method', choices=['2D', '3D', '3D_skeleton'], default='3D',
                        help='Measurement method: "2D", "3D", or "3D_skeleton"')
    parser.add_argument('--depth_dir', required=False, help='Directory containing precomputed depth maps (optional)')
    parser.add_argument('--camera_intrinsics', required=False, help='Path to a JSON file containing camera intrinsics (optional)')
    parser.add_argument('--camera_intrinsics_values', required=False,
                        help='Camera intrinsic parameters as a CSV string in the order fx,fy,cx,cy (optional)')
    parser.add_argument('--focal_length', type=float, required=False,
                        help='Focal length in millimeters (optional, requires sensor width or pixel size to compute intrinsics)')
    parser.add_argument('--sensor_width', type=float, required=False,
                        help='Sensor width in millimeters (optional, used with focal length)')
    parser.add_argument('--sensor_height', type=float, required=False,
                        help='Sensor height in millimeters (optional, used with focal length)')
    parser.add_argument('--pixel_size', type=float, required=False,
                        help='Pixel size in micrometers (optional, used with focal length)')
    parser.add_argument('--generate_depth', action='store_true', help='Generate depth maps using Metric3Dv2 if depth maps are not provided')
    parser.add_argument('--mm_scale', type=float, required=False, help='Default pixels per mm ratio to use for images without a detected scale bar', default=None)
    parser.add_argument('--save_results', action='store_true', help='Save measurement results to individual files')
    parser.add_argument('--output_file', type=str, required=False, help='Path to a single output CSV file for measurements')
    parser.add_argument('--jsonl_output', type=str, required=False, help='Path to the output JSONL file for measurements', default=None)
    parser.add_argument('--min_aspect_ratio', type=float, default=5.0, help='Minimum aspect ratio for scale bar detection')
    parser.add_argument('--max_aspect_ratio', type=float, default=200.0, help='Maximum aspect ratio for scale bar detection')
    parser.add_argument('--min_width', type=int, default=10, help='Minimum width in pixels for scale bar detection')
    parser.add_argument('--save_depth', action='store_true', help='Save depth maps to the output directory')
    parser.add_argument('--depth_output_dir', type=str, required=False, help='Directory to save depth maps', default=None)
    parser.add_argument('--save_annotated', action='store_true', help='Save annotated images with measurements')
    parser.add_argument('--annotated_output_dir', type=str, required=False, help='Directory to save annotated images', default=None)
    
    # New Argument for Grouping File
    parser.add_argument('--grouping_file', type=str, required=False, help='Path to the grouping file (CSV, TSV, or JSON) mapping specimen to groups', default=None)
    
    # New Argument for Flattening Output
    parser.add_argument('--flatten_output', action='store_true', help='Flatten the output directory by moving all files to the top-level output directory.')
    
    return parser.parse_args()

def load_annotations(json_path, image_dir, image_id=None, category_name=None):
    with open(json_path, 'r') as f:
        coco = json.load(f)
    
    # Create mappings
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    images = {img['id']: img for img in coco['images']}
    
    # Determine which images to process
    if image_id is not None:
        image_ids = [image_id]
    else:
        # Only include images that are in the specified image_dir (excluding subdirectories)
        image_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        image_filenames_set = set(image_filenames)
        image_ids = [img_id for img_id, img_info in images.items() if img_info['file_name'] in image_filenames_set]
    
    # Collect annotations for the selected image IDs
    annotations_per_image = {}
    for img_id in image_ids:
        img = images.get(img_id)
        if img is None:
            continue
        
        image_filename = img['file_name']
        
        # Skip images in subdirectories
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
    """
    if cache_dir is None:
        cache_dir = os.path.join(script_dir, '.EasyOCR')
    
    os.makedirs(cache_dir, exist_ok=True)
    
    reader = easyocr.Reader(languages, model_storage_directory=cache_dir)
    return reader

def find_scale_bar_length_in_pixels(image, args):
    """
    Detects the scale bar in the image and returns its length in pixels along with its bounding rectangle and contour.
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
    print(f"Extracted OCR Text: '{text}' with confidence {highest_confidence:.2f}")

    return text

def parse_scale_text(text):
    """
    Parses the extracted text to find scale length and unit.
    Handles both comma and dot as decimal separators.
    """
    # Replace comma with dot for European decimal separator
    text = text.replace(',', '.')

    # Regular expression to find patterns like '0.2 mm', '100 µm', etc.
    match = re.search(r'(\d+(\.\d+)?)\s*(mm|µm|um)', text.lower())
    if match:
        scale_length = float(match.group(1))
        scale_unit = match.group(3)
        return scale_length, scale_unit
    return None, None

def calculate_pixels_per_mm(scale_bar_length_px, scale_length, scale_unit):
    """
    Calculates the number of pixels per millimeter based on the scale bar.
    """
    if scale_unit == 'mm':
        return scale_bar_length_px / scale_length
    elif scale_unit in ['µm', 'um']:
        # Convert micrometers to millimeters
        return scale_bar_length_px / (scale_length / 1000.0)
    else:
        return None

def load_depth_map(image_basename, depth_dir):
    """
    Loads a depth map corresponding to the image basename from the depth directory.
    """
    depth_filename = f"{os.path.splitext(image_basename)[0]}_depth.npy"
    depth_path = os.path.join(depth_dir, depth_filename)
    if os.path.exists(depth_path):
        depth_map = np.load(depth_path)
        return depth_map
    else:
        print(f"Depth map not found for image {image_basename}. Expected at {depth_path}")
        return None

def generate_depth_map(image, model, device, camera_intrinsics):
    """
    Generates a depth map for the image using Metric3Dv2 model.
    Adjusts camera intrinsics accordingly.
    """
    # Adjust input size to fit pretrained model
    # Keep aspect ratio
    input_size = (616, 1064)  # For ViT model
    h, w = image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    resized_h = int(h * scale)
    resized_w = int(w * scale)
    rgb = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Padding to input_size
    padding_value = [123.675, 116.28, 103.53]
    pad_h = input_size[0] - resized_h
    pad_w = input_size[1] - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    rgb = cv2.copyMakeBorder(rgb, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=padding_value)
    
    # Adjust intrinsics for scaling and padding
    fx, fy, cx, cy = camera_intrinsics
    fx = fx * scale
    fy = fy * scale
    cx = cx * scale + pad_left
    cy = cy * scale + pad_top
    adjusted_intrinsics = (fx, fy, cx, cy)
    
    # Convert image to torch tensor and normalize
    rgb = torch.from_numpy(rgb.transpose(2, 0, 1)).float().to(device)  # HWC to CHW
    mean = torch.tensor([123.675, 116.28, 103.53], device=device).float().view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device=device).float().view(3, 1, 1)
    rgb = (rgb - mean) / std
    rgb = rgb.unsqueeze(0)  # Add batch dimension
    
    # Inference
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})
    
    # Remove padding from depth map
    pred_depth = pred_depth[:, :, pad_top:pad_top + resized_h, pad_left:pad_left + resized_w]
    # Resize depth map to original image size
    pred_depth = F.interpolate(pred_depth, size=(h, w), mode='bilinear', align_corners=False)
    pred_depth = pred_depth.squeeze().cpu().numpy()
    
    return pred_depth, adjusted_intrinsics

def backproject_2d_to_3d(points_2d, depth_map, camera_intrinsics):
    """
    Back-projects 2D points into 3D space using the depth map and camera intrinsics.
    """
    fx, fy, cx, cy = camera_intrinsics
    x_normalized = (points_2d[:, 0] - cx) / fx
    y_normalized = (points_2d[:, 1] - cy) / fy

    # Ensure indices are within the depth map dimensions
    x_indices = np.clip(points_2d[:, 0].astype(int), 0, depth_map.shape[1] - 1)
    y_indices = np.clip(points_2d[:, 1].astype(int), 0, depth_map.shape[0] - 1)

    depth_values = depth_map[y_indices, x_indices]
    points_3d = np.stack((x_normalized * depth_values, y_normalized * depth_values, depth_values), axis=-1)
    return points_3d

def estimate_camera_intrinsics(image_shape, scale_length_pixels, scale_length_mm):
    """
    Estimates camera intrinsics based on image size and scale bar.
    """
    # Assuming a pinhole camera model and that the scale bar is at the same depth as the objects
    # Estimate focal length in pixels using the known size of the scale bar
    width_in_pixels = image_shape[1]
    height_in_pixels = image_shape[0]
    cx = width_in_pixels / 2
    cy = height_in_pixels / 2

    # Estimate focal length in pixels per mm
    pixels_per_mm = scale_length_pixels / scale_length_mm
    fx = fy = pixels_per_mm * 1.0  # Assuming scale bar is at a unit depth; adjust as needed

    print(f"Estimated camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    return fx, fy, cx, cy

def calibrate_depth_map(depth_map, scale_bar_mask, scale_length_mm, scale_length_pixels):
    """
    Calibrates the depth map using the scale bar.

    Parameters:
        depth_map (numpy.ndarray): The estimated depth map.
        scale_bar_mask (numpy.ndarray): Binary mask where the scale bar is located.
        scale_length_mm (float): The known physical length of the scale bar in millimeters.
        scale_length_pixels (int): Length of the scale bar in pixels.

    Returns:
        depth_map_calibrated (numpy.ndarray): The calibrated depth map.
        scaling_factor (float): The scaling factor applied to the depth map.
    """
    # Extract depth values at the scale bar location
    depth_values = depth_map[scale_bar_mask > 0]

    if depth_values.size == 0:
        print("No depth values found at the scale bar location for calibration.")
        return depth_map, None

    # Compute the average depth value at the scale bar
    D_bar = np.mean(depth_values)  # In meters

    print(f"D_bar (average depth at scale bar): {D_bar} meters")

    # Calculate the scaling factor
    scaling_factor = (scale_length_mm / 1000.0) / D_bar

    print(f"Scaling factor: {scaling_factor}")

    # Adjust the depth map
    depth_map_calibrated = depth_map * scaling_factor

    return depth_map_calibrated, scaling_factor

def measure_metrics_3d(contour_3d, pixels_per_mm=None, img_id=None, ann_idx=None, category_name=None, image_basename=None):
    """
    Measures length and width of the object using PCA on its 3D contour.
    Length is the maximum distance along the principal component within the contour.
    Width is the maximum distance perpendicular to the principal component within the contour.
    """
    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(contour_3d)

    # Get the principal components
    pc1 = pca.components_[0]  # Principal component (length direction)
#    pc2 = pca.components_[1]  # Second principal component (width direction)

    # Project the contour points onto the principal components
    projections_pc1 = contour_3d @ pc1

    # Identify points corresponding to maximum and minimum projections along pc1 (length)
    idx_max_pc1 = np.argmax(projections_pc1)
    idx_min_pc1 = np.argmin(projections_pc1)
    point_max_pc1 = contour_3d[idx_max_pc1]
    point_min_pc1 = contour_3d[idx_min_pc1]

    # Length is the distance between these two points
    length = np.linalg.norm(point_max_pc1 - point_min_pc1)

    # Project the contour points onto the plane perpendicular to pc1
    # The projection onto pc1 is pc1 * (pc1 dot point)
    projections_onto_pc1 = np.outer(projections_pc1, pc1)
    contour_perpendicular = contour_3d - projections_onto_pc1

    # Compute all pairwise distances in the perpendicular plane
    distances = pdist(contour_perpendicular)
    if distances.size == 0:
        max_width = 0
        point1_in_perp_plane = point2_in_perp_plane = contour_3d[0]
    else:
        max_width = np.max(distances)
        distances_square = squareform(distances)
        idx1, idx2 = np.unravel_index(np.argmax(distances_square), distances_square.shape)
        point1_in_perp_plane = contour_3d[idx1]
        point2_in_perp_plane = contour_3d[idx2]

    # Convert length and width to mm
    length_mm = length * 1000
    width_mm = max_width * 1000

    # Convert to pixels if pixels_per_mm is provided
    length_pixels = length_mm * pixels_per_mm if pixels_per_mm else None
    width_pixels = width_mm * pixels_per_mm if pixels_per_mm else None

    # Prepare metrics dictionary
    metrics = {
        "image_id": img_id,
        "annotation_index": ann_idx + 1 if ann_idx is not None else None,
        "category_name": category_name,
        "method": "3D",
        "length_mm": length_mm,  # Convert to mm
        "width_mm": max_width_mm,
        "length_pixels": length_pixels,
        "width_pixels": width_pixels,
        "image_filename": image_basename,
        "length_line_start": point_min_pc1.tolist(),
        "length_line_end": point_max_pc1.tolist(),
        "width_line_start": point1_in_perp_plane.tolist(),
        "width_line_end": point2_in_perp_plane.tolist(),
    }

    return metrics, pca

def project_3d_point_to_2d(point_3d, camera_intrinsics):
    """
    Projects a 3D point into 2D image coordinates using camera intrinsics.
    """
    fx, fy, cx, cy = camera_intrinsics
    x, y, z = point_3d  # x, y, z in meters
    if z == 0:
        z = 1e-6  # Prevent division by zero
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    return np.array([int(u), int(v)])

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

    # Add a bright red dot on the first point
    plt.scatter(contour[0, 0], contour[0, 1], color='red', s=100, zorder=5)

    # Calculate the number of points to label based on dividing by 12
    num_points = len(contour)
    label_step = max(1, num_points // 12)  # Avoid step of 0

    # Label points every `label_step` interval
    for j in range(0, num_points, label_step):
        plt.text(contour[j, 0], contour[j, 1], str(j + 1), fontsize=12, color='blue', zorder=10)

    plt.title(f"Contour for Image {image_basename}, Annotation {ann_idx+1}, Category: {category_name}")
    plt.axis('off')

    # Save the image with contour
    img_output_dir = os.path.join(output_dir, f"image_{clean_filename(image_basename)}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    save_path = os.path.join(img_output_dir, f"ann_{clean_filename(image_basename)}_{ann_idx+1}_{category_name}_contour.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Contour visualization saved to {save_path}")

def visualize_length_width(image, contour, length_mm, width_mm, length_line_start, length_line_end, width_line_start, width_line_end, output_dir, img_id, ann_idx, category_name, image_basename):
    """
    Saves a visualization showing the length and width measurements on the image.
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

    # Plot the width line in green
    plt.plot([width_line_start[0], width_line_end[0]], [width_line_start[1], width_line_end[1]], 'g-', linewidth=2)

    # Optionally, mark the start and end points
    plt.scatter([length_line_start[0], length_line_end[0]], [length_line_start[1], length_line_end[1]], color='red', s=50, zorder=5)
    plt.scatter([width_line_start[0], width_line_end[0]], [width_line_start[1], width_line_end[1]], color='green', s=50, zorder=5)

    # Prepare title with length and width in mm if available
    title = f"Length: {length_mm:.2f} mm"
    title += f", Width: {width_mm:.2f} mm"

    plt.title(title)
    plt.axis('off')

    # Save the image with length and width
    img_output_dir = os.path.join(output_dir, f"image_{clean_filename(image_basename)}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    save_path = os.path.join(img_output_dir, f"ann_{clean_filename(image_basename)}_{ann_idx+1}_{category_name}_length_width.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Length and Width visualization saved to {save_path}")

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
        
        # Save the visualization
        img_output_dir = os.path.join(output_dir, f"image_{clean_filename(image_basename)}_{img_id}")
        os.makedirs(img_output_dir, exist_ok=True)
        save_path = os.path.join(img_output_dir, f"scale_bar_visualization.png")
        cv2.imwrite(save_path, image)
        print(f"Scale bar visualization saved to {save_path}")
    else:
        print("Invalid scale_bar_contour provided. Skipping scale bar visualization.")

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
    print(f"Plotting {method} with Thumbnails for category '{category_name}'...")
    print("Principal Components Shape:", principal_components.shape)
    print("First 5 Principal Components:\n", principal_components[:5])

    # Check if PCA data is valid
    if principal_components is None or principal_components.size == 0:
        print(f"Error: {method} data is empty or None.")
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
        print(f"Processing thumbnail {i+1}/{len(images)} at {method} coordinates: {pc}")

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
    if X.shape[0] < 2:
        print(f"Not enough data points ({X.shape[0]}) for hierarchical clustering. Skipping dendrogram plotting.")
        return

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
    if X.shape[0] < 2:
        print(f"Not enough data points ({X.shape[0]}) for DBSCAN. Skipping DBSCAN clustering.")
        return np.array([])

    dbscan = DBSCAN(eps=0.5, min_samples=2)
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
    if X.shape[0] < 2:
        print(f"Not enough data points ({X.shape[0]}) for UMAP. Skipping UMAP.")
        return None, None
    umap_model = umap.UMAP(n_components=n_components, random_state=random_state)
    umap_results = umap_model.fit_transform(X)
    print(f"UMAP completed with {n_components} components.")
    return umap_results, umap_model

def save_metrics_to_file(output_dir, filename, metrics, img_id):
    """
    Saves the measurement metrics to a text file.
    """
    img_output_dir = os.path.join(output_dir, f"image_{clean_filename(metrics['image_filename'])}_{img_id}")
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
            metrics = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in metrics.items()}
            writer.writerow(metrics)
    
    print(f"All metrics saved to {output_file}")

def save_metrics_to_jsonl(jsonl_output_path, all_metrics):
    """
    Saves each metric dictionary as a separate JSON object per line in a JSONL file.

    Parameters:
        jsonl_output_path (str): Path to the output JSONL file.
        all_metrics (list): List of metric dictionaries.
    """
    import json
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
        
        print(f"Parsed grouping file with {len(group_mapping)} entries.")
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
    
    return counts_per_specimen, counts_per_group

def compute_statistics(counts_per_group):
    """
    Computes mean, standard deviation, and range for counts per group.

    Parameters:
        counts_per_group (dict): Counts per species/group.

    Returns:
        dict: Statistics per group.
    """
    stats_per_group = {}
    
    for group, counts in counts_per_group.items():
        counts_array = np.array(counts)
        mean = np.mean(counts_array)
        std = np.std(counts_array)
        range_ = (int(np.min(counts_array)), int(np.max(counts_array)))
        stats_per_group[group] = {
            'mean': mean,
            'std': std,
            'range': range_,
            'specimens_examined': len(counts)
        }
    
    return stats_per_group

def save_group_statistics(stats_per_group, output_dir, category_name):
    """
    Saves the statistics per group to a text file.

    Parameters:
        stats_per_group (dict): Statistics per species/group.
        output_dir (str): Directory to save the statistics file.
        category_name (str): The category name for labeling.

    Returns:
        None
    """
    stats_file = os.path.join(output_dir, f"{category_name}_group_statistics.txt")
    with open(stats_file, 'w') as f:
        for group, stats in stats_per_group.items():
            f.write(f"{group}: {category_name} mean {stats['mean']:.2f} SD {stats['std']:.2f} range {stats['range']} specimens examined {stats['specimens_examined']}\n")
    print(f"Group statistics saved to {stats_file}")

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

# === Existing Functions ===
# All existing functions from your script remain unchanged
# (For brevity, assume all existing functions are present as provided in your initial script)
# === End of Existing Functions ===

# === Utility Functions ===

def clean_filename(filename):
    """
    Cleans the filename by removing or replacing invalid characters.
    """
    return re.sub(r'[^\w\-_\. ]', '_', filename)

# === Modified Main Function ===

def main():
    args = parse_args()
    annotations_dict, images_info, categories = load_annotations(args.json, args.image_dir, args.image_id, args.category_name)

    if not annotations_dict:
        if args.image_id:
            print(f"No annotations found for image ID {args.image_id}.")
        else:
            print("No annotations found in the JSON file.")
        return

    # Ensure EasyOCR models are present and initialize the reader
    reader = ensure_easyocr_models_present(languages=['en'])

    # Set the cache directory for torch.hub (Metric3D models)
    torch_cache_dir = os.path.join(script_dir, '.torch')
    os.environ['TORCH_HOME'] = torch_cache_dir
    os.makedirs(torch_cache_dir, exist_ok=True)

    # Initialize depth estimation model if needed
    if args.generate_depth:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
        model.to(device).eval()

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    if args.depth_output_dir:
        os.makedirs(args.depth_output_dir, exist_ok=True)
    if args.annotated_output_dir:
        os.makedirs(args.annotated_output_dir, exist_ok=True)

    # Parse the grouping file if provided
    if args.grouping_file:
        group_mapping = parse_grouping_file(args.grouping_file)
    else:
        group_mapping = {}
        print("No grouping file provided. Proceeding without grouping information.")

    # Collect all metrics
    all_metrics = []
    images_list = []
    masks_list = []
    categories_set = set()

    for img_id, annotations in annotations_dict.items():
        image_info = images_info.get(img_id)
        if image_info is None:
            print(f"No image found with ID {img_id} in the JSON annotations.")
            continue

        # Get the image filename from the metadata
        image_filename = image_info['file_name']
        image_basename = os.path.basename(os.path.normpath(image_filename))  # Extract base filename without directories

        image_path = os.path.join(args.image_dir, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found: {image_path}")
            continue
        height, width = image.shape[:2]

        if not annotations:
            print(f"Image ID ('{image_filename}') {img_id}: No annotations found.")
            continue

        # === New Integration: EXIF-based Intrinsics Extraction ===
        exif_intrinsics = extract_exif_and_intrinsics(image_path)
        if exif_intrinsics and exif_intrinsics["Camera Intrinsics"]["fx"] and exif_intrinsics["Camera Intrinsics"]["fy"]:
            camera_intrinsics = (
                exif_intrinsics["Camera Intrinsics"]["fx"],
                exif_intrinsics["Camera Intrinsics"]["fy"],
                exif_intrinsics["Camera Intrinsics"]["cx"],
                exif_intrinsics["Camera Intrinsics"]["cy"]
            )
            print(f"EXIF-based Camera Intrinsics for image {image_basename}: fx={camera_intrinsics[0]}, fy={camera_intrinsics[1]}, cx={camera_intrinsics[2]}, cy={camera_intrinsics[3]}")
        else:
            print("EXIF data incomplete or not available. Falling back to scale bar detection or user-provided intrinsics.")
            # === Existing Scale Bar Detection and Intrinsics Estimation ===
            scale_length_pixels, scale_bar_bbox, scale_bar_contour = find_scale_bar_length_in_pixels(image, args)
            scale_length_pixels, pixels_per_unit = find_scale_bar_length_and_label(image, scale_bar_bbox, reader, args)

            if scale_length_pixels and pixels_per_unit:
                scale_length_mm = scale_length_pixels / pixels_per_unit
                # Estimate camera intrinsics
                camera_intrinsics = estimate_camera_intrinsics(image.shape, scale_length_pixels, scale_length_mm)
                print(f"Estimated camera intrinsics for image {image_basename}: fx={camera_intrinsics[0]}, fy={camera_intrinsics[1]}, cx={camera_intrinsics[2]}, cy={camera_intrinsics[3]}")
            else:
                print("Scale bar not detected or failed to parse label.")
                # Check if user provided intrinsics via command-line arguments
                if args.focal_length and args.sensor_width and args.sensor_height:
                    focal_length_mm = args.focal_length
                    sensor_width_mm = args.sensor_width
                    sensor_height_mm = args.sensor_height
                    pixel_size_mm = args.pixel_size / 1000.0 if args.pixel_size else None

                    # Calculate fx and fy
                    fx = (focal_length_mm * width) / sensor_width_mm
                    fy = (focal_length_mm * height) / sensor_height_mm

                    # Principal point
                    cx, cy = width / 2, height / 2

                    camera_intrinsics = (fx, fy, cx, cy)
                    print(f"User-provided Camera Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                elif args.focal_length and (args.pixel_size or args.sensor_width):
                    # Partial intrinsics provided; attempt to compute missing parameters
                    focal_length_mm = args.focal_length
                    sensor_width_mm = args.sensor_width if args.sensor_width else None
                    sensor_height_mm = args.sensor_height if args.sensor_height else None
                    pixel_size_mm = args.pixel_size / 1000.0 if args.pixel_size else None

                    if pixel_size_mm and sensor_width_mm and sensor_height_mm:
                        # Calculate fx and fy using sensor size and pixel size
                        sensor_width_mm = pixel_size_mm * width
                        sensor_height_mm = pixel_size_mm * height
                        fx = (focal_length_mm * width) / sensor_width_mm
                        fy = (focal_length_mm * height) / sensor_height_mm
                        cx, cy = width / 2, height / 2
                        camera_intrinsics = (fx, fy, cx, cy)
                        print(f"Computed Camera Intrinsics from user-provided focal length and pixel size: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                    else:
                        print("Insufficient data provided via command-line arguments to compute camera intrinsics.")
                        print("Please provide focal_length, sensor_width, and sensor_height, or pixel_size.")
                        sys.exit(1)
                else:
                    print("No valid method to obtain camera intrinsics. Please provide EXIF data, a scale bar, or necessary command-line arguments.")
                    sys.exit(1)
        # === End of New Integration ===

        # Load or generate depth map
        if args.depth_dir:
            depth_map = load_depth_map(image_basename, args.depth_dir)
            adjusted_intrinsics = camera_intrinsics  # No adjustment needed
        elif args.generate_depth:
            depth_map, adjusted_intrinsics = generate_depth_map(image, model, device, camera_intrinsics)
        else:
            print(f"No depth map provided for image {image_basename}. Skipping.")
            continue

        if depth_map is None:
            print(f"Depth map not available for image {image_basename}. Skipping.")
            continue

        # Calibrate depth map if scale bar was detected
        if 'scale_length_mm' in locals() and scale_length_pixels and scale_length_mm and scale_bar_contour is not None:
            scale_bar_mask = np.zeros_like(depth_map, dtype=np.uint8)
            cv2.drawContours(scale_bar_mask, [scale_bar_contour], -1, 255, -1)
            depth_map, scaling_factor = calibrate_depth_map(depth_map, scale_bar_mask, scale_length_mm, scale_length_pixels)

            if scaling_factor is not None:
                print(f"Depth map calibrated with scaling factor {scaling_factor}")
            else:
                print("Calibration failed; proceeding with uncalibrated depth map.")
        else:
            print("Scale bar not detected or calibration failed; proceeding with uncalibrated depth map.")

        # Save depth maps if required
        if args.save_depth:
            depth_output_dir = args.depth_output_dir if args.depth_output_dir else args.output_dir
            depth_filename = f"{os.path.splitext(image_basename)[0]}_depth.npy"
            depth_path = os.path.join(depth_output_dir, depth_filename)
            np.save(depth_path, depth_map)
            print(f"Depth map saved to {depth_path}")

            # Save as .png for visualization
            depth_visual = depth_map.copy()
            depth_visual = depth_visual - np.min(depth_visual)
            depth_visual = depth_visual / np.max(depth_visual)
            depth_visual = (depth_visual * 255).astype(np.uint8)
            depth_image_filename = f"{os.path.splitext(image_basename)[0]}_depth.png"
            depth_image_path = os.path.join(depth_output_dir, depth_image_filename)
            cv2.imwrite(depth_image_path, depth_visual)
            print(f"Depth image saved to {depth_image_path}")

        for ann_idx, ann in enumerate(annotations):
            category_name = categories.get(ann['category_id'], 'Unknown')
            categories_set.add(category_name)
            mask = create_mask_from_annotation(ann, height, width)

            # Convert mask to binary
            binary_mask = (mask > 0).astype(np.uint8) * 255

            # Extract the object's contour from the mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f"Image ID {img_id}, Annotation {ann_idx+1}: No contours found.")
                continue

            object_contour = max(contours, key=cv2.contourArea)
            contour_points = object_contour.reshape(-1, 2)

            # Back-project 2D points into 3D
            contour_3d = backproject_2d_to_3d(contour_points, depth_map, adjusted_intrinsics)

            # Measure metrics in 3D
            metrics, pca = measure_metrics_3d(
                contour_3d=contour_3d,
                pixels_per_mm=pixels_per_unit,
                img_id=img_id,
                ann_idx=ann_idx,
                category_name=category_name,
                image_basename=image_basename
            )

            # Print metrics
            print(f"Image ID {img_id}, Annotation {ann_idx+1}:")
            print(f"Category: {category_name}")
            print(f"Method: 3D")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            print()

            # Save metrics to individual file if required
            if args.save_results:
                filename = f"{os.path.splitext(image_basename)[0]}_ann_{ann_idx+1}_{category_name}_3d_metrics.txt"
                save_metrics_to_file(args.output_dir, filename, metrics, img_id)

            # Collect metrics and corresponding images and masks for clustering and visualization
            if args.output_file or args.jsonl_output:
                all_metrics.append(metrics)
                images_list.append(image)
                masks_list.append(binary_mask)

            # Draw and save annotated images if required
            if args.save_annotated:
                annotated_output_dir = args.annotated_output_dir if args.annotated_output_dir else args.output_dir

                # Convert image to RGB for visualization
                annotated_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

                # Project 3D points back to 2D
                length_line_start = np.array(metrics['length_line_start'])
                length_line_end = np.array(metrics['length_line_end'])
                width_line_start = np.array(metrics['width_line_start'])
                width_line_end = np.array(metrics['width_line_end'])

                length_start_2d = project_3d_point_to_2d(length_line_start, adjusted_intrinsics)
                length_end_2d = project_3d_point_to_2d(length_line_end, adjusted_intrinsics)
                width_start_2d = project_3d_point_to_2d(width_line_start, adjusted_intrinsics)
                width_end_2d = project_3d_point_to_2d(width_line_end, adjusted_intrinsics)

                # Draw length line (Red)
                cv2.line(annotated_image, tuple(length_start_2d), tuple(length_end_2d), (255, 0, 0), 2)
                # Draw width line (Green)
                cv2.line(annotated_image, tuple(width_start_2d), tuple(width_end_2d), (0, 255, 0), 2)

                # Add text labels near the lines
                if metrics['length_pixels'] is not None:
                    cv2.putText(annotated_image, f"Length: {metrics['length_mm']:.2f} mm", 
                                (int(length_start_2d[0]), int(length_start_2d[1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if metrics['width_pixels'] is not None:
                    cv2.putText(annotated_image, f"Width: {metrics['width_mm']:.2f} mm", 
                                (int(width_start_2d[0]), int(width_start_2d[1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save annotated image
                annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                annotated_image_filename = f"{os.path.splitext(image_basename)[0]}_annotated.png"
                annotated_image_path = os.path.join(annotated_output_dir, annotated_image_filename)
                cv2.imwrite(annotated_image_path, annotated_image_bgr)
                print(f"Annotated image saved to {annotated_image_path}")

                # Save contour visualization
                visualize_contours(image, contour_points, args.output_dir, img_id, ann_idx, category_name, image_basename)

                # Save length and width visualization
                visualize_length_width(
                    image, 
                    contour_points, 
                    metrics["length_mm"], 
                    metrics["width_mm"], 
                    metrics["length_line_start"], 
                    metrics["length_line_end"],
                    metrics["width_line_start"], 
                    metrics["width_line_end"],
                    args.output_dir, 
                    img_id, 
                    ann_idx, 
                    category_name, 
                    image_basename
                )

                # Save scale bar visualization if detected
                if 'scale_length_pixels' in locals() and pixels_per_unit and scale_bar_contour is not None:
                    visualize_scale_bar(
                        image.copy(), 
                        scale_bar_contour, 
                        f"{scale_length_pixels} px / {pixels_per_unit:.2f} px/mm", 
                        args.output_dir, 
                        img_id, 
                        image_basename
                    )
                else:
                    print(f"Image ID {img_id}, Annotation {ann_idx+1}: No valid scale bar detected for visualization.")

    # === After Processing All Images and Annotations ===
    # Save all metrics to CSV if output_file is specified
    if args.output_file:
        save_all_metrics_to_csv(args.output_file, all_metrics)

    # Save all metrics to JSONL if jsonl_output is specified
    if args.jsonl_output:
        save_metrics_to_jsonl(args.jsonl_output, all_metrics)

    # === Counting Instances and Calculating Statistics ===
    if args.output_file or args.jsonl_output:
        counts_per_specimen, counts_per_group = count_instances(all_metrics, group_mapping, args.category_name)
        stats_per_group = compute_statistics(counts_per_group)
        save_group_statistics(stats_per_group, args.output_dir, args.category_name)

    # === Clustering and Visualization per Category ===
    # Convert all_metrics to DataFrame
    df = pd.DataFrame(all_metrics)

    # Select numerical features for clustering
    features = ['length_mm', 'width_mm']  # Add more features as needed

    # Drop rows with missing values
    df_clean = df.dropna(subset=features + ['category_name'])
    print(f"Data after dropping missing values: {len(df_clean)} entries.")

    # Check if there is enough data for clustering
    if df_clean.shape[0] < 2:
        print("Not enough data points for clustering and dimensionality reduction. Skipping these steps.")
        return

    # Group by category
    grouped = df_clean.groupby('category_name')

    for category, group in grouped:
        print(f"Processing category: {category}")

        # Extract features
        X = group[features].values

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Feature scaling completed.")

        # Perform PCA
        pca_model = PCA(n_components=2)
        principal_components = pca_model.fit_transform(X_scaled)
        group = group.copy()
        group['PC1'] = principal_components[:, 0]
        group['PC2'] = principal_components[:, 1]
        print("PCA transformation completed.")

        # Perform UMAP
        umap_results, umap_model = perform_umap(X_scaled, n_components=2)
        if umap_results is not None:
            group['UMAP1'] = umap_results[:, 0]
            group['UMAP2'] = umap_results[:, 1]

        # Collect corresponding images and masks
        indices = group.index.tolist()
        clustered_images = [images_list[i] for i in indices]
        clustered_masks = [masks_list[i] for i in indices]
        print(f"Collected {len(clustered_images)} images and masks for category '{category}'.")

        # Plot PCA with thumbnails
        plot_pca_umap(principal_components, clustered_images, clustered_masks, args.output_dir, category, method='PCA')

        # Plot UMAP with thumbnails if available
        if umap_results is not None:
            plot_pca_umap(umap_results, clustered_images, clustered_masks, args.output_dir, category, method='UMAP')
        else:
            print(f"UMAP results not available for category '{category}'. Skipping UMAP plotting.")

        # Perform DBSCAN
        dbscan_labels = perform_dbscan(X_scaled, args.output_dir, category, principal_components, clustered_images, clustered_masks)
        if dbscan_labels.size > 0:
            group = group.copy()
            group['DBSCAN_cluster'] = dbscan_labels
            print(f"DBSCAN labels assigned for category '{category}'.")
        else:
            print(f"DBSCAN clustering not performed for category '{category}' due to insufficient data.")

        # Perform Hierarchical Clustering
        if X_scaled.shape[0] >= 2:
            hierarchical = AgglomerativeClustering(n_clusters=min(3, X_scaled.shape[0]))
            hierarchical_labels = hierarchical.fit_predict(X_scaled)
            group['Hierarchical_cluster'] = hierarchical_labels
            print(f"Hierarchical Clustering labels assigned for category '{category}'.")

            # Visualize Hierarchical Clustering Dendrogram
            plot_hierarchical_dendrogram(X_scaled, args.output_dir, category)
        else:
            print(f"Not enough data points for hierarchical clustering in category '{category}'. Skipping.")

        # Optionally, save cluster labels back to CSV or JSONL
        if args.output_file:
            # Append to CSV without headers
            group.to_csv(args.output_file, mode='a', header=False, index=False)
            print(f"Cluster labels for category '{category}' appended to CSV: {args.output_file}")
        elif args.jsonl_output:
            group.to_json(args.jsonl_output, orient='records', lines=True, mode='a')
            print(f"Cluster labels for category '{category}' appended to JSONL: {args.jsonl_output}")

    # === Flatten Output Directory if Requested ===
    if args.flatten_output:
        flatten_output_directory(args.output_dir)
    # === End of Flattening ===

    print("Processing completed successfully.")

# === End of Modified Main Function ===

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
