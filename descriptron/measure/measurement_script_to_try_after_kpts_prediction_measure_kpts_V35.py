#!/usr/bin/env python
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
from math import sqrt
from scipy.spatial.distance import pdist, squareform
import shutil
import logging
import datetime
from scipy.spatial import cKDTree

# Florence-2 VLM for scale bar detection (optional, falls back to EasyOCR)
_FLORENCE2_MODEL = None
_FLORENCE2_PROCESSOR = None

def _load_florence2():
    global _FLORENCE2_MODEL, _FLORENCE2_PROCESSOR
    if _FLORENCE2_MODEL is not None:
        return _FLORENCE2_MODEL, _FLORENCE2_PROCESSOR
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _FLORENCE2_PROCESSOR = AutoProcessor.from_pretrained(
            'microsoft/Florence-2-base', trust_remote_code=True)
        _FLORENCE2_MODEL = AutoModelForCausalLM.from_pretrained(
            'microsoft/Florence-2-base', trust_remote_code=True).to(device)
        logging.info(f"Florence-2-base loaded on {device}")
        return _FLORENCE2_MODEL, _FLORENCE2_PROCESSOR
    except Exception as e:
        logging.warning(f"Florence-2 unavailable ({e}), will use EasyOCR only")
        return None, None

# ----------------------------
# Configure Logging
# ----------------------------
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO or WARNING to reduce verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

###############################
# NEW HELPER FUNCTIONS
###############################

def process_keypoints(annotation, img_id, ann_idx, category_name, mm_scale=None):
    """
    Processes keypoints to calculate pairwise distances.
    """
    keypoints = annotation.get('keypoints', [])
    num_keypoints = annotation.get('num_keypoints', 0)

    if len(keypoints) != num_keypoints * 3:
        logging.warning(f"Annotation {ann_idx + 1} for Image ID {img_id} ({category_name}) has inconsistent keypoints data.")
        return None

    # Extract (x, y) coords of visible keypoints
    keypoint_coords = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i:i+3]
        if v > 0:
            keypoint_coords.append((x, y))

    if len(keypoint_coords) < 2:
        logging.warning(f"Annotation {ann_idx + 1} for Image ID {img_id} ({category_name}) has <2 visible keypoints.")
        return None

    # Distance matrix in pixels
    distance_matrix = np.linalg.norm(
        np.array(keypoint_coords)[:, np.newaxis, :] - np.array(keypoint_coords)[np.newaxis, :, :],
        axis=2
    )
    distances = distance_matrix.flatten()

    # Summary (pixel-based)
    summary = {
        'num_visible_keypoints': len(keypoint_coords),
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'max_distance': float(np.max(distances)),
        'min_distance': float(np.min(distances)),
        'median_distance': float(np.median(distances)),
        'distance_matrix': distance_matrix.tolist(),  # original pixel matrix
    }

    # We initially set image_filename to None or 'Unknown'.
    # We'll override it in the main loop to the actual image basename.
    keypoints_metrics = {
        "image_id": img_id,
        "annotation_index": ann_idx + 1,
        "category_id": annotation['category_id'],
        "category_name": category_name,
        "method": "Keypoints Distance Matrix",
        "keypoints_distance_matrix": distance_matrix.tolist(),  # in pixels
        "image_filename": "Unknown",  # Will be fixed in main
        "summary": summary
    }

    # If mm_scale is provided, add mm-based data
    if mm_scale and mm_scale > 0:
        distance_matrix_mm = distance_matrix / mm_scale
        distances_mm = distance_matrix_mm.flatten()

        summary['mean_distance_mm'] = float(distances_mm.mean())
        summary['std_distance_mm'] = float(distances_mm.std())
        summary['max_distance_mm'] = float(distances_mm.max())
        summary['min_distance_mm'] = float(distances_mm.min())
        summary['median_distance_mm'] = float(np.median(distances_mm))

        # Store entire matrix in mm
        keypoints_metrics["keypoints_distance_matrix_mm"] = distance_matrix_mm.tolist()
    else:
        keypoints_metrics["keypoints_distance_matrix_mm"] = None

    return keypoints_metrics


def create_mask_from_annotation(annotation, height, width, img_id, ann_idx, category_name):
    """
    Creates a binary mask from an annotation.
    """
    if 'segmentation' not in annotation or not annotation['segmentation']:
        logging.warning(f"Annotation {ann_idx + 1} for Image ID {img_id} ({category_name}) has no segmentation data.")
        return None

    segmentation = annotation['segmentation']

    try:
        if isinstance(segmentation, list):
            valid_polygons = [poly for poly in segmentation if len(poly) >= 6]
            if not valid_polygons:
                logging.warning(f"Annotation {ann_idx + 1} for Image ID {img_id} ({category_name}) has no valid polygons.")
                return None
            rles = maskUtils.frPyObjects(valid_polygons, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segmentation, dict):
            if 'counts' not in segmentation or 'size' not in segmentation:
                logging.warning(f"Annotation {ann_idx + 1} for Image ID {img_id} ({category_name}) has invalid RLE segmentation.")
                return None
            rle = segmentation
        else:
            logging.warning(f"Annotation {ann_idx + 1} for Image ID {img_id} ({category_name}) has unknown segmentation format.")
            return None

        mask = maskUtils.decode(rle)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask
    except Exception as e:
        logging.error(f"Failed to create mask for Annotation {ann_idx + 1} in Image ID {img_id} ({category_name}): {e}")
        return None


def visualize_keypoints_distances(image, keypoint_coords, distance_matrix, output_dir, img_id, ann_idx, category_name, image_basename, mm_scale=None):
    """
    Visualizes keypoints and draws lines between them representing distances.
    Automatically shows distances in mm if mm_scale is provided (and > 0).
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    x_coords, y_coords = zip(*keypoint_coords)
    plt.scatter(x_coords, y_coords, c='red', s=100, label='Keypoints')

    num_keypoints = len(keypoint_coords)
    for i in range(num_keypoints):
        for j in range(i + 1, num_keypoints):
            x1, y1 = keypoint_coords[i]
            x2, y2 = keypoint_coords[j]
            dist_px = distance_matrix[i][j]
            if mm_scale and mm_scale > 0:
                dist_mm = dist_px / mm_scale
                label_str = f"{dist_mm:.1f} mm"
            else:
                label_str = f"{dist_px:.1f} px"
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=1)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            plt.text(mid_x, mid_y, label_str, color='yellow', fontsize=8)

    plt.title(f"Keypoints Distance Matrix for Image {image_basename} (ID: {img_id}), Annotation {ann_idx + 1}")
    plt.legend()
    plt.axis('off')

    img_output_dir = os.path.join(output_dir, f"image_{clean_filename(image_basename)}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    save_path = os.path.join(
        img_output_dir,
        f"ann_{clean_filename(image_basename)}_{ann_idx + 1}_{category_name}_keypoints_distance.png"
    )
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Keypoints distance visualization saved to {save_path}")


# === Updated function to match the same subfolder as the .png ===
def save_distance_matrices_npy_individual(
    output_dir,
    distance_matrix_px,
    distance_matrix_mm,
    keypoints_metrics,
    image_basename
):
    """
    Saves the keypoints distance matrices (px and mm) as .npy files for one annotation.
    Places them in the same folder as the .png output, using the real image basename.
    """
    img_id = keypoints_metrics["image_id"]
    ann_idx = keypoints_metrics["annotation_index"]
    category = keypoints_metrics["category_name"]

    # Folder where the .png is stored
    cleaned_basename = clean_filename(image_basename)
    img_output_dir = os.path.join(output_dir, f"image_{cleaned_basename}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)

    # Use the same "ann_..." naming
    file_prefix = f"ann_{cleaned_basename}_{ann_idx}_{category}"

    # Pixel-based .npy
    px_path = os.path.join(img_output_dir, file_prefix + "_distance_matrix_px.npy")
    np.save(px_path, distance_matrix_px)
    logging.info(f"Saved pixel distance matrix to {px_path}")

    # If we have an mm-based matrix, save that too
    if distance_matrix_mm is not None:
        mm_path = os.path.join(img_output_dir, file_prefix + "_distance_matrix_mm.npy")
        np.save(mm_path, distance_matrix_mm)
        logging.info(f"Saved mm distance matrix to {mm_path}")


###############################
# End New Helper Functions
###############################


def parse_args():
    parser = argparse.ArgumentParser(description='Measure length and width of objects in images.')
    parser.add_argument('--json', required=True, help='Path to the COCO JSON file')
    parser.add_argument('--image_dir', required=True, help='Directory containing the images')
    parser.add_argument('--output_dir', required=False, help='Directory to save outputs', default='./outputs')
    parser.add_argument('--image_id', type=int, required=False, help='ID of the image to process (optional)')
    parser.add_argument('--category_name', required=False, help='Name of the category to process', default=None)
    parser.add_argument('--method', choices=['skeleton', 'pca'], default='pca',
                        help='Method to measure length and width: "skeleton" or "pca"')
    parser.add_argument('--curve_measure_method', choices=['skeleton', 'bboxes'], default='skeleton',
                        help='For segmentation masks: use the standard skeleton method or the new overlapping bboxes method to measure curve length')
    parser.add_argument('--skeleton_method', choices=['opencv', 'skimage'], default='opencv',
                        help='Skeletonization method to use: "opencv" or "skimage"')
    parser.add_argument('--trim_branches', action='store_true', help='Enable branch trimming in skeletonization')
    parser.add_argument('--save_results', action='store_true', help='Save measurement results to individual files')
    parser.add_argument('--output_file', type=str, required=False, help='Path to a single output CSV file for measurements')
    parser.add_argument('--jsonl_output', type=str, required=False, help='Path to the output JSONL file for measurements', default=None)
    parser.add_argument('--min_aspect_ratio', type=float, default=5.0, help='Minimum aspect ratio for scale bar detection')
    parser.add_argument('--max_aspect_ratio', type=float, default=200.0, help='Maximum aspect ratio for scale bar detection')
    parser.add_argument('--min_width', type=int, default=10, help='Minimum width in pixels for scale bar detection')
    parser.add_argument('--mm_scale', type=float, required=False, help='Pixels per mm ratio (if available); if provided, distances will be shown in mm', default=None)
    parser.add_argument('--grouping_file', type=str, required=False, help='Path to the grouping file (CSV, TSV, or JSON)', default=None)
    parser.add_argument('--max_images', type=int, required=False, help='Process at most N images (for testing)', default=None)
    return parser.parse_args()


def load_annotations(json_path, image_id=None, category_name=None):
    with open(json_path, 'r') as f:
        coco = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    images = {img['id']: img for img in coco['images']}
    
    if image_id is not None:
        image_ids = [image_id]
    else:
        image_ids = list(images.keys())
    
    annotations_per_image = {}
    for img_id in image_ids:
        img = images.get(img_id)
        if img is None:
            continue
        
        image_filename = img['file_name']
        if '/' in image_filename or '\\' in image_filename:
            logging.info(f"Skipping image '{image_filename}' as it is in a subdirectory.")
            continue
        
        anns = [ann for ann in coco['annotations'] if ann['image_id'] == img_id]
        if category_name:
            category_ids = [cat_id for cat_id, name in categories.items() if name.lower() == category_name.lower()]
            anns = [ann for ann in anns if ann['category_id'] in category_ids]
        if anns:
            annotations_per_image[img_id] = anns
    
    return annotations_per_image, images, categories


# --- Modified EasyOCR initialization function with GPU check ---
def ensure_easyocr_models_present(languages=['en'], cache_dir=None):
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.EasyOCR")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    models_needed = False
    for lang in languages:
        detector_dir = os.path.join(cache_dir, lang, 'detector')
        recognizer_dir = os.path.join(cache_dir, lang, 'recognizer')
        if not os.path.isdir(detector_dir) or not os.path.isdir(recognizer_dir):
            models_needed = True
            break
    
    if models_needed:
        print("EasyOCR models not found. Downloading models, please wait...")
        reader = easyocr.Reader(languages, model_storage_directory=cache_dir, gpu=use_gpu)
        print("EasyOCR models downloaded successfully.")
    else:
        print("EasyOCR models already present. Skipping download.")
        reader = easyocr.Reader(languages, model_storage_directory=cache_dir, gpu=use_gpu)
    
    return reader


def find_scale_bar_length_in_pixels(image, args):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ranges = [
        {"lower": np.array([0, 0, 200]), "upper": np.array([180, 25, 255])},
        {"lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 50])},
        {"lower": np.array([0, 0, 200]), "upper": np.array([180, 25, 255])}
    ]
    
    for color_range in ranges:
        mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for scale_bar_contour in sorted(contours, key=cv2.contourArea, reverse=True):
                x, y, w, h = cv2.boundingRect(scale_bar_contour)
                if h < w and h > 0:
                    aspect_ratio = w / h
                    if w >= args.min_width and (args.min_aspect_ratio <= aspect_ratio <= args.max_aspect_ratio):
                        return w, (x, y, w, h), scale_bar_contour
    return None, None, None


def find_scale_bar_length_and_label(image, scale_bar_bbox, reader, args):
    if scale_bar_bbox is None:
        print("No scale bar bounding box provided.")
        return None, None
    x, y, w, h = scale_bar_bbox
    label_roi_y_start = max(y - h - 50, 0)
    label_roi_y_end = y
    label_roi_x_start = max(x - 50, 0)
    label_roi_x_end = x + w + 50
    label_roi = image[label_roi_y_start:label_roi_y_end, label_roi_x_start:label_roi_x_end]
    text = extract_scale_text(label_roi, reader)
    scale_length, scale_unit = parse_scale_text(text)
    if scale_length and scale_unit:
        pixels_per_unit = calculate_pixels_per_mm(w, scale_length, scale_unit)
        return w, pixels_per_unit
    else:
        print("Failed to parse scale bar label.")
        return w, None


def extract_scale_text(roi_image, reader):
    roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
    #results = reader.readtext(roi_rgb)
    # only let EasyOCR output digits, dots, commas and µ/u/m characters
    results = reader.readtext(roi_rgb,
                              allowlist='0123456789.,µum',
                              detail=1)
    text = ""
    highest_confidence = 0.0
    for res in results:
        confidence = res[2]
        detected_text = res[1]
        if confidence > highest_confidence:
            highest_confidence = confidence
            text = detected_text
    logging.debug(f"Extracted OCR Text: '{text}' with confidence {highest_confidence:.2f}")
    return text


def parse_scale_text(text):
#    text = text.replace(',', '.')
#    match = re.search(r'(\d+(\.\d+)?)\s*(mm|µm|um)', text.lower())
#    if match:
#        scale_length = float(match.group(1))
#        scale_unit = match.group(3)
#        return scale_length, scale_unit
#    return None, None
    # unify comma → dot
    txt = text.replace(',', '.').lower()

    # 1) try to find a decimal number first (xxx.yyy)
    m = re.search(r'(\d+\.\d+)\s*(mm|µm|um)', txt)
    if m:
        return float(m.group(1)), m.group(2)

    # 2) fallback to an integer if no decimal was found
    m2 = re.search(r'(\d+)\s*(mm|µm|um)', txt)
    if m2:
        return float(m2.group(1)), m2.group(2)

    return None, None


def calculate_pixels_per_mm(scale_bar_length_pixels, scale_length, scale_unit):
    if scale_unit == 'mm':
        return scale_bar_length_pixels / scale_length
    elif scale_unit in ['µm', 'um']:
        return scale_bar_length_pixels / (scale_length / 1000.0)
    else:
        return None


def clean_filename(filename):
    if not isinstance(filename, (str, bytes)):
        filename = str(filename)
    cleaned = re.sub(r'[^\w\-_.]', '_', filename)
    return cleaned


######## Patches to integrate global EasyOCR search and localized scale bar detection

# 1. Add this helper function near the other definitions (e.g., after `calculate_pixels_per_mm`):

def detect_image_scale(image, reader, args, expand_px=100):
    """
    Detect scale length in pixels per unit by: 1) OCR entire image for text like '50 µm' or '2 mm';
    2) For each detected text, crop a region expanded by `expand_px` around the text bbox;
    3) Run flat-line detection (`find_scale_bar_length_in_pixels`) on that region;
    4) Compute and return pixels_per_unit using `calculate_pixels_per_mm`.
    Returns None if detection fails.
    """
    # Convert to RGB for EasyOCR
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ocr_results = reader.readtext(rgb)
    for bbox, text, conf in ocr_results:
        # Try parsing the OCR text for numeric scale
        length_val, unit = parse_scale_text(text)
        if length_val and unit:
            # Compute text bounding box coords
            xs = [int(pt[0]) for pt in bbox]
            ys = [int(pt[1]) for pt in bbox]
            x_min, y_min = max(min(xs) - expand_px, 0), max(min(ys) - expand_px, 0)
            x_max, y_max = min(max(xs) + expand_px, image.shape[1]), min(max(ys) + expand_px, image.shape[0])
            roi = image[y_min:y_max, x_min:x_max]
            # Detect scale bar in the ROI
            bar_w, bar_bbox, bar_contour = find_scale_bar_length_in_pixels(roi, args)
            if bar_w and length_val:
                pixels_per_unit = calculate_pixels_per_mm(bar_w, length_val, unit)
                logging.info(f"Detected scale: {length_val}{unit} => {pixels_per_unit:.2f} px/unit")
                return pixels_per_unit
    logging.warning("Scale detection via OCR + bar search failed. Using default mm_scale if provided.")
    return None


def detect_image_scale_florence2(image, args, expand_px=100):
    """
    Use Florence-2 VLM for scale bar detection. Reads text via OCR_WITH_REGION
    to get text + bounding box, then finds the scale bar line near the text.
    Falls back to None if Florence-2 is unavailable or detection fails.

    Returns (pixels_per_unit, scale_text, bar_contour_in_full_image, text_bbox)
    or (None, None, None, None) on failure.
    """
    model, processor = _load_florence2()
    if model is None:
        return None, None, None, None

    import torch
    device = next(model.parameters()).device
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    inputs = processor(text='<OCR_WITH_REGION>', images=pil_img,
                       return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300)
    decoded = processor.batch_decode(out, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        decoded, task='<OCR_WITH_REGION>',
        image_size=(pil_img.width, pil_img.height))

    ocr_data = result.get('<OCR_WITH_REGION>', {})
    labels = ocr_data.get('labels', [])
    quad_boxes = ocr_data.get('quad_boxes', [])

    for label, qbox in zip(labels, quad_boxes):
        text = label.replace('</s>', '').strip()
        length_val, unit = parse_scale_text(text)
        if not (length_val and unit):
            continue
        xs = [qbox[i] for i in range(0, len(qbox), 2)]
        ys = [qbox[i] for i in range(1, len(qbox), 2)]
        x_min = max(int(min(xs)) - expand_px, 0)
        y_min = max(int(min(ys)) - expand_px, 0)
        x_max = min(int(max(xs)) + expand_px, image.shape[1])
        y_max = min(int(max(ys)) + expand_px, image.shape[0])
        roi = image[y_min:y_max, x_min:x_max]
        bar_w, bar_bbox, bar_contour = find_scale_bar_length_in_pixels(roi, args)
        if bar_w and length_val:
            pixels_per_unit = calculate_pixels_per_mm(bar_w, length_val, unit)
            logging.info(f"Florence-2 scale: '{text}' => {length_val}{unit} => {pixels_per_unit:.2f} px/unit")
            shifted_contour = None
            if bar_contour is not None:
                shifted_contour = bar_contour.copy()
                shifted_contour[:, :, 0] += x_min
                shifted_contour[:, :, 1] += y_min
            text_bbox = (int(min(xs)), int(min(ys)),
                         int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            return pixels_per_unit, f"{length_val} {unit}", shifted_contour, text_bbox

    logging.info("Florence-2 OCR found no scale text, falling back to EasyOCR")
    return None, None, None, None


#############################################################################################################


def compute_inter_mask_distances(masks_data, px_per_mm=None):
    """
    Compute pairwise distances between all segmentation masks on one image.

    Args:
        masks_data: list of dicts, each with:
            'mask': binary ndarray (H,W), 'category_name': str,
            'annotation_index': int, 'contour': ndarray (N,2)
        px_per_mm: pixels-per-mm scale factor (or None)

    Returns:
        list of dicts, one per pair:
            category_a, category_b, ann_idx_a, ann_idx_b,
            min_boundary_distance_px, centroid_distance_px,
            (+ _mm variants if px_per_mm is set),
            overlap: bool
    """
    n = len(masks_data)
    results = []
    if n < 2:
        return results

    centroids = []
    contour_points = []
    for md in masks_data:
        mask = md['mask']
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            ys, xs = np.where(mask > 0)
            cx, cy = (float(xs.mean()), float(ys.mean())) if len(xs) > 0 else (0, 0)
        centroids.append((cx, cy))
        if md.get('contour') is not None and len(md['contour']) > 0:
            contour_points.append(md['contour'].reshape(-1, 2).astype(np.float64))
        else:
            pts = np.column_stack(np.where(mask > 0))[:, ::-1].astype(np.float64)
            contour_points.append(pts)

    for i in range(n):
        for j in range(i + 1, n):
            cx_dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 +
                              (centroids[i][1] - centroids[j][1])**2)

            overlap = bool(np.any(
                (masks_data[i]['mask'] > 0) & (masks_data[j]['mask'] > 0)))

            if overlap:
                min_bd = 0.0
            else:
                tree_j = cKDTree(contour_points[j])
                dists_i, _ = tree_j.query(contour_points[i], k=1)
                min_bd = float(dists_i.min())

            pair = {
                'category_a': masks_data[i]['category_name'],
                'ann_idx_a': masks_data[i]['annotation_index'],
                'category_b': masks_data[j]['category_name'],
                'ann_idx_b': masks_data[j]['annotation_index'],
                'centroid_distance_px': float(cx_dist),
                'min_boundary_distance_px': float(min_bd),
                'overlap': overlap,
            }
            if px_per_mm and px_per_mm > 0:
                pair['centroid_distance_mm'] = float(cx_dist / px_per_mm)
                pair['min_boundary_distance_mm'] = float(min_bd / px_per_mm)
            results.append(pair)

    return results


def measure_metrics_pca(contour, mask, pixels_per_mm=None, img_id=None, ann_idx=None, category_name=None, category_id=None, image_basename=None):
    contour = contour.reshape(-1, 2)
    pca = PCA(n_components=2)
    pca.fit(contour)
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    projections_pc1 = contour @ pc1
    idx_max_pc1 = np.argmax(projections_pc1)
    idx_min_pc1 = np.argmin(projections_pc1)
    point_max_pc1 = contour[idx_max_pc1]
    point_min_pc1 = contour[idx_min_pc1]
    length = np.linalg.norm(point_max_pc1 - point_min_pc1)
    
    theta = -np.arctan2(pc1[1], pc1[0])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    rotated_contour = contour @ rotation_matrix.T
    x_vals = rotated_contour[:, 0]
    y_vals = rotated_contour[:, 1]
    x_bins = np.linspace(x_vals.min(), x_vals.max(), num=100)
    digitized = np.digitize(x_vals, x_bins)
    max_width = 0
    x_at_max_width = None
    y_top = None
    y_bottom = None
    for bin_num in np.unique(digitized):
        y_in_bin = y_vals[digitized == bin_num]
        if len(y_in_bin) > 0:
            y_min = y_in_bin.min()
            y_max = y_in_bin.max()
            width = y_max - y_min
            if width > max_width:
                max_width = width
                x_at_max_width = x_bins[bin_num - 1]
                y_top = y_max
                y_bottom = y_min
    height = max_width

    rotation_matrix_inv = np.array([[np.cos(-theta), -np.sin(-theta)],
                                    [np.sin(-theta), np.cos(-theta)]])
    point_top = np.array([x_at_max_width, y_top]) @ rotation_matrix_inv.T if x_at_max_width is not None else [None, None]
    point_bottom = np.array([x_at_max_width, y_bottom]) @ rotation_matrix_inv.T if x_at_max_width is not None else [None, None]

    length_mm = length / pixels_per_mm if (pixels_per_mm and pixels_per_mm > 0) else None
    height_mm = height / pixels_per_mm if (pixels_per_mm and pixels_per_mm > 0) else None
    length_to_height_ratio = length / height if height != 0 else None
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    rect_area = length * height if (length and height) else None
    extent = area / rect_area if (rect_area and rect_area != 0) else None
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if (hull_area and hull_area != 0) else None
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    angle = np.degrees(np.arctan2(pc1[1], pc1[0]))
    MA = 2 * np.sqrt(pca.explained_variance_[0]) * np.sqrt(len(contour))
    ma = 2 * np.sqrt(pca.explained_variance_[1]) * np.sqrt(len(contour))
    
    if pixels_per_mm and pixels_per_mm > 0:
        area_mm2 = area / (pixels_per_mm ** 2)
        perimeter_mm = perimeter / pixels_per_mm
        equivalent_diameter_mm = equivalent_diameter / pixels_per_mm
        MA_mm = MA / pixels_per_mm
        ma_mm = ma / pixels_per_mm
    else:
        area_mm2 = perimeter_mm = equivalent_diameter_mm = MA_mm = ma_mm = None
    
    metrics = {
        "image_id": img_id,
        "annotation_index": ann_idx + 1,
        "category_id": category_id,
        "category_name": category_name,
        "method": "PCA",
        "length_pixels": length,
        "height_pixels": height,
        "length_to_height_ratio": length_to_height_ratio,
        "area_pixels": area,
        "perimeter_pixels": perimeter,
        "aspect_ratio": length / height if height != 0 else None,
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
        "height_line_start": point_bottom.tolist() if point_bottom[0] is not None else None,
        "height_line_end": point_top.tolist() if point_top[0] is not None else None
    }
    return metrics


def preprocess_mask_for_skeleton(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def skeletonize_skimage(mask):
    mask_bool = mask.astype(bool)
    skeleton = skeletonize(mask_bool)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    return skeleton_uint8


def measure_length_skeleton(skeleton):
    return np.count_nonzero(skeleton)


def measure_width_skeleton(mask, skeleton):
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    radius = distance * (skeleton > 0)
    width_map = radius * 2
    non_zero_count = np.count_nonzero(width_map)
    if non_zero_count == 0:
        average_width = 0
    else:
        average_width = np.sum(width_map) / non_zero_count
    return average_width


def remove_branch_points(skeleton):
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]], dtype=np.uint8)
    while True:
        neighbor_count = cv2.filter2D(skeleton, -1, kernel)
        branch_points = np.where(neighbor_count > 12, 255, 0).astype(np.uint8)
        if not np.any(branch_points):
            break
        skeleton = cv2.subtract(skeleton, branch_points)
    return skeleton


def visualize_contours(image, contour, output_dir, img_id, ann_idx, category_name, image_basename):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if contour.ndim != 2 or contour.shape[1] != 2:
        logging.warning(f"Contour for Image {image_basename} {img_id}, Annotation {ann_idx+1} is invalid. Skipping visualization.")
        plt.close()
        return
    plt.plot(np.append(contour[:, 0], contour[0, 0]), np.append(contour[:, 1], contour[0, 1]), '-o', color='blue', linewidth=2)
    plt.scatter(contour[0, 0], contour[0, 1], color='red', s=100, zorder=5)
    num_points = len(contour)
    label_step = max(1, num_points // 12)
    for j in range(0, num_points, label_step):
        plt.text(contour[j, 0], contour[j, 1], str(j + 1), fontsize=12, color='blue', zorder=10)
    plt.title(f"Contour for Image {image_basename} {img_id}, Annotation {ann_idx+1}, Category: {category_name}")
    plt.axis('off')
    img_output_dir = os.path.join(output_dir, f"image_{image_basename}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    save_path = os.path.join(img_output_dir, f"ann_{image_basename}_{ann_idx+1}_{category_name}_contour.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Contour visualization saved to {save_path}")


def visualize_length_height(image, contour, length, height, length_mm, height_mm, orientation_degrees, output_dir, img_id, ann_idx, category_name, cleaned_image_basename, length_line_start, length_line_end, height_line_start, height_line_end):
    if contour.ndim != 2 or contour.shape[1] != 2:
        contour = contour.reshape(-1, 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.plot(np.append(contour[:, 0], contour[0, 0]), np.append(contour[:, 1], contour[0, 1]), 'b-', linewidth=2)
    if length_line_start is not None and length_line_end is not None:
        plt.plot([length_line_start[0], length_line_end[0]], [length_line_start[1], length_line_end[1]], 'r-', linewidth=2)
    if height_line_start is not None and height_line_end is not None:
        plt.plot([height_line_start[0], height_line_end[0]], [height_line_start[1], height_line_end[1]], 'g-', linewidth=2)
    if length_line_start is not None and length_line_end is not None:
        plt.scatter([length_line_start[0], length_line_end[0]], [length_line_start[1], length_line_end[1]], color='red', s=50, zorder=5)
    if height_line_start is not None and height_line_end is not None:
        plt.scatter([height_line_start[0], height_line_end[0]], [height_line_start[1], height_line_end[1]], color='green', s=50, zorder=5)
    title = f"Length: {length:.2f} px"
    if length_mm is not None:
        title += f" ({length_mm:.2f} mm)"
    title += f", Height: {height:.2f} px"
    if height_mm is not None:
        title += f" ({height_mm:.2f} mm)"
    plt.title(title)
    plt.axis('off')
    img_output_dir = os.path.join(output_dir, f"image_{cleaned_image_basename}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    save_path = os.path.join(img_output_dir, f"ann_{cleaned_image_basename}_{ann_idx+1}_{category_name}_length_height.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Length and Height visualization saved to {save_path}")


def visualize_skeleton_visual(image, skeleton, output_dir, img_id, ann_idx, category_name, image_basename):
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(skeleton, cmap='gray', alpha=0.5)
    plt.title(f"Skeleton for Image {image_basename} {img_id}, Annotation {ann_idx+1}, Category: {category_name}")
    plt.axis('off')
    img_output_dir = os.path.join(output_dir, f"image_{image_basename}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    save_path = os.path.join(img_output_dir, f"ann_{image_basename}_{ann_idx+1}_{category_name}_skeleton.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Skeleton visualization saved to: {save_path}")


def visualize_scale_bar(image, scale_bar_contour, scale_text, output_dir,
                        img_id, image_basename, text_bbox=None, method=""):
    vis = image.copy()
    if scale_bar_contour is not None and len(scale_bar_contour) >= 1:
        x, y, w, h = cv2.boundingRect(scale_bar_contour)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{scale_text}"
        if method:
            label += f" [{method}]"
        cv2.putText(vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if text_bbox is not None:
        tx, ty, tw, th = text_bbox
        cv2.rectangle(vis, (tx, ty), (tx + tw, ty + th), (0, 200, 255), 2)
    img_output_dir = os.path.join(output_dir, f"image_{clean_filename(image_basename)}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    save_path = os.path.join(img_output_dir, "scale_bar_visualization.png")
    cv2.imwrite(save_path, vis)
    logging.info(f"Scale bar visualization saved to: {save_path}")


#def save_metrics_to_file(output_dir, filename, metrics, img_id):
#    img_output_dir = os.path.join(output_dir, f"image_{metrics['image_filename']}_{img_id}")
#    os.makedirs(img_output_dir, exist_ok=True)
#    file_path = os.path.join(img_output_dir, filename)
#    with open(file_path, 'w') as f:
#        for key, value in metrics.items():
#            f.write(f"{key}: {value}\n")
#    logging.info(f"Metrics saved to {file_path}")


def save_all_metrics_to_csv(output_file, all_metrics):
    # If empty or None, we skip
    if not output_file or not output_file.strip():
        logging.warning("No valid --output_file provided; skipping combined CSV.")
        return

    if os.path.isdir(output_file):
        output_file = os.path.join(output_file, "combined_metrics.csv")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not all_metrics:
        logging.warning("No metrics to save.")
        return
    fieldnames = set()
    for metric in all_metrics:
        fieldnames.update(metric.keys())
    fieldnames = sorted(list(fieldnames))

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in all_metrics:
            # Convert any NumPy arrays to lists
            metrics_serializable = {
                k: (v if not isinstance(v, (np.ndarray, np.generic)) else v.tolist())
                for k, v in metrics.items()
            }
            writer.writerow(metrics_serializable)
    logging.info(f"All metrics saved to CSV file: {output_file}")


def save_metrics_to_jsonl(jsonl_output_path, all_metrics):
    if not jsonl_output_path or not jsonl_output_path.strip():
        logging.warning("No valid --jsonl_output provided; skipping JSONL.")
        return

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return [convert_types(v) for v in obj]
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
            serializable_metric = convert_types(metric)
            jsonl_file.write(json.dumps(serializable_metric) + '\n')
    logging.info(f"All metrics saved to JSONL file at {jsonl_output_path}")


def parse_grouping_file(grouping_file_path):
    if not os.path.exists(grouping_file_path):
        logging.warning(f"Grouping file not found at {grouping_file_path}. Proceeding without grouping.")
        return {}
    
    _, file_extension = os.path.splitext(grouping_file_path)
    group_mapping = {}
    try:
        if file_extension.lower() in ['.csv', '.tsv', '.txt']:
            try:
                if file_extension.lower() == '.csv':
                    df = pd.read_csv(grouping_file_path)
                else:
                    df = pd.read_csv(grouping_file_path, delimiter='\t')
                if list(df.columns)[:2] == ['specimen_name', 'group']:
                    group_mapping = pd.Series(df.group.values, index=df.specimen_name).to_dict()
                else:
                    group_mapping = pd.Series(df.iloc[:,1].values, index=df.iloc[:,0]).to_dict()
            except pd.errors.ParserError:
                if file_extension.lower() == '.csv':
                    df = pd.read_csv(grouping_file_path, header=None)
                else:
                    df = pd.read_csv(grouping_file_path, delimiter='\t', header=None)
                group_mapping = pd.Series(df.iloc[:,1].values, index=df.iloc[:,0]).to_dict()
        elif file_extension.lower() == '.json':
            with open(grouping_file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        specimen = entry.get('specimen_name') or entry.get('specimen') or entry.get('name')
                        group = entry.get('group') or entry.get('species') or entry.get('cluster')
                        if specimen and group:
                            group_mapping[specimen] = group
                elif isinstance(data, dict):
                    group_mapping = data
        else:
            logging.warning(f"Unsupported file format: {file_extension}. Proceeding without grouping.")
            return {}
        logging.debug(f"Parsed grouping file with {len(group_mapping)} entries.")
    except Exception as e:
        logging.error(f"Error parsing grouping file: {e}. Proceeding without grouping.")
        return {}
    return group_mapping


def count_instances(all_metrics, group_mapping, category_name):
    counts_per_specimen = defaultdict(int)
    counts_per_group = defaultdict(list)
    for metric in all_metrics:
        specimen_name = os.path.splitext(metric['image_filename'])[0]
        if metric['category_name'].lower() != category_name.lower():
            continue
        counts_per_specimen[specimen_name] += 1
        group_name = group_mapping.get(specimen_name, 'Unassigned')
        counts_per_group[group_name].append(1)
    if group_mapping:
        unique_groups = set(group_mapping.values())
        for group in unique_groups:
            if group not in counts_per_group:
                counts_per_group[group] = [0]
    logging.debug("Counts per specimen:")
    for specimen, count in counts_per_specimen.items():
        logging.debug(f"  {specimen}: {count}")
    logging.debug("\nCounts per group:")
    for group, counts in counts_per_group.items():
        logging.debug(f"  {group}: {counts}")
    return counts_per_specimen, counts_per_group


def compute_statistics(counts_per_group):
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
    logging.debug("\nStatistics per group:")
    for group, stats in stats_per_group.items():
        logging.debug(f"  {group}: {stats}")
    return stats_per_group


def save_group_measurement_statistics(stats_per_group, output_dir, category_name):
    csv_path = os.path.join(output_dir, f"{category_name}_group_measurement_statistics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['group_name']
        measurement_fields = [
            'length_mm', 'height_mm', 'area_mm2', 'perimeter_mm',
            'equivalent_diameter_mm', 'major_axis_length_mm', 'minor_axis_length_mm'
        ]
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
    logging.debug(f"Group measurement statistics saved to CSV: {csv_path}")
    json_path = os.path.join(output_dir, f"{category_name}_group_measurement_statistics.json")
    with open(json_path, 'w') as jsonfile:
        json.dump(stats_per_group, jsonfile, indent=4)
    logging.debug(f"Group measurement statistics saved to JSON: {json_path}")


def create_thumbnail(image, mask, thumbnail_size=(80, 80)):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    elif len(mask.shape) > 2:
        mask = mask[:, :, 0]
    binary_mask = (mask > 0).astype(np.uint8)
    foreground = np.zeros_like(image)
    foreground[binary_mask == 1] = image[binary_mask == 1]
    if image.ndim == 3 and image.shape[2] == 3:
        foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    else:
        foreground_rgb = foreground
    pil_img = Image.fromarray(foreground_rgb).convert("RGBA")
    pil_mask = Image.fromarray((binary_mask * 255).astype(np.uint8)).convert("L")
    pil_img.putalpha(pil_mask)
    pil_img.thumbnail(thumbnail_size, Image.LANCZOS)
    new_img = Image.new('RGBA', thumbnail_size, (255, 255, 255, 0))
    offset = ((thumbnail_size[0] - pil_img.size[0]) // 2,
              (thumbnail_size[1] - pil_img.size[1]) // 2)
    new_img.paste(pil_img, offset)
    foreground_rgba = np.array(new_img)
    return foreground_rgba


def plot_pca_umap(principal_components, images, masks, output_dir, category_name, method='PCA', thumbnail_size=(80, 80)):
    logging.debug(f"Plotting {method} with Thumbnails for category '{category_name}'...")
    if principal_components.size == 0:
        logging.error(f"{method} data is empty.")
        return
    if len(principal_components) != len(images) or len(images) != len(masks):
        logging.error("The lengths of principal_components, images, and masks do not match.")
        return
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title(f'{method} of {category_name} Measurement Metrics with Thumbnails')
    ax.set_xlabel(f'{method} Component 1')
    ax.set_ylabel(f'{method} Component 2')
    ax.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.0)
    x_min, x_max = principal_components[:, 0].min(), principal_components[:, 0].max()
    y_min, y_max = principal_components[:, 1].min(), principal_components[:, 1].max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    for i, (pc, mask, img) in enumerate(zip(principal_components, masks, images)):
        thumbnail = create_thumbnail(img, mask, thumbnail_size)
        if thumbnail is None:
            logging.warning(f"Skipping thumbnail {i+1} due to creation failure.")
            continue
        imagebox = OffsetImage(thumbnail, zoom=0.75)
        ab = AnnotationBbox(imagebox, (pc[0], pc[1]), frameon=False, pad=0.0)
        ax.add_artist(ab)
    plt.grid(True)
    cleaned = clean_filename(category_name)
    plot_filename = f'{method.lower()}_{cleaned}_with_thumbnails.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.debug(f"{method} plot with thumbnails saved to: {plot_path}")


from scipy.cluster.hierarchy import dendrogram, linkage

def plot_hierarchical_dendrogram(X, output_dir, category_name):
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
    logging.debug(f"Hierarchical dendrogram saved to: {plot_path}")


# --- Re-added perform_umap function ---
def perform_umap(X, n_components=2, random_state=42):
    umap_model = umap.UMAP(n_components=n_components, random_state=random_state)
    umap_results = umap_model.fit_transform(X)
    logging.debug(f"UMAP completed with {n_components} components.")
    return umap_results, umap_model


def perform_dbscan(X, output_dir, category_name, principal_components, images, masks):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    logging.debug(f"DBSCAN found {num_clusters} clusters")
    plt.figure(figsize=(16, 12))
    unique_labels = set(dbscan_labels)
    colors = sns.color_palette(None, len(unique_labels))
    for label, color in zip(unique_labels, colors):
        if label == -1:
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
    for i, (pc, img, mask, label) in enumerate(zip(principal_components, images, masks, dbscan_labels)):
        if label == -1:
            continue
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
    logging.debug(f"DBSCAN plot saved to: {plot_path}")
    return dbscan_labels


def perform_hierarchical_clustering(X, output_dir, category_name, principal_components, images, masks, method='ward', metric='euclidean', num_clusters=3):
    Z = linkage(X, method=method, metric=metric)
    labels = AgglomerativeClustering(n_clusters=num_clusters, affinity=metric, linkage=method).fit_predict(X)
    logging.debug(f"Hierarchical clustering formed {num_clusters} clusters using method '{method}' and metric '{metric}'.")
    plt.figure(figsize=(16,12))
    unique = np.unique(labels)
    colors = sns.color_palette(None, len(unique))
    for lab, color in zip(unique, colors):
        lab_name = f'Cluster {lab}'
        mask_lab = (labels == lab)
        xy = principal_components[mask_lab]
        plt.scatter(xy[:,0], xy[:,1], c=[color], label=lab_name, alpha=0.6, edgecolors='w', linewidths=0.5)
    plt.title(f'Hierarchical Clustering of {category_name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    for i, (pc, img, mask, lab) in enumerate(zip(principal_components, images, masks, labels)):
        thumbnail = create_thumbnail(img, mask)
        if thumbnail is None:
            continue
        imagebox = OffsetImage(thumbnail, zoom=0.5)
        ab = AnnotationBbox(imagebox, (pc[0], pc[1]), frameon=False, pad=0.0)
        plt.gca().add_artist(ab)
    plot_filename = f'hierarchical_clustering_{clean_filename(category_name)}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.debug(f"Hierarchical clustering plot saved to: {plot_path}")
    return labels


def flatten_output_directory(output_dir):
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            if root == output_dir:
                continue
            subdir_name = os.path.basename(root)
            new_filename = f"{subdir_name}_{file}"
            destination = os.path.join(output_dir, new_filename)
            shutil.move(file_path, destination)
            logging.debug(f"Moved '{file_path}' to '{destination}'")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
                logging.debug(f"Removed empty directory: {dir_path}")
            except OSError:
                logging.debug(f"Directory not empty, could not remove: {dir_path}")
    logging.debug("All files have been moved to the top-level output directory.")

# ---------------------------------------------------------------------
# (1) Add a list of relevant segmentation metrics to store in a numeric vector
# ---------------------------------------------------------------------
SEG_FEATURE_KEYS = [
    "length_pixels",
    "height_pixels",
    "area_pixels",
    "perimeter_pixels",
    "aspect_ratio",
    "extent",
    "solidity",
    "major_axis_length_pixels",
    "minor_axis_length_pixels"
]

### NEW: mm-based fields
SEG_FEATURE_KEYS_MM = [
    "length_mm",
    "height_mm",
    "area_mm2",
    "perimeter_mm",
    "aspect_ratio_mm",
    "major_axis_length_mm",
    "minor_axis_length_mm",
    # You can add 'equivalent_diameter_mm' or others if desired
]


###############################################################################
# Add these two functions somewhere above the 'def main():' line (for example,
# right after the SEG_FEATURE_KEYS_MM definition).
###############################################################################

def get_segmentation_headers_px():
    """
    Return the list of header names for the pixel-based segmentation metrics.
    This corresponds to the order in SEG_FEATURE_KEYS.
    """
    return [
        "length_pixels",
        "height_pixels",
        "area_pixels",
        "perimeter_pixels",
        "aspect_ratio",
        "extent",
        "solidity",
        "major_axis_length_pixels",
        "minor_axis_length_pixels"
    ]

def get_segmentation_headers_mm():
    """
    Return the list of header names for the mm-based segmentation metrics.
    This corresponds to the order in SEG_FEATURE_KEYS_MM.
    """
    return [
        "length_mm",
        "height_mm",
        "area_mm2",
        "perimeter_mm",
        "major_axis_length_mm",
        "minor_axis_length_mm"
    ]


def extract_seg_features_as_array(metrics_dict):
    """
    Convert a segmentation metrics dict into a numeric 1D array
    for the fields in SEG_FEATURE_KEYS, in a fixed order.
    If any field is missing or None, we store np.nan.
    """
    arr = []
    for key in SEG_FEATURE_KEYS:
        val = metrics_dict.get(key, None)
        if val is None:
            arr.append(np.nan)
        else:
            arr.append(val)
    return np.array(arr, dtype=np.float64)

### NEW: function to extract mm-based segmentation fields
def extract_seg_features_as_array_mm(metrics_dict):
    """
    Convert a segmentation metrics dict into a numeric 1D array
    of the mm-based fields in SEG_FEATURE_KEYS_MM.
    Missing fields => np.nan.
    """
    arr = []
    for key in SEG_FEATURE_KEYS_MM:
        val = metrics_dict.get(key, None)
        if val is None:
            arr.append(np.nan)
        else:
            arr.append(val)
    return np.array(arr, dtype=np.float64)

# ---------------------------------------------------------------------
# (2) Save segmentation metrics .npy for an annotation
# ---------------------------------------------------------------------
def save_segmentation_metrics_npy_individual(
    output_dir,
    seg_array,
    metrics_dict,
    suffix="px"  ### NEW optional suffix to differentiate px vs mm
):
    """
    Saves the segmentation metrics array to a .npy file in the same folder 
    structure used for the .png output.

    The annotation key naming: 
      "ann_{cleaned_basename}_{ann_idx}_{category}_segmentation_{suffix}.npy"
    """
    img_id    = metrics_dict["image_id"]
    ann_idx   = metrics_dict["annotation_index"]
    category  = metrics_dict["category_name"]
    basename  = metrics_dict["image_filename"]  # e.g. "myfile.jpg"
    
    # For folder naming
    cleaned_basename = re.sub(r'[^\w\-_.]', '_', basename)
    img_output_dir   = os.path.join(output_dir, f"image_{cleaned_basename}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)

    file_prefix = f"ann_{cleaned_basename}_{ann_idx}_{category}_segmentation_{suffix}"
    save_path   = os.path.join(img_output_dir, file_prefix + ".npy")
    
    np.save(save_path, seg_array)
    logging.info(f"Saved segmentation {suffix} metrics array to {save_path}")

# 3) NEW: function to save combined TPS landmarks with scale
# ---------------------------------------------------------------------
def save_tps_file(all_keypoints_metrics, combined_scales, output_dir, filename='combined_landmarks.tps'):
    tps_path = os.path.join(output_dir, filename)
    with open(tps_path, 'w') as f:
        for kp in all_keypoints_metrics:
            coords = kp.get('keypoint_coords', [])
            num = len(coords)
            f.write(f"LM={num}\n")
            f.write(f"IMAGE={kp['image_filename']}\n")
            # include scale if available
            scale = combined_scales.get(kp['image_filename'], '')
            if scale:
                f.write(f"SCALE={scale}\n")
            for x, y in coords:
                f.write(f"{x:.2f} {y:.2f}\n")
            f.write("\n")
    logging.info(f"TPS landmarks saved to {tps_path}")

def save_metrics_to_file(output_dir, filename, metrics, img_id):
    img_output_dir = os.path.join(output_dir, f"image_{metrics['image_filename']}_{img_id}")
    os.makedirs(img_output_dir, exist_ok=True)
    file_path = os.path.join(img_output_dir, filename)
    # NEW: switch to append mode and add timestamp
    mode = 'a' if os.path.exists(file_path) else 'w'
    with open(file_path, mode) as f:
        f.write(f"\n--- Run: {datetime.datetime.now().isoformat()} ---\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    logging.info(f"Metrics saved to {file_path} (mode={mode})")


# NEW: helper to write segmentation TPS
def save_seg_tps_file(seg_metrics, combined_scales, output_dir,
                      filename="combined_segmentation_landmarks.tps"):
    tps_path = os.path.join(output_dir, filename)
    with open(tps_path, "w") as f:
        for seg in seg_metrics:
            pts = seg["segmentation_coords"]
            f.write(f"LM={len(pts)}\n")
            f.write(f"IMAGE={seg['image_filename']}\n")
            sc = combined_scales.get(seg["image_filename"], "")
            if sc:
                f.write(f"SCALE={sc}\n")
            f.write("LINE\n")  # first point + rest as a line/polyline
            for x, y in pts:
                f.write(f"{x:.2f} {y:.2f}\n")
            f.write("\n")
    logging.info(f"Segmentation TPS saved to {tps_path}")


def save_augmented_coco_json(input_json_path, output_dir, annotations_dict,
                             images_info, categories, per_ann_measurements,
                             combined_scales, all_inter_mask_distances):
    """Write a proper COCO JSON with measurement fields injected into annotations."""
    with open(input_json_path, 'r') as f:
        coco = json.load(f)

    ann_lookup = {a['id']: a for a in coco.get('annotations', [])}
    imd_by_image = defaultdict(list)
    for d in all_inter_mask_distances:
        imd_by_image[d['image_id']].append(d)

    for ann in coco.get('annotations', []):
        aid = ann['id']
        iid = ann['image_id']
        fname = images_info.get(iid, {}).get('file_name', '')
        basename = os.path.basename(fname)

        ann['scale_pixels_per_mm'] = combined_scales.get(basename)

        meas = per_ann_measurements.get((iid, aid))
        if meas:
            for key in ['length_pixels', 'height_pixels', 'length_mm', 'height_mm',
                        'area_pixels', 'area_mm2', 'perimeter_pixels', 'perimeter_mm',
                        'aspect_ratio', 'solidity', 'extent',
                        'major_axis_length_pixels', 'minor_axis_length_pixels',
                        'major_axis_length_mm', 'minor_axis_length_mm',
                        'equivalent_diameter_pixels', 'equivalent_diameter_mm',
                        'orientation_degrees', 'length_to_height_ratio']:
                val = meas.get(key)
                if val is not None:
                    ann[key] = float(val) if isinstance(val, (np.floating, float)) else val

    coco['inter_mask_distances'] = []
    for d in all_inter_mask_distances:
        entry = {k: v for k, v in d.items()
                 if not isinstance(v, (np.ndarray, np.generic))}
        for k, v in entry.items():
            if isinstance(v, (np.floating, np.integer)):
                entry[k] = float(v) if isinstance(v, np.floating) else int(v)
        coco['inter_mask_distances'].append(entry)

    out_path = os.path.join(output_dir, 'measurements_coco.json')
    with open(out_path, 'w') as f:
        json.dump(coco, f, indent=2, default=str)
    logging.info(f"Augmented COCO JSON saved to {out_path}")


def save_jsonld_knowledge_graph(output_dir, annotations_dict, images_info,
                                categories, per_ann_measurements,
                                combined_scales, all_inter_mask_distances):
    """Write a JSON-LD file with schema.org + Darwin Core vocabulary."""
    specimens = []
    for img_id, anns in annotations_dict.items():
        info = images_info.get(img_id, {})
        fname = info.get('file_name', '')
        basename = os.path.basename(fname)
        scale = combined_scales.get(basename)

        regions = []
        for ann in anns:
            aid = ann.get('id', 0)
            cat_name = categories.get(ann['category_id'], 'Unknown')
            meas = per_ann_measurements.get((img_id, aid), {})

            region = {
                "@type": "DefinedTerm",
                "name": cat_name,
                "identifier": f"annotation-{aid}",
                "additionalProperty": []
            }
            prop_map = {
                'length_pixels': ('length', 'px'),
                'height_pixels': ('width', 'px'),
                'length_mm': ('length', 'mm'),
                'height_mm': ('width', 'mm'),
                'area_pixels': ('area', 'px^2'),
                'area_mm2': ('area', 'mm^2'),
                'perimeter_pixels': ('perimeter', 'px'),
                'perimeter_mm': ('perimeter', 'mm'),
                'aspect_ratio': ('aspectRatio', None),
                'solidity': ('solidity', None),
                'extent': ('extent', None),
                'orientation_degrees': ('orientation', 'degrees'),
            }
            for mkey, (pname, unit) in prop_map.items():
                val = meas.get(mkey)
                if val is not None:
                    prop = {
                        "@type": "PropertyValue",
                        "name": pname,
                        "value": round(float(val), 4) if isinstance(val, (float, np.floating)) else val,
                    }
                    if unit:
                        prop["unitCode"] = unit
                    region["additionalProperty"].append(prop)

            regions.append(region)

        img_dists = [d for d in all_inter_mask_distances if d['image_id'] == img_id]
        distance_entries = []
        for d in img_dists:
            entry = {
                "@type": "PropertyValue",
                "name": f"distance_{d['category_a']}_to_{d['category_b']}",
                "description": f"Pairwise distance between {d['category_a']} (ann {d['ann_idx_a']}) and {d['category_b']} (ann {d['ann_idx_b']})",
                "additionalProperty": [
                    {"@type": "PropertyValue", "name": "centroidDistance",
                     "value": round(d['centroid_distance_px'], 2), "unitCode": "px"},
                    {"@type": "PropertyValue", "name": "minBoundaryDistance",
                     "value": round(d['min_boundary_distance_px'], 2), "unitCode": "px"},
                    {"@type": "PropertyValue", "name": "overlap",
                     "value": d['overlap']},
                ]
            }
            if 'centroid_distance_mm' in d:
                entry["additionalProperty"].extend([
                    {"@type": "PropertyValue", "name": "centroidDistance",
                     "value": round(d['centroid_distance_mm'], 4), "unitCode": "mm"},
                    {"@type": "PropertyValue", "name": "minBoundaryDistance",
                     "value": round(d['min_boundary_distance_mm'], 4), "unitCode": "mm"},
                ])
            distance_entries.append(entry)

        specimen = {
            "@type": "Specimen",
            "name": basename,
            "identifier": f"image-{img_id}",
            "image": {"@type": "ImageObject", "name": fname},
            "hasPart": regions,
        }
        if scale:
            specimen["additionalProperty"] = [{
                "@type": "PropertyValue",
                "name": "scalePixelsPerMm",
                "value": round(float(scale), 2),
                "unitCode": "px/mm"
            }]
        if distance_entries:
            specimen.setdefault("additionalProperty", []).extend(distance_entries)

        specimens.append(specimen)

    jsonld = {
        "@context": {
            "@vocab": "https://schema.org/",
            "dwc": "http://rs.tdwg.org/dwc/terms/",
            "openbiodiv": "http://openbiodiv.net/ontology#",
            "Specimen": "openbiodiv:Specimen",
            "DefinedTerm": "schema:DefinedTerm",
            "PropertyValue": "schema:PropertyValue",
            "ImageObject": "schema:ImageObject",
        },
        "@type": "Dataset",
        "name": "Descriptron Morphometric Measurements",
        "dateCreated": datetime.datetime.now().isoformat(),
        "description": "Instance segmentation measurements with inter-mask distances",
        "hasPart": specimens
    }

    out_path = os.path.join(output_dir, 'measurements_kg.jsonld')
    with open(out_path, 'w') as f:
        json.dump(jsonld, f, indent=2, default=str)
    logging.info(f"JSON-LD knowledge graph saved to {out_path}")


###########################
# Main Measurement Function
###########################
def main():
    try:
        logging.debug("Starting main function")
        args = parse_args()
        logging.debug("Arguments parsed successfully.")
        
        os.makedirs(args.output_dir, exist_ok=True)
        logging.debug(f"Output directory is set to '{args.output_dir}'")
        
        annotations_dict, images_info, categories = load_annotations(
            args.json, args.image_id, args.category_name
        )
        logging.debug(f"Loaded annotations for {len(annotations_dict)} images.")
        if not annotations_dict:
            logging.warning("No annotations found in the JSON file.")
            return

        if args.max_images is not None:
            keys = list(annotations_dict.keys())[:args.max_images]
            annotations_dict = {k: annotations_dict[k] for k in keys}
            logging.info(f"Limiting to {len(annotations_dict)} images (--max_images={args.max_images})")

        # NEW: collect per-image scales
        combined_scales = {}
        # NEW: collect segmentation coords for TPS
        all_segmentation_metrics = []
        
    except Exception as e:
        logging.error(f"An error occurred in main(): {e}")
        traceback.print_exc()
        return
    
    try:
        if args.grouping_file:
            group_mapping = parse_grouping_file(args.grouping_file)
            logging.debug(f"Group mapping contains {len(group_mapping)} entries.")
        else:
            group_mapping = {}
            logging.debug("No grouping file provided. Proceeding without grouping information.")
        
        reader = ensure_easyocr_models_present(languages=['en'])
        logging.debug("EasyOCR reader initialized.")
        
        all_metrics = []
        images_list = []
        masks_list  = []
        all_inter_mask_distances = []
        per_ann_measurements = {}

        # For keypoints
        all_keypoints_metrics          = []
        keypoint_images_list           = []
        combined_distance_matrices_px  = {}
        combined_distance_matrices_mm  = {}

        # ---------------------------------------------------------------------
        # (3) Create a new dictionary to store segmentation vectors for all specs
        # ---------------------------------------------------------------------
        combined_segmentation_arrays = {}
        
        for img_id, annotations in annotations_dict.items():
            logging.debug(f"Processing image ID: {img_id}")
            image_info = images_info.get(img_id)
            if image_info is None:
                logging.warning(f"No image found with ID {img_id} in the JSON annotations.")
                continue
            
            image_filename = image_info['file_name']
            image_basename = os.path.basename(os.path.normpath(image_filename))
            cleaned_image_basename = clean_filename(image_basename)
            image_path = os.path.join(args.image_dir, image_filename)
            
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Image not found: {image_path}")
                continue
            
            height, width = image.shape[:2]
            logging.debug(f"Image dimensions: {width}x{height}")
            
            # --- Automatic scale detection (PER-IMAGE) ---
            scale_contour = None
            scale_text_bbox = None
            scale_text_str = None
            scale_method = ""
            if args.mm_scale is not None:
                px_per_mm = args.mm_scale
            else:
                px_per_mm, scale_text_str, scale_contour, scale_text_bbox = \
                    detect_image_scale_florence2(image, args)
                if px_per_mm is not None:
                    scale_method = "Florence-2"
                else:
                    px_per_mm = detect_image_scale(image, reader, args)
                    if px_per_mm is not None:
                        scale_method = "EasyOCR"

            combined_scales[image_basename] = px_per_mm

            if px_per_mm is not None:
                visualize_scale_bar(
                    image, scale_contour, scale_text_str or f"{px_per_mm:.1f} px/mm",
                    args.output_dir, img_id, image_basename,
                    text_bbox=scale_text_bbox, method=scale_method)

            if not annotations:
                logging.warning(f"Image ID ('{image_filename}') {img_id}: No annotations found.")
                continue

            per_image_masks = []

            for ann_idx, ann in enumerate(annotations):
                logging.debug(f"Processing annotation {ann_idx+1} for image ID {img_id}")
                category_name = categories.get(ann['category_id'], 'Unknown')
                
                metrics = {
                    "image_id": img_id,
                    "annotation_index": ann_idx + 1,
                    "category_id": ann['category_id'],
                    "category_name": category_name,
                    "image_filename": image_basename
                }
                
                # Process Keypoints
                if 'keypoints' in ann and len(ann['keypoints']) >= 3:
                    logging.debug(f"Processing keypoints for annotation {ann_idx+1}")
                    keypoints_metrics = process_keypoints(
                        ann, img_id, ann_idx, category_name, mm_scale=px_per_mm #args.mm_scale
                    )
                    if keypoints_metrics:
                        keypoints_metrics["image_filename"] = image_basename
                        all_metrics.append(keypoints_metrics)
                        all_keypoints_metrics.append(keypoints_metrics)
                        images_list.append(image)
                        keypoint_images_list.append(image)
                        
                        # Visualization: overlay & distances
                        raw = ann['keypoints']
#                        coords = [(raw[i], raw[i+1]) 
#                                  for i in range(0, len(raw), 3) if raw[i+2] > 0]
                        coords = [
                            (raw[i], raw[i+1]) 
                            for i in range(0, len(raw), 3) if raw[i+2] > 0
                        ]
                        # NEW: store coords for TPS output
                        keypoints_metrics["keypoint_coords"] = coords
                        
                        distance_matrix_px = np.array(keypoints_metrics["keypoints_distance_matrix"])
                        visualize_keypoints_distances(
                            image=image,
                            keypoint_coords=coords,
                            distance_matrix=distance_matrix_px,
                            output_dir=args.output_dir,
                            img_id=img_id,
                            ann_idx=ann_idx,
                            category_name=category_name,
                            image_basename=image_basename,
                            mm_scale=px_per_mm  #args.mm_scale
                        )

                        # record combined NPZ
                        annotation_key = f"{cleaned_image_basename}_{ann_idx+1}_{category_name}_keypoints"
                        combined_distance_matrices_px[annotation_key] = distance_matrix_px
                        if "keypoints_distance_matrix_mm" in keypoints_metrics:
                            dm_mm = np.array(keypoints_metrics["keypoints_distance_matrix_mm"])
                            combined_distance_matrices_mm[annotation_key] = dm_mm
                        
                        # save individual NPZ
                        if args.save_results:
                            save_distance_matrices_npy_individual(
                                args.output_dir,
                                distance_matrix_px,
                                keypoints_metrics.get("keypoints_distance_matrix_mm"),
                                keypoints_metrics,
                                image_basename
                            )
                
                # Process Segmentation
                if args.method in ['skeleton', 'pca']:
                    logging.debug(f"Processing segmentation using method: {args.method}")
                    mask = create_mask_from_annotation(
                        ann, height, width, img_id, ann_idx, category_name
                    )
                    if mask is None:
                        continue
                    binary_mask = (mask > 0).astype(np.uint8) * 255
                    
                    if args.method == 'skeleton':
                        logging.debug("Applying skeletonization")
                        pre = preprocess_mask_for_skeleton(binary_mask)
                        try:
                            sk_z = cv2.ximgproc.thinning(pre, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
                            sk_g = cv2.ximgproc.thinning(pre, thinningType=cv2.ximgproc.THINNING_GUOHALL)
                        except AttributeError as e:
                            logging.error(f"Skeletonization error: {e}")
                            continue
                        skeleton = sk_z if np.count_nonzero(sk_z) > np.count_nonzero(sk_g) else sk_g
                        
                        metrics.update({
                            "method": "Skeletonization",
                            "length_pixels": measure_length_skeleton(skeleton),
                            "average_width_pixels": measure_width_skeleton(binary_mask, skeleton)
                        })
                        if px_per_mm: #args.mm_scale:
                            metrics["length_mm"] = metrics["length_pixels"] / px_per_mm
                            metrics["width_mm"]  = metrics["average_width_pixels"] / px_per_mm

                        # NEW: capture skeleton coords
                        sk_pts = np.column_stack(np.where(skeleton > 0))[:, ::-1].tolist()
                        metrics["segmentation_coords"] = sk_pts
                        all_segmentation_metrics.append(metrics)
                        
                        all_metrics.append(metrics)
                        masks_list.append(binary_mask)
                        per_image_masks.append({
                            'mask': binary_mask, 'category_name': category_name,
                            'annotation_index': ann_idx + 1, 'contour': None,
                            'annotation_id': ann.get('id', ann_idx + 1),
                        })
                        per_ann_measurements[(img_id, ann.get('id', ann_idx + 1))] = metrics

                        visualize_skeleton_visual(
                            image=image, skeleton=skeleton, output_dir=args.output_dir,
                            img_id=img_id, ann_idx=ann_idx,
                            category_name=category_name,
                            image_basename=cleaned_image_basename
                        )
                    
                    else:  # PCA
                        logging.debug("Applying PCA-based measurements")
                        filled = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE,
                                                  np.ones((5, 5), np.uint8))
                        contours, _ = cv2.findContours(
                            filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if not contours:
                            continue
                        contour = max(contours, key=cv2.contourArea)
                        metrics_pca = measure_metrics_pca(
                            contour, filled,
                            pixels_per_mm=px_per_mm, #args.mm_scale,
                            img_id=img_id, ann_idx=ann_idx,
                            category_name=category_name,
                            category_id=ann['category_id'],
                            image_basename=cleaned_image_basename
                        )
                        # NEW: capture contour coords
                        contour_pts = contour.squeeze().tolist()
                        metrics_pca["segmentation_coords"] = contour_pts
                        all_segmentation_metrics.append(metrics_pca)
                        
                        all_metrics.append(metrics_pca)
                        masks_list.append(filled)
                        per_image_masks.append({
                            'mask': filled, 'category_name': category_name,
                            'annotation_index': ann_idx + 1, 'contour': contour,
                            'annotation_id': ann.get('id', ann_idx + 1),
                        })
                        per_ann_measurements[(img_id, ann.get('id', ann_idx + 1))] = metrics_pca

                        visualize_length_height(
                            image=image,
                            contour=contour,
                            length=metrics_pca['length_pixels'],
                            height=metrics_pca['height_pixels'],
                            length_mm=metrics_pca.get('length_mm'),
                            height_mm=metrics_pca.get('height_mm'),
                            orientation_degrees=metrics_pca.get('orientation_degrees'),
                            output_dir=args.output_dir,
                            img_id=img_id,
                            ann_idx=ann_idx,
                            category_name=category_name,
                            cleaned_image_basename=cleaned_image_basename,
                            length_line_start=metrics_pca['length_line_start'],
                            length_line_end=metrics_pca['length_line_end'],
                            height_line_start=metrics_pca['height_line_start'],
                            height_line_end=metrics_pca['height_line_end']
                        )

                        # build numeric segmentation arrays (existing)
                        seg_array_px = extract_seg_features_as_array(metrics_pca)
                        seg_key_px   = f"{cleaned_image_basename}_{ann_idx+1}_{category_name}_seg"
                        combined_segmentation_arrays[seg_key_px] = seg_array_px
                        if metrics_pca.get("length_mm") is not None:
                            seg_array_mm = extract_seg_features_as_array_mm(metrics_pca)
                            seg_key_mm   = f"{cleaned_image_basename}_{ann_idx+1}_{category_name}_seg_mm"
                            combined_segmentation_arrays[seg_key_mm] = seg_array_mm
                        
                        if args.save_results:
                            save_segmentation_metrics_npy_individual(
                                args.output_dir, seg_array_px, metrics_pca, suffix="px"
                            )
                            if metrics_pca.get("length_mm") is not None:
                                save_segmentation_metrics_npy_individual(
                                    args.output_dir, seg_array_mm, metrics_pca, suffix="mm"
                                )
                
                if args.save_results:
                    save_metrics_to_file(
                        args.output_dir, f"metrics_{img_id}.txt", metrics, img_id
                    )

            # --- Inter-mask pairwise distances for this image ---
            if len(per_image_masks) >= 2:
                inter_dists = compute_inter_mask_distances(per_image_masks, px_per_mm)
                for d in inter_dists:
                    d['image_id'] = img_id
                    d['image_filename'] = image_basename
                all_inter_mask_distances.extend(inter_dists)
                logging.info(f"Image {image_basename}: {len(inter_dists)} inter-mask pairs computed")

        # Save combined NPZs (existing) …
        if combined_distance_matrices_px:
            np.savez(
                os.path.join(args.output_dir, "combined_keypoints_distance_px.npz"),
                **combined_distance_matrices_px
            )
        if combined_distance_matrices_mm:
            np.savez(
                os.path.join(args.output_dir, "combined_keypoints_distance_mm.npz"),
                **combined_distance_matrices_mm
            )
        if combined_segmentation_arrays:
            headers_px = get_segmentation_headers_px()
            headers_mm = get_segmentation_headers_mm()
            combined_segmentation_arrays["HEADERS_PX"] = np.array(headers_px, object)
            combined_segmentation_arrays["HEADERS_MM"] = np.array(headers_mm, object)
            np.savez(
                os.path.join(args.output_dir, "combined_segmentation_metrics.npz"),
                **combined_segmentation_arrays
            )
        
        # NEW: write combined scales CSV
        scale_csv = os.path.join(args.output_dir, 'combined_scales.csv')
        with open(scale_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_filename', 'pixels_per_unit'])
            for fn, sc in combined_scales.items():
                writer.writerow([fn, sc])
        logging.info(f"Combined scales saved to {scale_csv}")
        
        # NEW: TPS for keypoints
        if all_keypoints_metrics:
            save_tps_file(all_keypoints_metrics, combined_scales, args.output_dir)
        else:
            logging.info("No keypoints to write TPS.")
        
        # NEW: TPS for segmentation
        if all_segmentation_metrics:
            save_seg_tps_file(all_segmentation_metrics, combined_scales, args.output_dir)
        else:
            logging.info("No segmentation to write TPS.")
        
        # --- Inter-mask distances CSV ---
        if all_inter_mask_distances:
            imd_csv = os.path.join(args.output_dir, 'inter_mask_distances.csv')
            imd_fields = ['image_filename', 'image_id', 'category_a', 'ann_idx_a',
                          'category_b', 'ann_idx_b', 'centroid_distance_px',
                          'min_boundary_distance_px', 'overlap',
                          'centroid_distance_mm', 'min_boundary_distance_mm']
            with open(imd_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=imd_fields, extrasaction='ignore')
                writer.writeheader()
                for row in all_inter_mask_distances:
                    writer.writerow(row)
            logging.info(f"Inter-mask distances saved to {imd_csv} ({len(all_inter_mask_distances)} pairs)")

        # --- Augmented COCO JSON with measurements ---
        save_augmented_coco_json(
            args.json, args.output_dir, annotations_dict, images_info,
            categories, per_ann_measurements, combined_scales,
            all_inter_mask_distances)

        # --- JSON-LD Knowledge Graph ---
        save_jsonld_knowledge_graph(
            args.output_dir, annotations_dict, images_info,
            categories, per_ann_measurements, combined_scales,
            all_inter_mask_distances)
     
        # ----- RE-ADD combined CSV/JSONL output -----
        if all_metrics and args.output_file:
            save_all_metrics_to_csv(args.output_file, all_metrics)
            logging.debug(f"Metrics saved to CSV: {args.output_file}")

        if all_metrics and args.jsonl_output:
            save_metrics_to_jsonl(args.jsonl_output, all_metrics)
            logging.debug(f"Metrics saved to JSONL: {args.jsonl_output}")

        # ----- RE-ADD Overall Metrics Analysis & Plots -----
        if all_metrics:
            feature_keys = [
                'length_pixels', 'height_pixels',
                'area_pixels', 'perimeter_pixels',
                'major_axis_length_pixels', 'minor_axis_length_pixels'
            ]
            feature_vectors = []
            for m in all_metrics:
                try:
                    feature_vectors.append([m[k] for k in feature_keys])
                except KeyError:
                    continue
            feature_vectors = np.array(feature_vectors)
            if feature_vectors.size:
                umap_results, _ = perform_umap(feature_vectors, n_components=2)
                plot_pca_umap(
                    umap_results, images_list, masks_list,
                    args.output_dir, args.category_name, method='UMAP'
                )

                pca = PCA(n_components=2)
                pca_results = pca.fit_transform(feature_vectors)
                plot_pca_umap(
                    pca_results, images_list, masks_list,
                    args.output_dir, args.category_name, method='PCA'
                )

                plot_hierarchical_dendrogram(
                    feature_vectors, args.output_dir, args.category_name
                )
            else:
                logging.warning("No valid feature vectors for overall analysis.")
        else:
            logging.warning("No metrics available for overall analysis.")

        # ----- RE-ADD Keypoints-Specific Analysis & Plots -----
        if all_keypoints_metrics:
            kp_keys = [
                'num_visible_keypoints', 'mean_distance', 'std_distance',
                'max_distance', 'min_distance', 'median_distance'
            ]
            kp_vectors = []
            for km in all_keypoints_metrics:
                try:
                    kp_vectors.append([km['summary'][k] for k in kp_keys])
                except KeyError:
                    continue
            kp_vectors = np.array(kp_vectors)
            if kp_vectors.size > 0:
                # PCA on keypoints
                kp_pca = PCA(n_components=2).fit_transform(kp_vectors)
                # NEW: default category_name if None
                kp_category = args.category_name if args.category_name else "unknown"
                plot_pca_umap(
                    kp_pca,
                    keypoint_images_list,
                    [np.ones_like(img[:, :, 0]) * 255 for img in keypoint_images_list],
                    args.output_dir,
                    kp_category + "_keypoints",
                    method='PCA'
                )
                # UMAP on keypoints
                kp_umap, _ = perform_umap(kp_vectors, n_components=2)
                plot_pca_umap(
                    kp_umap,
                    keypoint_images_list,
                    [np.ones_like(img[:, :, 0]) * 255 for img in keypoint_images_list],
                    args.output_dir,
                    kp_category + "_keypoints",
                    method='UMAP'
                )
                # Dendrogram on keypoints
                plot_hierarchical_dendrogram(
                    kp_vectors,
                    args.output_dir,
                    kp_category + "_keypoints"
                )
            else:
                logging.warning("No valid keypoint feature vectors for analysis.")
        else:
            logging.info("No keypoint metrics available for keypoints analysis.")

        logging.info("Processing completed successfully.")
    
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.DEBUG)
        main()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

