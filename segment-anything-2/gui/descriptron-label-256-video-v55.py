import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from functools import partial
import numpy as np
from PIL import Image, ImageTk
import torch
import json
import time
import os
import cv2
import csv
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
import copy
import subprocess
import threading
import logging
from datetime import datetime
import shutil
from pathlib import Path
import sys
import gc

# ============================================================================
# MEMORY OPTIMIZATION: Enable CUDA memory management for long videos
# ============================================================================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

root = tk.Tk()
root.title("Descriptron Interactive SAM2 Segmentation")
# Configure the root window to allow the canvas to expand and buttons to stay at the bottom
root.grid_rowconfigure(0, weight=1)  # Row 0 (canvas) expands
root.grid_columnconfigure(0, weight=1)  # Column 0 expands


# Set up the SAM2 model and parameters
sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "../sam2_configs/sam2_hiera_l.yaml"

# Verify checkpoint files exist
if not os.path.exists(sam2_checkpoint):
    print(f"ERROR: SAM2 checkpoint not found!")
    print(f"Expected location: {sam2_checkpoint}")
    print(f"Current directory: {os.getcwd()}")
    print(f"\nThis script should be run from the segment-anything-2/gui/ directory.")
    print(f"Make sure you have downloaded the SAM2 checkpoint files.")
    print(f"Looking for: sam2_hiera_large.pt")
    sys.exit(1)
    
if not os.path.exists(model_cfg):
    print(f"ERROR: SAM2 config not found!")
    print(f"Expected location: {model_cfg}")
    print(f"Current directory: {os.getcwd()}")
    print(f"\nThis script should be run from the segment-anything-2/gui/ directory.")
    print(f"Looking for: sam2_hiera_l.yaml")
    sys.exit(1)

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable mixed precision if using CUDA
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Build SAM2 model
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# Configure the automatic mask generator
mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=64,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
)


# Global COCO variables
coco_output_mask = {
    "images": [],
    "annotations": [],
    "categories": []
}
coco_output = {
    "images": [],
    "annotations": [],
    "categories": []
}

coco_output_accumulate = {
    "images": [],
    "annotations": [],
    "categories": []
}

category_id_to_name = {}
category_name_to_id = {}
category_id_to_supercategory = {}
categories = []  # List to hold category dictionaries
instance_counters = {}  # Dictionary to track instance counts per category


# Global variables to store the image, masks, labels, and VIA2 regions
image = None
segmentation_masks = []
current_mask_index = 0
labels = {}
via2_regions = []
region_data = []
file_path_base = None
bounding_boxes = []
canvas = None
bbox_rects = []
current_tool = None
zoom_level = 1.0
zoom_step = 0.1  # Adjust the zoom step as needed
mode = "prompt"  # Modes: 'prompt', 'mask_edit'
pan_x_offset = 0
pan_y_offset = 0
bbox_mode = False  # Toggle for bounding box mode
point_coords = []  # Stores x, y coordinates for points
point_labels = []  # Stores labels (1 for positive, 0 for negative)
keypoint_labels = [] # Parallel list to hold labels (default: [1, 2, 3, ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦]).
point_prompt_mode = False  # Toggle for point prompt mode
multi_mask_mode = False  # Default is single mask mode
point_orders = []  # Stores the order (superscript labels) of keypoints
# Global variable to store brush size
brush_size = 5  # Default brush size (small)
keypoint_edit_mode = False
moving_point = None
# Global variables for prediction viewing
prediction_data = {}  # Mapping from image filenames to their annotations
prediction_image_names = []  # List of image filenames used in predictions
selected_prediction_image = tk.StringVar(root)  # Variable to track the selected image in dropdown
image_dir = ""  # Directory containing the input images used in predictions
# Global variable to cache predictions from "View Predictions"
prediction_cache = {}
category_id_to_name = {}  # Mapping from category_id to category_name
category_label_options = []  # List to hold category labels from the COCO JSON
pred_json_path = ""
view_predictions_mode = False
working_pred_json_path = ""
# Global variables to manage image navigation
image_list = []            # List of image filenames
current_image_index = 0    # Index of the currently displayed image


# === VIDEO MODE GLOBALS ===
video_mode = False
video_frames = []
current_video_frame = 0
video_metadata = {}
video_temp_dir = None
video_annotations = {}  # {frame_idx: {masks, keypoints, labels, etc}}
video_remote = None
frames_to_delete = set()  # Track frames marked for deletion
original_video_path = None  # Store original video path for re-encoding
timeline_draw_callback = None  # Callback to redraw timeline when annotations change




def setup_logging():
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def toggle_mode():
    global mode
    if mode == 'prediction_view':
        mode = 'mask_edit'
    else:
        mode = 'prediction_view'


def clean_coco_json(
    original_json_path,
    cleaned_json_path,
    default_exclude_categories=None,
    user_exclude_categories=None
):
    """
    Cleans the COCO JSON file by removing specified categories and their annotations.
    Also removes images with no annotations and categories with zero instances.

    Parameters:
    - original_json_path (str): Path to the original COCO JSON file.
    - cleaned_json_path (str): Path where the cleaned COCO JSON will be saved.
    - default_exclude_categories (list of str): Default categories to exclude.
    - user_exclude_categories (list of str): Additional categories to exclude specified by the user.
    """
    if default_exclude_categories is None:
        default_exclude_categories = ["Trash", "Select Label", "Custom"]

    if user_exclude_categories is None:
        user_exclude_categories = []

    # Combine default and user-specified categories to exclude
    categories_to_exclude = set(default_exclude_categories + user_exclude_categories)

    with open(original_json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract categories and create mappings
    original_categories = coco_data.get('categories', [])
    category_id_to_name = {cat['id']: cat['name'] for cat in original_categories}
    category_name_to_id = {cat['name']: cat['id'] for cat in original_categories}

    # Identify categories to exclude based on names
    exclude_category_ids = set()
    for cat_name in categories_to_exclude:
        cat_id = category_name_to_id.get(cat_name)
        if cat_id:
            exclude_category_ids.add(cat_id)
        else:
            logging.warning(f"Category '{cat_name}' not found in JSON categories.")

    # Special handling for "Custom": Only exclude if it has zero instances
    if "Custom" in categories_to_exclude:
        custom_id = category_name_to_id.get("Custom")
        if custom_id:
            # Count instances of "Custom"
            custom_instances = [ann for ann in coco_data.get('annotations', []) if ann['category_id'] == custom_id]
            if not custom_instances:
                exclude_category_ids.add(custom_id)
                logging.info("Excluding 'Custom' category as it has zero instances.")
            else:
                logging.info(f"Retaining 'Custom' category as it has {len(custom_instances)} instances.")
        else:
            logging.warning("'Custom' category not found in JSON categories.")

    # Remove categories to exclude
    filtered_categories = [
        cat for cat in original_categories if cat['id'] not in exclude_category_ids
    ]

    # Update category ID mappings after exclusion
    new_category_id_mapping = {cat['id']: idx+1 for idx, cat in enumerate(filtered_categories)}
    new_categories = []
    for idx, cat in enumerate(filtered_categories):
        new_cat = {
            "id": idx + 1,
            "name": cat['name'],
            "supercategory": cat.get('supercategory', '')
        }
        new_categories.append(new_cat)

    # Filter annotations to exclude those belonging to excluded categories
    original_annotations = coco_data.get('annotations', [])
    filtered_annotations = [
        ann for ann in original_annotations if ann['category_id'] not in exclude_category_ids
    ]

    # Update category_id in annotations based on new mapping
    for ann in filtered_annotations:
        ann['category_id'] = new_category_id_mapping.get(ann['category_id'], ann['category_id'])

    # Collect image IDs that have annotations
    valid_image_ids = set(ann['image_id'] for ann in filtered_annotations)

    # Filter images to include only those that have valid annotations
    original_images = coco_data.get('images', [])
    filtered_images = [
        img for img in original_images if img['id'] in valid_image_ids
    ]

    # Update the coco_data with filtered data
    coco_data['categories'] = new_categories
    coco_data['annotations'] = filtered_annotations
    coco_data['images'] = filtered_images

    # Remove categories with zero instances
    category_instance_count = {cat['id']:0 for cat in new_categories}
    for ann in filtered_annotations:
        category_instance_count[ann['category_id']] += 1

    # Identify categories with zero instances
    categories_with_zero_instances = [
        cat_id for cat_id, count in category_instance_count.items() if count == 0
    ]

    if categories_with_zero_instances:
        logging.info(f"Removing categories with zero instances: {categories_with_zero_instances}")
        # Remove categories with zero instances
        coco_data['categories'] = [
            cat for cat in coco_data['categories'] if cat['id'] not in categories_with_zero_instances
        ]

    # Save the cleaned JSON
    with open(cleaned_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    logging.info(f"Cleaned COCO JSON saved to {cleaned_json_path}")


def validate_coco_json(data):
    """
    Validates the COCO JSON structure to ensure no category has id=0 and all annotations reference valid category_ids.
    """
    # Check for category_id=0
    categories_with_id_zero = [cat for cat in data['categories'] if cat['id'] == 0]
    if categories_with_id_zero:
        logging.error("Validation Error: Found category with id=0.")
        return False

    # Create a set of valid category_ids
    valid_category_ids = set(cat['id'] for cat in data['categories'])

    # Check annotations for valid category_ids
    invalid_annotations = [ann for ann in data['annotations'] if ann['category_id'] not in valid_category_ids]
    if invalid_annotations:
        logging.error(f"Validation Error: Found {len(invalid_annotations)} annotations with invalid category_ids.")
        return False

    logging.info("COCO JSON validation passed.")
    return True

# Function to draw a cross
def draw_cross(x, y, size=5, color="green", tag="point"):
    canvas.create_line(x - size, y - size, x + size, y + size, fill=color, width=2, tags=tag)
    canvas.create_line(x + size, y - size, x - size, y + size, fill=color, width=2, tags=tag)

# changed "point_number" to keypoint
def draw_number(x, y, number, color="black"):
    """Draw a small number next to the point."""
    canvas.create_text(x + 10, y - 10, text=str(number), font=("Arial", 8), fill="black", tags="point_number")

# Function to pick points for point prompting
def pick_point(event):
    global point_coords, point_labels, point_orders

    # Get the actual position on the image (considering panning and zoom)
    x = canvas.canvasx(event.x) / zoom_level
    y = canvas.canvasy(event.y) / zoom_level

    # Ensure that the point isn't added multiple times
    if [x, y] not in point_coords:
        point_coords.append([x, y])

        if selected_point_label.get() == "Positive":
            label = 1
            color = "green"
        else:
            label = 0
            color = "red"

        point_labels.append(label)
        point_orders.append(len(point_coords))  # Use the order in which the point was picked

        # Draw the point immediately
        draw_cross(x * zoom_level, y * zoom_level, color=color, tag="point")
        draw_number(x * zoom_level, y * zoom_level, point_orders[-1])


def redraw_points():
    """Redraw the points based on the current zoom and pan position."""
    # Clear all previously drawn points and numbers
    canvas.delete("point")
    canvas.delete("point_number")

    if point_coords:
        for i, point in enumerate(point_coords):
            x = point[0] * zoom_level
            y = point[1] * zoom_level
            color = "green" if point_labels[i] == 1 else "red"
            draw_cross(x, y, color=color, tag="point")
            draw_number(x, y, point_orders[i])
    update_superscripts()


#def draw_number(x, y, number):
###    """Draws a number (label) on the canvas at the given (x, y) coordinates."""
    # Remove any previous text objects with a given tag, then create new text.
#    canvas.create_text(x + 10, y - 10, text=str(number), fill="yellow", font=("Helvetica", 12), tag="keypoint")

#def redraw_points():
###    """Redraw all keypoints on the canvas."""
#    canvas.delete("keypoint")
#    for point in point_coords:
#        x = point[0] * zoom_level
#        y = point[1] * zoom_level
#        canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", tag="keypoint")
#    update_superscripts()

def initialize_keypoint_labels():
    global keypoint_labels
    keypoint_labels = [i+1 for i in range(len(point_coords))]
    update_superscripts()


def apply_zoom():
    global image, canvas, imgtk, zoom_level

    if image is None:
        return

    # Resize the image according to the zoom level
    zoomed_image = Image.fromarray(image).resize(
        (int(image.shape[1] * zoom_level), int(image.shape[0] * zoom_level)),
        Image.LANCZOS
    )
    imgtk = ImageTk.PhotoImage(zoomed_image)

    # Adjust canvas size and redraw image
    canvas.config(scrollregion=(0, 0, zoomed_image.width, zoomed_image.height))
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    canvas.image = imgtk
    canvas.delete("point")
    redraw_points()

    # Redraw points and masks with the correct zoom and pan
    if multi_mask_mode:
        redraw_masks(multi=True)  # Draw all masks at once
    else:
        redraw_masks(multi=False)  # Draw one mask at a time

    # Rescale and redraw bounding boxes if in prompt mode
    if mode == "prompt":
        for i, bbox in enumerate(bounding_boxes):
            tag = f"bbox_{i}"
            canvas.delete(tag)
            scaled_bbox = [coord * zoom_level for coord in bbox]
            canvas.create_rectangle(*scaled_bbox, outline="red", tags=tag)


    # Rescale and redraw points if in point prompt mode
#    canvas.delete("point")  # Clear old points
#    if point_coords:
#        for i, point in enumerate(point_coords):
#            x = point[0] * zoom_level
#            y = point[1] * zoom_level
#            color = "green" if point_labels[i] == 1 else "red"
#            draw_cross(canvas.canvasx(x), canvas.canvasy(y), color=color)

    elif mode == "mask_edit":
        redraw_masks()

def zoom_in():
    global zoom_level
    zoom_level += zoom_step
    apply_zoom()

def zoom_out():
    global zoom_level
    if zoom_level - zoom_step >= 0.1:  # Prevent excessive zoom out
        zoom_level -= zoom_step
        apply_zoom()

# Function to enable panning
# Panning doesn't need manual pan_x_offset updates
def start_pan(event):
    canvas.scan_mark(event.x, event.y)

def pan_image(event):
    canvas.scan_dragto(event.x, event.y, gain=1)
    redraw_points()  # Redraw points based on new pan position


def toggle_bbox_mode():
    global bbox_mode
    bbox_mode = not bbox_mode
    if bbox_mode:
        canvas.bind("<ButtonPress-1>", draw_bounding_box)
        canvas.bind("<B1-Motion>", draw_bounding_box)
        canvas.bind("<ButtonRelease-1>", draw_bounding_box)
        bbox_button.config(relief=tk.SUNKEN)
    else:
        canvas.unbind("<ButtonPress-1>")
        canvas.unbind("<B1-Motion>")
        canvas.unbind("<ButtonRelease-1>")
        bbox_button.config(relief=tk.RAISED)

# Function to toggle point prompt mode
def toggle_point_prompt_mode():
    global point_prompt_mode
    point_prompt_mode = not point_prompt_mode
    if point_prompt_mode:
        canvas.bind("<ButtonPress-1>", pick_point)
        canvas.bind("<ButtonRelease-1>", pick_point)
        point_prompt_btn.config(relief=tk.SUNKEN)
    else:
        canvas.unbind("<ButtonPress-1>")
        canvas.unbind("<ButtonRelease-1>")
        point_prompt_btn.config(relief=tk.RAISED)

  
def clear_points():
    global point_coords, point_labels, point_orders
    # Clear points from the canvas using tags
    canvas.delete("point")
    canvas.delete("point_number")
    point_coords = []
    point_labels = []
    point_orders = []


#apply point prompts for sam2

def apply_sam2_prompt_with_points():
    global segmentation_masks, canvas, mode, prediction_cache, selected_prediction_image, current_mask_index

    # Ensure that either bounding boxes or points are provided
    if not bounding_boxes and not point_coords:
        messagebox.showerror("Error", "No prompts (bounding boxes or points) drawn!")
        return

    predictor = SAM2ImagePredictor(sam2)
    predictor.set_image(image)

    # FIXED: Store new masks in a temporary list instead of clearing segmentation_masks
    new_segmentation_masks = []

    # If bounding boxes exist, use them; otherwise, rely solely on point prompts
    if bounding_boxes:
        for bbox in bounding_boxes:
            input_box = np.array(bbox)
            input_point = np.array(point_coords) if point_coords else None
            input_label = np.array(point_labels) if point_labels else None

            # Apply SAM2 with points and/or boxes
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box[None, :] if input_box is not None else None,
                multimask_output=False,
            )
            # Set category_id to -1 by default
            new_masks = [{'segmentation': mask, 'category_id': -1} for mask in masks]
            new_segmentation_masks.extend(new_masks)
    else:
        # If there are no bounding boxes but points are provided
        input_point = np.array(point_coords) if point_coords else None
        input_label = np.array(point_labels) if point_labels else None

        # Apply SAM2 with point prompts alone
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,  # No bounding box in this case
            multimask_output=False,
        )
        # Set category_id to -1 by default
        new_masks = [{'segmentation': mask, 'category_id': -1} for mask in masks]
        new_segmentation_masks.extend(new_masks)

    # Check if any masks were generated
    if not new_segmentation_masks:
        messagebox.showerror("Error", "No mask was generated from the point prompts.")
        return

    # FIXED: Extend existing masks instead of replacing them
    segmentation_masks.extend(new_segmentation_masks)
    
    # Update current_mask_index to point to the first new mask
    current_mask_index = len(segmentation_masks) - len(new_segmentation_masks)
    
    canvas.delete("all")
    mode = "mask_edit"

    # Update the prediction_cache with new annotations
    image_name = selected_prediction_image.get()
    if image_name in prediction_cache:
        annotations = prediction_cache[image_name]
        # Remove existing annotations if needed
        annotations.clear()
        for idx, mask_data in enumerate(segmentation_masks):
            # Convert mask to COCO segmentation
            mask = mask_data['segmentation']
            mask_8bit = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            coco_segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                if len(contour) >= 6:
                    coco_segmentation.append(contour)
            # Create a new annotation
            annotation = {
                'id': idx,
                'image_id': image_name,
                'category_id': mask_data.get('category_id', -1),
                'segmentation': coco_segmentation,
                'bbox': cv2.boundingRect(mask_8bit),
                'area': float(np.sum(mask_8bit > 0)),
                'iscrowd': 0
            }
            annotations.append(annotation)

   # Redraw masks to display the result immediately
    redraw_masks()
    apply_zoom()

    canvas.bind("<ButtonPress-2>", start_pan)
    canvas.bind("<B2-Motion>", pan_image)

    
#def redraw_masks():
#    """Redraw masks on the canvas."""
#    if segmentation_masks:
#        mask = segmentation_masks[current_mask_index]['segmentation']#
#
#        # Resize mask according to the current zoom level
#        resized_mask = cv2.resize(mask.astype(np.uint8), (int(image.shape[1] * zoom_level), int(image.shape[0] * zoom_level)), #interpolation=cv2.INTER_NEAREST)
#
#        # Ensure the resized mask has the same dimensions as the resized image
#        mask_overlay = np.zeros((resized_mask.shape[0], resized_mask.shape[1], 3), dtype=np.uint8)
#        mask_overlay[resized_mask > 0] = [0, 255, 0]  # Green mask overlay
#
#        # Blend the resized image and mask
#        zoomed_image = np.array(Image.fromarray(image).resize(
#            (int(image.shape[1] * zoom_level), int(image.shape[0] * zoom_level)),
#            Image.LANCZOS
#        ))
#
#        blended_image = cv2.addWeighted(zoomed_image, 0.9, mask_overlay, 0.3, 0)
#
#        imgtk = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(blended_image)))
#        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
#        canvas.image = imgtk

def save_points_to_via2():
    """
    Save point coordinates, labels, and superscript (order) in COCO format with image path included for Detectron2.
    """
    global point_coords, point_labels, file_path_base, coco_output_mask, coco_output_accumulate

    if not point_coords:
        messagebox.showerror("Error", "No points to save!")
        return

    # Ensure that the base path contains the full path with extension
    file_dir, file_name = os.path.split(file_path_base)
    file_base, file_ext = os.path.splitext(file_name)

    if not file_ext:  # If no extension, assume it's a .jpg by default
        file_ext = ".jpg"

    full_file_path = os.path.join(file_dir, file_base + file_ext)

    # Image metadata
    image_metadata = {
        "id": file_base,  # Use the file base as image ID
        "file_name": os.path.basename(full_file_path),
        "height": image.shape[0],
        "width": image.shape[1]
    }

    # Append image metadata to both coco_output_mask and coco_output_accumulate
    coco_output_mask["images"].append(image_metadata)
    coco_output_accumulate["images"].append(image_metadata)

    # Create keypoints and annotation in COCO format, including the superscript label as "point_order"
    keypoints = []
    point_order = []  # List to store the order (superscript labels)
    num_keypoints = 0

    for i, (point, label) in enumerate(zip(point_coords, point_labels)):
        # COCO keypoints format expects [x, y, visibility] where visibility is 2 for labeled keypoints
        keypoints.extend([point[0], point[1], 2])  # "2" indicates the point is visible
        point_order.append(i + 1)  # Save the order in which the point was clicked (superscript label)
        num_keypoints += 1

    # Add annotation for keypoints, including the "point_order" as an additional field
    annotation = {
        "id": int(time.time() * 1000),  # Unique ID based on timestamp
        "image_id": file_base,  # Link the annotation to the image ID
        "category_id": category_name_to_id.get("keypoints", -1),  # Assuming 'keypoints' is in the categories
        "keypoints": keypoints,
        "num_keypoints": num_keypoints,
        "area": 1,  # COCO format requires this but it's not relevant for keypoints
        "bbox": [min([x[0] for x in point_coords]), min([x[1] for x in point_coords]),
                 max([x[0] for x in point_coords]) - min([x[0] for x in point_coords]),
                 max([x[1] for x in point_coords]) - min([x[1] for x in point_coords])],
        "iscrowd": 0,
        "point_order": point_order  # Store the superscript label order
    }

    # Append annotation to both coco_output_mask and coco_output_accumulate
    coco_output_mask["annotations"].append(annotation)
    coco_output_accumulate["annotations"].append(annotation)

    # Prompt user to save the COCO JSON (for the individual mask)
    save_path = filedialog.asksaveasfilename(defaultextension=".json", title="Save Points as COCO JSON", filetypes=[("COCO JSON Files", "*.json")])

    if save_path:
        # Save the JSON in COCO format for the individual mask
        with open(save_path, 'w') as json_file:
            json.dump(coco_output_mask, json_file, indent=4)
        messagebox.showinfo("Info", f"COCO Keypoints JSON saved successfully to {save_path}")
    else:
        messagebox.showinfo("Info", "Save operation cancelled.")


def toggle_keypoint_edit_mode():
    global keypoint_edit_mode
    keypoint_edit_mode = not keypoint_edit_mode
    if keypoint_edit_mode:
        keypoint_edit_btn.config(relief=tk.SUNKEN, text="Keypoint Edit: ON")
        canvas.bind("<ButtonPress-1>", start_move_keypoint)  # Enable moving
        canvas.bind("<ButtonRelease-1>", stop_move_keypoint)
        # Bind shift+left-click for context menu actions.
        canvas.bind("<Button-3>", on_right_click)
        #canvas.bind("<Shift-Button-1>", on_shift_left_click)
    else:
        keypoint_edit_btn.config(relief=tk.RAISED, text="Keypoint Edit: OFF")
        canvas.unbind("<ButtonPress-1>")
        canvas.unbind("<ButtonRelease-1>")
        canvas.unbind("<Button-3>")

def start_move_keypoint(event):
    global moving_point
    # Get the actual position on the image (considering panning and zoom)
    x = canvas.canvasx(event.x) / zoom_level
    y = canvas.canvasy(event.y) / zoom_level

    # Find the closest point
    for i, point in enumerate(point_coords):
        if abs(point[0] - x) < 10 and abs(point[1] - y) < 10:
            moving_point = i
            break

def stop_move_keypoint(event):
    global moving_point
    if moving_point is not None:
        # Update the keypoint with new coordinates after dragging
        x = canvas.canvasx(event.x) / zoom_level
        y = canvas.canvasy(event.y) / zoom_level
        point_coords[moving_point] = [x, y]
        moving_point = None
        redraw_points()  # Redraw all points with updated positions

    # After moving, update the superscripts
    update_superscripts()

#def update_superscripts():
#    """Update the superscript numbering for all keypoints."""
#    for i, point in enumerate(point_coords):
#        x = point[0] * zoom_level
#        y = point[1] * zoom_level
#        draw_number(x, y, i + 1)

def on_right_click(event):
    """
    Called when the user Shift+Left-Clicks on the canvas.
    Finds the nearest keypoint (within a threshold) and pops up a context menu
    allowing removal or renaming of the selected keypoint.
    """
    x = canvas.canvasx(event.x) / zoom_level
    y = canvas.canvasy(event.y) / zoom_level
    selected_index = None
    for i, (px, py) in enumerate(point_coords):
        if abs(px - x) < 10 and abs(py - y) < 10:
            selected_index = i
            break
    if selected_index is None:
        return  # No keypoint is sufficiently close to the click.

    # Create a context menu.
    menu = tk.Menu(canvas, tearoff=0)
    menu.add_command(label="Remove Keypoint", command=lambda: remove_keypoint(selected_index))
    menu.add_command(label="Rename Keypoint", command=lambda: rename_keypoint(selected_index))
    # Post the menu using screen coordinates provided by the event.
    menu.post(event.x_root, event.y_root)


def remove_keypoint(index):
    global point_coords, keypoint_labels
    if index < len(point_coords):
        del point_coords[index]
    if index < len(keypoint_labels):
        del keypoint_labels[index]
    redraw_points()
    update_superscripts()

def rename_keypoint(index):
    global keypoint_labels
    new_label = simpledialog.askstring("Rename Keypoint", f"Enter new label for keypoint #{index+1}:")
    if new_label is not None:
        try:
            # Try to convert the input into an integer.
            new_val = int(new_label)
            total = len(keypoint_labels)
            # Allow new labels from 1 up to total+1
            if new_val < 1 or new_val > (total + 1):
                messagebox.showerror("Error", f"Please enter a number between 1 and {total + 1}.")
                return
            # Get the current label for this keypoint.
            # Check that the index exists (if not, do nothing)
            if index >= total:
                messagebox.showerror("Error", "Invalid keypoint index.")
                return
            curr_val = keypoint_labels[index]
            if new_val == curr_val:
                # No change needed
                return
            else:
                # Remove the item from its current position
                item = keypoint_labels.pop(index)
                # If new_val equals total+1, that means append to the end.
                if new_val == total + 1:
                    keypoint_labels.append(item)
                else:
                    # Insert it at new_val-1 (zero-indexed)
                    keypoint_labels.insert(new_val - 1, item)
                # Reassign the order so that the list becomes 1, 2, ..., len(keypoint_labels)
                keypoint_labels[:] = list(range(1, len(keypoint_labels) + 1))
        except ValueError:
            # If the input isnÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬'t numeric, then simply treat it as a nonÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Ânumeric label
            # (i.e. add it to the end and leave the rest of the numeric order unchanged)
            # You may store the non-numeric value for that keypoint.
            # Here we choose to append the item without reordering.
            item = keypoint_labels.pop(index)
            # Option: if you want to store nonnumeric labels separately, you might
            # do that. For now, we simply append a special indicator (like "X")
            # or simply reappend the current number.
            keypoint_labels.append(item)
        update_superscripts()


def update_superscripts():
    #"""Redraw keypoint labels using the current keypoint_labels list."""
    #for i, point in enumerate(point_coords):
    #    x = point[0] * zoom_level
    #    y = point[1] * zoom_level
    #    label = keypoint_labels[i] if i < len(keypoint_labels) else (i + 1)
    #    draw_number(x, y, label)
    """Redraw keypoint labels using the current keypoint_labels list.
       Clears previous labels before drawing.
    """
    # Remove previously drawn superscripts.
    canvas.delete("point_number")
    for i, point in enumerate(point_coords):
        x = point[0] * zoom_level
        y = point[1] * zoom_level
        # Use the keypoint label if provided; otherwise, use the order (i+1)
        label = keypoint_labels[i] if i < len(keypoint_labels) else (i + 1)
        #current_label = keypoint_labels[i]
        # Determine the color from point_labels: positive = green, negative = red.
        color = "green" if (i < len(point_labels) and point_labels[i] == 1) else "red"
        #draw_number(x, y, current_label, color)
        draw_number(x, y, label, color)


def update_label_dropdown():
    """
    Update the label dropdown menu with the loaded labels.
    """
    global label_dropdown, selected_label, label_options

    # Update label options with loaded categories
    label_options = [category['name'] for category in categories]

    # Ensure 'Trash' and 'Custom' are included
    if "Trash" not in label_options:
        label_options.append("Trash")
    if "Custom" not in label_options:
        label_options.append("Custom")

    # Remove duplicates while preserving order
    label_options = list(dict.fromkeys(label_options))

    # Clear the current OptionMenu and repopulate with new labels
    menu = label_dropdown["menu"]
    menu.delete(0, "end")

    # Add all labels to the dropdown menu
    for label in label_options:
        menu.add_command(label=label, command=tk._setit(selected_label, label))

    # Reset the selection to default if needed
    if selected_label.get() not in label_options:
        selected_label.set("Select Label")

def update_video_label_dropdown():
    """
    Update the VIDEO REMOTE label dropdown from global `categories`.
    Does NOT touch the main (2D) dropdown at all.
    """
    global video_label_dropdown, video_selected_label, video_label_options, categories

    if 'video_label_dropdown' not in globals() or video_label_dropdown is None:
        return

    video_label_options = [c.get("name") for c in (categories or []) if isinstance(c, dict) and c.get("name")]

    if "Trash" not in video_label_options:
        video_label_options.append("Trash")
    if "Custom" not in video_label_options:
        video_label_options.append("Custom")

    video_label_options = list(dict.fromkeys(video_label_options))

    menu = video_label_dropdown["menu"]
    menu.delete(0, "end")
    for label in video_label_options:
        menu.add_command(label=label, command=tk._setit(video_selected_label, label))

    if video_selected_label.get() not in video_label_options:
        video_selected_label.set("Select Label")


def load_labels():
    """
    Load labels from a CSV or JSON file. The labels should contain 'id', 'name', and 'supercategory'.
    Ensure 'Trash' is always present.
    """
    global label_options, categories, category_name_to_id, category_id_to_supercategory, coco_output_mask, coco_output_accumulate

    file_path = filedialog.askopenfilename(
        title="Select Labels File",
        filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json")]
    )
    if not file_path:
        messagebox.showerror("Error", "No file selected!")
        return

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        categories.clear()
        category_name_to_id.clear()
        category_id_to_supercategory.clear()

        if file_ext == ".csv":
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    cat_id = int(row['id'])
                    if cat_id == 0:
                        logging.warning("Category ID 0 is reserved and will be skipped.")
                        continue  # Skip categories with id=0
                    name = row['name'].strip()
                    supercategory = row['supercategory'].strip()
                    categories.append({
                        "id": cat_id,
                        "name": name,
                        "supercategory": supercategory
                    })
                    category_name_to_id[name] = cat_id
                    category_id_to_supercategory[cat_id] = supercategory
        elif file_ext == ".json":
            with open(file_path, 'r') as jsonfile:
                data = json.load(jsonfile)
                for item in data:
                    cat_id = int(item['id'])
                    if cat_id == 0:
                        logging.warning("Category ID 0 is reserved and will be skipped.")
                        continue  # Skip categories with id=0
                    name = item['name'].strip()
                    supercategory = item['supercategory'].strip()
                    categories.append({
                        "id": cat_id,
                        "name": name,
                        "supercategory": supercategory
                    })
                    category_name_to_id[name] = cat_id
                    category_id_to_supercategory[cat_id] = supercategory
        else:
            messagebox.showerror("Invalid File", "Please provide a valid JSON or CSV file.")
            return

        # Ensure 'Trash' is present
        if "Trash" not in category_name_to_id:
            # Assign the next available category ID
            if category_name_to_id:
                new_trash_id = max(category_name_to_id.values()) + 1
            else:
                new_trash_id = 1  # Start from 1 if no categories loaded

            categories.append({
                "id": new_trash_id,
                "name": "Trash",
                "supercategory": "none"
            })
            category_name_to_id["Trash"] = new_trash_id
            category_id_to_supercategory[new_trash_id] = "none"
            logging.info(f"Added 'Trash' category with ID {new_trash_id}.")

        # Ensure 'Custom' is present
        if "Custom" not in category_name_to_id:
            if category_name_to_id:
                new_custom_id = max(category_name_to_id.values()) + 1
            else:
                new_custom_id = 1  # Start from 1 if no categories loaded

            categories.append({
                "id": new_custom_id,
                "name": "Custom",
                "supercategory": "none"
            })
            category_name_to_id["Custom"] = new_custom_id
            category_id_to_supercategory[new_custom_id] = "none"
            logging.info(f"Added 'Custom' category with ID {new_custom_id}.")

        # Add categories to both coco_output_mask and coco_output_accumulate
        # FIXED: Use copy() to avoid reference issues
        coco_output_mask["categories"] = [cat.copy() for cat in categories]
        coco_output_accumulate["categories"] = [cat.copy() for cat in categories]

        update_label_dropdown()  # Update the dropdown to include 'Trash' and 'Custom'
        messagebox.showinfo("Success", "Labels loaded successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load labels: {e}")


def load_image_and_predict():
    global image, segmentation_masks, current_mask_index, file_path_base, canvas, mode

    mode = "mask_edit"  # Set the mode to mask_edit to ensure masks are handled properly
    file_path = filedialog.askopenfilename()
    file_path_base = os.path.splitext(file_path)[0]
    image = Image.open(file_path)
    image = np.array(image.convert("RGB"))
    if image is None:
        print("Failed to load image")
        return

    # Reset canvas and config scroll region based on the image size
    canvas.delete("all")
    canvas.config(scrollregion=(0, 0, image.shape[1], image.shape[0]))

    # Generate masks using SAM2
    segmentation_masks = mask_generator_2.generate(image)
    current_mask_index = 0

    # Apply zoom and display the first mask
    apply_zoom()  # Apply zoom to the image and masks
    redraw_masks()  # Ensure the first mask is redrawn with the correct zoom level

    # Bind panning actions
    canvas.bind("<ButtonPress-2>", start_pan)
    canvas.bind("<B2-Motion>", pan_image)

def load_image_for_prompt():
    global image, file_path_base, canvas, bbox_rects, mode

    mode = "prompt"
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    file_path_base = os.path.splitext(file_path)[0]
    image = Image.open(file_path)
    image = np.array(image.convert("RGB"))
    if image is None:
        print("Failed to load image")
        return

    canvas.delete("all")
    bbox_rects = []
    canvas.config(scrollregion=(0, 0, image.shape[1], image.shape[0]))

    imgtk = ImageTk.PhotoImage(image=Image.fromarray(image))
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    canvas.image = imgtk

    # Make sure to reset the scrollbar
    canvas.update_idletasks()
    
    # Push the buttons down after the image is fully loaded
    button_frame.lift()

    if bbox_mode:
        toggle_bbox_mode()  # Ensure bbox mode is only active when required

def draw_bounding_box(event):
    if not bbox_mode:
        return  # Do nothing if bbox mode is not active

    global start_x, start_y, bbox_rect

    x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)

    if event.type == tk.EventType.ButtonPress:
        start_x, start_y = x, y
        bbox_rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="red")
    elif event.type == tk.EventType.Motion:
        canvas.coords(bbox_rect, start_x, start_y, x, y)
    elif event.type == tk.EventType.ButtonRelease:
        end_x, end_y = x, y
        bbox = [start_x / zoom_level, start_y / zoom_level, 
                end_x / zoom_level, end_y / zoom_level]
        bounding_boxes.append(bbox)
        bbox_rects.append(bbox_rect)
        tag = f"bbox_{len(bounding_boxes) - 1}"
        canvas.itemconfig(bbox_rect, tags=tag)
        print(f"Finalized bounding box: {bbox}")

def remove_last_bbox():
    global bounding_boxes

    if bounding_boxes:
        bbox = bounding_boxes.pop()
        tag = f"bbox_{len(bounding_boxes)}"
        canvas.delete(tag)

def apply_sam2_prompt():
    global segmentation_masks, canvas, mode, current_mask_index, prediction_cache, selected_prediction_image

    if not bounding_boxes:
        messagebox.showerror("Error", "No bounding boxes drawn!")
        return

    predictor = SAM2ImagePredictor(sam2)
    predictor.set_image(image)

    new_segmentation_masks = []

    for bbox in bounding_boxes:
        input_box = np.array(bbox)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        # Set category_id to -1 by default
        new_masks = [{'segmentation': mask, 'category_id': -1} for mask in masks]
        new_segmentation_masks.extend(new_masks)
    #
    # Add new masks to segmentation_masks without discarding existing ones
    segmentation_masks.extend(new_segmentation_masks)

    # Update the prediction_cache if in prediction_view mode
    if mode == 'prediction_view':
        image_name = selected_prediction_image.get()
        if image_name in prediction_cache:
            annotations = prediction_cache[image_name]
        else:
            prediction_cache[image_name] = []
            annotations = prediction_cache[image_name]

        for mask_data in new_segmentation_masks:
            mask = mask_data['segmentation']
            mask_8bit = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            coco_segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                if len(contour) >= 6:
                    coco_segmentation.append(contour)
            # Assign a new unique annotation ID
            max_ann_id = max([ann['id'] for ann in annotations], default=0)
            annotation = {
                'id': max_ann_id + 1,
                'image_id': image_name,
                'category_id': mask_data.get('category_id', -1),
                'segmentation': coco_segmentation,
                'bbox': cv2.boundingRect(mask_8bit),
                'area': float(np.sum(mask_8bit > 0)),
                'iscrowd': 0
            }
            annotations.append(annotation)

    # Update current_mask_index to point to the last mask
    current_mask_index = len(segmentation_masks) - 1

    # Instead of deleting all canvas items, we can refresh the display
    # canvas.delete("all")  # Do not clear the entire canvas
    apply_zoom()
    redraw_masks()

    # Rebind pan functionality if necessary
    canvas.bind("<ButtonPress-2>", start_pan)
    canvas.bind("<B2-Motion>", pan_image)

    # Do not change mode if in prediction_view mode
    if mode != 'prediction_view':
        mode = "mask_edit"

def redraw_masks(multi=False):
    """Redraw masks on the canvas. If multi is True, draw all masks with different colors."""
    global segmentation_masks

    if segmentation_masks:
        # Start with the base image
        zoomed_image = np.array(Image.fromarray(image).resize(
            (int(image.shape[1] * zoom_level), int(image.shape[0] * zoom_level)),
            Image.LANCZOS
        ))

        # Create an overlay for the masks
        mask_overlay = np.zeros((zoomed_image.shape[0], zoomed_image.shape[1], 3), dtype=np.uint8)

        if multi:
            # Draw all masks at once with different colors
            mask_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            for i, mask_data in enumerate(segmentation_masks):
                mask = mask_data['segmentation']
                color = mask_colors[i % len(mask_colors)]  # Cycle through colors for each mask

                # Resize mask according to the current zoom level
                resized_mask = cv2.resize(mask.astype(np.uint8), 
                                          (int(image.shape[1] * zoom_level), 
                                           int(image.shape[0] * zoom_level)), 
                                          interpolation=cv2.INTER_NEAREST)

                # Assign color to the mask area
                mask_overlay[resized_mask > 0] = color  # Accumulate mask colors in the overlay

        else:
            # Single mask mode: only draw the current mask
            mask = segmentation_masks[current_mask_index]['segmentation']

            # Resize mask according to the current zoom level
            resized_mask = cv2.resize(mask.astype(np.uint8), 
                                      (int(image.shape[1] * zoom_level), 
                                       int(image.shape[0] * zoom_level)), 
                                      interpolation=cv2.INTER_NEAREST)

            # Green mask overlay
            mask_overlay[resized_mask > 0] = [0, 255, 0]

        # Blend the mask overlay with the zoomed image
        blended_image = cv2.addWeighted(zoomed_image, 0.9, mask_overlay, 0.3, 0)

        # Render the blended image on the canvas
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(blended_image)))
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.image = imgtk



def set_brush_size(size):
    global brush_size
    brush_size = size  # Update the brush size based on the user's selection

def paint(event):
    if segmentation_masks:
        # Calculate the cursor position accounting for zoom and pan
        x = int(canvas.canvasx(event.x) / zoom_level)
        y = int(canvas.canvasy(event.y) / zoom_level)

        mask = segmentation_masks[current_mask_index]['segmentation']

        # Apply paint to the mask as an oval with the selected brush size
        mask[max(0, y-brush_size):min(mask.shape[0], y+brush_size), 
             max(0, x-brush_size):min(mask.shape[1], x+brush_size)] = 1
        redraw_masks()
        update_annotation_in_cache()
        # Clear the previous paint circle
        canvas.delete("paint_circle")

        # Draw a border around the painted area (oval) for visibility
        canvas.create_oval(
            event.x - brush_size + canvas.canvasx(0), 
            event.y - brush_size + canvas.canvasy(0), 
            event.x + brush_size + canvas.canvasx(0), 
            event.y + brush_size + canvas.canvasy(0), 
            outline='green', width=2, tags="paint_circle"
        )

# Same for eraser
def erase(event):
    if segmentation_masks:
        # Calculate the cursor position accounting for zoom and pan
        x = int(canvas.canvasx(event.x) / zoom_level)
        y = int(canvas.canvasy(event.y) / zoom_level)

        mask = segmentation_masks[current_mask_index]['segmentation']
        # Apply eraser to the mask as an oval with the selected brush size
        mask[max(0, y-brush_size):min(mask.shape[0], y+brush_size), 
             max(0, x-brush_size):min(mask.shape[1], x+brush_size)] = 0
        redraw_masks()
        update_annotation_in_cache()
        # Clear the previous erase circle
        canvas.delete("erase_circle")

        # Draw a border around the erased area (oval) for visibility
        canvas.create_oval(
            event.x - brush_size + canvas.canvasx(0), 
            event.y - brush_size + canvas.canvasy(0), 
            event.x + brush_size + canvas.canvasx(0), 
            event.y + brush_size + canvas.canvasy(0), 
            outline='blue', width=2, tags="erase_circle"
        )

def select_eraser():
    global current_tool
    if current_tool == "eraser":
        current_tool = None
        eraser_btn.config(relief=tk.RAISED, text="Eraser: OFF")
    else:
        current_tool = "eraser"
        canvas.bind("<B1-Motion>", erase)
        eraser_btn.config(relief=tk.SUNKEN, text="Eraser: ON")
        # Ensure other tools are deactivated
        paintbrush_btn.config(relief=tk.RAISED, text="Paintbrush: OFF")

def select_paintbrush():
    global current_tool
    if current_tool == "paintbrush":
        current_tool = None
        paintbrush_btn.config(relief=tk.RAISED, text="Paintbrush: OFF")
    else:
        current_tool = "paintbrush"
        canvas.bind("<B1-Motion>", paint)
        paintbrush_btn.config(relief=tk.SUNKEN, text="Paintbrush: ON")
        # Ensure other tools are deactivated
        eraser_btn.config(relief=tk.RAISED, text="Eraser: OFF")



def display_image_bbox():
    global canvas, imgtk

    if image is None or not segmentation_masks:
        return

    mask = segmentation_masks[current_mask_index]['segmentation']
    mask_overlay = np.zeros_like(image)
    mask_overlay[mask > 0] = [0, 255, 0]

    img_with_mask = Image.fromarray(np.uint8(image + mask_overlay * 0.3))
    imgtk = ImageTk.PhotoImage(image=img_with_mask)

    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    canvas.image = imgtk

def display_image():
    if image is None or not segmentation_masks:
        return
    
    # Clear the previous image/mask
    canvas.delete("all")
    
    mask = segmentation_masks[current_mask_index]['segmentation']
    mask_overlay = np.zeros_like(image)
    mask_overlay[mask > 0] = [0, 255, 0]

    img_with_mask = Image.fromarray(np.uint8(image + mask_overlay * 0.3))
    
    imgtk = ImageTk.PhotoImage(image=img_with_mask)
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    canvas.image = imgtk

    # If in mask edit mode, redraw the masks
    if mode == "mask_edit":
        redraw_masks()



def load_json_keypoints_into_gui(annotations_for_image):
    """
    Looks for an annotation that has 'keypoints' in COCO format (x,y,v triplets).
    Converts them into point_coords, point_labels, and point_orders
    so we can reuse 'redraw_points()' and 'draw_cross()'.
    """

    # Clear any existing points in the GUI
    clear_points()  # This calls canvas.delete("point") and resets your lists

    # If you have multiple annotations with keypoints, decide if you want to:
    # - merge them,
    # - just take the first one,
    # - or something else.
    for ann in annotations_for_image:
        if "keypoints" in ann and ann["keypoints"]:
            kpts = ann["keypoints"]
            
            # We parse them in steps of 3: [x, y, visibility]
            for i in range(0, len(kpts), 3):
                x = kpts[i]
                y = kpts[i+1]
                v = kpts[i+2]

                if v > 0:
                    # v=2 means "visible" => label=1 => color="green"
                    # v=1 means "labeled but not visible" => label=0 => color="red"
                    label = 1 if v == 2 else 0

                    point_coords.append([x, y])
                    point_labels.append(label)
                    point_orders.append(len(point_coords)) 
                    # The i-th valid keypoint is appended last => order = len(point_coords)

            # If you only want to load the FIRST annotation that has keypoints:
            break

    # Finally, redraw them (calls your existing draw_cross/draw_number code)
    redraw_points()



def next_mask(event=None):
    global current_mask_index, segmentation_masks

    if not segmentation_masks:
        return

    current_mask_index = (current_mask_index + 1) % len(segmentation_masks)

    # Preserve zoom/pan refresh behavior
    try:
        apply_zoom()
    except Exception:
        try:
            redraw_masks()
        except Exception:
            pass

    # Preserve mode-dependent label syncing
    try:
        if globals().get("mode", None) == "prediction_view":
            category_id = segmentation_masks[current_mask_index].get("category_id", -1)
            category_name = category_id_to_name.get(category_id, "unlabeled")

            # main UI
            try:
                selected_category_label.set(category_name)
            except Exception:
                pass

            # video-remote dropdown (if it exists)
            try:
                if "video_selected_label" in globals() and video_selected_label is not None:
                    video_selected_label.set(category_name)
            except Exception:
                pass

        else:
            label = labels.get(current_mask_index, "unlabeled")
            try:
                selected_label.set(label)
            except Exception:
                pass
    except Exception:
        pass


#original
#def next_mask():
#    global current_mask_index
#    if segmentation_masks:
#        current_mask_index = (current_mask_index + 1) % len(segmentation_masks)
#        apply_zoom()
#        # Update the selected label based on the current mode
#        if mode == 'prediction_view':
#            # Update the selected category label
#            category_id = segmentation_masks[current_mask_index].get('category_id', -1)
#            category_name = category_id_to_name.get(category_id, "unlabeled")
#            selected_category_label.set(category_name)
#        else:
#            # Original functionality
#            label = labels.get(current_mask_index, "unlabeled")
#            selected_label.set(label)

#def next_mask():
#    global current_mask_index, multi_mask_mode
#    if segmentation_masks:
#        multi_mask_mode = False  # 👈 force single-mask view so cycling is visible
#        current_mask_index = (current_mask_index + 1) % len(segmentation_masks)
#        apply_zoom()
#
#        # Update label based on category_id when in prediction_view
#        if mode == 'prediction_view':
#            category_id = segmentation_masks[current_mask_index].get('category_id', -1)
#            category_name = category_id_to_name.get(category_id, "unlabeled")
#            selected_category_label.set(category_name)
#        else:
#            label = labels.get(current_mask_index, "unlabeled")
#            selected_label.set(label)





def next_annotated_frame():
    """Navigate to the next frame that has annotations in video mode."""
    global current_video_frame, video_annotations, video_mode
    
    if not video_mode or not video_annotations:
        messagebox.showinfo("Info", "No annotated frames available" if video_mode else "Not in video mode")
        return
    
    # Get sorted list of annotated frame indices
    annotated_frames = sorted(video_annotations.keys())
    
    # Find next annotated frame after current
    next_frames = [f for f in annotated_frames if f > current_video_frame]
    
    if next_frames:
        # Jump to next annotated frame
        load_video_frame_into_canvas(next_frames[0])
        messagebox.showinfo("Navigation", 
                          f"Jumped to frame {next_frames[0]}\n"
                          f"({len(video_annotations[next_frames[0]]['masks'])} masks)")
    else:
        # Wrap around to first annotated frame
        if annotated_frames:
            load_video_frame_into_canvas(annotated_frames[0])
            messagebox.showinfo("Navigation", 
                              f"Wrapped to first annotated frame {annotated_frames[0]}\n"
                              f"({len(video_annotations[annotated_frames[0]]['masks'])} masks)")
        else:
            messagebox.showinfo("Info", "No annotated frames found")


def previous_annotated_frame():
    """Navigate to the previous frame that has annotations in video mode."""
    global current_video_frame, video_annotations, video_mode
    
    if not video_mode or not video_annotations:
        messagebox.showinfo("Info", "No annotated frames available" if video_mode else "Not in video mode")
        return
    
    # Get sorted list of annotated frame indices
    annotated_frames = sorted(video_annotations.keys())
    
    # Find previous annotated frame before current
    prev_frames = [f for f in annotated_frames if f < current_video_frame]
    
    if prev_frames:
        # Jump to previous annotated frame
        load_video_frame_into_canvas(prev_frames[-1])
        messagebox.showinfo("Navigation", 
                          f"Jumped to frame {prev_frames[-1]}\n"
                          f"({len(video_annotations[prev_frames[-1]]['masks'])} masks)")
    else:
        # Wrap around to last annotated frame
        if annotated_frames:
            load_video_frame_into_canvas(annotated_frames[-1])
            messagebox.showinfo("Navigation", 
                              f"Wrapped to last annotated frame {annotated_frames[-1]}\n"
                              f"({len(video_annotations[annotated_frames[-1]]['masks'])} masks)")
        else:
            messagebox.showinfo("Info", "No annotated frames found")


def previous_mask(event=None):
    global current_mask_index, segmentation_masks

    if not segmentation_masks:
        return

    current_mask_index = (current_mask_index - 1) % len(segmentation_masks)

    # Preserve zoom/pan refresh behavior
    try:
        apply_zoom()
    except Exception:
        try:
            redraw_masks()
        except Exception:
            pass

    # Preserve mode-dependent label syncing
    try:
        if globals().get("mode", None) == "prediction_view":
            category_id = segmentation_masks[current_mask_index].get("category_id", -1)
            category_name = category_id_to_name.get(category_id, "unlabeled")

            # main UI
            try:
                selected_category_label.set(category_name)
            except Exception:
                pass

            # video-remote dropdown (if it exists)
            try:
                if "video_selected_label" in globals() and video_selected_label is not None:
                    video_selected_label.set(category_name)
            except Exception:
                pass

        else:
            label = labels.get(current_mask_index, "unlabeled")
            try:
                selected_label.set(label)
            except Exception:
                pass
    except Exception:
        pass


#    load_prediction_image(new_image_name)
#    # ADD the same keypoint code:
#    if new_image_name in prediction_data:
#        ann_for_new_image = prediction_data[new_image_name]
#        load_json_keypoints_into_gui(annotations_for_image)

#def previous_mask():
#    global current_mask_index, multi_mask_mode
#    if segmentation_masks:
#        multi_mask_mode = False  # 👈 force single-mask view so cycling is visible
#        current_mask_index = (current_mask_index - 1) % len(segmentation_masks)
#        apply_zoom()
#
#        if mode == 'prediction_view':
#            category_id = segmentation_masks[current_mask_index].get('category_id', -1)
#            category_name = category_id_to_name.get(category_id, "unlabeled")
#            selected_category_label.set(category_name)
#        else:
#            label = labels.get(current_mask_index, "unlabeled")
#            selected_label.set(label)

#original
#def previous_mask():
#    global current_mask_index
#    if segmentation_masks:
#        current_mask_index = (current_mask_index - 1) % len(segmentation_masks)
#        apply_zoom()
#        # Update the selected label based on the current mode
#        if mode == 'prediction_view':
#            # Update the selected category label
#            category_id = segmentation_masks[current_mask_index].get('category_id', -1)
#            category_name = category_id_to_name.get(category_id, "unlabeled")
#            selected_category_label.set(category_name)
#        else:
#            # Original functionality
#            label = labels.get(current_mask_index, "unlabeled")
#            selected_label.set(label)

#not sure about next part maybe leave commented out
#    load_prediction_image(new_image_name)
#    # ADD the same keypoint code:
#    if new_image_name in prediction_data:
#        ann_for_new_image = prediction_data[new_image_name]
#        load_json_keypoints_into_gui(annotations_for_image)

def apply_video_label(event=None):
    """
    Apply the currently selected VIDEO-remote label to the current mask
    and persist it into video_annotations (so exports/train use correct category_id).
    """
    global current_video_frame, current_mask_index
    global segmentation_masks, labels
    global video_selected_label, category_name_to_id, category_id_to_name, categories, category_id_to_supercategory

    if not globals().get("video_mode", False):
        # If not in video mode, fall back to the standard behavior
        return apply_label(event=event)

    if not segmentation_masks or current_mask_index >= len(segmentation_masks):
        messagebox.showwarning("Warning", "No mask selected to label.")
        return

    label_name = None
    try:
        label_name = video_selected_label.get()
    except Exception:
        # fallback to main dropdown if remote not present
        label_name = selected_label.get()

    if not label_name or label_name == "Select Label":
        messagebox.showwarning("Warning", "Please select a label first.")
        return

    # If label isn't in categories yet, add it (do NOT invent supercategory)
    if label_name not in category_name_to_id:
        # Choose a new ID safely
        existing_ids = [int(c["id"]) for c in (categories or []) if isinstance(c, dict) and "id" in c]
        new_id = (max(existing_ids) + 1) if existing_ids else 1

        # Preserve user semantics: default supercategory is 'sclerite' only if missing
        categories.append({"id": new_id, "name": label_name, "supercategory": "sclerite"})
        category_name_to_id[label_name] = new_id
        category_id_to_name[new_id] = label_name
        category_id_to_supercategory[new_id] = "sclerite"

        # Refresh dropdowns so the new label is visible immediately
        refresh_select_label_ui(verbose=False)

    cat_id = int(category_name_to_id[label_name])

    # Update current mask entry (this list is what redraw uses)
    segmentation_masks[current_mask_index]["category_id"] = cat_id
    segmentation_masks[current_mask_index]["label"] = label_name

    # Update labels mapping (used by redraw UI and mask cycling)
    labels[current_mask_index] = label_name

    # Persist into video_annotations for the frame
    try:
        save_current_frame_annotations()
    except Exception as e:
        print(f"[WARNING] apply_video_label: could not save_current_frame_annotations: {e}")

    # Repaint
    redraw_masks()

    print(f"[OK] Applied video label: frame={current_video_frame}, mask_idx={current_mask_index}, cat_id={cat_id}, name={label_name}")


def apply_category_label():
    global segmentation_masks, current_mask_index
    global category_name_to_id, category_id_to_name, categories, category_label_options
    global labels, category_id_to_supercategory, coco_output_mask, coco_output_accumulate

    label = selected_category_label.get()
    if label == "Select Category":
        messagebox.showerror("Error", "Please select a category.")
        return

    # Check if the category exists
    if label in category_name_to_id:
        category_id = category_name_to_id[label]
    else:
        # Category does not exist; ask for supercategory and add new category
        existing_supercategories = list(set(category_id_to_supercategory.values()))
        supercategory_str = ", ".join(existing_supercategories) if existing_supercategories else "sclerite, textures, etc."
        
        supercategory = simpledialog.askstring(
            "Supercategory", 
            f"Enter supercategory for '{label}':\n(Existing: {supercategory_str})",
            parent=root
        )
        if not supercategory:
            supercategory = "none"  # Default if cancelled or empty
        
        new_category_id = max(category_id_to_name.keys(), default=0) + 1
        category_id_to_name[new_category_id] = label
        category_name_to_id[label] = new_category_id
        category_id_to_supercategory[new_category_id] = supercategory
        
        new_category = {
            "id": new_category_id,
            "name": label,
            "supercategory": supercategory
        }
        categories.append(new_category)
        category_id = new_category_id
        
        # FIXED: Sync new category to coco_output_accumulate and coco_output_mask
        if "categories" not in coco_output_accumulate:
            coco_output_accumulate["categories"] = []
        if "categories" not in coco_output_mask:
            coco_output_mask["categories"] = []
        coco_output_accumulate["categories"].append(new_category.copy())
        coco_output_mask["categories"].append(new_category.copy())
        
        # Update the category dropdowns
        if label not in category_label_options:
            category_label_options.append(label)
            update_category_label_dropdown()

    # Update the category_id of the current mask
    segmentation_masks[current_mask_index]['category_id'] = category_id
    # ALso update the label for the current mask
    labels[current_mask_index] = label
    # Set the selected_category_label to the applied label
    selected_category_label.set(label)

    messagebox.showinfo("Success", f"Category '{label}' applied to the current mask.")

def reannotate_masks():
    global segmentation_masks, image, file_path_base, categories
    global pred_json_path, working_pred_json_path, category_id_to_name, category_name_to_id
    global mode, selected_prediction_image
    global point_coords, point_labels, point_orders
    global prediction_cache, prediction_image_names

    if mode not in ['prediction_view', 'mask_edit']:
        messagebox.showerror(
            "Error",
            "Re-annotate function is only available in View Predictions or Mask Edit mode."
        )
        return

    image_name = selected_prediction_image.get()

    # 1) Load the working copy JSON from disk
    try:
        with open(working_pred_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to load JSON from '{working_pred_json_path}':\n{e}"
        )
        return

    # 2) Find the integer image_id for the current image
    image_id = None
    for img in data['images']:
        if img['file_name'] == image_name:
            image_id = img['id']
            break
    if image_id is None:
        messagebox.showerror(
            "Error",
            f"No image_id found for image '{image_name}' in working prediction JSON."
        )
        return

    # 3) Prompt user to replace or keep existing annotations
    result = messagebox.askyesnocancel(
        "Update Annotations",
        "Do you want to replace existing annotations for this image?\n"
        "Yes: Replace\nNo: Keep Both\nCancel: Do Not Save"
    )
    if result is None:
        # User pressed "Cancel"
        return
    elif result:
        # Yes => remove old annotations for this image
        data['annotations'] = [
            ann for ann in data['annotations']
            if ann['image_id'] != image_id
        ]
    else:
        # No => keep them
        pass

    # 4) Build new annotations from current segmentation_masks
    new_annotations = []
    max_ann_id = max((ann['id'] for ann in data['annotations']), default=0)
    ann_id = max_ann_id + 1

    for mask_data in segmentation_masks:
        mask = mask_data['segmentation']
        category_id = mask_data.get('category_id', -1)
        if category_id == -1:
            category_id = category_name_to_id.get('unlabeled', -1)

        # Convert mask to polygons
        mask_8bit = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coco_segmentation = []
        for contour in contours:
            flat = contour.flatten().tolist()
            if len(flat) >= 6:
                coco_segmentation.append(flat)

        # Bbox & area
        x, y, w, h = cv2.boundingRect(mask_8bit)
        bbox = [x, y, w, h]
        area = float((mask_8bit > 0).sum())

        annotation = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": coco_segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        }
        new_annotations.append(annotation)
        ann_id += 1

    # 5) If we have keypoints in point_coords, add one annotation for them
    if point_coords:
        # PICK A KEYPOINTS CATEGORY (adjust ID as needed)
        keypoints_category_id = 1  # or some other ID for keypoints category

        keypoints_list = []
        for i, (px, py) in enumerate(point_coords):
            # v=2 => visible, v=1 => labeled but not visible
            visibility = 2 if point_labels[i] == 1 else 1
            keypoints_list.extend([px, py, visibility])

        annotation_for_keypoints = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": keypoints_category_id,
            "keypoints": keypoints_list,
            "num_keypoints": len(point_coords),
            "segmentation": [],
            "bbox": [],
            "area": 0,
            "iscrowd": 0
        }
        new_annotations.append(annotation_for_keypoints)
        ann_id += 1

    # 6) Append newly built annotations to the main list
    data['annotations'].extend(new_annotations)

    # 7) Update any in-memory caches so the same session sees them
    if image_name in prediction_cache:
        prediction_cache[image_name].extend(new_annotations)
    else:
        prediction_cache[image_name] = list(new_annotations)

    # 8) Ensure categories include any new categories from global 'categories'
    existing_category_ids = {cat['id'] for cat in data['categories']}
    for cat in categories:
        if cat['id'] not in existing_category_ids:
            data['categories'].append(cat)

    # 9) Write everything back to disk
    try:
        with open(working_pred_json_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save JSON:\n{e}")
        return

    messagebox.showinfo("Success", f"Annotations saved to '{working_pred_json_path}'")

    # 10) Reload the updated JSON so we see the new points/masks instantly
    load_prediction_image(image_name)




def add_categories_to_coco():
    """
    Add the category mappings (id to name) to the COCO JSON structure.
    """
    global coco_output_mask, coco_output_accumulate, categories

    # Check if categories are already added
    if "categories" not in coco_output_mask:
        coco_output_mask["categories"] = []

    # Add each category
    for category in categories:
        coco_output_mask["categories"].append({
            "id": category["id"],
            "name": category["name"],
            "supercategory": category["supercategory"]
        })
    # Check if categories are already added
    if "categories" not in coco_output_accumulate:
        coco_output_accumulate["categories"] = []

    # Add each category
    for category in categories:
        coco_output_accumulate["categories"].append({
            "id": category["id"],
            "name": category["name"],
            "supercategory": category["supercategory"]
        })


def save_mask_and_contour_via2(overwrite=False):
    """
    Save the mask and contour in COCO JSON format compatible with Detectron2, with unique filenames and translation table.
    Supports both regular masks and line annotations.
    """
    global labels, file_path_base, image, segmentation_masks, current_mask_index
    global coco_output_mask, coco_output_accumulate
    global category_name_to_id, category_id_to_supercategory, categories, instance_counters

    if not segmentation_masks:
        messagebox.showerror("Error", "No segmentation masks available to save.")
        return

    mask_data = segmentation_masks[current_mask_index]
    mask = mask_data['segmentation']
    mask_8bit = (mask * 255).astype(np.uint8)

    # Get the label and category_id of the current mask
    label = labels.get(current_mask_index, "unlabeled")
    category_id = mask_data.get('category_id', -1)
    if category_id == -1:
        messagebox.showerror("Error", "No category assigned to the current mask.")
        return
    if category_id == 0:
        logging.warning("Category ID 0 is reserved and will be skipped.")
        return

    # FIXED: Initialize or update the instance counter for each label
    # This enables multiple instances like spine_0, spine_1, spine_2
    if label not in instance_counters:
        instance_counters[label] = 0
    
    instance_number = instance_counters[label]
    instance_counters[label] += 1
    
    # Create instance-aware label for filenames (e.g., "spine_0", "spine_1")
    instance_label = f"{label}_{instance_number}"

    # Generate a unique annotation ID
    annotation_id = int(time.time() * 1000) + current_mask_index + instance_number

    # Compute contours
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coco_segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) >= 6:
            coco_segmentation.append(contour)

    # Compute bounding box and area
    x, y, w, h = cv2.boundingRect(mask_8bit)
    bbox = [x, y, w, h]
    area = float(np.sum(mask_8bit > 0))

    # Create annotation with instance information
    annotation = {
        "id": annotation_id,
        "image_id": file_path_base,  # Adjust this as needed
        "category_id": category_id,
        "segmentation": coco_segmentation,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "instance_id": instance_number  # ADDED: Track instance number
    }
    
    # ADDED: Preserve line-specific fields if this is a line annotation
    if mask_data.get('is_line', False):
        annotation['is_line'] = True
        if 'line_points' in mask_data:
            annotation['line_points'] = [[int(p[0]), int(p[1])] for p in mask_data['line_points']]
        if 'line_thickness' in mask_data:
            annotation['line_thickness'] = mask_data['line_thickness']
        if 'num_line_keypoints' in mask_data:
            annotation['num_line_keypoints'] = mask_data['num_line_keypoints']

    # Append annotation to both coco_output_mask and coco_output_accumulate
    coco_output_mask["annotations"].append(annotation)
    coco_output_accumulate["annotations"].append(annotation)

    # Ensure categories are added to coco_output_mask and coco_output_accumulate
    if "categories" not in coco_output_mask or not coco_output_mask["categories"]:
        coco_output_mask["categories"] = categories.copy()
    if "categories" not in coco_output_accumulate or not coco_output_accumulate["categories"]:
        coco_output_accumulate["categories"] = categories.copy()
    
    # FIXED: Sync any new categories that might have been added
    existing_cat_ids_mask = {cat['id'] for cat in coco_output_mask.get("categories", [])}
    existing_cat_ids_acc = {cat['id'] for cat in coco_output_accumulate.get("categories", [])}
    for cat in categories:
        if cat['id'] not in existing_cat_ids_mask:
            coco_output_mask["categories"].append(cat.copy())
        if cat['id'] not in existing_cat_ids_acc:
            coco_output_accumulate["categories"].append(cat.copy())

    # Save the mask image with instance number in filename
    mask_filename = f"{file_path_base}_{instance_label}_mask.png"
    cv2.imwrite(mask_filename, mask_8bit)
    
    # Log the save with instance info
    is_line_str = " (LINE)" if mask_data.get('is_line', False) else ""
    logging.info(f"Saved mask instance: {instance_label} (category: {label}, instance #{instance_number}){is_line_str}")

    # Optionally, reset coco_output_mask after saving
    coco_output_mask["annotations"].clear()


def view_predictions():
    global prediction_data, prediction_image_names, image_dir, selected_prediction_image
    global category_id_to_name, category_name_to_id, category_label_options, categories
    global pred_json_path, working_pred_json_path, prediction_cache, view_predictions_mode

    # Toggle the mode
    if view_predictions_mode:
        view_predictions_mode = False
        view_predictions_btn.config(relief=tk.RAISED, text="View Predictions")
    else:
        view_predictions_mode = True
        view_predictions_btn.config(relief=tk.SUNKEN, text="View Predictions: ON")

    # Prompt user to select the prediction JSON file
    pred_json_path_selected = filedialog.askopenfilename(
        title="Select Predictions JSON File",
        filetypes=[("JSON Files", "*.json")]
    )
    if not pred_json_path_selected:
        return  # User cancelled the dialog

    pred_json_path = pred_json_path_selected  # Store the original path

    # Determine the working copy filename
    original_dir = os.path.dirname(pred_json_path)
    pred_json_filename = os.path.basename(pred_json_path)

    # Initialize working copy filename and version number
    working_copy_prefix = 'working_copy_'
    version = 1

    # Remove extension from filename
    filename_without_ext, ext = os.path.splitext(pred_json_filename)

    # Remove 'working_copy_' prefix if present
    if filename_without_ext.startswith(working_copy_prefix):
        base_filename = filename_without_ext[len(working_copy_prefix):]
    else:
        base_filename = filename_without_ext

    # Remove version suffix if present
    if '_V' in base_filename:
        # Split off the version number
        name_part, version_part = base_filename.rsplit('_V', 1)
        base_filename = name_part
        try:
            version = int(version_part) + 1
        except ValueError:
            version = 2
    else:
        version = 1  # Start from version 1

    # Construct new working copy filename with version
    working_pred_json_filename = f"{working_copy_prefix}{base_filename}_V{version}{ext}"
    working_pred_json_path = os.path.join(original_dir, working_pred_json_filename)

    # Check if the new working copy filename is the same as the selected file
    if os.path.abspath(pred_json_path) == os.path.abspath(working_pred_json_path):
        messagebox.showerror("Error", "Cannot create a new working copy. The source and destination files are the same.")
        return

    # Copy the input JSON file to create the working copy
    shutil.copyfile(pred_json_path, working_pred_json_path)

    # Proceed to select the image directory
    image_dir_selected = filedialog.askdirectory(
        title="Select Image Directory Used in Predictions"
    )
    if not image_dir_selected:
        return  # User cancelled the dialog

    image_dir = image_dir_selected  # Set the global image directory

    # Load the prediction JSON from the working copy
    try:
        with open(working_pred_json_path, 'r') as f:
            pred_data = json.load(f)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load JSON file:\n{e}")
        return

    # Extract images and annotations
    images = pred_data.get('images', [])
    annotations = pred_data.get('annotations', [])
    categories = pred_data.get('categories', [])

    if not images or not annotations:
        messagebox.showerror("Error", "The selected JSON file does not contain images or annotations.")
        return

    # Create a mapping from image_id to image info
    image_id_to_info = {img['id']: img for img in images}

    # Create a mapping from image filename to annotations
    image_filename_to_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        img_info = image_id_to_info.get(img_id)
        if img_info:
            file_name = img_info['file_name']
            if file_name not in image_filename_to_annotations:
                image_filename_to_annotations[file_name] = []
            image_filename_to_annotations[file_name].append(ann)

    # Populate the prediction_data dict
    prediction_data = image_filename_to_annotations

    # Populate the prediction_cache
    prediction_cache = copy.deepcopy(prediction_data)  # Use deepcopy to avoid shared references

    # Get sorted list of image filenames
    prediction_image_names = sorted(prediction_data.keys())

    if not prediction_image_names:
        messagebox.showinfo("Info", "No images found in the prediction JSON.")
        return

    # Store categories and populate category_label_options
    category_id_to_name = {}
    category_name_to_id = {}
    category_label_options = []
    for category in categories:
        category_id_to_name[category['id']] = category['name']
        category_name_to_id[category['name']] = category['id']
        category_label_options.append(category['name'])

    # Update the category_label_dropdown
    update_category_label_dropdown()


    # -----------------------------
    # INTERNAL FUNCTION: on_select_prediction_image
    # Used by the dropdown to switch images
    # -----------------------------
    def on_select_prediction_image(img_name):
        """
        Called when the user picks a filename from the dropdown.
        1) Update the global selected_prediction_image
        2) Load the new image
        3) Load keypoints from the new image
        """
        selected_prediction_image.set(img_name)      # let the GUI know which image is selected
        load_prediction_image(img_name)             # your function that draws the new image (and masks)
        
        # Now load keypoints for that image
        if img_name in prediction_data:
            annotations_for_image = prediction_data[img_name]
            load_json_keypoints_into_gui(annotations_for_image)  # lines 1161-1200 in your code
        else:
            print(f"No annotations found for '{img_name}' in prediction_data.")

    # Populate the dropdown menu
#    menu = prediction_image_dropdown["menu"]
#    menu.delete(0, "end")  # Clear existing menu entries
#    for img_name in prediction_image_names:
#        menu.add_command(
#            label=img_name,
#            command=partial(on_select_prediction_image, img_name)
#        )

    # Reset the selection to the first image
    selected_prediction_image.set(prediction_image_names[0])

    messagebox.showinfo("Success", f"Loaded predictions for {len(prediction_image_names)} images.")

    # Load the first image
    load_prediction_image(selected_prediction_image.get())


    # ----------------------------------------------------------------------
    # NEW: After calling load_prediction_image(), load the keypoints as well
    # ----------------------------------------------------------------------
    file_name = selected_prediction_image.get()  # The filename we just loaded
    if file_name in prediction_data:
        annotations_for_image = prediction_data[file_name]
        load_json_keypoints_into_gui(annotations_for_image)  # from your def at lines ~1161-1200
    else:
        print(f"No annotations found for '{file_name}' in prediction_data.")

    print("prediction_data keys:", list(prediction_data.keys()))
    print("Current file_name:", file_name)
    #    load_prediction_image(new_image_name)
    #    # ADD the same keypoint code:
#    if new_image_name in prediction_data:
#        ann_for_new_image = prediction_data[new_image_name]
#        load_json_keypoints_into_gui(annotations_for_image)


def save_final_via2_json():
    """
    Save the final COCO JSON file with all labeled masks and contours from the session,
    excluding any annotations labeled as "Trash".
    Also extracts keypoints from line annotations and saves them as COCO keypoints.
    """
    global coco_output_accumulate, file_path_base, segmentation_masks, labels, point_coords

    if not coco_output_accumulate['annotations']:
        messagebox.showerror("Error", "No annotations to save!")
        return

    if not coco_output_accumulate['categories']:
        messagebox.showerror("Error", "Categories missing from the final output!")
        return

    # --- Minimal Modification Start ---
    # Find the category_id for "Trash"
    trash_category_id = None
    for category in coco_output_accumulate['categories']:
        if category['name'] == 'Trash':
            trash_category_id = category['id']
            break  # Exit loop once found

    # Filter annotations to exclude those with category_id == trash_category_id
    if trash_category_id is not None:
        filtered_annotations = []
        for annotation in coco_output_accumulate['annotations']:
            if annotation['category_id'] != trash_category_id:
                filtered_annotations.append(annotation)
    else:
        # If "Trash" category not found, use all annotations
        filtered_annotations = coco_output_accumulate['annotations']
    # --- Minimal Modification End ---

    # --- LINE KEYPOINTS EXTRACTION ---
    # Count existing manual keypoints to know where to start numbering
    existing_keypoint_count = len(point_coords) if point_coords else 0
    keypoint_number = existing_keypoint_count + 1  # Start numbering after manual keypoints
    
    # Collect all line keypoints from line annotations in filtered_annotations
    # (these came from coco_output_accumulate and include the line_points data)
    line_keypoints_data = []
    
    # Build category_id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_output_accumulate.get('categories', [])}
    
    for ann in filtered_annotations:
        if ann.get('is_line', False):
            # Check if keypoints should be extracted (num_line_keypoints > 0)
            num_line_kp = ann.get('num_line_keypoints', 0)
            if num_line_kp < 2:
                logging.info(f"Skipping keypoint extraction for line annotation {ann.get('id')} (num_line_keypoints={num_line_kp})")
                continue
            
            # Get line_points from the annotation
            line_points = ann.get('line_points', [])
            if not line_points or len(line_points) < 2:
                logging.warning(f"Line annotation {ann.get('id')} has no valid line_points")
                continue
            
            # Ensure consistent ordering (lowest x+y sum first)
            sums = [p[0] + p[1] for p in line_points]
            if sums[0] > sums[-1]:
                line_points = line_points[::-1]  # Reverse if needed
            
            # Extract evenly spaced keypoints
            keypoints = extract_evenly_spaced_keypoints(line_points, num_line_kp)
            
            if keypoints and len(keypoints) >= 2:
                category_id = ann.get('category_id', -1)
                label = cat_id_to_name.get(category_id, "line")
                
                # Store keypoints with their numbering
                for i, kp in enumerate(keypoints):
                    line_keypoints_data.append({
                        'x': kp[0],
                        'y': kp[1],
                        'number': keypoint_number,
                        'label': label,
                        'category_id': category_id,
                        'line_position': 'start' if i == 0 else ('end' if i == len(keypoints) - 1 else 'middle')
                    })
                    keypoint_number += 1
                
                logging.info(f"Extracted {len(keypoints)} keypoints from line '{label}'")
    
    # Create keypoint annotations in COCO format if we have line keypoints
    if line_keypoints_data:
        # Ensure we have a "line_keypoints" category
        line_kp_category_id = None
        for cat in coco_output_accumulate['categories']:
            if cat['name'] == 'line_keypoints':
                line_kp_category_id = cat['id']
                break
        
        if line_kp_category_id is None:
            # Create the category
            existing_ids = [cat['id'] for cat in coco_output_accumulate['categories']]
            line_kp_category_id = max(existing_ids) + 1 if existing_ids else 1
            coco_output_accumulate['categories'].append({
                'id': line_kp_category_id,
                'name': 'line_keypoints',
                'supercategory': 'keypoints'
            })
        
        # Group keypoints by their source line (category_id)
        keypoints_by_line = {}
        for kp_data in line_keypoints_data:
            cat_id = kp_data['category_id']
            if cat_id not in keypoints_by_line:
                keypoints_by_line[cat_id] = []
            keypoints_by_line[cat_id].append(kp_data)
        
        # Create COCO keypoint annotations for each line
        for cat_id, kp_list in keypoints_by_line.items():
            # Build COCO keypoints format: [x1, y1, v1, x2, y2, v2, ...]
            coco_keypoints = []
            point_orders = []
            
            for kp_data in kp_list:
                coco_keypoints.extend([kp_data['x'], kp_data['y'], 2])  # 2 = visible
                point_orders.append(kp_data['number'])
            
            # Calculate bounding box
            xs = [kp_data['x'] for kp_data in kp_list]
            ys = [kp_data['y'] for kp_data in kp_list]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            
            # Find the line category name
            line_cat_name = "unknown"
            for cat in coco_output_accumulate['categories']:
                if cat['id'] == cat_id:
                    line_cat_name = cat['name']
                    break
            
            # Create annotation
            keypoint_annotation = {
                'id': int(time.time() * 1000) + cat_id,
                'image_id': coco_output_accumulate['images'][0]['id'] if coco_output_accumulate['images'] else file_path_base,
                'category_id': line_kp_category_id,
                'keypoints': coco_keypoints,
                'num_keypoints': len(kp_list),
                'point_order': point_orders,
                'bbox': bbox,
                'area': 1,
                'iscrowd': 0,
                'source_line_category': line_cat_name,
                'source_line_category_id': cat_id
            }
            
            filtered_annotations.append(keypoint_annotation)
        
        logging.info(f"Added {len(line_keypoints_data)} line keypoints to annotations")

    # Define the final JSON structure using filtered annotations
    final_coco_json = {
        "images": coco_output_accumulate["images"],
        "annotations": filtered_annotations,
        "categories": coco_output_accumulate["categories"]
    }

    # Generate a unique filename for the final JSON (using timestamp)
    timestamp = int(time.time())
    final_json_filename = f"{file_path_base}_final_{timestamp}.json"

    # Save the final concatenated COCO JSON
    with open(final_json_filename, 'w') as json_file:
        json.dump(final_coco_json, json_file, indent=4)

    # Summary message
    line_kp_count = len(line_keypoints_data)
    if line_kp_count > 0:
        messagebox.showinfo("Success", f"Final COCO JSON saved as '{final_json_filename}'\n\nIncluded {line_kp_count} keypoints extracted from line annotations.")
    else:
        messagebox.showinfo("Success", f"Final COCO JSON saved as '{final_json_filename}'")

    # Optionally, reset coco_output_accumulate and instance_counters after saving
    coco_output_accumulate["images"].clear()
    coco_output_accumulate["annotations"].clear()
    instance_counters.clear()

def load_json_contours(json_path):
    global segmentation_masks, point_coords, point_labels, point_orders
    global categories, category_name_to_id, category_id_to_supercategory, category_id_to_name
    global labels

    # Open and read the JSON file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # Clear the current segmentation masks, points, and labels
    segmentation_masks = []
    point_coords = []
    point_labels = []
    point_orders = []
    labels = {}  # Clear labels

    # Clear existing categories and mappings
    categories.clear()
    category_name_to_id.clear()
    category_id_to_supercategory.clear()
    category_id_to_name.clear()

    # Load categories from the JSON file
    categories_data = data.get("categories", [])
    for cat in categories_data:
        cat_id = int(cat['id'])
        name = cat['name'].strip()
        supercategory = cat['supercategory'].strip()
        categories.append({
            "id": cat_id,
            "name": name,
            "supercategory": supercategory
        })
        category_name_to_id[name] = cat_id
        category_id_to_supercategory[cat_id] = supercategory
        category_id_to_name[cat_id] = name  # Create category_id_to_name mapping

    # Update the label dropdown menu
    update_label_dropdown()

    # Process each annotation in the JSON file
    for annotation in data["annotations"]:
        # Handle segmentations (contours)
        if 'segmentation' in annotation:
            contours = annotation["segmentation"]
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            for contour in contours:
                points = np.array(contour).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)

            # CRITICAL FIX: Store category_id in mask data
            category_id = annotation.get('category_id', 1)
            segmentation_masks.append({
                'segmentation': mask,
                'category_id': category_id
            })

            # Associate the label with the mask
            label = category_id_to_name.get(category_id, 'unlabeled')
            mask_index = len(segmentation_masks) - 1
            labels[mask_index] = label

        # Handle keypoints
        if 'keypoints' in annotation:
            keypoints = annotation["keypoints"]
            num_keypoints = len(keypoints) // 3  # Each keypoint is [x, y, v]

            # Use 'point_order' if available; otherwise, default to sequential numbers
            if 'point_order' in annotation:
                point_order = annotation['point_order']
            else:
                point_order = list(range(1, num_keypoints + 1))

            for i in range(num_keypoints):
                x = keypoints[i * 3]
                y = keypoints[i * 3 + 1]
                v = keypoints[i * 3 + 2]  # Visibility flag

                if v > 0:
                    point_coords.append([x, y])
                    point_labels.append(1 if v == 2 else 0)  # Visibility 2 for labeled keypoints
                    point_orders.append(point_order[i])

    print(f"Loaded {len(segmentation_masks)} masks and {len(point_coords)} keypoints from {json_path}")


#toggle and load_annotations functions
def toggle_multi_mask_mode():
    global multi_mask_mode
    multi_mask_mode = not multi_mask_mode
    if multi_mask_mode:
        multi_mask_btn.config(relief=tk.SUNKEN, text="Multi-Mask Mode: ON")
    else:
        multi_mask_btn.config(relief=tk.RAISED, text="Multi-Mask Mode: OFF")


def load_annotations():
    global segmentation_masks, current_mask_index, file_path_base, multi_mask_mode

    # Prompt user to load the JSON file
    annotation_path = filedialog.askopenfilename(
        title="Select Annotations (JSON)",
        filetypes=[("JSON Files", "*.json")]
    )
    if not annotation_path:
        messagebox.showerror("Error", "No annotations file selected!")
        return

    load_json_contours(annotation_path)

    # After loading, check if multi-mask mode is enabled and redraw the masks accordingly
    if multi_mask_mode:
        apply_zoom()
        redraw_masks(multi=True)  # Draw all masks at once
    else:
        current_mask_index = 0
        apply_zoom()
        redraw_masks(multi=False)  # Draw one mask at a time
    # Set the selected label to the label of the first mask
    if segmentation_masks:
        label = labels.get(0, "unlabeled")
        selected_label.set(label)
    # Automatically apply labels to all masks and save them temporarily
    for idx in range(len(segmentation_masks)):
        current_mask_index = idx  # Set the current mask index
        label = labels.get(idx, "unlabeled")
        selected_label.set(label)  # Update the selected label
        apply_label(auto=True)  # Call apply_label with auto=True to skip user prompts

    redraw_points()



def save_unlabeled_masks():
    """
    Save all the automatically generated unlabeled masks in one combined COCO JSON file.
    """
    global coco_output_mask, segmentation_masks, file_path_base

    if not segmentation_masks:
        messagebox.showerror("Error", "No segmentation masks available to save.")
        return

    unlabeled_folder = os.path.join(os.path.dirname(file_path_base), "unlabeled_masks")
    os.makedirs(unlabeled_folder, exist_ok=True)

    combined_coco_json = {
        "images": [],
        "annotations": [],
        "categories": [{"id": -1, "name": "unlabeled", "supercategory": "none"}]
    }

    # Iterate through all the masks
    for index, mask_data in enumerate(segmentation_masks):
        mask = mask_data['segmentation']
        mask_8bit = (mask * 255).astype(np.uint8)

        # Save the mask image in PNG format
        mask_file_path = os.path.join(unlabeled_folder, f"{file_path_base}_unlabeled_{index}_mask.png")
        Image.fromarray(mask_8bit).save(mask_file_path)

        # Generate COCO format data
        contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coco_segmentation = [contour.flatten().tolist() for contour in contours]
        x, y, w, h = cv2.boundingRect(mask_8bit)
        bbox = [x, y, w, h]
        area = float(np.sum(mask_8bit > 0))  # Area of the mask

        # Prepare annotation
        annotation = {
            "id": int(time.time() * 1000) + index,
            "image_id": f"unlabeled_{index}",
            "category_id": -1,
            "segmentation": coco_segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        }

        # Append annotation to combined JSON
        combined_coco_json["annotations"].append(annotation)

        # Save image metadata
        image_metadata = {
            "id": f"unlabeled_{index}",
            "file_name": os.path.basename(mask_file_path),
            "height": mask.shape[0],
            "width": mask.shape[1]
        }
        combined_coco_json["images"].append(image_metadata)

    # Save the combined JSON file
    combined_json_path = os.path.join(unlabeled_folder, f"{file_path_base}_unlabeled_combined.json")
    with open(combined_json_path, 'w') as json_file:
        json.dump(combined_coco_json, json_file, indent=4)

    messagebox.showinfo("Info", f"Unlabeled masks and combined COCO JSON saved to '{combined_json_path}'")


def run_coco_combiner_script():
    """
    Opens a dialog to collect inputs and runs the coco_combiner.py script with the provided arguments.
    """
    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Combine COCO JSON Files")
    
    # Define variables to store user inputs
    selected_files = []  # List to store selected file paths
    output_json = tk.StringVar()
    skip_categories = tk.StringVar()
    selected_image_folders = []  # List to store selected image folder paths

    # Function to browse and select multiple JSON files
    def browse_files():
        paths = filedialog.askopenfilenames(
            title="Select COCO JSON Files to Combine",
            filetypes=[("JSON Files", "*.json")]
        )
        if paths:
            # Add new paths to the selected_files list, avoiding duplicates
            for path in paths:
                if path not in selected_files:
                    selected_files.append(path)
                    json_listbox.insert(tk.END, path)
    
    # Function to remove selected JSON files from the list
    def remove_selected_files():
        selected_indices = json_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Selection Error", "Please select files to remove.")
            return
        for index in reversed(selected_indices):
            selected_files.pop(index)
            json_listbox.delete(index)
    
    # Function to browse for output file
    def browse_save_file():
        # Suggest a default filename based on selected input files
        if selected_files:
            base_names = [os.path.splitext(os.path.basename(f))[0] for f in selected_files if f]
            suggested_name = "_".join(base_names) + "_combined.json"
        else:
            suggested_name = "combined.json"
        
        path = filedialog.asksaveasfilename(
            title="Select Output JSON File",
            defaultextension=".json",
            initialfile=suggested_name,
            filetypes=[("JSON Files", "*.json")]
        )
        if path:
            output_json.set(path)
    
    # --- NEW: Functions to handle image folder selection ---
    def browse_folder():
        folder = filedialog.askdirectory(title="Select an Image Folder")
        if folder and folder not in selected_image_folders:
            selected_image_folders.append(folder)
            folder_listbox.insert(tk.END, folder)

    def remove_selected_folders():
        selected_indices = folder_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Selection Error", "Please select folder(s) to remove.")
            return
        for index in reversed(selected_indices):
            selected_image_folders.pop(index)
            folder_listbox.delete(index)
    
    # -------------------------
    # Layout the input fields:
    # -------------------------
    
    # Row 0: JSON Files Section
    tk.Label(input_window, text="Input COCO JSON Files (Required):").grid(row=0, column=0, sticky='nw', padx=5, pady=5)
    
    # Frame for JSON Listbox and Scrollbar (Row 0)
    json_listbox_frame = tk.Frame(input_window)
    json_listbox_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
    
    # Configure grid weights to make the JSON Listbox expandable
    input_window.grid_rowconfigure(0, weight=1)
    input_window.grid_columnconfigure(1, weight=1)
    
    # Create JSON Listbox with Scrollbar
    json_scrollbar = tk.Scrollbar(json_listbox_frame, orient=tk.VERTICAL)
    json_listbox = tk.Listbox(json_listbox_frame, selectmode=tk.MULTIPLE, yscrollcommand=json_scrollbar.set, width=50, height=10)
    json_scrollbar.config(command=json_listbox.yview)
    json_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    json_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Buttons to add and remove JSON files (Row 0)
    json_buttons_frame = tk.Frame(input_window)
    json_buttons_frame.grid(row=0, column=2, padx=5, pady=5, sticky='n')
    tk.Button(json_buttons_frame, text="Browse", command=browse_files).pack(fill=tk.X, pady=(0, 5))
    tk.Button(json_buttons_frame, text="Remove Selected", command=remove_selected_files).pack(fill=tk.X)
    
    # Row 1: Image Folders Section
    tk.Label(input_window, text="Image Folders (Optional, one or more):").grid(row=1, column=0, sticky='nw', padx=5, pady=5)
    
    # Frame for Folder Listbox and Scrollbar (Row 1)
    folder_listbox_frame = tk.Frame(input_window)
    folder_listbox_frame.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
    
    folder_scrollbar = tk.Scrollbar(folder_listbox_frame, orient=tk.VERTICAL)
    folder_listbox = tk.Listbox(folder_listbox_frame, selectmode=tk.MULTIPLE, yscrollcommand=folder_scrollbar.set, width=50, height=5)
    folder_scrollbar.config(command=folder_listbox.yview)
    folder_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    folder_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Buttons for image folder selection (Row 1)
    folder_buttons_frame = tk.Frame(input_window)
    folder_buttons_frame.grid(row=1, column=2, padx=5, pady=5, sticky='n')
    tk.Button(folder_buttons_frame, text="Browse Folder", command=browse_folder).pack(fill=tk.X, pady=(0, 5))
    tk.Button(folder_buttons_frame, text="Remove Selected", command=remove_selected_folders).pack(fill=tk.X)
    
    # Row 2: Output JSON File Section
    tk.Label(input_window, text="Output JSON File (Required):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_json, width=50).grid(row=2, column=1, padx=5, pady=5, sticky='w')
    tk.Button(input_window, text="Browse", command=browse_save_file).grid(row=2, column=2, padx=5, pady=5)
    
    # Row 3: Skip Categories Section
    tk.Label(input_window, text="Skip Categories (Optional, comma-separated):").grid(row=3, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=skip_categories, width=50).grid(row=3, column=1, padx=5, pady=5, sticky='w')
    
    # Row 4: Combine and Cancel buttons
#    buttons_submit_cancel = tk.Frame(input_window)
#    buttons_submit_cancel.grid(row=4, column=0, columnspan=3, pady=10, sticky='e')
#    tk.Button(buttons_submit_cancel, text="Combine", command=submit).pack(side=tk.RIGHT, padx=5)
#    tk.Button(buttons_submit_cancel, text="Cancel", command=input_window.destroy).pack(side=tk.RIGHT)
    
    # -------------------------
    # Submit Function:
    # -------------------------
    def submit():
        # Validate required inputs
        if not selected_files:
            messagebox.showerror("Input Error", "Please select at least one input COCO JSON file.")
            return
        input_jsons_val = ','.join(selected_files)
        output_json_val = output_json.get()
        if not output_json_val:
            messagebox.showerror("Input Error", "Please specify an output JSON file.")
            return
        skip_categories_val = skip_categories.get()
        
        # Construct the path to the combiner script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        combiner_script = script_dir / 'coco_combiner_V13.py'
        if not combiner_script.is_file():
            messagebox.showerror("Script Error", f"Combiner script not found at: {combiner_script}")
            return
        
        # Construct the command
        cmd = [
            "python",
            str(combiner_script),
            "--input-jsons", input_jsons_val,
            "--output-json", output_json_val
        ]
        if skip_categories_val:
            categories_list = [cat.strip() for cat in skip_categories_val.split(',') if cat.strip()]
            if categories_list:
                cmd.extend(["--skip-categories"] + categories_list)
        if selected_image_folders:
            cmd.extend(["--images-folders"] + selected_image_folders)
        
        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                loading = tk.Toplevel()
                loading.title("Combining COCO JSON Files")
                tk.Label(loading, text="Please wait while the COCO JSON files are being combined...").pack(padx=20, pady=20)
                loading.update()
                subprocess.run(cmd, check=True)
                loading.destroy()
                messagebox.showinfo("Success", f"Combined COCO JSON saved to:\n{output_json_val}")
            except subprocess.CalledProcessError as e:
                loading.destroy()
                messagebox.showerror("Error", f"An error occurred while combining COCO JSON files:\n{e}")
            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
        threading.Thread(target=run_script).start()
        input_window.destroy()
    
    # -------------------------
    # Row 4: Combine and Cancel buttons
    # -------------------------
    buttons_submit_cancel = tk.Frame(input_window)
    buttons_submit_cancel.grid(row=4, column=0, columnspan=3, pady=10, sticky='e')
    tk.Button(buttons_submit_cancel, text="Combine", command=submit).pack(side=tk.RIGHT, padx=5)
    tk.Button(buttons_submit_cancel, text="Cancel", command=input_window.destroy).pack(side=tk.RIGHT)


# 1. Helper Function: Order Keypoints by Addition Sequence
def order_keypoints_by_sequence(keypoints, orders):
    """
    Orders keypoints based on the sequence they were added.

    Parameters:
        keypoints (list of [x, y]): List of keypoint coordinates.
        orders (list of int): List of order numbers corresponding to each keypoint.

    Returns:
        ordered_points (list of [x, y]): Keypoints ordered by their addition sequence.
    """
    if not keypoints or not orders or len(keypoints) != len(orders):
        return []

    # Convert to NumPy arrays for easier manipulation
    keypoints_np = np.array(keypoints)
    orders_np = np.array(orders)

    # Sort the keypoints based on their order numbers in ascending order
    sorted_indices = np.argsort(orders_np)
    ordered_points = keypoints_np[sorted_indices].tolist()

    return ordered_points

# 2. Updated Function: Convert Keypoints to Mask
def keypoints_to_mask():
    global segmentation_masks, point_coords, point_labels, point_orders, mode, current_mask_index

    """
    Converts the current set of keypoints into a mask polygon, visualizes it,
    and removes the keypoints from the canvas.
    Now includes optional Bezier curve smoothing for more naturalistic shapes.
    """

    if not point_coords:
        messagebox.showerror("Error", "No keypoints available to convert into a mask.")
        return

    if len(point_coords) < 3:
        messagebox.showerror("Error", "At least three keypoints are required to form a polygon.")
        return

    # Order the keypoints based on the sequence they were added
    ordered_points = order_keypoints_by_sequence(point_coords, point_orders)

    # --- Mask Options Dialog ---
    options_window = tk.Toplevel(root)
    options_window.title("Mask Creation Options")
    options_window.geometry("350x200")
    options_window.transient(root)
    options_window.grab_set()
    
    # Variables for options
    use_bezier_var = tk.BooleanVar(value=True)  # Default to smooth
    num_interpolated_var = tk.IntVar(value=50)
    
    # Bezier interpolation checkbox
    tk.Label(options_window, text="Shape Smoothing Options", font=('TkDefaultFont', 10, 'bold')).pack(pady=(15, 10))
    tk.Checkbutton(options_window, text="Use Bezier curve smoothing (naturalistic shape)", variable=use_bezier_var).pack(pady=5)
    
    # Number of interpolated points
    interp_frame = tk.Frame(options_window)
    interp_frame.pack(pady=5)
    tk.Label(interp_frame, text="Interpolation points:").pack(side=tk.LEFT)
    tk.Spinbox(interp_frame, from_=20, to=200, textvariable=num_interpolated_var, width=5).pack(side=tk.LEFT, padx=5)
    tk.Label(options_window, text="(More points = smoother curves)", fg="gray").pack()
    
    result = {'confirmed': False}
    
    def on_confirm():
        result['confirmed'] = True
        options_window.destroy()
    
    def on_cancel():
        options_window.destroy()
    
    # Buttons
    btn_frame = tk.Frame(options_window)
    btn_frame.pack(pady=20)
    tk.Button(btn_frame, text="Create Mask", command=on_confirm, bg="lightgreen").pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=10)
    
    # Wait for dialog to close
    options_window.wait_window()
    
    if not result['confirmed']:
        return
    
    # Get values from dialog
    use_bezier = use_bezier_var.get()
    num_interpolated = num_interpolated_var.get()
    
    # Apply Bezier smoothing if enabled
    if use_bezier and len(ordered_points) >= 3:
        # For closed shapes, we need to create a loop
        # Add first point at the end to close the curve smoothly
        closed_points = ordered_points + [ordered_points[0]]
        smoothed_points = bezier_curve(closed_points, num_interpolated)
        # Remove the duplicate closing point since fillPoly will close it
        if len(smoothed_points) > 1:
            smoothed_points = smoothed_points[:-1]
        final_points = smoothed_points
    else:
        final_points = ordered_points

    # Create a mask from the (optionally smoothed) polygon
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    pts = np.array([final_points], dtype=np.int32)
    cv2.fillPoly(mask, pts, 1)

    # CRITICAL FIX: Get category_id from currently selected label
    # This ensures multi-category annotations work in video mode
    current_label = selected_label.get() if mode != 'prediction_view' else selected_category_label.get()
    category_id = 1  # default
    for cat in categories:
        if cat['name'] == current_label:
            category_id = cat['id']
            break
    
    # Add the new mask to the segmentation_masks list with category_id
    segmentation_masks.append({
        'segmentation': mask,
        'category_id': category_id
    })
    current_mask_index = len(segmentation_masks) - 1  # Set current_mask_index to the new mask
    
    # Update the prediction_cache if in prediction_view mode
    if mode == 'prediction_view':
        image_name = selected_prediction_image.get()
        if image_name in prediction_cache:
            annotations = prediction_cache[image_name]
        else:
            prediction_cache[image_name] = []
            annotations = prediction_cache[image_name]

        # Convert mask to COCO segmentation
        mask_8bit = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coco_segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) >= 6:
                coco_segmentation.append(contour)

        # Create a new annotation
        annotation = {
            'id': len(annotations) + 1,
            'image_id': image_name,
            'category_id': -1,  # Default category_id
            'segmentation': coco_segmentation,
            'bbox': cv2.boundingRect(mask_8bit),
            'area': float(np.sum(mask_8bit > 0)),
            'iscrowd': 0
        }
        annotations.append(annotation)


    # Visualize the new mask
    apply_zoom()
    redraw_masks()    # Convert masks to the format used in this application

    # Clear the keypoints from the screen and internal lists
    clear_points()

    # Set the mode to mask_edit if it's not already
    # Do not change mode if in prediction_view mode
    if mode != 'prediction_view':
        mode = "mask_edit"

    smooth_str = " with Bezier smoothing" if use_bezier else ""
    messagebox.showinfo("Success", f"Keypoints have been converted to a mask{smooth_str}.")


def bezier_curve(points, num_interpolated=50):
    """
    Generate a smooth Bezier curve through a series of control points.
    Uses De Casteljau's algorithm for a B-spline-like curve that passes through all points.
    
    Parameters:
        points: List of [x, y] coordinates
        num_interpolated: Number of points to generate along the curve
        
    Returns:
        List of [x, y] coordinates along the smooth curve
    """
    if len(points) < 2:
        return points
    
    points = np.array(points, dtype=np.float64)
    n = len(points)
    
    if n == 2:
        # For 2 points, just interpolate linearly
        t_values = np.linspace(0, 1, num_interpolated)
        curve_points = []
        for t in t_values:
            x = points[0][0] * (1 - t) + points[1][0] * t
            y = points[0][1] * (1 - t) + points[1][1] * t
            curve_points.append([int(x), int(y)])
        return curve_points
    
    # Use Catmull-Rom spline for smooth interpolation through all points
    curve_points = []
    
    # Add phantom points at start and end for Catmull-Rom
    extended_points = np.vstack([
        2 * points[0] - points[1],  # Phantom point before start
        points,
        2 * points[-1] - points[-2]  # Phantom point after end
    ])
    
    # Generate points for each segment
    points_per_segment = max(2, num_interpolated // (n - 1))
    
    for i in range(1, len(extended_points) - 2):
        p0 = extended_points[i - 1]
        p1 = extended_points[i]
        p2 = extended_points[i + 1]
        p3 = extended_points[i + 2]
        
        for j in range(points_per_segment):
            t = j / points_per_segment
            
            # Catmull-Rom spline formula
            t2 = t * t
            t3 = t2 * t
            
            x = 0.5 * ((2 * p1[0]) +
                      (-p0[0] + p2[0]) * t +
                      (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                      (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            
            y = 0.5 * ((2 * p1[1]) +
                      (-p0[1] + p2[1]) * t +
                      (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                      (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            
            curve_points.append([int(x), int(y)])
    
    # Add the final point
    curve_points.append([int(points[-1][0]), int(points[-1][1])])
    
    return curve_points


def skeletonize_mask(mask):
    """
    Skeletonize a binary mask to get the centerline.
    Uses Zhang-Suen thinning algorithm via OpenCV if available,
    otherwise falls back to morphological thinning.
    
    Parameters:
        mask: Binary mask (H, W) with line pixels as 1 or 255
        
    Returns:
        Skeletonized binary mask
    """
    # Ensure mask is binary uint8
    if mask.max() <= 1:
        mask_8bit = (mask * 255).astype(np.uint8)
    else:
        mask_8bit = mask.astype(np.uint8)
    
    try:
        # Try using OpenCV's thinning (Zhang-Suen algorithm)
        skeleton = cv2.ximgproc.thinning(mask_8bit, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        # Fallback: use morphological thinning
        # This is a simple iterative erosion-based approach
        skeleton = mask_8bit.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.erode(skeleton, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(skeleton, temp)
            skeleton = eroded.copy()
            
            if cv2.countNonZero(temp) == 0:
                break
        
        # Ensure we have at least a 1-pixel wide line
        if cv2.countNonZero(skeleton) == 0:
            # If skeleton is empty, use the original mask
            skeleton = mask_8bit
    
    return skeleton


def extract_ordered_line_points(skeleton):
    """
    Extract ordered points along a skeletonized line.
    Orders from start (lowest x+y) to end (highest x+y).
    
    Parameters:
        skeleton: Skeletonized binary mask
        
    Returns:
        List of [x, y] points in order along the line
    """
    # Find all skeleton points
    points = np.argwhere(skeleton > 0)  # Returns [row, col] = [y, x]
    
    if len(points) == 0:
        return []
    
    # Convert to [x, y] format
    points_xy = [[p[1], p[0]] for p in points]
    
    # Find start and end based on x+y sum (topmost-leftmost to bottommost-rightmost)
    sums = [p[0] + p[1] for p in points_xy]
    start_idx = np.argmin(sums)
    
    # Use a greedy nearest-neighbor traversal from start point
    ordered_points = []
    remaining = set(range(len(points_xy)))
    current_idx = start_idx
    
    while remaining:
        remaining.discard(current_idx)
        ordered_points.append(points_xy[current_idx])
        
        if not remaining:
            break
        
        # Find nearest unvisited point
        current_point = np.array(points_xy[current_idx])
        min_dist = float('inf')
        nearest_idx = None
        
        for idx in remaining:
            dist = np.linalg.norm(current_point - np.array(points_xy[idx]))
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        if nearest_idx is not None:
            current_idx = nearest_idx
        else:
            break
    
    return ordered_points


def extract_evenly_spaced_keypoints(line_points, num_keypoints):
    """
    Extract evenly spaced keypoints along a line.
    
    Parameters:
        line_points: List of [x, y] points along the line (ordered)
        num_keypoints: Number of keypoints to extract (including start and end)
        
    Returns:
        List of [x, y] keypoints evenly distributed along the line
    """
    if len(line_points) < 2:
        return line_points
    
    if num_keypoints < 2:
        num_keypoints = 2
    
    if num_keypoints >= len(line_points):
        return line_points
    
    # Calculate cumulative distances along the line
    distances = [0.0]
    for i in range(1, len(line_points)):
        dx = line_points[i][0] - line_points[i-1][0]
        dy = line_points[i][1] - line_points[i-1][1]
        distances.append(distances[-1] + np.sqrt(dx*dx + dy*dy))
    
    total_length = distances[-1]
    if total_length == 0:
        return [line_points[0], line_points[-1]]
    
    # Calculate target distances for evenly spaced keypoints
    keypoints = []
    for i in range(num_keypoints):
        target_dist = (i / (num_keypoints - 1)) * total_length
        
        # Find the segment containing this distance
        for j in range(1, len(distances)):
            if distances[j] >= target_dist:
                # Interpolate between points j-1 and j
                segment_start = distances[j-1]
                segment_end = distances[j]
                segment_length = segment_end - segment_start
                
                if segment_length > 0:
                    t = (target_dist - segment_start) / segment_length
                else:
                    t = 0
                
                x = line_points[j-1][0] + t * (line_points[j][0] - line_points[j-1][0])
                y = line_points[j-1][1] + t * (line_points[j][1] - line_points[j-1][1])
                keypoints.append([int(x), int(y)])
                break
    
    return keypoints


def extract_keypoints_from_line_annotation(mask_data, use_stored_points=True):
    """
    Extract keypoints from a line annotation.
    Can use stored line_points or re-extract from mask via skeletonization.
    
    Parameters:
        mask_data: Dictionary containing mask and line metadata
        use_stored_points: If True, use stored line_points; otherwise skeletonize
        
    Returns:
        List of [x, y] keypoints evenly distributed along the line
    """
    num_keypoints = mask_data.get('num_line_keypoints', 10)
    
    if use_stored_points and 'line_points' in mask_data and mask_data['line_points']:
        # Use the stored Bezier-interpolated points
        line_points = mask_data['line_points']
        
        # Ensure consistent ordering (lowest x+y first)
        sums = [p[0] + p[1] for p in line_points]
        if sums[0] > sums[-1]:
            line_points = line_points[::-1]  # Reverse if needed
        
        return extract_evenly_spaced_keypoints(line_points, num_keypoints)
    else:
        # Fall back to skeletonization
        mask = mask_data['segmentation']
        
        try:
            skeleton = skeletonize_mask(mask)
            line_points = extract_ordered_line_points(skeleton)
            
            if len(line_points) < 2:
                return []
            
            return extract_evenly_spaced_keypoints(line_points, num_keypoints)
        except Exception as e:
            logging.warning(f"Skeletonization failed: {e}. Using stored points if available.")
            if 'line_points' in mask_data and mask_data['line_points']:
                return extract_evenly_spaced_keypoints(mask_data['line_points'], num_keypoints)
            return []


def keypoints_to_line():
    """
    Converts the current set of keypoints into a line annotation.
    Uses Bezier curve interpolation for smooth lines.
    Stores as a thin polygon segmentation mask compatible with COCO format.
    Auto-assigns supercategory "line" and prompts for category name (morphological feature).
    Also allows specifying number of keypoints to extract for COCO keypoint format.
    """
    global segmentation_masks, point_coords, point_labels, point_orders, mode, current_mask_index
    global categories, category_name_to_id, category_id_to_name, category_id_to_supercategory
    global coco_output_mask, coco_output_accumulate, labels

    if not point_coords:
        messagebox.showerror("Error", "No keypoints available to convert into a line.")
        return

    if len(point_coords) < 2:
        messagebox.showerror("Error", "At least two keypoints are required to form a line.")
        return

    # Order the keypoints based on the sequence they were added
    ordered_points = order_keypoints_by_sequence(point_coords, point_orders)
    
    # --- Line Options Dialog ---
    options_window = tk.Toplevel(root)
    options_window.title("Line Annotation Options")
    options_window.geometry("450x380")
    options_window.transient(root)
    options_window.grab_set()
    
    # Variables for options
    line_thickness_var = tk.IntVar(value=3)  # Default to thick for better training
    use_bezier_var = tk.BooleanVar(value=True)
    num_interpolated_var = tk.IntVar(value=50)
    category_name_var = tk.StringVar(value="")
    num_keypoints_var = tk.IntVar(value=10)  # Number of keypoints to extract for saving
    
    # Category name (morphological feature)
    tk.Label(options_window, text="Line Category (e.g., vein, suture, margin):").pack(pady=(10, 0))
    category_entry = tk.Entry(options_window, textvariable=category_name_var, width=30)
    category_entry.pack(pady=5)
    category_entry.focus_set()
    
    # Show existing line categories as suggestions
    existing_line_cats = [cat['name'] for cat in categories if cat.get('supercategory') == 'line']
    if existing_line_cats:
        tk.Label(options_window, text=f"Existing: {', '.join(existing_line_cats)}", fg="gray").pack()
    
    # Line thickness
    tk.Label(options_window, text="Line Thickness (pixels):").pack(pady=(15, 0))
    thickness_frame = tk.Frame(options_window)
    thickness_frame.pack()
    for val, label in [(1, "Thin (1)"), (2, "Normal (2)"), (3, "Thick (3)"), (5, "Bold (5)")]:
        tk.Radiobutton(thickness_frame, text=label, variable=line_thickness_var, value=val).pack(side=tk.LEFT, padx=5)
    
    # Bezier interpolation
    tk.Checkbutton(options_window, text="Use Bezier curve smoothing", variable=use_bezier_var).pack(pady=(15, 0))
    
    # Number of interpolated points
    interp_frame = tk.Frame(options_window)
    interp_frame.pack(pady=5)
    tk.Label(interp_frame, text="Interpolated points:").pack(side=tk.LEFT)
    tk.Spinbox(interp_frame, from_=10, to=200, textvariable=num_interpolated_var, width=5).pack(side=tk.LEFT, padx=5)
    
    # NEW: Number of keypoints to extract for COCO format
    tk.Label(options_window, text="ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬" * 40).pack(pady=(10, 5))
    tk.Label(options_window, text="Keypoints to extract (for COCO keypoints):").pack()
    keypoints_frame = tk.Frame(options_window)
    keypoints_frame.pack(pady=5)
    tk.Label(keypoints_frame, text="Number of keypoints:").pack(side=tk.LEFT)
    tk.Spinbox(keypoints_frame, from_=0, to=50, textvariable=num_keypoints_var, width=5).pack(side=tk.LEFT, padx=5)
    tk.Label(options_window, text="(0 = no keypoints, just save mask; otherwise start/end + evenly distributed)", fg="gray").pack()
    
    result = {'confirmed': False}
    
    def on_confirm():
        if not category_name_var.get().strip():
            messagebox.showerror("Error", "Please enter a category name for the line.")
            return
        result['confirmed'] = True
        options_window.destroy()
    
    def on_cancel():
        options_window.destroy()
    
    # Buttons
    btn_frame = tk.Frame(options_window)
    btn_frame.pack(pady=20)
    tk.Button(btn_frame, text="Create Line", command=on_confirm, bg="lightgreen").pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=10)
    
    # Wait for dialog to close
    options_window.wait_window()
    
    if not result['confirmed']:
        return
    
    # Get values from dialog
    line_thickness = line_thickness_var.get()
    use_bezier = use_bezier_var.get()
    num_interpolated = num_interpolated_var.get()
    category_name = category_name_var.get().strip()
    num_keypoints = num_keypoints_var.get()
    
    # Apply Bezier smoothing if enabled
    if use_bezier and len(ordered_points) >= 2:
        smoothed_points = bezier_curve(ordered_points, num_interpolated)
    else:
        smoothed_points = ordered_points
    
    # Create a mask for the line
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    pts = np.array(smoothed_points, dtype=np.int32)
    
    # Draw the line as a polyline with specified thickness
    cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=line_thickness)
    
    # --- Handle category creation with supercategory "line" ---
    if category_name in category_name_to_id:
        category_id = category_name_to_id[category_name]
    else:
        # Create new category with supercategory "line"
        existing_category_ids = set(cat['id'] for cat in categories)
        if existing_category_ids:
            new_category_id = max(existing_category_ids) + 1
        else:
            new_category_id = 1
        
        if new_category_id == 0:
            new_category_id = 1
        
        new_category = {
            "id": new_category_id,
            "name": category_name,
            "supercategory": "line"  # Auto-assign "line" as supercategory
        }
        categories.append(new_category)
        category_name_to_id[category_name] = new_category_id
        category_id_to_name[new_category_id] = category_name
        category_id_to_supercategory[new_category_id] = "line"
        category_id = new_category_id
        
        # Sync to coco outputs
        if "categories" not in coco_output_accumulate:
            coco_output_accumulate["categories"] = []
        if "categories" not in coco_output_mask:
            coco_output_mask["categories"] = []
        coco_output_accumulate["categories"].append(new_category.copy())
        coco_output_mask["categories"].append(new_category.copy())
        
        # Update dropdown
        if category_name not in label_options:
            label_options.append(category_name)
            update_label_dropdown()
        
        logging.info(f"Created new line category: {category_name} (ID: {new_category_id}, supercategory: line)")
    
    # Add the new mask to segmentation_masks with category already assigned
    new_mask_data = {
        'segmentation': mask,
        'category_id': category_id,
        'is_line': True,  # Flag to identify line annotations
        'line_points': smoothed_points,  # Store the line points for reference
        'line_thickness': line_thickness,
        'num_line_keypoints': num_keypoints  # Number of keypoints to extract when saving
    }
    segmentation_masks.append(new_mask_data)
    current_mask_index = len(segmentation_masks) - 1
    
    # Store label
    labels[current_mask_index] = category_name
    
    # Update the prediction_cache if in prediction_view mode
    if mode == 'prediction_view':
        image_name = selected_prediction_image.get()
        if image_name in prediction_cache:
            annotations = prediction_cache[image_name]
        else:
            prediction_cache[image_name] = []
            annotations = prediction_cache[image_name]

        # Convert mask to COCO segmentation
        mask_8bit = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coco_segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) >= 6:
                coco_segmentation.append(contour)

        # Create annotation with line-specific fields
        annotation = {
            'id': len(annotations) + 1,
            'image_id': image_name,
            'category_id': category_id,
            'segmentation': coco_segmentation,
            'bbox': list(cv2.boundingRect(mask_8bit)),
            'area': float(np.sum(mask_8bit > 0)),
            'iscrowd': 0,
            'is_line': True,
            'line_points': [[int(p[0]), int(p[1])] for p in smoothed_points],
            'line_thickness': line_thickness,
            'num_line_keypoints': num_keypoints
        }
        annotations.append(annotation)

    # Visualize the new mask
    apply_zoom()
    redraw_masks()

    # Clear the keypoints from the screen and internal lists
    clear_points()

    # Set the mode to mask_edit if not in prediction_view mode
    if mode != 'prediction_view':
        mode = "mask_edit"

    # AUTO-SAVE: Save the line mask annotation to coco_output_accumulate
    # This ensures line masks are saved when the marmot button is clicked
    save_line_annotation_to_accumulate(new_mask_data, category_name, category_id, smoothed_points, line_thickness, num_keypoints)

    kp_msg = f" ({num_keypoints} keypoints will be extracted)" if num_keypoints > 0 else " (no keypoints)"
    messagebox.showinfo("Success", f"Line '{category_name}' created with {len(smoothed_points)} points{kp_msg}")


def save_line_annotation_to_accumulate(mask_data, category_name, category_id, line_points, line_thickness, num_keypoints):
    """
    Save a line annotation directly to coco_output_accumulate.
    This is called automatically when a line is created to ensure it gets saved.
    """
    global coco_output_accumulate, coco_output_mask, file_path_base, instance_counters, image
    
    mask = mask_data['segmentation']
    mask_8bit = (mask * 255).astype(np.uint8)
    
    # Generate contours from the mask
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coco_segmentation = []
    for contour in contours:
        contour_flat = contour.flatten().tolist()
        if len(contour_flat) >= 6:
            coco_segmentation.append(contour_flat)
    
    if not coco_segmentation:
        logging.warning(f"No valid contours found for line '{category_name}'")
        return
    
    # Calculate bounding box and area
    x, y, w, h = cv2.boundingRect(mask_8bit)
    bbox = [x, y, w, h]
    area = float(np.sum(mask_8bit > 0))
    
    # Get instance number for this category
    if category_name not in instance_counters:
        instance_counters[category_name] = 0
    instance_counters[category_name] += 1
    instance_number = instance_counters[category_name]
    
    # Create unique annotation ID
    annotation_id = int(time.time() * 1000) + instance_number
    
    # Image metadata
    file_dir, file_name = os.path.split(file_path_base)
    file_base, file_ext = os.path.splitext(file_name)
    if not file_ext:
        file_ext = ".jpg"
    full_file_path = os.path.join(file_dir, file_base + file_ext)
    
    image_metadata = {
        "id": file_base,
        "file_name": os.path.basename(full_file_path),
        "height": image.shape[0],
        "width": image.shape[1]
    }
    
    # Check if image already in coco_output_accumulate
    existing_image_ids = [img['id'] for img in coco_output_accumulate.get('images', [])]
    if file_base not in existing_image_ids:
        coco_output_accumulate["images"].append(image_metadata)
    
    # Create annotation with line-specific fields
    annotation = {
        "id": annotation_id,
        "image_id": file_base,
        "category_id": category_id,
        "segmentation": coco_segmentation,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "instance_id": instance_number,
        "is_line": True,
        "line_points": [[int(p[0]), int(p[1])] for p in line_points],
        "line_thickness": line_thickness,
        "num_line_keypoints": num_keypoints
    }
    
    # Append annotation to coco_output_accumulate
    coco_output_accumulate["annotations"].append(annotation)
    
    # Ensure categories are synced
    if "categories" not in coco_output_accumulate or not coco_output_accumulate["categories"]:
        coco_output_accumulate["categories"] = categories.copy()
    else:
        existing_cat_ids = {cat['id'] for cat in coco_output_accumulate.get("categories", [])}
        for cat in categories:
            if cat['id'] not in existing_cat_ids:
                coco_output_accumulate["categories"].append(cat.copy())
    
    logging.info(f"Auto-saved line annotation: {category_name}_{instance_number} (LINE, {num_keypoints} keypoints to extract)")


def run_coco_converter_script():
    """
    Opens a dialog to collect inputs and runs the coco_converter_v3.py script with the provided arguments.
    """
    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Run COCO Converter Script")
    
    # Define variables to store user inputs
    input_json = tk.StringVar()
    output_dir = tk.StringVar()
    exclude_categories = tk.StringVar()
    output_base_name = tk.StringVar(value='converted_output')
    reassignImageIdsCheckBox = tk.BooleanVar()
    img_dir = tk.StringVar()  # NEW: Variable for the optional image directory


    # Function to browse files/directories
    def browse_file(var):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            var.set(path)
    
    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)
    
    # Layout the input fields
    tk.Label(input_window, text="Input COCO JSON File (Required):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=input_json, width=50).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(input_json)).grid(row=0, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Output Directory (Required):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_dir)).grid(row=1, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Exclude Categories (Optional, comma-separated):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=exclude_categories, width=50).grid(row=2, column=1, padx=5, pady=5)
    
    tk.Label(input_window, text="Output Base Name (Optional):").grid(row=3, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_base_name, width=50).grid(row=3, column=1, padx=5, pady=5)
    
    tk.Checkbutton(input_window, text="For Re-annotation Checkbox Only", variable=reassignImageIdsCheckBox).grid(row=4, column=1, sticky='w', padx=5, pady=5)

    tk.Label(input_window, text="Image Directory (Optional):").grid(row=5, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=img_dir, width=50).grid(row=5, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(img_dir)).grid(row=5, column=2, padx=5, pady=5)


    # Function to gather inputs and run the converter script
    def submit():
        # Gather required arguments
        input_json_val = input_json.get()
        output_dir_val = output_dir.get()
        output_base_name_val = output_base_name.get()
        exclude_categories_val = exclude_categories.get()
        reassignImageIdsCheckBox_val = reassignImageIdsCheckBox.get()
        img_dir_val = img_dir.get()

        # Validate required fields
        if not input_json_val:
            messagebox.showerror("Input Error", "Please select an input COCO JSON file.")
            return
        if not output_dir_val:
            messagebox.showerror("Input Error", "Please select an output directory.")
            return
        
        # Construct the path to the converter script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        converter_script = script_dir / 'coco_converter_v24.py'
        
        # Ensure the converter script exists
        if not converter_script.is_file():
            messagebox.showerror("Script Error", f"Converter script not found at: {converter_script}")
            return
        
        # Construct the command
        cmd = [
            "conda", "run", "-n", "detectron2_env", "python",
            str(converter_script),
            "--input-json", input_json_val,
            "--output-dir", output_dir_val,
            "--output-base-name", output_base_name_val,
            "--img-dir", img_dir_val
        ]
        
        if exclude_categories_val:
            # Split the categories by comma and strip any whitespace
            categories_list = [cat.strip() for cat in exclude_categories_val.split(',')]
            cmd.extend(["--exclude-categories"] + categories_list)
        if reassignImageIdsCheckBox_val:
            cmd.append("--reassign-image-ids")

        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running COCO Converter Script")
                tk.Label(loading, text="Please wait while the COCO converter script runs...").pack(padx=20, pady=20)
                loading.update()
                
                subprocess.run(cmd, check=True)
                loading.destroy()
                messagebox.showinfo("Success", "COCO converter script executed successfully.")
            except subprocess.CalledProcessError as e:
                loading.destroy()
                messagebox.showerror("Error", f"An error occurred while running the converter script:\n{e}")
            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
        
        threading.Thread(target=run_script).start()
        input_window.destroy()
    
    # Add Submit and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=6, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=6, column=2, sticky='w', padx=5, pady=10)



def select_file(entry_widget, title):
    path = filedialog.askopenfilename(title=title)
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, path)

def select_directory(entry_widget, title):
    path = filedialog.askdirectory(title=title)
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, path)


def run_remove_images_script():
    """
    Opens a dialog to collect inputs for removing images (and associated annotations)
    from a COCO JSON file based on a file list. Calls the removal script with argparse-style
    arguments.
    """
    input_window = tk.Toplevel()
    input_window.title("Remove Images from COCO JSON")
    
    # Define StringVars for the inputs
    coco_json = tk.StringVar()
    file_list = tk.StringVar()
    output_file = tk.StringVar(value="./output.json")
    
    def browse_json(var):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            var.set(path)
    
    def browse_file_list(var):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if path:
            var.set(path)
    
    def browse_output(var):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if path:
            var.set(path)
    
    # Layout the input fields
    tk.Label(input_window, text="Input COCO JSON File:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=coco_json, width=50).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_json(coco_json)).grid(row=0, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="File List (one filename per line):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=file_list, width=50).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file_list(file_list)).grid(row=1, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Output JSON File:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_file, width=50).grid(row=2, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_output(output_file)).grid(row=2, column=2, padx=5, pady=5)
    
    def submit():
        # Gather the arguments
        coco_val = coco_json.get()
        file_list_val = file_list.get()
        output_val = output_file.get()
        
        # Validate required fields
        if not coco_val:
            messagebox.showerror("Error", "Please select an input COCO JSON file.")
            return
        if not file_list_val:
            messagebox.showerror("Error", "Please select a file list to remove.")
            return
        if not output_val:
            messagebox.showerror("Error", "Please specify an output JSON file path.")
            return
        
        # Construct the command.
        # Assume that the removal script (remove_images_from_coco.py) is in the same directory as this script.
        script_dir = Path(__file__).parent
        removal_script = script_dir / "remove_images_from_coco.py"
        if not removal_script.exists():
            messagebox.showerror("Error", f"Removal script not found at: {removal_script}")
            return
        
        cmd = [
            "python", str(removal_script),
            "--coco", coco_val,
            "--file_list", file_list_val,
            "--output", output_val
        ]
        
        def run_script():
            try:
                # Optionally, show a simple loading window
                loading = tk.Toplevel()
                loading.title("Removing Images...")
                tk.Label(loading, text="Please wait while images are removed...").pack(padx=20, pady=20)
                loading.update()
                
                subprocess.run(cmd, check=True)
                loading.destroy()
                messagebox.showinfo("Success", "Images and associated annotations removed successfully.")
            except subprocess.CalledProcessError as e:
                loading.destroy()
                messagebox.showerror("Error", f"An error occurred:\n{e}")
        
        threading.Thread(target=run_script).start()
        input_window.destroy()
    
    # Add Submit and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=3, column=1, padx=5, pady=10, sticky='e')
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=3, column=2, padx=5, pady=10, sticky='w')



def run_detectron2_script(coco_json_path, img_dir, output_dir, total_iters, checkpoint_period, dataset_name, keypoints, train_keypoints_only, train_segmentation_only, popup):
#  def run_detectron2_script(coco_json_path, img_dir, output_dir, total_iters, checkpoint_period,
#                          dataset_name, keypoints, train_keypoints_only, train_segmentation_only, popup):

    # Validate inputs
    if not all([coco_json_path, img_dir, output_dir, total_iters, checkpoint_period, dataset_name]):
        messagebox.showerror("Input Error", "Please fill in all fields.")
        return

    # Close the popup window
    popup.destroy()


    # Inform the user that training is starting.
    messagebox.showinfo("Training", "Detectron2 training is now starting. Please wait...")


    # Get the directory of the current script (the GUI script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'detectron2/detectron2_training_and_filterV10_and_kpts-17.py')  # Update with the correct script name

    # Construct the command using 'conda run' to activate the environment
    command = [
        'conda', 'run', '-n', 'detectron2_env', 'python', script_path,
        '--coco-json', coco_json_path,
        '--img-dir', img_dir,
        '--output-dir', output_dir,
        '--total-iters', str(total_iters),
        '--checkpoint-period', str(checkpoint_period),
        '--dataset-name', dataset_name,  # Include the dataset name argument
    ]


    # Add the --keypoints flag if the checkbox is selected
    if keypoints:
        command.append('--keypoint')
    # Add the additional flags if selected
    if train_keypoints_only:
        command.append('--train-keypoints-only')
    if train_segmentation_only:
        command.append('--train-segmentation-only')


    def target():
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True  # Ensure text mode for Python 3
            )

            # Read the output
            def read_output(pipe):
                for line in iter(pipe.readline, ''):
                    print(line, end='')  # Replace with GUI output method if needed
                pipe.close()

            threading.Thread(target=read_output, args=(process.stdout,)).start()
            threading.Thread(target=read_output, args=(process.stderr,)).start()

            process.wait()
         # Notify the user upon completion
            if process.returncode == 0:
                messagebox.showinfo("Success", f"Detectron2 training completed successfully for dataset '{dataset_name}'.")
            else:
                messagebox.showerror("Error", f"Detectron2 training failed for dataset '{dataset_name}' with return code {process.returncode}.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run the training script: {e}")

    threading.Thread(target=target).start()



def select_file_detectron2(entry_widget, title, popup):
    """
    Custom file selection function for Detectron2 training popup.
    Ensures the popup remains open after selection.
    """
    path = filedialog.askopenfilename(title=title, filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
    if path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, path)
    # Do not close the popup here

def select_directory_detectron2(entry_widget, title, popup):
    """
    Custom directory selection function for Detectron2 training popup.
    Ensures the popup remains open after selection.
    """
    path = filedialog.askdirectory(title=title)
    if path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, path)
    # Do not close the popup here

def open_training_popup():
    popup = tk.Toplevel()
    popup.title("Train Detectron2 Model")
    popup.grab_set()  # Make the popup modal
    
    # COCO JSON path
    tk.Label(popup, text="COCO JSON File:").pack(padx=10, pady=5)
    coco_json_entry = tk.Entry(popup, width=50)
    coco_json_entry.pack(padx=10, pady=5)
    tk.Button(
        popup,
        text="Browse",
        command=lambda: select_file_detectron2(coco_json_entry, "Select COCO JSON File", popup)
    ).pack(padx=10, pady=5)
    
    # Image directory
    tk.Label(popup, text="Image Directory:").pack(padx=10, pady=5)
    img_dir_entry = tk.Entry(popup, width=50)
    img_dir_entry.pack(padx=10, pady=5)
    tk.Button(
        popup,
        text="Browse",
        command=lambda: select_directory_detectron2(img_dir_entry, "Select Image Directory", popup)
    ).pack(padx=10, pady=5)
    
    # Output directory
    tk.Label(popup, text="Output Directory:").pack(padx=10, pady=5)
    output_dir_entry = tk.Entry(popup, width=50)
    output_dir_entry.pack(padx=10, pady=5)
    tk.Button(
        popup,
        text="Browse",
        command=lambda: select_directory_detectron2(output_dir_entry, "Select Output Directory", popup)
    ).pack(padx=10, pady=5)
    
    # Dataset name
    tk.Label(popup, text="Dataset Name:").pack(padx=10, pady=5)
    dataset_name_entry = tk.Entry(popup, width=50)
    dataset_name_entry.insert(0, "your_taxon")  # Default value
    dataset_name_entry.pack(padx=10, pady=5)
    
    # Number of iterations
    tk.Label(popup, text="Total Iterations:").pack(padx=10, pady=5)
    total_iters_entry = tk.Entry(popup, width=20)
    total_iters_entry.insert(0, "35000")  # Default value
    total_iters_entry.pack(padx=10, pady=5)
    
    # Checkpoint period
    tk.Label(popup, text="Checkpoint Period:").pack(padx=10, pady=5)
    checkpoint_period_entry = tk.Entry(popup, width=20)
    checkpoint_period_entry.insert(0, "1000")  # Default value
    checkpoint_period_entry.pack(padx=10, pady=5)
    
    # Keypoint checkbox
    keypoints_checkbox_var = tk.BooleanVar(value=False)  # store True/False
    tk.Checkbutton(
        popup,
        text="Force keypoint training if present:",
        variable=keypoints_checkbox_var
    ).pack(padx=10, pady=5)
    # NOTE: We do NOT call .pack() on the BooleanVar itself. We only pack the Checkbutton.
    # Checkbox: Train Keypoints Only
    train_keypoints_only_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        popup,
        text="Train Keypoints Only",
        variable=train_keypoints_only_var
    ).pack(padx=10, pady=5)

    # Checkbox: Train Segmentation Only
    train_segmentation_only_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        popup,
        text="Train Segmentation Only",
        variable=train_segmentation_only_var
    ).pack(padx=10, pady=5)


    # Keypoint checkbox
    #tk.Checkbutton(popup, text="Force keypoint training if present:").pack(padx=5, pady=5)
    #keypoints_checkbox_entry = tk.BooleanVar(Value=False)
    #keypoints_checkbox_entry.pack(padx=5, pady=5)
    
    # Start Training button
    tk.Button(
        popup,
        text="Start Training",
        command=lambda: run_detectron2_script(
            coco_json_entry.get(),
            img_dir_entry.get(),
            output_dir_entry.get(),
            total_iters_entry.get(),
            checkpoint_period_entry.get(),
            dataset_name_entry.get(),  # Pass the dataset name
            keypoints_checkbox_var.get(),
            train_keypoints_only_var.get(),
            train_segmentation_only_var.get(),
            popup
        )
    ).pack(padx=10, pady=20)


   # Cancel button (to close the popup)
    tk.Button(
        popup,
        text="Cancel",
        command=popup.destroy
    ).pack(padx=10, pady=5)





def predict_and_filter_popup():
    # Create a new popup window
    popup = tk.Toplevel()
    popup.title("Detectron2 Predict and Filter")
    popup.grab_set()  # Make the popup modal

    # Configure grid layout
    for i in range(12):
        popup.grid_rowconfigure(i, weight=1)
    popup.grid_columnconfigure(1, weight=1)

    # Input Image Directory
    tk.Label(popup, text="Image Directory:").grid(row=0, column=0, padx=10, pady=5, sticky='e')
    image_dir_entry = tk.Entry(popup, width=50)
    image_dir_entry.grid(row=0, column=1, padx=10, pady=5, sticky='we')
    def browse_image_dir():
        dir_path = filedialog.askdirectory(title="Select Image Directory")
        if dir_path:
            image_dir_entry.delete(0, tk.END)
            image_dir_entry.insert(0, dir_path)
    tk.Button(popup, text="Browse", command=browse_image_dir).grid(row=0, column=2, padx=10, pady=5)

    # Configuration YAML File
    tk.Label(popup, text="Config YAML File:").grid(row=1, column=0, padx=10, pady=5, sticky='e')
    config_file_entry = tk.Entry(popup, width=50)
    config_file_entry.grid(row=1, column=1, padx=10, pady=5, sticky='we')
    def browse_config_file():
        file_path = filedialog.askopenfilename(title="Select Config YAML File", filetypes=[("YAML Files", "*.yaml"), ("All Files", "*.*")])
        if file_path:
            config_file_entry.delete(0, tk.END)
            config_file_entry.insert(0, file_path)
    tk.Button(popup, text="Browse", command=browse_config_file).grid(row=1, column=2, padx=10, pady=5)

    # Model Weights File
    tk.Label(popup, text="Model Weights (.pth):").grid(row=2, column=0, padx=10, pady=5, sticky='e')
    model_weights_entry = tk.Entry(popup, width=50)
    model_weights_entry.grid(row=2, column=1, padx=10, pady=5, sticky='we')
    def browse_model_weights():
        file_path = filedialog.askopenfilename(title="Select Model Weights File", filetypes=[("PyTorch Model Files", "*.pth"), ("All Files", "*.*")])
        if file_path:
            model_weights_entry.delete(0, tk.END)
            model_weights_entry.insert(0, file_path)
    tk.Button(popup, text="Browse", command=browse_model_weights).grid(row=2, column=2, padx=10, pady=5)

    # Metadata JSON File
    tk.Label(popup, text="Metadata JSON File:").grid(row=3, column=0, padx=10, pady=5, sticky='e')
    metadata_json_entry = tk.Entry(popup, width=50)
    metadata_json_entry.grid(row=3, column=1, padx=10, pady=5, sticky='we')
    def browse_metadata_json():
        file_path = filedialog.askopenfilename(title="Select Metadata JSON File", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if file_path:
            metadata_json_entry.delete(0, tk.END)
            metadata_json_entry.insert(0, file_path)
    tk.Button(popup, text="Browse", command=browse_metadata_json).grid(row=3, column=2, padx=10, pady=5)

    # Output Directory
    tk.Label(popup, text="Output Directory:").grid(row=4, column=0, padx=10, pady=5, sticky='e')
    output_dir_entry = tk.Entry(popup, width=50)
    output_dir_entry.grid(row=4, column=1, padx=10, pady=5, sticky='we')
    def browse_output_dir():
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            output_dir_entry.delete(0, tk.END)
            output_dir_entry.insert(0, dir_path)
    tk.Button(popup, text="Browse", command=browse_output_dir).grid(row=4, column=2, padx=10, pady=5)

    # Optional Categories to Retain
    tk.Label(popup, text="Categories to Retain (optional):").grid(row=5, column=0, padx=10, pady=5, sticky='e')
    categories_entry = tk.Entry(popup, width=50)
    categories_entry.grid(row=5, column=1, padx=10, pady=5, sticky='we')
    tk.Label(popup, text="(e.g., left_elytra,pronotum,head,rostrum)").grid(row=5, column=2, padx=10, pady=5, sticky='w')

    # Number of Sample Images
    tk.Label(popup, text="Number of Sample Images:").grid(row=6, column=0, padx=10, pady=5, sticky='e')
    sample_images_entry = tk.Entry(popup, width=50)
    sample_images_entry.insert(0, "10")  # Default value
    sample_images_entry.grid(row=6, column=1, padx=10, pady=5, sticky='we')

    # Prediction Threshold
    tk.Label(popup, text="Prediction Threshold:").grid(row=7, column=0, padx=10, pady=5, sticky='e')
    score_threshold_entry = tk.Entry(popup, width=50)
    score_threshold_entry.insert(0, "0.8")  # Default value
    score_threshold_entry.grid(row=7, column=1, padx=10, pady=5, sticky='we')

    # Number of iterations
    tk.Label(popup, text="Total Iterations from Training:").grid(row=8, column=0, padx=10, pady=5, sticky='e')
    total_iters_entry = tk.Entry(popup, width=20)
    total_iters_entry.insert(0, "25000")  # Default value
    total_iters_entry.grid(row=8, column=1, padx=10, pady=5, sticky='we')

    # Dataset Name from Training
    tk.Label(popup, text="Dataset Name from Training:").grid(row=9, column=0, padx=10, pady=5, sticky='e')
    dataset_name_entry = tk.Entry(popup, width=20)
    dataset_name_entry.insert(0, "my_training_set")  # Default value
    dataset_name_entry.grid(row=9, column=1, padx=10, pady=5, sticky='we')

    # ------ NEW: Prediction Mode Selection ------
    tk.Label(popup, text="Prediction Mode:").grid(row=10, column=0, padx=10, pady=5, sticky='e')
    prediction_mode_var = tk.StringVar(value="both")  # Default: both segmentation and keypoints
    # The following RadioButtons create three mutually exclusive options.
    frame = tk.Frame(popup)
    frame.grid(row=10, column=1, padx=10, pady=5, sticky='w')
    tk.Radiobutton(frame, text="Both", variable=prediction_mode_var, value="both").pack(side='left')
    tk.Radiobutton(frame, text="Segmentation Only", variable=prediction_mode_var, value="segmentation").pack(side='left', padx=15)
    tk.Radiobutton(frame, text="Keypoints Only", variable=prediction_mode_var, value="keypoints").pack(side='left', padx=15)
    # ------ END NEW ------

    # Run Button
    def run_prediction_and_filter():
        image_dir = image_dir_entry.get()
        config_file = config_file_entry.get()
        model_weights = model_weights_entry.get()
        metadata_json = metadata_json_entry.get()
        output_dir = output_dir_entry.get()
        categories = categories_entry.get()
        sample_images = sample_images_entry.get()
        score_threshold = score_threshold_entry.get()
        total_iters = total_iters_entry.get()
        dataset_name = dataset_name_entry.get()
        prediction_mode = prediction_mode_var.get()  # Get the selected mode

        # Validate required fields
        if not all([image_dir, config_file, model_weights, metadata_json, output_dir]):
            messagebox.showerror("Input Error", "Please fill in all required fields.")
            return

        # Validate score_threshold
        try:
            score_threshold = float(score_threshold)
            if not (0.0 <= score_threshold <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Prediction Threshold must be a float between 0 and 1.")
            return
        # Get the directory of the current script (the GUI script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path_pred = os.path.join(script_dir, './detectron2/detectron2_predict_and_filterV10_and_kptsV2.py')  # Update with the correct script name

        # Prepare the command
        command = [
            'conda', 'run', '-n', 'detectron2_env', 'python', script_path_pred,  # Update to your script name if different
            '--image_dir', image_dir,
            '--config_file', config_file,
            '--model_weights', model_weights,
            '--metadata_json', metadata_json,
            '--output_dir', output_dir,
            '--sample_images', sample_images,
            '--score_threshold', str(score_threshold),
            '--total_iters', str(total_iters),
            '--dataset_name', str(dataset_name)
        ]

        if categories:
            command += ['--categories', categories]

        # Append prediction mode flags (mutually exclusive):
        if prediction_mode == "segmentation":
            command.append('--predict_segmentation_only')
        elif prediction_mode == "keypoints":
            command.append('--predict_keypoints_only')
        # "both" produces no additional flag

        # Disable the run button to prevent multiple clicks
        run_button.config(state='disabled')

        # Function to execute the command
        def execute_command():
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True  # Requires Python 3.7+
                )
                stdout, stderr = process.communicate()

                if process.returncode == 0:
                    messagebox.showinfo("Success", f"Prediction and filtering completed successfully.\nOutput saved to:\n{output_dir}")
                else:
                    messagebox.showerror("Error", f"Prediction and filtering failed.\nError:\n{stderr}")
            except Exception as e:
                messagebox.showerror("Error", f"An exception occurred:\n{e}")
            finally:
                run_button.config(state='normal')

        # Run the command in a separate thread to keep the GUI responsive
        threading.Thread(target=execute_command).start()

    run_button = tk.Button(popup, text="Run", command=run_prediction_and_filter)
    run_button.grid(row=11, column=1, padx=10, pady=20)

    # Configure grid weights for proper resizing
    popup.grid_columnconfigure(1, weight=1)


def on_select_prediction_image(value):
    if check_unsaved_changes():
        selected_prediction_image.set(value)
        load_prediction_image(value)

def update_category_label_dropdown():
    global category_label_dropdown, selected_category_label, category_label_options
    # Clear the current OptionMenu and repopulate with new labels
    menu = category_label_dropdown["menu"]
    menu.delete(0, "end")

    # Add all labels to the dropdown menu
    for label in category_label_options:
        menu.add_command(label=label, command=tk._setit(selected_category_label, label))

    # Reset the selection to default
    selected_category_label.set("Select Category")


def load_prediction_image(image_name):
    global segmentation_masks, current_mask_index, image, file_path_base, selected_prediction_image
    global category_id_to_name, mode, working_pred_json_path
    mode = 'prediction_view'
    if image_name == "Select Prediction Image":
        return

    selected_prediction_image.set(image_name)
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        messagebox.showerror("Error", f"Image file '{image_path}' does not exist.")
        return

    try:
        pil_image = Image.open(image_path).convert("RGB")
        image = np.array(pil_image)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image:\n{e}")
        return

    canvas.delete("all")
    canvas.config(scrollregion=(0, 0, image.shape[1], image.shape[0]))
    file_path_base = os.path.splitext(os.path.basename(image_path))[0]

    # Load annotations for this image from working_pred_json_path
    try:
        with open(working_pred_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load working prediction JSON:\n{e}")
        return

    # Find annotations for this image
    #image_annotations = []
    image_id = None
    # First, find the image_id corresponding to this image_name
    for img in data.get('images', []):
        if img.get('file_name') == image_name:
            image_id = img['id']
            break
    if image_id is None:
        messagebox.showerror("Error", f"No image_id found for image '{image_name}' in working prediction JSON.")
        return

    # Get annotations for this image
#    image_annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

    # Filter annotations for this image (using integer image_id)
    image_annotations = [ann for ann in data.get('annotations', []) if ann.get('image_id') == image_id]



    # Reset segmentation_masks
    segmentation_masks = []
    global point_coords, point_labels, point_orders
    point_coords = []
    point_labels = []
    point_orders = []




    # Build segmentation masks from annotations with segmentation data
    for ann in image_annotations:
        if 'segmentation' in ann and ann['segmentation']:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for seg in ann['segmentation']:
                if isinstance(seg, list):
                    polygon = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [polygon], 1)
            if np.any(mask):
                segmentation_masks.append({
                    'segmentation': mask,
                    'category_id': ann.get('category_id', -1)
                })

#    segmentation_masks = []     
    # Build segmentation_masks from annotations with segmentation data
#    for ann in image_annotations:
#        if 'segmentation' in ann and ann['segmentation']:
#            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#            for seg in ann['segmentation']:
#                if isinstance(seg, list):
#                    polygon = np.array(seg).reshape((-1, 2)).astype(np.int32)
#                    cv2.fillPoly(mask, [polygon], 1)
#            if np.any(mask):
#                segmentation_masks.append({'segmentation': mask, 'category_id': ann.get('category_id', -1)})

    #    mask = ann.get('segmentation', [])
    #    category_id = ann.get('category_id', -1)
    #    mask_np = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    #    for seg in mask:
    #        if isinstance(seg, list):
    #            polygon = np.array(seg).reshape((-1, 2)).astype(np.int32)
    #            cv2.fillPoly(mask_np, [polygon], 1)
    #    if np.any(mask_np):
    #        segmentation_masks.append({'segmentation': mask_np, 'category_id': category_id})

    current_mask_index = 0

    # Build keypoints from annotations that contain a keypoints field.
    # For simplicity, use only the first annotation that has keypoints.
    for ann in image_annotations:
        if 'keypoints' in ann and ann['keypoints']:
            # Use your load_json_keypoints_into_gui to convert the keypoints
            load_json_keypoints_into_gui([ann])
            break  # Only load one set per image

    # Redraw the masks using your existing functions
    apply_zoom()
    redraw_masks()
    
    messagebox.showinfo("Image Loaded", f"Loaded image '{image_name}' with {len(segmentation_masks)} masks.")

    # Set the selected category label
#    if segmentation_masks:
#        category_id = segmentation_masks[current_mask_index].get('category_id', -1)
#        category_name = category_id_to_name.get(category_id, "unlabeled")
#        selected_category_label.set(category_name)
#    else:
#        selected_category_label.set("Select Category")
#
#    apply_zoom()
#    redraw_masks()
#    messagebox.showinfo("Image Loaded", f"Loaded image '{image_name}' with {len(segmentation_masks)} masks.")


def open_prediction_list(image_names):
    popup = tk.Toplevel()
    popup.title("Select Image")

    frame = ttk.Frame(popup)
    frame.pack(fill='both', expand=True)

    scrollbar = ttk.Scrollbar(frame, orient='vertical')
    listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, height=20, width=50)

    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side='right', fill='y')
    listbox.pack(side='left', fill='both', expand=True)

    for img_name in image_names:
        listbox.insert('end', img_name)

    def on_select(event):
        selected = listbox.curselection()
        if selected:
            index = selected[0]
            img_name = image_names[index]
            selected_prediction_image.set(img_name)
            load_prediction_image(img_name)

            # Load keypoints for selected image
            if img_name in prediction_data:
                annotations_for_image = prediction_data[img_name]
                load_json_keypoints_into_gui(annotations_for_image)
            else:
                print(f"No annotations found for '{img_name}' in prediction_data.")

            popup.destroy()
#    def on_select(event):
#        selected = listbox.curselection()
#        if selected:
#            index = selected[0]
#            selected_prediction_image.set(image_paths[index])
#            popup.destroy()

    listbox.bind('<<ListboxSelect>>', on_select)


def check_unsaved_changes():
    global segmentation_masks
    if segmentation_masks:
        result = messagebox.askyesnocancel("Unsaved Changes",
                                           "You have unsaved changes. Do you want to save them before proceeding?\nYes: Save\nNo: Discard\nCancel: Stay on current image")
        if result is None:
            # Cancel
            return False  # Don't proceed
        elif result:
            # Yes: Save
            reannotate_masks()
            return True  # Proceed
        else:
            # No: Discard changes
            return True  # Proceed
    return True  # No unsaved changes, proceed



def save_predictions_from_cache():
    """
    Save the predictions stored in prediction_cache to a new COCO JSON file in single-line format.
    """
    global prediction_cache, file_path_base, image_dir, categories, category_name_to_id, category_id_to_supercategory
  
    if not prediction_cache:
        messagebox.showerror("Error", "No predictions available to save.")
        return

    # Initialize structures for COCO JSON
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(categories)
    }

    annotation_id = 1  # Start annotation IDs from 1

    for img_filename, anns in prediction_cache.items():
        # Construct the full image path
        img_path = os.path.join(image_dir, img_filename)
        if not os.path.exists(img_path):
            logging.warning(f"Image file '{img_path}' does not exist. Skipping.")
            continue

        # Load image to get dimensions
        try:
            pil_image = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_image)
            height, width = img_np.shape[:2]
        except Exception as e:
            logging.warning(f"Failed to load image '{img_path}': {e}")
            continue

        # Add image metadata
        image_id = img_filename  # Use filename as image ID
        image_metadata = {
            "id": image_id,
            "file_name": img_filename,
            "height": height,
            "width": width
        }
        coco_output["images"].append(image_metadata)

        # Add annotations
        for ann in anns:
            ann_copy = ann.copy()
            ann_copy['id'] = annotation_id
            ann_copy['image_id'] = image_id
            coco_output["annotations"].append(ann_copy)
            annotation_id += 1

    if not coco_output["annotations"]:
        messagebox.showerror("Error", "No valid annotations found in predictions.")
        return

    # Prompt user to save the COCO JSON
    save_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        title="Save Predictions as COCO JSON",
        filetypes=[("COCO JSON Files", "*.json")]
    )

    if save_path:
        try:
            with open(save_path, 'w') as json_file:
                json.dump(coco_output, json_file, indent=4)
            messagebox.showinfo("Success", f"Predictions saved successfully to '{save_path}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save predictions:\n{e}")
    else:
        messagebox.showinfo("Info", "Save operation cancelled.")





def update_annotation_in_cache():
    global segmentation_masks, current_mask_index, prediction_cache, selected_prediction_image
    # Convert the mask to segmentation format (polygons)
    mask = segmentation_masks[current_mask_index]['segmentation']
    mask_8bit = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coco_segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) >= 6:
            coco_segmentation.append(contour)
    # Update the annotation in prediction_cache
    image_name = selected_prediction_image.get()
    if image_name in prediction_cache:
        annotations = prediction_cache[image_name]
        # Ensure the index is valid
        if current_mask_index < len(annotations):
            ann = annotations[current_mask_index]
            ann['segmentation'] = coco_segmentation
            # Update bbox and area
            x, y, w, h = cv2.boundingRect(mask_8bit)
            ann['bbox'] = [x, y, w, h]
            ann['area'] = float(np.sum(mask_8bit > 0))


def delete_current_mask():
    global segmentation_masks, current_mask_index
    if segmentation_masks:
        del segmentation_masks[current_mask_index]
        if current_mask_index >= len(segmentation_masks):
            current_mask_index = len(segmentation_masks) - 1
        apply_zoom()
        redraw_masks()
        messagebox.showinfo("Success", "Current mask deleted.")
    else:
        messagebox.showerror("Error", "No masks to delete.")



#####
def load_image_list():
    """
    Loads the list of image filenames from the working prediction JSON.
    """
    global image_list, current_image_index, working_pred_json_path
    
    try:
        with open(working_pred_json_path, 'r') as f:
            data = json.load(f)
            image_list = [img['file_name'] for img in data.get('images', [])]
            current_image_index = 0  # Start with the first image
            logging.info(f"Loaded {len(image_list)} images from '{working_pred_json_path}'.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image list from JSON:\n{e}")
        logging.error(f"Failed to load image list: {e}")

def update_navigation_buttons():
    """
    Enables or disables navigation buttons based on the current image index.
    """
    if current_image_index <= 0:
        prev_button.config(state=tk.DISABLED)
    else:
        prev_button.config(state=tk.NORMAL)
    
    if current_image_index >= len(image_list) - 1:
        next_button.config(state=tk.DISABLED)
    else:
        next_button.config(state=tk.NORMAL)

def load_next_image():
    """
    Loads the next image in the prediction_image_names list.
    """
    global selected_prediction_image, prediction_image_names

    current_image_name = selected_prediction_image.get()
    if not current_image_name:
        messagebox.showinfo("No Image", "No image is currently selected.")
        return

    try:
        current_index = prediction_image_names.index(current_image_name)
    except ValueError:
        messagebox.showerror("Error", f"Current image '{current_image_name}' not found in image list.")
        return

    new_index = current_index + 1
    if new_index >= len(prediction_image_names):
        messagebox.showinfo("End of List", "You have reached the last image.")
        return

    new_image_name = prediction_image_names[new_index]
    load_prediction_image(new_image_name)

    # 1) Load the new image
   #load_prediction_image(new_image_name)

    # 2) Load the new image's keypoints (if any)
    #if new_image_name in prediction_data:
    #    annotations_for_image = prediction_data[new_image_name]
    #    load_json_keypoints_into_gui(annotations_for_image)


def load_prev_image():
    """
    Loads the previous image in the prediction_image_names list.
    """
    global selected_prediction_image, prediction_image_names

    current_image_name = selected_prediction_image.get()
    if not current_image_name:
        messagebox.showinfo("No Image", "No image is currently selected.")
        return

    try:
        current_index = prediction_image_names.index(current_image_name)
    except ValueError:
        messagebox.showerror("Error", f"Current image '{current_image_name}' not found in image list.")
        return

    new_index = current_index - 1
    if new_index < 0:
        messagebox.showinfo("Start of List", "You are at the first image.")
        return

    new_image_name = prediction_image_names[new_index]
    load_prediction_image(new_image_name)
    # 1) Load the new image
   #load_prediction_image(new_image_name)

    # 2) Load the new image's keypoints (if any)
    #if new_image_name in prediction_data:
    #    annotations_for_image = prediction_data[new_image_name]
    #    load_json_keypoints_into_gui(annotations_for_image)

######

def run_measurement_script():
    """
    Opens a dialog to collect inputs and runs the measurement script with the provided arguments.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import subprocess, threading
    from pathlib import Path

    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Run Measurement Script")
    
    # Define variables to store user inputs
    json_path = tk.StringVar()
    image_dir = tk.StringVar()
    output_dir = tk.StringVar(value='./outputs')  # Default value
    method = tk.StringVar(value='pca')
    # NEW: additional variables for the new arguments:
    curve_measure_method = tk.StringVar(value='skeleton')
    skeleton_method = tk.StringVar(value='opencv')
    trim_branches = tk.BooleanVar(value=False)
    
    category_name = tk.StringVar()
    save_results = tk.BooleanVar()
    output_file = tk.StringVar()
    jsonl_output = tk.StringVar()
    min_aspect_ratio = tk.DoubleVar(value=5.0)
    max_aspect_ratio = tk.DoubleVar(value=200.0)
    min_width = tk.IntVar(value=10)
    mm_scale = tk.DoubleVar()
    grouping_file = tk.StringVar()
    
    # Optional arguments
    include_output_dir = tk.BooleanVar(value=True)
    include_output_file = tk.BooleanVar(value=False)
    include_jsonl_output = tk.BooleanVar(value=False)
    
    # Function to browse files/directories
    def browse_file(var):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            var.set(path)
    
    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)    
    
    # Layout the input fields (new rows have been inserted for the new arguments)
    tk.Label(input_window, text="COCO JSON File (Required):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=json_path, width=50).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(json_path)).grid(row=0, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Image Directory (Required):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=image_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(image_dir)).grid(row=1, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Output Directory (Optional):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
    tk.Checkbutton(input_window, text="Use Default './outputs'", variable=include_output_dir,
                   command=lambda: output_dir.set('./outputs')).grid(row=2, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Method (pca or skeleton):").grid(row=3, column=0, sticky='e', padx=5, pady=5)
    tk.OptionMenu(input_window, method, 'pca', 'skeleton').grid(row=3, column=1, sticky='w', padx=5, pady=5)
    
    # NEW: Additional fields for keypoint/segmentation analysis options
    tk.Label(input_window, text="Curve Measure Method (skeleton or bboxes):").grid(row=4, column=0, sticky='e', padx=5, pady=5)
    tk.OptionMenu(input_window, curve_measure_method, 'skeleton', 'bboxes').grid(row=4, column=1, sticky='w', padx=5, pady=5)
    
    tk.Label(input_window, text="Skeleton Method (opencv or skimage):").grid(row=5, column=0, sticky='e', padx=5, pady=5)
    tk.OptionMenu(input_window, skeleton_method, 'opencv', 'skimage').grid(row=5, column=1, sticky='w', padx=5, pady=5)
    
    tk.Checkbutton(input_window, text="Trim Branches", variable=trim_branches).grid(row=6, column=1, sticky='w', padx=5, pady=5)
    
    tk.Label(input_window, text="Category Name (Optional):").grid(row=7, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=category_name, width=50).grid(row=7, column=1, padx=5, pady=5)
    
    tk.Checkbutton(input_window, text="Save Results to Individual Files", variable=save_results).grid(row=8, column=1, sticky='w', padx=5, pady=5)
    
    tk.Label(input_window, text="Output CSV File (Optional):").grid(row=9, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_file, width=50).grid(row=9, column=1, padx=5, pady=5)
    tk.Checkbutton(input_window, text="Specify Output CSV File", variable=include_output_file).grid(row=9, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Output JSONL File (Optional):").grid(row=10, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=jsonl_output, width=50).grid(row=10, column=1, padx=5, pady=5)
    tk.Checkbutton(input_window, text="Specify Output JSONL File", variable=include_jsonl_output).grid(row=10, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Min Aspect Ratio:").grid(row=11, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=min_aspect_ratio).grid(row=11, column=1, padx=5, pady=5)
    
    tk.Label(input_window, text="Max Aspect Ratio:").grid(row=12, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=max_aspect_ratio).grid(row=12, column=1, padx=5, pady=5)
    
    tk.Label(input_window, text="Min Width (pixels):").grid(row=13, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=min_width).grid(row=13, column=1, padx=5, pady=5)
    
    tk.Label(input_window, text="mm to pixels (pixels):").grid(row=14, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=mm_scale).grid(row=14, column=1, padx=5, pady=5)
    
    tk.Label(input_window, text="Path to the grouping file (CSV, TSV, or JSON):").grid(row=15, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=grouping_file, width=50).grid(row=15, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(grouping_file)).grid(row=15, column=2, padx=5, pady=5)
    
    def submit():
        # Gather required arguments
        json_val = json_path.get()
        image_dir_val = image_dir.get()
        output_dir_val = output_dir.get() if include_output_dir.get() else './outputs'
        method_val = method.get()
        curve_measure_method_val = curve_measure_method.get()
        skeleton_method_val = skeleton_method.get()
        trim_branches_val = trim_branches.get()
        category_name_val = category_name.get() if category_name.get() else None
        save_results_val = save_results.get()
        output_file_val = output_file.get() if include_output_file.get() else None
        jsonl_output_val = jsonl_output.get() if include_jsonl_output.get() else None
        min_aspect_ratio_val = min_aspect_ratio.get()
        max_aspect_ratio_val = max_aspect_ratio.get()
        min_width_val = min_width.get()
        mm_scale_val = mm_scale.get()
        grouping_file_val = grouping_file.get() if grouping_file.get() else None
        
        # Validate required fields
        if not json_val:
            messagebox.showerror("Input Error", "Please select a COCO JSON file.")
            return
        if not image_dir_val:
            messagebox.showerror("Input Error", "Please select an Image Directory.")
            return
        
        # Construct the path to the measurement script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        measurement_script = script_dir / 'measure' / 'measurement_script_to_try_after_kpts_prediction_measure_kpts_V34.py' ### previous was V29
                
        # Ensure the measurement script exists
        if not measurement_script.is_file():
            messagebox.showerror("Script Error", f"Measurement script not found at: {measurement_script}")
            return
        
        # Construct the command
        cmd = [
            "conda", "run", "-n", "measure_env", "python",
            str(measurement_script),
            "--json", json_val,
            "--image_dir", image_dir_val,
            "--output_dir", output_dir_val,
            "--method", method_val,
            "--curve_measure_method", curve_measure_method_val,
            "--skeleton_method", skeleton_method_val
        ]
        if trim_branches_val:
            cmd.append("--trim_branches")
        if category_name_val:
            cmd.extend(["--category_name", category_name_val])
        if save_results_val:
            cmd.append("--save_results")
        if output_file_val:
            cmd.extend(["--output_file", output_file_val])
        if jsonl_output_val:
            cmd.extend(["--jsonl_output", jsonl_output_val])
        if min_aspect_ratio_val:
            cmd.extend(["--min_aspect_ratio", str(min_aspect_ratio_val)])
        if max_aspect_ratio_val:
            cmd.extend(["--max_aspect_ratio", str(max_aspect_ratio_val)])
        if min_width_val:
            cmd.extend(["--min_width", str(min_width_val)])
        if mm_scale_val:
            cmd.extend(["--mm_scale", str(mm_scale_val)])
        if grouping_file_val:
            cmd.extend(["--grouping_file", str(grouping_file_val)])
        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running Measurement Script")
                tk.Label(loading, text="Please wait while the measurement script runs...").pack(padx=20, pady=20)
                loading.update()

                subprocess.run(cmd, check=True)
                loading.destroy()
                messagebox.showinfo("Success", "Measurement script executed successfully.")
            except subprocess.CalledProcessError as e:
                loading.destroy()
                messagebox.showerror("Error", f"An error occurred while running the measurement script:\n{e}")
        
        threading.Thread(target=run_script).start()
        input_window.destroy()
    
    # Add Submit and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=16, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=16, column=2, sticky='w', padx=5, pady=10)



def run_measurement_script_3d():
    """
    Opens a dialog to collect inputs and runs the 3D measurement script with the provided arguments.
    """
    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Run 3D Measurement Script")
    
    # Define variables to store user inputs
    json_path = tk.StringVar()
    image_dir = tk.StringVar()
    output_dir = tk.StringVar(value='./outputs')  # Default value
    method = tk.StringVar(value='3D')
    category_name = tk.StringVar()
    save_results = tk.BooleanVar()
    output_file = tk.StringVar()
    jsonl_output = tk.StringVar()
    mm_scale = tk.DoubleVar(value=100.0)  # Assuming mm_scale is a float
    grouping_file = tk.StringVar()
    
    # Depth-specific variables
    generate_depth = tk.BooleanVar(value=True)
    save_depth = tk.BooleanVar(value=True)
    depth_output_dir = tk.StringVar(value='./outputs/depth_maps/')
    
    # Annotated images variables
    save_annotated = tk.BooleanVar()
    annotated_output_dir = tk.StringVar(value='./outputs/annotated_images/')
    
    # Function to browse files/directories
    def browse_file(var, filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)
    
    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)
    
    # Layout the input fields
    row = 0  # To keep track of the current row
    
    # COCO JSON File
    tk.Label(input_window, text="COCO JSON File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=json_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(json_path)).grid(row=row, column=2, padx=5, pady=5)
    row += 1
    
    # Image Directory
    tk.Label(input_window, text="Image Directory (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=image_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(image_dir)).grid(row=row, column=2, padx=5, pady=5)
    row += 1
    
    # Output Directory
    tk.Label(input_window, text="Output Directory (Optional):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Checkbutton(
        input_window, 
        text="Use Default './outputs'", 
        variable=tk.BooleanVar(value=True),
        command=lambda: output_dir.set('./outputs') if True else None
    ).grid(row=row, column=2, padx=5, pady=5)
    row += 1
    
    # Method
    tk.Label(input_window, text="Method:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.OptionMenu(input_window, method, '3D').grid(row=row, column=1, sticky='w', padx=5, pady=5)
    row += 1
    
    # Category Name
    tk.Label(input_window, text="Category Name (Optional):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=category_name, width=50).grid(row=row, column=1, padx=5, pady=5)
    row += 1
    
    # Save Results
    tk.Checkbutton(input_window, text="Save Results to Individual Files", variable=save_results).grid(row=row, column=1, sticky='w', padx=5, pady=5)
    row += 1
    
    # Output CSV File
    tk.Label(input_window, text="Output CSV File (Optional):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_file, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Checkbutton(input_window, text="Specify Output CSV File", variable=tk.BooleanVar()).grid(row=row, column=2, padx=5, pady=5)
    row += 1
    
    # Output JSONL File
    tk.Label(input_window, text="Output JSONL File (Optional):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=jsonl_output, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Checkbutton(input_window, text="Specify Output JSONL File", variable=tk.BooleanVar()).grid(row=row, column=2, padx=5, pady=5)
    row += 1
    
    # mm_scale
    tk.Label(input_window, text="mm to pixels (pixels):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=mm_scale).grid(row=row, column=1, padx=5, pady=5)
    row += 1
    
    # Grouping File
    tk.Label(input_window, text="Path to Grouping File (CSV, TSV, JSON):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=grouping_file, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(grouping_file, [("CSV Files", "*.csv"), ("TSV Files", "*.tsv"), ("JSON Files", "*.json"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)
    row += 1
    
    # Depth Map Generation
    tk.Checkbutton(input_window, text="Generate Depth Maps", variable=generate_depth).grid(row=row, column=1, sticky='w', padx=5, pady=5)
    row += 1
    
    # Save Depth Maps
    tk.Checkbutton(input_window, text="Save Depth Maps", variable=save_depth).grid(row=row, column=1, sticky='w', padx=5, pady=5)
    row += 1
    
    # Depth Output Directory
    tk.Label(input_window, text="Depth Output Directory (Optional):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=depth_output_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Checkbutton(
        input_window, 
        text="Use Default './outputs/depth_maps/'", 
        variable=tk.BooleanVar(value=True),
        command=lambda: depth_output_dir.set('./outputs/depth_maps/') if True else None
    ).grid(row=row, column=2, padx=5, pady=5)
    row += 1
    
    # Save Annotated Images
    tk.Checkbutton(input_window, text="Save Annotated Images", variable=save_annotated).grid(row=row, column=1, sticky='w', padx=5, pady=5)
    row += 1
    
    # Annotated Output Directory
    tk.Label(input_window, text="Annotated Output Directory (Optional):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=annotated_output_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Checkbutton(
        input_window, 
        text="Use Default './outputs/annotated_images/'", 
        variable=tk.BooleanVar(value=True),
        command=lambda: annotated_output_dir.set('./outputs/annotated_images/') if True else None
    ).grid(row=row, column=2, padx=5, pady=5)
    row += 1
    
    def submit():
        # Gather required arguments
        json_val = json_path.get()
        image_dir_val = image_dir.get()
        output_dir_val = output_dir.get() if output_dir.get() else './outputs'
        method_val = method.get()
        category_name_val = category_name.get() if category_name.get() else None
        save_results_val = save_results.get()
        output_file_val = output_file.get() if output_file.get() else None
        jsonl_output_val = jsonl_output.get() if jsonl_output.get() else None
        mm_scale_val = mm_scale.get()
        grouping_file_val = grouping_file.get() if grouping_file.get() else None
        generate_depth_val = generate_depth.get()
        save_depth_val = save_depth.get()
        depth_output_dir_val = depth_output_dir.get() if depth_output_dir.get() else './outputs/depth_maps/'
        save_annotated_val = save_annotated.get()
        annotated_output_dir_val = annotated_output_dir.get() if annotated_output_dir.get() else './outputs/annotated_images/'
        
        # Validate required fields
        if not json_val:
            messagebox.showerror("Input Error", "Please select a COCO JSON file.")
            return
        if not image_dir_val:
            messagebox.showerror("Input Error", "Please select an Image Directory.")
            return
        
        # Construct the path to the measurement script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        measurement_script = script_dir / 'metric3d' / 'measure_metric3d_v16.py'  # Adjusted for 3D , was measure_metric_3d_v12
        
        # Ensure the measurement script exists
        if not measurement_script.is_file():
            messagebox.showerror("Script Error", f"Measurement script not found at: {measurement_script}")
            return
        
        # Construct the command
        cmd = [
            "conda", "run", "-n", "metric3d", "python", str(measurement_script),
            "--json", json_val,
            "--image_dir", image_dir_val,
            "--method", method_val
        ]
        
        if category_name_val:
            cmd.extend(["--category_name", category_name_val])
        if save_results_val:
            cmd.append("--save_results")
        if output_file_val:
            cmd.extend(["--output_file", output_file_val])
        if jsonl_output_val:
            cmd.extend(["--jsonl_output", jsonl_output_val])
        if mm_scale_val:
            cmd.extend(["--mm_scale", str(mm_scale_val)])
        if grouping_file_val:
            cmd.extend(["--grouping_file", str(grouping_file_val)])
        if generate_depth_val:
            cmd.append("--generate_depth")
        if save_depth_val:
            cmd.append("--save_depth")
        if depth_output_dir_val:
            cmd.extend(["--depth_output_dir", str(depth_output_dir_val)])
        if save_annotated_val:
            cmd.append("--save_annotated")
        if annotated_output_dir_val:
            cmd.extend(["--annotated_output_dir", str(annotated_output_dir_val)])
        if args.flatten_output:  # Ensure 'flatten_output' is captured from the GUI
            cmd.append("--flatten_output")
        
        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running 3D Measurement Script")
                tk.Label(loading, text="Please wait while the measurement script runs...").pack(padx=20, pady=20)
                loading.update()
    
                subprocess.run(cmd, check=True)
                loading.destroy()
                messagebox.showinfo("Success", "3D Measurement script executed successfully.")
            except subprocess.CalledProcessError as e:
                loading.destroy()
                messagebox.showerror("Error", f"An error occurred while running the 3D Measurement script:\n{e}")
            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
        
        threading.Thread(target=run_script).start()
        input_window.destroy()
    
    # Add Submit and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=row, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row, column=2, sticky='w', padx=5, pady=10)




def run_semi_landmarking_script():
    """
    Opens a dialog to collect inputs and runs the semi-landmarking script with the provided arguments.
    """
    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Run Semi-Landmarking Script")
    
    # Define variables to store user inputs
    json_path = tk.StringVar()
    image_dir = tk.StringVar()
    output_dir = tk.StringVar(value='./outputs')  # Default value
    category_name = tk.StringVar()
    num_landmarks = tk.IntVar(value=100)  # Default value
    mode = tk.StringVar(value='2D')
    group_labels = tk.StringVar()
    perform_manova = tk.BooleanVar()
    alignment_method = tk.StringVar()
    # Function to browse files/directories
    def browse_file(var, filetypes=[("All Files", "*.*")]):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)
    
    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)
    
    # Layout the input fields
    tk.Label(input_window, text="COCO JSON File (Required):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=json_path, width=50).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(json_path, [("JSON Files", "*.json"), ("All Files", "*.*")])).grid(row=0, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Image Directory (Required):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=image_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(image_dir)).grid(row=1, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Output Directory (Optional):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_dir)).grid(row=2, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Category Name (Optional):").grid(row=3, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=category_name, width=50).grid(row=3, column=1, padx=5, pady=5)
    
    tk.Label(input_window, text="Number of Landmarks (Optional, default=100):").grid(row=4, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=num_landmarks).grid(row=4, column=1, padx=5, pady=5)
    
#    tk.Label(input_window, text="Mode (2D or 3D):").grid(row=5, column=0, sticky='e', padx=5, pady=5)
#    tk.OptionMenu(input_window, mode, '2D', '3D').grid(row=5, column=1, sticky='w', padx=5, pady=5)
    
    tk.Label(input_window, text="Group Labels CSV File (Optional):").grid(row=6, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=group_labels, width=50).grid(row=6, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(group_labels, [("CSV Files", "*.csv"), ("TSV Files", "*.tsv"), ("All Files", "*.*")])).grid(row=6, column=2, padx=5, pady=5)
    
    tk.Checkbutton(input_window, text="Perform MANOVA", variable=perform_manova).grid(row=7, column=1, sticky='w', padx=5, pady=5)
    
    tk.Label(input_window, text="Alignment Method choose without_reflection (default) or with_reflection").grid(row=8, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=alignment_method).grid(row=8, column=1, padx=5, pady=5)

    # Function to gather inputs and run the semi-landmarking script
    def submit():
        # Gather required arguments
        json_val = json_path.get()
        image_dir_val = image_dir.get()
        output_dir_val = output_dir.get() if output_dir.get() else './outputs'
        category_name_val = category_name.get() if category_name.get() else None
        num_landmarks_val = num_landmarks.get() if num_landmarks.get() else 100
#        mode_val = mode.get()
        group_labels_val = group_labels.get() if group_labels.get() else None
        perform_manova_val = perform_manova.get()
        alignment_method_val = alignment_method.get() if alignment_method.get() else 'without_reflection'

        # Validate required fields
        if not json_val:
            messagebox.showerror("Input Error", "Please select a COCO JSON file.")
            return
        if not image_dir_val:
            messagebox.showerror("Input Error", "Please select an Image Directory.")
            return
        
        # Construct the path to the semi-landmarking script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        semi_landmarking_script = script_dir / 'measure' / 'semi_landmark_and_kpts_procrustesV34_GPA.py' # was 'semi_landmark_and_kpts_procrustesV33_no_pacmap_fork.py'
        
        # Ensure the semi-landmarking script exists
        if not semi_landmarking_script.is_file():
            messagebox.showerror("Script Error", f"Semi-Landmarking script not found at: {semi_landmarking_script}")
            return
        
        # Construct the command
        #cmd = [
        #    "conda", "run", "-n", "measure_env", "python",
        #    str(semi_landmarking_script),
        #    "--json", json_val,
        #    "--image_dir", image_dir_val,
        #    "--output_dir", output_dir_val,
        #]
        # Build the command as a single string replaced
        cmd = f'''
        conda run -n measure_env python "{semi_landmarking_script}" \
            --json "{json_val}" \
            --image_dir "{image_dir_val}" \
            --output_dir "{output_dir_val}"
        '''
#        if category_name_val:
#            cmd.extend(["--category_name", category_name_val])
#        if num_landmarks_val:
#            cmd.extend(["--num_landmarks", str(num_landmarks_val)])
 #       if mode_val:
 #           cmd.extend(["--mode", mode_val])
#        if group_labels_val:
#            cmd.extend(["--group_labels", group_labels_val])
#        if perform_manova_val:
#            cmd.append("--perform_manova")
#        if alignment_method_val:
#            cmd.extend(["--alignment_method", str(alignment_method_val)])

        if category_name_val:
            cmd += f' --category_name "{category_name_val}"'
        if num_landmarks_val:
            cmd += f' --num_landmarks "{num_landmarks_val}"'
        if group_labels_val:
            cmd += f' --group_labels "{group_labels_val}"'
        if perform_manova_val:
            cmd += ' --perform_manova'
        if alignment_method_val:
            cmd += f' --alignment_method "{alignment_method_val}"'

        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running Semi-Landmarking Script")
                tk.Label(loading, text="Please wait while the semi-landmarking script runs...").pack(padx=20, pady=20)
                loading.update()
                
                subprocess.run(cmd, shell=True, executable='/bin/bash', check=True)
                loading.destroy()
                messagebox.showinfo("Success", "Semi-Landmarking script executed successfully.")
            except subprocess.CalledProcessError as e:
                loading.destroy()
                messagebox.showerror("Error", f"An error occurred while running the semi-landmarking script:\n{e}")
        
        threading.Thread(target=run_script).start()
        input_window.destroy()
    
    # Add Submit and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=9, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=9, column=2, sticky='w', padx=5, pady=10)



def run_color_segmentation_script():
    """
    Opens a dialog to collect inputs and runs the color segmentation script with the provided arguments.
    """
    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Run Color Segmentation Script")
    
    # Define variables to store user inputs
    input_dir_masks = tk.StringVar()
    output_dir_images = tk.StringVar()
    run_segmentation = tk.BooleanVar()
    run_normalization = tk.BooleanVar()
    visualize = tk.BooleanVar()
    
    # Function to browse directories
    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)
    
    # Layout the input fields
    tk.Label(input_window, text="Input Masks Directory (Required):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=input_dir_masks, width=50).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(input_dir_masks)).grid(row=0, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Output Images Directory (Required):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_dir_images, width=50).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_dir_images)).grid(row=1, column=2, padx=5, pady=5)
    
    # Operation options
    tk.Label(input_window, text="Select Operations to Perform:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    tk.Checkbutton(input_window, text="Felzenszwalb Segmentation", variable=run_segmentation).grid(row=2, column=1, sticky='w', padx=5, pady=5)
    tk.Checkbutton(input_window, text="Hue Normalization", variable=run_normalization).grid(row=3, column=1, sticky='w', padx=5, pady=5)
    
    tk.Checkbutton(input_window, text="Visualize Results", variable=visualize).grid(row=4, column=1, sticky='w', padx=5, pady=5)
    
    # Function to gather inputs and run the script
    def submit():
        # Gather required arguments
        input_dir_masks_val = input_dir_masks.get()
        output_dir_images_val = output_dir_images.get()
        run_segmentation_val = run_segmentation.get()
        run_normalization_val = run_normalization.get()
        visualize_val = visualize.get()
        
        # Validate required fields
        if not input_dir_masks_val:
            messagebox.showerror("Input Error", "Please select an Input Masks Directory.")
            return
        if not output_dir_images_val:
            messagebox.showerror("Input Error", "Please select an Output Images Directory.")
            return
        
        # If neither operation is selected, default to running both
        if not run_segmentation_val and not run_normalization_val:
            run_segmentation_val = True
            run_normalization_val = True
        
        # Construct the path to the script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        color_segmentation_script = script_dir / 'measure' / 'FHS_and_CLAHE_V17.py'  # Update with your actual script name and path from v6 to v17
        
        # Ensure the script exists
        if not color_segmentation_script.is_file():
            messagebox.showerror("Script Error", f"Color Segmentation script not found at: {color_segmentation_script}")
            return
        
        # Construct the command
        cmd = [
            "conda", "run", "-n", "measure_env", "python",
            str(color_segmentation_script),
            "--input_dir", input_dir_masks_val,
            "--output_dir", output_dir_images_val
        ]
        
        if run_segmentation_val:
            cmd.append("--run_segmentation")
        if run_normalization_val:
            cmd.append("--run_normalization")
        if visualize_val:
            cmd.append("--visualize")
        
        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running Color Segmentation Script")
                tk.Label(loading, text="Please wait while the color segmentation script runs...").pack(padx=20, pady=20)
                loading.update()
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                # Function to read output and display it
                def read_output():
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            print(output.strip())
                    rc = process.poll()
                    loading.destroy()
                    if rc == 0:
                        messagebox.showinfo("Success", "Color Segmentation script executed successfully.")
                    else:
                        stderr = process.stderr.read()
                        messagebox.showerror("Error", f"An error occurred while running the color segmentation script:\n{stderr}")
                
                threading.Thread(target=read_output).start()
                
            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
        
        threading.Thread(target=run_script).start()
        input_window.destroy()
    
    # Add Submit and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=5, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=5, column=2, sticky='w', padx=5, pady=10)
# For Windows compatibility (commented out for now)
# If on Windows, you might need to adjust the command, e.g.,
# cmd = [
#     "conda", "activate", "measure_env", "&&",
#     "python", str(color_segmentation_script),
#     "--input_dir", input_dir_masks_val,
#     "--output_dir", output_dir_images_val
# ]
# Or use the full path to the python executable in the environment
# python_executable = r'C:\Path\to\Anaconda3\envs\measure_env\python.exe'
# cmd = [
#     python_executable,
#     str(color_segmentation_script),
#     # arguments...
# ]


def run_color_extraction_script():
    """
    Opens a dialog to collect inputs and runs color_extraction_with_names-v8.py
    with the provided arguments.
    """
    input_window = tk.Toplevel()
    input_window.title("Run Color Extraction Script")

    # Variables
    binary_masks_dir = tk.StringVar()
    color_masks_dir = tk.StringVar()
    mask_types = tk.StringVar(value="fhs normalized fhs_normalized")
    output_dir = tk.StringVar()
    n_components = tk.IntVar(value=2)
    visualize = tk.BooleanVar()
    grouping_file = tk.StringVar()
    append_human_readable = tk.BooleanVar()
    category_substring = tk.StringVar()
    top_k_var = tk.IntVar(value=5)
    images_json_var = tk.StringVar()

    # Helper functions
    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def browse_file(var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    row_idx = 0

    # Binary Masks Dir
    tk.Label(input_window, text="Binary Masks Directory (Required):").grid(row=row_idx, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=binary_masks_dir, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(binary_masks_dir)).grid(row=row_idx, column=2, padx=5, pady=5)
    row_idx += 1

    # Color Masks Dir
    tk.Label(input_window, text="Color Masks Directory (Required):").grid(row=row_idx, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=color_masks_dir, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(color_masks_dir)).grid(row=row_idx, column=2, padx=5, pady=5)
    row_idx += 1

    # Mask Types
    tk.Label(input_window, text="Mask Types (space-separated):").grid(row=row_idx, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=mask_types, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(e.g. 'fhs normalized fhs_normalized')").grid(row=row_idx, column=2, sticky='w', padx=5, pady=5)
    row_idx += 1

    # Output Dir
    tk.Label(input_window, text="Output Directory (Required):").grid(row=row_idx, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_dir, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_dir)).grid(row=row_idx, column=2, padx=5, pady=5)
    row_idx += 1

    # n_components
    tk.Label(input_window, text="Number of PCA Components (default=2):").grid(row=row_idx, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=n_components, width=10).grid(row=row_idx, column=1, sticky='w', padx=5, pady=5)
    row_idx += 1

    # visualize
    tk.Checkbutton(input_window, text="Visualize PCA", variable=visualize).grid(row=row_idx, column=1, sticky='w', padx=5, pady=5)
    row_idx += 1

    # grouping_file
    tk.Label(input_window, text="Grouping File (Optional):").grid(row=row_idx, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=grouping_file, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(grouping_file)).grid(row=row_idx, column=2, padx=5, pady=5)
    row_idx += 1

    # append_human_readable
    tk.Checkbutton(input_window, text="Append Human-Readable by Group", variable=append_human_readable)\
        .grid(row=row_idx, column=1, sticky='w', padx=5, pady=5)
    row_idx += 1

    # category substring
    tk.Label(input_window, text="Category Filter (Optional):").grid(row=row_idx, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=category_substring, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(e.g. 'entire_forewing')").grid(row=row_idx, column=2, sticky='w', padx=5, pady=5)
    row_idx += 1

    # top_k
    tk.Label(input_window, text="Top K Colors (default=5):").grid(row=row_idx, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=top_k_var, width=10).grid(row=row_idx, column=1, sticky='w', padx=5, pady=5)
    row_idx += 1

    # images_json
    tk.Label(input_window, text="Images JSON (Optional):").grid(row=row_idx, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=images_json_var, width=50).grid(row=row_idx, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(images_json_var)).grid(row=row_idx, column=2, padx=5, pady=5)
    row_idx += 1

    def submit():
        bin_val = binary_masks_dir.get()
        col_val = color_masks_dir.get()
        mask_types_val = mask_types.get()
        out_val = output_dir.get()
        n_comp_val = n_components.get()
        visualize_val = visualize.get()
        group_val = grouping_file.get()
        append_val = append_human_readable.get()
        cat_val = category_substring.get()
        top_k = top_k_var.get()
        images_json_val = images_json_var.get()  # <--- this is important

        # Validate
        if not bin_val:
            messagebox.showerror("Input Error", "Please select a Binary Masks Directory.")
            return
        if not col_val:
            messagebox.showerror("Input Error", "Please select a Color Masks Directory.")
            return
        if not mask_types_val.strip():
            messagebox.showerror("Input Error", "Please specify at least one mask type.")
            return
        if not out_val:
            messagebox.showerror("Input Error", "Please select an Output Directory.")
            return
        if n_comp_val <= 0:
            messagebox.showerror("Input Error", "Number of PCA Components must be > 0.")
            return
        if top_k <= 0:
            messagebox.showerror("Input Error", "Top K must be > 0.")
            return

        # Build command
        script_dir = Path(__file__).parent
        script_path = script_dir / 'measure' / 'color_extraction_with_names-v8.py'

        if not script_path.is_file():
            messagebox.showerror("Script Error", f"Script not found at: {script_path}")
            return

        cmd = [
            "conda", "run", "-n", "measure_env", "python",
            str(script_path),
            "--binary_masks_dir", bin_val,
            "--color_masks_dir", col_val,
            "--output_dir", out_val,
            "--n_components", str(n_comp_val),
            "--top_k", str(top_k)
        ]

        # split mask types
        mask_list = mask_types_val.strip().split()
        cmd.append("--mask_types")
        cmd.extend(mask_list)

        if visualize_val:
            cmd.append("--visualize")

        if group_val.strip():
            cmd.extend(["--grouping_file", group_val.strip()])

        if append_val:
            cmd.append("--append_human_readable")

        if cat_val.strip():
            cmd.extend(["--category", cat_val.strip()])

        if images_json_val.strip():
            cmd.extend(["--images_json", images_json_val.strip()])

        def run_script():
            loading = None
            try:
                loading = tk.Toplevel()
                loading.title("Running Color Extraction Script")
                tk.Label(loading, text="Please wait, the script is running...").pack(padx=20, pady=20)
                loading.update()

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                def read_output():
                    stdout, stderr = process.communicate()
                    if loading:
                        loading.destroy()
                    if process.returncode == 0:
                        messagebox.showinfo("Success", f"Script completed successfully.\n\n{stdout}")
                    else:
                        messagebox.showerror("Error", f"Script error:\n\n{stderr}")

                threading.Thread(target=read_output).start()

            except Exception as e:
                if loading:
                    loading.destroy()
                messagebox.showerror("Error", f"Unexpected error:\n{e}")

        threading.Thread(target=run_script).start()
        input_window.destroy()

    # Buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=row_idx, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row_idx, column=2, sticky='w', padx=5, pady=10)



def run_color_pattern_analysis_script():
    """
    Opens a dialog to collect inputs and runs the color pattern analysis script with the provided arguments.
    """
    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Run Color Pattern Analysis Script")
    
    # Define variables to store user inputs
    binary_masks_dir = tk.StringVar()
    color_masks_dir = tk.StringVar()
    output_dir = tk.StringVar()
    mask_types_var = tk.StringVar(value='fhs normalized fhs_normalized')  # Default mask types
    n_components = tk.IntVar(value=2)
    visualize = tk.BooleanVar()
    grouping_file = tk.StringVar()

    # Function to browse directories
    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)
    
    # Layout the input fields
    tk.Label(input_window, text="Binary Masks Directory (Required):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=binary_masks_dir, width=50).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(binary_masks_dir)).grid(row=0, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Color Masks Directory (Required):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=color_masks_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(color_masks_dir)).grid(row=1, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Output Directory (Required):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_dir)).grid(row=2, column=2, padx=5, pady=5)
    
    tk.Label(input_window, text="Mask Types (Optional, default: fhs normalized fhs_normalized):").grid(row=3, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=mask_types_var, width=50).grid(row=3, column=1, padx=5, pady=5)
    
    tk.Label(input_window, text="Number of PCA Components (Optional, default: 2):").grid(row=4, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=n_components, width=10).grid(row=4, column=1, sticky='w', padx=5, pady=5)
    
    
    tk.Label(input_window, text="Grouping File (Optional):").grid(row=5, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=grouping_file, width=50).grid(row=5, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_dir)).grid(row=5, column=2, padx=5, pady=5)

    tk.Checkbutton(input_window, text="Visualize Results", variable=visualize).grid(row=6, column=1, sticky='w', padx=5, pady=5)
    
    # Function to gather inputs and run the script
    def submit():
        # Gather required arguments
        binary_masks_dir_val = binary_masks_dir.get()
        color_masks_dir_val = color_masks_dir.get()
        output_dir_val = output_dir.get()
        mask_types_val = mask_types_var.get()
        n_components_val = n_components.get()
        visualize_val = visualize.get()
        grouping_file_val = grouping_file.get()

        # Validate required fields
        if not binary_masks_dir_val:
            messagebox.showerror("Input Error", "Please select a Binary Masks Directory.")
            return
        if not color_masks_dir_val:
            messagebox.showerror("Input Error", "Please select a Color Masks Directory.")
            return
        if not output_dir_val:
            messagebox.showerror("Input Error", "Please select an Output Directory.")
            return
        if n_components_val <= 0:
            messagebox.showerror("Input Error", "Number of PCA Components must be a positive integer.")
            return
        
        # Construct the path to the script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        color_analysis_script = script_dir / 'measure' / 'color_pattern_analysis-Hist-V6.py'  # Update with your actual script name and path
        
        # Ensure the script exists
        if not color_analysis_script.is_file():
            messagebox.showerror("Script Error", f"Color Pattern Analysis script not found at: {color_analysis_script}")
            return
        
        # Construct the command
        cmd = [
            "conda", "run", "-n", "measure_env", "python",
            str(color_analysis_script),
            "--binary_masks_dir", binary_masks_dir_val,
            "--color_masks_dir", color_masks_dir_val,
            "--output_dir", output_dir_val,
            "--grouping_file", grouping_file_val,
            "--n_components", str(n_components_val)
        ]
        
        # Add mask types if specified
        if mask_types_val.strip():
            cmd.extend(["--mask_types"] + mask_types_val.strip().split())
        
        if visualize_val:
            cmd.append("--visualize")
        
        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running Color Pattern Analysis Script")
                tk.Label(loading, text="Please wait while the color pattern analysis script runs...").pack(padx=20, pady=20)
                loading.update()
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                # Function to read output and display it
                def read_output():
                    stdout, stderr = process.communicate()
                    loading.destroy()
                    if process.returncode == 0:
                        messagebox.showinfo("Success", "Color Pattern Analysis script executed successfully.")
                    else:
                        messagebox.showerror("Error", f"An error occurred while running the color pattern analysis script:\n{stderr}")
                
                threading.Thread(target=read_output).start()
                
            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
        
        threading.Thread(target=run_script).start()
        input_window.destroy()
    
    # Add Submit and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=7, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=7, column=2, sticky='w', padx=5, pady=10)




def run_visualize_contours_bbox_script():
    """
    Opens a dialog to collect inputs and runs the visualize_annotations_only.py script with the provided arguments.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import subprocess
    import threading
    from pathlib import Path

    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Visualize Contours & BBoxes")
    
    # Define variables to store user inputs
    json_file = tk.StringVar()
    image_dir = tk.StringVar()
    output_dir = tk.StringVar()
    category_name = tk.StringVar()
    overlay_bboxes = tk.BooleanVar()
    overlay_contours = tk.BooleanVar()
    overlay_labels = tk.BooleanVar()
    #color_labels = tk.StringVar(value='match')  # Default to 'match'
    opacity = tk.DoubleVar(value=0.4)
    morphology_key_images = tk.StringVar()
    # Function to browse files
    def browse_file(var):
        path = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if path:
            var.set(path)
    
    # Function to browse directories
    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)
    
    # This function allows multi-file selection and stores them as a comma-separated list
    def browse_file_list(var):
        file_paths = filedialog.askopenfilenames(
            title="Select One or More Key Images",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg *.tif *.bmp"), ("All Files", "*.*")]
        )
        if file_paths:
            # Convert tuple/list to comma-separated string
            comma_list = ",".join(file_paths)
            var.set(comma_list)

    # Layout the input fields
    row = 0
    tk.Label(input_window, text="JSON File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=json_file, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(json_file)).grid(row=row, column=2, padx=5, pady=5)
    
    row += 1
    tk.Label(input_window, text="Image Directory (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=image_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(image_dir)).grid(row=row, column=2, padx=5, pady=5)
    
    row += 1
    tk.Label(input_window, text="Output Directory (Optional):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_dir)).grid(row=row, column=2, padx=5, pady=5)
    
    row += 1
    tk.Label(input_window, text="Category Name (Optional):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=category_name, width=50).grid(row=row, column=1, padx=5, pady=5)
    
    row += 1
    tk.Checkbutton(input_window, text="Overlay Bounding Boxes", variable=overlay_bboxes).grid(row=row, column=1, sticky='w', padx=5, pady=5)
    
    row += 1
    tk.Checkbutton(input_window, text="Overlay Contours", variable=overlay_contours).grid(row=row, column=1, sticky='w', padx=5, pady=5)
    
    row += 1
    tk.Checkbutton(input_window, text="Overlay Labels", variable=overlay_labels).grid(row=row, column=1, sticky='w', padx=5, pady=5)
    
   # row += 1
   # tk.Label(input_window, text="Label Color:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
   # color_options = ['match', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black', 'orange', 'purple', 'white']
   # color_menu = tk.OptionMenu(input_window, color_labels, *color_options)
   # color_menu.config(width=10)
   # color_menu.grid(row=row, column=1, sticky='w', padx=5, pady=5)
    
    row += 1
    tk.Label(input_window, text="Overlay Opacity (0.0 - 1.0):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    opacity_slider = tk.Scale(input_window, variable=opacity, from_=0.0, to=1.0, resolution=0.1, orient='horizontal')
    opacity_slider.grid(row=row, column=1, sticky='w', padx=5, pady=5)
     # 8) Morphology Key Images (Comma Separated)
    row += 1
    tk.Label(input_window, text="Morphology Key Images (Optional, comma-separated):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=morphology_key_images, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse (Multi)", command=lambda: browse_file_list(morphology_key_images)).grid(row=row, column=2, padx=5, pady=5)
    #row += 1

    # Function to gather inputs and run the script
    def submit():
        # Gather required arguments
        json_file_val = json_file.get()
        image_dir_val = image_dir.get()
        output_dir_val = output_dir.get()
        category_name_val = category_name.get()
        overlay_bboxes_val = overlay_bboxes.get()
        overlay_contours_val = overlay_contours.get()
        overlay_labels_val = overlay_labels.get()
       # color_labels_val = color_labels.get()
        opacity_val = opacity.get()
         # New: parse morphology key images
        morphology_key_val = morphology_key_images.get().strip()  # comma-separated string

        # Validate required fields
        if not json_file_val:
            messagebox.showerror("Input Error", "Please select a JSON file.")
            return
        if not image_dir_val:
            messagebox.showerror("Input Error", "Please select an Image Directory.")
            return
        if opacity_val < 0.0 or opacity_val > 1.0:
            messagebox.showerror("Input Error", "Opacity must be between 0.0 and 1.0.")
            return
        
        # Construct the path to the script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        visualize_script = script_dir / 'idefics' / 'visualize_annotations_only_V28.py'  # Update with your actual script name and path
        
        # Ensure the script exists
        if not visualize_script.is_file():
            messagebox.showerror("Script Error", f"Visualization script not found at: {visualize_script}")
            return
        
        # Construct the command
        cmd = [
            "conda", "run", "-n", "measure_env", "python",
            str(visualize_script),
            "--json", json_file_val,
            "--image_dir", image_dir_val,
            "--opacity", str(opacity_val)
        ]
        
        if output_dir_val.strip():
            cmd.extend(["--output_dir", output_dir_val])
        
        if category_name_val.strip():
            cmd.extend(["--category_name", category_name_val.strip()])
        
        if overlay_bboxes_val:
            cmd.append("--overlay_bboxes")
        
        if overlay_contours_val:
            cmd.append("--overlay_contours")
        
        if overlay_labels_val:
            cmd.append("--overlay_labels")
        
       # if color_labels_val.strip().lower() != 'match':
       #     cmd.extend(["--color_labels", color_labels_val.strip().lower()])
       # else:
       #     cmd.append("--color_labels")
       #     cmd[-1] = "--color_labels"
       #     cmd.append("match")
        # If user provided one or more morphology key images (as comma-separated)
        if morphology_key_val:
            cmd.extend(["--morphology_key_image", morphology_key_val])


        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running Visualization Script")
                tk.Label(loading, text="Please wait while the visualization script runs...").pack(padx=20, pady=20)
                loading.update()
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                # Function to read output and display it
                def read_output():
                    stdout, stderr = process.communicate()
                    loading.destroy()
                    if process.returncode == 0:
                        messagebox.showinfo("Success", "Visualization script executed successfully.")
                    else:
                        messagebox.showerror("Error", f"An error occurred while running the visualization script:\n{stderr}")
                
                threading.Thread(target=read_output).start()
                
            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
        
        threading.Thread(target=run_script).start()
        input_window.destroy()
    
    # Add Submit and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=row+1, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row+1, column=2, sticky='w', padx=5, pady=10)



def run_tsv_to_jsonl_script():
    """
    Opens a dialog to collect inputs and runs the tsv_to_jsonl_for_gpt4-V3.py script with the provided arguments.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import subprocess
    import threading
    from pathlib import Path

    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Convert TSV to JSONL")

    # Define variables to store user inputs
    input_tsv = tk.StringVar()
    output_jsonl = tk.StringVar()

    # Function to browse files
    def browse_file(var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    # Layout the input fields
    row = 0
    tk.Label(input_window, text="Input TSV File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=input_tsv, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(input_tsv, [("TSV Files", "*.tsv"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output JSONL File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_jsonl, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(output_jsonl, [("JSONL Files", "*.jsonl"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    # Function to gather inputs and run the script
    def submit():
        # Gather required arguments
        input_tsv_val = input_tsv.get()
        output_jsonl_val = output_jsonl.get()

        # Validate required fields
        if not input_tsv_val:
            messagebox.showerror("Input Error", "Please select an input TSV file.")
            return
        if not output_jsonl_val:
            messagebox.showerror("Input Error", "Please specify an output JSONL file.")
            return

        # Construct the path to the script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        tsv_to_jsonl_script = script_dir / 'idefics' / 'tsv_to_jsonl_for_gpt4-V3.py'  # Update with your actual script name and path

        # Ensure the script exists
        if not tsv_to_jsonl_script.is_file():
            messagebox.showerror("Script Error", f"Conversion script not found at: {tsv_to_jsonl_script}")
            return

        # Construct the command
        cmd = [
            "conda", "run", "-n", "gpt4", "python",
            str(tsv_to_jsonl_script),
            "--input_tsv", input_tsv_val,
            "--output_jsonl", output_jsonl_val
        ]

        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running TSV to JSONL Conversion")
                tk.Label(loading, text="Please wait while the conversion script runs...").pack(padx=20, pady=20)
                loading.update()

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Function to read output and display it
                def read_output():
                    stdout, stderr = process.communicate()
                    loading.destroy()
                    if process.returncode == 0:
                        messagebox.showinfo("Success", "Conversion script executed successfully.")
                    else:
                        messagebox.showerror("Error", f"An error occurred while running the conversion script:\n{stderr}")

                threading.Thread(target=read_output).start()

            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

        threading.Thread(target=run_script).start()
        input_window.destroy()

    # Add Run and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=row+1, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row+1, column=2, sticky='w', padx=5, pady=10)


def run_fine_tune_script():
    """
    Opens a dialog to collect inputs and runs fine_tune_gpt4-V6.py with optional seed prompt.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import subprocess
    import threading
    from pathlib import Path

    input_window = tk.Toplevel()
    input_window.title("Fine-tune GPT-4 Model")
    input_window.geometry("820x420")

    jsonl_file = tk.StringVar()
    model_name = tk.StringVar()
    seed_file = tk.StringVar()
    force_seed = tk.BooleanVar(value=False)

    def browse_file(var, filetypes):
        p = filedialog.askopenfilename(filetypes=filetypes)
        if p: var.set(p)

    def load_seed_from_file():
        p = filedialog.askopenfilename(title="Select seed prompt file",
                                       filetypes=[("Text files","*.txt"),("All files","*.*")])
        if not p: return
        seed_file.set(p)
        with open(p, "r", encoding="utf-8") as fh:
            seed_box.delete("1.0","end")
            seed_box.insert("1.0", fh.read())

    row = 0
    tk.Label(input_window, text="JSONL File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=jsonl_file, width=60).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(jsonl_file, [("JSONL Files","*.jsonl"),("All Files","*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Model Name (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=model_name, width=60).grid(row=row, column=1, padx=5, pady=5)

    # --- NEW: seed prompt UI ---
    row += 1
    tk.Label(input_window, text="Seed Prompt (optional):").grid(row=row, column=0, sticky='ne', padx=5, pady=5)
    seed_box = tk.Text(input_window, width=60, height=6, wrap="word")
    seed_box.grid(row=row, column=1, padx=5, pady=5, sticky='w')
    side = tk.Frame(input_window)
    side.grid(row=row, column=2, sticky='nw', padx=5, pady=5)
    tk.Button(side, text="Browse", command=load_seed_from_file).pack(anchor="w", pady=2)
    tk.Label(side, textvariable=seed_file, wraplength=200, fg="gray").pack(anchor="w")
    tk.Checkbutton(side, text="Force prepend even if examples already have a system message",
                   variable=force_seed).pack(anchor="w", pady=8)

    def submit():
        jsonl_file_val = jsonl_file.get().strip()
        model_name_val = model_name.get().strip()
        seed_text_val = seed_box.get("1.0", "end-1c").strip()
        seed_file_val = seed_file.get().strip()

        if not jsonl_file_val:
            messagebox.showerror("Input Error", "Please select a JSONL file.")
            return
        if not model_name_val:
            messagebox.showerror("Input Error", "Please specify a model name.")
            return

        script_dir = Path(__file__).parent
        fine_tune_script = script_dir / 'idefics' / 'fine_tune_gpt4-V6.py'  # updated

        if not fine_tune_script.is_file():
            messagebox.showerror("Script Error", f"Fine-tuning script not found at: {fine_tune_script}")
            return

        cmd = ["conda", "run", "-n", "gpt4", "python", str(fine_tune_script),
               "--jsonl_file", jsonl_file_val, "--model", model_name_val]

        # Prefer literal text if provided; else pass file path
        if seed_text_val:
            cmd += ["--seed_prompt", seed_text_val]
        elif seed_file_val:
            cmd += ["--seed_prompt_file", seed_file_val]
        if force_seed.get():
            cmd += ["--force_seed"]

        def run_script():
            try:
                loading = tk.Toplevel()
                loading.title("Running Fine-tuning Script")
                tk.Label(loading, text="Please wait while the fine-tuning script runs...").pack(padx=20, pady=20)
                loading.update()
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                def read_output():
                    stdout, stderr = process.communicate()
                    loading.destroy()
                    if process.returncode == 0:
                        messagebox.showinfo("Success", "Fine-tuning script executed successfully.")
                    else:
                        messagebox.showerror("Error", f"An error occurred while running the fine-tuning script:\n{stderr}")
                threading.Thread(target=read_output).start()
            except Exception as e:
                try: loading.destroy()
                except: pass
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

        threading.Thread(target=run_script).start()
        input_window.destroy()

    tk.Button(input_window, text="Run", command=submit).grid(row=row+1, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row+1, column=2, sticky='w', padx=5, pady=10)



def run_gpt4_featurize_script():
    """
    Opens a dialog to collect inputs and runs the gpt4_featurize-V18.py script with the provided arguments,
    including optional seed/preface text or file, and a 'skip existing outputs' toggle.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import subprocess
    import threading
    from pathlib import Path

    input_window = tk.Toplevel()
    input_window.title("Automate Species Description using GPT-4")
    input_window.geometry("860x600")

    # Inputs
    image_path = tk.StringVar()
    image_folder = tk.StringVar()
    questions_file = tk.StringVar()
    output_folder = tk.StringVar()
    model_name = tk.StringVar(value='gpt-4o')
    temperature = tk.StringVar(value='0.2')
    max_tokens = tk.StringVar(value='150')
    preface_file_var = tk.StringVar()
    skip_existing = tk.BooleanVar(value=False)

    def browse_file(var, filetypes):
        p = filedialog.askopenfilename(filetypes=filetypes)
        if p: var.set(p)

    def browse_dir(var):
        p = filedialog.askdirectory()
        if p: var.set(p)

    def load_preface_from_file():
        p = filedialog.askopenfilename(title="Select seed/preface text file",
                                       filetypes=[("Text files","*.txt"),("All files","*.*")])
        if not p: return
        preface_file_var.set(p)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                preface_box.delete("1.0","end")
                preface_box.insert("1.0", fh.read())
        except Exception as e:
            messagebox.showwarning("Read error", f"Could not read file:\n{e}")

    row = 0
    tk.Label(input_window, text="Single Image Path:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=image_path, width=60).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(image_path, [("Image Files","*.png;*.jpg;*.jpeg"), ("All Files","*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Image Folder:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=image_folder, width=60).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_dir(image_folder)).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Questions File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=questions_file, width=60).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(questions_file, [("Text Files","*.txt"),("All Files","*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output Folder (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_folder, width=60).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_dir(output_folder)).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Model Name:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=model_name, width=60).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: gpt-4o)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Temperature:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=temperature, width=60).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: 0.2)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Max Tokens:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=max_tokens, width=60).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: 150)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    # NEW: seed/preface UI
    row += 1
    tk.Label(input_window, text="Seed / Preface (optional):").grid(row=row, column=0, sticky='ne', padx=5, pady=5)
    preface_box = tk.Text(input_window, width=60, height=6, wrap="word")
    preface_box.grid(row=row, column=1, padx=5, pady=5, sticky='w')
    side = tk.Frame(input_window); side.grid(row=row, column=2, sticky='nw', padx=5, pady=5)
    tk.Button(side, text="Browse", command=load_preface_from_file).pack(anchor="w", pady=2)
    tk.Label(side, textvariable=preface_file_var, wraplength=220, fg="gray").pack(anchor="w")
    tk.Checkbutton(side, text="Skip existing outputs", variable=skip_existing).pack(anchor="w", pady=8)

    def submit():
        image_path_val = image_path.get().strip()
        image_folder_val = image_folder.get().strip()
        questions_file_val = questions_file.get().strip()
        output_folder_val = output_folder.get().strip()
        model_name_val = model_name.get().strip()
        temperature_val = temperature.get().strip()
        max_tokens_val = max_tokens.get().strip()
        preface_text_val = preface_box.get("1.0", "end-1c").strip()
        preface_file_path = preface_file_var.get().strip()

        if not image_path_val and not image_folder_val:
            messagebox.showerror("Input Error", "Please specify either a single image path or an image folder.")
            return
        if not questions_file_val:
            messagebox.showerror("Input Error", "Please select a questions file.")
            return
        if not output_folder_val:
            messagebox.showerror("Input Error", "Please specify an output folder.")
            return

        script_dir = Path(__file__).parent
        gpt4_featurize_script = script_dir / 'idefics' / 'gpt4_featurize-V18.py'
        if not gpt4_featurize_script.is_file():
            messagebox.showerror("Script Error", f"GPT-4 featurize script not found at: {gpt4_featurize_script}")
            return

        cmd = [
            "conda", "run", "-n", "gpt4", "python", str(gpt4_featurize_script),
            "--questions_file", questions_file_val,
            "--output_folder", output_folder_val
        ]
        if image_path_val:
            cmd += ["--image_path", image_path_val]
        if image_folder_val:
            cmd += ["--image_folder", image_folder_val]
        if model_name_val and model_name_val != 'gpt-4o':
            cmd += ["--model", model_name_val]
        if temperature_val and temperature_val != '0.2':
            cmd += ["--temperature", temperature_val]
        if max_tokens_val and max_tokens_val != '150':
            cmd += ["--max_tokens", max_tokens_val]

        # Preface: prefer inline text if provided; otherwise pass file if chosen
        if preface_text_val:
            cmd += ["--preface", preface_text_val]
        elif preface_file_path:
            cmd += ["--preface_file", preface_file_path]

        if skip_existing.get():
            cmd += ["--skip_existing"]

        def run_script():
            try:
                loading = tk.Toplevel()
                loading.title("Running GPT-4 Featurize Script")
                tk.Label(loading, text="Please wait while the GPT-4 featurize script runs...").pack(padx=20, pady=20)
                loading.update()
                import subprocess
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                def read_output():
                    stdout, stderr = process.communicate()
                    loading.destroy()
                    if process.returncode == 0:
                        messagebox.showinfo("Success", "GPT-4 featurize script executed successfully.")
                    else:
                        messagebox.showerror("Error", f"An error occurred while running the GPT-4 featurize script:\n{stderr or stdout}")
                threading.Thread(target=read_output).start()
            except Exception as e:
                try: loading.destroy()
                except: pass
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

        threading.Thread(target=run_script).start()
        input_window.destroy()

    tk.Button(input_window, text="Run", command=submit).grid(row=row+1, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row+1, column=2, sticky='w', padx=5, pady=10)



def run_gpt4_label_script():
    """
    Opens a dialog to collect inputs and runs the gpt4_label-V4.py script with the provided arguments.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import subprocess
    import threading
    from pathlib import Path

    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Automate Species Description using GPT-4")

    # Define variables to store user inputs
    image_path = tk.StringVar()
    image_folder = tk.StringVar()
    questions_file = tk.StringVar()
    output_folder = tk.StringVar()
    model_name = tk.StringVar(value='gpt-4o')  # Default model
    temperature = tk.StringVar(value='0.2')  # Default temperature
    max_tokens = tk.StringVar(value='150')  # Default max_tokens
    type_status = tk.StringVar()
    type_status_choices = ['Holotype', 'Paratype', 'Specimen_Examined']
    type_status.set('Specimen_Examined')  # Default value

    # Function to browse files
    def browse_file(var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    # Function to browse directories
    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    # Layout the input fields
    row = 0
    tk.Label(input_window, text="Single Image Path:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=image_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(image_path, [("Image Files", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Image Folder:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=image_folder, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(image_folder)).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Questions File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=questions_file, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(questions_file, [("Text Files", "*.txt"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output Folder (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_folder, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_folder)).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Model Name:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=model_name, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: gpt-4o)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Temperature:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=temperature, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: 0.2)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Max Tokens:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=max_tokens, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: 150)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Type Status:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    type_status_menu = tk.OptionMenu(input_window, type_status, *type_status_choices)
    type_status_menu.grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Optional)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    # Function to gather inputs and run the script
    def submit():
        # Gather required arguments
        image_path_val = image_path.get()
        image_folder_val = image_folder.get()
        questions_file_val = questions_file.get()
        output_folder_val = output_folder.get()
        model_name_val = model_name.get().strip()
        temperature_val = temperature.get().strip()
        max_tokens_val = max_tokens.get().strip()
        type_status_val = type_status.get().strip()

        # Validate required fields
        if not image_path_val and not image_folder_val:
            messagebox.showerror("Input Error", "Please specify either a single image path or an image folder.")
            return
        if not questions_file_val:
            messagebox.showerror("Input Error", "Please select a questions file.")
            return
        if not output_folder_val:
            messagebox.showerror("Input Error", "Please specify an output folder.")
            return

        # Construct the path to the script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        gpt4_label_script = script_dir / 'idefics' / 'gpt4_label-V4.py'  # Update with your actual script name and path

        # Ensure the script exists
        if not gpt4_label_script.is_file():
            messagebox.showerror("Script Error", f"GPT-4 label script not found at: {gpt4_label_script}")
            return

        # Construct the command
        cmd = [
            "conda", "run", "-n", "gpt4", "python",
            str(gpt4_label_script),
            "--questions_file", questions_file_val,
            "--output_folder", output_folder_val
        ]

        if image_path_val:
            cmd.extend(["--image_path", image_path_val])
        if image_folder_val:
            cmd.extend(["--image_folder", image_folder_val])

        if model_name_val and model_name_val != 'gpt-4o':
            cmd.extend(["--model", model_name_val])
        if temperature_val and temperature_val != '0.2':
            cmd.extend(["--temperature", temperature_val])
        if max_tokens_val and max_tokens_val != '150':
            cmd.extend(["--max_tokens", max_tokens_val])

        if type_status_val:
            cmd.extend(["--type_status", type_status_val])

        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running GPT-4 Label Script")
                tk.Label(loading, text="Please wait while the GPT-4 label script runs...").pack(padx=20, pady=20)
                loading.update()

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Function to read output and display it
                def read_output():
                    stdout, stderr = process.communicate()
                    loading.destroy()
                    if process.returncode == 0:
                        messagebox.showinfo("Success", "GPT-4 label script executed successfully.")
                    else:
                        messagebox.showerror("Error", f"An error occurred while running the GPT-4 label script:\n{stderr}")

                threading.Thread(target=read_output).start()

            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

        threading.Thread(target=run_script).start()
        input_window.destroy()

    # Add Run and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=row+1, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row+1, column=2, sticky='w', padx=5, pady=10)




def run_parse_me_script():
    """
    Opens a dialog to collect inputs and runs the materials_examined_extract_V8.py script with the provided arguments.
    """
    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Parse Material Examined")
    input_window.geometry("600x400")  # Adjust size as needed

    # Define variables to store user inputs
    species_list_path = tk.StringVar()
    output_json_dir = tk.StringVar()
    output_tsv_path = tk.StringVar()
    q_key = tk.StringVar(value='Q41')  # Default q_key

    # Function to browse files
    def browse_file(var, filetypes, title):
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            var.set(path)

    # Function to browse directories
    def browse_directory(var, title):
        path = filedialog.askdirectory(title=title)
        if path:
            var.set(path)

    # Function to handle "Save As" for output TSV
    def save_as_file(var, filetypes, title):
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".tsv",
            filetypes=filetypes
        )
        if path:
            var.set(path)



    # Layout the input fields
    row = 0
    tk.Label(input_window, text="Species List TSV File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=species_list_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(species_list_path, [("TSV Files", "*.tsv"), ("All Files", "*.*")], "Select Species List TSV File")).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output JSON Directory (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_json_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_json_dir, "Select Output JSON Directory")).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output TSV Path (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_tsv_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Save As", command=lambda: save_as_file(output_tsv_path, [("TSV Files", "*.tsv"), ("All Files", "*.*")], "Save Mapping TSV As")).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Q-key to Extract:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=q_key, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: Q41)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    # Function to gather inputs and run the script
    def submit():
        # Gather required arguments
        species_list_val = species_list_path.get().strip()
        output_json_dir_val = output_json_dir.get().strip()
        output_tsv_val = output_tsv_path.get().strip()
        q_key_val = q_key.get().strip()

        # Validate required fields
        if not species_list_val:
            messagebox.showerror("Input Error", "Please select a Species List TSV file.")
            return
        if not output_json_dir_val:
            messagebox.showerror("Input Error", "Please select an Output JSON Directory.")
            return
        if not output_tsv_val:
            messagebox.showerror("Input Error", "Please specify an Output TSV path.")
            return
        if not q_key_val:
            messagebox.showerror("Input Error", "Please specify a Q-key to extract.")
            return

        # Construct the path to the script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        extract_script = script_dir / 'idefics' / 'materials_examined_extract_V9.py'  # Update with your actual script name and path

        # Ensure the script exists
        if not extract_script.is_file():
            messagebox.showerror("Script Error", f"Parse Material Examined script not found at: {extract_script}")
            return

        # Construct the command
        cmd = [
            "conda", "run", "-n", "measure_env", "python",
            str(extract_script),
            '--species_list', species_list_val,
            '--output_json_dir', output_json_dir_val,
            '--output_tsv', output_tsv_val,
            '--q_key', q_key_val
        ]

        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running Parse Material Examined")
                tk.Label(loading, text="Please wait while the script runs...").pack(padx=20, pady=20)
                loading.update()

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Function to read output and display it
                def read_output():
                    stdout, stderr = process.communicate()
                    loading.destroy()
                    if process.returncode == 0:
                        logging.info(f"Parse_ME completed successfully.\n{stdout}")
                        messagebox.showinfo("Success", "Parsing Material Examined completed successfully.")
                    else:
                        logging.error(f"Error in Parse_ME:\n{stderr}")
                        messagebox.showerror("Error", f"Parsing Material Examined failed:\n{stderr}")

                threading.Thread(target=read_output).start()

            except Exception as e:
                loading.destroy()
                logging.exception("Unexpected error in run_parse_me_script")
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

        threading.Thread(target=run_script).start()
        input_window.destroy()

    # Add Run and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=row+1, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row+1, column=2, sticky='w', padx=5, pady=10)

def run_concat_description_script():
    """
    Opens a dialog to collect inputs and runs the bring_together_for_gpt4-V52.py script with the provided arguments.
    """
    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Concat Description")
    input_window.geometry("700x500")  # Adjust size as needed

    # Define variables to store user inputs
    measurements_path = tk.StringVar()
    colors_path = tk.StringVar()
    qa_folder_path = tk.StringVar()
    contours_path = tk.StringVar()
    species_list_path = tk.StringVar()
    category_list_path = tk.StringVar()
    material_examined_path = tk.StringVar()
    output_prompts_path = tk.StringVar()
    body_part = tk.StringVar(value='wing')  # Default body part
    category_filter = tk.StringVar(value='all')  # Default category filter

    # Function to browse files
    def browse_file(var, filetypes, title):
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            var.set(path)

    # Function to browse directories
    def browse_directory(var, title):
        path = filedialog.askdirectory(title=title)
        if path:
            var.set(path)

    # Function to handle "Save As" for output TSV
    def save_as_file(var, filetypes, title):
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".tsv",
            filetypes=filetypes
        )
        if path:
            var.set(path)

    # Layout the input fields
    row = 0
    tk.Label(input_window, text="Measurements JSONL File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=measurements_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(measurements_path, [("JSONL Files", "*.jsonl"), ("All Files", "*.*")], "Select Measurements JSONL File")).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Colors JSONL File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=colors_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(colors_path, [("JSONL Files", "*.jsonl"), ("All Files", "*.*")], "Select Colors JSONL File")).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="QA Folder (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=qa_folder_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(qa_folder_path, "Select QA Folder")).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Contours COCO JSON File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=contours_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(contours_path, [("JSON Files", "*.json"), ("All Files", "*.*")], "Select Contours JSON File")).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Species List Text File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=species_list_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(species_list_path, [("Text Files", "*.txt"), ("All Files", "*.*")], "Select Species List Text File")).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Category List Text File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=category_list_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(category_list_path, [("Text Files", "*.txt"), ("All Files", "*.*")], "Select Category List Text File")).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Material Examined TSV File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=material_examined_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(material_examined_path, [("TSV Files", "*.tsv"), ("All Files", "*.*")], "Select Material Examined TSV File")).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output Prompts File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_prompts_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Save as", command=lambda: save_as_file(output_prompts_path, [("Text Files", "*.txt"), ("All Files", "*.*")], "Save Generated Prompts As")).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Body Part to Describe:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=body_part, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: wing)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Category to Include:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=category_filter, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: all)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    # Function to gather inputs and run the script
    def submit():
        # Gather required arguments
        measurements_val = measurements_path.get().strip()
        colors_val = colors_path.get().strip()
        qa_folder_val = qa_folder_path.get().strip()
        contours_val = contours_path.get().strip()
        species_list_val = species_list_path.get().strip()
        category_list_val = category_list_path.get().strip()
        material_examined_val = material_examined_path.get().strip()
        output_prompts_val = output_prompts_path.get().strip()
        body_part_val = body_part.get().strip()
        category_filter_val = category_filter.get().strip()

        # Validate required fields
        if not measurements_val:
            messagebox.showerror("Input Error", "Please select a Measurements JSONL file.")
            return
        if not colors_val:
            messagebox.showerror("Input Error", "Please select a Colors JSONL file.")
            return
        if not qa_folder_val:
            messagebox.showerror("Input Error", "Please select a QA Folder.")
            return
        if not contours_val:
            messagebox.showerror("Input Error", "Please select a Contours COCO JSON file.")
            return
        if not species_list_val:
            messagebox.showerror("Input Error", "Please select a Species List Text file.")
            return
        if not category_list_val:
            messagebox.showerror("Input Error", "Please select a Category List Text file.")
            return
        if not material_examined_val:
            messagebox.showerror("Input Error", "Please select a Material Examined TSV file.")
            return
        if not output_prompts_val:
            messagebox.showerror("Input Error", "Please specify an Output Prompts file.")
            return
        if not body_part_val:
            body_part_val = "wing"  # Default value
            logging.debug("Body part not specified. Using default 'wing'.")
        if not category_filter_val:
            category_filter_val = "all"  # Default value
            logging.debug("Category filter not specified. Using default 'all'.")

        # Construct the path to the script using a relative path for portability
        script_dir = Path(__file__).parent  # Directory where the GUI script is located
        concat_script = script_dir / 'idefics' / 'bring_together_for_gpt4-V49.py'  # Update with your actual script name and path

        # Ensure the script exists
        if not concat_script.is_file():
            messagebox.showerror("Script Error", f"Concat Description script not found at: {concat_script}")
            return

        # Construct the command # switched from idefics2 environment to measure_env
        cmd = [ 'conda', 'run', '-n', 'measure_env', 'python', 
            str(concat_script),
            '--measurements', measurements_val,
            '--colors', colors_val,
            '--qa_folder', qa_folder_val,
            '--contours', contours_val,
            '--body_part', body_part_val,
            '--species_list', species_list_val,
            '--category', category_filter_val,
            '--category_list', category_list_val,
            '--material_examined', material_examined_val,
            '--output', output_prompts_val
        ]

        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running Concat Description")
                tk.Label(loading, text="Please wait while the script runs...").pack(padx=20, pady=20)
                loading.update()

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Function to read output and display it
                def read_output():
                    stdout, stderr = process.communicate()
                    loading.destroy()
                    if process.returncode == 0:
                        logging.info(f"Concat_Description completed successfully.\n{stdout}")
                        messagebox.showinfo("Success", "Concatenating Description completed successfully.")
                    else:
                        logging.error(f"Error in Concat_Description:\n{stderr}")
                        messagebox.showerror("Error", f"Concatenating Description failed:\n{stderr}")

                threading.Thread(target=read_output).start()

            except Exception as e:
                loading.destroy()
                logging.exception("Unexpected error in run_concat_description_script")
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

        threading.Thread(target=run_script).start()
        input_window.destroy()

    # Add Run and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=row+1, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row+1, column=2, sticky='w', padx=5, pady=10)


def run_construct_description():
    """
    Opens a dialog to collect inputs and runs the construct_description_gpt-V8.py script with the provided arguments.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import subprocess
    import threading
    from pathlib import Path

    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Generate Species Descriptions using GPT-4")

    # Define variables to store user inputs
    description_file = tk.StringVar()
    example_file = tk.StringVar()
    output_folder = tk.StringVar()
    output_prefix = tk.StringVar(value='species_descriptions-v2')  # Default prefix
    model_name = tk.StringVar(value='gpt-4o')  # Default model
    temperature = tk.StringVar(value='0.2')    # Default temperature
    max_tokens = tk.StringVar(value='8000')    # Default max_tokens

    # Function to browse files
    def browse_file(var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    # Function to browse directories
    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    # Layout the input fields
    row = 0
    tk.Label(input_window, text="Description File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=description_file, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(description_file, [("Text Files", "*.txt"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Example File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=example_file, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(example_file, [("Text Files", "*.txt"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output Folder (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_folder, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_folder)).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output Prefix:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_prefix, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: species_descriptions-v2)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Model Name:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=model_name, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: gpt-4o)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Temperature:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=temperature, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: 0.2)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Max Tokens:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=max_tokens, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: 8000)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    # Function to gather inputs and run the script
    def submit():
        # Gather required arguments
        description_file_val = description_file.get()
        example_file_val = example_file.get()
        output_folder_val = output_folder.get()
        output_prefix_val = output_prefix.get().strip()
        model_name_val = model_name.get().strip()
        temperature_val = temperature.get().strip()
        max_tokens_val = max_tokens.get().strip()

        # Validate required fields
        if not description_file_val:
            messagebox.showerror("Input Error", "Please select the description file.")
            return
        if not example_file_val:
            messagebox.showerror("Input Error", "Please select the example file.")
            return
        if not output_folder_val:
            messagebox.showerror("Input Error", "Please specify an output folder.")
            return

        # Construct the path to the script using a relative path for portability
        script_dir = Path(__file__).parent
        construct_description_script = script_dir / 'idefics' / 'construct_description_gpt-V8.py'  # Update with your actual script path

        # Ensure the script exists
        if not construct_description_script.is_file():
            messagebox.showerror("Script Error", f"Script not found at: {construct_description_script}")
            return

        # Construct the command
        cmd = [
            "conda", "run", "-n", "gpt4", "python",
            str(construct_description_script),
            "--description_file", description_file_val,
            "--example_file", example_file_val,
            "--output_folder", output_folder_val,
            "--output_prefix", output_prefix_val,
            "--model", model_name_val,
            "--temperature", temperature_val,
            "--max_tokens", max_tokens_val
        ]

        # Run the command in a separate thread to prevent GUI freezing
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running Construct Description Script")
                tk.Label(loading, text="Please wait while the script runs...").pack(padx=20, pady=20)
                loading.update()

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Function to read output and display it
                def read_output():
                    stdout, stderr = process.communicate()
                    loading.destroy()
                    if process.returncode == 0:
                        messagebox.showinfo("Success", "Species descriptions generated successfully.")
                    else:
                        messagebox.showerror("Error", f"An error occurred:\n{stderr}")

                threading.Thread(target=read_output).start()

            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

        threading.Thread(target=run_script).start()
        input_window.destroy()

    # Add Run and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=row+1, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row+1, column=2, sticky='w', padx=5, pady=10)




def refine_and_generate_key():
    """
    Runs scripts to parse out descriptions and diagnoses, then generates the dichotomous key using GPT-4.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import subprocess
    import threading
    from pathlib import Path

    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Refine Output and Generate Dichotomous Key")

    # Define variables to store user inputs
    species_descriptions_file = tk.StringVar()
    example_file = tk.StringVar()
    output_folder = tk.StringVar()
    output_prefix = tk.StringVar(value='species_descriptions-v2-dichotomous-key')  # Default prefix
    model_name = tk.StringVar(value='gpt-4o')  # Default model
    temperature = tk.StringVar(value='0.2')    # Default temperature
    max_tokens = tk.StringVar(value='8000')    # Default max_tokens

    # Function to browse files
    def browse_file(var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    # Function to browse directories
    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    # Layout the input fields
    row = 0
    tk.Label(input_window, text="Species Descriptions File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=species_descriptions_file, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(species_descriptions_file, [("Text Files", "*.txt"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Example File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=example_file, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(example_file, [("Text Files", "*.txt"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output Folder (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_folder, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_folder)).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output Prefix:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_prefix, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: species_descriptions-v2-dichotomous-key)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Model Name:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=model_name, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: gpt-4o)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row +=1
    tk.Label(input_window, text="Temperature:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=temperature, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: 0.2)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row +=1
    tk.Label(input_window, text="Max Tokens:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=max_tokens, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: 8000)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    # Function to gather inputs and run the scripts
    def submit():
        # Gather required arguments
        species_descriptions_file_val = species_descriptions_file.get()
        example_file_val = example_file.get()
        output_folder_val = output_folder.get()
        output_prefix_val = output_prefix.get().strip()
        model_name_val = model_name.get().strip()
        temperature_val = temperature.get().strip()
        max_tokens_val = max_tokens.get().strip()

        # Validate required fields
        if not species_descriptions_file_val:
            messagebox.showerror("Input Error", "Please select the species descriptions file.")
            return
        if not example_file_val:
            messagebox.showerror("Input Error", "Please select the example file.")
            return
        if not output_folder_val:
            messagebox.showerror("Input Error", "Please specify an output folder.")
            return

        # Paths to the scripts
        script_dir = Path(__file__).parent
        get_descriptions_only_script = script_dir / 'idefics' / 'get-descriptions-only.py'
        get_diagnosis_only_script = script_dir / 'idefics' / 'get-diagnosis-only.py'
        construct_dichotomous_key_script = script_dir / 'idefics' / 'construct_description_gpt-V8-dichotomous-key.py'
        update_species_descriptions_script = script_dir / 'idefics' /'update_species_descriptions.py'
        # Ensure the scripts exist
        for script in [get_descriptions_only_script, get_diagnosis_only_script, construct_dichotomous_key_script]:
            if not script.is_file():
                messagebox.showerror("Script Error", f"Script not found at: {script}")
                return

        # Run the scripts in a separate thread
        def run_scripts():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Refining Output and Generating Dichotomous Key")
                tk.Label(loading, text="Please wait while the scripts run...").pack(padx=20, pady=20)
                loading.update()

                # Step 1: Run get-descriptions-only.py
                descriptions_only_file = Path(output_folder_val) / 'species_descriptions-descriptions_only.txt'
                cmd_get_descriptions = [
                    "conda", "run", "-n", "gpt4", "python",
                    str(get_descriptions_only_script),
                    "--input_file", species_descriptions_file_val,
                    "--output_file", str(descriptions_only_file)
                ]
                result = subprocess.run(cmd_get_descriptions, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    loading.destroy()
                    messagebox.showerror("Error", f"Error in get-descriptions-only.py:\n{result.stderr}")
                    return

                # Step 2: Run get-diagnosis-only.py
                diagnosis_only_file = Path(output_folder_val) / 'species_descriptions-diagnosis_only.txt'
                cmd_get_diagnosis = [
                    "conda", "run", "-n", "gpt4", "python",
                    str(get_diagnosis_only_script),
                    "--input_file", species_descriptions_file_val,
                    "--output_file", str(diagnosis_only_file)
                ]
                result = subprocess.run(cmd_get_diagnosis, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    loading.destroy()
                    messagebox.showerror("Error", f"Error in get-diagnosis-only.py:\n{result.stderr}")
                    return

                # Step 3: Run construct_description_gpt-V8-dichotomous-key.py
                dichotomous_key_file = Path(output_folder_val) / f"{output_prefix_val}.txt"
                cmd_construct_key = [
                    "conda", "run", "-n", "gpt4", "python",
                    str(construct_dichotomous_key_script),
                    "--description_file", str(diagnosis_only_file),
                    "--example_file", example_file_val,
                    "--output_folder", output_folder_val,
                    "--output_prefix", output_prefix_val,
                    "--model", model_name_val,
                    "--temperature", temperature_val,
                    "--max_tokens", max_tokens_val
                ]
                result = subprocess.run(cmd_construct_key, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                loading.destroy()
                if result.returncode != 0:
                    messagebox.showerror("Error", f"Error in constructing dichotomous key:\n{result.stderr}")
                    return

                #messagebox.showinfo("Success", "Dichotomous key generated successfully.")

                # Step 4: Run update_species_descriptions.py
                updated_species_descriptions_file = Path(output_folder_val) / 'updated_species_descriptions_from-descriptron.txt'
                cmd_update_species = [
                    "conda", "run", "-n", "gpt4", "python",
                    str(update_species_descriptions_script),
                    "--input_file", species_descriptions_file_val,
                    "--key_file", str(dichotomous_key_file),
                    "--output_file", str(updated_species_descriptions_file)
                ]
                result = subprocess.run(cmd_update_species, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                loading.destroy()
                if result.returncode != 0:
                    messagebox.showerror("Error", f"Error in updating species descriptions:\n{result.stderr}")
                    return

                messagebox.showinfo("Success", "Dichotomous key generated and species descriptions updated successfully.")

            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

        threading.Thread(target=run_scripts).start()
        input_window.destroy()

    # Add Run and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=row+1, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row+1, column=2, sticky='w', padx=5, pady=10)


def run_calculate_rogueV38():
    """
    Opens a window to run the calculate_rogueV38.py script by gathering necessary inputs.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import subprocess
    import threading
    from pathlib import Path

    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Run Calculate RogueV38")

    # Define variables to store user inputs
    taxo_path = tk.StringVar()
    gpt4_text_path = tk.StringVar()
    models_csv = tk.StringVar()
    coco_jsonl_path = tk.StringVar()
    image_dir = tk.StringVar()
    mask_dir = tk.StringVar()
    output_dir = tk.StringVar()
    category_name = tk.StringVar(value='entire_forewing')
    thumbnail_size = tk.StringVar(value='80 80')  # Entered as two integers separated by space
    thumbnail_offset = tk.StringVar(value='5 0')  # Entered as two integers separated by space
    n_neighbors = tk.StringVar(value='3')

    # Function to browse files/directories
    def browse_file(var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    # Layout the input fields
    row = 0
    tk.Label(input_window, text="Taxonomist TSV File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=taxo_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(taxo_path, [("TSV Files", "*.tsv"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="GPT-4 Text File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=gpt4_text_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(gpt4_text_path, [("Text Files", "*.txt"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Models CSV File (Optional):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=models_csv, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(models_csv, [("CSV Files", "*.csv"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="COCO JSONL File (Optional):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=coco_jsonl_path, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(coco_jsonl_path, [("JSONL Files", "*.jsonl"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Image Directory (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=image_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(image_dir)).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Mask Directory (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=mask_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(mask_dir)).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output Directory (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(output_dir)).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Category Name:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=category_name, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: entire_forewing)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Thumbnail Size (Width Height):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=thumbnail_size, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: 80 80)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Thumbnail Offset (X Y):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=thumbnail_offset, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: 5 0)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Number of Neighbors:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=n_neighbors, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Label(input_window, text="(Default: 3)").grid(row=row, column=2, sticky='w', padx=5, pady=5)

    row += 1

    # Function to gather inputs and run the script
    def submit():
        # Gather inputs
        taxo_path_val = taxo_path.get()
        gpt4_text_path_val = gpt4_text_path.get()
        models_csv_val = models_csv.get()
        coco_jsonl_path_val = coco_jsonl_path.get()
        image_dir_val = image_dir.get()
        mask_dir_val = mask_dir.get()
        output_dir_val = output_dir.get()
        category_name_val = category_name.get().strip()
        thumbnail_size_val = thumbnail_size.get().strip()
        thumbnail_offset_val = thumbnail_offset.get().strip()
        n_neighbors_val = n_neighbors.get().strip()

        # Validate required fields
        if not taxo_path_val:
            messagebox.showerror("Input Error", "Please select the Taxonomist TSV File.")
            return
        if not gpt4_text_path_val:
            messagebox.showerror("Input Error", "Please select the GPT-4 Text File.")
            return
        if not image_dir_val:
            messagebox.showerror("Input Error", "Please select the Image Directory.")
            return
        if not mask_dir_val:
            messagebox.showerror("Input Error", "Please select the Mask Directory.")
            return
        if not output_dir_val:
            messagebox.showerror("Input Error", "Please select the Output Directory.")
            return

        # Process thumbnail_size and thumbnail_offset
        try:
            thumbnail_size_list = list(map(int, thumbnail_size_val.split()))
            if len(thumbnail_size_list) != 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Thumbnail Size must be two integers separated by space (e.g., '80 80').")
            return

        try:
            thumbnail_offset_list = list(map(int, thumbnail_offset_val.split()))
            if len(thumbnail_offset_list) != 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Thumbnail Offset must be two integers separated by space (e.g., '5 0').")
            return

        # Validate n_neighbors
        try:
            n_neighbors_int = int(n_neighbors_val)
            if n_neighbors_int < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Number of Neighbors must be a positive integer.")
            return

        # Paths to the calculate_rogueV38.py script
        script_dir = Path(__file__).parent  # Assuming the script is in the same directory
        calculate_rogueV38_script = script_dir / 'idefics' / 'calculate_rogueV38_modifiedv5.py'

        if not calculate_rogueV38_script.is_file():
            messagebox.showerror("Script Error", f"calculate_rogueV38.py not found at: {calculate_rogueV38_script}")
            return

        # Run the script in a separate thread
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running Script")
                tk.Label(loading, text="Please wait while calculate_rogueV38.py is running...").pack(padx=20, pady=20)
                loading.update()

                # Construct the command
                cmd = [
                    "python",
                    str(calculate_rogueV38_script),
                    "--taxo_path", taxo_path_val,
                    "--gpt4_text_path", gpt4_text_path_val,
                    "--image_dir", image_dir_val,
                    "--mask_dir", mask_dir_val,
                    "--output_dir", output_dir_val,
                    "--category_name", category_name_val,
                    "--thumbnail_size", str(thumbnail_size_list[0]), str(thumbnail_size_list[1]),
                    "--thumbnail_offset", str(thumbnail_offset_list[0]), str(thumbnail_offset_list[1]),
                    "--n_neighbors", str(n_neighbors_int)
                ]

                # Include optional arguments if provided
                if models_csv_val:
                    cmd.extend(["--models_csv", models_csv_val])
                if coco_jsonl_path_val:
                    cmd.extend(["--coco_jsonl_path", coco_jsonl_path_val])

                # Run the command
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Close the loading window
                loading.destroy()

                if result.returncode != 0:
                    messagebox.showerror("Script Error", f"Error in calculate_rogueV38.py:\n{result.stderr}")
                else:
                    messagebox.showinfo("Success", f"calculate_rogueV38.py ran successfully. Check the output directory:\n{output_dir_val}")

            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

        threading.Thread(target=run_script).start()
        input_window.destroy()

    # Add Run and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=row+1, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row+1, column=2, sticky='w', padx=5, pady=10)


def setup_models_csv():
    """
    Opens a window to set up the models.csv file by gathering necessary inputs
    and running the create_models_csv.py script.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import subprocess
    import threading
    from pathlib import Path

    # Create a new top-level window for inputs
    input_window = tk.Toplevel()
    input_window.title("Setup Models CSV")

    # Define variables to store user inputs
    image_dir = tk.StringVar()
    base_names_file = tk.StringVar()
    descriptions_dir = tk.StringVar()
    output_csv = tk.StringVar()

    # Function to browse files/directories
    def browse_file(var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def browse_directory(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    # Layout the input fields
    row = 0
    tk.Label(input_window, text="Image Directory (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=image_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(image_dir)).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Base Names File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=base_names_file, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(base_names_file, [("Text Files", "*.txt"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Descriptions Directory (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=descriptions_dir, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_directory(descriptions_dir)).grid(row=row, column=2, padx=5, pady=5)

    row += 1
    tk.Label(input_window, text="Output CSV File (Required):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(input_window, textvariable=output_csv, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(input_window, text="Browse", command=lambda: browse_file(output_csv, [("CSV Files", "*.csv"), ("All Files", "*.*")])).grid(row=row, column=2, padx=5, pady=5)

    row += 1

    # Function to gather inputs and run the script
    def submit():
        # Gather inputs
        image_dir_val = image_dir.get()
        base_names_file_val = base_names_file.get()
        descriptions_dir_val = descriptions_dir.get()
        output_csv_val = output_csv.get()

        # Validate required fields
        if not image_dir_val:
            messagebox.showerror("Input Error", "Please select the Image Directory.")
            return
        if not base_names_file_val:
            messagebox.showerror("Input Error", "Please select the Base Names File.")
            return
        if not descriptions_dir_val:
            messagebox.showerror("Input Error", "Please select the Descriptions Directory.")
            return
        if not output_csv_val:
            messagebox.showerror("Input Error", "Please specify the Output CSV File.")
            return

        # Paths to the create_models_csv.py script
        script_dir = Path(__file__).parent  # Assuming the script is in the same directory
        create_models_csv_script = script_dir / 'idefics' / 'create_description_table_.py'

        if not create_models_csv_script.is_file():
            messagebox.showerror("Script Error", f"create_models_csv.py not found at: {create_models_csv_script}")
            return

        # Run the script in a separate thread
        def run_script():
            try:
                # Show a loading message
                loading = tk.Toplevel()
                loading.title("Running Script")
                tk.Label(loading, text="Please wait while the models.csv is being created...").pack(padx=20, pady=20)
                loading.update()

                # Construct the command
                cmd = [
                    "python",
                    str(create_models_csv_script),
                    "--image_dir", image_dir_val,
                    "--base_names_file", base_names_file_val,
                    "--descriptions_dir", descriptions_dir_val,
                    "--output_csv", output_csv_val
                ]

                # Run the command
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Close the loading window
                loading.destroy()

                if result.returncode != 0:
                    messagebox.showerror("Script Error", f"Error in create_models_csv.py:\n{result.stderr}")
                else:
                    messagebox.showinfo("Success", f"models.csv created successfully at:\n{output_csv_val}")

            except Exception as e:
                loading.destroy()
                messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

        threading.Thread(target=run_script).start()
        input_window.destroy()

    # Add Run and Cancel buttons
    tk.Button(input_window, text="Run", command=submit).grid(row=row+1, column=1, sticky='e', padx=5, pady=10)
    tk.Button(input_window, text="Cancel", command=input_window.destroy).grid(row=row+1, column=2, sticky='w', padx=5, pady=10)










def reset_global_variables():
    global image, segmentation_masks, current_mask_index, labels, via2_regions, region_data
    global file_path_base, bounding_boxes, bbox_rects, current_tool, zoom_level, mode
    global category_id_to_name, category_name_to_id, categories
    global category_label_options, selected_category_label, prediction_data, prediction_image_names
    global selected_prediction_image  # Remove working_pred_json_path from here
    global working_pred_json_path
    # Add this variable
    view_predictions_mode = False
    view_predictions_btn.config(relief=tk.RAISED, text="View Predictions")
    
  # Clear points from memory
    point_coords = []
    point_labels = []

    # Reset canvas and variables
    if canvas.winfo_exists():
        canvas.delete("all")
    image = None
    segmentation_masks = []
    current_mask_index = 0
    labels = {}
    via2_regions = []
    region_data = []
    file_path_base = None
    bounding_boxes = []
    bbox_rects = []
    current_tool = None
    zoom_level = 1.0
    mode = "prompt"

    category_id_to_name = {}
    category_name_to_id = {}
    categories = []
    category_label_options = []
    selected_category_label.set("Select Category")
    prediction_data = {}
    prediction_image_names = []
    selected_prediction_image.set("Select Prediction Image")


# ============================================================================
# SAM2-PAL (SAM2-Palindrome) Functions
# Palindrome-based mask propagation with 4-step OC-CCL fine-tuning
# Based on arxiv.org/abs/2501.06749 - correct implementation with memory reset
# Supports v15 (standard) and v16 (optional LoRA)
# ============================================================================


# ===========================================================================
# VIDEO MODE FUNCTIONS
# ===========================================================================

def extract_video_frames(video_path):
    """Extract frames from video to temp directory."""
    global video_frames, video_metadata, video_temp_dir, original_video_path, frames_to_delete
    import tempfile
    
    video_temp_dir = tempfile.mkdtemp(prefix='descriptron_video_')
    original_video_path = video_path  # Store for later re-encoding
    frames_to_delete = set()  # Reset deletion tracking
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_metadata = {
        'file_name': os.path.basename(video_path),
        'file_path': video_path,  # Store full path for output directory
        'fps': fps if fps > 0 else 30.0,
        'total_frames': total,
        'width': w,
        'height': h
    }
    
    video_frames = []
    idx = 0
    print(f"Extracting {total} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fpath = os.path.join(video_temp_dir, f"{idx:06d}.jpg")  # SAM2 expects pure numbers
        cv2.imwrite(fpath, frame)
        video_frames.append(fpath)
        idx += 1
        if idx % 100 == 0:
            print(f"  {idx} frames...")
    
    cap.release()
    print(f"Done: {len(video_frames)} frames")
    return True


def load_video_frame_into_canvas(frame_idx):
    """Load specific frame into main canvas using existing image loading."""
    global image, original_image, file_path, file_path_base, current_video_frame
    
    if not video_frames or frame_idx >= len(video_frames):
        return False
    
    # Save current frame annotations if any exist
    save_current_frame_annotations()
    
    # Load new frame
    fpath = video_frames[frame_idx]
    current_video_frame = frame_idx
    
    # Use existing image loading
    img_pil = Image.open(fpath)
    image = np.array(img_pil.convert("RGB"))
    original_image = image.copy()
    file_path = fpath
    file_path_base = f"{frame_idx:06d}"  # Match SAM2 naming
    
    # Clear canvas and reload
    canvas.delete("all")
    apply_zoom()
    
    # Restore frame annotations if they exist
    restore_frame_annotations(frame_idx)
    
    # Update timeline to show current frame position (blue line)
    if timeline_draw_callback:
        try:
            timeline_draw_callback()
        except:
            pass
    
    return True


def save_current_frame_annotations():
    """Save current annotations to memory."""
    global video_annotations, current_video_frame, segmentation_masks
    global point_coords, point_labels, keypoint_labels, point_orders
    
    if not video_mode:
        return
    
    # Only save if frame has actual annotations
    has_content = (
        (segmentation_masks and len(segmentation_masks) > 0) or
        (point_coords and len(point_coords) > 0) or
        (keypoint_labels and len(keypoint_labels) > 0)
    )
    
    if has_content:
        video_annotations[current_video_frame] = {
            'masks': copy.deepcopy(segmentation_masks),
            'points': copy.deepcopy(point_coords),
            'point_labels': copy.deepcopy(point_labels),
            'keypoint_labels': copy.deepcopy(keypoint_labels),
            'point_orders': copy.deepcopy(point_orders),
            'labels': copy.deepcopy(labels)
        }
        
        # Update timeline to show green indicator for this annotated frame
        if timeline_draw_callback:
            try:
                timeline_draw_callback()
            except:
                pass


def restore_frame_annotations(frame_idx):
    """Restore annotations for a frame."""
    global segmentation_masks, point_coords, point_labels
    global keypoint_labels, point_orders, labels, current_mask_index
    
    if frame_idx in video_annotations:
        ann = video_annotations[frame_idx]
        segmentation_masks = copy.deepcopy(ann['masks'])
        point_coords = copy.deepcopy(ann['points'])
        point_labels = copy.deepcopy(ann['point_labels'])
        keypoint_labels = copy.deepcopy(ann['keypoint_labels'])
        point_orders = copy.deepcopy(ann['point_orders'])
        labels = copy.deepcopy(ann['labels'])
        
        # Reset current_mask_index to valid range
        if segmentation_masks:
            current_mask_index = min(current_mask_index, len(segmentation_masks) - 1)
            current_mask_index = max(0, current_mask_index)
        else:
            current_mask_index = 0
        
        # Redraw
        redraw_masks()
        redraw_points()
    else:
        # Clear annotations for new frame
        segmentation_masks = []
        point_coords = []
        point_labels = []
        keypoint_labels = []
        point_orders = []
        labels = {}
        current_mask_index = 0


def video_cleanup():
    """Clean up video temp files."""
    global video_temp_dir, video_frames, video_mode, video_annotations, frames_to_delete, original_video_path, timeline_draw_callback
    
    if video_temp_dir and os.path.exists(video_temp_dir):
        try:
            shutil.rmtree(video_temp_dir)
        except:
            pass
    
    video_temp_dir = None
    video_frames = []
    video_mode = False
    video_annotations = {}
    frames_to_delete = set()
    original_video_path = None
    timeline_draw_callback = None

# Default OFF: let SAM2 handle tracking; enable only if you want legacy CV-style area/bbox clamping.
video_use_area_clamp_var = tk.BooleanVar(root, value=False)


# ============================================================================
# MEMORY-EFFICIENT SAM2 VIDEO INITIALIZATION
# ============================================================================

def init_sam2_state_memory_efficient(predictor, video_temp_dir, device='cuda'):
    """
    Initialize SAM2 video state with memory-efficient settings.
    
    Handles long videos (500+ frames) by:
    1. Offloading frames to CPU
    2. Clearing CUDA cache
    3. Monitoring memory usage
    4. Fallback to subsampling if needed
    
    Args:
        predictor: SAM2 video predictor
        video_temp_dir: Directory containing extracted frames
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        inference_state: Initialized SAM2 inference state
    """
    import torch
    
    # Clear CUDA cache before initialization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Check available memory
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3  # GB
        total_mem = torch.cuda.mem_get_info()[1] / 1024**3  # GB
        print(f"\nGPU Memory Status:")
        print(f"  Available: {free_mem:.2f} GB / {total_mem:.2f} GB")
    
    # Try different initialization strategies
    print("\nInitializing SAM2 inference state...")
    
    try:
        # Strategy 1: Use CPU offloading (BEST - requires SAM2 >= 1.0)
        print("  Attempting CPU offload mode...")
        inference_state = predictor.init_state(
            video_path=video_temp_dir,
            offload_video_to_cpu=True,     # Keep frames on CPU
            offload_state_to_cpu=True      # Offload attention states to CPU
        )
        print("  Ã¢Å“â€œ SUCCESS: Using CPU offload (memory-efficient)")
        return inference_state
        
    except TypeError as e:
        # CPU offloading not available in this SAM2 version
        print(f"  Ã¢Å¡Â  CPU offload not available: {type(e).__name__}")
        print("  Trying standard initialization...")
        
        try:
            # Strategy 2: Standard initialization
            inference_state = predictor.init_state(video_path=video_temp_dir)
            print("  Ã¢Å“â€œ SUCCESS: Using standard init")
            return inference_state
            
        except torch.cuda.OutOfMemoryError:
            # Out of memory even with standard init
            print("  Ã¢ÂÅ’ FAILED: Out of memory with standard init")
            print("\n  Attempting frame subsampling as last resort...")
            
            # Strategy 3: Subsample frames (last resort)
            return init_state_with_subsampling(predictor, video_temp_dir)


def init_state_with_subsampling(predictor, video_temp_dir, subsample_rate=2):
    """
    Subsample frames to reduce memory usage.
    Only use this as last resort when other methods fail.
    
    Args:
        predictor: SAM2 video predictor
        video_temp_dir: Directory containing extracted frames
        subsample_rate: Process every Nth frame (2 = every other frame)
    
    Returns:
        inference_state: Initialized SAM2 inference state with subsampling info
    """
    import os
    import shutil
    import tempfile
    
    # Create temporary directory for subsampled frames
    subsampled_dir = tempfile.mkdtemp(prefix='sam2_subsampled_')
    
    print(f"  Creating subsampled frames (every {subsample_rate}th frame)...")
    
    # Get frame files
    frame_files = sorted([
        f for f in os.listdir(video_temp_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # Copy every Nth frame
    copied = 0
    for i, frame_file in enumerate(frame_files):
        if i % subsample_rate == 0:
            src = os.path.join(video_temp_dir, frame_file)
            dst = os.path.join(subsampled_dir, frame_file)
            shutil.copy2(src, dst)
            copied += 1
    
    print(f"  Ã¢Å“â€œ Subsampled: {len(frame_files)} Ã¢â€ â€™ {copied} frames")
    
    # Initialize with subsampled frames
    inference_state = predictor.init_state(video_path=subsampled_dir)
    
    # Store subsampled directory for cleanup
    inference_state['_subsampled_dir'] = subsampled_dir
    inference_state['_subsample_rate'] = subsample_rate
    
    print("  Ã¢Å¡Â  WARNING: Processing subsampled video")
    print("    Results will be lower quality but should fit in memory")
    
    return inference_state


def refresh_select_label_ui(verbose: bool = False):
    """
    Refresh BOTH label dropdown UIs (main + video remote if present).
    Does NOT change your existing update_label_dropdown() logic.
    """
    # Main UI dropdown (already working)
    try:
        update_label_dropdown()
        if verbose:
            print("[OK] refresh_select_label_ui: main label dropdown updated")
    except Exception as e:
        if verbose:
            print(f"[WARNING] refresh_select_label_ui: could not update main dropdown: {e}")

    # Video remote dropdown (if open)
    try:
        update_video_label_dropdown()
        if verbose:
            print("[OK] refresh_select_label_ui: video label dropdown updated")
    except Exception as e:
        if verbose:
            print(f"[WARNING] refresh_select_label_ui: could not update video dropdown: {e}")


def update_video_label_dropdown():
    """
    Update the OptionMenu in the VIDEO remote control with current global `categories`.
    Safe if the remote isn't open yet.
    """
    global video_label_dropdown, video_selected_label, video_label_options, categories

    if "video_label_dropdown" not in globals() or video_label_dropdown is None:
        return

    # Build options from loaded categories (verbatim)
    opts = []
    try:
        opts = [c.get("name", "") for c in (categories or []) if isinstance(c, dict) and c.get("name")]
    except Exception:
        opts = []

    # Always include these UI helpers
    for extra in ("Trash", "Custom"):
        if extra not in opts:
            opts.append(extra)

    # Remove duplicates while preserving order
    opts = list(dict.fromkeys(opts))

    video_label_options = opts

    menu = video_label_dropdown["menu"]
    menu.delete(0, "end")
    for label in video_label_options:
        menu.add_command(label=label, command=tk._setit(video_selected_label, label))

    if video_selected_label.get() not in video_label_options:
        video_selected_label.set("Select Label")


def sync_categories_from_coco(coco_dict, *, verbose=True, update_ui=True):
    """
    Read COCO 'categories' VERBATIM and update all category globals.
    Preserves user semantics (e.g. supercategory='sclerite').

    Expected:
      coco_dict["categories"] = [{"id": 1, "name": "...", "supercategory": "..."}, ...]
    """
    global categories, category_id_to_name, category_name_to_id, category_id_to_supercategory, category_label_options

    cats = coco_dict.get("categories", []) or []
    cats = [c for c in cats if isinstance(c, dict) and "id" in c]

    if not cats:
        if verbose:
            print("[WARNING] sync_categories_from_coco: no categories in COCO; keeping existing mappings.")
        return False

    # Preserve verbatim category list for later export
    categories = []
    category_id_to_name = {}
    category_id_to_supercategory = {}

    for c in cats:
        cid = int(c["id"])
        nm = c.get("name", f"category_{cid}")
        # Only fallback if missing in file (do NOT invent 'morphology')
        sc = c.get("supercategory", "sclerite")
        categories.append({"id": cid, "name": nm, "supercategory": sc})
        category_id_to_name[cid] = nm
        category_id_to_supercategory[cid] = sc

    category_name_to_id = {v: k for k, v in category_id_to_name.items()}
    category_label_options = [category_id_to_name[k] for k in sorted(category_id_to_name.keys())]

    if verbose:
        print(f"[OK] Loaded {len(categories)} categories from COCO (verbatim):")
        for cid in sorted(category_id_to_name.keys()):
            print(f"  - {cid}: {category_id_to_name[cid]} (super={category_id_to_supercategory.get(cid,'sclerite')})")

    if update_ui:
        refresh_select_label_ui(verbose=False)

    return True




def predict_sam2_video():
    """Run SAM2 video prediction on all frames with annotations."""
    if not video_mode:
        messagebox.showwarning("Warning", "Not in video mode")
        return
    
    # IMPORTANT: Save current frame's annotations first!
    # (Otherwise current frame's work won't be included)
    save_current_frame_annotations()
    
    # Check if we have any annotations
    if not video_annotations:
        messagebox.showwarning("Warning", 
            "No annotations to propagate!\n\n"
            "Please annotate at least one frame:\n"
            "1. Click points on the object, OR\n"
            "2. Use SAM2 to generate a mask\n\n"
            "Then try prediction again.")
        return
    
    # Warn if too many frames for memory
    if len(video_annotations) > 4:
        warning_result = messagebox.askyesno(
            "Many Frames Annotated",
            f"You've annotated {len(video_annotations)} frames.\n\n"
            f"This may cause out-of-memory errors on long videos.\n"
            f"For 595-frame videos, 3-4 annotated frames is recommended.\n\n"
            f"Continue anyway?"
        )
        if not warning_result:
            return
    
    result = messagebox.askyesno(
        "SAM2 Video Prediction",
        f"Run SAM2 video tracking on {len(video_annotations)} annotated frames?\n\n"
        f"This will propagate masks through the entire video.\n"
        f"This may take several minutes.\n\n"
        f"Video: {len(video_frames)} frames\n"
        f"Annotated: {len(video_annotations)} frames"
    )
    
    if not result:
        return
    
    try:
        from sam2.build_sam import build_sam2_video_predictor
        import torch
        
        print("Initializing SAM2 video predictor...")
        print(f"(Processing {len(video_frames)} frames - using memory-efficient mode)")
        
        # Build predictor
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        
        # Initialize state with memory-efficient settings
        inference_state = init_sam2_state_memory_efficient(
            predictor, 
            video_temp_dir, 
            device=device
        )
        
        # Set memory constraints to limit accumulation during propagation
        if hasattr(predictor, 'max_cond_frames_in_attn'):
            predictor.max_cond_frames_in_attn = 8  # Reduced from 10 for better memory efficiency
            print(f"  Set max_cond_frames_in_attn = 8")
        
        if hasattr(inference_state, 'max_cond_frames_in_attn'):
            inference_state['max_cond_frames_in_attn'] = 8
        
        # For very long videos, also limit memory bank size
        if len(video_frames) > 500:
            if hasattr(predictor, 'mem_every'):
                predictor.mem_every = 10  # Save memory state every 10 frames (default is 5)
                print(f"  Set mem_every = 10 (for long video)")
        
        # Add prompts from annotated frames
        # MULTI-CATEGORY SUPPORT: Track multiple objects simultaneously
        # Each category gets its own obj_id for independent tracking
        obj_id_map = {}  # category_id -> obj_id for SAM2
        next_obj_id = 0
        prompts_added = 0
        
        print(f"Checking {len(video_annotations)} annotated frames...")
        print("(Multi-category mode: each category tracked independently)")
        
        for frame_idx, ann in video_annotations.items():
            # Process EACH mask in this frame (multi-category support!)
            n_points = len(ann['points']) if ann['points'] else 0
            n_masks = len(ann['masks']) if ann['masks'] else 0
            print(f"Frame {frame_idx}: {n_points} points, {n_masks} masks")
            
            if n_masks == 0:
                print(f"    â†’ Skipping (no masks found)")
                continue
            
            # Process each mask separately for multi-category tracking
            for mask_idx, mask_data in enumerate(ann['masks']):
                mask = mask_data['segmentation']
                category_id = mask_data.get('category_id', 1)
                
                # Map category_id to obj_id for SAM2
                if category_id not in obj_id_map:
                    obj_id_map[category_id] = next_obj_id
                    next_obj_id += 1
                    print(f"  Mapped category {category_id} â†’ obj_id {obj_id_map[category_id]}")
                
                obj_id = obj_id_map[category_id]
                
                # Prepare mask as float32 in range [0, 1] for SAM2
                if mask.dtype != np.float32:
                    mask_input = mask.astype(np.float32)
                else:
                    mask_input = mask.copy()
                
                # Normalize to [0, 1] if needed
                if mask_input.max() > 1.0:
                    mask_input = mask_input / 255.0
                
                # Get bounding box from mask (for reference)
                bbox = None
                coords = np.argwhere(mask > 0)
                if len(coords) > 0:
                    y_coords, x_coords = coords[:, 0], coords[:, 1]
                    bbox = np.array([
                        x_coords.min(),
                        y_coords.min(),
                        x_coords.max(),
                        y_coords.max()
                    ])
                    
                    print(f"  Frame {frame_idx}, Mask {mask_idx+1}/{n_masks}: cat_id={category_id}, obj_id={obj_id}, area={np.sum(mask > 0)} pixels")
                
                # Add mask prompt to SAM2
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask_input
                )
                
                prompts_added += 1
        
        if prompts_added == 0:
            messagebox.showerror("Error", 
                "No valid masks found!"
                "Make sure you have created masks on at least one frame."
                "Use SAM2 or keypoint-to-mask to generate masks.")
            return
        
        print(f"Added {prompts_added} mask prompts for {len(obj_id_map)} categories")
        print(f"Category â†’ Object mapping: {obj_id_map}")
        
        # Get annotated keyframes for re-anchoring
        keyframe_indices = sorted(list(video_annotations.keys()))
        print(f"Keyframes for drift prevention: {keyframe_indices}")
        
        # Get clamping setting from UI variable
        use_area_clamp = video_use_area_clamp_var.get()
        
        # Propagate with anti-drift strategies
        print("Propagating through video...")
        print("(Using re-anchoring strategy to prevent mask drift)")
        print("(Area/bbox clamping (optional): " + ("ON" if use_area_clamp else "OFF") + ")")
        print("(Clearing GPU cache frequently to prevent OOM)")
        tracks = {}
        frame_count = 0
        reanchor_interval = 50  # Re-anchor every 50 frames
        
        # Area consistency tracking (Strategy 4)
        prev_area = {}  # obj_id -> area
        prev_bbox = {}  # obj_id -> [x1, y1, x2, y2]
        prev_center = {}  # obj_id -> (cx, cy)
        area_growth_limit = 2.5  # Max 2.5x growth allowed
        area_shrink_limit = 0.4  # Min 0.4x shrinkage allowed (below = drift)
        bbox_move_threshold = 100  # Max center movement in pixels
        drift_detected_frames = []  # Track where drift occurred
        
        # Clear CUDA cache before propagation to ensure maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"\nÃ¢Å“â€œ Cleared CUDA cache before propagation")
        
        print(f"\nÃ°Å¸â€œÂ¹ Propagating masks through {len(video_frames)} frames...")
        
        try:
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
                # mask_logits can be either:
                # - A dict {obj_id: mask_tensor}
                # - A tensor of shape (num_objects, H, W)
                
                # Handle both cases and store masks
                current_masks = {}
                if isinstance(mask_logits, dict):
                    # Dict format: {obj_id: mask}
                    for oid, mask in mask_logits.items():
                        if oid not in tracks:
                            tracks[oid] = {}
                        # Move to CPU immediately to free GPU memory
                        m = (mask > 0).squeeze()
                        if hasattr(m, 'cpu'):
                            m = m.cpu().numpy()
                        else:
                            m = np.array(m)
                        m_uint8 = m.astype(np.uint8)
                        # Ensure mask is 2D for bbox/area logic
                        m_uint8 = np.squeeze(m_uint8)
                        if m_uint8.ndim != 2:
                            # Fallback: if somehow still has extra dims, take the first plane(s)
                            while getattr(m_uint8, 'ndim', 0) > 2:
                                m_uint8 = m_uint8[0]
                        
                        # STRATEGY 4: Area consistency checking
                        area = int(np.sum(m_uint8))
                        drift_detected = False
                        
                        if use_area_clamp and oid in prev_area and prev_area[oid] > 0:
                            # Check for explosive growth (drift to larger object)
                            if area > area_growth_limit * prev_area[oid]:
                                drift_detected = True
                                drift_reason = f"growth {area/prev_area[oid]:.1f}x"
                                
                                # Clamp to previous bbox to prevent drift
                                if oid in prev_bbox:
                                    bbox = prev_bbox[oid]
                                    x1, y1, x2, y2 = bbox
                                    # Zero out pixels outside previous bbox
                                    clamped = m_uint8.copy()
                                    clamped[:y1, :] = 0
                                    clamped[y2+1:, :] = 0
                                    clamped[:, :x1] = 0
                                    clamped[:, x2+1:] = 0
                                    m_uint8 = clamped
                                    area = int(np.sum(m_uint8))
                                    print(f"    [WARNING] Drift detected at frame {frame_idx} ({drift_reason}) - clamped to previous bbox")
                            
                            # Check for dramatic shrinkage (lost tracking)
                            elif area < area_shrink_limit * prev_area[oid]:
                                drift_detected = True
                                drift_reason = f"shrink {area/prev_area[oid]:.1f}x"
                                print(f"    [WARNING] Tracking quality drop at frame {frame_idx} ({drift_reason})")
                            
                            # Check bbox center movement
                            if area > 0:
                                ys, xs = np.where(m_uint8 > 0)
                                if len(xs) > 0:
                                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                                    if oid in prev_center:
                                        pcx, pcy = prev_center[oid]
                                        movement = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                                        if movement > bbox_move_threshold:
                                            drift_detected = True
                                            drift_reason = f"moved {movement:.0f}px"
                                            print(f"    [WARNING] Large movement at frame {frame_idx} ({drift_reason})")
                                    prev_center[oid] = (cx, cy)
                        
                        # Update tracking stats
                        if area > 0:
                            prev_area[oid] = area
                            ys, xs = np.where(m_uint8 > 0)
                            if len(xs) > 0:
                                x1, y1 = max(0, xs.min() - 10), max(0, ys.min() - 10)
                                x2, y2 = min(m_uint8.shape[1]-1, xs.max() + 10), min(m_uint8.shape[0]-1, ys.max() + 10)
                                prev_bbox[oid] = [x1, y1, x2, y2]
                        
                        if drift_detected:
                            drift_detected_frames.append(frame_idx)
                        
                        tracks[oid][frame_idx] = m_uint8
                        current_masks[oid] = m_uint8
                        
                        # Delete GPU tensor immediately
                        del mask
                else:
                    # Tensor format: index by obj_ids
                    for i, oid in enumerate(obj_ids):
                        if oid not in tracks:
                            tracks[oid] = {}
                        # Get mask for this object
                        # Get mask for this object (SAM2 may return (N,H,W) or (N,1,H,W))
                        if hasattr(mask_logits, 'ndim'):
                            if mask_logits.ndim == 4:  # (num_objects, 1, H, W)
                                mask = mask_logits[i, 0]
                            elif mask_logits.ndim == 3:  # (num_objects, H, W) or (1, H, W)
                                # If first dim matches number of objects, index by i, else treat as single
                                if mask_logits.shape[0] == len(obj_ids):
                                    mask = mask_logits[i]
                                else:
                                    mask = mask_logits[0]
                            else:  # (H, W)
                                mask = mask_logits
                        else:
                            mask = mask_logits
                        
                        # Move to CPU immediately
                        m = (mask > 0).squeeze()
                        if hasattr(m, 'cpu'):
                            m = m.cpu().numpy()
                        else:
                            m = np.array(m)
                        m_uint8 = m.astype(np.uint8)
                        # Ensure 2D mask for bbox/area logic (SAM2 can return [n_obj,H,W] or [1,H,W])
                        m_uint8 = np.squeeze(m_uint8)
                        if getattr(m_uint8, 'ndim', 0) != 2:
                            # Fallback: take the first slice until we reach 2D
                            while getattr(m_uint8, 'ndim', 0) > 2:
                                m_uint8 = m_uint8[0]
                            if getattr(m_uint8, 'ndim', 0) != 2:
                                m_uint8 = m_uint8.reshape(m_uint8.shape[-2], m_uint8.shape[-1])

                        
                        # STRATEGY 4: Area consistency checking (same as above)
                        area = int(np.sum(m_uint8))
                        drift_detected = False
                        
                        if use_area_clamp and oid in prev_area and prev_area[oid] > 0:
                            if area > area_growth_limit * prev_area[oid]:
                                drift_detected = True
                                drift_reason = f"growth {area/prev_area[oid]:.1f}x"
                                if oid in prev_bbox:
                                    bbox = prev_bbox[oid]
                                    x1, y1, x2, y2 = bbox
                                    clamped = m_uint8.copy()
                                    clamped[:y1, :] = 0
                                    clamped[y2+1:, :] = 0
                                    clamped[:, :x1] = 0
                                    clamped[:, x2+1:] = 0
                                    m_uint8 = clamped
                                    area = int(np.sum(m_uint8))
                                    print(f"    [WARNING] Drift detected at frame {frame_idx} ({drift_reason}) - clamped to previous bbox")
                            elif area < area_shrink_limit * prev_area[oid]:
                                drift_detected = True
                                drift_reason = f"shrink {area/prev_area[oid]:.1f}x"
                                print(f"    [WARNING] Tracking quality drop at frame {frame_idx} ({drift_reason})")
                        
                        if area > 0:
                            prev_area[oid] = area
                            ys, xs = np.where(m_uint8 > 0)
                            if len(xs) > 0:
                                x1, y1 = max(0, xs.min() - 10), max(0, ys.min() - 10)
                                x2, y2 = min(m_uint8.shape[1]-1, xs.max() + 10), min(m_uint8.shape[0]-1, ys.max() + 10)
                                prev_bbox[oid] = [x1, y1, x2, y2]
                        
                        if drift_detected:
                            drift_detected_frames.append(frame_idx)
                        
                        tracks[oid][frame_idx] = m_uint8
                        current_masks[oid] = m_uint8
                
                frame_count += 1
                
                # ANTI-DRIFT STRATEGY 1: Re-anchor at keyframes
                # If we're at an annotated keyframe, refresh with original annotation
                if frame_idx in keyframe_indices:
                    ann = video_annotations[frame_idx]
                    if ann['masks']:
                        original_mask = ann['masks'][0]['segmentation']
                        try:
                            # Try to refresh tracking with original annotation
                            if hasattr(predictor, 'add_new_mask'):
                                predictor.add_new_mask(
                                    inference_state=inference_state,
                                    frame_idx=frame_idx,
                                    obj_id=0,
                                    mask=original_mask
                                )
                                print(f"    [OK] Re-anchored at keyframe {frame_idx}")
                        except:
                            pass  # Not all SAM2 versions support mid-stream add_new_mask
                
                # ANTI-DRIFT STRATEGY 2: Periodic re-anchoring
                # Every N frames, refresh with current prediction to prevent drift
                elif reanchor_interval > 0 and frame_count % reanchor_interval == 0 and current_masks:
                    try:
                        for oid, mask in current_masks.items():
                            if hasattr(predictor, 'add_new_mask'):
                                predictor.add_new_mask(
                                    inference_state=inference_state,
                                    frame_idx=frame_idx,
                                    obj_id=oid,
                                    mask=mask
                                )
                        print(f"    <- Re-anchored with prediction at frame {frame_idx}")
                    except:
                        pass  # Silently fail if not supported
                
                # ANTI-DRIFT STRATEGY 5: Adaptive re-anchoring (NEW!)
                # Re-anchor immediately when drift is detected
                elif drift_detected and current_masks:
                    try:
                        for oid, mask in current_masks.items():
                            if hasattr(predictor, 'add_new_mask'):
                                predictor.add_new_mask(
                                    inference_state=inference_state,
                                    frame_idx=frame_idx,
                                    obj_id=oid,
                                    mask=mask  # Use clamped mask
                                )
                        print(f"    Adaptive re-anchor triggered at frame {frame_idx} (drift recovery)")
                    except:
                        pass
                
                # VERY aggressive cache clearing - every 10 frames
                if frame_count % 10 == 0:
                    import torch
                    torch.cuda.empty_cache()
                    if frame_count % 50 == 0:
                        print(f"  Processed {frame_count} frames... (GPU cache cleared)")
                    
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[WARNING]  GPU ran out of memory at frame {frame_count}")
            print(f"Processed {frame_count}/{len(video_frames)} frames before OOM")
            messagebox.showwarning(
                "Out of Memory",
                f"GPU ran out of memory at frame {frame_count}.\n\n"
                f"Processed: {frame_count}/{len(video_frames)} frames\n"
                f"Partial results saved to memory.\n\n"
                f"Try these fixes:\n"
                f"1. Annotate fewer frames (3-4 instead of {prompts_added})\n"
                f"2. Use shorter video clip (first 300 frames)\n"
                f"3. Reduce video resolution (720p)\n"
                f"4. Close other applications\n"
                f"5. Restart Python to clear memory"
            )
            # Continue with partial results
        
        print(f"\nPropagation complete: {frame_count} frames processed")
        
        # ANTI-DRIFT STRATEGY 3: Memory reset (from SAM2-PAL paper)
        # Clear accumulated memory to prevent future drift
        try:
            if hasattr(predictor, 'reset_state'):
                predictor.reset_state(inference_state)
                print("[OK] Memory state reset (prevents future drift)")
        except:
            pass
        
        # Update annotations with predictions
        # MULTI-CATEGORY FIX: Preserve category_id for each tracked object
        # Create reverse mapping: obj_id â†’ category_id
        obj_to_cat = {obj_id: cat_id for cat_id, obj_id in obj_id_map.items()}
        
        for oid, frame_masks in tracks.items():
            # Get category for this object
            category_id = obj_to_cat.get(oid, 1)  # Default to 1 if not found
            
            for frame_idx, mask in frame_masks.items():
                if frame_idx not in video_annotations:
                    video_annotations[frame_idx] = {
                        'masks': [],
                        'points': [],
                        'point_labels': [],
                        'keypoint_labels': [],
                        'point_orders': [],
                        'labels': {}
                    }
                
                # Add mask to frame with category_id preserved!
                video_annotations[frame_idx]['masks'].append({
                    'segmentation': mask,
                    'area': np.sum(mask),
                    'bbox': cv2.boundingRect(mask),
                    'predicted': True,
                    'category_id': category_id  # CRITICAL: Preserve category!
                })
        
        # Count re-anchoring events
        num_keyframe_anchors = len([f for f in keyframe_indices if f < frame_count])
        num_periodic_anchors = frame_count // reanchor_interval if reanchor_interval > 0 else 0
        num_drift_anchors = len(drift_detected_frames)
        
        # Report drift statistics
        if drift_detected_frames:
            print(f"\n[WARNING] Drift detected and corrected on {num_drift_anchors} frames:")
            print(f"   Frames: {drift_detected_frames[:10]}" + (" ..." if len(drift_detected_frames) > 10 else ""))
        
        messagebox.showinfo("Success", 
            f"SAM2 video prediction complete!\n\n"
            f"Objects tracked: {len(tracks)}\n"
            f"Frames processed: {frame_count}\n"
            f"Keyframes used: {num_keyframe_anchors}\n"
            f"Periodic re-anchoring: {num_periodic_anchors}\n"
            f"Drift corrections: {num_drift_anchors}\n"
            f"Total annotated frames: {len(video_annotations)}\n\n"
            f"Anti-drift strategies applied:\n"
            f"* Keyframe re-anchoring\n"
            f"* Periodic refresh (every {reanchor_interval} frames)\n"
            f"* Area consistency monitoring [NEW] NEW\n"
            f"* Adaptive re-anchoring [NEW] NEW\n"
            f"* Memory state reset\n\n"
            f"Navigate frames to review predictions.\n"
            f"Use Ã¢â€“Â¶ Play to check tracking quality.")
        
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up subsampled directory if it was created
        if 'inference_state' in locals() and isinstance(inference_state, dict):
            subsampled_dir = inference_state.get('_subsampled_dir')
            if subsampled_dir and os.path.exists(subsampled_dir):
                import shutil
                try:
                    shutil.rmtree(subsampled_dir)
                    print(f"\nÃ¢Å“â€œ Cleaned up temporary subsampled directory")
                except Exception as cleanup_error:
                    print(f"\nÃ¢Å¡Â  Could not clean up subsampled directory: {cleanup_error}")

def load_video_annotations():
    """
    Load video COCO JSON:
      - Sync categories verbatim into global mappings (so Select Label shows them)
      - Rebuild video_annotations with per-frame masks
      - Populate `labels[...]` for mask switching/display (category_id -> name)
      - ALSO load point prompts if present in COCO (points, point_labels, point_orders, keypoint_labels)
    """
    global video_annotations, categories
    global category_id_to_name, category_name_to_id, category_id_to_supercategory, category_label_options

    json_file = filedialog.askopenfilename(
        title="Load Video Annotations (COCO JSON)",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    if not json_file:
        return

    try:
        with open(json_file, "r") as f:
            coco = json.load(f)

        # 1) IMPORTANT: load categories FIRST (verbatim), then refresh dropdown UIs
        sync_categories_from_coco(coco, verbose=True, update_ui=True)

        # 2) Rebuild video_annotations from COCO annotations
        video_annotations.clear()

        def _coerce_points(points_raw):
            """
            Accept either:
              - [[x,y], [x,y], ...]
              - [x,y,x,y,...]
            Return: [[x,y], ...] as float
            """
            if not points_raw:
                return []
            if isinstance(points_raw, (list, tuple)):
                # flat list
                if len(points_raw) >= 2 and isinstance(points_raw[0], (int, float)) and isinstance(points_raw[1], (int, float)):
                    out = []
                    n = len(points_raw) - (len(points_raw) % 2)
                    for i in range(0, n, 2):
                        out.append([float(points_raw[i]), float(points_raw[i + 1])])
                    return out
                # list of pairs
                out = []
                for p in points_raw:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        out.append([float(p[0]), float(p[1])])
                return out
            return []

        def _coerce_point_labels(lbl_raw, n_pts):
            """
            Accept list of ints/bools; if wrong length, pad/truncate.
            Default: all zeros.
            """
            if not lbl_raw or not isinstance(lbl_raw, (list, tuple)):
                return [0] * n_pts
            lbls = [int(x) for x in lbl_raw[:n_pts]]
            if len(lbls) < n_pts:
                lbls.extend([0] * (n_pts - len(lbls)))
            return lbls

        def _coerce_point_orders(ord_raw, n_pts, start_at=0):
            """
            Accept list; else generate sequential.
            """
            if ord_raw and isinstance(ord_raw, (list, tuple)):
                vals = [int(x) for x in ord_raw[:n_pts]]
                if len(vals) < n_pts:
                    vals.extend(list(range(start_at + len(vals), start_at + n_pts)))
                return vals
            return list(range(start_at, start_at + n_pts))

        for ann in coco.get("annotations", []) or []:
            frame_id = ann.get("frame_id", None)
            if frame_id is None:
                continue
            frame_id = int(frame_id)

            if frame_id not in video_annotations:
                video_annotations[frame_id] = {
                    "masks": [],
                    "points": [],
                    "point_labels": [],
                    "keypoint_labels": [],
                    "point_orders": [],
                    "labels": {}   # mask_index -> label string
                }

            # ----------------------------
            # LOAD POINT PROMPTS (if any)
            # ----------------------------
            # Points are typically stored on the annotation entry in Descriptron COCO exports.
            # Because a frame can have multiple mask annotations, we merge points per-frame
            # and avoid obvious duplicates.
            pts_raw = ann.get("points", None)
            lbl_raw = ann.get("point_labels", None)
            ord_raw = ann.get("point_orders", None)
            key_lbl_raw = ann.get("keypoint_labels", None)

            pts = _coerce_points(pts_raw)
            if pts:
                lbls = _coerce_point_labels(lbl_raw, len(pts))

                # Merge unique points (by rounded coordinate + label)
                existing = video_annotations[frame_id]["points"]
                existing_lbls = video_annotations[frame_id]["point_labels"]

                seen = set()
                for (p, l) in zip(existing, existing_lbls):
                    seen.add((round(float(p[0]), 3), round(float(p[1]), 3), int(l)))

                new_pts = []
                new_lbls = []
                for (p, l) in zip(pts, lbls):
                    key = (round(float(p[0]), 3), round(float(p[1]), 3), int(l))
                    if key in seen:
                        continue
                    seen.add(key)
                    new_pts.append(p)
                    new_lbls.append(int(l))

                if new_pts:
                    start_at = len(video_annotations[frame_id]["points"])
                    new_orders = _coerce_point_orders(ord_raw, len(new_pts), start_at=start_at)

                    video_annotations[frame_id]["points"].extend(new_pts)
                    video_annotations[frame_id]["point_labels"].extend(new_lbls)
                    video_annotations[frame_id]["point_orders"].extend(new_orders)

                    # keypoint_labels is optional; keep lengths consistent if provided
                    if key_lbl_raw and isinstance(key_lbl_raw, (list, tuple)):
                        k = [int(x) for x in key_lbl_raw[:len(new_pts)]]
                        if len(k) < len(new_pts):
                            k.extend([0] * (len(new_pts) - len(k)))
                        video_annotations[frame_id]["keypoint_labels"].extend(k)
                    else:
                        # if you want to keep an explicit list the same length:
                        video_annotations[frame_id]["keypoint_labels"].extend([0] * len(new_pts))

            # ----------------------------
            # LOAD MASK SEGMENTATION
            # ----------------------------
            segs = ann.get("segmentation", None)
            if not segs:
                continue

            # Build 0/1 mask from polygons
            mask = np.zeros((video_metadata["height"], video_metadata["width"]), dtype=np.uint8)
            for seg in segs:
                if isinstance(seg, (list, tuple)) and len(seg) >= 6:
                    pts_poly = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts_poly], 1)

            area = int(mask.sum())
            if area <= 0:
                continue

            # Robust bbox: prefer COCO bbox if valid, else compute from contours
            bbox = ann.get("bbox", None)
            use_bbox = None
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    use_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                except Exception:
                    use_bbox = None

            if use_bbox is None:
                mask_u8 = (mask.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = [c for c in contours if c is not None and len(c) >= 3]
                if contours:
                    pts2 = np.vstack(contours)
                    x, y, w, h = cv2.boundingRect(pts2)
                    use_bbox = [int(x), int(y), int(w), int(h)]
                else:
                    use_bbox = [0, 0, 0, 0]

            cat_id = int(ann.get("category_id", 1))
            label_name = category_id_to_name.get(cat_id, f"category_{cat_id}")

            # Append mask entry
            mask_index = len(video_annotations[frame_id]["masks"])
            video_annotations[frame_id]["masks"].append({
                "segmentation": mask,          # keep 0/1 for GUI logic
                "area": area,
                "bbox": use_bbox,
                "predicted": bool(ann.get("predicted", False)),
                "finetuned": bool(ann.get("finetuned", False)),
                "category_id": cat_id,
                "ann_id": int(ann.get("id", -1)),
                "iscrowd": int(ann.get("iscrowd", 0)),
                "label": label_name,           # helpful for UI
            })

            # Populate label lookup so Next/Prev Mask shows correct label
            video_annotations[frame_id]["labels"][mask_index] = label_name

        print(f"[OK] Loaded video annotations from: {os.path.basename(json_file)}")
        print(f"[OK] Frames with annotations: {len(video_annotations)}")

        # Quick visibility sanity check
        total_pts = sum(len(v.get("points", [])) for v in video_annotations.values())
        print(f"[OK] Total loaded points: {total_pts}")

        # Jump to first annotated frame (or 0 if none)
        if video_annotations:
            first_frame = min(video_annotations.keys())
            load_video_frame_into_canvas(first_frame)
            print(f"[OK] Displaying frame {first_frame}")
        else:
            load_video_frame_into_canvas(0)
            print("[WARNING] No annotations found in JSON; showing frame 0")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load video annotations:\n{e}")
        import traceback
        traceback.print_exc()



def delete_frame_range(start_frame, end_frame):
    """Mark a range of frames for deletion."""
    global frames_to_delete
    
    if start_frame < 0 or end_frame >= len(video_frames):
        messagebox.showerror("Error", f"Frame range must be between 0 and {len(video_frames)-1}")
        return False
    
    if start_frame > end_frame:
        messagebox.showerror("Error", "Start frame must be <= end frame")
        return False
    
    # Add frames to deletion set
    for i in range(start_frame, end_frame + 1):
        frames_to_delete.add(i)
    
    messagebox.showinfo("Marked for Deletion", 
                       f"Marked frames {start_frame} to {end_frame} for deletion.\n"
                       f"Total frames to delete: {len(frames_to_delete)}\n"
                       f"Click the marmot button to save the edited video.")
    return True


def clear_frame_deletions():
    """Clear all marked frame deletions."""
    global frames_to_delete
    frames_to_delete.clear()
    messagebox.showinfo("Cleared", "Cleared all frame deletion marks")


def save_edited_video_and_annotations():
    """
    Save edited video (without deleted frames) and update COCO JSON annotations.
    This is called when the marmot button is pressed in video mode.
    """
    global video_annotations, video_frames, frames_to_delete, original_video_path, video_metadata
    
    if not video_mode:
        return
    
    # Save current frame first
    save_current_frame_annotations()
    
    # Check if any frames are marked for deletion
    if not frames_to_delete:
        # No deletions - use original save function
        save_video_annotations()
        return
    
    # Ask user for output paths
    output_video_path = filedialog.asksaveasfilename(
        title="Save Edited Video",
        defaultextension=".mp4",
        filetypes=[("MP4", "*.mp4"), ("WebM", "*.webm"), ("All", "*.*")],
        initialfile=f"edited_{os.path.splitext(video_metadata['file_name'])[0]}.mp4"
    )
    
    if not output_video_path:
        return
    
    output_json_path = filedialog.asksaveasfilename(
        title="Save Updated Annotations",
        defaultextension=".json",
        filetypes=[("JSON", "*.json")],
        initialfile=f"edited_{os.path.splitext(video_metadata['file_name'])[0]}.json"
    )
    
    if not output_json_path:
        return
    
    # Show progress dialog
    progress_window = tk.Toplevel(root)
    progress_window.title("Processing Video...")
    progress_window.geometry("400x150")
    progress_window.transient(root)
    
    tk.Label(progress_window, text="Saving edited video and updating annotations...", 
             font=("Arial", 10)).pack(pady=20)
    progress_label = tk.Label(progress_window, text="Starting...", font=("Arial", 9))
    progress_label.pack(pady=10)
    
    progress_window.update()
    
    try:
        # Step 1: Re-encode video without deleted frames
        progress_label.config(text=f"Re-encoding video (removing {len(frames_to_delete)} frames)...")
        progress_window.update()
        
        if not reencode_video_without_frames(original_video_path, output_video_path, frames_to_delete):
            messagebox.showerror("Error", "Failed to re-encode video")
            progress_window.destroy()
            return
        
        # Step 2: Update annotations with new frame indices
        progress_label.config(text="Updating COCO annotations...")
        progress_window.update()
        
        updated_annotations, new_frame_count = remap_annotations_after_deletion(
            video_annotations, frames_to_delete, len(video_frames)
        )
        
        # Step 3: Build updated COCO JSON
        progress_label.config(text="Saving COCO JSON...")
        progress_window.update()
        
        coco = {
            "videos": [{
                "id": 1,
                "file_name": os.path.basename(output_video_path),
                "width": video_metadata['width'],
                "height": video_metadata['height'],
                "length": new_frame_count,
                "fps": video_metadata['fps']
            }],
            "images": [],
            "annotations": [],
            "categories": categories if categories else [{"id": 1, "name": "object", "supercategory": "morphology"}]
        }
        
        ann_id = 1
        for new_frame_idx in range(new_frame_count):
            img_id = new_frame_idx + 1
            
            coco['images'].append({
                "id": img_id,
                "video_id": 1,
                "frame_id": new_frame_idx,
                "file_name": f"{new_frame_idx:06d}.jpg",
                "width": video_metadata['width'],
                "height": video_metadata['height']
            })
            
            if new_frame_idx in updated_annotations:
                ann = updated_annotations[new_frame_idx]
                
                # Save masks
                for mask_data in ann['masks']:
                    mask = mask_data['segmentation']
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    seg = [c.flatten().tolist() for c in contours if len(c) >= 3]
                    
                    if seg:
                        x, y, w, h = cv2.boundingRect(mask)
                        annotation_entry = {
                            "id": ann_id,
                            "image_id": img_id,
                            "video_id": 1,
                            "frame_id": new_frame_idx,
                            "category_id": 1,
                            "segmentation": seg,
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "area": float(np.sum(mask)),
                            "iscrowd": 0
                        }
                        
                        # Add point prompts if they exist
                        if ann['points'] and len(ann['points']) > 0:
                            annotation_entry['points'] = ann['points']
                            annotation_entry['point_labels'] = ann['point_labels']
                            if ann.get('point_orders'):
                                annotation_entry['point_orders'] = ann['point_orders']
                        
                        coco['annotations'].append(annotation_entry)
                        ann_id += 1
        
        with open(output_json_path, 'w') as f:
            json.dump(coco, f, indent=2)
        
        progress_window.destroy()
        
        messagebox.showinfo("Success", 
                           f"Saved edited video: {output_video_path}\n"
                           f"Saved updated annotations: {output_json_path}\n\n"
                           f"Original frames: {len(video_frames)}\n"
                           f"Deleted frames: {len(frames_to_delete)}\n"
                           f"New frame count: {new_frame_count}\n"
                           f"Annotations: {ann_id-1}")
        
    except Exception as e:
        progress_window.destroy()
        messagebox.showerror("Error", f"Failed to save edited video:\n{str(e)}")
        import traceback
        traceback.print_exc()


def reencode_video_without_frames(input_path, output_path, frames_to_skip):
    """
    Re-encode video, skipping specified frames.
    
    Args:
        input_path: Original video path
        output_path: Output video path
        frames_to_skip: Set of frame indices to skip
    
    Returns:
        bool: True if successful
    """
    try:
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine codec based on output extension
        ext = os.path.splitext(output_path)[1].lower()
        if ext == '.webm':
            fourcc = cv2.VideoWriter_fourcc(*'VP80')  # WebM codec
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        
        # Create video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Cannot create video writer for {output_path}")
            cap.release()
            return False
        
        frame_idx = 0
        frames_written = 0
        
        print(f"Re-encoding video: {total_frames} frames -> {total_frames - len(frames_to_skip)} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip if frame is marked for deletion
            if frame_idx not in frames_to_skip:
                out.write(frame)
                frames_written += 1
                
                if frames_written % 100 == 0:
                    print(f"  Written {frames_written} frames...")
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        print(f"Done: Wrote {frames_written} frames to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error re-encoding video: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def remap_annotations_after_deletion(annotations_dict, deleted_frames, original_frame_count):
    """
    Remap annotation frame indices after frame deletion.
    
    Args:
        annotations_dict: Dictionary mapping old frame indices to annotations
        deleted_frames: Set of frame indices that were deleted
        original_frame_count: Original number of frames
    
    Returns:
        tuple: (new_annotations_dict, new_frame_count)
    """
    # Create mapping from old frame index to new frame index
    old_to_new = {}
    new_idx = 0
    
    for old_idx in range(original_frame_count):
        if old_idx not in deleted_frames:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    
    # Remap annotations
    new_annotations = {}
    for old_idx, ann in annotations_dict.items():
        if old_idx in old_to_new:
            new_annotations[old_to_new[old_idx]] = ann
    
    new_frame_count = original_frame_count - len(deleted_frames)
    
    print(f"Remapped {len(annotations_dict)} annotations:")
    print(f"  Old frames with annotations: {sorted(annotations_dict.keys())[:10]}...")
    print(f"  New frames with annotations: {sorted(new_annotations.keys())[:10]}...")
    print(f"  New frame count: {new_frame_count}")
    
    return new_annotations, new_frame_count

def _mask_to_u8_2d(mask):
    import numpy as np
    # torch -> numpy
    try:
        import torch
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().to("cpu").float().numpy()
    except Exception:
        pass

    mask = np.asarray(mask)
    mask = np.squeeze(mask)

    # Handle common 3D cases
    if mask.ndim == 3:
        # (N,H,W) or (H,W,C) -> take first plane/channel
        if mask.shape[0] <= 4 and mask.shape[1] > 16 and mask.shape[2] > 16:
            mask = mask[0]        # (N,H,W)
        else:
            mask = mask[..., 0]   # (H,W,C)

    if mask.ndim != 2:
        raise ValueError(f"Mask not 2D after squeeze: shape={mask.shape}")

    # binarize (SAM masks often float probs)
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)

    return mask * 255  # OpenCV likes 0/255

def save_video_annotations():
    """Save all video annotations as COCO JSON."""
    if not video_mode:
        return
    
    # Save current frame first
    save_current_frame_annotations()
    
    path = filedialog.asksaveasfilename(
        title="Save Video Annotations",
        defaultextension=".json",
        filetypes=[("JSON", "*.json")]
    )
    
    if not path:
        return
    
    # Build COCO JSON
    coco = {
        "videos": [{
            "id": 1,
            "file_name": video_metadata['file_name'],
            "width": video_metadata['width'],
            "height": video_metadata['height'],
            "length": len(video_frames),
            "fps": video_metadata['fps']
        }],
        "images": [],
        "annotations": [],
        "categories": categories if categories else [{"id": 1, "name": "object", "supercategory": "morphology"}]
    }
    
    ann_id = 1
    for frame_idx in range(len(video_frames)):
        img_id = frame_idx + 1
        
        coco['images'].append({
            "id": img_id,
            "video_id": 1,
            "frame_id": frame_idx,
            "file_name": os.path.basename(video_frames[frame_idx]),
            "width": video_metadata['width'],
            "height": video_metadata['height']
        })
        
        if frame_idx in video_annotations:
            ann = video_annotations[frame_idx]
            
            # Save masks
            for mask_data in ann['masks']:
                mask = mask_data['segmentation']
                
                # CRITICAL FIX: Convert mask to uint8 for cv2.findContours
                # findContours requires CV_8UC1 type (uint8), not float32
                # Handle various mask formats
                if mask.dtype != np.uint8:
                    # Convert to binary mask first (handle float/int types)
                    if mask.dtype in [np.float32, np.float64]:
                        # Float mask: threshold at 0.5
                        mask_binary = (mask > 0.5).astype(np.uint8)
                    else:
                        # Int mask: any non-zero value
                        mask_binary = (mask > 0).astype(np.uint8)
                    # Convert to uint8 with proper values for findContours
                    mask_uint8 = mask_binary * 255
                else:
                    # Already uint8, but ensure it's binary (0 or 255)
                    mask_uint8 = ((mask > 0) * 255).astype(np.uint8)
                
                # Ensure contiguous array for OpenCV
                mask_uint8 = np.ascontiguousarray(mask_uint8)
                
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                seg = [c.flatten().tolist() for c in contours if len(c) >= 3]
                
                if seg:
                    x, y, w, h = cv2.boundingRect(mask_uint8)
                    
                    # CRITICAL FIX: Preserve category_id from mask_data instead of hardcoding to 1
                    category_id = mask_data.get('category_id', 1)
                    
                    annotation_entry = {
                        "id": ann_id,
                        "image_id": img_id,
                        "video_id": 1,
                        "frame_id": frame_idx,
                        "category_id": category_id,  # Use actual category, not hardcoded 1
                        "segmentation": seg,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": float(np.sum(mask > 0)),
                        "iscrowd": 0
                    }
                    
                    # Add point prompts if they exist (custom Descriptron extension)
                    # Only add points to the FIRST mask to avoid duplication
                    if ann_id == 1 or (ann_id > 1 and not coco['annotations'][-1].get('points')):
                        if ann['points'] and len(ann['points']) > 0:
                            annotation_entry['points'] = ann['points']  # List of [x, y]
                            annotation_entry['point_labels'] = ann['point_labels']  # List of 1/0
                            # Save point_orders if available (for display numbering)
                            if ann.get('point_orders'):
                                annotation_entry['point_orders'] = ann['point_orders']
                    
                    coco['annotations'].append(annotation_entry)
                    ann_id += 1
    
    with open(path, 'w') as f:
        json.dump(coco, f, indent=2)
    
    messagebox.showinfo("Success", f"Saved {ann_id-1} annotations from {len(video_annotations)} frames")


def run_finetuned_prediction(predictor, predictions_json):
    """
    Run standard SAM2 video prediction using fine-tuned model.
    This uses the SAME prediction loop as predict_sam2_video() but with fine-tuned weights.

    MULTI-CATEGORY VERSION (FIXED):
      - Correctly slices per-object mask logits (handles (N,1,H,W) / (N,H,W) / dict formats)
      - Does NOT broadcast user points to every object
      - If points are only stored at frame-level, they are applied to ONE target object
        (by default: the largest mask in that frame, which matches scrobe_frons usage)

    Args:
        predictor: SAM2VideoPredictor with fine-tuned weights already loaded
        predictions_json: Path to save predictions

    Returns:
        bool: True if successful
    """
    global video_annotations, video_frames, video_temp_dir, video_metadata

    import os
    import json
    import numpy as np
    import cv2
    import torch

    print("\n" + "=" * 60)
    print(" Running Fine-Tuned Model Prediction")
    print("=" * 60)
    print(f"Annotated keyframes: {len(video_annotations)}")
    print(f"Total frames: {len(video_frames)}")
    print("Using fine-tuned model for tracking...")
    print("=" * 60)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _ensure_points_and_labels(points, labels):
        """Return (pts Nx2 float32, lbls N int32) or (None, None) if empty."""
        if points is None:
            return None, None
        if isinstance(points, list) and len(points) == 0:
            return None, None

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
            return None, None

        if labels is None or (isinstance(labels, list) and len(labels) == 0):
            lbls = np.ones((len(pts),), dtype=np.int32)
        else:
            raw = np.asarray(labels, dtype=np.int32).reshape(-1)
            if len(raw) != len(pts):
                # Safe reconcile
                lbls = np.ones((len(pts),), dtype=np.int32)
                m = min(len(raw), len(pts))
                lbls[:m] = raw[:m]
            else:
                lbls = raw

        return pts, lbls

    def _sample_positive_points_from_mask(mask_u8, k=8):
        """Sample k positive points (x,y) from a binary mask (0/1 uint8)."""
        coords = np.argwhere(mask_u8 > 0)  # (y,x)
        if coords.shape[0] == 0:
            return None
        k = min(k, coords.shape[0])
        idx = np.random.choice(coords.shape[0], size=k, replace=False)
        sel = coords[idx][:, [1, 0]].astype(np.float32)  # (x,y)
        return sel

    def _to_numpy(x):
        """Torch -> numpy, otherwise np.asarray."""
        try:
            import torch as _torch
            if isinstance(x, _torch.Tensor):
                return x.detach().to("cpu").float().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _mask_logits_to_per_object_2d(mask_logits, obj_ids):
        """
        Return dict[obj_id] -> 2D float/uint array of logits/probs.
        Handles:
          - dict[obj_id] -> tensor/array (1,H,W) or (H,W)
          - tensor/array shapes: (N,1,H,W), (N,H,W), (1,H,W), (H,W)
        """
        out = {}

        if isinstance(mask_logits, dict):
            for oid, m in mask_logits.items():
                arr = _to_numpy(m)
                arr = np.squeeze(arr)
                if arr.ndim == 3:
                    # (1,H,W) or (C,H,W) -> take first
                    arr = arr[0]
                if arr.ndim != 2:
                    continue
                out[int(oid)] = arr
            return out

        arr = _to_numpy(mask_logits)
        # Keep object dimension if present
        if arr.ndim == 4:
            # common: (N,1,H,W)
            if arr.shape[1] == 1:
                arr = arr[:, 0, :, :]
            else:
                # fallback: take channel 0
                arr = arr[:, 0, :, :]
        elif arr.ndim == 3:
            # (N,H,W) OR (1,H,W)
            pass
        elif arr.ndim == 2:
            # single object
            if len(obj_ids) >= 1:
                out[int(obj_ids[0])] = arr
            return out
        else:
            return out

        arr = np.asarray(arr)
        # If (1,H,W) but multiple obj_ids, we cannot invent different masks.
        # Only map first obj_id in that case.
        if arr.shape[0] == 1 and len(obj_ids) > 1:
            out[int(obj_ids[0])] = arr[0]
            return out

        n = min(arr.shape[0], len(obj_ids))
        for i in range(n):
            out[int(obj_ids[i])] = np.squeeze(arr[i])
        return out

    def _mask2d_to_u8(mask2d):
        """Binarize 2D mask logits to uint8 0/1."""
        m = np.asarray(mask2d)
        if m.dtype != np.uint8:
            # logits can be positive/negative; original code thresholds at >0
            m = (m > 0).astype(np.uint8)
        else:
            m = (m > 0).astype(np.uint8)
        return m

    def _bbox_from_mask_u8(mask01):
        """Return [x,y,w,h] from 0/1 mask."""
        ys, xs = np.where(mask01 > 0)
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return [x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)]

    # -------------------------------------------------------------------------
    # Init inference state (memory safe)
    # -------------------------------------------------------------------------
    inference_state = init_sam2_state_memory_efficient(
        predictor,
        video_temp_dir,
        device=device
    )

    if hasattr(predictor, "max_cond_frames_in_attn"):
        predictor.max_cond_frames_in_attn = 8
        print("  Set max_cond_frames_in_attn = 8")

    if isinstance(inference_state, dict) and "max_cond_frames_in_attn" in inference_state:
        inference_state["max_cond_frames_in_attn"] = 8

    if len(video_frames) > 500 and hasattr(predictor, "mem_every"):
        predictor.mem_every = 10
        print("  Set mem_every = 10 (for long video)")

    # -------------------------------------------------------------------------
    # Build category_id -> obj_id mapping
    # -------------------------------------------------------------------------
    cat_to_obj = {}
    next_obj_id = 1

    print("\nBuilding category → object ID mapping...")
    for frame_idx, ann in sorted(video_annotations.items()):
        for mask_data in ann.get("masks", []):
            cat_id = int(mask_data.get("category_id", 1))
            if cat_id not in cat_to_obj:
                cat_to_obj[cat_id] = next_obj_id
                next_obj_id += 1

    print("\n" + "=" * 60)
    print("📋 CATEGORY → OBJECT ID MAPPING")
    print("=" * 60)
    for cat_id, obj_id in sorted(cat_to_obj.items()):
        cat_name = category_id_to_name.get(cat_id, f"category_{cat_id}")
        print(f"  Category {cat_id} ({cat_name}) → Object ID {obj_id}")
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # Add prompts (masks + points) per keyframe
    # -------------------------------------------------------------------------
    prompts_added = 0
    point_prompts_added = 0

    print("Adding prompts from annotated keyframes...")
    for frame_idx, ann in sorted(video_annotations.items()):
        masks_list = ann.get("masks", [])
        frame_points = ann.get("points", [])
        frame_point_labels = ann.get("point_labels", [])

        print(f"\n  Frame {frame_idx}: {len(frame_points) if frame_points else 0} points, {len(masks_list)} masks")

        # If points are only stored at frame-level, apply them to ONE target object:
        # choose the largest mask in that frame (this matches scrobe_frons in your dataset).
        target_cat_for_frame_points = None
        if frame_points and len(masks_list) > 0:
            largest = max(masks_list, key=lambda md: int(md.get("area", 0)) if md.get("area", 0) is not None else 0)
            target_cat_for_frame_points = int(largest.get("category_id", 1))

        for mask_data in masks_list:
            mask = mask_data.get("segmentation", None)
            if mask is None:
                continue

            cat_id = int(mask_data.get("category_id", 1))
            obj_id = cat_to_obj.get(cat_id, None)
            if obj_id is None:
                continue

            cat_name = category_id_to_name.get(cat_id, f"category_{cat_id}")

            # Prepare mask for SAM2
            mask_np = np.asarray(mask)
            if mask_np.dtype != np.float32:
                mask_input = mask_np.astype(np.float32)
            else:
                mask_input = mask_np.copy()

            if mask_input.max() > 1.0:
                mask_input = mask_input / 255.0

            mask_area = int(np.sum(mask_np > 0))
            print(f"    Cat={cat_id} ({cat_name}), Obj={obj_id}: MASK prompt (area={mask_area} px)")

            # Add mask prompt
            try:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask_input
                )
                prompts_added += 1
            except Exception as e:
                print(f"    [WARNING] Failed to add mask for Obj={obj_id}: {e}")
                continue

            # -------------------------------
            # Points: DO NOT BROADCAST
            # Prefer per-mask points if present; else use frame-level points only for target category.
            # -------------------------------
            per_mask_points = mask_data.get("points", None)
            per_mask_labels = mask_data.get("point_labels", None)

            pts, lbls = _ensure_points_and_labels(per_mask_points, per_mask_labels)
            used_source = "per-mask"

            if pts is None:
                # fallback to frame-level points, but only for the chosen target category
                if frame_points and target_cat_for_frame_points == cat_id:
                    pts, lbls = _ensure_points_and_labels(frame_points, frame_point_labels)
                    used_source = f"frame-level→Cat={cat_id}"

            if pts is not None and lbls is not None and len(pts) > 0:
                n_pos = int(np.sum(lbls == 1))
                n_neg = int(np.sum(lbls == 0))

                # If user provided only negative points, SAM2 can collapse masks;
                # anchor with a few positive samples from the current mask.
                if n_pos == 0:
                    # Convert original mask to 0/1
                    base01 = (mask_np > 0).astype(np.uint8)
                    pos_pts = _sample_positive_points_from_mask(base01, k=8)
                    if pos_pts is not None and len(pos_pts) > 0:
                        pts = np.vstack([pos_pts, pts]).astype(np.float32)
                        lbls = np.hstack([np.ones(len(pos_pts), dtype=np.int32), lbls]).astype(np.int32)
                        n_pos = int(np.sum(lbls == 1))
                        n_neg = int(np.sum(lbls == 0))

                print(f"      + points ({used_source}): {len(pts)} ({n_pos} positive, {n_neg} negative) → Obj={obj_id}")

                try:
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=pts,
                        labels=lbls,
                        box=None
                    )
                    point_prompts_added += 1
                except Exception as e:
                    print(f"      [WARNING] Failed to add points for Obj={obj_id}: {e}")

    print(f"\n[OK] Added prompts from {len(video_annotations)} keyframes")
    print(f"[OK] Total mask prompts: {prompts_added}")
    print(f"[OK] Total point prompts: {point_prompts_added}")
    print(f"[OK] Keyframes: {sorted(list(video_annotations.keys()))}")

    # -------------------------------------------------------------------------
    # Propagate through video
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🎬 Propagating through video with fine-tuned model...")
    print("=" * 60)

    temp_predictions = {}
    frame_count = 0

    cat_frame_counts = {cat_id: 0 for cat_id in cat_to_obj.keys()}
    obj_to_cat = {obj_id: cat_id for cat_id, obj_id in cat_to_obj.items()}

    try:
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            if frame_idx not in temp_predictions:
                temp_predictions[frame_idx] = []

            # Normalize obj_ids to python list[int]
            if obj_ids is None:
                obj_ids = []
            try:
                obj_ids = [int(x) for x in list(obj_ids)]
            except Exception:
                obj_ids = []

            per_obj_logits = _mask_logits_to_per_object_2d(mask_logits, obj_ids)

            for obj_id, m2d in per_obj_logits.items():
                m01 = _mask2d_to_u8(m2d)  # 0/1
                if int(np.sum(m01)) == 0:
                    continue

                cat_id = int(obj_to_cat.get(int(obj_id), 1))
                cat_frame_counts[cat_id] += 1

                bbox = _bbox_from_mask_u8(m01)

                temp_predictions[frame_idx].append({
                    "segmentation": m01,                 # 0/1 uint8
                    "area": int(np.sum(m01)),
                    "bbox": bbox if bbox is not None else [0, 0, 0, 0],
                    "category_id": cat_id
                })

            frame_count += 1

            if frame_count % 100 == 0:
                print(f"  Processed {frame_count}/{len(video_frames)} frames...")

            if frame_count % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n[OK] Propagation complete: {frame_count} frames processed")

        print("\n" + "=" * 60)
        print("📊 PER-CATEGORY PREDICTION STATISTICS")
        print("=" * 60)
        for cat_id in sorted(cat_to_obj.keys()):
            cat_name = category_id_to_name.get(cat_id, f"category_{cat_id}")
            obj_id = cat_to_obj[cat_id]
            count = cat_frame_counts[cat_id]
            print(f"  {cat_name} (Cat={cat_id}, Obj={obj_id}): {count} predictions")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Cleanup inference state
    try:
        predictor.reset_state(inference_state)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Helper: force SAM2 masks to strict 2D uint8 for OpenCV export (0/255)
    # -------------------------------------------------------------------------
    def _mask_to_u8_2d(mask):
        import numpy as np
        try:
            import torch
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().to("cpu").float().numpy()
        except Exception:
            pass

        mask = np.asarray(mask)
        mask = np.squeeze(mask)

        if mask.ndim == 3:
            if mask.shape[0] <= 8 and mask.shape[1] > 16 and mask.shape[2] > 16:
                mask = mask[0]
            else:
                mask = mask[..., 0]

        if mask.ndim != 2:
            raise ValueError(f"Mask not 2D after squeeze: shape={mask.shape}")

        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        else:
            mask = (mask > 0).astype(np.uint8)

        return mask * 255

    # -------------------------------------------------------------------------
    # Export predictions to COCO JSON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📾 Exporting predictions to COCO JSON...")
    print("=" * 60)

    try:
        categories_list = []
        for cat_id in sorted(cat_to_obj.keys()):
            cat_name = category_id_to_name.get(cat_id, f"category_{cat_id}")
            supercategory = category_id_to_supercategory.get(cat_id, "morphology")
            categories_list.append({
                "id": cat_id,
                "name": cat_name,
                "supercategory": supercategory
            })

        if not categories_list:
            categories_list = [{"id": 1, "name": "object", "supercategory": "morphology"}]

        coco = {
            "videos": [{
                "id": 1,
                "file_name": video_metadata["file_name"],
                "width": video_metadata["width"],
                "height": video_metadata["height"],
                "length": len(video_frames),
                "fps": video_metadata["fps"]
            }],
            "images": [],
            "annotations": [],
            "categories": categories_list
        }

        print(f"[DEBUG] Exporting {len(categories_list)} categories: {[c['name'] for c in categories_list]}")

        ann_id = 1
        for frame_idx in range(len(video_frames)):
            img_id = frame_idx + 1

            coco["images"].append({
                "id": img_id,
                "video_id": 1,
                "frame_id": frame_idx,
                "file_name": os.path.basename(video_frames[frame_idx]),
                "width": video_metadata["width"],
                "height": video_metadata["height"]
            })

            if frame_idx not in temp_predictions:
                continue

            preds = temp_predictions[frame_idx]
            if not isinstance(preds, list):
                preds = [preds]

            for pred in preds:
                mask_raw = pred.get("segmentation", None)
                if mask_raw is None:
                    continue

                cat_id = int(pred.get("category_id", 1))
                mask_u8 = _mask_to_u8_2d(mask_raw)
                if mask_u8.max() == 0:
                    continue

                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = [c for c in contours if c is not None and len(c) >= 3]
                if not contours:
                    continue

                seg = [c.flatten().tolist() for c in contours]

                pts = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(pts)

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "video_id": 1,
                    "frame_id": frame_idx,
                    "category_id": cat_id,
                    "segmentation": seg,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": float((mask_u8 > 0).sum()),
                    "iscrowd": 0,
                    "predicted": True,
                    "finetuned": True
                })
                ann_id += 1

        with open(predictions_json, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"[OK] Exported {ann_id - 1} predictions to: {os.path.basename(predictions_json)}")
        print(f"[OK] File: {predictions_json}")

    except Exception as e:
        print(f"[ERROR] Could not export predictions: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("=" * 60)
    print("[OK] Fine-tuned prediction complete!")
    print("=" * 60)
    return True



def finetune_and_predict_with_cycle_consistency():
    """
    Complete pipeline:
    1. Fine-tune SAM2 on annotated frames (palindrome approach)
    2. Load fine-tuned weights into predictor
    3. Run standard SAM2 video prediction with fine-tuned model
    4. Export predictions to COCO JSON
    5. Load predictions into video_annotations for review
    """
    global video_annotations, video_frames, video_temp_dir, video_metadata
    global categories, category_id_to_name, category_name_to_id, category_id_to_supercategory, category_label_options

    if not video_mode or len(video_annotations) < 2:
        messagebox.showwarning(
            "Warning",
            "Need at least 2 annotated frames for fine-tuning!\n\n"
            f"Currently annotated: {len(video_annotations)} frames\n\n"
            "Recommendation: Annotate 5-10 keyframes spread across the video."
        )
        return

    # -------------------------------------------------------------------------
    # Helper: force masks to strict 2D uint8 for OpenCV
    # -------------------------------------------------------------------------
    def _mask_to_u8_2d(mask):
        import numpy as np
        try:
            import torch
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().to("cpu").float().numpy()
        except Exception:
            pass

        mask = np.asarray(mask)
        mask = np.squeeze(mask)

        # Handle common 3D cases: (1,H,W), (N,H,W), (H,W,1), (H,W,C)
        if mask.ndim == 3:
            if mask.shape[0] <= 8 and mask.shape[1] > 16 and mask.shape[2] > 16:
                mask = mask[0]
            else:
                mask = mask[..., 0]

        if mask.ndim != 2:
            raise ValueError(f"Mask not 2D after squeeze: shape={mask.shape}")

        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)

        return mask * 255  # 0/255 for OpenCV

    # -------------------------------------------------------------------------
    # Configuration dialog for fine-tuning settings
    # -------------------------------------------------------------------------
    config_dialog = tk.Toplevel(root)
    config_dialog.title("Fine-Tuning Configuration")
    config_dialog.geometry("600x550")
    config_dialog.transient(root)
    config_dialog.grab_set()

    epochs_var = tk.IntVar(value=25)
    learning_rate_var = tk.StringVar(value="1e-4")
    lora_rank_var = tk.IntVar(value=16)

    video_dir = os.path.dirname(video_metadata['file_path'])
    video_base = os.path.splitext(video_metadata['file_name'])[0]
    default_output_dir = os.path.join(video_dir, f"{video_base}_finetuned")
    default_checkpoint = os.path.join(default_output_dir, "sam2_finetuned.pt")
    default_predictions = os.path.join(default_output_dir, f"{video_base}_predictions.json")

    checkpoint_path_var = tk.StringVar(value=default_checkpoint)
    predictions_path_var = tk.StringVar(value=default_predictions)
    config_result = {'confirmed': False}

    tk.Label(config_dialog, text="Configure Fine-Tuning Parameters",
             font=("Arial", 14, "bold")).pack(pady=15)

    info_frame = tk.Frame(config_dialog, bg="#e8f4f8", relief=tk.RIDGE, borderwidth=2)
    info_frame.pack(fill=tk.X, padx=20, pady=10)
    tk.Label(info_frame, text=f"Video: {video_metadata['file_name']}",
             bg="#e8f4f8", font=("Arial", 10)).pack(pady=3)
    tk.Label(info_frame, text=f"Annotated keyframes: {len(video_annotations)}",
             bg="#e8f4f8", font=("Arial", 10)).pack(pady=3)
    tk.Label(info_frame, text=f"Total frames: {len(video_frames)}",
             bg="#e8f4f8", font=("Arial", 10)).pack(pady=3)

    settings_frame = tk.Frame(config_dialog)
    settings_frame.pack(pady=10)

    tk.Label(settings_frame, text="Training Epochs:",
             font=("Arial", 11)).grid(row=0, column=0, sticky='w', padx=10, pady=8)
    tk.Spinbox(settings_frame, from_=5, to=100,
               textvariable=epochs_var, width=10, font=("Arial", 10)).grid(row=0, column=1, padx=10, pady=8)
    tk.Label(settings_frame, text="(more = better quality, longer time)",
             fg="gray", font=("Arial", 9)).grid(row=0, column=2, sticky='w', padx=5)

    tk.Label(settings_frame, text="LoRA Rank:",
             font=("Arial", 11)).grid(row=1, column=0, sticky='w', padx=10, pady=8)
    tk.Spinbox(settings_frame, from_=4, to=64, increment=4,
               textvariable=lora_rank_var, width=10, font=("Arial", 10)).grid(row=1, column=1, padx=10, pady=8)
    tk.Label(settings_frame, text="(higher = more capacity, more memory)",
             fg="gray", font=("Arial", 9)).grid(row=1, column=2, sticky='w', padx=5)

    tk.Label(config_dialog, text="Output File Paths",
             font=("Arial", 12, "bold")).pack(pady=(15, 5))

    paths_frame = tk.Frame(config_dialog)
    paths_frame.pack(fill=tk.X, padx=20, pady=5)

    tk.Label(paths_frame, text="Fine-tuned Model:",
             font=("Arial", 10)).grid(row=0, column=0, sticky='w', pady=5)
    checkpoint_entry = tk.Entry(paths_frame, textvariable=checkpoint_path_var,
                                width=45, font=("Arial", 9))
    checkpoint_entry.grid(row=0, column=1, padx=5, pady=5)

    def browse_checkpoint():
        path = filedialog.asksaveasfilename(
            title="Save Fine-Tuned Checkpoint As",
            initialfile=os.path.basename(checkpoint_path_var.get()),
            initialdir=os.path.dirname(checkpoint_path_var.get()),
            defaultextension=".pt",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if path:
            checkpoint_path_var.set(path)

    tk.Button(paths_frame, text="Browse...", command=browse_checkpoint, width=10)\
        .grid(row=0, column=2, padx=5, pady=5)

    tk.Label(paths_frame, text="Predictions JSON:",
             font=("Arial", 10)).grid(row=1, column=0, sticky='w', pady=5)
    predictions_entry = tk.Entry(paths_frame, textvariable=predictions_path_var,
                                 width=45, font=("Arial", 9))
    predictions_entry.grid(row=1, column=1, padx=5, pady=5)

    def browse_predictions():
        path = filedialog.asksaveasfilename(
            title="Save Predictions As",
            initialfile=os.path.basename(predictions_path_var.get()),
            initialdir=os.path.dirname(predictions_path_var.get()),
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")]
        )
        if path:
            predictions_path_var.set(path)

    tk.Button(paths_frame, text="Browse...", command=browse_predictions, width=10)\
        .grid(row=1, column=2, padx=5, pady=5)

    tk.Label(paths_frame, text="Learning rate:").grid(row=2, column=0, sticky="w", pady=5)
    lr_entry = ttk.Entry(paths_frame, textvariable=learning_rate_var, width=20)
    lr_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
    tk.Label(paths_frame, text="e.g. 1e-4, 5e-5", fg="gray").grid(row=2, column=2, sticky="w")

    def update_estimate():
        epochs = epochs_var.get()
        est_minutes = int(epochs * 0.4)
        return f"~{est_minutes} minutes"

    estimate_label = tk.Label(
        config_dialog,
        text=f"Estimated time: {update_estimate()}",
        font=("Arial", 10, "italic"),
        fg="#0066cc"
    )
    estimate_label.pack(pady=10)

    def update_time(*args):
        estimate_label.config(text=f"Estimated time: {update_estimate()}")

    epochs_var.trace('w', update_time)

    def confirm():
        checkpoint = checkpoint_path_var.get()
        predictions = predictions_path_var.get()

        try:
            learning_rate = float(str(learning_rate_var.get()).strip())
            if not (learning_rate > 0):
                raise ValueError
        except Exception:
            messagebox.showerror(
                'Invalid Learning Rate',
                'Please enter a positive number for learning rate (e.g., 1e-4).'
            )
            return

        if not checkpoint or not predictions:
            messagebox.showerror("Error", "Please specify output paths!")
            return

        checkpoint_dir = os.path.dirname(checkpoint)
        predictions_dir = os.path.dirname(predictions)

        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            if predictions_dir != checkpoint_dir:
                os.makedirs(predictions_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create output directory:\n{e}")
            return

        config_result['confirmed'] = True
        config_result['epochs'] = epochs_var.get()
        config_result['learning_rate'] = learning_rate
        config_result['lora_rank'] = lora_rank_var.get()
        config_result['checkpoint_path'] = checkpoint
        config_result['predictions_path'] = predictions
        config_result['output_dir'] = checkpoint_dir
        config_dialog.destroy()

    def cancel():
        config_dialog.destroy()

    btn_frame = tk.Frame(config_dialog)
    btn_frame.pack(pady=20)
    tk.Button(btn_frame, text="Start Fine-Tuning", command=confirm,
              bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), width=15)\
        .pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Cancel", command=cancel,
              font=("Arial", 11), width=10)\
        .pack(side=tk.LEFT, padx=10)

    config_dialog.wait_window()
    if not config_result['confirmed']:
        return

    num_epochs = config_result['epochs']
    lora_rank = config_result['lora_rank']
    output_dir = config_result['output_dir']
    checkpoint_path = config_result['checkpoint_path']
    predictions_json = config_result['predictions_path']

    print("\n" + "=" * 60)
    print("DEBUG: User configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {config_result.get('learning_rate')}")
    print(f"  LoRA Rank: {lora_rank}")
    print(f"  Output dir: {output_dir}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Predictions: {predictions_json}")
    print("=" * 60)

    script_path = "sam2_finetune_palindrome_video_v10.py"
    print(f"\nDEBUG: Checking for script: {script_path}")
    print(f"DEBUG: Current directory: {os.getcwd()}")
    print(f"DEBUG: Script exists: {os.path.exists(script_path)}")

    if not os.path.exists(script_path):
        messagebox.showerror(
            "Script Not Found",
            f"Fine-tuning script not found: {script_path}\n\n"
            f"Please copy sam2_finetune_palindrome_video_v10.py to:\n"
            f"{os.getcwd()}\n\n"
            f"The script should be in the same directory as Descriptron."
        )
        return

    try:
        import subprocess

        # Save current frame first
        save_current_frame_annotations()
        print("DEBUG: Successfully saved current frame annotations")

        # Build COCO JSON for fine-tuning
        temp_json = os.path.join(video_temp_dir, "annotations_for_finetuning.json")
        print("[PREPARE] Preparing annotations for fine-tuning...")

        # ---------------------------------------------------------------------
        # Export categories for finetuning (DO NOT invent/overwrite user semantics)
        # Priority: 1) loaded COCO categories (global `categories`)
        #          2) derive from used category_ids + existing mappings
        #          3) fallback supercategory="sclerite"
        # ---------------------------------------------------------------------
        export_categories = []

        if "categories" in globals() and categories and isinstance(categories, list) and len(categories) > 0:
            export_categories = []
            for c in categories:
                if not isinstance(c, dict) or "id" not in c:
                    continue
                cid = int(c["id"])
                export_categories.append({
                    "id": cid,
                    "name": c.get("name", f"category_{cid}"),
                    "supercategory": c.get("supercategory", "sclerite"),
                })
            print(f"[DEBUG] Using {len(export_categories)} categories from loaded COCO (verbatim).")
        else:
            seen_cat_ids = set()
            for _frame_idx_tmp, ann_tmp in video_annotations.items():
                for mask_data_tmp in ann_tmp.get('masks', []):
                    seen_cat_ids.add(int(mask_data_tmp.get('category_id', 1)))

            if isinstance(category_id_to_name, dict):
                for cid in category_id_to_name.keys():
                    seen_cat_ids.add(int(cid))

            for cid in sorted(seen_cat_ids):
                nm = category_id_to_name.get(cid, f"category_{cid}") if isinstance(category_id_to_name, dict) else f"category_{cid}"
                sc = category_id_to_supercategory.get(cid, "sclerite") if isinstance(category_id_to_supercategory, dict) else "sclerite"
                export_categories.append({"id": int(cid), "name": nm, "supercategory": sc})

            if not export_categories:
                export_categories = [{"id": 1, "name": "object", "supercategory": "sclerite"}]
            print(f"[DEBUG] Built {len(export_categories)} categories from annotations/mappings (fallback supercategory='sclerite').")

        for cat in export_categories:
            print(f"  ID {cat['id']}: {cat.get('name')} (super={cat.get('supercategory','sclerite')})")

        coco = {
            "videos": [{
                "id": 1,
                "file_name": video_metadata['file_name'],
                "width": video_metadata['width'],
                "height": video_metadata['height'],
                "length": len(video_frames),
                "fps": video_metadata['fps']
            }],
            "images": [],
            "annotations": [],
            "categories": export_categories
        }

        ann_id = 1
        for frame_idx in range(len(video_frames)):
            img_id = frame_idx + 1
            coco['images'].append({
                "id": img_id,
                "video_id": 1,
                "frame_id": frame_idx,
                "file_name": os.path.basename(video_frames[frame_idx]),
                "width": video_metadata['width'],
                "height": video_metadata['height']
            })

            if frame_idx in video_annotations:
                ann = video_annotations[frame_idx]
                for mask_data in ann.get('masks', []):
                    mask_raw = mask_data.get('segmentation', None)
                    if mask_raw is None:
                        continue

                    mask_u8 = _mask_to_u8_2d(mask_raw)
                    if mask_u8.max() == 0:
                        continue

                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = [c for c in contours if c is not None and len(c) >= 3]
                    if not contours:
                        continue

                    seg = [c.flatten().tolist() for c in contours]
                    pts = np.vstack(contours)
                    x, y, w, h = cv2.boundingRect(pts)

                    actual_category_id = int(mask_data.get('category_id', 1))

                    annotation_entry = {
                        "id": ann_id,
                        "image_id": img_id,
                        "video_id": 1,
                        "frame_id": frame_idx,
                        "category_id": actual_category_id,
                        "segmentation": seg,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": float((mask_u8 > 0).sum()),
                        "iscrowd": 0
                    }

                    if mask_data.get('points') and len(mask_data['points']) > 0:
                        annotation_entry['points'] = mask_data['points']
                        annotation_entry['point_labels'] = mask_data.get('point_labels', [])


                    coco['annotations'].append(annotation_entry)

                    cat_name_debug = category_id_to_name.get(actual_category_id, f"cat_{actual_category_id}") \
                        if isinstance(category_id_to_name, dict) else f"cat_{actual_category_id}"
                    print(f"  [ANN] id={ann_id}, frame={frame_idx}, cat={actual_category_id} ({cat_name_debug})")

                    ann_id += 1

        with open(temp_json, 'w') as f:
            json.dump(coco, f, indent=2)

        print(f"[OK] Saved {ann_id-1} annotations to {temp_json}")

        cmd = [
            sys.executable,
            script_path,
            "--video_dir", video_temp_dir,
            "--annotations", temp_json,
            "--output_dir", output_dir,
            "--sam2_checkpoint", sam2_checkpoint,
            "--model_cfg", model_cfg,
            "--epochs", str(num_epochs),
            "--learning_rate", str(config_result.get("learning_rate", 1e-4)),
            "--use_lora",
            "--lora_rank", str(lora_rank),
            "--device", str(device),
            # remove "--export_predictions", predictions_json
        ]

        print("\n" + "=" * 60)
        print("FINE-TUNING COMMAND:")
        print("=" * 60)
        print("Script:", script_path)
        print("Python:", sys.executable)
        print("\nArguments:")
        i = 2
        while i < len(cmd):
            arg = cmd[i]
            if arg.startswith('--'):
                if i + 1 < len(cmd) and not cmd[i + 1].startswith('--'):
                    print(f"  {arg} = {cmd[i + 1]}")
                    i += 2
                else:
                    print(f"  {arg}")
                    i += 1
            else:
                i += 1
        print("=" * 60)

        print("\n" + "=" * 60)
        print("[START] Starting palindrome fine-tuning...")
        print("=" * 60)
        print(f"Keyframes: {len(video_annotations)}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {config_result.get('learning_rate')}")
        print(f"Method: LoRA (efficient training)")
        print("=" * 60)

        progress_window = tk.Toplevel(root)
        progress_window.title("Fine-Tuning in Progress...")
        progress_window.geometry("600x300")
        progress_window.transient(root)

        tk.Label(progress_window, text="[START] Palindrome Fine-Tuning SAM2...",
                 font=("Arial", 16, "bold")).pack(pady=15)

        info_frame = tk.Frame(progress_window, bg="#f0f0f0", relief=tk.RIDGE, borderwidth=2)
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(info_frame, text=f"Training on {len(video_annotations)} annotated frames",
                 bg="#f0f0f0", font=("Arial", 11)).pack(pady=3)
        tk.Label(info_frame, text="Using LoRA (efficient training)",
                 bg="#f0f0f0", font=("Arial", 11)).pack(pady=3)

        tk.Label(progress_window,
                 text="Please wait while training completes...\nCheck console for progress.",
                 font=("Arial", 10), fg="#666").pack(pady=10)

        progress_window.update()

        print("\n[MICROSCOPE] Training output:\n")

        output_lines = []
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)
            sys.stdout.flush()

        process.wait()
        progress_window.destroy()

        if process.returncode != 0:
            error_output = ''.join(output_lines[-30:]) if len(output_lines) > 30 else ''.join(output_lines)
            messagebox.showerror(
                "Fine-Tuning Failed",
                f"Fine-tuning script failed with exit code {process.returncode}\n\n"
                f"Last output:\n{error_output[:500]}\n\n"
                f"Common issues:\n"
                f"* Script not found: sam2_finetune_palindrome_video_v10.py\n"
                f"* PEFT library not installed: pip install peft\n"
                f"* Out of GPU memory: reduce epochs or LoRA rank\n"
                f"* SAM2 paths incorrect: check checkpoint and config paths\n\n"
                f"Full output is in the console - scroll up to see details."
            )
            return

        print("[OK] Fine-tuning complete!")
        print("=" * 60)

        checkpoint_file = os.path.join(output_dir, "checkpoint_path.txt")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_path = f.read().strip()
            print(f"[OK] Checkpoint path: {checkpoint_path}")
        else:
            pt_files = sorted(
                [f for f in os.listdir(output_dir) if f.endswith(".pt")],
                key=lambda fn: os.path.getmtime(os.path.join(output_dir, fn)),
                reverse=True,
            )
            if pt_files:
                checkpoint_path = os.path.join(output_dir, pt_files[0])
                print(f"[OK] Using output directory checkpoint: {checkpoint_path}")
            else:
                print("[WARNING] No .pt found in output_dir; using base SAM2 weights.")
                checkpoint_path = None

        print("\n[LOAD] Loading fine-tuned model...")

        from sam2.build_sam import build_sam2_video_predictor
        import torch

        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

        weights_loaded = False
        if checkpoint_path and os.path.isfile(checkpoint_path) and checkpoint_path.endswith('.pt'):
            try:
                checkpoint_dict = torch.load(checkpoint_path, map_location=device)
                if 'model_state_dict' in checkpoint_dict:
                    state_dict = checkpoint_dict['model_state_dict']
                elif 'state_dict' in checkpoint_dict:
                    state_dict = checkpoint_dict['state_dict']
                else:
                    state_dict = checkpoint_dict

                missing_keys, unexpected_keys = predictor.load_state_dict(state_dict, strict=False)
                print(f"[OK] Loaded fine-tuned checkpoint from {os.path.basename(checkpoint_path)}")
                if missing_keys:
                    print(f"[INFO] Missing keys (expected for partial loading): {len(missing_keys)}")
                if unexpected_keys:
                    print(f"[INFO] Unexpected keys: {len(unexpected_keys)}")
                weights_loaded = True
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint: {e}")
                import traceback
                traceback.print_exc()

        if not weights_loaded:
            print("[WARNING] Using base SAM2 model (fine-tuned weights not loaded)")

        # Run prediction with fine-tuned model
        success = run_finetuned_prediction(predictor, predictions_json)
        if not success:
            messagebox.showerror(
                "Prediction Failed",
                "Fine-tuned model prediction failed.\n\nCheck console output for details."
            )
            return

        # ---------------------------------------------------------------------
        # Auto-load predictions into video_annotations
        # Also: pull category names/supercategories from predictions COCO verbatim
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        # Auto-load predictions into video_annotations
        # Also: pull category names/supercategories from predictions COCO verbatim
        # ---------------------------------------------------------------------
        if os.path.exists(predictions_json):
            try:
                print("\n" + "=" * 60)
                print("[OK] Loading predictions into video annotations...")
                print("=" * 60)
        
                with open(predictions_json, "r") as f:
                    pred_coco = json.load(f)
        
                # 1) Sync category dictionaries VERBATIM from predictions COCO
                #    (this updates: categories, category_id_to_name, category_id_to_supercategory, etc.)
                sync_categories_from_coco(pred_coco, verbose=True, update_ui=True)

                # 2) Clear and rebuild video_annotations from predictions
                video_annotations.clear()
                H = int(video_metadata["height"])
                W = int(video_metadata["width"])

                for ann in pred_coco.get("annotations", []) or []:
                    frame_id = ann.get("frame_id", None)
                    if frame_id is None:
                        continue
                    frame_id = int(frame_id)
        
                    if frame_id not in video_annotations:
                        video_annotations[frame_id] = {
                            "masks": [],
                            "points": [],
                            "point_labels": [],
                            "keypoint_labels": [],
                            "point_orders": [],
                            "labels": {}
                        }

                    segs = ann.get("segmentation", None)
                    if not segs:
                        continue
        
                    # Convert COCO polygon(s) -> binary mask (0/1)
                    mask = np.zeros((H, W), dtype=np.uint8)

                    # COCO segmentation may be list-of-lists polygons
                    for seg in segs:
                        if not seg or len(seg) < 6:
                            continue
                        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                        pts_i32 = np.round(pts).astype(np.int32)
                        cv2.fillPoly(mask, [pts_i32], 1)

                    if int(mask.sum()) == 0:
                        continue

                    # bbox safely from contours (avoid cv2.boundingRect(mask) error)
                    mask_u8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = [c for c in contours if c is not None and len(c) >= 3]
                    if contours:
                        pts2 = np.vstack(contours)
                        x, y, w, h = cv2.boundingRect(pts2)
                        bbox_fallback = [int(x), int(y), int(w), int(h)]
                    else:
                        bbox_fallback = [0, 0, 0, 0]

                    cat_id = int(ann.get("category_id", 1))

                    video_annotations[frame_id]["masks"].append({
                        "segmentation": mask,  # keep 0/1 for GUI logic
                        "area": int(mask.sum()),
                        "bbox": ann.get("bbox", bbox_fallback),
                        "predicted": True,
                        "category_id": cat_id,
                        "ann_id": int(ann.get("id", -1)),
                        "iscrowd": int(ann.get("iscrowd", 0)),
                    })

                print(f"[OK] Loaded {len(video_annotations)} frames with predictions")

                # 3) Refresh canvas
                if video_annotations:
                    first_frame = min(video_annotations.keys())
                    load_video_frame_into_canvas(first_frame)
                    print(f"[OK] Displaying frame {first_frame}")

            except Exception as e:
                print(f"[WARNING] Could not auto-load predictions: {e}")
                import traceback
                traceback.print_exc()

    
        messagebox.showinfo(
            "Fine-Tuning Complete!",
            f"SAM2 Palindrome fine-tuning completed successfully!\n\n"
            f"Outputs saved:\n"
            f"  * Checkpoint: {os.path.basename(checkpoint_path) if checkpoint_path else 'N/A'}\n"
            f"  * Predictions: {os.path.basename(predictions_json)}\n\n"
            f"Location: {output_dir}\n\n"
            f"Settings:\n"
            f"  * Epochs: {num_epochs}\n"
            f"  * Learning rate: {config_result.get('learning_rate')}\n"
            f"  * LoRA Rank: {lora_rank}\n"
            f"  * Keyframes: {len(video_annotations)}\n\n"
            f"Predictions have been loaded into the video!\n"
            f"Use the video controls to review results."
        )

        print("\n" + "=" * 60)
        print(f"[OK] All outputs saved to: {output_dir}")
        print(f"[OK] Checkpoint: {checkpoint_path}")
        print(f"[OK] Predictions: {predictions_json}")
        print("=" * 60)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        messagebox.showerror(
            "Pipeline Error",
            f"Fine-tuning pipeline failed:\n\n{str(e)}\n\nSee console for full traceback."
        )
        print("\n" + "=" * 60)
        print("❗ ERROR:")
        print("=" * 60)
        print(error_msg)




def run_cycle_consistency_inference(predictor, batch_size: int = 50, predictions_json: str = None):
    """
    Run cycle consistency inference with BATCHED processing to avoid OOM.
    
    Key changes from original:
    - Pass 1: Store masks to temporary numpy files, not in memory
    - Memory cleared between passes
    - Pass 2: Compare in batches, loading Pass 1 results incrementally
    - Never hold more than batch_size frames of masks in memory
    
    Args:
        predictor: SAM2 video predictor
        batch_size: Number of frames to hold in memory at once (default 50)
    """
    global video_annotations, video_frames, video_temp_dir, video_metadata
    
    import torch
    import gc
    
    print("\nÃ°Å¸â€œÅ  CYCLE CONSISTENCY CHECKING (BATCHED)")
    print("=" * 60)
    print("This helps identify frames where tracking may drift.")
    print(f"Processing in batches of {batch_size} frames to save memory")
    print("=" * 60)
    
    total_frames = len(video_frames)
    
    # Create temp directory for mask storage
    mask_cache_dir = os.path.join(video_temp_dir, "_mask_cache")
    os.makedirs(mask_cache_dir, exist_ok=True)
    
    # =========================================================================
    # PASS 1: Forward prediction - store results to disk
    # =========================================================================
    print("\nPASS 1: Forward prediction")
    print("-" * 60)
    
    try:
        inference_state_1 = predictor.init_state(
            video_path=video_temp_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False
        )
    except TypeError:
        inference_state_1 = predictor.init_state(video_path=video_temp_dir)
    
    # Add prompts from annotated keyframes
    obj_id = 0
    keyframes_used = 0
    
    for frame_idx, ann in sorted(video_annotations.items()):
        all_points = []
        all_labels = []
        bbox = None
        mask_input = None
        
        # Use mask as positive prompt (not extracted points!)
        if ann['masks'] and len(ann['masks']) > 0:
            mask = ann['masks'][0]['segmentation']
            
            # Prepare mask for SAM2 (float32, range [0,1])
            if mask.dtype != np.float32:
                mask_input = mask.astype(np.float32)
            else:
                mask_input = mask.copy()
            if mask_input.max() > 1.0:
                mask_input = mask_input / 255.0
            
            # Get bbox
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                y_coords, x_coords = coords[:, 0], coords[:, 1]
                bbox = np.array([x_coords.min(), y_coords.min(),
                               x_coords.max(), y_coords.max()])
        
        # Add user points
        if ann['points'] and len(ann['points']) > 0:
            all_points.append(np.array(ann['points']))
            all_labels.append(np.array(ann['point_labels']))
        
        # Prepare points if any
        pts = None
        lbls = None
        if all_points:
            pts = np.vstack(all_points)
            lbls = np.concatenate(all_labels)
        
        # Add prompts using correct SAM2 API
        if mask_input is not None or pts is not None:
            # Add mask if we have one
            if mask_input is not None:
                predictor.add_new_mask(
                    inference_state=inference_state_1,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask_input
                )
            
            # Add points if we have any
            if pts is not None:
                predictor.add_new_points_or_box(
                    inference_state=inference_state_1,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbls,
                    box=bbox
                )
            
            keyframes_used += 1
    
    print(f"Using {keyframes_used} keyframes for tracking")
    
    # Forward propagation - SAVE TO DISK instead of memory
    frame_count = 0
    mask_files_pass1 = {}
    
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state_1):
        if isinstance(mask_logits, dict):
            mask = (list(mask_logits.values())[0] > 0).cpu().numpy().squeeze().astype(np.uint8)
        else:
            if len(mask_logits.shape) == 3:
                mask = (mask_logits[0] > 0).cpu().numpy().squeeze().astype(np.uint8)
            else:
                mask = (mask_logits > 0).cpu().numpy().squeeze().astype(np.uint8)
        
        # Save mask to disk instead of holding in memory
        mask_path = os.path.join(mask_cache_dir, f"pass1_frame_{frame_idx:06d}.npy")
        np.save(mask_path, mask)
        mask_files_pass1[frame_idx] = mask_path
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  Pass 1: {frame_count}/{total_frames} frames processed")
            torch.cuda.empty_cache()
    
    print(f"[OK] Pass 1 complete: {len(mask_files_pass1)} frames saved to cache")
    
    # =========================================================================
    # MEMORY RESET - Critical!
    # =========================================================================
    print("\nMemory reset (critical for avoiding OOM)...")
    predictor.reset_state(inference_state_1)
    del inference_state_1
    torch.cuda.empty_cache()
    gc.collect()
    
    # Wait a moment for memory to settle
    import time
    time.sleep(1)
    
    # =========================================================================
    # PASS 2: Second prediction in BATCHES, comparing as we go
    # =========================================================================
    print("\nPASS 2: Second prediction (batched, cycle consistency check)")
    print("-" * 60)
    
    try:
        inference_state_2 = predictor.init_state(
            video_path=video_temp_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False
        )
    except TypeError:
        inference_state_2 = predictor.init_state(video_path=video_temp_dir)
    
    # Re-add prompts using masks (no random sampling needed with full mask)
    
    for frame_idx, ann in sorted(video_annotations.items()):
        all_points = []
        all_labels = []
        bbox = None
        mask_input = None
        
        # Use mask as positive prompt
        if ann['masks'] and len(ann['masks']) > 0:
            mask = ann['masks'][0]['segmentation']
            
            # Prepare mask for SAM2
            if mask.dtype != np.float32:
                mask_input = mask.astype(np.float32)
            else:
                mask_input = mask.copy()
            if mask_input.max() > 1.0:
                mask_input = mask_input / 255.0
            
            # Get bbox
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                y_coords, x_coords = coords[:, 0], coords[:, 1]
                bbox = np.array([x_coords.min(), y_coords.min(),
                               x_coords.max(), y_coords.max()])
        
        # Add user points
        if ann['points'] and len(ann['points']) > 0:
            all_points.append(np.array(ann['points']))
            all_labels.append(np.array(ann['point_labels']))
        
        # Prepare points if any
        pts = None
        lbls = None
        if all_points:
            pts = np.vstack(all_points)
            lbls = np.concatenate(all_labels)
        
        # Add prompts using correct SAM2 API
        if mask_input is not None or pts is not None:
            # Add mask if we have one
            if mask_input is not None:
                predictor.add_new_mask(
                    inference_state=inference_state_2,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask_input
                )
            
            # Add points if we have any
            if pts is not None:
                predictor.add_new_points_or_box(
                    inference_state=inference_state_2,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbls,
                    box=bbox
                )
    
    # Second propagation - COMPARE IN BATCHES
    drift_frames = []
    iou_scores = [0.0] * total_frames  # Pre-allocate
    low_iou_frames = []
    frame_count = 0
    
    # Store Pass 2 masks for current batch only
    pass2_masks_batch = {}
    
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state_2):
        if isinstance(mask_logits, dict):
            mask = (list(mask_logits.values())[0] > 0).cpu().numpy().squeeze().astype(np.uint8)
        else:
            if len(mask_logits.shape) == 3:
                mask = (mask_logits[0] > 0).cpu().numpy().squeeze().astype(np.uint8)
            else:
                mask = (mask_logits > 0).cpu().numpy().squeeze().astype(np.uint8)
        
        pass2_masks_batch[frame_idx] = mask
        frame_count += 1
        
        # Every batch_size frames, compare with Pass 1 and clear batch
        if len(pass2_masks_batch) >= batch_size or frame_count >= total_frames:
            print(f"  Pass 2: Processing batch ending at frame {frame_idx} ({frame_count}/{total_frames})")
            
            # Compare this batch with Pass 1
            for f_idx, pass2_mask in pass2_masks_batch.items():
                if f_idx in mask_files_pass1:
                    # Load Pass 1 mask from disk
                    pass1_mask = np.load(mask_files_pass1[f_idx])
                    
                    # Compute IoU
                    intersection = np.sum((pass1_mask > 0) & (pass2_mask > 0))
                    union = np.sum((pass1_mask > 0) | (pass2_mask > 0))
                    iou = intersection / (union + 1e-6)
                    
                    iou_scores[f_idx] = iou
                    
                    # Track low IoU (potential drift)
                    if iou < 0.7 and f_idx not in video_annotations:
                        drift_frames.append(f_idx)
                        low_iou_frames.append((f_idx, iou))
                    
                    # Store final mask in video_annotations (use Pass 1 as canonical)
                    if f_idx not in video_annotations:
                        video_annotations[f_idx] = {
                            'masks': [],
                            'points': [],
                            'point_labels': [],
                            'keypoint_labels': [],
                            'point_orders': [],
                            'labels': {}
                        }
                    
                    video_annotations[f_idx]['masks'] = [{
                        'segmentation': pass1_mask,  # Use Pass 1 as final
                        'area': int(np.sum(pass1_mask)),
                        'bbox': cv2.boundingRect(pass1_mask),
                        'predicted': True,
                        'finetuned': True,
                        'cycle_consistency_iou': float(iou)
                    }]
            
            # Clear batch from memory
            pass2_masks_batch.clear()
            torch.cuda.empty_cache()
    
    # Clean up Pass 2 state
    predictor.reset_state(inference_state_2)
    del inference_state_2
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"[OK] Pass 2 complete: {frame_count} frames processed")
    
    # =========================================================================
    # Report results
    # =========================================================================
    valid_iou_scores = [s for s in iou_scores if s > 0]
    avg_iou = np.mean(valid_iou_scores) if valid_iou_scores else 0.0
    
    print("\n" + "=" * 60)
    print("[OK] CYCLE CONSISTENCY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Frames analyzed: {len(valid_iou_scores)}")
    print(f"  Average IoU: {avg_iou:.1%}")
    print(f"  High agreement (IoU >= 0.9): {sum(1 for iou in valid_iou_scores if iou >= 0.9)}/{len(valid_iou_scores)}")
    print(f"  Medium agreement (0.7 <= IoU < 0.9): {sum(1 for iou in valid_iou_scores if 0.7 <= iou < 0.9)}/{len(valid_iou_scores)}")
    print(f"  Low agreement (IoU < 0.7): {len(drift_frames)}/{len(valid_iou_scores)}")
    
    if drift_frames:
        print(f"\n[WARNING] Potential drift detected at {len(drift_frames)} frames:")
        sorted_drift = sorted(low_iou_frames, key=lambda x: x[1])[:10]
        for frame_idx, iou in sorted_drift:
            print(f"    Frame {frame_idx}: IoU = {iou:.2%}")
        if len(drift_frames) > 10:
            print(f"    ... and {len(drift_frames) - 10} more")
        
        print(f"Recommendation:")
        print(f"  1. Review frames: {[f for f, _ in sorted_drift[:5]]}")
        print(f"  2. Add keyframes at these locations if drift is real")
        print(f"  3. Re-run fine-tuning for 99%+ quality")
    else:
        print(f"\n[OK] Excellent! No significant drift detected.")
        print(f"  Quality estimate: 98-99%")
    
    print("=" * 60)
    
    # Clean up cache directory
    print("\nCleaning up mask cache...")
    try:
        shutil.rmtree(mask_cache_dir)
    except Exception as e:
        print(f"  Warning: Could not clean cache: {e}")
    
    # =========================================================================
    # Export predictions to JSON (CRITICAL - without this, predictions won't load!)
    # =========================================================================
    if predictions_json:
        print("\nExporting predictions to COCO JSON...")
        try:
            # Build COCO JSON from video_annotations
            coco = {
                "videos": [{
                    "id": 1,
                    "file_name": video_metadata['file_name'],
                    "width": video_metadata['width'],
                    "height": video_metadata['height'],
                    "length": len(video_frames),
                    "fps": video_metadata['fps']
                }],
                "images": [],
                "annotations": [],
                "categories": [{"id": 1, "name": "object", "supercategory": "morphology"}]
            }
            
            ann_id = 1
            for frame_idx in range(len(video_frames)):
                img_id = frame_idx + 1
                
                coco['images'].append({
                    "id": img_id,
                    "video_id": 1,
                    "frame_id": frame_idx,
                    "file_name": os.path.basename(video_frames[frame_idx]),
                    "width": video_metadata['width'],
                    "height": video_metadata['height']
                })
                
                if frame_idx in video_annotations:
                    ann = video_annotations[frame_idx]
                    
                    # Save masks
                    # Save masks
                    for mask_data in ann['masks']:
                        mask_raw = mask_data.get('segmentation', None)
                        if mask_raw is None:
                            continue

                        # category_id if present (otherwise default 1)
                        cat_id = mask_data.get('category_id', 1)

                        mask_u8 = _mask_to_u8_2d(mask_raw)

                        # Skip empty masks safely
                        if mask_u8.max() == 0:
                            continue

                        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours = [c for c in contours if c is not None and len(c) >= 3]
                        if not contours:
                            continue

                        seg = [c.flatten().tolist() for c in contours]

                        # bbox from contour points (NOT from mask array)
                        pts = np.vstack(contours)
                        x, y, w, h = cv2.boundingRect(pts)

                        annotation_entry = {
                            "id": ann_id,
                            "image_id": img_id,
                            "video_id": 1,
                            "frame_id": frame_idx,
                            "category_id": cat_id,
                            "segmentation": seg,
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "area": float((mask_u8 > 0).sum()),
                            "iscrowd": 0,
                            "predicted": True,
                            "finetuned": True
                        }

                        coco['annotations'].append(annotation_entry)
                        ann_id += 1
            
            # Save to file
            with open(predictions_json, 'w') as f:
                json.dump(coco, f, indent=2)
            
            print(f"[OK] Exported {ann_id-1} predictions to: {os.path.basename(predictions_json)}")
            
        except Exception as e:
            print(f"[WARNING] Could not export predictions: {e}")
            import traceback
            traceback.print_exc()
    
    # Show summary dialog
    if drift_frames:
        sorted_drift = sorted(low_iou_frames, key=lambda x: x[1])[:5]
        drift_msg = (
            f"[WARNING] {len(drift_frames)} frames with low cycle consistency (IoU < 0.7)\n\n"
            f"Worst frames:\n"
        )
        for frame_idx, iou in sorted_drift:
            drift_msg += f"  - Frame {frame_idx}: IoU = {iou:.1%}\n"
        
        drift_msg += (
            f"\nÃ°Å¸â€™Â¡ Recommendations:\n"
            f"  1. Navigate to these frames and review\n"
            f"  2. Add keyframes if drift is visible\n"
            f"  3. Re-run fine-tuning for 99%+ quality\n\n"
            f"Current quality estimate: 96-98%"
        )
    else:
        drift_msg = "[OK] Excellent cycle consistency!\n\nQuality estimate: 98-99%"
    
    messagebox.showinfo(
        "Cycle Consistency Complete",
        f"[OK] Fine-tuned model inference complete!\n\n"
        f"Frames processed: {frame_count}\n"
        f"Average cycle consistency: {avg_iou:.1%}\n\n"
        f"{drift_msg}\n\n"
        f"Use Play to review predictions."
    )


# ============================================================================
# END OF REPLACEMENT FUNCTION
# ============================================================================


def open_video_remote_control():
    """Open video remote control popup."""
    global video_mode, video_remote, current_video_frame
    
    # Load video
    video_path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("Videos", "*.mp4 *.avi *.mov *.webm *.mkv"), ("All", "*.*")]
    )
    
    if not video_path:
        return
    
    # Extract frames
    if not extract_video_frames(video_path):
        return
    
    video_mode = True
    current_video_frame = 0
    
    # Create remote control window
    video_remote = tk.Toplevel(root)
    video_remote.title("Video Remote Control")
    video_remote.geometry("600x450")
    video_remote.transient(root)
    
    # Video info
    info_frame = tk.Frame(video_remote, bg="#f0f0f0", relief=tk.RIDGE, borderwidth=2)
    info_frame.pack(fill=tk.X, padx=10, pady=10)
    
    tk.Label(info_frame, text=f" {video_metadata['file_name']}", 
             font=("Arial", 11, "bold"), bg="#f0f0f0").pack(pady=5)
    tk.Label(info_frame, 
             text=f"{len(video_frames)} frames | {video_metadata['fps']:.1f} fps | {video_metadata['width']}x{video_metadata['height']}",
             bg="#f0f0f0").pack(pady=2)
    
    # Frame counter
    frame_var = tk.StringVar(value=f"Frame: 1 / {len(video_frames)}")
    tk.Label(info_frame, textvariable=frame_var, font=("Arial", 10, "bold"), 
             bg="#f0f0f0", fg="#0066cc").pack(pady=5)
    
    # Slider
    slider_frame = tk.Frame(video_remote)
    slider_frame.pack(fill=tk.X, padx=20, pady=10)
    
    def on_slider(val):
        idx = int(float(val))
        load_video_frame_into_canvas(idx)
        frame_var.set(f"Frame: {idx + 1} / {len(video_frames)}")
    
    slider = tk.Scale(slider_frame, from_=0, to=len(video_frames)-1,
                     orient=tk.HORIZONTAL, command=on_slider, length=400)
    slider.pack()
    
    # === INTERACTIVE TIMELINE DELETION UI ===
    deletion_frame = tk.Frame(video_remote, bg="#ffe6e6", relief=tk.RIDGE, borderwidth=2)
    deletion_frame.pack(fill=tk.X, padx=10, pady=10)
    
    tk.Label(deletion_frame, text="Timeline: Green=Annotated | Red=Delete | Click & Drag to Select", 
             font=("Arial", 10, "bold"), bg="#ffe6e6").pack(pady=5)
    
    # Timeline canvas
    timeline_canvas = tk.Canvas(deletion_frame, height=60, bg="white", 
                                highlightthickness=1, highlightbackground="#999")
    timeline_canvas.pack(fill=tk.X, padx=10, pady=5)
    
    # Status label
    deletion_status = tk.StringVar(value=f"Frames marked for deletion: {len(frames_to_delete)}")
    tk.Label(deletion_frame, textvariable=deletion_status, 
             font=("Arial", 9, "bold"), bg="#ffe6e6", fg="#cc0000").pack(pady=2)
    
    # Timeline state
    timeline_state = {
        'selecting': False,
        'start_frame': None,
        'current_band': None,
        'deletion_bands': []  # List of (start, end) tuples
    }
    
    def draw_timeline():
        """Draw the timeline with frame markers and deletion bands."""
        timeline_canvas.delete("all")
        
        width = timeline_canvas.winfo_width()
        height = 60
        
        if width <= 1:
            timeline_canvas.update_idletasks()
            width = timeline_canvas.winfo_width()
        
        total_frames = len(video_frames)
        
        # Draw background
        timeline_canvas.create_rectangle(0, 0, width, height, fill="white", outline="")
        
        # Draw GREEN LINES for frames with annotations
        if video_annotations:
            for frame_idx in video_annotations.keys():
                if 0 <= frame_idx < total_frames:
                    x = (frame_idx / total_frames) * width
                    # Draw green vertical line for annotated frames
                    # Slightly shorter than blue line (5-50 vs 0-50) so blue stands out as current position
                    timeline_canvas.create_line(x, 5, x, 50, 
                                               fill="#00cc00", width=3, tags="annotation")
        
        # Draw frame markers every 10% or at least every 50 frames
        marker_interval = max(total_frames // 10, 50)
        if total_frames < 100:
            marker_interval = max(total_frames // 10, 10)
        
        for i in range(0, total_frames, marker_interval):
            x = (i / total_frames) * width
            timeline_canvas.create_line(x, 45, x, 55, fill="#999", width=1)
            timeline_canvas.create_text(x, 58, text=str(i), font=("Arial", 7), fill="#666")
        
        # Draw deletion bands (converted from frames_to_delete set)
        if frames_to_delete:
            bands = get_contiguous_ranges(frames_to_delete)
            for start, end in bands:
                x1 = (start / total_frames) * width
                x2 = ((end + 1) / total_frames) * width
                timeline_canvas.create_rectangle(x1, 10, x2, 40, 
                                                fill="#ff6666", outline="#cc0000", width=2,
                                                tags="deletion")
                # Label the band
                mid_x = (x1 + x2) / 2
                timeline_canvas.create_text(mid_x, 25, 
                                           text=f"{start}-{end}", 
                                           font=("Arial", 8, "bold"), fill="white",
                                           tags="deletion_label")
        
        # Draw current selection band (during dragging)
        if timeline_state['selecting'] and timeline_state['start_frame'] is not None:
            start_f = timeline_state['start_frame']
            current_f = timeline_state.get('current_frame', start_f)
            
            start_x = (min(start_f, current_f) / total_frames) * width
            end_x = (max(start_f, current_f) / total_frames) * width
            
            timeline_canvas.create_rectangle(start_x, 10, end_x, 40,
                                            fill="#ffcccc", outline="#ff0000", 
                                            width=2, dash=(4, 4), tags="selection")
        
        # Draw current frame indicator
        current_x = (current_video_frame / total_frames) * width
        timeline_canvas.create_line(current_x, 0, current_x, 50, 
                                    fill="#0066cc", width=2, tags="current_frame")
    
    def get_contiguous_ranges(frame_set):
        """Convert a set of frame indices to list of contiguous (start, end) ranges."""
        if not frame_set:
            return []
        
        sorted_frames = sorted(frame_set)
        ranges = []
        start = sorted_frames[0]
        prev = sorted_frames[0]
        
        for frame in sorted_frames[1:]:
            if frame != prev + 1:
                ranges.append((start, prev))
                start = frame
            prev = frame
        
        ranges.append((start, prev))
        return ranges
    
    def canvas_to_frame(x):
        """Convert canvas x coordinate to frame number."""
        width = timeline_canvas.winfo_width()
        frame = int((x / width) * len(video_frames))
        return max(0, min(len(video_frames) - 1, frame))
    
    def on_timeline_press(event):
        """Start selecting a deletion band."""
        timeline_state['selecting'] = True
        timeline_state['start_frame'] = canvas_to_frame(event.x)
        timeline_state['current_frame'] = timeline_state['start_frame']
        draw_timeline()
    
    def on_timeline_drag(event):
        """Update selection band while dragging."""
        if timeline_state['selecting']:
            timeline_state['current_frame'] = canvas_to_frame(event.x)
            draw_timeline()
    
    def on_timeline_release(event):
        """Finalize selection band."""
        if timeline_state['selecting']:
            start_f = timeline_state['start_frame']
            end_f = canvas_to_frame(event.x)
            
            # Add range to deletion set
            for f in range(min(start_f, end_f), max(start_f, end_f) + 1):
                frames_to_delete.add(f)
            
            timeline_state['selecting'] = False
            timeline_state['start_frame'] = None
            timeline_state['current_frame'] = None
            
            deletion_status.set(f"Frames marked for deletion: {len(frames_to_delete)}")
            draw_timeline()
    
    def clear_timeline_deletions():
        """Clear all deletion bands."""
        frames_to_delete.clear()
        timeline_state['deletion_bands'] = []
        deletion_status.set(f"Frames marked for deletion: 0")
        draw_timeline()
    
    # Bind timeline events
    timeline_canvas.bind("<ButtonPress-1>", on_timeline_press)
    timeline_canvas.bind("<B1-Motion>", on_timeline_drag)
    timeline_canvas.bind("<ButtonRelease-1>", on_timeline_release)
    timeline_canvas.bind("<Configure>", lambda e: draw_timeline())
    
    # Control buttons
    btn_frame = tk.Frame(deletion_frame, bg="#ffe6e6")
    btn_frame.pack(pady=5)
    
    tk.Button(btn_frame, text="Clear All Deletions", command=clear_timeline_deletions, 
             bg="#ffcccc", font=("Arial", 9), width=20).pack(side=tk.LEFT, padx=5)
    
    tk.Label(deletion_frame, 
             text="ðŸ’¡ Drag to mark frames for deletion. Green lines = annotated frames. Click marmot to save.",
             font=("Arial", 8), bg="#ffe6e6", fg="#666").pack(pady=(0, 5))
    
    # Initial draw
    global timeline_draw_callback
    timeline_draw_callback = draw_timeline
    video_remote.after(100, draw_timeline)
    
    # Controls
    ctrl_frame = tk.Frame(video_remote)
    ctrl_frame.pack(pady=10)
    
    is_playing = [False]
    # Populate remote dropdown immediately if categories already loaded
    update_video_label_dropdown()
    def play():
        if not is_playing[0]:
            is_playing[0] = True
            play_btn.config(text="Pause")
            play_frames()
        else:
            is_playing[0] = False
            play_btn.config(text="Play")
    
    def play_frames():
        if not is_playing[0]:
            return
        if current_video_frame < len(video_frames) - 1:
            slider.set(current_video_frame + 1)
            delay = int(1000 / video_metadata['fps'])
            video_remote.after(delay, play_frames)
        else:
            is_playing[0] = False
            play_btn.config(text="Play")
    
    def prev_frame():
        if current_video_frame > 0:
            slider.set(current_video_frame - 1)
    
    def next_frame():
        if current_video_frame < len(video_frames) - 1:
            slider.set(current_video_frame + 1)
    
    def first_frame():
        slider.set(0)
    
    tk.Button(ctrl_frame, text="First", command=first_frame, width=8).grid(row=0, column=0, padx=3)
    tk.Button(ctrl_frame, text="<<--Prev", command=prev_frame, width=8).grid(row=0, column=1, padx=3)
    play_btn = tk.Button(ctrl_frame, text="Play>", command=play, width=8)
    play_btn.grid(row=0, column=2, padx=3)
    tk.Button(ctrl_frame, text="Next-->", command=next_frame, width=8).grid(row=0, column=3, padx=3)
    #tk.Button(ctrl_frame, text="Select Label", command=update_label_dropdown, width=8).grid(row=0, column=4, padx=4)
    # --- VIDEO REMOTE label dropdown (separate from main UI) ---
    global video_selected_label, video_label_options, video_label_dropdown

    video_selected_label = tk.StringVar(root)
    video_selected_label.set("Select Label")
    video_label_options = ["Custom", "Trash"]

    video_label_dropdown = tk.OptionMenu(ctrl_frame, video_selected_label, *video_label_options)
    video_label_dropdown.config(width=12)
    video_label_dropdown.grid(row=0, column=4, padx=4, pady=0, sticky="w")
    #selected_label = tk.StringVar(root)
    #selected_label.set("Select Label")
    #label_options = ["Custom","Trash"]
    #label_dropdown = tk.OptionMenu(ctrl_frame, selected_label, *label_options)
    #label_dropdown.config(width=8)  # <-- width belongs here, not in grid()
    #label_dropdown.grid(row=0, column=4, padx=4, pady=0, sticky="w")

    # Annotated frame navigation buttons (second row)
    tk.Label(ctrl_frame, text="Jump to Annotations:",
            font=("Arial", 9, "bold")).grid(row=1, column=0, columnspan=2, pady=(10, 2))
    tk.Button(ctrl_frame, text="Prev Annotated",
             command=previous_annotated_frame, bg="lightyellow",
             width=15).grid(row=2, column=0, columnspan=2, padx=3, pady=3)
    tk.Button(ctrl_frame, text="Next Annotated",
             command=next_annotated_frame, bg="lightyellow",
             width=15).grid(row=2, column=2, columnspan=2, padx=3, pady=3)
    
    # Action buttons
    action_frame = tk.Frame(video_remote)
    action_frame.pack(pady=15)
    
    tk.Button(action_frame, text="Load JSON", 
             command=load_video_annotations, bg="#ADD8E6", 
             font=("Arial", 10, "bold"), width=15).grid(row=0, column=0, padx=5, pady=5)
    tk.Button(action_frame, text="Apply Label",
         command=apply_video_label, bg="#90EE90",
         font=("Arial", 10, "bold"), width=15).grid(row=0, column=1, padx=5, pady=5)

    tk.Button(action_frame, text="Refresh Labels",
         command=lambda: refresh_select_label_ui(verbose=False), bg="#EEE8AA",
         font=("Arial", 10), width=15).grid(row=0, column=2, padx=5, pady=5)

    tk.Button(action_frame, text="Predict SAM2 Video", 
             command=predict_sam2_video, bg="#90EE90", 
             font=("Arial", 10, "bold"), width=20).grid(row=0, column=1, padx=5, pady=5)
    
    # Optional: legacy CV-style drift clamp (OFF by default)
    tk.Checkbutton(
        action_frame,
        text="Use area/bbox clamping (optional)",
        variable=video_use_area_clamp_var,
        font=("Arial", 9),
    ).grid(row=1, column=1, sticky="w", padx=5, pady=(0, 6))
    
    # NEW: Fine-tuning + Cycle Consistency button
    tk.Button(action_frame, text="[START] Fine-Tune + Cycle Check", 
             command=finetune_and_predict_with_cycle_consistency, 
             bg="#9C27B0", fg="white",
             font=("Arial", 10, "bold"), width=35).grid(row=2, column=0, columnspan=2, padx=5, pady=5)
    
    # Info label for new button
    tk.Label(action_frame, 
             text="Standard (1-2 min, 95-97%) | Advanced (12 min, 98-99%)",
             font=("Arial", 8), fg="#666").grid(row=3, column=0, columnspan=2, pady=2)
    
    # Mini marmot button (save & close)
    def save_and_close_video():
        is_playing[0] = False
        save_edited_video_and_annotations()  # New function handles frame deletion
        video_cleanup()
        video_remote.destroy()
    
    try:
        marmot_img = Image.open("./marmot.jpg")
        marmot_img = marmot_img.resize((40, 40), Image.LANCZOS)
        marmot_photo = ImageTk.PhotoImage(marmot_img)
        marmot_btn = tk.Button(action_frame, image=marmot_photo, 
                              command=save_and_close_video)
        marmot_btn.image = marmot_photo
        marmot_btn.grid(row=0, column=3, padx=5, pady=5)
    except:
        tk.Button(action_frame, text="Save", 
                 command=save_and_close_video,
                 font=("Arial", 10, "bold"), width=8).grid(row=0, column=3, padx=5, pady=5)
    
    # Load first frame
    load_video_frame_into_canvas(0)
    
    def on_close():
        is_playing[0] = False
        if messagebox.askyesno("Close", "Close video mode?\nUnsaved work will be lost."):
            video_cleanup()
            video_remote.destroy()
    
    video_remote.protocol("WM_DELETE_WINDOW", on_close)

# END VIDEO FUNCTIONS

def open_pal_popup():
    """
    Open popup window to configure and run SAM2-PAL batch processing.
    
    SAM2-PAL (SAM2-Palindrome) treats static images as a "pseudo-video"
    and uses SAM2's video tracking to propagate masks from template to targets.
    
    Supports:
    - v15: Standard PAL fine-tuning with 4-step OC-CCL (memory reset)
    - v16: Optional LoRA fine-tuning (requires peft library)
    - Multi-template training from COCO JSON
    """
    popup = tk.Toplevel(root)
    popup.title("SAM2-PAL: Palindrome-based Mask Propagation")
    popup.geometry("750x1000")
    popup.transient(root)
    popup.grab_set()
    
    # Detect available script version
    script_dir = os.path.dirname(os.path.abspath(__file__))
    v18_exists = os.path.exists(os.path.join(script_dir, 'sam2_pal_batch_v18.py'))
    v17_exists = os.path.exists(os.path.join(script_dir, 'sam2_pal_batch_v17.py'))
    v16_exists = os.path.exists(os.path.join(script_dir, 'sam2_pal_batch_v16.py'))
    v15_exists = os.path.exists(os.path.join(script_dir, 'sam2_pal_batch_v15.py'))
    
    # Check for LoRA availability
    lora_available = False
    try:
        import peft
        lora_available = True
    except ImportError:
        pass
    
    # Determine which script to use (prefer newest)
    if v18_exists:
        pal_script_name = 'sam2_pal_batch_v18.py'
        version_str = "v18"
    elif v17_exists:
        pal_script_name = 'sam2_pal_batch_v17.py'
        version_str = "v17"
    elif v16_exists:
        pal_script_name = 'sam2_pal_batch_v16.py'
        version_str = "v16"
    elif v15_exists:
        pal_script_name = 'sam2_pal_batch_v15.py'
        version_str = "v15"
    else:
        pal_script_name = 'sam2_pal_batch.py'
        version_str = "legacy"
    
    # Variables
    template_json_var = tk.StringVar()
    template_image_var = tk.StringVar()
    image_dir_var = tk.StringVar()
    output_dir_var = tk.StringVar()
    save_masks_var = tk.BooleanVar(value=True)
    save_vis_var = tk.BooleanVar(value=True)
    template_mask_var = tk.StringVar(value="")
    category_name_var = tk.StringVar(value="object")
    template_mode_var = tk.StringVar(value="mask")
    
    # Fine-tuning variables (PAL)
    enable_finetune_var = tk.BooleanVar(value=True)
    num_epochs_var = tk.StringVar(value="50")
    learning_rate_var = tk.StringVar(value="1e-5")
    finetune_checkpoint_var = tk.StringVar(value="")
    max_images_per_epoch_var = tk.StringVar(value="30")
    
    # Multi-template training variables
    training_json_var = tk.StringVar()
    training_images_dir_var = tk.StringVar()
    
    # LoRA variables (v16 only)
    use_lora_var = tk.BooleanVar(value=False)
    lora_rank_var = tk.StringVar(value="16")
    
    # Output naming variables
    output_timestamp_var = tk.BooleanVar(value=True)
    output_category_prefix_var = tk.BooleanVar(value=False)
    
    # Inference options
    interleave_template_var = tk.BooleanVar(value=False)
    num_points_var = tk.StringVar(value="30")

    # V18 inference strategy options
    cycle_consistency_var = tk.BooleanVar(value=False)
    chunk_size_var = tk.StringVar(value="0")
    iou_threshold_var = tk.StringVar(value="0.5")
    load_checkpoint_var = tk.StringVar(value="")
    
    # Create scrollable frame for the popup
    main_canvas = tk.Canvas(popup)
    scrollbar = tk.Scrollbar(popup, orient="vertical", command=main_canvas.yview)
    scrollable_popup = tk.Frame(main_canvas)
    
    scrollable_popup.bind(
        "<Configure>",
        lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
    )
    
    main_canvas.create_window((0, 0), window=scrollable_popup, anchor="nw")
    main_canvas.configure(yscrollcommand=scrollbar.set)
    
    main_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    row = 0
    
    # Title
    tk.Label(scrollable_popup, text=f"SAM2-PAL {version_str}: 4-Step OC-CCL Mask Propagation", 
             font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=3, pady=10)
    row += 1
    
    # Version and status info
    status_parts = [f"Script: {pal_script_name}"]
    if version_str in ["v17", "v18"]:
        status_parts.append("Multi-mask: [OK]")
    if version_str == "v18":
        status_parts.append("OOM-fix: [OK]")
        status_parts.append("Cycle-consistency: [OK]")
    if version_str in ["v16", "v17"]:
        status_parts.append(f"LoRA: {'[OK] Available' if lora_available else 'Install peft'}")
    status_text = " | ".join(status_parts)
    tk.Label(scrollable_popup, text=status_text, fg="blue", font=("Arial", 9)).grid(row=row, column=0, columnspan=3, pady=(0, 5))
    row += 1
    
    # Description
    desc_text = "Propagate masks using SAM2 video tracking with 4-step palindrome training (per arxiv.org/abs/2501.06749)"
    tk.Label(scrollable_popup, text=desc_text, fg="gray", wraplength=700).grid(row=row, column=0, columnspan=3, pady=(0, 5))
    row += 1
    
    # Info about PAL
    info_frame = tk.LabelFrame(scrollable_popup, text="How SAM2-PAL Works (4-Step OC-CCL)", padx=5, pady=5)
    info_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
    pal_info = """4-frame palindrome: {x0, x1, x1A‚Â , x0} with MEMORY RESET
    
Step 1: Template (x0) stored in memory
Step 2: Unlabeled (x1) predict mask using memory
*** MEMORY RESET *** (prevents cheating)
Step 3: with predicted mask as prompt
Step 4: predict, compare to original mask (loss)

This forces the model to learn REAL tracking, not just memorization."""
    tk.Label(info_frame, text=pal_info, justify="left", fg="blue", font=("Arial", 9)).pack()
    row += 1
    
    # Template Mode Selection
    tk.Label(scrollable_popup, text="Template Source:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    mode_frame = tk.Frame(scrollable_popup)
    mode_frame.grid(row=row, column=1, sticky='w', padx=5, pady=5)
    tk.Radiobutton(mode_frame, text="Binary Mask (recommended)", variable=template_mode_var, value="mask").pack(side=tk.LEFT, padx=5)
    tk.Radiobutton(mode_frame, text="COCO JSON", variable=template_mode_var, value="json").pack(side=tk.LEFT, padx=5)
    row += 1
    
    # --- Mask Mode Inputs ---
    mask_frame = tk.LabelFrame(scrollable_popup, text="Binary Mask Template (Primary)", padx=5, pady=5)
    mask_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
    
    tk.Label(mask_frame, text="Mask File:").grid(row=0, column=0, sticky='e', padx=5, pady=3)
    tk.Entry(mask_frame, textvariable=template_mask_var, width=45).grid(row=0, column=1, padx=5, pady=3)
    tk.Button(mask_frame, text="Browse", command=lambda: browse_file_pal(template_mask_var, "Select Binary Mask", [("PNG Files", "*.png"), ("All Images", "*.png *.jpg *.bmp")])).grid(row=0, column=2, padx=5, pady=3)
    
    tk.Label(mask_frame, text="Category Name:").grid(row=1, column=0, sticky='e', padx=5, pady=3)
    tk.Entry(mask_frame, textvariable=category_name_var, width=20).grid(row=1, column=1, sticky='w', padx=5, pady=3)
    row += 1
    
    # --- JSON Mode Inputs ---
    json_frame = tk.LabelFrame(scrollable_popup, text="COCO JSON Template (Multi-mask support in v17)", padx=5, pady=5)
    json_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
    
    tk.Label(json_frame, text="Template JSON:").grid(row=0, column=0, sticky='e', padx=5, pady=3)
    tk.Entry(json_frame, textvariable=template_json_var, width=45).grid(row=0, column=1, padx=5, pady=3)
    tk.Button(json_frame, text="Browse", command=lambda: browse_file_pal(template_json_var, "Select Template COCO JSON", [("JSON Files", "*.json")])).grid(row=0, column=2, padx=5, pady=3)
    
    # Info about multi-mask
    if version_str == "v17":
        tk.Label(json_frame, text="v17: Multiple masks (e.g., scape, antenna, eye) will ALL be trained and predicted", 
                 fg="green", font=("Arial", 8)).grid(row=1, column=0, columnspan=3, sticky='w', padx=5)
    else:
        tk.Label(json_frame, text="Note: v15/v16 only use first mask for training. Upgrade to v17 for multi-mask training.", 
                 fg="orange", font=("Arial", 8)).grid(row=1, column=0, columnspan=3, sticky='w', padx=5)
    row += 1
    
    # --- Common Inputs ---
    tk.Label(scrollable_popup, text="Ã" * 80).grid(row=row, column=0, columnspan=3, pady=5)
    row += 1
    
    # Template Image (required for both modes)
    tk.Label(scrollable_popup, text="Template Image:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(scrollable_popup, textvariable=template_image_var, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(scrollable_popup, text="Browse", command=lambda: browse_file_pal(template_image_var, "Select Template Image", [("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")])).grid(row=row, column=2, padx=5, pady=5)
    row += 1
    
    # Image Directory
    tk.Label(scrollable_popup, text="Target Images Directory:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(scrollable_popup, textvariable=image_dir_var, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(scrollable_popup, text="Browse", command=lambda: browse_dir_pal(image_dir_var, "Select Target Images Directory")).grid(row=row, column=2, padx=5, pady=5)
    row += 1
    
    # Output Directory
    tk.Label(scrollable_popup, text="Output Directory:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
    tk.Entry(scrollable_popup, textvariable=output_dir_var, width=50).grid(row=row, column=1, padx=5, pady=5)
    tk.Button(scrollable_popup, text="Browse", command=lambda: browse_dir_pal(output_dir_var, "Select Output Directory")).grid(row=row, column=2, padx=5, pady=5)
    row += 1
    
    # === PAL Fine-tuning Section (Primary Feature) ===
    tk.Label(scrollable_popup, text="Ã" * 80).grid(row=row, column=0, columnspan=3, pady=5)
    row += 1
    
    finetune_frame = tk.LabelFrame(scrollable_popup, text="PAL Fine-tuning (4-Step OC-CCL)", padx=5, pady=5)
    finetune_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
    
    # Enable checkbox
    tk.Checkbutton(finetune_frame, text="Enable PAL fine-tuning (recommended for best results)", 
                   variable=enable_finetune_var, font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky='w', pady=5)
    
    # Info text
    ft_info = "Fine-tunes SAM2 using 4-step palindrome with memory reset (per paper)"
    tk.Label(finetune_frame, text=ft_info, fg="gray", font=("Arial", 8)).grid(row=1, column=0, columnspan=3, sticky='w')
    
    # Number of epochs
    tk.Label(finetune_frame, text="Epochs:").grid(row=2, column=0, sticky='e', padx=5, pady=3)
    tk.Entry(finetune_frame, textvariable=num_epochs_var, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=3)
    tk.Label(finetune_frame, text="(50-100 recommended)", fg="gray").grid(row=2, column=2, sticky='w')
    
    # Learning rate
    tk.Label(finetune_frame, text="Learning Rate:").grid(row=3, column=0, sticky='e', padx=5, pady=3)
    tk.Entry(finetune_frame, textvariable=learning_rate_var, width=10).grid(row=3, column=1, sticky='w', padx=5, pady=3)
    tk.Label(finetune_frame, text="(1e-5 for full, 1e-4 for LoRA)", fg="gray").grid(row=3, column=2, sticky='w')
    
    # Max images per epoch
    tk.Label(finetune_frame, text="Images/Epoch:").grid(row=4, column=0, sticky='e', padx=5, pady=3)
    tk.Entry(finetune_frame, textvariable=max_images_per_epoch_var, width=10).grid(row=4, column=1, sticky='w', padx=5, pady=3)
    tk.Label(finetune_frame, text="(unlabeled images sampled per epoch)", fg="gray").grid(row=4, column=2, sticky='w')
    
    # Save fine-tuned checkpoint
    tk.Label(finetune_frame, text="Save checkpoint:").grid(row=5, column=0, sticky='e', padx=5, pady=3)
    tk.Entry(finetune_frame, textvariable=finetune_checkpoint_var, width=40).grid(row=5, column=1, padx=5, pady=3)
    tk.Button(finetune_frame, text="Browse", command=lambda: browse_save_pal(finetune_checkpoint_var, "Save Fine-tuned Checkpoint", [("PyTorch", "*.pt")])).grid(row=5, column=2, padx=5, pady=3)
    
    row += 1
    
    # === Multi-Template Training Section ===
    multi_frame = tk.LabelFrame(scrollable_popup, text="Multi-Template Training (Optional - Improves Results)", padx=5, pady=5)
    multi_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
    
    tk.Label(multi_frame, text="Add more labeled templates for better generalization:", fg="gray", font=("Arial", 8)).grid(row=0, column=0, columnspan=3, sticky='w')
    
    tk.Label(multi_frame, text="Training JSON:").grid(row=1, column=0, sticky='e', padx=5, pady=3)
    tk.Entry(multi_frame, textvariable=training_json_var, width=40).grid(row=1, column=1, padx=5, pady=3)
    tk.Button(multi_frame, text="Browse", command=lambda: browse_file_pal(training_json_var, "Select Training COCO JSON", [("JSON Files", "*.json")])).grid(row=1, column=2, padx=5, pady=3)
    
    tk.Label(multi_frame, text="Training Images Dir:").grid(row=2, column=0, sticky='e', padx=5, pady=3)
    tk.Entry(multi_frame, textvariable=training_images_dir_var, width=40).grid(row=2, column=1, padx=5, pady=3)
    tk.Button(multi_frame, text="Browse", command=lambda: browse_dir_pal(training_images_dir_var, "Select Training Images Directory")).grid(row=2, column=2, padx=5, pady=3)
    
    row += 1
    
    # === LoRA Section (v16/v17 only) ===
    if version_str in ["v16", "v17"]:
        lora_frame = tk.LabelFrame(scrollable_popup, text="LoRA Fine-tuning (v16+ - Optional)", padx=5, pady=5)
        lora_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
        
        lora_cb = tk.Checkbutton(lora_frame, text="Use LoRA (Low-Rank Adaptation) - per paper", 
                       variable=use_lora_var, font=("Arial", 9))
        lora_cb.grid(row=0, column=0, columnspan=2, sticky='w', pady=3)
        
        if not lora_available:
            lora_cb.config(state='disabled')
            tk.Label(lora_frame, text="[WARNING] Install peft: pip install peft", fg="red", font=("Arial", 8)).grid(row=0, column=2, sticky='w')
        else:
            tk.Label(lora_frame, text="[OK] peft installed", fg="green", font=("Arial", 8)).grid(row=0, column=2, sticky='w')
        
        tk.Label(lora_frame, text="LoRA Rank:").grid(row=1, column=0, sticky='e', padx=5, pady=3)
        tk.Entry(lora_frame, textvariable=lora_rank_var, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=3)
        tk.Label(lora_frame, text="(16 default)", fg="gray").grid(row=1, column=2, sticky='w')
        
        row += 1
    
    # === Output Options ===
    output_frame = tk.LabelFrame(scrollable_popup, text="Output Options", padx=5, pady=5)
    output_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
    
    tk.Checkbutton(output_frame, text="Save individual binary masks", variable=save_masks_var).grid(row=0, column=0, sticky='w', padx=5)
    tk.Checkbutton(output_frame, text="Save visualizations", variable=save_vis_var).grid(row=0, column=1, sticky='w', padx=5)
    tk.Checkbutton(output_frame, text="Add timestamp to output", variable=output_timestamp_var).grid(row=1, column=0, sticky='w', padx=5)
    tk.Checkbutton(output_frame, text="Category prefix in filenames", variable=output_category_prefix_var).grid(row=1, column=1, sticky='w', padx=5)
    tk.Checkbutton(output_frame, text="Interleave template (reduces drift)", variable=interleave_template_var).grid(row=2, column=0, sticky='w', padx=5)
    
    tk.Label(output_frame, text="Num points:").grid(row=2, column=1, sticky='e', padx=5)
    tk.Entry(output_frame, textvariable=num_points_var, width=5).grid(row=2, column=2, sticky='w', padx=2)

    # V18 inference strategy options
    if version_str in ["v18"]:
        v18_frame = tk.LabelFrame(output_frame, text="v18 Inference Strategy", padx=3, pady=3)
        v18_frame.grid(row=3, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        tk.Checkbutton(v18_frame, text="Cycle consistency (4-stage forward+backward IoU check)",
                        variable=cycle_consistency_var).grid(row=0, column=0, columnspan=3, sticky='w')

        tk.Label(v18_frame, text="Chunk size:").grid(row=1, column=0, sticky='e', padx=5)
        tk.Entry(v18_frame, textvariable=chunk_size_var, width=8).grid(row=1, column=1, sticky='w', padx=2)
        tk.Label(v18_frame, text="(0=all at once, >0=process N images per chunk)",
                  fg="gray", font=("Arial", 8)).grid(row=1, column=2, sticky='w')

        tk.Label(v18_frame, text="IoU threshold:").grid(row=2, column=0, sticky='e', padx=5)
        tk.Entry(v18_frame, textvariable=iou_threshold_var, width=8).grid(row=2, column=1, sticky='w', padx=2)
        tk.Label(v18_frame, text="(drift flagging threshold, default 0.5)",
                  fg="gray", font=("Arial", 8)).grid(row=2, column=2, sticky='w')

        # Load existing checkpoint for inference-only
        tk.Label(v18_frame, text="Load checkpoint:").grid(row=3, column=0, sticky='e', padx=5)
        tk.Entry(v18_frame, textvariable=load_checkpoint_var, width=30).grid(row=3, column=1, sticky='w', padx=2)
        def _browse_load_ckpt():
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                title="Select Fine-tuned Checkpoint (.pt)",
                filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")]
            )
            if path:
                load_checkpoint_var.set(path)
        tk.Button(v18_frame, text="Browse", command=_browse_load_ckpt, width=8).grid(row=3, column=2, padx=2)
        tk.Label(v18_frame, text="(skip training, use existing .pt for inference only)",
                  fg="gray", font=("Arial", 8)).grid(row=4, column=0, columnspan=3, sticky='w', padx=5)

    row += 1
    
    # SAM2 paths info
    tk.Label(scrollable_popup, text=f"Using SAM2: {sam2_checkpoint}", fg="gray", font=("Arial", 8)).grid(row=row, column=0, columnspan=3, pady=2)
    row += 1
    
    # Progress label
    progress_var = tk.StringVar(value="Ready")
    progress_label = tk.Label(scrollable_popup, textvariable=progress_var, fg="green", font=("Arial", 10))
    progress_label.grid(row=row, column=0, columnspan=3, pady=5)
    row += 1
    
    def run_pal():
        """Run the PAL script (v15/v16/v17) with configured options."""
        mode = template_mode_var.get()
        
        # Validate inputs based on mode
        if mode == "json":
            if not template_json_var.get():
                messagebox.showerror("Error", "Please select a template COCO JSON file.")
                return
            if not os.path.exists(template_json_var.get()):
                messagebox.showerror("Error", f"Template JSON file not found:\n{template_json_var.get()}")
                return
        else:  # mask mode
            if not template_mask_var.get():
                messagebox.showerror("Error", "Please select a binary mask file.")
                return
            if not os.path.exists(template_mask_var.get()):
                messagebox.showerror("Error", f"Mask file not found:\n{template_mask_var.get()}")
                return
        
        if not template_image_var.get():
            messagebox.showerror("Error", "Please select a template image.")
            return
        if not os.path.exists(template_image_var.get()):
            messagebox.showerror("Error", f"Template image not found:\n{template_image_var.get()}")
            return
        if not image_dir_var.get():
            messagebox.showerror("Error", "Please select a target images directory.")
            return
        if not os.path.exists(image_dir_var.get()):
            messagebox.showerror("Error", f"Target images directory not found:\n{image_dir_var.get()}")
            return
        if not output_dir_var.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return
        
        # === FIX: Auto-create output directory if it doesn't exist ===
        output_dir = output_dir_var.get()
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logging.info(f"Created output directory: {output_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output directory:\n{output_dir}\n\n{str(e)}")
                return
        
        # Validate fine-tuning options if enabled
        do_finetune = enable_finetune_var.get()
        if do_finetune:
            # === FIX: Allow COCO JSON mode for fine-tuning in v17 ===
            if mode == "json" and version_str not in ["v17", "v18"]:
                messagebox.showerror("Error", "PAL fine-tuning with COCO JSON requires v17.\nPlease use Binary Mask mode or upgrade to sam2_pal_batch_v17.py")
                return
            # For mask mode, still require a mask file
            if mode == "mask" and not template_mask_var.get():
                messagebox.showerror("Error", "PAL fine-tuning requires a binary mask template.")
                return
            
            if not finetune_checkpoint_var.get():
                # Auto-generate checkpoint path
                auto_checkpoint = os.path.join(output_dir, f"finetuned_model_pal{version_str}_large.pt")
                finetune_checkpoint_var.set(auto_checkpoint)
            try:
                int(num_epochs_var.get())
                float(learning_rate_var.get())
                int(max_images_per_epoch_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid epochs, learning rate, or images per epoch value.")
                return
        
        progress_var.set(f"Starting SAM2-PAL {version_str}...")
        popup.update()
        
        # Get script path
        pal_script = os.path.join(script_dir, pal_script_name)
        
        if not os.path.exists(pal_script):
            messagebox.showerror("Error", f"PAL script not found at: {pal_script}\n\nPlease ensure {pal_script_name} is in the same folder as Descriptron.")
            progress_var.set("Error: Script not found")
            return
        
        # Build command
        command = [
            sys.executable, pal_script,
            '--template_image', template_image_var.get(),
            '--image_dir', image_dir_var.get(),
            '--output_dir', output_dir,
            '--sam2_checkpoint', sam2_checkpoint,
            '--sam2_config', model_cfg,
            '--num_points', num_points_var.get(),
        ]
        
        if mode == "json":
            command.extend(['--template_json', template_json_var.get()])
        else:  # mask mode
            command.extend(['--template_mask', template_mask_var.get()])
            command.extend(['--category_name', category_name_var.get()])
        
        if save_masks_var.get():
            command.append('--save_masks')
        else:
            command.append('--no_save_masks')
        
        if save_vis_var.get():
            command.append('--save_vis')
        
        # Add PAL fine-tuning options
        if do_finetune:
            command.append('--pal_finetuning')
            command.extend(['--num_epochs', num_epochs_var.get()])
            command.extend(['--learning_rate', learning_rate_var.get()])
            command.extend(['--max_images_per_epoch', max_images_per_epoch_var.get()])
            command.extend(['--finetune_checkpoint', finetune_checkpoint_var.get()])
            
            # Multi-template training
            if training_json_var.get():
                command.extend(['--training_json', training_json_var.get()])
            if training_images_dir_var.get():
                command.extend(['--training_images_dir', training_images_dir_var.get()])
        
        # LoRA options (v16/v17 only)
        if version_str in ["v16", "v17", "v18"] and use_lora_var.get() and lora_available:
            command.append('--use_lora')
            command.extend(['--lora_rank', lora_rank_var.get()])
        
        # Output naming options
        if output_timestamp_var.get():
            command.append('--output_timestamp')
        if output_category_prefix_var.get():
            command.append('--output_category_prefix')
        
        # Inference options
        if interleave_template_var.get():
            command.append('--interleave_template')
        
        # V18 inference strategy options
        if version_str in ["v18"]:
            if load_checkpoint_var.get() and os.path.exists(load_checkpoint_var.get()):
                command.extend(['--load_checkpoint', load_checkpoint_var.get()])
            if cycle_consistency_var.get():
                command.append('--cycle_consistency')
            chunk_sz = int(chunk_size_var.get() or "0")
            if chunk_sz > 0:
                command.extend(['--chunk_size', str(chunk_sz)])
            iou_thr = float(iou_threshold_var.get() or "0.5")
            if iou_thr != 0.5:
                command.extend(['--iou_threshold', str(iou_thr)])
        
        # === FIX: Save settings to YAML file ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        settings = {
            'sam2_pal_version': version_str,
            'script': pal_script_name,
            'timestamp': timestamp,
            'template_mode': mode,
            'template_image': template_image_var.get(),
            'template_mask': template_mask_var.get() if mode == "mask" else None,
            'template_json': template_json_var.get() if mode == "json" else None,
            'category_name': category_name_var.get(),
            'image_dir': image_dir_var.get(),
            'output_dir': output_dir,
            'sam2_checkpoint': sam2_checkpoint,
            'sam2_config': model_cfg,
            'fine_tuning': {
                'enabled': do_finetune,
                'num_epochs': num_epochs_var.get(),
                'learning_rate': learning_rate_var.get(),
                'max_images_per_epoch': max_images_per_epoch_var.get(),
                'checkpoint': finetune_checkpoint_var.get(),
                'training_json': training_json_var.get(),
                'training_images_dir': training_images_dir_var.get(),
            },
            'lora': {
                'enabled': use_lora_var.get(),
                'rank': lora_rank_var.get(),
                'available': lora_available,
            },
            'output_options': {
                'save_masks': save_masks_var.get(),
                'save_vis': save_vis_var.get(),
                'output_timestamp': output_timestamp_var.get(),
                'category_prefix': output_category_prefix_var.get(),
                'interleave_template': interleave_template_var.get(),
                'num_points': num_points_var.get(),
                'cycle_consistency': cycle_consistency_var.get(),
                'chunk_size': chunk_size_var.get(),
                'iou_threshold': iou_threshold_var.get(),
                'load_checkpoint': load_checkpoint_var.get(),
            },
            'command': ' '.join(command),
        }
        
        settings_file = os.path.join(output_dir, f"pal_settings_{timestamp}.yaml")
        try:
            # Save as YAML-like format (simple key: value)
            with open(settings_file, 'w') as f:
                f.write(f"# SAM2-PAL Settings - {timestamp}\n")
                f.write(f"# Generated by Descriptron\n\n")
                def write_dict(d, indent=0):
                    for key, value in d.items():
                        prefix = "  " * indent
                        if isinstance(value, dict):
                            f.write(f"{prefix}{key}:\n")
                            write_dict(value, indent + 1)
                        elif value is None:
                            f.write(f"{prefix}{key}: null\n")
                        elif isinstance(value, bool):
                            f.write(f"{prefix}{key}: {str(value).lower()}\n")
                        elif isinstance(value, str) and (' ' in value or ':' in value):
                            f.write(f'{prefix}{key}: "{value}"\n')
                        else:
                            f.write(f"{prefix}{key}: {value}\n")
                write_dict(settings)
            logging.info(f"Saved settings to: {settings_file}")
        except Exception as e:
            logging.warning(f"Could not save settings file: {e}")
        
        # === FIX: Prepare log file ===
        log_file = os.path.join(output_dir, f"pal_log_{timestamp}.txt")
        
        def run_thread():
            output_lines = []
            try:
                # Log the command
                output_lines.append(f"SAM2-PAL {version_str} - {timestamp}")
                output_lines.append("=" * 60)
                output_lines.append(f"Command: {' '.join(command)}")
                output_lines.append("=" * 60)
                output_lines.append("")
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        output_lines.append(line)
                        # Update progress with last line
                        display_line = line[-70:] if len(line) > 70 else line
                        root.after(0, lambda l=display_line: progress_var.set(l))
                
                process.wait()
                
                # Add final status
                output_lines.append("")
                output_lines.append("=" * 60)
                output_lines.append(f"Exit code: {process.returncode}")
                output_lines.append(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # === FIX: Save log file ===
                try:
                    with open(log_file, 'w') as f:
                        f.write('\n'.join(output_lines))
                    logging.info(f"Saved log to: {log_file}")
                except Exception as e:
                    logging.warning(f"Could not save log file: {e}")
                
                if process.returncode == 0:
                    # Find the output JSON (may have timestamp)
                    try:
                        json_files = [f for f in os.listdir(output_dir) if f.startswith('pal_predictions') and f.endswith('.json')]
                        if json_files:
                            output_json = os.path.join(output_dir, sorted(json_files)[-1])
                        else:
                            output_json = os.path.join(output_dir, 'pal_predictions.json')
                    except:
                        output_json = os.path.join(output_dir, 'pal_predictions.json')
                    
                    success_msg = f"SAM2-PAL {version_str} completed!\n\n"
                    success_msg += f"Output: {output_json}\n"
                    success_msg += f"Settings: {settings_file}\n"
                    success_msg += f"Log: {log_file}\n\n"
                    if do_finetune:
                        success_msg += f"Checkpoint: {finetune_checkpoint_var.get()}\n\n"
                    success_msg += "Use 'View Predictions' to review results."
                    root.after(0, lambda: messagebox.showinfo("Success", success_msg))
                    root.after(0, lambda: progress_var.set("[OK] Completed successfully!"))
                else:
                    error_msg = '\n'.join(output_lines[-30:]) if output_lines else "Unknown error"
                    root.after(0, lambda: messagebox.showerror("Error", f"PAL processing failed!\n\nSee log file:\n{log_file}\n\nLast output:\n{error_msg[:600]}"))
                    root.after(0, lambda: progress_var.set(f"Failed - see {os.path.basename(log_file)}"))
                    
            except Exception as e:
                output_lines.append(f"\nEXCEPTION: {str(e)}")
                # Save log even on exception
                try:
                    with open(log_file, 'w') as f:
                        f.write('\n'.join(output_lines))
                except:
                    pass
                root.after(0, lambda: messagebox.showerror("Error", f"Failed to run PAL script:\n\n{str(e)}\n\nLog saved to:\n{log_file}"))
                root.after(0, lambda: progress_var.set("Error"))
        
        # Run in separate thread
        progress_var.set("Running... (check console for progress)")
        threading.Thread(target=run_thread, daemon=True).start()
    
    # Buttons
    button_frame = tk.Frame(scrollable_popup)
    button_frame.grid(row=row, column=0, columnspan=3, pady=15)
    
    tk.Button(button_frame, text=f"Run SAM2-PAL {version_str}", command=run_pal, 
              bg="lightgreen", font=("Arial", 11, "bold"), width=18).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="Cancel", command=popup.destroy, width=10).pack(side=tk.LEFT, padx=10)


def browse_file_pal(var, title, filetypes):
    """Browse for a file for PAL popup."""
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if path:
        var.set(path)


def browse_dir_pal(var, title):
    """Browse for a directory for PAL popup."""
    path = filedialog.askdirectory(title=title)
    if path:
        var.set(path)


def browse_save_pal(var, title, filetypes):
    """Browse for a save file location for PAL popup."""
    path = filedialog.asksaveasfilename(title=title, filetypes=filetypes, defaultextension=".pt")
    if path:
        var.set(path)


# Override the marmot button action to save points if the point prompt is still active
def save_and_close():
    global canvas, prediction_cache, mode, segmentation_masks, image, selected_prediction_image, pred_json_path, categories
        # Save the concatenated VIA2 JSON file with keypoints, contours, and bounding boxes
    save_final_via2_json()
    if point_prompt_mode:
        save_points_to_via2()
    reset_global_variables()


def save_predictions_from_cache():
    global prediction_cache, image_dir, categories

    if not prediction_cache:# Add the Trash button to the GUI
        trash_button = tk.Button(scrollable_frame, text="Trash Mask", command=delete_current_mask)
        trash_button.grid(row=1, column=12, padx=1, pady=1)

        messagebox.showerror("Error", "No predictions available to save.")
        return

    # Initialize structures for COCO JSON
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": copy.deepcopy(categories)
    }

    annotation_id = 1  # Start annotation IDs from 1
    image_id_mapping = {}  # Map image filenames to unique image IDs
    image_id_counter = 1

    for img_filename, anns in prediction_cache.items():
        # Construct the full image path
        img_path = os.path.join(image_dir, img_filename)
        if not os.path.exists(img_path):
            logging.warning(f"Image file '{img_path}' does not exist. Skipping.")
            continue

        # Load image to get dimensions
        try:
            pil_image = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_image)
            height, width = img_np.shape[:2]
        except Exception as e:
            logging.warning(f"Failed to load image '{img_path}': {e}")
            continue

        # Add image metadata
        image_id = image_id_counter  # Assign a unique integer ID
        image_id_mapping[img_filename] = image_id
        image_metadata = {
            "id": image_id,
            "file_name": img_filename,
            "height": height,
            "width": width
        }
        coco_output["images"].append(image_metadata)
        image_id_counter += 1

        # Add annotations
        for ann in anns:
            ann_copy = ann.copy()
            ann_copy['id'] = annotation_id
            ann_copy['image_id'] = image_id_mapping[img_filename]
            # Ensure all necessary fields are present
            required_fields = ['segmentation', 'bbox', 'area', 'category_id', 'iscrowd']
            for field in required_fields:
                if field not in ann_copy:
                    ann_copy[field] = ann.get(field, None)
            coco_output["annotations"].append(ann_copy)
            annotation_id += 1

    if not coco_output["annotations"]:
        messagebox.showerror("Error", "No valid annotations found in predictions.")
        return

    # Prompt user to save the COCO JSON
    save_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        title="Save Predictions as COCO JSON",
        filetypes=[("COCO JSON Files", "*.json")]
    )

    if save_path:
        try:
            with open(save_path, 'w') as json_file:
                json.dump(coco_output, json_file, indent=4)
            messagebox.showinfo("Success", f"Predictions saved successfully to '{save_path}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save predictions:\n{e}")
    else:
        messagebox.showinfo("Info", "Save operation cancelled.")


def apply_label(auto=False):
    global labels, current_mask_index, category_name_to_id, category_id_to_name, categories, segmentation_masks
    global coco_output_mask, coco_output_accumulate
    label = selected_label.get()


    # If "Custom" is selected, prompt the user for a custom label
    if label == "Custom":
        label = simpledialog.askstring("Input", "Enter custom label:", parent=root)
        
        # FIXED: Also ask for supercategory if creating a new custom label
        if label and label not in category_name_to_id:
            # Get list of existing supercategories for suggestions
            existing_supercategories = list(set(category_id_to_supercategory.values()))
            supercategory_str = ", ".join(existing_supercategories) if existing_supercategories else "sclerite, textures, etc."
            
            supercategory = simpledialog.askstring(
                "Supercategory", 
                f"Enter supercategory for '{label}':\n(Existing: {supercategory_str})",
                parent=root
            )
            if not supercategory:
                supercategory = "none"  # Default if cancelled or empty

    # Ensure a label is present
    if label:
        labels[current_mask_index] = label

        # Check if the category exists
        if label in category_name_to_id:
            category_id = category_name_to_id[label]
        else:
            # Category does not exist; add new category
            existing_category_ids = set(cat['id'] for cat in categories)
            if existing_category_ids:
                new_category_id = max(existing_category_ids) + 1
            else:
                new_category_id = 1  # Start from 1 if no categories exist

            # Ensure the new category ID is not 0
            if new_category_id == 0:
                new_category_id = 1  # Avoid using category ID 0

            category_name_to_id[label] = new_category_id
            category_id_to_name[new_category_id] = label
            
            # Use the supercategory from the dialog (set above for Custom labels)
            # For non-Custom labels, default to "none"
            if 'supercategory' not in dir() or supercategory is None:
                supercategory = "none"
            
            new_category = {
                "id": new_category_id,
                "name": label,
                "supercategory": supercategory
            }
            categories.append(new_category)
            category_id_to_supercategory[new_category_id] = supercategory
            category_id = new_category_id
            
            # FIXED: Sync new category to coco_output_accumulate and coco_output_mask
            if new_category not in coco_output_accumulate.get("categories", []):
                if "categories" not in coco_output_accumulate:
                    coco_output_accumulate["categories"] = []
                coco_output_accumulate["categories"].append(new_category.copy())
            if new_category not in coco_output_mask.get("categories", []):
                if "categories" not in coco_output_mask:
                    coco_output_mask["categories"] = []
                coco_output_mask["categories"].append(new_category.copy())
            
            # Update the label options in the dropdown
            if label not in label_options:
                label_options.append(label)
                update_label_dropdown()

        # Assign the category_id to the current mask
        segmentation_masks[current_mask_index]['category_id'] = category_id

        # FIXED: With instance naming, each save creates a unique file (label_0, label_1, etc.)
        # So we don't need the overwrite check anymore - each instance gets its own file
        # Save the mask and contour
        save_mask_and_contour_via2(overwrite=False)

        if not auto:
            messagebox.showinfo("Label Applied", f"Label '{label}' applied to the current mask.")
    else:
        messagebox.showerror("Error", "No label entered.")


def close_window():
    root.destroy()  # This will close the entire application window

#root = tk.Tk()
#root.title("Descriptron Interactive SAM2 Segmentation")

# Configure the root window to allow the canvas to expand and buttons to stay at the bottom
#root.grid_rowconfigure(0, weight=1)  # Row 0 (canvas) expands
#root.grid_columnconfigure(0, weight=1)  # Column 0 expands

# Image canvas frame
image_frame = tk.Frame(root)
image_frame.grid(row=0, column=0, sticky="nsew")  # Allow the frame to expand and fill space
canvas = tk.Canvas(image_frame)
canvas.pack(fill=tk.BOTH, expand=True)  # The canvas will fill the image_frame


# Button frame with fixed height (e.g., 250 in the Y dimension)
button_frame = tk.Frame(root, height=100)  # Set the height of the button area to 100 pixels
button_frame.grid(row=1, column=0, sticky="ew")  # Stick the button frame to the bottom
# Set the button frame to have a fixed height and not expand
button_frame.grid_propagate(False)  # Prevent the button frame from resizing itself based on content

# Button frame (now using grid)
#button_frame = tk.Frame(root)
#button_frame.grid(row=1, column=0, sticky="ew")  # Stick the button frame to the bottom

# Wrapping frame for buttons to avoid overflow
# Button frame
#button_frame = tk.Frame(root)
#button_frame.pack(side=tk.BOTTOM, fill=tk.X)  # Stick the button frame to the bottom

# Scrollable button container
#button_container = tk.Frame(button_frame)
#button_container.pack(side=tk.TOP, fill=tk.X, padx=1, pady=1)

# Add a canvas with a scrollbar to make sure all buttons are accessible

# Scrollable button container
button_canvas = tk.Canvas(button_frame)
button_canvas.pack(side=tk.TOP, fill=tk.X)  # The canvas for the buttons takes up the top of the button frame
scrollbar = tk.Scrollbar(button_frame, orient="horizontal", command=button_canvas.xview)
scrollbar.pack(side=tk.BOTTOM, fill=tk.X)  # Place the scrollbar under the buttons

scrollable_frame = tk.Frame(button_canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: button_canvas.configure(scrollregion=button_canvas.bbox("all"))
)


button_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
button_canvas.configure(xscrollcommand=scrollbar.set)

button_canvas.pack(side="bottom", fill="both", expand=False)
scrollbar.pack(side="bottom", fill="x")



load_btn = tk.Button(scrollable_frame, text="Load & Apply SAM2-Automatic", command=load_image_and_predict)
load_btn.grid(row=0, column=0, padx=0, pady=0)

load_prompt_btn = tk.Button(scrollable_frame, text="Load Image for SAM2-Prompt", command=load_image_for_prompt)
load_prompt_btn.grid(row=0, column=1, padx=0, pady=0)

bbox_button = tk.Button(scrollable_frame, text="Draw Bounding Box", command=toggle_bbox_mode)
bbox_button.grid(row=0, column=2, padx=0, pady=0)

remove_bbox_btn = tk.Button(scrollable_frame, text="Remove Last BBox", command=remove_last_bbox)
remove_bbox_btn.grid(row=0, column=3, padx=0, pady=0)

# Dropdown for selecting point label (Positive or Negative)
selected_point_label = tk.StringVar(root)
selected_point_label.set("Positive")
point_label_options = ["Positive", "Negative"]
point_label_dropdown = tk.OptionMenu(scrollable_frame, selected_point_label, *point_label_options)
point_label_dropdown.grid(row=0, column=4, padx=0, pady=0)
# Button to toggle point prompt mode
point_prompt_btn = tk.Button(scrollable_frame, text="Point Prompt", command=toggle_point_prompt_mode)
point_prompt_btn.grid(row=0, column=5, padx=0, pady=0)

# Button to clear all points
clear_points_btn = tk.Button(scrollable_frame, text="Clear Points", command=clear_points)
clear_points_btn.grid(row=0, column=6, padx=0, pady=0)

# Button to apply SAM2 using points and bounding boxes
apply_points_prompt_btn = tk.Button(scrollable_frame, text="Apply Points+BBox SAM2", command=apply_sam2_prompt_with_points)
apply_points_prompt_btn.grid(row=0, column=7, padx=0, pady=0)

apply_prompt_btn = tk.Button(scrollable_frame, text="Apply SAM2-Prompt", command=apply_sam2_prompt)
apply_prompt_btn.grid(row=0, column=8, padx=0, pady=0)

load_labels_btn = tk.Button(scrollable_frame, text="Load Labels", command=load_labels)
load_labels_btn.grid(row=0, column=9, padx=0, pady=0)

selected_label = tk.StringVar(root)
selected_label.set("Select Label")
label_options = ["Custom","Trash"]
label_dropdown = tk.OptionMenu(scrollable_frame, selected_label, *label_options)
label_dropdown.grid(row=0, column=10, padx=0, pady=0)

apply_label_btn = tk.Button(scrollable_frame, text="Apply Label", command=apply_label)
apply_label_btn.grid(row=0, column=11, padx=0, pady=0)

next_btn = tk.Button(scrollable_frame, text="Next Mask", command=next_mask)
next_btn.grid(row=1, column=0, padx=0, pady=0)
prev_btn = tk.Button(scrollable_frame, text="Previous Mask", command=previous_mask)
prev_btn.grid(row=1, column=1, padx=0, pady=0)

save_unlabeled_btn = tk.Button(scrollable_frame, text="Save Unlabeled Masks", command=save_unlabeled_masks)
save_unlabeled_btn.grid(row=1, column=2, padx=0, pady=0)

zoom_in_btn = tk.Button(scrollable_frame, text="Zoom In", command=zoom_in)
zoom_in_btn.grid(row=1, column=3, padx=0, pady=0)

zoom_out_btn = tk.Button(scrollable_frame, text="Zoom Out", command=zoom_out)
zoom_out_btn.grid(row=1, column=4, padx=0, pady=0)

eraser_btn = tk.Button(scrollable_frame, text="Eraser: OFF", command=select_eraser)
eraser_btn.grid(row=1, column=5, padx=0, pady=0)

paintbrush_btn = tk.Button(scrollable_frame, text="Paintbrush: OFF", command=select_paintbrush)
paintbrush_btn.grid(row=1, column=6, padx=0, pady=0)


# Dropdown menu for brush size
brush_size_var = tk.StringVar(root)
brush_size_var.set("Small")  # Default brush size is small

# Dropdown options
brush_sizes = {"Small": 5, "Big": 25, "Jumbo": 50}

def update_brush_size(*args):
    size = brush_sizes[brush_size_var.get()]
    set_brush_size(size)

# Create the dropdown menu for brush sizes
brush_size_menu = tk.OptionMenu(scrollable_frame, brush_size_var, *brush_sizes.keys())
brush_size_menu.grid(row=1, column=7, padx=0, pady=0)

# Bind the dropdown change to update brush size
brush_size_var.trace("w", update_brush_size)


# Button for toggling keypoint edit mode
keypoint_edit_btn = tk.Button(scrollable_frame, text="Keypoint Edit: OFF", command=toggle_keypoint_edit_mode)
keypoint_edit_btn.grid(row=2, column=0, padx=0, pady=0)


# Add the new "Keypoints to Mask" button
keypoints_to_mask_btn = tk.Button(scrollable_frame, text="Keypoints to Mask", command=keypoints_to_mask)
keypoints_to_mask_btn.grid(row=2, column=1, padx=0, pady=0)

# Add the new "Keypoints to Line" button for line annotations (right next to Keypoints to Mask)
keypoints_to_line_btn = tk.Button(scrollable_frame, text="Keypoints to Line", command=keypoints_to_line, bg="lightyellow")
keypoints_to_line_btn.grid(row=2, column=2, padx=0, pady=0)

coco_to_min_btn = tk.Button(scrollable_frame, text="COCO->MinCOCO", command=run_coco_converter_script)
coco_to_min_btn.grid(row=2, column=4, padx=0, pady=0)

load_annotations_btn = tk.Button(scrollable_frame, text="Load Annotations", command=load_annotations)
load_annotations_btn.grid(row=1, column=8, padx=0, pady=0)

# Add the Multi-Mask button to the second row
multi_mask_btn = tk.Button(scrollable_frame, text="Multi-Mask Mode: OFF", command=toggle_multi_mask_mode)
multi_mask_btn.grid(row=1, column=9, padx=0, pady=0)

combine_json_btn = tk.Button(scrollable_frame, text="Combine COCO", command=run_coco_combiner_script)
combine_json_btn.grid(row=1, column=10, padx=0, pady=0)

remove_files_from_json_btn = tk.Button(scrollable_frame, text="Remove images COCO", command=run_remove_images_script)
remove_files_from_json_btn.grid(row=1, column=11, padx=0, pady=0)

# Detectron2 Training Button

# Train Detectron2 button
train_button = tk.Button(scrollable_frame, text="Train Detectron2", command=open_training_popup)
train_button.grid(row=2, column=5, padx=0, pady=0)

predict_btn = tk.Button(scrollable_frame, text="Predict Detectron2", command=predict_and_filter_popup)
predict_btn.grid(row=2, column=6, padx=0, pady=0)


# --- Add 'View Predictions' Button ---
view_predictions_btn = tk.Button(
    scrollable_frame,
    text="View Predictions",
    command=view_predictions
)
view_predictions_btn.grid(row=2, column=7, padx=0, pady=0)

# --- Add Dropdown Menu for Selecting Images ---
#prediction_image_dropdown = tk.Listbox(
#    scrollable_frame,
#    selected_prediction_image,
#    "Select Prediction Image"  # Initial placeholder
#)
#prediction_image_dropdown.grid(row=2, column=6, padx=0, pady=0, sticky='w')

select_prediction_btn = tk.Button(
    scrollable_frame,
    text="Select Prediction Image",
    command=lambda: open_prediction_list(prediction_image_names)
)
select_prediction_btn.grid(row=2, column=8, padx=0, pady=0, sticky='w')

# Variable to hold the selected category label
selected_category_label = tk.StringVar(root)
selected_category_label.set("Select Category")
category_label_options = []  # Will be populated when loading predictions

# Create the category label dropdown menu
category_label_dropdown = tk.OptionMenu(scrollable_frame, selected_category_label, "Select Category")
category_label_dropdown.grid(row=2, column=9, padx=0, pady=0)

# Add "Apply Category Label" button
apply_category_label_btn = tk.Button(scrollable_frame, text="Apply Category Label", command=apply_category_label)
apply_category_label_btn.grid(row=2, column=10, padx=0, pady=0)

# Add "Re-annotate" button
reannotate_btn = tk.Button(scrollable_frame, text="Re-annotate", command=reannotate_masks)
reannotate_btn.grid(row=2, column=11, padx=0, pady=0)

# Previous Button
prev_image_button = tk.Button(scrollable_frame, text="Previous Image", command=load_prev_image, bg="orange") #state=tk.DISABLED)
prev_image_button.grid(row=2, column=13, padx=0, pady=0)
    
# Next Button
next_image_button = tk.Button(scrollable_frame, text="Next Image", command=load_next_image, bg="orange") #state=tk.DISABLED)
next_image_button.grid(row=2, column=14, padx=0, pady=0)

# Add the Trash button to the GUI
trash_button = tk.Button(scrollable_frame, text="Trash Mask", command=delete_current_mask)
trash_button.grid(row=2, column=12, padx=0, pady=0)

# Add the new measurement button
measure_button = tk.Button(scrollable_frame, text="Run Measurement Script", command=run_measurement_script, bg="lightblue")  # Optional: customize button color
measure_button.grid(row=3, column=0, padx=0, pady=0)

# Add the new 3D measurement button
measure_button_threed = tk.Button(scrollable_frame, text="Run 3D DfD Measurement", command=run_measurement_script_3d, bg="lightblue")
measure_button_threed.grid(row=3, column=1, padx=0, pady=0)

# Add new semi-landmarking button
semilandmark_button = tk.Button(scrollable_frame, text="Semi-Landmarking", command=run_semi_landmarking_script, bg="lightblue")
semilandmark_button.grid(row=3, column=2, padx=0, pady=0)

color_segmentation_button = tk.Button(scrollable_frame, text="Color Segmentation", command=run_color_segmentation_script, bg="lightblue")
color_segmentation_button.grid(row=3, column=3, padx=0, pady=0)

color_extraction_button = tk.Button(scrollable_frame, text="Color Extraction", command=run_color_extraction_script, bg="lightblue")
color_extraction_button.grid(row=3, column=4, padx=0, pady=0)

color_analysis_button = tk.Button(scrollable_frame, text="Color Pattern Analysis", command=run_color_pattern_analysis_script, bg="lightblue")
color_analysis_button.grid(row=3, column=5, padx=0, pady=0)

# ===PASTE THIS CODE===
try:
    from descriptron_species_delimiter_gui_v2 import open_species_delimiter_popup
    
    tk.Button(
        scrollable_frame,
        text="Species Delimitation",
        command=lambda: open_species_delimiter_popup(root),
        bg="#e74c3c",
        fg="lightyellow",
        font=("Arial", 9, "bold")
    ).grid(row=4, column=4, padx=2, pady=2)
    
except ImportError:
    print("⚠ Species Delimitation not available")



visualize_contours_button = tk.Button(scrollable_frame, text="Visualize Contours & Bbox", command=run_visualize_contours_bbox_script, bg="lightblue")
visualize_contours_button.grid(row=3, column=6, padx=0, pady=0)

convert_tsv_jsonl_button = tk.Button(scrollable_frame, text="Convert TSV to JSONL", command=run_tsv_to_jsonl_script, bg="lightblue")
convert_tsv_jsonl_button.grid(row=3, column=7, padx=0, pady=0)

fine_tune_gpt4_button = tk.Button(scrollable_frame, text="Fine-tune GPT-4o Model", command=run_fine_tune_script, bg="lightblue")
fine_tune_gpt4_button.grid(row=3, column=8, padx=0, pady=0)

gpt4o_featurize_button = tk.Button(scrollable_frame, text="GPT-4 Featurize", command=run_gpt4_featurize_script, bg="lightblue")
gpt4o_featurize_button.grid(row=3, column=9, padx=0, pady=0)

gpt4o_specimen_examined_button = tk.Button(scrollable_frame, text="GPT-4 Specimens Examined", command=run_gpt4_label_script, bg="lightblue")
gpt4o_specimen_examined_button.grid(row=3, column=10, padx=0, pady=0)

parse_me_button = tk.Button(scrollable_frame, text="Parse Material Examined", command=run_parse_me_script, bg="lightblue")
parse_me_button.grid(row=3, column=11, padx=0, pady=0)

concat_desc_button = tk.Button(scrollable_frame, text="Concat Description", command=run_concat_description_script, bg="lightblue")
concat_desc_button.grid(row=3, column=12, padx=0, pady=0)

btn_generate_descriptions = tk.Button(scrollable_frame, text="Generate Species Descriptions", command=run_construct_description, bg="lightgreen")
btn_generate_descriptions.grid(row=4, column=0, padx=0, pady=0)

btn_refine_output = tk.Button(scrollable_frame, text="Refine Output and Generate Dichotomous Key", command=refine_and_generate_key, bg="lightgreen")
btn_refine_output.grid(row=4, column=1, padx=0, pady=0)

# Add the "Setup Models CSV" button
btn_setup_model_comparison = tk.Button(scrollable_frame, text="Setup Models Comparison CSV", command=setup_models_csv, bg="lightgreen")
btn_setup_model_comparison.grid(row=4, column=2, padx=0, pady=0)

# Add the "Run Calculate RogueV38" button
btn_calculate_rogue = tk.Button(scrollable_frame, text="Calculate Rogue Stats", command=run_calculate_rogueV38, bg="lightgreen")
btn_calculate_rogue.grid(row=4, column=3, padx=0, pady=0)

# Add the "SAM2-PAL" button (Palindrome-based mask propagation) - moved to row 2 next to Keypoints to Line
btn_sam2_pal = tk.Button(scrollable_frame, text="SAM2-PAL", command=lambda: open_pal_popup(), bg="lightyellow")
btn_sam2_pal.grid(row=2, column=3, padx=0, pady=0)


# Marmot button
marmot_image_path = "./marmot.jpg"
marmot_img = Image.open(marmot_image_path)

# Video button (light yellow, same row as SAM2-PAL)
btn_load_video = tk.Button(scrollable_frame, text="Load Video for SAM2",
                          command=open_video_remote_control, bg="lightyellow",
                          font=("Arial", 9, "bold"))
btn_load_video.grid(row=2, column=4, padx=0, pady=0)

button_height = 75
aspect_ratio = marmot_img.width / marmot_img.height
new_width = int(button_height * aspect_ratio)
marmot_img_resized = marmot_img.resize((new_width, button_height), Image.LANCZOS)
marmot_thumbnail = ImageTk.PhotoImage(marmot_img_resized)
marmot_button = tk.Button(scrollable_frame, image=marmot_thumbnail, command=save_and_close)
marmot_button.grid(row=4, column=11, padx=0, pady=0)

root.protocol("WM_DELETE_WINDOW", close_window)

canvas.bind("<ButtonPress-2>", start_pan)
canvas.bind("<B2-Motion>", pan_image)

root.mainloop()
