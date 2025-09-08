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


root = tk.Tk()
root.title("Descriptron Interactive SAM2 Segmentation")
# Configure the root window to allow the canvas to expand and buttons to stay at the bottom
root.grid_rowconfigure(0, weight=1)  # Row 0 (canvas) expands
root.grid_columnconfigure(0, weight=1)  # Column 0 expands


# Set up the SAM2 model and parameters
sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "../sam2_configs/sam2_hiera_l.yaml"

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
keypoint_labels = [] # Parallel list to hold labels (default: [1, 2, 3, …]).
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

    segmentation_masks = []
    # Convert masks to the format used in this application
    #segmentation_masks = [{'segmentation': mask} for mask in masks]
    current_mask_index = 0  # Reset current_mask_index

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
            segmentation_masks.extend(new_masks)
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
        segmentation_masks.extend(new_masks)

    # Check if any masks were generated
    if not segmentation_masks:
        messagebox.showerror("Error", "No mask was generated from the point prompts.")
        return

    # Update canvas by displaying the mask
    current_mask_index = 0
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
            # If the input isn’t numeric, then simply treat it as a non‐numeric label
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
        coco_output_mask["categories"] = categories
        coco_output_accumulate["categories"] = categories

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



def next_mask():
    global current_mask_index
    if segmentation_masks:
        current_mask_index = (current_mask_index + 1) % len(segmentation_masks)
        apply_zoom()
        # Update the selected label based on the current mode
        if mode == 'prediction_view':
            # Update the selected category label
            category_id = segmentation_masks[current_mask_index].get('category_id', -1)
            category_name = category_id_to_name.get(category_id, "unlabeled")
            selected_category_label.set(category_name)
        else:
            # Original functionality
            label = labels.get(current_mask_index, "unlabeled")
            selected_label.set(label)


#    load_prediction_image(new_image_name)
#    # ADD the same keypoint code:
#    if new_image_name in prediction_data:
#        ann_for_new_image = prediction_data[new_image_name]
#        load_json_keypoints_into_gui(annotations_for_image)

def previous_mask():
    global current_mask_index
    if segmentation_masks:
        current_mask_index = (current_mask_index - 1) % len(segmentation_masks)
        apply_zoom()
        # Update the selected label based on the current mode
        if mode == 'prediction_view':
            # Update the selected category label
            category_id = segmentation_masks[current_mask_index].get('category_id', -1)
            category_name = category_id_to_name.get(category_id, "unlabeled")
            selected_category_label.set(category_name)
        else:
            # Original functionality
            label = labels.get(current_mask_index, "unlabeled")
            selected_label.set(label)

#    load_prediction_image(new_image_name)
#    # ADD the same keypoint code:
#    if new_image_name in prediction_data:
#        ann_for_new_image = prediction_data[new_image_name]
#        load_json_keypoints_into_gui(annotations_for_image)


def apply_category_label():
    global segmentation_masks, current_mask_index
    global category_name_to_id, category_id_to_name, categories, category_label_options
    global labels

    label = selected_category_label.get()
    if label == "Select Category":
        messagebox.showerror("Error", "Please select a category.")
        return

    # Check if the category exists
    if label in category_name_to_id:
        category_id = category_name_to_id[label]
    else:
        # Category does not exist; add new category
        new_category_id = max(category_id_to_name.keys(), default=0) + 1
        category_id_to_name[new_category_id] = label
        category_name_to_id[label] = new_category_id
        categories.append({
            "id": new_category_id,
            "name": label,
            "supercategory": "none"  # Modify as needed
        })
        category_id = new_category_id
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



def apply_category_label():
    global segmentation_masks, current_mask_index
    global category_name_to_id, category_id_to_name, categories, category_label_options

    label = selected_category_label.get()
    if label == "Select Category":
        messagebox.showerror("Error", "Please select a category.")
        return

    # Check if the category exists
    if label in category_name_to_id:
        category_id = category_name_to_id[label]
    else:
        # Category does not exist; add new category
        new_category_id = max(category_id_to_name.keys(), default=0) + 1
        category_id_to_name[new_category_id] = label
        category_name_to_id[label] = new_category_id
        categories.append({
            "id": new_category_id,
            "name": label,
            "supercategory": "none"  # Modify as needed
        })
        category_id = new_category_id
        # Update the category dropdowns
        if label not in category_label_options:
            category_label_options.append(label)
            update_category_label_dropdown()

    # Update the category_id of the current mask
    segmentation_masks[current_mask_index]['category_id'] = category_id

    # Set the selected_category_label to the applied label
    selected_category_label.set(label)

    messagebox.showinfo("Success", f"Category '{label}' applied to the current mask.")




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
    """
def save_mask_and_contour_via2(overwrite=False):
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

    # Initialize or update the instance counter for each label
    if label not in instance_counters:
        instance_counters[label] = 1
    else:
        instance_counters[label] += 1

    # Generate a unique annotation ID
    annotation_id = int(time.time() * 1000) + current_mask_index

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

    # Create annotation
    annotation = {
        "id": annotation_id,
        "image_id": file_path_base,  # Adjust this as needed
        "category_id": category_id,
        "segmentation": coco_segmentation,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0
    }

    # Append annotation to both coco_output_mask and coco_output_accumulate
    coco_output_mask["annotations"].append(annotation)
    coco_output_accumulate["annotations"].append(annotation)

    # Ensure categories are added to coco_output_mask and coco_output_accumulate
    if "categories" not in coco_output_mask or not coco_output_mask["categories"]:
        coco_output_mask["categories"] = categories.copy()
    if "categories" not in coco_output_accumulate or not coco_output_accumulate["categories"]:
        coco_output_accumulate["categories"] = categories.copy()

    # Save the mask image if necessary
    mask_filename = f"{file_path_base}_{label}_{current_mask_index}_mask.png"
    cv2.imwrite(mask_filename, mask_8bit)

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
    """
    global coco_output_accumulate, file_path_base

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

            segmentation_masks.append({'segmentation': mask})

            # Associate the label with the mask
            category_id = annotation.get('category_id', -1)
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
        combiner_script = script_dir / 'coco_combiner_V11_updated.py'
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
    """


    if not point_coords:
        messagebox.showerror("Error", "No keypoints available to convert into a mask.")
        return

    if len(point_coords) < 3:
        messagebox.showerror("Error", "At least three keypoints are required to form a polygon.")
        return

    # Order the keypoints based on the sequence they were added
    ordered_points = order_keypoints_by_sequence(point_coords, point_orders)

    # Create a mask from the ordered polygon
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    pts = np.array([ordered_points], dtype=np.int32)
    cv2.fillPoly(mask, pts, 1)

    # Add the new mask to the segmentation_masks list
    segmentation_masks.append({'segmentation': mask})
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

    messagebox.showinfo("Success", "Keypoints have been converted to a mask and visualized.")


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
        semi_landmarking_script = script_dir / 'measure' / 'semi_landmark_and_kpts_procrustesV33_no_pacmap_fork.py'
        
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
    label = selected_label.get()


    # If "Custom" is selected, prompt the user for a custom label
    if label == "Custom":
        label = simpledialog.askstring("Input", "Enter custom label:", parent=root)

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
            categories.append({
                "id": new_category_id,
                "name": label,
                "supercategory": "none"  # Modify as needed
            })
            category_id = new_category_id
            # Update the label options in the dropdown
            if label not in label_options:
                label_options.append(label)
                update_label_dropdown()

        # Assign the category_id to the current mask
        segmentation_masks[current_mask_index]['category_id'] = category_id

        # Proceed to save the mask
        existing_mask_file = f"{file_path_base}_{label}_{current_mask_index}_mask.png"
        if os.path.exists(existing_mask_file) and not auto:
            # Prompt the user to overwrite or cancel
            result = messagebox.askyesno("File Exists", f"A mask file for this label already exists.\nDo you want to overwrite it?")
            if result:
                # Overwrite the existing file
                save_mask_and_contour_via2(overwrite=True)
            else:
                # Do not overwrite, do nothing
                messagebox.showinfo("Info", "Mask not saved.")
        else:
            # Save the mask and contour
            save_mask_and_contour_via2(overwrite=True if auto else False)

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
load_btn.grid(row=0, column=0, padx=1, pady=1)

load_prompt_btn = tk.Button(scrollable_frame, text="Load Image for SAM2-Prompt", command=load_image_for_prompt)
load_prompt_btn.grid(row=0, column=1, padx=1, pady=1)

bbox_button = tk.Button(scrollable_frame, text="Draw Bounding Box", command=toggle_bbox_mode)
bbox_button.grid(row=0, column=2, padx=1, pady=1)

remove_bbox_btn = tk.Button(scrollable_frame, text="Remove Last BBox", command=remove_last_bbox)
remove_bbox_btn.grid(row=0, column=3, padx=1, pady=1)

# Dropdown for selecting point label (Positive or Negative)
selected_point_label = tk.StringVar(root)
selected_point_label.set("Positive")
point_label_options = ["Positive", "Negative"]
point_label_dropdown = tk.OptionMenu(scrollable_frame, selected_point_label, *point_label_options)
point_label_dropdown.grid(row=0, column=4, padx=1, pady=1)
# Button to toggle point prompt mode
point_prompt_btn = tk.Button(scrollable_frame, text="Point Prompt", command=toggle_point_prompt_mode)
point_prompt_btn.grid(row=0, column=5, padx=1, pady=1)

# Button to clear all points
clear_points_btn = tk.Button(scrollable_frame, text="Clear Points", command=clear_points)
clear_points_btn.grid(row=0, column=6, padx=1, pady=1)

# Button to apply SAM2 using points and bounding boxes
apply_points_prompt_btn = tk.Button(scrollable_frame, text="Apply Points+BBox SAM2", command=apply_sam2_prompt_with_points)
apply_points_prompt_btn.grid(row=0, column=7, padx=1, pady=1)

apply_prompt_btn = tk.Button(scrollable_frame, text="Apply SAM2-Prompt", command=apply_sam2_prompt)
apply_prompt_btn.grid(row=0, column=8, padx=1, pady=1)

load_labels_btn = tk.Button(scrollable_frame, text="Load Labels", command=load_labels)
load_labels_btn.grid(row=0, column=9, padx=1, pady=1)

selected_label = tk.StringVar(root)
selected_label.set("Select Label")
label_options = ["Custom","Trash"]
label_dropdown = tk.OptionMenu(scrollable_frame, selected_label, *label_options)
label_dropdown.grid(row=0, column=10, padx=1, pady=1)

apply_label_btn = tk.Button(scrollable_frame, text="Apply Label", command=apply_label)
apply_label_btn.grid(row=0, column=11, padx=1, pady=1)

next_btn = tk.Button(scrollable_frame, text="Next Mask", command=next_mask)
next_btn.grid(row=1, column=0, padx=1, pady=1)
prev_btn = tk.Button(scrollable_frame, text="Previous Mask", command=previous_mask)
prev_btn.grid(row=1, column=1, padx=1, pady=1)

save_unlabeled_btn = tk.Button(scrollable_frame, text="Save Unlabeled Masks", command=save_unlabeled_masks)
save_unlabeled_btn.grid(row=1, column=2, padx=1, pady=1)

zoom_in_btn = tk.Button(scrollable_frame, text="Zoom In", command=zoom_in)
zoom_in_btn.grid(row=1, column=3, padx=1, pady=1)

zoom_out_btn = tk.Button(scrollable_frame, text="Zoom Out", command=zoom_out)
zoom_out_btn.grid(row=1, column=4, padx=1, pady=1)

eraser_btn = tk.Button(scrollable_frame, text="Eraser: OFF", command=select_eraser)
eraser_btn.grid(row=1, column=5, padx=1, pady=1)

paintbrush_btn = tk.Button(scrollable_frame, text="Paintbrush: OFF", command=select_paintbrush)
paintbrush_btn.grid(row=1, column=6, padx=1, pady=1)


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
brush_size_menu.grid(row=1, column=7, padx=1, pady=1)

# Bind the dropdown change to update brush size
brush_size_var.trace("w", update_brush_size)


# Button for toggling keypoint edit mode
keypoint_edit_btn = tk.Button(scrollable_frame, text="Keypoint Edit: OFF", command=toggle_keypoint_edit_mode)
keypoint_edit_btn.grid(row=2, column=0, padx=1, pady=1)


# Add the new "Keypoints to Mask" button
keypoints_to_mask_btn = tk.Button(scrollable_frame, text="Keypoints to Mask", command=keypoints_to_mask)
keypoints_to_mask_btn.grid(row=2, column=1, padx=1, pady=1)

coco_to_min_btn = tk.Button(scrollable_frame, text="COCO->MinCOCO", command=run_coco_converter_script)
coco_to_min_btn.grid(row=2, column=2, padx=1, pady=1)

load_annotations_btn = tk.Button(scrollable_frame, text="Load Annotations", command=load_annotations)
load_annotations_btn.grid(row=1, column=8, padx=1, pady=1)

# Add the Multi-Mask button to the second row
multi_mask_btn = tk.Button(scrollable_frame, text="Multi-Mask Mode: OFF", command=toggle_multi_mask_mode)
multi_mask_btn.grid(row=1, column=9, padx=1, pady=1)

combine_json_btn = tk.Button(scrollable_frame, text="Combine COCO", command=run_coco_combiner_script)
combine_json_btn.grid(row=1, column=10, padx=1, pady=1)

remove_files_from_json_btn = tk.Button(scrollable_frame, text="Remove images COCO", command=run_remove_images_script)
remove_files_from_json_btn.grid(row=1, column=11, padx=1, pady=1)

# Detectron2 Training Button

# Train Detectron2 button
train_button = tk.Button(scrollable_frame, text="Train Detectron2", command=open_training_popup)
train_button.grid(row=2, column=3, padx=1, pady=1)

predict_btn = tk.Button(scrollable_frame, text="Predict Detectron2", command=predict_and_filter_popup)
predict_btn.grid(row=2, column=4, padx=1, pady=1)  # Adjust row and column as needed


# --- Add 'View Predictions' Button ---
view_predictions_btn = tk.Button(
    scrollable_frame,
    text="View Predictions",
    command=view_predictions
)
view_predictions_btn.grid(row=2, column=5, padx=1, pady=1)  # Adjust row and column as needed

# --- Add Dropdown Menu for Selecting Images ---
#prediction_image_dropdown = tk.Listbox(
#    scrollable_frame,
#    selected_prediction_image,
#    "Select Prediction Image"  # Initial placeholder
#)
#prediction_image_dropdown.grid(row=2, column=6, padx=1, pady=1, sticky='w')

select_prediction_btn = tk.Button(
    scrollable_frame,
    text="Select Prediction Image",
    command=lambda: open_prediction_list(prediction_image_names)
)
select_prediction_btn.grid(row=2, column=6, padx=1, pady=1, sticky='w')

# Variable to hold the selected category label
selected_category_label = tk.StringVar(root)
selected_category_label.set("Select Category")
category_label_options = []  # Will be populated when loading predictions

# Create the category label dropdown menu
category_label_dropdown = tk.OptionMenu(scrollable_frame, selected_category_label, "Select Category")
category_label_dropdown.grid(row=2, column=7, padx=1, pady=1)

# Add "Apply Category Label" button
apply_category_label_btn = tk.Button(scrollable_frame, text="Apply Category Label", command=apply_category_label)
apply_category_label_btn.grid(row=2, column=8, padx=1, pady=1)

# Add "Re-annotate" button
reannotate_btn = tk.Button(scrollable_frame, text="Re-annotate", command=reannotate_masks)
reannotate_btn.grid(row=2, column=9, padx=1, pady=1)

# Previous Button
prev_image_button = tk.Button(scrollable_frame, text="Previous Image", command=load_prev_image, bg="orange") #state=tk.DISABLED)
prev_image_button.grid(row=2, column=11, padx=1, pady=1)
    
# Next Button
next_image_button = tk.Button(scrollable_frame, text="Next Image", command=load_next_image, bg="orange") #state=tk.DISABLED)
next_image_button.grid(row=2, column=12, padx=1, pady=1)

# Add the Trash button to the GUI
trash_button = tk.Button(scrollable_frame, text="Trash Mask", command=delete_current_mask)
trash_button.grid(row=2, column=10, padx=1, pady=1)

# Add the new measurement button
measure_button = tk.Button(scrollable_frame, text="Run Measurement Script", command=run_measurement_script, bg="lightblue")  # Optional: customize button color
measure_button.grid(row=3, column=0, padx=1, pady=1)

# Add the new 3D measurement button
measure_button_threed = tk.Button(scrollable_frame, text="Run 3D DfD Measurement", command=run_measurement_script_3d, bg="lightblue")
measure_button_threed.grid(row=3, column=1, padx=1, pady=1)

# Add new semi-landmarking button
semilandmark_button = tk.Button(scrollable_frame, text="Semi-Landmarking", command=run_semi_landmarking_script, bg="lightblue")
semilandmark_button.grid(row=3, column=2, padx=1, pady=1)

color_segmentation_button = tk.Button(scrollable_frame, text="Color Segmentation", command=run_color_segmentation_script, bg="lightblue")
color_segmentation_button.grid(row=3, column=3, padx=1, pady=1)

color_extraction_button = tk.Button(scrollable_frame, text="Color Extraction", command=run_color_extraction_script, bg="lightblue")
color_extraction_button.grid(row=3, column=4, padx=1, pady=1)

color_analysis_button = tk.Button(scrollable_frame, text="Color Pattern Analysis", command=run_color_pattern_analysis_script, bg="lightblue")
color_analysis_button.grid(row=3, column=5, padx=1, pady=1)

visualize_contours_button = tk.Button(scrollable_frame, text="Visualize Contours & Bbox", command=run_visualize_contours_bbox_script, bg="lightblue")
visualize_contours_button.grid(row=3, column=6, padx=1, pady=1)

convert_tsv_jsonl_button = tk.Button(scrollable_frame, text="Convert TSV to JSONL", command=run_tsv_to_jsonl_script, bg="lightblue")
convert_tsv_jsonl_button.grid(row=3, column=7, padx=1, pady=1)

fine_tune_gpt4_button = tk.Button(scrollable_frame, text="Fine-tune GPT-4o Model", command=run_fine_tune_script, bg="lightblue")
fine_tune_gpt4_button.grid(row=3, column=8, padx=1, pady=1)

gpt4o_featurize_button = tk.Button(scrollable_frame, text="GPT-4 Featurize", command=run_gpt4_featurize_script, bg="lightblue")
gpt4o_featurize_button.grid(row=3, column=9, padx=1, pady=1)

gpt4o_specimen_examined_button = tk.Button(scrollable_frame, text="GPT-4 Specimens Examined", command=run_gpt4_label_script, bg="lightblue")
gpt4o_specimen_examined_button.grid(row=3, column=10, padx=1, pady=1)

parse_me_button = tk.Button(scrollable_frame, text="Parse Material Examined", command=run_parse_me_script, bg="lightblue")
parse_me_button.grid(row=3, column=11, padx=1, pady=1)

concat_desc_button = tk.Button(scrollable_frame, text="Concat Description", command=run_concat_description_script, bg="lightblue")
concat_desc_button.grid(row=3, column=12, padx=1, pady=1)

btn_generate_descriptions = tk.Button(scrollable_frame, text="Generate Species Descriptions", command=run_construct_description, bg="lightgreen")
btn_generate_descriptions.grid(row=4, column=0, padx=1, pady=1)

btn_refine_output = tk.Button(scrollable_frame, text="Refine Output and Generate Dichotomous Key", command=refine_and_generate_key, bg="lightgreen")
btn_refine_output.grid(row=4, column=1, padx=1, pady=1)

# Add the "Setup Models CSV" button
btn_setup_model_comparison = tk.Button(scrollable_frame, text="Setup Models Comparison CSV", command=setup_models_csv, bg="lightgreen")
btn_setup_model_comparison.grid(row=4, column=2, padx=1, pady=1)

# Add the "Run Calculate RogueV38" button
btn_calculate_rogue = tk.Button(scrollable_frame, text="Calculate Rogue Stats", command=run_calculate_rogueV38, bg="lightgreen")
btn_calculate_rogue.grid(row=4, column=3, padx=1, pady=1)


# Marmot button
marmot_image_path = "./marmot.jpg"
marmot_img = Image.open(marmot_image_path)
button_height = 75
aspect_ratio = marmot_img.width / marmot_img.height
new_width = int(button_height * aspect_ratio)
marmot_img_resized = marmot_img.resize((new_width, button_height), Image.LANCZOS)
marmot_thumbnail = ImageTk.PhotoImage(marmot_img_resized)
marmot_button = tk.Button(scrollable_frame, image=marmot_thumbnail, command=save_and_close)
marmot_button.grid(row=4, column=11, padx=1, pady=1)

root.protocol("WM_DELETE_WINDOW", close_window)

canvas.bind("<ButtonPress-2>", start_pan)
canvas.bind("<B2-Motion>", pan_image)

root.mainloop()

