import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
import argparse
from shapely.geometry import Polygon  # For accurate area calculation
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import sys
import logging

# Setup logger
def setup_logging_custom():
    """
    Sets up logging to both console and a log file.
    """
    logger = logging.getLogger('Detectron2_Predict_Filter')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('detectron2_predict_filter.log', mode='w')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Detectron2 Prediction Script with Filtering and Subdirectory Support")
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Path to the input image directory containing .jpg, .jpeg, or .png files. Can include subdirectories for different categories.'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        required=True,
        help='Path to the Detectron2 configuration YAML file.'
    )
    parser.add_argument(
        '--model_weights',
        type=str,
        required=True,
        help='Path to the model weights file (.pth).'
    )
    parser.add_argument(
        '--metadata_json',
        type=str,
        required=True,
        help='Path to the metadata JSON file containing category information.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where the output JSON files and visualizations will be saved.'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='Name of the dataset to use as a suffix for output files.'
    )
    parser.add_argument(
        '--categories',
        type=str,
        default='',
        help='Optional comma-separated list of categories to retain (e.g., left_elytra,pronotum,head,rostrum). If not provided, all categories are retained.'
    )
    parser.add_argument(
        '--sample_images',
        type=int,
        default=10,
        help='Number of sample images to visualize predictions. Default is 10.'
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=0.8,
        help='Prediction threshold for classifying detections. Default is 0.8.'
    )
    parser.add_argument(
        '--total_iters',
        type=float,
        default=25000,
        help='Total number of iterations performed during your training run. Default is 25,000.'
    )

    return parser.parse_args()

# Function to get dataset dictionaries with subdirectory support and exclusion of output directories
def get_dicts(img_dir, logger, exclude_dirs=None):
    """
    Retrieves image information from the specified directory and its subdirectories, excluding certain directories.

    Args:
        img_dir (str): Path to the image directory.
        logger (logging.Logger): Logger for logging messages.
        exclude_dirs (list, optional): List of directory names to exclude. Defaults to None.

    Returns:
        list: List of image records in Detectron2 format.
    """
    if exclude_dirs is None:
        exclude_dirs = []
    dataset_dicts = []
    total_files_found = 0
    for root, dirs, files in os.walk(img_dir):
        # Modify dirs in-place to exclude specified directories
        dirs[:] = [d for d in dirs if d.lower() not in [ex.lower() for ex in exclude_dirs]]
        
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(root, filename)
                if not os.path.isfile(filepath):
                    continue  # Skip if not a file
                
                # Compute relative path to handle subdirectories
                rel_path = os.path.relpath(filepath, img_dir)
                
                img = cv2.imread(filepath)
                if img is None:
                    logger.warning(f"Failed to read image: {filepath}. Skipping.")
                    continue
                height, width = img.shape[:2]

                total_files_found += 1
                record = {
                    "file_name": rel_path,  # Relative path includes subdirectory
                    "id": total_files_found,  # Assign a unique integer ID
                    "height": height,
                    "width": width
                }
                dataset_dicts.append(record)
    logger.info(f"Total number of matching files: {total_files_found}")
    return dataset_dicts

def main():
    args = parse_arguments()
    logger = setup_logging_custom()
    
    # Load the metadata JSON file containing category information
    if not os.path.exists(args.metadata_json):
        logger.error(f"Metadata JSON file not found: {args.metadata_json}")
        sys.exit(1)
    
    with open(args.metadata_json, 'r') as f:
        try:
            coco_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON file: {e}")
            sys.exit(1)
    
    category_names = coco_data.get('thing_classes', [])
    if not category_names:
        logger.error("The metadata JSON does not contain the 'thing_classes' key or it's empty.")
        sys.exit(1)
    
    # Create a mapping from category ID to category name
    category_id_to_name = {i: name for i, name in enumerate(category_names)}
    
    # Log category information for debugging
    logger.info("Category ID to Name mapping:")
    for idx, name in category_id_to_name.items():
        logger.info(f"Category ID {idx}: {name}")
    
    # Register the dataset
    dataset_name = args.dataset_name  # Use the dataset_name from arguments
    exclude_dirs = ['melanastera_visualizations', 'visualizations']  # Directories to exclude from input images
    DatasetCatalog.register(dataset_name, lambda: get_dicts(args.image_dir, logger, exclude_dirs=exclude_dirs))
    MetadataCatalog.get(dataset_name).set(thing_classes=category_names)
    
    # Set the configuration parameters for instance segmentation
    cfg_instance = get_cfg()
    cfg_instance.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_instance.merge_from_file(args.config_file)
    cfg_instance.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold  # set a custom testing threshold
    cfg_instance.MODEL.WEIGHTS = args.model_weights
    cfg_instance.DATALOADER.NUM_WORKERS = 2
    cfg_instance.SOLVER.IMS_PER_BATCH = 2
    cfg_instance.SOLVER.BASE_LR = 0.0001
    cfg_instance.SOLVER.MAX_ITER = int(args.total_iters)
    cfg_instance.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg_instance.MODEL.ROI_HEADS.NUM_CLASSES = len(category_names)
    
    # Log the number of classes
    logger.info(f"Model is configured with {cfg_instance.MODEL.ROI_HEADS.NUM_CLASSES} classes.")
    
    # Initialize predictor
    predictor_instance = DefaultPredictor(cfg_instance)
    
    # Create output file paths with dataset_name as suffix
    os.makedirs(args.output_dir, exist_ok=True)
    all_results_json_path = os.path.join(args.output_dir, f"{args.dataset_name}_all_predictions.json")
    filtered_results_json_path = os.path.join(args.output_dir, f"{args.dataset_name}_filtered_predictions.json")
    
    # Place visualizations in a separate directory outside image_dir
    visualization_output_dir = os.path.join(args.output_dir, f"{args.dataset_name}_visualizations")
    os.makedirs(visualization_output_dir, exist_ok=True)
    
    # Initialize the results lists
    images = []
    annotations = []
    filtered_predictions = []
    annotation_id = 1  # Initialize annotation ID counter
    category_counts = {name: 0 for name in category_names}  # Initialize category counts
    
    # Function to extract segmentation and calculate area from masks
    def extract_segmentation(mask):
        """
        Extracts segmentation polygons and calculates the total area from a binary mask.
    
        Args:
            mask (numpy.ndarray): Binary mask of the object.
    
        Returns:
            tuple: (segmentation, area)
                - segmentation (list): List of segmentation polygons in COCO format.
                - area (float): Total area of all valid polygons.
        """
        # Find contours using OpenCV
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        segmentation = []
        area = 0.0

        for contour in contours:
            # Check if the contour has at least 4 points (required for Shapely's Polygon)
            if len(contour) < 4:
                logging.warning(f"Skipping contour with insufficient points: {len(contour)} points.")
                continue  # Skip this contour

            # Flatten the contour and convert to a list
            contour_flat = contour.flatten().tolist()
        
            # Check if the flattened contour has at least 6 coordinates (3 points)
            if len(contour_flat) >= 6:
                segmentation.append(contour_flat)
            else:
                logging.warning(f"Contour flattened length insufficient for segmentation: {len(contour_flat)} coordinates.")
                continue  # Skip this contour

            # Calculate area using Shapely's Polygon
            try:
                # Reshape contour to (num_points, 2) for Polygon
                contour_coords = contour.reshape(-1, 2)
                poly = Polygon(contour_coords)
                if poly.is_valid:
                    area += poly.area
                else:
                    logging.warning("Invalid polygon created; skipping area addition for this contour.")
            except ValueError as ve:
                logging.warning(f"ValueError encountered when creating Polygon: {ve}. Skipping this contour.")
                continue  # Skip this contour if Polygon creation fails

        return segmentation, area
    
    # Function to process each image and get annotations
    def process_image(image_entry):
        nonlocal annotation_id  # Declare nonlocal to modify the outer variable
        image_path = os.path.join(args.image_dir, image_entry["file_name"])
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Image file not found or cannot be read: {image_path}")
            return []
        outputs = predictor_instance(img)
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes.tensor.numpy()
        pred_classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        masks = instances.pred_masks.numpy()

        image_annotations = []
        for box, cls, score, mask in zip(pred_boxes, pred_classes, scores, masks):
            cls_int = int(cls)
            category_name = category_id_to_name.get(cls_int, "unknown")
            if category_name == "unknown":
                logger.warning(f"Unknown class ID {cls_int} encountered in image '{image_entry['file_name']}'. Skipping this prediction.")
                continue  # Skip unknown classes
            segmentation, area = extract_segmentation(mask)
            if not segmentation:
                logger.warning(f"No valid segmentation found for image '{image_entry['file_name']}', category '{category_name}'. Skipping this prediction.")
                continue  # Skip if no valid segmentation

            # Convert bbox from [x_min, y_min, x_max, y_max] to [x, y, width, height]
            x_min, y_min, x_max, y_max = box
            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
            
            annotation = {
                "id": annotation_id,
                "image_id": image_entry["id"],
                "category_id": cls_int + 1,  # COCO category IDs start at 1
                "segmentation": segmentation,
                "bbox": bbox,
                "area": float(area),
                "iscrowd": 0
            }
            annotations.append(annotation)
            image_annotations.append(annotation)
            annotation_id += 1

        return image_annotations

    # Get all image file paths
    dataset_dicts = get_dicts(args.image_dir, logger, exclude_dirs=exclude_dirs)
    logger.info(f"Starting prediction on {len(dataset_dicts)} images.")
    
    # Process the dataset and collect predictions
    for idx, entry in enumerate(dataset_dicts):
        image_name = entry["file_name"]
        image_id = entry["id"]

        logger.info(f"Processing image {idx + 1}/{len(dataset_dicts)}: {image_name}")

        # Add image information to the 'images' list
        image_info = {
            "id": image_id,
            "file_name": image_name,  # Relative path includes subdirectory
            "height": entry["height"],
            "width": entry["width"]
        }
        images.append(image_info)

        # Process image and get annotations
        image_annotations = process_image(entry)

        # Update category counts
        for annotation in image_annotations:
            category_id = annotation['category_id'] - 1  # Adjusting back to zero-based index
            category_name = category_id_to_name.get(category_id, "unknown")
            if category_name != "unknown":
                category_counts[category_name] += 1

        # Save filtered predictions if categories are specified
        if args.categories:
            desired_categories = set([cat.strip() for cat in args.categories.split(',') if cat.strip()])
            filtered_preds = [ann for ann in image_annotations if category_id_to_name.get(ann['category_id'] - 1, "") in desired_categories]
            if filtered_preds:
                filtered_entry = {
                    "image_id": image_id,
                    "file_name": image_name,
                    "predictions": filtered_preds
                }
                filtered_predictions.append(filtered_entry)

        # Save visualizations for sample images
        if idx < args.sample_images:
            img_path = os.path.join(args.image_dir, image_name)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Image file not found or cannot be read for visualization: {img_path}")
                continue
            outputs = predictor_instance(img)
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(dataset_name), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_img = v.get_image()[:, :, ::-1]
            # Sanitize filename to avoid issues with subdirectories
            base_image_name = os.path.basename(image_name)
            visualization_filename = f"{args.dataset_name}_sample_{idx + 1}_{base_image_name}"
            visualization_path = os.path.join(visualization_output_dir, visualization_filename)
            cv2.imwrite(visualization_path, result_img)

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1} files")
            logger.info(f"Category counts so far: {category_counts}")

    logger.info(f"Finished processing {len(dataset_dicts)} files.")
    logger.info(f"Total Category Counts: {category_counts}")

    # Compile the final COCO format JSON
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "sclerite"} for i, name in enumerate(category_names)
        ]
    }

    # Save all predictions to a JSON file in COCO format
    try:
        with open(all_results_json_path, 'w') as f:
            # Use separators to make JSON single-lined
            json.dump(coco_output, f, separators=(',', ':'))
        logger.info(f"Saved all predictions in COCO format to '{all_results_json_path}'.")
    except Exception as e:
        logger.error(f"Error saving all predictions: {e}")
        sys.exit(1)

    # Save filtered predictions to a separate JSON file if categories are specified
    if args.categories:
        # Create a filtered COCO format JSON
        filtered_coco_output = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i + 1, "name": name, "supercategory": "sclerite"} for i, name in enumerate(category_names) if name in desired_categories
            ]
        }
        # Map image_id to image info for filtered images
        filtered_image_ids = set([entry["image_id"] for entry in filtered_predictions])
        for image in images:
            if image["id"] in filtered_image_ids:
                filtered_coco_output["images"].append(image)
        # Add corresponding annotations
        for annotation in annotations:
            if annotation["image_id"] in filtered_image_ids and category_id_to_name.get(annotation["category_id"] - 1, "") in desired_categories:
                filtered_coco_output["annotations"].append(annotation)
        
        try:
            with open(filtered_results_json_path, 'w') as f:
                # Use separators to make JSON single-lined
                json.dump(filtered_coco_output, f, separators=(',', ':'))
            logger.info(f"Saved filtered predictions in COCO format to '{filtered_results_json_path}'.")
        except Exception as e:
            logger.error(f"Error saving filtered predictions: {e}")
            sys.exit(1)

    # Optionally, save the category counts to a CSV file
    try:
        summary_df = pd.DataFrame(list(category_counts.items()), columns=["Category", "Count"])
        summary_csv_path = os.path.join(args.output_dir, f"{args.dataset_name}_category_counts.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved category counts summary to '{summary_csv_path}'.")
    except Exception as e:
        logger.error(f"Error saving category counts summary: {e}")

    logger.info("Prediction and filtering completed successfully.")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
