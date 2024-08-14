import configparser
import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
import sys

# Setup logger
setup_logger()

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the paths from the config file
train_data = config['Paths']['train_data']
val_data = config['Paths']['val_data']
test_data = config['Paths']['test_data']
model_output = config['Paths']['model_output']
plot_output = config['Paths']['plot_output']
json_input = config['Paths']['json_input']
json_output = config['Paths']['json_output']
metadata_json_path = config['Paths']['metadata_json_path']
detectron2_config = config['Paths']['detectron2_config']  # Path to the .yaml config file
detectron2_weights = config['Paths']['detectron2_weights']  # Path to the model weights file

# Load the JSON file with category information (metadata from Detectron2 training)
with open(metadata_json_path, 'r') as f:
    coco_data = json.load(f)

# Extract category names
category_names = coco_data['thing_classes']
category_id_to_name = {i: name for i, name in enumerate(category_names)}
print(category_names)
sys.stdout.flush()

# Function to get the dataset dictionaries
def get_dicts(base_path, prefixes):
    dataset_dicts = []
    total_files_found = 0
    for view in prefixes:
        view_path = os.path.join(base_path, view)
        for root, _, files in os.walk(view_path):
            for filename in files:
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    total_files_found += 1
                    record = {}
                    filepath = os.path.join(root, filename)
                    height, width = cv2.imread(filepath).shape[:2]

                    record["file_name"] = filepath
                    record["image_id"] = filepath
                    record["height"] = height
                    record["width"] = width
                    record["annotations"] = []
                    dataset_dicts.append(record)
    print(f"Total number of matching files: {total_files_found}")
    sys.stdout.flush()
    return dataset_dicts

# Register the datasets
prefixes = ["dorsal", "lateral", "frontal"]
for view in prefixes:
    dataset_name = f"coleoptera_{view}"
    DatasetCatalog.register(dataset_name, lambda view=view: get_dicts(test_data, [view]))
    MetadataCatalog.get(dataset_name).set(thing_classes=category_names)

# Set the configuration parameters for instance segmentation
cfg_instance = get_cfg()
cfg_instance.MODEL.DEVICE = 'cuda'  # or 'cpu' if you don't have a GPU
cfg_instance.merge_from_file(detectron2_config)  # Load the config file from the config.ini path
cfg_instance.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
cfg_instance.MODEL.WEIGHTS = detectron2_weights  # Load the model weights from the config.ini path
cfg_instance.DATALOADER.NUM_WORKERS = 2
cfg_instance.SOLVER.IMS_PER_BATCH = 2
cfg_instance.SOLVER.BASE_LR = 0.0001
cfg_instance.SOLVER.MAX_ITER = 25000
cfg_instance.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg_instance.MODEL.ROI_HEADS.NUM_CLASSES = len(category_names)

# Initialize predictor
predictor_instance = DefaultPredictor(cfg_instance)

# Path to save the results
output_dir = json_output  # Use the json_output path from the config file
os.makedirs(output_dir, exist_ok=True)
results_json_path = os.path.join(output_dir, "inference_results.json")
visualization_output_dir = os.path.join(output_dir, "visualizations")
os.makedirs(visualization_output_dir, exist_ok=True)

# Load existing predictions
existing_predictions = set()
if os.path.exists(results_json_path):
    with open(results_json_path, 'r') as f:
        existing_results = json.load(f)
        for result in existing_results:
            existing_predictions.add(result["file_name"])

# Initialize the results file
all_predictions = []

# Function to extract contours from segmentation masks
def extract_contours(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.flatten().tolist() for contour in contours]

# Function to process each image and get predictions
def process_image(image_path):
    img = cv2.imread(image_path)
    outputs = predictor_instance(img)
    instances = outputs["instances"].to("cpu")
    pred_boxes = instances.pred_boxes.tensor.numpy()
    pred_classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    masks = instances.pred_masks.numpy()

    predictions = []
    for box, cls, score, mask in zip(pred_boxes, pred_classes, scores, masks):
        if int(cls) in category_id_to_name:  # Ensure the class is valid
            category_name = category_id_to_name[int(cls)]
        else:
            continue  # Skip unknown classes
        contours = extract_contours(mask)
        predictions.append({
            "bbox": box.tolist(),
            "category_id": int(cls),
            "category_name": category_name,  # Translate category ID to name
            "score": float(score),
            "contours": contours
        })

    return predictions, outputs

# Initialize counters
processed_files = 0
category_counts = {name: 0 for name in category_names}

# Process the dataset and collect predictions
for view in prefixes:
    dataset_name = f"coleoptera_{view}"
    dataset_dicts = get_dicts(test_data, [view])
    print(f"Dataset dicts for {view}: {dataset_dicts[:5]}")  # Debug output (printing only first 5 for brevity)
    sys.stdout.flush()
    if dataset_dicts:
        for idx, d in enumerate(dataset_dicts):
            file_name = d["file_name"]
            if file_name in existing_predictions:
                continue  # Skip already processed files
            image_id = d["image_id"]
            predictions, outputs = process_image(file_name)
            result = {
                "image_id": image_id,
                "file_name": file_name,
                "predictions": predictions
            }

            # Append prediction to results file
            if os.path.exists(results_json_path):
                with open(results_json_path, 'r+') as f:
                    all_predictions = json.load(f)
                    all_predictions.append(result)
                    f.seek(0)
                    json.dump(all_predictions, f)
            else:
                with open(results_json_path, 'w') as f:
                    json.dump([result], f)

            # Update counters every 100 files
            processed_files += 1
            for prediction in predictions:
                category_name = prediction['category_name']
                category_counts[category_name] += 1

            if processed_files % 100 == 0:
                print(f"Processed {processed_files} files")
                print(f"Category counts: {category_counts}")
                sys.stdout.flush()

            # Save visualizations for a few images
            if idx < 10:  # Save visualizations for the first 10 images
                img = cv2.imread(file_name)
                v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(dataset_name), scale=1.2)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                result_img = v.get_image()[:, :, ::-1]
                cv2.imwrite(os.path.join(visualization_output_dir, f"{view}_{idx}_prediction.jpg"), result_img)

# Check the structure of all_predictions before saving
if os.path.exists(results_json_path):
    with open(results_json_path, 'r') as f):
        all_predictions = json.load(f)
print(f"All predictions: {all_predictions[:5]}")  # Print the first 5 predictions for inspection
sys.stdout.flush()

# Save all predictions to a final file (optional if needed)
try:
    with open(results_json_path, 'w') as f:
        json.dump(all_predictions, f)
    print(f"Saved inference results to {results_json_path}.")
    sys.stdout.flush()
except Exception as e:
    print(f"Error saving inference results: {e}")
    sys.stdout.flush()
    raise

# Evaluate the model and print AP scores for all views
summary_table = {}
for view in prefixes:
    evaluator = COCOEvaluator(f"coleoptera_{view}", cfg_instance, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg_instance, f"coleoptera_{view}")
    results = inference_on_dataset(predictor_instance.model, val_loader, evaluator)

    # Extract AP scores for each class
    ap_scores = results['bbox']['AP-per-category']
    for idx, cls in enumerate(category_names):
        summary_table[f"{cls}_{view}"] = ap_scores[idx]

# Print summary table
summary_df = pd.DataFrame(list(summary_table.items()), columns=["Class_View", "AP"])
print(summary_df)
sys.stdout.flush()

# Save summary table to CSV
summary_df.to_csv(os.path.join(output_dir, "ap_pred_summary.csv"), index=False)
