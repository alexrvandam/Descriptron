#!/usr/bin/env python3
import numpy as np
import os
import json
import cv2
import torch
import random
import logging
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import warnings
import shutil
import subprocess
import threading
import sys

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, hooks, HookBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
    DatasetMapper,
)
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from shapely.geometry import Polygon

warnings.filterwarnings("ignore", category=shapely.errors.ShapelyDeprecationWarning)

###############################################################################
# Helper function: Make annotation ids unique.
def make_annotation_ids_unique(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    ann_ids = set()
    new_ann_id = 1
    for ann in data.get("annotations", []):
        if ann["id"] in ann_ids:
            ann["id"] = new_ann_id
            new_ann_id += 1
        else:
            ann_ids.add(ann["id"])
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    return

###############################################################################
# Helper function: Remove annotations with empty or invalid bounding boxes.
def remove_empty_bbox_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    new_annotations = []
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        new_annotations.append(ann)
    data["annotations"] = new_annotations
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    return

###############################################################################
# NEW helper: Remove annotations with missing or empty segmentation.
def remove_empty_segmentation_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    new_annotations = []
    for ann in data.get("annotations", []):
        segm = ann.get("segmentation", None)
        # For instance segmentation, expect a nonempty list (or a valid dict)
        if segm is None:
            continue
        if isinstance(segm, list) and len(segm) == 0:
            continue
        new_annotations.append(ann)
    data["annotations"] = new_annotations
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    return

###############################################################################
def train_model(coco_json_path, img_dir, output_dir, total_iters, checkpoint_period,
                dataset_name, keypoint_flag, train_keypoints_only, train_segmentation_only):
    # Set up logging
    os.makedirs(output_dir, exist_ok=True)
    log_filename = f"training_{dataset_name}.log"
    logging.basicConfig(
        filename=os.path.join(output_dir, log_filename),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()

    # Load the COCO JSON data
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract categories and verify
    categories = coco_data.get('categories', [])
    if not categories:
        logger.error("No categories found in COCO JSON file.")
        return

    # Build mapping from category IDs to names.
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}
    logger.info(f"Category IDs and Names: {category_id_to_name}")
    valid_category_ids = set(category_id_to_name.keys())

    # ----- [NEW BLOCK for keypoints-only training] -----
    if train_keypoints_only:
        filtered_annotations = []
        for ann in coco_data.get('annotations', []):
            if 'keypoints' in ann and len(ann['keypoints']) >= 3:
                # If bbox is missing or invalid, compute a bbox from visible keypoints.
                if (not ann.get('bbox')) or (len(ann.get('bbox')) < 4) or (ann['bbox'][2] <= 0) or (ann['bbox'][3] <= 0):
                    kpts = ann['keypoints']
                    vis = kpts[2::3]
                    xs = [x for x, v in zip(kpts[0::3], vis) if v != 0]
                    ys = [y for y, v in zip(kpts[1::3], vis) if v != 0]
                    if xs and ys:
                        min_x = min(xs)
                        min_y = min(ys)
                        max_x = max(xs)
                        max_y = max(ys)
                        ann['bbox'] = [min_x, min_y, max_x - min_x, max_y - min_y]
                        ann['area'] = (max_x - min_x) * (max_y - min_y)
                filtered_annotations.append(ann)
        coco_data['annotations'] = filtered_annotations
        new_kp_cat_id = max(valid_category_ids) + 1
        new_cat = {"id": new_kp_cat_id, "name": "keypoints", "supercategory": "keypoints"}
        coco_data['categories'].append(new_cat)
        for ann in coco_data.get('annotations', []):
            if 'keypoints' in ann and len(ann['keypoints']) >= 3:
                ann['category_id'] = new_kp_cat_id
        modified_json_path = os.path.splitext(coco_json_path)[0] + "_modified.json"
        with open(modified_json_path, "w") as f:
            json.dump(coco_data, f, indent=4)
        coco_json_path = modified_json_path
        logger.info("Modified coco JSON for keypoints-only training.")
    # -----------------------------------------------------

    # Filter out annotations with invalid category IDs.
    annotations = coco_data.get('annotations', [])
    annotation_category_ids = set(ann['category_id'] for ann in annotations)
    invalid_category_ids = annotation_category_ids - valid_category_ids
    if train_keypoints_only:
        invalid_category_ids = invalid_category_ids - {new_kp_cat_id}
    if invalid_category_ids:
        logger.error(f"Annotations contain invalid category_ids: {invalid_category_ids}")
        logger.info("Removing invalid annotations...")
        total_annotations_before = len(annotations)
        annotations = [ann for ann in annotations if ann['category_id'] in valid_category_ids or 
                       (train_keypoints_only and ann['category_id'] == new_kp_cat_id)]
        total_annotations_after = len(annotations)
        logger.info(f"Annotations: before={total_annotations_before}, after={total_annotations_after}")
        coco_data['annotations'] = annotations
        base_name, ext = os.path.splitext(coco_json_path)
        fixed_coco_json_path = base_name + '_fixed_' + ext
        with open(fixed_coco_json_path, 'w') as f:
            json.dump(coco_data, f)
        logger.info(f"Saved fixed COCO JSON file to {fixed_coco_json_path}")
        coco_json_path = fixed_coco_json_path
    else:
        logger.info("No invalid category_ids found in annotations.")

    category_names = [cat['name'] for cat in coco_data['categories']]
    logger.info("Extracted category names: %s", category_names)

    # Determine if polygons and keypoints exist.
    has_polygons = any(
        isinstance(ann.get("segmentation", []), list) and len(ann.get("segmentation", [])) > 0 
        for ann in coco_data.get("annotations", [])
    )
    has_keypoints = any(
        isinstance(ann.get("keypoints", []), list) and len(ann.get("keypoints", [])) > 0 
        for ann in coco_data.get("annotations", [])
    )
    logger.info(f"Detected polygons? {has_polygons}")
    logger.info(f"Detected keypoints? {has_keypoints}")
    if keypoint_flag:
        has_keypoints = True
        logger.info("Forcing keypoint training due to --keypoint flag.")

    # Create train and val directories under img_dir.
    train_dir = os.path.join(img_dir, 'train')
    val_dir = os.path.join(img_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy image files to train and val directories.
    image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    for img_file in image_files:
        if img_file in ['train', 'val']:
            continue
        src_path = os.path.join(img_dir, img_file)
        dst_train = os.path.join(train_dir, img_file)
        dst_val = os.path.join(val_dir, img_file)
        if not os.path.exists(dst_train):
            shutil.copy(src_path, dst_train)
        if not os.path.exists(dst_val):
            shutil.copy(src_path, dst_val)

    # Copy the (possibly modified) COCO JSON file into train and val directories.
    train_json_path = os.path.join(train_dir, 'train.json')
    val_json_path = os.path.join(val_dir, 'val.json')
    shutil.copy(coco_json_path, train_json_path)
    shutil.copy(coco_json_path, val_json_path)

    # For segmentation-only training, also remove annotations with empty segmentation.
    if train_segmentation_only:
        remove_empty_segmentation_annotations(train_json_path)
        remove_empty_segmentation_annotations(val_json_path)

    # Clean up the JSON copies.
    remove_empty_bbox_annotations(train_json_path)
    remove_empty_bbox_annotations(val_json_path)
    make_annotation_ids_unique(train_json_path)
    make_annotation_ids_unique(val_json_path)

    # Dataset Registration
    register_coco_instances(dataset_name + "_train", {}, train_json_path, train_dir)
    register_coco_instances(dataset_name + "_val", {}, val_json_path, val_dir)
    MetadataCatalog.get(dataset_name + "_train").thing_classes = category_names
    MetadataCatalog.get(dataset_name + "_val").thing_classes = category_names

    # --- KEYPOINT METADATA SETUP ---
    if has_keypoints:
        metadata_train = MetadataCatalog.get(dataset_name + "_train")
        if metadata_train.get("keypoint_names") is None:
            kp_names = None
            for cat in categories:
                if "keypoints" in cat and isinstance(cat["keypoints"], list) and len(cat["keypoints"]) > 0:
                    kp_names = cat["keypoints"]
                    break
            if kp_names is None:
                max_keypoints = 0
                for ann in coco_data.get('annotations', []):
                    kpts = ann.get("keypoints", [])
                    num = len(kpts) // 3
                    if num > max_keypoints:
                        max_keypoints = num
                kp_names = [f"kp{i+1}" for i in range(max_keypoints)]
            metadata_train.set(keypoint_names=kp_names, keypoint_flip_map=[])
            MetadataCatalog.get(dataset_name + "_val").set(keypoint_names=kp_names, keypoint_flip_map=[])
            logger.info("Assigned keypoint names: %s", kp_names)
        else:
            logger.info("Using existing keypoint names: %s", metadata_train.get("keypoint_names"))

    print("Registered datasets:", DatasetCatalog.list())

    # Save thing_classes externally.
    metadata = MetadataCatalog.get(dataset_name + "_train")
    thing_classes = metadata.get("thing_classes", None)
    if thing_classes is None:
        logger.error("No 'thing_classes' attribute in metadata.")
        return
    thing_classes_filename = f"thing_classes_{dataset_name}.json"
    with open(os.path.join(output_dir, thing_classes_filename), "w") as f:
        json.dump({"thing_classes": thing_classes}, f, indent=4)
    logger.info(f"Saved thing_classes to {thing_classes_filename}.")

    # --------------------------------------------------
    # Download Keypoint Model if needed.
    keypoint_model_url = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    keypoint_model_path = os.path.join(output_dir, "keypoint_rcnn_R_50_FPN_3x.pth")
    if not os.path.exists(keypoint_model_path):
        logger.info(f"Downloading keypoint model to {keypoint_model_path}...")
        torch.hub.download_url_to_file(keypoint_model_url, keypoint_model_path)
        logger.info("Keypoint model downloaded successfully.")
    else:
        logger.info("Keypoint model already exists. Skipping download.")

    # -------------------------------------------------------------------------
    # BASE CONFIGURATION: Set up model for segmentation and/or keypoint training.
    # -------------------------------------------------------------------------
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    if train_keypoints_only:
        cfg.MODEL.MASK_ON = False
        cfg.MODEL.KEYPOINT_ON = True
    elif train_segmentation_only:
        cfg.MODEL.KEYPOINT_ON = False
        cfg.MODEL.MASK_ON = True
    else:
        cfg.MODEL.KEYPOINT_ON = has_keypoints
        cfg.MODEL.MASK_ON = has_polygons

    if cfg.MODEL.KEYPOINT_ON:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        kp_names = MetadataCatalog.get(dataset_name + "_train").get("keypoint_names", [])
        num_kps = len(kp_names)
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_kps
        # Override TEST.KEYPOINT_OKS_SIGMAS to match the number of keypoints.
        cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.025] * num_kps
    else:
        cfg.MODEL.KEYPOINT_ON = False

    cfg.DATASETS.TRAIN = (dataset_name + "_train",)
    cfg.DATASETS.TEST  = (dataset_name + "_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # -------------------------------------------------------------------------
    # Solver Setup
    # -------------------------------------------------------------------------
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.WARMUP_ITERS = int(0.085 * total_iters)
    cfg.SOLVER.STEPS = (int(0.40 * total_iters), int(0.80 * total_iters))
    cfg.SOLVER.MAX_ITER = int(total_iters)
    cfg.SOLVER.CHECKPOINT_PERIOD = int(checkpoint_period)
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.OUTPUT_DIR = output_dir

    def setup_cfg(cfg):
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        from detectron2.utils.logger import setup_logger
        setup_logger(output=cfg.OUTPUT_DIR)
        return cfg

    cfg_local = setup_cfg(cfg)
    cfg_local.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        cfg_local.SOLVER.IMS_PER_BATCH = 2 * num_gpus
    else:
        cfg_local.SOLVER.IMS_PER_BATCH = 2
    cfg_local.SOLVER.BASE_LR = 0.00025 * cfg_local.SOLVER.IMS_PER_BATCH
    cfg_local.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS
    cfg_local.SOLVER.MAX_ITER = cfg.SOLVER.MAX_ITER
    cfg_local.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
    cfg_local.MODEL.ROI_HEADS.NUM_CLASSES = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    cfg_local.TEST.EVAL_PERIOD = cfg.TEST.EVAL_PERIOD

    config_filename = f"config_{dataset_name}.yaml"
    with open(os.path.join(cfg_local.OUTPUT_DIR, config_filename), "w") as f:
        f.write(cfg_local.dump())
    logger.info(f"Saved configuration to {config_filename}.")

    # -------------------------------------------------------------------------
    # Training Hooks
    # -------------------------------------------------------------------------
    class EarlyStoppingHook(HookBase):
        def __init__(self, patience, metric_name):
            self.patience = patience
            self.metric_name = metric_name
            self.best_metric = None
            self.counter = 0
        def after_step(self):
            eval_period = self.trainer.cfg.TEST.EVAL_PERIOD
            if eval_period > 0 and self.trainer.iter > 0 and self.trainer.iter % eval_period == 0:
                storage = self.trainer.storage
                latest_metrics = {k: v.median(20) for k, v in storage.histories().items()}
                metric_value = latest_metrics.get(self.metric_name, None)
                if metric_value is None:
                    return
                if self.best_metric is None or metric_value > self.best_metric:
                    self.best_metric = metric_value
                    self.counter = 0
                else:
                    self.counter += 1
                if self.counter >= self.patience:
                    logger.info(f"Early stopping triggered at iteration {self.trainer.iter} with best {self.metric_name}: {self.best_metric}")
                    self.trainer.storage.put_scalars(early_stop=self.trainer.iter)
                    raise StopIteration

    class AugmentedTrainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, cfg):
            augs = [
                T.ResizeShortestEdge(
                    short_edge_length=(640, 672, 704, 736, 768, 800),
                    max_size=1333,
                    sample_style='choice'
                ),
                T.RandomRotation(angle=[-45, 45]),
                T.RandomBrightness(0.8, 1.2),
                T.RandomContrast(0.8, 1.2),
                T.RandomSaturation(0.8, 1.2)
            ]
            return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs))
        @classmethod
        def build_evaluator(cls, cfg, dataset_name):
            return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
        def build_hooks(self):
            hooks_list = super().build_hooks()
            hooks_list.insert(-1, hooks.EvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                lambda: self.test(self.cfg, self.model, self.build_evaluator(self.cfg, self.cfg.DATASETS.TEST[0]))
            ))
            hooks_list.insert(-1, EarlyStoppingHook(patience=5, metric_name="bbox/AP50"))
            return hooks_list

    class LossPlotHook(HookBase):
        def __init__(self, dataset_name):
            super().__init__()
            self.losses = []
            self.dataset_name = dataset_name
        def after_step(self):
            if self.trainer.iter % 20 == 0 and self.trainer.iter > 0:
                total_loss = self.trainer.storage.history("total_loss").latest()
                self.losses.append(total_loss)
                plt.figure(figsize=(10, 5))
                plt.plot(np.arange(len(self.losses)) * 20, self.losses, label='Total Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Loss over Iterations')
                plt.legend()
                plt.grid()
                loss_curve_filename = f"loss_curve_{self.dataset_name}.png"
                plt.savefig(os.path.join(self.trainer.cfg.OUTPUT_DIR, loss_curve_filename))
                plt.close()

    class EvaluationMetricsPlotHook(HookBase):
        def __init__(self, dataset_name):
            super().__init__()
            self.iterations = []
            self.ap_values = []
            self.ap50_values = []
            self.dataset_name = dataset_name
        def after_step(self):
            eval_period = self.trainer.cfg.TEST.EVAL_PERIOD
            if eval_period > 0 and self.trainer.iter > 0 and self.trainer.iter % eval_period == 0:
                storage = self.trainer.storage
                histories = storage.histories()
                ap_history = histories.get('bbox/AP', None)
                ap50_history = histories.get('bbox/AP50', None)
                if ap_history is not None and ap50_history is not None:
                    ap = ap_history.median(20)
                    ap50 = ap50_history.median(20)
                    self.iterations.append(self.trainer.iter)
                    self.ap_values.append(ap)
                    self.ap50_values.append(ap50)
                    plt.figure(figsize=(10, 5))
                    plt.plot(self.iterations, self.ap_values, label='AP')
                    plt.plot(self.iterations, self.ap50_values, label='AP50')
                    plt.xlabel('Iteration')
                    plt.ylabel('Score')
                    plt.title('AP and AP50 over Iterations')
                    plt.legend()
                    plt.grid()
                    if not all(np.isnan(self.ap_values)):
                        ap_scores_filename = f"ap_scores_{self.dataset_name}.png"
                        plt.savefig(os.path.join(self.trainer.cfg.OUTPUT_DIR, ap_scores_filename))
                        logger.info(f"Saved AP scores plot to {ap_scores_filename}.")
                    else:
                        logger.warning("AP scores contain only NaN values. Plot not saved.")
                    plt.close()

    trainer = AugmentedTrainer(cfg_local)
    trainer.register_hooks([LossPlotHook(dataset_name), EvaluationMetricsPlotHook(dataset_name)])
    trainer.resume_or_load(resume=False)

    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training interrupted: {e}")

    evaluator = COCOEvaluator(dataset_name + "_val", cfg_local, False, output_dir=cfg_local.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg_local, dataset_name + "_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    logger.info("Results: %s", results)

    model_final_filename = f"model_final_{dataset_name}.pth"
    torch.save(trainer.model.state_dict(), os.path.join(cfg_local.OUTPUT_DIR, model_final_filename))
    logger.info(f"Saved trained model to {model_final_filename}.")

    results_plain_filename = f"results_plain_{dataset_name}.json"
    with open(os.path.join(cfg_local.OUTPUT_DIR, results_plain_filename), "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved as {results_plain_filename}.")

    try:
        if 'bbox' in results and 'segm' in results:
            bbox_results = results['bbox']
            segm_results = results['segm']
            results_dict = {
                "AP (bbox)": bbox_results['AP'],
                "AP50 (bbox)": bbox_results['AP50'],
                "AP75 (bbox)": bbox_results['AP75'],
                "APs (bbox)": bbox_results['APs'],
                "APm (bbox)": bbox_results['APm'],
                "APl (bbox)": bbox_results['APl'],
                "AP (segm)": segm_results['AP'],
                "AP50 (segm)": segm_results['AP50'],
                "AP75 (segm)": segm_results['AP75'],
                "APs (segm)": segm_results['APs'],
                "APm (segm)": segm_results['APm'],
                "APl (segm)": segm_results['APl'],
            }
            df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])
            logger.info("Evaluation Results:\n%s", df)
        else:
            logger.info("Missing bbox or segm in results.")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detectron2 Training Script for Segmentation and Keypoints")
    parser.add_argument('--coco-json', type=str, required=True, help='Path to custom COCO JSON file')
    parser.add_argument('--img-dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--total-iters', type=int, default=35000, help='Total number of iterations')
    parser.add_argument('--checkpoint-period', type=int, default=1000, help='Checkpoint period')
    parser.add_argument('--dataset-name', type=str, default='your_taxon', help='Name of the dataset')
    parser.add_argument('--keypoint', action='store_true', help='Force keypoint training if present.')
    parser.add_argument('--train-keypoints-only', action='store_true', help='Train keypoints only (disable segmentation).')
    parser.add_argument('--train-segmentation-only', action='store_true', help='Train segmentation only (disable keypoint training).')

    args = parser.parse_args()

    train_model(
        args.coco_json,
        args.img_dir,
        args.output_dir,
        args.total_iters,
        args.checkpoint_period,
        args.dataset_name,
        keypoint_flag=args.keypoint,
        train_keypoints_only=args.train_keypoints_only,
        train_segmentation_only=args.train_segmentation_only
    )
