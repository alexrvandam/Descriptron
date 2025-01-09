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


def train_model(coco_json_path, img_dir, output_dir, total_iters, checkpoint_period, dataset_name):
    # Set up logging
    os.makedirs(output_dir, exist_ok=True)
    log_filename = f"training_{dataset_name}.log"  # Modified: Added dataset_name suffix
    logging.basicConfig(
        filename=os.path.join(output_dir, log_filename),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()

    # Load COCO JSON data
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract categories
    categories = coco_data.get('categories', [])
    if not categories:
        logger.error("No categories found in COCO JSON file.")
        return

    # Build category_id to name mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}
    logger.info(f"Category IDs and Names: {category_id_to_name}")

    # Collect valid category IDs
    valid_category_ids = set(category_id_to_name.keys())

    # Collect annotations
    annotations = coco_data.get('annotations', [])
    annotation_category_ids = set(ann['category_id'] for ann in annotations)

    # Find invalid category_ids
    invalid_category_ids = annotation_category_ids - valid_category_ids

    if invalid_category_ids:
        logger.error(f"Annotations contain invalid category_ids: {invalid_category_ids}")
        logger.info("Proceeding to fix the COCO JSON file by removing invalid annotations.")

        # Remove annotations with invalid category_ids
        total_annotations_before = len(annotations)
        annotations = [ann for ann in annotations if ann['category_id'] in valid_category_ids]
        total_annotations_after = len(annotations)

        logger.info(f"Total annotations before fixing: {total_annotations_before}")
        logger.info(f"Total annotations after fixing: {total_annotations_after}")

        # Update the coco_data with fixed annotations
        coco_data['annotations'] = annotations

        # Save the fixed JSON file with '_fixed_' appended
        base_name, ext = os.path.splitext(coco_json_path)
        fixed_coco_json_path = base_name + '_fixed_' + ext
        with open(fixed_coco_json_path, 'w') as f:
            json.dump(coco_data, f)
        logger.info(f"Saved fixed COCO JSON file to {fixed_coco_json_path}")

        # Update coco_json_path to use the fixed file
        coco_json_path = fixed_coco_json_path
    else:
        logger.info("No invalid category_ids found in annotations.")

    # Extract category names
    category_names = [cat['name'] for cat in categories]
    logger.info("Category names extracted from COCO JSON: %s", category_names)

    # Create 'train' and 'val' directories inside img_dir
    train_dir = os.path.join(img_dir, 'train')
    val_dir = os.path.join(img_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy all images into 'train' and 'val' directories
    image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    for img_file in image_files:
        src_path = os.path.join(img_dir, img_file)
        # Avoid copying images from 'train' and 'val' directories
        if img_file in ['train', 'val']:
            continue
        dst_train = os.path.join(train_dir, img_file)
        dst_val = os.path.join(val_dir, img_file)
        if not os.path.exists(dst_train):
            shutil.copy(src_path, dst_train)
        if not os.path.exists(dst_val):
            shutil.copy(src_path, dst_val)

    # Copy COCO JSON annotation file into both directories as 'train.json' and 'val.json'
    train_json_path = os.path.join(train_dir, 'train.json')
    val_json_path = os.path.join(val_dir, 'val.json')
    shutil.copy(coco_json_path, train_json_path)
    shutil.copy(coco_json_path, val_json_path)

    # Use the provided dataset_name
    register_coco_instances(dataset_name + "_train", {}, train_json_path, train_dir)
    register_coco_instances(dataset_name + "_val", {}, val_json_path, val_dir)

    # Manually set thing_classes
    MetadataCatalog.get(dataset_name + "_train").thing_classes = category_names
    MetadataCatalog.get(dataset_name + "_val").thing_classes = category_names

    # Access metadata and verify
    metadata = MetadataCatalog.get(dataset_name + "_train")
    thing_classes = metadata.get("thing_classes", None)

    if thing_classes is not None:
        logger.info("Category indices and names:")
        for idx, name in enumerate(thing_classes):
            logger.info("%d: %s", idx, name)
    else:
        logger.error("No 'thing_classes' attribute found in metadata.")
        return

    # Save thing_classes to a JSON file
    #thing_classes_filename = f"thing_classes_{dataset_name}.json"  # Modified
    #with open(os.path.join(output_dir, thing_classes_filename), "w") as f:
    #    json.dump(thing_classes, f)
    #logger.info(f"Saved thing_classes to {thing_classes_filename}.")
    # Save thing_classes to a JSON file with correct formatting
    thing_classes_filename = f"thing_classes_{dataset_name}.json"  # Modified
    with open(os.path.join(output_dir, thing_classes_filename), "w") as f:
        json.dump({"thing_classes": thing_classes}, f, indent=4)
    logger.info(f"Saved thing_classes to {thing_classes_filename}.")

    # Configuration
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = (dataset_name + "_train",)
    cfg.DATASETS.TEST = (dataset_name + "_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001  # Reduced learning rate
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.WARMUP_ITERS = 3000
    cfg.SOLVER.STEPS = (14000, 28000)
    cfg.SOLVER.MAX_ITER = int(total_iters)  # Use total_iters from arguments
    cfg.SOLVER.CHECKPOINT_PERIOD = int(checkpoint_period)  # Use checkpoint_period from arguments
    cfg.TEST.EVAL_PERIOD = 1000  # Ensure this is set to a non-zero value

    cfg.SOLVER.WEIGHT_DECAY = 0.0001  # Add weight decay
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True  # Enable gradient clipping
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Adjust based on memory and accuracy needs
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)  # Set NUM_CLASSES based on thing_classes

    cfg.OUTPUT_DIR = output_dir

    # Check for GPU availability and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    logger.info(f"Using device: {device}")

    def setup(cfg):
        """
        Perform some basic common setups at the beginning of a job.
        """
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        from detectron2.utils.logger import setup_logger
        setup_logger(output=cfg.OUTPUT_DIR)
        return cfg

    cfg_local = setup(cfg)
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

    # Save the config at the beginning
    config_filename = f"config_{dataset_name}.yaml"  # Modified
    with open(os.path.join(cfg_local.OUTPUT_DIR, config_filename), "w") as f:
        f.write(cfg_local.dump())
    logger.info(f"Saved {config_filename}.")

    class EarlyStoppingHook(HookBase):
        def __init__(self, patience, metric_name):
            self.patience = patience
            self.metric_name = metric_name
            self.best_metric = None
            self.counter = 0

        def after_step(self):
            eval_period = self.trainer.cfg.TEST.EVAL_PERIOD
            if (
                eval_period > 0
                and self.trainer.iter > 0
                and self.trainer.iter % eval_period == 0
            ):
                # Access the latest metrics
                storage = self.trainer.storage
                latest_metrics = {
                    k: v.median(20) for k, v in storage.histories().items()
                }

                metric_value = latest_metrics.get(self.metric_name, None)
                if metric_value is None:
                    return

                if self.best_metric is None or metric_value > self.best_metric:
                    self.best_metric = metric_value
                    self.counter = 0
                else:
                    self.counter += 1

                if self.counter >= self.patience:
                    logger.info(
                        f"Early stopping triggered at iteration {self.trainer.iter} with best {self.metric_name}: {self.best_metric}"
                    )
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
            return build_detection_train_loader(
                cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs)
            )

        @classmethod
        def build_evaluator(cls, cfg, dataset_name):
            return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

        def build_hooks(self):
            hooks_list = super().build_hooks()
            hooks_list.insert(
                -1,
                hooks.EvalHook(
                    self.cfg.TEST.EVAL_PERIOD,
                    lambda: self.test(
                        self.cfg,
                        self.model,
                        self.build_evaluator(self.cfg, self.cfg.DATASETS.TEST[0]),
                    ),
                ),
            )
            hooks_list.insert(
                -1, EarlyStoppingHook(patience=5, metric_name="bbox/AP50")
            )
            return hooks_list

    # Custom hook for plotting loss curves
    class LossPlotHook(HookBase):
        def __init__(self, dataset_name):
            super().__init__()
            self.losses = []
            self.dataset_name = dataset_name  # Added dataset_name

        def after_step(self):
            if self.trainer.iter % 20 == 0 and self.trainer.iter > 0:
                # Access the most recent total loss value
                total_loss = self.trainer.storage.history("total_loss").latest()
                self.losses.append(total_loss)

                plt.figure(figsize=(10, 5))
                plt.plot(
                    np.arange(len(self.losses)) * 20,
                    self.losses,
                    label='Total Loss'
                )
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Training Loss over Iterations')
                plt.legend()
                plt.grid()
                loss_curve_filename = f"loss_curve_{self.dataset_name}.png"  # Modified
                plt.savefig(
                    os.path.join(self.trainer.cfg.OUTPUT_DIR, loss_curve_filename)
                )
                plt.close()

    # Custom hook for plotting evaluation metrics
    class EvaluationMetricsPlotHook(HookBase):
        def __init__(self, dataset_name):
            super().__init__()
            self.iterations = []
            self.ap_values = []
            self.ap50_values = []
            self.dataset_name = dataset_name  # Added dataset_name

        def after_step(self):
            eval_period = self.trainer.cfg.TEST.EVAL_PERIOD
            if (
                eval_period > 0
                and self.trainer.iter > 0
                and self.trainer.iter % eval_period == 0
            ):
                storage = self.trainer.storage
                histories = storage.histories()
                ap_history = histories.get('bbox/AP', None)
                ap50_history = histories.get('bbox/AP50', None)
                
                if ap_history is not None and ap50_history is not None:
                    # Get the most recent values
                    ap = ap_history.median(20)
                    ap50 = ap50_history.median(20)
                    self.iterations.append(self.trainer.iter)
                    self.ap_values.append(ap)
                    self.ap50_values.append(ap50)

                    # Plot the AP and AP50 scores
                    plt.figure(figsize=(10, 5))
                    plt.plot(self.iterations, self.ap_values, label='AP')
                    plt.plot(self.iterations, self.ap50_values, label='AP50')
                    plt.xlabel('Iteration')
                    plt.ylabel('Score')
                    plt.title('AP and AP50 over Iterations')
                    plt.legend()
                    plt.grid()
                    
                    # Handle NaN values by not plotting them
                    if not all(np.isnan(self.ap_values)):
                        ap_scores_filename = f"ap_scores_{self.dataset_name}.png"  # Modified
                        plt.savefig(
                            os.path.join(self.trainer.cfg.OUTPUT_DIR, ap_scores_filename)
                        )
                        logger.info(f"Saved AP scores plot to {ap_scores_filename}.")
                    else:
                        logger.warning("AP scores contain only NaN values. Plot not saved.")
                    plt.close()


    # Initialize the trainer
    trainer = AugmentedTrainer(cfg_local)
    trainer.register_hooks([LossPlotHook(dataset_name), EvaluationMetricsPlotHook(dataset_name)])
    trainer.resume_or_load(resume=False)

    try:
        # Run the training
        trainer.train()
    except Exception as e:
        logger.error(f"Training interrupted: {e}")
    finally:
        pass  # Add any cleanup code here if necessary

    # Evaluate the model
    evaluator = COCOEvaluator(
        dataset_name + "_val", cfg_local, False, output_dir=cfg_local.OUTPUT_DIR
    )
    val_loader = build_detection_test_loader(cfg_local, dataset_name + "_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    logger.info("Results: %s", results)

    # Save the trained model
    model_final_filename = f"model_final_{dataset_name}.pth"  # Modified
    torch.save(
        trainer.model.state_dict(), os.path.join(cfg_local.OUTPUT_DIR, model_final_filename)
    )
    logger.info(f"Saved trained model to {model_final_filename}.")

    # Save the evaluation results
    results_plain_filename = f"results_plain_{dataset_name}.json"  # Modified
    with open(os.path.join(cfg_local.OUTPUT_DIR, results_plain_filename), "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved as {results_plain_filename}.")

    # Process and print evaluation results
    try:
        if 'bbox' in results and 'segm' in results:
            logger.info("Results bbox: %s", results['bbox'])
            logger.info("Results segm: %s", results['segm'])

            bbox_results = results['bbox']
            segm_results = results['segm']

            # Convert the results to a DataFrame for better readability
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

    parser = argparse.ArgumentParser(description="Detectron2 Training Script")
    parser.add_argument('--coco-json', type=str, required=True, help='Path to COCO JSON file')
    parser.add_argument('--img-dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--total-iters', type=int, default=35000, help='Total number of iterations')
    parser.add_argument('--checkpoint-period', type=int, default=1000, help='Checkpoint period')
    parser.add_argument('--dataset-name', type=str, default='your_taxon', help='Name of the dataset')  # Added argument
    args = parser.parse_args()

    train_model(
        args.coco_json,
        args.img_dir,
        args.output_dir,
        args.total_iters,
        args.checkpoint_period,
        args.dataset_name  # Pass the dataset name
    )
