import configparser
import numpy as np
import os, json, cv2, random
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch, hooks, HookBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetMapper
import shapely
import warnings
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=shapely.errors.ShapelyDeprecationWarning)

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the paths
train_data = config['Paths']['train_data']
val_data = config['Paths']['val_data']
test_data = config['Paths']['test_data']
model_output = config['Paths']['model_output']
plot_output = config['Paths']['plot_output']
train_json_file_path = config['Paths']['train_json_file_path']  # Handcrafted polygons JSON
json_input = config['Paths']['json_input']
json_output = config['Paths']['json_output']
coco_json_path = config['Paths']['coco_json_path']  # COCO format JSON path
log_file_path = config['Paths']['log_file_path']  # Log file from training
metadata_json_path = config['Paths']['metadata_json_path']  # Metadata JSON from training

# Load JSON data
with open(train_json_file_path) as f:
    data = json.load(f)

# Extract unique sclerite types
sclerites = set()
for _, file_data in data.items():
    for region in file_data['regions']:
        sclerites.add(region['region_attributes']['sclerite'])

sclerites = list(sclerites)
print("Sclerites:", sclerites)

# Convert to COCO format
def convert_to_coco(data, img_dir, sclerites):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    for i, sclerite in enumerate(sclerites):
        coco_format["categories"].append({
            "id": i + 1,  # Ensure category IDs start from 1
            "name": sclerite,
            "supercategory": "sclerite",
        })

    annotation_id = 0
    for idx, (file_name, file_data) in enumerate(data.items()):
        image_id = idx
        filename = os.path.join(img_dir, file_data["filename"])
        height, width = cv2.imread(filename).shape[:2]

        coco_format["images"].append({
            "id": image_id,
            "file_name": file_data["filename"],
            "height": height,
            "width": width,
        })

        for region in file_data["regions"]:
            anno = region["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": sclerites.index(region["region_attributes"]["sclerite"]) + 1,  # Ensure category IDs start from 1
                "segmentation": [poly],
                "bbox": [min(px), min(py), max(px)-min(px), max(py)-min(py)],
                "bbox_mode": BoxMode.XYWH_ABS,
                "area": (max(px)-min(px)) * (max(py)-min(py)),
                "iscrowd": 0,
            })
            annotation_id += 1

    return coco_format

coco_data = convert_to_coco(data, train_data, sclerites)

# Save COCO format JSON
with open(coco_json_path, 'w') as f:
    json.dump(coco_data, f)

# Register the dataset in COCO format
register_coco_instances("coleoptera_train", {}, coco_json_path, train_data)
register_coco_instances("coleoptera_val", {}, coco_json_path, val_data)
coleoptera_metadata = MetadataCatalog.get("coleoptera_train")

# Configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("coleoptera_train",)
cfg.DATASETS.TEST = ("coleoptera_val",)
cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
cfg.MODEL.RESNETS.DEPTH = 50
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001  # Reduced learning rate
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
cfg.SOLVER.WARMUP_ITERS = 3000
cfg.SOLVER.STEPS = (14000, 28000)
cfg.SOLVER.MAX_ITER = 35000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.TEST.EVAL_PERIOD = 1000  # Ensure this is set to a non-zero value

cfg.SOLVER.WEIGHT_DECAY = 0.0001  # Add weight decay
cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True  # Enable gradient clipping
cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Default value. Adjust based on memory and accuracy needs.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(sclerites)

# Check for GPU availability and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.DEVICE = device
print(f"Using device: {device}")

def setup(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

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
                print(f"Early stopping triggered at iteration {self.trainer.iter} with best {self.metric_name}: {self.best_metric}")
                self.trainer.storage.put_scalars(early_stop=self.trainer.iter)
                raise StopIteration

class AugmentedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        augs = [
            T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'),
            T.RandomRotation(angle=[-45, 45]),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2)
        ]
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs))
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=model_output)
    
    def build_hooks(self):
        hooks_list = super().build_hooks()
        hooks_list.insert(-1, hooks.EvalHook(
            0, lambda: self.test(self.cfg, self.model, self.build_evaluator(self.cfg, self.cfg.DATASETS.TEST[0]))
        ))
        hooks_list.insert(-1, EarlyStoppingHook(patience=5, metric_name="bbox/AP50"))  # Use appropriate metric name
        return hooks_list

class LossPlotHook(HookBase):
    def __init__(self):
        super().__init__()
        self.losses = []

    def after_step(self):
        if self.trainer.iter % 20 == 0 and self.trainer.iter > 0:
            total_loss = self.trainer.storage.history("total_loss").values()[-1][0]
            self.losses.append(total_loss)
            
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(len(self.losses)) * 20, self.losses, label='Total Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Loss over Iterations')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(self.trainer.cfg.OUTPUT_DIR, 'loss_curve.png'))
            plt.close()

def main(args):
    cfg = setup(get_cfg())
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("coleoptera_train",)
    cfg.DATASETS.TEST = ("coleoptera_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        cfg.SOLVER.IMS_PER_BATCH = 2 * num_gpus
    else:
        cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025 * cfg.SOLVER.IMS_PER_BATCH

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.MAX_ITER = 35000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(sclerites)

    cfg.TEST.EVAL_PERIOD = 1000

    trainer = AugmentedTrainer(cfg)
    trainer.register_hooks([LossPlotHook()])
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("coleoptera_val", cfg, False, output_dir=model_output)
    val_loader = build_detection_test_loader(cfg, "coleoptera_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print("Results:", results)

    torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "model_final_big35kv4_rn50.pth"))

    with open(os.path.join(cfg.OUTPUT_DIR, "config_big35kv4_rn50.yaml"), "w") as f:
        f.write(cfg.dump())

    with open(metadata_json_path, "w") as f:  # Save metadata JSON
        json.dump({"thing_classes": sclerites}, f)

    try:
        if 'bbox' in results and 'segm' in results:
            print("Results bbox:", results['bbox'])
            print("Results segm:", results['segm'])

            bbox_results = results['bbox']
            segm_results = results['segm']

            if isinstance(bbox_results, dict) and 'bbox' in bbox_results:
                bbox_stats = bbox_results['bbox']['AP'] if 'bbox' in bbox_results else [0]*24
            else:
                bbox_stats = [0]*24

            if isinstance(segm_results, dict) and 'segm' in segm_results:
                segm_stats = segm_results['segm']['AP'] if 'segm' in segm_results else [0]*24
            else:
                segm_stats = [0]*24

            results_list = bbox_stats + segm_stats

            results_dict = {
                "AP (bbox)": results_list[0],
                "AP50 (bbox)": results_list[1],
                "AP75 (bbox)": results_list[2],
                "APs (bbox)": results_list[3],
                "APm (bbox)": results_list[4],
                "APl (bbox)": results_list[5],
                "AR (bbox)": results_list[6],
                "AR50 (bbox)": results_list[7],
                "AR75 (bbox)": results_list[8],
                "ARs (bbox)": results_list[9],
                "ARm (bbox)": results_list[10],
                "ARl (bbox)": results_list[11],
                "AP (segm)": results_list[12],
                "AP50 (segm)": results_list[13],
                "AP75 (segm)": results_list[14],
                "APs (segm)": results_list[15],
                "APm (segm)": results_list[16],
                "APl (segm)": results_list[17],
                "AR (segm)": results_list[18],
                "AR50 (segm)": results_list[19],
                "AR75 (segm)": results_list[20],
                "ARs (segm)": results_list[21],
                "ARm (segm)": results_list[22],
                "ARl (segm)": results_list[23],
            }
            df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])
            print(df)
        else:
            print("Missing bbox or segm in results.")
    except Exception as e:
        print(f"Error during evaluation: {e}")

    with open(os.path.join(cfg.OUTPUT_DIR, "results_plain_big35kv4_rn50.json"), "w") as f:
        json.dump(results, f)
    print("Results saved as plain JSON.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="number of gpus to use (default: use all available GPUs)"
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=1,
        help="number of machines (default=1)"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (default=0)"
    )
    args = parser.parse_args()

    num_gpus = args.num_gpus
    if num_gpus > 0:
        launch(
            main,
            num_gpus_per_machine=num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url="auto",
            args=(args,),
        )
    else:
        main(args)
