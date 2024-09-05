import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import os
import numpy as np
import json
import cv2
import random
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#### from google.colab.patches import cv2_imshow
print(torch.__version__, torch.cuda.is_available())
setup_logger()

def get_dicts(img_dir):
    json_file = os.path.join(img_dir, "elytra_polys_and_keyponts_via_project_7Jul2023_16h41m_28_json.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
        
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        keypoints = []
        for anno in annos:
            shape_attr = anno["shape_attributes"]
            
            if shape_attr["name"] == "point":
                # process keypoint
                keypoint = [shape_attr["cx"], shape_attr["cy"], 2]  # (x, y, visibility)
                keypoints.append(keypoint)
                
            elif shape_attr["name"] == "polygon":
                # process polygon
                px = shape_attr["all_points_x"]
                py = shape_attr["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
        
        # Add keypoints to each object
        for obj in objs:
            keypoints = [kp for sublist in keypoints for kp in sublist]  # flatten keypoints list
            obj["keypoints"] = keypoints

        record["annotations"] = objs
        dataset_dicts.append(record)
        
    return dataset_dicts 

from detectron2.data import DatasetCatalog, MetadataCatalog
path = "/Users/alexvandam/Mask_RCNN/Dataset/Keypoints/" # path to your image folder
for d in ["train", "val"]:
    DatasetCatalog.register("elytra_" + d, lambda d=d: get_dicts(path + "" +  d))
    MetadataCatalog.get("elytra_" + d).set(thing_classes=["elytra"])
    MetadataCatalog.get("elytra_" + d).set(thing_classes=["elytra"], keypoint_names=["point1", "point2", "point3"])
        # Here is where you should add your keypoint flip map.
    # Ensure your keypoints are paired properly if they are identifiable after flipping. 
    # If not, use an empty list.
    MetadataCatalog.get("elytra_" + d).set(keypoint_flip_map=[("point1", "point2"), ("point1", "point3"), ("point2", "point3") ])
dataset_dicts = get_dicts(path + "/train/")
for d in random.sample(dataset_dicts, 28):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("elytra_train"), scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    #cv2.imshow("preview", vis.get_image()[:, :, ::-1])
    #cv2.waitKey()
cfg = get_cfg()
cfg.OUTPUT_DIR = "/Users/alexvandam/Mask_RCNN/Dataset/Keypoints/"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (elytra)
cfg.MODEL.ROI_HEADS.NUM_KEYPOINTS = 3  # number of keypoints

cfg.DATASETS.TRAIN = ("elytra_train",)     # our training dataset
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0     # number of parallel data loading workers
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # use pretrained weights
cfg.SOLVER.IMS_PER_BATCH = 2     # in 1 iteration the model sees 2 images
cfg.SOLVER.BASE_LR = 0.00025     # learning rate
cfg.SOLVER.MAX_ITER = 2000        # number of iteration
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128     # number of proposals to sample for training
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (elytra)
cfg.SOLVER.STEPS = [] 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.1, 0.1, 0.1]  # adjust these values as necessary for your keypoints

# use GPU if available, else use CPU
if torch.cuda.is_available():
    cfg.MODEL.DEVICE = 'cuda'
else:
    cfg.MODEL.DEVICE = 'cpu'
# 
   
#cfg.MODEL.DEVICE='device'
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
### new stuff
os.makedirs("/Users/alexvandam/Mask_RCNN/Dataset/Keypoints/Updated_yml", exist_ok=True)
os.makedirs("/Users/alexvandam/Mask_RCNN/Dataset/Keypoints/Updated_pth", exist_ok=True)


config_str = cfg.dump()
with open("/Users/alexvandam/Mask_RCNN/Dataset/Keypoints/Updated_yml/elytra_keypoints_new_config.yaml", "w") as f:
    f.write(config_str)
####

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set a custom testing threshold

predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_dicts("/Users/alexvandam/Mask_RCNN/Dataset/Keypoints/val")
for d in random.sample(dataset_dicts, 28):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get("elytra_val"), 
                   scale=0.5 #, 
                   #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow("preview", vis.get_image()[:, :, ::-1])
    cv2.imshow("preview", out.get_image()[:, :, ::-1])
    cv2.waitKey()
    