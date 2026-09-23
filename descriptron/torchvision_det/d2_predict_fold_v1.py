#!/usr/bin/env python3
"""
d2_predict_fold_v1.py — Detectron2 predictions in the same format as tv_predict
===============================================================================

Runs in `detectron2_env`. Its only job is to make the Detectron2 arm of the
backend comparison scoreable by exactly the same evaluator as the torchvision
arm: it writes COCO *detections* (image_id, category_id, RLE segmentation, score)
and nothing else. If each arm were scored by its own evaluator the comparison
would measure the evaluators.

It takes a JSON list of image ids rather than a directory, because the held-out
fold is a subset of one folder and copying hundreds of TIFFs per fold, or
symlinking them, is the kind of thing that later turns into a path bug.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--model_weights", required=True)
    p.add_argument("--coco_json", required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--image_ids", required=True, help="JSON list of COCO image ids")
    p.add_argument("--detections_out", required=True)
    p.add_argument("--score_threshold", type=float, default=0.05)
    p.add_argument("--thing_classes", required=True,
                   help="JSON list of category names in the contiguous order used for training")
    a = p.parse_args()

    import cv2
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from pycocotools import mask as mask_utils

    cfg = get_cfg()
    cfg.merge_from_file(a.config_file)
    cfg.MODEL.WEIGHTS = a.model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = a.score_threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)

    names = json.loads(Path(a.thing_classes).read_text()
                       if Path(a.thing_classes).exists() else a.thing_classes)
    coco = json.loads(Path(a.coco_json).read_text())
    by_id = {im["id"]: im for im in coco["images"]}
    name_to_coco = {c["name"]: c["id"] for c in coco["categories"]}
    wanted = json.loads(Path(a.image_ids).read_text())

    detections = []
    for image_id in wanted:
        meta = by_id.get(image_id)
        if meta is None:
            continue
        path = Path(a.img_dir) / Path(meta["file_name"]).name
        if not path.exists():
            path = Path(a.img_dir) / meta["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            print(f"  ! unreadable: {path}")
            continue
        inst = predictor(img)["instances"].to("cpu")
        scores = inst.scores.numpy() if inst.has("scores") else np.zeros(0)
        classes = inst.pred_classes.numpy() if inst.has("pred_classes") else np.zeros(0, int)
        boxes = inst.pred_boxes.tensor.numpy() if inst.has("pred_boxes") else np.zeros((0, 4))
        masks = inst.pred_masks.numpy() if inst.has("pred_masks") else None
        for i in range(len(scores)):
            if scores[i] < a.score_threshold:
                continue
            name = names[int(classes[i])] if int(classes[i]) < len(names) else str(classes[i])
            x0, y0, x1, y1 = [float(v) for v in boxes[i]]
            det = {"image_id": int(image_id),
                   "category_id": int(name_to_coco.get(name, int(classes[i]))),
                   "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(scores[i])}
            if masks is not None:
                binary = masks[i].astype(np.uint8)
                if binary.sum() == 0:
                    continue
                rle = mask_utils.encode(np.asfortranarray(binary))
                rle["counts"] = rle["counts"].decode("ascii")
                det["segmentation"] = rle
            detections.append(det)

    Path(a.detections_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.detections_out).write_text(json.dumps(detections))
    print(f"[d2_predict] {len(detections)} detections over {len(wanted)} images "
          f"-> {a.detections_out}")


if __name__ == "__main__":
    main()
