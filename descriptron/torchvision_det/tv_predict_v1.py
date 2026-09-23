#!/usr/bin/env python3
"""
tv_predict_v1.py — predict with a fine-tuned torchvision detector, and score it
==============================================================================

The counterpart to `tv_train_v1.py`, replacing
`detectron2/detectron2_predict_and_filterV10_and_kptsV2.py`. It writes two things:

* **Descriptron annotations** (`--annotations_out`): a COCO file with polygon
  segmentations or keypoints, ready to load in the GUI and correct by hand.
* **COCO detections** (`--detections_out`): the scored results format that
  `pycocotools` evaluates, which is what the backend comparison needs.

With `--eval_against <ground truth COCO>` it runs COCOeval and prints AP, AP50 and
**AP75**. AP75 is the one to watch here: it is the strict-overlap score, so it is
where a coarse mask boundary shows up, and the boundary is what the measurements
are made from.

Polygons come from `cv2.findContours` rather than shapely, so this runs in the
`samm` environment with no extra dependency, and the area is taken from the mask
itself rather than from the polygon approximation — slightly more accurate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tv_coco_dataset import CocoInstanceDataset, collate_fn   # noqa: E402
from tv_train_v1 import build_model                            # noqa: E402


def mask_to_polygons(mask, min_points=3, epsilon=1.0):
    """Binary mask -> COCO polygon list. Small fragments are dropped, not merged."""
    import cv2
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if epsilon:
            c = cv2.approxPolyDP(c, epsilon, True)
        c = c.reshape(-1, 2)
        if c.shape[0] >= min_points:
            polys.append([float(v) for v in c.reshape(-1)])
    return polys


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = ck["meta"]
    # checkpoints written before --arch existed are v2, which was the only option then
    model = build_model(meta["task"], meta["num_classes"], meta.get("num_keypoints", 0),
                        pretrained=False, arch=meta.get("arch", "v2"))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, meta


@torch.no_grad()
def predict(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    model, meta = load_checkpoint(args.checkpoint, device)
    task = meta["task"]
    contig_to_name = {int(k): v for k, v in meta["contiguous_to_name"].items()}
    name_to_coco = {c["name"]: c["id"] for c in meta["categories"]}
    keep_names = set(n.strip() for n in args.categories.split(",") if n.strip()) or None

    image_ids = json.load(open(args.image_ids)) if args.image_ids else None
    ds = CocoInstanceDataset(args.coco_json, args.img_dir, task=task,
                             image_ids=image_ids, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=args.num_workers,
                                         collate_fn=collate_fn)
    print(f"[tv_predict] {len(ds)} images, task={task}, device={device}")

    detections, annotations, ann_id = [], [], 1
    t0 = time.time()
    for images, targets in loader:
        image_id = int(targets[0]["image_id"].item())
        out = model([images[0].to(device)])[0]
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        boxes = out["boxes"].cpu().numpy()
        keep = scores >= args.score_threshold
        masks = (out["masks"].cpu().numpy()[:, 0] if "masks" in out else None)
        kps = (out["keypoints"].cpu().numpy() if "keypoints" in out else None)

        for i in np.nonzero(keep)[0]:
            name = contig_to_name.get(int(labels[i]), str(labels[i]))
            if keep_names and name not in keep_names:
                continue
            x0, y0, x1, y1 = [float(v) for v in boxes[i]]
            bbox = [x0, y0, x1 - x0, y1 - y0]
            det = {"image_id": image_id, "category_id": int(name_to_coco.get(name, labels[i])),
                   "bbox": bbox, "score": float(scores[i])}
            ann = {"id": ann_id, "image_id": image_id, "bbox": bbox,
                   "category_id": det["category_id"], "category_name": name,
                   "score": float(scores[i]), "iscrowd": 0}
            if masks is not None:
                binary = (masks[i] >= args.mask_threshold).astype(np.uint8)
                if binary.sum() == 0:
                    continue
                from pycocotools import mask as mask_utils
                rle = mask_utils.encode(np.asfortranarray(binary))
                rle["counts"] = rle["counts"].decode("ascii")
                det["segmentation"] = rle
                ann["segmentation"] = mask_to_polygons(binary)
                ann["area"] = float(binary.sum())
                if not ann["segmentation"]:
                    continue
            if kps is not None:
                k = kps[i].reshape(-1, 3).copy()
                k[:, 2] = 2.0                       # COCO: 2 = labelled and visible
                det["keypoints"] = [float(v) for v in k.reshape(-1)]
                ann["keypoints"] = det["keypoints"]
                ann["num_keypoints"] = int(k.shape[0])
                ann["area"] = float(bbox[2] * bbox[3])
            detections.append(det)
            annotations.append(ann)
            ann_id += 1

    elapsed = time.time() - t0
    print(f"[tv_predict] {len(detections)} detections in {elapsed:.1f}s "
          f"({len(ds)/max(elapsed,1e-6):.2f} img/s)")

    if args.detections_out:
        Path(args.detections_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.detections_out).write_text(json.dumps(detections))
    if args.annotations_out:
        src = json.load(open(args.coco_json))
        wanted = {a["image_id"] for a in annotations}
        out_coco = {"images": [im for im in src["images"] if im["id"] in wanted],
                    "categories": src["categories"], "annotations": annotations,
                    "info": {"description": "torchvision predictions",
                             "model": meta["recipe"]["backbone"],
                             "score_threshold": args.score_threshold}}
        Path(args.annotations_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.annotations_out).write_text(json.dumps(out_coco, indent=2))

    if args.eval_against:
        return evaluate(args.eval_against, detections, task,
                        image_ids=[ds.samples[i][0] for i in range(len(ds))],
                        out_json=args.metrics_out)
    return None


def sanitize_gt(gt_json, task):
    """
    pycocotools cannot read an annotation whose segmentation is an empty list:
    `frPyObjects` indexes `pyobj[0]` and raises `IndexError: list index out of
    range`, which aborts the whole evaluation rather than skipping one row. Such
    an annotation carries no mask, so a segm evaluation could never match it
    anyway. They are dropped into a cleaned copy in the system temp directory —
    the source file is never modified — and the count is printed, so the number
    of excluded ground-truth rows is visible rather than silent.

    (In `diaphorina_29species_unified.json` there is exactly one: annotation 1909
    on image 523, the single keypoint annotation in an otherwise mask-only file.)
    """
    import hashlib
    import tempfile

    src = Path(gt_json)
    raw = json.loads(src.read_text())
    keep, dropped = [], 0
    for a in raw.get("annotations", []):
        if task == "masks":
            seg = a.get("segmentation")
            good = bool(seg) and (
                (isinstance(seg, list) and any(isinstance(q, list) and len(q) >= 6 for q in seg))
                or isinstance(seg, dict))
        else:
            kp = a.get("keypoints")
            good = bool(kp) and any(kp[2::3])
        if good:
            keep.append(a)
        else:
            dropped += 1
    if dropped == 0:
        return str(src), 0
    tag = hashlib.sha1(f"{src.resolve()}|{src.stat().st_mtime_ns}|{task}".encode()).hexdigest()[:12]
    out = Path(tempfile.gettempdir()) / f"cleaned_gt_{task}_{tag}.json"
    if not out.exists():
        raw["annotations"] = keep
        out.write_text(json.dumps(raw))
    print(f"[tv_predict] ground truth: {dropped} annotation(s) unusable for '{task}' "
          f"and excluded from scoring ({len(keep)} kept) -> {out.name}")
    return str(out), dropped


def evaluate(gt_json, detections, task, image_ids=None, out_json=None):
    """COCOeval on exactly the images that were predicted, so nothing is scored blind."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io

    gt_json, n_dropped = sanitize_gt(gt_json, task)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json)
    if not detections:
        print("[tv_predict] no detections — AP is 0 by definition")
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "n_detections": 0}
    iou_type = "segm" if task == "masks" else "keypoints"
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(list(detections))
        ev = COCOeval(gt, dt, iouType=iou_type)
        if image_ids is not None:
            ev.params.imgIds = sorted(set(int(i) for i in image_ids))
        ev.evaluate(); ev.accumulate()
    ev.summarize()
    s = ev.stats
    metrics = {"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
               "n_detections": len(detections), "iou_type": iou_type,
               "gt_annotations_excluded": n_dropped}
    print(f"[tv_predict] AP {metrics['AP']:.4f}  AP50 {metrics['AP50']:.4f}  "
          f"AP75 {metrics['AP75']:.4f}")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--coco-json", required=True,
                   help="COCO file naming the images to predict (its annotations are ignored "
                        "unless --eval_against points at it)")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--image_ids", default=None, help="JSON list of image ids to predict")
    p.add_argument("--annotations_out", default=None)
    p.add_argument("--detections_out", default=None)
    p.add_argument("--metrics_out", default=None)
    p.add_argument("--eval_against", default=None, help="ground-truth COCO to score against")
    p.add_argument("--categories", default="", help="comma-separated names to keep")
    p.add_argument("--score_threshold", type=float, default=0.5)
    p.add_argument("--mask_threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    predict(p.parse_args())


if __name__ == "__main__":
    main()
