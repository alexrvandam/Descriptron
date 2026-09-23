#!/usr/bin/env python3
"""
render_detections_v1.py — look at the masks, not just the AP
============================================================

AP is one number for a whole fold; it cannot tell you *how* a model is wrong —
whether it clips a wing cell, merges two sclerites, or misses pale structures.
This renders the saved COCO detections as overlays so the masks can be judged by
eye, side by side with the ground truth and, optionally, with another arm.

It reads the `detections_*.json` the sweep already wrote, so it needs no GPU and
no re-running of any model.

    python render_detections_v1.py \\
        --coco_json unified_sizefix.json --img_dir images/ \\
        --detections tv_v2=sweep/tv_v2/detections_fold0.json \\
                     d2=sweep/d2/detections_fold0.json \\
        --out_dir sweep/previews_fold0 --n 12

Each panel is: ground truth, then one column per arm, same image, same colours
per category. `--score_threshold` applies only to the display; the AP in the
sweep integrates over the whole curve and is unaffected by what is drawn here.
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np


def category_colours(categories):
    """One stable colour per category, evenly spread round the hue circle."""
    out = {}
    n = max(len(categories), 1)
    for i, c in enumerate(sorted(categories, key=lambda c: c["id"])):
        r, g, b = colorsys.hsv_to_rgb((i / n) % 1.0, 0.62, 0.98)
        out[c["id"]] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def draw(image, items, colours, names, alpha=0.45, outline=True):
    import cv2
    canvas = image.copy()
    overlay = image.copy()
    for it in items:
        m = it["mask"]
        if m is None or m.sum() == 0:
            continue
        col = colours.get(it["category_id"], (255, 0, 0))
        overlay[m.astype(bool)] = col
        if outline:
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cont, -1, col, 2)
    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    for it in items:
        m = it["mask"]
        if m is None or m.sum() == 0:
            continue
        ys, xs = np.nonzero(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        label = names.get(it["category_id"], str(it["category_id"]))
        if it.get("score") is not None:
            label += f" {it['score']:.2f}"
        cv2.putText(canvas, label, (max(cx - 40, 2), max(cy, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (max(cx - 40, 2), max(cy, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def decode(item, h, w):
    from pycocotools import mask as mu
    seg = item.get("segmentation")
    if seg is None:
        return None
    try:
        if isinstance(seg, list):
            polys = [p for p in seg if isinstance(p, list) and len(p) >= 6]
            if not polys:
                return None
            m = mu.decode(mu.merge(mu.frPyObjects(polys, h, w)))
        else:
            rle = dict(seg)
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("ascii")
            m = mu.decode(rle)
        return m.max(axis=2) if m.ndim == 3 else m
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco_json", required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--detections", nargs="+", required=True,
                   help="one or more name=path/to/detections.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=12, help="images to render")
    p.add_argument("--score_threshold", type=float, default=0.5,
                   help="display only; the sweep's AP is unaffected")
    p.add_argument("--image_ids", default=None)
    a = p.parse_args()

    import cv2
    from PIL import Image

    coco = json.loads(Path(a.coco_json).read_text())
    images = {im["id"]: im for im in coco["images"]}
    names = {c["id"]: c["name"] for c in coco["categories"]}
    colours = category_colours(coco["categories"])
    gt = {}
    for ann in coco["annotations"]:
        gt.setdefault(ann["image_id"], []).append(ann)

    arms = {}
    for spec in a.detections:
        name, _, path = spec.partition("=")
        dets = json.loads(Path(path).read_text())
        by_img = {}
        for d in dets:
            if d.get("score", 1.0) >= a.score_threshold:
                by_img.setdefault(d["image_id"], []).append(d)
        arms[name] = by_img

    candidates = sorted(set.intersection(*[set(v) for v in arms.values()])
                        if arms else set())
    if a.image_ids:
        want = set(json.loads(Path(a.image_ids).read_text()))
        candidates = [i for i in candidates if i in want]
    step = max(1, len(candidates) // max(a.n, 1))
    picks = candidates[::step][:a.n]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(candidates)} images predicted by every arm; rendering {len(picks)}")

    for image_id in picks:
        meta = images[image_id]
        path = Path(a.img_dir) / Path(meta["file_name"]).name
        if not path.exists():
            path = Path(a.img_dir) / meta["file_name"]
        img = np.asarray(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]

        panels, titles = [], []
        g = [{"mask": decode(x, h, w), "category_id": x["category_id"], "score": None}
             for x in gt.get(image_id, [])]
        panels.append(draw(img, g, colours, names))
        titles.append(f"ground truth ({len(g)})")
        for name, by_img in arms.items():
            items = [{"mask": decode(x, h, w), "category_id": x["category_id"],
                      "score": x.get("score")} for x in by_img.get(image_id, [])]
            panels.append(draw(img, items, colours, names))
            titles.append(f"{name} ({len(items)} @ score>={a.score_threshold})")

        bar = 28
        strip = []
        for panel, title in zip(panels, titles):
            head = np.full((bar, panel.shape[1], 3), 255, np.uint8)
            cv2.putText(head, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (20, 20, 20), 1, cv2.LINE_AA)
            strip.append(np.vstack([head, panel]))
        sheet = np.hstack(strip)
        name = Path(meta["file_name"]).stem[:70].replace(" ", "_")
        dst = out / f"{image_id:04d}_{name}.jpg"
        cv2.imwrite(str(dst), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"   {dst.name}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
