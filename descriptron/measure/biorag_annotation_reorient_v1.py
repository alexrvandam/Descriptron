#!/usr/bin/env python3
"""
biorag_annotation_reorient_v1.py — an image was turned after it was annotated: put the polygons back
=====================================================================================================

A photograph that is rotated or mirrored in an image editor AFTER it was annotated keeps its file
name, so nothing downstream notices: the polygons are still in the old frame and now lie partly on
blank slide. `biorag_annotation_screen_v1.py` finds such annotations and sets them aside. This script
repairs them where the edit was one of the eight rigid moves of a rectangle (the four rotations, each
with or without a mirror), which is what an image editor's "rotate" and "flip" buttons do:

  for every image, all of its polygons are moved together by each of the eight transforms, and the
  share of each polygon that falls on the specimen (darker than the mount) is measured;
  a transform other than "leave as is" is accepted only when the polygons sit badly as they are,
  sit well after it, and the gain is large — otherwise the image is left alone and reported.

Nothing is overwritten: a corrected copy of the COCO file is written next to a report. The same
per-image transform can be applied to other COCO files of the same images (landmarks), because a
landmark placed before the image was turned is in the old frame too.

  python biorag_annotation_reorient_v1.py --coco <coco.json> --image_dir <images> --out_dir <dir> \
      [--apply_to <keypoints.json> ...]

No model is called.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

VERSION = "1.0"
Image.MAX_IMAGE_PIXELS = None


def foreground(im: np.ndarray) -> np.ndarray:
    """Specimen against a pale mount: darker than the bright mode of the image (as in the screen)."""
    g = im.mean(2)
    return (g < np.percentile(g, 60) * 0.92).astype(np.uint8)


# (name, needs the source frame to be the actual frame turned a quarter, function of x, y, W, H)
# W, H = the ACTUAL image size; for the quarter-turn family the polygons were drawn on an H x W frame
TRANSFORMS = [
    ("as is", False, lambda x, y, W, H: (x, y)),
    ("rotated 180", False, lambda x, y, W, H: (W - x, H - y)),
    ("mirrored left-right", False, lambda x, y, W, H: (W - x, y)),
    ("mirrored top-bottom", False, lambda x, y, W, H: (x, H - y)),
    ("rotated 90 clockwise", True, lambda x, y, W, H: (W - y, x)),
    ("rotated 90 anticlockwise", True, lambda x, y, W, H: (y, H - x)),
    ("transposed", True, lambda x, y, W, H: (y, x)),
    ("anti-transposed", True, lambda x, y, W, H: (W - y, H - x)),
]


def move(flat, fn, W, H):
    p = np.asarray(flat, float).reshape(-1, 2)
    x, y = fn(p[:, 0], p[:, 1], W, H)
    return np.stack([x, y], 1)


def coverage(fg, pts, reduce):
    m = np.zeros(fg.shape, np.uint8)
    cv2.fillPoly(m, [(pts / reduce).astype(np.int32)], 1)
    s = m.sum()
    return float((fg & m).sum() / s) if s >= 25 else np.nan


def main():
    ap = argparse.ArgumentParser(description="Re-orient annotations whose image was rotated or mirrored after annotation")
    ap.add_argument("--coco", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--apply_to", nargs="*", default=[],
                    help="other COCO files of the same images (e.g. landmarks): the accepted transforms are "
                         "applied to their polygons and keypoints too")
    ap.add_argument("--poor", type=float, default=0.55, help="mean coverage below which an image is a candidate")
    ap.add_argument("--good", type=float, default=0.58,
                    help="mean coverage a transform must reach (transparent structures such as wing "
                         "membranes rarely exceed 0.8 even when the polygons fit)")
    ap.add_argument("--gain", type=float, default=0.18, help="least improvement over as-is")
    ap.add_argument("--reduce", type=int, default=2)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j = json.loads(Path(a.coco).read_text())
    imgs = {i["id"]: i for i in j["images"]}
    by_img = {}
    for ann in j["annotations"]:
        if isinstance(ann.get("segmentation"), list) and ann["segmentation"] and ann["segmentation"][0]:
            by_img.setdefault(ann["image_id"], []).append(ann)

    rows, accepted = [], {}
    for n, (iid, anns) in enumerate(sorted(by_img.items()), 1):
        rec = imgs.get(iid)
        p = Path(a.image_dir) / rec["file_name"] if rec else None
        if p is None or not p.exists():
            continue
        with Image.open(p) as raw:
            W, H = raw.size
            fg = foreground(np.asarray(raw.convert("RGB").reduce(a.reduce)).astype(np.float32))
        turned_frame = (rec.get("width"), rec.get("height")) == (H, W) and W != H
        score = {}
        for name, quarter, fn in TRANSFORMS:
            cov = [coverage(fg, move(ann["segmentation"][0], fn, W, H), a.reduce) for ann in anns]
            score[name] = float(np.nanmean(cov)) if np.isfinite(cov).any() else np.nan
        asis = score["as is"]
        best = max(score, key=lambda k: -1 if np.isnan(score[k]) else score[k])
        second = sorted((v for k, v in score.items() if k != best and not np.isnan(v)), reverse=True)
        take = (best != "as is" and (asis < a.poor or turned_frame) and score[best] >= a.good
                and score[best] - asis >= a.gain)
        # two transforms that fit equally well cannot be told apart by coverage: report, do not apply
        ambiguous = bool(take and second and score[best] - second[0] < 0.05)
        if ambiguous:
            take = False
        rows.append({"image": rec["file_name"], "polygons": len(anns), "coverage_as_is": round(asis, 3),
                     "best_transform": best, "coverage_best": round(score[best], 3),
                     "coco_frame": f"{rec.get('width')}x{rec.get('height')}", "actual_frame": f"{W}x{H}",
                     "applied": bool(take), "ambiguous": ambiguous,
                     **{f"cov[{k}]": round(v, 3) for k, v in score.items()}})
        if take:
            accepted[rec["file_name"]] = best
        if n % 80 == 0:
            print(f"  {n}/{len(by_img)} images")

    rep = pd.DataFrame(rows).sort_values("coverage_as_is")
    rep.to_csv(out / "reorientation_report.tsv", sep="\t", index=False)
    fns = {name: fn for name, _, fn in TRANSFORMS}

    def apply(coco_path: Path) -> int:
        d = json.loads(coco_path.read_text())
        by_name = {i["id"]: i for i in d["images"]}
        touched = 0
        sizes = {}
        for i in d["images"]:
            if i["file_name"] in accepted:
                with Image.open(Path(a.image_dir) / i["file_name"]) as raw:
                    sizes[i["id"]] = raw.size
                i["width"], i["height"] = sizes[i["id"]]
                i["reoriented"] = accepted[i["file_name"]]
        for ann in d["annotations"]:
            if ann["image_id"] not in sizes:
                continue
            W, H = sizes[ann["image_id"]]
            fn = fns[accepted[by_name[ann["image_id"]]["file_name"]]]
            if isinstance(ann.get("segmentation"), list):
                ann["segmentation"] = [np.round(move(s, fn, W, H), 2).reshape(-1).tolist()
                                       for s in ann["segmentation"] if s]
                if ann["segmentation"]:
                    allp = np.concatenate([np.asarray(s).reshape(-1, 2) for s in ann["segmentation"]])
                    x0, y0 = allp.min(0)
                    x1, y1 = allp.max(0)
                    ann["bbox"] = [round(float(x0), 2), round(float(y0), 2),
                                   round(float(x1 - x0), 2), round(float(y1 - y0), 2)]
            if ann.get("keypoints"):
                k = np.asarray(ann["keypoints"], float).reshape(-1, 3)
                k[:, :2] = move(k[:, :2].reshape(-1), fn, W, H)
                ann["keypoints"] = np.round(k, 2).reshape(-1).tolist()
            touched += 1
        dst = out / (coco_path.stem + "_reoriented" + coco_path.suffix)
        dst.write_text(json.dumps(d))
        print(f"{coco_path.name}: {touched} annotations moved on {len(sizes)} images -> {dst}")
        return touched

    apply(Path(a.coco))
    for other in a.apply_to:
        apply(Path(other))
    (out / "reorientation_summary.json").write_text(json.dumps(
        {"version": VERSION, "coco": str(a.coco), "images_tested": len(rows),
         "images_reoriented": accepted,
         "ambiguous_left_alone": rep.loc[rep["ambiguous"], "image"].tolist(),
         "rule": f"applied when coverage as is < {a.poor} (or the COCO frame is the actual frame turned a "
                 f"quarter), the best transform reaches >= {a.good}, gains >= {a.gain} and beats the "
                 f"runner-up by >= 0.05"}, indent=2))
    print(f"\n{len(accepted)} images re-oriented of {len(rows)} tested")
    print(rep[rep["applied"] | rep["ambiguous"]][["image", "coverage_as_is", "best_transform", "coverage_best",
                                                  "coco_frame", "actual_frame", "applied"]].to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
