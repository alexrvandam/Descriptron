#!/usr/bin/env python3
"""
fix_coco_image_sizes_v1.py — repair width/height recorded in a COCO file
========================================================================

Some COCO exports record `width: 0, height: 0` for an image. The consequences are
uneven, and only one of them is loud:

* **Detectron2 refuses to train**: `SizeMismatchError: got (1536, 1024),
  expect (0, 0)`. It is the only component that checks.
* **pycocotools is silent**: `frPyObjects(segmentation, 0, 0)` returns a 0x0 mask
  of area 0, so every annotation on such an image becomes ground truth that no
  prediction can ever match. AP is depressed and nothing says why.
* A loader that reads the size from the image file, as `tv_coco_dataset` does, is
  unaffected — which is how the defect survives.

This reads the true size from each image and writes a repaired copy. **The source
file is never modified**; the output is a new file and the changes are listed.

    python fix_coco_image_sizes_v1.py --coco_json in.json --img_dir images/ \
        --out repaired.json [--check_all]

`--check_all` verifies every image, not only those recorded as zero, and reports
any whose recorded size disagrees with the file — a rotated-after-annotation image
shows up here as a swapped width and height.
"""
import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco_json", required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--check_all", action="store_true")
    a = p.parse_args()

    d = json.loads(Path(a.coco_json).read_text())
    img_dir = Path(a.img_dir)
    fixed, swapped, missing, checked = [], [], [], 0

    for im in d.get("images", []):
        zero = not im.get("width") or not im.get("height")
        if not (zero or a.check_all):
            continue
        path = img_dir / Path(im["file_name"]).name
        if not path.exists():
            path = img_dir / im["file_name"]
        if not path.exists():
            missing.append(im["file_name"])
            continue
        with Image.open(path) as handle:
            w, h = handle.size
        checked += 1
        if zero:
            im["width"], im["height"] = w, h
            fixed.append((im["id"], im["file_name"], w, h))
        elif (im["width"], im["height"]) != (w, h):
            swapped.append((im["id"], im["file_name"],
                            (im["width"], im["height"]), (w, h)))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d))

    ann_by_image = {}
    for ann in d.get("annotations", []):
        ann_by_image[ann["image_id"]] = ann_by_image.get(ann["image_id"], 0) + 1
    n_ann = sum(ann_by_image.get(i, 0) for i, *_ in fixed)

    print(f"read   {a.coco_json}")
    print(f"images {len(d.get('images', []))}, inspected {checked}")
    print(f"FIXED  {len(fixed)} image(s) recorded as 0x0, carrying {n_ann} annotations")
    for i, name, w, h in fixed[:20]:
        print(f"   id {i:>5}  {w}x{h}  {name}")
    if len(fixed) > 20:
        print(f"   … and {len(fixed)-20} more")
    if swapped:
        print(f"MISMATCHED {len(swapped)} image(s) whose recorded size differs from the file:")
        for i, name, rec, real in swapped[:20]:
            note = " (width/height swapped)" if rec == real[::-1] else ""
            print(f"   id {i:>5}  recorded {rec} vs file {real}{note}  {name}")
    if missing:
        print(f"MISSING  {len(missing)} image file(s) not found under {img_dir}")
        for name in missing[:10]:
            print(f"   {name}")
    print(f"wrote  {a.out}")


if __name__ == "__main__":
    main()
