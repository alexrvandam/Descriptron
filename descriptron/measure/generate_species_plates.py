#!/usr/bin/env python3
"""
Generate per-species figure plates for BioRAG taxonomic descriptions.

For each species, produces a composite figure plate containing:
  1. Original specimen images (one per body region)
  2. Annotated overlays with category labels
  3. Selected analytical figures (PCA, CVA, etc.) from the pipeline output

Also collects general pipeline-level summary figures (whole-wing PCA,
confabulation charts, key quality plots) into a general_figures/ directory.

Usage:
  python generate_species_plates.py \
    --coco_json /path/to/coco.json \
    --image_dir /path/to/images \
    --group_labels /path/to/group_labels.csv \
    --output_base /path/to/pipeline_output \
    --plates_dir /path/to/species_plates
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate per-species figure plates for BioRAG descriptions")
    p.add_argument("--coco_json", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--group_labels", required=True)
    p.add_argument("--output_base", required=True,
                   help="Pipeline output base (contains measurements/, semilandmarks/, etc.)")
    p.add_argument("--plates_dir", required=True,
                   help="Where to write species plate PNGs")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--max_images_per_species", type=int, default=8)
    return p.parse_args()


CATEGORY_COLORS = {}

def _get_color(cat_name):
    if cat_name not in CATEGORY_COLORS:
        np.random.seed(hash(cat_name) % 2**31)
        CATEGORY_COLORS[cat_name] = tuple(np.random.randint(60, 230, 3).tolist())
    return CATEGORY_COLORS[cat_name]


def load_coco(coco_path):
    with open(coco_path) as f:
        coco = json.load(f)
    id_to_img = {img["id"]: img for img in coco["images"]}
    id_to_cat = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)
    return coco, id_to_img, id_to_cat, img_anns


def _normalize_fn(fn):
    return fn.replace(" ", "_")


def load_group_labels(path):
    df = pd.read_csv(path)
    fn_col = "filename" if "filename" in df.columns else df.columns[0]
    mapping = {}
    norm_mapping = {}
    for fn, sp in zip(df[fn_col], df["group_label"]):
        mapping[fn] = sp
        norm_mapping[_normalize_fn(fn)] = sp
    mapping.update(norm_mapping)
    return mapping


def decode_rle(rle, h, w):
    counts = rle["counts"]
    if isinstance(counts, str):
        import pycocotools.mask as mask_util
        return mask_util.decode(rle).astype(bool)
    mask = np.zeros(h * w, dtype=bool)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:
            mask[pos:pos + c] = True
        pos += c
    return mask.reshape((h, w), order="F")


def seg_to_mask(seg, h, w):
    if isinstance(seg, dict) and "counts" in seg:
        return decode_rle(seg, h, w)
    elif isinstance(seg, list):
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)
    return np.zeros((h, w), dtype=bool)


def draw_annotated_image(img_bgr, anns, id_to_cat):
    overlay = img_bgr.copy()
    labels_drawn = []

    for ann in anns:
        cat_name = id_to_cat.get(ann["category_id"], "?")
        color = _get_color(cat_name)
        h, w = img_bgr.shape[:2]

        seg = ann.get("segmentation")
        if seg is None:
            continue

        mask = seg_to_mask(seg, h, w)
        colored = np.zeros_like(overlay)
        colored[mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.35, 0)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)

        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                labels_drawn.append((cx, cy, cat_name, color))

    for cx, cy, cat_name, color in labels_drawn:
        label = cat_name.replace("_", " ")
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.3, min(0.5, img_bgr.shape[1] / 2000))
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        cv2.rectangle(overlay, (cx - 2, cy - th - 4), (cx + tw + 2, cy + 2),
                      (0, 0, 0), -1)
        cv2.putText(overlay, label, (cx, cy - 2), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay


def make_species_plate(species, images_data, output_path, dpi=150):
    n = len(images_data)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (img_name, img_rgb, body_part) in enumerate(images_data):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.imshow(img_rgb)
        ax.set_title(body_part, fontsize=7, pad=2)
        ax.axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Species: {species}", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def extract_body_part(filename):
    fn = filename.lower()
    for part in ["forewing", "rostrum", "metaleg", "terminalia", "head"]:
        if part in fn:
            return part
    return "specimen"


def collect_general_figures(output_base, plates_dir):
    general_dir = Path(plates_dir) / "general_figures"
    general_dir.mkdir(parents=True, exist_ok=True)

    base = Path(output_base)
    figure_sources = [
        (base / "biorag_descriptions" / "confabulation_comparison", "confab_"),
        (base / "biorag_descriptions" / "confabulation_report", "confab_report_"),
        (base / "biorag_descriptions" / "key_comparison", "key_"),
        (base / "biorag_descriptions" / "key_qualitative_check", "keyqual_"),
    ]

    semi_dir = base / "semilandmarks"
    if semi_dir.exists():
        for cat_dir in sorted(semi_dir.iterdir()):
            if cat_dir.is_dir():
                for png in cat_dir.glob("*.png"):
                    if any(k in png.name for k in ["pca_", "cva_", "umap_",
                           "allometry_", "pairwise_heatmap_", "mahalanobis_tree_"]):
                        figure_sources.append((cat_dir, f"semi_{cat_dir.name}_"))

    copied = 0
    for src_dir, prefix in figure_sources:
        if not src_dir.exists():
            continue
        for png in sorted(src_dir.glob("*.png")):
            dst = general_dir / f"{prefix}{png.name}"
            if not dst.exists():
                shutil.copy2(png, dst)
                copied += 1

    print(f"  Collected {copied} general figures → {general_dir}")
    return general_dir


def main():
    args = parse_args()

    coco, id_to_img, id_to_cat, img_anns = load_coco(args.coco_json)
    fn_to_species = load_group_labels(args.group_labels)

    plates_dir = Path(args.plates_dir)
    plates_dir.mkdir(parents=True, exist_ok=True)

    species_images = defaultdict(list)
    for img in coco["images"]:
        fn = img["file_name"]
        species = fn_to_species.get(fn) or fn_to_species.get(_normalize_fn(fn))
        if species:
            species_images[species].append(img)

    print(f"Generating plates for {len(species_images)} species...")

    for species in sorted(species_images.keys()):
        sp_dir = plates_dir / species
        sp_dir.mkdir(parents=True, exist_ok=True)

        imgs = species_images[species]
        if len(imgs) > args.max_images_per_species:
            imgs = imgs[:args.max_images_per_species]

        images_data = []
        for img_info in imgs:
            img_path = os.path.join(args.image_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            anns = img_anns.get(img_info["id"], [])
            annotated = draw_annotated_image(img_bgr, anns, id_to_cat)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            body_part = extract_body_part(img_info["file_name"])
            specimen_id = img_info["file_name"].replace(".tif", "").replace(".TIF", "")

            out_single = sp_dir / f"{specimen_id}_annotated.png"
            cv2.imwrite(str(out_single), annotated)

            original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            out_orig = sp_dir / f"{specimen_id}_original.png"
            cv2.imwrite(str(out_orig), img_bgr)

            images_data.append((img_info["file_name"], annotated_rgb, body_part))

        if images_data:
            plate_path = sp_dir / f"{species}_plate.png"
            make_species_plate(species, images_data, plate_path, dpi=args.dpi)
            print(f"  {species}: {len(images_data)} images → {plate_path}")

    print("\nCollecting general pipeline figures...")
    collect_general_figures(args.output_base, args.plates_dir)

    print(f"\nAll plates in: {plates_dir}")


if __name__ == "__main__":
    main()
