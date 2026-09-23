#!/usr/bin/env python3
"""
compile_specimen_data.py — Diaphorina Pipeline Data Compilation for BioRAG
==========================================================================

Joins outputs from 6 analysis pipelines (measurements, semilandmarks, color
extraction, color homology, texture homology, landmark GPA) into unified
machine-readable formats. Identifies statistically diagnostic features per
category using Kruskal-Wallis + Dunn's post-hoc tests.

Outputs:
  - Full archive CSV (all features, all specimens)
  - Diagnostic features CSV (only significant discriminating features)
  - Diagnostic report JSON (for VLM prompt injection)
  - NPY feature matrix (for ML downstream)
  - COCO JSON-LD knowledge graph (extends COCO with traits + ontology)
  - Per-specimen summary JSONs (diagnostic features only)
  - Coverage matrix CSV

Usage:
  python compile_specimen_data.py \\
    --measurements_dir "/path/to/Diaphorina_measurements" \\
    --semilandmarks_dir "/path/to/Diaphorina_semilandmarks" \\
    --color_dir "/path/to/Diaphorina_color_extraction" \\
    --color_homology_dir "/path/to/Diaphorina_color_homology" \\
    --texture_dir "/path/to/Diaphorina_texture_homology" \\
    --landmark_gpa_dirs "/path/to/Diaphorina_landmark_gpa_forewing" \\
    --coco_json "/path/to/diaphorina_combined_all_bodyparts.json" \\
    --group_labels "/path/to/group_labels.csv" \\
    --output_dir "/path/to/Diaphorina_compiled"
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kruskal

try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# FILENAME NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_specimen_key(filename: str, category_from_dir: str = None
                           ) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Normalize a filename from any pipeline into (image_base, annotation_index, category).

    Handles formats:
      - Measurements: separate columns (not parsed here)
      - Semilandmarks: "{image}.tif_{ann_idx}" in category subdir
      - Color extraction: "{stem}tif_{ann_idx}_fg_{category}"
      - Color/Texture homology: "{image}.tif_{ann_idx}" in category subdir
      - Landmark GPA: just image filename (no annotation index)
    """
    name = str(filename).strip()

    # Strip _fg_{category} suffix (color extraction format)
    fg_match = re.match(r"^(.+)_fg_(.+)$", name)
    category = None
    if fg_match:
        name = fg_match.group(1)
        category = fg_match.group(2)

    # Normalize tif vs .tif (color extraction drops the dot)
    name = re.sub(r"(?<!\.)tif", ".tif", name)

    # Try to split off numeric annotation index from the end
    ann_idx = None
    idx_match = re.match(r"^(.+?)_(\d+)$", name)
    if idx_match:
        base_candidate = idx_match.group(1)
        idx_candidate = int(idx_match.group(2))
        if base_candidate.endswith(".tif") or base_candidate.endswith(".png"):
            ann_idx = idx_candidate
            name = base_candidate
        elif not base_candidate[-1].isdigit():
            ann_idx = idx_candidate
            name = base_candidate

    # Normalize spaces to underscores
    name = name.replace(" ", "_")

    if category_from_dir and not category:
        category = category_from_dir

    return name, ann_idx, category


def _normalize_image_filename(fn: str) -> str:
    """Normalize an image filename for matching (spaces→underscores, strip path)."""
    return Path(fn).name.replace(" ", "_")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_scale_factors(measurements_dir: str) -> Dict[str, float]:
    """Load combined_scales.csv → {image_base: pixels_per_mm}."""
    csv_path = Path(measurements_dir) / "combined_scales.csv"
    if not csv_path.exists():
        logger.warning(f"Scale factors CSV not found: {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    scales = {}
    fn_col = "image_filename" if "image_filename" in df.columns else df.columns[0]
    for _, row in df.iterrows():
        fn = _normalize_image_filename(str(row[fn_col]).strip())
        val = row.get("pixels_per_unit")
        if pd.notna(val) and float(val) > 0:
            scales[fn] = float(val)
    logger.info(f"Scale factors: {len(scales)} images with calibration")
    return scales


# Columns that represent absolute sizes (need calibration)
_PIXEL_SIZE_COLS = {
    "length_pixels", "height_pixels", "area_pixels", "perimeter_pixels",
    "equivalent_diameter_pixels",
}
# PCA ellipse axes use 2*sqrt(eigenvalue)*sqrt(n_contour_points) — the sqrt(n)
# factor makes them scale with contour resolution, not specimen size. Exclude.
_SKIP_COLS = {
    "major_axis_length_pixels", "minor_axis_length_pixels",
    "major_axis_length_mm", "minor_axis_length_mm",
}
# Columns that are dimensionless (keep as-is)
_DIMENSIONLESS_COLS = {
    "aspect_ratio", "length_to_height_ratio", "solidity", "extent",
    "orientation_degrees",
}


def load_measurements(measurements_dir: str) -> Dict[Tuple[str, str], Dict]:
    """Load all_metrics.csv with scale-bar calibration.

    Pixel-based size measurements are converted to mm using combined_scales.csv.
    Dimensionless ratios (aspect_ratio, solidity, etc.) are kept unchanged.
    """
    csv_path = Path(measurements_dir) / "all_metrics.csv"
    if not csv_path.exists():
        logger.warning(f"Measurements CSV not found: {csv_path}")
        return {}

    scales = _load_scale_factors(measurements_dir)

    data = {}
    skip_cols = {"image_filename", "image_id", "category_id", "category_name",
                 "annotation_index", "segmentation_coords", "method",
                 "length_line_start", "length_line_end",
                 "height_line_start", "height_line_end"} | _SKIP_COLS
    # Skip raw pixel columns when we can calibrate (also skip pre-existing _mm
    # columns since we recompute them from pixels + scale factor for consistency)
    mm_cols = {"length_mm", "height_mm", "area_mm2", "perimeter_mm",
               "equivalent_diameter_mm"}

    df = pd.read_csv(csv_path, low_memory=False)
    n_calibrated = 0

    for _, row in df.iterrows():
        img = _normalize_image_filename(str(row.get("image_filename", "")))
        cat = str(row.get("category_name", "")).strip().replace(" ", "_")
        if not img or not cat:
            continue

        ppm = scales.get(img, 0)
        features = {}

        for col in df.columns:
            if col in skip_cols or col in mm_cols:
                continue

            val = row[col]
            if not pd.notna(val):
                continue

            try:
                fval = float(val)
            except (ValueError, TypeError):
                continue

            if col in _PIXEL_SIZE_COLS and ppm > 0:
                # Convert to mm
                if "area" in col:
                    cal_val = fval / (ppm * ppm)
                    cal_name = col.replace("_pixels", "_mm2")
                else:
                    cal_val = fval / ppm
                    cal_name = col.replace("_pixels", "_mm")
                features[f"meas_{cal_name}"] = cal_val
            elif col in _PIXEL_SIZE_COLS:
                # No scale factor — skip pixel measurement (not comparable)
                pass
            elif col in _DIMENSIONLESS_COLS:
                features[f"meas_{col}"] = fval
            else:
                features[f"meas_{col}"] = fval

        if ppm > 0:
            n_calibrated += 1

        key = (img, cat)
        data[key] = features

    logger.info(f"Measurements: {len(data)} entries, "
                f"{n_calibrated} calibrated with scale bar ({csv_path.name})")
    return data


def _load_category_csv(base_dir: str, prefix: str,
                       csv_pattern: str, skip_cols: set = None
                       ) -> Dict[Tuple[str, str], Dict]:
    """Generic loader for per-category-subdir CSVs."""
    base = Path(base_dir)
    if not base.exists():
        logger.warning(f"Directory not found: {base}")
        return {}

    skip = skip_cols or set()
    data = {}

    for cat_dir in sorted(base.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_name = cat_dir.name
        # Skip numeric-only dirs (semilandmark image indices)
        if cat_name.isdigit():
            continue

        csv_name = csv_pattern.format(cat=cat_name)
        csv_path = cat_dir / csv_name
        if not csv_path.exists():
            # Try alternate patterns
            candidates = list(cat_dir.glob(f"*{cat_name}*.csv"))
            csv_path = None
            for c in candidates:
                if "phylo" in c.name or "features" in c.name or "traits" in c.name:
                    csv_path = c
                    break
            if csv_path is None:
                continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            logger.warning(f"Error reading {csv_path}: {e}")
            continue

        fn_col = "filename" if "filename" in df.columns else df.columns[0]

        for _, row in df.iterrows():
            raw_fn = str(row[fn_col]).strip()
            if not raw_fn or raw_fn == "nan":
                continue

            img_base, ann_idx, _ = normalize_specimen_key(raw_fn, cat_name)

            features = {}
            for col in df.columns:
                if col == fn_col or col in skip:
                    continue
                val = row[col]
                if pd.notna(val):
                    try:
                        features[f"{prefix}{col}"] = float(val)
                    except (ValueError, TypeError):
                        features[f"{prefix}{col}"] = str(val)

            key = (img_base, cat_name)
            data[key] = features

    logger.info(f"{prefix.rstrip('_')}: {len(data)} entries across "
                f"{len(set(k[1] for k in data))} categories")
    return data


def _scan_phylo_csvs(base_dir: str, pipeline: str,
                     skip_cols: set = None) -> Dict[Tuple[str, str], Dict]:
    """
    Scan ALL *phylo* and *features* CSVs in each category subfolder.
    Each CSV gets a distinct prefix derived from its filename, e.g.:
      color_traits_phylo_{cat}_adaptive.csv  → color_adaptive_
      color_traits_phylo_{cat}_median.csv    → color_median_
      texture_traits_phylo_{cat}.csv         → tex_phylo_
      texture_homology_features_{cat}.csv    → tex_
      color_homology_features_{cat}_combined_hw1.csv → colhom_
    """
    base = Path(base_dir)
    if not base.exists():
        logger.warning(f"Directory not found: {base}")
        return {}

    skip = skip_cols or set()
    data = {}
    files_loaded = 0

    for cat_dir in sorted(base.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_name = cat_dir.name
        if cat_name.isdigit():
            continue

        csvs = sorted(cat_dir.glob("*.csv"))
        for csv_path in csvs:
            name = csv_path.name.lower()
            # Skip PCA variance, loadings, analysis summaries — not per-specimen data
            if any(skip_word in name for skip_word in
                   ["pca_variance", "loadings", "analysis_", "pattern_summary"]):
                continue
            # Must be a traits/features/homology file
            if not any(kw in name for kw in ["phylo", "features", "traits"]):
                continue

            # Derive prefix from filename
            prefix = _derive_prefix(pipeline, csv_path.name, cat_name)

            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except Exception as e:
                logger.warning(f"Error reading {csv_path}: {e}")
                continue

            fn_col = "filename" if "filename" in df.columns else df.columns[0]

            for _, row in df.iterrows():
                raw_fn = str(row[fn_col]).strip()
                if not raw_fn or raw_fn == "nan":
                    continue

                img_base, ann_idx, _ = normalize_specimen_key(raw_fn, cat_name)
                key = (img_base, cat_name)

                if key not in data:
                    data[key] = {}

                for col in df.columns:
                    if col == fn_col or col in skip:
                        continue
                    val = row[col]
                    if pd.notna(val):
                        try:
                            data[key][f"{prefix}{col}"] = float(val)
                        except (ValueError, TypeError):
                            pass

            files_loaded += 1

    n_cats = len(set(k[1] for k in data))
    logger.info(f"{pipeline}: {len(data)} entries across {n_cats} categories "
                f"({files_loaded} CSVs loaded)")
    return data


def _derive_prefix(pipeline: str, filename: str, cat_name: str) -> str:
    """Derive a column prefix from a CSV filename."""
    fn = filename.lower()
    fn = fn.replace(cat_name.lower(), "").replace(" ", "_")

    if pipeline == "color":
        if "adaptive" in fn:
            return "color_adaptive_"
        elif "median" in fn:
            return "color_median_"
        elif "mean" in fn and "homology" not in fn:
            return "color_mean_"
        return "color_"
    elif pipeline == "colhom":
        if "traits_phylo" in fn:
            return "colhom_phylo_"
        return "colhom_"
    elif pipeline == "tex":
        if "traits_phylo" in fn:
            return "tex_phylo_"
        return "tex_"
    elif pipeline == "shape":
        return "shape_"
    return f"{pipeline}_"


def load_semilandmarks(semilandmarks_dir: str) -> Dict[Tuple[str, str], Dict]:
    return _load_category_csv(
        semilandmarks_dir, "shape_",
        "shape_traits_phylo_{cat}.csv",
        skip_cols={"group", "cluster"}
    )


def load_color_extraction(color_dirs: List[str]) -> Dict[Tuple[str, str], Dict]:
    """Load all color phylo CSVs from one or more directories."""
    combined = {}
    for d in color_dirs:
        result = _scan_phylo_csvs(d, "color",
                                  skip_cols={"threshold", "n_clusters"})
        for key, feats in result.items():
            if key not in combined:
                combined[key] = {}
            combined[key].update(feats)
    n_cats = len(set(k[1] for k in combined))
    logger.info(f"color (all dirs): {len(combined)} entries across {n_cats} categories")
    return combined


def load_color_homology(color_homology_dirs: List[str]) -> Dict[Tuple[str, str], Dict]:
    """Load all color homology CSVs from one or more directories."""
    combined = {}
    for d in color_homology_dirs:
        result = _scan_phylo_csvs(d, "colhom")
        for key, feats in result.items():
            if key not in combined:
                combined[key] = {}
            combined[key].update(feats)
    n_cats = len(set(k[1] for k in combined))
    logger.info(f"colhom (all dirs): {len(combined)} entries across {n_cats} categories")
    return combined


def load_texture_homology(texture_dirs: List[str]) -> Dict[Tuple[str, str], Dict]:
    """Load all texture homology CSVs from one or more directories."""
    combined = {}
    for d in texture_dirs:
        result = _scan_phylo_csvs(d, "tex")
        for key, feats in result.items():
            if key not in combined:
                combined[key] = {}
            combined[key].update(feats)
    n_cats = len(set(k[1] for k in combined))
    logger.info(f"tex (all dirs): {len(combined)} entries across {n_cats} categories")
    return combined


def load_landmark_gpa(gpa_dirs: List[str]) -> Dict[Tuple[str, str], Dict]:
    """Load PC scores, centroid sizes, and inter-landmark distances from GPA dirs."""
    data = {}
    for gpa_dir in gpa_dirs:
        base = Path(gpa_dir)
        if not base.exists():
            logger.warning(f"GPA directory not found: {base}")
            continue

        for sub in base.iterdir():
            if not sub.is_dir():
                continue

            cat_name = sub.name
            pc_path = sub / f"{cat_name}_pc_scores.csv"
            cs_path = sub / f"{cat_name}_centroid_sizes.csv"
            proc_path = sub / f"{cat_name}_procrustes_coords.csv"

            if pc_path.exists():
                df_pc = pd.read_csv(pc_path)
                fn_col = "filename" if "filename" in df_pc.columns else df_pc.columns[0]

                for _, row in df_pc.iterrows():
                    raw_fn = str(row[fn_col]).strip()
                    img_base = _normalize_image_filename(raw_fn)

                    features = {}
                    for col in df_pc.columns:
                        if col in (fn_col, "group"):
                            continue
                        val = row[col]
                        if pd.notna(val):
                            try:
                                features[f"lmk_{col}"] = float(val)
                            except (ValueError, TypeError):
                                pass

                    key = (img_base, cat_name)
                    data[key] = features

            if cs_path.exists():
                df_cs = pd.read_csv(cs_path)
                fn_col = "filename" if "filename" in df_cs.columns else df_cs.columns[0]

                for _, row in df_cs.iterrows():
                    raw_fn = str(row[fn_col]).strip()
                    img_base = _normalize_image_filename(raw_fn)

                    key = (img_base, cat_name)
                    if key not in data:
                        data[key] = {}

                    val = row.get("centroid_size")
                    if pd.notna(val):
                        data[key]["lmk_centroid_size"] = float(val)

            if proc_path.exists():
                df_proc = pd.read_csv(proc_path)
                fn_col = "filename" if "filename" in df_proc.columns else df_proc.columns[0]
                lm_x_cols = sorted([c for c in df_proc.columns if c.endswith("_x")],
                                   key=lambda c: int(c.replace("lm", "").replace("_x", "")))
                lm_y_cols = [c.replace("_x", "_y") for c in lm_x_cols]
                n_lm = len(lm_x_cols)

                for _, row in df_proc.iterrows():
                    raw_fn = str(row[fn_col]).strip()
                    img_base = _normalize_image_filename(raw_fn)
                    key = (img_base, cat_name)
                    if key not in data:
                        data[key] = {}

                    coords = []
                    for xc, yc in zip(lm_x_cols, lm_y_cols):
                        if pd.notna(row.get(xc)) and pd.notna(row.get(yc)):
                            coords.append((float(row[xc]), float(row[yc])))
                        else:
                            coords.append(None)

                    for i in range(n_lm):
                        for j in range(i + 1, n_lm):
                            if coords[i] is not None and coords[j] is not None:
                                dx = coords[i][0] - coords[j][0]
                                dy = coords[i][1] - coords[j][1]
                                dist = (dx * dx + dy * dy) ** 0.5
                                data[key][f"lmk_dist_{i+1}_{j+1}"] = dist

    logger.info(f"Landmark GPA: {len(data)} entries")
    return data


def load_inter_mask_distances(measurements_dir: str
                              ) -> Dict[str, Dict[Tuple[str, str], Dict]]:
    """Load inter_mask_distances.csv with calibrated (mm) distances preferred.

    Uses mm columns when available; falls back to px/scale conversion;
    drops uncalibrated pixel distances entirely (not comparable across images).
    """
    csv_path = Path(measurements_dir) / "inter_mask_distances.csv"
    if not csv_path.exists():
        logger.warning(f"Inter-mask distances CSV not found: {csv_path}")
        return {}

    scales = _load_scale_factors(measurements_dir)
    df = pd.read_csv(csv_path, low_memory=False)
    data = {}
    n_calibrated = 0

    for _, row in df.iterrows():
        img = _normalize_image_filename(str(row.get("image_filename", "")))
        if not img:
            continue

        cat_a = str(row.get("category_a", "")).strip().replace(" ", "_")
        cat_b = str(row.get("category_b", "")).strip().replace(" ", "_")
        pair = tuple(sorted([cat_a, cat_b]))

        if img not in data:
            data[img] = {}

        ppm = scales.get(img, 0)
        dists = {}

        for metric in ["centroid", "boundary"]:
            mm_col = f"{metric}_distance_mm" if metric == "centroid" else f"min_{metric}_distance_mm"
            px_col = f"{metric}_distance_px" if metric == "centroid" else f"min_{metric}_distance_px"
            mm_val = row.get(mm_col)
            px_val = row.get(px_col)

            if pd.notna(mm_val):
                dists[f"{metric}_mm"] = float(mm_val)
            elif pd.notna(px_val) and ppm > 0:
                dists[f"{metric}_mm"] = float(px_val) / ppm
            # Drop uncalibrated px values — not comparable across images

        dists["overlap"] = bool(row.get("overlap", False))
        if any(k.endswith("_mm") for k in dists):
            n_calibrated += 1

        if pair not in data[img] or dists.get("centroid_mm", 0) > 0:
            data[img][pair] = dists

    n_images = len(data)
    n_pairs = sum(len(v) for v in data.values())
    logger.info(f"Inter-mask distances: {n_pairs} pairs across {n_images} images "
                f"({n_calibrated} calibrated)")
    return data


def load_group_labels(group_labels_path: str) -> Dict[str, str]:
    """Load specimen → species group mapping."""
    path = Path(group_labels_path)
    if not path.exists():
        logger.warning(f"Group labels not found: {path}")
        return {}

    df = pd.read_csv(path)
    fn_col = df.columns[0]
    grp_col = df.columns[1] if len(df.columns) > 1 else "group_label"

    labels = {}
    for _, row in df.iterrows():
        fn = _normalize_image_filename(str(row[fn_col]).strip())
        labels[fn] = str(row[grp_col]).strip()

    logger.info(f"Group labels: {len(labels)} specimens, "
                f"{len(set(labels.values()))} groups")
    return labels


def load_exclude_list(exclude_path: str) -> set:
    """Load exclusion list CSV → set of (image_filename, category_name) tuples."""
    path = Path(exclude_path)
    if not path.exists():
        logger.warning(f"Exclude list not found: {path}")
        return set()

    df = pd.read_csv(path)
    excludes = set()
    for _, row in df.iterrows():
        fn = _normalize_image_filename(str(row["image_filename"]).strip())
        cat = str(row["category_name"]).strip().replace(" ", "_")
        excludes.add((fn, cat))

    logger.info(f"Exclude list: {len(excludes)} specimen×category pairs")
    return excludes


def load_coco_json(coco_path: str) -> Dict:
    """Load COCO JSON and build lookup dicts."""
    with open(coco_path) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"].replace(" ", "_") for c in coco.get("categories", [])}
    img_map = {}
    for img in coco.get("images", []):
        fn = _normalize_image_filename(img["file_name"])
        img_map[fn] = img

    ann_lookup = {}
    for ann in coco.get("annotations", []):
        img_entry = next((im for im in coco["images"] if im["id"] == ann["image_id"]), None)
        if img_entry is None:
            continue
        fn = _normalize_image_filename(img_entry["file_name"])
        cat = cat_map.get(ann["category_id"], "")
        ann_lookup[(fn, cat)] = ann

    logger.info(f"COCO JSON: {len(coco.get('images', []))} images, "
                f"{len(coco.get('annotations', []))} annotations")
    return coco, cat_map, img_map, ann_lookup


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE
# ═══════════════════════════════════════════════════════════════════════════════

def merge_all_pipelines(
    measurements: Dict, semilandmarks: Dict, color: Dict,
    color_homology: Dict, texture: Dict, landmarks: Dict,
    inter_mask_dists: Dict, group_labels: Dict, coco_ann_lookup: Dict,
    scale_factors: Dict[str, float] = None
) -> pd.DataFrame:
    """Merge all pipeline outputs into a single DataFrame.

    scale_factors: {image_base: pixels_per_mm} for calibrating pixel-based
    features from non-measurement pipelines (semilandmarks, color, etc.).
    """
    scale_factors = scale_factors or {}

    all_keys = set()
    pipelines = {
        "measurement": measurements,
        "shape": semilandmarks,
        "color": color,
        "color_homology": color_homology,
        "texture": texture,
        "landmark": landmarks,
    }
    for p_data in pipelines.values():
        all_keys.update(p_data.keys())

    rows = []
    for img_base, category in sorted(all_keys):
        row = {"image_base": img_base, "category": category}

        # Group label (join on image base — strip annotation suffix if needed)
        row["group_label"] = group_labels.get(img_base, "")
        if not row["group_label"]:
            ib_nodot = img_base.replace(".", "").replace(".tif", "")
            for gl_key, gl_val in group_labels.items():
                gl_stripped = gl_key.replace(".tif", "")
                gl_nodot = gl_key.replace(".", "").replace("tif", "")
                if img_base.startswith(gl_stripped) or \
                   gl_key.startswith(img_base.replace(".tif", "")) or \
                   ib_nodot.startswith(gl_nodot) or \
                   gl_nodot.startswith(ib_nodot):
                    row["group_label"] = gl_val
                    break

        # Coverage flags
        for pname, pdata in pipelines.items():
            row[f"has_{pname}"] = (img_base, category) in pdata
        img_dists_check = inter_mask_dists.get(img_base, {})
        row["has_inter_mask_dist"] = any(
            category in pair for pair in img_dists_check
        )

        # Merge features
        for pdata in pipelines.values():
            features = pdata.get((img_base, category), {})
            for k, v in features.items():
                if isinstance(v, (int, float)):
                    row[k] = v

        # Inter-mask distances: attach distances where this category is a member
        img_dists = inter_mask_dists.get(img_base, {})
        for (cat_a, cat_b), dists in img_dists.items():
            if cat_a == category:
                other = cat_b
            elif cat_b == category:
                other = cat_a
            else:
                continue
            for metric, val in dists.items():
                if metric == "overlap":
                    continue
                row[f"imd_to_{other}_{metric}"] = val

        # COCO annotation ID
        ann = coco_ann_lookup.get((img_base, category))
        if ann:
            row["coco_annotation_id"] = ann.get("id")
            row["coco_image_id"] = ann.get("image_id")

        rows.append(row)

    df = pd.DataFrame(rows)

    # --- Scale factor columns and pixel→mm calibration ---
    if scale_factors:
        df["scale_px_per_mm"] = df["image_base"].map(scale_factors)
        df["scale_px_per_um"] = df["scale_px_per_mm"] * 1000.0
        n_with_scale = df["scale_px_per_mm"].notna().sum()
        logger.info(f"Scale factors attached: {n_with_scale}/{len(df)} rows")

        _calibrate_pixel_features(df)

    logger.info(f"Merged: {len(df)} rows, {len(df.columns)} columns, "
                f"{df['category'].nunique()} categories")
    return df


# Pixel-based columns from non-measurement pipelines that need calibration.
# Format: (original_col_substring, calibration_type, new_suffix)
#   calibration_type: "linear" = divide by px/mm, "area" = divide by px/mm²
_PIXEL_CALIBRATION_RULES = [
    # Semilandmark centroid size (linear, in pixels)
    ("shape_centroid_size", "linear", "shape_centroid_size_mm"),
    # Landmark GPA centroid size (linear, in pixels)
    ("lmk_centroid_size", "linear", "lmk_centroid_size_mm"),
    # Color extraction marking areas (area, in pixels²)
    ("_total_marking_area", "area", None),
    ("_mean_marking_area", "area", None),
    ("_std_marking_area", "area", None),
]


def _calibrate_pixel_features(df: pd.DataFrame) -> None:
    """In-place calibration of pixel-based features using scale_px_per_mm.

    For each pixel feature, creates a calibrated _mm or _mm2 column and
    drops the raw pixel column so downstream never sees uncalibrated values.
    """
    ppm = df["scale_px_per_mm"]
    n_calibrated = 0

    for col in list(df.columns):
        cal_col = None
        cal_type = None

        # Exact match for known columns
        if col == "shape_centroid_size":
            cal_col = "shape_centroid_size_mm"
            cal_type = "linear"
        elif col == "lmk_centroid_size":
            cal_col = "lmk_centroid_size_mm"
            cal_type = "linear"
        elif col.endswith("_total_marking_area"):
            cal_col = col + "_mm2"
            cal_type = "area"
        elif col.endswith("_mean_marking_area"):
            cal_col = col + "_mm2"
            cal_type = "area"
        elif col.endswith("_std_marking_area"):
            cal_col = col + "_mm2"
            cal_type = "area"
        elif col.endswith("_n_boundaries_diff"):
            # Boundary-crossing pixel count ≈ boundary perimeter (linear)
            cal_col = col + "_mm"
            cal_type = "linear_keep"
        elif col.endswith("_n_boundaries_same"):
            # Same-label adjacency count ≈ area proxy (area)
            cal_col = col + "_mm2"
            cal_type = "area_keep"
        else:
            continue

        if cal_type == "linear":
            df[cal_col] = df[col] / ppm
        elif cal_type == "area":
            df[cal_col] = df[col] / (ppm * ppm)
        elif cal_type == "linear_keep":
            df[cal_col] = df[col] / ppm
        elif cal_type == "area_keep":
            df[cal_col] = df[col] / (ppm * ppm)

        if cal_type not in ("linear_keep", "area_keep"):
            df.drop(columns=[col], inplace=True)
        n_calibrated += 1

    if n_calibrated > 0:
        logger.info(f"Pixel→mm calibration: {n_calibrated} columns converted")


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiagnosticReport:
    all_tests: pd.DataFrame
    diagnostic_features: Dict[str, List[str]]
    pairwise: Dict[str, pd.DataFrame]
    summary_text: Dict[str, str]


def select_diagnostic_features(df: pd.DataFrame, group_col: str = "group_label",
                               alpha: float = 0.05,
                               min_group_size: int = 3
                               ) -> DiagnosticReport:
    """
    Per category, per numeric feature: Kruskal-Wallis test across species groups.
    FDR correction, effect size ranking, Dunn's post-hoc for top features.
    """
    meta_cols = {"image_base", "category", "group_label",
                 "coco_annotation_id", "coco_image_id",
                 "scale_px_per_mm", "scale_px_per_um"}
    coverage_cols = {c for c in df.columns if c.startswith("has_")}
    skip = meta_cols | coverage_cols

    test_rows = []
    diagnostic_features = {}
    pairwise_results = {}
    summary_texts = {}

    for cat, cat_df in df.groupby("category"):
        # Filter groups with enough specimens
        group_counts = cat_df[group_col].value_counts()
        valid_groups = group_counts[group_counts >= min_group_size].index.tolist()
        if len(valid_groups) < 2:
            continue

        cat_filtered = cat_df[cat_df[group_col].isin(valid_groups)].copy()
        n_total = len(cat_filtered)
        k_groups = len(valid_groups)

        numeric_cols = [c for c in cat_filtered.columns
                        if c not in skip and pd.api.types.is_numeric_dtype(cat_filtered[c])]

        cat_tests = []
        for feat in numeric_cols:
            col_data = cat_filtered[[group_col, feat]].dropna()
            if len(col_data) < k_groups * min_group_size:
                continue

            groups = [g[feat].values for _, g in col_data.groupby(group_col)]
            groups = [g for g in groups if len(g) >= min_group_size]
            if len(groups) < 2:
                continue

            try:
                h_stat, p_val = kruskal(*groups)
            except Exception:
                continue

            n = sum(len(g) for g in groups)
            k = len(groups)
            eta_sq = (h_stat - k + 1) / (n - k) if n > k else 0.0
            eta_sq = max(0.0, min(1.0, eta_sq))

            cat_tests.append({
                "category": cat,
                "feature": feat,
                "H_stat": round(h_stat, 4),
                "p_value": p_val,
                "eta_squared": round(eta_sq, 4),
                "n_specimens": n,
                "n_groups": k,
            })

        if not cat_tests:
            continue

        cat_test_df = pd.DataFrame(cat_tests)

        # Benjamini-Hochberg FDR correction
        p_vals = cat_test_df["p_value"].values
        n_tests = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        q_vals = np.zeros(n_tests)
        for rank_i, orig_i in enumerate(sorted_idx):
            bh_critical = (rank_i + 1) / n_tests * alpha
            q_vals[orig_i] = p_vals[orig_i] * n_tests / (rank_i + 1)
        q_vals = np.minimum.accumulate(q_vals[np.argsort(sorted_idx)][::-1])[::-1]
        q_vals = np.clip(q_vals, 0, 1)
        # Reorder to original
        q_corrected = np.zeros(n_tests)
        for i, idx in enumerate(sorted_idx):
            q_corrected[idx] = q_vals[i]

        cat_test_df["q_value"] = q_corrected
        cat_test_df["is_diagnostic"] = cat_test_df["q_value"] < alpha

        # Sort by effect size
        cat_test_df = cat_test_df.sort_values("eta_squared", ascending=False)
        test_rows.append(cat_test_df)

        diag = cat_test_df[cat_test_df["is_diagnostic"]]
        diagnostic_features[cat] = diag["feature"].tolist()

        # Dunn's post-hoc for top diagnostic features (max 10)
        if HAS_POSTHOCS and len(diag) > 0:
            top_feats = diag.head(10)["feature"].tolist()
            pairwise_dfs = []
            for feat in top_feats:
                col_data = cat_filtered[[group_col, feat]].dropna()
                try:
                    pw = sp.posthoc_dunn(col_data, val_col=feat,
                                         group_col=group_col, p_adjust="fdr_bh")
                    pw_melted = pw.reset_index().melt(id_vars="index")
                    pw_melted.columns = ["group_a", "group_b", "p_value_dunn"]
                    pw_melted["feature"] = feat
                    pw_melted = pw_melted[pw_melted["group_a"] < pw_melted["group_b"]]
                    pairwise_dfs.append(pw_melted)
                except Exception:
                    pass
            if pairwise_dfs:
                pairwise_results[cat] = pd.concat(pairwise_dfs, ignore_index=True)

        # Summary text
        n_diag = len(diag)
        if n_diag > 0:
            top3 = diag.head(3)
            feat_strs = []
            for _, r in top3.iterrows():
                feat_strs.append(
                    f"{r['feature']} (H={r['H_stat']:.1f}, η²={r['eta_squared']:.3f}, "
                    f"q={r['q_value']:.4f})"
                )
            summary_texts[cat] = (
                f"{n_diag} diagnostic features. Top: {'; '.join(feat_strs)}"
            )

    all_tests = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
    n_diag_total = all_tests["is_diagnostic"].sum() if len(all_tests) > 0 else 0
    n_cats = len(diagnostic_features)

    logger.info(f"Diagnostic selection: {n_diag_total} significant features "
                f"across {n_cats} categories (alpha={alpha}, FDR corrected)")

    return DiagnosticReport(
        all_tests=all_tests,
        diagnostic_features=diagnostic_features,
        pairwise=pairwise_results,
        summary_text=summary_texts,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def write_full_csv(df: pd.DataFrame, output_dir: Path):
    path = output_dir / "diaphorina_full_features.csv"
    df.to_csv(path, index=False, float_format="%.8g")
    logger.info(f"Full CSV: {path} ({len(df)} rows, {len(df.columns)} columns)")


def write_diagnostic_csv(df: pd.DataFrame, report: DiagnosticReport, output_dir: Path):
    if report.all_tests.empty:
        logger.warning("No diagnostic tests — skipping diagnostic CSV")
        return

    path = output_dir / "diaphorina_diagnostic_features.csv"
    report.all_tests.to_csv(path, index=False, float_format="%.8g")
    logger.info(f"Diagnostic CSV: {path} ({len(report.all_tests)} tests)")

    # Pairwise results
    pw_path = output_dir / "diaphorina_pairwise_tests.csv"
    if report.pairwise:
        all_pw = pd.concat(
            [pw.assign(category=cat) for cat, pw in report.pairwise.items()],
            ignore_index=True
        )
        all_pw.to_csv(pw_path, index=False, float_format="%.8g")
        logger.info(f"Pairwise tests: {pw_path} ({len(all_pw)} comparisons)")


def write_diagnostic_report_json(report: DiagnosticReport, output_dir: Path):
    path = output_dir / "diaphorina_diagnostic_report.json"

    report_dict = {
        "generated": datetime.now().isoformat(),
        "method": "Kruskal-Wallis H-test + Benjamini-Hochberg FDR + Dunn's post-hoc",
        "categories": {},
    }

    for cat, features in report.diagnostic_features.items():
        cat_tests = report.all_tests[report.all_tests["category"] == cat]
        diag_tests = cat_tests[cat_tests["is_diagnostic"]].to_dict("records")

        cat_entry = {
            "n_diagnostic": len(features),
            "n_tested": len(cat_tests),
            "diagnostic_features": features,
            "tests": diag_tests,
            "summary": report.summary_text.get(cat, ""),
        }

        if cat in report.pairwise:
            pw = report.pairwise[cat]
            sig_pairs = pw[pw["p_value_dunn"] < 0.05]
            cat_entry["pairwise_significant"] = sig_pairs.to_dict("records")

        report_dict["categories"][cat] = cat_entry

    with open(path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    logger.info(f"Diagnostic report JSON: {path}")


def write_npy_matrix(df: pd.DataFrame, report: DiagnosticReport, output_dir: Path):
    meta_cols = {"image_base", "category", "group_label",
                 "coco_annotation_id", "coco_image_id"}
    coverage_cols = {c for c in df.columns if c.startswith("has_")}
    skip = meta_cols | coverage_cols

    # Use diagnostic features only if available
    if not report.all_tests.empty:
        diag_feats = set()
        for feats in report.diagnostic_features.values():
            diag_feats.update(feats)
        num_cols = [c for c in df.columns if c in diag_feats]
    else:
        num_cols = [c for c in df.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    if not num_cols:
        logger.warning("No numeric columns for NPY matrix")
        return

    matrix = df[num_cols].values.astype(np.float64)
    np.save(output_dir / "diaphorina_diagnostic_matrix.npy", matrix)

    with open(output_dir / "diaphorina_feature_names.json", "w") as f:
        json.dump(num_cols, f, indent=2)

    row_index = df[["image_base", "category"]].to_dict("records")
    with open(output_dir / "diaphorina_row_index.json", "w") as f:
        json.dump(row_index, f, indent=2)

    logger.info(f"NPY matrix: {matrix.shape} (features: {len(num_cols)})")


def write_coco_jsonld(df: pd.DataFrame, report: DiagnosticReport,
                      coco: Dict, coco_ann_lookup: Dict,
                      semilandmarks_dir: str, output_dir: Path):
    """Extend COCO JSON with JSON-LD context and diagnostic traits."""
    import copy
    enriched = copy.deepcopy(coco)

    enriched["@context"] = {
        "@vocab": "https://descriptron.org/ontology/",
        "dwc": "http://rs.tdwg.org/dwc/terms/",
        "uberon": "http://purl.obolibrary.org/obo/UBERON_",
        "hao": "http://purl.obolibrary.org/obo/HAO_",
        "aism": "http://purl.obolibrary.org/obo/AISM_",
        "pato": "http://purl.obolibrary.org/obo/PATO_",
    }

    enriched["generated"] = datetime.now().isoformat()
    enriched["diagnostic_method"] = "Kruskal-Wallis + BH-FDR + Dunn's post-hoc"

    # Build annotation ID → traits mapping
    ann_id_to_traits = {}
    diag_lookup = {}
    if not report.all_tests.empty:
        for _, row in report.all_tests[report.all_tests["is_diagnostic"]].iterrows():
            key = (row["category"], row["feature"])
            diag_lookup[key] = {
                "H_stat": row["H_stat"],
                "p_value": row["p_value"],
                "q_value": row["q_value"],
                "eta_squared": row["eta_squared"],
            }

    for _, specimen in df.iterrows():
        img_base = specimen["image_base"]
        cat = specimen["category"]
        ann = coco_ann_lookup.get((img_base, cat))
        if ann is None:
            continue

        ann_id = ann["id"]
        traits = []

        for col in df.columns:
            if col.startswith(("meas_", "shape_", "color_", "colhom_", "tex_", "lmk_", "imd_")):
                val = specimen.get(col)
                if pd.isna(val):
                    continue

                pipeline = col.split("_")[0]
                feature_name = "_".join(col.split("_")[1:])
                is_diag = (cat, col) in diag_lookup
                stats = diag_lookup.get((cat, col), {})

                trait = {
                    "traitType": feature_name,
                    "value": round(float(val), 6),
                    "pipeline": pipeline,
                    "is_diagnostic": is_diag,
                }
                if stats:
                    trait.update(stats)
                traits.append(trait)

        # Foreground mask path
        fg_mask = None
        if semilandmarks_dir:
            fg_pattern = Path(semilandmarks_dir) / cat / cat
            candidates = list(fg_pattern.glob(f"*{img_base.replace('.tif', '')}*_fg_{cat}.png"))
            if candidates:
                fg_mask = str(candidates[0])

        ann_id_to_traits[ann_id] = {
            "traits": traits,
            "foreground_mask_path": fg_mask,
            "group_label": specimen.get("group_label", ""),
        }

    for ann in enriched.get("annotations", []):
        extra = ann_id_to_traits.get(ann["id"], {})
        ann.update(extra)

    path = output_dir / "diaphorina_compiled.jsonld"
    with open(path, "w") as f:
        json.dump(enriched, f, indent=2, default=str)
    logger.info(f"COCO JSON-LD: {path}")


def write_specimen_summaries(df: pd.DataFrame, report: DiagnosticReport,
                             output_dir: Path):
    """Per-image JSONs with only diagnostic features, for VLM prompt injection."""
    summaries_dir = output_dir / "specimen_summaries"
    summaries_dir.mkdir(exist_ok=True)

    # Precompute group means/stds for diagnostic features
    group_stats = {}
    for cat, features in report.diagnostic_features.items():
        cat_df = df[df["category"] == cat]
        for feat in features:
            if feat not in cat_df.columns:
                continue
            for grp, grp_df in cat_df.groupby("group_label"):
                vals = grp_df[feat].dropna()
                if len(vals) > 0:
                    group_stats[(cat, feat, grp)] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std()),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "n": len(vals),
                    }

    # Pairwise significance lookup
    pw_sig = {}
    for cat, pw_df in report.pairwise.items():
        for _, row in pw_df.iterrows():
            if row["p_value_dunn"] < 0.05:
                key = (cat, row["feature"], row["group_a"], row["group_b"])
                pw_sig[key] = row["p_value_dunn"]

    # Group by image
    written = 0
    for img_base, img_df in df.groupby("image_base"):
        specimen = {
            "image": img_base,
            "group_label": img_df["group_label"].iloc[0],
            "categories": {},
        }

        for _, row in img_df.iterrows():
            cat = row["category"]
            diag_feats = report.diagnostic_features.get(cat, [])
            if not diag_feats:
                continue

            cat_entry = []
            for rank, feat in enumerate(diag_feats, 1):
                val = row.get(feat)
                if pd.isna(val):
                    continue

                grp = row["group_label"]
                stats = group_stats.get((cat, feat, grp), {})

                # Build comparison text
                comparisons = []
                for (c, f, ga, gb), p in pw_sig.items():
                    if c == cat and f == feat and ga == grp:
                        comparisons.append(f"differs from {gb} (p={p:.4f})")
                    elif c == cat and f == feat and gb == grp:
                        comparisons.append(f"differs from {ga} (p={p:.4f})")

                entry = {
                    "feature": feat,
                    "value": round(float(val), 6),
                    "rank": rank,
                    "group_mean": round(stats.get("mean", 0), 6),
                    "group_std": round(stats.get("std", 0), 6),
                    "group_min": round(stats.get("min", 0), 6),
                    "group_max": round(stats.get("max", 0), 6),
                    "group_n": stats.get("n", 0),
                }
                if comparisons:
                    entry["comparisons"] = comparisons

                cat_entry.append(entry)

            if cat_entry:
                specimen["categories"][cat] = cat_entry

        if specimen["categories"]:
            stem = Path(img_base).stem
            with open(summaries_dir / f"{stem}.json", "w") as f:
                json.dump(specimen, f, indent=2)
            written += 1

    logger.info(f"Specimen summaries: {written} files in {summaries_dir}")


def write_species_summary_table(df: pd.DataFrame, report: DiagnosticReport,
                                output_dir: Path):
    """Per-species, per-category summary with mean, std, min, max, n, CV for all features."""
    meta_cols = {"image_base", "category", "group_label",
                 "coco_annotation_id", "coco_image_id"}
    coverage_cols = {c for c in df.columns if c.startswith("has_")}
    skip = meta_cols | coverage_cols

    numeric_cols = [c for c in df.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        return

    # Build diagnostic feature lookup set
    diag_set = set()
    for cat, feats in report.diagnostic_features.items():
        for f in feats:
            diag_set.add((cat, f))

    # Vectorized: group by (group_label, category), aggregate numeric cols
    grouped = df[df["group_label"].notna() & (df["group_label"] != "")].groupby(
        ["group_label", "category"]
    )[numeric_cols]

    agg = grouped.agg(["mean", "std", "min", "max", "count"])

    rows = []
    for (grp, cat), stats in agg.iterrows():
        for feat in numeric_cols:
            n = int(stats[(feat, "count")])
            if n == 0:
                continue
            mean_val = float(stats[(feat, "mean")])
            std_val = float(stats[(feat, "std")]) if n > 1 else 0.0
            if pd.isna(mean_val):
                continue
            cv = (std_val / abs(mean_val) * 100) if mean_val != 0 else 0.0
            rows.append({
                "group_label": grp,
                "category": cat,
                "feature": feat,
                "n": n,
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "min": round(float(stats[(feat, "min")]), 6),
                "max": round(float(stats[(feat, "max")]), 6),
                "cv_percent": round(cv, 2),
                "is_diagnostic": (cat, feat) in diag_set,
            })

    if rows:
        summary_df = pd.DataFrame(rows)
        path = output_dir / "diaphorina_species_summary.csv"
        summary_df.to_csv(path, index=False, float_format="%.8g")
        logger.info(f"Species summary table: {path} "
                    f"({len(summary_df)} feature × species × category rows)")


def write_coverage_matrix(df: pd.DataFrame, output_dir: Path):
    coverage_cols = [c for c in df.columns if c.startswith("has_")]
    if not coverage_cols:
        return

    cov_df = df[["image_base", "category"] + coverage_cols].copy()
    path = output_dir / "coverage_matrix.csv"
    cov_df.to_csv(path, index=False)

    # Summary
    for col in coverage_cols:
        pct = cov_df[col].mean() * 100
        logger.info(f"  {col}: {pct:.1f}% coverage")

    logger.info(f"Coverage matrix: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_ratios_for_df(df: pd.DataFrame, ratio_config: str = None,
                           species_pattern: str = None,
                           output_dir: Path = None) -> pd.DataFrame:
    """Compute taxonomic ratios as new columns, matched at the specimen level.

    Ratios are cross-body-part (e.g., antenna length / head width requires
    measurements from different images of the same specimen). This function
    extracts a specimen_id from image_base, pivots measurements by category,
    computes ratios, and joins them back to the DataFrame.
    """
    try:
        from compute_proportions import build_default_config
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from compute_proportions import build_default_config

    abbrevs, ratios = build_default_config()

    specimen_col = "_specimen_id"
    sex_col = "_sex"

    def _extract_specimen(image_base):
        if species_pattern:
            m = re.search(rf'({species_pattern})_(\d+)', image_base)
        else:
            m = re.search(r'([a-zA-Z][a-zA-Z0-9_.]+?)_(\d+)_', image_base)
        return f'{m.group(1)}_{m.group(2)}' if m else None

    def _extract_sex(image_base):
        if '_m_' in image_base:
            return 'male'
        elif '_f_' in image_base:
            return 'female'
        return 'unknown'

    df[specimen_col] = df['image_base'].apply(_extract_specimen)
    df[sex_col] = df['image_base'].apply(_extract_sex)

    n_matched = df[specimen_col].notna().sum()
    if n_matched == 0:
        logger.warning("No specimen IDs extracted — skipping ratio computation")
        df.drop(columns=[specimen_col, sex_col], inplace=True)
        return df

    logger.info(f"Ratio computation: {n_matched}/{len(df)} rows matched to "
                f"{df[specimen_col].nunique()} specimens")

    spec_data = {}
    for _, row in df.iterrows():
        spec = row[specimen_col]
        if pd.isna(spec):
            continue
        cat = row['category']
        if spec not in spec_data:
            spec_data[spec] = {'sex': row[sex_col], 'cats': {}}
        for col_suffix, meas_col in [('length_mm', 'meas_length_mm'),
                                      ('height_mm', 'meas_height_mm')]:
            val = row.get(meas_col)
            if pd.notna(val) and val > 0:
                if cat not in spec_data[spec]['cats']:
                    spec_data[spec]['cats'][cat] = {}
                spec_data[spec]['cats'][cat][col_suffix] = val

    def _get_val(spec, abbrev):
        cfg = abbrevs.get(abbrev)
        if not cfg:
            return None
        cat = cfg['category']
        col = cfg['measurement']
        return spec_data.get(spec, {}).get('cats', {}).get(cat, {}).get(col)

    ratio_rows = {}
    for spec_id, sdata in spec_data.items():
        sex = sdata['sex']
        row_ratios = {}
        for ratio in ratios:
            rid = ratio['id']
            rsex = ratio.get('sex', 'all')
            if rsex != 'all' and sex != rsex:
                continue
            num_key = ratio['num']
            den_key = ratio['den']
            if '+' in num_key:
                parts = num_key.split('+')
                vals = [_get_val(spec_id, p.strip()) for p in parts]
                num_val = sum(vals) if all(v is not None for v in vals) else None
            else:
                num_val = _get_val(spec_id, num_key)
            den_val = _get_val(spec_id, den_key)
            if num_val is not None and den_val is not None and den_val > 0:
                row_ratios[f'ratio_{rid}'] = round(num_val / den_val, 6)
        ratio_rows[spec_id] = row_ratios

    n_ratios_added = 0
    for col_name in set().union(*(r.keys() for r in ratio_rows.values())):
        df[col_name] = df[specimen_col].map(
            lambda s, cn=col_name: ratio_rows.get(s, {}).get(cn))
        n_ratios_added += 1

    logger.info(f"  Added {n_ratios_added} ratio columns")

    if output_dir:
        ratio_summary = []
        for spec_id in sorted(ratio_rows.keys()):
            entry = {'specimen_id': spec_id, 'sex': spec_data[spec_id]['sex']}
            entry.update(ratio_rows[spec_id])
            ratio_summary.append(entry)
        if ratio_summary:
            rdf = pd.DataFrame(ratio_summary)
            rdf.to_csv(output_dir / "ratio_summary.csv", index=False)
            logger.info(f"  Ratio summary: {output_dir / 'ratio_summary.csv'}")

    df.drop(columns=[specimen_col, sex_col], inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compile Diaphorina pipeline outputs into unified formats for BioRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--measurements_dir", type=str, required=True)
    parser.add_argument("--semilandmarks_dir", type=str, required=True)
    parser.add_argument("--color_dirs", type=str, nargs="+", required=True,
                        help="Color extraction dir(s) — body parts and/or heads")
    parser.add_argument("--color_homology_dirs", type=str, nargs="+", required=True,
                        help="Color homology dir(s) — body parts and/or heads")
    parser.add_argument("--texture_dirs", type=str, nargs="+", required=True,
                        help="Texture homology dir(s) — body parts and/or heads")
    parser.add_argument("--landmark_gpa_dirs", type=str, nargs="+", default=[])
    parser.add_argument("--coco_json", type=str, required=True)
    parser.add_argument("--group_labels", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold for diagnostic features (default: 0.05)")
    parser.add_argument("--min_group_size", type=int, default=3,
                        help="Minimum specimens per group for statistical tests (default: 3)")
    parser.add_argument("--exclude_list", type=str, default=None,
                        help="CSV of outlier specimens to exclude (image_filename, category_name, ...)")
    parser.add_argument("--ratio_config", type=str, default=None,
                        help="TSV defining taxonomic ratios (abbreviations + ratio definitions)")
    parser.add_argument("--species_pattern", type=str, default=None,
                        help="Regex for extracting species from filenames (for ratio computation)")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Diaphorina Pipeline Data Compilation")
    logger.info("=" * 60)

    # Load all pipelines
    measurements = load_measurements(args.measurements_dir)
    semilandmarks = load_semilandmarks(args.semilandmarks_dir)
    color = load_color_extraction(args.color_dirs)
    color_homology = load_color_homology(args.color_homology_dirs)
    texture = load_texture_homology(args.texture_dirs)
    landmarks = load_landmark_gpa(args.landmark_gpa_dirs)
    inter_mask_dists = load_inter_mask_distances(args.measurements_dir)
    group_labels = load_group_labels(args.group_labels)
    coco, cat_map, img_map, ann_lookup = load_coco_json(args.coco_json)
    scale_factors = _load_scale_factors(args.measurements_dir)

    # Merge
    df = merge_all_pipelines(
        measurements, semilandmarks, color, color_homology,
        texture, landmarks, inter_mask_dists, group_labels, ann_lookup,
        scale_factors=scale_factors
    )

    # Compute taxonomic ratios (cross-body-part proportions)
    if args.ratio_config or args.species_pattern:
        df = _compute_ratios_for_df(df, args.ratio_config, args.species_pattern,
                                    output_dir)

    # Apply outlier exclusion list
    if args.exclude_list:
        excludes = load_exclude_list(args.exclude_list)
        if excludes:
            before = len(df)
            mask = df.apply(
                lambda r: (r.get("image_base", ""), r.get("category", "")) not in excludes,
                axis=1,
            )
            df = df[mask].reset_index(drop=True)
            n_removed = before - len(df)
            logger.info(f"Excluded {n_removed} rows ({before} → {len(df)})")

    # Diagnostic feature selection
    report = select_diagnostic_features(
        df, alpha=args.alpha, min_group_size=args.min_group_size
    )

    # Write all outputs
    write_full_csv(df, output_dir)
    write_diagnostic_csv(df, report, output_dir)
    write_diagnostic_report_json(report, output_dir)
    write_npy_matrix(df, report, output_dir)
    write_coco_jsonld(df, report, coco, ann_lookup, args.semilandmarks_dir, output_dir)
    write_specimen_summaries(df, report, output_dir)
    write_species_summary_table(df, report, output_dir)
    write_coverage_matrix(df, output_dir)

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Compilation Complete!")
    logger.info("=" * 60)
    logger.info(f"  Specimens × categories: {len(df)}")
    logger.info(f"  Total features: {len(df.columns)}")
    n_diag = sum(len(v) for v in report.diagnostic_features.values())
    logger.info(f"  Diagnostic features: {n_diag} across "
                f"{len(report.diagnostic_features)} categories")
    logger.info(f"  Output: {output_dir}")

    for cat in sorted(report.summary_text.keys()):
        logger.info(f"  [{cat}] {report.summary_text[cat]}")


if __name__ == "__main__":
    main()
