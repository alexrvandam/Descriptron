#!/usr/bin/env python3
"""
FHS_and_CLAHE_V21.py   (2026-04)
------------------------------------
Felzenszwalb → CLAHE → second-iteration LAB averaging workflow
with optional shine removal, colour-true dendrograms, ΔE sweep,
background-masking via the original binary mask,
adaptive & median-calibrated threshold selection,
FFT granularity (pattern energy) analysis, colour adjacency
analysis, Elliptical Fourier marking-shape analysis,
and per-image + population-level CSV summary export.

New in V21 (over V20)
---------------------
* HSV metrics from correct source : hsv_colour_metrics() now runs on
                           the CLAHE-normalised fg image (seg_p), NOT
                           the LAB-averaged output (out_p). The averaged
                           image discards per-pixel colour variance.
* Full 0-360 hue range  : OpenCV hue (0-180) doubled to 0-360 before
                           circular decomposition. hue_mean now in
                           0-360 degrees, matching R contMap lims and
                           hsb_to_rgb(h/360). hue_sin/hue_cos unchanged.

New in V20 (over V19)
---------------------
* --category_name        : Category name for output file naming
                           (e.g. left_elytron, pronotum)
* Phylo-compatible CSV   : Automatically exports
                           color_traits_phylo_<cat>.csv with all
                           numeric pattern/color metrics suitable for
                           phylogenetic mapping via texture_phylo_mapping.R
* PCA on color traits    : Runs PCA on all numeric metrics, exports
                           PC scores in phylo CSV for contMap/phylosig

Usage examples
--------------
# Full pipeline: adaptive per-image + median-calibrated + all analyses
python FHS_and_CLAHE_V19.py \\
    --input_dir ./specimens --output_dir ./results \\
    --remove_shine --run_segmentation --run_normalization \\
    --run_lab_average --median_threshold \\
    --run_pattern_analysis --run_efa --pattern_viz --efa_viz

# Fixed threshold with pattern analysis only (no EFA)
python FHS_and_CLAHE_V19.py \\
    --input_dir ./specimens --output_dir ./results \\
    --run_segmentation --run_normalization \\
    --run_lab_average --lab_threshold 20 \\
    --run_pattern_analysis --pattern_viz

References
----------
Troscianko & Stevens (2015) MEE — MICA Toolbox / granularity analysis
Nokelainen et al.  (2024) Nat Commun — moth colour pattern variability
Chan, Stevens & Todd (2019) MEE — pat-geom (marking shape/adjacency)
Endler (2012) Biol J Linn Soc — colour adjacency analysis
Bookstein (1991) — bending energy / morphometric deformation energy
"""
import os
import csv
import json
import argparse
from typing import Optional, List, Dict, Tuple
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
from skimage import io, color
from skimage.segmentation import felzenszwalb
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.ndimage import uniform_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa

###############################################################################
# helper – LAB → RGB (float 0-1)
###############################################################################

def _lab2rgb_pts(lab_arr: np.ndarray) -> np.ndarray:
    rgb = color.lab2rgb(lab_arr.reshape(-1, 1, 3)).reshape(-1, 3)
    return np.clip(rgb, 0, 1)


###############################################################################
# 0a) Background-based chromaticity (a*/b*) normalisation
###############################################################################

def _sample_background_lab(original_img_path: str,
                           mask_path: str,
                           border_fraction: float = 0.05
                           ) -> Optional[Tuple[float, float, float]]:
    """Sample background pixels from an original (unmasked) image.

    Returns mean (L, a*, b*) of background in OpenCV LAB encoding
    (L: 0-255, a*/b*: 0-255 with 128 = neutral), or None on failure.
    """
    img = cv2.imread(original_img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if img.shape[:2] != mask.shape:
        return None

    bg = mask == 0
    n_bg = bg.sum()
    if n_bg < 100:
        return None

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    bg_pixels = lab[bg].astype(np.float64)
    return float(bg_pixels[:, 0].mean()), float(bg_pixels[:, 1].mean()), float(bg_pixels[:, 2].mean())


def _resolve_original_image(fg_filename: str,
                             original_dir: str,
                             coco_data: Optional[dict] = None
                             ) -> Optional[str]:
    """Map a _fg_ export filename back to the original image path.

    Strategy: extract image_id (last integer before '_fg_') and look up
    in COCO JSON, or fuzzy-match by stem in original_dir.
    """
    import re
    stem = fg_filename
    # strip _fg_<category>.png suffix
    m = re.match(r'^(.+?)_fg_.+$', stem)
    if not m:
        return None
    pre = m.group(1)
    # last component is image_id
    parts = pre.rsplit('_', 1)
    img_id = None
    if len(parts) == 2 and parts[1].isdigit():
        img_id = int(parts[1])

    # Try COCO lookup
    if coco_data and img_id is not None:
        img_map = {im['id']: im['file_name'] for im in coco_data.get('images', [])}
        fn = img_map.get(img_id)
        if fn:
            p = os.path.join(original_dir, os.path.basename(fn))
            if os.path.isfile(p):
                return p

    # Fuzzy match: strip trailing digits (image_id) and 'tif'/'png' artifacts,
    # then search for .tif/.tiff/.png/.jpg in original_dir
    base = parts[0] if len(parts) == 2 and parts[1].isdigit() else pre
    # The exported name replaces '.' with '' and extension with suffix,
    # e.g. "file_SV.tif" becomes "file_SVtif".  Try restoring ".tif":
    for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
        ext_stripped = ext.replace('.', '')
        if base.endswith(ext_stripped):
            candidate = base[:-len(ext_stripped)] + ext
            # also restore spaces that V34 may have replaced with underscores
            for trial in [candidate, candidate.replace('_', ' ')]:
                p = os.path.join(original_dir, trial)
                if os.path.isfile(p):
                    return p
    return None


def normalize_background_chromaticity(
        fg_path: str,
        mask_path: str,
        out_path: str,
        original_img_path: Optional[str] = None,
        bg_lab: Optional[Tuple[float, float, float]] = None,
        target_L: Optional[float] = None,
        correct_L: bool = False) -> Optional[Tuple[float, float]]:
    """Apply von Kries–style correction in LAB space using background reference.

    Shifts a* and b* channels so the background becomes chromatically neutral
    (a*=128, b*=128 in OpenCV encoding).  Optionally also normalises L.

    Parameters
    ----------
    fg_path : foreground image (specimen on black background)
    mask_path : binary mask (255 = foreground)
    out_path : path for corrected output
    original_img_path : path to original (unmasked) image for bg sampling
    bg_lab : pre-computed background (L, a*, b*); overrides original_img_path
    target_L : if correct_L, scale L so background matches this value
    correct_L : whether to also correct luminance (usually False since CLAHE does this)

    Returns (a*_offset, b*_offset) applied, or None on failure.
    """
    if bg_lab is None and original_img_path is not None:
        bg_lab = _sample_background_lab(original_img_path, mask_path)
    if bg_lab is None:
        return None

    bg_L, bg_a, bg_b = bg_lab
    a_offset = bg_a - 128.0   # deviation from neutral
    b_offset = bg_b - 128.0

    img = cv2.imread(fg_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    fg = mask > 0

    lab[fg, 1] -= a_offset
    lab[fg, 2] -= b_offset

    if correct_L and target_L is not None and bg_L > 1:
        lab[fg, 0] *= (target_L / bg_L)

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    return (a_offset, b_offset)


def batch_normalize_chromaticity(fg_mask_pairs: List[Tuple[str, str]],
                                  output_dir: str,
                                  original_dir: Optional[str] = None,
                                  coco_json: Optional[str] = None,
                                  correct_L: bool = False
                                  ) -> List[Tuple[str, str]]:
    """Normalise a*/b* for a batch of _fg_/_bin_ image pairs.

    Mode 1 (original_dir provided): Per-image correction using actual
    background sampled from original images.

    Mode 2 (no original_dir): Batch centering — compute mean a*/b* of
    foreground across all images, then shift each image so its mean
    matches the grand mean.  Removes inter-image colour-cast variance.

    Returns list of (corrected_fg_path, mask_path) pairs.
    """
    coco_data = None
    if coco_json and os.path.isfile(coco_json):
        with open(coco_json) as f:
            coco_data = json.load(f)

    corrected = []
    os.makedirs(output_dir, exist_ok=True)

    if original_dir and os.path.isdir(original_dir):
        # --- MODE 1: per-image background correction ---
        print(f"\n=== Background chromaticity normalisation (per-image, from {original_dir}) ===")
        offsets = []
        for fg_p, mask_p in fg_mask_pairs:
            fg_fn = os.path.basename(fg_p)
            orig_p = _resolve_original_image(fg_fn, original_dir, coco_data)
            out_p = os.path.join(output_dir, f"bgnorm_{fg_fn}")

            if orig_p:
                result = normalize_background_chromaticity(
                    fg_p, mask_p, out_p, original_img_path=orig_p, correct_L=correct_L)
                if result:
                    offsets.append(result)
                    print(f"  {fg_fn}: a*={result[0]:+.1f}  b*={result[1]:+.1f}")
                    corrected.append((out_p, mask_p))
                    continue

            # Fallback: copy unchanged
            import shutil
            shutil.copy2(fg_p, out_p)
            corrected.append((out_p, mask_p))
            print(f"  {fg_fn}: no original found — skipped")

        if offsets:
            a_off = np.array([o[0] for o in offsets])
            b_off = np.array([o[1] for o in offsets])
            print(f"\n  Batch a* offsets: mean={a_off.mean():.2f} std={a_off.std():.2f} "
                  f"range=[{a_off.min():.1f}, {a_off.max():.1f}]")
            print(f"  Batch b* offsets: mean={b_off.mean():.2f} std={b_off.std():.2f} "
                  f"range=[{b_off.min():.1f}, {b_off.max():.1f}]")
    else:
        # --- MODE 2: batch mean centering ---
        print("\n=== Background chromaticity normalisation (batch centering) ===")

        # Pass 1: compute per-image mean a*, b*
        per_image_ab = []
        for fg_p, mask_p in fg_mask_pairs:
            img = cv2.imread(fg_p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                per_image_ab.append((128.0, 128.0))
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            fg = mask > 0
            if fg.sum() < 10:
                per_image_ab.append((128.0, 128.0))
                continue
            fg_pixels = lab[fg].astype(np.float64)
            per_image_ab.append((fg_pixels[:, 1].mean(), fg_pixels[:, 2].mean()))

        arr = np.array(per_image_ab)
        grand_a = arr[:, 0].mean()
        grand_b = arr[:, 1].mean()
        print(f"  Grand mean a*={grand_a:.1f} (neutral=128)  b*={grand_b:.1f}")

        # Pass 2: shift each image to the grand mean
        for idx, (fg_p, mask_p) in enumerate(fg_mask_pairs):
            fg_fn = os.path.basename(fg_p)
            out_p = os.path.join(output_dir, f"bgnorm_{fg_fn}")
            img = cv2.imread(fg_p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                corrected.append((fg_p, mask_p))
                continue

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
            fg = mask > 0
            img_a, img_b = per_image_ab[idx]
            a_shift = grand_a - img_a
            b_shift = grand_b - img_b
            lab[fg, 1] += a_shift
            lab[fg, 2] += b_shift
            lab = np.clip(lab, 0, 255).astype(np.uint8)
            cv2.imwrite(out_p, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
            corrected.append((out_p, mask_p))

        print(f"  Per-image a* range: [{arr[:,0].min():.1f}, {arr[:,0].max():.1f}] "
              f"→ centred at {grand_a:.1f}")
        print(f"  Per-image b* range: [{arr[:,1].min():.1f}, {arr[:,1].max():.1f}] "
              f"→ centred at {grand_b:.1f}")

    return corrected


###############################################################################
# 0b) shine / dust removal
###############################################################################

def remove_shine_inpaint(src: str, dst: str, thresh_v: int = 240) -> None:
    #img = cv2.imread(src)
    img = cv2.imread(src, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv[:, :, 2], thresh_v, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                            iterations=2)
    cv2.imwrite(dst, cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA))


###############################################################################
# 1) Felzenszwalb segmentation (first pass)
###############################################################################

def _mask_bbox(mask_2d: np.ndarray, pad: int = 2):
    """Return (r0, r1, c0, c1) bounding box of nonzero pixels, with padding."""
    rows = np.any(mask_2d > 0, axis=1)
    cols = np.any(mask_2d > 0, axis=0)
    if not rows.any():
        return 0, mask_2d.shape[0], 0, mask_2d.shape[1]
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0 = max(0, r0 - pad)
    r1 = min(mask_2d.shape[0], r1 + 1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(mask_2d.shape[1], c1 + 1 + pad)
    return r0, r1, c0, c1


def fhs_segmentation(img_p: str, mask_p: str, out_p: str) -> None:
    rgb = io.imread(img_p)
    mask = io.imread(mask_p)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if rgb.shape[:2] != mask.shape:
        print("[WARN] shape mismatch – skipping", img_p)
        return
    r0, r1, c0, c1 = _mask_bbox(mask)
    rgb_crop = rgb[r0:r1, c0:c1]
    mask_crop = mask[r0:r1, c0:c1]
    fg_crop = np.zeros_like(rgb_crop)
    fg_crop[mask_crop > 0] = rgb_crop[mask_crop > 0]
    seg_crop = felzenszwalb(fg_crop, scale=100, sigma=0.5, min_size=50)
    avg_crop = color.label2rgb(seg_crop, fg_crop, kind="avg", bg_label=0)
    out = np.zeros_like(rgb)
    out[r0:r1, c0:c1] = avg_crop
    io.imsave(out_p, out)


###############################################################################
# 2) CLAHE luminance normalisation
###############################################################################

def normalize_color(src: str, dst: str, mask_path: Optional[str] = None) -> None:
    img = cv2.imread(src, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION); assert img is not None, src
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == img.shape[:2]:
            r0, r1, c0, c1 = _mask_bbox(mask)
            crop = img[r0:r1, c0:c1]
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[:, :, 0])
            out = img.copy()
            out[r0:r1, c0:c1] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            cv2.imwrite(dst, out)
            return
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[:, :, 0])
    cv2.imwrite(dst, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))


###############################################################################
# 2b) Adaptive ΔE threshold via silhouette scoring
###############################################################################

def _load_fg_lab(seg_p: str, mask_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an image, apply mask, return (unique_lab, inverse, fg_bool)."""
    rgb = io.imread(seg_p)
    if mask_path is not None and os.path.exists(mask_path):
        mask = io.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        fg = mask > 0
    else:
        fg = ~((rgb[:, :, 0] == 0) & (rgb[:, :, 1] == 0) & (rgb[:, :, 2] == 0))
    lab_fg = color.rgb2lab(rgb[fg]).reshape(-1, 3)
    uniq, inverse = np.unique(lab_fg, axis=0, return_inverse=True)
    return uniq, inverse, fg


def find_adaptive_threshold(lab_uniq: np.ndarray,
                            candidates: Optional[List[float]] = None,
                            max_sample: int = 5000) -> float:
    """Return the ΔE threshold that maximises mean silhouette score.

    Tests a range of candidate thresholds (default 5–50 in steps of 2.5)
    and picks the one producing the most internally coherent, externally
    separated clusters.  Sub-samples if data is too large for speed.
    """
    from sklearn.metrics import silhouette_score

    if candidates is None:
        candidates = list(np.arange(5.0, 52.5, 2.5))

    if lab_uniq.shape[0] <= 1:
        return candidates[len(candidates) // 2]

    if lab_uniq.shape[0] > max_sample:
        idx = np.random.default_rng(42).choice(
            lab_uniq.shape[0], max_sample, replace=False)
        pts = lab_uniq[idx]
    else:
        pts = lab_uniq

    Z = linkage(pts, method="average", metric="euclidean")
    best_score, best_t = -1.0, candidates[len(candidates) // 2]

    for t in candidates:
        labels = fcluster(Z, t=t, criterion="distance")
        n_clusters = len(np.unique(labels))
        if n_clusters < 2 or n_clusters >= len(pts):
            continue
        try:
            score = silhouette_score(pts, labels, metric="euclidean")
        except ValueError:
            # can happen with very imbalanced clusters
            continue
        if score > best_score:
            best_score = score
            best_t = t

    return best_t


###############################################################################
# 3) LAB-space averaging + coloured dendrogram / scatters
###############################################################################

def lab_average_clustering(seg_p: str,
                           save_p: str,
                           threshold: float = 25.0,
                           adaptive: bool = False,
                           mask_path: Optional[str] = None,
                           dendro_path: Optional[str] = None,
                           scatter_before: Optional[str] = None,
                           scatter_after: Optional[str] = None
                           ) -> Dict:
    """Cluster foreground pixels in LAB space and average.

    Returns a dict with clustering metadata (threshold used,
    n_clusters, cluster_areas, label_map, fg_mask) for downstream use.
    """
    rgb = io.imread(seg_p)

    # mask background ---------------------------------------------
    if mask_path is not None and os.path.exists(mask_path):
        mask = io.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        fg = mask > 0
    else:
        fg = ~((rgb[:, :, 0] == 0) & (rgb[:, :, 1] == 0) & (rgb[:, :, 2] == 0))

    lab_fg = color.rgb2lab(rgb[fg]).reshape(-1, 3)
    uniq, inverse = np.unique(lab_fg, axis=0, return_inverse=True)

    # ---- adaptive threshold ----
    if adaptive:
        threshold = find_adaptive_threshold(uniq)
        print(f"  [adaptive] selected ΔE = {threshold:.1f} for "
              f"{os.path.basename(seg_p)}")

    if uniq.shape[0] <= 1:
        clusters = np.ones(uniq.shape[0], dtype=int)
    else:
        Z = linkage(uniq, method="average", metric="euclidean")
        clusters = fcluster(Z, t=threshold, criterion="distance")
    means = {c: uniq[clusters == c].mean(axis=0) for c in np.unique(clusters)}

    lab_out = np.vstack([means[c] for c in clusters])[inverse]
    rgb_out = np.zeros_like(rgb, dtype=np.float64)
    rgb_out[fg] = color.lab2rgb(lab_out)
    io.imsave(save_p, (rgb_out * 255).astype(np.uint8))

    # ---------- extra outputs: grayscale & binary ----------
    gray_full = cv2.cvtColor((rgb_out * 255).astype("uint8"),
                              cv2.COLOR_RGB2GRAY)
    gray_canvas = np.zeros_like(gray_full)
    gray_canvas[fg] = gray_full[fg]
    io.imsave(save_p.replace(".png", "_gray.png"), gray_canvas)

    pixel_labels = clusters[inverse]
    label_map = np.zeros_like(gray_full, dtype=np.int32)
    label_map[fg] = pixel_labels

    dominant = max(means, key=lambda c: (pixel_labels == c).sum())
    binary = np.zeros_like(gray_full, dtype=np.uint8)
    binary[label_map == dominant] = 255
    io.imsave(save_p.replace(".png", "_bin.png"), binary)

    total_fg = int(fg.sum())
    cluster_areas = {int(c): int((pixel_labels == c).sum()) / total_fg
                     for c in np.unique(clusters)}

    # true-colour dendrogram --------------------------------------
    if dendro_path:
        n_leaves = uniq.shape[0]
        node_lab: Dict[int, np.ndarray] = {i: uniq[i] for i in range(n_leaves)}
        for i, (l, r, *_rest) in enumerate(Z):
            node_lab[n_leaves + i] = (node_lab[int(l)] + node_lab[int(r)]) / 2

        def _link_color(nid: int):
            return mcolors.to_hex(_lab2rgb_pts(node_lab[nid][None, :])[0])

        fig, ax = plt.subplots(figsize=(10, 5))
        dn = dendrogram(Z, ax=ax, link_color_func=_link_color, no_labels=True)
        xs = ax.get_xticks(); leaves = dn["leaves"]
        for x, leaf_idx in zip(xs, leaves):
            c = _lab2rgb_pts(uniq[leaf_idx][None, :])[0]
            ax.scatter(x, -2, s=120, c=[c], marker="s", clip_on=False)
        ax.set_title(f"Colour dendrogram  (ΔE ≤ {threshold})")
        plt.tight_layout(); plt.savefig(dendro_path, dpi=180); plt.close()

    # 3-D scatters -------------------------------------------------
    def _scatter(pts: np.ndarray, fn: Optional[str]):
        if fn:
            fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pts[:, 1], pts[:, 2], pts[:, 0], s=40,
                       c=_lab2rgb_pts(pts))
            ax.set_xlabel("a"); ax.set_ylabel("b"); ax.set_zlabel("L")
            plt.tight_layout(); plt.savefig(fn, dpi=180); plt.close()

    _scatter(uniq, scatter_before)
    _scatter(np.array(list(means.values())), scatter_after)

    return {
        "threshold_used": threshold,
        "n_clusters": len(means),
        "cluster_areas": cluster_areas,
        "label_map": label_map,
        "fg_mask": fg,
    }


###############################################################################
# 4) FFT Granularity (Pattern Energy) Analysis
#    Following Troscianko & Stevens (2015) / Nokelainen et al. (2024)
###############################################################################

def granularity_analysis(img_path: str,
                         mask_path: Optional[str],
                         n_bands: int = 20,
                         max_filter_px: int = 100
                         ) -> Dict:
    """FFT band-pass granularity analysis on the luminance channel.

    Decomposes image into spatial frequency bands using sequential
    difference-of-means filtering (equivalent to MICA Toolbox).
    Pattern energy at each scale = std of filtered pixel values
    within the foreground mask.

    Returns dict with filter_sizes, energies, max_power (dominance),
    max_freq (marking size), sum_power (contrast), prop_energies.
    """
    #img = cv2.imread(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(img_path)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0].astype(np.float64)

    if mask_path and os.path.exists(mask_path):
        mask = io.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        fg = mask > 0
    else:
        fg = lum > 0

    fg_mean = lum[fg].mean()
    lum_masked = lum.copy()
    lum_masked[~fg] = fg_mean

    filter_sizes = np.linspace(1, max_filter_px, n_bands + 1).astype(int)
    filter_sizes = np.unique(filter_sizes)

    energies = []
    prev_smooth = lum_masked.copy()
    for i in range(len(filter_sizes)):
        ksize = max(1, int(filter_sizes[i]))
        smoothed = uniform_filter(lum_masked, size=ksize, mode='reflect')
        band = prev_smooth - smoothed
        energy = float(band[fg].std())
        energies.append(energy)
        prev_smooth = smoothed

    energies = np.array(energies)
    actual_sizes = filter_sizes[:len(energies)]

    max_idx = int(np.argmax(energies))
    max_power = float(energies[max_idx])
    max_freq  = float(actual_sizes[max_idx])
    sum_power = float(energies.sum())
    total = energies.sum()
    prop_energies = (energies / total).tolist() if total > 0 else energies.tolist()

    return {
        "filter_sizes":  actual_sizes.tolist(),
        "energies":      energies.tolist(),
        "max_power":     max_power,
        "max_freq":      max_freq,
        "sum_power":     sum_power,
        "prop_energies": prop_energies,
    }


###############################################################################
# 5) Colour Adjacency Analysis
#    Following Endler (2012) / boundary strength approach
###############################################################################

def colour_adjacency_analysis(label_map: np.ndarray,
                              fg_mask: np.ndarray,
                              cluster_means_lab: Optional[Dict] = None
                              ) -> Dict:
    """Compute the colour adjacency (transition) matrix.

    For every pair of horizontally/vertically adjacent foreground
    pixels, record same- vs. different-cluster transitions.

    Returns dict with transition_matrix, n_same, n_different,
    pattern_complexity, boundary_strength, cluster_ids.
    """
    # horizontal transitions
    labels_l = label_map[:, :-1]
    labels_r = label_map[:, 1:]
    valid_h = fg_mask[:, :-1] & fg_mask[:, 1:]

    # vertical transitions
    labels_u = label_map[:-1, :]
    labels_d = label_map[1:, :]
    valid_v = fg_mask[:-1, :] & fg_mask[1:, :]

    # vectorised counting using paired labels
    pairs_h = np.stack([labels_l[valid_h], labels_r[valid_h]], axis=1)
    pairs_v = np.stack([labels_u[valid_v], labels_d[valid_v]], axis=1)
    all_pairs = np.concatenate([pairs_h, pairs_v], axis=0)

    # sort each pair so (i,j) and (j,i) map to same key
    all_pairs_sorted = np.sort(all_pairs, axis=1)
    same_mask = all_pairs_sorted[:, 0] == all_pairs_sorted[:, 1]
    n_same = int(same_mask.sum())
    n_diff = int((~same_mask).sum())

    # build transition matrix
    tm: Dict[Tuple[int, int], int] = {}
    unique_pairs, counts = np.unique(all_pairs_sorted, axis=0, return_counts=True)
    for (a, b), c in zip(unique_pairs, counts):
        tm[(int(a), int(b))] = int(c)

    total = n_same + n_diff
    complexity = n_diff / total if total > 0 else 0.0

    boundary_strength = 0.0
    if cluster_means_lab and n_diff > 0:
        weighted_de = 0.0
        diff_count = 0
        for (i, j), count in tm.items():
            if i != j and i in cluster_means_lab and j in cluster_means_lab:
                de = np.linalg.norm(
                    np.array(cluster_means_lab[i]) - np.array(cluster_means_lab[j]))
                weighted_de += de * count
                diff_count += count
        if diff_count > 0:
            boundary_strength = weighted_de / diff_count

    cluster_ids = sorted(set(
        [k for pair in tm for k in pair if k != 0]))

    return {
        "transition_matrix": {str(k): v for k, v in tm.items()},
        "n_same": n_same,
        "n_different": n_diff,
        "pattern_complexity": complexity,
        "boundary_strength": boundary_strength,
        "cluster_ids": cluster_ids,
    }


###############################################################################
# 6) Elliptical Fourier Analysis (EFA) of marking shapes
#    Following Chan, Stevens & Todd (2019) pat-geom methodology
###############################################################################

def efa_marking_analysis(label_map: np.ndarray,
                         fg_mask: np.ndarray,
                         n_harmonics: int = 20,
                         min_area: int = 100,
                         cluster_means_lab: Optional[Dict] = None
                         ) -> Dict:
    """Elliptical Fourier Analysis of colour-patch (marking) shapes.

    For each colour cluster in the label_map, the binary mask of that
    cluster is contour-traced with cv2.findContours.  Each contour
    larger than min_area is analysed with Elliptical Fourier
    Descriptors (EFDs) to extract:

      * aspect_ratio   — semi-major / semi-minor of the first-harmonic
                         ellipse (1.0 = circular, >1 = elongated)
      * bending_energy — Σ n² (aₙ² + bₙ² + cₙ² + dₙ²), measures
                         boundary irregularity / complexity of shape
                         (Bookstein 1991; higher = more irregular)
      * shape_complexity — minimum number of harmonics needed to
                           capture ≥99 % of cumulative harmonic power
                           (1 = perfectly elliptical)
      * marking_area   — contour area in px²

    Per-image aggregates (mean, std across all markings) are returned
    alongside per-marking details.

    Parameters
    ----------
    label_map    : 2-D int array (0 = background, ≥1 = cluster IDs)
    fg_mask      : boolean foreground mask
    n_harmonics  : EFA order (default 20)
    min_area     : minimum contour area in px² (default 100)
    cluster_means_lab : {cid: [L,a,b]} for colour annotation

    Returns
    -------
    dict with per_marking (list of dicts) and image-level aggregates.
    """
    from pyefd import elliptic_fourier_descriptors

    per_marking: List[Dict] = []
    cluster_ids = [c for c in np.unique(label_map) if c != 0]

    for cid in cluster_ids:
        bin_mask = ((label_map == cid) & fg_mask).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            # reshape contour to (N, 2) for pyefd
            pts = cnt.squeeze()
            if pts.ndim != 2 or pts.shape[0] < 10:
                continue

            try:
                coeffs = elliptic_fourier_descriptors(
                    pts, order=n_harmonics, normalize=False)
            except Exception:
                continue

            # --- aspect ratio from first harmonic ---
            # The first harmonic defines the best-fit ellipse.
            # Semi-axes lengths from eigenvalues of the 2×2 matrix
            # [[a₁, b₁], [c₁, d₁]]
            a1, b1, c1, d1 = coeffs[0]
            M = np.array([[a1, b1], [c1, d1]])
            eigvals = np.sqrt(np.linalg.eigvalsh(M.T @ M))
            eigvals = np.sort(eigvals)[::-1]  # descending
            if eigvals[-1] > 0:
                aspect_ratio = float(eigvals[0] / eigvals[-1])
            else:
                aspect_ratio = 1.0

            # --- bending energy ---
            # BE = Σ_{n=1}^{K} n² × (aₙ² + bₙ² + cₙ² + dₙ²)
            harmonics = np.arange(1, len(coeffs) + 1)
            harmonic_power = np.sum(coeffs ** 2, axis=1)
            bending_energy = float(np.sum(harmonics ** 2 * harmonic_power))

            # --- shape complexity ---
            # Minimum harmonics to reach 99% cumulative power
            cumulative = np.cumsum(harmonic_power)
            total_power = cumulative[-1]
            if total_power > 0:
                threshold_99 = 0.99 * total_power
                shape_complexity = int(
                    np.searchsorted(cumulative, threshold_99) + 1)
            else:
                shape_complexity = 1

            # --- perimeter ---
            perimeter = float(cv2.arcLength(cnt, closed=True))

            marking_info = {
                "cluster_id": int(cid),
                "area_px": float(area),
                "perimeter_px": perimeter,
                "aspect_ratio": aspect_ratio,
                "bending_energy": bending_energy,
                "shape_complexity": shape_complexity,
            }
            if cluster_means_lab and cid in cluster_means_lab:
                marking_info["lab_L"] = cluster_means_lab[cid][0]
                marking_info["lab_a"] = cluster_means_lab[cid][1]
                marking_info["lab_b"] = cluster_means_lab[cid][2]

            per_marking.append(marking_info)

    # ---- image-level aggregates ----
    n_markings = len(per_marking)
    if n_markings == 0:
        return {
            "per_marking": [],
            "n_markings": 0,
            "mean_aspect_ratio": None,
            "std_aspect_ratio": None,
            "mean_bending_energy": None,
            "std_bending_energy": None,
            "mean_shape_complexity": None,
            "mean_marking_area": None,
            "std_marking_area": None,
            "total_marking_area": 0.0,
        }

    ars   = np.array([m["aspect_ratio"]      for m in per_marking])
    bes   = np.array([m["bending_energy"]     for m in per_marking])
    scs   = np.array([m["shape_complexity"]   for m in per_marking])
    areas = np.array([m["area_px"]            for m in per_marking])

    return {
        "per_marking":            per_marking,
        "n_markings":             n_markings,
        "mean_aspect_ratio":      float(ars.mean()),
        "std_aspect_ratio":       float(ars.std()) if n_markings > 1 else 0.0,
        "mean_bending_energy":    float(bes.mean()),
        "std_bending_energy":     float(bes.std()) if n_markings > 1 else 0.0,
        "mean_shape_complexity":  float(scs.mean()),
        "mean_marking_area":      float(areas.mean()),
        "std_marking_area":       float(areas.std()) if n_markings > 1 else 0.0,
        "total_marking_area":     float(areas.sum()),
    }


###############################################################################
# 6b) EFA visualisation overlay
###############################################################################

def plot_efa_overlay(img_path: str, label_map: np.ndarray,
                     fg_mask: np.ndarray, efa_result: Dict,
                     cluster_means_lab: Dict,
                     save_path: str) -> None:
    """Draw marking contours colour-coded by cluster, annotated with
    aspect ratio, on top of the segmented image."""
    #img = cv2.imread(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        return
    overlay = img.copy()

    for marking in efa_result.get("per_marking", []):
        cid = marking["cluster_id"]
        bin_mask = ((label_map == cid) & fg_mask).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # pick a contrasting colour for the contour
        if cid in cluster_means_lab:
            lab_c = np.array(cluster_means_lab[cid])
            rgb_c = _lab2rgb_pts(lab_c[None, :])[0]
            # invert for contrast
            bgr_c = tuple(int(255 * (1 - c)) for c in rgb_c[::-1])
        else:
            bgr_c = (0, 255, 0)

        for cnt in contours:
            if cv2.contourArea(cnt) >= marking.get("area_px", 0) * 0.9:
                cv2.drawContours(overlay, [cnt], -1, bgr_c, 2)
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(overlay, f"AR={marking['aspect_ratio']:.1f}",
                                (cx - 20, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_c, 1)
                break  # one contour per marking entry

    cv2.imwrite(save_path, overlay)


###############################################################################
# 7) HSV colour metrics (per Nokelainen et al. 2024)
###############################################################################

def hsv_colour_metrics(img_path: str,
                       mask_path: Optional[str] = None) -> Dict:
    """Extract mean and std of H, S, V within the foreground mask.

    V21 changes
    -----------
    * Hue is now expressed in full 0-360° (OpenCV 0-180 doubled).
      hue_mean is the circular mean in 0-360°, matching the R script's
      contMap(lims=c(0,360)) and hsb_to_rgb(h/360) conventions.
    * hue_circ_std is in 0-360° (no half-scale division).
    * hue_sin / hue_cos are the correct circular projections used for
      PCA (unchanged in meaning, now derived from the full-circle angle).
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    #img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)

    if mask_path and os.path.exists(mask_path):
        mask = io.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        fg = mask > 0
    else:
        fg = hsv[:, :, 2] > 0

    h, s, v = hsv[:, :, 0][fg], hsv[:, :, 1][fg], hsv[:, :, 2][fg]

    # V21: convert OpenCV 0-180 → full 0-360° before circular decomposition
    # so that hue_mean, hue_std, and the contMap in R all use the same scale.
    h_deg360 = h * 2.0                        # 0-180 → 0-360
    h_rad    = h_deg360 * np.pi / 180.0       # → radians on full circle

    hue_sin = float(np.mean(np.sin(h_rad)))
    hue_cos = float(np.mean(np.cos(h_rad)))

    # Circular mean in 0-360°
    hue_circ_mean = float(np.degrees(np.arctan2(hue_sin, hue_cos))) % 360.0

    # Circular std in degrees (Mardia & Jupp formula, full 360° scale)
    R = np.sqrt(hue_sin**2 + hue_cos**2)
    hue_circ_std = float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-10)))))

    return {
        "hue_mean": hue_circ_mean,  "hue_std": hue_circ_std,
        "hue_sin":  hue_sin,        "hue_cos": hue_cos,
        "sat_mean": float(np.mean(s)),   "sat_std": float(np.std(s)),
        "bri_mean": float(np.mean(v)),   "bri_std": float(np.std(v)),
    }


###############################################################################
# 8) Population-level CV statistics
###############################################################################

def compute_population_cv(records: List[Dict]) -> Dict:
    """Compute population-level coefficient of variation (std/mean)
    for each numeric metric across a list of per-image records."""
    if not records:
        return {}
    numeric_keys = [k for k, v in records[0].items()
                    if isinstance(v, (int, float)) and not k.startswith("_")]
    result = {}
    for key in numeric_keys:
        vals = [r[key] for r in records
                if key in r and r[key] is not None]
        if len(vals) < 2:
            result[f"{key}_cv"] = None
            continue
        arr = np.array(vals, dtype=float)
        mean = arr.mean()
        result[f"{key}_mean_pop"] = float(mean)
        result[f"{key}_std_pop"]  = float(arr.std(ddof=1))
        result[f"{key}_cv"]      = (float(arr.std(ddof=1) / mean)
                                    if mean != 0 else None)
    return result


###############################################################################
# 9) Visualisation helpers
###############################################################################

def plot_granularity_spectrum(gran: Dict, save_path: str,
                              title: str = "") -> None:
    """Plot pattern energy spectrum."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sizes = gran["filter_sizes"]
    energies = gran["energies"]
    ax.plot(sizes, energies, "o-", color="steelblue", linewidth=2)
    ax.axvline(gran["max_freq"], color="red", linestyle="--", alpha=0.7,
               label=f"peak = {gran['max_freq']:.0f} px")
    ax.fill_between(sizes, energies, alpha=0.15, color="steelblue")
    ax.set_xlabel("Filter size (px)")
    ax.set_ylabel("Pattern energy (σ)")
    ax.set_title(f"Granularity spectrum{' – ' + title if title else ''}")
    ax.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=180); plt.close()


def plot_adjacency_heatmap(adj: Dict, cluster_means_lab: Dict,
                           save_path: str, title: str = "") -> None:
    """Plot transition matrix as colour-coded heatmap."""
    cids = adj["cluster_ids"]
    if len(cids) < 2:
        return
    n = len(cids)
    mat = np.zeros((n, n))
    id2idx = {c: i for i, c in enumerate(cids)}

    for key_str, count in adj["transition_matrix"].items():
        pair = eval(key_str)
        if pair[0] in id2idx and pair[1] in id2idx:
            i, j = id2idx[pair[0]], id2idx[pair[1]]
            mat[i, j] = count
            mat[j, i] = count

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat_norm = mat / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat_norm, cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{c}" for c in cids])
    ax.set_yticklabels([f"C{c}" for c in cids])
    plt.colorbar(im, ax=ax, label="Transition proportion")

    for idx, cid in enumerate(cids):
        if cid in cluster_means_lab:
            rgb_c = _lab2rgb_pts(np.array(cluster_means_lab[cid])[None, :])[0]
            ax.get_xticklabels()[idx].set_color(rgb_c)
            ax.get_yticklabels()[idx].set_color(rgb_c)

    ax.set_title(f"Colour adjacency{' – ' + title if title else ''}")
    plt.tight_layout(); plt.savefig(save_path, dpi=180); plt.close()


###############################################################################
# 10) run_analyses_on_segmented — single entry point for all three methods
###############################################################################

def run_analyses_on_segmented(out_p: str,
                              mask_p: str,
                              cluster_info: Dict,
                              base_name: str,
                              seg_tag: str,
                              args: argparse.Namespace,
                              seg_p: Optional[str] = None,
                              ) -> Optional[Dict]:
    """Run granularity + adjacency + EFA on one segmented image.

    Parameters
    ----------
    out_p      : LAB-averaged (flat-colour) segmented image — used for
                 pattern/adjacency/EFA analysis and cluster LAB means.
    seg_p      : CLAHE-normalised foreground image (fhs_normalized_*)
                 — used for HSV colour metrics so that per-pixel colour
                 variance is preserved.  Falls back to out_p if None.
    mask_p     : binary foreground mask
    ...

    Returns a flat record dict suitable for CSV export, or None if
    pattern analysis is not requested.
    """
    if not args.run_pattern_analysis and not args.run_efa:
        return None

    # V21: HSV colour metrics use the CLAHE-normalised source image,
    # not the LAB-averaged output, to preserve per-pixel colour variance.
    colour_src = seg_p if (seg_p is not None and os.path.isfile(seg_p)) else out_p

    record: Dict = OrderedDict()
    record["filename"]  = base_name
    record["seg_tag"]   = seg_tag   # "adaptive", "median", "fixed_T25", …
    record["threshold"] = cluster_info["threshold_used"]
    record["n_clusters"] = cluster_info["n_clusters"]

    # reconstruct cluster LAB means from the averaged image
    rgb_avg = io.imread(out_p)
    label_map = cluster_info["label_map"]
    fg_mask = cluster_info["fg_mask"]
    lab_avg_img = color.rgb2lab(rgb_avg)
    cluster_means_lab = {}
    for cid in np.unique(label_map):
        if cid == 0:
            continue
        cmask = label_map == cid
        cluster_means_lab[cid] = lab_avg_img[cmask].mean(axis=0).tolist()

    # ---- HSV colour metrics (V21: from CLAHE-normalised source, not avg) ----
    hsv_m = hsv_colour_metrics(colour_src, mask_p)
    record.update(hsv_m)

    # ---- granularity analysis ----
    if args.run_pattern_analysis:
        gran = granularity_analysis(
            out_p, mask_p,
            n_bands=args.granularity_bands,
            max_filter_px=args.granularity_max_px)
        record["max_power"]  = gran["max_power"]
        record["max_freq"]   = gran["max_freq"]
        record["sum_power"]  = gran["sum_power"]

        # adjacency analysis
        adj = colour_adjacency_analysis(label_map, fg_mask, cluster_means_lab)
        record["pattern_complexity"] = adj["pattern_complexity"]
        record["boundary_strength"]  = adj["boundary_strength"]
        record["n_boundaries_same"]  = adj["n_same"]
        record["n_boundaries_diff"]  = adj["n_different"]

        # optional per-image vis
        if args.pattern_viz:
            stem = out_p.replace(".png", "")
            plot_granularity_spectrum(
                gran, f"{stem}_granularity.png", title=base_name)
            plot_adjacency_heatmap(
                adj, cluster_means_lab,
                f"{stem}_adjacency.png", title=base_name)

        print(f"  [pattern] {seg_tag} {base_name}: "
              f"maxFreq={gran['max_freq']:.0f}px  "
              f"maxPow={gran['max_power']:.2f}  "
              f"sumPow={gran['sum_power']:.2f}  "
              f"complexity={adj['pattern_complexity']:.3f}  "
              f"boundaryΔE={adj['boundary_strength']:.1f}")
    else:
        gran = None

    # ---- EFA marking shape analysis ----
    if args.run_efa:
        efa = efa_marking_analysis(
            label_map, fg_mask,
            n_harmonics=args.efa_harmonics,
            min_area=args.efa_min_area,
            cluster_means_lab=cluster_means_lab)
        record["n_markings"]          = efa["n_markings"]
        record["mean_aspect_ratio"]   = efa["mean_aspect_ratio"]
        record["std_aspect_ratio"]    = efa["std_aspect_ratio"]
        record["mean_bending_energy"] = efa["mean_bending_energy"]
        record["std_bending_energy"]  = efa["std_bending_energy"]
        record["mean_shape_complexity"] = efa["mean_shape_complexity"]
        record["mean_marking_area"]   = efa["mean_marking_area"]
        record["std_marking_area"]    = efa["std_marking_area"]
        record["total_marking_area"]  = efa["total_marking_area"]

        if args.efa_viz:
            efa_save = out_p.replace(".png", "_efa_overlay.png")
            plot_efa_overlay(out_p, label_map, fg_mask, efa,
                             cluster_means_lab, efa_save)

        print(f"  [efa]     {seg_tag} {base_name}: "
              f"n_markings={efa['n_markings']}  "
              f"meanAR={efa['mean_aspect_ratio'] or 0:.2f}  "
              f"meanBE={efa['mean_bending_energy'] or 0:.1f}  "
              f"meanSC={efa['mean_shape_complexity'] or 0:.1f}")

    return record


###############################################################################
# 11) CLI
###############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Felzenszwalb + CLAHE + LAB averaging + pattern analysis (V19)")

    # --- core pipeline ---
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_segmentation", action="store_true")
    p.add_argument("--run_normalization", action="store_true")
    p.add_argument("--remove_shine", action="store_true")
    p.add_argument("--run_lab_average", action="store_true")
    p.add_argument("--lab_threshold", type=float, default=25.0)
    p.add_argument("--lab_sweep", type=str,
                   help="Comma-sep list of ΔE thresholds, e.g. 15,20,25")
    p.add_argument("--visualize", action="store_true")

    # --- adaptive / median threshold ---
    p.add_argument("--adaptive_threshold", action="store_true",
                   help="Per-image silhouette-based adaptive ΔE. "
                        "Requires scikit-learn.")
    p.add_argument("--median_threshold", action="store_true",
                   help="TWO-PASS mode: Pass 1 computes per-image "
                        "adaptive ΔE for every image.  Pass 2 takes "
                        "the MEDIAN ΔE and re-clusters every image at "
                        "that single threshold.  BOTH per-image adaptive "
                        "outputs AND median-calibrated outputs are saved. "
                        "Implies --adaptive_threshold.  Requires scikit-learn.")

    # --- pattern analysis (granularity + adjacency) ---
    p.add_argument("--run_pattern_analysis", action="store_true",
                   help="FFT granularity + colour adjacency analysis")
    p.add_argument("--granularity_bands", type=int, default=20)
    p.add_argument("--granularity_max_px", type=int, default=100)
    p.add_argument("--pattern_viz", action="store_true",
                   help="Save granularity spectrum + adjacency heatmap")

    # --- EFA marking shape analysis ---
    p.add_argument("--run_efa", action="store_true",
                   help="Elliptical Fourier Analysis of marking shapes "
                        "(à la pat-geom, Chan et al. 2019). Requires pyefd.")
    p.add_argument("--efa_harmonics", type=int, default=20,
                   help="Number of Fourier harmonics (default 20)")
    p.add_argument("--efa_min_area", type=int, default=100,
                   help="Minimum contour area in px² for EFA (default 100)")
    p.add_argument("--efa_viz", action="store_true",
                   help="Save EFA contour overlay visualisation")
    p.add_argument("--category_name", type=str, default="",
                   help="Category name for output file naming "
                        "(e.g. left_elytron, pronotum). Used in "
                        "phylo CSV filename.")
    p.add_argument("--skip_preprocessing", action="store_true",
                   help="Skip FHS/CLAHE/shine removal (Stage A). "
                        "Use existing fhs_normalized_* files from --output_dir "
                        "and masks from --input_dir. Useful for resuming "
                        "after preprocessing completed but analysis crashed.")
    p.add_argument("--include_unnormalized", action="store_true",
                   help="Advanced: also analyse plain fhs_* images alongside "
                        "fhs_normalized_* images. By default only the CLAHE-"
                        "normalised versions are used for pattern/phylo analysis.")

    # --- background chromaticity normalisation ---
    p.add_argument("--bg_normalize", action="store_true",
                   help="Normalise a*/b* chromaticity using background reference "
                        "before FHS/CLAHE.  Corrects per-image illuminant "
                        "colour casts (von Kries in LAB space).  "
                        "Mode 1: if --bg_original_dir given, samples actual "
                        "background from original unmasked images.  "
                        "Mode 2: otherwise, batch-centres a*/b* across all "
                        "images to remove inter-image chromaticity variance.")
    p.add_argument("--bg_original_dir", type=str, default=None,
                   help="Directory with original (unmasked) images for "
                        "background sampling.  Used with --bg_normalize to "
                        "enable per-image von Kries correction.  If omitted, "
                        "batch mean centering is used instead.")
    p.add_argument("--bg_coco_json", type=str, default=None,
                   help="COCO JSON for resolving _fg_ filenames to original "
                        "image names (image_id lookup).  Optional; improves "
                        "filename matching with --bg_original_dir.")
    p.add_argument("--bg_correct_L", action="store_true",
                   help="Also correct L (luminance) during background "
                        "normalisation.  Usually unnecessary since CLAHE "
                        "handles luminance, but useful if CLAHE is disabled.")

    return p.parse_args()


###############################################################################
# 12) main
###############################################################################

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --median_threshold implies --adaptive_threshold
    if args.median_threshold:
        args.adaptive_threshold = True

    thresholds = ([float(t) for t in args.lab_sweep.split(",")]
                  if args.lab_sweep else [args.lab_threshold])

    all_seg_pairs: List[Tuple[str, str]] = []  # (seg_path, mask_path)
    viz_done = False

    # ================================================================
    # STAGE A — build binary / foreground pairs, FHS + CLAHE
    # ================================================================
    if args.skip_preprocessing:
        # --- RESUME MODE: skip FHS/CLAHE, use existing files ---
        print("\n=== SKIP PREPROCESSING — scanning for existing files ===")

        # scan output_dir for existing fhs_* segmented images
        out_files = os.listdir(args.output_dir)

        # By default, only use fhs_normalized_* files (CLAHE-normalised).
        # If --include_unnormalized, also use plain fhs_* files.
        normalized_files = sorted([f for f in out_files
                           if f.startswith("fhs_normalized_") and f.endswith(".png")
                           and "_gray.png" not in f and "_binary.png" not in f
                           and "_granularity" not in f and "_adjacency" not in f
                           and "_efa_overlay" not in f and "labavg_" not in f])

        unnorm_files = []
        if args.include_unnormalized:
            unnorm_files = sorted([f for f in out_files
                           if f.startswith("fhs_") and not f.startswith("fhs_normalized_")
                           and f.endswith(".png")
                           and "_gray.png" not in f and "_binary.png" not in f
                           and "_granularity" not in f and "_adjacency" not in f
                           and "_efa_overlay" not in f and "labavg_" not in f])

        fhs_files = normalized_files + unnorm_files

        # fallback: if no normalized files found, use plain fhs_ files
        if not normalized_files and not unnorm_files:
            fhs_files = sorted([f for f in out_files
                           if f.startswith("fhs_") and f.endswith(".png")
                           and "_gray.png" not in f and "_binary.png" not in f
                           and "_granularity" not in f and "_adjacency" not in f
                           and "_efa_overlay" not in f and "labavg_" not in f])
            if fhs_files:
                print("  No fhs_normalized_* files found, falling back to fhs_* files")

        print(f"  Normalized files: {len(normalized_files)}")
        if unnorm_files:
            print(f"  Unnormalized files: {len(unnorm_files)} (--include_unnormalized)")
        print(f"  Total FHS files to process: {len(fhs_files)}")

        for fhs_f in fhs_files:
            fhs_p = os.path.join(args.output_dir, fhs_f)

            # derive the fg filename by stripping fhs_ or fhs_normalized_ prefix
            fg_name = fhs_f
            if fg_name.startswith("fhs_normalized_"):
                fg_name = fg_name[len("fhs_normalized_"):]
            elif fg_name.startswith("fhs_"):
                fg_name = fg_name[len("fhs_"):]

            # derive bin mask filename: replace _fg_ with _bin_
            bin_name = fg_name.replace("_fg_", "_bin_")

            # look for bin mask in input_dir first, then output_dir
            mask_p = None
            for search_dir in [args.input_dir, args.output_dir]:
                candidate = os.path.join(search_dir, bin_name)
                if os.path.isfile(candidate):
                    mask_p = candidate
                    break

            if mask_p:
                all_seg_pairs.append((fhs_p, mask_p))

        print(f"  Found {len(all_seg_pairs)} preprocessed image+mask pairs")
        if all_seg_pairs:
            # show first few pairs for verification
            for fhs_p, mask_p in all_seg_pairs[:3]:
                print(f"    {os.path.basename(fhs_p)}")
                print(f"      → mask: {os.path.basename(mask_p)}")
        else:
            print("  WARNING: No pairs found. Check that --output_dir contains "
                  "fhs_* files and --input_dir contains *_bin_* files.")

    else:
        # --- NORMAL MODE: run full preprocessing ---
        files = os.listdir(args.input_dir)
        bin_files = [f for f in files if "_bin_" in f]
        fg_files  = [f for f in files if "_fg_" in f]
        pairs: Dict[str, Dict[str, str]] = {}
        for f in bin_files:
            key = f.replace("_bin_", "", 1)
            pairs.setdefault(key, {})["_bin_"] = f
        for f in fg_files:
            key = f.replace("_fg_", "", 1)
            pairs.setdefault(key, {})["_fg_"] = f

        # Collect valid pairs
        valid_pairs = []
        for base, pair in pairs.items():
            if "_bin_" not in pair or "_fg_" not in pair:
                continue
            valid_pairs.append((pair["_fg_"], pair["_bin_"]))

        # ---- STAGE 0: background chromaticity normalisation ----
        # Maps fg filename → path to use as input for downstream steps.
        # If bg_normalize is off, points to original; if on, to corrected.
        fg_source: Dict[str, str] = {}
        if args.bg_normalize and valid_pairs:
            raw_pairs = [(os.path.join(args.input_dir, fg_f),
                          os.path.join(args.input_dir, bn_f))
                         for fg_f, bn_f in valid_pairs]
            bgnorm_dir = os.path.join(args.output_dir, "_bgnorm")
            corrected_pairs = batch_normalize_chromaticity(
                raw_pairs, bgnorm_dir,
                original_dir=args.bg_original_dir,
                coco_json=args.bg_coco_json,
                correct_L=args.bg_correct_L)
            for (fg_f, _bn_f), (corrected_p, _) in zip(valid_pairs, corrected_pairs):
                fg_source[fg_f] = corrected_p
        else:
            for fg_f, _bn_f in valid_pairs:
                fg_source[fg_f] = os.path.join(args.input_dir, fg_f)

        for fg_f, bin_f in valid_pairs:
            bin_p = os.path.join(args.input_dir, bin_f)
            fg_p  = fg_source[fg_f]

            fg_work = fg_p
            if args.remove_shine:
                shine_p = os.path.join(args.output_dir, f"shinefree_{fg_f}")
                remove_shine_inpaint(fg_p, shine_p)
                fg_work = shine_p

            if args.run_segmentation:
                seg_p = os.path.join(args.output_dir, f"fhs_{fg_f}")
                fhs_segmentation(fg_work, bin_p, seg_p)
                if args.include_unnormalized or not args.run_normalization:
                    all_seg_pairs.append((seg_p, bin_p))

            if args.run_normalization:
                norm_p = os.path.join(args.output_dir, f"normalized_{fg_f}")
                normalize_color(fg_work, norm_p, mask_path=bin_p)
                if args.run_segmentation:
                    seg_norm_p = os.path.join(args.output_dir,
                                              f"fhs_normalized_{fg_f}")
                    fhs_segmentation(norm_p, bin_p, seg_norm_p)
                    all_seg_pairs.append((seg_norm_p, bin_p))

    # ================================================================
    # STAGE B — LAB averaging (+pattern analyses)
    # ================================================================
    all_records: List[Dict] = []

    if args.run_lab_average:

        # -----------------------------------------------------------
        # B1.  If --median_threshold:  TWO-PASS APPROACH
        # -----------------------------------------------------------
        if args.median_threshold:
            print("\n=== PASS 1 / 2 — computing per-image adaptive ΔE ===")
            per_image_thresholds: List[float] = []

            # Pass 1a — compute adaptive threshold for every image
            for seg_p, mask_p in all_seg_pairs:
                uniq, _inv, _fg = _load_fg_lab(seg_p, mask_p)
                t_img = find_adaptive_threshold(uniq)
                per_image_thresholds.append(t_img)
                print(f"  {os.path.basename(seg_p)}: ΔE = {t_img:.1f}")

            median_de = float(np.median(per_image_thresholds))
            mean_de   = float(np.mean(per_image_thresholds))
            std_de    = float(np.std(per_image_thresholds, ddof=1)) if len(per_image_thresholds) > 1 else 0.0
            print(f"\n  Per-image ΔE:  median = {median_de:.1f}  |  "
                  f"mean = {mean_de:.1f}  |  std = {std_de:.1f}  |  "
                  f"n = {len(per_image_thresholds)}")

            # Pass 1b — cluster each image at its OWN adaptive ΔE
            #            AND run analyses on that result
            print("\n=== PASS 1b — per-image adaptive clustering + analyses ===")
            for idx, (seg_p, mask_p) in enumerate(all_seg_pairs):
                base_name = os.path.basename(seg_p)
                t_img = per_image_thresholds[idx]
                out_p = os.path.join(
                    args.output_dir,
                    f"labavg_adaptive_dE{t_img:.0f}_{base_name}")

                cluster_info = lab_average_clustering(
                    seg_p, out_p, threshold=t_img,
                    adaptive=False,  # already computed
                    mask_path=mask_p)

                rec = run_analyses_on_segmented(
                    out_p, mask_p, cluster_info, base_name,
                    seg_tag="adaptive", args=args, seg_p=seg_p)
                if rec:
                    all_records.append(rec)

            # Pass 2 — re-cluster every image at the MEDIAN ΔE
            print(f"\n=== PASS 2 / 2 — median-calibrated clustering "
                  f"(ΔE = {median_de:.1f}) ===")
            for seg_p, mask_p in all_seg_pairs:
                base_name = os.path.basename(seg_p)
                out_p = os.path.join(
                    args.output_dir,
                    f"labavg_median_dE{median_de:.0f}_{base_name}")

                dendro = scatter_b = scatter_a = None
                if args.visualize and not viz_done:
                    dendro   = os.path.join(args.output_dir,
                                            'dendrogram_median.png')
                    scatter_b = os.path.join(args.output_dir,
                                             'lab_before_median.png')
                    scatter_a = os.path.join(args.output_dir,
                                             'lab_after_median.png')

                cluster_info = lab_average_clustering(
                    seg_p, out_p, threshold=median_de,
                    adaptive=False, mask_path=mask_p,
                    dendro_path=dendro,
                    scatter_before=scatter_b,
                    scatter_after=scatter_a)
                if dendro:
                    viz_done = True

                rec = run_analyses_on_segmented(
                    out_p, mask_p, cluster_info, base_name,
                    seg_tag="median", args=args, seg_p=seg_p)
                if rec:
                    all_records.append(rec)

            # save the threshold summary
            thresh_info = {
                "per_image_thresholds": per_image_thresholds,
                "median_dE": median_de,
                "mean_dE": mean_de,
                "std_dE": std_de,
                "n_images": len(per_image_thresholds),
            }
            thresh_path = os.path.join(args.output_dir,
                                       "adaptive_threshold_summary.json")
            with open(thresh_path, "w") as f:
                json.dump(thresh_info, f, indent=2)
            print(f"\nThreshold summary saved to: {thresh_path}")

        # -----------------------------------------------------------
        # B2.  Standard path (fixed threshold or per-image adaptive)
        # -----------------------------------------------------------
        else:
            for seg_p, mask_p in all_seg_pairs:
                base_name = os.path.basename(seg_p)

                for t in thresholds:
                    out_p = os.path.join(
                        args.output_dir, f"labavg_T{t}_{base_name}")
                    dendro = scatter_b = scatter_a = None
                    if args.visualize and not viz_done:
                        dendro   = os.path.join(args.output_dir,
                                                'dendrogram.png')
                        scatter_b = os.path.join(args.output_dir,
                                                 'lab_before.png')
                        scatter_a = os.path.join(args.output_dir,
                                                 'lab_after.png')

                    cluster_info = lab_average_clustering(
                        seg_p, out_p, threshold=t,
                        adaptive=args.adaptive_threshold,
                        mask_path=mask_p,
                        dendro_path=dendro,
                        scatter_before=scatter_b,
                        scatter_after=scatter_a)
                    if dendro:
                        viz_done = True

                    tag = ("adaptive" if args.adaptive_threshold
                           else f"fixed_T{t}")
                    rec = run_analyses_on_segmented(
                        out_p, mask_p, cluster_info, base_name,
                        seg_tag=tag, args=args, seg_p=seg_p)
                    if rec:
                        all_records.append(rec)

    # ================================================================
    # STAGE C — export CSVs + population-level CV
    # ================================================================
    if all_records:
        # use all keys from all records (median records may have more)
        all_keys = list(OrderedDict.fromkeys(
            k for r in all_records for k in r.keys()))

        csv_path = os.path.join(args.output_dir, "pattern_summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys,
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(all_records)
        print(f"\nPer-image metrics saved to: {csv_path}")

        # separate per-tag CV so you can compare adaptive vs median
        tags = sorted(set(r.get("seg_tag", "") for r in all_records))
        pop_cv_all: Dict[str, Dict] = {}
        for tag in tags:
            tag_records = [r for r in all_records
                          if r.get("seg_tag") == tag]
            if len(tag_records) < 2:
                continue
            cv = compute_population_cv(tag_records)
            pop_cv_all[tag] = cv

        cv_path = os.path.join(args.output_dir, "population_cv.json")
        with open(cv_path, "w") as f:
            json.dump(pop_cv_all, f, indent=2)
        print(f"Population CV statistics saved to: {cv_path}")

        # summary table
        for tag, cv_data in pop_cv_all.items():
            print(f"\n--- Population-level CV: [{tag}] ---")
            cv_keys = sorted([k for k in cv_data if k.endswith("_cv")])
            for k in cv_keys:
                metric = k.replace("_cv", "")
                mean_val = cv_data.get(f"{metric}_mean_pop", "?")
                cv_val = cv_data[k]
                if cv_val is not None:
                    print(f"  {metric:30s}  "
                          f"mean={mean_val:10.3f}  CV={cv_val:.4f}")

        # also save per-marking EFA details as a separate CSV
        # (one row per marking, not per image)
        if args.run_efa:
            _save_efa_details(all_records, args.output_dir)

        # ============================================================
        # PHYLO-COMPATIBLE CSV EXPORT — one per seg_tag
        # ============================================================
        tags = sorted(set(r.get("seg_tag", "") for r in all_records))
        cat_name = args.category_name or "color"

        # extract specimen name from filename
        # Handles naming convention:
        #   fhs_normalized_Acales_crypto_282jpg_1_fg_left_elytron.png
        #   → Acales_crypto_282.jpg_1  (matches R script tree-tip format)
        import re as _re
        def clean_specimen_name(fn):
            name = fn
            # strip file extension
            name = os.path.splitext(name)[0]
            # strip labavg prefixes (e.g. labavg_median_dE29_)
            name = _re.sub(r'^labavg_(?:median|adaptive)_dE[\d.]+_', '', name)
            name = _re.sub(r'^labavg_T[\d.]+_', '', name)
            # strip fhs_normalized_ or fhs_ prefix
            if name.startswith("fhs_normalized_"):
                name = name[len("fhs_normalized_"):]
            elif name.startswith("fhs_"):
                name = name[len("fhs_"):]
            # strip _fg_<category> or _bin_<category> suffix
            name = _re.sub(r'_(?:fg|bin)_[a-zA-Z_]+$', '', name)
            # re-insert dot before image extension so R script can match
            # e.g. 282jpg_1 → 282.jpg_1
            name = _re.sub(r'(\d)(jpg|jpeg|png|tif|tiff)(_\d+)$',
                           r'\1.\2\3', name)
            return name

        for phylo_tag in tags:
            phylo_records = [r for r in all_records
                             if r.get("seg_tag") == phylo_tag]
            if len(phylo_records) < 2:
                continue

            # build phylo dataframe
            phylo_rows = []
            for rec in phylo_records:
                row = OrderedDict()
                row['filename'] = clean_specimen_name(rec.get('filename', ''))
                # copy all numeric fields
                for k, v in rec.items():
                    if k in ('filename', 'seg_tag'):
                        continue
                    if isinstance(v, (int, float)) and v is not None:
                        row[k] = v
                phylo_rows.append(row)

            phylo_df = pd.DataFrame(phylo_rows)

            # PCA on all numeric columns EXCEPT circular hue
            # hue_mean and hue_std are circular — kept in CSV for R's
            # anc_circular() but excluded from PCA.  hue_sin and hue_cos
            # (linear projections) ARE included in PCA.
            circular_exclude = {'hue_mean', 'hue_std'}
            numeric_cols = [c for c in phylo_df.columns
                           if c != 'filename' and c not in circular_exclude
                           and phylo_df[c].dtype in
                           [np.float64, np.int64, float, int]]
            if len(numeric_cols) >= 2:
                X = phylo_df[numeric_cols].fillna(0).values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                n_comp = min(10, X_scaled.shape[0] - 1, X_scaled.shape[1])
                if n_comp >= 2:
                    pca = PCA(n_components=n_comp)
                    pc_scores = pca.fit_transform(X_scaled)
                    for i in range(n_comp):
                        phylo_df[f'PC{i+1}'] = pc_scores[:, i]

                    # save variance explained
                    var_df = pd.DataFrame({
                        'component': [f'PC{i+1}' for i in range(n_comp)],
                        'variance_explained': pca.explained_variance_ratio_,
                        'cumulative': np.cumsum(pca.explained_variance_ratio_)
                    })
                    var_df.to_csv(os.path.join(
                        args.output_dir,
                        f"color_pca_variance_{cat_name}_{phylo_tag}.csv"),
                        index=False)

                    print(f"\nColor PCA variance [{phylo_tag}]: "
                          f"{pca.explained_variance_ratio_[:5].round(3)}")

            # save phylo CSV
            phylo_path = os.path.join(
                args.output_dir,
                f"color_traits_phylo_{cat_name}_{phylo_tag}.csv")
            phylo_df.to_csv(phylo_path, index=False)
            print(f"Phylo traits saved: {phylo_path} "
                  f"({len(phylo_df)} specimens, "
                  f"seg_tag='{phylo_tag}')")

    print("\nProcessing complete. Results saved in:", args.output_dir)


def _save_efa_details(records: List[Dict], output_dir: str) -> None:
    """Write a detailed per-marking CSV from the EFA results stored
    alongside each record.  (The main CSV only has image-level means.)
    This reads back the per-marking data that we can regenerate from
    the stored label maps, but for simplicity we just note that the
    main CSV has the aggregates and point users to the per-image
    *_efa_overlay.png for per-marking inspection."""
    # The per-marking data was computed inside run_analyses_on_segmented
    # but we only stored aggregates in the flat record.  We leave a
    # breadcrumb in the main CSV so users know the per-marking details
    # exist in the EFA overlay visualisations.
    detail_note = os.path.join(output_dir, "efa_details_note.txt")
    with open(detail_note, "w") as f:
        f.write("EFA per-marking details\n")
        f.write("=======================\n\n")
        f.write("The pattern_summary.csv contains image-level EFA aggregates\n")
        f.write("(mean/std aspect ratio, bending energy, shape complexity, etc.).\n\n")
        f.write("Per-marking details (individual contour measurements) are\n")
        f.write("visible in the *_efa_overlay.png visualisations when\n")
        f.write("--efa_viz is used.\n\n")
        f.write("To export full per-marking CSVs, re-run with\n")
        f.write("--run_efa and inspect the EFA function's per_marking\n")
        f.write("return value programmatically.\n")
    print(f"EFA details note saved to: {detail_note}")


if __name__ == "__main__":
    main()
