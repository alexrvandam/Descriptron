#!/usr/bin/env python3
"""
color_phenomics_homology.py  (v2.1)
====================================
Homology-aware COLOR phenomics using TPS-warped quadrilateral grid cells.

v2.1 (2026-04) — HONEST PER-CELL ABSOLUTE LAB
----------------------------------------------
Adds three columns per cell to the output CSV: L_mean_abs, a_mean_abs,
b_mean_abs (absolute CIELAB means per cell).  These are already computed
internally by v2 but were not written to disk.  Surfacing them enables
honest ancestral state reconstruction of per-cell colour downstream:
  • reconstruct L_mean_abs, a_mean_abs, b_mean_abs at each ancestral node
    via fastAnc (or correlated multivariate equivalents) — these are
    measured quantities, not PCA back-projected
  • render the reconstructed LAB as actual colour at nodes without
    needing to re-combine relative-LAB with a specimen mean from a
    separate CSV
  • no changes to v2's CLI, dispatched modes, or PCA/UMAP outputs
  • adds ~240 columns for an 8×10 grid (80 cells × 3 LAB channels)
  • drop-in replacement for v2 — same filenames, same command line

Everything else below is unchanged from v2:

v2 adds three orthogonal colour analyses that answer different
biological questions. Select with --color_mode:

  pattern   : per-cell SPECIMEN-RELATIVE L*a*b* features (v1 behaviour).
              8 features × n_cells per specimen. Camera/shine-robust.
              Answers: "Which specimens have similar colour PATTERNS
              (spatial contrast structure)?"  Dark-green-with-yellow-apex
              clusters with dark-green-with-yellow-apex regardless of
              exact shade.  PCA/UMAP driven by within-specimen contrast.

  hue       : SPECIMEN-LEVEL ABSOLUTE features only.  6 features total:
                L_mean, a_mean, b_mean         — absolute L*a*b* means
                chroma_ab = sqrt(a*² + b*²)    — saturation in LAB plane
                hue_ab_sin, hue_ab_cos         — LAB chromaticity angle
                                                  atan2(b*,a*), decomposed
              Answers: "Which specimens have similar OVERALL COLOUR?"
              All greens cluster, all reds cluster, etc.  No spatial
              information.  Recommended input: V21 labavg_median_* images
              so shine is already cancelled.

  combined  : both concatenated, BLOCK-z-scored so feature count doesn't
              bias the result.  --hue_weight controls the variance ratio
              between the two blocks.  Default 1.0 = equal block weight.
              Answers: "Which specimens are similar in BOTH colour and
              pattern?"  Best general-purpose setting.

LAB hue angle rationale
-----------------------
In L*a*b* space the (a*, b*) plane is perceptually-meaningful
chromaticity: chroma = radial distance, hue = polar angle atan2(b*,a*).
Hue is circular so we decompose to sin/cos in the same way HSV hue is
handled in FHS_and_CLAHE_V21.py — this avoids 0°/360° wrap artefacts.

v2 feature-list per specimen (8 per cell):
  - dL_rel, da_rel, db_rel : specimen-relative L*a*b* means
    (deviation from whole-specimen foreground mean — camera-robust)
  - L_std, a_std, b_std    : within-cell L*a*b* standard deviations
    (measures color variability/patterning within this anatomical region)
  - color_entropy           : Shannon entropy of quantized color distribution
    within the cell (high = multi-colored, low = uniform)
  - dominant_proportion     : proportion of the most common color bin
    (1.0 = perfectly uniform, <0.5 = multi-colored region)

Pipeline:
  1. Load V34 GPA output (gpa_mean.csv + back_transformed_coco.json)
  2. Build regular grid inside mean contour
  3. Use subsampled semi-landmarks as TPS control points
  4. For each specimen: TPS-warp grid → build warped quadrilateral
     cell masks → intersect with foreground mask → extract color per cell
     PLUS extract specimen-level hue features from whole foreground.
  5. Per --color_mode, assemble the feature matrix
  6. PCA / UMAP / clustering + spatial PC loading heatmaps (pattern modes)
  7. Export phylo-compatible CSV for contMap / phylosig

Usage:
  # Pattern-only (v1 behaviour):
  python color_phenomics_homology.py \\
    --gpa_dir ./outputs/left_elytron \\
    --json annotations.json \\
    --image_dir ./v21_output \\
    --output_dir ./color_homology_out \\
    --category_name left_elytron \\
    --color_mode pattern

  # Hue-only (for clustering by overall colour):
  python color_phenomics_homology.py ... --color_mode hue

  # Combined with equal block weighting (default, recommended):
  python color_phenomics_homology.py ... --color_mode combined
"""

import sys, os, json, argparse, logging, re
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from scipy.interpolate import RBFInterpolator
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from pycocotools import mask as maskUtils
except ImportError:
    maskUtils = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("color_homology.log"),
              logging.StreamHandler(sys.stdout)]
)


###############################################################################
# 1. LOAD V34 GPA OUTPUTS
###############################################################################

def load_gpa_outputs(gpa_dir):
    """Load mean shape and back-transformed contours from V34 GPA output."""
    mean_path = os.path.join(gpa_dir, 'gpa_mean.csv')
    if not os.path.isfile(mean_path):
        raise FileNotFoundError(f"gpa_mean.csv not found in {gpa_dir}")
    mean_df = pd.read_csv(mean_path)
    mean_shape = mean_df[['x', 'y']].values

    bt_path = os.path.join(gpa_dir, 'back_transformed_coco.json')
    pa_path = os.path.join(gpa_dir, 'pre_alignment_coco.json')
    coco_path = bt_path if os.path.isfile(bt_path) else pa_path
    if not os.path.isfile(coco_path):
        raise FileNotFoundError(f"No COCO in {gpa_dir}")

    with open(coco_path) as f:
        coco = json.load(f)

    img_map = {i['id']: i['file_name'] for i in coco.get('images', [])}
    specimens = OrderedDict()
    for ann in coco.get('annotations', []):
        seg = ann.get('segmentation', [])
        if not seg or not isinstance(seg[0], list):
            continue
        pts = np.array(seg[0], dtype=np.float64).reshape(-1, 2)
        fn = img_map.get(ann['image_id'], '')
        if fn:
            specimens[fn] = pts

    logging.info(f"Loaded mean shape ({mean_shape.shape[0]} pts), "
                 f"{len(specimens)} specimens from {os.path.basename(coco_path)}")
    return mean_shape, specimens


def resolve_image_path(fn_from_gpa, image_dir):
    """
    Resolve V34 GPA filename to actual V20 preprocessed image path.
    Prioritizes labavg_median files for camera-robust homology analysis.
    """
    import re
    
    # 1. Strip the trailing _NNN (image ID appended by V34 GPA)
    # Example: 'sp23_7902.jpg_240' -> 'sp23_7902.jpg'
    m = re.match(r'^(.+\.\w+)_\d+$', fn_from_gpa)
    base = m.group(1) if m else fn_from_gpa
    base_noext = os.path.splitext(base)[0]
    
    # Create a "super-clean" version for matching V20 naming quirks
    # (e.g., matching 'sp23_7902.jpg' to 'sp23_7902jpg')
    base_clean = base.lower().replace(".", "").replace("_", "")
    base_noext_clean = base_noext.lower().replace(".", "").replace("_", "")

    if not os.path.isdir(image_dir):
        return None

    candidates = os.listdir(image_dir)
    
    # --- PRIORITY 1: Look for labavg_median images (V20 calibrated output) ---
    for f in candidates:
        f_lower = f.lower().replace(".", "").replace("_", "")
        if f.lower().startswith("labavg_median"):
            if base_clean in f_lower or base_noext_clean in f_lower:
                return os.path.join(image_dir, f)

    # --- PRIORITY 2: Look for any labavg or fhs prefixed images ---
    for f in candidates:
        f_lower = f.lower().replace(".", "").replace("_", "")
        if f.lower().startswith("labavg") or f.lower().startswith("fhs"):
            if base_clean in f_lower or base_noext_clean in f_lower:
                return os.path.join(image_dir, f)

    # --- PRIORITY 3: Try direct matches or extension swaps ---
    # Try as-is
    p = os.path.join(image_dir, fn_from_gpa)
    if os.path.isfile(p): return p
    
    # Try base with common extensions
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        p = os.path.join(image_dir, base_noext + ext)
        if os.path.isfile(p): return p

    # --- FINAL FALLBACK: Any filename containing the specimen string ---
    for f in candidates:
        f_lower = f.lower().replace(".", "").replace("_", "")
        if base_noext_clean in f_lower:
            return os.path.join(image_dir, f)

    return None


def get_original_filename(fn_from_gpa):
    """Extract original filename from V34 GPA name (strip _imageID suffix)."""
    m = re.match(r'^(.+\.\w+)_\d+$', fn_from_gpa)
    return m.group(1) if m else fn_from_gpa


###############################################################################
# 2. MASK LOOKUP FROM ORIGINAL COCO
###############################################################################

def build_mask_lookup(coco_json_path, category_name):
    """Build a lookup: original_filename → (image_id, category_id, annotation)
    for the specified category from the ORIGINAL COCO JSON."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    # find category ID by name
    cat_id = None
    for c in coco.get('categories', []):
        if c['name'].lower() == category_name.lower():
            cat_id = c['id']
            break
    if cat_id is None:
        # fallback: first non-keypoint category
        for c in coco.get('categories', []):
            if c['name'].lower() not in ('keypoints', 'line_keypoints'):
                cat_id = c['id']
                logging.warning(f"Category '{category_name}' not found, "
                                f"using '{c['name']}' (id={c['id']})")
                break

    img_map = {i['id']: i for i in coco.get('images', [])}

    # build lookup by basename
    lookup = {}  # basename → best annotation + image info
    for ann in coco.get('annotations', []):
        if ann.get('category_id') != cat_id:
            continue
        ii = img_map.get(ann['image_id'])
        if not ii:
            continue
        bn = os.path.basename(ii.get('file_name', ''))
        if bn not in lookup or ann.get('area', 0) > lookup[bn]['ann'].get('area', 0):
            lookup[bn] = {'ann': ann, 'img': ii, 'cat_id': cat_id}

    logging.info(f"Mask lookup: {len(lookup)} images for category "
                 f"'{category_name}' (id={cat_id})")
    return lookup, cat_id


def decode_mask(ann, h, w):
    seg = ann.get('segmentation')
    if not seg:
        return np.zeros((h, w), np.uint8)
    try:
        if isinstance(seg, list):
            rle = maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
        else:
            rle = seg
        return (maskUtils.decode(rle) > 0).astype(np.uint8)
    except:
        return np.zeros((h, w), np.uint8)


###############################################################################
# 3. TPS WARP
###############################################################################

def build_tps_warp(source_pts, target_pts):
    """TPS warp: source → target. Returns callable warp(query) → warped.
    Uses small smoothing to prevent singular matrix with dense control points."""
    warp_x = RBFInterpolator(source_pts, target_pts[:, 0],
                             kernel='thin_plate_spline', smoothing=1e-5)
    warp_y = RBFInterpolator(source_pts, target_pts[:, 1],
                             kernel='thin_plate_spline', smoothing=1e-5)
    def warp(qp):
        qp = np.asarray(qp, dtype=np.float64)
        return np.column_stack([warp_x(qp), warp_y(qp)])
    return warp


###############################################################################
# 4. DINOSAR-STYLE QUADRILATERAL GRID
###############################################################################

def create_grid_inside_contour(contour, n_rows=8, n_cols=10, margin=0.05):
    """Create a regular grid of quadrilateral cells inside the mean contour.

    Returns grid_info dict with x_lines, y_lines, n_rows, n_cols,
    and cell_inside flags indicating which cells fall inside the contour.
    """
    pts = contour[:-1] if len(contour) > 1 and np.array_equal(contour[0], contour[-1]) else contour
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    w, h = xmax - xmin, ymax - ymin
    xmin -= margin * w; xmax += margin * w
    ymin -= margin * h; ymax += margin * h

    x_lines = np.linspace(xmin, xmax, n_cols + 1)
    y_lines = np.linspace(ymin, ymax, n_rows + 1)

    # determine which cells are inside the contour
    contour_scaled = (pts * 10000).astype(np.int32)
    cell_inside = np.zeros((n_rows, n_cols), dtype=bool)
    for r in range(n_rows):
        for c in range(n_cols):
            cx = (x_lines[c] + x_lines[c + 1]) / 2
            cy = (y_lines[r] + y_lines[r + 1]) / 2
            pt = (int(cx * 10000), int(cy * 10000))
            if cv2.pointPolygonTest(contour_scaled, pt, False) >= 0:
                cell_inside[r, c] = True

    n_inside = cell_inside.sum()
    logging.info(f"Grid: {n_rows}×{n_cols} = {n_rows * n_cols} cells, "
                 f"{n_inside} inside contour")

    return {
        'x_lines': x_lines, 'y_lines': y_lines,
        'n_rows': n_rows, 'n_cols': n_cols,
        'cell_inside': cell_inside
    }


def get_grid_intersections(grid_info):
    """Get all grid line intersection points as (n_rows+1)×(n_cols+1)×2 array."""
    xx, yy = np.meshgrid(grid_info['x_lines'], grid_info['y_lines'])
    return np.stack([xx, yy], axis=-1)  # shape (n_rows+1, n_cols+1, 2)


def warp_grid_to_specimen(grid_info, warp_fn):
    """Warp all grid intersections from mean space to specimen image space.
    Returns warped intersections array (n_rows+1, n_cols+1, 2)."""
    intersections = get_grid_intersections(grid_info)
    shape = intersections.shape
    flat = intersections.reshape(-1, 2)
    warped_flat = warp_fn(flat)
    return warped_flat.reshape(shape)


def build_cell_mask(warped_intersections, row, col, h, w, fg_mask):
    """Build a pixel mask for one warped quadrilateral cell, intersected
    with the foreground mask.

    The 4 corners of cell (row, col) in specimen image space are:
      TL = warped_intersections[row, col]
      TR = warped_intersections[row, col+1]
      BR = warped_intersections[row+1, col+1]
      BL = warped_intersections[row+1, col]

    Returns binary mask (h, w) with 1s inside the warped quad AND fg_mask.
    """
    corners = np.array([
        warped_intersections[row, col],
        warped_intersections[row, col + 1],
        warped_intersections[row + 1, col + 1],
        warped_intersections[row + 1, col],
    ], dtype=np.int32)

    cell_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(cell_mask, [corners], 1)

    # CRITICAL: intersect with foreground mask — excludes legs, background, etc.
    cell_mask = cell_mask & fg_mask

    return cell_mask


###############################################################################
# 5. PER-CELL COLOR EXTRACTION
###############################################################################

def extract_cell_color_features(img_lab, cell_mask, specimen_lab_means,
                                n_color_bins=8):
    """Extract color vector from one warped grid cell.

    Features (11 per cell — v2.1 adds absolute LAB means):
      dL_rel, da_rel, db_rel : specimen-relative L*a*b* means
      L_mean_abs, a_mean_abs, b_mean_abs : absolute L*a*b* means (v2.1)
      L_std, a_std, b_std    : within-cell color standard deviations
      color_entropy          : Shannon entropy of quantized color distribution
      dominant_proportion    : proportion of most common color bin

    The absolute L_mean_abs / a_mean_abs / b_mean_abs columns are new in
    v2.1.  They are the untransformed per-cell CIELAB means (skimage
    convention: L in [0,100], a and b in ~[-128, 127]).  These enable
    honest ancestral-state reconstruction of per-cell colour at internal
    nodes without needing to re-combine relative values with specimen
    means from a separate CSV.

    Only pixels where cell_mask > 0 are used.
    """
    fg = cell_mask > 0
    if fg.sum() < 25:
        return None

    features = []

    # L*a*b* values for foreground pixels in this cell
    L_vals = img_lab[:, :, 0][fg]
    a_vals = img_lab[:, :, 1][fg]
    b_vals = img_lab[:, :, 2][fg]

    L_mean_abs = float(L_vals.mean())
    a_mean_abs = float(a_vals.mean())
    b_mean_abs = float(b_vals.mean())

    # specimen-relative means (camera-robust)
    features.append(L_mean_abs - specimen_lab_means[0])  # dL_rel
    features.append(a_mean_abs - specimen_lab_means[1])  # da_rel
    features.append(b_mean_abs - specimen_lab_means[2])  # db_rel

    # absolute L*a*b* means (v2.1 — for ancestral state reconstruction
    # of per-cell colour without needing specimen_lab_means separately)
    features.append(L_mean_abs)
    features.append(a_mean_abs)
    features.append(b_mean_abs)

    # within-cell standard deviations (color variability)
    features.append(float(L_vals.std()) if len(L_vals) > 1 else 0.0)  # L_std
    features.append(float(a_vals.std()) if len(a_vals) > 1 else 0.0)  # a_std
    features.append(float(b_vals.std()) if len(b_vals) > 1 else 0.0)  # b_std

    # color entropy — quantize L*a*b* into bins and compute Shannon entropy
    # Use L channel binned into n_color_bins levels
    L_q = ((L_vals - L_vals.min()) / max(1, L_vals.max() - L_vals.min())
           * (n_color_bins - 1)).astype(int).clip(0, n_color_bins - 1)
    hist = np.bincount(L_q, minlength=n_color_bins).astype(float)
    hist = hist / hist.sum()
    hist_nz = hist[hist > 0]
    entropy = float(-np.sum(hist_nz * np.log2(hist_nz)))
    features.append(entropy)

    # dominant proportion — fraction of pixels in the most common L bin
    dominant_prop = float(hist.max())
    features.append(dominant_prop)

    return features


COLOR_CELL_FEATURE_NAMES = [
    'dL_rel', 'da_rel', 'db_rel',
    'L_mean_abs', 'a_mean_abs', 'b_mean_abs',  # v2.1 absolute per-cell LAB
    'L_std', 'a_std', 'b_std',
    'color_entropy', 'dominant_proportion'
]


###############################################################################
# 5b. SPECIMEN-LEVEL HUE FEATURES (v2)
###############################################################################

HUE_FEATURE_NAMES = [
    'L_mean', 'a_mean', 'b_mean',
    'chroma_ab', 'hue_ab_sin', 'hue_ab_cos'
]


def extract_specimen_hue_features(img_lab, fg_bool, lab_offset=None):
    """Compute 6 specimen-level absolute colour features from LAB.

    Returns list:
      [L_mean, a_mean, b_mean, chroma_ab, hue_ab_sin, hue_ab_cos]

    The hue_ab angle is the L*a*b* chromaticity angle atan2(b*, a*).
    Decomposed into sin/cos so the circularity of hue is preserved
    in Euclidean feature space (no 0°/360° wrap artefacts).

    Scientific rationale
    --------------------
    In the (a*, b*) plane:
      - chroma (radius) = saturation / colourfulness
      - hue   (angle)   = perceptual colour-wheel position
          0° ≈ red, ~90° ≈ yellow, ~180° ≈ green, ~270° ≈ blue
    Both are approximately perceptually-uniform in CIELAB.  Clustering
    on (hue_sin, hue_cos, chroma) gives you the classic "all greens
    together, all reds together" behaviour — independent of how bright
    or pattern-rich the specimen is.

    a*/b* ENCODING — read before changing this function
    ---------------------------------------------------
    chroma and hue are NON-LINEAR in a*/b*, so the encoding matters.
    skimage returns a*/b* centred on 0 (neutral grey = 0), which is what
    the maths below assumes.  OpenCV's COLOR_BGR2LAB on an 8-bit image
    instead returns a*/b* offset by +128 (neutral grey = 128).  Feeding
    the OpenCV encoding in uncentred collapses every specimen onto
    atan2(128, 128) ~ 45 deg and puts a floor of sqrt(128^2 + 128^2) ~ 181
    under chroma — hue stops discriminating green from brown entirely.
    `lab_offset` is subtracted from a*/b* before either is computed.

    Parameters
    ----------
    img_lab : (H, W, 3) float64 array of L*a*b* values
    fg_bool : (H, W) boolean foreground mask
    lab_offset : float or None
        Neutral-grey offset of the a*/b* channels.  Pass 128.0 for OpenCV
        8-bit LAB, 0.0 for skimage LAB.  None (default) auto-detects:
        centred data straddles zero, OpenCV-encoded data sits near +128.

    Returns
    -------
    list of 6 floats
    """
    if fg_bool.sum() < 50:
        return [0.0] * 6

    L_vals = img_lab[:, :, 0][fg_bool]
    a_vals = img_lab[:, :, 1][fg_bool]
    b_vals = img_lab[:, :, 2][fg_bool]

    if lab_offset is None:
        # Centred a*/b* straddle zero; OpenCV-encoded ones sit near +128.
        # A real specimen never averages a* > 64 on the centred scale.
        lab_offset = 128.0 if (a_vals.min() >= 0.0 and b_vals.min() >= 0.0
                               and max(a_vals.mean(), b_vals.mean()) > 64.0) \
                     else 0.0

    L_mean = float(L_vals.mean())
    a_mean = float(a_vals.mean()) - float(lab_offset)
    b_mean = float(b_vals.mean()) - float(lab_offset)

    # Chroma = radial distance in the (a*, b*) plane.
    # How saturated the colour is, independent of hue and lightness.
    chroma_ab = float(np.sqrt(a_mean ** 2 + b_mean ** 2))

    # Hue angle in the (a*, b*) plane.  Decompose to sin/cos so that
    # 179° and -179° are neighbours in feature space (circularity).
    hue_ab = np.arctan2(b_mean, a_mean)
    hue_ab_sin = float(np.sin(hue_ab))
    hue_ab_cos = float(np.cos(hue_ab))

    return [L_mean, a_mean, b_mean, chroma_ab, hue_ab_sin, hue_ab_cos]


def block_zscore(X, eps=1e-8):
    """Column-wise z-score.  Used for block-wise scaling in combined mode.

    After this transformation every column has unit variance, so the
    total variance contributed by a block is exactly equal to its
    number of columns.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def combine_pattern_and_hue(X_pattern, X_hue, hue_weight=1.0):
    """Concatenate pattern and hue feature matrices with balanced block
    variance.

    After z-scoring each block separately, both blocks have per-column
    variance = 1.  We then rescale the hue block so that its TOTAL block
    variance equals ``hue_weight × (pattern total block variance)``.

    With hue_weight = 1.0 (default), pattern and hue contribute equally
    regardless of the column-count asymmetry (~500 pattern vs 6 hue).

    With hue_weight = 2.0, the hue block contributes twice as much
    variance as the pattern block.  With hue_weight = 0.5, the pattern
    block contributes twice as much as the hue block.
    """
    Zp = block_zscore(X_pattern)
    Zh = block_zscore(X_hue)

    n_p = Zp.shape[1]
    n_h = Zh.shape[1]

    # After z-scoring, block p has total variance = n_p, block h = n_h.
    # We want block_h_total_var == hue_weight * n_p, so we multiply
    # every hue column by sqrt(hue_weight * n_p / n_h).
    if n_h > 0 and n_p > 0:
        scale = float(np.sqrt(max(hue_weight, 1e-12) * n_p / n_h))
        Zh = Zh * scale

    return np.concatenate([Zp, Zh], axis=1)


###############################################################################
# 6. PROCESS ONE SPECIMEN
###############################################################################

def process_specimen(fn_gpa, specimen_contour, mean_shape, grid_info,
                     image_dir, mask_lookup, ctrl_step=5, color_bins=8):
    """Process one specimen: TPS warp grid, extract per-cell color."""

    # resolve image path
    orig_fn = get_original_filename(fn_gpa)
    ip = resolve_image_path(fn_gpa, image_dir)
    if ip is None:
        logging.warning(f"  Image not found: {fn_gpa} / {orig_fn}")
        return None

    img = cv2.imread(ip)
    if img is None:
        return None
    h, w = img.shape[:2]
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)

    # get foreground mask from original COCO
    fg_mask = np.zeros((h, w), np.uint8)
    ml = mask_lookup.get(orig_fn)
    if ml and maskUtils:
        fg_mask = decode_mask(ml['ann'], h, w)
    if fg_mask.sum() < 100:
        logging.warning(f"  No mask for {orig_fn}")
        return None

    fg_bool = fg_mask > 0

    # specimen-wide L*a*b* means for relative color
    specimen_lab_means = np.array([
        float(img_lab[:, :, ci][fg_bool].mean()) for ci in range(3)
    ])

    # v2: specimen-level hue features (used by --color_mode hue/combined)
    # Computed unconditionally — cheap, and lets main() select a mode
    # without re-reading the image.
    # img_lab came from cv2.COLOR_BGR2LAB on 8-bit input -> a*/b* are +128
    hue_feats = extract_specimen_hue_features(img_lab, fg_bool, lab_offset=128.0)

    # subsample semi-landmarks for TPS control points
    # using ALL 100 causes singular matrix (too dense in Procrustes space)
    # every 5th = 20 control points — more than enough for smooth warp
    spec_pts = specimen_contour
    mean_pts = mean_shape
    n = min(spec_pts.shape[0], mean_pts.shape[0])
    if spec_pts.shape[0] != n or mean_pts.shape[0] != n:
        from scipy.interpolate import interp1d
        def resample(c, nn):
            d = np.sqrt(np.sum(np.diff(c, axis=0)**2, axis=1))
            cd = np.insert(np.cumsum(d), 0, 0)
            if cd[-1] == 0: return c[:nn]
            fx = interp1d(cd, c[:, 0]); fy = interp1d(cd, c[:, 1])
            return np.column_stack([fx(np.linspace(0, cd[-1], nn)),
                                    fy(np.linspace(0, cd[-1], nn))])
        spec_pts = resample(spec_pts, n)
        mean_pts = resample(mean_pts, n)

    if n < 4:
        return None

    # pick every ctrl_step-th landmark as TPS control point
    ctrl_idx = list(range(0, n, ctrl_step))
    if len(ctrl_idx) < 4:
        ctrl_idx = list(range(0, n, max(1, n // 10)))
    ctrl_mean = mean_pts[ctrl_idx]
    ctrl_spec = spec_pts[ctrl_idx]

    # build TPS warp: mean → specimen
    try:
        warp = build_tps_warp(ctrl_mean, ctrl_spec)
    except Exception as e:
        logging.warning(f"  TPS failed for {fn_gpa}: {e}")
        return None

    # warp grid intersections
    warped_ints = warp_grid_to_specimen(grid_info, warp)
    cell_inside = grid_info['cell_inside']
    n_rows, n_cols = grid_info['n_rows'], grid_info['n_cols']

    # extract features per inside cell
    # v2.1: was 8, now 11 (added 3 absolute per-cell LAB means)
    n_feats = len(COLOR_CELL_FEATURE_NAMES)
    cell_features = []
    for r in range(n_rows):
        for c in range(n_cols):
            if not cell_inside[r, c]:
                cell_features.append([np.nan] * n_feats)
                continue
            cm = build_cell_mask(warped_ints, r, c, h, w, fg_mask)
            feats = extract_cell_color_features(img_lab, cm,
                                                 specimen_lab_means,
                                                 n_color_bins=color_bins)
            if feats is None:
                cell_features.append([np.nan] * n_feats)
            else:
                cell_features.append(feats)

    fv = np.array(cell_features).flatten()
    # v2: 6-tuple now — trailing hue_feats is the specimen-level block.
    return (fn_gpa, fv, img, fg_mask, warped_ints, hue_feats)


###############################################################################
# 7. VISUALISATION
###############################################################################

def viz_mean_grid(mean_shape, grid_info, od, cat, ctrl_step=5):
    """Plot mean shape with grid lines and control landmarks marked."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ms = mean_shape
    if len(ms) > 1 and not np.array_equal(ms[0], ms[-1]):
        ms = np.vstack([ms, ms[0]])
    ax.plot(ms[:, 0], ms[:, 1], 'k-', lw=2.5, label='Mean shape')

    # show ALL semi-landmarks as small dots
    ax.scatter(mean_shape[:, 0], mean_shape[:, 1], c='gray', s=8, zorder=4,
              alpha=0.5, label=f'{mean_shape.shape[0]} semi-landmarks')

    # show TPS control points (subsampled) as large red dots
    ctrl_idx = list(range(0, mean_shape.shape[0], ctrl_step))
    ctrl_pts = mean_shape[ctrl_idx]
    ax.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], c='red', s=60, zorder=5,
              label=f'{len(ctrl_pts)} TPS control pts (every {ctrl_step}th)')

    # grid lines
    xl, yl = grid_info['x_lines'], grid_info['y_lines']
    for x in xl:
        ax.plot([x, x], [yl[0], yl[-1]], 'b-', alpha=0.3, lw=0.5)
    for y in yl:
        ax.plot([xl[0], xl[-1]], [y, y], 'b-', alpha=0.3, lw=0.5)

    # inside cells
    ci = grid_info['cell_inside']
    for r in range(ci.shape[0]):
        for c in range(ci.shape[1]):
            if ci[r, c]:
                cx = (xl[c] + xl[c+1]) / 2
                cy = (yl[r] + yl[r+1]) / 2
                ax.plot(cx, cy, 'b.', ms=4)

    n_in = ci.sum()
    ax.set_aspect('equal', 'box'); ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.set_title(f"Mean shape + grid — {cat}\n"
                 f"({len(ctrl_pts)} TPS control pts, "
                 f"{ci.shape[0]}×{ci.shape[1]} grid, {n_in} cells inside)")
    plt.tight_layout()
    plt.savefig(os.path.join(od, f"mean_grid_{cat}.png"), dpi=300)
    plt.close()


def viz_warped_grid_on_specimen(fn, img, fg_mask, warped_ints, grid_info,
                                od, cat):
    """Draw warped quadrilateral grid on a specimen image."""
    vis = img.copy()
    vis[fg_mask == 0] = vis[fg_mask == 0] // 3
    ci = grid_info['cell_inside']
    n_rows, n_cols = grid_info['n_rows'], grid_info['n_cols']

    for r in range(n_rows):
        for c in range(n_cols):
            if not ci[r, c]:
                continue
            corners = np.array([
                warped_ints[r, c], warped_ints[r, c+1],
                warped_ints[r+1, c+1], warped_ints[r+1, c]
            ], dtype=np.int32)
            cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
            centre = corners.mean(axis=0).astype(int)
            cv2.circle(vis, tuple(centre), 3, (0, 0, 255), -1)

    bn = os.path.basename(fn).replace('.', '_')
    out = os.path.join(od, f"warped_grid_{bn}.png")
    cv2.imwrite(out, vis)


def mk_thumb(img, mask, sz=(80, 80)):
    bm = (mask > 0).astype(np.uint8)
    fg = np.zeros_like(img); fg[bm == 1] = img[bm == 1]
    rgb = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    pi = Image.fromarray(rgb).convert("RGBA")
    pi.putalpha(Image.fromarray((bm * 255).astype(np.uint8)).convert("L"))
    pi.thumbnail(sz, Image.LANCZOS)
    ni = Image.new('RGBA', sz, (255, 255, 255, 0))
    ni.paste(pi, ((sz[0] - pi.size[0]) // 2, (sz[1] - pi.size[1]) // 2))
    return np.array(ni)


###############################################################################
# 8. CLI + MAIN
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(
        description='Homology-aware COLOR phenomics via TPS-warped quad grid')
    p.add_argument('--gpa_dir', required=True,
                   help='V34 GPA output directory (with gpa_mean.csv, '
                        'back_transformed_coco.json)')
    p.add_argument('--json', required=True,
                   help='ORIGINAL COCO JSON (for foreground masks)')
    p.add_argument('--image_dir', required=True,
                   help='Directory containing specimen images. Can point to '
                        'raw images OR V20 color-segmented output directory '
                        '(labavg_* images) for camera-robust color analysis.')
    p.add_argument('--output_dir', default='./color_homology_out')
    p.add_argument('--category_name', default='',
                   help='Category name matching the original COCO JSON '
                        '(e.g. left_elytron, pronotum)')
    p.add_argument('--grid_rows', type=int, default=8,
                   help='Number of grid rows (default 8)')
    p.add_argument('--grid_cols', type=int, default=10,
                   help='Number of grid columns (default 10)')
    p.add_argument('--landmark_step', type=int, default=5,
                   help='Use every Nth semi-landmark as TPS control point '
                        '(default 5 = 20 from 100 landmarks)')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--n_clusters', type=int, default=0,
                   help='Number of clusters (0 = auto via silhouette score)')
    p.add_argument('--viz_specimens', type=int, default=5)
    p.add_argument('--umap_n_neighbors', type=int, default=15)
    p.add_argument('--color_bins', type=int, default=8,
                   help='Number of bins for within-cell color entropy '
                        'calculation (default 8)')
    # --- v2 additions ---
    p.add_argument('--color_mode', type=str, default='combined',
                   choices=['pattern', 'hue', 'combined'],
                   help="What to compute. 'pattern' = per-cell specimen-"
                        "relative LAB (v1 behaviour, clusters by colour "
                        "PATTERN). 'hue' = specimen-level absolute LAB + "
                        "chromaticity angle sin/cos (clusters by overall "
                        "COLOUR). 'combined' = both, block-z-scored. "
                        "Default: combined.")
    p.add_argument('--hue_weight', type=float, default=1.0,
                   help='Variance weight of the hue block relative to the '
                        'pattern block in combined mode. 1.0 (default) = '
                        'equal block weight, counteracts the column-count '
                        'asymmetry (~500 pattern cols vs 6 hue cols). '
                        '>1.0 pushes clustering toward hue; <1.0 toward '
                        'pattern. Ignored in pattern/hue-only modes.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"=== Texture Phenomics (Homology-Aware) ===")
    logging.info(f"GPA dir:   {args.gpa_dir}")
    logging.info(f"COCO JSON: {args.json}")
    logging.info(f"Image dir: {args.image_dir}")
    logging.info(f"Category:  {args.category_name}")
    logging.info(f"Grid:      {args.grid_rows}×{args.grid_cols}")
    logging.info(f"Ctrl pts:  every {args.landmark_step}th landmark")
    logging.info(f"Color mode: {args.color_mode}"
                 + (f" (hue_weight={args.hue_weight:g})"
                    if args.color_mode == 'combined' else ""))

    # feature names for color analysis
    feature_names = COLOR_CELL_FEATURE_NAMES
    n_feats_per_cell = len(feature_names)

    # --- Load GPA outputs ---
    mean_shape, specimens = load_gpa_outputs(args.gpa_dir)

    # --- Build mask lookup from ORIGINAL COCO ---
    cat_name = args.category_name or 'category'
    mask_lookup, cat_id = build_mask_lookup(args.json, cat_name)
    logging.info(f"Category '{cat_name}' → id={cat_id}, "
                 f"{len(mask_lookup)} masks available")

    # --- Build grid ---
    grid_info = create_grid_inside_contour(
        mean_shape, n_rows=args.grid_rows, n_cols=args.grid_cols)

    ctrl_step = args.landmark_step
    n_ctrl = len(range(0, mean_shape.shape[0], ctrl_step))
    logging.info(f"TPS control points: {n_ctrl} (every {ctrl_step}th of "
                 f"{mean_shape.shape[0]})")

    viz_mean_grid(mean_shape, grid_info, args.output_dir, cat_name,
                  ctrl_step=ctrl_step)

    # --- Process specimens ---
    color_bins = args.color_bins
    n_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    logging.info(f"Processing {len(specimens)} specimens ({n_workers} workers)")

    def proc(item):
        fn, contour = item
        return process_specimen(fn, contour, mean_shape, grid_info,
                                args.image_dir, mask_lookup,
                                ctrl_step=ctrl_step,
                                color_bins=color_bins)

    items = list(specimens.items())
    results = []
    if n_workers <= 1:
        for i, item in enumerate(items):
            r = proc(item)
            if r: results.append(r)
            if (i + 1) % 20 == 0:
                logging.info(f"  {i+1}/{len(items)} done")
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(proc, item): item for item in items}
            for i, f in enumerate(as_completed(futures)):
                r = f.result()
                if r: results.append(r)
                if (i + 1) % 20 == 0:
                    logging.info(f"  {i+1}/{len(items)} done")

    if not results:
        logging.error("No specimens processed. Check --category_name matches "
                      "the original COCO JSON categories, and images exist.")
        return

    logging.info(f"Successfully processed {len(results)} / {len(specimens)} specimens")

    # save grid overlays
    viz_dir = os.path.join(args.output_dir, 'grid_overlays')
    os.makedirs(viz_dir, exist_ok=True)
    # v2: 6-tuple — use *_rest to stay robust if tuple ever grows.
    for fn, fv, img, mask, wi, *_rest in results[:args.viz_specimens]:
        viz_warped_grid_on_specimen(fn, img, mask, wi, grid_info, viz_dir, cat_name)
    logging.info(f"Saved {min(len(results), args.viz_specimens)} grid overlays")

    # --- Assemble feature matrix ---
    fns = [r[0] for r in results]
    X_raw = np.array([r[1] for r in results])            # per-cell features
    all_imgs = [r[2] for r in results]
    all_masks = [r[3] for r in results]
    X_hue_raw = np.array([r[5] for r in results])        # v2: specimen-level hue block

    ci = grid_info['cell_inside']
    n_cells = ci.sum()

    # v2: filename tag so the three modes can coexist in the same output dir
    mode = args.color_mode
    mode_tag = f"{cat_name}_{mode}"
    if mode == 'combined':
        mode_tag = f"{mode_tag}_hw{args.hue_weight:g}"

    # ---- Build the per-cell pattern feature matrix (shared by all modes) ----
    pattern_col_names = []
    for r in range(ci.shape[0]):
        for c in range(ci.shape[1]):
            if ci[r, c]:
                for fn in feature_names:
                    pattern_col_names.append(f'r{r}c{c}_{fn}')

    inside_indices = []
    for r in range(ci.shape[0]):
        for c in range(ci.shape[1]):
            base = (r * ci.shape[1] + c) * n_feats_per_cell
            if ci[r, c]:
                inside_indices.extend(range(base, base + n_feats_per_cell))

    X_pattern = X_raw[:, inside_indices]
    # impute NaN in the pattern block
    pat_col_means = np.nanmean(X_pattern, axis=0)
    pat_col_means = np.where(np.isnan(pat_col_means), 0, pat_col_means)
    for j in range(X_pattern.shape[1]):
        m = np.isnan(X_pattern[:, j])
        X_pattern[m, j] = pat_col_means[j]

    # impute NaN in the hue block (rare: only when whole mask was tiny)
    hue_col_means = np.nanmean(X_hue_raw, axis=0)
    hue_col_means = np.where(np.isnan(hue_col_means), 0, hue_col_means)
    X_hue_filled = X_hue_raw.copy()
    for j in range(X_hue_filled.shape[1]):
        m = np.isnan(X_hue_filled[:, j])
        X_hue_filled[m, j] = hue_col_means[j]

    # ---- Dispatch on mode to produce X_filled, col_names, X_scaled ----
    n_pattern_cols = X_pattern.shape[1]  # needed for spatial heatmap

    # ---- v2.1: separate PCA-participating features from CSV-saved features ----
    # The absolute per-cell LAB means (L_mean_abs, a_mean_abs, b_mean_abs) are
    # written to the CSV for honest downstream ancestral state reconstruction,
    # but they are EXCLUDED from the PCA in pattern mode so that pattern-PCs
    # remain camera-robust (their whole purpose).  In 'hue' mode this doesn't
    # apply (hue features are specimen-level).  In 'combined' mode we also
    # exclude the absolute LABs from the pattern block of the PCA input.
    abs_lab_names = ('L_mean_abs', 'a_mean_abs', 'b_mean_abs')
    pattern_pca_mask = np.array([
        not any(col.endswith(f'_{n}') for n in abs_lab_names)
        for col in pattern_col_names
    ])
    X_pattern_for_pca = X_pattern[:, pattern_pca_mask]
    pattern_col_names_for_pca = [
        c for c, keep in zip(pattern_col_names, pattern_pca_mask) if keep
    ]
    logging.info(f"[v2.1] PCA uses {X_pattern_for_pca.shape[1]} of "
                 f"{X_pattern.shape[1]} pattern columns "
                 f"(absolute per-cell LAB excluded from PCA, still in CSV)")

    if mode == 'pattern':
        X_filled = X_pattern             # CSV gets EVERYTHING, including abs LAB
        col_names = pattern_col_names
        pca_input_col_names = pattern_col_names_for_pca  # v2.1: matches pca.components_
        # PCA gets the camera-robust subset only (v2.1 fix)
        X_scaled = StandardScaler().fit_transform(X_pattern_for_pca)
        logging.info(f"[pattern mode] CSV: {X_filled.shape} full columns; "
                     f"PCA: {X_pattern_for_pca.shape} relative-only columns")

    elif mode == 'hue':
        X_filled = X_hue_filled
        col_names = list(HUE_FEATURE_NAMES)
        pca_input_col_names = list(HUE_FEATURE_NAMES)
        # z-score (chroma and LAB means are on very different scales)
        X_scaled = StandardScaler().fit_transform(X_filled)
        logging.info(f"[hue mode] Feature matrix: {X_filled.shape} "
                     f"(specimen-level absolute LAB + hue-angle sin/cos)")

    elif mode == 'combined':
        # block-z-score each separately, then re-weight so hue block
        # contributes hue_weight × pattern-block variance.
        # v2.1: combined mode's pattern block also excludes abs-LAB from PCA
        X_scaled = combine_pattern_and_hue(
            X_pattern_for_pca, X_hue_filled, hue_weight=args.hue_weight)
        # For saving features pre-scaling we just concatenate everything.
        X_filled = np.concatenate([X_pattern, X_hue_filled], axis=1)
        col_names = pattern_col_names + list(HUE_FEATURE_NAMES)
        pca_input_col_names = pattern_col_names_for_pca + list(HUE_FEATURE_NAMES)
        logging.info(f"[combined mode] CSV: {X_filled.shape[1]} cols "
                     f"({X_pattern.shape[1]} pattern + {X_hue_filled.shape[1]} hue); "
                     f"PCA: {X_pattern_for_pca.shape[1]}+{X_hue_filled.shape[1]}, "
                     f"hue_weight={args.hue_weight:g}")
        logging.info(f"[combined mode] Scaled feature matrix: {X_scaled.shape}")

    else:
        raise ValueError(f"Unknown --color_mode: {mode}")

    nan_frac = np.isnan(X_filled).mean()
    logging.info(f"Feature matrix: {X_filled.shape} (NaN fraction after impute: {nan_frac:.3f})")

    # save raw features (pre-scaling) for reproducibility
    df = pd.DataFrame(X_filled, columns=col_names)
    df.insert(0, 'filename', fns)
    df.to_csv(os.path.join(args.output_dir,
              f"color_homology_features_{mode_tag}.csv"), index=False)

    # --- PCA ---
    # Note: X_scaled is already appropriately scaled per-mode.
    # We do NOT re-StandardScaler here for combined mode (would un-do
    # the block weighting).  For pattern/hue we already used StandardScaler.
    n_comp = min(X_scaled.shape[0] - 1, X_scaled.shape[1], 20)
    n_comp = max(n_comp, 2)
    pca = PCA(n_components=n_comp)
    pc = pca.fit_transform(X_scaled)
    logging.info(f"PCA variance: {pca.explained_variance_ratio_[:5].round(3)}")

    # --- PCA PLOT ---
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(f"Color PCA ({mode}) — {cat_name}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    # CRITICAL: invisible scatter sets axis limits (AnnotationBbox doesn't autoscale)
    ax.scatter(pc[:, 0], pc[:, 1], alpha=0)
    for i in range(len(pc)):
        t = mk_thumb(all_imgs[i], all_masks[i])
        ax.add_artist(AnnotationBbox(
            OffsetImage(t, zoom=0.6), (pc[i, 0], pc[i, 1]), frameon=False))
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f"color_pca_homology_{mode_tag}.png"), dpi=300)
    plt.close()

    # --- UMAP ---
    usc = None
    if HAS_UMAP and len(results) > 5:
        nn = min(args.umap_n_neighbors, len(results) - 1)
        if nn > 1:
            um = umap_lib.UMAP(n_components=2, n_neighbors=nn, random_state=42)
            usc = um.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_title(f"Color UMAP ({mode}) — {cat_name}")
            ax.scatter(usc[:, 0], usc[:, 1], alpha=0)
            for i in range(len(usc)):
                t = mk_thumb(all_imgs[i], all_masks[i])
                ax.add_artist(AnnotationBbox(
                    OffsetImage(t, zoom=0.6), (usc[i, 0], usc[i, 1]), frameon=False))
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir,
                        f"color_umap_homology_{mode_tag}.png"), dpi=300)
            plt.close()

    # --- SILHOUETTE-BASED OPTIMAL K ---
    from sklearn.metrics import silhouette_score
    best_k, best_sil = 2, -1
    sil_scores = {}
    max_k = min(10, len(results) - 1)
    for k in range(2, max_k + 1):
        Z_test = linkage(X_scaled, 'ward')
        lab_test = fcluster(Z_test, k, 'maxclust')
        if len(set(lab_test)) < 2:
            continue
        s = silhouette_score(X_scaled, lab_test)
        sil_scores[k] = s
        if s > best_sil:
            best_sil = s
            best_k = k
    logging.info(f"Silhouette scores: {sil_scores}")
    logging.info(f"Optimal k={best_k} (silhouette={best_sil:.3f})")

    # use optimal k if user didn't specify, otherwise use user's choice
    use_k = best_k if args.n_clusters <= 0 else args.n_clusters

    # --- CLUSTERING + DENDROGRAM ---
    import seaborn as sns
    Z = linkage(X_scaled, 'ward')
    labels = fcluster(Z, use_k, 'maxclust')

    # silhouette plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(sil_scores.keys())
    ax.plot(ks, [sil_scores[k] for k in ks], 'o-', color='steelblue', lw=2)
    ax.axvline(best_k, color='red', ls='--', label=f'optimal k={best_k} (sil={best_sil:.3f})')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette score')
    ax.set_title(f'Silhouette analysis ({mode}) — {cat_name}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f"color_silhouette_{mode_tag}.png"), dpi=200)
    plt.close()

    # dendrogram
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(Z, labels=fns, leaf_rotation=90, leaf_font_size=4, ax=ax)
    ax.set_title(f"Color dendrogram ({mode}) — {cat_name} "
                 f"(k={use_k}, silhouette={best_sil:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f"color_dendrogram_{mode_tag}.png"), dpi=200)
    plt.close()

    # cluster scatter on PCA
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette(None, len(np.unique(labels)))
    for lab, col in zip(np.unique(labels), palette):
        m = labels == lab
        ax.scatter(pc[m, 0], pc[m, 1], c=[col], label=f'Cluster {lab}',
                   alpha=0.7, s=60)
    for i in range(len(pc)):
        t = mk_thumb(all_imgs[i], all_masks[i], sz=(50, 50))
        ax.add_artist(AnnotationBbox(
            OffsetImage(t, zoom=0.5), (pc[i, 0], pc[i, 1]), frameon=False))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Color clusters on PCA ({mode}) — {cat_name} "
                 f"(k={use_k}, sil={best_sil:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f"color_clusters_{mode_tag}.png"), dpi=300)
    plt.close()

    # --- HEATMAP (all specimens) ---
    hm_df = pd.DataFrame(X_filled, columns=col_names)
    hm_df.insert(0, 'filename', fns)
    n_spec = len(hm_df)
    feat_hm_cols = [c for c in hm_df.columns if c != 'filename']
    sc_hm = StandardScaler()
    vals_z = sc_hm.fit_transform(hm_df[feat_hm_cols].values)
    row_height = max(0.12, min(0.3, 15.0 / n_spec))
    fig_h = max(8, n_spec * row_height)
    fontsize = max(3, min(8, int(400 / n_spec)))
    fig, ax = plt.subplots(figsize=(max(14, len(feat_hm_cols) * 0.3), fig_h))
    sns.heatmap(vals_z, xticklabels=feat_hm_cols,
                yticklabels=hm_df['filename'].values,
                cmap='RdBu_r', center=0, ax=ax)
    ax.set_title(f"Color features z-scored ({mode}) — {cat_name} — {n_spec} specimens")
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=5, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f"color_heatmap_{mode_tag}.png"), dpi=200)
    plt.close()
    logging.info(f"Heatmap saved: {n_spec} specimens × {len(feat_hm_cols)} features")

    # --- SAVE RESULTS ---
    res_df = pd.DataFrame({'filename': fns, 'cluster': labels,
                           'optimal_k': use_k, 'silhouette': best_sil})
    for i in range(min(10, pc.shape[1])):
        res_df[f'PC{i+1}'] = pc[:, i]
    if usc is not None:
        res_df['UMAP1'] = usc[:, 0]
        res_df['UMAP2'] = usc[:, 1]
    res_df.to_csv(os.path.join(args.output_dir,
                  f"color_analysis_homology_{mode_tag}.csv"), index=False)

    # loadings
    # v2.1 fix: loadings are indexed by the column names that actually fed
    # the PCA (pca_input_col_names), which may differ from the CSV-saved
    # column names (col_names) in pattern/combined modes where the
    # absolute per-cell LAB columns are written to the CSV but excluded
    # from the PCA input.
    pd.DataFrame(
        pca.components_[:min(5, n_comp)].T,
        columns=[f'PC{i+1}' for i in range(min(5, n_comp))],
        index=pca_input_col_names
    ).to_csv(os.path.join(args.output_dir,
             f"color_loadings_homology_{mode_tag}.csv"))

    # PCA variance explained
    pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'variance_explained': pca.explained_variance_ratio_,
        'cumulative': np.cumsum(pca.explained_variance_ratio_)
    }).to_csv(os.path.join(args.output_dir,
              f"color_pca_variance_{mode_tag}.csv"), index=False)

    # phylo export
    res_df.to_csv(os.path.join(args.output_dir,
                  f"color_traits_phylo_{mode_tag}.csv"), index=False)

    # --- SPATIAL PC LOADING HEATMAP on mean shape ---
    # Shows which anatomical regions drive each PC axis.
    # In 'hue' mode the features are specimen-level, not per-cell, so
    # there is no spatial structure to map — we skip the heatmap.
    # In 'combined' mode the pattern features come FIRST in the column
    # layout (pattern_col_names then HUE_FEATURE_NAMES), so the per-cell
    # loadings walk correctly; hue columns are trailing and ignored here.
    # v2.1: the PCA input uses only the 8 camera-robust features per cell,
    # not the 11-feature CSV layout.  Spatial indexing walks in 8-strides.
    n_feats_per_cell_for_pca = int(pattern_pca_mask.sum()) // n_cells
    n_pattern_cols_for_pca = X_pattern_for_pca.shape[1]
    if mode == 'hue':
        logging.info("[hue mode] Spatial PC heatmap skipped "
                     "(no per-cell features to map).")
    else:
        for pc_idx in range(min(3, n_comp)):
            pc_label = f'PC{pc_idx + 1}'
            pc_var = pca.explained_variance_ratio_[pc_idx] * 100

            # sum absolute loadings across features within each cell
            cell_importance = np.zeros((grid_info['n_rows'], grid_info['n_cols']))
            cell_counter = 0
            for r in range(grid_info['n_rows']):
                for c in range(grid_info['n_cols']):
                    if grid_info['cell_inside'][r, c]:
                        # loadings for this cell's features (pattern block only)
                        start = cell_counter * n_feats_per_cell_for_pca
                        end = start + n_feats_per_cell_for_pca
                        if end <= n_pattern_cols_for_pca:
                            cell_loadings = pca.components_[pc_idx, start:end]
                            cell_importance[r, c] = np.sum(cell_loadings ** 2)
                        cell_counter += 1

            # plot on mean shape
            fig, ax = plt.subplots(figsize=(10, 10))
            ms = mean_shape
            if len(ms) > 1 and not np.array_equal(ms[0], ms[-1]):
                ms = np.vstack([ms, ms[0]])
            ax.plot(ms[:, 0], ms[:, 1], 'k-', lw=2)

            xl, yl = grid_info['x_lines'], grid_info['y_lines']
            vmax = cell_importance[grid_info['cell_inside']].max()
            if vmax == 0: vmax = 1

            for r in range(grid_info['n_rows']):
                for c in range(grid_info['n_cols']):
                    if not grid_info['cell_inside'][r, c]:
                        continue
                    x0, x1 = xl[c], xl[c + 1]
                    y0, y1 = yl[r], yl[r + 1]
                    val = cell_importance[r, c] / vmax
                    color = plt.cm.hot(val)
                    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                         facecolor=color, edgecolor='gray',
                                         linewidth=0.5, alpha=0.8)
                    ax.add_patch(rect)

            sm = plt.cm.ScalarMappable(cmap=plt.cm.hot,
                                       norm=plt.Normalize(0, vmax))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Sum of squared loadings',
                         shrink=0.6)

            ax.set_aspect('equal', 'box')
            ax.invert_yaxis()
            ax.set_title(f"Color spatial contribution to {pc_label} ({pc_var:.1f}%) "
                         f"— {cat_name} [{mode}]\n"
                         f"Bright = high contribution to this color axis")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir,
                        f"color_spatial_{pc_label}_{mode_tag}.png"), dpi=300)
            plt.close()
            logging.info(f"Spatial heatmap saved for {pc_label}")

    logging.info(f"Done: {len(results)} specimens, mode={mode}, "
                 f"final feature matrix shape = {X_scaled.shape}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
