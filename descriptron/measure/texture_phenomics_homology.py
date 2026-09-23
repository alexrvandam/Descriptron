#!/usr/bin/env python3
"""
texture_phenomics_homology.py  (v2)
====================================
Homology-aware texture phenomics using TPS-warped quadrilateral grid cells.

Pipeline:
  1. Load V34 GPA output (gpa_mean.csv + back_transformed_coco.json)
  2. Build a regular grid inside the mean contour
  3. Use ALL semi-landmarks as TPS control points (full point correspondence)
  4. For each specimen: TPS-warp grid line intersections → build warped
     quadrilateral cell masks → intersect with foreground mask → extract
     texture per cell
  5. PCA / UMAP / clustering on the (specimens × cells×features) matrix

Grid cells are warped quadrilaterals (DINOSAR-style), NOT circles.
This ensures no gaps, no overlap, natural scaling with specimen size,
and each cell covers its full homologous region.

Usage:
  python texture_phenomics_homology.py \\
    --gpa_dir ./outputs/left_elytron \\
    --json annotations.json \\
    --image_dir ./images \\
    --output_dir ./texture_homology_out \\
    --category_name left_elytron \\
    --grid_rows 8 --grid_cols 10
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
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

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
    handlers=[logging.FileHandler("texture_homology.log"),
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
    """Resolve V34 GPA filename (e.g. 'Acalles_crypto_88.jpg_19') to actual
    image file path. V34 appends '_imageID' to filenames."""
    # try as-is first
    p = os.path.join(image_dir, fn_from_gpa)
    if os.path.isfile(p):
        return p

    # strip the trailing _NNN (image ID appended by V34)
    # pattern: everything up to the last _digits
    m = re.match(r'^(.+\.\w+)_\d+$', fn_from_gpa)
    if m:
        base = m.group(1)
        p = os.path.join(image_dir, base)
        if os.path.isfile(p):
            return p

    # try replacing _jpg_ with .jpg (in case of mangled extensions)
    for ext in ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp']:
        mangled = fn_from_gpa.replace(f'_{ext}_', f'.{ext}')
        if mangled != fn_from_gpa:
            # strip trailing _NNN
            m2 = re.match(r'^(.+\.\w+)_\d+$', mangled)
            candidate = m2.group(1) if m2 else mangled
            p = os.path.join(image_dir, candidate)
            if os.path.isfile(p):
                return p

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
# 5. PER-CELL TEXTURE EXTRACTION
###############################################################################

def _masked_glcm_single(gray_q, mask_bool, distance, angle, levels=16):
    """GLCM where BOTH pixels in each pair must be inside the mask."""
    h, w = gray_q.shape
    dx = int(round(distance * np.cos(angle)))
    dy = int(round(-distance * np.sin(angle)))
    if dy >= 0: ys, yd = slice(0, h - dy), slice(dy, h)
    else:       ys, yd = slice(-dy, h), slice(0, h + dy)
    if dx >= 0: xs, xd = slice(0, w - dx), slice(dx, w)
    else:       xs, xd = slice(-dx, w), slice(0, w + dx)
    src = gray_q[ys, xs]; dst = gray_q[yd, xd]
    valid = mask_bool[ys, xs] & mask_bool[yd, xd]
    if valid.sum() == 0:
        return np.zeros((levels, levels), dtype=np.float64)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (src[valid].astype(np.intp), dst[valid].astype(np.intp)), 1)
    np.add.at(glcm, (dst[valid].astype(np.intp), src[valid].astype(np.intp)), 1)
    total = glcm.sum()
    if total > 0: glcm /= total
    return glcm

def _haralick_invariant(glcm, levels=16):
    """Gray-level invariant Haralick features (Brynolfsson et al. 2019).

    Standard Haralick features depend on the number of gray levels N.
    The invariant versions normalize indices to (0,1] and multiply by
    differentials (1/N), making features asymptotically independent of
    quantization.

    Invariant contrast    = sum(p(i,j) * ((i-j)/N)^2)
    Invariant homogeneity = sum(p(i,j) / (1 + ((i-j)/N)^2))
    Invariant energy      = sum(p(i,j)^2) * N^2
                            (density-based: multiply by N^2 so Riemann sum → integral)
    """
    if glcm.sum() == 0:
        return [0., 0., 0.]
    N = float(levels)
    I, J = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
    diff_norm = (I - J) / N  # normalized difference in (0, 1] scale

    contrast    = float(np.sum(glcm * diff_norm ** 2))
    homogeneity = float(np.sum(glcm / (1.0 + diff_norm ** 2)))
    energy      = float(np.sum(glcm ** 2) * N * N)

    return [contrast, homogeneity, energy]

def quantize_foreground(gray, mask_bool, n_levels=16):
    """Min-max normalize foreground pixels to [0, n_levels-1].

    Maps each specimen's actual intensity range to the full quantization
    range, so two cameras with different dynamic ranges produce equivalent
    GLCM matrices for the same physical texture.
    """
    fg_pixels = gray[mask_bool].astype(np.float64)
    if len(fg_pixels) == 0:
        return (gray // (256 // n_levels)).astype(np.uint8)

    vmin, vmax = fg_pixels.min(), fg_pixels.max()
    if vmax == vmin:
        return np.zeros_like(gray, dtype=np.uint8)

    # normalize to [0, 1] then scale to [0, n_levels-1]
    normalized = (gray.astype(np.float64) - vmin) / (vmax - vmin)
    quantized = (normalized * (n_levels - 1)).clip(0, n_levels - 1).astype(np.uint8)
    return quantized

def extract_cell_features(gray, img_lab, cell_mask, specimen_lab_means=None,
                          glcm_levels=16, gray_quantized=None):
    """Extract texture vector from one warped grid cell.
    Only pixels where cell_mask > 0 are used.

    If specimen_lab_means is provided (3-element array of L*, a*, b* means
    across the specimen's entire foreground), color features are computed as
    deviations from the specimen mean — removing camera/lighting bias while
    preserving spatial colour patterns.

    gray_quantized: pre-quantized grayscale (from quantize_foreground).
    glcm_levels: number of gray levels used in quantization.

    Returns list of 8 features.
    """
    fg = cell_mask > 0
    if fg.sum() < 25:
        return None
    features = []

    # GLCM (d=3, 4 angles) — using foreground-normalized quantization
    ys, xs = np.where(fg)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    if gray_quantized is not None:
        crop_q = gray_quantized[y0:y1, x0:x1]
    else:
        crop_q = quantize_foreground(gray[y0:y1, x0:x1],
                                     cell_mask[y0:y1, x0:x1] > 0, glcm_levels)
    crop_m = cell_mask[y0:y1, x0:x1] > 0

    h_vals = []
    for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        g = _masked_glcm_single(crop_q, crop_m, 3, angle, levels=glcm_levels)
        h_vals.append(_haralick_invariant(g, levels=glcm_levels))
    features.extend(np.mean(h_vals, axis=0))

    # LBP
    if HAS_SKIMAGE:
        lbp = local_binary_pattern(gray, 16, 2, method='uniform')
        fg_lbp = lbp[fg]
        hist, _ = np.histogram(fg_lbp, bins=18, range=(0, 18), density=True)
        hist_nz = hist[hist > 0]
        features.append(float(-np.sum(hist_nz * np.log2(hist_nz))))
        features.append(float(hist[:-1].sum()))
    else:
        features.extend([0., 0.])

    # L*a*b* — specimen-relative or absolute
    for ci in range(3):
        cell_mean = float(img_lab[:, :, ci][fg].mean())
        if specimen_lab_means is not None:
            # deviation from specimen foreground mean
            features.append(cell_mean - specimen_lab_means[ci])
        else:
            features.append(cell_mean)

    return features


def get_cell_feature_names(normalize_color=True):
    """Return feature names reflecting whether color is relative or absolute."""
    if normalize_color:
        return ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy',
                'lbp_entropy', 'lbp_uniformity',
                'dL_rel', 'da_rel', 'db_rel']
    else:
        return ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy',
                'lbp_entropy', 'lbp_uniformity',
                'L_mean', 'a_mean', 'b_mean']


###############################################################################
# 6. PROCESS ONE SPECIMEN
###############################################################################

def process_specimen(fn_gpa, specimen_contour, mean_shape, grid_info,
                     image_dir, mask_lookup, ctrl_step=5, normalize_color=True,
                     normalize_intensity=True, glcm_levels=16):
    """Process one specimen with warped quad grid cells."""

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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)

    # get foreground mask from original COCO — match by original filename
    fg_mask = np.zeros((h, w), np.uint8)
    ml = mask_lookup.get(orig_fn)
    if ml and maskUtils:
        fg_mask = decode_mask(ml['ann'], h, w)
    if fg_mask.sum() < 100:
        logging.warning(f"  No mask for {orig_fn}")
        return None

    fg_bool = fg_mask > 0

    # --- Per-specimen histogram equalization of grayscale within mask ---
    # Standardises intensity distribution across cameras/lighting so GLCM
    # contrast measures spatial texture pattern, not camera dynamic range.
    if normalize_intensity:
        fg_pixels = gray[fg_bool]
        # compute equalization mapping from foreground histogram
        hist, bins = np.histogram(fg_pixels, bins=256, range=(0, 256))
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        n_fg = fg_pixels.shape[0]
        # normalized CDF → lookup table
        lut = np.zeros(256, dtype=np.uint8)
        if n_fg > cdf_min:
            lut = ((cdf - cdf_min) / (n_fg - cdf_min) * 255).clip(0, 255).astype(np.uint8)
        gray = lut[gray]  # apply to full image (background doesn't matter)

    # --- Foreground min-max quantization for GLCM ---
    # Maps each specimen's actual intensity range to [0, N-1] so cameras
    # with different dynamic ranges produce equivalent GLCM matrices.
    gray_quantized = quantize_foreground(gray, fg_bool, glcm_levels)

    # compute specimen-wide foreground L*a*b* means for relative color
    specimen_lab_means = None
    if normalize_color:
        specimen_lab_means = np.array([
            float(img_lab[:, :, ci][fg_bool].mean()) for ci in range(3)
        ])

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
    n_feats = 8  # always 8 features per cell
    cell_features = []
    for r in range(n_rows):
        for c in range(n_cols):
            if not cell_inside[r, c]:
                cell_features.append([np.nan] * n_feats)
                continue
            cm = build_cell_mask(warped_ints, r, c, h, w, fg_mask)
            feats = extract_cell_features(gray, img_lab, cm,
                                          specimen_lab_means=specimen_lab_means,
                                          glcm_levels=glcm_levels,
                                          gray_quantized=gray_quantized)
            if feats is None:
                cell_features.append([np.nan] * n_feats)
            else:
                cell_features.append(feats)

    fv = np.array(cell_features).flatten()
    return (fn_gpa, fv, img, fg_mask, warped_ints)


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
        description='Homology-aware texture phenomics via TPS-warped quad grid')
    p.add_argument('--gpa_dir', required=True,
                   help='V34 GPA output directory (with gpa_mean.csv, '
                        'back_transformed_coco.json)')
    p.add_argument('--json', required=True,
                   help='ORIGINAL COCO JSON (for foreground masks)')
    p.add_argument('--image_dir', required=True)
    p.add_argument('--output_dir', default='./texture_homology_out')
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
                   help='Number of clusters (0 = auto via silhouette score, '
                        'positive integer = force that k)')
    p.add_argument('--viz_specimens', type=int, default=5)
    p.add_argument('--umap_n_neighbors', type=int, default=15)
    p.add_argument('--normalize_color', action='store_true', default=True,
                   help='Use specimen-relative L*a*b* (deviation from specimen '
                        'foreground mean) to remove camera/lighting bias. '
                        'DEFAULT: enabled.')
    p.add_argument('--no_normalize_color', dest='normalize_color',
                   action='store_false',
                   help='Use absolute L*a*b* means (sensitive to camera/lighting).')
    p.add_argument('--normalize_intensity', action='store_true', default=True,
                   help='Per-specimen histogram equalization of grayscale within '
                        'the foreground mask before GLCM/LBP. DEFAULT: enabled.')
    p.add_argument('--no_normalize_intensity', dest='normalize_intensity',
                   action='store_false',
                   help='Skip grayscale histogram equalization.')
    p.add_argument('--glcm_levels', type=int, default=16,
                   help='Number of gray levels for GLCM quantization (default 16). '
                        'Lower = more robust to camera differences, higher = '
                        'finer texture detail. Common values: 8, 16, 32. '
                        'Uses foreground min-max normalization + invariant '
                        'Haralick features (Brynolfsson et al. 2019).')
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
    logging.info(f"Color:     {'specimen-relative (camera-robust)' if args.normalize_color else 'absolute L*a*b*'}")
    logging.info(f"Intensity: {'histogram-equalized (camera-robust)' if args.normalize_intensity else 'raw grayscale'}")
    logging.info(f"GLCM:      {args.glcm_levels} gray levels, invariant Haralick (Brynolfsson 2019)")

    # feature names depend on normalization mode
    feature_names = get_cell_feature_names(args.normalize_color)
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
    normalize_color = args.normalize_color
    normalize_intensity = args.normalize_intensity
    glcm_levels = args.glcm_levels
    n_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    logging.info(f"Processing {len(specimens)} specimens ({n_workers} workers)")

    def proc(item):
        fn, contour = item
        return process_specimen(fn, contour, mean_shape, grid_info,
                                args.image_dir, mask_lookup,
                                ctrl_step=ctrl_step,
                                normalize_color=normalize_color,
                                normalize_intensity=normalize_intensity,
                                glcm_levels=glcm_levels)

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
    for fn, fv, img, mask, wi in results[:args.viz_specimens]:
        viz_warped_grid_on_specimen(fn, img, mask, wi, grid_info, viz_dir, cat_name)
    logging.info(f"Saved {min(len(results), args.viz_specimens)} grid overlays")

    # --- Assemble feature matrix ---
    fns = [r[0] for r in results]
    X_raw = np.array([r[1] for r in results])
    all_imgs = [r[2] for r in results]
    all_masks = [r[3] for r in results]

    ci = grid_info['cell_inside']
    n_cells = ci.sum()
    col_names = []
    for r in range(ci.shape[0]):
        for c in range(ci.shape[1]):
            if ci[r, c]:
                for fn in feature_names:
                    col_names.append(f'r{r}c{c}_{fn}')

    # remove columns for outside cells (all NaN)
    inside_indices = []
    for r in range(ci.shape[0]):
        for c in range(ci.shape[1]):
            base = (r * ci.shape[1] + c) * n_feats_per_cell
            if ci[r, c]:
                inside_indices.extend(range(base, base + n_feats_per_cell))

    X_inside = X_raw[:, inside_indices]
    nan_frac = np.isnan(X_inside).mean()
    logging.info(f"Feature matrix: {X_inside.shape} (NaN fraction: {nan_frac:.3f})")

    # impute remaining NaN
    col_means = np.nanmean(X_inside, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    X_filled = X_inside.copy()
    for j in range(X_filled.shape[1]):
        m = np.isnan(X_filled[:, j])
        X_filled[m, j] = col_means[j]

    # save raw features
    df = pd.DataFrame(X_filled, columns=col_names)
    df.insert(0, 'filename', fns)
    df.to_csv(os.path.join(args.output_dir,
              f"texture_homology_features_{cat_name}.csv"), index=False)

    # --- PCA ---
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_filled)
    n_comp = min(X_scaled.shape[0] - 1, X_scaled.shape[1], 20)
    n_comp = max(n_comp, 2)
    pca = PCA(n_components=n_comp)
    pc = pca.fit_transform(X_scaled)
    logging.info(f"PCA variance: {pca.explained_variance_ratio_[:5].round(3)}")

    # --- PCA PLOT ---
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(f"Texture PCA (homology-aware) — {cat_name}")
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
                f"texture_pca_homology_{cat_name}.png"), dpi=300)
    plt.close()

    # --- UMAP ---
    usc = None
    if HAS_UMAP and len(results) > 5:
        nn = min(args.umap_n_neighbors, len(results) - 1)
        if nn > 1:
            um = umap_lib.UMAP(n_components=2, n_neighbors=nn, random_state=42)
            usc = um.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_title(f"Texture UMAP (homology-aware) — {cat_name}")
            ax.scatter(usc[:, 0], usc[:, 1], alpha=0)
            for i in range(len(usc)):
                t = mk_thumb(all_imgs[i], all_masks[i])
                ax.add_artist(AnnotationBbox(
                    OffsetImage(t, zoom=0.6), (usc[i, 0], usc[i, 1]), frameon=False))
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir,
                        f"texture_umap_homology_{cat_name}.png"), dpi=300)
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
    ax.set_title(f'Silhouette analysis — {cat_name}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f"texture_silhouette_{cat_name}.png"), dpi=200)
    plt.close()

    # dendrogram
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(Z, labels=fns, leaf_rotation=90, leaf_font_size=4, ax=ax)
    ax.set_title(f"Texture dendrogram — {cat_name} "
                 f"(k={use_k}, silhouette={best_sil:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f"texture_dendrogram_{cat_name}.png"), dpi=200)
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
    ax.set_title(f"Texture clusters on PCA — {cat_name} "
                 f"(k={use_k}, sil={best_sil:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f"texture_clusters_{cat_name}.png"), dpi=300)
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
    ax.set_title(f"Texture features (z-scored) — {cat_name} — {n_spec} specimens")
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=5, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f"texture_heatmap_{cat_name}.png"), dpi=200)
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
                  f"texture_analysis_homology_{cat_name}.csv"), index=False)

    # loadings
    pd.DataFrame(
        pca.components_[:min(5, n_comp)].T,
        columns=[f'PC{i+1}' for i in range(min(5, n_comp))],
        index=col_names
    ).to_csv(os.path.join(args.output_dir,
             f"texture_loadings_homology_{cat_name}.csv"))

    # PCA variance explained
    pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'variance_explained': pca.explained_variance_ratio_,
        'cumulative': np.cumsum(pca.explained_variance_ratio_)
    }).to_csv(os.path.join(args.output_dir,
              f"texture_pca_variance_{cat_name}.csv"), index=False)

    # phylo export
    res_df.to_csv(os.path.join(args.output_dir,
                  f"texture_traits_phylo_{cat_name}.csv"), index=False)

    # --- SPATIAL PC LOADING HEATMAP on mean shape ---
    # Shows which anatomical regions drive each PC axis
    for pc_idx in range(min(3, n_comp)):
        pc_label = f'PC{pc_idx + 1}'
        pc_var = pca.explained_variance_ratio_[pc_idx] * 100

        # sum absolute loadings across features within each cell
        cell_importance = np.zeros((grid_info['n_rows'], grid_info['n_cols']))
        cell_counter = 0
        for r in range(grid_info['n_rows']):
            for c in range(grid_info['n_cols']):
                if grid_info['cell_inside'][r, c]:
                    # loadings for this cell's features
                    start = cell_counter * n_feats_per_cell
                    end = start + n_feats_per_cell
                    if end <= len(pca.components_[pc_idx]):
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
        ax.set_title(f"Spatial contribution to {pc_label} ({pc_var:.1f}%) — {cat_name}\n"
                     f"Bright = high contribution to this texture axis")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir,
                    f"texture_spatial_{pc_label}_{cat_name}.png"), dpi=300)
        plt.close()
        logging.info(f"Spatial heatmap saved for {pc_label}")

    logging.info(f"Done: {len(results)} specimens, {n_cells} cells × "
                 f"{n_feats_per_cell} features = "
                 f"{n_cells * n_feats_per_cell} features/specimen")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
