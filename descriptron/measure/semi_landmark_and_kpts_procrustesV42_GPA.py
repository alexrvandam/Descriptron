# PATCH_DORSAL_MARGIN_V1_APPLIED
# V42 (2026-09-25) = V34 + mirror-image handling, found by checking against R's geomorph:
#  - with_reflection had no effect on closed contours: a reflected outline runs the other way round, so it
#    must also be read in reverse order before the cyclic-shift search. Fixed in align_closed.
#  - new --alignment_method reflect_mirrored (the default): the minority of specimens whose outline fits the
#    consensus better reflected (mirror-image photos, opposite body side) are reflected once, before the GPA
#    without reflection; listed in <category>/reflected_specimens.csv. On Diaphorina whole wings the 8 mirror
#    images sat 38% further from the mean than the rest under without_reflection.
import sys
import argparse
import json
import cv2
import numpy as np
import os
import traceback
from pycocotools import mask as maskUtils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import umap
import seaborn as sns
import re
from itertools import combinations
from collections import defaultdict
from math import sqrt
import shutil
import logging
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe
import networkx as nx

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

###############################################################################
# ARGUMENT PARSING
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Semi-landmark GPA for closed contours and open curves (lines). '
                    'Supports iterative mean-shape alignment, multiple anchor '
                    'strategies, TPS sliding semi-landmarks, and bidirectional '
                    'line alignment for full rotation invariance.'
    )
    # Core I/O
    parser.add_argument('--json', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output_dir', default='./outputs')
    parser.add_argument('--image_id', type=int, required=False)
    parser.add_argument('--category_name', default="",
                        help='Category to process (blank = all non-keypoint categories)')

    # Semi-landmarking
    parser.add_argument('--num_landmarks', type=int, default=100)
    parser.add_argument('--auto_landmarks', action='store_true')
    parser.add_argument('--min_auto_landmarks', type=int, default=50)
    parser.add_argument('--auto_scale_factor', type=float, required=False)

    # Keypoints
    parser.add_argument('--keypoints_indices', type=str, required=False,
                        help='Comma/dash list of 1-based keypoint indices to keep')
    parser.add_argument('--require_keypoints_and_segmentation', action='store_true')

    # Anchor / homology (CLOSED contours only)
    parser.add_argument('--anchor_method', choices=['none', 'origin', 'curvature', 'keypoint'],
                        default='none',
                        help='Starting-point strategy for closed contours. '
                             'NONE (default, recommended): fully rotation-invariant exhaustive '
                             'cyclic-shift GPA, then post-GPA origin labelling for '
                             'interpretability. ORIGIN: pre-anchor at min(x+y) — fast but '
                             'REQUIRES consistent photo orientation. CURVATURE: pre-anchor '
                             'at sharpest corner — mostly rotation-invariant. '
                             'KEYPOINT: pre-anchor nearest to a specified keypoint — '
                             'rotation-invariant if keypoints are reliable.')
    parser.add_argument('--anchor_keypoint_index', type=int, default=1,
                        help='1-based keypoint index for keypoint anchor mode')
    parser.add_argument('--anchor_search_window', type=int, default=0,
                        help='Cyclic-shift search window (0=auto). Ignored in none mode.')

    # Line / open-curve handling
    parser.add_argument('--line_categories', type=str, default="",
                        help='Comma-separated category names to treat as lines')
    parser.add_argument('--skeletonize_lines', action='store_true')
    parser.add_argument('--num_line_landmarks', type=int, default=50)

    # TPS sliding semi-landmarks
    parser.add_argument('--slide_semilandmarks', action='store_true',
                        help='Enable TPS bending-energy sliding. Activates when keypoints '
                             'lie on/near the same curve as semi-landmarks. Fixed keypoints '
                             'anchor the curve; semi-landmarks between them slide along '
                             'tangent vectors to minimise bending energy.')
    parser.add_argument('--slide_tolerance', type=float, default=5.0,
                        help='Max pixel distance for a keypoint to be considered "on" the '
                             'curve for sliding purposes (default 5.0 px)')
    parser.add_argument('--slide_iterations', type=int, default=5,
                        help='Number of sliding iterations per GPA iteration (default 5)')
    # V42: sliding semilandmarks exactly as geomorph's gpagen(curves = ...) does them
    parser.add_argument('--slide_method', choices=['none', 'procd', 'bending'], default='none',
                        help='V42: slide every semilandmark along the outline (closed outlines: all points; open '
                             'curves: all but the two end points) as geomorph gpagen does: procd = minimise '
                             'Procrustes distance to the consensus (ProcD = TRUE), bending = minimise thin-plate-'
                             'spline bending energy (ProcD = FALSE, geomorph\'s default). none (default) = fixed '
                             'semilandmarks at equal arc length. (--slide_semilandmarks is the older keypoint-'
                             'anchored option.)')
    parser.add_argument('--slide_max_iter', type=int, default=10, help='V42 sliding: at most this many rounds (geomorph 10)')
    parser.add_argument('--slide_tol', type=float, default=1e-4, help='V42 sliding: convergence tolerance (geomorph 1e-4)')

    # Group labels & MANOVA
    parser.add_argument('--group_labels', required=False)
    parser.add_argument('--perform_manova', action='store_true')

    # UMAP
    parser.add_argument('--umap_n_neighbors', type=int, default=15)
    parser.add_argument('--umap_min_dist', type=float, default=0.1)

    # Alignment
    parser.add_argument('--alignment_method', choices=['reflect_mirrored', 'without_reflection', 'with_reflection'],
                        default='reflect_mirrored',
                        help='reflect_mirrored (V42 default): reflect the minority of mirror-image specimens once, '
                             'then GPA without reflection. without_reflection = V34 default. with_reflection: each '
                             'specimen may be reflected at every alignment.')

    # GPA
    parser.add_argument('--gpa_max_iterations', type=int, default=20)
    parser.add_argument('--gpa_tolerance', type=float, default=1e-6)


    # Dorsal margin extraction
    parser.add_argument('--dorsal_margin_mode', action='store_true',
                        help='Extract only the dorsal margin arc from closed contours '
                             'using keypoint x-coordinate cutoffs.  Requires '
                             '--margin_keypoint_pairs.  Extracted arcs are treated as '
                             'open curves and aligned via gpa_lines().')
    parser.add_argument('--margin_keypoint_pairs', type=str, default="",
                        help='Category-to-keypoint mapping for dorsal margin cutoffs. '
                             'Format: "cat_name:kp_left,kp_right;..." where kp indices '
                             'are 1-based.  Example: "pronotum:1,2;left_elytron:3,4". '
                             'For each category, the two keypoints define x-coordinate '
                             'boundaries; only the upper (dorsal) arc between them is '
                             'kept.')
    parser.add_argument('--outlier_sd_threshold', type=float, default=2.0,
                        help='Procrustes-distance SD threshold for outlier flagging '
                             'in dorsal margin mode (default 2.0).  Shapes whose '
                             'mean distance to other shapes exceeds mean+N*SD are '
                             'flagged and optionally removed.')
    parser.add_argument('--remove_outliers', action='store_true',
                        help='Actually remove flagged outliers before final GPA '
                             '(default: flag only).')
    parser.add_argument('--num_dorsal_landmarks', type=int, default=60,
                        help='Number of equidistant semi-landmarks to resample each '
                             'margin arc to (default 60).')
    parser.add_argument('--margin_arc_select', type=str, default='dorsal',
                        choices=['dorsal', 'ventral', 'anterior', 'posterior',
                                 'shorter', 'longer'],
                        help='Which arc to keep when two keypoints split a closed '
                             'contour: dorsal (top), ventral (bottom), anterior (left), '
                             'posterior (right), shorter, or longer (default: dorsal).')
    parser.add_argument('--pre_align_full_contour', action='store_true',
                        help='In dorsal margin mode: first align all FULL contours '
                             'via standard GPA, then extract margin arcs from the '
                             'aligned shapes. Prevents orientation-dependent arc '
                             'extraction errors (e.g. vertically-oriented specimens '
                             'getting wrong arc). Recommended for x_extrema and '
                             'y_extrema split methods.')
    parser.add_argument('--margin_split_method', type=str, default='keypoints',
                        choices=['keypoints', 'x_extrema', 'y_extrema', 'curvature'],
                        help='How to find the two split points on the contour. '
                             'KEYPOINTS (default): use --margin_keypoint_pairs. '
                             'X_EXTREMA: auto-split at leftmost/rightmost contour '
                             'points (ideal for lateral-view specimens). '
                             'Y_EXTREMA: auto-split at topmost/bottommost. '
                             'CURVATURE: auto-split at the two sharpest corners. '
                             'The auto modes need NO keypoints — just masks.')

    return parser.parse_args()


###############################################################################
# LOADING HELPERS
###############################################################################
def load_annotations(json_path, image_id=None):
    try:
        with open(json_path, 'r') as f:
            coco = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}"); sys.exit(1)
    cats_d = {c['id']: c['name'] for c in coco.get('categories', [])}
    cats_l = coco.get('categories', [])
    imgs = {i['id']: i for i in coco.get('images', [])}
    iids = [image_id] if image_id else list(imgs.keys())
    anns = {}
    for iid in iids:
        im = imgs.get(iid)
        if not im: continue
        fn = im.get('file_name', '')
        if '/' in fn or '\\' in fn: continue
        al = [a for a in coco.get('annotations', []) if a['image_id'] == iid]
        if al: anns[iid] = al
    return anns, imgs, cats_l, cats_d

def load_group_labels(path):
    if not path or not os.path.isfile(path): return {}
    try:
        df = pd.read_csv(path, sep='\t' if path.endswith('.tsv') else ',')
        df.columns = [c.strip().lower() for c in df.columns]
        if 'filename' not in df.columns or 'group_label' not in df.columns: return {}
        return dict(zip(df['filename'].astype(str), df['group_label'].astype(str)))
    except: return {}


###############################################################################
# BASIC UTILITIES
###############################################################################
def clean_fn(s):
    return "".join(c for c in s.replace(' ', '_') if c.isalnum() or c in ('_', '-')).rstrip()

def rm_dup(c, tol=1e-3):
    if len(c) == 0: return c
    u = [c[0]]
    for p in c[1:]:
        if np.linalg.norm(p - u[-1]) > tol: u.append(p)
    return np.array(u)

def mk_mask(ann, h, w):
    if 'segmentation' not in ann: return np.zeros((h, w), np.uint8)
    seg = ann['segmentation']
    try:
        rle = maskUtils.merge(maskUtils.frPyObjects(seg, h, w)) if isinstance(seg, list) else seg
        m = maskUtils.decode(rle)
    except: return np.zeros((h, w), np.uint8)
    b = (m > 0).astype(np.uint8); k = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(b, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)

def fg_mask(contour, shape):
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [contour.astype(np.int32)], 1); return m

def get_kpts(ann):
    kp = ann.get('keypoints', []); pts = []
    if len(kp) % 3 == 0:
        for i in range(0, len(kp), 3):
            x, y, v = kp[i:i+3]
            if v > 0: pts.append((x, y))
    elif len(kp) % 2 == 0:
        pts = [tuple(a) for a in np.array(kp, dtype=np.float32).reshape(-1, 2)]
    return pts

def rel_bend(c):
    if not np.array_equal(c[0], c[-1]): c = np.vstack([c, c[0]])
    d = np.diff(c, axis=0); angles = []
    for i in range(1, len(d)):
        n1, n2 = np.linalg.norm(d[i-1]), np.linalg.norm(d[i])
        angles.append(np.arccos(np.clip(np.dot(d[i-1], d[i])/(n1*n2), -1, 1)) if n1 > 0 and n2 > 0 else 0)
    tl = np.sum(np.linalg.norm(d, axis=1))
    return np.sum(np.array(angles)**2) / tl**2 if tl > 0 else 0


###############################################################################
# RESAMPLING
###############################################################################
def resample_closed(c, n):
    """Resample closed contour to n equidistant points (closed output)."""
    if len(c) < 2: raise ValueError("Need >= 2 pts")
    if not np.array_equal(c[0], c[-1]): c = np.vstack([c, c[0]])
    d = np.sqrt(np.sum(np.diff(c, axis=0)**2, axis=1))
    cd = np.insert(np.cumsum(d), 0, 0)
    fx, fy = interp1d(cd, c[:, 0]), interp1d(cd, c[:, 1])
    nd = np.linspace(0, cd[-1], n)
    r = np.column_stack([fx(nd), fy(nd)])
    if not np.array_equal(r[0], r[-1]): r = np.vstack([r, r[0]])
    return r

def resample_open(c, n):
    """Resample open curve to n equidistant points (open output)."""
    c = np.asarray(c, dtype=np.float64)
    if len(c) < 2: raise ValueError("Need >= 2 pts")
    d = np.sqrt(np.sum(np.diff(c, axis=0)**2, axis=1))
    cd = np.insert(np.cumsum(d), 0, 0)
    if cd[-1] == 0: return np.tile(c[0], (n, 1))
    fx, fy = interp1d(cd, c[:, 0]), interp1d(cd, c[:, 1])
    nd = np.linspace(0, cd[-1], n)
    return np.column_stack([fx(nd), fy(nd)])


###############################################################################
# LINE HELPERS
###############################################################################
def skeletonize(mask):
    m8 = (mask * 255 if mask.max() <= 1 else mask).astype(np.uint8)
    try:
        return cv2.ximgproc.thinning(m8, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)); sk = m8.copy()
        while True:
            er = cv2.erode(sk, el); t = cv2.subtract(sk, cv2.dilate(er, el)); sk = er.copy()
            if cv2.countNonZero(t) == 0: break
        return sk if cv2.countNonZero(sk) > 0 else m8

def ordered_line_pts(skeleton):
    pts = np.argwhere(skeleton > 0)
    if len(pts) == 0: return np.empty((0, 2))
    xy = pts[:, ::-1].astype(np.float64)
    start = int(np.argmin(xy[:, 0] + xy[:, 1]))
    rem = set(range(len(xy))); ordered = []; cur = start
    while rem:
        rem.discard(cur); ordered.append(xy[cur])
        if not rem: break
        idxs = list(rem)
        cur = idxs[int(np.argmin(np.linalg.norm(xy[idxs] - xy[cur], axis=1)))]
    return np.array(ordered)


###############################################################################
# PROCRUSTES CORE
###############################################################################
def procrustes(X, Y):
    """Ordinary Procrustes superimposition (no reflection).
    Returns (distance, aligned_Y, tform_dict)."""
    assert X.shape == Y.shape and X.shape[1] == 2, f"Shape mismatch: {X.shape} vs {Y.shape}"
    mX, mY = X.mean(0), Y.mean(0)
    X0, Y0 = X - mX, Y - mY
    nX, nY = np.sqrt((X0**2).sum()), np.sqrt((Y0**2).sum())
    if nX == 0 or nY == 0:
        return 0., Y0, {'mean_Y': mY, 'normY': nY, 'R': np.eye(2), 'mean_X': mX, 'normX': nX}
    X0 /= nX; Y0 /= nY
    U, s, Vt = np.linalg.svd(Y0.T @ X0)
    R = U @ Vt
    if np.linalg.det(R) < 0: Vt[-1] *= -1; R = U @ Vt
    Z = Y0 @ R
    return float(((X0 - Z)**2).sum()), Z, {'mean_Y': mY, 'normY': nY, 'R': R, 'mean_X': mX, 'normX': nX}

def back_xform(pts, tf):
    """Invert Procrustes: Y = (Z @ R^T) * normY + mean_Y."""
    return (pts @ tf['R'].T) * tf['normY'] + tf['mean_Y']

def reflect(c, axis):
    cen = c.mean(0); r = c.copy()
    if axis == 'horizontal': r[:, 0] = 2*cen[0] - r[:, 0]
    elif axis == 'vertical': r[:, 1] = 2*cen[1] - r[:, 1]
    return r


###############################################################################
# ANCHOR STRATEGIES (closed contours)
###############################################################################
def anchor_origin(c):
    """Index of point closest to top-left (min x+y). NOT rotation-invariant."""
    pts = c[:-1] if np.array_equal(c[0], c[-1]) else c
    return int(np.argmin(pts[:, 0] + pts[:, 1]))

def anchor_curvature(c):
    """Index of point with max discrete curvature. Mostly rotation-invariant."""
    pts = c[:-1] if np.array_equal(c[0], c[-1]) else c
    n = len(pts)
    if n < 5: return 0
    curv = np.zeros(n)
    for i in range(n):
        v1, v2 = pts[(i-1)%n] - pts[i], pts[(i+1)%n] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        curv[i] = (np.pi - np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))) if n1>0 and n2>0 else 0
    return int(np.argmax(curv))

def anchor_keypoint(c, kp_xy):
    """Index of point nearest to a keypoint. Rotation-invariant."""
    pts = c[:-1] if np.array_equal(c[0], c[-1]) else c
    return int(np.argmin(np.linalg.norm(pts - np.array(kp_xy), axis=1)))

def normalise_start(c, idx):
    """Roll closed contour so idx becomes position 0."""
    pts = c[:-1] if np.array_equal(c[0], c[-1]) else c
    r = np.roll(pts, -idx, axis=0)
    return np.vstack([r, r[0]])

def post_gpa_reorder(aligned, mean):
    """After GPA (none mode), apply origin rule to mean and roll all shapes identically."""
    idx = anchor_origin(mean)
    logging.info(f"Post-GPA reorder: rolling all shapes by {idx} (origin rule on mean)")
    return [normalise_start(s, idx) for s in aligned], normalise_start(mean, idx)



###############################################################################
# DORSAL MARGIN EXTRACTION (keypoint x-coordinate cutoff)
###############################################################################
def parse_margin_kp_pairs(spec):
    """Parse 'cat:kpL,kpR;cat2:kpL2,kpR2' into {cat_name: (idx_L, idx_R)} (0-based)."""
    pairs = {}
    if not spec or not spec.strip():
        return pairs
    for part in spec.split(';'):
        part = part.strip()
        if not part or ':' not in part:
            continue
        cat, idxs = part.split(':', 1)
        cat = cat.strip()
        lr = [int(x.strip()) - 1 for x in idxs.split(',') if x.strip()]
        if len(lr) == 2:
            pairs[cat.lower()] = (lr[0], lr[1])
        else:
            logging.warning(f"Ignoring margin spec '{part}' — need exactly 2 indices")
    return pairs


def extract_margin_arc(contour, kp_A_xy, kp_B_xy, arc_select='dorsal'):
    """Extract a specific arc of a closed contour between two keypoints.

    The two keypoints split the closed contour into two arcs.  The
    ``arc_select`` parameter chooses which arc to keep:

      dorsal    — arc with lower mean y  (top of image)
      ventral   — arc with higher mean y (bottom of image)
      anterior  — arc with lower mean x  (left of image)
      posterior  — arc with higher mean x (right of image)
      shorter   — arc with fewer points  (the shorter path between kps)
      longer    — arc with more points   (the longer path between kps)

    Parameters
    ----------
    contour : np.ndarray (M, 2)
        Closed contour — last point may or may not duplicate the first.
    kp_A_xy, kp_B_xy : tuple/array (x, y)
        The two keypoints defining the arc boundaries.
    arc_select : str
        Which arc to keep (see above).

    Returns
    -------
    selected_arc : np.ndarray (K, 2)  — open curve
    idx_A, idx_B : int  — contour indices of the two split points
    """
    pts = contour[:-1] if np.array_equal(contour[0], contour[-1]) else contour
    n = len(pts)

    # find nearest contour vertex to each keypoint
    idx_A = int(np.argmin(np.linalg.norm(pts - np.array(kp_A_xy), axis=1)))
    idx_B = int(np.argmin(np.linalg.norm(pts - np.array(kp_B_xy), axis=1)))

    # walk forward (A → B) and backward (B → ... → A)
    if idx_A <= idx_B:
        arc_fwd = pts[idx_A:idx_B + 1]
        arc_bwd = np.vstack([pts[idx_B:], pts[:idx_A + 1]])
    else:
        arc_fwd = np.vstack([pts[idx_A:], pts[:idx_B + 1]])
        arc_bwd = pts[idx_B:idx_A + 1]

    if arc_fwd.shape[0] < 3 and arc_bwd.shape[0] < 3:
        logging.warning("Both arcs too short — returning forward arc")
        return arc_fwd, idx_A, idx_B

    # --- Arc selection ---
    sel = arc_select.lower().strip()

    if sel == 'dorsal':
        # lower mean y = higher in image = dorsal
        chosen = arc_fwd if arc_fwd[:, 1].mean() <= arc_bwd[:, 1].mean() else arc_bwd
    elif sel == 'ventral':
        # higher mean y = lower in image = ventral
        chosen = arc_fwd if arc_fwd[:, 1].mean() >= arc_bwd[:, 1].mean() else arc_bwd
    elif sel == 'anterior':
        # lower mean x = left of image = anterior
        chosen = arc_fwd if arc_fwd[:, 0].mean() <= arc_bwd[:, 0].mean() else arc_bwd
    elif sel == 'posterior':
        # higher mean x = right of image = posterior
        chosen = arc_fwd if arc_fwd[:, 0].mean() >= arc_bwd[:, 0].mean() else arc_bwd
    elif sel == 'shorter':
        chosen = arc_fwd if arc_fwd.shape[0] <= arc_bwd.shape[0] else arc_bwd
    elif sel == 'longer':
        chosen = arc_fwd if arc_fwd.shape[0] >= arc_bwd.shape[0] else arc_bwd
    else:
        logging.warning(f"Unknown arc_select='{arc_select}', defaulting to 'dorsal'")
        chosen = arc_fwd if arc_fwd[:, 1].mean() <= arc_bwd[:, 1].mean() else arc_bwd

    # orient consistently: lower coordinate first along the primary axis
    if sel in ('anterior', 'posterior'):
        # y-axis ordering (top to bottom)
        if chosen[0, 1] > chosen[-1, 1]:
            chosen = chosen[::-1]
    else:
        # x-axis ordering (left to right)
        if chosen[0, 0] > chosen[-1, 0]:
            chosen = chosen[::-1]

    return chosen, idx_A, idx_B


# backward-compatible alias
def extract_dorsal_margin(contour, kp_left_xy, kp_right_xy):
    """Backward-compatible wrapper — calls extract_margin_arc with arc_select='dorsal'."""
    return extract_margin_arc(contour, kp_left_xy, kp_right_xy, arc_select='dorsal')


def find_contour_split_points(contour, split_method='x_extrema'):
    """Find two split points on a closed contour automatically (no keypoints needed).

    split_method:
      x_extrema  — leftmost & rightmost points (lateral view: anterior/posterior)
      y_extrema  — topmost & bottommost points (dorsal view: left/right)
      curvature  — two points with highest discrete curvature (corners)

    Returns (pt_A_xy, pt_B_xy) as numpy arrays.
    """
    pts = contour[:-1] if np.array_equal(contour[0], contour[-1]) else contour

    if split_method == 'x_extrema':
        idx_A = int(np.argmin(pts[:, 0]))  # leftmost
        idx_B = int(np.argmax(pts[:, 0]))  # rightmost
    elif split_method == 'y_extrema':
        idx_A = int(np.argmin(pts[:, 1]))  # topmost (min y)
        idx_B = int(np.argmax(pts[:, 1]))  # bottommost (max y)
    elif split_method == 'curvature':
        n = len(pts)
        if n < 5:
            idx_A, idx_B = 0, n // 2
        else:
            curv = np.zeros(n)
            for i in range(n):
                v1 = pts[(i - 1) % n] - pts[i]
                v2 = pts[(i + 1) % n] - pts[i]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    curv[i] = np.pi - np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
            # find two highest-curvature points that are far enough apart
            sorted_idx = np.argsort(curv)[::-1]
            idx_A = sorted_idx[0]
            # second peak must be at least 20% of contour length away
            min_sep = max(5, n // 5)
            for ci in sorted_idx[1:]:
                fwd = (ci - idx_A) % n
                bwd = (idx_A - ci) % n
                if min(fwd, bwd) >= min_sep:
                    idx_B = ci
                    break
            else:
                idx_B = sorted_idx[1]
    else:
        logging.warning(f"Unknown split_method='{split_method}', using x_extrema")
        idx_A = int(np.argmin(pts[:, 0]))
        idx_B = int(np.argmax(pts[:, 0]))

    return pts[idx_A], pts[idx_B]


def detect_outliers_procrustes(shapes, sd_threshold=2.0):
    """Flag shapes whose mean pairwise Procrustes distance exceeds mean+sd*threshold.

    Parameters
    ----------
    shapes : list of np.ndarray (all same shape)
    sd_threshold : float

    Returns
    -------
    is_outlier : list of bool
    distances : np.ndarray (N,) — mean distance per shape
    """
    n = len(shapes)
    if n < 3:
        return [False] * n, np.zeros(n)

    # equalise point counts
    mp = min(s.shape[0] for s in shapes)
    eq = [resample_open(s, mp) for s in shapes]

    # pairwise Procrustes distances
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d, _, _ = procrustes(eq[i], eq[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    mean_dists = dist_mat.mean(axis=1)
    mu, sigma = mean_dists.mean(), mean_dists.std()
    threshold = mu + sd_threshold * sigma
    is_outlier = [bool(d > threshold) for d in mean_dists]

    for i, (out, d) in enumerate(zip(is_outlier, mean_dists)):
        if out:
            logging.warning(f"  OUTLIER: shape {i} — mean Procrustes dist {d:.6f} "
                            f"> threshold {threshold:.6f} (mean={mu:.6f}, sd={sigma:.6f})")
    return is_outlier, mean_dists


def viz_dorsal_margins(margins, fns, od, cat_name, outlier_flags=None, arc_label='dorsal'):
    """Overlay plot of all margin arcs with outliers highlighted."""
    plt.figure(figsize=(12, 6))
    for i, (m, fn) in enumerate(zip(margins, fns)):
        is_out = outlier_flags[i] if outlier_flags else False
        c = 'red' if is_out else plt.cm.tab20(i % 20)
        ls = '--' if is_out else '-'
        lw = 0.8 if is_out else 1.2
        lab = f"{fn} [OUTLIER]" if is_out else ""
        plt.plot(m[:, 0], m[:, 1], linestyle=ls, color=c, linewidth=lw,
                 alpha=0.4, label=lab if is_out else "")
    plt.title(f"{arc_label.capitalize()} margin arcs — {cat_name}")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', 'box')
    if outlier_flags and any(outlier_flags):
        plt.legend(fontsize=7)
    plt.savefig(os.path.join(od, f"dorsal_margins_{clean_fn(cat_name)}.png"), dpi=300)
    plt.close()


###############################################################################
# TPS BENDING ENERGY & SLIDING SEMI-LANDMARKS
###############################################################################
def _tps_kernel(r):
    """TPS kernel U(r) = r^2 * ln(r) for 2D; 0 when r=0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(r > 0, r**2 * np.log(r), 0.0)

def tps_bending_energy(source, target):
    """Compute TPS bending energy for mapping source -> target (2D).
    Both are (n,2) arrays of corresponding landmarks."""
    n = source.shape[0]
    # build kernel matrix L
    dists = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
    K = _tps_kernel(dists)
    # build affine part P = [1, x, y]
    P = np.hstack([np.ones((n, 1)), source])
    # build system [K P; P^T 0] * [w; a] = [v; 0]
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T
    # solve for each coordinate
    BE = 0.0
    for dim in range(2):
        rhs = np.zeros(n + 3)
        rhs[:n] = target[:, dim]
        try:
            coeffs = np.linalg.solve(L, rhs)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(L, rhs, rcond=None)[0]
        w = coeffs[:n]
        BE += w @ K @ w
    return BE

def compute_tangents(pts, is_closed=True):
    """Compute unit tangent vectors at each point along a curve."""
    n = len(pts)
    tangents = np.zeros_like(pts)
    for i in range(n):
        if is_closed:
            p = pts if np.array_equal(pts[0], pts[-1]) else pts
            nn = len(p) - 1 if np.array_equal(p[0], p[-1]) else len(p)
            t = p[(i+1) % nn] - p[(i-1) % nn]
        else:
            if i == 0: t = pts[1] - pts[0]
            elif i == n - 1: t = pts[-1] - pts[-2]
            else: t = pts[i+1] - pts[i-1]
        norm = np.linalg.norm(t)
        tangents[i] = t / norm if norm > 0 else np.array([1., 0.])
    return tangents

def find_kpts_on_curve(curve_pts, keypoints, tolerance):
    """Find which keypoints lie within `tolerance` of the curve.
    Returns list of (kp_index, curve_index) pairs."""
    if len(keypoints) == 0: return []
    pts = curve_pts[:-1] if np.array_equal(curve_pts[0], curve_pts[-1]) else curve_pts
    matches = []
    for ki, kp in enumerate(keypoints):
        dists = np.linalg.norm(pts - np.array(kp), axis=1)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= tolerance:
            matches.append((ki, min_idx))
    return matches

def slide_semilandmarks(shape, reference, fixed_indices, is_closed=True, n_iter=5):
    """Slide non-fixed semi-landmarks along tangent vectors to minimise TPS
    bending energy relative to reference.

    shape:         (n,2) current shape (in Procrustes space)
    reference:     (n,2) current mean/reference shape
    fixed_indices: set of indices that must not move (Type I landmarks)
    is_closed:     whether this is a closed contour
    n_iter:        number of relaxation iterations
    """
    s = shape.copy()
    n = s.shape[0]
    # don't slide the closing duplicate if present
    effective_n = n - 1 if is_closed and np.array_equal(s[0], s[-1]) else n
    sliding_indices = [i for i in range(effective_n) if i not in fixed_indices]

    if not sliding_indices:
        return s  # nothing to slide

    for iteration in range(n_iter):
        tangents = compute_tangents(s, is_closed)
        moved = False
        for i in sliding_indices:
            # current displacement from reference
            disp = s[i] - reference[i]
            # project onto tangent
            t = tangents[i]
            proj = np.dot(disp, t) * t
            # slide: move semi-landmark along tangent to reduce displacement
            # use a damped step to avoid oscillation
            new_pos = s[i] - 0.5 * proj
            if np.linalg.norm(new_pos - s[i]) > 1e-10:
                moved = True
            s[i] = new_pos
        # re-close if needed
        if is_closed and effective_n < n:
            s[-1] = s[0]
        if not moved:
            break

    return s


###############################################################################
# CYCLIC-SHIFT ALIGNMENT (closed contours)
###############################################################################
def align_closed(ref, contour, method='without_reflection', window=None):
    """Align closed contour to ref via cyclic shift + Procrustes.
    window=None means full exhaustive sweep (rotation-invariant)."""
    pts = contour[:-1] if np.array_equal(contour[0], contour[-1]) else contour.copy()
    n = len(pts)
    refls = ['none','horizontal','vertical','both'] if method == 'with_reflection' else ['none']
    if window and window > 0:
        shifts = list(set(d % n for d in range(-window//2, window//2+1)))
    else:
        shifts = range(n)
    best_d, best_z, best_tf = np.inf, None, None
    for sh in shifts:
        sc = np.roll(pts, sh, 0); sc = np.vstack([sc, sc[0]])
        # V42: a reflected outline runs the other way round, so read it in reverse before reflecting
        sr = np.roll(pts[::-1], sh, 0); sr = np.vstack([sr, sr[0]])
        for refl in refls:
            tc = sr.copy() if refl in ('horizontal', 'vertical') else sc.copy()
            if refl == 'horizontal': tc = reflect(tc, 'horizontal')
            elif refl == 'vertical': tc = reflect(tc, 'vertical')
            elif refl == 'both': tc = reflect(reflect(tc, 'horizontal'), 'vertical')
            if ref.shape != tc.shape:
                m = min(ref.shape[0], tc.shape[0])
                r2, t2 = resample_closed(ref, m), resample_closed(tc, m)
            else:
                r2, t2 = ref, tc
            d, z, tf = procrustes(r2, t2)
            tf['reflection'] = refl            # V42: which variant won
            if d < best_d: best_d, best_z, best_tf = d, z, tf
    return best_z, best_tf


# ─── V42: sliding semilandmarks, following geomorph's gpagen (pGpa.wSliders, procD.slide, BE.slide) ───
def _gm_center(x):
    return x - x.mean(0)


def _gm_cs_scale(x):
    return x / np.sqrt((_gm_center(x) ** 2).sum())


def _gm_rotate_to(M, Ys):
    """geomorph apply.pPsup: rotate each configuration onto the (unit-size) reference, rotation only."""
    M = _gm_cs_scale(M)
    out = []
    for y in Ys:
        MY = M.T @ y
        u, _, vt = np.linalg.svd(MY)
        u[:, -1] *= np.sign(np.linalg.det(MY)) or 1.0
        out.append(y @ (u @ vt).T)
    return out


def _gm_Ltemplate(ref):
    """Bending-energy matrix of the reference (upper-left p x p block of the inverse TPS L matrix)."""
    p, k = ref.shape
    P = np.linalg.norm(ref[:, None] - ref[None], axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(P > 0, P ** 2 * np.log(P), 0.0)
    L = np.zeros((p + k + 1, p + k + 1))
    L[:p, :p] = P
    L[:p, p:] = np.column_stack([np.ones(p), ref])
    L[p:, :p] = L[:p, p:].T
    try:
        Li = np.linalg.inv(L)
    except np.linalg.LinAlgError:
        Li = -np.linalg.pinv(L)
    return Li[:p, :p]


def _gm_tangents(y, sliders):
    T = np.zeros_like(y)
    t = y[sliders[:, 2]] - y[sliders[:, 0]]
    T[sliders[:, 1]] = t / np.linalg.norm(t, axis=1, keepdims=True)
    return T


def _gm_slide_one(y, ref, T, sliders, method, Lk=None):
    yc = y - ref
    idx = sliders[:, 1]
    p = len(y)
    if method == 'procd':                         # semilandmarks.slide.ProcD
        return y - T * (T * yc).sum(1, keepdims=True)
    m = len(idx)                                  # semilandmarks.slide.BE (exact, approxBE = FALSE)
    U = np.zeros((2 * p, m))
    U[idx, np.arange(m)] = T[idx, 0]
    U[p + idx, np.arange(m)] = T[idx, 1]
    yvec = np.concatenate([yc[:, 0], yc[:, 1]])  # R's as.vector: all x, then all y
    tULk = U.T @ Lk
    res = U @ np.linalg.solve(tULk @ U, tULk @ yvec)
    return y - np.column_stack([res[:p], res[p:]])


def slide_gpa(shapes, sliders, method='bending', max_iter=10, tol=1e-4):
    """V42: GPA with sliding semilandmarks, step for step as geomorph gpagen(curves = sliders, ProcD = ...,
    Proj = FALSE): centre and scale, rotate onto the first specimen, then repeatedly slide every specimen
    against the current consensus and rotate the slid sample back onto it. Returns (coords, consensus)."""
    sliders = np.asarray(sliders, int)
    Ya = [_gm_center(np.asarray(y, float)) for y in shapes]
    Ya = [y / np.sqrt((y ** 2).sum()) for y in Ya]
    Ya = _gm_rotate_to(Ya[0], Ya)
    n = len(Ya)
    ref = sum(Ya) / n
    slid0 = Ya
    Q = ss0 = float((sum(Ya) ** 2).sum() / n)
    it = 1
    while Q > tol:
        it += 1
        Lk = None
        if method == 'bending':
            L = _gm_Ltemplate(ref)
            Lk = np.kron(np.eye(2), L)
        slid = [_gm_slide_one(y, ref, _gm_tangents(y, sliders), sliders, method, Lk) for y in slid0]
        ss = float((sum(slid) ** 2).sum() / n)
        slid0 = _gm_rotate_to(ref, slid)
        ref = _gm_cs_scale(sum(slid0) / n)
        Q = abs(ss0 - ss)
        ss0 = ss
        if it >= max_iter:
            break
    return slid0, ref


def closed_sliders(p):
    """every point of a closed outline slides between its two neighbours (0-based: before, slider, after)"""
    i = np.arange(p)
    return np.column_stack([(i - 1) % p, i, (i + 1) % p])


def open_sliders(p):
    """all points of an open curve except the two end points slide"""
    i = np.arange(1, p - 1)
    return np.column_stack([i - 1, i, i + 1])


def mirror_flags_closed(shapes, max_iter, tol, window):
    """V42: which closed contours are mirror images of the majority. A GPA that may reflect each specimen;
    the minority whose best fit is a reflection are the mirror images."""
    _, _, tfs = gpa_closed(shapes, 'with_reflection', max_iter, tol, window=window)
    flags = np.array([tf.get('reflection') in ('horizontal', 'vertical') for tf in tfs])
    return flags if flags.sum() <= len(flags) / 2 else ~flags


def reflect_closed(s):
    """V42: mirror image of a closed contour, re-read in the same direction round the outline."""
    pts = s[:-1] if np.array_equal(s[0], s[-1]) else s.copy()
    r = pts.copy(); r[:, 0] = -r[:, 0]; r = r[::-1]
    return np.vstack([r, r[0]])


###############################################################################
# BIDIRECTIONAL LINE ALIGNMENT (rotation-invariant for open curves)
###############################################################################
def align_line_bidirectional(ref, line, method='without_reflection'):
    """Try both forward and reversed orientations; keep the one with lower
    Procrustes distance.  This makes line alignment fully rotation-invariant
    — even if two specimens are photographed upside-down relative to each
    other, the correct endpoint correspondence will be found."""
    # forward
    d_fwd, z_fwd, tf_fwd = procrustes(ref, line)
    # reversed
    rev = line[::-1].copy()
    d_rev, z_rev, tf_rev = procrustes(ref, rev)

    if method == 'with_reflection':
        # also try reflections of both orientations
        for refl_axis in ['horizontal', 'vertical']:
            rf = reflect(line.copy(), refl_axis)
            d_rf, z_rf, tf_rf = procrustes(ref, rf)
            if d_rf < d_fwd: d_fwd, z_fwd, tf_fwd = d_rf, z_rf, tf_rf
            rr = reflect(rev.copy(), refl_axis)
            d_rr, z_rr, tf_rr = procrustes(ref, rr)
            if d_rr < d_rev: d_rev, z_rev, tf_rev = d_rr, z_rr, tf_rr

    if d_fwd <= d_rev:
        return d_fwd, z_fwd, tf_fwd, False
    else:
        return d_rev, z_rev, tf_rev, True  # reversed=True


###############################################################################
# GPA — CLOSED CONTOURS (with optional sliding)
###############################################################################
def _norm1(s):
    s = s.copy(); s -= s.mean(0)
    n = np.sqrt((s**2).sum())
    if n > 0: s /= n
    return s, {'mean_Y': s.mean(0), 'normY': n, 'R': np.eye(2), 'mean_X': np.zeros(2), 'normX': 1.}

def gpa_closed(shapes, method='without_reflection', max_iter=20, tol=1e-6,
               window=None, do_slide=False, fixed_indices_per_shape=None,
               slide_iters=5):
    """Iterative GPA for closed contours with optional TPS sliding."""
    ns = len(shapes)
    if ns == 0: return [], np.empty((0, 2)), []
    if ns == 1:
        s, tf = _norm1(shapes[0]); return [s], s.copy(), [tf]

    # bootstrap on shapes[0] (full sweep for robustness)
    ca, ct = [], []
    for i, s in enumerate(shapes):
        if i == 0: _, z, tf = procrustes(shapes[0], s)
        else: z, tf = align_closed(shapes[0], s, method)
        ca.append(z); ct.append(tf)

    mean = np.mean(ca, 0); mean -= mean.mean(0)
    nm = np.sqrt((mean**2).sum())
    if nm > 0: mean /= nm
    delta = np.inf

    for it in range(max_iter):
        na, nt = [], []
        for i, s in enumerate(shapes):
            z, tf = align_closed(mean, s, method, window)
            # optional TPS sliding
            if do_slide and fixed_indices_per_shape:
                fi = fixed_indices_per_shape[i] if i < len(fixed_indices_per_shape) else set()
                z = slide_semilandmarks(z, mean, fi, is_closed=True, n_iter=slide_iters)
            na.append(z); nt.append(tf)

        nm2 = np.mean(na, 0); nm2 -= nm2.mean(0)
        nn = np.sqrt((nm2**2).sum())
        if nn > 0: nm2 /= nn
        delta = ((nm2 - mean)**2).sum()
        logging.info(f"GPA iter {it+1}: delta={delta:.10f}" +
                     (" [with sliding]" if do_slide else ""))
        mean, ca, ct = nm2, na, nt
        if delta < tol:
            logging.info(f"GPA converged at iter {it+1}."); break
    else:
        logging.warning(f"GPA: {max_iter} iters, delta={delta:.10f}")
    return ca, mean, ct


###############################################################################
# GPA — OPEN CURVES (LINES) with bidirectional alignment
###############################################################################
def gpa_lines(shapes, method='without_reflection', max_iter=20, tol=1e-6,
              do_slide=False, fixed_indices_per_shape=None, slide_iters=5):
    """Iterative GPA for open curves using bidirectional alignment.
    Fully rotation-invariant: each specimen is tried in both orientations
    (forward and reversed) and the better fit is kept."""
    ns = len(shapes)
    if ns == 0: return [], np.empty((0, 2)), []
    if ns == 1:
        s, tf = _norm1(shapes[0]); return [s], s.copy(), [tf]

    # bootstrap: align all to shapes[0] bidirectionally
    ca, ct = [], []
    for s in shapes:
        d, z, tf, rev = align_line_bidirectional(shapes[0], s, method)
        ca.append(z); ct.append(tf)
    mean = np.mean(ca, 0); mean -= mean.mean(0)
    nm = np.sqrt((mean**2).sum())
    if nm > 0: mean /= nm

    delta = np.inf
    for it in range(max_iter):
        na, nt = [], []
        for i, s in enumerate(shapes):
            d, z, tf, rev = align_line_bidirectional(mean, s, method)
            if do_slide and fixed_indices_per_shape:
                fi = fixed_indices_per_shape[i] if i < len(fixed_indices_per_shape) else set()
                z = slide_semilandmarks(z, mean, fi, is_closed=False, n_iter=slide_iters)
            na.append(z); nt.append(tf)
        nm2 = np.mean(na, 0); nm2 -= nm2.mean(0)
        nn = np.sqrt((nm2**2).sum())
        if nn > 0: nm2 /= nn
        delta = ((nm2 - mean)**2).sum()
        logging.info(f"Line-GPA iter {it+1}: delta={delta:.10f}" +
                     (" [bidirectional+sliding]" if do_slide else " [bidirectional]"))
        mean, ca, ct = nm2, na, nt
        if delta < tol:
            logging.info(f"Line-GPA converged at iter {it+1}."); break
    else:
        logging.warning(f"Line-GPA: {max_iter} iters, delta={delta:.10f}")
    return ca, mean, ct


###############################################################################
# SAVE MASKS
###############################################################################
def save_masks(contour, image, out_dir, cat, prefix):
    bm = fg_mask(contour, image.shape[:2])
    fg = np.zeros_like(image); fg[bm==1] = image[bm==1]
    d = os.path.join(out_dir, cat); os.makedirs(d, exist_ok=True)
    bc, cc = clean_fn(prefix), clean_fn(cat)
    cv2.imwrite(os.path.join(d, f"{bc}_bin_{cc}.png"), bm*255)
    cv2.imwrite(os.path.join(d, f"{bc}_fg_{cc}.png"), fg)


###############################################################################
# VISUALIZATION
###############################################################################
def label_pts(ax, s, color='purple'):
    n = s.shape[0]; step = max(1, n//5)
    for i in sorted(set([0] + [step*j for j in range(5)] + [n-1])):
        if i < n: ax.text(s[i,0], s[i,1], str(i+1), color=color, fontsize=8)

def viz_aligned(shapes, fns, od, cat, mean=None, is_line=False):
    plt.figure(figsize=(10,10))
    for i, s in enumerate(shapes):
        p = s if is_line or np.array_equal(s[0], s[-1]) else np.vstack([s, s[0]])
        plt.plot(p[:,0], p[:,1], '-o', alpha=.4, ms=2, label=fns[i] if i==0 else "")
        label_pts(plt.gca(), s)
    if mean is not None:
        mp = mean if is_line or np.array_equal(mean[0], mean[-1]) else np.vstack([mean, mean[0]])
        plt.plot(mp[:,0], mp[:,1], 'k-', lw=2.5, label='GPA mean')
    plt.title(f"GPA {'line' if is_line else 'contour'} — {cat}")
    plt.legend(fontsize=7); plt.gca().invert_yaxis(); plt.gca().set_aspect('equal','box')
    plt.savefig(os.path.join(od, f"aligned_{clean_fn(cat)}.png"), dpi=300); plt.close()

def viz_kpts(shapes, fns, od, cat):
    plt.figure(figsize=(10,10))
    for i, kp in enumerate(shapes):
        c = plt.cm.tab20(i%20)
        for j, pt in enumerate(kp): plt.scatter(pt[0], pt[1], color=c, s=50)
    plt.title(f"Aligned KP — {cat}")
    plt.gca().invert_yaxis(); plt.gca().set_aspect('equal','box')
    plt.savefig(os.path.join(od, f"aligned_kp_{clean_fn(cat)}.png"), dpi=300); plt.close()


###############################################################################
# PCA / UMAP / CLUSTERING
###############################################################################
def mk_thumb(img, mask, sz=(80,80)):
    if img.ndim == 2 or img.shape[2] == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if mask is None: mask = np.ones(img.shape[:2], np.uint8)*255
    elif mask.ndim > 2: mask = mask[:,:,0]
    bm = (mask > 0).astype(np.uint8)
    fg = np.zeros_like(img); fg[bm==1] = img[bm==1]
    rgb = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB) if img.ndim==3 and img.shape[2]==3 else fg
    pi = Image.fromarray(rgb).convert("RGBA")
    pi.putalpha(Image.fromarray((bm*255).astype(np.uint8)).convert("L"))
    pi.thumbnail(sz, Image.LANCZOS)
    ni = Image.new('RGBA', sz, (255,255,255,0))
    ni.paste(pi, ((sz[0]-pi.size[0])//2, (sz[1]-pi.size[1])//2))
    return np.array(ni)

def do_pca(shapes, nc=2):
    data = np.array([s.flatten() for s in shapes])
    sc = StandardScaler(); ds = sc.fit_transform(data)
    nc = min(nc, data.shape[0], data.shape[1])  # clamp to min(n_samples, n_features)
    pca = PCA(n_components=nc); pc = pca.fit_transform(ds)
    return pc, pca, ds

def do_umap(X, nn=15, md=0.1):
    m = umap.UMAP(n_components=2, random_state=42, n_neighbors=nn, min_dist=md)
    return m.fit_transform(X), m

def plot_emb(pc, imgs, msks, od, cat, method='PCA'):
    if pc.ndim==2 and pc.shape[1]==1: pc = np.hstack([pc, np.zeros((len(pc),1))])
    fig, ax = plt.subplots(figsize=(12,8)); ax.set_title(f"{method} — {cat}")
    ax.scatter(pc[:,0], pc[:,1], alpha=0)
    for p, im, mk in zip(pc, imgs, msks):
        t = mk_thumb(im, mk)
        if t is not None: ax.add_artist(AnnotationBbox(OffsetImage(t, zoom=.75), (p[0],p[1]), frameon=False))
    plt.savefig(os.path.join(od, f"{method.lower()}_{clean_fn(cat)}.png"), dpi=300); plt.close()

def do_hclust(X, od, cat, pc, imgs, msks, nc=3):
    if pc.ndim==2 and pc.shape[1]==1: pc = np.hstack([pc, np.zeros((len(pc),1))])
    Z = linkage(X, 'ward', 'euclidean'); labels = fcluster(Z, nc, 'maxclust')
    plt.figure(figsize=(16,12))
    for lab, col in zip(np.unique(labels), sns.color_palette(None, len(np.unique(labels)))):
        m = labels==lab; plt.scatter(pc[m,0], pc[m,1], c=[col], label=f'Cl {lab}', alpha=.6)
    plt.title(f'HClust — {cat}'); plt.legend()
    for p, im, mk in zip(pc, imgs, msks):
        t = mk_thumb(im, mk)
        if t is not None: plt.gca().add_artist(AnnotationBbox(OffsetImage(t, zoom=.5), (p[0],p[1]), frameon=False))
    plt.savefig(os.path.join(od, f"hclust_{clean_fn(cat)}.png"), dpi=300); plt.close()
    return labels

def do_manova(pc, gl, od, max_pcs=3):
    try:
        n_pcs = min(max_pcs, pc.shape[1])
        cols = [f'PC{i+1}' for i in range(n_pcs)]
        df = pd.DataFrame(pc[:, :n_pcs], columns=cols)
        df['species'] = gl
        v = df[df['species'].notnull() & (df['species'] != 'Unknown')].copy()
        if v.empty or len(v['species'].unique()) < 2:
            logging.warning("MANOVA needs >= 2 groups with data. Skipping.")
            return
        groups = sorted(v['species'].unique())
        n_groups = len(groups)
        counts = v['species'].value_counts()
        lines = []
        lines.append("=" * 70)
        lines.append("MANOVA — shape variation among species")
        lines.append("=" * 70)
        lines.append(f"PCs used:   {n_pcs}  ({', '.join(cols)})")
        lines.append(f"Species:    {n_groups}")
        lines.append(f"Specimens:  {len(v)}")
        lines.append("")
        lines.append("Sample sizes per species:")
        for sp in groups:
            lines.append(f"  {sp:30s}  n = {counts[sp]}")
        lines.append("")

        # --- omnibus test: do species differ overall? ---
        min_needed = n_groups + n_pcs
        if len(v) <= min_needed:
            lines.append(f"OMNIBUS TEST skipped: {len(v)} specimens but need "
                         f"> {min_needed} (groups + PCs) for valid test.")
        else:
            formula = ' + '.join(cols) + ' ~ C(species)'
            r = MANOVA.from_formula(formula, data=v).mv_test()
            lines.append("-" * 70)
            lines.append("OMNIBUS TEST: Do species differ in shape?")
            lines.append("-" * 70)
            omnibus = r.results['C(species)']['stat']
            for idx in omnibus.index:
                row = omnibus.loc[idx]
                lines.append(f"  {idx:28s}  Value={row['Value']:.4f}  "
                             f"F({row['Num DF']:.0f},{row['Den DF']:.0f})="
                             f"{row['F Value']:.4f}  p={row['Pr > F']:.6f}"
                             f"{'  ***' if row['Pr > F'] < 0.001 else '  **' if row['Pr > F'] < 0.01 else '  *' if row['Pr > F'] < 0.05 else ''}")
            lines.append("")

        # --- pairwise tests between each species pair ---
        pairs = list(combinations(groups, 2))
        n_pairs = len(pairs)
        if n_pairs > 0:
            lines.append("-" * 70)
            lines.append(f"PAIRWISE COMPARISONS ({n_pairs} pairs, Bonferroni-corrected)")
            lines.append("-" * 70)
            pw_results = []
            for sp_a, sp_b in pairs:
                sub = v[v['species'].isin([sp_a, sp_b])].copy()
                na, nb = len(sub[sub['species'] == sp_a]), len(sub[sub['species'] == sp_b])
                if na < 2 or nb < 2 or (na + nb) <= (2 + n_pcs):
                    pw_results.append((sp_a, sp_b, na, nb, None, None, None,
                                       "too few specimens"))
                    continue
                try:
                    formula = ' + '.join(cols) + ' ~ C(species)'
                    pr = MANOVA.from_formula(formula, data=sub).mv_test()
                    st = pr.results['C(species)']['stat']
                    wilks_row = st.loc["Wilks' lambda"]
                    f_val = wilks_row['F Value']
                    p_val = wilks_row['Pr > F']
                    p_adj = min(p_val * n_pairs, 1.0)
                    pw_results.append((sp_a, sp_b, na, nb, f_val, p_val,
                                       p_adj, None))
                except Exception as e:
                    pw_results.append((sp_a, sp_b, na, nb, None, None, None,
                                       str(e)))
            pw_results.sort(key=lambda x: x[6] if x[6] is not None else 999)
            for sp_a, sp_b, na, nb, f_val, p_raw, p_adj, err in pw_results:
                label = f"{sp_a} vs {sp_b}"
                if err:
                    lines.append(f"  {label:45s}  ({err})")
                else:
                    sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 \
                          else "*" if p_adj < 0.05 else "ns"
                    lines.append(f"  {label:45s}  F={f_val:7.3f}  "
                                 f"p={p_raw:.6f}  p_adj={p_adj:.6f}  "
                                 f"(n={na},{nb})  {sig}")
            lines.append("")
            lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, "
                         "ns = not significant")
            lines.append("p_adj = Bonferroni-corrected (p_raw * "
                         f"{n_pairs} pairs)")
        lines.append("=" * 70)
        out = os.path.join(od, 'manova_results.txt')
        with open(out, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        logging.info(f"MANOVA results written to {out}")
    except Exception as e:
        logging.error(f"MANOVA: {e}")
        import traceback; traceback.print_exc()


def _species_palette(groups):
    n = len(groups)
    if n <= 10:
        base = plt.cm.tab10(np.linspace(0, 1, 10))
    elif n <= 20:
        base = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        base = plt.cm.turbo(np.linspace(0.05, 0.95, n))
    return {g: base[i] for i, g in enumerate(sorted(groups))}


def _confidence_ellipse(x, y, ax, n_std=1.96, **kw):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    e = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle, **kw)
    ax.add_patch(e)


def do_cva(pc, gl, od, cat_name, max_pcs=10):
    try:
        v_mask = np.array([(g is not None and g != 'Unknown') for g in gl])
        labels = np.array(gl)[v_mask]
        scores = pc[v_mask]
        groups = sorted(set(labels))
        if len(groups) < 2:
            return
        n_pcs = min(max_pcs, scores.shape[1])
        scores = scores[:, :n_pcs]
        min_per_group = {g: np.sum(labels == g) for g in groups}
        usable = [g for g, c in min_per_group.items() if c >= 2]
        if len(usable) < 2:
            logging.warning("CVA: need >= 2 groups with n>=2. Skipping.")
            return
        mask2 = np.isin(labels, usable)
        labels2, scores2 = labels[mask2], scores[mask2]
        n_components = min(len(usable) - 1, scores2.shape[1], 2)
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        cv = lda.fit_transform(scores2, labels2)
        palette = _species_palette(usable)
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#fafafa')
        ax.set_facecolor('#fafafa')
        for g in usable:
            m = labels2 == g
            c = palette[g]
            if cv.shape[1] >= 2:
                ax.scatter(cv[m, 0], cv[m, 1], color=c, s=60, alpha=0.85,
                           edgecolors='white', linewidths=0.5, zorder=3,
                           label=f"{g} (n={min_per_group[g]})")
                _confidence_ellipse(cv[m, 0], cv[m, 1], ax,
                                    facecolor=c, alpha=0.12, edgecolor=c,
                                    linewidth=1.5, linestyle='--')
                cx, cy = cv[m, 0].mean(), cv[m, 1].mean()
                ax.annotate(g, (cx, cy), fontsize=8, fontweight='bold',
                            ha='center', va='center',
                            path_effects=[pe.withStroke(linewidth=3,
                                                        foreground='white')])
            else:
                ax.scatter(cv[m, 0], np.zeros(m.sum()), color=c, s=60,
                           alpha=0.85, edgecolors='white', linewidths=0.5,
                           zorder=3, label=f"{g} (n={min_per_group[g]})")
        ev = lda.explained_variance_ratio_ * 100
        ax.set_xlabel(f"CV1 ({ev[0]:.1f}%)", fontsize=12)
        if cv.shape[1] >= 2:
            ax.set_ylabel(f"CV2 ({ev[1]:.1f}%)", fontsize=12)
        else:
            ax.set_ylabel("", fontsize=12)
        ax.set_title(f"Canonical Variate Analysis — {cat_name}", fontsize=14,
                     fontweight='bold')
        ax.legend(loc='best', fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='grey', lw=0.5, alpha=0.3)
        ax.axvline(0, color='grey', lw=0.5, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(od, f"cva_{clean_fn(cat_name)}.png"), dpi=300)
        plt.close()
        logging.info(f"CVA plot saved")

        # cross-validated classification accuracy
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        loo = LeaveOneOut()
        y_pred = cross_val_predict(
            LinearDiscriminantAnalysis(), scores2, labels2, cv=loo)
        acc = np.mean(y_pred == labels2) * 100
        conf = pd.crosstab(pd.Series(labels2, name='True'),
                           pd.Series(y_pred, name='Predicted'))
        with open(os.path.join(od, 'cva_classification.txt'), 'w') as f:
            f.write(f"Leave-one-out cross-validated classification accuracy: "
                    f"{acc:.1f}%\n\n")
            f.write("Confusion matrix:\n")
            f.write(conf.to_string())
            f.write("\n")
        logging.info(f"CVA LOO accuracy: {acc:.1f}%")
    except Exception as e:
        logging.error(f"CVA: {e}")
        import traceback; traceback.print_exc()


def do_pairwise_heatmap(pc, gl, od, cat_name, max_pcs=3):
    try:
        n_pcs = min(max_pcs, pc.shape[1])
        cols = [f'PC{i+1}' for i in range(n_pcs)]
        df = pd.DataFrame(pc[:, :n_pcs], columns=cols)
        df['species'] = gl
        v = df[df['species'].notnull() & (df['species'] != 'Unknown')].copy()
        groups = sorted(v['species'].unique())
        counts = v['species'].value_counts()
        testable = [g for g in groups if counts[g] >= 2]
        if len(testable) < 3:
            logging.warning("Pairwise heatmap: need >= 3 testable species. Skipping.")
            return
        pairs = list(combinations(testable, 2))
        n_pairs = len(pairs)
        pmat = pd.DataFrame(np.nan, index=testable, columns=testable)
        fmat = pd.DataFrame(np.nan, index=testable, columns=testable)
        for sp_a, sp_b in pairs:
            sub = v[v['species'].isin([sp_a, sp_b])].copy()
            na = len(sub[sub['species'] == sp_a])
            nb = len(sub[sub['species'] == sp_b])
            if (na + nb) <= (2 + n_pcs):
                continue
            try:
                formula = ' + '.join(cols) + ' ~ C(species)'
                pr = MANOVA.from_formula(formula, data=sub).mv_test()
                st = pr.results['C(species)']['stat']
                p_raw = st.loc["Wilks' lambda"]['Pr > F']
                f_val = st.loc["Wilks' lambda"]['F Value']
                p_adj = min(p_raw * n_pairs, 1.0)
                pmat.loc[sp_a, sp_b] = p_adj
                pmat.loc[sp_b, sp_a] = p_adj
                fmat.loc[sp_a, sp_b] = f_val
                fmat.loc[sp_b, sp_a] = f_val
            except Exception:
                pass
        neg_log_p = -np.log10(pmat.astype(float).clip(lower=1e-20))
        mask = np.triu(np.ones_like(neg_log_p, dtype=bool), k=0)
        fig, ax = plt.subplots(figsize=(max(8, len(testable) * 0.7),
                                        max(7, len(testable) * 0.6)))
        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        sns.heatmap(neg_log_p, mask=mask, cmap=cmap, ax=ax,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': '-log₁₀(p_adj)', 'shrink': 0.7},
                    vmin=0, vmax=max(4, neg_log_p.values[~mask].max() * 1.1)
                    if not np.all(np.isnan(neg_log_p.values[~mask])) else 4,
                    annot=True, fmt='.1f', annot_kws={'size': 7})
        ax.axhline(y=0, color='k', linewidth=2)
        ax.axhline(y=len(testable), color='k', linewidth=2)
        ax.axvline(x=0, color='k', linewidth=2)
        ax.axvline(x=len(testable), color='k', linewidth=2)
        thresh_y = -np.log10(0.05)
        ax.set_title(f"Pairwise shape divergence — {cat_name}\n"
                     f"(-log₁₀ p_adj;  dashed line = 0.05 threshold = {thresh_y:.2f})",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(od, f"pairwise_heatmap_{clean_fn(cat_name)}.png"),
                    dpi=300)
        plt.close()
        logging.info("Pairwise heatmap saved")
    except Exception as e:
        logging.error(f"Pairwise heatmap: {e}")
        import traceback; traceback.print_exc()


def do_mahalanobis_tree(pc, gl, od, cat_name, max_pcs=10):
    try:
        n_pcs = min(max_pcs, pc.shape[1])
        df = pd.DataFrame(pc[:, :n_pcs])
        df['species'] = gl
        v = df[df['species'].notnull() & (df['species'] != 'Unknown')].copy()
        groups = sorted(v['species'].unique())
        counts = v['species'].value_counts()
        usable = [g for g in groups if counts[g] >= 2]
        if len(usable) < 3:
            logging.warning("Mahalanobis tree: need >= 3 groups with n>=2.")
            return
        data_cols = list(range(n_pcs))
        centroids = {}
        for g in usable:
            centroids[g] = v.loc[v['species'] == g, data_cols].values.mean(axis=0)
        pooled_cov = np.zeros((n_pcs, n_pcs))
        total_n = 0
        for g in usable:
            gd = v.loc[v['species'] == g, data_cols].values
            if len(gd) > 1:
                pooled_cov += (len(gd) - 1) * np.cov(gd, rowvar=False)
                total_n += len(gd) - 1
        pooled_cov /= total_n
        try:
            cov_inv = np.linalg.inv(pooled_cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(pooled_cov)
        n = len(usable)
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                diff = centroids[usable[i]] - centroids[usable[j]]
                d = np.sqrt(diff @ cov_inv @ diff)
                dist_mat[i, j] = d
                dist_mat[j, i] = d
        from scipy.spatial.distance import squareform
        condensed = squareform(dist_mat)
        Z = linkage(condensed, method='average')
        fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 6))
        fig.patch.set_facecolor('#fafafa')
        ax.set_facecolor('#fafafa')
        dn = dendrogram(Z, labels=usable, ax=ax, leaf_rotation=45,
                        leaf_font_size=9, color_threshold=0,
                        above_threshold_color='#2c3e50')
        ax.set_ylabel("Mahalanobis distance", fontsize=12)
        ax.set_title(f"Shape phenogram (UPGMA) — {cat_name}", fontsize=14,
                     fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(od, f"mahalanobis_tree_{clean_fn(cat_name)}.png"),
                    dpi=300)
        plt.close()
        logging.info("Mahalanobis tree saved")
        dist_df = pd.DataFrame(dist_mat, index=usable, columns=usable)
        dist_df.to_csv(os.path.join(od, 'mahalanobis_distances.csv'))
    except Exception as e:
        logging.error(f"Mahalanobis tree: {e}")
        import traceback; traceback.print_exc()


def do_shape_network(pc, gl, od, cat_name, max_pcs=3):
    try:
        n_pcs = min(max_pcs, pc.shape[1])
        cols = [f'PC{i+1}' for i in range(n_pcs)]
        df = pd.DataFrame(pc[:, :n_pcs], columns=cols)
        df['species'] = gl
        v = df[df['species'].notnull() & (df['species'] != 'Unknown')].copy()
        groups = sorted(v['species'].unique())
        counts = v['species'].value_counts()
        testable = [g for g in groups if counts[g] >= 2]
        if len(testable) < 3:
            logging.warning("Shape network: need >= 3 testable species.")
            return
        # CVA positions for node layout
        mask2 = v['species'].isin(testable)
        scores2 = v.loc[mask2, cols].values
        labels2 = v.loc[mask2, 'species'].values
        n_comp = min(len(testable) - 1, scores2.shape[1], 2)
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        cv = lda.fit_transform(scores2, labels2)
        centroids_cv = {}
        for g in testable:
            m = labels2 == g
            centroids_cv[g] = cv[m].mean(axis=0)
        # pairwise tests
        pairs = list(combinations(testable, 2))
        n_pairs = len(pairs)
        sig_pairs = []
        nonsig_pairs = []
        for sp_a, sp_b in pairs:
            sub = v[v['species'].isin([sp_a, sp_b])].copy()
            na = len(sub[sub['species'] == sp_a])
            nb = len(sub[sub['species'] == sp_b])
            if (na + nb) <= (2 + n_pcs):
                nonsig_pairs.append((sp_a, sp_b, None))
                continue
            try:
                formula = ' + '.join(cols) + ' ~ C(species)'
                pr = MANOVA.from_formula(formula, data=sub).mv_test()
                st = pr.results['C(species)']['stat']
                p_raw = st.loc["Wilks' lambda"]['Pr > F']
                p_adj = min(p_raw * n_pairs, 1.0)
                if p_adj < 0.05:
                    sig_pairs.append((sp_a, sp_b, p_adj))
                else:
                    nonsig_pairs.append((sp_a, sp_b, p_adj))
            except Exception:
                nonsig_pairs.append((sp_a, sp_b, None))
        G = nx.Graph()
        for g in testable:
            G.add_node(g)
        for sp_a, sp_b, _ in nonsig_pairs:
            G.add_edge(sp_a, sp_b)
        if cv.shape[1] >= 2:
            pos = {g: (centroids_cv[g][0], centroids_cv[g][1]) for g in testable}
        else:
            pos = {g: (centroids_cv[g][0], 0) for g in testable}
        palette = _species_palette(testable)
        node_colors = [palette[g] for g in testable]
        node_sizes = [counts[g] * 200 + 300 for g in testable]
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        # draw non-significant edges (species that CAN'T be told apart)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#44aacc',
                               alpha=0.25, width=1.5, style='solid')
        # draw significant edges as dashed red
        sig_edge_list = [(a, b) for a, b, _ in sig_pairs]
        if sig_edge_list:
            G_sig = nx.Graph()
            G_sig.add_edges_from(sig_edge_list)
            nx.draw_networkx_edges(G_sig, pos, ax=ax, edge_color='#ff6b6b',
                                   alpha=0.5, width=2.0, style=(0, (5, 3)))
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.9,
                               edgecolors='white', linewidths=1.5)
        for g in testable:
            x, y = pos[g]
            ax.annotate(f"{g}\nn={counts[g]}", (x, y), fontsize=8,
                        fontweight='bold', ha='center', va='center',
                        color='white',
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground='#1a1a2e')])
        n_sig = len(sig_pairs)
        n_nonsig = len(nonsig_pairs)
        ax.set_title(f"Shape discrimination network — {cat_name}\n"
                     f"Solid blue = not significantly different  |  "
                     f"Dashed red = significantly different (p<0.05)\n"
                     f"{n_sig} significant / {n_sig + n_nonsig} pairs tested",
                     fontsize=12, fontweight='bold', color='white')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(od, f"shape_network_{clean_fn(cat_name)}.png"),
                    dpi=300, facecolor=fig.get_facecolor())
        plt.close()
        logging.info(f"Shape network saved ({n_sig} significant pairs)")
    except Exception as e:
        logging.error(f"Shape network: {e}")
        import traceback; traceback.print_exc()


###############################################################################
# SIZE ANALYSES
###############################################################################
def do_size_anova(csizes, gl, od, cat_name):
    try:
        from scipy.stats import f_oneway, ttest_ind
        df = pd.DataFrame({'centroid_size': csizes, 'species': gl})
        v = df[df['species'].notnull() & (df['species'] != 'Unknown')].copy()
        groups = sorted(v['species'].unique())
        counts = v['species'].value_counts()
        usable = [g for g in groups if counts[g] >= 2]
        if len(usable) < 2:
            logging.warning("Size ANOVA: need >= 2 groups with n>=2.")
            return
        grp_data = [v.loc[v['species'] == g, 'centroid_size'].values for g in usable]
        f_stat, p_val = f_oneway(*grp_data)
        lines = []
        lines.append("=" * 70)
        lines.append(f"SIZE ANALYSIS — centroid size among species — {cat_name}")
        lines.append("=" * 70)
        lines.append("")
        lines.append("Species means (centroid size):")
        means = {}
        for g in usable:
            vals = v.loc[v['species'] == g, 'centroid_size']
            means[g] = vals.mean()
            lines.append(f"  {g:30s}  mean={vals.mean():.4f}  "
                         f"sd={vals.std():.4f}  n={len(vals)}")
        lines.append("")
        lines.append(f"One-way ANOVA:  F={f_stat:.4f}  p={p_val:.6f}"
                     f"{'  ***' if p_val < 0.001 else '  **' if p_val < 0.01 else '  *' if p_val < 0.05 else '  ns'}")
        lines.append("")
        # pairwise t-tests with Bonferroni
        pairs = list(combinations(usable, 2))
        n_pairs = len(pairs)
        if n_pairs > 0:
            lines.append(f"Pairwise t-tests ({n_pairs} pairs, Bonferroni-corrected):")
            pw = []
            for sp_a, sp_b in pairs:
                da = v.loc[v['species'] == sp_a, 'centroid_size'].values
                db = v.loc[v['species'] == sp_b, 'centroid_size'].values
                t, p = ttest_ind(da, db)
                p_adj = min(p * n_pairs, 1.0)
                pw.append((sp_a, sp_b, t, p, p_adj, len(da), len(db)))
            pw.sort(key=lambda x: x[4])
            for sp_a, sp_b, t, p, p_adj, na, nb in pw:
                sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 \
                      else "*" if p_adj < 0.05 else "ns"
                lines.append(f"  {sp_a} vs {sp_b:30s}  t={t:7.3f}  "
                             f"p_adj={p_adj:.6f}  (n={na},{nb})  {sig}")
        lines.append("=" * 70)
        with open(os.path.join(od, 'size_anova.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        # box plot
        fig, ax = plt.subplots(figsize=(max(8, len(usable) * 0.6), 6))
        fig.patch.set_facecolor('#fafafa')
        ax.set_facecolor('#fafafa')
        palette = _species_palette(usable)
        plot_df = v[v['species'].isin(usable)].copy()
        order = sorted(usable, key=lambda g: means[g], reverse=True)
        bp = sns.boxplot(data=plot_df, x='species', y='centroid_size', order=order,
                         ax=ax, palette=palette, width=0.6, linewidth=1.2,
                         fliersize=3)
        sns.stripplot(data=plot_df, x='species', y='centroid_size', order=order,
                      ax=ax, color='black', alpha=0.4, size=4, jitter=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        sig_str = f"p={p_val:.2e}" if p_val < 0.001 else f"p={p_val:.4f}"
        ax.set_title(f"Centroid size by species — {cat_name}\n"
                     f"ANOVA F={f_stat:.2f}, {sig_str}", fontsize=12,
                     fontweight='bold')
        ax.set_ylabel("Centroid size", fontsize=11)
        ax.set_xlabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(od, f"size_boxplot_{clean_fn(cat_name)}.png"),
                    dpi=300)
        plt.close()
        logging.info(f"Size ANOVA saved (F={f_stat:.2f}, p={p_val:.4e})")
    except Exception as e:
        logging.error(f"Size ANOVA: {e}")
        import traceback; traceback.print_exc()


def do_form_space(pc, csizes, gl, od, cat_name, max_pcs=3):
    """Form space = shape PCs + log(centroid size). Tests shape+size together."""
    try:
        n_pcs = min(max_pcs, pc.shape[1])
        log_cs = np.log(np.array(csizes, dtype=float))
        form = np.column_stack([pc[:, :n_pcs], log_cs])
        form_cols = [f'PC{i+1}' for i in range(n_pcs)] + ['logCS']
        df = pd.DataFrame(form, columns=form_cols)
        df['species'] = gl
        v = df[df['species'].notnull() & (df['species'] != 'Unknown')].copy()
        groups = sorted(v['species'].unique())
        counts = v['species'].value_counts()
        n_groups = len(groups)
        n_vars = n_pcs + 1
        if len(v) <= n_groups + n_vars:
            logging.warning(f"Form space MANOVA: too few specimens. Skipping.")
            return
        # omnibus MANOVA
        formula = ' + '.join(form_cols) + ' ~ C(species)'
        r = MANOVA.from_formula(formula, data=v).mv_test()
        lines = []
        lines.append("=" * 70)
        lines.append(f"FORM SPACE MANOVA — shape + size — {cat_name}")
        lines.append("=" * 70)
        lines.append(f"Variables: {n_pcs} shape PCs + log(centroid size)")
        lines.append(f"Species: {n_groups}  |  Specimens: {len(v)}")
        lines.append("")
        lines.append("OMNIBUS: Do species differ in form (shape + size)?")
        lines.append("-" * 70)
        omnibus = r.results['C(species)']['stat']
        for idx in omnibus.index:
            row = omnibus.loc[idx]
            sig = '***' if row['Pr > F'] < 0.001 else '**' if row['Pr > F'] < 0.01 \
                  else '*' if row['Pr > F'] < 0.05 else ''
            lines.append(f"  {idx:28s}  Value={row['Value']:.4f}  "
                         f"F({row['Num DF']:.0f},{row['Den DF']:.0f})="
                         f"{row['F Value']:.4f}  p={row['Pr > F']:.6f}  {sig}")
        lines.append("=" * 70)
        with open(os.path.join(od, 'form_space_manova.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        # form space CVA
        usable = [g for g, c in counts.items() if c >= 2]
        if len(usable) >= 2:
            mask2 = v['species'].isin(usable)
            scores2 = v.loc[mask2, form_cols].values.astype(float)
            labels2 = v.loc[mask2, 'species'].values
            n_comp = min(len(usable) - 1, scores2.shape[1], 2)
            lda = LinearDiscriminantAnalysis(n_components=n_comp)
            cv = lda.fit_transform(scores2, labels2)
            palette = _species_palette(usable)
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('#fafafa')
            ax.set_facecolor('#fafafa')
            for g in usable:
                m = labels2 == g
                c = palette[g]
                if cv.shape[1] >= 2:
                    ax.scatter(cv[m, 0], cv[m, 1], color=c, s=60, alpha=0.85,
                               edgecolors='white', linewidths=0.5, zorder=3,
                               label=f"{g} (n={counts[g]})")
                    _confidence_ellipse(cv[m, 0], cv[m, 1], ax,
                                        facecolor=c, alpha=0.12, edgecolor=c,
                                        linewidth=1.5, linestyle='--')
                else:
                    ax.scatter(cv[m, 0], np.zeros(m.sum()), color=c, s=60,
                               alpha=0.85, edgecolors='white', linewidths=0.5,
                               label=f"{g} (n={counts[g]})")
            ev = lda.explained_variance_ratio_ * 100
            ax.set_xlabel(f"CV1 ({ev[0]:.1f}%)", fontsize=12)
            if cv.shape[1] >= 2:
                ax.set_ylabel(f"CV2 ({ev[1]:.1f}%)", fontsize=12)
            ax.set_title(f"Form space CVA (shape + size) — {cat_name}",
                         fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=7, framealpha=0.9)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(od, f"form_cva_{clean_fn(cat_name)}.png"),
                        dpi=300)
            plt.close()
            # LOO accuracy
            from sklearn.model_selection import LeaveOneOut, cross_val_predict
            y_pred = cross_val_predict(
                LinearDiscriminantAnalysis(), scores2, labels2, cv=LeaveOneOut())
            acc = np.mean(y_pred == labels2) * 100
            with open(os.path.join(od, 'form_cva_classification.txt'), 'w') as f:
                f.write(f"Form space LOO classification accuracy: {acc:.1f}%\n\n")
                conf = pd.crosstab(pd.Series(labels2, name='True'),
                                   pd.Series(y_pred, name='Predicted'))
                f.write("Confusion matrix:\n")
                f.write(conf.to_string() + "\n")
            logging.info(f"Form space CVA saved (LOO accuracy: {acc:.1f}%)")
    except Exception as e:
        logging.error(f"Form space: {e}")
        import traceback; traceback.print_exc()


def do_allometry(pc, csizes, gl, od, cat_name, max_pcs=3):
    """Regress shape PCs on log(centroid size), analyse residuals (allometry-free)."""
    try:
        n_pcs = min(max_pcs, pc.shape[1])
        log_cs = np.log(np.array(csizes, dtype=float))
        residuals = np.zeros((len(csizes), n_pcs))
        lines = []
        lines.append("=" * 70)
        lines.append(f"ALLOMETRY ANALYSIS — {cat_name}")
        lines.append("=" * 70)
        lines.append("Regression of shape PCs on log(centroid size):")
        lines.append("")
        from scipy.stats import pearsonr, linregress
        for i in range(n_pcs):
            slope, intercept, r, p, se = linregress(log_cs, pc[:, i])
            residuals[:, i] = pc[:, i] - (slope * log_cs + intercept)
            r2 = r ** 2
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            lines.append(f"  PC{i+1}:  R²={r2:.4f}  slope={slope:.4f}  "
                         f"p={p:.6f}  {sig}")
        lines.append("")
        lines.append("R² = proportion of shape variation explained by size")
        lines.append("High R² = strong allometry (shape changes with size)")
        lines.append("")
        # MANOVA on residuals (allometry-free shape)
        cols = [f'resPC{i+1}' for i in range(n_pcs)]
        df = pd.DataFrame(residuals, columns=cols)
        df['species'] = gl
        v = df[df['species'].notnull() & (df['species'] != 'Unknown')].copy()
        groups = sorted(v['species'].unique())
        n_groups = len(groups)
        if len(v) > n_groups + n_pcs and n_groups >= 2:
            formula = ' + '.join(cols) + ' ~ C(species)'
            r = MANOVA.from_formula(formula, data=v).mv_test()
            lines.append("ALLOMETRY-FREE MANOVA (shape residuals ~ species):")
            lines.append("-" * 70)
            omnibus = r.results['C(species)']['stat']
            for idx in omnibus.index:
                row = omnibus.loc[idx]
                sig = '***' if row['Pr > F'] < 0.001 else '**' if row['Pr > F'] < 0.01 \
                      else '*' if row['Pr > F'] < 0.05 else ''
                lines.append(f"  {idx:28s}  Value={row['Value']:.4f}  "
                             f"F({row['Num DF']:.0f},{row['Den DF']:.0f})="
                             f"{row['F Value']:.4f}  p={row['Pr > F']:.6f}  {sig}")
        lines.append("=" * 70)
        with open(os.path.join(od, 'allometry_analysis.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        # allometry scatter — PC1 vs log(centroid size)
        v_mask = np.array([(g is not None and g != 'Unknown') for g in gl])
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#fafafa')
        ax.set_facecolor('#fafafa')
        groups_present = sorted(set(np.array(gl)[v_mask]))
        palette = _species_palette(groups_present)
        for g in groups_present:
            m = np.array(gl) == g
            ax.scatter(log_cs[m], pc[m, 0], color=palette[g], s=40,
                       alpha=0.7, edgecolors='white', linewidths=0.3,
                       label=g)
        slope, intercept, r_val, p_val, _ = linregress(log_cs[v_mask],
                                                        pc[v_mask, 0])
        xs = np.array([log_cs[v_mask].min(), log_cs[v_mask].max()])
        ax.plot(xs, slope * xs + intercept, 'k--', lw=1.5, alpha=0.6)
        ax.set_xlabel("log(centroid size)", fontsize=12)
        ax.set_ylabel("PC1", fontsize=12)
        ax.set_title(f"Allometry — {cat_name}\n"
                     f"PC1 vs size:  R²={r_val**2:.3f}, p={p_val:.4e}",
                     fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=6, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(od, f"allometry_{clean_fn(cat_name)}.png"),
                    dpi=300)
        plt.close()
        logging.info(f"Allometry analysis saved (PC1~size R²={r_val**2:.3f})")
    except Exception as e:
        logging.error(f"Allometry: {e}")
        import traceback; traceback.print_exc()


###############################################################################
# OUTPUT HELPERS
###############################################################################
def build_coco(fns, shapes, cid, cname, kpf=None, kps=None):
    co = {"images":[], "annotations":[], "categories":[{"id":cid, "name":cname}]}
    if kps: co["categories"].append({"id":cid+1, "name":cname+"_keypoints"})
    aid = 1
    for i, fn in enumerate(fns):
        iid = i+1; co["images"].append({"id":iid, "file_name":fn})
        s = shapes[i]; sf = s.flatten().tolist(); xs, ys = s[:,0], s[:,1]
        co["annotations"].append({"id":aid, "image_id":iid, "category_id":cid,
            "segmentation":[sf], "bbox":[float(xs.min()),float(ys.min()),float(np.ptp(xs)),float(np.ptp(ys))],
            "area":float(np.ptp(xs)*np.ptp(ys)), "iscrowd":0}); aid += 1
        if kps and kpf and fn in kpf:
            ki = kpf.index(fn); kp = kps[ki]
            kf = [float(c) for pt in kp for c in (*pt, 2)]; xk, yk = kp[:,0], kp[:,1]
            co["annotations"].append({"id":aid, "image_id":iid, "category_id":cid+1,
                "keypoints":kf, "num_keypoints":int(kp.shape[0]),
                "bbox":[float(xk.min()),float(yk.min()),float(np.ptp(xk)),float(np.ptp(yk))],
                "area":float(np.ptp(xk)*np.ptp(yk)), "iscrowd":0}); aid += 1
    return co

def write_tps(fns, shapes, path):
    with open(path, 'w') as f:
        for i, fn in enumerate(fns):
            s = shapes[i]; f.write(f"LM={s.shape[0]}\n")
            for x, y in s: f.write(f"{x} {y}\n")
            f.write(f"IMAGE={fn}\n\n")


###############################################################################
# MAIN PIPELINE
###############################################################################
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # parse kp indices
    kp_idx = None
    if args.keypoints_indices:
        if '-' in args.keypoints_indices:
            p = args.keypoints_indices.split('-')
            kp_idx = [i-1 for i in range(int(p[0]), int(p[1])+1)]
        else:
            kp_idx = [int(x.strip())-1 for x in args.keypoints_indices.split(',') if x.strip()]

    gl_map = load_group_labels(args.group_labels)
    all_anns, imgs_info, cats_l, cats_d = load_annotations(args.json, args.image_id)
    if not all_anns: print("No annotations."); return

    kp_cids = [c for c, n in cats_d.items() if n.lower() == "keypoints"]

    # detect line categories
    expl = set(n.strip().lower() for n in args.line_categories.split(',') if n.strip()) if args.line_categories else set()
    line_ids = set()
    for c in cats_l:
        if c.get('supercategory','').lower() == 'line': line_ids.add(c['id'])
        if c['name'].lower() in expl: line_ids.add(c['id'])
    for al in all_anns.values():
        for a in al:
            if a.get('is_line'): line_ids.add(a['category_id'])

    # seg categories
    if args.category_name.strip():
        seg_cats = [(c, n) for c, n in cats_d.items() if n.lower() == args.category_name.lower()]
        if not seg_cats: print(f"No match: '{args.category_name}'"); return
    else:
        seg_cats = [(c, n) for c, n in cats_d.items() if n.lower() not in ("keypoints","line_keypoints")]
    if not seg_cats: print("No categories."); return

    # global accumulators
    g_io, g_ao = 0, 0
    g_raw = {"images":[],"annotations":[],"categories":[{"id":1,"name":"pre_alignment"},{"id":2,"name":"pre_alignment_kp"}]}
    g_back = {"images":[],"annotations":[],"categories":[{"id":1,"name":"back_transformed"},{"id":2,"name":"back_transformed_kp"}]}
    g_tm = {}

    # =====================================================================
    for cat_id, cat_name in seg_cats:
        is_line = cat_id in line_ids
        logging.info(f"=== {cat_name} ({'LINE' if is_line else 'CONTOUR'}) ===")

        shapes, kpts_all, fns, imgs, msks, gls, anchor_kps = [],[],[],[],[],[],[]
        od = os.path.join(args.output_dir, clean_fn(cat_name)); os.makedirs(od, exist_ok=True)

        # --- GATHER DATA ---
        for iid, anns in all_anns.items():
            ii = imgs_info.get(iid)
            if not ii: continue
            fn = ii['file_name']; bn = os.path.basename(fn)
            ip = os.path.join(args.image_dir, fn)
            #im = cv2.imread(ip)
            im = cv2.imread(ip, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if im is None: continue
            h, w = im.shape[:2]

            sa = [a for a in anns if a['category_id'] == cat_id]
            if not sa: continue
            best = max(sa, key=lambda a: a.get('area', 0))

            if is_line:
                lp = best.get('line_points')
                if lp and len(lp) >= 2:
                    pts = np.array(lp, dtype=np.float64)
                else:
                    mask = mk_mask(best, h, w)
                    pts = ordered_line_pts(skeletonize(mask))
                    if len(pts) < 2: continue
                pts = rm_dup(pts)
                # NOTE: We do NOT orient here — bidirectional GPA handles it
                resampled = resample_open(pts, args.num_line_landmarks)
            else:
                mask = mk_mask(best, h, w)
                cts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not cts: continue
                cp = max(cts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
                cp = rm_dup(cp)
                if not np.array_equal(cp[0], cp[-1]): cp = np.vstack([cp, cp[0]])
                nlm = args.num_landmarks
                if args.auto_landmarks:
                    if not args.auto_scale_factor: print("Need --auto_scale_factor"); sys.exit(1)
                    nlm = max(args.min_auto_landmarks, int(rel_bend(cp) * args.auto_scale_factor))
                resampled = resample_closed(cp, nlm)

            akp = []
            for a in anns:
                if a['category_id'] in kp_cids: akp.extend(get_kpts(a))
            if args.require_keypoints_and_segmentation and not akp: continue
            if kp_idx and akp: akp = [akp[j] for j in range(len(akp)) if j in kp_idx]

            sl = f"{bn}_{iid}"
            shapes.append(resampled)
            kpts_all.append(np.array(akp) if akp else np.empty((0, 2)))
            fns.append(sl)
            gls.append(gl_map.get(bn, gl_map.get(sl)))
            anchor_kps.append(akp if akp else None)

            if not is_line:
                msks.append(fg_mask(cp, (h, w))); imgs.append(im)
                save_masks(cp, im, od, cat_name, sl)
            else:
                lm = np.zeros((h, w), np.uint8)
                cv2.polylines(lm, [resampled.astype(np.int32)], False, 1, 5)
                msks.append(lm); imgs.append(im)

        if not shapes: print(f"No shapes for '{cat_name}'."); continue

        # --- MARGIN ARC EXTRACTION (if enabled) ---
        dorsal_margin_active = False
        margin_pairs = {}
        if args.dorsal_margin_mode:
            margin_pairs = parse_margin_kp_pairs(args.margin_keypoint_pairs)
            cat_key = cat_name.lower()
            arc_sel = args.margin_arc_select
            split_method = args.margin_split_method

            # Determine if we can process this category
            use_keypoints = (split_method == 'keypoints' and cat_key in margin_pairs)
            use_auto = (split_method in ('x_extrema', 'y_extrema', 'curvature'))

            if use_keypoints:
                kp_L_idx, kp_R_idx = margin_pairs[cat_key]
                logging.info(f"MARGIN ARC mode: category=\'{cat_name}\', "
                             f"arc_select=\'{arc_sel}\', split=keypoints "
                             f"({kp_L_idx+1} & {kp_R_idx+1})")
            elif use_auto:
                logging.info(f"MARGIN ARC mode: category=\'{cat_name}\', "
                             f"arc_select=\'{arc_sel}\', split={split_method} "
                             f"(auto, no keypoints needed)")
            else:
                if split_method == 'keypoints' and cat_key not in margin_pairs:
                    logging.info(f"Margin arc mode ON but no kp pair for \'{cat_name}\' "
                                 f"-- using full contour")

            if use_keypoints or use_auto:
                dorsal_margin_active = True
                
                # --- PRE-ALIGN FULL CONTOURS (if enabled) ---
                # Aligns all complete outlines first so every specimen is in
                # the same Procrustes orientation before arc extraction.
                # This prevents x_extrema from picking wrong split points
                # on vertically-oriented specimens.
                if args.pre_align_full_contour:
                    logging.info("Pre-aligning full contours before arc extraction...")
                    # equalise landmark count
                    n_lm = min(s.shape[0] for s in shapes)
                    eq_shapes = []
                    for s in shapes:
                        if s.shape[0] != n_lm:
                            from scipy.interpolate import interp1d
                            d = np.sqrt(np.sum(np.diff(s, axis=0)**2, axis=1))
                            cd = np.insert(np.cumsum(d), 0, 0)
                            if cd[-1] == 0:
                                eq_shapes.append(s[:n_lm])
                            else:
                                fx = interp1d(cd, s[:, 0])
                                fy = interp1d(cd, s[:, 1])
                                nd = np.linspace(0, cd[-1], n_lm)
                                eq_shapes.append(np.column_stack([fx(nd), fy(nd)]))
                        else:
                            eq_shapes.append(s.copy())
 
                    # run GPA on full contours to get consistent orientation
                    if is_line:
                        pre_aligned, pre_mean, _ = gpa_lines(eq_shapes, args.alignment_method,
                                                              args.gpa_max_iterations, args.gpa_tolerance)
                    else:
                        pre_aligned, pre_mean, _ = gpa_closed(eq_shapes, args.alignment_method,
                                                               args.gpa_max_iterations, args.gpa_tolerance)
                    # replace shapes with pre-aligned versions for arc extraction
                    shapes = pre_aligned
                    logging.info(f"Pre-aligned {len(shapes)} full contours, "
                                 f"now extracting arcs from aligned shapes")
 
# === END EDIT 2 ===

                dorsal_shapes, dorsal_fns, dorsal_imgs, dorsal_msks = [], [], [], []
                dorsal_gls, dorsal_kpts = [], []
                for si in range(len(shapes)):
                    contour = shapes[si]

                    if use_keypoints:
                        # --- keypoint-based split ---
                        kps_this = kpts_all[si]
                        if kps_this.shape[0] == 0:
                            logging.warning(f"  {fns[si]}: no keypoints -- skipping")
                            continue
                        if max(kp_L_idx, kp_R_idx) >= kps_this.shape[0]:
                            logging.warning(f"  {fns[si]}: only {kps_this.shape[0]} kpts, "
                                            f"need {max(kp_L_idx, kp_R_idx)+1} -- skipping")
                            continue
                        kp_A_xy = kps_this[kp_L_idx]
                        kp_B_xy = kps_this[kp_R_idx]
                    else:
                        # --- auto split from contour geometry ---
                        kp_A_xy, kp_B_xy = find_contour_split_points(contour, split_method)

                    dorsal, _, _ = extract_margin_arc(contour, kp_A_xy, kp_B_xy,
                                                     arc_select=arc_sel)
                    if dorsal.shape[0] < 5:
                        logging.warning(f"  {fns[si]}: arc too short ({dorsal.shape[0]} pts) -- skipping")
                        continue
                    dorsal_resampled = resample_open(dorsal, args.num_dorsal_landmarks)
                    dorsal_shapes.append(dorsal_resampled)
                    dorsal_fns.append(fns[si])
                    dorsal_gls.append(gls[si])
                    dorsal_kpts.append(kpts_all[si] if kpts_all[si].shape[0] > 0 else np.empty((0,2)))
                    if si < len(imgs):
                        dorsal_imgs.append(imgs[si])
                        dorsal_msks.append(msks[si])

                if not dorsal_shapes:
                    logging.warning(f"No arcs extracted for \'{cat_name}\' -- full contour fallback")
                    dorsal_margin_active = False
                else:
                    logging.info(f"Extracted {len(dorsal_shapes)} {arc_sel} arcs for \'{cat_name}\'")

                    # --- OUTLIER DETECTION ---
                    outlier_flags, outlier_dists = detect_outliers_procrustes(
                        dorsal_shapes, args.outlier_sd_threshold)
                    n_outliers = sum(outlier_flags)
                    if n_outliers > 0:
                        logging.warning(f"{n_outliers} outlier(s) detected")
                        pd.DataFrame({
                            'filename': dorsal_fns,
                            'mean_procrustes_dist': outlier_dists,
                            'is_outlier': outlier_flags
                        }).to_csv(os.path.join(od, f"dorsal_outliers_{clean_fn(cat_name)}.csv"),
                                  index=False)

                    viz_dorsal_margins(dorsal_shapes, dorsal_fns, od, cat_name, outlier_flags,
                                       arc_label=arc_sel)

                    if args.remove_outliers and n_outliers > 0:
                        keep = [i for i, o in enumerate(outlier_flags) if not o]
                        dorsal_shapes = [dorsal_shapes[i] for i in keep]
                        dorsal_fns    = [dorsal_fns[i]    for i in keep]
                        dorsal_gls    = [dorsal_gls[i]    for i in keep]
                        dorsal_kpts   = [dorsal_kpts[i]   for i in keep]
                        if dorsal_imgs:
                            dorsal_imgs = [dorsal_imgs[i] for i in keep]
                            dorsal_msks = [dorsal_msks[i] for i in keep]
                        logging.info(f"Removed {n_outliers} outlier(s), {len(dorsal_shapes)} remain")

                    # REPLACE working lists so downstream GPA uses margin arcs
                    shapes   = dorsal_shapes
                    fns      = dorsal_fns
                    gls      = dorsal_gls
                    kpts_all = [np.array(k) if len(k) > 0 else np.empty((0,2)) for k in dorsal_kpts]
                    if dorsal_imgs:
                        imgs = dorsal_imgs
                        msks = dorsal_msks
                    is_line = True
                    logging.info(f"Margin arcs -> OPEN CURVES via gpa_lines()")
                    if not shapes:
                        logging.warning(f"All removed as outliers -- skipping")
                        continue



        # --- EQUALISE POINT COUNTS ---
        mp = min(s.shape[0] for s in shapes)
        eq = [resample_open(s, mp) if is_line else resample_closed(s, mp) for s in shapes]

        # --- DETECT KEYPOINTS ON CURVE (for sliding) ---
        fixed_per_shape = [set() for _ in eq]
        do_slide = False
        if args.slide_semilandmarks:
            for i in range(len(eq)):
                if kpts_all[i].shape[0] > 0:
                    matches = find_kpts_on_curve(eq[i], kpts_all[i], args.slide_tolerance)
                    if matches:
                        fi = set(ci for _, ci in matches)
                        fixed_per_shape[i] = fi
                        logging.info(f"  {fns[i]}: {len(fi)} keypoints on curve -> fixed indices {fi}")
            # enable sliding if at least one shape has fixed landmarks on the curve
            do_slide = any(len(fi) > 0 for fi in fixed_per_shape)
            if do_slide:
                logging.info(f"TPS sliding ENABLED for '{cat_name}' — "
                             f"{sum(1 for fi in fixed_per_shape if fi)} shapes have on-curve keypoints")
            else:
                logging.info(f"TPS sliding requested but no keypoints found on curve "
                             f"(tolerance={args.slide_tolerance}px). Proceeding without sliding.")

        # --- ANCHOR NORMALISE (closed only, skip for 'none' mode) ---
        if not is_line and args.anchor_method != 'none':
            logging.info(f"Anchor: {args.anchor_method}")
            if args.anchor_method == 'origin':
                logging.warning("NOTE: 'origin' anchor is NOT rotation-invariant. "
                                "Use --anchor_method=none if specimens are not "
                                "consistently oriented in images.")
            for i in range(len(eq)):
                if args.anchor_method == 'origin':
                    ai = anchor_origin(eq[i])
                elif args.anchor_method == 'curvature':
                    ai = anchor_curvature(eq[i])
                elif args.anchor_method == 'keypoint':
                    kp = anchor_kps[i]
                    if kp and len(kp) >= args.anchor_keypoint_index:
                        ai = anchor_keypoint(eq[i], kp[args.anchor_keypoint_index - 1])
                    else:
                        ai = anchor_origin(eq[i])
                else:
                    ai = 0
                eq[i] = normalise_start(eq[i], ai)
        elif not is_line:
            logging.info("Anchor: none — full cyclic sweep (rotation-invariant)")

        # --- GPA ---
        if args.anchor_method == 'none' and not is_line:
            sw = None  # full exhaustive sweep
        else:
            sw = args.anchor_search_window if args.anchor_search_window > 0 else max(20, mp // 5)

        # V42: reflect the minority of mirror-image outlines once, then align without reflection
        method_eff = args.alignment_method
        mflags = np.zeros(len(eq), bool)          # V42: which specimens were reflected (outputs un-reflect them)
        if args.alignment_method == 'reflect_mirrored':
            method_eff = 'without_reflection'
            if not is_line and len(eq) >= 3:
                mflags = mirror_flags_closed(eq, args.gpa_max_iterations, args.gpa_tolerance, sw)
                pd.DataFrame({'file': fns, 'reflected': mflags.astype(int)}).to_csv(
                    os.path.join(od, "reflected_specimens.csv"), index=False)
                if mflags.any():
                    logging.warning(f"{int(mflags.sum())} of {len(eq)} '{cat_name}' outlines are mirror images and "
                                    f"were reflected before GPA: {', '.join(str(f) for f, m in zip(fns, mflags) if m)}")
                    eq = [reflect_closed(e) if m else e for e, m in zip(eq, mflags)]
            elif is_line:
                method_eff = 'with_reflection'      # open curves: reflection handled per specimen (V34 behaviour)

        if is_line:
            aligned, mean, tfs = gpa_lines(
                eq, method_eff, args.gpa_max_iterations, args.gpa_tolerance,
                do_slide=do_slide, fixed_indices_per_shape=fixed_per_shape,
                slide_iters=args.slide_iterations)
        else:
            aligned, mean, tfs = gpa_closed(
                eq, method_eff, args.gpa_max_iterations, args.gpa_tolerance,
                window=sw, do_slide=do_slide, fixed_indices_per_shape=fixed_per_shape,
                slide_iters=args.slide_iterations)

        # --- V42: sliding semilandmarks (geomorph method), on the correspondence the GPA found ---
        if args.slide_method != 'none' and len(aligned) >= 3:
            closed_dup = (not is_line) and np.allclose(aligned[0][0], aligned[0][-1])
            A0 = [a[:-1] if closed_dup else a for a in aligned]
            sl = open_sliders(len(A0[0])) if is_line else closed_sliders(len(A0[0]))
            slid, _ = slide_gpa(A0, sl, args.slide_method, args.slide_max_iter, args.slide_tol)
            logging.info(f"Semilandmarks slid ({args.slide_method}, geomorph method): {len(slid)} specimens, "
                         f"{len(sl)} sliders")
            new_al, new_tf = [], []
            for i, a in enumerate(slid):
                full = np.vstack([a, a[:1]]) if closed_dup else a
                # the specimen's raw points in the correspondence the GPA found (not eq[i]'s own start point)
                raw_corr = back_xform(aligned[i], tfs[i])
                # a fresh similarity fit maps the slid points back onto this specimen's image
                an = _gm_center(full); an = an / np.sqrt((an ** 2).sum())
                _, _, tf_i = procrustes(an, raw_corr)
                new_al.append(full); new_tf.append(tf_i)
            aligned, tfs = new_al, new_tf
            mean = np.mean(aligned, 0)

        # --- POST-GPA REORDER (none mode only) ---
        if args.anchor_method == 'none' and not is_line:
            aligned, mean = post_gpa_reorder(aligned, mean)

        viz_aligned(aligned, fns, od, cat_name, mean, is_line)
        pd.DataFrame(mean, columns=['x','y']).to_csv(os.path.join(od, "gpa_mean.csv"), index=False)

        # --- KEYPOINTS (separate Procrustes, non-cyclic) ---
        iwk = [i for i, a in enumerate(kpts_all) if a.shape[0] > 0]
        akps, kpf, rkps, ktfs = [], [], [], []
        if iwk:
            mk = min(kpts_all[i].shape[0] for i in iwk)
            ekp = [kpts_all[i][:mk] for i in iwk]
            kpf = [fns[i] for i in iwk]; rkps = [k.copy() for k in ekp]
            for arr in ekp:
                _, z, tf = procrustes(ekp[0], arr)
                akps.append(z); ktfs.append(tf)
            viz_kpts(akps, kpf, od, cat_name)
            rows = [[fn, f'KP{j+1}', pt[0], pt[1]] for fn, kp in zip(kpf, akps) for j, pt in enumerate(kp)]
            pd.DataFrame(rows, columns=['filename','keypoint','x','y']).to_csv(
                os.path.join(od, f"aligned_kp_{clean_fn(cat_name)}.csv"), index=False)

        # --- ALIGNED COCO ---
        ac = build_coco(fns, aligned, 1, "aligned", kpf if akps else None, akps if akps else None)
        with open(os.path.join(od, "aligned_coco.json"), "w") as f: json.dump(ac, f, indent=2)

        # --- DOWNSTREAM ANALYSES ---
        ns = len(aligned)
        csizes = [np.sqrt(np.sum((s - s.mean(axis=0))**2)) for s in eq]
        pc, pca_obj, ds = do_pca(aligned, max(1, ns-1))
        pc2 = np.hstack([pc[:,:2], np.zeros((len(pc), max(0, 2-pc.shape[1])))]) if pc.shape[1] < 2 else pc[:,:2]
        plot_emb(pc2, imgs, msks, od, cat_name, 'PCA')
        ur = None; nnb = min(args.umap_n_neighbors, ns-1)
        if nnb > 1 and ns >= 4:
            try:
                ur, _ = do_umap(ds, nnb, args.umap_min_dist)
                plot_emb(ur, imgs, msks, od, cat_name, 'UMAP')
            except (TypeError, ValueError) as e:
                logging.warning(f"UMAP failed for {cat_name} (n={ns}): {e}")
                ur = None
        dbl = DBSCAN(eps=.5, min_samples=5).fit_predict(ds)
        hl = do_hclust(ds, od, cat_name, pc2, imgs, msks)
        dd = {'filename':fns, 'PCA1':pc[:,0], 'PCA2':pc2[:,1], 'DBSCAN':dbl, 'Hierarchical':hl}
        if ur is not None: dd['UMAP1']=ur[:,0]; dd['UMAP2']=ur[:,1]
        else: dd['UMAP1']=np.nan; dd['UMAP2']=np.nan
        pd.DataFrame(dd).to_csv(os.path.join(od, f"{clean_fn(cat_name)}_analysis.csv"), index=False)
        # --- PHYLO TRAITS EXPORT ---
        # Save PC scores in format compatible with texture_phylo_mapping.R
        # Includes PC1-PC10, centroid size, and PCA variance explained
        try:
            n_pcs = min(10, pc.shape[1])
            phylo_dd = {'filename': fns}
            for pi in range(n_pcs):
                phylo_dd[f'PC{pi+1}'] = pc[:, pi]
            phylo_dd['centroid_size'] = csizes
            phylo_df = pd.DataFrame(phylo_dd)
            phylo_path = os.path.join(od, f"shape_traits_phylo_{clean_fn(cat_name)}.csv")
            phylo_df.to_csv(phylo_path, index=False)
            logging.info(f"Phylo traits saved: {phylo_path} ({n_pcs} PCs + centroid_size)")
            # also save variance explained
            var_dd = {
                'component': [f'PC{i+1}' for i in range(len(pca_obj.explained_variance_ratio_))],
                'variance_explained': pca_obj.explained_variance_ratio_,
                'cumulative': np.cumsum(pca_obj.explained_variance_ratio_)
            }
            pd.DataFrame(var_dd).to_csv(
                os.path.join(od, f"shape_pca_variance_{clean_fn(cat_name)}.csv"), index=False)
        except Exception as e:
            logging.warning(f"Phylo traits export: {e}")

        # --- SPATIAL PC LOADING HEATMAP ON MEAN SHAPE ---
        # Color each semi-landmark by its contribution to each PC axis
        try:
            n_landmarks = mean.shape[0]
            for pc_idx in range(min(3, pca_obj.n_components_)):
                pc_label = f'PC{pc_idx + 1}'
                pc_var = pca_obj.explained_variance_ratio_[pc_idx] * 100
                loadings = pca_obj.components_[pc_idx]  # length = n_landmarks * 2

                # sum squared loadings of x,y for each landmark
                lm_importance = np.zeros(n_landmarks)
                for li in range(n_landmarks):
                    ix, iy = 2 * li, 2 * li + 1
                    if iy < len(loadings):
                        lm_importance[li] = loadings[ix]**2 + loadings[iy]**2

                fig, ax = plt.subplots(figsize=(10, 10))

                # draw mean contour as closed line
                ms_plot = mean
                if n_landmarks > 1 and not np.array_equal(mean[0], mean[-1]):
                    ms_plot = np.vstack([mean, mean[0]])
                ax.plot(ms_plot[:, 0], ms_plot[:, 1], 'k-', lw=1.5, alpha=0.4)

                # color landmarks by contribution
                sc = ax.scatter(mean[:, 0], mean[:, 1],
                                c=lm_importance, cmap='hot', s=120,
                                edgecolors='black', linewidths=0.5,
                                zorder=5)
                plt.colorbar(sc, ax=ax, label='Sum of squared loadings (x² + y²)',
                             shrink=0.6)

                # annotate top 5 contributing landmarks
                top5 = np.argsort(lm_importance)[-5:]
                for ti in top5:
                    ax.annotate(f'LM{ti}', (mean[ti, 0], mean[ti, 1]),
                                fontsize=7, ha='center', va='bottom',
                                xytext=(0, 8), textcoords='offset points',
                                color='red', fontweight='bold')

                ax.set_aspect('equal', 'box')
                ax.invert_yaxis()
                ax.set_title(f"Shape {pc_label} ({pc_var:.1f}%) — {cat_name}\n"
                             f"Landmark contribution to shape variation\n"
                             f"(bright = high, dark = low)")
                plt.tight_layout()
                plt.savefig(os.path.join(od, f"shape_spatial_{pc_label}_{clean_fn(cat_name)}.png"),
                            dpi=300)
                plt.close()
                logging.info(f"Shape spatial heatmap: {pc_label}")
        except Exception as e:
            logging.warning(f"Shape spatial heatmap: {e}")
            import traceback; traceback.print_exc()
        if args.perform_manova:
            do_manova(pc, gls, od)
            do_cva(pc, gls, od, cat_name)
            do_pairwise_heatmap(pc, gls, od, cat_name)
            do_mahalanobis_tree(pc, gls, od, cat_name)
            do_shape_network(pc, gls, od, cat_name)
            do_size_anova(csizes, gls, od, cat_name)
            do_form_space(pc, csizes, gls, od, cat_name)
            do_allometry(pc, csizes, gls, od, cat_name)

        # --- PRE-ALIGNMENT OUTPUT ---
        try:
            # V42: reflected specimens go back to their own image coordinates (x -> -x undoes the reflection;
            # the landmark order is kept, so landmark i still corresponds across specimens)
            eq_img = [e * np.array([-1.0, 1.0]) if mflags[i] else e for i, e in enumerate(eq)]
            rc = build_coco(fns, eq_img, 1, "pre_alignment", kpf if rkps else None, rkps if rkps else None)
            with open(os.path.join(od, "pre_alignment_coco.json"), "w") as f: json.dump(rc, f)
            write_tps(fns, eq_img, os.path.join(od, "pre_alignment.TPS"))
        except Exception as e: logging.error(f"Pre-align output: {e}")

        # --- BACK-TRANSFORM ---
        try:
            bs = [back_xform(aligned[i], tfs[i]) for i in range(len(aligned))]
            # V42: on the specimen's own image (colour homology and homology frames look pixels up here)
            bs = [b * np.array([-1.0, 1.0]) if mflags[i] else b for i, b in enumerate(bs)]
            bkp = [back_xform(akps[j], ktfs[j]) for j in range(len(akps))] if akps else []
            bc = build_coco(fns, bs, 1, "back_transformed", kpf if bkp else None, bkp if bkp else None)
            with open(os.path.join(od, "back_transformed_coco.json"), "w") as f: json.dump(bc, f)
            write_tps(fns, bs, os.path.join(od, "back_transformed.TPS"))
            tm = {"semilandmarks":{}, "keypoints":{}}
            for i, fn in enumerate(fns):
                t = tfs[i]; tm["semilandmarks"][fn] = {"R":t['R'].tolist(), "mean_Y":t['mean_Y'].tolist(), "normY":float(t['normY']),
                                                     "reflected_x_before_transform": bool(mflags[i])}   # V42
            for j, fn in enumerate(kpf):
                t = ktfs[j]; tm["keypoints"][fn] = {"R":t['R'].tolist(), "mean_Y":t['mean_Y'].tolist(), "normY":float(t['normY'])}
            with open(os.path.join(od, "transforms.json"), "w") as f: json.dump(tm, f, indent=2)
        except Exception as e: logging.error(f"Back-xform: {e}"); traceback.print_exc()

        # --- GLOBAL ACCUMULATE (unique IDs) ---
        try:
            for x in rc["images"]: x["id"] += g_io
            for x in rc["annotations"]: x["image_id"] += g_io; x["id"] += g_ao
            g_raw["images"].extend(rc["images"]); g_raw["annotations"].extend(rc["annotations"])
            for x in bc["images"]: x["id"] += g_io
            for x in bc["annotations"]: x["image_id"] += g_io; x["id"] += g_ao
            g_back["images"].extend(bc["images"]); g_back["annotations"].extend(bc["annotations"])
            for dt in tm:
                for fn, v in tm[dt].items(): g_tm[f"{cat_name}/{dt}/{fn}"] = v
            ni = len(rc["images"])
            ma = max((a["id"] for a in rc["annotations"]), default=0)
            mb = max((a["id"] for a in bc["annotations"]), default=0)
            g_io += ni; g_ao += max(ma, mb)
        except Exception as e: logging.error(f"Accumulate: {e}")
        logging.info(f"Done: {cat_name}")

    # global outputs
    try:
        for nm, d in [("concat_pre_alignment_coco.json", g_raw), ("concat_back_transformed_coco.json", g_back)]:
            with open(os.path.join(args.output_dir, nm), "w") as f: json.dump(d, f)
        with open(os.path.join(args.output_dir, "concat_transforms.json"), "w") as f: json.dump(g_tm, f, indent=2)
    except Exception as e: logging.error(f"Global output: {e}")
    logging.info("All categories complete.")


if __name__ == "__main__":
    try: main()
    except Exception as e: logging.error(f"Fatal: {e}"); traceback.print_exc(); sys.exit(1)
