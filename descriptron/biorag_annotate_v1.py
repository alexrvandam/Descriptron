"""BioRAG Annotate (v1) - DINOv3 patch tokens of a folder of images in PCA space <-> the images.

Opened from Descriptron v89 (tab "6  BioRAG Annotate"). The window
  1. TOKENS: extracts DINOv3 ViT-L/16 patch tokens (layer 12/18/23) for every image of the folder
     session (or the open image) in a background SUBPROCESS (biorag_annotate_tokens_v1.py, run with the
     python of an env whose transformers knows DINOv3; BIORAG_ANNOTATE_PY overrides it) and caches them
     per image (npz in <image folder>/biorag_annotate_cache, configurable). The long side is resized
     (cubic, small images UPSAMPLED) to 1024/1536/2048, sides multiples of 16.
     ZOOMED REGION: "Re-extract zoomed region" extracts tokens of the visible crop only, upsampled so the
     crop's long side is the region size (at most 8x its own pixels), cached per crop; the "Token source"
     switch chooses whole-image or zoomed-region tokens for the plot, similarity and masks (fine
     structures such as setae are only a few px wide and vanish inside one patch at whole-image scale).
  2. PLOT: all tokens of the active source in one PCA scatter (PC1-2 / 1-3 / 2-3; raw / high-pass /
     both), coloured PCA->RGB or by the category of the mask covering each patch (folder COCO or a
     picked COCO / rlejson, e.g. DINOMask predictions); legend entries select their points.
  3. STRIP: one image at a time (Prev / Next, slider, arrow keys); wheel = zoom about the cursor,
     middle-drag or space+drag = pan, 0 = reset; the photo is re-rendered from the full-resolution file
     when zoomed. View (button or 'v'): photo -> annotations (category masks + outlines) -> DINOv3
     PCA-RGB -> trained DINOMask (argmax classes with legend, or PCA-RGB of the head's 512-d hidden
     activations); a blend slider sets the view layer's opacity; the selection halo is always on top.
  4. SELECT: lasso on the plot -> the patches light up on every image (PCA colours + white halo); click /
     lasso on the image -> their points light up on the plot; right-click the image -> Select this patch
     / Select similar (this image / all images) = cosine similarity of the token vectors (same variant
     as the plot) to the clicked patch, or to the mean of the selection if the click is inside it,
     threshold slider updates live. Brush / eraser on the strip edit a per-image PIXEL selection (start:
     selected patches upsampled + guided filter); a patch counts as selected on the plot if >= 50% of its
     pixels are; Escape cancels a stroke, Ctrl+Z undoes the last stroke. "Not this" (19 Oct) paints a
     per-image NEGATIVE pixel mask with the same brush: it removes those pixels from the selection
     (negative wins; a later Brush stroke removes not-this paint again), selections / masks never include
     it, and the learned rule trains on those patches as explicit (up-weighted) negatives.
  5. ASSIGN: right-click -> category (the editor's category menu). Selected patches -> mask (patch grid
     -> guided filter on the grey working image -> original size, threshold 0.5, tiny components
     removed); brush-edited images use their pixel mask as is. Crop masks map back into the image. The
     mask of the image open in the editor is added to the editor session (the Lasso -> New mask path, so
     it is editable and saved by the marmot); the masks of ALL images go to biorag_annotate_<timestamp>
     .json (polygons) + .rlejson (RLE) in the cache folder. "Merge into folder annotations..." writes a
     NEW merged COCO next to the folder COCO; the folder COCO (and any picked mask file) is never modified.

  6. APPLY A REGION SELECTION (18 Oct): on zoomed-region tokens, "Apply selection rule" carries the selection to
     this whole image or all images AT THE REGION'S SCALE. Rule = colour windows (the region's PCA incl. its
     foreground split + both windows), similar (prototype + threshold) or learned (logistic regression, numpy,
     balanced); auto = how the selection was made (lasso / brush -> learned). The token subprocess tokenises
     each image whole at that scale in tiles of the region's own size (on the region's image one tile IS the
     region, same patch-grid phase), applies the rule and keeps only the per-patch result; the result is a new
     token source "applied (fine)" (no plot; brush / assign as usual). Rule file + id go with the masks.
     REGIONS list: every cached zoomed region (any size) can be re-opened without re-extracting; "Extract this
     view" adds the visible part as a new region, at the same scale as the region it was panned from. Regions
     a category was assigned from are kept (settings file) and outlined on the strip.

The window talks to the editor only through a small `host` object (see BioRAGAnnotateWindow); the pure
parts are unit-tested in test_brush/test_biorag_annotate.py.

Foreground only (simple, documented): Otsu threshold on PC1 of a first PCA of all tokens; the side
that holds most of the image-border patches is called background (the "flip" box swaps the sides);
the PCA is then refitted on the foreground tokens only, and only those are shown and selectable.

Trained DINOMask view: the head was trained on LAST-layer tokens at the run's size with 4-shift
averaging, so it is run on its own features (computed by the token CLI with --dinomask_run), not on the
layer-18 cache. Class map = argmax of the guided-filtered 4-shift probabilities (as DINOMask, without
test-time rotation); hidden activations from the unshifted pass.
"""
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# a copy named *_dev.py pairs with biorag_annotate_tokens_v1_dev.py (development); renamed, it uses the live files
_SFX = "_dev" if os.path.basename(__file__).endswith("_dev.py") else ""
import importlib  # noqa: E402
bat = importlib.import_module("biorag_annotate_tokens_v1" + _SFX)  # pure helpers; torch is imported only by its CLI

TOKENS_SCRIPT = os.path.join(_HERE, "biorag_annotate_tokens_v1%s.py" % _SFX)
SIZES = (1024, 1536, 2048)
LAYERS = (6, 9, 12, 15, 18, 20, 21, 22, 23)   # 0-based blocks of ViT-L (0-23); 18 Oct: full sweep for AVD
VARIANTS = ("raw", "high-pass", "raw+high-pass")
PC_PAIRS = {"PC1 - PC2": (0, 1), "PC1 - PC3": (0, 2), "PC2 - PC3": (1, 2)}
MAX_FIT = 60000
MAX_SHOW = 50000
MAX_HILITE = 15000
NONE_GREY = (0.55, 0.55, 0.55)


def default_token_python():
    """BIORAG_ANNOTATE_PY, else the python of the sibling conda env 'biorag' of the running python
    (.../envs/samm/bin/python -> .../envs/biorag/bin/python), else this python."""
    p = os.environ.get("BIORAG_ANNOTATE_PY")
    if p:
        return p
    exe = os.path.abspath(sys.executable)
    envs = os.path.dirname(os.path.dirname(os.path.dirname(exe)))     # .../envs
    cand = os.path.join(envs, "biorag", "bin", "python")
    if os.path.basename(envs) == "envs" and os.path.isfile(cand):
        return cand
    return exe


# ===================================================================== pure helpers
def highpass_grid(G, k=3):
    """G [gh, gw, D] -> G minus its k x k box mean (in patch units; border windows average only the
    patches that exist)."""
    G = np.asarray(G, np.float32)
    gh, gw = G.shape[:2]
    r = k // 2
    P = np.pad(G, ((r, r), (r, r), (0, 0)))
    one = np.pad(np.ones((gh, gw), np.float32), r)
    S = np.zeros_like(G)
    C = np.zeros((gh, gw), np.float32)
    for dy in range(k):
        for dx in range(k):
            S += P[dy:dy + gh, dx:dx + gw]
            C += one[dy:dy + gh, dx:dx + gw]
    return G - S / C[..., None]


def token_features(T, grid, variant):
    """T (gh*gw, D) -> float32 features for the PCA: raw, high-pass or both concatenated."""
    gh, gw = int(grid[0]), int(grid[1])
    X = np.asarray(T, np.float32)
    if variant == "raw":
        return X
    H = highpass_grid(X.reshape(gh, gw, -1), 3).reshape(gh * gw, -1)
    if variant == "high-pass":
        return H
    return np.concatenate([X, H], 1)


def fit_pca(X, ncomp=3):
    """X (n, d) -> (mean (d,), components (d, ncomp)). numpy SVD of the d x d covariance (same
    eigenvectors as the SVD of X - mean, cheaper for n >> d). Sign fixed: largest |loading| > 0."""
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    Xc = X - mu
    C = Xc.T @ Xc / max(1, len(X) - 1)
    U, _, _ = np.linalg.svd(C, full_matrices=False)
    V = U[:, :ncomp].copy()
    for j in range(V.shape[1]):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] *= -1
    return mu.astype(np.float32), V.astype(np.float32)


def project(X, mu, V):
    return (np.asarray(X, np.float32) - mu) @ V


def minmax_rgb(P, ref=None):
    """PC scores (n, 3) -> RGB in [0, 1], each PC min-max scaled over `ref` (default P itself)."""
    P = np.asarray(P, np.float32)
    R = P if ref is None or len(ref) == 0 else np.asarray(ref, np.float32)
    lo, hi = R.min(0), R.max(0)
    return np.clip((P - lo) / np.maximum(hi - lo, 1e-12), 0, 1)


def rgb_to_hsv(rgb):
    """RGB in [0, 1] (n, 3) -> hue in degrees [0, 360), saturation and value in [0, 1] (18 Oct)."""
    rgb = np.asarray(rgb, np.float32).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    return hsv[:, 0] % 360.0, hsv[:, 1], hsv[:, 2]


def hue_window_mask(hue, val, centre, width, vmin=0.0, vmax=1.0):
    """Tokens whose PCA colour hue lies within +-width/2 degrees of centre (wrapping at 360) and whose
    brightness (HSV value) lies in [vmin, vmax]."""
    d = np.abs((np.asarray(hue, np.float32) - float(centre) + 180.0) % 360.0 - 180.0)
    return (d <= float(width) / 2.0) & (np.asarray(val) >= vmin) & (np.asarray(val) <= vmax)


def grow_near(a, b, k):
    """Patches of grid b that lie within k patches (chessboard distance) of a patch of grid a, grown step by
    step THROUGH b, so a chain of b patches touching a is kept (18 Oct: setae edges next to the seta colour)."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    cur = a.copy()
    ker = np.ones((3, 3), np.uint8)
    for _ in range(int(k)):
        nxt = cv2.dilate(cur.astype(np.uint8), ker) > 0
        nxt &= (b | a)
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur & b


def otsu_threshold(v, bins=256):
    v = np.asarray(v, np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    h, e = np.histogram(v, bins=bins, range=(lo, hi))
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = w0 * w1 * (mu0 - mu1) ** 2
    best = np.flatnonzero(between >= between.max() - 1e-12 * max(1.0, between.max()))
    return float((e[best[0] + 1] + e[best[-1] + 1]) / 2)       # middle of the best plateau (upper bin edges)


def border_index(grid):
    """Flat indices of the patches on the edge of a gh x gw grid."""
    gh, gw = int(grid[0]), int(grid[1])
    m = np.zeros((gh, gw), bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return np.flatnonzero(m.ravel())


def foreground_from_pc1(pc1, border_flat_idx, flip=False):
    """Otsu on PC1; the side holding most border patches is background (flip swaps). -> bool fg, t."""
    pc1 = np.asarray(pc1, np.float32)
    t = otsu_threshold(pc1)
    hi = pc1 > t
    bh = hi[border_flat_idx].mean() if len(border_flat_idx) else 0.0
    fg = ~hi if bh > 0.5 else hi
    return (~fg if flip else fg), t


def gfilter(I, p, r=8, eps=1e-3):
    """Guided filter (He et al.), cv2.boxFilter implementation, exactly as in DINOMask
    (dinomask_predict_rot4_v1.py). I: HxW float32 guide; p: HxWxK float32."""
    mean = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))  # noqa: E731
    mI = mean(I)
    vI = mean(I * I) - mI * mI
    out = np.empty_like(p)
    for k in range(p.shape[2]):
        q = p[..., k]
        mp = mean(q)
        aa = (mean(I * q) - mI * mp) / (vI + eps)
        b = mp - aa * mI
        out[..., k] = mean(aa) * I + mean(b)
    return out


def remove_small_components(m, min_px):
    m = (np.asarray(m) > 0).astype(np.uint8)
    if min_px <= 0 or not m.any():
        return m
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]


def selection_to_mask(selgrid, image_hw, grey_work=None, r=8, eps=1e-3, min_frac=0.001, box=None, min_px=None):
    """Selected patches (gh, gw bool) -> uint8 0/1 mask at the ORIGINAL image size.
    box (x0, y0, x1, y1): the tokens cover only this crop of the image; the mask is made at the crop's
    size (min_frac then relative to the crop) and pasted into a full-size mask.
    Patch grid -> working size (nearest, exact 16 px blocks) -> guided filter on the grey working
    image (radius r, eps; as DINOMask, where r = 8 px is half a patch) -> original size (linear) ->
    > 0.5 -> components smaller than min_frac of the image removed. Without grey_work the blocks are
    only resized (no refinement)."""
    sel = np.asarray(selgrid, bool)
    H, W = int(image_hw[0]), int(image_hw[1])
    if box is not None and tuple(box) != (0, 0, W, H):
        x0, y0, x1, y1 = [int(v) for v in box]
        m = selection_to_mask(sel, (y1 - y0, x1 - x0), grey_work, r, eps, min_frac, min_px=min_px)
        return paste_mask(m, box, (H, W))
    if not sel.any():
        return np.zeros((H, W), np.uint8)
    gh, gw = sel.shape
    if grey_work is not None:
        nh, nw = grey_work.shape[:2]
        p = cv2.resize(sel.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST)
        q = gfilter(np.asarray(grey_work, np.float32), p[..., None], r, eps)[..., 0]
    else:
        q = sel.astype(np.float32)
    q = cv2.resize(q, (W, H), interpolation=cv2.INTER_LINEAR if grey_work is not None else cv2.INTER_NEAREST)
    m = (q > 0.5).astype(np.uint8)
    # min_px (18 Oct): an absolute minimum component size instead of min_frac (fine structures: 1 patch)
    return remove_small_components(m, int(round(min_frac * H * W)) if min_px is None else int(min_px))


def paste_mask(m, box, image_hw):
    """Crop mask m (y1-y0, x1-x0) -> full-size uint8 mask with m at box."""
    H, W = int(image_hw[0]), int(image_hw[1])
    x0, y0, x1, y1 = [int(v) for v in box]
    out = np.zeros((H, W), np.uint8)
    out[y0:y1, x0:x1] = (np.asarray(m)[:y1 - y0, :x1 - x0] > 0)
    return out


def selection_to_work_mask(selgrid, grey_work, r=8, eps=1e-3, min_frac=0.001):
    """Selected patches -> uint8 0/1 pixel mask at the WORKING size (the start of a brush edit):
    nearest upsampling + guided filter, > 0.5, tiny components removed."""
    sel = np.asarray(selgrid, bool)
    nh, nw = grey_work.shape[:2]
    if not sel.any():
        return np.zeros((nh, nw), np.uint8)
    p = cv2.resize(sel.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST)
    q = gfilter(np.asarray(grey_work, np.float32), p[..., None], r, eps)[..., 0]
    return remove_small_components((q > 0.5).astype(np.uint8), int(round(min_frac * nh * nw)))


def work_mask_to_image(pix, image_hw, box=None):
    """Brush-edited working-size mask -> uint8 mask at the original image size, as is (linear resize,
    > 0.5; no guided filter, no component removal)."""
    H, W = int(image_hw[0]), int(image_hw[1])
    x0, y0, x1, y1 = (0, 0, W, H) if box is None else [int(v) for v in box]
    q = cv2.resize(np.asarray(pix, np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    return paste_mask(q > 0.5, (x0, y0, x1, y1), (H, W))


def pix_to_patch_sel(pix, grid, thresh=0.5):
    """Working-size pixel mask -> patch selection: a patch is selected if >= thresh of its pixels are."""
    gh, gw = int(grid[0]), int(grid[1])
    m = np.asarray(pix, np.float32)
    if m.shape != (gh * 16, gw * 16):
        m = cv2.resize(m, (gw * 16, gh * 16), interpolation=cv2.INTER_NEAREST)
    return m.reshape(gh, 16, gw, 16).mean((1, 3)) >= thresh


def patch_centres(grid, box):
    gh, gw = int(grid[0]), int(grid[1])
    x0, y0, x1, y1 = box
    return x0 + (np.arange(gw) + 0.5) * (x1 - x0) / gw, y0 + (np.arange(gh) + 0.5) * (y1 - y0) / gh


def point_to_patch(x, y, grid, box):
    """Image point -> (row, col) of the patch under it, or None outside the token box."""
    gh, gw = int(grid[0]), int(grid[1])
    x0, y0, x1, y1 = box
    if x is None or y is None or not (x0 <= x < x1 and y0 <= y < y1):
        return None
    return min(gh - 1, int((y - y0) * gh / (y1 - y0))), min(gw - 1, int((x - x0) * gw / (x1 - x0)))


def normalize_rows(F):
    F = np.asarray(F, np.float32)
    return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)


def cosine_to(Fn, q):
    """Rows of Fn (already L2-normalised) . q/|q| -> cosine similarities."""
    q = np.asarray(q, np.float32)
    return np.asarray(Fn, np.float32) @ (q / max(float(np.linalg.norm(q)), 1e-8))


def zoom_box(vbox, cx, cy, factor, full_wh, min_size=8.0):
    """View box (x0, x1, y0, y1) zoomed by factor (< 1 = in) about (cx, cy); never larger than the image."""
    x0, x1, y0, y1 = [float(v) for v in vbox]
    W, H = float(full_wh[0]), float(full_wh[1])
    w = min(W, max(min_size, (x1 - x0) * factor))
    h = min(H, max(min_size, (y1 - y0) * factor))
    fx = (cx - x0) / max(x1 - x0, 1e-9)
    fy = (cy - y0) / max(y1 - y0, 1e-9)
    nx0, ny0 = cx - fx * w, cy - fy * h
    return pan_box((nx0, nx0 + w, ny0, ny0 + h), 0, 0, full_wh)


def pan_box(vbox, dx, dy, full_wh):
    """Move the view box by (dx, dy) image px, kept inside the image."""
    x0, x1, y0, y1 = [float(v) for v in vbox]
    W, H = float(full_wh[0]), float(full_wh[1])
    w, h = x1 - x0, y1 - y0
    nx0 = min(max(0.0, x0 + dx), max(0.0, W - w))
    ny0 = min(max(0.0, y0 + dy), max(0.0, H - h))
    return (nx0, nx0 + w, ny0, ny0 + h)


def grey_working_image(rgb, work_hw):
    nh, nw = int(work_hw[0]), int(work_hw[1])
    g = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (nw, nh), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0


# ---------------------------------------------------------------- RLE / polygons
def mask_to_rle(m):
    """Uncompressed COCO RLE {'size': [h, w], 'counts': [...]}, column-major, first run = 0s (the
    format the Descriptron loader reads)."""
    m = (np.asarray(m) > 0)
    h, w = m.shape
    flat = m.T.reshape(-1).astype(np.int8)
    if flat.size == 0:
        return {"size": [int(h), int(w)], "counts": [0]}
    ch = np.flatnonzero(np.diff(flat)) + 1
    b = np.concatenate([[0], ch, [flat.size]])
    runs = np.diff(b).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    return {"size": [int(h), int(w)], "counts": [int(x) for x in runs]}


def rle_string_to_counts(s):
    """Compressed COCO RLE string -> counts (pycocotools rleFrString, in pure python)."""
    if isinstance(s, bytes):
        s = s.decode("ascii")
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def rle_to_mask(rle):
    h, w = int(rle["size"][0]), int(rle["size"][1])
    counts = rle["counts"]
    if isinstance(counts, (str, bytes)):
        counts = rle_string_to_counts(counts)
    counts = np.asarray(counts, np.int64)
    vals = (np.arange(len(counts)) % 2).astype(np.uint8)
    flat = np.repeat(vals, counts)
    if flat.size < h * w:
        flat = np.concatenate([flat, np.zeros(h * w - flat.size, np.uint8)])
    return np.ascontiguousarray(flat[:h * w].reshape(w, h).T)


def mask_to_polygons(m):
    """Outer contours as COCO polygons, as the GUI's _segmentation_from_mask (RETR_EXTERNAL)."""
    m8 = (np.asarray(m) > 0).astype(np.uint8) * 255
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.flatten().tolist() for c in cs if c.size >= 6]


def segmentation_to_mask(seg, hw):
    """Any COCO segmentation (polygon list / flat polygon / json string of polygons / RLE dict with
    list or compressed counts) -> uint8 0/1 mask of size hw. Unknown -> empty mask."""
    H, W = int(hw[0]), int(hw[1])
    out = np.zeros((H, W), np.uint8)
    if seg is None:
        return out
    if isinstance(seg, str):
        try:
            seg = json.loads(seg)
        except Exception:
            return out
    if isinstance(seg, dict) and "counts" in seg:
        m = rle_to_mask(seg)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return m
    if isinstance(seg, list):
        if seg and not isinstance(seg[0], (list, tuple, str, dict)):
            seg = [seg]
        for s in seg:
            if isinstance(s, str):
                try:
                    s = json.loads(s)
                except Exception:
                    continue
            if isinstance(s, dict):
                out |= segmentation_to_mask(s, hw)
                continue
            a = np.asarray(s, np.float64).reshape(-1)
            if a.size >= 6:
                cv2.fillPoly(out, [np.round(a[:a.size // 2 * 2].reshape(-1, 1, 2)).astype(np.int32)], 1)
    return out


# ---------------------------------------------------------------- COCO in / out
def _key(name):
    return os.path.basename(str(name))


def _stem(name):
    return os.path.splitext(_key(name))[0]


def coco_annotations_by_image(coco, image_names):
    """{image file name: [annotation, ...]} for the names given, matching images by file name, then by
    stem (image_id strings that are paths are understood too)."""
    want = {_key(n): _key(n) for n in image_names}
    want_stem = {_stem(n): _key(n) for n in image_names}
    id2name = {}
    for im in coco.get("images", []):
        fn = im.get("file_name", "")
        nm = want.get(_key(fn)) or want_stem.get(_stem(fn))
        if nm:
            id2name[str(im.get("id"))] = nm
    out = {_key(n): [] for n in image_names}
    for a in coco.get("annotations", []):
        nm = id2name.get(str(a.get("image_id")))
        if nm is None and isinstance(a.get("image_id"), str):
            nm = want.get(_key(a["image_id"])) or want_stem.get(_stem(a["image_id"]))
        if nm is not None:
            out[nm].append(a)
    return out


def category_grid(anns, cat_names, image_hw, grid, box=None):
    """Category index per patch (gh, gw int, -1 = none): the category of the mask covering the patch
    centre; the smallest mask wins where masks overlap. cat_names: {category_id: index}.
    box: the tokens cover only this crop (x0, y0, x1, y1) of the image."""
    H, W = int(image_hw[0]), int(image_hw[1])
    gh, gw = int(grid[0]), int(grid[1])
    cx, cy = patch_centres(grid, box if box is not None else (0, 0, W, H))
    ys = np.clip(cy.astype(int), 0, H - 1)
    xs = np.clip(cx.astype(int), 0, W - 1)
    out = np.full((gh, gw), -1, np.int32)
    items = []
    for a in anns:
        if a.get("category_id") not in cat_names:
            continue
        seg = a.get("segmentation_rle") or a.get("segmentation")
        if not seg:
            continue
        m = segmentation_to_mask(seg, (H, W))
        area = int(m.sum())
        if area:
            items.append((area, cat_names[a["category_id"]], m[np.ix_(ys, xs)] > 0))
    for _, ci, hit in sorted(items, key=lambda t: -t[0]):
        out[hit] = ci
    return out


def build_coco(assignments, images_info, category_ids=None):
    """assignments: [{'category': name, 'params': {...}, 'masks': {file_name: rle}}, ...]
    images_info: {file_name: (H, W)}. category_ids: {name: id} to reuse (editor ids); new names get
    the next free ids. -> (polygon COCO dict, RLE COCO dict)."""
    category_ids = dict(category_ids or {})
    cats, cid = [], {}
    nxt = max([int(v) for v in category_ids.values() if isinstance(v, (int, np.integer))] + [0]) + 1
    for asg in assignments:
        nm = asg["category"]
        if nm in cid:
            continue
        if nm in category_ids:
            cid[nm] = int(category_ids[nm])
        else:
            while nxt in cid.values():
                nxt += 1
            cid[nm] = nxt
            nxt += 1
        cats.append({"id": cid[nm], "name": nm, "supercategory": "none"})
    names = []
    for asg in assignments:
        for fn in asg["masks"]:
            if fn not in names:
                names.append(fn)
    imgs = [{"id": k + 1, "file_name": fn, "height": int(images_info[fn][0]), "width": int(images_info[fn][1])}
            for k, fn in enumerate(names)]
    iid = {fn: k + 1 for k, fn in enumerate(names)}
    poly, rle = [], []
    aid = 1
    for asg in assignments:
        for fn, r in asg["masks"].items():
            m = rle_to_mask(r)
            if not m.any():
                continue
            x, y, w, h = cv2.boundingRect(m)
            base = {"id": aid, "image_id": iid[fn], "category_id": cid[asg["category"]],
                    "bbox": [int(x), int(y), int(w), int(h)], "area": float(m.sum()), "iscrowd": 0,
                    "source": "biorag_annotate", "biorag_annotate": dict(asg.get("params", {}))}
            poly.append(dict(base, segmentation=mask_to_polygons(m)))
            rle.append(dict(base, segmentation=r))
            aid += 1
    return ({"images": imgs, "annotations": poly, "categories": cats},
            {"images": [dict(i) for i in imgs], "annotations": rle, "categories": [dict(c) for c in cats]})


def merge_coco(base, new):
    """Append the annotations of `new` to a COPY of `base`. Existing annotations, images and categories
    are untouched; images are matched by file name (then stem), categories by name (new names get new
    ids); new annotations get new ids. -> merged dict."""
    import copy
    out = copy.deepcopy(base)
    out.setdefault("images", [])
    out.setdefault("annotations", [])
    out.setdefault("categories", [])

    def _int_max(vals):
        return max([int(v) for v in vals if isinstance(v, (int, np.integer)) and not isinstance(v, bool)] + [0])
    cat_by_name = {c.get("name"): c.get("id") for c in out["categories"]}
    cmap = {}
    nxt_c = _int_max([c.get("id") for c in out["categories"]]) + 1
    for c in new.get("categories", []):
        if c["name"] in cat_by_name:
            cmap[c["id"]] = cat_by_name[c["name"]]
        else:
            out["categories"].append({"id": nxt_c, "name": c["name"], "supercategory": c.get("supercategory", "none")})
            cat_by_name[c["name"]] = nxt_c
            cmap[c["id"]] = nxt_c
            nxt_c += 1
    by_name = {_key(i.get("file_name", "")): i for i in out["images"]}
    by_stem = {_stem(i.get("file_name", "")): i for i in out["images"]}
    ids_int = all(isinstance(i.get("id"), (int, np.integer)) for i in out["images"])
    nxt_i = _int_max([i.get("id") for i in out["images"]]) + 1
    imap = {}
    for im in new.get("images", []):
        hit = by_name.get(_key(im["file_name"])) or by_stem.get(_stem(im["file_name"]))
        if hit is None:
            nid = nxt_i if ids_int else _stem(im["file_name"])
            nxt_i += 1
            hit = {"id": nid, "file_name": im["file_name"], "height": im["height"], "width": im["width"]}
            out["images"].append(hit)
            by_name[_key(im["file_name"])] = hit
        imap[im["id"]] = hit["id"]
    nxt_a = _int_max([a.get("id") for a in out["annotations"]]) + 1
    for a in new.get("annotations", []):
        b = dict(a)
        b["id"] = nxt_a
        nxt_a += 1
        b["image_id"] = imap[a["image_id"]]
        b["category_id"] = cmap[a["category_id"]]
        out["annotations"].append(b)
    return out


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def category_palette(n):
    """n distinct colours (tab20, then golden-angle hues)."""
    import colorsys
    tab20 = [(0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
             (0.58, 0.4, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.74, 0.74, 0.13),
             (0.09, 0.75, 0.81), (0.68, 0.78, 0.91), (1.0, 0.73, 0.47), (0.6, 0.87, 0.54),
             (1.0, 0.6, 0.59), (0.77, 0.69, 0.84), (0.77, 0.61, 0.58), (0.97, 0.71, 0.82),
             (0.86, 0.86, 0.55), (0.62, 0.85, 0.9)]
    out = []
    for i in range(n):
        if i < len(tab20):
            out.append(tab20[i])
        else:
            out.append(colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.65, 0.95))
    return out


# ===================================================================== selection rules (18 Oct, AVD)
# A selection built on zoomed-region tokens (colour windows, similar, lasso, brush) can be APPLIED to the whole
# image / all images at the same fine resolution. The rule is everything needed to redo the selection on new
# tokens: the crop's PCA (incl. its foreground split) + colour windows, or a prototype + threshold, or a
# logistic regression. pca_fit_tokens / colour_window_selection are used by the window itself, so applying a
# colour rule to the crop's own tokens reproduces the crop selection exactly (tested).
RULE_KINDS = ("colour windows", "similar", "learned")


def pca_fit_tokens(items, variant, fg=False, flip=False, max_fit=MAX_FIT, seed=0):
    """items: [(T (n, D), grid), ...] -> (P (N, 3), rgb (N, 3), vis (N,), prm). The window's PCA: fit on a
    random subsample (<= max_fit), all tokens projected; foreground = Otsu on PC1 of a first PCA, the side with
    most border patches is background (flip swaps), PCA refitted on the foreground (if >= 10 tokens); PCA-RGB
    = min-max over the visible tokens. prm holds what a rule needs to redo this on other tokens."""
    N = sum(int(g[0]) * int(g[1]) for _, g in items)
    offs = np.cumsum([0] + [int(g[0]) * int(g[1]) for _, g in items])
    rng = np.random.default_rng(seed)
    frac = min(1.0, max_fit / max(1, N))
    samp = [np.sort(rng.choice(int(g[0]) * int(g[1]), size=max(1, int(round(int(g[0]) * int(g[1]) * frac))),
                               replace=False)) for _, g in items]

    def fit_and_project(keep):
        Xs = []
        for (T, g), s, o in zip(items, samp, offs):
            F = token_features(T, g, variant)
            s2 = s if keep is None else s[keep[o + s]]
            Xs.append(F[s2])
        mu, V = fit_pca(np.concatenate(Xs, 0))
        P = np.zeros((N, 3), np.float32)
        for (T, g), o in zip(items, offs):
            P[o:o + int(g[0]) * int(g[1])] = project(token_features(T, g, variant), mu, V)
        return P, mu, V
    P, mu1, V1 = fit_and_project(None)
    mu, V = mu1, V1
    vis = np.ones(N, bool)
    prm = {"variant": variant, "fg": bool(fg), "mu1": mu1, "V1": V1}
    if fg:
        bidx = np.concatenate([o + border_index(g) for (_, g), o in zip(items, offs)])
        t = otsu_threshold(P[:, 0])
        hi = P[:, 0] > t
        border_hi = hi[bidx].mean() > 0.5
        vis = ~hi if border_hi else hi
        if flip:
            vis = ~vis
        prm.update({"t": float(t), "fg_hi": bool((not border_hi) != bool(flip))})
        if vis.sum() >= 10:
            P, mu, V = fit_and_project(vis)
    R = P[vis] if vis.any() else P
    rgb = minmax_rgb(P, R)
    prm.update({"mu": mu, "V": V, "lo": np.asarray(R, np.float32).min(0), "hi": np.asarray(R, np.float32).max(0)})
    return P, rgb, vis, prm


def colour_window_selection(rgb, grids, windows):
    """PCA-RGB (N, 3) -> (selected, m1, m2) by the two colour windows. grids: [(offset, grid), ...] for the
    colour-2 growth per image grid. windows: [{'on', 'c', 'w', 'lo', 'hi'}, {... 'near'}] (lo/hi in %)."""
    hue, _, val = rgb_to_hsv(rgb)
    n = len(hue)

    def one(d):
        if not d.get("on"):
            return np.zeros(n, bool)
        return hue_window_mask(hue, val, float(d["c"]), int(d["w"]), int(d["lo"]) / 100.0, int(d["hi"]) / 100.0)
    m1 = one(windows[0])
    m2 = one(windows[1]) if len(windows) > 1 else np.zeros(n, bool)
    near = int(windows[1].get("near", 0)) if len(windows) > 1 else 0
    if len(windows) > 1 and windows[1].get("on") and near > 0:
        keep2 = np.zeros(n, bool)
        for o, g in grids:
            sl = slice(o, o + int(g[0]) * int(g[1]))
            keep2[sl] = grow_near(m1[sl].reshape(g), m2[sl].reshape(g), near).ravel()
        m2 = keep2
    return m1 | m2, m1, m2


def train_logreg(X, y, l2=1.0, iters=400, lr=0.05, seed=0, sample_weight=None):
    """Logistic regression, numpy only (deterministic): standardised features, balanced class weights, L2,
    Adam on the full batch. -> {'mean', 'std', 'w', 'b'}. sample_weight (19 Oct, optional): per-sample
    multipliers on the balanced weights (None = unchanged)."""
    X = np.asarray(X, np.float32)
    y = np.asarray(y, bool)
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    Z = (X - mean) / std
    n1, n0 = max(1, int(y.sum())), max(1, int((~y).sum()))
    sw = np.where(y, len(y) / (2.0 * n1), len(y) / (2.0 * n0)).astype(np.float32)
    if sample_weight is not None:              # 19 Oct: explicit "not this" negatives up-weighted
        sw = (sw * np.asarray(sample_weight, np.float32)).astype(np.float32)
    w = np.zeros(Z.shape[1], np.float32)
    b = 0.0                       # a Python float: a numpy float64 scalar would turn every product into float64
    mw, vw, mb, vb = np.zeros_like(w), np.zeros_like(w), 0.0, 0.0
    yt = y.astype(np.float32)
    for it in range(1, iters + 1):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        g = (sw * (p - yt) / len(yt)).astype(np.float32)
        gw = Z.T @ g + l2 * w / len(yt)
        gb = float(g.sum())
        mw = 0.9 * mw + 0.1 * gw
        vw = 0.999 * vw + 0.001 * gw * gw
        mb = 0.9 * mb + 0.1 * gb
        vb = 0.999 * vb + 0.001 * gb * gb
        w -= lr * (mw / (1 - 0.9 ** it)) / (np.sqrt(vw / (1 - 0.999 ** it)) + 1e-8)
        b = float(b - lr * (mb / (1 - 0.9 ** it)) / (np.sqrt(vb / (1 - 0.999 ** it)) + 1e-8))
    return {"mean": mean, "std": std.astype(np.float32), "w": w, "b": float(b)}


def logreg_proba(X, m):
    Z = (np.asarray(X, np.float32) - m["mean"]) / m["std"]
    return 1.0 / (1.0 + np.exp(-np.clip(Z @ m["w"] + m["b"], -30, 30)))


# ------------------------------------------------------------ "not this" (negative) brush, 19 Oct
NOT_THIS_WEIGHTING = ("explicit negatives x max(1, n_negatives / n_explicit), so together they weigh at least as "
                      "much as all positives; balanced class weights otherwise unchanged")
_KEEP_NEG = object()          # undo entries without a not-this mask to restore


def paint_stroke(m, pa, pb, rad, val):
    """One brush segment (as the strip brush): a line of width 2 rad + 1 with round ends, in place."""
    cv2.line(m, pa, pb, int(val), thickness=2 * rad + 1)
    cv2.circle(m, pa, rad, int(val), -1)
    cv2.circle(m, pb, rad, int(val), -1)
    return m


def paint_not_this(neg, pos, pa, pb, rad):
    """Not-this stroke: marks neg and removes the same pixels from pos (negative wins; pos may be None)."""
    paint_stroke(neg, pa, pb, rad, 1)
    if pos is not None:
        paint_stroke(pos, pa, pb, rad, 0)


def paint_positive(pos, neg, pa, pb, rad):
    """Brush stroke: adds to pos and removes not-this paint there (last stroke wins; neg may be None)."""
    paint_stroke(pos, pa, pb, rad, 1)
    if neg is not None:
        paint_stroke(neg, pa, pb, rad, 0)


def negative_patches(neg_pix, grid, thresh=0.5):
    """Not-this pixel mask -> patch grid: a patch is an explicit negative if >= thresh of its pixels are."""
    return pix_to_patch_sel(neg_pix, grid, thresh)


def drop_negatives(sel, neg):
    """Selection minus explicit negatives (any shape; neg None = unchanged)."""
    if neg is None:
        return sel
    return np.asarray(sel, bool) & ~np.asarray(neg, bool)


def map_negatives(neg_work, src_box, image_hw, dst_box, dst_work_hw):
    """Not-this paint of one view (working size, covering src_box of the image) -> working-size mask of
    another view of the same image (dst_box, dst_work_hw); zero outside src_box."""
    full = work_mask_to_image(neg_work, image_hw, src_box)
    x0, y0, x1, y1 = [int(v) for v in dst_box]
    nh, nw = int(dst_work_hw[0]), int(dst_work_hw[1])
    q = cv2.resize(full[y0:y1, x0:x1].astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR)
    return (q > 0.5).astype(np.uint8)


def explicit_negative_weights(y, neg):
    """-> (sample weights | None, factor). Explicit negatives (never positives) get factor
    max(1, n_negatives / n_explicit) on top of the balanced class weights, everything else 1."""
    y = np.asarray(y, bool)
    neg = np.asarray(neg, bool) & ~y
    ne = int(neg.sum())
    if ne == 0:
        return None, 1.0
    f = max(1.0, int((~y).sum()) / float(ne))
    w = np.ones(len(y), np.float32)
    w[neg] = f
    return w, float(f)


def learned_training_set(sel, vis, neg=None):
    """Training set of the learned rule -> (keep, y, sample_weight | None, info for the rule).
    No explicit negatives: keep = vis, y = sel, no weights (the training before 19 Oct, bit-identical).
    With them: positives vs explicit negatives + the other unselected visible patches; explicit negatives
    are kept even outside the foreground, are never positive and are up-weighted (explicit_negative_weights).
    sample_weight is over the kept patches (X[keep])."""
    sel = np.asarray(sel, bool)
    vis = np.asarray(vis, bool)
    if neg is None or not np.asarray(neg, bool).any():
        return vis, sel, None, {}
    neg = np.asarray(neg, bool)
    keep = vis | neg
    y = sel & ~neg
    w, f = explicit_negative_weights(y[keep], neg[keep])
    info = {"lr_n_explicit_neg": int(neg[keep].sum()), "lr_explicit_neg_weight": round(f, 4),
            "lr_weighting": NOT_THIS_WEIGHTING}
    return keep, y, w, info


def rule_foreground(F, rule):
    """Foreground of new tokens with the crop's FITTED split (first PCA, Otsu threshold, side); no refit."""
    if not rule.get("fg"):
        return np.ones(len(F), bool)
    hi = project(F, rule["mu1"], rule["V1"])[:, 0] > rule["t"]
    return hi if rule["fg_hi"] else ~hi


def rule_scores(T, grid, rule):
    """Apply a selection rule to the tokens of one grid -> (selected (n,) bool, score (n,) float32).
    Scores: colour windows = 1 - hue distance / half width of the nearest active window (0..1, selected
    tokens >= 0); similar = cosine similarity; learned = logistic probability."""
    F = token_features(T, grid, rule["variant"])
    vis = rule_foreground(F, rule)
    kind = rule["kind"]
    if kind == "colour windows":
        P = project(F, rule["mu"], rule["V"])
        lo, hi = np.asarray(rule["lo"], np.float32), np.asarray(rule["hi"], np.float32)
        rgb = np.clip((P - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
        sel, _, _ = colour_window_selection(rgb, [(0, grid)], rule["windows"])
        hue, _, _ = rgb_to_hsv(rgb)
        score = np.full(len(F), -1.0, np.float32)
        for d in rule["windows"]:
            if d.get("on"):
                dist = np.abs((hue - float(d["c"]) + 180.0) % 360.0 - 180.0)
                score = np.maximum(score, 1.0 - dist / max(float(d["w"]) / 2.0, 1e-6))
    elif kind == "similar":
        score = cosine_to(normalize_rows(F), rule["proto"]).astype(np.float32)
        sel = score >= float(rule["threshold"])
    elif kind == "learned":
        score = logreg_proba(F, {"mean": rule["lr_mean"], "std": rule["lr_std"], "w": rule["lr_w"],
                                 "b": float(rule["lr_b"])}).astype(np.float32)
        sel = score >= float(rule["threshold"])
    else:
        raise ValueError(f"unknown rule kind {kind!r}")
    return sel & vis, score


_RULE_ARRAYS = ("mu1", "V1", "mu", "V", "lo", "hi", "proto", "lr_mean", "lr_std", "lr_w")


def save_rule(path, rule):
    """Rule -> npz (arrays as arrays, everything else as JSON in 'meta'). Returns the rule id (md5 of the
    content, 10 hex), which names the per-image result files."""
    import hashlib
    arrs = {k: np.asarray(rule[k], np.float32) for k in _RULE_ARRAYS if k in rule and rule[k] is not None}
    meta = {k: v for k, v in rule.items() if k not in _RULE_ARRAYS}
    h = hashlib.md5(json.dumps(meta, sort_keys=True, default=str).encode())
    for k in sorted(arrs):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrs[k]).tobytes())
    rid = h.hexdigest()[:10]
    meta["rule_id"] = rid
    tmp = path + ".tmp.npz"
    np.savez(tmp, meta=np.str_(json.dumps(meta, default=str)), **arrs)
    os.replace(tmp, path)
    return rid


def load_rule(path):
    with np.load(path, allow_pickle=False) as z:
        rule = json.loads(str(z["meta"]))
        for k in z.files:
            if k != "meta":
                rule[k] = z[k]
    return rule


def rule_summary(rule):
    """The JSON-able part of a rule (for the assign metadata): no arrays."""
    return {k: v for k, v in rule.items() if k not in _RULE_ARRAYS}


# ===================================================================== the window
VIEWS = ("photo", "annotations", "pca", "dinomask")
DM_SIZES = (1024, 1344, 1536, 2048)
MAX_FEAT_CACHE_BYTES = 1.5e9


class ImageRec:
    """Tokens of one image, or of one crop box of it (zoomed region)."""

    def __init__(self, path, z):
        self.path = path
        self.name = os.path.basename(path)
        self.T = z["tokens"]
        self.grid = (int(z["grid"][0]), int(z["grid"][1]))
        self.image_hw = (int(z["image_hw"][0]), int(z["image_hw"][1]))
        self.work_hw = (int(z["work_hw"][0]), int(z["work_hw"][1]))
        H, W = self.image_hw
        self.box = tuple(int(v) for v in z["box"]) if "box" in z else (0, 0, W, H)
        self.is_crop = self.box != (0, 0, W, H)
        self.label = self.name + (" [region %d,%d-%d,%d]" % self.box if self.is_crop else "")
        self.n = self.grid[0] * self.grid[1]
        self.offset = 0
        self._grey = None

    def grey(self):
        if self._grey is None:
            rgb = bat.load_rgb8(self.path)
            x0, y0, x1, y1 = self.box
            self._grey = grey_working_image(rgb[y0:y1, x0:x1], self.work_hw)
        return self._grey


class FineRec(ImageRec):
    """Result of a selection rule applied to a whole image at the zoomed region's fine scale (18 Oct): the
    per-patch selection + score only; the fine tokens themselves are not kept."""

    def __init__(self, path, z):
        self.path = path
        self.name = os.path.basename(path)
        self.T = None
        self.fine = True
        self.grid = (int(z["grid"][0]), int(z["grid"][1]))
        self.image_hw = (int(z["image_hw"][0]), int(z["image_hw"][1]))
        self.work_hw = (int(z["work_hw"][0]), int(z["work_hw"][1]))
        H, W = self.image_hw
        self.box = tuple(int(v) for v in z["box"]) if "box" in z else (0, 0, W, H)   # <= 1 patch trimmed at edges
        self.is_crop = False
        self.label = self.name + " [applied, fine]"
        self.n = self.grid[0] * self.grid[1]
        self.offset = 0
        self._grey = None
        self.sel0 = np.asarray(z["sel"], bool).ravel()
        self.score = np.asarray(z["score"], np.float32).ravel()


class BioRAGAnnotateWindow:
    """host: object with
        root, images() -> [paths], current_image_path() -> path|None, editor_image_shape() -> (H, W)|None,
        add_mask_to_editor(mask, name|None) -> (final_name, annotation_id)|None,
        delete_editor_mask(annotation_id) -> bool, category_cascade(menu, command_for) -> tk.Menu,
        category_ids() -> {name: id}, post_menu(menu, event), folder_coco_path() -> path|None"""

    def __init__(self, host):
        import tkinter as tk
        from tkinter import ttk
        from collections import OrderedDict
        self.tk, self.ttk, self.host = tk, ttk, host
        self.win = tk.Toplevel(host.root)
        self.win.title("BioRAG Annotate")
        self.win.geometry("1450x1000")
        self.paths = [p for p in host.images() if p]
        first = self.paths[0] if self.paths else ""
        self.recs, self.recs_whole, self.recs_region, self.crops = [], [], [], []
        self.failed = {}
        self.P = self.pca_rgb = self.colors = self.vis = self.sel = self.show_idx = None
        self.catidx, self.cat_names, self.cat_cols, self._cat_per, self._cid2idx = None, [], [], None, {}
        self.pix, self._stroke, self._undo_strokes = {}, None, []
        self.neg_pix = {}                        # 19 Oct: "not this" pixel masks per image (working size)
        self.sim, self._sim_base, self._sim_label = None, None, ""
        self._featn = {}
        self.dm, self.dm_classes, self.dm_cols, self.dm_run = {}, [], [], None
        self._thumbs, self._full, self._ann_cache, self._dm_thumb = {}, OrderedDict(), {}, {}
        self.cur, self.vbox = 0, None
        self.proc, self._proc_kind = None, None
        self.q = queue.Queue()
        self.assignments = []
        self.session_ts = time.strftime("%Y%m%d_%H%M%S")
        self.last_written = None
        self._busy = False
        self._pca_job = 0
        self._contours, self._dm_legend = [], None
        self._press, self._moved, self._pan, self._space = None, False, None, False
        self._lod_job = None
        self.params_tokens = {}
        self._applied_mode, self.recs_applied, self._pca_prm, self._sel_origin, self._sim_query = False, [], None, None, None
        self._crop_size, self._annotated_regions, self._region_artists = {}, [], []   # 18 Oct: regions list

        self.v_size = tk.StringVar(value="1536")
        self.v_layer = tk.StringVar(value="18")
        self.v_cache = tk.StringVar(value=os.path.join(os.path.dirname(first), "biorag_annotate_cache") if first else "")
        self.v_variant = tk.StringVar(value="raw")
        self.v_pcs = tk.StringVar(value="PC1 - PC2")
        self.v_fg = tk.BooleanVar(value=False)
        self.v_flip = tk.BooleanVar(value=False)
        self.v_colour = tk.StringVar(value="PCA-RGB")
        self.v_lasso = tk.StringVar(value="replace")
        self.v_src = tk.StringVar(value="whole")
        self.v_region_size = tk.StringVar(value="2048")
        self.v_sim = tk.DoubleVar(value=0.80)
        self.v_view = tk.StringVar(value="pca")
        self.v_dmview = tk.StringVar(value="classes")
        self.v_alpha = tk.DoubleVar(value=0.55)
        self.v_tool = tk.StringVar(value="")
        self.v_brush = tk.IntVar(value=40)
        self.v_dm_size = tk.StringVar(value="2048")
        self.v_status = tk.StringVar(value=f"{len(self.paths)} image(s). Press 'Extract / load tokens'.")
        self.v_selinfo = tk.StringVar(value="")
        self.v_catsrc = tk.StringVar(value="")
        self.v_dmsrc = tk.StringVar(value="(none)")
        self.v_python = tk.StringVar(value=default_token_python())
        self.v_rule = tk.StringVar(value="auto")              # 18 Oct: apply a zoomed-region selection elsewhere
        self.v_lr_thr = tk.DoubleVar(value=0.5)
        self.v_apply_scope = tk.StringVar(value="image")
        self.v_same_scale = tk.BooleanVar(value=True)
        self.v_region_pick = tk.StringVar(value="")
        self._build()
        w = self.win
        w.bind("<Left>", lambda e: None if self._typing(e) else self._step(-1))
        w.bind("<Right>", lambda e: None if self._typing(e) else self._step(1))
        w.bind("<Key-v>", lambda e: None if self._typing(e) else self.cycle_view())
        w.bind("<Key-0>", lambda e: None if self._typing(e) else self.reset_zoom())
        for _k in ("<Key-plus>", "<Key-equal>", "<KP_Add>"):
            w.bind(_k, lambda e: None if self._typing(e) else self.zoom_step(0.5))
        for _k in ("<Key-minus>", "<KP_Subtract>"):
            w.bind(_k, lambda e: None if self._typing(e) else self.zoom_step(2.0))
        w.bind("<Key-b>", lambda e: None if self._typing(e) else self.toggle_big())
        w.bind("<KeyPress-space>", lambda e: None if self._typing(e) else self._space_key(True))
        w.bind("<KeyRelease-space>", lambda e: self._space_key(False))
        w.bind("<Escape>", lambda e: self._cancel_stroke())
        w.bind("<Control-z>", lambda e: None if self._typing(e) else self.undo_stroke())
        w.protocol("WM_DELETE_WINDOW", self.close)
        w.after(200, self._poll)
        self._want_region = False
        self._restore_settings()                 # 18 Oct: last settings + zoomed regions come back on reopen
        self._try_load_cached()

    def _typing(self, e):
        try:
            return e.widget.winfo_class() in ("Entry", "TEntry", "TCombobox", "Text", "Spinbox")
        except Exception:
            return False

    # ------------------------------------------------------------ layout
    def _build(self):
        tk, ttk = self.tk, self.ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.widgets import LassoSelector
        from matplotlib.backend_bases import MouseButton
        from matplotlib.patches import Circle

        r1 = tk.Frame(self.win)
        r1.pack(side="top", fill="x", padx=4, pady=2)
        tk.Label(r1, text="Size").pack(side="left")
        ttk.Combobox(r1, textvariable=self.v_size, values=[str(s) for s in SIZES], width=5, state="readonly").pack(side="left")
        tk.Label(r1, text="Layer").pack(side="left", padx=(6, 0))
        ttk.Combobox(r1, textvariable=self.v_layer, values=[str(s) for s in LAYERS], width=4, state="readonly").pack(side="left")
        tk.Label(r1, text="Cache folder").pack(side="left", padx=(6, 0))
        tk.Entry(r1, textvariable=self.v_cache, width=44).pack(side="left")
        tk.Button(r1, text="...", command=self._pick_cache).pack(side="left")
        self.b_extract = tk.Button(r1, text="Extract / load tokens", bg="#cfe8d5", command=self.extract)
        self.b_extract.pack(side="left", padx=(8, 0))
        self.b_stop = tk.Button(r1, text="Stop", state="disabled", command=self.stop)
        self.b_stop.pack(side="left")
        tk.Label(r1, textvariable=self.v_status, anchor="w", fg="#333").pack(side="left", padx=8, fill="x", expand=True)

        r2 = tk.Frame(self.win)
        r2.pack(side="top", fill="x", padx=4, pady=1)
        tk.Label(r2, text="Tokens").pack(side="left")
        cb = ttk.Combobox(r2, textvariable=self.v_variant, values=VARIANTS, width=13, state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_variant())
        tk.Label(r2, text="PCs").pack(side="left", padx=(6, 0))
        cb = ttk.Combobox(r2, textvariable=self.v_pcs, values=list(PC_PAIRS), width=10, state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._redraw_scatter(full=True))
        tk.Checkbutton(r2, text="foreground only", variable=self.v_fg, command=self.compute_pca).pack(side="left", padx=(6, 0))
        tk.Checkbutton(r2, text="flip fg/bg", variable=self.v_flip, command=self.compute_pca).pack(side="left")
        tk.Label(r2, text="Colour points by").pack(side="left", padx=(10, 0))
        cb = ttk.Combobox(r2, textvariable=self.v_colour, values=("PCA-RGB", "mask category"), width=13, state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_colour_mode())
        tk.Button(r2, text="Mask file...", command=self.pick_category_file).pack(side="left")
        tk.Label(r2, textvariable=self.v_catsrc, fg="#555").pack(side="left")
        tk.Label(r2, text="Select:").pack(side="left", padx=(10, 0))
        for v in ("replace", "add", "subtract"):
            tk.Radiobutton(r2, text=v, value=v, variable=self.v_lasso).pack(side="left")
        tk.Button(r2, text="Clear selection", command=self.clear_selection).pack(side="left", padx=(6, 0))

        r3 = tk.Frame(self.win)
        r3.pack(side="top", fill="x", padx=4, pady=1)
        tk.Label(r3, text="Token source").pack(side="left")
        tk.Radiobutton(r3, text="whole image", value="whole", variable=self.v_src, command=self._on_src).pack(side="left")
        tk.Radiobutton(r3, text="zoomed region", value="region", variable=self.v_src, command=self._on_src).pack(side="left")
        self.b_region = tk.Button(r3, text="Extract this view (new region)", bg="#cfe8d5", command=self.extract_region)
        self.b_region.pack(side="left", padx=(6, 0))
        tk.Label(r3, text="region size").pack(side="left", padx=(4, 0))
        ttk.Combobox(r3, textvariable=self.v_region_size, values=("1024", "1536", "2048"), width=5,
                     state="readonly").pack(side="left")
        tk.Checkbutton(r3, text="same scale as this region", variable=self.v_same_scale).pack(side="left", padx=(4, 0))
        tk.Label(r3, text="Regions").pack(side="left", padx=(8, 0))
        self.cb_regions = ttk.Combobox(r3, textvariable=self.v_region_pick, values=[], width=34, state="readonly")
        self.cb_regions.pack(side="left")
        self.cb_regions.bind("<<ComboboxSelected>>", lambda e: self.open_region_pick())
        self.cb_regions.bind("<Button-1>", lambda e: self._refresh_regions_list(), add="+")
        tk.Label(r3, text="Similar >=").pack(side="left", padx=(14, 0))
        tk.Scale(r3, from_=0.30, to=1.0, resolution=0.01, orient="horizontal", variable=self.v_sim, length=180,
                 command=lambda v: self._update_sim()).pack(side="left")
        tk.Label(r3, text="(right-click the image: Select similar)", fg="#666").pack(side="left", padx=4)
        tk.Radiobutton(r3, text="applied (fine)", value="applied", variable=self.v_src,
                       command=self._on_src).pack(side="left", padx=(10, 0))

        r4 = tk.Frame(self.win)                 # 18 Oct: apply the zoomed-region selection to the whole image(s)
        r4.pack(side="top", fill="x", padx=4, pady=1)
        tk.Label(r4, text="Selection rule").pack(side="left")
        for v in ("auto",) + RULE_KINDS:
            tk.Radiobutton(r4, text=v, value=v, variable=self.v_rule).pack(side="left")
        tk.Label(r4, text="learned threshold").pack(side="left", padx=(6, 0))
        tk.Spinbox(r4, from_=0.05, to=0.95, increment=0.05, width=5, textvariable=self.v_lr_thr).pack(side="left")
        tk.Label(r4, text="Apply to:").pack(side="left", padx=(12, 0))
        tk.Radiobutton(r4, text="this image", value="image", variable=self.v_apply_scope).pack(side="left")
        tk.Radiobutton(r4, text="all images", value="all", variable=self.v_apply_scope).pack(side="left")
        self.b_apply = tk.Button(r4, text="Apply selection rule", bg="#cfe8d5", state="disabled",
                                 command=self.apply_rule)
        self.b_apply.pack(side="left", padx=(6, 0))
        self.v_rule_info = tk.StringVar(value="(zoomed region + a selection -> same selection, same fine scale, "
                                              "on the whole image or all images)")
        tk.Label(r4, textvariable=self.v_rule_info, fg="#666").pack(side="left", padx=4)

        body = tk.Frame(self.win)
        body.pack(side="top", fill="both", expand=True)
        side = tk.Frame(body, width=270)
        side.pack(side="right", fill="y", padx=4)
        side.pack_propagate(False)
        tk.Label(side, text="Categories (click = select)", font=("Arial", 9, "bold")).pack(anchor="w")
        lf = tk.Frame(side)
        lf.pack(fill="both", expand=True)
        self.legend_canvas = tk.Canvas(lf, highlightthickness=0)
        sb = tk.Scrollbar(lf, orient="vertical", command=self.legend_canvas.yview)
        self.legend_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.legend_canvas.pack(side="left", fill="both", expand=True)
        self.legend = tk.Frame(self.legend_canvas)
        self.legend_canvas.create_window((0, 0), window=self.legend, anchor="nw")
        self.legend.bind("<Configure>", lambda e: self.legend_canvas.configure(scrollregion=self.legend_canvas.bbox("all")))
        tk.Label(side, text="Assigned (category, images, patches)", font=("Arial", 9, "bold")).pack(anchor="w", pady=(6, 0))
        self.lb = tk.Listbox(side, height=8)
        self.lb.pack(fill="x")
        tk.Button(side, text="Undo last", command=self.undo_last).pack(fill="x")
        tk.Button(side, text="Merge into folder annotations...", command=self.merge_into_folder).pack(fill="x", pady=(4, 0))
        tk.Label(side, text="Trained DINOMask (view)", font=("Arial", 9, "bold")).pack(anchor="w", pady=(8, 0))
        dmf = tk.Frame(side)
        dmf.pack(fill="x")
        self.b_dm = tk.Button(dmf, text="DINOMask run...", command=self.pick_dinomask)
        self.b_dm.pack(side="left")
        tk.Label(dmf, text="size").pack(side="left", padx=(4, 0))
        ttk.Combobox(dmf, textvariable=self.v_dm_size, values=[str(s) for s in DM_SIZES], width=5,
                     state="readonly").pack(side="left")
        tk.Label(side, textvariable=self.v_dmsrc, fg="#555", anchor="w").pack(fill="x")
        tk.Label(side, text="Token env python").pack(anchor="w", pady=(8, 0))
        tk.Entry(side, textvariable=self.v_python).pack(fill="x")

        self._build_hue_column(body)          # 18 Oct: colour-window selector (vertical rainbow)
        pw = tk.PanedWindow(body, orient="vertical", sashrelief="raised", sashwidth=6)
        self._pw = pw
        pw.pack(side="left", fill="both", expand=True)
        top = tk.Frame(pw)
        bot = tk.Frame(pw)
        pw.add(top, stretch="always", height=480)
        pw.add(bot, stretch="always", height=460)

        self.fig_s = Figure(figsize=(9, 5.2), dpi=100, facecolor="#2a2a2a")
        self.ax_s = self.fig_s.add_axes([0.05, 0.08, 0.93, 0.9], facecolor="#1b1b1b")
        self.ax_s.tick_params(colors="#bbbbbb", labelsize=7)
        self.sc_all = self.ax_s.scatter([], [], s=2, linewidths=0, rasterized=True)
        self.sc_sel = self.ax_s.scatter([], [], s=16, linewidths=0.7, edgecolors="white", zorder=3)
        self.sc_neg = self.ax_s.scatter([], [], s=14, linewidths=0, c="red", zorder=4)   # 19 Oct: not this
        self.cv_s = FigureCanvasTkAgg(self.fig_s, master=top)
        self.cv_s.get_tk_widget().pack(fill="both", expand=True)
        self.lasso_s = LassoSelector(self.ax_s, self._on_lasso_scatter, button=MouseButton.LEFT,
                                     props={"color": "yellow", "linewidth": 1.2})
        self.cv_s.mpl_connect("button_press_event", self._on_press_scatter)

        self.fig_i = Figure(figsize=(9, 3.4), dpi=100, facecolor="#2a2a2a")
        self.ax_i = self.fig_i.add_axes([0.0, 0.0, 1.0, 1.0])
        self.ax_i.set_axis_off()
        self.im_photo = self.ax_i.imshow(np.zeros((2, 2, 3), np.uint8), interpolation="antialiased")
        self.im_over = self.ax_i.imshow(np.zeros((2, 2, 4), np.float32), interpolation="nearest", zorder=2)
        self.im_neg = self.ax_i.imshow(np.zeros((2, 2, 4), np.float32), interpolation="nearest", zorder=3)
        self.im_sel = self.ax_i.imshow(np.zeros((2, 2, 4), np.float32), interpolation="nearest", zorder=4)
        self.cursor = Circle((0, 0), 10, fill=False, ec="yellow", lw=1.2, visible=False, zorder=8)
        self.ax_i.add_patch(self.cursor)

        info = tk.Frame(bot)
        info.pack(side="bottom", fill="x")
        tk.Label(info, textvariable=self.v_selinfo, fg="#333", anchor="w").pack(side="left", padx=4)
        tools = tk.Frame(bot)
        tools.pack(side="bottom", fill="x")
        self.b_brush = tk.Button(tools, text="Brush: OFF", bg="#fbdcb8", command=lambda: self.set_tool("brush"))
        self.b_brush.pack(side="left")
        self.b_eraser = tk.Button(tools, text="Eraser: OFF", bg="#fbdcb8", command=lambda: self.set_tool("eraser"))
        self.b_eraser.pack(side="left")
        self.b_notthis = tk.Button(tools, text="Not this: OFF", bg="#f4b6c2", command=lambda: self.set_tool("notthis"))
        self.b_notthis.pack(side="left")
        tk.Button(tools, text="Clear not-this", command=self.clear_not_this).pack(side="left")
        tk.Label(tools, text="size (image px)").pack(side="left", padx=(4, 0))
        tk.Scale(tools, from_=1, to=400, orient="horizontal", variable=self.v_brush, length=150,
                 command=lambda v: self._update_cursor_radius()).pack(side="left")
        tk.Button(tools, text="Undo stroke (Ctrl+Z)", command=self.undo_stroke).pack(side="left")
        tk.Button(tools, text="Zoom + (+)", command=lambda: self.zoom_step(0.5)).pack(side="left", padx=(10, 0))
        tk.Button(tools, text="Zoom - (-)", command=lambda: self.zoom_step(2.0)).pack(side="left")
        tk.Button(tools, text="Reset zoom (0)", command=self.reset_zoom).pack(side="left")
        self.b_big = tk.Button(tools, text="Big image (b)", command=self.toggle_big)
        self.b_big.pack(side="left", padx=(6, 0))
        tk.Label(tools, text="wheel = zoom at cursor, middle-drag or space+drag = pan", fg="#666").pack(side="left", padx=4)
        nav = tk.Frame(bot)
        nav.pack(side="bottom", fill="x")
        tk.Button(nav, text="<- Prev", command=lambda: self._step(-1)).pack(side="left")
        self.slider = tk.Scale(nav, from_=1, to=max(1, len(self.paths)), orient="horizontal", showvalue=True,
                               length=220, command=lambda v: self._goto(int(float(v)) - 1))
        self.slider.pack(side="left")
        tk.Button(nav, text="Next ->", command=lambda: self._step(1)).pack(side="left")
        self.b_view = tk.Button(nav, text="View: pca (v)", width=20, command=self.cycle_view)
        self.b_view.pack(side="left", padx=(10, 0))
        tk.Label(nav, text="DINOMask:").pack(side="left", padx=(4, 0))
        for v in ("classes", "hidden PCA"):
            tk.Radiobutton(nav, text=v, value=v, variable=self.v_dmview, command=self._redraw_image).pack(side="left")
        tk.Label(nav, text="blend").pack(side="left", padx=(8, 0))
        tk.Scale(nav, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", variable=self.v_alpha, length=110,
                 showvalue=False, command=lambda v: self._redraw_image()).pack(side="left")
        self.cv_i = FigureCanvasTkAgg(self.fig_i, master=bot)
        self.cv_i.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.lasso_i = LassoSelector(self.ax_i, self._on_lasso_image, button=MouseButton.LEFT,
                                     props={"color": "yellow", "linewidth": 1.2})
        self.cv_i.mpl_connect("button_press_event", self._img_press)
        self.cv_i.mpl_connect("button_release_event", self._img_release)
        self.cv_i.mpl_connect("motion_notify_event", self._img_motion)
        self.cv_i.mpl_connect("scroll_event", self._img_scroll)
        self.cv_i.mpl_connect("axes_leave_event", lambda e: self._hide_cursor())

    # ------------------------------------------------------------ images (display)
    def _rec(self):
        return self.recs[self.cur] if self.recs and 0 <= self.cur < len(self.recs) else None

    def _full_rgb(self, path):
        if path in self._full:
            self._full.move_to_end(path)
            return self._full[path]
        rgb = bat.load_rgb8(path)
        self._full[path] = rgb
        while len(self._full) > 2:
            self._full.popitem(last=False)
        return rgb

    def _thumb(self, path, long_side=1400):
        if path not in self._thumbs:
            rgb = self._full_rgb(path)
            H, W = rgb.shape[:2]
            s = min(1.0, long_side / max(H, W))
            self._thumbs[path] = cv2.resize(rgb, (max(1, int(W * s)), max(1, int(H * s))),
                                            interpolation=cv2.INTER_AREA) if s < 1 else rgb
        return self._thumbs[path]

    # ------------------------------------------------------------ tokens
    def _pick_cache(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(parent=self.win, title="Token cache folder", initialdir=self.v_cache.get() or None)
        if d:
            self.v_cache.set(d)

    def _cache_files(self):
        size, layer, cdir = int(self.v_size.get()), int(self.v_layer.get()), self.v_cache.get().strip()
        return [(p, bat.cache_path(cdir, p, layer, size)) for p in self.paths]

    def _try_load_cached(self):
        """Load what is already cached for the current settings (nothing is extracted)."""
        if not self.paths or not self.v_cache.get().strip():
            return
        size, layer = int(self.v_size.get()), int(self.v_layer.get())
        if all(bat.cache_is_valid(c, p, layer, size, bat.DEFAULT_MODEL) for p, c in self._cache_files()):
            self._load_tokens()

    def _start_proc(self, lines, args, kind, label):
        if self.proc is not None:
            self.v_status.set("A token job is already running (one GPU job at a time).")
            return False
        cdir = self.v_cache.get().strip()
        if not cdir:
            self.v_status.set("Set a cache folder first.")
            return False
        try:
            os.makedirs(cdir, exist_ok=True)
        except OSError as e:
            self.v_status.set(f"Cannot create the cache folder: {e}")
            return False
        fd, lst = tempfile.mkstemp(prefix="biorag_annotate_", suffix=".txt")
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        self._list_file = lst
        cmd = [self.v_python.get().strip() or default_token_python(), TOKENS_SCRIPT, "--image_list", lst,
               "--cache_dir", cdir] + list(args)
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception as e:
            self.proc = None
            self.v_status.set(f"Could not start the token extractor: {e}")
            return False
        self._proc_kind = kind
        self.failed = {}
        for b in (self.b_extract, self.b_region, self.b_dm, self.b_apply):
            b.configure(state="disabled")
        self.b_stop.configure(state="normal")
        self.v_status.set(f"{label}: 0 / {len(lines)} ...")
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        return True

    def extract(self):
        if not self.paths:
            self.v_status.set("No images: load a folder (or an image) in the editor first.")
            return
        self._start_proc(self.paths, ["--size", self.v_size.get(), "--layer", self.v_layer.get()], "tokens", "Tokens")

    def extract_region(self):
        """Tokens of the visible part of the current image only, upsampled (cubic) so the region's long side
        is the region size (at most 8x its own pixels)."""
        from tkinter import messagebox
        r = self._rec()
        if r is None or self.vbox is None:
            messagebox.showinfo("BioRAG Annotate", "Load the whole-image tokens first, then zoom in on the strip "
                                "image (mouse wheel) and press this button.", parent=self.win)
            return
        H, W = r.image_hw
        x0, x1, y0, y1 = self.vbox
        box = (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))), min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1))))
        if (box[2] - box[0]) * (box[3] - box[1]) > 0.9 * W * H:
            messagebox.showinfo("BioRAG Annotate", "Zoom in first (mouse wheel on the image): the region is "
                                "the part of the image you see.", parent=self.win)
            return
        if (r.path, box) not in self.crops:
            self.crops.append((r.path, box))
        size = int(self.v_region_size.get())
        if r.is_crop and self.v_src.get() == "region" and self.v_same_scale.get():
            # 18 Oct: a view panned / zoomed from a region is extracted at THAT region's scale (cap 8x)
            s_cur = r.work_hw[1] / float(r.box[2] - r.box[0])
            size = max(16, int(round(max(box[2] - box[0], box[3] - box[1]) * s_cur)))
        self._crop_size[(r.path, box)] = size
        self._new_crop = (r.path, box)
        lines = ["%s\t%d,%d,%d,%d\t%d" % ((p,) + tuple(b) + (self._crop_sz(p, b),)) for p, b in self.crops]
        self._start_proc(lines, ["--size", self.v_region_size.get(), "--layer", self.v_layer.get()], "region",
                         "Region tokens")

    def _crop_sz(self, path, box):
        return int(self._crop_size.get((path, tuple(box)), int(self.v_region_size.get())))

    def _reader(self, proc):
        tail = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("BRA_"):
                self.q.put(("line", line))
            elif line.strip():
                tail.append(line)
                del tail[:-15]
        proc.wait()
        self.q.put(("exit", (proc.returncode, "\n".join(tail))))

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "line":
                    self._on_line(val)
                elif kind == "exit":
                    self._on_exit(*val)
                elif kind == "pca":
                    self._on_pca(val)
                elif kind == "masks":
                    self._finish_assign(*val)
                elif kind == "rule":
                    self._start_apply(*val)
                elif kind == "error":
                    self._busy = False
                    self.v_status.set(val)
        except queue.Empty:
            pass
        try:
            self.win.after(150, self._poll)
        except Exception:
            pass

    def _on_line(self, line):
        parts = line.split("\t")
        if parts[0] == "BRA_PROGRESS" and len(parts) >= 6:
            i, n, st, sec, name = parts[1], parts[2], parts[3], parts[4], parts[5]
            msg = parts[6] if len(parts) > 6 else ""
            if st == "failed":
                self.failed[name] = msg
            self.v_status.set(f"{self._proc_kind} {i} / {n}: {name[:40]} {st} ({sec} s)")
        elif line.startswith("BRA_INFO"):
            self.v_status.set(line[9:][:120])

    def _on_exit(self, code, tail):
        kind = self._proc_kind
        self.proc, self._proc_kind = None, None
        for b in (self.b_extract, self.b_region, self.b_dm):
            b.configure(state="normal")
        self.b_stop.configure(state="disabled")
        try:
            os.remove(self._list_file)
        except Exception:
            pass
        if code not in (0, 2):
            from tkinter import messagebox
            messagebox.showwarning("BioRAG Annotate", f"The {kind} job stopped (exit code {code}).\n\n{tail[-1500:]}",
                                   parent=self.win)
        if kind == "tokens":
            self._load_tokens()
        elif kind == "region":
            self._load_region()
        elif kind == "dinomask":
            self._load_dinomask()
        elif kind == "apply":
            self._load_applied()

    def _load_tokens(self):
        size, layer = int(self.v_size.get()), int(self.v_layer.get())
        recs = []
        for p, c in self._cache_files():
            if not bat.cache_is_valid(c, p, layer, size, bat.DEFAULT_MODEL):
                self.failed.setdefault(os.path.basename(p), "no valid token cache")
                continue
            try:
                recs.append(ImageRec(p, bat.load_cache(c)))
            except Exception as e:
                self.failed[os.path.basename(p)] = str(e)
        self.recs_whole = recs
        self.params_whole = {"layer": layer, "size": size, "model": bat.DEFAULT_MODEL, "token_source": "whole image"}
        if self.v_src.get() == "whole" or not self.recs_region:
            self.v_src.set("whole")
            self._activate(recs, self.params_whole)
        if self._want_region:                    # 18 Oct: reopen on the zoomed region the user had
            self._want_region = False
            self.win.after(100, self._restore_region)

    def _load_region(self):
        size, layer, cdir = int(self.v_region_size.get()), int(self.v_layer.get()), self.v_cache.get().strip()
        recs = []
        for p, box in self.crops:
            size = self._crop_sz(p, box)                     # 18 Oct: each region keeps its own size
            c = bat.cache_path(cdir, p, layer, size, box)
            if bat.cache_is_valid(c, p, layer, size, bat.DEFAULT_MODEL):
                try:
                    recs.append(ImageRec(p, bat.load_cache(c)))
                except Exception as e:
                    self.failed[os.path.basename(c)] = str(e)
        if not recs:
            self.v_status.set("Region tokens failed: " + "; ".join(f"{k} ({v[:80]})" for k, v in self.failed.items()))
            return
        self.recs_region = recs
        self.params_region = {"layer": layer, "size": size, "model": bat.DEFAULT_MODEL, "token_source": "zoomed region",
                              "max_upsampling": bat.MAX_CROP_SCALE}
        new = getattr(self, "_new_crop", None)
        cur = next((k for k, r in enumerate(recs) if new and r.path == new[0] and r.box == tuple(new[1])), 0)
        self.v_src.set("region")
        self._refresh_regions_list()
        self._activate(recs, self.params_region, cur=cur)

    def _on_src(self):
        src = self.v_src.get()
        if src == "applied":
            if self.recs_applied:
                self._activate_applied()
            else:
                self.v_src.set("region" if self.recs_region and not self._applied_mode else "whole")
                self.v_status.set("No applied results yet: build a selection on a zoomed region, then "
                                  "'Apply selection rule'.")
            return
        if src == "region" and not self.recs_region and self._scan_crops():
            self._restore_region()               # 18 Oct: regions extracted in an earlier session
            return
        if src == "region" and not self.recs_region:
            self.v_src.set("whole")
            self.v_status.set("No zoomed-region tokens yet: zoom in on the image and press 'Extract this view (new region)'.")
            return
        if src == "whole":
            self._activate(self.recs_whole, getattr(self, "params_whole", {}))
        else:
            self._activate(self.recs_region, self.params_region)

    def _activate(self, recs, params, cur=0):
        self._applied_mode = False
        self._refresh_regions_list()
        off = 0
        for r in recs:
            r.offset = off
            off += r.n
        self.recs = recs
        self.params_tokens = dict(params)
        self.slider.configure(to=max(1, len(recs)))
        self.cur = min(cur, max(0, len(recs) - 1))
        self.vbox = None
        self.P = self.colors = self.vis = None
        self.sel = np.zeros(off, bool)
        self.pix, self._undo_strokes, self.sim, self._featn = {}, [], None, {}
        self.neg_pix = {}
        self.catidx = None
        msg = f"{len(recs)} token set(s) ({params.get('token_source', '')}), {off} patches."
        if self.failed:
            msg += f" Skipped {len(self.failed)}: " + "; ".join(f"{k} ({v[:60]})" for k, v in list(self.failed.items())[:3])
        self.v_status.set(msg)
        if self.v_colour.get() == "mask category":
            self._load_categories(self._cat_path(), quiet=True)
        try:
            self.slider.set(self.cur + 1)
        except Exception:
            pass
        if recs:
            self.compute_pca()
        else:
            self._redraw_scatter(full=True)

    # ------------------------------------------------------------ PCA
    def _on_variant(self):
        self._featn = {}
        self.compute_pca()

    def compute_pca(self):
        if not self.recs or getattr(self, "_applied_mode", False):
            return
        self._pca_job += 1
        job = self._pca_job
        variant, fg, flip = self.v_variant.get(), self.v_fg.get(), self.v_flip.get()
        self.v_status.set("Computing PCA ...")
        recs = list(self.recs)

        def work():
            try:
                # 18 Oct: same maths as before, now in the pure pca_fit_tokens (its parameters make the rules)
                P, rgb, vis, prm = pca_fit_tokens([(r.T, r.grid) for r in recs], variant, fg, flip)
                self.q.put(("pca", (job, P, rgb, vis, prm)))
            except Exception as e:
                self.q.put(("error", f"PCA failed: {e}"))
        threading.Thread(target=work, daemon=True).start()

    def _on_pca(self, val):
        job, P, rgb, vis = val[:4]
        if job != self._pca_job:
            return
        self._pca_prm = val[4] if len(val) > 4 else None
        self.P, self.pca_rgb, self.vis = P, rgb, vis
        if self.sel is None or len(self.sel) != len(P):
            self.sel = np.zeros(len(P), bool)
        self.sel &= vis
        idx = np.flatnonzero(vis)
        rng = np.random.default_rng(1)
        self.show_idx = idx if len(idx) <= MAX_SHOW else np.sort(rng.choice(idx, MAX_SHOW, replace=False))
        self._apply_colours()
        self.v_status.set(f"{len(self.recs)} token set(s), {int(vis.sum())} of {len(vis)} patches shown"
                          f"{' (foreground)' if self.v_fg.get() else ''}; PCA on {self.v_variant.get()} tokens.")
        self._redraw_scatter(full=True)
        self._redraw_image()
        if any(d["on"].get() for d in getattr(self, "hw", [])):
            self._hue_base = None
            self._hue_changed()                  # 18 Oct: colours changed with the PCA
        self._save_settings()

    # ------------------------------------------------------------ colours / categories
    def _cat_path(self):
        return getattr(self, "_cat_file", None) or self.host.folder_coco_path()

    def pick_category_file(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(parent=self.win, title="COCO masks to colour by (json / rlejson)",
                                       filetypes=[("COCO", "*.json *.rlejson"), ("All", "*.*")])
        if p:
            self._cat_file = p
            self._cat_per, self.catidx, self._ann_cache = None, None, {}
            self.v_colour.set("mask category")
            self._on_colour_mode()

    def _on_colour_mode(self):
        if self.v_colour.get() == "mask category" and self.catidx is None:
            self._load_categories(self._cat_path())
        self._apply_colours()
        self._redraw_scatter(full=True)
        self._redraw_image()

    def _read_categories(self, path):
        """Annotations per image + category list from a COCO file (kept for the annotations view)."""
        with open(path) as fh:
            coco = json.load(fh)
        names_by_id = {c["id"]: c["name"] for c in coco.get("categories", [])}
        per = coco_annotations_by_image(coco, self.paths)
        used = sorted({names_by_id.get(a.get("category_id")) for v in per.values() for a in v} - {None})
        self._cat_per = per
        self.cat_names = used
        self.cat_cols = category_palette(len(used))
        self._cid2idx = {cid: used.index(nm) for cid, nm in names_by_id.items() if nm in used}
        self._ann_cache = {}
        self.v_catsrc.set(os.path.basename(path)[:40])

    def _load_categories(self, path, quiet=False):
        from tkinter import messagebox
        if not path or not os.path.isfile(path):
            if not quiet:
                messagebox.showinfo("BioRAG Annotate", "No mask file: load the folder annotations in the editor, "
                                    "or pick a COCO / rlejson file with 'Mask file...'.", parent=self.win)
            self.v_colour.set("PCA-RGB")
            return False
        try:
            if self._cat_per is None:
                self._read_categories(path)
            N = sum(r.n for r in self.recs)
            ci = np.full(N, -1, np.int32)
            for r in self.recs:
                ci[r.offset:r.offset + r.n] = category_grid(self._cat_per.get(r.name, []), self._cid2idx,
                                                            r.image_hw, r.grid, r.box).ravel()
            self.catidx = ci
            return True
        except Exception as e:
            messagebox.showerror("BioRAG Annotate", f"Could not read masks from\n{path}\n\n{e}", parent=self.win)
            self.v_colour.set("PCA-RGB")
            return False

    def _apply_colours(self):
        if self.P is None:
            return
        if self.v_colour.get() == "mask category" and self.catidx is not None:
            pal = np.array(list(self.cat_cols) + [NONE_GREY], np.float32)
            self.colors = pal[np.where(self.catidx < 0, len(self.cat_cols), self.catidx)]
        elif self._applied_mode:
            self.colors = self._applied_colors
        else:
            self.colors = self.pca_rgb
        self._build_legend()

    def _build_legend(self):
        tk = self.tk
        for w in self.legend.winfo_children():
            w.destroy()
        if self.v_colour.get() != "mask category" or self.catidx is None:
            tk.Label(self.legend, text="(colour by 'mask category'\nto list categories)", fg="#777",
                     justify="left").pack(anchor="w")
            return
        vis = self.vis if self.vis is not None else np.ones(len(self.catidx), bool)
        rows = [(nm, self.cat_cols[k], int(((self.catidx == k) & vis).sum()), k) for k, nm in enumerate(self.cat_names)]
        rows.append(("none", NONE_GREY, int(((self.catidx < 0) & vis).sum()), -1))
        for nm, col, n, k in rows:
            hexc = "#%02x%02x%02x" % tuple(int(255 * c) for c in col)
            f = tk.Frame(self.legend)
            f.pack(anchor="w", fill="x")
            tk.Label(f, bg=hexc, width=2).pack(side="left", padx=(0, 4))
            tk.Button(f, text=f"{nm}  ({n})", anchor="w", relief="flat",
                      command=lambda k=k: self.select_category(k)).pack(side="left", fill="x")

    def select_category(self, k):
        if self.catidx is None or self.vis is None:
            return
        self._apply_selection((self.catidx == k) & self.vis)

    # ------------------------------------------------------------ selection
    def _drop_stale_pix(self, before):
        """Brush masks are dropped for images whose patch selection was changed by another tool."""
        gone = []
        for i in list(self.pix):
            r = self.recs[i]
            sl = slice(r.offset, r.offset + r.n)
            if not np.array_equal(before[sl], self.sel[sl]):
                del self.pix[i]
                gone.append(i)
        if gone:
            self._undo_strokes = [u for u in self._undo_strokes if u[0] not in gone]

    # ------------------------------------------------------------ colour windows (vertical rainbows), 18 Oct
    # Window 1 selects a colour range anywhere; window 2 (optional) adds a second colour range, but only tokens
    # within N patches of window 1's tokens (0 = anywhere), e.g. the darker edges of setae.
    def _build_hue_column(self, body):
        tk = self.tk
        self.hw = []
        outer = tk.Frame(body)
        outer.pack(side="left", fill="y", padx=(2, 4))
        for k, (title, c0) in enumerate((("Colour 1", 220.0), ("Colour 2\nnear 1", 60.0))):
            col = tk.Frame(outer, width=92)
            col.pack(side="left", fill="y", padx=(0, 3))
            d = {"c": tk.DoubleVar(value=c0), "w": tk.IntVar(value=30), "lo": tk.IntVar(value=0),
                 "hi": tk.IntVar(value=100), "on": tk.BooleanVar(value=False)}
            tk.Label(col, text=title, font=("Arial", 8, "bold")).pack()
            tk.Checkbutton(col, text="on", variable=d["on"], command=self._hue_changed).pack()
            cv = tk.Canvas(col, width=40, height=360, highlightthickness=1, highlightbackground="#888")
            cv.pack(pady=2)
            for y in range(360):                  # hue 0 (top) -> 359 (bottom), the PCA-RGB hues
                r, g, b = cv2.cvtColor(np.float32([[[y, 1.0, 1.0]]]), cv2.COLOR_HSV2RGB)[0, 0]
                cv.create_line(0, y, 40, y, fill="#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)))
            cv.bind("<Button-1>", lambda e, k=k: self._hue_click(k, e))
            cv.bind("<B1-Motion>", lambda e, k=k: self._hue_click(k, e))
            cv.bind("<Button-4>", lambda e, k=k: self._hue_nudge(k, -1))
            cv.bind("<Button-5>", lambda e, k=k: self._hue_nudge(k, 1))
            cv.bind("<MouseWheel>", lambda e, k=k: self._hue_nudge(k, -1 if e.delta > 0 else 1))
            d["cv"] = cv
            tk.Label(col, text="width (deg)").pack()
            tk.Spinbox(col, from_=1, to=180, increment=1, width=5, textvariable=d["w"], command=self._hue_changed).pack()
            tk.Label(col, text="brightness %").pack(pady=(4, 0))
            fr = tk.Frame(col)
            fr.pack()
            tk.Spinbox(fr, from_=0, to=100, increment=5, width=3, textvariable=d["lo"], command=self._hue_changed).pack(side="left")
            tk.Spinbox(fr, from_=0, to=100, increment=5, width=3, textvariable=d["hi"], command=self._hue_changed).pack(side="left")
            if k == 1:
                d["near"] = tk.IntVar(value=1)
                tk.Label(col, text="within N patches\nof colour 1\n(0 = anywhere)", justify="center").pack(pady=(4, 0))
                tk.Spinbox(col, from_=0, to=20, increment=1, width=4, textvariable=d["near"], command=self._hue_changed).pack()
            for v in [d["w"], d["lo"], d["hi"]] + ([d["near"]] if k == 1 else []):
                v.trace_add("write", lambda *a: self._hue_changed())
            self.hw.append(d)
        self.v_hue_scope = tk.StringVar(value="all")
        self.v_hue_info = tk.StringVar(value="")
        tk.Radiobutton(outer.winfo_children()[0], text="all images", value="all", variable=self.v_hue_scope,
                       command=self._hue_changed).pack(anchor="w", pady=(8, 0))
        tk.Radiobutton(outer.winfo_children()[0], text="this image", value="image", variable=self.v_hue_scope,
                       command=self._hue_changed).pack(anchor="w")
        tk.Label(outer.winfo_children()[0], textvariable=self.v_hue_info, fg="#555", wraplength=88,
                 justify="left").pack(pady=(4, 0))
        self._hue_base = None
        for k in range(len(self.hw)):
            self._draw_hue_window(k)

    def _draw_hue_window(self, k):
        tk = self.tk
        d = self.hw[k]
        cv = d["cv"]
        cv.delete("win")
        try:
            c, w = float(d["c"].get()), max(1, int(d["w"].get()))
        except (tk.TclError, ValueError):
            return
        a, b = (c - w / 2) % 360, (c + w / 2) % 360
        col = "white" if d["on"].get() else "#777"
        if w >= 360:
            cv.create_rectangle(1, 0, 39, 360, outline=col, width=2, tags="win")
        elif a <= b:
            cv.create_rectangle(1, a, 39, b, outline=col, width=2, tags="win")
        else:                                      # wraps around red
            cv.create_rectangle(1, a, 39, 360, outline=col, width=2, tags="win")
            cv.create_rectangle(1, 0, 39, b, outline=col, width=2, tags="win")
        cv.create_line(0, c, 40, c, fill="black", width=1, tags="win")

    def _hue_click(self, k, ev):
        self.hw[k]["c"].set(float(min(359, max(0, ev.y))))
        if not self.hw[k]["on"].get():
            self.hw[k]["on"].set(True)
        self._hue_changed()

    def _hue_nudge(self, k, d):
        self.hw[k]["c"].set((float(self.hw[k]["c"].get()) + d * 2) % 360)
        self._hue_changed()

    def _hue_mask(self, k, hue, val):
        tk = self.tk
        d = self.hw[k]
        try:
            w, lo, hi = int(d["w"].get()), int(d["lo"].get()), int(d["hi"].get())
        except (tk.TclError, ValueError):
            return None
        return hue_window_mask(hue, val, d["c"].get(), w, lo / 100.0, hi / 100.0)

    def _hue_changed(self):
        tk = self.tk
        for k in range(len(self.hw)):
            self._draw_hue_window(k)
        on1, on2 = self.hw[0]["on"].get(), self.hw[1]["on"].get()
        if not (on1 or on2):
            self._hue_base = None
            return
        if (self.pca_rgb is None or self.sel is None or self.vis is None
                or not (len(self.pca_rgb) == len(self.sel) == len(self.vis))):
            self.v_hue_info.set("load tokens first")   # also while a new PCA is being computed
            return
        if self._hue_base is None or len(self._hue_base) != len(self.sel):
            self._hue_base = self.sel.copy()    # what was selected before the windows moved (for add/subtract)
        wins = self._window_params()               # 18 Oct: shared with the selection rules (same maths)
        if wins is None:
            return
        new, m1, m2 = colour_window_selection(self.pca_rgb, [(r.offset, r.grid) for r in self.recs], wins)
        if self.v_hue_scope.get() == "image":
            r = self._rec()
            keep = np.zeros(len(new), bool)
            if r is not None:
                keep[r.offset:r.offset + r.n] = True
            new &= keep
        before = self.sel.copy()
        self.sim = None
        self.sel = self._combine(self._hue_base, new)
        self._drop_stale_pix(before)
        self._sel_origin = "colour windows"
        self.v_hue_info.set(f"{int(self.sel.sum())} patches" + (f"\n(+{int((m2 & ~m1).sum())} from colour 2)" if on2 else ""))
        self._redraw_scatter()
        self._redraw_image()

    def _window_params(self):
        """The two colour windows as plain values (None while a spinbox holds no number)."""
        tk = self.tk
        out = []
        try:
            for k, d in enumerate(self.hw):
                w = {"on": bool(d["on"].get()), "c": float(d["c"].get()), "w": int(d["w"].get()),
                     "lo": int(d["lo"].get()), "hi": int(d["hi"].get())}
                if k == 1:
                    w["near"] = int(d["near"].get())
                out.append(w)
        except (tk.TclError, ValueError):
            return None
        return out

    def _combine(self, base, new):
        mode = self.v_lasso.get()
        if mode == "add":
            out = base | new
        elif mode == "subtract":
            out = base & ~new
        else:
            out = new.copy()
        out = out & self.vis if self.vis is not None else out
        if self.neg_pix:                          # 19 Oct: never select "not this" patches
            out = drop_negatives(out, self._neg_vec(len(out)))
        return out

    def _neg_vec(self, n=None):
        """Explicit-negative patches of all images (>= 50% of a patch's pixels painted 'not this')."""
        n = (len(self.sel) if self.sel is not None else 0) if n is None else n
        out = np.zeros(n, bool)
        for i, nm in self.neg_pix.items():
            if nm is not None and i < len(self.recs) and nm.any():
                r = self.recs[i]
                if r.offset + r.n <= n:
                    out[r.offset:r.offset + r.n] = negative_patches(nm, r.grid).ravel()
        return out

    def _drop_neg_patches(self, i):
        nm = self.neg_pix.get(i)
        if nm is not None and nm.any() and self.sel is not None:
            r = self.recs[i]
            sl = slice(r.offset, r.offset + r.n)
            self.sel[sl] = drop_negatives(self.sel[sl], negative_patches(nm, r.grid).ravel())

    def clear_not_this(self):
        """Remove the not-this paint of this image (Ctrl+Z brings it back; the selection is not changed)."""
        r, i = self._rec(), self.cur
        nm = self.neg_pix.get(i)
        if r is None or nm is None or self.sel is None:
            self.win.bell()
            return
        pm = self.pix.get(i)
        self._undo_strokes.append((i, None if pm is None else pm.copy(), self.sel[r.offset:r.offset + r.n].copy(), nm))
        del self._undo_strokes[:-30]
        del self.neg_pix[i]
        self._redraw_scatter()
        self._redraw_image()

    def _apply_selection(self, new):
        self._sel_origin = "lasso"
        self._hue_base = None                  # 18 Oct: the colour window restarts from the new selection
        if self.sel is None or len(self.sel) != len(new):
            self.sel = np.zeros(len(new), bool)
        before = self.sel.copy()
        self.sim = None
        self.sel = self._combine(before, new)
        self._drop_stale_pix(before)
        self._redraw_scatter()
        self._redraw_image()

    def clear_selection(self):
        if self.sel is not None:
            self.sel[:] = False
        self.pix, self._undo_strokes, self.sim = {}, [], None
        self._redraw_scatter()
        self._redraw_image()

    def _xy(self):
        a, b = PC_PAIRS[self.v_pcs.get()]
        return self.P[:, [a, b]]

    def _on_lasso_scatter(self, verts):
        if self.P is None or len(verts) < 3 or self._applied_mode:
            return
        from matplotlib.path import Path
        idx = np.flatnonzero(self.vis)
        inside = Path(verts).contains_points(self._xy()[idx])
        new = np.zeros(len(self.P), bool)
        new[idx[inside]] = True
        self._apply_selection(new)

    def _on_lasso_image(self, verts):
        r = self._rec()
        if r is None or self.P is None or len(verts) < 3 or self.v_tool.get() or self._pan is not None:
            return
        v = np.asarray(verts)
        bw = r.box[2] - r.box[0]
        if np.ptp(v[:, 0]) + np.ptp(v[:, 1]) < 2 * bw / r.grid[1]:
            return                                           # a click; handled on release
        from matplotlib.path import Path
        cx, cy = patch_centres(r.grid, r.box)
        X, Y = np.meshgrid(cx, cy)
        inside = Path(verts).contains_points(np.c_[X.ravel(), Y.ravel()])
        new = np.zeros(len(self.P), bool)
        new[r.offset + np.flatnonzero(inside)] = True
        self._apply_selection(new)

    # ------------------------------------------------------------ similarity
    def _features_norm(self, i):
        if i in self._featn:
            return self._featn[i]
        r = self.recs[i]
        F = normalize_rows(token_features(r.T, r.grid, self.v_variant.get()))
        used = sum(v.nbytes for v in self._featn.values())
        if used + F.size * 2 < MAX_FEAT_CACHE_BYTES:
            self._featn[i] = F.astype(np.float16)
        return F

    def select_similar(self, i, row, col, scope):
        """Cosine similarity (same token variant as the plot) to the clicked patch, or to the mean of the
        current selection if the clicked patch is selected; threshold = the 'Similar >=' slider (live)."""
        if self.P is None:
            return
        r = self.recs[i]
        gi = r.offset + row * r.grid[1] + col
        if self.sel is not None and self.sel[gi]:
            acc, n = None, 0
            for k, q in enumerate(self.recs):
                s = self.sel[q.offset:q.offset + q.n]
                if s.any():
                    v = self._features_norm(k)[s].astype(np.float32).sum(0)
                    acc = v if acc is None else acc + v
                    n += int(s.sum())
            qv, what = acc / max(n, 1), f"mean of {n} selected patches"
        else:
            qv, what = self._features_norm(i)[row * r.grid[1] + col].astype(np.float32), "the clicked patch"
        self._sim_query = np.asarray(qv, np.float32)
        sims = np.full(len(self.P), -2.0, np.float32)
        ks = [i] if scope == "image" else range(len(self.recs))
        for k in ks:
            q = self.recs[k]
            sims[q.offset:q.offset + q.n] = cosine_to(self._features_norm(k), qv)
        before = self.sel.copy()
        self.sim = sims
        self._sim_base = before
        self._sim_label = f"similar to {what} ({'this image' if scope == 'image' else 'all images'})"
        self._update_sim()

    def _update_sim(self):
        if self.sim is None or self._sim_base is None:
            return
        before = self.sel.copy()
        new = self.sim >= float(self.v_sim.get())
        self.sel = self._combine(self._sim_base, new)
        self._sel_origin = "similar"
        self._drop_stale_pix(before)
        self._redraw_scatter()
        self._redraw_image()

    # ------------------------------------------------------------ brush / eraser on the strip
    def set_tool(self, tool):
        self.v_tool.set("" if self.v_tool.get() == tool else tool)
        t = self.v_tool.get()
        self.b_brush.configure(text="Brush: ON" if t == "brush" else "Brush: OFF",
                               relief="sunken" if t == "brush" else "raised")
        self.b_eraser.configure(text="Eraser: ON" if t == "eraser" else "Eraser: OFF",
                                relief="sunken" if t == "eraser" else "raised")
        self.b_notthis.configure(text="Not this: ON" if t == "notthis" else "Not this: OFF",
                                 relief="sunken" if t == "notthis" else "raised")
        self.lasso_i.set_active(not t and not self._space)
        if not t:
            self._hide_cursor()

    def _update_cursor_radius(self):
        self.cursor.set_radius(max(1, int(self.v_brush.get())))
        self.cv_i.draw_idle()

    def _hide_cursor(self):
        if self.cursor.get_visible():
            self.cursor.set_visible(False)
            self.cv_i.draw_idle()

    def _ensure_pix(self, i):
        if i not in self.pix:
            r = self.recs[i]
            s = self.sel[r.offset:r.offset + r.n].reshape(r.grid)
            self.pix[i] = selection_to_work_mask(s, r.grey(), min_frac=0)   # 18 Oct: keep every selected part
            nm = self.neg_pix.get(i)
            if nm is not None:
                self.pix[i][nm > 0] = 0                  # 19 Oct: not this
        return self.pix[i]

    def _to_work(self, r, x, y):
        x0, y0, x1, y1 = r.box
        nh, nw = r.work_hw
        return (x - x0) * nw / (x1 - x0), (y - y0) * nh / (y1 - y0)

    def _paint(self, x0, y0, x1, y1):
        if self.v_tool.get() == "notthis":            # 19 Oct
            self._paint_neg(x0, y0, x1, y1)
            return
        st = self._stroke
        r = self.recs[st["i"]]
        m = self.pix[st["i"]]
        a = self._to_work(r, x0, y0)
        b = self._to_work(r, x1, y1)
        rad = max(1, int(round(self.v_brush.get() * r.work_hw[1] / (r.box[2] - r.box[0]))))
        val = 1 if self.v_tool.get() == "brush" else 0
        pa, pb = (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))
        cv2.line(m, pa, pb, val, thickness=2 * rad + 1)
        cv2.circle(m, pa, rad, val, -1)
        cv2.circle(m, pb, rad, val, -1)
        if val == 1 and self.neg_pix.get(st["i"]) is not None:
            paint_stroke(self.neg_pix[st["i"]], pa, pb, rad, 0)        # 19 Oct: last stroke wins

    def _paint_neg(self, x0, y0, x1, y1):
        st = self._stroke
        r = self.recs[st["i"]]
        a = self._to_work(r, x0, y0)
        b = self._to_work(r, x1, y1)
        rad = max(1, int(round(self.v_brush.get() * r.work_hw[1] / (r.box[2] - r.box[0]))))
        pa, pb = (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))
        paint_not_this(self.neg_pix[st["i"]], self.pix.get(st["i"]), pa, pb, rad)

    def _stroke_begin(self, x, y):
        r = self._rec()
        if r is None or self.sel is None or self.P is None:
            return
        i = self.cur
        prev = self.pix.get(i)
        self._stroke = {"i": i, "prev": None if prev is None else prev.copy(),
                        "prev_sel": self.sel[r.offset:r.offset + r.n].copy(), "last": (x, y)}
        pn = self.neg_pix.get(i)
        self._stroke["prev_neg"] = None if pn is None else pn.copy()
        if self.v_tool.get() == "notthis":            # 19 Oct: pixel selection only where there is one
            if i not in self.neg_pix:
                self.neg_pix[i] = np.zeros(r.work_hw, np.uint8)
            if i in self.pix or self.sel[r.offset:r.offset + r.n].any():
                self._ensure_pix(i)
        else:
            self._ensure_pix(i)
        self._paint(x, y, x, y)
        self._redraw_image(fast=True)

    def _stroke_to(self, x, y):
        st = self._stroke
        lx, ly = st["last"]
        self._paint(lx, ly, x, y)
        st["last"] = (x, y)
        self._redraw_image(fast=True)

    def _sync_sel_from_pix(self, i):
        r = self.recs[i]
        s = pix_to_patch_sel(self.pix[i], r.grid).ravel()
        if self.vis is not None:
            s &= self.vis[r.offset:r.offset + r.n]
        self.sel[r.offset:r.offset + r.n] = s

    def _stroke_end(self):
        st, self._stroke = self._stroke, None
        if st is None:
            return
        self._undo_strokes.append((st["i"], st["prev"], st["prev_sel"], st.get("prev_neg", _KEEP_NEG)))
        del self._undo_strokes[:-30]
        if self.v_tool.get() != "notthis":         # 19 Oct: not this keeps how the selection was made
            self.sim = None
            self._sel_origin = "brush"
        if st["i"] in self.pix:
            self._sync_sel_from_pix(st["i"])
        self._drop_neg_patches(st["i"])
        self._redraw_scatter()
        self._redraw_image()

    def _restore(self, i, prev, prev_sel, prev_neg=_KEEP_NEG):
        r = self.recs[i]
        if prev_neg is not _KEEP_NEG:                  # 19 Oct: not-this paint
            if prev_neg is None:
                self.neg_pix.pop(i, None)
            else:
                self.neg_pix[i] = prev_neg
        if prev is None:
            self.pix.pop(i, None)
        else:
            self.pix[i] = prev
        self.sel[r.offset:r.offset + r.n] = prev_sel
        self._redraw_scatter()
        self._redraw_image()

    def _cancel_stroke(self):
        st, self._stroke = self._stroke, None
        if st is not None:
            self._restore(st["i"], st["prev"], st["prev_sel"], st.get("prev_neg", _KEEP_NEG))

    def undo_stroke(self):
        if self._stroke is not None:
            self._cancel_stroke()
        elif self._undo_strokes:
            self._restore(*self._undo_strokes.pop())
        else:
            self.win.bell()

    # ------------------------------------------------------------ mouse on the strip
    def _space_key(self, down):
        self._space = down
        try:
            self.lasso_i.set_active(not down and not self.v_tool.get())
        except Exception:
            pass

    def _img_press(self, ev):
        if ev.inaxes is not self.ax_i:
            return
        if ev.button == 3:
            self._image_menu(ev)
            return
        if ev.button == 2 or (ev.button == 1 and self._space):
            if self.vbox is not None:
                self._pan = (ev.x, ev.y, self.vbox)
            return
        if ev.button == 1 and self.v_tool.get() and ev.xdata is not None:
            self._stroke_begin(ev.xdata, ev.ydata)
            return
        self._press = (ev.x, ev.y) if ev.button == 1 else None
        self._moved = False

    def _img_motion(self, ev):
        if self._pan is not None:
            px, py, vb = self._pan
            bb = self.ax_i.bbox
            dx = -(ev.x - px) * (vb[1] - vb[0]) / max(bb.width, 1)
            dy = (ev.y - py) * (vb[3] - vb[2]) / max(bb.height, 1)
            r = self._rec()
            self.vbox = pan_box(vb, dx, dy, (r.image_hw[1], r.image_hw[0]))
            self._apply_view()
            return
        if self.v_tool.get():
            if ev.inaxes is self.ax_i and ev.xdata is not None:
                self.cursor.center = (ev.xdata, ev.ydata)
                self.cursor.set_radius(max(1, int(self.v_brush.get())))
                self.cursor.set_edgecolor({"brush": "yellow", "notthis": "magenta"}.get(self.v_tool.get(), "red"))
                self.cursor.set_visible(True)
                if self._stroke is not None:
                    self._stroke_to(ev.xdata, ev.ydata)
                else:
                    self.cv_i.draw_idle()
            return
        if self._press is not None and abs(ev.x - self._press[0]) + abs(ev.y - self._press[1]) > 4:
            self._moved = True                                # a lasso, not a click

    def _img_release(self, ev):
        if self._pan is not None:
            self._pan = None
            self._schedule_lod()
            return
        if self._stroke is not None:
            self._stroke_end()
            return
        r = self._rec()
        pressed, self._press = self._press, None
        if ev.button != 1 or pressed is None or self._moved or r is None or self.P is None or ev.xdata is None:
            return
        if abs(ev.x - pressed[0]) + abs(ev.y - pressed[1]) > 4:
            return
        hit = point_to_patch(ev.xdata, ev.ydata, r.grid, r.box)
        if hit is None:
            return
        new = np.zeros(len(self.P), bool)
        new[r.offset + hit[0] * r.grid[1] + hit[1]] = True
        self._apply_selection(new)

    def _img_scroll(self, ev):
        r = self._rec()
        if r is None or ev.xdata is None or self.vbox is None:
            return
        f = 0.67 ** (ev.step if ev.step else (1 if ev.button == "up" else -1))   # 18 Oct: faster (was 0.8)
        self.vbox = zoom_box(self.vbox, ev.xdata, ev.ydata, f, (r.image_hw[1], r.image_hw[0]))
        self._apply_view()

    def _initial_view(self, r):
        H, W = r.image_hw
        if not r.is_crop:
            return (0.0, float(W), 0.0, float(H))
        x0, y0, x1, y1 = r.box
        mx, my = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
        return pan_box((x0 - mx, x1 + mx, y0 - my, y1 + my), 0, 0, (W, H))

    def zoom_step(self, factor):
        """18 Oct: zoom about the centre of the view (buttons / + - keys); factor < 1 = in."""
        r = self._rec()
        if r is None or self.vbox is None:
            self.v_status.set("Load tokens first (Extract / load tokens), then zoom.")
            return
        x0, x1, y0, y1 = self.vbox
        self.vbox = zoom_box(self.vbox, (x0 + x1) / 2, (y0 + y1) / 2, factor, (r.image_hw[1], r.image_hw[0]))
        self._apply_view()

    def toggle_big(self):
        """18 Oct: give the image strip most of the height (or back to the plot)."""
        tk = self.tk
        try:
            h = self._pw.winfo_height()
            big = getattr(self, "_big", False)
            self._pw.sash_place(0, 0, int(h * (0.5 if big else 0.18)))
            self._big = not big
            self.b_big.configure(relief="raised" if big else "sunken")
        except tk.TclError:
            pass

    def reset_zoom(self):
        r = self._rec()
        if r is None:
            return
        self.vbox = self._initial_view(r)
        self._apply_view()

    def _apply_view(self):
        x0, x1, y0, y1 = self.vbox
        self.ax_i.set_xlim(x0, x1)
        self.ax_i.set_ylim(y1, y0)
        self.cv_i.draw_idle()
        self._schedule_lod()

    def _schedule_lod(self):
        if self._lod_job is not None:
            try:
                self.win.after_cancel(self._lod_job)
            except Exception:
                pass
        self._lod_job = self.win.after(150, self._refresh_photo)

    def _refresh_photo(self):
        """Sharp photo when zoomed in: the visible part of the full-resolution image (plus a margin)."""
        self._lod_job = None
        r = self._rec()
        if r is None or self.vbox is None:
            return
        H, W = r.image_hw
        x0, x1, y0, y1 = self.vbox
        if (x1 - x0) * (y1 - y0) > 0.6 * W * H:
            th = self._thumb(r.path)
            self.im_photo.set_data(th)
            self.im_photo.set_extent((0, W, H, 0))
        else:
            mx, my = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
            cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
            cx1, cy1 = min(W, int(np.ceil(x1 + mx))), min(H, int(np.ceil(y1 + my)))
            crop = self._full_rgb(r.path)[cy0:cy1, cx0:cx1]
            s = min(1.0, 1600.0 / max(crop.shape[:2]))
            if s < 1:
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                                  interpolation=cv2.INTER_AREA)
            self.im_photo.set_data(crop)
            self.im_photo.set_extent((cx0, cx1, cy1, cy0))
        self.cv_i.draw_idle()

    def _on_press_scatter(self, ev):
        if ev.button == 3:
            self._context_menu()

    # ------------------------------------------------------------ drawing
    def _redraw_scatter(self, full=False):
        if self._applied_mode:                 # 18 Oct: applied results carry no tokens -> no plot
            self.sc_all.set_offsets(np.zeros((0, 2)))
            self.sc_sel.set_offsets(np.zeros((0, 2)))
            self.sc_neg.set_offsets(np.zeros((0, 2)))
            self.ax_s.set_xlabel("applied (fine) results: no token plot - edit / assign on the image below",
                                 color="#cccccc", fontsize=9)
            self.ax_s.set_ylabel("")
            self.cv_s.draw_idle()
            self._update_selinfo()
            return
        if self.P is None:
            self.sc_all.set_offsets(np.zeros((0, 2)))
            self.sc_sel.set_offsets(np.zeros((0, 2)))
            self.sc_neg.set_offsets(np.zeros((0, 2)))
            self.cv_s.draw_idle()
            return
        xy = self._xy()
        if full:
            si = self.show_idx
            self.sc_all.set_offsets(xy[si])
            self.sc_all.set_facecolors(self.colors[si])
            v = xy[self.vis] if self.vis.any() else xy
            lo, hi = v.min(0), v.max(0)
            pad = (hi - lo) * 0.04 + 1e-6
            self.ax_s.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            self.ax_s.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            a, b = PC_PAIRS[self.v_pcs.get()]
            self.ax_s.set_xlabel(f"PC{a + 1}", color="#cccccc", fontsize=8)
            self.ax_s.set_ylabel(f"PC{b + 1}", color="#cccccc", fontsize=8)
        sidx = np.flatnonzero(self.sel) if self.sel is not None else np.zeros(0, int)
        if len(sidx) > MAX_HILITE:
            sidx = np.sort(np.random.default_rng(2).choice(sidx, MAX_HILITE, replace=False))
        if len(sidx):
            self.sc_sel.set_offsets(xy[sidx])
            self.sc_sel.set_facecolors(self.colors[sidx])
        else:
            self.sc_sel.set_offsets(np.zeros((0, 2)))
        nidx = np.flatnonzero(self._neg_vec(len(xy))) if self.neg_pix else np.zeros(0, int)   # 19 Oct
        if len(nidx) > MAX_HILITE:
            nidx = np.sort(np.random.default_rng(3).choice(nidx, MAX_HILITE, replace=False))
        self.sc_neg.set_offsets(xy[nidx] if len(nidx) else np.zeros((0, 2)))
        self.cv_s.draw_idle()
        self._update_selinfo()

    def _update_selinfo(self):
        try:                                   # 18 Oct: rule can be applied from a zoomed region with a selection
            ok = (self.v_src.get() == "region" and not self._applied_mode and self.proc is None
                  and self.sel is not None and bool(self.sel.any()) and self._pca_prm is not None)
            self.b_apply.configure(state="normal" if ok else "disabled")
        except Exception:
            pass
        r = self._rec()
        if r is None or self.sel is None or not len(self.sel):
            self.v_selinfo.set("")
            return
        here = int(self.sel[r.offset:r.offset + r.n].sum())
        per = [int(self.sel[q.offset:q.offset + q.n].sum()) for q in self.recs]
        n_img = sum(1 for v in per if v)
        txt = (f"{self.cur + 1}/{len(self.recs)} {r.label[:60]}   selected: {here} patches here, "
               f"{int(self.sel.sum())} on {n_img} image(s)")
        if self.pix.get(self.cur) is not None:
            txt += "  (brush-edited)"
        nm = self.neg_pix.get(self.cur)
        if nm is not None and nm.any():
            txt += f"  not this: {int(negative_patches(nm, r.grid).sum())} patches"
        if self.sim is not None:
            txt += f"   | {self._sim_label} >= {float(self.v_sim.get()):.2f}"
            if len(per) > 1:
                txt += "  per image: " + ", ".join(str(v) for v in per[:12]) + (" ..." if len(per) > 12 else "")
        self.v_selinfo.set(txt)

    def cycle_view(self):
        v = VIEWS[(VIEWS.index(self.v_view.get()) + 1) % len(VIEWS)]
        self.v_view.set(v)
        self.b_view.configure(text=f"View: {v} (v)")
        if v == "annotations" and self._cat_per is None:
            p = self._cat_path()
            if p and os.path.isfile(p):
                try:
                    self._read_categories(p)
                except Exception as e:
                    self.v_status.set(f"Cannot read annotations: {e}")
            else:
                self.v_status.set("Annotations view: load the folder annotations or pick a 'Mask file...'.")
        if v == "dinomask" and not self.dm:
            self.v_status.set("DINOMask view: pick a trained DINOMask run ('DINOMask run...').")
        self._redraw_image()

    def _ann_layer(self, r, alpha):
        if self._cat_per is None:
            return None
        if r.path not in self._ann_cache:
            th = self._thumb(r.path)
            tht, tw = th.shape[:2]
            H, W = r.image_hw
            items = []
            for a in self._cat_per.get(r.name, []):
                ci = self._cid2idx.get(a.get("category_id"))
                seg = a.get("segmentation_rle") or a.get("segmentation")
                if ci is None or not seg:
                    continue
                m = segmentation_to_mask(seg, (H, W))
                area = int(m.sum())
                if area:
                    items.append((area, ci, cv2.resize(m, (tw, tht), interpolation=cv2.INTER_NEAREST)))
            lab = np.full((tht, tw), -1, np.int32)
            outl = np.zeros((tht, tw), np.uint8)
            for _, ci, mt in sorted(items, key=lambda t: -t[0]):
                lab[mt > 0] = ci
                cs, _ = cv2.findContours(mt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(outl, cs, -1, int(ci) + 1, thickness=2)
            self._ann_cache[r.path] = (lab, outl)
        lab, outl = self._ann_cache[r.path]
        pal = np.array(self.cat_cols, np.float32).reshape(-1, 3)
        rgba = np.zeros(lab.shape + (4,), np.float32)
        m = lab >= 0
        if m.any():
            rgba[m, :3] = pal[lab[m]]
            rgba[m, 3] = alpha
        o = outl > 0
        if o.any():
            rgba[o, :3] = pal[outl[o].astype(int) - 1]
            rgba[o, 3] = 1.0
        return rgba

    def _dm_layer(self, r, alpha):
        d = self.dm.get(r.path)
        if d is None:
            return None, None
        H, W = r.image_hw
        if self.v_dmview.get() == "hidden PCA":
            g = d["rgb"]
            rgba = np.concatenate([g, np.full(g.shape[:2] + (1,), alpha, np.float32)], -1)
            return rgba, (0, W, H, 0)
        if r.path not in self._dm_thumb:
            th = self._thumb(r.path)
            self._dm_thumb[r.path] = cv2.resize(d["lab"], (th.shape[1], th.shape[0]), interpolation=cv2.INTER_NEAREST)
        lab = self._dm_thumb[r.path]
        pal = np.array(self.dm_cols, np.float32)
        rgba = np.zeros(lab.shape + (4,), np.float32)
        rgba[..., :3] = pal[lab]
        rgba[..., 3] = np.where(lab > 0, alpha, 0.0)                 # class 0 = background: transparent
        return rgba, (0, W, H, 0)

    def _redraw_image(self, fast=False):
        r = self._rec()
        if r is None:
            self.cv_i.draw_idle()
            return
        H, W = r.image_hw
        gh, gw = r.grid
        bx0, by0, bx1, by1 = r.box
        bext = (bx0, bx1, by1, by0)
        if self.vbox is None:
            self.vbox = self._initial_view(r)
        if not fast:
            try:
                th = self._thumb(r.path)
            except Exception as e:
                self.v_status.set(f"Cannot read {r.name}: {e}")
                return
            self.im_photo.set_data(th)
            self.im_photo.set_extent((0, W, H, 0))
            view, alpha = self.v_view.get(), float(self.v_alpha.get())
            layer, ext = None, (0, W, H, 0)
            if self._dm_legend is not None:
                try:
                    self._dm_legend.remove()
                except Exception:
                    pass
                self._dm_legend = None
            if view == "pca" and self.pca_rgb is not None:
                sl = slice(r.offset, r.offset + r.n)
                layer = np.zeros((gh, gw, 4), np.float32)
                layer[..., :3] = self.pca_rgb[sl].reshape(gh, gw, 3)
                layer[..., 3] = np.where(self.vis[sl].reshape(gh, gw), alpha, 0.0)
                ext = bext
            elif view == "annotations":
                layer = self._ann_layer(r, alpha)
            elif view == "dinomask":
                layer, e2 = self._dm_layer(r, alpha)
                ext = e2 or ext
                if layer is not None and self.v_dmview.get() == "classes":
                    from matplotlib.patches import Patch
                    present = np.unique(self._dm_thumb.get(r.path, np.zeros(1, np.uint8)))
                    hs = [Patch(color=self.dm_cols[k], label=self.dm_classes[k]) for k in present if k > 0]
                    if hs:
                        self._dm_legend = self.ax_i.legend(handles=hs, loc="upper right", fontsize=7, framealpha=0.75)
            self.im_over.set_data(layer if layer is not None else np.zeros((2, 2, 4), np.float32))
            self.im_over.set_extent(ext)
        for c in self._contours:
            try:
                c.remove()
            except Exception:
                pass
        self._contours = []
        sg = np.zeros((2, 2, 4), np.float32)
        sext = bext
        if self.sel is not None and len(self.sel) and self.colors is not None:
            cols = self.colors[r.offset:r.offset + r.n].reshape(gh, gw, 3)
            pm = self.pix.get(self.cur)
            if pm is not None:
                nh, nw = pm.shape
                s = min(1.0, 1400.0 / max(nh, nw))
                dh, dw = max(1, int(nh * s)), max(1, int(nw * s))
                m = cv2.resize(pm, (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
                k = max(3, int(2 * dw / gw) | 1)
                dil = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
                sg = np.zeros((dh, dw, 4), np.float32)
                sg[dil & ~m] = (1.0, 1.0, 1.0, 0.6)
                cu = cv2.resize(cols, (dw, dh), interpolation=cv2.INTER_NEAREST)
                sg[m, :3] = cu[m]
                sg[m, 3] = 0.8
                cgrid, cs_hw = dil, (dh, dw)
            else:
                s = self.sel[r.offset:r.offset + r.n].reshape(gh, gw)
                cgrid = None
                if s.any():
                    dil = cv2.dilate(s.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                    sg = np.zeros((gh, gw, 4), np.float32)
                    sg[dil & ~s] = (1.0, 1.0, 1.0, 0.6)
                    sg[s, :3] = cols[s]
                    sg[s, 3] = 1.0
                    cgrid = cv2.resize(dil.astype(np.uint8), (gw * 4, gh * 4), interpolation=cv2.INTER_NEAREST) > 0
                    cs_hw = (gh * 4, gw * 4)
            if cgrid is not None and cgrid.any() and not fast:
                ch, cw = cs_hw
                xs = bx0 + (np.arange(cw) + 0.5) * (bx1 - bx0) / cw
                ys = by0 + (np.arange(ch) + 0.5) * (by1 - by0) / ch
                cs = self.ax_i.contour(xs, ys, cgrid.astype(np.float32), levels=[0.5], colors="white",
                                       linewidths=1.3, zorder=5)
                self._contours = [cs]
        ng = np.zeros((2, 2, 4), np.float32)                  # 19 Oct: not-this paint, under the halo
        nm = self.neg_pix.get(self.cur)
        if nm is not None and nm.any():
            nh, nw = nm.shape
            s = min(1.0, 1400.0 / max(nh, nw))
            dh, dw = max(1, int(nh * s)), max(1, int(nw * s))
            m = cv2.resize(nm, (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
            ng = np.zeros((dh, dw, 4), np.float32)
            ng[m] = (1.0, 0.1, 0.45, 0.45)
            if not fast:
                xs = bx0 + (np.arange(dw) + 0.5) * (bx1 - bx0) / dw
                ys = by0 + (np.arange(dh) + 0.5) * (by1 - by0) / dh
                self._contours.append(self.ax_i.contour(xs, ys, m.astype(np.float32), levels=[0.5], colors="red",
                                                        linewidths=1.1, zorder=3.5))
        self.im_neg.set_data(ng)
        self.im_neg.set_extent(bext)
        self.im_sel.set_data(sg)
        self.im_sel.set_extent(sext)
        self.ax_i.set_aspect("equal")
        x0, x1, y0, y1 = self.vbox
        self.ax_i.set_xlim(x0, x1)
        self.ax_i.set_ylim(y1, y0)
        self.cv_i.draw_idle()
        if not fast:
            self._draw_regions(r)
            self._schedule_lod()
            self._update_selinfo()

    def _goto(self, i):
        if not self.recs:
            return
        i = int(np.clip(i, 0, len(self.recs) - 1))
        if i != self.cur:
            if self._stroke is not None:
                self._stroke_end()
            self.cur = i
            self.vbox = None
            self._redraw_image()

    def _step(self, d):
        if not self.recs:
            return
        i = int(np.clip(self.cur + d, 0, len(self.recs) - 1))
        self.slider.set(i + 1)
        self._goto(i)

    # ------------------------------------------------------------ DINOMask view
    def pick_dinomask(self):
        from tkinter import filedialog, messagebox
        d = filedialog.askdirectory(parent=self.win, title="Trained DINOMask run folder (head.pt + results.json)")
        if not d:
            return
        if not (os.path.isfile(os.path.join(d, "head.pt")) and os.path.isfile(os.path.join(d, "results.json"))):
            messagebox.showerror("BioRAG Annotate", "That folder has no head.pt + results.json.", parent=self.win)
            return
        self.run_dinomask(d)

    def run_dinomask(self, run_dir):
        if not self.paths:
            return
        self.dm_run = run_dir
        self.v_dmsrc.set(os.path.basename(os.path.normpath(run_dir))[:40])
        self._start_proc(self.paths, ["--size", self.v_dm_size.get(), "--dinomask_run", run_dir], "dinomask",
                         "DINOMask")

    def _load_dinomask(self):
        cdir, size, run = self.v_cache.get().strip(), int(self.v_dm_size.get()), self.dm_run
        loaded = {}
        for p in self.paths:
            try:
                c = bat.dinomask_cache_path(cdir, p, run, size)
                if bat.dinomask_cache_is_valid(c, p):
                    with np.load(c, allow_pickle=False) as z:
                        loaded[p] = {k: z[k] for k in ("classes_map", "hidden", "grid", "classes")}
            except Exception as e:
                self.failed[os.path.basename(p)] = str(e)
        if not loaded:
            self.v_status.set("DINOMask failed: " + "; ".join(f"{k} ({v[:80]})" for k, v in self.failed.items()))
            return
        rng = np.random.default_rng(0)
        N = sum(len(v["hidden"]) for v in loaded.values())
        frac = min(1.0, MAX_FIT / max(N, 1))
        X = np.concatenate([v["hidden"][rng.choice(len(v["hidden"]), max(1, int(len(v["hidden"]) * frac)), replace=False)]
                            for v in loaded.values()]).astype(np.float32)
        mu, V = fit_pca(X)
        ref = project(X, mu, V)
        self.dm, self._dm_thumb = {}, {}
        for p, v in loaded.items():
            gh, gw = (int(x) for x in v["grid"])
            rgb = minmax_rgb(project(v["hidden"].astype(np.float32), mu, V), ref).reshape(gh, gw, 3)
            self.dm[p] = {"lab": v["classes_map"], "rgb": rgb}
        self.dm_classes = [str(c) for c in next(iter(loaded.values()))["classes"]]
        self.dm_cols = [(0, 0, 0)] + category_palette(len(self.dm_classes) - 1)
        self.v_view.set("dinomask")
        self.b_view.configure(text="View: dinomask (v)")
        self.v_status.set(f"DINOMask view: {len(self.dm)} image(s), classes {', '.join(self.dm_classes[1:])[:90]}")
        self._redraw_image()

    # ------------------------------------------------------------ menus + assign
    def _assign_entries(self, menu):
        n = int(self.sel.sum()) if self.sel is not None else 0
        npix = sum(1 for m in self.pix.values() if m.any())
        if (n == 0 and npix == 0) or self._busy:
            menu.add_command(label="Nothing selected" if not self._busy else "Busy ...", state="disabled")
            return
        n_img = len(self._jobs_recs())
        menu.add_command(label=f"{n} patches on {n_img} image(s)", state="disabled")
        menu.add_cascade(label="Assign category >", menu=self.host.category_cascade(menu, self.assign))
        menu.add_command(label="Clear selection", command=self.clear_selection)

    def _post(self, menu):
        import types
        ev = types.SimpleNamespace(x_root=self.win.winfo_pointerx(), y_root=self.win.winfo_pointery())
        self.host.post_menu(menu, ev)

    def _context_menu(self):
        menu = self.tk.Menu(self.win, tearoff=0)
        self._assign_entries(menu)
        self._post(menu)

    def _image_menu(self, ev):
        r = self._rec()
        menu = self.tk.Menu(self.win, tearoff=0)
        hit = point_to_patch(ev.xdata, ev.ydata, r.grid, r.box) if (r is not None and self.P is not None) else None
        if hit is not None:
            i, (row, col) = self.cur, hit
            gi = r.offset + row * r.grid[1] + col

            def one():
                new = np.zeros(len(self.P), bool)
                new[gi] = True
                self._apply_selection(new)
            inside = self.sel is not None and self.sel[gi]
            menu.add_command(label="Select this patch", command=one)
            ref = " (to the selection mean)" if inside else ""
        if hit is not None and not self._applied_mode:     # 18 Oct: no tokens behind applied results
            menu.add_command(label="Select similar (this image)" + ref,
                             command=lambda: self.select_similar(i, row, col, "image"))
            menu.add_command(label="Select similar (all images)" + ref,
                             command=lambda: self.select_similar(i, row, col, "all"))
        elif hit is None:
            menu.add_command(label="(outside the tokens of this view)", state="disabled")
        menu.add_separator()
        self._assign_entries(menu)
        self._post(menu)

    def current_params(self):
        p = dict(self.params_tokens)
        p.update({"tokens": self.v_variant.get(), "pcs": self.v_pcs.get(), "foreground_only": bool(self.v_fg.get()),
                  "flip_fg": bool(self.v_flip.get()), "colour_by": self.v_colour.get(),
                  "guided_filter": {"r": 8, "eps": 1e-3, "threshold": 0.5},
                  "min_component": ("1 patch" if self._rec() is not None and (self._rec().is_crop or getattr(self._rec(), "fine", False))
                                    else "0.1% of the image")})
        if self.sim is not None:
            p["similarity"] = {"what": self._sim_label, "threshold": float(self.v_sim.get())}
        if self._applied_mode:
            p["applied_rule"] = dict(getattr(self, "_applied_rule_summary", {}))
        return p

    def _jobs_recs(self):
        out = []
        for k, r in enumerate(self.recs):
            s = self.sel[r.offset:r.offset + r.n]
            pm = self.pix.get(k)
            if s.any() or (pm is not None and pm.any()):
                out.append(k)
        return out

    def assign(self, name):
        """name None = new category (asked here, or by the editor's own dialog when the open image has
        selected patches)."""
        if self._busy or self.sel is None:
            return
        ks = self._jobs_recs()
        if not ks:
            return
        self._busy = True
        jobs = []
        negs = []                                      # 19 Oct: not-this paint per job
        for k in ks:
            r = self.recs[k]
            pm = self.pix.get(k)
            jobs.append((r, self.sel[r.offset:r.offset + r.n].reshape(r.grid).copy(), None if pm is None else pm.copy()))
            nm = self.neg_pix.get(k)
            negs.append(nm.copy() if nm is not None and nm.any() else None)
        params = self.current_params()
        regs = [[r.name, [int(v) for v in r.box], self._crop_sz(r.path, r.box)] for r, _, _ in jobs if r.is_crop]
        if regs:
            params["regions"] = regs                 # 18 Oct: annotated from these zoomed regions
        if any(pm is not None for _, _, pm in jobs):
            params["brush_edited"] = True
        self.v_status.set(f"Making masks for {len(jobs)} token set(s) ...")

        def work():
            try:
                out = {}
                for k, (r, s, pm) in enumerate(jobs, 1):
                    if pm is not None:
                        m = work_mask_to_image(pm, r.image_hw, r.box)                   # brush-edited: as is
                    else:
                        # 18 Oct: regions and applied (fine) results hold small structures (setae): there the
                        # minimum component is ONE patch (in image px), not 0.1% of the image / region
                        mpx = None
                        if r.is_crop or getattr(r, "fine", False):
                            mpx = (r.box[2] - r.box[0]) / r.grid[1] * (r.box[3] - r.box[1]) / r.grid[0]
                        m = selection_to_mask(s, r.image_hw, r.grey(), box=r.box, min_px=mpx)
                        if getattr(r, "fine", False):
                            r._grey = None                  # fine working images are large: not kept
                    if negs[k - 1] is not None:             # 19 Oct: masks never include "not this" paint
                        m[work_mask_to_image(negs[k - 1], r.image_hw, r.box) > 0] = 0
                    if r.path in out:
                        out[r.path] = (np.maximum(out[r.path][0], m), out[r.path][1] + int(s.sum()))
                    else:
                        out[r.path] = (m, int(s.sum()))
                    self.q.put(("line", f"BRA_INFO masks {k} / {len(jobs)}"))
                self.q.put(("masks", (name, out, params)))
            except Exception as e:
                self.q.put(("error", f"Mask making failed: {e}"))
        threading.Thread(target=work, daemon=True).start()

    def _is_editor_image(self, path):
        cur = self.host.current_image_path()
        return bool(cur) and os.path.splitext(os.path.abspath(cur))[0] == os.path.splitext(os.path.abspath(path))[0]

    def _finish_assign(self, name, out, params):
        from tkinter import simpledialog, messagebox
        try:
            out = {p: v for p, v in out.items() if v[0].any()}
            if not out:
                messagebox.showinfo("BioRAG Annotate", "The selection gives no mask after refinement "
                                    "(too small).", parent=self.win)
                return
            ed_path = next((p for p in out if self._is_editor_image(p)), None)
            ed_ann = None
            if ed_path is not None:
                m = out[ed_path][0]
                shp = self.host.editor_image_shape()
                if shp is not None and tuple(shp[:2]) == m.shape:
                    res = self.host.add_mask_to_editor(m, name)
                    if res is None:
                        if name is None:
                            return                      # the new-category dialog was cancelled
                    else:
                        name, ed_ann = res
                else:
                    messagebox.showwarning("BioRAG Annotate", "The editor image does not have the size of the "
                                           "token image; its mask goes to the output file only.", parent=self.win)
            if name is None:
                name = simpledialog.askstring("New category", "Name of the new category:", parent=self.win)
                if not name or not name.strip():
                    return
                name = name.strip()
            asg = {"category": name, "params": params,
                   "masks": {os.path.basename(p): mask_to_rle(v[0]) for p, v in out.items()},
                   "hw": {os.path.basename(p): v[0].shape for p, v in out.items()},
                   "n_patches": sum(v[1] for v in out.values()), "editor_ann": ed_ann,
                   "editor_path": ed_path}
            self.assignments.append(asg)
            for nm, bx, sz in params.get("regions", []):          # 18 Oct: outlined on the strip, re-openable
                self._annotated_regions.append({"name": nm, "box": bx, "size": sz, "category": name,
                                                "session": self.session_ts})
            asg["regions"] = params.get("regions", [])
            self._save_settings()
            self._write_outputs()
            self._refresh_list()
            self.v_status.set(f"'{name}': {len(out)} mask(s) -> {os.path.basename(self.last_written or '')}"
                              + ("; added to the editor image" if ed_ann is not None else ""))
        finally:
            self._busy = False

    def _refresh_list(self):
        self.lb.delete(0, "end")
        for a in self.assignments:
            self.lb.insert("end", f"{a['category']}  ({len(a['masks'])} img, {a['n_patches']} patches)")

    def _out_paths(self):
        cdir = self.v_cache.get().strip()
        base = os.path.join(cdir, f"biorag_annotate_{self.session_ts}")
        return base + ".json", base + ".rlejson"

    def _write_outputs(self):
        pj, pr = self._out_paths()
        info = {}
        for a in self.assignments:
            info.update(a["hw"])
        poly, rle = build_coco(self.assignments, info, self.host.category_ids())
        write_json(pj, poly)
        write_json(pr, rle)
        self.last_written = pj

    def undo_last(self):
        from tkinter import messagebox
        if not self.assignments:
            return
        a = self.assignments.pop()
        for nm, bx, sz in a.get("regions", []):
            for k in range(len(self._annotated_regions) - 1, -1, -1):
                e = self._annotated_regions[k]
                if e["name"] == nm and list(e["box"]) == list(bx) and e["category"] == a["category"]:
                    del self._annotated_regions[k]
                    break
        self._save_settings()
        self._write_outputs()
        self._refresh_list()
        note = ""
        if a.get("editor_ann") is not None:
            if not self.host.delete_editor_mask(a["editor_ann"]):
                note = ("\n\nThe mask added to the editor image was not found on the open image "
                        "(you moved on?): delete it there by hand if needed.")
        messagebox.showinfo("BioRAG Annotate", f"Removed '{a['category']}' ({len(a['masks'])} mask(s)) from "
                            f"{os.path.basename(self.last_written)}.{note}", parent=self.win)

    def merge_into_folder(self):
        from tkinter import filedialog, messagebox
        if not self.assignments:
            messagebox.showinfo("BioRAG Annotate", "Nothing assigned yet.", parent=self.win)
            return
        base = self.host.folder_coco_path()
        if not base or not os.path.isfile(base):
            base = filedialog.askopenfilename(parent=self.win, title="Folder annotations (COCO) to merge with",
                                              filetypes=[("COCO", "*.json *.rlejson"), ("All", "*.*")])
            if not base:
                return
        try:
            with open(base) as fh:
                coco = json.load(fh)
            info = {}
            for a in self.assignments:
                info.update(a["hw"])
            poly, rle = build_coco(self.assignments, info, self.host.category_ids())
            new = rle if base.lower().endswith(".rlejson") else poly
            merged = merge_coco(coco, new)
            stem, ext = os.path.splitext(base)
            outp = f"{stem}_with_biorag_{time.strftime('%Y%m%d_%H%M%S')}{ext or '.json'}"
            write_json(outp, merged)
        except Exception as e:
            messagebox.showerror("BioRAG Annotate", f"Merge failed:\n{e}", parent=self.win)
            return
        messagebox.showinfo("BioRAG Annotate",
                            f"Wrote a NEW file (the folder annotations were not changed):\n{outp}\n\n"
                            f"{len(new['annotations'])} annotation(s) appended.\n"
                            f"Load it with 'Load Annotations' to work on it.", parent=self.win)

    # ------------------------------------------------------------ apply a zoomed-region selection (18 Oct)
    def _rule_kind(self):
        k = self.v_rule.get()
        if k != "auto":
            return k
        o = self._sel_origin
        return o if o in ("colour windows", "similar") else "learned"   # lasso / brush / patch clicks -> learned

    def _selected_mean(self):
        acc, n = None, 0
        for k, q in enumerate(self.recs):
            s = self.sel[q.offset:q.offset + q.n]
            if s.any():
                v = self._features_norm(k)[s].astype(np.float32).sum(0)
                acc = v if acc is None else acc + v
                n += int(s.sum())
        return acc / max(n, 1)

    def apply_rule(self):
        """Carry the selection of the zoomed region over to the whole image / all images, at the region's
        fine scale (tokens re-extracted whole, tiled, in the token subprocess; only results are kept)."""
        from tkinter import messagebox
        if self.proc is not None or self._busy:
            self.v_status.set("A job is running; wait for it to finish.")
            return
        prm, r = self._pca_prm, self._rec()
        if self.v_src.get() != "region" or prm is None or r is None or self.sel is None or not self.sel.any():
            messagebox.showinfo("BioRAG Annotate", "Use zoomed-region tokens and make a selection first.", parent=self.win)
            return
        kind = self._rule_kind()
        x0, y0, x1, y1 = r.box
        rule = {"kind": kind, "variant": prm["variant"], "fg": prm["fg"], "mu1": prm["mu1"], "V1": prm["V1"],
                "layer": int(self.params_tokens.get("layer", int(self.v_layer.get()))),
                "model": self.params_tokens.get("model", bat.DEFAULT_MODEL),
                "scale": [r.work_hw[0] / float(y1 - y0), r.work_hw[1] / float(x1 - x0)],
                "region_size": self.params_tokens.get("size"), "from_region": [r.name, list(r.box)],
                "region_work_hw": [int(r.work_hw[0]), int(r.work_hw[1])],
                "selection_origin": self._sel_origin, "n_selected_in_region": int(self.sel.sum())}
        if prm["fg"]:
            rule.update({"t": prm["t"], "fg_hi": prm["fg_hi"]})
        targets = [r.path] if self.v_apply_scope.get() == "image" else list(self.paths)
        self._apply_negs_pending = [(self.recs[i].path, self.recs[i].box, nm.copy()) for i, nm in self.neg_pix.items()
                                    if nm is not None and nm.any() and i < len(self.recs)]   # 19 Oct
        if kind == "colour windows":
            wins = self._window_params()
            if wins is None or not any(w["on"] for w in wins):
                messagebox.showinfo("BioRAG Annotate", "Rule 'colour windows': switch on a colour window first.",
                                    parent=self.win)
                return
            rule.update({"mu": prm["mu"], "V": prm["V"], "lo": prm["lo"], "hi": prm["hi"], "windows": wins})
            self._start_apply(rule, targets)
        elif kind == "similar":
            q = self._sim_query if (self._sel_origin == "similar" and self._sim_query is not None) else self._selected_mean()
            rule.update({"proto": np.asarray(q, np.float32), "threshold": float(self.v_sim.get()),
                         "proto_from": "similarity query" if self._sel_origin == "similar" else "selection mean"})
            self._start_apply(rule, targets)
        else:
            try:
                thr = float(self.v_lr_thr.get())
            except Exception:
                thr = 0.5
            rule["threshold"] = thr
            recs, sel, vis = list(self.recs), self.sel.copy(), self.vis.copy()
            negv = self._neg_vec(len(sel))                 # 19 Oct: "not this" = explicit negatives
            self._busy = True
            self.v_status.set("Training the learned rule (logistic regression) on the region tokens ...")

            def work():
                try:
                    X = np.concatenate([token_features(q.T, q.grid, rule["variant"]) for q in recs], 0)
                    keep, sel_t, sw_t, ninfo = learned_training_set(sel, vis, negv)
                    m = train_logreg(X[keep], sel_t[keep], sample_weight=sw_t)
                    acc = float(((logreg_proba(X[keep], m) >= thr) == sel_t[keep]).mean())
                    rule.update({"lr_mean": m["mean"], "lr_std": m["std"], "lr_w": m["w"], "lr_b": m["b"],
                                 "lr_train_accuracy": round(acc, 4), "lr_n_pos": int(sel_t[keep].sum()),
                                 "lr_n_neg": int((~sel_t[keep]).sum())})
                    rule.update(ninfo)
                    self.q.put(("rule", (rule, targets)))
                except Exception as e:
                    self.q.put(("error", f"Learned rule failed: {e}"))
            threading.Thread(target=work, daemon=True).start()

    def _start_apply(self, rule, targets):
        self._busy = False
        cdir = self.v_cache.get().strip()
        path = os.path.join(cdir, f"biorag_annotate_rule_{time.strftime('%Y%m%d_%H%M%S')}.npz")
        try:
            os.makedirs(cdir, exist_ok=True)
            rid = save_rule(path, rule)
        except Exception as e:
            self.v_status.set(f"Cannot write the rule: {e}")
            return
        self._apply_rid, self._apply_targets = rid, list(targets)
        self._apply_negs = list(getattr(self, "_apply_negs_pending", []))   # 19 Oct
        self._applied_rule_summary = dict(rule_summary(rule), rule_id=rid, rule_file=os.path.basename(path))
        self._apply_from = self._rec().path if self._rec() is not None else None
        extra = f", train acc {rule['lr_train_accuracy']}" if "lr_train_accuracy" in rule else ""
        self.v_rule_info.set(f"rule '{rule['kind']}' ({rid}{extra}) -> {len(targets)} image(s)")
        self._start_proc(targets, ["--apply_rule", path], "apply", "Apply rule (fine)")

    def _load_applied(self):
        cdir, rid = self.v_cache.get().strip(), getattr(self, "_apply_rid", None)
        recs = []
        for p in getattr(self, "_apply_targets", []):
            c = bat.applied_cache_path(cdir, p, rid)
            if os.path.isfile(c):
                try:
                    with np.load(c, allow_pickle=False) as z:
                        recs.append(FineRec(p, {k: z[k] for k in z.files}))
                except Exception as e:
                    self.failed[os.path.basename(c)] = str(e)
        if not recs:
            self.v_status.set("Applying the rule failed: " + "; ".join(f"{k} ({v[:80]})" for k, v in self.failed.items()))
            return
        self.recs_applied = recs
        self._activate_applied()

    def _activate_applied(self):
        from matplotlib import colormaps
        recs = self.recs_applied
        prev = self._rec().path if self._rec() is not None else getattr(self, "_apply_from", None)
        off = 0
        for q in recs:
            q.offset = off
            off += q.n
        self.recs, self._applied_mode = recs, True
        rs = getattr(self, "_applied_rule_summary", {})
        self.params_tokens = {"token_source": "applied (fine)", "layer": rs.get("layer"), "model": rs.get("model"),
                              "scale": rs.get("scale")}
        self.slider.configure(to=max(1, len(recs)))
        self.cur = next((k for k, q in enumerate(recs) if q.path == prev), 0)
        self.vbox = None
        self.P = np.zeros((off, 3), np.float32)
        self.pca_rgb, self.vis, self.show_idx = None, np.ones(off, bool), np.zeros(0, int)
        self.sel = np.concatenate([q.sel0 for q in recs])
        score = np.concatenate([q.score for q in recs])
        lo, hi = np.percentile(score, [1, 99]) if len(score) else (0.0, 1.0)
        self._applied_colors = colormaps["turbo"](np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1))[:, :3].astype(np.float32)
        self.colors = self._applied_colors
        self.pix, self._undo_strokes, self.sim, self._featn, self.catidx, self._hue_base = {}, [], None, {}, None, None
        self.neg_pix = {}
        for k, q in enumerate(recs):            # 19 Oct: the region's not-this paint carries to its own image
            for p, box, nm in getattr(self, "_apply_negs", []):
                if p == q.path:
                    mm = map_negatives(nm, box, q.image_hw, q.box, q.work_hw)
                    self.neg_pix[k] = mm if k not in self.neg_pix else np.maximum(self.neg_pix[k], mm)
        for k in list(self.neg_pix):
            if self.neg_pix[k].any():
                self._drop_neg_patches(k)
            else:
                del self.neg_pix[k]
        self._sel_origin = "applied"
        if self.v_colour.get() == "mask category":
            self._load_categories(self._cat_path(), quiet=True)
            self._apply_colours()
        self.v_src.set("applied")
        try:
            self.slider.set(self.cur + 1)
        except Exception:
            pass
        per = ", ".join(f"{q.name[:24]} {int(q.sel0.sum())}/{q.n}" for q in recs[:4])
        self.v_status.set(f"Applied rule '{rs.get('kind')}' at fine scale: {len(recs)} image(s); selected patches {per}")
        self._redraw_scatter(full=True)
        self._redraw_image()

    # ------------------------------------------------------------ remembered settings (18 Oct)
    _SETTING_VARS = ("v_size", "v_layer", "v_variant", "v_pcs", "v_fg", "v_flip", "v_colour", "v_lasso", "v_src",
                     "v_region_size", "v_sim", "v_view", "v_dmview", "v_alpha", "v_brush", "v_hue_scope")

    def _settings_path(self):
        cdir = self.v_cache.get().strip()
        return os.path.join(cdir, "biorag_annotate_settings.json") if cdir else None

    def _save_settings(self):
        p = self._settings_path()
        if not p:
            return
        try:
            st = {k: getattr(self, k).get() for k in self._SETTING_VARS if hasattr(self, k)}
            st["colour_windows"] = [{k: v.get() for k, v in d.items() if k != "cv"} for d in getattr(self, "hw", [])]
            st["crops"] = [[os.path.basename(pp), list(map(int, b)), self._crop_sz(pp, b)] for pp, b in self.crops]
            st["annotated_regions"] = list(self._annotated_regions)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                json.dump(st, fh, indent=1)
        except Exception:
            pass                                   # a convenience only

    def _restore_settings(self):
        p = self._settings_path()
        try:
            with open(p) as fh:
                st = json.load(fh)
        except Exception:
            return
        for k in self._SETTING_VARS:
            if k in st and hasattr(self, k):
                try:
                    getattr(self, k).set(st[k])
                except Exception:
                    pass
        for d, sv in zip(getattr(self, "hw", []), st.get("colour_windows", [])):
            for k, v in sv.items():
                if k in d:
                    try:
                        d[k].set(v)
                    except Exception:
                        pass
        by_name = {os.path.basename(pp): pp for pp in self.paths}
        self.crops = []
        for c in st.get("crops", []):
            if c and c[0] in by_name:
                self.crops.append((by_name[c[0]], tuple(c[1])))
                if len(c) > 2 and c[2]:
                    self._crop_size[(by_name[c[0]], tuple(c[1]))] = int(c[2])
        self._annotated_regions = [a for a in st.get("annotated_regions", []) if a.get("name") in by_name]
        self._want_region = st.get("v_src") == "region"
        for k in range(len(getattr(self, "hw", []))):
            self._draw_hue_window(k)

    def _scan_crops(self):
        """Zoomed regions extracted earlier (any session): crop token caches of these images at this layer, any
        size (18 Oct: views extracted at the same scale as their region have their own size)."""
        import glob, re
        cdir, layer = self.v_cache.get().strip(), int(self.v_layer.get())
        out = list(self.crops)
        for pp in self.paths:
            pat = glob.escape(os.path.join(cdir, os.path.basename(pp))) + "__crop_*__L%d_s*.npz" % layer
            for f in sorted(glob.glob(pat)):
                m = re.search(r"__crop_(\d+)_(\d+)_(\d+)_(\d+)__L\d+_s(\d+)\.npz$", os.path.basename(f))
                if not m:
                    continue
                box = tuple(int(v) for v in m.groups()[:4])
                self._crop_size.setdefault((pp, box), int(m.group(5)))
                if (pp, box) not in out:
                    out.append((pp, box))
        return out

    # ------------------------------------------------------------ regions list + annotated regions (18 Oct)
    def _region_label(self, p, box):
        cats = [a["category"] for a in self._annotated_regions
                if a.get("name") == os.path.basename(p) and tuple(a.get("box", ())) == tuple(box)]
        return ("%s [%d,%d-%d,%d] s%d" % ((os.path.basename(p)[:30],) + tuple(box) + (self._crop_sz(p, box),))
                + (("  * " + ", ".join(sorted(set(cats)))) if cats else ""))

    def _refresh_regions_list(self):
        try:
            self._region_entries = self._scan_crops()
            self.cb_regions.configure(values=[self._region_label(p, b) for p, b in self._region_entries])
        except Exception:
            pass

    def open_region_pick(self):
        """Jump to a cached zoomed region (its tokens are loaded from the cache, not re-extracted)."""
        lab = self.v_region_pick.get()
        ents = getattr(self, "_region_entries", None) or self._scan_crops()
        hit = next(((p, b) for p, b in ents if self._region_label(p, b) == lab), None)
        if hit is None:
            return
        self.open_region(*hit)

    def open_region(self, path, box):
        box = tuple(int(v) for v in box)
        k = next((i for i, q in enumerate(self.recs_region) if q.path == path and q.box == box), None)
        if k is not None:
            if self.v_src.get() == "region" and not self._applied_mode and self.recs is self.recs_region:
                self.slider.set(k + 1)
                self._goto(k)
            else:
                self.v_src.set("region")
                self._activate(self.recs_region, self.params_region, cur=k)
            return
        if (path, box) not in self.crops:
            self.crops.append((path, box))
        self._new_crop = (path, box)
        cdir, layer, size = self.v_cache.get().strip(), int(self.v_layer.get()), self._crop_sz(path, box)
        if bat.cache_is_valid(bat.cache_path(cdir, path, layer, size, box), path, layer, size, bat.DEFAULT_MODEL):
            self._load_region()
        else:                                  # not cached (e.g. the image changed): extract it
            lines = ["%s\t%d,%d,%d,%d\t%d" % ((p,) + tuple(b) + (self._crop_sz(p, b),)) for p, b in self.crops]
            self._start_proc(lines, ["--size", self.v_region_size.get(), "--layer", self.v_layer.get()], "region",
                             "Region tokens")

    def _draw_regions(self, r):
        """Outline the regions of this image on the strip: annotated ones (solid orange + category), other cached
        regions (thin dashed white)."""
        from matplotlib.patches import Rectangle
        for a in self._region_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._region_artists = []
        if r is None:
            return
        done = set()
        for a in self._annotated_regions:
            if a.get("name") != r.name:
                continue
            x0, y0, x1, y1 = a["box"]
            self._region_artists.append(self.ax_i.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                                                      ec="#ff9f1c", lw=1.6, zorder=6)))
            self._region_artists.append(self.ax_i.text(x0, y0, " " + a.get("category", ""), color="#ff9f1c",
                                                       fontsize=7, va="bottom", zorder=6, clip_on=True))
            done.add(tuple(a["box"]))
        for p, b in getattr(self, "_region_entries", []) or []:
            if p == r.path and tuple(b) not in done:
                x0, y0, x1, y1 = b
                self._region_artists.append(self.ax_i.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                                                          ec="white", lw=0.8, ls="--", zorder=6)))

    def _restore_region(self):
        self.crops = self._scan_crops()
        if self.crops:
            self._new_crop = self.crops[-1]
            self._load_region()

    def close(self):
        self._save_settings()
        self.stop()
        try:
            self.win.destroy()
        except Exception:
            pass


_open = {"win": None}


def open_window(host):
    """Open the window (or raise the one already open)."""
    w = _open.get("win")
    try:
        if w is not None and w.win.winfo_exists():
            w.win.lift()
            return w
    except Exception:
        pass
    _open["win"] = BioRAGAnnotateWindow(host)
    return _open["win"]
