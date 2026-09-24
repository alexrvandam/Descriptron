#!/usr/bin/env python3
"""
DINOv3 landmark transfer  —  v52 (v51 + --rotate_refs orientation search)
=================================
Successor to DINOV3-patch-match-v50.py, rebuilt as a landmark *transfer*
pipeline rather than a patch-similarity demo.

Fixes carried over from v50
---------------------------
(1) GRID SCRAMBLING.  v50 pooled patch tokens across flips/rotations WITHOUT
    inverse-warping the token grid, so the descriptor at patch i was the mean of
    features from up to 7 anatomically different locations.  Here every geometric
    augmentation is un-warped back to canonical orientation before averaging.
(2) REGISTER OFFSET.  v50 sliced tokens as [1 : 1+num_patches].  Models with
    register tokens (dinov2-with-registers, dinov3) put CLS + N registers first,
    so that slice ingests registers as patches and shifts the whole grid.  Here
    the offset is derived as (seq_len - num_patches).
(3) ATTENTION != IDENTITY.  Matching is on patch tokens (or the key facet),
    never on attention maps, which encode saliency rather than location identity.

Pipeline (5 stages)
-------------------
    1. PCA-affine alignment of A -> B from the foreground masks
    2. affine-constrained LOCAL search  (not global argmax)
    3. forward-backward CYCLE CONSISTENCY gate
    4. RANSAC geometric outlier rejection
    5. soft-argmax sub-patch refinement

Modes
-----
    --selftest   synthetic warp of one image, known ground truth, reports
                 median pixel error per configuration (this is what proves or
                 disproves the augmentation fix)
    --sweep      layer sweep
    --pair       cross-species transfer A -> B with figures
"""
from __future__ import annotations
import os, argparse, json, math, warnings
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# validated palette (see dataviz palette.md)
C_OK   = "#1baf7a"   # accepted match
C_BAD  = "#e34948"   # rejected / flagged
# Tk's "goldenrod", so a v=1 landmark is the same colour here as in Descriptron's
# KP_COLOR.  Red is deliberately NOT reused for flagged points: in the GUI red means
# v=0, the landmark no reference supplied, which is a different thing to check.
C_FLAG = "#DAA520"
C_REF  = "#2a78d6"   # reference landmark
C_PRED = "#eb6834"   # predicted
C_GRID = "#e8e8e4"
SURF   = "#fcfcfb"
INK    = "#0b0b0b"
INK2   = "#52514e"


# ───────────────────────────── backbone ──────────────────────────────────────
class Backbone:
    def __init__(self, model_id: str, img_dim: int, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        cfg = self.model.config
        self.patch = getattr(cfg, "patch_size", 14)
        self.img_dim = (img_dim // self.patch) * self.patch
        self.grid = self.img_dim // self.patch
        self.n_patches = self.grid * self.grid
        self.n_layers = getattr(cfg, "num_hidden_layers", 12)
        self._keys = {}
        self._hooked = False

    def _prep(self, img: Image.Image):
        inp = self.processor(images=img, return_tensors="pt",
                             do_resize=False, do_center_crop=False)
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inp.items()}

    def _hook_keys(self):
        """Best-effort capture of the key projection of each block."""
        if self._hooked:
            return
        def mk(i):
            def fn(_m, _in, out):
                self._keys[i] = out.detach()
            return fn
        n = 0
        for name, mod in self.model.named_modules():
            if name.endswith(("attention.key", "attn.k_proj", "attention.k_proj", "self_attn.k_proj")):
                try:
                    idx = int([p for p in name.split(".") if p.isdigit()][0])
                except (IndexError, ValueError):
                    continue
                mod.register_forward_hook(mk(idx)); n += 1
        self._hooked = True
        self.keys_available = n > 0

    def raw_tokens(self, img: Image.Image, layer: int, facet: str = "token") -> torch.Tensor:
        """(n_patches, D) tokens for one image, register offset handled."""
        if facet == "key":
            self._hook_keys()
        with torch.inference_mode():
            out = self.model(**self._prep(img), output_hidden_states=True)
        if facet == "key" and getattr(self, "keys_available", False) and layer in self._keys:
            seq = self._keys[layer][0]
        else:
            seq = out.hidden_states[layer][0]
        offset = seq.shape[0] - self.n_patches      # CLS + registers
        if offset < 0:
            raise RuntimeError(f"seq {seq.shape[0]} < n_patches {self.n_patches}")
        return F.normalize(seq[offset: offset + self.n_patches].float(), dim=-1)

    # ---- augmentation pooling ------------------------------------------------
    _GEOM = [
        ("id",     lambda im: im,                                   lambda g: g),
        ("fliplr", lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),  lambda g: torch.flip(g, dims=[1])),
        ("fliptb", lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),  lambda g: torch.flip(g, dims=[0])),
        ("rot90",  lambda im: im.rotate(90,  expand=False),          lambda g: torch.rot90(g, k=-1, dims=(0, 1))),
        ("rot180", lambda im: im.rotate(180, expand=False),          lambda g: torch.rot90(g, k=-2, dims=(0, 1))),
        ("rot270", lambda im: im.rotate(270, expand=False),          lambda g: torch.rot90(g, k=-3, dims=(0, 1))),
    ]

    def embed(self, img: Image.Image, layer: int, aug: str = "fixed",
              facet: str = "token") -> torch.Tensor:
        """aug: none | broken | fixed | photo
        broken = v50 behaviour (geometric pooling with NO inverse warp)."""
        base = self.raw_tokens(img, layer, facet)
        if aug == "none":
            return base
        pool = []
        if aug in ("broken", "fixed"):
            for name, fwd, inv in self._GEOM:
                t = self.raw_tokens(fwd(img), layer, facet)
                if aug == "fixed":
                    g = t.reshape(self.grid, self.grid, -1)
                    t = inv(g).reshape(self.n_patches, -1)
                pool.append(t)
        else:
            pool.append(base)
        if aug in ("photo", "fixed"):
            for Enh in (ImageEnhance.Brightness, ImageEnhance.Contrast):
                for f in (0.6, 1.4):
                    pool.append(self.raw_tokens(Enh(img).enhance(f), layer, facet))
        return F.normalize(torch.stack(pool, 0).mean(0), dim=-1)


# ─────────────────────────── geometry helpers ────────────────────────────────
def load_pair(img_path: str, mask_path: str | None, dim: int):
    img = Image.open(img_path).convert("RGB")
    if mask_path and os.path.exists(mask_path):
        m = np.array(Image.open(mask_path).convert("L").resize(img.size)) > 127
    else:                                     # fall back: threshold off white bg
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        m = g < 240
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((5, 5), np.uint8)).astype(bool)
    ys, xs = np.where(m)
    if len(xs) == 0:
        raise RuntimeError(f"empty mask for {img_path}")
    pad = 0.06
    x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - pad * w)); x1 = min(img.size[0], int(x1 + pad * w))
    y0 = max(0, int(y0 - pad * h)); y1 = min(img.size[1], int(y1 + pad * h))
    img = img.crop((x0, y0, x1, y1)).resize((dim, dim), Image.BICUBIC)
    mk = Image.fromarray((m * 255).astype(np.uint8)).crop((x0, y0, x1, y1)) \
              .resize((dim, dim), Image.NEAREST)
    return img, (np.array(mk) > 127)


def letterbox(img: Image.Image, dim: int, fill=(128, 128, 128)):
    """Resize preserving aspect ratio and pad to a dim x dim square.

    Forcing a non-square plate into a square stretches the specimen by a factor that
    differs per image (AntWeb plates run 817x808 to 1075x808), so every specimen would
    arrive at the backbone under a different anisotropic distortion — which no single
    affine can undo. Letterboxing keeps shape comparable across specimens.

    Returns (padded_image, scale, ox, oy):  x_pad = x_orig * scale + ox
    """
    W, H = img.size
    scale = dim / max(W, H)
    nw, nh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    im = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (dim, dim), fill)
    ox, oy = (dim - nw) // 2, (dim - nh) // 2
    canvas.paste(im, (ox, oy))
    return canvas, scale, ox, oy


def prepare_image(bb, raw: Image.Image, layer: int, img_path: str,
                  mask_dir: str | None, margin: float = 0.06):
    """Crop to the specimen, then letterbox — undistorted AND high resolution.

    Plain letterboxing of a 1075x808 plate wastes ~25% of the frame on padding, so
    fewer patches land on the animal. Cropping to the mask bounding box first lets
    the specimen fill the square. Two passes: a coarse mask to find the subject,
    then the real mask on the crop.

    Returns (image, mask, (scale, ox, oy, cx0, cy0)) where
        x_pad = (x_orig - cx0) * scale + ox
    """
    W, H = raw.size
    coarse, s0, ox0, oy0 = letterbox(raw, bb.img_dim)
    m0 = find_user_mask(img_path, mask_dir, bb.img_dim)
    if m0 is None:
        m0 = auto_mask(bb, coarse, layer)
    ys, xs = np.nonzero(m0)
    if len(xs) == 0:                       # nothing found: fall back to whole frame
        return coarse, np.ones((bb.img_dim, bb.img_dim), bool), (s0, ox0, oy0, 0, 0)
    x0 = (xs.min() - ox0) / s0; x1 = (xs.max() - ox0) / s0
    y0 = (ys.min() - oy0) / s0; y1 = (ys.max() - oy0) / s0
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    cx0 = int(max(0, x0 - mw)); cx1 = int(min(W, x1 + mw))
    cy0 = int(max(0, y0 - mh)); cy1 = int(min(H, y1 + mh))
    if cx1 - cx0 < 32 or cy1 - cy0 < 32:
        return coarse, m0, (s0, ox0, oy0, 0, 0)
    crop = raw.crop((cx0, cy0, cx1, cy1))
    im, s1, ox1, oy1 = letterbox(crop, bb.img_dim)
    mk = auto_mask(bb, im, layer) if find_user_mask(img_path, mask_dir, bb.img_dim) is None \
         else None
    if mk is None:
        mk = auto_mask(bb, im, layer)
    return im, mk, (s1, ox1, oy1, cx0, cy0)


def find_user_mask(img_path: str, mask_dir: str | None, dim: int):
    """Look for a user-supplied mask (e.g. from SAM2/SAM3) for this image.

    Tried in order: <mask_dir>/<stem>_mask.png, <mask_dir>/<stem>.png, then the
    same two beside the image.  Returns None if absent, in which case the caller
    falls back to the generic auto-mask.  Nothing here is taxon-specific."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    cands = []
    for d in ([mask_dir] if mask_dir else []) + [os.path.dirname(img_path)]:
        cands += [os.path.join(d, stem + "_mask.png"), os.path.join(d, stem + ".png")]
    for c in cands:
        if os.path.exists(c):
            m = Image.open(c).convert("L")
            W, H = m.size
            scale = dim / max(W, H)
            nw, nh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
            m = m.resize((nw, nh), Image.NEAREST)
            canvas = Image.new("L", (dim, dim), 0)      # padding is background
            canvas.paste(m, ((dim - nw) // 2, (dim - nh) // 2))
            return np.array(canvas) > 127
    return None


def auto_mask(bb, img: Image.Image, layer: int) -> np.ndarray:
    """Foreground mask from DINOv3 patch features, no extra dependencies.

    AntWeb plates have an in-focus specimen on a blurred background, which the first
    principal component of the patch tokens separates cleanly.  Sign of PC1 is
    ambiguous, so the side whose patches are more central is taken as foreground;
    then largest connected component + hole fill + dilation back to pixel space.
    """
    t = bb.raw_tokens(img, layer).float().cpu().numpy()
    t = t - t.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(t, full_matrices=False)
    pc1 = (t @ vt[0]).reshape(bb.grid, bb.grid)
    thr = np.median(pc1)
    a = pc1 > thr
    rr, cc = np.mgrid[0:bb.grid, 0:bb.grid]
    ctr = (bb.grid - 1) / 2.0
    d = np.hypot(rr - ctr, cc - ctr)
    fg = a if d[a].mean() < d[~a].mean() else ~a
    m = (fg.astype(np.uint8) * 255)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = ((lab == big).astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.resize(m, (bb.img_dim, bb.img_dim), interpolation=cv2.INTER_NEAREST)
    return m > 127


def load_landmarks_coco(path: str, img_w: int, img_h: int, dim: int,
                        crop_box=None, lb=None):
    """Read COCO keypoints -> (xy, ids, visibility).

    ALL slots are returned, including v=0 ('absent on this specimen'), so landmark
    numbering stays stable end to end — detectron2 requires a fixed keypoint count
    per category, and the index is the landmark identity."""
    d = json.load(open(path))
    ann = d["annotations"][0]
    kp = ann["keypoints"]
    xs, ys, vs = kp[0::3], kp[1::3], kp[2::3]
    order = ann.get("point_order", list(range(1, len(vs) + 1)))
    pts, idx, vis = [], [], []
    for x, y, v, o in zip(xs, ys, vs, order):
        pts.append([x, y]); idx.append(int(o)); vis.append(int(v))
    pts = np.array(pts, dtype=float)
    if lb is not None:                      # letterbox: uniform scale + offset
        scale, ox, oy = lb
        pts[:, 0] = pts[:, 0] * scale + ox
        pts[:, 1] = pts[:, 1] * scale + oy
    elif crop_box is not None:
        x0, y0, x1, y1 = crop_box
        pts[:, 0] = (pts[:, 0] - x0) * dim / max(x1 - x0, 1)
        pts[:, 1] = (pts[:, 1] - y0) * dim / max(y1 - y0, 1)
    else:
        pts[:, 0] *= dim / img_w
        pts[:, 1] *= dim / img_h
    return pts, idx, vis


def pca_frame(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(np.float64)
    mu = P.mean(0)
    C = np.cov((P - mu).T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    return mu, V[:, order], np.sqrt(np.maximum(w[order], 1e-9))


def pca_affine(maskA, maskB):
    """2x3 affine taking A-coords to B-coords via mask PCA frames."""
    muA, VA, sA = pca_frame(maskA)
    muB, VB, sB = pca_frame(maskB)
    for k in range(2):                       # resolve axis sign ambiguity
        if np.dot(VA[:, k], VB[:, k]) < 0:
            VB[:, k] = -VB[:, k]
    S = np.diag(sB / np.maximum(sA, 1e-9))
    M = VB @ S @ VA.T
    t = muB - M @ muA
    return np.hstack([M, t[:, None]])


def feature_affine(fa: torch.Tensor, fb: torch.Tensor, maskA: np.ndarray,
                   maskB: np.ndarray, patch: int, grid: int,
                   min_inliers: int = 8, cycle_tol: float = 1.5):
    """Zero-shot alignment from the tokens themselves — no mask PCA.

    1. mutual nearest neighbour over FOREGROUND patches only (unconstrained)
    2. forward-backward cycle filter
    3. RANSAC partial-affine (rotation + uniform scale + translation)

    Returns a 2x3 affine A->B, or None if too few consistent matches. Robust where
    mask PCA is not: an antenna in a different pose changes the mask's principal
    axes, but it only adds outliers here, which RANSAC discards.
    """
    fill_a = make_patch_fill(maskA, grid, patch) > 0.15
    fill_b = make_patch_fill(maskB, grid, patch) > 0.15
    ia, ib = np.where(fill_a)[0], np.where(fill_b)[0]
    if len(ia) < min_inliers or len(ib) < min_inliers:
        return None
    S = (fa[ia] @ fb[ib].T).float().cpu().numpy()
    fwd = S.argmax(1)                 # each A-patch -> best B-patch
    bwd = S.argmax(0)                 # each B-patch -> best A-patch
    keep = bwd[fwd] == np.arange(len(ia))     # mutual NN == cycle consistent
    if keep.sum() < min_inliers:
        return None
    pa = patch_to_xy(ia[keep], patch, grid)
    pb = patch_to_xy(ib[fwd[keep]], patch, grid)
    M, inl = cv2.estimateAffinePartial2D(
        pa.astype(np.float32), pb.astype(np.float32), method=cv2.RANSAC,
        ransacReprojThreshold=patch * 1.5, maxIters=8000, confidence=0.995)
    if M is None or inl is None or int(inl.sum()) < min_inliers:
        return None
    return M


def make_patch_fill(mask: np.ndarray, grid: int, patch: int) -> np.ndarray:
    """Fraction of each patch cell covered by the mask."""
    m = mask.astype(float)
    m = m[:grid * patch, :grid * patch]
    return m.reshape(grid, patch, grid, patch).mean(axis=(1, 3)).ravel()


def apply_affine(pts: np.ndarray, A: np.ndarray) -> np.ndarray:
    p = np.hstack([pts, np.ones((len(pts), 1))])
    return (A @ p.T).T


def xy_to_patch(xy, patch, grid):
    c = np.clip((xy[:, 0] // patch).astype(int), 0, grid - 1)
    r = np.clip((xy[:, 1] // patch).astype(int), 0, grid - 1)
    return r * grid + c


def patch_to_xy(idx, patch, grid):
    r, c = idx // grid, idx % grid
    return np.stack([c * patch + patch / 2, r * patch + patch / 2], 1).astype(float)


# ─────────────────────────── the 5-stage matcher ─────────────────────────────
@dataclass
class MatchResult:
    pred: np.ndarray          # (n,2) predicted xy in B
    accepted: np.ndarray      # (n,) bool
    stage: np.ndarray         # (n,) which stage rejected: 0 ok,1 cycle,2 ransac
    cycle_rate: float


def transfer(fa: torch.Tensor, fb: torch.Tensor, lm_xy: np.ndarray,
             A: np.ndarray, patch: int, grid: int,
             radius_patches: float = 4.0, cycle_tol: float = 1.5,
             use_affine: bool = True, use_cycle: bool = True,
             use_ransac: bool = True, use_soft: bool = True) -> MatchResult:
    n = len(lm_xy)
    qidx = xy_to_patch(lm_xy, patch, grid)
    sims = (fa @ fb.T).float().cpu().numpy()          # (P, P)

    pred_patch = np.zeros(n, dtype=int)
    # ---- stage 1+2: affine-constrained local search -------------------------
    pred_aff = apply_affine(lm_xy, A) if use_affine else lm_xy
    rc = np.stack([np.arange(grid * grid) // grid, np.arange(grid * grid) % grid], 1)
    for i in range(n):
        s = sims[qidx[i]].copy()
        if use_affine:
            cy, cx = pred_aff[i, 1] / patch, pred_aff[i, 0] / patch
            d = np.hypot(rc[:, 0] + .5 - cy, rc[:, 1] + .5 - cx)
            s[d > radius_patches] = -np.inf
            if not np.isfinite(s).any():
                s = sims[qidx[i]].copy()
        pred_patch[i] = int(np.argmax(s))

    # ---- stage 3: forward-backward cycle consistency -------------------------
    stage = np.zeros(n, dtype=int)
    if use_cycle:
        back = sims[:, pred_patch].argmax(axis=0)      # B-patch -> best A-patch
        d = np.hypot(rc[back, 0] - rc[qidx, 0], rc[back, 1] - rc[qidx, 1])
        stage[d > cycle_tol] = 1
    cycle_rate = float((stage == 0).mean())

    pred_xy = patch_to_xy(pred_patch, patch, grid)

    # ---- stage 5: soft-argmax sub-patch refinement ---------------------------
    if use_soft:
        for i in range(n):
            r0, c0 = pred_patch[i] // grid, pred_patch[i] % grid
            rr = np.clip(np.arange(r0 - 1, r0 + 2), 0, grid - 1)
            cc = np.clip(np.arange(c0 - 1, c0 + 2), 0, grid - 1)
            idx = (rr[:, None] * grid + cc[None, :]).ravel()
            w = sims[qidx[i], idx]
            w = np.exp((w - w.max()) / 0.02); w /= w.sum()
            xy = patch_to_xy(idx, patch, grid)
            pred_xy[i] = (w[:, None] * xy).sum(0)

    # ---- stage 4: RANSAC geometric outlier rejection -------------------------
    if use_ransac and (stage == 0).sum() >= 4:
        ok = np.where(stage == 0)[0]
        M, inl = cv2.estimateAffinePartial2D(
            lm_xy[ok].astype(np.float32), pred_xy[ok].astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=patch * 2.0,
            maxIters=5000, confidence=0.995)
        if inl is not None:
            bad = ok[inl.ravel() == 0]
            stage[bad] = 2
    return MatchResult(pred_xy, stage == 0, stage, cycle_rate)


# ─────────────────────────────── self test ───────────────────────────────────
def synth_warp(img: Image.Image, mask: np.ndarray, angle=18.0, scale=0.88,
               tx=0.05, ty=-0.04):
    d = img.size[0]
    M = cv2.getRotationMatrix2D((d / 2, d / 2), angle, scale)
    M[0, 2] += tx * d; M[1, 2] += ty * d
    im2 = cv2.warpAffine(np.array(img), M, (d, d), flags=cv2.INTER_CUBIC,
                         borderValue=(255, 255, 255))
    mk2 = cv2.warpAffine(mask.astype(np.uint8) * 255, M, (d, d),
                         flags=cv2.INTER_NEAREST) > 127
    return Image.fromarray(im2), mk2, M


def sample_landmarks(mask: np.ndarray, n: int, patch: int, rng) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    keep = (xs > patch * 2) & (xs < mask.shape[1] - patch * 2) & \
           (ys > patch * 2) & (ys < mask.shape[0] - patch * 2)
    xs, ys = xs[keep], ys[keep]
    sel = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return np.stack([xs[sel], ys[sel]], 1).astype(float)


ALIGN_MODE = "mask"


def run_selftest(bb: Backbone, img, mask, layers, facet, outdir, n_lm=60, radius=4.0):
    rng = np.random.default_rng(SEED)
    img2, mask2, M = synth_warp(img, mask)
    lm = sample_landmarks(mask, n_lm, bb.patch, rng)
    truth = apply_affine(lm, M)
    A = pca_affine(mask, mask2)

    rows = []
    for layer in layers:
        for aug in ("none",):
            fa = bb.embed(img,  layer, aug, facet)
            fb = bb.embed(img2, layer, aug, facet)
            for tag, kw in (("global argmax", dict(use_affine=False, use_cycle=False,
                                                   use_ransac=False, use_soft=False)),
                            ("+affine",       dict(use_affine=True,  use_cycle=False,
                                                   use_ransac=False, use_soft=False)),
                            ("+cycle",        dict(use_affine=True,  use_cycle=True,
                                                   use_ransac=False, use_soft=False)),
                            ("+ransac",       dict(use_affine=True,  use_cycle=True,
                                                   use_ransac=True,  use_soft=False)),
                            ("+soft-argmax",  dict(use_affine=True,  use_cycle=True,
                                                   use_ransac=True,  use_soft=True))):
                A_use = A
                if ALIGN_MODE == "feature":
                    Af = feature_affine(fa, fb, mask, mask2, bb.patch, bb.grid)
                    A_use = Af if Af is not None else A
                r = transfer(fa, fb, lm, A_use, bb.patch, bb.grid, radius_patches=radius, **kw)
                err = np.hypot(*(r.pred - truth).T)
                sel = r.accepted if r.accepted.any() else np.ones(len(err), bool)
                rows.append(dict(layer=layer, aug=aug, align=ALIGN_MODE, stage=tag,
                                 median_err=float(np.median(err[sel])),
                                 p90_err=float(np.percentile(err[sel], 90)),
                                 kept=float(sel.mean()), cycle=r.cycle_rate))
                print(f"L{layer:<3} {ALIGN_MODE:<8} {tag:<14} "
                      f"med {rows[-1]['median_err']:6.2f}px  "
                      f"p90 {rows[-1]['p90_err']:6.2f}px  kept {sel.mean():.2f}")
    json.dump(rows, open(os.path.join(outdir, "selftest.json"), "w"), indent=1)
    return rows, (img, img2, lm, truth, A)


# ─────────────────────────────── figures ─────────────────────────────────────
def fig_aug(rows, outdir, patch):
    import collections
    best = collections.defaultdict(dict)
    for r in rows:
        if r["stage"] != "+soft-argmax":
            continue
        best[r["aug"]][r["layer"]] = r["median_err"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    style = {"none": ("#6b6a66", "no augmentation"),
             "broken": (C_BAD, "v50 pooling (grid not un-warped)"),
             "fixed": (C_OK, "v51 pooling (grid un-warped)")}
    for aug, d in best.items():
        ls = sorted(d)
        col, lab = style[aug]
        ax.plot(ls, [d[l] for l in ls], "-o", color=col, lw=1.8, ms=4.5,
                mfc=SURF, mew=1.4, label=lab)
    ax.axhline(patch, ls="--", lw=.8, color="#9a9a94")
    ax.text(ax.get_xlim()[0], patch * 1.06, "1 patch", size=7, color=INK2)
    ax.set_xlabel("DINOv3 block (layer)", size=8, color=INK2)
    ax.set_ylabel("median landmark error (px)", size=8, color=INK2)
    ax.set_title("Augmentation pooling: does un-warping the token grid matter?",
                 size=9, color=INK, weight="bold")
    ax.set_yscale("log")
    ax.grid(color=C_GRID, lw=.5); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(labelsize=7, colors=INK2)
    ax.legend(fontsize=7, frameon=False)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_augmentation.png"))
    plt.close(fig)


def fig_stages(rows, outdir, layer, patch):
    order = ["global argmax", "+affine", "+cycle", "+ransac", "+soft-argmax"]
    aug_used = rows[0]["aug"] if rows else "fixed"
    sub = [r for r in rows if r["layer"] == layer and r["aug"] == aug_used]
    d = {r["stage"]: r for r in sub}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0), dpi=200)
    x = np.arange(len(order))
    a1.bar(x, [d[o]["median_err"] for o in order], color=C_REF, width=.62)
    a1.axhline(patch, ls="--", lw=.8, color="#9a9a94")
    a1.set_ylabel("median error (px)", size=8, color=INK2)
    a1.set_yscale("log")
    a2.bar(x, [d[o]["kept"] * 100 for o in order], color=C_OK, width=.62)
    a2.set_ylabel("landmarks retained (%)", size=8, color=INK2)
    a2.set_ylim(0, 105)
    for ax, t in ((a1, "Accuracy by pipeline stage"), (a2, "Coverage by pipeline stage")):
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right", size=7)
        ax.set_title(t, size=9, color=INK, weight="bold")
        ax.grid(axis="y", color=C_GRID, lw=.5); ax.set_axisbelow(True)
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(labelsize=7, colors=INK2); ax.set_facecolor(SURF)
    fig.patch.set_facecolor(SURF)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_stages.png"))
    plt.close(fig)


def fig_transfer(imgA, imgB, lm, pred, accepted, outdir, name, truth=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.6), dpi=200)
    a1.imshow(imgA); a2.imshow(imgB)
    a1.set_title("reference (landmarks placed)", size=9, color=INK, weight="bold")
    a2.set_title("target (transferred)", size=9, color=INK, weight="bold")
    a1.scatter(lm[:, 0], lm[:, 1], s=16, c=C_REF, edgecolors="white", linewidths=.5)
    ok, bad = accepted, ~accepted
    a2.scatter(pred[ok, 0], pred[ok, 1], s=18, c=C_OK, edgecolors="white",
               linewidths=.5, label=f"accepted ({ok.sum()})")
    if bad.any():
        a2.scatter(pred[bad, 0], pred[bad, 1], s=18, facecolors="none",
                   edgecolors=C_FLAG, linewidths=1.0, label=f"flagged ({bad.sum()})")
    if truth is not None:
        a2.scatter(truth[:, 0], truth[:, 1], s=8, marker="x", c=INK2,
                   linewidths=.7, label="ground truth")
    a2.legend(fontsize=7, frameon=False, loc="lower right")
    for a in (a1, a2):
        a.axis("off"); a.set_facecolor(SURF)
    fig.patch.set_facecolor(SURF)
    fig.suptitle(name, size=10, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"fig3_transfer_{name}.png"))
    plt.close(fig)


# ──────────────────────────────── main ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--facet", default="token", choices=["token", "key"])
    ap.add_argument("--imgA", required=True); ap.add_argument("--maskA", default=None)
    ap.add_argument("--imgB", default=None);  ap.add_argument("--maskB", default=None)
    ap.add_argument("--layers", default="6,8,9,10,11")
    ap.add_argument("--outdir", default="lm_out")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--landmarks", default=None, help="COCO keypoints JSON for imgA")
    ap.add_argument("--batch_glob", default=None, help="glob of target images")
    ap.add_argument("--batch_n", type=int, default=6)
    ap.add_argument("--align", default="mask", choices=["mask", "feature"])
    ap.add_argument("--ref_dir", default=None,
                    help="folder holding the reference images named in the landmark JSONs")
    ap.add_argument("--mask_dir", default=None,
                    help="optional dir of user masks (<stem>_mask.png), e.g. from SAM2/SAM3")
    ap.add_argument("--n_lm", type=int, default=60)
    ap.add_argument("--radius", type=float, default=4.0)
    # v51b: review artefacts for large batches
    ap.add_argument("--preview", default="original", choices=["original", "letterbox"],
                    help="draw the overlay on the untouched plate (true aspect ratio, no "
                         "padding) or on the 512x512 letterboxed tensor the model saw")
    ap.add_argument("--fig_per_page", type=int, default=12,
                    help="panels per preview page; 170 targets in one figure is unusable")
    ap.add_argument("--mirror_refs", action="store_true",
                    help="mirror-aware transfer: add a left-right flipped copy of every reference "
                         "(image flipped, landmark x mirrored, landmark numbers kept) and, per target, "
                         "keep only the handedness group whose references match best (share of "
                         "landmarks accepted). Needed when some specimens are photographed as mirror "
                         "images; the two handednesses are never mixed in one vote.")
    ap.add_argument("--rotate_refs", action="store_true",
                    help="v52 orientation search: add copies of every reference rotated 90/180/270 deg "
                         "(landmarks rotated with the image, numbers kept) and, per target, keep only "
                         "the orientation group whose references match best (share of landmarks "
                         "accepted). For specimens photographed turned or upside-down; combine with "
                         "--mirror_refs for all 8 rotation/flip combinations. 4x slower. Off by default: "
                         "image specimens in the references' orientation when you can.")
    ap.add_argument("--orientation_search", default="none", choices=["none", "rot4"],
                    help="same name as in SAM2-PAL: rot4 = --rotate_refs (off by default)")
    ap.add_argument("--emit_refs", action="store_true",
                    help="also write one COCO file per target under <outdir>/per_image/ and "
                         "list the fully-confirmed ones in refs_confirmed.txt, so a checked "
                         "batch can be fed straight back in via --landmarks")
    a = ap.parse_args()
    if a.orientation_search == "rot4":
        a.rotate_refs = True
    os.makedirs(a.outdir, exist_ok=True)

    global ALIGN_MODE
    ALIGN_MODE = a.align
    bb = Backbone(a.model, a.dim)
    print(f"model={a.model} patch={bb.patch} grid={bb.grid} "
          f"dim={bb.img_dim} layers={bb.n_layers} device={bb.device}")
    layers = [int(x) for x in a.layers.split(",") if int(x) <= bb.n_layers]

    # ---- batch transfer: N annotated references -> many unannotated targets -----
    if a.batch_glob:
        import glob as _glob
        LAYER = layers[0]
        ref_files = [x.strip() for x in a.landmarks.split(",") if x.strip()] if a.landmarks else []
        if not ref_files:
            raise SystemExit("--batch_glob requires --landmarks (comma-separated for multi-reference)")
        ref_dir = a.ref_dir or os.path.dirname(a.imgA)

        refs = []          # (name, feats, mask, pts_resized, slots)
        slot_ids = None
        for rf in ref_files:
            meta = json.load(open(rf))
            fname = meta["images"][0]["file_name"]
            ipath = os.path.join(ref_dir, fname)
            if not os.path.exists(ipath):
                ipath = a.imgA if os.path.basename(a.imgA) == fname else ipath
            if not os.path.exists(ipath):
                raise SystemExit(f"reference image not found for {rf}: {fname} (use --ref_dir)")
            raw = Image.open(ipath).convert("RGB")
            im, mk, (rs, rox, roy, rcx, rcy) = prepare_image(bb, raw, LAYER, ipath, a.mask_dir)
            pts_all, ids, vis = load_landmarks_coco(rf, raw.size[0], raw.size[1],
                                                    bb.img_dim,
                                                    lb=(rs, rox - rcx * rs, roy - rcy * rs))
            if slot_ids is None:
                slot_ids = ids
            elif ids != slot_ids:
                raise SystemExit(f"landmark numbering differs between references:\n"
                                 f"  {ref_files[0]}: {slot_ids}\n  {rf}: {ids}")
            present = [k for k, v in enumerate(vis) if v > 0]
            refs.append((os.path.basename(rf), bb.embed(im, LAYER, "none", a.facet),
                         mk, pts_all[present], present, im))
            print(f"reference {fname}: {len(present)}/{len(ids)} landmarks present")
        print(f"{len(refs)} reference(s); landmark ids {slot_ids}")

        # ---- mirror-aware mode: handedness of each reference, plus flipped copies ----
        # Handedness = sign of the signed area (shoelace) of the landmarks in their fixed
        # numbering order: a mirror image reverses it. A flipped reference keeps its landmark
        # numbers, so its votes stay in the same slots.
        def _handedness(pts):
            x, y = pts[:, 0], pts[:, 1]
            return 1 if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) >= 0 else -1
        ref_hand = [_handedness(r[3]) for r in refs]
        if a.mirror_refs:
            for (rname, _fa, mk, pts, present, im), hnd in list(zip(refs, ref_hand)):
                im_f = im.transpose(Image.FLIP_LEFT_RIGHT)
                pts_f = pts.copy()
                pts_f[:, 0] = bb.img_dim - pts_f[:, 0]
                refs.append((rname + " [flipped]", bb.embed(im_f, LAYER, "none", a.facet),
                             mk[:, ::-1].copy(), pts_f, present, im_f))
                ref_hand.append(-hnd)
            print(f"mirror-aware: {len(refs)} references "
                  f"({ref_hand.count(1)} one handedness, {ref_hand.count(-1)} the other)")
        two_hands = a.mirror_refs and len(set(ref_hand)) == 2

        # ---- v52 orientation search: rotated copies of every reference (incl. flipped ones) ----
        # DINOv3 patch tokens are not rotation-invariant, so a turned target matches a turned
        # reference far better than an upright one. The square canvas rotates exactly; a
        # clockwise quarter turn maps (x, y) -> (D - y, x). Rotation keeps handedness.
        ref_rot = [0] * len(refs)
        if a.rotate_refs:
            D = bb.img_dim
            _pil = {90: Image.ROTATE_270, 180: Image.ROTATE_180, 270: Image.ROTATE_90}   # PIL turns ccw
            for (rname, _fa, mk, pts, present, im), hnd in list(zip(refs, ref_hand)):
                for k in (90, 180, 270):
                    x, y = pts[:, 0], pts[:, 1]
                    pts_r = {90: np.c_[D - y, x], 180: np.c_[D - x, D - y], 270: np.c_[y, D - x]}[k]
                    im_r = im.transpose(_pil[k])
                    refs.append((f"{rname} [rot{k}]", bb.embed(im_r, LAYER, "none", a.facet),
                                 np.rot90(mk, -k // 90).copy(), pts_r, present, im_r))
                    ref_hand.append(hnd)
                    ref_rot.append(k)
            print(f"orientation search: {len(refs)} references (0/90/180/270 deg"
                  f"{' x 2 handednesses' if two_hands else ''})")
        rot_used_by_name = {}

        targets = sorted(_glob.glob(a.batch_glob))
        ref_names = {os.path.basename(r[0]).replace('_keypoints.json', '') for r in refs}
        targets = [t for t in targets if os.path.abspath(t) != os.path.abspath(a.imgA)]
        targets = targets[:a.batch_n]

        results = []
        for tp in targets:
            rawB = Image.open(tp).convert("RGB")
            W0, H0 = rawB.size
            imgB, maskB, (tscale, tox, toy, tcx, tcy) = prepare_image(
                bb, rawB, LAYER, tp, a.mask_dir)
            fb = bb.embed(imgB, LAYER, "none", a.facet)

            group_votes, group_score = {}, {}
            for gi, ((rname, fa, maskA_r, pts, present, _im), hnd, rot) in enumerate(zip(refs, ref_hand, ref_rot)):
                g = (hnd if two_hands else 0, rot)
                votes = group_votes.setdefault(g, {k: [] for k in range(len(slot_ids))})
                A = None
                if a.align == "feature":
                    A = feature_affine(fa, fb, maskA_r, maskB, bb.patch, bb.grid)
                if A is None:
                    A = pca_affine(maskA_r, maskB)
                r = transfer(fa, fb, pts, A, bb.patch, bb.grid, radius_patches=a.radius)
                group_score.setdefault(g, []).append(float(np.mean(r.accepted)))
                for j, slot in enumerate(present):
                    votes[slot].append((r.pred[j], bool(r.accepted[j])))
            # RANSAC fits rotation/scale/shift only, never a reflection, so references of the
            # wrong handedness get most landmarks rejected: the higher mean accepted share wins
            chosen = max(group_votes, key=lambda g: np.mean(group_score[g]))
            votes = group_votes[chosen]
            hand_used = chosen[0] if two_hands else None
            if a.rotate_refs:
                rot_used_by_name[os.path.basename(tp).rsplit('.', 1)[0]] = chosen[1]

            # ---- consensus: median of the votes, spread = agreement ------------
            cons = {}
            for slot, vs in votes.items():
                if not vs:
                    cons[slot] = (None, 0, 0, 0.0); continue
                acc = [xy for xy, ok in vs if ok]
                use = acc if acc else [xy for xy, _ in vs]
                arr = np.array(use)
                med = np.median(arr, axis=0)
                spread = float(np.median(np.hypot(*(arr - med).T))) if len(arr) > 1 else 0.0
                cons[slot] = (med, len(acc), len(vs), spread)

            results.append((os.path.basename(tp).rsplit('.', 1)[0], imgB, maskB,
                            cons, (W0, H0), os.path.basename(tp),
                            (tscale, tox, toy, tcx, tcy), tp, hand_used))
            agree = sum(1 for k in cons if cons[k][0] is not None and cons[k][1] > 0)
            print(f"{os.path.basename(tp)[:44]:<46} confirmed {agree}/{len(slot_ids)}")

        # ---- write ONE combined COCO file in ORIGINAL pixel coordinates -------
        # Descriptron's prediction viewer walks the `images` list of a single JSON
        # (load_image_list -> prediction_image_names -> load_prediction_image), so
        # per-image files cannot drive it. One file, many images.
        TIGHT = bb.patch * 1.5      # spread below this = references agree
        coco = {"images": [], "annotations": [],
                "categories": [{"id": 1, "name": "landmarks", "supercategory": "none",
                                "keypoints": [f"kp{sid}" for sid in slot_ids],
                                "skeleton": []}]}
        ann_id = 1
        per_image_dir = os.path.join(a.outdir, "per_image")
        if a.emit_refs:
            os.makedirs(per_image_dir, exist_ok=True)
        confirmed_refs = []
        anns_by_name = {}                     # name -> annotation, for the preview
        for nm, imgB, mkB, cons, (W0, H0), fname, (tscale, tox, toy, tcx, tcy), tp, hand_used in results:
            kp, nvis, detail = [], 0, []
            xs, ys = [], []
            for k, sid in enumerate(slot_ids):
                med, n_acc, n_tot, spread = cons[k]
                if med is None or n_tot == 0:
                    kp += [0, 0, 0]
                    detail.append({"id": sid, "v": 0, "n_confirmed": 0,
                                   "n_refs": 0, "spread_px": None})
                    continue
                v = 2 if (n_acc * 2 >= n_tot and n_acc > 0 and spread <= TIGHT) else 1
                X = (float(med[0]) - tox) / tscale + tcx    # undo letterbox + crop
                Y = (float(med[1]) - toy) / tscale + tcy
                kp += [X, Y, v]; xs.append(X); ys.append(Y); nvis += 1
                detail.append({"id": sid, "v": v, "n_confirmed": n_acc, "n_refs": n_tot,
                               "spread_px": round(spread / tscale, 1)})
            bbox = ([min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    if xs else [0, 0, 0, 0])
            img_entry = {"id": nm, "file_name": fname, "height": H0, "width": W0}
            ann = {"id": ann_id, "image_id": nm, "category_id": 1,
                   "keypoints": kp, "num_keypoints": nvis,
                   "area": max(1.0, bbox[2] * bbox[3]),
                   "bbox": bbox, "iscrowd": 0,
                   "point_order": slot_ids,
                   "n_references": len(refs),
                   "handedness_used": hand_used,
                   "orientation_used": rot_used_by_name.get(nm),
                   "landmark_detail": detail}
            coco["images"].append(img_entry)
            coco["annotations"].append(ann)
            anns_by_name[nm] = ann
            ann_id += 1

            # ---- per-image file, and promotion of clean ones to references ----
            # A target is promotable only if EVERY landmark the references could
            # supply came back v=2; a v=1 slot is exactly the case a human must
            # look at, and feeding it back would launder a guess into a reference.
            if a.emit_refs:
                one = {"images": [img_entry], "annotations": [dict(ann, id=1)],
                       "categories": coco["categories"]}
                dst = os.path.join(per_image_dir, nm + "_keypoints_pred.json")
                json.dump(one, open(dst, "w"), indent=1)
                usable = [d for d in detail if d["n_refs"] > 0]
                if usable and all(d["v"] == 2 for d in usable):
                    confirmed_refs.append(dst)
        combined = os.path.join(a.outdir, "predictions_keypoints.json")
        json.dump(coco, open(combined, "w"), indent=1)
        if a.emit_refs:
            with open(os.path.join(a.outdir, "refs_confirmed.txt"), "w") as fh:
                fh.write("\n".join(confirmed_refs) + ("\n" if confirmed_refs else ""))

        # ---- preview pages ----------------------------------------------------
        # One 170-panel figure is both unreadable and past matplotlib's pixel
        # ceiling, so pages of --fig_per_page.  --preview original draws on the
        # untouched plate: true aspect ratio, no letterbox padding, and it checks
        # the back-transform to original pixels at the same time.
        per_page = max(1, a.fig_per_page)
        pages = [results[i:i + per_page] for i in range(0, len(results), per_page)]
        n_pages = len(pages)
        written = []
        for pi, page in enumerate(pages, 1):
            n = len(page); cols = min(3, n); rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.2 * rows), dpi=150)
            axes = np.atleast_1d(axes).ravel()
            for ax_, (nm, imgB, mkB, cons, (W0, H0), _fn, _lb, tp, _hand) in zip(axes, page):
                if a.preview == "original":
                    ax_.imshow(Image.open(tp).convert("RGB"))
                    kpx = anns_by_name[nm]["keypoints"]
                    pts = [(kpx[3 * k], kpx[3 * k + 1], kpx[3 * k + 2])
                           for k in range(len(slot_ids))]
                    off = max(4.0, 0.008 * max(W0, H0))
                else:
                    ax_.imshow(imgB)
                    pts, off = [], 6.0
                    for k in range(len(slot_ids)):
                        med, n_acc, n_tot, spread = cons[k]
                        if med is None or n_tot == 0:
                            pts.append((0, 0, 0)); continue
                        v = 2 if (n_acc * 2 >= n_tot and n_acc > 0
                                  and spread <= TIGHT) else 1
                        pts.append((float(med[0]), float(med[1]), v))
                for k, sid in enumerate(slot_ids):
                    X, Y, v = pts[k]
                    if v == 0:
                        continue
                    ok = v == 2
                    col = C_OK if ok else C_FLAG
                    ax_.scatter([X], [Y], s=30, c=col if ok else "none",
                                edgecolors="white" if ok else col, linewidths=1.2)
                    ax_.text(X + off, Y - off, str(sid), fontsize=7,
                             weight="bold", color=col)
                ax_.set_title(nm[:30], size=7); ax_.axis("off")
            for ax_ in axes[len(page):]: ax_.axis("off")
            # Same colour language as Descriptron: green v=2, goldenrod v=1.  v=0
            # (no reference supplied the landmark) is not drawn here — it has no
            # position to draw; in the GUI it appears as a red placeholder to drag.
            fig.suptitle(f"Multi-reference consensus ({len(refs)} references) — "
                         f"green = agreed, goldenrod = check"
                         + (f"  [page {pi}/{n_pages}]" if n_pages > 1 else ""),
                         size=10, weight="bold")
            fig.patch.set_facecolor(SURF); fig.tight_layout()
            out_png = os.path.join(a.outdir, "fig4_batch_transfer.png" if n_pages == 1
                                   else f"fig4_batch_transfer_p{pi:02d}.png")
            fig.savefig(out_png); plt.close(fig)
            written.append(os.path.basename(out_png))
        print(f"\nwrote predictions_keypoints.json — {len(coco['images'])} images, "
              f"{len(coco['annotations'])} annotations, original pixel coords")
        print(f"       + {len(written)} preview page(s) ({a.preview} frame): "
              f"{written[0]}{' ... ' + written[-1] if len(written) > 1 else ''}")
        if a.emit_refs:
            print(f"       + per_image/ ({len(results)} files), "
                  f"refs_confirmed.txt ({len(confirmed_refs)} fully-confirmed)")
        return

    imgA, maskA = load_pair(a.imgA, a.maskA, bb.img_dim)

    if a.selftest:
        rows, (iA, iB, lm, truth, A) = run_selftest(bb, imgA, maskA, layers,
                                                    a.facet, a.outdir, a.n_lm, a.radius)
        fig_aug(rows, a.outdir, bb.patch)
        # run_selftest only evaluates aug="none" now; summarise whatever aug the rows
        # actually carry instead of the stale "fixed" filter (which left min() empty)
        aug_used = rows[0]["aug"]
        bestl = min({r["layer"] for r in rows},
                    key=lambda L: min(r["median_err"] for r in rows
                                      if r["layer"] == L and r["aug"] == aug_used))
        fig_stages(rows, a.outdir, bestl, bb.patch)
        fa = bb.embed(iA, bestl, aug_used, a.facet)
        fb = bb.embed(iB, bestl, aug_used, a.facet)
        r = transfer(fa, fb, lm, A, bb.patch, bb.grid, radius_patches=a.radius)
        fig_transfer(iA, iB, lm, r.pred, r.accepted, a.outdir, "selftest", truth)
        print(f"\nbest layer = {bestl}")

    if a.imgB:
        imgB, maskB = load_pair(a.imgB, a.maskB, bb.img_dim)
        A = pca_affine(maskA, maskB)
        rng = np.random.default_rng(SEED)
        lm = sample_landmarks(maskA, a.n_lm, bb.patch, rng)
        best = None
        for layer in layers:
            fa = bb.embed(imgA, layer, "fixed", a.facet)
            fb = bb.embed(imgB, layer, "fixed", a.facet)
            r = transfer(fa, fb, lm, A, bb.patch, bb.grid, radius_patches=a.radius)
            print(f"cross-species  L{layer:<3} cycle-consistency {r.cycle_rate:.2f}  "
                  f"retained {r.accepted.mean():.2f}")
            if best is None or r.cycle_rate > best[1].cycle_rate:
                best = (layer, r, imgB)
        layer, r, imgB = best
        nm = os.path.basename(a.imgB).split("_")[0] + "_L" + str(layer)
        fig_transfer(imgA, imgB, lm, r.pred, r.accepted, a.outdir, nm)
        json.dump(dict(layer=layer, cycle=r.cycle_rate,
                       retained=float(r.accepted.mean())),
                  open(os.path.join(a.outdir, "cross_species.json"), "w"), indent=1)
        print(f"\nbest cross-species layer = {layer}")


if __name__ == "__main__":
    main()
