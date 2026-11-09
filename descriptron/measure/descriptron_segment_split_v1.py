#!/usr/bin/env python3
"""
descriptron_segment_split_v1.py — split COARSE part masks into FINE homologous parts (DINOv3: L23 says what, L18 says where)
===========================================================================================================================

A patch-token model (DINOMask) learns a few COARSE classes well from one or two hand-labelled specimens: head, pronotum,
elytra, abdomen, leg, antenna, mouthparts, eye, plus 'body' = the specimen not covered by any part. Those classes come from
the last block of the frozen DINOv3 ViT-L/16 (L23), which knows WHAT a patch is (its homology) but places outlines only to
about a patch (16 px). This program keeps L23's homology and turns the coarse masks into FINE parts with sharp outlines:

    elytra  -> left_elytron / right_elytron                 abdomen -> tergite 3 ... tergite 8 (+ tergite 9 / terminalia)
    leg     -> left_profemur, right_mesotibia, ...           antenna -> left_antenna / right_antenna (+ left_mandible ...)
    body    -> scutellum / mesonotum / unassigned            mouthparts -> labrum / left_maxillary_palp / ... or mouthparts
    eye     -> left_eye / right_eye (found again from L18)   head, pronotum, ... kept, with sharpened outlines
    (antenna also -> left_/right_mandible where the coarse model put the mandibles into 'antenna')

How it works, per image
  1. Foreground. Every coarse part is intersected with a specimen foreground mask, so the background a patch model fills
     between appendages (the window between an antenna and a foreleg, between a leg and the pronotum) is removed; every
     covered background piece of at least --window_min_px is reported as a window. --fg_masks DIR reads <stem>_mask.png
     per image (as descriptron_plate_split_v1.py --mask writes; --fg_stem_regex maps image stems to mask stems);
     otherwise the foreground is the colour distance from the border colour (CIE L*a*b*, Otsu threshold with a floor,
     --fg_min_dist, and hysteresis down to --fg_weak_frac x that threshold for pale appendages attached to the body).
     Holes are filled only when they are PINHOLES (< --pinhole_frac x image area) or glare (rim >= 90% in one
     non-appendage part); every other enclosed hole is a window and stays background. The coarse masks are painted large
     to small (small parts on top, as DINOMask paints its training labels), 'body' below everything; thin uncovered
     foreground (setae, at most --fringe_px half width) joins the part it grows from, the rest becomes 'body'. Specks
     (below --min_piece_frac of the foreground, or a 'body' piece whose rim lies >= 85% in one part) join a neighbour.
  2. Boundary maps. Frozen DINOv3 ViT-L/16, block L = hidden_states[L + 1] (as biorag_probe_panels_v2.py and
     dinomask_hypercolumn_probe_v1.py number it), tiled; the image is enlarged --token_scale x (default 2: 8 native px per
     patch) and read in four passes shifted by half a patch, so the token grid is 4 native px. HIGH-PASS = token minus
     the mean of its foreground neighbours in a 7 x 7-patch window (probe panels v2), reduced to --hp_pca whitened PCA
     components per image. Two STRUCTURE TENSORS per layer (L18, L23), so that the strength ACROSS any direction n can be
     read off (sqrt(n' J n)): EDGE = first differences of neighbouring tokens (a step between two sclerites; used for the
     sharpening) and LINE = second differences within one shifted pass (a token on a thin line differs from both
     neighbours across it; used for the suture, intersegmental lines and joints). Differences touching a background token
     (< --token_fg foreground) are not counted. The photograph gives the same tensor at full resolution (Gaussian
     derivatives, sigma 1 px), added with --image_weight (sharpening) and --split_image_weight (splits). Each map is
     scaled to a 95th percentile of 1 on the specimen. Why these choices (Oxytelus hand labels, the 6 lines between its 7
     abdominal segments): the top 6 peaks of the along-axis profile hit 6/6 lines for the L18 line map in the lateral
     margin bands at token scale 2, 4/6 at native scale, 0/6 across the middle of the abdomen (basal carinae inside each
     tergite) and <= 2/6 for any first-difference map; adding L23 at the margins lowered it to 4/6. --maps_dir keeps
     <stem>_segsplit_maps.npz so the geometry can be rerun without a GPU; only this step needs torch.
  3. Sharpening (--sharpen_layer L18, default; L23 or none to compare). Marker-controlled watershed: every coarse part
     keeps its core (pixels farther than --band_px from any boundary with another part; pieces thinner than that keep
     their medial pixels); the band is flooded from the cores in order of boundary strength, inside the foreground only.
     Why a watershed: L23 already says which part each region is, and its error is a shift of the boundary by up to
     about a patch; flooding from the cores can move a boundary only within that band, onto the strongest ridge, and can
     neither invent nor lose a part. A CRF or graph cut would need weights tuned per class; a random walker needs
     scikit-image. The specimen outline itself comes from the foreground.
  4. Body frame (descriptron_serial_homology_v1.body_frame, imported: principal axis of the trunk = foreground minus the
     appendages, anterior end from the head, midline from the trunk's edges slice by slice). Side rule as there: with the
     head up, the specimen's left is on the image's left in DORSAL view and on the image's right in VENTRAL view. The
     view is per image: an image field (--view_field), else --view_regex on the file name, else --view (serial
     homology's image_view, imported). Lateral views: nothing is split by side (warning).
  5. Splits. All cuts are CURVES: dynamic programming (Viterbi) finds the path of greatest line strength through a strip
     of the part straightened along the body axis, one pixel step at a time, then maps it back.
     a. elytra -> left/right: the suture = the strongest across-axis line within --suture_window x the elytra's half
        width of the midline; its mean strength must reach --suture_contrast x the median in that window, else the
        midline itself is used ('rule fallback'). Two separate elytra (spread apart) are split by their pieces.
     b. abdomen -> segments numbered from the anterior (--first_segment, --last_regular_segment, --segment_name; ventral
        views --segment_name_ventral). Candidates = peaks of the along-axis line strength in the two lateral MARGIN bands
        (--margin_band of the width; by eye: L23 shows the intersegmental nodes on the margins, L18 the lines), as a
        z-score; --mid_weight, --notch_weight and --l23_margin_weight add the middle band, the outline notch and L23
        (default 0, see 2.). Each candidate is traced across as a curve (within --cut_bend x the expected segment length)
        and placed on the photograph's gradient within --refine_px. Setal and colour bands cross the abdomen as lines
        too, but cut a segment into a short piece and a long one, so the cuts are CHOSEN jointly (dynamic programming
        over candidate subsets): sum of (score - --cut_threshold), minus --ratio_weight x the squared log ratio of
        neighbouring segment lengths, minus --missing_weight per regular segment fewer than expected, no piece shorter
        than --min_segment_frac x the expected length, at most the expected number of regular segments. Every candidate
        and the reason it was taken or not are in <out>_cuts.csv. APEX (--apex auto): the terminalia behind the last
        regular segment are an optional extra piece that must look like one: mean width < --apex_width_frac x the
        regular width and < --apex_prev_frac x the segment before it (tergite VIII tapers too), at least --apex_min_frac
        x the expected length, and an abrupt narrowing (--apex_step), separate lobes (--apex_lobes) or a very narrow
        shape (--apex_narrow), so a protruding apex never becomes a regular segment that shifts the numbers. It is named
        --apex_name (default the segment after --last_regular_segment, 'tergite 9'); separate median pieces behind it
        are numbered on (tergite 10); paired lateral pieces, and an apex whose boundary is weaker than --apex_min_line,
        are named --terminalia_name with a warning. apex_present yes/no per specimen (CSV); sex is not inferred.
        --apex none: no apex; the last regular segment runs to the tip (Oxytelus, 'abdominal segment 1-7').
     c. body leftovers: a piece between the pronotum and the elytra, touching both, on the midline and narrowing
        posteriorly -> 'scutellum'; a transverse piece across the midline there -> 'mesonotum'; any other exposed piece
        -> 'unassigned' (warning, with where it lies).
     d. leg -> coxa / trochanter / femur / tibia / tarsus (--leg_parts_dorsal, --leg_parts_ventral, with expected length
        fractions). Each leg piece is unrolled by geodesic distance from its base (the leg pixel nearest the trunk's
        CORE: a leg lying along the body touches its outline over a long stretch); per 1-px step the joint score = along-
        leg line strength (z, clipped) + --leg_width_weight x the narrowing after a wider stretch + --leg_bend_weight x
        the bend of the centre line; joints are chosen jointly with the length fractions as a prior, and an expected
        position is used where no candidate fits ('rule fallback'). Names and sides come from
        descriptron_serial_homology_v1.run (imported; --leg_name_format prefix -> left_profemur).
     e. antenna -> left_/right_antenna by the side of its base. Mandibles inside the coarse antenna class (Platydracus:
        the sickles; --mandible_from_antenna auto = only when the input has no mandible class): a broad piece (wider than
        --mandible_width x the antenna's radius) at the anterior corner of the head, either hanging off the antenna's
        base (the antenna = the geodesic path from its tip to the head plus its width; the line between them by
        watershed) or a short piece of its own; sided by its bulk. Coarse 'mandible' masks are sided by their pieces.
        --rename 'mandibles=mouthparts' renames input classes first (the Oxytelus class 'mandibles' lands on the labrum
        and maxillary palps of Platydracus).
     f. mouthparts -> labrum (median piece at the head's anterior margin) and left_/right_maxillary_palp (paired lateral
        pieces, elongation >= --palp_elongation) by a three-marker watershed; otherwise 'mouthparts' with a warning.
     g. eyes from L18: the head's own outline (head + eye + antenna + mandible opened by --eye_open_frac x its half
        width, so the antennae fall away and the bulging eyes stay) is clustered on its L18 tokens (PCA, k-means, k =
        2..5); the cluster whose lateral pieces look most like a PAIR of eyes wins: compact (--eye_min_compact), oval
        (--eye_max_elongation), --eye_min_area..--eye_max_area of the head, at the lateral outline, absent from the
        medial head, similar on both sides, far from the medial head in L18 (--eye_distinct_weight), near the coarse eye
        (--eye_prior_weight) and in front of the temples (--eye_position anterior, the default for dorsal Staphylinidae).
        Each eye is sided and sharpened. Where L18 finds none on a side, a coarse 'eye' piece is kept only when coherent
        and at the lateral head margin (warning), else dropped. Eye confidence is never 'high' (a cluster rule).
  6. Output: a NEW COCO json (polygons, as DINOMask v6 writes predictions_polygons.json: pieces >= 10% of the largest,
     outer outlines, simplified to --simplify_px) that the Descriptron editor opens; <out>_parts.csv (one row per part:
     source coarse part, method, confidence, evidence, warnings, apex_present); <out>_cuts.csv (every candidate cut);
     <out>_images.csv (per image: windows removed, segments, apex_present, scutellum, eyes, unassigned pieces); QC PNGs
     (--qc_dir): photo + coarse outlines + removed windows (red) | boundary map with accepted (green) and rejected (red)
     cuts, suture (blue), joints (orange) | fine parts. --gt FILE (+ --gt_catmap raw->coarse) scores the coarse,
     sharpened and fine masks against hand labels: IoU and boundary F at 2 and 4 px, on the whole outline and on the
     internal part-to-part boundaries only (<out>_eval.csv).

Validation (Oxytelus jessoensis, GUI-corrected dorsal labels; one image). Split only (hand labels merged to coarse
classes, --sharpen_layer none): left/right elytron IoU 0.96 (BF2 0.88), abdominal segments 1-7 all numbered right, mean
IoU 0.87 (BF4 0.81; cuts within about 4 px), 16/18 leg parts IoU >= 0.5 (mean 0.71; the misses are two tarsi), antennae
0.85. End to end on a DINOMask L23 coarse prediction (trained on this same image, so optimistic): sharpening raised the
coarse internal boundary F (2 px) from 0.07 to 0.33 (L18) / 0.33 (L23) and mean IoU from 0.62 to 0.67 / 0.68; fine
parts: L23-only 0.64 mean IoU, L23 + L18 0.68 (segments 0.79 -> 0.86, elytra 0.86 -> 0.91, legs 13 -> 15/18 IoU >= 0.5).
The L18 eye rule failed on Oxytelus (small eyes; it took the temples).

Known failure modes. (i) A coarse error larger than --band_px cannot be fixed by sharpening; a thin femur lying along the
pronotum can be handed to the pronotum. (ii) Intersegmental lines that are weak at the margins (dense lateral setae)
can be missed, after which the posterior segments are numbered too low and an apex can be numbered as the last regular
segment (Platydracus basicornis); strong setal or colour-band edges are taken only when the length prior allows. (iii)
Legs that touch each other or an antenna form one piece (warning 'branched') and are split as one leg; legs cut by the
image border or hidden give the length prior a wrong total length. (iv) Mandible, mouthpart and eye splits are shape
rules on top of the coarse classes: check them (eyes: right on 2 of 5 Platydracus dorsal templates, partial on the rest).

Examples
    python descriptron_segment_split_v1.py coarse.json --images imgs/ --fg_masks masks/ --fg_stem_regex '^\\d+_(.*)$' \\
        --view_regex '_(dorsal|ventral|lateral)' --rename mandibles=mouthparts --first_segment 3 \\
        --maps_dir maps/ --out fine.json --qc_dir qc/
    python descriptron_segment_split_v1.py coarse.json --images imgs/ --view dorsal --first_segment 1 \\
        --last_regular_segment 7 --apex none --segment_name 'abdominal segment {n}' --out fine.json \\
        --gt hand.json --gt_catmap catmap_coarse.json
    python descriptron_segment_split_v1.py coarse.json --images imgs/ --maps_dir maps/ --maps_only   # GPU step only
"""
from __future__ import annotations

import argparse
import copy
import csv
import heapq
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))
import descriptron_serial_homology_v1 as sh  # noqa: E402

VERSION = "segment_split_v1"
Image.MAX_IMAGE_PIXELS = None
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
# coarse roles and their default category patterns (case-insensitive, matched after --rename)
ROLE_DEFAULTS = {
    "head": r"^\s*head\s*$",
    "pronotum": r"pronot",
    "elytra": r"^\s*elytr(a|on)\s*$",
    "abdomen": r"^\s*(abdomen|gaster|metasoma)\s*$",
    "leg": r"^\s*legs?\s*$",
    "antenna": r"^\s*antenna[e]?\s*$",
    "mandible": r"^\s*mandibles?\s*$",
    "mouthparts": r"^\s*(mouth[\s_]*parts?|palps?)\s*$",
    "eye": r"^\s*eyes?\s*$",
    "body": r"^\s*body\b|habitus",
}
APPENDAGES = ("leg", "antenna", "mandible", "mouthparts")
SINGLE = ("head", "pronotum", "elytra", "abdomen")        # one piece per specimen (extra pieces are specks)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(path):
    with open(path) as f:
        return json.load(f)


def save(d, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f)


# ------------------------------------------------------------------ small mask tools
def edt(m: np.ndarray) -> np.ndarray:
    return ndimage.distance_transform_edt(m)


def dilate(m: np.ndarray, r: float) -> np.ndarray:
    if r <= 0 or not m.any():                                 # (an EDT with nothing to measure from is undefined)
        return m.copy()
    return edt(~m) <= r


def erode(m: np.ndarray, r: float) -> np.ndarray:
    return edt(m) > r if r > 0 else m.copy()


def label8(m: np.ndarray):
    return ndimage.label(m, structure=np.ones((3, 3), bool))


def ring(m: np.ndarray, r: float = 2.0) -> np.ndarray:
    return dilate(m, r) & ~m


def pca_elongation(m: np.ndarray) -> float:
    ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return 1.0
    ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys])))
    return float(math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))


def otsu(v: np.ndarray, bins: int = 256) -> float:
    v = v[np.isfinite(v)]
    if not len(v) or v.max() <= v.min():
        return float(v.max()) if len(v) else 0.0
    h, e = np.histogram(v, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    k = int(np.argmax(w0 * w1 * (m0 - m1) ** 2))
    return float(c[k])


def robust_z(x, ref) -> np.ndarray:
    ref = np.asarray(ref, float); ref = ref[np.isfinite(ref)]
    med = float(np.median(ref)) if len(ref) else 0.0
    mad = float(np.median(np.abs(ref - med))) * 1.4826 if len(ref) else 1.0
    return (np.asarray(x, float) - med) / max(mad, 1e-6)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (uint8) -> CIE L*a*b* (D65)."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = (c @ M.T) / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


# ------------------------------------------------------------------ names and roles
def parse_rename(spec: str) -> Dict[str, str]:
    """'mandibles=mouthparts, foo=bar' -> {'mandibles': 'mouthparts', ...} (case-insensitive keys)."""
    out = {}
    for item in (spec or "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            if k.strip():
                out[k.strip().lower()] = v.strip()
    return out


def role_matcher(opts):
    rx = {r: re.compile(getattr(opts, f"{r}_regex") or ROLE_DEFAULTS[r], re.I) for r in ROLE_DEFAULTS}

    def role(name: str) -> Optional[str]:
        for r in ("body", "head", "pronotum", "elytra", "abdomen", "leg", "antenna", "mandible", "mouthparts", "eye"):
            if rx[r].search(name or ""):
                return r
        return None
    return role


def sided(fmt: str, side: str, part: str) -> str:
    return fmt.format(side=side, part=part)


def canon(name: str) -> str:
    """Comparable key for a part name: 'profemur (left)' / 'left_profemur' -> 'left_profemur'; 'antenna (left)' ->
    'left_antenna'; 'elytra'/'elytron', 'mandibles', 'eyes' singular; anything else lower case."""
    n = (name or "").strip()
    p = sh.parse_name(n)
    if p and p[0]:
        return f"{p[0]}_{p[1]}{p[2]}"
    m = re.match(r"^\s*(.+?)\s*\(\s*(left|right)\s*\)\s*$", n, re.I) or None
    side, base = (m.group(2).lower(), m.group(1)) if m else (None, n)
    m2 = re.match(r"^\s*(left|right)[\s_\-]+(.+)$", base, re.I)
    if m2 and side is None:
        side, base = m2.group(1).lower(), m2.group(2)
    base = re.sub(r"\s+", " ", base.strip().lower())
    base = {"elytra": "elytron", "mandibles": "mandible", "eyes": "eye", "antennae": "antenna"}.get(base, base)
    return f"{side}_{base}" if side else base


# ------------------------------------------------------------------ foreground
def pinhole_px(shape, opts) -> int:
    return max(4, int(round(opts.pinhole_frac * shape[0] * shape[1])))


def fill_pinholes(fg: np.ndarray, max_px: int, labels: Optional[np.ndarray] = None,
                  glare_ok=None) -> Tuple[np.ndarray, int, int]:
    """Fill holes of the foreground that are pinholes (< max_px) or glare (rim >= 90% in one part for which glare_ok(label)
    is True). -> (filled fg, pinholes filled, glare holes filled). Every other hole is a window and stays background."""
    holes = ndimage.binary_fill_holes(fg) & ~fg
    hl, n = ndimage.label(holes)
    if not n:
        return fg, 0, 0
    sizes = np.bincount(hl.ravel())[1:]
    out = fg.copy(); n_pin = n_glare = 0
    for k, sl in enumerate(ndimage.find_objects(hl), 1):
        if sizes[k - 1] < max_px:
            out[hl == k] = True; n_pin += 1; continue
        if labels is None or glare_ok is None:
            continue
        y0, y1 = max(0, sl[0].start - 4), sl[0].stop + 4; x0, x1 = max(0, sl[1].start - 4), sl[1].stop + 4
        h = hl[y0:y1, x0:x1] == k
        rim = ring(h, 3.0) & fg[y0:y1, x0:x1]
        lv = labels[y0:y1, x0:x1][rim]
        lv = lv[lv > 0]
        if len(lv):
            top, cnt = Counter(lv.tolist()).most_common(1)[0]
            if cnt >= 0.9 * len(lv) and glare_ok(top):
                out[y0:y1, x0:x1][h] = True; n_glare += 1
    return out, n_pin, n_glare


def derive_fg(rgb: np.ndarray, opts) -> np.ndarray:
    """Specimen foreground from a plain background: colour distance (L*a*b*) from the border median, Otsu threshold with a
    floor (--fg_min_dist), opened/closed, the largest piece plus pieces near it (detached legs). Holes are NOT filled here."""
    lab = rgb_to_lab(rgb)
    h, w = lab.shape[:2]
    b = max(2, int(0.04 * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3),
                             lab[:, -b:].reshape(-1, 3)])
    d = np.linalg.norm(lab - np.median(border, 0), axis=-1)
    t_strong = max(otsu(np.clip(d, 0, np.percentile(d, 99.9))), opts.fg_min_dist)
    strong = ndimage.binary_opening(d > t_strong, np.ones((3, 3), bool))
    # hysteresis: pale parts (yellow tarsi, palps on white) are weakly different from the background but attached to
    # the strongly different body
    weak = d > max(opts.fg_min_dist, opts.fg_weak_frac * t_strong)
    wl, _ = label8(weak)
    keep_l = np.unique(wl[strong]); keep_l = keep_l[keep_l > 0]
    m = np.isin(wl, keep_l) | strong
    m = ndimage.binary_opening(m, np.ones((2, 2), bool)) | strong
    k = max(1.0, 0.002 * max(h, w))
    m = erode(dilate(m, k), k) | m                          # closing by a small disc
    lab_, n = label8(m)
    if n > 1:
        sizes = np.bincount(lab_.ravel())[1:]
        big = int(np.argmax(sizes)) + 1
        sl = ndimage.find_objects(lab_)
        bs = sl[big - 1]; py, px = int(0.08 * (bs[0].stop - bs[0].start)), int(0.08 * (bs[1].stop - bs[1].start))
        keep = np.zeros(n + 1, bool); keep[big] = True
        for i, s in enumerate(sl, 1):
            near = (s[0].stop >= bs[0].start - py and s[0].start <= bs[0].stop + py and
                    s[1].stop >= bs[1].start - px and s[1].start <= bs[1].stop + px)
            if i != big and near and sizes[i - 1] >= 0.002 * sizes[big - 1]:
                keep[i] = True
        m = keep[lab_]
    return m


def load_fg_mask(path: Path, shape) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L")) > 127
    if m.shape != tuple(shape):
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) > 127
    return m


def fg_mask_path(file_name: str, opts) -> Optional[Path]:
    if not opts.fg_masks:
        return None
    stem = Path(file_name).stem
    if opts.fg_stem_regex and (m := re.search(opts.fg_stem_regex, stem)):
        stem = m.group(1) if m.groups() else m.group(0)
    p = Path(opts.fg_masks) / f"{stem}{opts.fg_suffix}"
    return p if p.is_file() else None


# ------------------------------------------------------------------ coarse label map
class Labels:
    """Label map of one image: lab (int16, 0 = background), names[k], roles[k]."""

    def __init__(self, lab: np.ndarray, names: List[str], roles: List[Optional[str]]):
        self.lab, self.names, self.roles = lab, names, roles

    def copy(self) -> "Labels":
        return Labels(self.lab.copy(), list(self.names), list(self.roles))

    def idx(self, role: str) -> List[int]:
        return [k for k, r in enumerate(self.roles) if r == role and k > 0]

    def mask(self, role: str) -> np.ndarray:
        ks = self.idx(role)
        return np.isin(self.lab, ks) if ks else np.zeros(self.lab.shape, bool)

    def add(self, name: str, role: Optional[str]) -> int:
        self.names.append(name); self.roles.append(role)
        return len(self.names) - 1


def build_labels(masks: Dict[str, np.ndarray], fg: np.ndarray, role_of, opts) -> Labels:
    """Paint the coarse masks large to small (small parts on top), 'body' first (lowest); keep foreground only; foreground
    no part covers becomes 'body'."""
    names, roles = ["background"], [None]
    order = sorted(masks, key=lambda n: (role_of(n) != "body", -int(masks[n].sum())))
    lab = np.zeros(fg.shape, np.int16)
    for n in order:
        names.append(n); roles.append(role_of(n))
        lab[masks[n]] = len(names) - 1
    lab[~fg] = 0
    L = Labels(lab, names, roles)
    left = fg & (lab == 0)
    if left.any() and opts.fringe_px > 0 and (lab > 0).any():
        # thin uncovered foreground (setae, the fringe of hairs the specimen mask keeps) joins the part it grows from
        cl, n = label8(left)
        thick = ndimage.maximum(edt(np.pad(left, 1))[1:-1, 1:-1], cl, index=np.arange(1, n + 1)) if n else []
        thin = np.zeros(n + 1, bool); thin[1:] = np.asarray(thick) <= opts.fringe_px
        fr = thin[cl] & left
        if fr.any():
            _, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
            lab[fr] = lab[iy[fr], ix[fr]]
            left = fg & (lab == 0)
    if left.any():
        ks = L.idx("body")
        k = ks[0] if ks else L.add(opts.body_name, "body")
        L.lab[left] = k
    return L


def neighbour_label(lab: np.ndarray, piece: np.ndarray, exclude=(0,)) -> Tuple[Optional[int], float]:
    """Label sharing the longest border with a piece -> (label, share of the rim) (background excluded)."""
    sl = ndimage.find_objects(piece.astype(np.int8))[0]
    y0, y1 = max(0, sl[0].start - 3), sl[0].stop + 3; x0, x1 = max(0, sl[1].start - 3), sl[1].stop + 3
    p = piece[y0:y1, x0:x1]
    v = lab[y0:y1, x0:x1][ring(p, 1.5)]
    v = v[~np.isin(v, exclude)]
    if not len(v):
        return None, 0.0
    k, c = Counter(v.tolist()).most_common(1)[0]
    return int(k), c / len(v)


def absorb_specks(L: Labels, fg_area: int, opts) -> List[str]:
    """Pieces below --min_piece_frac x foreground (single-instance parts: also below 10% of their largest piece) join the
    part they touch most; a 'body' piece whose rim lies >= 85% in one part joins that part (it is enclosed by it)."""
    notes = []
    min_px = opts.min_piece_frac * fg_area
    for k in range(1, len(L.names)):
        m = L.lab == k
        if not m.any():
            continue
        cl, n = label8(m)
        if n == 0:
            continue
        sizes = np.bincount(cl.ravel())[1:]
        big = sizes.max()
        for i in range(1, n + 1):
            sz = sizes[i - 1]
            role = L.roles[k]
            small = sz < min_px or (role in SINGLE and sz < 0.1 * big)
            piece = cl == i
            if role == "body":
                nb, share = neighbour_label(L.lab, piece, exclude=(0, k))
                if nb is not None and (share >= 0.85 or (sz < min_px and share >= 0.5)):
                    L.lab[piece] = nb
                    notes.append(f"body piece of {sz} px enclosed by {L.names[nb]} ({share:.0%} of its rim): joined it")
                continue
            if small and sz < big:
                nb, share = neighbour_label(L.lab, piece, exclude=(0, k))
                if nb is not None:
                    L.lab[piece] = nb
                    notes.append(f"speck of {L.names[k]} ({sz} px) joined {L.names[nb]}")
    return notes


# ------------------------------------------------------------------ boundary maps
class Maps:
    """Boundary-strength structure tensors at full resolution, per source ('L18', 'L23', 'img'): (Jxx, Jxy, Jyy). Strength
    across a direction n = sqrt(n' J n); total strength = sqrt(Jxx + Jyy). F18 = L18 tokens (PCA) on the 8-px grid."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.J: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.F18: Optional[np.ndarray] = None                   # (k, gh, gw)
        self.grid = (8.0, 7.5)                                  # step, pixel index of cell 0's centre

    @classmethod
    def from_boundary(cls, b: Dict[str, np.ndarray], normal: Optional[Dict[str, Tuple[float, float]]] = None) -> "Maps":
        """Synthetic maps (tests): b[source] = strength; normal[source] = unit direction of the gradient, else isotropic."""
        mp = cls(next(iter(b.values())).shape)
        for k, v in b.items():
            v = np.asarray(v, np.float32)
            if normal and k in normal:
                nx, ny = normal[k]
                mp.J[k] = (v ** 2 * nx * nx, v ** 2 * nx * ny, v ** 2 * ny * ny)
            else:
                mp.J[k] = (v ** 2 / 2, np.zeros_like(v), v ** 2 / 2)
        return mp

    def has(self, key) -> bool:
        return key in self.J

    def mag(self, key) -> np.ndarray:
        if key not in self.J:
            return np.zeros(self.shape, np.float32)
        a, _, c = self.J[key]
        return np.sqrt(np.maximum(a + c, 0))

    def across(self, key, nx, ny) -> np.ndarray:
        """Strength of the gradient along direction (nx, ny) (scalars or per-pixel arrays)."""
        if key not in self.J:
            return np.zeros(self.shape, np.float32)
        a, b, c = self.J[key]
        return np.sqrt(np.maximum(nx * nx * a + 2 * nx * ny * b + ny * ny * c, 0))

    def combo_mag(self, layer: Optional[str], w_img: float) -> np.ndarray:
        out = self.mag(layer) if layer else np.zeros(self.shape, np.float32)
        return out + w_img * self.mag("img") if w_img else out

    def combo_across(self, layer: Optional[str], w_img: float, nx, ny) -> np.ndarray:
        out = self.across(layer, nx, ny) if layer else np.zeros(self.shape, np.float32)
        return out + w_img * self.across("img", nx, ny) if w_img else out


def normalise_tensor(J, fg: np.ndarray):
    a, b, c = J
    s = np.sqrt(np.maximum(a + c, 0))
    v = s[fg] if fg is not None and fg.any() else s.ravel()
    p = float(np.percentile(v, 95)) if len(v) else 1.0
    p = p if p > 1e-9 else 1.0
    return a / p ** 2, b / p ** 2, c / p ** 2


def image_tensor(rgb: np.ndarray, fg: np.ndarray, sigma: float = 1.0, rho: float = 1.5):
    """Colour structure tensor of the photograph (Gaussian derivatives, sigma px; smoothed by rho px), normalised."""
    x = rgb.astype(np.float32) / 255.0
    a = np.zeros(x.shape[:2], np.float32); b = np.zeros_like(a); c = np.zeros_like(a)
    for ch in range(x.shape[2]):
        gx = ndimage.gaussian_filter(x[..., ch], sigma, order=(0, 1))
        gy = ndimage.gaussian_filter(x[..., ch], sigma, order=(1, 0))
        a += gx * gx; b += gx * gy; c += gy * gy
    J = tuple(ndimage.gaussian_filter(v, rho) for v in (a, b, c))
    return normalise_tensor(J, fg)


def upsample_grid(g: np.ndarray, shape, step: float, origin: float, order: int = 1) -> np.ndarray:
    """Grid values (gh, gw) at pixel centres origin + step * i -> full resolution (bilinear, edges held)."""
    H, W = shape
    yy = (np.arange(H, dtype=np.float32) - origin) / step
    xx = (np.arange(W, dtype=np.float32) - origin) / step
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return ndimage.map_coordinates(np.asarray(g, np.float32), [Y, X], order=order, mode="nearest")


def maps_from_npz(npz, shape, fg: np.ndarray, rgb: Optional[np.ndarray], opts) -> Maps:
    mp = Maps(shape)
    step, origin = float(npz["step"]), float(npz["origin"])
    mp.grid = (step, origin)
    for key in ("L18", "L23"):
        for pre, suf in (("J", ""), ("Q", "line")):           # edge tensor 'L18', line tensor 'L18line'
            if f"{pre}{key[1:]}" in npz:
                J = tuple(upsample_grid(np.asarray(npz[f"{pre}{key[1:]}"][i], np.float32), shape, step, origin)
                          for i in range(3))
                if pre == "Q":                                # line tensor: off-diagonal may exceed the PSD bound
                    a, b, c = J
                    b = np.clip(b, -np.sqrt(np.maximum(a * c, 0)), np.sqrt(np.maximum(a * c, 0)))
                    J = (a, b, c)
                mp.J[key + suf] = normalise_tensor(J, fg)
    if "F18" in npz:
        mp.F18 = np.asarray(npz["F18"], np.float32)
    if rgb is not None:
        mp.J["img"] = image_tensor(rgb, fg)
    return mp


# ------------------------------------------------------------------ DINOv3 token maps (the only torch part)
_NET = {}


def compute_token_maps(rgb: np.ndarray, fg: np.ndarray, opts) -> Dict[str, np.ndarray]:
    """L18 and L23 high-pass structure tensors and L18 PCA features on the 8-px grid (four half-patch-shifted passes, tiled
    native resolution; block L = hidden_states[L + 1]). -> dict for np.savez."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    dev = "cuda" if torch.cuda.is_available() and opts.device != "cpu" else "cpu"
    if opts.model not in _NET:
        _NET[opts.model] = AutoModel.from_pretrained(opts.model).to(dev).eval()
    net = _NET[opts.model]
    nreg = getattr(net.config, "num_register_tokens", 4); D = net.config.hidden_size
    layers = (18, 23)
    TILE, OV = opts.tile, 128
    STEP = TILE - 2 * OV

    def one(t):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            x = torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1)))[None].to(dev)
            hs = net(pixel_values=x, output_hidden_states=True).hidden_states
            return {L: hs[L + 1][0, 1 + nreg:, :].float().reshape(t.shape[0] // 16, t.shape[1] // 16, D) for L in layers}

    def grid(x):
        H, W = x.shape[:2]; gh, gw = H // 16, W // 16
        if H <= TILE and W <= TILE:
            return one(x)
        out = {L: torch.zeros((gh, gw, D), device=dev) for L in layers}
        for y0 in range(0, H, STEP):
            for x0 in range(0, W, STEP):
                ya, xa = max(0, y0 - OV), max(0, x0 - OV)
                t = x[ya:min(H, y0 + STEP + OV), xa:min(W, x0 + STEP + OV)]
                t = t[:t.shape[0] // 16 * 16, :t.shape[1] // 16 * 16]
                T = one(t); ky, kx = (y0 - ya) // 16, (x0 - xa) // 16
                ny, nx = min(STEP, H - y0) // 16, min(STEP, W - x0) // 16
                for L in layers:
                    out[L][y0 // 16:y0 // 16 + ny, x0 // 16:x0 // 16 + nx] = T[L][ky:ky + ny, kx:kx + nx]
                del T
        return out

    def highpass(G, w):                                       # token minus the mean of its fg neighbours (7 x 7 patches)
        Gw = (G * w[..., None]).permute(2, 0, 1)[None]
        num = F.avg_pool2d(Gw, 7, 1, 3, count_include_pad=False)[0].permute(1, 2, 0)
        den = F.avg_pool2d(w[None, None], 7, 1, 3, count_include_pad=False)[0, 0]
        return G - num / den.clamp_min(1e-3)[..., None]

    sc = float(opts.token_scale)
    H0, W0 = rgb.shape[:2]
    if sc != 1.0:                                             # e.g. 2 = 8 native px per patch (finer lines)
        Wn, Hn = int(round(W0 * sc)), int(round(H0 * sc))
        rgb = np.asarray(Image.fromarray(rgb).resize((Wn, Hn), Image.BICUBIC))
        fg = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).resize((Wn, Hn), Image.NEAREST)) > 127
    S, st = 2, 8
    H, W = rgb.shape[:2]
    fh, fw = H // st, W // st
    xn = (rgb.astype(np.float32) / 255 - MEAN) / STD
    HP = {L: torch.zeros((fh, fw, D), device=dev) for L in layers}
    RAW18 = torch.zeros((fh, fw, D), device=dev)
    WG = np.zeros((fh, fw), np.float32)
    filled = np.zeros((fh, fw), bool)
    gen = torch.Generator(device=dev).manual_seed(0)
    for sy in range(S):
        for sx in range(S):
            J_ = xn[sy * st:, sx * st:]
            J_ = J_[:J_.shape[0] // 16 * 16, :J_.shape[1] // 16 * 16]
            G = grid(J_)
            gh, gw = G[layers[0]].shape[:2]
            fgc = fg[sy * st:sy * st + gh * 16, sx * st:sx * st + gw * 16].astype(np.float32)
            wc = fgc.reshape(gh, 16, gw, 16).mean((1, 3))
            wt = torch.from_numpy((wc >= 0.5).astype(np.float32)).to(dev)
            ys, xs = slice(sy, sy + S * gh, S), slice(sx, sx + S * gw, S)
            ny, nx = len(range(fh)[ys]), len(range(fw)[xs])
            for L in layers:
                HP[L][ys, xs] = highpass(G[L], wt)[:ny, :nx]
            RAW18[ys, xs] = G[18][:ny, :nx]
            WG[ys, xs] = wc[:ny, :nx]; filled[ys, xs] = True
            del G
    # native pixel index of grid cell 0's centre and the grid step in native px
    out = {"step": np.float32(st / sc), "origin": np.float32(st / sc - 0.5),
           "fgw": WG.astype(np.float16), "shape": np.array([H0, W0], np.int32), "token_scale": np.float32(sc)}
    ok = torch.from_numpy((WG >= opts.token_fg) & filled).to(dev)
    sel = torch.from_numpy(WG >= 0.5).to(dev)

    def pca(X, k, whiten):
        """PCA of the foreground tokens (k components), optionally each scaled to unit variance on the foreground."""
        V_ = X[sel]
        mu = V_.mean(0)
        idx = torch.randperm(len(V_), device=dev, generator=gen)[:20000]
        _, _, Vt = torch.linalg.svd(V_[idx] - mu, full_matrices=False)
        Z = (X - mu) @ Vt[:min(k, Vt.shape[0])].T
        if whiten:
            Z = Z / (Z[sel].std(0) + 1e-6)
        return Z

    def sh_(X, dy, dx):                                      # X shifted: value at (i, j) = X[i + dy, j + dx] (zero outside)
        Y = torch.zeros_like(X)
        ys_ = slice(max(0, -dy), X.shape[0] - max(0, dy)); yd = slice(max(0, dy), X.shape[0] - max(0, -dy))
        xs_ = slice(max(0, -dx), X.shape[1] - max(0, dx)); xd = slice(max(0, dx), X.shape[1] - max(0, -dx))
        Y[ys_, xs_] = X[yd, xd]
        return Y
    okf = ok.float()
    for L in layers:
        if int(sel.sum()) < 20:
            break
        Z = pca(HP[L], opts.hp_pca, True)
        # EDGE tensor: first differences (tokens 16 px apart: the neighbouring cells of the other shifted pass)
        gx = (sh_(Z, 0, 1) - sh_(Z, 0, -1)) / 2 * (sh_(okf, 0, 1) * sh_(okf, 0, -1))[..., None]
        gy = (sh_(Z, 1, 0) - sh_(Z, -1, 0)) / 2 * (sh_(okf, 1, 0) * sh_(okf, -1, 0))[..., None]
        out[f"J{L}"] = torch.stack([(gx * gx).sum(-1), (gx * gy).sum(-1), (gy * gy).sum(-1)]).cpu().numpy().astype(np.float32)
        del gx, gy
        # LINE tensor: second differences within one shifted pass (cells 2 apart on the interleaved grid): a token on a
        # thin line differs from both neighbours across it
        def l2(dy, dx):
            m = sh_(okf, dy, dx) * sh_(okf, -dy, -dx) * okf
            return (((sh_(Z, dy, dx) + sh_(Z, -dy, -dx) - 2 * Z) ** 2).sum(-1)) * m
        a, c = l2(0, 2), l2(2, 0)
        b = (l2(2, 2) - l2(2, -2)) / 2
        out[f"Q{L}"] = torch.stack([a, b, c]).cpu().numpy().astype(np.float32)
        del Z
    Fp = pca(RAW18, opts.pca, False) if int(sel.sum()) >= 20 else None
    if Fp is not None:
        out["F18"] = Fp.permute(2, 0, 1).cpu().numpy().astype(np.float16)
    del HP, RAW18
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ watershed
def flood(markers: np.ndarray, cost: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """Marker-controlled watershed (priority flood, 4-connected) inside `domain`: unlabelled domain pixels (marker 0) take
    the label of the marker that reaches them first in order of cost (ties: first come, first served)."""
    H, W = markers.shape
    lab = np.pad(markers.astype(np.int32), 1)
    dom = np.pad(domain, 1)
    cst = np.pad(cost.astype(np.float64), 1)
    Wp = W + 2
    flat_l, flat_d, flat_c = lab.ravel(), dom.ravel(), cst.ravel()
    heap, cnt = [], 0
    inq = np.zeros(flat_l.size, bool)
    seeds = np.flatnonzero((flat_l > 0) & flat_d)
    for off in (1, -1, Wp, -Wp):                            # unlabelled domain pixels next to a marker
        nb = seeds + off
        ok = flat_d[nb] & (flat_l[nb] == 0) & ~inq[nb]
        for p, q in zip(nb[ok], seeds[ok]):
            if not inq[p]:
                inq[p] = True
                heapq.heappush(heap, (flat_c[p], cnt, int(p), int(flat_l[q]))); cnt += 1
    while heap:
        _, _, p, k = heapq.heappop(heap)
        if flat_l[p]:
            continue
        flat_l[p] = k
        for off in (1, -1, Wp, -Wp):
            q = p + off
            if flat_d[q] and not flat_l[q] and not inq[q]:
                inq[q] = True
                heapq.heappush(heap, (flat_c[q], cnt, q, k)); cnt += 1
    return lab[1:-1, 1:-1]


def part_boundary(lab: np.ndarray) -> np.ndarray:
    """Pixels with a 4-neighbour of another non-zero label."""
    b = np.zeros(lab.shape, bool)
    for a, c in ((lab[1:], lab[:-1]), (lab[:, 1:], lab[:, :-1])):
        d = (a != c) & (a > 0) & (c > 0)
        if a.shape[0] != lab.shape[0]:
            b[1:] |= d; b[:-1] |= d
        else:
            b[:, 1:] |= d; b[:, :-1] |= d
    return b


def sharpen(lab: np.ndarray, cost: np.ndarray, domain: np.ndarray, band: float,
            keep: Optional[np.ndarray] = None) -> np.ndarray:
    """Move the boundaries between labelled parts onto the strongest ridge of `cost` within `band` px of where they are.
    Each piece keeps a core (outside the band, or its medial pixels when it is thinner than the band). `keep`: labels
    not to touch (their pixels stay as they are)."""
    if band <= 0:
        return lab.copy()
    bd = part_boundary(lab)
    unk = (edt(~bd) <= band) & domain & (lab > 0)
    if keep is not None:
        unk &= ~np.isin(lab, keep)
    markers = np.where(unk, 0, lab).astype(np.int32)
    # pieces that lost every pixel to the band keep their medial pixels
    for k in np.unique(lab[unk]):
        if k == 0:
            continue
        cl, n = label8(lab == k)
        for i, sl in enumerate(ndimage.find_objects(cl), 1):
            piece = cl[sl] == i
            if (markers[sl][piece] == k).any():
                continue
            d = edt(np.pad(piece, 1))[1:-1, 1:-1]
            core = piece & (d >= 0.7 * d.max())
            markers[sl][core] = k
    markers[~domain] = 0
    out = flood(markers, cost, domain & (lab > 0))
    out[~(domain & (lab > 0))] = lab[~(domain & (lab > 0))]
    return out.astype(lab.dtype)


# ------------------------------------------------------------------ body frame (serial homology's, imported)
class Body:
    """AP coordinate s (positive = anterior) and lateral coordinate t (positive = image-left with the head up, from the
    local midline) for any pixel; from descriptron_serial_homology_v1.body_frame."""

    def __init__(self, fr):
        self.fr = fr
        if fr.mode == "edges" and len(fr.mid_s):
            g = np.linspace(fr.mid_s.min(), fr.mid_s.max(), 200)
            off = []
            for x in g:
                near = np.abs(fr.mid_s - x) <= fr.window
                off.append(float(np.median(fr.mid_t[near])) if near.any() else
                           float(fr.mid_t[np.argmin(np.abs(fr.mid_s - x))]))
            self._g, self._off = g, np.array(off)
        else:
            self._g, self._off = np.array([0.0, 1.0]), np.zeros(2)

    def mid(self, s):
        return np.interp(s, self._g, self._off)

    def st(self, xs, ys):
        P = np.stack([np.asarray(xs, float), np.asarray(ys, float)], -1) - self.fr.c
        s = P @ self.fr.a
        return s, P @ self.fr.L - self.mid(s)

    def xy(self, s, t):
        o = self.mid(s); c, a, L = self.fr.c, self.fr.a, self.fr.L
        return c[0] + s * a[0] + (t + o) * L[0], c[1] + s * a[1] + (t + o) * L[1]

    def st_mask(self, m: np.ndarray):
        ys, xs = np.nonzero(m)
        s, t = self.st(xs, ys)
        return ys, xs, s, t


def frame_opts(opts):
    return argparse.Namespace(work_size=opts.work_size, open_frac=0.3, min_elongation=1.25, anterior=opts.anterior,
                              midline_window=0.10, midline="edges", _cat=lambda a: a.get("_name", ""))


def body_of(L: Labels, opts) -> Tuple[Optional[Body], List[str]]:
    H, W = L.lab.shape
    app = np.zeros(L.lab.shape, bool)
    for r in APPENDAGES:
        app |= L.mask(r)
    trunk = (L.lab > 0) & ~app
    head = L.mask("head") | L.mask("eye")
    body_ann = {"segmentation": sh.mask_rle(trunk), "_name": "trunk (foreground minus appendages)"}
    heads = [{"segmentation": sh.mask_rle(head), "_name": "head"}] if head.any() else []
    fr, warn = sh.body_frame(W, H, [body_ann], heads, [], [], frame_opts(opts))
    return (Body(fr) if fr is not None else None), warn


def side_name(t: float, view: Optional[str]) -> Optional[str]:
    """Anatomical side of a lateral coordinate (t > 0 = image-left with the head up): dorsal = same, ventral = mirrored."""
    if view not in ("dorsal", "ventral") or t == 0:
        return None
    return "left" if (t > 0) == (view == "dorsal") else "right"


# ------------------------------------------------------------------ straightened strips and curved cuts
class Strip:
    """A part straightened along the body axis: rows = s from anterior to posterior (1 px), columns = t (1 px)."""

    def __init__(self, body: Body, m: np.ndarray, pad: int = 3):
        ys, xs, s, t = body.st_mask(m)
        self.body = body
        self.S = np.arange(np.ceil(s.max()) + pad, np.floor(s.min()) - pad - 1, -1.0)
        self.T = np.arange(np.floor(t.min()) - pad, np.ceil(t.max()) + pad + 1, 1.0)
        SS, TT = np.meshgrid(self.S, self.T, indexing="ij")
        self.X, self.Y = body.xy(SS, TT)
        self.mask = self.sample(m, order=0) > 0.5
        self.t0 = int(np.argmin(np.abs(self.T)))             # column of the midline

    def sample(self, a: np.ndarray, order: int = 1) -> np.ndarray:
        return ndimage.map_coordinates(np.asarray(a, np.float32), [self.Y, self.X], order=order, mode="constant", cval=0)

    def row_of(self, s):
        return (self.S[0] - np.asarray(s, float))

    def s_of(self, row):
        return self.S[0] - np.asarray(row, float)


def viterbi(E: np.ndarray, lo: np.ndarray, hi: np.ndarray, lam: float = 0.1, max_step: int = 1,
            prior: Optional[np.ndarray] = None, mu: float = 0.0) -> np.ndarray:
    """Best path through E (steps x states): one state per step within [lo, hi], moving at most max_step states per
    step; maximises sum E - lam x |move| - mu x |state - prior|."""
    n, m = E.shape
    NEG = -1e18
    ar = np.arange(m)
    back = np.zeros((n, m), np.int32)

    def allowed(i):
        a = np.full(m, NEG)
        ok = (ar >= lo[i]) & (ar <= hi[i])
        a[ok] = E[i, ok] - (mu * np.abs(ar[ok] - prior[i]) if prior is not None else 0.0)
        return a
    score = allowed(0)
    for i in range(1, n):
        best, arg = score.copy(), ar.copy()
        for d in range(1, max_step + 1):
            for sgn in (1, -1):
                c = np.full(m, NEG)
                if sgn > 0:
                    c[d:] = score[:-d] - lam * d; src = ar - d
                else:
                    c[:-d] = score[d:] - lam * d; src = ar + d
                better = c > best
                best[better] = c[better]; arg[better] = src[better]
        a = allowed(i)
        score = np.where(a > NEG / 2, best + a, NEG)
        back[i] = arg
    path = np.zeros(n, np.int32)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


# ------------------------------------------------------------------ elytra
def line_key(maps: Maps, layer: str) -> str:
    """The LINE tensor of a layer ('L18' -> 'L18line') when the maps have it, else its edge tensor."""
    return layer + "line" if maps.has(layer + "line") else layer


def split_elytra(m: np.ndarray, maps: Maps, body: Body, view: Optional[str], opts) -> Tuple[List[dict], dict]:
    """elytra -> left/right elytron along the suture (ridge near the midline) or the midline. -> (parts, qc info)."""
    qc = {"suture": []}
    if view not in ("dorsal", "ventral"):
        return [dict(mask=m, name=opts.elytra_name or "elytra", method="not split", confidence="medium",
                     warnings=[f"view {view or 'unknown'}: elytra not split by side"])], qc
    cl, n = label8(m)
    sizes = np.bincount(cl.ravel())[1:] if n else np.array([])
    if n >= 2 and np.sort(sizes)[-2] >= 0.3 * sizes.max():      # two elytra spread apart: by their pieces
        out = []
        for i in np.argsort(-sizes)[:2] + 1:
            piece = cl == i
            ys, xs, s, t = body.st_mask(piece)
            side = side_name(float(np.median(t)), view)
            out.append(dict(mask=piece, name=sided(opts.side_format, side, "elytron"), method="separate pieces",
                            confidence="high", warnings=[], side=side))
        rest = m & ~(out[0]["mask"] | out[1]["mask"])
        if rest.any():
            out[0]["mask"] = out[0]["mask"] | (rest & dilate(out[0]["mask"], 4))
        return out, qc
    st = Strip(body, m)
    a = body.fr.a; Lv = body.fr.L
    E = st.sample(maps.combo_across(line_key(maps, opts.split_layer), opts.split_image_weight, Lv[0], Lv[1]))
    rows = np.flatnonzero(st.mask.any(1))
    widths = st.mask.sum(1)
    hw = float(np.median(widths[rows])) / 2
    win = max(2, int(round(opts.suture_window * hw)))
    lo = np.full(len(st.S), st.t0 - win); hi = np.full(len(st.S), st.t0 + win)
    Er = np.where(st.mask, E, 0.0)
    path = viterbi(Er[rows], lo[rows], hi[rows], lam=0.05 * float(np.median(Er[st.mask]) + 1e-6) * 4, max_step=1,
                   prior=np.full(len(rows), st.t0), mu=0.002)
    on = np.array([Er[r, c] for r, c in zip(rows, path)])
    band = Er[rows][:, max(0, st.t0 - win):st.t0 + win + 1]
    inside = st.mask[rows][:, max(0, st.t0 - win):st.t0 + win + 1]
    contrast = float(np.mean(on) / max(float(np.median(band[inside])) if inside.any() else 0.0, 1e-6)) if np.mean(on) > 0 else 0.0
    method, conf, warn = "L18 ridge (suture)" if opts.split_layer == "L18" else f"{opts.split_layer} ridge (suture)", \
        ("high" if contrast >= 1.6 else "medium"), []
    if contrast < opts.suture_contrast:
        path = np.full(len(rows), st.t0)
        method, conf = "rule fallback (midline)", "medium"
        warn.append(f"suture ridge weak (contrast {contrast:.2f} < --suture_contrast {opts.suture_contrast}): midline used")
    tcut = st.T[path]
    scut = st.s_of(rows)
    ys, xs, s, t = body.st_mask(m)
    tc = np.interp(-s, -scut, tcut)                             # np.interp needs increasing x: s decreasing with rows
    left_img = t > tc
    out = []
    for flag in (True, False):
        pm = np.zeros(m.shape, bool); pm[ys[left_img == flag], xs[left_img == flag]] = True
        side = side_name(1.0 if flag else -1.0, view)
        out.append(dict(mask=pm, name=sided(opts.side_format, side, "elytron"), method=method, confidence=conf,
                        warnings=list(warn), side=side, evidence=f"suture contrast {contrast:.2f}"))
    X, Y = body.xy(scut, tcut)
    qc["suture"] = list(zip(X.tolist(), Y.tolist()))
    return out, qc


# ------------------------------------------------------------------ abdomen
def abdomen_cut_paths(st: Strip, E: np.ndarray, rows: List[int], R: int, smooth: float = 1.0,
                      E_fine: Optional[np.ndarray] = None, refine: int = 0) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Trace each candidate line across the strip: for every column (t) a row within +-R of the candidate. With E_fine
    (the photograph's full-resolution strength) the path is then re-traced within +-refine px of itself for the final
    pixel placement. -> [(cols, rows of the path, mean strength along it inside the mask)]"""
    out = []
    cols_all = np.flatnonzero(st.mask.any(0))
    nR = st.mask.shape[0]
    for r0 in rows:
        a0, a1 = max(0, r0 - R), min(nR, r0 + R + 1)
        cols = np.array([c for c in cols_all if st.mask[a0:a1, c].any()])
        if len(cols) < 3:
            out.append((cols, np.full(len(cols), r0), 0.0)); continue
        sub = np.where(st.mask[a0:a1][:, cols], E[a0:a1][:, cols], 0.0).T     # (columns, rows)
        lo = np.zeros(len(cols), int); hi = np.full(len(cols), a1 - a0 - 1)
        scale = float(np.median(sub[sub > 0])) if (sub > 0).any() else 1.0
        p = viterbi(sub, lo, hi, lam=smooth * scale, max_step=1, prior=np.full(len(cols), r0 - a0), mu=0.03 * scale)
        prow = p + a0
        if E_fine is not None and refine > 0:                 # final pixel placement on the photograph
            offs = np.arange(-refine, refine + 1)
            rr = np.clip(prow[:, None] + offs[None], 0, nR - 1)
            sub2 = np.where(st.mask[rr, cols[:, None]], E_fine[rr, cols[:, None]], 0.0)
            sc2 = float(np.median(sub2[sub2 > 0])) if (sub2 > 0).any() else 1.0
            p2 = viterbi(sub2, np.zeros(len(cols), int), np.full(len(cols), 2 * refine), lam=smooth * sc2, max_step=1,
                         prior=np.full(len(cols), refine), mu=0.05 * sc2)
            prow = np.clip(prow + offs[p2], 0, nR - 1)
        inside = st.mask[prow, cols]
        val = float(np.mean(E[prow[inside], cols[inside]])) if inside.any() else 0.0
        out.append((cols, prow, val))
    return out


def width_runs(row: np.ndarray, min_len: int = 3) -> int:
    """Number of separate runs (lobes) in a strip row at least min_len px long (setae do not count), ignoring 1-px gaps."""
    if not row.any():
        return 0
    r = ndimage.binary_closing(row, np.ones(3, bool)) | row
    lab, n = ndimage.label(r)
    return int((np.bincount(lab)[1:] >= min_len).sum()) if n else 0


def choose_cuts(pos: np.ndarray, q: np.ndarray, start: float, end: float, nmax: int, opts) -> Tuple[List[int], float]:
    """Joint choice of cuts among candidates (positions between start and end, increasing): at most nmax pieces; score =
    sum (q - cut_threshold) - ratio_weight x sum log(len_j / len_j-1)^2 - missing_weight x (nmax - pieces); no piece
    shorter than min_segment_frac x (end - start) / nmax. -> (chosen candidate indices, score)."""
    m = len(pos)
    lmin = opts.min_segment_frac * (end - start) / max(nmax, 1)
    pts = [start] + list(pos) + [end]                         # 0 = start, 1..m = candidates, m+1 = end
    gain = [0.0] + [float(v) - opts.cut_threshold for v in q] + [0.0]
    NEG = -1e18

    def ratio(a, b):
        return opts.ratio_weight * math.log(max(a, 1e-6) / max(b, 1e-6)) ** 2
    # best[(j, i)][k]: pieces so far k (ending at point i, previous point j)
    best: Dict[Tuple[int, int], Dict[int, Tuple[float, Optional[Tuple[int, int]]]]] = {}
    for i in range(1, m + 2):
        if pts[i] - pts[0] >= lmin:
            best[(0, i)] = {1: (gain[i], None)}
    order = sorted(best.keys(), key=lambda x: x[1])
    keys_by_end = defaultdict(list)
    for key in best:
        keys_by_end[key[1]].append(key)
    for i in range(1, m + 1):                                  # extend states ending at candidate i
        for key in list(keys_by_end[i]):
            j = key[0]
            for k, (sc, _) in list(best[key].items()):
                if k >= nmax:
                    continue
                for l in range(i + 1, m + 2):
                    ln = pts[l] - pts[i]
                    if ln < lmin:
                        continue
                    v = sc + gain[l] - ratio(ln, pts[i] - pts[j])
                    nk = (i, l)
                    d = best.setdefault(nk, {})
                    if k + 1 not in d or v > d[k + 1][0]:
                        d[k + 1] = (v, key)
                        if nk not in keys_by_end[l]:
                            keys_by_end[l].append(nk)
    bestv, bestst = NEG, None
    for key, d in best.items():
        if key[1] != m + 1:
            continue
        for k, (sc, _) in d.items():
            v = sc - opts.missing_weight * max(0, nmax - k)
            if v > bestv:
                bestv, bestst = v, (key, k)
    if bestst is None:
        return [], NEG
    chosen = []
    key, k = bestst
    while key is not None:
        if 1 <= key[1] <= m:
            chosen.append(key[1] - 1)
        prev = best[key][k][1]
        key, k = prev, k - 1
    return sorted(chosen), bestv


def split_abdomen(m: np.ndarray, maps: Maps, body: Body, view: Optional[str], opts) -> Tuple[List[dict], dict, List[dict]]:
    """abdomen -> regular segments (+ apex) along curved intersegmental lines. -> (parts, qc, candidate rows)."""
    qc = {"accepted": [], "rejected": []}
    cand_rows: List[dict] = []
    fmt = (opts.segment_name_ventral if view == "ventral" and opts.segment_name_ventral else opts.segment_name)
    first, last = opts.first_segment, opts.last_regular_segment
    N = max(1, last - first + 1)
    a = body.fr.a
    st = Strip(body, m)
    lk = line_key(maps, opts.split_layer)
    E = st.sample(maps.combo_across(lk, opts.split_image_weight, a[0], a[1]))      # lines across the body axis
    E23 = st.sample(maps.across(line_key(maps, "L23"), a[0], a[1])) if maps.has("L23") else None
    M = st.mask
    # shape (widths, lobes, the tip) from the mask without its setae: opened by a small disc
    Ms = erode(dilate(erode(M, opts.setae_px), opts.setae_px), 0) & M if opts.setae_px > 0 else M
    Ms = Ms if Ms.sum() > 0.5 * M.sum() else M
    w = Ms.sum(1).astype(float)
    rows = np.flatnonzero(w >= 3)
    if len(rows) < 10:
        return [dict(mask=m, name=fmt.format(n=first), method="too small to split", confidence="low",
                     warnings=["abdomen too small to split"])], qc, cand_rows
    r_front, r_end = int(rows[0]), int(rows[-1]) + 1
    Labd = r_end - r_front
    Lseg = Labd / (N + (0.5 if opts.apex == "auto" else 0.0))
    # line strength per row: in the two lateral MARGIN bands (where the intersegmental nodes are; the middle of a
    # tergite carries basal lines, setal and colour bands) and across the middle 80%
    Pm, Pc, P23 = np.zeros(M.shape[0]), np.zeros(M.shape[0]), np.zeros(M.shape[0])
    tl = np.full(M.shape[0], -1); tr = np.full(M.shape[0], -1)
    for r in rows:
        c = np.flatnonzero(Ms[r]); tl[r], tr[r] = c[0], c[-1]
        band = max(2, int(round(opts.margin_band * (c[-1] - c[0] + 1))))
        Pm[r] = float(np.concatenate([E[r, c[0]:c[0] + band], E[r, c[-1] - band + 1:c[-1] + 1]]).mean())
        lo, hi = int(c[0] + 0.1 * (c[-1] - c[0])), int(c[-1] - 0.1 * (c[-1] - c[0]))
        seg = E[r, lo:hi + 1][M[r, lo:hi + 1]]
        Pc[r] = float(seg.mean()) if len(seg) else 0.0
        if E23 is not None:
            P23[r] = float(np.concatenate([E23[r, c[0]:c[0] + band], E23[r, c[-1] - band + 1:c[-1] + 1]]).mean())
    zm = robust_z(ndimage.gaussian_filter1d(Pm, 1.0), ndimage.gaussian_filter1d(Pm, 1.0)[rows])
    zc = robust_z(ndimage.gaussian_filter1d(Pc, 1.0), ndimage.gaussian_filter1d(Pc, 1.0)[rows])
    z23 = robust_z(ndimage.gaussian_filter1d(P23, 1.0), ndimage.gaussian_filter1d(P23, 1.0)[rows])
    # outline notch (the paratergites' posterior corners stand out at each intersegmental node)
    ws = ndimage.gaussian_filter1d(w, 1.0)
    wmed = float(np.median(w[rows]))
    h = max(3, int(round(0.3 * Lseg)))
    notch = np.zeros(M.shape[0])
    for r in rows:
        left = ws[max(r_front, r - h):r]; right = ws[r + 1:min(r_end, r + h + 1)]
        if len(left) and len(right):
            notch[r] = max(0.0, (min(left.max(), right.max()) - ws[r]) / max(wmed, 1))
    prof = zm + opts.mid_weight * zc + opts.l23_margin_weight * z23 + opts.notch_weight * notch / max(opts.notch_scale, 1e-6)
    sep = max(2, int(round(opts.min_segment_frac * Lseg * 0.5)))
    edge_gap = max(2, int(round(0.2 * Lseg)))
    peaks = [r for r in range(r_front + edge_gap, r_end - max(2, int(0.08 * Lseg)))
             if prof[r] == prof[max(0, r - sep):r + sep + 1].max()]
    R = max(2, int(round(opts.cut_bend * Lseg)))
    E_img = st.sample(maps.across("img", a[0], a[1])) if (maps.has("img") and opts.refine_px > 0) else None
    paths = abdomen_cut_paths(st, E, peaks, R, opts.cut_smooth, E_img, int(opts.refine_px))
    cands = []
    for (cols, prow, val), r0 in zip(paths, peaks):
        k0, k1 = max(r_front, r0 - 2), min(r_end, r0 + 3)
        q = float(prof[k0:k1].max())
        cands.append(dict(row=float(np.median(prow)), r0=r0, cols=cols, prow=prow, line_z=float(zm[k0:k1].max()),
                          mid_z=float(zc[k0:k1].max()), notch=float(notch[k0:k1].max()),
                          margin23_z=float(z23[k0:k1].max()) if E23 is not None else float("nan"), score=q))
    pos = np.array([c["row"] for c in cands]); qv = np.array([c["score"] for c in cands])
    # options: no apex, or an apex behind one of the posterior candidates
    best = (None, None, -1e18)                               # (chosen, apex candidate index, score)
    ch, sc = choose_cuts(pos, qv, r_front, r_end, N, opts)
    if opts.apex == "none":
        best = (ch, None, sc)
    else:
        best = (ch, None, sc)
        for ai, c in enumerate(cands):
            ra = c["row"]
            if ra - r_front < 0.5 * Labd or r_end - ra < 3:
                continue
            ap_rows = np.arange(int(ra) + 1, r_end)
            reg_rows = rows[rows < ra]
            if len(ap_rows) < 3 or len(reg_rows) < 5:
                continue
            wref = float(np.median(w[reg_rows]))
            rho = float(w[ap_rows].mean()) / max(wref, 1)
            before = float(np.mean(w[max(r_front, int(ra) - 4):int(ra) - 1])) if int(ra) - 1 > r_front else wref
            after = float(np.mean(w[int(ra) + 2:int(ra) + 6])) if int(ra) + 6 <= r_end else float(w[ap_rows].mean())
            step = before / max(after, 1)
            lobes = float(np.mean([width_runs(Ms[r], max(3, int(0.06 * wref))) >= 2 for r in ap_rows]))
            prev = w[max(r_front, int(ra - Lseg)):int(ra)]
            rho_prev = float(w[ap_rows].mean()) / max(float(prev.mean()) if len(prev) else wref, 1)
            c["apex_rho"], c["apex_step"], c["apex_lobes"], c["apex_rho_prev"] = rho, step, lobes, rho_prev
            # narrower than the regular segments AND than the segment just before it (tergite VIII tapers too), with an
            # abrupt narrowing at its base, separate lobes, or a very narrow shape
            if not (rho < opts.apex_width_frac and rho_prev < opts.apex_prev_frac and
                    (step >= opts.apex_step or lobes >= opts.apex_lobes or rho < opts.apex_narrow)):
                continue
            if r_end - ra > 1.5 * Lseg or r_end - ra < opts.apex_min_frac * Lseg:
                continue
            sub = [i for i in range(len(cands)) if cands[i]["row"] < ra]
            ch2, sc2 = choose_cuts(pos[sub], qv[sub], r_front, ra, N, opts)
            v = sc2 + (c["score"] - opts.cut_threshold) + opts.apex_bonus
            if v > best[2]:
                best = ([sub[i] for i in ch2], ai, v)
    chosen, apex_i, _ = best
    chosen = list(chosen or [])
    # pieces: by the chosen curved cuts (and the apex cut)
    cut_idx = chosen + ([apex_i] if apex_i is not None else [])
    cut_idx = sorted(set(cut_idx), key=lambda i: cands[i]["row"])
    ys, xs, s, t = body.st_mask(m)
    prow_pix = st.row_of(s)                                   # row coordinate of each pixel
    piece = np.zeros(len(s), int)
    for i in cut_idx:
        c = cands[i]
        rc = np.interp(t, st.T[c["cols"]], c["prow"].astype(float))
        piece += (prow_pix > rc).astype(int)
    n_reg = len(chosen) + 1
    warn_all = []
    if n_reg < N:
        warn_all.append(f"found {n_reg} of {N} expected regular segments: numbered from the anterior, so the posterior "
                        f"ones may be numbered too low")
    parts = []
    apex_present = apex_i is not None
    for k in range(len(cut_idx) + 1):
        pm = np.zeros(m.shape, bool); sel = piece == k
        pm[ys[sel], xs[sel]] = True
        if not pm.any():
            continue
        is_apex = apex_present and k == len(cut_idx)
        bounds = ([cands[cut_idx[k - 1]]] if k > 0 else []) + ([cands[cut_idx[k]]] if k < len(cut_idx) else [])
        strong = all(b["score"] >= opts.cut_threshold + 1.0 for b in bounds)
        if is_apex:
            parts += apex_pieces(pm, cands[apex_i], cands, body, fmt, opts, warn_all)
            continue
        n = first + k
        conf = "high" if strong and not warn_all else ("low" if n_reg < N - 1 else "medium")
        ev = "; ".join(f"cut at s={st.s_of(b['row']):.0f}: margin line z {b['line_z']:.1f}, middle z {b['mid_z']:.1f}, "
                       f"notch {b['notch']:.2f}, L23 margin z {b['margin23_z']:.1f}" for b in bounds)
        parts.append(dict(mask=pm, name=fmt.format(n=n), method=f"{opts.split_layer} ridges (curved cuts)",
                          confidence=conf, warnings=list(warn_all), segment=n, evidence=ev))
    for p in parts:
        p["apex_present"] = "yes" if apex_present else "no"
    # candidate table and QC
    chosen_set = set(chosen)
    for i, c in enumerate(cands):
        X, Y = body.xy(st.s_of(c["prow"]), st.T[c["cols"]])
        line = list(zip(X.tolist(), Y.tolist()))
        acc = i in chosen_set or i == apex_i
        why = ("apex boundary" if i == apex_i else "accepted") if acc else (
            f"not chosen (score below --cut_threshold {opts.cut_threshold})" if c["score"] < opts.cut_threshold else
            "rejected by the joint choice (segment-length ratios / expected count): setal or colour band?")
        mid = len(line) // 2
        cand_rows.append(dict(x=round(line[mid][0], 1) if line else "", y=round(line[mid][1], 1) if line else "",
                              s=round(float(st.s_of(c["row"])), 1), margin_line_z=round(c["line_z"], 2),
                              middle_z=round(c["mid_z"], 2), notch=round(c["notch"], 3),
                              margin23_z=round(c["margin23_z"], 2), score=round(c["score"], 2), accepted=acc, reason=why,
                              apex_rho=round(c.get("apex_rho", float("nan")), 3), apex_step=round(c.get("apex_step", float("nan")), 3),
                              apex_lobes=round(c.get("apex_lobes", float("nan")), 3),
                              apex_rho_prev=round(c.get("apex_rho_prev", float("nan")), 3)))
        (qc["accepted"] if acc else qc["rejected"]).append(line)
    return parts, qc, cand_rows


def apex_pieces(pm: np.ndarray, cut: dict, cands: List[dict], body: Body, fmt: str, opts, warn_all) -> List[dict]:
    """The apex behind the last regular segment: one piece -> apex name (or two, across a clear transverse ridge, are
    numbered on); separate median pieces numbered on; paired lateral pieces and a weak apex boundary -> terminalia."""
    nxt = opts.last_regular_segment + 1
    apex_name = opts.apex_name or fmt.format(n=nxt)
    out = []
    weak = cut["line_z"] < opts.apex_min_line
    cl, n = label8(pm)
    sizes = np.bincount(cl.ravel())[1:] if n else np.array([0])
    big = [i + 1 for i in range(n) if sizes[i] >= 0.15 * sizes.max()]
    if weak:
        rest = pm.copy()
        return [dict(mask=rest, name=opts.terminalia_name, method="apex (shape), boundary weak", confidence="low",
                     warnings=warn_all + [f"apex boundary weak (line z {cut['line_z']:.1f}): named {opts.terminalia_name}"],
                     segment="apex", evidence=f"apex width ratio {cut.get('apex_rho', float('nan')):.2f}")]
    if len(big) <= 1:
        return [dict(mask=pm, name=apex_name, method="apex (shape + ridge)", confidence="medium", warnings=list(warn_all),
                     segment=nxt, evidence=f"apex width ratio {cut.get('apex_rho', float('nan')):.2f}, narrowing "
                                           f"{cut.get('apex_step', float('nan')):.2f}")]
    info = []
    hw = None
    for i in big:
        q = cl == i
        ys, xs, s, t = body.st_mask(q)
        info.append((i, float(np.median(s)), float(np.median(t)), q))
    allys, allxs, alls, allt = body.st_mask(pm)
    hw = max(1.0, (allt.max() - allt.min()) / 2)
    median = sorted([x for x in info if abs(x[2]) < 0.25 * hw], key=lambda x: -x[1])
    lateral = [x for x in info if abs(x[2]) >= 0.25 * hw]
    for j, (i, s_, t_, q) in enumerate(median):
        out.append(dict(mask=q, name=apex_name if j == 0 else fmt.format(n=nxt + j), method="apex piece (median)",
                        confidence="medium", warnings=list(warn_all), segment=nxt + j))
    for i, s_, t_, q in lateral:
        out.append(dict(mask=q, name=opts.terminalia_name, method="apex piece (paired lateral)", confidence="low",
                        warnings=warn_all + ["paired lateral apex piece (stylus or gonocoxite?): named "
                                             f"{opts.terminalia_name}"], segment="apex"))
    if not median:
        for o in out:
            o["warnings"].append("no median apex piece")
    small = pm & ~np.isin(cl, big)
    if small.any() and out:
        out[0]["mask"] = out[0]["mask"] | small
    return out


# ------------------------------------------------------------------ chains (legs, antennae): geodesic unrolling
def geodesic(mask: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Geodesic distance (8-connected, diagonal sqrt 2) inside mask from the src pixels; inf outside / unreachable."""
    ys, xs = np.nonzero(mask)
    n = len(ys)
    out = np.full(mask.shape, np.inf)
    if not n:
        return out
    idx = -np.ones(mask.shape, np.int64); idx[ys, xs] = np.arange(n)
    r_, c_, w_ = [], [], []
    H, W = mask.shape
    for dy, dx, wt in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        ok[ok] = mask[y2[ok], x2[ok]]
        r_.append(idx[ys[ok], xs[ok]]); c_.append(idx[y2[ok], x2[ok]]); w_.append(np.full(int(ok.sum()), wt))
    G = csr_matrix((np.concatenate(w_), (np.concatenate(r_), np.concatenate(c_))), shape=(n, n))
    s = idx[src & mask]
    s = s[s >= 0]
    if not len(s):
        return out
    d = dijkstra(G, directed=False, indices=s, min_only=True)
    out[ys, xs] = d
    return out


class Chain:
    """A thin part unrolled by geodesic distance from its base: per 1-px bin the width, centre and direction."""

    def __init__(self, m: np.ndarray, base: np.ndarray):
        self.m = m
        self.g = geodesic(m, base)
        ok = np.isfinite(self.g) & m
        self.ys, self.xs = np.nonzero(ok)
        self.d = self.g[self.ys, self.xs]
        self.length = float(self.d.max()) if len(self.d) else 0.0
        nb = int(self.length) + 1
        self.bin = np.minimum(self.d.astype(int), nb - 1)
        cnt = np.bincount(self.bin, minlength=nb).astype(float)
        self.width = ndimage.gaussian_filter1d(cnt, 1.5)
        cx = np.bincount(self.bin, self.xs, nb) / np.maximum(cnt, 1)
        cy = np.bincount(self.bin, self.ys, nb) / np.maximum(cnt, 1)
        good = cnt > 0
        k = np.arange(nb)
        cx = np.interp(k, k[good], cx[good]); cy = np.interp(k, k[good], cy[good])
        sm = max(1.5, 0.03 * nb)
        self.cx, self.cy = ndimage.gaussian_filter1d(cx, sm), ndimage.gaussian_filter1d(cy, sm)
        ux, uy = np.gradient(self.cx), np.gradient(self.cy)
        nrm = np.hypot(ux, uy) + 1e-9
        self.ux, self.uy = ux / nrm, uy / nrm
        # pieces per bin (a leg that branches = two parts touching)
        self.branched = False
        if nb > 10:
            for b in range(int(0.2 * nb), nb, max(1, nb // 20)):
                sel = self.bin == b
                if sel.sum() > 3:
                    pts = np.zeros(m.shape, bool); pts[self.ys[sel], self.xs[sel]] = True
                    if label8(dilate(pts, 1.5))[1] >= 2:
                        self.branched = True
                        break

    def along(self, maps: Maps, layer, w_img) -> np.ndarray:
        """Mean strength along the chain's own direction, per bin."""
        nb = len(self.width)
        ux, uy = self.ux[self.bin], self.uy[self.bin]
        v = np.zeros(len(self.bin))
        if layer and maps.has(layer):
            a, b, c = (x[self.ys, self.xs] for x in maps.J[layer])
            v += np.sqrt(np.maximum(ux * ux * a + 2 * ux * uy * b + uy * uy * c, 0))
        if w_img and maps.has("img"):
            a, b, c = (x[self.ys, self.xs] for x in maps.J["img"])
            v += w_img * np.sqrt(np.maximum(ux * ux * a + 2 * ux * uy * b + uy * uy * c, 0))
        return np.bincount(self.bin, v, nb) / np.maximum(np.bincount(self.bin, minlength=nb), 1)

    def bend(self, h: int) -> np.ndarray:
        nb = len(self.width); out = np.zeros(nb)
        for b in range(nb):
            i0, i1 = max(0, b - h), min(nb - 1, b + h)
            c = self.ux[i0] * self.ux[i1] + self.uy[i0] * self.uy[i1]
            out[b] = math.acos(float(np.clip(c, -1, 1)))
        return out


def base_pixels(m: np.ndarray, target: np.ndarray, tol: float = 2.0, core: bool = True) -> np.ndarray:
    """The base of an appendage piece: the pixel of m nearest the target's CORE (target pixels at least half its largest
    inscribed radius deep; core=False: nearest the target itself) and the pixels of m within tol px of it. A leg lying
    along the body touches its outline over a long stretch; its base is where it goes under the body."""
    if not target.any() or not m.any():
        return m & (edt(m) >= edt(m).max()) if m.any() else m
    if core:
        dt = edt(target)
        target = dt >= 0.5 * dt.max()
    d = edt(~target)
    dm = np.where(m, d, np.inf)
    y, x = np.unravel_index(np.argmin(dm), dm.shape)
    disc = np.zeros(m.shape, bool); disc[y, x] = True
    return m & dilate(disc, tol)


def parse_leg_parts(spec: str) -> List[Tuple[str, float]]:
    out = []
    for item in spec.split(","):
        if ":" in item:
            k, v = item.split(":")
            out.append((k.strip(), float(v)))
        elif item.strip():
            out.append((item.strip(), 1.0))
    tot = sum(v for _, v in out) or 1.0
    return [(k, v / tot) for k, v in out]


def split_leg(m: np.ndarray, trunk: np.ndarray, maps: Maps, view: Optional[str], opts) -> Tuple[List[dict], dict]:
    """One leg piece -> generic parts proximal to distal at its joints. -> (parts with generic names, qc)."""
    spec = opts.leg_parts_ventral if view == "ventral" else opts.leg_parts_dorsal
    parts = parse_leg_parts(spec)
    ch = Chain(m, base_pixels(m, trunk))
    qc = {"cuts": []}
    warn = []
    if ch.branched:
        warn.append("leg piece branches (two legs, or a leg and an antenna, touching?): split as one leg")
    D = ch.length
    K = len(parts)
    if D < 8 or K < 2:
        return [dict(mask=m, name="leg", generic="leg", method="too short to split", confidence="low",
                     warnings=warn + ["leg piece too short to split"])], qc
    E = ch.along(maps, line_key(maps, opts.split_layer) if opts.leg_map == "line" else opts.split_layer,
                 opts.split_image_weight)
    nb = len(E)
    zE = robust_z(E, E)
    h1 = max(2, int(0.1 * D))
    notch = np.zeros(nb)
    wmed = float(np.median(ch.width[ch.width > 0])) if (ch.width > 0).any() else 1.0
    for b in range(nb):
        prev = ch.width[max(0, b - h1):b]
        notch[b] = max(0.0, (prev.max() - ch.width[b]) / max(wmed, 1)) if len(prev) else 0.0
    bend = ch.bend(max(2, int(0.08 * D)))
    J = np.clip(zE, -3, 4) + opts.leg_width_weight * notch + opts.leg_bend_weight * bend / (math.pi / 4)
    Js = ndimage.gaussian_filter1d(J, 1.0)
    sep = max(2, int(0.05 * D))
    lo_b, hi_b = int(0.06 * D), int(0.96 * D)
    cand = [b for b in range(lo_b, hi_b) if Js[b] == Js[max(0, b - sep):b + sep + 1].max()]
    cum = np.cumsum([f for _, f in parts])[:-1] * D
    fall = [float(c) for c in cum]
    pos = [float(b) for b in cand] + fall
    sc = [float(Js[b]) for b in cand] + [opts.fallback_score] * len(fall)
    kind = ["ridge"] * len(cand) + ["fallback"] * len(fall)
    order = np.argsort(pos)
    pos = [pos[i] for i in order]; sc = [sc[i] for i in order]; kind = [kind[i] for i in order]
    # DP: choose K-1 increasing cuts; score = sum J - leg_ratio_weight * sum log(len_k / (f_k D))^2
    fr = [f for _, f in parts]
    NEG = -1e18
    n = len(pos)
    best = [[(NEG, -1)] * n for _ in range(K - 1)]
    mingap = max(2.0, 0.04 * D)

    def pen(ln, k):
        return opts.leg_ratio_weight * math.log(max(ln, 1e-3) / max(fr[k] * D, 1e-3)) ** 2
    for i in range(n):
        if pos[i] >= mingap:
            best[0][i] = (sc[i] - pen(pos[i], 0), -1)
    for k in range(1, K - 1):
        for i in range(n):
            for j in range(i):
                if best[k - 1][j][0] <= NEG / 2 or pos[i] - pos[j] < mingap:
                    continue
                v = best[k - 1][j][0] + sc[i] - pen(pos[i] - pos[j], k)
                if v > best[k][i][0]:
                    best[k][i] = (v, j)
    fin = [(best[K - 2][i][0] - pen(D - pos[i], K - 1), i) for i in range(n)
           if best[K - 2][i][0] > NEG / 2 and D - pos[i] >= mingap]
    if not fin:
        return [dict(mask=m, name="leg", generic="leg", method="no joints found", confidence="low",
                     warnings=warn + ["no joint layout fits"])], qc
    _, i = max(fin)
    sel = []
    for k in range(K - 2, -1, -1):
        sel.append(i); i = best[k][i][1]
    sel = sel[::-1]
    cuts = [pos[i] for i in sel]; kinds = [kind[i] for i in sel]
    lab = np.zeros(m.shape, np.int32)
    part_of = np.searchsorted(np.array(cuts), ch.d, side="right")
    lab[ch.ys, ch.xs] = part_of + 1
    lab[m & (lab == 0)] = 1                                   # unreachable specks: proximal part
    cost = maps.combo_mag(opts.split_layer, opts.image_weight)
    lab = sharpen(lab, cost, m, opts.leg_band_px)
    out = []
    for k, (pname, _) in enumerate(parts):
        pmk = lab == k + 1
        if not pmk.any():
            continue
        kk = [kinds[j] for j in (k - 1, k) if 0 <= j < len(kinds)]
        method = "L18 joint ridge" if all(x == "ridge" for x in kk) else (
            "rule fallback (expected length)" if all(x == "fallback" for x in kk) else "L18 joint ridge + rule fallback")
        w = list(warn)
        if "fallback" in kk:
            w.append("a joint placed at the expected length fraction (no ridge there)")
        out.append(dict(mask=pmk, generic=pname, name=pname, method=method,
                        confidence="medium" if "fallback" in kk or w else "high", warnings=w,
                        evidence=f"leg length {D:.0f} px; cuts at " + ", ".join(f"{c / D:.2f}" for c in cuts)))
    for c in cuts:
        b = int(min(len(ch.cx) - 1, round(c)))
        qc["cuts"].append((float(ch.cx[b]), float(ch.cy[b]), float(-ch.uy[b]), float(ch.ux[b]),
                           float(max(3.0, ch.width[b] / 2 + 2))))
    return out, qc


def name_legs(leg_parts: List[dict], L: Labels, im: dict, view: Optional[str], opts) -> None:
    """Serial homologue and side of every generic leg part, by descriptron_serial_homology_v1.run (imported)."""
    H, W = L.lab.shape
    cats = {"body": 1, "head": 2, "pronotum": 3, "elytra": 4, "abdomen": 5}
    gen = {}
    for p in leg_parts:
        if p.get("generic") and p["generic"] != "leg":
            gen.setdefault(p["generic"], len(cats) + len(gen) + 1)
    cats.update(gen)
    app = np.zeros(L.lab.shape, bool)
    for r in APPENDAGES:
        app |= L.mask(r)
    anns = []
    for nm, m in (("body", (L.lab > 0) & ~app), ("head", L.mask("head") | L.mask("eye")), ("pronotum", L.mask("pronotum")),
                  ("elytra", L.mask("elytra")), ("abdomen", L.mask("abdomen"))):
        if m.any():
            anns.append({"id": len(anns) + 1, "image_id": 1, "category_id": cats[nm], "segmentation": sh.mask_rle(m)})
    ids = {}
    for p in leg_parts:
        if p.get("generic") in gen:
            aid = 1000 + len(ids)
            ids[aid] = p
            anns.append({"id": aid, "image_id": 1, "category_id": cats[p["generic"]], "segmentation": sh.mask_rle(p["mask"])})
    if not ids:
        return
    coco = {"images": [{"id": 1, "file_name": im.get("file_name", ""), "width": W, "height": H}],
            "categories": [{"id": v, "name": k} for k, v in cats.items()], "annotations": anns}
    args = ["_", "--out", "_", "--name_format", opts.leg_name_format]
    if view:
        args += ["--view", view]
    if opts.anterior:
        args += ["--anterior", opts.anterior]
    shopts = sh.build_parser().parse_args(args)
    _, rows, _ = sh.run(coco, shopts)
    for r in rows:
        p = ids.get(r["annotation_id"])
        if p is None:
            continue
        if r["final_name"]:
            p["name"] = r["final_name"]
            p["side"] = r.get("side", "")
            if r.get("confidence") == "low" or (r.get("confidence") == "medium" and p["confidence"] == "high"):
                p["confidence"] = r["confidence"]
        else:
            p["name"] = p["generic"]
            p["confidence"] = "low"
            p["warnings"].append("serial homology: not named (" + "; ".join(x for x in (r.get("reason"), r.get("warnings")) if x) + ")")
        if r.get("warnings"):
            p["warnings"].append("serial homology: " + r["warnings"])


# ------------------------------------------------------------------ antennae, mandibles, mouthparts
def split_antenna(m: np.ndarray, L: Labels, maps: Maps, body: Body, view: Optional[str], opts) -> Tuple[List[dict], dict]:
    """Antenna pieces -> left/right antenna by the side of the base; a broad piece at the antennal base (anterior corner of
    the head) -> mandible."""
    head = L.mask("head") | L.mask("eye")
    hys, hxs, hs, ht = body.st_mask(head) if head.any() else (None, None, np.array([0.0]), np.array([0.0]))
    head_front = float(np.percentile(hs, 98)); head_len = float(np.ptp(hs)) or 1.0
    qc = {"lines": []}
    out = []
    cl, n = label8(m)
    if not n:
        return out, qc
    sizes = np.bincount(cl.ravel())[1:]
    cost = maps.combo_mag(opts.split_layer, opts.image_weight)
    lengths = {}
    chains = {}
    for i in range(1, n + 1):
        piece = cl == i
        if sizes[i - 1] < 10:
            continue
        base = base_pixels(piece, head) if head.any() else piece & (edt(piece) > 0)
        chains[i] = Chain(piece, base)
        lengths[i] = chains[i].length
    longest = max(lengths.values()) if lengths else 1.0
    mand_on = opts.mandible_from_antenna == "on" or (opts.mandible_from_antenna == "auto" and not L.idx("mandible"))
    # typical antenna radius from the long pieces (for short pieces that may be a mandible on their own)
    radii = []
    for i, chn in chains.items():
        if lengths[i] >= 0.4 * longest:
            d_ = edt(np.pad(cl == i, 1))[1:-1, 1:-1]
            v = d_[(cl == i) & (chn.g >= 0.3 * chn.length)]
            if len(v):
                radii.append(float(np.median(v)))
    ra_ref = float(np.median(radii)) if radii else 2.0
    for i, chn in chains.items():
        piece = cl == i
        warn = []
        if mand_on and lengths[i] < 0.4 * longest:            # a short piece on its own: a mandible if broad and anterior
            wide = float(edt(np.pad(piece, 1))[1:-1, 1:-1].max())
            pys, pxs, ps_, pt_ = body.st_mask(piece)
            if (wide >= opts.mandible_width * ra_ref and piece.sum() >= opts.mandible_min_area * np.pi * ra_ref ** 2
                    and float(np.median(ps_)) >= head_front - opts.mandible_zone * head_len):
                mys, mxs, ms, mt = body.st_mask(piece)            # sided by its bulk: the curved tip may cross the midline
                mside = side_name(float(np.median(mt)), view)
                out.append(dict(mask=piece, name=sided(opts.side_format, mside, "mandible") if mside else "mandible",
                                source="antenna", method="broad short piece of the coarse antenna class at the anterior "
                                "corner of the head", confidence="medium", warnings=["mandible separated from the coarse "
                                "antenna class"], side=mside, evidence=f"width x{wide / max(ra_ref, 1e-6):.1f} the antenna radius"))
                continue
        ys, xs, s, t = body.st_mask(base_pixels(piece, head) if head.any() else piece)
        tb = float(np.median(t))
        side = side_name(tb, view)
        if side is None:
            warn.append(f"view {view or 'unknown'} or base on the midline: antenna not sided")
        # the antenna proper: geodesic path from its tip back to the base, plus its own width
        tip = np.unravel_index(np.argmax(np.where(np.isfinite(chn.g), chn.g, -1)), chn.g.shape)
        path = np.zeros(piece.shape, bool)
        y, x = tip
        g = np.where(np.isfinite(chn.g), chn.g, np.inf)
        for _ in range(int(chn.length * 2) + 10):
            path[y, x] = True
            if g[y, x] <= 0.5:
                break
            y0, y1, x0, x1 = max(0, y - 1), min(piece.shape[0], y + 2), max(0, x - 1), min(piece.shape[1], x + 2)
            win = g[y0:y1, x0:x1]
            k = np.unravel_index(np.argmin(win), win.shape)
            ny, nx = y0 + k[0], x0 + k[1]
            if (ny, nx) == (y, x):
                break
            y, x = ny, nx
        dpiece = edt(np.pad(piece, 1))[1:-1, 1:-1]
        rads = dpiece[path & (chn.g >= 0.3 * chn.length)]
        ra = float(np.median(rads)) if len(rads) else 2.0
        zone = (edt(~path) <= max(2.5, 1.6 * ra)) & piece
        rest = piece & ~zone
        mand = None
        if rest.any() and lengths[i] >= 0.4 * longest and mand_on:
            rl, rn = label8(rest)
            rs = np.bincount(rl.ravel())[1:]
            j = int(np.argmax(rs)) + 1
            blob = rl == j
            bys, bxs, bs, bt = body.st_mask(blob)
            wide = float(edt(np.pad(blob | (zone & dilate(blob, 2)), 1))[1:-1, 1:-1].max())
            anterior = float(np.median(bs)) >= head_front - opts.mandible_zone * head_len
            if rs.max() >= max(30, opts.mandible_min_area * np.pi * ra * ra) and wide >= opts.mandible_width * ra and anterior:
                mand = blob
        if mand is not None:                                  # markers: the antenna's centre line, the blob's core
            mk = np.zeros(piece.shape, np.int32)
            mk[path & piece & ~dilate(mand, 1)] = 1
            core = erode(mand, max(1.0, 0.3 * float(edt(np.pad(mand, 1))[1:-1, 1:-1].max())))
            mk[core if core.any() else mand] = 2
            lab = flood(mk, cost, piece)
            mand = lab == 2; ant = (lab == 1) | (piece & (lab == 0))
            mys, mxs, ms, mt = body.st_mask(mand)                 # sided by its bulk: the curved tip may cross the midline
            mside = side_name(float(np.median(mt)), view) or side
            out.append(dict(mask=mand, name=sided(opts.side_format, mside, "mandible") if mside else "mandible",
                            source="antenna", method="broad piece at the antennal base (width, position) + L18 watershed",
                            confidence="medium", warnings=["mandible separated from the coarse antenna class"], side=mside,
                            evidence=f"antenna radius {ra:.1f} px, piece width x{wide / max(ra, 1e-6):.1f}"))
        else:
            ant = piece
        if lengths[i] < 0.4 * longest:
            warn.append(f"antenna piece short ({lengths[i]:.0f} px vs {longest:.0f}): fragment?")
        out.append(dict(mask=ant, name=sided(opts.side_format, side, "antenna") if side else "antenna", source="antenna",
                        method="side of the base (serial-homology side rule)", confidence="high" if not warn else "medium",
                        warnings=warn, side=side, evidence=f"base {tb:+.0f} px from the midline"))
    # same name twice (two pieces of one antenna): fine in COCO, but say so
    cnt = Counter(p["name"] for p in out)
    for p in out:
        if cnt[p["name"]] > 1:
            p["warnings"].append(f"{cnt[p['name']]} pieces named {p['name']}")
    return out, qc


def split_sided_pieces(m: np.ndarray, base_name: str, source: str, body: Body, view: Optional[str], opts,
                       head: Optional[np.ndarray] = None) -> List[dict]:
    """Pieces of a paired class (coarse 'mandibles') -> left/right by the side of each piece."""
    out = []
    cl, n = label8(m)
    for i in range(1, n + 1):
        piece = cl == i
        ref = base_pixels(piece, head) if head is not None and head.any() else piece
        ys, xs, s, t = body.st_mask(ref)
        side = side_name(float(np.median(t)), view)
        out.append(dict(mask=piece, name=sided(opts.side_format, side, base_name) if side else base_name, source=source,
                        method="side of the piece", confidence="medium" if side else "low",
                        warnings=[] if side else ["not sided"], side=side))
    return out


def split_mouthparts(m: np.ndarray, L: Labels, maps: Maps, body: Body, view: Optional[str], opts) -> List[dict]:
    """mouthparts -> labrum + left/right maxillary palp when a three-marker watershed gives that shape; else mouthparts."""
    fallback = [dict(mask=m, name=opts.mouthparts_name, source="mouthparts", method="kept (coarse class)",
                     confidence="medium", warnings=[])]
    head = L.mask("head") | L.mask("eye")
    if view not in ("dorsal", "ventral") or not head.any() or m.sum() < 50:
        fallback[0]["warnings"].append("mouthparts not split (view, head or size)")
        return fallback
    hys, hxs, hs, ht = body.st_mask(head)
    hw = float(np.percentile(np.abs(ht), 95)); hfront = float(np.percentile(hs, 98))
    ys, xs, s, t = body.st_mask(m)
    lab = np.zeros(m.shape, np.int32)
    lab_core = (np.abs(t) < 0.15 * hw) & (s <= hfront + 0.15 * hw) & (s >= hfront - 0.2 * hw)
    if lab_core.sum() < 5:
        fallback[0]["warnings"].append("no median piece at the head's anterior margin: mouthparts not split")
        return fallback
    lab[ys[lab_core], xs[lab_core]] = 1
    for k, sgn in ((2, 1), (3, -1)):
        lat = (sgn * t > 0.3 * hw)
        if lat.sum() < 5:
            fallback[0]["warnings"].append("no paired lateral pieces: mouthparts not split")
            return fallback
        # the most anterolateral pixels of that side
        score = s[lat] + 0.5 * np.abs(t[lat])
        thr = np.percentile(score, 85)
        sel = np.flatnonzero(lat)[score >= thr]
        lab[ys[sel], xs[sel]] = k
    cost = maps.combo_mag(opts.split_layer, opts.image_weight)
    res = flood(lab, cost, m)
    pieces = {k: res == k for k in (1, 2, 3)}
    ok = all(pieces[k].sum() >= 0.08 * m.sum() for k in (2, 3)) and pieces[1].sum() >= 0.08 * m.sum()
    el = {k: pca_elongation(pieces[k]) for k in (2, 3)}
    if not ok or min(el.values()) < opts.palp_elongation:
        fallback[0]["warnings"].append("labrum/palp split not supported by shape (pieces too small or palps not elongate: "
                                       + ", ".join(f"{v:.1f}" for v in el.values()) + "): kept as mouthparts")
        return fallback
    out = [dict(mask=pieces[1], name="labrum", source="mouthparts", method="median piece (3-marker watershed)",
                confidence="low", warnings=["labrum split by a shape rule: check"])]
    for k, sgn in ((2, 1), (3, -1)):
        side = side_name(float(sgn), view)
        out.append(dict(mask=pieces[k], name=sided(opts.side_format, side, "maxillary_palp"), source="mouthparts",
                        method="paired lateral piece (3-marker watershed)", confidence="low",
                        warnings=["maxillary palp split by a shape rule: check"], side=side,
                        evidence=f"elongation {el[k]:.1f}"))
    rest = m & ~(pieces[1] | pieces[2] | pieces[3])
    if rest.any():
        out[0]["mask"] = out[0]["mask"] | rest
    return out


# ------------------------------------------------------------------ eyes from L18
def kmeans(X: np.ndarray, k: int, it: int = 30, seed: int = 0) -> np.ndarray:
    """Plain k-means (k-means++ start, fixed seed) -> labels."""
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    C = np.array(C)
    lab = np.zeros(len(X), int)
    for _ in range(it):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab


def find_eyes(L: Labels, maps: Maps, body: Body, view: Optional[str], opts) -> Tuple[List[dict], List[str], np.ndarray]:
    """Eyes from L18: the head's whole lateral extent (the coarse model puts parts of the eyes into the antenna class) is
    clustered on its L18 tokens (k-means, k = 2..5); the cluster whose lateral pieces look most like a PAIR of eyes
    (compact, oval, at the lateral outline, similar size and AP level on both sides, absent from the medial head) is
    taken; each eye is then sharpened against its surroundings. -> (eyes, notes, pixels taken)."""
    notes = []
    none = np.zeros(L.lab.shape, bool)
    hm = L.mask("head") | L.mask("eye")
    if maps.F18 is None or not hm.any() or view not in ("dorsal", "ventral"):
        return [], ["eyes: no L18 features, head or dorsal/ventral view"], none
    hys, hxs, hs, ht = body.st_mask(hm)
    s_lo, s_hi = float(np.percentile(hs, 1)), float(np.percentile(hs, 99))
    hw = float(np.percentile(np.abs(ht), 95)) or 1.0
    hlen = max(s_hi - s_lo, 1.0)
    # the head's own outline: head + eye + antenna + mandible OPENED by a disc, so that thin antennae and mandible tips
    # fall away and the bulging eyes stay
    cand = hm | L.mask("antenna") | L.mask("mandible")
    r_open = opts.eye_open_frac * hw
    cand = dilate(erode(cand, r_open), r_open) & cand | hm
    ys, xs, s, t = body.st_mask(cand)
    sel = (s >= s_lo) & (s <= s_hi) & (np.abs(t) <= opts.eye_reach * hw)
    ys, xs, s, t = ys[sel], xs[sel], s[sel], t[sel]
    if len(ys) < 50:
        return [], ["eyes: head region too small"], none
    region = np.zeros(L.lab.shape, bool); region[ys, xs] = True
    head_area = float(hm.sum())
    # local half width of the region per AP slice (for 'lateral' and 'at the outline')
    sb = np.round(s).astype(int)
    hw_s = np.zeros(sb.max() - sb.min() + 1)
    np.maximum.at(hw_s, sb - sb.min(), np.abs(t))
    loc_hw = hw_s[sb - sb.min()]
    lateral = np.abs(t) >= opts.eye_lateral_frac * loc_hw
    depth = edt(np.pad(region, 1))[1:-1, 1:-1]               # distance inside the region from its outline
    step, origin = maps.grid
    F = maps.F18
    _, gh, gw = F.shape
    gi = np.clip(np.round((ys - origin) / step).astype(int), 0, gh - 1)
    gj = np.clip(np.round((xs - origin) / step).astype(int), 0, gw - 1)
    uc, inv = np.unique(gi * gw + gj, return_inverse=True)
    X = F[:, uc // gw, uc % gw].T.astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Xp = X[inv]                                               # per pixel
    med_mu = Xp[np.abs(t) < 0.3 * hw].mean(0) if (np.abs(t) < 0.3 * hw).any() else Xp.mean(0)
    coarse_eye = L.mask("eye")
    best = None
    for k in range(2, 6):
        lab = kmeans(X, k)[inv]
        dist = {c: float(np.linalg.norm(Xp[lab == c].mean(0) - med_mu)) if (lab == c).any() else 0.0 for c in range(k)}
        dmax = max(dist.values()) or 1.0
        for c in range(k):
            inc = lab == c
            if (inc & (np.abs(t) < 0.3 * hw)).sum() > 0.2 * max(1, inc.sum()):
                continue                                      # this cluster is the head capsule
            sides = {}
            for sgn in (1, -1):
                q_sel = inc & lateral & (sgn * t > 0)
                if q_sel.sum() < 10:
                    continue
                cm = np.zeros(L.lab.shape, bool); cm[ys[q_sel], xs[q_sel]] = True
                cm = erode(dilate(cm, 1.5), 1.5) & region
                cl, n = label8(cm)
                for i in range(1, n + 1):
                    q = cl == i
                    a = float(q.sum())
                    if not (opts.eye_min_area * head_area <= a <= opts.eye_max_area * head_area):
                        continue
                    qy, qx, qs, qt = body.st_mask(q)
                    compact = a / max(1.0, float((np.ptp(qs) + 1) * (np.ptp(qt) + 1)))
                    el = pca_elongation(q)
                    touch = float(depth[q].min()) <= max(3.0, 0.06 * hw)      # reaches the head's lateral outline
                    if compact < opts.eye_min_compact or el > opts.eye_max_elongation or not touch:
                        continue
                    if sgn not in sides or a > sides[sgn][1]:
                        sides[sgn] = (q, a, float(np.median(qs)), compact)
            if not sides:
                continue
            score = 2.0 * len(sides) + float(np.mean([v[3] for v in sides.values()])) + opts.eye_distinct_weight * dist[c] / dmax
            if opts.eye_position != "any":                    # where the eyes sit along the head (temples lie behind them)
                pos = float(np.mean([(v[2] - s_lo) / hlen for v in sides.values()]))       # 0 = back, 1 = front
                score += opts.eye_position_weight * ((pos - 0.5) if opts.eye_position == "anterior" else (0.5 - pos))
            if coarse_eye.any():                              # L23 says 'eye' near here
                near_eye = dilate(coarse_eye, opts.eye_prior_px)
                score += opts.eye_prior_weight * float(np.mean([(v[0] & near_eye).sum() / v[1] for v in sides.values()]))
            if len(sides) == 2:
                (q1, a1, s1, _), (q2, a2, s2, _) = sides[1], sides[-1]
                score += (1 - abs(a1 - a2) / max(a1, a2)) + (1 - min(1.0, abs(s1 - s2) / (0.3 * hlen)))
            if best is None or score > best[0]:
                best = (score, k, c, sides)
    if best is None:
        return [], ["eyes: no paired compact lateral L18 cluster in the head"], none
    score, k, c, sides = best
    cost = maps.combo_mag(opts.sharpen_layer if opts.sharpen_layer != "none" else opts.split_layer, opts.image_weight)
    eyes, taken = [], np.zeros(L.lab.shape, bool)
    for sgn, (q, a, sm, compact) in sides.items():
        q = ndimage.binary_fill_holes(q)
        lab = np.zeros(L.lab.shape, np.int32)
        lab[region] = 1; lab[q] = 2
        lab = sharpen(lab, cost, region, opts.eye_band_px)
        q = lab == 2
        side = side_name(float(sgn), view)
        eyes.append(dict(mask=q, name=sided(opts.side_format, side, "eye"), source="head (L18 tokens)",
                         method=f"L18 token cluster (k-means k={k}), paired compact lateral pieces; watershed outline",
                         confidence="medium" if len(sides) == 2 and score >= opts.eye_min_score else "low",
                         warnings=["eye found by an unsupervised L18 cluster rule: check"] +
                                  ([] if len(sides) == 2 else ["only one eye found from L18"]) +
                                  ([f"weak pair score {score:.1f} < --eye_min_score {opts.eye_min_score}"]
                                   if score < opts.eye_min_score else []), side=side,
                         evidence=f"area {int(q.sum())} px ({q.sum() / head_area:.1%} of head), compactness {compact:.2f}, "
                                  f"pair score {score:.2f}"))
        taken |= q
    return eyes, notes, taken


def coarse_eyes(L: Labels, body: Body, view: Optional[str], opts, found_sides) -> List[dict]:
    """Coarse 'eye' pieces kept only when coherent and at the lateral head margin (for sides L18 did not find)."""
    out = []
    em = L.mask("eye")
    if not em.any():
        return out
    head = L.mask("head") | em
    hys, hxs, hs, ht = body.st_mask(head)
    hwh = float(np.percentile(np.abs(ht), 95)) or 1.0
    cl, n = label8(em)
    for i in range(1, n + 1):
        q = cl == i
        qy, qx, qs, qt = body.st_mask(q)
        side = side_name(float(np.median(qt)), view)
        if side in found_sides:
            continue
        compact = q.sum() / max(1.0, float((np.ptp(qy) + 1) * (np.ptp(qx) + 1)))
        lateral = float(np.median(np.abs(qt))) >= 0.5 * hwh
        ok = q.sum() >= 30 and compact >= 0.3 and lateral
        out.append(dict(mask=q, name=(sided(opts.side_format, side, "eye") if side else "eye"), source="eye",
                        method="coarse eye kept", confidence="low", keep=ok, side=side,
                        warnings=[("eye not found from L18 on this side: coarse eye kept (coherent, lateral)" if ok else
                                   f"coarse eye piece dropped (area {int(q.sum())} px, compactness {compact:.2f}, "
                                   f"{'lateral' if lateral else 'not at the lateral margin'})")]))
    return out


# ------------------------------------------------------------------ body leftovers
def classify_leftovers(L: Labels, body: Body, view: Optional[str], fg_area: int, opts) -> List[dict]:
    out = []
    bm = L.mask("body")
    if not bm.any():
        return out
    pro, ely, abd = L.mask("pronotum"), L.mask("elytra"), L.mask("abdomen")
    trunk = (L.lab > 0)
    for r in APPENDAGES:
        trunk &= ~L.mask(r)
    tys, txs, ts, tt = body.st_mask(trunk)
    thw = float(np.percentile(np.abs(tt), 95)) or 1.0
    p_post = float(np.percentile(body.st_mask(pro)[2], 2)) if pro.any() else None
    e_ant = float(np.percentile(body.st_mask(ely)[2], 98)) if ely.any() else None
    e_post = float(np.percentile(body.st_mask(ely)[2], 2)) if ely.any() else None
    e_len = (e_ant - e_post) if ely.any() else 0.0
    cl, n = label8(bm)
    for i in range(1, n + 1):
        q = cl == i
        a = int(q.sum())
        qy, qx, qs, qt = body.st_mask(q)
        rim = ring(q, 2)
        touch = {nm: bool((rim & mm).any()) for nm, mm in (("pronotum", pro), ("elytra", ely), ("abdomen", abd))}
        sc, tc = float(np.median(qs)), float(np.median(qt))
        if a < opts.min_piece_frac * fg_area:
            nb, share = neighbour_label(L.lab, q, exclude=(0,) + tuple(L.idx("body")))
            out.append(dict(mask=q, name=L.names[nb] if nb else "unassigned", source="body", method="speck joined its neighbour",
                            confidence="medium", warnings=[], absorb=nb))
            continue
        zone_pe = (p_post is not None and e_ant is not None and touch["pronotum"] and touch["elytra"]
                   and e_ant - 0.4 * e_len <= sc <= p_post + 0.1 * e_len)
        if zone_pe and view in ("dorsal", None):
            rng = float(np.ptp(qs)) or 1.0
            front = np.abs(qt[qs >= qs.max() - rng / 3]); back = np.abs(qt[qs <= qs.min() + rng / 3])
            w_front = float(np.ptp(qt[qs >= qs.max() - rng / 3])) if len(front) else 0
            w_back = float(np.ptp(qt[qs <= qs.min() + rng / 3])) if len(back) else 0
            centred = abs(tc) <= 0.15 * thw
            width = float(np.ptp(qt))
            if centred and w_front >= 1.4 * max(w_back, 1) and width < 0.6 * 2 * thw:
                out.append(dict(mask=q, name=opts.scutellum_name, source="body", method="leftover on the midline between "
                                "pronotum and elytra, narrowing posteriorly", confidence="medium", warnings=[],
                                evidence=f"width front/back {w_front:.0f}/{w_back:.0f} px, centre {tc:+.0f} px"))
                continue
            if abs(tc) <= 0.25 * thw and width >= 0.4 * 2 * thw:
                out.append(dict(mask=q, name=opts.mesonotum_name, source="body", method="transverse leftover across the "
                                "midline between pronotum and elytra", confidence="low", warnings=["mesonotum by position: check"]))
                continue
        where = ("between elytra and abdomen" if touch["elytra"] and touch["abdomen"] else
                 "between pronotum and elytra" if zone_pe else "elsewhere")
        out.append(dict(mask=q, name=opts.unassigned_name, source="body", method="leftover", confidence="low",
                        warnings=[f"exposed region {where} ({a} px) not assigned to a part"]))
    return out


# ------------------------------------------------------------------ one image
def split_image(rgb: np.ndarray, masks: Dict[str, np.ndarray], fg: np.ndarray, maps: Maps, view: Optional[str],
                im: dict, opts, fg_note: str = "") -> dict:
    """Coarse masks + foreground + boundary maps -> fine parts. -> dict(parts, stages, qc, notes, cuts)."""
    role_of = role_matcher(opts)
    H, W = fg.shape
    union = np.zeros(fg.shape, bool)
    for v in masks.values():
        union |= v
    # background the coarse masks covered: windows between appendages (enclosed or open to the outside) and spill over
    # the outline; pieces of at least --window_min_px count as windows
    holes = ndimage.binary_fill_holes(fg) & ~fg
    win = union & ~fg
    wl, wn = label8(win)
    wsz = np.bincount(wl.ravel())[1:] if wn else np.array([])
    big = wsz >= opts.window_min_px
    n_windows = int(big.sum())
    n_encl = int(sum(1 for i in np.flatnonzero(big) if (holes[wl == i + 1]).mean() > 0.5)) if n_windows else 0
    notes = [fg_note] if fg_note else []
    notes.append(f"windows removed: {n_windows} background pieces >= {opts.window_min_px} px that the coarse masks covered "
                 f"({int(wsz[big].sum()) if n_windows else 0} px; {n_encl} of them enclosed by the specimen); "
                 f"{int(win.sum())} px of background removed in all")
    summary = {"windows_removed": n_windows, "windows_enclosed": n_encl, "background_removed_px": int(win.sum())}
    L0 = build_labels(masks, fg, role_of, opts)
    fg_area = int(fg.sum())
    notes += absorb_specks(L0, fg_area, opts)
    stages = {"coarse": L0.copy()}
    # sharpen
    if opts.sharpen_layer != "none":
        cost = maps.combo_mag(opts.sharpen_layer, opts.image_weight)
        L1 = L0.copy()
        L1.lab = sharpen(L0.lab, cost, fg, opts.band_px)
    else:
        L1 = L0.copy()
    stages["sharpened"] = L1.copy()
    body, fwarn = body_of(L1, opts)
    parts: List[dict] = []
    qc = {"windows": np.isin(wl, 1 + np.flatnonzero(big)) if wn else win,
          "suture": [], "accepted": [], "rejected": [], "leg_cuts": [], "coarse": L0}
    cuts_rows: List[dict] = []
    if body is None:
        notes.append("no body frame: " + "; ".join(fwarn) + "; parts kept as coarse classes")
        for k in range(1, len(L1.names)):
            mk = L1.lab == k
            if mk.any():
                parts.append(dict(mask=mk, name=L1.names[k], source=L1.names[k], method="kept (no body frame)",
                                  confidence="low", warnings=["no body frame"]))
        return dict(parts=parts, stages=stages, qc=qc, notes=notes, cuts=cuts_rows, body=None, summary=summary)
    notes += [f"body frame: {w}" for w in fwarn]
    # eyes from L18 first: they take pixels from the head region
    eyes, enotes, taken = find_eyes(L1, maps, body, view, opts)
    notes += enotes
    found = {e.get("side") for e in eyes}
    ce = coarse_eyes(L1, body, view, opts, found)
    eye_pixels = taken.copy()
    for e in ce:
        if e.get("keep"):
            eyes.append(e); eye_pixels |= e["mask"]
        else:
            notes.append(e["warnings"][0])
    # pixels of the coarse 'eye' not kept go to the head (or the part around them)
    head_k = (L1.idx("head") or [None])[0]
    leftover_eye = L1.mask("eye") & ~eye_pixels
    if leftover_eye.any() and head_k is not None:
        L1.lab[leftover_eye] = head_k
    for k in range(1, len(L1.names)):
        mk = (L1.lab == k) & ~eye_pixels
        if not mk.any():
            continue
        role, nm = L1.roles[k], L1.names[k]
        if role == "eye":
            continue
        if role == "elytra":
            ps, q = split_elytra(mk, maps, body, view, opts)
            for p in ps:
                p["source"] = nm
            parts += ps; qc["suture"] = q["suture"]
        elif role == "abdomen":
            ps, q, cr = split_abdomen(mk, maps, body, view, opts)
            for p in ps:
                p["source"] = nm
            parts += ps; qc["accepted"] += q["accepted"]; qc["rejected"] += q["rejected"]
            for c in cr:
                c.update(image=im.get("file_name", ""), structure=nm)
            cuts_rows += cr
        elif role == "leg":
            trunk = (L1.lab > 0)
            for r in APPENDAGES:
                trunk &= ~L1.mask(r)
            cl, n = label8(mk)
            sizes = np.bincount(cl.ravel())[1:] if n else []
            legs = []
            for i in range(1, n + 1):
                if sizes[i - 1] < opts.min_piece_frac * fg_area:
                    continue
                ps, q = split_leg(cl == i, trunk, maps, view, opts)
                for p in ps:
                    p["source"] = nm
                legs += ps; qc["leg_cuts"] += q["cuts"]
            name_legs(legs, L1, im, view, opts)
            parts += legs
        elif role == "antenna":
            ps, _ = split_antenna(mk, L1, maps, body, view, opts)
            parts += ps
        elif role == "mandible":
            parts += split_sided_pieces(mk, "mandible", nm, body, view, opts, L1.mask("head"))
        elif role == "mouthparts":
            parts += split_mouthparts(mk, L1, maps, body, view, opts)
        elif role == "body":
            continue
        else:
            parts.append(dict(mask=mk, name=nm, source=nm, method=(f"L23 class, {opts.sharpen_layer}-sharpened outline"
                                                                   if opts.sharpen_layer != "none" else "L23 class, coarse outline"),
                              confidence="high", warnings=[]))
    for e in eyes:
        e.setdefault("source", "eye")
    parts += eyes
    # body leftovers (after the eyes took their pixels)
    Lb = L1.copy()
    Lb.lab[eye_pixels] = 0
    for p in classify_leftovers(Lb, body, view, fg_area, opts):
        if p.get("absorb"):
            tgt = next((q for q in parts if q.get("source") == L1.names[p["absorb"]] and
                        (dilate(q["mask"], 2) & p["mask"]).any()), None)
            if tgt is not None:
                tgt["mask"] = tgt["mask"] | p["mask"]
                continue
            p["name"] = L1.names[p["absorb"]]
        parts.append(p)
    for p in parts:
        p.setdefault("warnings", [])
        p.setdefault("source", "")
    segs = [p["name"] for p in parts if p.get("segment") not in (None, "")]
    summary.update(segments=len([p for p in parts if isinstance(p.get("segment"), int)]), segment_names=", ".join(segs),
                   apex_present=next((p["apex_present"] for p in parts if p.get("apex_present")), ""),
                   scutellum="yes" if any(p["name"] == opts.scutellum_name for p in parts) else "no",
                   mesonotum="yes" if any(p["name"] == opts.mesonotum_name for p in parts) else "no",
                   eyes="; ".join(f"{e['name']} ({e['method'].split(' (')[0].split(',')[0]})" for e in eyes) or "none",
                   unassigned_pieces=sum(1 for p in parts if p["name"] == opts.unassigned_name))
    return dict(parts=parts, stages=stages, qc=qc, notes=notes, cuts=cuts_rows, body=body, summary=summary)


# ------------------------------------------------------------------ polygons
MOORE = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]   # (dx, dy) clockwise, y down


def trace_outer(m: np.ndarray) -> np.ndarray:
    """Outer boundary of one 8-connected piece (Moore-neighbour tracing) -> (n, 2) pixel centres (x, y)."""
    p = np.pad(m, 1)
    ys, xs = np.nonzero(p)
    i = int(np.lexsort((xs, ys))[0])
    start = (int(xs[i]), int(ys[i]))
    cont = [start]
    cur, b = start, 4                                       # backtrack = west neighbour (background by raster order)
    first_next = None
    for _ in range(8 * p.size):
        nxt = None
        for k in range(1, 9):
            d = (b + k) % 8
            x, y = cur[0] + MOORE[d][0], cur[1] + MOORE[d][1]
            if p[y, x]:
                pb = (b + k - 1) % 8
                bx, by = cur[0] + MOORE[pb][0], cur[1] + MOORE[pb][1]
                nxt = (x, y)
                b = MOORE.index((bx - x, by - y))
                break
        if nxt is None:
            break                                            # single pixel
        if cur == start:
            if first_next is None:
                first_next = nxt
            elif nxt == first_next:
                break
        cont.append(nxt); cur = nxt
    if len(cont) > 1 and cont[-1] == start:
        cont = cont[:-1]
    return np.array(cont, float) - 1


def rdp(P: np.ndarray, eps: float) -> np.ndarray:
    """Douglas-Peucker on an open polyline."""
    if len(P) < 3:
        return P
    keep = np.zeros(len(P), bool); keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a; n = np.hypot(*ab)
        seg = P[i + 1:j]
        d = (np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / n if n > 0 else
             np.hypot(*(seg - a).T))
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return P[keep]


def simplify_closed(P: np.ndarray, eps: float) -> np.ndarray:
    if len(P) < 8 or eps <= 0:
        return P
    k = int(np.argmax(np.hypot(*(P - P[0]).T)))
    a = rdp(P[:k + 1], eps); b = rdp(np.vstack([P[k:], P[:1]]), eps)
    return np.vstack([a[:-1], b[:-1]])


def mask_polygons(m: np.ndarray, eps: float = 1.0, min_px: int = 10, min_frac: float = 0.1) -> List[List[float]]:
    """Outer polygons of the pieces of a mask (pieces >= min_frac of the largest and >= min_px, as DINOMask v6 keeps)."""
    cl, n = label8(m)
    if not n:
        return []
    sizes = np.bincount(cl.ravel())[1:]
    out = []
    for i, sl in enumerate(ndimage.find_objects(cl), 1):
        if sizes[i - 1] < max(min_px, min_frac * sizes.max()):
            continue
        sub = cl[sl] == i
        P = trace_outer(sub) + [sl[1].start, sl[0].start]
        P = simplify_closed(P, eps)
        if len(P) >= 3:
            out.append(P.ravel().round(1).tolist())
    return out


# ------------------------------------------------------------------ evaluation against hand labels
def boundary_px(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, np.ones((3, 3), bool), border_value=0)


def bf_score(pred: np.ndarray, gt: np.ndarray, tol: float, region: Optional[np.ndarray] = None) -> float:
    bp, bg = boundary_px(pred), boundary_px(gt)
    if region is not None:
        bp &= region; bg &= region
    if not bp.any() and not bg.any():
        return float("nan")
    if not bp.any() or not bg.any():
        return 0.0
    P = float((edt(~bg)[bp] <= tol).mean()); R = float((edt(~bp)[bg] <= tol).mean())
    return 2 * P * R / (P + R) if P + R else 0.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def gt_masks(gt: dict, file_name: str, shape) -> Dict[str, np.ndarray]:
    """{category name: union mask} for the GT image with this base name (line annotations skipped)."""
    H, W = shape
    cats = {c["id"]: c["name"] for c in gt.get("categories", [])}
    ids = {im["id"] for im in gt.get("images", []) if Path(im.get("file_name", "")).name == Path(file_name).name}
    out = {}
    for a in gt.get("annotations", []):
        if a.get("image_id") not in ids or a.get("is_line"):
            continue
        m = sh.ann_mask(a, W, H, 1.0)
        if m is None or not m.any():
            continue
        nm = cats.get(a.get("category_id"), "")
        out[nm] = out.get(nm, np.zeros((H, W), bool)) | m
    return out


def evaluate(res: dict, gt: dict, catmap: Dict[str, str], file_name: str, fg: np.ndarray, opts,
             final_masks: Dict[str, np.ndarray]) -> List[dict]:
    """Coarse and sharpened coarse classes vs the GT merged with catmap; fine parts vs GT fine names."""
    rows = []
    H, W = fg.shape
    G = gt_masks(gt, file_name, (H, W))
    if not G:
        return rows
    role_of = role_matcher(opts)
    ren = parse_rename(opts.rename)
    # coarse GT label map: catmap, then rename, painted like the coarse input
    merged = defaultdict(lambda: np.zeros((H, W), bool))
    for nm, m in G.items():
        c = catmap.get(nm, nm)
        c = ren.get(c.lower(), c)
        merged[c] |= m
    gtL = build_labels(dict(merged), np.ones((H, W), bool) & np.any(np.stack(list(merged.values())), 0), role_of, opts)
    gfg = gtL.lab > 0
    internal = erode(gfg & fg, 2)
    for stage in ("coarse", "sharpened"):
        Ls = res["stages"][stage]
        for k in range(1, len(gtL.names)):
            nm = gtL.names[k]
            g = gtL.lab == k
            if nm not in Ls.names or not g.any():
                continue
            p = np.isin(Ls.lab, [i for i, n in enumerate(Ls.names) if n == nm])
            rows.append(dict(image=file_name, stage=stage, part=nm, iou=iou(p, g), bf2=bf_score(p, g, 2), bf4=bf_score(p, g, 4),
                             bf2_internal=bf_score(p, g, 2, internal), bf4_internal=bf_score(p, g, 4, internal),
                             gt_px=int(g.sum()), pred_px=int(p.sum())))
    # fine: by canonical names; unsided GT names compared with the union of the sided predictions
    P = defaultdict(lambda: np.zeros((H, W), bool))
    for nm, m in final_masks.items():
        P[canon(nm)] |= m
    for nm, g in G.items():
        key = canon(nm)
        if role_of(catmap.get(nm, nm)) == "body":
            continue
        if key in P:
            p = P[key]; how = ""
        else:
            sides = [P[k] for k in (f"left_{key}", f"right_{key}") if k in P]
            if not sides:
                rows.append(dict(image=file_name, stage="fine", part=nm, iou=0.0, bf2=0.0, bf4=0.0, bf2_internal=float("nan"),
                                 bf4_internal=float("nan"), gt_px=int(g.sum()), pred_px=0, note="no prediction of this name"))
                continue
            p = np.any(np.stack(sides), 0); how = "union of left_/right_"
        rows.append(dict(image=file_name, stage="fine", part=nm, iou=iou(p, g), bf2=bf_score(p, g, 2), bf4=bf_score(p, g, 4),
                         bf2_internal=bf_score(p, g, 2, internal), bf4_internal=bf_score(p, g, 4, internal),
                         gt_px=int(g.sum()), pred_px=int(p.sum()), note=how))
    return rows


# ------------------------------------------------------------------ QC figure
def colour_of(name: str) -> Tuple[int, int, int]:
    h = (sum(ord(c) * (i + 7) for i, c in enumerate(name)) * 0.61803) % 1.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size=12):
    for f in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def qc_figure(rgb: np.ndarray, res: dict, final: List[dict], maps: Maps, title: str, path: Path, opts) -> None:
    H, W = rgb.shape[:2]
    base = Image.fromarray(rgb)
    font = _font(max(10, int(H / 90)))
    # panel 1: photo, coarse outlines, removed windows in red
    p1 = base.copy().convert("RGBA")
    ov = np.zeros((H, W, 4), np.uint8)
    ov[res["qc"]["windows"]] = (255, 0, 0, 170)
    L0 = res["qc"]["coarse"]
    for k in range(1, len(L0.names)):
        b = boundary_px(L0.lab == k)
        ov[b] = colour_of(L0.names[k]) + (255,)
    p1 = Image.alpha_composite(p1, Image.fromarray(ov)).convert("RGB")
    # panel 2: boundary strength (L18 + image), cuts
    B = maps.combo_mag(opts.split_layer, opts.image_weight)
    g = np.clip(255 - 255 * B / max(np.percentile(B, 99), 1e-6), 0, 255).astype(np.uint8)
    p2 = Image.fromarray(np.stack([g] * 3, -1))
    d2 = ImageDraw.Draw(p2)
    for line in res["qc"]["rejected"]:
        d2.line(line, fill=(230, 0, 0), width=2)
    for line in res["qc"]["accepted"]:
        d2.line(line, fill=(0, 170, 0), width=3)
    if res["qc"]["suture"]:
        d2.line(res["qc"]["suture"], fill=(0, 160, 255), width=2)
    for (x, y, nx, ny, r) in res["qc"]["leg_cuts"]:
        d2.line([(x - nx * r, y - ny * r), (x + nx * r, y + ny * r)], fill=(255, 140, 0), width=3)
    # panel 3: fine parts
    p3 = base.copy().convert("RGBA")
    ov = np.zeros((H, W, 4), np.uint8)
    for p in final:
        c = colour_of(p["name"])
        ov[p["mask"]] = c + (110,)
    for p in final:
        ov[boundary_px(p["mask"])] = colour_of(p["name"]) + (255,)
    p3 = Image.alpha_composite(p3, Image.fromarray(ov)).convert("RGB")
    d3 = ImageDraw.Draw(p3)
    for p in final:
        ys, xs = np.nonzero(p["mask"])
        if not len(xs):
            continue
        i = int(np.argmax(edt(np.pad(p["mask"], 1))[1:-1, 1:-1][ys, xs]))
        lbl = p["name"].replace("tergite ", "T").replace("sternite ", "S").replace("abdominal segment ", "A")
        d3.text((int(xs[i]) - 3 * len(lbl), int(ys[i]) - 6), lbl, fill=(0, 0, 0), font=font, stroke_width=2,
                stroke_fill=(255, 255, 255))
    gap = 10
    fig = Image.new("RGB", (3 * W + 2 * gap, H + 30), "white")
    for i, p in enumerate((p1, p2, p3)):
        fig.paste(p, (i * (W + gap), 30))
    d = ImageDraw.Draw(fig)
    d.text((6, 6), f"{title}   1 coarse (L23) + removed windows (red) | 2 boundary {opts.split_layer}"
                   f"{' + image' if opts.image_weight else ''}: cuts accepted green, rejected red, suture blue, joints orange"
                   f" | 3 fine parts", fill=(0, 0, 0), font=_font(12))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.save(path)


# ------------------------------------------------------------------ COCO in / out
def coarse_masks(coco: dict, im_ids, W: int, H: int, cats: Dict[int, str], ren: Dict[str, str],
                 catmap: Optional[Dict[str, str]] = None) -> Dict[str, np.ndarray]:
    """{coarse name: union mask} of one image; names through --input_catmap, then --rename; line annotations skipped."""
    out = {}
    for a in coco.get("annotations", []):
        if a.get("image_id") not in im_ids or a.get("is_line"):
            continue
        nm = cats.get(a.get("category_id"), "")
        nm = (catmap or {}).get(nm, nm)
        nm = ren.get(nm.lower(), nm)
        m = sh.ann_mask(a, W, H, 1.0)
        if m is None or not m.any():
            continue
        out[nm] = out.get(nm, np.zeros((H, W), bool)) | m
    return out


def find_image(file_name: str, opts) -> Optional[Path]:
    p = Path(file_name)
    cands = ([p] if p.is_absolute() else []) + ([Path(opts.images) / p.name, Path(opts.images) / file_name] if opts.images else [])
    return next((c for c in cands if c.is_file()), None)


def run(coco: dict, opts) -> Tuple[dict, List[dict], List[dict], List[dict], List[str]]:
    """-> (new COCO, part rows, cut rows, eval rows, notes)."""
    ren = parse_rename(opts.rename)
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    gt = load(opts.gt) if opts.gt else None
    in_catmap = load(opts.input_catmap) if opts.input_catmap else {}
    catmap = load(opts.gt_catmap) if opts.gt_catmap else {}
    if isinstance(catmap, dict) and "categories" in catmap:
        catmap = {}
    ims, seen = [], {}
    for im in coco.get("images", []):
        key = im.get("id")
        if key in seen:
            continue
        seen[key] = im; ims.append(im)
    out = {"images": [], "categories": [], "annotations": []}
    cat_ids: Dict[str, int] = {}
    rows, cut_rows, eval_rows, notes, img_rows = [], [], [], [], []
    for im in ims:
        fn = im.get("file_name", "")
        if opts.only and not re.search(opts.only, fn):
            continue
        path = find_image(fn, opts)
        if path is None:
            notes.append(f"{fn}: image not found (--images): skipped"); continue
        rgb = np.asarray(Image.open(path).convert("RGB"))
        H, W = rgb.shape[:2]
        masks = coarse_masks(coco, {im.get("id")}, W, H, cats, ren, in_catmap)
        L_tmp = None
        fgp = fg_mask_path(fn, opts)
        if fgp is not None:
            fg = load_fg_mask(fgp, (H, W))
            fg, npin, _ = fill_pinholes(fg, pinhole_px((H, W), opts))
            fg_note = f"foreground from {fgp.name} ({npin} pinholes filled)"
        else:
            fg = derive_fg(rgb, opts)
            role_of = role_matcher(opts)
            L_tmp = build_labels(masks, np.ones((H, W), bool), role_of, opts) if masks else None
            glare_ok = (lambda k: L_tmp.roles[k] not in APPENDAGES) if L_tmp is not None else None
            fg, npin, nglare = fill_pinholes(fg, pinhole_px((H, W), opts), L_tmp.lab if L_tmp is not None else None, glare_ok)
            fg_note = (f"foreground from the background colour ({npin} pinholes and {nglare} glare holes filled; "
                       f"other holes kept as windows)" + (" [--fg_masks given but no mask for this image]" if opts.fg_masks else ""))
        # boundary maps
        mp = None
        npz_path = Path(opts.maps_dir) / f"{Path(fn).stem}_segsplit_maps.npz" if opts.maps_dir else None
        if npz_path is not None and npz_path.is_file() and not opts.recompute_maps:
            npz = dict(np.load(npz_path))
            if tuple(npz.get("shape", (H, W))) != (H, W):
                notes.append(f"{fn}: {npz_path.name} is for another image size: recomputed"); npz = None
        else:
            npz = None
        if npz is None and opts.tokens != "off":
            npz = compute_token_maps(rgb, fg, opts)
            if npz_path is not None:
                npz_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(npz_path, **npz)
        if opts.maps_only:
            print(f"maps: {fn}" + (f" -> {npz_path}" if npz_path else ""))
            continue
        if npz is not None:
            mp = maps_from_npz(npz, (H, W), fg, rgb, opts)
        else:
            mp = Maps((H, W)); mp.J["img"] = image_tensor(rgb, fg)
            notes.append(f"{fn}: no token maps (--tokens off): image gradient only")
        if not masks:
            notes.append(f"{fn}: no coarse masks: skipped"); continue
        view, vsrc = sh.image_view(im, opts)
        res = split_image(rgb, masks, fg, mp, view, im, opts, fg_note)
        final = [p for p in res["parts"] if p["mask"].any()]
        new_im = {k: v for k, v in im.items()}
        new_im.update(width=W, height=H)
        out["images"].append(new_im)
        final_masks: Dict[str, np.ndarray] = {}
        for p in final:
            polys = mask_polygons(p["mask"], opts.simplify_px)
            if not polys:
                continue
            if p["name"] not in cat_ids:
                cat_ids[p["name"]] = len(cat_ids) + 1
            aid = len(out["annotations"]) + 1
            ys, xs = np.nonzero(p["mask"])
            out["annotations"].append({"id": aid, "image_id": im.get("id"), "category_id": cat_ids[p["name"]],
                                       "segmentation": polys, "area": float(p["mask"].sum()),
                                       "bbox": [int(xs.min()), int(ys.min()), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)],
                                       "iscrowd": 0})
            polym = sh.raster([np.asarray(q).reshape(-1, 2) for q in polys], W, H)
            final_masks[p["name"]] = final_masks.get(p["name"], np.zeros((H, W), bool)) | polym
            holes = int((ndimage.binary_fill_holes(p["mask"]) & ~p["mask"]).sum())
            w = list(p.get("warnings", []))
            if holes > 0.02 * p["mask"].sum():
                w.append(f"mask has holes ({holes} px) that the outer polygon fills")
            rows.append(dict(image=Path(fn).name, image_id=im.get("id"), annotation_id=aid, part=p["name"],
                             source_part=p.get("source", ""), method=p.get("method", ""), confidence=p.get("confidence", ""),
                             view=view or "", side=p.get("side", "") or "", segment=p.get("segment", ""),
                             apex_present=p.get("apex_present", ""), area_px=int(p["mask"].sum()),
                             evidence=p.get("evidence", ""), warnings="; ".join(dict.fromkeys(w))))
        cut_rows += res["cuts"]
        img_rows.append(dict(image=Path(fn).name, view=view or "", **res.get("summary", {})))
        notes += [f"{Path(fn).name}: {n}" for n in res["notes"]]
        if opts.qc_dir:
            qc_figure(rgb, res, final, mp, Path(fn).name, Path(opts.qc_dir) / f"{Path(fn).stem}_segsplit_qc.png", opts)
        if gt is not None:
            eval_rows += evaluate(res, gt, catmap, fn, fg, opts, final_masks)
    out["categories"] = [{"id": i, "name": n, "supercategory": ""} for n, i in cat_ids.items()]
    out["info"] = {"descriptron_segment_split": {
        "program": VERSION, "parts": len(out["annotations"]),
        "options": {k: v for k, v in vars(opts).items() if not k.startswith("_") and k not in ("input", "out")},
        "images": img_rows}}
    opts._img_rows = img_rows
    return out, rows, cut_rows, eval_rows, notes


def write_csv(rows: List[dict], path: str):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog="\n\n".join(__doc__.split("\n\n")[1:]),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="COCO json with COARSE part masks (polygons or RLE), e.g. DINOMask predictions_polygons.json")
    ap.add_argument("--out", required=True, help="new COCO json with the fine parts (polygons; the input is not changed)")
    ap.add_argument("--images", default="", help="folder of the images (looked up by file name; explicit names only)")
    ap.add_argument("--csv", help="per-part report (default: <out>_parts.csv); cuts go to <out>_cuts.csv")
    ap.add_argument("--qc_dir", "--qc-dir", default="", help="write a QC PNG per image here")
    ap.add_argument("--only", default="", help="process only images whose file name matches this regex")
    g = ap.add_argument_group("foreground (step 1)")
    g.add_argument("--fg_masks", "--fg-masks", default="", help="folder of binary specimen masks <stem><--fg_suffix>")
    g.add_argument("--fg_suffix", default="_mask.png")
    g.add_argument("--fg_stem_regex", default="", help="regex on the image stem; its first group (or match) is the mask stem, "
                   "e.g. '^\\d+_(.*)$' drops a numeric prefix")
    g.add_argument("--fg_min_dist", type=float, default=10.0, help="derived foreground: colour distance floor (L*a*b*)")
    g.add_argument("--fg_weak_frac", type=float, default=0.3, help="derived foreground: pixels above this x the Otsu "
                   "threshold count when they connect to the body (hysteresis; pale appendages)")
    g.add_argument("--pinhole_frac", type=float, default=0.0002, help="holes below this x image area are filled")
    g.add_argument("--window_min_px", type=int, default=50, help="covered background pieces counted as windows from this size")
    g.add_argument("--min_piece_frac", type=float, default=0.0003, help="pieces below this x foreground area are specks")
    g.add_argument("--fringe_px", type=float, default=4.0, help="uncovered foreground pieces at most this thick (half width, "
                   "px; setae) join the nearest part; thicker ones become 'body'")
    g = ap.add_argument_group("coarse classes (regex on the category name after --rename; case-insensitive)")
    g.add_argument("--input_catmap", default="", help="json {input name: coarse class} merging fine input labels into coarse "
                   "classes first (e.g. a hand-labelled file, to test the splits on their own)")
    g.add_argument("--rename", default="", help="comma list old=new applied to the input names, e.g. 'mandibles=mouthparts'")
    for r, rx in ROLE_DEFAULTS.items():
        g.add_argument(f"--{r}_regex", default=rx, help=f"coarse class(es) of the {r} (default {rx!r})")
    g = ap.add_argument_group("view and orientation (as descriptron_serial_homology_v1)")
    g.add_argument("--view", choices=["dorsal", "ventral", "lateral"])
    g.add_argument("--view_regex", "--view-regex", help="regex on the file name; first group: dorsal/ventral/lateral")
    g.add_argument("--view_field", "--view-field", default="view")
    g.add_argument("--anterior", choices=list(sh.ANTERIOR_VEC), help="image direction of the head when there is no head mask")
    g.add_argument("--work_size", type=int, default=1200)
    g = ap.add_argument_group("DINOv3 boundary maps (step 2)")
    g.add_argument("--maps_dir", "--maps-dir", default="", help="keep / reuse <stem>_segsplit_maps.npz here")
    g.add_argument("--recompute_maps", action="store_true", help="recompute maps even when an .npz exists")
    g.add_argument("--maps_only", action="store_true", help="only compute the maps (GPU step), no splitting")
    g.add_argument("--tokens", choices=["on", "off"], default="on", help="off = image gradient only (no torch)")
    g.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    g.add_argument("--tile", type=int, default=1024); g.add_argument("--device", default="cuda")
    g.add_argument("--token_fg", type=float, default=0.25, help="tokens with less foreground than this do not count in gradients")
    g.add_argument("--pca", type=int, default=16, help="L18 PCA components kept for the eye step")
    g.add_argument("--hp_pca", type=int, default=64, help="whitened PCA components of the high-pass tokens for the maps")
    g.add_argument("--token_scale", type=float, default=2.0, help="image scale fed to DINOv3: 2 = 8 native px per patch "
                   "(4-px grid with the shifted passes); 1 = native (8-px grid)")
    g = ap.add_argument_group("sharpening and splitting (steps 3, 5)")
    g.add_argument("--sharpen_layer", choices=["L18", "L23", "none"], default="L18")
    g.add_argument("--split_layer", choices=["L18", "L23"], default="L18")
    g.add_argument("--image_weight", type=float, default=0.5, help="sharpening: weight of the photograph's gradient added to "
                   "the token edge map")
    g.add_argument("--split_image_weight", type=float, default=0.3, help="splits: weight of the photograph's gradient added "
                   "to the token line map")
    g.add_argument("--band_px", type=float, default=12.0, help="watershed band: how far a coarse boundary may move (px)")
    g.add_argument("--suture_window", type=float, default=0.15, help="suture search: +- this x half elytra width from the midline")
    g.add_argument("--suture_contrast", type=float, default=1.25, help="suture ridge must be this x the window median")
    g.add_argument("--first_segment", type=int, default=3, help="number of the first visible segment (Staphylininae dorsal: 3)")
    g.add_argument("--last_regular_segment", type=int, default=8, help="number of the last regular segment")
    g.add_argument("--segment_name", default="tergite {n}", help="name format of the segments")
    g.add_argument("--segment_name_ventral", default="sternite {n}", help="name format in ventral views ('' = --segment_name)")
    g.add_argument("--apex", choices=["auto", "none"], default="auto", help="auto = look for terminalia behind the last regular segment")
    g.add_argument("--apex_name", default="", help="name of the apex (default: the segment after --last_regular_segment)")
    g.add_argument("--terminalia_name", default="terminalia")
    g.add_argument("--apex_width_frac", type=float, default=0.6, help="apex: mean width below this x the regular width")
    g.add_argument("--apex_step", type=float, default=1.2, help="apex: width ratio across its base (abrupt narrowing)")
    g.add_argument("--apex_bonus", type=float, default=0.5, help="score added for an apex that fits")
    g.add_argument("--apex_min_frac", type=float, default=0.3, help="apex: at least this x the expected segment length long")
    g.add_argument("--setae_px", type=float, default=0.0, help="abdomen shape (widths, lobes, tip) measured without setae: "
                   "mask opened by a disc of this radius (px)")
    g.add_argument("--apex_lobes", type=float, default=0.15, help="apex: share of its rows split into separate lobes that counts")
    g.add_argument("--apex_narrow", type=float, default=0.45, help="apex: mean width below this x the regular width counts alone")
    g.add_argument("--apex_prev_frac", type=float, default=0.7, help="apex: mean width below this x the segment before it")
    g.add_argument("--apex_min_line", type=float, default=0.5, help="apex: its boundary's line z below this -> terminalia")
    g.add_argument("--cut_threshold", type=float, default=1.5, help="candidate cut score needed to gain from a cut")
    g.add_argument("--margin_band", type=float, default=0.15, help="abdomen: lateral margin bands, x the width on each side")
    g.add_argument("--mid_weight", type=float, default=0.0, help="abdomen: weight of the line strength across the middle")
    g.add_argument("--l23_margin_weight", type=float, default=0.0, help="abdomen: weight of the L23 line strength at the margins")
    g.add_argument("--notch_weight", type=float, default=0.0, help="abdomen: weight of the outline notch")
    g.add_argument("--notch_scale", type=float, default=0.03, help="outline notch depth (x median width) that counts 1")
    g.add_argument("--ratio_weight", type=float, default=2.0, help="penalty x log(neighbouring segment length ratio)^2")
    g.add_argument("--missing_weight", type=float, default=1.0, help="penalty per regular segment fewer than expected")
    g.add_argument("--min_segment_frac", type=float, default=0.35, help="shortest segment, x expected length")
    g.add_argument("--cut_bend", type=float, default=0.2, help="a cut may bend +- this x expected segment length")
    g.add_argument("--cut_smooth", type=float, default=1.0, help="cost of a 1-px step of a cut, x the median line strength")
    g.add_argument("--refine_px", type=float, default=3, help="final placement of each cut on the photograph's gradient "
                   "within +- this many px (0 = token map only)")
    g.add_argument("--leg_parts_dorsal", default="femur:0.36,tibia:0.36,tarsus:0.28")
    g.add_argument("--leg_parts_ventral", default="coxa:0.10,trochanter:0.07,femur:0.30,tibia:0.30,tarsus:0.23")
    g.add_argument("--leg_width_weight", type=float, default=2.0); g.add_argument("--leg_bend_weight", type=float, default=0.5)
    g.add_argument("--leg_ratio_weight", type=float, default=1.5); g.add_argument("--fallback_score", type=float, default=-1.0)
    g.add_argument("--leg_band_px", type=float, default=4.0, help="watershed band at the leg joints (px)")
    g.add_argument("--leg_map", choices=["line", "edge"], default="line", help="leg joints from the line or the edge tensor")
    g.add_argument("--leg_name_format", default="prefix", help="serial homology --name_format (prefix = left_profemur)")
    g.add_argument("--mandible_from_antenna", choices=["auto", "on", "off"], default="auto",
                   help="separate mandibles from the coarse antenna class; auto = only when the input has no mandible class")
    g.add_argument("--mandible_width", type=float, default=1.3, help="mandible: wider than this x the antenna's radius")
    g.add_argument("--mandible_min_area", type=float, default=6.0, help="mandible: at least this x pi x antenna radius^2")
    g.add_argument("--mandible_zone", type=float, default=0.25, help="mandible: centre anterior of head front - this x head length")
    g.add_argument("--palp_elongation", type=float, default=1.8)
    g.add_argument("--eye_lateral_frac", type=float, default=0.5, help="eyes: lateral zone, x the local half width of the head")
    g.add_argument("--eye_reach", type=float, default=1.3, help="eyes: search out to this x the head's half width")
    g.add_argument("--eye_open_frac", type=float, default=0.12, help="eyes: opening radius of the head outline, x the "
                   "head's half width (removes antennae)")
    g.add_argument("--eye_min_area", type=float, default=0.008, help="eye: at least this x the head area")
    g.add_argument("--eye_max_area", type=float, default=0.15, help="eye: at most this x the head area")
    g.add_argument("--eye_band_px", type=float, default=6.0, help="eyes: watershed band of the outline (px)")
    g.add_argument("--eye_distinct_weight", type=float, default=2.0, help="eyes: weight of the cluster's L18 distance from "
                   "the medial head")
    g.add_argument("--eye_prior_weight", type=float, default=2.0, help="eyes: weight of the overlap with the coarse eye "
                   "prediction (within --eye_prior_px)")
    g.add_argument("--eye_prior_px", type=float, default=8.0)
    g.add_argument("--eye_position", choices=["anterior", "posterior", "any"], default="anterior",
                   help="eyes: prior on their AP position in the head (dorsal Staphylinidae: anterolateral; temples behind)")
    g.add_argument("--eye_position_weight", type=float, default=4.0)
    g.add_argument("--eye_min_compact", type=float, default=0.4, help="eye: area / bounding box (s, t) at least this")
    g.add_argument("--eye_min_score", type=float, default=8.0, help="eyes: pair score below this -> confidence low")
    g.add_argument("--eye_max_elongation", type=float, default=4.5, help="eye: length / width at most this")
    g = ap.add_argument_group("names")
    g.add_argument("--side_format", default="{side}_{part}", help="sided names (DINOMask style: left_elytron)")
    g.add_argument("--elytra_name", default="", help="name for unsplit elytra (lateral views)")
    g.add_argument("--body_name", default="body"); g.add_argument("--scutellum_name", default="scutellum")
    g.add_argument("--mesonotum_name", default="mesonotum"); g.add_argument("--unassigned_name", default="unassigned")
    g.add_argument("--mouthparts_name", default="mouthparts")
    g.add_argument("--simplify_px", type=float, default=1.0, help="polygon simplification tolerance (px)")
    g = ap.add_argument_group("evaluation")
    g.add_argument("--gt", default="", help="hand-labelled COCO of the same images: IoU and boundary F -> <out>_eval.csv")
    g.add_argument("--gt_catmap", default="", help="json {hand-label name: coarse class} to merge the hand labels")
    return ap


def main(argv=None):
    opts = build_parser().parse_args(argv)
    coco = load(opts.input)
    out, rows, cut_rows, eval_rows, notes = run(coco, opts)
    if opts.maps_only:
        return 0
    save(out, opts.out)
    stem = str(Path(opts.out).with_suffix(""))
    write_csv(rows, opts.csv or stem + "_parts.csv")
    write_csv(cut_rows, stem + "_cuts.csv")
    write_csv(getattr(opts, "_img_rows", []), stem + "_images.csv")
    if eval_rows:
        write_csv(eval_rows, stem + "_eval.csv")
    for n in notes:
        print("note:", n)
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["image"]].append(r)
    for im, rs in by_img.items():
        segs = [r["part"] for r in rs if r["segment"] not in ("", None)]
        print(f"{im}: {len(rs)} parts; segments: {', '.join(segs) or '-'}; apex_present "
              f"{next((r['apex_present'] for r in rs if r['apex_present']), '-')}; "
              f"low confidence: {sum(r['confidence'] == 'low' for r in rs)}")
    if eval_rows:
        for stage in ("coarse", "sharpened", "fine"):
            rs = [r for r in eval_rows if r["stage"] == stage and np.isfinite(r["iou"])]
            if rs:
                print(f"eval {stage}: mean IoU {np.nanmean([r['iou'] for r in rs]):.3f}, BF2 {np.nanmean([r['bf2'] for r in rs]):.3f}, "
                      f"BF4 {np.nanmean([r['bf4'] for r in rs]):.3f}, internal BF2 {np.nanmean([r['bf2_internal'] for r in rs]):.3f} "
                      f"({len(rs)} parts)")
    print(f"wrote {opts.out}, {opts.csv or stem + '_parts.csv'}, {stem}_cuts.csv, {stem}_images.csv"
          + (f", {stem}_eval.csv" if eval_rows else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
