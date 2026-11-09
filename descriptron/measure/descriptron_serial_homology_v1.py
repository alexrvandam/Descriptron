#!/usr/bin/env python3
"""
descriptron_serial_homology_v1.py — name generic leg parts by serial homologue and side
=========================================================================================

Leg segmentation (DINOMask, SAM2-PAL, Detectron2) learns better from many examples of a few GENERIC classes
(coxa, trochanter, femur, tibia, tarsus) than from a handful each of 18-30 side- and segment-specific classes.
This program gives every predicted generic instance its serial homologue and side back, by rule, from the
geometry of the specimen in the image, as the names Descriptron already uses:

    femur -> "profemur (left)"    tibia -> "mesotibia (right)"    tarsus -> "metatarsus (left)"    coxa -> "mesocoxa (right)"
    trochanter -> "metatrochanter (left)"      (--name_format prefix writes DINOMask's style: "left_profemur")

With --mode verify it checks a model's own full names instead (model first, rule second; see below).
It writes a NEW COCO file (only the category of each named instance changes; the categories are added when they
do not exist yet; every other annotation, image and category is kept) and a CSV with the evidence for every
instance (model_name, rule_name, final_name, status, reason, side margin, anchor, rank evidence, links,
confidence of the rule's name, warnings). Instances that cannot be named keep their category, are listed, and
say why: no guessing.

The rule, per image
  1. Body frame. Body region = union of the masks whose category matches --body_regex ("body (SAM2)", "habitus";
     else every mask in the image) and the head, minus the leg parts, opened with a disc of --open_frac x its
     largest inscribed radius, so that the thin appendages left in a whole-specimen mask (antennae, trochanters,
     unlabelled bits of legs) fall away. Anteroposterior (AP) axis = principal axis (PCA) of that core's pixels;
     for a nearly round body (length/width < --min_elongation) the axis is taken from the head (or --anterior)
     instead. Anterior end = the end nearer the head (--head_regex); without a head mask, --anterior
     up|down|left|right (when both exist and disagree, the head wins and a warning is written).
     MIDLINE (--midline edges, default): for every slice across the axis, the midpoint of the core's two edges;
     the midline at a given AP level is the median of those midpoints within +-10% of body length. Why not the
     axis through the centroid (--midline axis): the centroid moves towards whichever side carries more mass (an
     extended leg or antenna in a SAM whole-body mask, a lifted elytron), the edge midpoints do not; and the local
     median follows a curved body (a tilted ant gaster) where one straight line would cut across the thorax.
  2. Legs as chains: coxa -> trochanter -> femur -> tibia -> tarsus. Each instance gets its own principal axis and
     two end caps (mean of its pixels in the extreme 10% of its length). Tibia -> femur, tarsus -> tibia,
     femur -> trochanter, trochanter -> coxa (and femur -> coxa where there is no trochanter) are linked by a global
     (Hungarian) assignment on end-cap distances, so two tibiae can never claim one femur; a link longer than
     --max_gap x body length is refused, and a tibia may not take the femur end that clearly lies at the body.
     The linked ends define proximal and distal (a femur's distal end is the one its tibia articulates with);
     unlinked parts take as proximal the end nearer the body core (at equal distance, the one nearer the
     midline). Absent coxae and trochanters (the rule in dorsal views, where they are hidden) are simply skipped.
     A second pass links across one missing level (tibia -> trochanter or coxa when the femur is hidden, tarsus ->
     femur when the tibia is) over up to 1.5 x the missing segment's typical length, but only when clear (any
     other assignment at least half that allowance worse), so that a hidden segment does not split one leg in two.
     Every link has a MARGIN: how much worse the best assignment is in which that child takes another parent. A
     margin below --link_margin x body length (touching legs) makes the link ambiguous: the parts beyond it are
     left generic. A chain that starts at a tarsus, or at a tibia more than --max_gap from the body, is a
     FRAGMENT (the parts joining it to the body are missing): never ranked, never named. Why: in a randomized
     test (asymmetric postures, knees folded up to 150 deg, any rotation, 10% of parts deleted) fragments
     ranked as legs gave most of the wrong names; these rules took wrong names from 0.5% to 0.1% of instances.
  3. Side. A chain's ANCHOR is its most proximal part: the coxa (centroid) if present, else the trochanter
     (centroid), else the proximal end of the femur or tibia. Dorsal views get no warning for missing coxae or
     trochanters; ventral and lateral views note "no coxa or trochanter found" (it does not lower confidence).
     The anchor's signed distance from the midline gives the side; every part of the chain inherits it (a tarsus
     folded across the midline keeps its leg's side, flagged in the CSV).
     Convention: with the anterior end up, the specimen's left is on the image's left in DORSAL view and on the
     image's right in VENTRAL view. The view is per image: an image field (--view_field, default "view"), else a
     regular expression on the file name (--view_regex; its first group or the whole match: "d", "dors..." =
     dorsal, "v", "ven..." = ventral, "l", "lat...", "prof..." = lateral), else --view for all images; with no
     view, no side and so no name. Checked on hand labels of Oxytelus jessoensis (dorsal, head up):
     left_elytron and every "... (left)" leg part lie on the image's left, i.e. the anatomical convention above.
  3L. LATERAL views (profile) are not bilaterally symmetrical, so there is no midline step: every leg belongs to
     the side facing the camera. With the dorsum up, head on the image's LEFT = the specimen's LEFT side, head on
     the RIGHT = its RIGHT side; dorsum down inverts (in general: left when anterior x dorsal > 0 in image
     coordinates, so any rotation works). Dorsum = --dorsum up|down|left|right (image direction, default up), or
     auto: the side of the body axis on which the dorsal-structure masks lie (--dorsal_regex: pronotum, elytra,
     nota, tergites), else the side opposite the legs (legs hang from the ventral side); a disagreement is
     written. Far-side legs often show behind the body; their side is NOT guessed from geometry. Default
     (--lateral_far_side ignore): every leg is near side, so with more than three legs the segments cannot be
     ranked (only matched to landmark positions; extras stay generic). With --lateral_far_side flag and more than
     three legs: a far-side leg shows only where it emerges from behind the body, so its visible proximal end lies
     ON the body outline, while a near-side leg starts inside the outline (in front of the body). The three legs
     whose anchors lie deepest inside the (closed, hole-filled, leg-free) body are the near side; the others are
     marked "far side?" with the near leg at the same AP level named in the warning (e.g. "possibly the far-side
     mesofemur"), and are not named. If the 3rd and 4th depths are too close to call, no leg is named.
  4. Segment. Per side (lateral: the near side), chains are ordered by the AP position of their anchors, from the
     head backwards: three chains -> pro, meso, meta. Fewer (a leg hidden, lost or folded away) or more
     (fragmented masks) are NOT forced into ranks: they are matched in AP order to expected positions from (a)
     the other side of the same specimen when it has exactly three chains (not in lateral views), else (b) body
     landmarks: prothoracic legs at 65% of the pronotum/prothorax length (--pro_regex), mesothoracic 15% behind
     the front of the elytra/forewings/mesothorax (--meso_regex; without it, halfway), metathoracic at the front
     of the abdomen (--meta_regex). A chain is named only when it is at least 1/--ambiguity (default 2) times
     closer to its own expected position than to any other; otherwise it stays generic with a warning. With
     three chains per side the other side and the landmarks are cross-checks (warning, confidence low).
  5. Names: --name_format suffix ("profemur (left)", default), prefix ("left_profemur") or a format with
     {segment} {part} {side}. If two instances would get the same name in one image, both stay generic.
     Confidence of the rule's name: high = ranked from three coxa/trochanter/femur anchors with no warning;
     medium = expected positions, a tibia anchor or one warning; low = several warnings or a failed cross-check.

Model first, rule second (--mode verify). A patch-token model (DINOMask, trained on full names) names the parts;
the rule runs blind next to it and steps in only where the model is unsure or contradicted. A part with a model
name keeps it, written in --name_format, unless
  (a) another part in the same image carries the same model name (duplicate);
  (b) the rule puts it on the other side with a clear margin (anchor >= --verify_margin, default 0.2, x half body
      width from the midline, no long or skipped link); never in lateral views, where far-side legs are real;
  (c) the rule ranks it pro/meso/meta differently with clear evidence (three coxa/trochanter/femur anchors without
      warnings, or an expected-position match twice as decisive as --ambiguity asks);
  (d) a part it is linked to (unambiguous links only) carries a different segment or side: the chain is
      inconsistent.
Then the rule's name replaces it ((b) the side, (c) the segment, (a) and (d) both). Where the rule has no name, a
duplicate (a) becomes generic, while an inconsistent chain (d) keeps the model name, flagged (the links may be what
is wrong). Side-less model names (DINOMask --fold_lr: "profemur") take the rule's side, which is known even when
the leg cannot be ranked (in a lateral view: the near side). Parts with a
'score' below --min_score are named by the rule alone. When a name would still be given twice, the model's name
wins and the rule's stays generic. In the default rule mode, parts already carrying a full name are left as they
are (status 'input') but take part in the geometry. In a randomized test with simulated model errors (each part's
side and segment independently wrong with probability 0.05, all parts present) verify mode fixed 315 of 316 wrong
model names and changed no correct one; at 0.10 with 10% of parts deleted it fixed 489 of 512 (6 stayed wrong,
flagged) and changed 4 of 2356 correct names (19 more were left generic).

Other leg counts. The names assume three thoracic leg pairs (insects). Up to three chains per side are named as
above; a side with more than three chains (fragments, or an arachnid's four pairs) is named only through expected
positions and the extras stay generic. Taxa with reduced forelegs (nymphalid butterflies) show two legs per side:
those are named only with landmarks (or a complete opposite side).

Known failure modes. (i) Dorsal views hide coxae and often femora: a chain then anchors on a tibia, whose
proximal end lies wherever the femur points (profemur held forwards, metafemur backwards), so the AP order is
usually but not always right (warning "leg anchored on a tibia"); a tibia away from the body without its femur
is a fragment and stays generic. (ii) Anchors on or across the midline (legs folded under the body in ventral
view) give an unreliable side: no side below --min_side_margin x half body width. (iii) Strongly bent or curled
parts: the end caps along the instance's own axis may not be its true ends. (iv) Touching legs of one side,
above all with a segment missing, can swap links; the link margin catches near-ties, not every case. (v)
Lateral views: the far-side test assumes the body mask outlines the body itself (the masks of all leg parts are
removed from it, and its outline is where far-side legs disappear); it cannot tell a near-side leg whose base is
hidden from a far-side leg, and it has been tested only on synthetic profiles. Frontal and posterior views are
not supported.

Inputs: COCO/Descriptron files with polygon or RLE masks (DINOMask's compressed RLE strings are decoded here,
no pycocotools needed; a mask in several pieces is flagged: two parts under one label?). Generic categories are
matched by name, case-insensitively, with an optional number ("femur", "Femur_1", "tibia 2");
--coxa/--trochanter/--femur/--tibia/--tarsus take comma-separated alternatives. Full names are read in both
styles, "<seg><part> (<side>)" and "<side>_<seg><part>", and side-less ("profemur"). --check_existing makes the
rule name every part and reports agreement with the existing names (a test of the rule on hand labels).
Duplicate image entries (same id listed twice) are processed once.

Examples
    python descriptron_serial_homology_v1.py predicted.json --out named.json --view dorsal
    python descriptron_serial_homology_v1.py batch.json --out named.json --view_regex "_(dorsal|ventral|lateral)_"
    python descriptron_serial_homology_v1.py rotated.json --out named.json --view dorsal --anterior right
    python descriptron_serial_homology_v1.py profiles.json --out named.json --view lateral --dorsum auto --lateral_far_side flag
    python descriptron_serial_homology_v1.py hand_named.json --out check.json --view dorsal --check_existing
    python descriptron_serial_homology_v1.py dinomask_out/predictions.json --out checked.json --view dorsal \
        --mode verify --name_format prefix
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

VERSION = "serial_homology_v1"
PARTS = ("coxa", "trochanter", "femur", "tibia", "tarsus")   # proximal -> distal
SEGMENTS = ("pro", "meso", "meta")                       # anterior -> posterior
_P = "|".join(PARTS)
# full names in the two styles in use: "profemur (left)" (Descriptron editor) and "left_profemur" (DINOMask, whose
# hflip augmentation swaps only the left_/right_ prefix); DINOMask --fold_lr writes side-less names ("profemur")
SUFFIX_RE = re.compile(r"^\s*(pro|meso|meta)[\s_\-]?(%s)\s*\(\s*(left|right)\s*\)\s*$" % _P, re.I)
PREFIX_RE = re.compile(r"^\s*(left|right)[\s_\-]+(pro|meso|meta)[\s_\-]?(%s)\s*$" % _P, re.I)
SEGONLY_RE = re.compile(r"^\s*(pro|meso|meta)[\s_\-]?(%s)\s*$" % _P, re.I)
FULL_NAME_RE = SUFFIX_RE
NAME_PRESETS = {"suffix": "{segment}{part} ({side})", "prefix": "{side}_{segment}{part}"}
CAP = 0.10                 # end cap = pixels in the extreme 10% of a part's length
RANK_GAP_WARN = 0.02       # AP gap between neighbouring anchors below this x body length -> warning
LINK_WARN = 0.03           # link longer than this x body length -> warning
BIG = 1e12
ANTERIOR_VEC = {"up": (0.0, -1.0), "down": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}


def load(path):
    with open(path) as f:
        return json.load(f)


def save(d, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=1)


# ------------------------------------------------------------------ names
def part_matcher(spec: Dict[str, str]) -> Callable[[str], Optional[str]]:
    """{part: "femur,femora"} -> f(category name) -> part or None (case-insensitive, optional trailing number)."""
    pats = {}
    for part, alts in spec.items():
        names = [re.escape(n.strip()) for n in alts.split(",") if n.strip()]
        if names:
            pats[part] = re.compile(r"^\s*(?:%s)(?:[\s_\-]*\d+)?\s*$" % "|".join(names), re.I)

    def match(name: str) -> Optional[str]:
        for part, rx in pats.items():
            if rx.match(name or ""):
                return part
        return None
    return match


def full_name(fmt: str, segment: str, part: str, side: str) -> str:
    return NAME_PRESETS.get(fmt, fmt).format(segment=segment, part=part, side=side)


def parse_name(name: str) -> Optional[Tuple[Optional[str], Optional[str], str]]:
    """'profemur (left)' / 'left_profemur' -> ('left', 'pro', 'femur'); 'profemur' -> (None, 'pro', 'femur')."""
    if (m := SUFFIX_RE.match(name or "")):
        return m.group(3).lower(), m.group(1).lower(), m.group(2).lower()
    if (m := PREFIX_RE.match(name or "")):
        return m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
    if (m := SEGONLY_RE.match(name or "")):
        return None, m.group(1).lower(), m.group(2).lower()
    return None


# ------------------------------------------------------------------ masks
def ann_polygons(ann) -> List[np.ndarray]:
    """Polygons of an annotation as (n, 2) arrays (flat un-nested polygons tolerated; RLE -> [])."""
    seg = ann.get("segmentation")
    if isinstance(seg, list) and seg and not isinstance(seg[0], list):
        seg = [seg]
    if not isinstance(seg, list):
        return []
    return [np.asarray(p, float).reshape(-1, 2) for p in seg if isinstance(p, list) and len(p) >= 6]


def rle_counts(s) -> List[int]:
    """Compressed COCO RLE string (as DINOMask writes) -> run lengths; the code of pycocotools' rleFrString."""
    s = s.decode() if isinstance(s, bytes) else s
    counts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = bool(c & 0x20)
            p += 1; k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def rle_mask(seg: dict) -> Optional[np.ndarray]:
    """COCO RLE (compressed string or uncompressed counts) -> bool mask; no pycocotools needed."""
    counts = seg.get("counts")
    if isinstance(counts, (str, bytes)):
        try:
            counts = rle_counts(counts)
        except Exception:
            return None
    h, w = seg["size"]
    flat, pos = np.zeros(h * w, bool), 0
    for i, c in enumerate(counts):
        if i % 2:
            flat[pos:pos + c] = True
        pos += c
    return flat.reshape(w, h).T                              # COCO RLE is column-major


def rle_string(counts: List[int]) -> str:
    """Run lengths -> compressed COCO RLE string (pycocotools' rleToString), the inverse of rle_counts."""
    out = []
    for i, x in enumerate(counts):
        x = int(x) - (int(counts[i - 2]) if i > 2 else 0)
        more = True
        while more:
            c = x & 0x1f
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def mask_rle(m: np.ndarray) -> dict:
    """Bool mask (h, w) -> compressed COCO RLE {'size', 'counts'} (column-major runs, starting with background)."""
    flat = np.asarray(m, bool).ravel(order="F")
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(flat.astype(np.int8))) + 1, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        runs = [0] + runs
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": rle_string(runs)}


def raster(polys: List[np.ndarray], w: int, h: int, scale: float = 1.0, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """Union of polygons drawn into a w x h window (top-left at image point x0, y0) at `scale` px per image px."""
    img = Image.new("L", (max(1, int(w)), max(1, int(h))), 0)
    dr = ImageDraw.Draw(img)
    for p in polys:
        q = (p - [x0, y0]) * scale
        dr.polygon([(float(x), float(y)) for x, y in q], fill=1, outline=1)
    return np.asarray(img, bool)


def ann_mask(ann, W: int, H: int, scale: float) -> Optional[np.ndarray]:
    """Annotation mask at working scale (polygons drawn directly; RLE decoded and resampled)."""
    polys = ann_polygons(ann)
    w, h = int(math.ceil(W * scale)), int(math.ceil(H * scale))
    if polys:
        return raster(polys, w, h, scale)
    seg = ann.get("segmentation")
    if isinstance(seg, dict) and "counts" in seg:
        m = rle_mask(seg)
        if m is None:
            return None
        if scale != 1.0:                                     # nearest-neighbour sampling (fast for big masks)
            ri = np.minimum(((np.arange(h) + 0.5) / scale).astype(int), m.shape[0] - 1)
            ci = np.minimum(((np.arange(w) + 0.5) / scale).astype(int), m.shape[1] - 1)
            m = m[np.ix_(ri, ci)]
        return m
    return None


def n_pieces(m: np.ndarray) -> int:
    """Connected pieces of a mask that hold at least 10% of the largest (a class mask covering two parts?)."""
    lab, n = ndimage.label(m)
    if n <= 1:
        return n
    sizes = np.bincount(lab.ravel())[1:]
    return int((sizes >= max(5, 0.1 * sizes.max())).sum())


def part_pixels(ann, W: int, H: int, max_px: int = 40000) -> Tuple[Optional[np.ndarray], int]:
    """-> ((N, 2) image coordinates (x, y) of an instance's pixels at full resolution, rasterised in its own box;
    number of pieces of the mask)."""
    polys = ann_polygons(ann)
    if polys:
        allp = np.vstack(polys)
        x0, y0 = np.floor(allp.min(0)) - 2
        x1, y1 = np.ceil(allp.max(0)) + 2
        m = raster(polys, x1 - x0 + 1, y1 - y0 + 1, 1.0, x0, y0)
        ys, xs = np.nonzero(m)
        pts = np.column_stack([xs + x0, ys + y0]).astype(float)
    else:
        seg = ann.get("segmentation")
        m = rle_mask(seg) if isinstance(seg, dict) and "counts" in seg else None
        if m is None:
            return None, 0
        ys, xs = np.nonzero(m)
        pts = np.column_stack([xs, ys]).astype(float)
        if len(pts) >= 3:
            m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]       # its own box only
    if len(pts) < 3:
        return None, 0
    pieces = n_pieces(m)
    if len(pts) > max_px:                                     # deterministic thinning for very large masks
        pts = pts[:: int(math.ceil(len(pts) / max_px))]
    return pts, pieces


# ------------------------------------------------------------------ leg parts
class Part:
    """One generic leg-part instance: its pixels, own principal axis and two end caps."""

    def __init__(self, ann, part: str, pts: np.ndarray, existing: Optional[str] = None, pieces: int = 1):
        self.ann, self.id, self.part, self.existing, self.pieces = ann, ann.get("id"), part, existing, pieces
        self.c = pts.mean(0)
        ev, evec = np.linalg.eigh(np.cov((pts - self.c).T))
        self.axis = evec[:, 1]
        pr = (pts - self.c) @ self.axis
        lo, hi = float(pr.min()), float(pr.max())
        self.length = max(hi - lo, 1.0)
        self.width = 4 * math.sqrt(max(ev[0], 0.0))
        cap = max(CAP * self.length, 1.0)
        self.ends = np.array([pts[pr <= lo + cap].mean(0), pts[pr >= hi - cap].mean(0)])
        self.prox: Optional[int] = None                       # index of the proximal end once known
        self.prox_how = ""
        self.parent: Optional["Part"] = None
        self.child: Optional["Part"] = None
        self.link_d: Optional[float] = None
        self.link_margin: float = math.inf
        self.skipped: str = ""                                 # name of a missing segment bridged by the link

    def end(self, which: str) -> np.ndarray:
        return self.ends[self.prox] if which == "prox" else self.ends[1 - self.prox]


# ------------------------------------------------------------------ body frame
class Frame:
    """Body axis, anterior direction and midline of one specimen image (image px throughout)."""

    def __init__(self):
        self.c = np.zeros(2); self.a = np.array([0.0, -1.0]); self.L = np.array([-1.0, 0.0])
        self.length = 1.0; self.half_width = 1.0; self.scale = 1.0
        self.mid_s = np.zeros(0); self.mid_t = np.zeros(0); self.window = 0.0; self.mode = "edges"
        self.dist_out: Optional[np.ndarray] = None
        self.core0: Optional[np.ndarray] = None               # body minus leg masks, before the opening
        self.notes: List[str] = []

    def s(self, p) -> float:
        """AP coordinate (px, positive = anterior)."""
        return float((np.asarray(p, float) - self.c) @ self.a)

    def t(self, p) -> float:
        """Signed distance from the midline (px, positive = image-left when the anterior end is up)."""
        p = np.asarray(p, float)
        t = float((p - self.c) @ self.L)
        if self.mode == "edges" and len(self.mid_s):
            s = self.s(p)
            near = np.abs(self.mid_s - s) <= self.window
            if not near.any():                                # beyond the core: nearest slices
                near = np.abs(self.mid_s - s) <= np.abs(self.mid_s - s).min() + self.window
            t -= float(np.median(self.mid_t[near]))
        return t

    def dist_to_body(self, p) -> float:
        if self.dist_out is None:
            return 0.0
        h, w = self.dist_out.shape
        x = int(np.clip(round(p[0] * self.scale), 0, w - 1)); y = int(np.clip(round(p[1] * self.scale), 0, h - 1))
        return float(self.dist_out[y, x]) / self.scale


def edt(m: np.ndarray) -> np.ndarray:
    return ndimage.distance_transform_edt(m)


def body_frame(W: int, H: int, body: List[dict], head: List[dict], legs: List[dict], others: List[dict],
               opts) -> Tuple[Optional[Frame], List[str]]:
    """-> (Frame, warnings) or (None, warnings) when no axis or anterior end can be found."""
    warn: List[str] = []
    fr = Frame()
    sc = min(1.0, opts.work_size / max(W, H))
    fr.scale = sc
    w, h = int(math.ceil(W * sc)), int(math.ceil(H * sc))

    def union(anns):
        m = np.zeros((h, w), bool)
        for a in anns:
            x = ann_mask(a, W, H, sc)
            if x is not None and x.shape == m.shape:
                m |= x
        return m
    src = body if body else others
    region = union(src + head)
    fr.notes.append("body = " + ("+".join(sorted({opts._cat(a) for a in body})) if body else
                                 f"union of {len(others)} non-leg masks (no --body_regex match)"))
    legm = union(legs)
    if legm.any():
        legm = edt(~legm) <= max(2.0, 2 * sc)                # leg parts dilated by ~2 px
    core0 = region & ~legm
    if core0.sum() < 30:
        if region.sum() >= 30:
            core0 = region; warn.append("no body region outside the leg masks: frame from all masks incl. legs")
        else:
            return None, warn + ["no body mask (no --body_regex match and no other masks)"]
    # opening by a disc via two distance transforms (fast for large radii)
    R = float(edt(core0).max()); r = opts.open_frac * R
    core = core0
    if r >= 1.0:
        er = edt(core0) > r
        if er.any():
            core = (edt(~er) <= r) & core0
    lab, n = ndimage.label(core)
    if n > 1:
        sizes = ndimage.sum(core, lab, range(1, n + 1))
        core = np.isin(lab, 1 + np.nonzero(sizes >= 0.02 * sizes.max())[0])
    if core.sum() < 0.25 * core0.sum():
        core = core0; warn.append("opening removed most of the body; frame from the un-opened body mask")
    ys, xs = np.nonzero(core)
    P = np.column_stack([xs, ys]).astype(float) / sc
    c = P.mean(0)
    ev, evec = np.linalg.eigh(np.cov((P - c).T))
    u = evec[:, 1]
    elong = math.sqrt(ev[1] / max(ev[0], 1e-9))
    # anterior end
    hc = None
    if head:
        hm = union(head)
        if hm.any():
            hy, hx = np.nonzero(hm); hc = np.array([hx.mean(), hy.mean()]) / sc
    opt_vec = np.array(ANTERIOR_VEC[opts.anterior]) if opts.anterior else None
    if elong < opts.min_elongation:
        ref = (hc - c) if hc is not None and np.linalg.norm(hc - c) > 0 else opt_vec
        if ref is None:
            return None, warn + [f"body nearly round (length/width {elong:.2f}) and no head mask or --anterior for the axis"]
        u = ref / np.linalg.norm(ref)
        warn.append(f"body nearly round (length/width {elong:.2f}): AP axis taken from the "
                    + ("head" if hc is not None else "--anterior direction"))
    pr = (P - c) @ u
    length = float(pr.max() - pr.min())
    if hc is not None:
        d = float((hc - c) @ u)
        if abs(d) < 0.05 * length:
            warn.append("head centroid lies near the body centre along the axis: anterior end uncertain")
        a = u if d >= 0 else -u
        fr.notes.append("anterior from head")
        if opt_vec is not None and float(a @ opt_vec) < 0:
            warn.append(f"--anterior {opts.anterior} disagrees with the head mask; head used")
    elif opt_vec is not None:
        if abs(float(u @ opt_vec)) < 0.5:
            warn.append(f"body axis is more than 60 deg from --anterior {opts.anterior}")
        a = u if float(u @ opt_vec) >= 0 else -u
        fr.notes.append(f"anterior from --anterior {opts.anterior}")
    else:
        return None, warn + ["anterior end unknown: no head mask (--head_regex) and no --anterior"]
    fr.c, fr.a = c, a
    fr.L = np.array([a[1], -a[0]])                             # image-left when the anterior end is up
    fr.length = length
    # midline from the edges of the core, slice by slice
    S = (P - c) @ a; T = (P - c) @ fr.L
    nb = int(np.clip(length * sc / 3, 20, 120))
    edges = np.linspace(S.min(), S.max(), nb + 1)
    k = np.clip(np.digitize(S, edges) - 1, 0, nb - 1)
    tmin = np.full(nb, np.inf); tmax = np.full(nb, -np.inf)
    np.minimum.at(tmin, k, T); np.maximum.at(tmax, k, T)
    ok = np.isfinite(tmin) & (tmax > tmin)
    fr.mid_s = ((edges[:-1] + edges[1:]) / 2)[ok]
    fr.mid_t = (tmin[ok] + tmax[ok]) / 2
    fr.half_width = float(np.median(tmax[ok] - tmin[ok]) / 2) if ok.any() else 1.0
    fr.window = opts.midline_window * length
    fr.mode = opts.midline
    fr.notes.append(f"axis PCA (length/width {elong:.2f}), midline {opts.midline}")
    fr.dist_out = edt(~core)
    fr.core0 = core0
    return fr, warn


def landmark_extent(anns: List[dict], fr: Frame, W: int, H: int) -> Optional[Tuple[float, float]]:
    """(anterior, posterior) AP extent (2nd/98th percentile) of the union of landmark masks."""
    sc = fr.scale; w, h = int(math.ceil(W * sc)), int(math.ceil(H * sc))
    m = np.zeros((h, w), bool)
    for a in anns:
        x = ann_mask(a, W, H, sc)
        if x is not None and x.shape == m.shape:
            m |= x
    if m.sum() < 3:
        return None
    ys, xs = np.nonzero(m)
    S = (np.column_stack([xs, ys]) / sc - fr.c) @ fr.a
    return float(np.percentile(S, 98)), float(np.percentile(S, 2))


def landmark_positions(lm: Dict[str, List[dict]], fr: Frame, W: int, H: int) -> Tuple[Optional[List[float]], str]:
    """Expected AP positions [pro, meso, meta] of the leg attachments from body landmarks, or None."""
    ext = {k: landmark_extent(v, fr, W, H) for k, v in lm.items() if v}
    if "pro" not in ext or "meta" not in ext or ext["pro"] is None or ext["meta"] is None:
        return None, ""
    pa, pp = ext["pro"]; ma, _ = ext["meta"]
    e_pro, e_meta = pa - 0.65 * (pa - pp), ma
    if ext.get("meso"):
        sa, sp = ext["meso"]; e_meso = sa - 0.15 * (sa - sp); how = "pronotum/meso/abdomen landmarks"
    else:
        e_meso = (e_pro + e_meta) / 2; how = "pronotum/abdomen landmarks (meso halfway)"
    if not e_pro > e_meso > e_meta:
        return None, ""
    return [e_pro, e_meso, e_meta], how


# ------------------------------------------------------------------ linking legs into chains
def link(parents: List[Part], children: List[Part], cost: Callable, max_cost: float) -> Dict[int, tuple]:
    """Global (Hungarian) child -> parent assignment; a child may stay unlinked (pairs costlier than max_cost never link).
    MARGIN of a link = how much worse (px) the best assignment is in which that child takes ANOTHER parent (inf if no
    other parent is within max_cost): a small margin means another leg's part fits almost as well (touching legs).
    -> {child index: (parent index, cost, (parent end, child end), margin)}"""
    if not parents or not children:
        return {}
    n, m = len(children), len(parents)
    C = np.full((n, m + n), BIG); E = {}
    for i, ch in enumerate(children):
        C[i, m + i] = max_cost                                  # the 'stays unlinked' option of child i
        for j, pa in enumerate(parents):
            r = cost(pa, ch)
            if r is not None and r[0] <= max_cost:
                C[i, j], E[i, j] = r[0], r[1:]

    def solve(M):
        ri, cj = linear_sum_assignment(M)
        return float(M[ri, cj].sum()), dict(zip(ri.tolist(), cj.tolist()))
    total, asg = solve(C)
    out = {}
    for i, j in asg.items():
        if j < m:
            M = C.copy(); M[i, j] = BIG; M[i, m + i] = BIG       # child i must take another parent
            alt, _ = solve(M)
            out[i] = (j, float(C[i, j]), E[i, j], alt - total if alt < BIG / 2 else math.inf)
    return out


def end_cost(fr: Frame, body_end_penalty: bool = False):
    """Distance between the parent's distal end and the child's proximal end (any end while unknown)."""
    def cost(pa: Part, ch: Part):
        pe = [1 - pa.prox] if pa.prox is not None else [0, 1]
        ce = [ch.prox] if ch.prox is not None else [0, 1]
        best = None
        for i in pe:
            for j in ce:
                d = float(np.linalg.norm(pa.ends[i] - ch.ends[j]))
                if body_end_penalty and pa.prox is None:
                    # a tibia may not take the femur end that clearly lies at the body while the other end is out
                    if fr.dist_to_body(pa.ends[i]) + 0.25 * pa.length < fr.dist_to_body(pa.ends[1 - i]):
                        d += pa.length
                if best is None or d < best[0]:
                    best = (d, i, j)
        return best
    return cost


def proximal_by_body(p: Part, fr: Frame):
    d0, d1 = fr.dist_to_body(p.ends[0]), fr.dist_to_body(p.ends[1])
    if abs(d0 - d1) > 0.1 * p.length:
        p.prox, p.prox_how = (0 if d0 < d1 else 1), "end nearer the body"
    else:
        t0, t1 = abs(fr.t(p.ends[0])), abs(fr.t(p.ends[1]))
        p.prox, p.prox_how = (0 if t0 <= t1 else 1), "end nearer the midline"


def typical_length(by: Dict[str, List[Part]], part: str, fr: Frame) -> float:
    """Median length of a part class in this image (for a missing segment), else of the femora, else 0.2 x body."""
    for q in (part, "femur", "tibia"):
        if by.get(q):
            return float(np.median([p.length for p in by[q]]))
    return 0.2 * fr.length


def build_chains(parts: List[Part], fr: Frame, max_gap: float) -> List[List[Part]]:
    """Link the parts of each leg (see the header, step 2) -> chains ordered proximal -> distal."""
    by = defaultdict(list)
    for p in parts:
        by[p.part].append(p)

    def attach(links, parents, children):
        for i, (j, d, (pe, ce), margin) in links.items():
            pa, ch = parents[j], children[i]
            pa.child, ch.parent, ch.link_d, ch.link_margin = ch, pa, d, margin
            if pa.prox is None:
                pa.prox, pa.prox_how = 1 - pe, f"distal end articulates with {ch.part} {ch.id}"
            if ch.prox is None:
                ch.prox, ch.prox_how = ce, f"proximal end articulates with {pa.part} {pa.id}"
    # 1. tibia -> femur
    attach(link(by["femur"], by["tibia"], end_cost(fr, body_end_penalty=True), max_gap), by["femur"], by["tibia"])
    for f in by["femur"]:
        if f.prox is None:
            proximal_by_body(f, fr)
    # 2. tarsus -> tibia (for a tibia without femur, its distal end is the one its tarsus takes)
    attach(link(by["tibia"], by["tarsus"], end_cost(fr), max_gap), by["tibia"], by["tarsus"])
    for p in by["tibia"] + by["tarsus"] + by["coxa"]:
        if p.prox is None:
            proximal_by_body(p, fr)

    # 3. femur -> trochanter -> coxa, and femur -> coxa where there is no trochanter (usual in dorsal views, where
    #    coxae and trochanters are hidden: then nothing is linked here and nothing is missed). Cost = nearest of the
    #    parent's end caps (and, for a coxa, centroid) to the child's proximal end; the trochanter end that the femur
    #    takes is its distal end.
    def basal_cost(pa: Part, ch: Part):
        cands = [pa.ends[0], pa.ends[1]] + ([pa.c] if pa.part == "coxa" else [])
        ds = [float(np.linalg.norm(e - ch.end("prox"))) for e in cands]
        k = int(np.argmin(ds))
        return ds[k], (k if k < 2 else 0), ch.prox
    attach(link(by["trochanter"], by["femur"], basal_cost, max_gap), by["trochanter"], by["femur"])
    for q in by["trochanter"]:
        if q.prox is None:
            proximal_by_body(q, fr)
    attach(link(by["coxa"], by["trochanter"], basal_cost, max_gap), by["coxa"], by["trochanter"])
    free_cx = [q for q in by["coxa"] if q.child is None]
    free_fe = [q for q in by["femur"] if q.parent is None]
    attach(link(free_cx, free_fe, basal_cost, max_gap), free_cx, free_fe)

    # 4. across one missing level: tibia -> trochanter or coxa (femur hidden), tarsus -> femur (tibia hidden). The gap
    #    may be as long as the missing segment (allowance = max_gap + 1.5 x its typical length in this image), so the
    #    link must also be clear: any other assignment at least half the allowance worse (else it stays a fragment).
    def skip_cost(pa: Part, ch: Part):
        pe = pa.c if pa.part == "coxa" else pa.end("dist")
        return float(np.linalg.norm(pe - ch.end("prox"))), 0, ch.prox
    for pa_part, ch_part, missing in (("trochanter", "tibia", "femur"), ("coxa", "tibia", "femur"),
                                      ("femur", "tarsus", "tibia")):
        free_pa = [q for q in by[pa_part] if q.child is None]
        free_ch = [q for q in by[ch_part] if q.parent is None]
        allow = max_gap + 1.5 * typical_length(by, missing, fr)
        sl = {i: v for i, v in link(free_pa, free_ch, skip_cost, allow).items() if v[3] >= 0.5 * allow}
        attach(sl, free_pa, free_ch)
        for i in sl:
            free_ch[i].skipped = missing
    chains = []
    for p in parts:
        if p.parent is None:
            ch, q = [], p
            while q is not None:
                ch.append(q); q = q.child
            chains.append(ch)
    return chains


BASAL = ("coxa", "trochanter")                            # short basal parts: anchored on the centroid


def anchor_point(chain: List[Part]) -> np.ndarray:
    head = chain[0]
    return head.c if head.part in BASAL else head.end("prox")


# ------------------------------------------------------------------ segment ranks
def match_expected(svals: List[float], expected: List[float], ambiguity: float):
    """Order-preserving match of anchors (sorted anterior -> posterior) to expected positions [pro, meso, meta].
    -> list of (segment index or None, distance, next distance) per anchor."""
    n, K = len(svals), len(expected)
    best = None
    chosen_sets = itertools.combinations(range(n), min(n, K))
    for chosen in chosen_sets:
        for ks in itertools.combinations(range(K), len(chosen)):
            cost = sum(abs(svals[i] - expected[k]) for i, k in zip(chosen, ks))
            if best is None or cost < best[0]:
                best = (cost, dict(zip(chosen, ks)))
    out = []
    for i in range(n):
        k = best[1].get(i) if best else None
        if k is None:
            out.append((None, None, None)); continue
        d = abs(svals[i] - expected[k])
        dn = min(abs(svals[i] - expected[j]) for j in range(K) if j != k)
        out.append((k if d < ambiguity * dn else None, d, dn))
    return out


def rank_group(chs: List[List[Part]], other: Optional[List[List[Part]]], info: dict, fr: Frame,
               lm_pos: Optional[List[float]], lm_how: str, opts, seg_of: dict, where: str):
    """Segments for the leg chains of one group (one side, or the near side of a lateral view), sorted anterior
    first -> seg_of[id(chain)] = (segment index or None, method, evidence, warnings). `other` = the opposite
    side's chains (None in a lateral view: no contralateral evidence)."""
    other = other or []
    sv = [info[id(c)]["s"] for c in chs]
    fmt_s = ", ".join(f"{v:.0f}" for v in sv)
    if len(chs) == 3:
        for r, c in enumerate(chs):
            gaps = [abs(sv[r] - sv[q]) for q in range(3) if q != r]
            w = []
            if min(gaps) < RANK_GAP_WARN * fr.length:
                w.append(f"anchor only {min(gaps):.0f} px along the axis from a neighbour")
            seg_of[id(c)] = (r, "rank", f"{r + 1} of 3 {where} (anchor AP positions {fmt_s} px)", w, True)
        checks = []
        if len(other) == 3:
            ov = [info[id(c)]["s"] for c in other]
            if [int(np.argmin([abs(v - o) for o in ov])) for v in sv] != [0, 1, 2]:
                checks.append("ranks disagree with the other side's anchor positions")
        if lm_pos:
            mm = match_expected(sv, lm_pos, opts.ambiguity)
            if any(m[0] is not None and m[0] != r for r, m in enumerate(mm)):
                checks.append(f"ranks disagree with {lm_how}")
        for c in chs:
            seg_of[id(c)][3].extend(checks)
        return
    exp, how = None, ""
    if len(other) == 3:
        exp, how = [info[id(c)]["s"] for c in other], "other side's three anchors"
    elif lm_pos:
        exp, how = lm_pos, lm_how
    if exp is None:
        hint = (f"the other side has {len(other)}" if not where.startswith("in this lateral") else
                ("more than three: far-side legs? try --lateral_far_side flag" if len(chs) > 3 else "no other side"))
        for c in chs:
            seg_of[id(c)] = (None, "unresolved", f"{len(chs)} leg(s) {where} (AP {fmt_s} px)",
                             [f"{len(chs)} leg(s) {where} ({hint}) and no body landmarks "
                              f"(--pro_regex/--meta_regex): segment not forced"], False)
        return
    mm = match_expected(sv, exp, opts.ambiguity)
    exp_s = "/".join(f"{e:.0f}" for e in exp)
    for c, (kk, d, dn) in zip(chs, mm):
        if kk is None:
            why = ("extra leg instance " + where if d is None else
                   f"between expected positions (nearest {d:.0f} px, next {dn:.0f} px)")
            seg_of[id(c)] = (None, "unresolved", f"{how}: expected AP {exp_s} px", [why + ": segment not forced"], False)
        else:                                                # strong = twice as decisive as needed to name it
            seg_of[id(c)] = (kk, "contralateral" if how.startswith("other") else "landmarks",
                             f"{how}: expected AP {exp_s} px; {d:.0f} px from {SEGMENTS[kk]}, next {dn:.0f} px", [],
                             d < 0.5 * opts.ambiguity * dn)


def mask_centroid(anns: List[dict], fr: Frame, W: int, H: int) -> Optional[np.ndarray]:
    sc = fr.scale; w, h = int(math.ceil(W * sc)), int(math.ceil(H * sc))
    m = np.zeros((h, w), bool)
    for a in anns:
        x = ann_mask(a, W, H, sc)
        if x is not None and x.shape == m.shape:
            m |= x
    if m.sum() < 3:
        return None
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()]) / sc


def lateral_near_side(fr: Frame, dorsal: List[dict], parts: List[Part], W: int, H: int, opts):
    """Side facing the camera in a lateral view -> (left/right or None, how, warnings).
    With the anterior direction a and the dorsal direction d (image vectors, y down), the camera sees the
    specimen's LEFT side when a x d > 0: dorsum up + head on the image's left -> left; head right -> right;
    dorsum down inverts. d is perpendicular to the body axis, on the side of --dorsum (up/down/left/right in
    the image) or, with --dorsum auto, on the side of the dorsal-structure masks (--dorsal_regex: pronotum,
    elytra, ...), else opposite the legs (legs hang from the ventral side)."""
    perp, warn = fr.L, []
    hh = max(fr.half_width, 1e-9)
    d, how = None, ""
    if opts.dorsum in ANTERIOR_VEC:
        v = np.array(ANTERIOR_VEC[opts.dorsum]); proj = float(v @ perp)
        if abs(proj) < 0.5:
            return None, "", [f"the body axis is within 30 deg of --dorsum {opts.dorsum}: dorsum unknown, side not given"]
        d, how = perp * np.sign(proj), f"dorsum {opts.dorsum} (--dorsum)"
    else:                                                    # auto
        cues = []
        c = mask_centroid(dorsal, fr, W, H) if dorsal else None
        if c is not None and abs(fr.t(c)) > 0.1 * hh:
            cues.append((np.sign(fr.t(c)), "dorsal-structure mask"))
        legs_t = np.median([fr.t(p.c) for p in parts]) if parts else 0.0
        if abs(legs_t) > 0.1 * hh:
            cues.append((-np.sign(legs_t), "legs on the ventral side"))
        if not cues:
            return None, "", ["--dorsum auto: no dorsal-structure mask off the axis and legs on neither side: side not given"]
        if len(cues) == 2 and cues[0][0] != cues[1][0]:
            warn.append("--dorsum auto: dorsal-structure mask and legs disagree; mask used")
        d, how = perp * cues[0][0], f"dorsum from the {cues[0][1]}"
    cross = float(fr.a[0] * d[1] - fr.a[1] * d[0])
    return ("left" if cross > 0 else "right"), f"lateral: near side from head direction + {how}", warn


def flag_far_side(legs_: List[List[Part]], info: dict, fr: Frame, parts: List[Part], opts, seg_of: dict):
    """More than three legs in a lateral view: the far-side legs show only where they emerge from behind the body,
    so their visible proximal ends lie ON the body outline, while near-side legs start inside it (in front of the
    body). Depth = distance of the anchor inside the leg-free body (closed and hole-filled, so that the notch a
    near-side femur cuts into the body mask does not count). The three deepest anchors are the near side; the
    others are flagged 'far side?' and not named. If the 3rd and 4th depths are too close to call, none is named."""
    sc = fr.scale
    widths = [p.width for p in parts if p.part == "femur"] or [p.width for p in parts]
    r = max(2.0, float(np.median(widths)) * sc)
    m = fr.core0 if fr.core0 is not None else fr.dist_out == 0
    closed = edt(edt(~m) <= r) > r                           # closing by a disc (dilate, then erode)
    B = ndimage.binary_fill_holes(closed | m)
    dB = edt(B)
    h, w = dB.shape

    def depth(p):
        x = int(np.clip(round(p[0] * sc), 0, w - 1)); y = int(np.clip(round(p[1] * sc), 0, h - 1))
        return float(dB[y, x]) / sc
    dep = {id(ch): depth(info[id(ch)]["ap"]) for ch in legs_}
    for ch in legs_:
        info[id(ch)]["anchor"] += f" ({dep[id(ch)]:.1f} px inside the body outline)"
    order = sorted(legs_, key=lambda ch: -dep[id(ch)])
    tol = max(1.5, 0.15 * float(np.median(widths)))
    if dep[id(order[2])] - dep[id(order[3])] < tol:
        for ch in legs_:
            seg_of[id(ch)] = (None, "far side?", f"{len(legs_)} legs in a lateral view",
                              [f"far side? {len(legs_)} legs in a lateral view and near/far side unclear (anchor depths "
                               + ", ".join(f"{dep[id(c)]:.1f}" for c in order) + " px): not named"], False)
        return []
    near = sorted(order[:3], key=lambda ch: -info[id(ch)]["s"])
    for ch in order[3:]:
        twin = min(range(3), key=lambda k: abs(info[id(near[k])]["s"] - info[id(ch)]["s"]))
        seg_of[id(ch)] = (None, "far side?",
                          f"anchor {dep[id(ch)]:.1f} px inside the outline vs {dep[id(near[twin])]:.1f} px for the near "
                          f"leg at the same AP level ({SEGMENTS[twin]})",
                          [f"far side? extra leg behind the body, at the AP level of the near-side {SEGMENTS[twin]}"
                           f" leg (possibly the far-side {SEGMENTS[twin]}{ch[0].part}): not named"], False)
    return near


# ------------------------------------------------------------------ split semantic masks into pieces
def split_pieces(coco: dict, opts) -> List[str]:
    """--split_pieces: a semantic model (DINOMask) writes ONE mask per class and image, so a generic 'femur' mask
    holds every visible femur and a side-less 'profemur' both profemora. Each leg-part annotation is cut into its
    pieces (the polygons of a polygon annotation, as DINOMask's predictions_polygons.json writes them; the connected
    pieces of an RLE mask): pieces of at least 10% of the largest and --min_piece_px become annotations of their own
    (new ids, 'split_from' = the original id); the largest piece, with any smaller specks, stays on the original
    annotation. Works in place on `coco` (a copy). -> notes."""
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    match = part_matcher({"coxa": opts.coxa, "trochanter": opts.trochanter, "femur": opts.femur,
                          "tibia": opts.tibia, "tarsus": opts.tarsus})
    next_id = max([a["id"] for a in coco.get("annotations", []) if isinstance(a.get("id"), int)] + [0]) + 1
    new, n_split = [], 0
    for a in coco.get("annotations", []):
        name = cats.get(a.get("category_id"), "")
        if not (match(name) or parse_name(name)):
            continue
        seg = a.get("segmentation")
        pieces = []                                          # (area, segmentation, bbox)
        if isinstance(seg, list) and len(ann_polygons(a)) > 1:
            for P in ann_polygons(a):
                x, y = P[:, 0], P[:, 1]
                area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
                pieces.append((area, [P.ravel().round(2).tolist()],
                               [float(x.min()), float(y.min()), float(np.ptp(x)), float(np.ptp(y))]))
        elif isinstance(seg, dict) and "counts" in seg:
            m = rle_mask(seg)
            if m is None or not m.any():
                continue
            rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            y0, x0 = int(rows[0]), int(cols[0])
            sub = m[y0:rows[-1] + 1, x0:cols[-1] + 1]           # label inside the mask's box only
            lab, n = ndimage.label(sub, structure=np.ones((3, 3)))
            if n < 2:
                continue
            for k, sl in enumerate(ndimage.find_objects(lab), 1):
                pm = np.zeros_like(m)
                pm[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = lab == k
                pieces.append((float((lab[sl] == k).sum()), pm, [float(x0 + sl[1].start), float(y0 + sl[0].start),
                                                                  float(sl[1].stop - sl[1].start), float(sl[0].stop - sl[0].start)]))
        if len(pieces) < 2:
            continue
        pieces.sort(key=lambda q: -q[0])
        big = pieces[0][0]
        keep = [q for q in pieces[1:] if q[0] >= max(opts.min_piece_px, 0.1 * big)]
        if not keep:
            continue
        kept = {id(q) for q in keep}
        rest = [q for q in pieces if id(q) not in kept]
        if isinstance(seg, dict):
            m0 = np.zeros_like(rest[0][1])
            for q in rest:
                m0 |= q[1]
            a["segmentation"], a["area"] = mask_rle(m0), float(m0.sum())
            ys, xs = np.nonzero(m0)
            a["bbox"] = [float(xs.min()), float(ys.min()), float(np.ptp(xs) + 1), float(np.ptp(ys) + 1)]
        else:
            a["segmentation"] = [p for q in rest for p in q[1]]
            a["area"] = float(sum(q[0] for q in rest)); a["bbox"] = rest[0][2]
        for area, sg, bb in keep:
            b = dict(a, id=next_id, segmentation=mask_rle(sg) if isinstance(sg, np.ndarray) else sg, area=area,
                     bbox=bb, split_from=a.get("id"))
            new.append(b); next_id += 1
        n_split += 1
    coco["annotations"] = coco.get("annotations", []) + new
    return [f"--split_pieces: {n_split} leg-part masks cut into pieces, {len(new)} new annotations"] if n_split else []


# ------------------------------------------------------------------ views
def image_view(im: dict, opts) -> Tuple[Optional[str], str]:
    raw, src = None, ""
    attrs = im.get("attributes") if isinstance(im.get("attributes"), dict) else {}
    if opts.view_field and (im.get(opts.view_field) or attrs.get(opts.view_field)):
        raw, src = im.get(opts.view_field) or attrs.get(opts.view_field), f"image field '{opts.view_field}'"
    elif opts.view_regex and (m := re.search(opts.view_regex, im.get("file_name", ""), re.I)):
        raw, src = (m.group(1) if m.groups() and m.group(1) else m.group(0)), "file name (--view_regex)"
    elif opts.view:
        raw, src = opts.view, "--view"
    if raw is None:
        return None, "no view"
    v = str(raw).strip().lower()
    if v == "d" or v.startswith("dors"):
        return "dorsal", src
    if v == "v" or v.startswith("ven"):
        return "ventral", src
    if v == "l" or v.startswith("lat") or v.startswith("prof"):
        return "lateral", src
    return None, f"view '{raw}' from {src} is not dorsal, ventral or lateral"


# ------------------------------------------------------------------ one image
def assign_image(im: dict, anns: List[dict], opts) -> List[dict]:
    """Rule for one image -> one result row per generic leg-part instance."""
    W, H = int(im.get("width") or 0), int(im.get("height") or 0)
    if not W or not H:                                       # size from the polygons
        allp = [p for a in anns for p in ann_polygons(a)]
        if allp:
            mx = np.vstack(allp).max(0); W, H = int(mx[0]) + 2, int(mx[1]) + 2
    legs, body, head, others, lm, lm_dorsal = [], [], [], [], defaultdict(list), []
    for a in anns:
        part, existing = opts._leg(a)
        name = opts._cat(a)
        if part:
            legs.append((a, part, existing)); continue
        if a.get("is_line"):
            continue
        if opts._body_rx.search(name):
            body.append(a)
        elif opts._head_rx.search(name):
            head.append(a)
        else:
            others.append(a)
        for k, rx in opts._lm_rx.items():
            if rx.search(name):
                lm[k].append(a)
        if opts._dorsal_rx and opts._dorsal_rx.search(name):
            lm_dorsal.append(a)
    rows = []

    def row(a, part, existing, **kw):
        raw = opts._cat(a)
        parsed = parse_name(raw)
        score = a.get("score")
        r = {"image": im.get("file_name", ""), "image_id": im.get("id"), "annotation_id": a.get("id"),
             "split_from": a.get("split_from", ""), "generic_class": part, "input_category": raw, "model_name": "", "rule_name": "", "final_name": "",
             "status": "unresolved", "reason": "", "side": "", "segment": "", "view": "", "side_margin_px": "",
             "side_margin_rel": "", "anchor": "", "rank_method": "", "rank_evidence": "", "linked_to": "",
             "link_distance_px": "", "confidence": "", "warnings": "", "body_frame": "",
             "_rule": None, "_rule_side": None, "_side_clear": False, "_rank_clear": False, "_chain": None, "_raw": raw,
             "_model": (parsed[0], parsed[1]) if parsed else None,
             "_score": float(score) if isinstance(score, (int, float)) else None}
        if opts.check_existing:
            r["existing_name"] = existing or ""
        r.update(kw)
        return r
    if not legs:
        return rows
    parts, nogeo = [], []
    for a, part, existing in legs:
        pts, pieces = part_pixels(a, W, H) if W and H else (None, 0)
        if pts is None:
            nogeo.append(row(a, part, existing, warnings="no mask geometry (empty or unreadable segmentation)"))
        else:
            parts.append(Part(a, part, pts, existing, pieces))
    fr, fwarn = body_frame(W, H, body, head, [p.ann for p in parts], others, opts) if parts else (None, [])
    if fr is None:
        rows = nogeo + [row(p.ann, p.part, p.existing, warnings="; ".join(fwarn)) for p in parts]
        decide(rows, opts)
        return rows
    view, vsrc = image_view(im, opts)
    lateral = view == "lateral"
    if lateral:
        fr.mode = "axis"                                     # no midline in a profile
    max_gap = opts.max_gap * fr.length
    chains = build_chains(parts, fr, max_gap)
    # anchor of every chain; fragments (a tarsus, or a tibia away from the body, without the parts that join it to
    # the body) are not legs and are not ranked
    info, seg_of = {}, {}
    for ch in chains:
        ap = anchor_point(ch)
        t = fr.t(ap)
        info[id(ch)] = {"s": fr.s(ap), "t": t, "img_side": "L" if t > 0 else "R", "ap": ap,
                        "anchor": f"{ch[0].part} {ch[0].id} " + ("centroid" if ch[0].part in BASAL else
                                                                  f"proximal end ({ch[0].prox_how})")}
        gap = fr.dist_to_body(ap)
        if ch[0].part == "tarsus" or (ch[0].part == "tibia" and gap > max_gap):
            seg_of[id(ch)] = (None, "fragment", f"starts at a {ch[0].part} {gap:.0f} px from the body",
                              [f"fragment: starts at a {ch[0].part} with no {'tibia/femur' if ch[0].part == 'tarsus' else 'femur'}"
                               f" joining it to the body ({gap:.0f} px away); not ranked"], False)
    legs_ = sorted([ch for ch in chains if id(ch) not in seg_of], key=lambda ch: -info[id(ch)]["s"])   # anterior first
    lm_pos, lm_how = landmark_positions(lm, fr, W, H)
    near, near_how, lwarn = None, "", []
    if lateral:
        near, near_how, lwarn = lateral_near_side(fr, lm_dorsal, parts, W, H, opts)
        group = legs_
        if len(legs_) > 3 and opts.lateral_far_side == "flag":
            group = flag_far_side(legs_, info, fr, parts, opts, seg_of)
        rank_group(group, None, info, fr, lm_pos, lm_how, opts, seg_of, "in this lateral view")
    else:
        sides = {k: [ch for ch in legs_ if info[id(ch)]["img_side"] == k] for k in ("L", "R")}
        for k, chs in sides.items():
            rank_group(chs, sides["R" if k == "L" else "L"], info, fr, lm_pos, lm_how, opts, seg_of, "on this side")
    # the rule's name for every part (blind to any name the part already has)
    out = []
    for ci, ch in enumerate(chains):
        inf = info[id(ch)]
        r, method, evidence, w, strong = seg_of[id(ch)]
        margin = abs(inf["t"]); rel = margin / max(fr.half_width, 1e-9)
        cw = list(fwarn) + list(w)
        side = None
        if view is None:
            cw.append(f"view unknown ({vsrc}): give --view, --view_regex or an image '{opts.view_field}' field")
        elif lateral:
            cw += lwarn
            side = near if method != "far side?" else None
        elif rel < opts.min_side_margin:
            cw.append(f"anchor {margin:.1f} px from the midline (< {opts.min_side_margin} x half width): side not given")
        else:
            side = "left" if (inf["img_side"] == "L") == (view == "dorsal") else "right"
        if view in ("ventral", "lateral") and ch[0].part == "femur" and r is not None:
            cw.append("no coxa or trochanter found: anchored on the femur")   # (not said for dorsal views)
        ambiguous = None                                     # first ambiguous link from the anchor outwards
        for p in ch:
            pw = list(cw)
            if p.link_d is not None and p.link_d > LINK_WARN * fr.length:
                pw.append(f"long link to {p.parent.part} {p.parent.id} ({p.link_d:.0f} px)")
            if p.link_margin < opts.link_margin * fr.length and ambiguous is None:
                ambiguous = (f"ambiguous link to {p.parent.part} {p.parent.id}: another assignment is only "
                             f"{p.link_margin:.1f} px worse; this part and those beyond it not named")
            if ambiguous:
                pw.append(ambiguous)
            if p.skipped:
                pw.append(f"linked to {p.parent.part} {p.parent.id} across a missing {p.skipped}")
            if not lateral and p is not ch[0] and (fr.t(p.c) > 0) != (inf["t"] > 0):
                pw.append("lies across the midline from its leg's anchor (side inherited)")
            if ch[0].part in ("tibia", "tarsus") and r is not None:
                pw.append(f"leg anchored on a {ch[0].part} (no femur found)")
            if p.pieces > 1:
                pw.append(f"mask in {p.pieces} pieces (two parts under one label?)")
            named = side and r is not None and not ambiguous
            name = full_name(opts.name_format, SEGMENTS[r], p.part, side) if named else ""
            conf = ""
            if name:                                         # the "no coxa" note never lowers confidence
                real = [x for x in pw if not x.startswith("no coxa")]
                minor = [x for x in real if not x.startswith(("long link", "linked to"))]
                conf = "high" if method == "rank" and not real and ch[0].part in BASAL + ("femur",) else \
                       "low" if len(minor) > 1 or any("disagree" in x for x in pw) else "medium"
            if lateral:
                shown_side = side or ("far side?" if method == "far side?" else "")
            else:
                shown_side = side or ("image-" + ("left" if inf["img_side"] == "L" else "right"))
            weak_link = any(x.startswith(("long link", "linked to")) for x in pw)
            rank_clear = bool(named and strong and not weak_link and ch[0].part in BASAL + ("femur",) and
                              not any(("anchor only" in x) or ("disagree" in x) for x in pw))
            out.append(row(p.ann, p.part, p.existing, rule_name=name, side=shown_side,
                           segment=SEGMENTS[r] if r is not None else "",
                           view=view or "", side_margin_px="" if lateral else round(margin, 1),
                           side_margin_rel="" if lateral else round(rel, 3),
                           anchor=inf["anchor"] + (f"; {near_how}" if lateral and near_how else ""),
                           rank_method=method, rank_evidence=evidence,
                           linked_to=f"{p.parent.part} {p.parent.id}" if p.parent else "",
                           link_distance_px=round(p.link_d, 1) if p.link_d is not None else "",
                           confidence=conf, warnings="; ".join(pw), body_frame="; ".join(fr.notes),
                           _rule=(side, SEGMENTS[r]) if named else None,
                           _rule_side=side if side and not ambiguous else None,   # known even without a rank
                           _side_clear=bool(named and not lateral and not weak_link and rel >= opts.verify_margin),
                           _rank_clear=rank_clear,
                           _chain=f"{ci}b" if ambiguous else str(ci)))   # an ambiguous link splits the chain
    rows = nogeo + out
    decide(rows, opts)
    return rows


def decide(rows: List[dict], opts) -> None:
    """Final name of every leg-part row of one image -> final_name, status, reason, _out (category to write).

    rule mode: generic parts get the rule's name; parts already named in the input are left as they are
      (status 'input'), unless --check_existing, where the rule names every part.
    verify mode (model first, rule second): a part with a model name keeps it (written in --name_format) unless
      (a) another part of this image has the same model name; (b) the rule puts it on the other side with a clear
      margin (>= --verify_margin x half body width; never in lateral views, where far-side legs exist); (c) the
      rule ranks it differently with clear evidence (three femur/trochanter/coxa anchors with no warning, or an
      expected-position match twice as decisive as --ambiguity asks); (d) a part it is linked to (unambiguous
      links only) carries a different segment or side. Then the rule's name replaces it (both side and segment for
      a and d); where the rule has no name, (a) makes the part generic and (d) keeps the model name, flagged. A
      side-less model name (DINOMask --fold_lr) takes the rule's side, even where the rule cannot rank the leg.
      Parts with a 'score' below --min_score are named by the rule alone.
    Names given twice at the end: a name from the model or the input wins over one from the rule (which stays
    generic), so the model comes first."""
    fmt = opts.name_format
    use = {}                                                 # the model name actually used: (side, segment)
    for r in rows:
        m = r["_model"]
        if (m is not None and opts.mode == "verify" and opts.min_score is not None and r["_score"] is not None
                and r["_score"] < opts.min_score):
            r["reason"] = f"model score {r['_score']:.2f} < --min_score {opts.min_score}: the rule decides"
            m = None
        use[id(r)] = m
    dup = Counter((m[0], m[1], r["generic_class"]) for r in rows if (m := use[id(r)]) and m[0] and m[1])
    chain = defaultdict(list)
    for r in rows:
        if r["_chain"] is not None:
            chain[r["_chain"]].append(r)
    for r in rows:
        part, rule, m = r["generic_class"], r["_rule"], use[id(r)]
        why, generic = [], False
        if opts.mode == "verify" and m is not None:
            side, seg = m
            if side and seg and dup[(side, seg, part)] > 1:
                why.append(f"(a) '{r['_raw']}' is given to {dup[(side, seg, part)]} parts in this image")
                side, seg = rule or (None, None); generic = rule is None
            else:
                if rule and side and rule[0] != side and r["_side_clear"]:
                    why.append(f"(b) side: the rule puts it on the {rule[0]} (anchor {r['side_margin_rel']} x half "
                               f"body width from the midline)")
                    side = rule[0]
                if rule and seg and rule[1] != seg and r["_rank_clear"]:
                    why.append(f"(c) segment: the rule makes it {rule[1]} ({r['rank_evidence']})")
                    seg = rule[1]
                odd = [o for o in chain.get(r["_chain"], []) if o is not r and (om := use[id(o)]) and
                       ((om[0] and m[0] and om[0] != m[0]) or (om[1] and m[1] and om[1] != m[1]))]
                if odd:
                    why.append("(d) leg chain inconsistent: linked " + ", ".join(
                        f"{o['generic_class']} {o['annotation_id']} is '{o['_raw']}'" for o in odd))
                    if rule:
                        side, seg = rule
                    else:                                    # the links may be what is wrong: model name kept
                        why.append("the rule cannot name it: model name kept")
            if not generic:                                  # a side-less model name (--fold_lr) is completed
                if side is None and r["_rule_side"]:
                    side = r["_rule_side"]; why.append("side from the rule (the model name has none)")
                if seg is None and rule:
                    seg = rule[1]; why.append("segment from the rule (the model name has none)")
            final = (side, seg) if side and seg else None
            if final is None:
                status = "unresolved"
                why.append("the rule cannot name it" if generic else "model name incomplete and no rule name")
            elif final == (m[0] or final[0], m[1] or final[1]):
                status = "model"
                if rule and rule != final:
                    why.append(f"kept: the rule says {full_name(fmt, rule[1], part, rule[0])} but its evidence is "
                               f"not clear")
                elif not rule and not any("model name kept" in x for x in why):
                    why.append("kept: the rule cannot name it")
            else:
                status = "rule"
        elif opts.mode == "rule" and m is not None and not opts.check_existing:
            final, status = ((m[0], m[1]) if m[0] and m[1] else None), "input"     # named in the input: untouched
        else:
            final, status = rule, ("rule" if rule else "unresolved")
            generic = r["_model"] is not None and opts.mode == "verify" and not rule   # low score, no rule name
        r["_final"], r["status"], r["_generic"] = final, status, generic
        r["reason"] = "; ".join(x for x in [r["reason"]] + why if x)
    # a name given twice: model and input names win over rule names, which stay generic
    by_name = defaultdict(list)
    for r in rows:
        if r["_final"]:
            by_name[(r["_final"][0], r["_final"][1], r["generic_class"])].append(r)
    for (side, seg, part), rs in by_name.items():
        if len(rs) < 2:
            continue
        keep = [r for r in rs if r["status"] in ("model", "input")]
        for r in rs:
            if r["status"] == "input" or (len(keep) == 1 and r is keep[0]):
                continue
            r["reason"] = "; ".join(x for x in [r["reason"], f"{full_name(fmt, seg, part, side)} would be given "
                                                f"twice in this image: left generic"] if x)
            r["_generic"] = r["_model"] is not None and opts.mode == "verify"
            r["_final"], r["status"] = None, "unresolved"
    for r in rows:
        f = r["_final"]
        r["final_name"] = (r["_raw"] if r["status"] == "input" else
                           full_name(fmt, f[1], r["generic_class"], f[0]) if f else "")
        r["model_name"] = r["_raw"] if r["_model"] is not None else ""
        # category to write: the final name; generic when a model name had to go; else unchanged
        r["_out"] = (None if r["status"] == "input" else r["final_name"] if r["final_name"] else
                     r["generic_class"] if r["_generic"] else None)
        if opts.check_existing:
            ex = parse_name(r["existing_name"]) if r.get("existing_name") else None
            r["agrees"] = "" if not ex else str(r["_rule"] == (ex[0], ex[1]))


# ------------------------------------------------------------------ COCO in / out
def prepare(opts, coco: dict):
    cats = {c["id"]: c for c in coco.get("categories", [])}
    match = part_matcher({"coxa": opts.coxa, "trochanter": opts.trochanter, "femur": opts.femur,
                          "tibia": opts.tibia, "tarsus": opts.tarsus})
    opts._cat = lambda a: cats.get(a.get("category_id"), {}).get("name", "")

    def leg(a) -> Tuple[Optional[str], Optional[str]]:
        """-> (part, existing name for --check_existing). Generic names and full names in either style (and
        side-less ones) are all leg parts: every leg part takes part in the geometry."""
        name = opts._cat(a)
        p = match(name)
        if p:
            return p, None
        if (q := parse_name(name)):
            return q[2], (name if opts.check_existing else None)
        return None, None
    opts._leg = leg
    opts._body_rx = re.compile(opts.body_regex, re.I)
    opts._head_rx = re.compile(opts.head_regex, re.I)
    opts._lm_rx = {k: re.compile(v, re.I) for k, v in (("pro", opts.pro_regex), ("meso", opts.meso_regex),
                                                        ("meta", opts.meta_regex)) if v}
    opts._dorsal_rx = re.compile(opts.dorsal_regex, re.I) if opts.dorsal_regex else None
    return cats


def run(coco: dict, opts) -> Tuple[dict, List[dict], List[str]]:
    """-> (new COCO, result rows, notes). The input dict is not modified."""
    coco = copy.deepcopy(coco)
    notes = split_pieces(coco, opts) if opts.split_pieces else []
    prepare(opts, coco)
    ims, seen = [], set()
    for im in coco.get("images", []):
        if im.get("id") in seen:
            notes.append(f"image id {im.get('id')} ({im.get('file_name')}) is listed more than once: processed once")
            continue
        seen.add(im.get("id")); ims.append(im)
    by_im = defaultdict(list)
    for a in coco.get("annotations", []):
        by_im[a.get("image_id")].append(a)
    rows = []
    for im in ims:
        rows += assign_image(im, by_im.get(im.get("id"), []), opts)
    out = coco
    cats = out.setdefault("categories", [])
    by_name = {c["name"].strip().lower(): c for c in cats}
    next_id = max([c["id"] for c in cats if isinstance(c.get("id"), int)] + [0]) + 1
    src_cat = {c["id"]: c for c in cats}
    new_cat = {}
    ann_by_id = {a.get("id"): a for a in coco.get("annotations", [])}
    for r in rows:
        target = r["_out"]
        if not target or target == r["input_category"]:
            continue
        key = target.strip().lower()
        if key not in by_name:
            sup = src_cat.get(ann_by_id[r["annotation_id"]].get("category_id"), {}).get("supercategory", "")
            c = {"id": next_id, "name": target, "supercategory": sup}
            cats.append(c); by_name[key] = c; next_id += 1
        new_cat[r["annotation_id"]] = by_name[key]["id"]
    for a in out.get("annotations", []):
        if a.get("id") in new_cat:
            a["category_id"] = new_cat[a["id"]]
    info = out.get("info") if isinstance(out.get("info"), dict) else {}
    info["descriptron_serial_homology"] = {
        "program": VERSION, "mode": opts.mode, "named": sum(1 for r in rows if r["final_name"]),
        "status": dict(Counter(r["status"] for r in rows)),
        "unresolved": [{"annotation_id": r["annotation_id"], "image": r["image"], "generic_class": r["generic_class"],
                        "why": "; ".join(x for x in (r["reason"], r["warnings"]) if x)}
                       for r in rows if r["status"] == "unresolved"],
        "options": {k: v for k, v in vars(opts).items() if not k.startswith("_") and k not in ("input", "out", "csv")}}
    out["info"] = info
    return out, rows, notes


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog="\n\n".join(__doc__.split("\n\n")[2:]),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="COCO json with generic leg-part masks (and body/head masks)")
    ap.add_argument("--out", required=True, help="new COCO json with the full names (the input is not changed)")
    ap.add_argument("--csv", help="per-instance report (default: <out>_serial_homology.csv)")
    g = ap.add_argument_group("view and orientation")
    g.add_argument("--view", choices=["dorsal", "ventral", "lateral"],
                   help="view of every image (unless a field or --view_regex says otherwise)")
    g.add_argument("--view_regex", "--view-regex",
                   help="regex on the file name; first group or match: d/dorsal, v/ventral, l/lateral/profile")
    g.add_argument("--view_field", "--view-field", default="view", help="image-level field with the view (default: view)")
    g.add_argument("--anterior", choices=list(ANTERIOR_VEC), help="image direction of the head when there is no head mask")
    g.add_argument("--dorsum", choices=list(ANTERIOR_VEC) + ["auto"], default="up",
                   help="lateral views: image direction of the back (default up); auto = from --dorsal_regex masks, else the legs")
    g.add_argument("--dorsal_regex", "--dorsal-regex", default=r"pronot|elytr|notum|scutell|tergite|tegmin|hemelytr",
                   help="lateral views, --dorsum auto: categories on the dorsal side")
    g.add_argument("--lateral_far_side", "--lateral-far-side", choices=["ignore", "flag"], default="ignore",
                   help="lateral views with more than three legs: ignore = every leg is near side (extras not ranked); "
                        "flag = the legs whose bases lie on the body outline are flagged 'far side?'")
    g.add_argument("--body_regex", "--body-regex", default=r"\bbody\b|habitus|\bsoma\b|whole[\s_]*specimen",
                   help="categories that outline the whole body")
    g.add_argument("--head_regex", "--head-regex", default=r"\bhead\b", help="categories of the head (anterior end)")
    g.add_argument("--pro_regex", "--pro-regex", default=r"pronot|prothora|prostern", help="landmark: prothorax")
    g.add_argument("--meso_regex", "--meso-regex", default=r"elytr|hemelytr|tegm|forewing|mesonot|mesothora|mesostern|mesoventr",
                   help="landmark: front of the mesothorax / base of the forewings")
    g.add_argument("--meta_regex", "--meta-regex", default=r"abdom|gaster|metasoma|ventrite|urite|tergite|sternite",
                   help="landmark: abdomen (its front = metathoracic legs)")
    g = ap.add_argument_group("model first, rule second")
    g.add_argument("--mode", choices=["rule", "verify"], default="rule",
                   help="rule = name generic parts (default); verify = keep model names (e.g. DINOMask 'left_profemur') "
                        "unless duplicated, clearly on the other side, clearly another segment or inconsistent with "
                        "the leg chain")
    g.add_argument("--min_score", type=float, help="verify: parts with a 'score' below this are named by the rule alone")
    g.add_argument("--verify_margin", type=float, default=0.2,
                   help="verify: the rule's side overrides the model's when the anchor is this x half body width "
                        "from the midline or more")
    g = ap.add_argument_group("generic categories (comma-separated alternatives, case-insensitive, optional number)")
    g.add_argument("--coxa", default="coxa,coxae"); g.add_argument("--trochanter", default="trochanter,trochanters")
    g.add_argument("--femur", default="femur,femora")
    g.add_argument("--tibia", default="tibia,tibiae"); g.add_argument("--tarsus", default="tarsus,tarsi")
    g.add_argument("--split_pieces", "--split-pieces", action="store_true",
                   help="cut leg-part masks into their pieces first (needed for semantic models such as DINOMask, "
                        "which write one mask per class and image)")
    g.add_argument("--min_piece_px", type=float, default=50, help="--split_pieces: smallest piece (px) of its own")
    g.add_argument("--check_existing", "--check-existing", action="store_true",
                   help='treat "profemur (left)"-style categories as generic and report agreement with their names')
    g = ap.add_argument_group("rule parameters")
    g.add_argument("--midline", choices=["edges", "axis"], default="edges",
                   help="edges = local median of slice edge midpoints (default); axis = PCA axis through the centroid")
    g.add_argument("--midline_window", type=float, default=0.10, help="half window of the local midline, x body length")
    g.add_argument("--open_frac", type=float, default=0.3, help="opening radius, x the body's largest inscribed radius")
    g.add_argument("--min_elongation", type=float, default=1.25, help="below this length/width the axis comes from the head")
    g.add_argument("--max_gap", type=float, default=0.06, help="longest link between articulating parts, x body length")
    g.add_argument("--min_side_margin", type=float, default=0.05, help="anchor closer to the midline (x half width) -> no side")
    g.add_argument("--ambiguity", type=float, default=0.5, help="name a leg only if d(own) < this x d(next expected position)")
    g.add_argument("--link_margin", type=float, default=0.01,
                   help="a link whose best alternative is less than this x body length worse is ambiguous (parts beyond it not named)")
    g.add_argument("--name_format", default="suffix",
                   help="suffix = 'profemur (left)' (Descriptron editor), prefix = 'left_profemur' (DINOMask), or a "
                        "format with {segment} {part} {side}")
    g.add_argument("--work_size", type=int, default=1200, help="longest side (px) for the body raster")
    return ap


def main(argv=None):
    opts = build_parser().parse_args(argv)
    coco = load(opts.input)
    out, rows, notes = run(coco, opts)
    save(out, opts.out)
    csv_path = opts.csv or str(Path(opts.out).with_suffix("")) + "_serial_homology.csv"
    if rows:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if not k.startswith("_")], extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    for n in notes:
        print("note:", n)
    st = Counter(r["status"] for r in rows)
    print(f"{len(rows)} leg-part instances ({opts.mode} mode): " + ", ".join(
        f"{st[k]} {lab}" for k, lab in (("model", "model names kept"), ("rule", "named by the rule"),
                                        ("input", "already named, left as they are"), ("unresolved", "unresolved"))
        if st[k]))
    for r in rows:
        if r["status"] == "rule" and r["model_name"]:
            print(f"  replaced: {r['image']} {r['annotation_id']} {r['model_name']!r} -> {r['final_name']!r}: {r['reason']}")
    for r in rows:
        if r["status"] == "unresolved":
            print(f"  unresolved: {r['image']} annotation {r['annotation_id']} ({r['generic_class']}): "
                  + "; ".join(x for x in (r["reason"], r["warnings"]) if x))
    if opts.check_existing:
        cmp = [r for r in rows if r.get("existing_name")]
        ok = sum(1 for r in cmp if r.get("agrees") == "True")
        print(f"agreement with the existing names: {ok}/{len(cmp)}")
        for r in cmp:
            if r.get("agrees") != "True":
                print(f"  {r['image']} {r['annotation_id']}: existing {r['existing_name']!r} -> rule {r['rule_name'] or '(none)'!r}")
    print(f"wrote {opts.out} and {csv_path if rows else '(no CSV: no leg parts)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
