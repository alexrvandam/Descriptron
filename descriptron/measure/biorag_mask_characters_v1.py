#!/usr/bin/env python3
"""
biorag_mask_characters_v1.py — RETIRED (2026-09-20): discrete shape characters from mask outlines
===================================================================================================
NOT PART OF THE PIPELINE OR THE PAPER. Kept for reproducibility of the negative result only.

Procrustes GPA already analyses outline shape and its principal components are in the
Tier-1 matrix, so the elliptic-Fourier, elongation, solidity and circularity descriptors
computed here are a parallel encoding of the same signal and could not add anything. They
did not: 0 confirmed autapomorphies from 18,734 tests, and a novelty margin of 0.179
against 0.483 for the matrix. Use the matrix. The only non-redundant part was the counts
(lobes, notches, concavities), which are topological rather than Procrustes shape.

ORIGINAL HEADER FOLLOWS
-----------------------
biorag_mask_characters_v1.py — discrete shape characters mined from the masks
=============================================================================

Why discrete, and why this exists
---------------------------------
A range measured on n specimens is not a species' range: a further specimen falls
outside the observed minimum and maximum with probability about 2/(n+1) per
character however diagnostic that character is, so at the six specimens per species
typical of a revision, more than half of ordinary conspecifics look "outside".
That is a property of continuous characters read against sample ranges, and it is
what defeats range-based novelty detection on small series.

A discrete, invariant character escapes it entirely. A state present in all six
specimens of a species and in none of the other hundred-odd is overwhelming
evidence at exactly the sample size where a measurement is helpless — which is why
descriptive taxonomy has always leaned on qualitative diagnostic characters rather
than on measurements. This module mines such characters from the segmentation
masks the pipeline already produces, deterministically: no model is involved, so
the scores are reproducible by construction and there is no scorer reliability to
establish.

What it extracts, per specimen and structure
--------------------------------------------
Each is computed on the outline, then binned into a small number of ordered or
binary states against thresholds derived from the whole assemblage (never from one
species), so that a state means the same thing for every specimen:

  lobes / notches      curvature extrema on the smoothed outline: how many
                       protrusions and indentations the margin carries
  concavity            convexity defects deeper than a fraction of the outline
                       scale — the classic "emarginate / entire" distinction
  margin regularity    dispersion of curvature: a smooth margin against a
                       crenulate or serrate one
  outline harmonics    the first elliptic Fourier harmonics after normalisation
                       for size, rotation and starting point — shape independent
                       of how large the structure is or how it was mounted
  symmetry             agreement between the outline and its mirror image about
                       the principal axis
  elongation, solidity, rectangularity, circularity — binned, not continuous
  apex and base form   local curvature at the two ends of the principal axis
  skeleton topology    branch and end points: how many processes a structure has

  python biorag_mask_characters_v1.py --coco <coco.json> --taxon_profile <p.yaml> \\
      --out_dir "$M/mask_characters" [--image_dir <dir> --states 3]
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                       # noqa: E402

VERSION = "1.0"
MIN_POINTS = 12                     # an outline with fewer points carries no shape
SMOOTH = 0.012                      # contour smoothing, as a fraction of perimeter
N_HARMONICS = 8


# ─────────────────────────────────────────────────────────────────────────────
# outline geometry
# ─────────────────────────────────────────────────────────────────────────────
def resample(pts: np.ndarray, n: int = 256) -> np.ndarray:
    """Even arc-length resampling, so curvature is comparable between outlines
    traced with different numbers of vertices."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0, append=pts[:1]), axis=1))]
    if d[-1] <= 0:
        return pts
    u = np.linspace(0, d[-1], n, endpoint=False)
    return np.c_[np.interp(u, d, np.r_[pts[:, 0], pts[0, 0]]),
                 np.interp(u, d, np.r_[pts[:, 1], pts[0, 1]])]


def curvature(pts: np.ndarray, smooth: float = SMOOTH) -> np.ndarray:
    """Signed curvature of a closed outline, smoothed at a fixed fraction of the
    perimeter so the scale of detail is the same for every structure."""
    n = len(pts)
    k = max(3, int(round(smooth * n)) | 1)
    ker = np.ones(k) / k
    x = np.convolve(np.r_[pts[-k:, 0], pts[:, 0], pts[:k, 0]], ker, "same")[k:-k]
    y = np.convolve(np.r_[pts[-k:, 1], pts[:, 1], pts[:k, 1]], ker, "same")[k:-k]
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5
    denom[denom == 0] = np.nan
    return (dx * ddy - dy * ddx) / denom


def efd(pts: np.ndarray, harmonics: int = N_HARMONICS) -> np.ndarray:
    """Elliptic Fourier descriptors, normalised for size, rotation and starting
    point, so two outlines of the same shape give the same numbers whatever their
    size or how the specimen happened to be oriented on the slide."""
    dxy = np.diff(pts, axis=0, append=pts[:1])
    dt = np.linalg.norm(dxy, axis=1)
    dt[dt == 0] = 1e-9
    t = np.r_[0, np.cumsum(dt)]
    T = t[-1]
    out = np.zeros((harmonics, 4))
    for n in range(1, harmonics + 1):
        c = 2 * n * math.pi / T
        cs, sn = np.cos(c * t[1:]) - np.cos(c * t[:-1]), np.sin(c * t[1:]) - np.sin(c * t[:-1])
        out[n - 1, 0] = (T / (2 * n * n * math.pi ** 2)) * np.sum(dxy[:, 0] / dt * cs)
        out[n - 1, 1] = (T / (2 * n * n * math.pi ** 2)) * np.sum(dxy[:, 0] / dt * sn)
        out[n - 1, 2] = (T / (2 * n * n * math.pi ** 2)) * np.sum(dxy[:, 1] / dt * cs)
        out[n - 1, 3] = (T / (2 * n * n * math.pi ** 2)) * np.sum(dxy[:, 1] / dt * sn)
    a, b, c, d = out[0]
    theta = 0.5 * math.atan2(2 * (a * b + c * d), a * a + c * c - b * b - d * d)
    ct, st = math.cos(theta), math.sin(theta)
    rot = np.array([[ct, -st], [st, ct]])
    scale = math.hypot(a * ct + b * st, c * ct + d * st) or 1.0
    norm = []
    for n in range(harmonics):
        m = np.array([[out[n, 0], out[n, 1]], [out[n, 2], out[n, 3]]]) @ rot / scale
        norm.extend(m.ravel())
    return np.array(norm[4:])                      # first harmonic is the normaliser


def geometry(pts: np.ndarray) -> dict:
    """Everything measured on one outline, before any binning."""
    r = resample(pts)
    k = curvature(r)
    kf = k[np.isfinite(k)]
    hull = cv2.convexHull(r.astype(np.float32))
    area = abs(cv2.contourArea(r.astype(np.float32)))
    hull_area = abs(cv2.contourArea(hull)) or 1.0
    peri = cv2.arcLength(r.astype(np.float32), True) or 1.0

    c = r - r.mean(axis=0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    length = float(np.ptp(proj[:, 0])) or 1.0
    width = float(np.ptp(proj[:, 1])) or 1.0
    scale = math.sqrt(area) or 1.0

    # protrusions and indentations: curvature extrema beyond a scale-free threshold
    thr = 1.5 / scale
    pos = kf > thr
    neg = kf < -thr
    runs = lambda m: int(np.sum(np.diff(np.r_[0, m.astype(int), 0]) == 1))   # noqa: E731

    # departure from the convex hull: how deeply is the margin indented, and how many
    # separate indentations are there? Measured against the hull polygon directly, because
    # cv2.convexityDefects needs hull indices in traversal order and fails silently otherwise.
    hv = hull.reshape(-1, 2)
    # pointPolygonTest is POSITIVE inside the polygon, so an outline point lying well
    # inside its own convex hull is exactly a point in an indentation, and its distance
    # to the hull boundary is the depth of that indentation.
    dist = np.array([cv2.pointPolygonTest(hv.astype(np.float32), (float(x), float(y)), True)
                     for x, y in r])
    dist = np.maximum(dist, 0.0) / scale
    depth = float(dist.max())
    deep = runs(dist > 0.02)

    # symmetry about the principal axis
    mir = proj.copy()
    mir[:, 1] *= -1
    sym = float(np.mean(np.min(np.linalg.norm(proj[:, None, :] - mir[None, ::8, :], axis=2), axis=1))) / scale

    # ends of the principal axis: how pointed is the apex, how pointed the base
    order = np.argsort(proj[:, 0])
    apex = float(np.nanmean(k[order[-8:]])) * scale
    base = float(np.nanmean(k[order[:8]])) * scale

    return {"lobes": runs(pos), "notches": runs(neg), "deep_concavities": deep,
            "margin_irregularity": float(np.nanstd(kf) * scale),
            "elongation": length / width,
            "solidity": area / hull_area,
            "circularity": 4 * math.pi * area / (peri * peri),
            "rectangularity": area / (length * width),
            "symmetry": sym, "apex_curvature": apex, "base_curvature": base,
            "concavity_depth": depth,
            **{f"efd{i + 1}": v for i, v in enumerate(efd(r))}}


def skeleton_topology(pts: np.ndarray) -> dict:
    """How many processes does the structure have? Rasterise the outline small,
    thin it, and count branch and end points of the medial axis."""
    p = pts - pts.min(axis=0)
    s = 160.0 / max(p.max(), 1)
    q = np.round(p * s).astype(np.int32)
    h, w = q[:, 1].max() + 3, q[:, 0].max() + 3
    if h < 5 or w < 5:
        return {"skeleton_ends": 0, "skeleton_branches": 0}
    img = np.zeros((h, w), np.uint8)
    cv2.fillPoly(img, [q], 1)
    try:
        sk = cv2.ximgproc.thinning(img * 255) // 255
    except Exception:                               # opencv-contrib absent: erosion thinning
        sk = img.copy()
        for _ in range(60):
            er = cv2.erode(sk, np.ones((3, 3), np.uint8))
            tmp = cv2.subtract(sk, cv2.dilate(cv2.morphologyEx(sk, cv2.MORPH_OPEN,
                                                               np.ones((3, 3), np.uint8)),
                                              np.ones((3, 3), np.uint8)))
            sk = er if er.any() else sk
            if not er.any():
                break
        sk = (sk > 0).astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8),
                      borderType=cv2.BORDER_CONSTANT)
    return {"skeleton_ends": int(np.sum((sk == 1) & (nb == 1))),
            "skeleton_branches": int(np.sum((sk == 1) & (nb >= 3)))}


# ─────────────────────────────────────────────────────────────────────────────
from biorag_specimen_id import specimen_of   # moved out; kept for back-compatibility

def main():
    ap = argparse.ArgumentParser(description="Discrete shape characters from segmentation masks")
    ap.add_argument("--coco", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--matrix_dir", required=True, help="for the specimen -> species table")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--states", type=int, default=3, help="bins for the continuous quantities")
    ap.add_argument("--min_specimens_per_structure", type=int, default=8)
    ap.add_argument("--exclude_list", default=None,
                    help="annotation_screen/auto_exclusion_list.csv (and/or a manual list): every "
                         "annotation on a listed image is dropped. A polygon that has drifted off "
                         "the specimen still yields a number, and nothing downstream can tell that "
                         "number from a measurement")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    long = pd.read_csv(Path(a.matrix_dir) / "specimen_matrix_long.csv")
    species_of = dict(zip(long["specimen_id"], long["species"]))
    codes = sorted(set(species_of.values()))

    drop_ids, drop_imgs = load_exclusions(a.exclude_list)
    if drop_ids or drop_imgs:
        print(f"excluding {len(drop_ids)} flagged annotations"
              + (f" and every annotation on {len(drop_imgs)} images" if drop_imgs else ""))

    j = json.loads(Path(a.coco).read_text())
    cats = {c["id"]: c["name"] for c in j.get("categories", [])}
    imgs = {i["id"]: i["file_name"] for i in j.get("images", [])}
    rows, skipped, report = [], 0, {}
    for ann in j.get("annotations", []):
        seg = ann.get("segmentation")
        if not seg or not isinstance(seg, list) or not seg[0]:
            continue
        pts = np.asarray(seg[0], dtype=float).reshape(-1, 2)
        if len(pts) < MIN_POINTS:
            skipped += 1
            continue
        fn = imgs.get(ann["image_id"], "")
        if ann.get("id") in drop_ids or fn in drop_imgs:
            skipped += 1
            continue
        code, sid = specimen_of(fn, profile, codes, report)
        if not sid or sid not in species_of:
            skipped += 1
            continue
        try:
            g = geometry(pts)
        except Exception:                                      # a degenerate outline
            skipped += 1
            continue
        rows.append({"specimen_id": sid, "species": species_of[sid], "image": fn,
                     "structure": cats.get(ann["category_id"], "?"), **g})
    raw = pd.DataFrame(rows)
    if not len(raw):
        print("no usable outlines found")
        return
    # one row per specimen and structure: the median over repeated annotations
    num = [c for c in raw.columns if c not in ("specimen_id", "species", "image", "structure")]
    raw = (raw.groupby(["specimen_id", "species", "structure"], as_index=False)[num].median())
    raw.to_csv(out / "mask_geometry_raw.tsv", sep="\t", index=False)

    # ── binning ──────────────────────────────────────────────────────────────
    # Thresholds come from the whole assemblage, never from one species, so a state
    # label means the same thing for every specimen and cannot encode its identity.
    COUNTS = {"lobes", "notches", "deep_concavities"}
    states, defs = [], []
    for struct, g in raw.groupby("structure"):
        if g["specimen_id"].nunique() < a.min_specimens_per_structure:
            continue
        for col in num:
            v = pd.to_numeric(g[col], errors="coerce")
            if v.notna().sum() < a.min_specimens_per_structure or v.nunique() < 2:
                continue
            name = f"{struct}:{col}"
            if col in COUNTS:
                vals = v.round().astype("Int64")
                top = vals.value_counts().index[:4].tolist()
                st = vals.where(vals.isin(top), pd.NA).astype("string")
                cut = f"counts: {sorted([int(x) for x in top])}"
            else:
                qs = np.nanquantile(v, np.linspace(0, 1, a.states + 1))
                qs[0], qs[-1] = -np.inf, np.inf
                if len(set(np.round(qs[1:-1], 9))) < len(qs) - 2:
                    continue                                    # ties: not binnable
                labels = ["low", "mid", "high"] if a.states == 3 else \
                    [f"q{i + 1}" for i in range(a.states)]
                st = pd.cut(v, bins=qs, labels=labels).astype("string")
                cut = "quantiles: " + ", ".join(f"{x:.4g}" for x in qs[1:-1])
            for sid, s in zip(g["specimen_id"], st):
                if pd.notna(s):
                    states.append({"specimen_id": sid, "species": g.set_index("specimen_id")
                                   .loc[sid, "species"] if False else species_of.get(sid),
                                   "character": name, "state": str(s)})
            defs.append({"character": name, "structure": struct, "quantity": col,
                         "kind": "count" if col in COUNTS else "binned",
                         "thresholds": cut,
                         "n_specimens": int(v.notna().sum())})
    st = pd.DataFrame(states)
    st.to_csv(out / "mask_character_states.tsv", sep="\t", index=False)
    pd.DataFrame(defs).to_csv(out / "mask_character_definitions.tsv", sep="\t", index=False)

    summary = {"version": VERSION, "annotations_used": int(len(rows)),
               "annotations_skipped": int(skipped),
               "specimens": int(raw["specimen_id"].nunique()),
               "structures": int(raw["structure"].nunique()),
               "characters": int(st["character"].nunique()) if len(st) else 0,
               "state_assignments": int(len(st)),
               "states_per_character": a.states,
               "annotations_excluded_by_screen": len(drop_ids),
               "images_excluded_by_screen": len(drop_imgs),
               "abbreviations_inferred": report.get("abbreviations", {}),
               "images_not_matched_to_a_specimen": sorted(set(report.get("unmapped", []))),
               "note": "thresholds derived from the whole assemblage, never per species"}
    (out / "mask_characters_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
