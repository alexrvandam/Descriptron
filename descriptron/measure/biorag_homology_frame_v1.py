#!/usr/bin/env python3
"""
biorag_homology_frame_v1.py — put every specimen in the same anatomical frame
=============================================================================

Why this has to come first
--------------------------
When a vision model is asked where on a structure it saw a feature, and answers with a
point on the image it was given, that point is in the coordinate frame of *that crop*.
Two specimens photographed at different magnifications, mounted at different angles, or
simply differing in proportion, put the same anatomical place at different coordinates.
Pooling such points produces a heat map of nothing: it mixes non-homologous locations
and will look diffuse whether the character is real or invented.

So the images are brought into a common frame before anyone — model or person — looks
at them. This is the same homology machinery the colour and texture phenomics steps
already use, applied to the pictures rather than to the measurements:

  1  contour            the annotated polygon, resampled to N equally spaced
                        semilandmarks with the starting point fixed at one end of the
                        principal axis, so landmark i means the same thing everywhere
  2  Procrustes (GPA)   translation, rotation and scale removed, iterated to a
                        consensus; reflection is allowed only when asked for, because
                        left and right structures are not the same structure
  3  similarity only    each specimen is rotated to the common orientation and scaled
                        to a common size with Lanczos resampling, and NOTHING ELSE.
                        The pixels are never warped onto the consensus: the shape of
                        the structure is the thing being described, and a specimen
                        warped to the average shape no longer has a shape of its own.
                        Every image comes out the same size and orientation while
                        remaining a faithful picture of that individual.
  4  grid, warped LATER the homology grid is defined once on the CONSENSUS, then
                        thin-plate-spline warped from the consensus onto each
                        specimen's own landmarks, so its cells follow that
                        individual's anatomy. This happens AFTER scoring, purely to
                        look up which homologous cell a reported point fell in, and
                        to draw the correspondence for a reader to check. The grid is
                        never drawn on the image the model sees, because a visible
                        grid tells the model where to look and the whole point is to
                        find out where it looks unprompted.

After this, a point reported at (x, y) means the same anatomical place on every
specimen, a heat map of such points is interpretable, and "is it looking in the same
region across specimens?" becomes a measurement instead of an impression.

  python biorag_homology_frame_v1.py --coco <coco.json> --image_dir <dir> \\
      --taxon_profile <p.yaml> --matrix_dir "$M/compiled_key_tier" \\
      --out_dir "$M/homology_frames" --structures whole_wing pterostigma [--canvas 768]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                       # noqa: E402
from biorag_specimen_id import specimen_of                         # noqa: E402
from biorag_vlm_characters_v1 import load_exclusions                      # noqa: E402

VERSION = "1.0"
Image.MAX_IMAGE_PIXELS = None
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


# ─────────────────────────────────────────────────────────────────────────────
# 1. semilandmarks
# ─────────────────────────────────────────────────────────────────────────────
def load_gpa(gpa_dir: Path, struct: str):
    """Reuse the consensus and aligned contours the semilandmark step already produced.

    `color_phenomics_homology_v2_1.load_gpa_outputs` reads exactly these two files, so
    taking them here means the frames built for the model and the colour-homology cells
    computed for the matrix refer to the same consensus. Recomputing a second GPA would
    give a slightly different mean and quietly break that correspondence.
    Returns (consensus Nx2 in normalised units, {specimen_key: contour in image pixels}).
    """
    d = Path(gpa_dir) / struct
    mean_f, coco_f = d / "gpa_mean.csv", d / "back_transformed_coco.json"
    if not (mean_f.exists() and coco_f.exists()):
        return None, None
    mean = pd.read_csv(mean_f)[["x", "y"]].to_numpy(float)
    # gpa_mean.csv stores a closed ring whose last row repeats the first, and the contours in
    # back_transformed_coco.json carry the same repeat, so the duplicate is LEFT IN PLACE: the
    # two must have equal length for landmark i to correspond, and dropping it here silently
    # broke every match.
    j = json.loads(coco_f.read_text())
    imgs = {i["id"]: i["file_name"] for i in j.get("images", [])}
    out = {}
    for ann in j.get("annotations", []):
        seg = ann.get("segmentation")
        if not seg or not seg[0]:
            continue
        out[imgs.get(ann["image_id"], "")] = np.asarray(seg[0], float).reshape(-1, 2)
    return mean, out


def semilandmarks(pts: np.ndarray, n: int = 64) -> np.ndarray:
    """N points equally spaced along the outline, starting at one end of the principal
    axis and running in a fixed rotational direction.

    Both conventions matter. Without a fixed start, landmark i is an arbitrary point and
    the correspondence between specimens is random; without a fixed direction, half the
    specimens are traced backwards and the consensus collapses toward a circle.
    """
    p = np.asarray(pts, float)
    if len(p) < 4:
        return None
    # fixed direction: counter-clockwise in image coordinates
    area2 = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area2 < 0:
        p = p[::-1]
    # NOTE: the starting point is NOT fixed here. semi_landmark_and_kpts_procrustesV34_GPA
    # leaves it to an exhaustive cyclic-shift sweep inside the GPA (align_closed), which is
    # more robust than any single landmark rule: on a smooth outline the "furthest point
    # along the principal axis" moves between specimens and silently breaks correspondence.
    # equal arc length
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0, append=p[:1]), axis=1))]
    if d[-1] <= 0:
        return None
    u = np.linspace(0, d[-1], n, endpoint=False)
    return np.c_[np.interp(u, d, np.r_[p[:, 0], p[0, 0]]),
                 np.interp(u, d, np.r_[p[:, 1], p[0, 1]])]


# ─────────────────────────────────────────────────────────────────────────────
# 2. generalised Procrustes
# ─────────────────────────────────────────────────────────────────────────────
def _align(a: np.ndarray, b: np.ndarray, reflect: bool = False):
    """Rotate, scale and translate a onto b (ordinary Procrustes)."""
    ac, bc = a - a.mean(0), b - b.mean(0)
    sa = np.sqrt((ac ** 2).sum()) or 1.0
    sb = np.sqrt((bc ** 2).sum()) or 1.0
    ac, bc = ac / sa, bc / sb
    u, s, vt = np.linalg.svd(ac.T @ bc)
    R = u @ vt
    if not reflect and np.linalg.det(R) < 0:      # forbid mirroring: a left wing is not a right one
        vt[-1] *= -1
        s = s.copy()
        s[-1] *= -1
        R = u @ vt
    return ac @ R, R, sa


def align_closed(cons: np.ndarray, shape: np.ndarray, reflect: bool = False):
    """Best Procrustes fit over every cyclic shift of a closed outline.

    Semilandmarks on a closed curve have no natural first point, so correspondence has to
    be found rather than assumed. This is the sweep `align_closed` performs in
    semi_landmark_and_kpts_procrustesV34_GPA; it returns the aligned shape and the shift
    that produced it, so the same roll can be applied to the unaligned landmarks later.
    """
    best, best_k, best_d = None, 0, np.inf
    for k in range(len(shape)):
        rolled = np.roll(shape, -k, axis=0)
        a, _R, _s = _align(rolled, cons, reflect)
        d = float(((a - cons) ** 2).sum())
        if d < best_d:
            best, best_k, best_d = a, k, d
    return best, best_k


def gpa(shapes: np.ndarray, iters: int = 12, reflect: bool = False, tol: float = 1e-7):
    """Generalised Procrustes with a cyclic-shift sweep, as the semilandmark step does."""
    X = np.stack(shapes).astype(float)
    cons = X[0] - X[0].mean(0)
    cons /= np.sqrt((cons ** 2).sum()) or 1.0
    shifts = np.zeros(len(X), int)
    for _ in range(iters):
        res = [align_closed(cons, x, reflect) for x in X]
        al = np.stack([r[0] for r in res])
        shifts = np.array([r[1] for r in res])
        new = al.mean(0)
        new -= new.mean(0)
        new /= np.sqrt((new ** 2).sum()) or 1.0
        if np.linalg.norm(new - cons) < tol:
            cons = new
            break
        cons = new
    res = [align_closed(cons, x, reflect) for x in X]
    aligned = np.stack([r[0] for r in res])
    shifts = np.array([r[1] for r in res])
    return cons, aligned, shifts


# ─────────────────────────────────────────────────────────────────────────────
# 3-4. thin-plate spline warp of the pixels, onto a fixed canvas
# ─────────────────────────────────────────────────────────────────────────────
def to_canvas(shape: np.ndarray, canvas: int, margin: float = 0.08) -> np.ndarray:
    """Consensus (Procrustes units, centred at 0) to pixel coordinates on the canvas."""
    s = shape - shape.mean(0)
    half = np.abs(s).max() or 1.0
    k = (canvas * (0.5 - margin)) / half
    return s * k + canvas / 2.0


def similarity_to_frame(lm: np.ndarray, cons_px: np.ndarray):
    """The rotation, uniform scale and translation that best place this specimen in the
    common frame — and nothing more.

    A similarity transform cannot change shape: angles and ratios inside the structure
    survive untouched. That is the point. Using the full Procrustes fit here (or a thin
    plate spline) would pull the specimen onto the average outline and destroy the very
    differences the scorer is being asked to judge.
    """
    a = lm - lm.mean(0)
    b = cons_px - cons_px.mean(0)
    na = np.sqrt((a ** 2).sum()) or 1.0
    nb = np.sqrt((b ** 2).sum()) or 1.0
    u, sv, vt = np.linalg.svd((a / na).T @ (b / nb))
    R = u @ vt
    if np.linalg.det(R) < 0:
        vt[-1] *= -1
        R = u @ vt
    scale = (nb / na)
    t = cons_px.mean(0) - (lm.mean(0) @ R) * scale
    M = np.zeros((2, 3), np.float64)
    M[:, :2] = (R * scale).T
    M[:, 2] = t
    return M


def place(img: np.ndarray, M: np.ndarray, canvas: int) -> np.ndarray:
    """Apply the similarity transform and resample once, with Lanczos."""
    return cv2.warpAffine(img, M, (canvas, canvas), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def apply_M(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def build_tps_warp(src: np.ndarray, dst: np.ndarray, step: int = 5, smoothing: float = 1e-5):
    """Thin-plate spline mapping src landmarks onto dst, as build_tps_warp does in
    color_phenomics_homology_v2_1.

    Only every `step`-th landmark is used as a control point. That is not an economy: the
    interpolation matrix built from a full set of densely spaced semilandmarks is singular,
    because neighbouring landmarks carry almost the same constraint. Twenty control points
    out of a hundred is the ratio that pipeline settled on.
    """
    from scipy.interpolate import RBFInterpolator
    s = np.asarray(src, float)[::step]
    d = np.asarray(dst, float)[::step]
    wx = RBFInterpolator(s, d[:, 0], kernel="thin_plate_spline", smoothing=smoothing)
    wy = RBFInterpolator(s, d[:, 1], kernel="thin_plate_spline", smoothing=smoothing)

    def warp(q):
        q = np.asarray(q, float).reshape(-1, 2)
        return np.column_stack([wx(q), wy(q)])
    return warp


def warp_points(t, pts: np.ndarray) -> np.ndarray:
    return t(pts)


# ─────────────────────────────────────────────────────────────────────────────
# 5. the homology grid (saved, never drawn on what the model sees)
# ─────────────────────────────────────────────────────────────────────────────
def homology_grid(cons_px: np.ndarray, canvas: int, n_rows: int = 8, n_cols: int = 10,
                  margin: float = 0.05):
    """A lattice over the consensus bounding box, as create_grid_inside_contour builds it.

    A cell is kept when its CENTRE lies inside the consensus outline — the same test that
    pipeline uses. Cells are numbered by row letter and column number so a reported
    location has a name a person can look up on the reference sheet.
    """
    mask = np.zeros((canvas, canvas), np.uint8)
    cv2.fillPoly(mask, [cons_px.astype(np.int32)], 1)
    x0b, y0b = cons_px.min(0)
    x1b, y1b = cons_px.max(0)
    mx, my = margin * (x1b - x0b), margin * (y1b - y0b)
    xs = np.linspace(x0b - mx, x1b + mx, n_cols + 1)
    ys = np.linspace(y0b - my, y1b + my, n_rows + 1)
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            cx, cy = (xs[c] + xs[c + 1]) / 2, (ys[r] + ys[r + 1]) / 2
            if cv2.pointPolygonTest(cons_px.astype(np.float32), (float(cx), float(cy)), False) >= 0:
                cells.append({"cell": f"{chr(65 + r)}{c + 1}", "row": r, "col": c,
                              "x0": xs[c], "y0": ys[r], "x1": xs[c + 1], "y1": ys[r + 1]})
    return cells, mask


def main():
    ap = argparse.ArgumentParser(description="Warp every specimen into one homologous frame")
    ap.add_argument("--coco", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--structures", nargs="*", default=None)
    ap.add_argument("--landmarks", type=int, default=100)
    ap.add_argument("--canvas", type=int, default=768)
    ap.add_argument("--grid_rows", type=int, default=8)
    ap.add_argument("--grid_cols", type=int, default=10)
    ap.add_argument("--min_specimens", type=int, default=6)
    ap.add_argument("--allow_reflection", action="store_true",
                    help="permit mirroring during alignment. Off by default: a left and a "
                         "right structure are different structures, and allowing reflection "
                         "silently pools them")
    ap.add_argument("--exclude_list", default=None)
    ap.add_argument("--gpa_dir", default=None,
                    help="the semilandmark step's output (…/Diaphorina_semilandmarks). When given, "
                         "the consensus and the landmark correspondence are taken from "
                         "gpa_mean.csv and back_transformed_coco.json instead of being recomputed, "
                         "so these frames share a consensus with the colour-homology cells")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    long = pd.read_csv(Path(a.matrix_dir) / "specimen_matrix_long.csv")
    species_of = dict(zip(long["specimen_id"], long["species"]))
    codes = sorted(set(species_of.values()))
    drop_ids, drop_imgs = load_exclusions(a.exclude_list)

    j = json.loads(Path(a.coco).read_text())
    cats = {c["id"]: c["name"] for c in j.get("categories", [])}
    imgs = {i["id"]: i["file_name"] for i in j.get("images", [])}
    want = set(a.structures or [])
    per_struct = {}
    for ann in j.get("annotations", []):
        st = cats.get(ann["category_id"], "?")
        if want and st not in want:
            continue
        if ann.get("id") in drop_ids or imgs.get(ann["image_id"], "") in drop_imgs:
            continue
        seg = ann.get("segmentation")
        if not seg or not isinstance(seg, list) or not seg[0]:
            continue
        pts = np.asarray(seg[0], float).reshape(-1, 2)
        lm = semilandmarks(pts, a.landmarks)
        if lm is None:
            continue
        fn = imgs.get(ann["image_id"], "")
        _c, sid = specimen_of(fn, profile, codes)
        if not sid or sid not in species_of:
            continue
        per_struct.setdefault(st, {})
        if sid not in per_struct[st] or len(pts) > len(per_struct[st][sid][2]):
            per_struct[st][sid] = (lm, fn, pts)

    img_dir = Path(a.image_dir)
    report = []
    for st, recs in sorted(per_struct.items()):
        if len(recs) < a.min_specimens:
            continue
        sids = sorted(recs)
        cons_ext, lm_ext = (load_gpa(Path(a.gpa_dir), st) if a.gpa_dir else (None, None))
        if cons_ext is not None:
            cons = cons_ext
            shifts = np.zeros(len(sids), int)
            n_sub = 0
            for i_s, sid in enumerate(sids):
                lm0, fn0, pts0 = recs[sid]
                hit = next((v for k, v in lm_ext.items() if k.startswith(Path(fn0).name)), None)
                if hit is not None and len(hit) == len(cons):
                    recs[sid] = (hit, fn0, pts0)        # the pipeline's own landmarks
                    n_sub += 1
            print(f"  {st}: consensus and {n_sub}/{len(sids)} contours taken from the GPA step")
        else:
            cons, aligned, shifts = gpa([recs[s][0] for s in sids], reflect=a.allow_reflection)
        cons_px = to_canvas(cons, a.canvas)
        sdir = out / st
        sdir.mkdir(parents=True, exist_ok=True)
        cells, cmask = homology_grid(cons_px, a.canvas, a.grid_rows, a.grid_cols)
        np.save(sdir / "consensus_px.npy", cons_px)
        pd.DataFrame(cells).to_csv(sdir / "homology_grid.tsv", sep="\t", index=False)

        # a reference sheet for a person: consensus outline with the grid and its labels.
        # This file is for us, not for the model.
        ref = np.full((a.canvas, a.canvas, 3), 255, np.uint8)
        cv2.polylines(ref, [cons_px.astype(np.int32)], True, (40, 40, 40), 2)
        for c in cells:
            cv2.rectangle(ref, (int(c["x0"]), int(c["y0"])), (int(c["x1"]), int(c["y1"])),
                          (170, 190, 205), 1)
            cv2.putText(ref, c["cell"], (int(c["x0"]) + 4, int(c["y0"]) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 140, 160), 1, cv2.LINE_AA)
        Image.fromarray(ref).save(sdir / "_consensus_grid_reference.png")

        n_ok = 0
        cell_poly = {c["cell"]: np.array([[c["x0"], c["y0"]], [c["x1"], c["y0"]],
                                          [c["x1"], c["y1"]], [c["x0"], c["y1"]]], float)
                     for c in cells}
        warped_grids = {}
        for i_s, sid in enumerate(sids):
            lm, fn, pts = recs[sid]
            if int(shifts[i_s]):
                lm = np.roll(lm, -int(shifts[i_s]), axis=0)   # the shift GPA chose for this shape
            p = img_dir / fn
            if not p.exists():
                continue
            with Image.open(p) as raw:
                im = np.asarray(raw.convert("RGB"))
            m = np.zeros(im.shape[:2], np.uint8)
            cv2.fillPoly(m, [pts.astype(np.int32)], 255)
            masked = np.where(m[..., None] > 0, im, np.uint8(255))

            M = similarity_to_frame(lm, cons_px)
            placed = place(masked, M, a.canvas)
            lm_frame = apply_M(M, lm)                       # this specimen's landmarks, in frame
            Image.fromarray(placed).save(sdir / f"{sid}.png")

            # the grid follows the individual: consensus landmarks -> this specimen's
            t_fwd = build_tps_warp(cons_px, lm_frame)      # consensus -> this specimen
            warped_grids[sid] = {name: warp_points(t_fwd, poly).tolist()
                                 for name, poly in cell_poly.items()}
            np.save(sdir / f"{sid}_landmarks.npy", lm_frame)
            n_ok += 1
        (sdir / "warped_grids.json").write_text(json.dumps(warped_grids))
        report.append({"structure": st, "specimens": len(sids), "written": n_ok,
                       "grid_cells": len(cells)})
        print(f"  {st:20s} {n_ok}/{len(sids)} warped, {len(cells)} grid cells")

    summary = {"version": VERSION, "canvas": a.canvas, "landmarks": a.landmarks,
               "grid": f"{a.grid_rows}x{a.grid_cols}", "reflection_allowed": bool(a.allow_reflection),
               "structures": report,
               "note": "the grid is saved beside each structure and is NOT drawn on the "
                       "specimen images; a visible grid would tell a scorer where to look"}
    (out / "homology_frames_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
