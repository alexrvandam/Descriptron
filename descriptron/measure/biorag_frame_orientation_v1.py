#!/usr/bin/env python3
"""
biorag_frame_orientation_v1.py — which homology frames are half a turn out?
===========================================================================

The problem this exists to fix
------------------------------
`biorag_homology_frame_v1.py` puts every specimen of a structure into one frame by a
SIMILARITY transform onto the Procrustes consensus: rotation, uniform scale, translation,
no reflection, no warping. Correspondence along a closed outline has no natural first
point, so the fit sweeps every cyclic shift of the semilandmarks and keeps the best. That
sweep is what makes the method robust on a lumpy outline — and it is also what makes it
blind on an elongate one.

An elongate structure that is nearly invariant under a half turn (a tibia, a labial
segment, a femur, a paramere, a pterostigma, most wing cells) fits its own consensus
almost as well upside down as the right way up: the two minima of the Procrustes residual
differ by less than the between-specimen noise, so which one wins is decided by noise.
The frames of ONE structure then sit in TWO orientations, half a turn apart, and nothing
downstream can tell. A vision model asked where it saw a character follows the anatomy to
whichever end that anatomy is on, and its reports split between the two ends of the frame
with nothing in between. The localisation statistic computed on those reports —
spread = mean distance of the reports from their own centroid — then measures frame
orientation, not whether the model looked at the same anatomical place twice. In the run
that prompted this script, `tibia: Proximal knob with notched excavation` had its 61
reports split 37/24 between the two ends and a spread of 0.483, and 9 of the 16 characters
classed "scattered" were two-ended in the same way.

Why not simply re-run the Procrustes fit
----------------------------------------
Because the full-outline Procrustes distance is the statistic that already failed: on a
nearly two-fold-symmetric outline it is, by construction, almost the same for both
orientations. Deciding the flip needs a cue that is LARGE where the Procrustes residual is
small — something that differs between the two ENDS.

The cue: the width profile along the long axis
----------------------------------------------
A tibia is not symmetric end to end even when its outline is: one end carries the
proximal knob and is broad, the other tapers. Project a specimen's outline onto the
consensus long axis, rasterise it, and take the filled width of each column: that is a
profile w(t), t running from one end of the structure to the other. Under a half turn the
profile reverses, w(t) -> w(1-t), and nothing else about it changes. So:

  * split each (centred, unit-norm) profile into its symmetric and antisymmetric parts.
    Reversal flips the sign of the antisymmetric part and leaves the symmetric part alone,
    so the ENTIRE orientation signal lives in the antisymmetric part and the symmetric
    part — which is most of the profile, and most of the Procrustes residual — is
    correctly ignored;
  * the reference direction is the leading right singular vector of the matrix of
    antisymmetric parts. This is deliberately NOT a mean over the specimens as they stand:
    if the frames are in two orientations, that mean is a mixture and cancels towards zero,
    which is precisely how the consensus lost the information in the first place. The
    singular vector is indifferent to the signs, and the majority of the projections onto
    it then fixes which orientation is called "as is";
  * with the signs known, the specimens are turned the same way up and averaged into a
    reference profile w_ref. The two reported scores are the correlations of the specimen's
    profile with w_ref, as it stands and reversed; since both profiles are centred and
    unit-norm their difference is exactly twice the agreement of the antisymmetric parts,
    so the margin between the two scores is a direct measure of how much end-to-end
    asymmetry this individual actually has AND how well it agrees with the structure's.
    Half that margin is reported as the confidence.

A specimen with no end-to-end asymmetry of its own gets a margin near zero, and is left
alone and flagged rather than guessed at. Where the profile is undecided, a second pass
uses image content: the greyscale frame is correlated with the mean frame — built from
frames already turned the same way up, so it is not itself a mixture — as it stands and
turned half a turn. Pigment, sculpture and setal pits are strongly end-biased in these
structures. Anything still undecided keeps method "undecided" and is NOT flipped.

The second cue is not only a fallback: it is the check
------------------------------------------------------
The width profile alone over-flips, and the way it fails is instructive. Where the
structure's end-to-end asymmetry is weak, each specimen's antisymmetric profile is mostly
noise, and a minority of specimens will have one pointing the other way for no reason at
all. Flipping those turns a set of frames that was already consistent INTO a mixture. It
cannot be caught by looking at the reports: moving a minority of reports onto the majority
always lowers the spread, whether or not the frame was ever out, so the statistic this
script exists to repair would applaud the damage. It is plainly visible in the mean frame,
which acquires a ghost of the reversed structure — and that is a picture, which is the
other cue. So:

  * the picture verdict is computed for EVERY frame, not only the undecided ones;
  * a structure keeps its flips only if the picture agrees to turn at least
    `--min_corroboration` of the frames the profile wants turned. Below that, every flip on
    that structure is withdrawn (method "uncorroborated") and the frames stand as they are;
  * within a corroborated structure, a single flip that the picture confidently contradicts
    is withdrawn too (method "conflict"). A half turn changes the outline AND the picture;
    one cue alone saying so is not a half turn.

On this dataset that gate matters: the profile pass wanted to turn frames on twelve
structures, and the picture confirmed only two of them (tibia 0.84, LAB2 0.77; every other
structure scored 0.00–0.33 and was withdrawn). The two it confirmed are the two whose mean
frame is measurably sharper afterwards.

Structures at risk, and the negative control
--------------------------------------------
Two per-structure numbers say whether the ambiguity can arise at all, and both are
reported in every row:

  elongation      extent of the consensus along its principal axis over the extent across
                  it. A compact outline has no two ends to confuse.
  end_asymmetry   how far the consensus is from being invariant under a half turn: the
                  consensus, centred and scaled to unit centroid size, is rotated 180
                  degrees and matched back to itself over every cyclic shift of the
                  landmarks — WITHOUT re-fitting the rotation, since re-fitting it would
                  simply undo the half turn and return zero for every shape. The residual
                  is that distance. Small means the two orientations are nearly
                  indistinguishable to the Procrustes fit; large means they are not.

A structure that is elongate AND nearly half-turn-symmetric is at risk. A structure that is
not (whole_wing, LAB1) is the negative control: the same detector run unchanged on it
should find essentially no flips, and if it finds many the detector is wrong rather than
the frames. These two numbers flag the candidates; they do not decide, and are not used to
gate anything — some structures they flag (pterostigma, the wing cells) turn out to be
consistently oriented, and only the corroboration test above tells them apart.

Output
------
  frame_orientation.tsv   structure, specimen_id, flipped (0/1), method
                          (profile | ncc | undecided | conflict | uncorroborated),
                          score_as_is, score_rotated, confidence, elongation,
                          end_asymmetry, plus at_risk and the two picture agreements.

Downstream, `biorag_vlm_heatmap_figures_v1.py --orientation <this file>` maps a reported
(x, y) on a flipped frame to (1 - x, 1 - y), turns that frame half a turn before it enters
the mean background, and turns its outline likewise before the off-the-specimen test. The
frames on disk are NOT rewritten: the transform is a half turn about the frame centre,
exactly invertible, and a table of flips is auditable in a way that silently re-saved
images are not.

  python biorag_frame_orientation_v1.py --frames_dir "$M/homology_frames" \\
      --out "$M/homology_frames/frame_orientation.tsv" [--structures tibia LAB2]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

VERSION = "1.0"
Image.MAX_IMAGE_PIXELS = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. the consensus: is this structure at risk at all?
# ─────────────────────────────────────────────────────────────────────────────
def ring(pts: np.ndarray) -> np.ndarray:
    """The outline without the repeated closing point.

    gpa_mean.csv and everything derived from it store a CLOSED ring whose last row repeats
    the first. The duplicate is left in place everywhere else because landmark i must mean
    the same thing in the consensus and in the contours; here it would be counted twice in
    a cyclic sweep, so it is dropped locally and nothing on disk is touched."""
    p = np.asarray(pts, float)
    return p[:-1] if len(p) > 2 and np.allclose(p[0], p[-1]) else p


def principal_axis(pts: np.ndarray):
    """Unit vector along the long axis, and the unit vector across it."""
    c = np.asarray(pts, float).mean(0)
    u = np.linalg.svd(np.asarray(pts, float) - c, full_matrices=False)[2][0]
    return u / (np.linalg.norm(u) or 1.0), np.array([-u[1], u[0]]) / (np.linalg.norm(u) or 1.0)


def elongation_of(pts: np.ndarray) -> float:
    u, v = principal_axis(pts)
    c = np.asarray(pts, float).mean(0)
    du = float(np.ptp((pts - c) @ u))
    dv = float(np.ptp((pts - c) @ v))
    return du / dv if dv > 0 else float("inf")


def rotation_asymmetry(pts: np.ndarray) -> float:
    """Distance from having a two-fold rotational symmetry.

    The shape is centred and scaled to unit centroid size, turned half a turn, and matched
    back to itself over every cyclic shift of the landmarks. The rotation is NOT re-fitted:
    a free rotation would cancel the half turn and give zero for every shape, which is the
    degenerate answer, not the question. Zero means a half turn maps the outline onto
    itself — the case in which the Procrustes fit that built the frames cannot tell the two
    orientations apart."""
    p = ring(pts)
    p = p - p.mean(0)
    n = np.sqrt((p ** 2).sum()) or 1.0
    p = p / n
    r = -p                                                 # the half turn, about the centroid
    # only cyclic shifts: a half turn preserves the direction in which the ring is traced,
    # so re-tracing it backwards is not one of the correspondences on offer — allowing it
    # would report a mirror symmetry as though it were a rotational one.
    best = np.inf
    for k in range(len(p)):
        best = min(best, float(np.sqrt(((r - np.roll(p, k, axis=0)) ** 2).sum())))
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 2. the cue: width along the long axis
# ─────────────────────────────────────────────────────────────────────────────
def width_profile(lm: np.ndarray, u: np.ndarray, v: np.ndarray, nbins: int, rows: int = 256):
    """Filled width of the outline in each of `nbins` slices taken across the long axis.

    The outline is rasterised rather than binned by landmark, so a concave margin, an
    uneven landmark spacing and an empty bin all behave. Both axes are normalised by this
    specimen's own extent: a profile is then a shape, independent of how long or how broad
    the individual is, which is what has to be compared across specimens."""
    p = np.asarray(lm, float)
    c = p.mean(0)
    q = np.c_[(p - c) @ u, (p - c) @ v]
    span = q.max(0) - q.min(0)
    if span[0] <= 0 or span[1] <= 0:
        return None
    xy = (q - q.min(0)) / span * np.array([nbins - 1, rows - 1])
    m = np.zeros((rows, nbins), np.uint8)
    cv2.fillPoly(m, [np.round(xy).astype(np.int32)], 1)
    w = m.sum(0).astype(float)
    return w if w.sum() > 0 else None


def unit(w: np.ndarray):
    """Centred and scaled to unit norm, so a dot product is a correlation."""
    x = np.asarray(w, float) - np.mean(w)
    n = np.linalg.norm(x)
    return (x / n) if n > 0 else None


def orient_by_profile(profiles: dict, iters: int = 3):
    """Turn every profile the same way up, without ever averaging a mixture.

    Reversal flips the sign of the antisymmetric part of a profile and leaves the symmetric
    part untouched, so the orientation lives entirely in the antisymmetric part. The
    leading right singular vector of those antisymmetric parts is the direction they share
    REGARDLESS of their signs — a mean would cancel it. Projections onto that direction
    give the signs; the majority sign is called "as is". The reference profile is then a
    mean over profiles already turned the same way, and is re-derived a couple of times.
    Returns {sid: (score_as_is, score_rotated)} and the reference profile."""
    sids = sorted(profiles)
    W = np.stack([profiles[s] for s in sids])
    A = (W - W[:, ::-1]) / 2.0                                  # antisymmetric parts
    if np.linalg.norm(A) <= 0:
        return {s: (0.0, 0.0) for s in sids}, W.mean(0)
    v1 = np.linalg.svd(A, full_matrices=False)[2][0]
    proj = A @ v1
    if (proj < 0).sum() > (proj > 0).sum():                     # majority orientation = "as is"
        v1, proj = -v1, -proj
    sign = np.where(proj >= 0, 1.0, -1.0)
    ref = None
    for _ in range(iters):
        turned = np.where(sign[:, None] > 0, W, W[:, ::-1])
        ref = unit(turned.mean(0))
        if ref is None:
            ref = turned.mean(0)
            break
        s_as, s_rot = W @ ref, W[:, ::-1] @ ref
        new = np.where(s_as >= s_rot, 1.0, -1.0)
        if np.array_equal(new, sign):
            break
        sign = new
    return {s: (float(W[i] @ ref), float(W[i, ::-1] @ ref)) for i, s in enumerate(sids)}, ref


# ─────────────────────────────────────────────────────────────────────────────
# 3. the fallback cue: image content against a cleaned mean frame
# ─────────────────────────────────────────────────────────────────────────────
def grey(path: Path, px: int):
    with Image.open(path) as im:
        g = im.convert("L")
        if g.width != px or g.height != px:
            g = g.resize((px, px), Image.BILINEAR)
        return 255.0 - np.asarray(g, np.float32)                # ink, not paper


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    x, y = unit(a.ravel()), unit(b.ravel())
    return float(x @ y) if x is not None and y is not None else 0.0


def orient_by_image(sdir: Path, sids, anchor: dict, px: int, min_anchor: int = 4):
    """Correlate each frame with the MEAN frame, as it stands and turned half a turn.

    The mean has to be built from frames that are already the same way up: a mean over a
    mixture of orientations is symmetric by construction and answers every question with a
    tie — the same cancellation that lost the information in the consensus. The profile
    pass supplies those frames where it decided confidently, but on a structure whose
    outline carries no end-to-end asymmetry at all (a femur) it decides almost nothing, so
    the frames are turned the same way up by the same argument used on the profiles,
    applied to pictures: reversal here is a half turn, the half-turn-antisymmetric part of
    each frame flips sign under it, and the leading right singular vector of those parts is
    the direction they share whatever their signs.

    That fixes the orientations only up to ONE global sign, which the profile pass anchors
    by majority agreement; with no confident profile frame to anchor it, the majority of the
    frames is called "as is", as in the profile pass. Anchoring can only inflate the
    agreement, which is why the test that matters downstream is not this overall agreement
    but whether the picture confirms the particular frames the profile wants turned.

    Pigment, sculpture and setal pits are strongly end-biased in these structures, which is
    why picture content can separate ends that a symmetric outline cannot. Scores are
    returned for EVERY frame, not only the ones the profile could not decide: they are also
    the second opinion that decides whether the profile's flips are real (see main).
    Returns ({sid: (ncc_as_is, ncc_rotated)}, agreement with the profile pass)."""
    keep = [s for s in sids if (sdir / f"{s}.png").exists()]
    if len(keep) < 4:
        return {}, float("nan")
    G = np.stack([grey(sdir / f"{s}.png", px) for s in keep])
    A = (G - G[:, ::-1, ::-1]).reshape(len(G), -1) / 2.0
    A -= A.mean(1, keepdims=True)
    if not np.isfinite(A).all() or np.linalg.norm(A) <= 0:
        return {}, float("nan")
    v1 = np.linalg.svd(A, full_matrices=False)[2][0]
    proj = A @ v1
    at = np.array([1.0 if not anchor.get(s, False) else -1.0 for s in keep if s in anchor])
    ap = np.array([proj[i] for i, s in enumerate(keep) if s in anchor])
    agree = float("nan")
    if len(at) >= min_anchor:
        agree = float(np.mean(np.sign(ap) == at))
        if agree < 0.5:                       # the singular vector's sign is arbitrary
            proj, agree = -proj, 1.0 - agree
    elif (proj < 0).sum() > (proj > 0).sum():
        proj = -proj
    m = np.where(proj[:, None, None] >= 0, G, G[:, ::-1, ::-1]).mean(0)
    return {s: (ncc(G[i], m), ncc(G[i, ::-1, ::-1], m)) for i, s in enumerate(keep)}, agree


# ─────────────────────────────────────────────────────────────────────────────
# 4. the cue that needs no threshold: the anatomy next door   (added 2026-09-21)
#
# Everything above infers the polarity of a frame from the frame. The photographs
# themselves carry the answer: a metaleg image holds the femur AND the tibia, a rostrum
# image holds both labial segments, a male terminalia image holds the two halves of the
# aedeagus. The end of the tibia that touches the femur is the proximal end, wherever the
# frame happens to have put it, and that is a fact about the specimen rather than a
# statistic over the sample. Reading it takes no threshold and no corroboration; it takes
# only the neighbouring polygon, which the COCO file already has.
#
# Getting from the picture into the frame needs the transform the frame was built with.
# biorag_homology_frame_v1 does not save it, but it saves the LANDMARKS in frame
# coordinates, and when it is run with --gpa_dir those landmarks are the semilandmark
# step's own back-transformed contour, point for point. Matching the two by index recovers
# the rotation exactly (residual ~1e-15 on this data set), so a direction or a point read
# off the photograph can be carried into the frame with nothing estimated.
# ─────────────────────────────────────────────────────────────────────────────
def _ring_np(p):
    p = np.asarray(p, float)
    return p[:-1] if len(p) > 2 and np.allclose(p[0], p[-1]) else p


def _unit_shape(p):
    q = np.asarray(p, float)
    q = q - q.mean(0)
    return q / (np.sqrt((q ** 2).sum()) or 1.0)


def kabsch_rotation(src: np.ndarray, dst: np.ndarray):
    """The rotation R with src @ R ~ dst, for point sets already in correspondence.

    No reflection: biorag_homology_frame_v1.similarity_to_frame forbids it, so admitting
    one here would fit a transform the frames were never built with and quietly absorb the
    very mirroring part (B) exists to detect. Returns (R, residual after centring and
    scaling both)."""
    a, b = _unit_shape(src), _unit_shape(dst)
    u, _s, vt = np.linalg.svd(a.T @ b)
    R = u @ vt
    if np.linalg.det(R) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        R = u @ vt
    return R, float(np.sqrt((((a @ R) - b) ** 2).sum()))


def recover_similarity(lm_img: np.ndarray, lm_frame: np.ndarray):
    """The rotation, uniform scale and translation biorag_homology_frame_v1 used, recovered
    from the landmarks it saved. `p_frame = p_image @ R * scale + t`, so any point of the
    photograph — the centroid of a neighbouring polygon, say — can be placed in the frame
    without re-fitting anything. Returns (R, scale, t, residual)."""
    a, b = _ring_np(lm_img), _ring_np(lm_frame)
    R, res = kabsch_rotation(a, b)
    na = np.sqrt(((a - a.mean(0)) ** 2).sum()) or 1.0
    nb = np.sqrt(((b - b.mean(0)) ** 2).sum()) or 1.0
    scale = nb / na
    t = b.mean(0) - (a.mean(0) @ R) * scale
    return R, float(scale), t, res


def load_gpa_contours(gpa_dir, struct: str) -> dict:
    """{image file name: contour in IMAGE pixels} from the semilandmark step's
    back_transformed_coco.json — the same points biorag_homology_frame_v1 placed in the
    frame, in the same order, so index i means the same landmark in both."""
    f = Path(gpa_dir) / struct / "back_transformed_coco.json"
    if not f.exists():
        return {}
    j = json.loads(f.read_text())
    names = {i["id"]: i["file_name"] for i in j.get("images", [])}
    out = {}
    for ann in j.get("annotations", []):
        seg = ann.get("segmentation")
        if not seg or not seg[0]:
            continue
        out[names.get(ann["image_id"], "")] = np.asarray(seg[0], float).reshape(-1, 2)
    return out


def image_key(name: str) -> str:
    """back_transformed_coco.json appends the annotation index to the file name
    ('…_SV.tif_175'); the COCO the neighbours come from does not."""
    s = str(name)
    for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
        if ext in s.lower():
            i = s.lower().rindex(ext)
            return s[: i + len(ext)]
    return s


def load_coco_polygons(coco_path) -> dict:
    """{image file name: {structure: largest polygon in image pixels}}."""
    j = json.loads(Path(coco_path).read_text())
    cats = {c["id"]: c["name"] for c in j.get("categories", [])}
    names = {i["id"]: i["file_name"] for i in j.get("images", [])}
    out = {}
    for ann in j.get("annotations", []):
        seg = ann.get("segmentation")
        if not seg or not isinstance(seg, list) or not seg[0]:
            continue
        if isinstance(seg[0], (int, float)):
            seg = [seg]
        rings = [np.asarray(s, float).reshape(-1, 2) for s in seg
                 if isinstance(s, (list, tuple)) and len(s) >= 8]
        if not rings:
            continue
        r = max(rings, key=lambda q: abs(np.sum(q[:, 0] * np.roll(q[:, 1], -1)
                                                - np.roll(q[:, 0], -1) * q[:, 1])))
        st, fn = cats.get(ann["category_id"], "?"), names.get(ann["image_id"], "")
        d = out.setdefault(fn, {})
        if st not in d or len(r) > len(d[st]):
            d[st] = r
    return out


def densify(p: np.ndarray, step: float = 3.0) -> np.ndarray:
    """A polygon resampled densely enough that the nearest vertex is the nearest point."""
    p = _ring_np(p)
    out = []
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        n = max(1, int(np.linalg.norm(b - a) / max(step, 1e-6)))
        out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(out) if out else p


def axis_coordinate(lm_frame: np.ndarray, cons: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Every landmark's position along the frame's long axis, scaled to [0, 1] over this
    specimen's own extent. 0 is one end of the frame and 1 the other; which end is which is
    fixed by the consensus, so the number is comparable between specimens."""
    t = (np.asarray(lm_frame, float) - np.asarray(cons, float).mean(0)) @ u
    return (t - t.min()) / (np.ptp(t) or 1.0)


def neighbour_end(lm_img: np.ndarray, lm_frame: np.ndarray, neighbour: np.ndarray,
                  cons: np.ndarray, u: np.ndarray, k: int = 5) -> float:
    """Where along the FRAME the end that touches the neighbour lies.

    The k landmarks of this structure closest to the neighbouring polygon are found in
    IMAGE coordinates — where the adjacency is a physical fact — and their position along
    the frame's long axis is averaged. A joint is a bend, so the direction from one
    centroid to the other is not the structure's own axis (on the femur it is oblique in
    half the specimens); the point of contact is not affected by the bend."""
    d = np.sqrt(((_ring_np(lm_img)[:, None, :] - neighbour[None, :, :]) ** 2).sum(-1)).min(1)
    t = axis_coordinate(_ring_np(lm_frame), cons, u)
    idx = np.argsort(d)[: max(1, k)]
    return float(np.mean(t[idx]))


def axis_direction_cos(lm_img: np.ndarray, lm_frame: np.ndarray, d_img: np.ndarray,
                       u: np.ndarray) -> float:
    """Cosine between an anatomical direction read off the photograph and the frame's long
    axis. Used where the neighbour is not at an end but the pair of neighbours defines a
    direction — the forewing, whose cells give a base and an apex but whose margin touches
    nothing."""
    R, _res = kabsch_rotation(_ring_np(lm_img), _ring_np(lm_frame))
    dv = np.asarray(d_img, float) @ R
    n = float(np.linalg.norm(dv)) or 1.0
    return float((dv @ u) / n)


def parse_pairs(spec) -> dict:
    """--neighbours tibia=femura LAB2=LAB1 subgenital_plate=male_proctiger,female_proctiger"""
    out = {}
    for item in (spec or []):
        if "=" not in str(item):
            continue
        k, v = str(item).split("=", 1)
        out.setdefault(k.strip(), []).extend([x.strip() for x in v.split(",") if x.strip()])
    return out


def parse_axes(spec) -> dict:
    """--axis_pairs whole_wing=cell-cu2,cell-a>cell-r2,cell-Rs  (from > to)"""
    out = {}
    for item in (spec or []):
        if "=" not in str(item) or ">" not in str(item):
            continue
        k, v = str(item).split("=", 1)
        lo, hi = v.split(">", 1)
        out[k.strip()] = ([x.strip() for x in lo.split(",") if x.strip()],
                          [x.strip() for x in hi.split(",") if x.strip()])
    return out


def orient_by_neighbour(sdir: Path, struct: str, gpa_dir, polys: dict, pairs: dict,
                        axes: dict, cons: np.ndarray, u: np.ndarray, sid_of,
                        min_end_margin: float = 0.30, min_axis_cosine: float = 0.50,
                        max_fit_residual: float = 1e-3):
    """{sid: (turned, margin, cue)} for every frame whose photograph settles the question.

    `turned` is relative to the MAJORITY of this structure, exactly as the profile pass
    defines "as is": the neighbour says which end of the frame the joint is on, and the
    frames that put it on the other end are the ones out of step. Nothing is thresholded
    except the quality of the reading itself — a contact point in the middle of the frame
    (`margin`) or an anatomical direction across the long axis rather than along it
    (`cosine`) has not answered the question and is left to the profile pass."""
    contours = load_gpa_contours(gpa_dir, struct) if gpa_dir else {}
    if not contours or (struct not in pairs and struct not in axes):
        return {}
    by_sid = {}
    for fn, g in contours.items():
        s = sid_of(image_key(fn))
        if s:
            by_sid.setdefault(s, (fn, g))
    raw = {}
    for f in sorted(sdir.glob("*_landmarks.npy")):
        sid = f.name[: -len("_landmarks.npy")]
        hit = by_sid.get(sid)
        if hit is None:
            continue
        fn, g = hit
        lmf = np.load(f)
        if len(_ring_np(g)) != len(_ring_np(lmf)):
            continue
        _R, res = kabsch_rotation(_ring_np(g), _ring_np(lmf))
        if res > max_fit_residual:            # not the contour these landmarks came from
            continue
        got = polys.get(image_key(fn), {})
        best = None
        for nb in pairs.get(struct, []):
            if nb not in got:
                continue
            t = neighbour_end(g, lmf, densify(got[nb]), cons, u)
            m = abs(t - 0.5) * 2.0
            if best is None or m > best[1]:
                best = (1.0 if t > 0.5 else -1.0, m, f"neighbour:{nb}")
        if best is None and struct in axes:
            lo, hi = axes[struct]
            a = [got[c].mean(0) for c in lo if c in got and c != struct]
            b = [got[c].mean(0) for c in hi if c in got and c != struct]
            if a and b:
                c = axis_direction_cos(g, lmf, np.mean(b, 0) - np.mean(a, 0), u)
                best = (1.0 if c > 0 else -1.0, abs(c), "axis")
        if best is None:
            continue
        need = min_end_margin if best[2].startswith("neighbour") else min_axis_cosine
        if best[1] < need:
            continue
        raw[sid] = best
    if not raw:
        return {}
    side = np.array([v[0] for v in raw.values()])
    majority = 1.0 if (side > 0).sum() >= (side < 0).sum() else -1.0
    return {s: (int(v[0] != majority), float(v[1]), v[2]) for s, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 5. mirrored frames                                           (added 2026-09-21)
#
# A similarity transform cannot turn a left forewing into a right one, so a mirrored
# specimen is fitted by the best compromise available and lands in the frame with its two
# SIDES exchanged. Nothing above notices: the width profile along the long axis is
# unchanged by a reflection across that axis, and the picture correlation is measured only
# against the half turn. The frame is nevertheless wrong everywhere, and a report on it is
# read at the wrong place.
#
# The decision is not taken per structure. All the structures cut out of ONE photograph
# share one handedness, so the near-symmetric ones (whole_wing, pterostigma, cell-cu2)
# that cannot decide for themselves are carried by the vote of those that can — which is
# how a person would do it, and it is also testable: biorag_outline_shape_v1 registers
# every outline mirror-aware and reports a relative gap between the two fits, and the wing
# cells whose gap is wide recover an independent handedness (read from the cell centroids,
# not from any outline) at 98-100 %.
# ─────────────────────────────────────────────────────────────────────────────
def load_outline_registration(path):
    """(structure, specimen) -> (mirrored, ambiguous, relative gap) from
    biorag_outline_shape_v1.py's outline_registration.tsv."""
    t = pd.read_csv(path, sep="\t")
    out = {}
    for r in t.itertuples():
        # the registration files a sexed structure under two names ('subgenital_plate__male')
        # while the frames carry one; the sex is not a second structure
        st = str(r.structure)
        for suff in ("__male", "__female"):
            if st.endswith(suff):
                st = st[: -len(suff)]
        out[(st, str(r.specimen_id))] = (
            bool(str(getattr(r, "mirrored", "False")).lower() in ("true", "1")),
            bool(str(getattr(r, "ambiguous", "False")).lower() in ("true", "1")),
            float(getattr(r, "relative_gap", float("nan"))))
    return out


def mirror_vote(reg: dict, image_of: dict, reliable=None, min_votes: int = 1):
    """One handedness per PHOTOGRAPH, then that verdict for every structure on it.

    Only structures whose outline registration is unambiguous get a vote (and only those
    listed in `reliable`, when a list is given); the vote is by simple majority, and a tie
    leaves the image undecided rather than guessed at. Returns
    ({(structure, specimen): mirrored}, {image: (mirrored, votes_for, votes_total)})."""
    ballots = {}
    for (st, sid), (mir, amb, _gap) in reg.items():
        img = image_of.get((st, sid))
        if img is None or amb:
            continue
        if reliable and st not in reliable:
            continue
        ballots.setdefault(img, []).append(1 if mir else 0)
    verdict, per_image = {}, {}
    for img, v in ballots.items():
        if len(v) < min_votes:
            continue
        yes, n = int(sum(v)), len(v)
        if yes * 2 == n:
            continue                                    # a tie is not a decision
        per_image[img] = (yes * 2 > n, yes, n)
    for (st, sid) in reg:
        img = image_of.get((st, sid))
        if img in per_image:
            verdict[(st, sid)] = int(per_image[img][0])
    if verdict and sum(verdict.values()) * 2 > len(verdict):
        # the registration's sign is arbitrary; the majority of the frames is "as is"
        verdict = {k: 1 - v for k, v in verdict.items()}
        per_image = {k: (not v[0], v[2] - v[1], v[2]) for k, v in per_image.items()}
    return verdict, per_image


TRANSFORMS = ("identity", "halfturn", "flipx", "flipy", "mirror", "mirror_halfturn")


def apply_frame_transform(xy: np.ndarray, name: str, axis_deg: float = 0.0) -> np.ndarray:
    """A reported (x, y) on the unit frame, mapped onto the consensus anatomy.

    The four transforms are the symmetries a similarity fit can confuse: nothing
    (`identity`), the half turn about the frame centre (`halfturn`), and the two
    reflections — across the frame's long axis (`mirror`) and across the axis at right
    angles to it (`mirror_halfturn`, which is the same as a mirror followed by a half
    turn). `flipx`/`flipy` are the axis-aligned special cases, kept because they are what
    a reader expects to see written down when the long axis happens to lie along an axis
    of the frame."""
    p = np.asarray(xy, float).reshape(-1, 2) - 0.5
    n = str(name or "identity")
    if n == "identity":
        q = p
    elif n == "halfturn":
        q = -p
    elif n == "flipx":
        q = np.c_[-p[:, 0], p[:, 1]]
    elif n == "flipy":
        q = np.c_[p[:, 0], -p[:, 1]]
    elif n in ("mirror", "mirror_halfturn"):
        a = np.deg2rad(float(axis_deg) + (90.0 if n == "mirror_halfturn" else 0.0))
        c, s = np.cos(2 * a), np.sin(2 * a)
        q = p @ np.array([[c, s], [s, -c]])                     # reflection about that line
    else:
        raise ValueError(f"unknown frame transform {name!r}")
    return (q + 0.5).reshape(np.asarray(xy, float).shape)


def best_read_time_transform(lm_frame: np.ndarray, cons: np.ndarray, canvas: float,
                             axis_deg: float, candidates=None):
    """Which of the candidate transforms brings this frame's own outline onto the
    consensus outline, and by how much it beats the runner-up.

    The test is the mean distance from each transformed landmark to the nearest point of
    the consensus, in both directions, so it does not depend on the landmarks being in
    correspondence — they are not, once a reflection is in play: a mirror reverses the
    order in which a closed outline is traced."""
    c = _ring_np(cons) / float(canvas)
    scores = []
    for name in (candidates or TRANSFORMS):
        q = apply_frame_transform(_ring_np(lm_frame) / float(canvas), name, axis_deg)
        d = np.sqrt(((q[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        scores.append((0.5 * (d.min(1).mean() + d.min(0).mean()), name))
    scores.sort()
    return scores[0][1], float(scores[0][0]), float(scores[1][0] - scores[0][0])


def consensus_axis_degrees(cons: np.ndarray) -> float:
    u, _v = principal_axis(_ring_np(cons))
    return float(np.degrees(np.arctan2(u[1], u[0])) % 180.0)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Decide, per specimen, whether its homology frame sits half a turn out")
    ap.add_argument("--frames_dir", required=True, help="output of biorag_homology_frame_v1.py")
    ap.add_argument("--out", default=None,
                    help="TSV to write (default <frames_dir>/frame_orientation.tsv)")
    ap.add_argument("--structures", nargs="*", default=None)
    ap.add_argument("--bins", type=int, default=64, help="slices across the long axis")
    ap.add_argument("--min_confidence", type=float, default=0.05,
                    help="half the margin between the two profile scores; below this the "
                         "profile pass has not decided and the image pass is tried")
    ap.add_argument("--min_ncc_margin", type=float, default=0.02,
                    help="difference in correlation with the mean frame below which the "
                         "image pass has not decided either; such frames are left unflipped")
    ap.add_argument("--min_corroboration", type=float, default=0.60,
                    help="share of the frames the profile wants turned that the picture must "
                         "agree to turn. Below this the structure's frames were already "
                         "consistent and the apparent flips are noise in a weakly "
                         "end-asymmetric profile: every flip on that structure is withdrawn")
    ap.add_argument("--min_corroborated_flips", type=int, default=3,
                    help="how many profile flips a structure needs before that test is "
                         "meaningful; with fewer, the flips stand or fall one by one")
    ap.add_argument("--no_ncc", action="store_true", help="skip the image-content fallback")
    ap.add_argument("--ncc_px", type=int, default=192, help="frames are compared at this size")
    ap.add_argument("--min_elongation", type=float, default=2.0,
                    help="below this the structure has no two ends to confuse (reported only)")
    ap.add_argument("--max_end_asymmetry", type=float, default=0.30,
                    help="above this a half turn is plainly not a symmetry of the consensus "
                         "and the Procrustes fit could not have confused the two (reported only)")
    ap.add_argument("--min_specimens", type=int, default=6)
    # ── the anatomical adjacency cue (added 2026-09-21). Off unless asked for: without
    #    --coco and --gpa_dir every line below behaves exactly as it did before.
    ap.add_argument("--coco", default=None,
                    help="COCO file holding the NEIGHBOURING structures. With --gpa_dir this "
                         "turns on the adjacency cue, which is tried FIRST and needs no "
                         "threshold: the end of the tibia that touches the femur is proximal")
    ap.add_argument("--gpa_dir", default=None,
                    help="the semilandmark step's directory, the one biorag_homology_frame_v1 "
                         "was run with. Its back_transformed_coco.json holds the same landmarks "
                         "the frames were built from, in image pixels, so the frame transform "
                         "is recovered exactly rather than estimated")
    ap.add_argument("--neighbours", nargs="*", default=None,
                    help="structure=neighbour[,neighbour] pairs, e.g. tibia=femura LAB2=LAB1. "
                         "Read from the taxon profile key `orientation_neighbours` when that "
                         "is present and this is not given")
    ap.add_argument("--axis_pairs", nargs="*", default=None,
                    help="structure=base_structures>apex_structures, for structures no "
                         "neighbour touches end on: the forewing and its cells, whose base and "
                         "apex are given by the cells at either end of the wing")
    ap.add_argument("--taxon_profile", default=None,
                    help="used to read specimen ids out of image file names, and "
                         "`orientation_neighbours` / `orientation_axes` if they are in it")
    ap.add_argument("--matrix_dir", default=None,
                    help="compiled_key_tier, for the species codes specimen ids are built from")
    ap.add_argument("--min_end_margin", type=float, default=0.30,
                    help="how far from the middle of the frame the contact with the neighbour "
                         "must lie, as a fraction of the frame's length. A joint reported in "
                         "the middle has not said which end it is")
    ap.add_argument("--min_axis_cosine", type=float, default=0.50,
                    help="for --axis_pairs: how nearly the anatomical direction must lie along "
                         "the frame's long axis before it can say which end is which")
    # ── mirrored frames (added 2026-09-21)
    ap.add_argument("--outline_registration", default=None,
                    help="outline_registration.tsv from biorag_outline_shape_v1.py. When given, "
                         "`mirrored` and `read_time_transform` columns are added: a mirror image "
                         "cannot be fitted by a similarity transform, so such a frame has its "
                         "two sides exchanged and a report on it lands on the wrong one")
    ap.add_argument("--mirror_reliable", nargs="*", default=None,
                    help="structures whose outline registration may vote on an image's "
                         "handedness. Default: every structure with an unambiguous reading")
    ap.add_argument("--mirror_min_votes", type=int, default=1)
    a = ap.parse_args()

    frames = Path(a.frames_dir)
    out_f = Path(a.out) if a.out else frames / "frame_orientation.tsv"
    structs = sorted(d.name for d in frames.iterdir()
                     if d.is_dir() and (d / "consensus_px.npy").exists()
                     and (not a.structures or d.name in a.structures))
    if not structs:
        raise SystemExit(f"no structures with a consensus under {frames}")

    # ── adjacency and mirror set-up. Both stay empty unless the new options were given,
    #    and every line below is guarded on that, so the old run is reproduced exactly.
    pairs, axes, polys, sid_of = {}, {}, {}, None
    profile = None
    if a.taxon_profile:
        import biorag_feature_policy as _pol
        profile = _pol.load_taxon_profile(a.taxon_profile)
        pairs = parse_pairs(profile.get("orientation_neighbours") and
                            [f"{k}={','.join(v) if isinstance(v, list) else v}"
                             for k, v in profile["orientation_neighbours"].items()])
        axes = parse_axes(profile.get("orientation_axes") and
                          [f"{k}={','.join(v['from'])}>{','.join(v['to'])}"
                           for k, v in profile["orientation_axes"].items()])
    if a.neighbours:
        pairs = parse_pairs(a.neighbours)
    if a.axis_pairs:
        axes = parse_axes(a.axis_pairs)
    use_adjacency = bool(a.coco and a.gpa_dir and (pairs or axes))
    if use_adjacency:
        from biorag_specimen_id import specimen_of
        codes = []
        if a.matrix_dir:
            codes = sorted(set(pd.read_csv(Path(a.matrix_dir) / "specimen_matrix_long.csv")
                               ["species"].astype(str)))
        polys = load_coco_polygons(a.coco)
        sid_of = lambda fn: specimen_of(fn, profile or {}, codes)[1]          # noqa: E731
        print(f"adjacency cue: {len(pairs)} neighbour rule(s), {len(axes)} axis rule(s), "
              f"{len(polys)} images")

    mirror_of, mirror_images, image_of = {}, {}, {}
    if a.outline_registration:
        reg = load_outline_registration(a.outline_registration)
        if use_adjacency:
            for st in structs:
                for fn in load_gpa_contours(a.gpa_dir, st):
                    s = sid_of(image_key(fn))
                    if s:
                        image_of[(st, s)] = image_key(fn)
        reliable = a.mirror_reliable or ((profile or {}).get("orientation_mirror_voters"))
        mirror_of, mirror_images = mirror_vote(reg, image_of, reliable, a.mirror_min_votes)
        print(f"mirror vote: {len(mirror_images)} images decided, "
              f"{sum(1 for v in mirror_images.values() if v[0])} mirrored; "
              f"{len(mirror_of)} frames carried")

    rows, table = [], []
    for st in structs:
        sdir = frames / st
        cons = np.load(sdir / "consensus_px.npy")
        u, v = principal_axis(ring(cons))
        elong = elongation_of(ring(cons))
        easym = rotation_asymmetry(cons)
        at_risk = bool(elong >= a.min_elongation and easym <= a.max_end_asymmetry)
        axis_deg = consensus_axis_degrees(cons)      # the long axis, for the mirror columns

        prof = {}
        for f in sorted(sdir.glob("*_landmarks.npy")):
            sid = f.name[: -len("_landmarks.npy")]
            w = width_profile(np.load(f), u, v, a.bins)
            if w is None:
                continue
            wu = unit(w)
            if wu is not None:
                prof[sid] = wu
        if len(prof) < a.min_specimens:
            print(f"  {st:<20} only {len(prof)} usable outlines — skipped")
            continue

        scores, _ref = orient_by_profile(prof)
        dec, undec = {}, []
        rec = {}
        for sid, (s_as, s_rot) in scores.items():
            conf = abs(s_as - s_rot) / 2.0
            if conf >= a.min_confidence:
                flip = bool(s_rot > s_as)
                dec[sid] = flip
                rec[sid] = dict(flipped=int(flip), method="profile", score_as_is=s_as,
                                score_rotated=s_rot, confidence=conf)
            else:
                undec.append(sid)
                rec[sid] = dict(flipped=0, method="undecided", score_as_is=s_as,
                                score_rotated=s_rot, confidence=conf)

        n_ncc, n_conflict, agree, corrob = 0, 0, float("nan"), float("nan")
        if not a.no_ncc:
            img, agree = orient_by_image(sdir, sorted(rec), dec, a.ncc_px)
            # the picture's own verdict, positive when it wants the frame turned
            d = {sid: (c_rot - c_as) for sid, (c_as, c_rot) in img.items()}
            for sid in undec:                          # the picture decides what the outline could not
                if abs(d.get(sid, 0.0)) >= a.min_ncc_margin:
                    c_as, c_rot = img[sid]
                    rec[sid] = dict(flipped=int(d[sid] > 0), method="ncc", score_as_is=c_as,
                                    score_rotated=c_rot, confidence=abs(d[sid]) / 2.0)
                    n_ncc += 1
            # a half turn changes the outline AND the picture, so one cue saying so and the
            # other confidently saying the opposite is not a half turn. Withdraw, don't guess.
            for sid, flip in dec.items():
                if flip and d.get(sid, 0.0) <= -a.min_ncc_margin:
                    rec[sid].update(flipped=0, method="conflict")
                    n_conflict += 1

            # Does the picture confirm the frames the outline wants turned? A width profile
            # that is only weakly end-asymmetric produces a minority whose antisymmetric part
            # points the other way through noise alone, and flipping those manufactures a
            # mixture out of frames that were consistent — visible afterwards as a ghost of
            # the reversed structure in the mean frame. The reports cannot arbitrate: moving a
            # minority of them onto the majority always lowers the spread, whether or not the
            # frame was ever out. Picture content can, and it is independent of the outline.
            want = [s for s, f in dec.items() if f and s in d]
            if len(want) >= a.min_corroborated_flips:
                corrob = float(np.mean([d[s] >= a.min_ncc_margin for s in want]))
                if corrob < a.min_corroboration:
                    for sid, r in rec.items():
                        if r["flipped"]:
                            r.update(flipped=0, method="uncorroborated")
                    n_ncc = 0

        # ── the adjacency cue, taken FIRST where the photograph has it (2026-09-21).
        #    It overrides the profile and the picture rather than being averaged with them:
        #    it is an observation of this specimen, not a statistic over the sample, and it
        #    has no threshold to be wrong about. Two housekeeping points. (1) "as is" is a
        #    majority, and the profile pass and the adjacency pass each take their own; if
        #    they disagree the two halves of the table would contradict each other, so the
        #    profile's convention is turned round to match the anatomy — for DECIDED frames
        #    only, since an undecided frame carries flipped=0 meaning "not known", not
        #    "this way up". (2) the count of adjacency decisions is reported separately so
        #    the fall-back can be audited against it.
        n_adj, n_swap = 0, 0
        if use_adjacency:
            adj = orient_by_neighbour(sdir, st, a.gpa_dir, polys, pairs, axes,
                                      ring(cons), u, sid_of, a.min_end_margin,
                                      a.min_axis_cosine)
            both = [s for s in adj if s in rec and rec[s]["method"] in ("profile", "ncc")]
            if len(both) >= 4:
                agr = float(np.mean([adj[s][0] == rec[s]["flipped"] for s in both]))
                if agr < 0.5:
                    n_swap = sum(1 for s in rec if rec[s]["method"] in ("profile", "ncc"))
                    for s in rec:
                        if rec[s]["method"] in ("profile", "ncc"):
                            rec[s]["flipped"] = 1 - rec[s]["flipped"]
            for s, (turned, margin, cue) in adj.items():
                if s in rec:
                    rec[s].update(flipped=int(turned), method="neighbour",
                                  confidence=float(margin), neighbour_cue=cue)
                    n_adj += 1

        for sid in sorted(rec):
            r = rec[sid]
            row = {"structure": st, "specimen_id": sid, "flipped": r["flipped"],
                   "method": r["method"], "score_as_is": round(r["score_as_is"], 5),
                   "score_rotated": round(r["score_rotated"], 5),
                   "confidence": round(r["confidence"], 5),
                   "elongation": round(elong, 3), "end_asymmetry": round(easym, 4),
                   "at_risk": int(at_risk),
                   "picture_profile_agreement": (round(agree, 3) if agree == agree else ""),
                   "picture_corroboration": (round(corrob, 3) if corrob == corrob else "")}
            if use_adjacency:
                row["neighbour_cue"] = r.get("neighbour_cue", "")
            if a.outline_registration:
                mir = mirror_of.get((st, sid))
                row["mirrored"] = "" if mir is None else int(mir)
                row["mirror_axis_deg"] = round(axis_deg, 2)
                row["read_time_transform"] = (
                    ("mirror_halfturn" if r["flipped"] else "mirror") if mir
                    else ("halfturn" if r["flipped"] else "identity"))
            rows.append(row)
        n_flip = sum(r["flipped"] for r in rec.values())
        n_low = sum(r["method"] == "undecided" for r in rec.values())
        n_unc = sum(r["method"] == "uncorroborated" for r in rec.values())
        table.append({"structure": st, "n": len(rec), "elongation": round(elong, 2),
                      "end_asym": round(easym, 3), "at_risk": "yes" if at_risk else "no",
                      "flipped": n_flip, "by_ncc": n_ncc, "undecided": n_low,
                      "withdrawn": n_unc + n_conflict, "corrob": corrob})
        print(f"  {st:<20} n={len(rec):>4}  elong={elong:>5.2f}  end_asym={easym:.3f}  "
              f"{'AT RISK' if at_risk else '  —    '}  flipped={n_flip:>3} "
              f"(ncc {n_ncc})  undecided={n_low:>3}  withdrawn={n_unc + n_conflict:>3}"
              + (f"  corroboration {corrob:.2f}" if corrob == corrob else "")
              + (f"  | anatomy {n_adj}" + (f", convention turned round ({n_swap})"
                                           if n_swap else "") if use_adjacency else ""))

    df = pd.DataFrame(rows)
    out_f.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_f, sep="\t", index=False)

    t = pd.DataFrame(table)
    print("\n" + "=" * 96)
    print(f"{'structure':<20}{'n':>5}{'elong':>8}{'end_asym':>10}{'at risk':>9}"
          f"{'flipped':>9}{'by ncc':>8}{'undecided':>11}{'withdrawn':>11}{'corrob':>9}")
    print("-" * 96)
    for r in table:
        ag = f"{r['corrob']:.2f}" if r["corrob"] == r["corrob"] else "—"
        print(f"{r['structure']:<20}{r['n']:>5}{r['elongation']:>8.2f}{r['end_asym']:>10.3f}"
              f"{r['at_risk']:>9}{r['flipped']:>9}{r['by_ncc']:>8}{r['undecided']:>11}"
              f"{r['withdrawn']:>11}{ag:>9}")
    print("-" * 96)
    risk = t[t["at_risk"] == "yes"]
    ctrl = t[t["at_risk"] == "no"]
    print(f"{'AT RISK':<20}{int(risk['n'].sum()):>5}{'':>8}{'':>10}{'':>9}"
          f"{int(risk['flipped'].sum()):>9}{int(risk['by_ncc'].sum()):>8}"
          f"{int(risk['undecided'].sum()):>11}")
    print(f"{'control (not at risk)':<20}{int(ctrl['n'].sum()):>5}{'':>8}{'':>10}{'':>9}"
          f"{int(ctrl['flipped'].sum()):>9}{int(ctrl['by_ncc'].sum()):>8}"
          f"{int(ctrl['undecided'].sum()):>11}")
    print("=" * 96)
    print(json.dumps({"version": VERSION, "frames": int(len(df)),
                      "flipped": int(df["flipped"].sum()),
                      "undecided": int((df["method"] == "undecided").sum()),
                      "structures": int(df["structure"].nunique())}, indent=1))
    print(f"-> {out_f}")


if __name__ == "__main__":
    main()
