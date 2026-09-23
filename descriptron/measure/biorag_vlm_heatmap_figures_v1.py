#!/usr/bin/env python3
"""
biorag_vlm_heatmap_figures_v1.py — where on the structure did the model say it looked?
=====================================================================================

The scorer in `biorag_vlm_characters_v1.py` is asked, for every character, WHERE on the
image it saw the feature. Because every specimen was first placed in a common homologous
frame (`biorag_homology_frame_v1.py`: rotation, uniform scale and translation only, the
outline left alone), a reported (x, y) is an anatomical coordinate, and the reports of many
specimens can be pooled. `biorag_vlm_roi_figures_v1.py` draws those reports specimen by
specimen — the comparison a reader has to make to believe one character. This script draws
the other view: all reports of a character at once, as a density laid over the anatomy, so
that "is the model looking at the same place on every specimen, and is it the place the
character's name says" can be read off a single panel.

What is drawn
-------------
  background   the MEAN of the aligned frames of the structure, in greyscale. The frames
               share a frame, so their mean is meaningful: it shows the anatomy without
               privileging one specimen, and its soft edge shows how much the outline
               varies. The Procrustes consensus outline is drawn over it.
  heat map     a Gaussian-smoothed density of the reported points, scaled to its own
               maximum in each panel, in a rainbow colour map. A rainbow is used on purpose
               here: the background is a greyscale photograph, so a light-to-dark ramp
               would be confounded with the anatomy underneath it, whereas hue is not.
               Opacity rises with density, so empty regions stay transparent.
  dots         the individual reports, so the n behind the colour is visible. Three reports
               and three hundred make the same red blob once each is scaled to its maximum.

Three cases look identical in a table of states and different here (the localisation audit):

  tight and off-centre   a genuinely localised character — the reports concentrate on one
                         part of the structure, away from the middle of the frame;
  tight and centred      a whole-structure character (an outline, an overall proportion):
                         there is no local place to point, so the model points at the
                         middle. Legitimate, and its name says which it is;
  scattered              for a character named after a LOCAL feature, the signature of a
                         scorer that has not localised anything.

A large spread has a second cause that is not a failure of the scorer, and it is flagged
separately (`two_ended`): reports divided between the two ENDS of the structure's long axis
with little in between. Either the feature really occurs at both ends, or the frames of an
elongate, nearly two-fold-symmetric outline sit in two orientations half a turn apart — a
Procrustes fit without reflection cannot tell the ends of such an outline apart — and the
model has followed the anatomy to whichever end it is on. The summary figure prefers a true
scatter as its "scattered" example and labels a two-ended character as what it is.

The second of those two causes is no longer left standing: `biorag_frame_orientation_v1.py`
decides, per specimen, whether its frame is the other way up — from the width profile along
the long axis, which reverses under a half turn while the Procrustes residual barely moves,
and from picture content where the outline cannot say — and `--orientation` applies that
table here. A report on a flipped frame is mapped to (1 - x, 1 - y), and the frame and its
outline are turned to match, so a character that is two-ended only because its frames were
collapses to one place and one that is two-ended because the reports really go to both ends
stays as it is. That is the distinction the flag is for: after it, `two_ended` means the
reports, not the frames.

A small spread has a second cause too, and it is the more dangerous one because it looks
like success: a STEREOTYPED answer. Asked where the apex of a slender structure is, a model
may answer "top, middle" for every specimen; the reports are then tightly clustered and well
off-centre, and lie beside the structure rather than on it. Spread and offset cannot see
this. A third statistic can: the share of reports that fall outside the specimen's OWN
outline (its landmarks in the frame), beyond a tolerance that still admits a feature on the
margin. It is written for every character, characters whose reports are mostly off the
specimen are never offered as examples of localisation, and the summary figure shows the
worst of them as a case of its own.

The two statistics are those of the audit and of `biorag_vlm_roi_figures_v1.py`:
spread = mean distance of the reports from their own centroid, offset = distance of that
centroid from the centre of the frame, both in frame units (0-1).

Output
------
  atlas__<structure>.png/.pdf    first panel: every character of the structure pooled
                                 ("where does the model look on this structure at all");
                                 then one panel per character with >= --min_reports.
  summary_vlm_attention.png/.pdf the cases side by side, chosen from the statistics (never
                                 by hand): the tightest off-centre characters whose reports
                                 lie on the specimen, one per structure; a whole-structure
                                 character; the tight character lying furthest off the
                                 specimen; a high-spread one; and a two-ended one.
  vlm_heatmap_index.tsv          one row per character: n, spread, offset, category, the
                                 share of reports falling outside the specimen's own
                                 outline, the figure it is in, and its role in the summary.

  python biorag_vlm_heatmap_figures_v1.py --states "$M/vlm_combined/vlm_character_states.tsv" \\
      --frames_dir "$M/homology_frames" --out_dir "$M/vlm_combined/figures_heatmap" \\
      [--orientation "$M/homology_frames/frame_orientation.tsv"]
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.cm import ScalarMappable                                 # noqa: E402
from matplotlib.colors import Normalize                                  # noqa: E402
from matplotlib.path import Path as MplPath                              # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
from PIL import Image                                                    # noqa: E402

VERSION = "1.0"
Image.MAX_IMAGE_PIXELS = None
INK, INK2, RULE = "#263238", "#546e7a", "#cfd8dc"        # text and frame: never the data colour
POOLED = "(all characters pooled)"
ROLE_LABEL = {"localised": "LOCALISED  ·  tight, off-centre, on the specimen",
              "whole_structure": "WHOLE STRUCTURE  ·  tight, centred",
              "scattered": "HIGH SPREAD  ·  reports not in one place",
              "two_ended": "TWO-ENDED  ·  reports at both ends",
              "off_specimen": "TIGHT BUT OFF THE SPECIMEN  ·  a stereotyped answer"}


# ─────────────────────────────────────────────────────────────────────────────
# statistics
# ─────────────────────────────────────────────────────────────────────────────
def localisation(p: np.ndarray):
    """(spread, offset) exactly as the localisation audit computed them — replicated from
    biorag_vlm_roi_figures_v1.py (heat() and the index rows), where they exist only inline:
    spread = mean distance of the reports from their centroid, offset = distance of the
    centroid from the centre of the frame (0.5, 0.5)."""
    if not len(p):
        return math_nan(), math_nan()
    spread = float(np.hypot(*(p - p.mean(0)).T).mean())
    offset = float(np.hypot(*(p.mean(0) - 0.5)))
    return spread, offset


def math_nan():
    return float("nan")


def two_ended(p: np.ndarray, cons01):
    """Are the reports split between the two ENDS of the structure's long axis, with little in
    between? That pattern inflates the spread exactly as a scatter does but means something
    else: either the feature really occurs at both ends (articulations), or the frames of an
    elongate, nearly two-fold-symmetric structure sit in two orientations half a turn apart and
    the model has followed the anatomy to whichever end it is on. Neither is a scorer that
    failed to localise, so the two must not be read as one. Returns (flag, share at the
    less-reported end)."""
    if cons01 is None or len(p) < 6:
        return False, math_nan()
    c = cons01.mean(0)
    u = np.linalg.svd(cons01 - c, full_matrices=False)[2][0]
    half = float(np.abs((cons01 - c) @ u).max()) or 1.0
    t = ((p - c) @ u) / half                                  # -1 .. 1 along the long axis
    lo, hi, mid = float((t < -0.5).mean()), float((t > 0.5).mean()), float((np.abs(t) <= 0.5).mean())
    return bool(min(lo, hi) >= 0.20 and mid <= 0.25), min(lo, hi)


def categorise(n, spread, offset, a):
    if n < a.min_reports:
        return "too_few_reports"
    if spread >= a.scattered:
        return "scattered"
    if spread < a.tight and offset > a.offcentre:
        return "tight_offcentre"
    if spread < a.tight and offset < a.centred:
        return "tight_centred"
    return "intermediate"


# ─────────────────────────────────────────────────────────────────────────────
# frames
# ─────────────────────────────────────────────────────────────────────────────
def load_orientation(path):
    """The half-turn table from biorag_frame_orientation_v1.py, if one was given.

    An elongate, nearly half-turn-symmetric outline fits its own Procrustes consensus about
    as well upside down as the right way up, so the frames of such a structure sit in two
    orientations and a report follows the anatomy to whichever end it is on. That inflates
    the spread of a perfectly well localised character. The table says which frames are the
    other way up; nothing here rewrites them, the half turn is applied as the reports, the
    background and the outlines are read. Returns {(structure, specimen_id): flipped} and
    the per-structure counts."""
    if not path:
        return {}, {}, {}
    t = pd.read_csv(path, sep="\t")
    flip = {(r.structure, r.specimen_id): bool(int(r.flipped)) for r in t.itertuples()}
    nfl = {s: int(g["flipped"].sum()) for s, g in t.groupby("structure")}
    nlo = {s: int((g["method"] == "undecided").sum()) for s, g in t.groupby("structure")}
    return flip, nfl, nlo


# ── mirrored frames (added 2026-09-21) ───────────────────────────────────────
# A half turn is not the only symmetry a similarity fit can confuse. A specimen that is a
# MIRROR image of the consensus cannot be fitted by a similarity transform at all — the
# fit forbids reflection, because a left structure is not a right one — so its frame comes
# out with its two SIDES exchanged, and a report on it lands on the wrong side of the
# anatomy however well the half turn was decided. `biorag_frame_orientation_v1.py`, given
# an outline registration, adds two columns saying so. A table without them is read
# exactly as before: `read_time_transform` then defaults to the half turn or to nothing,
# which is what this script has always applied.
def load_transforms(path):
    """{(structure, specimen_id): (transform name, mirror axis in degrees)}.

    Older tables carry only `flipped`; they map to `identity`/`halfturn`, which is the
    behaviour this script had before the column existed."""
    if not path:
        return {}, {}
    t = pd.read_csv(path, sep="\t")
    has = "read_time_transform" in t.columns
    tr, counts = {}, {}
    for r in t.itertuples():
        name = (str(getattr(r, "read_time_transform", "") or "") if has else "")
        if not name or name == "nan":
            name = "halfturn" if int(r.flipped) else "identity"
        ang = float(getattr(r, "mirror_axis_deg", 0.0) or 0.0) if has else 0.0
        if ang != ang:
            ang = 0.0
        tr[(r.structure, r.specimen_id)] = (name, ang)
        counts.setdefault(r.structure, {}).setdefault(name, 0)
        counts[r.structure][name] += 1
    return tr, counts


def _reflection(axis_deg: float) -> np.ndarray:
    a = np.deg2rad(float(axis_deg))
    c, s = np.cos(2 * a), np.sin(2 * a)
    return np.array([[c, s], [s, -c]])


def transform_unit(xy, name: str, axis_deg: float = 0.0):
    """A point on the unit frame, mapped onto the consensus anatomy."""
    p = np.asarray(xy, float).reshape(-1, 2) - 0.5
    if name == "identity":
        q = p
    elif name == "halfturn":
        q = -p
    elif name == "flipx":
        q = np.c_[-p[:, 0], p[:, 1]]
    elif name == "flipy":
        q = np.c_[p[:, 0], -p[:, 1]]
    elif name == "mirror":
        q = p @ _reflection(axis_deg)
    elif name == "mirror_halfturn":
        q = p @ _reflection(axis_deg + 90.0)
    else:
        q = p
    return (q + 0.5).reshape(np.asarray(xy, float).shape)


def transform_image(v: np.ndarray, name: str, axis_deg: float = 0.0) -> np.ndarray:
    """The same map applied to a square frame. Every one of these transforms is its own
    inverse, so the forward matrix also serves as the inverse map PIL needs."""
    if name in ("identity", "", None):
        return v
    if name == "halfturn":
        return v[::-1, ::-1]
    if name == "flipx":
        return v[:, ::-1]
    if name == "flipy":
        return v[::-1, :]
    Mx = _reflection(axis_deg + (90.0 if name == "mirror_halfturn" else 0.0))
    n = v.shape[0]
    c = (n - 1) / 2.0
    coeff = (Mx[0, 0], Mx[0, 1], c - Mx[0, 0] * c - Mx[0, 1] * c,
             Mx[1, 0], Mx[1, 1], c - Mx[1, 0] * c - Mx[1, 1] * c)
    im = Image.fromarray(np.asarray(v, np.float32), mode="F")
    return np.asarray(im.transform((v.shape[1], n), Image.AFFINE, coeff,
                                   resample=Image.BILINEAR), np.float32)


def flipped_of(flip: dict, struct: str) -> set:
    return {sid for (s, sid), f in flip.items() if f and s == struct}


def transforms_of(tr: dict, struct: str) -> dict:
    return {sid: v for (s, sid), v in tr.items() if s == struct and v[0] != "identity"}


def mean_frame(sdir: Path, sids, px: int, flipped: set = None, tmap: dict = None):
    """Greyscale mean of the aligned frames (0-255), and how many went into it."""
    acc, n = None, 0
    for sid in sids:
        f = sdir / f"{sid}.png"
        if not f.exists():
            continue
        im = Image.open(f).convert("L")
        if px and im.width != px:
            im = im.resize((px, px), Image.BILINEAR)
        v = np.asarray(im, np.float32)
        if tmap is not None:
            if sid in tmap:
                v = transform_image(v, tmap[sid][0], tmap[sid][1])
        elif flipped and sid in flipped:
            v = v[::-1, ::-1]                  # the same half turn applied to the reports
        acc = v if acc is None else acc + v
        n += 1
    return (acc / n if n else None), n


def own_outline_miss(sdir: Path, rows: pd.DataFrame, canvas: int, tol: float,
                     flipped: set = None, tmap: dict = None):
    """Share of reports that fall outside the specimen's OWN outline (its landmarks in the
    frame): beyond `tol` of the frame, which still allows a feature reported on the margin
    itself, and beyond 3 x `tol`, which does not. The specimen's outline is used, not the
    consensus: a report on the margin of a broad individual lies outside the average outline
    without being wrong. Returns (share beyond tol, share beyond 3 tol, reports tested)."""
    def inside(path, xy, r):            # +r and -r: whichever way the ring is wound, one of them expands it
        return path.contains_point(xy, radius=r) or path.contains_point(xy, radius=-r)
    near = far = tested = 0
    cache = {}
    for sid, x, y in zip(rows["specimen_id"], rows["x"], rows["y"]):
        if sid not in cache:
            f = sdir / f"{sid}_landmarks.npy"
            if f.exists():
                lm = np.load(f)
                if tmap is not None:
                    if sid in tmap:
                        lm = transform_unit(lm / float(canvas), tmap[sid][0],
                                            tmap[sid][1]) * float(canvas)
                elif flipped and sid in flipped:
                    lm = float(canvas) - lm          # half turn about the frame centre
                cache[sid] = MplPath(lm / float(canvas))
            else:
                cache[sid] = None
        path = cache[sid]
        if path is None:
            continue
        tested += 1
        if not inside(path, (x, y), tol):
            near += 1
            if not inside(path, (x, y), 3 * tol):
                far += 1
    if not tested:
        return math_nan(), math_nan(), 0
    return near / tested, far / tested, tested


# ─────────────────────────────────────────────────────────────────────────────
# density
# ─────────────────────────────────────────────────────────────────────────────
def smooth(z: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(z, sigma=sigma, mode="constant")
    except Exception:                                        # separable fallback, no scipy
        r = max(1, int(round(3 * sigma)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        z = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, z)
        return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, z)


def density(p: np.ndarray, grid: int, bandwidth: float) -> np.ndarray:
    """Gaussian-smoothed 2-D histogram on the unit frame, rows = y (down), scaled to max 1."""
    z, _, _ = np.histogram2d(p[:, 1], p[:, 0], bins=grid, range=[[0, 1], [0, 1]])
    z = smooth(z, bandwidth * grid)
    return z / z.max() if z.max() > 0 else z


def rgba(z: np.ndarray, cmap, alpha: float, gamma: float, floor: float) -> np.ndarray:
    """Colour by density; opacity rises with density so the anatomy shows through where
    nothing was reported."""
    out = cmap(z)
    out[..., 3] = np.where(z < floor, 0.0, alpha * np.clip(z, 0, 1) ** gamma)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# drawing
# ─────────────────────────────────────────────────────────────────────────────
def view_of(cons01, pts=None, pad=0.07, square=False, min_aspect=0.45):
    """A window on the structure: the consensus bounding box, widened to hold every report,
    so a report that lies off the structure is shown rather than cropped away. Returns the
    limits (x0, x1, y_bottom, y_top) and the height/width of the window."""
    lo, hi = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    if cons01 is not None and len(cons01):
        lo, hi = cons01.min(0) - pad, cons01.max(0) + pad
    if pts is not None and len(pts):
        lo, hi = np.minimum(lo, pts.min(0) - 0.02), np.maximum(hi, pts.max(0) + 0.02)
    c, (w, h) = (lo + hi) / 2, (hi - lo)
    if square:
        w = h = max(w, h)
    else:                                   # never a sliver: a very long structure keeps some air
        h = max(h, min_aspect * w)
        w = max(w, min_aspect * h)
    return (c[0] - w / 2, c[0] + w / 2, c[1] + h / 2, c[1] - h / 2), float(h / w)


def layout(n, ncols, aspect, title_in, header_in=0.80, pw=3.0, gap=0.26, left=0.30, right=1.20,
           bottom=0.22, vgap=0.14):
    """Panels placed in inches, so a two-line title can never run into the row above it."""
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    ph = pw * aspect
    W = left + ncols * pw + (ncols - 1) * gap + right
    H = header_in + nrows * (title_in + ph) + (nrows - 1) * vgap + bottom
    fig = plt.figure(figsize=(W, H))
    axes = []
    for k in range(n):
        r, c = divmod(k, ncols)
        x0 = left + c * (pw + gap)
        top = H - header_in - r * (title_in + ph + vgap) - title_in
        axes.append(fig.add_axes([x0 / W, (top - ph) / H, pw / W, ph / H]))
    cb_h = min(ph, 2.6)
    top0 = H - header_in - title_in
    cax = fig.add_axes([(left + ncols * pw + (ncols - 1) * gap + 0.28) / W, (top0 - cb_h) / H,
                        0.15 / W, cb_h / H])
    return fig, axes, cax, W, H


def header(fig, W, H, title, sub):
    fig.text(0.30 / W, 1 - 0.30 / H, title, fontsize=11, color=INK, weight="bold", ha="left", va="center")
    fig.text(0.30 / W, 1 - 0.56 / H, sub, fontsize=7.3, color=INK2, ha="left", va="center")


def panel(ax, bg, cons01, pts, a, cmap, view, title, sub, dot=7.0, weight="normal", head=None):
    ax.set_facecolor("white")
    if bg is not None:
        # lifted towards white so the heat map, not the photograph, carries the contrast
        ax.imshow(255 - (255 - bg) * a.bg_strength, cmap="gray", vmin=0, vmax=255,
                  extent=(0, 1, 1, 0), origin="upper", interpolation="bilinear", zorder=1)
    if len(pts):
        z = density(pts, a.grid, a.bandwidth)
        ax.imshow(rgba(z, cmap, a.alpha, a.gamma, a.floor), extent=(0, 1, 1, 0), origin="upper",
                  interpolation="bilinear", zorder=2)
    if cons01 is not None:
        ring = np.vstack([cons01, cons01[:1]])
        ax.plot(ring[:, 0], ring[:, 1], color="white", lw=1.6, alpha=0.9, zorder=3)
        ax.plot(ring[:, 0], ring[:, 1], color=INK, lw=0.7, zorder=4)
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=dot, facecolor="#111111", edgecolor="white",
                   linewidth=0.35, alpha=0.9, zorder=5)
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(RULE)
        sp.set_linewidth(0.8)
    note = dict(xy=(0, 1), xycoords="axes fraction", textcoords="offset points", ha="left", va="bottom")
    ax.annotate(sub, xytext=(0, 3), fontsize=6.3, color=INK2, **note)
    ax.annotate(title, xytext=(0, 12.5), fontsize=7.6, color=INK, weight=weight, linespacing=1.15, **note)
    if head:
        ax.annotate(head, xytext=(0, 35), fontsize=6.5, color=INK2, weight="bold", **note)


def colourbar(fig, cax, cmap):
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax)
    cb.set_label("relative density of reports\n(each panel scaled to its own maximum)",
                 fontsize=7, color=INK2)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["none", "", "most"])
    cb.ax.tick_params(labelsize=6.5, colors=INK2, length=2)
    cb.outline.set_edgecolor(RULE)


def wrap(s, width=38):
    return "\n".join(textwrap.wrap(s, width)[:2])


def stat_line(n, spread, offset, cat=None):
    s = f"n = {n}   spread {spread:.3f}   offset {offset:.3f}"
    return s + (f"   {cat.replace('_', ' ')}" if cat else "")


def save(fig, stem: Path, dpi: int):
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Rainbow heat maps of where the model reported "
                                             "seeing each character, over the mean aligned frame")
    ap.add_argument("--states", required=True,
                    help="vlm_character_states.tsv (specimen_id, character 'structure:name', state, x, y)")
    ap.add_argument("--frames_dir", required=True, help="output of biorag_homology_frame_v1.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--orientation", default=None,
                    help="frame_orientation.tsv from biorag_frame_orientation_v1.py. When "
                         "given, a frame marked flipped is turned half a turn before anything "
                         "is read from it — the report (x, y) -> (1 - x, 1 - y), the frame in "
                         "the mean background, and the outline in the off-the-specimen test — "
                         "so that the spread of a character measures where the model looked "
                         "and not which way up the frame happened to come out of the "
                         "Procrustes fit. Omitted, nothing changes")
    ap.add_argument("--structures", nargs="*", default=None)
    ap.add_argument("--state", default="present", choices=["present", "absent", "both"],
                    help="which reports to draw. A location means 'where I saw it' only for "
                         "PRESENT; for ABSENT it is merely where the model looked")
    ap.add_argument("--min_reports", type=int, default=4)
    ap.add_argument("--cmap", default="turbo", help="turbo (default), jet, or any matplotlib map")
    ap.add_argument("--alpha", type=float, default=0.82, help="opacity of the densest region")
    ap.add_argument("--gamma", type=float, default=0.55, help="how fast opacity rises with density")
    ap.add_argument("--floor", type=float, default=0.03, help="densities below this stay transparent")
    ap.add_argument("--bandwidth", type=float, default=0.035,
                    help="Gaussian smoothing, as a fraction of the frame")
    ap.add_argument("--grid", type=int, default=256, help="density raster, cells per side")
    ap.add_argument("--bg_px", type=int, default=512, help="mean frame is resampled to this size")
    ap.add_argument("--bg_strength", type=float, default=0.75,
                    help="contrast of the mean frame, 1 = as photographed, 0 = blank")
    ap.add_argument("--tight", type=float, default=0.10, help="spread below this is 'tight'")
    ap.add_argument("--offcentre", type=float, default=0.15, help="offset above this is 'off-centre'")
    ap.add_argument("--centred", type=float, default=0.08, help="offset below this is 'centred'")
    ap.add_argument("--scattered", type=float, default=0.20, help="spread at or above this is 'scattered'")
    ap.add_argument("--outside_tol", type=float, default=0.02,
                    help="a report this close to the specimen's outline still counts as on it")
    ap.add_argument("--summary_localised", type=int, default=4)
    ap.add_argument("--off_specimen", type=float, default=0.25,
                    help="a character with more than this share of its reports well outside the "
                         "specimen's own outline (beyond 3 x --outside_tol) is 'off the specimen'")
    ap.add_argument("--summary_min_reports", type=int, default=10)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=250)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = Path(a.frames_dir)
    cmap = matplotlib.colormaps[a.cmap]

    canvas = 768
    sj = frames / "homology_frames_summary.json"
    if sj.exists():
        canvas = int(json.loads(sj.read_text()).get("canvas", canvas))

    st = pd.read_csv(a.states, sep="\t")
    if "x" not in st.columns or "y" not in st.columns:
        raise SystemExit("the states table carries no reported locations (columns x, y)")
    st["structure"] = st["character"].str.split(":", n=1).str[0]
    st["name"] = st["character"].str.split(":", n=1).str[1]
    n_all = len(st)
    if a.state != "both":
        st = st[st["state"] == a.state]
    st = st.dropna(subset=["x", "y"])
    st = st[st["x"].between(0, 1) & st["y"].between(0, 1)].copy()
    print(f"{len(st)} of {n_all} state assignments carry a usable location (state = {a.state})")

    flip_map, n_flipped, n_lowconf = load_orientation(a.orientation)
    tr_map, tr_counts = load_transforms(a.orientation)
    # a table carrying `read_time_transform` may ask for a reflection as well as a half
    # turn; one that does not resolves to exactly the half turn applied here before
    mirrored_table = any(v[0] in ("mirror", "mirror_halfturn", "flipx", "flipy")
                         for v in tr_map.values())
    if flip_map:
        if mirrored_table:
            keys = list(zip(st["structure"], st["specimen_id"]))
            moved = 0
            xy = st[["x", "y"]].to_numpy(float)
            for i, k in enumerate(keys):
                nm, ang = tr_map.get(k, ("identity", 0.0))
                if nm != "identity":
                    xy[i] = transform_unit(xy[i], nm, ang)
                    moved += 1
            st.loc[:, "x"], st.loc[:, "y"] = xy[:, 0], xy[:, 1]
            nm_all = {}
            for v in tr_map.values():
                nm_all[v[0]] = nm_all.get(v[0], 0) + 1
            print(f"orientation: {len(tr_map)} frames, "
                  + ", ".join(f"{k} {v}" for k, v in sorted(nm_all.items()))
                  + f"; {moved} of {len(st)} reports moved with them")
        else:
            fl = np.array([flip_map.get((s, i), False)
                           for s, i in zip(st["structure"], st["specimen_id"])])
            st.loc[fl, "x"] = 1.0 - st.loc[fl, "x"]
            st.loc[fl, "y"] = 1.0 - st.loc[fl, "y"]
            print(f"orientation: {sum(n_flipped.values())} frames of "
                  f"{len(flip_map)} turned half a turn ({sum(n_lowconf.values())} undecided, left "
                  f"as they are); {int(fl.sum())} of {len(st)} reports moved with them")

    structures = [s for s in sorted(st["structure"].unique())
                  if (not a.structures or s in a.structures)]
    index, store = [], {}

    for struct in structures:
        sdir = frames / struct
        g = st[st["structure"] == struct]
        if not sdir.is_dir():
            print(f"  {struct}: no frames — skipped (reports are only comparable in a common frame)")
            continue
        sids = sorted(p.stem for p in sdir.glob("*.png") if not p.name.startswith("_"))
        fset = flipped_of(flip_map, struct) if flip_map else None
        tset = transforms_of(tr_map, struct) if mirrored_table else None
        bg, n_bg = mean_frame(sdir, sids, a.bg_px, fset, tset)
        cf = sdir / "consensus_px.npy"
        cons01 = np.load(cf) / float(canvas) if cf.exists() else None
        store[struct] = (bg, cons01)

        rows = []
        for name, gc in g.groupby("name", sort=False):
            p = gc[["x", "y"]].values.astype(float)
            spread, offset = localisation(p)
            miss, far, _ = own_outline_miss(sdir, gc, canvas, a.outside_tol, fset, tset)
            ends, minor = two_ended(p, cons01)
            rows.append({"structure": struct, "character": name, "n_reports": len(p),
                         "two_ended": ends, "minor_end_share": minor,
                         "spread": spread, "offset": offset,
                         "category": categorise(len(p), spread, offset, a),
                         "share_outside_own_outline": miss, "share_far_outside": far, "_pts": p})
        rows.sort(key=lambda r: (r["category"] == "too_few_reports", r["spread"]))
        drawn = [r for r in rows if r["category"] != "too_few_reports"]

        pall = g[["x", "y"]].values.astype(float)
        view, aspect = view_of(cons01, pall)
        fig, axes, cax, W, H = layout(1 + len(drawn), a.ncols, aspect, title_in=0.50)
        sp_all, off_all = localisation(pall)
        panel(axes[0], bg, cons01, pall, a, cmap, view, "ALL CHARACTERS POOLED",
              stat_line(len(pall), sp_all, off_all), dot=3.0, weight="bold")
        for ax, r in zip(axes[1:], drawn):
            panel(ax, bg, cons01, r["_pts"], a, cmap, view, wrap(r["character"]),
                  stat_line(r["n_reports"], r["spread"], r["offset"],
                            r["category"] + (", two-ended" if r["two_ended"] else "")))
        colourbar(fig, cax, cmap)
        header(fig, W, H, f"{struct} — where the model reported seeing its characters",
               f"grey: mean of {n_bg} aligned specimens with the Procrustes consensus outline   ·   colour: "
               f"density of reported locations ({a.state})   ·   dots: the individual reports")
        stem = out / f"atlas__{struct}"
        save(fig, stem, a.dpi)

        index.append({"structure": struct, "character": POOLED, "n_reports": len(pall),
                      "spread": round(sp_all, 4), "offset": round(off_all, 4), "category": "pooled",
                      "share_outside_own_outline": None, "share_far_outside": None,
                      "two_ended": None, "minor_end_share": None,
                      "figure": stem.name + ".png", "summary_role": ""})
        for r in rows:
            index.append({"structure": struct, "character": r["character"], "n_reports": r["n_reports"],
                          "spread": round(r["spread"], 4), "offset": round(r["offset"], 4),
                          "category": r["category"],
                          "share_outside_own_outline": (round(r["share_outside_own_outline"], 3)
                                                        if r["share_outside_own_outline"] == r["share_outside_own_outline"]
                                                        else None),
                          "share_far_outside": (round(r["share_far_outside"], 3)
                                                if r["share_far_outside"] == r["share_far_outside"] else None),
                          "two_ended": bool(r["two_ended"]),
                          "minor_end_share": (round(r["minor_end_share"], 3)
                                              if r["minor_end_share"] == r["minor_end_share"] else None),
                          "figure": stem.name + ".png" if r["category"] != "too_few_reports" else "",
                          "summary_role": "", "_pts": r["_pts"]})
        print(f"  {struct}: {len(drawn)} characters drawn, {len(pall)} reports, background = {n_bg} frames")

    idx = pd.DataFrame(index)
    if not len(idx):
        raise SystemExit("nothing to draw")

    # ── the three cases, chosen from the statistics ──────────────────────────
    ch = idx[(idx["character"] != POOLED) & (idx["n_reports"] >= a.summary_min_reports)]
    on = ch["share_far_outside"].fillna(0) <= a.off_specimen
    loc = (ch[(ch["category"] == "tight_offcentre") & on].sort_values("spread")
           .drop_duplicates("structure").head(a.summary_localised))
    offsp = (ch[(ch["category"] == "tight_offcentre") & ~on]
             .sort_values(["share_far_outside", "n_reports"], ascending=False).head(1))
    whole = ch[ch["category"] == "tight_centred"].sort_values("spread").head(1)
    sc = ch[ch["category"] == "scattered"].sort_values("spread", ascending=False)
    scat = sc[sc["two_ended"] != True].head(1)                               # noqa: E712
    ends = sc[sc["two_ended"] == True].head(1)                               # noqa: E712
    chosen = [("localised", i) for i in loc.index] + [("whole_structure", i) for i in whole.index] \
             + [("off_specimen", i) for i in offsp.index] \
             + [("scattered", i) for i in scat.index] + [("two_ended", i) for i in ends.index]
    if chosen:
        for role, i in chosen:
            idx.loc[i, "summary_role"] = role
        fig, axes, cax, W, H = layout(len(chosen), 4, 1.0, title_in=0.68, header_in=0.85)
        for k, ((role, i), ax) in enumerate(zip(chosen, axes)):
            r = idx.loc[i]
            bg, cons01 = store[r["structure"]]
            view, _ = view_of(cons01, r["_pts"], square=True)
            panel(ax, bg, cons01, r["_pts"], a, cmap, view, wrap(f"{r['structure']}: {r['character']}"),
                  stat_line(int(r["n_reports"]), r["spread"], r["offset"]),
                  head=f"{'abcdefghijklmnop'[k]}   {ROLE_LABEL[role]}")
        colourbar(fig, cax, cmap)
        header(fig, W, H, "Where the model says it looked: cases that a table of states cannot tell apart",
               "grey: mean of the aligned specimens with the consensus outline   ·   colour: density of reported "
               "locations   ·   dots: individual reports   ·   spread and offset in frame units (0–1)")
        save(fig, out / "summary_vlm_attention", a.dpi)

    cols = ["structure", "character", "n_reports", "spread", "offset", "category",
            "share_outside_own_outline", "share_far_outside", "two_ended", "minor_end_share",
            "figure", "summary_role"]
    idx[cols].to_csv(out / "vlm_heatmap_index.tsv", sep="\t", index=False)

    ch = idx[idx["character"] != POOLED]
    cats = ch["category"].value_counts().to_dict()
    ok = ch[ch["category"] != "too_few_reports"]
    off = ok[ok["share_far_outside"].fillna(0) > a.off_specimen]
    summary = {"version": VERSION, "state_drawn": a.state, "canvas": canvas,
               "structures": int(ch["structure"].nunique()), "characters": int(len(ch)),
               "characters_drawn": int(len(ok)), "reports": int(ok["n_reports"].sum()),
               "median_spread": round(float(ok["spread"].median()), 4),
               "median_offset": round(float(ok["offset"].median()), 4),
               "categories": {k: int(v) for k, v in cats.items()},
               "scattered_but_two_ended": [f"{r.structure}:{r.character} (n={r.n_reports})" for r in
                                           ok[(ok["category"] == "scattered") & (ok["two_ended"] == True)].itertuples()],  # noqa: E712
               "thresholds": {"tight": a.tight, "offcentre": a.offcentre, "centred": a.centred,
                              "scattered": a.scattered, "min_reports": a.min_reports,
                              "outside_tol": a.outside_tol, "off_specimen": a.off_specimen},
               "reports_outside_own_outline": {
                   "beyond_tol": round(float((ok["share_outside_own_outline"] * ok["n_reports"]).sum()
                                             / max(1, ok["n_reports"].sum())), 3),
                   "beyond_3_tol": round(float((ok["share_far_outside"] * ok["n_reports"]).sum()
                                               / max(1, ok["n_reports"].sum())), 3)},
               "tight_offcentre_and_on_the_specimen": int(((ok["category"] == "tight_offcentre")
                                                           & (ok["share_far_outside"].fillna(0) <= a.off_specimen)).sum()),
               "characters_with_over_a_quarter_of_reports_well_off_the_specimen":
                   [f"{r.structure}:{r.character} ({r.share_far_outside:.0%} of {r.n_reports})"
                    for r in off.itertuples()],
               "summary_figure": [{"role": role, "structure": idx.loc[i, "structure"],
                                   "character": idx.loc[i, "character"],
                                   "n_reports": int(idx.loc[i, "n_reports"]),
                                   "spread": float(idx.loc[i, "spread"]),
                                   "offset": float(idx.loc[i, "offset"])} for role, i in chosen]}
    if flip_map:
        summary["orientation_table"] = str(a.orientation)
        summary["frames_flipped"] = int(sum(n_flipped.values()))
        summary["frames_low_confidence"] = int(sum(n_lowconf.values()))
        summary["frames_per_structure"] = {
            s: {"frames": int(sum(1 for k in flip_map if k[0] == s)),
                "flipped": int(n_flipped.get(s, 0)),
                "low_confidence": int(n_lowconf.get(s, 0))}
            for s in sorted({k[0] for k in flip_map})}
    (out / "vlm_heatmap_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{len(ok)} characters over {ch['structure'].nunique()} structures "
          f"(median spread {summary['median_spread']}, median offset {summary['median_offset']})")
    for k in ("tight_offcentre", "tight_centred", "intermediate", "scattered", "too_few_reports"):
        print(f"  {k:<18}{cats.get(k, 0):>4}")
    if len(off):
        print(f"  {len(off)} characters have more than a quarter of their reports well off the specimen's "
              f"own outline (beyond 3 x --outside_tol) — see share_far_outside")
    for role, i in chosen:
        r = idx.loc[i]
        print(f"  summary [{role}] {r['structure']}: {r['character']}  "
              f"n={int(r['n_reports'])} spread={r['spread']:.3f} offset={r['offset']:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
