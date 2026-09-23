#!/usr/bin/env python3
"""
biorag_character_roi_figures_v1.py — RETIRED (2026-09-20)
==========================================================
Drew the RETIRED outline-descriptor characters on the specimens. The VLM arm has its own
and better ROI tool, `biorag_vlm_roi_figures_v1.py`, which also marks where the scorer
says it looked and assigns that to a homologous grid cell. Use that one.

ORIGINAL HEADER FOLLOWS
-----------------------
biorag_character_roi_figures_v1.py — show the character on the specimen
=======================================================================

A diagnostic character is worth nothing to a reader who cannot find it on a
specimen. For every candidate this draws the structure on the actual image with
the region the character is computed from picked out, so the claim can be checked
by eye rather than taken on trust:

  lobes, notches        the curvature extrema themselves are marked on the outline
  deep concavities      the convexity defect and the hull chord it departs from
  apex / base curvature the end of the principal axis the quantity is measured at
  symmetry              the principal axis and the mirrored outline over the real one
  elongation and the    the principal-axis box, so the proportion being binned is
  other whole-outline   visible; the whole outline is emphasised for the harmonics
  quantities

Each plate puts specimens carrying the state beside specimens that do not, drawn
at the same scale, which is the comparison a reader would have to make anyway.

  python biorag_character_roi_figures_v1.py --candidates "$M/autapomorphies/autapomorphy_candidates.tsv" \\
      --coco <coco.json> --image_dir <dir> --states "$M/mask_characters/mask_character_states.tsv" \\
      --taxon_profile <p.yaml> --matrix_dir "$M/compiled_key_tier" --out_dir "$M/autapomorphies/figures"
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402
import pandas as pd                                                    # noqa: E402
from PIL import Image                                                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                    # noqa: E402
from biorag_mask_characters_v1 import curvature, resample, specimen_of  # noqa: E402

VERSION = "1.0"
Image.MAX_IMAGE_PIXELS = None
HAVE, LACK = "#2e7d32", "#c62828"


def load_outlines(coco: Path, profile, codes):
    """{(specimen, structure): (points, image file name)}"""
    j = json.loads(coco.read_text())
    cats = {c["id"]: c["name"] for c in j.get("categories", [])}
    imgs = {i["id"]: i["file_name"] for i in j.get("images", [])}
    outl = {}
    for a in j.get("annotations", []):
        seg = a.get("segmentation")
        if not seg or not isinstance(seg, list) or not seg[0]:
            continue
        pts = np.asarray(seg[0], dtype=float).reshape(-1, 2)
        if len(pts) < 12:
            continue
        fn = imgs.get(a["image_id"], "")
        _code, sid = specimen_of(fn, profile, codes)
        if not sid:
            continue
        key = (sid, cats.get(a["category_id"], "?"))
        if key not in outl or len(pts) > len(outl[key][0]):
            outl[key] = (pts, fn)
    return outl


def highlight(ax, pts, quantity):
    """Mark the part of the outline the quantity is actually computed from."""
    r = resample(pts)
    k = curvature(r)
    c = r - r.mean(axis=0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    scale = np.sqrt(abs(np.cross(r - r.mean(0), np.roll(r, -1, 0) - r.mean(0)).sum()) / 2) or 1.0

    if quantity in ("lobes", "notches"):
        thr = 1.5 / scale
        m = (k > thr) if quantity == "lobes" else (k < -thr)
        ax.scatter(r[m, 0], r[m, 1], s=16, color="#ffb300", zorder=5,
                   edgecolor="#5d4037", linewidth=0.4)
        return "curvature extrema marked"
    if quantity == "deep_concavities":
        hull = r[np.sort(np.unique(_hull_idx(r)))]
        ax.plot(np.r_[hull[:, 0], hull[0, 0]], np.r_[hull[:, 1], hull[0, 1]],
                color="#ffb300", lw=1.2, ls="--", zorder=4)
        return "convex hull dashed; the gap is the concavity"
    if quantity in ("apex_curvature", "base_curvature"):
        order = np.argsort(proj[:, 0])
        idx = order[-10:] if quantity == "apex_curvature" else order[:10]
        ax.scatter(r[idx, 0], r[idx, 1], s=22, color="#ffb300", zorder=5,
                   edgecolor="#5d4037", linewidth=0.4)
        return "end of the principal axis"
    if quantity == "symmetry":
        mir = proj.copy()
        mir[:, 1] *= -1
        back = mir @ vt + r.mean(axis=0)
        ax.plot(np.r_[back[:, 0], back[0, 0]], np.r_[back[:, 1], back[0, 1]],
                color="#ffb300", lw=1.0, ls=":", zorder=4)
        return "outline mirrored about its long axis, dotted"
    if quantity in ("skeleton_ends", "skeleton_branches"):
        ax.plot(r[:, 0], r[:, 1], color="#ffb300", lw=2.0, alpha=0.5, zorder=4)
        return "whole outline (medial-axis topology)"
    # whole-outline quantities: show the principal-axis box being binned
    box = np.array([[proj[:, 0].min(), proj[:, 1].min()], [proj[:, 0].max(), proj[:, 1].min()],
                    [proj[:, 0].max(), proj[:, 1].max()], [proj[:, 0].min(), proj[:, 1].max()]])
    b = box @ vt + r.mean(axis=0)
    ax.plot(np.r_[b[:, 0], b[0, 0]], np.r_[b[:, 1], b[0, 1]], color="#ffb300", lw=1.0, ls="--",
            zorder=4)
    return "principal-axis box"


def _hull_idx(r):
    import cv2
    return cv2.convexHull(r.astype(np.float32), returnPoints=False)[:, 0]


def panel(ax, pts, fn, image_dir: Path, quantity, colour, title):
    img = None
    p = Path(image_dir) / fn
    if p.exists():
        try:
            im = Image.open(p)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            pad = 0.18 * max(x1 - x0, y1 - y0)
            box = (max(0, int(x0 - pad)), max(0, int(y0 - pad)),
                   min(im.width, int(x1 + pad)), min(im.height, int(y1 + pad)))
            img = np.asarray(im.convert("RGB").crop(box))
            ax.imshow(img, extent=(box[0], box[2], box[3], box[1]))
        except Exception:                                    # unreadable image: outline only
            img = None
    ax.plot(np.r_[pts[:, 0], pts[0, 0]], np.r_[pts[:, 1], pts[0, 1]], color=colour, lw=1.4,
            zorder=3)
    note = highlight(ax, pts, quantity)
    if img is None:
        ax.set_aspect("equal")
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(colour)
        s.set_linewidth(1.6)
    ax.set_title(title, fontsize=7.4, color=colour, pad=2)
    return note


def main():
    ap = argparse.ArgumentParser(description="Draw each candidate character on the specimens")
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--informativeness", default=None,
                    help="character_informativeness.tsv — draw the most species-informative "
                         "characters when no state is fixed-and-unique, which is the usual case")
    ap.add_argument("--states", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top", type=int, default=12, help="how many candidates to draw")
    ap.add_argument("--per_group", type=int, default=4)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    long = pd.read_csv(Path(a.matrix_dir) / "specimen_matrix_long.csv")
    codes = sorted(set(long["species"]))
    if a.candidates and Path(a.candidates).exists() and \
            len(pd.read_csv(a.candidates, sep="\t")):
        cand = pd.read_csv(a.candidates, sep="\t")
    elif a.informativeness:
        inf = pd.read_csv(a.informativeness, sep="\t")
        cand = pd.DataFrame({"character": inf["character"], "state": inf["purest_state"],
                             "species": inf["purest_state_species"],
                             "q": inf["adjusted_mutual_information"],
                             "singleton": False, "confirmed": False,
                             "verdict": ["most informative — the purest state of this character, "
                                         "not a fixed one"] * len(inf)})
    else:
        raise SystemExit("pass --candidates or --informativeness")
    st = pd.read_csv(a.states, sep="\t")
    outl = load_outlines(Path(a.coco), profile, codes)

    keep = cand[cand["verdict"].str.startswith("autapomorphy")]
    if not len(keep):
        keep = cand[cand["confirmed"] == True] if "confirmed" in cand else cand    # noqa: E712
    if not len(keep):
        keep = cand
    keep = keep.head(a.top)
    index = []
    for i, (_, r) in enumerate(keep.iterrows(), 1):
        ch, state, sp = r["character"], str(r["state"]), r["species"]
        struct, quantity = ch.split(":", 1)
        rows = st[(st["character"] == ch)]
        have = [s for s in rows[rows["state"].astype(str) == state]["specimen_id"]
                if (s, struct) in outl]
        lack = [s for s in rows[rows["state"].astype(str) != state]["specimen_id"]
                if (s, struct) in outl]
        have = [s for s in have if s.rsplit("_", 1)[0] == sp][:a.per_group] or have[:a.per_group]
        rng = np.random.default_rng(i)
        lack = list(rng.choice(lack, size=min(a.per_group, len(lack)), replace=False)) if lack else []
        if not have:
            continue
        n = max(len(have), len(lack))
        fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.6), squeeze=False)
        note = ""
        for col in range(n):
            for row, group, colour, lab in ((0, have, HAVE, sp), (1, lack, LACK, "other species")):
                ax = axes[row][col]
                if col < len(group):
                    sid = group[col]
                    pts, fn = outl[(sid, struct)]
                    note = panel(ax, pts, fn, Path(a.image_dir), quantity, colour, sid)
                else:
                    ax.axis("off")
            axes[0][col].set_ylabel("")
        axes[0][0].text(-0.08, 0.5, f"{sp}\n(state present)", transform=axes[0][0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8, color=HAVE, weight="bold")
        axes[1][0].text(-0.08, 0.5, "other species\n(state absent)", transform=axes[1][0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8, color=LACK, weight="bold")
        q = r.get("q", float("nan"))
        kind = ("singleton candidate" if r.get("singleton")
                else "confirmed on held-back specimens" if r.get("confirmed")
                else "most informative state (NOT fixed within the species)")
        fig.suptitle(f"{struct} — {quantity.replace('_', ' ')} = {state}   |   {sp}   |   {kind}"
                     f"{'' if pd.isna(q) else f'   score = {q:.2g}'}\n"
                     f"highlighted in amber: {note}",
                     fontsize=9.5, x=0.02, ha="left", color="#1a237e")
        fig.tight_layout(rect=(0.02, 0, 1, 0.9))
        f = out / f"char{i:02d}_{sp}_{struct}_{quantity}.png".replace("/", "-")
        fig.savefig(f, dpi=160)
        plt.close(fig)
        index.append({"figure": f.name, "species": sp, "structure": struct,
                      "quantity": quantity, "state": state, "verdict": r["verdict"],
                      "q": q, "highlighted": note})
    pd.DataFrame(index).to_csv(out / "figure_index.tsv", sep="\t", index=False)
    print(f"{len(index)} plates -> {out}")


if __name__ == "__main__":
    main()
