#!/usr/bin/env python3
"""
biorag_vlm_roi_figures_v1.py — show what the model says it saw, and where
=========================================================================

A character proposed by a model is a claim, and a claim about an image can fail in
three different ways that look identical in a table of states:

  the character is real          specimens scored present differ visibly from those
                                 scored absent, in the place the model points to;
  the model is confabulating     the states are assigned but the two groups look the
                                 same, or the pointed-at place holds nothing;
  the model is looking elsewhere the states may even be informative, but the location
                                 it reports is off the structure, or is always the
                                 centre of the image, or is scattered at random —
                                 the character is then about something other than
                                 what its name says.

Only the third is detectable from the scores alone, and only barely. So the scorer is
asked where on the structure it saw each feature, and this draws that back onto the
masked foreground:

  <character>_plate.png    specimens scored PRESENT above, ABSENT below, each on its
                           own masked foreground with the reported point marked. The
                           comparison a reader would have to make anyway.
  <character>_heatmap.png  every reported location for that character, pooled in the
                           structure's own normalised frame and shown as a density
                           over the mean outline: present on the left, absent on the
                           right. A real character concentrates somewhere; a
                           confabulated one spreads out or piles up in the middle.

The middle-of-the-image pile-up is worth naming, because it is the common failure: a
model that has not localised anything tends to answer (0.5, 0.5). The dispersion of
the reported points, and how far they sit from the centre, are printed on the figure
so that pattern is visible rather than inferred.

  python biorag_vlm_roi_figures_v1.py --states "$M/vlm_characters_full/vlm_character_states.tsv" \\
      --characters "$M/vlm_characters_full/vlm_proposed_characters.tsv" \\
      --coco <coco.json> --image_dir <dir> --taxon_profile <p.yaml> \\
      --matrix_dir "$M/compiled_key_tier" --out_dir "$M/vlm_characters_full/figures"
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
from PIL import Image                                                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                      # noqa: E402
from biorag_vlm_characters_v1 import crop, outlines                      # noqa: E402

VERSION = "1.0"
Image.MAX_IMAGE_PIXELS = None
PRESENT, ABSENT = "#2e7d32", "#c62828"


def plate(ax, img, pt, colour, title):
    ax.imshow(img)
    if pt is not None and all(v == v for v in pt):
        x, y = pt[0] * img.width, pt[1] * img.height
        ax.scatter([x], [y], s=190, facecolor="none", edgecolor="#ff6f00", lw=2.2, zorder=5)
        ax.scatter([x], [y], s=16, color="#ff6f00", zorder=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour)
        sp.set_linewidth(1.8)
    ax.set_title(title, fontsize=7, color=colour, pad=2)


def heat(ax, pts, colour, label):
    """Density of the reported locations in the structure's normalised frame."""
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="#90a4ae", lw=1.0))
    if len(pts):
        p = np.asarray(pts, float)
        try:
            from scipy.stats import gaussian_kde
            if len(p) >= 5 and np.ptp(p[:, 0]) > 1e-6 and np.ptp(p[:, 1]) > 1e-6:
                k = gaussian_kde(p.T, bw_method=0.35)
                gx, gy = np.mgrid[0:1:120j, 0:1:120j]
                z = k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
                ax.imshow(z.T, extent=(0, 1, 1, 0), origin="upper", cmap="magma_r", alpha=0.85)
        except Exception:
            pass
        ax.scatter(p[:, 0], p[:, 1], s=14, color=colour, edgecolor="white", lw=0.4, zorder=4)
        spread = float(np.hypot(*(p - p.mean(0)).T).mean())
        centre = float(np.hypot(*(p.mean(0) - 0.5)))
        ax.set_xlabel(f"n={len(p)}   spread={spread:.3f}   offset from centre={centre:.3f}",
                      fontsize=6.6)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(1.03, -0.03)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=8, color=colour, loc="left")


def cell_of(point, warped_cells):
    """Which homologous cell does a reported point fall in?

    The grid was warped from the consensus onto this individual, so its cells follow that
    specimen's own anatomy. A point is assigned by containment in a warped quad, as
    build_cell_mask does, not by nearest centre — a long thin cell near a margin would
    otherwise capture points that lie well outside it.
    """
    import cv2 as _cv
    if point is None or any(v != v for v in point):
        return None
    for name, poly in warped_cells.items():
        q = np.asarray(poly, np.float32)
        if _cv.pointPolygonTest(q, (float(point[0]), float(point[1])), False) >= 0:
            return name
    return None


def main():
    ap = argparse.ArgumentParser(description="Draw the model's own reported locations")
    ap.add_argument("--states", required=True)
    ap.add_argument("--characters", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude_list", default=None)
    ap.add_argument("--frames_dir", default=None,
                    help="homology frames. With these the plates show the aligned images and each "
                         "reported point is assigned to a homologous grid cell, so 'is it looking "
                         "at the same place on every specimen' becomes a count rather than an "
                         "impression")
    ap.add_argument("--per_group", type=int, default=5)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--informativeness", default=None,
                    help="character_informativeness.tsv, to draw the most informative first")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    codes = sorted(set(pd.read_csv(Path(a.matrix_dir) / "specimen_matrix_long.csv")["species"]))
    st = pd.read_csv(a.states, sep="\t")
    chars = pd.read_csv(a.characters, sep="\t")
    if "x" not in st.columns:
        st["x"] = np.nan
        st["y"] = np.nan

    order = list(dict.fromkeys(st["character"]))
    if a.informativeness and Path(a.informativeness).exists():
        inf = pd.read_csv(a.informativeness, sep="\t")
        order = [c for c in inf["character"] if c in set(order)] + \
                [c for c in order if c not in set(inf["character"])]
    order = order[:a.top]

    structures = {c.split(":", 1)[0] for c in order}
    from biorag_vlm_characters_v1 import load_exclusions
    drop_ids, drop_imgs = load_exclusions(a.exclude_list)
    outl = outlines(Path(a.coco), profile, codes, structures, drop_ids, drop_imgs)
    img_dir = Path(a.image_dir)
    grids = {}
    if a.frames_dir:
        for stc in structures:
            f = Path(a.frames_dir) / stc / "warped_grids.json"
            if f.exists():
                grids[stc] = json.loads(f.read_text())
    index = []

    for ch in order:
        struct, name = ch.split(":", 1)
        g = st[st["character"] == ch]
        if not len(g):
            continue
        meta = chars[(chars["structure"] == struct) & (chars["name"] == name)]
        where = str(meta.iloc[0]["where"]) if len(meta) else ""
        pres = g[g["state"] == "present"]
        abst = g[g["state"] == "absent"]
        picks = {"present": pres.head(a.per_group), "absent": abst.head(a.per_group)}
        wc_all = grids.get(struct, {})
        cells_hit = []
        if wc_all:
            for _, r2 in pres.iterrows():
                if r2.get("x") == r2.get("x") and r2["specimen_id"] in wc_all:
                    cells_hit.append(cell_of((r2["x"] * 768, r2["y"] * 768),
                                             wc_all[r2["specimen_id"]]))
        cells_hit = [c for c in cells_hit if c]
        top_cell, top_share = None, None
        if cells_hit:
            vc = pd.Series(cells_hit).value_counts()
            top_cell, top_share = vc.index[0], round(float(vc.iloc[0] / len(cells_hit)), 3)
        n = max(len(picks["present"]), len(picks["absent"]), 1)

        fig = plt.figure(figsize=(2.35 * n + 6.2, 5.8))
        gs = fig.add_gridspec(2, n + 2, width_ratios=[1] * n + [1.25, 1.25], wspace=0.12,
                              hspace=0.22, left=0.03, right=0.985, top=0.80, bottom=0.07)
        for r, (state, colour) in enumerate((("present", PRESENT), ("absent", ABSENT))):
            sub = picks[state]
            for c in range(n):
                ax = fig.add_subplot(gs[r, c])
                if c < len(sub):
                    row = sub.iloc[c]
                    key = (row["specimen_id"], struct)
                    fp = (Path(a.frames_dir) / struct / f"{row['specimen_id']}.png"
                          if a.frames_dir else None)
                    im = None
                    if fp is not None and fp.exists():
                        im = Image.open(fp).convert("RGB")
                    elif key in outl:
                        pts, fn = outl[key]
                        p = img_dir / fn
                        if p.exists():
                            im = crop(p, pts, mask_mode="hard")
                    if im is not None:
                        pt = (row.get("x"), row.get("y"))
                        cname = None
                        wc = grids.get(struct, {}).get(row["specimen_id"])
                        if wc and pt[0] == pt[0]:
                            cname = cell_of((pt[0] * im.width, pt[1] * im.height), wc)
                        plate(ax, im, pt, colour,
                              f"{row['specimen_id']}" + (f"  [{cname}]" if cname else ""))
                        continue
                ax.axis("off")
        axp = fig.add_subplot(gs[0, n:])
        axa = fig.add_subplot(gs[1, n:])
        heat(axp, pres[["x", "y"]].dropna().values, PRESENT, "where it says it saw it — PRESENT")
        heat(axa, abst[["x", "y"]].dropna().values, ABSENT, "where it looked — ABSENT")
        fig.suptitle(f"{struct} — “{name}”\n"
                     f"instructed to look at: {where[:120]}\n"
                     f"present {len(pres)}   absent {len(abst)}   "
                     f"orange ring = the location the model reported"
                     + (f"   |   modal homologous cell {top_cell} "
                        f"({100 * top_share:.0f}% of reports)" if top_cell else ""),
                     fontsize=9.5, x=0.02, ha="left", color="#1a237e")
        f = out / f"{struct}__{name[:40].replace(' ', '_').replace('/', '-')}.png"
        fig.savefig(f, dpi=150)
        plt.close(fig)
        pxy = pres[["x", "y"]].dropna().values
        index.append({"figure": f.name, "structure": struct, "character": name,
                      "n_present": len(pres), "n_absent": len(abst),
                      "localised": int(g["x"].notna().sum()),
                      "spread_present": (round(float(np.hypot(*(pxy - pxy.mean(0)).T).mean()), 3)
                                         if len(pxy) > 2 else None),
                      "offset_from_centre": (round(float(np.hypot(*(pxy.mean(0) - 0.5))), 3)
                                             if len(pxy) else None),
                      "modal_homologous_cell": top_cell,
                      "share_in_modal_cell": top_share,
                      "cells_assigned": len(cells_hit)})
    pd.DataFrame(index).to_csv(out / "vlm_roi_index.tsv", sep="\t", index=False)
    print(f"{len(index)} character plates -> {out}")


if __name__ == "__main__":
    main()
