#!/usr/bin/env python3
"""Side-by-side recoloured specimens: input | each method's colour map, with the palette (area-scaled) under each.
Methods are given as in compare_palettes.py (--method NAME=descriptron:<dir>:<glob>  or  NAME=recolorize:<dir>).
Usage: make_side_by_side.py --pairs_dir <dir> --out <png> --specimens a,b,c --method ... [--title_map NAME=Title ...]"""
import argparse, os, sys
import numpy as np, pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_palettes import load_descriptron, load_recolorize, load_rgb

ap = argparse.ArgumentParser()
ap.add_argument("--pairs_dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--specimens", required=True)
ap.add_argument("--method", action="append", required=True); ap.add_argument("--title_map", action="append", default=[])
a = ap.parse_args()
titles = dict(t.split("=", 1) for t in a.title_map)
methods = []
for spec in a.method:
    name, rest = spec.split("=", 1); parts = rest.split(":")
    if parts[0] == "descriptron":
        methods.append((name, "d", parts[1], parts[2], None))
    else:
        methods.append((name, "r", parts[1], None, pd.read_csv(os.path.join(parts[1], "palettes.csv")).groupby("image")))
specs = a.specimens.split(",")
nc = 1 + len(methods)
fig = plt.figure(figsize=(2.9 * nc, 2.75 * len(specs)))
gs = fig.add_gridspec(len(specs) * 2, nc, height_ratios=[1, 0.12] * len(specs), hspace=0.08, wspace=0.04)
for r, stem in enumerate(specs):
    fg = stem + ".png"
    raw = load_rgb(os.path.join(a.pairs_dir, fg))
    mk = np.array(Image.open(os.path.join(a.pairs_dir, fg.replace("_fg_", "_bin_", 1)))); m = (mk[:, :, 0] if mk.ndim == 3 else mk) > 0
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    panels = [("input", np.where(m[..., None], raw, 255), None)]
    for name, kind, d, pat, pal in methods:
        res = load_descriptron(d, pat, fg, m) if kind == "d" else load_recolorize(d, stem, m, pal)
        if res is None:
            panels.append((name, np.full_like(raw, 255), None)); continue
        rgbp = lab2rgb(res["pal_lab"].reshape(-1, 1, 3)).reshape(-1, 3)
        img = np.ones(raw.shape, float); img[m] = rgbp[res["labels"][m] - 1]
        frac = np.bincount(res["labels"][m], minlength=len(rgbp) + 1)[1:] / m.sum()
        panels.append((name, (img * 255).astype(np.uint8), (rgbp, frac)))
    for c, (name, img, pal) in enumerate(panels):
        ax = fig.add_subplot(gs[2 * r, c]); ax.imshow(img[y0:y1, x0:x1]); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 0:
            ax.set_title(titles.get(name, name), fontsize=8)
        if c == 0:
            ax.set_ylabel(stem.split("png_")[0].replace("__", "\n").replace("_", " "), fontsize=7)
        axp = fig.add_subplot(gs[2 * r + 1, c]); axp.set_xlim(0, 1); axp.set_ylim(0, 1); axp.axis("off")
        if pal is not None:
            rgbp, frac = pal; o = np.argsort(-frac); x = 0.0
            for i in o:
                axp.add_patch(plt.Rectangle((x, 0), frac[i], 1, color=rgbp[i], ec="0.5", lw=0.3)); x += frac[i]
            axp.text(1.0, -0.15, f"k={len(frac)}", ha="right", va="top", fontsize=6, transform=axp.transAxes)
plt.savefig(a.out, dpi=150, bbox_inches="tight"); print(a.out)
