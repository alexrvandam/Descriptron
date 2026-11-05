#!/usr/bin/env python3
"""Summary figure from compare_palettes.py / trees.R outputs.
 a  CIEDE2000 of Hungarian-matched palette colours, per comparison (all matched pairs pooled)
 b  pixel agreement (fraction of specimen pixels assigned to matched colours), per image
 c  colour fidelity: CIEDE2000 between each input pixel and its palette colour, per image and method
 d  species colour distances (mean between-specimen EMD in CIELAB), Descriptron vs recolorize, with Mantel r
Usage: make_summary_figure.py --compare_dir <dir> --trees_dir <dir> --out <png>
        --pairs A:B=label ... --methods NAME=label ... --scatter A:B=label ..."""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--compare_dir", required=True); ap.add_argument("--trees_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--pairs", nargs="+", required=True); ap.add_argument("--methods", nargs="+", required=True)
ap.add_argument("--scatter", nargs="+", default=[])
a = ap.parse_args()
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
pairs = [(p.split("=")[0].split(":"), p.split("=", 1)[1]) for p in a.pairs]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.0, 1.1]))

def box(ax, data, labels, ylabel, title):
    bp = ax.boxplot(data, vert=True, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
    for b in bp["boxes"]:
        b.set_facecolor("#dbe8f8"); b.set_edgecolor(SER[0])
    for i, d in enumerate(data, 1):
        ax.text(i, np.percentile(d, 75), f"{np.median(d):.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=9, color=INK); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)

d1, d2, lab = [], [], []
for (A, B), l in pairs:
    mc = pd.read_csv(os.path.join(a.compare_dir, f"matched_colours_{A}__{B}.csv")); pi = pd.read_csv(os.path.join(a.compare_dir, f"per_image_pair_{A}__{B}.csv"))
    d1.append(mc.dE00.values); d2.append(pi.pixel_agreement.values); lab.append(f"{l}\n(n={len(pi)})")
box(axs[0], d1, lab, "CIEDE2000, matched palette colours", "a  Palette colours")
box(axs[1], d2, lab, "fraction of specimen pixels", "b  Pixel assignment agreement")
axs[1].set_ylim(0, 1.05)
d3, l3 = [], []
for m in a.methods:
    n, l = m.split("=", 1); pi = pd.read_csv(os.path.join(a.compare_dir, f"per_image_{n}.csv")); d3.append(pi.fidelity_dE00_mean.values); l3.append(f"{l}\n(k med {pi.k.median():.0f})")
box(axs[2], d3, l3, "CIEDE2000, input pixel vs its palette colour", "c  Colour fidelity")
ax = axs[3]
tc = pd.read_csv(os.path.join(a.trees_dir, "tree_comparisons.csv"))
for i, s in enumerate(a.scatter):
    (A, B), l = s.split("=")[0].split(":"), s.split("=", 1)[1]
    MA = pd.read_csv(os.path.join(a.trees_dir, f"species_dist_{A}.csv"), index_col=0); MB = pd.read_csv(os.path.join(a.trees_dir, f"species_dist_{B}.csv"), index_col=0)
    S = MA.index.intersection(MB.index); iu = np.triu_indices(len(S), 1)
    x = MA.loc[S, S].values[iu]; y = MB.loc[S, S].values[iu]
    r = tc[tc.comparison == f"{A} vs {B}"]
    lbl = f"{l}: Mantel r {r.mantel_r.iloc[0]:.2f}, RF {r.RF_unrooted_norm.iloc[0]:.2f}" if len(r) else l
    ax.scatter(x, y, s=6, alpha=0.45, color=SER[i % 3], edgecolor="none", label=lbl)
lim = ax.get_xlim()[1], ax.get_ylim()[1]; mx = max(lim); ax.plot([0, mx], [0, mx], color=MUTED, lw=0.8, ls="--")
ax.set_xlabel("Descriptron species distance (EMD, CIELAB)"); ax.set_ylabel("recolorize species distance (EMD, CIELAB)")
ax.set_title("d  Species colour distances (43 spp.)", loc="left", fontsize=9, color=INK); ax.legend(fontsize=6.5, frameon=False, loc="upper left")
ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(a.out, dpi=200); plt.savefig(os.path.splitext(a.out)[0] + ".pdf"); print(a.out)
