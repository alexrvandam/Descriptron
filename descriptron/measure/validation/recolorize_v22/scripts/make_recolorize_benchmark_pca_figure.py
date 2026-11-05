"""PC1 x PC2 of each method in the forewing colour-pattern benchmark, one panel per method (plotting only).
Scores come from `colour_benchmark_v1.py analyse` (pc_scores_<method>.csv); the identification numbers in each title
come from `descriptron_trait_stats` (summary.csv: leave-one-out 1-NN on all PCs, Wilson 95% CI, PERMANOVA R2).
Usage: make_recolorize_benchmark_pca_figure.py --panel "Label|pc_scores.csv|trait_stats summary.csv|trait_set" [...]
       --out <prefix> [--suptitle TEXT]"""
import argparse, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--panel", action="append", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--suptitle", default="")
a = ap.parse_args()

PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MK = "osD^v<>Ph*"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
panels = [p.split("|") for p in a.panel]
fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 4.6), facecolor="white")
for ax, (label, sc, st, ts) in zip(np.atleast_1d(axes), panels):
    Z = pd.read_csv(sc)
    S = pd.read_csv(st).set_index("trait_set").loc[ts]
    sp = sorted(Z.species.unique(), key=lambda s: int(re.sub(r"\D", "", s) or 0))
    for q, s in enumerate(sp):
        m = Z.species == s
        ax.scatter(Z.PC1[m], Z.PC2[m], s=36, c=PAL[q % 10], marker=MK[q % 10], edgecolor="k", linewidth=0.4, label=s)
    ax.axhline(0, c="0.88", lw=0.6, zorder=0); ax.axvline(0, c="0.88", lw=0.6, zorder=0)
    ax.set_xlabel(f"PC1 ({100 * S.PC1_var:.1f}%)"); ax.set_ylabel(f"PC2 ({100 * S.PC2_var:.1f}%)")
    ax.set_title(f"{label}\nspecies identified {100 * S.loo_acc_allPCs:.0f}% "
                 f"[{100 * S.loo_ci_low_allPCs:.0f}-{100 * S.loo_ci_high_allPCs:.0f}]  ·  PERMANOVA R² {S.permanova_R2:.2f}",
                 fontsize=9.5, color=INK, loc="left")
np.atleast_1d(axes)[-1].legend(title="species", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
if a.suptitle:
    fig.suptitle(a.suptitle, x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight")
print("wrote", a.out + ".png/.pdf")
