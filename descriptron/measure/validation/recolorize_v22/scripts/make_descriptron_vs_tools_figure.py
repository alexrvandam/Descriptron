"""Combined figure: Descriptron colour-pattern homology vs patternize / Colormesh (forewings) and the moth demo.
  a  forewing species identification (leave-one-out 1-NN, all PCs, Wilson 95% CI; descriptron_trait_stats summary.csv)
  b  forewing frame slippage at the wing margin: share of the warped consensus outside the wing (slippage_summary.csv)
  c  forewing interior slippage: landmark left out, error % of sqrt(wing area) (same file)
  d  moth TPS fit vs number of outline control points, slid semilandmarks before / after projection onto the outline
  e  moth phylogenetic signal (Kmult) of the colour-pattern grid by segmentation input (phylogenetic_signal.csv)
  f  moth phylomorphospace (image from descriptron_phylo)       g  grid cells that drive colour PC1 (image)
Usage: make_descriptron_vs_tools_figure.py --forewing_stats <trait_stats summary.csv> --slippage <slippage_summary.csv>
   --fit_before <outline_fit.csv> --fit_after <outline_fit.csv> --kmult NAME=CSV [...] --phylomorphospace <png>
   --spatial <png> --out <prefix>"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

ap = argparse.ArgumentParser()
for k in ("--forewing_stats", "--slippage", "--fit_before", "--fit_after", "--phylomorphospace", "--spatial", "--out"):
    ap.add_argument(k, required=True)
ap.add_argument("--kmult", action="append", required=True, help="label=phylogenetic_signal.csv")
a = ap.parse_args()

DESC, PAT, CM, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=8.5, color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)


fig = plt.figure(figsize=(13, 13), facecolor="white")
gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.7, height_ratios=[1, 1, 1.5])

# a  species identification
S = pd.read_csv(a.forewing_stats).set_index("trait_set")
names = {"Colormesh": "Colormesh", "patternize": "patternize", "Descriptron_colour_recolorize": "Descriptron: recolorize",
         "Descriptron_colour_photo": "Descriptron: photo", "Descriptron_colour": "Descriptron: normalise + recolorize",
         "Descriptron_texture": "Descriptron: texture grid"}
order = [k for k in names if k in S.index]
ax = fig.add_subplot(gs[0, 0:2])
for i, k in enumerate(order):
    acc, lo, hi = S.loc[k, ["loo_acc_allPCs", "loo_ci_low_allPCs", "loo_ci_high_allPCs"]] * 100
    c = PAT if k == "patternize" else CM if k == "Colormesh" else DESC
    ax.barh(i, acc, height=0.6, color=c); ax.plot([lo, hi], [i, i], color=INK, lw=1)
    ax.text(hi + 1.5, i, f"{acc:.0f}%", va="center", color=INK, fontsize=8)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in order]); ax.set_xlim(0, 110)
ax.axvline(S["chance"].iloc[0] * 100, color=GREY, lw=0.8, ls=":"); ax.text(S["chance"].iloc[0] * 100 + 1, -0.75, "chance", color=INK2, fontsize=7)
ax.set_xlabel("species identified, leave-one-out (%, all PCs; bar = Wilson 95% CI)")
style(ax, f"a  Forewings ({int(S['loo_scored'].iloc[0])} wings, {int(S['n_species'].iloc[0])} spp.): species identification")

# b, c  slippage
L = pd.read_csv(a.slippage).set_index("controls").loc[["landmarks", "outline", "both"]]
lab = ["17 landmarks\n(patternize)", "100 semi-\nlandmarks", "landmarks +\nsemilandmarks"]
cols = [PAT, GREY, DESC]
for j, (col, ttl, xl) in enumerate([("outside_share", "b  Margin slippage", "warped grid outside the wing (%)"),
                                    ("interior_error_pct_median", "c  Interior slippage", "left-out landmark error\n(% of size)")]):
    ax = fig.add_subplot(gs[0, 2 + j]); v = L[col] * (100 if col == "outside_share" else 1)
    ax.barh(range(3), v, height=0.6, color=cols)
    for i, x in enumerate(v):
        ax.text(x + v.max() * 0.03, i, f"{x:.1f}" if col != "outside_share" else f"{x:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(3)); ax.set_yticklabels(lab); ax.set_xlim(0, v.max() * 1.3); ax.set_xlabel(xl); ax.invert_yaxis()
    style(ax, ttl)

# d  TPS fit vs control points
ax = fig.add_subplot(gs[1, 0:2]); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
for f, c, lb in [(a.fit_before, GREY, "slid, off the outline"), (a.fit_after, DESC, "slid + projected (V42 default)")]:
    m = pd.read_csv(f).groupby("n_control").IoU.median()
    ax.plot(m.index, m.values, color=c, lw=2, marker="o", ms=4, label=lb)
    ax.text(m.index[-1] * 1.08, m.values[-1], f"{m.values[-1]:.3f}", va="center", fontsize=7.5, color=INK)
ax.set_xscale("log"); ax.set_xticks([4, 10, 20, 50, 100]); ax.set_xticklabels(["4", "10", "20", "50", "100"])
ax.set_xlim(3, 190); ax.set_xlabel("TPS control points on the outline"); ax.set_ylabel("IoU, warped grid vs AI mask (median)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.set_title("d  Moths (423): grid fit vs control points", loc="left", fontsize=8.5, color=INK, fontweight="bold")

# e  Kmult
ax = fig.add_subplot(gs[1, 2:4]); K = []
for s in a.kmult:
    nm, p = s.split("=", 1); r = pd.read_csv(p).iloc[0]; K.append((nm, r.Kmult, r.P))
for i, (nm, k, p) in enumerate(K):
    ax.barh(i, k, height=0.6, color=DESC if "Descriptron" in nm else GREY)
    ax.text(k + 0.01, i, f"{k:.2f} (P={p:.3f})", va="center", fontsize=7.5, color=INK)
ax.set_yticks(range(len(K))); ax.set_yticklabels([k[0] for k in K]); ax.set_xlim(0, 1.05); ax.invert_yaxis()
ax.set_xlabel("Kmult (colour-pattern grid, 43 spp.)"); style(ax, "e  Moths: phylogenetic signal")

for j, (img, ttl) in enumerate([(a.phylomorphospace, "f  Phylomorphospace (recolorize + grid)"),
                                (a.spatial, "g  Grid cells driving colour PC1")]):
    ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2]); im = mpimg.imread(img)
    if j == 1:                      # crop the white margins of the spatial map
        h, w = im.shape[:2]; im = im[int(h * 0.22):int(h * 0.78), int(w * 0.04):int(w * 0.97)]
    ax.imshow(im); ax.axis("off"); ax.set_title(ttl, loc="left", fontsize=8.5, color=INK, fontweight="bold")

fig.suptitle("Descriptron colour-pattern homology: TPS grid warped by landmarks + sliding semilandmarks on AI masks",
             x=0.01, y=0.915, ha="left", fontsize=10.5, color=INK, fontweight="bold")
fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(a.out + ".pdf", bbox_inches="tight")
print("wrote", a.out + ".png/.pdf")
