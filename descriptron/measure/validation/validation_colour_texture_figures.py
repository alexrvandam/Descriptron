#!/usr/bin/env python3
"""Summary figures for the colour and texture validation (Supplementary Text S13), in the style of the geomorph
comparison figures.

  colour figure  : species separation of Descriptron (standard homology-cell features), patternize and Colormesh on the
                   same forewings; PC1-PC2 ordinations; why absolute colour alone is dominated by brightness.
  texture figure : Descriptron's masked GLCM against scikit-image's; grey-level invariance of the texture features.

  python validation_colour_texture_figures.py --colour_dir <colour_benchmark> --descriptron_features <color_homology_features_*.csv>
      --texture_script <texture_phenomics_homology.py> --image_glob "<wing crops>/*.png" --out_dir <dir>
"""
import argparse
import glob
import importlib.util
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix

ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
C_DESC, C_PAT, C_CM, C_BASE = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"


def loo(S, y):
    ok = [i for i in range(len(y)) if (y == y[i]).sum() >= 2]; hit = 0
    for i in ok:
        d = np.linalg.norm(S - S[i], axis=1); d[i] = np.inf; hit += y[d.argmin()] == y[i]
    return hit / len(ok), len(ok)


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def colour_figure(a, out):
    res = Path(a.colour_dir) / "results"
    sp = pd.read_csv(Path(a.colour_dir) / "inputs" / "specimens.csv").sort_values("order")
    F = pd.read_csv(a.descriptron_features)
    X = sp[["filename", "species"]].merge(F, on="filename", how="left")
    assert len(X) == 48 and X.iloc[:, 2:].notna().any(axis=1).all()
    y = X.species.values
    feats = [c for c in F.columns if c != "filename"]

    def desc_pca(cols, std):
        M = X[cols].astype(float); M = M.loc[:, M.notna().mean() > 0.9]; M = M.fillna(M.mean()); M = M.loc[:, M.std() > 0]
        Z = (M - M.mean()) / M.std() if std else M - M.mean()
        return PCA().fit_transform(Z.values)
    P_std = desc_pca(feats, True)
    P_abs = desc_pca([c for c in feats if c.endswith(("_L_mean_abs", "_a_mean_abs", "_b_mean_abs"))], False)
    meanL = X[[c for c in feats if c.endswith("_L_mean_abs")]].mean(1).values
    others = {}
    for m in ("patternize", "Colormesh"):
        t = pd.read_csv(res / f"pc_scores_{m}.csv"); t = t.set_index("id").loc[[Path(f).stem for f in X.filename.str.split(".tif").str[0] + ".tif"]]
        assert (t.species.values == y).all()
        others[m] = t[[c for c in t.columns if c.startswith("PC")]].values
    base = X[[c for c in feats if c.endswith("_L_mean_abs")]].mean(1).to_frame("L").assign(
        a=X[[c for c in feats if c.endswith("_a_mean_abs")]].mean(1), b=X[[c for c in feats if c.endswith("_b_mean_abs")]].mean(1)).values
    base_raw = base - base.mean(0)
    base = (base - base.mean(0)) / base.std(0)
    methods = [("Descriptron\n(standard features)", P_std, C_DESC), ("patternize", others["patternize"], C_PAT),
               ("Colormesh", others["Colormesh"], C_CM), ("Descriptron\n(absolute L*a*b* only)", P_abs, "#9ecae1"),
               ("wing mean L*a*b*\nstandardised", base, C_BASE), ("wing mean L*a*b*\nraw", base_raw, "#c7c7c7")]

    fig = plt.figure(figsize=(20, 10)); gs = fig.add_gridspec(2, 4)
    fig.suptitle("Descriptron colour-pattern features vs patternize and Colormesh (R), same 48 forewings of 9 species",
                 fontsize=14)
    ax = fig.add_subplot(gs[0, 0:2])
    x = np.arange(len(methods)); w = 0.6
    accs, lo, hi = [], [], []
    for _, P, _c in methods:                      # PC1-5 (or all 3 axes for the baselines): same for every method
        acc, n = loo(P[:, :5], y); l, h = wilson(acc, n)
        accs.append(acc); lo.append(acc - l); hi.append(h - acc)
    ax.bar(x, accs, w, yerr=[lo, hi], capsize=3, color=[c for _, _, c in methods], edgecolor="black", linewidth=0.5)
    for xi, v in zip(x, accs):
        ax.text(xi + 0.05, v + 0.02, f"{v:.2f}", ha="left", va="bottom", fontsize=9)
    ax.axhline(1 / 9, ls="--", color="grey", lw=0.8); ax.text(len(methods) - 0.5, 1 / 9 + 0.01, "chance", fontsize=8, ha="right", color="grey")
    ax.set_xticks(x); ax.set_xticklabels([m for m, _, _ in methods], fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("leave-one-out 1-NN species accuracy")
    ax.set_title("Species identified, PC1-5 (95% CI). Mean wing colour alone identifies most species", fontsize=11)
    cs = None
    if getattr(a, "colour_structure_csv", None) and Path(a.colour_structure_csv).exists():
        cs = pd.read_csv(a.colour_structure_csv).set_index("method")
    from scipy.spatial import ConvexHull
    for j, (lab, P, col) in enumerate(methods[:3]):
        ax = fig.add_subplot(gs[1, j])
        for s in sorted(set(y)):
            pts = ax.scatter(P[y == s, 0], P[y == s, 1], s=22, label=s, edgecolor="white", linewidth=0.4)
            q = P[y == s][:, :2]
            if len(q) >= 3:                         # outline each species' cluster
                h = ConvexHull(q); v = np.r_[h.vertices, h.vertices[:1]]
                ax.fill(q[v, 0], q[v, 1], color=pts.get_facecolor()[0], alpha=0.12, lw=0)
                ax.plot(q[v, 0], q[v, 1], color=pts.get_facecolor()[0], lw=0.8, alpha=0.8)
        key = lab.split("\n")[0]; stats = ""
        if cs is not None and key in cs.index:
            r = cs.loc[key]
            stats = (f"\nPERMANOVA (PC1-10): species R2 {r.species_R2:.2f}, p = {r.p:.4f}"
                     f"\nbeyond mean colour R2 {r.species_R2_beyond_mean_colour:.2f}, p = {r.p_beyond:.4f}")
        ax.set_title(lab.replace("\n", " ") + ": PC1 vs PC2" + (stats if cs is not None and key in cs.index else ""),
                     fontsize=10 if cs is not None else 11); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        if j == 0:
            ax.legend(fontsize=7, ncol=2, frameon=False, title="species", title_fontsize=7)
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(meanL, P_abs[:, 0], s=18, color="#9ecae1", edgecolor="black", linewidth=0.3, label="absolute L*a*b* only")
    ax.scatter(meanL, P_std[:, 0], s=18, color=C_DESC, edgecolor="white", linewidth=0.3, label="standard features")
    rs = pd.Series(P_abs[:, 0]).corr(pd.Series(meanL), method="spearman")
    rt = pd.Series(P_std[:, 0]).corr(pd.Series(meanL), method="spearman")
    ax.set_title(f"PC1 vs overall wing brightness\n(Spearman: absolute {rs:+.2f}, standard {rt:+.2f})", fontsize=11)
    ax.set_xlabel("wing mean L* (as stored)"); ax.set_ylabel("Descriptron PC1"); ax.legend(fontsize=8, frameon=False)
    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    ax.text(0, 1, "Same 48 wings, same order, per-method PCA.\n\n"
            "Descriptron: 52 homologous TPS cells on the wing\noutline; per cell L*a*b* means, the same\n"
            "relative to the wing's own mean, spread,\nentropy, dominant share (578 features,\nstandardised).\n\n"
            "patternize 0.0.5: patLanK (k = 3), TPS to mean\nlandmarks, darkest cluster; 17 landmarks.\n"
            "Colormesh 2.1: 17 landmarks + 55 outline\nsemilandmarks, tps.unwarp, tri.surf x3,\nrgb.measure.\n\n"
            "Accuracy CIs are Wilson 95% (n = 48).\nAbsolute colour alone follows overall\nbrightness (illumination, clearing); the\n"
            "relative features remove it.\n\nOn these psyllid wings the species differ\nmainly in overall colour: the standardised\n"
            "wing mean (3 numbers) identifies 0.81,\nclose to every pattern method. Beyond\nmean colour, species still explain about\n"
            "half of each method's pattern variation\n(PERMANOVA, bottom panels): the species\ndiffer in pattern too, and all three\n"
            "methods detect it.", va="top", family="monospace", fontsize=9)
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"validation_colour_vs_patternize_colormesh.{ext}", dpi=150)
    plt.close(fig)


def texture_figure(a, out):
    spec = importlib.util.spec_from_file_location("tx", a.texture_script)
    tx = importlib.util.module_from_spec(spec); spec.loader.exec_module(tx)
    imgs = sorted(glob.glob(a.image_glob))[:15]
    ours_all, ref_all = [], []
    ratios = {k: {"inv": [], "std": []} for k in ("contrast", "homogeneity", "energy")}
    pairs = {k: {"inv": [], "std": []} for k in ("contrast", "homogeneity", "energy")}
    for f in imgs:
        g = cv2.imread(f, 0); q = (g // 16).astype(np.uint8); h, w = q.shape; yy, xx = np.mgrid[:h, :w]
        for s in range(3):
            cy, cx, ry, rx = h * (0.3 + 0.2 * s), w * (0.3 + 0.2 * s), h * 0.25, w * 0.2
            m = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
            for ang in ANGLES:
                o = tx._masked_glcm_single(q, m, 3, ang, levels=16)
                sk = {np.pi / 4: 3 * np.pi / 4, 3 * np.pi / 4: np.pi / 4}.get(ang, ang)
                R = graycomatrix(np.where(m, q, 16).astype(np.uint8), [3], [sk], levels=17, symmetric=True,
                                 normed=False)[:16, :16, 0, 0].astype(float); R /= R.sum()
                ours_all.append(o.ravel()); ref_all.append(R.ravel())
        gf = g.astype(float); full = np.ones(g.shape, bool); v = {}
        for L in (8, 32):
            ql = np.clip((gf - gf.min()) / (gf.max() - gf.min()) * (L - 1), 0, L - 1).astype(np.uint8)
            G = np.mean([tx._masked_glcm_single(ql, full, 3, an, levels=L) for an in ANGLES], axis=0)
            I, J = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
            v[L] = (tx._haralick_invariant(G, levels=L),
                    [float((G * (I - J) ** 2).sum()), float((G / (1 + (I - J) ** 2)).sum()), float((G ** 2).sum())])
        for k, name in enumerate(("contrast", "homogeneity", "energy")):
            ratios[name]["inv"].append(v[32][0][k] / v[8][0][k]); ratios[name]["std"].append(v[32][1][k] / v[8][1][k])
            pairs[name]["inv"].append((v[8][0][k], v[32][0][k])); pairs[name]["std"].append((v[8][1][k], v[32][1][k]))
    ours_all = np.concatenate(ours_all); ref_all = np.concatenate(ref_all)
    fig = plt.figure(figsize=(20, 10)); gs = fig.add_gridspec(2, 4)
    fig.suptitle("Descriptron texture features: masked GLCM vs scikit-image, and grey-level invariance "
                 f"({len(imgs)} forewing images)", fontsize=14)
    ax = fig.add_subplot(gs[0, 0]); nz = (ours_all > 0) | (ref_all > 0)
    ax.scatter(ref_all[nz], ours_all[nz], s=6, color=C_DESC, alpha=0.5)
    lim = [ref_all[nz].min() * 0.8, ref_all[nz].max() * 1.2]
    ax.plot(lim, lim, "k--", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("scikit-image graycomatrix (mask via extra grey level)"); ax.set_ylabel("Descriptron masked GLCM")
    ax.set_title(f"Masked GLCM entries, 180 matrices\nmax |difference| = {np.abs(ours_all - ref_all).max():.1e}", fontsize=11)
    for j, name in enumerate(("contrast", "homogeneity", "energy")):
        ax = fig.add_subplot(gs[1, j])
        for kind, col, lab in (("inv", C_DESC, "invariant (Descriptron)"), ("std", C_BASE, "standard Haralick")):
            p = np.array(pairs[name][kind]); p = p / p[:, :1]
            ax.scatter(np.ones(len(p)) * (0 if kind == "inv" else 1) + np.random.default_rng(0).normal(0, 0.04, len(p)),
                       p[:, 1], color=col, s=20, label=lab, edgecolor="white", linewidth=0.3)
        ax.axhline(1, ls="--", color="black", lw=0.8); ax.set_yscale("log")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["invariant\n(Descriptron)", "standard"])
        ax.set_ylabel("value at 32 levels / value at 8 levels")
        ax.set_title(f"{name}: median ratio invariant {np.median(ratios[name]['inv']):.2f}, "
                     f"standard {np.median(ratios[name]['std']):.2f}", fontsize=10)
    ax = fig.add_subplot(gs[0, 1:3])
    names = ("contrast", "homogeneity", "energy"); x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, [np.median(ratios[n]["inv"]) for n in names], w, color=C_DESC, label="invariant (Descriptron)")
    ax.bar(x + w / 2, [np.median(ratios[n]["std"]) for n in names], w, color=C_BASE, label="standard Haralick")
    ax.axhline(1, ls="--", color="black", lw=0.8); ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("median ratio, 32 vs 8 grey levels (1 = invariant)"); ax.legend(frameon=False)
    ax.set_title("Grey-level invariance: contrast and homogeneity invariant; energy not", fontsize=11)
    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    ax.text(0, 1, "GLCM: distance 3 px, four directions\naveraged, 16 grey levels (cell min-max);\n"
            "only pairs with both pixels in the mask.\nReference: scikit-image graycomatrix with\n"
            "outside pixels as a 17th level, whose row\nand column are dropped. The programs name\n"
            "the two diagonals oppositely; averaging\nmakes the features identical.\n\n"
            "Invariant formulas: Lofstedt et al. 2019.\nEnergy's correction overshoots on real\nimages (co-occurrences near the diagonal).\n"
            "Descriptron always uses 16 levels, so\nenergy is comparable among all images\nanalysed with Descriptron, not with values\n"
            "computed at other quantisations.\nTexture depends on image scale.\n\nLBP: scikit-image; PCA: scikit-learn.",
            va="top", family="monospace", fontsize=9)
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"validation_texture_glcm.{ext}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colour_dir", required=True); ap.add_argument("--descriptron_features", required=True)
    ap.add_argument("--texture_script", required=True); ap.add_argument("--image_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--colour_structure_csv", default=None,
                    help="colour_species_structure.csv (colour_species_structure_v1.py): adds the PERMANOVA to each scatter")
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    colour_figure(a, out); texture_figure(a, out); print("wrote figures to", out)


if __name__ == "__main__":
    main()
