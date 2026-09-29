#!/usr/bin/env python3
"""Summary figure: Descriptron's components against established alternatives (the Descriptron-v2 paper's Fig. 7).
Every value is computed from the benchmark output files; nothing is typed in.

  python validation_summary_figure.py \
      --validation_dir <benchmark outputs>            (segmentation_benchmark/, KEYPOINT_BENCHMARK_ALL_CONDITIONS.csv,
                                                        colour_benchmark/, keypoint_benchmark_reoriented/crops/)
      --geomorph_csv <validate_shape_stats_vs_geomorph.py output>/python_vs_geomorph_values.csv \
      --semilandmark_dir <validate_semilandmarks_vs_geomorph.py output> \
      --colour_structure_csv <colour_species_structure_v1.py output>.csv \
      --descriptron_features <color_homology_features_whole_wing_*.csv> \
      --measure_dir <descriptron>/measure --out validation_summary
"""
import argparse, importlib.util, io, contextlib, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1d2327", "#6b7780", "#e3e7ea"
C_DESC, C_REF, C_REF2, C_BASE = "#0072B2", "#E69F00", "#CC79A7", "#9aa5ad"   # Okabe-Ito: Descriptron / references / baseline


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=8.6, color=INK, pad=6, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validation_dir", required=True); ap.add_argument("--geomorph_csv", required=True)
    ap.add_argument("--measure_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--semilandmark_dir", required=True,
                    help="output of validate_semilandmarks_vs_geomorph.py (geomorph_pdist.csv + descriptron_pdist.csv)")
    ap.add_argument("--colour_structure_csv", required=True,
                    help="colour_species_structure.csv from validation/colour_species_structure_v1.py")
    ap.add_argument("--descriptron_features", required=True,
                    help="whole-wing colour-homology features (color_homology_features_whole_wing_*.csv)")
    a = ap.parse_args()
    V = Path(a.validation_dir); vdir = Path(a.measure_dir) / "validation"
    report = []

    fig = plt.figure(figsize=(7.2, 9.9))
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 0.95], hspace=0.66, wspace=1.1)

    # ---- a: segmentation, SAM2-PAL vs SST (mean IoU against hand masks)
    ax = fig.add_subplot(gs[0, 0:3]); style(ax, "a  Segmentation (mean IoU)")
    seg = pd.read_csv(V / "segmentation_benchmark" / "segmentation_benchmark_summary.csv").set_index(["group", "arm"])
    arms = [("sst_oneshot", "SST, as distributed", C_REF2), ("sst_perimage", "SST, one image at a time", C_REF),
            ("pal_ft1_orient", "SAM2-PAL, 1 template", C_DESC), ("pal_ft5_orient", "SAM2-PAL, 5 templates", "#56B4E9")]
    groups = [("forewing", "Forewing\n(63 wings, 10 structures)"), ("rostrum", "Rostrum\n(40 rostra, 2 segments)")]
    w = 0.19
    for j, (arm, lab, col) in enumerate(arms):
        vals = [seg.loc[(g, arm), "mean"] for g, _ in groups]
        xs = np.arange(len(groups)) + (j - 1.5) * w
        ax.bar(xs, vals, w * 0.92, color=col, label=lab, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=5.6, color=INK)
        report.append(("a", arm, vals))
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[1] for g in groups], fontsize=6.8)
    ax.set_ylim(0, 1.28); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("mean IoU", fontsize=7.2)
    ax.legend(fontsize=5.8, frameon=False, loc="upper left", ncol=2, handlelength=1.0, columnspacing=0.8)
    ax.text(0.99, -0.30, "SAM2-PAL with orientation search", transform=ax.transAxes, ha="right", fontsize=5.8, color=MUTED)

    # ---- b: landmarks, wings as photographed
    ax = fig.add_subplot(gs[0, 3:6]); style(ax, "b  Landmarks, wings as photographed")
    kp = pd.read_csv(V / "KEYPOINT_BENCHMARK_ALL_CONDITIONS.csv")
    lines = [("as photographed, DINOLand + orientation search", "DINOLand", "DINOLand + orientation search", C_DESC, "-", (3, -9)),
             ("as photographed, no orientation search", "DINOLand", "DINOLand, no orientation search", "#56B4E9", "--", (3, 4)),
             ("as photographed, no orientation search", "Keypoint R-CNN", "Keypoint R-CNN (trained)", C_REF, "-", (-9, -9))]
    for cond, meth, lab, col, ls, off in lines:
        t = kp[(kp.condition == cond) & (kp.method == meth)].sort_values("n_train")
        ax.plot(t.n_train, t.median_pct, ls, color=col, marker="o", ms=3.4, lw=1.4, label=lab, zorder=3)
        for x, y, f in zip(t.n_train, t.median_pct, t.failing_wings):
            ax.annotate(f"{int(f)}", (x, y), textcoords="offset points", xytext=off, fontsize=5.8, color=col, weight="bold")
        report.append(("b", lab, list(zip(t.n_train, t.median_pct, t.failing_wings))))
    ax.set_xscale("log"); ax.set_xlim(0.8, 100); ax.set_ylim(0, 14); ax.set_xticks([1, 5, 20, 76]); ax.set_xticklabels(["1", "5", "20", "76"])
    ax.set_xlabel("labelled training wings", fontsize=7.2); ax.set_ylabel("median error (% wing length)", fontsize=7.2)
    ax.legend(fontsize=5.8, frameon=False, loc="upper right", handlelength=1.8)
    ax.text(0.99, -0.30, "small numbers: wings failed (of 20; median error > 20%)", transform=ax.transAxes,
            ha="right", fontsize=5.8, color=MUTED)

    # ---- c: colour pattern, species separation (same computation as validation_colour_texture_figures.py)
    ax = fig.add_subplot(gs[1, 0:3]); style(ax, "c  Colour pattern: species")
    cf = load(vdir / "validation_colour_texture_figures.py", "cfig")
    cdir = V / "colour_benchmark"; res = cdir / "results"
    sp = pd.read_csv(cdir / "inputs" / "specimens.csv").sort_values("order")
    F = pd.read_csv(a.descriptron_features)
    X = sp[["filename", "species"]].merge(F, on="filename", how="left"); y = X.species.values
    feats = [c for c in F.columns if c != "filename"]
    M = X[feats].astype(float); M = M.loc[:, M.notna().mean() > 0.9]; M = M.fillna(M.mean()); M = M.loc[:, M.std() > 0]
    from sklearn.decomposition import PCA
    P_std = PCA().fit_transform(((M - M.mean()) / M.std()).values)
    others = {}
    for m in ("patternize", "Colormesh"):
        t = pd.read_csv(res / f"pc_scores_{m}.csv"); t = t.set_index("id").loc[[Path(f).stem for f in X.filename.str.split(".tif").str[0] + ".tif"]]
        assert (t.species.values == y).all(); others[m] = t[[c for c in t.columns if c.startswith("PC")]].values
    base = X[[c for c in feats if c.endswith("_L_mean_abs")]].mean(1).to_frame("L").assign(
        a=X[[c for c in feats if c.endswith("_a_mean_abs")]].mean(1), b=X[[c for c in feats if c.endswith("_b_mean_abs")]].mean(1)).values
    base = (base - base.mean(0)) / base.std(0)
    meths = [("Descriptron", P_std, C_DESC), ("patternize", others["patternize"], C_REF),
             ("Colormesh", others["Colormesh"], C_REF2), ("mean wing\ncolour only", base, C_BASE)]
    cs = pd.read_csv(a.colour_structure_csv).set_index("method")
    bw = 0.27
    for i, (lab, S, col) in enumerate(meths):
        key = lab.split("\n")[0].replace("mean wing", "mean wing colour only").split(" colour only colour")[0]
        key = "mean wing colour only" if i == 3 else key
        r = cs.loc[key]
        acc, n = r.loo_acc_PC1to5, int(r.loo_scored)
        assert abs(acc - cf.loo(S[:, :5], y)[0]) < 1e-12          # the same computation as the S13 figure
        bars = [(acc, dict(color=col), "solid")]
        if i < 3:
            bars += [(r.loo_acc_allPCs, dict(color=col, alpha=0.35), "all"),
                     (r.species_R2_beyond_mean_colour, dict(color="white", edgecolor=col, hatch="////", lw=0.9), "r2")]
        k = len(bars)
        for b, (v, kw, kind) in enumerate(bars):
            x0 = i + (b - (k - 1) / 2) * bw
            ax.bar(x0, v, bw, zorder=3, **kw)
            top = v
            if kind != "r2":
                lo, hi = cf.wilson(v, n)
                ax.errorbar(x0, v, yerr=[[v - lo], [hi - v]], color=INK, lw=0.6, capsize=1.2, zorder=4); top = hi
            ax.text(x0, top + 0.012, f"{v:.2f}", fontsize=5.0, color=INK, va="bottom", ha="center")
        report.append(("c", key, acc, r.loo_acc_allPCs, r.get("species_R2_beyond_mean_colour", np.nan)))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MUTED, label="species identified, PC1–5 (leave-one-out nearest neighbour)"),
                       Patch(facecolor=MUTED, alpha=0.35, label="species identified, all PCs"),
                       Patch(facecolor="white", edgecolor=MUTED, hatch="////",
                             label=f"species share of pattern variation beyond mean colour\n(PERMANOVA R², PC1–10; all p = {cs.p_beyond.dropna().max():.4f})")],
              fontsize=5.0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=1, handlelength=1.4)
    chance = 1 / len(set(y))
    ax.axhline(chance, color=MUTED, lw=0.7, ls=":"); ax.text(-0.42, chance + 0.015, "chance", fontsize=5.6, color=MUTED, ha="left")
    ax.set_xticks(range(len(meths))); ax.set_xticklabels([m[0] for m in meths], fontsize=6.8)
    ax.set_ylim(0, 1.45); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("proportion\n(48 wings, 9 species)", fontsize=7.0)

    # ---- d: texture, GLCM identical to scikit-image, grey-level invariance
    ax = fig.add_subplot(gs[1, 3:6]); style(ax, "d  Texture: grey-level invariance")
    tv = load(vdir / "validate_texture_glcm.py", "tglcm")
    tex_script = Path(a.measure_dir) / "texture_phenomics_homology.py"
    img_glob = str(V / "keypoint_benchmark_reoriented" / "crops" / "*.png")
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "t.csv"; buf = io.StringIO()
        sys.argv = ["x", "--image_glob", img_glob, "--texture_script", str(tex_script), "--out_csv", str(out_csv)]
        with contextlib.redirect_stdout(buf):
            tv.main()
        tt = pd.read_csv(out_csv); glcm_line = buf.getvalue().splitlines()[0]
    names = ["contrast", "homogeneity", "energy"]
    inv = [tt[tt.feature == n].invariant_ratio_32_vs_8.median() for n in names]
    std = [tt[tt.feature == n].standard_ratio_32_vs_8.median() for n in names]
    xs = np.arange(3)
    ax.bar(xs - 0.19, inv, 0.36, color=C_DESC, label="invariant formulas (Descriptron)", zorder=3)
    ax.bar(xs + 0.19, std, 0.36, color=C_BASE, label="standard Haralick formulas", zorder=3)
    for x, v in zip(xs - 0.19, inv):
        ax.text(x, v * 1.12, f"{v:.2f}", ha="center", fontsize=5.8, color=INK)
    for x, v in zip(xs + 0.19, std):
        ax.text(x, max(v, 1.0) * 1.12, f"{v:.2f}", ha="center", fontsize=5.8, color=INK)
    ax.axhline(1, color=INK, lw=0.7, ls="--"); ax.set_yscale("log"); ax.set_ylim(0.2, 60)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=6.8)
    ax.set_ylabel("value at 32 / at 8 grey levels\n(1 = invariant)", fontsize=7.0)
    ax.legend(fontsize=5.8, frameon=False, loc="upper right")
    nmat = glcm_line.split("over")[1].split("matrices")[0].strip()
    maxd = glcm_line.split("|difference|")[1].split("over")[0].strip()
    ax.text(0.0, -0.30, f"co-occurrence matrices identical to scikit-image\n(largest difference {float(maxd):.0f}, {nmat} matrices)",
            transform=ax.transAxes, fontsize=5.8, color=MUTED)
    report.append(("d", dict(zip(names, inv)), dict(zip(names, std)), glcm_line))

    # ---- e: shape statistics against geomorph
    ax = fig.add_subplot(gs[2, :]); style(ax, "e  Shape statistics: Descriptron against geomorph 4.1.1 (same raw landmarks)")
    gv = load(vdir / "validate_shape_stats_vs_geomorph.py", "gval")
    g = pd.read_csv(a.geomorph_csv)
    g = g[(g.projection == "noproj") & (g.statistic.isin(gv.DETERMINISTIC))].copy()
    g["rel"] = (g.python - g.R).abs() / g.R.abs().clip(lower=1e-12)
    labels = {"centroid_size": "centroid\nsize", "procrustes_distance": "Procrustes\ndistance",
              "procrustes_variance": "Procrustes\nvariance", "difference": "disparity\ndifferences",
              "SS": "ANOVA\nSS", "Rsq": "ANOVA\nR²", "F": "ANOVA\nF", "CR": "modularity\nCR",
              "r_PLS": "two-block\nPLS r", "Kmult": "Kmult", "individual_asymmetry": "individual\nasymmetry"}
    order = [s for s in labels if s in set(g.statistic)]
    floor = 1e-16
    rng = np.random.default_rng(0)
    for i, s in enumerate(order):
        v = g[g.statistic == s].rel.clip(lower=floor).values
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=5, color=C_DESC, alpha=0.35, lw=0, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [v.max()] * 2, color=INK, lw=1.0, zorder=4)
    worst = g.rel.max()
    ax.axhline(worst, color=C_REF, lw=0.8, ls="--")
    ax.text(len(order) - 0.5, worst * 2.2, f"largest relative difference {worst:.1e} ({len(g):,} values)",
            ha="right", fontsize=6.2, color=INK)
    ax.set_yscale("log"); ax.set_ylim(floor / 3, 1e-1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([labels[s] for s in order], fontsize=6.0)
    ax.set_ylabel("|Descriptron − geomorph|\n/ |geomorph|", fontsize=7.0)
    ax.text(0.0, -0.42, "dots: individual values (identical values plotted at 10⁻¹⁶); bars: largest per statistic; "
            "permutation P-values not shown (they differ by random sampling)", transform=ax.transAxes, fontsize=5.8, color=MUTED)
    report.append(("e", worst, len(g)))

    # ---- f-h: one against the other, on the line of identity
    def identity(ax, pairs, title, xlab, ylab, log=False):
        style(ax, title); ax.yaxis.grid(False)
        allv = np.concatenate([np.r_[x, y] for x, y, *_ in pairs])
        lo_, hi_ = allv.min(), allv.max(); pad = (hi_ - lo_) * 0.04
        lims = (lo_ / 1.3, hi_ * 1.3) if log else (lo_ - pad, hi_ + pad)
        ax.plot(lims, lims, color=MUTED, lw=0.7, ls="--", zorder=1)
        for x, y, col, lab, sz in pairs:
            ax.scatter(x, y, s=sz, color=col, alpha=0.55, lw=0, label=lab, zorder=3)
        if log:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlab, fontsize=6.6); ax.set_ylabel(ylab, fontsize=6.6); ax.tick_params(labelsize=6)
        rel = max(np.max(np.abs(y - x) / np.abs(x)) for x, y, *_ in pairs)
        ax.text(0.04, 0.96, f"max rel. diff.\n{rel:.1e}", transform=ax.transAxes, fontsize=5.8, va="top", color=INK)
        ax.legend(fontsize=5.2, frameon=False, loc="lower right", handletextpad=0.2, markerscale=1.5)
        return rel
    gall = pd.read_csv(a.geomorph_csv); gall = gall[gall.projection == "noproj"]
    def pts(stat):
        t = gall[gall.statistic == stat]
        real = t[t.dataset == "real"]; syn = t[t.dataset != "real"]
        return [(syn.R.values, syn.python.values, C_BASE, "synthetic designs", 5),
                (real.R.values, real.python.values, C_DESC, "Diaphorina forewings", 5)]
    ax = fig.add_subplot(gs[3, 0:2])
    rf = identity(ax, pts("centroid_size"), "f  Size", "geomorph centroid size", "Descriptron")
    ax = fig.add_subplot(gs[3, 2:4])
    rg = identity(ax, pts("procrustes_distance"), "g  Landmark shape", "geomorph Procrustes distance", "Descriptron")
    sd = Path(a.semilandmark_dir)
    gd = pd.read_csv(sd / "geomorph_pdist.csv"); dd = pd.read_csv(sd / "descriptron_pdist.csv")
    ax = fig.add_subplot(gs[3, 4:6])
    rh = identity(ax, [(gd.slid_bending.values, dd.slid_bending.values, C_REF2, "sliding, bending energy", 7),
                       (gd.slid_procd.values, dd.slid_procd.values, C_REF, "sliding, Procrustes distance", 4),
                       (gd.fixed.values, dd.fixed.values, C_DESC, "fixed", 1.5)],
                  "h  Outline semilandmarks", "geomorph Procrustes distance", "Descriptron")
    report.append(("f-h", rf, rg, rh, len(gd)))

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    with open(out.with_suffix(".values.txt"), "w") as fh:
        for r in report:
            print(r, file=fh); print(r)
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
