#!/usr/bin/env python3
"""
validate_shape_stats_vs_rrpp.py — Descriptron's Python shape statistics side by side with R's RRPP
===================================================================================================

Runs identical Procrustes-aligned data and identical models through descriptron_shape_stats.py and through
RRPP (Collyer & Adams), then plots every statistic from both on identity axes and writes the paired values.

Datasets: (1) a seeded synthetic design with known effects — 3 species x 3 localities (6 each), a species x
locality interaction for one species, elevation; (2) optionally real landmark data (--real COCO, with
--real-meta / --real-schema), e.g. the Diaphorina forewings with species labels.

R is needed only to run this check, not to use Descriptron:
    python validate_shape_stats_vs_rrpp.py --rscript /path/to/Rscript --rlib /path/to/Rlib --out-dir val/ \\
        [--real forewings.json --real-meta meta.csv --real-schema schema.json --real-formula "species*sex"]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import descriptron_metadata as dm       # noqa: E402
import descriptron_shape_stats as ss    # noqa: E402

R_CODE = r'''
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], .libPaths()))
suppressMessages(library(RRPP))
d <- read.csv(args[2], check.names = FALSE); out <- args[3]; formula_txt <- args[4]; traj <- args[5]; group <- args[6]
Y <- as.matrix(d[, grep("^y[0-9]+$", names(d))])
vars <- setdiff(names(d), grep("^y[0-9]+$", names(d), value = TRUE))
df <- rrpp.data.frame(Y = Y)
for (v in vars) df[[v]] <- if (is.numeric(d[[v]])) d[[v]] else factor(d[[v]])
set.seed(1)
tab <- function(fit) { a <- anova(fit)$table; data.frame(term = rownames(a), a, check.names = FALSE) }
fit <- lm.rrpp(as.formula(paste("Y ~", formula_txt)), data = df, iter = 999, SS.type = "I", print.progress = FALSE)
write.csv(tab(fit), file.path(out, "R_anova.csv"), row.names = FALSE)
fa <- lm.rrpp(as.formula(paste("Y ~ logCS *", group)), data = df, iter = 999, print.progress = FALSE)
write.csv(tab(fa), file.path(out, "R_allometry.csv"), row.names = FALSE)
fg <- lm.rrpp(as.formula(paste("Y ~", group)), data = df, iter = 999, print.progress = FALSE)
pw <- summary(pairwise(fg, groups = df[[group]]), test.type = "var")
write.csv(data.frame(pair = rownames(pw$summary.table), pw$summary.table, check.names = FALSE), file.path(out, "R_disparity.csv"), row.names = FALSE)
if (traj != "none") {
  ft <- lm.rrpp(as.formula(paste("Y ~", group, "*", traj)), data = df, iter = 999, print.progress = FALSE)
  TA <- trajectory.analysis(ft, groups = df[[group]], traj.pts = df[[traj]], print.progress = FALSE)
  md <- summary(TA, attribute = "MD")$summary.table; tc <- summary(TA, attribute = "TC", angle.type = "deg")$summary.table
  write.csv(data.frame(pair = rownames(md), md, check.names = FALSE), file.path(out, "R_traj_MD.csv"), row.names = FALSE)
  write.csv(data.frame(pair = rownames(tc), tc, check.names = FALSE), file.path(out, "R_traj_TC.csv"), row.names = FALSE)
}
cat("R done\n")
'''


def synthetic(out: Path):
    rng = np.random.default_rng(3)
    base = np.array([[0, 0], [40, 5], [80, 0], [100, 30], [80, 60], [40, 65], [0, 60], [-15, 30]], float)
    sp_eff = {"sp_A": np.zeros((8, 2)), "sp_B": rng.normal(0, 2.0, (8, 2)), "sp_C": rng.normal(0, 2.0, (8, 2))}
    loc_eff = {"L1": np.zeros((8, 2)), "L2": rng.normal(0, 1.2, (8, 2)), "L3": rng.normal(0, 1.2, (8, 2))}
    X, meta = [], []
    for sp in sp_eff:
        for loc in loc_eff:
            for _ in range(6):
                elev = {"L1": 300, "L2": 1200, "L3": 2100}[loc] + rng.normal(0, 100)
                P = base + sp_eff[sp] + loc_eff[loc] + (loc_eff[loc] * 1.5 if sp == "sp_C" else 0) + rng.normal(0, 0.8, (8, 2))
                a = rng.uniform(0, 6.28)
                X.append((np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ P.T).T * rng.uniform(4, 6) + 500)
                meta.append({"species": sp, "locality": loc, "elevation": round(elev)})
    return np.array(X), meta


def aligned_table(X, meta, cols, path):
    refl = ss.mirrored_specimens(X)
    if refl.any():
        X = X.copy(); X[refl, :, 0] *= -1
    Z, cs = ss.gpa(X)
    Y = ss.flat(Z)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols + ["logCS"] + [f"y{i}" for i in range(Y.shape[1])])
        for m, y, c in zip(meta, Y, cs):
            w.writerow([m[k] for k in cols] + [math.log(c)] + list(y))
    return Y, np.log(cs), int(refl.sum())


def python_side(Y, logcs, meta, formula, group, traj, iters=999):
    rng = np.random.default_rng(1)
    kinds = {k: ("continuous" if isinstance(meta[0][k], (int, float)) else "factor") for k in meta[0]}
    data = {k: [m[k] for m in meta] for k in meta[0]}
    kinds["logCS"] = "continuous"; data["logCS"] = list(logcs)
    res = {"anova": ss.procrustes_anova(Y, ss.Design(formula, data, kinds), iters, rng),
           "allometry": ss.procrustes_anova(Y, ss.Design(f"logCS*{group}", data, kinds), iters, rng)}
    res["disparity"] = ss.disparity(Y, data[group], iters, rng)[1]
    if traj:
        res["trajectory"] = ss.trajectory(Y, data[group], data[traj], iters, rng)["pairs"]
    return res


def read_csv(p):
    return list(csv.DictReader(open(p)))


def pair_up(name, py, rdir, has_traj):
    """-> list of dicts: dataset, analysis, statistic, label, python, R."""
    rows = []
    def add(analysis, stat, label, a, b):
        try:
            rows.append({"dataset": name, "analysis": analysis, "statistic": stat, "label": label,
                         "python": float(a), "R": float(b)})
        except (TypeError, ValueError):
            pass
    for key, fn in (("anova", "R_anova.csv"), ("allometry", "R_allometry.csv")):
        R = {r["term"]: r for r in read_csv(rdir / fn)}
        for r in py[key]:
            t = r["term"]
            if t in R and t not in ("Total",):
                for stat, rc in (("SS", "SS"), ("Rsq", "Rsq"), ("F", "F"), ("Z", "Z"), ("P", "Pr(>F)")):
                    if r.get(stat) not in ("", None) and R[t].get(rc) not in ("", None, "NA"):
                        add(key, stat, t, r[stat], R[t][rc])
    Rd = {r["pair"]: r for r in read_csv(rdir / "R_disparity.csv")}
    for r in py["disparity"]:
        k = f"{r['group_1']}:{r['group_2']}"
        if k in Rd:
            add("disparity", "difference", k, r["difference"], Rd[k]["d"]); add("disparity", "P", k, r["P"], Rd[k]["Pr > d"])
    if has_traj:
        md = {r["pair"]: r for r in read_csv(rdir / "R_traj_MD.csv")}
        tc = {r["pair"]: r for r in read_csv(rdir / "R_traj_TC.csv")}
        for r in py["trajectory"]:
            k = f"{r['group_1']}:{r['group_2']}"
            if k in md:
                add("trajectory", "magnitude", k, r["magnitude_difference"], md[k]["d"])
                add("trajectory", "P_magnitude", k, r["P_magnitude"], md[k]["Pr > d"])
                add("trajectory", "angle_deg", k, r["angle_deg"], tc[k]["angle"])
                add("trajectory", "P_direction", k, r["P_direction"], tc[k]["Pr > angle"])
    return rows


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("Sums of squares", lambda r: r["statistic"] == "SS", True),
              ("R-squared", lambda r: r["statistic"] == "Rsq", False),
              ("F", lambda r: r["statistic"] == "F", True),
              ("Effect size Z", lambda r: r["statistic"] == "Z", False),
              ("Trajectory: magnitude difference and angle (deg)",
               lambda r: r["statistic"] in ("magnitude", "angle_deg"), True),
              ("Disparity: pairwise difference", lambda r: r["analysis"] == "disparity" and r["statistic"] == "difference", True),
              ("P-values (permutation)", lambda r: r["statistic"].startswith("P"), False)]
    colours = {"synthetic": "#2a78d6", "real": "#e8743b"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
    for ax, (title, sel, logscale) in zip(axes.ravel(), panels):
        sub = [r for r in rows if sel(r)]
        for ds in ("synthetic", "real"):
            pts = [(r["R"], r["python"]) for r in sub if r["dataset"] == ds]
            if pts:
                x, y = np.array(pts).T
                if logscale:
                    keep = (x > 0) & (y > 0); x, y = x[keep], y[keep]
                ax.scatter(x, y, s=26, c=colours[ds], alpha=0.8, edgecolors="white", linewidths=0.5, label=ds)
        if sub:
            v = np.array([[r["R"], r["python"]] for r in sub], float)
            v = v[(v > 0).all(1)] if logscale else v
            lo, hi = v.min(), v.max()
            ax.plot([lo, hi], [lo, hi], color="#555", lw=0.8, ls="--")
        if logscale:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title, fontsize=10); ax.set_xlabel("R (RRPP)"); ax.set_ylabel("Python (Descriptron)")
    ax = axes.ravel()[-1]; ax.axis("off")
    det = [r for r in rows if r["statistic"] in ("SS", "Rsq", "F", "magnitude", "angle_deg") or
           (r["analysis"] == "disparity" and r["statistic"] == "difference")]
    rel = max(abs(r["python"] - r["R"]) / max(abs(r["R"]), 1e-12) for r in det) if det else float("nan")
    ps = [r for r in rows if r["statistic"].startswith("P")]
    ax.text(0, 0.95, "Deterministic statistics\n(SS, R2, F, trajectory distance and angle,\ndisparity):\n"
            f"largest relative difference {rel:.1e}\n({len(det)} values)\n\n"
            f"Permutation-based values (P, Z) differ only\nby random sampling ({len(ps)} P-values).\n\n"
            "Same Procrustes-aligned data and models in both;\nRRPP 999 permutations, Descriptron 999.",
            va="top", fontsize=10, family="monospace")
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Descriptron shape statistics (Python) vs RRPP (R), identical data", fontsize=13)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return rel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rscript", required=True); ap.add_argument("--rlib", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--real"); ap.add_argument("--real-meta"); ap.add_argument("--real-schema")
    ap.add_argument("--real-formula", default=None); ap.add_argument("--real-category")
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "rrpp_side.R").write_text(R_CODE)
    rows = []
    # synthetic
    X, meta = synthetic(out)
    d = out / "synthetic"; d.mkdir(exist_ok=True)
    Y, lcs, _ = aligned_table(X, meta, ["species", "locality", "elevation"], d / "aligned.csv")
    py = python_side(Y, lcs, meta, "species*locality + elevation", "species", "locality")
    subprocess.run([a.rscript, str(out / "rrpp_side.R"), a.rlib, str(d / "aligned.csv"), str(d),
                    "species * locality + elevation", "locality", "species"], check=True)
    rows += pair_up("synthetic", py, d, True)
    # real
    if a.real:
        files, Xr, _ = ss.load_shapes(a.real, a.real_category)
        cols, trows = dm.load_table(a.real_meta); schema = json.load(open(a.real_schema))
        links, _ = dm.link_images(trows, schema, files)
        group = next(c for c, r in schema.items() if r == "group")
        keep = [i for i, f in enumerate(files) if f in links and trows[links[f]].get(group)]
        metar = [{group: trows[links[files[i]]][group]} for i in keep]
        for c, r in schema.items():
            if r == "factor":
                for m, i in zip(metar, keep):
                    m[c] = trows[links[files[i]]].get(c, "") or "NA"
        # groups with a single specimen cannot enter pairwise variance comparisons
        from collections import Counter
        cnt = Counter(m[group] for m in metar)
        ok = [j for j, m in enumerate(metar) if cnt[m[group]] >= 2]
        metar = [metar[j] for j in ok]; Xr = Xr[[keep[j] for j in ok]]
        d = out / "real"; d.mkdir(exist_ok=True)
        cols_r = list(metar[0])
        Y, lcs, nrefl = aligned_table(Xr, metar, cols_r, d / "aligned.csv")
        f = a.real_formula or group
        py = python_side(Y, lcs, metar, f, group, None)
        subprocess.run([a.rscript, str(out / "rrpp_side.R"), a.rlib, str(d / "aligned.csv"), str(d), f.replace("*", " * "),
                        "none", group], check=True)
        rows += pair_up("real", py, d, False)
        print(f"real data: {len(metar)} specimens, {nrefl} mirror images reflected")
    with open(out / "python_vs_R_values.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rel = figure(rows, out / "python_vs_R.png")
    print(f"{len(rows)} paired values; largest relative difference of deterministic statistics {rel:.2e}")
    print(f"figure {out/'python_vs_R.png'}; values {out/'python_vs_R_values.csv'}")


if __name__ == "__main__":
    main()
