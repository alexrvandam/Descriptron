#!/usr/bin/env python3
"""
validate_shape_stats_vs_geomorph.py — Descriptron's shape statistics side by side with R's geomorph
===================================================================================================

Unlike validate_shape_stats_vs_rrpp.py (identical ALIGNED data into both), this gives both sides the same RAW
landmarks, so the Procrustes superimposition itself is compared too: geomorph's gpagen vs Descriptron's gpa, then
procD.lm, morphol.disparity, modularity.test (CR), two.b.pls, physignal (Kmult) and bilat.symmetry (object symmetry)
against their Descriptron counterparts, each side on its own aligned shapes.

geomorph projects aligned shapes into tangent space by default (gpagen Proj = TRUE); Descriptron does not. The main
comparison uses gpagen(Proj = FALSE); the size of the tangent-projection difference is measured and reported separately.

Datasets (seeded synthetic, known effects): a 3 species x 3 localities design with elevation and temperature
covariates and two landmark modules; 16 taxa evolving by Brownian motion on a tree; bilaterally symmetric
configurations with directional asymmetry. Optionally real landmarks (--real COCO + --real-meta/--real-schema);
mirror-image specimens are reflected BEFORE both sides (geomorph has no mirror detection).

R is needed only to run this check, not to use Descriptron:
    python validate_shape_stats_vs_geomorph.py --rscript /path/to/Rscript --rlib /path/to/Rlib --out-dir val/ \\
        [--real forewings.json --real-meta meta.csv --real-schema schema.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import descriptron_metadata as dm       # noqa: E402
import descriptron_shape_stats as ss    # noqa: E402

ITER = 999

R_CODE = r'''
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], .libPaths()))
suppressMessages({library(geomorph); library(ape)})
d <- args[2]
raw <- as.matrix(read.csv(file.path(d, "raw.csv")))
meta <- read.csv(file.path(d, "meta.csv"), stringsAsFactors = TRUE, check.names = FALSE)
task <- readLines(file.path(d, "task.txt"))
kv <- setNames(sub("^[^=]*=", "", task), sub("=.*", "", task))
has <- function(k) !is.na(kv[k]) && nzchar(kv[k])
p <- ncol(raw) / 2; n <- nrow(raw)
A <- arrayspecs(raw, p, 2)
wr <- function(x, name) write.csv(x, file.path(d, name), row.names = FALSE)
tab <- function(fit) { a <- anova(fit)$table; data.frame(term = rownames(a), a, check.names = FALSE) }
for (proj in c(FALSE, TRUE)) {
  tag <- if (proj) "proj" else "noproj"
  g <- gpagen(A, Proj = proj, print.progress = FALSE)
  wr(data.frame(Csize = g$Csize), paste0("G_csize_", tag, ".csv"))
  wr(as.matrix(dist(two.d.array(g$coords))), paste0("G_pdist_", tag, ".csv"))
  gdf <- geomorph.data.frame(coords = g$coords, Csize = g$Csize)
  for (v in names(meta)) gdf[[v]] <- meta[[v]]
  if (has("formula")) {
    set.seed(1)
    wr(tab(procD.lm(as.formula(paste("coords ~", kv["formula"])), data = gdf, iter = 999, SS.type = "I",
                    print.progress = FALSE)), paste0("G_anova_", tag, ".csv"))
  }
  if (has("group")) {
    set.seed(1)
    wr(tab(procD.lm(as.formula(paste("coords ~ log(Csize) *", kv["group"])), data = gdf, iter = 999,
                    print.progress = FALSE)), paste0("G_allometry_", tag, ".csv"))
    set.seed(1)
    md <- morphol.disparity(as.formula(paste("coords ~", kv["group"])), groups = as.formula(paste("~", kv["group"])), data = gdf, iter = 999,
                            print.progress = FALSE)
    wr(data.frame(group = names(md$Procrustes.var), pv = as.numeric(md$Procrustes.var)), paste0("G_pv_", tag, ".csv"))
    D <- md$PV.dist; P <- md$PV.dist.Pval; rows <- NULL
    for (i in seq_len(nrow(D))) for (j in seq_len(ncol(D))) if (i < j)
      rows <- rbind(rows, data.frame(pair = paste(rownames(D)[i], colnames(D)[j], sep = ":"), d = D[i, j], P = P[i, j]))
    wr(rows, paste0("G_pvdist_", tag, ".csv"))
  }
  if (has("modules")) {
    set.seed(1)
    gp <- as.integer(strsplit(kv["modules"], ",")[[1]])
    mt <- modularity.test(g$coords, partition.gp = gp, iter = 999, CI = FALSE, opt.rot = FALSE, print.progress = FALSE)
    set.seed(1)
    mo <- modularity.test(g$coords, partition.gp = gp, iter = 999, CI = FALSE, print.progress = FALSE)
    wr(data.frame(CR = mt$CR, P = mt$P.value, CR_default_opt_rot = mo$CR), paste0("G_modularity_", tag, ".csv"))
  }
  if (has("pls")) {
    set.seed(1)
    X2 <- as.matrix(meta[, strsplit(kv["pls"], ",")[[1]], drop = FALSE])
    tb <- two.b.pls(g$coords, X2, iter = 999, print.progress = FALSE)
    wr(data.frame(r = tb$r.pls, P = tb$P.value), paste0("G_pls_", tag, ".csv"))
  }
  if (has("tree")) {
    set.seed(1)
    phy <- read.tree(file.path(d, kv["tree"]))
    C <- g$coords; dimnames(C)[[3]] <- as.character(meta$taxon)
    ps <- physignal(C, phy, iter = 999, print.progress = FALSE)
    wr(data.frame(K = ps$phy.signal, P = ps$pvalue), paste0("G_kmult_", tag, ".csv"))
  }
}
if (has("pairs")) {
  pr <- matrix(as.integer(strsplit(kv["pairs"], ",")[[1]]), ncol = 2, byrow = TRUE)
  set.seed(1)
  bs <- bilat.symmetry(A, ind = factor(seq_len(n)), object.sym = TRUE, land.pairs = pr, iter = 999,
                       print.progress = FALSE)
  # unsigned.AI adds the mean shape back in (values near 1); take the asymmetry component itself
  mn <- mshape(bs$symm.shape)
  D <- two.d.array(bs$asymm.shape) - matrix(as.vector(t(mn)), nrow = n, ncol = 2 * p, byrow = TRUE)
  wr(data.frame(unsigned_AI = as.numeric(bs$unsigned.AI), asym_component_norm = sqrt(rowSums(D^2))), "G_asym_AI.csv")
}
cat("geomorph done\n")
'''


def _rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _scatter_raw(P, rng):
    """random position, rotation and scale: what GPA has to remove"""
    return (_rot(rng.uniform(0, 6.28)) @ P.T).T * rng.uniform(4, 6) + rng.uniform(200, 800, 2)


def design_dataset():
    rng = np.random.default_rng(3)
    base = np.array([[0, 0], [40, 5], [80, 0], [100, 30], [80, 60], [40, 65], [0, 60], [-15, 30]], float)
    sp_eff = {"sp_A": np.zeros((8, 2)), "sp_B": rng.normal(0, 2.0, (8, 2)), "sp_C": rng.normal(0, 2.0, (8, 2))}
    loc_eff = {"L1": np.zeros((8, 2)), "L2": rng.normal(0, 1.2, (8, 2)), "L3": rng.normal(0, 1.2, (8, 2))}
    elev_dir = rng.normal(0, 1.0, (8, 2))
    X, meta = [], []
    for sp in sp_eff:
        for loc in loc_eff:
            for _ in range(6):
                elev = {"L1": 300, "L2": 1200, "L3": 2100}[loc] + rng.normal(0, 100)
                temp = 25 - elev / 180 + rng.normal(0, 1)
                P = base + sp_eff[sp] + loc_eff[loc] + (loc_eff[loc] * 1.5 if sp == "sp_C" else 0) \
                    + elev_dir * (elev - 1200) / 900 + rng.normal(0, 0.8, (8, 2))
                P[:4] += rng.normal(0, 1.0) * np.array([[0, 1], [0, 1], [0, 1], [0, 1]])   # module 1 moves together
                X.append(_scatter_raw(P, rng))
                meta.append({"species": sp, "locality": loc, "elevation": round(elev), "temperature": round(temp, 2)})
    task = {"formula": "species * locality + elevation", "group": "species",
            "modules": "1,1,1,1,2,2,2,2", "pls": "elevation,temperature"}
    return np.array(X), meta, task


def tree_dataset():
    rng = np.random.default_rng(11)
    nwk = "((((t1:1,t2:1):1,(t3:1,t4:1):1):1,((t5:1,t6:1):1,(t7:1,t8:1):1):1):1," \
          "(((t9:1,t10:1):1,(t11:1,t12:1):1):1,((t13:1,t14:1):1,(t15:1,t16:1):1):1):1);"
    tips, C = ss.parse_newick(nwk)
    L = np.linalg.cholesky(C)
    base = np.array([[0, 0], [40, 5], [80, 0], [100, 30], [80, 60], [40, 65], [0, 60], [-15, 30]], float)
    E = (L @ rng.normal(0, 1.5, (len(tips), 16))).reshape(len(tips), 8, 2)
    X = np.array([_scatter_raw(base + e, rng) for e in E])
    return X, [{"taxon": t} for t in tips], {"tree": "tree.nwk"}, nwk


def symmetry_dataset():
    rng = np.random.default_rng(5)
    sym = np.array([[-30, 0], [-20, 30], [0, 40], [20, 30], [30, 0], [0, -20]], float)   # 1<->5, 2<->4
    X = []
    for _ in range(25):
        P = sym + rng.normal(0, 0.6, sym.shape)
        P[0, 1] += 3.0                                                                  # directional asymmetry
        X.append(_scatter_raw(P, rng))
    return np.array(X), [{"ind": f"i{k}"} for k in range(25)], {"pairs": "1,5,2,4"}, [(1, 5), (2, 4)]


def write_inputs(d: Path, X, meta, task, extra=None):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "raw.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow([f"{c}{i + 1}" for i in range(X.shape[1]) for c in "xy"])
        for P in X:
            w.writerow([f"{v:.10f}" for v in P.ravel()])
    with open(d / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0])); w.writeheader(); w.writerows(meta)
    (d / "task.txt").write_text("\n".join(f"{k}={v}" for k, v in task.items()) + "\n")
    for name, text in (extra or {}).items():
        (d / name).write_text(text)


def python_side(X, meta, task, pairs=None, nwk=None):
    rng = np.random.default_rng(1)
    Z, cs = ss.gpa(X)
    Y = ss.flat(Z)
    iu = np.triu_indices(len(Y), 1)
    res = {"csize": cs, "pdist": np.linalg.norm(Y[:, None] - Y[None], axis=2)[iu]}
    kinds = {k: ("continuous" if isinstance(meta[0][k], (int, float)) else "factor") for k in meta[0]}
    data = {k: [m[k] for m in meta] for k in meta[0]}
    kinds["logCS"] = "continuous"; data["logCS"] = list(np.log(cs))
    if "formula" in task:
        res["anova"] = ss.procrustes_anova(Y, ss.Design(task["formula"].replace(" ", ""), data, kinds), ITER, rng)
    if "group" in task:
        g = task["group"]
        res["allometry"] = ss.procrustes_anova(Y, ss.Design(f"logCS*{g}", data, kinds), ITER, rng)
        res["pv"], res["pvdist"] = ss.disparity(Y, data[g], ITER, rng)
    if "modules" in task:
        res["modularity"] = ss.modularity(Y, [int(v) for v in task["modules"].split(",")], ITER, rng)
    if "pls" in task:
        B = np.array([[float(m[c]) for c in task["pls"].split(",")] for m in meta])
        res["pls"] = ss.two_block_pls(Y, B, ITER, rng)
    if nwk:
        tips, C = ss.parse_newick(nwk)
        order = [tips.index(m["taxon"]) for m in meta]
        res["kmult"] = ss.phylosignal(Y, C[np.ix_(order, order)], ITER, rng)
    if pairs:
        res["asym"] = ss.asymmetry(X, pairs, ITER, rng)
    return res


def read_csv(p):
    return list(csv.DictReader(open(p)))


def pair_up(name, py, d: Path):
    """-> list of dicts: dataset, analysis, statistic, label, python, R, projection."""
    rows = []

    def add(analysis, stat, label, a, b, proj="noproj"):
        try:
            rows.append({"dataset": name, "analysis": analysis, "statistic": stat, "label": label,
                         "python": float(a), "R": float(b), "projection": proj})
        except (TypeError, ValueError):
            pass
    for tag in ("noproj", "proj"):
        cs = [float(r["Csize"]) for r in read_csv(d / f"G_csize_{tag}.csv")]
        for i, (a, b) in enumerate(zip(py["csize"], cs)):
            add("gpa", "centroid_size", str(i), a, b, tag)
        M = np.array([[float(v) for v in r.values()] for r in read_csv(d / f"G_pdist_{tag}.csv")])
        for k, (a, b) in enumerate(zip(py["pdist"], M[np.triu_indices(len(M), 1)])):
            add("gpa", "procrustes_distance", str(k), a, b, tag)
        for key, fn in (("anova", "G_anova"), ("allometry", "G_allometry")):
            f = d / f"{fn}_{tag}.csv"
            if key not in py or not f.exists():
                continue
            R = {r["term"].replace("log(Csize)", "logCS"): r for r in read_csv(f)}
            for r in py[key]:
                t = r["term"]
                if t in R and t != "Total":
                    for stat, rc in (("SS", "SS"), ("Rsq", "Rsq"), ("F", "F"), ("Z", "Z"), ("P", "Pr(>F)")):
                        if r.get(stat) not in ("", None) and R[t].get(rc) not in ("", None, "NA"):
                            add(key, stat, t, r[stat], R[t][rc], tag)
        if "pv" in py:
            G = {r["group"]: r for r in read_csv(d / f"G_pv_{tag}.csv")}
            for r in py["pv"]:
                add("disparity", "procrustes_variance", r["group"], r["procrustes_variance"], G[r["group"]]["pv"], tag)
            Gd = {r["pair"]: r for r in read_csv(d / f"G_pvdist_{tag}.csv")}
            for r in py["pvdist"]:
                k = f"{r['group_1']}:{r['group_2']}"
                if k in Gd:
                    add("disparity", "difference", k, r["difference"], Gd[k]["d"], tag)
                    add("disparity", "P", k, r["P"], Gd[k]["P"], tag)
        if "modularity" in py:
            g = read_csv(d / f"G_modularity_{tag}.csv")[0]
            add("modularity", "CR", "CR", py["modularity"]["CR"], g["CR"], tag)
            add("modularity", "P", "CR", py["modularity"]["P_CR (share of random partitions with CR <= observed)"], g["P"], tag)
        if "pls" in py:
            g = read_csv(d / f"G_pls_{tag}.csv")[0]
            add("pls", "r_PLS", "shape~environment", py["pls"]["r_PLS"], g["r"], tag)
            add("pls", "P", "shape~environment", py["pls"]["P"], g["P"], tag)
        if "kmult" in py:
            g = read_csv(d / f"G_kmult_{tag}.csv")[0]
            add("kmult", "Kmult", "Kmult", py["kmult"]["Kmult"], g["K"], tag)
            add("kmult", "P", "Kmult", py["kmult"]["P"], g["P"], tag)
    if "asym" in py and (d / "G_asym_AI.csv").exists():
        # geomorph's asymmetry component is the whole left-right difference; Descriptron reports half of it
        # (the asymmetric component of Klingenberg et al. 2002), so it is doubled here
        ai = [float(r["asym_component_norm"]) for r in read_csv(d / "G_asym_AI.csv")]
        for i, (a, b) in enumerate(zip(py["asym"]["per_specimen"], ai)):
            add("asymmetry", "individual_asymmetry", str(i), 2 * a, b)
    return rows


DETERMINISTIC = ("centroid_size", "procrustes_distance", "SS", "Rsq", "F", "procrustes_variance", "difference",
                 "CR", "r_PLS", "Kmult", "individual_asymmetry")


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main_rows = [r for r in rows if r["projection"] == "noproj"]
    panels = [("GPA: centroid size (gpagen)", lambda r: r["statistic"] == "centroid_size", True),
              ("GPA: Procrustes distances (gpagen)", lambda r: r["statistic"] == "procrustes_distance", True),
              ("procD.lm: SS, R2 and F", lambda r: r["statistic"] in ("SS", "Rsq", "F"), True),
              ("morphol.disparity", lambda r: r["analysis"] == "disparity" and r["statistic"] != "P", True),
              ("modularity.test CR, two.b.pls r,\nphysignal Kmult", lambda r: r["statistic"] in ("CR", "r_PLS", "Kmult"), False),
              ("bilat.symmetry: individual asymmetry\n(Descriptron x2: it reports half)", lambda r: r["statistic"] == "individual_asymmetry", False),
              ("P-values (permutation)", lambda r: r["statistic"] == "P", False)]
    colours = {"design": "#2a78d6", "phylogeny": "#1baf7a", "symmetry": "#8e44ad", "real": "#e8743b"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
    for ax, (title, sel, logscale) in zip(axes.ravel(), panels):
        sub = [r for r in main_rows if sel(r)]
        for ds, col in colours.items():
            pts = [(r["R"], r["python"]) for r in sub if r["dataset"] == ds]
            if pts:
                x, y = np.array(pts).T
                if logscale:
                    keep = (x > 0) & (y > 0); x, y = x[keep], y[keep]
                ax.scatter(x, y, s=24, c=col, alpha=0.8, edgecolors="white", linewidths=0.5, label=ds)
        if sub:
            v = np.array([[r["R"], r["python"]] for r in sub], float)
            v = v[(v > 0).all(1)] if logscale else v
            if len(v):
                lo, hi = v.min(), v.max()
                ax.plot([lo, hi], [lo, hi], color="#555", lw=0.8, ls="--")
        if logscale:
            from matplotlib.ticker import NullFormatter
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=10); ax.set_xlabel("R (geomorph)"); ax.set_ylabel("Python (Descriptron)")
        ax.legend(frameon=False, fontsize=7)
    ax = axes.ravel()[-1]; ax.axis("off")
    det = [r for r in main_rows if r["statistic"] in DETERMINISTIC]
    rel = max(abs(r["python"] - r["R"]) / max(abs(r["R"]), 1e-12) for r in det) if det else float("nan")
    tp = [r for r in rows if r["projection"] == "proj" and r["statistic"] in ("procrustes_distance", "SS", "Rsq", "F")]
    rel_tp = float(np.median([abs(r["python"] - r["R"]) / max(abs(r["R"]), 1e-12) for r in tp])) if tp else float("nan")
    ps = [r for r in main_rows if r["statistic"] == "P"]
    ax.text(0, 0.97, "Same RAW landmarks into both;\neach side does its own GPA.\n\n"
            "Deterministic statistics\n(centroid size, Procrustes distances,\nSS, R2, F, disparity, CR, r-PLS,\n"
            f"Kmult, asymmetry): largest relative\ndifference {rel:.1e} ({len(det)} values)\n\n"
            f"Permutation P-values differ only by\nrandom sampling ({len(ps)} values).\n\n"
            "geomorph settings: gpagen(Proj = FALSE),\nmorphol.disparity(coords ~ group),\n"
            "modularity.test(opt.rot = FALSE).\nWith gpagen's default tangent\n"
            f"projection the median relative\ndifference is {rel_tp:.1e}.",
            va="top", fontsize=9.5, family="monospace")
    fig.suptitle("Descriptron shape statistics (Python) vs geomorph (R), same raw landmarks", fontsize=13)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return rel, rel_tp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rscript", required=True); ap.add_argument("--rlib", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--real"); ap.add_argument("--real-meta"); ap.add_argument("--real-schema")
    ap.add_argument("--real-category")
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rfile = out / "geomorph_side.R"; rfile.write_text(R_CODE)

    def run_r(d):
        subprocess.run([a.rscript, str(rfile), a.rlib, str(d)], check=True)

    rows = []
    X, meta, task = design_dataset()
    d = out / "design"; write_inputs(d, X, meta, task); run_r(d)
    rows += pair_up("design", python_side(X, meta, task), d)

    X, meta, task, nwk = tree_dataset()
    d = out / "phylogeny"; write_inputs(d, X, meta, task, {"tree.nwk": nwk}); run_r(d)
    rows += pair_up("phylogeny", python_side(X, meta, task, nwk=nwk), d)

    X, meta, task, pairs = symmetry_dataset()
    d = out / "symmetry"; write_inputs(d, X, meta, task); run_r(d)
    rows += pair_up("symmetry", python_side(X, meta, task, pairs=pairs), d)

    if a.real:
        files, Xr, _ = ss.load_shapes(a.real, a.real_category)
        cols, trows = dm.load_table(a.real_meta); schema = json.load(open(a.real_schema))
        links, _ = dm.link_images(trows, schema, files)
        group = next(c for c, r in schema.items() if r == "group")
        keep = [i for i, f in enumerate(files) if f in links and trows[links[f]].get(group)]
        metar = [{group: str(trows[links[files[i]]][group])} for i in keep]
        cnt = Counter(m[group] for m in metar)
        ok = [j for j, m in enumerate(metar) if cnt[m[group]] >= 2]
        metar = [metar[j] for j in ok]; Xr = Xr[[keep[j] for j in ok]]
        refl = ss.mirrored_specimens(Xr)
        Xr = Xr.copy(); Xr[refl, :, 0] *= -1            # same reflected input to both sides
        task = {"formula": group, "group": group}
        d = out / "real"; write_inputs(d, Xr, metar, task); run_r(d)
        rows += pair_up("real", python_side(Xr, metar, task), d)
        print(f"real data: {len(metar)} specimens, {int(refl.sum())} mirror images reflected before both sides")

    with open(out / "python_vs_geomorph_values.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rel, rel_tp = figure(rows, out / "python_vs_geomorph.png")
    print(f"{len(rows)} paired values; largest relative difference of deterministic statistics {rel:.2e} "
          f"(gpagen Proj=FALSE); median with geomorph's default tangent projection {rel_tp:.2e}")
    print(f"figure {out/'python_vs_geomorph.png'}; values {out/'python_vs_geomorph_values.csv'}")


if __name__ == "__main__":
    main()
