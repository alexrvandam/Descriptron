#!/usr/bin/env python3
"""
validate_community_run_vs_r.py - check one descriptron_community_v1 run against R, number by number
===================================================================================================

Run descriptron_community_v1.py with --save_matrices; it writes the exact inputs behind its results to
<run>/validation_inputs/. This script gives the same inputs to R and compares every number:

  variance partitioning   adjusted-R2 fractions [a] [b] [c] [d] (vegan::varpart) and the F of treatment | phylogeny
                          and phylogeny | treatment (vegan::anova of rda(Y, X, Z)), for every trait set;
  community phylogenetics Faith's PD with the root (ape::keep.tip + root-to-MRCA depth, = picante::pd), MPD and
                          MNTD (ape::cophenetic), for every site;
  phylomorphospace        the ancestral states drawn in the phylomorphospace (generalised least squares under
                          Brownian motion) against phytools::fastAnc, node by node, PC1 and PC2, for every trait set.

  python validate_community_run_vs_r.py --run_dir community_out/ --out_dir community_out/validation_vs_R/

Writes community_vs_R.tsv (every comparison: quantity, item, Descriptron, R, |difference|) and community_vs_R.png/.pdf
(Descriptron against R on a 1:1 line, one panel per kind of quantity). Exits non-zero if any difference > 1e-8.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

R_CODE = r"""
suppressMessages({library(vegan); library(ape); library(phytools); library(jsonlite)})
d <- commandArgs(TRUE)[1]; res <- list()
for (f in list.files(d, pattern = "^varpart_.*_Y\\.csv$")) {
  set <- sub("^varpart_(.*)_Y\\.csv$", "\\1", f)
  Y <- as.matrix(read.csv(file.path(d, f))); E <- as.matrix(read.csv(file.path(d, sprintf("varpart_%s_E.csv", set))))
  P <- as.matrix(read.csv(file.path(d, sprintf("varpart_%s_P.csv", set))))
  v <- varpart(Y, E, P)$part$indfract$Adj.R.square
  res[[paste0("varpart|", set)]] <- list(a = v[1], c = v[2], b = v[3], d = v[4],
      Fa = anova(rda(Y, E, P), permutations = 0)$F[1], Fc = anova(rda(Y, P, E), permutations = 0)$F[1])
}
if (file.exists(file.path(d, "site_species.tsv"))) {
  tr <- read.tree(file.path(d, "tree_sites.nwk")); co <- cophenetic(tr); dep <- node.depth.edgelength(tr)
  s <- read.delim(file.path(d, "site_species.tsv"), stringsAsFactors = FALSE)
  for (i in seq_len(nrow(s))) {
    sp <- strsplit(s$tips[i], ",")[[1]]
    mpd <- NA; mntd <- NA
    if (length(sp) >= 2) { D <- co[sp, sp]; mpd <- mean(D[lower.tri(D)]); diag(D) <- Inf; mntd <- mean(apply(D, 1, min)) }
    pd <- if (length(sp) >= 2) sum(keep.tip(tr, sp)$edge.length) + dep[getMRCA(tr, sp)] else dep[match(sp, tr$tip.label)]
    res[[paste0("site|", s$site[i])]] <- list(PD = pd, MPD = mpd, MNTD = mntd)
  }
}
for (f in list.files(d, pattern = "^morpho_.*_tree\\.nwk$")) {
  set <- sub("^morpho_(.*)_tree\\.nwk$", "\\1", f)
  tr <- read.tree(file.path(d, f)); tips <- read.csv(file.path(d, sprintf("morpho_%s_tips.csv", set)), row.names = 1)
  out <- list()
  for (pc in c("PC1", "PC2")) {
    x <- setNames(tips[[pc]], rownames(tips)); a <- fastAnc(tr, x)
    nodes <- as.integer(names(a))
    key <- sapply(nodes, function(n) paste(sort(tr$tip.label[getDescendants(tr, n)[getDescendants(tr, n) <= Ntip(tr)]]), collapse = ","))
    out[[pc]] <- setNames(as.list(unname(a)), key)
  }
  res[[paste0("morpho|", set)]] <- out
}
cat(toJSON(res, digits = NA, auto_unbox = TRUE))
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--rscript", default="Rscript")
    a = ap.parse_args()
    run = Path(a.run_dir); vi = run / "validation_inputs"
    if not vi.exists():
        sys.exit(f"{vi} not found - run descriptron_community_v1.py with --save_matrices")
    out = Path(a.out_dir) if a.out_dir else run / "validation_vs_R"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "v.R").write_text(R_CODE)
        r = subprocess.run([a.rscript, str(Path(t) / "v.R"), str(vi)], capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    R = json.loads(r.stdout)
    rows = []
    vp = pd.read_csv(run / "varpart.tsv", sep="\t").set_index("set") if (run / "varpart.tsv").exists() else None
    for key, val in R.items():
        kind, item = key.split("|", 1)
        if kind == "varpart" and vp is not None and item in vp.index:
            d = vp.loc[item]
            for lab, ours, rv in (("fraction [a] treatment only", d["treatment_only_[a]"], val["a"]),
                                  ("fraction [b] shared", d["shared_[b]"], val["b"]),
                                  ("fraction [c] phylogeny only", d["phylogeny_only_[c]"], val["c"]),
                                  ("fraction [d] unexplained", d["unexplained_[d]"], val["d"]),
                                  ("F treatment | phylogeny", d["F_treatment_given_phylogeny"], val["Fa"]),
                                  ("F phylogeny | treatment", d["F_phylogeny_given_treatment"], val["Fc"])):
                rows.append({"panel": "variance partitioning" if lab.startswith("fraction") else "partial F",
                             "quantity": lab, "item": item, "descriptron": float(ours), "R": float(rv)})
    if (vi / "site_species.tsv").exists():
        ss = pd.read_csv(vi / "site_species.tsv", sep="\t").set_index("site")
        for key, val in R.items():
            kind, item = key.split("|", 1)
            if kind != "site":
                continue
            row = ss.loc[item] if item in ss.index else ss.loc[ss.index.astype(str) == item].iloc[0]
            for q in ("PD", "MPD", "MNTD"):
                ours, rv = row[q], val[q]
                if rv is None or (isinstance(rv, float) and np.isnan(rv)) or pd.isna(ours):
                    continue
                rows.append({"panel": "community phylogenetics", "quantity": q, "item": item,
                             "descriptron": float(ours), "R": float(rv)})
    for key, val in R.items():
        kind, item = key.split("|", 1)
        if kind != "morpho":
            continue
        nodes = pd.read_csv(vi / f"morpho_{item}_nodes.csv").set_index("descendants")
        for pc in ("PC1", "PC2"):
            for desc, rv in val[pc].items():
                if desc in nodes.index:
                    rows.append({"panel": "phylomorphospace ancestral states", "quantity": f"node {pc}",
                                 "item": f"{item}: {desc[:40]}", "descriptron": float(nodes.loc[desc, pc]), "R": float(rv)})
    df = pd.DataFrame(rows)
    df["abs_difference"] = (df["descriptron"] - df["R"]).abs()
    df.to_csv(out / "community_vs_R.tsv", sep="\t", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [p for p in ("variance partitioning", "partial F", "community phylogenetics",
                          "phylomorphospace ancestral states") if p in set(df["panel"])]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.8), squeeze=False)
    refs = {"variance partitioning": "vegan::varpart", "partial F": "vegan::rda / anova",
            "community phylogenetics": "ape (PD with root, MPD, MNTD)", "phylomorphospace ancestral states": "phytools::fastAnc"}
    colours = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    for ax, pnl in zip(axs.ravel(), panels):
        g = df[df["panel"] == pnl]
        for k, (q, gg) in enumerate(g.groupby("quantity")):
            ax.scatter(gg["R"], gg["descriptron"], s=16, color=colours[k % len(colours)], label=q, alpha=0.85)
        lo, hi = min(g["R"].min(), g["descriptron"].min()), max(g["R"].max(), g["descriptron"].max())
        pad = 0.05 * (hi - lo or 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#999", lw=0.8, zorder=0)
        ax.set_xlabel(f"R: {refs[pnl]}", fontsize=8); ax.set_ylabel("Descriptron", fontsize=8)
        ax.set_title(f"{pnl}\nn = {len(g)}, max |diff| = {g['abs_difference'].max():.1e}", fontsize=8.5)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"community_vs_R.{ext}", dpi=200)
    plt.close(fig)
    worst = df.groupby("panel")["abs_difference"].max()
    for pnl, w in worst.items():
        print(f"  {pnl:36s} n={int((df.panel == pnl).sum()):4d}  max |difference| = {w:.2e}")
    print(f"-> {out}")
    sys.exit(1 if (df["abs_difference"] > 1e-8).any() else 0)


if __name__ == "__main__":
    main()
