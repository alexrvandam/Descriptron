#!/usr/bin/env python3
"""Reproduce published phylogenetic-signal results with Descriptron: Waldron et al. (2025, Evolution 79: 2369;
doi:10.1093/evolut/qpaf167), 57 Plethodon species, data and trees on Dryad (doi:10.5061/dryad.bvq83bknf).

Their Table S5 gives Blomberg's K (and Pagel's lambda) for six traits on the pruned MCMCtree timetree: costal-groove
count (CG) and five body-size-corrected measurements (divided by SVL, log-transformed: S.E, HL, BW, FLL, HLL). Here the
same K is computed by Descriptron (descriptron_phylo.kmult on one trait = Blomberg's K), phytools::phylosig and
geomorph::physignal, all on their processed data and their tree, and compared with the published values. Their standard
PCA (PC1 83.4%, PC2 6.5% in the SI) is recomputed as well. Pagel's lambda is not implemented in Descriptron.

Download the Dryad files by hand (Dryad blocks scripted downloads) and unzip them into one folder:

  python validate_phylo_published_waldron2025.py --data_dir <folder> --rscript /path/Rscript --rlib /path/Rlib \
      --out-dir waldron_check/
"""
import argparse, csv, importlib.util, json, subprocess
from pathlib import Path
import numpy as np


def _sibling(name):
    """measure/<name> in the source tree; beside this file once packaged (tools/)."""
    here = Path(__file__).resolve()
    return next((c for c in (here.parents[1] / name, here.parent / name) if c.exists()), here.parents[1] / name)

TRAITS = ["CG", "S.E", "HL", "BW", "FLL", "HLL"]

R_CODE = r'''
args <- commandArgs(TRUE); .libPaths(c(args[1], .libPaths())); d <- args[2]
suppressMessages({library(ape); library(geomorph); library(phytools)})
phy <- read.tree(file.path(d, "tree.nwk"))
X <- read.csv(file.path(d, "traits.csv"), row.names = 1, check.names = FALSE)
X <- X[phy$tip.label, , drop = FALSE]
if (anyNA(X)) {       # as the authors did: phylogenetic imputation with Rphylopars (Brownian motion, defaults)
  suppressMessages(library(Rphylopars))
  td <- data.frame(species = rownames(X), X, check.names = FALSE)
  pr <- phylopars(trait_data = td, tree = phy)
  imp <- pr$anc_recon[rownames(X), colnames(X), drop = FALSE]
  X[is.na(X)] <- imp[is.na(X)]
}
write.csv(X, file.path(d, "traits_used.csv"))
out <- data.frame()
for (v in colnames(X)) {
  x <- X[[v]]; names(x) <- rownames(X)
  set.seed(1); k <- phylosig(phy, x, method = "K", test = TRUE, nsim = 999)
  set.seed(1); p <- physignal(matrix(x, ncol = 1, dimnames = list(names(x), v)), phy, iter = 999, print.progress = FALSE)
  out <- rbind(out, data.frame(trait = v, K_phytools = k$K, P_phytools = k$P, K_geomorph = p$phy.signal, P_geomorph = p$pvalue))
}
write.csv(out, file.path(d, "R_K.csv"), row.names = FALSE)
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True); ap.add_argument("--rscript", required=True)
    ap.add_argument("--rlib", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iter", type=int, default=999)
    a = ap.parse_args(argv)
    import pandas as pd
    D, out = Path(a.data_dir), Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    morph = pd.read_csv(next(D.rglob("Plethodon_morphology_R_Processed*.csv")))
    tree_txt = next(D.rglob("plethodon-pruned.tree")).read_text()
    pub = pd.read_excel(next(D.rglob("Table_S5_Plethodon_phylosignal.xlsx")), header=None)
    hdr = pub.index[pub[0].astype(str).eq("variable")][0]
    pub = pub.iloc[hdr + 1:, :5]; pub.columns = ["trait", "lambda", "p_lambda", "K", "p_K"]
    pub = pub.dropna(subset=["trait"]).set_index("trait").astype(float)

    spec = importlib.util.spec_from_file_location("dp", _sibling("descriptron_phylo.py"))
    dp = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp)
    tree = dp.parse_newick(tree_txt)
    tips = dp.tips_of(tree); tn = [t.name for t in tips]
    col = {"CG": "CG", "S.E": "S.E.sc", "HL": "HL.sc", "BW": "BW.sc", "FLL": "FLL.sc", "HLL": "HLL.sc"}
    M = morph.set_index("Species").loc[tn, [col[t] for t in TRAITS]].astype(float)
    M.columns = TRAITS
    missing = [(i, c) for c in M.columns for i in M.index[M[c].isna()]]
    (out / "tree.nwk").write_text(tree_txt)
    M.to_csv(out / "traits.csv")
    (out / "compare.R").write_text(R_CODE)
    subprocess.run([a.rscript, str(out / "compare.R"), a.rlib, str(out)], check=True)
    R = pd.read_csv(out / "R_K.csv").set_index("trait")
    Mi = pd.read_csv(out / "traits_used.csv", index_col=0).loc[tn, TRAITS]
    assert np.allclose(Mi.values[~M.isna().values], M.values[~M.isna().values])   # only the gaps were filled
    M = Mi

    C = dp.vcv(tips, tips)
    rows = []
    for t in TRAITS:
        y = M[t].values[:, None]
        r = dp.phylo_signal(y, C, a.iter, np.random.default_rng(1))
        rows.append({"trait": t, "K_published": pub.loc[t, "K"], "K_descriptron": r["Kmult"],
                     "K_phytools": R.loc[t, "K_phytools"], "K_geomorph": R.loc[t, "K_geomorph"],
                     "diff_descriptron_vs_published": abs(r["Kmult"] - pub.loc[t, "K"]),
                     "rel_diff_descriptron_vs_phytools": abs(r["Kmult"] - R.loc[t, "K_phytools"]) / R.loc[t, "K_phytools"],
                     "P_published": pub.loc[t, "p_K"], "P_descriptron": r["P"],
                     "P_phytools": R.loc[t, "P_phytools"], "P_geomorph": R.loc[t, "P_geomorph"],
                     "lambda_published": pub.loc[t, "lambda"]})
    res = pd.DataFrame(rows)
    res.to_csv(out / "waldron2025_K_comparison.csv", index=False)
    # their standard (non-phylogenetic) PCA: PC1 83.4%, PC2 6.5%
    pcs = {}
    for how in ("covariance", "correlation"):
        Z = M.values - M.values.mean(0)
        if how == "correlation":
            Z = Z / M.values.std(0, ddof=1)
        sv = np.linalg.svd(Z, compute_uv=False); ev = sv ** 2 / (sv ** 2).sum()
        pcs[how] = [round(float(ev[0]) * 100, 1), round(float(ev[1]) * 100, 1)]
    Mm = M.drop(columns=["CG"])
    sv = np.linalg.svd(Mm.values - Mm.values.mean(0), compute_uv=False); ev = sv ** 2 / (sv ** 2).sum()
    pcs["covariance, size-corrected traits only"] = [round(float(ev[0]) * 100, 1), round(float(ev[1]) * 100, 1)]
    summary = {"species": len(tn), "imputed_cells": [f"{i}:{c}" for i, c in missing], "K": rows, "pca_percent_PC1_PC2": pcs, "pca_published": [83.4, 6.5]}
    json.dump(summary, open(out / "waldron2025_summary.json", "w"), indent=2, default=float)
    print(res.round(6).to_string()); print("PCA:", pcs, "published [83.4, 6.5]")


if __name__ == "__main__":
    main()
