#!/usr/bin/env python3
"""Descriptron's phylogenetic statistics on a REAL tree, against geomorph and ape.

geomorph's bundled `plethspecies` (Plethodon, 9 species, 11 landmarks, published tree with its branch lengths):
  1. the tree: Descriptron's Newick parser -> phylogenetic covariance matrix, vs ape::vcv.phylo
  2. geometric morphometrics: centroid size and Procrustes distances (gpagen), Kmult on shape (physignal)
  3. morphometrics: Kmult on log centroid size, and on all inter-landmark distances (linear measurements)
Each program works from the same raw landmarks. There is no colour in this dataset.

  python validate_phylo_real_tree.py --rscript /path/to/Rscript --rlib /path/to/Rlib --out-dir val_phylo/
"""
import argparse, csv, importlib.util, json, subprocess
from pathlib import Path
import numpy as np


def _sibling(name):
    """measure/<name> in the source tree; beside this file once packaged (tools/)."""
    here = Path(__file__).resolve()
    return next((c for c in (here.parents[1] / name, here.parent / name) if c.exists()), here.parents[1] / name)

R_CODE = r'''
args <- commandArgs(TRUE); .libPaths(c(args[1], .libPaths())); d <- args[2]
suppressMessages({library(geomorph); library(ape)})
data(plethspecies)
A <- plethspecies$land; phy <- plethspecies$phy
sp <- dimnames(A)[[3]]
stopifnot(setequal(sp, phy$tip.label))
raw <- two.d.array(A); rownames(raw) <- sp
write.csv(raw, file.path(d, "raw_landmarks.csv"))
write.tree(phy, file.path(d, "tree.nwk"))
write.csv(vcv.phylo(phy), file.path(d, "R_vcv.csv"))
g <- gpagen(A, Proj = FALSE, print.progress = FALSE)
write.csv(data.frame(species = dimnames(g$coords)[[3]], Csize = as.numeric(g$Csize)), file.path(d, "R_csize.csv"), row.names = FALSE)
Y <- two.d.array(g$coords); rownames(Y) <- dimnames(g$coords)[[3]]
write.csv(as.matrix(dist(Y)), file.path(d, "R_pdist.csv"))
write.csv(Y, file.path(d, "R_aligned.csv"))
lin <- t(apply(A, 3, function(m) as.vector(dist(m)))); rownames(lin) <- sp
out <- list()
set.seed(1); ps <- physignal(g$coords, phy, iter = 999, print.progress = FALSE)
out$shape <- c(ps$phy.signal, ps$pvalue)
lcs <- log(g$Csize); names(lcs) <- dimnames(g$coords)[[3]]
set.seed(1); pc <- physignal(lcs, phy, iter = 999, print.progress = FALSE)
out$logCS <- c(pc$phy.signal, pc$pvalue)
set.seed(1); pl <- physignal(lin, phy, iter = 999, print.progress = FALSE)
out$linear <- c(pl$phy.signal, pl$pvalue)
res <- data.frame(trait = names(out), K = sapply(out, `[`, 1), P = sapply(out, `[`, 2))
write.csv(res, file.path(d, "R_kmult.csv"), row.names = FALSE)
# phylogenetic regression: shape ~ log centroid size (phylogenetic allometry)
gdf <- geomorph.data.frame(coords = g$coords, lcs = log(g$Csize), phy = phy)
set.seed(1); pg <- procD.pgls(coords ~ lcs, phy = phy, data = gdf, iter = 999, print.progress = FALSE)
at <- pg$aov.table
write.csv(data.frame(SS = at[1, "SS"], Rsq = at[1, "Rsq"], F = at[1, "F"], P = at[1, "Pr(>F)"],
                     SS_residual = at[2, "SS"]), file.path(d, "R_pgls.csv"), row.names = FALSE)
# phylomorphospace: ancestral states and PCA of the tips
pc <- gm.prcomp(g$coords, phy = phy)
anc <- pc$ancestors
pp <- prop.part(phy)
lab <- sapply(seq_along(pp), function(i) paste(sort(phy$tip.label[pp[[i]]]), collapse = "|"))
rownames(anc) <- lab
write.csv(anc, file.path(d, "R_ancestors.csv"))
ve <- pc$sdev^2 / sum(pc$sdev^2)
write.csv(data.frame(pc = seq_along(ve), var = ve), file.path(d, "R_pc_variance.csv"), row.names = FALSE)
'''


def read_matrix(p):
    rows = list(csv.reader(open(p)))
    names = [r[0] for r in rows[1:]]
    return rows[0][1:], names, np.array([[float(v) for v in r[1:]] for r in rows[1:]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rscript", required=True); ap.add_argument("--rlib", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--iter", type=int, default=999)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "geomorph_side.R").write_text(R_CODE)
    subprocess.run([a.rscript, str(out / "geomorph_side.R"), a.rlib, str(out)], check=True)
    spec = importlib.util.spec_from_file_location("ss", _sibling("descriptron_shape_stats.py"))
    ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)

    cols, sp, raw = read_matrix(out / "raw_landmarks.csv")
    X = raw.reshape(len(sp), raw.shape[1] // 2, 2)       # two.d.array writes x1, y1, x2, y2, ...
    tips, C = ss.parse_newick((out / "tree.nwk").read_text())
    rc, rn, RV = read_matrix(out / "R_vcv.csv")
    order = [tips.index(s) for s in rn]
    C_r = C[np.ix_(order, order)]
    res = {"species": len(sp), "landmarks": X.shape[1],
           "vcv_max_abs_diff": float(np.abs(C_r - RV).max()), "vcv_max_value": float(RV.max())}

    Z, cs = ss.gpa(X); Y = ss.flat(Z)
    Rcs = {r["species"]: float(r["Csize"]) for r in csv.DictReader(open(out / "R_csize.csv"))}
    res["csize_max_rel_diff"] = float(max(abs(c - Rcs[s]) / Rcs[s] for s, c in zip(sp, cs)))
    _, pn, RD = read_matrix(out / "R_pdist.csv")
    D = np.linalg.norm(Y[:, None] - Y[None], axis=2)
    pix = [sp.index(s) for s in pn]; D = D[np.ix_(pix, pix)]
    iu = np.triu_indices(len(sp), 1)
    res["pdist_max_rel_diff"] = float(np.max(np.abs(D[iu] - RD[iu]) / RD[iu]))

    ordp = [tips.index(s) for s in sp]; Cp = C[np.ix_(ordp, ordp)]
    lin = np.array([np.linalg.norm(x[:, None] - x[None], axis=2)[np.triu_indices(x.shape[0], 1)] for x in X])
    traits = {"shape": Y, "logCS": np.log(cs)[:, None], "linear": lin}
    Rk = {r["trait"]: (float(r["K"]), float(r["P"])) for r in csv.DictReader(open(out / "R_kmult.csv"))}
    rng = np.random.default_rng(1); rows = []
    for t, M in traits.items():
        r = ss.phylosignal(M, Cp, a.iter, rng)
        rows.append({"trait": t, "K_descriptron": r["Kmult"], "K_geomorph": Rk[t][0],
                     "K_rel_diff": abs(r["Kmult"] - Rk[t][0]) / abs(Rk[t][0]),
                     "P_descriptron": r["P"], "P_geomorph": Rk[t][1]})
    res["kmult"] = rows

    # ---- descriptron_phylo.py: its own tree reader, Kmult, PGLS and ancestral states
    spec2 = importlib.util.spec_from_file_location("dp", _sibling("descriptron_phylo.py"))
    dp = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(dp)
    tree = dp.parse_newick((out / "tree.nwk").read_text())
    ptips = dp.tips_of(tree); tn = [t.name for t in ptips]
    Ct = dp.vcv(ptips, ptips)
    Rmap = [rn.index(t) for t in tn]
    res["phylo_module_vcv_max_abs_diff"] = float(np.abs(Ct - RV[np.ix_(Rmap, Rmap)]).max())
    Yt = Y[[sp.index(t) for t in tn]]
    res["phylo_module_kmult_shape"] = dp.kmult(Yt, Ct)
    csT = cs[[sp.index(t) for t in tn]]
    pr = dp.pgls(Yt, np.log(csT), Ct, a.iter, np.random.default_rng(1))
    Rp = next(csv.DictReader(open(out / "R_pgls.csv")))
    res["pgls"] = {k: {"descriptron": pr[k], "geomorph": float(Rp[k]),
                       "rel_diff": abs(pr[k] - float(Rp[k])) / abs(float(Rp[k]))}
                   for k in ("SS", "Rsq", "F", "SS_residual")}
    res["pgls"]["P"] = {"descriptron": pr["P"], "geomorph": float(Rp["P"])}
    nodes = dp.internal_nodes(tree)
    # the ancestral-state computation is compared on geomorph's own aligned coordinates: the final orientation of a
    # superimposition is arbitrary (Descriptron's frame is geomorph's mirrored in x), which changes coordinates but no
    # distance, Kmult, regression or variance
    _, an, RA = read_matrix(out / "R_aligned.csv")
    anc = dp.ancestral_states(RA[[an.index(t) for t in tn]], ptips, nodes)
    Ra = {r[0]: np.array([float(v) for v in r[1:]]) for r in list(csv.reader(open(out / "R_ancestors.csv")))[1:]}
    worst = 0.0; matched = 0
    for n, a_ in zip(nodes, anc):
        key = "|".join(sorted(t.name for t in dp.tips_of(n)))
        if key in Ra:
            matched += 1
            worst = max(worst, float(np.abs(a_ - Ra[key]).max() / np.abs(Ra[key]).max()))
    res["ancestral_states"] = {"nodes": len(nodes), "matched": matched, "max_rel_diff": worst}
    sv = np.linalg.svd(Yt - Yt.mean(0), compute_uv=False); ev = sv ** 2 / (sv ** 2).sum()
    Rv = [float(r["var"]) for r in csv.DictReader(open(out / "R_pc_variance.csv"))]
    res["pc_variance_max_abs_diff"] = float(max(abs(e - r) for e, r in zip(ev, Rv)))
    with open(out / "phylo_real_tree_kmult.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(res, open(out / "phylo_real_tree_summary.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
