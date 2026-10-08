#!/usr/bin/env python3
"""
validate_community_vs_r.py - descriptron_community_v1 statistics against R (vegan, ape)
========================================================================================

  1. variance partitioning: adjusted-R2 fractions [a] [b] [c] [d] against vegan::varpart(Y, E, P), and the F of
     treatment | phylogeny and phylogeny | treatment against vegan::anova(rda(Y, E, P)) / rda(Y, P, E);
  2. community phylogenetics on the example tree: MPD and MNTD (from ape::cophenetic) and Faith's PD including
     the root (sum of the branch lengths of ape::keep.tip plus the root-to-MRCA depth = picante::pd include.root)
     for random species sets.

  python validate_community_vs_r.py [--rscript Rscript]
Prints the largest absolute difference for each quantity and exits non-zero if any exceeds 1e-8.
(Kmult is descriptron_phylo's, already checked against geomorph::physignal in validate_phylo_traits.py.)
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))                   # pip packages keep every program in one flat folder
import descriptron_community_v1 as cm   # noqa: E402
import descriptron_phylo as ph          # noqa: E402

R_CODE = r"""
suppressMessages({library(vegan); library(ape)})
a <- commandArgs(TRUE); d <- a[1]
Y <- as.matrix(read.csv(file.path(d, "Y.csv"))); E <- as.matrix(read.csv(file.path(d, "E.csv")))
P <- as.matrix(read.csv(file.path(d, "P.csv")))
v <- varpart(Y, E, P)
f <- v$part$indfract$Adj.R.square
Fa <- anova(rda(Y, E, P), permutations = 0)$F[1]
Fc <- anova(rda(Y, P, E), permutations = 0)$F[1]
tr <- read.tree(file.path(d, "tree.nwk")); co <- cophenetic(tr); dep <- node.depth.edgelength(tr)
sets <- strsplit(readLines(file.path(d, "sets.txt")), ",")
out <- lapply(sets, function(s) {
  D <- co[s, s]; mpd <- mean(D[lower.tri(D)]); diag(D) <- Inf; mntd <- mean(apply(D, 1, min))
  k <- keep.tip(tr, s); mr <- getMRCA(tr, s)
  c(mpd = mpd, mntd = mntd, pd = sum(k$edge.length) + dep[mr])
})
cat(jsonlite::toJSON(list(fract = f, Fa = Fa, Fc = Fc, comm = out), digits = NA))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rscript", default="Rscript")
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    n = 32
    Y = rng.normal(size=(n, 5)); E = np.column_stack([rng.integers(0, 2, n), rng.normal(size=n)]).astype(float)
    P = rng.normal(size=(n, 3)); Y[:, 0] += 1.5 * E[:, 0] + P[:, 0]
    py = cm.varpart(Y - Y.mean(0), E, P, iters=0, rng=rng)
    ex = HERE.parent / "examples" / "community_example" / "tree_species.nwk"
    nwk = ex.read_text() if ex.exists() else (       # the pip package has no examples folder: a fixed test tree
        "(((A:1.2,B:1.2):0.8,(C:0.5,D:0.5):1.5):1.0,((E:2.1,(F:0.9,G:0.9):1.2):0.4,(H:1.6,(I:0.3,J:0.3):1.3):0.9):0.5);")
    root = ph.parse_newick(nwk)
    tips = sorted(t.name for t in ph.tips_of(root))
    phy = type("P", (), {})()
    sets = [sorted(rng.choice(tips, min(k, len(tips)), replace=False).tolist()) for k in (3, 5, 8, 12)]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for nm, M in (("Y", Y), ("E", E), ("P", P)):
            np.savetxt(d / f"{nm}.csv", M, delimiter=",", header=",".join(f"v{i}" for i in range(M.shape[1])), comments="")
        (d / "tree.nwk").write_text(nwk)
        (d / "sets.txt").write_text("\n".join(",".join(s) for s in sets))
        (d / "v.R").write_text(R_CODE)
        r = subprocess.run([a.rscript, str(d / "v.R"), str(d)], capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr)
        R = json.loads(r.stdout)
    # vegan's indfract rows are X1|X2, X2|X1, shared, residual (vegan calls the shared fraction [c]; we follow
    # Desdevises et al. 2003, where [b] is the shared fraction and [c] phylogeny only)
    ours = [py["treatment_only_[a]"], py["phylogeny_only_[c]"], py["shared_[b]"], py["unexplained_[d]"]]
    worst = {}
    worst["varpart fractions"] = float(np.max(np.abs(np.array(ours) - np.array(R["fract"], float))))
    Fa, _ = cm.partial_F_test(Y - Y.mean(0), E, P, 0, rng)
    Fc, _ = cm.partial_F_test(Y - Y.mean(0), P, E, 0, rng)
    worst["F treatment | phylogeny"] = abs(Fa - float(np.ravel(R["Fa"])[0]))
    worst["F phylogeny | treatment"] = abs(Fc - float(np.ravel(R["Fc"])[0]))
    dm = dp = 0.0
    T = {t.name: t for t in ph.tips_of(root)}
    for s, rr in zip(sets, R["comm"]):
        names = s
        D = np.array([[0.0 if x == y else T[x].depth + T[y].depth - 2 * ph.mrca_depth(T[x], T[y]) for y in names] for x in names])
        mpd, mntd = cm.mpd_mntd(D)
        rr = [float(np.ravel(v)[0]) for v in (rr if isinstance(rr, list) else rr.values())] if not isinstance(rr, dict) else [float(rr[k]) for k in ("mpd", "mntd", "pd")]
        dm = max(dm, abs(mpd - rr[0]), abs(mntd - rr[1]))
        dp = max(dp, abs(cm.faith_pd(root, set(names)) - rr[2]))
    worst["MPD / MNTD"] = dm
    worst["Faith's PD (with root)"] = dp
    bad = False
    for k, v in worst.items():
        ok = v < 1e-8
        bad |= not ok
        print(f"  {k:28s} max |difference| = {v:.2e}  {'OK' if ok else 'DIFFERS'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
