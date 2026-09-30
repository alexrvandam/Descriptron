#!/usr/bin/env python3
"""descriptron_phylo.py against geomorph, ape and phytools on any tree and any continuous trait sets.

For every trait set (landmark or outline shape, measurements, colour pattern, texture, ...), on the same species means:
  - Kmult (descriptron_phylo.kmult)                vs geomorph::physignal
  - PGLS on a predictor (descriptron_phylo.pgls)   vs geomorph::procD.pgls          (SS, R2, F; P by permutation)
  - ancestral states, all internal nodes           vs geomorph::gm.prcomp(phy = )$ancestors
  - PCA variance of the species means               vs geomorph::gm.prcomp
and, for the first principal component of each set as a single trait:
  - Blomberg's K                                    vs phytools::phylosig(method = "K")
  - ML ancestral states                             vs ape::ace(method = "ML") and phytools::fastAnc
plus the tree itself: descriptron_phylo.vcv        vs ape::vcv.phylo.

Trait tables are CSVs with a species/tip column (--species_col) and numeric columns; --traits NAME=CSV[:REGEX] keeps
the columns matching REGEX; --scale NAME=standardize standardises a set before every program sees it (so all three
receive the identical matrix). --predictor SET:COLUMN gives the PGLS predictor (one value per species).

  python validate_phylo_traits.py --tree tree.nwk --species_col tip --traits shape=shape.csv \
      --traits colour=colour.csv --scale colour=standardize --predictor meas:aspect_ratio \
      --rscript /path/Rscript --rlib /path/Rlib --out-dir out/
"""
import argparse, csv, importlib.util, json, re, subprocess
from pathlib import Path
import numpy as np


def _sibling(name):
    """measure/<name> in the source tree; beside this file once packaged (tools/)."""
    here = Path(__file__).resolve()
    return next((c for c in (here.parents[1] / name, here.parent / name) if c.exists()), here.parents[1] / name)

R_CODE = r'''
args <- commandArgs(TRUE); .libPaths(c(args[1], .libPaths())); d <- args[2]
suppressMessages({library(ape); library(geomorph); library(phytools)})
phy <- read.tree(file.path(d, "tree.nwk"))
write.csv(vcv.phylo(phy), file.path(d, "R_vcv.csv"))
pp <- prop.part(phy)
lab <- sapply(seq_along(pp), function(i) paste(sort(phy$tip.label[pp[[i]]]), collapse = "|"))
pred <- read.csv(file.path(d, "predictor.csv"), row.names = 1)
for (f in list.files(d, pattern = "^set_.*\\.csv$")) {
  nm <- sub("^set_(.*)\\.csv$", "\\1", f)
  Y <- as.matrix(read.csv(file.path(d, f), row.names = 1, check.names = FALSE))
  Y <- Y[phy$tip.label, , drop = FALSE]
  set.seed(1); ps <- physignal(Y, phy, iter = 999, print.progress = FALSE)
  x <- pred[phy$tip.label, 1]; names(x) <- phy$tip.label
  gdf <- rrpp.data.frame(Y = Y, x = x)
  set.seed(1); pg <- procD.pgls(Y ~ x, phy = phy, data = gdf, iter = 999, print.progress = FALSE)
  at <- pg$aov.table
  pc <- gm.prcomp(Y, phy = phy)
  anc <- pc$ancestors; rownames(anc) <- lab
  write.csv(anc, file.path(d, paste0("R_anc_", nm, ".csv")))
  ve <- pc$sdev^2 / sum(pc$sdev^2)
  # first principal component of the species means, as one trait, for the univariate programs
  z <- as.numeric(scale(Y, scale = FALSE) %*% svd(scale(Y, scale = FALSE))$v[, 1]); names(z) <- phy$tip.label
  kb <- phylosig(phy, z, method = "K")
  ac <- ace(z, phy, type = "continuous", method = "ML")$ace
  fa <- fastAnc(phy, z)
  names(ac) <- lab[as.integer(names(ac)) - Ntip(phy)]; names(fa) <- lab[as.integer(names(fa)) - Ntip(phy)]
  write.csv(data.frame(node = names(ac), ace = ac, fastAnc = fa[names(ac)]), file.path(d, paste0("R_pc1anc_", nm, ".csv")), row.names = FALSE)
  write.csv(data.frame(z = z), file.path(d, paste0("R_pc1_", nm, ".csv")))
  write.csv(data.frame(K = ps$phy.signal, P = ps$pvalue, SS = at[1, "SS"], Rsq = at[1, "Rsq"], F = at[1, "F"],
                       Ppgls = at[1, "Pr(>F)"], SS_residual = at[2, "SS"], pc1_var = ve[1], pc2_var = ve[2],
                       K_univ = as.numeric(kb)),
            file.path(d, paste0("R_stats_", nm, ".csv")), row.names = FALSE)
}
'''


def read_matrix(p):
    rows = list(csv.reader(open(p)))
    return [r[0] for r in rows[1:]], rows[0][1:], np.array([[float(v) for v in r[1:]] for r in rows[1:]])


def load_set(path, species_col, regex):
    rows = list(csv.DictReader(open(path, newline="")))
    cols = [c for c in rows[0] if c != species_col]
    num = [c for c in cols if all(_num(r[c]) for r in rows)]
    if regex:
        num = [c for c in num if re.search(regex, c)]
    by = {}
    for r in rows:
        by.setdefault(r[species_col], []).append([float(r[c]) for c in num])
    sp = sorted(by)
    return sp, num, np.array([np.mean(by[s], axis=0) for s in sp])


def _num(v):
    try:
        float(v); return True
    except ValueError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True); ap.add_argument("--species_col", required=True)
    ap.add_argument("--traits", action="append", required=True, metavar="NAME=CSV[:REGEX]")
    ap.add_argument("--scale", action="append", default=[], metavar="NAME=standardize")
    ap.add_argument("--predictor", required=True, metavar="SET:COLUMN")
    ap.add_argument("--rscript", required=True); ap.add_argument("--rlib", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--iter", type=int, default=999)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("dp", _sibling("descriptron_phylo.py"))
    dp = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp)
    tree = dp.parse_newick(Path(a.tree).read_text())
    scales = dict(x.split("=", 1) for x in a.scale)
    sets = {}
    for t in a.traits:
        name, rest = t.split("=", 1)
        path, _, rx = rest.partition(":") if ":" in rest[2:] else (rest, "", "")
        sp, cols, M = load_set(path, a.species_col, rx or None)
        sets[name] = (sp, cols, M)
    common = set.intersection(*[set(s[0]) for s in sets.values()]) & {t.name for t in dp.tips_of(tree)}
    tree = dp.prune(tree, common)
    tips = dp.tips_of(tree); tn = [t.name for t in tips]
    Path(out / "tree.nwk").write_text(_newick(tree) + ";\n")
    ps, pcol = a.predictor.split(":", 1)
    sp, cols, M = sets[ps]
    xv = np.array([M[sp.index(t), cols.index(pcol)] for t in tn])
    with open(out / "predictor.csv", "w") as f:
        f.write("tip,x\n" + "".join(f"{t},{float(v)!r}\n" for t, v in zip(tn, xv)))
    Ys = {}
    for name, (sp, cols, M) in sets.items():
        Y = M[[sp.index(t) for t in tn]]
        if scales.get(name) == "standardize":
            sd = Y.std(0, ddof=1); Y = (Y[:, sd > 0] - Y[:, sd > 0].mean(0)) / sd[sd > 0]
        Ys[name] = Y
        with open(out / f"set_{name}.csv", "w") as f:        # the identical matrix every program receives
            f.write("tip," + ",".join(f"v{i}" for i in range(Y.shape[1])) + "\n")
            for t, row in zip(tn, Y):
                f.write(t + "," + ",".join(repr(float(v)) for v in row) + "\n")
    (out / "compare.R").write_text(R_CODE)
    subprocess.run([a.rscript, str(out / "compare.R"), a.rlib, str(out)], check=True)

    rn, _, RV = read_matrix(out / "R_vcv.csv")
    C = dp.vcv(tips, tips)
    summary = {"tips": len(tn), "vcv_max_abs_diff": float(np.abs(C - RV[np.ix_([rn.index(t) for t in tn], [rn.index(t) for t in tn])]).max())}
    nodes = dp.internal_nodes(tree)
    keys = ["|".join(sorted(x.name for x in dp.tips_of(n))) for n in nodes]
    rows = []
    for name, Y in Ys.items():
        R = next(csv.DictReader(open(out / f"R_stats_{name}.csv")))
        K = dp.kmult(Y, C); sig = dp.phylo_signal(Y, C, a.iter, np.random.default_rng(1))
        pg = dp.pgls(Y, xv, C, a.iter, np.random.default_rng(1))
        anc = dp.ancestral_states(Y, tips, nodes)
        an, _, RA = read_matrix(out / f"R_anc_{name}.csv")
        ad = max(float(np.abs(anc[i] - RA[an.index(k)]).max()) for i, k in enumerate(keys)) / float(np.abs(RA).max())
        sv = np.linalg.svd(Y - Y.mean(0), compute_uv=False); ev = sv ** 2 / (sv ** 2).sum()
        zr = {r[""] if "" in r else list(r.values())[0]: float(r["z"]) for r in csv.DictReader(open(out / f"R_pc1_{name}.csv"))}
        z = np.array([zr[t] for t in tn])[:, None]
        k1 = dp.kmult(z, C)
        a1 = dp.ancestral_states(z, tips, nodes)[:, 0]
        ua = {r["node"]: (float(r["ace"]), float(r["fastAnc"])) for r in csv.DictReader(open(out / f"R_pc1anc_{name}.csv"))}
        scale_ = float(np.abs(z).max())
        rel = lambda x, y: abs(x - y) / abs(y)
        rows.append({
            "trait_set": name, "n_tips": len(tn), "n_variables": Y.shape[1],
            "Kmult_descriptron": K, "Kmult_geomorph": float(R["K"]), "Kmult_rel_diff": rel(K, float(R["K"])),
            "P_descriptron": sig["P"], "P_geomorph": float(R["P"]),
            "PGLS_Rsq_descriptron": pg["Rsq"], "PGLS_Rsq_geomorph": float(R["Rsq"]),
            "PGLS_max_rel_diff_SS_R2_F": max(rel(pg["SS"], float(R["SS"])), rel(pg["Rsq"], float(R["Rsq"])), rel(pg["F"], float(R["F"]))),
            "PGLS_P_descriptron": pg["P"], "PGLS_P_geomorph": float(R["Ppgls"]),
            "ancestral_max_rel_diff_geomorph": ad,
            "PC1_var_abs_diff": abs(ev[0] - float(R["pc1_var"])),
            "K_univariate_PC1_descriptron": k1, "K_univariate_PC1_phytools": float(R["K_univ"]),
            "K_univariate_rel_diff": rel(k1, float(R["K_univ"])),
            "ancestral_PC1_max_diff_ape_ace": max(abs(a1[i] - ua[k][0]) for i, k in enumerate(keys)) / scale_,
            "ancestral_PC1_max_diff_phytools_fastAnc": max(abs(a1[i] - ua[k][1]) for i, k in enumerate(keys)) / scale_,
        })
    with open(out / "phylo_traits_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary["sets"] = rows
    json.dump(summary, open(out / "phylo_traits_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=1))


def _newick(n):
    if not n.kids:
        return f"{n.name}:{n.length!r}"
    inner = ",".join(_newick(k) for k in n.kids)
    return f"({inner}){n.name}:{n.length!r}" if n.parent is not None else f"({inner})"


if __name__ == "__main__":
    main()
