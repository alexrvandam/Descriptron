#!/usr/bin/env python3
"""
descriptron_phylo.py - phylogenetic comparative analyses of any continuous Descriptron traits
=============================================================================================

Given a Newick tree and one or more trait tables (landmark shape, linear measurements, colour pattern, texture, or
any other continuous features), computes on species means:

  signal     multivariate phylogenetic signal Kmult (Adams 2014), permutation P and effect size Z,
             as geomorph::physignal
  pgls       phylogenetic regression of a trait set on a predictor (for example log centroid size: phylogenetic
             allometry) by generalised least squares under Brownian motion, SS / R2 / F with residual
             randomisation, as geomorph::procD.pgls
  morphospace principal components of the species means with ancestral states estimated under Brownian motion
             (generalised least squares) projected into the same space and the tree drawn through them, as
             geomorph::gm.prcomp(phy = )

Trait tables: CSV with one row per specimen (or per species) and numeric columns. The species of each row comes
from a column (--species_col) or from a regular expression applied to an id column (--species_regex). Rows of the
same species are averaged. Species absent from the tree, or tree tips without data, are dropped (the tree is pruned).

  python descriptron_phylo.py --tree tree.nwk --out_dir phylo/ \
      --traits shape=landmark_gpa/forewing/forewing_procrustes_coords.csv \
      --traits colour=color_homology/whole_wing/color_homology_features_whole_wing.csv \
      --species_regex "morph_([A-Za-z]+\\d*?)_?\\d+_" --scale colour=standardize \
      --pgls shape~logCS --analyses signal pgls morphospace

Scaling (--scale NAME=none|standardize|log): Kmult and the PGLS statistics weight every column by its variance, so a
set that mixes units (millimetres with ratios, colour channels with entropies) should be standardised; Procrustes
shape coordinates are analysed as they are (the default for a set whose columns look like x/y coordinates).

Validated against geomorph 4.1.1 and ape on geomorph's plethspecies data (validation/validate_phylo_real_tree.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================ tree
class Node:
    __slots__ = ("name", "length", "kids", "parent", "depth", "idx")

    def __init__(self, name="", length=0.0):
        self.name, self.length, self.kids, self.parent, self.depth, self.idx = name, length, [], None, 0.0, -1


def parse_newick(s: str) -> Node:
    s = s.strip().rstrip(";")
    pos = 0

    def node() -> Node:
        nonlocal pos
        n = Node()
        if s[pos] == "(":
            pos += 1
            n.kids.append(node())
            while s[pos] == ",":
                pos += 1
                n.kids.append(node())
            pos += 1                                                  # ')'
        m = re.match(r"([^:,();\[]*)(?:\[[^\]]*\])?(?::([0-9.eE+\-]+))?", s[pos:])
        pos += m.end()
        n.name = m.group(1).strip().strip("'\"")
        n.length = float(m.group(2) or 0.0)
        for k in n.kids:
            k.parent = n
        return n

    root = node()
    root.length = 0.0                                                 # the root's own branch does not count
    _set_depths(root)
    return root


def _set_depths(n: Node, d: float = 0.0):
    n.depth = d
    for k in n.kids:
        _set_depths(k, d + k.length)


def tips_of(n: Node) -> List[Node]:
    return [n] if not n.kids else [t for k in n.kids for t in tips_of(k)]


def internal_nodes(n: Node) -> List[Node]:
    return ([n] if n.kids else []) + [x for k in n.kids for x in internal_nodes(k)]


def prune(root: Node, keep: set) -> Node:
    """Drop tips not in `keep`; collapse nodes left with one child (their branch lengths add up)."""
    def rec(n: Node) -> Optional[Node]:
        if not n.kids:
            return n if n.name in keep else None
        kids = [c for c in (rec(k) for k in n.kids) if c is not None]
        if not kids:
            return None
        if len(kids) == 1:
            kids[0].length += n.length
            return kids[0]
        n.kids = kids
        for k in kids:
            k.parent = n
        return n

    r = rec(root)
    r.length, r.parent = 0.0, None
    _set_depths(r)
    return r


def mrca_depth(a: Node, b: Node) -> float:
    anc = set()
    x = a
    while x is not None:
        anc.add(id(x)); x = x.parent
    y = b
    while id(y) not in anc:
        y = y.parent
    return y.depth


def vcv(nodes_a: List[Node], nodes_b: List[Node]) -> np.ndarray:
    """Brownian-motion covariance: shared path length from the root (tips and internal nodes alike)."""
    return np.array([[mrca_depth(a, b) for b in nodes_b] for a in nodes_a])


# ============================================================================ statistics
def _whiten(C: np.ndarray) -> np.ndarray:
    """P with P'P = C^-1 (any such P gives the same GLS sums of squares)."""
    return np.linalg.inv(np.linalg.cholesky(C))


def kmult(Y: np.ndarray, C: np.ndarray) -> float:
    """Adams (2014) multivariate K, as geomorph::physignal."""
    n = len(Y)
    Ci = np.linalg.inv(C)
    one = np.ones((n, 1))
    a = np.linalg.solve(one.T @ Ci @ one, one.T @ Ci @ Y)
    R = Y - one @ a
    obs = np.trace(R.T @ R) / np.trace(R.T @ Ci @ R)
    expct = (np.trace(C) - n / Ci.sum()) / (n - 1)
    return float(obs / expct)


def effect_size(obs: float, rand: np.ndarray) -> float:
    r = np.r_[obs, rand]
    sd = r.std(ddof=1)
    return float((obs - r.mean()) / sd) if sd > 0 else float("nan")


def phylo_signal(Y, C, iters, rng) -> Dict:
    K = kmult(Y, C)
    rand = np.array([kmult(Y[rng.permutation(len(Y))], C) for _ in range(iters)])
    return {"Kmult": K, "P": float((1 + (rand >= K - 1e-12).sum()) / (iters + 1)), "Z": effect_size(K, rand)}


def pgls(Y: np.ndarray, x: np.ndarray, C: np.ndarray, iters: int, rng) -> Dict:
    """Y ~ x under Brownian motion by GLS; SS from the whitened data, residual randomisation of the null model
    (intercept only), as RRPP / geomorph::procD.pgls with a single predictor."""
    n = len(Y)
    P = _whiten(C)
    Yt = P @ Y
    X0 = P @ np.ones((n, 1))
    X1 = P @ np.column_stack([np.ones(n), x])

    def rss(X, Yy):
        B, *_ = np.linalg.lstsq(X, Yy, rcond=None)
        R = Yy - X @ B
        return float((R ** 2).sum())

    B0, *_ = np.linalg.lstsq(X0, Yt, rcond=None)
    fit0, res0 = X0 @ B0, Yt - X0 @ B0
    ss_tot, ss_res = rss(X0, Yt), rss(X1, Yt)
    ss = ss_tot - ss_res
    df, dfr = 1, n - 2
    F = (ss / df) / (ss_res / dfr)
    rand = []
    for _ in range(iters):
        Yp = fit0 + res0[rng.permutation(n)]
        s_t, s_r = rss(X0, Yp), rss(X1, Yp)
        rand.append(((s_t - s_r) / df) / (s_r / dfr))
    rand = np.array(rand)
    return {"Df": df, "SS": ss, "MS": ss / df, "Rsq": ss / ss_tot, "F": F, "Z": effect_size(F, rand),
            "P": float((1 + (rand >= F - 1e-12).sum()) / (iters + 1)), "SS_residual": ss_res, "SS_total": ss_tot}


def ancestral_states(Y: np.ndarray, tips: List[Node], nodes: List[Node]) -> np.ndarray:
    """Ancestral states under Brownian motion by GLS: a = mu + C_nt C_tt^-1 (Y - 1 mu), with mu the GLS root state."""
    Ctt = vcv(tips, tips)
    Cnt = vcv(nodes, tips)
    Ci = np.linalg.inv(Ctt)
    one = np.ones((len(tips), 1))
    mu = np.linalg.solve(one.T @ Ci @ one, one.T @ Ci @ Y)
    return mu + Cnt @ Ci @ (Y - one @ mu)


# ============================================================================ data
def _is_number(v: str) -> bool:
    try:
        float(v); return True
    except ValueError:
        return False


def _key_forms(s: str) -> List[str]:
    """an id with and without Descriptron's annotation suffix ("x.png_3") and extension."""
    s = str(s).strip(); base = re.sub(r"\.(png|jpe?g|tiff?|bmp)_\d+$", r".\1", s, flags=re.I)
    stem = re.sub(r"\.(png|jpe?g|tiff?|bmp)$", "", base, flags=re.I)
    forms = [s, base, stem, Path(stem).name]
    # some steps write file names with spaces replaced by underscores ("Scan 001" -> "Scan_001")
    return forms + [f.replace(" ", "_") for f in forms if " " in f]


def load_group_map(path: str, group_col: Optional[str] = None) -> Dict[str, str]:
    """group labels CSV (filename + group_label/species): id -> species."""
    rows = list(csv.DictReader(open(path, newline=""), delimiter="\t" if str(path).endswith(".tsv") else ","))
    lc = {c.strip().lower(): c for c in rows[0]}
    fcol = lc.get("filename") or lc.get("file") or lc.get("specimen") or list(rows[0])[0]
    gcol = group_col or lc.get("group_label") or lc.get("species") or list(rows[0])[1]
    m: Dict[str, str] = {}
    for r in rows:
        for k in _key_forms(r[fcol]):
            m.setdefault(k, r[gcol])
    return m


def load_traits(path: str, species_col: Optional[str], species_regex: Optional[str], id_col: Optional[str],
                groups: Optional[Dict[str, str]] = None):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        sys.exit(f"{path}: no rows")
    cols = list(rows[0])
    if groups is not None and not (species_col and species_col in cols):     # species from a group-labels file
        idc = id_col or next((c for c in ("filename", "specimen_id", "specimen", "image", "id", "file") if c in cols), cols[0])
        sp = [next((groups[k] for k in _key_forms(r[idc]) if k in groups), None) for r in rows]
        species_col = idc                                                # the id column is not a trait
    elif species_col and species_col in cols:
        sp = [r[species_col] for r in rows]
    else:
        idc = id_col or next((c for c in ("filename", "specimen_id", "specimen", "image", "id", "file") if c in cols), cols[0])
        if not species_regex:
            sys.exit(f"{path}: give --species_col or --species_regex (no species column found)")
        rx = re.compile(species_regex)
        sp = []
        for r in rows:
            m = rx.search(r[idc])
            sp.append(m.group(1) if m else None)
    num = [c for c in cols if c != species_col and all(_is_number(r[c]) or r[c] == "" for r in rows)
           and any(r[c] != "" for r in rows)]
    M = np.array([[float(r[c]) if r[c] != "" else np.nan for c in num] for r in rows])
    by: Dict[str, List[int]] = {}
    for i, s in enumerate(sp):
        if s:
            by.setdefault(s, []).append(i)
    species = sorted(by)
    if not species:
        sys.exit(f"{path}: no row could be given a species (check the id column against the group labels / regex)")
    means = np.array([np.nanmean(M[by[s]], axis=0) for s in species])
    keep = ~np.isnan(means).any(axis=0)                  # drop columns missing for any species
    return species, [c for c, k in zip(num, keep) if k], means[:, keep], {s: len(v) for s, v in by.items()}


def scale(M: np.ndarray, how: str) -> np.ndarray:
    if how == "standardize":
        sd = M.std(axis=0, ddof=1)
        ok = sd > 0
        return (M[:, ok] - M[:, ok].mean(0)) / sd[ok]
    if how == "log":
        if (M <= 0).any():
            sys.exit("--scale log needs positive values")
        return np.log(M)
    return M


# ============================================================================ plot
def plot_morphospace(pc_tips, pc_nodes, tips, nodes, ev, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = {id(t): pc_tips[i] for i, t in enumerate(tips)}
    pos.update({id(n): pc_nodes[i] for i, n in enumerate(nodes)})
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n in nodes:
        for k in n.kids:
            a, b = pos[id(n)], pos[id(k)]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#6b7780", lw=0.9, zorder=1)
    ax.scatter(pc_nodes[:, 0], pc_nodes[:, 1], s=14, color="white", edgecolor="#6b7780", zorder=2, label="ancestral states")
    ax.scatter(pc_tips[:, 0], pc_tips[:, 1], s=30, color="#0072B2", zorder=3, label="species means")
    for i, t in enumerate(tips):
        ax.annotate(t.name, pc_tips[i, :2], xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, help="Newick tree whose tips are species names")
    ap.add_argument("--traits", action="append", required=True, metavar="NAME=CSV",
                    help="a trait table; repeat for several sets (shape, measurements, colour, texture, ...)")
    ap.add_argument("--species_col", default=None, help="column holding the species (default: use --species_regex)")
    ap.add_argument("--species_regex", default=None, help="regex with one group extracting the species from the id column")
    ap.add_argument("--groups", default=None, help="group labels CSV (filename + group_label/species) giving each "
                    "specimen's species, as for the rest of the pipeline (instead of --species_col/--species_regex)")
    ap.add_argument("--id_col", default=None, help="id column for --species_regex (default: filename/specimen_id/...)")
    ap.add_argument("--columns", action="append", default=[], metavar="NAME=REGEX",
                    help="keep only the columns of trait set NAME that match REGEX")
    ap.add_argument("--scale", action="append", default=[], metavar="NAME=none|standardize|log")
    ap.add_argument("--size_col", default=r"centroid|csize",
                    help="regex for a size column kept out of the trait set and used as the logCS predictor")
    ap.add_argument("--pgls", action="append", default=[], metavar="SET~PREDICTOR",
                    help="PREDICTOR is logCS (from a set's centroid-size column) or SET2:COLUMN")
    ap.add_argument("--analyses", nargs="+", default=["signal", "morphospace"], choices=["signal", "pgls", "morphospace"])
    ap.add_argument("--iterations", type=int, default=999); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    root = parse_newick(Path(a.tree).read_text())
    colsel = dict(x.split("=", 1) for x in a.columns)
    gmap = load_group_map(a.groups) if a.groups else None
    scales = dict(x.split("=", 1) for x in a.scale)

    sets = {}
    for spec in a.traits:
        name, path = spec.split("=", 1)
        sp, cols, M, counts = load_traits(path, a.species_col, a.species_regex, a.id_col, gmap)
        if name in colsel:
            rx = re.compile(colsel[name]); k = [i for i, c in enumerate(cols) if rx.search(c)]
            cols, M = [cols[i] for i in k], M[:, k]
        # a size column (centroid size) is kept aside: it is a predictor (logCS), not part of the trait set
        szi = [i for i, c in enumerate(cols) if re.search(a.size_col, c, re.I)]
        size = M[:, szi[0]] if szi else None
        keep = [i for i in range(len(cols)) if i not in szi]
        sets[name] = {"species": sp, "cols": [cols[i] for i in keep], "raw": M[:, keep], "size": size,
                      "counts": counts}

    report = ["# Phylogenetic comparative analyses", "", f"Tree: {a.tree}", ""]
    signal_rows, pgls_rows = [], []
    for name, S in sets.items():
        tipnames = {t.name for t in tips_of(root)}
        common = [s for s in S["species"] if s in tipnames]
        if len(common) < 4:
            report.append(f"{name}: only {len(common)} species match the tree - skipped"); continue
        tree = prune(parse_newick(Path(a.tree).read_text()), set(common))
        tips = tips_of(tree)
        idx = [S["species"].index(t.name) for t in tips]
        how = scales.get(name, "none")
        Y = scale(S["raw"][idx], how)
        if Y.shape[1] == 0 or not np.isfinite(Y).all() or float(Y.var(axis=0).sum()) == 0:
            report += [f"## {name}", "no variable differs among the species (for example empty homology cells) - skipped", ""]
            continue
        C = vcv(tips, tips)
        S.update(tree=tree, tips=tips, Y=Y, C=C, idx=idx)
        dropped = sorted(set(S["species"]) - set(common))
        report += [f"## {name}", f"{len(tips)} species, {Y.shape[1]} variables (scale: {how})"
                   + (f"; not in the tree: {', '.join(dropped)}" if dropped else ""), ""]
        if "signal" in a.analyses:
            r = phylo_signal(Y, C, a.iterations, rng)
            signal_rows.append({"trait_set": name, "n_species": len(tips), "n_variables": Y.shape[1], "scale": how, **r})
            report.append(f"Phylogenetic signal: Kmult = {r['Kmult']:.4f}, P = {r['P']:.4f}, Z = {r['Z']:.2f}")
        if "morphospace" in a.analyses and Y.shape[1] < 2:
            report.append("Phylomorphospace: fewer than two variables - not drawn")
        elif "morphospace" in a.analyses:
            nodes = internal_nodes(tree)
            anc = ancestral_states(Y, tips, nodes)
            mu = Y.mean(0)
            U, sv, Vt = np.linalg.svd(Y - mu, full_matrices=False)
            ev = sv ** 2 / (sv ** 2).sum()
            pc_t, pc_n = (Y - mu) @ Vt.T, (anc - mu) @ Vt.T
            np.savetxt(out / f"{name}_morphospace_tips.csv", pc_t[:, :min(5, pc_t.shape[1])], delimiter=",",
                       header=",".join(f"PC{i + 1}" for i in range(min(5, pc_t.shape[1]))), comments="")
            plot_morphospace(pc_t, pc_n, tips, nodes, ev, out / f"{name}_phylomorphospace.png",
                             f"{name}: phylomorphospace ({len(tips)} species)")
            report.append(f"Phylomorphospace: {name}_phylomorphospace.png (PC1 {100 * ev[0]:.1f}%, PC2 {100 * ev[1]:.1f}%)")
        report.append("")

    if "pgls" in a.analyses:
        for f in a.pgls:
            ys, pred = [x.strip() for x in f.split("~")]
            S = sets.get(ys)
            if S is None or "Y" not in S:
                report.append(f"PGLS {f}: trait set {ys} not analysed - skipped"); continue
            if pred == "logCS":
                if S["size"] is None:
                    report.append(f"PGLS {f}: no size column in {ys} (--size_col) - skipped"); continue
                xv = np.log(np.array([S["size"][S["species"].index(t.name)] for t in S["tips"]]))
            else:
                s2, cc = pred.split(":", 1); src = sets.get(s2)
                if src is None or cc not in src["cols"]:
                    report.append(f"PGLS {f}: predictor column not found - skipped"); continue
                xv = np.array([src["raw"][src["species"].index(t.name), src["cols"].index(cc)] for t in S["tips"]])
            r = pgls(S["Y"], xv, S["C"], a.iterations, rng)
            pgls_rows.append({"model": f, **r})
            report.append(f"PGLS {f}: R2 = {r['Rsq']:.4f}, F = {r['F']:.3f}, P = {r['P']:.4f}")

    for fn, rows in (("phylogenetic_signal.csv", signal_rows), ("pgls.csv", pgls_rows)):
        if rows:
            with open(out / fn, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "phylo_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
