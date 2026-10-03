#!/usr/bin/env python3
"""
coi_species_tree_v1.py - a species tree for the phylogenetic comparative analyses, from a barcode gene tree
==========================================================================================================

  1. midpoint-root the gene tree (e.g. IQ-TREE .treefile; support values on internal nodes are kept)
  2. per morphospecies: is it monophyletic in the rooted tree? (species with >= 2 sequences)
  3. keep one sequence per morphospecies and name the tip by the species code, so the tree can be passed to
     descriptron_phylo.py --tree with the group labels. The sequence kept is the first, in this order, of:
     a sequence whose DNA nearest neighbour is its own species and that is also a matrix specimen; one whose
     nearest neighbour is its own species; any. Sequences are mapped to species with the table written by
     dna_vs_morphology_v1.py (specimens_dna_vs_matrix.tsv); other sequences (references) are dropped.

  python coi_species_tree_v1.py --tree coi.treefile --specimens dna/specimens_dna_vs_matrix.tsv --out_dir tree/
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# measure/ in the source tree; the same folder in the installed package (tools/)
for _d in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
    sys.path.insert(0, str(_d))
from descriptron_phylo import Node, parse_newick, prune, tips_of, _set_depths      # noqa: E402


def to_newick(n: Node, support: bool = True) -> str:
    if not n.kids:
        return f"{n.name}:{n.length:.6g}"
    lab = n.name if support else ""
    return "(" + ",".join(to_newick(k, support) for k in n.kids) + f"){lab}:{n.length:.6g}"


def midpoint_root(root: Node) -> Node:
    """re-root at the midpoint of the longest tip-to-tip path."""
    nb = {}                                                    # undirected graph: node -> [(node, length)]
    stack = [root]
    while stack:
        n = stack.pop(); nb.setdefault(n, [])
        for k in n.kids:
            nb[n].append((k, k.length)); nb.setdefault(k, []).append((n, k.length)); stack.append(k)

    def far(src):
        dist, prev, st = {src: 0.0}, {src: None}, [src]
        while st:
            u = st.pop()
            for v, w in nb[u]:
                if v not in dist:
                    dist[v] = dist[u] + w; prev[v] = u; st.append(v)
        t = max((x for x in dist if not x.kids), key=dist.get)
        return t, dist, prev
    a, _, _ = far(next(x for x in nb if not x.kids))
    b, dist, prev = far(a)
    half = dist[b] / 2
    x = b                                                      # walk back from b towards a until past the midpoint
    while dist[prev[x]] > half:
        x = prev[x]
    u, v = prev[x], x                                          # the midpoint lies on edge u-v
    new = Node("", 0.0)
    w = next(l for y, l in nb[u] if y is v)
    nb[u] = [(y, l) for y, l in nb[u] if y is not v]; nb[v] = [(y, l) for y, l in nb[v] if y is not u]
    nb[new] = [(u, half - dist[u]), (v, dist[v] - half)]
    nb[u].append((new, half - dist[u])); nb[v].append((new, dist[v] - half))
    del w
    # rebuild the parent/child structure from the new root; support labels stay with the clade below each node
    old_parent = {}
    st = [root]
    while st:
        n = st.pop()
        for k in n.kids:
            old_parent[k] = n; st.append(k)

    def build(n, parent, length):
        n.parent, n.length = parent, length
        kids = [(y, l) for y, l in nb[n] if y is not parent]
        if kids and parent is not None and old_parent.get(parent) is n:
            n.name = ""                                        # direction reversed: the old support no longer applies
        n.kids = []
        for y, l in kids:
            n.kids.append(build(y, n, l))
        if n.kids and len(n.kids) == 1:                        # the old root, left with one child: collapse it
            c = n.kids[0]; c.length += n.length; c.parent = parent
            return c
        return n
    r = build(new, None, 0.0)
    _set_depths(r)
    return r


def clade_tips(n):
    return {t.name for t in tips_of(n)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--specimens", required=True, help="specimens_dna_vs_matrix.tsv from dna_vs_morphology_v1.py")
    ap.add_argument("--min_tip_length", type=float, default=1e-4,
                    help="floor for tip branch lengths in the species tree: identical sequences of two species give a "
                         "zero-length branch and a singular phylogenetic covariance matrix (default 1e-4 subst./site)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    root = midpoint_root(parse_newick(Path(a.tree).read_text()))
    (out / "gene_tree_midpoint.nwk").write_text(to_newick(root) + ";\n")
    S = pd.read_csv(a.specimens, sep="\t")
    sp_of = dict(zip(S["sequence"], S["species"]))
    tips = clade_tips(root)
    # monophyly: the smallest clade holding all sequences of the species, and what else it holds
    mono = []
    for s, g in S.groupby("species"):
        seqs = set(g["sequence"]) & tips
        if len(seqs) < 2:
            continue
        best = None
        st = [root]
        while st:
            n = st.pop()
            ct = clade_tips(n)
            if seqs <= ct and (best is None or len(ct) < len(best[1])):
                best = (n, ct)
            st.extend(n.kids)
        others = sorted(sp_of.get(t, t) for t in best[1] - seqs)
        mono.append({"species": s, "sequences": len(seqs), "monophyletic": not others,
                     "support": best[0].name if not others else "", "intruding": ", ".join(sorted(set(others)))})
    M = pd.DataFrame(mono, columns=["species", "sequences", "monophyletic", "support", "intruding"]); M.to_csv(out / "morphospecies_monophyly.tsv", sep="\t", index=False)
    # one tip per species
    S["rank"] = [0 if (r.dna_agrees_with_label is True or r.dna_agrees_with_label == "True") and isinstance(r.matrix_named, str) and r.matrix_named
                 else 1 if (r.dna_agrees_with_label is True or r.dna_agrees_with_label == "True") else 2 for r in S.itertuples()]
    keep = S[S["sequence"].isin(tips)].sort_values(["species", "rank", "sequence"]).groupby("species").head(1)
    rep = dict(zip(keep["sequence"], keep["species"]))
    sp_tree = prune(root, set(rep))
    floored = []
    for t in tips_of(sp_tree):
        t.name = rep[t.name]
        if t.length < a.min_tip_length:
            floored.append(t.name); t.length = a.min_tip_length
    _set_depths(sp_tree)
    (out / "species_tree.nwk").write_text(to_newick(sp_tree, support=False) + ";\n")
    keep[["species", "sequence", "specimen_id"]].to_csv(out / "species_tree_tips.tsv", sep="\t", index=False)
    summ = {"gene_tree_tips": len(tips), "species_tips": len(rep), "tips_given_min_length": floored,
            "morphospecies_tested": len(M), "monophyletic": int(M["monophyletic"].astype(bool).sum()),
            "not_monophyletic": M.loc[~M["monophyletic"].astype(bool), ["species", "intruding"]].to_dict("records")}
    json.dump(summ, open(out / "species_tree_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
