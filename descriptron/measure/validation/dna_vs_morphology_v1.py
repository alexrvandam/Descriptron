#!/usr/bin/env python3
"""
dna_vs_morphology_v1.py - do DNA barcodes support the morphospecies, and the specimens Descriptron flags?
=======================================================================================================

Independent evidence for the species hypotheses that Descriptron's hold-out tests take as given.

  1. distances      Kimura 2-parameter (K2P) distances between aligned barcodes (pairwise deletion of gaps / N)
  2. barcode gap    per morphospecies: largest distance within it vs smallest distance to another species
  3. DNA groups     single-linkage groups at a range of K2P thresholds; agreement with the morphospecies
                    (adjusted Rand index; species split, species merged)
  4. specimens      for sequences that are also matrix specimens: the nearest other sequence's species (DNA verdict)
                    against the morphospecies label and the matrix's leave-one-out name (calibration distances)

Sequence names are mapped to species codes and specimen ids by a regex and an alias table, so the script is
reusable for any taxon:
  --name_regex 'Diaphorina-([A-Za-z0-9.]+)-(\\d+)$'  (group 1 = species code, group 2 = specimen number)
  --extra_regex 'Diaphorina-(cf-carrisae|cf-enderleini|sp1A)$'  (names with a species code but no specimen number)
  (these two are the settings used for the Diaphorina barcodes)
  --alias virgata=virg --alias turneri=turn --alias 'cf-carrisae=cf.carrisae' ...
Names that do not match, or map to a species outside --species, are kept as extra taxa (references, other
species) in distances and groups but are not scored as morphospecies.

  python dna_vs_morphology_v1.py --fasta barcodes.fas --species_list species.txt --calibration_dir M/calibration \
      --out_dir out/
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PURINES, PYRIMIDINES = set("AG"), set("CT")


def read_fasta(path):
    seqs, cur = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            cur = line[1:].strip(); seqs[cur] = []
        elif cur is not None:
            seqs[cur].append(line.upper())
    return {k: "".join(v) for k, v in seqs.items()}


def k2p_matrix(seqs):
    """K2P distance, pairwise deletion of positions with gaps or ambiguity codes in either sequence."""
    names = list(seqs)
    L = max(len(s) for s in seqs.values())
    A = np.array([list(seqs[n].ljust(L, "-")) for n in names])
    valid = np.isin(A, list("ACGT"))
    pur = np.isin(A, list("AG"))
    n = len(names); D = np.zeros((n, n)); NS = np.zeros((n, n), int)
    for i in range(n):
        v = valid[i] & valid
        diff = (A[i] != A) & v
        trans = diff & (pur[i] == pur)                    # both purines or both pyrimidines: transition
        tv = diff & ~(pur[i] == pur)
        nsite = v.sum(1)
        P = trans.sum(1) / np.maximum(nsite, 1); Q = tv.sum(1) / np.maximum(nsite, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = -0.5 * np.log(1 - 2 * P - Q) - 0.25 * np.log(1 - 2 * Q)
        D[i] = np.where(np.isfinite(d), d, np.nan); NS[i] = nsite
    np.fill_diagonal(D, 0.0)
    return names, D, NS


def single_linkage(D, t):
    n = len(D); parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] == D[i, j] and D[i, j] <= t:
                parent[find(i)] = find(j)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


def ari(a, b):
    from math import comb
    ct = Counter(zip(a, b)); ra = Counter(a); rb = Counter(b); n = len(a)
    s_ij = sum(comb(v, 2) for v in ct.values()); s_a = sum(comb(v, 2) for v in ra.values()); s_b = sum(comb(v, 2) for v in rb.values())
    exp = s_a * s_b / comb(n, 2) if n > 1 else 0
    mx = (s_a + s_b) / 2
    return (s_ij - exp) / (mx - exp) if mx != exp else 1.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--name_regex", default=r"([A-Za-z0-9.]+)[-_](\d+)$",
                    help="regex on the sequence name: group 1 = species code, group 2 = specimen number "
                         "(default: '<code>-<n>' or '<code>_<n>' at the end of the name)")
    ap.add_argument("--extra_regex", nargs="*", default=[],
                    help="regexes for names whose group 1 is a species code without a specimen number")
    ap.add_argument("--alias", action="append", default=[], metavar="NAME=CODE")
    ap.add_argument("--species", nargs="*", default=None, help="the morphospecies codes to score (default: from --calibration_dir)")
    ap.add_argument("--calibration_dir", default=None, help="Descriptron calibration folder (matrix_identification_distances.tsv)")
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08])
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    alias = dict(x.split("=", 1) for x in a.alias)
    seqs = read_fasta(a.fasta)
    mat = None
    if a.calibration_dir:
        mat = pd.read_csv(Path(a.calibration_dir) / "matrix_identification_distances.tsv", sep="\t")
    morph = set(a.species or (mat["species"].unique() if mat is not None else []))
    rows = []
    for nm in seqs:
        sp = sid = None
        m = re.search(a.name_regex, nm)
        if m:
            sp = alias.get(m.group(1), m.group(1)); sid = f"{sp}_{m.group(2)}"
        else:
            for rx in a.extra_regex:
                m2 = re.search(rx, nm)
                if m2:
                    sp = alias.get(m2.group(1), m2.group(1)); break
        rows.append({"sequence": nm, "species": sp, "specimen_id": sid, "scored": sp in morph,
                     "in_matrix": bool(mat is not None and sid in set(mat["specimen_id"]))})
    meta = pd.DataFrame(rows)
    names, D, NS = k2p_matrix(seqs)
    meta = meta.set_index("sequence").loc[names].reset_index()
    pd.DataFrame(D, index=names, columns=names).to_csv(out / "k2p_distances.tsv", sep="\t")
    sp = meta["species"].fillna("(other)").values; sc = meta["scored"].values

    # barcode gap per morphospecies
    gap = []
    for s in sorted(set(sp[sc])):
        idx = np.where(sp == s)[0]; oth = np.where(sc & (sp != s))[0]
        intra = D[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)] if len(idx) > 1 else np.array([])
        inter = D[np.ix_(idx, oth)]
        j = np.unravel_index(np.nanargmin(inter), inter.shape)
        gap.append({"species": s, "sequences": len(idx), "max_intra": round(float(np.nanmax(intra)), 4) if intra.size else None,
                    "min_inter": round(float(inter[j]), 4), "nearest_species": sp[oth[j[1]]],
                    "gap": round(float(inter[j] - np.nanmax(intra)), 4) if intra.size else None})
    G = pd.DataFrame(gap); G.to_csv(out / "barcode_gap.tsv", sep="\t", index=False)

    # DNA groups vs morphospecies (scored sequences only)
    Ds = D[np.ix_(sc, sc)]; lab = list(sp[sc])
    thr = []
    for t in a.thresholds:
        g = single_linkage(Ds, t)
        by_sp = defaultdict(set); by_g = defaultdict(set)
        for s_, gg in zip(lab, g):
            by_sp[s_].add(gg); by_g[gg].add(s_)
        split = sorted(s_ for s_, v in by_sp.items() if len(v) > 1)
        merged = sorted("+".join(sorted(v)) for v in by_g.values() if len(v) > 1)
        thr.append({"k2p_threshold": t, "dna_groups": len(set(g)), "morphospecies": len(by_sp), "ARI": round(ari(lab, g), 3),
                    "morphospecies_split": ", ".join(split), "morphospecies_merged": "; ".join(merged)})
    T = pd.DataFrame(thr); T.to_csv(out / "dna_groups_by_threshold.tsv", sep="\t", index=False)

    # specimen level: DNA nearest neighbour vs label vs matrix name
    spec = []
    mnamed = {}
    if mat is not None:
        for _d in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):   # source tree / package
            sys.path.insert(0, str(_d))
        from biorag_congruence_compare_v1 import identify
        m_ = identify(mat, sorted(mat["set"].unique()))
        mnamed = m_["named"].to_dict()
    for k, row in meta.iterrows():
        if not row["scored"]:
            continue
        d = D[k].copy(); d[k] = np.inf
        d[~sc] = np.inf                                     # name from the morphospecies only
        j2 = int(np.nanargmin(d))
        dna_named = sp[j2]
        spec.append({"sequence": row["sequence"], "specimen_id": row["specimen_id"], "species": row["species"],
                     "dna_nearest": dna_named, "dna_nearest_k2p": round(float(d[j2]), 4),
                     # one sequence of a species: its nearest neighbour is necessarily another species - uninformative
                     "dna_agrees_with_label": (dna_named == row["species"]) if (sp == row["species"]).sum() > 1 else None,
                     "matrix_named": mnamed.get(row["specimen_id"], "") if row["in_matrix"] else "",
                     "matrix_agrees_with_label": (mnamed.get(row["specimen_id"]) == row["species"]) if row["in_matrix"] else None})
    S = pd.DataFrame(spec); S.to_csv(out / "specimens_dna_vs_matrix.tsv", sep="\t", index=False)
    inm = S[S["matrix_named"] != ""]
    summ = {"sequences": len(seqs), "scored_sequences": int(sc.sum()), "morphospecies_with_dna": int(len(G)),
            "morphospecies_with_2plus": int((G["sequences"] > 1).sum()),
            "positive_gap": int((G["gap"] > 0).sum()), "negative_gap": int((G["gap"] <= 0).sum()),
            "dna_nearest_agrees": f"{int((S['dna_agrees_with_label'] == True).sum())}/{int(S['dna_agrees_with_label'].notna().sum())} (species with >=2 sequences)",
            "matched_matrix_specimens": len(inm),
            "dna_and_matrix_both_agree": int(((inm["dna_agrees_with_label"] == True) & (inm["matrix_agrees_with_label"] == True)).sum()),
            "best_threshold_by_ARI": T.loc[T["ARI"].idxmax()].to_dict()}
    json.dump(summ, open(out / "summary.json", "w"), indent=2, default=str)
    print(json.dumps(summ, indent=1, default=str))
    return S


if __name__ == "__main__":
    main()
