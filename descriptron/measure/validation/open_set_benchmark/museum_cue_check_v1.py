#!/usr/bin/env python3
"""
museum_cue_check_v1.py - does an instrument see the museum (photographic set-up) rather than the animal?
=====================================================================================================
Within every species photographed in two or more collections, each specimen's nearest OTHER specimen of the same
species is found; if it comes from the same collection more often than a random same-species specimen would, the
instrument carries a collection (lighting, camera, background, mounting) signal. Species identity is held fixed, so
this is not confounded with species differences.

  observed  share of specimens whose nearest same-species neighbour is from the same collection
  expected  the same share if the neighbour were drawn at random from the specimen's own species
  P         permutation test: collection labels shuffled within species (default 2,000 permutations)

Instruments: continuous tables (specimen_id x features; standardised, Euclidean distance on the shared columns) and
embedding tables (cosine). Each is given as NAME=TABLE.tsv[:cosine].

  python museum_cue_check_v1.py --institution specimen_institution.csv \
      --table colour=colour_set.tsv --table bioclip_raw=embeddings_raw/bioclip.tsv:cosine --out_dir museum_check/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def dist_matrix(T, cosine):
    X = T.values.astype(float)
    if cosine:
        X = np.nan_to_num(X); X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return 1 - X @ X.T
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0); sd[sd == 0] = 1
    Z = (X - mu) / sd; ok = ~np.isnan(Z); Z = np.nan_to_num(Z)
    n = ok.astype(float) @ ok.T.astype(float)
    sq = (Z ** 2) @ ok.T.astype(float) + ok.astype(float) @ (Z ** 2).T - 2 * Z @ Z.T
    return np.sqrt(np.maximum(sq, 0) / np.maximum(n, 1))


def same_share(D, sp, inst, idx):
    hits = []
    for i in idx:
        cand = [j for j in idx if j != i and sp[j] == sp[i]]
        j = cand[int(np.argmin(D[i, cand]))]
        hits.append(inst[j] == inst[i])
    return float(np.mean(hits))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--institution", required=True, help="CSV: specimen_id, species, institution")
    ap.add_argument("--table", action="append", required=True, metavar="NAME=TABLE.tsv[:cosine]")
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    I = pd.read_csv(a.institution)
    multi = I.groupby("species").institution.nunique()
    keep_sp = set(multi[multi > 1].index)
    I = I[I.species.isin(keep_sp)].set_index("specimen_id")
    rng = np.random.default_rng(a.seed); rows = []
    for spec in a.table:
        name, rest = spec.split("=", 1)
        cosine = rest.endswith(":cosine"); path = rest[:-7] if cosine else rest
        T = pd.read_csv(path, sep="\t").set_index("specimen_id") if path.endswith(".tsv") else pd.read_csv(path).set_index("specimen_id")
        T = T.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        ids = [i for i in I.index if i in T.index]
        D = dist_matrix(T.loc[ids], cosine)
        sp = I.loc[ids, "species"].values; inst = I.loc[ids, "institution"].values; idx = range(len(ids))
        obs = same_share(D, sp, inst, idx)
        exp = float(np.mean([np.mean([inst[j] == inst[i] for j in idx if j != i and sp[j] == sp[i]]) for i in idx]))
        null = []
        for _ in range(a.permutations):
            perm = inst.copy()
            for s in set(sp):
                k = np.where(sp == s)[0]; perm[k] = rng.permutation(perm[k])
            null.append(same_share(D, sp, perm, idx))
        p = (1 + sum(n >= obs for n in null)) / (1 + a.permutations)
        rows.append({"instrument": name, "specimens": len(ids), "species": len(set(sp)), "observed_same_collection": round(obs, 3),
                     "expected_at_random": round(exp, 3), "excess": round(obs - exp, 3), "P": round(p, 4)})
        print(rows[-1], flush=True)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "museum_cue.tsv", sep="\t", index=False)
    json.dump({"species_in_2plus_collections": len(keep_sp), "permutations": a.permutations}, open(out / "museum_cue.json", "w"), indent=2)


if __name__ == "__main__":
    main()
