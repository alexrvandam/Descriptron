#!/usr/bin/env python3
"""
traits_to_matrix_dir_v1.py - per-specimen trait tables -> the character-matrix folder the hold-out tests read
===========================================================================================================
For a taxon run outside the full BioRAG pipeline (e.g. photographs with no scale: no millimetre sizes), this writes
`specimen_matrix_long.csv` and `feature_dictionary.tsv` so biorag_congruence_compare_v1.py and
cosine_knn_check_v1.py score it with exactly the machinery used for the Diaphorina matrix.

Each --set gives a trait CSV (one row per image, a species column, a file column, numeric features) and the
feature FAMILY that places it in one of the matrix's character sets (biorag_novelty_score_v1.SETS):
  ratio           proportions (aspect ratio, solidity, ...)          -> set "ratio"
  landmark_ratio  scale-free shape (e.g. Procrustes outline coords)  -> set "landmark"
  colour          colour pattern (homology-cell colour features)    -> set "colour"
  length / area   sizes in mm                                        -> set "size"
Specimen ids follow the pipeline's rule (species + taxon profile specimen_id.number_regex). Feature ids are
prefixed with the set name so two tables never share a column.

  python traits_to_matrix_dir_v1.py --taxon_profile moths.yaml --out_dir matrix/ \
      --set meas=meas.csv:ratio --set shape=shape.csv:landmark_ratio --set colour=colour.csv:colour
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import biorag_feature_policy as pol                  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, metavar="NAME=CSV:FAMILY")
    ap.add_argument("--species_col", default="species")
    ap.add_argument("--file_col", default="file")
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--keep", default=None, help="optional CSV with a file column: only these images are used")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    prof = pol.load_taxon_profile(a.taxon_profile)
    keep = set(pd.read_csv(a.keep)[a.file_col]) if a.keep else None
    longs, fd, report = [], [], []
    for spec in a.set:
        name, rest = spec.split("=", 1); path, fam = rest.rsplit(":", 1)
        t = pd.read_csv(path)
        if keep is not None:
            t = t[t[a.file_col].isin(keep)]
        sid = [pol.specimen_id(str(f), str(s), prof) for f, s in zip(t[a.file_col], t[a.species_col])]
        num = t.drop(columns=[a.species_col, a.file_col]).apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        num.columns = [f"{name}:{c}" for c in num.columns]
        num.insert(0, "specimen_id", sid); num.insert(1, "species", t[a.species_col].astype(str).values)
        L = num.melt(id_vars=["specimen_id", "species"], var_name="feature_id", value_name="value").dropna(subset=["value"])
        longs.append(L)
        fd += [{"feature_id": c, "family": fam, "set_source": name} for c in num.columns[2:]]
        report.append({"set": name, "family": fam, "file": path, "rows": len(t), "specimens": len(set(sid)),
                       "features": num.shape[1] - 2})
    long = pd.concat(longs)
    long.insert(2, "sex", "unknown")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "specimen_matrix_long.csv", index=False)
    pd.DataFrame(fd).to_csv(out / "feature_dictionary.tsv", sep="\t", index=False)
    json.dump({"sets": report, "specimens": int(long.specimen_id.nunique()), "species": int(long.species.nunique())},
              open(out / "matrix_report.json", "w"), indent=2)
    print(json.dumps(report, indent=1), long.specimen_id.nunique(), "specimens", long.species.nunique(), "species")


if __name__ == "__main__":
    main()
