#!/usr/bin/env python3
"""
biorag_label_conflict_check_v1.py — is any photograph counted under two species?
=================================================================================

A compiled feature table that was merged from several runs, or whose group labels were revised
after a species was split, can hold rows measured on ONE photograph under TWO group labels. The
matrix builder then averages one species' measurements into another's specimens, and may create
specimens that do not exist. This check reads a compiled directory (`*_full_features.csv` with
`image_base`, `group_label`, `category`) and reports

  * every image that occurs under more than one group label, and
  * every image whose file name names a group other than the one it is labelled with, when the
    name of another group of the table occurs in it as a whole token (e.g. `..._sp1A_3_...`
    labelled `sp1`).

  python biorag_label_conflict_check_v1.py --compiled_dir <dir> --taxon_profile <yaml> --out_dir <dir>

Exit status 1 when a conflict is found, so a pipeline can stop on it. No model is called.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol      # noqa: E402

VERSION = "1.0"


def main():
    ap = argparse.ArgumentParser(description="Photographs counted under two group labels")
    ap.add_argument("--compiled_dir", required=True)
    ap.add_argument("--taxon_profile", default=None, help="when given, the specimens affected are named")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = next(iter(sorted(Path(a.compiled_dir).glob("*_full_features.csv"))), None)
    if src is None:
        sys.exit(f"no *_full_features.csv in {a.compiled_dir}")
    f = pd.read_csv(src, usecols=lambda c: c in ("image_base", "group_label", "category"), low_memory=False)
    groups = sorted(f["group_label"].dropna().astype(str).unique(), key=len, reverse=True)

    two = f.groupby("image_base")["group_label"].nunique()
    two = sorted(two[two > 1].index)

    def named(image):
        toks = re.split(r"[^A-Za-z0-9.]+", str(image))
        return [g for g in groups if g in toks]
    rows = []
    for img, lab in f[["image_base", "group_label"]].drop_duplicates().itertuples(index=False):
        n = named(img)
        if n and str(lab) not in n:
            rows.append({"image_base": img, "labelled": lab, "file_name_says": "; ".join(n),
                         "rows": int(((f["image_base"] == img) & (f["group_label"] == lab)).sum())})
    bad = pd.DataFrame(rows)
    affected = []
    if len(bad) and a.taxon_profile:
        prof = pol.load_taxon_profile(a.taxon_profile)
        bad["specimen_it_is_merged_into"] = [pol.specimen_id(i, l, prof) for i, l in zip(bad["image_base"], bad["labelled"])]
        ok = f[~f["image_base"].isin(bad["image_base"])]
        real = {pol.specimen_id(i, l, prof) for i, l in ok[["image_base", "group_label"]].drop_duplicates().itertuples(index=False)}
        affected = sorted(set(bad["specimen_it_is_merged_into"]))
        phantom = [s for s in affected if s not in real]
    else:
        phantom = []
    bad.to_csv(out / "label_conflicts.tsv", sep="\t", index=False)
    summary = {"version": VERSION, "table": src.name, "images": int(f["image_base"].nunique()),
               "images_under_two_labels": two,
               "rows_labelled_against_their_file_name": int(bad["rows"].sum()) if len(bad) else 0,
               "images_labelled_against_their_file_name": int(len(bad)),
               "by_label_pair": (bad.groupby(["labelled", "file_name_says"])["rows"].sum().reset_index().to_dict("records")
                                 if len(bad) else []),
               "specimens_receiving_foreign_rows": affected,
               "specimens_that_exist_only_from_foreign_rows": phantom}
    (out / "label_conflicts_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    sys.exit(1 if (two or len(bad)) else 0)


if __name__ == "__main__":
    main()
