#!/usr/bin/env python3
"""
biorag_add_discrete_characters_v1.py — put present/absent characters into a Tier-1 matrix
=========================================================================================

The Tier-1 matrix built by biorag_key_feature_filter_v2.py holds measurements only. Many
groups are keyed on states — a seta present or absent, a carina complete or interrupted —
and a taxonomist usually has those already, coded per specimen. This script appends such a
table to a copy of the matrix so every downstream step (key builder, calibration, graded key,
instrument comparison) sees them as ordinary characters. Where the states come from is not
its business: a spreadsheet coded at the bench, or biorag_vlm_characters_v1.py.

Two things are decided here and recorded in the feature dictionary, because both have bitten:

  tier     the key builder reads tier == "key" and nothing else. A character appended with
           any other tier is silently invisible to it, and a key that comes back unchanged
           then looks like a finding. Default "key"; the script prints how many features the
           builder will see before and after so the difference cannot go unnoticed.
  family   "presence_absence", so that a step which must treat states differently from
           measurements can tell them apart (the builder grades them by series length,
           not by a gap in standard deviations; see binary_quality there).

  python biorag_add_discrete_characters_v1.py --matrix_dir "$M/compiled_key_tier" \\
      --states "$M/vlm_combined/vlm_character_states.tsv" --out_dir "$M/compiled_key_tier_plus_vlm" \\
      [--tier key] [--prefix vlm] [--min_species 2] [--reliability <tsv> --min_agreement 0.8]
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

VERSION = "1.0"
PRESENT = {"present", "1", "1.0", "true", "yes", "y"}
ABSENT = {"absent", "0", "0.0", "false", "no", "n"}


def slug(s: str) -> str:
    return re.sub(r"[^\w,\-]+", "_", str(s).strip()).strip("_")


def main():
    ap = argparse.ArgumentParser(description="Append present/absent characters to a Tier-1 matrix")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--states", required=True,
                    help="long TSV: specimen_id, character ('structure:name'), state")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tier", choices=["key", "description"], default="key")
    ap.add_argument("--prefix", default="state", help="marks the column, e.g. 'vlm' or 'coded'")
    ap.add_argument("--min_species", type=int, default=2,
                    help="drop a character scored in fewer species than this")
    ap.add_argument("--reliability", default=None,
                    help="optional TSV with columns character, agreement (test-retest)")
    ap.add_argument("--min_agreement", type=float, default=0.8)
    a = ap.parse_args()

    src, out = Path(a.matrix_dir), Path(a.out_dir)
    if out.resolve() == src.resolve():
        sys.exit("refusing to write over the source matrix; give a different --out_dir")
    out.mkdir(parents=True, exist_ok=True)
    fd = pd.read_csv(src / "feature_dictionary.tsv", sep="\t")
    long = pd.read_csv(src / "specimen_matrix_long.csv")
    st = pd.read_csv(a.states, sep="\t")
    st["state"] = st["state"].astype(str).str.strip().str.lower()
    unknown = sorted(set(st["state"]) - PRESENT - ABSENT)
    if unknown:
        print(f"  ignoring {int(st['state'].isin(unknown).sum())} rows with states that are neither "
              f"present nor absent: {unknown[:6]}")
    st = st[st["state"].isin(PRESENT | ABSENT)].copy()
    st["value"] = st["state"].isin(PRESENT).astype(float)

    known = dict(zip(long["specimen_id"], long["species"]))
    sex = dict(zip(long["specimen_id"], long["sex"]))
    stray = sorted(set(st["specimen_id"]) - set(known))
    if stray:
        print(f"  {len(stray)} specimens in the states table are not in the matrix and are skipped: "
              f"{stray[:5]}")
    st = st[st["specimen_id"].isin(known)]

    dropped = {}
    if a.reliability and Path(a.reliability).exists():
        rel = pd.read_csv(a.reliability, sep="\t")
        col = next((c for c in ("agreement", "retest_agreement", "exact") if c in rel.columns), None)
        if col:
            bad = set(rel.loc[rel[col] < a.min_agreement, "character"])
            dropped["not reproducible on retest"] = sorted(bad & set(st["character"]))
            st = st[~st["character"].isin(bad)]
    nsp = st.assign(sp=st["specimen_id"].map(known)).groupby("character")["sp"].nunique()
    thin = set(nsp[nsp < a.min_species].index)
    const = set(st.groupby("character")["value"].nunique().loc[lambda s: s < 2].index)
    dropped["scored in too few species"] = sorted(thin)
    dropped["same state in every specimen"] = sorted(const - thin)
    st = st[~st["character"].isin(thin | const)]

    rows_fd, rows_long = [], []
    for ch, g in st.groupby("character"):
        struct, _, name = str(ch).partition(":")
        name = name or struct
        fid = f"{struct}.{a.prefix}_{slug(name)}"
        g = g.groupby("specimen_id")["value"].mean().round()          # one state per specimen
        rows_fd.append({"feature_id": fid, "category": struct, "base_category": struct,
                        "column": f"{a.prefix}_state", "tier": a.tier, "family": "presence_absence",
                        "label": f"{struct}: {name}", "definition": f"{name} (scored present/absent)",
                        "unit": "", "structure_sex": "both", "section": struct, "key_priority": 1,
                        "n_species": int(nsp[ch]), "n_specimens": int(len(g)), "conversion": ""})
        for sid, v in g.items():
            rows_long.append({"species": known[sid], "specimen_id": sid, "sex": sex.get(sid),
                              "category": struct, "base_category": struct,
                              "column": f"{a.prefix}_state", "feature_id": fid, "tier": a.tier,
                              "family": "presence_absence", "value": float(v), "n_images": 1})
    add_fd = pd.DataFrame(rows_fd)
    clash = set(add_fd["feature_id"]) & set(fd["feature_id"]) if len(add_fd) else set()
    if clash:
        sys.exit(f"feature ids already in the matrix: {sorted(clash)[:5]} — choose another --prefix")
    fd2 = pd.concat([fd, add_fd], ignore_index=True)
    long2 = pd.concat([long, pd.DataFrame(rows_long)], ignore_index=True)
    fd2.to_csv(out / "feature_dictionary.tsv", sep="\t", index=False)
    long2.to_csv(out / "specimen_matrix_long.csv", index=False)
    for extra in ("outlier_flags.tsv", "filter_report_v2.json"):
        if (src / extra).exists():
            shutil.copy2(src / extra, out / extra)

    before, after = int((fd["tier"] == "key").sum()), int((fd2["tier"] == "key").sum())
    report = {"version": VERSION, "source_matrix": str(src), "states": str(a.states),
              "tier": a.tier, "characters_added": len(add_fd), "values_added": len(rows_long),
              "characters_dropped": {k: len(v) for k, v in dropped.items()},
              "dropped": dropped,
              "features_the_key_builder_reads": {"before": before, "after": after}}
    (out / "discrete_characters_report.json").write_text(json.dumps(report, indent=2))
    print(f"added {len(add_fd)} present/absent characters ({len(rows_long)} values); dropped "
          + ", ".join(f"{len(v)} {k}" for k, v in dropped.items()))
    print(f"features the key builder reads (tier == 'key'): {before} -> {after}")
    if a.tier != "key":
        print("  NOTE: tier is not 'key', so the key builder will not see these characters")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
