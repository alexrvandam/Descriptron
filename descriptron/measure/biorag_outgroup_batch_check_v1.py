#!/usr/bin/env python3
"""
biorag_outgroup_batch_check_v1.py — is the outgroup separation biology or provenance?
=====================================================================================

When material from outside the reference set is scored as novel, the result is
only as good as the assumption that those images were produced like the
reference images. If they were annotated by a different hand, at a different
magnification, or with a different convention for where a cell boundary lies,
the whole set is displaced from the reference cloud in one direction and every
one of them scores as an outlier — for reasons that have nothing to do with the
animals.

This separates the two explanations without needing new material:

  shared direction   The mean z-score per character over all outgroup images. A
                     provenance artefact pushes every image the same way in the
                     same characters; different genera should not agree on which
                     characters are extreme.
  agreement between  Mean pairwise correlation of the outgroup images' z-vectors,
  outgroups          against the same statistic computed on the reference species
                     themselves (which differ only biologically, by construction).
                     If unrelated genera resemble each other more than congeneric
                     species do, that resemblance is the pipeline.
  batch vs batch     Distance between images from different source folders
                     compared with distance within a folder. A folder should not
                     predict shape.

  python biorag_outgroup_batch_check_v1.py --matrix_dir "$M/compiled_key_tier" \\
      --taxon_profile <profile.yaml> --candidate_coco <outgroups.json> \\
      --category_map '{"entire_forewing": "whole_wing"}' --out_dir "$M/outgroup_check"
"""

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                    # noqa: E402
from biorag_novelty_score_v1 import Reference, features_from_coco      # noqa: E402

VERSION = "1.0"


def batch_of(name: str) -> str:
    """Source folder, or 'standalone' for an image sitting on its own."""
    p = str(name).replace("\\", "/")
    return p.rsplit("/", 1)[0] if "/" in p else "standalone"


def mean_pairwise_corr(M: np.ndarray) -> float:
    """Mean Pearson correlation between rows, over the characters both rows have."""
    vals = []
    for i, j in combinations(range(len(M)), 2):
        a, b = M[i], M[j]
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() >= 4 and np.std(a[ok]) > 1e-9 and np.std(b[ok]) > 1e-9:
            vals.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Provenance check on out-of-reference material")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--candidate_coco", required=True)
    ap.add_argument("--category_map", default=None)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    ref = Reference(Path(a.matrix_dir), profile)
    ext = features_from_coco(Path(a.candidate_coco), profile,
                            json.loads(a.category_map) if a.category_map else {})
    if ext.empty:
        sys.exit("no usable outgroup features")

    rt = ref.tables["ratio"]
    shared = [c for c in ext.columns if c in rt.columns]
    ext = ext[shared]
    print(f"{len(ext)} outgroup images, {len(shared)} characters shared with the reference set")

    mu, sd = rt[shared].mean(), rt[shared].std(ddof=1).replace(0, np.nan)
    z_ext = (ext - mu) / sd

    # 1. is there a direction they all share?
    per_char = pd.DataFrame({"mean_z": z_ext.mean(), "sd_z": z_ext.std(ddof=1),
                             "n": z_ext.notna().sum(),
                             "share_same_sign": z_ext.apply(
                                 lambda c: max((c > 0).sum(), (c < 0).sum()) / max(1, c.notna().sum()))})
    per_char = per_char.sort_values("mean_z", key=abs, ascending=False)

    # 2. do unrelated genera agree with each other more than congeneric species do?
    sp_ids = {s: [i for i in rt.index if ref.species_of.get(i) == s] for s in ref.species}
    cent = pd.DataFrame({s: ((rt.loc[ids, shared].mean() - mu) / sd) for s, ids in sp_ids.items()
                         if ids}).T
    agree_out = mean_pairwise_corr(z_ext.values)
    agree_ref = mean_pairwise_corr(cent.values)

    # 3. does the source folder predict shape?
    ext = ext.copy()
    ext["batch"] = [batch_of(i) for i in ext.index]
    z_ext2 = z_ext.copy()
    z_ext2["batch"] = ext["batch"].values
    within, between = [], []
    for i, j in combinations(range(len(z_ext)), 2):
        a_, b_ = z_ext.values[i], z_ext.values[j]
        ok = ~np.isnan(a_) & ~np.isnan(b_)
        if ok.sum() < 4:
            continue
        d = float(np.sqrt(np.mean((a_[ok] - b_[ok]) ** 2)))
        (within if ext["batch"].values[i] == ext["batch"].values[j] else between).append(d)

    summary = {
        "version": VERSION, "outgroup_images": int(len(z_ext)), "characters": shared,
        "batches": {k: int(v) for k, v in ext["batch"].value_counts().items()},
        "mean_abs_z": round(float(np.nanmean(np.abs(z_ext.values))), 3),
        "characters_with_a_consistent_shift": per_char.head(5).round(3).to_dict("index"),
        "agreement_between_outgroup_images": round(agree_out, 3),
        "agreement_between_reference_species": round(agree_ref, 3),
        "interpretation_agreement":
            ("outgroup images resemble each other MORE than the reference species resemble each "
             "other: consistent with a shared provenance signature"
             if agree_out > agree_ref + 0.15 else
             "outgroup images do not resemble each other more than the reference species do: "
             "no evidence of a shared provenance signature in this statistic"),
        "distance_within_a_batch": round(float(np.mean(within)), 3) if within else None,
        "distance_between_batches": round(float(np.mean(between)), 3) if between else None,
        "interpretation_batch":
            ("images from the same folder are markedly closer than images from different folders: "
             "the folder predicts shape, which a taxon should not"
             if within and between and np.mean(within) < 0.75 * np.mean(between) else
             "the source folder does not predict shape"),
    }
    (out / "outgroup_batch_check.json").write_text(json.dumps(summary, indent=2))
    per_char.round(3).to_csv(out / "outgroup_z_by_character.tsv", sep="\t")
    z_ext.round(3).to_csv(out / "outgroup_z_by_image.tsv", sep="\t")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("characters", "characters_with_a_consistent_shift")}, indent=1))
    print("\nper character (largest shifts first):")
    print(per_char.head(10).round(2).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
