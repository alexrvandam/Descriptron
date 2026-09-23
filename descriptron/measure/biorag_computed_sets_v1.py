#!/usr/bin/env python3
"""
biorag_computed_sets_v1.py — the computed colour-pattern and texture metrics as
character sets the hold-out harness can read
================================================================================

The pipeline measures far more per structure than the identification matrix uses.
Two families are per-specimen quantities and are simply never looked at:

  colour pattern   the colour-extraction pattern metrics (how many markings, how
                   complex, how sharp their boundaries, the spread of hue /
                   saturation / brightness, the marking area) and the condensed
                   colour-homology grid — the mean and spread of cell entropy and
                   the spread of cell lightness and of a*/b*: what a taxonomist
                   means by "uniform" versus "mottled".
                   This is exactly COMPUTED_GROUPS["colour pattern"] +
                   GRID_SUMMARIES of biorag_novelty_score_v1.

  texture          the texture-homology grid: GLCM contrast / energy /
                   homogeneity and LBP entropy / uniformity, measured cell by
                   cell in a TPS-aligned grid. Condensed per specimen to the mean
                   and the spread over cells (--texture_mode summary, default) or
                   kept cell by cell (--texture_mode grid).

NOT included, and why (written into the report):
  * tex_phylo_PC*, tex_phylo_UMAP*, tex_phylo_cluster / silhouette / optimal_k,
    color_*_PC*, shape_PC*, lmk_* ordinations — these are BATCH statistics: every
    specimen's value is a function of all the others, so a specimen withheld from
    a fold is still inside the number it is scored against. They cannot be held
    out without recomputing the ordination per fold, and here they additionally
    come from two separate batch runs (the 25-species and the 4-species batches),
    so they are not even comparable across the sample.
  * tex_r*c*_dL_rel / da_rel / db_rel — relative colour deviations carried in the
    texture grid; they are colour, not texture, and the colour-homology grid is
    already summarised in the colour-pattern set.

Rows are matched to the matrix's own specimen ids through the compiled table the
matrix was built from (--identity_dir), so the category aliases, the sex split
and the exclusion list are honoured without repeating them here.

Images that were REORIENTED after annotation (--reorient_summary) had their
colour and texture measured off the specimen: those image x structure rows are
set to missing before anything is averaged.

Usage:
  python biorag_computed_sets_v1.py \\
      --compiled_dir "<Diaphorina_compiled_29species>" \\
      --identity_dir "<compiled_key_tier>" \\
      --taxon_profile biorag_prompts/taxon_profiles/diaphorina_taxon_profile.yaml \\
      --reorient_summary "<annotation_reorient/reorientation_summary.json>" \\
      --out_dir "<out>"
"""

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                       # noqa: E402
import biorag_outline_shape_v1 as osh                                     # noqa: E402
from biorag_novelty_score_v1 import COMPUTED_GROUPS, GRID_SUMMARIES       # noqa: E402

VERSION = "1.0"

TEXTURE_CELL_STATS = ["glcm_contrast", "glcm_energy", "glcm_homogeneity",
                      "lbp_entropy", "lbp_uniformity"]
TEXTURE_EXCLUDED = {
    r'^tex_phylo_(PC\d+|UMAP\d*|cluster|silhouette|optimal_k)$':
        "batch ordination: every value depends on the whole sample, so it cannot be held out; "
        "here also computed in two separate batches",
    r'^tex_r\d+c\d+_d[Lab]_rel$':
        "relative colour deviation carried in the texture grid — colour, not texture, and the "
        "colour-homology grid is already condensed in the colour-pattern set",
}
DEFAULT_REORIENT_CATEGORIES = "whole_wing,pterostigma,cell-*"


def add_grid_summaries(f: pd.DataFrame) -> list:
    made = []
    for name, (pat, how) in GRID_SUMMARIES.items():
        grid = [c for c in f.columns if re.match(pat, c)]
        if grid:
            f[name] = f[grid].mean(axis=1) if how == "mean" else f[grid].std(axis=1)
            made.append(name)
    return made


def texture_columns(f: pd.DataFrame, mode: str) -> tuple:
    cells = {}
    for st in TEXTURE_CELL_STATS:
        cols = [c for c in f.columns if re.match(r'^tex_r\d+c\d+_' + re.escape(st) + r'$', c)]
        if cols:
            cells[st] = sorted(cols)
    made = []
    if mode == "grid":
        for st, cols in cells.items():
            made += cols
        return made, cells
    for st, cols in cells.items():
        f[f"texgrid_{st}_mean"] = f[cols].mean(axis=1)
        f[f"texgrid_{st}_sd"] = f[cols].std(axis=1)
        made += [f"texgrid_{st}_mean", f"texgrid_{st}_sd"]
    return made, cells


def to_specimen_table(f: pd.DataFrame, cols: list, idmap: dict, aliases: dict) -> pd.DataFrame:
    """image x category rows -> specimen x '<matrix category>.<column>'."""
    rows = []
    for r in f[["image_base", "category"] + cols].itertuples(index=False):
        img, cat = r[0], r[1]
        ckey = osh.norm_category(cat)
        ckey = osh.norm_category(aliases[ckey]) if ckey in aliases else ckey
        for sid, _species, matrix_cat in idmap.get((osh.norm_image(img), ckey), []):
            rows.append((sid, matrix_cat) + tuple(r[2:]))
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows, columns=["specimen_id", "category"] + cols)
    g = d.groupby(["specimen_id", "category"])[cols].mean()
    wide = g.unstack("category")
    wide.columns = [f"{cat}.{col}" for col, cat in wide.columns]
    return wide.dropna(axis=1, how="all").sort_index(axis=1)


def main():
    ap = argparse.ArgumentParser(description="Colour-pattern and texture sets for the harness")
    ap.add_argument("--compiled_dir", required=True,
                    help="the compiled dir that still carries color_* / colhom_* / tex_* columns")
    ap.add_argument("--identity_dir", required=True,
                    help="the compiled dir the matrix was built from (*_full_features.csv with "
                         "specimen_id); its aliases, sex split and exclusions are inherited")
    ap.add_argument("--taxon_profile", default=None)
    ap.add_argument("--reorient_summary", default=None,
                    help="annotation_reorient/reorientation_summary.json — the images whose "
                         "annotations were turned AFTER the colour and texture were measured")
    ap.add_argument("--reorient_categories", default=DEFAULT_REORIENT_CATEGORIES,
                    help="comma-separated glob patterns of the structures on those images")
    ap.add_argument("--texture_mode", choices=["summary", "grid"], default="summary")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    idmap, aliases = osh.load_identity(Path(a.identity_dir), profile)

    src = sorted(Path(a.compiled_dir).glob("*_full_features.csv"))
    if not src:
        sys.exit(f"no *_full_features.csv in {a.compiled_dir}")
    f = pd.read_csv(src[0], low_memory=False)
    print(f"{src[0].name}: {f.shape[0]} image x structure rows, {f.shape[1]} columns")

    grid_made = add_grid_summaries(f)
    pats = [re.compile(p) for p in COMPUTED_GROUPS["colour pattern"]]
    colour_cols = [c for c in f.columns if any(p.match(c) for p in pats)] + grid_made
    tex_cols, tex_cells = texture_columns(f, a.texture_mode)
    print(f"colour pattern: {len(colour_cols)} columns per structure "
          f"({len(colour_cols) - len(grid_made)} pattern metrics + {len(grid_made)} grid summaries)")
    print(f"texture ({a.texture_mode}): {len(tex_cols)} columns per structure from "
          f"{sum(len(v) for v in tex_cells.values())} grid cells over "
          f"{len(tex_cells)} statistics")

    # ── the reoriented images ───────────────────────────────────────────────
    blanked = {"images": [], "rows": 0, "columns": len(colour_cols) + len(tex_cols)}
    if a.reorient_summary and Path(a.reorient_summary).exists():
        ro = json.loads(Path(a.reorient_summary).read_text()).get("images_reoriented", {})
        keys = {osh.norm_image(k) for k in ro}
        globs = [g.strip() for g in a.reorient_categories.split(",") if g.strip()]
        hit = f["image_base"].map(osh.norm_image).isin(keys) & \
            f["category"].map(lambda c: any(fnmatch.fnmatch(str(c), g) for g in globs))
        f.loc[hit, colour_cols + tex_cols] = np.nan
        blanked = {"images": sorted(ro), "rule": ro, "categories": globs,
                   "rows": int(hit.sum()), "columns": len(colour_cols) + len(tex_cols),
                   "reason": "the annotations were turned after the colour and texture were "
                             "measured, so those values were read off the wrong part of the slide"}
        print(f"reoriented images: {len(ro)}; {int(hit.sum())} image x structure rows set to "
              f"missing for colour pattern and texture")

    report = {"version": VERSION, "compiled": str(src[0]), "identity_dir": str(a.identity_dir),
              "texture_mode": a.texture_mode, "reoriented": blanked,
              "colour_pattern_columns": colour_cols,
              "texture_columns_per_structure": tex_cols,
              "texture_grid_cells_per_statistic": {k: len(v) for k, v in tex_cells.items()},
              "excluded": {}}
    for pat, why in TEXTURE_EXCLUDED.items():
        hits = [c for c in f.columns if re.match(pat, c)]
        if hits:
            report["excluded"][pat] = {"columns": len(hits), "why": why,
                                       "examples": hits[:4]}
    for name, cols in (("colour_pattern", colour_cols), ("texture", tex_cols)):
        t = to_specimen_table(f, cols, idmap, aliases)
        if t.empty:
            print(f"{name}: nothing matched the matrix")
            continue
        t.index.name = "specimen_id"
        t.to_csv(out / f"{name}.tsv", sep="\t")
        report[name] = {"specimens": int(len(t)), "features": int(t.shape[1]),
                        "structures": len(sorted({c.rsplit('.', 1)[0] for c in t.columns})),
                        "filled": round(float(t.notna().mean().mean()), 3)}
        print(f"{name}: {t.shape[0]} specimens x {t.shape[1]} features "
              f"({report[name]['structures']} structures, "
              f"{100 * report[name]['filled']:.0f}% filled)")
    (out / "computed_sets_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
