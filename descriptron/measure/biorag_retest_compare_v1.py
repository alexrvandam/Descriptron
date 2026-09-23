#!/usr/bin/env python3
"""
biorag_retest_compare_v1.py — the same images read more than once: agreement, kappa, and drift
===============================================================================================

`biorag_descriptive_scoring_v1.py` asks a model for one state per descriptive categorical
character on each specimen x structure image. Whether those states can be relied on is an
empirical question with three parts that a single percentage hides:

  exact agreement   how often a second reading returns the same state
  Cohen's kappa     the same, with the agreement expected by chance removed: a character on
                    which nine structures in ten are "glabrous" agrees with itself 82% of the
                    time by luck. For ordinal characters a linearly weighted kappa is given as
                    well, which counts a one-step disagreement (shining / subopaque) as a
                    smaller error than a three-step one
  direction         whether the disagreements run both ways (noise) or mostly one way (the
                    scorer has shifted between the two readings). Two readings taken in the
                    same session and two taken days apart are different tests, and only a
                    comparison of both can tell noise from drift

Any number of readings can be given; every pair is compared on the cells they share, where a
cell is one (specimen, structure, character, image) and both readings returned a state.

  python biorag_retest_compare_v1.py --readings first=<tsv> A=<tsv> B=<tsv> --out_dir <dir>

Each TSV is a `descriptive_states_by_specimen.tsv` (columns specimen_id, category, character,
image, state, type). No model is called.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biorag_novelty_score_v1 import DESCRIPTIVE_CHARACTERS, state_ranks      # noqa: E402

VERSION = "1.1"      # 1.1: --exclude_images; agreement of the SPECIES-level state (what a treatment prints)
KEY = ["specimen_id", "category", "character", "image"]
BLANK = {"", "not assessable", "nan", "none"}


def load(path: Path, exclude_images=()) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    d["state"] = d["state"].astype(str).str.strip().str.lower()
    d = d[~d["state"].isin(BLANK)]
    if exclude_images:
        d = d[~d["image"].astype(str).map(lambda p: any(x in p for x in exclude_images))]
    keep = KEY + ["state"] + (["species"] if "species" in d.columns else [])
    return d.drop_duplicates(KEY)[keep]


def species_state(d: pd.DataFrame, ranks_for) -> pd.DataFrame:
    """The state a treatment would print for a species x structure x character: the median state of
    an ordinal character over that species' specimens (lower median, so it is always a state that
    was scored), the modal state of a nominal one (ties broken alphabetically)."""
    rows = []
    for (sp, cat, ch), g in d.groupby(["species", "category", "character"]):
        ranks = ranks_for(ch)
        if ranks:
            r = g["state"].map(ranks).dropna().sort_values()
            if not len(r):
                continue
            inv = {v: k for k, v in ranks.items()}
            st = inv[r.iloc[(len(r) - 1) // 2]]
        else:
            st = sorted(g["state"].mode())[0]
        rows.append({"species": sp, "category": cat, "character": ch, "state": st, "n": len(g)})
    return pd.DataFrame(rows)


def kappa(a: pd.Series, b: pd.Series, ranks=None):
    """Cohen's kappa; with `ranks` (state -> position on the scale) a linearly weighted kappa."""
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return None
    ia = pd.Categorical(a, categories=cats).codes
    ib = pd.Categorical(b, categories=cats).codes
    n = len(cats)
    O = np.zeros((n, n))
    for x, y in zip(ia, ib):
        O[x, y] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    if ranks is None:
        W = 1.0 - np.eye(n)
    else:
        r = np.array([ranks.get(c, np.nan) for c in cats], float)
        if np.isnan(r).any() or np.nanmax(r) == np.nanmin(r):
            return None
        W = np.abs(r[:, None] - r[None, :]) / (np.nanmax(r) - np.nanmin(r))
    den = (W * E).sum()
    return None if den <= 0 else float(1.0 - (W * O).sum() / den)


def direction(a: pd.Series, b: pd.Series, ranks):
    """Among disagreements on an ordinal scale: the share in which the SECOND reading is higher.
    0.5 is noise; near 0 or 1 is a shift of the scorer."""
    if not ranks:
        return None, 0
    ra, rb = a.map(ranks), b.map(ranks)
    ok = ra.notna() & rb.notna() & (ra != rb)
    n = int(ok.sum())
    return (float((rb[ok] > ra[ok]).mean()) if n else None), n


def main():
    ap = argparse.ArgumentParser(description="Agreement, kappa and drift between readings of the same images")
    ap.add_argument("--readings", nargs="+", required=True, help="name=path, two or more")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude_images", nargs="*", default=[],
                    help="substrings of image paths to leave out of every reading (e.g. photographs whose "
                         "annotations were found to be misplaced, so the scorer was shown blank mount)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    R = {}
    for spec in a.readings:
        name, _, path = spec.partition("=")
        R[name] = load(Path(path), a.exclude_images)
    # a retest file may not carry the species column: take it from any reading that does
    sp_of = {}
    for d in R.values():
        if "species" in d.columns:
            sp_of.update(dict(zip(d["specimen_id"], d["species"])))
    for name in R:
        if "species" not in R[name].columns and sp_of:
            R[name] = R[name].assign(species=R[name]["specimen_id"].map(sp_of))
        print(f"{name}: {len(R[name])} cells with a state")

    rows, pooled = [], []
    for n1, n2 in itertools.combinations(R, 2):
        m = R[n1].merge(R[n2], on=KEY, suffixes=("_1", "_2"))
        if not len(m):
            continue
        pooled.append({"pair": f"{n1} ~ {n2}", "cells": int(len(m)),
                       "specimens": int(m["specimen_id"].nunique()),
                       "exact_agreement": round(float((m.state_1 == m.state_2).mean()), 4)})
        for ch, g in m.groupby("character"):
            cfg = DESCRIPTIVE_CHARACTERS.get(ch, {})
            ranks = state_ranks(cfg) if cfg.get("type") == "ordinal" else None
            up, n_dis = direction(g.state_1, g.state_2, ranks)
            within = None
            if ranks:
                r1, r2 = g.state_1.map(ranks), g.state_2.map(ranks)
                okr = r1.notna() & r2.notna()
                within = float(((r1[okr] - r2[okr]).abs() <= 1).mean()) if okr.any() else None
            k, kw = kappa(g.state_1, g.state_2), kappa(g.state_1, g.state_2, ranks) if ranks else None
            rows.append({"pair": f"{n1} ~ {n2}", "character": ch, "type": cfg.get("type", ""),
                         "cells": int(len(g)),
                         "exact_agreement": round(float((g.state_1 == g.state_2).mean()), 4),
                         "within_one_step": None if within is None else round(within, 4),
                         "cohen_kappa": None if k is None else round(k, 4),
                         "weighted_kappa": None if kw is None else round(kw, 4),
                         "disagreements": n_dis,
                         "share_second_reading_higher": None if up is None else round(up, 3)})
    per = pd.DataFrame(rows)
    per.to_csv(out / "retest_agreement_by_character.tsv", sep="\t", index=False)
    pool = pd.DataFrame(pooled)
    for i, p in pool.iterrows():
        g = per[per.pair == p["pair"]]
        w = g["cells"] / g["cells"].sum()
        pool.loc[i, "mean_cohen_kappa"] = round(float((g["cohen_kappa"].fillna(0) * w).sum()), 4)
    pool.to_csv(out / "retest_agreement_pooled.tsv", sep="\t", index=False)
    # the same comparison one level up: the state a TREATMENT prints is a species-level summary, and
    # per-image noise averages out of it while a shift of the scorer does not
    def ranks_for(ch):
        cfg = DESCRIPTIVE_CHARACTERS.get(ch, {})
        return state_ranks(cfg) if cfg.get("type") == "ordinal" else None
    sp_rows = []
    for n1, n2 in itertools.combinations(R, 2):
        if "species" not in R[n1].columns or "species" not in R[n2].columns:
            continue
        shared = R[n1].merge(R[n2][KEY], on=KEY)                      # summarise each reading over the SAME cells
        s1 = species_state(shared, ranks_for)
        s2 = species_state(R[n2].merge(R[n1][KEY], on=KEY), ranks_for)
        m = s1.merge(s2, on=["species", "category", "character"], suffixes=("_1", "_2"))
        for ch, g in list(m.groupby("character")) + [("ALL CHARACTERS", m)]:
            ranks = ranks_for(ch) if ch != "ALL CHARACTERS" else None
            within = None
            if ranks:
                r1, r2 = g.state_1.map(ranks), g.state_2.map(ranks)
                within = float(((r1 - r2).abs() <= 1).mean())
            k = kappa(g.state_1, g.state_2) if ch != "ALL CHARACTERS" else None
            sp_rows.append({"pair": f"{n1} ~ {n2}", "character": ch, "species_x_structure_cells": int(len(g)),
                            "same_state_printed": round(float((g.state_1 == g.state_2).mean()), 4),
                            "within_one_step": None if within is None else round(within, 4),
                            "cohen_kappa": None if k is None else round(k, 4)})
    if sp_rows:
        pd.DataFrame(sp_rows).to_csv(out / "retest_agreement_species_level.tsv", sep="\t", index=False)
        print("\nspecies-level state (what a treatment prints), all characters:")
        print(pd.DataFrame(sp_rows).query("character == 'ALL CHARACTERS'").to_string(index=False))

    (out / "retest_compare_summary.json").write_text(json.dumps(
        {"version": VERSION, "readings": {k: int(len(v)) for k, v in R.items()},
         "pooled": pool.to_dict("records"),
         "note": "cells = (specimen, structure, character, image) on which both readings returned a "
                 "state; share_second_reading_higher near 0 or 1 means the scorer shifted between the "
                 "two readings, near 0.5 means its disagreements are noise"}, indent=2))
    pd.set_option("display.width", 220)
    print("\n" + pool.to_string(index=False))
    wide = per.pivot(index="character", columns="pair", values="exact_agreement")
    print("\nexact agreement by character:\n" + wide.round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
