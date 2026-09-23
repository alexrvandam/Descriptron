#!/usr/bin/env python3
"""
biorag_character_reliability_v1.py — which descriptive characters can be trusted
===============================================================================

A vision model can put a word on any structure; that does not mean the word is
reproducible, and reproducibility differs between taxa. This script measures it
and says, per character, whether to use it, use it only at the coarse level, or
flag it as unreliable for this taxon:

  1. REPEATABILITY — the same images scored twice (biorag_descriptive_scoring_v1.py
     run a second time into another --out_dir): the share of states repeated
     exactly, the share within one step of the scale, and the share repeated
     after the states are merged into coarse bands (smooth / finely sculptured /
     coarsely sculptured, uniform / two-toned / patterned ...). Coarse bands
     usually recover the characters that only fail on fine distinctions.
  2. CONGRUENCE — does the model's state track a metric the pipeline computes
     from the same image? Elongation against length/width, margin incision
     against solidity, colour heterogeneity against the colour-pattern metrics,
     surface roughness against the texture components. Ordinal characters are
     compared with Spearman's rho, nominal characters with eta-squared.
     Characters with no computed counterpart are marked as such: for those the
     model is the only source and its repeatability is all the evidence there is.

Verdict per character (thresholds are options, not constants):
  use             repeatable at the fine scale (>= --min_fine)
  use coarse      repeatable only once the states are merged (>= --min_coarse)
  flag            below both: keep the word in the text if you like, but do not
                  score with it, and say so in the paper

Usage:
  python biorag_character_reliability_v1.py \\
     --scoring   "$M/descriptive_states/descriptive_states_by_specimen.tsv" \\
     --rescoring "$M/descriptive_states_retest/descriptive_states_by_specimen.tsv" \\
     --computed_dir "/media/.../Diaphorina_compiled_29species" \\
     --taxon_profile <profile.yaml> --out_dir "$M/descriptive_states"
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import biorag_feature_policy as pol  # noqa: E402
from biorag_novelty_score_v1 import (DESCRIPTIVE_CHARACTERS, canonical_states,  # noqa: E402
                                     coarse_map, coarse_ranks, computed_table, state_ranks)

VERSION = "1.0"
NA = "not assessable"

# Which computed metrics a character should track, if any. The metric is matched
# within the same structure; the best |correlation| is reported.
COUNTERPARTS = {
    "elongation": [r'meas_aspect_ratio$', r'shape_PC[1-3]$'],
    "curvature": [r'meas_solidity$', r'meas_extent$', r'shape_PC[1-3]$'],
    "margin incision": [r'meas_solidity$', r'meas_extent$'],
    "apex": [r'shape_PC[1-5]$', r'meas_solidity$'],
    "outline shape": [r'shape_PC[1-5]$', r'meas_aspect_ratio$'],
    "surface roughness": [r'tex_phylo_PC[1-5]$', r'grid_entropy_(mean|sd)$'],
    "surface pattern": [r'tex_phylo_PC[1-5]$'],
    "colour heterogeneity": [r'grid_lightness_sd$', r'grid_entropy_(mean|sd)$', r'grid_[ab]_sd$',
                             r'color_(adaptive|median)_(n_markings|pattern_complexity|boundary_strength|'
                             r'n_boundaries_diff)$'],
    "lustre": [r'color_(adaptive|median)_bri_(mean|std)$', r'grid_lightness_sd$'],
    "transparency": [r'color_(adaptive|median)_(bri_mean|sat_mean)$'],
    "vestiture": [],                      # nothing in the pipeline measures setae
}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 6:
        return math.nan
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) < 1e-9 or np.std(ry) < 1e-9:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def eta_squared(groups: List[np.ndarray]) -> float:
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return math.nan
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((allv - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else math.nan


def repeatability(a: pd.DataFrame, b: pd.DataFrame, characters: Dict) -> Dict[str, Dict]:
    key = ["specimen_id", "category", "character"]
    m = a.merge(b, on=key, suffixes=("_1", "_2"))
    m = m[(m["state_1"] != NA) & (m["state_2"] != NA)]
    out = {}
    for ch, g in m.groupby("character"):
        cfg = characters.get(ch)
        if cfg is None:
            continue
        ranks = state_ranks(cfg)
        cmap, cranks = coarse_map(ch), coarse_ranks(ch)
        s1 = g["state_1"].str.lower()
        s2 = g["state_2"].str.lower()
        exact = float((s1.values == s2.values).mean())
        if cfg["type"] == "ordinal":
            steps = np.array([abs(ranks.get(x, 0) - ranks.get(y, 0)) for x, y in zip(s1, s2)])
            within1 = float((steps <= 1).mean())
        else:
            steps, within1 = np.array([]), exact
        if cmap:
            c1 = s1.map(cmap)
            c2 = s2.map(cmap)
            ok = c1.notna() & c2.notna()
            coarse = float((c1[ok].values == c2[ok].values).mean()) if ok.any() else math.nan
            n_bands = len(set(cmap.values()))
        else:
            coarse, n_bands = math.nan, 0
        out[ch] = {"n": int(len(g)), "type": cfg["type"], "states": len(canonical_states(cfg)),
                   "exact": round(exact, 3), "within_one_step": round(within1, 3),
                   "coarse_bands": n_bands,
                   "coarse_exact": (round(coarse, 3) if coarse == coarse else None),
                   "mean_steps": (round(float(steps.mean()), 2) if len(steps) else None)}
    return out


def congruence(scores: pd.DataFrame, computed: pd.DataFrame, characters: Dict,
               min_specimens=8) -> Dict[str, Dict]:
    """Does the model's state track something the pipeline measures?"""
    out = {}
    for ch, pats in COUNTERPARTS.items():
        cfg = characters.get(ch)
        if cfg is None:
            continue
        if not pats:
            out[ch] = {"counterpart": None, "note": "nothing in the pipeline measures this character"}
            continue
        ranks = state_ranks(cfg)
        rx = [re.compile(p) for p in pats]
        best = {"metric": None, "value": math.nan, "structures": 0, "n": 0}
        per_metric = defaultdict(list)
        sub = scores[(scores["character"] == ch) & (scores["state"] != NA)]
        for cat, g in sub.groupby("category"):
            cols = [c for c in computed.columns
                    if c.startswith(f"{cat}.") and any(p.search(c.split('.', 1)[1]) for p in rx)]
            if not cols:
                continue
            g = g.drop_duplicates("specimen_id").set_index("specimen_id")
            ids = [i for i in g.index if i in computed.index]
            if len(ids) < min_specimens:
                continue
            states = g.loc[ids, "state"].str.lower()
            for col in cols:
                y = computed.loc[ids, col].astype(float).values
                keep = ~np.isnan(y)
                if keep.sum() < min_specimens:
                    continue
                if cfg["type"] == "ordinal":
                    x = np.array([ranks.get(s, math.nan) for s in states])[keep]
                    if np.isnan(x).any():
                        continue
                    r = spearman(x, y[keep])
                    val = abs(r) if r == r else math.nan
                else:
                    groups = [y[keep][(states.values[keep] == st)] for st in states.unique()]
                    val = eta_squared(groups)
                if val == val:
                    per_metric[col.split(".", 1)[1]].append((val, int(keep.sum())))
        for metric, vals in per_metric.items():
            med = float(np.median([v for v, _ in vals]))
            if not (best["value"] == best["value"]) or med > best["value"]:
                best = {"metric": metric, "value": round(med, 3), "structures": len(vals),
                        "n": int(np.median([n for _, n in vals])),
                        # the median is across structures and hides a character that tracks the metric
                        # strongly on a few structures and not at all on the rest, so report both
                        "value_max": round(float(max(v for v, _ in vals)), 3),
                        "structures_strong": int(sum(1 for v, _ in vals if v >= 0.5))}
        out[ch] = {"counterpart": best["metric"],
                   "statistic": "|Spearman rho|" if cfg["type"] == "ordinal" else "eta squared",
                   "value": best["value"], "structures_compared": best["structures"],
                   "median_specimens": best["n"],
                   "value_max": best.get("value_max"),
                   "structures_strong": best.get("structures_strong")}
    return out


def main():
    ap = argparse.ArgumentParser(description="Repeatability and congruence of the descriptive characters")
    ap.add_argument("--scoring", required=True, help="descriptive_states_by_specimen.tsv (first pass)")
    ap.add_argument("--rescoring", default=None, help="the same images scored again (a sample is enough)")
    ap.add_argument("--computed_dir", default=None, help="compiled feature dir, for the congruence test")
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_fine", type=float, default=0.85,
                    help="repeatability at the fine scale to use a character as scored (default 0.85)")
    ap.add_argument("--min_coarse", type=float, default=0.80,
                    help="repeatability once the states are merged into coarse bands (default 0.80)")
    a = ap.parse_args()

    profile = pol.load_taxon_profile(a.taxon_profile)
    characters = {k: dict(v) for k, v in DESCRIPTIVE_CHARACTERS.items()}
    characters.update(profile.get("descriptive_characters") or {})
    scores = pd.read_csv(a.scoring, sep="\t")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rep = repeatability(scores, pd.read_csv(a.rescoring, sep="\t"), characters) if a.rescoring else {}
    comp = {}
    if a.computed_dir:
        ct = computed_table(Path(a.computed_dir), profile)
        comp = congruence(scores, ct, characters)

    rows = []
    for ch in characters:
        r = rep.get(ch, {})
        c = comp.get(ch, {})
        fine = r.get("exact", math.nan)
        coarse = r.get("coarse_exact")
        coarse = coarse if coarse is not None else math.nan
        if fine == fine and fine >= a.min_fine:
            verdict, why = "use", f"repeats {100 * fine:.0f}% of states exactly"
        elif coarse == coarse and coarse >= a.min_coarse:
            verdict, why = "use coarse", (f"only {100 * fine:.0f}% exact, but {100 * coarse:.0f}% once the "
                                          f"states are merged into {r.get('coarse_bands')} bands")
        elif fine == fine:
            verdict, why = "flag", (f"{100 * fine:.0f}% exact"
                                    + (f", {100 * coarse:.0f}% coarse" if coarse == coarse else "")
                                    + " — not reproducible enough to score with")
        else:
            verdict, why = "not measured", "no second scoring run supplied"
        if verdict == "flag" and c.get("counterpart") and (c.get("value") or 0) >= 0.5:
            why += f"; use the measured {c['counterpart']} instead ({c['statistic']} {c['value']})"
        rows.append({"character": ch, "type": characters[ch]["type"],
                     "states": len(canonical_states(characters[ch])),
                     "coarse_bands": r.get("coarse_bands", 0), "n_compared": r.get("n"),
                     "repeat_exact": fine, "repeat_within_one_step": r.get("within_one_step"),
                     "repeat_coarse": coarse,
                     "computed_counterpart": c.get("counterpart"),
                     "congruence_statistic": c.get("statistic"), "congruence": c.get("value"),
                     "congruence_best_structure": c.get("value_max"),
                     "structures_congruent": c.get("structures_strong"),
                     "structures_compared": c.get("structures_compared"),
                     "verdict": verdict, "reason": why})
    df = pd.DataFrame(rows).sort_values(["verdict", "repeat_exact"], ascending=[True, False])
    df.to_csv(out / "character_reliability.tsv", sep="\t", index=False)
    report = {"version": VERSION, "generated": datetime.now().isoformat(),
              "min_fine": a.min_fine, "min_coarse": a.min_coarse,
              "scoring": str(a.scoring), "rescoring": str(a.rescoring),
              "characters": {r["character"]: r for r in rows},
              "use": [r["character"] for r in rows if r["verdict"] == "use"],
              "use_coarse": [r["character"] for r in rows if r["verdict"] == "use coarse"],
              "flag": [r["character"] for r in rows if r["verdict"] == "flag"]}
    (out / "character_reliability.json").write_text(json.dumps(report, indent=2))
    pd.set_option("display.width", 200)
    print(df[["character", "n_compared", "repeat_exact", "repeat_within_one_step", "repeat_coarse",
              "computed_counterpart", "congruence", "congruence_best_structure",
              "structures_congruent", "structures_compared", "verdict"]].to_string(index=False))
    print(f"\nuse: {report['use']}\nuse coarse: {report['use_coarse']}\nflag: {report['flag']}")
    print(f"-> {out / 'character_reliability.tsv'}")


if __name__ == "__main__":
    main()
