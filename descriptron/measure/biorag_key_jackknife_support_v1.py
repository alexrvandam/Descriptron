#!/usr/bin/env python3
"""
biorag_key_jackknife_support_v1.py — a support value for every couplet of the key
==================================================================================

A phylogeny carries a support value on every node; a computed key can carry one on every couplet,
obtained the same way. The key is rebuilt with one specimen deleted, once for each specimen (a
delete-one JACKKNIFE — not a bootstrap, which would resample specimens with replacement), and two
things are read off the rebuilt keys for every couplet of the published key:

  recovered   the share of jackknife keys that contain the couplet — the same two groups of species
              set apart — and, of those, the share that set them apart on the same leading character.
              A couplet that is rebuilt whichever specimen is left out rests on the species; one that
              comes and goes rests on a specimen.
  routed      of the withheld specimens that passed through that couplet in their own jackknife key,
              the share sent to the side holding their own species, with a 95% Wilson interval. This is
              the number a user of the key needs: how often a specimen the key has never seen leaves
              this couplet by the right door. A specimen sent the wrong way higher up never reaches the
              couplets below, so n falls down the key.

  python biorag_key_jackknife_support_v1.py --matrix_dir <compiled_key_tier> --taxon_profile <yaml> \
      --key_tree <key/key_tree.json> --key_loo_dir <key_fuzzy_loo/keys_loo> --out_dir <dir>

`--key_loo_dir` holds one rebuilt key per specimen (`<specimen_id>.json`), as written by
`biorag_key_fuzzy_v1.py --known_arm leave_one_specimen_out`. Writes key_couplet_support.tsv and
key_couplet_support_summary.json; `biorag_calibration_figures_v1.py` prints the values on the key.
No model is called.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                  # noqa: E402
from biorag_novelty_score_v1 import Reference                        # noqa: E402
from biorag_graph_support_figure_v1 import walk_key                  # noqa: E402

VERSION = "1.0"


def wilson(k, n, z=1.96):
    if not n:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def split_of(c):
    """The couplet as an unordered pair of species sets."""
    return frozenset((frozenset(c.get("A_species", [])), frozenset(c.get("B_species", []))))


def lead(c):
    ch = (c.get("characters") or [{}])[0]
    return ch.get("feature_id"), ch.get("label", "")


def main():
    ap = argparse.ArgumentParser(description="Delete-one jackknife support for every couplet of a computed key")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--key_tree", required=True)
    ap.add_argument("--key_loo_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref = Reference(Path(a.matrix_dir), pol.load_taxon_profile(a.taxon_profile))
    pub = json.loads(Path(a.key_tree).read_text())["couplets"]
    pub_split = {c["number"]: split_of(c) for c in pub}
    by_split = {v: k for k, v in pub_split.items()}

    keys, recovered, same_char = 0, Counter(), Counter()
    routed, right = Counter(), Counter()
    for sid in ref.raw.index:
        p = Path(a.key_loo_dir) / f"{sid}.json"
        if not p.exists():
            continue
        recs = json.loads(p.read_text())["couplets"]
        keys += 1
        here = {split_of(r): r for r in recs}
        for num, sp in pub_split.items():
            r = here.get(sp)
            if r is not None:
                recovered[num] += 1
                same_char[num] += int(lead(r)[0] == lead(next(c for c in pub if c["number"] == num))[0])
        row = ref.raw.loc[sid]
        values = {k: float(v) for k, v in row.items() if v == v}
        truth = ref.species_of.get(sid)
        _ans, steps, _ = walk_key(values, ref.sex_of.get(sid), recs)
        for r, _c, side in steps:
            num = by_split.get(split_of(r))
            if num is None:
                continue                                  # a couplet the published key does not have
            routed[num] += 1
            right[num] += int(truth in r.get(f"{side}_species", []))

    rows = []
    for c in pub:
        n = c["number"]
        lo, hi = wilson(right[n], routed[n])
        rows.append({"couplet": n, "leading_character": lead(c)[1], "feature_id": lead(c)[0],
                     "species_A": len(c.get("A_species", [])), "species_B": len(c.get("B_species", [])),
                     "jackknife_keys": keys,
                     "recovered": round(recovered[n] / keys, 4) if keys else None,
                     "recovered_on_same_character": round(same_char[n] / recovered[n], 4) if recovered[n] else None,
                     "withheld_specimens_routed": routed[n], "routed_to_own_species": right[n],
                     "routing_support": round(right[n] / routed[n], 4) if routed[n] else None,
                     "routing_ci_low": None if lo != lo else round(lo, 4),
                     "routing_ci_high": None if hi != hi else round(hi, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "key_couplet_support.tsv", sep="\t", index=False)
    rs = df["routing_support"].dropna()
    summary = {"version": VERSION, "couplets": int(len(df)), "jackknife_keys": keys,
               "couplets_recovered_in_every_key": int((df["recovered"] == 1).sum()),
               "median_recovered": float(df["recovered"].median()),
               "least_recovered": df.sort_values("recovered").head(3)[["couplet", "leading_character", "recovered"]].to_dict("records"),
               "median_routing_support": float(rs.median()) if len(rs) else None,
               "couplets_routing_below_0.75": int((rs < 0.75).sum()),
               "weakest_routing": df.dropna(subset=["routing_support"]).query("withheld_specimens_routed >= 5")
                                    .sort_values("routing_support").head(3)[["couplet", "leading_character", "routing_support",
                                                                              "withheld_specimens_routed"]].to_dict("records"),
               "note": "delete-one-specimen jackknife: recovered = share of rebuilt keys with the same two groups of "
                       "species set apart; routing_support = share of withheld specimens passing through the couplet "
                       "that left it on the side of their own species"}
    (out / "key_couplet_support_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(df[["couplet", "leading_character", "recovered", "recovered_on_same_character",
              "withheld_specimens_routed", "routing_support"]].to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("note",)}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
