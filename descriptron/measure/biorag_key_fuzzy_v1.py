#!/usr/bin/env python3
"""
biorag_key_fuzzy_v1.py — a graded key: how well does a specimen fit the path it took?
====================================================================================

A dichotomous couplet is a hard threshold, so a specimen a hair either side of it is
sent one way with no record that the call was close, and a specimen belonging to no
described species is routed somewhere regardless. This scores the same key in two
graded ways, without changing the tree:

  decisiveness   how clearly the value chose its side of the couplet. 1 = far from
                 the threshold, 0 = a coin flip. Derived from the two observed
                 ranges, so it needs no distributional assumption — which matters
                 with two to seven specimens per species.
  fit            how compatible the value is with the range ever observed on the
                 side it was sent down. 1 = inside, falling to 0 one range-width
                 outside. This is the existing conflict flag with its step edges
                 softened.

Both are carried down the path with a t-norm (minimum by default, so the weakest
couplet governs, as a taxonomist would read it). A missing character no longer ends
the path: the couplet's supporting characters are tried first.

The point of the exercise is the 30% of specimens that, in the hold-out test, reached
an existing species with nothing to warn the user. A graded score should give those a
low value.

The delimitation test (`biorag_novelty_score_v1.py --holdout_all`) scores a
held-out species against the character matrices. It does **not** test the key:
there the key is built from all species and the path is only a record of where
the specimen runs. This script runs the other experiment, the one a taxonomist
actually performs.

For each described species in turn:

  1. the key is rebuilt from the OTHER species only, deterministically
     (no model wording is needed for the tree, so this costs nothing);
  2. every specimen of the held-out species is run down that key;
  3. what happens to it is recorded.

Three outcomes, and only one of them is dangerous:

  no character      the path stops because a couplet needs a character that was
                    not measured on this specimen. This is a coverage failure,
                    NOT evidence of novelty: the same happens to correctly
                    identified specimens, and the baseline rate is reported
                    alongside it (--baseline).
  key exhausted     the path stops although every character was available. That
                    is a property of the specimen, not of the data.
  keyed, conflict   the specimen reaches a species, but at one or more couplets
                    its value lay outside the range ever observed on the side it
                    was sent down. A careful user, with the ranges printed in the
                    couplet, has a reason to doubt the answer.
  keyed, clean      the specimen reaches an existing species with every value
                    inside the observed ranges. The key gives a confident wrong
                    answer and there is nothing in it to warn the user.

  python biorag_key_holdout_v1.py --matrix_dir "$M/compiled_key_tier" \\
      --taxon_profile <profile.yaml> --out_dir "$M/key_holdout" [--species sp1 sp2]
"""

import argparse
import json
import math
import subprocess
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                   # noqa: E402
from biorag_novelty_score_v1 import (Reference, features_from_coco,   # noqa: E402
                                     key_path)

VERSION = "1.0-fuzzy"
TNORMS = {"min": min, "product": lambda a, b: a * b}
BUILDER = Path(__file__).resolve().parent / "biorag_key_builder_v1.py"


def build_key(matrix_dir: Path, species: list, out_dir: Path, python: str, profile: str,
              exclude_specimens=None) -> dict:
    """Rebuild the key from a species list. The tree is computed from the matrix, so no model
    is needed and the result is deterministic."""
    cmd = [python, str(BUILDER), "--matrix_dir", str(matrix_dir), "--output_dir", str(out_dir),
           "--species", *species, "--no-loo", "--llm-backend", "none",
           "--taxon-profile", profile]
    if exclude_specimens:
        cmd += ["--exclude_specimens", *exclude_specimens]
    r = subprocess.run(cmd, capture_output=True, text=True)
    tree = out_dir / "key_tree.json"
    if not tree.exists():
        return {"error": (r.stderr or r.stdout or "")[-400:]}
    return json.loads(tree.read_text())


def _edges(a_range, b_range, a_is_low):
    """The transition zone between the two sides, whether the ranges are separated by a
    gap or overlap. Returns (low_edge, high_edge) with low_edge <= high_edge."""
    lo_side, hi_side = (a_range, b_range) if a_is_low else (b_range, a_range)
    top_of_low, bottom_of_high = max(lo_side), min(hi_side)
    return (min(top_of_low, bottom_of_high), max(top_of_low, bottom_of_high))


def membership(val, ch):
    """(decisiveness, side) for one character. decisiveness is |mu_low - mu_high|, so a
    value deep inside one side scores 1 and a value at the crossover scores 0."""
    a_low = str(ch.get("A_operator", "<=")).startswith("<")
    lo_e, hi_e = _edges(ch["A_range"], ch["B_range"], a_low)
    if hi_e <= lo_e:                       # degenerate: fall back to the printed threshold
        mu_low = 1.0 if val <= ch["threshold"] else 0.0
    elif val <= lo_e:
        mu_low = 1.0
    elif val >= hi_e:
        mu_low = 0.0
    else:
        mu_low = (hi_e - val) / (hi_e - lo_e)
    mu_high = 1.0 - mu_low
    side = ("A" if a_low else "B") if mu_low >= mu_high else ("B" if a_low else "A")
    return abs(mu_low - mu_high), side


def binary_membership(val, ch):
    """(decisiveness, side, fit) for a present/absent character.

    A state has no range to grade against, so the trapezoid above would return 1 for every
    specimen and the character could never be doubted. What can be graded is how well the
    state is established on each side: with the state seen in k of the n specimens behind a
    lead, the chance that a further specimen of that side shows it is (k + 1) / (n + 2)
    (Laplace). A state fixed in three specimens therefore counts for 0.8, not 1, and one
    fixed in thirty for 0.97 - the same small-sample caution the builder applies when it
    decides whether the character deserves a couplet at all (binary_quality)."""
    present = val > 0.5
    mu = {}
    for side in ("A", "B"):
        n = max(1, int(ch.get(f"{side}_n") or 1))
        lo, hi = min(ch[f"{side}_range"]), max(ch[f"{side}_range"])
        # share of that side showing 'present': exact when the side is fixed, else mid-range
        share = 1.0 if lo >= 1 else 0.0 if hi <= 0 else 0.5
        k = share * n
        p_present = (k + 1.0) / (n + 2.0)
        mu[side] = p_present if present else 1.0 - p_present
    side = "A" if mu["A"] >= mu["B"] else "B"
    return abs(mu["A"] - mu["B"]), side, mu[side]


def fit_to_range(val, rng):
    """1 inside the observed range, falling linearly to 0 one range-width outside."""
    lo, hi = min(rng), max(rng)
    if lo <= val <= hi:
        return 1.0
    w = hi - lo
    if w <= 0:
        return 0.0
    d = (lo - val) if val < lo else (val - hi)
    return max(0.0, 1.0 - d / w)


def fuzzy_key_path(key, cand, tnorm="min"):
    """Run a specimen down the key, grading every couplet instead of only branching.

    Falls back to a couplet's supporting characters when the primary one was not
    measured, so a missing character narrows the evidence rather than ending the path.
    """
    if not key:
        return {}
    T = TNORMS[tnorm]
    by_no = {c["number"]: c for c in key["couplets"]}
    node = key["couplets"][0]["number"]
    steps, seen = [], set()
    dec_path = fit_path = 1.0
    weakest = (1.0, None)
    used_support = missing = 0
    while node in by_no and node not in seen:
        seen.add(node)
        c = by_no[node]
        chars = c["characters"]
        ch = next((x for x in chars if cand.get(x["feature_id"], math.nan) ==
                   cand.get(x["feature_id"], math.nan)), None)
        if ch is None:
            steps.append(f"couplet {c['number']}: no character of this couplet was measured")
            missing += 1
            return {"terminal": None, "resolved": False, "steps": steps,
                    "decisiveness": dec_path, "fit": fit_path, "missing": missing,
                    "used_support": used_support, "weakest": weakest[1]}
        if ch is not chars[0]:
            used_support += 1
        val = float(cand[ch["feature_id"]])
        if ch.get("binary"):
            dec, side, f = binary_membership(val, ch)
        else:
            dec, side = membership(val, ch)
            f = fit_to_range(val, ch[f"{side}_range"])
        # the other characters of the couplet corroborate the fit where they exist
        others = [(binary_membership(float(cand[x["feature_id"]]), x)[2]
                   if x.get("binary") and binary_membership(float(cand[x["feature_id"]]), x)[1] == side
                   else 0.0 if x.get("binary")
                   else fit_to_range(float(cand[x["feature_id"]]), x[f"{side}_range"]))
                  for x in chars if x is not ch and
                  cand.get(x["feature_id"], math.nan) == cand.get(x["feature_id"], math.nan)
                  and f"{side}_range" in x]
        if others:
            f = (f + sum(others)) / (1 + len(others))
        dec_path, fit_path = T(dec_path, dec), T(fit_path, f)
        if min(dec, f) < weakest[0]:
            weakest = (min(dec, f), f"couplet {c['number']} ({ch['label']})")
        steps.append(f"couplet {c['number']}{side.lower()}: {ch['label']} = {val:.4g} "
                     f"(decisiveness {dec:.2f}, fit {f:.2f})")
        nxt = c[f"{side}_goto"]
        if nxt in by_no:
            node = nxt
        else:
            return {"terminal": nxt, "resolved": True, "steps": steps,
                    "decisiveness": dec_path, "fit": fit_path, "score": min(dec_path, fit_path),
                    "missing": missing, "used_support": used_support, "weakest": weakest[1]}
    return {"terminal": None, "resolved": False, "steps": steps, "decisiveness": dec_path,
            "fit": fit_path, "missing": missing, "used_support": used_support,
            "weakest": weakest[1]}


def main():
    ap = argparse.ArgumentParser(description="Grade the key: decisiveness and fit per specimen")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--key_tree", required=True, help="the published key, for the known specimens")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tnorm", choices=list(TNORMS), default="min")
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--known_arm", choices=["leave_one_specimen_out", "published"],
                    default="leave_one_specimen_out",
                    help="How the known specimens are graded. The default rebuilds the key without "
                         "each specimen before grading it; 'published' reproduces the earlier, "
                         "optimistic run in which a specimen was graded against ranges it had "
                         "helped define")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    ref = Reference(Path(a.matrix_dir), profile)
    published = json.loads(Path(a.key_tree).read_text())
    all_species = list(ref.species)
    targets = a.species or all_species
    print(f"{len(all_species)} species; grading with the '{a.tnorm}' t-norm")

    rows = []
    # (1) known specimens.
    # The specimen is withheld from the key it is graded against. Grading it on the published key
    # would score it against thresholds and ranges it helped compute — it cannot fall outside its
    # own contribution — and every rate that follows would be flattering. Rebuilt trees are cached
    # so a re-run costs nothing.
    cache = out / "keys_loo"
    cache.mkdir(exist_ok=True)
    loo_n = 0
    for sid in ref.raw.index:
        if a.known_arm == "published":
            key_for_sid = published
        else:
            cf = cache / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', str(sid))}.json"
            if cf.exists():
                key_for_sid = json.loads(cf.read_text())
            else:
                with tempfile.TemporaryDirectory() as td:
                    key_for_sid = build_key(Path(a.matrix_dir), all_species, Path(td), a.python,
                                            a.taxon_profile, exclude_specimens=[str(sid)])
                if "error" not in key_for_sid:
                    cf.write_text(json.dumps(key_for_sid))
            if "error" in key_for_sid:
                print(f"  rebuild without {sid} FAILED — skipped")
                continue
            loo_n += 1
            if loo_n % 25 == 0:
                print(f"  known arm, leave-one-specimen-out: {loo_n}/{len(ref.raw.index)}")
        kp = fuzzy_key_path(key_for_sid, ref.raw.loc[sid], a.tnorm)
        if not kp:
            continue
        rows.append({"specimen_id": sid, "species": ref.species_of.get(sid), "arm": "known",
                     "resolved": kp["resolved"], "keys_to": kp.get("terminal"),
                     "correct": kp.get("terminal") == ref.species_of.get(sid),
                     "decisiveness": round(kp["decisiveness"], 4), "fit": round(kp["fit"], 4),
                     "score": round(min(kp["decisiveness"], kp["fit"]), 4) if kp["resolved"] else None,
                     "used_support": kp["used_support"], "weakest": kp["weakest"]})

    # (2) each species held out, key rebuilt without it — the specimen the key has never seen
    for code in targets:
        others = [x for x in all_species if x != code]
        with tempfile.TemporaryDirectory() as td:
            key = build_key(Path(a.matrix_dir), others, Path(td), a.python, a.taxon_profile)
        if "error" in key:
            print(f"  {code:12s} key rebuild FAILED")
            continue
        ids = [i for i in ref.raw.index if ref.species_of.get(i) == code]
        for sid in ids:
            kp = fuzzy_key_path(key, ref.raw.loc[sid], a.tnorm)
            if not kp:
                continue
            rows.append({"specimen_id": sid, "species": code, "arm": "held_out",
                         "resolved": kp["resolved"], "keys_to": kp.get("terminal"),
                         "correct": False,
                         "decisiveness": round(kp["decisiveness"], 4), "fit": round(kp["fit"], 4),
                         "score": round(min(kp["decisiveness"], kp["fit"]), 4) if kp["resolved"] else None,
                         "used_support": kp["used_support"], "weakest": kp["weakest"]})
        print(f"  {code:12s} {len(ids):2d} specimens graded")

    d = pd.DataFrame(rows)
    d.to_csv(out / "fuzzy_key_specimens.tsv", sep="\t", index=False)

    def auc(pos, neg):
        """P(a random known scores above a random held-out). 0.5 = no separation."""
        pos, neg = [x for x in pos if x == x], [x for x in neg if x == x]
        if not pos or not neg:
            return None
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return round(wins / (len(pos) * len(neg)), 3)

    res = d[d["resolved"] == True]                                        # noqa: E712
    known, held = res[res.arm == "known"], res[res.arm == "held_out"]
    summary = {
        "version": VERSION,
        "known_arm": a.known_arm, "tnorm": a.tnorm,
        "specimens_graded": int(len(d)),
        "reached_a_species": int(len(res)),
        "unresolved_known": int((d[d.arm == "known"]["resolved"] == False).sum()),     # noqa: E712
        "unresolved_held_out": int((d[d.arm == "held_out"]["resolved"] == False).sum()),  # noqa: E712
        "median_score_known": float(known["score"].median()) if len(known) else None,
        "median_score_held_out": float(held["score"].median()) if len(held) else None,
        "AUC_known_vs_held_out_score": auc(known["score"], held["score"]),
        "AUC_known_vs_held_out_fit": auc(known["fit"], held["fit"]),
        "AUC_known_vs_held_out_decisiveness": auc(known["decisiveness"], held["decisiveness"]),
        "specimens_rescued_from_dead_end_by_supporting_characters":
            int((d["used_support"] > 0).sum()),
    }
    # does a low score catch the silent misidentifications the crisp key missed?
    if len(known):
        right, wrong = known[known.correct], known[~known.correct]
        summary["known_correct"] = int(len(right))
        summary["known_misidentified"] = int(len(wrong))
        summary["AUC_correct_vs_misidentified"] = auc(right["score"], wrong["score"])
        summary["median_score_correct"] = float(right["score"].median()) if len(right) else None
        summary["median_score_misidentified"] = float(wrong["score"].median()) if len(wrong) else None
    (out / "fuzzy_key_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
