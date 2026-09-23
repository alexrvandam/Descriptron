#!/usr/bin/env python3
"""
biorag_key_holdout_v1.py — would the key itself notice a species it has never seen?
===================================================================================

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
import subprocess
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

VERSION = "1.0"
BUILDER = Path(__file__).resolve().parent / "biorag_key_builder_v1.py"


def build_key(matrix_dir: Path, species: list, out_dir: Path, python: str, profile: str) -> dict:
    """Rebuild the key from a species list. The tree is computed from the matrix, so no model
    is needed and the result is deterministic."""
    cmd = [python, str(BUILDER), "--matrix_dir", str(matrix_dir), "--output_dir", str(out_dir),
           "--species", *species, "--no-loo", "--llm-backend", "none",
           "--taxon-profile", profile]
    r = subprocess.run(cmd, capture_output=True, text=True)
    tree = out_dir / "key_tree.json"
    if not tree.exists():
        return {"error": (r.stderr or r.stdout or "")[-400:]}
    return json.loads(tree.read_text())


def main():
    ap = argparse.ArgumentParser(description="Hold out each species and rebuild the key without it")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--species", nargs="*", default=None, help="only these (default: all)")
    ap.add_argument("--baseline", default=None,
                    help="identification_test.tsv from the published key: gives the rate at which "
                         "KNOWN specimens fail to key out, which is the coverage baseline the dead "
                         "ends here have to be read against")
    ap.add_argument("--candidate_coco", default=None,
                    help="instead of holding species out, run outside material down the EXISTING key "
                         "(--key_tree) and record what the key does with it")
    ap.add_argument("--category_map", default=None)
    ap.add_argument("--key_tree", default=None, help="the published key, for --candidate_coco")
    ap.add_argument("--label", default="external material")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    if a.candidate_coco:
        if not a.key_tree or not Path(a.key_tree).exists():
            sys.exit("--candidate_coco needs --key_tree")
        profile = pol.load_taxon_profile(a.taxon_profile)
        key = json.loads(Path(a.key_tree).read_text())
        ext = features_from_coco(Path(a.candidate_coco), profile,
                                 json.loads(a.category_map) if a.category_map else {})
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for img, row in ext.iterrows():
            kp = key_path(key, row)
            rows.append({"image": img,
                         "outcome": ("no character" if not kp.get("resolved") and kp.get("missing")
                                     else "key exhausted" if not kp.get("resolved")
                                     else "keyed, conflict" if kp.get("conflicts", 0) else "keyed, clean"),
                         "keys_to": kp.get("terminal"), "conflicts": kp.get("conflicts", 0),
                         "couplets": len(kp.get("steps") or []),
                         "stopped_at": (kp.get("steps") or [""])[-1][:120]})
        d = pd.DataFrame(rows)
        d.to_csv(out / "key_external_outcomes.tsv", sep="\t", index=False)
        tally = Counter(d["outcome"])
        n = max(1, len(d))
        summary = {"version": VERSION, "label": a.label, "images": len(d),
                   "outcomes": dict(tally),
                   "percent": {k: round(100 * v / n, 1) for k, v in tally.items()},
                   "note": "the key asks for measurements in mm; material without a scale bar cannot "
                           "answer those couplets, which is why a 'no character' result here is a "
                           "limit of the key's applicability, not evidence about the specimen"}
        (out / "key_external_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{a.label}: {len(d)} images down the published key")
        for k, v in tally.most_common():
            print(f"   {k:<18}{v:4d}  {100 * v / n:5.1f}%")
        print(f"-> {out}")
        return

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    ref = Reference(Path(a.matrix_dir), profile)
    all_species = list(ref.species)
    targets = a.species or all_species
    print(f"{len(all_species)} species in the matrix; holding out {len(targets)}")

    rows, per_species = [], []
    for code in targets:
        others = [s for s in all_species if s != code]
        with tempfile.TemporaryDirectory() as td:
            key = build_key(Path(a.matrix_dir), others, Path(td), a.python, a.taxon_profile)
        if "error" in key:
            print(f"  {code:12s} key rebuild FAILED: {key['error'][:120]}")
            per_species.append({"species": code, "n_specimens": 0, "outcome": "key rebuild failed"})
            continue
        ids = [i for i in ref.raw.index if ref.species_of.get(i) == code]
        outcomes = Counter()
        for sid in ids:
            kp = key_path(key, ref.raw.loc[sid])
            if not kp.get("resolved"):
                # separate the two reasons a path can stop: a character this specimen never had
                # (coverage, and it happens to correctly identified specimens too) versus a key
                # that ran out with everything available (a property of the specimen)
                outcome = "no character" if kp.get("missing", 0) else "key exhausted"
            elif kp.get("conflicts", 0) > 0:
                outcome = "keyed, conflict"
            else:
                outcome = "keyed, clean"
            outcomes[outcome] += 1
            rows.append({"species": code, "specimen_id": sid, "outcome": outcome,
                         "keys_to": kp.get("terminal"), "conflicts": kp.get("conflicts", 0),
                         "couplets": len(kp.get("steps") or []),
                         "missing_character": kp.get("missing", 0)})
        worst = ("keyed, clean" if outcomes["keyed, clean"] else
                 "keyed, conflict" if outcomes["keyed, conflict"] else
                 "key exhausted" if outcomes["key exhausted"] else "no character")
        per_species.append({"species": code, "n_specimens": len(ids), "outcome": worst,
                            **{k.replace(", ", "_"): v for k, v in outcomes.items()}})
        print(f"  {code:12s} {len(ids):2d} specimens: "
              + ", ".join(f"{v} {k}" for k, v in outcomes.most_common()))

    det = pd.DataFrame(rows)
    sp = pd.DataFrame(per_species)
    det.to_csv(out / "key_holdout_specimens.tsv", sep="\t", index=False)
    sp.to_csv(out / "key_holdout_species.tsv", sep="\t", index=False)
    tally = Counter(det["outcome"]) if len(det) else Counter()
    n = max(1, len(det))
    # the fair denominator for "would the key mislead?" is the specimens that reached a terminal
    reached = det[det["outcome"].isin(["keyed, conflict", "keyed, clean"])] if len(det) else det
    baseline, false_alarm = {}, {}
    if a.baseline and Path(a.baseline).exists():
        b = pd.read_csv(a.baseline, sep="\t")
        if "loo" in b.columns:
            unres = int((b["loo"] == "unresolved").sum())
            baseline = {"known_specimens": int(len(b)), "unresolved": unres,
                        "unresolved_percent": round(100 * unres / max(1, len(b)), 1),
                        "note": "known specimens, own record withheld, species still in the key: "
                                "the coverage baseline for the dead ends above"}
        # ── the false-alarm arm ──────────────────────────────────────────────
        # Detection without a false-alarm rate says nothing: a rule that fires on everything
        # "detects" every unseen species. The comparison has to be made against described species
        # scored the same way — each specimen keyed against a key rebuilt WITHOUT it, so it is
        # never asked whether it falls outside a range it helped define. The key builder records
        # this as loo_conflicts; without that column the arm is reported as unavailable rather
        # than silently substituted with the published key.
        if "loo_conflicts" in b.columns:
            r = b[~b["loo"].astype(str).isin(["unresolved", "species has no other specimen"])].copy()
            r["c"] = pd.to_numeric(r["loo_conflicts"], errors="coerce").fillna(0)
            g = r.groupby("species")["c"].agg(any_=lambda x: (x > 0).any(),
                                              maj=lambda x: (x > 0).mean() > 0.5,
                                              all_=lambda x: (x > 0).all(), n="size")
            false_alarm = {
                "described_species_scored": int(len(g)),
                "falsely_flagged_any_specimen": int(g["any_"].sum()),
                "falsely_flagged_majority_of_series": int(g["maj"].sum()),
                "falsely_flagged_whole_series": int(g["all_"].sum()),
                "note": "described species wrongly called new, each specimen keyed against a key "
                        "rebuilt without it. Read every detection figure above against these"}
        else:
            false_alarm = {"unavailable": "the baseline identification_test.tsv has no "
                                          "loo_conflicts column; rebuild the key with the current "
                                          "builder. Detection figures above have no false-alarm "
                                          "rate to be read against and must not be quoted alone"}
    summary = {"version": VERSION, "generated": datetime.now().isoformat(),
               "species_held_out": len(per_species), "specimens": len(det),
               "specimen_outcomes": dict(tally),
               "specimen_percent": {k: round(100 * v / n, 1) for k, v in tally.items()},
               "species_with_any_clean_misidentification":
                   int((sp["outcome"] == "keyed, clean").sum()) if len(sp) else 0,
               "species_where_every_specimen_fails_to_key":
                   int((sp["outcome"].isin(["no character", "key exhausted"])).sum()) if len(sp) else 0,
               "specimens_reaching_a_species": len(reached),
               "of_those_flagged_by_a_conflict_percent":
                   round(100 * (reached["outcome"] == "keyed, conflict").mean(), 1) if len(reached) else None,
               "of_those_misidentified_without_warning_percent":
                   round(100 * (reached["outcome"] == "keyed, clean").mean(), 1) if len(reached) else None,
               "species_with_a_novelty_signal":
                   int(det[det["outcome"].isin(["keyed, conflict", "key exhausted"])]["species"].nunique())
                   if len(det) else 0,
               # the default rule: one aberrant specimen is an outlier, a series of them is a species.
               # Reported here so the series-level verdict travels with the per-specimen one; which
               # rule to use on a given taxon is measured by biorag_calibrate_v1.py, not assumed.
               "species_flagged_by_majority_of_series":
                   int(sum(1 for _, g in reached.groupby("species")
                           if (g["outcome"] == "keyed, conflict").mean() > 0.5)) if len(reached) else 0,
               "species_flagged_by_whole_series":
                   int(sum(1 for _, g in reached.groupby("species")
                           if (g["outcome"] == "keyed, conflict").all())) if len(reached) else 0,
               "species_reaching_a_terminal": int(reached["species"].nunique()) if len(reached) else 0,
               "margin_majority_of_series": None,
               "coverage_baseline": baseline,
               "false_alarm_arm": false_alarm}
    if false_alarm.get("described_species_scored"):
        det_m = summary["species_flagged_by_majority_of_series"] / max(1, summary["species_reaching_a_terminal"])
        fa_m = false_alarm["falsely_flagged_majority_of_series"] / max(1, false_alarm["described_species_scored"])
        summary["margin_majority_of_series"] = round(det_m - fa_m, 3)
    (out / "key_holdout_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary["specimen_outcomes"], indent=1))
    print(f"species whose specimens key cleanly to a wrong species: "
          f"{summary['species_with_any_clean_misidentification']} of {len(sp)}")
    print(f"species where no specimen keys out at all: "
          f"{summary['species_where_every_specimen_fails_to_key']} of {len(sp)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
