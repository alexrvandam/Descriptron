#!/usr/bin/env python3
"""
biorag_graph_identify_v1.py — can the knowledge graph do what the key does?
==========================================================================

Three instruments are built from the same specimen matrix and they work in
completely different ways. This scores the third one so that all three can be
compared on the same specimens:

  the key        an ordered sequence of thresholded questions. Uses few
                 characters, in a fixed order, and stops when one is missing.
  the matrices   one global distance in units of within-species SD, averaged
                 over a character set. Uses whatever characters the candidate
                 shares with the reference.
  the graph      the published treatments as machine-readable assertions: for
                 every species, the range it is asserted to occupy in each
                 character. Identification becomes constraint satisfaction —
                 which species' assertions is this specimen consistent with? —
                 and novelty becomes "consistent with none of them".

The graph is read from the JSON-LD treatments (`dsc:numericCitations`, the
min–max values the description actually states), so what is tested is the
published record, not a re-derivation of the matrix.

  python biorag_graph_identify_v1.py --jsonld_dir "$M/treatments/machine_readable/jsonld" \\
      --matrix_dir "$M/compiled_key_tier" --taxon_profile <profile.yaml> \\
      --out_dir "$M/graph_test" [--candidate_coco <outgroup.json> --category_map '{...}']

Two experiments, matching the ones run on the key and the matrices:
  * leave one SPECIES out — its node is deleted from the graph and its specimens
    are scored against the rest. Nothing leaks, because the species it came from
    is gone.
  * identification — every specimen against the whole graph. Note that the
    published ranges were computed WITH that specimen, so this is optimistic and
    is reported as an upper bound.
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                    # noqa: E402
from biorag_novelty_score_v1 import Reference, features_from_coco      # noqa: E402

VERSION = "1.0"
RANGE = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*[–—-]\s*([-+]?\d*\.?\d+)\s*$")
TOL = 1e-9


def load_graph(jsonld_dir: Path) -> dict:
    """{species: {featureId: (lo, hi)}} from what the treatments actually assert."""
    graph = {}
    for f in sorted(Path(jsonld_dir).glob("*.jsonld")):
        d = json.loads(f.read_text())
        code = f.stem
        rngs = {}
        for c in d.get("dsc:numericCitations", []):
            fid, val = c.get("dsc:featureId"), str(c.get("dsc:value", ""))
            m = RANGE.match(val)
            if fid and m:
                lo, hi = float(m.group(1)), float(m.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                prev = rngs.get(fid)
                rngs[fid] = (min(lo, prev[0]), max(hi, prev[1])) if prev else (lo, hi)
        if rngs:
            graph[code] = rngs
    return graph


def match(row: pd.Series, rngs: dict, min_shared: int = 3):
    """How consistent is one specimen with one species' assertions?"""
    shared = violated = 0
    worst = 0.0
    for fid, (lo, hi) in rngs.items():
        v = row.get(fid, math.nan)
        if not (v == v):
            continue
        shared += 1
        if v < lo - TOL or v > hi + TOL:
            violated += 1
            span = max(hi - lo, abs(hi) * 1e-3, 1e-9)
            worst = max(worst, (lo - v if v < lo else v - hi) / span)
    if shared < min_shared:
        return None
    return {"shared": shared, "violated": violated, "rate": violated / shared, "worst": worst}


def score(row: pd.Series, graph: dict, exclude: str = None, min_shared: int = 3):
    """Best-matching species and how badly even that one is violated."""
    best = None
    for sp, rngs in graph.items():
        if sp == exclude:
            continue
        m = match(row, rngs, min_shared)
        if m is None:
            continue
        m["species"] = sp
        if best is None or (m["rate"], -m["shared"]) < (best["rate"], -best["shared"]):
            best = m
    return best


def regraph_without(graph: dict, ref, specimen_id, species) -> dict:
    """The graph as it would read if one specimen had never been collected.

    The published assertions are per-species ranges over the specimens measured, so a specimen
    tested against its own species' asserted range cannot fall outside it: it is inside its own
    contribution by construction. Only that one species' ranges change, and only for the features
    the treatment actually asserts, so the graph stays faithful to what was published.
    """
    if species not in graph:
        return graph
    ids = [i for i in ref.raw.index if ref.species_of.get(i) == species and i != specimen_id]
    if not ids:
        return {k: v for k, v in graph.items() if k != species}       # nothing left to assert
    sub = ref.raw.loc[ids]
    rngs = {}
    for fid in graph[species]:
        if fid not in sub.columns:
            continue
        col = sub[fid].dropna()
        if len(col):
            rngs[fid] = (float(col.min()), float(col.max()))
    g = dict(graph)
    if rngs:
        g[species] = rngs
    else:
        g.pop(species, None)
    return g


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser(description="Identification and novelty from the knowledge graph")
    ap.add_argument("--jsonld_dir", required=True)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--candidate_coco", default=None)
    ap.add_argument("--category_map", default=None)
    ap.add_argument("--candidate_label", default="external material")
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--min_shared", type=int, default=3)
    ap.add_argument("--arm", choices=["leave_one_specimen_out", "published"],
                    default="leave_one_specimen_out",
                    help="The default re-derives each species' asserted ranges without the specimen "
                         "being scored, so a specimen is never tested against a range it helped "
                         "compute. 'published' reproduces the earlier, optimistic run")
    ap.add_argument("--ranges_from", choices=["jsonld", "matrix"], default="jsonld",
                    help="'jsonld' reads each asserted range as published. 'matrix' keeps the "
                         "FEATURES each treatment asserts but recomputes every range from the "
                         "matrix given here, which is what to use when the matrix has been "
                         "rebuilt since the treatments were exported: otherwise the species under "
                         "test is scored on current values and every other species on stale ones")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    ref = Reference(Path(a.matrix_dir), profile)
    graph = load_graph(Path(a.jsonld_dir))
    if a.ranges_from == "matrix":
        for sp in list(graph):
            graph = regraph_without(graph, ref, None, sp)      # None withholds nobody
    print(f"graph: {len(graph)} species, "
          f"{int(np.median([len(v) for v in graph.values()]))} asserted ranges each (median)")

    raw = ref.raw
    rows = []
    for sid in raw.index:
        sp = ref.species_of.get(sid)
        g_loo = graph if a.arm == "published" else regraph_without(graph, ref, sid, sp)
        full = score(raw.loc[sid], g_loo, min_shared=a.min_shared)          # identification arm
        held = score(raw.loc[sid], graph, exclude=sp, min_shared=a.min_shared)  # novelty arm
        # the false-alarm arm: the specimen scored against the WHOLE graph, its own species
        # included but re-derived without it — the operational case for a described specimen
        rows.append({"specimen_id": sid, "species": sp,
                     "id_species": full["species"] if full else None,
                     "id_correct": bool(full and full["species"] == sp),
                     "id_rate": full["rate"] if full else None,
                     "id_shared": full["shared"] if full else None,
                     "false_alarm_rate": full["rate"] if full else None,
                     "novel_rate": held["rate"] if held else None,
                     "novel_nearest": held["species"] if held else None,
                     "novel_shared": held["shared"] if held else None})
    d = pd.DataFrame(rows)
    d.to_csv(out / "graph_specimen_scores.tsv", sep="\t", index=False)

    # threshold: how badly may the best OTHER species be violated before we call it novel?
    # calibrated so that a known specimen, compared with its own species removed, is called
    # novel at the chosen rate only as often as we allow
    known = d["false_alarm_rate"].dropna()
    thr = float(np.quantile(known, 1 - a.fpr)) if len(known) else 1.0
    d["called_novel"] = d["novel_rate"] > thr
    d["falsely_called_novel"] = d["false_alarm_rate"] > thr
    per_sp = d.groupby("species").agg(n=("specimen_id", "size"),
                                      id_accuracy=("id_correct", "mean"),
                                      flagged=("called_novel", "sum"),
                                      false_flagged=("falsely_called_novel", "sum")).reset_index()
    per_sp["all_flagged"] = per_sp["flagged"] == per_sp["n"]
    per_sp["majority_flagged"] = per_sp["flagged"] > per_sp["n"] / 2
    per_sp["any_falsely_flagged"] = per_sp["false_flagged"] > 0
    per_sp["majority_falsely_flagged"] = per_sp["false_flagged"] > per_sp["n"] / 2

    # the whole trade-off, not one point: sweep the violation-rate threshold and record how many
    # unseen species are recognised against how many described ones are falsely flagged
    sweep = []
    for q in np.linspace(0.50, 0.999, 60):
        t = float(np.quantile(known, q)) if len(known) else 1.0
        fl = d.assign(a_=d["novel_rate"] > t, b_=d["false_alarm_rate"] > t)
        g = fl.groupby("species").agg(n=("specimen_id", "size"), a_=("a_", "sum"), b_=("b_", "sum"))
        for rule, fn in (("any specimen", lambda x: x > 0),
                         ("majority of the series", lambda x: None),
                         ("the whole series", lambda x: None)):
            if rule == "any specimen":
                det, fa_ = int((g["a_"] > 0).sum()), int((g["b_"] > 0).sum())
            elif rule == "majority of the series":
                det, fa_ = int((g["a_"] > g["n"] / 2).sum()), int((g["b_"] > g["n"] / 2).sum())
            else:
                det, fa_ = int((g["a_"] == g["n"]).sum()), int((g["b_"] == g["n"]).sum())
            sweep.append({"quantile": round(q, 4), "threshold": round(t, 4), "rule": rule,
                          "caught": det, "of_unseen": len(g), "false_alarms": fa_,
                          "of_described": len(g),
                          "detection": round(det / max(1, len(g)), 3),
                          "false_alarm": round(fa_ / max(1, len(g)), 3),
                          "margin": round((det - fa_) / max(1, len(g)), 3)})
    sw = pd.DataFrame(sweep)
    sw.to_csv(out / "graph_tradeoff.tsv", sep="\t", index=False)
    bestsw = sw.sort_values(["margin", "detection"], ascending=False).iloc[0]
    per_sp.to_csv(out / "graph_species_summary.tsv", sep="\t", index=False)

    n_sp = len(per_sp)
    any_fl = int((per_sp["flagged"] > 0).sum())
    lo, hi = wilson(any_fl, n_sp)
    summary = {"version": VERSION, "species": n_sp, "specimens": int(len(d)),
               "median_ranges_per_species": int(np.median([len(v) for v in graph.values()])),
               "identification_accuracy_upper_bound": round(float(d["id_correct"].mean()), 3),
               "arm": a.arm,
               "identification_note": ("each specimen scored against a graph re-derived without it, "
                                       "so this is a hold-out result"
                                       if a.arm == "leave_one_specimen_out" else
                                       "the published ranges were computed with these specimens "
                                       "included, so this is an upper bound, not a hold-out result"),
               "species_falsely_flagged_any": int(per_sp["any_falsely_flagged"].sum()),
               "species_falsely_flagged_majority": int(per_sp["majority_falsely_flagged"].sum()),
               "species_flagged_majority": int(per_sp["majority_flagged"].sum()),
               "best_operating_point": {k: (float(v) if isinstance(v, (int, float)) else v)
                                        for k, v in bestsw.to_dict().items()},
               "novelty_threshold_violation_rate": round(thr, 3),
               "species_flagged_when_held_out": any_fl,
               "species_flagged_percent": round(100 * any_fl / max(1, n_sp), 1),
               "species_flagged_95CI": [round(100 * lo, 1), round(100 * hi, 1)],
               "species_all_specimens_flagged": int(per_sp["all_flagged"].sum()),
               "specimens_flagged_percent": round(100 * float(d["called_novel"].mean()), 1)}

    if a.candidate_coco:
        cmap = json.loads(a.category_map) if a.category_map else {}
        ext = features_from_coco(Path(a.candidate_coco), profile, cmap)
        erows = []
        for img, row in ext.iterrows():
            m = score(row, graph, min_shared=a.min_shared)
            erows.append({"image": img, "best_species": m["species"] if m else None,
                          "violation_rate": m["rate"] if m else None,
                          "characters_compared": m["shared"] if m else 0,
                          "called_novel": bool(m and m["rate"] > thr)})
        e = pd.DataFrame(erows)
        e.to_csv(out / "graph_external_scores.tsv", sep="\t", index=False)
        ok = e["violation_rate"].notna()
        k = int(e.loc[ok, "called_novel"].sum())
        elo, ehi = wilson(k, int(ok.sum()))
        summary["external"] = {
            "label": a.candidate_label, "images": int(len(e)),
            "comparable": int(ok.sum()),
            "not_comparable": int((~ok).sum()),
            "flagged": k,
            "flagged_percent": round(100 * k / max(1, int(ok.sum())), 1),
            "flagged_95CI": [round(100 * elo, 1), round(100 * ehi, 1)],
            "median_characters_compared": int(e.loc[ok, "characters_compared"].median()) if ok.any() else 0}
        print(f"\n{a.candidate_label}: {int(ok.sum())} of {len(e)} images comparable with the graph, "
              f"{k} called novel ({summary['external']['flagged_percent']}%)")

    (out / "graph_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "external"}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
