#!/usr/bin/env python3
"""
biorag_calibrate_v1.py — measure your own cutoffs before you trust them
======================================================================

Every threshold this workflow uses is a property of the material, not of the
method: which descriptive characters a scorer can reproduce, how far a specimen
must sit from a species before the distance means anything, and how many couplets
a specimen may violate before it is worth a second look. The numbers obtained on
one genus are not transferable, so this step measures them on yours and writes
the answers out as the cutoffs to use.

Three measurements, all hold-outs on your own reference set:

  leave one SPECIMEN out   how often the key identifies a specimen correctly when
                           its own record is withheld, reported per series size.
                           This is what tells you whether you have enough
                           specimens yet.
  leave one SPECIES  out   the key is rebuilt without each species in turn and
                           that species is run down it. Gives, for every candidate
                           rule, how many unseen species it catches and how many
                           described ones it falsely flags.
  character reliability    (read from the retest, if it was run) which descriptive
                           characters survive on your images.

It then recommends a novelty rule, the one with no false alarms where possible,
and says plainly when the reference set is too small to calibrate anything.

  python biorag_calibrate_v1.py --matrix_dir "$M/compiled_key_tier" \\
      --taxon_profile <profile.yaml> --key_tree "$M/key/key_tree.json" \\
      --out_dir "$M/calibration" [--reliability "$M/descriptive_states/character_reliability.json"]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                    # noqa: E402
from biorag_novelty_score_v1 import Reference, key_path                # noqa: E402
from biorag_key_builder_v1 import tolerance_range                      # noqa: E402

VERSION = "1.0"
BUILDER = Path(__file__).resolve().parent / "biorag_key_builder_v1.py"

# what counts as enough material to calibrate at all
MIN_SPECIES = 8
MIN_SPECIMENS_PER_SPECIES = 3
GOOD_SPECIMENS_PER_SPECIES = 6

RULES = {
    "any specimen leaves the observed range":
        lambda g: bool((g["conflicts"] > 0).any()),
    "two or more couplets left the range, in one specimen":
        lambda g: bool((g["conflicts"] >= 2).any()),
    "a majority of the series leaves the range":
        lambda g: bool((g["conflicts"] > 0).mean() > 0.5),
    "the whole series leaves the range":
        lambda g: bool((g["conflicts"] > 0).all()),
}


def walk(tree, cand, coverage=0.95):
    """key_path, but counting the tolerance-widened conflicts alongside the raw ones so the
    detection arm and the false-alarm arm are always compared on the same footing."""
    if not tree or "couplets" not in tree:
        return {"resolved": False, "conflicts": 0, "tol_conflicts": 0}
    by_no = {c["number"]: c for c in tree["couplets"]}
    node, conflicts, tol, seen = tree["couplets"][0]["number"], 0, 0, set()
    exc = []
    while node in by_no and node not in seen:
        seen.add(node)
        c = by_no[node]
        if not c.get("characters"):
            return {"resolved": False, "conflicts": conflicts, "tol_conflicts": tol, "excursions": exc}
        ch = c["characters"][0]
        v = cand.get(ch["feature_id"], float("nan"))
        if v != v:
            return {"resolved": False, "conflicts": conflicts, "tol_conflicts": tol, "excursions": exc}
        is_A = (v <= ch["threshold"]) if ch["A_operator"] == "<=" else (v > ch["threshold"])
        rng = ch["A_range"] if is_A else ch["B_range"]
        if rng:
            lo_, hi_ = min(rng), max(rng)
            conflicts += int(v < lo_ or v > hi_)
            w = (hi_ - lo_) or abs(hi_) or 1.0
            exc.append(round(max(0.0, lo_ - v, v - hi_) / w, 4))
            tl, th = tolerance_range(rng, ch.get("A_n" if is_A else "B_n"), coverage)
            tol += int(v < tl or v > th)
        nxt = c["A_goto" if is_A else "B_goto"]
        if nxt in by_no:
            node = nxt
        else:
            return {"resolved": True, "conflicts": conflicts, "tol_conflicts": tol, "excursions": exc}
    return {"resolved": False, "conflicts": conflicts, "tol_conflicts": tol, "excursions": exc}


def key_calls(held: pd.DataFrame, known: pd.DataFrame) -> dict:
    """Per-species calls of the key at every operating point it has: each rule, at every
    widening of the observed range that changes a call. `held` is the species-withheld arm,
    `known` the specimen-withheld arm; both carry the per-couplet excursions as a string."""
    def ex(v):
        return [float(x) for x in str(v).split(",") if x not in ("", "nan")]
    ks = sorted({0.0} | {e for fr in (held, known) if len(fr) for v in fr["excursions"] for e in ex(v)})
    out = {}
    for k in ks:
        for name, fn in RULES.items():
            calls = []
            for fr in (held, known):
                if not len(fr):
                    calls.append(pd.Series(dtype=bool))
                    continue
                f = fr.assign(conflicts=[sum(1 for e in ex(v) if e > k) for v in fr["excursions"]])
                calls.append(f.groupby("species").apply(fn).astype(bool))
            out[(k, name)] = tuple(calls)
    return out


def _has_loo_conflicts(p: Path) -> bool:
    try:
        return p.exists() and "loo_conflicts_tol" in pd.read_csv(p, sep="\t", nrows=1).columns
    except Exception:                       # unreadable or empty — treat as absent
        return False


def build_key(matrix_dir: Path, species: list, out_dir: Path, python: str, profile: str,
              loo: bool = False):
    """A key with no wording: the tree is deterministic, so no model is needed to evaluate it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, str(BUILDER), "--matrix_dir", str(matrix_dir),
           "--output_dir", str(out_dir), "--species", *species,
           "--llm-backend", "none", "--taxon-profile", profile]
    if not loo:
        cmd.append("--no-loo")
    r = subprocess.run(cmd, capture_output=True, text=True)
    tree = out_dir / "key_tree.json"
    return json.loads(tree.read_text()) if tree.exists() else {"error": (r.stderr or "")[-300:]}


def main():
    ap = argparse.ArgumentParser(description="Calibrate the cutoffs on your own reference set")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--key_tree", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--reliability", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--no_matrix", action="store_true",
                    help="skip scoring the character matrix beside the key (saves a few minutes; "
                         "the report then cannot say which instrument to use)")
    ap.add_argument("--min_sets", type=int, default=2,
                    help="character sets that must agree before the matrix calls a specimen novel")
    ap.add_argument("--no_figures", action="store_true",
                    help="skip the two summary sheets")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    ref = Reference(Path(a.matrix_dir), profile)
    published = json.loads(Path(a.key_tree).read_text())
    species = list(ref.species)
    per_sp = {s: [i for i in ref.raw.index if ref.species_of.get(i) == s] for s in species}
    n_spec = {s: len(v) for s, v in per_sp.items()}

    # ── is there enough material to calibrate at all? ─────────────────────────
    warnings = []
    if len(species) < MIN_SPECIES:
        warnings.append(
            f"only {len(species)} species in the reference set. Below about {MIN_SPECIES} the "
            f"hold-out estimates are too noisy to set a cutoff from; treat everything below as "
            f"indicative and add material before relying on it")
    thin = [s for s, n in n_spec.items() if n < MIN_SPECIMENS_PER_SPECIES]
    if thin:
        warnings.append(
            f"{len(thin)} species have fewer than {MIN_SPECIMENS_PER_SPECIES} specimens "
            f"({', '.join(sorted(thin)[:8])}{' …' if len(thin) > 8 else ''}). A range computed from "
            f"one or two specimens is not a range, and every rule below inherits that")
    med = int(np.median(list(n_spec.values()))) if n_spec else 0
    if med < GOOD_SPECIMENS_PER_SPECIES:
        warnings.append(
            f"median series size is {med}. Identification accuracy rises steeply with series size, "
            f"so expect the key to under-perform until most species reach about "
            f"{GOOD_SPECIMENS_PER_SPECIES}")

    # ── leave one SPECIMEN out: does the key identify what it already knows? ──
    # The key builder already withholds each specimen and rebuilds, writing identification_test.tsv.
    # Use that when it is there; only fall back to resubstitution, clearly labelled, when it is not,
    # because resubstitution scores a specimen against a key built from that specimen and flatters.
    id_test = Path(a.key_tree).parent / "identification_test.tsv"
    # The published key may predate the loo_conflicts column. Rather than rebuild over it and lose
    # the wording, build a plain, unworded copy once into the calibration directory and cache it.
    if not _has_loo_conflicts(id_test):
        side = out / "key_loo" / "identification_test.tsv"
        if not _has_loo_conflicts(side):
            print("  building an unworded copy of the key with leave-one-specimen-out "
                  "(once; cached in calibration/key_loo) ...")
            build_key(Path(a.matrix_dir), species, out / "key_loo", a.python, a.taxon_profile,
                      loo=True)
        if _has_loo_conflicts(side):
            id_test = side
    arm = "leave_one_specimen_out"
    if id_test.exists():
        t = pd.read_csv(id_test, sep="\t")
        kdf = pd.DataFrame({"specimen_id": t["specimen_id"], "species": t["species"],
                            "resolved": t["loo"].astype(str).ne("unresolved"),
                            "correct": t["loo_ok"].astype(bool)})
    else:
        arm = "resubstitution_only"
        warnings.append("identification_test.tsv not found beside the key, so the accuracy below is "
                        "resubstitution — each specimen scored against a key built from it — which "
                        "overstates performance. Rebuild the key without --no-loo for the real figure")
        kdf = pd.DataFrame([{"specimen_id": sid, "species": ref.species_of.get(sid),
                             "resolved": bool(key_path(published, ref.raw.loc[sid]).get("resolved")),
                             "correct": key_path(published, ref.raw.loc[sid]).get("terminal")
                             == ref.species_of.get(sid)} for sid in ref.raw.index])
    kdf["series"] = kdf["species"].map(n_spec)
    res = kdf[kdf.resolved]
    by_size = (res.assign(bin=pd.cut(res["series"], [0, 3, 5, 7, 999],
                                     labels=["2-3", "4-5", "6-7", "8+"]))
               .groupby("bin", observed=True)["correct"].agg(["mean", "size"]))

    # ── the false-alarm arm ──────────────────────────────────────────────────
    # How often does a specimen of a species that IS in the key trip the novelty rule?
    # It must be counted under a key rebuilt without that specimen: asking a specimen whether
    # it lies outside a range it helped define is the same mistake as resubstitution, and it
    # makes the false-alarm rate look better than it is. The key builder records this as
    # loo_conflicts; if the column is absent the published key is used and the report says so.
    fa_arm = "leave_one_specimen_out"
    if _has_loo_conflicts(id_test):
        t2 = pd.read_csv(id_test, sep="\t")
        t2 = t2[t2["loo"].astype(str).isin(["unresolved", "species has no other specimen"]) == False]
        fa_df = pd.DataFrame({"specimen_id": t2["specimen_id"],
                              "species": t2["species"], "resolved": True,
                              "conflicts": pd.to_numeric(t2["loo_conflicts"],
                                                         errors="coerce").fillna(0).astype(int),
                              "tol_conflicts": pd.to_numeric(t2["loo_conflicts_tol"],
                                                             errors="coerce").fillna(0).astype(int),
                              "excursions": t2["loo_excursions"].fillna("")})
    else:
        fa_arm = "published_key"
        warnings.append("the key's identification_test.tsv carries no loo_conflicts column, so the "
                        "false-alarm rates below were counted on the published key — where each "
                        "specimen helped define the ranges it is being tested against, which makes "
                        "them optimistic. Rebuild the key with the current builder for the honest "
                        "figure")
        fa_df = pd.DataFrame([{"specimen_id": sid,
                               "species": ref.species_of.get(sid), "resolved": kp["resolved"],
                               "conflicts": kp["conflicts"], "tol_conflicts": kp["tol_conflicts"],
                               "excursions": ",".join(str(x) for x in kp.get("excursions", []))}
                              for sid in ref.raw.index
                              for kp in [walk(published, ref.raw.loc[sid])]])
    fa_res = fa_df[fa_df.resolved]

    # ── leave one SPECIES out: rebuild without it and run it down ────────────
    held = []
    for code in species:
        others = [s for s in species if s != code]
        with tempfile.TemporaryDirectory() as td:
            key = build_key(Path(a.matrix_dir), others, Path(td), a.python, a.taxon_profile)
        if "error" in key:
            warnings.append(f"key rebuild failed without {code}; it is excluded from the estimates")
            continue
        for sid in per_sp[code]:
            kp = walk(key, ref.raw.loc[sid])
            held.append({"specimen_id": sid, "species": code, "resolved": kp["resolved"],
                         "conflicts": kp["conflicts"], "tol_conflicts": kp["tol_conflicts"],
                         "excursions": ",".join(str(x) for x in kp.get("excursions", []))})
        print(f"  held out {code}: {len(per_sp[code])} specimens")
    hdf = pd.DataFrame(held)

    # ── every candidate rule, on both arms ────────────────────────────────────
    # Both range definitions are scored side by side. The observed min/max is what a reader
    # assumes "outside the range ever recorded" means; the tolerance interval is what that phrase
    # has to become before it is a statement about the species rather than about how many
    # specimens happened to be measured.
    rows = []
    hres = hdf[hdf.resolved] if len(hdf) else hdf
    for col, kind in (("conflicts", "observed min/max"), ("tol_conflicts", "95% tolerance interval")):
        for name, fn in RULES.items():
            g_det = hres.assign(conflicts=hres[col]) if len(hres) else hres
            g_fa = fa_res.assign(conflicts=fa_res[col])
            det = sum(fn(g) for _, g in g_det.groupby("species")) if len(g_det) else 0
            fa = sum(fn(g) for _, g in g_fa.groupby("species"))
            n_h = hres["species"].nunique() if len(hres) else 0
            n_d = fa_res["species"].nunique()
            rows.append({"range_definition": kind, "rule": name,
                         "unseen_species_flagged": int(det), "of_unseen": int(n_h),
                         "described_falsely_flagged": int(fa), "of_described": int(n_d),
                         "detection": round(det / n_h, 3) if n_h else None,
                         "false_alarm": round(fa / max(1, n_d), 3),
                         "margin": round(det / max(1, n_h) - fa / max(1, n_d), 3)})
    rules = pd.DataFrame(rows)
    rules.to_csv(out / "novelty_rules.tsv", sep="\t", index=False)

    # ── the trade-off curve ──────────────────────────────────────────────────
    # A range of n specimens is not the species' range, so "outside the range" needs a margin
    # before it means anything. Rather than pick one, sweep it: widen every range by k times its
    # own width and re-score both arms at each k, from the raw min/max (k = 0) outwards. The
    # excursions were recorded during the single walk, so this costs nothing.
    def _ex(v):
        return [float(x) for x in str(v).split(",") if x not in ("", "nan")]

    sweep = []
    for k in [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]:
        for name, fn in RULES.items():
            def cnt(frame):
                f = frame.copy()
                f["conflicts"] = [sum(1 for e in _ex(v) if e > k) for v in f["excursions"]]
                return f
            d, f_ = cnt(hres), cnt(fa_res)
            det = sum(fn(g) for _, g in d.groupby("species")) if len(d) else 0
            fa_ = sum(fn(g) for _, g in f_.groupby("species"))
            nh, nd = (d["species"].nunique() if len(d) else 0), f_["species"].nunique()
            sweep.append({"widening_k": k, "rule": name, "caught": int(det), "of_unseen": int(nh),
                          "false_alarms": int(fa_), "of_described": int(nd),
                          "detection": round(det / max(1, nh), 3),
                          "false_alarm": round(fa_ / max(1, nd), 3),
                          "margin": round(det / max(1, nh) - fa_ / max(1, nd), 3)})
    sw = pd.DataFrame(sweep)
    sw.to_csv(out / "novelty_tradeoff.tsv", sep="\t", index=False)
    bestsw = sw.sort_values(["margin", "detection"], ascending=False).iloc[0]

    # both arms, specimen by specimen, so any other script can re-score the key at every
    # operating point (and choose one without looking at the species it reports on)
    arms = pd.concat([hres.assign(arm="species_withheld") if len(hres) else hres,
                      fa_res.assign(arm="specimen_withheld")], ignore_index=True, sort=False)
    arms.to_csv(out / "novelty_specimens.tsv", sep="\t", index=False)

    # ── the same two questions put to the character matrix ───────────────────
    # The key is one instrument. The matrix it was computed from is another: a distance to the
    # nearest described species, in units of that species' own spread, called novel when at
    # least two independent character sets agree. Which of the two is the better guide — for
    # a name, and for a warning that the material may be undescribed — is a property of the
    # reference set, so it is measured here rather than assumed.
    inst = {}
    if not a.no_matrix:
        from biorag_congruence_compare_v1 import (score_arms, sweep as g_sweep, nested,
                                                  identify, wilson)
        from biorag_novelty_score_v1 import SETS
        base = [n for n in SETS if n in ref.tables and len(ref.tables[n].columns)]
        print(f"  scoring the character matrix on both arms ({', '.join(base)}) ...")
        det, fal, ranks = score_arms(ref, base)
        det.to_csv(out / "matrix_detection_scores.tsv", sep="\t", index=False)
        fal.to_csv(out / "matrix_false_alarm_scores.tsv", sep="\t", index=False)
        ranks.to_csv(out / "matrix_identification_distances.tsv", sep="\t", index=False)
        msw, mcalls = g_sweep(det, fal, base, a.min_sets)
        msw.to_csv(out / "matrix_tradeoff.tsv", sep="\t", index=False)
        mb = msw.sort_values(["margin", "caught"], ascending=False).iloc[0]
        mn = nested(msw, mcalls, species)
        kn = nested(None, key_calls(hres, fa_res), species)
        named = identify(ranks, base)
        m_ok = named["named"] == named["species"]
        kname = kdf.set_index("specimen_id")
        k_named = kname["resolved"]
        both = named.join(kname[["resolved", "correct"]], how="inner")
        key_names = pd.read_csv(id_test, sep="\t").set_index("specimen_id")["loo"] \
            if id_test.exists() else pd.Series(dtype=str)
        agree = both.index[both["resolved"] & (key_names.reindex(both.index) == both["named"])]
        inst = {
            "novelty": {
                "key": {"best_in_sample_margin": float(bestsw["margin"]),
                        "nested_margin": kn["margin"],
                        "nested_point": f"{kn['caught']}/{kn['of']} caught, "
                                        f"{kn['false_alarms']}/{kn['of']} false"},
                "character_matrix": {"best_in_sample_margin": round(float(mb["margin"]), 3),
                                     "best_in_sample_rule": mb["rule"],
                                     "best_in_sample_threshold": round(float(mb["threshold"]), 3),
                                     "nested_margin": mn["margin"],
                                     "nested_point": f"{mn['caught']}/{mn['of']} caught, "
                                                     f"{mn['false_alarms']}/{mn['of']} false",
                                     "nested_detection_95CI": wilson(mn["caught"], mn["of"]),
                                     "min_sets": a.min_sets}},
            "identification_leave_one_specimen_out": {
                "key": {"named": int(k_named.sum()), "correct": int(kname["correct"].sum()),
                        "of": int(len(kname))},
                "character_matrix": {"named": int(len(named)), "correct": int(m_ok.sum()),
                                     "of": int(len(ref.raw))},
                "key_and_matrix_agree": {"specimens": int(len(agree)),
                                         "correct": int(m_ok.reindex(agree).sum())}},
        }
        inst["recommended_for_novelty"] = ("character_matrix" if mn["margin"] >= kn["margin"]
                                           else "key")
        inst["recommended_for_identification"] = (
            "character_matrix" if m_ok.sum() >= kname["correct"].sum() else "key")

    # per-species accuracy, so the key figure can colour each terminal by its own support
    persp = (kdf.groupby("species")
             .agg(n=("specimen_id", "size"), resolved=("resolved", "sum"),
                  loo_accuracy=("correct", "mean")).reset_index())
    persp["n"] = persp["species"].map(n_spec)
    persp.to_csv(out / "per_species.tsv", sep="\t", index=False)

    # A rule is only worth recommending if it separates the two arms at all. Among rules with no
    # false alarms take the most sensitive; otherwise take the widest margin between the rate at
    # which it catches unseen species and the rate at which it cries wolf on described ones.
    clean = rules[rules.described_falsely_flagged == 0].sort_values("unseen_species_flagged",
                                                                    ascending=False)
    best = rules.sort_values(["margin", "detection"], ascending=False).iloc[0]
    if len(clean) and float(clean.iloc[0]["margin"]) >= float(best["margin"]):
        best = clean.iloc[0]
    if float(bestsw["margin"]) > float(best["margin"]) + 0.02:
        warnings.append(
            f"the raw observed range is not the best operating point here: widening every range by "
            f"{bestsw['widening_k']:g} times its own width and requiring \u201c{bestsw['rule']}\u201d "
            f"gives {bestsw['caught']}/{bestsw['of_unseen']} against "
            f"{bestsw['false_alarms']}/{bestsw['of_described']} (margin {bestsw['margin']:.2f} vs "
            f"{float(best['margin']):.2f}). See novelty_tradeoff.tsv")
    if float(best["margin"]) <= 0.15 and float(bestsw["margin"]) <= 0.15:
        warnings.append(
            f"no rule separates unseen species from described ones on this material (best margin "
            f"{best['margin']:.2f}). Treat the key as an identification tool only, and do not use it "
            f"to claim a specimen is undescribed until there are more specimens per species")

    # why the false-alarm rate is what it is: the observed range of n specimens is not the
    # species' range — a further specimen lands outside it with probability about 2/(n+1) per
    # character however good that character is. Recorded so the figure can show it.
    diag = []
    fr = fa_res.copy()
    fr["series"] = fr["species"].map(n_spec)
    fr["any_out"] = [any(float(x) > 0 for x in str(v).split(",") if x not in ("", "nan"))
                     for v in fr["excursions"]]
    fr["path_len"] = [len([x for x in str(v).split(",") if x not in ("", "nan")])
                      for v in fr["excursions"]]
    for lo_, hi_, lab in ((2, 3, "2-3"), (4, 5, "4-5"), (6, 7, "6-7"), (8, 999, "8+")):
        g = fr[(fr.series >= lo_) & (fr.series <= hi_)]
        if not len(g):
            continue
        k = max(1, int(g["series"].median()) - 1)                  # specimens defining the range
        L_ = max(1.0, float(g["path_len"].mean()))
        diag.append({"series": lab, "n": int(len(g)),
                     "observed": round(float(g["any_out"].mean()), 3),
                     "expected_from_sample_size": round(1 - (1 - 2 / (k + 1)) ** L_, 3)})

    rel = {}
    if a.reliability and Path(a.reliability).exists():
        rel = json.loads(Path(a.reliability).read_text())

    report = {
        "version": VERSION,
        "reference_set": {"species": len(species), "specimens": int(len(ref.raw)),
                          "median_series": med, "smallest_series": int(min(n_spec.values()) if n_spec else 0)},
        "specimen_level_arm": arm,
        "false_alarm_arm": fa_arm,
        "leave_one_specimen_out": {
            "overall_correct": round(float(res["correct"].mean()), 3) if len(res) else None,
            "unresolved": int((~kdf.resolved).sum()),
            "by_series_size": {str(k): {"accuracy": round(float(v["mean"]), 3), "n": int(v["size"])}
                               for k, v in by_size.iterrows()}},
        "novelty_rules": rows,
        "range_diagnostic": diag,
        "recommended_rule": best["rule"],
        "recommended_range_definition": best["range_definition"],
        "recommended_rule_margin": float(best["margin"]),
        "best_operating_point": {k: (float(v) if isinstance(v, (int, float)) else v)
                                 for k, v in bestsw.to_dict().items()},
        "recommended_rule_detection": best["detection"],
        "recommended_rule_false_alarm": best["false_alarm"],
        "descriptive_characters": {"use": rel.get("use", []), "use_coarse": rel.get("use_coarse", []),
                                   "set_aside": rel.get("flag", [])} if rel else
        {"note": "character reliability was not measured; run descriptive_states then char_reliability"},
        "instruments": inst or {"note": "the character matrix was not scored (--no_matrix)"},
        "warnings": warnings,
    }
    (out / "calibration.json").write_text(json.dumps(report, indent=2))

    L = [f"Calibration for {(profile.get('taxon') or {}).get('genus') or 'this taxon'}", "",
         f"Reference set: {len(species)} species, {len(ref.raw)} specimens, "
         f"median {med} per species (smallest {report['reference_set']['smallest_series']}).", ""]
    if warnings:
        L += ["Before using any number below:"] + [f"  - {w}" for w in warnings] + [""]
    L += ["Does the key identify a specimen when its own record is withheld?"
          if arm == "leave_one_specimen_out" else
          "Resubstitution only (no hold-out available — this flatters the key):"]
    for k, v in report["leave_one_specimen_out"]["by_series_size"].items():
        L.append(f"   {k:>4} specimens per species: {100 * v['accuracy']:5.1f}%  (n={v['n']})")
    L += ["", "Recognising a species the key has never seen — your rules, your material:"]
    for kind in ("observed min/max", "95% tolerance interval"):
        L += [f"   range taken as the {kind}:",
              f"     {'rule':<50}{'caught':>9}{'false alarms':>14}{'margin':>9}"]
        for r in [x for x in rows if x["range_definition"] == kind]:
            L.append(f"     {r['rule']:<50}{r['unseen_species_flagged']:>4}/{r['of_unseen']:<4}"
                     f"{r['described_falsely_flagged']:>8}/{r['of_described']:<5}{r['margin']:>8.2f}")
    L += ["", f"Recommended: {best['rule']}  ({best['range_definition']})",
          f"   catches {best['unseen_species_flagged']} of {best['of_unseen']} unseen species, "
          f"falsely flags {best['described_falsely_flagged']} of {best['of_described']} described ones.",
          "   A flag is a reason to examine the specimens, never a determination."]
    if inst:
        nv, idn = inst["novelty"], inst["identification_leave_one_specimen_out"]
        L += ["", "Key or character matrix? Both measured on your material, both arms held out.",
              "   (nested = threshold and rule chosen on the other species, applied to the one left out;",
              "    this is the figure to expect in use, the in-sample best is always higher)",
              f"   recognising an unseen species   key: nested margin {nv['key']['nested_margin']:+.2f} "
              f"({nv['key']['nested_point']})",
              f"                                matrix: nested margin "
              f"{nv['character_matrix']['nested_margin']:+.2f} ({nv['character_matrix']['nested_point']})",
              f"   naming a described specimen     key: {idn['key']['correct']}/{idn['key']['of']} "
              f"({idn['key']['named']} reached a name)",
              f"                                matrix: {idn['character_matrix']['correct']}/"
              f"{idn['character_matrix']['of']}",
              f"   where key and matrix give the same name: {idn['key_and_matrix_agree']['correct']} of "
              f"{idn['key_and_matrix_agree']['specimens']} correct",
              f"   -> for a warning use the {inst['recommended_for_novelty'].replace('_', ' ')}; "
              f"for a name use the {inst['recommended_for_identification'].replace('_', ' ')}, "
              f"and treat agreement between the two as confirmation."]
    if rel:
        L += ["", "Descriptive characters on your images:",
              f"   use as scored: {', '.join(rel.get('use', [])) or 'none'}",
              f"   use at band level: {', '.join(rel.get('use_coarse', [])) or 'none'}",
              f"   set aside (not reproducible here): {', '.join(rel.get('flag', [])) or 'none'}"]
    (out / "calibration_report.txt").write_text("\n".join(L))
    print("\n".join(L))

    # the two summary sheets: everything above in one place, the way a phylogeny carries
    # its topology and its support values together
    if not a.no_figures:
        fig_cmd = [sys.executable, str(Path(__file__).resolve().parent /
                                       "biorag_calibration_figures_v1.py"),
                   "--calibration", str(out), "--key_tree", a.key_tree]
        if a.reliability and Path(a.reliability).exists():
            fig_cmd += ["--reliability", str(Path(a.reliability).with_suffix(".tsv"))]
        r = subprocess.run(fig_cmd, capture_output=True, text=True)
        print(r.stdout.strip() or (r.stderr or "")[-400:])
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
