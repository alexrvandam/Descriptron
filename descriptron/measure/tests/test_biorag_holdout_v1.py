#!/usr/bin/env python3
"""
Hold-out invariants for the BioRAG evaluation scripts.

One family of error has been made in this project four times: a specimen (or a species) was
withheld from the prediction but left inside a statistic it was then scored against — the
range of a couplet, the asserted interval of a treatment, the pooled spread distances are
divided by, the list of characters a model drew up. Every such leak flatters the instrument
and none is visible in the output. The tests below state the invariant directly:

    changing the data of whatever is withheld must not change its score.

Run:  python measure/tests/test_biorag_holdout_v1.py        (plain python)
  or: python -m pytest measure/tests/test_biorag_holdout_v1.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import biorag_key_builder_v1 as kb                                        # noqa: E402
import biorag_novelty_score_v1 as nv                                      # noqa: E402
import biorag_congruence_compare_v1 as cc                                 # noqa: E402
import biorag_key_fuzzy_v1 as fz                                          # noqa: E402
import biorag_graph_identify_v1 as gi                                     # noqa: E402

FEATS = [("tibia.length_mm", "length", "mm"), ("tibia.height_mm", "length", "mm"),
         ("LAB1.length_mm", "length", "mm"), ("LAB1.height_mm", "length", "mm"),
         ("tibia.aspect_ratio", "aspect_ratio", "x"), ("LAB1.aspect_ratio", "aspect_ratio", "x"),
         ("head.aspect_ratio", "aspect_ratio", "x"), ("wing.aspect_ratio", "aspect_ratio", "x")]
SERIES = {"spA": 2, "spB": 5, "spC": 5, "spD": 4}


def write_matrix(tmp: Path, wild=()):
    """Four species; `wild` lists specimen ids whose every value is replaced by nonsense."""
    rng = np.random.default_rng(3)
    rows = []
    for s, (sp, n) in enumerate(SERIES.items()):
        for i in range(1, n + 1):
            sid = f"{sp}_{i}"
            for k, (f, fam, unit) in enumerate(FEATS):
                v = 1.0 + 0.35 * ((s * (k + 2)) % 4) + 0.1 * k + rng.normal(0, 0.02)
                if sid in wild:
                    v = 40.0 + 7.0 * k + i
                rows.append({"species": sp, "specimen_id": sid, "sex": "male",
                             "category": f.split(".")[0], "base_category": f.split(".")[0],
                             "column": f.split(".")[1], "feature_id": f, "tier": "key",
                             "family": fam, "value": v, "n_images": 1})
    pd.DataFrame(rows).to_csv(tmp / "specimen_matrix_long.csv", index=False)
    pd.DataFrame([{"feature_id": f, "category": f.split(".")[0], "base_category": f.split(".")[0],
                   "column": f.split(".")[1], "tier": "key", "family": fam, "label": f,
                   "definition": f, "unit": unit, "structure_sex": "both", "section": "S",
                   "key_priority": 1, "n_species": 4, "n_specimens": 16, "conversion": ""}
                  for f, fam, unit in FEATS]).to_csv(tmp / "feature_dictionary.tsv", sep="\t", index=False)


PROFILE = {"taxon": {"genus": "Testia"}, "species": {k: {"name": f"Testia {k}"} for k in SERIES},
           "structures": {}}


def _refs(wild, **kw):
    out = []
    for w in ((), wild):
        with tempfile.TemporaryDirectory() as td:
            write_matrix(Path(td), wild=w)
            out.append(nv.Reference(Path(td), PROFILE, **kw))
    return out


def test_a_withheld_specimen_cannot_move_its_own_score():
    """False-alarm arm: the specimen's own record is replaced by nonsense in the reference;
    scored with that record withheld, nothing may change — not the distances, not the spread
    they are divided by, not the weights."""
    sid = "spA_1"
    clean, dirty = _refs((sid,))
    for name in ("size", "ratio"):
        cand = clean.tables[name].loc[sid]
        a = nv.score_candidate(clean, name, cand, exclude_specimens=(sid,))
        b = nv.score_candidate(dirty, name, cand, exclude_specimens=(sid,))
        assert a["nearest"] == b["nearest"], (name, a["nearest"], b["nearest"])
        assert abs(a["g"] - b["g"]) < 1e-12, (name, a["g"], b["g"])


def test_that_test_has_teeth_the_legacy_scale_does_move():
    """The same perturbation DOES move the score when the pooled spread is computed once over
    everything (the behaviour before 2026-09-20), so the test above is not vacuous."""
    sid = "spA_1"
    clean, dirty = _refs((sid,), holdout_scale=False)
    moved = False
    for name in ("size", "ratio"):
        cand = clean.tables[name].loc[sid]
        a = nv.score_candidate(clean, name, cand, exclude_specimens=(sid,))
        b = nv.score_candidate(dirty, name, cand, exclude_specimens=(sid,))
        moved |= abs(a["g"] - b["g"]) > 1e-9
    assert moved


def test_a_withheld_species_cannot_move_its_members_scores():
    """Detection arm: every specimen of the withheld species is replaced by nonsense."""
    sp = "spC"
    ids = tuple(f"{sp}_{i}" for i in range(1, SERIES[sp] + 1))
    clean, dirty = _refs(ids)
    for name in ("size", "ratio"):
        cand = clean.tables[name].loc[ids[0]]
        a = nv.score_candidate(clean, name, cand, exclude_species=(sp,), exclude_specimens=(ids[0],))
        b = nv.score_candidate(dirty, name, cand, exclude_species=(sp,), exclude_specimens=(ids[0],))
        assert a["nearest"] == b["nearest"] and abs(a["g"] - b["g"]) < 1e-12, (name, a, b)


def test_the_full_ranking_is_held_out_too():
    """biorag_congruence_compare_v1.score_full feeds identification; same invariant."""
    sid = "spB_2"
    clean, dirty = _refs((sid,))
    cand = clean.tables["ratio"].loc[sid]
    ga, da = cc.score_full(clean, "ratio", cand, (), (sid,))
    gb, db = cc.score_full(dirty, "ratio", cand, (), (sid,))
    assert abs(ga - gb) < 1e-12 and all(abs(da[k] - db[k]) < 1e-12 for k in da)


def test_key_ranges_are_rebuilt_without_the_withheld_specimen():
    with tempfile.TemporaryDirectory() as td:
        write_matrix(Path(td), wild=("spB_3",))
        long = pd.read_csv(Path(td) / "specimen_matrix_long.csv")
        fd = pd.read_csv(Path(td) / "feature_dictionary.tsv", sep="\t")
    M = kb.Matrix(long, fd, PROFILE)
    assert M.obs["tibia.length_mm"]["spB"].max() > 30
    M.rebuild(exclude_specimen="spB_3")
    assert M.obs["tibia.length_mm"]["spB"].max() < 5 and M.n_spec["spB"] == SERIES["spB"] - 1


def test_graph_ranges_leave_out_the_withheld_specimen():
    with tempfile.TemporaryDirectory() as td:
        write_matrix(Path(td), wild=("spB_3",))
        ref = nv.Reference(Path(td), PROFILE)
    graph = {"spB": {"tibia.length_mm": (0.0, 1.0)}, "spC": {"tibia.length_mm": (0.0, 1.0)}}
    full = gi.regraph_without(graph, ref, None, "spB")               # nobody withheld
    held = gi.regraph_without(graph, ref, "spB_3", "spB")
    assert full["spB"]["tibia.length_mm"][1] > 30 > held["spB"]["tibia.length_mm"][1]
    assert held["spC"] == graph["spC"]                               # other species untouched


def test_adding_a_set_does_not_change_the_thresholds_tried_for_the_others():
    """The 0.483 -> 0.552 'gain' came from a threshold grid that depended on which sets were
    in the pot. The exact sweep over the original sets must not notice an extra column."""
    rng = np.random.default_rng(0)
    sp = np.repeat([f"s{i}" for i in range(6)], 4)
    det = pd.DataFrame({"species": sp, "a": rng.gamma(2, 1, 24), "b": rng.gamma(2, 1, 24)})
    fal = pd.DataFrame({"species": sp, "a": rng.gamma(1, 1, 24), "b": rng.gamma(1, 1, 24)})
    s1, _ = cc.sweep(det, fal, ["a", "b"], 2)
    s2, _ = cc.sweep(det.assign(extra=rng.gamma(9, 3, 24)), fal.assign(extra=rng.gamma(9, 3, 24)),
                     ["a", "b"], 2)
    assert s1[["threshold", "rule", "caught", "false_alarms"]].equals(
        s2[["threshold", "rule", "caught", "false_alarms"]])


def test_nested_choice_cannot_see_the_species_it_is_applied_to():
    """Operating point P is perfect for species j and useless for everyone else; Q is good
    for everyone else and misses j. Chosen without j, the answer is Q, so j is missed."""
    species = [f"s{i}" for i in range(6)]
    j = "s0"
    P = (pd.Series({s: s == j for s in species}), pd.Series({s: s != j for s in species}))
    Q = (pd.Series({s: s != j for s in species}), pd.Series({s: False for s in species}))
    ne = cc.nested(None, {(0.1, "P"): P, (0.2, "Q"): Q}, species)
    assert ne["caught"] == 5 and ne["false_alarms"] == 0, ne


def test_a_species_an_instrument_cannot_score_counts_as_missed():
    species = ["s0", "s1", "s2", "s3"]
    det = pd.Series({"s0": True, "s1": True})                        # s2, s3 never reached a name
    fal = pd.Series({s: False for s in species})
    ne = cc.nested(None, {(0.0, "r"): (det, fal)}, species)
    assert ne["of"] == 4 and ne["caught"] == 2


def test_a_state_is_graded_by_the_length_of_the_series_it_was_seen_in():
    q = [kb.binary_quality([np.ones(n), np.zeros(n)]) for n in (2, 3, 6, 12, 60)]
    assert q[0] == 0 and q[1] == 0 and q[2] < q[3] < q[4] < 1
    # the graded key doubts a state seen three times more than one seen thirty times
    thin = fz.binary_membership(1.0, {"A_range": [0, 0], "B_range": [1, 1], "A_n": 3, "B_n": 3})
    long_ = fz.binary_membership(1.0, {"A_range": [0, 0], "B_range": [1, 1], "A_n": 30, "B_n": 30})
    assert thin[1] == long_[1] == "B" and thin[0] < long_[0] and thin[2] < long_[2] < 1


def test_the_builder_does_not_prefer_a_thinly_supported_state_to_a_clear_measurement():
    with tempfile.TemporaryDirectory() as td:
        write_matrix(Path(td))
        long = pd.read_csv(Path(td) / "specimen_matrix_long.csv")
        fd = pd.read_csv(Path(td) / "feature_dictionary.tsv", sep="\t")
    state = long[long.feature_id == "tibia.length_mm"].copy()
    state["feature_id"], state["column"], state["family"] = "tibia.st_x", "st_state", "presence_absence"
    state["value"] = state["species"].isin(["spA", "spB"]).astype(float)
    row = fd[fd.feature_id == "tibia.length_mm"].iloc[0].to_dict()
    row.update(feature_id="tibia.st_x", column="st_state", family="presence_absence", unit="", label="x")
    M = kb.Matrix(pd.concat([long, state]), pd.concat([fd, pd.DataFrame([row])]), PROFILE)
    assert "tibia.st_x" in M.binary and "tibia.length_mm" not in M.binary
    B = kb.KeyBuilder(M, PROFILE)
    S = list(SERIES)
    s_state = B.best_split("tibia.st_x", S)
    assert s_state is not None and s_state.perfect
    # spA has two specimens: the state is 'fixed' in it on the evidence of two insects
    assert s_state.score < max(B.best_split(f, S).score for f, _, _ in FEATS if B.best_split(f, S))


def test_characters_added_with_the_wrong_tier_are_reported_as_invisible():
    """The void 'VLM characters do not help the key' result: they had been appended with a tier
    the builder does not read. The helper must say how many features the builder will see."""
    script = HERE.parents[1] / "biorag_add_discrete_characters_v1.py"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_matrix(td)
        st = pd.DataFrame([{"specimen_id": f"{sp}_{i}", "character": "tibia:knob",
                            "state": "present" if sp in ("spA", "spB") else "absent"}
                           for sp, n in SERIES.items() for i in range(1, n + 1)])
        st.to_csv(td / "states.tsv", sep="\t", index=False)
        seen = {}
        for tier in ("key", "description"):
            r = subprocess.run([sys.executable, str(script), "--matrix_dir", str(td), "--states",
                                str(td / "states.tsv"), "--out_dir", str(td / tier), "--tier", tier],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-400:]
            seen[tier] = json.loads((td / tier / "discrete_characters_report.json").read_text())[
                "features_the_key_builder_reads"]
            fd = pd.read_csv(td / tier / "feature_dictionary.tsv", sep="\t")
            long = pd.read_csv(td / tier / "specimen_matrix_long.csv")
            n_read = len(kb.Matrix(long, fd, PROFILE).features)
            assert n_read == seen[tier]["after"], (tier, n_read, seen[tier])
    assert seen["key"]["after"] == seen["key"]["before"] + 1
    assert seen["description"]["after"] == seen["description"]["before"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    print(f"{'ALL PASSED' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)
