#!/usr/bin/env python3
"""
Outline shape and the other added character sets: registration and the hold-out invariant.

Two properties have to hold before a number from these sets means anything.

REGISTRATION. A closed outline has no origin and no handedness of its own: the same
structure photographed from the other side, or with the slide turned, is the same shape
traversed from a different point, or mirrored. Correspondence — which point of one
outline answers to which point of another — has to be fixed before shapes can be
compared, exactly as a morphometrician fixes it by digitising landmarks in an agreed
order. The legacy GPA step cannot do it (its reflection negates the coordinates without
reversing the point order, so a mirrored outline never lands on its original), which is
why the outline had to be built again.

HOLD-OUT. The consensus an outline is aligned to is a statistic of the sample, so it is
something the specimen under test is scored against. Leaving a specimen out of the
distances while leaving it inside the consensus those distances are measured in is not a
hold-out. The tests below state it the same way the rest of the suite does:

    changing the data of whatever is withheld must not change its score.

Run:  python measure/tests/test_biorag_shape_sets_v1.py
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
import biorag_novelty_score_v1 as nv                                      # noqa: E402
import biorag_congruence_compare_v1 as cc                                 # noqa: E402
import biorag_outline_shape_v1 as osh                                     # noqa: E402
import test_biorag_holdout_v1 as ho                                       # noqa: E402

N_POINTS = 64


# ─────────────────────────────────────────────────────────────────────────────
# synthetic outlines
# ─────────────────────────────────────────────────────────────────────────────
def blob(seed=0, noise=0.0, n_vertices=240, scale=100.0, offset=(500.0, 300.0)):
    """A closed outline with no mirror symmetry and no rotational symmetry: three
    harmonics with phases that no reflection can bring back onto themselves."""
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    r = 1 + 0.35 * np.cos(th) + 0.25 * np.sin(2 * th) + 0.12 * np.sin(3 * th + 0.7)
    p = np.column_stack([r * np.cos(th), r * np.sin(th)]) * scale
    if noise:
        p = p + rng.normal(0, noise, p.shape)
    return p + np.asarray(offset)


def pipeline(raw):
    """The script's own path from a polygon to a normalised outline."""
    return osh.normalise_shape(osh.resample_closed(raw, N_POINTS))


def turn(p, deg):
    t = math.radians(deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    c = p.mean(axis=0)
    return (p - c) @ R.T + c


def procrustes_gap(a, b):
    """Rotation-only Procrustes distance between two normalised outlines,
    correspondence as given (no shift search, no mirroring)."""
    return osh.align_to(a, b)[1]


# ─────────────────────────────────────────────────────────────────────────────
# (i) a mirrored outline is recognised and lands on its original
# ─────────────────────────────────────────────────────────────────────────────
def test_a_mirrored_outline_registers_as_mirrored_and_then_coincides():
    raws = [blob(seed=i, noise=0.25) for i in range(5)]
    mirrored_raw = blob(seed=0, noise=0.25) * np.array([-1.0, 1.0])        # the slide turned over
    ids = [f"s{i}" for i in range(5)] + ["mirror_of_s0"]
    S = np.stack([pipeline(r) for r in raws] + [pipeline(mirrored_raw)])
    R = osh.register_structure(ids, S)
    flags = R["mirrored"]
    assert flags[5] != flags[:5].mean().round(), \
        f"the mirrored copy was not called the odd one out: {flags}"
    assert (flags[:5] == flags[0]).all(), f"the five unmirrored copies disagree: {flags}"
    d = procrustes_gap(R["shapes"][5], R["shapes"][0])
    assert d < 0.02, f"after registration the mirrored copy does not coincide (d={d:.4f})"
    # and the registration is confident about it, not a coin toss
    assert not R["ambiguous"][5], "an outline with no mirror symmetry was called ambiguous"


def test_the_legacy_reflection_cannot_do_it():
    """Negating x WITHOUT reversing the point order leaves the outline running backwards,
    and no rotation brings it back — which is why the legacy shape_PC* columns mix real
    shape with which way round the slide was mounted. This test has teeth: it fails if
    mirror_shape is ever 'simplified' to a sign flip."""
    a = pipeline(blob(seed=1))
    good = osh.mirror_shape(a)
    bad = a * np.array([-1.0, 1.0])
    IDX = osh._roll_index(N_POINTS)
    _, d_good = osh._best_over_shifts(good[None, :, :], osh.mirror_shape(a), IDX)
    _, d_bad = osh._best_over_shifts(bad[None, :, :], osh.mirror_shape(a), IDX)
    assert d_good[0] < 1e-6 < 0.1 < d_bad[0], (d_good[0], d_bad[0])
    assert osh.signed_area(good) > 0 > osh.signed_area(bad)


# ─────────────────────────────────────────────────────────────────────────────
# (ii) a turned outline started somewhere else registers onto the original
# ─────────────────────────────────────────────────────────────────────────────
def test_a_turned_outline_started_elsewhere_registers_onto_the_original():
    raws = [blob(seed=i, noise=0.25) for i in range(5)]
    moved = np.roll(turn(blob(seed=0, noise=0.25), 180.0), 97, axis=0)     # half a turn, new start
    ids = [f"s{i}" for i in range(5)] + ["turned_s0"]
    S = np.stack([pipeline(r) for r in raws] + [pipeline(moved)])
    R = osh.register_structure(ids, S)
    assert R["mirrored"][5] == R["mirrored"][0], "a rotation was mistaken for a reflection"
    d = procrustes_gap(R["shapes"][5], R["shapes"][0])
    assert d < 0.02, f"the turned copy did not register onto the original (d={d:.4f})"


def test_a_symmetric_outline_is_reported_as_ambiguous():
    """An ellipse is its own mirror image, so which way round it was laid down cannot be
    recovered and the call must be FLAGGED, not made silently. The chiral blob, measured
    the same way, is never flagged. This contrast is what the `ambiguous` fraction sorts
    structures by, and why the forewing membrane and the head are dropped from the
    'rarely ambiguous' set while the wing cells are kept."""
    th = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    ids, sym = [], []
    for i in range(8):
        rng = np.random.default_rng(i)
        a = 3 + 0.3 * rng.normal()                       # specimens differ in proportion
        p = np.column_stack([a * np.cos(th), np.sin(th)]) * 100.0 + rng.normal(0, 0.25, (240, 2))
        ids.append(f"e{i}")
        sym.append(pipeline(p + np.array([400.0, 400.0])))
    Rs = osh.register_structure(ids, np.stack(sym))
    Rc = osh.register_structure([f"b{i}" for i in range(8)],
                                np.stack([pipeline(blob(seed=i, noise=0.25)) for i in range(8)]))
    assert Rs["ambiguous"].mean() > 0.5, f"a symmetric outline was called confidently: {Rs['ambiguous']}"
    assert Rc["ambiguous"].mean() == 0.0, f"a chiral outline was called ambiguous: {Rc['ambiguous']}"


# ─────────────────────────────────────────────────────────────────────────────
# (iii) the hold-out invariant for outline shape
# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_reg(structures=("alpha", "beta")):
    """One outline per specimen of the four synthetic species, species-specific shape."""
    ids = [f"{sp}_{i}" for sp, n in ho.SERIES.items() for i in range(1, n + 1)]
    reg = {}
    for k, st in enumerate(structures):
        S = []
        for j, sid in enumerate(ids):
            sp = sid.split("_")[0]
            seed = 100 * k + 7 * list(ho.SERIES).index(sp) + j
            p = pipeline(blob(seed=seed, noise=0.4,
                              scale=100.0 + 9.0 * list(ho.SERIES).index(sp)))
            S.append(p)
        S = np.stack(S)
        reg[st] = {"ids": list(ids), "shapes": S,
                   "consensus": S[0], "mirrored": np.zeros(len(ids), bool),
                   "shift": np.zeros(len(ids), int), "ambiguous": np.zeros(len(ids), bool),
                   "d_as_is": np.zeros(len(ids)), "d_mirrored": np.ones(len(ids))}
    return reg, ids


def _spoil(reg, victims, seed=5):
    rng = np.random.default_rng(seed)
    out = {}
    for st, R in reg.items():
        S = R["shapes"].copy()
        for i, sid in enumerate(R["ids"]):
            if sid in victims:
                S[i] = rng.normal(0, 1, S[i].shape)
        out[st] = {**R, "shapes": S}
    return out


def _ref_with_shape(tmp: Path, table: pd.DataFrame, name="shape", **kw):
    ho.write_matrix(tmp)
    ref = nv.Reference(tmp, ho.PROFILE, **kw)
    cc.register_continuous(ref, name, table)
    return ref


def test_the_fold_frame_is_built_from_the_training_outlines_only():
    reg, ids = _synthetic_reg()
    victim = "spA_1"
    clean = osh.FoldAligner(reg).frame([victim])
    dirty = osh.FoldAligner(_spoil(reg, {victim})).frame([victim])
    keep = [i for i in clean.index if i != victim]
    assert clean.loc[keep].equals(dirty.loc[keep]), \
        "replacing the withheld outline moved the other specimens' coordinates"
    # teeth: with the withheld specimen inside the consensus, it does move everyone
    full_clean = osh.FoldAligner(reg).frame([])
    full_dirty = osh.FoldAligner(_spoil(reg, {victim})).frame([])
    assert not np.allclose(full_clean.loc[keep].values, full_dirty.loc[keep].values), \
        "the full-sample frame did not move — the test cannot detect a leak"


def test_a_withheld_specimen_cannot_move_its_own_outline_score():
    """False-alarm arm. The candidate's own coordinates are taken from the clean frame;
    the reference they are scored against is the same fold frame built from a sample in
    which that specimen's outline is nonsense. Nothing may change."""
    reg, ids = _synthetic_reg()
    sid = "spB_2"
    fc = osh.FoldAligner(reg).frame([sid])
    fd = osh.FoldAligner(_spoil(reg, {sid})).frame([sid])
    with tempfile.TemporaryDirectory() as td:
        clean = _ref_with_shape(Path(td), fc)
    with tempfile.TemporaryDirectory() as td:
        dirty = _ref_with_shape(Path(td), fd)
    cand = clean.tables["shape"].loc[sid]
    ga, da = cc.score_full(clean, "shape", cand, (), (sid,))
    gb, db = cc.score_full(dirty, "shape", cand, (), (sid,))
    assert ga == ga and abs(ga - gb) < 1e-12, (ga, gb)
    assert all(abs(da[k] - db[k]) < 1e-12 for k in da)


def test_a_withheld_species_cannot_move_its_members_outline_scores():
    """Detection arm: every outline of the withheld species is replaced by nonsense."""
    reg, ids = _synthetic_reg()
    sp = "spC"
    members = [i for i in ids if i.startswith(sp + "_")]
    fc = osh.FoldAligner(reg).frame(members)
    fd = osh.FoldAligner(_spoil(reg, set(members))).frame(members)
    with tempfile.TemporaryDirectory() as td:
        clean = _ref_with_shape(Path(td), fc)
    with tempfile.TemporaryDirectory() as td:
        dirty = _ref_with_shape(Path(td), fd)
    for sid in members:
        cand = clean.tables["shape"].loc[sid]
        ga, _ = cc.score_full(clean, "shape", cand, (sp,), (sid,))
        gb, _ = cc.score_full(dirty, "shape", cand, (sp,), (sid,))
        assert ga == ga and abs(ga - gb) < 1e-12, (sid, ga, gb)


def test_the_full_sample_frame_would_have_leaked():
    """The same score computed on a frame aligned to the FULL-SAMPLE consensus does move
    when the withheld specimen's outline is replaced, so the test above is not vacuous."""
    reg, ids = _synthetic_reg()
    sid = "spB_2"
    fc = osh.FoldAligner(reg).frame([])
    fd = osh.FoldAligner(_spoil(reg, {sid})).frame([])
    with tempfile.TemporaryDirectory() as td:
        clean = _ref_with_shape(Path(td), fc)
    with tempfile.TemporaryDirectory() as td:
        dirty = _ref_with_shape(Path(td), fd)
    cand = clean.tables["shape"].loc[sid]
    ga, _ = cc.score_full(clean, "shape", cand, (), (sid,))
    gb, _ = cc.score_full(dirty, "shape", cand, (), (sid,))
    assert abs(ga - gb) > 1e-9, (ga, gb)


def test_the_fold_set_hands_the_scorer_a_different_table_per_fold():
    """FoldSet must register one table per exclusion set and unregister the old ones —
    a stale cached scale would silently score one fold with another fold's spread."""
    reg, ids = _synthetic_reg()
    aligner = osh.FoldAligner(reg)
    folds = osh.FoldTables(aligner, max_cached=2)
    with tempfile.TemporaryDirectory() as td:
        ho.write_matrix(Path(td))
        ref = nv.Reference(Path(td), ho.PROFILE)
    fs = cc.FoldSet(ref, "shape", folds, aligner.columns, max_registered=2)
    t1, t2, t3 = fs.table(["spA_1"]), fs.table(["spB_1"]), fs.table(["spC_1"])
    assert len({t1, t2, t3}) == 3
    assert t1 not in ref.tables and t2 in ref.tables and t3 in ref.tables
    assert not [k for k in ref._scache if k[0] == t1], "a stale scale survived eviction"
    assert fs.table(["spB_1"]) == t2 and ref.tables[t2].shape[1] == len(aligner.columns)
    assert not ref.tables[t2].equals(ref.tables[t3])


# ─────────────────────────────────────────────────────────────────────────────
# (iv) an added continuous set goes through the hold-out scale
# ─────────────────────────────────────────────────────────────────────────────
def _continuous_table(ids, wild=(), seed=11):
    rng = np.random.default_rng(seed)
    rows = {}
    for sid in ids:
        s = list(ho.SERIES).index(sid.split("_")[0])
        v = np.array([1.0 + 0.4 * ((s * (k + 3)) % 5) + rng.normal(0, 0.03) for k in range(9)])
        rows[sid] = np.full(9, 80.0) + np.arange(9) if sid in wild else v
    return pd.DataFrame.from_dict(rows, orient="index",
                                  columns=[f"struct.f{k}" for k in range(9)])


def test_an_added_continuous_set_is_scaled_without_the_withheld_specimen():
    ids = [f"{sp}_{i}" for sp, n in ho.SERIES.items() for i in range(1, n + 1)]
    sid = "spA_2"
    clean_t, dirty_t = _continuous_table(ids), _continuous_table(ids, wild=(sid,))
    with tempfile.TemporaryDirectory() as td:
        clean = _ref_with_shape(Path(td), clean_t, name="extra")
    with tempfile.TemporaryDirectory() as td:
        dirty = _ref_with_shape(Path(td), dirty_t, name="extra")
    assert "extra" not in getattr(clean, "unit_scale", set()), \
        "a measured set was registered on the unit scale used for present/absent characters"
    assert not clean.scales["extra"].equals(dirty.scales["extra"]), \
        "the perturbation did not even reach the scale — the test has no teeth"
    cand = clean.tables["extra"].loc[sid]
    ga, da = cc.score_full(clean, "extra", cand, (), (sid,))
    gb, db = cc.score_full(dirty, "extra", cand, (), (sid,))
    assert ga == ga and abs(ga - gb) < 1e-12, (ga, gb)
    assert all(abs(da[k] - db[k]) < 1e-12 for k in da)


def test_the_added_set_moves_when_the_hold_out_scale_is_switched_off():
    """The same perturbation DOES move the score with the single sample-wide scale, so the
    added set really is routed through scale_for and not past it."""
    ids = [f"{sp}_{i}" for sp, n in ho.SERIES.items() for i in range(1, n + 1)]
    sid = "spA_2"
    with tempfile.TemporaryDirectory() as td:
        clean = _ref_with_shape(Path(td), _continuous_table(ids), name="extra",
                                holdout_scale=False)
    with tempfile.TemporaryDirectory() as td:
        dirty = _ref_with_shape(Path(td), _continuous_table(ids, wild=(sid,)), name="extra",
                                holdout_scale=False)
    cand = clean.tables["extra"].loc[sid]
    ga, _ = cc.score_full(clean, "extra", cand, (), (sid,))
    gb, _ = cc.score_full(dirty, "extra", cand, (), (sid,))
    assert abs(ga - gb) > 1e-9, (ga, gb)


def test_the_specimen_hook_leaves_the_old_behaviour_alone():
    """score_arms without the new hook must score exactly what it scored before."""
    with tempfile.TemporaryDirectory() as td:
        ho.write_matrix(Path(td))
        ref = nv.Reference(Path(td), ho.PROFILE)
    a_det, a_fal, a_rk = cc.score_arms(ref, ["size", "ratio"])
    b_det, b_fal, b_rk = cc.score_arms(ref, ["size", "ratio"], None, lambda name, sid: name)
    assert a_det.equals(b_det) and a_fal.equals(b_fal) and a_rk.equals(b_rk)


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
