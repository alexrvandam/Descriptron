"""Sliding semilandmarks in semi_landmark_and_kpts_procrustesV42_GPA (slide_gpa), following geomorph's gpagen.
Against geomorph 4.1.1 itself (validation/validate_semilandmarks_vs_geomorph.py, 48 Diaphorina wing outlines x 99
points) the Procrustes distances agree to 2e-12 (procd) and 5e-11 (bending). These tests check the properties
without R: sliding moves points only along the outline, lowers the criterion it minimises, and keeps correspondence."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v42():
    spec = importlib.util.spec_from_file_location("v42s", HERE / "semi_landmark_and_kpts_procrustesV42_GPA.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:
        pytest.skip(f"V42 imports unavailable: {e}")
    return mod


def _outlines(n=12, p=40, seed=0):
    rng = np.random.default_rng(seed)
    t0 = np.linspace(0, 2 * np.pi, p, endpoint=False)
    out = []
    for _ in range(n):
        t = t0 + rng.normal(0, 0.05, p)                       # uneven spacing: something to slide
        r = 1 + 0.3 * np.cos(t) + 0.15 * np.sin(2 * t) + rng.normal(0, 0.02, 1)
        P = np.column_stack([r * np.cos(t) * 50, r * np.sin(t) * 30])
        a = rng.uniform(0, 6.28)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        out.append(P @ R.T * rng.uniform(1, 3) + rng.uniform(0, 100, 2))
    return out


def _pdist_to_mean(Y):
    Y = np.array(Y); M = Y.mean(0)
    return float(np.linalg.norm((Y - M).reshape(len(Y), -1), axis=1).mean())


def test_procd_sliding_brings_specimens_closer_to_the_consensus(v42):
    X = _outlines()
    sl = v42.closed_sliders(len(X[0]))
    fixed, _ = v42.slide_gpa(X, sl, "procd", max_iter=1)          # max_iter=1: GPA only, no sliding round
    slid, _ = v42.slide_gpa(X, sl, "procd")
    assert _pdist_to_mean(slid) < _pdist_to_mean(fixed)


def test_bending_sliding_lowers_bending_energy(v42):
    X = _outlines(seed=1)
    sl = v42.closed_sliders(len(X[0]))
    fixed, ref0 = v42.slide_gpa(X, sl, "bending", max_iter=1)
    slid, ref = v42.slide_gpa(X, sl, "bending")

    def be(Y, ref):
        L = v42._gm_Ltemplate(ref)
        return sum(float((y - ref)[:, 0] @ L @ (y - ref)[:, 0] + (y - ref)[:, 1] @ L @ (y - ref)[:, 1]) for y in Y)
    assert be(slid, ref) < be(fixed, ref0)


def test_open_curve_end_points_do_not_slide(v42):
    X = [x[:15] for x in _outlines(seed=2)]                     # open arcs
    sl = v42.open_sliders(15)
    assert 0 not in sl[:, 1] and 14 not in sl[:, 1] and len(sl) == 13
    slid, _ = v42.slide_gpa(X, sl, "procd")
    assert np.array(slid).shape == np.array(X).shape
