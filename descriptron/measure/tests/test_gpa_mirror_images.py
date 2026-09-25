"""Mirror-image specimens (photographed from the other side, or the opposite body side) must be reflected before GPA.
Regression tests for landmark_gpa_V2 and semi_landmark_and_kpts_procrustesV42_GPA (found by checking against geomorph:
V1 let 19 mirrored forewings dominate PC1; V34's with_reflection never matched a reflected outline)."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import landmark_gpa_V2 as lg  # noqa: E402

BASE = np.array([[0, 0], [40, 5], [80, 0], [100, 30], [80, 60], [40, 65], [0, 60], [-15, 30]], float)


def _rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def test_landmark_v2_finds_the_mirror_images():
    rng = np.random.default_rng(0)
    cfgs, truth = [], []
    for i in range(12):
        P = BASE + rng.normal(0, 1.0, BASE.shape)
        m = i in (2, 5, 9)
        if m:
            P = P * [-1, 1]
        cfgs.append((_rot(rng.uniform(0, 6.28)) @ P.T).T * rng.uniform(2, 4) + rng.uniform(0, 300, 2))
        truth.append(m)
    assert list(lg.mirrored_specimens(cfgs)) == truth


def test_landmark_v2_gpa_ends_on_principal_axes():
    rng = np.random.default_rng(1)
    cfgs = [(_rot(rng.uniform(0, 6.28)) @ (BASE + rng.normal(0, 1, BASE.shape)).T).T for _ in range(6)]
    _, mean, _ = lg.gpa(cfgs)
    c = np.cov((mean - mean.mean(0)).T)
    assert abs(c[0, 1]) < 1e-10 and c[0, 0] >= c[1, 1]


@pytest.fixture(scope="module")
def v42():
    spec = importlib.util.spec_from_file_location("v42", HERE / "semi_landmark_and_kpts_procrustesV42_GPA.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:            # the script's plotting / UMAP stack is optional for this test
        pytest.skip(f"V42 imports unavailable: {e}")
    return mod


def _outline(n=60):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + 0.35 * np.cos(t) + 0.2 * np.sin(2 * t) + 0.1 * np.cos(3 * t)      # clearly asymmetric
    P = np.column_stack([r * np.cos(t) * 60, r * np.sin(t) * 40])
    return np.vstack([P, P[0]])


def test_v42_with_reflection_matches_a_mirror_image_outline(v42):
    ref = _outline()
    pts = ref[:-1]
    # the mirror-image photo, traced in the same rotational sense as every other outline (as findContours does)
    mir = (pts * [-1, 1])[::-1]
    mir = np.roll(mir, 17, 0) @ _rot(0.7).T * 1.8 + [200, 50]
    mir = np.vstack([mir, mir[0]])
    z_no, _ = v42.align_closed(ref, mir, "without_reflection")
    z_yes, tf = v42.align_closed(ref, mir, "with_reflection")
    r0 = (ref - ref.mean(0)) / np.sqrt(((ref - ref.mean(0)) ** 2).sum())
    assert ((r0 - z_yes) ** 2).sum() < 1e-10 < ((r0 - z_no) ** 2).sum()
    assert tf["reflection"] in ("horizontal", "vertical")


def test_v42_reflect_closed_restores_the_outline(v42):
    ref = _outline()
    mir = np.vstack([(ref[:-1] * [-1, 1])[::-1], (ref[:-1] * [-1, 1])[::-1][0]])
    back = v42.reflect_closed(mir)
    z, _ = v42.align_closed(ref, back, "without_reflection")
    r0 = (ref - ref.mean(0)) / np.sqrt(((ref - ref.mean(0)) ** 2).sum())
    assert ((r0 - z) ** 2).sum() < 1e-10
