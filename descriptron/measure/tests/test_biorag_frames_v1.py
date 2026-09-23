#!/usr/bin/env python3
"""
Frame orientation: which homology frames lie end for end?

A Procrustes fit that sweeps the starting point of a closed outline cannot tell the two ends of
a nearly symmetric elongate structure apart, so some frames come out half a turn from the rest,
and a statistic meant to ask "does the model look in the same place each time?" measures the
frames instead. biorag_frame_orientation_v1.py decides from the width profile along the long
axis. These tests fix the two properties that matter: a clubbed outline turned half a turn is
found, and a symmetric outline is never confidently turned.

Run:  python measure/tests/test_biorag_frames_v1.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import biorag_frame_orientation_v1 as fo                                  # noqa: E402


def club(n=120, knob=0.55, noise=0.0, rng=None, turned=False, symmetric=False):
    """A closed outline: a shaft 10 long and 1 wide with a knob at x = 0 (a tibia, roughly).
    `symmetric` puts the same knob at both ends."""
    x = np.linspace(0, 10, n // 2)
    half = 0.5 + knob * np.exp(-(x / 1.2) ** 2)
    if symmetric:
        half = half + knob * np.exp(-((10 - x) / 1.2) ** 2)
    top = np.c_[x, half]
    bot = np.c_[x[::-1], -half[::-1]]
    pts = np.r_[top, bot]
    if rng is not None and noise:
        pts = pts + rng.normal(0, noise, pts.shape)
    if turned:
        pts = -pts                                                        # half a turn about the origin
    th = np.deg2rad(37.0)                                                 # frames are not axis-aligned
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return pts @ R.T + 384.0


def profiles(outlines, axes_from):
    u, v = fo.principal_axis(axes_from)
    out = {}
    for k, lm in outlines.items():
        w = fo.width_profile(lm, u, v, 64)
        out[k] = fo.unit(w)
    return out


def test_frames_half_a_turn_out_are_the_ones_found():
    rng = np.random.default_rng(4)
    turned = {f"s{i}" for i in (2, 5, 9, 11, 14, 17)}
    outl = {f"s{i}": club(noise=0.03, rng=rng, turned=f"s{i}" in turned) for i in range(20)}
    scores, _ = fo.orient_by_profile(profiles(outl, club()))
    found = {k for k, (as_is, rot) in scores.items() if rot > as_is}
    assert found == turned, (sorted(found), sorted(turned))
    margins = [abs(a - b) for a, b in scores.values()]
    assert min(margins) > 0.05                                            # and none of them a close call


def test_the_majority_orientation_is_the_reference():
    """With 14 one way and 6 the other, it is the 6 that are turned, never the 14."""
    rng = np.random.default_rng(1)
    outl = {f"s{i}": club(noise=0.03, rng=rng, turned=i < 6) for i in range(20)}
    scores, _ = fo.orient_by_profile(profiles(outl, club()))
    found = {k for k, (as_is, rot) in scores.items() if rot > as_is}
    assert found == {f"s{i}" for i in range(6)}


def test_a_symmetric_outline_is_never_confidently_turned():
    rng = np.random.default_rng(7)
    outl = {f"s{i}": club(noise=0.03, rng=rng, symmetric=True, turned=i % 3 == 0) for i in range(18)}
    scores, _ = fo.orient_by_profile(profiles(outl, club(symmetric=True)))
    assert max(abs(a - b) for a, b in scores.values()) < 0.05             # below --min_confidence


def test_asymmetry_measure_separates_the_two():
    assert fo.rotation_asymmetry(fo.ring(club())) > 3 * fo.rotation_asymmetry(fo.ring(club(symmetric=True)))
    assert fo.elongation_of(club()) > 4


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
