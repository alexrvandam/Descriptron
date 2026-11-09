"""vein_pilosity_profile_v1 and vein_macrotrichia_detect_v1 without DINOv3: the command lines, and the vein centre line
(longest skeleton path, arc length) both programs use to place chunks along a vein. The scripts run top to bottom, so the
centre-line function is taken out of the source with ast and run on synthetic vein masks. CPU only, no weights."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
PILOSITY = HERE / "vein_pilosity_profile_v1.py"
MACRO = HERE / "vein_macrotrichia_detect_v1.py"


def _env():
    return {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


def _funcs(path, names):
    """the named top-level functions of a script, compiled without running the script"""
    pytest.importorskip("skimage")
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from skimage.morphology import skeletonize
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), f"{path.name}: {names} not found"
    ns = {"np": np, "coo_matrix": coo_matrix, "connected_components": connected_components,
          "dijkstra": dijkstra, "skeletonize": skeletonize}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


@pytest.mark.parametrize("script", [PILOSITY, MACRO], ids=lambda p: p.stem)
def test_help_and_required_args(script):
    pytest.importorskip("torch")
    r = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, env=_env(), timeout=300)
    assert r.returncode == 0, r.stderr
    for opt in ("--images", "--coco", "--out", "--chunk_px", "--kmax", "--base_part"):
        assert opt in r.stdout
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=_env(), timeout=300)
    assert r.returncode == 2 and "required" in r.stderr


def _bar(h=60, w=160, x0=10, x1=130, y=30, half=3):
    m = np.zeros((h, w), np.uint8); m[y - half:y + half + 1, x0:x1 + 1] = 1
    return m


def test_centreline_straight_vein():
    cl = _funcs(PILOSITY, ["centreline"])["centreline"]
    P, s = cl(_bar())
    assert P.shape[1] == 2 and len(P) == len(s)
    assert s[0] == 0 and np.all(np.diff(s) > 0)                      # arc length increases along the path
    assert 105 <= s[-1] <= 121                                        # ~ bar length (skeleton loses the rounded ends)
    assert np.all(np.abs(P[:, 1] - 30) <= 1)                          # on the midline
    xs = P[:, 0]
    assert np.all(np.diff(xs) > 0) or np.all(np.diff(xs) < 0)         # one end to the other, in order


def test_centreline_bent_vein_and_largest_piece():
    cl = _funcs(PILOSITY, ["centreline"])["centreline"]
    m = np.zeros((120, 120), np.uint8)
    m[17:24, 10:91] = 1          # horizontal arm, 80 px
    m[17:101, 84:91] = 1         # vertical arm, 80 px
    m[110:113, 5:15] = 1         # a small separate speck: ignored (largest skeleton component only)
    P, s = cl(m)
    assert 140 <= s[-1] <= 165
    assert P[:, 1].max() < 105


def test_centreline_empty_mask():
    cl = _funcs(PILOSITY, ["centreline"])["centreline"]
    assert cl(np.zeros((20, 20), np.uint8)) == (None, None)


def test_macrotrichia_uses_the_same_centreline():
    a = _funcs(PILOSITY, ["centreline"])["centreline"]
    b = _funcs(MACRO, ["centreline"])["centreline"]
    m = _bar(); m[30:55, 125:131] = 1                                 # bar with a hook at the apex
    (Pa, sa), (Pb, sb) = a(m), b(m)
    assert np.array_equal(Pa, Pb) and np.allclose(sa, sb)
