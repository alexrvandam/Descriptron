"""Skeleton lengths in measurement_script_to_try_after_kpts_prediction_measure_kpts_V35 (--method skeleton), 2.8.4.
Before 2.8.4 the length was the number of skeleton pixels (a 45-degree line came out 29% short) and the 3x3 opening in
preprocess_mask_for_skeleton erased or cut masks only a few pixels wide. These tests draw lines of known length with
OpenCV, run V35's preprocessing and thinning, and check measure_length_skeleton_path against the true length.
The two functions are taken out of V35 with ast (V35 imports easyocr, umap, ... at the top)."""
import ast
import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
if not hasattr(cv2, "ximgproc"):
    pytest.skip("cv2.ximgproc (opencv-contrib) not available", allow_module_level=True)

V35 = Path(__file__).resolve().parents[1] / "measurement_script_to_try_after_kpts_prediction_measure_kpts_V35.py"
NAMES = ("preprocess_mask_for_skeleton", "measure_length_skeleton_path")


@pytest.fixture(scope="module")
def v35():
    tree = ast.parse(V35.read_text(encoding="utf-8"))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    assert {f.name for f in funcs} == set(NAMES), "V35 lost one of the skeleton-length functions"
    ns = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(V35), "exec"), ns)
    return ns


def _skeleton(ns, mask):
    """As V35 does it: preprocess, thin with Zhang-Suen and Guo-Hall, keep the one with more pixels."""
    pre = ns["preprocess_mask_for_skeleton"](mask)
    sk_z = cv2.ximgproc.thinning(pre, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    sk_g = cv2.ximgproc.thinning(pre, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    return sk_z if np.count_nonzero(sk_z) > np.count_nonzero(sk_g) else sk_g


def _horizontal(w):
    m = np.zeros((200, 500), np.uint8)
    cv2.line(m, (100, 100), (400, 100), 255, w)
    return m, 300.0


def _diagonal(w):
    m = np.zeros((500, 500), np.uint8)
    cv2.line(m, (50, 50), (330, 330), 255, w)
    return m, 280 * math.sqrt(2)


def _arc(w, a=150, b=90, t0=10, t1=170):
    m = np.zeros((300, 400), np.uint8)
    cv2.ellipse(m, (200, 200), (a, b), 0, t0, t1, 255, w)
    t = np.radians(np.linspace(t0, t1, 20001))
    x, y = a * np.cos(t), b * np.sin(t)
    return m, float(np.sum(np.hypot(np.diff(x), np.diff(y))))


@pytest.mark.parametrize("shape", [_horizontal, _diagonal, _arc], ids=["horizontal", "45deg", "arc"])
@pytest.mark.parametrize("width", [2, 3, 5, 8])
def test_length_within_2_percent(v35, shape, width):
    mask, true_len = shape(width)
    sk = _skeleton(v35, mask)
    assert np.any(sk), "skeleton is empty"
    length = v35["measure_length_skeleton_path"](sk)
    assert abs(length - true_len) / true_len < 0.02, f"{length:.1f} vs true {true_len:.1f}"


def test_thin_curve_not_split(v35):
    mask, _ = _arc(3)
    pre = v35["preprocess_mask_for_skeleton"](mask)
    n_in = cv2.connectedComponents((mask > 0).astype(np.uint8))[0]
    n_out = cv2.connectedComponents((pre > 0).astype(np.uint8))[0]
    assert n_out == n_in == 2          # background + one piece
    assert np.count_nonzero(pre) >= 0.8 * np.count_nonzero(mask)


def test_empty_skeleton_is_zero(v35):
    assert v35["measure_length_skeleton_path"](np.zeros((50, 50), np.uint8)) == 0
