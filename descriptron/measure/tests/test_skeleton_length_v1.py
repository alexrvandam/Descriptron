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
# 2.8.5: only the cv2.ximgproc tests need opencv-contrib; the scikit-image fallback (pip installs) is tested without it
needs_ximgproc = pytest.mark.skipif(not hasattr(cv2, "ximgproc"), reason="cv2.ximgproc (opencv-contrib) not available")

V35 = Path(__file__).resolve().parents[1] / "measurement_script_to_try_after_kpts_prediction_measure_kpts_V35.py"
NAMES = ("preprocess_mask_for_skeleton", "measure_length_skeleton_path", "skeletonize_skimage", "thin_skimage")


@pytest.fixture(scope="module")
def v35():
    tree = ast.parse(V35.read_text(encoding="utf-8"))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    assert {f.name for f in funcs} == set(NAMES), "V35 lost one of the skeleton-length functions"
    from skimage.morphology import skeletonize
    ns = {"np": np, "cv2": cv2, "skeletonize": skeletonize}
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


@needs_ximgproc
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



def _skeleton_skimage(ns, mask):
    """2.8.5, as V35 does it without cv2.ximgproc: skimage skeletonize (Zhang-Suen) and thin (Guo-Hall type), keep the
    one with more pixels."""
    pre = ns["preprocess_mask_for_skeleton"](mask)
    sk_z, sk_g = ns["skeletonize_skimage"](pre), ns["thin_skimage"](pre)
    return sk_z if np.count_nonzero(sk_z) > np.count_nonzero(sk_g) else sk_g


def _via_coco_polygon(mask):
    """The mask as V35 gets it: outline -> COCO polygon -> pycocotools raster (line_drawing / GUI path)."""
    mu = pytest.importorskip("pycocotools.mask")
    cs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    seg = [c.reshape(-1).astype(float).tolist() for c in cs if len(c) >= 3]
    h, w = mask.shape
    return (mu.decode(mu.merge(mu.frPyObjects(seg, h, w))) > 0).astype(np.uint8) * 255


def _diagonal_offset(w):
    """The 45-degree line of the end-to-end test (Zhang-Suen alone deletes its polygon raster at 2 and 5 px)."""
    m = np.zeros((340, 500), np.uint8)
    cv2.line(m, (60, 20), (340, 300), 255, w)
    return m, 280 * math.sqrt(2)


SHAPES = [_horizontal, _diagonal, _diagonal_offset, _arc]
SHAPE_IDS = ["horizontal", "45deg", "45deg_offset", "arc"]


@pytest.mark.parametrize("coco", [False, True], ids=["drawn", "coco_polygon"])
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("width", [2, 3, 5, 8])
def test_length_within_2_percent_skimage_fallback(v35, shape, width, coco):
    """2.8.5: without cv2.ximgproc (pip: opencv-python-headless) V35 thins with scikit-image."""
    mask, true_len = shape(width)
    if coco:
        mask = _via_coco_polygon(mask)
    sk = _skeleton_skimage(v35, mask)
    assert np.any(sk), "skeleton is empty"
    length = v35["measure_length_skeleton_path"](sk)
    assert abs(length - true_len) / true_len < 0.02, f"{length:.1f} vs true {true_len:.1f}"


@needs_ximgproc
@pytest.mark.parametrize("shape", [_diagonal_offset, _arc], ids=["45deg_offset", "arc"])
@pytest.mark.parametrize("width", [2, 3, 5, 8])
def test_length_within_2_percent_coco_polygon(v35, shape, width):
    mask, true_len = shape(width)
    sk = _skeleton(v35, _via_coco_polygon(mask))
    length = v35["measure_length_skeleton_path"](sk)
    assert abs(length - true_len) / true_len < 0.02, f"{length:.1f} vs true {true_len:.1f}"


def test_v35_falls_back_to_skimage_and_imports_easyocr_lazily():
    """2.8.5: the AttributeError branch of --method skeleton thins with scikit-image (it used to `continue`), and
    a missing easyocr does not stop V35 at import."""
    src = V35.read_text(encoding="utf-8")
    assert "sk_z = skeletonize_skimage(pre)" in src and "sk_g = thin_skimage(pre)" in src
    assert "\nimport easyocr\n" not in src and "    import easyocr\nexcept ImportError:\n    easyocr = None" in src
