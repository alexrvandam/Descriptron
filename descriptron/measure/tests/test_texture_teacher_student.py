"""texture_teacher_student_v1 without DINOv3: the command line, and the pure image helpers (region decoding from polygons
and RLE, token coverage, per-channel resize, guided filter). The script runs top to bottom, so the helpers are taken out of
the source with ast and run on synthetic arrays. CPU only, no weights."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "texture_teacher_student_v1.py"
cv2 = pytest.importorskip("cv2")


def _funcs(names):
    tree = ast.parse(SCRIPT.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"), ns)
    return ns


def test_help_and_required_args():
    pytest.importorskip("torch")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    for opt in ("--images", "--coco", "--region", "--labels", "--size_mode", "--lo", "--hi", "--teacher_files", "--out"):
        assert opt in r.stdout
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 2 and "required" in r.stderr


def test_dec_polygon_and_rle_agree():
    mu = pytest.importorskip("pycocotools.mask")
    dec = _funcs(["dec"])["dec"]
    hw = (40, 50)
    poly = {"segmentation": [[5, 5, 30, 5, 30, 25, 5, 25]]}
    m = dec(poly, hw)
    assert m.dtype == bool and m.shape == hw and 400 <= m.sum() <= 600
    rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode()                       # COCO JSON stores counts as text
    assert np.array_equal(dec({"segmentation": rle}, hw), m)


def test_cover_and_rz():
    ns = _funcs(["cover", "rz"])
    m = np.zeros((64, 64), bool); m[:, :32] = True
    c = ns["cover"](m, 4, 4)
    assert c.shape == (4, 4) and np.allclose(c[:, :2], 1) and np.allclose(c[:, 2:], 0)
    a = np.random.default_rng(0).random((4, 5, 3)).astype(np.float32)
    r = ns["rz"](a, (10, 8), cv2.INTER_NEAREST)
    assert r.shape == (8, 10, 3)


def test_guided_filter_keeps_constants_and_edges():
    gf = _funcs(["gfilter"])["gfilter"]
    I = np.zeros((64, 64), np.float32); I[:, 32:] = 1                # grey image with one edge
    p = np.full((64, 64, 2), 0.3, np.float32)
    assert np.allclose(gf(I, p), 0.3, atol=1e-5)                       # a constant map stays constant
    q = np.zeros((64, 64, 1), np.float32); q[:, 32:] = 1               # map edge on the image edge survives
    out = gf(I, q, r=4)[..., 0]
    assert out[:, :28].max() < 0.05 and out[:, 36:].min() > 0.95
