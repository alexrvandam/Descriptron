"""descriptron_check_cross_image_copies_v1: a copied outline under another image is a 'copy'; similar
independent tracings are not flagged; the cleaned copy drops what was named and never touches the input."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import descriptron_check_cross_image_copies_v1 as cc        # noqa: E402


def _circle(cx, cy, r, n=200, jitter=0.0, seed=0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = r + np.random.default_rng(seed).normal(0, jitter, n)
    return np.c_[cx + rr * np.cos(t), cy + rr * np.sin(t)].round(1).ravel().tolist()


def _coco(poly_a, poly_b):
    return {"images": [{"id": 1, "file_name": "a.tif"}, {"id": 2, "file_name": "b.tif"}],
            "categories": [{"id": 1, "name": "head"}],
            "annotations": [{"id": 10, "image_id": 1, "category_id": 1, "segmentation": [poly_a]},
                            {"id": 20, "image_id": 2, "category_id": 1, "segmentation": [poly_b]}]}


def test_exact_copy_is_found():
    p = _circle(300, 300, 150)
    pairs = cc.find_copies(_coco(p, list(p)))
    assert len(pairs) == 1 and pairs[0]["verdict"] == "copy"


def test_independent_similar_tracing_is_not_flagged():
    pairs = cc.find_copies(_coco(_circle(300, 300, 150, n=200, jitter=0.6, seed=1), _circle(300, 300, 150, n=173, jitter=0.6, seed=2)))
    assert pairs == []


def test_cleaned_copy_drops_image_and_keeps_input(tmp_path):
    f = tmp_path / "x.json"; p = _circle(300, 300, 150)
    f.write_text(json.dumps(_coco(p, list(p)))); before = f.read_text()
    r = subprocess.run([sys.executable, str(HERE / "descriptron_check_cross_image_copies_v1.py"), "--coco", str(f),
                        "--drop_image", "a.jpg"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "x_cleaned.json").read_text())
    assert [a["id"] for a in out["annotations"]] == [20] and [i["id"] for i in out["images"]] == [2]
    assert f.read_text() == before and (tmp_path / "x_cross_image_copies.tsv").exists()
