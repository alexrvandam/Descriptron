"""Tests for descriptron_centerline: lengths of shapes with known centre lines, branch mode, the CLI."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_centerline as dc  # noqa: E402

cv2 = pytest.importorskip("cv2")


def line_mask(pts, width, size=1000):
    m = np.zeros((size, size), np.uint8)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1, width)
    return m.astype(bool)


def arclen(p):
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


T = np.linspace(0, 1, 2000)
ARC = np.stack([500 + 300 * np.cos(np.radians(-150 + 120 * T)), 700 + 300 * np.sin(np.radians(-150 + 120 * T))], 1)
BAR = np.stack([200 + 346.4 * T, 300 + 200 * T], 1)


@pytest.mark.parametrize("pts", [BAR, ARC], ids=["straight", "arc"])
@pytest.mark.parametrize("width", [6, 30])
def test_length_within_one_percent(pts, width):
    r = dc.centerline(line_mask(pts, width))
    true = arclen(pts) + width                       # round caps add width/2 at each end
    assert r["length_px"] == pytest.approx(true, rel=0.01)
    assert r["width_mean_px"] == pytest.approx(width, abs=2)


def test_straight_axis_under_reads_the_arc():
    m = line_mask(ARC, 10)
    ys, xs = np.nonzero(m); P = np.stack([xs, ys], 1).astype(float); P -= P.mean(0)
    pr = P @ np.linalg.svd(P, full_matrices=False)[2][0]
    assert (pr.max() - pr.min()) < 0.9 * dc.centerline(m)["length_px"]
    assert dc.centerline(m)["sinuosity"] > 1.1


def y_shape():
    m = np.zeros((600, 600), np.uint8)
    cv2.line(m, (300, 550), (300, 300), 1, 12); cv2.line(m, (300, 300), (150, 100), 1, 12)
    cv2.line(m, (300, 300), (450, 100), 1, 12); cv2.line(m, (300, 450), (320, 445), 1, 12)
    return m.astype(bool)


def test_main_path_ignores_branches():
    r = dc.centerline(y_shape())
    assert len(r["paths"]) == 1 and r["length_px"] == pytest.approx(512, rel=0.02)


def test_branch_mode_merges_junction_pixels_and_prunes_spurs():
    r = dc.centerline(y_shape(), branches=True)
    assert r["n_segments"] == 5 and r["n_branch_points"] == 2 and r["n_tips"] == 4
    r2 = dc.centerline(y_shape(), branches=True, min_branch=40)
    assert sorted(round(s["length_px"] / 10) for s in r2["segments"]) == [10, 15, 25, 25]


def test_cli_writes_csv_in_mm_and_gui_lines(tmp_path):
    poly = cv2.findContours(line_mask(ARC, 20).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    coco = {"images": [{"id": 1, "file_name": "a.png", "width": 1000, "height": 1000, "scale_px_per_mm": 500.0}],
            "annotations": [{"id": 7, "image_id": 1, "category_id": 1, "segmentation": [poly.ravel().tolist()]}],
            "categories": [{"id": 1, "name": "vein"}]}
    f = tmp_path / "c.json"; f.write_text(json.dumps(coco))
    dc.main([str(f), "--out-dir", str(tmp_path / "o")])
    row = next(csv.DictReader(open(tmp_path / "o" / "centerline_measurements.csv")))
    assert float(row["length_mm"]) == pytest.approx(float(row["length_px"]) / 500, rel=1e-3)
    assert float(row["length_px"]) == pytest.approx(arclen(ARC) + 20, rel=0.02)
    out = json.load(open(tmp_path / "o" / "centerlines_coco.json"))
    assert out["annotations"][0]["is_line"] and out["categories"][0]["name"] == "vein_centerline"
