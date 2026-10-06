"""descriptron_labelled_plates_v1: leader lines end on the structure's own outline; labels never overlap;
the program runs end to end on a tiny synthetic COCO set."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import descriptron_labelled_plates_v1 as lp        # noqa: E402


def test_exit_point_is_on_the_outline_facing_the_label():
    m = np.zeros((100, 100), bool); m[40:60, 30:70] = True
    x, y = lp.exit_point(m, (50, 50), (200, 50))
    assert (x, y) == (69, 50) and m[y, x] and not m[y, x + 1]


def test_container_points_at_its_margin_not_at_a_part_inside():
    wing = np.zeros((100, 200), bool); cv2.ellipse(wing.view(np.uint8), (100, 50), (90, 40), 0, 0, 360, 1, -1)
    wing = wing.astype(bool)
    cell = np.zeros_like(wing); cell[35:65, 60:140] = True
    own = lp.own_regions([("whole_wing", wing), ("cell", cell)])
    assert not (own[0] & cell).any() and own[1].sum() == cell.sum()


def test_spread_keeps_gap_and_bounds():
    ys = lp.spread([10, 11, 12, 13, 90], 0, 100, 8)
    s = sorted(ys)
    assert all(b - a >= 8 - 1e-6 for a, b in zip(s, s[1:])) and min(s) >= 0 and max(s) <= 100


def test_end_to_end(tmp_path):
    img = np.full((120, 200, 3), 200, np.uint8); cv2.imwrite(str(tmp_path / "a_forewing.tif"), img)
    coco = {"images": [{"id": 1, "file_name": "a_forewing.tif", "height": 120, "width": 200}],
            "categories": [{"id": 1, "name": "whole_wing"}, {"id": 2, "name": "cell_a"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[20, 20, 180, 20, 180, 100, 20, 100]]},
                            {"id": 2, "image_id": 1, "category_id": 2, "segmentation": [[60, 40, 100, 40, 100, 80, 60, 80]]}]}
    (tmp_path / "c.json").write_text(json.dumps(coco))
    (tmp_path / "g.csv").write_text("filename,group_label\na_forewing.tif,spX\n")
    r = subprocess.run([sys.executable, str(HERE / "descriptron_labelled_plates_v1.py"), "--coco_json", str(tmp_path / "c.json"),
                        "--image_dir", str(tmp_path), "--group_labels", str(tmp_path / "g.csv"),
                        "--plates_dir", str(tmp_path / "out"), "--dpi", "60"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out/spX/a_forewing_labelled.png").exists()
    assert (tmp_path / "out/spX/spX_labelled_plate.png").exists() and (tmp_path / "out/spX/spX_labelled_plate.pdf").exists()


def test_round_robin_shows_every_body_part_first():
    imgs = [{"file_name": f"x_rostrum_{i}.tif"} for i in range(5)] + [{"file_name": "x_forewing_1.tif"}]
    first = [lp.gsp.extract_body_part(i["file_name"]) for i in lp.round_robin(imgs)[:2]]
    assert set(first) == {"rostrum", "forewing"}


def test_centre_anchor_option_runs(tmp_path):
    img = np.full((120, 200, 3), 200, np.uint8); cv2.imwrite(str(tmp_path / "a_forewing.tif"), img)
    coco = {"images": [{"id": 1, "file_name": "a_forewing.tif", "height": 120, "width": 200}],
            "categories": [{"id": 1, "name": "whole_wing"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[20, 20, 180, 20, 180, 100, 20, 100]]}]}
    (tmp_path / "c.json").write_text(json.dumps(coco))
    (tmp_path / "g.csv").write_text("filename,group_label\na_forewing.tif,spX\n")
    r = subprocess.run([sys.executable, str(HERE / "descriptron_labelled_plates_v1.py"), "--coco_json", str(tmp_path / "c.json"),
                        "--image_dir", str(tmp_path), "--group_labels", str(tmp_path / "g.csv"),
                        "--plates_dir", str(tmp_path / "out"), "--dpi", "60", "--anchor", "centre"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr and (tmp_path / "out/spX/a_forewing_labelled.png").exists()


def test_pipeline_passes_plate_anchor():
    src = (HERE / "run_full_pipeline_v2.py").read_text()
    assert '"--anchor", cfg.get("plate_anchor", "edge")' in src and '"--plate_anchor"' in src


def test_container_detection():
    wing = np.zeros((100, 200), bool); wing[10:90, 10:190] = True
    cell = np.zeros_like(wing); cell[40:60, 40:80] = True
    other = np.zeros_like(wing); other[0:5, 0:5] = True
    s = [("whole_wing", wing), ("cell", cell), ("other", other)]
    assert lp.is_container(s, 0) and not lp.is_container(s, 1) and not lp.is_container(s, 2)
