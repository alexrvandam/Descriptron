"""descriptron_sam3_instances: tiling covers every pixel, duplicates across tiles merge, exemplar boxes are read
from COCO. (Running SAM 3 itself needs the sam3 environment and gated checkpoints, so it is not tested here.)"""
import json
import sys
from pathlib import Path

import numpy as np

GUI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GUI))

import descriptron_sam3_instances as s3        # noqa: E402  (the module imports SAM 3 only inside main())


def test_tiles_cover_the_image():
    for w, h in ((5000, 3000), (1008, 1008), (700, 400), (2100, 1009)):
        cov = np.zeros((h, w), bool)
        for x, y, tw, th in s3.tiles(w, h, 1008, 0.25):
            assert tw <= 1008 and th <= 1008 and x + tw <= w and y + th <= h
            cov[y:y + th, x:x + tw] = True
        assert cov.all(), (w, h)


def test_small_image_is_one_tile():
    assert s3.tiles(640, 480, 1008, 0.25) == [(0, 0, 640, 480)]


def test_merge_keeps_best_of_overlapping_duplicates():
    dets = [{"box": [0, 0, 10, 10], "score": 0.6}, {"box": [1, 1, 11, 11], "score": 0.9},
            {"box": [50, 50, 60, 60], "score": 0.7}]
    kept = s3.merge(dets, 0.5)
    assert len(kept) == 2 and kept[0]["score"] == 0.9 and kept[1]["box"] == [50, 50, 60, 60]


def test_box_iou():
    assert s3.box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert s3.box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_load_exemplars(tmp_path):
    coco = {"images": [{"id": 1, "file_name": "dir/wing1.tif"}],
            "categories": [{"id": 1, "name": "seta"}, {"id": 2, "name": "cell"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 5, 30]},
                            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 100, 100]}]}
    f = tmp_path / "ex.json"; f.write_text(json.dumps(coco))
    assert s3.load_exemplars(str(f), "seta") == {"wing1.tif": [[10, 20, 15, 50]]}


def test_gui_prompts_parse():
    assert s3.parse_boxes("30,40,10,20;1,2,3,4") == [[10, 20, 30, 40], [1, 2, 3, 4]]
    assert s3.parse_points("5.5,6,1;7,8,0") == ([[5.5, 6.0], [7.0, 8.0]], [1, 0])
    assert s3.parse_boxes("") == [] and s3.parse_points(None) == ([], [])


def test_tile_zero_is_whole_image():
    assert s3.tiles(5000, 3000, 0, 0.25) == [(0, 0, 5000, 3000)]


def test_point_window_native_resolution():
    x, y, w, h = s3.point_window([[100, 100]], 5000, 4000, 1008)
    assert (x, y, w, h) == (0, 0, 1008, 1008)                 # clipped at the corner, full size
    x, y, w, h = s3.point_window([[2500, 2000]], 5000, 4000, 1008)
    assert w == h == 1008 and x <= 2500 < x + w and y <= 2000 < y + h
    x, y, w, h = s3.point_window([[10, 10]], 600, 400, 1008)
    assert (x, y, w, h) == (0, 0, 600, 400)                   # small image: the whole image
    x, y, w, h = s3.point_window([[100, 100], [3000, 200]], 5000, 4000, 1008)
    assert x <= 100 and x + w > 3000                          # spread points: one window holds all
