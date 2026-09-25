"""Tests for descriptron_video_track (pose carrying, label-to-individual assignment, DeepLabCut export).
SAM2 tracking itself is exercised on a synthetic two-animal video in the validation notes (IoU 0.98-1.00)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import descriptron_video_track as vt  # noqa: E402

H, W, N = 120, 160, 12


@pytest.fixture
def clip(tmp_path):
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.integers(30, 220, (40, 40, 3)).astype(np.uint8), (3, 3), 0)
    fr_dir = tmp_path / "frames"; fr_dir.mkdir()
    images, tracks, truth = [], [], []
    for t in range(N):
        im = np.full((H, W, 3), 200, np.uint8)
        x0, y0 = 20 + 3 * t, 30 + t                        # the square moves 3 px right, 1 px down per frame
        im[y0:y0 + 40, x0:x0 + 40] = tex
        cv2.imwrite(str(fr_dir / f"{t:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 98])
        poly = [x0, y0, x0 + 40, y0, x0 + 40, y0 + 40, x0, y0 + 40]
        images.append({"id": t + 1, "file_name": f"{t:05d}.jpg", "width": W, "height": H})
        tracks.append({"id": t + 1, "image_id": t + 1, "category_id": 1, "track_id": 1, "segmentation": [poly],
                       "bbox": [x0, y0, 40, 40], "area": 1600})
        truth.append(np.array([[x0 + 10, y0 + 10], [x0 + 30, y0 + 25]], float))
    (tmp_path / "tracks.json").write_text(json.dumps({"images": images, "annotations": tracks, "categories": [{"id": 1, "name": "ind"}]}))
    lab = {"images": [images[0]], "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "point_order": [1, 2],
           "keypoints": [truth[0][0][0], truth[0][0][1], 2, truth[0][1][0], truth[0][1][1], 2]}],       # no track_id: found by mask
           "categories": [{"id": 2, "name": "keypoints", "keypoints": ["a", "b"], "skeleton": [[1, 2]]}]}
    (tmp_path / "labels.json").write_text(json.dumps(lab))
    return tmp_path, truth


def test_mask_polygon_round_trip():
    m = np.zeros((50, 60), bool); m[10:30, 5:45] = True
    seg, bbox, area = vt.polygon_of(m)
    back = vt.mask_from_seg(seg, 50, 60)
    assert bbox == [5.0, 10.0, 40.0, 20.0] and (back == m).mean() > 0.99


def test_pose_is_carried_and_exported(clip):
    d, truth = clip
    vt.main(["pose", str(d / "frames"), "--tracks", str(d / "tracks.json"), "--labels", str(d / "labels.json"),
             "--out", str(d / "pose.json")])
    po = json.load(open(d / "pose.json"))
    assert len(po["annotations"]) == N and po["categories"][0]["skeleton"] == [[1, 2]]
    err = [np.linalg.norm(np.array(a["keypoints"]).reshape(-1, 3)[:, :2] - truth[a["image_id"] - 1], axis=1).max()
           for a in po["annotations"]]
    assert max(err) < 1.5 and min(min(a["keypoint_scores"]) for a in po["annotations"]) > 0.5
    vt.main(["export-dlc", str(d / "pose.json"), "--out", str(d / "dlc.csv")])
    rows = list(csv.reader(open(d / "dlc.csv")))
    assert [r[0] for r in rows[:4]] == ["scorer", "individuals", "bodyparts", "coords"]
    assert rows[2][1:4] == ["a", "a", "a"] and rows[3][1:4] == ["x", "y", "likelihood"] and len(rows) == 4 + N
