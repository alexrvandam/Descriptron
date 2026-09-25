"""Tests for descriptron_joints: exact angles, joints from the skeleton, pose standardisation removes posture."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_joints as dj  # noqa: E402

BASE = np.array([[0, 0], [100, 0], [150, 10], [200, 15], [250, 12]], float)


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def legs(tmp_path, n=25, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    ims, anns, true = [], [], []
    for i in range(n):
        P = BASE.copy(); bend = math.radians(rng.uniform(-40, 40))
        P[2:] = (rot(bend) @ (P[2:] - P[1]).T).T + P[1]
        u, v = P[0] - P[1], P[2] - P[1]; true.append(abs(math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], u @ v))))
        P = (rot(rng.uniform(0, 6.28)) @ P.T).T * rng.uniform(.8, 1.2) + 500 + rng.normal(0, noise, P.shape)
        ims.append({"id": i + 1, "file_name": f"l{i}.png", "width": 1000, "height": 1000})
        anns.append({"id": i + 1, "image_id": i + 1, "category_id": 1, "point_order": [1, 2, 3, 4, 5],
                     "keypoints": [x for p in P for x in (float(p[0]), float(p[1]), 2)]})
    f = tmp_path / "legs.json"
    f.write_text(json.dumps({"images": ims, "annotations": anns, "categories": [
        {"id": 1, "name": "keypoints", "keypoints": list("abcde"), "skeleton": [[1, 2], [2, 3], [3, 4], [4, 5]]}]}))
    return f, np.array(true)


def gpa_variance(path):
    d = json.load(open(path))
    X = np.array([np.array(a["keypoints"]).reshape(-1, 3)[:, :2] for a in d["annotations"]])
    X = X - X.mean(1, keepdims=True); X /= np.linalg.norm(X, axis=(1, 2), keepdims=True)
    M = X[0]
    for _ in range(10):
        for k in range(len(X)):
            u, _, vt = np.linalg.svd(X[k].T @ M); X[k] = X[k] @ (u @ vt)
        M = X.mean(0); M /= np.linalg.norm(M)
    return float(((X - X.mean(0)) ** 2).sum(axis=(1, 2)).mean())


def test_joints_from_skeleton():
    assert dj.joints_from_skeleton([[1, 2], [2, 3], [2, 4]]) == [(2, 1, 3), (2, 1, 4), (2, 3, 4)]


def test_angles_are_exact(tmp_path):
    f, true = legs(tmp_path)
    dj.main(["angles", str(f), "--out-dir", str(tmp_path / "a")])
    rows = list(csv.DictReader(open(tmp_path / "a" / "joint_angles.csv")))
    assert len([k for k in rows[0] if k.startswith("angle_")]) == 3
    meas = np.array([float(r["angle_a-b-c_deg"]) for r in rows])
    assert np.abs(meas - true).max() < 1e-3


def test_standardise_removes_posture_and_keeps_original(tmp_path):
    f, _ = legs(tmp_path, noise=1.0)
    before = f.read_text()
    dj.main(["standardise", str(f), "--rotate", "2:1,3", "--out", str(tmp_path / "s.json")])
    assert f.read_text() == before                                     # original untouched
    assert gpa_variance(tmp_path / "s.json") < 0.1 * gpa_variance(f)
    rep = json.load(open(tmp_path / "s.json"))["info"]["descriptron_joint_standardisation"][0]
    assert rep["rotated_landmarks"] == [3, 4, 5]                       # distal part found from the skeleton
    angs = [dj.angle(dj.points(a), 2, 1, 3, signed=True) for a in json.load(open(tmp_path / "s.json"))["annotations"]]
    assert np.ptp(np.mod(angs, 360)) < 0.01                            # every specimen now has the same angle (coords rounded to 0.001 px)
