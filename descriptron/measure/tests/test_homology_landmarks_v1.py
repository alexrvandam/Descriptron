"""homology_landmarks_v1: fixed landmarks as extra TPS control points of the homology grids (4 Nov 2026).

Synthetic structures with a known interior landmark field:
  * consensus landmarks are recovered in the consensus frame (rotated, scaled, translated specimens);
  * a specimen drawn as a true mirror image (outline read the other way round, as V42 reflect_mirrored writes it)
    gets its landmarks reflected correctly;
  * a mirror image whose outline was NOT reversed (an old GPA) is flagged and warped from its outline alone;
  * keypoints of another structure in the same image are ignored (outside the outline + tolerance);
  * with outline semilandmarks alone the TPS misplaces an interior landmark that moved, with landmarks it hits it;
  * no keypoints -> nothing changes (controls returned unchanged)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).parent.parent))
import homology_landmarks_v1 as H  # noqa: E402

N = 100
t = np.linspace(0, 2 * np.pi, N, endpoint=False)
MEAN = np.c_[2 * np.cos(t), np.sin(t) + 0.25 * np.sin(3 * t)]          # consensus outline (clockwise in image y-down)
LM = np.array([[0.6, 0.1], [-0.8, 0.2], [0.0, -0.4]])                   # interior landmarks, consensus frame


def _pose(p, th, s, tx, ty, mirror=False):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q = p * [-1, 1] if mirror else p
    return s * q @ R.T + [tx, ty]


def _coco(tmp, specs):
    """specs: name -> (outline, landmarks); writes a COCO file with the outline as a 'wing' polygon and the
    landmarks as a keypoint annotation, plus one stray keypoint set far away (another structure)."""
    imgs, anns = [], []
    for i, (name, (ol, lm)) in enumerate(specs.items(), 1):
        imgs.append({"id": i, "file_name": name + ".png", "width": 1000, "height": 1000})
        anns.append({"id": 10 * i, "image_id": i, "category_id": 1, "segmentation": [ol.ravel().tolist()]})
        kp = np.c_[lm, np.full(len(lm), 2)].ravel().tolist()
        anns.append({"id": 10 * i + 1, "image_id": i, "category_id": 2, "keypoints": kp, "num_keypoints": len(lm)})
        stray = np.c_[lm + 600, np.full(len(lm), 2)].ravel().tolist()
        anns.append({"id": 10 * i + 2, "image_id": i, "category_id": 2, "keypoints": stray, "num_keypoints": len(lm)})
    cats = [{"id": 1, "name": "wing"}, {"id": 2, "name": "line_keypoints", "keypoints": ["a", "b", "c"]}]
    p = tmp / "kp.json"; p.write_text(json.dumps({"images": imgs, "annotations": anns, "categories": cats}))
    return str(p)


def _specimens(rng, n=8, mirror_reversed=(), mirror_unreversed=()):
    specs = {}
    for k in range(n):
        th, s, tx, ty = rng.uniform(-0.5, 0.5), rng.uniform(80, 120), rng.uniform(300, 500), rng.uniform(300, 500)
        jit = rng.normal(0, 0.01, MEAN.shape); ljit = rng.normal(0, 0.01, LM.shape)
        mir = k in mirror_reversed or k in mirror_unreversed
        ol = _pose(MEAN + jit, th, s, tx, ty, mirror=mir); lm = _pose(LM + ljit, th, s, tx, ty, mirror=mir)
        if k in mirror_reversed:          # V42: a mirror image keeps homologous point order -> runs the other way
            pass
        if k in mirror_unreversed:        # old GPA: re-ordered to the common winding -> correspondence broken
            ol = ol[::-1]
        specs[f"s{k}"] = (ol, lm)
    return specs


def test_consensus_and_mirror(tmp_path):
    rng = np.random.default_rng(0)
    specs = _specimens(rng, mirror_reversed=(2,), mirror_unreversed=(5,))
    js = _coco(tmp_path, specs)
    outlines = {k + ".png_1": v[0] for k, v in specs.items()}
    lc = H.LandmarkControls.from_coco(js, outlines, MEAN)
    assert lc.n_used == 3
    assert np.abs(lc.mean_lm - LM).max() < 0.03                     # recovered in the consensus frame
    assert lc.info["outline_landmark_disagree"] == ["s5.png_1"]     # broken mirror flagged, true mirror kept
    assert lc.n_for("s2.png_1") == 3 and lc.n_for("s5.png_1") == 0
    assert np.all(np.abs(lc.spec_lm["s0.png_1"] - specs["s0"][1]) < 1e-9)   # the stray set was not chosen


def test_landmarks_pin_the_interior(tmp_path):
    from scipy.interpolate import RBFInterpolator
    rng = np.random.default_rng(1)
    specs = _specimens(rng, n=6)
    ol, lm = specs["s3"]
    lm = lm.copy(); lm[0] += [25.0, -15.0]                         # this specimen's landmark a moved inside
    specs["s3"] = (ol, lm)
    js = _coco(tmp_path, specs)
    outlines = {k + ".png_1": v[0] for k, v in specs.items()}
    lc = H.LandmarkControls.from_coco(js, outlines, MEAN)

    def tps(s, d):
        wx = RBFInterpolator(s, d[:, 0], kernel="thin_plate_spline", smoothing=1e-5)
        wy = RBFInterpolator(s, d[:, 1], kernel="thin_plate_spline", smoothing=1e-5)
        return lambda q: np.c_[wx(q), wy(q)]
    idx = list(range(0, N, 5))
    w0 = tps(MEAN[idx], ol[idx])
    cm, cs = lc.controls("s3.png_1", MEAN[idx], ol[idx])
    w1 = tps(cm, cs)
    e0 = np.linalg.norm(w0(lc.mean_lm[:1]) - lm[0]); e1 = np.linalg.norm(w1(lc.mean_lm[:1]) - lm[0])
    assert e1 < 0.5 and e0 > 10                                    # outline alone misses it, with landmarks it is pinned


def test_no_keypoints_changes_nothing(tmp_path):
    p = tmp_path / "nokp.json"
    p.write_text(json.dumps({"images": [{"id": 1, "file_name": "a.png", "width": 9, "height": 9}],
                             "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0, 0, 5, 0, 5, 5]]}],
                             "categories": [{"id": 1, "name": "wing"}]}))
    lc = H.LandmarkControls.from_coco(str(p), {"a.png_1": MEAN * 50 + 100}, MEAN)
    assert lc.n_used == 0
    cm, cs = lc.controls("a.png_1", MEAN[:5], MEAN[:5])
    assert cm.shape == (5, 2) and cs.shape == (5, 2)
