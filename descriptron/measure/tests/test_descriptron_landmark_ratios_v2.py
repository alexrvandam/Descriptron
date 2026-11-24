"""Tests for descriptron_landmark_ratios_v2.py: term geometry on shapes with known answers, ratio parsing, BH, Dunn, PERMANOVA/PERMDISP,
and an end-to-end run on a synthetic COCO file (two groups that differ in width/length ratio). Run: pytest -q measure/tests/test_descriptron_landmark_ratios_v2.py"""
import csv
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('lr2', os.path.join(HERE, '..', 'descriptron_landmark_ratios_v2.py'))
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)

# a 'head' rotated by 30 deg: 1 = top midline (0, 0), 2 = bottom midline (0, 10), 3/4 = sides (-3, 4) and (4, 4), 5 = a point 2 below the 3-4 line
BASE = np.array([[0, 0], [0, 10], [-3, 4], [4, 4], [0.5, 6]], float)
th = np.radians(30)
ROT = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
K = BASE @ ROT.T + [100, 50]


def ev(s, masks=None):
    return lr.evaluate(lr.parse_term(s), K, masks or {})


def test_distance_along_across_rotation_invariant():
    assert ev('1-2') == pytest.approx(10)
    assert ev('3-4^1-2') == pytest.approx(7)          # width across the midline
    assert ev('3-4|1-2') == pytest.approx(0)          # 3 and 4 at the same level
    assert ev('1-3|1-2') == pytest.approx(4)
    assert ev('mean(1-3|1-2;1-4|1-2)') == pytest.approx(4)


def test_point_line_signed():
    assert ev('5>3-4') == pytest.approx(2)
    assert ev('5>3-4@2') == pytest.approx(2)           # same side as landmark 2 -> positive
    assert ev('5>3-4@1') == pytest.approx(-2)          # opposite side to landmark 1 -> negative


def test_missing_landmark_gives_nan():
    k = K.copy(); k[2] = np.nan
    assert np.isnan(lr.evaluate(lr.parse_term('3-4^1-2'), k, {}))
    assert np.isnan(lr.evaluate(lr.parse_term('9-10'), K, {}))


def test_mask_terms_rectangle():
    ys, xs = np.mgrid[0:20, 0:60]
    X = np.c_[xs.ravel(), ys.ravel()].astype(float)
    m = {'body': X}
    k = np.array([[0, 10], [59, 10]], float)            # axis along x
    assert lr.evaluate(lr.parse_term('body:length'), k, m) == pytest.approx(60)
    assert lr.evaluate(lr.parse_term('body:width'), k, m) == pytest.approx(20)
    assert lr.evaluate(lr.parse_term('body:width^1-2'), k, m) == pytest.approx(20)
    assert lr.evaluate(lr.parse_term('body:length|1-2'), k, m) == pytest.approx(60)
    assert lr.evaluate(lr.parse_term('body:area'), k, m) == pytest.approx(1200)


def test_parse_errors():
    with pytest.raises(ValueError):
        lr.parse_term('1+2')
    with pytest.raises(ValueError):
        lr.parse_ratio('HW*HL')


def test_bh_matches_definition():
    p = [0.01, 0.04, 0.03, 0.2]
    q = lr.bh(p)
    assert q == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.2])      # by hand: 0.01*4/1; min(0.04*4/3, 0.03*4/2); 0.2
    sm = pytest.importorskip('statsmodels.stats.multitest')
    assert q == pytest.approx(sm.multipletests(p, method='fdr_bh')[1])


def test_permanova_and_permdisp():
    rng = np.random.default_rng(1)
    X = np.r_[rng.normal(0, 1, (10, 3)), rng.normal(3, 1, (10, 3))]
    g = np.array(['a'] * 10 + ['b'] * 10)
    F, R2, p = lr.permanova(X, g, 199, 0)
    assert p < 0.05 and R2 > 0.4
    _, pd_ = lr.permdisp(X, g, 199, 0)
    assert pd_ > 0.05                                   # same spread
    _, _, p0 = lr.permanova(rng.normal(0, 1, (20, 3)), g, 199, 0)
    assert p0 > 0.05


def test_dunn_finds_the_shifted_group():
    rng = np.random.default_rng(2)
    xs = [rng.normal(0, 1, 8), rng.normal(0, 1, 8), rng.normal(4, 1, 8)]
    out = {(r[0], r[1]): r for r in lr.dunn(xs, ['a', 'b', 'c'])}
    assert out[('a', 'c')][4] < 0.05 and out[('b', 'c')][4] < 0.05 and out[('a', 'b')][4] > 0.05


def test_end_to_end(tmp_path):
    rng = np.random.default_rng(3)
    images, anns = [], []
    for i in range(16):
        sp = 'SPA' if i < 8 else 'SPB'
        w = (6 if sp == 'SPA' else 9) + rng.normal(0, 0.3)
        pts = [[0, 0], [0, 10], [-w / 2, 4], [w / 2, 4]]
        images.append({'id': i + 1, 'file_name': f'{sp}_x{i}.png', 'height': 50, 'width': 50})
        anns.append({'id': i + 1, 'image_id': i + 1, 'category_id': 1, 'keypoints': [v for p in pts for v in (p[0] + 20, p[1] + 20, 2)]})
    js = tmp_path / 'c.json'
    js.write_text(json.dumps({'images': images, 'annotations': anns, 'categories': [{'id': 1, 'name': 'line_keypoints'}]}))
    out = tmp_path / 'out'
    lr.main(['--json', str(js), '--measure', 'L=1-2', '--measure', 'W=3-4^1-2', '--ratio', 'WI=W/L*100', '--out_dir', str(out), '--permutations', '99'])
    rows = list(csv.DictReader(open(out / 'specimen_measurements.tsv'), delimiter='\t'))
    assert len(rows) == 16 and {r['group'] for r in rows} == {'SPA', 'SPB'}
    assert float(rows[0]['L']) == pytest.approx(10)
    tests = list(csv.DictReader(open(out / 'tests.tsv'), delimiter='\t'))
    loo = [t for t in tests if t['test'].startswith('LDA')][0]
    assert float(loo['statistic_or_accuracy_pct']) == pytest.approx(100) and float(loo['p']) < 0.05
    assert (out / 'pairwise_dunn.tsv').exists() and (out / 'species_summary.tsv').exists()
