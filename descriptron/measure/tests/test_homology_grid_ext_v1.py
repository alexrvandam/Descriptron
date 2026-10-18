"""Tests for homology_grid_ext.py, homology_batch_meanshift_v1.py and the pipeline options (next release, GUI v86)."""
import os, sys, json, tempfile
import numpy as np
import cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import homology_grid_ext as hg
import homology_batch_meanshift_v1 as bm


def square_contour(n=100, s=100.0):
    t = np.linspace(0, 4, n, endpoint=False); pts = []
    for u in t:
        k, f = int(u), u - int(u)
        pts.append([(0, f), (f, 1), (1, 1 - f), (1 - f, 0)][k])
    return np.array(pts) * s


def test_extended_grid_adds_cells_and_keeps_margin_cells():
    c = square_contour()
    g0 = hg.create_grid_extended(c, 8, 10, ext=0, min_overlap=0.02)
    g2 = hg.create_grid_extended(c, 8, 10, ext=2, min_overlap=0.02)
    assert (g0['n_rows'], g0['n_cols']) == (8, 10) and (g2['n_rows'], g2['n_cols']) == (12, 14)
    assert g0['cell_inside'].sum() == 80               # a square fills every core cell
    assert g2['cell_inside'].sum() == 80               # extension cells lie fully outside a square
    assert np.isclose(g2['x_lines'][2], 0) and np.isclose(g2['x_lines'][-3], 100)


def test_overlap_rule_keeps_cell_whose_centre_is_outside():
    # square with one corner cut off (x + y <= 160): the corner cell's centre (87.5, 87.5) is outside,
    # but about 8% of that cell lies inside the outline
    poly = np.array([[0, 0], [100, 0], [100, 60], [60, 100], [0, 100]], float)
    g = hg.create_grid_extended(poly, 4, 4, ext=0, min_overlap=0.02)
    assert not (87.5 + 87.5 <= 160)
    assert g['cell_inside'][3, 3] and 0.05 < g['cell_overlap'][3, 3] < 0.12
    assert g['cell_inside'].sum() == 16


def test_visible_and_expected_and_cell_fraction():
    fg = np.zeros((100, 100), np.uint8); fg[20:80, 20:80] = 1
    occ_ann = {'segmentation': None}
    def dec(a, h, w):
        m = np.zeros((h, w), np.uint8); m[40:50, 0:100] = 1; return m   # a bar (an antenna) across the structure
    fg_cut = fg & (1 - dec(None, 100, 100))                              # the annotator's mask excludes the antenna
    vis, exp = hg.visible_and_expected(fg_cut, [occ_ann], dec)
    assert vis.sum() == fg_cut.sum()
    assert exp.sum() == fg.sum()                                         # hidden band restored only inside the hull
    frac = hg.cell_visible_fraction(lambda m: m[30:60, 30:60], vis, exp)
    assert np.isclose(frac, 20 / 30)


def test_no_occluders_changes_nothing():
    fg = np.zeros((50, 50), np.uint8); fg[10:40, 10:40] = 1
    vis, exp = hg.visible_and_expected(fg, [], lambda a, h, w: np.zeros((h, w), np.uint8))
    assert (vis == fg).all() and (exp == fg).all()


def test_repair_outline_bridges_the_occluded_notch():
    m = np.zeros((200, 200), np.uint8); cv2.circle(m, (100, 100), 60, 1, -1)
    m[95:105, :] = 0                                                     # antenna cuts the outline in two
    def dec(a, h, w):
        z = np.zeros((h, w), np.uint8); z[95:105, :] = 1; return z
    rep = hg.repair_outline(m, [{}], 200, 200, dec)
    full = np.zeros((200, 200), np.uint8); cv2.circle(full, (100, 100), 60, 1, -1)
    assert abs(int(rep.sum()) - int(full.sum())) < 0.01 * full.sum()   # the disc is whole again
    assert rep[100, 100] == 1 and rep[100, 5] == 0                       # nothing added outside the convex hull


def test_occluder_lookup_never_falls_back_to_another_category():
    coco = {'images': [{'id': 1, 'file_name': 'a.jpg'}], 'categories': [{'id': 1, 'name': 'head'}, {'id': 2, 'name': 'scape'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1}, {'id': 2, 'image_id': 1, 'category_id': 2}]}
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as fh:
        json.dump(coco, fh)
    try:
        assert hg.build_occluder_lookup(fh.name, ['antenna']) == {}     # unknown name: nothing, not 'head'
        lk = hg.build_occluder_lookup(fh.name, ['scape'])
        assert list(lk) == ['a.jpg'] and lk['a.jpg'][0]['category_id'] == 2
    finally:
        os.unlink(fh.name)


def test_batch_mean_shift_removes_offset_from_shared_species():
    rng = np.random.default_rng(0)
    sp = np.array(['A'] * 6 + ['B'] * 6 + ['C'] * 3); bt = np.array((['r'] * 3 + ['o'] * 3) * 2 + ['o'] * 3)
    mu = {'A': 0.0, 'B': 5.0, 'C': 10.0}
    X = np.array([[mu[s] + (2.0 if b == 'o' else 0.0)] for s, b in zip(sp, bt)]) + rng.normal(0, 0.01, (15, 1))
    Y, info = bm.correct(X, sp, bt, 'r')
    assert info['o'] == ['A', 'B']                                      # only species imaged in both batches
    assert np.allclose(Y[bt == 'o', 0] - np.array([mu[s] for s in sp[bt == 'o']]), 0, atol=0.05)
    Y2, _ = bm.correct(X, sp, bt, 'r', exclude='A')                     # leave-one-species-out still corrects
    assert np.allclose(Y2[(bt == 'o') & (sp == 'A'), 0], 0, atol=0.05)


def test_pipeline_grid_options():
    sys.path.insert(0, os.path.dirname(HERE))
    import run_full_pipeline_v2 as rp
    assert rp._homology_grid_opts({}) == []
    assert rp._homology_grid_opts({'grid_ext': 2, 'occluder_categories': 'scape', 'grid_figures_all': True}) == \
        ['--grid_ext', '2', '--occluder_categories', 'scape', '--viz_specimens', '-1']


def _coco_two_legs_and_body():
    return {'categories': [{'id': 1, 'name': 'elytron'}, {'id': 2, 'name': 'metafemur', 'occluder': True},
                           {'id': 3, 'name': 'metatibia', 'occluder': True}, {'id': 4, 'name': 'pronotum', 'occludes': ['head']},
                           {'id': 5, 'name': 'head'}, {'id': 6, 'name': 'stria'}],
            'images': [], 'annotations': []}


def test_occluders_per_pair_and_auto():
    coco = _coco_two_legs_and_body()
    occ = hg.resolve_occluders('auto', coco)
    assert occ == {'metafemur': {'*'}, 'metatibia': {'*'}, 'pronotum': {'head'}}
    # the pronotum hides the head but only abuts the elytron
    assert hg.occluders_for('elytron', occ, {}) == {'metafemur', 'metatibia'}
    assert hg.occluders_for('head', occ, {}) == {'metafemur', 'metatibia', 'pronotum'}
    # legs crossing legs: each leg is hidden by the other, never by itself
    assert hg.occluders_for('metafemur', occ, {}) == {'metatibia'}
    # command-line form
    assert hg.resolve_occluders('pronotum>head|elytron, metafemur', coco) == {'pronotum': {'head', 'elytron'}, 'metafemur': {'*'}}


def test_nested_parts_are_never_occluders():
    occ = {'stria': {'*'}, 'metafemur': {'*'}}
    related = {'elytron': {'stria'}, 'stria': {'elytron'}}      # striae lie inside the elytron
    assert hg.occluders_for('elytron', occ, related) == {'metafemur'}


def test_containment_detected_from_masks():
    def poly(x0, y0, x1, y1): return [[x0, y0, x1, y0, x1, y1, x0, y1]]
    coco = {'categories': [{'id': 1, 'name': 'head_capsule'}, {'id': 2, 'name': 'cephalic_dorsum'}, {'id': 3, 'name': 'scape'}],
            'images': [{'id': i, 'file_name': f'{i}.jpg', 'height': 200, 'width': 200} for i in (1, 2)], 'annotations': []}
    for i in (1, 2):
        coco['annotations'] += [{'id': 10 * i + 1, 'image_id': i, 'category_id': 1, 'segmentation': poly(20, 20, 180, 180)},
                                {'id': 10 * i + 2, 'image_id': i, 'category_id': 2, 'segmentation': poly(60, 40, 140, 160)},
                                {'id': 10 * i + 3, 'image_id': i, 'category_id': 3, 'segmentation': poly(0, 120, 100, 135)}]
    rel = hg.containment_related(coco)
    assert 'cephalic_dorsum' in rel['head_capsule'] and 'scape' not in rel.get('head_capsule', set())


def test_repair_ignores_occluder_that_does_not_touch():
    m = np.zeros((200, 200), np.uint8); cv2.fillPoly(m, [np.array([[20, 20], [180, 20], [180, 180], [100, 100], [20, 180]], np.int32)], 1)
    def dec(a, h, w):     # a leg lying inside the notch of the outline without touching the structure
        z = np.zeros((h, w), np.uint8); z[150:170, 95:105] = 1; return z
    rep = hg.repair_outline(m, [{}], 200, 200, dec)
    assert rep[160, 100] == 0


def test_gui_occlusion_flags_round_trip():
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    import descriptron_category_edit as ce
    coco = {'categories': [{'id': 1, 'name': 'elytron'}, {'id': 2, 'name': 'metafemur'}, {'id': 3, 'name': 'pronotum'}, {'id': 4, 'name': 'head'}],
            'annotations': []}
    coco = ce.set_occlusion(coco, 'metafemur', '*')
    coco = ce.set_occlusion(coco, 'pronotum', ['head'])
    assert ce.occlusion_of(coco) == {'elytron': None, 'metafemur': '*', 'pronotum': ['head'], 'head': None}
    # what the GUI writes is what the measurement side reads
    assert hg.resolve_occluders('auto', coco) == {'metafemur': {'*'}, 'pronotum': {'head'}}
    coco = ce.set_occlusion(coco, 'metafemur', None)
    assert ce.occlusion_of(coco)['metafemur'] is None and 'occluder' not in coco['categories'][1]
