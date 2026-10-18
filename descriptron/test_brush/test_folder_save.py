"""Tests for descriptron_folder_save_v1 (v89 folder-session save fix, 19 Oct 2026).
Run: python3 test_brush/test_folder_save.py"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import descriptron_folder_save_v1 as fs

D = '/data/imgs'
P = [f'{D}/sp{k}.tif' for k in (1, 2, 3, 4)]
CATS = [{'id': 1, 'name': 'eye'}, {'id': 2, 'name': 'line_keypoints', 'supercategory': 'keypoints'}]


def img(k, stem_id=False):
    return {'id': f'sp{k}' if stem_id else f'{D}/sp{k}', 'file_name': f'sp{k}.tif', 'height': 10, 'width': 10}


def mask(i, k):
    return {'id': i, 'image_id': f'{D}/sp{k}', 'category_id': 1, 'segmentation': [[0, 0, 5, 0, 5, 5]], 'area': 12}


def autosaved_kp(i, k, x):
    return {'id': i, 'image_id': f'sp{k}', 'category_id': -1, 'keypoints': [x, x, 2], 'keypoints_clicked': [x, x, 2]}


def acc_avd_case():
    """sp1 and sp2 visited earlier (keypoints auto-saved on leaving), sp3 on screen; sp1 is images[0]."""
    return {'images': [img(1), img(1, True), img(2), img(2, True), img(3)],
            'annotations': [mask(1, 1), autosaved_kp(2, 1, 11.0), mask(3, 2), autosaved_kp(4, 2, 22.0), mask(5, 3),
                            autosaved_kp(6, 3, 99.0)],
            'categories': [dict(c) for c in CATS]}


CACHE = {P[0]: {'coords': [[11.0, 11.0]], 'vis': [2], 'orders': [1]},
         P[1]: {'coords': [[22.0, 22.0]], 'vis': [2], 'orders': [1]},
         P[2]: {'coords': [[33.0, 33.0]], 'vis': [2], 'orders': [1]}}
CUR = img(3)
T = []
def test(f): T.append(f); return f


@test
def current_image_first_so_live_points_land_on_it():
    ims, _ = fs.build_folder_save(acc_avd_case(), CUR, P, CACHE, {}, None)
    assert ims[0]['file_name'] == 'sp3.tif', ims[0]


@test
def one_image_entry_per_file():
    ims, _ = fs.build_folder_save(acc_avd_case(), CUR, P, CACHE, {}, None)
    assert sorted(i['file_name'] for i in ims) == ['sp1.tif', 'sp2.tif', 'sp3.tif']
    assert all(str(i['id']).startswith(D) for i in ims)          # the entry the masks use, not the stem duplicate


@test
def each_visited_image_keeps_its_own_keypoints():
    ims, anns = fs.build_folder_save(acc_avd_case(), CUR, P, CACHE, {}, None)
    kp = {a['image_id']: a['keypoints'][:2] for a in anns if a.get('keypoints')}
    assert kp == {f'{D}/sp1': [11.0, 11.0], f'{D}/sp2': [22.0, 22.0]}, kp   # sp3's live points are added by the save
    assert all(a['category_id'] == 2 for a in anns if a.get('keypoints'))   # never -1


@test
def stale_autosave_of_current_image_dropped():
    _, anns = fs.build_folder_save(acc_avd_case(), CUR, P, CACHE, {}, None)
    assert not any(a.get('keypoints', [None])[0] == 99.0 for a in anns)


@test
def masks_are_the_same_objects_so_brush_edits_reach_the_session():
    acc = acc_avd_case(); _, anns = fs.build_folder_save(acc, CUR, P, CACHE, {}, None)
    assert any(a is acc['annotations'][0] for a in anns)


@test
def session_lists_untouched():
    acc = acc_avd_case(); before = json.dumps(acc['annotations']); n = len(acc['images'])
    fs.build_folder_save(acc, CUR, P, CACHE, {}, None)
    assert json.dumps(acc['annotations']) == before and len(acc['images']) == n


@test
def unvisited_images_copied_from_source_with_category_names_mapped():
    src = {'images': [img(1), img(4)],
           'categories': [{'id': 7, 'name': 'line_keypoints'}, {'id': 8, 'name': 'eye'}, {'id': 9, 'name': 'gaster'}],
           'annotations': [{'id': 1, 'image_id': f'{D}/sp4', 'category_id': 9, 'segmentation': [[1, 1, 4, 1, 4, 4]]},
                           {'id': 2, 'image_id': f'{D}/sp4', 'category_id': 7, 'keypoints': [5, 5, 2], 'keypoints_clicked': [5, 5, 2]},
                           {'id': 3, 'image_id': f'{D}/sp1', 'category_id': 8, 'segmentation': [[0, 0, 1, 0, 1, 1]]}]}
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, 'src.json'); json.dump(src, open(p, 'w'))
        acc = acc_avd_case(); ims, anns = fs.build_folder_save(acc, CUR, P, CACHE, {}, p)
    assert 'sp4.tif' in [i['file_name'] for i in ims]
    sp4 = [a for a in anns if a['image_id'] == f'{D}/sp4']
    names = {c['id']: c['name'] for c in acc['categories']}
    assert sorted(names[a['category_id']] for a in sp4) == ['gaster', 'line_keypoints'], sp4
    assert not any(a['image_id'] == f'{D}/sp1' and a.get('area') is None and a['id'] == 3 for a in anns)  # visited sp1 not re-copied
    assert len({a['id'] for a in anns}) == len(anns)                              # ids unique


@test
def rle_sibling_attached():
    src = {'images': [img(4)], 'categories': [{'id': 9, 'name': 'gaster'}],
           'annotations': [{'id': 1, 'image_id': f'{D}/sp4', 'category_id': 9, 'segmentation': [[1, 1, 4, 1, 4, 4]]}]}
    rle = dict(src, annotations=[dict(src['annotations'][0], segmentation={'size': [10, 10], 'counts': 'abc'})])
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, 'src.json'); json.dump(src, open(p, 'w')); json.dump(rle, open(os.path.join(t, 'src.rlejson'), 'w'))
        _, anns = fs.build_folder_save(acc_avd_case(), CUR, P, CACHE, {}, p)
    a = [a for a in anns if a['image_id'] == f'{D}/sp4'][0]
    assert a['segmentation_rle']['counts'] == 'abc' and isinstance(a['segmentation'], list)


@test
def current_image_without_masks_still_gets_an_entry():
    acc = acc_avd_case(); cur = img(4)
    ims, _ = fs.build_folder_save(acc, cur, P, CACHE, {}, None)
    assert ims[0]['file_name'] == 'sp4.tif'


@test
def absent_keypoints_written_as_zero_with_clicked_kept():
    a = fs.kp_annotation({'coords': [[3, 4], [5, 6]], 'vis': [2, 0], 'orders': [1, 2]}, 'x', 2, 1)
    assert a['keypoints'] == [3.0, 4.0, 2, 0, 0, 0] and a['keypoints_clicked'][3:] == [5.0, 6.0, 0] and a['num_keypoints'] == 1


if __name__ == '__main__':
    bad = 0
    for f in T:
        try:
            f(); print('ok  ', f.__name__)
        except Exception as e:
            bad += 1; print('FAIL', f.__name__, repr(e))
    print(f'{len(T) - bad}/{len(T)} passed'); sys.exit(1 if bad else 0)
