"""Unit tests for descriptron_category_nav_v1 (v91 category navigator). Run: python test_brush/test_category_nav.py"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import descriptron_category_nav_v1 as cn

T = []


def test(f):
    T.append(f); return f


MASKS = [{'category_id': 1}, {'category_id': 2}, {'category_id': 2}, {'category_id': 3}, {'category_id': 2}]
LABELS = {0: 'pronotum', 1: 'seta', 2: 'seta', 3: 'elytron', 4: 'seta'}
ID2N = {1: 'pronotum', 2: 'seta', 3: 'elytron'}


@test
def counts_and_first_appearance_order():
    names = cn.mask_category_names(MASKS, LABELS, ID2N)
    assert cn.category_counts(names) == [('pronotum', 1), ('seta', 3), ('elytron', 1)]


@test
def order_stable_across_calls():
    names = cn.mask_category_names(MASKS, LABELS, ID2N)
    assert cn.category_counts(names) == cn.category_counts(list(names))


@test
def format_and_parse_roundtrip():
    assert cn.format_entry('seta', 3) == 'seta (3)'
    assert cn.parse_entry('seta (3)') == 'seta'
    assert cn.parse_entry('cell (c+sc) (2)') == 'cell (c+sc)'
    assert cn.parse_entry('cell (c+sc)') == 'cell (c+sc)'
    assert cn.parse_entry('') == ''


@test
def first_of_category():
    names = cn.mask_category_names(MASKS, LABELS, ID2N)
    assert cn.first_index(names, 'seta') == 1
    assert cn.first_index(names, 'absent') is None


@test
def cycling_and_wrap():
    names = cn.mask_category_names(MASKS, LABELS, ID2N)
    seq, cur = [], 0
    for _ in range(4):
        cur = cn.target_index(names, 'seta', cur); seq.append(cur)
    assert seq == [1, 2, 4, 1]


@test
def jump_from_other_category_goes_to_first():
    names = cn.mask_category_names(MASKS, LABELS, ID2N)
    assert cn.target_index(names, 'seta', 3) == 1      # current is elytron -> first seta, not next after 3
    assert cn.next_index(names, 'seta', 3) == 4        # "next of this" from mask 3


@test
def single_mask_category_stays():
    names = cn.mask_category_names(MASKS, LABELS, ID2N)
    assert cn.target_index(names, 'elytron', 3) == 3


@test
def empty_image():
    assert cn.mask_category_names([], {}, {}) == []
    assert cn.category_counts([]) == []
    assert cn.target_index([], 'seta', 0) is None
    assert cn.next_index([], 'seta', 0) is None


@test
def masks_without_labels():
    masks = [{'category_id': 2}, {}, {'label': 'eye'}, {'category_id': 99}]
    names = cn.mask_category_names(masks, {}, ID2N)
    assert names == ['seta', 'unlabeled', 'eye', 'unlabeled']
    assert cn.category_counts(names) == [('seta', 1), ('unlabeled', 2), ('eye', 1)]


@test
def labels_beat_category_id_unless_prediction_view():
    masks = [{'category_id': 1}]
    assert cn.mask_category_names(masks, {0: 'seta'}, ID2N) == ['seta']
    assert cn.mask_category_names(masks, {0: 'seta'}, ID2N, prefer_category_id=True) == ['pronotum']


@test
def deleted_masks_and_stale_labels():
    masks = list(MASKS); labels = dict(LABELS)
    del masks[1]                                        # delete_current_mask re-keys labels
    labels = {(k if k < 1 else k - 1): v for k, v in labels.items() if k != 1}
    labels[10] = 'ghost'                                # a stale key past the end is ignored
    names = cn.mask_category_names(masks, labels, ID2N)
    assert cn.category_counts(names) == [('pronotum', 1), ('seta', 2), ('elytron', 1)]
    assert cn.target_index(names, 'seta', 3) == 1      # current index 3 is seta -> wraps to 1


@test
def current_out_of_range():
    names = ['a', 'b', 'a']
    assert cn.target_index(names, 'a', 7) == 0
    assert cn.next_index(names, 'a', None) == 0


@test
def folder_counts_from_coco():
    coco = {'images': [{'id': 1, 'file_name': 'sp1.tif'}, {'id': 2, 'file_name': 'sp2.tif'},
                       {'id': 3, 'file_name': 'other.tif'}],
            'categories': [{'id': 1, 'name': 'seta'}, {'id': 2, 'name': 'Elytron'}],
            'annotations': [{'image_id': 1, 'category_id': 1, 'segmentation': [[0, 0, 1, 1, 1, 0]]},
                            {'image_id': 1, 'category_id': 1, 'segmentation': [[0, 0, 1, 1, 1, 0]]},
                            {'image_id': 2, 'category_id': 1, 'segmentation': [[0, 0, 1, 1, 1, 0]]},
                            {'image_id': 2, 'category_id': 2, 'segmentation': [[0, 0, 1, 1, 1, 0]]},
                            {'image_id': 2, 'category_id': 2, 'keypoints': [1, 1, 2]},       # keypoints only: no
                            {'image_id': 3, 'category_id': 2, 'segmentation': [[0, 0, 1, 1, 1, 0]]}]}
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, 'a.json'); json.dump(coco, open(p, 'w'))
        r = cn.folder_category_counts(p, ['/x/sp1.tif', '/x/sp2.tif'])
    assert r == [('Elytron', 1, 1), ('seta', 3, 2)], r
    assert cn.folder_category_counts('/no/such.json', ['a']) == []


if __name__ == '__main__':
    bad = 0
    for f in T:
        try:
            f(); print('ok  ', f.__name__)
        except Exception as e:
            bad += 1; print('FAIL', f.__name__, repr(e))
    print(f'{len(T) - bad}/{len(T)} passed'); sys.exit(1 if bad else 0)
