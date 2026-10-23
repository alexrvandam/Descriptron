"""Unit tests for descriptron_undo_v1 (v96 general undo + unsaved-changes tracking).
Run: python test_brush/test_undo.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import descriptron_undo_v1 as u

T = []


def test(f):
    T.append(f); return f


def make_ns():
    m0 = np.zeros((4, 4), np.uint8); m0[1, 1] = 1
    m1 = np.zeros((4, 4), np.uint8); m1[2, 2] = 1
    return {
        "point_coords": [[1.0, 2.0], [3.0, 4.0]], "point_labels": [1, 1], "point_orders": [1, 2],
        "keypoint_labels": [1, 2], "keypoint_visibility": [2, 2], "keypoint_bank": [],
        "segmentation_masks": [{"segmentation": m0, "category_id": 1, "annotation_id": 10},
                               {"segmentation": m1, "category_id": 2, "annotation_id": 11}],
        "labels": {0: "head", 1: "eye"}, "current_mask_index": 1, "structure_status": {},
        "coco_output_accumulate": {"images": [{"id": 1}], "categories": [{"id": 1, "name": "head"}],
                                   "annotations": [{"id": 10, "segmentation": [[0, 0, 1, 1, 2, 2]]},
                                                   {"id": 11, "segmentation": [[3, 3, 4, 4, 5, 5]]}]},
        "coco_output_mask": {"images": [], "categories": [], "annotations": []},
        "_UNDO_STACK": [],
    }


def clear_points(ns):
    ns["point_coords"] = []; ns["point_labels"] = []; ns["point_orders"] = []; ns["keypoint_labels"] = []
    ns["keypoint_visibility"].clear()


def trash_current(ns):
    i = ns["current_mask_index"]
    aid = ns["segmentation_masks"][i]["annotation_id"]
    acc = ns["coco_output_accumulate"]
    acc["annotations"] = [a for a in acc["annotations"] if a["id"] != aid]
    del ns["segmentation_masks"][i]
    ns["labels"] = {k: v for k, v in ns["labels"].items() if k != i}
    ns["current_mask_index"] = max(0, i - 1)


def do(mgr, ns, fn, label, key="A", group="all", keep=False):
    e = mgr.begin(ns, label, key, group)
    fn(ns)
    return mgr.commit(e, ns, keep_if_noop=keep)


def undo(mgr, ns, key="A"):
    kind, info = mgr.decide(ns, key, ns["_UNDO_STACK"][-1][1] if ns["_UNDO_STACK"] else None)
    if kind == "state":
        e = mgr.pop_state()
        r = mgr.begin(ns, e.label, key, e.fields)
        u.restore(ns, e.data)
        mgr.push_redo(r)
    elif kind == "pixel":
        idx, saved = ns["_UNDO_STACK"].pop()
        ns["segmentation_masks"][idx]["segmentation"][...] = saved
    return kind, info


def pixel_edit(mgr, ns, key="A", i=0):
    seg = ns["segmentation_masks"][i]["segmentation"]
    saved = seg.copy()
    ns["_UNDO_STACK"].append((i, saved)); mgr.stamp_pixel(saved, key)
    seg[0, 0] = 1


@test
def clear_points_then_undo_restores_all_parallel_lists():
    ns, mgr = make_ns(), u.UndoManager()
    before = u.capture(ns, u.ALL_FIELDS)
    assert do(mgr, ns, clear_points, "Clear points", group="kp")
    assert ns["point_coords"] == []
    kind, info = undo(mgr, ns)
    assert kind == "state" and info["label"] == "Clear points"
    assert not u.differs(ns, before)
    assert ns["point_coords"] == [[1.0, 2.0], [3.0, 4.0]] and ns["keypoint_visibility"] == [2, 2]


@test
def lifo_order_of_state_entries():
    ns, mgr = make_ns(), u.UndoManager()
    s0 = u.capture(ns, u.ALL_FIELDS)
    do(mgr, ns, trash_current, "Trash mask")
    s1 = u.capture(ns, u.ALL_FIELDS)
    do(mgr, ns, clear_points, "Clear points", group="kp")
    assert undo(mgr, ns)[1]["label"] == "Clear points"
    assert not u.differs(ns, s1)
    assert undo(mgr, ns)[1]["label"] == "Trash mask"
    assert not u.differs(ns, s0)
    assert len(ns["segmentation_masks"]) == 2 and len(ns["coco_output_accumulate"]["annotations"]) == 2
    assert undo(mgr, ns)[0] == "none"


@test
def masks_share_arrays_not_copied():
    ns, mgr = make_ns(), u.UndoManager()
    arr = ns["segmentation_masks"][0]["segmentation"]
    e = mgr.begin(ns, "x", "A")
    assert e.data["segmentation_masks"][0]["segmentation"] is arr
    assert e.data["segmentation_masks"][0] is not ns["segmentation_masks"][0]


@test
def restore_is_in_place_for_lists_and_dicts():
    ns, mgr = make_ns(), u.UndoManager()
    masks, acc = ns["segmentation_masks"], ns["coco_output_accumulate"]
    do(mgr, ns, trash_current, "Trash mask")
    undo(mgr, ns)
    assert ns["segmentation_masks"] is masks and ns["coco_output_accumulate"] is acc


@test
def snapshot_stays_pristine_after_restore_and_later_edits():
    ns, mgr = make_ns(), u.UndoManager()
    do(mgr, ns, clear_points, "Clear points", group="kp")
    undo(mgr, ns)
    ns["point_coords"][0][0] = 99.0                    # edit the restored list in place
    kind, _ = mgr.decide(ns, "A")                       # redo entry must not be affected; nothing left to undo
    assert kind == "none"
    r = mgr.pop_redo()
    assert r.data["point_coords"] == []


@test
def per_image_refusal():
    ns, mgr = make_ns(), u.UndoManager()
    do(mgr, ns, clear_points, "Clear points", key="A", group="kp")
    kind, info = mgr.decide(ns, "B")
    assert kind == "refuse" and info["key"] == "A" and info["label"] == "Clear points"
    assert len(mgr.states) == 1                         # refused, not popped
    assert mgr.decide(ns, "A")[0] == "state"


@test
def pixel_entry_of_other_image_is_refused():
    ns, mgr = make_ns(), u.UndoManager()
    pixel_edit(mgr, ns, key="A")
    kind, info = mgr.decide(ns, "B", ns["_UNDO_STACK"][-1][1])
    assert kind == "refuse" and info["key"] == "A"


@test
def max_size():
    ns, mgr = make_ns(), u.UndoManager(max_states=30)
    for k in range(40):
        do(mgr, ns, lambda n, k=k: n["point_coords"].append([float(k), 0.0]), f"add {k}", group="kp")
    assert len(mgr.states) == 30
    assert mgr.states[0].label == "add 10" and mgr.states[-1].label == "add 39"


@test
def noop_actions_are_not_kept():
    ns, mgr = make_ns(), u.UndoManager()
    assert not do(mgr, ns, lambda n: None, "nothing")
    assert mgr.states == [] and not mgr.dirty


@test
def pending_dialog_entry_becomes_real_on_change():
    ns, mgr = make_ns(), u.UndoManager()
    mgr.mark_clean(ns)
    assert do(mgr, ns, lambda n: None, "Record absent", keep=True)
    assert mgr.states[-1].pending and not mgr.is_dirty(ns, "A")
    ns["structure_status"]["x.tif"] = {"eye": "absent"}          # the dialog's OK, later
    assert mgr.is_dirty(ns, "A") and not mgr.states[-1].pending
    assert undo(mgr, ns)[0] == "state" and ns["structure_status"] == {}


@test
def cancelled_dialog_entry_is_skipped_by_undo():
    ns, mgr = make_ns(), u.UndoManager()
    do(mgr, ns, clear_points, "Clear points", group="kp")
    do(mgr, ns, lambda n: None, "Hide keypoints", group="kp", keep=True)      # dialog cancelled
    kind, info = undo(mgr, ns)
    assert kind == "state" and info["label"] == "Clear points"


@test
def consecutive_noop_pending_entries_collapse():
    ns, mgr = make_ns(), u.UndoManager()
    for _ in range(5):
        do(mgr, ns, lambda n: None, "Characters", keep=True)
    assert len(mgr.states) == 1


@test
def mixed_pixel_and_state_ordering():
    ns, mgr = make_ns(), u.UndoManager()
    seg0 = ns["segmentation_masks"][0]["segmentation"]
    orig = seg0.copy()
    do(mgr, ns, clear_points, "Clear points", group="kp")   # 1 state
    pixel_edit(mgr, ns)                                       # 2 pixel
    do(mgr, ns, lambda n: n["labels"].__setitem__(0, "gena"), "Apply label")   # 3 state
    pixel_edit(mgr, ns, i=1)                                  # 4 pixel
    order = [undo(mgr, ns)[0] for _ in range(5)]
    assert order == ["pixel", "state", "pixel", "state", "none"], order
    assert np.array_equal(seg0, orig) and ns["labels"][0] == "head" and len(ns["point_coords"]) == 2


@test
def state_restore_brings_back_the_pixel_stack():
    """Trash Mask rewrites the pixel stack (v88 shift); undoing it must bring the old stack back."""
    ns, mgr = make_ns(), u.UndoManager()
    pixel_edit(mgr, ns, i=1)
    stack_before = list(ns["_UNDO_STACK"])

    def trash_and_shift(n):
        trash_current(n)
        n["_UNDO_STACK"][:] = [e for e in n["_UNDO_STACK"] if e[0] != 1]
    do(mgr, ns, trash_and_shift, "Trash mask")
    assert ns["_UNDO_STACK"] == []
    assert undo(mgr, ns)[0] == "state"
    assert len(ns["_UNDO_STACK"]) == 1 and ns["_UNDO_STACK"][0][1] is stack_before[0][1]
    assert undo(mgr, ns)[0] == "pixel"


@test
def dirty_flag_lifecycle():
    ns, mgr = make_ns(), u.UndoManager()
    mgr.mark_clean(ns)
    assert not mgr.is_dirty(ns)
    do(mgr, ns, clear_points, "Clear points", group="kp")
    assert mgr.is_dirty(ns)
    mgr.mark_clean(ns)
    assert not mgr.is_dirty(ns)
    pixel_edit(mgr, ns)
    assert mgr.is_dirty(ns)
    mgr.mark_clean(ns)
    ns["coco_output_accumulate"]["annotations"].pop()       # an unrecorded change of the session file
    assert mgr.is_dirty(ns)


@test
def redo_after_undo_and_new_action_clears_redo():
    ns, mgr = make_ns(), u.UndoManager()
    do(mgr, ns, clear_points, "Clear points", group="kp")
    undo(mgr, ns)
    r = mgr.pop_redo()
    back = mgr.begin(ns, r.label, "A", r.fields)
    u.restore(ns, r.data)
    mgr.push_after_redo(back)
    assert ns["point_coords"] == []
    assert undo(mgr, ns)[0] == "state" and len(ns["point_coords"]) == 2
    do(mgr, ns, trash_current, "Trash mask")
    assert mgr.redo == []


@test
def same_handles_arrays_and_nesting():
    a = np.zeros(3)
    assert u.same({"x": [a, 1]}, {"x": [a, 1]})
    assert not u.same({"x": [a, 1]}, {"x": [a.copy(), 1]})
    assert not u.same([1, 2], [1, 2, 3]) and u.same((1, [2]), (1, [2]))
    assert u.copy_value([1, 2.5, None]) == [1, 2.5, None]


@test
def save_keypoints_clears_only_keypoint_changes():
    ns, mgr = make_ns(), u.UndoManager()
    mgr.mark_clean(ns)
    do(mgr, ns, clear_points, "Clear points", group="kp")
    was = mgr.session_differs(ns)
    ns["coco_output_accumulate"]["annotations"].append({"id": 99, "keypoints": []})   # the keypoint save adds this
    mgr.mark_keypoints_saved(ns, was)
    assert not mgr.is_dirty(ns)
    do(mgr, ns, trash_current, "Trash mask")
    was = mgr.session_differs(ns)
    mgr.mark_keypoints_saved(ns, was)
    assert mgr.is_dirty(ns)                              # the trashed mask is still unsaved


if __name__ == '__main__':
    bad = 0
    for f in T:
        try:
            f(); print('ok  ', f.__name__)
        except Exception as e:
            bad += 1; print('FAIL', f.__name__, repr(e))
    print(f'{len(T) - bad}/{len(T)} passed'); sys.exit(1 if bad else 0)
