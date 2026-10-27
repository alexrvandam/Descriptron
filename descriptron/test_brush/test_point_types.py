"""Unit tests for descriptron_point_types_v1 (v98 keypoint "Point type"): pure functions, no display.
The keypoint operations are replayed exactly as the GUI performs them on its parallel lists (del, pop / insert,
replace on move, rebinding on clear / hide, copies on Show All), and the v96 undo module snapshots the types.
Run: python test_brush/test_point_types.py"""
import copy, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import descriptron_point_types_v1 as pt
import descriptron_undo_v1 as u

T = []


def test(f):
    T.append(f); return f


class GUI(object):
    """The GUI's keypoint state and its operations, as written in descriptron-v2-v97/v98."""
    def __init__(self):
        self.point_coords, self.point_labels, self.point_orders, self.keypoint_labels = [], [], [], []
        self.keypoint_visibility, self.keypoint_bank, self.keypoint_types = [], [], []
        self.tr = pt.TypeTracker(self.keypoint_types)
        self.current = pt.DEFAULT

    def sync(self, new_type=None):
        return self.tr.sync(self.point_coords, self.keypoint_bank, new_type=new_type)

    def pick(self, x, y, positive=True):
        self.sync()
        if [x, y] not in self.point_coords:
            self.point_coords.append([x, y]); self.point_labels.append(1 if positive else 0)
            self.keypoint_visibility.append(2 if positive else 0); self.point_orders.append(len(self.point_coords))
            self.keypoint_labels.append(len(self.point_coords))
        self.sync(new_type=self.current)

    def remove(self, i):
        self.sync()
        for l in (self.point_coords, self.keypoint_visibility, self.point_labels, self.point_orders, self.keypoint_labels):
            del l[i]
        self.keypoint_labels[:] = list(range(1, len(self.keypoint_labels) + 1))
        self.point_orders[:] = list(range(1, len(self.point_orders) + 1))
        self.sync()

    def rename(self, i, new_val):
        self.sync()
        for l in (self.keypoint_labels, self.point_coords, self.point_labels, self.point_orders, self.keypoint_visibility):
            l.insert(new_val - 1, l.pop(i))
        self.keypoint_labels[:] = list(range(1, len(self.keypoint_labels) + 1))
        self.point_orders[:] = list(range(1, len(self.point_orders) + 1))
        self.sync()

    def move(self, i, x, y):
        self.sync()
        t = self.keypoint_types[i]
        self.point_coords[i] = [x, y]               # stop_move_keypoint replaces the point object
        self.sync()                                 # (the inner redraw)
        self.tr.set_one(self.point_coords, i, t)

    def clear(self):
        self.sync()
        self.point_coords = []; self.point_labels = []; self.point_orders = []; self.keypoint_labels = []
        self.keypoint_visibility.clear()
        self.sync()

    def snapshot(self):
        self.sync()
        return {"name": "Set", "coords": [list(c) for c in self.point_coords], "labels": list(self.point_labels),
                "orders": list(self.point_orders), "kp_labels": list(self.keypoint_labels),
                "vis": list(self.keypoint_visibility), "types": list(self.keypoint_types)}

    def hide(self):
        self.keypoint_bank.append(self.snapshot())
        self.point_coords = []; self.point_labels = []; self.point_orders = []; self.keypoint_labels = []
        self.sync()

    def show(self):
        snap = self.keypoint_bank.pop()
        self.point_coords = [list(c) for c in snap["coords"]]; self.point_labels = list(snap["labels"])
        self.point_orders = list(snap["orders"]); self.keypoint_labels = list(snap["kp_labels"])
        self.keypoint_visibility[:] = list(snap["vis"])
        self.tr.set(self.point_coords, snap.get("types"))

    def show_all(self):                             # the manager's Show All: copies, bank emptied, then redraw
        self.sync()
        all_c = [list(c) for c in self.point_coords]
        for snap in self.keypoint_bank:
            all_c.extend([list(c) for c in snap["coords"]])
        self.keypoint_bank.clear()
        self.point_coords = all_c
        self.sync()

    def table(self):
        return [(tuple(c), t) for c, t in zip(self.point_coords, self.keypoint_types)]


def placed():
    g = GUI()
    g.pick(1, 1); g.pick(2, 2)
    g.current = "seta"; g.pick(3, 3); g.pick(4, 4, positive=False)
    g.current = "pit"; g.pick(5, 5)
    return g


@test
def t_list_and_persistence():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "deep", "dir", "point_types.json")
    assert pt.load_types(p) == ["landmark"]                       # no file
    lst, name = pt.add_type(pt.load_types(p), "  scale seta ")
    assert name == "scale_seta" and lst == ["landmark", "scale_seta"]
    lst, name = pt.add_type(lst, "Scale_Seta")
    assert name == "scale_seta" and len(lst) == 2                  # existing, any case
    lst, name = pt.add_type(lst, "")
    assert name is None
    assert pt.save_types(lst + ["pit"], p) and os.path.exists(p)   # dirs created
    assert pt.load_types(p) == ["landmark", "scale_seta", "pit"]
    open(p, "w").write("{not json")
    assert pt.load_types(p) == ["landmark"]                       # unreadable -> default, no exception
    assert pt.save_types(["x"], "/proc/forbidden/point_types.json") is False
    os.environ["DESCRIPTRON_POINT_TYPES"] = p
    assert pt.config_path() == p
    del os.environ["DESCRIPTRON_POINT_TYPES"]
    assert pt.config_path().endswith(os.path.join(".config", "descriptron", "point_types.json"))


@test
def t_colour_and_tag():
    assert pt.colour_for("landmark") is None and pt.colour_for(None) is None
    assert pt.colour_for("seta") == pt.colour_for("seta") and pt.colour_for("seta") in pt.PALETTE
    assert pt.tag_for("landmark") == ""
    assert pt.tag_for("seta", ["seta", "pit"]) == "s"
    assert pt.tag_for("seta", ["seta", "scale"]) == "se" and pt.tag_for("scale", ["seta", "scale"]) == "sc"
    assert all(ord(ch) < 128 for t in ("seta", "pit") for ch in pt.tag_for(t, ["seta", "pit"]))


@test
def t_add_keeps_types():
    g = placed()
    assert g.keypoint_types == ["landmark", "landmark", "seta", "seta", "pit"]
    assert len(g.keypoint_types) == len(g.point_coords) == len(g.point_labels) == len(g.point_orders) == 5
    assert g.point_labels[3] == 0 and g.keypoint_types[3] == "seta"      # a negative keeps its type
    g.pick(5, 5)                                                         # duplicate click (press + release)
    assert len(g.keypoint_types) == 5


@test
def t_remove_keeps_alignment():
    g = placed(); before = dict(g.table())
    g.remove(2)
    assert g.keypoint_types == ["landmark", "landmark", "seta", "pit"]
    assert all(before[k] == v for k, v in g.table())
    g.remove(0)
    assert g.table() == [((2, 2), "landmark"), ((4, 4), "seta"), ((5, 5), "pit")]
    assert g.point_orders == [1, 2, 3]


@test
def t_rename_reorders_types():
    g = placed(); before = dict(g.table())
    g.rename(4, 1)                                                      # pit to slot 1
    assert g.keypoint_types == ["pit", "landmark", "landmark", "seta", "seta"]
    assert all(before[k] == v for k, v in g.table())
    g.rename(0, 5)
    assert dict(g.table()) == before and g.keypoint_types[-1] == "pit"


@test
def t_move_keeps_type():
    g = placed()
    g.move(2, 30, 30)
    assert g.table()[2] == ((30, 30), "seta") and g.keypoint_types == ["landmark", "landmark", "seta", "seta", "pit"]


@test
def t_clear_and_new_points():
    g = placed(); g.clear()
    assert g.keypoint_types == []
    g.current = "landmark"; g.pick(1, 1)
    assert g.keypoint_types == ["landmark"]


@test
def t_hide_show_one_set():
    g = placed(); before = g.table()
    g.hide()
    assert g.keypoint_types == [] and g.keypoint_bank[0]["types"][2] == "seta"
    g.show()
    assert g.table() == before


@test
def t_show_all_merges_types():
    g = placed(); g.hide()
    g.current = "pore"; g.pick(9, 9)
    g.show_all()                                     # active (pore) first, then the hidden set, all copies
    assert g.keypoint_types == ["pore", "landmark", "landmark", "seta", "seta", "pit"]


@test
def t_undo_snapshot_carries_types():
    g = placed()
    groups = dict(u.GROUPS)
    try:
        u.GROUPS["kp"] = tuple(u.GROUPS["kp"]) + ("keypoint_types",)
        ns = g.__dict__
        mgr = u.UndoManager()
        e = mgr.begin(ns, "Remove keypoint", "A", "kp")
        g.remove(1)
        assert mgr.commit(e, ns)
        assert g.keypoint_types == ["landmark", "seta", "seta", "pit"]
        e = mgr.pop_state()
        back = mgr.begin(ns, e.label, "A", e.fields)
        u.restore(ns, e.data)                        # refills the lists in place (types too)
        g.tr.sync(g.point_coords, g.keypoint_bank, trust=True)
        assert g.keypoint_types == ["landmark", "landmark", "seta", "seta", "pit"]
        assert len(g.point_coords) == 5
        mgr.push_redo(back)
        r = mgr.pop_redo(); u.restore(ns, r.data); g.tr.sync(g.point_coords, trust=True)
        assert g.keypoint_types == ["landmark", "seta", "seta", "pit"]
        # a type-only difference is seen as a change (so an undo entry is kept)
        e = mgr.begin(ns, "x", "A", "kp"); g.keypoint_types[0] = "seta"
        assert u.differs(ns, e.data)
    finally:
        u.GROUPS.clear(); u.GROUPS.update(groups)


def _ann(coords, vis, types=None, clicked=True):
    kps, cl = [], []
    for (x, y), v in zip(coords, vis):
        kps += [x, y, v] if v > 0 else [0, 0, 0]
        cl += [x, y, max(v, 0)]
    a = {"id": 1, "image_id": "img", "category_id": 1, "keypoints": kps, "num_keypoints": sum(v > 0 for v in vis),
         "point_order": list(range(1, len(coords) + 1)), "area": 1, "iscrowd": 0}
    if clicked:
        a["keypoints_clicked"] = cl
    if types is not None:
        a["keypoint_types"] = types
    return a


@test
def t_json_field_round_trip():
    g = placed()
    a = _ann(g.point_coords, g.keypoint_visibility)
    data = {"images": [{"id": "img"}], "annotations": [a], "categories": []}
    assert pt.add_types_to_coco(data, [(g.point_coords, g.keypoint_types)]) == 1
    assert a["keypoint_types"] == ["landmark", "landmark", "seta", "seta", "pit"]
    d = tempfile.mkdtemp(); p = os.path.join(d, "x_keypoints.json")
    json.dump(data, open(p, "w"), indent=4)
    back = json.load(open(p))["annotations"][0]
    # load as the GUI does (negative slot from keypoints_clicked)
    coords = [list(alts[0]) for alts in pt.annotation_coords(back)]
    assert coords == g.point_coords
    assert pt.find_types_for_coords(json.load(open(p))["annotations"], coords) == g.keypoint_types
    assert pt.types_from_annotation(back) == g.keypoint_types


@test
def t_file_without_field_is_landmark():
    a = _ann([[1, 1], [2, 2]], [2, 0])
    assert pt.types_from_annotation(a) == ["landmark", "landmark"]
    assert pt.find_types_for_coords([a], [[1, 1], [2, 2]]) == ["landmark", "landmark"]
    assert pt.find_types_for_coords([a], [[7, 7]]) == ["landmark"]                  # no match
    bad = dict(a, keypoint_types=["seta"])                                           # wrong length -> ignored
    assert pt.types_from_annotation(bad) == ["landmark", "landmark"]
    line = dict(_ann([[1, 1], [2, 2]], [2, 2], ["seta", "seta"]), source_line_category="antenna")
    assert pt.find_types_for_coords([line], [[1, 1], [2, 2]]) == ["landmark", "landmark"]   # line-derived skipped


@test
def t_patch_only_adds_the_list():
    coords = [[10.5, 20.25], [30.0, 40.0], [50.0, 60.0]]
    v97 = {"images": [{"id": "img", "file_name": "img.jpg", "height": 100, "width": 100}],
           "annotations": [_ann(coords, [2, 2, 0]), _ann([[1, 1]], [2])],
           "categories": [{"id": 1, "name": "keypoints", "supercategory": "keypoints"}]}
    d = tempfile.mkdtemp(); p = os.path.join(d, "a.json")
    json.dump(v97, open(p, "w"), indent=4); os.chmod(p, 0o644)
    n = pt.patch_json_file(p, [(coords, ["landmark"] * 3)])
    v98 = json.load(open(p))
    assert n == 1 and v98["annotations"][0]["keypoint_types"] == ["landmark"] * 3
    assert "keypoint_types" not in v98["annotations"][1]                           # unknown set untouched
    strip = copy.deepcopy(v98); strip["annotations"][0].pop("keypoint_types")
    assert strip == v97                                                              # nothing else changed
    assert oct(os.stat(p).st_mode & 0o777) == oct(0o644)
    assert list(v98["annotations"][0].keys())[:-1] == list(v97["annotations"][0].keys())
    mt = os.path.getmtime(p)
    assert pt.patch_json_file(p, [(coords, ["seta"] * 3)]) == 0                     # already has the field
    assert json.load(open(p))["annotations"][0]["keypoint_types"] == ["landmark"] * 3
    assert pt.patch_json_file(os.path.join(d, "missing.json"), [(coords, ["x"] * 3)]) == 0


@test
def t_fit():
    assert pt.fit(None, 2) == ["landmark", "landmark"]
    assert pt.fit(["a", "", None, "b"], 3) == ["a", "landmark", "landmark"]
    assert pt.fit(["a"], 0) == []


if __name__ == "__main__":
    for f in T:
        f()
        print("ok", f.__name__)
    print(f"{len(T)} / {len(T)} point-type tests passed")
