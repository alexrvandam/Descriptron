"""
Headless tests for the v88 mask tools (right-click menu, lasso, push/pull edge, category change, undo, save).

The functions under test are taken VERBATIM from descriptron-v2-v88.py (parsed with ast), then run in a
namespace where Tk, the canvas and dialogs are stubs. The category change goes through the REAL apply_label and
save_mask_and_contour_via2, and the save test runs the REAL save_final_via2_json (the v70 body) behind the v88
wrapper, writing .json and .rlejson into a temporary folder.

Run:  <python of the GUI env> test_brush/test_v88_context_tools.py
"""
import ast, json, logging, os, sys, tempfile, time, types

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V88 = os.path.join(os.path.dirname(HERE), "descriptron-v2-v88.py")
SRC = open(V88).read()
DEFS = {}
for _n in ast.parse(SRC).body:
    if isinstance(_n, (ast.FunctionDef, ast.ClassDef)):
        DEFS.setdefault(_n.name, []).append(ast.get_source_segment(SRC, _n))


def src(name, which=-1):
    return DEFS[name][which]


class Var:
    def __init__(self, v=""):
        self.v = v

    def set(self, v):
        self.v = v

    def get(self):
        return self.v


def _noop(*a, **k):
    return None


PURE = ["_v88_canvas_points_to_image", "_v88_polygon_to_mask", "_v88_fg_value", "_v88_combine", "_v88_masks_at",
        "_v88_pushpull_prepare", "_v88_pushpull_moved", "_v88_refill", "_v88_pushpull",
        "_v88_annotation_geometry", "_v88_update_annotations", "_v88_shift_undo_after_delete", "_V88CatState",
        "_segmentation_from_mask", "_mask_to_uncompressed_rle", "_mask_has_interior_holes"]
GLUE = ["_v88_select_mask", "_v88_mark_dirty", "_v88_sync_mask_annotation", "_v88_commit_mask",
        "_v88_set_category", "_v88_undo_category", "_v88_delete_mask", "_v88_lasso_commit", "_v88_lasso_new_mask",
        "_v88_cancel_stroke", "_v88_category_names", "_v88_mask_name",
        "_sync_label_to_current_mask", "update_annotation_in_cache", "delete_current_mask",
        "apply_label", "save_mask_and_contour_via2", "_snapshot_if_new_stroke"]


def make_ns(tmp):
    msg = types.SimpleNamespace(showinfo=_noop, showerror=_noop, showwarning=_noop,
                                askyesno=lambda *a, **k: True)
    ns = dict(np=np, cv2=cv2, os=os, json=json, logging=logging, time=time,
              tk=types.SimpleNamespace(), messagebox=msg, simpledialog=types.SimpleNamespace(askstring=_noop),
              root=types.SimpleNamespace(bell=_noop),
              canvas=types.SimpleNamespace(delete=_noop),
              segmentation_masks=[], labels={}, current_mask_index=0, mode="mask_edit",
              categories=[{"id": 1, "name": "head", "supercategory": "s"},
                          {"id": 2, "name": "eye", "supercategory": "s"}],
              category_name_to_id={"head": 1, "eye": 2}, category_id_to_name={1: "head", 2: "eye"},
              category_id_to_supercategory={1: "s", 2: "s"},
              coco_output_accumulate={"images": [], "annotations": [], "categories": []},
              coco_output_mask={"images": [], "annotations": [], "categories": []},
              selected_label=Var("Select Label"), selected_category_label=Var("Select Category"),
              selected_prediction_image=Var(""), prediction_cache={},
              file_path_base=os.path.join(tmp, "spec"), image=np.zeros((80, 120, 3), np.uint8),
              label_options=["head", "eye", "Custom"], category_label_options=[], instance_counters={},
              _DCHARS_AVAILABLE=False, _ONTOLOGY_AVAILABLE=False, _onto_mgr=None, dchar_ask_var=Var(False),
              scale_bar_info=None, folder_mask_cache={}, _UNDO_STACK=[], _UNDO_MAX=20,
              _stroke={"t": 0.0, "idx": None}, _v88={"lasso": [], "pending": None, "pp": None, "pp_idx": None},
              _V88_EXCLUDED_NAMES={"", "Custom", "Select Label", "Select Category", "unlabeled",
                                   "line_keypoints", "keypoints"},
              structure_status={}, category_occlusion={}, point_coords=[], point_orders=[], point_labels=[])
    for stub in ("apply_zoom", "redraw_masks", "update_label_dropdown", "update_category_label_dropdown",
                 "edit_descriptive_characters", "_attach_structure_status"):
        ns[stub] = _noop
    ns["_ask_parent_region_for_custom_category"] = lambda name: (None, None)
    ns["_image_file_name"] = lambda p: os.path.basename(str(p)) + ".jpg"
    code = "\n\n".join(src(n) for n in PURE + GLUE)
    exec(compile(code, V88, "exec"), ns)
    # v74 brush undo (original) behind the v88 wrapper
    exec(compile(src("undo_last_stroke", 0).replace("def undo_last_stroke", "def _v88_undo_prev"), V88, "exec"), ns)
    exec(compile(src("undo_last_stroke", -1), V88, "exec"), ns)
    # the real save (v70 body) behind the v88 wrapper
    exec(compile(src("save_final_via2_json", 0).replace("def save_final_via2_json", "def _v88_save_final_orig"),
                 V88, "exec"), ns)
    exec(compile(src("save_final_via2_json", -1), V88, "exec"), ns)
    return ns


RESULTS = []


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({info})" if info and not ok else ""))


def n_holes(mask):
    m8 = (mask > 0).astype(np.uint8)
    _, hier = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return 0 if hier is None else int(np.sum(hier[0][:, 3] != -1))


def ring_mask(h=200, w=200):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (100, 100), 60, 1, -1)
    cv2.circle(m, (100, 100), 25, 0, -1)       # hole
    cv2.circle(m, (100, 100), 8, 1, -1)        # island inside the hole
    return m


def decode_rle(rle):
    h, w = rle["size"]
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in rle["counts"]:
        flat[pos:pos + c] = val
        pos += c
        val = 1 - val
    return flat.reshape(w, h).T


def main():
    tmp = tempfile.mkdtemp(prefix="v88test_")
    ns = make_ns(tmp)

    # ---- 1. lasso rasterisation through the zoom/pan transform --------------------------------------------
    zoom = 2.5
    # a square drawn at canvas coords 100..200 (canvasx/canvasy already include the scroll offset)
    canvas_pts = [(100, 100), (200, 100), (200, 200), (100, 200)]
    img_pts = ns["_v88_canvas_points_to_image"](canvas_pts, zoom)
    check("canvas -> image transform divides by zoom", img_pts[0] == (40.0, 40.0) and img_pts[2] == (80.0, 80.0))
    poly = ns["_v88_polygon_to_mask"](img_pts, (120, 160, 3))
    ys, xs = np.nonzero(poly)
    check("lasso rasterised at full image resolution", poly.shape == (120, 160)
          and xs.min() == 40 and xs.max() == 80 and ys.min() == 40 and ys.max() == 80,
          f"{poly.shape} x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}")
    # scrolled view: event (10,10) with scroll offset (300,250) at zoom 2 is image (155,130)
    ip = ns["_v88_canvas_points_to_image"]([(300 + 10, 250 + 10)], 2.0)[0]
    check("panned view maps to the right image pixel", ip == (155.0, 130.0), str(ip))
    tiny = ns["_v88_polygon_to_mask"]([(1, 1), (5, 5)], (20, 20))
    check("fewer than 3 points rasterises to nothing", not tiny.any())
    # freehand (many points, concave)
    t = np.linspace(0, 2 * np.pi, 200)
    star = [(60 + (30 + 12 * np.cos(5 * a)) * np.cos(a), 60 + (30 + 12 * np.cos(5 * a)) * np.sin(a)) for a in t]
    sm = ns["_v88_polygon_to_mask"](star, (120, 120))
    check("freehand concave lasso fills a sensible area", 2000 < sm.sum() < 4500, str(sm.sum()))

    # ---- 2. add / cut keep dtype and foreground value ---------------------------------------------------------
    base = np.zeros((100, 100), np.uint8); base[20:60, 20:60] = 1
    lp = np.zeros((100, 100), np.uint8); lp[40:80, 40:80] = 1
    add = ns["_v88_combine"](base, lp, "add")
    cut = ns["_v88_combine"](base, lp, "cut")
    check("lasso add = union", add.sum() == 1600 + 1600 - 400 and add.dtype == np.uint8 and add.max() == 1)
    check("lasso cut = mask minus lasso", cut.sum() == 1600 - 400 and cut[50, 50] == 0 and cut[25, 25] == 1)
    b255 = base * 255
    check("0/255 mask stays 0/255", set(np.unique(ns["_v88_combine"](b255, lp, "add"))) == {0, 255})
    bb = base.astype(bool)
    check("bool mask stays bool", ns["_v88_combine"](bb, lp, "cut").dtype == bool)
    inner = np.zeros((100, 100), np.uint8); inner[30:50, 30:50] = 1
    holed = ns["_v88_combine"](base, inner, "cut")
    check("lasso cut inside makes a hole", n_holes(holed) == 1)

    # ---- 3. mask hit-test: smallest first, lines ignored -----------------------------------------------------
    big = np.zeros((100, 100), np.uint8); big[10:90, 10:90] = 1
    small = np.zeros((100, 100), np.uint8); small[40:60, 40:60] = 1
    line = np.zeros((100, 100), np.uint8); line[50, :] = 1
    masks = [{"segmentation": big}, {"segmentation": line, "is_line": True}, {"segmentation": small}]
    check("overlap: smallest mask first, line excluded", ns["_v88_masks_at"](masks, 50.4, 50.7) == [2, 0])
    check("click outside all masks -> none", ns["_v88_masks_at"](masks, 5, 5) == [])
    check("click off the image -> none", ns["_v88_masks_at"](masks, -3, 500) == [])

    # ---- 4. push / pull ---------------------------------------------------------------------------------------
    ring = ring_mask()
    same, ch = ns["_v88_pushpull"](ring, (160, 100), (0, 0), 15)
    check("zero drag leaves the mask (with hole + island) unchanged", not ch and np.array_equal(same, ring))
    prep = ns["_v88_pushpull_prepare"](ring, (160, 100), 15)
    check("contours keep the hole (RETR_CCOMP)", len(prep["pts"]) == 3 and int(np.sum(prep["hier"][:, 3] != -1)) == 1)
    exact = ns["_v88_refill"](prep["pts"], prep["hier"], ring.shape)
    check("re-fill from unmoved contours is pixel-exact", np.array_equal(exact > 0, ring > 0),
          f"{int(np.sum((exact > 0) != (ring > 0)))} px differ")
    pulled, ch = ns["_v88_pushpull"](ring, (160, 100), (12, 0), 15)
    a0, a1 = int(ring.sum()), int((pulled > 0).sum())
    check("pull outward grows the mask a little", ch and 0 < a1 - a0 < 600, f"{a0} -> {a1}")
    check("pulled edge reaches the drag target", pulled[100, 170] > 0 and ring[100, 170] == 0)
    pushed, ch = ns["_v88_pushpull"](ring, (160, 100), (-12, 0), 15)
    a2 = int((pushed > 0).sum())
    check("push inward shrinks the mask a little", ch and 0 < a0 - a2 < 600, f"{a0} -> {a2}")
    check("push/pull keeps the hole and the island", n_holes(pulled) == 1 and n_holes(pushed) == 1
          and pulled[100, 100] > 0 and pulled[100, 115] == 0)
    far = np.ones_like(ring, bool)
    yy, xx = np.ogrid[:200, :200]
    far[(xx - 160) ** 2 + (yy - 100) ** 2 <= (15 + 12 + 2) ** 2] = False
    check("pixels outside the influence circle are untouched", np.array_equal((pulled > 0)[far], (ring > 0)[far]))
    # pushing the inner hole edge (hole grows when its boundary is dragged towards the outside of the hole)
    hole_push, ch = ns["_v88_pushpull"](ring, (125, 100), (8, 0), 8)
    check("dragging a hole edge changes the hole", ch and int((hole_push > 0).sum()) < a0)
    nohit = ns["_v88_pushpull_prepare"](ring, (5, 5), 10)
    check("press far from any edge is not a hit", not nohit["hit"])
    check("dtype preserved by push/pull", pulled.dtype == ring.dtype and pulled.max() == 1)

    # ---- 5. category change through the real Apply Label path -----------------------------------------------
    m0 = np.zeros((80, 120), np.uint8); m0[10:40, 10:50] = 1
    m1 = np.zeros((80, 120), np.uint8); m1[45:70, 60:110] = 1
    ns["segmentation_masks"][:] = [{"segmentation": m0, "category_id": 1}, {"segmentation": m1, "category_id": 1}]
    acc = ns["coco_output_accumulate"]
    ns["_v88_set_category"](0, "head")
    ann0 = [a for a in acc["annotations"]]
    check("first category on an unlabelled mask -> one annotation", len(ann0) == 1 and ann0[0]["category_id"] == 1
          and ns["labels"][0] == "head" and ns["segmentation_masks"][0]["annotation_id"] == ann0[0]["id"])
    time.sleep(0.002)
    ns["_v88_set_category"](0, "eye")
    check("changing category replaces, no duplicate", len(acc["annotations"]) == 1
          and acc["annotations"][0]["category_id"] == 2 and ns["labels"][0] == "eye"
          and ns["segmentation_masks"][0]["category_id"] == 2
          and ns["segmentation_masks"][0]["annotation_id"] == acc["annotations"][0]["id"])
    ns["simpledialog"].askstring = lambda *a, **k: ("sclerite" if "upercategory" in str(a) else "gena")
    ns["_v88_set_category"](0, None)          # New category...
    gid = ns["category_name_to_id"].get("gena")
    check("New category... registers it everywhere", gid is not None
          and any(c["name"] == "gena" for c in ns["categories"])
          and any(c["name"] == "gena" for c in acc["categories"])
          and "gena" in ns["label_options"] and len(acc["annotations"]) == 1
          and acc["annotations"][0]["category_id"] == gid and ns["labels"][0] == "gena",
          str([c["name"] for c in ns["categories"]]))
    ns["_v88_set_category"](1, "head")
    check("second mask labelled separately", len(acc["annotations"]) == 2
          and sorted(a["category_id"] for a in acc["annotations"]) == sorted([gid, 1]))
    names = ns["_v88_category_names"]()
    check("menu category list = categories, no 'Custom'", names[:3] == ["head", "eye", "gena"] and "Custom" not in names)

    # undo category changes (Ctrl+Z wrapper)
    ns["undo_last_stroke"]()                  # undo mask 1 first labelling
    check("undo first labelling removes its annotation", len(acc["annotations"]) == 1
          and ns["segmentation_masks"][1].get("annotation_id") is None and 1 not in ns["labels"])
    ns["undo_last_stroke"]()                  # gena -> eye
    check("undo category change restores the previous one, no duplicate", len(acc["annotations"]) == 1
          and acc["annotations"][0]["category_id"] == 2 and ns["labels"][0] == "eye")

    # ---- 6. geometry edits are written into the same annotation ---------------------------------------------
    ns["current_mask_index"] = 0
    aid = ns["segmentation_masks"][0]["annotation_id"]
    lp = np.zeros((80, 120), np.uint8); lp[20:30, 20:30] = 1      # cut a hole inside mask 0
    ns["_v88"]["pending"] = lp
    ns["_v88_lasso_commit"]("cut")
    a = [x for x in acc["annotations"] if x["id"] == aid]
    check("lasso cut updates the existing annotation (same id)", len(acc["annotations"]) == 1 and len(a) == 1
          and a[0]["area"] == float(30 * 40 - 100) and a[0].get("segmentation_rle") is not None,
          str(a[0]["area"] if a else None))
    ns["undo_last_stroke"]()
    a = [x for x in acc["annotations"] if x["id"] == aid]
    check("Ctrl+Z restores the mask and its annotation", int(ns["segmentation_masks"][0]["segmentation"].sum()) == 1200
          and a[0]["area"] == 1200.0 and "segmentation_rle" not in a[0])
    lp2 = np.zeros((80, 120), np.uint8); lp2[10:40, 50:70] = 1
    ns["_v88"]["pending"] = lp2
    ns["_v88_lasso_commit"]("add")
    a = [x for x in acc["annotations"] if x["id"] == aid][0]
    check("lasso add updates bbox and area", a["bbox"] == [10, 10, 60, 30] and a["area"] == 1800.0, str(a["bbox"]))

    # lasso new mask: registered like any labelled mask
    ns["simpledialog"].askstring = _noop
    lp3 = np.zeros((80, 120), np.uint8); lp3[50:75, 5:30] = 1
    ns["_v88"]["pending"] = lp3
    ns["_v88_lasso_new_mask"]("eye")
    nm = ns["segmentation_masks"][-1]
    check("lasso new mask -> mask + annotation + label", len(ns["segmentation_masks"]) == 3
          and nm.get("annotation_id") is not None and ns["labels"][2] == "eye" and len(acc["annotations"]) == 2)
    ns["_v88"]["pending"] = lp3.copy()
    ns["_v88_lasso_new_mask"](None)          # New category... cancelled -> nothing added
    check("cancelled new category leaves no mask behind", len(ns["segmentation_masks"]) == 3
          and len(acc["annotations"]) == 2)

    # brush stroke on a labelled mask (dirty flag) is saved by the marmot save
    ns["current_mask_index"] = 2
    cv2.circle(ns["segmentation_masks"][2]["segmentation"], (17, 62), 3, 0, -1)   # what erase() does
    ns["_v88_mark_dirty"]()

    # delete via menu path shifts undo entries
    stack = [(0, "a"), (1, "b"), (2, "c")]
    ns["_v88_shift_undo_after_delete"](stack, 1)
    check("undo entries follow a deleted mask", stack == [(0, "a"), (1, "c")])

    # ---- 7. the marmot save writes the edited shapes (.json polygons + .rlejson RLE) -------------------------
    ns["coco_output_accumulate"]["images"].append({"id": ns["file_path_base"], "file_name": "spec.jpg",
                                                   "height": 80, "width": 120})
    ns["current_mask_index"] = 0
    lp4 = np.zeros((80, 120), np.uint8); lp4[20:30, 20:30] = 1
    ns["_v88"]["pending"] = lp4
    ns["_v88_lasso_commit"]("cut")
    expect0 = ns["segmentation_masks"][0]["segmentation"] > 0
    expect2 = ns["segmentation_masks"][2]["segmentation"] > 0
    aid2 = ns["segmentation_masks"][2]["annotation_id"]
    ns["save_final_via2_json"]()
    js = sorted(f for f in os.listdir(tmp) if f.endswith(".json") and "_final_" in f)
    rl = sorted(f for f in os.listdir(tmp) if f.endswith(".rlejson"))
    ok = len(js) == 1 and len(rl) == 1
    if ok:
        J = json.load(open(os.path.join(tmp, js[0])))
        R = json.load(open(os.path.join(tmp, rl[0])))
        ok_poly = len(J["annotations"]) == 2 and all(isinstance(x["segmentation"], list) for x in J["annotations"])
        r0 = [x for x in R["annotations"] if x["id"] == aid][0]
        dec = decode_rle(r0["segmentation"]) if isinstance(r0["segmentation"], dict) else None
        check("saved .json: one polygon annotation per mask", ok_poly, str(len(J["annotations"])))
        check("saved .rlejson: lasso-cut hole preserved exactly", dec is not None and np.array_equal(dec > 0, expect0))
        j2 = [x for x in J["annotations"] if x["id"] == aid2][0]
        check("saved .json: brush-edited labelled mask has its edited area", j2["area"] == float(expect2.sum()),
              f"{j2['area']} vs {expect2.sum()}")
    else:
        check("marmot save wrote .json and .rlejson", False, str(os.listdir(tmp)))

    n_pass = sum(ok for _, ok in RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} passed")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
