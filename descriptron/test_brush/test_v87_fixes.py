"""
Headless tests for the three annotation bugs fixed in descriptron-v2-v87.py.

The functions under test are taken VERBATIM from the GUI source (parsed with ast, no copy-paste), then run in a
namespace where Tk, the canvas and dialogs are stubs. Each test runs on v86 (must show the bug) and on v87
(must not).

  Bug 1  After Load Annotations the selection pointed at the LAST mask while the FIRST was drawn, so Trash
         removed a mask that was not on screen and Next wrapped back to the first mask.
  Bug 2  In Review predictions (prediction_view) Trash showed the next mask but the category indicator and the
         status-bar label still named the trashed / a stale mask.
  Bug 3  In a folder session, each image's Load Annotations wiped categories added earlier in the session; the
         new category vanished from the dropdown and re-creating it reused an id of another category.
  Bug 4  In Review predictions, saving an edited mask with Re-annotate reloaded the image and put the selection
         back on mask 1; it now stays on the edited mask (Replace and Keep Both), and Trash still works after it.

Run:  <python of the GUI env> test_brush/test_v87_fixes.py
"""
import ast, json, logging, os, sys, tempfile, types

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.dirname(HERE)
V86 = os.path.join(GUI_DIR, "descriptron-v2-v86.py")
V87 = os.path.join(GUI_DIR, "descriptron-v2-v87.py")

_SRC = {}


def _functions(path, names):
    """Source of the named top-level functions of a GUI file (only those that exist in it)."""
    if path not in _SRC:
        src = open(path).read()
        _SRC[path] = (src, {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)})
    src, defs = _SRC[path]
    return "\n\n".join(ast.get_source_segment(src, defs[n]) for n in names if n in defs)


class Var:                                   # tk.StringVar stand-in
    def __init__(self, v=""):
        self.v = v

    def set(self, v):
        self.v = v

    def get(self):
        return self.v


def _noop(*a, **k):
    return None


def _namespace(path, names):
    msg = types.SimpleNamespace(showinfo=_noop, showerror=_noop, showwarning=_noop,
                                askyesno=lambda *a, **k: True, askyesnocancel=lambda *a, **k: False)
    ns = dict(np=np, cv2=cv2, os=os, json=json, logging=logging, Image=Image, tk=types.SimpleNamespace(),
              messagebox=msg, filedialog=types.SimpleNamespace(askopenfilename=_noop),
              simpledialog=types.SimpleNamespace(askstring=_noop),
              canvas=types.SimpleNamespace(delete=_noop, config=_noop),
              segmentation_masks=[], labels={}, current_mask_index=0, mode="prompt",
              multi_mask_mode=False, folder_images=[], folder_coco={"path": None},
              point_coords=[], point_labels=[], point_orders=[], keypoint_labels=[], keypoint_visibility=[],
              keypoint_bank=[], category_occlusion={}, IMAGE_EXTS=(".jpg", ".png", ".tif"),
              categories=[], category_name_to_id={}, category_id_to_name={}, category_id_to_supercategory={},
              coco_output_accumulate={"images": [], "annotations": [], "categories": []},
              coco_output_mask={"images": [], "annotations": [], "categories": []},
              scale_bar_info=None, selected_label=Var("Select Label"), selected_category_label=Var("Select Category"),
              selected_prediction_image=Var(""), file_path_base=None, image=None,
              label_options=[], category_label_options=[], instance_counters={},
              _DCHARS_AVAILABLE=False, _ONTOLOGY_AVAILABLE=False, _onto_mgr=None, dchar_ask_var=Var(False),
              prediction_cache={}, prediction_image_names=[], pred_json_path=None, DRAWN=[])
    for stub in ("apply_zoom", "redraw_points", "update_label_dropdown", "update_category_label_dropdown",
                 "_load_structure_status", "load_json_keypoints_into_gui", "save_mask_and_contour_via2",
                 "edit_descriptive_characters", "_ask_parent_region_for_custom_category"):
        ns[stub] = _noop

    def redraw_masks(multi=False):           # records which mask the canvas shows
        if ns["segmentation_masks"]:
            ns["DRAWN"].append(ns["segmentation_masks"][ns["current_mask_index"]].get("name"))
    ns["redraw_masks"] = redraw_masks
    ns["apply_zoom"] = lambda *a, **k: ns["redraw_masks"]()   # the real apply_zoom redraws the current mask
    exec(compile(_functions(path, names), path, "exec"), ns)
    return ns


def _mask(x, h=60, w=120):
    m = np.zeros((h, w), np.uint8)
    m[10:30, x:x + 10] = 1
    return m


def _coco(tmp, n_masks=5):
    imgs, anns = [], []
    for k in range(2):
        fn = f"spec{k}.jpg"
        Image.fromarray(np.full((60, 120, 3), 90, np.uint8)).save(os.path.join(tmp, fn))
        imgs.append({"id": k + 1, "file_name": fn, "height": 60, "width": 120})
        for j in range(n_masks):
            x = 5 + 20 * j
            anns.append({"id": 100 * k + j + 1, "image_id": k + 1, "category_id": 1 + j % 2,
                         "segmentation": [[x, 10, x + 10, 10, x + 10, 30, x, 30]],
                         "bbox": [x, 10, 10, 20], "area": 200, "iscrowd": 0})
    p = os.path.join(tmp, "coco.json")
    json.dump({"images": imgs, "annotations": anns,
               "categories": [{"id": 1, "name": "head", "supercategory": "s"},
                              {"id": 2, "name": "eye", "supercategory": "s"}]}, open(p, "w"))
    return p


# --------------------------------------------------------------------------------------------- bug 1
def bug1_fixed(path):
    """Load Annotations, then Trash: the mask removed must be the one on screen."""
    ns = _namespace(path, ["load_annotations", "delete_current_mask", "_sync_label_to_current_mask",
                           "next_mask", "previous_mask"])

    def fake_load_json_contours(p):          # five labelled masks m0..m4
        ns["segmentation_masks"] = [{"segmentation": _mask(5 + 20 * i), "category_id": 1, "name": f"m{i}",
                                     "annotation_id": i} for i in range(5)]
        ns["labels"] = {i: f"label{i}" for i in range(5)}
    ns["load_json_contours"] = fake_load_json_contours
    ns["apply_label"] = _noop
    ns["load_annotations"]("x.json", quiet=True)
    on_screen = ns["DRAWN"][-1]
    idx_ok = ns["current_mask_index"] == 0 and ns["selected_label"].get() == "label0"
    ns["delete_current_mask"]()
    names_left = [m["name"] for m in ns["segmentation_masks"]]
    trashed_shown = on_screen not in names_left
    ns["next_mask"]()
    no_wrap = ns["DRAWN"][-1] == "m2"        # after trashing m0 (now m1 shown), Next shows m2
    return idx_ok and trashed_shown and no_wrap, dict(shown_before_trash=on_screen, left_after_trash=names_left)


# --------------------------------------------------------------------------------------------- bug 2
def bug2_fixed(path, tmp):
    """Review predictions: after Trash, the indicator names the mask now shown."""
    ns = _namespace(path, ["load_prediction_image", "delete_current_mask", "_sync_label_to_current_mask",
                           "next_mask", "previous_mask"])
    ns.update(image_dir=tmp, working_pred_json_path=_coco(tmp), labels={0: "stale_a", 1: "stale_b"},
              category_id_to_name={1: "head", 2: "eye"}, category_name_to_id={"head": 1, "eye": 2})
    _orig_rd = ns["redraw_masks"]

    def rd(multi=False):                     # name masks by category for the record
        for i, m in enumerate(ns["segmentation_masks"]):
            m.setdefault("name", ns["category_id_to_name"][m["category_id"]] + f"@{int(np.where(m['segmentation'].any(0))[0].min())}")
        _orig_rd(multi)
    ns["redraw_masks"] = rd
    ns["load_prediction_image"]("spec0.jpg")
    ok_load = ns["labels"].get(0) == "head"              # status bar "mask 1/5: <label>"
    ns["next_mask"]()                                    # mask 2 (eye); indicator -> eye
    ns["delete_current_mask"]()                          # mask 3 (head) is now shown at index 1
    shown = ns["segmentation_masks"][ns["current_mask_index"]]
    want = ns["category_id_to_name"][shown["category_id"]]
    ok_trash = (ns["selected_category_label"].get() == want and ns["labels"].get(ns["current_mask_index"]) == want)
    return ok_load and ok_trash, dict(shown=want, indicator=ns["selected_category_label"].get(),
                                      status_label=ns["labels"].get(ns["current_mask_index"]))


# --------------------------------------------------------------------------------------------- bug 3
def bug3_fixed(path, tmp):
    """Folder session: a category added on image 1 is still there on image 2, with the same id."""
    ns = _namespace(path, ["load_json_contours", "apply_label"])
    coco = _coco(tmp)
    ns.update(file_path_base=os.path.join(tmp, "spec0"), image=np.zeros((60, 120, 3), np.uint8))
    ns["load_json_contours"](coco)                       # image 1: file categories head=1, eye=2
    ns["coco_output_accumulate"]["categories"] = [dict(c) for c in ns["categories"]]
    for new in ("pronotum", "scape"):                    # add two new categories on image 1
        ns["selected_label"].set(new)
        ns["apply_label"](auto=True)
    ids_img1 = dict(ns["category_name_to_id"])
    ns["file_path_base"] = os.path.join(tmp, "spec1")    # Next image: folder_goto re-reads the same COCO file
    ns["load_json_contours"](coco)
    kept = all(ns["category_name_to_id"].get(n) == ids_img1[n] for n in ("pronotum", "scape"))
    ns["category_name_to_id"].pop("scape", None)         # worst case: the name is missing and is typed again
    ns["categories"][:] = [c for c in ns["categories"] if c["name"] != "scape"]
    ns["selected_label"].set("scape")
    ns["apply_label"](auto=True)
    acc_ids = [c["id"] for c in ns["coco_output_accumulate"]["categories"]]
    unique = len(acc_ids) == len(set(acc_ids))
    return kept and unique, dict(img1=ids_img1, img2=dict(ns["category_name_to_id"]), accumulator_ids=acc_ids)


# --------------------------------------------------------------------------------------------- bug 4
def bug4_fixed(path, tmp, replace=True):
    """Review predictions: edit mask 3, Re-annotate (save the edit) -> still on the edited mask; Trash still works."""
    ns = _namespace(path, ["load_prediction_image", "reannotate_masks", "delete_current_mask",
                           "_sync_label_to_current_mask", "next_mask", "previous_mask"])
    ns.update(image_dir=tmp, working_pred_json_path=_coco(tmp),
              category_id_to_name={1: "head", 2: "eye"}, category_name_to_id={"head": 1, "eye": 2})
    ns["messagebox"].askyesnocancel = lambda *a, **k: replace     # Yes = Replace, No = Keep Both
    ns["load_prediction_image"]("spec0.jpg")
    ns["next_mask"](); ns["next_mask"]()                           # mask 3 (index 2)
    ns["segmentation_masks"][2]["segmentation"][40:50, 45:55] = 1  # brush stroke below it
    ns["reannotate_masks"]()
    i = ns["current_mask_index"]
    m = ns["segmentation_masks"][i]["segmentation"]
    want_idx = 2 if replace else 5 + 2
    edited_shown = i == want_idx and m[45, 50] == 1 and m[20, 50] == 1
    cat = ns["category_id_to_name"][ns["segmentation_masks"][i]["category_id"]]
    indicator_ok = ns["selected_category_label"].get() == cat and ns["labels"].get(i) == cat
    n = len(ns["segmentation_masks"])
    ns["delete_current_mask"]()                                    # Trash in this mode still works
    j = ns["current_mask_index"]
    trash_ok = (len(ns["segmentation_masks"]) == n - 1 and j == min(i, n - 2) and ns["selected_category_label"].get()
                == ns["category_id_to_name"][ns["segmentation_masks"][j]["category_id"]])
    return edited_shown and indicator_ok and trash_ok, dict(index_after_save=i, wanted=want_idx,
                                                             indicator=ns["selected_category_label"].get(), trash_ok=trash_ok)


def main():
    results = []
    for label, fn in (("bug 1  trash removes the mask on screen", lambda p, t: bug1_fixed(p)),
                      ("bug 2  indicator follows the mask after trash (review)", bug2_fixed),
                      ("bug 3  session categories survive the next image", bug3_fixed),
                      ("bug 4  Re-annotate (Replace) keeps the edited mask", bug4_fixed),
                      ("bug 4  Re-annotate (Keep Both) keeps the edited mask", lambda p, t: bug4_fixed(p, t, False))):
        for path, want in ((V86, False), (V87, True)):
            with tempfile.TemporaryDirectory() as tmp:
                got, info = fn(path, tmp)
            ok = got == want
            results.append(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] {os.path.basename(path)}: {label}: "
                  f"{'fixed' if got else 'BUG reproduced'}  {info if not ok or not got else ''}")
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
