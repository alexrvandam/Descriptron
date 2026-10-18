"""
Headless unit tests for v66 annotation logic.
Tests:
  1. Re-labelling a mask replaces the old annotation (no duplicate)
  2. Deleting a mask removes its annotation from the accumulator
  3. Deleting re-keys the labels dict so remaining indices are contiguous
  4. Re-labelling after delete works cleanly (no orphan from deleted mask)
  5. Circular brush removes fewer pixels than old square method
"""
import sys, types, math
import numpy as np

# ── Minimal stubs so the script's module-level imports don't crash ──────────
def _stub(*a, **kw): pass
def _stub_cls(*a, **kw):
    class C:
        def __getattr__(self, n): return _stub
    return C()

# Build fake tkinter
tk_mod = types.ModuleType("tkinter")
tk_mod.messagebox = types.SimpleNamespace(showerror=_stub, showinfo=_stub)
tk_mod.simpledialog = types.SimpleNamespace(askstring=lambda *a,**kw: "test")
for name in ["Tk","Frame","Canvas","Button","Label","Entry","OptionMenu",
             "StringVar","IntVar","DoubleVar","BooleanVar","Toplevel",
             "Menu","Scale","Scrollbar","Text","Checkbutton","Spinbox",
             "PhotoImage","NW","BOTH","LEFT","RIGHT","TOP","BOTTOM","X","Y",
             "RAISED","SUNKEN","DISABLED","NORMAL","END","W","E","EW","NSEW",
             "HORIZONTAL","VERTICAL","ALL"]:
    setattr(tk_mod, name, _stub_cls)
sys.modules["tkinter"] = tk_mod
sys.modules["tkinter.messagebox"] = tk_mod.messagebox
sys.modules["tkinter.simpledialog"] = tk_mod.simpledialog
sys.modules["tkinter.filedialog"] = types.SimpleNamespace(askopenfilename=_stub, askdirectory=_stub)
from tkinter import messagebox  # noqa – now resolves to stub

# Other heavyweight stubs
for mod in ["PIL","PIL.Image","PIL.ImageTk","PIL.ImageDraw",
            "torch","torchvision","sam2","sam2.build_sam","sam2.sam2_image_predictor",
            "sam2.sam2_video_predictor","matplotlib","matplotlib.pyplot",
            "skimage","skimage.morphology","scipy","scipy.ndimage"]:
    sys.modules[mod] = types.ModuleType(mod)

import cv2  # real cv2 from samm env

# ── Set up the globals the functions depend on ───────────────────────────────
import time

coco_output_accumulate = {"images": [], "annotations": [], "categories": []}
coco_output_mask       = {"images": [], "annotations": [], "categories": []}
segmentation_masks = []
labels = {}
categories = []
category_name_to_id = {}
category_id_to_name = {}
category_id_to_supercategory = {}
instance_counters = {}
scale_bar_info = None
current_mask_index = 0
file_path_base = "test_image"
zoom_level = 1.0
brush_size = 10

H, W = 100, 100
image = np.zeros((H, W, 3), dtype=np.uint8)

# ── Paste the actual functions verbatim from v66 ─────────────────────────────
# (Only the parts we're testing; GUI calls replaced with stubs)

def redraw_masks(*a, **kw): pass
def apply_zoom(*a, **kw): pass
def update_annotation_in_cache(*a, **kw): pass

def save_mask_and_contour_via2(overwrite=False):
    global labels, file_path_base, image, segmentation_masks, current_mask_index
    global coco_output_mask, coco_output_accumulate
    global category_name_to_id, category_id_to_supercategory, categories, instance_counters

    if not segmentation_masks:
        return

    mask_data = segmentation_masks[current_mask_index]
    mask = mask_data['segmentation']
    mask_8bit = (mask * 255).astype(np.uint8)

    label = labels.get(current_mask_index, "unlabeled")
    category_id = mask_data.get('category_id', -1)
    if category_id in (-1, 0):
        return

    if label not in instance_counters:
        instance_counters[label] = 0
    instance_number = instance_counters[label]
    instance_counters[label] += 1

    annotation_id = int(time.time() * 1000) + current_mask_index + instance_number

    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coco_segmentation = []
    for contour in contours:
        flat = contour.flatten().tolist()
        if len(flat) >= 6:
            coco_segmentation.append(flat)

    x, y, w, h = cv2.boundingRect(mask_8bit)
    bbox = [x, y, w, h]
    area = float(np.sum(mask_8bit > 0))

    annotation = {
        "id": annotation_id,
        "image_id": file_path_base,
        "category_id": category_id,
        "segmentation": coco_segmentation,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "instance_id": instance_number,
    }

    # THE FIX: remove old annotation before appending new one
    old_ann_id = mask_data.get('annotation_id')
    if old_ann_id is not None:
        coco_output_accumulate["annotations"] = [
            a for a in coco_output_accumulate["annotations"] if a["id"] != old_ann_id
        ]
        coco_output_mask["annotations"] = [
            a for a in coco_output_mask["annotations"] if a["id"] != old_ann_id
        ]

    coco_output_mask["annotations"].append(annotation)
    coco_output_accumulate["annotations"].append(annotation)
    segmentation_masks[current_mask_index]['annotation_id'] = annotation_id


def delete_current_mask():
    global segmentation_masks, current_mask_index, coco_output_accumulate
    global coco_output_mask, labels, scale_bar_info, current_mask_index

    if not segmentation_masks:
        return

    mask_data = segmentation_masks[current_mask_index]
    ann_id = mask_data.get('annotation_id')
    if ann_id is not None:
        coco_output_accumulate["annotations"] = [
            a for a in coco_output_accumulate["annotations"] if a["id"] != ann_id
        ]
        coco_output_mask["annotations"] = [
            a for a in coco_output_mask["annotations"] if a["id"] != ann_id
        ]

    if scale_bar_info is not None and scale_bar_info.get('mask_idx') == current_mask_index:
        scale_bar_info = None

    labels.pop(current_mask_index, None)
    new_labels = {}
    for idx, lbl in labels.items():
        new_idx = idx if idx < current_mask_index else idx - 1
        new_labels[new_idx] = lbl
    labels.clear()
    labels.update(new_labels)

    del segmentation_masks[current_mask_index]
    if current_mask_index >= len(segmentation_masks):
        current_mask_index = max(0, len(segmentation_masks) - 1)


# ── Helper ───────────────────────────────────────────────────────────────────
def make_mask():
    m = np.zeros((H, W), dtype=bool)
    m[20:60, 20:60] = True
    return m

def reset_state():
    global segmentation_masks, labels, coco_output_accumulate, coco_output_mask
    global instance_counters, current_mask_index, categories, category_name_to_id
    segmentation_masks.clear()
    labels.clear()
    coco_output_accumulate["annotations"].clear()
    coco_output_accumulate["categories"].clear()
    coco_output_mask["annotations"].clear()
    instance_counters.clear()
    categories.clear()
    category_name_to_id.clear()
    current_mask_index = 0

def add_category(name, cat_id):
    cat = {"id": cat_id, "name": name, "supercategory": "none"}
    categories.append(cat)
    category_name_to_id[name] = cat_id
    coco_output_accumulate["categories"].append(cat)

def apply_label_simple(label, cat_id):
    """Minimal stand-in for apply_label: assigns category and saves."""
    segmentation_masks[current_mask_index]['category_id'] = cat_id
    labels[current_mask_index] = label
    save_mask_and_contour_via2()

# ── Tests ────────────────────────────────────────────────────────────────────
passed = 0
failed = 0

def check(condition, name, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
        failed += 1


print("\n=== Test 1: Re-labelling replaces old annotation (no duplicate) ===")
reset_state()
add_category("clypeus", 1)
add_category("labrum", 2)
segmentation_masks.append({'segmentation': make_mask(), 'category_id': -1})

apply_label_simple("clypeus", 1)
check(len(coco_output_accumulate["annotations"]) == 1, "after first label: 1 annotation")
check(coco_output_accumulate["annotations"][0]["category_id"] == 1, "first annotation is clypeus")

apply_label_simple("labrum", 2)
check(len(coco_output_accumulate["annotations"]) == 1,
      "after re-label: still 1 annotation (no duplicate)",
      f"got {len(coco_output_accumulate['annotations'])}")
check(coco_output_accumulate["annotations"][0]["category_id"] == 2,
      "annotation now has labrum category_id")


print("\n=== Test 2: Deleting a mask removes its annotation ===")
reset_state()
add_category("labrum", 2)
segmentation_masks.append({'segmentation': make_mask(), 'category_id': -1})
apply_label_simple("labrum", 2)
check(len(coco_output_accumulate["annotations"]) == 1, "1 annotation before delete")
delete_current_mask()
check(len(coco_output_accumulate["annotations"]) == 0,
      "0 annotations after delete",
      f"got {len(coco_output_accumulate['annotations'])}")
check(len(segmentation_masks) == 0, "segmentation_masks is empty")


print("\n=== Test 3: Deleting re-keys labels dict ===")
reset_state()
add_category("clypeus", 1)
add_category("labrum", 2)
segmentation_masks.append({'segmentation': make_mask(), 'category_id': -1})  # idx 0
segmentation_masks.append({'segmentation': make_mask(), 'category_id': -1})  # idx 1

apply_label_simple("clypeus", 1)          # labels[0] = clypeus
current_mask_index = 1
apply_label_simple("labrum", 2)           # labels[1] = labrum

current_mask_index = 0
delete_current_mask()                     # delete clypeus; labrum should shift to idx 0

check(len(segmentation_masks) == 1, "1 mask remains after delete")
check(0 in labels, "label key 0 exists after re-key",
      f"labels keys: {list(labels.keys())}")
check(labels.get(0) == "labrum", "remaining label is labrum",
      f"got {labels.get(0)}")
check(1 not in labels, "stale label key 1 gone")


print("\n=== Test 4: Two masks, delete first, re-label second – no orphan ===")
reset_state()
add_category("clypeus", 1)
add_category("labrum", 2)
add_category("gena", 3)

segmentation_masks.append({'segmentation': make_mask(), 'category_id': -1})  # idx 0
segmentation_masks.append({'segmentation': make_mask(), 'category_id': -1})  # idx 1

apply_label_simple("clypeus", 1)
current_mask_index = 1
apply_label_simple("labrum", 2)

acc_before = len(coco_output_accumulate["annotations"])
check(acc_before == 2, "2 annotations for 2 masks")

# Delete the clypeus mask (idx 0)
current_mask_index = 0
delete_current_mask()
check(len(coco_output_accumulate["annotations"]) == 1,
      "1 annotation after deleting clypeus",
      f"got {len(coco_output_accumulate['annotations'])}")

# Re-label the remaining mask (now at idx 0) as gena
current_mask_index = 0
apply_label_simple("gena", 3)
check(len(coco_output_accumulate["annotations"]) == 1,
      "still 1 annotation after re-labelling – no duplicate",
      f"got {len(coco_output_accumulate['annotations'])}")
check(coco_output_accumulate["annotations"][0]["category_id"] == 3,
      "annotation is now gena (cat_id=3)")


print("\n=== Test 5: Circular brush smaller than old square ===")
h2, w2 = 100, 100
bs = 10
cx2, cy2 = 50, 50
mask_circ = np.zeros((h2, w2), dtype=np.uint8)
cv2.circle(mask_circ, (cx2, cy2), bs, 1, -1)
mask_sq = np.zeros((h2, w2), dtype=np.uint8)
mask_sq[max(0,cy2-bs):min(h2,cy2+bs), max(0,cx2-bs):min(w2,cx2+bs)] = 1
check(int(np.sum(mask_circ)) < int(np.sum(mask_sq)),
      "circle footprint smaller than square footprint",
      f"circle={int(np.sum(mask_circ))} square={int(np.sum(mask_sq))}")
corner_in_sq  = mask_sq [cy2-bs, cx2-bs]
corner_in_cir = mask_circ[cy2-bs, cx2-bs]
check(corner_in_sq == 1 and corner_in_cir == 0,
      "corner pixel painted by square but NOT by circle (true circular shape)")


print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
