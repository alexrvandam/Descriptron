"""v98 "Point type" tested in the REAL GUI, booted with every window withdrawn (nothing is shown on screen; Tk only
needs a display connection). Each GUI runs in its own child process:
  * v97 and v98 place the same landmark points and save (Save Keypoints + marmot): the v98 files must equal the
    v97 files apart from the new keypoint_types list (and the time-based annotation ids);
  * v98 full run: Point type box placement, typed points + drawing, remove / rename / move / visibility / undo /
    redo / hide-show / Show All, Save Keypoints, marmot, Load Annotations (v98 file, v97 file, two sets -> hidden
    set), folder session (auto-save, navigation, final save), persistence of the type list.
Run:  <python of the GUI env> test_brush/test_point_types_gui.py"""
import glob, json, os, subprocess, sys, tempfile, types as _types

HERE = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.dirname(HERE)
V97 = os.path.join(GUI_DIR, "descriptron-v2-v97.py")
V98 = os.path.join(GUI_DIR, "descriptron-v2-v98.py")


def _img(path, w=200, h=150):
    import numpy as np
    from PIL import Image
    a = (np.random.RandomState(0).rand(h, w, 3) * 255).astype("uint8")
    Image.fromarray(a).save(path)


def _ev(x, y):
    return _types.SimpleNamespace(x=x, y=y, type=None, widget=None)


def _cats(g):
    """A session normally has its categories once a mask is labelled; the marmot refuses to save without any."""
    for store in (g["coco_output_accumulate"], g["coco_output_mask"]):
        if not store.get("categories"):
            store.setdefault("categories", []).append({"id": 1, "name": "head", "supercategory": ""})


# ------------------------------------------------------------------------------------------------ child: landmark save
def child_landmark(gui, out):
    sys.path.insert(0, HERE)
    import gui_boot_withdrawn as b
    os.environ["DESCRIPTRON_POINT_TYPES"] = os.path.join(out, "pt_config.json")
    g = b.boot(gui)
    img = os.path.join(out, "spec.png"); _img(img)
    g["_display_image_path"](img)
    g["selected_point_label"].set("Positive")
    for x, y in ((10, 12), (40.5, 33.25), (90, 70)):
        g["pick_point"](_ev(x, y)); g["pick_point"](_ev(x, y))      # press + release, as the canvas binding does
    g["selected_point_label"].set("Negative")
    g["pick_point"](_ev(120, 100))
    b.ANSWERS["asksaveasfilename"] = os.path.join(out, "saved_keypoints.json")
    g["save_keypoints_now"]()
    _cats(g)
    g["save_and_close"]()
    finals = sorted(glob.glob(os.path.join(out, "spec_final_*.json")) + glob.glob(os.path.join(out, "spec_final_*.rlejson")))
    json.dump({"keypoints": os.path.join(out, "saved_keypoints.json"), "finals": finals}, open(os.path.join(out, "result.json"), "w"))


# ------------------------------------------------------------------------------------------------ child: full v98 run
def child_full(gui, out):
    sys.path.insert(0, HERE)
    import gui_boot_withdrawn as b
    cfg = os.path.join(out, "cfg", "descriptron", "point_types.json")
    os.environ["DESCRIPTRON_POINT_TYPES"] = cfg
    g = b.boot(gui)
    n = [0]
    fails = []

    def check(cond, name):
        n[0] += 1
        if not cond:
            fails.append(name)
            print("FAIL", name)

    def types():
        return list(g["keypoint_types"])

    def table():
        return [(tuple(c), t) for c, t in zip(g["point_coords"], g["keypoint_types"])]

    def aligned():
        L = len(g["point_coords"])
        return all(len(g[k]) == L for k in ("point_labels", "point_orders", "keypoint_types")) and \
            len(g["_sync_kp_vis"]()) == L

    def set_type(name):
        g["_v98_type_var"].set(name); g["_v98_commit_type"]()

    def markers():
        return len(g["canvas"].find_withtag("v98_ptype"))

    # -- UI --
    fr, dd = g["_v98_frame"], g["point_label_dropdown"]
    gi, gd = fr.grid_info(), dd.grid_info()
    check(str(gi["in"]) == str(gd["in"]) and int(gi["column"]) == int(gd["column"]) + 1, "box right of Positive/Negative")
    cols = sorted(int(w.grid_info()["column"]) for w in gi["in"].grid_slaves())
    check(len(cols) == len(set(cols)), "no two widgets share a column in the SAM2 row")
    check(g["_v98_type_var"].get() == "landmark" and list(g["_v98_combo"].cget("values")) == ["landmark"], "default landmark")
    check(all(ord(ch) < 128 for ch in g["_v98_label"].cget("text")), "ASCII label")

    img = os.path.join(out, "spec.png"); _img(img)
    g["_display_image_path"](img)
    g["selected_point_label"].set("Positive")
    for x, y in ((10, 10), (20, 20)):
        g["pick_point"](_ev(x, y)); g["pick_point"](_ev(x, y))
    check(types() == ["landmark", "landmark"] and markers() == 0, "landmarks drawn as before (no marker)")
    set_type("  seta ")
    check(g["_v98_current"]["t"] == "seta" and os.path.exists(cfg) and "seta" in json.load(open(cfg))["point_types"],
          "new type added + persisted (dirs created)")
    g["pick_point"](_ev(30, 30)); g["pick_point"](_ev(30, 30))
    g["selected_point_label"].set("Negative")
    g["pick_point"](_ev(40, 40))
    g["selected_point_label"].set("Positive")
    set_type("pit"); g["pick_point"](_ev(50, 50))
    set_type("landmark"); g["pick_point"](_ev(60, 60))
    check(types() == ["landmark", "landmark", "seta", "seta", "pit", "landmark"] and aligned(), "add: types recorded")
    check(g["point_labels"][3] == 0 and g["keypoint_types"][3] == "seta", "negative point keeps its type")
    check(markers() == 2 * 3, "ring + tag on each of the 3 typed points")
    check(g["point_orders"] == [1, 2, 3, 4, 5, 6], "numbering unchanged")
    texts = sorted(g["canvas"].itemcget(i, "text") for i in g["canvas"].find_withtag("v98_ptype")
                   if g["canvas"].type(i) == "text")
    check(texts == ["p", "s", "s"], "tags s / p")
    g["apply_zoom"]()
    check(markers() == 6, "markers survive a redraw / zoom")
    before = dict(table())

    # -- remove / rename / move / visibility --
    g["remove_keypoint"](2)
    check(types() == ["landmark", "landmark", "seta", "pit", "landmark"] and aligned()
          and all(before[k] == v for k, v in table()), "remove_keypoint")
    g["v96_undo"]()
    check(table() == list(before.items()) and aligned(), "undo remove brings the type back")
    g["v96_redo"]()
    check(types() == ["landmark", "landmark", "seta", "pit", "landmark"] and aligned(), "redo remove")
    g["v96_undo"]()
    b.ANSWERS["askstring"] = "1"
    g["rename_keypoint"](4)                                # pit -> slot 1
    check(types()[0] == "pit" and all(before[k] == v for k, v in table()) and aligned(), "rename reorders types")
    g["v96_undo"]()
    check(table() == list(before.items()), "undo rename")
    g["moving_point"] = 2
    g["stop_move_keypoint"](_ev(33, 37))
    check(table()[2][1] == "seta" and table()[2][0] != list(before)[2][0] and aligned(), "move keeps the type")
    g["v96_undo"]()
    check(table() == list(before.items()), "undo move")
    g["set_keypoint_visibility"](2, 0)
    check(types()[2] == "seta" and g["keypoint_visibility"][2] == 0, "visibility change keeps type")
    g["v96_undo"]()

    # -- hide / show --
    g["toggle_hide_keypoints"]()
    check(g["point_coords"] == [] and g["keypoint_bank"][-1]["types"] == [t for _, t in before.items()], "hide stores types")
    g["toggle_hide_keypoints"]()
    check(table() == list(before.items()) and markers() == 6, "show restores types + markers")
    g["toggle_hide_keypoints"]()                           # hide again, place a new set, Show All via the manager
    set_type("pore"); g["pick_point"](_ev(5, 90))
    g["toggle_hide_keypoints"]()                           # 2 sets -> manager popup (withdrawn)
    pops = [w for w in g["root"].winfo_children() if w.winfo_class() == "Toplevel"]
    btns = []
    def walk(w):
        for c in w.winfo_children():
            if c.winfo_class() == "Button":
                btns.append(c)
            walk(c)
    walk(pops[-1])
    [x for x in btns if x.cget("text") == "Show All"][0].invoke()
    check(types() == ["pore"] + [t for _, t in before.items()] and aligned(), "Show All keeps every set's types")
    g["v96_undo"]()                                        # undo Show All: one hidden + the pore point
    check(types() == ["pore"] and g["keypoint_bank"] and g["keypoint_bank"][-1].get("types") == [t for _, t in before.items()],
          "undo Show All")
    g["clear_points"]()
    check(types() == [] and aligned(), "clear points")
    g["v96_undo"]()
    check(types() == ["pore"], "undo clear")
    g["clear_points"](); g["keypoint_bank"].clear()

    # -- save keypoints + marmot --
    set_type("landmark")
    for (x, y), t in zip([(11, 11), (22, 22), (33, 33), (44, 44)], ["landmark", "seta", "seta", "pit"]):
        set_type(t); g["pick_point"](_ev(x, y))
    want = ["landmark", "seta", "seta", "pit"]
    saved_xy = [list(c) for c in g["point_coords"]]
    kp_path = os.path.join(out, "kp_save.json")
    b.ANSWERS["asksaveasfilename"] = kp_path
    g["save_keypoints_now"]()
    d = json.load(open(kp_path)); a = d["annotations"][0]
    check(a.get("keypoint_types") == want and len(a["keypoints_clicked"]) == 12, "Save Keypoints writes keypoint_types")
    acc = [x for x in g["coco_output_accumulate"]["annotations"] if "keypoints_clicked" in x]
    check(acc and acc[-1].get("keypoint_types") == want, "session annotation carries the list")
    g["point_prompt_mode"] = True                          # marmot also writes <image>_keypoints.json then
    _cats(g)
    g["save_and_close"]()
    g["point_prompt_mode"] = False
    finals = sorted(glob.glob(os.path.join(out, "spec_final_*.json")))
    rles = sorted(glob.glob(os.path.join(out, "spec_final_*.rlejson")))
    fa = [x for x in json.load(open(finals[-1]))["annotations"] if x.get("keypoints") and not x.get("source_line_category")]
    check(fa and all(x.get("keypoint_types") == want for x in fa), "marmot final .json has keypoint_types")
    ra = [x for x in json.load(open(rles[-1]))["annotations"] if x.get("keypoints")]
    check(ra and all(x.get("keypoint_types") == want for x in ra), "marmot .rlejson has keypoint_types")
    per = json.load(open(os.path.join(out, "spec_keypoints.json")))["annotations"][0]
    check(per.get("keypoint_types") == want, "<image>_keypoints.json has keypoint_types")
    check(g["point_coords"] == [] and types() == [] or aligned(), "after marmot reset")

    # -- load back --
    g["_display_image_path"](img)
    g["load_annotations"](finals[-1], quiet=True)
    check(types() == want and [list(map(float, c)) for c in g["point_coords"]] ==
          [list(map(float, c)) for c in saved_xy], "Load Annotations restores types")
    check(markers() == 6, "loaded typed points drawn")
    # a v97 file (no list) -> all landmark
    v97 = json.load(open(kp_path)); v97["annotations"][0].pop("keypoint_types")
    p97 = os.path.join(out, "v97_style.json"); json.dump(v97, open(p97, "w"), indent=4)
    g["clear_points"](); g["load_annotations"](p97, quiet=True)
    check(types() == ["landmark"] * 4 and len(g["point_coords"]) == 4 and markers() == 0, "v97 file loads as landmarks")
    # two sets in one file: the second goes to the hidden bank with its own types
    two = json.load(open(kp_path)); s2 = dict(two["annotations"][0])
    s2["id"] = 99; s2["keypoints"] = [70, 70, 2, 80, 80, 2]; s2["keypoints_clicked"] = [70, 70, 2, 80, 80, 2]
    s2["point_order"] = [1, 2]; s2["keypoint_status"] = ["present"] * 2; s2["keypoint_types"] = ["pore", "landmark"]
    s2["num_keypoints"] = 2
    two["annotations"].append(s2)
    p2 = os.path.join(out, "two_sets.json"); json.dump(two, open(p2, "w"), indent=4)
    g["clear_points"](); g["keypoint_bank"].clear(); g["load_annotations"](p2, quiet=True)
    check(types() == want and g["keypoint_bank"] and g["keypoint_bank"][-1].get("types") == ["pore", "landmark"],
          "second set -> hidden bank with its types")
    g["toggle_hide_keypoints"]()                           # 2 sets -> manager; Show Selected the hidden one
    pops = [w for w in g["root"].winfo_children() if w.winfo_class() == "Toplevel"]
    btns.clear(); walk(pops[-1])
    lbs = []
    def walk_lb(w):
        for c in w.winfo_children():
            if c.winfo_class() == "Listbox":
                lbs.append(c)
            walk_lb(c)
    walk_lb(pops[-1]); lbs[0].selection_set(1)
    [x for x in btns if x.cget("text") == "Show Selected"][0].invoke()
    check(types() == ["pore", "landmark"] and g["keypoint_bank"][-1].get("types") == want, "Show Selected swaps sets with types")
    # second argv JSON path = load_annotations(path, quiet=True) (same call as above); keypoints-into-gui loader:
    g["load_json_keypoints_into_gui"]([s2])
    check(types() == ["pore", "landmark"], "load_json_keypoints_into_gui restores types")
    g["load_json_keypoints_into_gui"]([v97["annotations"][0]])
    check(types() == ["landmark"] * 4, "load_json_keypoints_into_gui without list -> landmark")
    g["clear_points"](); g["keypoint_bank"].clear()

    # -- folder session --
    fd = os.path.join(out, "folder"); os.makedirs(fd)
    _img(os.path.join(fd, "a.png")); _img(os.path.join(fd, "b.png"))
    g["load_image_folder"](fd)
    set_type("seta"); g["pick_point"](_ev(15, 15)); set_type("landmark"); g["pick_point"](_ev(25, 25))
    g["folder_goto"](1)
    ak = json.load(open(os.path.join(fd, "a_keypoints.json")))["annotations"][0]
    check(ak.get("keypoint_types") == ["seta", "landmark"], "folder auto-save <image>_keypoints.json has types")
    check(types() == [], "next image starts empty")
    set_type("pit"); g["pick_point"](_ev(35, 35)); g["pick_point"](_ev(45, 45))
    g["folder_goto"](-1)
    check(types() == ["seta", "landmark"] and markers() == 2, "back to image a: types restored from the folder cache")
    _cats(g)
    g["save_and_close"]()
    ff = sorted(glob.glob(os.path.join(fd, "a_final_*.json")))
    fdat = json.load(open(ff[-1])) if ff else {"images": [], "annotations": []}
    fname = {im["id"]: os.path.basename(str(im.get("file_name"))) for im in fdat["images"]}
    anns = [x for x in fdat["annotations"] if fname.get(x.get("image_id")) in ("a.png", "b.png")]
    got = sorted(tuple(x.get("keypoint_types") or ["MISSING"]) for x in anns if x.get("keypoints"))
    check(got == sorted([("seta", "landmark"), ("pit", "pit")]), f"folder marmot save: each image's types ({got})")

    # -- persistence across sessions --
    check(g["_v98pt"].load_types(cfg) == ["landmark", "seta", "pit", "pore"], "type list persisted")
    json.dump({"n": n[0], "fails": fails}, open(os.path.join(out, "result.json"), "w"))


def _run(mode, gui, out):
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", mode, gui, out],
                       capture_output=True, text=True, timeout=900)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-6000:])
        raise SystemExit(f"child {mode} {os.path.basename(gui)} failed")
    return json.load(open(os.path.join(out, "result.json")))


def _strip(d, drop_types):
    d = json.loads(json.dumps(d))
    for a in d.get("annotations", []):
        a.pop("id", None)
        if drop_types:
            a.pop("keypoint_types", None)
    return d


def main():
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        print("SKIP: no display connection (Tk needs one; no window is shown)"); return
    passed = total = 0
    base = tempfile.mkdtemp(prefix="v98_pt_gui_")
    o97, o98, of = (os.path.join(base, k) for k in ("v97", "v98", "full"))
    for p in (o97, o98, of):
        os.makedirs(p)
    r97, r98 = _run("landmark", V97, o97), _run("landmark", V98, o98)
    k97, k98 = json.load(open(r97["keypoints"])), json.load(open(r98["keypoints"]))
    checks = [
        ("v97 Save Keypoints file has no keypoint_types", "keypoint_types" not in k97["annotations"][0]),
        ("v98 landmark-only Save Keypoints adds the list", k98["annotations"][0].get("keypoint_types") == ["landmark"] * 4),
        ("v98 Save Keypoints == v97 apart from the list (and time ids)", _strip(k98, True) == _strip(k97, False)),
        ("same field order (list appended last)", list(k98["annotations"][0])[:-1] == list(k97["annotations"][0])),
        ("both wrote final .json + .rlejson", len(r97["finals"]) == 2 and len(r98["finals"]) == 2),
    ]
    for f97, f98 in zip(r97["finals"], r98["finals"]):
        d97, d98 = json.load(open(f97)), json.load(open(f98))
        kp98 = [a for a in d98["annotations"] if a.get("keypoints")]
        checks.append((f"marmot {os.path.splitext(f98)[1]} == v97 apart from the list",
                       _strip(d98, True) == _strip(d97, False)))
        checks.append((f"marmot {os.path.splitext(f98)[1]} landmark list present",
                       kp98 and all(a.get("keypoint_types") == ["landmark"] * 4 for a in kp98)))
    for name, ok in checks:
        total += 1; passed += bool(ok)
        print(("ok  " if ok else "FAIL") + " " + name)
    rf = _run("full", V98, of)
    total += rf["n"]; passed += rf["n"] - len(rf["fails"])
    for f in rf["fails"]:
        print("FAIL", f)
    print(f"full v98 GUI run: {rf['n'] - len(rf['fails'])} / {rf['n']}")
    print(f"{passed} / {total} point-type GUI checks passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        {"landmark": child_landmark, "full": child_full}[sys.argv[2]](sys.argv[3], sys.argv[4])
    else:
        main()
