"""v99 scrollable label picker, tested in the REAL GUI booted with every window withdrawn (gui_boot_withdrawn):
  * Select Label and Select Category open the picker instead of the OptionMenu's own menu (which cannot scroll);
  * 120 structure names: the list is sorted alphabetically, 18 rows high, scrollable, without Custom/Trash;
  * Custom, Trash and Cancel stay at the bottom as buttons;
  * choosing a name, Custom or Trash sets the same variable the menu entry sets; Cancel changes nothing;
  * the filter box narrows the list; Enter with one match picks it.
Run:  <python of the GUI env> test_brush/test_v99_label_picker.py"""
import json, os, random, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.dirname(HERE)
V99 = os.path.join(GUI_DIR, "descriptron-v2-v99.py")


def _widgets(w):
    out = [w]
    for c in w.winfo_children():
        out += _widgets(c)
    return out


def child(out):
    sys.path.insert(0, HERE)
    import gui_boot_withdrawn as b
    import tkinter as tk
    g = b.boot(V99)
    res = {}
    names = [f"structure_{i:03d}" for i in range(120)] + ["Antenna", "vertex", "Zygoma"]
    random.Random(1).shuffle(names)
    g["categories"][:] = [{"id": i + 1, "name": n, "supercategory": "none"} for i, n in enumerate(names)]
    g["update_label_dropdown"]()
    om, var = g["label_dropdown"], g["selected_label"]
    res["click_bound"] = bool(om.bind("<Button-1>"))
    menu_labels = [om["menu"].entrycget(i, "label") for i in range(om["menu"].index("end") + 1)]
    res["menu_tail"] = menu_labels[-2:]

    def picker():
        win = g["_v99_open_label_picker"](om)
        ws = _widgets(win)
        lb = [w for w in ws if isinstance(w, tk.Listbox)][0]
        btns = [w for w in ws if isinstance(w, tk.Button)]
        ent = [w for w in ws if isinstance(w, tk.Entry)][0]
        return win, lb, btns, ent

    win, lb, btns, ent = picker()
    items = list(lb.get(0, "end"))
    res["n_items"] = len(items)
    res["sorted"] = items == sorted(items, key=lambda s: (s.casefold(), s))
    res["first3"] = items[:3]
    res["no_pinned_in_list"] = not ({"Custom", "Trash"} & set(items))
    res["height"] = int(lb.cget("height"))
    res["has_scrollbar"] = any(isinstance(w, tk.Scrollbar) for w in _widgets(win))
    res["buttons"] = [w.cget("text") for w in btns]
    # pick a name from the list
    lb.selection_clear(0, "end"); lb.selection_set(items.index("vertex"))
    win._v99_pick()
    res["after_list_pick"] = var.get()
    res["closed_after_pick"] = not win.winfo_exists()
    # Custom button: same variable the menu entry sets ("Custom" prompt itself is in apply_label, unchanged)
    win, lb, btns, ent = picker()
    [w for w in btns if w.cget("text") == "Custom"][0].invoke()
    res["after_custom"] = var.get()
    # Cancel: nothing changes
    win, lb, btns, ent = picker()
    [w for w in btns if w.cget("text") == "Cancel"][0].invoke()
    res["after_cancel"] = var.get()
    res["closed_after_cancel"] = not win.winfo_exists()
    # filter + Enter
    win, lb, btns, ent = picker()
    ent.insert(0, "zyg")
    res["filtered"] = list(lb.get(0, "end"))
    win._v99_pick()
    res["after_filter_enter"] = var.get()
    # Trash button
    win, lb, btns, ent = picker()
    [w for w in btns if w.cget("text") == "Trash"][0].invoke()
    res["after_trash"] = var.get()
    # prediction category dropdown
    g["category_label_options"][:] = ["wing", "Antenna", "head"]
    g["update_category_label_dropdown"]()
    com, cvar = g["category_label_dropdown"], g["selected_category_label"]
    res["cat_click_bound"] = bool(com.bind("<Button-1>"))
    cw = g["_v99_open_label_picker"](com)
    clb = [w for w in _widgets(cw) if isinstance(w, tk.Listbox)][0]
    res["cat_items"] = list(clb.get(0, "end"))
    clb.selection_clear(0, "end"); clb.selection_set(2); cw._v99_pick()
    res["cat_after_pick"] = cvar.get()
    json.dump(res, open(os.path.join(out, "result.json"), "w"))


def main():
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, __file__, "--child", d], capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-3000:]
        res = json.load(open(os.path.join(d, "result.json")))
    checks = [
        ("Select Label click opens the picker", res["click_bound"]),
        ("menu still ends with Trash, Custom (unchanged)", res["menu_tail"] == ["Trash", "Custom"]),
        ("123 names in the list", res["n_items"] == 123),
        ("list sorted alphabetically (case-insensitive)", res["sorted"] and res["first3"][0] == "Antenna"),
        ("Custom/Trash not in the scrolling list", res["no_pinned_in_list"]),
        ("18 rows high + scrollbar", res["height"] == 18 and res["has_scrollbar"]),
        ("bottom buttons: Custom, Trash, Cancel", res["buttons"] == ["Custom", "Trash", "Cancel"]),
        ("picking from the list sets the label", res["after_list_pick"] == "vertex" and res["closed_after_pick"]),
        ("Custom button", res["after_custom"] == "Custom"),
        ("Cancel changes nothing", res["after_cancel"] == "Custom" and res["closed_after_cancel"]),
        ("filter narrows the list", res["filtered"] == ["Zygoma"]),
        ("Enter with one match picks it", res["after_filter_enter"] == "Zygoma"),
        ("Trash button", res["after_trash"] == "Trash"),
        ("Select Category click opens the picker", res["cat_click_bound"]),
        ("prediction categories sorted", res["cat_items"] == ["Antenna", "head", "wing"]),
        ("prediction category picked", res["cat_after_pick"] == "wing"),
    ]
    fails = 0
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
        fails += not ok
    if fails:
        print(json.dumps(res, indent=1))
    print(f"{len(checks) - fails}/{len(checks)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
