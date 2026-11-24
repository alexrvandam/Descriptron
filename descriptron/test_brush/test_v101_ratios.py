"""v101 Ratios button, tested in the REAL GUI booted with every window withdrawn (gui_boot_withdrawn):
  * the button exists, sits LAST in the Measure tab's SHAPE row and carries a tooltip; the other SHAPE buttons keep their places;
  * the v100 PhyloRAG button is still in the Describe tab's MATRIX row;
  * clicking it opens the Ratios dialog with the ant-head preset selected and Stop disabled.
Run:  <python of the GUI env> test_brush/test_v101_ratios.py"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.dirname(HERE)
V101 = os.path.join(GUI_DIR, "descriptron-v2-v101.py")


def _widgets(w):
    out = [w]
    for c in w.winfo_children():
        out += _widgets(c)
    return out


def child(out):
    sys.path.insert(0, HERE)
    import gui_boot_withdrawn as b
    import tkinter as tk
    g = b.boot(V101)
    res = {}
    btn = g.get("btn_ratios")
    res["button"] = btn is not None and btn.cget("text") == "Ratios"
    shape = [g.get(n) for n in ("measure_button", "semilandmark_button", "btn_centerline", "btn_joints")]
    row = btn.grid_info().get("in") if btn is not None else None
    res["same_row_as_semilandmarks"] = row is not None and g["semilandmark_button"].grid_info().get("in") == row
    cols = [w.grid_info().get("column") for w in shape + [btn]] if btn is not None else []
    res["last_in_row"] = bool(cols) and int(cols[-1]) == max(int(c) for c in cols)
    res["others_unchanged"] = [int(c) for c in cols[:-1]] == sorted(int(c) for c in cols[:-1])
    rows = g.get("_describe_rows") or []
    res["phylorag_kept"] = bool(rows) and g.get("btn_phylorag") in rows[-1].grid_slaves()
    res["tooltip"] = "btn_ratios" in g.get("_TIPS", {})
    before = set(g["root"].winfo_children())
    btn.invoke()
    new = [w for w in g["root"].winfo_children() if w not in before and isinstance(w, tk.Toplevel)]
    res["dialog"] = len(new) == 1 and new[0].title().startswith("Ratios")
    if new:
        ws = _widgets(new[0])
        res["preset"] = any(isinstance(w, tk.Menubutton) and "ant_head_frontal_v1" in str(w.cget("text")) for w in ws)
        ents = [w.get() for w in ws if isinstance(w, tk.Entry)]
        res["config_filled"] = any(e.endswith("ant_head_frontal_v1.tsv") for e in ents)
        stops = [w for w in ws if isinstance(w, tk.Button) and w.cget("text") == "Stop"]
        res["stop_disabled"] = bool(stops) and str(stops[0].cget("state")) == "disabled"
        new[0].destroy()
    res["messages"] = [m[0] for m in b.MESSAGES]
    json.dump(res, open(os.path.join(out, "result.json"), "w"))


def main():
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, __file__, "--child", d], capture_output=True, text=True, timeout=300)
        p = os.path.join(d, "result.json")
        if not os.path.exists(p):
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit("FAIL: the GUI did not boot")
        res = json.load(open(p))
    checks = [("button exists", res["button"]), ("in the SHAPE row", res["same_row_as_semilandmarks"]),
              ("last in the row", res["last_in_row"]), ("other SHAPE buttons in order", res["others_unchanged"]),
              ("PhyloRAG (v100) kept", res["phylorag_kept"]), ("tooltip", res["tooltip"]), ("dialog opens", res["dialog"]),
              ("ant-head preset selected", res.get("preset")), ("config filled from preset", res.get("config_filled")),
              ("Stop disabled", res.get("stop_disabled")), ("no error message", "showerror" not in res["messages"])]
    fails = 0
    for name, ok in checks:
        fails += not ok
        print(("PASS " if ok else "FAIL ") + name)
    if fails:
        print(json.dumps(res, indent=1))
    print(f"{len(checks) - fails}/{len(checks)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
