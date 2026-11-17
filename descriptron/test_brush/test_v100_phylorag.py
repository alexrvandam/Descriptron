"""v100 PhyloRAG button, tested in the REAL GUI booted with every window withdrawn (gui_boot_withdrawn):
  * the button exists, sits in the Describe tab (row MATRIX, after WRITE and EXPORT) and carries a tooltip;
  * the EXPORT row keeps its index (the 'Ontology: not available' note is placed by index);
  * clicking it opens the PhyloRAG dialog: four tabs (Characters, Literature, Extract, Review & export), model
    claude-code / claude-opus-5-5 and near-verbatim 0.90 preset, Stop disabled until a step runs.
Run:  <python of the GUI env> test_brush/test_v100_phylorag.py"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.dirname(HERE)
V100 = os.path.join(GUI_DIR, "descriptron-v2-v100.py")


def _widgets(w):
    out = [w]
    for c in w.winfo_children():
        out += _widgets(c)
    return out


def child(out):
    sys.path.insert(0, HERE)
    import gui_boot_withdrawn as b
    import tkinter as tk
    from tkinter import ttk
    g = b.boot(V100)
    res = {}
    btn = g.get("btn_phylorag")
    res["button"] = btn is not None and btn.cget("text") == "PhyloRAG"
    rows = g.get("_describe_rows") or []
    res["describe_rows"] = len(rows)
    # buttons stay children of scrollable_frame and are gridded INTO each row frame (in_=), so look at grid_slaves
    res["in_matrix_row"] = bool(rows) and btn is not None and btn in rows[-1].grid_slaves()
    res["export_row_index_kept"] = len(rows) >= 2 and all(w in rows[1].grid_slaves() for w in g.get("_export_row", []))
    res["write_row_unchanged"] = bool(rows) and all(w in rows[0].grid_slaves() for w in g.get("_describe_row", []))
    res["tooltip"] = "btn_phylorag" in g.get("_TIPS", {})
    before = set(g["root"].winfo_children())
    btn.invoke()
    new = [w for w in g["root"].winfo_children() if w not in before and isinstance(w, tk.Toplevel)]
    res["dialog"] = len(new) == 1 and new[0].title().startswith("PhyloRAG")
    if new:
        ws = _widgets(new[0])
        nb = [w for w in ws if isinstance(w, ttk.Notebook)]
        res["tabs"] = [nb[0].tab(i, "text") for i in range(nb[0].index("end"))] if nb else []
        ents = [w.get() for w in ws if isinstance(w, tk.Entry)]
        res["model_preset"] = "claude-opus-5-5" in ents and "0.90" in ents
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
    checks = [("button exists", res["button"]), ("three Describe rows", res["describe_rows"] == 3),
              ("button in the MATRIX row", res["in_matrix_row"]), ("EXPORT row keeps index 1", res["export_row_index_kept"]),
              ("WRITE row unchanged", res["write_row_unchanged"]),
              ("tooltip", res["tooltip"]), ("dialog opens", res["dialog"]),
              ("four tabs", res.get("tabs") == ["1 Characters", "2 Literature", "3 Extract", "4 Review & export"]),
              ("model and threshold preset", res.get("model_preset")), ("Stop disabled", res.get("stop_disabled")),
              ("no error message", "showerror" not in res["messages"])]
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
