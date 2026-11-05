"""v99 colour segmentation dialog tested in the REAL GUI, booted with every window withdrawn (gui_boot_withdrawn):
  * default (recolorize ticked): the command runs FHS_and_CLAHE_V22.py with --recolorize, cutoff 20, 8 bins, average
    linkage, and none of the Descriptron-method switches (Felzenszwalb, LAB averaging, dE threshold mode);
  * recolorize unticked: the V21-style Descriptron command (Felzenszwalb + CLAHE + LAB averaging + median dE), on V22;
  * the cutoff box accepts 'median';
  * colour and texture homology dialogs: '--landmarks auto' by default (keypoints join the semilandmarks as TPS control
    points), '--landmarks none' when the box is unticked, '--landmark_json' when a file is given.
The command is captured, not run. Each case runs in its own child process.
Run:  <python of the GUI env> test_brush/test_v99_recolorize.py"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.dirname(HERE)
V99 = os.path.join(GUI_DIR, "descriptron-v2-v99.py")


def _widgets(w):
    out = [w]
    for c in w.winfo_children():
        out += _widgets(c)
    return out


def child(case, out):
    sys.path.insert(0, HERE)
    import gui_boot_withdrawn as b
    import tkinter as tk
    g = b.boot(V99)
    seen = []
    g["subprocess"].run = lambda cmd, *a, **k: seen.append(cmd)

    class _SyncThread:
        def __init__(self, target=None, *a, **k): self.t = target
        def start(self): self.t()
    g["threading"].Thread = _SyncThread
    before = set(g["root"].winfo_children())
    if case == "slmdlg":
        fn = [k for k in g if k.startswith("run_semi") and callable(g[k])][0]
        g[fn]()
        win = [w for w in g["root"].winfo_children() if w not in before and isinstance(w, tk.Toplevel)][-1]
        ws = _widgets(win)
        for e in [w for w in ws if isinstance(w, tk.Entry) and not isinstance(w, tk.Spinbox)][:3]:
            if not e.get().strip():
                e.insert(0, os.path.join(out, "x"))
        btn = [w for w in ws if isinstance(w, tk.Button) and str(w.cget("text")).lower().startswith("run")][0]
        btn.invoke()
        json.dump({"cmds": seen}, open(os.path.join(out, "result.json"), "w"))
        return
    if case.startswith(("colhom", "texhom")):
        g["run_color_homology_script" if case.startswith("colhom") else "run_texture_homology_script"]()
        win = [w for w in g["root"].winfo_children() if w not in before and isinstance(w, tk.Toplevel)][-1]
        ws = _widgets(win)
        ents = [w for w in ws if isinstance(w, tk.Entry) and not isinstance(w, tk.Spinbox)]
        for e in ents[:4]:
            if not e.get().strip():
                e.insert(0, os.path.join(out, "x"))
        if case.endswith("_off"):
            [w for w in ws if isinstance(w, tk.Checkbutton) and str(w.cget("text")).startswith("Use landmarks")][0].deselect()
        if case.endswith("_file"):
            lab = [w for w in ws if isinstance(w, tk.Label) and str(w.cget("text")).startswith("Landmark JSON")][0]
            e = [w for w in ws if isinstance(w, tk.Entry) and int(w.grid_info().get("row", -1)) == int(lab.grid_info()["row"])][0]
            e.insert(0, os.path.join(out, "kp.json"))
        [w for w in ws if isinstance(w, tk.Button) and w.cget("text") == "Run Analysis"][0].invoke()
        json.dump({"cmds": seen}, open(os.path.join(out, "result.json"), "w"))
        return
    g["run_color_segmentation_script"]()
    win = [w for w in g["root"].winfo_children() if w not in before and isinstance(w, tk.Toplevel)][-1]
    ws = _widgets(win)
    entries = [w for w in ws if isinstance(w, tk.Entry) and not isinstance(w, tk.Spinbox)]
    entries[0].delete(0, "end"); entries[0].insert(0, os.path.join(out, "in"))
    entries[1].delete(0, "end"); entries[1].insert(0, os.path.join(out, "out"))
    rc = [w for w in ws if isinstance(w, tk.Checkbutton) and str(w.cget("text")).startswith("recolorize")][0]
    if case == "descriptron":
        rc.deselect()
    if case == "norm":
        [w for w in ws if isinstance(w, tk.Checkbutton) and str(w.cget("text")).startswith("Normalise first")][0].select()
    if case == "median":
        cb = [w for w in ws if w.winfo_class() == "TCombobox" and w.get() == "20"][0]
        cb.set("median")
    btn = [w for w in ws if isinstance(w, tk.Button) and w.cget("text") == "Run Analysis"][0]
    btn.invoke()
    json.dump({"cmds": seen}, open(os.path.join(out, "result.json"), "w"))


def run_case(case):
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, __file__, "--child", case, d], capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-3000:]
        return json.load(open(os.path.join(d, "result.json")))["cmds"]


def main():
    fails = 0
    cmds = run_case("recolorize")
    c = cmds[0] if cmds else ""
    checks = [("V22 script", "FHS_and_CLAHE_V22.py" in c), ("--recolorize", " --recolorize " in c),
              ("cutoff 20", "--recolorize_cutoff 20" in c), ("8 bins", "--recolorize_bins 8" in c),
              ("average", "--recolorize_linkage average" in c), ("no --run_segmentation", "--run_segmentation" not in c),
              ("no --run_lab_average", "--run_lab_average" not in c), ("no --median_threshold", "--median_threshold" not in c),
              ("no normalisation by default", "--bg_normalize" not in c and "--run_normalization" not in c)]
    cmds = run_case("descriptron")
    c = cmds[0] if cmds else ""
    checks += [("descriptron: V22", "FHS_and_CLAHE_V22.py" in c), ("descriptron: no --recolorize", "--recolorize" not in c),
               ("descriptron: FHS", "--run_segmentation" in c), ("descriptron: CLAHE", "--run_normalization" in c),
               ("descriptron: LAB averaging", "--run_lab_average" in c), ("descriptron: median dE", "--median_threshold" in c)]
    cmds = run_case("median")
    c = cmds[0] if cmds else ""
    checks += [("median cutoff", "--recolorize_cutoff median" in c)]
    sl = (run_case("slmdlg") or [""])[0]
    checks += [("semilandmark dialog: --slide_method bending by default", "--slide_method bending" in sl)]
    c = (run_case("norm") or [""])[0]
    checks += [("normalise first", "--recolorize " in c and "--bg_normalize" in c and "--run_normalization" in c)]
    for tool in ("colhom", "texhom"):
        c = (run_case(tool) or [""])[0]
        checks += [(f"{tool}: --landmarks auto by default", "--landmarks auto" in c)]
        c = (run_case(tool + "_off") or [""])[0]
        checks += [(f"{tool}: unticked -> --landmarks none", "--landmarks none" in c)]
        c = (run_case(tool + "_file") or [""])[0]
        checks += [(f"{tool}: --landmark_json passed", "--landmark_json" in c and "kp.json" in c)]
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
        fails += not ok
    print(f"{len(checks) - fails}/{len(checks)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
