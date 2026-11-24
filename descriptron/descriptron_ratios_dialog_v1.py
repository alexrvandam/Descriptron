"""Ratios dialog for the Descriptron GUI (v101, 17 Nov 2026).

Runs measure/descriptron_landmark_ratios_v2.py: measurements and taxonomic indices (e.g. ant CI = HW/HL x 100, OI = EL/HW x 100)
from the landmarks and masks of a COCO file, with the tests that say whether they separate species (Kruskal-Wallis with false
discovery rate, Dunn pairwise, leave-one-out species ID with permutations, PERMANOVA + PERMDISP, agreement with a coded character,
optional allometry check). Measurements are defined by a ratio config (TSV; ready-made ones in measure/ratio_configs/) and/or typed
in the dialog, one NAME=TERM per line, e.g.
    HL=9-10            HW=11-12^9-10            EL=mean(1-2|9-10;3-4|9-10)            CI=HW/HL*100
This module holds the dialog and the pure functions behind it (command building, checks), which the tests call without a display.
"""
import glob
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

PROGRAM = "descriptron_landmark_ratios_v2.py"
SYNTAX = ("Landmark terms (1-based numbers): A-B distance | A-B|C-D along axis C-D | A-B^C-D across axis C-D | "
          "P>A-B distance of P from line A-B (P>A-B@R signed, + towards R) | mean(t1;t2).  Mask terms: cat:length, cat:width, "
          "cat:width^A-B, cat:length|A-B, cat:area, cat:feret.  Ratios: NAME=NUM/DEN or NUM/DEN*100.")


def script_path(gui_dir):
    """measure/descriptron_landmark_ratios_v2.py beside the GUI (source tree, Docker) or in descriptron-core (pip)."""
    p = Path(gui_dir) / "measure" / PROGRAM
    if p.is_file():
        return p
    try:
        from descriptron_core.cli import tools_dir
        p = Path(tools_dir()) / PROGRAM
        if p.is_file():
            return p
    except Exception:
        pass
    return None


def presets(gui_dir):
    """Ready-made ratio configs: measure/ratio_configs/*.tsv (name -> path)."""
    s = script_path(gui_dir)
    d = (s.parent if s else Path(gui_dir) / "measure") / "ratio_configs"
    return {os.path.splitext(os.path.basename(p))[0]: p for p in sorted(glob.glob(str(d / "*.tsv")))}


def python_for(environ=None):
    """numpy/scipy/scikit-learn/pycocotools/matplotlib: the GUI's own Python, unless DESCRIPTRON_RATIOS_PYTHON names another."""
    environ = os.environ if environ is None else environ
    return environ.get("DESCRIPTRON_RATIOS_PYTHON", "").strip() or sys.executable


def _lines(s):
    return [x.strip() for x in str(s or "").replace(";;", "\n").splitlines() if x.strip() and not x.strip().startswith("#")]


def build_cmd(v, script, python):
    """Command line from the dialog's values (strings and booleans)."""
    c = [python, "-u", str(script), "--json", v["json"], "--out_dir", v["out_dir"]]
    if v.get("config"):
        c += ["--config", v["config"]]
    for x in _lines(v.get("measures")):
        c += ["--measure", x]
    for x in _lines(v.get("ratios")):
        c += ["--ratio", x]
    for key, flag in (("keypoint_category", "--keypoint_category"), ("group_labels", "--group_labels"),
                      ("group_regex", "--group_regex"), ("px_per_mm", "--px_per_mm"), ("size_measure", "--size_measure"),
                      ("character", "--character"), ("positive", "--positive"), ("loo_on", "--loo_on"),
                      ("permutations", "--permutations")):
        if str(v.get(key, "")).strip():
            c += [flag, str(v[key]).strip()]
    if v.get("assume_same_scale"):
        c.append("--assume_same_scale")
    return c


LABELS = {"json": "COCO file", "out_dir": "output folder", "config": "ratio config", "group_labels": "group labels",
          "px_per_mm": "px/mm table", "character": "character table"}


def check(v):
    """Error text, or '' when the values can run."""
    for k in ("json", "out_dir"):
        if not v.get(k):
            return f"Choose the {LABELS[k]}."
    if not v.get("config") and not _lines(v.get("measures")):
        return "Choose a ratio config (or a preset), or type at least one measure (NAME=TERM)."
    for k in ("json", "config", "group_labels", "px_per_mm", "character"):
        if v.get(k) and not os.path.isfile(v[k]):
            return f"The {LABELS[k]} was not found:\n{v[k]}"
    for x in _lines(v.get("measures")) + _lines(v.get("ratios")):
        if "=" not in x:
            return f'"{x}" needs the form NAME=DEFINITION.'
    if v.get("positive") and not v.get("character"):
        return "The positive label needs a character table."
    try:
        if v.get("permutations") and int(v["permutations"]) < 0:
            raise ValueError
    except ValueError:
        return "Permutations must be a whole number (0 = none)."
    if v.get("size_measure") and not (v.get("px_per_mm") or v.get("assume_same_scale")):
        return "The allometry check needs real sizes: a px/mm table, or tick 'all images at one magnification'."
    return ""


def open_file(path):
    if sys.platform.startswith("win"):
        os.startfile(path)                                     # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_dialog(root, gui_dir, coco=None, tooltip=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    script = script_path(gui_dir)
    if script is None:
        messagebox.showerror("Ratios", f"measure/{PROGRAM} was not found beside the GUI or in descriptron-core.")
        return None
    py = python_for()
    win = tk.Toplevel(root)
    win.title("Ratios - measurements and indices from landmarks and masks")
    S = {k: tk.StringVar() for k in ("json", "keypoint_category", "preset", "config", "group_labels", "group_regex", "px_per_mm",
                                     "size_measure", "character", "positive", "loo_on", "permutations", "out_dir")}
    B = {"assume_same_scale": tk.BooleanVar()}
    S["group_regex"].set(r"^([^_]+)_"); S["loo_on"].set("ratios"); S["permutations"].set("999")
    if coco:
        S["json"].set(coco)
        S["out_dir"].set(os.path.join(os.path.dirname(coco), "ratios"))
    P = presets(gui_dir)
    state = {"proc": None, "q": queue.Queue()}
    f = tk.Frame(win)
    f.pack(fill="x", padx=8, pady=6)
    r = [0]

    def browse(var, kind, title, types=None):
        def go():
            if kind == "dir":
                p = filedialog.askdirectory(parent=win, title=title)
            else:
                p = filedialog.askopenfilename(parent=win, title=title, filetypes=types or [("All", "*.*")])
            if p:
                var.set(p)
        return go

    def row(label, var, kind=None, types=None, tip=None, width=60):
        tk.Label(f, text=label).grid(row=r[0], column=0, sticky="e", padx=5, pady=2)
        e = tk.Entry(f, textvariable=var, width=width)
        e.grid(row=r[0], column=1, sticky="w", padx=5, pady=2)
        if kind:
            tk.Button(f, text="Browse", command=browse(var, kind, label, types)).grid(row=r[0], column=2, padx=4)
        if tip and tooltip:
            try:
                tooltip(e, tip)
            except Exception:
                pass
        r[0] += 1
        return e

    def head(text):
        tk.Label(f, text=text, font=("TkDefaultFont", 9, "bold")).grid(row=r[0], column=0, columnspan=3, sticky="w", padx=4, pady=(8, 1))
        r[0] += 1

    JS = [("COCO", "*.json"), ("All", "*.*")]
    TAB = [("Tables", "*.tsv *.csv *.txt"), ("All", "*.*")]
    head("Data")
    row("COCO file:", S["json"], "file", JS, "Descriptron save with one landmark set per image and/or the masks the terms name")
    row("Landmark set (category):", S["keypoint_category"], tip="blank = the first keypoint set of each image (e.g. line_keypoints)")
    head("Measurements and ratios")
    tk.Label(f, text="Preset:").grid(row=r[0], column=0, sticky="e", padx=5)
    names = ["(none)"] + list(P)
    S["preset"].set(names[1] if len(names) > 1 else names[0])

    def _preset(*_):
        S["config"].set(P.get(S["preset"].get(), ""))
    tk.OptionMenu(f, S["preset"], *names, command=lambda _v: _preset()).grid(row=r[0], column=1, sticky="w", padx=5)
    r[0] += 1
    _preset()
    row("Ratio config (TSV):", S["config"], "file", TAB, "columns kind (measure|ratio), name, definition, description")
    tk.Label(f, text="More measures:").grid(row=r[0], column=0, sticky="ne", padx=5)
    tm = tk.Text(f, height=3, width=60)
    tm.grid(row=r[0], column=1, sticky="w", padx=5, pady=2)
    r[0] += 1
    tk.Label(f, text="More ratios:").grid(row=r[0], column=0, sticky="ne", padx=5)
    tr = tk.Text(f, height=2, width=60)
    tr.grid(row=r[0], column=1, sticky="w", padx=5, pady=2)
    r[0] += 1
    tk.Label(f, text=SYNTAX, fg="gray35", wraplength=640, justify="left").grid(row=r[0], column=0, columnspan=3, sticky="w", padx=8)
    r[0] += 1
    head("Species and tests")
    row("Group labels (CSV):", S["group_labels"], "file", TAB, "filename,group_label (the file the semilandmark GPA uses)")
    row("or group from file name (regex):", S["group_regex"], width=30)
    tk.Label(f, text="Leave-one-out on:").grid(row=r[0], column=0, sticky="e", padx=5)
    tk.OptionMenu(f, S["loo_on"], "ratios", "measures", "both").grid(row=r[0], column=1, sticky="w", padx=5)
    r[0] += 1
    row("Permutations:", S["permutations"], width=8)
    row("Compare with character (TSV):", S["character"], "file", TAB, "taxon, label: e.g. a published character matrix column")
    row("  positive label:", S["positive"], width=30, tip="binary character: the state counted as positive (blank = all states)")
    head("Size (optional)")
    row("px/mm (CSV file,px_per_mm):", S["px_per_mm"], "file", TAB, "measures also in mm; needed for the allometry check")
    row("Size measure for allometry:", S["size_measure"], width=12, tip="e.g. HL; each ratio is checked for a size effect")
    tk.Checkbutton(f, text="all images at one magnification (pixel sizes comparable)", variable=B["assume_same_scale"]).grid(
        row=r[0], column=1, sticky="w", padx=5)
    r[0] += 1
    head("Output")
    row("Output folder:", S["out_dir"], "dir")

    lf = tk.Frame(win)
    lf.pack(fill="both", expand=True, padx=8)
    log = tk.Text(lf, height=14, width=110)
    sb = tk.Scrollbar(lf, command=log.yview)
    log.configure(yscrollcommand=sb.set)
    log.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    status = tk.StringVar(value=f"Runs with {py}")
    bf = tk.Frame(win)
    bf.pack(fill="x", padx=8, pady=(2, 8))
    tk.Label(bf, textvariable=status, fg="gray30").pack(side="left")
    tk.Button(bf, text="Close", command=win.destroy, width=8).pack(side="right", padx=4)
    stop_btn = tk.Button(bf, text="Stop", state="disabled", width=8, command=lambda: state["proc"] and state["proc"].terminate())
    stop_btn.pack(side="right", padx=4)

    def values():
        v = {k: x.get().strip() for k, x in S.items()}
        v.update({k: x.get() for k, x in B.items()})
        v["measures"] = tm.get("1.0", "end")
        v["ratios"] = tr.get("1.0", "end")
        return v

    def write(text):
        log.insert("end", text)
        log.see("end")

    def open_out(name):
        p = os.path.join(values()["out_dir"], name)
        if os.path.exists(p):
            open_file(p)
        else:
            messagebox.showinfo("Ratios", f"Not there yet: {p}", parent=win)

    def start():
        if state["proc"] is not None and state["proc"].poll() is None:
            messagebox.showinfo("Ratios", "Still running.", parent=win)
            return
        v = values()
        err = check(v)
        if err:
            messagebox.showerror("Ratios", err, parent=win)
            return
        cmd = build_cmd(v, script, py)
        write("\n$ " + " ".join(f'"{c}"' if (" " in c or "|" in c or "^" in c) else c for c in cmd) + "\n")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except OSError as e:
            messagebox.showerror("Ratios", f"Could not start: {e}", parent=win)
            return
        state["proc"] = proc
        stop_btn.configure(state="normal")
        status.set("running ...")

        def reader():
            for line in proc.stdout:
                state["q"].put(line)
            state["q"].put(None)
        threading.Thread(target=reader, daemon=True).start()
        win.after(200, poll)

    def poll():
        done = False
        try:
            while True:
                line = state["q"].get_nowait()
                if line is None:
                    done = True
                    break
                write(line)
        except queue.Empty:
            pass
        if not done:
            win.after(300, poll)
            return
        state["proc"].wait()
        stop_btn.configure(state="disabled")
        ok = state["proc"].returncode == 0
        status.set("finished: summary.txt, specimen_measurements.tsv, species_summary.tsv, pairwise_dunn.tsv" if ok
                   else f"stopped (exit code {state['proc'].returncode})")

    run = tk.Frame(win)
    run.pack(fill="x", padx=8, pady=(0, 4), before=lf)
    for text, cmd, colour in (("Run", start, "lightgreen"), ("Open summary", lambda: open_out("summary.txt"), "lightyellow"),
                              ("Open species table", lambda: open_out("species_summary.tsv"), "lightyellow"),
                              ("Open plot", lambda: open_out("ratios_by_group.png"), "lightyellow")):
        tk.Button(run, text=text, command=cmd, bg=colour, width=max(10, len(text) + 2)).pack(side="left", padx=4)
    return win
