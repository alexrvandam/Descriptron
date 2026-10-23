"""DINOMask dialog for the Descriptron GUI (v94, 21 Oct 2026).

DINOMask (dinomask/dinomask_train_predict_v4.py, dinomask/dinomask_predict_rot4_v1.py) trains a small per-patch
classifier on frozen DINOv3 ViT-L tokens of a few hand-annotated images and predicts every structure on the rest.
Until v94 it could only be run from the command line. This module holds the dialog and the pure functions behind it
(command building, interpreter choice, vocabulary), which the tests call without a display.

Three modes:
  train_predict  train on a COCO file, then predict a folder of images   -> predictions_polygons.json (opens in the GUI)
  holdout        train on a COCO file with some images held out          -> per-part IoU on those images (results.json)
  predict_run    predict a folder with an already trained run (rot4)     -> predictions_polygons.json
"""
import json
import os
import sys
from pathlib import Path

MODEL_VITL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MODEL_URL = "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"
NOT_PARTS = {"line_keypoints", "Select Label", "Trash", "keypoints"}
AUGS = ("rot90", "rot180", "rot270", "hflip")


def script_dir(gui_dir):
    """dinomask/ beside the GUI (source tree, Docker) or in the descriptron-vision package (pip)."""
    d = Path(gui_dir) / "dinomask"
    if (d / "dinomask_train_predict_v4.py").is_file():
        return d
    try:
        from descriptron_vision.cli import tools_dir
        d = Path(tools_dir()) / "dinomask"
        if (d / "dinomask_train_predict_v4.py").is_file():
            return d
    except Exception:
        pass
    return None


def _version_ok(v, need=(4, 56)):
    try:
        parts = tuple(int(x) for x in v.split(".")[:2])
        return parts >= need
    except Exception:
        return False


def dinov3_python(gui_python=None, environ=None):
    """(python, why). DINOv3 needs transformers >= 4.56. Order: DESCRIPTRON_DINOV3_PYTHON; this GUI's own Python
    when its transformers is new enough (pip installs); a sibling conda env 'biorag' (the source-tree setup, where
    the GUI env 'samm' carries an older transformers); otherwise this Python (the run log then names the problem)."""
    environ = os.environ if environ is None else environ
    gui_python = gui_python or sys.executable
    p = environ.get("DESCRIPTRON_DINOV3_PYTHON", "").strip()
    if p:
        return p, "DESCRIPTRON_DINOV3_PYTHON"
    try:
        from importlib.metadata import version
        if _version_ok(version("transformers")):
            return gui_python, "this Python (transformers >= 4.56)"
    except Exception:
        pass
    exe = Path(gui_python).resolve()
    for envs in (exe.parents[2] if len(exe.parents) > 2 else None, exe.parents[1] if len(exe.parents) > 1 else None):
        if envs is None:
            continue
        for cand in (envs / "biorag" / "bin" / "python", envs / "biorag" / "python.exe"):
            if cand.is_file():
                return str(cand), "conda env biorag (transformers >= 4.56)"
    return gui_python, "this Python (transformers may be too old for DINOv3; see the run log)"


def coco_summary(coco_path):
    """(image file_names, part names with >= 1 mask, n masks) of a COCO file; parts exclude keypoint / helper categories."""
    d = json.load(open(coco_path))
    names = {c["id"]: c["name"] for c in d.get("categories", [])}
    parts, n = set(), 0
    for a in d.get("annotations", []):
        if a.get("segmentation") and not a.get("keypoints"):
            nm = names.get(a.get("category_id"), "")
            if nm and nm not in NOT_PARTS:
                parts.add(nm); n += 1
    return [im.get("file_name", "") for im in d.get("images", [])], sorted(parts), n


def images_root(coco_path, file_names, folder=None):
    """--images for the training script: '/' when every file_name is absolute; else the folder that holds them
    (the COCO file's own folder first, then the GUI's current folder)."""
    if file_names and all(os.path.isabs(f) for f in file_names):
        return "/"
    for base in [os.path.dirname(os.path.abspath(coco_path))] + ([folder] if folder else []):
        if base and file_names and sum(os.path.exists(os.path.join(base, f)) for f in file_names) == len(file_names):
            return base
    return os.path.dirname(os.path.abspath(coco_path))


def count_found(root, file_names):
    return sum(1 for f in file_names if os.path.exists(f if os.path.isabs(f) or root == "/" else os.path.join(root, f)))


def build_cmd(v, sdir, python):
    """v: dict with mode, coco, images, out, model, size, epochs, augs (list), partial (bool), parts (list),
    predict_dir, test_list (list of file_names), run_dir, fg_border (bool), extra (str)."""
    sdir = Path(sdir)
    mode = v["mode"]
    if mode == "predict_run":
        cmd = [python, "-u", str(sdir / "dinomask_predict_rot4_v1.py"), "--run", v["run_dir"],
               "--images", v["predict_dir"], "--out", v["out"], "--model", v.get("model") or MODEL_VITL,
               "--size", str(int(v.get("size") or 2048) or 2048)]
    else:
        cmd = [python, "-u", str(sdir / "dinomask_train_predict_v4.py"), "--coco", v["coco"], "--images", v["images"],
               "--out", v["out"], "--model", v.get("model") or MODEL_VITL, "--size", str(int(v.get("size", 2048))),
               "--epochs", str(int(v.get("epochs") or 30)), "--containers", "auto", "--min_count", "1",
               "--species_regex", "^$", "--drop", "Select Label,line_keypoints,Trash"]
        augs = [a for a in v.get("augs", []) if a in AUGS]
        if augs:
            cmd += ["--aug", ",".join(augs)]
        if v.get("partial") and v.get("parts"):
            cmd += ["--partial_bg", ",".join(v["parts"])]
        if mode == "holdout":
            cmd += ["--test_list", ",".join(v["test_list"])]
        else:
            cmd += ["--predict_dir", v["predict_dir"]]
            if v.get("fg_border"):
                cmd += ["--fg_mask", "border"]
    if v.get("extra", "").strip():
        cmd += v["extra"].split()
    return cmd


def check(v):
    """Error text, or '' when the values can run."""
    m = v.get("mode")
    if not v.get("out"):
        return "Choose an output folder."
    if m in ("train_predict", "holdout"):
        if not v.get("coco") or not os.path.isfile(v["coco"]):
            return "Choose the training annotations (COCO JSON with hand-drawn masks)."
        if not v.get("parts"):
            return "The training file has no masks (only keypoints or empty)."
        if v.get("n_found", 1) == 0:
            return f"None of the training images was found under {v.get('images')}. Set the images folder."
    if m == "holdout" and not v.get("test_list"):
        return "Pick at least one image to hold out."
    if m in ("train_predict", "predict_run") and (not v.get("predict_dir") or not os.path.isdir(v["predict_dir"])):
        return "Choose the folder of images to predict."
    if m == "predict_run" and not os.path.isfile(os.path.join(v.get("run_dir", ""), "head.pt")):
        return "The trained run folder must hold head.pt and results.json (the output folder of an earlier training)."
    return ""


def summarize(out_dir, mode):
    """Short text for the end-of-run message."""
    out = Path(out_dir)
    lines = []
    r = out / "results.json"
    if mode == "holdout" and r.is_file():
        res = json.load(open(r))
        ious = {c: v[0] for c, v in res.get("ious", {}).items()}
        if ious:
            lines.append(f"Held-out patch IoU, mean over {len(ious)} parts: {sum(ious.values()) / len(ious):.2f}")
            lines += [f"  {c}: {v:.2f}" for c, v in sorted(ious.items(), key=lambda kv: -kv[1])]
        lines.append(f"Overlays (truth | prediction): {out}/overlay_*.png")
    pp = out / "predictions_polygons.json"
    if pp.is_file():
        d = json.load(open(pp))
        lines.append(f"{len(d.get('images', []))} images, {len(d.get('annotations', []))} masks -> {pp}")
        ab = out / "absent_parts.json"
        if ab.is_file():
            n = sum(1 for v in json.load(open(ab)).values() if v)
            if n:
                lines.append(f"Parts reported absent on {n} images (absent_parts.json)")
    return "\n".join(lines) or f"Output in {out}"


# ------------------------------------------------------------------ Tk dialog
def open_dialog(root, gui_dir, folder=None, coco=None, open_results=None, ensure_token=None, tooltip=None):
    """open_results(image_folder, predictions_json) loads the predictions into the editor;
    ensure_token(model_id, parent) makes HF_TOKEN available (GUI's key store / prompt)."""
    import subprocess
    import tkinter as tk
    from tkinter import filedialog, messagebox

    sdir = script_dir(gui_dir)
    if sdir is None:
        messagebox.showerror("DINOMask", "The DINOMask scripts (dinomask/) were not found beside the GUI or in the "
                                         "descriptron-vision package.")
        return
    win = tk.Toplevel(root)
    win.title("DINOMask - masks for every structure from a few annotated images")
    V = {k: tk.StringVar() for k in ("coco", "images", "predict_dir", "out", "run_dir", "extra", "test_list")}
    V["coco"].set(coco or ""); V["predict_dir"].set(folder or "")
    mode = tk.StringVar(value="train_predict"); model = tk.StringVar(value=MODEL_VITL)
    size = tk.StringVar(value="2048"); epochs = tk.StringVar(value="30")
    aug = {a: tk.BooleanVar(value=a in ("rot180", "hflip")) for a in AUGS}
    partial = tk.BooleanVar(value=True); fg = tk.BooleanVar(value=False)
    info = tk.StringVar()
    state = {"files": [], "parts": [], "n": 0}

    def refresh(*_):
        p = V["coco"].get().strip()
        if p and os.path.isfile(p):
            try:
                files, parts, n = coco_summary(p)
            except Exception as e:
                info.set(f"cannot read: {e}"); return
            state.update(files=files, parts=parts, n=n)
            if not V["images"].get().strip():
                V["images"].set(images_root(p, files, folder))
            found = count_found(V["images"].get().strip(), files)
            info.set(f"{len(files)} images ({found} found), {n} masks, {len(parts)} parts: " + ", ".join(parts[:14])
                     + (" ..." if len(parts) > 14 else ""))
            if not V["out"].get().strip():
                V["out"].set(os.path.join(os.path.dirname(os.path.abspath(p)), "dinomask_run"))
        else:
            info.set("")
    V["coco"].trace_add("write", refresh)

    r = 0
    tk.Label(win, text="What to do:", font=("Arial", 9, "bold")).grid(row=r, column=0, sticky="e", padx=5, pady=4)
    mf = tk.Frame(win); mf.grid(row=r, column=1, columnspan=2, sticky="w")
    for val, txt in (("train_predict", "Train, then predict a folder"), ("holdout", "Test: hold out images, report IoU"),
                     ("predict_run", "Predict with a trained run")):
        tk.Radiobutton(mf, text=txt, variable=mode, value=val, command=lambda: show()).pack(side="left", padx=4)
    r += 1
    rows = {}

    def row(key, label, var, browse=None):
        nonlocal r
        w = [tk.Label(win, text=label), tk.Entry(win, textvariable=var, width=60)]
        w[0].grid(row=r, column=0, sticky="e", padx=5, pady=2); w[1].grid(row=r, column=1, sticky="w", padx=5)
        if browse:
            b = tk.Button(win, text="Browse", command=browse); b.grid(row=r, column=2, padx=5); w.append(b)
        rows[key] = w; r += 1

    row("coco", "Training annotations (COCO JSON):", V["coco"],
        lambda: V["coco"].set(filedialog.askopenfilename(parent=win, filetypes=[("COCO JSON", "*.json"), ("All", "*.*")]) or V["coco"].get()))
    tk.Label(win, textvariable=info, fg="gray30", wraplength=560, justify="left").grid(row=r, column=1, columnspan=2, sticky="w", padx=5)
    rows["info"] = []; r += 1
    row("images", "Training images folder:", V["images"],
        lambda: (V["images"].set(filedialog.askdirectory(parent=win) or V["images"].get()), refresh()))
    row("run_dir", "Trained run folder (head.pt):", V["run_dir"],
        lambda: V["run_dir"].set(filedialog.askdirectory(parent=win) or V["run_dir"].get()))
    row("predict_dir", "Images to predict (folder):", V["predict_dir"],
        lambda: V["predict_dir"].set(filedialog.askdirectory(parent=win) or V["predict_dir"].get()))

    def pick_holdout():
        if not state["files"]:
            messagebox.showinfo("DINOMask", "Choose the training annotations first.", parent=win); return
        t = tk.Toplevel(win); t.title("Images to hold out")
        lb = tk.Listbox(t, selectmode="multiple", width=70, height=min(20, len(state["files"])))
        cur = set(x for x in V["test_list"].get().split(",") if x)
        for i, f in enumerate(state["files"]):
            lb.insert("end", f)
            if f in cur:
                lb.selection_set(i)
        lb.pack(padx=6, pady=6)
        tk.Button(t, text="OK", command=lambda: (V["test_list"].set(",".join(state["files"][i] for i in lb.curselection())),
                                                 t.destroy())).pack(pady=4)
    row("test_list", "Hold out (test) images:", V["test_list"], pick_holdout)
    rows["test_list"][-1].configure(text="Pick...")
    row("out", "Output folder:", V["out"], lambda: V["out"].set(filedialog.askdirectory(parent=win) or V["out"].get()))
    row("model", "DINOv3 model:", model)
    tk.Label(win, text=f"ViT-L/16 (as in all reported results). Gated: request access once at {MODEL_URL}",
             fg="gray30").grid(row=r, column=1, columnspan=2, sticky="w", padx=5); r += 1
    of = tk.Frame(win); of.grid(row=r, column=1, columnspan=2, sticky="w", padx=5, pady=3)
    tk.Label(of, text="Resolution (longest side):").pack(side="left")
    tk.OptionMenu(of, size, "2048", "1344", "0").pack(side="left")
    tk.Label(of, text="(0 = native: sharper small parts, more time and memory)").pack(side="left")
    r += 1
    af = tk.Frame(win); af.grid(row=r, column=1, columnspan=2, sticky="w", padx=5)
    tk.Label(af, text="Training copies:").pack(side="left")
    for a_ in AUGS:
        tk.Checkbutton(af, text=a_, variable=aug[a_]).pack(side="left")
    tk.Label(af, text="  Epochs:").pack(side="left"); tk.Entry(af, textvariable=epochs, width=4).pack(side="left")
    rows["aug"] = [af]; r += 1
    pc = tk.Checkbutton(win, variable=partial, text="Some training images have only some parts drawn (do not teach the "
                                                    "missing parts as background)")
    pc.grid(row=r, column=1, columnspan=2, sticky="w", padx=5); rows["partial"] = [pc]; r += 1
    fc = tk.Checkbutton(win, variable=fg, text="Plain background: nothing outside the specimen is a part (--fg_mask border)")
    fc.grid(row=r, column=1, columnspan=2, sticky="w", padx=5); rows["fg"] = [fc]; r += 1
    row("extra", "Extra arguments:", V["extra"])
    py, why = dinov3_python()
    tk.Label(win, text=f"Runs with {py}  [{why}]. A GPU is strongly advised.", fg="gray40", wraplength=600,
             justify="left").grid(row=r, column=0, columnspan=3, sticky="w", padx=8, pady=4); r += 1

    def show():
        m = mode.get()
        vis = {"coco": m != "predict_run", "info": m != "predict_run", "images": m != "predict_run",
               "run_dir": m == "predict_run", "predict_dir": m != "holdout", "test_list": m == "holdout",
               "aug": m != "predict_run", "partial": m != "predict_run", "fg": m == "train_predict"}
        for k, ws in rows.items():
            for w in ws:
                (w.grid() if vis.get(k, True) else w.grid_remove())
        if m == "predict_run" and size.get() == "0":
            size.set("2048")

    def submit():
        v = {k: x.get().strip() for k, x in V.items()}
        v.update(mode=mode.get(), model=model.get().strip(), size=size.get(), epochs=epochs.get(),
                 augs=[a_ for a_ in AUGS if aug[a_].get()], partial=partial.get(), fg_border=fg.get(),
                 parts=state["parts"], test_list=[x for x in v["test_list"].split(",") if x])
        if v["mode"] != "predict_run":
            v["n_found"] = count_found(v["images"], state["files"])
        err = check(v)
        if err:
            messagebox.showerror("DINOMask", err, parent=win); return
        if ensure_token:
            try:
                ensure_token(v["model"], win)
            except Exception:
                pass
        cmd = build_cmd(v, sdir, py)
        os.makedirs(v["out"], exist_ok=True)
        log = os.path.join(v["out"], "dinomask_run.log")
        fh = open(log, "w"); fh.write(" ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n\n"); fh.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(sdir), env=env)
        win.destroy()
        prog = tk.Toplevel(root); prog.title("DINOMask running")
        last = tk.StringVar(value="starting ...")
        tk.Label(prog, text=f"DINOMask ({v['mode']}) - log: {log}", wraplength=620, justify="left").pack(padx=12, pady=(10, 4))
        tk.Label(prog, textvariable=last, fg="gray30", wraplength=620, justify="left").pack(padx=12, pady=4)
        tk.Button(prog, text="Stop", command=lambda: proc.terminate()).pack(pady=(4, 10))

        def poll():
            try:
                tail = [l for l in open(log, errors="replace").read().splitlines() if l.strip()]
                last.set(tail[-1][-200:] if tail else "")
            except Exception:
                pass
            if proc.poll() is None:
                prog.after(2000, poll); return
            fh.close()
            try:
                prog.destroy()
            except Exception:
                pass
            if proc.returncode != 0:
                try:
                    t = "".join(open(log, errors="replace").readlines()[-20:])
                except Exception:
                    t = ""
                messagebox.showerror("DINOMask error", f"Exit code {proc.returncode}. Last lines of {log}:\n\n{t}")
                return
            msg = summarize(v["out"], v["mode"])
            pp = os.path.join(v["out"], "predictions_polygons.json")
            if v["mode"] != "holdout" and os.path.isfile(pp) and open_results:
                if messagebox.askyesno("DINOMask finished", msg + "\n\nOpen the predicted masks in the editor now?"):
                    open_results(v["predict_dir"], pp)
            else:
                messagebox.showinfo("DINOMask finished", msg)
        prog.after(2000, poll)

    bf = tk.Frame(win); bf.grid(row=r, column=0, columnspan=3, pady=8)
    tk.Button(bf, text="Run", command=submit, bg="lightgreen", width=10).pack(side="left", padx=6)
    tk.Button(bf, text="Cancel", command=win.destroy, width=10).pack(side="left", padx=6)
    refresh(); show()
    return win
