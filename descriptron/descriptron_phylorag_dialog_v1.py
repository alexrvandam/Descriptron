"""PhyloRAG dialog for the Descriptron GUI (v100, 17 Nov 2026).

PhyloRAG (measure/phylorag_v1.py) builds a morphological character matrix from the taxonomic literature: a language
model scores your characters from BioSysLit treatments and your PDFs, every state must carry a verbatim quote that is
found in its source, states are flagged for review where the quote may not be about the taxon, and only states you
accept enter the NEXUS matrix, on its own or merged into an existing matrix and matched to your molecular data.

The dialog runs the program's five steps from four tabs and shows the output below them:
  1 Characters        init --from-nexus      characters file from a published matrix (MorphoBank, Mesquite, PAUP)
  2 Literature        fetch                  BioSysLit treatments by keyword or taxon name, plus PDFs / text files
  3 Extract           extract [--dry-run]    the model scores each taxon from its own sources -> review sheet
  4 Review & export   audit / export         re-check an edited sheet; accepted states -> NEXUS + provenance
This module holds the dialog and the pure functions behind it (command building, checks), which the tests call
without a display.
"""
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"
STEPS = ("init", "fetch", "dry_run", "extract", "audit", "export")


def script_path(gui_dir):
    """measure/phylorag_v1.py beside the GUI (source tree, Docker) or in the descriptron-core package (pip)."""
    p = Path(gui_dir) / "measure" / "phylorag_v1.py"
    if p.is_file():
        return p
    try:
        from descriptron_core.cli import tools_dir
        p = Path(tools_dir()) / "phylorag_v1.py"
        if p.is_file():
            return p
    except Exception:
        pass
    return None


def python_for(environ=None):
    """PhyloRAG needs only the standard library (plus `anthropic` for --llm-backend api): the GUI's own Python,
    unless DESCRIPTRON_PHYLORAG_PYTHON names another."""
    environ = os.environ if environ is None else environ
    return environ.get("DESCRIPTRON_PHYLORAG_PYTHON", "").strip() or sys.executable


def _split(s):
    return [x.strip() for x in str(s or "").replace("\n", ";").split(";") if x.strip()]


def review_path(v):
    return v.get("review") or (os.path.join(v["out_dir"], "review_sheet.tsv") if v.get("out_dir") else "")


def build_cmd(step, v, script, python):
    """Command line for one step from the dialog's values (strings and booleans)."""
    base = [python, "-u", str(script)]
    if step == "init":
        return base + ["init", "--from-nexus", v["nexus"], "--out", v["chars"]] + (["--overwrite"] if v.get("overwrite") else [])
    if step == "fetch":
        c = base + ["fetch", "--out", v["sources"]]
        kw = v.get("keywords", "").replace(",", " ").split()
        if kw:
            c += ["--keywords"] + kw
        if v.get("names"):
            c += ["--names", "--taxa", v["taxa"]]
        if v.get("pdf_dir"):
            c += ["--pdf-dir", v["pdf_dir"]]
        if v.get("text_dir"):
            c += ["--text-dir", v["text_dir"]]
        return c
    if step in ("extract", "dry_run"):
        c = base + ["extract", "--characters", v["chars"], "--taxa", v["taxa"], "--sources", v["sources"],
                    "--out-dir", v["out_dir"], "--llm-backend", v.get("backend") or "claude-code",
                    "--model", v.get("model") or DEFAULT_MODEL, "--near", str(v.get("near") or "0.90"),
                    "--max-sources", str(v.get("max_sources") or "5")]
        if v.get("group"):
            c += ["--group", v["group"]]
        if v.get("context"):
            c += ["--context-file", v["context"]]
        if v.get("include_figures"):
            c += ["--include-figures"]
        if v.get("prefill"):
            c += ["--prefill", "clean"]
        if step == "dry_run":
            c += ["--dry-run"]
        return c
    if step == "audit":
        return base + ["audit", "--states", review_path(v), "--characters", v["chars"], "--taxa", v["taxa"],
                       "--sources", v["sources"], "--out-dir", os.path.join(os.path.dirname(review_path(v)) or ".",
                                                                            "reaudit"),
                       "--near", str(v.get("near") or "0.90")]
    if step == "export":
        c = base + ["export", "--review", review_path(v), "--characters", v["chars"], "--taxa", v["taxa"],
                    "--out-dir", v.get("export_dir") or os.path.join(os.path.dirname(review_path(v)) or ".", "export"),
                    "--labels", v.get("labels") or "tip"]
        if v.get("base_nexus"):
            c += ["--base-nexus", v["base_nexus"]]
        mols = _split(v.get("molecular"))
        if mols:
            c += ["--molecular"] + mols
        if v.get("add_missing"):
            c += ["--add-missing-taxa"]
        if v.get("include_unreviewed"):
            c += ["--include-unreviewed"]
        return c
    raise ValueError(f"unknown step {step}")


def check(step, v):
    """None if the step can run, else what is missing (shown to the user)."""
    def need(*keys):
        for k in keys:
            if not str(v.get(k) or "").strip():
                return f"Choose the {LABELS[k]} first."
        return None

    def exists(*keys):
        for k in keys:
            if v.get(k) and not os.path.exists(v[k]):
                return f"{LABELS[k]} not found: {v[k]}"
        return None
    if step == "init":
        return need("nexus", "chars") or exists("nexus")
    if step == "fetch":
        if not (v.get("keywords", "").strip() or v.get("names") or v.get("pdf_dir") or v.get("text_dir")):
            return "Give BioSysLit keywords, tick 'search each taxon name', or choose a PDF or text folder."
        if v.get("names") and not v.get("taxa"):
            return "Searching by taxon name needs the taxa file."
        return need("sources") or exists("taxa", "pdf_dir", "text_dir")
    if step in ("extract", "dry_run"):
        err = need("chars", "taxa", "sources", "out_dir") or exists("chars", "taxa", "sources", "context")
        if err:
            return err
        try:
            n = float(v.get("near") or 0.9)
            m = int(v.get("max_sources") or 5)
        except ValueError:
            return "Near-verbatim threshold must be a number and max sources a whole number."
        if not 0.5 <= n <= 1.0 or m < 1:
            return "Near-verbatim threshold must be between 0.5 and 1, max sources at least 1."
        return None
    if step in ("audit", "export"):
        if not review_path(v):
            return "Choose the review sheet (or the extract output folder) first."
        err = need("chars", "taxa") or exists("chars", "taxa", "base_nexus")
        if err:
            return err
        if not os.path.exists(review_path(v)):
            return f"review sheet not found: {review_path(v)}"
        if step == "audit":
            return need("sources") or exists("sources")
        for m in _split(v.get("molecular")):
            if not os.path.exists(m):
                return f"molecular file not found: {m}"
        return None
    return f"unknown step {step}"


LABELS = {"nexus": "published matrix (NEXUS)", "chars": "characters file", "taxa": "taxa file",
          "sources": "sources file", "out_dir": "output folder", "context": "terminology notes file",
          "pdf_dir": "PDF folder", "text_dir": "text folder", "base_nexus": "matrix to fill",
          "review": "review sheet"}


def open_file(path):
    """Open a file with the desktop's default program (the review sheet in a spreadsheet)."""
    if sys.platform.startswith("win"):
        os.startfile(path)                                     # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_dialog(root, gui_dir, tooltip=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    script = script_path(gui_dir)
    if script is None:
        messagebox.showerror("PhyloRAG", "measure/phylorag_v1.py was not found beside the GUI or in the "
                                         "descriptron-core package.")
        return None
    py = python_for()
    win = tk.Toplevel(root)
    win.title("PhyloRAG - character matrix from the literature")
    S = {k: tk.StringVar() for k in ("nexus", "chars", "taxa", "sources", "keywords", "pdf_dir", "text_dir", "out_dir",
                                     "group", "context", "backend", "model", "near", "max_sources", "review",
                                     "base_nexus", "molecular", "export_dir", "labels")}
    B = {k: tk.BooleanVar() for k in ("overwrite", "names", "include_figures", "prefill", "add_missing",
                                      "include_unreviewed")}
    S["backend"].set("claude-code"); S["model"].set(DEFAULT_MODEL); S["near"].set("0.90"); S["max_sources"].set("5")
    S["labels"].set("tip")
    state = {"proc": None, "q": queue.Queue(), "step": None}

    tk.Label(win, text="The model proposes, you decide: every state needs a verbatim quote found in its source, and "
                       "only states you accept in the review sheet enter the matrix.", fg="gray25", wraplength=760,
             justify="left").pack(padx=10, pady=(8, 2), anchor="w")
    nb = ttk.Notebook(win)
    nb.pack(fill="x", padx=8, pady=4)
    tabs = {}
    for key, title in (("chars", "1 Characters"), ("lit", "2 Literature"), ("ext", "3 Extract"),
                       ("exp", "4 Review & export")):
        f = tk.Frame(nb)
        nb.add(f, text=title)
        tabs[key] = f
    rowno = {k: 0 for k in tabs}

    def browse(var, kind, title, types=None):
        def f():
            if kind == "dir":
                p = filedialog.askdirectory(parent=win, title=title)
            elif kind == "save":
                p = filedialog.asksaveasfilename(parent=win, title=title, filetypes=types or [("All", "*.*")])
            elif kind == "many":
                ps = filedialog.askopenfilenames(parent=win, title=title, filetypes=types or [("All", "*.*")])
                p = ";".join(ps) if ps else ""
            else:
                p = filedialog.askopenfilename(parent=win, title=title, filetypes=types or [("All", "*.*")])
            if p:
                var.set(p)
        return f

    def row(tab, label, var, kind=None, types=None, tip=None, width=62):
        r = rowno[tab]
        tk.Label(tabs[tab], text=label).grid(row=r, column=0, sticky="e", padx=5, pady=3)
        e = tk.Entry(tabs[tab], textvariable=var, width=width)
        e.grid(row=r, column=1, sticky="w", padx=5, pady=3)
        if kind:
            tk.Button(tabs[tab], text="Browse", command=browse(var, kind, label, types)).grid(row=r, column=2, padx=4)
        if tip and tooltip:
            try:
                tooltip(e, tip)
            except Exception:
                pass
        rowno[tab] += 1
        return e

    def note(tab, text):
        tk.Label(tabs[tab], text=text, fg="gray35", wraplength=720, justify="left").grid(
            row=rowno[tab], column=0, columnspan=3, sticky="w", padx=8, pady=(2, 6))
        rowno[tab] += 1

    def check_row(tab, text, var):
        tk.Checkbutton(tabs[tab], text=text, variable=var).grid(row=rowno[tab], column=1, columnspan=2, sticky="w", padx=5)
        rowno[tab] += 1

    def buttons(tab, spec):
        f = tk.Frame(tabs[tab])
        f.grid(row=rowno[tab], column=0, columnspan=3, pady=8)
        for text, cmd, colour in spec:
            tk.Button(f, text=text, command=cmd, bg=colour, width=max(10, len(text) + 2)).pack(side="left", padx=5)
        rowno[tab] += 1

    NEX = [("NEXUS", "*.nex *.nexus *.nxs"), ("All", "*.*")]
    JS = [("JSON", "*.json"), ("All", "*.*")]
    TAB = [("Tables", "*.tsv *.csv *.txt"), ("All", "*.*")]

    # 1 Characters
    note("chars", "Characters are read from a published matrix: names, state labels and ordered characters. The file "
                  "can also be written by hand (JSON; discrete, meristic = counts, continuous = measurements).")
    row("chars", "Published matrix (NEXUS):", S["nexus"], "open", NEX)
    row("chars", "Characters file to write:", S["chars"], "save", JS)
    check_row("chars", "Overwrite an existing characters file", B["overwrite"])
    buttons("chars", [("Make characters file", lambda: start("init"), "lightgreen")])

    # 2 Literature
    note("lit", "BioSysLit holds the treatments of Plazi TreatmentBank on Zenodo. Keywords are higher taxa as tagged "
                "there (an order or family, e.g. Scolopendromorpha). PDFs and text files are added as they are; assign "
                "a monograph to its taxa in the taxa file's 'sources' column.")
    row("lit", "Taxa file (TSV):", S["taxa"], "open", TAB,
        tip="columns: taxon, tip_label (name in your alignment), synonyms (;), sources (;)")
    row("lit", "BioSysLit keywords:", S["keywords"])
    check_row("lit", "Also search BioSysLit for each taxon name and synonym", B["names"])
    row("lit", "PDF folder (optional):", S["pdf_dir"], "dir")
    row("lit", "Text folder (optional):", S["text_dir"], "dir")
    row("lit", "Sources file (JSON):", S["sources"], "save", JS)
    buttons("lit", [("Fetch literature", lambda: start("fetch"), "lightgreen")])

    # 3 Extract
    note("ext", "Each taxon is scored only from its own treatments (titles that begin with its name) and the sources "
                "the taxa file assigns to it. Check the sources first, then extract.")
    row("ext", "Characters file:", S["chars"], "open", JS)
    row("ext", "Taxa file:", S["taxa"], "open", TAB)
    row("ext", "Sources file:", S["sources"], "open", JS)
    row("ext", "Output folder:", S["out_dir"], "dir")
    row("ext", "Group (for the prompt):", S["group"], tip="e.g. centipede, weevil, orchid")
    row("ext", "Terminology notes (optional):", S["context"], "open", [("Text", "*.txt *.md"), ("All", "*.*")])
    of = tk.Frame(tabs["ext"])
    of.grid(row=rowno["ext"], column=1, columnspan=2, sticky="w", padx=5, pady=3)
    tk.Label(tabs["ext"], text="Model:").grid(row=rowno["ext"], column=0, sticky="e", padx=5)
    tk.OptionMenu(of, S["backend"], "claude-code", "api").pack(side="left")
    tk.Entry(of, textvariable=S["model"], width=20).pack(side="left", padx=4)
    tk.Label(of, text="near-verbatim").pack(side="left")
    tk.Entry(of, textvariable=S["near"], width=5).pack(side="left", padx=2)
    tk.Label(of, text="max sources").pack(side="left")
    tk.Entry(of, textvariable=S["max_sources"], width=3).pack(side="left", padx=2)
    rowno["ext"] += 1
    note("ext", "claude-code = your Claude subscription (claude -p); api = an API key. 0.90 accepts only spacing and "
                "punctuation differences between a quote and its source.")
    check_row("ext", "Also use figure and table records (captions often describe other taxa)", B["include_figures"])
    check_row("ext", "Pre-mark unflagged, verified states 'accept' in the review sheet (you still check them)",
              B["prefill"])
    buttons("ext", [("Check sources (no model)", lambda: start("dry_run"), "lightyellow"),
                    ("Extract", lambda: start("extract"), "lightgreen")])

    # 4 Review & export
    note("exp", "Open the review sheet, write accept or reject in the 'decision' column (flags say what to look at), "
                "save it as TSV or CSV, then export. Re-audit checks an edited sheet again and keeps your decisions.")
    row("exp", "Review sheet:", S["review"], "open", TAB)
    row("exp", "Matrix to fill (optional):", S["base_nexus"], "open", NEX,
        tip="only its missing cells (?) are filled; nothing is overwritten; disagreements are logged")
    row("exp", "Molecular data (optional):", S["molecular"], "many",
        [("Alignments / trees", "*.fasta *.fas *.fa *.phy *.phylip *.nex *.nexus *.tre *.tree *.nwk *.newick"),
         ("All", "*.*")], tip="FASTA, PHYLIP, NEXUS or Newick; ; between several files")
    row("exp", "Export folder:", S["export_dir"], "dir")
    lf = tk.Frame(tabs["exp"])
    lf.grid(row=rowno["exp"], column=1, columnspan=2, sticky="w", padx=5)
    tk.Label(tabs["exp"], text="Row names:").grid(row=rowno["exp"], column=0, sticky="e", padx=5)
    for val, txt in (("tip", "tip_label"), ("underscore", "Genus_epithet"), ("taxon", "Genus epithet")):
        tk.Radiobutton(lf, text=txt, variable=S["labels"], value=val).pack(side="left")
    rowno["exp"] += 1
    check_row("exp", "Add an empty row for every taxon of the molecular data without morphology", B["add_missing"])
    check_row("exp", "Include unreviewed states without flags (a quick look only; marked in the files)",
              B["include_unreviewed"])
    buttons("exp", [("Open review sheet", lambda: _open_sheet(), "white"),
                    ("Re-audit sheet (no model)", lambda: start("audit"), "lightyellow"),
                    ("Export matrix", lambda: start("export"), "lightgreen")])

    # output
    lf2 = tk.Frame(win)
    lf2.pack(fill="both", expand=True, padx=8, pady=(4, 2))
    log = tk.Text(lf2, height=16, width=110, wrap="word", font=("Courier", 9))
    sb = tk.Scrollbar(lf2, command=log.yview)
    log.configure(yscrollcommand=sb.set)
    log.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    status = tk.StringVar(value=f"Runs with {py}")
    bf = tk.Frame(win)
    bf.pack(fill="x", padx=8, pady=(2, 8))
    tk.Label(bf, textvariable=status, fg="gray30").pack(side="left")
    tk.Button(bf, text="Close", command=win.destroy, width=8).pack(side="right", padx=4)
    stop_btn = tk.Button(bf, text="Stop", state="disabled", width=8,
                         command=lambda: state["proc"] and state["proc"].terminate())
    stop_btn.pack(side="right", padx=4)

    def values():
        v = {k: x.get().strip() for k, x in S.items()}
        v.update({k: x.get() for k, x in B.items()})
        return v

    def _open_sheet():
        p = review_path(values())
        if not p or not os.path.exists(p):
            messagebox.showinfo("PhyloRAG", "No review sheet yet: run Extract first, or choose the sheet.", parent=win)
            return
        S["review"].set(p)
        open_file(p)

    def write(text):
        log.insert("end", text)
        log.see("end")

    def start(step):
        if state["proc"] is not None and state["proc"].poll() is None:
            messagebox.showinfo("PhyloRAG", "A step is still running.", parent=win)
            return
        v = values()
        err = check(step, v)
        if err:
            messagebox.showerror("PhyloRAG", err, parent=win)
            return
        if step == "extract" and not messagebox.askyesno(
                "PhyloRAG", f"Extract calls the model ({v['backend']}, {v['model']}) once per taxon or more. "
                            "Run 'Check sources' first to see how many calls. Start now?", parent=win):
            return
        cmd = build_cmd(step, v, script, py)
        write("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except OSError as e:
            messagebox.showerror("PhyloRAG", f"Could not start: {e}", parent=win)
            return
        state.update(proc=proc, step=step)
        stop_btn.configure(state="normal")
        status.set(f"running: {step.replace('_', ' ')} ...")

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
        proc, step = state["proc"], state["step"]
        proc.wait()
        stop_btn.configure(state="disabled")
        ok = proc.returncode == 0
        status.set(f"{step.replace('_', ' ')}: {'finished' if ok else f'stopped (exit code {proc.returncode})'}")
        if not ok:
            return
        v = values()
        if step == "extract":
            S["review"].set(os.path.join(v["out_dir"], "review_sheet.tsv"))
            if not S["export_dir"].get():
                S["export_dir"].set(os.path.join(v["out_dir"], "export"))
            nb.select(tabs["exp"])
            write("\nNext: Open review sheet, decide, save, then Export matrix.\n")
        if step == "audit":
            S["review"].set(os.path.join(os.path.dirname(review_path(v)), "reaudit", "review_sheet.tsv"))
            write("\nThe re-audited sheet (your decisions kept) is now the review sheet.\n")

    return win
