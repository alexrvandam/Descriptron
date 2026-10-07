"""
descriptron_descriptive_characters.py - descriptive characters for any labelled structure (GUI panel)
======================================================================================================

The Descriptron-GBIF Annotator lets a user record, for each structure, its texture, sculpture, setae, colour,
colour pattern, lustre and so on. This is the same panel for the Descriptron GUI, on by default: after a mask is
labelled (with any label - custom, imported or one used before) its descriptive characters can be set, and they
are saved with the annotation.

Storage (the same field the GBIF annotator writes, so files move between the two tools):
    COCO annotations[].attributes = {"texture": "punctate", "setae": "dense", "setae_count": "12", ...}
    COCO annotations[].notes      = "free text"
Only characters that were set are stored; an empty choice means "not assessed".

The vocabulary is descriptron_descriptive_characters.json beside this file (states from the GBIF annotator's
ROSETTA vocabulary, grouped into sets, plus meristic counts). Characters are categorical (one state from a list;
an "extensible" character also accepts a state typed in) or counts (a whole number).

The Tk dialog is `edit_characters()`; everything else here is plain Python and tested without a display.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOCAB_FILE = "descriptron_descriptive_characters.json"


def load_vocabulary(path: str | os.PathLike | None = None) -> dict:
    """the vocabulary: beside this file, else DESCRIPTRON_GUI_DATA (pip install), else an explicit path"""
    cands = [Path(path)] if path else []
    cands += [HERE / VOCAB_FILE]
    if os.environ.get("DESCRIPTRON_GUI_DATA"):
        cands.append(Path(os.environ["DESCRIPTRON_GUI_DATA"]) / VOCAB_FILE)
    for c in cands:
        if c.exists():
            with open(c, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"{VOCAB_FILE} not found (looked in: {', '.join(map(str, cands))})")


def characters_in_sets(vocab: dict, sets) -> list[str]:
    """character keys of the chosen sets, in set order, without repeats"""
    out = []
    for s in sets:
        for c in vocab["sets"].get(s, []):
            if c not in out:
                out.append(c)
    return out


def clean_attributes(raw: dict, vocab: dict) -> tuple[dict, list[str]]:
    """keep only characters that were set; counts must be whole numbers >= 0. Returns (attributes, problems)."""
    out, problems = {}, []
    chars = vocab.get("characters", {})
    for k, v in raw.items():
        v = ("" if v is None else str(v)).strip()
        if not v:
            continue
        spec = chars.get(k, {"type": "categorical"})
        if spec.get("type") == "count":
            if not v.isdigit():
                problems.append(f"{spec.get('label', k)}: '{v}' is not a whole number")
                continue
            v = str(int(v))
        out[k] = v
    return out, problems


def summary(attrs: dict, vocab: dict, limit: int = 4) -> str:
    """short text for a status line: 'texture: punctate; setae: dense; +2 more'"""
    if not attrs:
        return "no descriptive characters"
    chars = vocab.get("characters", {})
    parts = [f"{chars.get(k, {}).get('label', k)}: {v}" for k, v in attrs.items()]
    more = len(parts) - limit
    return "; ".join(parts[:limit]) + (f"; +{more} more" if more > 0 else "")


class PanelState:
    """what the panel remembers during a session: which sets are shown, and the last values used per structure
    (so 'Same as last <structure>' can fill a new specimen in one click)"""

    def __init__(self, vocab: dict):
        self.sets = list(vocab.get("default_sets", []))
        self.last = {}          # structure name -> attributes
        self.custom = {}        # character key -> spec, characters the user added this session

    def remember(self, structure: str, attrs: dict):
        if attrs:
            self.last[structure] = dict(attrs)


def _ascii(s: str) -> str:
    return str(s).encode("ascii", "replace").decode()


def edit_characters(root, structure: str, current: dict, notes: str, vocab: dict, state: PanelState):
    """modal dialog; returns (attributes, notes) or None if cancelled"""
    import tkinter as tk
    from tkinter import ttk, simpledialog, messagebox

    chars = dict(vocab["characters"]); chars.update(state.custom)
    dlg = tk.Toplevel(root)
    dlg.title(_ascii(f"Descriptive characters - {structure}"))
    dlg.transient(root)
    result = {"value": None}
    values = {k: tk.StringVar(value=current.get(k, "")) for k in set(chars) | set(current)}
    for k in current:                                   # a state from a file that is not in the vocabulary
        chars.setdefault(k, {"label": k.replace("_", " "), "type": "categorical", "extensible": True,
                             "values": [current[k]], "description": "from the loaded file"})

    top = tk.Frame(dlg); top.pack(fill="x", padx=8, pady=(8, 2))
    tk.Label(top, text=_ascii(f"Structure: {structure}"), font=("Arial", 10, "bold")).pack(side="left")
    sets_frame = tk.LabelFrame(dlg, text="Character sets shown"); sets_frame.pack(fill="x", padx=8, pady=2)
    set_vars = {}
    for i, s in enumerate(vocab["sets"]):
        set_vars[s] = tk.BooleanVar(value=s in state.sets)
        tk.Checkbutton(sets_frame, text=_ascii(s), variable=set_vars[s], command=lambda: rebuild()).grid(
            row=i // 4, column=i % 4, sticky="w", padx=4)

    body_outer = tk.Frame(dlg); body_outer.pack(fill="both", expand=True, padx=8, pady=4)
    canvas = tk.Canvas(body_outer, height=360, highlightthickness=0)
    sb = tk.Scrollbar(body_outer, orient="vertical", command=canvas.yview)
    body = tk.Frame(canvas)
    body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=body, anchor="nw"); canvas.configure(yscrollcommand=sb.set)
    canvas.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")

    def shown():
        keys = characters_in_sets(vocab, [s for s, v in set_vars.items() if v.get()])
        keys += [k for k in state.custom if k not in keys]
        keys += [k for k in current if k not in keys]              # never hide a value that is set
        keys += [k for k, v in values.items() if v.get() and k not in keys]
        return keys

    def rebuild():
        for w in body.winfo_children():
            w.destroy()
        for r, k in enumerate(shown()):
            spec = chars.get(k, {})
            tk.Label(body, text=_ascii(spec.get("label", k)), anchor="e", width=24).grid(row=r, column=0, sticky="e", padx=(0, 6), pady=1)
            if spec.get("type") == "count":
                w = tk.Spinbox(body, from_=0, to=100000, textvariable=values[k], width=10)
                if not values[k].get():
                    values[k].set("")
            else:
                w = ttk.Combobox(body, textvariable=values[k], width=30, values=[""] + list(spec.get("values", [])),
                                 state="normal" if spec.get("extensible", True) else "readonly")
            w.grid(row=r, column=1, sticky="w", pady=1)
            tk.Label(body, text=_ascii(spec.get("description", ""))[:70], fg="#666", font=("Arial", 8),
                     anchor="w").grid(row=r, column=2, sticky="w", padx=6)

    note_fr = tk.Frame(dlg); note_fr.pack(fill="x", padx=8, pady=2)
    tk.Label(note_fr, text="Notes").pack(side="left")
    notes_var = tk.StringVar(value=notes or "")
    tk.Entry(note_fr, textvariable=notes_var, width=70).pack(side="left", fill="x", expand=True, padx=4)

    btns = tk.Frame(dlg); btns.pack(fill="x", padx=8, pady=(4, 8))

    def same_as_last():
        prev = state.last.get(structure)
        if not prev:
            messagebox.showinfo("Same as last", _ascii(f"No earlier '{structure}' in this session."), parent=dlg)
            return
        for k, v in prev.items():
            values.setdefault(k, tk.StringVar()).set(v)
        rebuild()

    def add_character():
        name = simpledialog.askstring("Add character", "Name of the new character (e.g. 'spine number'):", parent=dlg)
        if not name:
            return
        is_count = messagebox.askyesno("Add character", "Is it a count (a whole number)?", parent=dlg)
        key = name.strip().lower().replace(" ", "_")
        spec = {"label": name.strip(), "type": "count" if is_count else "categorical", "extensible": True,
                "values": [], "description": "added by the user"}
        state.custom[key] = spec; chars[key] = spec
        values.setdefault(key, tk.StringVar())
        rebuild()

    def clear():
        for v in values.values():
            v.set("")

    def ok():
        raw = {k: v.get() for k, v in values.items()}
        attrs, problems = clean_attributes(raw, {"characters": chars})
        if problems:
            messagebox.showerror("Descriptive characters", "\n".join(_ascii(p) for p in problems), parent=dlg)
            return
        state.sets = [s for s, v in set_vars.items() if v.get()]
        state.remember(structure, attrs)
        result["value"] = (attrs, notes_var.get().strip())
        dlg.destroy()

    tk.Button(btns, text=_ascii(f"Same as last {structure}")[:40], command=same_as_last).pack(side="left")
    tk.Button(btns, text="Add character...", command=add_character).pack(side="left", padx=4)
    tk.Button(btns, text="Clear all", command=clear).pack(side="left")
    tk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="right")
    tk.Button(btns, text="OK", width=8, command=ok, bg="#9fd8b0").pack(side="right", padx=4)
    dlg.bind("<Return>", lambda e: ok()); dlg.bind("<Escape>", lambda e: dlg.destroy())
    rebuild()
    dlg.grab_set()
    root.wait_window(dlg)
    return result["value"]
