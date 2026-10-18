"""
descriptron_category_edit.py - rename, merge or delete annotation categories (GUI utility)
==========================================================================================

A misspelt category ("mesosomu" for "mesosoma") is fixed by renaming it; renaming to a name that already exists
merges the two (their annotations end up under one category). Deleting a category removes it AND every
annotation filed under it, so the GUI asks first and says how many.

Works on a COCO dict (a saved file: the GUI writes the result to a NEW file beside the original) and on the
GUI session (the same functions are applied to the session's accumulators). Plain Python, tested without Tk.
"""
from __future__ import annotations

from collections import Counter


def counts_by_category(coco: dict) -> dict:
    """category name -> number of annotations"""
    names = {c["id"]: c["name"] for c in coco.get("categories", [])}
    n = Counter(a.get("category_id") for a in coco.get("annotations", []))
    return {names[i]: n.get(i, 0) for i in names}


def rename_category(coco: dict, old: str, new: str) -> dict:
    """rename `old` to `new`; if `new` already exists the two are merged under `new`'s id. Returns a new dict."""
    new = new.strip()
    if not new:
        raise ValueError("the new name is empty")
    cats = coco.get("categories", [])
    by_name = {c["name"]: c for c in cats}
    if old not in by_name:
        raise KeyError(f"no category named {old!r}")
    if new == old:
        return coco
    src = by_name[old]
    if new in by_name:                                         # merge
        tgt = by_name[new]["id"]
        new_cats = [c for c in cats if c["name"] != old]
        anns = [dict(a, category_id=tgt) if a.get("category_id") == src["id"] else a
                for a in coco.get("annotations", [])]
    else:
        new_cats = [dict(c, name=new) if c["name"] == old else c for c in cats]
        anns = list(coco.get("annotations", []))
    return {**coco, "categories": new_cats, "annotations": anns}


def delete_category(coco: dict, name: str) -> tuple[dict, int]:
    """remove the category and every annotation under it; returns (new dict, annotations removed)"""
    cats = coco.get("categories", [])
    ids = {c["id"] for c in cats if c["name"] == name}
    if not ids:
        raise KeyError(f"no category named {name!r}")
    anns = coco.get("annotations", [])
    keep = [a for a in anns if a.get("category_id") not in ids]
    return {**coco, "categories": [c for c in cats if c["id"] not in ids], "annotations": keep}, len(anns) - len(keep)


def corrected_path(path: str) -> str:
    """where a corrected copy of a file goes: beside it, never over it"""
    import os
    base, ext = os.path.splitext(path)
    return f"{base}_categories_fixed{ext or '.json'}"


# --- occlusion (GUI v86): which category can hide which -------------------------------------------------------------
# Stored on the COCO category: "occluder": true = can hide any structure (legs, antennae lying over the body);
# "occludes": ["head", "compound_eye"] = can hide only those (a pronotum hides the head and eye but only abuts the
# elytron). Read by the measurement scripts with --repair_outline_with auto / --occluder_categories auto
# (measure/homology_grid_ext.py). A part nested inside another (cephalic_dorsum in head_capsule) is NOT an occlusion.

def occlusion_of(coco: dict) -> dict:
    """category name -> None (not an occluder), '*' (hides anything) or a sorted list of names it can hide"""
    out = {}
    for c in coco.get("categories", []):
        if c.get("occludes"):
            out[c["name"]] = sorted(c["occludes"])
        elif c.get("occluder"):
            out[c["name"]] = "*"
        else:
            out[c["name"]] = None
    return out


def set_occlusion(coco: dict, name: str, hides) -> dict:
    """hides: None (not an occluder), '*' (anything) or an iterable of category names. Returns a new dict."""
    cats = coco.get("categories", [])
    names = {c["name"] for c in cats}
    if name not in names:
        raise KeyError(f"no category named {name!r}")
    if hides not in (None, "*"):
        hides = sorted(set(hides) - {name})
        unknown = set(hides) - names
        if unknown:
            raise KeyError(f"unknown categories: {sorted(unknown)}")
    new = []
    for c in cats:
        if c["name"] == name:
            c = {k: v for k, v in c.items() if k not in ("occluder", "occludes")}
            if hides == "*":
                c["occluder"] = True
            elif hides:
                c["occludes"] = list(hides)
        new.append(c)
    return {**coco, "categories": new}
