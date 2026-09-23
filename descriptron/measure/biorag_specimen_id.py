#!/usr/bin/env python3
"""
biorag_specimen_id.py — read species and specimen number from an image file name
=================================================================================

Extracted so the modules that need it do not have to import a retired analysis script.
Two naming conventions are in use in the same collection and both must be read, because a
matcher that silently understands only one drops whole species rather than random images.
"""

import re
from pathlib import Path


def specimen_of(filename: str, profile: dict, species_codes, report: dict = None) -> tuple:
    """Species code and specimen id from an image file name.

    Two conventions are in use in the same collection and both must be read, because a
    matcher that silently understands only one drops whole species rather than random
    images: `..._morph_sp1_1_forewing_...` separates the code from the number, while
    `..._morph_virg5_rostrum_...` runs them together. A code may also be abbreviated
    (`punc7` for punctulata); an abbreviation is accepted only when exactly one species
    code begins with it, and every such inference is recorded in `report` so it can be
    checked rather than trusted.
    """
    rx = (profile.get("specimen_id") or {}).get("number_regex", r"_(\d+)_")
    base = Path(filename).stem
    code, num = None, None
    for c in sorted(species_codes, key=len, reverse=True):
        m = re.search(rf"(?:^|[_.]){re.escape(c)}(\d*)(?:[_.]|$)", base)
        if m:
            code = c
            num = m.group(1) or None          # digits glued to the code, if any
            break
    if code is None:                          # an abbreviation, if it is unambiguous
        for tok, digits in re.findall(r"(?:^|[_.])([A-Za-z][A-Za-z.]{2,})(\d+)(?=[_.]|$)", base):
            hits = [c for c in species_codes if c.startswith(tok)]
            if len(hits) == 1:
                code, num = hits[0], digits
                if report is not None:
                    report.setdefault("abbreviations", {}).setdefault(tok, hits[0])
                break
    if code and not num:
        m = list(re.finditer(rx, base))
        num = m[-1].group(1) if m else None
    if report is not None and not (code and num):
        report.setdefault("unmapped", []).append(filename)
    return code, (f"{code}_{num}" if code and num else None)


def load_exclusions(spec):
    """Which annotations must be ignored, by id where the screen gives one.

    A drifted polygon still produces a number and nothing downstream can tell that number
    from a measurement, so it has to go. But dropping every annotation on an image because
    one of them drifted throws away good work: on most flagged images only one or two
    cells moved. Annotation ids are used when the screen supplies them and whole images
    only when it does not.
    """
    ids, images = set(), set()
    for f in str(spec or "").split(","):
        f = f.strip()
        if not f or not Path(f).exists():
            continue
        t = pd.read_csv(f, sep="\t" if f.endswith(".tsv") else ",")
        if "annotation_id" in t.columns:
            flag = t["flag"] if "flag" in t.columns else True
            ids |= set(pd.to_numeric(t.loc[flag, "annotation_id"], errors="coerce").dropna().astype(int))
        else:
            col = next((c for c in ("image", "file_name", "image_name") if c in t.columns), None)
            if col:
                images |= set(t[col].astype(str))
    return ids, images
