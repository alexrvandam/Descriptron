#!/usr/bin/env python3
"""
biorag_rename_species_v1.py — apply final species names to a finished monograph
===============================================================================

Working names ("Diaphorina sp. 1A", "D. sp. 'kenya'") are replaced by the names
the taxonomist settles on, everywhere they occur, without regenerating any text
with a language model.

Why not sed: every species appears in at least two forms (the display name and
the abbreviated form used in comparisons), the forms overlap ("D. sp. 1" inside
"D. sp. 1A"), and the folder codes must NOT change (they are the join key of
the data matrix). This script builds one regular expression per species from
the taxon profile, replaces the longest form first, and reports exactly how
many replacements it made in each file.

Input: a table (TSV or CSV) with a row per species to rename:

    code        name                        status       authority
    sp1A        Diaphorina liliyae          undescribed  Serbina & Van Dam, 2026
    kenya       Diaphorina kenyensis        undescribed
    acok        Diaphorina acokantherae     described    Munyiri, 1970

`code` must match the folder code / group label; `name` is the new display
name; `status` and `authority` are optional (status defaults to the profile's
current value, and `verify: true` is cleared for every renamed species).

Usage:
  python biorag_rename_species_v1.py --names final_names.tsv \\
      --taxon_profile biorag_prompts/taxon_profiles/diaphorina_taxon_profile.yaml \\
      --monograph "/media/.../Diaphorina_monograph" [--apply]

Without --apply nothing is written (a dry run listing every change).
With --apply, each file is backed up as <file>.bak_rename_<date> first.

Afterwards, regenerate the derived files (no model calls, a few minutes):
  1. data sheets:  biorag_description_refiner_v1.py ... --sheets_only
  2. key wording:  biorag_key_builder_v1.py ... --llm-backend none   (or rerun
     with a backend to reword the couplets with the new names)
  3. audit:        biorag_confabulation_checker_v2.py ...
  4. ontology:     biorag_ontology_annotator_v2.py ...
  5. monograph:    build_species_treatment_docx_v2.py and
                   build_monograph_exports_v1.py
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import biorag_feature_policy as pol  # noqa: E402

TEXT_SUFFIXES = {".txt", ".json", ".jsonld", ".md", ".xml", ".tsv", ".csv", ".yaml", ".yml"}


def name_forms(display: str, genus: str) -> List[str]:
    """Display name and the abbreviated form(s) used in comparisons."""
    forms = [display]
    if genus and display.startswith(genus + " "):
        forms.append(f"{genus[0]}. {display[len(genus) + 1:]}")
    return forms


def build_patterns(mapping: Dict[str, Dict], profile: Dict) -> List[tuple]:
    """(compiled regex, replacement, code) — longest old form first."""
    genus = profile.get("taxon", {}).get("genus", "")
    pats = []
    for code, new in mapping.items():
        old_display = pol.species_display_name(code, profile)
        new_display = new["name"]
        olds = name_forms(old_display, genus)
        news = name_forms(new_display, new.get("genus") or genus)
        for o, n in zip(olds, news):
            if o != n:
                pats.append((o, n, code))
    pats.sort(key=lambda t: -len(t[0]))
    return [(re.compile(re.escape(o) + r"(?![\w'])"), n, code) for o, n, code in pats]


def rewrite(path: Path, pats, apply: bool, stamp: str) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return 0
    new, total = text, 0
    for rx, repl, _ in pats:
        new, k = rx.subn(repl, new)
        total += k
    if total and apply:
        shutil.copy2(path, path.with_name(path.name + f".bak_rename_{stamp}"))
        path.write_text(new, encoding="utf-8")
    return total


def update_profile(profile_path: Path, mapping: Dict[str, Dict], apply: bool, stamp: str) -> int:
    """Rewrite the species block of the taxon profile (line-wise, comments kept)."""
    lines = profile_path.read_text(encoding="utf-8").split("\n")
    changed = 0
    for i, line in enumerate(lines):
        m = re.match(r'^(\s+)([\w+\-.]+):\s*\{(.*)\}\s*$', line)
        if not m or m.group(2) not in mapping:
            continue
        indent, code, body = m.groups()
        new = mapping[code]
        body = re.sub(r'name:\s*"[^"]*"', f'name: "{new["name"]}"', body)
        if new.get("status"):
            body = re.sub(r'status:\s*[\w]+', f'status: {new["status"]}', body)
        body = re.sub(r'verify:\s*true', 'verify: false', body)
        if new.get("authority"):
            if "authority:" in body:
                body = re.sub(r'authority:\s*"[^"]*"', f'authority: "{new["authority"]}"', body)
            else:
                body += f', authority: "{new["authority"]}"'
        lines[i] = f'{indent}{code}: {{{body}}}'
        changed += 1
    if changed and apply:
        shutil.copy2(profile_path, profile_path.with_name(profile_path.name + f".bak_rename_{stamp}"))
        profile_path.write_text("\n".join(lines), encoding="utf-8")
    return changed


def main():
    ap = argparse.ArgumentParser(description="Apply final species names to a finished monograph")
    ap.add_argument("--names", required=True, help="TSV/CSV with columns code, name[, status, authority]")
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--monograph", required=True, help="Monograph directory (descriptions, key, treatments ...)")
    ap.add_argument("--extra_dir", nargs="*", default=[], help="Other directories to rewrite")
    ap.add_argument("--apply", action="store_true", help="Write the changes (default: dry run)")
    a = ap.parse_args()

    sep = "\t" if str(a.names).endswith((".tsv", ".tab")) else ","
    tab = pd.read_csv(a.names, sep=sep, dtype=str).fillna("")
    missing = {"code", "name"} - set(tab.columns)
    if missing:
        sys.exit(f"--names is missing the column(s): {', '.join(sorted(missing))}")
    profile = pol.load_taxon_profile(a.taxon_profile)
    known = set(profile.get("species", {}))
    mapping, unknown = {}, []
    for r in tab.to_dict("records"):
        code = r["code"].strip()
        if code in known:
            mapping[code] = r
        else:
            unknown.append(code)
    if unknown:
        print(f"WARNING: these codes are not in the taxon profile and are ignored: {', '.join(unknown)}")
    if not mapping:
        sys.exit("nothing to rename")
    pats = build_patterns(mapping, profile)
    print(f"{len(mapping)} species to rename, {len(pats)} name forms:")
    for rx, repl, code in pats:
        print(f"   [{code}] {rx.pattern.split('(?!')[0].replace(chr(92), '')}  ->  {repl}")

    stamp = datetime.now().strftime("%Y%m%d")
    roots = [Path(a.monograph)] + [Path(x) for x in a.extra_dir]
    files, total = 0, 0
    per_dir = {}
    for root in roots:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES or ".bak" in p.name:
                continue
            n = rewrite(p, pats, a.apply, stamp)
            if n:
                files += 1
                total += n
                per_dir[str(p.parent)] = per_dir.get(str(p.parent), 0) + n
    nprof = update_profile(Path(a.taxon_profile), mapping, a.apply, stamp)
    print(f"\n{'APPLIED' if a.apply else 'DRY RUN'}: {total} replacements in {files} files; "
          f"{nprof} profile entries updated")
    for d, n in sorted(per_dir.items(), key=lambda kv: -kv[1])[:15]:
        print(f"   {n:6d}  {d}")
    if not a.apply:
        print("\nNothing was written. Re-run with --apply (every changed file is backed up first).")
    else:
        print("\nNow regenerate the derived files (see the header of this script): data sheets, key wording, "
              "audit, ontology, DOCX and the machine-readable exports.")


if __name__ == "__main__":
    main()
