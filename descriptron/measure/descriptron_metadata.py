#!/usr/bin/env python3
"""
descriptron_metadata.py — specimen metadata for analyses: import, column roles, link to images
===============================================================================================

Reads the metadata people already have and turns it into one table that Descriptron's analyses can use:

  * a table you filled in (CSV, TSV or Excel), one row per specimen, any columns;
  * Darwin Core, as downloaded from GBIF or exported from a collection database: an occurrence.txt
    (tab separated) or a whole Darwin Core Archive (.zip with meta.xml and occurrence.txt).

Every column gets a ROLE, guessed from its name (Darwin Core terms are recognised) and editable:

  specimen_id   the specimen's identifier; also used to find its images (it must appear in the file name)
  group         the main grouping, usually the species
  factor        any other categorical variable: locality, sex, host, collection, season...
  continuous    a number used as a covariate: elevation, temperature, body length...
  latitude / longitude   decimal degrees (used for geographic distance)
  image         a column holding the image file name (instead of matching specimen IDs)
  ignore        not used in analyses

Outputs: specimen_metadata.csv (the table), metadata_schema.json (the roles), and, with --coco, a
link report: which images belong to which specimen, and which could not be matched.

    python descriptron_metadata.py import occurrence.txt --out-dir meta/
    python descriptron_metadata.py import dwca.zip --out-dir meta/ --coco annotations.json
    python descriptron_metadata.py import my_table.xlsx --out-dir meta/ --role locality=factor --role elev=continuous
    python descriptron_metadata.py link meta/specimen_metadata.csv meta/metadata_schema.json annotations.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROLES = ["specimen_id", "group", "factor", "continuous", "latitude", "longitude", "image", "ignore"]

# Darwin Core and common column names -> role (checked case-insensitively, after removing spaces/underscores)
_GUESS = [
    (("catalognumber", "occurrenceid", "specimenid", "specimen", "id", "voucher", "recordnumber", "fieldnumber",
      "materialsampleid", "individualid", "code"), "specimen_id"),
    (("scientificname", "species", "acceptedscientificname", "taxon", "specificepithet", "verbatimscientificname"), "group"),
    (("decimallatitude", "latitude", "lat"), "latitude"),
    (("decimallongitude", "longitude", "lon", "long", "lng"), "longitude"),
    (("filename", "file", "image", "imagefile", "associatedmedia", "image_filename"), "image"),
    (("elevation", "minimumelevationinmeters", "maximumelevationinmeters", "altitude", "depth",
      "minimumdepthinmeters", "temperature", "year", "individualcount"), "continuous"),
    (("locality", "country", "countrycode", "stateprovince", "county", "municipality", "sex", "lifestage", "caste",
      "host", "associatedtaxa", "habitat", "recordedby", "institutioncode", "collectioncode", "month", "season",
      "population", "site", "morph", "genus", "family"), "factor"),
]
_IGNORE_HINTS = ("gbifid", "datasetkey", "license", "rightsholder", "modified", "issue", "lastinterpreted",
                 "mediatype", "publishingorgkey", "basisofrecord", "occurrencestatus", "eventdate", "dateidentified",
                 "identifiedby", "references", "remarks")


def _norm(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


def guess_role(name: str, values: List[str]) -> str:
    n = _norm(name)
    for keys, role in _GUESS:
        if n in keys:
            return role
    if any(h in n for h in _IGNORE_HINTS):
        return "ignore"
    vals = [v for v in values if str(v).strip() not in ("", "NA", "nan", "None")]
    if vals and all(_isnum(v) for v in vals):
        return "continuous"
    if vals and len(set(vals)) <= max(2, len(vals) // 2):
        return "factor"
    return "ignore"


def _isnum(v) -> bool:
    try:
        float(str(v).replace(",", "."))
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ reading
def _read_delimited(text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    first = text.split("\n", 1)[0]
    delim = "\t" if first.count("\t") >= max(first.count(","), first.count(";")) else \
            (";" if first.count(";") > first.count(",") else ",")
    rd = csv.DictReader(io.StringIO(text), delimiter=delim)
    rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k is not None} for r in rd]
    return [c.strip() for c in rd.fieldnames or []], rows


def read_any(path: str) -> Tuple[List[str], List[Dict[str, str]], str]:
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            core = None
            if "meta.xml" in names:
                m = re.search(r'<core[^>]*>.*?<location>([^<]+)</location>', z.read("meta.xml").decode("utf-8", "replace"), re.S)
                core = m.group(1).strip() if m else None
            core = core or next((n for n in names if n.lower().endswith(("occurrence.txt", "occurrence.csv"))), None)
            if not core:
                raise SystemExit(f"{path}: no occurrence file found in the Darwin Core Archive")
            cols, rows = _read_delimited(z.read(core).decode("utf-8", "replace"))
            return cols, rows, f"Darwin Core Archive ({core})"
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        import openpyxl
        ws = openpyxl.load_workbook(p, read_only=True, data_only=True).active
        it = ws.iter_rows(values_only=True)
        cols = [str(c).strip() if c is not None else f"column{i + 1}" for i, c in enumerate(next(it))]
        rows = [{c: ("" if v is None else str(v).strip()) for c, v in zip(cols, r)} for r in it if any(v is not None for v in r)]
        return cols, rows, "Excel"
    text = p.read_text(encoding="utf-8-sig", errors="replace")
    cols, rows = _read_delimited(text)
    dwc = sum(1 for c in cols if _norm(c) in ("catalognumber", "occurrenceid", "scientificname", "decimallatitude"))
    return cols, rows, "Darwin Core table" if dwc >= 2 else "table"


def default_schema(cols: List[str], rows: List[Dict[str, str]]) -> Dict[str, str]:
    schema = {c: guess_role(c, [r.get(c, "") for r in rows[:500]]) for c in cols}
    # exactly one specimen_id and one group: keep the first guess of each, demote the rest to factor/ignore
    for role, fallback in (("specimen_id", "ignore"), ("group", "factor"), ("latitude", "ignore"), ("longitude", "ignore")):
        seen = False
        for c in cols:
            if schema[c] == role:
                if seen:
                    schema[c] = fallback
                seen = True
    return schema


# ------------------------------------------------------------------ linking to images
def link_images(rows: List[Dict[str, str]], schema: Dict[str, str], file_names: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """image file name -> row index. By an 'image' column if present, else by the specimen ID occurring in the
    file name (the longest matching ID wins, case-insensitive, bounded by non-alphanumerics)."""
    img_col = next((c for c, r in schema.items() if r == "image"), None)
    id_col = next((c for c, r in schema.items() if r == "specimen_id"), None)
    out, unmatched = {}, []
    if img_col:
        by_stem = {Path(r.get(img_col, "")).stem.lower(): i for i, r in enumerate(rows) if r.get(img_col)}
        for fn in file_names:
            i = by_stem.get(Path(fn).stem.lower())
            (out.__setitem__(fn, i) if i is not None else unmatched.append(fn))
        return out, unmatched
    if not id_col:
        return {}, list(file_names)
    ids = sorted(((r.get(id_col, "").strip(), i) for i, r in enumerate(rows) if r.get(id_col, "").strip()),
                 key=lambda t: -len(t[0]))
    pats = [(re.compile(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])", re.I), i) for s, i in ids]
    for fn in file_names:
        stem = Path(fn).stem
        hit = next((i for p, i in pats if p.search(stem)), None)
        (out.__setitem__(fn, hit) if hit is not None else unmatched.append(fn))
    return out, unmatched


def write_table(cols, rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def load_table(path) -> Tuple[List[str], List[Dict[str, str]]]:
    return _read_delimited(Path(path).read_text(encoding="utf-8-sig", errors="replace"))


def cmd_import(a):
    cols, rows, kind = read_any(a.input)
    if not rows:
        sys.exit(f"{a.input}: no rows")
    schema = default_schema(cols, rows)
    for r in a.role or []:
        c, _, role = r.partition("=")
        if c not in schema or role not in ROLES:
            sys.exit(f"--role {r}: column must be one of {cols} and role one of {ROLES}")
        schema[c] = role
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    keep = [c for c in cols if schema[c] != "ignore"] if a.drop_ignored else cols
    write_table(keep, rows, out / "specimen_metadata.csv")
    json.dump({c: schema[c] for c in keep}, open(out / "metadata_schema.json", "w"), indent=1)
    print(f"read {len(rows)} specimens, {len(cols)} columns ({kind})")
    for role in ROLES[:-1]:
        cs = [c for c in keep if schema[c] == role]
        if cs:
            print(f"  {role:12s} {', '.join(cs)}")
    print(f"wrote {out/'specimen_metadata.csv'} and {out/'metadata_schema.json'} (edit the roles there or in the GUI)")
    if a.coco:
        _report_link(rows, {c: schema[c] for c in keep}, a.coco, out)


def _report_link(rows, schema, coco_path, out):
    coco = json.load(open(coco_path))
    names = [im["file_name"] for im in coco["images"]]
    m, unmatched = link_images(rows, schema, names)
    json.dump({"matched": m, "unmatched": unmatched}, open(Path(out) / "image_links.json", "w"), indent=1)
    print(f"linked {len(m)}/{len(names)} images to {len(set(m.values()))} specimens -> {Path(out)/'image_links.json'}")
    if unmatched:
        print(f"  not matched ({len(unmatched)}), e.g.: {', '.join(unmatched[:5])}")


def cmd_link(a):
    cols, rows = load_table(a.table)
    schema = json.load(open(a.schema))
    _report_link(rows, schema, a.coco, Path(a.table).parent)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("Outputs:")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import"); p.add_argument("input"); p.add_argument("--out-dir", required=True)
    p.add_argument("--role", action="append", help="COLUMN=ROLE to override a guess (repeat)")
    p.add_argument("--coco", help="also link the metadata to this COCO's images and report")
    p.add_argument("--drop-ignored", action="store_true", help="leave 'ignore' columns out of the table")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("link"); p.add_argument("table"); p.add_argument("schema"); p.add_argument("coco")
    p.set_defaults(fn=cmd_link)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
