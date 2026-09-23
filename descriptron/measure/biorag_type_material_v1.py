#!/usr/bin/env python3
"""
biorag_type_material_v1.py — turn the collaborator's type designations into the
statement the Code requires
==============================================================================

A new species-group name published after 1999 is only available if the work
explicitly fixes a holotype (ICZN Art. 16.4.1) and states the collection in
which it is deposited (Art. 16.4.2). Until those come back from the collectors
the monograph prints a placeholder. This reads the filled-in designations and
writes, per species, the sentence that replaces it — and refuses to write one
that would not satisfy the Code, rather than producing something that looks
finished and is not.

Input: the type-designation sheet from `biorag_collaborator_forms_v1.py`
(columns `specimen_id`, `type_status`, `institution`, `accession_number`), or
any table with those columns.

  python biorag_type_material_v1.py \\
      --types "$M/collaborator_forms/..._type_designations.tsv" \\
      --localities "$M/localities/Diaphorina_localities_verified.tsv" \\
      --taxon_profile <profile.yaml> --out_dir "$M/localities"   [--write_localities]

Outputs `type_material.json` (per species: the statement, the holotype, the
paratypes, and any problem found) and `type_material_report.txt`. With
--write_localities the three columns are merged back into the localities table,
which is what the DOCX, TaxPub, JSON-LD and Darwin Core exports read.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

VERSION = "1.2"

# Under the principle of typification (Art. 61.1) a name-bearing type is fixed permanently by the
# publication that designated it. A redescription adds material examined, never types: only the
# original author's designated specimens are types (Art. 72.4), so no one can add paratypes later.
# The two legitimate later acts are a lectotype from an original syntype series (Art. 74) and a
# neotype where the name-bearing types are lost (Art. 75) — and both are NEW acts that cite the
# original, not edits to it. The register below is therefore APPEND-ONLY: an act that records the
# publication which fixed it is never rewritten, only superseded by a later act that cites it.
ACT_FIELDS = ("kind", "specimen_id", "species", "institution", "accession_number",
              "sex", "life_stage", "type_purpose", "fixed_by", "fixed_year", "zoobank_lsid")


def act_key(a: dict) -> tuple:
    return (a.get("species", ""), a.get("kind", ""), a.get("specimen_id", ""))


def is_published(a: dict) -> bool:
    """An act is settled once the publication that fixed it is recorded."""
    return bool(str(a.get("fixed_by", "")).strip())


def merge_acts(existing: list, incoming: list) -> tuple:
    """Add new acts without touching settled ones. Returns (acts, refusals)."""
    by_key = {act_key(a): a for a in existing}
    refused = []
    for a in incoming:
        k = act_key(a)
        old = by_key.get(k)
        if old is None:
            by_key[k] = a
            continue
        if is_published(old) and any(str(old.get(f, "")) != str(a.get(f, ""))
                                     for f in ("institution", "accession_number", "kind")):
            refused.append(
                f"{a.get('species')} / {a.get('specimen_id')}: this {old.get('kind')} was fixed by "
                f"{old.get('fixed_by')} and cannot be altered (Art. 61.1). To change which specimen "
                f"bears the name, publish a lectotype or neotype designation citing that work")
            continue
        by_key[k] = {**old, **{f: v for f, v in a.items() if v not in ("", None)}}
    return list(by_key.values()), refused

# The kinds of type material, what each means under the Code, and where each is legitimate.
# "new" = may be designated for a species being described here; "existing" = applies to a name
# that already exists (so it is an error on a new species).
KINDS = {
    "holotype":      dict(rank=0, group="new",      unique=True,
                          note="the single name-bearing specimen (Art. 73.1)"),
    "syntype":       dict(rank=1, group="new",      unique=False,
                          note="name-bearing series where no holotype is fixed (Art. 73.2); "
                               "valid, but a holotype is preferable"),
    "allotype":      dict(rank=2, group="new",      unique=True,
                          note="a specimen of the opposite sex to the holotype. NOT a name-bearing "
                               "type under the Code: formally it is a paratype (Rec. 72A)"),
    "paratype":      dict(rank=3, group="new",      unique=False,
                          note="the remaining specimens of the type series (Art. 72.4.5)"),
    "lectotype":     dict(rank=4, group="existing", unique=True,
                          note="designated from an existing syntype series (Art. 74)"),
    "paralectotype": dict(rank=5, group="existing", unique=False,
                          note="the remaining former syntypes (Art. 73.2.2)"),
    "neotype":       dict(rank=6, group="existing", unique=True,
                          note="replaces lost or destroyed name-bearing types (Art. 75)"),
    "non-type":      dict(rank=9, group="any",      unique=False,
                          note="additional material examined, not part of the type series"),
}
LABEL = {"holotype": "Holotype", "syntype": "Syntypes", "allotype": "Allotype",
         "paratype": "Paratypes", "lectotype": "Lectotype", "paralectotype": "Paralectotypes",
         "neotype": "Neotype", "non-type": "Other material examined"}
# a life stage worth naming; adults are the unmarked default in taxonomic usage
STAGES = {"adult": "", "larva": "larva", "larval": "larva", "nymph": "nymph",
          "immature": "immature", "pupa": "pupa", "egg": "egg"}


def kind_of(raw: str) -> str:
    """Normalise whatever the collaborator typed to one of KINDS, or "" if unrecognised."""
    t = re.sub(r"[^a-z]", "", str(raw or "").lower())
    if not t:
        return ""
    for k in sorted(KINDS, key=lambda k: -len(k)):
        if t.startswith(re.sub(r"[^a-z]", "", k)) or re.sub(r"[^a-z]", "", k).startswith(t):
            return k
    return ""


def sex_symbol(s: str) -> str:
    v = str(s or "").strip().lower()
    return {"male": "♂", "m": "♂", "female": "♀", "f": "♀"}.get(v, "")


def stage_of(s: str) -> str:
    return STAGES.get(str(s or "").strip().lower(), str(s or "").strip().lower())


def cite(r: dict) -> str:
    """One specimen, as a type citation: sex, life stage if not adult, repository and number."""
    sx = sex_symbol(r.get("sex"))
    st = stage_of(r.get("life_stage"))
    where = " ".join(x for x in (r.get("institution", ""), r.get("accession_number", "")) if x)
    head = " ".join(x for x in (sx, st) if x)
    out = ", ".join(x for x in (f"{head} {where}".strip(), r.get("locality", "")) if x)
    return out or r.get("specimen_id", "")


def summarise(rows: list) -> str:
    """Several specimens of one kind: counts by sex and stage, then the repositories."""
    by = defaultdict(int)
    for r in rows:
        by[" ".join(x for x in (sex_symbol(r.get("sex")) or "?",
                                stage_of(r.get("life_stage"))) if x)] += 1
    counts = ", ".join(f"{n} {k}" for k, n in sorted(by.items()))
    wheres = sorted({r.get("institution", "") for r in rows if r.get("institution")})
    return counts + (f" ({'; '.join(wheres)})" if wheres else "")


def statement(groups: dict) -> str:
    """The Material-examined opening, in the Code's order of precedence."""
    out = []
    for kind in sorted(groups, key=lambda k: KINDS[k]["rank"]):
        rows = groups[kind]
        if not rows:
            continue
        if KINDS[kind]["unique"] and len(rows) == 1:
            line = f"{LABEL[kind]}: {cite(rows[0])}"
            if kind == "allotype":
                line += " (a paratype under the Code)"
            purpose = str(rows[0].get("type_purpose", "") or "").strip()
            if kind in ("lectotype", "neotype") and purpose:
                line += f", {purpose}"
            out.append(line + ".")
        else:
            out.append(f"{LABEL[kind]}: {summarise(rows)}.")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser(description="Build the type-material statements from the designations")
    ap.add_argument("--types", required=True)
    ap.add_argument("--localities", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--acts", default=None,
                    help="an existing type-act register to append to (default: <out_dir>/type_acts.json). "
                         "Acts already carrying a publication in 'fixed_by' are never rewritten")
    ap.add_argument("--fixed_by", default=None,
                    help="the publication making these designations, e.g. 'Serbina & Van Dam, 2026'. "
                         "Recording it is what makes an act settled and protects it from later edits")
    ap.add_argument("--zoobank", default=None, help="ZooBank LSID of that work, if registered")
    ap.add_argument("--write_localities", action="store_true",
                    help="merge type_status / institution / accession_number into the localities "
                         "table (backed up first), which is what the exports read")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prof = yaml.safe_load(open(a.taxon_profile))
    species = prof.get("species") or {}
    t = pd.read_csv(a.types, sep="\t" if str(a.types).endswith(".tsv") else ",", dtype=str).fillna("")
    loc = pd.read_csv(a.localities, sep="\t", dtype=str).fillna("")
    lo_by_id = {r["specimen_id"]: dict(r) for _, r in loc.iterrows()}

    per_sp, problems = {}, []
    for code, rows in t.groupby("species"):
        meta = species.get(code) or {}
        status = meta.get("status", "")
        name = meta.get("name", code)
        new = status == "undescribed"
        groups = defaultdict(list)
        for _, r in rows.iterrows():
            d = dict(r)
            d.update({k: lo_by_id.get(d["specimen_id"], {}).get(k, "")
                      for k in ("locality", "country", "date", "collector", "sex", "life_stage",
                                "type_purpose")
                      if not d.get(k)})
            raw = d.get("type_status", "")
            if not str(raw).strip():
                continue
            k = kind_of(raw)
            if not k:
                problems.append(f"{code} ({name}): specimen {d['specimen_id']} has type status "
                                f"'{raw}', which is not one of: {', '.join(sorted(KINDS))}")
                continue
            if new and KINDS[k]["group"] == "existing":
                problems.append(f"{code} ({name}): {d['specimen_id']} is marked {k}, which applies to "
                                f"a name that already exists ({KINDS[k]['note']}); a species being "
                                f"described here takes a holotype or syntypes")
                continue
            groups[k].append(d)

        entry = {"species": code, "name": name, "status": status,
                 "counts": {k: len(v) for k, v in sorted(groups.items())}}
        if not new:
            # an existing name may still have a nomenclatural act performed on it here: a lectotype
            # fixed from a syntype series (Art. 74) or a neotype replacing lost types (Art. 75).
            # Both carry requirements of their own.
            for kind in ("lectotype", "neotype"):
                for r in groups.get(kind, []):
                    if not r.get("institution"):
                        problems.append(f"{code} ({name}): the {kind} {r['specimen_id']} has no "
                                        f"institution; Art. 74.7.4 / 75.3.7 require the collection "
                                        f"to be named")
                    # Art. 74.7.3: a lectotype designation after 1999 must state its purpose
                    if kind == "lectotype" and not str(r.get("type_purpose", "")).strip():
                        problems.append(f"{code} ({name}): the lectotype designation for "
                                        f"{r['specimen_id']} needs an express statement of purpose "
                                        f"(Art. 74.7.3) — put it in 'type_purpose', e.g. 'designated "
                                        f"to fix the application of the name'")
            if len(groups.get("lectotype", [])) > 1:
                problems.append(f"{code} ({name}): more than one lectotype; only one specimen can be "
                                f"the name-bearing type")
            entry["statement"] = statement(groups) if groups else ""
            entry["note"] = ("nomenclatural act on an existing name" if
                             (groups.get("lectotype") or groups.get("neotype")) else
                             "not being described as new; type designation optional")
            entry["specimens"] = {k: [r["specimen_id"] for r in v] for k, v in groups.items()}
            per_sp[code] = entry
            continue

        holo, syn = groups.get("holotype", []), groups.get("syntype", [])
        ok = True
        # Art. 73: a nominal species has EITHER one holotype OR a series of syntypes, never both
        if holo and syn:
            problems.append(f"{code} ({name}): both a holotype and {len(syn)} syntypes are marked. "
                            f"A species has one or the other (Art. 73), not both")
            ok = False
        elif len(holo) > 1:
            problems.append(f"{code} ({name}): {len(holo)} specimens marked holotype "
                            f"({', '.join(h['specimen_id'] for h in holo)}) — exactly one is allowed. "
                            f"If no single specimen can be chosen, mark them all syntype instead")
            ok = False
        elif not holo and len(syn) == 1:
            problems.append(f"{code} ({name}): a single syntype is not a valid type fixation; mark it "
                            f"holotype (Art. 73.1) or add the rest of the series")
            ok = False
        elif not holo and not syn:
            problems.append(f"{code} ({name}): no holotype or syntype series marked — the name cannot "
                            f"be made available (ICZN Art. 16.4.1)")
            ok = False

        for r in (holo + syn):
            if not r.get("institution"):
                problems.append(f"{code} ({name}): name-bearing specimen {r['specimen_id']} has no "
                                f"institution — the Code requires the collection to be named "
                                f"(Art. 16.4.2)")
                ok = False
            elif not r.get("accession_number"):
                problems.append(f"{code} ({name}): {r['specimen_id']} has no accession number; the "
                                f"collection should be able to supply one (recommended, not required)")

        allo = groups.get("allotype", [])
        if len(allo) > 1:
            problems.append(f"{code} ({name}): {len(allo)} allotypes; only one is conventional")
        if allo and holo and sex_symbol(allo[0].get("sex")) and \
                sex_symbol(allo[0].get("sex")) == sex_symbol(holo[0].get("sex")):
            problems.append(f"{code} ({name}): the allotype is the same sex as the holotype, which "
                            f"defeats its purpose (it denotes the opposite sex; Rec. 72A)")

        entry["statement"] = statement(groups) if ok else ""
        entry["specimens"] = {k: [r["specimen_id"] for r in v] for k, v in groups.items()}
        if "syntype" in entry["counts"]:
            entry["note"] = ("name fixed on a syntype series; a later worker may designate a "
                             "lectotype from it (Art. 74)")
        per_sp[code] = entry

    ready = [c for c, e in per_sp.items() if e.get("statement") and e.get("status") == "undescribed"]
    todo = [c for c, e in per_sp.items() if e.get("status") == "undescribed" and not e.get("statement")]
    report = {"version": VERSION, "species": len(per_sp),
              "to_be_described": sum(1 for e in per_sp.values() if e["status"] == "undescribed"),
              "ready": len(ready), "not_ready": len(todo), "problems": problems,
              "per_species": per_sp}
    # ── the append-only register of nomenclatural acts ────────────────────────────────────────
    acts_path = Path(a.acts) if a.acts else out / "type_acts.json"
    existing = []
    if acts_path.exists():
        try:
            existing = json.loads(acts_path.read_text()).get("acts", [])
        except Exception:  # noqa: BLE001
            existing = []
    incoming = []
    for code, e in per_sp.items():
        for kind, ids in (e.get("specimens") or {}).items():
            if KINDS[kind]["group"] == "any":          # non-type material is not an act
                continue
            for sid in ids:
                row = next((dict(r) for _, r in t.iterrows() if r["specimen_id"] == sid), {})
                incoming.append({"kind": kind, "specimen_id": sid, "species": code,
                                 "institution": row.get("institution", ""),
                                 "accession_number": row.get("accession_number", ""),
                                 "sex": row.get("sex", ""), "life_stage": row.get("life_stage", ""),
                                 "type_purpose": row.get("type_purpose", ""),
                                 # An act belongs to the publication that MADE it. For a species
                                 # being described here that is the present work (--fixed_by); for a
                                 # name that already exists the holotype was fixed by its original
                                 # author, so the sheet must say who, and the present work only
                                 # cites it. Stamping the current paper on an old type would credit
                                 # it with a designation it did not make.
                                 # A lectotype or neotype designated HERE is an act of the present
                                 # work even on an old name; a holotype or syntype of an old name
                                 # was fixed by its original author and is only cited here.
                                 "fixed_by": (row.get("fixed_by", "") or
                                              (a.fixed_by if (e.get("status") == "undescribed" or
                                                              KINDS[kind]["group"] == "existing")
                                               else "")),
                                 "fixed_year": (row.get("fixed_year", "") or
                                                ((a.fixed_by or "").split(",")[-1].strip()
                                                 if a.fixed_by and
                                                 (e.get("status") == "undescribed" or
                                                  KINDS[kind]["group"] == "existing") else "")),
                                 "zoobank_lsid": a.zoobank or row.get("zoobank_lsid", "")})
    for row in incoming:
        if not row["fixed_by"] and KINDS[row["kind"]]["group"] == "new":
            e = per_sp.get(row["species"], {})
            if e.get("status") != "undescribed":
                problems.append(
                    f"{row['species']} ({e.get('name', '')}): the {row['kind']} "
                    f"{row['specimen_id']} belongs to a name that already exists, so record the "
                    f"ORIGINAL author and year in 'fixed_by' (e.g. 'Pettey, 1924'). The present "
                    f"work cites this type, it does not designate it")
    acts, refused = merge_acts(existing, incoming)
    problems += refused
    acts_path.write_text(json.dumps(
        {"version": VERSION, "note": "append-only: a name-bearing type fixed by a publication is "
                                     "permanent (ICZN Art. 61.1). Later workers add material "
                                     "examined, or publish a lectotype / neotype act citing the "
                                     "original — they do not edit these rows.",
         "acts": sorted(acts, key=lambda x: (x["species"], KINDS[x["kind"]]["rank"],
                                             x["specimen_id"]))}, indent=2, ensure_ascii=False))
    report["acts_registered"] = len(acts)
    report["acts_file"] = str(acts_path)
    (out / "type_material.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lines = [f"Type material — {len(ready)} of {report['to_be_described']} new species ready", ""]
    for c in sorted(ready):
        lines += [f"  {c}: {per_sp[c]['statement']}"]
    if problems:
        lines += ["", "Still needed before any of these names can be published:"] + \
                 [f"  - {p}" for p in problems]
    (out / "type_material_report.txt").write_text("\n".join(lines))
    print("\n".join(lines[:40]))

    if a.write_localities:
        bak = Path(a.localities).with_suffix(".tsv.bak_pre_types")
        if not bak.exists():
            bak.write_text(Path(a.localities).read_text())
        cols = ["type_status", "institution", "accession_number", "life_stage", "type_purpose"]
        add = t.set_index("specimen_id")[[c for c in cols if c in t.columns]]
        for c in cols:
            if c not in add.columns:
                continue
            new_vals = loc["specimen_id"].map(add[c]).fillna("")
            if c in loc.columns:
                # never silently overwrite a designation that is already recorded: fill blanks only
                loc[c] = [old if str(old).strip() else nv for old, nv in zip(loc[c], new_vals)]
            else:
                loc[c] = new_vals
        loc.to_csv(a.localities, sep="\t", index=False)
        print(f"\nmerged {', '.join(cols)} into {a.localities} (backup {bak.name})")
    print(f"-> {out / 'type_material.json'}")


if __name__ == "__main__":
    main()
