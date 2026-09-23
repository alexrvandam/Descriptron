#!/usr/bin/env python3
"""
biorag_collaborator_forms_v1.py — the two things only a collaborator can supply
==============================================================================

A monograph built from images still needs two things that are not in any image:
what each species is actually called, and where each specimen came from. This
writes one workbook to send out, pre-filled with everything the pipeline already
knows so the recipient only fills the blanks.

  Sheet "1 Species names"   one row per species; the recipient fills final_name
                            (and optionally authority / status / notes)
  Sheet "2 Localities"      one row per specimen that has no locality yet; the
                            recipient fills country, locality, coordinates,
                            elevation, date, collector, host, method
  Sheet "3 Already supplied"  the specimens that are complete, as a worked
                            example of the format expected
  Sheet "4 How to fill this in"  notes, including what must not be changed

Everything is also written as TSV beside the workbook, for anyone who would
rather not open Excel.

  python biorag_collaborator_forms_v1.py --monograph "$M" \\
      --taxon_profile <profile.yaml> --out_dir "$M/collaborator_forms"
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

VERSION = "1.0"

NAME_FILL = ["final_name", "authority", "status_confirmed", "notes_from_collaborator"]
# ICZN Art. 16.4: a new species-group name published after 1999 needs an explicitly fixed holotype
# (16.4.1) and a statement of the collection where it is deposited (16.4.2). That is per SPECIMEN,
# not per species, so it is asked for on the specimen sheet.
TYPE_FILL = ["type_status", "life_stage", "institution", "accession_number", "fixed_by",
             "type_purpose", "notes_from_collaborator"]
# offered as dropdowns in the workbook so the spelling always matches what the pipeline parses
TYPE_STATUS = ["holotype", "syntype", "allotype", "paratype", "lectotype", "paralectotype",
               "neotype", "non-type"]
LIFE_STAGE = ["adult", "nymph", "larva", "pupa", "egg", "immature"]
LOC_FILL = ["country", "locality", "latitude_decimal", "longitude_decimal", "elevation_m",
            "date_collected", "collector", "host_plant", "method", "notes_from_collaborator"]
# The Methods section of a data paper cannot be reconstructed from the files: only the people who
# prepared and photographed the specimens know these. Each question names why it is being asked, so
# the answer comes back in a usable form rather than as "Leica".
METHODS_Q = [
    ("SPECIMEN PREPARATION", ""),
    ("How were the specimens preserved before mounting?", "e.g. 70% ethanol, dried and pinned, frozen"),
    ("Clearing: what agent, what concentration, how long, at what temperature?", "e.g. 10% KOH, 12 h, room temperature"),
    ("Was anything stained? With what?", ""),
    ("Mounting medium", "e.g. Canada balsam, Euparal, Hoyer's"),
    ("Which structures were dissected and mounted separately?", "e.g. terminalia, wings, legs on separate slides"),
    ("Who prepared the slides, and when?", "for the acknowledgements and the data provenance"),
    ("IMAGING", ""),
    ("Microscope: make and model", ""),
    ("Camera: make and model", ""),
    ("Objective magnifications used, and for which body parts", "e.g. x20 head and terminalia, x10 wings"),
    ("Illumination", "e.g. transmitted brightfield, DIC, phase, incident"),
    ("Was focus stacking used? Software and typical number of slices", "e.g. Helicon Focus, 30-60 slices"),
    ("How was the scale bar produced?", "e.g. burned in by the capture software from a stage-micrometer calibration. Every measurement in millimetres depends on this"),
    ("Image format, bit depth and pixel dimensions", "e.g. 16-bit TIFF, 2048x2048"),
    ("COLOUR - was any white balance or colour standard used?", "e.g. a grey card, an X-Rite target, or the software's auto white balance. This decides how far the CIE L*a*b* values can be compared BETWEEN images, and if the answer is 'none' we must say so as a limitation - so please answer even if the answer is no"),
    ("Capture software and version", ""),
    ("Was any image adjusted after capture?", "e.g. levels, sharpening, background removal - and if so, before or after the measurements were taken"),
    ("DEPOSITION AND RIGHTS", ""),
    ("Which institution holds the slides?", ""),
    ("Who owns the images, and under what licence may they be published?", "e.g. CC BY 4.0 - a data paper requires an open licence"),
    ("Anyone to be acknowledged or added as an author?", ""),
]

HELP = [
    ("What we need", "Two things no image can supply: the accepted name of each species, and where "
                     "each specimen was collected."),
    ("Sheet 1 — Species names",
     "One row per species. Please fill 'final_name' with the name the monograph should print. Leave "
     "it blank if the species is still undescribed and the working name should stand. 'authority' "
     "is the author and year for a described species."),
    ("Sheet 2 — Localities",
     "One row per specimen still without locality data. Fill whatever is known; blanks are fine and "
     "are better than a guess. Coordinates in decimal degrees please, negative for South and West "
     "(for example -4.328 for 4.328 S)."),
    ("Please do not change", "The 'code' and 'specimen_id' columns. They are the keys that join "
                             "these rows to the measurements, images and descriptions."),
    ("If a label is ambiguous", "Say so in the notes column rather than resolving it silently. One "
                                "label in this set gave coordinates with no hemisphere, and that "
                                "ambiguity is recorded in the monograph rather than hidden."),
    ("Sheet 3 — Type designations",
     "For a species being described as new: mark exactly ONE specimen 'holotype' and the others "
     "'paratype', and give the institution and its accession number. If no single specimen can be "
     "chosen, mark them all 'syntype' instead — a syntype series is valid (Art. 73.2), it simply "
     "leaves a lectotype to be fixed later. 'allotype' marks a specimen of the opposite sex to the "
     "holotype; under the Code it is really a paratype, so it is optional. Use 'life_stage' when "
     "the specimen is not an adult, so that a nymphal or larval type is cited correctly."),
    ("Sheet 3 — existing names",
     "For a species that already has a name, you may also fix a 'lectotype' from an old syntype "
     "series (Art. 74) or a 'neotype' where the original types are lost (Art. 75). A lectotype "
     "designation made now must state WHY, so fill 'type_purpose' — for example 'designated to fix "
     "the application of the name' (Art. 74.7.3). Leave the whole sheet blank for species where no "
     "type act is intended."),
    ("Sheet 3 — 'fixed_by'",
     "Leave this blank for a species being described as new: the present work is what fixes the "
     "type. For a species that ALREADY has a name, put the original author and year (for example "
     "'Pettey, 1924'), because that publication designated the type and this one only cites it. A "
     "redescription never creates types and never adds paratypes — any further specimens are "
     "'non-type' material examined."),
    ("Sheet 5 — Methods",
     "These are for the methods section of the data paper and only you can answer them. Please type "
     "the answer in the 'answer' column; 'not recorded' is a perfectly good answer and is much more "
     "useful to us than a guess, because we will state it as a limitation rather than imply a "
     "precision the data does not have."),
    ("A personal collection is not enough",
     "The repository must be a collection that maintains a research collection and is likely to "
     "persist (Rec. 16C). If the specimens are not yet accessioned, say when they will be — this "
     "is the step that sets the timetable for publication."),
    ("Sheet 4", "Specimens already complete, as an example of the level of detail that is useful."),
]


def main():
    ap = argparse.ArgumentParser(description="Build the collaborator forms for names and localities")
    ap.add_argument("--monograph", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--matrix_dir", default=None, help="default: <monograph>/compiled_key_tier")
    ap.add_argument("--localities", default=None,
                    help="default: <monograph>/localities/*_localities_verified.tsv")
    ap.add_argument("--project", default=None, help="name used in the file names and the help sheet")
    a = ap.parse_args()

    M = Path(a.monograph)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = Path(a.matrix_dir) if a.matrix_dir else M / "compiled_key_tier"
    lp = Path(a.localities) if a.localities else next(
        iter(sorted((M / "localities").glob("*_localities_verified.tsv"))), None)
    project = a.project or M.name

    prof = yaml.safe_load(open(a.taxon_profile))
    species = prof.get("species") or {}
    long = pd.read_csv(md / "specimen_matrix_long.csv")
    spec = (long[["species", "specimen_id", "sex"]].drop_duplicates("specimen_id")
            .sort_values(["species", "specimen_id"]))
    loc = pd.read_csv(lp, sep="\t", dtype=str).fillna("") if lp and lp.exists() else pd.DataFrame()
    # the image file names are the only handle a collaborator has on a specimen we hold no label for,
    # so carry a couple of them into the form
    images = {}
    full = next(iter(sorted(md.glob("*_full_features.csv"))), None)
    if full is not None:
        f = pd.read_csv(full, low_memory=False, usecols=lambda c: c in ("specimen_id", "image_base"))
        if {"specimen_id", "image_base"} <= set(f.columns):
            images = (f.dropna().groupby("specimen_id")["image_base"]
                      .apply(lambda x: "; ".join(sorted(set(x))[:2])).to_dict())

    have_loc = set()
    if len(loc):
        for _, r in loc.iterrows():
            if r.get("country") or r.get("locality") or r.get("latitude"):
                have_loc.add(r["specimen_id"])
    by_id = {r["specimen_id"]: r for _, r in loc.iterrows()} if len(loc) else {}

    # ── sheet 1: names ────────────────────────────────────────────────────────
    n_spec = spec.groupby("species")["specimen_id"].nunique().to_dict()
    rows = []
    for code, meta in sorted(species.items()):
        meta = meta if isinstance(meta, dict) else {}
        sub = loc[loc["species"] == code] if len(loc) else pd.DataFrame()

        def first(col):
            v = [x for x in sub.get(col, []) if x] if len(sub) else []
            return v[0] if v else ""
        rows.append({"code": code, "our_working_name": meta.get("name", ""),
                     "our_status": meta.get("status", ""),
                     "we_need_this_confirmed": "yes" if meta.get("verify") else "no",
                     "specimens": n_spec.get(code, 0),
                     "country_known": first("country"), "locality_known": first("locality"),
                     "host_known": first("host_plant"),
                     "collector_determination": first("determination"),
                     **{c: "" for c in NAME_FILL}})
    names = pd.DataFrame(rows)

    # ── sheet 2: localities still wanted, and sheet 3: those already supplied ──
    want, done = [], []
    for _, s in spec.iterrows():
        sid = s["specimen_id"]
        r = by_id.get(sid, {})
        base = {"specimen_id": sid, "species": s["species"],
                "our_working_name": (species.get(s["species"]) or {}).get("name", ""),
                "sex": s["sex"] if isinstance(s["sex"], str) else "",
                "slide_code": r.get("collaborator_code", ""), "field_code": r.get("field_code", ""),
                "image_files": images.get(sid, ""),
                "label_text_we_have": r.get("verbatim", "")}
        if sid in have_loc:
            done.append({**base, "country": r.get("country", ""), "locality": r.get("locality", ""),
                         "latitude_decimal": r.get("latitude", ""),
                         "longitude_decimal": r.get("longitude", ""),
                         "elevation_m": r.get("elevation", ""), "date_collected": r.get("date", ""),
                         "collector": r.get("collector", ""), "host_plant": r.get("host_plant", ""),
                         "method": r.get("method", ""), "note": r.get("discrepancy_notes", "")})
        else:
            want.append({**base, **{c: "" for c in LOC_FILL}})
    want_df, done_df = pd.DataFrame(want), pd.DataFrame(done)
    methods_df = pd.DataFrame([{"question": q, "answer": "", "why we ask / example": h}
                               for q, h in METHODS_Q])
    # every specimen is a candidate type, so the type sheet covers them all
    types_df = pd.DataFrame([{"specimen_id": r["specimen_id"], "species": r["species"],
                              "our_working_name": r["our_working_name"],
                              "our_status": (species.get(r["species"]) or {}).get("status", ""),
                              "sex": r["sex"], "image_files": r.get("image_files", ""),
                              **{c: "" for c in TYPE_FILL}}
                             for r in (want + done)])
    help_df = pd.DataFrame(HELP, columns=["", "  "])

    stem = f"{project}_collaborator_form_{date.today():%Y%m%d}"
    for df, name in ((names, "species_names"), (want_df, "localities_needed"),
                     (types_df, "type_designations"), (methods_df, "methods_questions"),
                     (done_df, "localities_supplied")):
        df.to_csv(out / f"{stem}_{name}.tsv", sep="\t", index=False)

    xlsx = out / f"{stem}.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
            names.to_excel(xw, sheet_name="1 Species names", index=False)
            want_df.to_excel(xw, sheet_name="2 Localities", index=False)
            types_df.to_excel(xw, sheet_name="3 Type designations", index=False)
            methods_df.to_excel(xw, sheet_name="4 Methods", index=False)
            done_df.to_excel(xw, sheet_name="5 Already supplied", index=False)
            help_df.to_excel(xw, sheet_name="6 How to fill this in", index=False)
            from openpyxl.styles import Alignment, Font, PatternFill
            fill_need = PatternFill("solid", fgColor="FFF2CC")      # columns to complete
            fill_head = PatternFill("solid", fgColor="D9E2F3")
            for sheet, df, to_fill in (("1 Species names", names, NAME_FILL),
                                       ("2 Localities", want_df, LOC_FILL),
                                       ("3 Type designations", types_df, TYPE_FILL),
                                       ("4 Methods", methods_df, ["answer"]),
                                       ("5 Already supplied", done_df, []),
                                       ("6 How to fill this in", help_df, [])):
                ws = xw.sheets[sheet]
                ws.freeze_panes = "A2"
                for j, col in enumerate(df.columns, start=1):
                    width = max(14, min(52, int(df[col].astype(str).str.len().max() or 0) + 4,
                                        len(str(col)) + 6) if len(df) else len(str(col)) + 6)
                    ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = width
                    h = ws.cell(row=1, column=j)
                    h.font = Font(bold=True)
                    h.fill = fill_need if col in to_fill else fill_head
                    h.alignment = Alignment(wrap_text=True, vertical="center")
                # dropdowns, so a typo cannot reach the pipeline
                if sheet.startswith("3"):
                    from openpyxl.worksheet.datavalidation import DataValidation
                    for col, choices in (("type_status", TYPE_STATUS), ("life_stage", LIFE_STAGE)):
                        if col not in df.columns:
                            continue
                        j = list(df.columns).index(col) + 1
                        letter = ws.cell(row=1, column=j).column_letter
                        dv = DataValidation(type="list",
                                            formula1='"' + ",".join(choices) + '"',
                                            allow_blank=True, showDropDown=False)
                        dv.error = "Please choose one of the listed values."
                        dv.promptTitle, dv.prompt = col, "Choose from the list"
                        ws.add_data_validation(dv)
                        dv.add(f"{letter}2:{letter}{len(df) + 1}")
                if sheet.startswith("4 Methods"):
                    ws.column_dimensions["A"].width = 62
                    ws.column_dimensions["B"].width = 46
                    ws.column_dimensions["C"].width = 70
                    for row in ws.iter_rows(min_row=1, max_col=3):
                        for c in row:
                            c.alignment = Alignment(wrap_text=True, vertical="top")
                    for i, (q, _h) in enumerate(METHODS_Q, start=2):
                        if q.isupper():                      # section banners
                            ws.cell(row=i, column=1).font = Font(bold=True)
                if sheet.startswith("6"):
                    ws.column_dimensions["A"].width = 30
                    ws.column_dimensions["B"].width = 110
                    for row in ws.iter_rows(min_row=1, max_col=2):
                        for c in row:
                            c.alignment = Alignment(wrap_text=True, vertical="top")
        print(f"workbook -> {xlsx}")
    except Exception as e:  # noqa: BLE001
        print(f"  (no workbook written: {e}; the TSVs are there)")

    print(f"  sheet 1: {len(names)} species, {int((names['we_need_this_confirmed'] == 'yes').sum())} "
          f"needing a confirmed name")
    print(f"  sheet 2: {len(want_df)} specimens with no locality yet "
          f"({want_df['species'].nunique()} species)")
    print(f"  sheet 3: {len(types_df)} specimens to designate as holotype / paratype")
    print(f"  sheet 4: {sum(1 for q, _ in METHODS_Q if not q.isupper())} methods questions")
    print(f"  sheet 5: {len(done_df)} specimens already supplied "
          f"({done_df['species'].nunique() if len(done_df) else 0} species)")
    print(f"  TSVs    -> {out}")


if __name__ == "__main__":
    main()
