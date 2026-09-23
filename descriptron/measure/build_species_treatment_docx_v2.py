#!/usr/bin/env python3
"""
build_species_treatment_docx_v2.py — monograph DOCX from BioRAG v2 outputs
=========================================================================

v2 of build_species_treatment_docx.py (v1 is kept unchanged and still used
for its map and formatting helpers). v2 assembles the monograph from:

  --descriptions-dir   output of biorag_description_refiner_v1.py
                       (<code>/<code>_treatment.json; falls back to v1 parsing
                       of TOWLEY/BioRAG .txt files for species without one)
  --key-dir            output of biorag_key_builder_v1.py (key_tree.json,
                       key_validation_report.json)
  --localities         verified localities TSV (see columns below)
  --plates-dir         consolidated plates with plate_index.tsv
  --taxon-profile      display names and species status

Treatment layout per species (publication order):
  heading (display name, italic) · key couplet reference
  Material examined   — verbatim slide-label text, structured fields, sex,
                        field code; type designations ONLY for species whose
                        profile status is 'undescribed', and labelled
                        "proposed (to be confirmed)"
  Diagnosis · Description (one paragraph per section) · Sexual dimorphism
  Host plant · Distribution (countries; map only from confirmed decimal
                        coordinates — verbatim coordinates are never parsed)
  Remarks · Figure (plate + caption)

Front matter: key (formatted couplets, species in italics) and a data-quality
note; back matter: provenance appendix (validation of key and descriptions,
names to verify, localities discrepancies, outlier flags).

Changes from v1 (bug fixes): Florence-2 model cache keyed by model name;
verbatim label text is printed even when structured fields exist; the
'verbatim_coordinates' column is not mapped to decimal coordinates.

Localities TSV columns used (header names exactly as written by the
Diaphorina_localities_verified.tsv): specimen_id, species, collaborator_code,
field_code, sex, country, locality, verbatim, verbatim_coordinates, latitude,
longitude, elevation, date, collector, host_plant, method, label_image,
source, discrepancy_notes.

Usage:
  python build_species_treatment_docx_v2.py \
    --descriptions-dir ".../Diaphorina_monograph/descriptions" \
    --key-dir ".../Diaphorina_monograph/key" \
    --localities ".../Diaphorina_monograph/localities/Diaphorina_localities_verified.tsv" \
    --plates-dir ".../Diaphorina_monograph/plates" \
    --taxon-profile biorag_prompts/taxon_profiles/diaphorina_taxon_profile.yaml \
    --title "..." --authors "..." \
    --output ".../Diaphorina_monograph/treatments/Diaphorina_monograph_treatments.docx"
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_species_treatment_docx as v1  # noqa: E402  (map + formatting helpers)
import biorag_feature_policy as pol  # noqa: E402

from docx import Document  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

BUILDER_V2 = "2.0"


# --- v1 bug fix: Florence-2 cache keyed by model name --------------------------
_FLORENCE_BY_MODEL = {}


def _load_florence2_keyed(model_name='microsoft/Florence-2-base'):
    if model_name in _FLORENCE_BY_MODEL:
        m, p, d = _FLORENCE_BY_MODEL[model_name]
        v1._FLORENCE.update({'model': m, 'processor': p, 'device': d})
        return m, p
    v1._FLORENCE.update({'model': None, 'processor': None})
    m, p = v1._orig_load_florence2(model_name)
    _FLORENCE_BY_MODEL[model_name] = (m, p, v1._FLORENCE['device'])
    return m, p


if not hasattr(v1, "_orig_load_florence2"):
    v1._orig_load_florence2 = v1._load_florence2
    v1._load_florence2 = _load_florence2_keyed


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────

def load_verified_localities(path):
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding='utf-8-sig', newline='') as f:
        delim = '\t' if '\t' in f.readline() else ','
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f, delimiter=delim):
            r = {k: (v or '').strip() for k, v in r.items() if k}
            r['lat'] = v1._to_float(r.get('latitude'))
            r['lon'] = v1._to_float(r.get('longitude'))
            rows.append(r)
    return rows


def load_plate_index(plates_dir, species_order=None, first_figure=1):
    """plate_index.tsv if present; otherwise <plates_dir>/<sp>/<sp>_plate.png numbered
    from first_figure in key/natural order (the index is then written for reuse)."""
    idx = {}
    p = Path(plates_dir or '') / 'plate_index.tsv'
    if plates_dir and not p.exists() and Path(plates_dir).is_dir():
        found = {d.name: d / f"{d.name}_plate.png" for d in Path(plates_dir).iterdir()
                 if d.is_dir() and (d / f"{d.name}_plate.png").exists()}
        order = [s for s in (species_order or []) if s in found] + \
            sorted((s for s in found if s not in (species_order or [])), key=natural_key)
        if order:
            with open(p, 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f, delimiter='\t')
                w.writerow(['figure_number', 'species', 'plate_file', 'n_panels', 'source_dir', 'note'])
                for i, sp in enumerate(order):
                    n = len(list((Path(plates_dir) / sp).glob('*_annotated.png')))
                    w.writerow([first_figure + i, sp, f"{sp}/{sp}_plate.png", n, str(Path(plates_dir) / sp),
                                'auto-indexed by build_species_treatment_docx_v2'])
    if p.exists():
        with open(p, encoding='utf-8') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                idx[r['species']] = {'figure': int(r['figure_number']),
                                     'png': str(Path(plates_dir) / r['plate_file']),
                                     'n_panels': r.get('n_panels', '')}
    return idx


def discover(descriptions_dir, only=None):
    found = OrderedDict()
    for name in sorted(os.listdir(descriptions_dir), key=natural_key):
        sub = Path(descriptions_dir) / name
        if not sub.is_dir() or (only and name not in only):
            continue
        tj = sub / f"{name}_treatment.json"
        tx = sub / f"{name}.txt"
        if tj.exists():
            found[name] = ('v2', tj)
        elif tx.exists() and v1._looks_like_description(str(tx)):
            found[name] = ('v1', tx)
    return found


def matrix_specimens(matrix_dir):
    out = defaultdict(dict)
    p = Path(matrix_dir or '') / 'specimen_matrix_long.csv'
    if p.exists():
        import pandas as pd
        d = pd.read_csv(p, usecols=['species', 'specimen_id', 'sex'])
        for sp, g in d.groupby('species'):
            s = g[g['sex'] != 'unknown'].groupby('specimen_id')['sex'].first()
            for sid in sorted(g['specimen_id'].unique(), key=natural_key):
                out[sp][sid] = s.get(sid, 'unknown')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

_ITALIC_NAME_CACHE = {}


def italic_names_markup(text, profile):
    """Wrap species names in italic markers (see add_runs)."""
    key = id(profile)
    if key not in _ITALIC_NAME_CACHE:
        names = set()
        genus = profile.get('taxon', {}).get('genus', '')
        for code in profile.get('species', {}):
            nm = pol.species_display_name(code, profile)
            names.add(nm)
            if genus and nm.startswith(genus + ' '):
                names.add(f"{genus[0]}. {nm[len(genus) + 1:]}")
        names = sorted(names, key=len, reverse=True)
        _ITALIC_NAME_CACHE[key] = re.compile('|'.join(re.escape(n) for n in names)) if names else None
    rx = _ITALIC_NAME_CACHE[key]
    if not rx or not text:
        return text or ''
    return rx.sub(lambda m: italicise_binomial(m.group(0)), text)


IT_OPEN, IT_CLOSE, B_OPEN, B_CLOSE = "\u27e8i\u27e9", "\u27e8/i\u27e9", "\u27e8b\u27e9", "\u27e8/b\u27e9"
_RUN_RX = re.compile(r'(\u27e8i\u27e9.*?\u27e8/i\u27e9|\u27e8b\u27e9.*?\u27e8/b\u27e9)')


def add_runs(paragraph, text):
    """Render ⟨i⟩…⟨/i⟩ and ⟨b⟩…⟨/b⟩ markers (asterisks such as CIE L* stay literal)."""
    for tok in _RUN_RX.split(text or ""):
        if not tok:
            continue
        if tok.startswith(IT_OPEN):
            paragraph.add_run(tok[len(IT_OPEN):-len(IT_CLOSE)]).italic = True
        elif tok.startswith(B_OPEN):
            paragraph.add_run(tok[len(B_OPEN):-len(B_CLOSE)]).bold = True
        else:
            paragraph.add_run(tok)


def italicise_binomial(name):
    """Italicise genus/epithet but not 'sp.', 'cf.' or quoted working names."""
    out = []
    for tok in name.split(' '):
        if tok in ('sp.', 'cf.', 'aff.') or tok.startswith("'") or re.fullmatch(r'\d+[A-Za-z]?', tok):
            out.append(tok)
        else:
            out.append(f"{IT_OPEN}{tok}{IT_CLOSE}")
    return ' '.join(out)


def material_line(r, status):
    sex = v1._sex_symbol(r.get('sex'))
    head = "1" + (sex or " specimen")
    bits = []
    if r.get('verbatim'):
        label = r['verbatim'].split(' | ')[0]
        bits.append(f'"{label}"')
    struct = [r.get('country'), r.get('locality')]
    if r.get('lat') is not None and r.get('lon') is not None:
        struct.append(v1._fmt_coord(r['lat'], r['lon']))
    if r.get('date'):
        struct.append(r['date'])
    if r.get('collector'):
        struct.append(f"leg. {r['collector']}")
    if r.get('method'):
        struct.append(r['method'])
    struct = [s for s in struct if s]
    if struct:
        bits.append(", ".join(struct))
    ids = [f"specimen {r['specimen_id']}"]
    if r.get('collaborator_code'):
        ids.append(f"slide {r['collaborator_code']}")
    if r.get('field_code'):
        ids.append(f"field code {r['field_code']}")
    return f"{head}: " + "; ".join(bits) + f" ({', '.join(ids)})."


def add_para(doc, text, profile, bold_label=None, size=None):
    p = doc.add_paragraph()
    if bold_label:
        r = p.add_run(bold_label + " ")
        r.bold = True
    add_runs(p, italic_names_markup(text, profile))
    if size:
        for run in p.runs:
            run.font.size = Pt(size)
    return p


def add_key(doc, key_tree, profile):
    v1._add_heading(doc, "Key to the species", 1)
    rep = key_tree.get('validation', {})
    note = (f"Dichotomous key computed from the hand-measurable data matrix "
            f"({rep.get('n_species')} species, {rep.get('n_couplets')} couplets). Numbers in parentheses "
            f"are the observed ranges and specimen counts. Couplets marked † use characters whose ranges "
            f"overlap slightly; check all characters listed. ♂/♀ mark characters that can be seen only "
            f"in that sex.")
    add_para(doc, note, profile, size=9)
    for r in key_tree['couplets']:
        dag = "†" if r['kind'] in ('overlap', 'unresolved') else ""
        for side in ('A', 'B'):
            g = r[f'{side}_goto']
            dest = str(g) if isinstance(g, int) else italicise_binomial(pol.species_display_name(g, profile))
            num = f"{r['number']}{dag}" if side == 'A' else "–"
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.45)
            p.paragraph_format.first_line_indent = Inches(-0.45)
            p.paragraph_format.space_after = Pt(1 if side == 'A' else 6)
            p.add_run(f"{num}\t")
            add_runs(p, italic_names_markup(r[f'{side}_text'].rstrip('.'), profile))
            p.add_run(" … ")
            add_runs(p, dest)
            for run in p.runs:
                run.font.size = Pt(9.5)


# ─────────────────────────────────────────────────────────────────────────────
# Main assembly
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Monograph DOCX from BioRAG v2 outputs")
    ap.add_argument('--descriptions-dir', required=True)
    ap.add_argument('--key-dir', default=None)
    ap.add_argument('--localities', default=None)
    ap.add_argument('--plates-dir', default=None)
    ap.add_argument('--matrix-dir', default=None, help="Tier-1 matrix (lists specimens examined)")
    ap.add_argument('--taxon-profile', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--species', nargs='*', default=None)
    ap.add_argument('--title', default=None)
    ap.add_argument('--authors', default=None)
    ap.add_argument('--map-mode', choices=['auto', 'cartopy', 'simple', 'none'], default='auto')
    ap.add_argument('--coastline-geojson', default=None)
    ap.add_argument('--include-not-assessable', action='store_true')
    ap.add_argument('--no-plates', action='store_true', help="Omit plate images (small file for review)")
    ap.add_argument('--first-figure', type=int, default=1,
                    help="First figure number when plates have no plate_index.tsv")
    ap.add_argument('--emit-locality-template', default=None, metavar='TSV',
                    help="Also write a blank verified-localities TSV (one row per specimen in the matrix)")
    args = ap.parse_args()
    n_types = pol.load_type_material(args.descriptions_dir) if hasattr(pol, "load_type_material") else 0
    if n_types:
        print(f"type material: {n_types} species have a designated type statement")

    profile = pol.load_taxon_profile(args.taxon_profile)
    genus = profile.get('taxon', {}).get('genus', '')
    found = discover(args.descriptions_dir, set(args.species) if args.species else None)
    if not found:
        sys.exit("No treatments found.")
    order = []
    if args.key_dir and (Path(args.key_dir) / 'key_tree.json').exists():
        order = list(json.loads((Path(args.key_dir) / 'key_tree.json').read_text()).get('species', {}))
    plates = load_plate_index(args.plates_dir, order or list(found), args.first_figure)
    locs = load_verified_localities(args.localities)
    loc_by_sp = defaultdict(list)
    for r in locs:
        loc_by_sp[r.get('species', '')].append(r)
    mspec = matrix_specimens(args.matrix_dir)
    if args.emit_locality_template:
        cols = ["specimen_id", "species", "collaborator_code", "field_code", "sex", "country", "locality",
                "verbatim", "verbatim_coordinates", "latitude", "longitude", "elevation", "date",
                "collector", "host_plant", "method", "label_image", "source", "discrepancy_notes"]
        Path(args.emit_locality_template).parent.mkdir(parents=True, exist_ok=True)
        with open(args.emit_locality_template, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, cols, delimiter='\t')
            w.writeheader()
            for sp in sorted(mspec, key=natural_key):
                for sid, sx in mspec[sp].items():
                    w.writerow({"specimen_id": sid, "species": sp, "sex": sx if sx != "unknown" else "",
                                "source": "fill from the slide/pin labels; leave unknown fields blank"})
        print(f"Locality template: {args.emit_locality_template}")
    key_tree = None
    if args.key_dir and (Path(args.key_dir) / 'key_tree.json').exists():
        key_tree = json.loads((Path(args.key_dir) / 'key_tree.json').read_text())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = out.parent / 'treatment_figures'
    fig_dir.mkdir(exist_ok=True)

    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(11)

    if args.title:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(args.title)
        r.bold = True
        r.font.size = Pt(16)
    if args.authors:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(args.authors)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Draft for co-author review — generated {datetime.now():%Y-%m-%d} with Descriptron "
                  f"BioRAG v2 (build_species_treatment_docx_v2 {BUILDER_V2}). Not for citation.")
    r.italic = True
    r.font.color.rgb = RGBColor(0x77, 0x77, 0x77)
    verify = [f"{c} → {v.get('name')}" for c, v in profile.get('species', {}).items()
              if isinstance(v, dict) and v.get('verify')]
    add_para(doc, "Names are working names from the specimen folders and collaborator notes; "
                  f"{len(verify)} of {len(profile.get('species', {}))} require confirmation before "
                  "publication (see Appendix). Type designations are shown only for undescribed "
                  "species and only as proposals.", profile, size=9)
    doc.add_page_break()

    if key_tree:
        add_key(doc, key_tree, profile)
        doc.add_page_break()

    points = {}
    for sp in found:
        pts = [(r['lat'], r['lon']) for r in loc_by_sp.get(sp, []) if r['lat'] is not None and r['lon'] is not None]
        if pts:
            points[sp] = pts

    val_rows = []
    for sp, (kind, path) in found.items():
        name = pol.species_display_name(sp, profile)
        status = pol.species_status(sp, profile)
        v1._add_heading(doc, "", 1)
        h = doc.paragraphs[-1]
        add_runs(h, italicise_binomial(name))
        if key_tree:
            refs = [f"{r['number']}{s.lower()}" for r in key_tree['couplets'] for s in ('A', 'B')
                    if r[f'{s}_goto'] == sp]
            if refs:
                add_para(doc, f"Key couplet {', '.join(refs)}.", profile, size=9)
        if sp in plates:
            add_para(doc, f"Figure {plates[sp]['figure']}.", profile, size=9)

        # Material examined
        v1._add_heading(doc, "Material examined", 2)
        recs = sorted(loc_by_sp.get(sp, []), key=lambda r: natural_key(r['specimen_id']))
        if status == 'undescribed' and recs:
            tm = pol.type_material(sp) if hasattr(pol, "type_material") else ""
            add_para(doc, tm if tm else
                     "Proposed type material (to be confirmed by the authors; holotype not yet "
                     "designated).", profile, size=9)
        for r in recs:
            add_para(doc, material_line(r, status), profile)
        listed = {r['specimen_id'] for r in recs}
        rest = [s for s in mspec.get(sp, {}) if s not in listed]
        if rest:
            txt = ", ".join(f"{s} ({v1._sex_symbol(mspec[sp][s]) or '?'})" for s in rest)
            add_para(doc, f"Specimens in the morphometric data set without locality data: {txt}.",
                     profile)
        if not recs and not rest:
            add_para(doc, "No specimen data supplied.", profile)

        if kind == 'v2':
            t = json.loads(Path(path).read_text())
            v1._add_heading(doc, "Diagnosis", 2)
            add_para(doc, t.get('diagnosis', '') or '[missing]', profile)
            v1._add_heading(doc, "Description", 2)
            for d in t.get('description', []) or []:
                if d.get('text', '').strip():
                    add_para(doc, d['text'].strip(), profile, bold_label=f"{d.get('section', '')}.")
            if t.get('sexual_dimorphism', '').strip():
                v1._add_heading(doc, "Sexual dimorphism", 2)
                add_para(doc, t['sexual_dimorphism'], profile)
            if args.include_not_assessable and t.get('not_assessable'):
                add_para(doc, "; ".join(t['not_assessable']) + ".", profile,
                         bold_label="Not assessable from the material:", size=9)
            v = t.get('_validation', {})
            val_rows.append((sp, len(v.get('attempts', [])), len(v.get('final_problems', [])),
                             len(v.get('removed_sentences', [])), len(t.get('citations', []))))
            remarks = t.get('remarks', '')
        else:
            tr = v1.parse_treatment(str(path))
            v1._add_heading(doc, "Diagnosis", 2)
            add_para(doc, tr.get('synthesis_diagnosis') or '[v1 treatment — not refined]', profile)
            remarks = "Treatment not yet refined with the v2 evidence policy."
            val_rows.append((sp, 0, -1, 0, 0))

        hosts = sorted({r['host_plant'] for r in recs if r.get('host_plant')})
        if hosts:
            v1._add_heading(doc, "Host plant", 2)
            add_para(doc, "; ".join(hosts) + " (slide labels).", profile)
        v1._add_heading(doc, "Distribution", 2)
        countries = sorted({r['country'] for r in recs if r.get('country')})
        add_para(doc, (", ".join(countries) + ".") if countries else "No locality data supplied.", profile)
        if sp in points and args.map_mode != 'none':
            png = fig_dir / f"{sp}_map.png"
            try:
                v1.draw_range_map({sp: points[sp]}, str(png), name, args.map_mode, args.coastline_geojson)
                doc.add_picture(str(png), width=Inches(4.5))
            except Exception as e:  # noqa: BLE001
                print(f"  map failed for {sp}: {e}", file=sys.stderr)
        notes = sorted({r['discrepancy_notes'] for r in recs if r.get('discrepancy_notes')})
        v1._add_heading(doc, "Remarks", 2)
        add_para(doc, remarks or "—", profile)
        for n in notes:
            add_para(doc, f"Label data note: {n}.", profile, size=9)

        if sp in plates and not args.no_plates and os.path.isfile(plates[sp]['png']):
            doc.add_picture(plates[sp]['png'], width=Inches(6.0))
            cap = doc.add_paragraph()
            add_runs(cap, f"{B_OPEN}Figure {plates[sp]['figure']}.{B_CLOSE} {italicise_binomial(name)}: "
                                   f"specimen images with annotated structures (coloured overlays) used "
                                   f"for measurement ({plates[sp]['n_panels']} panels).")
            for run in cap.runs:
                run.font.size = Pt(9)
        doc.add_page_break()

    # Appendix
    v1._add_heading(doc, "Appendix — provenance and checks", 1)
    if key_tree:
        rep = key_tree['validation']
        add_para(doc, (f"All species reachable: {rep.get('all_species_reachable')}; species at more than "
                       f"one terminal: {len(rep.get('species_with_multiple_terminals', []))}; E_Dicho "
                       f"{rep.get('e_dicho')}; mean path {rep.get('mean_steps')} couplets (max "
                       f"{rep.get('max_steps')}); threshold violations: "
                       f"{len(rep.get('threshold_violations_on_perfect_characters', []))}; statistical terms "
                       f"in leads: {len(rep.get('tier2_terms_in_leads', []))}; identification of the "
                       f"measured specimens (specimens with data at every couplet): "
                       f"{100 * (rep.get('identification_resubstitution_resolved_only') or 0):.0f}% (same data), "
                       f"{100 * (rep.get('identification_leave_one_out_resolved_only') or 0):.0f}% "
                       f"(leave-one-specimen-out)."),
                 profile, bold_label="Key.")
    ok = sum(1 for r in val_rows if r[2] == 0)
    add_para(doc, (f"{ok} of {len(val_rows)} treatments passed every automatic check (all numbers "
                   f"traced to the data matrix; no statistical terms outside Remarks). Numbers cited per "
                   f"treatment: {', '.join(f'{pol.species_display_name(r[0], profile)} {r[4]}' for r in val_rows)}."),
             profile, bold_label="Descriptions.")
    sj = Path(args.descriptions_dir) / "subjective_check_summary.json"
    st = Path(args.descriptions_dir) / "subjective_character_flags.tsv"
    if sj.exists() and st.exists():
        import pandas as pd
        summ = json.loads(sj.read_text())
        kinds = summ.get("by_kind_status", {})
        add_para(doc, ("Qualitative words read from the structure images are kept in the treatments. "
                       "They were flagged in the machine-readable outputs (<code>_treatment.jsonld, "
                       "subjective_character_flags.tsv) for future versions of Descriptron. Colour words were "
                       "compared with the measured colour (KD-tree colour names, CIE L*a*b*): "
                       + "; ".join(f"{k.replace('/', ': ').replace('_', ' ')} {v}" for k, v in sorted(kinds.items()))
                       + ". Texture, shape and setation words have no quantitative lookup yet and are marked "
                         "as unverified."), profile, bold_label="Subjective characters.")
        fl = pd.read_csv(st, sep="\t")
        rev = fl[fl["status"].isin(["contradicts", "check_image"])]
        for r in rev.itertuples():
            img = f" Image: {Path(r.image_checked).name}." if isinstance(r.image_checked, str) and r.image_checked else ""
            add_para(doc, (f"{pol.species_display_name(r.species, profile)}, {r.structure}, {r.status.replace('_', ' ')}: "
                           f"\u201c{r.text[:160]}\u201d — measured L* {r.measured_L}, a* {r.measured_a}, "
                           f"b* {r.measured_b} (nearest named colour: {r.kdtree_colour_name}).{img}"),
                     profile, size=8)
    add_para(doc, "; ".join(verify) + ".", profile, bold_label="Names to verify.", size=9)
    doc.save(str(out))
    print(f"Wrote {out} ({len(found)} treatments; {len(points)} species with mapped coordinates)")


if __name__ == "__main__":
    main()
