#!/usr/bin/env python3
"""
build_species_treatment_docx.py

Assemble a review-ready Microsoft Word (.docx) taxonomic treatment from the
outputs of the Descriptron-v2 / TOWLEY pipeline.

For each species it collates, in publication order:

    1. Heading            (Genus + species epithet/code, italicised)
    2. Materials Examined  (from a flexible localities CSV/TSV; type series first)
    3. Diagnosis           (per-category DIAGNOSIS blocks from {species}.txt)
    4. Description          (per-category DESCRIPTION blocks from {species}.txt)
    5. Distribution         (a basic range map drawn from GPS coordinates,
                             plus a plain-text locality list)
    6. Figure               (the species plate PNG + caption)

A single dichotomous Key section (from taxonomic_key.txt) is inserted once,
before the individual treatments.

The locality table may supply coordinates / locality fields *directly*, or a
column of specimen-label image paths.  With --ocr-labels the script runs
Florence-2 on those label images to recover verbatim text and coordinates
(basic OCR; see label_locality_ocr()).

This is a standalone, argparse-driven CLI.  It is also invoked as a button in
the Descriptron-v2 GUI (conda env: measure_env).

Example
-------
    python build_species_treatment_docx.py \
        --descriptions-dir "/media/.../Diaphorina_towley_descriptions/descriptions_with_figures" \
        --plates-dir       "/media/.../Diaphorina_towley_descriptions/species_plates" \
        --key-file         "/media/.../Diaphorina_towley_descriptions/taxonomic_key.txt" \
        --localities        localities.csv \
        --genus            Diaphorina \
        --title            "A revision of the Diaphorina of East Africa" \
        --authors          "A.E.Z. Short" \
        --output           Diaphorina_treatments.docx
"""

import argparse
import csv
import os
import re
import sys
from collections import OrderedDict, defaultdict

# ---------------------------------------------------------------------------
# python-docx (required)
# ---------------------------------------------------------------------------
try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    sys.exit("ERROR: python-docx is required.  `pip install python-docx` "
             "(it is present in the measure_env conda environment).")


# ===========================================================================
# 1. Coordinate parsing
# ===========================================================================

# Decimal like  -1.234   or   36.789 E
_DEC_RE = re.compile(
    r'(?P<val>[-+]?\d{1,3}(?:\.\d+)?)\s*[°]?\s*(?P<hemi>[NSEWnsew])?')

# Degrees-minutes-seconds like  1°14'30"S  or  36 49 12 E
_DMS_RE = re.compile(
    r'(?P<deg>\d{1,3})\s*[°:\s]\s*(?P<min>\d{1,2})\s*[\'′:\s]\s*'
    r'(?P<sec>\d{1,2}(?:\.\d+)?)?\s*["″]?\s*(?P<hemi>[NSEWnsew])')


def _dms_to_decimal(deg, minutes, seconds, hemi):
    dec = float(deg) + float(minutes) / 60.0 + (float(seconds) if seconds else 0.0) / 3600.0
    if hemi and hemi.upper() in ('S', 'W'):
        dec = -dec
    return dec


def parse_coordinate_pair(text):
    """Best-effort extraction of a (lat, lon) decimal pair from free text.

    Returns (lat, lon) floats or (None, None).  Handles decimal and DMS.
    This is intentionally forgiving; label OCR is noisy.
    """
    if not text:
        return None, None
    text = str(text).strip()

    # First try DMS (two matches expected: lat then lon).
    dms = list(_DMS_RE.finditer(text))
    if len(dms) >= 2:
        vals = {}
        for m in dms[:2]:
            dec = _dms_to_decimal(m.group('deg'), m.group('min'),
                                  m.group('sec'), m.group('hemi'))
            axis = 'lat' if m.group('hemi').upper() in ('N', 'S') else 'lon'
            vals[axis] = dec
        if 'lat' in vals and 'lon' in vals:
            return vals['lat'], vals['lon']

    # Fall back to a decimal pair separated by comma / whitespace / slash.
    parts = re.split(r'[;,/]|\s{2,}', text)
    if len(parts) < 2:
        parts = text.split()
    nums = []
    for p in parts:
        m = re.search(r'[-+]?\d{1,3}\.\d+', p)
        if m:
            hemi = re.search(r'[NSEWnsew]', p)
            v = float(m.group())
            if hemi and hemi.group().upper() in ('S', 'W'):
                v = -abs(v)
            nums.append(v)
    if len(nums) >= 2:
        # Heuristic: latitude is the value with abs <= 90.
        a, b = nums[0], nums[1]
        if abs(a) <= 90 and abs(b) <= 180:
            return a, b
        if abs(b) <= 90:
            return b, a
    return None, None


def _to_float(x):
    try:
        if x is None or str(x).strip() == '':
            return None
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


# ===========================================================================
# 2. Locality table loading (flexible header matching)
# ===========================================================================

# canonical field -> list of accepted header aliases (lower-cased, punctuation stripped)
_FIELD_ALIASES = {
    'specimen_id': ['specimen_id', 'specimenid', 'specimen', 'catalog',
                    'catalognumber', 'catalog_number', 'id', 'code'],
    'species':     ['species', 'species_name', 'taxon', 'sp'],
    'type_status': ['type_status', 'typestatus', 'type', 'status'],
    'sex':         ['sex'],
    'country':     ['country'],
    'state':       ['state', 'stateprovince', 'state_province', 'province', 'region'],
    'locality':    ['locality', 'verbatimlocality', 'place', 'site', 'location'],
    'elevation':   ['elevation', 'elev', 'altitude', 'minimumelevationinmeters'],
    'habitat':     ['habitat', 'ecology'],
    'latitude':    ['latitude', 'lat', 'decimallatitude', 'decimal_latitude'],
    'longitude':   ['longitude', 'lon', 'lng', 'long', 'decimallongitude',
                    'decimal_longitude'],
    'coordinates': ['coordinates', 'coords', 'latlon', 'lat_lon', 'gps',
                    'verbatimcoordinates'],
    'date':        ['date', 'eventdate', 'event_date', 'collection_date',
                    'collectiondate', 'datecollected'],
    'collector':   ['collector', 'recordedby', 'recorded_by', 'leg', 'legit'],
    'label_image': ['label_image', 'label_image_path', 'labelimage', 'label',
                    'label_path', 'labelpath', 'image', 'image_path'],
    'verbatim':    ['verbatim', 'verbatim_label', 'verbatimlabel', 'label_text',
                    'labeltext'],
}


def _norm_header(h):
    return re.sub(r'[^a-z0-9]', '', str(h).lower())


def _build_header_map(fieldnames):
    """Map canonical field -> actual column name present in the file."""
    lookup = {}
    normed = {_norm_header(f): f for f in fieldnames if f is not None}
    for canon, aliases in _FIELD_ALIASES.items():
        for a in aliases:
            if a in normed:
                lookup[canon] = normed[a]
                break
    return lookup


def _sniff_delimiter(path):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        sample = f.readline()
    if '\t' in sample:
        return '\t'
    if sample.count(';') > sample.count(','):
        return ';'
    return ','


def load_localities(path, species_from_specimen=None):
    """Load a localities CSV/TSV into a list of dicts keyed by canonical fields.

    species_from_specimen: optional callable(specimen_id) -> species, used when
    the file has no explicit species column.
    """
    if not path or not os.path.isfile(path):
        return []
    delim = _sniff_delimiter(path)
    rows = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=delim)
        hmap = _build_header_map(reader.fieldnames or [])
        for raw in reader:
            rec = {canon: (raw.get(col) or '').strip()
                   for canon, col in hmap.items()}
            # Resolve coordinates.
            lat = _to_float(rec.get('latitude'))
            lon = _to_float(rec.get('longitude'))
            if (lat is None or lon is None) and rec.get('coordinates'):
                lat2, lon2 = parse_coordinate_pair(rec['coordinates'])
                lat = lat if lat is not None else lat2
                lon = lon if lon is not None else lon2
            rec['lat'] = lat
            rec['lon'] = lon
            # Resolve species.
            if not rec.get('species') and species_from_specimen:
                rec['species'] = species_from_specimen(rec.get('specimen_id', ''))
            rows.append(rec)
    return rows


def default_species_from_specimen(specimen_id):
    """Diaphorina convention: 'acok3' -> 'acok', 'sp1A_2' -> 'sp1A'.

    Strips a trailing specimen number (optionally after an underscore).
    """
    if not specimen_id:
        return ''
    s = specimen_id.strip()
    # 'kenya_6' or 'kenya6' -> 'kenya'
    m = re.match(r'^(.*?)[_-]?(\d+)$', s)
    if m and m.group(1):
        return m.group(1).rstrip('_-')
    return s


# ===========================================================================
# 3. Florence-2 label OCR (optional)
# ===========================================================================

_FLORENCE = {'model': None, 'processor': None}


def _load_florence2(model_name='microsoft/Florence-2-base'):
    """Lazy-load Florence-2 (mirrors measurement_script V35 loader)."""
    if _FLORENCE['model'] is not None:
        return _FLORENCE['model'], _FLORENCE['processor']
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True).to(device)
    _FLORENCE['model'] = model
    _FLORENCE['processor'] = proc
    _FLORENCE['device'] = device
    return model, proc


def florence_ocr(image_path, model_name='microsoft/Florence-2-base'):
    """Run Florence-2 <OCR> on a label image, returning the recognised text."""
    import torch
    from PIL import Image
    model, proc = _load_florence2(model_name)
    device = _FLORENCE['device']
    image = Image.open(image_path).convert('RGB')
    prompt = '<OCR>'
    inputs = proc(text=prompt, images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        gen = model.generate(input_ids=inputs['input_ids'],
                             pixel_values=inputs['pixel_values'],
                             max_new_tokens=1024, num_beams=3)
    text = proc.batch_decode(gen, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(text, task='<OCR>',
                                          image_size=(image.width, image.height))
    return parsed.get('<OCR>', '').strip()


def ocr_fill_localities(rows, model_name='microsoft/Florence-2-base',
                        enriched_out=None):
    """For rows lacking coordinates but carrying a label image, OCR the label
    and fill verbatim text + parsed coordinates.  Returns the same rows.

    Basic/best-effort: the label reader is a known-imperfect step (per project
    notes).  Rows that fail are left unchanged and reported.
    """
    n_ocr = n_coord = 0
    for rec in rows:
        img = rec.get('label_image')
        if not img or (rec.get('lat') is not None and rec.get('lon') is not None):
            continue
        if not os.path.isfile(img):
            print(f"  [ocr] label image not found: {img}", file=sys.stderr)
            continue
        try:
            text = florence_ocr(img, model_name)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  [ocr] failed on {img}: {e}", file=sys.stderr)
            continue
        n_ocr += 1
        if text:
            rec['verbatim'] = (rec.get('verbatim') or text).strip()
            lat, lon = parse_coordinate_pair(text)
            if lat is not None and lon is not None:
                rec['lat'], rec['lon'] = lat, lon
                n_coord += 1
    print(f"  [ocr] read {n_ocr} labels, recovered coordinates for {n_coord}.")
    if enriched_out:
        _write_enriched_localities(rows, enriched_out)
        print(f"  [ocr] enriched localities written to {enriched_out}")
    return rows


def _write_enriched_localities(rows, out_path):
    cols = ['specimen_id', 'species', 'type_status', 'sex', 'country', 'state',
            'locality', 'lat', 'lon', 'elevation', 'date', 'collector',
            'verbatim', 'label_image']
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


# ===========================================================================
# 4. Description .txt parsing
#
# Two formats exist in the pipeline output and both must be supported:
#
#   TOWLEY  (older, e.g. acok.txt):   header "Method: TOWLEY"; body is a series
#           of "Category: X" blocks, each with DIAGNOSIS:/DESCRIPTION:/TRAITS:.
#
#   BioRAG  (newer, deconfabulated 4-species, e.g. kenya.txt): header
#           "SPECIES DESCRIPTION: X" / "Method: BioRAG"; sections are an
#           UPPERCASE title on one line followed by a line of dashes:
#               TYPE MATERIAL / SYNTHESIS DIAGNOSIS / DIAGNOSIS (per category) /
#               DESCRIPTION / <CAT> — Trait details ...
#
# Both are normalised into one "treatment" dict:
#   { format, type_material:[{type_status,specimen_id,sex}],
#     synthesis_diagnosis:str,
#     diagnosis_blocks:[(cat,txt)], description_blocks:[(cat,txt)],
#     traits_blocks:[(cat,txt)] }
# ===========================================================================

_SEP_RE = re.compile(r'^={20,}\s*$')
_DASH_RE = re.compile(r'^-{10,}\s*$')
_CAT_RE = re.compile(r'^Category:\s*(?P<name>.+?)\s*(?:\((?P<mode>[^)]+)\))?\s*$')
_TYPE_RE = re.compile(
    r'^(?P<status>HOLOTYPE|ALLOTYPE|NEOTYPE|LECTOTYPE):\s*'
    r'(?P<id>.+?)\s*(?:\((?P<sex>[^)]+)\))?\s*$')
_PARATYPE_RE = re.compile(r'^PARATYPES?\s*(?:\(\d+\))?:\s*(?P<body>.+)$')
_SPECIMEN_SEX_RE = re.compile(r'^(?P<id>.+?)\s*(?:\((?P<sex>[^)]+)\))?\s*$')


def detect_format(head_text):
    """'towley' or 'biorag' from the first lines of a description file."""
    if 'Method: BioRAG' in head_text or head_text.lstrip().startswith(
            'SPECIES DESCRIPTION:'):
        return 'biorag'
    return 'towley'


def parse_treatment(path):
    """Parse a {species}.txt (either format) into the normalised treatment dict."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    fmt = detect_format('\n'.join(lines[:6]))
    if fmt == 'biorag':
        return _parse_biorag(lines)
    return _parse_towley(lines)


def _empty_treatment(fmt):
    return {'format': fmt, 'type_material': [], 'synthesis_diagnosis': '',
            'diagnosis_blocks': [], 'description_blocks': [], 'traits_blocks': []}


def _parse_towley(lines):
    t = _empty_treatment('towley')
    i, n = 0, len(lines)
    while i < n:
        m = _CAT_RE.match(lines[i])
        if not m:
            i += 1
            continue
        cat = m.group('name').strip()
        i += 1
        body = []
        while i < n and not _CAT_RE.match(lines[i]):
            body.append(lines[i])
            i += 1
        sections = _split_sections(body)
        if sections.get('DIAGNOSIS', '').strip():
            t['diagnosis_blocks'].append((cat, sections['DIAGNOSIS'].strip()))
        if sections.get('DESCRIPTION', '').strip():
            t['description_blocks'].append((cat, sections['DESCRIPTION'].strip()))
        if sections.get('TRAITS', '').strip():
            t['traits_blocks'].append((cat, sections['TRAITS'].strip()))
    return t


def _split_sections(body_lines):
    """Split a TOWLEY category body into DIAGNOSIS / DESCRIPTION / TRAITS text."""
    sections = OrderedDict()
    current = None
    buf = []
    header_re = re.compile(r'^(DIAGNOSIS|DESCRIPTION|TRAITS)\s*:(.*)$')
    for ln in body_lines:
        if _SEP_RE.match(ln) or ln.startswith('[See '):
            continue
        h = header_re.match(ln)
        if h:
            if current is not None:
                sections[current] = '\n'.join(buf).strip()
            current = h.group(1)
            rest = h.group(2).strip()
            buf = [rest] if rest else []
        elif current is not None:
            buf.append(ln)
    if current is not None:
        sections[current] = '\n'.join(buf).strip()
    return sections


def _biorag_sections(lines):
    """Yield (header, body_lines) for BioRAG '<HEADER>\\n----' sections."""
    sections = []
    i, n = 0, len(lines)
    while i < n:
        if i + 1 < n and _DASH_RE.match(lines[i + 1]) and lines[i].strip():
            header = lines[i].strip()
            j = i + 2
            body = []
            while j < n and not (j + 1 < n and _DASH_RE.match(lines[j + 1])
                                 and lines[j].strip()):
                body.append(lines[j])
                j += 1
            sections.append((header, body))
            i = j
        else:
            i += 1
    return sections


def _parse_biorag(lines):
    t = _empty_treatment('biorag')
    for header, body in _biorag_sections(lines):
        text = '\n'.join(body).strip()
        up = header.upper()
        if up == 'TYPE MATERIAL':
            t['type_material'] = _parse_type_material(body)
        elif up == 'SYNTHESIS DIAGNOSIS':
            t['synthesis_diagnosis'] = text
        elif up.startswith('DIAGNOSIS'):
            if text:
                t['diagnosis_blocks'].append(('per character', text))
        elif up == 'DESCRIPTION':
            if text:
                t['description_blocks'].append(('', text))
        elif up.endswith('TRAIT DETAILS') or up.endswith('— TRAIT DETAILS'):
            cat = re.sub(r'\s*[—-]\s*Trait details\s*$', '', header,
                         flags=re.IGNORECASE).strip()
            if text:
                t['traits_blocks'].append((cat, text))
    return t


def _parse_type_material(body_lines):
    """Parse TYPE MATERIAL lines into [{type_status, specimen_id, sex}]."""
    out = []
    for ln in body_lines:
        ln = ln.strip()
        if not ln:
            continue
        m = _TYPE_RE.match(ln)
        if m:
            out.append({'type_status': m.group('status').lower(),
                        'specimen_id': m.group('id').strip(),
                        'sex': (m.group('sex') or '').strip()})
            continue
        p = _PARATYPE_RE.match(ln)
        if p:
            for chunk in p.group('body').split(','):
                sm = _SPECIMEN_SEX_RE.match(chunk.strip())
                if sm and sm.group('id'):
                    out.append({'type_status': 'paratype',
                                'specimen_id': sm.group('id').strip(),
                                'sex': (sm.group('sex') or '').strip()})
    return out


def prettify_category(name):
    """'circumanal_ring' -> 'Circumanal ring'; keep LAB1 etc. as-is."""
    if not name:
        return ''
    if name.isupper() or re.match(r'^[A-Z]{2,}\d*$', name):
        return name
    return name.replace('_', ' ').strip().capitalize()


# --- Specimen enumeration (for the blank locality template) -----------------

_CANDIDATE_ID_RE = re.compile(r'[A-Za-z][A-Za-z0-9.]*[_-]?\d+[A-Za-z]?')
_TYPE_DESIG_RE = re.compile(
    r'\b(holotype|allotype|paratype|neotype|lectotype)s?\b[^.;()]{0,30}\(([^)]*)\)',
    re.IGNORECASE)


def _specimen_matches(token, species):
    """True if `token` is a specimen id of `species` (e.g. kenya_6, acok3, sp1_2).

    Requires a separator when the species name ends in a digit, so 'sp11' is NOT
    mistaken for specimen 1 of 'sp1'.
    """
    t = token.strip()
    tl, sl = t.lower(), species.lower()
    for sep in ('_', '-'):
        pre = sl + sep
        if tl.startswith(pre) and re.fullmatch(r'\d+[A-Za-z]?', t[len(pre):]):
            return True
    if not species[-1:].isdigit() and tl.startswith(sl):
        if re.fullmatch(r'\d+[A-Za-z]?', t[len(species):]):
            return True
    return False


def collect_specimens(species, treatment):
    """Enumerate specimen records for `species` from its parsed treatment.

    Uses the structured TYPE MATERIAL (BioRAG) plus type designations and
    specimen-id tokens mined from the description text (TOWLEY).  Returns
    [{specimen_id, species, type_status, sex}] with the type series first.
    """
    specimens = OrderedDict()

    def add(sid, ts='', sex=''):
        sid = (sid or '').strip()
        if not sid:
            return
        cur = specimens.setdefault(sid, {'type_status': '', 'sex': ''})
        if ts and not cur['type_status']:
            cur['type_status'] = ts.lower()
        if sex and not cur['sex']:
            cur['sex'] = sex

    for tm in treatment.get('type_material', []):
        add(tm.get('specimen_id', ''), tm.get('type_status', ''), tm.get('sex', ''))

    texts = [treatment.get('synthesis_diagnosis', '')]
    for k in ('diagnosis_blocks', 'description_blocks', 'traits_blocks'):
        texts += [t for _, t in treatment.get(k, [])]
    blob = '\n'.join(texts)

    for m in _TYPE_DESIG_RE.finditer(blob):
        ts, inside = m.group(1), m.group(2)
        for tok in _CANDIDATE_ID_RE.findall(inside):
            if _specimen_matches(tok, species):
                add(tok, ts)
    for tok in _CANDIDATE_ID_RE.findall(blob):
        if _specimen_matches(tok, species):
            add(tok)

    def order(item):
        return (_TYPE_ORDER.get(item[1]['type_status'], 99), item[0])
    return [{'specimen_id': sid, 'species': species,
             'type_status': meta['type_status'], 'sex': meta['sex']}
            for sid, meta in sorted(specimens.items(), key=order)]


_TEMPLATE_COLS = ['specimen_id', 'species', 'type_status', 'sex', 'country',
                  'state', 'locality', 'latitude', 'longitude', 'elevation',
                  'date', 'collector', 'label_image', 'verbatim']


def write_locality_template(species_txt, out_path):
    """Write a blank localities CSV (one row per discovered specimen) to fill in."""
    rows = []
    for sp, txt in species_txt.items():
        rows.extend(collect_specimens(sp, parse_treatment(txt)))
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_TEMPLATE_COLS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in _TEMPLATE_COLS})
    return len(rows)


# ===========================================================================
# 5. Range map drawing
# ===========================================================================

def _species_color(idx):
    import matplotlib
    try:
        cmap = matplotlib.colormaps['tab20']
    except (AttributeError, KeyError):
        cmap = matplotlib.cm.get_cmap('tab20')
    return cmap(idx % 20)


def _auto_extent(points, margin=2.0):
    lons = [p[1] for p in points]
    lats = [p[0] for p in points]
    return (min(lons) - margin, max(lons) + margin,
            min(lats) - margin, max(lats) + margin)


def _draw_basemap(ax, extent, map_mode, coastline_geojson):
    """Draw coastlines/borders on ax.  Returns True if a basemap was drawn."""
    if map_mode in ('auto', 'cartopy'):
        try:
            import cartopy.crs as ccrs  # noqa: F401
            import cartopy.feature as cfeature
            ax.add_feature(cfeature.LAND, facecolor='#f2efe9')
            ax.add_feature(cfeature.OCEAN, facecolor='#dce6f0')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#888888')
            gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                              color='#cccccc', alpha=0.6)
            gl.top_labels = gl.right_labels = False
            return 'cartopy'
        except Exception:
            if map_mode == 'cartopy':
                print("  [map] cartopy requested but unavailable; falling back.",
                      file=sys.stderr)
    # Simple fallback: optional GeoJSON coastline + graticule.
    if coastline_geojson and os.path.isfile(coastline_geojson):
        _draw_geojson(ax, coastline_geojson)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(True, linewidth=0.3, color='#cccccc', alpha=0.6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal', adjustable='box')
    return 'simple'


def _draw_geojson(ax, geojson_path):
    """Draw a GeoJSON of country/coastline polygons with pure matplotlib."""
    import json
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    with open(geojson_path, 'r', encoding='utf-8') as f:
        gj = json.load(f)
    patches = []

    def add_ring(ring):
        if len(ring) >= 3:
            patches.append(MplPolygon(ring, closed=True))

    for feat in gj.get('features', []):
        geom = feat.get('geometry', {})
        gtype = geom.get('type')
        coords = geom.get('coordinates', [])
        if gtype == 'Polygon':
            for ring in coords:
                add_ring(ring)
        elif gtype == 'MultiPolygon':
            for poly in coords:
                for ring in poly:
                    add_ring(ring)
    if patches:
        pc = PatchCollection(patches, facecolor='#f2efe9',
                             edgecolor='#999999', linewidths=0.4)
        ax.add_collection(pc)


def draw_range_map(points_by_species, out_png, title, map_mode='auto',
                   coastline_geojson=None, extent=None, highlight=None,
                   dpi=200):
    """Draw a range map PNG.

    points_by_species: {species: [(lat, lon), ...]}
    highlight: if given, only that species is drawn in strong colour, others grey.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_pts = [pt for pts in points_by_species.values() for pt in pts]
    if not all_pts:
        return None
    if extent is None:
        extent = _auto_extent(all_pts)

    use_cartopy = map_mode in ('auto', 'cartopy')
    fig = plt.figure(figsize=(6.5, 5.0))
    proj_kw = {}
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            proj_kw = {'transform': ccrs.PlateCarree()}
        except Exception:
            ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.add_subplot(1, 1, 1)

    _draw_basemap(ax, extent, map_mode, coastline_geojson)

    for idx, (sp, pts) in enumerate(sorted(points_by_species.items())):
        if not pts:
            continue
        lons = [p[1] for p in pts]
        lats = [p[0] for p in pts]
        if highlight is not None and sp != highlight:
            ax.scatter(lons, lats, s=14, c='#bbbbbb', edgecolors='none',
                       zorder=3, **proj_kw)
        else:
            color = 'crimson' if highlight is not None else _species_color(idx)
            ax.scatter(lons, lats, s=40, color=[color], edgecolors='black',
                       linewidths=0.4, zorder=4,
                       label=sp if highlight is None else None, **proj_kw)

    if highlight is None and len(points_by_species) > 1:
        ax.legend(loc='best', fontsize=6, framealpha=0.8, markerscale=0.8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_png


# ===========================================================================
# 6. Materials Examined formatting
# ===========================================================================

_TYPE_ORDER = {'holotype': 0, 'lectotype': 0, 'neotype': 0,
               'allotype': 1, 'paratype': 2, 'paralectotype': 2}
_SEX_SYMBOL = {'male': '♂', 'm': '♂', '♂': '♂',
               'female': '♀', 'f': '♀', '♀': '♀'}


def _sex_symbol(sex):
    if not sex:
        return ''
    return _SEX_SYMBOL.get(str(sex).strip().lower(), str(sex).strip())


def _fmt_coord(lat, lon):
    if lat is None or lon is None:
        return ''
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    return f"{abs(lat):.4f}°{ns} {abs(lon):.4f}°{ew}"


def format_material_entry(rec):
    """Render one specimen record as a taxonomic 'material examined' string."""
    parts = []
    ts = (rec.get('type_status') or '').strip()
    sex = _sex_symbol(rec.get('sex'))
    lead = ts.capitalize() if ts else 'Material'
    tag = f"{lead}"
    if sex:
        tag += f" {sex}"
    sid = rec.get('specimen_id', '')
    if sid:
        tag += f" ({sid})"
    parts.append(tag + ':')

    loc_bits = []
    for k in ('country', 'state', 'locality'):
        v = rec.get(k)
        if v:
            loc_bits.append(v)
    coord = _fmt_coord(rec.get('lat'), rec.get('lon'))
    if coord:
        loc_bits.append(coord)
    if rec.get('elevation'):
        loc_bits.append(f"{rec['elevation']} m")
    if rec.get('date'):
        loc_bits.append(rec['date'])
    if rec.get('collector'):
        loc_bits.append(f"leg. {rec['collector']}")
    body = ', '.join(loc_bits)
    if body:
        parts.append(body + '.')
    elif rec.get('verbatim'):
        parts.append(f'"{rec["verbatim"]}".')
    return ' '.join(parts)


def sort_material(records):
    """Type series first (holotype, allotype, paratypes), then other material."""
    def key(r):
        ts = (r.get('type_status') or '').strip().lower()
        return (_TYPE_ORDER.get(ts, 99), r.get('specimen_id', ''))
    return sorted(records, key=key)


# ===========================================================================
# 7. DOCX assembly
# ===========================================================================

def _add_heading(doc, text, level, italic=False):
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    run.italic = italic
    return h


_MD_RE = re.compile(r'(\*\*.+?\*\*|\*[^*]+?\*)')


def _add_rich_runs(paragraph, text):
    """Render inline markdown (**bold**, *italic*) as docx runs.

    The BioRAG descriptions use **category** and *Genus* markdown heavily.
    """
    for tok in _MD_RE.split(text):
        if not tok:
            continue
        if tok.startswith('**') and tok.endswith('**') and len(tok) > 4:
            paragraph.add_run(tok[2:-2]).bold = True
        elif tok.startswith('*') and tok.endswith('*') and len(tok) > 2:
            paragraph.add_run(tok[1:-1]).italic = True
        else:
            paragraph.add_run(tok)


def _add_rich_paragraph(doc, text, style=None):
    p = doc.add_paragraph(style=style)
    _add_rich_runs(p, text)
    return p


def _add_label_para(doc, label, text, italic_label=True):
    p = doc.add_paragraph()
    r = p.add_run(f"{label} ")
    r.bold = True
    r.italic = italic_label
    _add_rich_runs(p, text)
    return p


def _add_monospace_block(doc, text):
    for line in text.splitlines():
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        run = p.add_run(line if line else ' ')
        run.font.name = 'Courier New'
        run.font.size = Pt(8)


def build_docx(species_data, key_text, output_path, genus='',
               title=None, authors=None, overview_map=None,
               include_traits=False):
    """Assemble the full treatment document.

    species_data: ordered list of dicts with keys:
        species, treatment (normalised dict), materials (list of strings),
        plate_png, plate_caption, range_map_png, locality_lines
    """
    doc = Document()

    # --- Title page ---------------------------------------------------------
    if title:
        tp = doc.add_paragraph()
        tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = tp.add_run(title)
        r.bold = True
        r.font.size = Pt(18)
    if authors:
        ap = doc.add_paragraph()
        ap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        ap.add_run(authors).font.size = Pt(12)
    if title or authors:
        note = doc.add_paragraph()
        note.alignment = WD_ALIGN_PARAGRAPH.CENTER
        nr = note.add_run("Draft taxonomic treatments — for review, not for citation")
        nr.italic = True
        nr.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        doc.add_page_break()

    # --- Overview distribution map -----------------------------------------
    if overview_map and os.path.isfile(overview_map):
        _add_heading(doc, "Distribution of all treated species", 1)
        doc.add_picture(overview_map, width=Inches(6.0))
        cap = doc.add_paragraph()
        cr = cap.add_run("Figure 1. Collection localities of all treated species.")
        cr.italic = True
        cr.font.size = Pt(9)
        doc.add_page_break()

    # --- Key ----------------------------------------------------------------
    if key_text:
        _add_heading(doc, "Key to species", 1)
        _add_monospace_block(doc, key_text)
        doc.add_page_break()

    # --- Per-species treatments --------------------------------------------
    for sp in species_data:
        name = sp['species']
        display = f"{genus} {name}".strip() if genus else name
        _add_heading(doc, display, 1, italic=True)

        # Materials Examined
        _add_heading(doc, "Material examined", 2)
        if sp['materials']:
            for m in sp['materials']:
                doc.add_paragraph(m, style=None)
        else:
            doc.add_paragraph("No locality data supplied.")

        tr = sp['treatment']

        # Diagnosis (BioRAG synthesis diagnosis first, then per-category)
        _add_heading(doc, "Diagnosis", 2)
        wrote_diag = False
        if tr.get('synthesis_diagnosis'):
            _add_rich_paragraph(doc, tr['synthesis_diagnosis'])
            wrote_diag = True
        for cat, txt in tr.get('diagnosis_blocks', []):
            label = prettify_category(cat)
            if label:
                _add_label_para(doc, label + '.', txt)
            else:
                _add_rich_paragraph(doc, txt)
            wrote_diag = True
        if not wrote_diag:
            doc.add_paragraph("[No diagnosis text available.]")

        # Description
        _add_heading(doc, "Description", 2)
        desc = tr.get('description_blocks', [])
        if desc:
            for cat, txt in desc:
                label = prettify_category(cat)
                if label:
                    _add_label_para(doc, label + '.', txt)
                else:
                    _add_rich_paragraph(doc, txt)
        else:
            doc.add_paragraph("[No description text available.]")

        if include_traits and tr.get('traits_blocks'):
            _add_heading(doc, "Traits (structured)", 3)
            for cat, txt in tr['traits_blocks']:
                _add_monospace_block(doc, f"{cat}:\n{txt}")

        # Distribution
        _add_heading(doc, "Distribution", 2)
        if sp.get('locality_lines'):
            for line in sp['locality_lines']:
                doc.add_paragraph(line, style='List Bullet')
        if sp.get('range_map_png') and os.path.isfile(sp['range_map_png']):
            doc.add_picture(sp['range_map_png'], width=Inches(5.0))
            cap = doc.add_paragraph()
            cr = cap.add_run(f"Range map of {display}.")
            cr.italic = True
            cr.font.size = Pt(9)
        elif not sp.get('locality_lines'):
            doc.add_paragraph("[No georeferenced records.]")

        # Figure plate
        if sp.get('plate_png') and os.path.isfile(sp['plate_png']):
            _add_heading(doc, "Figure", 2)
            doc.add_picture(sp['plate_png'], width=Inches(6.0))
            if sp.get('plate_caption'):
                cap = doc.add_paragraph()
                cr = cap.add_run(sp['plate_caption'])
                cr.italic = True
                cr.font.size = Pt(9)

        doc.add_page_break()

    doc.save(output_path)
    return output_path


# ===========================================================================
# 8. Orchestration
# ===========================================================================

def _looks_like_description(txt_path):
    """A real species description opens with a TOWLEY or BioRAG header.
    Guards against report/check subdirs (confabulation_report/, key_num_check/,
    ...) that also hold a like-named txt.
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            head = f.read(400)
    except OSError:
        return False
    h = head.lstrip()
    return (h.startswith('Species:') or 'Method: TOWLEY' in head
            or h.startswith('SPECIES DESCRIPTION:') or 'Method: BioRAG' in head)


def discover_species(descriptions_dir, only=None):
    """Return ordered {species: txt_path} for subdirs containing a real
    TOWLEY/BioRAG {name}.txt description."""
    found = OrderedDict()
    if not os.path.isdir(descriptions_dir):
        sys.exit(f"ERROR: descriptions dir not found: {descriptions_dir}")
    for name in sorted(os.listdir(descriptions_dir)):
        sub = os.path.join(descriptions_dir, name)
        if not os.path.isdir(sub):
            continue
        txt = os.path.join(sub, f"{name}.txt")
        if os.path.isfile(txt) and _looks_like_description(txt):
            if only and name not in only:
                continue
            found[name] = txt
    return found


def load_figure_captions(plates_dir):
    """Best-effort load of a figure_index.csv / figure_captions.csv mapping
    species -> (plate_path, caption)."""
    captions = {}
    for cand in ('figure_index.csv', 'figure_captions.csv'):
        for base in (plates_dir, os.path.dirname(plates_dir.rstrip('/'))):
            p = os.path.join(base, cand)
            if os.path.isfile(p):
                with open(p, 'r', encoding='utf-8-sig', newline='') as f:
                    for row in csv.DictReader(f):
                        sp = (row.get('species') or '').strip()
                        if sp:
                            captions[sp] = (row.get('plate_path', ''),
                                            row.get('caption', ''))
                if captions:
                    return captions
    return captions


def main():
    ap = argparse.ArgumentParser(
        description="Assemble a review-ready DOCX taxonomic treatment "
                    "(Materials Examined, Diagnosis, Description, Key, range maps).")
    ap.add_argument('--descriptions-dir', required=True,
                    help="Dir with {species}/{species}.txt (e.g. descriptions_with_figures).")
    ap.add_argument('--plates-dir', default=None,
                    help="Dir with {species}/{species}_plate.png.")
    ap.add_argument('--key-file', default=None, help="taxonomic_key.txt.")
    ap.add_argument('--localities', default=None,
                    help="CSV/TSV of specimen localities (coords and/or label images).")
    ap.add_argument('--output', default=None, help="Output .docx path.")
    ap.add_argument('--emit-locality-template', default=None, metavar='CSV',
                    help="Instead of building the DOCX, scan the descriptions and "
                         "write a blank localities CSV (one row per specimen) here.")
    ap.add_argument('--species', nargs='*', default=None,
                    help="Restrict to these species codes (default: all found).")
    ap.add_argument('--genus', default='', help="Genus name for headings/citations.")
    ap.add_argument('--title', default=None)
    ap.add_argument('--authors', default=None)
    ap.add_argument('--include-traits', action='store_true',
                    help="Append the structured TRAITS blocks (verbose).")
    # Maps
    ap.add_argument('--map-mode', choices=['auto', 'cartopy', 'simple', 'none'],
                    default='auto')
    ap.add_argument('--coastline-geojson', default=None,
                    help="GeoJSON of coastlines/countries for the simple basemap.")
    ap.add_argument('--map-extent', nargs=4, type=float, default=None,
                    metavar=('WEST', 'EAST', 'SOUTH', 'NORTH'),
                    help="Fixed map extent; default auto-fit to points.")
    ap.add_argument('--figures-dir', default=None,
                    help="Where to write generated map PNGs (default: alongside output).")
    # Label OCR
    ap.add_argument('--ocr-labels', action='store_true',
                    help="Run Florence-2 on label images lacking coordinates.")
    ap.add_argument('--florence-model', default='microsoft/Florence-2-base')
    args = ap.parse_args()

    if not args.output and not args.emit_locality_template:
        ap.error("provide --output (to build the DOCX) or "
                 "--emit-locality-template (to write a blank localities CSV).")

    # --- Species ------------------------------------------------------------
    only = set(args.species) if args.species else None
    species_txt = discover_species(args.descriptions_dir, only)
    if not species_txt:
        sys.exit("ERROR: no {species}/{species}.txt found under descriptions dir.")
    print(f"Found {len(species_txt)} species: {', '.join(species_txt)}")

    # --- Locality template mode (write blank CSV and exit) ------------------
    if args.emit_locality_template:
        n = write_locality_template(species_txt, args.emit_locality_template)
        print(f"Wrote locality template with {n} specimen rows to "
              f"{args.emit_locality_template}")
        return

    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.output)) or '.', 'treatment_figures')
    os.makedirs(figures_dir, exist_ok=True)

    # --- Localities ---------------------------------------------------------
    localities = load_localities(args.localities,
                                 species_from_specimen=default_species_from_specimen)
    if localities and args.ocr_labels:
        enriched = os.path.join(figures_dir, 'localities_enriched.csv')
        ocr_fill_localities(localities, args.florence_model, enriched)
    loc_by_species = defaultdict(list)
    for rec in localities:
        sp = (rec.get('species') or '').strip()
        if sp:
            loc_by_species[sp].append(rec)
    if localities:
        print(f"Loaded {len(localities)} locality records "
              f"({sum(1 for r in localities if r.get('lat') is not None)} georeferenced).")

    # --- Points for maps ----------------------------------------------------
    points_by_species = {}
    for sp in species_txt:
        pts = [(r['lat'], r['lon']) for r in loc_by_species.get(sp, [])
               if r.get('lat') is not None and r.get('lon') is not None]
        if pts:
            points_by_species[sp] = pts

    overview_map = None
    if points_by_species and args.map_mode != 'none':
        overview_map = os.path.join(figures_dir, 'overview_map.png')
        try:
            draw_range_map(points_by_species, overview_map,
                           "All treated species", args.map_mode,
                           args.coastline_geojson, args.map_extent)
        except Exception as e:  # noqa: BLE001
            print(f"  [map] overview map failed: {e}", file=sys.stderr)
            overview_map = None

    # --- Figure captions ----------------------------------------------------
    captions = {}
    if args.plates_dir:
        captions = load_figure_captions(args.plates_dir)

    # --- Key ----------------------------------------------------------------
    key_text = ''
    if args.key_file and os.path.isfile(args.key_file):
        with open(args.key_file, 'r', encoding='utf-8') as f:
            key_text = f.read()

    # --- Per species assembly ----------------------------------------------
    species_data = []
    for sp, txt in species_txt.items():
        treatment = parse_treatment(txt)

        # Materials examined: prefer the localities CSV (has coordinates); fall
        # back to the parsed TYPE MATERIAL block (BioRAG descriptions) so the
        # type series still appears even without a locality file.
        recs = sort_material(loc_by_species.get(sp, []))
        if not recs and treatment.get('type_material'):
            recs = sort_material(treatment['type_material'])
        materials = [format_material_entry(r) for r in recs]

        # Plate
        plate_png = None
        plate_caption = ''
        if sp in captions and captions[sp][0] and os.path.isfile(captions[sp][0]):
            plate_png, plate_caption = captions[sp]
        elif args.plates_dir:
            cand = os.path.join(args.plates_dir, sp, f"{sp}_plate.png")
            if os.path.isfile(cand):
                plate_png = cand

        # Locality lines (plain-text list) + per-species map
        locality_lines = []
        for r in recs:
            bits = [b for b in (r.get('country'), r.get('state'),
                                r.get('locality'), _fmt_coord(r.get('lat'), r.get('lon')))
                    if b]
            if bits:
                locality_lines.append(', '.join(bits))

        range_map = None
        if sp in points_by_species and args.map_mode != 'none':
            range_map = os.path.join(figures_dir, f"{sp}_range_map.png")
            try:
                draw_range_map(points_by_species, range_map,
                               f"{args.genus + ' ' if args.genus else ''}{sp}",
                               args.map_mode, args.coastline_geojson,
                               args.map_extent, highlight=sp)
            except Exception as e:  # noqa: BLE001
                print(f"  [map] {sp} map failed: {e}", file=sys.stderr)
                range_map = None

        species_data.append({
            'species': sp,
            'treatment': treatment,
            'materials': materials,
            'plate_png': plate_png,
            'plate_caption': plate_caption,
            'range_map_png': range_map,
            'locality_lines': locality_lines,
        })

    out = build_docx(species_data, key_text, args.output, genus=args.genus,
                     title=args.title, authors=args.authors,
                     overview_map=overview_map,
                     include_traits=args.include_traits)
    print(f"\nDONE. Wrote {out}")
    print(f"      {len(species_data)} treatments, "
          f"{len(points_by_species)} with range maps.")


if __name__ == '__main__':
    main()
