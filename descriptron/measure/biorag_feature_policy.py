#!/usr/bin/env python3
"""
biorag_feature_policy.py — single source of truth for BioRAG evidence tiers
===========================================================================

Every column of the compiled phenomic matrix is assigned to one tier:

  key          human-measurable, allowed in KEY + DIAGNOSIS + DESCRIPTION
               (lengths/widths in mm, ratios, landmark-distance ratios,
               CIE L*a*b* colour of the whole structure)
  description  also observable, allowed in DIAGNOSIS + DESCRIPTION only
               (areas, perimeters)
  remarks      derived / statistical, allowed ONLY in REMARKS
               (PCs, UMAP, clusters, Procrustes, texture statistics, p-values,
               centroid distances, processed-image HSV colour ...)
  exclude      never used (image artefacts, duplicates, invalid encodings)
  meta         bookkeeping columns

The classification is a WHITELIST for the key tier: an unknown column falls
into 'remarks', so a new statistical feature can never leak into a key.

Taxon-agnostic. Organism-specific wording (structure terms, ratio
definitions, species display names) comes from a taxon profile YAML.

Used by: biorag_key_feature_filter_v2.py, biorag_key_builder_v1.py,
         biorag_description_refiner_v1.py, biosyslit_rag_retrieval_v2.py
"""

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

POLICY_VERSION = "1.0"

TIER_KEY = "key"
TIER_DESC = "description"
TIER_REMARKS = "remarks"
TIER_EXCLUDE = "exclude"
TIER_META = "meta"
TIER1 = (TIER_KEY, TIER_DESC)

# (regex, tier, family). First match wins. Derived columns (cie_*, lmkrel_*,
# lmkmm_*, ratio_*) are created by biorag_key_feature_filter_v2.py.
_COLUMN_RULES: List[Tuple[str, str, str]] = [
    (r'^(image_base|category|group_label|specimen_id|sex)$', TIER_META, "meta"),
    (r'^(has_|scale_|coco_|_)', TIER_META, "meta"),
    # --- measurements -------------------------------------------------------
    (r'^meas_length_mm$', TIER_KEY, "length"),
    (r'^meas_height_mm$', TIER_KEY, "length"),
    (r'^meas_aspect_ratio$', TIER_KEY, "aspect_ratio"),
    (r'^meas_length_to_height_ratio$', TIER_EXCLUDE, "duplicate"),
    (r'^meas_orientation', TIER_EXCLUDE, "image_artefact"),
    (r'^meas_area_mm2$', TIER_DESC, "area"),
    (r'^meas_perimeter_mm$', TIER_DESC, "length"),
    (r'^meas_', TIER_REMARKS, "derived_shape"),          # solidity, extent, eq. diameter
    # --- derived human-readable columns --------------------------------------
    (r'^ratio_', TIER_KEY, "ratio"),
    (r'^cie_L$', TIER_KEY, "colour"),
    (r'^cie_(a|b|C|h)$', TIER_DESC, "colour"),          # hue/chroma need a colorimeter
    (r'^cie_L_(end_contrast|darkest_third|palest_third)$', TIER_KEY, "colour"),
    (r'^lmkrel_\d+_\d+$', TIER_KEY, "landmark_ratio"),
    (r'^lmkmm_\d+_\d+$', TIER_KEY, "landmark_mm"),
    # --- raw colour encodings replaced by cie_* --------------------------------
    (r'^colhom_(L|a|b)_mean$', TIER_EXCLUDE, "raw_encoding"),
    (r'^colhom_(chroma_ab|hue_ab_sin|hue_ab_cos)$', TIER_EXCLUDE, "raw_encoding"),
    # --- everything statistical ----------------------------------------------
    (r'^lmk_dist_', TIER_REMARKS, "procrustes_distance"),
    (r'centroid_size', TIER_REMARKS, "centroid_size"),
    (r'(PC\d+|UMAP\d*|_cluster|silhouette|optimal_k)', TIER_REMARKS, "ordination_cluster"),
    (r'^colhom_', TIER_REMARKS, "colour_grid"),
    (r'^tex_', TIER_REMARKS, "texture_statistic"),
    (r'^color_', TIER_REMARKS, "processed_image_colour"),
    (r'^imd_', TIER_REMARKS, "centroid_distance"),
    (r'^shape_', TIER_REMARKS, "shape_morphometrics"),
    (r'^lmk_', TIER_REMARKS, "landmark_morphometrics"),
]
_COMPILED_RULES = [(re.compile(p), t, f) for p, t, f in _COLUMN_RULES]


def classify_column(col: str) -> Tuple[str, str]:
    """Return (tier, family) for a compiled-matrix column name."""
    for rx, tier, fam in _COMPILED_RULES:
        if rx.search(col):
            return tier, fam
    return TIER_REMARKS, "unclassified"


# Units and generic wording per derived/measured column
_MEASURE_WORDS = {
    "meas_length_mm": ("length", "mm"),
    "meas_height_mm": ("width", "mm"),
    "meas_aspect_ratio": ("length/width", "x"),
    "meas_area_mm2": ("area", "mm2"),
    "meas_perimeter_mm": ("perimeter", "mm"),
    "cie_L": ("colour (lightness)", "L*"),
    "cie_a": ("CIE a* (red-green)", "a*"),
    "cie_b": ("CIE b* (yellow-blue)", "b*"),
    "cie_C": ("chroma CIE C*ab", "C*"),
    "cie_h": ("hue angle CIE h_ab", "deg"),
    "cie_L_end_contrast": ("difference in lightness between its two end thirds", "L*"),
    "cie_L_darkest_third": ("darkest third (lightness)", "L*"),
    "cie_L_palest_third": ("palest third (lightness)", "L*"),
}

UNIT_DECIMALS = {"mm": 3, "mm2": 4, "x": 2, "L*": 1, "a*": 1, "b*": 1, "C*": 1, "deg": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Taxon profile
# ─────────────────────────────────────────────────────────────────────────────

def load_taxon_profile(path: Optional[str]) -> Dict:
    """Load a taxon profile YAML (None -> empty generic profile)."""
    prof: Dict = {}
    if path:
        import yaml
        p = Path(path).expanduser().resolve()
        with open(p) as f:
            prof = yaml.safe_load(f) or {}
        prof["_path"] = str(p)
        qf = prof.get("questions_file")
        if qf:
            qp = Path(qf)
            if not qp.is_absolute():
                qp = (p.parent / qp).resolve()
            prof["_questions_path"] = str(qp)
    prof.setdefault("taxon", {})
    prof.setdefault("species", {})
    prof.setdefault("structures", {})
    prof.setdefault("ratios", [])
    prof.setdefault("measurement_abbreviations", {})
    prof.setdefault("landmark_sets", {})
    prof.setdefault("key", {})
    prof.setdefault("description_sections", [])
    return prof


def species_display_name(code: str, profile: Dict) -> str:
    sp = profile.get("species", {}).get(code)
    if isinstance(sp, dict) and sp.get("name"):
        return sp["name"]
    genus = profile.get("taxon", {}).get("genus")
    return f"{genus} {code}" if genus else code


def species_status(code: str, profile: Dict) -> str:
    sp = profile.get("species", {}).get(code)
    return (sp or {}).get("status", "unknown") if isinstance(sp, dict) else "unknown"


def structure_info(category: str, profile: Dict) -> Dict:
    """Profile entry for a (possibly sex-split) category name."""
    base, sex = split_category_name(category)
    info = dict(profile.get("structures", {}).get(base, {}) or {})
    info.setdefault("term", base.replace("_", " "))
    if sex:
        info["sex"] = sex
        sec_key = f"section_{sex}"
        if info.get(sec_key):
            info["section"] = info[sec_key]
        info["term"] = f"{'male' if sex == 'male' else 'female'} {info.get('short', info['term'])}"
        info["short"] = info["term"]
    info.setdefault("short", info["term"])
    info.setdefault("sex", "both")
    info.setdefault("section", "Other structures")
    info.setdefault("key_priority", 2)
    return info


def split_category_name(category: str) -> Tuple[str, Optional[str]]:
    """'subgenital_plate__male' -> ('subgenital_plate', 'male')."""
    m = re.match(r'^(.*)__(male|female)$', category)
    return (m.group(1), m.group(2)) if m else (category, None)


def sex_split_category(category: str, sex: str) -> str:
    return f"{category}__{sex}"


def build_taxon_context(profile: Dict, include_glossary: bool = True) -> str:
    """Text block injected into prompts as {taxon_context}."""
    t = profile.get("taxon", {})
    lines = ["TAXON CONTEXT"]
    if t.get("genus"):
        lines.append(f"Genus: {t['genus']}   Classification: {t.get('higher_classification', t.get('family', ''))}")
    if t.get("imaging"):
        lines.append(f"Material: {t['imaging']}")
    if t.get("colour_caveat"):
        lines.append(f"Colour caveat: {t['colour_caveat']}")
    if include_glossary:
        abbr = profile.get("measurement_abbreviations", {})
        if abbr:
            lines.append("GLOSSARY OF MEASUREMENT ABBREVIATIONS (use only these meanings):")
            for k, v in abbr.items():
                lines.append(f"  {k} = {v.get('meaning', '')}")
        rat = profile.get("ratios", [])
        if rat:
            lines.append("STANDARD PROPORTIONS (ratio IDs -> meaning):")
            for r in rat:
                sx = "" if r.get("sex", "all") == "all" else f" ({r['sex']}s only)"
                lines.append(f"  {r['id']} = {r.get('words', r['num'] + ' / ' + r['den'])}{sx}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Specimen identity
# ─────────────────────────────────────────────────────────────────────────────

def specimen_number(image_base: str, profile: Dict) -> Optional[str]:
    rx = profile.get("specimen_id", {}).get("number_regex")
    if not rx:
        return None
    hits = re.findall(rx, str(image_base), flags=re.I)
    if not hits:
        return None
    h = hits[-1]
    return h[0] if isinstance(h, tuple) else h


def specimen_id(image_base: str, species: str, profile: Dict) -> str:
    num = specimen_number(image_base, profile)
    if num is None:
        return f"{species}__{image_base}"
    return f"{species}_{num}"


def specimen_sex(image_base: str, profile: Dict) -> str:
    sx = profile.get("specimen_id", {}).get("sex_regex") or {
        "male": r'_m_|_male', "female": r'_f_|_female'}
    s = str(image_base).lower()
    for sex, rx in sx.items():
        if re.search(rx, s):
            return sex
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Colour
# ─────────────────────────────────────────────────────────────────────────────

def detect_lab_encoding(L_values, a_values) -> Dict:
    """Detect OpenCV 8-bit LAB (L 0-255, a/b +128) vs true CIE LAB."""
    import numpy as np
    L = np.asarray([v for v in L_values if v == v], dtype=float)
    a = np.asarray([v for v in a_values if v == v], dtype=float)
    L_scale = 100.0 / 255.0 if (L.size and np.nanmax(L) > 100.5) else 1.0
    ab_offset = 128.0 if (a.size and np.nanmedian(a) > 64.0) else 0.0
    return {"L_scale": L_scale, "ab_offset": ab_offset,
            "source_encoding": ("OpenCV 8-bit LAB (L*x2.55, a*/b*+128)"
                                if L_scale != 1.0 or ab_offset else "CIE L*a*b*")}


def cie_from_encoded(L, a, b, enc: Dict) -> Tuple[float, float, float, float, float]:
    """Return (L*, a*, b*, C*ab, h_ab degrees)."""
    Lc = L * enc["L_scale"]
    ac = a - enc["ab_offset"]
    bc = b - enc["ab_offset"]
    C = math.hypot(ac, bc)
    h = math.degrees(math.atan2(bc, ac)) % 360.0
    return Lc, ac, bc, C, h


def lightness_term(L: float) -> str:
    """Colour-neutral term from CIE L* alone."""
    if L is None or L != L:
        return ""
    if L < 18:
        return "blackish"
    if L < 35:
        return "dark"
    if L < 60:
        return "medium-toned"
    if L < 80:
        return "pale"
    return "very pale"


def colour_name(L: float, a: float, b: float) -> str:
    """Approximate colour term from CIE L*a*b* (documented, deterministic)."""
    if any(v is None or v != v for v in (L, a, b)):
        return ""
    C = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a)) % 360.0
    if L < 18:
        return "blackish"
    if C < 8:
        if L >= 85:
            return "whitish"
        return "pale grey" if L >= 65 else ("grey" if L >= 40 else "dark grey")
    if 110 <= h < 170:
        return "pale greenish" if L >= 65 else ("greenish" if L >= 40 else "dark greenish brown")
    if 170 <= h < 330:
        return "bluish or purplish grey"
    if 70 <= h < 110:                       # yellow sector
        if L >= 80:
            return "pale yellow" if C < 40 else "yellow"
        if L >= 60:
            return "ochreous yellow" if C >= 30 else "pale ochreous"
        if L >= 40:
            return "ochreous brown"
        return "dark brown" if L < 28 else "brown"
    if 40 <= h < 70:                        # orange-yellow / brown sector
        if L >= 80:
            return "pale ochreous"
        if L >= 60:
            return "ochreous" if C >= 25 else "light brown"
        if L >= 40:
            return "light brown" if C < 25 else "orange-brown"
        return "dark brown" if L < 28 else "brown"
    # red sector (h < 40 or h >= 330)
    if L >= 70:
        return "pinkish"
    if L >= 45:
        return "reddish brown"
    return "dark reddish brown"


# ─────────────────────────────────────────────────────────────────────────────
# Feature labels, IDs, number formatting
# ─────────────────────────────────────────────────────────────────────────────

def feature_id(category: str, column: str) -> str:
    """Stable, human-readable data-matrix ID, e.g. 'tibia.length_mm'."""
    short = re.sub(r'^meas_', '', column)
    return f"{category}.{short}"


def feature_unit(column: str) -> str:
    if column in _MEASURE_WORDS:
        return _MEASURE_WORDS[column][1]
    if column.startswith(("ratio_", "lmkrel_")):
        return "x"
    if column.startswith("lmkmm_"):
        return "mm"
    return ""


def feature_label(category: str, column: str, profile: Dict,
                  landmark_ref: Optional[Dict] = None) -> Tuple[str, str]:
    """Return (label, definition) in words for a feature."""
    info = structure_info(category, profile)
    term = info.get("short") or info.get("term")
    if column.startswith("ratio_"):
        rid = column[len("ratio_"):]
        for r in profile.get("ratios", []):
            if r.get("id") == rid:
                return f"{rid} ratio", r.get("words", rid)
        return f"{rid} ratio", rid
    if column.startswith(("lmkrel_", "lmkmm_")):
        _, i, j = column.split("_")
        ls = profile.get("landmark_sets", {}).get(split_category_name(category)[0], {})
        lname = ls.get("landmark_name", "landmark")
        lterm = ls.get("term", term)
        if column.startswith("lmkmm_"):
            return (f"{lterm}: distance {lname}s {i}-{j} (mm)",
                    f"straight-line distance between {lname} {i} and {lname} {j}")
        ref = (landmark_ref or {}).get(split_category_name(category)[0])
        refs = f"{ref[0]}-{ref[1]}" if ref else "reference span"
        return (f"{lterm}: distance {lname}s {i}-{j} / span {refs}",
                f"distance between {lname} {i} and {j} divided by the distance between "
                f"{lname} {refs} (the longest landmark span)")
    words, _unit = _MEASURE_WORDS.get(column, (column, ""))
    return f"{term} {words}", f"{words} of the {info.get('term')}"


def decimals_for(unit: str, value: Optional[float] = None) -> int:
    d = UNIT_DECIMALS.get(unit, 3)
    if unit in ("mm", "mm2") and value is not None and value == value and value != 0:
        # keep 3 significant figures for small lengths
        mag = int(math.floor(math.log10(abs(value))))
        d = max(d, 2 - mag)
    return min(d, 6)


def fmt(value: Optional[float], unit: str, ref: Optional[float] = None) -> str:
    """Format one value with the precision rule used everywhere (sheets, key)."""
    if value is None or value != value:
        return "n/a"
    d = decimals_for(unit, ref if ref is not None else value)
    if unit == "deg":
        return f"{value:.0f}"
    return f"{value:.{d}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Text checks (Tier-2 terms that must not appear in key/diagnosis/description)
# ─────────────────────────────────────────────────────────────────────────────

TIER2_TEXT_PATTERNS = [
    r'\bPC\s?\d', r'principal component', r'\bUMAP', r'eigen', r'ordination',
    r'\bcluster\s*(\d|assignment|membership|analysis)', r'homology cluster',
    r'Procrustes', r'centroid size', r'\bp\s*[<=>≤≥]\s*0?\.\d', r'p-value',
    r'Kruskal', r'\bDunn', r'MANOVA', r'\bCVA\b', r'canonical variate',
    r'\bGLCM\b', r'\bLBP\b', r'Gabor', r'Fourier', r'spectral', r'entropy',
    r'silhouette', r'eta[²2]', r'η²', r'shape space', r'n_boundaries',
    r'0\s*[–-]\s*255', r'\bcolhom_', r'\btex_', r'\blmk_', r'\bimd_', r'_phylo',
    r'statistically significant', r'FDR',
]
_TIER2_RX = [re.compile(p, re.I) for p in TIER2_TEXT_PATTERNS]


def find_tier2_terms(text: str) -> List[str]:
    hits = []
    for rx in _TIER2_RX:
        m = rx.search(text or "")
        if m:
            hits.append(m.group(0))
    return hits


def numbers_in(text: str) -> List[str]:
    """All decimal numbers in text (ignores numbers glued to letters like 'sp1')."""
    out = []
    for m in re.finditer(r'(?<![A-Za-z0-9_.])[-−]?\d+(?:\.\d+)?(?![A-Za-wyz0-9_])', text or ""):
        out.append(m.group(0).replace("−", "-"))
    return out


def number_is_allowed(num: str, allowed: List[float], tol_digits: bool = True) -> bool:
    """True if num equals an allowed value at num's own printed precision."""
    try:
        v = float(num)
    except ValueError:
        return True
    d = len(num.split(".")[1]) if "." in num else 0
    for a in allowed:
        if a is None or a != a:
            continue
        if round(a, d) == round(v, d) or abs(a - v) <= 0.5 * 10 ** (-d) + 1e-12:
            return True
    return False


_TYPE_MATERIAL: dict = {}


def load_type_material(anchor) -> int:
    """Find the type designations written by biorag_type_material_v1.py, starting from any path
    inside the monograph (the descriptions directory, say) and walking up. Returns how many
    species have a statement; 0 means the exporters keep their placeholder, so a monograph can
    never silently claim a holotype that has not been designated."""
    import json
    from pathlib import Path
    _TYPE_MATERIAL.clear()
    p = Path(anchor).resolve()
    for base in [p] + list(p.parents)[:4]:
        for cand in (base / "localities" / "type_material.json", base / "type_material.json"):
            if cand.exists():
                try:
                    per = json.loads(cand.read_text()).get("per_species", {})
                except Exception:  # noqa: BLE001
                    return 0
                _TYPE_MATERIAL.update({k: v.get("statement", "") for k, v in per.items()
                                       if v.get("statement")})
                return len(_TYPE_MATERIAL)
    return 0


def type_material(code: str) -> str:
    """The Code-compliant type statement for a species, or "" if none has been supplied."""
    return _TYPE_MATERIAL.get(code, "")
