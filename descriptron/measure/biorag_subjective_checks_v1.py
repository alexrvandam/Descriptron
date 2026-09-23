#!/usr/bin/env python3
"""
biorag_subjective_checks_v1.py — flag subjective (image-read) characters
========================================================================

Species treatments contain two kinds of qualitative words:

  measured   colour words computed from the data matrix (CIE L*a*b*)
  subjective colour, texture/sculpture, shape and setation words that a
             vision model read from the isolated-structure images

The human-readable treatments keep the subjective words as written. This
script finds them, checks the colour words against the measured colour of the
same structure, and records the result in machine-readable outputs, so later
Descriptron versions (e.g. with a texture-to-term lookup) can find and correct
them.

Colour check
  * The measured colour of a structure (median CIE L*a*b* of its specimens)
    is named with the KD-tree colour lookup of
    color_extraction_with_names-v8.py (nearest named colour in OpenCV LAB
    space; the table is read from that script's source, not duplicated).
  * Lightness words (dark, blackish, piceous, fuscous ... / pale, whitish,
    cream, hyaline ...) are compared with measured L*: whole-structure words
    against the structure mean, regional words (apex, base, band, third ...)
    against the darkest / palest third.
  * Hue words (yellow, ochreous, orange, red, ferruginous, green, grey ...)
    are compared with the measured hue angle and chroma.
  Status per mention: consistent | contradicts | not_verifiable
  (no colour measurement, or a pattern word such as 'mottled').

Texture / sculpture / shape / setation words
  status 'subjective_unverified' with the note that no quantitative
  texture-to-term lookup exists yet (the pipeline's GLCM/LBP/Gabor texture
  statistics are Tier 2 and are not mapped to words).

For every flag the image the observation came from is listed when the
earlier BioRAG cache records it (_foreground_mask), so a reviewer can
re-check the colour on the mask of that structure.

Outputs (in --descriptions_dir):
  <code>/<code>_subjective_flags.json   per species
  <code>/<code>_treatment.jsonld        treatment as JSON-LD incl. flags + citations
  subjective_character_flags.tsv        all flags, one row per mention
  subjective_check_summary.json

Usage:
  python biorag_subjective_checks_v1.py \
    --descriptions_dir .../Diaphorina_monograph/descriptions \
    --matrix_dir .../Diaphorina_monograph/compiled_key_tier \
    --prior_cache .../Diaphorina_29species_key/biorag_cache \
    --taxon-profile biorag_prompts/taxon_profiles/diaphorina_taxon_profile.yaml
"""

import argparse
import ast
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import biorag_feature_policy as pol  # noqa: E402
import biorag_novelty_score_v1 as nvs  # noqa: E402  (shared descriptive vocabulary)

CHECKER_VERSION = "1.0"
DEFAULT_LOOKUP_SCRIPT = Path(__file__).parent / "color_extraction_with_names-v8.py"

# ── vocabularies (generic arthropod descriptive terms) ───────────────────────
DARK_WORDS = r"black|blackish|piceous|fuscous|infuscate[d]?|dark|dusky|castaneous|brunneous"
PALE_WORDS = r"pale|whitish|white|cream|creamy|hyaline|straw|stramineous|translucent|colou?rless"
COMPARATIVE = r"\b(darker|paler|lighter|darkened|darkest|palest|lightest)\b|\bthan\b"
HUE_FAMILIES = {
    "yellow": (r"yellow|yellowish|ochreous|ochre|ochraceous|flavous|luteous|straw|stramineous|golden|tan", (60, 105)),
    "orange": (r"orange|orange-brown|fulvous|amber|testaceous", (40, 80)),
    "red": (r"red|reddish|ferruginous|rufous|rust|rusty|chestnut|castaneous|maroon", (0, 65)),
    "green": (r"green|greenish|olive", (95, 180)),
    "grey": (r"grey|gray|greyish|grayish|silvery", None),
}
COLOUR_ANY = (rf"\b(?:{DARK_WORDS}|{PALE_WORDS}|"
              + "|".join(v[0] for v in HUE_FAMILIES.values())
              + r"|brown|brownish|bicolou?red|mottled|maculate|spotted|banded|marbled|infuscation)\b")
PATTERN_WORDS = (r"\b(bicolou?red|mottled|maculate|maculations?|maculae|spotted|spots?|banded|bands?|marbled|"
                 r"blotch|blotches|patch|patches|markings?|pattern|annulate|stripes?|streaks?)\b")
REGION_WORDS = (r"\b(apex|apical|apically|base|basal|basally|distal|distally|proximal|proximally|tip|band|third|"
                r"margin|border|half|end|ends|segments?|terminal|stripe|zone|region|area|shaft|condyle|lobe|"
                r"mid-region|midline|shoulder|borders?|margins?|veins?|ventral|dorsal|face|edges?|angle|"
                r"costal|anal|flagellar|flagellomeres?)\b")
TEXTURE_WORDS = (r"\b(smooth|rough|rugose|rugulose|coriaceous|alutaceous|punctate|punctures?|pitted|granulate|"
                 r"granular|striate|striolate|reticulate|imbricate|shagreened|shining|shiny|glossy|matt|matte|"
                 r"opaque|subopaque|microsculpture|sculptured?|sculpture|costate|costulae|carinate|wrinkled|"
                 r"tuberculate|scabrous)\b")
SHAPE_WORDS = (r"\b(concave|convex|conical|cordate|reniform|lamellar|digitiform|spatulate|subglobular|globular|"
               r"triangular|elongate|arcuate|sinuate|sinuous|truncate|acuminate|subacute|rounded|oval|"
               r"cuneate|campanulate|falcate|clavate|fusiform|lanceolate|expanded|constricted|flattened)\b")
SETATION_WORDS = r"\b(setae|setose|setulose|glabrous|pubescent|pilose|hairs?|bristles?|spinules?)\b"


def load_colour_lookup(script: Path):
    """Read the `color_lookup = {...}` dict literal from the KD-tree script."""
    src = script.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "color_lookup" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"color_lookup not found in {script}")


class ColourNamer:
    def __init__(self, script: Path):
        import cv2
        from scipy.spatial import cKDTree
        self.cv2 = cv2
        lookup = load_colour_lookup(script)
        self.names = list(lookup.values())
        rgb = np.array(list(lookup.keys()), dtype=np.uint8).reshape(-1, 1, 3)
        self.lab8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
        self.tree = cKDTree(self.lab8)
        self.source = str(script)

    def name(self, L, a, b):
        q = np.array([L * 255 / 100, a + 128, b + 128])
        d, i = self.tree.query(q)
        return self.names[i], float(d)


def structure_terms(profile):
    """term -> category (longest first), from the taxon profile."""
    out = {}
    for cat, info in profile.get("structures", {}).items():
        if not isinstance(info, dict) or info.get("exclude"):
            continue
        for key in ("term", "short"):
            t = info.get(key)
            if t:
                t = re.sub(r"\s*\(.*?\)", "", t).strip().lower()
                out[t] = cat
        out[cat.replace("_", " ").lower()] = cat
    extra = {"proctiger": None, "subgenital plate": "subgenital_plate", "paramere": "paramere",
             "circumanal ring": "circumanal_ring", "metafemur": "femura", "metatibia": "tibia",
             "forewing": "whole_wing", "pterostigma": "pterostigma", "vertex": "vertex",
             "genal process": "genal_processes", "antenna": "antenna", "head": "whole_head"}
    for k, v in extra.items():
        out.setdefault(k, v)
    for t, c in list(out.items()):
        if t.startswith("forewing ") and c not in (None, "whole_wing", "forewing_keypoints"):
            out.setdefault(t[len("forewing "):], c)
    for c in list(profile.get("structures", {})):
        m = re.match(r'^cell-(.+)$', c)
        if m:
            out.setdefault(f"cell {m.group(1).lower()}", c)
    out.setdefault("anal cell", "cell-a")
    return dict(sorted(out.items(), key=lambda kv: -len(kv[0])))


def section_default(section, sex_hint=None):
    s = (section or "").lower()
    return {"head": "whole_head", "antenna": "antenna", "forewing": "whole_wing", "metaleg": None,
            "rostrum": None, "male terminalia": None, "female terminalia": None}.get(s)


def resolve_category(term_cat, clause, section):
    """Resolve sex-dependent structures (proctiger, subgenital plate)."""
    sec = (section or "").lower()
    low = clause.lower()
    sex = "male" if ("male" in low and "female" not in low) or sec.startswith("male") else \
        "female" if "female" in low or sec.startswith("female") else None
    if term_cat is None and "proctiger" in low:
        return "male_proctiger" if sex == "male" else "female_proctiger" if sex == "female" else None
    if term_cat == "subgenital_plate" and sex:
        return pol.sex_split_category("subgenital_plate", sex)
    return term_cat


def measured_colour(summary, code, cat):
    rows = summary[(summary["species"] == code) & (summary["category"] == cat)]
    if rows.empty:
        return None
    g = rows.set_index("column")
    need = ["cie_L", "cie_a", "cie_b"]
    if not all(c in g.index for c in need):
        return None
    m = {c: float(g.loc[c, "median"]) for c in need}
    for c in ("cie_L_darkest_third", "cie_L_palest_third", "cie_L_end_contrast"):
        if c in g.index:
            m[c] = float(g.loc[c, "median"])
    m["n"] = int(g.loc["cie_L", "n"])
    m["C"] = math.hypot(m["cie_a"], m["cie_b"])
    m["h"] = math.degrees(math.atan2(m["cie_b"], m["cie_a"])) % 360
    return m


def judge_colour(clause, m):
    low = clause.lower()
    if m is None:
        return "not_verifiable", "no colour measurement for this structure"
    stripped = re.sub(COMPARATIVE, " ", low)
    if not re.search(COLOUR_ANY, stripped):
        return "not_verifiable", "relative colour comparison only"
    low = stripped
    regional = bool(re.search(REGION_WORDS, low))
    pattern = bool(re.search(PATTERN_WORDS, low))
    reasons, contra = [], []
    dark = re.search(rf"\b(?:{DARK_WORDS})\b", low)
    pale = re.search(rf"\b(?:{PALE_WORDS})\b", low)
    if regional:
        Ld, Lp = m.get("cie_L_darkest_third"), m.get("cie_L_palest_third")
        if dark and Ld is not None:
            (contra if Ld > 55 else reasons).append(f"regional dark word vs darkest third L* {Ld:.1f}")
        if pale and Lp is not None:
            (contra if Lp < 40 else reasons).append(f"regional pale word vs palest third L* {Lp:.1f}")
        if (dark and Ld is None) or (pale and Lp is None):
            return "not_verifiable", "regional colour word; no regional colour measurement"
        if contra:
            return "check_image", ("regional claim not supported by the thirds of the structure (a small "
                                   "region may not shift a third's mean): " + "; ".join(contra))
        if reasons:
            return "consistent", "; ".join(reasons)
        return "not_verifiable", "regional hue or pattern word; only regional lightness is measured"
    else:
        L = m["cie_L"]
        if dark and not pale:
            (contra if L > 60 else reasons).append(f"dark word vs L* {L:.1f}")
        if pale and not dark:
            (contra if L < 35 else reasons).append(f"pale word vs L* {L:.1f}")
        for fam, (rx, hue_rng) in HUE_FAMILIES.items():
            if pattern or not re.search(rf"\b(?:{rx})\b", low):
                continue
            if fam == "grey":
                (contra if m["C"] > 25 else reasons).append(f"grey word vs chroma C* {m['C']:.1f}")
            elif m["C"] >= 10 and hue_rng:
                lo, hi = hue_rng
                ok = lo - 15 <= m["h"] <= hi + 15
                (reasons if ok else contra).append(f"{fam} word vs hue {m['h']:.0f}° (C* {m['C']:.1f})")
    if contra:
        return "contradicts", "; ".join(contra)
    if reasons:
        return "consistent", "; ".join(reasons)
    if pattern:
        return "not_verifiable", "colour-pattern word; whole-structure colour cannot confirm a pattern"
    return "not_verifiable", "colour word without a lightness or hue family that can be tested"


def clauses_of(text):
    for sent in re.split(r'(?<=[.;])(?<![A-Z]\.)(?<!sp\.)(?<!cf\.)\s+', text or ""):
        for cl in re.split(r',\s+(?=(?:with|the|and)\b)', sent):
            if cl.strip():
                yield cl.strip()


def prior_masks(prior_cache, code):
    out = {}
    d = Path(prior_cache or "") / code
    if d.is_dir():
        for fp in d.glob("*_foreground.json"):
            try:
                out[fp.name[:-len("_foreground.json")]] = json.loads(fp.read_text()).get("_foreground_mask")
            except Exception:  # noqa: BLE001
                pass
    return out


def attribute_by_L(summary, code, clause):
    """If the clause quotes 'L* a–b', return the structure of this species with that exact L* range."""
    m = re.search(r'L\*\s*(-?\d+(?:\.\d+)?)\s*[–-]\s*(-?\d+(?:\.\d+)?)', clause)
    if not m:
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    rows = summary[(summary["species"] == code) & (summary["column"] == "cie_L")]
    hits = rows[(rows["min"].round(1) == round(lo, 1)) & (rows["max"].round(1) == round(hi, 1))]
    return hits["category"].iloc[0] if len(hits) == 1 else None


class ScoredStates:
    """Per-specimen descriptive states (biorag_descriptive_scoring_v1.py), used to
    test the image-read words in the text: does the word's state actually occur in
    the specimens of this species, and how far is it from the usual state?"""

    def __init__(self, path, characters):
        self.by = {}          # (species, category, character) -> Counter(state)
        self.characters = characters
        self.word = {}        # word -> (character, canonical state, rank)
        for ch, cfg in characters.items():
            canon = nvs.canonical_states(cfg)
            for w, r in nvs.state_ranks(cfg).items():
                self.word.setdefault(w, (ch, canon[int(r)], r))
        self.ok = False
        if not path or not Path(path).exists():
            return
        df = pd.read_csv(path, sep="\t")
        df = df[df["state"].astype(str).str.lower() != "not assessable"]
        for r in df.itertuples():
            self.by.setdefault((r.species, r.category, r.character), Counter())[str(r.state)] += 1
        self.ok = bool(self.by)

    def judge(self, code, cat, words):
        """(status, reason) for the image-read words of one clause."""
        if not self.ok or not cat:
            return None
        checked = []
        for w in words:
            info = self.word.get(w)
            if not info:
                continue
            char, canon_state, rank = info
            counts = self.by.get((code, cat, char))
            if not counts:
                continue
            n = sum(counts.values())
            modal, modal_n = counts.most_common(1)[0]
            cfg = self.characters[char]
            if canon_state in counts:
                share = counts[canon_state] / n
                checked.append((("consistent" if share >= 0.5 else "check_image"), char,
                                f"'{w}' scored in {counts[canon_state]} of {n} specimens "
                                f"(usual state {modal})"))
                continue
            if cfg["type"] == "ordinal":
                ranks = nvs.state_ranks(cfg)
                steps = abs(rank - ranks.get(modal.lower(), rank))
                status = "check_image" if steps <= 1 else "contradicts"
                checked.append((status, char, f"'{w}' not scored in any of {n} specimens; usual state "
                                              f"{modal} ({steps:.0f} step(s) away)"))
            else:
                checked.append(("contradicts", char,
                                f"'{w}' not scored in any of {n} specimens (scored: "
                                f"{', '.join(sorted(counts))})"))
        if not checked:
            return None
        order = {"contradicts": 0, "check_image": 1, "consistent": 2}
        checked.sort(key=lambda t: order[t[0]])
        return checked[0][0], "; ".join(f"{c[1]}: {c[2]}" for c in checked[:3])


def check_species(code, treat, summary, profile, namer, masks, states=None):
    terms = structure_terms(profile)
    flags = []
    fields = [("diagnosis", None, treat.get("diagnosis", ""))]
    fields += [(f"description/{d.get('section', '')}", d.get("section"), d.get("text", ""))
               for d in treat.get("description", []) or []]
    fields.append(("sexual_dimorphism", None, treat.get("sexual_dimorphism", "")))
    for field, section, text in fields:
        current = section_default(section)
        for cl in clauses_of(text):
            low = cl.lower()
            hit = next((t for t in terms if re.search(rf"\b{re.escape(t)}s?\b", low)), None)
            if hit:
                current = resolve_category(terms[hit], cl, section) or current
            cat = resolve_category(current, cl, section) if current else None
            by_L = attribute_by_L(summary, code, cl)
            if by_L:
                cat = current = by_L
            base_cat = pol.split_category_name(cat)[0] if cat else None
            colour_words = sorted(set(m.group(0).lower() for m in re.finditer(COLOUR_ANY, low)))
            has_numbers = bool(pol.numbers_in(cl))
            if colour_words:
                m = measured_colour(summary, code, cat) if cat else None
                status, why = judge_colour(cl, m)
                rec = {"species": code, "field": field, "structure": cat, "kind": "colour",
                       "words": colour_words, "text": cl, "status": status, "reason": why,
                       "evidence": "measured+words" if has_numbers else "image_observation",
                       "image_checked": masks.get(base_cat)}
                if m:
                    nm, dist = namer.name(m["cie_L"], m["cie_a"], m["cie_b"])
                    rec.update({"measured_L": round(m["cie_L"], 1), "measured_a": round(m["cie_a"], 1),
                                "measured_b": round(m["cie_b"], 1), "measured_n": m["n"],
                                "kdtree_colour_name": nm, "kdtree_distance": round(dist, 1),
                                "policy_colour_word": pol.colour_name(m["cie_L"], m["cie_a"], m["cie_b"])})
                flags.append(rec)
            for kind, rx, note in (
                    ("texture", TEXTURE_WORDS, "subjective; no quantitative texture-to-term lookup yet "
                                               "(GLCM/LBP/Gabor statistics are not mapped to words) — TODO"),
                    ("shape", SHAPE_WORDS, "subjective shape term read from the image; outline shape is "
                                           "measured only as statistics (semilandmarks)"),
                    ("setation", SETATION_WORDS, "subjective; setae are not segmented or measured")):
                words = sorted(set(x.group(0).lower() for x in re.finditer(rx, low)))
                if words:
                    status, reason = "subjective_unverified", note
                    judged = states.judge(code, cat, words) if states is not None else None
                    if judged:
                        status, reason = judged[0], "per-specimen character scoring — " + judged[1]
                    flags.append({"species": code, "field": field, "structure": cat, "kind": kind,
                                  "words": words, "text": cl, "status": status,
                                  "reason": reason, "evidence": "image_observation",
                                  "image_checked": masks.get(base_cat)})
    return flags


def write_jsonld(path, code, treat, flags, profile, meta):
    doc = {
        "@context": {"@vocab": "https://schema.org/", "dsc": "https://descriptron.org/ontology/",
                     "dwc": "http://rs.tdwg.org/dwc/terms/"},
        "@type": "dsc:TaxonomicTreatment",
        "@id": f"#treatment-{code}",
        "about": {"@type": "Taxon", "identifier": code,
                  "name": pol.species_display_name(code, profile),
                  "dsc:status": pol.species_status(code, profile)},
        "dateCreated": treat.get("_meta", {}).get("generated"),
        "dsc:generator": treat.get("_meta", {}),
        "dsc:diagnosis": treat.get("diagnosis", ""),
        "dsc:description": [{"dsc:section": d.get("section"), "text": d.get("text")}
                            for d in treat.get("description", []) or []],
        "dsc:sexualDimorphism": treat.get("sexual_dimorphism", ""),
        "dsc:remarks": {"text": treat.get("remarks", ""), "dsc:evidenceTier": 2},
        "dsc:numericCitations": [{"dsc:value": c.get("value"), "dsc:featureId": c.get("id"),
                                  "dsc:evidenceTier": 1} for c in treat.get("citations", [])],
        "dsc:qualitativeObservations": [
            {"@type": "dsc:SubjectiveCharacter", "dsc:kind": f["kind"], "dsc:structure": f["structure"],
             "dsc:terms": f["words"], "text": f["text"], "dsc:section": f["field"],
             "dsc:evidenceSource": f["evidence"], "dsc:checkStatus": f["status"],
             "dsc:checkReason": f["reason"], "dsc:reviewImage": f.get("image_checked"),
             **({"dsc:measuredCIELab": [f["measured_L"], f["measured_a"], f["measured_b"]],
                 "dsc:kdtreeColourName": f["kdtree_colour_name"]} if "measured_L" in f else {})}
            for f in flags],
        "dsc:validation": treat.get("_validation", {}),
        "dsc:subjectiveCheck": meta,
    }
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Flag subjective characters in BioRAG v2 treatments")
    ap.add_argument("--descriptions_dir", required=True)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--prior_cache", default=None)
    ap.add_argument("--descriptive_matrix", default=None,
                    help="descriptive_states_by_specimen.tsv: check the image-read texture, shape and "
                         "setation words against the states scored on each specimen")
    ap.add_argument("--taxon-profile", required=True)
    ap.add_argument("--colour_lookup_script", default=str(DEFAULT_LOOKUP_SCRIPT),
                    help="Script holding the `color_lookup` dict used for the KD-tree colour names")
    args = ap.parse_args()

    profile = pol.load_taxon_profile(args.taxon_profile)
    summary = pd.read_csv(Path(args.matrix_dir) / "species_feature_summary.csv")
    namer = ColourNamer(Path(args.colour_lookup_script))
    dd = Path(args.descriptions_dir)
    meta = {"checker_version": CHECKER_VERSION, "generated": datetime.now().isoformat(),
            "colour_lookup": namer.source, "n_named_colours": len(namer.names),
            "note": "subjective words are kept in the human-readable text; flags are for review and "
                    "for future Descriptron versions"}
    characters = {k: dict(v) for k, v in nvs.DESCRIPTIVE_CHARACTERS.items()}
    characters.update(profile.get("descriptive_characters") or {})
    states = ScoredStates(args.descriptive_matrix, characters)
    if states.ok:
        print(f"per-specimen descriptive states: {len(states.by)} species x structure x character cells")
        meta["descriptive_matrix"] = str(args.descriptive_matrix)
    all_flags = []
    for tj in sorted(dd.glob("*/*_treatment.json")):
        code = tj.parent.name
        treat = json.loads(tj.read_text())
        flags = check_species(code, treat, summary, profile, namer, prior_masks(args.prior_cache, code),
                              states)
        (tj.parent / f"{code}_subjective_flags.json").write_text(json.dumps(flags, indent=1, ensure_ascii=False))
        write_jsonld(tj.parent / f"{code}_treatment.jsonld", code, treat, flags, profile, meta)
        all_flags += flags
    df = pd.DataFrame(all_flags)
    if len(df):
        df["words"] = df["words"].map(lambda w: ", ".join(w))
        df.to_csv(dd / "subjective_character_flags.tsv", sep="\t", index=False)
    summ = {**meta, "n_species": int(df["species"].nunique()) if len(df) else 0,
            "n_flags": len(df),
            "by_kind_status": {f"{k}/{s}": int(n) for (k, s), n in
                               df.groupby(["kind", "status"]).size().items()} if len(df) else {},
            "colour_contradictions": df[(df["status"] == "contradicts") & (df["kind"] == "colour")][
                ["species", "structure", "words", "reason"]].to_dict("records") if len(df) else [],
            "state_contradictions": df[(df["status"] == "contradicts") & (df["kind"] != "colour")][
                ["species", "structure", "kind", "words", "reason"]].to_dict("records") if len(df) else []}
    (dd / "subjective_check_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps({k: summ[k] for k in ("n_species", "n_flags", "by_kind_status")}, indent=1))
    print(f"colour contradictions: {len(summ['colour_contradictions'])}; "
          f"descriptive-state contradictions: {len(summ['state_contradictions'])}")


if __name__ == "__main__":
    main()
