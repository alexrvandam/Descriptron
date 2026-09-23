#!/usr/bin/env python3
"""
biorag_confabulation_checker_v2.py — independent audit of BioRAG v2 treatments
==============================================================================

v2 of biorag_confabulation_checker.py (v1 kept unchanged; it audits the v1
``*_foreground.jsonld`` trait format). This checker audits the evidence-tiered
treatments written by biorag_description_refiner_v1.py
(``<code>/<code>_treatment.json``) and is deliberately INDEPENDENT of the
refiner's own validator:

  * The refiner only checks that every number occurs *somewhere* in the
    species data sheet. This checker recomputes each species' n / min / max /
    mean for every feature from the per-specimen matrix
    (specimen_matrix_long.csv) and attributes every number to ONE feature,
    using the words around it (structure + measurement) and the model's own
    citation list. A correct value attached to the wrong structure is therefore
    an error here (column confusion), not a pass.
  * Comparative statements ("greater than in all other species measured (22)
    except ...") are re-derived from the specimen values.
  * Remarks (Tier-2 statistics, natural history) are checked against the
    statements the model was given (the data sheet) and the localities table.

Claim kinds and error types (v1 names kept where the failure is the same):

  measurement   value_rounding, value_fabrication, column_confusion,
                unit_confusion, sample_inflation, sample_deflation,
                cross_species_value, group_value_as_species
  comparison    fabricated_comparison
  wording       statistic_outside_remarks (v1: cluster_as_continuous),
                unknown_taxon
  remarks       remarks_misattribution, remarks_fabrication,
                unsupported_natural_history

Rate = claims with >= 1 error / claims checked (a claim is one reported
measurement with its mean and n, one comparative statement, or one Remarks
statement). The per-number totals are reported as well.

Outputs (--output_dir): confabulation_summary.json (v1-compatible keys),
confabulation_claims.tsv (every claim), confabulation_issues.tsv,
per_species/<code>.json, confabulation_report.txt

Usage:
  python biorag_confabulation_checker_v2.py \
     --descriptions_dir ".../Diaphorina_monograph/descriptions" \
     --matrix_dir ".../Diaphorina_monograph/compiled_key_tier" \
     --taxon_profile biorag_prompts/taxon_profiles/diaphorina_taxon_profile.yaml \
     --key_tree ".../Diaphorina_monograph/key/key_tree.json" \
     --localities ".../localities/Diaphorina_localities_verified.tsv" \
     --output_dir ".../Diaphorina_monograph/descriptions/confabulation_report_v2"
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import biorag_feature_policy as pol  # noqa: E402

CHECKER_VERSION = "2.0"

ERROR_TYPES_V1_EQUIVALENT = {
    "value_rounding": "value_rounding",
    "value_fabrication": "value_fabrication",
    "column_confusion": "column_confusion",
    "unit_confusion": "unit_confusion",
    "sample_inflation": "sample_inflation",
    "fabricated_comparison": "fabricated_pairwise",
    "statistic_outside_remarks": "cluster_as_continuous",
}

NUM = r'[-−]?\d+(?:\.\d+)?'
FEATURE_TAG_RX = re.compile(r'\[[A-Za-z_+\-0-9]+\.[\w/()+\-]+\]')
DASH = r'\s*(?:–|—|-|to)\s*'
UP_WORDS = r'greater|larger|longer|wider|broader|higher|paler|stronger|more\s+\w+'
DOWN_WORDS = r'smaller|shorter|narrower|lower|darker|weaker|less\s+\w+'
MEASURE_MAP = {"length": "length_mm", "width": "height_mm", "area": "area_mm2",
               "perimeter": "perimeter_mm", "aspect": "aspect_ratio"}
COLOUR_MAP = {"L*": "cie_L", "a*": "cie_a", "b*": "cie_b", "C*": "cie_C", "h": "cie_h"}
ABBREV_BEFORE_DOT = re.compile(r'(?:\b[A-Z]|\bsp|\bspp|\bcf|\baff|\be\.g|\bi\.e|\bca|\bapprox|\bvs|\bFig|\bfig|'
                               r'\bno|\bvar|\bssp|\bsubsp|\bet al)$')
SENTENCE_START_WORDS = {"In", "The", "No", "A", "An", "Its", "This", "These", "Both", "Label", "Specimens",
                        "Sexes", "Sexual", "Head", "Forewing", "Male", "Female", "Among", "Within", "For",
                        "It", "On", "All", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Material",
                        "Standard", "Data", "Only", "As", "Of", "However", "Although", "Colour", "Cell",
                        "Beyond", "Aside", "Individual", "Relative", "Remarks", "Further", "See", "Here",
                        "Kruskal", "Wallis", "Dunn", "Benjamini", "Hochberg", "Procrustes", "Diaphorina",
                        "Psylloidea", "Hemiptera", "Liviidae", "Rostrum", "Metaleg", "Antenna", "Proportions",
                        "Description", "Diagnosis", "Key", "Figure", "CIE", "LAB", "OCR", "Florence"}


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Sentence boundaries that survive taxonomic abbreviations (D., sp., cf.)."""
    spans, start = [], 0
    for m in re.finditer(r'[.!?](?=\s+[A-Z(])', text):
        before = text[start:m.start()]
        if ABBREV_BEFORE_DOT.search(before):
            continue
        spans.append((start, m.end()))
        start = m.end()
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def matches(printed: str, actual: Optional[float]) -> bool:
    """printed number equals actual at the printed precision (half-unit tolerance)."""
    if actual is None or actual != actual:
        return False
    v = float(printed.replace("−", "-"))
    d = decimals(printed)
    return abs(actual - v) <= 0.5 * 10 ** (-d) + 1e-9 + 1e-9 * abs(actual)


def rel_err(printed: str, actual: float) -> float:
    v = float(printed.replace("−", "-"))
    return abs(v - actual) / max(abs(actual), 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Evidence
# ─────────────────────────────────────────────────────────────────────────────

class Evidence:
    def __init__(self, args):
        self.profile = pol.load_taxon_profile(args.taxon_profile)
        md = Path(args.matrix_dir)
        self.long = pd.read_csv(md / "specimen_matrix_long.csv")
        self.fdict = pd.read_csv(md / "feature_dictionary.tsv", sep="\t").set_index("feature_id")
        g = self.long.groupby(["species", "feature_id"])["value"]
        self.stats = pd.DataFrame({"n": g.count(), "min": g.min(), "max": g.max(), "mean": g.mean(),
                                   "median": g.median()}).reset_index()
        self.by_sp = {sp: d.set_index("feature_id") for sp, d in self.stats.groupby("species")}
        self.genus = self.stats.groupby("feature_id").agg(gmin=("min", "min"), gmax=("max", "max"),
                                                          nsp=("species", "nunique"))
        self.species = sorted(self.stats["species"].unique(), key=pol_natural_key)
        # consistency with the summary the data sheets were built from
        self.summary_mismatch = 0
        sp_sum = md / "species_feature_summary.csv"
        if sp_sum.exists():
            s = pd.read_csv(sp_sum).merge(self.stats, on=["species", "feature_id"], suffixes=("_s", ""))
            for c in ("n", "min", "max"):
                self.summary_mismatch += int((abs(s[f"{c}_s"] - s[c]) > 1e-9).sum())
        self.key = json.loads(Path(args.key_tree).read_text()) if args.key_tree and Path(args.key_tree).exists() else None
        self.loc = None
        if args.localities and Path(args.localities).exists():
            self.loc = pd.read_csv(args.localities, sep="\t" if args.localities.endswith(".tsv") else ",",
                                   dtype=str).fillna("")
        # category aliases (same structure under two names), e.g. cell-csc == cell-c+sc
        self.alias = {}
        for a, b in (self.profile.get("category_aliases") or {}).items():
            self.alias[a] = b
        cats = sorted(self.fdict["category"].unique())
        norm = defaultdict(list)
        for c in cats:
            norm[re.sub(r'[^a-z0-9]', '', c.lower())].append(c)
        for group in norm.values():
            if len(group) > 1:
                canon = max(group, key=len)
                for c in group:
                    if c != canon:
                        self.alias.setdefault(c, canon)
        self._build_lexicon()
        lm = []
        for col in self.fdict["column"]:
            mm = re.match(r'^lmk(?:rel|mm)_(\d+)_(\d+)$', str(col))
            if mm:
                lm += [int(mm.group(1)), int(mm.group(2))]
        self.max_landmark = max(lm) if lm else 0
        self.genus_can = {}
        for fid, r in self.genus.iterrows():
            self.genus_can.setdefault(self.canon_fid(fid), r)
        self._rows_cache = {}

    def row(self, sp, fid):
        d = self.by_sp.get(sp)
        if d is None:
            return None
        for f in d.index:
            if self.canon_fid(f) == fid:
                return d.loc[f]
        return None

    def rows_n2(self, fid):
        if fid not in self._rows_cache:
            out = {}
            for sp in self.by_sp:
                r = self.row(sp, fid)
                if r is not None and r["n"] >= 2:
                    out[sp] = r
            self._rows_cache[fid] = out
        return self._rows_cache[fid]

    def landmark_sets_for(self, col, sentence, section):
        sets = [c for c in (self.profile.get("landmark_sets") or {})]
        have = [s_ for s_ in sets if f"{s_}.{col}" in self.fdict.index]
        if len(have) == 1:
            return have[0]
        for s_ in have or sets:
            info = self.profile["landmark_sets"][s_]
            if re.search(re.escape(info.get("landmark_name", "~")), sentence, re.I) or \
                    info.get("term", "~").lower() == section.lower():
                return s_
        return (have or sets or [None])[0]

    def canon_cat(self, cat: str) -> str:
        return self.alias.get(cat, cat)

    def canon_fid(self, fid: str) -> str:
        cat, _, col = fid.partition(".")
        return f"{self.canon_cat(cat)}.{col}"

    # names ------------------------------------------------------------------
    def display(self, code):
        return pol.species_display_name(code, self.profile)

    def _build_lexicon(self):
        genus = self.profile.get("taxon", {}).get("genus", "")
        forms = {}
        for code in self.profile.get("species", {}):
            name = self.display(code)
            forms[name] = code
            if genus and name.startswith(genus + " "):
                rest = name[len(genus) + 1:]
                forms[f"{genus[0]}. {rest}"] = code
                if rest.startswith(("sp. ", "cf. ")) or " " not in rest:
                    pass
        self.name_forms = forms
        alts = sorted(forms, key=len, reverse=True)
        self.name_rx = re.compile("(" + "|".join(re.escape(a) for a in alts) + r")(?![\w'])")
        # structures: term, short, plus common variants
        lex = []
        lsets = set(self.profile.get("landmark_sets") or {})
        for cat, info in (self.profile.get("structures") or {}).items():
            if info.get("exclude") or cat in lsets:
                continue
            terms = {info.get("term", ""), info.get("short", ""), *(info.get("synonyms") or [])}
            for t in list(terms):
                t2 = re.sub(r'\s*\(.*?\)', '', t).strip()
                terms.add(t2)
                mcell = re.match(r'(?:\w+ )?cell (\S+)$', t2)
                if mcell:
                    terms.add(t2.split(" ", 1)[1] if t2.count(" ") >= 2 else t2)
                    if len(mcell.group(1)) >= 2:
                        terms.add(mcell.group(1))          # bare cell code: "m1", "c+sc"
                if t2.endswith("es"):
                    terms.add(t2[:-2])
                if t2.endswith("s") and not t2.endswith("ss"):
                    terms.add(t2[:-1])
            for t in terms:
                if t and "landmark" not in t:
                    lex.append((t.lower(), cat))
        extra = {"cell a": "cell-a", "anal cell": "cell-a", "cell c+sc": "cell-c+sc", "cell csc": "cell-csc",
                 "cell rs": "cell-Rs", "fore wing": "whole_wing", "wing": "whole_wing",
                 "basal aedeagal segment": "proximal_aedeagus", "proximal segment of the aedeagus": "proximal_aedeagus",
                 "distal segment of the aedeagus": "distal_aedeagus", "aedeagal head": "distal_aedeagus",
                 "hind femur": "femura", "hind tibia": "tibia", "metafemora": "femura", "metatibiae": "tibia",
                 "parameres": "paramere", "genal cones": "genal_processes", "genal process": "genal_processes",
                 "antennae": "antenna", "labium": None, "rostrum": None, "terminalia": None, "legs": None,
                 "metaleg": None, "aedeagus": None, "proctiger": "proctiger?", "subgenital plate": "subgenital_plate"}
        for t, c in extra.items():
            lex.append((t, c))
        lex = sorted(set(lex), key=lambda x: -len(x[0]))
        self.struct_rx = re.compile(r'\b(' + "|".join(re.escape(t).replace(r'\ ', r'[\s-]+') for t, _ in lex)
                                    + r')(?:e?s)?(?![\w+])', re.I)
        self.cat_section = {}
        for cat, info in (self.profile.get("structures") or {}).items():
            for key, suffix in (("section", ""), ("section_male", "__male"), ("section_female", "__female")):
                if info.get(key):
                    self.cat_section[cat + suffix] = info[key]
        self.struct_alt = "|".join(re.escape(t).replace(r'\ ', r'[\s-]+') for t, _ in lex)
        self.struct_map = {}
        for t, c in lex:
            self.struct_map.setdefault(t, c)
        self.ratio_ids = [r["id"] for r in (self.profile.get("ratios") or [])]
        self.ratio_rx = re.compile("(" + "|".join(re.escape(r) for r in sorted(self.ratio_ids, key=len, reverse=True))
                                   + r")(?![\w])") if self.ratio_ids else None
        self.ratio_pairs = {}
        self.col_equiv = {}
        abbr = self.profile.get("measurement_abbreviations") or {}
        for r in self.profile.get("ratios") or []:
            num, den = abbr.get(str(r.get("num")), {}), abbr.get(str(r.get("den")), {})
            if num and den:
                self.ratio_pairs[(num["category"], den["category"])] = r["id"]
                if num["category"] == den["category"] and {num.get("measurement"), den.get("measurement")} == \
                        {"length_mm", "height_mm"}:
                    self.col_equiv[f"ratio_{r['id']}"] = (num["category"], "aspect_ratio")


def pol_natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


# ─────────────────────────────────────────────────────────────────────────────
# Data-sheet blocks (what the model was given)
# ─────────────────────────────────────────────────────────────────────────────

def sheet_blocks(sheet: str) -> Dict[str, List[str]]:
    blocks, cur = defaultdict(list), None
    for line in sheet.splitlines():
        m = re.match(r'^=== (.+?) ===', line)
        if m:
            cur = m.group(1)
            continue
        if cur and line.strip():
            blocks[cur].append(line.strip().lstrip("- ").strip())
    return blocks


def block(blocks, prefix):
    for k, v in blocks.items():
        if k.startswith(prefix):
            return v
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Claim extraction
# ─────────────────────────────────────────────────────────────────────────────

class Claim(dict):
    pass


UNIT_RX = r'(?:mm²|mm2|mm|µm|um|×|x(?![a-z])|times\b|°|%)'
SKIP_BEFORE_STRUCT = re.compile(r'(?:relative to|than|as long as|as wide as|as broad as|exceeding|compared with|'
                                r'with that of|to that of|of the length of|half the|twice the|times the)\s+(?:the\s+)?$',
                                re.I)


def landmark_context(sentence: str) -> bool:
    return bool(re.search(r'junction|landmark|\bspan\b|\bdistance', sentence, re.I))


def mask_spans(text: str, ev: Evidence) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, int]]]:
    """Spans whose digits are identifiers, not values; plus landmark pairs (pos, a, b)."""
    spans = []
    for m in ev.name_rx.finditer(text):
        spans.append((m.start(), m.end(), "name"))
    lmk = []
    INT = r'(\d{1,2})(?![.\d])'
    for m in re.finditer(r'(?:vein[- ])?junctions?\s+' + INT + r'\s+(?:and|to)\s+(?:vein[- ])?junctions?\s+' + INT,
                         text, re.I):
        spans.append((m.start(), m.end(), "landmark"))
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        if not re.search(r'(?:span|relative to|divided by)\s*\(?\s*$', text[max(0, m.start() - 20):m.start()], re.I):
            lmk.append((m.start(), a, b))
    for m in re.finditer(r'\b(?:vein[- ])?junctions?\s+\d{1,2}(?![.\d])', text, re.I):
        spans.append((m.start(), m.end(), "landmark"))
    for m in re.finditer(r'(?:vein[- ]junctions?|junctions?|landmarks?|distance|between)\s+' + INT +
                         r'\s*(?:[-–]|and|to)\s*' + INT, text, re.I):
        spans.append((m.start(), m.end(), "landmark"))
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        if not re.search(r'(?:span|relative to|divided by)\s*\(?\s*$', text[max(0, m.start() - 20):m.start()], re.I):
            lmk.append((m.start(), a, b))
    # bare "a–b" pairs inside sentences about landmarks: "2–16, 0.66–0.68×", "4-8 measures"
    for sa, sb in sentence_spans(text):
        sent = text[sa:sb]
        if not landmark_context(sent):
            continue
        for m in re.finditer(r'(?<![\w.=])' + INT + r'\s*[-–]\s*' + INT + r'(?=\s*(?:,|\(|:|/|is\b|measures\b|$|and\b|;|\)))',
                             sent):
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            if 1 <= a < b <= ev.max_landmark and not re.search(r'n\s*=\s*$', sent[:m.start()]):
                spans.append((sa + m.start(), sa + m.end(), "landmark"))
                if not re.search(r'(?:span|relative to|divided by)\s*(?:\(\s*)?(?:(?:vein[- ])?junctions?\s*)?$',
                                 sent[max(0, m.start() - 30):m.start()], re.I):
                    lmk.append((sa + m.start(), a, b))
    patterns = [
        (r'\bspan\s+\d+\s*[-–]\s*\d+(?![.\d])', "landmark"),
        (r'\bcouplets?\s+\d+(?![.\d])[ab]?(?:\s*(?:,|and|–|-)\s*\d+(?![.\d])[ab]?)*', "couplet"),
        (r'\bsegments?\s+\d{1,2}(?![.\d])(?:\s*(?:,|and|to|or|–|-)\s*\d{1,2}(?![.\d]))*', "segment"),
        (r'\b(?:Figures?|Figs?\.)\s*\d+[a-z]?(?:\s*[–-]\s*\d*[a-z]?)?', "figure"),
        (r'\b\d{4}-\d{2}-\d{2}\b', "date"), (r'\b\d{1,2}\.[ivx]+\.\d{4}\b', "date"),
        (r'\b\d{1,2}\.\d{1,2}\.\d{4}\b', "date"), (r'(?<![.\d])\b(?:18|19|20)\d{2}\b(?![.\d])', "year"),
        (r'\b(?:LAB|lab)\d\b', "code"), (r'\bcu\d|\bm\d\b|\br\d\b', "code"),
        (r'\b[A-Z][A-Za-z]{1,6}-\d+[A-Z]?(?:\s*[–-]\s*\d+)?(?:-\d+)?\b', "code"),
        (r'\((?:first|1st) two principal components[^)]*\)', "tier2"),
        (r'\b\d+\s+species\s+compared', "count"), (r'\(\d+\s*spp?\.\)', "count"),
        (r'\bmeasured\s*\(\d+\)', "count"),
        (r'\b\d+\s+(?:males?|females?|specimens?|individuals?|slides?|images?|species|vein[- ]junction landmarks|'
         r'landmarks|semilandmarks|principal components|cells?|segments)\b', "count"),
    ]
    if ev.ratio_rx:
        for m in ev.ratio_rx.finditer(text):
            spans.append((m.start(), m.end(), "ratio_id"))
    for p, kind in patterns:
        for m in re.finditer(p, text, re.I if kind in ("couplet", "segment") else 0):
            spans.append((m.start(), m.end(), kind))
    for m in re.finditer(r'\bsegments?\s*\((?:e\.g\.\s*)?[\d,\sorand–-]+\)|\(e\.g\.\s*[\d,\sorand–-]+\)', text):
        spans.append((m.start(), m.end(), "segment"))
    for m in re.finditer(r'\[[A-Za-z_+\-0-9]+\.[\w/()+\-]+\]', text):
        spans.append((m.start(), m.end(), "feature_id"))
    lmk.sort()
    return spans, lmk


def in_spans(pos, spans):
    for a, b, k in spans:
        if a <= pos < b:
            return k
    return None


def structure_mentions(text: str, ev: Evidence, section: str, skip_relative: bool = True) -> List[Tuple[int, str]]:
    out = []
    sl = section.lower()
    sec_sex = "female" if sl.startswith("female") else "male" if sl.startswith("male") else None
    for m in ev.struct_rx.finditer(text):
        t = re.sub(r'[\s-]+', ' ', m.group(1).lower())
        if skip_relative and SKIP_BEFORE_STRUCT.search(text[max(0, m.start() - 30):m.start()]):
            continue
        if t in ("wing", "forewing", "fore wing", "membrane") and re.match(r'[\s-]*cells?\b', text[m.end():], re.I):
            continue                                  # "forewing cell X": the cell is the structure
        cat = ev.struct_map.get(t)
        if cat is None:
            out.append((m.start(), None))
            continue
        pre = text[max(0, m.start() - 12):m.start()].lower()
        sex = "male" if re.search(r'(?<!fe)male\s*$|♂\s*$', pre) else \
            "female" if re.search(r'female\s*$|♀\s*$', pre) else sec_sex
        if cat == "proctiger?":
            cat = {"male": "male_proctiger", "female": "female_proctiger"}.get(sex)
        elif cat == "subgenital_plate":
            cat = {"male": "subgenital_plate__male", "female": "subgenital_plate__female"}.get(sex)
        out.append((m.start(), cat))
    return out


def ratio_for(ev: Evidence, num_cat: Optional[str], den_text: str, section: str) -> Optional[str]:
    if not num_cat or not den_text:
        return None
    den = [c for _, c in structure_mentions(den_text, ev, section, skip_relative=False) if c]
    if not den:
        return None
    base = lambda c: re.sub(r'__(?:male|female)$', '', ev.canon_cat(c))  # noqa: E731
    return ev.ratio_pairs.get((base(num_cat), base(den[0])))


def extract_claims(text: str, ev: Evidence, section: str) -> Tuple[List[Claim], List[Tuple], List[Tuple]]:
    spans, lmk = mask_spans(text, ev)
    tokens = []
    for m in re.finditer(r'(?<![\w.])' + NUM + r'(?=$|[^\w]|x(?![a-z])|mm|µm|um\b)', text):
        if in_spans(m.start(), spans):
            continue
        tokens.append(m)
    groups, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if i + 1 < len(tokens):
            gap = text[t.end():tokens[i + 1].start()]
            pre = text[max(0, t.start() - 9):t.start()]
            if re.fullmatch(r'\s*' + UNIT_RX + r'?' + DASH, gap) or \
                    (gap == "" and tokens[i + 1].group(0).startswith("-")) or \
                    (re.fullmatch(r'\s*and\s*', gap) and re.search(r'between\s*$', pre)):
                groups.append((t, tokens[i + 1]))
                i += 2
                continue
        groups.append((t, None))
        i += 1
    structs = structure_mentions(text, ev, section)
    sents = sentence_spans(text)
    claims: List[Claim] = []
    pending_mean = None
    two_means = False
    for lo, hi in groups:
        start, end = lo.start(), (hi or lo).end()
        pre = text[max(0, start - 30):start]
        post = text[end:end + 100]
        lo_s = lo.group(0).replace("−", "-")
        hi_s = hi.group(0).replace("−", "-") if hi else None
        if hi is not None and text[lo.end():hi.start()] == "":
            hi_s = hi_s.lstrip("-")
        last = claims[-1] if claims else None
        gap_last = text[last["end"]:start] if last else ""
        # ---- n
        if re.search(r'\bn\s*=\s*$', pre):
            direct = last is not None and re.fullmatch(
                r'\s*' + UNIT_RX + r'?[^()\d]{0,25}[(,;]?\s*(?:[^()\d]{0,20}' + NUM +
                r'\s*' + UNIT_RX + r'?\s*[;,]\s*)?n\s*=\s*', gap_last) and \
                not re.search(NUM + r'\s*' + UNIT_RX + r'?\s*' + DASH + NUM, gap_last)
            if direct:
                last["n_claim"] = lo_s
                if two_means and len(claims) >= 2:
                    claims[-2]["n_claim"] = lo_s
            two_means = False
            continue
        # ---- mean
        if re.search(r'\bmean\s*[=:]?\s*$', pre):
            if re.match(r'\s*' + UNIT_RX + r'?\s*\(\s*' + NUM, post):
                pending_mean = lo_s                     # "mean 0.35x (0.34–0.37x; n=3)"
            elif last is not None:
                am = re.match(r'\s*' + UNIT_RX + r'?\s*(?:and|,)\s*(' + NUM + r')(?!\s*' + DASH + ')', post)
                if am and len(claims) >= 2:             # "(mean 0.212 and 0.210 mm; ...)"
                    claims[-2]["mean_claim"] = lo_s
                    last["mean_claim"] = am.group(1)
                    last["_skip_next"] = am.group(1)
                    two_means = True
                else:
                    last["mean_claim"] = lo_s
            continue
        if last is not None and last.get("_skip_next") == lo_s and re.search(r'(?:and|,)\s*$', pre):
            last.pop("_skip_next", None)
            continue
        if re.search(r'(?:±|\bSD\s*[=:]?)\s*$', pre) and last is not None:
            last["sd_claim"] = lo_s
            continue
        sent = next(((a, b) for a, b in sents if a <= start < b), (0, len(text)))
        prev_end = last["end"] if last and last["end"] > sent[0] else sent[0]
        window = text[prev_end:start]
        semi = max(text.rfind(";", sent[0], start), text.rfind(":", sent[0], start), sent[0])
        clause = text[semi:start]
        c = Claim(kind="measurement", section=section, start=start, end=end, lo=lo_s, hi=hi_s,
                  sentence=text[sent[0]:sent[1]].strip(),
                  context=text[max(sent[0], start - 90):min(sent[1], end + 40)],
                  _post=re.split(NUM, text[end:end + 70])[0])
        if pending_mean is not None:
            c["mean_claim"] = pending_mean
            pending_mean = None
        c["approx"] = bool(re.search(r'(?:about|approximately|approx\.|ca\.|c\.|nearly|almost|roughly|some)\s*$',
                                     pre, re.I))
        unit = None
        m = re.match(r'\s*(' + UNIT_RX + r')', post)
        if m:
            unit = {"mm²": "mm2", "mm2": "mm2", "mm": "mm", "µm": "um", "um": "um", "×": "x", "x": "x",
                    "times": "x", "°": "deg", "%": "%"}.get(m.group(1), m.group(1))
        after = post[m.end():] if m else post
        cm = re.search(r'(Δ\s*)?(L\*|a\*|b\*|C\*)\s*(?:values?\s*)?(?:of\s*)?[:=]?\s*$', pre)
        col = None
        rel_target = None
        rid_after = None
        if cm:
            ch = cm.group(2)
            kw = None
            if ch == "L*":
                KW = (r'(?P<contrast>(?:lightness\s+)?(?:contrast|difference)[^;()]{0,45}(?:thirds|ends)|'
                      r'end[- ]thirds? contrast|contrast in lightness|lightness contrast|end contrast|ΔL|'
                      r'\bcontrast\s*(?:L\*)?\s*$)|'
                      r'(?P<dark>darkest[^;()]{0,20}third|darkest (?:end|part|region)|darker (?:basally|apically|'
                      r'distally|proximally)|darker (?:basal|apical|distal|proximal) third|darkest\s*\(?\s*(?:L\*)?\s*$)|'
                      r'(?P<pale>palest[^;()]{0,20}third|palest (?:end|part|region)|paler (?:apically|distally|'
                      r'basally|proximally)|paler (?:basal|apical|distal|proximal) third|palest\s*\(?\s*(?:L\*)?\s*$)|'
                      r'(?P<whole>whole)')
                window_k = FEATURE_TAG_RX.sub(" ", window)
                clause_k = FEATURE_TAG_RX.sub(" ", clause)
                src = window_k if re.search(KW, window_k, re.I) else clause_k[-60:]
                hits = list(re.finditer(KW, src, re.I))
                if hits:
                    h = hits[-1]
                    kw = {"contrast": "contrast", "dark": "darkest", "pale": "palest", "whole": "whole"}[h.lastgroup]
                    # bare "darkest (L* ..)" / "palest (L* ..)" means a third only if "thirds" follows
                    if kw in ("darkest", "palest") and re.search(r'(?:darkest|palest)\s*\(?\s*(?:L\*)?\s*$', h.group(0)) \
                            and not re.search(r'\bthirds?\b', text[end:end + 70]):
                        kw = None
            if ch == "L*" and re.match(r'\s*[;,]\s*a\*', post):
                col = "cie_L"                       # whole-structure colour triple (L*; a*; b*)
            elif cm.group(1):
                col = "cie_L_end_contrast"
            elif kw == "darkest":
                col = "cie_L_darkest_third"
            elif kw == "palest":
                col = "cie_L_palest_third"
            elif kw == "contrast":
                col = "cie_L_end_contrast"
            else:
                col = COLOUR_MAP[ch]
            unit = ch
        elif unit is None and re.search(r'(?:contrast|difference)[^;()]{0,50}(?:thirds|ends)?[^;()\d]{0,12}$',
                                        window, re.I) and not re.search(r'contrasting', window[-30:]):
            col, unit = "cie_L_end_contrast", "L*"
        elif unit is None and re.search(r'darkest third[^;()\d]{0,8}$', window, re.I):
            col, unit = "cie_L_darkest_third", "L*"
        elif unit is None and re.search(r'palest third[^;()\d]{0,8}$', window, re.I):
            col, unit = "cie_L_palest_third", "L*"
        elif unit == "mm2" or (unit is None and re.search(r'\barea\b[^;\d]{0,12}$', window, re.I)):
            col = "area_mm2"
        elif unit == "x" or (unit is None and re.search(r'^\s*(?:as long as|times)', after)):
            unit = "x"
            rid = rid_clause = rid_after = None
            if ev.ratio_rx:
                semi_only = max(text.rfind(";", sent[0], start), sent[0])
                ms = list(ev.ratio_rx.finditer(text[max(prev_end, semi_only):start]))
                if ms:
                    rid = ms[-1].group(1)
                ma = re.match(r'\s*[(\[]\s*(?:proportions\.ratio_)?(' + ev.ratio_rx.pattern + r')', after)
                if ma:
                    rid_after = ma.group(2)
                mc = list(ev.ratio_rx.finditer(clause))
                rid_clause = mc[-1].group(1) if mc else None
            lp = [p for p in lmk if semi - 5 <= p[0] < start] or [p for p in lmk if prev_end - 60 <= p[0] < start]
            aft = post[m.end():] if m else post
            aft = re.sub(r'^\s*\([^)]*\)', ' ', aft)
            sm = re.match(r'\s*(?:times\s+)?as long as (?:that of\s+)?(?:the\s+)?([a-z +\-]+)', aft, re.I) or \
                re.match(r'\s*(?:the\s+)?length of (?:the\s+)?((?:' + ev.struct_alt + r'))', aft, re.I) or \
                re.match(r'\s*(?:the\s+|its\s+)?((?:' + ev.struct_alt + r'))\s+length', aft, re.I)
            if sm and not re.match(r'\s*(wide|broad)', sm.group(1)):
                rel_target = sm.group(1)
            ratio_words = bool(re.search(r'\bratio\b|\w/\w', clause)) and not re.search(r'length/width[^;]{0,6}$', window)
            if rid_after:
                col = f"ratio_{rid_after}"
            elif rel_target:
                col = None                          # resolved below from the structures
            elif rid:
                col = f"ratio_{rid}"
            elif lp:
                col = f"lmkrel_{lp[-1][1]}_{lp[-1][2]}"
            elif not rel_target and not ratio_words:
                col = "aspect_ratio"
            elif not rel_target and rid_clause:
                col = f"ratio_{rid_clause}"
        elif unit in ("mm", "um"):
            lp = [p for p in lmk if prev_end - 60 <= p[0] < start]
            wm = re.match(r'\s*(long|in length|wide|broad|in width|in diameter)\b', after)
            words = re.findall(r'\b(length|long|width|wide|broad|breadth|perimeter|circumference|distance|span)\b',
                               window, re.I)
            if lp and (not words or words[-1].lower() in ("distance", "span")):
                col = f"lmkmm_{lp[-1][1]}_{lp[-1][2]}"
            elif wm:
                col = "length_mm" if wm.group(1) in ("long", "in length") else "height_mm"
            elif words:
                col = {"length": "length_mm", "long": "length_mm", "width": "height_mm", "wide": "height_mm",
                       "broad": "height_mm", "breadth": "height_mm", "perimeter": "perimeter_mm",
                       "circumference": "perimeter_mm"}.get(words[-1].lower())
        # structure: last mention in the sentence before the value, else carried over
        cands = [(p, cat) for p, cat in structs if p < start and
                 (not section or not cat or ev.cat_section.get(cat, section) == section or cat == "proportions")]
        in_sent = [x for x in cands if x[0] >= sent[0]]
        pick = (in_sent or cands[-1:] or [(None, None)])[-1]
        cat = pick[1]
        carried = not in_sent
        if rel_target and not col:
            rid = ratio_for(ev, cat, rel_target, section)
            col = f"ratio_{rid}" if rid else None
            if rid and re.search(r'\bcombined\b|\bsum\b|\btogether\b', rel_target) and "+" not in rid.split("/")[-1]:
                c["definition_error"] = f"denominator described as '{rel_target.strip()}' but {rid} uses a single structure"
        elif rid_after and rel_target:
            rr = ratio_for(ev, cat, rel_target, section)
            if rr and re.search(r'\bcombined\b|\bsum\b|\btogether\b', rel_target) and "+" not in rr.split("/")[-1]:
                c["definition_error"] = f"denominator described as '{rel_target.strip()}' but {rr} uses a single structure"
        if col and col.startswith(("lmkrel_", "lmkmm_")):
            s_txt = text[sent[0]:sent[1]]
            sets = ev.landmark_sets_for(col, s_txt, section)
            cat = sets
            carried = False
        elif col and col.startswith("ratio_"):
            cat, carried = "proportions", False
        all_m = structure_mentions(text[sent[0]:sent[1]], ev, section, skip_relative=False)
        c.update(unit=unit, column=col, category=cat, category_carried=carried,
                 cat_mentions={cc for pp, cc in in_sent if cc} | {cc for pp, cc in all_m if cc},
                 lmk_candidates={f"lmkrel_{a}_{b}" for p_, a, b in lmk if sent[0] <= p_ < sent[1]} |
                 {f"lmkmm_{a}_{b}" for p_, a, b in lmk if sent[0] <= p_ < sent[1]},
                 genus_context=bool(re.search(r'all species|across (?:the )?(?:genus|species)|genus range|'
                                              r'among (?:the )?species|species treated', c["context"], re.I)),
                 key_context=bool(re.search(r'couplet|\bkey\b|observed', c["context"], re.I)))
        claims.append(c)
    for k, c in enumerate(claims):
        stop = claims[k + 1]["start"] if k + 1 < len(claims) else len(text)
        tm = FEATURE_TAG_RX.search(text, c["end"], min(stop, c["end"] + 90))
        c["_tag"] = tm.group(0)[1:-1] if tm else ""
    # comparisons
    comps = []
    name_alt = ev.name_rx.pattern[:-len(r"(?![\w'])")]
    lst = r'(?:' + name_alt + r')(?:(?:,\s*(?:and\s+)?|\s+and\s+|\s+or\s+)(?:' + name_alt + r'))*'
    comp_rx = re.compile(
        r'(?P<dir>' + UP_WORDS + '|' + DOWN_WORDS + r')(?:\s*\([^)]*\))?(?P<mid>[^.;]{0,80}?)\s+than\s+'
        r'(?:in\s+|that of\s+|those of\s+|any other\s+)?'
        r'(?:(?P<all>all other species measured\s*\((?P<N>\d+)\))(?:\s*,?\s*except\s+(?P<exc>' + lst + r'))?'
        r'|(?P<list>' + lst + r'))')
    for m in comp_rx.finditer(text):
        who = m.group("all") and (m.group("all") + (f" except {m.group('exc')}" if m.group("exc") else "")) \
            or m.group("list")
        comps.append((m.start(), m.end(), m.group("dir"), m.group("N"), m.group("exc"), m.group("list"), "than",
                      who, m.start("dir")))
    exc_rx = re.compile(r'(?P<dir>exceeded|surpassed)\s+only\s+by\s+(?P<list>' + lst + r')')
    for m in exc_rx.finditer(text):
        comps.append((m.start(), m.end(), "greater", None, m.group("list"), None, "exceeded_only_by",
                      m.group("list"), m.start()))
    return claims, comps, structs


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def parse_citations(treat: Dict) -> List[Tuple[str, Optional[str], str]]:
    out = []
    for c in treat.get("citations", []) or []:
        val = str(c.get("value", ""))
        val = re.sub(r'n\s*=\s*\d+', ' ', val)
        val = re.sub(r'mean\s*' + NUM, ' ', val)
        nums = [n.replace("−", "-") for n in re.findall(NUM, re.sub(r'[LabC]\*', ' ', val))]
        if not nums:
            continue
        out.append((nums[0], nums[1].lstrip("-") if len(nums) > 1 and re.search(r'\d-\d', val) else
                    (nums[1] if len(nums) > 1 else None), str(c.get("id", ""))))
    return out


def feature_matches(row, lo, hi, approx=False) -> bool:
    if row is None:
        return False
    if approx and hi is None:
        v = float(lo)
        return row["min"] * 0.9 - 1e-9 <= v <= row["max"] * 1.1 + 1e-9
    if hi is None:
        return (matches(lo, row["min"]) and matches(lo, row["max"])) or matches(lo, row["mean"]) or \
            matches(lo, row["median"])
    return matches(lo, row["min"]) and matches(hi, row["max"])


def verify_measurement(c: Claim, code: str, ev: Evidence, cits, issues: List[Dict]):
    sp = ev.by_sp.get(code)
    lo, hi = c["lo"], c["hi"]
    declared = [ev.canon_fid(fid) for a, b, fid in cits if a == lo and b == hi]
    if c.get("_tag"):
        declared.insert(0, ev.canon_fid(c["_tag"]))
    ctx_fid = ev.canon_fid(f"{c['category']}.{c['column']}") if c.get("category") and c.get("column") else None
    sp_can = {ev.canon_fid(f): r for f, r in sp.iterrows()} if sp is not None else {}
    matching = [f for f, r in sp_can.items() if feature_matches(r, lo, hi, c.get("approx"))]
    c["declared"] = ";".join(dict.fromkeys(declared))
    c["context_feature"] = ctx_fid or ""
    c["matching_features"] = ";".join(matching[:6])
    target, status, etype, note = None, "ok", "", ""
    mentioned = {ev.canon_cat(x) for x in c.get("cat_mentions", set())}
    if ctx_fid and ctx_fid in matching:
        target = ctx_fid
        if declared and ctx_fid not in declared:
            note = f"citation says {declared[0]}"
    else:
        dec_ok = [d for d in declared if d in matching]
        if dec_ok:
            target = dec_ok[0]
            dcat, dcol = target.split(".", 1)
            ccol = c.get("column")
            eq = ev.col_equiv.get(dcol) or ev.col_equiv.get(ccol or "")
            col_ok = ccol in (None, dcol) or dcol in c.get("lmk_candidates", set()) or \
                (eq is not None and {dcol, ccol} <= {eq[1], *[k for k, v in ev.col_equiv.items() if v == eq]})
            cat_ok = (dcat in mentioned or not mentioned or c.get("category") in (None, dcat)
                      or (c.get("category_carried") and c.get("section") and
                          ev.cat_section.get(dcat) == c.get("section"))
                      or dcat in ("proportions", "forewing_keypoints", "head_keypoints"))
            if not (col_ok and cat_ok):
                status, etype = "review", "attribution_mismatch"
                note = f"words suggest {ctx_fid or c.get('column')}; value matches cited {target}"
                ctx_row = sp_can.get(ctx_fid) if ctx_fid else None
                if ctx_row is not None and not feature_matches(ctx_row, lo, hi, c.get("approx")) and \
                        not c.get("category_carried"):
                    # the words name a different measurement that exists for this species
                    status, etype = "error", "column_confusion"
                    note = f"text names {ctx_fid} ({ctx_row['min']:.4g}–{ctx_row['max']:.4g}); value is {target}"
        elif matching and not ctx_fid:
            cand = [f for f in matching if f.split(".", 1)[0] in mentioned] or matching
            target = cand[0]
            if len(cand) > 1 and len({f.split('.', 1)[1] for f in cand}) > 1:
                status, etype, note = "review", "unresolved_attribution", f"value fits {', '.join(cand[:3])}"
            else:
                note = "uncited; matched by value"
    if target is None:
        ref = ctx_fid or (declared[0] if declared else None)
        ref_row = sp_can.get(ref) if ref else None
        if c["genus_context"] and ref and ref in ev.genus_can:
            g = ev.genus_can[ref]
            if matches(lo, g["gmin"]) and (hi is None or matches(hi, g["gmax"])):
                target, note = ref, "genus-wide range"
        if target is None and c["key_context"] and ev.key:
            for r in ev.key["couplets"]:
                for ch in r["characters"]:
                    for side in ("A", "B"):
                        rng = ch.get(f"{side}_range") or [None, None]
                        if (matches(lo, rng[0]) and (hi is None or matches(hi, rng[1]))) or \
                                (hi is None and matches(lo, ch.get("threshold"))):
                            target, note = ev.canon_fid(ch["feature_id"]), f"key couplet {r['number']} value"
        if target is None:
            status = "error"
            other, grp = [], False
            if ref:
                for osp in ev.by_sp:
                    if osp == code:
                        continue
                    r = ev.row(osp, ref)
                    if r is not None and feature_matches(r, lo, hi):
                        other.append(osp)
                if ev.key:
                    for r in ev.key["couplets"]:
                        for ch in r["characters"]:
                            if ev.canon_fid(ch["feature_id"]) != ref:
                                continue
                            for side in ("A", "B"):
                                rng = ch.get(f"{side}_range") or [None, None]
                                if matches(lo, rng[0]) and (hi is None or matches(hi, rng[1])):
                                    grp = True
            unit_hit = ref_row is not None and any(
                abs(float(lo) - ref_row["min"] * k) <= 0.02 * abs(ref_row["min"] * k) + 1e-9 for k in (1000.0, 0.001))
            if matching and ref:
                etype = "column_confusion"
                note = f"value belongs to {matching[0]}; text/citation names {ref}"
            elif matching:
                status, etype, note = "review", "unresolved_attribution", f"value fits {', '.join(matching[:3])}"
                target = matching[0]
            elif other:
                etype, note = "cross_species_value", f"{ref}: value of {', '.join(other[:3])}"
            elif grp:
                etype, note = "group_value_as_species", "pooled key-couplet range quoted as the species' own"
            elif unit_hit or c.get("unit") == "um":
                etype = "unit_confusion"
            elif ref_row is not None and rel_err(lo, ref_row["min"]) <= 0.10 and \
                    (hi is None or rel_err(hi, ref_row["max"]) <= 0.10):
                etype, note = "value_rounding", f"{ref}: actual {ref_row['min']:.4g}–{ref_row['max']:.4g}"
            elif ref is None:
                etype, note = "value_fabrication", "no feature of this species has this value; wording unresolved"
            else:
                etype = "value_fabrication"
                note = (f"{ref}: actual {ref_row['min']:.4g}–{ref_row['max']:.4g} (n={int(ref_row['n'])})"
                        if ref_row is not None else f"species has no data for {ref}")
            target = target or ref
    if c.get("definition_error") and status != "error":
        status, etype, note = "error", "column_confusion", c["definition_error"]
    c["feature"] = target or ""
    c["status"], c["type"], c["note"] = status, etype, note
    row = sp_can.get(target) if target and status != "error" else None
    if row is not None and "genus" not in note and "couplet" not in note:
        if c.get("mean_claim") and not matches(c["mean_claim"], row["mean"]) and \
                not matches(c["mean_claim"], row["median"]):
            ty = "value_rounding" if rel_err(c["mean_claim"], row["mean"]) <= 0.10 else "value_fabrication"
            c["status"], c["type"] = "error", ty
            c["note"] = (c["note"] + "; " if c["note"] else "") + f"mean {c['mean_claim']} vs {row['mean']:.4g}"
        if c.get("n_claim"):
            n_c, n_a = int(float(c["n_claim"])), int(row["n"])
            if n_c != n_a:
                ty = "sample_inflation" if n_c > n_a else "sample_deflation"
                c["status"], c["type"] = "error", ty
                c["note"] = (c["note"] + "; " if c["note"] else "") + f"n={n_c} claimed, {n_a} in matrix"
    if c["status"] in ("error", "review"):
        issues.append(dict(kind="measurement", type=c["type"], feature=c["feature"], note=c["note"],
                           context=c["context"], severity=c["status"]))


WORD_COLUMN = [
    (r'palest third', "cie_L_palest_third"), (r'darkest third', "cie_L_darkest_third"),
    (r'contrast', "cie_L_end_contrast"), (r'paler|darker|lightness|colou?r', "cie_L"),
    (r'\blength\b|longer|shorter', "length_mm"), (r'\bwidth\b|wider|narrower|broader', "height_mm"),
    (r'perimeter', "perimeter_mm"), (r'\barea\b', "area_mm2"),
    (r'elongate|slender|stout|length/width', "aspect_ratio"),
]


def parse_sheet_comparisons(lines: List[str], ev: Evidence) -> List[Dict]:
    out = []
    for ln in lines:
        m = re.match(r'\[C\d+\]\s*\[([^\]]+)\]\s*(.*?):\s*(.*)$', ln)
        if not m:
            continue
        fid, body = ev.canon_fid(m.group(1)), m.group(3)
        for part in body.split("; "):
            pm = re.match(r'(.*?) than in (.*)$', part)
            if pm:
                out.append({"feature": fid, "dir": pm.group(1), "who": pm.group(2).strip()})
    return out


def norm_who(s: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r',\s*and\s+', ' and ', s or "")).strip(" .,;")


def comparison_feature(clause_start, pos_end, claims, clause_text, dword, structs, ev):
    inside = [c for c in claims if clause_start <= c["start"] < pos_end and c.get("feature")]
    tl = re.search(r'\b(?:this|its|the)\s+(length|width|lightness|colou?r|palest third|darkest third|contrast)\b',
                   clause_text, re.I)
    col = None
    for rx, cc in WORD_COLUMN:
        if re.search(rx, tl.group(1) if tl else "", re.I):
            col = cc
            break
    if col is None and not inside:
        for rx, cc in WORD_COLUMN:
            if re.search(rx, dword, re.I) or re.search(rx, clause_text, re.I):
                col = cc
                break
    if inside and not tl:
        c = inside[-1]
        if re.match(r'paler|darker', dword) and not c["feature"].split(".", 1)[1].startswith("cie_L"):
            L = [x for x in inside if x["feature"].split(".", 1)[1].startswith("cie_L")]
            if L:
                return L[-1]["feature"], "value"
        else:
            return c["feature"], "value"
    cats = [c for p, c in structs if p < pos_end and c]
    if col and cats:
        return ev.canon_fid(f"{cats[-1]}.{col}"), "words"
    if inside:
        return inside[-1]["feature"], "value"
    return None, None


def check_comparison(fid, code, ev, dword, N, exc, lst, form):
    up = bool(re.match(UP_WORDS, dword)) or form == "exceeded_only_by"
    rows = ev.rows_n2(fid)
    me = rows.get(code)
    if me is None:
        return [f"species has fewer than 2 specimens for {fid}"]
    others = [s for s in rows if s != code]
    named = lambda txt: [ev.name_forms[m.group(1)] for m in ev.name_rx.finditer(txt or "")]  # noqa: E731
    problems = []
    if form == "exceeded_only_by":
        exceeders = named(exc)
        for s in others:
            if s not in exceeders and not rows[s]["max"] < me["min"]:
                problems.append(f"{s} overlaps or exceeds")
        for s in exceeders:
            if s not in rows:
                problems.append(f"{s} not measured (n>=2)")
            elif not rows[s]["max"] > me["max"]:
                problems.append(f"{s} does not exceed")
        return problems
    if N is not None:
        group = [s for s in others if s not in named(exc)]
        if int(N) != len(others):
            problems.append(f"'all other species measured ({N})' but {len(others)} measured")
        for s in named(exc):
            if s not in rows:
                problems.append(f"excepted {s} not measured (n>=2)")
    else:
        group = named(lst)
    for s in group:
        if s == code:
            continue
        if s not in rows:
            problems.append(f"{s} not measured (n>=2)")
            continue
        ok = rows[s]["max"] < me["min"] if up else rows[s]["min"] > me["max"]
        if not ok:
            problems.append(f"{s} {rows[s]['min']:.4g}–{rows[s]['max']:.4g} vs {me['min']:.4g}–{me['max']:.4g}")
    return problems


def verify_comparisons(text, comps, claims, code, ev: Evidence, issues, section, structs, sheet_comps):
    out = []
    sents = sentence_spans(text)
    prev_end, prev_fid = 0, None
    for a, b, dword, N, exc, lst, form, who, dpos in sorted(comps):
        sent = next(((x, y) for x, y in sents if x <= a < y), (0, len(text)))
        cstart = max(sent[0], prev_end)
        connector = text[prev_end:dpos] if prev_end > sent[0] else None
        fid, how = comparison_feature(cstart, b, claims, text[cstart:b], dword, structs, ev)
        if connector is not None and re.fullmatch(r'\s*[,;]?\s*(?:but|and|while|whereas)?\s*(?:is\s+)?', connector) \
                and prev_fid and how != "value":
            fid, how = prev_fid, "previous comparison"
        # the data-sheet statement this comparison quotes (verbatim species list)
        wn = norm_who(who)
        dir_word = re.match(r'\w+', dword).group(0)
        sheet = [s for s in sheet_comps if norm_who(s["who"]) == wn]
        rec = dict(kind="comparison", section=section, context=text[max(sent[0], a - 60):b], feature=fid or "",
                   status="ok", type="", note=f"feature from {how}" if how else "")
        out.append(rec)
        if sheet and fid not in {s["feature"] for s in sheet}:
            same_dir = [s for s in sheet if s["dir"].split()[0] == dir_word] or sheet
            src = same_dir[0]["feature"]
            probs = check_comparison(src, code, ev, dword, N, exc, lst, form)
            if not probs:
                rec.update(feature=src, status="warning", type="ambiguous_comparison_anchor",
                           note=f"statement for {src} placed after {fid or 'another value'}")
                prev_end, prev_fid = b, src
                continue
        if not fid:
            rec.update(status="unverifiable", note="no measured feature found for this comparison")
            prev_end = b
            continue
        probs = check_comparison(fid, code, ev, dword, N, exc, lst, form)
        if probs:
            rec.update(status="error", type="fabricated_comparison",
                       note=("(not a data-sheet statement) " if not sheet else "") + "; ".join(probs[:4]))
            issues.append(dict(kind="comparison", **{k: rec[k] for k in ("type", "feature", "note", "context")}))
        elif not sheet:
            rec["note"] += " (model-composed list; verified against the matrix)"
        prev_end, prev_fid = b, fid
    for r in out:
        if r["status"] == "warning":
            issues.append(dict(kind="comparison", type=r["type"], feature=r["feature"], note=r["note"],
                               context=r["context"], severity="warning"))
    return out


def verify_wording(field, text, code, ev: Evidence, issues, tier1=True):
    recs = []
    if tier1:
        for term in pol.find_tier2_terms(text):
            recs.append(dict(kind="wording", section=field, status="error", type="statistic_outside_remarks",
                             feature="", note=term, context=text[:120]))
    genus = ev.profile.get("taxon", {}).get("genus", "") or "Diaphorina"
    stop = {"species", "specimens", "spp", "and", "with", "from", "differs", "differ", "also", "which", "that",
            "have", "this", "these", "there", "where", "were", "being", "treated", "examined", "measured"}
    for m in re.finditer(rf'\b(?:{re.escape(genus)}|{re.escape(genus[0])}\.)\s+(?:sp\.\s*[\w\']+|cf\.\s*\w+|[a-z]{{4,}})',
                         text):
        s = m.group(0)
        if s.split()[-1] in stop or ev.name_rx.match(s):
            continue
        recs.append(dict(kind="wording", section=field, status="error", type="unknown_taxon",
                         feature="", note=s, context=text[max(0, m.start() - 40):m.end() + 20]))
    tags = FEATURE_TAG_RX.findall(text)
    if tags:
        recs.append(dict(kind="format", section=field, status="warning", type="internal_identifier_in_text",
                         feature="", note=f"{len(tags)} bracketed feature identifiers, e.g. {tags[0]}",
                         context=text[:120]))
    for r in recs:
        issues.append({k: r[k] for k in ("kind", "type", "feature", "note", "context")} |
                      {"severity": r["status"]})
    return recs


# ── Remarks ─────────────────────────────────────────────────────────────────

ANALYSIS_WORDS = [
    ("landmark", r'landmark-based|vein-junction landmarks|landmark analysis'),
    ("semilandmark", r'semilandmark|outline|mean shape|\bshape\b'),
    ("texture", r'textur'),
    ("colour-homology", r'colou?r[- ](?:pattern[- ])?homology|colou?r[- ]grid|colou?r(?:-pattern)?\b'),
    ("pairwise", r'Kruskal|Dunn|pairwise'),
]


def analyses_in(txt: str) -> List[str]:
    return [a for a, rx in ANALYSIS_WORDS if re.search(rx, txt, re.I)]


def parse_tier2_lines(lines: List[str], ev: Evidence) -> List[Dict]:
    out = []
    for ln in lines:
        if ln.startswith("("):
            continue
        rec = {"text": ln, "analysis": None, "category": None, "near": set(), "far": set(), "shared": set()}
        if "landmark-based" in ln:
            rec["analysis"] = "landmark"
        elif "semilandmark" in ln:
            rec["analysis"] = "semilandmark"
        elif "texture cluster" in ln:
            rec["analysis"] = "texture"
        elif "colour-homology" in ln:
            rec["analysis"] = "colour-homology"
        elif "shape cluster" in ln:
            rec["analysis"] = "semilandmark"
        elif re.search(r'Kruskal|Dunn|Feature groups', ln):
            rec["analysis"] = "pairwise"
        head = ln.split(":")[0] if ":" in ln else ln
        cats = [c for _, c in structure_mentions(head, ev, "", skip_relative=False) if c]
        rec["category"] = ev.canon_cat(cats[0]) if cats else None
        if rec["analysis"] == "landmark":
            rec["category"] = "forewing_keypoints" if "forewing" in ln else "head_keypoints"
        m = re.search(r'closest to (.*?)(?: and farthest from (.*?))?(?: \(|$)', ln)
        if m:
            rec["near"] = {ev.name_forms[x.group(1)] for x in ev.name_rx.finditer(m.group(1))}
            if m.group(2):
                rec["far"] = {ev.name_forms[x.group(1)] for x in ev.name_rx.finditer(m.group(2))}
        m = re.search(r'shared with (.*?)\)', ln)
        if m:
            rec["shared"] = {ev.name_forms[x.group(1)] for x in ev.name_rx.finditer(m.group(1))}
        if rec["analysis"] == "pairwise":
            rec["shared"] = {ev.name_forms[x.group(1)] for x in ev.name_rx.finditer(ln)}
        out.append(rec)
    return out


REL_RX = re.compile(r'closest|nearest|farthest|furthest|group(?:s|ed)?\b|cluster(?:s|ed)?\b|shared with|'
                    r'differ(?:s|ed)? significantly|pairwise')


def verify_remarks(text, code, ev: Evidence, blocks, issues, sheet_comps=None) -> List[Dict]:
    recs = []
    t2 = parse_tier2_lines(block(blocks, "TIER 2"), ev)
    nh = [x for x in block(blocks, "SUPPLIED NATURAL HISTORY") if not x.startswith("(")]
    nh_text = " ".join(nh)
    allowed_nums = []
    for ln in block(blocks, "TIER 2") + nh:
        allowed_nums += [float(x) for x in pol.numbers_in(ln)]
    loc_text = ""
    if ev.loc is not None:
        loc_text = " ".join(" ".join(r) for r in ev.loc[ev.loc["species"] == code].values.tolist())
    structs = structure_mentions(text, ev, "", skip_relative=False)
    for sa, sb in sentence_spans(text):
        sent = text[sa:sb]
        # a Tier-1 comparative statement of the data sheet, copied into Remarks
        norm_sent = norm_who(sent)
        quoted = [c for c in (sheet_comps or []) if norm_who(c["who"]) and norm_who(c["who"]) in norm_sent]
        if quoted:
            probs = []
            for q in quoted[:2]:
                m = re.match(r'all other species measured\s*\((\d+)\)(?:\s*,?\s*except\s+(.*))?$',
                             q["who"].strip())
                N, exc, lst = (m.group(1), m.group(2), None) if m else (None, None, q["who"])
                probs += check_comparison(q["feature"], code, ev, q["dir"].split()[0], N, exc, lst, "than")
            rec = dict(kind="comparison", section="Remarks", context=sent.strip()[:220],
                       feature=quoted[0]["feature"],
                       status="error" if probs else "warning",
                       type="fabricated_comparison" if probs else "tier1_statement_in_remarks",
                       note="; ".join(probs[:3]) or "measurable comparative statement placed in REMARKS; "
                                                   "it belongs in the Diagnosis or Description")
            recs.append(rec)
            issues.append({k: rec[k] for k in ("kind", "type", "feature", "note", "context")} |
                          {"severity": rec["status"]})
            continue
        names = [(m.start() + sa, ev.name_forms[m.group(1)]) for m in ev.name_rx.finditer(sent)
                 if ev.name_forms[m.group(1)] != code]
        if names:
            sent_an = analyses_in(sent)
            lm_sent = bool(re.search(r'landmark-based|landmark analysis|vein-junction', sent)) and \
                not re.search(r'semilandmark|outline', sent)
            cuts = {sa, sb}
            for m in re.finditer(r';|,\s+(?:and\s+)?(?=(?:that of|the |cell |forewing|head|vertex|metatibia|'
                                 r'metafemur|paramere|female|male|circumanal|apical|median|distal|proximal|'
                                 r'with (?:Diaphorina|D\.)|to (?:that of |those of )?(?:Diaphorina|D\.)))|'
                                 r'\band that of\b|\bwhile\b|\bwhereas\b', sent):
                cuts.add(sa + m.start())
            for m in re.finditer(r'\b(?:and\s+)?(?:farthest|furthest)\b', sent):
                cuts.add(sa + m.start())
            cuts = sorted(cuts)
            carry_cat = None
            for ca, cb in zip(cuts, cuts[1:]):
                cl = text[ca:cb]
                cl_names = [c for p, c in names if ca <= p < cb]
                cats_here = [ev.canon_cat(c) for p, c in structs if ca <= p < cb and c]
                if not cl_names:
                    if cats_here:
                        carry_cat = cats_here[-1]
                    continue
                rel = "far" if re.search(r'farthest|furthest', cl) else "near" if re.search(r'closest|nearest', cl) \
                    else "shared"
                an = analyses_in(cl) or sent_an
                cats = cats_here or ([carry_cat] if carry_cat else [])
                if cats_here:
                    carry_cat = cats_here[-1]
                if lm_sent or rel == "far":
                    an = ["landmark"]
                    cats = ["head_keypoints" if re.search(r'\bhead\b', sent) and not re.search(r'wing', sent)
                            else "forewing_keypoints"]
                if "respectively" in cl and len(cats_here) > 1:
                    pool = set()
                    for r in t2:
                        if r["category"] in cats_here:
                            pool |= r["near"] | r["shared"]
                    ok = set(cl_names) <= pool
                    rec = dict(kind="remarks", section="Remarks", context=cl.strip()[:220], feature="+".join(cats_here),
                               status="ok" if ok else "error", type="" if ok else "remarks_misattribution",
                               note="respectively-list checked against the union of the named structures")
                    recs.append(rec)
                    continue
                def pool_of(r):
                    return r["far"] if rel == "far" else (r["near"] | r["shared"])
                exact = [r for r in t2 if set(cl_names) <= pool_of(r) and (not cats or r["category"] in cats
                                                                           or r["category"] is None)]
                rec = dict(kind="remarks", section="Remarks", context=cl.strip()[:220], feature="+".join(cats),
                           status="ok", type="", note=f"{'/'.join(an) or '?'} / {rel}")
                if exact:
                    good = [r for r in exact if not an or r["analysis"] in an or r["analysis"] is None]
                    if not good:
                        rec.update(status="warning", type="analysis_label_mismatch",
                                   note=f"species and structure match a {exact[0]['analysis']} statement; text names "
                                        f"{'/'.join(an)}")
                else:
                    anywhere = set().union(*[pool_of(r) | r["near"] | r["shared"] | r["far"] for r in t2]) \
                        if t2 else set()
                    if set(cl_names) <= anywhere:
                        rec.update(status="error", type="remarks_misattribution",
                                   note=f"{'/'.join(an)}/{rel} {'+'.join(cats) or '?'}: {sorted(set(cl_names))} "
                                        f"not given for this structure/relation")
                    else:
                        rec.update(status="error", type="remarks_fabrication",
                                   note=f"{sorted(set(cl_names) - anywhere)} not in any supplied statement")
                recs.append(rec)
        # natural history
        nh_hit = re.search(r'host|collected|locality|label|coordinates?|elevation|altitude|trap|slide', sent, re.I)
        props = []
        for m in re.finditer(r'\b[A-Z][a-z]{2,}\b', sent):
            w = m.group(0)
            if w in SENTENCE_START_WORDS or any(w in f for f in ev.name_forms):
                continue
            if re.search(rf'\b{re.escape(w)}\b', nh_text + " " + loc_text, re.I):
                continue
            if any(w.lower() in r["text"].lower() for r in t2):
                continue
            props.append(w)
        dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\.[ivx]+\.\d{4}\b|\b\d{1,2}\.\d{1,2}\.\d{4}\b', sent)
        bad_dates = [d for d in dates if d not in nh_text and d not in loc_text]
        scrub = re.sub(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\.[ivx]+\.\d{4}\b|\b\d{1,2}\.\d{1,2}\.\d{4}\b|'
                       r'\b[A-Z][A-Za-z]{1,6}-\d+[A-Z]?(?:\s*[–-]\s*\d+)?(?:-\d+)?\b', ' ', ev.name_rx.sub(' ', sent))
        bad_nums = [x for x in pol.numbers_in(scrub)
                    if not pol.number_is_allowed(x, allowed_nums) and x not in loc_text]
        if nh_hit or props or bad_dates or bad_nums:
            rec = dict(kind="remarks", section="Remarks", context=sent.strip()[:220], feature="natural_history",
                       status="ok", type="", note="")
            probs = []
            if props:
                probs.append("proper nouns not in the supplied data: " + ", ".join(sorted(set(props))))
            if bad_dates:
                probs.append("dates not in the supplied data: " + ", ".join(bad_dates))
            if bad_nums:
                probs.append("numbers not in the supplied data: " + ", ".join(bad_nums))
            if nh_hit and not nh and re.search(r'\b(?:collected|host plant (?:is|was)|reared)\b', sent) \
                    and not re.search(r'\bno\b|not (?:supplied|available|recorded)', sent, re.I):
                probs.append("natural-history statement although none was supplied")
            if probs:
                rec.update(status="error", type="unsupported_natural_history", note="; ".join(probs))
            recs.append(rec)
    for r in recs:
        if r["status"] in ("error", "warning"):
            issues.append({k: r[k] for k in ("kind", "type", "feature", "note", "context")} |
                          {"severity": r["status"]})
    return recs


def absence_claims(treat, code, ev: Evidence, issues) -> List[Dict]:
    """Sections that declare data absent although the matrix has it (warning)."""
    recs = []
    sp = ev.by_sp.get(code)
    if sp is None:
        return recs
    have = defaultdict(int)
    for f in sp.index:
        if f in ev.fdict.index and ev.fdict.loc[f, "tier"] in ("key", "description"):
            have[ev.fdict.loc[f, "section"]] += 1
    for d in treat.get("description", []) or []:
        t = d.get("text", "")
        if not pol.numbers_in(t) and re.search(r'no .{0,60}(?:measurements?|data)|not assessable', t, re.I) \
                and have.get(d.get("section"), 0) > 0:
            recs.append(dict(kind="absence", section=d.get("section"), status="warning",
                             type="omitted_available_data",
                             feature="", note=f"{have[d.get('section')]} Tier-1 features available",
                             context=t[:160]))
    for r in recs:
        issues.append({k: r[k] for k in ("kind", "type", "feature", "note", "context")} | {"severity": "warning"})
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def audit_species(code: str, treat: Dict, sheet: str, ev: Evidence) -> Dict:
    issues: List[Dict] = []
    records: List[Dict] = []
    cits = parse_citations(treat)
    blocks = sheet_blocks(sheet)
    sheet_comps = parse_sheet_comparisons(block(blocks, "COMPARATIVE STATEMENTS"), ev)
    fields = [("Diagnosis", treat.get("diagnosis", "") or "")]
    fields += [(d.get("section", "?"), d.get("text", "") or "") for d in treat.get("description", []) or []]
    fields += [("Sexual dimorphism", treat.get("sexual_dimorphism", "") or "")]
    numbers_checked = 0
    for name, text in fields:
        sec = name if name not in ("Diagnosis", "Sexual dimorphism") else ""
        claims, comps, structs = extract_claims(text, ev, sec)
        for c in claims:
            verify_measurement(c, code, ev, cits, issues)
            numbers_checked += 1 + (c["hi"] is not None) + bool(c.get("mean_claim")) + bool(c.get("n_claim"))
            records.append(dict(kind="measurement", section=name,
                                **{k: c.get(k, "") for k in ("lo", "hi", "mean_claim", "n_claim", "unit",
                                                             "category", "column", "declared", "context_feature",
                                                             "feature", "status", "type", "note", "context")}))
        records += verify_comparisons(text, comps, claims, code, ev, issues, name, structs, sheet_comps)
        records += verify_wording(name, text, code, ev, issues)
    records += verify_wording("Remarks", treat.get("remarks", "") or "", code, ev, issues, tier1=False)
    records += verify_remarks(treat.get("remarks", "") or "", code, ev, blocks, issues, sheet_comps)
    records += absence_claims(treat, code, ev, issues)
    # citations that point at nothing in the text
    all_text = " ".join(t for _, t in fields)
    orphan = [f"{a}{'–' + b if b else ''} [{fid}]" for a, b, fid in cits
              if not re.search(rf'(?<![\d.]){re.escape(a)}(?![\d])', all_text)]
    repair = []
    for r in records:
        if r["status"] in ("error", "review") or r.get("type") in ("ambiguous_comparison_anchor",
                                                                     "internal_identifier_in_text",
                                                                     "analysis_label_mismatch",
                                                                     "tier1_statement_in_remarks"):
            repair.append(f"[{r.get('section', '')}] {r.get('type')}: {r.get('note', '')} — in: "
                          f"\"{str(r.get('context', ''))[:160]}\"")
    for r in records:
        r["species"] = code
    for i in issues:
        i["species"] = code
        i.setdefault("severity", "error")
    return {"species": code, "records": records, "issues": issues, "numbers_checked": numbers_checked,
            "repair_requests": repair,
            "orphan_citations": orphan, "n_citations": len(cits)}


def write_outputs(results: List[Dict], out: Path, ev: Evidence, args):
    out.mkdir(parents=True, exist_ok=True)
    recs = [r for res in results for r in res["records"]]
    counted = [r for r in recs if r["kind"] in ("measurement", "comparison", "remarks", "wording")
               and not (r["kind"] == "wording" and r["status"] == "ok")]
    errors = [r for r in counted if r["status"] == "error"]
    review = [r for r in counted if r["status"] == "review"]
    unver = [r for r in counted if r["status"] == "unverifiable"]
    warnings = [r for r in recs if r["status"] == "warning"]
    checked = [r for r in counted if r["kind"] != "wording"]
    n_wording_errors = sum(1 for r in counted if r["kind"] == "wording")
    denom = len(checked)
    etypes = Counter(r["type"] for r in errors)
    by_kind = Counter(r["kind"] for r in checked)
    summary = {
        "generated": datetime.now().isoformat(),
        "checker": f"biorag_confabulation_checker_v2 {CHECKER_VERSION}",
        "descriptions_dir": str(args.descriptions_dir), "matrix_dir": str(args.matrix_dir),
        "total_species": len(results),
        "total_traits_checked": denom,
        "total_traits_verified_ok": sum(1 for r in checked if r["status"] == "ok"),
        "claims_by_kind": dict(by_kind),
        "numbers_checked": sum(r["numbers_checked"] for r in results),
        "citations": sum(r["n_citations"] for r in results),
        "orphan_citations": sum(len(r["orphan_citations"]) for r in results),
        "errors": len(errors),
        "wording_errors_not_in_denominator": n_wording_errors,
        "needs_review": len(review),
        "unverifiable": len(unver),
        "warnings": len(warnings),
        "confabulation_rate": 100.0 * sum(1 for r in checked if r["status"] == "error") / denom if denom else 0.0,
        "confabulation_type_errors": dict(etypes.most_common()),
        "confabulation_type_errors_v1_names": dict(Counter(ERROR_TYPES_V1_EQUIVALENT.get(t, t)
                                                           for t in (r["type"] for r in errors)).most_common()),
        "review_types": dict(Counter(r["type"] for r in review).most_common()),
        "confabulation_type_warnings": dict(Counter(r["type"] for r in warnings).most_common()),
        "issues_by_species": dict(Counter(r["species"] for r in errors).most_common()),
        "errors_by_kind": dict(Counter(r["kind"] for r in errors)),
        "summary_vs_recomputed_mismatches": ev.summary_mismatch,
    }
    (out / "confabulation_summary.json").write_text(json.dumps(summary, indent=2))
    cols = ["species", "kind", "section", "status", "type", "feature", "lo", "hi", "mean_claim", "n_claim",
            "unit", "category", "column", "declared", "context_feature", "note", "context"]
    with open(out / "confabulation_claims.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow({k: str(r.get(k, "")).replace("\t", " ").replace("\n", " ") for k in cols})
    with open(out / "confabulation_issues.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in recs:
            if r["status"] in ("error", "review", "warning", "unverifiable"):
                w.writerow({k: str(r.get(k, "")).replace("\t", " ").replace("\n", " ") for k in cols})
    pdir = out / "per_species"
    pdir.mkdir(exist_ok=True)
    for res in results:
        (pdir / f"{res['species']}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    L = ["BioRAG v2 CONFABULATION AUDIT", f"Generated: {summary['generated']}", "=" * 70, "",
         f"Species: {summary['total_species']}   claims checked: {denom}   numbers checked: "
         f"{summary['numbers_checked']}",
         f"Errors: {summary['errors']} claims ({summary['confabulation_rate']:.2f}%)   "
         f"needs review: {len(review)}   unverifiable: {len(unver)}   warnings: {len(warnings)}", "",
         "Error types:"] + [f"  {k:32s} {v}" for k, v in etypes.most_common()] + ["", "Errors:"]
    for r in errors + review:
        L.append(f"  [{r['species']}] {r['status']}/{r['type']} {r.get('feature', '')}: {r.get('note', '')}")
        L.append(f"      \"{r.get('context', '')[:200]}\"")
    (out / "confabulation_report.txt").write_text("\n".join(L))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Independent confabulation audit of BioRAG v2 treatments")
    ap.add_argument("--descriptions_dir", required=True)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--key_tree", default=None)
    ap.add_argument("--localities", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--species", nargs="*", default=None)
    args = ap.parse_args()
    ev = Evidence(args)
    dd = Path(args.descriptions_dir)
    out = Path(args.output_dir) if args.output_dir else dd / "confabulation_report_v2"
    results = []
    for tj in sorted(dd.glob("*/*_treatment.json"), key=lambda p: pol_natural_key(p.parent.name)):
        code = tj.parent.name
        if args.species and code not in args.species:
            continue
        treat = json.loads(tj.read_text())
        sp = tj.parent / f"{code}_data_sheet.txt"
        sheet = sp.read_text() if sp.exists() else ""
        results.append(audit_species(code, treat, sheet, ev))
    s = write_outputs(results, out, ev, args)
    print(json.dumps({k: s[k] for k in ("total_species", "total_traits_checked", "claims_by_kind",
                                        "numbers_checked", "errors", "confabulation_rate",
                                        "confabulation_type_errors", "needs_review", "review_types",
                                        "unverifiable", "warnings", "confabulation_type_warnings",
                                        "orphan_citations", "summary_vs_recomputed_mismatches")}, indent=1))
    print(f"reports -> {out}")


if __name__ == "__main__":
    main()
