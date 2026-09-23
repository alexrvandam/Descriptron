#!/usr/bin/env python3
"""
biorag_description_refiner_v1.py — evidence-tiered species treatments
====================================================================

Turns the Tier-1 data matrix (biorag_key_feature_filter_v2.py), the
identification key (biorag_key_builder_v1.py), the statistical results of the
compiled phenomic data, and the qualitative image observations of an earlier
BioRAG/TOWLEY run into one concise treatment per species with the sections

    DIAGNOSIS            Tier 1 only (hand-measurable / visible)
    DESCRIPTION          Tier 1 only, one paragraph per body region
    SEXUAL DIMORPHISM    Tier 1 only
    REMARKS              Tier 2 (statistics, morphometric spaces, clusters)
                         + supplied natural history (host plant, locality)

How confabulation is prevented
  1. A deterministic SPECIES DATA SHEET is built from the matrix. Every number
     the model may use is printed there, next to its data-matrix feature ID.
  2. Prior image observations are passed with ALL numbers removed.
  3. The universal system prompt (biorag_system_prompts_v1.txt, sections
     [evidence_policy] + [refine_treatment]) states the tier rules.
  4. The reply is checked automatically: every number in Diagnosis /
     Description / Sexual dimorphism must be a Tier-1 value of the sheet,
     every number in Remarks must be in the sheet, no Tier-2 vocabulary may
     appear outside Remarks, and folder codes may not replace species names.
     Violations are sent back for up to two repair rounds; sentences that
     still fail are removed and listed in the validation report.

LLM backends: --llm-backend api | claude-code  (see biorag_llm_backend.py)

Outputs (--output_dir):
  <code>/<code>.txt                 monograph-style treatment text
  <code>/<code>_treatment.json      structured treatment + citations + validation
  <code>/<code>_data_sheet.txt      the exact evidence given to the model
  refine_validation_summary.tsv     one row per species
  llm_calls.jsonl                   provenance of every model call

Usage:
  python biorag_description_refiner_v1.py \
    --matrix_dir  .../Diaphorina_monograph/compiled_key_tier \
    --compiled_dir .../Diaphorina_compiled_29species \
    --prior_cache  .../Diaphorina_29species_key/biorag_cache \
    --key_tree    .../Diaphorina_monograph/key/key_tree.json \
    --localities  .../Diaphorina_monograph/localities/Diaphorina_localities_verified.tsv \
    --taxon-profile biorag_prompts/taxon_profiles/diaphorina_taxon_profile.yaml \
    --output_dir  .../Diaphorina_monograph/descriptions \
    --llm-backend claude-code
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import biorag_feature_policy as pol  # noqa: E402
from biorag_llm_backend import (add_backend_args, load_prompt_library,  # noqa: E402
                                make_llm_client, parse_json_response,
                                answering_model)

REFINER_VERSION = "1.1"
# 1.1 (2026-09-17): landmark shape-space neighbours need >= 5 species and the
#     farthest species is never also a nearest one; the independent audit
#     (biorag_confabulation_checker_v2.py) is part of validation; bracketed
#     data-sheet IDs are removed from the text; --repair_existing repairs
#     earlier treatments instead of rewriting them; citation IDs follow the
#     matrix's category aliases.


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

class Evidence:
    def __init__(self, args, profile):
        self.profile = profile
        md = Path(args.matrix_dir)
        self.long = pd.read_csv(md / "specimen_matrix_long.csv")
        self.fdict = pd.read_csv(md / "feature_dictionary.tsv", sep="\t").set_index("feature_id")
        self.summary = pd.read_csv(md / "species_feature_summary.csv")
        self.species = sorted(self.long["species"].unique(), key=natural_key)
        self.full = None
        self.diag = {}
        if args.compiled_dir:
            cd = Path(args.compiled_dir)
            fp = next(iter(sorted(cd.glob("*_full_features.csv"))), None)
            if fp is not None:
                self.full = pd.read_csv(fp, low_memory=False)
            dp = next(iter(sorted(cd.glob("*_diagnostic_report.json"))), None)
            if dp is not None:
                with open(dp) as f:
                    self.diag = json.load(f)
        self.key = None
        if args.key_tree and Path(args.key_tree).exists():
            with open(args.key_tree) as f:
                self.key = json.load(f)
        self.loc = None
        if args.localities and Path(args.localities).exists():
            self.loc = pd.read_csv(args.localities, sep="\t" if args.localities.endswith(".tsv") else ",",
                                   dtype=str).fillna("")
        self.prior_cache = Path(args.prior_cache) if args.prior_cache else None

    def display(self, code):
        return pol.species_display_name(code, self.profile)


# ─────────────────────────────────────────────────────────────────────────────
# Data sheet
# ─────────────────────────────────────────────────────────────────────────────

def fmt_range(row, unit):
    ref = row["median"]
    lo, hi, mean = pol.fmt(row["min"], unit, ref), pol.fmt(row["max"], unit, ref), pol.fmt(row["mean"], unit, ref)
    u = {"mm": " mm", "mm2": " mm²", "x": "×", "deg": "°"}.get(unit, "")
    pre = f"{unit} " if unit in ("L*", "a*", "b*", "C*") else ""
    if lo == hi:
        return f"{pre}{lo}{u} (n={int(row['n'])})", [row["min"]]
    return f"{pre}{lo}–{hi}{u} (mean {mean}; n={int(row['n'])})", [row["min"], row["max"], row["mean"]]


def build_sheet(ev: Evidence, code: str) -> Tuple[str, Dict]:
    prof = ev.profile
    allowed_t1: List[float] = []
    allowed_all: List[float] = []
    lines = []
    sp_long = ev.long[ev.long["species"] == code]
    specs = sp_long.groupby("specimen_id")["sex"].first()
    n_m, n_f = int((specs == "male").sum()), int((specs == "female").sum())
    lines.append(f"SPECIES: {ev.display(code)}   (folder code '{code}' — never print the code)")
    lines.append(f"Status: {pol.species_status(code, prof)}")
    lines.append(f"Specimens in the data matrix: {len(specs)} ({n_m} males, {n_f} females)")
    allowed_t1 += [len(specs), n_m, n_f]
    secs = prof.get("description_sections") or []
    lines.append(f"DESCRIPTION SECTIONS (use these names, in this order): {', '.join(secs)}")
    lines.append("")
    lines.append("=== TIER 1 — MEASUREMENTS, PROPORTIONS AND COLOUR (usable in every section) ===")
    summ = ev.summary[ev.summary["species"] == code].set_index("feature_id")
    genus = ev.summary.groupby("feature_id").agg(gmin=("min", "min"), gmax=("max", "max"),
                                                 nsp=("species", "nunique"))
    fd = ev.fdict
    by_section = defaultdict(list)
    for fid, row in summ.iterrows():
        if fid not in fd.index:
            continue
        meta = fd.loc[fid]
        if meta["tier"] not in pol.TIER1:
            continue
        col = meta["column"]
        if col.startswith("lmkmm_") or (col.startswith("lmkrel_") and row["n"] < 2):
            continue                       # keep sheets readable: landmark ratios with n>=2 only
        by_section[meta["section"]].append((fid, meta, row))
    order = {s: i for i, s in enumerate(secs)}
    for sec in sorted(by_section, key=lambda s: order.get(s, 99)):
        lines.append(f"[{sec}]")
        items = sorted(by_section[sec], key=lambda x: (x[1]["category"], x[1]["column"]))
        # colour: one line per structure (L*, a*, b*, colour word)
        colour = defaultdict(dict)
        for fid, meta, row in items:
            if meta["column"].startswith("cie_"):
                colour[meta["category"]][meta["column"]] = (fid, row)
                continue
            unit = meta["unit"] if isinstance(meta["unit"], str) else ""
            txt, nums = fmt_range(row, unit)
            g = genus.loc[fid]
            gtxt, gnums = "", []
            if g["nsp"] >= 3:
                gtxt = f"   | all species: {pol.fmt(g['gmin'], unit, row['median'])}–{pol.fmt(g['gmax'], unit, row['median'])} ({int(g['nsp'])} spp.)"
                gnums = [g["gmin"], g["gmax"], g["nsp"]]
            defin = meta["definition"] if meta["column"].startswith(("ratio_", "lmkrel_")) else ""
            lab = meta["label"] + (f" = {defin}" if defin and defin not in meta["label"] else "")
            lines.append(f"  [{fid}] {lab}: {txt}{gtxt}")
            allowed_t1 += nums + gnums + [row["n"]]
            # identifiers inside labels (landmark numbers, ratio names) are not measurements
            allowed_t1 += [float(x) for x in re.findall(r'\d+', lab)]
        for cat, d in colour.items():
            if "cie_L" not in d:
                continue
            fidL, rL = d["cie_L"]
            info = pol.structure_info(cat, prof)
            parts = []
            vals = {}
            for ch, lab in (("cie_L", "L*"), ("cie_a", "a*"), ("cie_b", "b*")):
                if ch in d:
                    fid_, r_ = d[ch]
                    parts.append(f"{lab} {pol.fmt(r_['min'], lab)}–{pol.fmt(r_['max'], lab)} [{fid_}]")
                    allowed_t1 += [r_["min"], r_["max"], r_["mean"], r_["median"]]
                    vals[ch] = r_["median"]
            word = pol.colour_name(vals.get("cie_L"), vals.get("cie_a"), vals.get("cie_b")) \
                if len(vals) == 3 else pol.lightness_term(vals.get("cie_L"))
            extra = []
            for ch, lab in (("cie_L_darkest_third", "darkest third L*"), ("cie_L_palest_third", "palest third L*"),
                            ("cie_L_end_contrast", "L* difference between the two end thirds")):
                if ch in d:
                    fid_, r_ = d[ch]
                    extra.append(f"{lab} {pol.fmt(r_['min'], 'L*')}–{pol.fmt(r_['max'], 'L*')} [{fid_}]")
                    allowed_t1 += [r_["min"], r_["max"], r_["mean"], r_["median"]]
            lines.append(f"  {info['term']} colour ≈ {word}: " + "; ".join(parts + extra)
                         + f" (n={int(rL['n'])})")
            allowed_t1.append(rL["n"])
        lines.append("")

    # comparative statements (pre-phrased; the model must quote the species lists verbatim)
    lines.append("=== COMPARATIVE STATEMENTS (non-overlapping specimen ranges, n>=2 in both species; Tier 1) ===")
    lines.append("    Quote the species list of a statement VERBATIM when you compare; do not reword or invert it.")
    comp = []
    s_all = ev.summary[(ev.summary["n"] >= 2)]
    for fid, grp in s_all.groupby("feature_id"):
        if fid not in fd.index or fd.loc[fid]["tier"] != pol.TIER_KEY:
            continue
        me = grp[grp["species"] == code]
        if me.empty:
            continue
        me = me.iloc[0]
        others = [x for x in grp["species"] if x != code]
        lower = sorted(grp[(grp["species"] != code) & (grp["max"] < me["min"])]["species"], key=natural_key)
        higher = sorted(grp[(grp["species"] != code) & (grp["min"] > me["max"])]["species"], key=natural_key)
        if len(lower) + len(higher) >= 3:
            comp.append((max(len(lower), len(higher)), fid, lower, higher, others, me))
    comp.sort(key=lambda x: (-x[0], x[1]))
    phrases = []
    for i, (n, fid, lower, higher, others, me) in enumerate(comp[:12], 1):
        meta = fd.loc[fid]
        unit = meta["unit"] if isinstance(meta["unit"], str) else ""
        rng, nums = fmt_range(me, unit)
        colour = meta["family"] == "colour"
        contrast = meta["column"] == "cie_L_end_contrast"
        up, down = (("stronger lightness contrast", "weaker lightness contrast") if contrast else
                    ("paler (higher L*)", "darker (lower L*)") if colour else ("greater", "smaller"))
        clauses = []
        for group, word in ((lower, up), (higher, down)):
            if not group:
                continue
            lst = species_list_phrase(ev, group, others)
            phrases.append(lst)
            clauses.append(f"{word} than in {lst}")
        lines.append(f"  [C{i}] [{fid}] {meta['label']} ({rng.split(' (')[0]}): " + "; ".join(clauses))
        allowed_t1 += nums + [len(others)]
    if not comp:
        lines.append("  (none — sample sizes too small or ranges overlap)")
    lines.append("")
    # key characters
    if ev.key:
        lines.append("=== CHARACTERS USED FOR THIS SPECIES IN THE IDENTIFICATION KEY (Tier 1) ===")
        for r in ev.key["couplets"]:
            for side in ("A", "B"):
                if code in r[f"{side}_species"] and (r[f"{side}_goto"] == code or len(r[f"{side}_species"]) <= 6):
                    lines.append(f"  couplet {r['number']}{side.lower()}: {r[f'{side}_text']}")
                    for c in r["characters"]:
                        allowed_t1 += [c["threshold"], *c[f"{side}_range"], c[f"{side}_n"]]
                    allowed_t1.append(r["number"])
        lines.append("")

    # tier 2
    lines.append("=== TIER 2 — STATISTICAL RESULTS (REMARKS ONLY; never in Diagnosis/Description) ===")
    t2 = tier2_lines(ev, code)
    for t in t2:
        lines.append(f"  - {t}")
        allowed_all += [float(x) for x in pol.numbers_in(t)]
    if not t2:
        lines.append("  (none)")
    lines.append("")

    # natural history (supplied only)
    lines.append("=== SUPPLIED NATURAL HISTORY (REMARKS; use ONLY these facts) ===")
    nh = natural_history(ev, code)
    for t in nh:
        lines.append(f"  - {t}")
        allowed_all += [float(x) for x in pol.numbers_in(t)]
    if not nh:
        lines.append("  (no host plant, locality or natural-history data supplied — do not mention any)")
    lines.append("")

    # prior observations
    lines.append("=== PRIOR VISUAL OBSERVATIONS (numbers removed; qualitative use only) ===")
    pv = prior_observations(ev, code)
    lines.extend(pv if pv else ["  (none available)"])
    sheet = "\n".join(lines)
    allowed_all += allowed_t1
    return sheet, {"t1": [float(x) for x in allowed_t1 if x == x],
                   "all": [float(x) for x in allowed_all if x == x],
                   "phrases": phrases,
                   "names": sorted({ev.display(x) for x in ev.species if x != code} |
                                   {short_name(ev, x) for x in ev.species if x != code}, key=len, reverse=True)}


def short_name(ev: Evidence, code: str) -> str:
    """'Diaphorina sp. 3' -> 'D. sp. 3' for species lists."""
    name = ev.display(code)
    genus = ev.profile.get("taxon", {}).get("genus", "")
    if genus and name.startswith(genus + " "):
        return f"{genus[0]}. {name[len(genus) + 1:]}"
    return name


def join_names(names: List[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def species_list_phrase(ev: Evidence, group: List[str], others: List[str]) -> str:
    """Compact, reproducible list: 'all other species measured (27) except D. sp. 3'
    when the group is the large majority, else an explicit list."""
    rest = sorted(set(others) - set(group), key=natural_key)
    if len(group) >= 6 and len(rest) <= len(group) / 2:
        if not rest:
            return f"all other species measured ({len(others)})"
        return (f"all other species measured ({len(others)}) except "
                + join_names([short_name(ev, x) for x in rest]))
    return join_names([short_name(ev, x) for x in group])


def tier2_lines(ev: Evidence, code: str) -> List[str]:
    out = []
    cats = ev.diag.get("categories", {})
    partners = Counter()
    fam = Counter()
    for cat, cd in cats.items():
        for p in cd.get("pairwise_significant", []):
            if code in (p.get("group_a"), p.get("group_b")):
                other = p["group_b"] if p.get("group_a") == code else p["group_a"]
                partners[other] += 1
                pref = p.get("feature", "").split("_")[0]
                fam[{"meas": "size measurements", "shape": "semilandmark shape",
                     "color": "colour-pattern statistics", "colhom": "colour-grid (homology) values",
                     "tex": "texture statistics", "lmk": "landmark morphometrics",
                     "imd": "distances between structures", "ratio": "proportions"}.get(pref, pref)] += 1
    if partners:
        top = ", ".join(f"{pol.species_display_name(s, ev.profile)} ({n})"
                        for s, n in sorted(partners.items(), key=lambda kv: (-kv[1], natural_key(kv[0])))[:8])
        out.append(f"Kruskal–Wallis tests with Dunn's post-hoc comparisons (FDR-corrected) found significant "
                   f"pairwise differences from: {top} [numbers = significant features]")
        out.append("Feature groups contributing most of these differences: "
                   + ", ".join(f"{k} ({v})" for k, v in fam.most_common(5)))
    if ev.full is not None:
        f = ev.full
        for col, what in (("shape_phylo_cluster", "semilandmark shape"), ("colhom_phylo_cluster", "colour-homology"),
                          ("tex_phylo_cluster", "texture")):
            if col not in f.columns:
                continue
            sub = f[f["group_label"] == code].dropna(subset=[col])
            for cat, g in sub.groupby("category"):
                if len(g) < 2:
                    continue
                k = g[col].mode().iloc[0]
                share = g[col].eq(k).mean()
                if share < 0.75:
                    continue
                others = f[(f["category"] == cat) & (f[col] == k)].groupby("group_label").size()
                others = [s for s in others.index if s != code]
                if len(others) <= 3:
                    who = ", ".join(pol.species_display_name(s, ev.profile) for s in sorted(others, key=natural_key)) or "no other species"
                    out.append(f"{pol.structure_info(cat, ev.profile)['term']}: in the {what} cluster analysis the "
                               f"specimens group together (shared with {who})")
        # semilandmark shape space neighbours (mean of the first two shape PCs)
        n_shape = 0
        for cat in sorted(f["category"].dropna().unique()):
            if n_shape >= 6 or not {"shape_PC1", "shape_PC2"} <= set(f.columns):
                break
            sub = f[f["category"] == cat].dropna(subset=["shape_PC1", "shape_PC2"])
            if code not in set(sub["group_label"]) or sub["group_label"].nunique() < 5:
                continue
            means = sub.groupby("group_label")[["shape_PC1", "shape_PC2"]].mean()
            d = np.sqrt(((means - means.loc[code]) ** 2).sum(axis=1)).drop(code).sort_values()
            near = ", ".join(pol.species_display_name(x, ev.profile) for x in d.index[:2])
            out.append(f"{pol.structure_info(cat, ev.profile)['term']}: in the semilandmark (outline) "
                       f"geometric-morphometric analysis the species mean shape is closest to {near} "
                       f"(first two principal components; {int(means.shape[0])} species compared)")
            n_shape += 1
        # landmark shape space neighbours
        for cat in ("forewing_keypoints", "head_keypoints"):
            pcs = [c for c in ("lmk_PC1", "lmk_PC2") if c in f.columns]
            sub = f[(f["category"] == cat)].dropna(subset=pcs)
            if len(pcs) < 2 or code not in set(sub["group_label"]):
                continue
            means = sub.groupby("group_label")[pcs].mean()
            if means.shape[0] < 5:          # too few species for a meaningful nearest/farthest statement
                continue
            d = np.sqrt(((means - means.loc[code]) ** 2).sum(axis=1)).drop(code).sort_values()
            if len(d) >= 3:
                near = ", ".join(pol.species_display_name(s, ev.profile) for s in d.index[:2])
                far = pol.species_display_name(d.index[-1], ev.profile)
                out.append(f"In the landmark-based geometric-morphometric analysis of the "
                           f"{pol.structure_info(cat, ev.profile)['term']}, the species mean lies closest "
                           f"to {near} and farthest from {far} (first two principal components; "
                           f"{int(means.shape[0])} species compared)")
    return out


def natural_history(ev: Evidence, code: str) -> List[str]:
    if ev.loc is None:
        return []
    sub = ev.loc[ev.loc["species"] == code]
    if sub.empty:
        return []
    out = []
    hosts = sorted(set(h for h in sub.get("host_plant", []) if h))
    if hosts:
        out.append("host plant on slide labels: " + "; ".join(hosts))
    countries = sorted(set(c for c in sub.get("country", []) if c))
    dates = sorted(set(d for d in sub.get("date", []) if d))
    colls = sorted(set(c for c in sub.get("collector", []) if c))
    meth = sorted(set(m for m in sub.get("method", []) if m))
    if countries:
        out.append("country: " + ", ".join(countries) + (f"; collected {', '.join(dates)}" if dates else "")
                   + (f"; collector {', '.join(colls)}" if colls else "") + (f"; method: {', '.join(meth)}" if meth else ""))
    # named localities and coordinates, where the collector has supplied them: one line per
    # distinct locality so a treatment can state where the material actually came from
    seen = set()
    for _, r in sub.iterrows():
        locality = str(r.get("locality", "") or "").strip()
        if not locality or locality in seen:
            continue
        seen.add(locality)
        lat, lon = str(r.get("latitude", "") or "").strip(), str(r.get("longitude", "") or "").strip()
        elev = str(r.get("elevation", "") or "").strip()
        bits = [locality]
        if lat and lon:
            bits.append(f"{lat}, {lon}")
        if elev:
            bits.append(elev)
        out.append("locality: " + ", ".join(bits))
    dets = sorted(set(x for x in sub.get("determination", []) if x))
    for x in dets:
        out.append(f"collector's determination note: {x}")
    notes = sorted(set(n for n in sub.get("discrepancy_notes", []) if n))
    for n in notes[:3]:
        out.append(f"label note: {n}")
    return out


_BAD_PRIOR = re.compile(r'(holotype|allotype|paratype|type series|statistic|significan|cluster|'
                        r'principal comp|\bPC\d|UMAP|Procrustes|p\s*[=<]|Dunn|Kruskal|diagnostic feature|'
                        r'meas_|colhom|tex_|lmk_|imd_|no image|data only|pairwise|entropy|GLCM|LBP)', re.I)


def prior_observations(ev: Evidence, code: str, max_chars: int = 14000) -> List[str]:
    if not ev.prior_cache:
        return []
    d = ev.prior_cache / code
    if not d.is_dir():
        return []
    out, total = [], 0
    for fp in sorted(d.glob("*_foreground.json"), key=lambda p: natural_key(p.name)):
        cat = fp.name[: -len("_foreground.json")]
        try:
            j = json.loads(fp.read_text())
        except Exception:  # noqa: BLE001
            continue
        eo = j.get("expected_output", j)
        texts = [str(eo.get("description", "")), str(eo.get("diagnosis", ""))]
        for t in eo.get("traits", []) or []:
            if isinstance(t, dict) and str(t.get("evidenceSource", "")).startswith("image"):
                texts.append(f"{t.get('traitType', '')}: {t.get('value', '')}")
        keep = []
        for txt in texts:
            for sent in re.split(r'(?<=[.;])\s+', txt):
                s = sent.strip()
                if len(s) < 15 or re.search(r'\d', s) or _BAD_PRIOR.search(s):
                    continue
                s = re.sub(rf'\b{re.escape(code)}\b', 'this species', s)
                if s not in keep:
                    keep.append(s)
        if keep:
            block = f"[{pol.structure_info(cat, ev.profile)['term']}] " + " ".join(keep)
            if total + len(block) > max_chars:
                block = block[: max(0, max_chars - total)]
            out.append("  " + block)
            total += len(block)
        if total >= max_chars:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(treat: Dict, allowed: Dict, codes: List[str], sections: List[str]) -> List[str]:
    probs = []
    t1_fields = [("diagnosis", treat.get("diagnosis", "")),
                 ("sexual_dimorphism", treat.get("sexual_dimorphism", ""))]
    for d in treat.get("description", []) or []:
        t1_fields.append((f"description/{d.get('section', '?')}", d.get("text", "")))
        if sections and d.get("section") not in sections:
            probs.append(f"unknown description section '{d.get('section')}'")
    for name, text in t1_fields:
        for n in pol.numbers_in(text):
            if not pol.number_is_allowed(n, allowed["t1"]):
                probs.append(f"{name}: number {n} is not a Tier-1 value of the data sheet")
        for term in pol.find_tier2_terms(text):
            probs.append(f"{name}: statistical term '{term}' belongs in REMARKS")
    diag_words = len((treat.get("diagnosis") or "").split())
    if diag_words > 230:
        probs.append(f"diagnosis has {diag_words} words; shorten it to at most 180 words")
    for name, text in t1_fields:
        for sent in re.split(r'(?<=[.;])\s+', text or ""):
            mentioned = [nm for nm in allowed.get("names", []) if nm in sent]
            if mentioned and not any(ph in sent for ph in allowed.get("phrases", [])):
                probs.append(f"{name}: sentence compares with other species without quoting a COMPARATIVE "
                             f"STATEMENT list verbatim: '{sent[:120]}...'")
    rem = treat.get("remarks", "")
    for n in pol.numbers_in(rem):
        if not pol.number_is_allowed(n, allowed["all"]):
            probs.append(f"remarks: number {n} is not in the data sheet")
    all_text = " ".join(t for _, t in t1_fields) + " " + rem
    for c in codes:
        if re.search(rf'(?<![\w.\']){re.escape(c)}(?![\w\'])', all_text):
            probs.append(f"folder code '{c}' used instead of the species name")
    return list(dict.fromkeys(probs))


FEATURE_TAG_RX = re.compile(r'\s*\[[A-Za-z_+\-0-9]+\.[\w/()+\-]+\]')


def independent_problems(code: str, treat: Dict, sheet: str, cev) -> List[str]:
    """Problems found by the independent audit (biorag_confabulation_checker_v2)."""
    if cev is None or not treat:
        return []
    import biorag_confabulation_checker_v2 as cchk
    res = cchk.audit_species(code, treat, sheet, cev)
    return [f"independent check {r}" for r in res.get("repair_requests", [])
            if "internal_identifier_in_text" not in r]


def clean_treatment(treat: Dict, cev=None) -> Dict:
    """Deterministic formatting clean-up: bracketed data-sheet IDs out of the
    text (they stay in 'citations'); citation IDs mapped to canonical names."""
    n = 0

    def strip(t):
        nonlocal n
        new, k = FEATURE_TAG_RX.subn("", t or "")
        n += k
        return new
    for key in ("diagnosis", "sexual_dimorphism", "remarks"):
        treat[key] = strip(treat.get(key, ""))
    for d in treat.get("description", []) or []:
        d["text"] = strip(d.get("text", ""))
    if cev is not None:
        for c in treat.get("citations", []) or []:
            c["id"] = cev.canon_fid(str(c.get("id", "")))
    return {"feature_ids_removed_from_text": n}


def strip_bad_sentences(treat: Dict, allowed: Dict, codes: List[str]) -> List[str]:
    removed = []

    def clean(text, pool, t1=True):
        keep = []
        for s in re.split(r'(?<=[.!?])\s+', text or ""):
            bad = any(not pol.number_is_allowed(n, pool) for n in pol.numbers_in(s))
            if t1 and pol.find_tier2_terms(s):
                bad = True
            if bad:
                removed.append(s)
            else:
                keep.append(s)
        return " ".join(keep)
    treat["diagnosis"] = clean(treat.get("diagnosis", ""), allowed["t1"])
    treat["sexual_dimorphism"] = clean(treat.get("sexual_dimorphism", ""), allowed["t1"])
    for d in treat.get("description", []) or []:
        d["text"] = clean(d.get("text", ""), allowed["t1"])
    treat["remarks"] = clean(treat.get("remarks", ""), allowed["all"], t1=False)
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def write_text(path: Path, ev: Evidence, code: str, treat: Dict, meta: Dict):
    name = ev.display(code)
    status = pol.species_status(code, ev.profile)
    L = [name, "=" * len(name),
         f"(folder code: {code}; status: {status}; generated {meta['generated']} by "
         f"biorag_description_refiner_v1 via {meta['backend']})", ""]
    if ev.key:
        for r in ev.key["couplets"]:
            for side in ("A", "B"):
                if r[f"{side}_goto"] == code:
                    L.append(f"Key: couplet {r['number']}{side.lower()}")
                    L.append("")
    L += ["DIAGNOSIS", treat.get("diagnosis", "").strip(), "", "DESCRIPTION"]
    for d in treat.get("description", []) or []:
        if d.get("text", "").strip():
            L.append(f"{d.get('section', '')}. {d['text'].strip()}")
    L.append("")
    if treat.get("sexual_dimorphism", "").strip():
        L += ["SEXUAL DIMORPHISM", treat["sexual_dimorphism"].strip(), ""]
    L += ["REMARKS", treat.get("remarks", "").strip() or "—", ""]
    if treat.get("not_assessable"):
        L += ["NOT ASSESSABLE FROM THE MATERIAL", "; ".join(treat["not_assessable"]), ""]
    L += ["PROVENANCE",
          f"Data matrix: {meta['matrix_dir']}",
          f"Numbers cited: {len(treat.get('citations', []))}; validation problems after repair: "
          f"{len(treat.get('_validation', {}).get('final_problems', []))}; "
          f"sentences removed: {len(treat.get('_validation', {}).get('removed_sentences', []))}"]
    path.write_text("\n".join(L), encoding="utf-8")


def codes_to_check(profile: Dict) -> List[str]:
    """Folder codes that must not appear in text (codes identical to a word of
    their own display name, e.g. 'zebrana', are legitimate epithets)."""
    out = []
    for c in profile.get("species", {}):
        name = pol.species_display_name(c, profile)
        if not re.search(rf'(?<![\w.]){re.escape(c)}(?![\w])', name):
            out.append(c)
    return out


def refine_one(code, ev, client, prompts, args, out_dir, log, cev=None):
    sp_dir = out_dir / code
    sp_dir.mkdir(parents=True, exist_ok=True)
    tj = sp_dir / f"{code}_treatment.json"
    previous = None
    adding = bool(getattr(args, "add_sections", None))
    if tj.exists() and (args.repair_existing or adding):
        previous = json.loads(tj.read_text())
        tag = "before_added_sections" if adding else "before_repair"
        bk = sp_dir / f"{code}_treatment_{tag}_{datetime.now():%Y%m%d}.json"
        if not bk.exists():
            bk.write_text(tj.read_text(), encoding="utf-8")
    elif tj.exists() and not args.force:
        log(f"  {code}: cached")
        return json.loads(tj.read_text())
    sheet, allowed = build_sheet(ev, code)
    (sp_dir / f"{code}_data_sheet.txt").write_text(sheet, encoding="utf-8")
    sections = ev.profile.get("description_sections") or []
    ctx = pol.build_taxon_context(ev.profile)
    system = prompts.get("taxonomist_persona") + "\n\n" + prompts.get("refine_treatment", taxon_context=ctx)
    questions = ""
    qp = ev.profile.get("_questions_path")
    if qp and Path(qp).exists():
        questions = ("\n\n=== TAXONOMIST QUESTIONS (answer those the sheet or observations allow; "
                     "proportions go in the 'Proportions' section) ===\n" + Path(qp).read_text())
    user = f"Write the treatment for {ev.display(code)}.\n\n{sheet}{questions}"
    add_only = [x for x in (args.add_sections or []) if x]
    if add_only:
        # only the named sections are written; the rest of the treatment is carried over untouched
        user = (f"Write ONLY the following sections of the description for {ev.display(code)}: "
                f"{', '.join(add_only)}. Return the same JSON structure, with 'description' "
                f"containing only those sections and no other field populated. Every number must "
                f"come from the data sheet exactly as it appears there.\n\n{sheet}{questions}")
    msgs = [{"role": "user", "content": user}]
    history = []
    treat, probs = {}, ["not run"]
    t0 = time.time()
    repair = prompts.get("refine_treatment_repair")

    def check(t):
        tt = json.loads(json.dumps(t, ensure_ascii=False))
        clean_treatment(tt, cev)
        return validate(tt, allowed, codes_to_check(ev.profile), sections) + \
            independent_problems(code, tt, sheet, cev)
    start = 1
    if add_only and previous is None:
        log(f"  {code}: --add_sections needs an existing treatment to merge into; skipped")
        return {}
    if previous is not None and not add_only:
        treat = {k: v for k, v in previous.items() if not k.startswith("_")}
        probs = check(treat)
        history = list(previous.get("_validation", {}).get("attempts", []))
        history.append({"attempt": "existing (audit)", "problems": probs})
        if not probs:
            start = args.max_repairs + 2          # nothing to repair
        else:
            msgs = [{"role": "user", "content": user},
                    {"role": "assistant", "content": json.dumps(treat, ensure_ascii=False)},
                    {"role": "user", "content": repair + "\n\nPROBLEMS:\n- " + "\n- ".join(probs)}]
    for attempt in range(start, args.max_repairs + 2):
        try:
            r = client.messages.create(model=args.model, max_tokens=12000, system=system, messages=msgs)
            new = parse_json_response(r.content[0].text)
        except Exception as e:  # noqa: BLE001
            history.append({"attempt": attempt, "problems": [f"LLM/JSON error: {e}"]})
            continue
        treat = new
        probs = check(treat)
        history.append({"attempt": attempt, "problems": probs})
        if not probs:
            break
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(treat, ensure_ascii=False)},
                {"role": "user", "content": repair + "\n\nPROBLEMS:\n- " + "\n- ".join(probs)}]
    if add_only and treat:
        # merge: the new sections replace or append, everything else is the previous text verbatim
        merged = {k: v for k, v in previous.items() if not k.startswith("_")}
        old_secs = {b.get("section"): b for b in (merged.get("description") or [])}
        stamp = args.added_by or ""
        for b in (treat.get("description") or []):
            sec = b.get("section")
            if sec not in add_only:
                continue
            if stamp:
                b["added_by"] = stamp
                b["added"] = datetime.now().strftime("%Y-%m-%d")
            old_secs[sec] = b
        order = {s: i for i, s in enumerate(sections)}
        merged["description"] = sorted(old_secs.values(),
                                       key=lambda b: order.get(b.get("section"), 99))
        merged["citations"] = (merged.get("citations") or []) + \
            [c for c in (treat.get("citations") or []) if c not in (merged.get("citations") or [])]
        treat = merged
        probs = check(treat)
        history.append({"attempt": "merged into the existing treatment", "problems": probs})
    cleaned = clean_treatment(treat, cev) if treat else {}
    removed = strip_bad_sentences(treat, allowed, codes_to_check(ev.profile)) if probs and treat else []
    final = (validate(treat, allowed, codes_to_check(ev.profile), sections) +
             independent_problems(code, treat, sheet, cev)) if treat else probs
    treat["_validation"] = {"attempts": history, "final_problems": final, "removed_sentences": removed,
                            "cleanup": cleaned, "independent_check": cev is not None,
                            "repaired_existing": previous is not None and not add_only,
                            "added_sections": add_only or None,
                            "added_by": args.added_by if add_only else None,
                            "seconds": round(time.time() - t0, 1)}
    prev_meta = (previous or {}).get("_meta", {})
    treat["_meta"] = {"species_code": code, "display_name": ev.display(code),
                      "status": pol.species_status(code, ev.profile),
                      "refiner_version": REFINER_VERSION, "policy_version": pol.POLICY_VERSION,
                      "model": args.model, "backend": args.llm_backend,
                      # "model" is what was asked for; the call log records what answered
                      "model_that_answered": answering_model(args.llm_log or str(out_dir / "llm_calls.jsonl")),
                      "generated": datetime.now().isoformat(),
                      "prompts_file": str(getattr(prompts, "path", "")),
                      "first_generated": prev_meta.get("first_generated", prev_meta.get("generated", "")) or None,
                      "first_refiner_version": prev_meta.get("first_refiner_version",
                                                             prev_meta.get("refiner_version")) or None}
    tj.write_text(json.dumps(treat, indent=1, ensure_ascii=False), encoding="utf-8")
    write_text(sp_dir / f"{code}.txt", ev, code, treat,
               {"generated": treat["_meta"]["generated"], "backend": args.llm_backend,
                "matrix_dir": args.matrix_dir})
    log(f"  {code}: attempts={len(history)} final_problems={len(final)} removed={len(removed)} "
        f"({treat['_validation']['seconds']}s)")
    return treat


def main():
    ap = argparse.ArgumentParser(description="Evidence-tiered species treatments (BioRAG v2)")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--compiled_dir", default=None, help="Original compiled data (Tier-2 statistics)")
    ap.add_argument("--prior_cache", default=None, help="biorag_cache/towley_cache with image observations")
    ap.add_argument("--key_tree", default=None)
    ap.add_argument("--localities", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max_repairs", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="Regenerate species that already have output")
    ap.add_argument("--sheets_only", action="store_true", help="Write data sheets without calling a model")
    ap.add_argument("--add_sections", nargs="*", default=None,
                    help="Describe ONLY these sections and merge them into the existing treatment, "
                         "leaving every other sentence byte-identical. This is how a later worker "
                         "adds a structure the original description left out (mouthparts, say) "
                         "without regenerating — and therefore without re-auditing — text that has "
                         "already been checked. Annotate the structure, add it to the profile's "
                         "structures and description_sections, rebuild the matrix, then run this.")
    ap.add_argument("--added_by", default=None,
                    help="Who is adding those sections, e.g. 'Smith, 2031'. Recorded per section so "
                         "a reader can tell the original description from what was added later")
    ap.add_argument("--repair_existing", action="store_true",
                    help="Audit existing treatments and send only the flagged problems back to the model "
                         "(the previous JSON is kept as <code>_treatment_before_repair_<date>.json)")
    ap.add_argument("--no_independent_check", action="store_true",
                    help="Validate with the data-sheet check only (refiner 1.0 behaviour)")
    add_backend_args(ap)
    args = ap.parse_args()

    profile = pol.load_taxon_profile(args.taxon_profile)
    ev = Evidence(args, profile)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "refiner.log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    species = [s for s in ev.species if not args.species or s in args.species]
    log(f"[{datetime.now().isoformat()}] refining {len(species)} species with {args.llm_backend}")
    if args.sheets_only or args.llm_backend == "none":
        for code in species:
            (out / code).mkdir(exist_ok=True)
            sheet, _ = build_sheet(ev, code)
            (out / code / f"{code}_data_sheet.txt").write_text(sheet, encoding="utf-8")
        log("data sheets written")
        return
    cev = None
    if not args.no_independent_check:
        import biorag_confabulation_checker_v2 as cchk
        cev = cchk.Evidence(argparse.Namespace(taxon_profile=args.taxon_profile, matrix_dir=args.matrix_dir,
                                               key_tree=args.key_tree, localities=args.localities))
    prompts = load_prompt_library(args.system_prompts)
    client = make_llm_client(args.llm_backend, claude_bin=args.claude_bin, cc_model=args.cc_model,
                             log_path=args.llm_log or str(out / "llm_calls.jsonl"))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = dict(zip(species, ex.map(lambda c: refine_one(c, ev, client, prompts, args, out, log, cev),
                                           species)))
    rows = []
    for code, t in results.items():
        v = t.get("_validation", {})
        rows.append({"species_code": code, "display_name": ev.display(code),
                     "attempts": len(v.get("attempts", [])),
                     "problems_first_attempt": len((v.get("attempts") or [{}])[0].get("problems", [])),
                     "final_problems": len(v.get("final_problems", [])),
                     "repaired_existing": bool(v.get("repaired_existing")),
                     "feature_ids_removed": (v.get("cleanup") or {}).get("feature_ids_removed_from_text", 0),
                     "sentences_removed": len(v.get("removed_sentences", [])),
                     "numbers_cited": len(t.get("citations", [])),
                     "diagnosis_words": len(t.get("diagnosis", "").split()),
                     "remarks_words": len(t.get("remarks", "").split())})
    pd.DataFrame(rows).to_csv(out / "refine_validation_summary.tsv", sep="\t", index=False)
    log(pd.DataFrame(rows).to_string(index=False))
    logf.close()


if __name__ == "__main__":
    main()
