#!/usr/bin/env python3
"""
biorag_ontology_annotator_v2.py — EQ ontology annotation of BioRAG v2 treatments
===============================================================================

v2 of biorag_ontology_assigner.py (v1 kept unchanged; it fills a URI field in
the v1 trait JSON-LD from a hand-written lookup table). v2 differs in three
ways:

  1. Every URI is resolved from an OBO release downloaded to --ontology_dir
     (pato.obo, aism.obo, ...). Nothing is hard-coded, and the release version
     of each ontology is recorded, so a reviewer can repeat the lookup.
  2. Annotations are entity-quality (EQ) pairs, as used in phenotype
     ontologies: the ENTITY is the annotated structure (AISM/UBERON) and the
     QUALITY is what was measured or observed (PATO). v1 recorded only a
     quality.
  3. Both measured claims (from the treatment's citation list, via the feature
     dictionary) and subjective image-read words (from
     biorag_subjective_checks_v1.py) are annotated, and coverage is reported
     separately for the two, with exact / synonym / broader matches counted
     apart.

Taxon-agnostic: structure terms come from the taxon profile. A structure that
has no term of its own in the ontologies (e.g. the psyllid circumanal ring)
gets `ontology_broader: <CURIE>` in the profile; the script verifies that the
CURIE exists in the release and marks those annotations "broader".

Usage:
  python biorag_ontology_annotator_v2.py \
     --descriptions_dir ".../Diaphorina_monograph/descriptions" \
     --matrix_dir ".../Diaphorina_monograph/compiled_key_tier" \
     --taxon_profile biorag_prompts/taxon_profiles/diaphorina_taxon_profile.yaml \
     [--ontology_dir biorag_prompts/ontologies] [--download]
"""

import argparse
import csv
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import biorag_feature_policy as pol  # noqa: E402

ANNOTATOR_VERSION = "2.0"
# Namespaces that may supply an ENTITY (anatomy). Quality namespaces (PATO) and
# imported non-anatomical namespaces (GO, CL, CHEBI, NCBITaxon) are excluded.
ENTITY_PREFIXES = ("AISM", "UBERON", "HAO", "FBbt", "TGMA", "SPD", "COLAO", "MA", "EMAPA")
DEFAULT_ONTOLOGIES = ["pato", "aism"]
OBO_URL = "http://purl.obolibrary.org/obo/{}.obo"

# Column -> PATO term LABEL (resolved to a CURIE in the release; taxon-agnostic)
QUALITY_LABELS = [
    (r'^meas_length_mm$|^length_mm$', "length"),
    (r'^meas_height_mm$|^height_mm$', "width"),
    (r'^meas_area_mm2$|^area_mm2$', "area"),
    (r'^meas_perimeter_mm$|^perimeter_mm$', "perimeter"),
    (r'^meas_aspect_ratio$|^aspect_ratio$|^ratio_|^lmkrel_', "ratio"),
    (r'^lmkmm_|^imd_', "distance"),
    (r'^cie_L', "color brightness"),
    (r'^cie_C$', "color saturation"),
    (r'^cie_h$', "color hue"),
    (r'^cie_[ab]$', "color"),
    (r'^shape_|^lmk_PC', "shape"),
    (r'^tex_', "texture"),
    (r'^color_', "color"),
    (r'^scale_', "size"),
]
# Subjective character kinds -> PATO term LABEL used when the word itself is
# not a PATO term (the word is still recorded verbatim).
KIND_FALLBACK = {"colour": "color", "texture": "texture", "sculpture": "texture",
                 "shape": "shape", "setation": "setose", "luster": "texture"}


# ─────────────────────────────────────────────────────────────────────────────
# OBO
# ─────────────────────────────────────────────────────────────────────────────

def parse_obo(path: Path) -> Tuple[Dict[str, Dict], str]:
    terms, cur, version = {}, None, ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("data-version:"):
                version = line.split(":", 1)[1].strip()
            if line.startswith("[") :
                if cur and cur.get("id") and not cur.get("obsolete"):
                    terms[cur["id"]] = cur
                cur = {"syn": [], "is_a": []} if line == "[Term]" else None
                continue
            if cur is None or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            if k == "id":
                cur["id"] = v
            elif k == "name":
                cur["name"] = re.sub(r'\s*\{.*\}\s*$', '', v)
            elif k == "synonym":
                m = re.match(r'"(.*?)"\s+(\w+)', v)
                if m:
                    cur["syn"].append((re.sub(r'\s*\{.*\}\s*$', '', m.group(1)), m.group(2)))
            elif k == "is_obsolete" and v == "true":
                cur["obsolete"] = True
            elif k == "is_a":
                cur["is_a"].append(v.split("!")[0].split("{")[0].strip())
    if cur and cur.get("id") and not cur.get("obsolete"):
        terms[cur["id"]] = cur
    return terms, version


class Ontologies:
    def __init__(self, directory: Path, names: List[str], download: bool = False):
        self.dir = directory
        self.terms: Dict[str, Dict] = {}
        self.versions: Dict[str, str] = {}
        for name in names:
            p = directory / f"{name}.obo"
            if not p.exists() and download:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"downloading {OBO_URL.format(name)} ...")
                urllib.request.urlretrieve(OBO_URL.format(name), p)  # noqa: S310
            if not p.exists():
                raise FileNotFoundError(f"{p} not found (run with --download)")
            t, v = parse_obo(p)
            self.versions[name] = v or "unknown"
            self.terms.update(t)
        self.by_label: Dict[str, str] = {}
        self.by_syn: Dict[str, Tuple[str, str]] = {}
        for cid, t in self.terms.items():
            lab = t.get("name", "").lower()
            if lab:
                self.by_label.setdefault(lab, cid)
            for syn, scope in t["syn"]:
                self.by_syn.setdefault(syn.lower(), (cid, scope))

    def label(self, curie: str) -> Optional[str]:
        t = self.terms.get(curie)
        return t.get("name") if t else None

    def lookup(self, text: str) -> Optional[Tuple[str, str, str]]:
        """(curie, label, match) for a term label or synonym; None if unknown."""
        if not text:
            return None
        s = re.sub(r'\s*\(.*?\)', '', str(text)).strip().lower()
        for cand in (s, s.rstrip("s"), s.replace("-", " ")):
            if cand in self.by_label:
                cid = self.by_label[cand]
                return cid, self.label(cid), "exact"
            if cand in self.by_syn:
                cid, scope = self.by_syn[cand]
                return cid, self.label(cid), f"{scope.lower()}_synonym"
        return None

    @staticmethod
    def iri(curie: str) -> str:
        return "http://purl.obolibrary.org/obo/" + curie.replace(":", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Mapping
# ─────────────────────────────────────────────────────────────────────────────

def entity_for(cat: str, profile: Dict, onto: Ontologies, report: Dict) -> Optional[Dict]:
    """Ontology entity for one annotation category."""
    base = re.sub(r'__(?:male|female)$', '', cat)
    info = (profile.get("structures") or {}).get(base) or (profile.get("structures") or {}).get(cat) or {}
    if base in (profile.get("landmark_sets") or {}):
        info = (profile.get("landmark_sets") or {}).get(base, info)
    for key, match in (("ontology", "declared"), ("ontology_broader", "broader")):
        curie = info.get(key)
        if curie:
            lab = onto.label(curie)
            if lab is None:
                report.setdefault("invalid_curies", []).append(f"{cat}: {curie} not in the releases")
                continue
            return {"curie": curie, "label": lab, "match": match, "source": f"taxon profile ({key})"}
    prefixes = tuple(profile.get("ontology_entity_prefixes") or ENTITY_PREFIXES)
    for text in (info.get("term"), info.get("short"), *(info.get("synonyms") or []), base.replace("_", " ")):
        hit = onto.lookup(text)
        if hit and hit[0].split(":")[0] in prefixes:
            return {"curie": hit[0], "label": hit[1], "match": hit[2], "source": f"label lookup: '{text}'"}
    return None


def quality_for(column: str, onto: Ontologies, report: Dict) -> Optional[Dict]:
    for rx, label in QUALITY_LABELS:
        if re.search(rx, column):
            hit = onto.lookup(label)
            if hit:
                return {"curie": hit[0], "label": hit[1], "match": hit[2], "source": f"column rule: '{label}'"}
            report.setdefault("missing_quality_labels", []).append(label)
            return None
    return None


def ratio_entities(rid: str, profile: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Numerator and denominator categories of a standard proportion."""
    abbr = profile.get("measurement_abbreviations") or {}
    def cat_of(key):
        key = str(key)
        if key in abbr:
            return abbr[key].get("category")
        parts = [p for p in re.split(r'[+&]', key) if p in abbr]      # combined measurement, e.g. "LAB2+LAB3"
        return abbr[parts[0]].get("category") if parts else None
    for r in profile.get("ratios") or []:
        if str(r.get("id")) == rid:
            return cat_of(r.get("num")), cat_of(r.get("den"))
    return None, None


def annotate_species(code: str, treat: Dict, fdict: pd.DataFrame, profile: Dict, onto: Ontologies,
                     ent_cache: Dict, report: Dict, subjective: pd.DataFrame) -> Dict:
    rows = []
    for c in treat.get("citations", []) or []:
        fid = str(c.get("id", ""))
        if re.fullmatch(r'\[?C\d+\.?\]?', fid.strip()):
            report["citations_to_comparative_statements"] = \
                report.get("citations_to_comparative_statements", 0) + 1
            continue
        cat, _, col = fid.partition(".")
        meta = fdict.loc[fid] if fid in fdict.index else None
        rel = None
        if col.startswith("ratio_"):
            num, den = ratio_entities(col[len("ratio_"):], profile)
            if num:
                cat = num
            if den and den not in ent_cache:
                ent_cache[den] = entity_for(den, profile, onto, report)
            rel = ent_cache.get(den) if den else None
        if cat not in ent_cache:
            ent_cache[cat] = entity_for(cat, profile, onto, report)
        ent = ent_cache[cat]
        qual = quality_for(col, onto, report)
        rows.append({"species": code, "kind": "measured", "value": c.get("value", ""), "feature_id": fid,
                     "structure": cat, "column": col,
                     "label": (meta["label"] if meta is not None else ""),
                     "unit": (meta["unit"] if meta is not None else ""),
                     "entity_curie": ent["curie"] if ent else "", "entity_label": ent["label"] if ent else "",
                     "entity_match": ent["match"] if ent else "none",
                     "quality_curie": qual["curie"] if qual else "", "quality_label": qual["label"] if qual else "",
                     "quality_match": qual["match"] if qual else "none",
                     "relative_to_curie": rel["curie"] if rel else "",
                     "relative_to_label": rel["label"] if rel else ""})
    if subjective is not None and len(subjective):
        sub = subjective[subjective["species"] == code]
        expanded = []
        for _, r in sub.iterrows():
            for w in str(r.get("words", "") or "").split(","):
                if w.strip():
                    expanded.append({"structure": r.get("structure", ""), "kind": r.get("kind", ""),
                                     "word": w.strip(), "field": r.get("field", "")})
        for r in expanded:
            cat = str(r.get("structure", "") or "")
            if cat and cat not in ent_cache:
                ent_cache[cat] = entity_for(cat, profile, onto, report)
            ent = ent_cache.get(cat)
            word = str(r.get("word", "") or "")
            hit = onto.lookup(word)
            if hit is None:
                fb = KIND_FALLBACK.get(str(r.get("kind", "")), None)
                h2 = onto.lookup(fb) if fb else None
                qual = ({"curie": h2[0], "label": h2[1], "match": "broader"} if h2 else None)
            else:
                qual = {"curie": hit[0], "label": hit[1], "match": hit[2]}
            rows.append({"species": code, "kind": "subjective", "value": word, "feature_id": "",
                         "structure": cat, "column": str(r.get("kind", "")), "label": word, "unit": "",
                         "entity_curie": ent["curie"] if ent else "", "entity_label": ent["label"] if ent else "",
                         "entity_match": ent["match"] if ent else "none",
                         "quality_curie": qual["curie"] if qual else "", "quality_label": qual["label"] if qual else "",
                         "quality_match": qual["match"] if qual else "none",
                         "relative_to_curie": "", "relative_to_label": ""})
    return rows


def main():
    ap = argparse.ArgumentParser(description="EQ ontology annotation of BioRAG v2 treatments")
    ap.add_argument("--descriptions_dir", required=True)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--ontology_dir", default=str(Path(__file__).parent / "biorag_prompts" / "ontologies"))
    ap.add_argument("--ontologies", nargs="*", default=None,
                    help=f"OBO names to load (default: taxon profile 'ontologies' or {DEFAULT_ONTOLOGIES})")
    ap.add_argument("--download", action="store_true", help="Download missing OBO releases")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    profile = pol.load_taxon_profile(args.taxon_profile)
    names = args.ontologies or profile.get("ontologies") or DEFAULT_ONTOLOGIES
    onto = Ontologies(Path(args.ontology_dir), names, args.download)
    fdict = pd.read_csv(Path(args.matrix_dir) / "feature_dictionary.tsv", sep="\t").set_index("feature_id")
    dd = Path(args.descriptions_dir)
    out = Path(args.output_dir) if args.output_dir else dd
    sub_path = dd / "subjective_character_flags.tsv"
    subjective = pd.read_csv(sub_path, sep="\t").fillna("") if sub_path.exists() else None
    report = {"annotator_version": ANNOTATOR_VERSION, "generated": datetime.now().isoformat(),
              "ontologies": {n: onto.versions[n] for n in names}, "taxon_profile": profile.get("_path")}
    ent_cache: Dict[str, Optional[Dict]] = {}
    all_rows = []
    for tj in sorted(dd.glob("*/*_treatment.json")):
        code = tj.parent.name
        treat = json.loads(tj.read_text())
        rows = annotate_species(code, treat, fdict, profile, onto, ent_cache, report, subjective)
        all_rows += rows
        jl = {"@context": {"@vocab": "https://schema.org/", "dsc": "https://descriptron.org/ontology/",
                           "obo": "http://purl.obolibrary.org/obo/"},
              "@type": "dsc:OntologyAnnotationSet", "about": {"@type": "Taxon", "identifier": code,
                                                              "name": treat.get("_meta", {}).get("display_name", code)},
              "dsc:ontologies": report["ontologies"], "dateCreated": report["generated"],
              "dsc:annotations": [
                  {"dsc:kind": r["kind"], "dsc:value": r["value"], "dsc:featureId": r["feature_id"],
                   "dsc:label": r["label"], "dsc:unit": r["unit"],
                   "dsc:entity": ({"@id": Ontologies.iri(r["entity_curie"]), "dsc:curie": r["entity_curie"],
                                   "name": r["entity_label"], "dsc:match": r["entity_match"]}
                                  if r["entity_curie"] else None),
                   "dsc:quality": ({"@id": Ontologies.iri(r["quality_curie"]), "dsc:curie": r["quality_curie"],
                                    "name": r["quality_label"], "dsc:match": r["quality_match"]}
                                   if r["quality_curie"] else None),
                   "dsc:relativeTo": ({"@id": Ontologies.iri(r["relative_to_curie"]),
                                       "dsc:curie": r["relative_to_curie"], "name": r["relative_to_label"]}
                                      if r.get("relative_to_curie") else None)}
                  for r in rows]}
        (tj.parent / f"{code}_ontology.jsonld").write_text(json.dumps(jl, indent=1, ensure_ascii=False),
                                                           encoding="utf-8")
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "ontology_annotations.tsv", sep="\t", index=False)
    # structure -> term table (supplement)
    st_rows = [{"category": cat, "curie": (e or {}).get("curie", ""), "term": (e or {}).get("label", ""),
                "match": (e or {}).get("match", "none"), "source": (e or {}).get("source", "")}
               for cat, e in sorted(ent_cache.items())]
    pd.DataFrame(st_rows).to_csv(out / "ontology_structure_terms.tsv", sep="\t", index=False)

    def cov(sub: pd.DataFrame, col: str) -> Dict:
        if not len(sub):
            return {}
        m = sub[f"{col}_match"].value_counts().to_dict()
        n_ok = int((sub[f"{col}_curie"] != "").sum())
        return {"n": len(sub), "annotated": n_ok, "coverage_percent": round(100 * n_ok / len(sub), 1),
                "by_match": m}
    meas, subj = df[df["kind"] == "measured"], df[df["kind"] == "subjective"]
    both = int(((df["entity_curie"] != "") & (df["quality_curie"] != "")).sum())
    report["coverage"] = {
        "claims_total": len(df), "eq_pairs_complete": both,
        "eq_coverage_percent": round(100 * both / max(1, len(df)), 1),
        "measured": {"entity": cov(meas, "entity"), "quality": cov(meas, "quality")},
        "subjective": {"entity": cov(subj, "entity"), "quality": cov(subj, "quality")},
        "structures": {"n": len(st_rows), "unmapped": [r["category"] for r in st_rows if not r["curie"]],
                       "by_match": dict(Counter(r["match"] for r in st_rows))},
        "unmapped_subjective_words": sorted({r["value"] for _, r in subj.iterrows() if not r["quality_curie"]})[:50],
    }
    (out / "ontology_coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"ontologies": report["ontologies"], **report["coverage"]}, indent=1)[:2000])
    print(f"annotations -> {out / 'ontology_annotations.tsv'}")


if __name__ == "__main__":
    main()
