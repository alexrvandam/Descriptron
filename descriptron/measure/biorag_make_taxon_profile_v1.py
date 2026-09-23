#!/usr/bin/env python3
"""
biorag_make_taxon_profile_v1.py — draft a taxon profile for BioRAG v2
=====================================================================

Writes a starter taxon profile YAML from the inputs every Descriptron project
already has, so the BioRAG v2 workflow runs on any taxon out of the box:

  * species codes      <- group_labels CSV (filename, group_label)
  * structures         <- COCO categories with annotations
  * landmark sets      <- COCO keypoint categories
  * sex regex          <- default _m_/_f_ convention
  * specimen number    <- inferred from file names (number before a structure word),
                          checked against the group labels
  * questions file     <- --user_prompts (optional)

Everything it cannot know is marked `verify: true` or left as a TODO comment:
display names (defaults to "<Genus> <code>"), species status, sex-specific
structures, description sections, measurement abbreviations and standard
ratios. Edit the file, then rerun the pipeline with --taxon_profile.

Usage:
  python biorag_make_taxon_profile_v1.py --coco_json all.json \
      --group_labels group_labels.csv --genus Tetramorium --family Formicidae \
      --output tetramorium_taxon_profile.yaml [--user_prompts questions.txt]
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import yaml

GENERIC_SECTIONS = [
    (r"head|vertex|gena|clypeus|frons|eye|ocell|mandib|antenn|scape", "Head"),
    (r"rostrum|lab(ium|ial)?\d*|proboscis|mouth", "Mouthparts"),
    (r"thorax|mesosoma|pronot|mesonot|propod|scutel|petiol", "Thorax"),
    (r"wing|cell|vein|pterostigma|tegmen|elytr", "Wings"),
    (r"leg|femur|femura|tibia|tars|coxa", "Legs"),
    (r"gaster|abdom|tergit|sternit|metasom", "Abdomen"),
    (r"genital|aedeag|paramer|proctiger|subgenital|circumanal|ovipositor|terminal", "Terminalia"),
]
SEX_HINTS = {"male": r"^male_|paramere|aedeag|phallus|genital_capsule",
             "female": r"^female_|ovipositor|spermath|circumanal"}


def guess_section(cat):
    c = cat.lower()
    for rx, sec in GENERIC_SECTIONS:
        if re.search(rx, c):
            return sec
    return "Other structures"


def main():
    ap = argparse.ArgumentParser(description="Draft a BioRAG v2 taxon profile")
    ap.add_argument("--coco_json", required=True)
    ap.add_argument("--group_labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--genus", default="")
    ap.add_argument("--family", default="")
    ap.add_argument("--higher_classification", default="")
    ap.add_argument("--keypoints_json", default=None)
    ap.add_argument("--user_prompts", default=None)
    args = ap.parse_args()

    coco = json.loads(Path(args.coco_json).read_text())
    n_ann = Counter(a["category_id"] for a in coco.get("annotations", []))
    cats = [c for c in coco.get("categories", []) if n_ann.get(c["id"], 0) > 0]
    kp_cats = [c for c in coco.get("categories", []) if c.get("keypoints")]
    if args.keypoints_json:
        kp = json.loads(Path(args.keypoints_json).read_text())
        kp_cats += [c for c in kp.get("categories", []) if c.get("keypoints")]

    with open(args.group_labels, newline="") as f:
        rows = list(csv.DictReader(f))
    fn_col = next((c for c in rows[0] if c.lower() in ("filename", "image_base", "image", "specimen_id")),
                  list(rows[0])[0])
    species = sorted({r["group_label"] for r in rows})

    words = sorted({re.split(r"[_\s]", c["name"])[0].lower() for c in cats} |
                   {"head", "wing", "forewing", "leg", "metaleg", "rostrum", "terminalia", "thorax",
                    "mesosoma", "gaster", "abdomen", "body", "habitus"}, key=len, reverse=True)
    number_regex = r"_?(\d+)_(?:" + "|".join(re.escape(w) for w in words) + ")"
    hits = sum(1 for r in rows if re.search(number_regex, r[fn_col], re.I))
    number_note = f"matched {hits} of {len(rows)} file names"

    genus = args.genus or "Genus"
    prof = {
        "profile_version": 1,
        "taxon": {"genus": args.genus, "family": args.family,
                  "higher_classification": args.higher_classification or args.family,
                  "group_label_for_text": f"species of {genus}",
                  "imaging": "TODO: describe how specimens were imaged",
                  "colour_caveat": "TODO: state whether photographs were colour-calibrated"},
        "specimen_id": {"number_regex": number_regex,
                        "sex_regex": {"male": "_m_|_male", "female": "_f_|_female"}},
        "species": {sp: {"name": f"{genus} {sp}", "status": "undescribed", "verify": True,
                         "source": "auto-generated from group labels — set name and status"}
                    for sp in species},
        "structures": {},
        "description_sections": [],
        "measurement_abbreviations": {},
        "ratios": [],
        "landmark_sets": {c["name"]: {"term": c["name"].replace("_", " "),
                                      "landmark_name": "landmark", "reference_pair": "auto"}
                          for c in kp_cats},
        "questions_file": str(Path(args.user_prompts).resolve()) if args.user_prompts else None,
        "key": {"feature_type_weights": {"ratio": 1.0, "landmark_ratio": 0.95, "aspect_ratio": 0.9,
                                         "length": 0.8, "landmark_mm": 0.75, "colour": 0.55},
                "min_specimens_primary": 2, "max_secondary_characters": 3},
    }
    secs = []
    for c in cats:
        name = c["name"].replace(" ", "_")
        sec = guess_section(name)
        sex = next((s for s, rx in SEX_HINTS.items() if re.search(rx, name.lower())), "both")
        prof["structures"][name] = {"term": name.replace("_", " "), "section": sec, "sex": sex,
                                    "key_priority": 3 if sec == "Terminalia" else 1}
        if sec not in secs:
            secs.append(sec)
    prof["description_sections"] = secs + ["Proportions"]

    header = (
        "# =============================================================================\n"
        "# BioRAG taxon profile — AUTO-GENERATED DRAFT by biorag_make_taxon_profile_v1.py\n"
        f"# specimen number regex {number_note}\n"
        "# TODO before publication:\n"
        "#  * species: set display names and status (described | cf | undescribed)\n"
        "#  * structures: check terms, sections, sex (male|female|both); set split_by_sex: true\n"
        "#    for structures that differ between the sexes (e.g. subgenital plates)\n"
        "#  * measurement_abbreviations + ratios: add your group's standard proportions\n"
        "#    (see TEMPLATE_taxon_profile.yaml and diaphorina_taxon_profile.yaml)\n"
        "# =============================================================================\n")
    Path(args.output).write_text(header + yaml.safe_dump(prof, sort_keys=False, allow_unicode=True))
    print(f"Draft taxon profile: {args.output}  ({len(species)} species, {len(cats)} structures, "
          f"{len(kp_cats)} landmark sets; specimen regex {number_note})")


if __name__ == "__main__":
    main()
