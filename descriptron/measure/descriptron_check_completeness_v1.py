#!/usr/bin/env python3
"""
descriptron_check_completeness_v1.py - which structures and keypoints each specimen has, lacks, or was never scored for
=====================================================================================================================

A structure with no mask can mean three different things, and only the taxonomist can say which:

  present   a mask of the structure, or a visible keypoint (v > 0);
  absent    recorded as lost in the GUI (Record absent -> Absent, or a Lost keypoint): the specimen does not have
            it, e.g. no eyes in a subterranean weevil. This is a CHARACTER STATE and enters the matrix, key and
            descriptions as 'absent';
  unknown   recorded as not visible (hidden, broken, out of focus) or a negative keypoint: MISSING DATA;
  gap       nothing recorded at all: not annotated, skipped or not predicted. Also missing data, but nobody has
            looked - so every gap is listed here for checking.

Only an explicit 'absent' is a character loss. A gap never becomes one by itself.

Reads the COCO files written by the Descriptron GUI (v84+: images[].structure_status and
annotations[].keypoint_status; older files simply have no records, so every missing structure is a gap).

  python descriptron_check_completeness_v1.py --coco annotations.json [more.json ...] \\
      [--group_labels group_labels.csv | --metadata specimens.csv | --species_regex '^([^_]+_[^_]+)'] \\
      [--taxon_profile profile.yaml | --specimen_regex '(sp\\d+_\\d+)'] \\
      [--structures eye antenna ...] [--strict] --out_dir completeness/

The unit is the SPECIMEN: images of one specimen (head, leg, wing photographed separately) are merged, using the
taxon profile's specimen pattern (as the matrix does), --specimen_regex, or the Darwin Core occurrenceID of
--metadata; without any of these each image counts as a specimen. A structure is present if any image of the
specimen shows it, absent if recorded absent and shown in none. With --taxon_profile, structures limited to one sex
(profile 'sex: male' / 'female') are not expected in specimens of the other sex.

Writes
  completeness_matrix.tsv    one row per specimen x character (structure, or structure keypoint N): state
  completeness_worklist.csv  every gap, most likely omissions first: missing in this specimen while
                             conspecifics have it (probably forgotten), then missing in the whole species
                             (possibly a real loss - check and record it), then specimens with no species
  completeness_conflicts.csv recorded absent but a mask / visible keypoint exists (or the reverse)
  completeness_by_character.tsv  per character: present / absent / unknown / gap and the share scored
                             (present + absent); use it to drop poorly scored characters (--min_completeness in
                             biorag_key_feature_filter_v2)
  completeness_heatmap.png   specimens x characters coloured by state
  completeness_summary.json
--strict exits with status 1 while any gap or conflict remains (for pipelines that should stop).
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "1.0"
IMAGE_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp")
NOT_STRUCTURES = {"Trash", "Custom", "trash", "custom"}
STATES = ("present", "absent", "unknown", "gap")


def stem(name):
    b = os.path.basename(str(name))
    return b[: -len(os.path.splitext(b)[1])] if os.path.splitext(b)[1].lower() in IMAGE_EXTS else b


def _has_geometry(a):
    seg = a.get("segmentation")
    if isinstance(seg, dict):
        return bool(seg.get("counts"))
    if isinstance(seg, list):
        return any(isinstance(p, list) and len(p) >= 6 for p in seg)
    return False


def keypoint_names(cat, n):
    names = cat.get("keypoints") if isinstance(cat.get("keypoints"), list) else []
    return [str(names[i]) if i < len(names) and names[i] else f"keypoint {i + 1}" for i in range(n)]


def structure_states(coco_dicts, aliases=None):
    """{image stem: {character: state}} plus the character list, from one or more COCO dicts.
    Characters are structure names (masks) and '<structure> <keypoint name>' (keypoint slots). Images with
    several COCO files are merged: present beats absent beats unknown (a conflict is reported separately)."""
    aliases = aliases or {}
    st = defaultdict(dict)            # stem -> char -> set of evidence
    chars_mask, chars_kp = set(), {}
    images = set()
    conflicts = []
    for d in coco_dicts:
        cats = {c["id"]: c for c in d.get("categories", [])}
        imgs = {im["id"]: im for im in d.get("images", [])}
        for im in d.get("images", []):
            s = stem(im.get("file_name") or im.get("id"))
            images.add(s)
            for name, val in (im.get("structure_status") or {}).items():
                name = aliases.get(name, name)
                if name in NOT_STRUCTURES:
                    continue
                chars_mask.add(name)
                st[s].setdefault(name, set()).add("absent" if val == "absent" else "unknown")
        for a in d.get("annotations", []):
            im = imgs.get(a.get("image_id"), {})
            s = stem(im.get("file_name") or a.get("image_id"))
            images.add(s)
            cat = cats.get(a.get("category_id"), {"name": str(a.get("category_id"))})
            cname = aliases.get(cat.get("name"), cat.get("name"))
            if cname in NOT_STRUCTURES or a.get("source_line_category"):
                continue
            if a.get("keypoints"):
                k = a["keypoints"]; n = len(k) // 3
                names = keypoint_names(cat, n)
                kst = a.get("keypoint_status") or []
                chars_kp.setdefault(cname, names if len(names) >= len(chars_kp.get(cname, [])) else chars_kp[cname])
                for i in range(n):
                    v = int(k[3 * i + 2])
                    state = "present" if v > 0 else ("absent" if len(kst) > i and kst[i] == "absent" else "unknown")
                    st[s].setdefault(f"{cname} {names[i]}", set()).add(state)
            elif _has_geometry(a) and not a.get("is_line"):
                chars_mask.add(cname)
                st[s].setdefault(cname, set()).add("present")
    out = {}
    for s in sorted(images):
        rec = {}
        for ch, ev in st.get(s, {}).items():
            if "present" in ev and "absent" in ev:
                conflicts.append({"image": s, "character": ch,
                                  "problem": "recorded absent but a mask / visible keypoint exists"})
            rec[ch] = "present" if "present" in ev else ("absent" if "absent" in ev else "unknown")
        out[s] = rec
    chars = sorted(chars_mask) + [f"{c} {n}" for c in sorted(chars_kp) for n in chars_kp[c]]
    return out, chars, conflicts


def specimen_map(images, species, profile=None, specimen_regex=None, metadata=None, media_col="associatedMedia",
                 id_col="occurrenceID"):
    """image stem -> (specimen key, sex or None)"""
    occ = {}
    if metadata:
        import csv
        with open(metadata, newline="") as f:
            for row in csv.DictReader(f):
                for m in str(row.get(media_col, "")).split("|"):
                    if m.strip() and row.get(id_col):
                        occ[stem(m.strip())] = row[id_col]
    rx = re.compile(specimen_regex) if specimen_regex else None
    pol = None
    if profile is not None:
        import biorag_feature_policy as pol
    out = {}
    for s in images:
        sp = species.get(s) or "?"
        sex = pol.specimen_sex(s, profile) if pol else None
        if s in occ:
            key = occ[s]
        elif rx is not None and rx.search(s):
            m = rx.search(s); key = f"{sp}|{m.group(1) if m.groups() else m.group(0)}"
        elif pol is not None:
            key = pol.specimen_id(s, sp, profile)
        else:
            key = s
        out[s] = (key, None if sex in (None, "unknown") else sex)
    return out


def merge_specimens(states, spec):
    """per-image states -> per-specimen states (present > absent > unknown); conflicts across images reported"""
    merged, images_of, conflicts = defaultdict(lambda: defaultdict(set)), defaultdict(list), []
    for s, rec in states.items():
        key = spec[s][0]
        images_of[key].append(s)
        for ch, v in rec.items():
            merged[key][ch].add(v)
    out = {}
    for key, rec in merged.items():
        out[key] = {}
        for ch, ev in rec.items():
            if "present" in ev and "absent" in ev:
                conflicts.append({"image": key, "character": ch,
                                  "problem": "absent on one image of the specimen, present on another"})
            out[key][ch] = "present" if "present" in ev else ("absent" if "absent" in ev else "unknown")
    for key in images_of:
        out.setdefault(key, {})
    return out, images_of, conflicts


def species_map(images, group_labels=None, metadata=None, species_regex=None, media_col="associatedMedia",
                species_col="scientificName"):
    sp = {}
    if group_labels:
        import csv
        with open(group_labels, newline="") as f:
            r = csv.DictReader(f)
            fn = "filename" if "filename" in r.fieldnames else r.fieldnames[0]
            for row in r:
                sp[stem(row[fn])] = row.get("group_label") or row.get("species")
    if metadata:
        import csv
        with open(metadata, newline="") as f:
            for row in csv.DictReader(f):
                for m in str(row.get(media_col, "")).split("|"):
                    if m.strip():
                        sp[stem(m.strip())] = row.get(species_col)
    if species_regex:
        rx = re.compile(species_regex)
        for s in images:
            if s not in sp:
                m = rx.search(s)
                if m:
                    sp[s] = m.group(1) if m.groups() else m.group(0)
    return {s: sp.get(s) for s in images}


def check(states, chars, species, structures=None, sex_of=None, structure_sex=None):
    """states/species keyed by specimen (or image). sex_of: specimen -> 'male'/'female'/None; structure_sex:
    structure -> 'male'/'female' for structures limited to one sex (not expected in the other)"""
    if structures:
        keep = set(structures)
        chars = [c for c in chars if c in keep or c.split(" ")[0] in keep]
    sex_of, structure_sex = sex_of or {}, structure_sex or {}
    rows = []
    for s, rec in states.items():
        for ch in chars:
            need = structure_sex.get(ch) or structure_sex.get(ch.split(" ")[0])
            if need and sex_of.get(s) and sex_of[s] != need and ch not in rec:
                continue                                 # e.g. an aedeagus is not expected in a female
            rows.append({"image": s, "species": species.get(s) or "", "character": ch, "state": rec.get(ch, "gap")})
    # worklist: for each gap, what the conspecifics show
    by_sp_char = defaultdict(Counter)
    for r in rows:
        by_sp_char[(r["species"], r["character"])][r["state"]] += 1
    work, seen_species_gap = [], {}
    for r in rows:
        if r["state"] != "gap":
            continue
        c = by_sp_char[(r["species"], r["character"])]
        n_other = sum(c.values()) - 1
        if not r["species"]:
            pri, why = 3, "no species given: check this specimen"
        elif c["present"] > 0:
            pri, why = 1, (f"probably not annotated: {c['present']} of {n_other} other specimen(s) of the species "
                           "have it - annotate it, or record it as absent / not visible")
        elif c["absent"] > 0:
            pri, why = 2, (f"recorded absent in {c['absent']} other specimen(s) of the species - if this one lacks "
                           "it too, record it as absent")
        else:
            # nothing recorded in any specimen of the species: one row for the species (often a body part that
            # was not imaged for it, or a real loss that has not been recorded yet)
            key = ("__species__", r["species"], r["character"])
            if key in seen_species_gap:
                seen_species_gap[key]["image"] += f"; {r['image']}"
                continue
            pri, why = 2, ("nothing recorded in any specimen of the species: if the species lacks it, record it as "
                           "absent; if the part was not imaged, ignore (or record not visible)")
            seen_species_gap[key] = {"priority": pri, "image": r["image"], "species": r["species"],
                                     "character": r["character"], "reason": why}
            work.append(seen_species_gap[key])
            continue
        work.append({"priority": pri, "image": r["image"], "species": r["species"], "character": r["character"],
                     "reason": why})
    work.sort(key=lambda w: (w["priority"], w["species"], w["image"], w["character"]))
    per = []
    for ch in chars:
        c = Counter(r["state"] for r in rows if r["character"] == ch)
        n = sum(c.values())
        per.append({"character": ch, **{k: c.get(k, 0) for k in STATES}, "n_specimens": n,
                    "share_scored": round((c.get("present", 0) + c.get("absent", 0)) / n, 4) if n else 0.0,
                    "species_with_absence": len({r["species"] for r in rows
                                                 if r["character"] == ch and r["state"] == "absent"})})
    return rows, work, per, chars


def heatmap(rows, chars, images, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    code = {"present": 0, "absent": 1, "unknown": 2, "gap": 3}
    idx = {s: i for i, s in enumerate(images)}; jdx = {c: j for j, c in enumerate(chars)}
    M = np.full((len(images), len(chars)), 3)
    for r in rows:
        M[idx[r["image"]], jdx[r["character"]]] = code[r["state"]]
    h = min(0.18 * len(images) + 2, 40); w = min(0.22 * len(chars) + 3, 40)
    fig, ax = plt.subplots(figsize=(w, h))
    ax.imshow(M, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["#4caf50", "#9b30ff", "#b0b0b0", "#ffffff"]), vmin=-0.5, vmax=3.5)
    ax.set_xticks(range(len(chars))); ax.set_xticklabels(chars, rotation=90, fontsize=6)
    ax.set_yticks(range(len(images))); ax.set_yticklabels(images, fontsize=5)
    ax.set_xticks(np.arange(-0.5, len(chars)), minor=True); ax.set_yticks(np.arange(-0.5, len(images)), minor=True)
    ax.grid(which="minor", color="#dddddd", lw=0.3); ax.tick_params(which="minor", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4caf50", label="present"), Patch(color="#9b30ff", label="absent (lost)"),
                       Patch(color="#b0b0b0", label="not visible (missing data)"),
                       Patch(facecolor="#ffffff", edgecolor="#999", label="nothing recorded (check)")],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, frameon=False)
    ax.set_title("Completeness: what each specimen has, lacks, or was never scored for", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", nargs="+", required=True)
    ap.add_argument("--group_labels", default=None, help="filename,group_label CSV (the pipeline's species table)")
    ap.add_argument("--metadata", default=None, help="Darwin Core specimen table (associatedMedia, scientificName)")
    ap.add_argument("--species_regex", default=None, help="regex on the image name; group 1 = species")
    ap.add_argument("--taxon_profile", default=None,
                    help="taxon profile YAML: its specimen pattern merges the images of one specimen, and its "
                         "sex-limited structures are not expected in the other sex")
    ap.add_argument("--specimen_regex", default=None,
                    help="regex on the image name whose group 1 identifies the specimen (images of one specimen merged)")
    ap.add_argument("--structures", nargs="*", default=None, help="check only these structures (default: all)")
    ap.add_argument("--aliases", default=None, help="JSON {old name: new name} for category names")
    ap.add_argument("--strict", action="store_true", help="exit 1 while any gap or conflict remains")
    ap.add_argument("--no_figure", action="store_true")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    aliases = json.load(open(a.aliases)) if a.aliases else {}
    dicts = [json.load(open(p)) for p in a.coco]
    profile, structure_sex = None, {}
    if a.taxon_profile:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import biorag_feature_policy as pol
        profile = pol.load_taxon_profile(a.taxon_profile)
        aliases = {**dict(profile.get("category_aliases") or {}), **aliases}
    states, chars, conflicts = structure_states(dicts, aliases)
    species = species_map(list(states), a.group_labels, a.metadata, a.species_regex)
    if profile is not None:
        for ch in chars:
            sx = pol.structure_info(ch.split(" ")[0], profile).get("sex")
            if sx in ("male", "female"):
                structure_sex[ch.split(" ")[0]] = sx
    spec = specimen_map(list(states), species, profile, a.specimen_regex, a.metadata)
    states, images_of, c2 = merge_specimens(states, spec)
    conflicts += c2
    sex_of = {}
    for s, (key, sx) in spec.items():
        if sx:
            sex_of[key] = sx
    species = {spec[s][0]: species.get(s) for s in spec}
    rows, work, per, chars = check(states, chars, species, a.structures, sex_of, structure_sex)
    for w in work:                                       # which images to open to fix it
        w["images"] = "; ".join(images_of.get(w["image"], [])[:6])
    import csv

    def write(path, recs, cols, sep=","):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter=sep); w.writeheader(); w.writerows(recs)
    write(out / "completeness_matrix.tsv", rows, ["image", "species", "character", "state"], "\t")
    write(out / "completeness_worklist.csv", work, ["priority", "image", "species", "character", "reason", "images"])
    write(out / "completeness_conflicts.csv", conflicts, ["image", "character", "problem"])
    write(out / "completeness_by_character.tsv", per, ["character", *STATES, "n_specimens", "share_scored",
                                                       "species_with_absence"], "\t")
    if not a.no_figure and rows:
        heatmap(rows, chars, list(states), out / "completeness_heatmap.png")
    tot = Counter(r["state"] for r in rows)
    summ = {"version": VERSION, "coco": [str(p) for p in a.coco], "specimens": len(states), "images": len(spec),
            "characters": len(chars),
            "states": dict(tot), "gaps": tot.get("gap", 0), "worklist_rows": len(work),
            "gaps_priority_1": sum(w["priority"] == 1 for w in work),
            "conflicts": len(conflicts), "no_species": sum(1 for s in states if not species.get(s))}
    json.dump(summ, open(out / "completeness_summary.json", "w"), indent=1)
    print(f"{len(states)} specimens ({len(spec)} images) x {len(chars)} characters: present {tot.get('present', 0)}, absent (lost) "
          f"{tot.get('absent', 0)}, not visible {tot.get('unknown', 0)}, nothing recorded {tot.get('gap', 0)}")
    if work:
        print(f"  worklist: {summ['gaps_priority_1']} gap(s) probably not annotated (conspecifics have the structure), "
              f"{sum(w['priority'] == 2 for w in work)} species x structure with nothing recorded in any specimen "
              f"-> {out / 'completeness_worklist.csv'}")
    if conflicts:
        print(f"  {len(conflicts)} conflict(s): recorded absent but annotated -> {out / 'completeness_conflicts.csv'}")
    print(f"-> {out}")
    if a.strict and (work or conflicts):
        sys.exit(1)


if __name__ == "__main__":
    main()
