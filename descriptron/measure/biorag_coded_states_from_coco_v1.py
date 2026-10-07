#!/usr/bin/env python3
"""
biorag_coded_states_from_coco_v1.py - descriptive characters recorded in the GUI -> the character matrix
========================================================================================================

The Descriptron GUI (v82+) and the Descriptron-GBIF Annotator store a structure's descriptive characters in
COCO, as annotations[].attributes = {"texture": "punctate", "setae": "dense", "setae_count": "12"}. This
script turns them into what the pipeline reads:

  1. coded_states_by_specimen.tsv - one row per specimen x structure x character, in the format of
     biorag_descriptive_scoring_v1 (specimen_id, species, category, structure, character, type, state, image)
     plus source=human, so it can be given wherever model-scored states are accepted (--descriptive_matrix);
  2. a COPY of a Tier-1 matrix folder with the characters appended as ordinary features, so the key builder,
     calibration, novelty and the exports see them:
       categorical  one 0/1 column per state ("texture = punctate"), family "coded_state"
       count        the number, family "meristic", unit "count"
     A specimen photographed more than once gets the state seen most often (a tie is left out) and the median
     count. A character scored in fewer than --min_species species, or with one state everywhere, is dropped
     and reported, as in biorag_add_discrete_characters_v1.

States a taxonomist records are bench observations, so they enter at --tier key by default (model-scored
states stay at the tier their reliability check allows). The source matrix folder is never written to.

  python biorag_coded_states_from_coco_v1.py --coco annotations.json [more.json ...] \\
      --group_labels group_labels.csv [--taxon_profile profile.yaml] \\
      --matrix_dir compiled_key_tier --out_dir compiled_key_tier_coded
  (without --matrix_dir only the TSV is written, to --out_dir)
"""
import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
VERSION = "1.0"


def slug(s):
    return re.sub(r"[^\w,\-]+", "_", str(s).strip()).strip("_")


def _stem(name):
    b = os.path.basename(str(name))
    return os.path.splitext(b)[0] if os.path.splitext(b)[1].lower() in (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp") else b


def load_vocabulary():
    """character types (count vs categorical) from the GUI vocabulary, if it can be found"""
    for c in (Path(__file__).resolve().parent / "descriptron_descriptive_characters.json",        # pip: beside it
              Path(__file__).resolve().parent.parent / "descriptron_descriptive_characters.json",
              Path(os.environ.get("DESCRIPTRON_GUI_DATA", "/nonexistent")) / "descriptron_descriptive_characters.json"):
        if c.exists():
            return json.load(open(c))
    return {"characters": {}}


def read_states(coco_paths, group_labels, profile, vocab):
    import biorag_feature_policy as pol
    gl = pd.read_csv(group_labels)
    fn_col = "filename" if "filename" in gl.columns else gl.columns[0]
    species_of = {}
    for fn, sp in zip(gl[fn_col], gl["group_label"]):
        species_of[_stem(fn)] = sp; species_of[_stem(fn).replace(" ", "_")] = sp
    aliases = dict(profile.get("category_aliases") or {})
    chars = vocab.get("characters", {})
    rows, unmatched = [], Counter()
    for path in coco_paths:
        d = json.load(open(path))
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        imgs = {im["id"]: im for im in d.get("images", [])}
        for a in d.get("annotations", []):
            attrs = a.get("attributes") or {}
            if not attrs:
                continue
            im = imgs.get(a.get("image_id"), {})
            stem = _stem(im.get("file_name") or a.get("image_id"))
            sp = species_of.get(stem) or species_of.get(stem.replace(" ", "_"))
            if sp is None:
                unmatched[stem] += 1
                continue
            cat = cats.get(a.get("category_id"), str(a.get("category_id")))
            cat = aliases.get(cat, cat)
            for ch, st in attrs.items():
                if str(ch).startswith("_") or st in (None, ""):
                    continue
                typ = chars.get(ch, {}).get("type", "categorical")
                rows.append({"specimen_id": pol.specimen_id(stem, sp, profile), "species": sp, "category": cat,
                             "structure": cat, "character": ch, "type": "count" if typ == "count" else "nominal",
                             "state": str(st).strip(), "image": stem, "source": "human"})
    df = pd.DataFrame(rows, columns=["specimen_id", "species", "category", "structure", "character", "type",
                                     "state", "image", "source"])
    # a character the vocabulary does not know (added by the user, or no vocabulary found) is a count when every
    # value recorded for it is a whole number
    for ch, g in df.groupby("character"):
        if ch not in chars and g["state"].str.fullmatch(r"\d+").all():
            df.loc[g.index, "type"] = "count"
    return df, unmatched


def one_per_specimen(df):
    """modal state (ties left out) / median count per specimen x structure x character"""
    out = []
    for (sid, sp, cat, ch, typ), g in df.groupby(["specimen_id", "species", "category", "character", "type"]):
        if typ == "count":
            v = pd.to_numeric(g["state"], errors="coerce").dropna()
            if len(v):
                out.append((sid, sp, cat, ch, typ, float(v.median())))
        else:
            c = Counter(g["state"].str.lower()).most_common()
            if len(c) == 1 or c[0][1] > c[1][1]:
                out.append((sid, sp, cat, ch, typ, c[0][0]))
    return pd.DataFrame(out, columns=["specimen_id", "species", "category", "character", "type", "state"])


def recorded_states(fd: pd.DataFrame, long: pd.DataFrame) -> dict:
    """the recorded states back from a matrix: {species: {(category, character): {"label", "states": {state: n},
    "n": specimens scored}}}. Shared by the treatment writer and the audit, so both read the same thing."""
    fd = fd.reset_index() if "feature_id" not in fd.columns else fd
    cs = fd[fd["family"] == "coded_state"] if "family" in fd.columns else fd.iloc[0:0]
    if cs.empty:
        return {}
    meta = {}
    for _, r in cs.iterrows():
        cat = r["category"]
        if "character" in cs.columns and isinstance(r.get("character"), str):
            ch, st = r["character"], r["state"]
            lab = r.get("character_label") if isinstance(r.get("character_label"), str) else ch.replace("_", " ")
        else:                                            # older files: "category: character = state"
            left, _, st = str(r["label"]).rpartition(" = ")
            ch = left.split(": ", 1)[-1]; lab = ch.replace("_", " ")
        meta[r["feature_id"]] = (cat, ch, str(st), lab)
    sub = long[long["feature_id"].isin(meta)]
    out = {}
    for (sp, fid), g in sub.groupby(["species", "feature_id"]):
        cat, ch, st, lab = meta[fid]
        rec = out.setdefault(sp, {}).setdefault((cat, ch), {"label": lab, "states": {}, "specimens": set()})
        rec["specimens"] |= set(g["specimen_id"])
        k = int((g["value"] >= 0.5).sum())
        if k:
            rec["states"][st] = rec["states"].get(st, 0) + k
    for sp in out:
        for rec in out[sp].values():
            rec["n"] = len(rec.pop("specimens"))
    return out


def to_features(per, tier, min_species, known, labels=None):
    """feature-dictionary rows and long-matrix rows; returns (fd, long, dropped)"""
    labels = labels or {}
    rows_fd, rows_long, dropped = [], [], {"scored in too few species": [], "same state in every specimen": []}
    per = per[per["specimen_id"].isin(known)] if known is not None else per
    for (cat, ch, typ), g in per.groupby(["category", "character", "type"]):
        name = f"{cat}:{ch}"
        if g["species"].nunique() < min_species:
            dropped["scored in too few species"].append(name); continue
        if g["state"].nunique() < 2:
            dropped["same state in every specimen"].append(name); continue
        lab = labels.get(ch, ch.replace("_", " "))
        if typ == "count":
            feats = [(f"{cat}.coded_{slug(ch)}", f"{cat}: {lab}", "meristic", "count", "",
                      g.set_index("specimen_id")["state"].astype(float))]
        else:
            feats = [(f"{cat}.coded_{slug(ch)}__{slug(s)}", f"{cat}: {lab} = {s}", "coded_state", "", s,
                      (g.set_index("specimen_id")["state"] == s).astype(float)) for s in sorted(g["state"].unique())]
        meta = g.drop_duplicates("specimen_id").set_index("specimen_id")
        for fid, label, fam, unit, state, vals in feats:
            rows_fd.append({"feature_id": fid, "category": cat, "base_category": cat, "column": "coded_state",
                            "tier": tier, "family": fam, "label": label, "character": ch, "state": state,
                            "character_label": lab,
                            "definition": f"{ch} (recorded by the taxonomist)", "unit": unit, "structure_sex": "both",
                            "section": cat, "key_priority": 1, "n_species": int(g["species"].nunique()),
                            "n_specimens": int(len(vals)), "conversion": ""})
            for sid, v in vals.items():
                rows_long.append({"species": meta.loc[sid, "species"], "specimen_id": sid, "category": cat,
                                  "base_category": cat, "column": "coded_state", "feature_id": fid, "tier": tier,
                                  "family": fam, "value": float(v), "n_images": 1})
    return pd.DataFrame(rows_fd), pd.DataFrame(rows_long), dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", nargs="+", required=True)
    ap.add_argument("--group_labels", required=True)
    ap.add_argument("--taxon_profile", default=None)
    ap.add_argument("--matrix_dir", default=None, help="Tier-1 matrix folder to copy and extend (optional)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tier", choices=["key", "description"], default="key")
    ap.add_argument("--min_species", type=int, default=2)
    a = ap.parse_args()
    import biorag_feature_policy as pol
    profile = pol.load_taxon_profile(a.taxon_profile) if a.taxon_profile else {}
    out = Path(a.out_dir)
    if a.matrix_dir and out.resolve() == Path(a.matrix_dir).resolve():
        sys.exit("refusing to write over the source matrix; give a different --out_dir")
    out.mkdir(parents=True, exist_ok=True)
    df, unmatched = read_states(a.coco, a.group_labels, profile, load_vocabulary())
    df.to_csv(out / "coded_states_by_specimen.tsv", sep="\t", index=False)
    print(f"{len(df)} coded states from {df['specimen_id'].nunique() if len(df) else 0} specimens, "
          f"{df['character'].nunique() if len(df) else 0} characters -> {out / 'coded_states_by_specimen.tsv'}")
    if unmatched:
        print(f"  {sum(unmatched.values())} annotation(s) on {len(unmatched)} image(s) not in --group_labels, skipped: "
              f"{list(unmatched)[:4]}")
    report = {"version": VERSION, "coco": [str(p) for p in a.coco], "n_states": int(len(df)),
              "unmatched_images": dict(unmatched)}
    if a.matrix_dir:
        src = Path(a.matrix_dir)
        fd = pd.read_csv(src / "feature_dictionary.tsv", sep="\t")
        long = pd.read_csv(src / "specimen_matrix_long.csv")
        known = set(long["specimen_id"])
        per = one_per_specimen(df) if len(df) else df
        stray = sorted(set(per["specimen_id"]) - known) if len(per) else []
        if stray:
            print(f"  {len(stray)} specimen(s) with coded states are not in the matrix, skipped: {stray[:4]}")
        labels = {k: v.get("label", k) for k, v in load_vocabulary().get("characters", {}).items()}
        add_fd, add_long, dropped = (to_features(per, a.tier, a.min_species, known, labels) if len(per)
                                     else (pd.DataFrame(), pd.DataFrame(), {}))
        if len(add_fd):
            sex = dict(zip(long["specimen_id"], long["sex"])) if "sex" in long else {}
            add_long["sex"] = add_long["specimen_id"].map(sex)
            clash = set(add_fd["feature_id"]) & set(fd["feature_id"])
            if clash:
                sys.exit(f"feature ids already in the matrix: {sorted(clash)[:5]}")
        pd.concat([fd, add_fd], ignore_index=True).to_csv(out / "feature_dictionary.tsv", sep="\t", index=False)
        pd.concat([long, add_long], ignore_index=True).to_csv(out / "specimen_matrix_long.csv", index=False)
        for extra in ("outlier_flags.tsv", "filter_report_v2.json"):
            if (src / extra).exists():
                shutil.copy2(src / extra, out / extra)
        n_key = lambda f: int((f["tier"] == "key").sum()) if len(f) else 0      # noqa: E731
        print(f"  matrix: {len(add_fd)} features added ({n_key(add_fd)} at tier key); key features "
              f"{n_key(fd)} -> {n_key(fd) + n_key(add_fd)} -> {out}")
        for why, names in dropped.items():
            if names:
                print(f"  dropped ({why}): {names[:6]}{' ...' if len(names) > 6 else ''}")
        report.update(features_added=int(len(add_fd)), dropped=dropped, source_matrix=str(src))
    json.dump(report, open(out / "coded_states_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
