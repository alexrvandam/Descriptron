#!/usr/bin/env python3
"""
biorag_diagnosis_builder_v1.py - a deterministic diagnosis per species, from the key and the matrix (no model)
==========================================================================================================

The model-written diagnosis (biorag_description_refiner_v1) chooses which of the supplied true statements to lead
with, so two runs can stress different characters. This program writes the diagnosis itself, by fixed rules, so the
same matrix and key always give the same text:

  1. KEY CHARACTERS - the couplets on the species' own path through the identification key (biorag_key_builder_v1),
     in key order; for each couplet its primary character, worded as the key words it, but with THIS species'
     observed range and n (a key couplet describes a group of species).
  2. LINEAGE-SPECIFIC CHARACTERS
     - categorical states and presence / absence recorded by the taxonomist (GUI descriptive characters, Record
       absent, Lost keypoints): a state recorded in no other species, or in at most --max_shared others (named);
     - meristic characters (counts): the species' range does not overlap that of at least half of the other species
       scored (n >= 2 in both), with the species named, as in the comparative statements;
     - measurements whose range overlaps no other species scored (n >= 2 in both): unique values.
  Every number is the species' own specimen range from specimen_matrix_long.csv; every species list is written by
  the same function as the comparative statements, so the independent audit (biorag_confabulation_checker_v2) can
  re-derive each claim.

  python biorag_diagnosis_builder_v1.py --matrix_dir compiled_key_tier --key_tree key_v2/key_tree.json \\
      --taxon_profile profile.yaml --out_dir diagnoses/

Writes diagnoses.json ({species: {text, items: [{source, feature_id, couplet, text}], words}}), diagnoses.md and
diagnoses.tsv. biorag_description_refiner_v1 --deterministic_diagnosis diagnoses.json uses the text verbatim in
place of the model's diagnosis (run_full_pipeline_v2 --deterministic_diagnosis does both).
"""
import argparse
import copy
import json
import re
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                       # noqa: E402
import biorag_description_refiner_v1 as rf                # noqa: E402
import biorag_key_builder_v1 as kb                        # noqa: E402

VERSION = "1.0"


class _NS(types.SimpleNamespace):
    def __getattr__(self, k):
        return None


def species_values(ev, code, fid):
    v = ev.long.loc[(ev.long["species"] == code) & (ev.long["feature_id"] == fid), "value"].dropna()
    return v.to_numpy(dtype=float)


def key_items(ev, code, max_group=6):
    """the primary character of each couplet on the species' path whose side holds at most max_group species (the
    couplets that separate it from its nearest species; the same rule as the data sheet), with its own range"""
    items = []
    if not ev.key:
        return items
    max_group = min(max_group, max(1, len(ev.species) // 4))   # scales with the genus: 6 of 29, 1 of 4
    for r in sorted(ev.key["couplets"], key=lambda r: r["number"]):
        for side in ("A", "B"):
            if code not in r[f"{side}_species"] or (r[f"{side}_goto"] != code and
                                                     len(r[f"{side}_species"]) > max_group):
                continue
            chars = sorted(r["characters"], key=lambda c: c.get("role") != "primary")
            for c in chars:                              # the first character this species was scored for
                vals = species_values(ev, code, c["feature_id"])
                if not len(vals):
                    continue
                c2 = copy.deepcopy(c)
                c2[f"{side}_range"] = [float(vals.min()), float(vals.max())]
                c2[f"{side}_n"] = int(len(vals))
                thr = c["threshold"]
                ok = (vals <= thr) if c[f"{side}_operator"] == "<=" else (vals > thr)
                c2["perfect_separation"] = bool(ok.all())
                c2.pop(f"{side}_colour_term", None)          # the group's colour word; not this species'
                txt = kb.template_lead([c2], side)
                where = f"key couplet {r['number']}{side.lower()}"
                txt = txt[:-1] + f"; {where})" if txt.endswith(")") else f"{txt} ({where})"
                m = re.search(r"observed ([^;)]+)", txt)
                cit = [{"value": m.group(1).strip(), "id": c["feature_id"]}] if m else []
                items.append({"source": "key", "couplet": f"{r['number']}{side.lower()}",
                              "feature_id": c["feature_id"], "text": txt, "citations": cit})
                break
    return items


def categorical_items(ev, code, max_shared):
    from biorag_coded_states_from_coco_v1 import recorded_states
    rec = recorded_states(ev.fdict, ev.long)
    mine = rec.get(code) or {}
    items = []
    for (cat, ch), r in sorted(mine.items()):
        scored = [sp for sp in rec if sp != code and (cat, ch) in rec[sp]]
        if len(scored) < 2:
            continue
        term = pol.structure_info(cat, ev.profile).get("term", cat.replace("_", " "))
        for st, k in sorted(r["states"].items(), key=lambda kv: -kv[1]):
            with_ = sorted([sp for sp in scored if st in rec[sp][(cat, ch)]["states"]], key=rf.natural_key)
            if len(with_) > max_shared or len(with_) >= len(scored) / 2:   # specific: few, and under half
                continue
            what = f"{term} {st}" if ch == "presence" else f"{term} {r['label']} {st}"
            count = f"{k} of {r['n']} specimens"
            if not with_:
                tail = f"recorded in no other species, {len(scored)} scored"
            else:
                tail = "otherwise recorded only in " + rf.join_names([rf.short_name(ev, x) for x in with_])
            fid = f"{cat}.coded_{pol_slug(ch)}__{pol_slug(st)}"
            items.append({"source": "categorical", "feature_id": fid, "category": cat, "character": ch,
                          "unique": not with_, "text": f"{what} ({count}; {tail})"})
    items.sort(key=lambda x: (not x["unique"], x["text"]))
    return items


def pol_slug(s):
    from biorag_coded_states_from_coco_v1 import slug
    return slug(s)


def range_items(ev, code, family_filter, min_share, label_source):
    """non-overlapping specimen ranges (n >= 2 in both), as in the comparative statements"""
    fd = ev.fdict
    s_all = ev.summary[ev.summary["n"] >= 2]
    out = []
    for fid, grp in s_all.groupby("feature_id"):
        if fid not in fd.index or not family_filter(fd.loc[fid]):
            continue
        me = grp[grp["species"] == code]
        if me.empty:
            continue
        me = me.iloc[0]
        others = [x for x in grp["species"] if x != code]
        if len(others) < 2:
            continue
        lower = sorted(grp[(grp["species"] != code) & (grp["max"] < me["min"])]["species"], key=rf.natural_key)
        higher = sorted(grp[(grp["species"] != code) & (grp["min"] > me["max"])]["species"], key=rf.natural_key)
        sep = len(lower) + len(higher)
        if sep < max(1, int(np.ceil(min_share * len(others)))):
            continue
        meta = fd.loc[fid]
        unit = meta["unit"] if isinstance(meta["unit"], str) else ""
        rng = rf.fmt_range(me, unit)[0].split(" (")[0]
        count = meta.get("family") == "meristic"
        up, down = ("more", "fewer") if count else ("greater", "smaller")
        if meta.get("family") == "colour":
            up, down = "paler (higher L*)", "darker (lower L*)"
        # the data sheet's own wording (species_list_phrase), so the audit re-derives the same statement
        clauses = [f"{word} than in {rf.species_list_phrase(ev, group, others)}"
                   for group, word in ((lower, up), (higher, down)) if group]
        term = meta["label"]
        out.append({"source": label_source, "feature_id": fid, "separated": sep, "of": len(others),
                    "text": f"{term} {rng} (n = {int(me['n'])}): " + "; ".join(clauses),
                    "citations": [{"value": rng, "id": fid}]})
    out.sort(key=lambda x: (-x["separated"] / max(x["of"], 1), x["feature_id"]))
    return out


def build(ev, code, max_shared=2, meristic_share=0.5, max_unique_measurements=3, key_max_group=6):
    keys = key_items(ev, code, key_max_group)
    cats = categorical_items(ev, code, max_shared)
    # a categorical state already used by the key is not repeated: its 'recorded in no other species' note is
    # appended to the key item instead
    key_cc = {}
    for it in keys:
        if it["feature_id"] in ev.fdict.index and ev.fdict.loc[it["feature_id"]].get("family") == "coded_state":
            key_cc[(ev.fdict.loc[it["feature_id"]]["category"], ev.fdict.loc[it["feature_id"]].get("character"))] = it
    rest = []
    for c in cats:
        k = key_cc.get((c["category"], c["character"]))
        if k is not None:
            tail = c["text"][c["text"].rfind("; ") + 2:-1]
            k["text"] = k["text"][:-1] + f"; {tail})"
        else:
            rest.append(c)
    used = {it["feature_id"] for it in keys}
    meris = [m for m in range_items(ev, code, lambda m: m.get("family") == "meristic", meristic_share, "meristic")
             if m["feature_id"] not in used]
    uniq = [m for m in range_items(ev, code, lambda m: m.get("tier") == pol.TIER_KEY and
                                   m.get("family") not in ("meristic", "coded_state"), 1.0, "unique_measurement")
            if m["feature_id"] not in used][:max_unique_measurements]
    n_other = len(ev.species) - 1
    name = ev.display(code)
    parts = []
    if keys:
        parts.append(f"{name} is separated from its nearest species in the key by: "
                     + "; ".join(it["text"] for it in keys) + ".")
    lin = rest + meris + uniq
    if lin:                                              # one sentence per character (lists contain ';')
        parts.append(("Further diagnostic characters: " if keys else
                      f"{name} differs from the other {n_other} species examined by the following characters: ")
                     + " ".join(f"{it['text'][0].upper()}{it['text'][1:]}." for it in lin))
    text = " ".join(parts) or f"No character separates {name} from the other species in the current matrix."
    cits = [c for it in keys + lin for c in it.get("citations", [])]
    return {"text": text, "items": keys + lin, "citations": cits, "words": len(text.split())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--key_tree", required=True, help="key_tree.json from biorag_key_builder_v1")
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--species", nargs="*", default=None, help="only these species codes (default: all)")
    ap.add_argument("--max_shared", type=int, default=2,
                    help="a categorical state counts as lineage-specific when recorded in at most this many other "
                         "species (named); 0 = only states recorded in no other species (default 2)")
    ap.add_argument("--meristic_share", type=float, default=0.5,
                    help="a count is reported when its range does not overlap this share of the other species (0.5)")
    ap.add_argument("--max_unique_measurements", type=int, default=3)
    ap.add_argument("--key_max_group", type=int, default=6,
                    help="use a key couplet when the species' side holds at most this many species (default 6)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    profile = pol.load_taxon_profile(a.taxon_profile)
    ev = rf.Evidence(_NS(matrix_dir=a.matrix_dir, key_tree=a.key_tree), profile)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    res = {}
    for code in (a.species or ev.species):
        res[code] = build(ev, code, a.max_shared, a.meristic_share, a.max_unique_measurements, a.key_max_group)
    meta = {"version": VERSION, "matrix_dir": str(a.matrix_dir), "key_tree": str(a.key_tree),
            "max_shared": a.max_shared, "meristic_share": a.meristic_share}
    json.dump({"_meta": meta, **res}, open(out / "diagnoses.json", "w"), indent=1, ensure_ascii=False)
    with open(out / "diagnoses.md", "w") as f:
        f.write("# Diagnoses (deterministic: key path + lineage-specific characters)\n\n")
        for code, r in res.items():
            f.write(f"**{ev.display(code)}.** {r['text']}\n\n")
    with open(out / "diagnoses.tsv", "w") as f:
        f.write("species\tsource\tcouplet\tfeature_id\ttext\n")
        for code, r in res.items():
            for it in r["items"]:
                f.write(f"{code}\t{it['source']}\t{it.get('couplet', '')}\t{it['feature_id']}\t{it['text']}\n")
    n_k = sum(any(i["source"] == "key" for i in r["items"]) for r in res.values())
    n_l = sum(any(i["source"] != "key" for i in r["items"]) for r in res.values())
    print(f"{len(res)} diagnoses: {n_k} with key characters, {n_l} with lineage-specific characters; median "
          f"{int(np.median([r['words'] for r in res.values()]))} words -> {out / 'diagnoses.md'}")


if __name__ == "__main__":
    main()
