#!/usr/bin/env python3
"""
biorag_descriptive_scoring_v1.py — score descriptive characters PER SPECIMEN
===========================================================================

Descriptive characters (surface roughness, margin incision, curvature, apex,
colour heterogeneity ...) separate species far better than continuous
measurements, but until now they existed only as free words recorded ONCE PER
SPECIES, which makes them untestable: no within-species replicate, so no
repeatability, no false-positive rate, and no justified weight in a
delimitation score.

This script scores them per specimen and per structure, from the isolated-
structure images the pipeline already produces (`*_fg_<structure>.png` written
by the semilandmark step). For each image the model must pick exactly ONE state
per character from a fixed list, or say "not assessable" — no free text, so the
result is a character x specimen matrix that can be calibrated like any other.

The vocabulary is the one shared with biorag_novelty_score_v1.py
(DESCRIPTIVE_CHARACTERS: ordinal where degree applies, nominal where the states
are kinds). A taxon profile may replace it under `descriptive_characters`.

Usage:
  python biorag_descriptive_scoring_v1.py \\
     --matrix_dir "$M/compiled_key_tier" --taxon_profile <profile.yaml> \\
     --image_roots "/media/.../Diaphorina_semilandmarks" \\
                   "/media/.../Diaphorina_semilandmarks_heads" \\
                   "/media/.../Diaphorina_4species_pipeline/semilandmarks" \\
     --out_dir "$M/descriptive_states" --llm-backend claude-code [--workers 4]
     [--species sp5 sp8] [--limit 20] [--dry_run]

Output:
  descriptive_states_by_specimen.tsv   specimen x structure x character -> state
  scoring_report.json                  coverage, refusals, invalid answers
  cache/<specimen>_<batch>.json        one file per model call (resumable)
"""

import argparse
import base64
import random
import io
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import biorag_feature_policy as pol  # noqa: E402
from biorag_llm_backend import (add_backend_args, client_from_args,  # noqa: E402
                                load_prompt_library, make_llm_client, parse_json_response,
                                models_in_log, answering_model)
from biorag_novelty_score_v1 import (DESCRIPTIVE_CHARACTERS, canonical_states,  # noqa: E402
                                     state_ranks)

VERSION = "1.0"
NOT_ASSESSABLE = "not assessable"
MAX_PX = 640
JPEG_QUALITY = 82

SYSTEM = """You are a taxonomist scoring morphological characters from images of
single, isolated structures of insect specimens. You are filling one column of a
character matrix, not writing prose.

Rules:
 1. For every structure shown, give exactly ONE state per character, chosen from
    the list of allowed states for that character. Copy the state word exactly.
 2. If the character cannot be judged from the image — the structure is damaged,
    out of focus, obscured, seen from the wrong angle, or the character does not
    apply to this structure — answer "{na}". Never guess, and never invent a
    state that is not in the list.
 3. Judge only what is visible in the image you are given. Do not use what you
    know about the group, and do not let one structure influence another.
 4. Return ONLY valid JSON, no markdown fences, in exactly this form:
    {{"<structure id>": {{"<character>": "<state>", ...}}, ...}}
    using the structure ids given in the prompt.""".format(na=NOT_ASSESSABLE)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def norm(s) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


# ─────────────────────────────────────────────────────────────────────────────
# Images
# ─────────────────────────────────────────────────────────────────────────────

def index_images(roots: List[Path]) -> Dict[Tuple[str, str], List[Path]]:
    """(normalised image_base, normalised structure) -> foreground PNGs."""
    idx: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*_fg_*.png"):
            m = re.match(r'(.*)tif_(\d+)_fg_(.+)\.png$', p.name)
            if m:
                idx[(norm(m.group(1)), norm(m.group(3)))].append(p)
    return idx


def crop_to_structure(path: Path, max_px=MAX_PX) -> Optional[str]:
    """Crop the foreground to its bounding box and return base64 JPEG."""
    try:
        im = Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    a = np.asarray(im)
    fg = (a.sum(axis=2) > 12) & (a.std(axis=2) > 2) | (a.sum(axis=2) > 30)
    ys, xs = np.where(fg)
    if len(xs) > 50:
        pad = 12
        im = im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                      min(a.shape[1], xs.max() + pad), min(a.shape[0], ys.max() + pad)))
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

def character_block(characters: Dict) -> str:
    lines = ["CHARACTERS AND THEIR ALLOWED STATES (one state per character, or "
             f"\"{NOT_ASSESSABLE}\"):"]
    for name, cfg in characters.items():
        kind = "a degree scale, in order" if cfg["type"] == "ordinal" else "kinds, unordered"
        lines.append(f"  {name} ({kind}): " + ", ".join(canonical_states(cfg)))
    return "\n".join(lines)


def build_message(batch: List[Dict], characters: Dict, profile: Dict) -> List[Dict]:
    blocks: List[Dict] = [{"type": "text", "text":
                           "Score the following structures of ONE specimen.\n\n"
                           + character_block(characters)
                           + "\n\nStructures in this batch (use these ids as the JSON keys):\n"
                           + "\n".join(f"  {b['category']} = {b['term']}" for b in batch)}]
    for b in batch:
        blocks.append({"type": "text", "text": f"Structure id: {b['category']}  ({b['term']})"})
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                   "data": b["b64"]}})
    blocks.append({"type": "text", "text":
                   "Return only the JSON object described in the instructions, with one entry per "
                   "structure id above."})
    return blocks


def validate(answer: Dict, batch: List[Dict], characters: Dict) -> Tuple[List[Dict], List[str]]:
    rows, problems = [], []
    # every accepted word (synonyms included) maps to the canonical state
    state_index = {}
    for c, cfg in characters.items():
        canon = canonical_states(cfg)
        state_index[c] = {w: canon[int(r)] for w, r in state_ranks(cfg).items()}
    for b in batch:
        got = answer.get(b["category"]) or answer.get(b["term"]) or {}
        if not isinstance(got, dict):
            problems.append(f"{b['category']}: no character block returned")
            continue
        for char in characters:
            raw = got.get(char, NOT_ASSESSABLE)
            val = str(raw).strip().lower()
            if val in ("", "na", "n/a", "none", "null", NOT_ASSESSABLE, "not applicable",
                       "not assessable from the image"):
                state = NOT_ASSESSABLE
            elif val in state_index[char]:
                state = state_index[char][val]
            else:
                first = next((state_index[char][w] for w in re.split(r'[,;/ ]+', val)
                              if w in state_index[char]), None)
                if first:
                    state = first
                    problems.append(f"{b['category']}/{char}: '{raw}' -> '{first}'")
                else:
                    state = NOT_ASSESSABLE
                    problems.append(f"{b['category']}/{char}: '{raw}' is not an allowed state")
            rows.append({"specimen_id": b["specimen_id"], "species": b["species"],
                         "category": b["category"], "structure": b["term"], "character": char,
                         "type": characters[char]["type"], "state": state, "image": str(b["path"])})
    return rows, problems


# ─────────────────────────────────────────────────────────────────────────────

def score_batch(batch, characters, profile, client, args, out_dir, log):
    key = f"{batch[0]['specimen_id']}_{batch[0]['batch_no']}"
    cache = out_dir / "cache" / f"{key}.json"
    if cache.exists() and not args.force:
        d = json.loads(cache.read_text())
        return d.get("rows", []), d.get("problems", []), True
    msgs = [{"role": "user", "content": build_message(batch, characters, profile)}]
    rows, problems = [], []
    for attempt in (1, 2):
        try:
            r = client.messages.create(model=args.model, max_tokens=3000, system=SYSTEM, messages=msgs)
            answer = parse_json_response(r.content[0].text)
        except Exception as e:  # noqa: BLE001
            problems = [f"call failed: {e}"]
            continue
        rows, problems = validate(answer, batch, characters)
        if rows and len(problems) <= len(batch):
            break
        msgs.append({"role": "assistant", "content": json.dumps(answer)[:2000]})
        msgs.append({"role": "user", "content":
                     "Some answers were not allowed states. Return the JSON again, copying each state "
                     "exactly from the allowed list, or \"" + NOT_ASSESSABLE + "\".\nProblems:\n- "
                     + "\n- ".join(problems[:12])})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"rows": rows, "problems": problems,
                                 "structures": [b["category"] for b in batch],
                                 "model": args.model, "generated": datetime.now().isoformat()},
                                indent=1))
    log(f"  {key}: {len(batch)} structures, {sum(1 for r in rows if r['state'] != NOT_ASSESSABLE)}"
        f"/{len(rows)} states, {len(problems)} problems")
    return rows, problems, False


def main():
    ap = argparse.ArgumentParser(description="Score descriptive characters per specimen from images")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--image_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=6, help="structures (images) per model call")
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--specimens", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N batches (a trial)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="build the batches and stop")
    ap.add_argument("--retest_fraction", type=float, default=0.10,
                    help="after scoring, read a sample of the SAME images a second time (a separate "
                         "cache, so the calls really are repeated) and write it to <out_dir>/retest. "
                         "This is what makes the scorer's own repeatability measurable, so it is on by "
                         "default and works whether one species is described or five hundred. 0 disables")
    ap.add_argument("--retest_min_cells", type=int, default=600,
                    help="keep enlarging the retest sample until it reaches about this many scored "
                         "cells, or the whole set is retested (whichever comes first)")
    add_backend_args(ap)
    args = ap.parse_args()

    profile = pol.load_taxon_profile(args.taxon_profile)
    characters = {k: dict(v) for k, v in DESCRIPTIVE_CHARACTERS.items()}
    characters.update(profile.get("descriptive_characters") or {})
    md = Path(args.matrix_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "scoring.log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    full = next(iter(sorted(md.glob("*_full_features.csv"))), None)
    if full is None:
        sys.exit(f"no *_full_features.csv in {md}")
    f = pd.read_csv(full, low_memory=False)
    long = pd.read_csv(md / "specimen_matrix_long.csv")
    keep = {(r.specimen_id, r.category) for r in long.itertuples()}   # structures that survived the filter
    pairs = f[["specimen_id", "category", "image_base", "group_label"]].dropna().drop_duplicates()
    idx = index_images([Path(r) for r in args.image_roots])
    log(f"[{datetime.now().isoformat()}] {len(pairs)} specimen x structure pairs, "
        f"{len(idx)} image keys, {len(characters)} characters")

    items, missing = [], Counter()
    for r in pairs.itertuples():
        if (r.specimen_id, r.category) not in keep:
            continue
        if args.species and r.group_label not in args.species:
            continue
        if args.specimens and r.specimen_id not in args.specimens:
            continue
        base_cat = re.sub(r'__(male|female)$', '', r.category)
        hit = idx.get((norm(str(r.image_base).replace(".tif", "")), norm(base_cat)))
        if not hit:
            missing[r.category] += 1
            continue
        items.append({"specimen_id": r.specimen_id, "species": r.group_label, "category": r.category,
                      "term": pol.structure_info(base_cat, profile).get("term", base_cat),
                      "path": sorted(hit)[0]})
    log(f"{len(items)} structures with an image; no image for "
        f"{sum(missing.values())} ({', '.join(f'{k} {v}' for k, v in missing.most_common(6))})")

    batches: List[List[Dict]] = []
    for sid, group in pd.DataFrame(items).groupby("specimen_id", sort=False):
        recs = group.to_dict("records")
        rng = random.Random(hash(sid) & 0xFFFF)
        rng.shuffle(recs)                    # no fixed structure order within a call
        for i in range(0, len(recs), args.batch_size):
            chunk = recs[i:i + args.batch_size]
            for c in chunk:
                c["batch_no"] = i // args.batch_size
            batches.append(chunk)
    batches.sort(key=lambda b: natural_key(b[0]["specimen_id"]))
    if args.limit:
        batches = batches[:args.limit]
    log(f"{len(batches)} model calls ({args.batch_size} structures each at most)")
    if args.dry_run:
        for b in batches[:3]:
            log(f"  e.g. {b[0]['specimen_id']}: " + ", ".join(x["category"] for x in b))
        return

    for b in batches:                       # encode images once, in the main thread
        for x in b:
            x["b64"] = crop_to_structure(x["path"])
        b[:] = [x for x in b if x["b64"]]
    batches = [b for b in batches if b]

    client = client_from_args(args, log_path=args.llm_log or str(out / "llm_calls.jsonl"))
    rows, problems, cached = [], [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rws, prob, was_cached in ex.map(
                lambda b: score_batch(b, characters, profile, client, args, out, log), batches):
            rows += rws
            problems += prob
            cached += int(was_cached)
    df = pd.DataFrame(rows)
    if df.empty:
        log("no rows scored")
        return
    df.to_csv(out / "descriptive_states_by_specimen.tsv", sep="\t", index=False)
    scored = df[df["state"] != NOT_ASSESSABLE]
    per_char = (scored.groupby("character")["state"].nunique().to_dict())
    degenerate = {}
    for ch, g in scored.groupby("character"):
        vc = g["state"].value_counts(normalize=True)
        degenerate[ch] = {"states": int(g["state"].nunique()),
                          "most_common": vc.index[0], "share": round(float(vc.iloc[0]), 2),
                          "n": int(len(g))}
    report = {
        "version": VERSION, "generated": datetime.now().isoformat(), "model": args.model,
        # "model" is what was asked for; these two are what the call log says answered
        "model_that_answered": answering_model(args.llm_log or str(out / "llm_calls.jsonl")),
        "models_in_call_log": models_in_log(args.llm_log or str(out / "llm_calls.jsonl")),
        "backend": args.llm_backend, "batches": len(batches), "from_cache": cached,
        "specimens": int(df["specimen_id"].nunique()), "structures": int(df["category"].nunique()),
        "cells_total": len(df), "cells_scored": len(scored),
        "coverage_percent": round(100 * len(scored) / max(1, len(df)), 1),
        "states_per_character": per_char,
        "variability_per_character": degenerate,
        "cells_per_character": scored["character"].value_counts().to_dict(),
        "problems": problems[:200], "n_problems": len(problems),
        "images_missing": dict(missing),
    }
    (out / "scoring_report.json").write_text(json.dumps(report, indent=2))
    log(json.dumps({k: v for k, v in report.items() if k not in ("problems",)}, indent=1)[:1500])
    log(f"matrix -> {out / 'descriptive_states_by_specimen.tsv'}")

    # ── read a sample of the same images again, so repeatability can be measured ──────────────
    if args.retest_fraction and args.retest_fraction > 0 and batches:
        cells_per_batch = max(1, len(df) // max(1, len(batches)))
        want = max(int(round(args.retest_fraction * len(batches))),
                   min(len(batches), -(-args.retest_min_cells // cells_per_batch)))
        want = min(want, len(batches))
        step = max(1, len(batches) // want)
        sample = batches[::step][:want]              # spread over the whole set, not the first N
        rout = out / "retest"
        rout.mkdir(parents=True, exist_ok=True)
        log(f"retest: reading {len(sample)} of {len(batches)} batches a second time "
            f"(~{len(sample) * cells_per_batch} cells) -> {rout}")
        rrows: List[Dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rws, _prob, _c in ex.map(
                    lambda b: score_batch(b, characters, profile, client, args, rout, log), sample):
                rrows += rws
        if rrows:
            rdf = pd.DataFrame(rrows)
            rdf.to_csv(rout / "descriptive_states_by_specimen.tsv", sep="\t", index=False)
            (rout / "scoring_report.json").write_text(json.dumps(
                {"version": VERSION, "generated": datetime.now().isoformat(),
                 "note": "second reading of a sample of the same images, for the repeatability test",
                 "batches": len(sample), "of_batches": len(batches),
                 "cells": len(rdf), "specimens": int(rdf["specimen_id"].nunique())}, indent=2))
            log(f"retest matrix -> {rout / 'descriptive_states_by_specimen.tsv'} ({len(rdf)} cells)")
    logf.close()


if __name__ == "__main__":
    main()
