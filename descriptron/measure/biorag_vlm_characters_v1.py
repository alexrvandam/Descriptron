#!/usr/bin/env python3
"""
biorag_vlm_characters_v1.py — let a model propose characters, then test whether they hold
==========================================================================================

The deterministic miner reads the outline of a segmented structure. A great deal of what
a taxonomist actually uses lies *inside* that outline — setae and their insertions,
sculpture, pigment boundaries, the shape of a single process — and no outline descriptor
will ever recover it. This arm asks a vision model to propose such characters, and then
subjects every proposal to exactly the machinery that governs everything else here.

Three stages, and the separation between them is the whole point
----------------------------------------------------------------
1. PROPOSE   Small contrast sets: specimens of one species beside specimens of the species
             it is hardest to separate from, one structure at a time, cropped to the mask.
             Only the *proposing half* of each series is ever shown. The model is asked for
             binary, observable features with a stated place to look — never for a verdict
             about which species is which.
2. SCORE     Every specimen is then scored against the fixed character list ONE IMAGE AT A
             TIME, with no species label and no sight of any other specimen. This is the
             expensive stage and it is deliberately the dull one: a single image per call
             keeps each judgement independent, removes position and ordering effects, makes
             every call individually re-runnable, and keeps a re-score cheap. Showing the
             model a hundred images at once would be cheaper in calls and worthless in
             evidence, because a character proposed and scored in the same context is
             scored by a model that already knows the answer.
3. RETEST    A sample is scored a second time, in a fresh context, to measure how often the
             model gives the same image the same answer. A character the scorer cannot
             reproduce cannot help anyone identify a specimen, whatever it means.

The output is a states table in the same shape the deterministic miner produces, so it
goes through `biorag_autapomorphy_v1.py` unchanged: the same held-back confirmation, the
same FDR over every test attempted, the same two-arm novelty evaluation. A character that
a model invented and a character that a contour algorithm computed are then judged by the
same standard, which is the only way to find out whether the model added anything.

Singletons are scored but never used to propose: a lone specimen cannot show that a state
is fixed within its species, and letting it propose would simply describe that individual.

  python biorag_vlm_characters_v1.py --coco <coco.json> --image_dir <dir> \\
      --taxon_profile <p.yaml> --matrix_dir "$M/compiled_key_tier" \\
      --out_dir "$M/vlm_characters" --structures whole_wing vertex --llm-backend claude-code
"""

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                        # noqa: E402
from biorag_llm_backend import (add_backend_args, client_from_args,          # noqa: E402
                                answering_model, models_in_log)
from biorag_specimen_id import specimen_of                          # noqa: E402

VERSION = "1.0"
Image.MAX_IMAGE_PIXELS = None
MAX_PX = 900

PROPOSE_SYSTEM = """You are assisting a morphological revision. You will be shown the same
anatomical structure from several insect specimens, in two groups. Every image shows only the
annotated structure: everything outside it is blank white. The specimens have been rotated to a
common orientation and scaled to a common size, but their shapes are unaltered, so differences
in outline, proportion and pattern are real differences between the specimens.

Propose character states that could be scored by a person with a microscope and the specimen
in front of them. Each must be:
  - BINARY: present or absent, with no intermediate;
  - LOCALISED: say where on the structure to look, in words a taxonomist would use;
  - OBSERVABLE: about form, sculpture, setation, margin or pigment boundary — something in the
    specimen, not in the photograph. Never about image quality, brightness, focus, staining,
    mounting medium, bubbles, debris, orientation on the slide, or how the crop was framed.
{colour_rule}

Do not say which group is which, do not name species, and do not guess a taxonomy. If nothing
in a group is reliably distinguishable, say so: an empty list is a valid and useful answer.

Reply as JSON only:
{"characters":[{"name":"<=6 words","where":"where to look","present":"what presence looks like",
"absent":"what absence looks like"}]}"""

COLOUR_PATTERN_ONLY = """  - PATTERN, NOT SHADE, where pigment is concerned. The arrangement of pigment is a
    character: where it sits, how far it extends, whether its edges are sharp or diffuse,
    whether markings are coarse or fine, discrete or confluent, and which areas stay clear.
    The SHADE is not: how brown, how yellow, how dark or how saturated something looks
    cannot be compared between these specimens, because the images are not colour-calibrated
    and preparation bleaches the specimen to different degrees. Say "pigment confined to the
    apical third" or "markings with sharp margins", never "dark brown ground" or "paler
    yellow than usual"."""

# A proposed character whose whole content is a shade judgement. Colour WORDS are fine when
# they name a region that is being placed or measured; the test is whether anything survives
# if the colour term is removed.
SHADE_ONLY = re.compile(
    r"^(?:\s*(?:the|a|an)\s+)?(?:(?:very|slightly|mostly|generally|overall)\s+)?"
    r"(?:dark|darker|darkest|pale|paler|palest|light|lighter|deep|intense|saturated|dull|"
    r"bright|brown|brownish|yellow|yellowish|amber|ochre|tan|fuscous|testaceous|black|"
    r"ground|background|body)\W*"
    r"(?:colou?r|shade|tone|tint|hue|ground|background|pigmentation)?\s*$", re.I)


# Words that give a pigment character spatial content: where it is, how far it reaches, how
# its edges behave, what grain it has. A character carrying any of these is about PATTERN and
# survives bleaching; one carrying none is a judgement of shade and does not.
SPATIAL = re.compile(
    r"\b(extent|extends?|confined|restricted|covers?|covering|reach(?:es|ing)?|margin|margins|"
    r"boundar(?:y|ies)|edge|edges|sharp|sharply|diffuse|abrupt|gradual|coarse|fine|discrete|"
    r"separate|confluent|merged|isolated|patch(?:es)?|blotch(?:es)?|fleck(?:s)?|spot(?:s)?|"
    r"macula(?:e|te)?|band(?:s|ed)?|stripe(?:s|d)?|streak(?:s)?|mottl(?:ed|ing)|speckl(?:ed|ing)|"
    r"apical|basal|distal|proximal|anterior|posterior|median|lateral|marginal|submarginal|"
    r"dominant|throughout|along|between|around|near|adjacent|third|half|quarter|cell|cells|"
    r"vein|veins|area|areas|region|hyaline|clear|transparent|pattern|arrangement|distribution|"
    r"dense|sparse|scattered|uniform|even|irregular)\b", re.I)


def shade_only(name: str, where: str = "") -> bool:
    """True when the character says nothing beyond how dark or how brown something is.

    Pigment PATTERN is a legitimate character here and pigment SHADE is not, so the test is
    not whether a colour word appears — "coarse brown blotches" is a pattern — but whether
    anything spatial appears alongside it. A name that ends in "colour", "shade" or "tone"
    and carries no word about placement, extent, margin or grain is a shade judgement, and
    on uncalibrated images of variably bleached specimens it cannot be repeated.
    """
    n = str(name).strip()
    # Only the character's own name is tested. The `where` field is an instruction to the
    # scorer and is spatial by construction ("the membrane between the veins"), so counting
    # it would let every shade judgement through on the strength of its own directions.
    if SPATIAL.search(n):
        return False
    if SHADE_ONLY.match(n):
        return True
    if re.search(r"\b(colou?r|shade|tone|tint|hue|pigmentation|darkness|paleness)\s*$", n, re.I):
        return True
    # strip every colour and intensity word; if almost nothing is left, it was a shade call
    rest = re.sub(r"\b(dark|darker|pale|paler|light|lighter|deep|bright|dull|intense|saturated|"
                  r"brown|brownish|yellow|yellowish|amber|ochre|tan|fuscous|testaceous|black|"
                  r"colou?r|shade|tone|tint|hue)\b", " ", n, flags=re.I)
    rest = re.sub(r"\b(the|a|an|of|is|with|and|very|mostly|overall|general|generally)\b", " ",
                  rest, flags=re.I)
    return len(re.sub(r"\W+", "", rest)) <= 3


SCORE_SYSTEM = """You are scoring one specimen against a fixed list of morphological characters.
The image shows only the annotated structure; everything outside it is blank white and is not
part of the specimen. The specimen has been rotated and scaled to a standard presentation, but
its shape is unaltered. Score only the structure itself.

For each character, answer present or absent or unclear, AND say where on the structure you
saw it. Give the location as two numbers between 0 and 1: x across the image from left, y down
from the top. Point at the feature itself, not at the middle of the structure. If you answer
unclear, give the location you looked at. Answer unclear whenever the region is
not visible, is damaged, or you are not confident — unclear is always preferable to a guess,
because a character you cannot score consistently is worse than no character at all.

Judge only the specimen. Never let image brightness, focus, staining, mounting medium, debris
or framing influence an answer.

Reply as JSON only:
{"scores":{"<character name>":{"state":"present|absent|unclear","x":0.0-1.0,"y":0.0-1.0}, ...}}"""


def crop(img_path: Path, pts: np.ndarray, pad: float = 0.12, mask_mode: str = "hard"):
    """Show the model the ANNOTATED STRUCTURE, not a rectangle of the slide.

    A bounding-box crop of a wing cell contains its neighbours, the slide background and
    whatever debris lies nearby, so a character proposed from it may be about something
    outside the annotation entirely — the attribution failure the rest of this workflow
    exists to prevent. The polygon is therefore composited out:

      hard  everything outside the polygon is replaced with flat white, so nothing but the
            structure can be scored. Attribution is unambiguous.
      dim   the outside is lightened towards white but left faintly visible, which keeps the
            structure's position relative to its neighbours readable at some cost in rigour.
      none  the raw bounding-box crop; kept only to reproduce the unmasked pilot.

    The outline is drawn in either masked mode so the boundary is explicit rather than
    inferred from where the image happens to stop.
    """
    im = Image.open(img_path).convert("RGB")
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    p = pad * max(x1 - x0, y1 - y0)
    box = (max(0, int(x0 - p)), max(0, int(y0 - p)),
           min(im.width, int(x1 + p)), min(im.height, int(y1 + p)))
    c = im.crop(box)
    if mask_mode != "none":
        local = [(float(x) - box[0], float(y) - box[1]) for x, y in pts]
        m = Image.new("L", c.size, 0)
        ImageDraw.Draw(m).polygon(local, fill=255)
        white = Image.new("RGB", c.size, (255, 255, 255))
        if mask_mode == "dim":
            faded = Image.blend(c, white, 0.80)
            c = Image.composite(c, faded, m)
        else:
            c = Image.composite(c, white, m)
        d = ImageDraw.Draw(c)
        d.line(local + [local[0]], fill=(40, 40, 40), width=2)
    if max(c.size) > MAX_PX:
        c.thumbnail((MAX_PX, MAX_PX))
    return c


def framed(frames_dir, struct, sid):
    """The pre-aligned image for this specimen, if one was built."""
    if not frames_dir:
        return None
    p = Path(frames_dir) / struct / f"{sid}.png"
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def block(img: Image.Image) -> dict:
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=88)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(b.getvalue()).decode()}}


def ask(client, model, system, content, max_tokens=1800):
    r = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": content}])
    txt = "".join(getattr(b, "text", "") for b in r.content) if hasattr(r, "content") else str(r)
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def outlines(coco: Path, profile, codes, structures, drop_ids=(), drop_imgs=()):
    j = json.loads(coco.read_text())
    cats = {c["id"]: c["name"] for c in j.get("categories", [])}
    imgs = {i["id"]: i["file_name"] for i in j.get("images", [])}
    out = {}
    for a in j.get("annotations", []):
        seg = a.get("segmentation")
        if not seg or not isinstance(seg, list) or not seg[0]:
            continue
        st = cats.get(a["category_id"], "?")
        if structures and st not in structures:
            continue
        if a.get("id") in drop_ids or imgs.get(a["image_id"], "") in drop_imgs:
            continue
        pts = np.asarray(seg[0], dtype=float).reshape(-1, 2)
        if len(pts) < 12:
            continue
        fn = imgs.get(a["image_id"], "")
        _c, sid = specimen_of(fn, profile, codes)
        if not sid:
            continue
        k = (sid, st)
        if k not in out or len(pts) > len(out[k][0]):
            out[k] = (pts, fn)
    return out


def load_exclusions(spec):
    """Which annotations must be ignored, by id where the screen gives one.

    A drifted polygon still produces a number and nothing downstream can tell that number
    from a measurement, so it has to go. But dropping every annotation on an image because
    one of them drifted throws away good work: on most flagged images only one or two
    cells moved. Annotation ids are used when the screen supplies them and whole images
    only when it does not.
    """
    ids, images = set(), set()
    for f in str(spec or "").split(","):
        f = f.strip()
        if not f or not Path(f).exists():
            continue
        t = pd.read_csv(f, sep="\t" if f.endswith(".tsv") else ",")
        if "annotation_id" in t.columns:
            flag = t["flag"] if "flag" in t.columns else True
            ids |= set(pd.to_numeric(t.loc[flag, "annotation_id"], errors="coerce").dropna().astype(int))
        else:
            col = next((c for c in ("image", "file_name", "image_name") if c in t.columns), None)
            if col:
                images |= set(t[col].astype(str))
    return ids, images


def main():
    ap = argparse.ArgumentParser(description="Model-proposed characters, independently tested")
    ap.add_argument("--coco", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--structures", nargs="*", default=None,
                    help="restrict to these COCO categories (default: all)")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--propose_frac", type=float, default=0.5)
    ap.add_argument("--contrast", type=int, default=3, help="specimens shown per group")
    ap.add_argument("--max_characters", type=int, default=0,
                    help="0 = no global cap. A global cap spends the whole budget on the first "
                         "structures and leaves the rest unexamined; use --max_per_structure")
    ap.add_argument("--max_per_structure", type=int, default=30,
                    help="characters kept per structure after de-duplication. Be generous: "
                         "scoring costs ONE call per specimen with every character in the same "
                         "prompt, so raising this adds prompt length, not calls. The kept ones "
                         "are simply the first proposed, not the best, so a tight cap discards "
                         "arbitrarily")
    ap.add_argument("--retest_n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_species", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue a run that was interrupted, skipping specimens already in "
                         "vlm_character_states.partial.tsv")
    ap.add_argument("--keep_shade", action="store_true",
                    help="keep characters that are only a judgement of how dark or how brown "
                         "something is. Dropped by default unless the taxon profile declares "
                         "colour_calibrated: true, because uncalibrated images and uneven "
                         "bleaching make shade unrepeatable between specimens while the PATTERN "
                         "of pigment survives both")
    ap.add_argument("--frames_dir", default=None,
                    help="output of biorag_homology_frame_v1.py. When given, the model is shown "
                         "<frames_dir>/<structure>/<specimen>.png — every specimen already at a "
                         "common orientation and size, shape untouched — instead of a crop cut "
                         "from the raw image. A location reported on such an image is comparable "
                         "between specimens; one reported on a raw crop is not")
    ap.add_argument("--exclude_list", default=None,
                    help="annotation_screen/annotation_screen.tsv (per annotation) and/or an "
                         "exclusion CSV: a polygon that has drifted off the specimen must never be "
                         "shown to the model, or it will describe the slide")
    ap.add_argument("--mask_mode", choices=["hard", "dim", "none"], default="hard",
                    help="what the model is shown outside the annotated polygon (see crop())")
    add_backend_args(ap)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    long = pd.read_csv(Path(a.matrix_dir) / "specimen_matrix_long.csv")
    species_of = dict(zip(long["specimen_id"], long["species"]))
    codes = sorted(set(species_of.values()))
    drop_ids, drop_imgs = load_exclusions(a.exclude_list)
    outl = outlines(Path(a.coco), profile, codes, set(a.structures or []), drop_ids, drop_imgs)
    if drop_ids or drop_imgs:
        print(f"screened out {len(drop_ids)} annotations, {len(drop_imgs)} images")
    client = client_from_args(a, log_path=str(out / "llm_calls.jsonl"))
    rng = np.random.default_rng(a.seed)
    img_dir = Path(a.image_dir)

    cal = bool((profile.get("taxon") or {}).get("colour_calibrated", False))
    # a plain replace, not .format(): the prompt ends with a literal JSON example and
    # str.format would try to read its braces as fields
    propose_system = PROPOSE_SYSTEM.replace("{colour_rule}",
                                            "" if cal else COLOUR_PATTERN_ONLY)
    if not cal:
        print("colour is not calibrated for this taxon: pigment PATTERN may be proposed, "
              "shade may not")

    by_struct = {}
    for (sid, st) in outl:
        by_struct.setdefault(st, []).append(sid)

    # ── stage 1: propose, on the proposing half only ─────────────────────────
    resumed = False
    proposing = {}
    for sp in codes:
        ids = sorted([i for i in {s for s, _ in outl} if species_of.get(i) == sp])
        if len(ids) < 2:
            proposing[sp] = []                       # singleton: scored, never proposes
        else:
            k = min(max(1, int(round(a.propose_frac * len(ids)))), len(ids) - 1)
            proposing[sp] = [str(x) for x in rng.choice(ids, size=k, replace=False)]

    # Proposals are expensive and already on disk after a first pass. A resume that
    # re-proposes spends the whole proposal budget again and, worse, gets a DIFFERENT
    # character list, so the scores either side of the interruption stop being comparable.
    cfile = out / "vlm_proposed_characters.tsv"
    if a.resume and cfile.exists() and len(pd.read_csv(cfile, sep="\t")):
        cdf = pd.read_csv(cfile, sep="\t")
        print(f"resuming: reusing {len(cdf)} characters already proposed "
              f"({cdf['structure'].nunique()} structures) - proposal stage skipped")
        chars, prop_log = [], []
        resumed = True
    else:
        chars, prop_log = [], []
        for st, sids in sorted(by_struct.items()):
            sp_here = sorted({species_of.get(s) for s in sids} - {None})
            if a.limit_species:
                sp_here = sp_here[:a.limit_species]
            for sp in sp_here:
                mine = [s for s in proposing.get(sp, []) if (s, st) in outl][:a.contrast]
                other = [s for s in sids if species_of.get(s) not in (sp, None)
                         and s in sum(proposing.values(), [])][:a.contrast]
                if len(mine) < 2 or len(other) < 2:
                    continue
                content = [{"type": "text", "text": f"Structure: {st}. Group A ({len(mine)} specimens):"}]
                ok = True
                for s in mine + other:
                    pts, fn = outl[(s, st)]
                    p = img_dir / fn
                    if not p.exists():
                        ok = False
                        break
                    if s == other[0]:
                        content.append({"type": "text", "text": f"Group B ({len(other)} specimens):"})
                    fr = framed(a.frames_dir, st, s)
                    content.append(block(fr if fr is not None
                                         else crop(p, pts, mask_mode=a.mask_mode)))
                if not ok:
                    continue
                content.append({"type": "text", "text":
                                "Propose characters that separate Group A from Group B, following "
                                "the rules. An empty list is acceptable."})
                r = ask(client, a.model, propose_system, content)
                for c in (r.get("characters") or [])[:6]:
                    if not c.get("name"):
                        continue
                    nm = str(c["name"])[:60]
                    flag = (not cal) and shade_only(nm, c.get("where", ""))
                    chars.append({"structure": st, "name": nm,
                                  "where": c.get("where", ""), "present": c.get("present", ""),
                                  "absent": c.get("absent", ""), "proposed_from": sp,
                                  "seen_species": ";".join(sorted({str(species_of.get(x))
                                                                   for x in mine + other})),
                                  "shade_only": bool(flag)})
                prop_log.append({"structure": st, "species": sp,
                                 "shown_A": ";".join(map(str, mine)),
                                 "shown_B": ";".join(map(str, other)),
                                 "n_returned": len(r.get("characters") or [])})
                print(f"  propose {st}/{sp}: {len(r.get('characters') or [])}")
    if chars:
        cdf = pd.DataFrame(chars)
    if chars and len(cdf):
        # The same character is often proposed from several contrast sets. Keep one row, but
        # keep every species that was on screen for any of them: when one of those species is
        # later withheld to stand in for an undescribed one, the character must be dropped for
        # that fold, because no undescribed species could have helped draw up the list
        # (biorag_congruence_compare_v1.py --proposals reads seen_species for exactly this).
        seen = (cdf.groupby(["structure", "name"])["seen_species"]
                .agg(lambda v: ";".join(sorted({x for s_ in v for x in str(s_).split(";") if x}))))
        cdf = cdf.drop_duplicates(subset=["structure", "name"])
        cdf["seen_species"] = [seen[(a_, b_)] for a_, b_ in zip(cdf["structure"], cdf["name"])]
        n_shade = int(cdf["shade_only"].sum())
        if n_shade and not a.keep_shade:
            print(f"dropping {n_shade} shade-only characters: "
                  + ", ".join(cdf[cdf["shade_only"]]["name"].head(6)))
            cdf = cdf[~cdf["shade_only"]]
        # Cap PER STRUCTURE, never globally. Proposals arrive structure by structure, so a
        # global head() silently spends the whole quota on whichever structures came first and
        # leaves the rest with no characters at all — the run then looks complete while having
        # examined a fraction of the anatomy.
        cdf = (cdf.groupby("structure", group_keys=False)
               .apply(lambda g: g.head(a.max_per_structure)).reset_index(drop=True))
        if a.max_characters:
            keep = []
            for stx, g in cdf.groupby("structure"):
                keep.append(g)
            cdf = pd.concat(keep, ignore_index=True) if keep else cdf
    if not resumed:
        # On --resume both files are already on disk from the first pass. Rewriting them here
        # replaced the proposal log with an empty table, losing the only record of which
        # specimens the model had seen when it drew up the characters.
        cdf.to_csv(out / "vlm_proposed_characters.tsv", sep="\t", index=False)
        pd.DataFrame(prop_log).to_csv(out / "vlm_proposal_log.tsv", sep="\t", index=False)
    print(f"proposed {len(cdf)} distinct characters")
    if not len(cdf):
        return

    # ── stage 2: score, one image at a time, blind ───────────────────────────
    def score_one(sid, st, chs):
        pts, fn = outl[(sid, st)]
        p = img_dir / fn
        if not p.exists():
            return {}
        listing = "\n".join(f"- {c['name']}: look at {c['where']}. Present = {c['present']}. "
                            f"Absent = {c['absent']}." for _, c in chs.iterrows())
        fr = framed(a.frames_dir, st, sid)
        content = [block(fr if fr is not None else crop(p, pts, mask_mode=a.mask_mode)),
                   {"type": "text", "text": f"Structure: {st}.\nCharacters:\n{listing}"}]
        return (ask(client, a.model, SCORE_SYSTEM, content) or {}).get("scores", {})

    def unpack(v):
        """Accept both the plain string form and the localised {state,x,y} form."""
        if isinstance(v, dict):
            return (str(v.get("state", "")).lower(),
                    _f(v.get("x")), _f(v.get("y")))
        return str(v).lower(), None, None

    def _f(x):
        try:
            x = float(x)
            return x if 0.0 <= x <= 1.0 else None
        except (TypeError, ValueError):
            return None

    # Scoring is written as it happens, not at the end. A long run WILL be interrupted —
    # a session limit, a dropped connection, a machine asleep — and a version that only
    # saves on completion loses every call it already paid for. Each specimen is appended
    # as soon as it is scored, and a re-run with --resume skips what is already there.
    part = out / "vlm_character_states.partial.tsv"
    done = set()
    if a.resume and part.exists():
        try:
            prev = pd.read_csv(part, sep="\t")
            done = set(zip(prev["specimen_id"], prev["character"].str.split(":").str[0]))
            print(f"resuming: {len(prev)} state assignments already on disk, "
                  f"{len({d[0] for d in done})} specimens done")
        except Exception:
            done = set()
    fh = part.open("a", encoding="utf-8")
    if not part.stat().st_size:
        fh.write("specimen_id\tspecies\tcharacter\tstate\tx\ty\n")
        fh.flush()

    rows, stopped = [], None
    for st, g in cdf.groupby("structure"):
        if stopped:
            break
        targets = sorted({s for (s, t) in outl if t == st})
        for i, sid in enumerate(targets, 1):
            if (sid, st) in done:
                continue
            try:
                sc = score_one(sid, st, g)
            except RuntimeError as e:
                # the backend is exhausted or unreachable: stop cleanly, keep everything
                stopped = str(e)[:200]
                print(f"\nSTOPPED at {st}/{sid}: {stopped}", flush=True)
                print(f"work so far is in {part.name}; re-run with --resume to continue",
                      flush=True)
                break
            for name, v in sc.items():
                state, x, y = unpack(v)
                if state in ("present", "absent"):
                    r = {"specimen_id": sid, "species": species_of.get(sid),
                         "character": f"{st}:{name}", "state": state, "x": x, "y": y}
                    rows.append(r)
                    fh.write(f"{r['specimen_id']}\t{r['species']}\t{r['character']}\t"
                             f"{r['state']}\t{'' if x is None else x}\t"
                             f"{'' if y is None else y}\n")
            fh.flush()
            if i % 20 == 0:
                print(f"  scored {st}: {i}/{len(targets)}", flush=True)
    fh.close()

    sdf = pd.read_csv(part, sep="\t") if part.exists() and part.stat().st_size else pd.DataFrame(rows)
    sdf.to_csv(out / "vlm_character_states.tsv", sep="\t", index=False)
    if stopped:
        print(f"\nINCOMPLETE: {sdf['specimen_id'].nunique() if len(sdf) else 0} specimens scored "
              f"before the backend stopped. Retest and summary reflect only these.", flush=True)

    # ── stage 3: retest a sample in a fresh context ──────────────────────────
    retest = []
    # The retest is checkpointed like the scoring, and for the same reason: it is several hundred
    # calls, and a version that writes only at the end loses all of them to one interruption.
    rpart = out / "vlm_character_states_retest.partial.tsv"
    rdone = set()
    if a.resume and rpart.exists() and rpart.stat().st_size:
        try:
            prevr = pd.read_csv(rpart, sep="\t")
            retest = prevr.to_dict("records")
            rdone = set(zip(prevr["specimen_id"], prevr["character"].str.split(":").str[0]))
            print(f"resuming the retest: {len(rdone)} specimen x structure readings already on disk")
        except Exception:
            retest, rdone = [], set()
    rfh = rpart.open("a", encoding="utf-8")
    if not rpart.stat().st_size:
        rfh.write("specimen_id\tcharacter\tstate\tx\ty\n")
        rfh.flush()
    if len(sdf):
        pool = sorted(sdf["specimen_id"].unique())
        pick = list(rng.choice(pool, size=min(a.retest_n, len(pool)), replace=False))
        retest_stopped = False
        for sid in pick:
            if retest_stopped:
                break
            for st, g in cdf.groupby("structure"):
                if (sid, st) not in outl or (sid, st) in rdone:
                    continue
                try:
                    _sc = score_one(sid, st, g) or {}
                except RuntimeError as e:
                    # the backend is exhausted: stop, keep what is on disk, resume later — calling a
                    # dead backend for every remaining reading only burns the retries
                    print(f"\nRETEST STOPPED at {st}/{sid}: {str(e)[:160]}\n"
                          f"readings so far are in {rpart.name}; re-run with --resume to continue", flush=True)
                    retest_stopped = True
                    break
                for name, v in _sc.items():
                    state, _x, _y = unpack(v)
                    if state in ("present", "absent"):
                        # keep WHERE the second reading pointed as well as what it called: the
                        # distance between the two readings of one image is the direct measure of
                        # whether the model looks in the same place twice, and it was being discarded
                        retest.append({"specimen_id": sid, "character": f"{st}:{name}",
                                       "state": state, "x": _x, "y": _y})
                        rfh.write(f"{sid}\t{st}:{name}\t{state}\t{'' if _x is None else _x}\t"
                                  f"{'' if _y is None else _y}\n")
                rfh.flush()
    rfh.close()
    rdf = pd.DataFrame(retest)
    rdf.to_csv(out / "vlm_character_states_retest.tsv", sep="\t", index=False)

    agree = {}
    if len(rdf):
        m = sdf.merge(rdf, on=["specimen_id", "character"], suffixes=("_1", "_2"))
        relocate = {}
        for ch, g in m.groupby("character"):
            agree[ch] = round(float((g["state_1"] == g["state_2"]).mean()), 3)
            if {"x_1", "y_1", "x_2", "y_2"} <= set(g.columns):
                both = g[(g["state_1"] == "present") & (g["state_2"] == "present")] \
                    .dropna(subset=["x_1", "y_1", "x_2", "y_2"])
                if len(both):
                    d = np.hypot(both["x_1"] - both["x_2"], both["y_1"] - both["y_2"])
                    relocate[ch] = (int(len(both)), round(float(d.median()), 4))
    pd.DataFrame([{"character": k, "retest_agreement": v,
                   # same image read twice, both times 'present': how far apart the two
                   # reported locations are, in frame units (0-1)
                   "relocated_pairs": relocate.get(k, (0, None))[0],
                   "median_relocation_distance": relocate.get(k, (0, None))[1]}
                  for k, v in agree.items()]) \
        .to_csv(out / "vlm_character_reliability.tsv", sep="\t", index=False)

    summary = {"version": VERSION, "model": a.model, "backend": a.llm_backend,
               # "model" is what was asked for; these two are what the call log says answered
               "model_that_answered": answering_model(str(out / "llm_calls.jsonl")),
               "models_in_call_log": models_in_log(str(out / "llm_calls.jsonl")),
               "mask_mode": a.mask_mode,
               "annotations_excluded": len(drop_ids),
               "frames_dir": a.frames_dir,
               "frame": ("homologous: Procrustes orientation + common size, shape untouched"
                         if a.frames_dir else "raw crop, per-specimen frame (locations NOT comparable)"),
               "structures": sorted(by_struct),
               "characters_proposed": int(len(cdf)),
               "colour_calibrated": cal,
               "shade_only_characters_dropped": int(sum(1 for c in chars if c.get("shade_only")))
               if not a.keep_shade else 0,
               "specimens_scored": int(sdf["specimen_id"].nunique()) if len(sdf) else 0,
               "state_assignments": int(len(sdf)),
               "completed": stopped is None,
               "stopped_reason": stopped,
               "localised_assignments": int(sdf["x"].notna().sum()) if len(sdf) else 0,
               "retest_specimens": int(rdf["specimen_id"].nunique()) if len(rdf) else 0,
               "median_retest_agreement": (round(float(np.median(list(agree.values()))), 3)
                                           if agree else None),
               "characters_reproducible_80pc": int(sum(1 for v in agree.values() if v >= 0.8)),
               "next": "run biorag_autapomorphy_v1.py --states vlm_character_states.tsv "
                       "to judge these by the same standard as the deterministic characters"}
    (out / "vlm_characters_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
