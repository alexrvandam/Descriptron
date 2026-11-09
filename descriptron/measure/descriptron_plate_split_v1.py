#!/usr/bin/env python3
"""
descriptron_plate_split_v1.py - cut a published habitus plate into one image per specimen, named from its caption
=================================================================================================================

A taxonomic plate shows several specimens on a white background, each panel marked with a letter (A, B, C ...),
some panels with a scale bar beside the letter. The caption says which letters show which species and which view
("A and B) Platydracus martini; C and D) P. basicornis. A and C) dorsal view; B and D) ventral view. Scale bar
2 mm."). This program finds the specimens, reads the panel letters off the plate, names every specimen from the
caption and turns each scale bar into pixels per millimetre.

How it works
  specimens    every pixel darker than --bg_threshold in its darkest colour channel is foreground. Scale bars and
               panel letters are taken out first; what remains is labelled into pieces. Pieces of at least
               --min_area_frac of the plate are specimens; smaller pieces (a detached antenna, a leg, setae) are
               joined to the nearest specimen within --max_gap_frac. When the caption names more panels than
               were found and one piece is clearly two specimens touching (antennae of neighbours often do), the
               piece is split: it is eroded until it falls into two or more bodies, and every pixel goes to the
               body it is connected to by the shortest path through the specimen (so an antenna stays with its
               own head).
  scale bars   thin, straight, solid, neutral-black lines (vertical or horizontal) with background on both sides;
               length measured along the bar with sub-pixel ends (the anti-aliased end pixels count by darkness).
  letters      small black marks of letter size that are not part of a specimen, read by comparing each mark's
               shape with the caption's letters drawn in several sans-serif fonts (no OCR needed). Each letter
               then goes to the nearest specimen (one letter per specimen). --letter_order auto (default) uses
               the letters read on the plate; a single unread letter left over is assigned by elimination; when
               several are unread, positional order is used only if it agrees with every letter that was read.
               --letter_order column (A, B down the first column, then the next column) or row (A, B, C along
               the first row) assigns letters by position alone, and still warns where it disagrees with the
               letters read on the plate.
  caption      letter groups "A and B)", "A, C, and E)", "A to D)", "A-D)", "B, D, F," and the text after each: a
               species ("Platydracus martini", "P. basicornis": an abbreviated genus is the last genus spelled out
               with that initial, or --genus) and/or a view ("dorsal view"). A caption that runs on into the next
               figure's caption ("... 0.5 mm. Fig. 32. Distribution maps ...") is cut at "Fig. N.". Scale bars:
               "Scale bar 2 mm", "Scale bars = 0.5 mm", qualified "Habitus scale bar 2 mm, aedeagus scale bar
               0.5 mm" (habitus views take the habitus bar, genitalia views the aedeagus bar; see --view_class)
               and letter-scoped "1 mm (A-C), 0.5 mm (D-F)". --bar_mm overrides the caption.
  px/mm        a panel with its own bar: bar length / bar mm. A panel without a bar takes the px/mm of the panel of
               the SAME species and the same kind of view (habitus or genitalia) with a bar, nearest in letter
               order, preferring the letter before it (A dorsal -> B ventral). ASSUMPTION: that both panels show
               the same specimen at the same magnification, which holds on plates where the dorsal and ventral
               views of one specimen are photographed and laid out at one scale; --no_share_scale turns this off.
               A warning is written when the two specimens' lengths differ by more than 15%.
  never silent warnings for: panels found != letters in the caption, a letter not found or not read, a caption
               label not understood, a letter given two species or two views, a scale bar not found or its
               length not readable, a shared scale between differently sized specimens. A file whose background
               is not white (a map, a photograph) or whose caption cannot be parsed is reported and skipped.

Outputs in --out_dir
  <plate>_<letter>_<Genus_species>_<view>.png    one per panel: the specimen within its outline on white, padded
                                                  by --pad; no letter, no scale bar, no neighbouring specimen
  ..._mask.png                                    with --mask: the specimen's foreground (pinholes filled; the
                                                  background enclosed by legs and antennae stays background)
  plate_split_panels.csv     one row per panel: plate, letter, species, view, crop_file, bbox_x/y/w/h (on the
                             plate), scale_bar_px, scale_bar_mm, px_per_mm, px_per_mm_source, warnings, and how
                             the letter was assigned (letter_method, letter_score)
  plate_split_plates.csv     one row per plate: status (ok / warnings / skipped), panels, caption letters, layout
                             (column, row or other: the order in which the letters were read), warnings
  qc/<plate>_qc.png          the plate, downscaled, with every panel boxed and labelled (letter, species, view,
                             px/mm and its source), the scale bars in red and the letter marks in blue

  python descriptron_plate_split_v1.py --plates plates/ --captions plate_manifest.csv --where habitus=True \\
      --out_dir specimens/ [--key_col fig --key_regex "fig0*(\\d+)"] [--mask]
  python descriptron_plate_split_v1.py --plates fig5.png --caption "A and B) Platydracus alesi; C and D) P. \\
      maculicollis. A and C) dorsal view; B and D) ventral view. Scale bar 2 mm." --out_dir specimens/

Plates and captions: --captions is a CSV with one row per figure; --key_col names the column that ties a row to
a plate file. If its values are file names, they are matched to the plates' file names; otherwise (a figure
number) --key_regex pulls the same key out of each plate's file name (default: the number after "fig"), and
leading zeros are ignored. --where COL=VALUE keeps only matching rows (e.g. habitus=True).
Requires numpy, scipy and Pillow only; nothing is sent anywhere.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# views measured with the habitus scale bar and with the genitalia (aedeagus) scale bar, when a caption gives
# both ("Habitus scale bar 2 mm, aedeagus scale bar 0.5 mm"); anything else (e.g. 'lateral', which can be either)
# needs --view_class lateral=aedeagus
DEFAULT_VIEW_CLASS = {v: "habitus" for v in ("dorsal", "ventral", "habitus", "frontal", "dorsolateral",
                                             "dorso-lateral", "ventrolateral", "ventro-lateral", "anterior",
                                             "posterior", "oblique")}
DEFAULT_VIEW_CLASS.update({v: "genitalia" for v in ("paramere", "parameral", "aedeagus", "aedeagal", "genitalia",
                                                    "genital", "penis", "phallus", "endophallus", "median-lobe",
                                                    "spermatheca", "ovipositor")})
CLASS_SYNONYMS = {"habitus": "habitus", "body": "habitus", "adult": "habitus", "specimen": "habitus",
                  "aedeagus": "genitalia", "aedeagal": "genitalia", "aedeagi": "genitalia", "genitalia": "genitalia",
                  "genital": "genitalia", "paramere": "genitalia", "penis": "genitalia", "phallus": "genitalia"}
NOT_EPITHET = {"sp", "spp", "cf", "aff", "n", "nov", "view", "views", "male", "female", "and", "to", "the"}
UNITS_MM = {"mm": 1.0, "µm": 1e-3, "μm": 1e-3, "um": 1e-3, "cm": 10.0}
TEMPLATE_FONTS = ("LiberationSans-Regular.ttf", "Arial.ttf", "arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf",
                  "LiberationSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf",
                  "FreeSans.ttf", "NotoSans-Regular.ttf", "Ubuntu-R.ttf", "Lato-Regular.ttf")


def canon_class(word: Optional[str]) -> Optional[str]:
    if not word:
        return None
    w = word.lower().strip()
    return CLASS_SYNONYMS.get(w, w)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", s).strip("_") or "unknown"


# ============================================================================ caption
def _letter_group_re(lower: bool = False):
    L = "[a-z]" if lower else "[A-Z]"
    conn = r"(?:\s*,\s*(?:and\s+|&\s*)?|\s+and\s+|\s*&\s*|\s*(?:to|–|—|-)\s*)"
    # a group of single panel letters ended by ')' ',' or ':'; not preceded by a letter, digit or '.', so a
    # genus abbreviation ("P.") or a word is never taken for a panel letter
    return re.compile(rf"(?<![A-Za-z0-9.\-])\(?\s*(?P<g>{L}(?:{conn}{L})*)\s*(?P<end>\)|,|:)")


def expand_letters(g: str, lower: bool = False) -> List[str]:
    L = "[a-z]" if lower else "[A-Z]"
    toks = re.findall(rf"{L}|\bto\b|[–—-]", g)
    out, k = [], 0
    while k < len(toks):
        t = toks[k]
        if t in ("to", "–", "—", "-") and out and k + 1 < len(toks):
            a, b = ord(out[-1]), ord(toks[k + 1])
            out.extend(chr(c) for c in range(a + 1, b + 1))
            k += 2
            continue
        if t not in ("to", "–", "—", "-"):
            out.append(t)
        k += 1
    return out


SPECIES_RE = re.compile(r"^(?P<genus>[A-Z][a-z]+|[A-Z][a-z]?\.)\s+(?:\((?P<subg>[A-Z][a-z]+)\)\s+)?"
                        r"(?P<ep>[a-z][a-z-]+)(?:\s+(?P<ssp>[a-z][a-z-]+))?")
VIEW_RE = re.compile(r"\b([A-Za-z]+(?:-[A-Za-z]+)?)\s+views?\b")
SCALE_RE = re.compile(r"(?:\b(?P<qual>[A-Za-z]+)\s+)?\bscale[\s-]*bars?\b", re.I)
VALUE_RE = re.compile(r"(?P<val>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|µm|μm|um|cm)\b(?:\s*\((?P<letters>[^)]*)\))?")
NEXT_FIG_RE = re.compile(r"(?<=\.)\s+Fig(?:ure)?s?\.?\s*\d+[A-Za-z]?\.\s+(?=[A-Z])")
LEAD_FIG_RE = re.compile(r"^\s*Fig(?:ure)?s?\.?\s*\d+[A-Za-z]?\s*[.:]\s*")


def parse_caption(text: str, genus: Optional[str] = None, lower: bool = False) -> dict:
    """letters, species and view of each letter, and the scale bars of a caption. Never guesses: anything not
    understood is returned in 'warnings'."""
    warnings: List[str] = []
    t = " ".join(str(text or "").split())
    t = LEAD_FIG_RE.sub("", t)
    m = NEXT_FIG_RE.search(t)
    if m:
        warnings.append(f"caption cut before '{t[m.start():m.end()].strip()}' (the next figure's caption)")
        t = t[:m.start()]
    gre = _letter_group_re(lower)
    groups = list(gre.finditer(t))
    letters: List[str] = []
    species: Dict[str, List[str]] = defaultdict(list)
    views: Dict[str, List[str]] = defaultdict(list)
    labels: Dict[str, List[str]] = defaultdict(list)
    genera: List[str] = [genus] if genus else []
    rest = t[:groups[0].start()] if groups else t
    for i, g in enumerate(groups):
        ls = expand_letters(g.group("g"), lower)
        for x in ls:
            if x not in letters:
                letters.append(x)
        end = groups[i + 1].start() if i + 1 < len(groups) else len(t)
        raw = t[g.end():end]
        lab = raw.lstrip(" ,;:)")
        lab = lab.split(";")[0]
        parts = re.split(r"\.\s+(?=[A-Z][a-z]{2,}|\d)", lab, maxsplit=1)
        if len(parts) > 1:
            rest += " " + parts[1]
        lab = parts[0].strip().rstrip(".,; ")
        sp = SPECIES_RE.match(lab)
        if sp and sp.group("ep").lower() in NOT_EPITHET - {"sp", "spp"}:
            sp = None                                  # "Dorsal view", "Male and ...": not a species name
        vw = VIEW_RE.search(lab)
        name = None
        if sp:
            gen = sp.group("genus")
            if gen.endswith("."):
                full = next((x for x in reversed(genera) if x.startswith(gen[:-1])), None)
                if full is None:
                    warnings.append(f"genus '{gen}' ({', '.join(ls)}) is not spelled out earlier in the caption; "
                                    f"give --genus")
                    full = gen.rstrip(".")
                gen = full
            else:
                genera.append(gen)
            ep = sp.group("ep")
            if ep.lower() in ("sp", "spp"):
                name = f"{gen} sp"
            else:
                name = f"{gen} {ep}"
                ssp = sp.group("ssp")
                if ssp and ssp.lower() not in NOT_EPITHET:
                    name += f" {ssp}"
        view = vw.group(1).lower() if vw else None
        if not sp and not vw and re.fullmatch(r"[a-z][a-z-]*(?: [a-z][a-z-]*){0,2}", lab or "X"):
            view = lab.replace(" ", "-")               # "habitus", "aedeagus": a view named without 'view'
        if name is None and view is None:
            warnings.append(f"label '{lab}' for {', '.join(ls)} not understood")
        for x in ls:
            labels[x].append(lab)
            if name:
                species[x].append(name)
            if view:
                views[x].append(view)
    if not groups:
        warnings.append("no panel letters found in the caption")
    # a view stated once for the whole plate ("Dorsal views.") applies to letters without one
    gv = VIEW_RE.search(rest)
    if gv:
        for x in letters:
            if not views.get(x):
                views[x].append(gv.group(1).lower())
    out_sp, out_vw = {}, {}
    for x in sorted(letters):
        s = list(dict.fromkeys(species.get(x, [])))
        v = list(dict.fromkeys(views.get(x, [])))
        if len(s) > 1:
            warnings.append(f"{x} is given {len(s)} species ({'; '.join(s)}); species left unknown")
        out_sp[x] = s[0] if len(s) == 1 else None
        if len(v) > 1:
            warnings.append(f"{x} is given {len(v)} views ({', '.join(v)}); view left unknown")
        out_vw[x] = v[0] if len(v) == 1 else None
    nosp = [x for x in sorted(letters) if out_sp[x] is None and len(species.get(x, [])) == 0]
    if nosp and groups:
        warnings.append(f"no species named for {', '.join(nosp)}")
    novw = [x for x in sorted(letters) if out_vw[x] is None and len(views.get(x, [])) == 0]
    if novw and groups:
        warnings.append(f"no view named for {', '.join(novw)}")
    bars, bw = parse_scale_bars(t, lower)
    warnings += bw
    return {"text": t, "letters": sorted(letters), "species": out_sp, "views": out_vw, "bars": bars,
            "labels": dict(labels), "warnings": warnings}


def parse_scale_bars(t: str, lower: bool = False) -> Tuple[List[dict], List[str]]:
    """every 'scale bar' statement: [{'cls': 'habitus'|None, 'mm': 2.0|None, 'letters': set|None, 'raw': ...}]"""
    out, warnings = [], []
    hits = list(SCALE_RE.finditer(t))
    for i, h in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(t)
        if i + 1 < len(hits) and hits[i + 1].group("qual"):
            end = hits[i + 1].start("qual")
        seg = t[h.end():end]
        m_end = re.search(r"\.\s+(?=[A-Z])", seg)
        if m_end:
            seg = seg[:m_end.start()]
        qual = h.group("qual")
        cls = canon_class(qual) if qual and qual.lower() not in ("the", "a", "and", "all", "with") else None
        raw = t[h.start():h.end() + len(seg)].strip().rstrip(",;")
        vals = list(VALUE_RE.finditer(seg))
        if not vals:
            warnings.append(f"scale bar length not readable in '{raw}'")
            out.append({"cls": cls, "mm": None, "letters": None, "raw": raw})
            continue
        for v in vals:
            mm = float(v.group("val").replace(",", ".")) * UNITS_MM[v.group("unit")]
            lets = set(expand_letters(v.group("letters"), lower)) if v.group("letters") else None
            out.append({"cls": cls, "mm": mm, "letters": lets or None, "raw": raw})
    return out, warnings


# ============================================================================ image: scale bars and letters
def _side_clear(fg, r0, r1, c0, c1, gap=2, width=3, max_frac=0.5):
    """the background beside a vertical run (rows r0:r1, columns c0:c1) is mostly empty on both sides"""
    H, W = fg.shape
    lo = fg[r0:r1, max(0, c0 - gap - width):max(0, c0 - gap)]
    hi = fg[r0:r1, min(W, c1 + gap):min(W, c1 + gap + width)]
    return all(s.size == 0 or s.mean() <= max_frac for s in (lo, hi))


def find_bars(gray, sat, fg, dark, longer, min_len_frac=0.012, max_thick_frac=0.012, orient="both",
              letter_boxes=()):
    """thin straight solid neutral lines with background on both sides. Returns dicts with the bar's bbox
    (x, y, w, h on the plate), its orientation and its length in pixels along the bar. A line inside the box of
    a mark that reads as a letter (the stem of an E, the arms of an F) is not a bar."""
    Lmin = max(12, int(round(min_len_frac * longer)))
    max_thick = max(3, int(round(max_thick_frac * longer)))
    bars = []
    for o in ("v", "h"):
        if orient not in ("both", {"v": "vertical", "h": "horizontal"}[o]):
            continue
        D, G, S, F = (dark, gray, sat, fg) if o == "v" else (dark.T, gray.T, sat.T, fg.T)
        V = ndi.binary_opening(D, structure=np.ones((Lmin, 1), bool))
        lab, n = ndi.label(V)
        for k, s in enumerate(ndi.find_objects(lab), 1):
            if s is None:
                continue
            sub = lab[s] == k
            h, w = sub.shape
            widths = sub.sum(1)
            medw = float(np.median(widths))
            if not (2 <= medw <= max_thick) or h < max(Lmin, 4 * medw) or w > medw + 3:
                continue
            if np.abs(widths - medw).mean() > 1.0:                     # not of even width: not a ruled line
                continue
            r0, r1, c0, c1 = s[0].start, s[0].stop, s[1].start, s[1].stop
            inner = slice(c0 + 1, c1 - 1) if (c1 - c0) >= 3 else slice(c0, c1)
            g = G[r0:r1, inner]                                       # ink of the line, without its soft edges
            if g.std() > 30 or S[s][sub].mean() > 30:
                continue
            if not _side_clear(F, r0, r1, c0, c1, max_frac=0.25):
                continue
            bx, by, bw, bh = (c0, r0, c1 - c0, r1 - r0) if o == "v" else (r0, c0, r1 - r0, c1 - c0)
            if any(bx >= lx - 2 and by >= ly - 2 and bx + bw <= lx + lw + 2 and by + bh <= ly + lh + 2
                   for lx, ly, lw, lh in letter_boxes):
                continue
            # length along the bar: darkness profile across its ends, so the anti-aliased end pixels count
            # by how dark they are (sub-pixel)
            e0, e1 = max(0, r0 - 3), min(G.shape[0], r1 + 3)
            prof = G[e0:e1, inner].mean(1)
            core = float(np.median(g))
            bgv = 255.0
            d = np.clip((bgv - prof) / max(1.0, bgv - core), 0, 1)
            length = float(d.sum())
            if o == "v":
                bars.append({"x": c0, "y": r0, "w": c1 - c0, "h": r1 - r0, "orient": "vertical",
                             "px": length, "thick": medw})
            else:
                bars.append({"x": r0, "y": c0, "w": r1 - r0, "h": c1 - c0, "orient": "horizontal",
                             "px": length, "thick": medw})
    return bars


_TEMPLATE_CACHE: Dict[Tuple[str, bool], List[Tuple[np.ndarray, int]]] = {}


_FONTS: List = []


def _template_fonts(size=96):
    if _FONTS:
        return _FONTS
    fonts = _FONTS
    for name in TEMPLATE_FONTS:
        try:
            fonts.append(ImageFont.truetype(name, size))
        except Exception:
            pass
    try:
        fonts.append(ImageFont.load_default(size))     # Pillow >= 10.1 with FreeType: a scalable sans font
    except Exception:
        pass
    return fonts


def holes(b: np.ndarray) -> int:
    """number of enclosed background regions in a binary glyph"""
    bg, n = ndi.label(~np.pad(b, 1))
    return n - 1


def glyph_vec(b: np.ndarray, size=32) -> Optional[np.ndarray]:
    """binary glyph -> tight crop, padded to a square (aspect kept), resized, zero-mean unit vector"""
    ys, xs = np.nonzero(b)
    if len(ys) == 0:
        return None
    b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = b.shape
    n = max(h, w)
    sq = np.zeros((n, n), np.float32)
    sq[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = b
    v = np.asarray(Image.fromarray((sq * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR), np.float32)
    v = ndi.gaussian_filter(v, 0.8).ravel()
    v -= v.mean()
    nv = np.linalg.norm(v)
    return v / nv if nv > 0 else None


def templates(letter: str) -> List[Tuple[np.ndarray, int]]:
    key = (letter, True)
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]
    out = []
    for f in _template_fonts():
        im = Image.new("L", (160, 160), 0)
        ImageDraw.Draw(im).text((20, 10), letter, fill=255, font=f)
        b = np.asarray(im) > 127
        if b.sum() < 20:
            continue
        v = glyph_vec(b)
        if v is not None:
            out.append((v, holes(b)))
    _TEMPLATE_CACHE[key] = out
    return out


def glyph_score(b: np.ndarray, letter: str) -> float:
    """best cosine similarity of a mark's shape with the letter drawn in the template fonts; a different number
    of enclosed holes (A and D have one, B two, C, E, F none) costs 0.3"""
    v = glyph_vec(b)
    if v is None:
        return 0.0
    nh = holes(b)
    best = 0.0
    for tv, th in templates(letter):
        s = float(v @ tv) - (0.3 if th != nh else 0.0)
        best = max(best, s)
    return best


def find_letter_marks(gray, sat, dark, fg_lab, fg_areas, longer, hfrac=(0.008, 0.08)):
    """small isolated black marks of letter size (not part of anything larger in the foreground)"""
    lab, n = ndi.label(dark, np.ones((3, 3), bool))
    marks = []
    for k, s in enumerate(ndi.find_objects(lab), 1):
        if s is None:
            continue
        h, w = s[0].stop - s[0].start, s[1].stop - s[1].start
        if not (hfrac[0] * longer <= h <= hfrac[1] * longer) or not (0.1 * h <= w <= 1.6 * h):
            continue
        sub = lab[s] == k
        area = int(sub.sum())
        if not (0.12 <= area / (h * w) <= 0.92):
            continue
        if sat[s][sub].mean() > 30:
            continue
        fl = fg_lab[s][sub]
        fl = fl[fl > 0]
        if fl.size == 0:
            continue
        host = np.bincount(fl).argmax()
        if fg_areas[host] > 3.0 * area:                # part of a specimen (an antenna segment, a dark leg)
            continue
        # binarise at half-way between the mark's ink and white, inside its own box
        g = gray[s]
        ink = np.percentile(g[sub], 10)
        b = (g < (ink + 255) / 2) & ndi.binary_dilation(sub, iterations=1)
        marks.append({"x": s[1].start, "y": s[0].start, "w": w, "h": h, "bin": b, "area": area})
    return marks


# ============================================================================ panels
def split_component(cmask: np.ndarray, radii, min_core_frac=0.15) -> Optional[np.ndarray]:
    """split one foreground piece into the bodies it contains: erode (by distance) until two or more large cores
    appear, then give each pixel to the core it reaches first through the piece (geodesic growth)"""
    edt = ndi.distance_transform_edt(np.pad(cmask, 1))[1:-1, 1:-1]
    seeds = None
    for r in radii:
        cores, n = ndi.label(edt >= r)
        if n < 2:
            continue
        a = ndi.sum(np.ones_like(cmask), cores, range(1, n + 1))
        keep = np.nonzero(a >= min_core_frac * a.max())[0] + 1
        if len(keep) >= 2:
            seeds = np.zeros(cmask.shape, np.int32)
            for j, kk in enumerate(keep, 1):
                seeds[cores == kk] = j
            break
    if seeds is None:
        return None
    lab = seeds
    for _ in range(100000):
        grown = ndi.grey_dilation(lab, size=(3, 3))
        new = (lab == 0) & cmask & (grown > 0)
        if not new.any():
            break
        lab[new] = grown[new]
    return lab


def find_panels(fg, n_expected, min_area_px, longer, max_gap_px, split="auto"):
    """label image of specimens (1..n) and their bboxes; pieces too small to be a specimen join the nearest one"""
    warnings = []
    lab, n = ndi.label(fg, np.ones((3, 3), bool))
    if n == 0:
        return np.zeros(fg.shape, np.int32), [], ["no foreground found"]
    areas = ndi.sum(fg, lab, range(1, n + 1))
    big = [k + 1 for k in np.nonzero(areas >= min_area_px)[0]]
    if not big:
        return np.zeros(fg.shape, np.int32), [], [f"no piece of at least {min_area_px} px (--min_area_frac)"]
    med = float(np.median([areas[k - 1] for k in big]))
    big = [k for k in big if areas[k - 1] >= 0.25 * med]
    plab = np.zeros(fg.shape, np.int32)
    slices = ndi.find_objects(lab)
    pieces = []                                   # (slice, mask)
    for k in big:
        s = slices[k - 1]
        pieces.append((s, lab[s] == k))
    # split pieces that hold two or more specimens while the caption names more panels than were found
    if split == "auto" and n_expected and len(pieces) < n_expected:
        radii = [max(2, int(round(f * longer))) for f in (0.004, 0.006, 0.009, 0.013, 0.02, 0.03, 0.045)]
        order = sorted(range(len(pieces)), key=lambda i: -pieces[i][1].sum())
        for i in order:
            if len(pieces) >= n_expected:
                break
            s, m = pieces[i]
            if m.sum() < 1.3 * med:
                continue
            sub = split_component(m, radii)
            if sub is None:
                continue
            k_new = int(sub.max())
            pieces[i] = (s, sub == 1)
            for j in range(2, k_new + 1):
                pieces.append((s, sub == j))
            warnings.append(f"one piece at x={s[1].start}, y={s[0].start} held {k_new} touching specimens; split "
                            f"along the shortest paths through each")
    for j, (s, m) in enumerate(pieces, 1):
        plab[s][m] = j
    # remaining small pieces (detached antennae, legs, setae) join the nearest specimen within max_gap_px
    rest = fg & (plab == 0)
    if rest.any() and pieces:
        dist, (iy, ix) = ndi.distance_transform_edt(plab == 0, return_indices=True)
        near = plab[iy, ix]
        rl, rn = ndi.label(rest, np.ones((3, 3), bool))
        rsl = ndi.find_objects(rl)
        lost = 0
        for k, s in enumerate(rsl, 1):
            m = rl[s] == k
            d = dist[s][m]
            if d.min() <= max_gap_px:
                tgt = np.bincount(near[s][m][d <= d.min() + 1]).argmax()
                plab[s][m] = tgt
            else:
                lost += int(m.sum())
                if m.sum() >= 0.02 * min_area_px:
                    warnings.append(f"a piece of {int(m.sum())} px at x={s[1].start}, y={s[0].start} is farther "
                                    f"than {max_gap_px} px from every specimen; left out")
    boxes = []
    for j, s in enumerate(ndi.find_objects(plab), 1):
        boxes.append((s[1].start, s[0].start, s[1].stop - s[1].start, s[0].stop - s[0].start))
    return plab, boxes, warnings


def _boundary_points(m: np.ndarray, off, maxn=4000):
    b = m & ~ndi.binary_erosion(m)
    ys, xs = np.nonzero(b)
    if len(ys) > maxn:
        k = np.linspace(0, len(ys) - 1, maxn).astype(int)
        ys, xs = ys[k], xs[k]
    return np.c_[xs + off[0], ys + off[1]].astype(np.float32)


def dist_to_panels(points: np.ndarray, plab, boxes) -> np.ndarray:
    """distance from each point (x, y) to each specimen's outline (0 when inside its pixels)"""
    D = np.zeros((len(points), len(boxes)))
    for j, (x, y, w, h) in enumerate(boxes, 1):
        m = plab[y:y + h, x:x + w] == j
        bp = _boundary_points(m, (x, y))
        for i, (px, py) in enumerate(points):
            ix, iy = int(px) - x, int(py) - y
            if 0 <= ix < w and 0 <= iy < h and m[iy, ix]:
                D[i, j - 1] = 0.0
            else:
                D[i, j - 1] = float(np.sqrt(((bp - (px, py)) ** 2).sum(1)).min()) if len(bp) else 1e9
    return D


def positional_order(boxes, mode="column") -> List[int]:
    """panel indices in reading order: by columns (top to bottom, then the next column) or by rows"""
    if not boxes:
        return []
    a, b = (0, 1) if mode == "column" else (1, 0)            # group along x for columns, y for rows
    items = sorted(range(len(boxes)), key=lambda i: boxes[i][a] + boxes[i][a + 2] / 2)
    groups: List[List[int]] = []
    for i in items:
        lo, hi = boxes[i][a], boxes[i][a] + boxes[i][a + 2]
        for g in groups:
            glo = min(boxes[j][a] for j in g); ghi = max(boxes[j][a] + boxes[j][a + 2] for j in g)
            ov = min(hi, ghi) - max(lo, glo)
            if ov >= 0.5 * min(hi - lo, ghi - glo):
                g.append(i)
                break
        else:
            groups.append([i])
    groups.sort(key=lambda g: np.mean([boxes[j][a] + boxes[j][a + 2] / 2 for j in g]))
    out = []
    for g in groups:
        out += sorted(g, key=lambda j: boxes[j][b] + boxes[j][b + 2] / 2)
    return out


# ============================================================================ one plate
def split_plate(path: Path, cap: dict, a, out_dir: Path) -> Tuple[List[dict], dict]:
    stem = path.stem
    plate_row = {"plate": path.name, "status": "", "panels": 0, "caption_letters": " ".join(cap["letters"]),
                 "layout": "", "warnings": ""}
    pw: List[str] = list(cap["warnings"])

    def skipped(why):
        plate_row["status"] = "skipped"
        plate_row["warnings"] = " | ".join([why] + pw)
        print(f"  {path.name}: SKIPPED - {why}")
        return [], plate_row

    if not cap["letters"]:
        return skipped("caption not parsed (no panel letters)")
    rgb = np.asarray(Image.open(path).convert("RGB"))
    H, W = rgb.shape[:2]
    longer = max(H, W)
    a16 = rgb.astype(np.int16)
    mn = a16.min(2)
    sat = (a16.max(2) - mn).astype(np.float32)
    gray = rgb.mean(2).astype(np.float32)
    fg = mn < a.bg_threshold
    fr = max(5, int(0.01 * min(H, W)))
    frame = np.r_[fg[:fr].ravel(), fg[-fr:].ravel(), fg[:, :fr].ravel(), fg[:, -fr:].ravel()]
    if 1 - frame.mean() < a.min_border_white or 1 - fg.mean() < 0.2:
        return skipped(f"background is not white ({100 * (1 - frame.mean()):.0f}% of the border and "
                       f"{100 * (1 - fg.mean()):.0f}% of the plate is background); not a plate on white")
    dark = (gray < a.dark_threshold) & (sat < 40)

    # scale bars, then letter marks, both taken out of the foreground before the specimens are found; the
    # strokes of marks that already read as letters are not bars
    fl0, fn0 = ndi.label(fg, np.ones((3, 3), bool))
    fa0 = np.r_[0, ndi.sum(fg, fl0, range(1, fn0 + 1))] if fn0 else np.zeros(1)
    lboxes = [(m_["x"], m_["y"], m_["w"], m_["h"])
              for m_ in find_letter_marks(gray, sat, dark, fl0, fa0, longer, tuple(a.letter_height_frac))
              if max(glyph_score(m_["bin"], L) for L in cap["letters"]) >= a.glyph_min_score]
    bars = find_bars(gray, sat, fg, dark, longer, orient=a.bar_orientation, letter_boxes=lboxes)
    taken = np.zeros_like(fg)
    for b in bars:
        taken[max(0, b["y"] - 2):b["y"] + b["h"] + 2, max(0, b["x"] - 2):b["x"] + b["w"] + 2] = True
    fg1 = fg & ~taken
    fl, fn = ndi.label(fg1, np.ones((3, 3), bool))
    fareas = np.r_[0, ndi.sum(fg1, fl, range(1, fn + 1))] if fn else np.zeros(1)
    marks = find_letter_marks(gray, sat, dark & ~taken, fl, fareas, longer, tuple(a.letter_height_frac))
    for mk in marks:
        mk["scores"] = {L: glyph_score(mk["bin"], L) for L in cap["letters"]}
    # the marks that read best as the caption's letters (one mark per letter)
    letters = cap["letters"]
    read: Dict[str, dict] = {}
    if marks:
        S = np.array([[mk["scores"][L] for L in letters] for mk in marks])
        ri, ci = linear_sum_assignment(-S)
        for r, c in zip(ri, ci):
            if S[r, c] >= a.glyph_min_score:
                read[letters[c]] = {**marks[r], "score": float(S[r, c])}
        if read:                                           # panel letters on one plate share one size
            hmed = float(np.median([m["h"] for m in read.values()]))
            for L in list(read):
                if abs(read[L]["h"] - hmed) > 0.3 * hmed:
                    pw.append(f"mark read as {L} ({read[L]['h']} px tall) differs in size from the other letters "
                              f"({hmed:.0f} px); not used")
                    del read[L]
    for mk_l in read.values():
        taken[max(0, mk_l["y"] - 2):mk_l["y"] + mk_l["h"] + 2, max(0, mk_l["x"] - 2):mk_l["x"] + mk_l["w"] + 2] = True
    # isolated marks that look like one of the caption's letters but were not accepted are kept out of the
    # specimens too (other small dark pieces stay: they may be a detached tarsus of a black specimen)
    for mk in marks:
        if max(mk["scores"].values(), default=0) >= 0.45:
            taken[max(0, mk["y"] - 1):mk["y"] + mk["h"] + 1, max(0, mk["x"] - 1):mk["x"] + mk["w"] + 1] = True
    fg2 = fg & ~taken

    plab, boxes, w_p = find_panels(fg2, len(letters), int(a.min_area_frac * H * W), longer,
                                   int(a.max_gap_frac * longer), split=a.split)
    pw += w_p
    if not boxes:
        return skipped("no specimens found")
    if len(boxes) != len(letters):
        pw.append(f"{len(boxes)} specimens found but the caption has {len(letters)} letters ({' '.join(letters)})")

    # ---------------------------------------------------------------- letters -> panels
    assign: Dict[int, Tuple[str, str, float]] = {}          # panel index -> (letter, method, score)
    glyph_assign: Dict[int, str] = {}
    if read:
        Ls = sorted(read)
        pts = np.array([[read[L]["x"] + read[L]["w"] / 2, read[L]["y"] + read[L]["h"] / 2] for L in Ls])
        D = dist_to_panels(pts, plab, boxes)
        ri, ci = linear_sum_assignment(D)
        for r, c in zip(ri, ci):
            if D[r, c] <= a.link_max_frac * longer:
                glyph_assign[c] = Ls[r]
            else:
                pw.append(f"letter {Ls[r]} is {D[r, c]:.0f} px from the nearest unclaimed specimen; not linked")
    orders = {m: positional_order(boxes, m) for m in ("column", "row")}

    def by_order(mode):
        return {p: L for p, L in zip(orders[mode], letters)}

    def agrees(mode):
        po = by_order(mode)
        return all(po.get(p) == L for p, L in glyph_assign.items())

    layout = "other"
    for mode in ("column", "row"):
        if len(glyph_assign) >= 2 and agrees(mode):
            layout = mode
            break
    if a.letter_order in ("column", "row"):
        for p, L in by_order(a.letter_order).items():
            assign[p] = (L, a.letter_order, 1.0 if glyph_assign.get(p) == L else 0.0)
        bad = [f"{glyph_assign[p]} read where {a.letter_order} order puts {assign[p][0]}" for p in glyph_assign
               if p in assign and assign[p][0] != glyph_assign[p]]
        if bad:
            pw.append(f"--letter_order {a.letter_order} disagrees with the letters on the plate: " + "; ".join(bad))
    else:
        for p, L in glyph_assign.items():
            assign[p] = (L, "read", read[L]["score"])
        left_p = [p for p in range(len(boxes)) if p not in assign]
        left_L = [L for L in letters if L not in assign_vals(assign)]
        if left_p and left_L:
            if len(left_p) == 1 and len(left_L) == 1:
                assign[left_p[0]] = (left_L[0], "elimination", 0.0)
                pw.append(f"letter {left_L[0]} not read on the plate; given to the one specimen left (elimination)")
            elif layout in ("column", "row") or not glyph_assign:
                mode = layout if layout in ("column", "row") else a.fallback_order
                po = by_order(mode)
                for p in left_p:
                    if p in po and po[p] in left_L:
                        assign[p] = (po[p], mode, 0.0)
                done = [assign[p][0] for p in left_p if p in assign]
                why = (f"which agrees with the {len(glyph_assign)} letters read" if glyph_assign
                       else "no letters could be read on the plate (--fallback_order)")
                if done:
                    pw.append(f"letters {', '.join(done)} not read; assigned by {mode} order, {why}")
            else:
                pw.append(f"letters {', '.join(left_L)} not read and the letters read follow neither column nor "
                          f"row order; those specimens are left unassigned")
    plate_row["layout"] = layout
    missing = [L for L in letters if L not in assign_vals(assign)]
    if missing:
        pw.append(f"letter(s) {', '.join(missing)} of the caption not found on the plate")

    # ---------------------------------------------------------------- scale bars -> panels
    panel_bars: Dict[int, List[dict]] = defaultdict(list)
    if bars:
        bpts = np.array([[b["x"] + b["w"] / 2, b["y"] + b["h"] / 2] for b in bars])
        D = dist_to_panels(bpts, plab, boxes)
        for i, b in enumerate(bars):
            # a bar beside a letter belongs to that letter's specimen; otherwise to the nearest specimen
            p = None
            for pp, L in glyph_assign.items():
                mk = read[L]
                gapx = max(0, max(mk["x"] - (b["x"] + b["w"]), b["x"] - (mk["x"] + mk["w"])))
                gapy = max(0, max(mk["y"] - (b["y"] + b["h"]), b["y"] - (mk["y"] + mk["h"])))
                if max(gapx, gapy) <= 2.5 * mk["h"] and (p is None or max(gapx, gapy) < p[1]):
                    p = (pp, max(gapx, gapy))
            if p is not None:
                b["panel"] = p[0]
            elif D[i].min() <= a.link_max_frac * longer:
                b["panel"] = int(np.argmin(D[i]))
            else:
                b["panel"] = None
                pw.append(f"scale bar at x={b['x']}, y={b['y']} is not near any specimen; not used")
            if b["panel"] is not None:
                panel_bars[b["panel"]].append(b)

    # ---------------------------------------------------------------- per panel: names, scale
    rows = []
    vc = {**DEFAULT_VIEW_CLASS, **a.view_class}
    for p, (x, y, w, h) in enumerate(boxes):
        L, method, score = assign.get(p, ("", "unassigned", 0.0))
        r = {"plate": path.name, "letter": L, "species": cap["species"].get(L) or "" if L else "",
             "view": (cap["views"].get(L) or "") if L else "", "crop_file": "", "bbox_x": x, "bbox_y": y,
             "bbox_w": w, "bbox_h": h, "scale_bar_px": "", "scale_bar_mm": "", "px_per_mm": "",
             "px_per_mm_source": "none", "warnings": [], "mask_file": "", "letter_method": method,
             "letter_score": round(score, 3), "_p": p}
        if not L:
            r["warnings"].append("no letter assigned")
        elif not r["species"]:
            r["warnings"].append("species not given for this letter in the caption")
        r["_cls"] = canon_class(vc.get(r["view"])) if r["view"] else None
        bl = panel_bars.get(p, [])
        if len(bl) > 1:
            r["warnings"].append(f"{len(bl)} scale bars next to this specimen; the longest used")
        if bl:
            b = max(bl, key=lambda q: q["px"])
            r["scale_bar_px"] = round(b["px"], 2)
            mm, why = bar_mm_for(L, r["_cls"], cap["bars"], a.bar_mm)
            if mm:
                r["scale_bar_mm"] = mm
                r["px_per_mm"] = round(b["px"] / mm, 3)
                r["px_per_mm_source"] = "own bar"
            else:
                r["warnings"].append(why)
        rows.append(r)
    if a.bar_scope == "plate":
        own = [r for r in rows if r["px_per_mm_source"] == "own bar"]
        if own:
            vals = [r["px_per_mm"] for r in own]
            if (max(vals) - min(vals)) > 0.02 * np.mean(vals):
                pw.append(f"--bar_scope plate but the bars give {min(vals):.2f}-{max(vals):.2f} px/mm")
            for r in rows:
                if r["px_per_mm_source"] != "own bar":
                    r["px_per_mm"] = round(float(np.mean(vals)), 3)
                    r["px_per_mm_source"] = "plate bar"
    elif a.share_scale:
        # kind of view: its class (habitus / genitalia) or, for a view not in the class map, the view itself;
        # nothing is shared into or out of a panel whose view is unknown
        kind = lambda q: q["_cls"] or (f"view:{q['view']}" if q["view"] else None)    # noqa: E731
        for r in rows:
            if r["px_per_mm_source"] != "none" or not r["letter"] or not r["species"] or kind(r) is None:
                continue
            donors = [d for d in rows if d["px_per_mm_source"] == "own bar" and d["species"] == r["species"]
                      and kind(d) == kind(r) and d is not r]
            if not donors:
                continue
            o = ord(r["letter"])
            d = min(donors, key=lambda q: (abs(ord(q["letter"]) - o), ord(q["letter"]) > o))
            r["px_per_mm"] = d["px_per_mm"]
            r["scale_bar_mm"] = d["scale_bar_mm"]
            r["px_per_mm_source"] = f"shared from {d['letter']} (same species, {d['view'] or 'same kind of view'})"
            la, lb = max(r["bbox_w"], r["bbox_h"]), max(d["bbox_w"], d["bbox_h"])
            if abs(la - lb) > 0.15 * max(la, lb):
                r["warnings"].append(f"px/mm shared from {d['letter']} but the two specimens' lengths differ by "
                                     f"{100 * abs(la - lb) / max(la, lb):.0f}%; magnification may differ")
    for r in rows:
        if r["px_per_mm_source"] == "none" and r["letter"]:
            if not panel_bars.get(r["_p"]):
                why = "no scale bar found for this specimen"
                if not a.share_scale:
                    why += " (--no_share_scale)"
                elif r["species"]:
                    why += (", and no panel of the same species and kind of view (" + (r["_cls"] or
                            f"view '{r['view'] or '?'}' not mapped; see --view_class") + ") has a bar of known length")
                r["warnings"].append(why)
    rows.sort(key=lambda r: (r["letter"] == "", r["letter"], r["_p"]))

    # ---------------------------------------------------------------- crops, masks
    others = (plab > 0)
    for r in rows:
        p = r["_p"]
        x, y, w, h = r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]
        m = plab[y:y + h, x:x + w] == p + 1
        outline = ndi.binary_fill_holes(m)
        e = a.edge_px
        X0, Y0, X1, Y1 = max(0, x - e), max(0, y - e), min(W, x + w + e), min(H, y + h + e)
        big = np.zeros((Y1 - Y0, X1 - X0), bool)
        big[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = outline
        own = np.zeros_like(big)
        own[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = m
        block = (others[Y0:Y1, X0:X1] & ~own) | taken[Y0:Y1, X0:X1]      # other specimens, letters, bars
        if e > 0:
            big = ndi.binary_dilation(big, iterations=e)
        big &= ~block
        pad = a.pad
        canvas = np.full((Y1 - Y0 + 2 * pad, X1 - X0 + 2 * pad, 3), 255, np.uint8)
        sub = rgb[Y0:Y1, X0:X1]
        canvas[pad:pad + Y1 - Y0, pad:pad + X1 - X0][big] = sub[big]
        name = f"{stem}_{r['letter'] or f'panel{p + 1}'}_{slug(r['species'] or 'unknown')}_{slug(r['view'] or 'unknown')}"
        if not r["letter"]:
            name = f"{stem}_panel{p + 1}_unassigned"
        cf = out_dir / f"{name}.png"
        Image.fromarray(canvas).save(cf)
        r["crop_file"] = cf.name
        if a.mask:
            # foreground pixels of the specimen plus pinholes (enclosed, < --mask_hole_frac of its area: glints
            # on the cuticle); background enclosed between legs, antennae and body stays background
            fgm = own.copy()
            hol, hn = ndi.label(ndi.binary_fill_holes(fgm) & ~fgm)
            if hn:
                ha = ndi.sum(np.ones_like(fgm), hol, range(1, hn + 1))
                small = np.r_[False, ha < a.mask_hole_frac * m.sum()]
                fgm |= small[hol]
            mk = np.zeros(canvas.shape[:2], np.uint8)
            mk[pad:pad + Y1 - Y0, pad:pad + X1 - X0][fgm] = 255
            mf = out_dir / f"{name}_mask.png"
            Image.fromarray(mk).save(mf)
            r["mask_file"] = mf.name
    for r in rows:
        r["warnings"] = " | ".join(r["warnings"])
    plate_row["panels"] = len(rows)
    plate_row["status"] = "warnings" if (pw or any(r["warnings"] for r in rows)) else "ok"
    plate_row["warnings"] = " | ".join(pw)
    qc_sheet(rgb, rows, bars, read, pw, out_dir / "qc" / f"{stem}_qc.png", a.qc_px)
    n_ppm = sum(1 for r in rows if r["px_per_mm"] != "")
    print(f"  {path.name}: {len(rows)} specimens, letters {''.join(sorted(r['letter'] for r in rows if r['letter']))}"
          f" ({layout}), {len(bars)} bars, px/mm for {n_ppm}" + (f"; {len(pw)} plate warning(s)" if pw else ""))
    return rows, plate_row


def assign_vals(assign):
    return {v[0] for v in assign.values()}


def bar_mm_for(letter, cls, specs, override) -> Tuple[Optional[float], str]:
    """the length in mm of the scale bar next to a panel, from --bar_mm or the caption"""
    if override:
        if None in override:
            return override[None], ""
        if cls in override:
            return override[cls], ""
    lettered = [s for s in specs if s["letters"] and letter in s["letters"]]
    if lettered:
        s = lettered[0]
        return (s["mm"], "") if s["mm"] else (None, f"scale bar length for {letter} not readable ('{s['raw']}')")
    qual = [s for s in specs if s["cls"] and not s["letters"]]
    plain = [s for s in specs if not s["cls"] and not s["letters"]]
    if qual:
        if cls:
            hit = [s for s in qual if s["cls"] == cls]
            if hit:
                s = hit[0]
                return (s["mm"], "") if s["mm"] else (None, f"{cls} scale bar length not readable ('{s['raw']}'); "
                                                            f"give --bar_mm {cls}=<mm>")
        if len(qual) == 1 and not plain and qual[0]["mm"] and not cls:
            return qual[0]["mm"], ""
        if not plain:
            return None, (f"caption gives {', '.join(s['cls'] for s in qual)} scale bars and the view of this panel "
                          f"is not mapped to either; give --view_class <view>=<{'|'.join(s['cls'] for s in qual)}>")
    if plain:
        mms = {s["mm"] for s in plain if s["mm"]}
        if len(mms) == 1:
            return mms.pop(), ""
        if len(mms) > 1:
            return None, f"caption gives several scale bar lengths ({', '.join(map(str, sorted(mms)))}) not tied to letters"
        return None, f"scale bar length not readable in the caption ('{plain[0]['raw']}'); give --bar_mm"
    return None, "caption gives no scale bar length; give --bar_mm"


# ============================================================================ QC sheet
def _font(size):
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


PALETTE = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14), (23, 190, 207),
           (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34)]


def qc_sheet(rgb, rows, bars, read, plate_warnings, out_path: Path, qc_px=1100):
    H, W = rgb.shape[:2]
    s = min(1.0, qc_px / max(H, W))
    im = Image.fromarray(rgb).resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS)
    f = _font(max(11, int(qc_px / 75)))
    fs = _font(max(10, int(qc_px / 95)))
    lines = []
    for w in plate_warnings:
        lines += _wrap("! " + w, 130)
    for r in rows:
        if r["warnings"]:
            lines += _wrap(f"! {r['letter'] or 'unassigned'}: {r['warnings']}", 130)
    lh = getattr(fs, "size", 11) + 4
    foot = (len(lines) + 1) * lh if lines else 0
    sheet = Image.new("RGB", (im.width, im.height + foot), "white")
    sheet.paste(im, (0, 0))
    d = ImageDraw.Draw(sheet, "RGBA")
    for L, mk in read.items():
        d.rectangle([mk["x"] * s - 2, mk["y"] * s - 2, (mk["x"] + mk["w"]) * s + 2, (mk["y"] + mk["h"]) * s + 2],
                    outline=(0, 90, 255, 255), width=2)
    for b in bars:
        d.rectangle([b["x"] * s - 1, b["y"] * s - 1, (b["x"] + b["w"]) * s + 1, (b["y"] + b["h"]) * s + 1],
                    outline=(230, 0, 0, 255), width=2)
        d.text(((b["x"] + b["w"]) * s + 3, b["y"] * s), f"{b['px']:.1f}px", fill=(230, 0, 0, 255), font=fs)
    for i, r in enumerate(rows):
        col = PALETTE[i % len(PALETTE)] if r["letter"] else (120, 120, 120)
        x0, y0 = r["bbox_x"] * s, r["bbox_y"] * s
        x1, y1 = (r["bbox_x"] + r["bbox_w"]) * s, (r["bbox_y"] + r["bbox_h"]) * s
        d.rectangle([x0, y0, x1, y1], outline=col + (255,), width=3)
        sp = r["species"] or "species?"
        sp = sp if len(sp) <= 26 else re.sub(r"^(\w)\w+ ", r"\1. ", sp)
        t1 = f"{r['letter'] or '?'}  {sp}"
        t2 = f"{r['view'] or 'view?'}  [{r['letter_method']}]"
        src = r["px_per_mm_source"]
        src = "own bar" if src == "own bar" else (f"from {src.split()[2]}" if src.startswith("shared") else src)
        t3 = (f"{r['px_per_mm']:.2f} px/mm ({src})" if r["px_per_mm"] != "" else "no px/mm")
        ty = y0 + 3
        for t, ff in ((t1, f), (t2, fs), (t3, fs)):
            bb = d.textbbox((x0 + 4, ty), t, font=ff)
            d.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=(255, 255, 255, 215))
            d.text((x0 + 4, ty), t, fill=col + (255,), font=ff)
            ty = bb[3] + 3
    y = im.height + 4
    for ln in lines:
        d.text((6, y), ln, fill=(170, 0, 0), font=fs)
        y += lh
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _wrap(t, n):
    out, cur = [], ""
    for w in t.split(" "):
        if cur and len(cur) + len(w) + 1 > n:
            out.append(cur)
            cur = "    " + w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        out.append(cur)
    return out


# ============================================================================ plates <-> captions
def list_plates(items) -> List[Path]:
    out = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT and not q.name.startswith("._"))
        elif p.exists():
            out.append(p)
        else:
            print(f"warning: {it} not found", file=sys.stderr)
    return out


def _norm_key(k: str) -> str:
    k = str(k).strip()
    return k.lstrip("0") or "0" if k.isdigit() else k.lower()


def match_captions(plates: List[Path], csv_path, key_col, caption_col, key_regex, where) -> Tuple[Dict[Path, str], List[str]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}, [f"{csv_path} has no rows"]
    cols = list(rows[0].keys())
    for k, v in where:
        if k not in cols:
            raise SystemExit(f"--where column '{k}' not in {csv_path} (columns: {', '.join(cols)})")
        rows = [r for r in rows if str(r[k]).strip().lower() == v.strip().lower()]
    if caption_col not in cols:
        raise SystemExit(f"--caption_col '{caption_col}' not in {csv_path} (columns: {', '.join(cols)})")
    if key_col is None:
        key_col = next((c for c in ("file", "filename", "file_name", "image", "plate", "fig", "figure") if c in cols), None)
        if key_col is None:
            raise SystemExit(f"give --key_col (columns: {', '.join(cols)})")
    if key_col not in cols:
        raise SystemExit(f"--key_col '{key_col}' not in {csv_path} (columns: {', '.join(cols)})")
    vals = [str(r[key_col]).strip() for r in rows]
    by_file = all(Path(v).suffix.lower() in IMG_EXT for v in vals if v)
    table: Dict[str, List[str]] = defaultdict(list)
    for r, v in zip(rows, vals):
        table[Path(v).stem.lower() if by_file else _norm_key(v)].append(r[caption_col])
    notes, out, used = [], {}, set()
    rx = re.compile(key_regex)
    for p in plates:
        if by_file:
            key = p.stem.lower()
        else:
            m = rx.search(p.stem)
            key = _norm_key(m.group(1) if m and m.groups() else (m.group(0) if m else ""))
        caps = table.get(key)
        if not caps:
            notes.append(f"{p.name}: no caption row for key '{key}' ({key_col}); skipped")
            continue
        if len(caps) > 1:
            notes.append(f"{p.name}: {len(caps)} caption rows for key '{key}'; the first used")
        out[p] = caps[0]
        used.add(key)
    for k in table:
        if k not in used:
            notes.append(f"caption row {key_col}={k} matches no plate file")
    return out, notes


# ============================================================================ main
def _kv_float(items):
    out = {}
    for it in items or []:
        if "=" in it:
            k, v = it.split("=", 1)
            out[canon_class(k)] = float(v)
        else:
            out[None] = float(it)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plates", nargs="+", required=True, help="plate images and/or folders of them")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--captions", help="CSV with a caption per figure (see --key_col, --caption_col, --where)")
    src.add_argument("--caption", help="one caption for the plate(s) given")
    ap.add_argument("--caption_col", default="caption", help="caption column of --captions (default caption)")
    ap.add_argument("--key_col", default=None,
                    help="column tying a caption row to a plate: file names, or a figure number matched with "
                         "--key_regex (default: the first of file/filename/image/plate/fig/figure)")
    ap.add_argument("--caption_file_col", dest="key_col", help=argparse.SUPPRESS)
    ap.add_argument("--key_regex", default=r"(?i)fig(?:ure)?[\s_-]*0*(\d+)",
                    help="regex with one group taking the key out of a plate's file name (default: the number "
                         "after 'fig'; leading zeros ignored)")
    ap.add_argument("--where", action="append", default=[], metavar="COL=VALUE",
                    help="use only caption rows with this value (repeatable), e.g. habitus=True")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--genus", default=None, help="genus for abbreviations ('P.') not spelled out in a caption")
    ap.add_argument("--bar_mm", action="append", default=[], metavar="MM | CLASS=MM",
                    help="scale bar length in mm, overriding the caption: one value for every bar, or per kind "
                         "of view, e.g. --bar_mm habitus=2 --bar_mm aedeagus=0.5")
    ap.add_argument("--view_class", action="append", default=[], metavar="VIEW=CLASS",
                    help="which caption scale bar a view uses when the caption gives qualified bars, e.g. "
                         "lateral=aedeagus (defaults: dorsal/ventral/... = habitus; paramere/aedeagus/... = "
                         "genitalia)")
    ap.add_argument("--no_share_scale", dest="share_scale", action="store_false",
                    help="do not carry px/mm to a panel without a bar from the same species' panel with one")
    ap.add_argument("--bar_scope", default="panel", choices=["panel", "plate"],
                    help="panel (default): a bar measures the specimen beside it; plate: the plate's bar(s) "
                         "measure every panel (one magnification for the whole plate)")
    ap.add_argument("--letter_order", default="auto", choices=["auto", "column", "row"],
                    help="auto: read the letters on the plate; column / row: assign letters by position")
    ap.add_argument("--fallback_order", default="column", choices=["column", "row"],
                    help="positional order used by auto when no letter can be read on the plate")
    ap.add_argument("--lowercase_letters", action="store_true", help="panel letters are lower case (a, b, c)")
    ap.add_argument("--bg_threshold", type=int, default=235,
                    help="a pixel is background when its darkest channel is >= this (default 235)")
    ap.add_argument("--dark_threshold", type=int, default=140,
                    help="grey level below which letters and scale bars are ink (default 140)")
    ap.add_argument("--min_area_frac", type=float, default=0.005,
                    help="smallest specimen as a fraction of the plate area (default 0.005)")
    ap.add_argument("--max_gap_frac", type=float, default=0.02,
                    help="a detached piece joins a specimen within this fraction of the plate's longer side")
    ap.add_argument("--min_border_white", type=float, default=0.85,
                    help="fraction of the plate's border that must be background (else the file is skipped)")
    ap.add_argument("--letter_height_frac", type=float, nargs=2, default=[0.008, 0.08], metavar=("MIN", "MAX"),
                    help="letter height range as fractions of the plate's longer side")
    ap.add_argument("--glyph_min_score", type=float, default=0.6, help="lowest shape match accepted as a letter")
    ap.add_argument("--link_max_frac", type=float, default=0.12,
                    help="a letter or bar farther than this fraction of the longer side from every specimen is "
                         "not linked")
    ap.add_argument("--bar_orientation", default="both", choices=["both", "vertical", "horizontal"])
    ap.add_argument("--split", default="auto", choices=["auto", "never"],
                    help="auto: split a piece holding touching specimens when the caption names more panels")
    ap.add_argument("--pad", type=int, default=20, help="white margin around each crop (px)")
    ap.add_argument("--edge_px", type=int, default=2, help="keep this many px of soft edge around each specimen")
    ap.add_argument("--mask", action="store_true", help="also write a binary foreground mask per crop")
    ap.add_argument("--mask_hole_frac", type=float, default=0.001,
                    help="enclosed holes smaller than this fraction of the specimen are filled in the mask")
    ap.add_argument("--qc_px", type=int, default=1100, help="longer side of the QC sheets")
    a = ap.parse_args(argv)
    a.bar_mm = _kv_float(a.bar_mm)
    vcl = {}
    for it in a.view_class:
        if "=" not in it:
            ap.error(f"--view_class needs VIEW=CLASS, got '{it}'")
        k, v = it.split("=", 1)
        vcl[k.strip().lower()] = v.strip().lower()
    a.view_class = vcl
    where = []
    for it in a.where:
        if "=" not in it:
            ap.error(f"--where needs COL=VALUE, got '{it}'")
        where.append(tuple(it.split("=", 1)))

    plates = list_plates(a.plates)
    if not plates:
        raise SystemExit("no plate images found")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qc").mkdir(exist_ok=True)
    notes = []
    if a.caption:
        captions = {p: a.caption for p in plates}
        if len(plates) > 1:
            notes.append(f"--caption used for all {len(plates)} plates")
    else:
        captions, notes = match_captions(plates, a.captions, a.key_col, a.caption_col, a.key_regex, where)
    unmatched = [n_ for n_ in notes if n_.startswith("caption row ")]
    for n_ in notes:
        if n_ not in unmatched:
            print("note:", n_)
    if unmatched:
        print(f"note: {len(unmatched)} caption row(s) match no plate file given: "
              + ", ".join(n_.split()[2] for n_ in unmatched))
    all_rows, plate_rows = [], []
    print(f"Splitting {len(captions)} plate(s) -> {out_dir}")
    for p in plates:
        if p not in captions:
            plate_rows.append({"plate": p.name, "status": "skipped", "panels": 0, "caption_letters": "",
                               "layout": "", "warnings": "no caption"})
            continue
        cap = parse_caption(captions[p], genus=a.genus, lower=a.lowercase_letters)
        try:
            rows, prow = split_plate(p, cap, a, out_dir)
        except Exception as e:                    # one bad file must not stop the batch; it is reported
            rows, prow = [], {"plate": p.name, "status": "skipped", "panels": 0,
                              "caption_letters": " ".join(cap["letters"]), "layout": "",
                              "warnings": f"error: {type(e).__name__}: {e}"}
            print(f"  {p.name}: SKIPPED - error {type(e).__name__}: {e}")
        all_rows += rows
        plate_rows.append(prow)
    cols = ["plate", "letter", "species", "view", "crop_file", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "scale_bar_px", "scale_bar_mm", "px_per_mm", "px_per_mm_source", "warnings", "mask_file",
            "letter_method", "letter_score"]
    with open(out_dir / "plate_split_panels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    with open(out_dir / "plate_split_plates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["plate", "status", "panels", "caption_letters", "layout", "warnings"])
        w.writeheader()
        for r in plate_rows:
            w.writerow(r)
    st = defaultdict(int)
    for r in plate_rows:
        st[r["status"]] += 1
    print(f"{len(all_rows)} specimens from {len(plate_rows)} plate(s): "
          + ", ".join(f"{v} {k}" for k, v in sorted(st.items())))
    print(f"wrote {out_dir / 'plate_split_panels.csv'}, {out_dir / 'plate_split_plates.csv'}, {out_dir / 'qc'}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
