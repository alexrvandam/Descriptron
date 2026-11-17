#!/usr/bin/env python3
"""
phylorag_v1.py - PhyloRAG: morphological character matrices from the taxonomic literature
=========================================================================================

PhyloRAG reads published descriptions (BioSysLit / Plazi TreatmentBank treatments on Zenodo, your own PDFs or
text files) and scores a fixed list of characters for a list of taxa with a language model. A state enters the
matrix only if it can be traced to a sentence about the taxon itself and an expert has accepted it. The result
is a NEXUS matrix that can be analysed on its own or combined with molecular data (total evidence, tip dating),
with a provenance table that gives the source and the verbatim quote of every cell.

It generalises the pipeline written for the Scolopendromorpha UCE study (centipede project scripts
biosyslit_morph_rag.py, claude_rag_morph_extraction_v2.py, audit_rag_morph_scores_v1.py,
merge_audited_rag_states_v1.py, build_noproxy_matrix_v1.py). Its audit showed what a model's output needs before
it can enter a matrix, and each rule below answers one failure seen there:

  sources   only treatments of the taxon (or a synonym): BioSysLit records whose title begins with its name, PDFs
            and text files whose title or file name contains it, and sources you assign in the taxa file; figure,
            table and article records are left out by default (in that study, figure captions of Scolopendra
            alternans and Cormocephalus guildingii had been retrieved for Cryptops amicitia because the title of
            the paper they came from named it)
  quotes    every state needs a verbatim quote, and the quote must be found in the source text (exact, or a
            difflib ratio >= --near over a sliding window); a third of the quotes of the first, unchecked
            extraction could not be found
  silence   never scored from absence ("not mentioned" is not a state); no knowledge of the genus or family
  states    a state must be an allowed state; counts and measurements must contain the numbers they report
  flags     states whose quote follows the name of another taxon, names another taxon, compares taxa, comes
            from a source that is not a treatment of the taxon, or disagrees with another quote, are flagged
            for the reviewer (in that study 19 states quoted the species' own treatment but described a
            different character, and 29 came from treatments of other species or of a genus)
  review    the model proposes, the taxonomist decides: export uses only rows marked "accept" in the review sheet
  merging   an existing matrix is never overwritten: only its missing cells (?) are filled; disagreements are
            logged, not applied

Subcommands
  init     characters file (JSON) from an existing NEXUS matrix (MorphoBank, Mesquite, PAUP), or example files
  fetch    literature: BioSysLit treatments by Zenodo keyword and/or by taxon name, plus PDFs and text files
  extract  the model scores each taxon from its own sources; every state is checked without a model
  audit    re-run the checks (no model) on any states table: an edited review sheet, a hand-coded table, an old run
  export   accepted states -> NEXUS + CSV + provenance (+ counts/measurements table); optionally fill only the
           missing cells of an existing matrix and use the taxon names of a molecular alignment or tree

Input files
  characters.json  [{"id": 1, "description": "Shape of head capsule", "states": {"0": "domed", "1": "flattened"},
                     "ordered": false}, {"id": 2, "description": "Number of antennal articles", "type": "meristic"},
                    {"id": 3, "description": "Body length", "type": "continuous", "unit": "mm"}]
                   type: discrete (default) | meristic (a count or range of counts) | continuous (number + unit).
                   Discrete characters become the NEXUS columns in the order of the file; counts and measurements
                   go to a separate table. `init --from-nexus` writes this file from a published matrix.
  taxa.tsv         columns: taxon (required; "Genus epithet"), tip_label (the name in your matrix or alignment),
                   synonyms (;-separated names whose treatments also count as the taxon's own), sources
                   (;-separated source ids or PDF/text file names that describe the taxon, e.g. a monograph).
                   A plain list of names, one per line, also works.
  sources.json     written by `fetch`: [{"id", "title", "doi", "text", "kind"}]; the treatment databases of the
                   centipede project (field description_text) are read too.

Examples
  python phylorag_v1.py init --from-nexus published_matrix.nex --out characters.json
  python phylorag_v1.py fetch --keywords Scolopendromorpha --taxa taxa.tsv --names --pdf-dir pdfs/ --out sources.json
  python phylorag_v1.py extract --characters characters.json --taxa taxa.tsv --sources sources.json \\
      --out-dir run1 --group centipede --llm-backend claude-code            # add --dry-run to see the sources first
  #   ... open run1/review_sheet.tsv, write accept / reject in the "decision" column ...
  python phylorag_v1.py export --review run1/review_sheet.tsv --characters characters.json --taxa taxa.tsv \\
      --out-dir run1/export --base-nexus published_matrix.nex --molecular uce_75p.phy --add-missing-taxa

Models: --llm-backend claude-code (a Claude subscription through `claude -p`) or api (an API key; any endpoint with
the Anthropic messages API via --api-base-url), through Descriptron's biorag_llm_backend. Every call is logged
(llm_calls.jsonl) with the model that actually answered, and every reply is saved in raw/, so a run can be audited
again, or re-read with other thresholds (--replay), without calling a model. audit and export never call a model.
"""

import argparse
import bisect
import csv
import difflib
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

VERSION = "1.0"
HERE = Path(__file__).resolve().parent
ZENODO_API = "https://zenodo.org/api/records"
BIOSYSLIT = "biosyslit"
USER_AGENT = "Descriptron-PhyloRAG/1.0"
DEFAULT_MODEL = "claude-opus-5-5"

ACCEPT = {"accept", "accepted", "a", "yes", "y", "keep", "k", "1", "true", "x", "ok"}
REJECT = {"reject", "rejected", "r", "no", "n", "drop", "0", "false"}
FIGURE_TITLE = re.compile(r"^\s*(fig(ure)?s?|plates?|tab(le)?s?|maps?)\b\.?\s*[0-9IVXivx]", re.I)
COMPARATIVE = re.compile(r"\b(differs?|differing|unlike|than in|than that|than those|compared|similar to|as in|"
                         r"resembl\w*|in contrast|vs|versus)\b", re.I)
OWN_WINDOW = 400   # in a taxon's own treatment, another name counts only this close before the quote (characters)
NOT_GENUS = {"key", "keys", "notes", "note", "description", "diagnosis", "remarks", "distribution", "figure", "fig",
             "figs", "table", "plate", "material", "synonymy", "checklist", "catalogue", "introduction", "the", "a",
             "new", "family", "genus", "order", "subfamily", "tribe", "redescription", "type", "types", "list"}
NOT_EPITHET = {"to", "of", "the", "and", "in", "from", "for", "with", "on", "at", "by", "species", "new", "gen",
               "sp", "spp", "nov", "var", "subsp", "sensu", "cf", "aff", "et", "al", "is", "was", "are", "has",
               "have", "group", "complex", "genus", "family", "nomen", "nudum", "auct"}
STATE_FIELDS = ["decision", "note", "taxon", "label", "char_id", "char_pos", "character", "type", "state",
                "state_label", "value", "unit", "evidence", "quote_match", "quote_score", "source_id",
                "source_title", "source_doi", "own_source", "flags", "chunk"]


# ════════════════════════════════════════════════════════════════════════ text ══
def norm(s):
    """For matching only: HTML entities decoded, lower case, every dash and odd punctuation to a space, quotes
    unified, whitespace collapsed. Dashes become spaces on both sides because sources differ in how they carry
    them ('1&ndash;19', '1–19', '1-19', and '1 19' where an older reader dropped the entity)."""
    s = html.unescape(str(s))
    s = re.sub(r"&[a-z]+;", " ", s).lower()
    s = re.sub(r"[\-‐‑‒–—―−]", " ", s).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\.\,\;\:\(\)\'/]", " ", s)).strip()


def find_quote(quote, src, near):
    """(status, score, position) of a quote in a normalised source text: exact, near (best difflib ratio over a
    sliding window >= near), absent, or too short (< 8 characters). The window is 20% longer than the quote and
    moves by a quarter of its length, as in the centipede audit (claude_rag_morph_extraction_v2.py), so a perfect
    alignment scores about 0.91 and 0.90 means near-verbatim."""
    q = norm(quote)
    if len(q) < 8:
        return "too short", 0.0, -1
    i = src.find(q)
    if i >= 0:
        return "exact", 1.0, i
    n = len(q)
    best, pos = 0.0, -1
    sm = difflib.SequenceMatcher(None, q, "", autojunk=False)
    for j in range(0, max(1, len(src) - n + 1), max(1, n // 4)):
        sm.set_seq2(src[j:j + n + n // 5])
        r = sm.ratio()
        if r > best:
            best, pos = r, j
        if best >= 0.99:
            break
    return ("near" if best >= near else "absent"), round(best, 3), pos


def numbers_in(s):
    return [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", str(s))]


def parse_value(v):
    """'21' -> (21, 21); '19-23' / '19 to 23' / '2.5–3.1' -> (min, max); else None."""
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:(?:-|–|—|to)\s*(\d+(?:[.,]\d+)?))?\s*[a-zA-Zµ]*\s*$", str(v))
    if not m:
        return None
    lo = float(m.group(1).replace(",", "."))
    hi = float(m.group(2).replace(",", ".")) if m.group(2) else lo
    return (min(lo, hi), max(lo, hi))


def fmt_num(x):
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def clean_text(s):
    """Source text as the model sees it: entities decoded ('&ndash;' -> '–'), whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"&[a-z]+;", " ", html.unescape(str(s)))).strip()


def slug(s, n=60):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")[:n] or "x"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_tsv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_tsv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = "\t" if "\t" in sample.split("\n")[0] else ","
        return list(csv.DictReader(f, delimiter=delim))


# ═════════════════════════════════════════════════════════════════════ NEXUS ══
_TOK = re.compile(r"'(?:[^']|'')*'|\[[^\]]*\]|,|/|;|[^\s,/;'\[]+")


def _tokens(s):
    out = []
    for t in _TOK.findall(s):
        if t.startswith("["):
            continue
        out.append(t[1:-1].replace("''", "'") if t.startswith("'") else t)
    return out


def _blocks(text):
    """{block name (upper): body} for every BEGIN ...; END; block (later blocks of the same name are appended)."""
    out = {}
    for m in re.finditer(r"begin\s+(\w+)\s*;(.*?)\bend(?:block)?\s*;", text, re.I | re.S):
        out[m.group(1).upper()] = out.get(m.group(1).upper(), "") + "\n" + m.group(2)
    return out


def _command(body, name):
    """Text of a NEXUS command (e.g. CHARLABELS) up to its terminating semicolon (quotes and comments respected)."""
    m = re.search(r"(?<![A-Za-z])" + name + r"(?![A-Za-z])", body, re.I)
    if not m:
        return None
    i, inq, depth = m.end(), False, 0
    while i < len(body):
        c = body[i]
        if c == "'" and depth == 0:
            inq = not inq
        elif c == "[" and not inq:
            depth += 1
        elif c == "]" and not inq and depth:
            depth -= 1
        elif c == ";" and not inq and depth == 0:
            return body[m.end():i]
        i += 1
    return body[m.end():]


def _ranges(s, n):
    out = set()
    for a, b in re.findall(r"(\d+)\s*(?:-\s*(\d+|\.))?", s):
        lo = int(a)
        hi = n if b == "." else int(b) if b else lo
        out.update(range(lo, hi + 1))
    return out


def characters_from_nexus(path):
    """Characters (1-based ids = column numbers) from CHARLABELS + STATELABELS (MorphoBank) or CHARSTATELABELS
    (Mesquite, PAUP); ordered characters from TYPESET / CTYPE 'ord:' lists."""
    text = open(path, encoding="utf-8", errors="replace").read()
    bl = _blocks(text)
    body = bl.get("CHARACTERS") or bl.get("DATA") or ""
    m = re.search(r"nchar\s*=\s*(\d+)", body, re.I)
    if not m:
        raise SystemExit(f"{path}: no NCHAR in a CHARACTERS/DATA block")
    n = int(m.group(1))
    labels, states = {}, {}
    csl = _command(body, "CHARSTATELABELS")
    if csl is not None:
        entry = []
        for t in _tokens(csl) + [","]:
            if t == ",":
                if entry and entry[0].isdigit():
                    k = int(entry[0])
                    rest = entry[1:]
                    if "/" in rest:
                        j = rest.index("/")
                        labels[k] = " ".join(rest[:j]).strip()
                        states[k] = rest[j + 1:]
                    else:
                        labels[k] = " ".join(rest).strip()
                entry = []
            else:
                entry.append(t)
    else:
        cl = _command(body, "CHARLABELS")
        if cl is not None:
            for k, t in enumerate([t for t in _tokens(cl) if t not in (",", "/")], 1):
                labels[k] = t
        sl = _command(body, "STATELABELS")
        if sl is not None:
            entry = []
            for t in _tokens(sl) + [","]:
                if t == ",":
                    if entry and entry[0].isdigit():
                        states[int(entry[0])] = entry[1:]
                    entry = []
                else:
                    entry.append(t)
    ordered = set()
    for b in (bl.get("ASSUMPTIONS", ""), bl.get("MESQUITECHARMODELS", ""), body):
        for mm in re.finditer(r"(?<![A-Za-z])ord\s*:\s*([0-9\s\-\.]+)", b, re.I):
            ordered |= _ranges(mm.group(1), n)
    chars = []
    for k in range(1, n + 1):
        lab = labels.get(k, f"character {k}")
        st = states.get(k, [])
        chars.append({"id": k, "name": slug(lab, 40).lower(), "description": lab, "type": "discrete",
                      "states": {str(i): s for i, s in enumerate(st)}, "ordered": k in ordered})
    return chars


def _cell_tokens(seq):
    toks = []
    for t in re.findall(r"\([^)]*\)|\{[^}]*\}|\S", seq):
        if t[0] in "({":
            inner = sorted(set(re.sub(r"[\s,&/]", "", t[1:-1])))
            t = "(" + "".join(inner) + ")" if len(inner) > 1 else (inner[0] if inner else "?")
        toks.append(t)
    return toks


def read_nexus_matrix(path):
    """(labels in order, {label: cell tokens}, nchar). Polymorphisms written with spaces ('(0 1)') or braces are
    one cell each (a matrix reader that split them shifted every later cell of the row)."""
    text = open(path, encoding="utf-8", errors="replace").read()
    bl = _blocks(text)
    body = bl.get("CHARACTERS") or bl.get("DATA") or ""
    m = re.search(r"nchar\s*=\s*(\d+)", body, re.I)
    nchar = int(m.group(1)) if m else None
    mat = _command(body, "MATRIX")
    if mat is None:
        raise SystemExit(f"{path}: no MATRIX")
    rows = OrderedDict()
    for line in mat.split("\n"):
        line = re.sub(r"\[[^\]]*\]", "", line).strip()
        if not line:
            continue
        mm = re.match(r"('(?:[^']|'')*'|\S+)\s+(.*)$", line)
        if not mm:
            continue
        lab = mm.group(1)
        lab = lab[1:-1].replace("''", "'") if lab.startswith("'") else lab
        rows.setdefault(lab, []).extend(_cell_tokens(mm.group(2)))
    return list(rows), rows, nchar


def nexus_quote(s):
    s = str(s)
    return s if re.fullmatch(r"[A-Za-z0-9_.\-]+", s) else "'" + s.replace("'", "''") + "'"


def write_nexus(path, labels, rows, chars, comment=""):
    codes = sorted({c for ch in chars for c in ch["states"]} | {t for r in rows.values() for cell in r
                    for t in re.sub(r"[()?\-]", "", cell)}, key=lambda c: (not c.isdigit(), c))
    with open(path, "w", encoding="utf-8") as f:
        f.write("#NEXUS\n")
        if comment:
            f.write("[ " + comment.replace("[", "(").replace("]", ")") + " ]\n")
        f.write(f"BEGIN TAXA;\n\tDIMENSIONS NTAX={len(labels)};\n\tTAXLABELS\n")
        for lab in labels:
            f.write(f"\t\t{nexus_quote(lab)}\n")
        f.write("\t;\nEND;\n\nBEGIN CHARACTERS;\n")
        f.write(f"\tDIMENSIONS NCHAR={len(chars)};\n")
        f.write(f"\tFORMAT DATATYPE=STANDARD GAP=- MISSING=? SYMBOLS=\"{' '.join(codes)}\";\n\tCHARSTATELABELS\n")
        ent = []
        for k, ch in enumerate(chars, 1):
            st = ch["states"]
            ordered_codes = sorted(st, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else c))
            labs = []
            if ordered_codes and all(c.isdigit() for c in ordered_codes):
                for i in range(int(ordered_codes[-1]) + 1):
                    labs.append(st.get(str(i), f"state {i}"))
            else:
                labs = [st[c] for c in ordered_codes]
            ent.append(f"\t\t{k} {nexus_quote(ch['description'])}"
                       + (" / " + " ".join(nexus_quote(x) for x in labs) if labs else ""))
        f.write(",\n".join(ent) + "\n\t;\n\tMATRIX\n")
        w = max([len(nexus_quote(x)) for x in labels] + [10]) + 2
        for lab in labels:
            f.write(f"\t{nexus_quote(lab).ljust(w)}{''.join(rows[lab])}\n")
        f.write("\t;\nEND;\n")
        ords = [str(k) for k, ch in enumerate(chars, 1) if ch.get("ordered")]
        if ords:
            f.write(f"\nBEGIN ASSUMPTIONS;\n\tTYPESET * UNTITLED = unord: 1-{len(chars)};\n"
                    f"\tTYPESET * default = ORD: {' '.join(ords)};\nEND;\n")


def molecular_labels(path):
    """Taxon names of a FASTA, PHYLIP, NEXUS alignment or a Newick tree."""
    text = open(path, encoding="utf-8", errors="replace").read()
    s = text.lstrip()
    if s.startswith(">"):
        return [ln[1:].split()[0] for ln in text.splitlines() if ln.startswith(">") and ln[1:].split()]
    if s[:6].upper() == "#NEXUS":
        bl = _blocks(text)
        if "TAXA" in bl and _command(bl["TAXA"], "TAXLABELS") is not None:
            return [t for t in _tokens(_command(bl["TAXA"], "TAXLABELS")) if t not in (",", "/")]
        if "TREES" in bl:
            tr = _command(bl["TREES"], "TREE")
            tips = _newick_tips(tr.split("=", 1)[1]) if tr and "=" in tr else []
            tl = _command(bl["TREES"], "TRANSLATE")
            if tl is not None:
                tk = [t for t in _tokens(tl) if t != ","]
                tmap = dict(zip(tk[0::2], tk[1::2]))
                tips = [tmap.get(t, t) for t in tips]
            return tips
        return read_nexus_matrix(path)[0]
    if s.startswith("("):
        return _newick_tips(s)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and re.fullmatch(r"\s*\d+\s+\d+.*", lines[0]):
        ntax = int(lines[0].split()[0])
        return [ln.split()[0] for ln in lines[1:ntax + 1]]
    raise SystemExit(f"{path}: not FASTA, PHYLIP, NEXUS or Newick")


def _newick_tips(s):
    s = re.sub(r"\[[^\]]*\]", "", s)
    out = []
    for m in re.finditer(r"[(,]\s*('(?:[^']|'')*'|[^():,;\s]+)", s):
        t = m.group(1)
        out.append(t[1:-1].replace("''", "'") if t.startswith("'") else t)
    return out


def name_key(s):
    return re.sub(r"\s+", " ", str(s).replace("_", " ").strip().strip("'\"").lower())


# ══════════════════════════════════════════════════════════════════════ inputs ══
def load_characters(path):
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("characters", list(raw.values()))
    chars, seen = [], set()
    for c in raw:
        cid = str(c["id"])
        if cid in seen:
            raise SystemExit(f"{path}: character id {cid} is used twice")
        seen.add(cid)
        typ = str(c.get("type", "discrete")).lower()
        if typ not in ("discrete", "meristic", "continuous"):
            raise SystemExit(f"{path}: character {cid} has type '{typ}' (use discrete, meristic or continuous)")
        st = {str(k): str(v) for k, v in (c.get("states") or {}).items()}
        if typ == "discrete" and not st:
            raise SystemExit(f"{path}: discrete character {cid} has no states")
        chars.append({"id": cid, "name": c.get("name", ""), "description": c.get("description") or c.get("name", ""),
                      "type": typ, "states": st, "ordered": bool(c.get("ordered", False)),
                      "unit": c.get("unit", "")})
    pos = 0
    for c in chars:
        if c["type"] == "discrete":
            pos += 1
            c["pos"] = pos
        else:
            c["pos"] = ""
    return chars


def load_taxa(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        raw = json.load(open(p, encoding="utf-8"))
        items = raw.items() if isinstance(raw, dict) else enumerate(raw)
        rows = []
        for k, v in items:
            v = v if isinstance(v, dict) else {"taxon": v}
            syn = v.get("synonyms") or [x for x in v.get("search_terms", []) if len(str(x).split()) >= 2]
            rows.append({"taxon": v.get("taxon") or v.get("species"), "tip_label": v.get("tip_label") or
                         (k if isinstance(k, str) else ""), "synonyms": syn, "sources": v.get("sources", [])})
    else:
        lines = open(p, encoding="utf-8-sig").read().splitlines()
        if lines and re.split(r"[\t,]", lines[0])[0].strip().lower() == "taxon":
            rows = read_tsv(p)
        else:
            rows = [{"taxon": ln.strip()} for ln in lines if ln.strip() and not ln.startswith("#")]
    out, seen = [], set()
    for r in rows:
        t = re.sub(r"\s+", " ", str(r.get("taxon") or "").strip())
        if not t or t in seen:
            continue
        seen.add(t)
        sp = lambda x: [y.strip() for y in (x if isinstance(x, list) else str(x or "").split(";")) if y.strip()]
        out.append({"taxon": t, "tip_label": str(r.get("tip_label") or "").strip(),
                    "synonyms": [s for s in sp(r.get("synonyms")) if s != t], "sources": sp(r.get("sources"))})
    if not out:
        raise SystemExit(f"{path}: no taxa")
    return out


def load_sources(path):
    raw = json.load(open(path, encoding="utf-8"))
    out = []
    for r in raw:
        txt = r.get("text") or r.get("description_text") or ""
        out.append({"id": str(r.get("id")), "title": str(r.get("title") or ""), "doi": str(r.get("doi") or ""),
                    "text": str(txt), "kind": r.get("kind", "biosyslit"), "file": r.get("file", "")})
    return out


# ═════════════════════════════════════════════════════════════════════ names ══
def _gen_epi(name):
    w = re.sub(r"\([^)]*\)", " ", name).split()
    w = [x for x in w if not re.fullmatch(r"(subsp|var|ssp|f|cf|aff)\.?", x, re.I)]
    return (w[0].lower(), w[1].lower()) if len(w) >= 2 else (w[0].lower() if w else "", "")


def names_in_title(title, names, at_start=False):
    """True if the title (or file name) names one of the taxa: genus and epithet as whole words (a subgenus in
    parentheses between them is fine). at_start: the title must begin with the genus, as the title of a
    treatment does (BioSysLit also holds article and figure records whose titles merely mention the taxon)."""
    t = " " + re.sub(r"[_\W]+", " ", str(title).lower()) + " "
    for n in names:
        g, e = _gen_epi(n)
        if not g:
            continue
        if (t.startswith(f" {g} ") if at_start else f" {g} " in t) and (not e or f" {e} " in t):
            return True
    return False


def is_figure(rec):
    return bool(FIGURE_TITLE.match(rec["title"]))


def select_sources(tx, sources, by_id, max_sources=5, include_figures=False, min_chars=100, title_anywhere=False):
    """(own, extra): own = treatments of the taxon or a synonym: BioSysLit records whose title begins with the name
    (anywhere with title_anywhere), PDFs and text files whose title or file name contains it (longest first, at
    most max_sources); extra = sources listed for the taxon in the taxa file."""
    names = [tx["taxon"]] + tx["synonyms"]
    own = [r for r in sources if len(r["text"]) > min_chars and (include_figures or not is_figure(r))
           and (names_in_title(r["title"], names, at_start=(r["kind"] == "biosyslit" and not title_anywhere))
                or (r["file"] and names_in_title(Path(r["file"]).stem, names)))]
    own = sorted(own, key=lambda r: -len(r["text"]))[:max_sources]
    ids = {r["id"] for r in own}
    extra = []
    for s in tx["sources"]:
        r = by_id.get(s) or by_id.get(f"pdf:{s}") or by_id.get(f"txt:{s}")
        if r is None:
            r = next((x for x in sources if x["file"] and Path(x["file"]).name == s), None)
        if r is None:
            print(f"  ! {tx['taxon']}: source '{s}' is not in the sources file")
        elif r["id"] not in ids:
            extra.append(r)
            ids.add(r["id"])
    return own, extra


# a treatment title: Genus [(Subgenus)] epithet, then the end, an author, a year or a nomenclatural note
TITLE_NAME = re.compile(r"^\s*([A-Z][a-z]+)\s+(?:\([A-Z][a-z]+\)\s+)?([a-z][a-z\-]{2,})"
                        r"(?=\s*$|\s*,|\s+(?:\(?[A-Z]|\d|sp\b|n\.|nov|subsp|var\b|ssp|stat\b|comb\b|syn\b|&|de\b|van\b|von\b))")


def name_vocabulary(sources, taxa):
    """Binomials known to the run: the taxa (and synonyms) plus every binomial that opens a source title (Plazi
    treatment titles are taxon names). -> {(genus, epithet): canonical} and {(initial., epithet): [canonical]}."""
    full, abbr = {}, defaultdict(list)

    def add(g, e, canon):
        if not g or not e or g in NOT_GENUS or e in NOT_EPITHET:
            return
        if (g, e) not in full:
            full[(g, e)] = canon
            abbr[(g[0] + ".", e)].append(canon)
    for tx in taxa:
        for n in [tx["taxon"]] + tx["synonyms"]:
            g, e = _gen_epi(n)
            add(g, e, tx["taxon"])
    for r in sources:
        m = TITLE_NAME.match(r["title"])
        if m:
            g, e = m.group(1).lower(), m.group(2).lower()
            add(g, e, f"{m.group(1)} {e}")
    return full, abbr


_WORD = re.compile(r"\([a-z]+\)|\b[a-z] ?\.|[a-z][a-z]+")     # 's.' and Plazi's 's .' are abbreviated genera


def mentions(ntext, vocab):
    """[(position, canonical name)] of every binomial of the vocabulary in a normalised text, in full or with
    the genus abbreviated ('s. cretica'); a subgenus in parentheses may stand between genus and epithet."""
    full, abbr = vocab
    toks = [(m.start(), m.group().replace(" ", "")) for m in _WORD.finditer(ntext)]
    out = []
    for i, (p, w) in enumerate(toks):
        j = i + 1
        if j < len(toks) and toks[j][1].startswith("("):
            j += 1
        if j >= len(toks):
            break
        e = toks[j][1]
        if (w, e) in full:
            e3 = toks[j + 1][1] if j + 1 < len(toks) else ""
            # an old trinomial ('rhysida nuda immarginata') names the subspecies, now often a species of its own
            out.append((p, full[(w, e3)] if (w, e3) in full else full[(w, e)]))
        elif len(w) == 2 and w.endswith(".") and (w, e) in abbr:
            out.append((p, abbr[(w, e)]))
    return out


def context_flags(target_names, own, ment, pos, qlen, quote, window=OWN_WINDOW):
    """Flags for the reviewer from where the quote sits in its source (no model). In the taxon's own treatment,
    another name counts only within `window` characters before the quote (synonymy lists and comparisons further
    up do not make a description about another taxon); in any other source, the nearest name before the quote
    must be the taxon's."""
    flags = []
    tset = {name_key(n) for n in target_names}

    def is_target(c):
        return any(name_key(x) in tset for x in (c if isinstance(c, list) else [c]))

    def show(c):
        return c[0] if isinstance(c, list) else c
    if pos >= 0 and ment:
        ps = [m[0] for m in ment]
        k = bisect.bisect_left(ps, pos)
        prev = ment[k - 1][1] if k else None
        if own and k and pos - ment[k - 1][0] > window:
            prev = None
        inside = [c for p, c in ment[k:] if p < pos + qlen and not is_target(c)]
        if prev is not None and not is_target(prev):
            flags.append(f"other taxon named before the quote: {show(prev)}")
        elif prev is None and not own:
            flags.append("taxon not named before the quote in this source")
        if inside:
            flags.append(f"quote names another taxon: {show(inside[0])}")
    elif not own:
        flags.append("taxon not named before the quote in this source")
    if COMPARATIVE.search(quote):
        flags.append("comparative statement")
    if not own:
        flags.append("source is not a treatment of this taxon")
    return flags


# ══════════════════════════════════════════════════════════════════ checking ══
def split_states(s):
    s = re.sub(r"[(){}\[\]]", " ", str(s))
    return [x for x in re.split(r"[\s,&/;|]+|\bor\b|\band\b", s) if x]


def check_item(it, tx, chars_by_id, src_norm, own_ids, ment, near, allow_poly=True):
    """Check one proposed state against the character list and its sources (no model). -> row dict with status
    'kept' or 'rejected' and the reason; kept rows carry the source the quote was found in and reviewer flags."""
    cid = str(it.get("id", "")).strip()
    ev = str(it.get("evidence") or it.get("quote") or "").strip()
    row = {"taxon": tx["taxon"], "label": tx.get("tip_label", ""), "char_id": cid, "evidence": ev, "status": "rejected"}
    c = chars_by_id.get(cid)
    if c is None:
        row["reason"] = "unknown character id"
        return row
    row.update(character=c["description"], type=c["type"], char_pos=c["pos"], unit=c.get("unit", ""))
    if c["type"] == "discrete":
        st = split_states(it.get("state", ""))
        row["state"] = "&".join(st)          # '0&1', not '0,1': spreadsheets read that as a number
        if not st:
            row["reason"] = "no state"
            return row
        bad = [x for x in st if x not in c["states"]]
        if bad:
            row["reason"] = f"state {','.join(bad)} not allowed"
            return row
        if len(st) > 1 and not allow_poly:
            row["reason"] = "more than one state (--no-polymorphism)"
            return row
        row["state_label"] = " | ".join(c["states"][x] for x in st)
    else:
        v = str(it.get("value", it.get("state", ""))).strip()
        row["value"] = v
        pv = parse_value(v)
        if pv is None:
            row["reason"] = "value is not a number or a range"
            return row
        if c["type"] == "meristic" and not all(float(x).is_integer() for x in pv):
            row["reason"] = "count is not a whole number"
            return row
        row["value"] = fmt_num(pv[0]) if pv[0] == pv[1] else f"{fmt_num(pv[0])}–{fmt_num(pv[1])}"   # en dash: '1-3' becomes a date in spreadsheets
        qn = numbers_in(ev)
        if not all(any(abs(x - y) < 1e-9 for y in qn) for x in pv):
            row["reason"] = "value not in the quote"
            return row
    best = ("absent", 0.0, -1, None)
    for sid, ntext in src_norm:
        st, sc, p = find_quote(ev, ntext, near)
        if sc > best[1] or best[3] is None:
            best = (st, sc, p, sid)
        if st == "exact":
            break
    status, score, pos, sid = best
    row.update(quote_match=status, quote_score=score, source_id=sid or "")
    if status not in ("exact", "near"):
        row["reason"] = f"quote {status} ({score})"
        return row
    own = sid in own_ids
    row["own_source"] = "yes" if own else "no"
    fl = context_flags([tx["taxon"]] + tx["synonyms"], own, ment.get(sid, []), pos, len(norm(ev)), ev)
    if c["type"] == "discrete" and len(split_states(row["state"])) > 1:
        fl.append("more than one state (variation?)")
    row["flags"] = "; ".join(fl)
    row["status"] = "kept"
    row["reason"] = ""
    return row


def mark_conflicts(rows):
    """Flag cells given different states (or values) by different quotes; drop exact duplicates (same cell, same
    state) after the first, noting where else the state was quoted."""
    out, first = [], {}
    by_cell = defaultdict(set)
    for r in rows:
        by_cell[(r["taxon"], r["char_id"])].add(r.get("state") or r.get("value"))
    for r in rows:
        key = (r["taxon"], r["char_id"], r.get("state") or r.get("value"))
        if key in first:
            f = first[key]
            f["note"] = (f.get("note", "") + f"; also quoted in {r.get('source_id')}").lstrip("; ")
            continue
        if len(by_cell[(r["taxon"], r["char_id"])]) > 1:
            r["flags"] = "; ".join([x for x in (r.get("flags", ""), "conflicting states for this cell") if x])
        first[key] = r
        out.append(r)
    return out


def prefill(rows, how):
    for r in rows:
        if how == "clean" and not r.get("decision") and not r.get("flags") and r.get("quote_match") in ("exact", "near"):
            r["decision"] = "accept"
    return rows


# ═════════════════════════════════════════════════════════════════════ init ══
EXAMPLE_CHARS = [
    {"id": 1, "description": "Shape of head capsule", "type": "discrete", "states": {"0": "domed", "1": "flattened"},
     "ordered": False},
    {"id": 2, "description": "Ocelli", "type": "discrete", "states": {"0": "absent", "1": "present"}, "ordered": False},
    {"id": 3, "description": "Number of antennal articles", "type": "meristic"},
    {"id": 4, "description": "Body length", "type": "continuous", "unit": "mm"},
]


def cmd_init(a):
    if a.from_nexus:
        chars = characters_from_nexus(a.from_nexus)
        out = Path(a.out or Path(a.from_nexus).with_suffix(".characters.json"))
        if out.exists() and not a.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        json.dump(chars, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        no_states = [c["id"] for c in chars if not c["states"]]
        print(f"{len(chars)} characters ({sum(c['ordered'] for c in chars)} ordered) -> {out}")
        if no_states:
            print(f"  ! {len(no_states)} characters have no state labels in the NEXUS (ids {no_states[:10]}); "
                  "add them to the JSON before extracting, or the model cannot score them")
        return
    d = Path(a.out_dir or ".")
    d.mkdir(parents=True, exist_ok=True)
    cp, tp = d / "characters_example.json", d / "taxa_example.tsv"
    for p in (cp, tp):
        if p.exists() and not a.overwrite:
            raise SystemExit(f"{p} exists (use --overwrite)")
    json.dump(EXAMPLE_CHARS, open(cp, "w"), indent=1)
    with open(tp, "w") as f:
        f.write("taxon\ttip_label\tsynonyms\tsources\n")
        f.write("Genus species\tGenus_species_voucher1\tOldgenus species\t\n")
        f.write("Genus otherspecies\t\t\tMonograph_2019.pdf\n")
    print(f"examples -> {cp}, {tp}")


# ═════════════════════════════════════════════════════════════════════ fetch ══
def _get_json(url, tries=4):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    err = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e
            if e.code == 429:
                time.sleep(min(int(e.headers.get("Retry-After", 30) or 30), 120))
                continue
            if e.code >= 500:
                time.sleep(5 * (i + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            err = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"Zenodo did not answer: {err}")


def _record(rec):
    meta = rec.get("metadata", {})
    desc = meta.get("description", "") or ""
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", desc))).strip()
    doi = rec.get("doi") or (rec.get("pids", {}).get("doi", {}) or {}).get("identifier", "")
    kw = meta.get("keywords") or [s.get("subject", "") for s in meta.get("subjects", []) if isinstance(s, dict)]
    return {"id": str(rec.get("id")), "doi": doi, "title": html.unescape(meta.get("title", "")), "text": text,
            "kind": "biosyslit", "keywords": kw, "publication_date": meta.get("publication_date", ""),
            "related_identifiers": meta.get("related_identifiers", [])}


def zenodo_search(q, max_records, label):
    out, page, size = [], 1, 25
    while len(out) < max_records:
        url = f"{ZENODO_API}?" + urllib.parse.urlencode({"communities": BIOSYSLIT, "q": q, "size": size,
                                                          "page": page, "sort": "mostrecent"})
        data = _get_json(url)
        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", 0)
        total = total.get("value", 0) if isinstance(total, dict) else total
        if page == 1:
            print(f"  {label}: {total} records")
        if not hits:
            break
        out += [_record(h) for h in hits]
        if page * size >= min(total, 10000):
            break
        page += 1
        time.sleep(1.0)
    return out[:max_records]


def pdf_text(path):
    if shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True)
        txt = r.stdout
    else:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SystemExit("reading PDFs needs pdftotext (poppler-utils) or the pypdf package")
        txt = "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    txt = re.sub(r"(\w)-\n([a-z])", r"\1\2", txt)          # words hyphenated at line ends
    txt = re.sub(r"[ \t]*\n(?!\n)[ \t]*", " ", txt)         # single line breaks -> spaces, paragraphs kept
    return re.sub(r"[ \t]+", " ", txt).strip()


def pdf_title(path):
    if shutil.which("pdfinfo"):
        r = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True)
        m = re.search(r"^Title:\s*(.+)$", r.stdout, re.M)
        if m and len(m.group(1).strip()) > 8:
            return m.group(1).strip()
    return Path(path).stem.replace("_", " ")


def cmd_fetch(a):
    out = Path(a.out)
    recs = OrderedDict()
    if out.exists() and not a.overwrite:
        for r in json.load(open(out, encoding="utf-8")):
            recs[str(r["id"])] = r
        print(f"{len(recs)} records already in {out} (new ones are added; --overwrite starts again)")
    n0 = len(recs)
    for kw in a.keywords or []:
        for r in zenodo_search(f'keywords:"{kw}"', a.max_records, f"BioSysLit keyword {kw}"):
            recs.setdefault(r["id"], r)
    if a.names:
        if not a.taxa:
            raise SystemExit("--names needs --taxa")
        for tx in load_taxa(a.taxa):
            for n in [tx["taxon"]] + tx["synonyms"]:
                for r in zenodo_search(f'"{n}"', a.max_records_per_name, f"BioSysLit name {n}"):
                    recs.setdefault(r["id"], r)
    for d, kind, pat in ((a.pdf_dir, "pdf", "*.pdf"), (a.text_dir, "txt", "*.txt")):
        if not d:
            continue
        for p in sorted(Path(d).glob(pat)):
            text = pdf_text(p) if kind == "pdf" else p.read_text(encoding="utf-8", errors="replace").strip()
            title = pdf_title(p) if kind == "pdf" else p.stem.replace("_", " ")
            recs[f"{kind}:{p.name}"] = {"id": f"{kind}:{p.name}", "doi": "", "title": title, "text": text,
                                        "kind": kind, "file": str(p.resolve())}
            print(f"  {kind}: {p.name} ({len(text)} characters)")
    short = [k for k, r in recs.items() if len(r.get("text") or r.get("description_text") or "") <= a.min_chars]
    for k in short:
        del recs[k]
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(list(recs.values()), open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(recs)} sources ({len(recs) - n0} new; {len(short)} with {a.min_chars} characters or fewer "
          f"left out) -> {out}")


# ═══════════════════════════════════════════════════════════════════ extract ══
def chunks_for(blocks, max_chars):
    """Pack (record, text) blocks into model calls of at most max_chars characters; a longer source is split at
    paragraph or sentence ends."""
    pieces = []
    for rec, txt in blocks:
        if len(txt) <= max_chars:
            pieces.append((rec, txt))
            continue
        parts, cur = re.split(r"(?<=[.!?])\s+|\n\n+", txt), ""
        for p in parts:
            if cur and len(cur) + len(p) + 1 > max_chars:
                pieces.append((rec, cur))
                cur = ""
            while len(p) > max_chars:
                pieces.append((rec, p[:max_chars]))
                p = p[max_chars:]
            cur = (cur + " " + p).strip()
        if cur:
            pieces.append((rec, cur))
    out, cur, n = [], [], 0
    for rec, txt in pieces:
        if cur and n + len(txt) > max_chars:
            out.append(cur)
            cur, n = [], 0
        cur.append((rec, txt))
        n += len(txt)
    if cur:
        out.append(cur)
    return out


def char_lines(chars):
    out = []
    for c in chars:
        if c["type"] == "discrete":
            out.append(f"[{c['id']}] {c['description']} — States: "
                       + "; ".join(f"{k}={v}" for k, v in c["states"].items()))
        elif c["type"] == "meristic":
            out.append(f"[{c['id']}] {c['description']} — COUNT (a whole number, or a range such as 19-23)")
        else:
            out.append(f"[{c['id']}] {c['description']} — MEASUREMENT" + (f" in {c['unit']}" if c["unit"] else "")
                       + " (a number, or a range such as 2.5-3.1)")
    return "\n".join(out)


def build_prompt(taxon, group, chars, text, context="", allow_poly=True):
    rank = "species" if len(taxon.split()) >= 2 else "taxon"
    grp = f"{group} " if group else ""
    types = {c["type"] for c in chars}
    fmt = '{"id": <character id>, "state": "<state number>", "evidence": "<verbatim quote>"}'
    if types & {"meristic", "continuous"}:
        fmt += (' for characters with states, and {"id": <character id>, "value": "<number or min-max>", '
                '"evidence": "<verbatim quote>"} for counts and measurements')
    rules = [
        f"Score a character only if the text states it about {taxon} itself. Never score from absence: if a "
        "structure is not mentioned, do not score it.",
        '"evidence" must be copied VERBATIM from the text above (one sentence or clause, at most 200 characters), '
        "with no paraphrase, no ellipsis and no added words. If you cannot quote it, do not score it.",
        "Do not use knowledge of the genus or family; only this text.",
    ]
    if types & {"meristic", "continuous"}:
        rules.append("A count or measurement must be written in the quote exactly as you report it.")
    if allow_poly:
        rules.append('If the text gives more than one state for this taxon (variation), give all of them in "state", '
                     'separated by commas, with a quote that shows the variation.')
    rules.append(f"Return ONLY a JSON array of objects {fmt}, or [] if nothing can be scored.")
    ctx = (f"\nBACKGROUND FROM THE USER (terminology only; never a source of states):\n{context.strip()}\n"
           if context.strip() else "")
    return (f"Score characters of the {grp}{rank} {taxon} from the text below.\n\nTEXT:\n{text}\n\n"
            f"CHARACTER DEFINITIONS (ids are as given):\n{char_lines(chars)}\n{ctx}\nRULES\n"
            + "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1)))


def system_prompt(group):
    return (f"You are a {group + ' ' if group else ''}taxonomist scoring a fixed morphological character matrix "
            "from published text. You copy evidence word for word and never infer a state from silence.")


def parse_items(raw):
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return [], "no JSON array in the reply"
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return [], f"reply is not valid JSON ({e})"
    return [x for x in items if isinstance(x, dict)], ""


def _model_args(p):
    p.add_argument("--llm-backend", default="claude-code", choices=["claude-code", "api", "none"],
                   help="claude-code = a Claude subscription via `claude -p`; api = an API key ($ANTHROPIC_API_KEY or "
                        "--api-key-env); none = no model (only with --replay)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"model id (default {DEFAULT_MODEL}; the claude-code "
                                                          "backend passes the family alias, e.g. opus)")
    p.add_argument("--cc-model", default=None, help="model alias for claude-code (default: derived from --model)")
    p.add_argument("--claude-bin", default=None, help="path to the Claude Code CLI (default: `claude` on PATH)")
    p.add_argument("--api-base-url", default=None, help="with --llm-backend api: any endpoint with the Anthropic "
                                                        "messages API")
    p.add_argument("--api-key-env", default="ANTHROPIC_API_KEY", help="variable that holds the API key")
    p.add_argument("--max-tokens", type=int, default=16000, help="longest reply per call")


def _checks_args(p):
    p.add_argument("--near", type=float, default=0.90,
                   help="a quote not found verbatim is accepted if its best difflib ratio against a sliding window of "
                        "the source (20%% longer than the quote, as in the centipede audit) is at least this. A perfect "
                        "alignment then scores about 0.91, so the default 0.90 accepts only spacing and punctuation "
                        "differences (about 99%% of the quote's characters in order); 0.85 allows a few changed words"),
    p.add_argument("--max-sources", type=int, default=5, help="own sources per taxon, longest first (default 5)")
    p.add_argument("--min-chars", type=int, default=100, help="sources with this many characters or fewer are ignored")
    p.add_argument("--include-figures", action="store_true",
                   help="also use figure / table records (left out by default: captions often describe other taxa)")
    p.add_argument("--title-anywhere", action="store_true",
                   help="a BioSysLit record counts as the taxon's own if its title names it anywhere (default: the "
                        "title must begin with the name, as treatment titles do; article records are left out)")
    p.add_argument("--no-polymorphism", action="store_true", help="reject states that list more than one state")
    p.add_argument("--prefill", choices=["none", "clean"], default="none",
                   help="clean = write 'accept' in the decision column of states with a verified quote and no flag "
                        "(you still read and can change them); default none = every decision is yours")


class Run:
    """Everything extract and audit share: characters, taxa, sources, names, per-source normalised text."""

    def __init__(self, a):
        self.chars = load_characters(a.characters)
        self.cby = {c["id"]: c for c in self.chars}
        self.taxa = load_taxa(a.taxa)
        self.sources = load_sources(a.sources)
        self.by_id = {r["id"]: r for r in self.sources}
        self.vocab = name_vocabulary(self.sources, self.taxa)
        self._norm, self._ment = {}, {}
        self.a = a

    def ntext(self, sid):
        if sid not in self._norm:
            self._norm[sid] = norm(self.by_id[sid]["text"]) if sid in self.by_id else ""
        return self._norm[sid]

    def ment(self, sid):
        if sid not in self._ment:
            self._ment[sid] = mentions(self.ntext(sid), self.vocab)
        return self._ment[sid]

    def select(self, tx):
        return select_sources(tx, self.sources, self.by_id, self.a.max_sources, self.a.include_figures,
                              self.a.min_chars, self.a.title_anywhere)

    def check(self, it, tx, sids, own_ids):
        return check_item(it, tx, self.cby, [(s, self.ntext(s)) for s in sids], own_ids,
                          {s: self.ment(s) for s in sids}, self.a.near, not self.a.no_polymorphism)


def _raw_path(raw_dir, taxon, k):
    return Path(raw_dir) / f"{slug(taxon, 80)}__c{k}.json"


def cmd_extract(a):
    R = Run(a)
    out = Path(a.out_dir)
    if not a.dry_run:
        (out / "raw").mkdir(parents=True, exist_ok=True)
    ctx = Path(a.context_file).read_text(encoding="utf-8") if a.context_file else ""
    use_model = not a.dry_run and not a.replay
    if a.llm_backend == "none" and use_model:
        raise SystemExit("extraction needs a model: --llm-backend claude-code or api (or --replay a saved run)")
    client, log = None, str(out / "llm_calls.jsonl")
    if use_model:
        sys.path.insert(0, str(HERE))
        from biorag_llm_backend import client_from_args
        a.llm_log = log
        client = client_from_args(a, log)
    kept, rejected, summary, used = [], [], [], []
    fails = 0
    print(f"{len(R.taxa)} taxa, {len(R.chars)} characters, {len(R.sources)} sources")
    for tx in R.taxa:
        own, extra = R.select(tx)
        chosen = own + extra
        for r in chosen:
            used.append({"taxon": tx["taxon"], "source_id": r["id"], "own": "yes" if r in own else "no (taxa file)",
                         "title": r["title"], "doi": r["doi"], "characters": len(r["text"])})
        if not chosen:
            summary.append({"taxon": tx["taxon"], "sources": 0, "calls": 0, "proposed": 0, "kept": 0, "rejected": 0,
                            "flagged": 0, "note": "no source names this taxon"})
            print(f"{tx['taxon']}: no source names this taxon")
            continue
        blocks = [(r, clean_text(r["text"])) for r in chosen]
        chunks = chunks_for(blocks, a.max_chars)
        if a.dry_run:
            print(f"{tx['taxon']}: {len(chosen)} sources ({len(own)} own), {sum(len(t) for _, t in blocks)} "
                  f"characters, {len(chunks)} call(s)")
            for r in chosen:
                print(f"    {r['id']}  {r['title'][:90]}")
            summary.append({"taxon": tx["taxon"], "sources": len(chosen), "calls": len(chunks)})
            continue
        own_ids = {r["id"] for r in own}
        n_prop, t_rows, notes = 0, [], []
        for k, ch in enumerate(chunks):
            text = "\n\n".join(f"[Source {r['id']}: {r['title']}]\n{t}" for r, t in ch)
            prompt = build_prompt(tx["taxon"], a.group, R.chars, text, ctx, not a.no_polymorphism)
            sha = hashlib.sha256((system_prompt(a.group) + "\n" + prompt).encode()).hexdigest()
            rp = _raw_path(out / "raw", tx["taxon"], k)
            raw = None
            if a.replay:
                src = _raw_path(Path(a.replay) / "raw" if (Path(a.replay) / "raw").is_dir() else a.replay, tx["taxon"], k)
                if src.exists():
                    rec = json.load(open(src))
                    raw = rec["response"]
                    if rec.get("prompt_sha256") != sha:
                        notes.append(f"call {k}: replayed reply was made with a different prompt")
                    if src.resolve() != rp.resolve():
                        shutil.copy2(src, rp)
                else:
                    notes.append(f"call {k}: no saved reply to replay")
                    continue
            elif rp.exists() and not a.no_reuse:
                rec = json.load(open(rp))
                if rec.get("prompt_sha256") == sha:
                    raw = rec["response"]
            if raw is None:
                t0 = time.time()
                try:
                    r = client.messages.create(model=a.model, max_tokens=a.max_tokens, system=system_prompt(a.group),
                                               messages=[{"role": "user", "content": prompt}])
                    raw = r.content[0].text
                    fails = 0
                except Exception as e:                     # one failed call should not lose the run
                    fails += 1
                    notes.append(f"call {k}: model error ({str(e)[:200]})")
                    print(f"  ! {tx['taxon']} call {k}: model error: {e}", flush=True)
                    if fails >= 3:
                        raise SystemExit("three model calls in a row failed; the replies saved so far are in "
                                         f"{out / 'raw'} and are reused when you run the same command again")
                    continue
                json.dump({"taxon": tx["taxon"], "chunk": k, "sources": [x["id"] for x, _ in ch],
                           "prompt_sha256": sha, "model_requested": a.model, "backend": a.llm_backend,
                           "seconds": round(time.time() - t0, 1), "time": datetime.now().isoformat(timespec="seconds"),
                           "response": raw}, open(rp, "w"), indent=1)
            items, err = parse_items(raw)
            if err:
                notes.append(f"call {k}: {err}")
            n_prop += len(items)
            sids = list(dict.fromkeys(x["id"] for x, _ in ch))
            for it in items:
                row = R.check(it, tx, sids, own_ids)
                row["chunk"] = k
                (t_rows if row["status"] == "kept" else rejected).append(row)
        t_rows = mark_conflicts(t_rows)
        for row in t_rows:
            rec = R.by_id.get(row["source_id"], {})
            row.update(source_title=rec.get("title", ""), source_doi=rec.get("doi", ""))
        kept += t_rows
        nflag = sum(1 for r in t_rows if r.get("flags"))
        summary.append({"taxon": tx["taxon"], "sources": len(chosen), "calls": len(chunks), "proposed": n_prop,
                        "kept": len(t_rows), "rejected": n_prop - len(t_rows), "flagged": nflag,
                        "note": "; ".join(notes)})
        print(f"{tx['taxon']}: {len(chosen)} sources, proposed {n_prop}, kept {len(t_rows)} ({nflag} flagged)"
              + (f"  [{'; '.join(notes)}]" if notes else ""), flush=True)
    if a.dry_run:
        print(f"dry run: {sum(s['calls'] for s in summary)} model calls would be made; nothing written")
        return
    prefill(kept, a.prefill)
    write_tsv(out / "review_sheet.tsv", kept, STATE_FIELDS)
    write_tsv(out / "rejected.tsv", rejected, ["taxon", "char_id", "character", "state", "value", "reason",
                                               "evidence", "quote_score", "chunk"])
    write_tsv(out / "summary.tsv", summary, ["taxon", "sources", "calls", "proposed", "kept", "rejected", "flagged",
                                             "note"])
    write_tsv(out / "sources_used.tsv", used, ["taxon", "source_id", "own", "title", "doi", "characters"])
    answered = {}
    if os.path.exists(log):
        sys.path.insert(0, str(HERE))
        from biorag_llm_backend import models_in_log
        answered = models_in_log(log)
    prov = {"program": "phylorag_v1.py", "version": VERSION, "step": "extract", "argv": sys.argv,
            "time": datetime.now().isoformat(timespec="seconds"),
            "characters": {"file": str(Path(a.characters).resolve()), "sha256": sha256_file(a.characters)},
            "taxa": {"file": str(Path(a.taxa).resolve()), "sha256": sha256_file(a.taxa)},
            "sources": {"file": str(Path(a.sources).resolve()), "sha256": sha256_file(a.sources),
                        "records": len(R.sources)},
            "backend": "replay" if a.replay else a.llm_backend, "model_requested": a.model,
            "models_that_answered": answered,
            "rules": {"near": a.near, "max_sources": a.max_sources, "min_chars": a.min_chars,
                      "include_figures": a.include_figures, "polymorphism": not a.no_polymorphism,
                      "max_chars_per_call": a.max_chars, "prefill": a.prefill},
            "totals": {k: sum(s.get(k, 0) for s in summary) for k in ("proposed", "kept", "rejected", "flagged")}}
    json.dump(prov, open(out / "run.json", "w"), indent=1)
    t = prov["totals"]
    print(f"\nproposed {t['proposed']}, kept {t['kept']} ({t['flagged']} flagged for review), rejected {t['rejected']}")
    if answered:
        print(f"models that answered: {answered}")
    print(f"review: {out / 'review_sheet.tsv'}  (write accept or reject in the 'decision' column, then run export)")


# ═════════════════════════════════════════════════════════════════════ audit ══
def cmd_audit(a):
    """Re-check states without a model: a review sheet (decisions and notes are kept), a hand-coded table
    (taxon, char_id, state or value, evidence[, source_id]) or the results JSON of the centipede scripts
    ({key: {species, scores, evidences, treatment_ids}})."""
    R = Run(a)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tx_by = {t["taxon"]: t for t in R.taxa}
    items = []
    if a.results_json:
        for key, v in json.load(open(a.results_json)).items():
            sc = v["scores"] if isinstance(v["scores"], dict) else {}
            ev = v.get("evidences", {})
            for cid, st in sc.items():
                items.append({"taxon": v.get("species", key), "label": key, "char_id": str(cid), "state": str(st),
                              "evidence": ev.get(cid, ev.get(str(cid), "")),
                              "source_ids": [str(x) for x in v.get("treatment_ids", [])]})
    else:
        for r in read_tsv(a.states):
            items.append({**r, "char_id": str(r.get("char_id", "")).strip(),
                          "source_ids": [s for s in str(r.get("source_id", "")).split(";") if s.strip()]})
    kept, rejected = [], []
    missing_tx = Counter()
    for it in items:
        tx = tx_by.get(it.get("taxon", ""))
        if tx is None:
            missing_tx[it.get("taxon", "")] += 1
            tx = {"taxon": it.get("taxon", ""), "tip_label": it.get("label", ""), "synonyms": [], "sources": []}
        own, extra = R.select(tx)
        own_ids = {r["id"] for r in own}
        sids = [s for s in it["source_ids"] if s in R.by_id] or [r["id"] for r in own + extra]
        row = R.check({"id": it["char_id"], "state": it.get("state", ""), "value": it.get("value", ""),
                       "evidence": it.get("evidence", "")}, tx, sids, own_ids)
        row["label"] = it.get("label") or tx.get("tip_label", "")
        row["decision"], row["note"] = it.get("decision", ""), it.get("note", "")
        row["chunk"] = it.get("chunk", "")
        rec = R.by_id.get(row.get("source_id", ""), {})
        row.update(source_title=rec.get("title", ""), source_doi=rec.get("doi", ""))
        (kept if row["status"] == "kept" else rejected).append(row)
    kept = mark_conflicts(kept)
    prefill(kept, a.prefill)
    write_tsv(out / "review_sheet.tsv", kept, STATE_FIELDS)
    write_tsv(out / "rejected.tsv", rejected, ["taxon", "label", "char_id", "character", "state", "value", "reason",
                                               "evidence", "quote_score", "decision", "note"])
    fl = Counter(f.split(":")[0] for r in kept for f in r.get("flags", "").split("; ") if f)
    rs = Counter(re.sub(r"\s*\(.*", "", r["reason"]) for r in rejected)
    rep = {"program": "phylorag_v1.py", "version": VERSION, "step": "audit", "argv": sys.argv,
           "time": datetime.now().isoformat(timespec="seconds"), "input": a.results_json or a.states,
           "states": len(items), "kept": len(kept), "rejected": len(rejected),
           "quote_match": dict(Counter(r["quote_match"] for r in kept)), "flags": dict(fl),
           "rejection_reasons": dict(rs), "taxa_not_in_taxa_file": dict(missing_tx), "near": a.near}
    json.dump(rep, open(out / "audit_report.json", "w"), indent=1)
    print(f"{len(items)} states: kept {len(kept)} ({sum(1 for r in kept if r.get('flags'))} flagged), "
          f"rejected {len(rejected)}")
    for k, v in rs.most_common():
        print(f"  rejected, {k}: {v}")
    for k, v in fl.most_common():
        print(f"  flag, {k}: {v}")
    if missing_tx:
        print(f"  ! taxa not in the taxa file (checked against their listed sources only): {dict(missing_tx)}")
    print(f"-> {out / 'review_sheet.tsv'}")


# ════════════════════════════════════════════════════════════════════ export ══
def cmd_export(a):
    chars = load_characters(a.characters)
    disc = [c for c in chars if c["type"] == "discrete"]
    cby = {c["id"]: c for c in chars}
    taxa = load_taxa(a.taxa)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(a.review)
    dec = lambda r: str(r.get("decision", "")).strip().lower()
    acc = [r for r in rows if dec(r) in ACCEPT]
    unrev = [r for r in rows if dec(r) not in ACCEPT | REJECT]
    rej = [r for r in rows if dec(r) in REJECT]
    if a.include_unreviewed:
        acc += [r for r in unrev if not r.get("flags")]
    if not acc:
        raise SystemExit(f"no accepted states in {a.review}: write 'accept' in the decision column "
                         f"({len(unrev)} rows not reviewed)")
    mol = []
    for p in a.molecular or []:
        mol += [x for x in molecular_labels(p) if x not in mol]
    mol_by = {name_key(x): x for x in mol}

    def label_for(tx):
        for n in [tx["tip_label"], tx["taxon"]] + tx["synonyms"]:
            if n and name_key(n) in mol_by:
                return mol_by[name_key(n)]          # the name used in the molecular data
        if a.labels == "taxon":
            return tx["taxon"]
        if a.labels == "underscore":
            return tx["taxon"].replace(" ", "_")
        return tx["tip_label"] or tx["taxon"].replace(" ", "_")
    tx_by = {t["taxon"]: t for t in taxa}
    base_labels, base_rows, log = [], {}, []
    if a.base_nexus:
        base_labels, base_rows, nchar = read_nexus_matrix(a.base_nexus)
        if (nchar or 0) != len(disc):
            raise SystemExit(f"{a.base_nexus} has {nchar} characters, the characters file {len(disc)} discrete ones; "
                             "make the characters file from this matrix (init --from-nexus)")
        bad = [lab for lab in base_labels if len(base_rows[lab]) != nchar]
        if bad:
            raise SystemExit(f"{a.base_nexus}: rows with a different number of cells: {bad[:5]}")
    base_key = {name_key(x): x for x in base_labels}
    labels, mat = list(base_labels), OrderedDict((k, list(v)) for k, v in base_rows.items())
    lab_of = {}
    for tx in taxa:
        hit = next((base_key[name_key(n)] for n in [tx["tip_label"], tx["taxon"]] + tx["synonyms"]
                    if n and name_key(n) in base_key), None)
        lab_of[tx["taxon"]] = hit or label_for(tx)
    # the review sheet names each character; a characters file with other ids would put states in wrong columns
    wrong = [(r.get("char_id"), r.get("character"), cby[str(r.get("char_id", "")).strip()]["description"]) for r in acc
             if r.get("character") and str(r.get("char_id", "")).strip() in cby
             and norm(r["character"]) != norm(cby[str(r.get("char_id", "")).strip()]["description"])]
    if wrong:
        raise SystemExit(f"{len(wrong)} accepted rows name a different character than the characters file has under "
                         f"that id, e.g. id {wrong[0][0]}: sheet '{wrong[0][1]}', file '{wrong[0][2]}'. Export with the "
                         "characters file the review sheet was made with.")
    cells = defaultdict(list)
    meristic, unknown = [], Counter()
    for r in acc:
        tx = tx_by.get(r.get("taxon", ""))
        c = cby.get(str(r.get("char_id", "")).strip())
        if tx is None or c is None:
            unknown["taxon not in taxa file" if tx is None else "character not in characters file"] += 1
            continue
        if c["type"] == "discrete":
            for s in split_states(r.get("state", "")):
                if s in c["states"]:
                    cells[(tx["taxon"], c["id"])].append((s, r))
        else:
            pv = parse_value(r.get("value", ""))
            if pv:
                meristic.append({"label": lab_of[tx["taxon"]], "taxon": tx["taxon"], "char_id": c["id"],
                                 "character": c["description"], "type": c["type"], "unit": c.get("unit", ""),
                                 "min": fmt_num(pv[0]), "max": fmt_num(pv[1]), "evidence": r.get("evidence", ""),
                                 "source_id": r.get("source_id", ""), "source_doi": r.get("source_doi", "")})
    prov = []
    for tx in taxa:
        has = any(k[0] == tx["taxon"] for k in cells)
        lab = lab_of[tx["taxon"]]
        if lab not in mat:
            if not has and not any(m["taxon"] == tx["taxon"] for m in meristic) and not a.keep_empty:
                continue
            labels.append(lab)
            mat[lab] = ["?"] * len(disc)
        for c in disc:
            got = cells.get((tx["taxon"], c["id"]))
            if not got:
                continue
            states = sorted({s for s, _ in got}, key=lambda s: (not s.isdigit(), s))
            new = states[0] if len(states) == 1 else "(" + "".join(states) + ")"
            i = c["pos"] - 1
            old = mat[lab][i]
            if old == "?":
                mat[lab][i] = new
                act = "filled"
            elif old == new:
                act = "agrees with base"
            elif old == "-":
                act = "not filled: inapplicable (-) in base"
            else:
                act = f"not filled: base has {old}"
            if a.base_nexus and lab in base_rows:
                log.append({"label": lab, "taxon": tx["taxon"], "char_pos": c["pos"], "character": c["description"],
                            "base": old, "phylorag": new, "action": act})
            if act == "filled" or not a.base_nexus or lab not in base_rows:
                for s, r in got:
                    prov.append({"label": lab, "taxon": tx["taxon"], "char_pos": c["pos"], "char_id": c["id"],
                                 "character": c["description"], "state": s, "state_label": c["states"].get(s, ""),
                                 "evidence": r.get("evidence", ""), "source_id": r.get("source_id", ""),
                                 "source_title": r.get("source_title", ""), "source_doi": r.get("source_doi", ""),
                                 "quote_match": r.get("quote_match", ""), "decision": r.get("decision", "") or
                                 "unreviewed (--include-unreviewed)", "note": r.get("note", "")})
    added_mol = []
    if a.add_missing_taxa and mol:
        have = {name_key(x) for x in labels}
        for m in mol:
            if name_key(m) not in have:
                labels.append(m)
                mat[m] = ["?"] * len(disc)
                added_mol.append(m)
    scored = sum(1 for lab in labels for x in mat[lab] if x not in "?-")
    stem = out / (a.name or "phylorag_matrix")
    note = (f"PhyloRAG {VERSION} (Descriptron), {datetime.now():%Y-%m-%d}: {len(prov)} states from the literature, "
            f"each traced to a verbatim quote ({stem.name}_provenance.tsv)"
            + (f"; filled only missing cells of {Path(a.base_nexus).name}" if a.base_nexus else "")
            + ("; WARNING: includes states nobody reviewed (--include-unreviewed)" if a.include_unreviewed else ""))
    write_nexus(f"{stem}.nex", labels, mat, disc, note)
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["taxon"] + [f"{c['pos']}:{c['description']}" for c in disc])
        for lab in labels:
            w.writerow([lab] + mat[lab])
    write_tsv(f"{stem}_provenance.tsv", prov, ["label", "taxon", "char_pos", "char_id", "character", "state",
                                               "state_label", "evidence", "source_id", "source_title", "source_doi",
                                               "quote_match", "decision", "note"])
    if meristic:
        write_tsv(f"{stem}_counts_measurements.tsv", meristic, ["label", "taxon", "char_id", "character", "type",
                                                                "unit", "min", "max", "evidence", "source_id",
                                                                "source_doi"])
    if log:
        write_tsv(f"{stem}_merge_log.tsv", log, ["label", "taxon", "char_pos", "character", "base", "phylorag",
                                                 "action"])
    mol_rep = {}
    if mol:
        mk = {name_key(x) for x in mol}
        lk = {name_key(x) for x in labels}
        mol_rep = {"molecular_taxa": len(mol), "in_both": len(mk & lk),
                   "only_in_matrix": sorted(x for x in labels if name_key(x) not in mk),
                   "only_in_molecular": sorted(x for x in mol if name_key(x) not in lk),
                   "added_as_empty_rows": added_mol}
    rep = {"program": "phylorag_v1.py", "version": VERSION, "step": "export", "argv": sys.argv,
           "time": datetime.now().isoformat(timespec="seconds"), "review_sheet": str(Path(a.review).resolve()),
           "accepted": len(acc), "rejected": len(rej), "not_reviewed": len(unrev),
           "include_unreviewed": a.include_unreviewed, "skipped": dict(unknown),
           "taxa": len(labels), "characters": len(disc), "scored_cells": scored, "states_entered": len(prov),
           "counts_measurements": len(meristic),
           "merge": dict(Counter(x["action"].split(":")[0] for x in log)) if log else {}, "molecular": mol_rep}
    json.dump(rep, open(f"{stem}_report.json", "w"), indent=1)
    print(f"accepted {len(acc)}, rejected {len(rej)}, not reviewed {len(unrev)}"
          + (" (clean unreviewed states INCLUDED)" if a.include_unreviewed else " (not exported)"))
    print(f"{len(labels)} taxa x {len(disc)} characters, {len(prov)} states entered -> {stem}.nex")
    if log:
        print("  base matrix: " + ", ".join(f"{v} {k}" for k, v in rep["merge"].items()) + f" ({stem}_merge_log.tsv)")
    if meristic:
        print(f"  {len(meristic)} counts / measurements -> {stem}_counts_measurements.tsv")
    if mol_rep:
        print(f"  molecular data: {mol_rep['in_both']} taxa in both, {len(mol_rep['only_in_matrix'])} only in the "
              f"matrix, {len(mol_rep['only_in_molecular'])} only in the molecular data"
              + (f" ({len(added_mol)} added as empty rows)" if added_mol else ""))
    if unknown:
        print(f"  ! skipped: {dict(unknown)}")


# ══════════════════════════════════════════════════════════════════════ main ══
def main(argv=None):
    ap = argparse.ArgumentParser(prog="phylorag_v1.py", description=__doc__.split("\n\n")[1].replace("\n", " "),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"PhyloRAG {VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="characters file from a NEXUS matrix, or example files")
    p.add_argument("--from-nexus", help="published matrix (MorphoBank, Mesquite, PAUP NEXUS)")
    p.add_argument("--out", help="characters JSON to write (default <matrix>.characters.json)")
    p.add_argument("--out-dir", help="folder for the example files (without --from-nexus)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("fetch", help="BioSysLit treatments (Zenodo) and your PDFs / text files -> sources JSON")
    p.add_argument("--out", required=True, help="sources JSON (records are added to it if it exists)")
    p.add_argument("--keywords", nargs="*", default=[],
                   help="Zenodo keywords of the BioSysLit records, usually an order or family (e.g. Scolopendromorpha)")
    p.add_argument("--names", action="store_true", help="also search BioSysLit for each taxon name (and synonym)")
    p.add_argument("--taxa", help="taxa file (for --names)")
    p.add_argument("--pdf-dir", help="folder of PDFs (text by pdftotext, else pypdf)")
    p.add_argument("--text-dir", help="folder of .txt files (e.g. OCR of old descriptions)")
    p.add_argument("--max-records", type=int, default=20000, help="per keyword")
    p.add_argument("--max-records-per-name", type=int, default=50)
    p.add_argument("--min-chars", type=int, default=100, help="drop records with this many characters or fewer")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("extract", help="score the characters of each taxon from its own sources (model)")
    p.add_argument("--characters", required=True)
    p.add_argument("--taxa", required=True)
    p.add_argument("--sources", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--group", default="", help="who is scoring, for the prompt (e.g. centipede, weevil, orchid)")
    p.add_argument("--context-file", help="terminology notes for the prompt (never a source of states)")
    p.add_argument("--max-chars", type=int, default=60000, help="source text per model call (longer runs are split)")
    p.add_argument("--dry-run", action="store_true", help="list the sources each taxon would use; no model calls")
    p.add_argument("--replay", help="re-read the saved replies of an earlier run (its folder) instead of calling a model")
    p.add_argument("--no-reuse", action="store_true",
                   help="call the model again even where this folder holds a reply to the same prompt")
    _model_args(p)
    _checks_args(p)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("audit", help="re-check states without a model (review sheet, hand-coded table, old run)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--states", help="TSV/CSV with taxon, char_id, state or value, evidence[, source_id, decision]")
    g.add_argument("--results-json", help="results JSON of the centipede scripts ({key: {species, scores, ...}})")
    p.add_argument("--characters", required=True)
    p.add_argument("--taxa", required=True)
    p.add_argument("--sources", required=True)
    p.add_argument("--out-dir", required=True)
    _checks_args(p)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("export", help="accepted states -> NEXUS, CSV and provenance")
    p.add_argument("--review", required=True, help="review sheet with a decision column (accept / reject)")
    p.add_argument("--characters", required=True)
    p.add_argument("--taxa", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default="phylorag_matrix", help="file name stem")
    p.add_argument("--base-nexus", help="existing matrix: only its missing cells (?) are filled; nothing is overwritten")
    p.add_argument("--molecular", nargs="*", default=[],
                   help="alignments or trees (FASTA, PHYLIP, NEXUS, Newick): their taxon names are used for matching "
                        "rows, and the overlap is reported")
    p.add_argument("--add-missing-taxa", action="store_true",
                   help="add an empty row (?) for every taxon of the molecular data that has no morphology")
    p.add_argument("--labels", choices=["tip", "taxon", "underscore"], default="tip",
                   help="row names for taxa not matched to the base matrix or molecular data: tip = tip_label "
                        "(else Genus_epithet), taxon = 'Genus epithet', underscore = Genus_epithet")
    p.add_argument("--keep-empty", action="store_true", help="also write taxa without any accepted state")
    p.add_argument("--include-unreviewed", action="store_true",
                   help="also export unreviewed states that carry no flag (a quick look only; marked in the files)")
    p.set_defaults(func=cmd_export)

    a = ap.parse_args(argv)
    if a.cmd == "init" and not a.from_nexus and not a.out_dir:
        ap.error("init needs --from-nexus or --out-dir")
    a.func(a)


if __name__ == "__main__":
    main()
