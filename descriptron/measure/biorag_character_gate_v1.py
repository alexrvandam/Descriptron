#!/usr/bin/env python3
"""
biorag_character_gate_v1.py — take the unreliable words out of the treatments
============================================================================

The scorer's repeatability is measured per character by
`biorag_character_reliability_v1.py`. That test decides three things, and until
now only the delimitation step acted on them. This step applies the same verdicts
to the *text*.

**This is not a confabulation check, and the statements it removes are not
necessarily false.** The structures may be exactly as described. What has been
measured is that a second reading of the same image does not return the same
state, so the *reading* is not repeatable — which means the statement cannot be
offered to a reader as an observation they could confirm, and cannot carry weight
in an identification. It is removed from the treatment and recorded in the report,
not deleted from the record: the character stays in the scored matrix, and the
reliability table says exactly why it was set aside for this taxon.

  verdict        what happens to the wording
  ─────────────  ────────────────────────────────────────────────────────────
  use            left exactly as written
  use coarse     left exactly as written. The bands exist to score a character,
                 not to reword a description: a character that clears the bar at
                 band level has passed, and its own wording stands. (`--coarsen`
                 will substitute the band anyway, but it is off, and the band
                 names do not always carry the same sense in prose: a margin
                 that is "straight" is not "flattened", "arcuate" sulci are not
                 "convex" ones.)
  flag           the statement is removed: a clause that says nothing else goes
                 whole, a clause that says more loses just that statement, and a
                 clause that labels the measurements after it keeps its label

Nothing here is taxon-specific: the vocabulary comes from the character list
(the shared one, or `descriptive_characters` in the taxon profile) and the
verdicts come from the retest of *this* taxon's own images, so a group whose
sculpture reads cleanly keeps its sculpture words while one whose outline words
wander loses them.

A word that belongs to more than one character (crenulate is both a surface
pattern and a kind of margin incision) is only touched when EVERY character it
could belong to was flagged — otherwise it is left alone and reported.

  python biorag_character_gate_v1.py \\
      --descriptions_dir "$M/descriptions_v2" \\
      --reliability      "$M/descriptive_states/character_reliability.json" \\
      --taxon_profile    <profile.yaml> \\
      --out_dir          "$M/descriptions_v2/character_gate"      [--dry_run]

Writes the edited `<code>_treatment.json` and `<code>.txt` in place (the
previous versions are saved as `<code>_treatment_before_gate_<date>.json` and
`<code>_before_gate_<date>.txt`), plus a report of every edit made.
Re-run `biorag_ontology_annotator_v2.py` and the DOCX build afterwards.
"""

import argparse
import json
import re
from difflib import SequenceMatcher
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                    # noqa: E402
from biorag_llm_backend import (add_backend_args, client_from_args,    # noqa: E402
                                make_llm_client)
from biorag_novelty_score_v1 import (DESCRIPTIVE_CHARACTERS, MODIFIERS,  # noqa: E402
                                     canonical_states, coarse_map, state_ranks)

VERSION = "1.0"
TEXT_FIELDS = ("diagnosis", "sexual_dimorphism", "remarks")            # plain strings
COLOUR_HINT = re.compile(r"\b(black|brown|ochreous|yellow|white|pale|dark|red|orange|green|blue|grey|"
                         r"gray|fuscous|testaceous|castaneous|piceous|infuscate\w*|banded|bicolou?red)\b",
                         re.I)
# a band name is a phrase already; a few read better with a word added
BAND_PHRASES = {"moderate": "moderately elongate", "flat": "flattened", "shining": "shining",
                "dull": "dull", "compact": "compact", "elongate": "elongate"}


def band_phrase(band: str) -> str:
    return BAND_PHRASES.get(band, band)


def vocabulary(characters: Dict) -> Dict[str, Set[str]]:
    """word -> the characters it could belong to (a word may serve two)."""
    v: Dict[str, Set[str]] = defaultdict(set)
    for name, cfg in characters.items():
        for state in state_ranks(cfg):
            v[state.lower()].add(name)
    return v


def decisions(characters: Dict, reliability: Dict) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """word -> "strike" | "coarsen" | "keep", and word -> {character: band phrase}."""
    verdict = {}
    for key, v in (("use", "use"), ("use_coarse", "use coarse"), ("flag", "flag")):
        for ch in reliability.get(key, []):
            verdict[ch] = v
    vocab = vocabulary(characters)
    action, bands = {}, {}
    for word, chars in vocab.items():
        vs = {verdict.get(c, "use") for c in chars}
        if vs == {"flag"}:                       # every character it could be was flagged
            action[word] = "strike"
        elif vs <= {"use coarse", "flag"} and "use coarse" in vs and len(chars) == 1:
            ch = next(iter(chars))
            band = coarse_map(ch).get(word)
            if band and band.lower() != word:
                action[word] = "coarsen"
                bands[word] = {"character": ch, "band": band_phrase(band)}
        else:
            action[word] = "keep"
    return action, bands


def structure_names(profile: Dict) -> Set[str]:
    """Every word by which this taxon's structures are named, from the profile — the labels a
    measurement hangs off. Taxon-agnostic: the profile supplies them."""
    words: Set[str] = set()
    for cat, info in (profile.get("structures") or {}).items():
        for name in (cat, (info or {}).get("term") or ""):
            for tok in re.findall(r"[a-z][a-z-]{2,}", str(name).lower()):
                words.add(tok)
    words.update({"cell", "segment", "segments", "flagellomere", "flagellomeres", "lobe", "margin"})
    return words


def names_a_structure(clause: str, names: Set[str]) -> bool:
    return any(w in names for w in re.findall(r"[a-z][a-z-]+", clause.lower()))


TRAILING_FILLER = re.compile(r"\b(?:is|are|was|were|not|the|a|an|its|with|and|or|to)\s*$", re.I)


def trim_filler(clause: str) -> str:
    """After a state word is removed, drop the copula or article it hung on: "The paramere is
    digitiform" -> "The paramere", not "The paramere is"."""
    c = clean(clause)
    while True:
        c2 = clean(TRAILING_FILLER.sub("", c))
        if c2 == c:
            return c
        c = c2


def carries_content(clause: str, vocab: Set[str]) -> bool:
    """Is there anything left in the clause worth printing? A measurement, another
    descriptive word or a colour word means yes; a bare structure name means no.

    The number has to look like a measurement: the digit in a structure's own name
    ("cell cu1", "flagellomere 3") is part of the label, not a statement, so
    "Cell cu1 triangular to deltoid" must end up removed entirely rather than
    reduced to "Cell cu1"."""
    if MEASUREMENT.search(clause):
        return True
    if COLOUR_HINT.search(clause):
        return True
    return any(w in vocab for w in re.findall(r"[a-z-]+", clause.lower()))


RANGE_NEAR = re.compile(r"\b(?:to|or|and|than|between)\b", re.I)
MAX_SAFE_WORDS = 5          # "apex acuminate" yes; a clause with a subordinate phrase no


CONNECTOR = r"(?:to|or|and|through|versus|vs\.?)"
# a number that is a measurement rather than part of a structure's name ("cell cu1")
MEASUREMENT = re.compile(r"\d+[.,]\d|\d+\s*(?:mm|cm|\u00b5m|um|nm|%)\b|\u2013|\u2014")


def delete_word(clause: str, word: str, mods: str) -> str:
    """Remove one state word from a clause WITHOUT leaving a dangling connector.

    Most of the hard cases are coordinations — "smooth to weakly striate", "cuneate to
    digitiform in outline", "not clavate or spatulate" — where the word must go together
    with the connector that joins it to its neighbour, or the sentence loses its grammar.
    Handled in order: parenthetical, right-hand side of a coordination, left-hand side,
    then a bare adjective.
    """
    # any adverb may qualify the state, not only the scoring modifiers: "broadly triangular"
    w = rf"(?:(?:{mods})\s+|\w+ly\s+)*{re.escape(word)}"
    for pattern in (rf"\s*\(\s*{w}\s*\)",              # "crescent-like (falcate)"
                    rf"\s*/\s*{w}\b",                    # "globular/triangular/elongate"
                    rf"\b{w}\s*/\s*",
                    rf"\s+{CONNECTOR}\s+{w}\b",          # "smooth to striate"
                    rf"\b{w}\s+{CONNECTOR}\s+",          # "striate to smooth"
                    rf"\b{w}\b"):                        # on its own
        new = re.sub(pattern, "", clause, count=1, flags=re.I)
        if new != clause:
            return new
    return clause


def dangling(clause: str, original: str = "") -> bool:
    """Did a deletion leave the clause hanging — a preposition or connector with nothing
    after it, a doubled connector, a stray adverb or a bracket it has broken?

    Judged against the clause as it was: a clause split off mid-parenthesis was already
    unbalanced before anything was removed, and whitespace is not evidence of damage
    because clean() normalises it."""
    c = clean(clause).rstrip(".")
    if not c:
        return False
    if re.search(rf"\b(?:{CONNECTOR}|in|of|with|at|on|from|the|a|an)\s*$", c, re.I):
        return True
    if re.search(rf"\b{CONNECTOR}\s+{CONNECTOR}\b", c, re.I):
        return True
    # "cuneate to broadly" — the adjective the adverb qualified has gone. But a clause that
    # already ended in an adverb ("compressed dorso-ventrally") is not damaged by the edit.
    if re.search(r"\b\w+ly$", c, re.I) and not re.search(r"\b\w+ly$", clean(original).rstrip("."), re.I):
        return True
    if original:                                 # brackets broken BY the edit, not before it
        o = clean(original)
        if (c.count("(") - c.count(")")) != (o.count("(") - o.count(")")):
            return True
    elif c.count("(") != c.count(")"):
        return True
    return bool(re.search(r",\s*,", c))


def clean(s: str) -> str:
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s+([;,.])", r"\1", s)
    s = re.sub(r"[;,]\s*\.", ".", s)
    return re.sub(r"^[;,]\s*", "", s).strip()


def edit_block(text: str, action: Dict[str, str], bands: Dict[str, Dict[str, str]],
               vocab: Set[str], edits: List[Dict], where: Dict, names: Set[str] = frozenset()) -> str:
    """Apply the verdicts clause by clause, and only where the edit is safe.

    Two things learned the hard way, both enforced here:

    * A word is never deleted from the middle of a clause. "Cell cu1 triangular to
      deltoid" does not become "Cell cu1 to". Either the whole clause goes (because
      the flagged character was all it said) or nothing is touched and the clause is
      reported for rewriting.
    * A band label is only substituted when it occupies the same grammatical slot on
      its own — "apex acuminate" -> "apex pointed". Inside a range or beside other
      qualifiers it is reported instead, because the band names bin scores and do not
      always carry the same sense in prose ("margin straight" is not "margin
      flattened"; "arcuate sulci" are not "convex sulci").
    """
    if not text:
        return text
    out_sentences = []
    for sentence in re.split(r"(?<=\.)\s+", text):
        if not sentence.strip():
            continue
        parts = re.split(r"(\s*[;,]\s*)", sentence)          # keep the separators
        kept: List[str] = []
        for i_part, part in enumerate(parts):
            # do the numbers later in this sentence depend on this clause to say what they measure?
            anchors_what_follows = (bool(MEASUREMENT.search("".join(parts[i_part + 1:])))
                                    and names_a_structure(part, names))
            if re.fullmatch(r"\s*[;,]\s*", part or ""):
                kept.append(part)
                continue
            clause = part
            words = {w for w in re.findall(r"[a-z-]+", clause.lower())
                     if action.get(w) in ("strike", "coarsen")}
            mods = "|".join(sorted(MODIFIERS, key=len, reverse=True))
            struck = sorted(w for w in words if action[w] == "strike")
            if struck:
                stripped = clause
                for word in sorted(struck, key=len, reverse=True):
                    stripped = delete_word(stripped, word, mods)
                if carries_content(clean(stripped), vocab):
                    if dangling(stripped, clause):
                        edits.append({**where, "action": "needs rewrite (struck word in a clause that "
                                                        "says more)", "word": ", ".join(struck),
                                      "character": "", "replacement": "",
                                      "before": part.strip(), "after": part.strip()})
                    else:
                        new_clause = clean(stripped)
                        edits.append({**where, "action": "statement removed", "word": ", ".join(struck),
                                      "character": "", "replacement": "",
                                      "before": part.strip(), "after": new_clause})
                        kept.append((" " if part.startswith(" ") else "") + new_clause)
                        continue
                elif anchors_what_follows:
                    # This clause names the structure that the numbers later in the same
                    # sentence belong to ("Cell cu1 triangular to deltoid; length 0.397-0.504 mm").
                    # Deleting it whole would leave those measurements attached to whatever was
                    # named before, which is exactly the mis-attribution the audit exists to catch,
                    # so the label stays and only the unreliable statement goes.
                    anchor = trim_filler(stripped)
                    edits.append({**where, "action": "statement removed (label kept for the "
                                                     "measurements that follow)",
                                  "word": ", ".join(struck), "character": "", "replacement": "",
                                  "before": part.strip(), "after": anchor})
                    kept.append((" " if part.startswith(" ") else "") + anchor)
                    continue
                else:
                    edits.append({**where, "action": "clause removed", "word": ", ".join(struck),
                                  "character": "", "replacement": "",
                                  "before": part.strip(), "after": ""})
                    if kept and re.fullmatch(r"\s*[;,]\s*", kept[-1] or ""):
                        kept.pop()
                    continue
            for word in sorted((w for w in words if action[w] == "coarsen"), key=len, reverse=True):
                new_word = bands[word]["band"]
                rx = re.compile(rf"\b(?:(?:{mods})\s+)?{re.escape(word)}\b", re.I)
                only_state = sum(1 for w in re.findall(r"[a-z-]+", clause.lower()) if w in vocab) == 1
                bare = clause.strip().rstrip(".")
                safe = (only_state                                    # nothing else qualifies it
                        and not RANGE_NEAR.search(bare)               # not one end of a range
                        and re.search(rf"{re.escape(word)}$", bare, re.I) is not None   # final slot
                        and len(re.findall(r"[A-Za-z]+", bare)) <= MAX_SAFE_WORDS)
                if safe:
                    clause2 = rx.sub(new_word, clause)
                    if clause2 != clause:
                        edits.append({**where, "action": "coarsened", "word": word,
                                      "character": bands[word]["character"], "replacement": new_word,
                                      "before": part.strip(), "after": clause2.strip()})
                        clause = clause2
                else:
                    edits.append({**where, "action": "needs rewrite (finer than the scorer can repeat)",
                                  "word": word, "character": bands[word]["character"],
                                  "replacement": new_word, "before": part.strip(),
                                  "after": part.strip()})
            kept.append(clause)
        s = clean("".join(kept))
        if re.sub(r"[^A-Za-z0-9]", "", s):
            out_sentences.append(s if s.endswith((".", ";")) else s + ".")
        else:
            edits.append({**where, "action": "sentence removed", "word": "", "character": "",
                          "replacement": "", "before": sentence.strip(), "after": ""})
    return " ".join(out_sentences)


def gate_treatment(treat: Dict, action, bands, vocab, code: str, edits: List[Dict],
                   names: Set[str] = frozenset(), engine: str = "python",
                   struck_chars: List[str] = (), client=None, model: str = "") -> Dict:
    def run(text, where):
        if engine == "llm":
            return edit_block_llm(text, list(struck_chars),
                                  {w for w, v in action.items() if v == "strike"},
                                  edits, where, client, model,
                                  {w for w, v in action.items() if v != "strike"})
        return edit_block(text, action, bands, vocab, edits, where, names)

    for field in TEXT_FIELDS:
        if isinstance(treat.get(field), str):
            treat[field] = run(treat[field], {"species": code, "field": field, "section": ""})
    for block in treat.get("description") or []:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            block["text"] = run(block["text"], {"species": code, "field": "description",
                                                "section": block.get("section", "")})
    return treat


REWRITE_SYSTEM = (
    "You are editing a taxonomic description. You may ONLY DELETE words from the sentence you are "
    "given. You may not add, reorder, replace or rephrase anything, and you may not introduce any "
    "number, name or word that is not already there. Fixing the punctuation left behind by a deletion "
    "is allowed. Return only the edited sentence, or an empty line if nothing should remain.")


def removed_tokens(before: str, after: str) -> List[str]:
    """The words an edit took out, in order."""
    a = re.findall(r"[A-Za-z0-9.\u2013-]+", before or "")
    b = re.findall(r"[A-Za-z0-9.\u2013-]+", after or "")
    gone: List[str] = []
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            gone += a[i1:i2]
    return gone


def collateral(before: str, after: str, keepable: Set[str]) -> List[str]:
    """Statements the edit removed that it had no business removing: words belonging to a
    character that PASSED the repeatability test, or a colour word.

    This is the guard the subsequence check does not give. A subsequence cannot ADD anything,
    but it can drop a neighbouring statement, and the numeric audit will not see it because
    nothing numeric changed. Measured on this data, a model asked to delete two characters
    also deleted a statement about a character that had passed in 47% of its edits."""
    out = []
    for w in removed_tokens(before, after):
        lw = w.lower().strip(".,;")
        if lw in keepable or COLOUR_HINT.fullmatch(lw):
            out.append(w)
    return out


def tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def is_deletion_only(before: str, after: str) -> bool:
    """Is `after` exactly `before` with words removed? Checked as a subsequence, so the
    model cannot slip in a value, a name or a claim while it edits."""
    a, b = tokens(after), tokens(before)
    it = iter(b)
    return all(t in it for t in a)


def rewrite_sentence(client, model: str, sentence: str, character: str, why: str) -> str:
    """Ask for the statement about one character to be deleted, and accept the answer only
    if it really is a deletion."""
    msg = (f"Delete from this sentence every statement about the {character} of the structure, "
           f"because {why}. Keep every other statement word for word.\n\nSentence:\n{sentence}")
    try:
        r = client.messages.create(model=model, max_tokens=700, system=REWRITE_SYSTEM,
                                   messages=[{"role": "user", "content": msg}])
        out = (r.content[0].text or "").strip().strip('"')
    except Exception:  # noqa: BLE001
        return ""
    out = out.splitlines()[0].strip() if out else ""
    if out and not is_deletion_only(sentence, out):
        return ""                      # the model wrote something new: refuse it
    return out


def edit_block_llm(text: str, struck_chars: List[str], struck_words: Set[str], edits: List[Dict],
                   where: Dict, client, model: str, keepable: Set[str] = frozenset()) -> str:
    """The same job done by the model instead of by the rules, sentence by sentence.

    The model is allowed to do exactly one thing: delete. Three guards are applied to every
    reply, and any failure leaves the sentence untouched:
      1. the reply's words must be a subsequence of the original (nothing added, nothing
         reordered, no value or name invented);
      2. every measurement in the sentence must survive (a description may not quietly lose
         its numbers);
      3. no struck word may remain, or the edit did not do its job;
      4. nothing may be removed that belongs to a character which PASSED the test, nor any
         colour word — the deletion has to be confined to what was asked for.
    Guard 4 is the one that matters in practice: guards 1-3 all passed on a run in which the
    model nevertheless deleted a neighbouring statement in 47% of its edits.
    """
    if not text:
        return text
    out = []
    for sentence in re.split(r"(?<=\.)\s+", text):
        hits = sorted({w for w in re.findall(r"[a-z-]+", sentence.lower()) if w in struck_words})
        if not hits:
            out.append(sentence)
            continue
        others = sorted({w for w in re.findall(r"[a-z-]+", sentence.lower())
                         if w in keepable})[:25]
        msg = ("Delete from this sentence every statement about the following characters, because a "
               "second reading of the same image does not reproduce them: "
               + ", ".join(struck_chars) + ".\n"
               "The words to remove, with whatever qualifies them: " + ", ".join(hits) + ".\n"
               "Keep every other statement word for word, keep all measurements, and keep the name of "
               "any structure that the measurements belong to."
               + (("\nThese words describe characters that passed and must all still be present: "
                   + ", ".join(others) + ".") if others else "")
               + "\n\nSentence:\n" + sentence)
        try:
            r = client.messages.create(model=model, max_tokens=900, system=REWRITE_SYSTEM,
                                       messages=[{"role": "user", "content": msg}])
            new = (r.content[0].text or "").strip().strip('"')
        except Exception as e:  # noqa: BLE001
            edits.append({**where, "action": f"llm call failed ({type(e).__name__})", "word": ", ".join(hits),
                          "character": "", "replacement": "", "before": sentence, "after": sentence})
            out.append(sentence)
            continue
        new = " ".join(line.strip() for line in new.splitlines() if line.strip())
        why = ""
        if new and not is_deletion_only(sentence, new):
            why = "refused: not a deletion"
        elif MEASUREMENT.findall(sentence) != MEASUREMENT.findall(new or ""):
            why = "refused: a measurement was lost"
        elif new and any(w in re.findall(r"[a-z-]+", new.lower()) for w in hits):
            why = "refused: the struck word is still there"
        else:
            over = collateral(sentence, new or "", keepable)
            if over:
                why = f"refused: it also removed {', '.join(sorted(set(over))[:6])}"
        if why:
            edits.append({**where, "action": f"llm {why}", "word": ", ".join(hits), "character": "",
                          "replacement": "", "before": sentence, "after": sentence})
            out.append(sentence)
            continue
        edits.append({**where, "action": "llm sentence removed" if not new else "llm edited",
                      "word": ", ".join(hits), "character": "", "replacement": "",
                      "before": sentence, "after": new})
        if new:
            out.append(new)
    return " ".join(out)


def replace_in_treatment(treat: Dict, before: str, after: str) -> bool:
    """Swap one clause for its edited form (or drop it) wherever it occurs in the treatment."""
    done = False

    def swap(text: str) -> str:
        nonlocal done
        if not text or before not in text:
            return text
        done = True
        return clean(text.replace(before, after))

    for field in TEXT_FIELDS:
        if isinstance(treat.get(field), str):
            treat[field] = swap(treat[field])
    for block in treat.get("description") or []:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            block["text"] = swap(block["text"])
    return done


def apply_rewrites(treat: Dict, my_edits: List[Dict], client, model: str, coarse_too: bool) -> int:
    """Send the clauses that could not be edited safely back as deletions."""
    n = 0
    for e in my_edits:
        if not e["action"].startswith("needs rewrite"):
            continue
        if "finer than" in e["action"] and not coarse_too:
            continue
        character = e["character"] or "shape or surface pattern"
        why = ("a second reading of the same image does not reproduce it"
               if "struck word" in e["action"] else
               f"it is finer than the scoring reproduces; only \"{e['replacement']}\" is supported")
        out = rewrite_sentence(client, model, e["before"], character, why)
        if out == e["before"]:
            e["action"] = "rewrite declined by the model"
            continue
        if not out and not is_deletion_only(e["before"], ""):
            e["action"] = "rewrite refused (not a deletion)"
            continue
        if replace_in_treatment(treat, e["before"], out):
            e["action"] = "rewritten" if out else "clause removed (rewrite)"
            e["after"] = out
            n += 1
        else:
            e["action"] = "rewrite not applied (clause not found)"
    return n


def main():
    ap = argparse.ArgumentParser(description="Apply the scorer-reliability verdicts to the treatment text")
    ap.add_argument("--descriptions_dir", required=True)
    ap.add_argument("--reliability", required=True, help="character_reliability.json")
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--out_dir", default=None, help="where the report goes (default: descriptions_dir)")
    ap.add_argument("--key", default=None, help="key text to check as well (reported, never edited)")
    ap.add_argument("--coarsen", action="store_true",
                    help="also replace a reliable-only-at-band-level word with its band name. OFF by "
                         "default: the bands are a scoring device, and a character that passes at band "
                         "level keeps the wording it was given")
    ap.add_argument("--dry_run", action="store_true", help="report what would change and stop")
    ap.add_argument("--engine", choices=["python", "llm"], default="python",
                    help="'python' (default) makes the deletions by rule: deterministic, free, needing "
                         "no account, and it removes only what it was told to. 'llm' asks a model "
                         "instead and reads better in connected prose, but on this data it removed five "
                         "times as many words as the rules and, in 47%% of its edits, also deleted a "
                         "statement about a character that had PASSED the test - which neither the "
                         "subsequence check nor the numeric audit can see, because nothing was added "
                         "and no number changed. Guard 4 (collateral) was written for exactly that. "
                         "Compare the two with biorag_gate_compare_v1.py before trusting either on a "
                         "new taxon")
    ap.add_argument("--rewrite", action="store_true",
                    help="for a flagged character sitting in a clause that says more, ask the model to "
                         "delete just that statement. The answer is accepted only if it is the original "
                         "sentence with words removed (checked as a subsequence), so no new claim, name "
                         "or number can enter the text this way")
    ap.add_argument("--rewrite_coarse", action="store_true",
                    help="with --coarsen, also rewrite the statements that are finer than the scoring "
                         "reproduces; ignored otherwise")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    add_backend_args(ap)
    a = ap.parse_args()

    d = Path(a.descriptions_dir)
    out = Path(a.out_dir) if a.out_dir else d
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    characters = {k: dict(v) for k, v in DESCRIPTIVE_CHARACTERS.items()}
    characters.update(profile.get("descriptive_characters") or {})
    rel = json.loads(Path(a.reliability).read_text())
    action, bands = decisions(characters, rel)
    if not a.coarsen:
        action = {w: ("keep" if v == "coarsen" else v) for w, v in action.items()}
    vocab = set(vocabulary(characters))
    names = structure_names(profile)
    struck_words = sorted(w for w, v in action.items() if v == "strike")
    print(f"verdicts: {len(rel.get('use', []))} characters used as scored, "
          f"{len(rel.get('use_coarse', []))} coarsened, {len(rel.get('flag', []))} struck "
          f"({', '.join(rel.get('flag', [])) or 'none'})")
    print(f"  words struck: {', '.join(struck_words) or 'none'}")
    print(f"  words coarsened: {sum(1 for v in action.values() if v == 'coarsen')} "
          f"({'--coarsen given' if a.coarsen else 'bands are for scoring only; wording left as written'})")

    edits: List[Dict] = []
    client = None
    stamp = datetime.now().strftime("%Y%m%d")
    n_species = 0
    for tj in sorted(d.glob("*/*_treatment.json")):
        code = tj.parent.name
        treat = json.loads(tj.read_text())
        before = json.dumps(treat, sort_keys=True)
        n0 = len(edits)
        if a.engine == "llm" and client is None:
            client = client_from_args(a, log_path=a.llm_log or str(out / "llm_calls.jsonl"))
        treat = gate_treatment(treat, action, bands, vocab, code, edits, names,
                               a.engine, rel.get("flag", []), client, a.model)
        if (a.rewrite or a.rewrite_coarse) and not a.dry_run:
            if client is None:
                client = make_llm_client(a.llm_backend, claude_bin=a.claude_bin, cc_model=a.cc_model,
                                         log_path=a.llm_log or str(out / "llm_calls.jsonl"))
            apply_rewrites(treat, edits[n0:], client, a.model, a.rewrite_coarse)
        if json.dumps(treat, sort_keys=True) == before:
            continue
        n_species += 1
        if a.dry_run:
            continue
        tj.with_name(f"{code}_treatment_before_gate_{stamp}.json").write_text(
            before if False else json.dumps(json.loads(before), indent=1, ensure_ascii=False),
            encoding="utf-8")
        treat.setdefault("_validation", {})["character_gate"] = {
            "version": VERSION, "applied": datetime.now().isoformat(),
            "edits": len(edits) - n0,
            "characters_struck": rel.get("flag", []),
            "characters_coarsened": rel.get("use_coarse", []) if a.coarsen else []}
        tj.write_text(json.dumps(treat, indent=1, ensure_ascii=False), encoding="utf-8")
        # the monograph text carries the same sentences verbatim: apply the same edits to it
        txt = tj.with_name(f"{code}.txt")
        if txt.exists():
            body = txt.read_text(encoding="utf-8")
            txt.with_name(f"{code}_before_gate_{stamp}.txt").write_text(body, encoding="utf-8")
            for e in edits[n0:]:
                if e["before"] and e["before"] in body:
                    body = body.replace(e["before"], e["after"])
            body = re.sub(r"\s{2,}", " ", body)
            txt.write_text(body, encoding="utf-8")

    key_hits: List[Dict] = []
    if a.key and Path(a.key).exists():
        for i, line in enumerate(Path(a.key).read_text(encoding="utf-8").splitlines(), 1):
            hit = sorted({w for w in re.findall(r"[a-z-]+", line.lower()) if action.get(w) == "strike"})
            if hit:
                key_hits.append({"line": i, "words": ", ".join(hit), "text": line.strip()[:200]})

    by_action = Counter(e["action"] for e in edits)
    report = {"version": VERSION, "generated": datetime.now().isoformat(),
              "dry_run": bool(a.dry_run), "coarsen": bool(a.coarsen), "engine": a.engine,
              "reliability": str(a.reliability),
              "characters": {"use": rel.get("use", []), "use_coarse": rel.get("use_coarse", []),
                             "flag": rel.get("flag", [])},
              "words_struck": struck_words,
              "species_edited": n_species, "edits": len(edits),
              "by_action": dict(by_action),
              "by_character": dict(Counter(e["character"] for e in edits if e["character"])),
              "key_lines_with_struck_words": key_hits}
    (out / "character_gate_report.json").write_text(json.dumps(report, indent=2))
    if edits:
        import pandas as pd
        pd.DataFrame(edits).to_csv(out / "character_gate_edits.tsv", sep="\t", index=False)
    print(f"{'would edit' if a.dry_run else 'edited'} {n_species} treatments, {len(edits)} changes: "
          + ", ".join(f"{k} {v}" for k, v in by_action.items()))
    if key_hits:
        print(f"  NOTE: {len(key_hits)} key lines still use a struck word — the key is built from "
              f"measured characters, so check the wording by hand")
    print(f"report -> {out / 'character_gate_report.json'}")


if __name__ == "__main__":
    main()
