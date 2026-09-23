#!/usr/bin/env python3
"""
biorag_gate_compare_v1.py — rules or model? compare the two gate engines
=======================================================================

`biorag_character_gate_v1.py` can remove the statements about an unreliable
character either by rule (`--engine python`) or by asking a model to delete them
(`--engine llm`). Both are constrained to deletion, but they do not necessarily
delete the same things. This compares two gated copies of the same treatments
against the ungated original and says where they agree, where they differ, and
what each one did that the other did not — so the choice is made on evidence for
each taxon rather than by preference.

  python biorag_gate_compare_v1.py \\
      --original "$M/descriptions.bak_pre_gate" \\
      --a "$S/gate_py"  --label_a "rules" \\
      --b "$S/gate_llm" --label_b "model" \\
      --out_dir "$M/descriptions/gate_comparison"

Writes `gate_comparison.tsv` (every block where anything changed) and
`gate_comparison.json` (the counts), and prints the summary.
"""

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List

VERSION = "1.0"


def blocks(treat: Dict) -> Dict[str, str]:
    """species text as {where: text} — the same keys in every copy."""
    out = {}
    for field in ("diagnosis", "sexual_dimorphism", "remarks"):
        if isinstance(treat.get(field), str):
            out[field] = treat[field]
    for i, b in enumerate(treat.get("description") or []):
        if isinstance(b, dict) and isinstance(b.get("text"), str):
            out[f"description/{b.get('section') or i}"] = b["text"]
    return out


def read(d: Path) -> Dict[str, Dict[str, str]]:
    out = {}
    for tj in sorted(Path(d).glob("*/*_treatment.json")):
        out[tj.parent.name] = blocks(json.loads(tj.read_text()))
    return out


def words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9.–-]+", s or "")


def removed(before: str, after: str) -> List[str]:
    """Which words the edit took out (order preserved)."""
    a, b = words(before), words(after)
    gone = []
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            gone += a[i1:i2]
    return gone


def main():
    ap = argparse.ArgumentParser(description="Compare two engines of the character gate")
    ap.add_argument("--original", required=True, help="the treatments before either gate ran")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    O, A, B = read(Path(args.original)), read(Path(args.a)), read(Path(args.b))
    rows: List[Dict] = []
    same = both = only_a = only_b = 0
    for code in sorted(set(O) & set(A) & set(B)):
        for where, o in O[code].items():
            a, b = A[code].get(where, o), B[code].get(where, o)
            ca, cb = a != o, b != o
            if not ca and not cb:
                continue
            if ca and cb:
                both += 1
            elif ca:
                only_a += 1
            else:
                only_b += 1
            agree = a.strip() == b.strip()
            same += int(agree)
            ra, rb = removed(o, a), removed(o, b)
            rows.append({"species": code, "where": where,
                         "identical_result": agree,
                         f"{args.label_a}_changed": ca, f"{args.label_b}_changed": cb,
                         f"{args.label_a}_words_removed": len(ra),
                         f"{args.label_b}_words_removed": len(rb),
                         f"removed_only_by_{args.label_a}": " ".join(w for w in ra if w not in rb),
                         f"removed_only_by_{args.label_b}": " ".join(w for w in rb if w not in ra),
                         "original": o, args.label_a: a, args.label_b: b})
    n = len(rows)
    summary = {"version": VERSION, "blocks_touched": n,
               "identical_result": same,
               "agreement_percent": round(100 * same / n, 1) if n else None,
               f"changed_by_{args.label_a}_only": only_a,
               f"changed_by_{args.label_b}_only": only_b,
               "changed_by_both": both,
               f"words_removed_{args.label_a}": int(sum(r[f"{args.label_a}_words_removed"] for r in rows)),
               f"words_removed_{args.label_b}": int(sum(r[f"{args.label_b}_words_removed"] for r in rows))}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_comparison.json").write_text(json.dumps(summary, indent=2))
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out / "gate_comparison.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=1))
    for r in rows:
        if not r["identical_result"]:
            print(f"\n[{r['species']} {r['where']}]")
            print(f"  only {args.label_a} removed: {r[f'removed_only_by_{args.label_a}'][:160]}")
            print(f"  only {args.label_b} removed: {r[f'removed_only_by_{args.label_b}'][:160]}")
    print(f"\n-> {out / 'gate_comparison.tsv'}")


if __name__ == "__main__":
    main()
