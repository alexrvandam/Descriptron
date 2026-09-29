#!/usr/bin/env python3
"""
biorag_provenance_v1.py — stamp every output with what produced it
==================================================================

A number in a manuscript is only checkable if a reader can get from it back to
the code and the inputs that made it. `manuscript_numbers_v18.tsv` already maps
each number to the report file that supplied it; this closes the remaining link,
from the report back to the script, the commit and the exact inputs.

Three lines in a script:

    from biorag_provenance_v1 import stamp, write_json
    ...
    write_json(out_dir / "report.json", payload)        # adds "_provenance"

or, for anything that is not JSON (a CSV, a figure, a directory of tables):

    stamp(out_dir, inputs=[args.matrix, args.coco])     # writes provenance.json

What is recorded: the script and its SHA-256, the git commit and whether the tree
was dirty, the full command line, the interpreter and platform, the run time in
UTC, and a SHA-256 for every named input file. Hashing the inputs is what makes
the record worth anything — "produced by version X from these exact files" is a
claim a sceptic can test, "produced by version X" is not.

Deliberately: no network, no imports beyond the standard library, and every
failure is swallowed. A provenance stamp must never be the reason an analysis
dies.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["provenance", "stamp", "write_json", "sha256"]

_MAX_HASH_BYTES = 2_000_000_000     # 2 GB: beyond this, record the size only


def sha256(path) -> str | None:
    """SHA-256 of a file, or None if it cannot be read. Never raises."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        if p.stat().st_size > _MAX_HASH_BYTES:
            return f"(not hashed: {p.stat().st_size} bytes)"
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _git(*args, cwd=None):
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def provenance(inputs=None, script=None, extra=None) -> dict:
    """The record itself. Safe to call from anywhere; never raises."""
    try:
        script_path = Path(script or sys.argv[0]).resolve()
    except Exception:
        script_path = Path(sys.argv[0] if sys.argv else "unknown")
    repo = script_path.parent if script_path.exists() else None

    # Only report a commit if THIS FILE is tracked by that repository. The
    # Descriptron scripts currently sit untracked inside a clone of
    # facebookresearch/segment-anything-2, so a naive `git rev-parse HEAD` returns
    # SAM2's commit — a hash that says nothing about the code that produced the
    # output, and is worse than no hash because it looks authoritative.
    commit = dirty = tracked = None
    if repo is not None:
        tracked = bool(_git("ls-files", "--error-unmatch", str(script_path), cwd=repo))
        if tracked:
            commit = _git("rev-parse", "HEAD", cwd=repo)
            dirty = bool(_git("status", "--porcelain", "--", str(script_path), cwd=repo))

    rec = {
        "script": script_path.name,
        "script_path": str(script_path),
        "script_sha256": sha256(script_path),
        "git_commit": commit,
        "git_dirty": dirty,
        # a run from an untracked tree cannot be pinned to a version; say so in
        # the record rather than leaving a reader to infer it
        "git_tracked": tracked,
        "git_note": (None if commit else
                     "this script is NOT tracked by any git repository, so the output "
                     "cannot be tied to a version of the code"),
        "command": " ".join(sys.argv),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {},
    }
    for item in (inputs or []):
        if item is None:
            continue
        try:
            rec["inputs"][str(item)] = sha256(item)
        except Exception:
            rec["inputs"][str(item)] = None
    if extra:
        rec["extra"] = extra
    return rec


def stamp(out_dir, inputs=None, script=None, extra=None, name="provenance.json"):
    """Write a provenance record beside an output that is not itself JSON."""
    try:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        rec = provenance(inputs=inputs, script=script, extra=extra)
        (d / name).write_text(json.dumps(rec, indent=2))
        return d / name
    except Exception:
        return None


def write_json(path, payload, inputs=None, script=None, extra=None, indent=2):
    """json.dump with a `_provenance` key added. Drop-in for an existing dump."""
    try:
        rec = provenance(inputs=inputs, script=script, extra=extra)
        if isinstance(payload, dict):
            payload = {**payload, "_provenance": rec}
        else:
            payload = {"data": payload, "_provenance": rec}
    except Exception:
        pass
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=indent, default=str))
    return p


def verify(record_path) -> dict:
    """
    Re-check a stamped record: does the script still hash the same, do the inputs
    still hash the same? This is what a sceptical reader runs.
    """
    rec = json.loads(Path(record_path).read_text())
    rec = rec.get("_provenance", rec)
    out = {"script": None, "inputs": {}, "ok": True}
    now = sha256(rec.get("script_path"))
    out["script"] = ("unchanged" if now and now == rec.get("script_sha256")
                     else "CHANGED or missing")
    if out["script"] != "unchanged":
        out["ok"] = False
    for path, want in (rec.get("inputs") or {}).items():
        got = sha256(path)
        same = (got is not None and got == want)
        out["inputs"][path] = "unchanged" if same else "CHANGED or missing"
        if not same:
            out["ok"] = False
    # records written by run_full_pipeline_v2 also list what the step wrote
    if isinstance(rec.get("outputs"), dict):
        out["outputs"] = {}
        for path, want in rec["outputs"].items():
            if path.startswith("("):
                continue
            got = sha256(path)
            same = (got is not None and got == want)
            out["outputs"][path] = "unchanged" if same else "CHANGED or missing"
            if not same:
                out["ok"] = False
    return out


def which(result_file, search_dir) -> list:
    """Which pipeline step wrote this file? Hash it and look for the hash among the outputs of
    every provenance record under search_dir. A match means this exact content was written by
    that step; no match means the file was changed afterwards or made outside the pipeline."""
    want = sha256(result_file)
    hits = []
    if want is None:
        return hits
    for rp in sorted(Path(search_dir).rglob("*provenance*.json")):
        try:
            rec = json.loads(rp.read_text())
        except Exception:
            continue
        rec = rec.get("_provenance", rec)
        outs = rec.get("outputs") or {}
        if isinstance(outs, dict) and want in outs.values():
            hits.append({"record": str(rp),
                         "step": (rec.get("extra") or {}).get("pipeline_step"),
                         "script": rec.get("script"), "script_sha256": rec.get("script_sha256"),
                         "git_commit": rec.get("git_commit"), "run_utc": rec.get("run_utc"),
                         "command": rec.get("command")})
    return hits


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", help="a provenance.json (or a report with _provenance) to re-check")
    ap.add_argument("--show", action="store_true", help="print a record for this invocation")
    ap.add_argument("--which", help="a result file: find the provenance record of the step that wrote it")
    ap.add_argument("--search", default=".", help="folder to search for provenance records (with --which)")
    a = ap.parse_args()
    if a.verify:
        result = verify(a.verify)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)
    if a.which:
        hits = which(a.which, a.search)
        print(json.dumps(hits, indent=2) if hits else
              "no provenance record lists this exact file content (changed since, or not made by the pipeline)")
        sys.exit(0 if hits else 1)
    if a.show:
        print(json.dumps(provenance(), indent=2))
