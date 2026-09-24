#!/usr/bin/env python3
"""
descriptron_credentials.py — where Descriptron finds your API keys and tokens
=============================================================================

Two keys are ever needed, each by one optional part of Descriptron:

  ANTHROPIC_API_KEY   species descriptions with --llm_backend api
  HF_TOKEN            DINOLand, to download Meta's gated DINOv3 weights once

Where a key is looked up, in this order:

  1. the environment (`export ANTHROPIC_API_KEY=...`) — always wins;
  2. the credentials file, one `NAME=value` per line:
        $DESCRIPTRON_CREDENTIALS, or ~/.config/descriptron/credentials
     You can write this file yourself, or let Descriptron write it:
  3. a prompt, the first time the key is actually needed — only in an interactive
     terminal (or the GUI), never in batch jobs, and never when
     DESCRIPTRON_NO_KEY_PROMPT=1. Pressing Enter skips it; the step that needed the
     key then reports what is missing instead of failing obscurely.

The file is created readable by you only (0600). Keys are never written to logs,
project folders or the Docker image.

    python descriptron_credentials.py --list          # which keys are set (values hidden)
    python descriptron_credentials.py --set HF_TOKEN  # enter or replace one
    python descriptron_credentials.py --remove HF_TOKEN
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import Dict, Optional

KNOWN = {
    "ANTHROPIC_API_KEY": ("species descriptions (--llm_backend api)",
                          "https://console.anthropic.com/settings/keys"),
    "HF_TOKEN": ("DINOLand: downloading Meta's DINOv3 weights (accept the licence first at "
                 "https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)",
                 "https://huggingface.co/settings/tokens"),
}


def credentials_path() -> Path:
    p = os.environ.get("DESCRIPTRON_CREDENTIALS")
    if p:
        return Path(p).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "descriptron" / "credentials"


def read_file(path: Optional[Path] = None) -> Dict[str, str]:
    path = path or credentials_path()
    out: Dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v:
                out[k] = v
    except OSError:
        pass
    return out


def write_file(values: Dict[str, str], path: Optional[Path] = None) -> Path:
    path = path or credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    body = "# Descriptron keys — one NAME=value per line; keep this file private\n" + \
           "".join(f"{k}={v}\n" for k, v in sorted(values.items()))
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(body)
    os.chmod(path, 0o600)
    return path


def load_into_environment() -> Dict[str, str]:
    """Copy keys from the file into os.environ where not already set (the environment wins).
    Returns the names that were loaded. Child processes then inherit them."""
    loaded = {}
    for k, v in read_file().items():
        if not os.environ.get(k):
            os.environ[k] = v
            loaded[k] = v
    return loaded


def prompting_allowed() -> bool:
    if os.environ.get("DESCRIPTRON_NO_KEY_PROMPT", "").strip() not in ("", "0", "false", "no"):
        return False
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def save_key(name: str, value: str) -> Path:
    vals = read_file()
    vals[name] = value
    os.environ[name] = value
    return write_file(vals)


def ensure_key(name: str, purpose: Optional[str] = None, where: Optional[str] = None,
               ask=None) -> Optional[str]:
    """Return the key: environment, then the credentials file, then (first use only) a prompt.

    `ask` replaces the terminal prompt (the GUI passes a dialog); it receives
    (name, purpose, where) and returns the value or None. Nothing is asked when prompting
    is switched off or no one is there to answer; then None is returned."""
    if os.environ.get(name):
        return os.environ[name]
    stored = read_file().get(name)
    if stored:
        os.environ[name] = stored
        return stored
    purpose = purpose or KNOWN.get(name, ("", ""))[0]
    where = where or KNOWN.get(name, ("", ""))[1]
    if os.environ.get("DESCRIPTRON_NO_KEY_PROMPT", "").strip() not in ("", "0", "false", "no"):
        return None
    if ask is not None:
        value = ask(name, purpose, where)
    elif prompting_allowed():
        sys.stderr.write(f"\nDescriptron needs {name} for {purpose}.\n")
        if where:
            sys.stderr.write(f"  Get one at: {where}\n")
        sys.stderr.write(f"  It will be saved (readable by you only) in {credentials_path()}\n"
                         f"  so you are asked only once. Press Enter to skip.\n")
        value = getpass.getpass(f"  {name}: ")
    else:
        return None
    value = (value or "").strip()
    if not value:
        return None
    path = save_key(name, value)
    sys.stderr.write(f"  saved {name} to {path}\n")
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description="Show, set or remove the keys Descriptron uses.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="which keys are set (values hidden) — the default")
    g.add_argument("--set", metavar="NAME", help="enter or replace a key (asked for without echo)")
    g.add_argument("--remove", metavar="NAME", help="delete a key from the file")
    a = ap.parse_args(argv)
    path = credentials_path()
    if a.set:
        v = getpass.getpass(f"{a.set}: ").strip()
        if not v:
            print("nothing entered; unchanged"); return 1
        print(f"saved {a.set} to {save_key(a.set, v)}"); return 0
    if a.remove:
        vals = read_file()
        if vals.pop(a.remove, None) is None:
            print(f"{a.remove} is not in {path}"); return 1
        write_file(vals); print(f"removed {a.remove} from {path}"); return 0
    stored = read_file()
    print(f"credentials file: {path}{'' if path.exists() else ' (not created yet)'}")
    for k in sorted(set(KNOWN) | set(stored)):
        src = "environment" if os.environ.get(k) else ("file" if k in stored else "not set")
        print(f"  {k:18s} {src:12s} {KNOWN.get(k, ('',))[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
