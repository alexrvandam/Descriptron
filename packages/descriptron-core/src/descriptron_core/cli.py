"""
descriptron_core.cli — one entry point for the analysis programs
================================================================

The 62 programs in this package are run with `runpy` exactly as they are run
from a shell, rather than being imported as modules. That is deliberate: they
were written as scripts, they use `if __name__ == "__main__"`, and several
import their siblings by plain name. Executing them as `__main__` with their own
directory on `sys.path` preserves that behaviour exactly, so packaging cannot
quietly change what a program does — which matters when the output is a
published species description.

    descriptron --list
    descriptron biorag_key_builder_v1 --help
    biorag-audit --help
"""
from __future__ import annotations

import os
import runpy
import sys
from importlib.resources import files
from pathlib import Path

__all__ = ["main", "tools_dir", "data_path", "available"]


def tools_dir() -> Path:
    return Path(str(files("descriptron_core") / "tools"))


def data_path(*parts: str) -> Path:
    """Prompts, taxon profiles, ontology releases and schemas that ship with the package."""
    return Path(str(files("descriptron_core") / "data")).joinpath(*parts)


def available() -> list[str]:
    d = tools_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.iterdir()
                  if p.suffix in (".py", ".sh") and not p.name.startswith("_"))


def _resolve(name: str) -> Path:
    d = tools_dir()
    for candidate in (d / name, d / f"{name}.py", d / f"{name}.sh"):
        if candidate.exists():
            return candidate
    matches = [t for t in available() if t.startswith(name)]
    if len(matches) == 1:
        return _resolve(matches[0])
    if matches:
        raise SystemExit(f"'{name}' is ambiguous: {', '.join(matches)}")
    raise SystemExit(f"unknown program '{name}'. `descriptron --list` shows them all.")


def run(name: str, argv: list[str] | None = None) -> None:
    path = _resolve(name)
    argv = list(sys.argv[1:] if argv is None else argv)
    # the programs' own directory goes on sys.path so a script that imports a
    # sibling by plain name keeps working, and the packaged prompts and ontology
    # releases are findable without anyone passing a path
    sys.path.insert(0, str(path.parent))
    os.environ.setdefault("DESCRIPTRON_DATA", str(data_path()))
    if path.suffix == ".sh":
        os.execvp("bash", ["bash", str(path), *argv])
    sys.argv = [str(path), *argv]
    runpy.run_path(str(path), run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        print(f"\n{len(available())} programs available; --list shows them.")
        return
    if args[0] in ("--list", "list"):
        for name in available():
            print(name)
        return
    if args[0] in ("--where", "where"):
        print(f"tools: {tools_dir()}\ndata:  {data_path()}")
        return
    run(args[0], args[1:])


def _fixed(script: str):
    def entry() -> None:
        run(script, sys.argv[1:])
    return entry


run_pipeline = _fixed("run_full_pipeline_v2")
run_key_builder = _fixed("biorag_key_builder_v1")
run_confabulation_checker = _fixed("biorag_confabulation_checker_v2")
run_novelty_score = _fixed("biorag_novelty_score_v1")
run_calibrate = _fixed("biorag_calibrate_v1")
run_keys = _fixed("descriptron_credentials")
