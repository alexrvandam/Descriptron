"""
descriptron_mcp.catalogue — find the Descriptron programs and how to run them
=============================================================================

The programs are not imported. They are located on disk (inside the installed
descriptron-core / descriptron-vision packages, plus any extra directories the
user names) and run as subprocesses, exactly as `descriptron <program>` runs
them. Nothing about a program's behaviour changes by being called through MCP.

Which interpreter runs them is configurable, because the MCP SDK needs
Python >= 3.10 while an existing analysis environment may be older, and the
GPU programs usually live in their own environment:

    DESCRIPTRON_CORE_PYTHON     interpreter for descriptron-core programs
    DESCRIPTRON_VISION_PYTHON   interpreter for descriptron-vision programs
    DESCRIPTRON_PYTHON          fallback for both
    (default: the interpreter running this server)

    DESCRIPTRON_MCP_TOOL_DIRS   extra program directories, separated by the
                                OS path separator (':' on Linux/macOS, ';' on
                                Windows) — e.g. a development checkout
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# library modules shipped next to the programs; importable, not runnable
_SKIP_DIRS = {"tests", "__pycache__", "data", "biorag_prompts"}


@dataclass
class Program:
    name: str
    path: Path
    package: str            # "core", "vision" or "extra"
    summary: str = ""
    runnable: bool = True
    search_path: list[Path] = field(default_factory=list)   # dirs put on PYTHONPATH

    def as_dict(self) -> dict:
        return {"name": self.name, "package": self.package, "summary": self.summary,
                "path": str(self.path)}


def _package_dir(module: str) -> Path | None:
    """Directory of an installed package, found WITHOUT importing it (vision imports torch)."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def core_data_dir() -> Path | None:
    d = _package_dir("descriptron_core")
    return d / "data" if d and (d / "data").exists() else None


def _summary_py(path: Path) -> tuple[str, bool]:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", False
    runnable = "__main__" in src
    try:
        doc = ast.get_docstring(ast.parse(src)) or ""
    except SyntaxError:
        doc = ""
    lines = [ln.strip() for ln in doc.splitlines() if ln.strip()]
    # skip a heading line that only repeats the file name, and its ==== underline
    lines = [ln for ln in lines if set(ln) - set("=-~")]
    for ln in lines:
        head = ln.split("—")[0].split(" - ")[0].strip()
        if head.replace(".py", "") == path.stem and "—" in ln:
            return ln.split("—", 1)[1].strip(), runnable
        if head.replace(".py", "") != path.stem:
            return ln, runnable
    return "", runnable


def _summary_sh(path: Path) -> str:
    try:
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:15]:
            t = ln.lstrip("#").strip()
            if t and set(t) - set("=-~") and not t.startswith(("shellcheck", "!")):
                return t.split("—", 1)[-1].strip() if path.stem in t else t
    except OSError:
        pass
    return ""


def _scan(root: Path, package: str, extra_path: list[Path]) -> list[Program]:
    out: list[Program] = []
    if not root or not root.is_dir():
        return out
    dirs = [root] + sorted(p for p in root.iterdir() if p.is_dir() and p.name not in _SKIP_DIRS)
    for d in dirs:
        for f in sorted(d.iterdir()):
            if f.name.startswith(("_", ".")) or f.suffix not in (".py", ".sh"):
                continue
            if f.suffix == ".py":
                summary, runnable = _summary_py(f)
            else:
                summary, runnable = _summary_sh(f), True
            out.append(Program(f.stem, f, package, summary, runnable,
                               [f.parent, root, *extra_path]))
    return out


def discover() -> dict[str, Program]:
    """All programs, keyed by name. Earlier sources win on a name clash:
    extra dirs (a development checkout) > core > vision."""
    core_pkg = _package_dir("descriptron_core")
    vision_pkg = _package_dir("descriptron_vision")
    core_tools = core_pkg / "tools" if core_pkg else None
    found: list[Program] = []
    for raw in os.environ.get("DESCRIPTRON_MCP_TOOL_DIRS", "").split(os.pathsep):
        if raw.strip():
            found += _scan(Path(raw).expanduser(), "extra", [])
    if core_tools:
        found += _scan(core_tools, "core", [])
    if vision_pkg:
        # vision programs import core programs by plain name, so core tools go on the path too
        found += _scan(vision_pkg / "tools", "vision", [core_tools] if core_tools else [])
    catalogue: dict[str, Program] = {}
    for p in found:
        catalogue.setdefault(p.name, p)
    return catalogue


def python_for(program: Program) -> str:
    key = {"core": "DESCRIPTRON_CORE_PYTHON", "vision": "DESCRIPTRON_VISION_PYTHON"}.get(program.package)
    for var in filter(None, (key, "DESCRIPTRON_PYTHON")):
        if os.environ.get(var):
            return os.environ[var]
    return sys.executable


def command_for(program: Program, args: list[str]) -> list[str]:
    if program.path.suffix == ".sh":
        return ["bash", str(program.path), *args]
    return [python_for(program), str(program.path), *args]


def environment_for(program: Program) -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(p) for p in program.search_path if p]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    data = core_data_dir()
    if data:
        env.setdefault("DESCRIPTRON_DATA", str(data))
    env.setdefault("MPLBACKEND", "Agg")          # no program may open a window here
    env["PYTHONUNBUFFERED"] = "1"                # so job logs stream
    return env


def resolve(catalogue: dict[str, Program], name: str) -> Program:
    name = name.removesuffix(".py").removesuffix(".sh")
    if name in catalogue:
        return catalogue[name]
    matches = [n for n in catalogue if n.startswith(name)]
    if len(matches) == 1:
        return catalogue[matches[0]]
    if matches:
        raise ValueError(f"'{name}' is ambiguous: {', '.join(sorted(matches))}")
    raise ValueError(f"unknown program '{name}'; list_programs shows what is available")
