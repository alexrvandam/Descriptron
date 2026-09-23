"""
descriptron_mcp.server — the Descriptron stack as an MCP server
===============================================================

Every analysis program in descriptron-core and every mask/landmark program in
descriptron-vision becomes callable by an MCP client (Claude Desktop, Claude
Code, or any other), together with readers for their inputs and outputs and
two curated tools for the step that matters most — writing a treatment from
the evidence and auditing it independently.

    descriptron-mcp              # stdio server, what MCP clients launch
    descriptron-mcp --check      # print what it found and exit

The annotation GUI is not exposed: annotation is interactive by nature, and the
server consumes the COCO files it writes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:                                             # mcp >= 2
    from mcp.server.mcpserver import Image, MCPServer
except ImportError:                              # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer, Image  # type: ignore[no-redef]

from . import __version__, catalogue, data_tools, jobs

INSTRUCTIONS = """\
Descriptron: morphology-driven species descriptions, keys and delimitation.

Typical order: COCO annotations -> measurements / semilandmark GPA / colour /
texture (run_full_pipeline_v2 chains them) -> compile -> biorag_key_feature_filter_v2
(the Tier-1 matrix) -> biorag_key_builder_v1 (key) -> treatments -> independent audit.

Use list_programs to see the programs and program_help before running one; run
short programs with run_program and anything long (pipelines, SAM2-PAL, DINOLand,
detector training) with start_job, then poll job_status.

Programs that can call a language model take --llm-backend (run_full_pipeline_v2
spells it --llm_backend and defaults to api, which needs an API key). The key tree,
matrix, audits and delimitation need no model. For single programs, pass
--llm-backend none unless the user asks for a model (a model only rewords key
couplets; you can write prose yourself from species_evidence).
For run_full_pipeline_v2, ASK the user which backend to use before starting:
claude-code (their Claude subscription; not available inside the Docker image) or
api (an API key). Its 'biorag' step writes the image descriptions, and is where
literature given with --pdf_dir is used; without a model that step stops the
pipeline, so for a numbers-only run pass --llm_backend none --skip_steps biorag.

When writing a species treatment yourself: get the numbers from species_evidence,
never from memory; put each number next to the structure it measures; keep
statistical results in Remarks; then call audit_treatment and fix ONLY the
sentences it flags. The audit is independent of whoever wrote the text.
Read the resource descriptron://workflow for the full recipe.
"""

mcp = MCPServer("descriptron", instructions=INSTRUCTIONS)
_CATALOGUE: dict[str, catalogue.Program] | None = None


def _cat(refresh: bool = False) -> dict[str, catalogue.Program]:
    global _CATALOGUE
    if _CATALOGUE is None or refresh:
        _CATALOGUE = catalogue.discover()
    return _CATALOGUE


def _workdir(workdir: str | None) -> Path:
    d = Path(workdir or os.environ.get("DESCRIPTRON_MCP_WORKDIR") or os.getcwd()).expanduser()
    d = data_tools.check_path(str(d))
    if not d.is_dir():
        raise ValueError(f"working directory does not exist: {d}")
    return d


# ------------------------------------------------------------------ programs --
@mcp.tool()
def list_programs(filter: str = "", package: str = "", include_libraries: bool = False) -> dict:
    """List the Descriptron programs that can be run.

    filter: substring to match in the name or summary (case-insensitive).
    package: "core" (CPU analysis), "vision" (GPU: detectors, SAM2-PAL, DINOLand,
        Florence-2 measurements) or "extra" (directories from DESCRIPTRON_MCP_TOOL_DIRS).
    include_libraries: also list helper modules that are imported, not run.
    """
    f = filter.lower()
    rows = [p.as_dict() for p in _cat().values()
            if (include_libraries or p.runnable)
            and (not package or p.package == package)
            and (not f or f in p.name.lower() or f in p.summary.lower())]
    rows.sort(key=lambda r: (r["package"], r["name"]))
    return {"count": len(rows), "programs": rows}


@mcp.tool()
def program_help(name: str) -> dict:
    """Show a program's command-line options (its --help) and the top of its documentation.

    Always read this before run_program or start_job: every option is passed as a
    command-line argument, exactly as in a shell.
    """
    prog = catalogue.resolve(_cat(), name)
    doc = ""
    if prog.path.suffix == ".py":
        import ast
        try:
            doc = ast.get_docstring(ast.parse(prog.path.read_text(encoding="utf-8", errors="replace"))) or ""
        except SyntaxError:
            pass
    res = jobs.run_now(prog, ["--help"], Path(tempfile.gettempdir()), timeout_s=180, tail_lines=400)
    return {"program": prog.name, "package": prog.package, "path": str(prog.path),
            "interpreter": catalogue.python_for(prog),
            "documentation": doc[:6000], "help": res["stdout_tail"] or res["stderr_tail"],
            "help_exit_code": res["exit_code"]}


@mcp.tool()
def run_program(name: str, args: list[str] | None = None, workdir: str | None = None,
                timeout_s: int = 900, tail_lines: int = 60) -> dict:
    """Run a Descriptron program to completion and return the end of its output.

    name: program name from list_programs (a unique prefix is enough).
    args: command-line arguments as a list, e.g. ["--matrix_dir", "/data/m", "--output_dir", "/data/key"].
        No shell is involved, so do not quote or escape.
    workdir: directory the program runs in (default: DESCRIPTRON_MCP_WORKDIR or the server's cwd).
    timeout_s: stop after this many seconds. Anything that may take longer than a few
        minutes should be started with start_job instead.
    Returns the exit code, the last lines of stdout/stderr, the full log path and the
    files written to any --output*/--out* directory.
    """
    prog = catalogue.resolve(_cat(), name)
    return jobs.run_now(prog, list(args or []), _workdir(workdir), timeout_s, tail_lines)


@mcp.tool()
def start_job(name: str, args: list[str] | None = None, workdir: str | None = None) -> dict:
    """Start a long program in the background and return a job id at once.

    Use for run_full_pipeline_v2, sam2_pal_batch_v21, dinov3_landmark_transfer_v51,
    detector training and anything else that runs for more than a few minutes.
    The job keeps running if the client disconnects. Poll with job_status.
    """
    prog = catalogue.resolve(_cat(), name)
    return jobs.start(prog, list(args or []), _workdir(workdir))


@mcp.tool()
def job_status(job_id: str, tail_lines: int = 20) -> dict:
    """State of a background job (running / finished / failed / cancelled / lost),
    the end of its log, and, once it has ended, the files it wrote."""
    return jobs.status(job_id, tail_lines)


@mcp.tool()
def job_log(job_id: str, tail_lines: int = 200) -> str:
    """The last lines of a background job's combined output."""
    return jobs.log(job_id, tail_lines)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Stop a running background job (and every process it started)."""
    return jobs.cancel(job_id)


@mcp.tool()
def list_jobs(limit: int = 30) -> list[dict]:
    """Background jobs started by this server, newest first."""
    return jobs.all_jobs(limit)


# --------------------------------------------------------------- read data --
@mcp.tool()
def list_files(directory: str, pattern: str = "*", recursive: bool = False, max_items: int = 200) -> dict:
    """List files in a directory, optionally recursively, filtered by a glob pattern like '*.csv'."""
    return data_tools.list_files(directory, pattern, recursive, max_items)


@mcp.tool()
def read_text(path: str, max_chars: int = 20000, start_char: int = 0) -> dict:
    """Read a text file (reports, logs, data sheets, JSON, YAML taxon profiles), in chunks."""
    return data_tools.read_text(path, max_chars, start_char)


@mcp.tool()
def read_table(path: str, max_rows: int = 50, columns: list[str] | None = None,
               where: dict[str, str] | None = None) -> dict:
    """Preview a CSV/TSV results table.

    columns: keep only these columns. where: keep rows whose column equals a value
    exactly, e.g. {"species": "acok"}. Reports the total and matching row counts.
    """
    return data_tools.read_table(path, max_rows, columns, where)


@mcp.tool()
def coco_summary(path: str) -> dict:
    """Summarise a COCO annotation file: images, categories, annotations per category,
    polygon/RLE/keypoint counts, unannotated images and images with zero width/height
    (which silently rescale every coordinate downstream)."""
    return data_tools.coco_summary(path)


@mcp.tool()
def view_image(path: str, max_side: int = 1568):
    """Look at an image: a specimen photograph, micro-CT slice, mask, plate or figure.

    Any format Pillow reads (TIFF incl. 16-bit, PNG, JPEG). Downscaled so the longest
    side is at most max_side pixels; the original size is reported so coordinates
    can be scaled back.
    """
    data, info = data_tools.view_image(path, max_side)
    return [Image(data=data, format="jpeg"), json.dumps(info)]


# ---------------------------------------------------- write + audit (BioRAG) --
@mcp.tool()
def species_evidence(matrix_dir: str, species: str, tier: str = "", structure: str = "",
                     max_rows: int = 500) -> dict:
    """The measured evidence for one species: per feature n, min, max, mean and unit,
    with the range across all species beside it.

    matrix_dir: the Tier-1 matrix directory (output of biorag_key_feature_filter_v2,
        containing species_feature_summary.csv).
    species: the species code used in the matrix.
    tier: "key" or "description" to restrict; structure: one category, e.g. "paramere".
    Write treatments from these numbers only.
    """
    return data_tools.species_evidence(matrix_dir, species, tier or None, structure or None, max_rows)


@mcp.tool()
def audit_treatment(species: str, treatment: dict, matrix_dir: str, taxon_profile: str,
                    data_sheet: str = "", key_tree: str = "", localities: str = "") -> dict:
    """Independently audit a species treatment against the data (biorag_confabulation_checker_v2).

    The checker recomputes every species' n/min/max/mean from the per-specimen matrix,
    attributes each number to ONE feature from the words around it, re-derives every
    comparison, and checks Remarks against their sources. A correct number printed
    next to the wrong structure is an error.

    species: species code. treatment: {"diagnosis": str, "description": [{"section": str,
        "text": str}, ...], "sexual_dimorphism": str, "remarks": str,
        "citations": [{"value": "0.240–0.242", "id": "distal_aedeagus.length_mm"}, ...]}.
    matrix_dir: Tier-1 matrix directory. taxon_profile: the taxon profile YAML.
    data_sheet: optional <code>_data_sheet.txt (needed to verify Remarks).
    key_tree, localities: optional, as for the pipeline.
    Returns the summary and every claim that is not "ok"; fix only those sentences.
    """
    for k in ("diagnosis", "description"):
        if k not in treatment:
            raise ValueError(f"treatment needs a '{k}' field; see this tool's description for the shape")
    root = jobs.state_dir() / "audits" / f"{datetime.now():%Y%m%d_%H%M%S}_{species}"
    sp_dir = root / "descriptions" / species
    sp_dir.mkdir(parents=True)
    (sp_dir / f"{species}_treatment.json").write_text(json.dumps(treatment, indent=1, ensure_ascii=False),
                                                     encoding="utf-8")
    if data_sheet:
        (sp_dir / f"{species}_data_sheet.txt").write_text(
            data_tools.check_path(data_sheet).read_text(encoding="utf-8"), encoding="utf-8")
    args = ["--descriptions_dir", str(root / "descriptions"), "--matrix_dir", matrix_dir,
            "--taxon_profile", taxon_profile, "--output_dir", str(root / "report"), "--species", species]
    if key_tree:
        args += ["--key_tree", key_tree]
    if localities:
        args += ["--localities", localities]
    prog = catalogue.resolve(_cat(), "biorag_confabulation_checker_v2")
    run = jobs.run_now(prog, args, root, timeout_s=900, tail_lines=30)
    if run["exit_code"] != 0:
        return {"ok": False, "error": "the checker did not complete", **run}
    summary = json.loads((root / "report" / "confabulation_summary.json").read_text(encoding="utf-8"))
    per = root / "report" / "per_species" / f"{species}.json"
    records = json.loads(per.read_text(encoding="utf-8")).get("records", []) if per.exists() else []
    flagged = [{k: r.get(k) for k in ("section", "kind", "status", "type", "feature", "lo", "hi",
                                      "unit", "note", "context")}
               for r in records if r.get("status") != "ok"]
    keep = ("claims_by_kind", "numbers_checked", "errors", "needs_review", "unverifiable", "warnings",
            "confabulation_rate", "confabulation_type_errors", "confabulation_type_warnings",
            "orphan_citations")
    return {"ok": True, "species": species, "claims_checked": len(records),
            "flagged_count": len(flagged), "flagged": flagged,
            "summary": {k: summary.get(k) for k in keep}, "report_dir": str(root / "report")}


# ---------------------------------------------------------- resource/prompt --
WORKFLOW = """\
# Descriptron workflow (v2, BioRAG)

1. Annotate structures (Descriptron GUI, or predict with descriptron-vision:
   tv_train_v1 / tv_predict_v1, sam2_pal_batch_v21, dinov3_landmark_transfer_v51).
   Check the COCO file with coco_summary; zero width/height images must be fixed.
2. Run the data pipeline: run_full_pipeline_v2 (start_job; needs --coco_json,
   --image_dir, --group_labels, --output_base, --taxon_profile). The description
   steps need a model: --llm_backend claude-code (Claude subscription) or api (key).
   Numbers only: --llm_backend none --skip_steps biorag. Literature for the
   descriptions: --pdf_dir <folder of PDFs>.
3. The Tier-1 matrix (compiled_key_tier/) holds species_feature_summary.csv,
   specimen_matrix_long.csv and feature_dictionary.tsv.
4. Key: biorag_key_builder_v1 --matrix_dir <matrix> --output_dir <key>. The tree is
   computed from the data; a model may only reword couplets.
5. Treatments: from species_evidence (never from memory), then audit_treatment.
6. Exports: build_monograph_exports_v1 (TaxPub, DwC-A, SDD, JSON-LD),
   build_species_treatment_docx_v2.

## Evidence tiers
- key: mm lengths/widths, length/width, standard ratios, landmark-span ratios, CIE L*
- description: adds areas, perimeters, a*, b*, C*, h
- everything statistical (GPA, clusters, texture, novelty): Remarks only

## House rules
- Every number sits next to the structure it measures; ranges are min-max of n specimens.
- A comparison ("greater than in all other species") must be true of the specimen values.
- Never calibrate a threshold on the species it will be judged on.
- The audit is independent of the writer. Fix only what it flags; do not rewrite the rest.
"""


@mcp.resource("descriptron://workflow", name="workflow", mime_type="text/markdown",
              description="The Descriptron v2 workflow, evidence tiers and house rules")
def workflow() -> str:
    return WORKFLOW


@mcp.prompt(name="write_audited_treatment",
            description="Write one species treatment from the matrix and audit it until it is clean")
def write_audited_treatment(species: str, matrix_dir: str, taxon_profile: str) -> str:
    return (f"Write the treatment for species '{species}'.\n"
            f"1. Call species_evidence(matrix_dir='{matrix_dir}', species='{species}').\n"
            f"2. Read the taxon profile {taxon_profile} with read_text for section names and terms.\n"
            "3. Write diagnosis, description (by section), sexual_dimorphism and remarks using ONLY those "
            "numbers, citing each as {value, id=feature_id}.\n"
            f"4. Call audit_treatment(species='{species}', treatment=..., matrix_dir='{matrix_dir}', "
            f"taxon_profile='{taxon_profile}').\n"
            "5. Fix only the flagged sentences and audit again, until flagged_count is 0. "
            "Report what was flagged and how it was fixed.")


# ---------------------------------------------------------------------- main --
def _check() -> int:
    cat = _cat()
    by_pkg: dict[str, int] = {}
    for p in cat.values():
        if p.runnable:
            by_pkg[p.package] = by_pkg.get(p.package, 0) + 1
    print(f"descriptron-mcp {__version__}  (server python: {sys.executable})")
    for pkg in ("core", "vision", "extra"):
        n = by_pkg.get(pkg, 0)
        sample = next((p for p in cat.values() if p.package == pkg), None)
        interp = catalogue.python_for(sample) if sample else "-"
        print(f"  {pkg:<6} {n:>3} programs   runs with: {interp}")
    if not cat:
        print("  no programs found: install descriptron-core (and descriptron-vision), or set "
              "DESCRIPTRON_MCP_TOOL_DIRS")
    print(f"  data     : {catalogue.core_data_dir()}")
    print(f"  state    : {jobs.state_dir()}")
    return 0 if cat else 1


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="descriptron-mcp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print what was found and exit")
    ap.add_argument("--version", action="version", version=f"descriptron-mcp {__version__}")
    a = ap.parse_args(argv)
    if a.check:
        sys.exit(_check())
    mcp.run()                                    # stdio


if __name__ == "__main__":
    main()
