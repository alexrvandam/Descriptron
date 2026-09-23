# descriptron-mcp

**The Descriptron programs as tools for Claude and any other MCP client.**

[MCP](https://modelcontextprotocol.io) (Model Context Protocol) is an open
standard that lets an AI assistant call software directly. With this server
installed, you can ask Claude Desktop or Claude Code things like *"summarise
this COCO file, run the pipeline on it, and build the key"*, or *"write the
treatment for acok from the matrix and audit it"*, and it runs the same
Descriptron programs you would run from a shell.

The server runs **on your own computer** and reads **your own files**. Nothing
is uploaded anywhere except what the assistant itself reads through the tools.

## Install

Python ≥ 3.10 is needed for the server itself. The programs can run in a
different environment. See *Configuration* below.

**Now (from GitHub).** The Descriptron packages are attached to the
[v2.0.1 release](https://github.com/alexrvandam/Descriptron/releases/tag/v2.0.1).
Install the analysis package first, then the server:

```bash
python -m venv descriptron-mcp-env
source descriptron-mcp-env/bin/activate        # Windows: descriptron-mcp-env\Scripts\activate

pip install https://github.com/alexrvandam/Descriptron/releases/download/v2.0.1/descriptron_core-2.0.1-py3-none-any.whl
pip install "git+https://github.com/alexrvandam/Descriptron#subdirectory=packages/descriptron-mcp"

descriptron-mcp --check                        # shows what it found
```

For the GPU programs (detectors, SAM2-PAL, DINOLand) also install
`descriptron_vision-2.0.1-py3-none-any.whl` from the same release page.

**Later (from PyPI):**

```bash
pip install descriptron-mcp            # analysis programs (CPU)
pip install "descriptron-mcp[vision]"  # + detectors, SAM2-PAL, DINOLand (GPU)
```

**With Docker instead** (no Python setup; includes the GPU programs):

```bash
claude mcp add descriptron -- docker run -i --rm --gpus all \
  --user "$(id -u):$(id -g)" -v "$HOME:$HOME" \
  ghcr.io/alexrvandam/descriptron:2.0.2 mcp
```

Without an NVIDIA GPU (e.g. on a Mac), leave out `--gpus all`: Docker refuses to start
with it, and everything except the GPU programs works the same.
`-v "$HOME:$HOME"` makes your files appear inside the container at the same
paths Claude uses; add another `-v /path:/path` for data elsewhere (e.g. an
external drive). `--user` makes the files it writes yours rather than root's.
Background jobs stop when the Claude session ends, because the container does.

## Connect it

**Claude Code**

```bash
claude mcp add descriptron -- descriptron-mcp
```

**Claude Desktop**: in *Settings → Developer → Edit config*
(`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "descriptron": {
      "command": "descriptron-mcp",
      "env": { "DESCRIPTRON_MCP_WORKDIR": "/path/to/your/project" }
    }
  }
}
```

If `descriptron-mcp` is not on the PATH Claude Desktop sees, give the full
path, e.g. `/home/me/miniforge3/envs/descriptron/bin/descriptron-mcp`.

## Tools

| tool | what it does |
|---|---|
| `list_programs` | the runnable programs (core: CPU analysis; vision: GPU) |
| `program_help` | a program's `--help` and documentation |
| `run_program` | run a program to completion; exit code, output tail, files written |
| `start_job` / `job_status` / `job_log` / `cancel_job` / `list_jobs` | long runs (pipelines, SAM2-PAL, training) in the background |
| `coco_summary` | images, categories, annotation counts, zero-size images |
| `read_table` / `read_text` / `list_files` | look at results |
| `view_image` | show a specimen photograph, micro-CT slice, mask or figure to the model |
| `species_evidence` | one species' measured n/min/max/mean per feature, with the all-species range |
| `audit_treatment` | independent audit of a treatment against the per-specimen data |

Resource `descriptron://workflow` gives the recipe; prompt
`write_audited_treatment` runs the write → audit → fix loop.

**Writing and auditing are separate.** When the assistant writes a
treatment, it gets the numbers from `species_evidence` and then submits the
text to `audit_treatment`. That audit is `biorag_confabulation_checker_v2`,
which recomputes every value from the specimen matrix and checks each number
against the structure it is printed next to. The writer never marks its own
work.

## Configuration (environment variables)

| variable | meaning | default |
|---|---|---|
| `DESCRIPTRON_CORE_PYTHON` | interpreter that runs the analysis programs | the server's |
| `DESCRIPTRON_VISION_PYTHON` | interpreter that runs the GPU programs | the server's |
| `DESCRIPTRON_PYTHON` | fallback for both | |
| `DESCRIPTRON_MCP_WORKDIR` | directory programs run in | server's cwd |
| `DESCRIPTRON_MCP_TOOL_DIRS` | extra program directories (e.g. a development checkout); these win on a name clash | |
| `DESCRIPTRON_MCP_ALLOWED_ROOTS` | if set, file tools refuse paths outside these directories | unrestricted |
| `DESCRIPTRON_MCP_STATE` | job logs, run logs, audits | `~/.cache/descriptron-mcp` |

Example: run the server from a small Python 3.11 environment, but run the
analysis in an existing conda environment and the GPU programs in another:

```json
"env": {
  "DESCRIPTRON_CORE_PYTHON": "/opt/conda/envs/measure_env/bin/python",
  "DESCRIPTRON_VISION_PYTHON": "/opt/conda/envs/samm/bin/python"
}
```

## What is not exposed

The annotation GUI. Annotation is interactive; the server works from the COCO
files the GUI (or a detector) writes.

## Licence

Apache-2.0. Part of [Descriptron](https://github.com/alexrvandam/Descriptron).
