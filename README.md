# Descriptron v2 with BioRAG

**Morphology-driven species descriptions, keys and delimitation for dark taxa.**

Descriptron takes photographs or micro-CT slices of specimens and produces the
things a taxonomist actually needs: measured structures, a character matrix, a
dichotomous key, formatted species treatments, and a record of how every number
was obtained. Annotation is model-assisted; the analysis is deterministic; and
every claim in the output is checked against the data it came from.

> **v2 is a rewrite of everything downstream of annotation.** v1 produced
> descriptions by handing measurements to GPT-4o. v2 replaces that with an
> evidence-tiered pipeline (**BioRAG**) in which the key, the matrix, the
> delimitation and the audits are computed, and a language model is used only to
> write prose from numbers it is not allowed to invent.

---

## Contents

- [What v2 adds](#what-v2-adds)
- [Installing](#installing) — pip · Docker · conda
- [The workflow](#the-workflow)
- [What the programs produce](#what-the-programs-produce)
- [Reproducing an analysis](#reproducing-an-analysis)
- [Provenance and auditing](#provenance-and-auditing)
- [Licence](#licence)
- [Citation](#citation)

---

## What v2 adds

| | v1 | v2 |
|---|---|---|
| descriptions | GPT-4o from a feature table | **BioRAG**: evidence tiers, numbers checked against the matrix, every sentence attributable |
| key | — | computed from the matrix, leave-one-specimen-out validated, jackknife couplet support |
| delimitation | — | matrix / key / graph instruments, calibrated on your own reference set |
| mask propagation | Detectron2 only | **SAM2-PAL** palindrome propagation, **DINOLand** DINOv3 landmark transfer, torchvision detectors |
| checking | — | independent confabulation audit, subjective-word checks, ontology annotation |
| install | eight conda environments | **`pip install descriptron`**, or one Docker image |

Nothing in the analysis half needs a GPU, and the key, matrix, audits and
delimitation need no model at all.

---

## Installing

### 1. pip — the normal route

```bash
pip install descriptron          # everything
descriptron-gui                  # the annotation GUI
descriptron --list               # the 62 analysis programs
```

Or take only what you need:

| package | what it gives you | needs |
|---|---|---|
| `descriptron-core` | the whole analysis half — measurements, matrix, key, treatments, audits | **nothing but pip.** No torch, no CUDA, no compiler |
| `descriptron-vision` | mask and landmark prediction: torchvision detectors, SAM2-PAL, DINOLand | torch (ordinary wheels; Apple GPU via MPS on macOS) |
| `descriptron-gui` | the annotation GUI | core + vision + tkinter |

**If you only want to reproduce a published analysis from deposited COCO files,
`pip install descriptron-core` is enough**, and it installs in about a minute on
Linux, macOS and Windows.

For an NVIDIA GPU, install torch from PyTorch's index first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install descriptron
```

Writing treatments is the only step that calls a language model:
`pip install "descriptron-core[llm]"`.

### 2. Docker — everything, including the parts pip cannot carry

docker pull ghcr.io/alexrvandam/descriptron:2.0.0

Two dependencies are not on PyPI and can never be declared by a published
package: **SAM2** and **Detectron2**. The image carries both, already built.

```bash
docker pull ghcr.io/alexrvandam/descriptron:latest

docker run --rm --gpus all -v "$PWD:/data" -v descriptron-weights:/weights \
  ghcr.io/alexrvandam/descriptron:latest train-d2 \
    --coco-json /data/annotations.json --img-dir /data/images \
    --output-dir /data/out --dataset-name mytaxon --total-iters 35000

docker run --rm ghcr.io/alexrvandam/descriptron:latest help
```

**GPU support by platform** — read this before planning around Docker:

| platform | GPU inside the container |
|---|---|
| Linux + NVIDIA | yes (`nvidia-container-toolkit`, `--gpus all`) |
| Windows + NVIDIA | yes (Docker Desktop, WSL2 backend) |
| **macOS** | **no — none, ever** |

Docker Desktop on macOS runs a Linux VM with no passthrough to the Apple GPU, and
Apple Silicon has no CUDA. **Mac users should install with pip and use the
torchvision backend**, which reaches the Apple GPU through PyTorch's MPS device.

`podman` works as a drop-in and has no licensing condition, which matters for
larger institutions.

### 3. conda — for development, or to match the published environment exactly

```bash
git clone https://github.com/alexrvandam/Descriptron.git && cd Descriptron
conda env create -f environments/measure_env_environment.yml     # analysis
conda env create -f environments/samm_environment.yml            # SAM2, SAM2-PAL, DINOLand
conda env create -f environments/detectron2_env_environment.yml  # Mask R-CNN
```

Detectron2 must then be built from source; see `docker/Dockerfile` for the exact
sequence, including the two flags it needs (`--no-build-isolation`, and a
non-editable install).

---

## The workflow

```
  images ──► ANNOTATE ──► PROPAGATE ──► MEASURE ──► SCREEN ──► MATRIX
                GUI        SAM2-PAL      mm, GPA,    outliers,   evidence
             SAM2-assisted  DINOLand     colour,     flips,      tiers
                                         texture     conflicts      │
                                                                    ▼
   treatments ◄── AUDIT ◄── KEY ◄── DELIMIT ◄──────────────────── characters
    .docx, XML,  numeric   couplets  matrix / key / graph,
    DwC-A, SDD   + words   + support  calibrated on your own set
```

Each stage is a set of command-line programs; the GUI is a front end that runs
them for you. Every step that calls a model takes the same flags
(`--llm-backend claude-code | api | none`, `--api-base-url`, `--api-key-env`), so
a subscription, an API key, another provider, or **no model at all** are
interchangeable.

---

## What the programs produce

67 programs, 51,107 lines. `docs/FigS1_script_inventory.png` draws all of them,
and `docs/FigS1_script_inventory.tsv` is the same information as a table: for
every program, its stage, line count, whether it calls a model, which pip
distribution installs it, **and what it writes**.

| stage | programs | lines | typical outputs |
|---|---|---|---|
| Annotation (the GUI) | 3 | 14,832 | COCO JSON, masks, visualisations |
| Images to measurements | 9 | 9,712 | measurement tables (mm), GPA coordinates, colour and texture tables, diagnostic figures |
| Screening, policy and matrix | 8 | 1,619 | exclusion lists, the character matrix, evidence-tier policy |
| Key and treatments | 5 | 5,928 | the key (text + JSON), species treatments |
| Checking the text | 5 | 2,705 | confabulation reports, ontology coverage, gate decisions |
| Descriptive categorical characters | 5 | 1,093 | character state tables, repeatability figures |
| Naming specimens, recognising new species | 16 | 5,980 | identification tables, novelty scores, calibration sheets, ROC figures |
| Model-proposed binary characters | 9 | 5,390 | proposed characters, congruence tables, heat-map atlases |
| Names, types and outputs | 7 | 4,017 | treatment .docx, TaxPub XML, DwC-A, SDD, JSON-LD, collaborator workbooks |

The output column in the figure is **read from the source**, not written by hand:
a program is listed as writing a figure when its code calls `savefig`, a table
when it calls `to_csv`, and so on. The figure therefore cannot drift from the
code, and any program that writes nothing is a step that only feeds the next one.

---

## Reproducing an analysis

```bash
pip install descriptron-core
descriptron run_full_pipeline_v2 --help
```

With the deposited COCO files and the taxon profile, the pipeline regenerates the
measurements, the matrix, the key, the delimitation and every figure. The steps
that write prose are skipped unless a model backend is given, and none of the
numbers depends on one.

---

## Provenance and auditing

Two mechanisms, both meant for a reader who does not take your word for it.

**Every number is traceable.** `manuscript_numbers_v18.tsv` maps each number in
the manuscript to the report file that produced it, and the manuscript builder
refuses to typeset a number that no report supplies.

**Every output records how it was made.** `biorag_provenance_v1.py` stamps
outputs with the script and its SHA-256, the git commit, the full command line,
**a SHA-256 for every input file**, the interpreter and the time:

```bash
descriptron biorag_provenance_v1 --verify path/to/report.json
```

re-hashes the script and the inputs and tells you whether either has changed
since the output was written.

**And the text is audited independently of the model that wrote it.**
`biorag_confabulation_checker_v2.py` recomputes every statistic from the specimen
matrix, attributes every number in a treatment to one feature, and re-derives
every comparison — a value that is correct but attached to the wrong structure
fails, which is exactly what a generator checking its own output cannot catch.

---
## Use Descriptron from Claude (MCP server)

Descriptron can also be driven by an AI assistant. `descriptron-mcp` is an
[MCP](https://modelcontextprotocol.io) (Model Context Protocol) server: it lets
Claude Code, Claude Desktop, or any other MCP client run the Descriptron
programs for you. For example, you can ask *"summarise this COCO file and check it
for problems"*, *"build the key from this matrix"*, *"run SAM2-PAL on this folder
in the background"*, or *"write the treatment for this species from the matrix
and audit it"*.

The server runs **on your own computer** and works on **your own files**. It
runs the same programs you would run from a shell, so their results are the same.

**Writing and checking are kept apart.** When the assistant writes a species
treatment, it takes every number from the measured data and then submits the
text to the independent audit (`biorag_confabulation_checker_v2`). That audit
recomputes each value from the specimen matrix and flags any number printed
next to the wrong structure. The writer never marks its own work.

### Install

Python 3.10 or newer is needed for the server.

```bash
python -m venv descriptron-mcp-env
source descriptron-mcp-env/bin/activate        # Windows: descriptron-mcp-env\Scripts\activate

pip install https://github.com/alexrvandam/Descriptron/releases/download/v2.0.0/descriptron_core-2.0.0-py3-none-any.whl
pip install "git+https://github.com/alexrvandam/Descriptron#subdirectory=packages/descriptron-mcp"

descriptron-mcp --check                        # lists the programs it found
```

For the GPU programs (torchvision detectors, SAM2-PAL, DINOLand), also install
`descriptron_vision-2.0.0-py3-none-any.whl` from the
[v2.0.0 release page](https://github.com/alexrvandam/Descriptron/releases/tag/v2.0.0).

### Connect it

**Claude Code**

```bash
claude mcp add descriptron -- /full/path/to/descriptron-mcp-env/bin/descriptron-mcp
```

**Claude Desktop**: Settings → Developer → Edit config, then add:

```json
{
  "mcpServers": {
    "descriptron": {
      "command": "/full/path/to/descriptron-mcp-env/bin/descriptron-mcp"
    }
  }
}
```

Restart the client and the Descriptron tools are available.

### What it offers

| tools | what they do |
|---|---|
| `list_programs`, `program_help` | the available programs and their options |
| `run_program` | run a program and report its output and the files it wrote |
| `start_job`, `job_status`, `job_log`, `cancel_job`, `list_jobs` | long runs (pipeline, SAM2-PAL, training) in the background |
| `coco_summary`, `read_table`, `read_text`, `list_files`, `view_image` | look at annotations, results, figures and specimen images |
| `species_evidence`, `audit_treatment` | write a treatment from the data, then audit it independently |

The annotation GUI is not part of the server. Annotation is interactive, and the
server works from the COCO files the GUI (or a detector) writes.

Configuration (for example, running the analysis in an existing conda
environment), tests and details: [packages/descriptron-mcp/README.md](packages/descriptron-mcp/README.md).

---

## Licence

Apache License 2.0. Some bundled or downloaded components carry their own terms:
Detectron2 (Apache-2.0), SAM2 (Apache-2.0), Metric3D (BSD-2-Clause), EasyOCR
(Apache-2.0). **Model weights are licensed separately from code** — DINOv3 and
some Florence-2 checkpoints are gated and carry their own conditions, which you
should read before any commercial use. Descriptron's own licence does not
restrict commercial use; some model weights may.

---

## Citation

Please cite Descriptron and the components you used. See `CITATION.cff` for the
machine-readable entry and the BibTeX below for the dependencies.

Cite the **Zenodo DOI of the release you ran**, not this page: a DOI pins the
exact version your results came from.
