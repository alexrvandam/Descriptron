# Descriptron v2 with Docker: a recipe

Descriptron turns images of specimens into measured structures, a character
matrix, a dichotomous key, species delimitation and species descriptions. The
Docker image contains everything: all programs, all Python environments, SAM2
and Detectron2. You install nothing except Docker.

Tested on 23 September 2026 with 8 *Diaphorina* head images: the full run
finished in about 6 minutes on a laptop without a GPU.

Software DOI: https://doi.org/10.5281/zenodo.22918056
Code and documentation: https://github.com/alexrvandam/Descriptron


## 1. What you need

- **Linux**, or **Windows 10/11** with Docker Desktop (WSL2 backend).
- **About 40 GB free disk space.** The image is 32 GB; model weights add about
  0.5 GB on first use.
- **An NVIDIA GPU is optional.** It is needed only for training detectors and for
  SAM2-PAL mask propagation. Measurements, the matrix, the key and the
  descriptions run without one.
- **Mac:** the image is built for Intel/AMD processors and has no GPU access on
  macOS. Mac users should install with pip instead (see the GitHub README).
- **For species descriptions:** an Anthropic API key. Descriptions are written by
  a vision-language model; without one you still get everything computed from
  the data (matrix, key, delimitation, per-species evidence sheets).


## 2. Install Docker and download Descriptron (once)

Install Docker: https://docs.docker.com/get-docker/ (Windows and Mac: Docker
Desktop; Linux: Docker Engine).

Then, in a terminal:

    docker pull ghcr.io/alexrvandam/descriptron:2.0.2

This downloads 32 GB, so it takes a while.


## 3. Prepare a project folder

Put everything for one analysis in one folder, for example `my_project/`:

    my_project/
      images/                 the specimen images (.tif, .jpg, .png)
      annotations.json        COCO annotations of the structures
                              (made with the Descriptron GUI or a detector)
      group_labels.csv        which species each image belongs to
      taxon_profile.yaml      optional: structure names, ratios, species names
      literature/             optional: PDFs of revisions or descriptions

`group_labels.csv` has two columns, the image file name and the species:

    filename,group_label
    specimen01_head.tif,species_a
    specimen02_head.tif,species_a
    specimen03_head.tif,species_b

If you leave out `taxon_profile.yaml`, a draft is written for you
(`output/taxon_profile_DRAFT.yaml`). Correct the names in it and run again with
`--taxon_profile /data/output/taxon_profile_DRAFT.yaml`.


## 4. Run the whole pipeline

Open a terminal **in your project folder**.

**Linux:**

    export ANTHROPIC_API_KEY=sk-ant-...        # your key, for the descriptions

    docker run --rm --user "$(id -u):$(id -g)" \
      -e ANTHROPIC_API_KEY \
      -v "$PWD:/data" -v descriptron-weights:/weights \
      ghcr.io/alexrvandam/descriptron:2.0.2 pipeline \
        --coco_json /data/annotations.json \
        --image_dir /data/images \
        --group_labels /data/group_labels.csv \
        --taxon_profile /data/taxon_profile.yaml \
        --output_base /data/output \
        --llm_backend api

**Windows (PowerShell):**

    $env:ANTHROPIC_API_KEY = "sk-ant-..."

    docker run --rm -e ANTHROPIC_API_KEY `
      -v "${PWD}:/data" -v descriptron-weights:/weights `
      ghcr.io/alexrvandam/descriptron:2.0.2 pipeline `
        --coco_json /data/annotations.json `
        --image_dir /data/images `
        --group_labels /data/group_labels.csv `
        --taxon_profile /data/taxon_profile.yaml `
        --output_base /data/output `
        --llm_backend api

What the parts mean:

- `-v "$PWD:/data"` makes your project folder visible inside the container as
  `/data`, so every path in the command starts with `/data/`.
- `-v descriptron-weights:/weights` keeps downloaded model weights between runs,
  so they download only once.
- `--user "$(id -u):$(id -g)"` (Linux) makes the output files yours, not root's.
- Add `--gpus all` after `docker run` if you have an NVIDIA GPU with the NVIDIA
  Container Toolkit installed. Without that toolkit, Docker refuses to start with it.

**Variations:**

- **No API key / data only:** use `--llm_backend none` and leave out
  `-e ANTHROPIC_API_KEY`. You get the matrix, the key, delimitation and the
  evidence sheets, but no descriptions.
- **Add literature** (terminology from published treatments):
  add `--pdf_dir /data/literature`. Scanned PDFs without a text layer contribute
  nothing; run OCR on them first (e.g. `ocrmypdf scan.pdf text.pdf`).
- **Stopped part-way?** Run the same command again: finished steps are skipped.


## 5. Where the results are

Everything is written to `my_project/output/`:

- `compiled_key_tier/`: the character matrix (per specimen and per species)
- `key_v2/`: the dichotomous key (text, JSON, SDD) and its validation
- `descriptions_v2/<species>/`: the evidence sheet per species and, with a
  model, the treatment
- `descriptions_v2/confabulation_report_v2/`: the independent audit of every
  number in the treatments
- `treatments/` (with a model): the formatted treatments (.docx) and machine-readable exports
  (TaxPub XML, Darwin Core Archive, SDD, JSON-LD)
- `logs/`: one log per step, for when something goes wrong


## 6. Other commands

    docker run --rm ghcr.io/alexrvandam/descriptron:2.0.2 help
    docker run --rm ghcr.io/alexrvandam/descriptron:2.0.2 pipeline --help

Training detectors, SAM2-PAL, DINOLand and the annotation GUI are described in
the GitHub README.


## 7. Please cite

If you use Descriptron in published work, cite the software and the first paper:

- Van Dam, A. R. (2026). Descriptron: morphology-driven species descriptions,
  keys and delimitation for dark taxa (Version 2.0.1). Zenodo.
  https://doi.org/10.5281/zenodo.22918056
- Van Dam, A. R. & Štarhová Serbina, L. (2026). Descriptron: Artificial
  intelligence for automating taxonomic species descriptions with a
  user-friendly software package. Systematic Entomology, 51(1), e70005.
  https://doi.org/10.1111/syen.70005

Questions and problems: https://github.com/alexrvandam/Descriptron/issues
