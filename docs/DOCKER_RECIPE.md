# Descriptron v2 with Docker: a recipe

Descriptron turns images of specimens into measured structures, a character
matrix, a dichotomous key, species delimitation and species descriptions. The
Docker image contains everything: all programs, all Python environments, SAM2
and Detectron2. You install nothing except Docker.

Tested on 23 September 2026 with 8 *Diaphorina* head images: the full run
finished in about 6 minutes on a laptop without a GPU. Version 2.0.3 (24
September 2026): SAM2-PAL and DINOLand re-tested inside the image, with and
without a GPU.

Software DOI (always the newest version): https://doi.org/10.5281/zenodo.17077224
Code and documentation: https://github.com/alexrvandam/Descriptron


## 1. What you need

- **Linux**, or **Windows 10/11** with Docker Desktop (WSL2 backend).
- **About 40 GB free disk space.** The image is 32 GB; model weights add about
  0.5 GB on first use.
- **An NVIDIA GPU is optional.** It is needed only for training detectors and for
  SAM2-PAL mask propagation. DINOLand landmark transfer, measurements, the
  matrix, the key and the descriptions run without one.
- **Mac:** the image is built for Intel/AMD processors and has no GPU access on
  macOS. Mac users should install with pip instead (see the GitHub README).
- **For species descriptions:** an Anthropic API key. Descriptions are written by
  a vision-language model; without one you still get everything computed from
  the data (matrix, key, delimitation, per-species evidence sheets).


## 2. Install Docker and download Descriptron (once)

Install Docker: https://docs.docker.com/get-docker/ (Windows and Mac: Docker
Desktop; Linux: Docker Engine).

Then, in a terminal:

    docker pull ghcr.io/alexrvandam/descriptron:2.1.0

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
      ghcr.io/alexrvandam/descriptron:2.1.0 pipeline \
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
      ghcr.io/alexrvandam/descriptron:2.1.0 pipeline `
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

**Keys, saved once (instead of `export` in every terminal).** Descriptron looks for
a key first in the environment, then in a small private file, and otherwise asks
the first time it needs it. To save them in that file:

    docker run -it --rm -v "$HOME/.config/descriptron:/config" \
      ghcr.io/alexrvandam/descriptron:2.1.0 keys --set ANTHROPIC_API_KEY
    docker run -it --rm -v "$HOME/.config/descriptron:/config" \
      ghcr.io/alexrvandam/descriptron:2.1.0 keys --set HF_TOKEN

Then add `-v "$HOME/.config/descriptron:/config"` to your runs and leave out the
`export` and `-e` parts. `keys` alone shows which keys are set (never their
values); `keys --remove NAME` deletes one. With `docker run -it`, a run that needs a
missing key asks for it and saves it the same way. Or write the file yourself:
`~/.config/descriptron/credentials`, one `NAME=value` per line.


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


## 6. Masks and landmarks: SAM2-PAL and DINOLand

These copy what you drew on a few reference images to the rest of a
collection: SAM2-PAL propagates masks, DINOLand transfers numbered landmarks.
How many references to draw, how to image, and how to check the results:
[SAM2-PAL & DINOLand annotation SOP](SAM2PAL_DINOLand_Annotation_SOP.md).

**Model weights (once).** SAM2 checkpoints are not in the image; download the
large one into your project folder:

    curl -L -o sam2_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

DINOLand uses Meta's DINOv3, which is gated: accept the licence at
https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m, create a read
token in your Hugging Face settings, and pass it with `-e HF_TOKEN`. The weights
then download once into the `descriptron-weights` volume.

**SAM2-PAL (Linux, NVIDIA GPU):**

    docker run --rm --gpus all --user "$(id -u):$(id -g)" \
      -v "$PWD:/data" -v descriptron-weights:/weights \
      ghcr.io/alexrvandam/descriptron:2.1.0 sam2-pal \
        --template_image /data/refs/template.png --template_json /data/refs/annotations.json \
        --training_json /data/refs/annotations.json --training_images_dir /data/refs \
        --image_dir /data/images --output_dir /data/out_sam2pal \
        --sam2_checkpoint /data/sam2_hiera_large.pt --sam2_config sam2_hiera_l.yaml \
        --pal_finetuning --num_epochs 40 --learning_rate 1e-5 --max_images_per_epoch 60 \
        --num_points 30 --chunk_size 1 --interleave_template --cycle_consistency --save_vis

**DINOLand (CPU is fine):**

    export HF_TOKEN=hf_...
    docker run --rm --user "$(id -u):$(id -g)" -e HF_TOKEN \
      -v "$PWD:/data" -v descriptron-weights:/weights \
      ghcr.io/alexrvandam/descriptron:2.1.0 dinoland \
        --imgA /data/refs/ref1.tif \
        --landmarks /data/refs/ref1.json,/data/refs/ref2.json,/data/refs/ref3.json \
        --ref_dir /data/refs --batch_glob "/data/images/*.tif" --batch_n 999 \
        --align feature --outdir /data/out_dinoland

**If specimens are not all photographed the same way** (both options are off by
default; best practice is to image every specimen in the references'
orientation):

- turned or upside-down specimens: add `--orientation_search rot4` (SAM2-PAL and
  DINOLand);
- mirror images, e.g. left and right wings: add `--mirror_refs` (DINOLand);
- not sure: for DINOLand use both.

Masks and landmarks come out as COCO JSON that the Descriptron GUI opens for
checking and correcting, and that the pipeline above measures.


## 7. Other commands

    docker run --rm ghcr.io/alexrvandam/descriptron:2.1.0 help
    docker run --rm ghcr.io/alexrvandam/descriptron:2.1.0 pipeline --help

Training detectors and the annotation GUI are described in the GitHub README.
**Hosting Descriptron for a group** on one institutional machine with a shared
GPU: [Shared VM hosting recipe](SHARED_VM_RECIPE.md).


## 8. Please cite

If you use Descriptron in published work, cite the software and the first paper:

- Van Dam, A. R. (2026). Descriptron: morphology-driven species descriptions,
  keys and delimitation for dark taxa. Zenodo.
  https://doi.org/10.5281/zenodo.17077224 (or the DOI of the version you used)
- Van Dam, A. R. & Štarhová Serbina, L. (2026). Descriptron: Artificial
  intelligence for automating taxonomic species descriptions with a
  user-friendly software package. Systematic Entomology, 51(1), e70005.
  https://doi.org/10.1111/syen.70005

Questions and problems: https://github.com/alexrvandam/Descriptron/issues
