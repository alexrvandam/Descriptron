# Descriptron in a container

One image, every environment, all the scripts. It exists because four of the
tools need Python environments that cannot co-exist — the conflict is about
co-installation, not about sharing a disk, and a container is where they can all
live at once.

## What is inside

| environment | python / torch | what runs there |
|---|---|---|
| `measure_env` | 3.9.20 · torch 2.5.0 | the BioRAG v2 pipeline, measurements, colour, texture, landmark and semilandmark GPA, compile, key, treatments, exports — 64 analysis programs |
| `samm` | 3.12.4 · torch 2.4.0+cu121 | the annotation GUI, SAM2, **SAM2-PAL v21**, **DINOLand v51**, and the torchvision Mask R-CNN / Keypoint R-CNN trainers |
| `detectron2_env` | 3.9.20 · torch 2.4.1+cu121 | Detectron2 Mask R-CNN training and prediction (compiled at build time) |
| `metric3d` | 3.9 · torch 2.0.1+cu117 | depth from defocus — **opt-in**, `--build-arg INCLUDE_METRIC3D=true` |

Also built from source, because neither is on PyPI and so neither can ever be a
declared dependency of a published package: **SAM2** and **Detectron2**.

**Not** included: model weights (they download on first use into the `/weights`
volume, so rebuilding the image does not re-download them), your data, and the
superseded `gpt4` v1 description scripts.

## Size, honestly

About **25 GB** with three environments, **31 GB** with `metric3d` — four torch
stacks, each carrying its own CUDA runtime. That is a real download. If you only
need one job, a single-environment image is far smaller; the Dockerfile is
structured so the stages can be cut down.

## Build

```bash
cd /path/to/Descriptron
docker build -f docker/Dockerfile -t descriptron:latest .
docker build -f docker/Dockerfile --build-arg INCLUDE_METRIC3D=true -t descriptron:depth .
```

## Run

Everything after the subcommand is passed through unchanged, so the flags are
identical to a native install.

```bash
# train the torchvision backend (works on any NVIDIA machine)
docker run --rm --gpus all -v "$PWD:/data" -v descriptron-weights:/weights \
  descriptron:latest train-masks \
    --coco-json /data/annotations.json --img-dir /data/images \
    --output-dir /data/out --total-iters 35000

# train Detectron2 (the more accurate backend)
docker run --rm --gpus all -v "$PWD:/data" -v descriptron-weights:/weights \
  descriptron:latest train-d2 \
    --coco-json /data/annotations.json --img-dir /data/images \
    --output-dir /data/out --total-iters 35000 --dataset-name mytaxon

# SAM2-PAL propagation, DINOLand landmarks, measurements, the whole pipeline
docker run --rm --gpus all -v "$PWD:/data" descriptron:latest sam2-pal --help
docker run --rm --gpus all -v "$PWD:/data" descriptron:latest dinoland --help
docker run --rm -v "$PWD:/data" descriptron:latest measure --help
docker run --rm -v "$PWD:/data" descriptron:latest pipeline --help

docker run --rm descriptron:latest envs      # which environments this image has
docker run --rm -it -v "$PWD:/data" descriptron:latest shell samm
```

`podman` works as a drop-in (`podman run --device nvidia.com/gpu=all …`) and has
no licensing condition, which matters for larger institutions.

## GPU support, by platform — read this before planning around Docker

| platform | GPU inside the container |
|---|---|
| Linux + NVIDIA | **yes** — install `nvidia-container-toolkit`, pass `--gpus all` |
| Windows + NVIDIA | **yes** — Docker Desktop with the WSL2 backend |
| **macOS (Apple Silicon or Intel)** | **no — none, ever** |

Docker Desktop on macOS runs containers in a Linux VM with no passthrough to the
Apple GPU, and Apple Silicon has no CUDA. Training in this image on a Mac is
CPU-only and impractical: a run that takes ten minutes on an RTX 3500 takes
hours. **Mac users should install natively with pip and use the torchvision
backend, which can use the Apple GPU through PyTorch's MPS device.** The
container is still useful on a Mac for short prediction jobs and for
reproducing an analysis exactly.

## The GUI

The annotation GUI is Tkinter and needs an X server, which is straightforward on
Linux and awkward everywhere else:

```bash
xhost +local:docker
docker run --rm --gpus all -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v "$PWD:/data" descriptron:latest gui
```

On macOS this needs XQuartz and on Windows VcXsrv, neither of which is one
click. **The recommended arrangement is the other way round**: install the GUI
natively (`pip install descriptron-gui` — Tkinter ships with Python, and
annotation wants to be responsive anyway) and let the GUI call this container
for the batch steps. Annotation stays native; training and prediction go in the
container.
