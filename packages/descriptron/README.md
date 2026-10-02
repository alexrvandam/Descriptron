# descriptron

Image annotation, phenomics and evidence-tiered species descriptions.

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128   # NVIDIA; Mac/CPU: drop --index-url
SAM2_BUILD_CUDA=0 pip install git+https://github.com/facebookresearch/sam2.git   # SAM2 library (Windows: set SAM2_BUILD_CUDA=0 first)
pip install descriptron          # analysis programs, detectors and the GUI
descriptron-gui                  # first start downloads the SAM2 model file (898 MB) once
descriptron --list               # the 62 analysis programs
```

**SAM2 is not on PyPI**, so it cannot be installed with this package: install it from Meta's GitHub as above, into
the same Python (3.10 or newer; 3.12 recommended). The GUI downloads the SAM2 model file `sam2_hiera_large.pt` once
into `~/.cache/descriptron/sam2/`; if you already have it, copy it there or set `DESCRIPTRON_SAM2_CHECKPOINT`.
Step by step for Windows, macOS and Linux:
https://github.com/alexrvandam/Descriptron/blob/main/docs/INSTALL_PIP.md

This is a convenience package with no code of its own. It installs:

| | |
|---|---|
| [`descriptron-core`](https://pypi.org/project/descriptron-core/) | the analysis half — **pure pip, no torch, no GPU, no compiler** |
| [`descriptron-vision`](https://pypi.org/project/descriptron-vision/) | torchvision detectors, SAM2-PAL propagation, DINOv3 landmark transfer |
| [`descriptron-gui`](https://pypi.org/project/descriptron-gui/) | the annotation GUI |

If you only want to reproduce an analysis from deposited COCO files, install
`descriptron-core` alone: it needs no model of any kind and installs in about a
minute on Linux, macOS and Windows.

Detectron2 is an optional advanced backend and is not installed here — it has no
PyPI wheel and needs a compiler. See the repository for the Docker image that
carries it ready to run.
