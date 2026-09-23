# descriptron

Image annotation, phenomics and evidence-tiered species descriptions.

```bash
pip install descriptron          # everything
descriptron-gui                  # the annotation GUI
descriptron --list               # the 62 analysis programs
```

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
