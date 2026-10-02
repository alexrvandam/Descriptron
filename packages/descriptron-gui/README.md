# descriptron-gui

The [Descriptron](https://github.com/alexrvandam/Descriptron) annotation GUI:
SAM2-assisted segmentation, keypoints, line annotations and COCO export.

```bash
SAM2_BUILD_CUDA=0 pip install git+https://github.com/facebookresearch/sam2.git   # SAM2 library, not on PyPI (Windows: set SAM2_BUILD_CUDA=0 first)
pip install descriptron-gui
descriptron-gui          # first start downloads the SAM2 model file (898 MB) into ~/.cache/descriptron/sam2/
```

SAM2 must be installed from Meta's GitHub into the same environment (PyPI does not allow it as a dependency). If you
already have `sam2_hiera_large.pt`, copy it to `~/.cache/descriptron/sam2/` or set `DESCRIPTRON_SAM2_CHECKPOINT`
to its path. Step by step for Windows, macOS and Linux:
https://github.com/alexrvandam/Descriptron/blob/main/docs/INSTALL_PIP.md

Pulls in `descriptron-core` and `descriptron-vision`: the GUI drives the analysis
programs directly, so it is not a thin front end over them.

**tkinter** is in the standard library but is not always packaged with Python on
Linux. If the GUI does not start:

```bash
sudo apt install python3-tk      # Debian/Ubuntu
sudo dnf install python3-tkinter # Fedora/RHEL
```

Model weights are not bundled; they download on first use and are cached. Drag an image or a folder onto the
window, or start with one: `descriptron-gui path/to/images`.
