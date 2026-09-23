# descriptron-gui

The [Descriptron](https://github.com/alexrvandam/Descriptron) annotation GUI:
SAM2-assisted segmentation, keypoints, line annotations and COCO export.

```bash
pip install descriptron-gui
descriptron-gui
```

Pulls in `descriptron-core` and `descriptron-vision`: the GUI drives the analysis
programs directly, so it is not a thin front end over them.

**tkinter** is in the standard library but is not always packaged with Python on
Linux. If the GUI does not start:

```bash
sudo apt install python3-tk      # Debian/Ubuntu
sudo dnf install python3-tkinter # Fedora/RHEL
```

Model weights are not bundled; they download on first use and are cached.
