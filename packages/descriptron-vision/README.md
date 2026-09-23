# descriptron-vision

Mask and landmark prediction for [Descriptron](https://github.com/alexrvandam/Descriptron):
torchvision detectors, SAM2-PAL propagation and DINOv3 landmark transfer.

```bash
pip install descriptron-vision
descriptron-train --task masks --coco-json annotations.json --img-dir images/ \
                  --output-dir out/ --total-iters 20000
descriptron-predict --checkpoint out/model_final_*.pth --coco-json annotations.json \
                    --img-dir images/ --annotations_out predictions.json
```

**No compiler required.** torch and torchvision are ordinary wheels on Linux,
macOS and Windows. For CUDA, install torch from PyTorch's own index first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install descriptron-vision
```

On a Mac the detectors can use the Apple GPU through PyTorch's MPS device, which
is why this backend exists: Detectron2 has no macOS wheels and no Apple Silicon
support.

## Two optional dependencies that cannot be declared

PyPI rejects a git dependency in package metadata, so neither **SAM2** nor
**Detectron2** is listed. Both are imported lazily, and the error names the
command to install them. The Docker image carries both ready to run.

## Choosing a backbone

`--arch v1` (default) is `maskrcnn_resnet50_fpn`; `--arch v2` is
`maskrcnn_resnet50_fpn_v2`. v2 scores higher on COCO, but on a few hundred
training images v1 won on 3 of 3 folds of a grouped species hold-out — its
lighter two-FC box head converges further in the same budget. Measured on the
data, not assumed from the model zoo.
