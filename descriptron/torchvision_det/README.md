# torchvision_det — instance segmentation and keypoints without a source build

A pip-installable replacement for the Detectron2 half of Descriptron. Same recipe,
same COCO files, same buttons; no compiler, no CUDA toolkit, and it works on macOS
as well as Linux, which Detectron2 does not.

| file | what it is |
|---|---|
| `tv_coco_dataset.py` | COCO JSON → the targets torchvision's detectors take; augmentation incl. **mosaic** |
| `tv_train_v1.py` | fine-tunes `maskrcnn_resnet50_fpn_v2` (masks) or `keypointrcnn_resnet50_fpn` (keypoints) |
| `tv_predict_v1.py` | predictions as Descriptron annotations **and** COCO detections; COCOeval with AP/AP50/AP75 |
| `d2_predict_fold_v1.py` | the same detections format from a Detectron2 checkpoint, so one evaluator scores both |
| `tv_sweep_v1.py` | grouped k-fold over species: does torchvision match Detectron2 on species it has never seen? |
| `tests/test_tv_dataset_v1.py` | 14 tests; the important one checks a rotated keypoint still lands on its rotated mask |

Runs in `samm` (torch 2.4 + torchvision 0.19). The Detectron2 arm of the sweep
shells out to `detectron2_env`; nothing else needs it.

## Two models, deliberately separate

`maskrcnn_resnet50_fpn_v2` for sclerites and `keypointrcnn_resnet50_fpn` for
landmarks are **separate models, separate runs, separate checkpoints**, exactly as
`--task` says. torchvision has no combined factory. (`RoIHeads` does accept a mask
head and a keypoint head at once, so one model for both is possible later by
composition — it is not what this ships.)

This also matches the data: in `diaphorina_29species_unified.json` 2,094 of 2,095
annotations carry a segmentation and exactly one carries keypoints. The landmarks
live in `diaphorina_forewing_keypoints.json` (97 images × 17 landmarks), which has
**no boxes at all** — Keypoint R-CNN is a detector and needs one per instance, so
the box is derived from the extent of the visible landmarks plus padding
(`boxes_from_keypoints`).

## Mosaic augmentation

Four images tiled into one canvas at a random split, annotations carried across;
works for **both** masks and keypoints. It could not be done in Detectron2 without
replacing its data loader — here it is ordinary Dataset code.

```
--mosaic_p 0.5 --mosaic-size 512 --close-mosaic-frac 0.15
```

Off by default, and worth knowing why before turning it on: each specimen ends up
at about half its linear size, and half a wing can end up beside half a head, so
the model sees scales and adjacencies that never occur at inference. That is a
real risk for morphometric structures, which is why `--close-mosaic-frac` switches
it off for the last stretch of training and why the sweep carries it as its own
arm rather than assuming it helps. Measured, not asserted.

## The recipe

Copied from `detectron2_training_and_filterV10_and_kpts-17.py`, in one place
(`tv_train_v1.RECIPE`): batch 2, SGD momentum 0.9, **lr 5e-4** (= 0.00025 × batch,
which is what `cfg_local` actually sets — the earlier `cfg.SOLVER.BASE_LR = 0.0001`
in that script is overwritten and never used), linear warmup over 8.5%, steps at
40%/80%, γ 0.1, weight decay 1e-4, gradient clipping by value 1.0, 128 RoIs per
image, shortest edge sampled from (640…800) with max 1333, rotation ±45°,
brightness/contrast/saturation 0.8–1.2, **no horizontal flip** (left/right
structures are distinct categories and a flipped wing is a different hand).

Two deliberate differences, both for comparability:

* **no early stopping by default.** The Detectron2 script stops on `bbox/AP50`
  with patience 5. Two backends must get the same number of iterations or the
  comparison measures the stopping rule.
* **no mixed precision by default** (`--amp` enables it). It is faster but it is
  not what the Detectron2 arm does.

## A finding about the Detectron2 script

It writes the *same* COCO file to both `train.json` and `val.json`
(lines 213–216) and copies every image into both directories. Its reported AP, its
`EvalHook` and its early stopping are therefore all **in-sample**. That is not a
reason to distrust the trained models, but any accuracy figure from that script is
optimistic, and the sweep imposes a real held-out split from outside.

## Running it

```bash
PY=/home/localuser/Desktop/Descriptron/conda/envs/samm/bin/python

# masks
$PY tv_train_v1.py --task masks --coco-json <coco.json> --img-dir <images/> \
    --output-dir <out/> --dataset-name <name> --total-iters 3000

# landmarks (separate model, separate run)
$PY tv_train_v1.py --task keypoints --coco-json <keypoints.json> --img-dir <images/> \
    --output-dir <out_kp/> --dataset-name <name> --total-iters 1500

# predict + score
$PY tv_predict_v1.py --checkpoint <out/model_final_<name>.pth> \
    --coco-json <coco.json> --img-dir <images/> \
    --annotations_out pred.json --detections_out det.json \
    --eval_against <coco.json>

# the backend comparison
$PY tv_sweep_v1.py --coco_json <coco.json> --img_dir <images/> \
    --group_labels <group_labels.csv> --output_dir <sweep/> \
    --folds 3 --iters 3000 --arms tv_v2,tv_v2_mosaic,d2
```

Measured on an RTX 3500 Ada (12 GB), 548 images at 1536×1024: **4.6 it/s** masks,
**5.5 it/s** masks with mosaic, **4.9 it/s** keypoints. So 3,000 iterations is
about 11 minutes, and the nine-run sweep is roughly two hours.

The sweep is resumable — a run whose `metrics_fold*.json` exists is skipped, so a
GPU hang costs one run, not the sweep.
