#!/usr/bin/env python3
"""
tv_train_v1.py — fine-tune torchvision Mask R-CNN v2 or Keypoint R-CNN on COCO
==============================================================================

A drop-in replacement for `detectron2/detectron2_training_and_filterV10_and_kpts-17.py`
that needs no compiler and no conda environment of its own: torchvision ships as
an ordinary wheel on Linux and macOS, so this is the arm a user can install with
one pip command.

**The recipe is copied from the Detectron2 script, not reinvented**, because the
point of the swap is that nothing about the training changes:

    batch 2 · SGD momentum 0.9 · lr 5e-4 (= 0.00025 x batch, as cfg_local sets it)
    WarmupMultiStepLR, linear warmup over 8.5% of iterations, factor 0.001
    steps at 40% and 80% of training, gamma 0.1 · weight decay 1e-4
    gradient clipping by value 1.0 · 128 RoIs per image
    multi-scale shortest edge (640,672,704,736,768,800), max 1333
    rotation +/-45 deg, brightness/contrast/saturation 0.8-1.2, NO horizontal flip

One thing the Detectron2 script did that this deliberately does NOT do by default:
early stopping on bbox/AP50 with patience 5. When two backends are being compared
they must get the same number of iterations, or the comparison measures the
stopping rule instead of the model. `--early_stop_patience` turns it back on for
ordinary use.

    python tv_train_v1.py --task masks \
        --coco-json specimens.json --img-dir images/ --output-dir out/ \
        --total-iters 2000 --dataset-name diaphorina

    python tv_train_v1.py --task keypoints \
        --coco-json forewing_keypoints.json --img-dir images/ --output-dir out_kp/ \
        --total-iters 1500 --dataset-name forewing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tv_coco_dataset import CocoInstanceDataset, collate_fn   # noqa: E402

# the Detectron2 recipe, in one place so the sweep and the GUI cannot drift from it
RECIPE = dict(batch_size=2, lr_per_image=0.00025, momentum=0.9, weight_decay=1e-4,
              warmup_frac=0.085, warmup_factor=0.001, steps_frac=(0.40, 0.80),
              gamma=0.1, clip_grad_value=1.0, rois_per_image=128,
              min_size=(640, 672, 704, 736, 768, 800), max_size=1333,
              rotation=45.0)


def build_model(task, num_classes, num_keypoints=0, trainable_layers=3, pretrained=True,
                arch="v2"):
    """
    maskrcnn_resnet50_fpn_v2 or keypointrcnn_resnet50_fpn, with the heads resized
    to this dataset. COCO weights are kept for the backbone, FPN and RPN — that
    transfer is the whole reason fine-tuning on a few hundred images works — and
    only the final predictors, which are category- and keypoint-specific, are new.
    """
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if task == "masks":
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        # v2 has heavier heads (deeper box and mask heads with normalisation) and scores
        # higher on COCO; v1 is the lighter original and is architecturally the closer
        # match to Detectron2's mask_rcnn_R_50_FPN. On a few hundred images the smaller
        # model can converge further in the same budget, so both are selectable.
        if arch == "v1":
            from torchvision.models.detection import (maskrcnn_resnet50_fpn,
                                                      MaskRCNN_ResNet50_FPN_Weights)
            weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
            builder = maskrcnn_resnet50_fpn
        else:
            from torchvision.models.detection import (maskrcnn_resnet50_fpn_v2,
                                                      MaskRCNN_ResNet50_FPN_V2_Weights)
            weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
            builder = maskrcnn_resnet50_fpn_v2
        model = builder(
            weights=weights, trainable_backbone_layers=trainable_layers,
            box_batch_size_per_image=RECIPE["rois_per_image"])
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
        in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, hidden, num_classes)
    elif task == "keypoints":
        from torchvision.models.detection import (keypointrcnn_resnet50_fpn,
                                                  KeypointRCNN_ResNet50_FPN_Weights)
        from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
        weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
        # load with COCO's 17 person keypoints, then replace the predictor: the
        # pretrained keypoint head is still a better start than random init even
        # when the landmark count differs
        model = keypointrcnn_resnet50_fpn(
            weights=weights, trainable_backbone_layers=trainable_layers,
            box_batch_size_per_image=RECIPE["rois_per_image"])
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
        in_k = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_k, num_keypoints)
    else:
        raise ValueError(f"task must be masks or keypoints, got {task!r}")

    # multi-scale training, matching ResizeShortestEdge(choice) in the D2 recipe
    model.transform.min_size = tuple(RECIPE["min_size"])
    model.transform.max_size = RECIPE["max_size"]
    return model


def make_scheduler(optimizer, total_iters):
    """Linear warmup then MultiStep — the shape of Detectron2's WarmupMultiStepLR."""
    warmup_iters = max(1, int(RECIPE["warmup_frac"] * total_iters))
    steps = [int(f * total_iters) for f in RECIPE["steps_frac"]]

    def lr_lambda(it):
        if it < warmup_iters:
            alpha = it / warmup_iters
            return RECIPE["warmup_factor"] * (1 - alpha) + alpha
        return RECIPE["gamma"] ** sum(1 for s in steps if it >= s)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), warmup_iters, steps


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ids = json.load(open(args.train_ids)) if args.train_ids else None
    ds = CocoInstanceDataset(args.coco_json, args.img_dir, task=args.task,
                             image_ids=train_ids, augment=not args.no_augment,
                             mosaic_p=args.mosaic_p, mosaic_size=args.mosaic_size,
                             rotation=RECIPE["rotation"], seed=args.seed)
    if len(ds) == 0:
        raise SystemExit("no usable training images — check --coco-json and --task")

    batch = args.batch_size or RECIPE["batch_size"]
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate_fn, pin_memory=(device.type == "cuda"),
                        drop_last=len(ds) > batch, persistent_workers=args.num_workers > 0)

    model = build_model(args.task, ds.num_classes, ds.num_keypoints,
                        trainable_layers=args.trainable_layers,
                        pretrained=not args.no_pretrained, arch=args.arch).to(device)

    lr = args.lr if args.lr is not None else RECIPE["lr_per_image"] * batch
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=lr, momentum=RECIPE["momentum"],
                            weight_decay=RECIPE["weight_decay"])
    sched, warmup_iters, steps = make_scheduler(optim, args.total_iters)
    # torch.amp.* rather than torch.cuda.amp.*: the latter is deprecated from 2.4
    # and emits a FutureWarning on 2.14
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    meta = dict(task=args.task, arch=args.arch, dataset_name=args.dataset_name,
                num_classes=ds.num_classes, num_keypoints=ds.num_keypoints,
                categories=[{"id": c["id"], "name": c["name"]} for c in ds.categories],
                contiguous_to_name={str(k): v for k, v in ds.contig_to_name.items()},
                recipe={**RECIPE, "lr": lr, "batch_size": batch,
                        "total_iters": args.total_iters, "mosaic_p": args.mosaic_p,
                        "close_mosaic_frac": args.close_mosaic_frac,
                        "backbone": (("maskrcnn_resnet50_fpn" if args.arch == "v1"
                                      else "maskrcnn_resnet50_fpn_v2")
                                     if args.task == "masks" else "keypointrcnn_resnet50_fpn"),
                        "warmup_iters": warmup_iters, "steps": steps},
                train_images=len(ds))
    (out / f"metadata_{args.dataset_name}.json").write_text(json.dumps(meta, indent=2))

    close_at = int((1.0 - args.close_mosaic_frac) * args.total_iters)
    print(f"[tv_train] task={args.task} images={len(ds)} classes={ds.num_classes} "
          f"keypoints={ds.num_keypoints} device={device}")
    print(f"[tv_train] lr={lr:g} batch={batch} iters={args.total_iters} "
          f"warmup={warmup_iters} steps={steps} mosaic_p={args.mosaic_p}"
          + (f" (closed at {close_at})" if args.mosaic_p else ""))

    model.train()
    it, t0, losses, history = 0, time.time(), [], []
    while it < args.total_iters:
        for images, targets in loader:
            if it >= args.total_iters:
                break
            if args.mosaic_p and it == close_at:
                ds.mosaic_p = 0.0
                print(f"[tv_train] iter {it}: mosaic closed for the final stretch")
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                       for t in targets]
            if not any(len(t["boxes"]) for t in targets):
                it += 1
                continue
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                print(f"[tv_train] iter {it}: non-finite loss, batch skipped")
                optim.zero_grad(set_to_none=True)
                it += 1
                continue
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_value_(params, RECIPE["clip_grad_value"])
            scaler.step(optim)
            scaler.update()
            sched.step()
            losses.append(float(loss))
            it += 1

            if it % args.log_period == 0:
                rate = it / (time.time() - t0)
                mean = float(np.mean(losses[-args.log_period:]))
                eta = (args.total_iters - it) / max(rate, 1e-6)
                print(f"[tv_train] iter {it}/{args.total_iters} loss {mean:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {rate:.2f} it/s ETA {eta/60:.1f} min",
                      flush=True)
                history.append(dict(iter=it, loss=mean, lr=sched.get_last_lr()[0],
                                    iters_per_sec=rate))
            if args.checkpoint_period and it % args.checkpoint_period == 0:
                torch.save({"model": model.state_dict(), "iter": it, "meta": meta},
                           out / f"model_{args.dataset_name}_{it:06d}.pth")

    final = out / f"model_final_{args.dataset_name}.pth"
    torch.save({"model": model.state_dict(), "iter": it, "meta": meta}, final)
    elapsed = time.time() - t0
    (out / f"train_history_{args.dataset_name}.json").write_text(json.dumps(
        dict(history=history, seconds=elapsed, iters=it,
             iters_per_sec=it / max(elapsed, 1e-6)), indent=2))
    print(f"[tv_train] done: {it} iters in {elapsed/60:.1f} min "
          f"({it/max(elapsed,1e-6):.2f} it/s) -> {final}")
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["masks", "keypoints"], default="masks")
    p.add_argument("--arch", choices=["v2", "v1"], default="v2",
                   help="masks only: v2 = maskrcnn_resnet50_fpn_v2 (heavier heads, higher "
                        "COCO AP), v1 = maskrcnn_resnet50_fpn (lighter, closest match to "
                        "Detectron2's config). Ignored for keypoints.")
    # names match the Detectron2 script so the GUI can pass the same strings
    p.add_argument("--coco-json", required=True)
    p.add_argument("--img-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-name", default="your_taxon")
    p.add_argument("--total-iters", type=int, default=2000)
    p.add_argument("--checkpoint-period", type=int, default=1000)
    p.add_argument("--train-ids", default=None,
                   help="JSON list of COCO image ids to train on (used by the sweep)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 0.00025 x batch size, as the Detectron2 cfg_local sets it")
    p.add_argument("--trainable-layers", type=int, default=3,
                   help="ResNet stages left unfrozen (torchvision default 3)")
    p.add_argument("--mosaic_p", type=float, default=0.0,
                   help="probability of composing a 4-image mosaic (0 = off)")
    p.add_argument("--mosaic-size", type=int, default=512)
    p.add_argument("--close-mosaic-frac", type=float, default=0.15,
                   help="fraction of training at the end with mosaic switched off")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--amp", action="store_true",
                   help="mixed precision: faster, but leave it off when comparing backends")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-period", type=int, default=50)
    train(p.parse_args())


if __name__ == "__main__":
    main()
