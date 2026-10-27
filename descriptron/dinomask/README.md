# DINOMask - part segmentation from frozen DINOv3 tokens

DINOMask labels every structure in a specimen image from a few hand-drawn examples. It keeps the DINOv3 ViT-L/16
backbone frozen (`facebook/dinov3-vitl16-pretrain-lvd1689m`), trains a small per-patch classifier (LayerNorm + MLP)
on the patch tokens of your annotated images, and predicts with four half-patch-shifted passes, bilinear
probabilities and a guided filter (radius 8, eps 1e-3, grey photo as guide). The background class doubles as
"part absent" (character loss). Unlike SAM2-PAL there is no template-to-specimen propagation, so a template that
looks different from the target does not break it.

Files (all in this folder; run them from here or through the pip commands below):

| file | what it does |
|---|---|
| `dinomask_train_predict_v4.py` | train on a COCO file, report held-out IoU, and optionally predict a folder or a list of images |
| `dinomask_predict_rot4_v1.py` | predict with a trained run at 0/90/180/270 degrees and keep the most confident rotation (figures with the head on the right or sideways) |
| `fg_mask_v1.py` | specimen foreground from a plain background, used by `--fg_mask border` |
| `dinomask_hypercolumn_probe_v1.py` | 2.8.3: **points probe** - learn one small, repeated structure (setae, scales, punctures, pores) from keypoint clicks and find it on every image; see below |
| `dinomask_train_predict_v5.py` | 2.8.1: v4 plus `--cache_dir` (DINOv3 tokens computed once per image and reused), `--bg_keep` (keep only this share of background patches far from the specimen), `--lr` / `--hidden` / `--wd`, `--curve_json` (per-epoch train and held-out loss, per-class held-out IoU) and `--grid` (several lr:hidden settings on the same tokens, curves only); less memory (training array built in place). Same defaults and outputs as v4; v4 stays the default and is what the GUI runs |

Installed with pip (`descriptron-vision`): `descriptron-dinomask` (v4), `descriptron-dinomask-rot4` and, from 2.8.1,
`descriptron-dinomask-v5`; any of the files can also be run with `descriptron-vision dinomask/<file> ...`. A GPU is
strongly advised; all scripts fall back to the CPU. The DINOv3 weights are gated on Hugging Face: accept the licence once
and set up a token.

**Model:** use the ViT-L/16, `facebook/dinov3-vitl16-pretrain-lvd1689m`
(https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m; LVD-1689M web images, not the `sat493m` satellite
model); every result below used it. From 2.8.1 the three pip commands use it unless you pass `--model`, and so does the
GUI's DINOMask dialog. Run as a file (`python dinomask_train_predict_v4.py`, `descriptron-vision dinomask/<file>`) the
script's own default is still the small ViT-S/16, so pass `--model` there.
Request access on that page and set up a token once: step by step in [docs/HUGGINGFACE_TOKEN.md](../../docs/HUGGINGFACE_TOKEN.md).
The token can be in the environment (`HF_TOKEN`), saved with `hf auth login`, or - from 2.8.1, for the pip commands, the
GUI and the Docker `dinomask` commands - in Descriptron's key store (`descriptron-keys --set HF_TOKEN`).

## Recipe (used for the profiles of undescribed Tetramorium species, October 2026)

```bash
VOCAB=eye,gaster,head_capsule,mesosoma,...      # the full list of structures you annotate
descriptron-dinomask --coco <annotations.json> --images <image_dir> --out <run_dir> \
    --model facebook/dinov3-vitl16-pretrain-lvd1689m \
    --size 0 --aug rot180,hflip --containers auto --min_count 1 --species_regex '^$' \
    --partial_bg "$VOCAB" \
    --predict_list <targets.txt>                 # or --test_list a.tif,b.tif to score held-out images instead
```

- `--size 0` works at the native resolution (overlapping 1536 px tiles); `--size 2048` resizes the longest side.
- `--aug rot180,hflip` adds rotated and mirrored training copies; a mirrored copy swaps `left_`/`right_` names.
- `--containers auto` detects classes that enclose others (e.g. `head_capsule` around the clypeus) and gives each its
  own yes/no head, so the container keeps its full extent instead of "the head minus its parts".
- `--partial_bg <vocabulary>`: on an image where some structures were not drawn, unlabelled cells near the specimen
  are ignored in the loss instead of being taught as background.
- `--species_regex '^$'` with `--test_list` holds out exactly the listed images.
- Prediction writes `predictions.json` / `predictions.rlejson` (RLE masks) and `predictions_polygons.json`
  (polygons that open and edit in the Descriptron GUI), `absent_parts.json` and overlays.

Rotation-robust prediction with a trained run:

```bash
descriptron-dinomask-rot4 --run <run_dir> --images <image_dir> --out <out_dir> [--list targets.txt] [--size 2048]
```

## Points probe: small structures from a few clicks (2.8.3)

DINOMask masks the parts; the points probe finds the small things *on* them. Read the same spot at several depths of
DINOv3 (A. R. Van Dam's idea): the patch tokens of layers 6, 12, 18 and 23 are reduced to 64 PCA components each and
concatenated into one 256-number "hypercolumn" per 16-px patch (shallow layers carry the fine detail, deep layers what the
thing is; Hariharan et al. 2015 for the hypercolumn). Each feature has its local average inside the specimen subtracted
(`--highpass 3`), so the probe learns the structure rather than the body region the clicks happen to sit on. A logistic
regression separates the structure from its surroundings; four half-patch-shifted passes give ~8-px precision.

How to use it:
1. In the GUI, open an image and place **positive keypoints** on 15-30 examples of the structure, spread over the specimen
   (and over two or three specimens if you can). Mark a few look-alikes it must ignore (glare, debris, empty sockets) as
   **negative** points. Leaving the image auto-saves `<image>_keypoints.json`.
2. DINOMask dialog -> *Points probe: learn one structure from clicks*: choose the save (or the folder of saves), name the
   structure, and optionally a folder to predict.
3. The result opens in the editor as points: delete the wrong ones, click the missed ones, save, run again. Each round is
   a better probe.

Automatic negatives come from the cuticle around each positive click (2-8 patches away, `--neg_dist` / `--neg_far`).
Validation is printed and saved in `results.json`: leave-one-image-out when two or more images are clicked, otherwise a
2-fold left/right split of the one image (`--ablate` adds each layer alone). Outputs: `probe.joblib`,
`predictions_keypoints.json` (GUI keypoint format), `counts.json` (per image; per part with `--within_coco` = DINOMask
predictions), `overlay_*.jpg`.

```bash
descriptron-vision dinomask/dinomask_hypercolumn_probe_v1.py --points Tri-20_keypoints.json --images <image_dir> \
    --name scale_seta --out <run_dir> --predict_dir <folder> [--within_coco <dinomask predictions.json> --within elytron,pronotum]
descriptron-vision dinomask/dinomask_hypercolumn_probe_v1.py --run <run_dir> --predict_dir <folder> --out <out_dir>
```

First test (Tri-20, *Trigonopterus*, cream appressed scale-like setae, dense colour-derived labels; taught on the pronotum,
tested on the unseen elytra): AUC .997, AP .972; layer 23 alone .962 / .867, so combining depths is what sharpens it.
With 25 sparse clicks and no negatives the first version learned "pronotum" instead of "scale" (automatic negatives from
the whole body included the elytral scales); local negatives and the high-pass fixed that. Results from real clicks will be
added here.

## Held-out results so far

All are intersection-over-union against hand-drawn masks on images the model never saw.

| data | test | DINOMask | comparison |
|---|---|---|---|
| Trigonopterus weevils, lateral, 11 parts | leave-one-specimen-out over 12 hand-annotated images, ViT-L 2048 + guided filter | 0.60 (0.63 with SAM2 prompted by each mask on the same image) | SAM2-PAL with one template 0.54; SAM2-PAL fine-tuned on 11 images 0.52 |
| Ant mesosoma | 67 held-out images with reviewed masks, ViT-L 1344 | 0.84 | SAM2-PAL one template 0.81; SAM2-PAL fine-tuned on 12 0.85 |
| Tetramorium profiles (undescribed species), 12 structures | 4-fold cross-validation over 8 hand-corrected images (2 held out per fold), native resolution, recipe above; patch-level IoU | mean 0.65 (gaster 0.90, head capsule 0.87, petiole 0.82, eye 0.79; hind-leg segments 0.37-0.49) | adding 12 profiles from another source with a mapped vocabulary: 0.64 |
| Micro-CT renders of ant heads / mesosoma | 5 held-out species, ViT-L 1024 | head 0.82, mesosoma 0.71 | ViT-S 336 + SAM2 0.71 / 0.49 |

Weevil parts where DINOMask was clearly better than SAM2-PAL: femur, tibia, head, ventrites, coxa; SAM2-PAL kept the
rostrum. Learning curve on the weevils (3 held-out species): 3 training images 0.52, 6 0.57, 9 0.59 - still rising.

Method and numbers: A. R. Van Dam (DINOMask is his name for it and his idea of using the high-resolution ViT-L
features); built and tested in the Descriptron project, October 2026.
