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

Installed with pip (`descriptron-vision`): `descriptron-dinomask` and `descriptron-dinomask-rot4`; any of the files can
also be run with `descriptron-vision dinomask/<file> ...`. A GPU is strongly advised; both scripts fall back to the
CPU. The DINOv3 weights are gated on Hugging Face: accept the licence once and set `HF_TOKEN`.

**Model:** always pass `--model facebook/dinov3-vitl16-pretrain-lvd1689m` - the ViT-L/16 at
https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m (LVD-1689M web images, not the `sat493m` satellite
model). Without `--model` the script falls back to the small ViT-S/16; every result below used the ViT-L.
Request access on that page and set up a token once: step by step in [docs/HUGGINGFACE_TOKEN.md](../../docs/HUGGINGFACE_TOKEN.md)
(the token must be in the environment or saved with `hf auth login`; Descriptron's own key store is not read by DINOMask
in 2.8.0).

## Recipe (used for the Austromyrma profiles, October 2026)

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

## Held-out results so far

All are intersection-over-union against hand-drawn masks on images the model never saw.

| data | test | DINOMask | comparison |
|---|---|---|---|
| Trigonopterus weevils, lateral, 11 parts | leave-one-specimen-out over 12 hand-annotated images, ViT-L 2048 + guided filter | 0.60 (0.63 with SAM2 prompted by each mask on the same image) | SAM2-PAL with one template 0.54; SAM2-PAL fine-tuned on 11 images 0.52 |
| Ant mesosoma | 67 held-out images with reviewed masks, ViT-L 1344 | 0.84 | SAM2-PAL one template 0.81; SAM2-PAL fine-tuned on 12 0.85 |
| Austromyrma profiles, 12 structures | 4-fold cross-validation over 8 hand-corrected images (2 held out per fold), native resolution, recipe above; patch-level IoU | mean 0.65 (gaster 0.90, head capsule 0.87, petiole 0.82, eye 0.79; hind-leg segments 0.37-0.49) | adding 12 profiles from another source with a mapped vocabulary: 0.64 |
| Micro-CT renders of ant heads / mesosoma | 5 held-out species, ViT-L 1024 | head 0.82, mesosoma 0.71 | ViT-S 336 + SAM2 0.71 / 0.49 |

Weevil parts where DINOMask was clearly better than SAM2-PAL: femur, tibia, head, ventrites, coxa; SAM2-PAL kept the
rostrum. Learning curve on the weevils (3 held-out species): 3 training images 0.52, 6 0.57, 9 0.59 - still rising.

Method and numbers: A. R. Van Dam (DINOMask is his name for it and his idea of using the high-resolution ViT-L
features); built and tested in the Descriptron project, October 2026.
