#!/usr/bin/env bash
# =============================================================================
# run_20k_comparison.sh — Detectron2 vs torchvision v1 at a realistic length
# =============================================================================
# The 22 Sep sweep ran 3,000 iterations to fit an afternoon and found Detectron2
# ahead by 0.102 AP / 0.131 AP75. That budget is about a tenth of a real training
# run, and the torchvision arm may simply have been further from convergence.
# This settles it at 20,000 iterations, which is long enough for both arms to
# converge on ~365 training images (about 110 epochs at batch 2) without paying
# for the 35,000 the GUI defaults to.
#
# Only the two arms worth the time are run: d2 (the incumbent) and tv_v1 (the
# best torchvision arm — it beat tv_v2 on 3 of 3 folds). Mosaic is excluded: it
# was measured at 3,000 and did not help.
#
# FAIRNESS: the Detectron2 arm uses detectron2_training_sweep_v1.py, which is the
# production script with early stopping removed and its in-sample evaluation
# moved to the end. At 3,000 iterations early stopping could not fire (2
# evaluations); at 20,000 there are 20 and it would stop that arm early, so the
# comparison would measure the stopping rule instead of the model.
#
#   ./run_20k_comparison.sh            # 3 folds, both arms  (~8 h)
#   ./run_20k_comparison.sh 1          # 1 fold,  both arms  (~2.7 h)
#   ./run_20k_comparison.sh 3 dry      # print the plan and stop
#
# Resumable: a run whose <arm>/metrics_fold<N>.json exists is skipped, so a
# crash or a GSP freeze costs one run, not the night. Just re-run this script.
# =============================================================================
set -euo pipefail

FOLDS="${1:-3}"
DRY="${2:-}"
ITERS=20000

PY=/home/localuser/Desktop/Descriptron/conda/envs/samm/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
U="/media/localuser/T7 Shield/Diaphorina_unified_20260921"
IMAGES="/media/localuser/T7 Shield/Diaphorina_images"

# A NEW output directory on purpose. The sweep skips any run whose metrics file
# already exists, so pointing this at the 3,000-iteration directory would skip
# every run and produce nothing.
OUT="/media/localuser/T7 Shield/Diaphorina_backbone_sweep_20k_$(date +%Y%m%d)"

# reuse the staged images the 3,000-iteration run already copied, if they are
# still there — it saves recopying 3.4 GB, and the Detectron2 script must never
# be pointed at the master image folder (it writes train/ and val/ copies into
# whatever it is given)
STAGE_OLD="/media/localuser/T7 Shield/Diaphorina_backbone_sweep_20260922/d2_stage/images"
if [ -d "$STAGE_OLD" ]; then D2IMG="$STAGE_OLD"; else D2IMG="$OUT/d2_stage/images"; fi

mkdir -p "$OUT"
cat <<EOF
Detectron2 vs torchvision v1, $ITERS iterations, $FOLDS fold(s)
  data    : $U/diaphorina_29species_unified_sizefix.json   (the size-repaired file)
  images  : $IMAGES
  d2 stage: $D2IMG
  output  : $OUT
  arms    : tv_v1 (faster, runs first) then d2

  Measured rates on the RTX 3500 Ada: tv_v1 ~5.4 it/s, d2 ~3.4 it/s.
  Estimate: tv_v1 ~62 min/fold, d2 ~98 min/fold  =>  ~$(( FOLDS * 160 / 60 )) h for $FOLDS fold(s).
  Run it detached (tmux, or nohup) — it outlives the terminal either way.
EOF
[ "$DRY" = "dry" ] && { echo "(dry run — nothing started)"; exit 0; }

echo "started $(date)" | tee -a "$OUT/run.log"

# tv_v1 first so a partial result is useful if the night is interrupted
"$PY" -u "$HERE/tv_sweep_v1.py" \
  --coco_json "$U/diaphorina_29species_unified_sizefix.json" \
  --img_dir "$IMAGES" --group_labels "$U/group_labels.csv" \
  --output_dir "$OUT" --folds "$FOLDS" --iters "$ITERS" \
  --arms tv_v1 --num-workers 4 --seed 0 2>&1 | tee -a "$OUT/run.log"

"$PY" -u "$HERE/tv_sweep_v1.py" \
  --coco_json "$U/diaphorina_29species_unified_sizefix.json" \
  --img_dir "$IMAGES" --group_labels "$U/group_labels.csv" \
  --output_dir "$OUT" --folds "$FOLDS" --iters "$ITERS" \
  --arms d2 --d2_img_dir "$D2IMG" \
  --d2_train_script "$HERE/../detectron2/detectron2_training_sweep_v1.py" \
  --num-workers 4 --seed 0 2>&1 | tee -a "$OUT/run.log"

echo "finished $(date)" | tee -a "$OUT/run.log"
echo
echo "Results: $OUT/sweep_results.tsv"
echo "Compare with the 3,000-iteration run:"
echo "  d2 0.4731 AP / 0.5064 AP75   tv_v1 0.3714 / 0.3756"
echo "If the gap has closed, torchvision ships as the default and Detectron2 becomes"
echo "a genuine niche. If it holds, the README states the cost of the default plainly."
