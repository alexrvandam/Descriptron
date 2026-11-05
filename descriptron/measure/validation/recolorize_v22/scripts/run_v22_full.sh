#!/usr/bin/env bash
# V22 --recolorize on all 423 moths, two settings matching the R runs (default_raw, b8_avg_T20_raw), one after the other,
# then V22-vs-R comparison per image. Env: O (test root), PAIRS (fg/bin pairs), PY (measure_env python), MEASURE (gui/measure).
set -euo pipefail
: "${O:?}" "${PAIRS:?}" "${PY:?}" "${MEASURE:?}"
export CUDA_VISIBLE_DEVICES=""
cd "$MEASURE"
for c in default b8_avg_T20; do
  D="$O/v22_tests/full_$c"; [ -f "$D/DONE" ] && { echo "skip $c"; continue; }
  mkdir -p "$D"; X=""; [ $c = b8_avg_T20 ] && X="--recolorize_bins 8 --recolorize_linkage average --recolorize_cutoff 20"
  echo "$(date -Is) start V22 $c"
  nice -n 10 "$PY" FHS_and_CLAHE_V22.py --input_dir "$PAIRS" --output_dir "$D" --recolorize $X --category_name moth_dorsal > "$D/log.txt" 2>&1
  "$PY" "$O/v22_tests/scripts/compare_v22_vs_R.py" "$O/recolorize/${c}_raw" "$D" "$O/v22_tests/compare_full_$c.csv" > /dev/null
  touch "$D/DONE"; echo "$(date -Is) end V22 $c"
done
echo "$(date -Is) V22 FULL COMPLETE"
