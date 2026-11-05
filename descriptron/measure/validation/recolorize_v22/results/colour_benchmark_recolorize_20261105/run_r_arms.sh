#!/usr/bin/env bash
# patternize then Colormesh on the recolorized inputs, one after the other (same settings as the 28 Sep benchmark).
set -euo pipefail
: "${R:?}" "${V:?}" "${L:?}"
echo "$(date -Is) start patternize"
/usr/bin/Rscript "$V/colour_benchmark_patternize.R" --lib "$L" --inputs "$R/inputs" --outdir "$R/patternize" > "$R/patternize_stdout.log" 2>&1
echo "$(date -Is) start colormesh"
/usr/bin/Rscript "$V/colour_benchmark_colormesh.R" --lib "$L" --inputs "$R/inputs" --outdir "$R/colormesh" > "$R/colormesh_stdout.log" 2>&1
echo "$(date -Is) R ARMS DONE"
