#!/usr/bin/env bash
# biorag_rerun_descriptive_tables_v1.sh — Supplementary Tables S5 and S6 on the current matrix
# ============================================================================================
#
# Tables S5 (the six character sets, calibrated one by one) and S6 (the gating ablation: are the
# descriptive characters used as scored or banded, and are the unrepeatable ones set aside?)
# were first computed on the Tier-1 matrix of 17 September, before the annotation screen removed
# the polygons that did not sit on the specimen and before the pooled spread was re-derived
# without the item under test. This script recomputes both on whatever matrix it is given, so
# the two tables stand on the same footing as the rest of the paper.
#
# No model is called: the descriptive states and their retest were scored once and are read
# from disk. Nothing is overwritten without a copy — the previous runs are kept beside the new
# ones with the suffix given as the fifth argument.
#
#   usage: biorag_rerun_descriptive_tables_v1.sh <python> <monograph_dir> <taxon_profile> \
#                                                <compiled_dir> [suffix_for_the_old_runs]
#
#   <compiled_dir> is the compiled feature directory the "measured stand-ins" set is read from
#   (the --compiled_dir given to biorag_key_feature_filter_v2.py).
#
# The four cells of the ablation differ only in two switches of biorag_novelty_score_v1.py:
#   --reliability <json>      drop the characters whose states did not repeat on a second reading
#   --coarse_descriptive      score the rest at their two-or-three-state bands
# and each is calibrated to the same target rate of calling a described specimen novel, so the
# detection rates are directly comparable.
set -euo pipefail

PY="${1:?python interpreter}"
M="${2:?monograph directory}"
PROFILE="${3:?taxon profile yaml}"
COMPILED="${4:?compiled feature directory}"
SUFFIX="${5:-previous_$(date +%Y%m%d)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# optional 6th and 7th arguments: another reading of the descriptive characters (a folder holding
# descriptive_states_by_specimen.tsv and character_reliability.json/.tsv) and where its six-set run goes;
# the defaults are the folders the paper has always read
DS="${6:-$M/descriptive_states}"
NG="${7:-$M/novelty_gated}"
STATES="$DS/descriptive_states_by_specimen.tsv"
REL="$DS/character_reliability.json"
AB="$DS/gating_ablation"
for f in "$STATES" "$REL" "$M/compiled_key_tier/specimen_matrix_long.csv" "$M/key/key_tree.json"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

keep() {  # keep the previous run beside the new one, once
    if [ -e "$1" ] && [ ! -e "${1}_${SUFFIX}" ]; then cp -r "$1" "${1}_${SUFFIX}"; echo "kept: ${1}_${SUFFIX}"; fi
}
keep "$AB"; keep "$NG"; keep "$DS/figures"

score() {  # score <out_dir> [extra switches...]
    local out="$1"; shift
    mkdir -p "$out"
    "$PY" "$HERE/biorag_novelty_score_v1.py" --matrix_dir "$M/compiled_key_tier" \
        --taxon_profile "$PROFILE" --key_tree "$M/key/key_tree.json" \
        --descriptive_matrix "$STATES" --out_dir "$out" "$@" > "$out/run.log" 2>&1
    echo "done: $out"
}

echo "[1/3] gating ablation, four cells (calibration only)"
score "$AB/ungated"
score "$AB/coarse_only"      --coarse_descriptive
score "$AB/gate_only"        --reliability "$REL"
score "$AB/gate_and_coarse"  --reliability "$REL" --coarse_descriptive

echo "[2/3] the run the paper reads: gated, full scales, six character sets, every species withheld in turn"
score "$NG" --reliability "$REL" --computed_dir "$COMPILED" --holdout_all

echo "[3/3] tables and figures"
"$PY" "$HERE/biorag_reliability_figures_v1.py" \
    --reliability "$DS/character_reliability.tsv" \
    --ablation_dir "$AB" --final_run "$NG" \
    --scoring_report "$DS/scoring_report.json" \
    --repeatability "$DS/repeatability.json" \
    --out_dir "$DS/figures"
echo "tables: $DS/figures/table3_gating_ablation.tsv, table4_character_sets.tsv"
