#!/usr/bin/env bash
# biorag_rebuild_downstream_v1.sh — everything downstream of the compiled feature table, in order
# ==================================================================================================
# When the annotations or the compiled table change, every result below changes with them, and doing
# the steps by hand is how a manuscript ends up quoting two versions of the same number. This runs
# them in the one order that is valid (matrix -> key -> calibration -> hold-out keys -> graded key ->
# knowledge graph -> instrument comparison -> optional arms -> figures) and stops at the first failure.
#
#   biorag_rebuild_downstream_v1.sh <python> <output_base> <taxon_profile> <compiled_dir> <exclude_list> \
#        [llm_backend: none|claude-code] [suffix for the folders it replaces]
#
# llm_backend   "none" (default) computes the key and leaves the couplets unworded - every number in the
#               paper is then already final, because the model only words the key, it never chooses a
#               character or a threshold. "claude-code" words the couplets through a Claude Code
#               subscription. The API is never used from here.
# suffix        existing result folders are MOVED to <name>_<suffix> first (nothing is deleted).
#
# Optional inputs, used when present under <output_base>:
#   treatments/machine_readable/jsonld        -> knowledge-graph arm
#   vlm_combined/vlm_character_states.tsv     -> model-proposed binary characters (congruence + key arm)
#   descriptive_states/                       -> the descriptive categorical character tables (S5-S6)
# Optional environment: OUTGROUPS = one "label=coco.json" per line;  CATEGORY_MAP='{"a":"b"}'
set -euo pipefail
PY="$1"; M="$2"; PROFILE="$3"; COMPILED="$4"; EXCL="$5"; BACKEND="${6:-none}"; SUF="${7:-}"
S="$(cd "$(dirname "$0")" && pwd)"
log() { printf '\n==== %s  [%s]\n' "$1" "$(date +%H:%M:%S)"; }
keep() { if [ -n "$SUF" ] && [ -e "$M/$1" ] && [ ! -e "$M/$1_$SUF" ]; then mv "$M/$1" "$M/$1_$SUF"; echo "   kept the previous $1 as $1_$SUF"; fi; }

cd "$S"
for d in compiled_key_tier key calibration key_holdout_v2 key_fuzzy_loo graph_test_loo instrument_comparison_v3 \
         congruence_compare compiled_key_tier_plus_vlm key_plus_vlm calibration_key_plus_vlm graph_support methods_summary character_robustness key_jackknife_support; do keep "$d"; done

log "0  is any photograph counted under two species? (stops here if so)"
"$PY" biorag_label_conflict_check_v1.py --compiled_dir "$COMPILED" --taxon_profile "$PROFILE" --out_dir "$M/label_conflicts" > "$M/label_conflicts.log"
tail -4 "$M/label_conflicts.log"

log "1  Tier-1 matrix"
"$PY" biorag_key_feature_filter_v2.py --compiled_dir "$COMPILED" --output_dir "$M/compiled_key_tier" \
      --taxon_profile "$PROFILE" --exclude_list "$EXCL" --exclude_flagged

log "2  key (computed; worded by: $BACKEND)"
"$PY" biorag_key_builder_v1.py --matrix_dir "$M/compiled_key_tier" --output_dir "$M/key" \
      --taxon-profile "$PROFILE" --llm-backend "$BACKEND"

log "3  calibration: key and matrix, both hold-out arms"
REL=(); [ -f "$M/descriptive_states/character_reliability.json" ] && REL=(--reliability "$M/descriptive_states/character_reliability.json")
"$PY" biorag_calibrate_v1.py --matrix_dir "$M/compiled_key_tier" --taxon_profile "$PROFILE" \
      --key_tree "$M/key/key_tree.json" --out_dir "$M/calibration" --python "$PY" "${REL[@]}"

log "4  hold-out keys (each species withheld)"
"$PY" biorag_key_holdout_v1.py --matrix_dir "$M/compiled_key_tier" --taxon_profile "$PROFILE" \
      --out_dir "$M/key_holdout_v2" --baseline "$M/key/identification_test.tsv" --python "$PY"

log "5  graded key, each specimen withheld"
"$PY" biorag_key_fuzzy_v1.py --matrix_dir "$M/compiled_key_tier" --taxon_profile "$PROFILE" \
      --key_tree "$M/key/key_tree.json" --out_dir "$M/key_fuzzy_loo" --python "$PY" --known_arm leave_one_specimen_out

log "5b jackknife support for every couplet (recovered in the rebuilt keys; withheld specimens routed the right way)"
"$PY" biorag_key_jackknife_support_v1.py --matrix_dir "$M/compiled_key_tier" --taxon_profile "$PROFILE" \
      --key_tree "$M/key/key_tree.json" --key_loo_dir "$M/key_fuzzy_loo/keys_loo" --out_dir "$M/key_jackknife_support" | tail -30

GRAPH=()
if [ -d "$M/treatments/machine_readable/jsonld" ]; then
  log "6  knowledge graph, each specimen withheld from its own ranges"
  "$PY" biorag_graph_identify_v1.py --jsonld_dir "$M/treatments/machine_readable/jsonld" --matrix_dir "$M/compiled_key_tier" \
        --taxon_profile "$PROFILE" --out_dir "$M/graph_test_loo" --arm leave_one_specimen_out --ranges_from matrix
  GRAPH=(--graph_scores "$M/graph_test_loo/graph_specimen_scores.tsv")
else
  echo "   no treatments/machine_readable/jsonld: knowledge-graph arm skipped"
fi

log "7  key, graph and matrix on both questions"
"$PY" biorag_instrument_compare_v3.py --calibration "$M/calibration" --key_loo "$M/key/identification_test.tsv" \
      "${GRAPH[@]}" --fuzzy "$M/key_fuzzy_loo/fuzzy_key_specimens.tsv" --out_dir "$M/instrument_comparison_v3"

if [ -f "$M/vlm_combined/vlm_character_states.tsv" ]; then
  log "8a model-proposed binary characters: do they add to the matrix?"
  "$PY" biorag_congruence_compare_v1.py --matrix_dir "$M/compiled_key_tier" --taxon_profile "$PROFILE" \
        --extra_states "$M/vlm_combined/vlm_character_states.tsv" --proposals "$M/vlm_combined/vlm_proposed_characters.tsv" \
        --key_loo "$M/key/identification_test.tsv" --out_dir "$M/congruence_compare"
  log "8b ... and in the key"
  "$PY" biorag_add_discrete_characters_v1.py --matrix_dir "$M/compiled_key_tier" --states "$M/vlm_combined/vlm_character_states.tsv" \
        --out_dir "$M/compiled_key_tier_plus_vlm" || echo "   (check the options of biorag_add_discrete_characters_v1.py; step 8b skipped)"
  if [ -f "$M/compiled_key_tier_plus_vlm/specimen_matrix_long.csv" ]; then
    "$PY" biorag_key_builder_v1.py --matrix_dir "$M/compiled_key_tier_plus_vlm" --output_dir "$M/key_plus_vlm" --taxon-profile "$PROFILE" --llm-backend none
    "$PY" biorag_key_fuzzy_v1.py --matrix_dir "$M/compiled_key_tier_plus_vlm" --taxon_profile "$PROFILE" --key_tree "$M/key_plus_vlm/key_tree.json" \
          --out_dir "$M/key_plus_vlm/fuzzy_loo" --python "$PY" --known_arm leave_one_specimen_out
    "$PY" biorag_calibrate_v1.py --matrix_dir "$M/compiled_key_tier_plus_vlm" --taxon_profile "$PROFILE" --key_tree "$M/key_plus_vlm/key_tree.json" \
          --out_dir "$M/calibration_key_plus_vlm" --python "$PY" --no_matrix --no_figures
  fi
fi

if [ -d "$M/descriptive_states" ] && [ -f "$S/biorag_rerun_descriptive_tables_v1.sh" ]; then
  log "9  descriptive categorical characters: tables S5-S6"
  bash "$S/biorag_rerun_descriptive_tables_v1.sh" "$PY" "$M" "$PROFILE" "$COMPILED" "${SUF:-}"
fi

if [ -n "${OUTGROUPS:-}" ]; then
  log "10 out-of-reference sets"
  while IFS= read -r pair; do                       # one "label=path" per LINE, so that paths may contain spaces
    [ -z "$pair" ] && continue
    label="${pair%%=*}"; coco="${pair#*=}"
    [ -n "$SUF" ] && [ -e "$M/novelty_$label" ] && [ ! -e "$M/novelty_${label}_$SUF" ] && mv "$M/novelty_$label" "$M/novelty_${label}_$SUF"
    "$PY" biorag_novelty_score_v1.py --matrix_dir "$M/compiled_key_tier" --taxon_profile "$PROFILE" --key_tree "$M/key/key_tree.json" \
          --out_dir "$M/novelty_$label" --candidate_coco "$coco" --candidate_label "$label" ${CATEGORY_MAP:+--category_map "$CATEGORY_MAP"}
  done <<< "$OUTGROUPS"
fi

log "10b what each character is worth on its own (both hold-outs, one character at a time)"
"$PY" biorag_character_robustness_v1.py --matrix_dir "$M/compiled_key_tier" --out_dir "$M/character_robustness"

log "11 figures"
"$PY" biorag_key_fuzzy_figures_v1.py --matrix_dir "$M/compiled_key_tier" --taxon_profile "$PROFILE" --key_loo_dir "$M/key_fuzzy_loo/keys_loo" \
      --fuzzy_specimens "$M/key_fuzzy_loo/fuzzy_key_specimens.tsv" --identification_test "$M/key/identification_test.tsv" \
      --out_dir "$M/key_fuzzy_loo/figures" --python "$PY"
if [ -d "$M/treatments/machine_readable/jsonld" ]; then
  "$PY" biorag_graph_support_figure_v1.py --matrix_dir "$M/compiled_key_tier" --taxon_profile "$PROFILE" \
        --jsonld_dir "$M/treatments/machine_readable/jsonld" --distances "$M/calibration/matrix_identification_distances.tsv" \
        --identification "$M/instrument_comparison_v3/identification_by_specimen.tsv" --key_loo_dir "$M/key_fuzzy_loo/keys_loo" \
        --out_dir "$M/graph_support"
fi
"$PY" biorag_methods_summary_figure_v1.py --monograph "$M" --out_dir "$M/methods_summary" || echo "   (methods summary figure skipped)"
"$PY" biorag_calibration_figures_v1.py --calibration "$M/calibration" --key_tree "$M/key/key_tree.json" \
      ${REL:+--reliability "$M/descriptive_states/character_reliability.tsv"}

log "done. Next: audit the treatments against the new matrix (biorag_confabulation_checker_v2.py), repair what is stale, rebuild the manuscript."
