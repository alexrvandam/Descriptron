#!/usr/bin/env bash
# Moths (Noctuidae + Erebidae, 43 spp.; Nokelainen et al. 2024): GBIF photographs -> Descriptron traits ->
# phylogenetic comparative analyses, compared with geomorph/ape/phytools. Supplementary Text S13.6.
# Set these (no defaults):
#   WORK       working folder; must contain clade_sources.tsv (from the JYX dataset, Supplementary data 2) and
#              SourceData/ (Nat Commun Source Data: SourceData_Fig3_ultrametric-1.tre, SourceData_phylanova_colourmetrics.txt)
#   DESC_GUI   Descriptron gui/ folder;  SAM2_DIR   segment-anything-2 folder (checkpoints/sam2_hiera_large.pt)
#   PY_SAM     python with sam2 + transformers (Florence-2);  PY_MEASURE  python for the measure scripts
#   RSCRIPT, RLIB   Rscript and a library with ape, geomorph, phytools
set -euo pipefail
: "${WORK:?}" "${DESC_GUI:?}" "${SAM2_DIR:?}" "${PY_SAM:?}" "${PY_MEASURE:?}" "${RSCRIPT:?}" "${RLIB:?}"
V="$(cd "$(dirname "$0")" && pwd)"; M="$DESC_GUI/measure"; cd "$WORK"
[ -f exclude_qc.txt ] || cp "$V/exclude_qc.txt" .       # visual QC exclusions of the published run
"$RSCRIPT" "$V/prune_tree.R" "$RLIB" SourceData/SourceData_Fig3_ultrametric-1.tre tree_noctuoidea43.nwk
[ -f gbif_candidates.csv ] || python3 "$V/gbif_query.py"
[ -f manifest.csv ] || python3 "$V/download_select.py"
python3 "$V/retry_failed.py"
if [ ! -f seg/qc.csv ]; then ls "$WORK"/images_raw/*.jpg > seg_list.txt
  (cd "$SAM2_DIR" && "$PY_SAM" "$V/seg_moths.py" --list "$WORK/seg_list.txt" --out "$WORK/seg" --sam2_ckpt checkpoints/sam2_hiera_large.pt --sam2_cfg sam2_hiera_l.yaml > "$WORK/log_seg.txt" 2>&1); fi
rm -rf images semilandmarks color_homology texture_homology measurements; "$PY_MEASURE" "$V/select_build_coco.py"
(cd "$M"
 "$PY_MEASURE" semi_landmark_and_kpts_procrustesV42_GPA.py --json "$WORK/coco.json" --image_dir "$WORK/images" --output_dir "$WORK/semilandmarks" --num_landmarks 100 --anchor_method none --alignment_method reflect_mirrored > "$WORK/log_semilandmarks.txt" 2>&1
 "$PY_MEASURE" color_phenomics_homology_v2_1.py --gpa_dir "$WORK/semilandmarks/moth_dorsal" --json "$WORK/coco.json" --image_dir "$WORK/images" --output_dir "$WORK/color_homology" --category_name moth_dorsal --color_mode combined > "$WORK/log_colour.txt" 2>&1
 "$PY_MEASURE" texture_phenomics_homology.py --gpa_dir "$WORK/semilandmarks/moth_dorsal" --json "$WORK/coco.json" --image_dir "$WORK/images" --output_dir "$WORK/texture_homology" --category_name moth_dorsal > "$WORK/log_texture.txt" 2>&1
 "$PY_MEASURE" measurement_script_to_try_after_kpts_prediction_measure_kpts_V35.py --json "$WORK/coco.json" --image_dir "$WORK/images" --output_dir "$WORK/measurements" --method pca --save_results --output_file "$WORK/measurements/all_metrics.csv" --jsonl_output "$WORK/measurements/all_metrics.jsonl" > "$WORK/log_measurements.txt" 2>&1)
"$PY_MEASURE" "$V/assemble_traits.py"
T="--species_col species --traits shape=shape.csv --traits meas=meas.csv --traits colour=colour.csv --traits texture=texture.csv --scale colour=standardize --scale texture=standardize"
"$PY_MEASURE" "$M/descriptron_phylo.py" --tree tree_noctuoidea43.nwk $T --pgls "shape~meas:aspect_ratio" --pgls "colour~meas:aspect_ratio" --pgls "texture~meas:aspect_ratio" --analyses signal pgls morphospace --iterations 999 --seed 1 --out_dir descriptron_phylo_out > log_phylo.txt 2>&1
"$PY_MEASURE" "$M/validation/validate_phylo_traits.py" --tree tree_noctuoidea43.nwk $T --predictor meas:aspect_ratio --rscript "$RSCRIPT" --rlib "$RLIB" --out-dir compare > log_compare.txt 2>&1
"$PY_MEASURE" "$V/compare_published.py" --measure_dir "$M" --rscript "$RSCRIPT" --rlib "$RLIB" --published SourceData/SourceData_phylanova_colourmetrics.txt
# robustness: two museums only; the authors' full tree pruned by descriptron_phylo; PGLS on anti-predator strategy
"$PY_MEASURE" "$V/extra_checks.py"
(cd robust_ethz_tam && "$PY_MEASURE" "$M/descriptron_phylo.py" --tree ../tree_noctuoidea43.nwk $T --analyses signal --iterations 999 --seed 1 --out_dir out > log.txt 2>&1)
"$PY_MEASURE" "$M/descriptron_phylo.py" --tree tree_full82_renamed.nwk $T --pgls "shape~meas:aspect_ratio" --pgls "colour~meas:aspect_ratio" --pgls "texture~meas:aspect_ratio" --analyses signal pgls --iterations 999 --seed 1 --out_dir descriptron_phylo_fulltree > log_phylo_fulltree.txt 2>&1
diff <(grep -E "Kmult|PGLS|species," log_phylo.txt) <(grep -E "Kmult|PGLS|species," log_phylo_fulltree.txt) && echo "full 82-tip tree: identical results"
S="--species_col species --traits colour=colour.csv --traits texture=texture.csv --traits shape=shape.csv --traits strat=strategy.csv --scale colour=standardize --scale texture=standardize"
"$PY_MEASURE" "$M/descriptron_phylo.py" --tree tree_noctuoidea43.nwk $S --pgls "colour~strat:aposematic" --pgls "texture~strat:aposematic" --pgls "shape~strat:aposematic" --analyses pgls --iterations 999 --seed 1 --out_dir descriptron_phylo_strategy > log_phylo_strategy.txt 2>&1
"$PY_MEASURE" "$M/validation/validate_phylo_traits.py" --tree tree_noctuoidea43.nwk $S --predictor strat:aposematic --rscript "$RSCRIPT" --rlib "$RLIB" --out-dir compare_strategy > log_compare_strategy.txt 2>&1
echo "done: $WORK/log_phylo.txt, compare/, compare_published/, robust_ethz_tam/, compare_strategy/"
