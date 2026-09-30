# Colour pattern, texture and shape on a phylogeny: moths (Supplementary Text S13.6)

43 species of Noctuidae and Erebidae on the published time tree of Nokelainen et al. (2024, *Nature Communications*
15: 1678; tree and species values from the article's Source Data; species list and image sources from the JYX dataset
doi:10.17011/jyx/dataset/92453, CC BY 4.0). The authors do not list vouchers, so the photographs are new: museum
specimens of the same species retrieved from GBIF. This validates Descriptron's image-to-phylogeny pipeline and its
statistics against geomorph, ape and phytools; it does not reproduce the authors' colour measurements.

`run_all.sh` (all paths via environment variables; see its header) runs:
1. `gbif_query.py`, `download_select.py`, `retry_failed.py`: preserved-specimen photographs with licences (`manifest.csv`);
2. `seg_moths.py`: Florence-2 "a moth" → SAM2 box → mask, antennae removed, automatic QC;
3. `select_build_coco.py`: one image per specimen, up to 10 per species, visual-QC exclusions in `exclude_qc.txt`;
4. Descriptron semilandmarks (V42), colour and texture homology cells, measurements; `assemble_traits.py`;
5. `descriptron_phylo.py` and `../validate_phylo_traits.py` (geomorph/ape/phytools);
6. `compare_published.py`: Kmult of the authors' 14 species traits (Descriptron vs phytools) and Descriptron's
   saturation/brightness vs the published values;
7. `extra_checks.py`: two-museum robustness, the full 82-tip tree pruned by descriptron_phylo, PGLS on strategy.

Downloads depend on the museums' servers (in September 2026 the Natural History Museum, London, refused them and
Symbiota refused parallel requests); the specimens actually used are listed by GBIF occurrence id in `specimens_used.csv`.
