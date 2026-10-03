# Open-set benchmark: the character matrix against biodiversity foundation models

Scripts behind supplement S13.7 (recognising a withheld species, naming a withheld specimen; *Diaphorina*, 148
specimens, 29 species). They compare Descriptron's character matrix with BioCLIP, BioCLIP 2 and the CLIBD image
encoder (BIOSCAN-5M) under the same hold-outs.

| script | what it does |
|---|---|
| `extract_foundation_embeddings_v1.py` | embeds every image with one model (`--model bioclip / bioclip2 / clibd5m`), averages per specimen and view; cutouts with transparency are composited on white (`--alpha_background`) |
| `cosine_knn_check_v1.py` | scores the embeddings by cosine similarity (species prototypes and nearest specimen) under the same nested hold-outs; the matrix must reproduce the comparison's own result (control) |
| `benchmark_statistics_v1.py` | species-cluster bootstrap intervals, Holm-corrected paired tests, specimen-level rates, logistic-regression baseline |
| `traits_to_matrix_dir_v1.py` | turns per-specimen trait tables (any taxon) into the matrix folder the hold-out tests read |
| `museum_cue_check_v1.py` | does an instrument see the collection (photographic set-up) rather than the animal? Within-species nearest-neighbour test with permutations |

The matrix side is scored by `../../biorag_congruence_compare_v1.py --extra_continuous NAME=<model>.tsv`.

## Environment

The embedding script needs PyTorch, `open_clip_torch`, `timm` and `huggingface_hub` (Python 3.10+); the other
scripts run in the standard Descriptron environment. For example:

    python -m venv openset && . openset/bin/activate
    pip install torch torchvision open_clip_torch timm huggingface_hub pandas pillow

## Fair use of foundation models in an open-set test

A withheld species must be unseen by every instrument. Check the models' training catalogues (TreeOfLife-10M,
TreeOfLife-200M, BIOSCAN-5M) for your species and specimens first: a species the model was trained on is not new to
it, whatever the hold-out says. For mixed-collection photographs, run `museum_cue_check_v1.py`: colour, texture and
image embeddings can all identify the collection, and single-collection replicates are then needed.
