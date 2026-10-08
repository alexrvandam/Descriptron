# Descriptron v2 with BioRAG - 
## Production use ready - installable with pip, Docker or MCP (eg. Claude-Code)
 
<p>
  <a href="https://doi.org/10.5281/zenodo.22958109"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22958109.svg" alt="DOI 10.5281/zenodo.22958109" align="middle"></a>
  &nbsp;<img src="docs/marmot_icon.png" alt="Descriptron marmot" height="40" align="middle">
  &nbsp;<a href="#mcp-server"><img src="docs/claude_code_crab.png" alt="Works with Claude Code (MCP server)" height="40" align="middle"></a>
</p>

**Morphology-driven species descriptions, keys and delimitation for dark taxa.**

Descriptron takes photographs or micro-CT slices of specimens and produces the
things a taxonomist actually needs: measured structures, a character matrix, a
dichotomous key, formatted species treatments, and a record of how every number
was obtained. Annotation is model-assisted; the analysis is deterministic; and
every claim in the output is checked against the data it came from.

> **v2 is a rewrite of everything downstream of annotation.** v1 produced
> descriptions by handing measurements to GPT-4o. v2 replaces that with an
> evidence-tiered pipeline (**BioRAG**) in which the key, the matrix, the
> delimitation and the audits are computed, and a language model is used only to
> write prose from numbers it is not allowed to invent. The model's context is
> retrieved first and foremost from **the measured data matrix**, which is the
> primary retrieval and the basis of every analysis; **the literature**
> (BioSysLit and your own PDFs) is a secondary, optional source.

<p align="center">
  <img src="docs/weevil_instance_segmentation.jpg" alt="A weevil in lateral view with every structure automatically segmented as a labelled, coloured mask" width="640">
  <br>
  <sub>Automated instance segmentation of a weevil in lateral view: each structure is a labelled mask with the model's confidence.</sub>
</p>

---

## Contents

- [The approach at a glance](#the-approach-at-a-glance)
- [What v2 adds](#what-v2-adds)
- [Installing](#installing) — pip · Docker · conda
  - **[Docker how-to](docs/DOCKER_RECIPE.md)** — step-by-step recipe: project folder, full pipeline, Linux and Windows
  - **[Shared VM hosting recipe](docs/SHARED_VM_RECIPE.md)** — for IT staff: one institutional VM, shared GPU, many users
- **[Tutorial: the GUI tab by tab](docs/TUTORIAL.md)** — annotation, prediction, measurement, shape statistics, utilities
- [Automated trait collection: measurements, shape, colour and texture](#automated-trait-collection-measurements-shape-colour-and-texture)
  - [Validation at a glance](#validation-at-a-glance) · [Checked against geomorph](#checked-against-geomorph) · [Colour and texture checked](#colour-and-texture-checked-against-patternize-colormesh-and-scikit-image) · [Traits on a phylogeny](#traits-on-a-phylogeny-checked-against-geomorph-ape-and-phytools-from-220) · [Do the species differ?](#do-the-species-differ-and-in-which-traits-from-230) · [An alternative to geomorph for 2D images](#an-alternative-to-geomorph-for-2d-images) · [Examples](#what-it-produces-examples)
- [The workflow](#the-workflow)
  - **[SAM2-PAL & DINOLand annotation SOP](docs/SAM2PAL_DINOLand_Annotation_SOP.md)** — imaging, references, recipes and the orientation/mirror options
- [What BioRAG retrieves: the data matrix and the literature](#what-biorag-retrieves-the-data-matrix-and-the-literature)
- [What the programs produce](#what-the-programs-produce)
- [Reproducing an analysis](#reproducing-an-analysis)
- [Provenance and auditing](#provenance-and-auditing)
- [Use Descriptron from Claude (MCP server)](#mcp-server)
- [Licence](#licence)
- [Citation](#citation)

---

## The approach at a glance

From specimen images to audited species treatments: (A) a few specimens are annotated and the rest are
predicted and corrected; (B) every structure is measured, its shape, colour and texture recorded, and the
features assigned to evidence tiers in a checkable matrix; (C) the key is computed from the matrix, and a
language model writes the treatments, which an independent audit checks against the matrix; (D) which
descriptive words can be trusted, and whether an unseen species can be recognised, are tested on the same
specimens. The numbers are from the 29-species *Diaphorina* run.

```
  images ──► ANNOTATE ──► PROPAGATE ──► MEASURE ──► SCREEN ──► MATRIX
                GUI        SAM2-PAL      mm, GPA,    outliers,   evidence
             SAM2-assisted  DINOLand     colour,     flips,      tiers
                                         texture     conflicts      │
                                                                    ▼
   treatments ◄── AUDIT ◄── KEY ◄── DELIMIT ◄──────────────────── characters
    .docx, XML,  numeric   couplets  matrix / key / graph,
    DwC-A, SDD   + words   + support  calibrated on your own set
```

![The Descriptron and BioRAG workflow](docs/tutorial/showcase_biorag_workflow.jpg)

---

## What v2 adds

| | v1 | v2 |
|---|---|---|
| descriptions | GPT-4o from a feature table | **BioRAG**: evidence tiers, numbers checked against the matrix, every sentence attributable |
| key | — | computed from the matrix, leave-one-specimen-out validated, jackknife couplet support |
| delimitation | — | matrix / key / graph instruments, calibrated on your own reference set |
| mask propagation | Detectron2 only | **SAM2-PAL** palindrome propagation, **DINOLand** DINOv3 landmark transfer, torchvision detectors |
| checking | — | independent confabulation audit, subjective-word checks, ontology annotation |
| install | eight conda environments | **`pip install descriptron`**, or one Docker image |
| shape and form (2.1) | GPA, PCA, MANOVA/CVA | + Procrustes ANOVA with metadata (species x locality...), trajectories, disparity, allometry slopes, 2B-PLS, Mantel, asymmetry, modularity, Kmult, assignment; mirror images reflected automatically; the Procrustes statistics checked against geomorph and RRPP (Mantel and assignment by known-answer tests) |
| many small structures (2.5) | — | **SAM 3** (optional; in the Docker image, `pip install descriptron-sam3`): outline one or a few setae with a box or clicks and it proposes every similar instance on that image, shown together and labelled in one step; drag and drop of images and folders onto the GUI |
| more tools (2.1) | — | centre lines of thin structures, joint angles and posture standardisation, pose skeletons, video tracking with pose (DeepLabCut export), TPS / MorphoJ / StereoMorph / VIA converters, Darwin Core metadata import |

Nothing in the analysis half needs a GPU, and the key, matrix, audits and
delimitation need no model at all.

---

## Automated trait collection: measurements, shape, colour and texture

Descriptron turns specimen photographs into a table of traits, specimen by specimen, with every step
automated after a few annotated examples:

1. **Annotate a few, predict the rest.** Structures are outlined with SAM2 and carried to the rest of the
   collection by SAM2-PAL; landmarks are carried by DINOLand. You check and correct, not draw.
2. **Morphometrics.** Lengths, widths, areas and ratios in millimetres, with scale bars read from the images;
   curved length, width and curvature of thin structures along their centre line (within 0.3 % of the true
   length on test shapes, where a straight measurement under-reads curved structures by 17–23 %); joint angles
   from a pose skeleton.
3. **Geometric morphometrics.** Landmark GPA and outline semilandmarks (fixed, or slid by Procrustes distance
   or bending energy), mirror-image specimens found and reflected, shape PCA, and shape statistics driven by your
   specimen metadata: Procrustes ANOVA with permutation (RRPP), allometry, trajectories, disparity,
   shape vs environment, Mantel, asymmetry, modularity and integration, phylogenetic signal, assignment of
   unidentified specimens.
4. **Colour, colour pattern and texture.** CIE L\*a\*b\* and HSV colour with optional background-based correction of
   the illumination cast, colour classes and markings, and colour and texture measured in a grid of
   **homologous cells**: after the superimposition the semilandmarks are frozen in their settled
   correspondence and mapped back onto each photograph, so the same cell covers the same anatomy on every
   specimen.
5. **One table.** Everything is joined per specimen, calibrated to millimetres, and feeds the character
   matrix, the key and the descriptions, or your own analyses.

**How this is checked.** Every number in the geometric-morphometric part was compared with R on the same data:
with geomorph from raw landmarks (below), and with RRPP on identical aligned data. The comparison scripts are
in `descriptron/measure/validation/` and can be rerun by anyone with R. The checks found real problems, which
2.1 fixes and which are described below rather than hidden. The colour and texture values themselves are
standard measures (CIE L\*a\*b\*, GLCM, LBP); what is checked is that they are taken from homologous places
(panel g of the third figure).

### Validation at a glance

Each component was run against an established alternative on the same *Diaphorina* data, and every number in the
figure is computed from the benchmark outputs by `descriptron/measure/validation/validation_summary_figure.py`:

- **Segmentation:** SAM2-PAL against SST, the method it builds on (mean IoU 0.68 against 0.59 on forewings, 0.82 against
  0.58–0.63 on rostra).
- **Landmarks:** DINOLand against a trained Keypoint R-CNN on wings as photographed: with one to five labelled
  wings DINOLand lost 1 of 20 test wings and the detector 6 to 8; with many labelled wings the detector is more precise.
- **Colour pattern:** Descriptron identifies species as well as patternize and better than Colormesh (a paired test
  finds no difference from patternize), and all three detect the same species differences in pattern beyond overall
  colour.
- **Texture:** the co-occurrence matrices are identical to scikit-image's.
- **Shape statistics:** agreement with geomorph to at most 5.6 × 10⁻⁴, with size, landmark shape and outline
  semilandmarks plotted value against value.

![Descriptron validated against established tools](docs/tutorial/validation_summary.png)

### Checked against geomorph

Descriptron's geometric morphometrics are written in Python, so we ran the same data through R's
**geomorph** (4.1.1) and compared every number. Raw landmarks go into both, and each side does its own
Procrustes superimposition. Centroid sizes, Procrustes distances, Procrustes ANOVA (SS, R², F),
disparity, modularity (CR), two-block PLS, phylogenetic signal (Kmult) and asymmetry agree to at most
5.6 × 10⁻⁴ relative difference, most to 10⁻⁶ or better; permutation P-values differ only by random sampling.
Real data: 84 Diaphorina forewings plus synthetic designs with known effects.

![Descriptron vs geomorph, same raw landmarks](docs/tutorial/validation_python_vs_geomorph.png)

Outline semilandmarks agree with `gpagen` on the same points, fixed (the default) and with both of geomorph's
sliding methods, new in 2.1 (`--slide_method procd`: minimum Procrustes distance; `bending`: minimum bending
energy) to 5 × 10⁻¹¹. After the superimposition the semilandmarks are frozen in their settled correspondence
and mapped back onto each photograph, so colour and texture are measured in the same anatomical cell of every
specimen, something geomorph does not do.

![Descriptron outline semilandmarks vs geomorph](docs/tutorial/validation_semilandmarks_vs_geomorph.png)

The pipeline's other geometric-morphometric steps get the same check: landmark GPA, the outline shape PCA
(now on the covariance of the Procrustes coordinates, as geomorph's `gm.prcomp`; earlier versions standardised
every coordinate first) and the homology frames all agree with geomorph, and the colour and texture measured
in homologous cells land on the same anatomy for mirror-image specimens once they are reflected.

![Descriptron pipeline steps vs geomorph, and the mirror-image fix](docs/tutorial/validation_pipeline_gm_vs_geomorph.png)

The checks found, and 2.1 fixes, one problem worth knowing about: **mirror-image specimens** (a wing
photographed from the other side) cannot be superimposed without reflection, and dominated the first
principal component (19 of 96 Diaphorina forewings: PC1 95.7 % -> 30.0 % once reflected). Landmark and
outline GPA now find the mirror images, reflect them before alignment and list them; the colour pattern of the
mirror-image wings' homologous cells then matches the other wings again (r = 0.69, was 0.36). The comparison scripts
are in `descriptron/measure/validation/` (R and geomorph are needed only to rerun them); a comparison with
RRPP on identical aligned data is in the [tutorial](docs/TUTORIAL.md#metadata-and-shape-statistics).

### Colour and texture checked against patternize, Colormesh and scikit-image

**Colour pattern.** On the same 48 Diaphorina forewings (9 species), Descriptron's colour features from 52
homologous grid cells were compared with the R packages **patternize** 0.0.5 (landmark + k-means route) and
**Colormesh** 2.1 (landmarks + outline semilandmarks, triangulated sampling). Scored by leave-one-out
nearest-neighbour species accuracy on PC1-5 (chance about 0.11), Descriptron reached 0.83, patternize 0.85
and Colormesh 0.75 (±0.12 with 48 wings, so Descriptron and patternize do not differ); on all PCs 0.92, 0.71
and 0.52. The standardised mean wing colour alone (three numbers) reached 0.81, because these species differ
strongly in overall colour, so identification accuracy cannot show a gain from pattern. The species do differ in
pattern as well: with each wing's mean colour regressed out, species still explained 47%, 48% and 50% of the
colour-pattern variation of Descriptron, patternize and Colormesh (PERMANOVA on PC1-10, 9,999 permutations,
p = 0.0001 each), so all three methods detect the same pattern differences. Absolute colour values follow overall brightness (illumination, slide clearing);
Descriptron's colour relative to the wing's own mean removes it.

![Descriptron colour features vs patternize and Colormesh](docs/tutorial/validation_colour_vs_patternize_colormesh.png)

**Texture.** Descriptron computes its grey-level co-occurrence matrix (GLCM) itself, because it counts only
pixel pairs with both pixels inside the mask. Checked against scikit-image's `graycomatrix` on 180 masked
matrices from 15 wing images, every entry is identical (difference 0). Contrast and homogeneity use the
grey-level invariant formulas of Löfstedt et al. (2019, *PLOS ONE* 14: e0212110): quantising the same image
to 32 instead of 8 grey levels changes them by a factor of 1.15 and 1.00 (standard formulas 18.4 and 0.80).
Invariant energy is not invariant on real images (factor 6.99). Descriptron always quantises every cell to 16
grey levels, so energy is comparable among all images analysed with Descriptron, from the same or different
sources, but not with energy computed at other quantisations. Texture, like any fixed-pixel-distance measure,
depends on image scale.

![Descriptron texture features vs scikit-image, and grey-level invariance](docs/tutorial/validation_texture_glcm.png)

The scripts (`colour_benchmark_v1.py` with `colour_benchmark_patternize.R` and `colour_benchmark_colormesh.R`,
`colour_species_structure_v1.py`, `validate_texture_glcm.py`, `validation_colour_texture_figures.py`) are in
`descriptron/measure/validation/`.

### Traits on a phylogeny: checked against geomorph, ape and phytools (from 2.2.0)

`descriptron descriptron_phylo` takes a Newick tree and any continuous trait tables (landmark or
outline shape, measurements, colour pattern, texture) and computes, on species means: multivariate phylogenetic
signal Kmult (Adams 2014; Blomberg's K for one variable) with a permutation test; phylogenetic regression (PGLS) on
a predictor with residual randomisation, as geomorph's `procD.pgls`; ancestral states by generalised least squares;
and the phylomorphospace. Species missing from the tree, and tips without data, are dropped and the tree is pruned.

    descriptron descriptron_phylo --tree tree.nwk --species_col species \
        --traits shape=shape.csv --traits colour=colour.csv --scale colour=standardize \
        --size_col centroid_size --pgls "shape~logCS" --analyses signal pgls morphospace --out_dir phylo/

- **Landmarks on a published tree** (geomorph's `plethspecies`, 9 *Plethodon* species): tree covariance, Kmult,
  PGLS and ancestral states agree with geomorph and ape to 5 × 10⁻⁸ or better.
- **A published result from the authors' data** (Waldron et al. 2025, *Evolution* 79: 2369; 57 *Plethodon*
  species): Blomberg's K reproduced exactly for the three traits without imputed values, and identical to
  phytools and geomorph for all six.
- **Colour pattern and texture from photographs** (43 moth species of Noctuidae and Erebidae on the published tree
  of Nokelainen et al. 2024, *Nature Communications* 15: 1678; 423 museum specimens from GBIF, segmented with
  Florence-2 + SAM2): colour pattern Kmult 0.66 and texture 0.69 (P = 0.001), unchanged when only two museums'
  photographs are used; every statistic agrees with geomorph and phytools to 1.4 × 10⁻¹² or better; saturation
  ranks the species as the published values do (ρ = 0.84).

| Trait set (moths) | Kmult, 43 spp. | P | Kmult, two museums, 32 spp. | P |
|---|---|---|---|---|
| Outline shape | 0.72 | 0.001 | 0.71 | 0.016 |
| Proportions | 0.65 | 0.028 | 0.70 | 0.111 |
| Colour pattern | 0.66 | 0.001 | 0.73 | 0.011 |
| Texture | 0.69 | 0.001 | 0.73 | 0.001 |

![Colour pattern, texture and shape on the moth phylogeny](docs/tutorial/validation_phylo_moths.png)

![Phylogenetic analyses vs geomorph, ape, phytools and published values (Plethodon)](docs/tutorial/validation_phylo_plethodon.png)

Moth photographs: GBIF, CC BY 4.0 or CC0 (credits per image in the validation folder). The scripts
(`validate_phylo_real_tree.py`, `validate_phylo_traits.py`, `validate_phylo_published_waldron2025.py`, and
`moths_nokelainen2024/` for the whole moth run) are in `descriptron/measure/validation/`.

### Do the species differ, and in which traits? (from 2.3.0)

`descriptron descriptron_trait_stats` runs, for every trait set (colour pattern, texture, outline or landmark shape,
measurements, descriptive categorical characters) against the species labels:
- PCA with each species' convex hull;
- PERMANOVA of species, also after removing each specimen's mean colour ("does the pattern differ beyond overall
  colour?") and log size;
- pairwise PERMANOVA for every species pair (Benjamini–Hochberg): which species each trait set separates, and which
  pairs no trait set separates;
- leave-one-out identification per trait set and per species, with Wilson 95% intervals, and exact McNemar tests
  between trait sets;
- allometry;
- Cramér's V for each descriptive character.

The pipeline runs it for every structure (step `trait_stats`, and `trait_stats_states` for the descriptive
characters), and with `--tree` also `descriptron_phylo` (step `phylo`). These are extra results beside the
descriptions and never change or stop them. In the GUI: Measure tab → STATS → **Morphological statistics**.

It reproduces the colour benchmark above exactly (species R² 0.606, 0.473 beyond mean colour; 40 and 44 of 48 wings
identified), and every statistic matches R to machine precision (`validate_trait_stats.py`: vegan `adonis2`,
`class::knn.cv`, `prop.test`, `binom.test`, `chisq.test`).

On the forewings of 13 *Diaphorina* species (69 wings), colour pattern and texture each identified 84% of wings,
outline shape 59% and measurements 48%; species still explain 45% of colour-pattern variation after each wing's mean
colour is removed.

    descriptron descriptron_trait_stats --groups group_labels.csv --out_dir stats/ \
        --traits "colour pattern=color_homology/forewing/color_homology_features_forewing_combined_hw1.csv" \
        --traits "outline shape=semilandmarks/forewing/aligned_coco.json" --scale "outline shape=none" \
        --categorical "states=descriptive_states.csv" --size measurements/all_metrics.csv:area_mm2:forewing

![Forewing colour pattern of 13 Diaphorina species, with species hulls and PERMANOVA](docs/tutorial/trait_stats_diaphorina_forewing_colour_pca.png)

![Species identified by each trait set, Diaphorina forewings](docs/tutorial/trait_stats_diaphorina_forewing_identification.png)

### Which characters, which specimens, and does DNA agree? (from 2.6.0)

Three more extra results of the pipeline (they never change or stop the descriptions):
- **What one character is worth** (step `char_signal`, `descriptron_character_signal_v1`). For every character of
  the matrix: how often it alone names a withheld specimen, whether it tells a described species from an unseen one,
  how strongly it separates species (Kruskal–Wallis η², FDR) and, with `--tree`, its phylogenetic signal (Blomberg's
  K). It writes the 25 best characters of each structure as a table, the character-robustness figure, and a
  companion figure of phylogenetic signal in the same colours. Characters that identify species well but carry no
  phylogenetic signal behave like autapomorphies (useful in keys); conserved characters place a species among its
  relatives.
- **Specimens to re-examine** (step `reexamine`, `descriptron_reexamine_v1`). After calibration, the specimens the
  character matrix names as another species (identity), and those with a flagged outline or a value at least twice
  or at most half the species median (annotation). It also writes a report and a zoomable map with support values.
- **A phylogenetic summary figure** (end of step `phylo`): the species tree, the two most informative colour-pattern
  phylomorphospaces (chosen by how well colour pattern identifies species, never by the phylogenetic result) and the
  Kmult of every structure and trait set.

With DNA barcodes, `descriptron dna_vs_morphology_v1` gives each morphospecies' barcode gap, DNA groups against the
morphospecies, and a DNA verdict beside the matrix's name for every specimen. `descriptron coi_species_tree_v1` turns
the gene tree into a species tree for `--tree`; pass the DNA results with `--dna_dir` to add them to the summary
figure. Practical rule: start from expert morphospecies, use DNA where available as outside evidence; species both
support are very likely real, and the conflicts show where to look again.

    descriptron run_full_pipeline_v2 ... --tree species_tree.nwk --dna_dir dna/

### Annotations stay on their own image; labelled plates (from 2.7.0)

- **Check older annotation files.** Before GUI v81, *Load Annotations* drew every annotation in a COCO file on the
  image on screen and saved it under that image, so a file loaded while another image was showing can hold an
  outline filed under an image it was never drawn on. `descriptron descriptron_check_cross_image_copies_v1 --coco
  file.json` finds such copies (same position and the same outline points; independent tracings of similar
  structures share only a few per cent of their points), reports them and changes nothing; `--drop_image` or
  `--drop_ids` writes a cleaned copy beside the original. GUI v81 loads only the annotations of the image on screen,
  and *Load Folder* keeps each image's masks when you move on and come back.
- **Labelled species plates** (step 18, `descriptron_labelled_plates_v1`): beside the overlay plates, the specimens in
  their own colours with every structure named by a leader line that ends on its outline (or, with
  `--plate_anchor centre` / the GUI checkbox, in its middle).

### Descriptive characters per structure (from 2.7.5)

After a structure is labelled in the GUI (v82), its descriptive characters can be recorded with it: texture,
sculpture, setae, vestiture, colour, colour pattern, lustre and the other characters of the Descriptron-GBIF
Annotator, plus counts (number of setae, number of punctures) and characters you add yourself. They are saved
with the mask (`annotations[].attributes`, the field the GBIF annotator uses) and, with
`descriptron run_full_pipeline_v2 ... --coded_states_coco annotations.json` (or the checkbox in the BioRAG window),
become characters of the matrix: the key states them in words ("elytron: setae dense"), the data sheet gives them
in words for the Diagnosis and Description, and the audit checks every state the text names against what was
recorded for that species. *Utilities > Rename/delete categories* fixes a misspelt category or merges two, and the
mouse wheel zooms the image.

### Phylogeny, habitat or both? Community phylogenetics (from 2.7.6)

`descriptron descriptron_community_v1` (GUI: *Measure & analyse > Community & phylogeny*) takes a Darwin Core
specimen table (site, treatment such as forest type, image names), the colour and texture tables, and a species
tree (one exemplar per species, pruned per site from the metadata, or one sequence per species per site with a tip
map). For colour, colour pattern (split automatically from the colour table) and texture it reports how much
variation the treatment explains alone, phylogeny alone, both together, or neither (variance partitioning with
phylogenetic eigenvectors); whether species found under both treatments shift; phylogenetic signal per treatment;
and, with sites as replicates, each site's phylogenetic diversity, MPD and MNTD with null-model effect sizes and
its trait dispersion. Phylomorphospaces are drawn with dots and with specimen thumbnails. Checked against R (vegan,
ape). A worked example with Darwin Core metadata is in `descriptron/measure/examples/community_example/`. Image
every site the same way: a camera or lighting difference shows up as a shift of absolute colour and texture in
every species, which the paired test detects (colour pattern is much less affected).

![Community phylogenetics on GBIF moth photographs: variance partitioning, a phylomorphospace with specimen thumbnails, and every number against R](docs/tutorial/showcase_community_phylogenetics.jpg)

*(a)* Swiss (ETH Zurich) against Estonian (Tartu) moths, 31 species: the collection shifts absolute colour and
texture in every species (photography) but hardly colour pattern, which follows phylogeny. *(b)* Colour-pattern
phylomorphospace of 19 Swiss species north and south of the Alps, each point drawn as the specimen nearest its
mean: phylogeny explains 0.26, the side of the Alps nothing. *(c)* The same run given to R. To check your own run
(from 2.7.7; needs R with vegan, ape and phytools), add `--save_matrices` and then run
`descriptron validate_community_run_vs_r --run_dir <out_dir>` (pip) or `python descriptron/measure/validation/validate_community_run_vs_r.py --run_dir <out_dir>` (GitHub), which writes
`community_vs_R.tsv` and panel (c). On the moth run:

| Quantity | R reference | Comparisons | Largest difference |
|---|---|---|---|
| Variance-partition fractions [a] [b] [c] [d] | `vegan::varpart` | 12 | 4.2e-16 |
| Partial F (treatment given phylogeny and vice versa) | `vegan::rda` + `anova` | 6 | 1.2e-14 |
| Faith's PD (with root), MPD, MNTD per site | `ape` (`keep.tip`, `cophenetic`) | 47 | 3.4e-13 |
| Phylomorphospace ancestral states, PC1 and PC2 | `phytools::fastAnc` | 108 | 4.8e-14 |

The site-level metrics in this museum example are shown only to check the arithmetic: the specimens are
museum records, not community samples, and the program warns when sites look like that.

### An alternative to geomorph for 2D images

For two-dimensional landmark and outline data, Descriptron now covers most of what taxonomists and
morphologists use geomorph (and tpsDig, MorphoJ or StereoMorph, whose files it reads and writes) for, gives
the same numbers, and adds the steps those programs leave to you: collecting the landmarks and outlines,
calibrating to millimetres, finding mirror-image specimens, and measuring colour and texture in homologous
regions. You can stay in one Python workflow from photograph to statistics.

| | geomorph | Descriptron | checked |
|---|---|---|---|
| GPA, landmarks | `gpagen` | `landmark_gpa_V2`, `descriptron_shape_stats` | same numbers |
| outline semilandmarks, fixed or sliding (Procrustes distance, bending energy) | `gpagen(curves=)` | V42 `--slide_method` | same numbers |
| shape PCA | `gm.prcomp` | V42, landmark GPA | same numbers |
| Procrustes ANOVA, allometry | `procD.lm` | `anova`, `allometry` | same numbers |
| trajectory analysis | `trajectory.analysis` (RRPP) | `trajectory` | same numbers (RRPP) |
| disparity | `morphol.disparity` | `disparity` | same numbers |
| 2B-PLS, modularity (CR) | `two.b.pls`, `modularity.test` | `pls`, `modularity` | same numbers |
| phylogenetic signal | `physignal` | `phylosignal` | same numbers |
| object symmetry | `bilat.symmetry` | `asymmetry` | same numbers (Descriptron reports half the left-right difference) |
| posture standardisation | `fixed.angle` | `descriptron_joints standardise` | known-answer tests |
| Mantel, assignment with typicality | — (vegan, MASS) | `mantel`, `assign` | known-answer tests |
| collection, mm calibration, mirror images, colour and texture in homologous cells | — | built in | see above |
| phylogenetic signal of any trait set (shape, measurements, colour pattern, texture) | `physignal` | `descriptron_phylo` (2.2.0) | same numbers (geomorph, phytools) |
| phylogenetic GLS | `procD.pgls` | `descriptron_phylo --pgls` (2.2.0) | same numbers |
| ancestral states, phylomorphospace | `gm.prcomp(phy =)` | `descriptron_phylo` (2.2.0) | same numbers (geomorph, phytools `fastAnc`) |
| PERMANOVA of species, pairwise species separation, leave-one-out identification, any trait set | — (vegan `adonis2`, `class::knn.cv`) | `descriptron_trait_stats` (2.3.0) | same numbers |
| **not (yet) in Descriptron:** 3D landmarks and surface semilandmarks, evolutionary rate comparisons (`compare.evol.rates`) | yes | no | — |

For these, and for any analysis we have not listed, use geomorph: Descriptron's converters write TPS and
MorphoJ files, and its aligned coordinates are plain tables.

### How the predictions check themselves

Both propagation tools check their own work by going there and back.

**SAM2-PAL** treats the annotated template and each new specimen as a short video that runs forwards and then
backwards: template, specimen, specimen again, template (a "palindrome"). The masks are carried from the template
to the specimen and then back; the memory is reset halfway, so the model cannot simply remember the template's
masks. During fine-tuning, the difference between the masks that come back and the template's own masks is the
training signal, so the model learns from specimens nobody has annotated. During prediction, the same round trip
is scored for every specimen. It was high almost everywhere in our runs (mean overlap 0.98 on 317 ant heads), so
it catches gross failures; every predicted mask also carries the model's own confidence, and a person checks them.

**DINOLand** matches each landmark from a reference specimen to the new one using DINOv3 features (a vision
transformer whose features give the same colour to the same structure on different specimens, below), searching
only near where an affine alignment of the two outlines says the landmark should be. The match is then sent back:
if the point found on the new specimen does not lead back to within 1.5 feature patches of where it started, the
landmark is rejected. Matches that survive are cleaned of geometric outliers (RANSAC) and refined below patch
size; with several reference specimens, a landmark counts as agreed (green in the examples) when the references
place it in the same spot, and is marked for checking (gold) when they do not.

![DINOv3 features give the same colour to the same structure on two ant heads, with a homology grid](docs/tutorial/showcase_dinov3_correspondence.jpg)

<sub>The homology grid is a thin-plate spline from the consensus shape (geometric morphometrics); the colours are the
first three principal components of DINOv3's patch features, computed jointly for both ants, so the same colour
marks the same structure.</sub>

**Detectron2** (Meta's Mask R-CNN) is the third route. Once enough images have been annotated and corrected, it
is trained on them and then segments every structure of new images in one pass, each mask with a confidence
score; Descriptron's training keeps a held-out validation set, split by specimen. The weevil at the top of this
page was segmented this way.

### What it produces: examples

Assembled from the programs' output (panel titles added; the animation is drawn from SAM2-PAL's
predictions file).

**Annotate one, predict the rest, correct what needs it.** Step 0: one ant head annotated by hand (18
structures). Step 1: SAM2-PAL carries them to the other heads (examples drawn at random from 317 colour
photographs). Step 2: on hard plates, with the head small in a wide frame, some structures land in the wrong
place; they are corrected in the GUI, and the corrected heads can be used to fine-tune SAM2-PAL for the next
round. The last two frames show two such plates as predicted and after correction.

![SAM2-PAL predictions on ant heads](docs/tutorial/showcase_sam2pal_heads.gif)

**Video, too.** Load a video (here a rotating micro-CT render of an ant head), mark a structure on a few
frames, and SAM2 follows it through every frame; the Video Remote Control steps through the frames, shows
which are annotated and lets you delete bad ones before fine-tuning (clip at 2x speed, earlier GUI layout).
Videos are of 3D Ant Models from Francisco Hita Garcia (OIST) 

![SAM2 video mode in the GUI](docs/tutorial/showcase_sam2pal_video_gui.gif)

The same region followed through the full rotation of four heads with the fine-tuned model (about 3x speed):

![SAM2-PAL on four rotating ant heads](docs/tutorial/showcase_sam2pal_video_4heads.gif)

<sub>3D models: Francisco Hita Garcia (OIST) (Sketchfab).</sub>

**Landmarks, placed automatically.** DINOLand carries 17 vein-junction landmarks from hand-landmarked
reference wings to new psyllid forewings, whichever way up and whichever side they were photographed from
(`--orientation_search rot4 --mirror_refs`). Each landmark is marked as agreed by the references (filled green)
or to be checked (open gold), so you look only where it is needed. On 80 test wings, turned copies and mirror
images included, the median error was 2.5 % of wing size, with 76 % of landmarks within 5 % and 89 % within 10 %.
Psyllidae images are from Liliya Serbina (LIB Hamburg).

![DINOLand landmarks on psyllid forewings](docs/tutorial/showcase_dinoland.jpg)

How much work that saves: in the test run below, DINOLand placed 1,360 landmarks (20 wings, each in four
orientations) in about four minutes on a laptop without a GPU. By hand that is 1,360 careful clicks; with
DINOLand you look at the gold points and move the ones that are off.

![DINOLand on 20 psyllid forewings](docs/tutorial/showcase_dinoland_wall.jpg)

Once about 20 or more specimens are corrected and imaged the same way up, a trained keypoint detector is more
precise: Keypoint R-CNN (GUI *Predict* tab, *Train KP R-CNN* / *Predict KP R-CNN*, or `descriptron-train --task
keypoints` / `descriptron-predict --image_folder`). With one to five references on specimens photographed as they
come, DINOLand lost 1 of 20 test wings and the trained detector 6 to 8; see the
[SOP](docs/SAM2PAL_DINOLand_Annotation_SOP.md#when-to-switch-from-dinoland-to-a-trained-detector).

**Measurements** in millimetres, with the scale bar read from the image, and the centre line of a thin,
curved structure, whose curved length a straight measurement under-reads.

![Length, width and centre line of a pterostigma](docs/tutorial/showcase_measurements.jpg)

Measurements are not limited to masks. From your own keypoints (here 17 vein junctions of a psyllid forewing)
the measurement step gives every pairwise distance in millimetres, 136 of them, alongside the landmark
geometric morphometrics.

![Every pairwise distance between 17 forewing keypoints, in mm](docs/tutorial/showcase_keypoint_distances.jpg)

**Geometric morphometrics** of wing outlines: where along the outline the shape varies, and the specimens in
shape space.

![Semilandmark PC heat maps and shape space](docs/tutorial/showcase_shape.jpg)

**Colour and colour pattern**: shine removal and illumination normalisation, colour classes and the pattern
mask, then colour measured in homologous cells, so the same cell covers the same anatomy on every wing.

![Colour pipeline and homologous cells](docs/tutorial/showcase_colour.jpg)

**Texture**: ant heads placed by the texture of their head capsule, measured in homologous cells.

![Ant heads in texture space](docs/tutorial/showcase_texture_ant_heads.jpg)

**Statistics from your metadata.** Import a specimen table or a Darwin Core download, mark the columns, and one
command runs the analyses the metadata support. Below, 96 psyllid forewings: species explain 72 % of forewing
shape and sex nothing; size has a small effect with the same slope in every species; and the 12 wings of
species outside the reference set are all flagged as unlike every known species rather than forced into the
nearest one.

![Shape statistics from specimen metadata](docs/tutorial/showcase_stats.jpg)

### From traits to species descriptions (BioRAG)

The traits of every specimen become a character matrix, and from it a key, species descriptions and a check
on whether a specimen belongs to a known species at all ([the approach at a glance](#the-approach-at-a-glance)).
Results on 29 *Diaphorina* species (148 specimens):

**Naming a specimen, and recognising a species never seen.** Each specimen's own record is withheld before it is
named. The character matrix names 132 of 148 correctly, and of the three instruments it best recognises a
species it has never seen (A: every operating point, and the estimate when the threshold is chosen without the
species being scored).

![Species prediction: key, knowledge graph and character matrix](docs/tutorial/showcase_biorag_species_prediction.jpg)

**What one character is worth.** For each structure, how often its best single character alone points to the
right species for a withheld specimen, and every character's worth for naming against novelty. Single
characters rarely suffice; the matrix works because it combines many.

![Character robustness](docs/tutorial/showcase_biorag_character_robustness.jpg)

**Which characters reflect shared ancestry** (from 2.6.0, with `--tree`). The same characters, in the same colours,
against their phylogenetic signal on a species tree (Blomberg's K; filled: P ≤ 0.05; ringed and labelled: signal
and K > 1; label colour: naming at least 3× chance). In *Diaphorina* the forewing venation proportions are the most
conserved; the best single identifiers carry no detectable signal, as species-specific, autapomorphy-like
characters should.

![Phylogenetic signal of every character](docs/tutorial/showcase_biorag_character_phylo_signal.jpg)

**How far apart the species are, and which characters hold up.** Species placed by their distances in the
matrix (a), the gap between every pair (b), and the support for the characters the knowledge graph and the key
use, each tested on specimens withheld from it (c, d).

![Species distances and character support](docs/tutorial/showcase_biorag_species_distances.jpg)

**A species treatment, written from the data and audited.** A language model writes the sentences; every
number comes from the matrix and an independent audit re-derives each one (numbers, the structure they
belong to, and every comparison) before the treatment is accepted. Descriptive words about shape and surface
come from images read by the model and are reported with their measured repeatability. Excerpt, *Diaphorina*
sp. 'kenya', collected in Kenya (undescribed species carry working codes until they are named):

![Diaphorina sp. 'kenya': specimens and measured structures](docs/tutorial/showcase_treatment_plate_kenya.jpg)

> **Diagnosis.** Diaphorina sp. 'kenya' differs from most congeners in characters of the distal aedeagal segment, male proctiger and male subgenital plate. Distal aedeagal segment length 0.138–0.163 mm, smaller than in all other species measured (19). Distal aedeagal segment colour L\* 51.0–57.0, darker (lower L\*) than in all other species measured (20) except D. cf. enderleini, D. sp. 1A and D. sp. 3; its palest third L\* 58.8–62.9, darker (lower L\*) than in all other species measured (20) except D. sp. 3. Male proctiger relatively long: MP/PL ratio 1.39–1.50×, greater than in all other species measured (22) except D. cf. carrisae and D. sp. 1A; MP/DL ratio 1.96–2.23×. Male subgenital plate width 0.064–0.193 mm, smaller than in all other species measured (21) except D. sp. 9, D. sp. 19 and D. sp. 21; male subgenital plate length/width 1.34–3.84×, greater than in all other species measured (22) except D. sp. 4, D. sp. 5, D. sp. 9, D. sp. 17, D. turneri and D. virgata. Head and forewing ochreous brown; metafemur orange-brown.
>
> **Head.** Head width (HW) 0.339–0.378 mm (mean 0.355; n=6), head length 0.556–0.617 mm (mean 0.580; n=6), head longer than wide (length/width 1.58–1.73×, mean 1.63; n=6). Vertex width 0.191–0.249 mm (mean 0.216; n=6), vertex length 0.357–0.397 mm (mean 0.372; n=6), distinctly longer than wide (VL/VW; see Proportions). Genal processes present, elongate in outline (length/width 1.21–2.28×, mean 1.91; n=6), length 0.164–0.317 mm (mean 0.279; n=6), width 0.125–0.166 mm (mean 0.146; n=6); their length relative to vertex length, precise shape and apex form not assessable from the material examined. Head ochreous brown (L\* 57.3–64.1, a\* 8.0–10.0, b\* 34.3–38.5); vertex slightly paler, ochreous yellow (L\* 64.5–68.1, a\* 5.7–7.4, b\* 34.8–37.9); genal processes ochreous brown (L\* 52.1–64.4, a\* 8.2–13.4, b\* 37.9–41.1).
>
> **Metaleg.** Metafemur length (MF) 0.394–0.416 mm (mean 0.403; n=6), width 0.105–0.119 mm (mean 0.113; n=6), length/width 3.42–3.77× (mean 3.58; n=6), orange-brown (L\* 40.7–53.2, a\* 15.3–19.1, b\* 37.8–45.2), with a lightness contrast between its two end thirds (L\* difference 8.4–19.9). Metatibia length (MT) 0.562–0.588 mm (mean 0.574; n=6), width 0.0621–0.0682 mm (mean 0.0656; n=6), length/width 8.32–9.24× (mean 8.76; n=6), ochreous yellow (L\* 71.5–75.8, a\* 2.6–4.5, b\* 30.9–37.8). Metafemur shorter than metatibia (MF/MT; see Proportions); metatibia distinctly longer than head width (MT/HW; see Proportions). Metatarsus colour not assessable from the material examined.
>
> *(Excerpt, verbatim. The full treatment covers antenna, rostrum, forewing cells, male and female terminalia,
> proportions, sexual dimorphism, remarks and what could not be assessed from the material.)*

---

## Installing

### 1. pip — the normal route

**Step by step for Windows, macOS and Linux, from a computer with nothing installed:
[docs/INSTALL_PIP.md](docs/INSTALL_PIP.md).** In short, in one Python 3.12 environment:

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128   # NVIDIA; Mac/CPU: drop --index-url
SAM2_BUILD_CUDA=0 pip install git+https://github.com/facebookresearch/sam2.git   # SAM2 library (Windows: set SAM2_BUILD_CUDA=0 first)
pip install descriptron          # analysis programs, detectors and the GUI
descriptron-gui                  # first start downloads the SAM2 model file (898 MB) once
descriptron --list               # the analysis programs
```

To update an existing install: `pip install --upgrade descriptron` (SAM 3, in its own environment:
`pipx upgrade descriptron-sam3`; Docker: `docker pull ghcr.io/alexrvandam/descriptron:latest`).

**SAM2 is not on PyPI**, so `pip install descriptron` cannot bring it: install it from Meta's GitHub as above,
into the same environment. The **SAM2 model file** `sam2_hiera_large.pt` is downloaded by the GUI the first time
it starts, into `~/.cache/descriptron/sam2/` (Windows: `C:\Users\<you>\.cache\descriptron\sam2\`); if you
already have it, copy it there or set `DESCRIPTRON_SAM2_CHECKPOINT` to its path. The analysis half alone
(`pip install descriptron-core`) needs neither.

Or take only what you need:

| package | what it gives you | needs |
|---|---|---|
| `descriptron-core` | the whole analysis half — measurements, matrix, key, treatments, audits | **nothing but pip.** No torch, no CUDA, no compiler |
| `descriptron-vision` | mask and landmark prediction: torchvision detectors, SAM2-PAL, DINOLand | torch (ordinary wheels; Apple GPU via MPS on macOS) |
| `descriptron-gui` | the annotation GUI | core + vision + tkinter |
| `descriptron-sam3` (2.5.1) | SAM 3: find every instance of a small structure (setae...) from a box, clicks or a word; the GUI's SAM 3 buttons use it | **Python 3.12**, its own environment (below) |

**SAM 3 (optional).** It needs Python 3.12 and PyTorch >= 2.7, so it goes in its own environment; pipx makes that
environment and puts the `descriptron-sam3` command on your PATH, where the GUI finds it:

```bash
pip install pipx && pipx ensurepath          # once; open a new terminal afterwards
pipx install --python python3.12 descriptron-sam3
```

No pipx? `python3.12 -m venv ~/sam3env && ~/sam3env/bin/pip install descriptron-sam3`, then tell the GUI where it
is: `export DESCRIPTRON_SAM3_PYTHON=~/sam3env/bin/python`. The SAM 3 checkpoints are gated: request access at
https://huggingface.co/facebook/sam3, then log in once with the token from your Hugging Face settings
(`pipx run --spec huggingface_hub hf auth login`). The first run downloads 3.3 GB; a CUDA GPU is strongly
recommended.

**If you only want to reproduce a published analysis from deposited COCO files,
`pip install descriptron-core` is enough**, and it installs in about a minute on
Linux, macOS and Windows.

For an NVIDIA GPU, install torch from PyTorch's index first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install descriptron
```

Writing treatments is the only step that calls a language model:
`pip install "descriptron-core[llm]"`. **Keys** (an Anthropic API key for
descriptions with `--llm_backend api`; a Hugging Face token the first time DINOLand
downloads DINOv3) are read from the environment or from
`~/.config/descriptron/credentials`; if neither has one, you are asked the first
time it is needed, and it is saved there (`descriptron-keys --set NAME` to change). Literature retrieval from BioSysLit and
your own PDFs: `pip install "descriptron-core[rag]"` (see
[What BioRAG retrieves](#what-biorag-retrieves-the-data-matrix-and-the-literature)).

### 2. Docker — everything, including the parts pip cannot carry

    docker pull ghcr.io/alexrvandam/descriptron:2.5.1

SAM 3 is included (from 2.5.1): pass your Hugging Face token (`-e HF_TOKEN=hf_...`, after requesting access at
https://huggingface.co/facebook/sam3) and the GUI's SAM 3 buttons and the `sam3` command work; see
[docs/DOCKER_RECIPE.md](docs/DOCKER_RECIPE.md#7-sam-3-every-instance-of-a-small-structure).

Two dependencies are not on PyPI and can never be declared by a published
package: **SAM2** and **Detectron2**. The image carries both, already built.

**Step-by-step recipe** (project folder, the full pipeline, Linux and Windows
commands, where the results go): [docs/DOCKER_RECIPE.md](docs/DOCKER_RECIPE.md).

```bash
docker pull ghcr.io/alexrvandam/descriptron:latest

docker run --rm --gpus all -v "$PWD:/data" -v descriptron-weights:/weights \
  ghcr.io/alexrvandam/descriptron:latest train-d2 \
    --coco-json /data/annotations.json --img-dir /data/images \
    --output-dir /data/out --dataset-name mytaxon --total-iters 35000

docker run --rm ghcr.io/alexrvandam/descriptron:latest help
```

**GPU support by platform** — read this before planning around Docker:

| platform | GPU inside the container |
|---|---|
| Linux + NVIDIA | yes (`nvidia-container-toolkit`, `--gpus all`) |
| Windows + NVIDIA | yes (Docker Desktop, WSL2 backend) |
| **macOS** | **no — none, ever** |

Docker Desktop on macOS runs a Linux VM with no passthrough to the Apple GPU, and
Apple Silicon has no CUDA. **Mac users should install with pip and use the
torchvision backend**, which reaches the Apple GPU through PyTorch's MPS device.

`podman` works as a drop-in and has no licensing condition, which matters for
larger institutions.

### 3. conda — for development, or to match the published environment exactly

```bash
git clone https://github.com/alexrvandam/Descriptron.git && cd Descriptron
conda env create -f environments/measure_env_environment.yml     # analysis
conda env create -f environments/samm_environment.yml            # SAM2, SAM2-PAL, DINOLand
conda env create -f environments/detectron2_env_environment.yml  # Mask R-CNN
```

Detectron2 must then be built from source; see `docker/Dockerfile` for the exact
sequence, including the two flags it needs (`--no-build-isolation`, and a
non-editable install).

### Optional: SAM 3 for many small structures (from 2.5.0; one-step install from 2.5.1)

SAM 3 finds every instance of a structure (for example all setae on a wing or genitalia) from one or a few
examples. It needs Python 3.12 and PyTorch >= 2.7, so it is installed beside the rest rather than into it:

| route | how |
|---|---|
| **Docker** | already inside the image (its own `sam3` environment): the GUI buttons work, and `docker run --rm --gpus all -e HF_TOKEN=... -v "$PWD:/data" -v descriptron-weights:/weights ghcr.io/alexrvandam/descriptron:latest sam3 --images /data/wings --text bristle --out /data/setae.json` runs it headless |
| **pip** | `pipx install --python python3.12 descriptron-sam3` (its own environment, and the `descriptron-sam3` command the GUI looks for); or `pip install descriptron-sam3` in any Python 3.12 environment and set `DESCRIPTRON_SAM3_PYTHON` to that environment's python |
| **conda** | `bash environments/build_sam3_env.sh` (a conda env `sam3`, found by the GUI) |

The checkpoints are gated (SAM License): request access at https://huggingface.co/facebook/sam3, then log in once
(`hf auth login`) or pass your token (`-e HF_TOKEN=...` with Docker). The first run downloads 3.3 GB. In the GUI,
on the SAM2 row, draw a box around one example (several boxes = several examples) or click points, then press the
dark green **Find all instances (SAM 3)**. With clicked points you choose: outline only what you clicked (one mask
per point, outlined at full image resolution), use each click as an example and find all like them, or use all
clicks as one example. The instances appear together, one colour each; trash the false ones and Apply Label once
to label them all. Examples work within the image they are drawn on; a typed word (e.g. "bristle") searches the
image tile by tile. The grey **Find all instances (SAM 3)** button in the Predict tab runs the same tool on a folder.
Results are proposals to review, not finished annotations.

---

## The workflow

```
  images ──► ANNOTATE ──► PROPAGATE ──► MEASURE ──► SCREEN ──► MATRIX
                GUI        SAM2-PAL      mm, GPA,    outliers,   evidence
             SAM2-assisted  DINOLand     colour,     flips,      tiers
                                         texture     conflicts      │
                                                                    ▼
   treatments ◄── AUDIT ◄── KEY ◄── DELIMIT ◄──────────────────── characters
    .docx, XML,  numeric   couplets  matrix / key / graph,
    DwC-A, SDD   + words   + support  calibrated on your own set
```

Each stage is a set of command-line programs; the GUI is a front end that runs
them for you. Every step that calls a model takes the same flags
(`--llm-backend claude-code | api | none`, `--api-base-url`, `--api-key-env`), so
a subscription, an API key, another provider, or **no model at all** are
interchangeable.

### Image specimens in one orientation

SAM2-PAL and DINOLand transfer masks and landmarks from reference images, and
both work best when every specimen is photographed in the **same orientation as
the references**. On 14 held-out ant heads, SAM2-PAL's mean mask IoU was 0.78
upright, 0.35 upside-down and 0.14 turned 90°. Opt-in fallbacks exist for
collections where this cannot be controlled; all are **off by default**:

| option | tool | what it does | tested gain |
|---|---|---|---|
| `--orientation_search rot4` (GUI: *Orientation search*) | SAM2-PAL | predicts each image at 0/90/180/270°, keeps the most confident rotation (mean SAM2 object confidence) and maps the masks back; 4× slower | turned heads 0.14 → 0.78; right rotation chosen 56/56 |
| `--flip_augment all` (GUI: *Flip augmentation*) | SAM2-PAL training | adds h, v and hv flipped copies of the labelled images | upright heads 0.76 → 0.78; does **not** fix turned specimens |
| `--orientation_search rot4` (GUI: *specimens may be turned or upside-down*) | DINOLand (v52) | adds copies of every reference rotated 90/180/270° and keeps, per target, the orientation whose landmarks are accepted most often; little extra time | psyllid wings turned 90/180/270°: median error 27–43% → 3.1–3.2% of wing length, wings failing 48/60 → 3/60 |
| `--mirror_refs` (GUI: *specimens may be mirror images*) | DINOLand | adds a flipped copy of every reference; each target uses whichever handedness matches | mirrored psyllid wings failing 16/17 → 1–2/17 |
| both DINOLand options together | DINOLand (v52) | covers all 8 rotation/mirror combinations; use when you do not know how specimens were imaged; little extra time (80 wings: 3.6 → 3.9 min) | mixed mirrored + turned wings: failing 53/80 → 0/80, median error 2.5% |

Neither tool can tell which side is anatomically left on a bilaterally
symmetric structure photographed mirrored, so `left_`/`right_` names follow the
image as taken.

The full recipe — imaging, how many references to draw, commands, and how to
check the output — is in the **[SAM2-PAL & DINOLand annotation SOP](docs/SAM2PAL_DINOLand_Annotation_SOP.md)**.

---

## What BioRAG retrieves: the data matrix and the literature

BioRAG is retrieval-augmented generation: the model writes only from context
that is retrieved for it, never from its own memory. The **data matrix is the
primary retrieval** and by far the most important: the key, the delimitation,
the audits and every number in a treatment rest on it. The literature is a
secondary, optional source of terminology and context.

### 1. The data matrix (primary, always used)

For each species, BioRAG retrieves that species' measurements from the
character matrix: for every feature, the number of specimens, the minimum,
maximum and mean, and the range across all species beside it, grouped by
evidence tier. This evidence sheet is the only source of numbers the model may
use. Every number in a treatment must come from it, and the independent audit
(`biorag_confabulation_checker_v2`) checks each one against the specimen matrix
afterwards. This retrieval is what makes the numbers in a description accurate.

### 2. The literature (secondary, optional)

Before the model describes each structure from the images, BioRAG can also give it
passages from **published treatments of the group**: the terminology taxonomists
use, and the characters they have found informative. The retrieved text is
reference only. The prompt tells the model not to copy a character state unless
it is visible in the image, and **every number still comes from the measured
matrix**, never from the literature.

Two sources, which can be combined in one index:

- **BioSysLit**: the Zenodo community of published taxonomic treatments,
  searched by taxon name.
- **Your own PDFs**: revisions, original descriptions, your own drafts. A PDF
  needs a text layer. Scanned papers have none and contribute nothing, so run OCR
  first (for example `ocrmypdf scan.pdf searchable.pdf`). To use descriptions you
  wrote yourself, save them as PDF (Word: *Save as PDF*).

```bash
pip install "descriptron-core[rag]"

# 1. build the index once: BioSysLit records for the taxon + a folder of PDFs
descriptron biosyslit_rag_retrieval_v2 index \
    --taxon Diaphorina --family Liviidae --max-records 100 \
    --pdf-dir literature/ --output diaphorina_index.json

# 2. check what it retrieves
descriptron biosyslit_rag_retrieval_v2 retrieve \
    --index diaphorina_index.json --taxon Diaphorina --family Liviidae --k 5

# 3. use it: pass the index to the image-based descriptions ...
descriptron biosyslit_rag_retrieval_v2 describe-biorag --index diaphorina_index.json ...
#    ... or give the whole pipeline the index (BioSysLit + PDFs) or just a folder of PDFs
descriptron run_full_pipeline_v2 --rag_index diaphorina_index.json ...
descriptron run_full_pipeline_v2 --pdf_dir literature/ ...
```

The literature is used by the step that describes each structure from the
images, which needs a vision-language model (`--llm_backend claude-code` or
`api`). **Species descriptions cannot be produced without one.** With
`--llm_backend none` the run stops at what can be computed (the data matrix, the
key, delimitation and the per-species evidence sheets) and writes no descriptions. If both `--rag_index` and `--pdf_dir` are given, only the
index is used, so index the PDFs into it.

Retrieval is by keyword and metadata by default (taxon, family, section type),
which needs nothing beyond `[rag]`. For retrieval by meaning, build the index
with `--embeddings`; this needs `pip install "descriptron-core[rag-embeddings]"`
(sentence-transformers and FAISS, which bring in PyTorch). The index is a plain
JSON file: build it once per group and reuse it.

The taxonomist's own questions (which characters to examine and what matters in
the group) are a separate input: a plain-text, .docx or .csv file given with
`--user_prompts`, or named as `questions_file` in the taxon profile.

---

## What the programs produce

Descriptron-v2 contains 67 programs, 51,107 lines. `docs/FigS1_script_inventory.png` draws all of them,
and `docs/FigS1_script_inventory.tsv` is the same information as a table: for
every program, its stage, line count, whether it calls a model, which pip
distribution installs it, **and what it writes**.

| stage | programs | lines | typical outputs |
|---|---|---|---|
| Annotation (the GUI) | 3 | 14,832 | COCO JSON, masks, visualisations |
| Images to measurements | 9 | 9,712 | measurement tables (mm), GPA coordinates, colour and texture tables, diagnostic figures |
| Screening, policy and matrix | 8 | 1,619 | exclusion lists, the character matrix, evidence-tier policy |
| Key and treatments | 5 | 5,928 | the key (text + JSON), species treatments |
| Checking the text | 5 | 2,705 | confabulation reports, ontology coverage, gate decisions |
| Descriptive categorical characters | 5 | 1,093 | character state tables, repeatability figures |
| Naming specimens, recognising new species | 16 | 5,980 | identification tables, novelty scores, calibration sheets, ROC figures |
| Model-proposed binary characters | 9 | 5,390 | proposed characters, congruence tables, heat-map atlases |
| Names, types and outputs | 7 | 4,017 | treatment .docx, TaxPub XML, DwC-A, SDD, JSON-LD, collaborator workbooks |

<p align="center">
  <a href="docs/FigS1_script_inventory.png"><img src="docs/FigS1_script_inventory.png" alt="Inventory of the 67 Descriptron programs by pipeline stage: line counts, model use, pip distribution and outputs" width="100%"></a>
  <br>
  <sub>All 67 programs by stage, with their size, whether they call a model, the pip distribution that installs them, and what they write. Click for full resolution.</sub>
</p>

The output column in the figure is **read from the source**, not written by hand:
a program is listed as writing a figure when its code calls `savefig`, a table
when it calls `to_csv`, and so on. The figure therefore cannot drift from the
code, and any program that writes nothing is a step that only feeds the next one.

---

## Reproducing an analysis

```bash
pip install descriptron-core
descriptron run_full_pipeline_v2 --help
```

With the deposited COCO files and the taxon profile, the pipeline regenerates the
measurements, the matrix, the key, the delimitation and every figure. The steps
that write prose are skipped unless a model backend is given, and none of the
numbers depends on one.

---

## Provenance and auditing

Three mechanisms, each meant for a reader who does not take your word for it.

**Every number is traceable.** `manuscript_numbers_v18.tsv` maps each number in
the manuscript to the report file that produced it, and the manuscript builder
refuses to typeset a number that no report supplies.

**Every pipeline step records how it was made** (from 2.1.3). Each step run by
`run_full_pipeline_v2` writes `logs/<step>.provenance.json`, with a copy
`provenance_<step>.json` in the step's output folder. It records:

- the step's script and its SHA-256, and the git commit when the script is tracked;
- the full command line and the installed Descriptron package versions;
- a SHA-256 of every input file named on the command line, taken before the step ran;
- a fingerprint (names, sizes, dates) of every input folder, such as the image folder;
- a SHA-256 of every file the step wrote.

Two checks:

```bash
# has the script, an input or an output changed since the step ran?
descriptron biorag_provenance_v1 --verify path/to/logs/step6_landmark_gpa.provenance.json

# which step, script and command wrote this result file?
descriptron biorag_provenance_v1 --which path/to/some_result.csv --search path/to/output_base
```

`--which` matches the file's content, not its name, so a result edited after the run is
reported as not made by the pipeline. Records hold absolute paths, so check them where the run's
files are (in Docker, with the same `-v` mounts as the run). Scripts run on their own, outside the pipeline, are not
stamped. Every language-model call is logged separately (`llm_calls.jsonl`: time, the model
asked for and the model that answered, tokens), because the two can differ.

**And the text is audited independently of the model that wrote it.**
`biorag_confabulation_checker_v2.py` recomputes every statistic from the specimen
matrix, attributes every number in a treatment to one feature, and re-derives
every comparison — a value that is correct but attached to the wrong structure
fails, which is exactly what a generator checking its own output cannot catch.

---
<a id="mcp-server"></a>

## Use Descriptron from Claude (MCP server)

Descriptron can also be driven by an AI assistant. `descriptron-mcp` is an
[MCP](https://modelcontextprotocol.io) (Model Context Protocol) server: it lets
Claude Code, Claude Desktop, or any other MCP client run the Descriptron
programs for you. For example, you can ask *"summarise this COCO file and check it
for problems"*, *"build the key from this matrix"*, *"run SAM2-PAL on this folder
in the background"*, or *"write the treatment for this species from the matrix
and audit it"*.

The server runs **on your own computer** and works on **your own files**. It
runs the same programs you would run from a shell, so their results are the same.

**Writing and checking are kept apart.** When the assistant writes a species
treatment, it takes every number from the measured data and then submits the
text to the independent audit (`biorag_confabulation_checker_v2`). That audit
recomputes each value from the specimen matrix and flags any number printed
next to the wrong structure. The writer never marks its own work.

### Install

Python 3.10 or newer is needed for the server.

```bash
python -m venv descriptron-mcp-env
source descriptron-mcp-env/bin/activate        # Windows: descriptron-mcp-env\Scripts\activate

pip install descriptron-mcp                    # analysis programs (CPU)
pip install "descriptron-mcp[vision]"          # + detectors, SAM2-PAL, DINOLand (GPU)

descriptron-mcp --check                        # lists the programs it found
```

**With Docker instead** (no Python setup; includes the GPU programs):

```bash
claude mcp add descriptron -- docker run -i --rm --gpus all \
  --user "$(id -u):$(id -g)" -v "$HOME:$HOME" \
  ghcr.io/alexrvandam/descriptron:2.1.1 mcp
```

Without an NVIDIA GPU (e.g. on a Mac), leave out `--gpus all`: Docker refuses to start
with it, and everything except the GPU programs works the same.
`-v "$HOME:$HOME"` makes your files appear inside the container at the same
paths Claude uses; add another `-v /path:/path` for data elsewhere (e.g. an
external drive). `--user` makes the files it writes yours rather than root's.
Background jobs stop when the Claude session ends, because the container does.

### Connect it

**Claude Code**

```bash
claude mcp add descriptron -- /full/path/to/descriptron-mcp-env/bin/descriptron-mcp
```

**Claude Desktop**: Settings → Developer → Edit config, then add:

```json
{
  "mcpServers": {
    "descriptron": {
      "command": "/full/path/to/descriptron-mcp-env/bin/descriptron-mcp"
    }
  }
}
```

Restart the client and the Descriptron tools are available.

### What it offers

| tools | what they do |
|---|---|
| `list_programs`, `program_help` | the available programs and their options |
| `run_program` | run a program and report its output and the files it wrote |
| `start_job`, `job_status`, `job_log`, `cancel_job`, `list_jobs` | long runs (pipeline, SAM2-PAL, training) in the background |
| `coco_summary`, `read_table`, `read_text`, `list_files`, `view_image` | look at annotations, results, figures and specimen images |
| `species_evidence`, `audit_treatment` | write a treatment from the data, then audit it independently |

The annotation GUI is not part of the server. Annotation is interactive, and the
server works from the COCO files the GUI (or a detector) writes.

Configuration (for example, running the analysis in an existing conda
environment), tests and details: [packages/descriptron-mcp/README.md](packages/descriptron-mcp/README.md).

---

## Licence

Apache License 2.0 ([LICENSE](LICENSE)). Any redistribution of Descriptron, or of
software derived from it, must include the [NOTICE](NOTICE) file. Some bundled or downloaded components carry their own terms:
Detectron2 (Apache-2.0), SAM2 (Apache-2.0), Metric3D (BSD-2-Clause), EasyOCR
(Apache-2.0). **Model weights are licensed separately from code** — DINOv3 and
some Florence-2 checkpoints are gated and carry their own conditions, which you
should read before any commercial use. Descriptron's own licence does not
restrict commercial use; some model weights may.

---

## Citation

If you use Descriptron, or any software derived from it (including
descriptron-core, descriptron-vision, descriptron-gui and descriptron-mcp), in
work that is published, presented or distributed, cite:

1. **The software (Descriptron v2)**, by the DOI of the version you used:
   Van Dam, A. R. (2026). Descriptron: morphology-driven species descriptions,
   keys and delimitation for dark taxa (Version 2.1.1). Zenodo.
   https://doi.org/10.5281/zenodo.22958109
   (all versions: https://doi.org/10.5281/zenodo.17077224)
2. **The first Descriptron paper:**
   Van Dam, A. R. & Štarhová Serbina, L. (2026). Descriptron: Artificial
   intelligence for automating taxonomic species descriptions with a
   user-friendly software package. *Systematic Entomology*, 51(1), e70005.
   https://doi.org/10.1111/syen.70005
3. **The Descriptron v2 paper,** once it is published. Its reference will be
   added here.

```bibtex
@software{vandam_2026_descriptron_v211,
  author    = {Van Dam, Alex R.},
  title     = {Descriptron: morphology-driven species descriptions, keys and
               delimitation for dark taxa},
  version   = {v2.1.1},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22958109},
  url       = {https://doi.org/10.5281/zenodo.22958109}
}

@article{vandam2026descriptron,
  author  = {Van Dam, Alex R. and Štarhová Serbina, Liliya},
  title   = {Descriptron: Artificial intelligence for automating taxonomic species
             descriptions with a user-friendly software package},
  journal = {Systematic Entomology},
  volume  = {51},
  number  = {1},
  pages   = {e70005},
  year    = {2026},
  doi     = {10.1111/syen.70005}
}
```

The same information is in [CITATION.cff](CITATION.cff) (GitHub's "Cite this
repository" button) and in [NOTICE](NOTICE). Please also cite the tools
Descriptron builds on that you used (SAM2, DINOv3, Detectron2 and others; see
*Licence* above).
