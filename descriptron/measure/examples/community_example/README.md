# Example: does phylogeny, habitat, or both shape the colour of a community?

A small synthetic data set for `descriptron_community_v1.py` (24 ant species, 22 sites: 10 primary forest and
12 secondary forest, 324 specimens). It shows the file formats; the numbers are simulated (`make_example.py`
rebuilds everything from a seed) so the analysis has known answers to recover.

## The metadata file (Darwin Core)

`metadata_darwin_core_example.csv`: one row per specimen. The program reads these columns by default (any can be
renamed with an option):

| Column | Darwin Core term | Used as | Option |
|---|---|---|---|
| `occurrenceID` | occurrenceID | the specimen | `--specimen_col` |
| `scientificName` | scientificName | the species (matched to tree tips; spaces or underscores) | `--species_col` |
| `locationID` | locationID | the site | `--site_col` |
| `habitat` | habitat | the treatment (here primary / secondary forest) | `--factor` (first one) |
| `associatedMedia` | associatedMedia | image file name(s) of the specimen, `|` separated: links trait rows to specimens | `--media_col` |
| `associatedSequences` | associatedSequences | which specimen was sequenced (the species' exemplar); informational | |
| `minimumElevationInMeters` | minimumElevationInMeters | an example site-level covariate | `--covariate` |

Other columns (catalogNumber, institutionCode, basisOfRecord, family, genus, specificEpithet, locality, country,
decimalLatitude, decimalLongitude, eventDate, recordedBy, identifiedBy) are standard Darwin Core and are carried
along. Further categorical variables (season, collecting method, ...) go in with more `--factor` options; further
site-level numbers (canopy cover, distance to edge, ...) with more `--covariate` options. If no
`associatedMedia` is given, a trait row is linked to the specimen whose `occurrenceID` appears in its file name.

## Two ways to give the phylogeny

1. **One exemplar per species, one tree, pruned per site** (`tree_species.nwk`): tip names are the species. A
   site's community is the tree pruned to the species recorded there in the metadata.

       python ../../descriptron_community_v1.py --metadata metadata_darwin_core_example.csv \
           --tree tree_species.nwk --traits colour=traits_colour_example.csv \
           --factor habitat --covariate minimumElevationInMeters --out_dir out_species_tree

2. **Every species sequenced at every site** (`tree_site_tips.nwk` + `tip_map.csv`): one tree of all sequences;
   `tip_map.csv` (columns `tip, species, site`) says which species and site each tip is. A site's community is its
   own sequences; analyses at the species level use one tip per species.

       python ../../descriptron_community_v1.py --metadata metadata_darwin_core_example.csv \
           --tree tree_site_tips.nwk --tip_map tip_map.csv --traits colour=traits_colour_example.csv \
           --factor habitat --out_dir out_site_tips

A site can also have a tree of its own: `--site_tree P01=P01.nwk` (repeatable).

## Traits

`traits_colour_example.csv` stands in for Descriptron's colour / colour-pattern / texture tables (for example
`color_homology_features_<structure>.csv` and the texture homology features): one row per image, a file-name
column and numeric features. Give each with `--traits NAME=path`. Alternatively give the annotations directly,
`--coco annotations.json --image_dir images/ [--structure head]`, and basic colour (CIE L*a*b*) and texture
(grey-level co-occurrence) features are computed from the masks.

## What the example should show

The traits were simulated with Brownian motion on the tree, a shift in secondary forest on two of the twelve
features (the same in every species), and habitat preferences that follow the phylogeny. The analysis recovers
strong phylogenetic signal (phylogeny-only fraction about 0.5, Kmult about 0.8), the within-species habitat shift
in the paired test (species found in both forests), and phylogenetically clustered primary-forest communities
(negative SES of MPD, compared with secondary forest across sites).
