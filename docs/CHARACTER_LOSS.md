# Recording character loss: absent structures vs missing data

Some specimens do not have a structure at all: a subterranean weevil without eyes, a species that has lost a
pronotal spine, a wingless female. That absence is a character, often a diagnostic one, and it belongs in the
matrix, the key and the description. A structure you simply did not annotate, or could not see, is something
else: missing data. Descriptron (from 2.7.8, GUI v84) keeps the two apart. **Only what you record as absent is
treated as a loss; a gap is never turned into one.**

## The four states

| What you do in the GUI | Coordinates / outline | Character state | In the matrix, key and description |
|---|---|---|---|
| Draw or predict a mask | used | **present** | measured as usual; "present" for a structure that is absent in other specimens |
| Place a **Positive** point | used | **present** | used as a landmark |
| **Record absent → Absent (lost)** for a structure | none | **absent** | a present/absent character: "eyes absent" |
| Place a **Lost** point (keypoint) | none (COCO v = 0) | **absent** | a present/absent character for that keypoint; the landmark is missing in GPA |
| **Record absent → Not visible** for a structure | none | **unknown** | missing data (not scored) |
| Place a **Negative** point | none (COCO v = 0) | **unknown** | missing data; also a SAM2 "not here" prompt |
| Nothing at all | none | **unknown (gap)** | missing data, **listed in the completeness worklist for you to check** |

Measurements of an absent structure are left empty, never written as 0.

## How to record a structure that is absent (e.g. no eyes)

1. Open the image of the specimen.
2. In the toolbar's **MASKS** row press **Record absent** (lilac button).
3. Choose the structure (or type its name, e.g. `eye`) and press **Absent (lost)**.
   - If it may be there but cannot be seen (hidden under the pronotum, broken off, out of focus), press
     **Not visible** instead: that is missing data, not a loss.
4. The list in the window shows what you recorded for this image; **Remove selected** undoes a record.
5. Save with the marmot as usual. The record is written on the image entry of the COCO file:
   `"structure_status": {"eye": "absent"}`. Detectron2, SAM2-PAL and other COCO readers ignore it.

Record every specimen that lacks the structure. One record per specimen: an eyeless species with ten specimens
needs ten records, so the matrix knows it was checked in all ten.

## How to record a keypoint that is absent (e.g. a lost spine)

Your landmark scheme has 16 points and points 10 and 11 sit on a spine this species does not have:

1. Place points 1 to 9 with **Positive** as usual (**Point Prompt** on).
2. Switch the point type (the Positive / Negative / Lost drop-down in the toolbar's **SAM2** row) to **Lost** and
   click twice anywhere on the image. The two purple points take the numbers 10 and 11.
3. Switch back to **Positive**: the next point is 12.

The numbers stay fixed, so landmark 12 is landmark 12 in every specimen. A point already placed can be changed
later: **Keypoint Edit** on, right-click the point, **Mark Lost - character loss (purple)** (or **Mark Not visible
- missing data** for a point that exists but cannot be placed). Saved as COCO visibility 0 at (0, 0) with
`"keypoint_status": [..., "absent", "absent", ...]`; lost points are never given to SAM2 as prompts.

Use **Negative** only for SAM2 background clicks and for landmarks you cannot see: a negative point is missing
data, not a loss.

## Checking your annotations: the completeness check

Run it from **Utilities → CHECK → Completeness check**, or

    descriptron descriptron_check_completeness_v1 --coco annotations.json --group_labels group_labels.csv --out_dir completeness/

The full pipeline runs it as its first step (`completeness`). The unit is the **specimen**: when a specimen is
photographed in several images (head, legs, terminalia), they are merged using the taxon profile's specimen
pattern (`--taxon_profile`), `--specimen_regex`, or the Darwin Core `occurrenceID` (`--metadata`), so a head photo
is not reported as 'missing' its legs. Structures the taxon profile limits to one sex are not expected in the
other. It writes:

- `completeness_heatmap.png`: specimens × structures and keypoints: green present, purple absent, grey not
  visible, white nothing recorded;
- `completeness_worklist.csv`: every gap, most likely omissions first:
  1. missing in this specimen while other specimens of the species have it: probably forgotten - annotate it, or
     record it as absent / not visible;
  2. nothing recorded in any specimen of the species (one row per species and structure): either a loss not yet
     recorded - record it as absent - or a body part that was not imaged for that species;
  3. specimens without a species;
- `completeness_conflicts.csv`: recorded absent but annotated as well;
- `completeness_by_character.tsv`: per structure and keypoint, how many specimens are present / absent / not
  visible / gaps, and the share scored (present + absent).

## Leaving out poorly scored characters

- `--min_completeness 0.8` (pipeline, or `biorag_key_feature_filter_v2 --completeness_table ...`) drops every
  feature of a structure scored in fewer than 80% of specimens.
- `--completeness_strict` stops the pipeline at the first step while gaps or conflicts remain.
- `--landmarks_present_in_all` (landmark GPA) keeps specimens that lack some landmarks and analyses the landmarks
  every specimen has; by default those specimens are left out of the GPA. Either way a missing landmark is never
  given a position.

## What happens downstream

- **Matrix**: each structure recorded absent in at least one specimen gets a character *presence* (present /
  absent), at the key tier. Not visible and gaps are empty cells.
- **Key**: a couplet can use it ("eyes absent (n = 10)" vs "eyes present (n = 24)").
- **Descriptions**: written in words; a state recorded in no other species is offered for the Diagnosis
  ("eyes absent: recorded in no other species").
- **Audit**: a description that calls a structure present in a species where it was recorded absent (or the
  reverse) is an error.
- **Geometric morphometrics**: missing landmarks are never used as coordinates; TPS export writes them as -1 -1
  (tpsDig / geomorph convention); keypoint distances keep the real landmark numbers.

## Files from before 2.7.8

Older COCO files have no records, so every missing structure appears as a gap and every negative point as not
visible. Nothing old is turned into a loss: open the files in v84 and record the absences. Note: before v84 the
marmot's final JSON saved every manually placed keypoint as visible (v = 2), negative points included, at the
spot clicked; the separate `_keypoints.json` saved them correctly as v = 0. Re-save keypoint files made with
negative points in v84, or use their `_keypoints.json`.
