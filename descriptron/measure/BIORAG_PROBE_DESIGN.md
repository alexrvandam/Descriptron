# BioRAG-Probe: design (17 Oct 2026)

Goal: a user brings images and literature (PDFs or a literature search) for ANY character and gets measured, validated
character states, without writing code and without us building it for them. Proof of concept on our own undescribed
weevils, ants and flies.

## Three parts, all taxon-neutral
1. **Literature -> labels.** Statements harvested from PDFs/Plazi/BHL with the verbatim quote, source, licence and the level
   they apply to (genus, species group, species). Checked by an independent quote validator. Taxon specifics live ONLY here.
2. **Images -> DINOv3 tokens -> regions.** Native resolution, overlapping tiles, 8-px shifted grid for maps. No training needed:
   - region masks from the user, or proposals (deep layer, e.g. L23 = whole parts, mirror-symmetric) that an expert only NAMES;
   - thin structures (veins, bristles, carinae): high-pass tokens + dark-ridge rule;
   - broad regions (spots, bands, colour patches): raw tokens + darkness/colour rule;
   - parts named by order along a margin (veins from the costa): a rule plus a one-line PROFILE per group.
3. **Probe = labels x tokens, validated.** Taxon (or part) held out; simple baseline alongside (dark dots, darkness+width,
   pixel threshold); shuffled-label null. Specimen level (A) and patch level (B, declared primary).

## Character types (routes); each route is general across taxa
| Type | Examples | Route | Layer / signal |
|---|---|---|---|
| Region / part | elytron, ant head parts, genital lobes, wing cells | masks or L23 proposals named by expert | L23 |
| Thin line | veins, carinae, striae | token regions, thin mode | L12 high-pass |
| Broad colour region | spots, bands, maculae | token regions, broad mode | raw L12/L18 |
| Dense surface texture | sculpture, rugosity, punctation, microtrichia, scales, palmate setae, tomentum | texture patches + probe | L18 raw + high-pass |
| Sparse setae on a structure | vein macrotrichia (setulae), pilosity of a sclerite | patches on that structure + probe, calibrated on the same structure | L18 raw + high-pass |
| Individual bristles | genitalic macrosetae, aristate setae over scales | instances (seta class + L12 high-pass + tracing), counted per named region | L12 high-pass |
| Pits / pores | punctures, sensilla, pores | instances or patches by density (sparse -> instances, dense -> patches) | L12 / L18 |
| Surface relief (key weevil terms) | tubercles, warts, granules, callosities, costae (raised intervals), crests, humeral/apical calli, spines | instances located on the named part (count, size, which interval) + outline protrusion in profile; dense granulation as texture patches | L12 high-pass / outline |
| Tufts / fascicles | erect scale or seta tufts, often on tubercles | instances + outline protrusion in profile | L12 / outline |
| Hairs against background | wing fringe, outline setae | direct measurement along the outline normal | native pixels |
| Hidden structures | striae under scales | measure where visible, else "not observable" (never "absent") | any |

## Rules learned (keep)
- Native resolution; never judge setae on downscaled sheets (3-4 px sockets vanish in 16-px patches).
- High-pass alone can flip sign between taxa: classify on raw + high-pass together.
- Hold out what you claim to generalise to (taxon, genus, vein); every result next to a simple baseline and a null.
- Calibrate each setae type on itself (sparse setulae are not dense setae; a reference from another setting does not transfer).
- Validated example: the sculpture probe (L18) works on BOTH ants and weevils (same route, two orders): ant sculpture scale from the Hita Garcia & Fisher key (L18 AUC .97 vs GLCM .83) and weevil texture/punctation.

## Scripts (gui/measure)
biorag_probe_v1.py (taxon-level probe; --exclude, --keep_majority_cluster), biorag_probe_parts_v1.py (part-level probe,
--apply_parts), biorag_probe_panels_v2.py (photo | PCA | high-pass PCA | segmentation), wing_vein_unsupervised_eval_v1.py and
wing_spot_tokens_v1.py (to be merged into one token-regions tool, thin/broad modes), wing_vein_naming_rule_v1.py (--profiles),
wing_fringe_character_v1.py (outline hairs), wing_spot_pattern_v1.py (pixel-threshold baseline only).
