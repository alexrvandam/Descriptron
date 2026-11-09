#!/usr/bin/env python3
"""
descriptron_phylo.py - phylogenetic comparative analyses of any continuous Descriptron traits
=============================================================================================

Given a Newick tree and one or more trait tables (landmark shape, linear measurements, colour pattern, texture, or
any other continuous features), computes on species means:

  signal     multivariate phylogenetic signal Kmult (Adams 2014), permutation P and effect size Z,
             as geomorph::physignal
  pgls       phylogenetic regression of a trait set on a predictor (for example log centroid size: phylogenetic
             allometry) by generalised least squares under Brownian motion, SS / R2 / F with residual
             randomisation, as geomorph::procD.pgls
  morphospace principal components of the species means with ancestral states estimated under Brownian motion
             (generalised least squares) projected into the same space and the tree drawn through them, as
             geomorph::gm.prcomp(phy = )
  models     for every variable on its own: Brownian motion (BM), Ornstein-Uhlenbeck with one optimum (OU), early
             burst (EB), Pagel's lambda and white noise (no phylogeny) fitted by maximum likelihood and compared by
             AICc and Akaike weights, as geiger::fitContinuous (same covariance forms and default bounds); lambda
             is also tested against 0 (no signal) and 1 (BM) by likelihood ratio
  rates      for every variable on its own: a separate evolutionary rate on every branch (variable-rate Brownian
             motion by penalised likelihood, Revell 2021), as phytools::multirateBM; the branches with the fastest
             change are listed and the tree is drawn with its branches coloured by rate. Each branch's change
             between the Brownian-motion ancestral states is also given, divided by its expected standard deviation
             (sqrt(sigma^2 x branch length)), so single large jumps stand out from a generally fast clade

  colour_tree a tree of colour-pattern similarity: the species-mean L*, a*, b* of every homologous grid cell (the
             colour-homology table; --colour_cells), the species distance = the mean over the cells both species have
             of the per-cell colour difference (--colour_distance de2000: CIEDE2000, Sharma et al. 2005; or de76),
             clustered by UPGMA or neighbour joining (--colour_tree_method, as phangorn::upgma / ape::nj); clade
             support = share of --bootstrap trees built from grid cells resampled with replacement that contain the
             clade (bipartition for NJ), as ape::prop.clades
  cophylo    the input tree and the colour tree face to face (tanglegram, PNG and PDF), nodes rotated to minimise line
             crossings (as phytools::cophylo(rotate = TRUE), counting crossings exactly); Mantel test of patristic
             against colour distance (Pearson and Spearman, --iterations, as vegan::mantel); Robinson-Foulds distance
             (unrooted, as phangorn::RF.dist)
  heatmap    the input tree at the left, the colour tree on top (support printed) and the species x species colour
             distance in between (one-hue scale, dark = similar colour)
  convergence candidate convergent groups, given (--convergent_groups species,group) or found: colour-tree clades with
             support >= --min_support that are not clades of the input tree, each member's molecular lineage and its
             sister group listed. For each group, on the first colour principal components (--conv_pcs; default the
             PCs reaching 90% of the variance, at most --conv_pcs_max) or every standardised colour variable:
             - Stayton's C1-C4 over all pairs of members, as convevol 2.2.1 calcConv (Dmax = the largest distance
               between any node of one lineage and any node of the other, tips and MRCA included; C4 divides by the sum
               of the distances between all pairs of nodes descending from the MRCA, as convevol 2.2.1 codes it)
             - Ct1-Ct4 over pairs from different molecular lineages, as convevol 2.2.1 calcConvCt with groups
               (Grossnickle et al. 2024: Dmax only between an ancestor and the other lineage at the same time). Ct4
               divides by the change along the branches of the MRCA's clade; convevol 2.2.1 indexes the whole tree's
               edge table there (the first m branches of the tree, m the clade's branch count) - Descriptron uses the
               clade, and reproduces convevol's form with conv_stats(..., ct4="convevol") for the validation
             - significance as convevol's convSig / convSigCt: --conv_sims data sets simulated under multivariate
               Brownian motion with root 0 and the rate matrix phytools::phyl.vcv(X, C, lambda = 0), P = #{simulated
               >= observed} / simulations
             - the Wheatsheaf index (Arbuckle et al. 2014) as windex 2.1.0 windex (with its leave-one-species-out 95%
               interval) and test.windex (P against --conv_sims resamplings of the species' trait rows); ultrametric
               trees only (windex's fossil option is not implemented)
             Candidates chosen from the colour tree are chosen because they look alike, which the Brownian-motion null
             does not model: their P values are descriptive.
             Colour distances (--colour_distance, one or several; several write <set>__<distance>_... files and a
             <set>_distance_comparison.csv):
             - de2000 / de76: mean per-cell CIEDE2000 / CIE76 between species-mean L*a*b* of the homology grid cells
               (bootstrap: grid cells resampled; convergence traits: PCs of the cells' L*a*b*)
             - palette_emd: the specimens' recolorize palettes (--palettes NAME=GLOB_OR_CSV), specimen distance = earth
               mover's distance (pixel-share weights, Euclidean CIELAB ground distance, as emdist::emd), species distance
               = mean over their specimen pairs (as validation/recolorize_v22/scripts/trees.R). Bootstrap: specimens
               resampled within species (a species with one specimen is fixed, so support only measures stability to
               which specimens represent a species). Convergence traits: principal coordinates (PCoA) of the species
               EMD matrix. Also <set>__palette_emd_colour_bars.png: the tree with each species' mean palette as a bar
             - pca: Euclidean distance on the first --pca_pcs PCs of the standardised colour-pattern features (those
               the phylomorphospace uses). Bootstrap: features resampled. Convergence traits: the same PCs
  surface    SURFACE (Ingram & Mahler 2013, Methods Ecol. Evol. 4: 416) on the first --surface_pcs colour-pattern PCs:
             Ornstein-Uhlenbeck with several optima (Hansen 1997; Butler & King 2004; ouch 2.20 hansen, its
             stationary-root covariance and glssoln, R's Nelder-Mead reproduced step by step), each PC its own alpha
             and sigma^2, regimes shared; forward phase adding the regime shift that lowers AICc most (n = species x
             PCs, parameters = shifts + PCs x (2 + regimes), as surface 0.6), backward phase merging regimes when that
             lowers AICc: regimes reached by two or more separate shifts are convergent, found without a priori
             groups. Writes the path, every branch's regime, alpha / sigma^2 / optima, a comparison of BM, OU1, EB,
             lambda and SURFACE by AICc on the same PCs, and chronograms coloured by regime and by the variable-rate BM
             rate (mean of the PCs' rates). The tanglegram's molecular tree is coloured by the regimes when SURFACE has
             the lowest AICc, otherwise by the rates (--tangle_tree_colour)

  reference_scores  every --colour_distance scored against expert groupings (--reference_groups CSV with a species
             column, --reference_columns e.g. group,mimicry_type), on all species with colour data and a label
             (tree or not; the distances need no tree) and on the tree species (--score_species): nearest-neighbour
             agreement (share of species whose nearest neighbour has the same label, among species whose label has >= 2
             members; tied neighbours share the vote), the adjusted Rand index (Hubert & Arabie 1985) of the UPGMA tree
             cut into k = number of labels (cutree(hclust(average), k)) and of PAM k-medoids at k (cluster::pam;
             k-medoids rather than a second average-linkage cut, which would repeat UPGMA), each with a permutation P
             (labels shuffled, --iterations). Writes <set>__<distance>_reference_scores.csv, _reference_nearest.csv and
             reference_scores_summary.csv
  mk_maps    a discrete trait (--discrete_trait CSV:COLUMN, e.g. mimicry type): Mk models ER, SYM, ARD (--mk_models)
             by maximum likelihood, root prior 1/k (phytools::fitMk pi = "equal"; ape::ace's log-likelihood omits the
             1/k and is this + log k), chosen by AICc (n = species); marginal ancestral states (phytools::ancr); --simmaps
             stochastic character maps with Q fixed at the estimate (make.simmap(Q = fitted)), each branch history
             drawn by uniformization (exact, as make.simmap's rejection sampling); per state the transitions into it and
             the separate origins the living species descend from, with 95% intervals over maps; the dated tree
             painted by the state most maps are in along each branch, with pies of the marginal probabilities
  oum        one colour-pattern optimum per discrete state (Hansen 1997; Butler & King 2004) on the SURFACE PCs: each
             branch in the state at the node it leads to (ouch's node painting), from the most probable marginal states
             and from each of --oum_maps stochastic maps (as OUwie users fit over a set of simmaps); compared by AICc
             (n = species x PCs, as surface) with BM, OU1, BMS (one rate per state; phytools::brownie.lite), white noise
             (one mean, or one mean per state: OUM's limit for a very short half-life) and SURFACE on the same PCs;
             optima per state per PC, alpha, sigma^2 and the phylogenetic half-life ln 2 / alpha
  ancestral_pattern  the colour pattern of every ancestor, as a picture: L*, a*, b* of every homology grid cell
             (--colour_cells) reconstructed at every internal node, each cell and channel on its own, by maximum
             likelihood under Brownian motion with 95% intervals, as phytools::fastAnc(CI = TRUE) (the same estimates as
             ape::ace(method = "ML"), which its numerical optimiser reaches on standardised data; on raw CIELAB
             values it can stop early; ace's intervals are computed differently); a cell some species lack is reconstructed on the species that
             have it (fastAnc on the tree pruned to them; nodes the pruning removes get the conditional expectation),
             --anc_missing complete uses only cells every species has. With --cell_geometry (the grid written by
             color_phenomics_homology_v2_1.py --save_cell_geometry) each node's and species' pattern is drawn on the
             consensus outline (cells filled with the CIELAB colour in sRGB, clipped to the outline): the dated tree with
             a picture at every node and tip (<set>_ancestral_pattern_tree), every node with its interval
             (<set>_ancestral_pattern_nodes), per-cell maps of Blomberg's K / Kmult(L*a*b*) with permutation P and of the
             REML Brownian-motion rate (<set>_ancestral_pattern_cell_maps); with --discrete_trait, per state the origins
             found by the stochastic maps (per species, the branch carrying the most recent change into its state in
             most maps), the reconstructed ancestor before each change next to its descendants
             (<set>_ancestral_pattern_convergence) and the mean CIEDE2000 between those ancestors and between their
             descendants. Tables: node x cell x channel estimates, SE and intervals (<set>_ancestral_pattern_states.csv)
             Branches coloured by the rate of colour-pattern change (--anc_branch_colour, one figure per mode):
             rate_de00 = the mean CIEDE2000 over the grid cells between the reconstructed parent and the child (a species:
             its mean pattern) divided by the branch length (Delta E00 per Myr on a chronogram; log colour scale relative
             to the median; under BM change grows with sqrt(time), so short branches read faster), bm_rate = the
             variable-rate BM rate (multirate_bm) averaged over the first --model_pcs pattern PCs; every branch's change,
             rates and the cells that changed most in <set>_ancestral_pattern_branches.csv
  part_sets  (--part_sets A B C ..., any names: parts of the same specimens, each run through ancestral_pattern) the
             parts' rates compared: Delta E00 per Myr over the whole tree, Kmult of the part's L*a*b* cells, Adams'
             (2014) sigma^2_mult per part with the test of equal rates by simulation, as
             geomorph::compare.multi.evol.rates (geomorph 4.0.7 - 4.1.1 divide every part by the total number of
             variables in its simulations; Descriptron divides by the part's own, and reports both), the correlation
             between parts of the branch changes scaled by sqrt(time); body_map_part_rates: the organism assembled from
             the parts' consensus outlines (stacked along y or x, or at their median positions in a COCO file with the
             parts as categories: --body_map_layout coco) with each part filled by its value
  discrete_tree  a discrete trait (--discrete_trait) on the tree: Fitch parsimony branch colours, codes and colours
             (--state_codes), a label column and tip pictures (--tip_images), the separate clades per state
  --combine_sets NEW=SET1:W1,SET2:W2  (with reference_scores) the sets' species colour distances, each divided by its
             median, weighted and summed (/ sum of weights), scored against the reference groups like a set

Models and rates are univariate: fit them to the original variables (for colour, L*, a*, b* or chroma of each part,
never a raw hue angle, which is circular). --model_pcs K|auto fits them to the colour-pattern PCs instead, one PC at a
time; PCA of species data can favour OU and EB spuriously (Uyeda et al. 2015, Syst. Biol. 64: 677), so every PC is
reported, not only PC1. OU and EB parameters are in the tree's units: on a tree in substitutions
per site (not time) alpha and a are per substitution, and geiger's default bounds (alpha <= e, a >= ln(1e-5)/depth)
depend on those units; estimates at a bound are flagged and the bounds can be changed with --model_bounds.

Trait tables: CSV with one row per specimen (or per species) and numeric columns. The species of each row comes
from a column (--species_col) or from a regular expression applied to an id column (--species_regex). Rows of the
same species are averaged. Species absent from the tree, or tree tips without data, are dropped (the tree is pruned).

  python descriptron_phylo.py --tree tree.nwk --out_dir phylo/ \
      --traits shape=landmark_gpa/forewing/forewing_procrustes_coords.csv \
      --traits colour=color_homology/whole_wing/color_homology_features_whole_wing.csv \
      --species_regex "morph_([A-Za-z]+\\d*?)_?\\d+_" --scale colour=standardize \
      --pgls shape~logCS --analyses signal pgls morphospace

Scaling (--scale NAME=none|standardize|log): Kmult and the PGLS statistics weight every column by its variance, so a
set that mixes units (millimetres with ratios, colour channels with entropies) should be standardised; Procrustes
shape coordinates are analysed as they are (the default for a set whose columns look like x/y coordinates).

Tree tips holding more than the species (voucher codes such as "QH001-S1_Abemus chloropterus"): --tip_regex with one
group extracts the species; a species appearing on several tips keeps the first. Spaces and underscores count as the
same character when tips are matched to species.

Colour-pattern tree and convergence, for a colour-homology table (the colour cells are found by --colour_cells):

  python descriptron_phylo.py --tree tree.nwk --groups group_labels.csv --out_dir phylo_colour/ \
      --traits body=color_homology/body/color_homology_features_body_combined_hw1.csv \
      --analyses colour_tree cophylo heatmap convergence --bootstrap 1000 --conv_sims 1000 --iterations 9999

Three colour distances, models on the colour-pattern PCs, SURFACE, tip pictures:

  python descriptron_phylo.py --tree chronogram.nwk --groups group_labels.csv --out_dir phylo_colour/ \
      --traits body=color_homology/body/color_homology_features_body_combined_hw1.csv \
      --palettes "body=color_extraction/body/*_palette.csv" --colour_distance de2000 palette_emd pca \
      --analyses colour_tree cophylo heatmap convergence models rates surface --model_pcs auto \
      --tip_images tip_images.csv --tip_image_rotate 90

Validated against geomorph 4.1.1 and ape on geomorph's plethspecies data (validation/validate_phylo_real_tree.py);
models and rates against geiger 2.0.11 and phytools 2.4.4 (validation/validate_phylo_models_vs_r.py); colour trees,
tanglegram and convergence against farver 2.1.2, phangorn 2.12.1, ape 5.8.1, vegan 2.7.5, phytools 2.4.4, convevol
2.2.1 and windex 2.1.0 (validation/validate_phylo_colour_convergence_vs_r.py); palette EMD, PCoA, PCA distance,
their bootstrap supports, the multi-optimum OU likelihood and SURFACE against emdist 0.3-3, stats, ape, phangorn,
ouch 2.20, surface 0.6 and geiger (validation/validate_phylo_palette_surface_vs_r.py); reference scores, Mk models,
ancestral states, stochastic maps, OUM and BMS against stats::cutree/hclust, cluster 2.1.6 pam, an R transcription of
the adjusted Rand index, phytools 2.4.4 fitMk / ancr / make.simmap / describe.simmap / brownie.lite, ape 5.8.1 ace and
ouch 2.20 hansen (validation/validate_phylo_mimicry_oum_vs_r.py); ancestral patterns, their intervals (also with cells
missing in some species), per-cell K and rates and the CIELAB-to-sRGB drawing colours against phytools fastAnc /
phylosig, ape ace / pic and grDevices / farver (validation/validate_phylo_ancestral_pattern_vs_r.py); branch Delta E00
rates, the variable-rate BM branch rates on the pattern PCs, Adams' sigma^2_mult with geomorph's makePD and statistic,
and Kmult per part against phytools fastAnc / multirateBM, farver compare_colour and geomorph 4.0.7
compare.multi.evol.rates / physignal (validation/validate_phylo_branch_rates_vs_r.py).

Ancestral colour pattern as pictures (cell geometry from color_phenomics_homology_v2_1.py --save_cell_geometry):

  python descriptron_phylo.py --tree chronogram.nwk --groups group_labels.csv --out_dir ancestral/ \
      --traits soma=fine_grid_soma/color_homology_features_soma_combined_hw1.csv \
      --cell_geometry soma=fine_grid_soma/color_homology_cells_soma.json \
      --discrete_trait lookalike_groups.csv:mimicry_type --state_codes non_mimic=n:#8a8a8a \
      --analyses ancestral_pattern --simmaps 200 --iterations 999

Look-alike scores, Mk maps of a mimicry type and OUM:

  python descriptron_phylo.py --tree chronogram.nwk --groups group_labels.csv --out_dir mimicry/ \
      --traits soma=color_homology/soma/color_homology_features_soma_combined_hw1.csv \
      --palettes "soma=color_extraction/soma/*_palette.csv" --colour_distance de2000 palette_emd pca \
      --reference_groups lookalike_groups.csv --reference_columns group,mimicry_type \
      --discrete_trait lookalike_groups.csv:mimicry_type --analyses reference_scores mk_maps oum --iterations 9999
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================ tree
class Node:
    __slots__ = ("name", "length", "kids", "parent", "depth", "idx")

    def __init__(self, name="", length=0.0):
        self.name, self.length, self.kids, self.parent, self.depth, self.idx = name, length, [], None, 0.0, -1


def parse_newick(s: str) -> Node:
    s = s.strip().rstrip(";")
    pos = 0

    def node() -> Node:
        nonlocal pos
        n = Node()
        if s[pos] == "(":
            pos += 1
            n.kids.append(node())
            while s[pos] == ",":
                pos += 1
                n.kids.append(node())
            pos += 1                                                  # ')'
        m = re.match(r"([^:,();\[]*)(?:\[[^\]]*\])?(?::([0-9.eE+\-]+))?", s[pos:])
        pos += m.end()
        n.name = m.group(1).strip().strip("'\"")
        n.length = float(m.group(2) or 0.0)
        for k in n.kids:
            k.parent = n
        return n

    root = node()
    root.length = 0.0                                                 # the root's own branch does not count
    _set_depths(root)
    return root


def _set_depths(n: Node, d: float = 0.0):
    n.depth = d
    for k in n.kids:
        _set_depths(k, d + k.length)


def tips_of(n: Node) -> List[Node]:
    return [n] if not n.kids else [t for k in n.kids for t in tips_of(k)]


def internal_nodes(n: Node) -> List[Node]:
    return ([n] if n.kids else []) + [x for k in n.kids for x in internal_nodes(k)]


def prune(root: Node, keep: set) -> Node:
    """Drop tips not in `keep`; collapse nodes left with one child (their branch lengths add up)."""
    def rec(n: Node) -> Optional[Node]:
        if not n.kids:
            return n if n.name in keep else None
        kids = [c for c in (rec(k) for k in n.kids) if c is not None]
        if not kids:
            return None
        if len(kids) == 1:
            kids[0].length += n.length
            return kids[0]
        n.kids = kids
        for k in kids:
            k.parent = n
        return n

    r = rec(root)
    r.length, r.parent = 0.0, None
    _set_depths(r)
    return r


def mrca_depth(a: Node, b: Node) -> float:
    anc = set()
    x = a
    while x is not None:
        anc.add(id(x)); x = x.parent
    y = b
    while id(y) not in anc:
        y = y.parent
    return y.depth


def vcv(nodes_a: List[Node], nodes_b: List[Node]) -> np.ndarray:
    """Brownian-motion covariance: shared path length from the root (tips and internal nodes alike)."""
    return np.array([[mrca_depth(a, b) for b in nodes_b] for a in nodes_a])


# ============================================================================ statistics
def _whiten(C: np.ndarray) -> np.ndarray:
    """P with P'P = C^-1 (any such P gives the same GLS sums of squares)."""
    return np.linalg.inv(np.linalg.cholesky(C))


def kmult(Y: np.ndarray, C: np.ndarray) -> float:
    """Adams (2014) multivariate K, as geomorph::physignal."""
    n = len(Y)
    Ci = np.linalg.inv(C)
    one = np.ones((n, 1))
    a = np.linalg.solve(one.T @ Ci @ one, one.T @ Ci @ Y)
    R = Y - one @ a
    obs = np.trace(R.T @ R) / np.trace(R.T @ Ci @ R)
    expct = (np.trace(C) - n / Ci.sum()) / (n - 1)
    return float(obs / expct)


def effect_size(obs: float, rand: np.ndarray) -> float:
    r = np.r_[obs, rand]
    sd = r.std(ddof=1)
    return float((obs - r.mean()) / sd) if sd > 0 else float("nan")


def phylo_signal(Y, C, iters, rng) -> Dict:
    K = kmult(Y, C)
    rand = np.array([kmult(Y[rng.permutation(len(Y))], C) for _ in range(iters)])
    return {"Kmult": K, "P": float((1 + (rand >= K - 1e-12).sum()) / (iters + 1)), "Z": effect_size(K, rand)}


def pgls(Y: np.ndarray, x: np.ndarray, C: np.ndarray, iters: int, rng) -> Dict:
    """Y ~ x under Brownian motion by GLS; SS from the whitened data, residual randomisation of the null model
    (intercept only), as RRPP / geomorph::procD.pgls with a single predictor."""
    n = len(Y)
    P = _whiten(C)
    Yt = P @ Y
    X0 = P @ np.ones((n, 1))
    X1 = P @ np.column_stack([np.ones(n), x])

    def rss(X, Yy):
        B, *_ = np.linalg.lstsq(X, Yy, rcond=None)
        R = Yy - X @ B
        return float((R ** 2).sum())

    B0, *_ = np.linalg.lstsq(X0, Yt, rcond=None)
    fit0, res0 = X0 @ B0, Yt - X0 @ B0
    ss_tot, ss_res = rss(X0, Yt), rss(X1, Yt)
    ss = ss_tot - ss_res
    df, dfr = 1, n - 2
    F = (ss / df) / (ss_res / dfr)
    rand = []
    for _ in range(iters):
        Yp = fit0 + res0[rng.permutation(n)]
        s_t, s_r = rss(X0, Yp), rss(X1, Yp)
        rand.append(((s_t - s_r) / df) / (s_r / dfr))
    rand = np.array(rand)
    return {"Df": df, "SS": ss, "MS": ss / df, "Rsq": ss / ss_tot, "F": F, "Z": effect_size(F, rand),
            "P": float((1 + (rand >= F - 1e-12).sum()) / (iters + 1)), "SS_residual": ss_res, "SS_total": ss_tot}


def ancestral_states(Y: np.ndarray, tips: List[Node], nodes: List[Node]) -> np.ndarray:
    """Ancestral states under Brownian motion by GLS: a = mu + C_nt C_tt^-1 (Y - 1 mu), with mu the GLS root state."""
    Ctt = vcv(tips, tips)
    Cnt = vcv(nodes, tips)
    Ci = np.linalg.inv(Ctt)
    one = np.ones((len(tips), 1))
    mu = np.linalg.solve(one.T @ Ci @ one, one.T @ Ci @ Y)
    return mu + Cnt @ Ci @ (Y - one @ mu)


# ============================================================================ evolutionary models
MODELS = ("BM", "OU", "EB", "lambda", "white")
MODEL_PARAM = {"OU": "alpha", "EB": "a", "lambda": "lambda"}
MODEL_K = {"BM": 2, "white": 2, "OU": 3, "EB": 3, "lambda": 3}      # with sigma^2 and the root state, as geiger counts


def model_vcv(C: np.ndarray, model: str, p: float = 0.0) -> np.ndarray:
    """Tip covariance per unit sigma^2 under each model, geiger::fitContinuous's forms:
    BM      C, the shared path length from the root
    OU      one optimum at the root state: exp(-alpha (t_i + t_j - 2 s_ij)) (1 - exp(-2 alpha s_ij)) / (2 alpha),
            geiger's covariance form (its branch-length transform gives the same on ultrametric trees)
    EB      rate sigma^2 exp(a t) at time t from the root: (exp(a C) - 1) / a
    lambda  shared paths times lambda, root-to-tip lengths unchanged
    white   the identity: no phylogeny"""
    if model == "BM":
        return C
    if model == "white":
        return np.eye(len(C))
    if model == "lambda":
        V = C * p
        np.fill_diagonal(V, np.diag(C))
        return V
    if model == "EB":
        return C if abs(p) < 1e-12 else np.expm1(p * C) / p
    if model == "OU":
        d = np.diag(C)
        return np.exp(-p * (d[:, None] + d[None, :] - 2 * C)) * -np.expm1(-2 * p * C) / (2 * p)
    raise ValueError(model)


def profile_loglik(y: np.ndarray, V: np.ndarray) -> Tuple[float, float, float]:
    """log-likelihood of y ~ N(z0, sigma^2 V) with z0 (the GLS mean) and sigma^2 = Q / n at their maximum."""
    from scipy.linalg import solve_triangular
    n = len(y)
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return -np.inf, np.nan, np.nan
    ty = solve_triangular(L, y, lower=True)
    t1 = solve_triangular(L, np.ones(n), lower=True)
    z0 = float(t1 @ ty / (t1 @ t1))
    s2 = float(((ty - z0 * t1) ** 2).sum()) / n
    if not s2 > 0:
        return -np.inf, s2, z0
    return float(-0.5 * n * (np.log(2 * np.pi) + np.log(s2) + 1) - np.log(np.diag(L)).sum()), s2, z0


def default_bounds(C: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """geiger::fitContinuous's defaults: alpha in (exp(-500), e], a in [ln(1e-5) / depth, -1e-6], lambda in (0, 1]."""
    return {"alpha": (np.exp(-500), np.e), "a": (np.log(1e-5) / float(np.diag(C).max()), -1e-6),
            "lambda": (np.exp(-500), 1.0)}


def fit_model(y: np.ndarray, C: np.ndarray, model: str, bounds: Dict[str, Tuple[float, float]]) -> Dict:
    """Maximum-likelihood fit of one model to one variable: a grid over the model's parameter, then Brent's method
    between the grid points either side of the best (alpha on a log scale, where its likelihood surface is smooth)."""
    from scipy.optimize import minimize_scalar
    n = len(y)
    par, at_bound = np.nan, ""
    if model in MODEL_PARAM:
        name = MODEL_PARAM[model]
        lo, hi = bounds[name]
        logscale = name == "alpha"
        if logscale:            # below alpha * depth = 1e-8 OU is BM to machine precision: no need to search lower
            lo_s = np.log(max(lo, 1e-8 / float(np.diag(C).max())))
            to_p = np.exp
            grid = np.linspace(lo_s, np.log(hi), 61)
        else:
            to_p = float
            grid = np.linspace(lo, hi, 61)
        f = lambda u: -profile_loglik(y, model_vcv(C, model, to_p(u)))[0]
        vals = np.array([f(u) for u in grid])
        i = int(np.argmin(vals))
        best_u, best_v = grid[i], vals[i]
        a_, b_ = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        if b_ > a_:
            r = minimize_scalar(f, bounds=(a_, b_), method="bounded", options={"xatol": 1e-10})
            if r.fun < best_v:
                best_u, best_v = float(r.x), float(r.fun)
        par = float(to_p(best_u))
        tol = 1e-6 * (grid[-1] - grid[0])
        at_bound = "lower" if best_u - grid[0] < tol else "upper" if grid[-1] - best_u < tol else ""
        V = model_vcv(C, model, par)
    else:
        V = model_vcv(C, model)
    ll, s2, z0 = profile_loglik(y, V)
    k = MODEL_K[model]
    aicc = -2 * ll + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)
    return {"model": model, "logLik": ll, "k": k, "AICc": aicc, "sigsq": s2, "z0": z0,
            "param": MODEL_PARAM.get(model, ""), "value": par, "at_bound": at_bound}


def fit_models(y: np.ndarray, C: np.ndarray, models=MODELS, bounds=None) -> List[Dict]:
    """every model for one variable, with delta-AICc, Akaike weights, OU half-life and the lambda tests."""
    from scipy.stats import chi2
    bounds = {**default_bounds(C), **(bounds or {})}
    fits = [fit_model(y, C, m, bounds) for m in models]
    best = min(f["AICc"] for f in fits)
    w = np.array([np.exp(-0.5 * (f["AICc"] - best)) for f in fits])
    depth = float(np.diag(C).max())
    for f, wi in zip(fits, w / w.sum()):
        f["dAICc"], f["weight"] = f["AICc"] - best, float(wi)
        f["half_life"] = float(np.log(2) / f["value"]) if f["model"] == "OU" else np.nan
        f["half_life_over_depth"] = f["half_life"] / depth if f["model"] == "OU" else np.nan
        if f["model"] == "lambda":
            ll0 = profile_loglik(y, model_vcv(C, "lambda", 0.0))[0]
            ll1 = profile_loglik(y, C)[0]
            f["P_lambda_vs_0"] = float(chi2.sf(max(0.0, 2 * (f["logLik"] - ll0)), 1))
            f["P_lambda_vs_1"] = float(chi2.sf(max(0.0, 2 * (f["logLik"] - ll1)), 1))
    return fits


# ============================================================================ variable-rate Brownian motion
def tree_arrays(root: Node):
    """nodes in preorder (root first), each node's parent index (-1 for the root) and branch length, and the
    ancestry matrix B (B[i, j] = 1 when the branch above node j lies on the path from the root to node i)."""
    nodes: List[Node] = []

    def rec(n):
        nodes.append(n)
        for k in n.kids:
            rec(k)

    rec(root)
    ix = {id(n): i for i, n in enumerate(nodes)}
    parent = np.array([ix[id(n.parent)] if n.parent is not None else -1 for n in nodes])
    length = np.array([0.0 if n.parent is None else n.length for n in nodes])
    B = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        j = i
        while parent[j] >= 0:
            B[i, j] = 1.0
            j = parent[j]
    return nodes, parent, length, B


def _logmean(u: np.ndarray, v: np.ndarray):
    """mean rate along a branch whose log rate runs linearly from u (parent) to v (child), and its derivatives:
    (e^u - e^v) / (u - v), phytools' ln.mean."""
    d = u - v
    ev = np.exp(v)
    small = np.abs(d) < 1e-6
    ds = np.where(small, 1.0, d)
    phi = np.where(small, 1 + d / 2 + d * d / 6, np.expm1(ds) / ds)                     # (e^d - 1) / d
    dphi = np.where(small, 0.5 + d / 3, (np.exp(ds) * ds - np.expm1(ds)) / (ds * ds))   # its derivative
    return ev * phi, ev * dphi, ev * (phi - dphi)                                        # L, dL/du, dL/dv


def multirate_bm(root: Node, tips: List[Node], y: np.ndarray, lam: float = 1.0, maxiter: int = 20000) -> Dict:
    """Variable-rate Brownian motion (Revell 2021, phytools::multirateBM): a log rate at every node, each branch
    evolving at the mean of the rates at its two ends, the tip data normal given those rates, and the log rates
    themselves penalised as Brownian motion along the tree with weight lam. Maximised with L-BFGS-B within
    phytools' bounds (log(var(y) / tree height) +/- 10) from its start (the Brownian-motion ML rate)."""
    from scipy.optimize import minimize
    nodes, parent, length, B = tree_arrays(root)
    N = len(nodes)
    tip_ix = {id(n): i for i, n in enumerate(nodes)}
    ti = np.array([tip_ix[id(t)] for t in tips])
    A = B[ti]                                                    # tips x branches
    nr = np.arange(1, N)                                         # every node but the root (index 0)
    pen_len = np.maximum(length, 1e-12 * length.max())           # zero-length branches would make Vn singular
    Vn = (B[nr] * pen_len) @ B[nr].T                             # shared depth among the non-root nodes
    Vn_i = np.linalg.inv(Vn)
    _, Vn_logdet = np.linalg.slogdet(Vn)
    n = len(y)

    def obj(th):
        L_, dLu, dLv = _logmean(th[parent[nr]], th[nr])
        w = np.zeros(N)
        w[nr] = length[nr] * L_
        C = (A * w) @ A.T
        try:
            Lc = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return 1e300, np.zeros(N)
        Ci = np.linalg.inv(C)
        one = np.ones(n)
        a = (one @ Ci @ y) / (one @ Ci @ one)
        r = y - a
        Cr = Ci @ r
        ll1 = -0.5 * (n * np.log(2 * np.pi) + 2 * np.log(np.diag(Lc)).sum() + r @ Cr)
        z = th[nr] - th[0]
        s = Vn_i @ z
        ll2 = -0.5 * ((N - 1) * np.log(2 * np.pi) + Vn_logdet + z @ s)
        # gradients: d ll1 / d w_j = 1/2 A_j' (C^-1 r r' C^-1 - C^-1) A_j (the GLS mean drops out at its optimum)
        M = np.outer(Cr, Cr) - Ci
        gw = 0.5 * np.einsum("ij,ik,kj->j", A, M, A)
        g = np.zeros(N)
        np.add.at(g, parent[nr], gw[nr] * length[nr] * dLu)
        g[nr] += gw[nr] * length[nr] * dLv
        g[nr] += -lam * s
        g[0] += lam * s.sum()
        return -(ll1 + lam * ll2), -g

    height = float((B * length).sum(axis=1).max())
    vv = np.log(np.var(y, ddof=1) / height)
    C0 = (A * length) @ A.T
    s2_bm = profile_loglik(y, C0)[1]
    th0 = np.full(N, np.clip(np.log(s2_bm), vv - 10, vv + 10))
    res = minimize(obj, th0, jac=True, method="L-BFGS-B", bounds=[(vv - 10, vv + 10)] * N,
                   options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": 1e-13, "gtol": 1e-8})
    th = res.x
    edge_rate = np.full(N, np.nan)
    edge_rate[nr] = _logmean(th[parent[nr]], th[nr])[0]
    w = np.zeros(N)
    w[nr] = length[nr] * edge_rate[nr]
    C = (A * w) @ A.T
    one = np.ones(n); Ci = np.linalg.inv(C); a = (one @ Ci @ y) / (one @ Ci @ one); r = y - a
    ll = float(-0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(C)[1] + r @ Ci @ r))    # unpenalised, as phytools
    return {"nodes": nodes, "parent": parent, "length": length, "node_rate": np.exp(th), "edge_rate": edge_rate,
            "penalised_logLik": float(-res.fun), "logLik": ll, "k": N + 1, "converged": bool(res.success),
            "message": str(res.message), "lambda": lam}


def branch_changes(root: Node, tips: List[Node], y: np.ndarray, nodes, parent, length) -> Tuple[np.ndarray, np.ndarray]:
    """each branch's change between the Brownian-motion ancestral states (tips observed) and that change divided by
    its standard deviation under the fitted Brownian motion, sqrt(sigma^2 x branch length)."""
    inner = [n for n in nodes if n.kids]
    anc = ancestral_states(y[:, None], tips, inner)[:, 0]
    val = {id(n): float(v) for n, v in zip(inner, anc)}
    val.update({id(t): float(v) for t, v in zip(tips, y)})
    s2 = profile_loglik(y, vcv(tips, tips))[1]
    state = np.array([val[id(n)] for n in nodes])
    delta = np.full(len(nodes), np.nan)
    delta[1:] = state[1:] - state[parent[1:]]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = delta / np.sqrt(s2 * length)
    return delta, z


def clade_label(n: Node) -> str:
    t = tips_of(n)
    return t[0].name if len(t) == 1 else f"{t[0].name} .. {t[-1].name} ({len(t)} tips)"


# ============================================================================ data
def _is_number(v: str) -> bool:
    try:
        float(v); return True
    except ValueError:
        return False


def _key_forms(s: str) -> List[str]:
    """an id with and without Descriptron's annotation suffix ("x.png_3") and extension."""
    s = str(s).strip(); base = re.sub(r"\.(png|jpe?g|tiff?|bmp)_\d+$", r".\1", s, flags=re.I)
    stem = re.sub(r"\.(png|jpe?g|tiff?|bmp)$", "", base, flags=re.I)
    forms = [s, base, stem, Path(stem).name]
    # some steps write file names with spaces replaced by underscores ("Scan 001" -> "Scan_001")
    return forms + [f.replace(" ", "_") for f in forms if " " in f]


def load_group_map(path: str, group_col: Optional[str] = None) -> Dict[str, str]:
    """group labels CSV (filename + group_label/species): id -> species."""
    rows = list(csv.DictReader(open(path, newline=""), delimiter="\t" if str(path).endswith(".tsv") else ","))
    lc = {c.strip().lower(): c for c in rows[0]}
    fcol = lc.get("filename") or lc.get("file") or lc.get("specimen") or list(rows[0])[0]
    gcol = group_col or lc.get("group_label") or lc.get("species") or list(rows[0])[1]
    m: Dict[str, str] = {}
    for r in rows:
        for k in _key_forms(r[fcol]):
            m.setdefault(k, r[gcol])
    return m


def load_traits(path: str, species_col: Optional[str], species_regex: Optional[str], id_col: Optional[str],
                groups: Optional[Dict[str, str]] = None, keep_missing: bool = False,
                blank: Optional[Dict[str, set]] = None):
    """species means of every numeric column; columns missing for any species are dropped unless keep_missing
    (then a species' mean is over its specimens that have the value, NaN if none has). blank: {specimen id: set of
    cell names}: those cells' columns (named <cell>_...) are set missing for that specimen before averaging (the cells
    a specimen left empty, which the colour-homology table fills with column means; --cell_geometry)."""
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        sys.exit(f"{path}: no rows")
    cols = list(rows[0])
    if groups is not None and not (species_col and species_col in cols):     # species from a group-labels file
        idc = id_col or next((c for c in ("filename", "specimen_id", "specimen", "image", "id", "file") if c in cols), cols[0])
        sp = [next((groups[k] for k in _key_forms(r[idc]) if k in groups), None) for r in rows]
        species_col = idc                                                # the id column is not a trait
    elif species_col and species_col in cols:
        sp = [r[species_col] for r in rows]
    else:
        idc = id_col or next((c for c in ("filename", "specimen_id", "specimen", "image", "id", "file") if c in cols), cols[0])
        if not species_regex:
            sys.exit(f"{path}: give --species_col or --species_regex (no species column found)")
        rx = re.compile(species_regex)
        sp = []
        for r in rows:
            m = rx.search(r[idc])
            sp.append(m.group(1) if m else None)
    num = [c for c in cols if c != species_col and all(_is_number(r[c]) or r[c] == "" for r in rows)
           and any(r[c] != "" for r in rows)]
    M = np.array([[float(r[c]) if r[c] != "" else np.nan for c in num] for r in rows])
    if blank:
        bidc = id_col or next((c for c in ("filename", "specimen_id", "specimen", "image", "id", "file") if c in cols), cols[0])
        bk = {k: v for key, v in blank.items() for k in _key_forms(key)}
        pref = np.array([c.split("_", 1)[0] for c in num])
        for i, r in enumerate(rows):
            cs = next((bk[k] for k in _key_forms(r[bidc]) if k in bk), None)
            if cs:
                M[i, np.isin(pref, list(cs))] = np.nan
    by: Dict[str, List[int]] = {}
    for i, s in enumerate(sp):
        if s:
            by.setdefault(s, []).append(i)
    species = sorted(by)
    if not species:
        sys.exit(f"{path}: no row could be given a species (check the id column against the group labels / regex)")
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN cells of a species stay NaN
            means = np.array([np.nanmean(M[by[s]], axis=0) for s in species])
    keep = np.ones(means.shape[1], bool) if keep_missing else ~np.isnan(means).any(axis=0)
    return species, [c for c, k in zip(num, keep) if k], means[:, keep], {s: len(v) for s, v in by.items()}


def scale(M: np.ndarray, how: str) -> np.ndarray:
    if how == "standardize":
        sd = M.std(axis=0, ddof=1)
        ok = sd > 0
        return (M[:, ok] - M[:, ok].mean(0)) / sd[ok]
    if how == "log":
        if (M <= 0).any():
            sys.exit("--scale log needs positive values")
        return np.log(M)
    return M


# ============================================================================ plot
def plot_morphospace(pc_tips, pc_nodes, tips, nodes, ev, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = {id(t): pc_tips[i] for i, t in enumerate(tips)}
    pos.update({id(n): pc_nodes[i] for i, n in enumerate(nodes)})
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n in nodes:
        for k in n.kids:
            a, b = pos[id(n)], pos[id(k)]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#6b7780", lw=0.9, zorder=1)
    ax.scatter(pc_nodes[:, 0], pc_nodes[:, 1], s=14, color="white", edgecolor="#6b7780", zorder=2, label="ancestral states")
    ax.scatter(pc_tips[:, 0], pc_tips[:, 1], s=30, color="#0072B2", zorder=3, label="species means")
    for i, t in enumerate(tips):
        ax.annotate(t.name, pc_tips[i, :2], xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_rate_tree(nodes, parent, length, edge_rate, path, title):
    """rectangular tree, each branch coloured by its rate relative to the median branch (log scale: slow blue,
    median grey, fast red)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    N = len(nodes)
    x = np.zeros(N)
    for i in range(1, N):
        x[i] = x[parent[i]] + length[i]
    tips = [i for i in range(N) if not nodes[i].kids]
    yv = np.zeros(N)
    yv[tips] = np.arange(len(tips))[::-1]
    for i in range(N - 1, -1, -1):                      # preorder reversed: children before parents
        if nodes[i].kids:
            kids = [j for j in range(N) if parent[j] == i]
            yv[i] = np.mean(yv[kids])
    rel = np.log10(edge_rate / np.nanmedian(edge_rate[1:]))
    lim = max(0.5, float(np.nanmax(np.abs(rel[1:]))))
    cmap = LinearSegmentedColormap.from_list("rate", ["#2166ac", "#b8bec4", "#b2182b"])
    norm = Normalize(-lim, lim)
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.13 * len(tips) + 1.2)))
    for i in range(1, N):
        p = parent[i]
        c = cmap(norm(rel[i]))
        ax.plot([x[p], x[i]], [yv[i], yv[i]], color=c, lw=1.6, solid_capstyle="butt")
        ax.plot([x[p], x[p]], [yv[p], yv[i]], color="#8a9399", lw=0.6)
    for i in tips:
        ax.text(x[i] + 0.01 * x.max(), yv[i], nodes[i].name, va="center", fontsize=5.5)
    ax.set_xlim(0, x.max() * 1.45); ax.set_yticks([]); ax.set_xlabel("branch length")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("log10 rate / median rate", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:80]


def _norm_name(s: str) -> str:
    """spaces and underscores count as the same character when tips are matched to species."""
    return re.sub(r"[\s_]+", "_", str(s).strip())


def load_tree(path: str, tip_regex: Optional[str] = None) -> Tuple[Node, List[str]]:
    """the tree with tips renamed to species (--tip_regex) and duplicate species pruned to their first tip."""
    root = parse_newick(Path(path).read_text().strip().split(";")[0] + ";")
    rx = re.compile(tip_regex) if tip_regex else None
    notes, seen, keep = [], set(), set()
    for t in tips_of(root):
        if rx:
            m = rx.search(t.name)
            if not m:
                notes.append(f"tip '{t.name}' does not match --tip_regex - kept as it is")
            else:
                t.name = m.group(1)
        t.name = _norm_name(t.name)
        if t.name in seen:
            notes.append(f"species '{t.name}' is on several tips - the first is kept")
            t.name = f"__dup__{id(t)}"
        else:
            seen.add(t.name)
        keep.add(t.name)
    if any(n.startswith("__dup__") for n in keep):
        root = prune(root, {n for n in keep if not n.startswith("__dup__")})
    return root, notes


def _copy_tree(n: Node, parent: Optional[Node] = None) -> Node:
    c = Node(n.name, n.length)
    c.parent = parent
    c.kids = [_copy_tree(k, c) for k in n.kids]
    return c


# ============================================================================ colour-pattern trees
COLOUR_CELLS_DEFAULT = r"^(?P<cell>r\d+c\d+)_(?P<ch>[Lab])_mean_abs$"


def ciede2000(lab1, lab2, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma, Wu & Dalal 2005, Color Res. Appl. 30: 21), element-wise over arrays whose
    last axis is (L*, a*, b*). Hue angles follow the paper's conventions (h' = 0 when a' = b' = 0; the hue difference and
    mean hue are taken as 0 / the plain sum when either chroma is 0)."""
    lab1, lab2 = np.asarray(lab1, float), np.asarray(lab2, float)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    Cb = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    zero = C1p * C2p == 0
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(zero, 0.0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp, Cbp, hs = (L1 + L2) / 2, (C1p + C2p) / 2, h1p + h2p
    hbp = np.where(zero, hs, np.where(np.abs(h1p - h2p) <= 180, hs / 2,
                                      np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-((hbp - 275) / 25) ** 2)
    RC = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC, SH = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC
    tL, tC, tH = dLp / (kL * SL), dCp / (kC * SC), dHp / (kH * SH)
    return np.sqrt(np.maximum(tL ** 2 + tC ** 2 + tH ** 2 + RT * tC * tH, 0.0))


def de76(lab1, lab2) -> np.ndarray:
    """CIE 1976 colour difference: Euclidean distance in CIELAB."""
    return np.sqrt(((np.asarray(lab1, float) - np.asarray(lab2, float)) ** 2).sum(-1))


def colour_cell_index(cols: List[str], regex: str = COLOUR_CELLS_DEFAULT) -> Tuple[List[str], np.ndarray]:
    """grid cells with all three of L*, a*, b* among `cols`: (cell names, index array cells x 3 into cols). The regex
    needs the named groups 'cell' and 'ch' (ch = L, a or b)."""
    rx = re.compile(regex)
    found: Dict[str, Dict[str, int]] = {}
    for i, c in enumerate(cols):
        m = rx.search(c)
        if m:
            found.setdefault(m.group("cell"), {})[m.group("ch")] = i
    cells = [c for c in found if all(k in found[c] for k in "Lab")]
    cells.sort(key=lambda c: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", c)])
    return cells, np.array([[found[c][k] for k in "Lab"] for c in cells], dtype=int).reshape(-1, 3)


def per_cell_distances(lab: np.ndarray, metric: str = "de2000") -> np.ndarray:
    """lab: species x cells x 3 (species-mean L*, a*, b* per cell, NaN where a cell is missing).
    Returns cells x species x species colour differences (NaN where either species lacks the cell)."""
    f = ciede2000 if metric == "de2000" else de76
    d = f(lab[:, None, :, :], lab[None, :, :, :])               # species x species x cells
    return np.transpose(d, (2, 0, 1))


def colour_distance(Dc: np.ndarray, cells: Optional[np.ndarray] = None) -> np.ndarray:
    """species distance = mean over the cells present in both species of the per-cell colour difference; `cells` is
    a vector of cell indices (repeats count several times: a bootstrap sample)."""
    X = Dc if cells is None else Dc[cells]
    ok = np.isfinite(X)
    cnt = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(ok, X, 0.0).sum(0) / cnt
    np.fill_diagonal(D, 0.0)
    return D


def upgma(D: np.ndarray, names: List[str]) -> Node:
    """UPGMA (average linkage) on a distance matrix: node heights are half the merging distance, as
    phangorn::upgma / hclust(method = "average") + as.phylo."""
    n = len(names)
    W = np.array(D, float).copy()
    np.fill_diagonal(W, np.inf)
    nodes = [Node(names[i]) for i in range(n)]
    height, size = np.zeros(n), np.ones(n)
    active = np.ones(n, bool)
    for _ in range(n - 1):
        M = np.where(active[:, None] & active[None, :], W, np.inf)
        i, j = np.unravel_index(int(np.argmin(M)), M.shape)
        i, j = min(i, j), max(i, j)
        h = W[i, j] / 2
        u = Node()
        for k in (i, j):
            nodes[k].length = h - height[k]; nodes[k].parent = u; u.kids.append(nodes[k])
        new = (size[i] * W[i] + size[j] * W[j]) / (size[i] + size[j])
        W[i, :], W[:, i] = new, new
        W[i, i] = np.inf
        active[j] = False
        W[j, :], W[:, j] = np.inf, np.inf
        nodes[i], height[i], size[i] = u, h, size[i] + size[j]
    root = nodes[int(np.flatnonzero(active)[0])]
    root.length = 0.0
    _set_depths(root)
    return root


def nj(D: np.ndarray, names: List[str]) -> Node:
    """neighbour joining (Saitou & Nei 1987, Studier & Keppler 1988's form), as ape::nj: the unrooted tree is returned
    with a three-way basal node (the last three clusters joined)."""
    D = np.array(D, float).copy()
    nodes = [Node(nm) for nm in names]
    act = list(range(len(names)))
    if len(act) < 3:
        raise ValueError("neighbour joining needs at least three species")
    while len(act) > 3:
        m = len(act)
        Da = D[np.ix_(act, act)]
        S = Da.sum(1)
        Q = (m - 2) * Da - S[:, None] - S[None, :]
        np.fill_diagonal(Q, np.inf)
        a, b = np.unravel_index(int(np.argmin(Q)), Q.shape)
        a, b = min(a, b), max(a, b)
        i, j = act[a], act[b]
        li = Da[a, b] / 2 + (S[a] - S[b]) / (2 * (m - 2))
        lj = Da[a, b] - li
        u = Node()
        for k, l in ((i, li), (j, lj)):
            nodes[k].length = l; nodes[k].parent = u; u.kids.append(nodes[k])
        du = (D[i] + D[j] - D[i, j]) / 2
        D[i, :], D[:, i] = du, du
        D[i, i] = 0.0
        nodes[i] = u
        act.remove(j)
    a, b, c = act
    r = Node()
    for k, (x, y, z) in ((a, (a, b, c)), (b, (b, a, c)), (c, (c, a, b))):
        nodes[k].length = (D[x, y] + D[x, z] - D[y, z]) / 2
        nodes[k].parent = r; r.kids.append(nodes[k])
    _set_depths(r)
    return r


def clade_sets(root: Node) -> List[Tuple[Node, frozenset]]:
    """every internal node but the root, with the set of its descendant tips."""
    return [(n, frozenset(t.name for t in tips_of(n))) for n in internal_nodes(root) if n is not root]


def split_key(s: frozenset, names: frozenset) -> Optional[frozenset]:
    """a bipartition as the side without the alphabetically first species; None if trivial (fewer than 2 tips a side)."""
    if min(names) in s:
        s = names - s
    return s if 2 <= len(s) <= len(names) - 2 else None


def tree_keys(root: Node, rooted: bool) -> set:
    names = frozenset(t.name for t in tips_of(root))
    if rooted:
        return {s for _, s in clade_sets(root)}
    return {k for k in (split_key(s, names) for _, s in clade_sets(root)) if k is not None}


def build_colour_tree(D: np.ndarray, names: List[str], method: str) -> Node:
    return upgma(D, names) if method == "upgma" else nj(D, names)


def bootstrap_support(Dc: np.ndarray, names: List[str], method: str, B: int, rng,
                      ref: Optional[Node] = None) -> Tuple[Node, Dict[int, float], np.ndarray, int]:
    """the colour tree from all cells and, for each of its internal nodes, the share of B trees built from cells
    resampled with replacement that contain it (UPGMA: the same clade; NJ, unrooted: the same bipartition).
    Returns (tree, {id(node): support}, the B x cells resampled index matrix, number of pair distances that had no
    cell in a replicate and kept their all-cell value)."""
    D0 = colour_distance(Dc)
    tree = ref if ref is not None else build_colour_tree(D0, names, method)
    nc = Dc.shape[0]
    idx = rng.integers(0, nc, size=(B, nc))
    counts = support_counts(Dc, names, method, idx, tree)
    nfill = counts.pop("__filled__")
    return tree, {k: v / B for k, v in counts.items()}, idx, nfill


def support_counts(Dc: np.ndarray, names: List[str], method: str, idx: np.ndarray, tree: Node) -> Dict:
    """how many of the trees built from the resampled cell sets `idx` (rows) contain each internal node of `tree`."""
    return support_counts_fn(lambda row: colour_distance(Dc, row), colour_distance(Dc), names, method, idx, tree)


def to_newick(n: Node, labels: Optional[Dict[int, str]] = None, fmt: str = "{:.10g}") -> str:
    labels = labels or {}
    if not n.kids:
        s = n.name
    else:
        s = "(" + ",".join(to_newick(k, labels, fmt) for k in n.kids) + ")" + labels.get(id(n), "")
    return s + (":" + fmt.format(n.length) if n.parent is not None else "")


def patristic(tips: List[Node]) -> np.ndarray:
    d = np.array([t.depth for t in tips])
    return d[:, None] + d[None, :] - 2 * vcv(tips, tips)


def mantel(D1: np.ndarray, D2: np.ndarray, method: str = "pearson", perms: int = 999, rng=None) -> Dict:
    """Mantel test as vegan::mantel: r between the lower triangles (Pearson, or Spearman on ranks), rows and columns of
    D1 permuted together, P = (#{r_perm >= r} + 1) / (perms + 1). For Spearman, distances within 1e-8 x the largest
    of each other count as tied (_tie), so that distances equal on the tree (all pairs across one node of an
    ultrametric tree) are not ordered by floating-point noise."""
    from scipy.stats import rankdata
    rng = rng or np.random.default_rng(1)
    n = len(D1)
    iu = np.triu_indices(n, 1)
    X, Y = np.array(D1, float), np.array(D2, float)
    if method == "spearman":                          # ranks travel with their entries when D1 is permuted
        X, Y = np.zeros((n, n)), np.zeros((n, n))
        X[iu] = rankdata(_tie(np.asarray(D1, float)[iu])); Y[iu] = rankdata(_tie(np.asarray(D2, float)[iu]))
        X, Y = X + X.T, Y + Y.T
    y = Y[iu]
    r = float(np.corrcoef(X[iu], y)[0, 1])
    rp = np.empty(perms)
    for k in range(perms):
        p = rng.permutation(n)
        rp[k] = np.corrcoef(X[np.ix_(p, p)][iu], y)[0, 1]
    return {"r": r, "P": float(((rp >= r).sum() + 1) / (perms + 1)), "permutations": perms}


def _tie(x: np.ndarray, rel: float = 1e-8) -> np.ndarray:
    """values whose sorted neighbours are within rel x max|x| are given the first value of their run (ties)."""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="stable")
    xs = x[o].copy()
    tol = rel * float(np.abs(x).max() or 1.0)
    for k in range(1, len(xs)):
        if xs[k] - xs[k - 1] <= tol or abs(x[o[k]] - xs[k - 1]) <= tol:
            xs[k] = xs[k - 1]
    out = np.empty_like(x); out[o] = xs
    return out


def _signif(x: np.ndarray, digits: int = 10) -> np.ndarray:
    """R's signif(): round each value to `digits` significant digits."""
    return np.array([float(f"{v:.{digits}g}") for v in np.ravel(x)]).reshape(np.shape(x))


def rf_distance(t1: Node, t2: Node) -> Dict:
    """Robinson-Foulds distance between two trees on the same tips, unrooted, as phangorn::RF.dist (number of
    bipartitions in one tree but not the other), and normalised by its maximum 2(n - 3)."""
    k1, k2 = tree_keys(t1, False), tree_keys(t2, False)
    n = len(tips_of(t1))
    rf = len(k1 ^ k2)
    return {"RF": rf, "RF_normalised": rf / (2 * (n - 3)) if n > 3 else float("nan"), "shared_splits": len(k1 & k2)}


# ---------------------------------------------------------------------------- tanglegram
def crossings(order_left: List[str], order_right: List[str]) -> int:
    """line crossings when the same species are joined between two tip orders (pairs in opposite order)."""
    pos = {s: i for i, s in enumerate(order_right)}
    p = [pos[s] for s in order_left if s in pos]
    return int(sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j]))


def _orient(root: Node, pos: Dict[str, float]):
    """rotate the children of every node of `root` to minimise crossings with the other tree's tip positions `pos`.
    A pair of species is crossed or not depending only on the order of the children at their most recent common
    ancestor, so choosing each node's best order is the exact one-sided optimum; ties keep the smaller mean
    position first (shorter lines, as phytools::cophylo's squared-distance criterion)."""
    def tipsets(n):
        return [[pos[t.name] for t in tips_of(k) if t.name in pos] for k in n.kids]

    for n in internal_nodes(root):
        sets = tipsets(n)

        def cost(a, b):                                  # crossings among a's and b's tips with a placed first
            return sum(1 for x in a for y in b if x > y)

        def bary(a):
            return np.mean(a) if a else np.inf
        order = sorted(range(len(n.kids)), key=lambda i: bary(sets[i]))
        changed = True
        while changed:                                   # adjacent swaps (exact for two children)
            changed = False
            for k in range(len(order) - 1):
                i, j = order[k], order[k + 1]
                if cost(sets[j], sets[i]) < cost(sets[i], sets[j]):
                    order[k], order[k + 1] = j, i; changed = True
        n.kids = [n.kids[i] for i in order]


def untangle(left: Node, right: Node, max_rounds: int = 50) -> int:
    """rotate the nodes of both trees (in place) to reduce line crossings, alternating sides until no change (as
    phytools::cophylo(rotate = TRUE), but counting crossings rather than squared tip offsets); both starting sides are
    tried and the better result kept. Returns the crossings left."""
    allnodes = internal_nodes(left) + internal_nodes(right)
    start = {id(n): list(n.kids) for n in allnodes}
    ncross = lambda: crossings([t.name for t in tips_of(left)], [t.name for t in tips_of(right)])
    results = []
    for first in ("left", "right"):
        for n in allnodes:
            n.kids = list(start[id(n)])
        best = ncross()
        for _ in range(max_rounds):
            for side in ((left, right), (right, left)) if first == "left" else ((right, left), (left, right)):
                _orient(side[0], {t.name: i for i, t in enumerate(tips_of(side[1]))})
            c = ncross()
            if c >= best:
                best = min(best, c); break
            best = c
        results.append((best, {id(n): list(n.kids) for n in allnodes}))
    best, kids = min(results, key=lambda r: r[0])
    for n in allnodes:
        n.kids = kids[id(n)]
    return best


# ---------------------------------------------------------------------------- convergence
class ConvTree:
    """index arrays of a tree for the convergence statistics: nodes in preorder, parents, heights, tip and internal
    node indices, and the weights giving the Brownian-motion (GLS = maximum-likelihood) ancestral states from the tips,
    the same values as phytools::fastAnc."""

    def __init__(self, root: Node):
        self.nodes, self.parent, self.length, B = tree_arrays(root)
        N = len(self.nodes)
        self.height = np.zeros(N)
        for i in range(1, N):
            self.height[i] = self.height[self.parent[i]] + self.length[i]
        self.tip_ix = np.array([i for i in range(N) if not self.nodes[i].kids])
        self.inner_ix = np.array([i for i in range(N) if self.nodes[i].kids])
        self.tips = [self.nodes[i] for i in self.tip_ix]
        self.names = [t.name for t in self.tips]
        inner = [self.nodes[i] for i in self.inner_ix]
        self.C = vcv(self.tips, self.tips)
        Ci = np.linalg.inv(self.C)
        one = np.ones(len(self.tips))
        self.w = Ci @ one / (one @ Ci @ one)
        self.W = vcv(inner, self.tips) @ Ci
        self.Wr = 1 - self.W.sum(1)
        self.desc = [np.flatnonzero((B[:, v] > 0) & (np.arange(N) != v)) if v else np.arange(1, N) for v in range(N)]
        self.N = N

    def all_values(self, Y: np.ndarray) -> np.ndarray:
        """tips and ancestral states, in node (preorder) order."""
        out = np.zeros((self.N, Y.shape[1]))
        out[self.tip_ix] = Y
        out[self.inner_ix] = self.W @ Y + self.Wr[:, None] * (self.w @ Y)[None, :]
        return out

    def path(self, i: int, stop: int) -> List[int]:
        p = [i]
        while p[-1] != stop:
            p.append(int(self.parent[p[-1]]))
        return p

    def mrca(self, i: int, j: int) -> int:
        a = set(self.path(i, 0))
        x = j
        while x not in a:
            x = int(self.parent[x])
        return x

    def pair(self, ti: int, tj: int) -> Dict:
        """lineages of two tips (node indices) up to and including their most recent common ancestor, and for every
        internal node on them the branch of the other lineage spanning its height, as convevol's calcCsCt."""
        m = self.mrca(ti, tj)
        u1, u2 = self.path(ti, m), self.path(tj, m)
        edges = [(int(self.parent[c]), c) for c in u1[:-1] + u2[:-1]]
        interp = []
        for v in u1[1:-1] + u2[1:-1]:
            h = self.height[v]
            sp = [(p, c) for p, c in edges if p != v and c != v and self.height[p] < h < self.height[c]]
            if sp:
                p, c = sp[0]
                interp.append((v, p, c, (h - self.height[p]) / (self.height[c] - self.height[p])))
            else:
                interp.append((v, -1, -1, np.nan))
        sub_edges = self.desc[m]
        return {"mrca": m, "u1": np.array(u1), "u2": np.array(u2), "interp": interp,
                "sub_child": sub_edges, "sub_parent": self.parent[sub_edges], "desc": self.desc[m]}


def _pairs(T: ConvTree, focal: List[str], groups: Optional[Dict[str, str]] = None) -> List[Dict]:
    pos = {nm: int(T.tip_ix[k]) for k, nm in enumerate(T.names)}
    out = []
    for a in range(len(focal)):
        for b in range(a + 1, len(focal)):
            if groups is not None and groups[focal[a]] == groups[focal[b]]:
                continue
            out.append(T.pair(pos[focal[a]], pos[focal[b]]))
    return out


def conv_stats(T: ConvTree, Yall: np.ndarray, pairs: List[Dict], ct4: str = "clade") -> Tuple[np.ndarray, np.ndarray]:
    """Stayton's convergence measures for each pair of focal tips, as convevol 2.2.1, averaged over the pairs.
    C1-C4 (calcConv / calcCs): Dmax = the largest distance between any node of one lineage (tip, ancestral nodes and
      the MRCA) and any node of the other; C1 = 1 - Dtip / Dmax, C2 = Dmax - Dtip, C3 = C2 / (phenotypic path length of
      both lineages), C4 = C2 / (sum of the distances between ALL pairs of nodes descending from the MRCA, as
      convevol 2.2.1 codes it).
    Ct1-Ct4 (calcConvCt / calcCsCt, Grossnickle et al. 2024): Dmax is taken only between each ancestral node of one
      lineage and the other lineage at the same time (linear interpolation along its branch), so ancestors of
      different ages are never compared; Ct4 divides by the summed change along all branches of the MRCA's clade.
    Pairs whose lineages have no ancestral node give convevol's values (Ct1 = 1, Ct2-Ct4 = -Inf).
    ct4 = "clade" (default) sums the change along the branches of the MRCA's clade, the definition. ct4 = "convevol"
    reproduces convevol 2.2.1's code, which indexes the whole tree's edge table instead of the clade's
    (ptraits / ctraits built from `edge`, not `C4.edge`) and so sums the change along the FIRST m branches of the tree
    in its cladewise edge order, m being the number of branches in the clade; Ct1-Ct3 are not affected."""
    diff = Yall[:, None, :] - Yall[None, :, :]
    Dall = np.sqrt((diff ** 2).sum(-1))
    Cs, Ct = [], []
    for p in pairs:
        u1, u2 = p["u1"], p["u2"]
        dtip = Dall[u1[0], u2[0]]
        lin = Dall[u1[:-1], u1[1:]].sum() + Dall[u2[:-1], u2[1:]].sum()
        d = p["desc"]
        dmax = Dall[np.ix_(u1, u2)].max()
        tot_all = Dall[np.ix_(d, d)].sum() / 2
        c2 = dmax - dtip
        Cs.append([1 - dtip / dmax, c2, c2 / lin, c2 / tot_all])
        diffs = [np.sqrt(((Yall[v] - ((1 - f) * Yall[pp] + f * Yall[cc])) ** 2).sum()) for v, pp, cc, f in p["interp"]
                 if pp >= 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            dmt = max(diffs) if diffs else -np.inf
            if ct4 == "convevol":                     # first m edges of the cladewise edge table = preorder nodes 1..m
                m = len(p["sub_child"]); kk = np.arange(1, m + 1)
                tot = Dall[kk, T.parent[kk]].sum()
            else:
                tot = Dall[p["sub_child"], p["sub_parent"]].sum()
            c2t = dmt - dtip
            Ct.append([1 - dtip / dmt, c2t, c2t / lin, c2t / tot])
    Cs, Ct = np.array(Cs).reshape(-1, 4), np.array(Ct).reshape(-1, 4)
    with np.errstate(invalid="ignore"):
        return (Cs.mean(0) if len(Cs) else np.full(4, np.nan)), (Ct.mean(0) if len(Ct) else np.full(4, np.nan))


def bm_rate_matrix_lambda0(T: ConvTree, Y: np.ndarray) -> np.ndarray:
    """the rate matrix convevol's convSig / convSigCt simulate with: phytools::phyl.vcv(X, vcv(tree), lambda = 0),
    i.e. the covariance of the tips about their 1/root-to-tip-weighted mean, ignoring shared branches."""
    d = np.diag(T.C)
    a = (Y / d[:, None]).sum(0) / (1 / d).sum()
    R = Y - a
    return (R / d[:, None]).T @ R / (len(Y) - 1)


def _psd_sqrt(R: np.ndarray) -> np.ndarray:
    ev, V = np.linalg.eigh((R + R.T) / 2)
    return V * np.sqrt(np.clip(ev, 0, None))


def convergence_test(T: ConvTree, Y: np.ndarray, focal: List[str], lineages: Optional[Dict[str, str]],
                     nsim: int, rng, ct4: str = "clade") -> Dict:
    """C1-C4 over all pairs of focal species (convevol::convSig) and Ct1-Ct4 over pairs from different molecular
    lineages (convevol::convSigCt with groups); significance as convevol: nsim data sets simulated under multivariate
    Brownian motion (root 0, rate matrix phyl.vcv(lambda = 0)), P = #{simulated >= observed} / nsim."""
    pa, pb = _pairs(T, focal), (_pairs(T, focal, lineages) if lineages is not None else _pairs(T, focal))
    Yall = T.all_values(Y)
    obs_c, _ = conv_stats(T, Yall, pa, ct4)
    _, obs_t = conv_stats(T, Yall, pb, ct4)
    Lc = np.linalg.cholesky(T.C)
    Lr = _psd_sqrt(bm_rate_matrix_lambda0(T, Y))
    sc, st = np.zeros((nsim, 4)), np.zeros((nsim, 4))
    for s in range(nsim):
        Ys = Lc @ rng.standard_normal((len(Y), Lr.shape[1])) @ Lr.T
        Ya = T.all_values(Ys)
        sc[s] = conv_stats(T, Ya, pa, ct4)[0]
        st[s] = conv_stats(T, Ya, pb, ct4)[1]
    return {"C": obs_c, "P_C": (sc >= obs_c).sum(0) / nsim, "Ct": obs_t, "P_Ct": (st >= obs_t).sum(0) / nsim,
            "n_pairs_C": len(pa), "n_pairs_Ct": len(pb), "sim_C": sc, "sim_Ct": st}


def wheatsheaf(X: np.ndarray, C: np.ndarray, focal: np.ndarray, se: bool = True) -> Dict:
    """Wheatsheaf index (Arbuckle, Bennett & Speed 2014) as windex 2.1.0 windex(): traits divided by their standard
    errors (sd / sqrt(n)), Euclidean distances penalised by phylogenetic proximity, d / (1 - ln(p + 0.01)) with p the
    shared path length of the tree scaled to height 1, w = mean over all pairs / mean over focal pairs (diagonals
    included, as windex). The 95% interval is windex's: quantiles of the n leave-one-species-out values."""
    X = np.asarray(X, float).reshape(len(X), -1)
    n = len(X)
    f = np.asarray(focal, bool)
    if se:
        X = X / (X.std(axis=0, ddof=1) / np.sqrt(n))
    P = C / np.diag(C).max()
    Dm = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)) / (1 - np.log(P + 0.01))
    np.fill_diagonal(Dm, 0.0)
    nf = int(f.sum())
    tot, totf = Dm.sum(), Dm[np.ix_(f, f)].sum()
    with np.errstate(divide="ignore", invalid="ignore"):       # identical focal phenotypes: Inf, as windex
        return _wheatsheaf_tail(Dm, f, n, nf, tot, totf)


def _wheatsheaf_tail(Dm, f, n, nf, tot, totf) -> Dict:
    w = (tot / n ** 2) / (totf / nf ** 2)
    jack = np.empty(n)
    for i in range(n):
        da = (tot - 2 * Dm[i].sum()) / (n * n - 2 * (n - 1))
        if f[i]:
            dfj = (totf - 2 * Dm[i, f].sum()) / (nf * nf - 2 * (nf - 1))
        else:
            dfj = totf / nf ** 2
        jack[i] = da / dfj
    return {"w": float(w), "low95": quantile7(jack, 0.025), "up95": quantile7(jack, 0.975), "jackknife": jack}


def quantile7(x: np.ndarray, q: float) -> float:
    """R's quantile(type = 7), which keeps Inf where two infinite neighbours meet (numpy gives NaN): with two focal
    species, leaving one out makes the focal mean 0 and the index infinite, as in windex."""
    x = np.sort(np.asarray(x, float))
    h = (len(x) - 1) * q
    lo, hi = int(np.floor(h)), int(np.ceil(h))
    if x[hi] == x[lo]:
        return float(x[lo])
    return float((1 - (h - lo)) * x[lo] + (h - lo) * x[hi])


def test_wheatsheaf(X: np.ndarray, C: np.ndarray, focal: np.ndarray, reps: int, rng, se: bool = True) -> Dict:
    """windex::test.windex: the index against reps data sets in which the species' trait rows are resampled with
    replacement (the focal labels and the tree stay); P = #{resampled >= observed} / reps."""
    r = wheatsheaf(X, C, focal, se)
    X = np.asarray(X, float).reshape(len(X), -1)
    boot = np.array([wheatsheaf(X[rng.integers(0, len(X), len(X))], C, focal, True)["w"] for _ in range(reps)])
    r.update(P=float((boot >= r["w"]).sum() / reps), boot=boot, reps=reps)
    return r


def molecular_lineages(root: Node, members: List[str]) -> Dict[str, str]:
    """each member's lineage: the largest clade of the (molecular) tree containing it whose tips are all members."""
    mem = set(members)
    tipnode = {t.name: t for t in tips_of(root)}
    out = {}
    for s in members:
        x = tipnode[s]
        while x.parent is not None and {t.name for t in tips_of(x.parent)} <= mem:
            x = x.parent
        out[s] = "+".join(sorted(t.name for t in tips_of(x)))
    return out


def candidate_groups(colour_tree: Node, support: Dict[int, float], mol_tree: Node, min_support: float,
                     max_size: int) -> List[Dict]:
    """colour-tree clades with bootstrap support >= min_support whose species are not a clade of the molecular tree."""
    moltips = {t.name: t for t in tips_of(mol_tree)}
    out = []
    for n, s in clade_sets(colour_tree):
        sup = support.get(id(n), np.nan)
        if not (sup >= min_support) or len(s) > max_size:
            continue
        mem = sorted(s)
        anc = moltips[mem[0]]
        while not s <= {t.name for t in tips_of(anc)}:
            anc = anc.parent
        mrca_tips = [t.name for t in tips_of(anc)]
        if len(mrca_tips) == len(s):
            continue                                      # a clade on the molecular tree too: not convergence
        lin = molecular_lineages(mol_tree, mem)
        out.append({"members": mem, "support": float(sup), "lineages": lin, "mrca_tips": mrca_tips})
    return out


def lineage_sisters(root: Node, lineage: str) -> str:
    """the sister group, on the molecular tree, of a lineage given as 'A+B+...'."""
    mem = set(lineage.split("+"))
    for n in [root] + [x for x in internal_nodes(root)] + tips_of(root):
        if {t.name for t in tips_of(n)} == mem and n.parent is not None:
            sis = [k for k in n.parent.kids if k is not n]
            return "; ".join(clade_label(k) for k in sis)
    return ""


# ---------------------------------------------------------------------------- colour figures
def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB (D65) to sRGB in [0, 1], for drawing."""
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    finv = lambda t: np.where(t > 6 / 29, t ** 3, 3 * (6 / 29) ** 2 * (t - 4 / 29))
    xyz = np.stack([0.95047 * finv(fx), finv(fy), 1.08883 * finv(fz)], -1)
    M = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    rgb = xyz @ M.T
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1)


def pattern_thumbnails(lab: np.ndarray, cells: List[str]) -> Optional[List[np.ndarray]]:
    """each species' mean colour per grid cell drawn as a small image (cells named r<row>c<col>); None otherwise."""
    rc = [re.match(r"^r(\d+)c(\d+)$", c) for c in cells]
    if not all(rc):
        return None
    r = np.array([int(m.group(1)) for m in rc]); c = np.array([int(m.group(2)) for m in rc])
    r -= r.min(); c -= c.min()
    out = []
    for s in range(lab.shape[0]):
        img = np.ones((r.max() + 1, c.max() + 1, 4)); img[..., 3] = 0
        rgb = lab_to_rgb(lab[s])
        ok = np.isfinite(lab[s]).all(-1)
        img[r[ok], c[ok], :3] = rgb[ok]; img[r[ok], c[ok], 3] = 1
        out.append(img)
    return out


def _layout(root: Node, depth_axis: bool = True) -> Tuple[Dict[int, float], Dict[int, float]]:
    """x = distance from the root, y = tip rank (0 = first tip) with internal nodes at their children's midpoint."""
    tips = tips_of(root)
    y = {id(t): float(i) for i, t in enumerate(tips)}
    x = {}

    def rec(n, d):
        x[id(n)] = d
        for k in n.kids:
            rec(k, d + k.length)
        if n.kids:
            y[id(n)] = (y[id(n.kids[0])] + y[id(n.kids[-1])]) / 2
    rec(root, 0.0)
    return x, y


def _draw_tree(ax, root: Node, x: Dict, y: Dict, flip_x: bool = False, vertical: bool = False, lw: float = 0.9,
               color: str = "#4a5560", xscale: float = 1.0, xoff: float = 0.0, bcol=None, blw: Optional[float] = None):
    """rectangular tree on ax. Horizontal: tips at the right (or left with flip_x). Vertical: root at the top, the
    tree's x becoming the downward axis and y the column. bcol(node) -> colour (or None) colours the branch leading
    to each node (width blw); the connecting vertical lines keep `color`."""
    def X(n):
        v = x[id(n)] * xscale
        return xoff - v if flip_x else xoff + v
    for n in internal_nodes(root):
        ys = [y[id(k)] for k in n.kids]
        for k in n.kids:
            c = (bcol(k) if bcol is not None else None) or color
            w = blw if (bcol is not None and blw) else lw
            if vertical:
                ax.plot([y[id(k)], y[id(k)]], [X(n), X(k)], color=c, lw=w, solid_capstyle="butt")
            else:
                ax.plot([X(n), X(k)], [y[id(k)], y[id(k)]], color=c, lw=w, solid_capstyle="butt")
        if vertical:
            ax.plot([min(ys), max(ys)], [X(n), X(n)], color=color, lw=lw)
        else:
            ax.plot([X(n), X(n)], [min(ys), max(ys)], color=color, lw=lw)


def _support_text(s: float) -> str:
    return f"{100 * s:.0f}"


def plot_colour_tree(tree: Node, support: Dict[int, float], thumbs: Optional[Dict[str, np.ndarray]], path,
                     title: str, xlabel: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tips = tips_of(tree)
    n = len(tips)
    x, y = _layout(tree)
    H = max(x.values()) or 1.0
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.24 * n + 1.3)))
    _draw_tree(ax, tree, x, y)
    for nd in internal_nodes(tree):
        s = support.get(id(nd))
        if s is not None and nd is not tree:
            ax.text(x[id(nd)] - 0.008 * H, y[id(nd)] - 0.12, _support_text(s), ha="right", va="bottom", fontsize=5.5,
                    color="#1b5e8a" if s >= 0.7 else "#9aa3aa", fontweight="bold" if s >= 0.95 else "normal")
    tx = max(x[id(t)] for t in tips)
    for t in tips:
        xx = tx + 0.02 * H
        if thumbs is not None and t.name in thumbs:
            img = thumbs[t.name]; h, w = img.shape[:2]; asp = w / h
            ax.imshow(img, extent=(xx, xx + 0.85 * asp * 0.075 * H, y[id(t)] + 0.42, y[id(t)] - 0.42),
                      interpolation="nearest", zorder=3, aspect="auto")
            xx += 0.85 * asp * 0.075 * H + 0.015 * H
        ax.plot([x[id(t)], tx], [y[id(t)], y[id(t)]], color="#d0d5d9", lw=0.5, ls=":")
        ax.text(xx, y[id(t)], t.name.replace("_", " "), va="center", fontsize=6.5, style="italic")
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(-0.02 * H, H * 1.75)
    ax.set_xticks([v for v in ax.get_xticks() if 0 <= v <= H * 1.0001])
    ax.spines["bottom"].set_bounds(0, H)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


_TIP_IMAGE_CACHE: Dict[str, Dict] = {}


def load_tip_images(a, set_name: str, names: List[str]) -> Optional[Dict[str, np.ndarray]]:
    """--tip_images: species -> RGBA float image (white background made transparent), rotated by --tip_image_rotate."""
    if not getattr(a, "tip_images", None):
        return None
    key = f"{a.tip_images}|{set_name}|{a.tip_image_rotate}"
    if key not in _TIP_IMAGE_CACHE:
        import matplotlib.image as mpimg
        rows = list(csv.DictReader(open(a.tip_images)))
        base = Path(a.tip_images).parent
        out = {}
        for r in rows:
            sp = (r.get("species") or "").strip().replace(" ", "_")
            if r.get("trait_set") and r["trait_set"] != set_name or sp not in names or sp in out:
                continue
            f = Path(r["image"]); f = f if f.is_absolute() else base / f
            if not f.exists():
                continue
            img = mpimg.imread(str(f)).astype(float)
            img = img / 255.0 if img.max() > 1.0 else img
            if img.ndim == 2:
                img = np.dstack([img] * 3)
            if img.shape[2] == 3:
                alpha = (img.min(axis=2) < 0.97).astype(float)
                img = np.dstack([img, alpha])
            img = np.rot90(img, a.tip_image_rotate // 90)
            ys, xs = np.nonzero(img[..., 3] > 0)
            if len(ys):
                img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            out[sp] = img
        _TIP_IMAGE_CACHE[key] = out
    return _TIP_IMAGE_CACHE[key] or None


GROUP_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7a4fa3", "#8c6d31"]


REGIME_COLOURS = ["#882255", "#117733", "#332288", "#999933", "#AA4499", "#44AA99", "#DDCC77", "#88CCEE", "#661100",
                  "#6699CC"]                      # Tol's muted scheme: unlike the Okabe-Ito group colours of the links
RATE_CMAP = "viridis"


def regime_colours(info: Dict) -> Dict[str, str]:
    """SURFACE regime -> colour: convergent regimes first (strong colours), the root's regime grey."""
    reg = info["regimes"]
    root_lab = reg[max(reg, key=len)]
    labs = sorted(set(reg.values()), key=lambda l: (l == root_lab, l not in info["convergent"], l))
    out, k = {}, 0
    for l in labs:
        if l == root_lab:
            out[l] = "#9aa3aa"
        else:
            out[l] = REGIME_COLOURS[k % len(REGIME_COLOURS)]; k += 1
    return out


def _branch_colour_fn(left_colour: Optional[Dict]):
    """(bcol(node) -> colour, legend data) for a tree coloured by regimes or rates."""
    if not left_colour:
        return None, None
    import matplotlib
    from matplotlib.colors import Normalize
    br = left_colour["branch"]
    if left_colour["kind"] == "regimes":
        cols = regime_colours(left_colour["info"])
        return (lambda nd: cols.get(br.get(branch_key(nd)))), {"kind": "regimes", "colours": cols,
                                                                  "info": left_colour["info"]}
    vals = np.array(list(br.values()), float)
    norm = Normalize(float(np.nanmin(vals)), float(np.nanmax(vals)) if np.nanmax(vals) > np.nanmin(vals) else
                     float(np.nanmin(vals)) + 1e-9)
    cmap = matplotlib.colormaps[RATE_CMAP]
    return (lambda nd: cmap(norm(br[branch_key(nd)])) if branch_key(nd) in br else None), \
        {"kind": "rates", "norm": norm, "cmap": cmap}


def _regime_legend_rows(leg: Dict) -> List[Tuple[str, str]]:
    info = leg["info"]
    rows = []
    reg = info["regimes"]
    root_lab = reg[max(reg, key=len)]
    for l, c in leg["colours"].items():
        tipsl = info.get("regime_tips", {}).get(l, [])
        sp = ", ".join(s.replace("Platydracus_", "P. ").replace("_", " ") for s in tipsl)
        kind = ("convergent: " + str(info["regime_count"][l]) + " separate shifts" if l in info["convergent"] else
                "ancestral (root) regime" if l == root_lab else "one shift")
        rows.append((c, f"regime {l} ({kind}): {sp}" if l != root_lab else f"regime {l} ({kind}): the other species"))
    return rows


def plot_tanglegram(left: Node, right: Node, support: Dict[int, float], path_stem, title: str,
                    highlight: Optional[List] = None, left_title="molecular tree",
                    right_title="colour tree", thumbs: Optional[Dict[str, np.ndarray]] = None, ncross=None,
                    highlight_note: str = "", thumb_note: str = "", left_colour: Optional[Dict] = None,
                    links: str = "curved"):
    """the two trees face to face; each species' picture beside its name on both sides, joined by a link from
    picture to picture (links = 'curved': a cubic Bezier S-curve leaving and arriving horizontally, control points
    40% of the gap from each end; 'straight': a segment). highlight: groups ({'members': [...], 'label': str} or
    plain lists) whose links are coloured. left_colour: the molecular tree's branches coloured by SURFACE regimes or
    variable rates (left_tree_colouring)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    tl, tr = tips_of(left), tips_of(right)
    n = len(tl)
    xl, yl = _layout(left); xr, yr = _layout(right)
    Hl, Hr = max(xl.values()) or 1, max(xr.values()) or 1
    groups = [g if isinstance(g, dict) else {"members": list(g), "label": f"group {k + 1}"}
              for k, g in enumerate(highlight or [])]
    bcol, leg = _branch_colour_fn(left_colour)
    reg_rows = _regime_legend_rows(leg) if leg and leg["kind"] == "regimes" else []
    row_in = 0.34 if thumbs else 0.25                      # inches per species
    n_notes = 4
    n_key = n_notes + max(len(groups) + 1, len(reg_rows) + 1, 3 if leg and leg["kind"] == "rates" else 0)
    fig_w, fig_h = 13.0, max(4.5, row_in * n + 1.4 + 0.17 * n_key)
    lab_w = 0.40 * max(len(t.name) for t in tl) / 22       # label width, in tree widths
    asp = float(np.median([im.shape[1] / im.shape[0] for im in thumbs.values()])) if thumbs else 0.0
    gap_w = 0.55
    xr_span_guess = 2 + 2 * lab_w + gap_w + 0.1
    for _ in range(3):                                     # x units per inch depend on the thumbnail width itself
        x_per_in = (xr_span_guess + 0.06) / fig_w
        th = 0.84 * row_in * asp * x_per_in if thumbs else 0.0
        xr_span_guess = 2 + 2 * lab_w + gap_w + 2 * th + 0.1
    l_lab = 1.015; l_img = l_lab + lab_w; l_end = l_img + th + (0.01 if thumbs else 0)
    r_beg = l_end + gap_w; r_img = r_beg + (0.01 if thumbs else 0); r_lab = r_img + th + lab_w
    rstart = r_lab + 0.015
    fig = plt.figure(figsize=(fig_w, fig_h))
    leg_h = 0.17 * n_key / fig_h
    ax = fig.add_axes([0.01, leg_h + 0.02, 0.98, 1 - leg_h - 0.06])
    _draw_tree(ax, left, xl, yl, xscale=1 / Hl, bcol=bcol, blw=1.8 if bcol else None)
    _draw_tree(ax, right, xr, yr, flip_x=True, xscale=1 / Hr, xoff=rstart + 1)
    col, sty = {}, {}
    for gi, g in enumerate(groups):                       # more groups than colours: the colours again, dashed
        g["colour"] = GROUP_COLOURS[gi % len(GROUP_COLOURS)]
        g["style"] = "-" if gi < len(GROUP_COLOURS) else (0, (4, 2))
        for s_ in g["members"]:
            col.setdefault(s_, g["colour"]); sty.setdefault(s_, g["style"])

    def picture(name, x0, y, align):
        """draws the picture in its slot (left side: against the slot's right edge, right side: against its left
        edge, so every link can start at the picture); returns the x where the link meets it"""
        if thumbs is None or name not in thumbs:
            return x0 + th if align == "right" else x0
        img = thumbs[name]; h, w = img.shape[:2]
        ww = min(th, 0.84 * row_in * (w / h) * x_per_in)          # keep the aspect
        xa = x0 + th - ww if align == "right" else x0
        ax.imshow(img, extent=(xa, xa + ww, y + 0.42, y - 0.42), interpolation="antialiased", aspect="auto", zorder=3)
        return xa + ww if align == "right" else xa
    lx = {}
    for t in tl:
        ax.text(xl[id(t)] / Hl + 0.015, yl[id(t)], t.name.replace("_", " "), va="center", fontsize=6.3, style="italic")
        lx[t.name] = picture(t.name, l_img, yl[id(t)], "right")
    rpos = {t.name: t for t in tr}
    rx = {}
    for t in tr:
        ax.text(rstart + 1 - xr[id(t)] / Hr - 0.015, yr[id(t)], t.name.replace("_", " "), va="center", ha="right",
                fontsize=6.3, style="italic")
        rx[t.name] = picture(t.name, r_img, yr[id(t)], "left")
    x_left_end = max(lx.values()); x_right_start = min(rx.values())
    for t in tl:
        c = col.get(t.name, "#a9b1b8")
        x0, y0 = x_left_end, yl[id(t)]
        x1, y1 = x_right_start, yr[id(rpos[t.name])]
        lw_ = 1.8 if t.name in col else 0.7
        z = 4 if t.name in col else 3
        if links == "curved":
            dx = x1 - x0
            pth = MPath([(x0, y0), (x0 + 0.4 * dx, y0), (x1 - 0.4 * dx, y1), (x1, y1)],
                        [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
            ax.add_patch(PathPatch(pth, fill=False, edgecolor=c, lw=lw_, linestyle=sty.get(t.name, "-"), zorder=z,
                                   capstyle="butt"))
        else:
            ax.plot([x0, x1], [y0, y1], color=c, lw=lw_, ls=sty.get(t.name, "-"), zorder=z, solid_capstyle="butt")
    for nd in internal_nodes(right):
        s = support.get(id(nd))
        if s is not None and nd is not right:
            ax.text(rstart + 1 - xr[id(nd)] / Hr + 0.008, yr[id(nd)] - 0.12, _support_text(s), ha="left", va="bottom",
                    fontsize=5.3, color="#1b5e8a" if s >= 0.7 else "#9aa3aa")
    ax.text(0.0, -1.3, left_title, fontsize=8, fontweight="bold")
    ax.text(rstart + 1, -1.3, right_title, fontsize=8, fontweight="bold", ha="right")
    ax.set_ylim(n - 0.3, -1.9)
    ax.set_xlim(-0.03, rstart + 1.03)
    ax.axis("off")
    ax.set_title(title + (f"  ({ncross} line crossings)" if ncross is not None else ""), fontsize=9)
    # key
    lax = fig.add_axes([0.03, 0.01, 0.94, leg_h])
    lax.axis("off")
    word = "curve" if links == "curved" else "line"
    handles = [Line2D([], [], color="#a9b1b8", lw=0.7, label=f"thin grey {word}: species not in a highlighted group")]
    handles += [Line2D([], [], color=g["colour"], lw=1.8, ls=g["style"],
                       label=f"{g['label']}: " + ", ".join(m.replace("_", " ").replace("Platydracus ", "P. ")
                                                         for m in g["members"])) for g in groups]
    notes = [x for x in (highlight_note, f"thick coloured {word}s: members of that group (dashed when the colours repeat)",
                         "numbers on the colour tree: bootstrap support % (blue >= 70)", thumb_note) if x]
    l1 = lax.legend(handles=handles, loc="upper left", fontsize=6.5, frameon=False, ncol=1 if len(handles) < 12 else 2,
                    title="\n".join(notes), title_fontsize=6.3, alignment="left")
    if reg_rows:
        lax.add_artist(l1)
        lax.legend(handles=[Line2D([], [], color=c, lw=2.4, label=t) for c, t in reg_rows], loc="upper left",
                   bbox_to_anchor=(0.52, 1.0), fontsize=6.5, frameon=False, alignment="left",
                   title="molecular-tree branches: " + left_colour["what"], title_fontsize=6.5)
    if leg and leg["kind"] == "rates":
        cax = fig.add_axes([0.62, 0.01 + 0.55 * leg_h, 0.3, 0.12 / fig_h])
        sm = plt.cm.ScalarMappable(cmap=leg["cmap"], norm=leg["norm"])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("molecular-tree branches: log10(rate / median branch rate)\n" + left_colour["what"], fontsize=6.3)
        cb.ax.tick_params(labelsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200)
    plt.close(fig)


def plot_colour_heatmap(mol: Node, col: Node, support: Dict[int, float], D: np.ndarray, names: List[str], path_stem,
                        title: str, metric_label: str, thumbs: Optional[Dict[str, np.ndarray]] = None):
    """molecular tree at the left (rows in its tip order), colour dendrogram on top (columns in its order, bootstrap
    support printed), species x species colour distance in the centre, darker = more similar colour. With thumbs
    (--tip_images) each row's picture sits between the molecular tree and the heatmap and each column's picture
    (turned upright) between the colour dendrogram and the heatmap; the names stay at the right and bottom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [t.name for t in tips_of(mol)]
    cols = [t.name for t in tips_of(col)]
    ix = {s: i for i, s in enumerate(names)}
    M = D[np.ix_([ix[s] for s in rows], [ix[s] for s in cols])]
    n = len(rows)
    cell = (0.19 if not thumbs else 0.24) if n <= 40 else max(0.11, 7.6 / n)
    fs = 6.0 if n <= 40 else max(4.2, min(6.0, cell * 72 * 0.62))
    lab = 0.052 * max(len(s) for s in names) * fs / 6.0
    asp = float(np.median([im.shape[1] / im.shape[0] for im in thumbs.values()])) if thumbs else 0.0
    tw = min(0.9, 0.9 * cell * asp) if thumbs else 0.0               # thumbnail strip width (rows) / height (cols)
    tree_w, tree_h = 1.6, 1.3
    W = tree_w + tw + n * cell + lab + 1.1
    Hh = tree_h + tw + n * cell + lab + 0.7
    fig = plt.figure(figsize=(W, Hh))
    fx = lambda v: v / W; fy = lambda v: v / Hh
    x0, y0 = tree_w + tw, lab + 0.35
    hm = fig.add_axes([fx(x0), fy(y0), fx(n * cell), fy(n * cell)])
    tl = fig.add_axes([fx(0.15), fy(y0), fx(tree_w - 0.2), fy(n * cell)])
    tt = fig.add_axes([fx(x0), fy(y0 + n * cell + tw + 0.05), fx(n * cell), fy(tree_h - 0.15)])
    cax = fig.add_axes([fx(x0 + n * cell + lab + 0.25), fy(y0 + n * cell * 0.25), fx(0.13), fy(n * cell * 0.5)])
    im = hm.imshow(M, cmap="Blues_r", interpolation="nearest", aspect="auto", vmin=0, vmax=float(np.nanmax(M)))
    hm.set_xticks(range(n)); hm.set_yticks(range(n))
    hm.set_xticklabels([s.replace("_", " ") for s in cols], rotation=90, fontsize=fs, style="italic")
    hm.set_yticklabels([s.replace("_", " ") for s in rows], fontsize=fs, style="italic")
    hm.yaxis.tick_right(); hm.tick_params(length=0, pad=1.5)
    for s in hm.spines.values():
        s.set_visible(False)
    if thumbs:
        ra = fig.add_axes([fx(tree_w - 0.02), fy(y0), fx(tw), fy(n * cell)])
        ca = fig.add_axes([fx(x0), fy(y0 + n * cell + 0.02), fx(n * cell), fy(tw)])
        for k, s in enumerate(rows):
            if s in thumbs:
                img = thumbs[s]; h, w = img.shape[:2]
                ww = min(tw, 0.86 * cell * w / h) / tw              # fraction of the strip width
                ra.imshow(img, extent=(1 - ww, 1, k + 0.43, k - 0.43), aspect="auto", interpolation="antialiased")
        for k, s in enumerate(cols):
            if s in thumbs:
                img = np.rot90(thumbs[s], -1); h, w = img.shape[:2]   # upright in the narrow column
                hh = min(tw, 0.86 * cell * h / w) / tw
                ca.imshow(img, extent=(k - 0.43, k + 0.43, 0, hh), aspect="auto", interpolation="antialiased")
        ra.set_xlim(0, 1.02); ra.set_ylim(n - 0.5, -0.5); ra.axis("off")
        ca.set_xlim(-0.5, n - 0.5); ca.set_ylim(-0.02, 1.0); ca.axis("off")
    x, y = _layout(mol)
    H = max(x.values()) or 1
    _draw_tree(tl, mol, x, y, lw=0.8)
    tl.set_ylim(n - 0.5, -0.5); tl.set_xlim(-0.02 * H, H * 1.01); tl.axis("off")
    tl.text(0, -0.9, "molecular tree", fontsize=7, fontweight="bold", va="bottom")
    x2, y2 = _layout(col)
    H2 = max(x2.values()) or 1
    _draw_tree(tt, col, x2, y2, vertical=True, lw=0.8)
    for nd in internal_nodes(col):
        s = support.get(id(nd))
        if s is not None and nd is not col and s >= 0.5:
            tt.text(y2[id(nd)], x2[id(nd)] - 0.01 * H2, _support_text(s), ha="center", va="bottom",
                    fontsize=max(3.5, fs - 1.2), color="#1b5e8a" if s >= 0.7 else "#9aa3aa")
    tt.set_xlim(-0.5, n - 0.5); tt.set_ylim(H2 * 1.01, -0.06 * H2); tt.axis("off")
    tt.text(-0.5, -0.05 * H2, "colour tree (bootstrap support %, shown when >= 50)", fontsize=7, fontweight="bold",
            va="bottom")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"{metric_label} between species means\n(dark = similar colour)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=9, x=fx(x0 + n * cell / 2), y=1 - fy(0.12))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200)
    plt.close(fig)


def plot_palette_tree(tree: Node, support: Dict[int, float], palettes: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      nspec: Dict[str, int], photos: Optional[Dict[str, np.ndarray]], path_stem, title: str,
                      xlabel: str):
    """the palette-EMD colour tree with each species' mean recolorize palette as a horizontal bar (colour blocks,
    width = pixel share, largest first), its picture (--tip_images) and its name."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    tips = tips_of(tree)
    n = len(tips)
    x, y = _layout(tree)
    H = max(x.values()) or 1.0
    fig, ax = plt.subplots(figsize=(10.0, max(4.0, 0.30 * n + 1.3)))
    _draw_tree(ax, tree, x, y)
    for nd in internal_nodes(tree):
        s = support.get(id(nd))
        if s is not None and nd is not tree:
            ax.text(x[id(nd)] - 0.008 * H, y[id(nd)] - 0.12, _support_text(s), ha="right", va="bottom", fontsize=5.5,
                    color="#1b5e8a" if s >= 0.7 else "#9aa3aa")
    tx = max(x[id(t)] for t in tips)
    bar0, barw = tx + 0.03 * H, 0.75 * H
    asp = float(np.median([im.shape[1] / im.shape[0] for im in photos.values()])) if photos else 0.0
    fig.canvas.draw()
    # x units per inch, to keep the pictures' aspect
    bb = ax.get_position(); xspan = H * 2.35
    x_per_in = xspan / (bb.width * fig.get_figwidth())
    y_per_in = (n + 0.2) / (bb.height * fig.get_figheight())
    pw = 0.8 * asp * x_per_in / y_per_in if photos else 0.0
    for t in tips:
        yy = y[id(t)]
        ax.plot([x[id(t)], tx], [yy, yy], color="#d0d5d9", lw=0.5, ls=":")
        w, lab_ = palettes.get(t.name, (np.array([]), np.zeros((0, 3))))
        rgb = lab_to_rgb(lab_) if len(w) else []
        xx = bar0
        for wi, c in zip(w / max(w.sum(), 1e-12), rgb):
            ax.add_patch(Rectangle((xx, yy - 0.36), wi * barw, 0.72, facecolor=c, edgecolor="none", zorder=3))
            xx += wi * barw
        ax.add_patch(Rectangle((bar0, yy - 0.36), barw, 0.72, facecolor="none", edgecolor="#6b7780", lw=0.4, zorder=4))
        xt = bar0 + barw + 0.02 * H
        if photos and t.name in photos:
            img = photos[t.name]; h, wd = img.shape[:2]
            ww = min(pw, 0.8 * (wd / h) * x_per_in / y_per_in)
            ax.imshow(img, extent=(xt, xt + ww, yy + 0.4, yy - 0.4), aspect="auto", interpolation="antialiased", zorder=3)
            xt += pw + 0.02 * H
        k = nspec.get(t.name, 0)
        ax.text(xt, yy, t.name.replace("_", " ") + (f"  ({k} specimens)" if k > 1 else ""), va="center", fontsize=6.5,
                style="italic")
    ax.text(bar0, -0.9, "mean recolorize palette (block width = pixel share; specimens averaged)", fontsize=6.5,
            va="bottom")
    ax.set_ylim(n - 0.4, -1.2)
    ax.set_xlim(-0.02 * H, H * 2.35)
    ax.set_xticks([v for v in ax.get_xticks() if 0 <= v <= H * 1.0001])
    ax.spines["bottom"].set_bounds(0, H)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200)
    plt.close(fig)


def plot_chronogram(tree: Node, branch: Dict[frozenset, object], kind: str, path_stem, title: str,
                    regime_info: Optional[Dict] = None):
    """the dated tree (time before present on the x axis) with its branches coloured by SURFACE regime (kind =
    'regimes', a key naming each regime's species) or by the log rate relative to the median branch (kind = 'rates',
    a perceptually uniform scale with a colour bar)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    t = _copy_tree(tree); _set_depths(t)
    tips = tips_of(t)
    n = len(tips)
    x, y = _layout(t)
    H = max(x.values()) or 1.0
    lc = {"kind": kind, "branch": branch, "info": regime_info, "what": ""}
    bcol, leg = _branch_colour_fn(lc)
    reg_rows = _regime_legend_rows(leg) if kind == "regimes" else []
    key_h = 0.17 * (len(reg_rows) + 1) if reg_rows else 0.0
    fig_h = max(4.0, 0.2 * n + 1.3) + key_h
    fig = plt.figure(figsize=(8.5, fig_h))
    ax = fig.add_axes([0.04, (key_h + 0.45) / fig_h, 0.80 if kind == "rates" else 0.92, 1 - (key_h + 0.85) / fig_h])
    _draw_tree(ax, t, x, y, bcol=bcol, blw=2.0)
    for tp in tips:
        ax.text(x[id(tp)] + 0.01 * H, y[id(tp)], tp.name.replace("_", " "), va="center", fontsize=6.5, style="italic")
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(-0.01 * H, H * 1.32)
    from matplotlib.ticker import MaxNLocator
    ages = [v for v in MaxNLocator(6).tick_values(0, H) if 0 <= v <= H * 1.0001]
    ax.set_xticks([H - v for v in ages]); ax.set_xticklabels([f"{v:g}" for v in ages], fontsize=6.5)
    ax.spines["bottom"].set_bounds(0, H)
    ax.set_xlabel("time before present (tree units; Ma on a chronogram)", fontsize=7.5)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=9)
    if kind == "rates" and leg:
        cax = fig.add_axes([0.87, 0.3, 0.02, 0.4])
        sm = plt.cm.ScalarMappable(cmap=leg["cmap"], norm=leg["norm"])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("log10(rate / median branch rate)", fontsize=7); cb.ax.tick_params(labelsize=6)
    if reg_rows:
        lax = fig.add_axes([0.04, 0.01, 0.92, key_h / fig_h])
        lax.axis("off")
        lax.legend(handles=[Line2D([], [], color=c, lw=2.4, label=tx) for c, tx in reg_rows], loc="upper left",
                   fontsize=6.3, frameon=False, title="branches: SURFACE regime (the branch leading to each node)",
                   title_fontsize=6.5, alignment="left")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200)
    plt.close(fig)


# ============================================================================ palette distance (recolorize palettes)
def emd(wa: np.ndarray, A: np.ndarray, wb: np.ndarray, B: np.ndarray) -> float:
    """earth mover's distance between two weighted colour sets (Rubner et al. 2000), as emdist::emd(cbind(wa, A),
    cbind(wb, B)) with weights normalised to sum 1 (as validation/recolorize_v22/scripts/trees.R does) and the Euclidean
    ground distance (in CIELAB: CIE76). Solved as a transportation problem by linear programming (HiGHS); the flows on
    the optimal support are then re-solved from the supply and demand equations, so the cost is exact to double
    precision (emdist itself works in single precision: its values agree to about 1e-6 relative)."""
    from scipy.optimize import linprog
    wa = np.asarray(wa, float) / np.sum(wa); wb = np.asarray(wb, float) / np.sum(wb)
    A = np.asarray(A, float).reshape(len(wa), -1); B = np.asarray(B, float).reshape(len(wb), -1)
    G = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    m, n = G.shape
    if m == 1 or n == 1:                                      # one colour on a side: every flow is fixed
        return float((wa[:, None] * wb[None, :] * G).sum())
    Aeq = np.zeros((m + n, m * n))
    for i in range(m):
        Aeq[i, i * n:(i + 1) * n] = 1.0
    for j in range(n):
        Aeq[m + j, j::n] = 1.0
    beq = np.r_[wa, wb]
    r = linprog(G.ravel(), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs",
                options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if r.status != 0:
        raise RuntimeError(f"EMD: linear program failed ({r.message})")
    x = r.x
    sup = x > 1e-12
    f, *_ = np.linalg.lstsq(Aeq[:, sup], beq, rcond=None)     # basic flows from the equations: machine precision
    if (f >= -1e-12).all() and np.abs(Aeq[:, sup] @ f - beq).max() < 1e-12:
        x = np.zeros_like(x); x[sup] = np.clip(f, 0, None)
    return float(G.ravel() @ x / x.sum())


def load_palette_csv(path) -> Tuple[np.ndarray, np.ndarray]:
    """a recolorize palette (columns share or n_pixels, and L, a, b): (weights, CIELAB array colours x 3)."""
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ValueError(f"{path}: empty palette")
    wcol = "share" if "share" in rows[0] else "n_pixels" if "n_pixels" in rows[0] else "frac"
    w = np.array([float(r[wcol]) for r in rows])
    lab = np.array([[float(r["L"]), float(r["a"]), float(r["b"])] for r in rows])
    keep = w > 0
    return w[keep], lab[keep]


def load_palettes(spec: str, groups: Optional[Dict[str, str]], names: List[str]) -> List[Dict]:
    """--palettes NAME=GLOB_OR_CSV for one set: the palettes of the specimens of the species in `names`.
    GLOB: palette files whose name contains the photo's file name stem (as Descriptron's colour extraction writes
    them: recolorize_dE20_<stem>png_<id>_fg_<set>_palette.csv), the photo -> species from --groups.
    CSV: columns filename (or photo / image / specimen) and palette (path, relative to the CSV), and optionally species.
    Returns [{'photo', 'species', 'path', 'w', 'lab'}], sorted by species then photo."""
    import glob as _glob
    nameset = set(names)
    out = []
    p = Path(spec)
    if p.suffix.lower() == ".csv" and p.exists() and "palette" in (list(csv.DictReader(open(p)))[0] if p.stat().st_size else {}):
        rows = list(csv.DictReader(open(p, newline="")))
        lc = {c.lower(): c for c in rows[0]}
        fcol = next(lc[c] for c in ("filename", "photo", "image", "specimen") if c in lc)
        for r in rows:
            sp = _norm_name(r[lc["species"]]) if "species" in lc and r[lc["species"]] else \
                next((_norm_name(groups[k]) for k in _key_forms(r[fcol]) if groups and k in groups), None)
            f = Path(r[lc["palette"]]); f = f if f.is_absolute() else p.parent / f
            if sp in nameset and f.exists():
                out.append({"photo": r[fcol], "species": sp, "path": str(f)})
    else:
        if groups is None:
            sys.exit("--palettes with a file pattern needs --groups (photo file name -> species)")
        stems = {}
        for k, sp in groups.items():
            st = re.sub(r"\.(png|jpe?g|tiff?|bmp)$", "", Path(k).name, flags=re.I)
            stems.setdefault(st, _norm_name(sp))
        for f in sorted(_glob.glob(spec)):
            fn = Path(f).name
            hits = [s for s in stems                          # the stem as a whole word, case as written
                    if re.search(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?:\.?(?i:png|jpe?g|tiff?|bmp)_|[._])", fn)]
            if not hits:
                continue
            st = max(hits, key=len)                            # the longest stem: "..._A" never matches "..._AB"
            if stems[st] in nameset:
                out.append({"photo": st, "species": stems[st], "path": f})
    seen, uniq = set(), []
    for r in sorted(out, key=lambda r: (r["species"], r["photo"])):
        if r["path"] in seen:
            continue
        seen.add(r["path"])
        r["w"], r["lab"] = load_palette_csv(r["path"])
        uniq.append(r)
    return uniq


def specimen_emd_matrix(pals: List[Dict]) -> np.ndarray:
    n = len(pals)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = emd(pals[i]["w"], pals[i]["lab"], pals[j]["w"], pals[j]["lab"])
    return M


def species_mean_distance(M: np.ndarray, slot_species: np.ndarray, nsp: int, row: Optional[np.ndarray] = None) -> np.ndarray:
    """species distance = mean specimen distance over all pairs of their specimens (trees.R's species_mat).
    slot_species[s] = species index of specimen slot s; row[s] = the specimen in that slot (a bootstrap resample of
    specimens within species; default every specimen once)."""
    spec = np.arange(len(slot_species)) if row is None else np.asarray(row)
    X = np.zeros((len(spec), nsp)); X[np.arange(len(spec)), slot_species] = 1.0
    cnt = X.sum(0)
    S = X.T @ M[np.ix_(spec, spec)] @ X
    D = S / np.outer(cnt, cnt)
    np.fill_diagonal(D, 0.0)
    return D


def specimen_resamples(slot_species: np.ndarray, B: int, rng) -> np.ndarray:
    """B x slots: for each slot, a specimen drawn with replacement from the same species (species with one specimen
    are fixed)."""
    slot_species = np.asarray(slot_species)
    idx = np.empty((B, len(slot_species)), dtype=int)
    for s in np.unique(slot_species):
        members = np.flatnonzero(slot_species == s)
        idx[:, members] = members[rng.integers(0, len(members), size=(B, len(members)))]
    return idx


def mean_palette(pals: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """a species' colours from all its specimens' palettes, each specimen weighing 1 / specimens."""
    w = np.concatenate([p["w"] / p["w"].sum() / len(pals) for p in pals])
    lab = np.vstack([p["lab"] for p in pals])
    o = np.argsort(-w, kind="stable")
    return w[o], lab[o]


def pcoa(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """principal coordinates analysis (classical scaling, Gower 1966), as stats::cmdscale / ape::pcoa(correction =
    "none"): eigen-decomposition of -1/2 J D^2 J; returns (coordinates on the axes with positive eigenvalues,
    every eigenvalue in decreasing order)."""
    n = len(D)
    J = np.eye(n) - 1.0 / n
    G = -0.5 * J @ (np.asarray(D, float) ** 2) @ J
    ev, V = np.linalg.eigh((G + G.T) / 2)
    o = np.argsort(-ev); ev, V = ev[o], V[:, o]
    pos = ev > 1e-10 * max(ev[0], 1e-300)
    X = V[:, pos] * np.sqrt(ev[pos])
    for j in range(X.shape[1]):                               # a fixed sign: the largest |value| positive
        if X[np.argmax(np.abs(X[:, j])), j] < 0:
            X[:, j] = -X[:, j]
    return X, ev


def pattern_pca(Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """principal components of the (standardised) species x features matrix: (scores, share of variance)."""
    Z = Xs - Xs.mean(0)
    U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = sv ** 2 / (sv ** 2).sum()
    sc = U * sv
    for j in range(sc.shape[1]):
        if sc[np.argmax(np.abs(sc[:, j])), j] < 0:
            sc[:, j] = -sc[:, j]
    return sc, ev


def n_axes(ev_share: np.ndarray, spec, kmax: int, n: int) -> int:
    """'auto': the first axes reaching 90% of the variance, at most kmax and n - 2; or a given number."""
    if str(spec) == "auto":
        k = int(np.searchsorted(np.cumsum(ev_share), 0.9 - 1e-12) + 1)
        return max(1, min(k, kmax, n - 2, len(ev_share)))
    return max(1, min(int(spec), len(ev_share)))


def broken_stick_count(ev_share: np.ndarray, n: int) -> int:
    """axes whose share of variance exceeds the broken-stick expectation (Jackson 1993), over the min(n - 1, p)
    axes species data can have."""
    p = min(n - 1, len(ev_share))
    bs = np.array([sum(1.0 / j for j in range(i, p + 1)) / p for i in range(1, p + 1)])
    k = 0
    while k < p and ev_share[k] > bs[k]:
        k += 1
    return k


# ============================================================================ SURFACE (multi-optimum OU, ouch / surface)
R_SQRT_EPS = 1.4901161193847656e-08                           # sqrt(.Machine$double.eps): optim's default reltol


def r_nelder_mead(fn, x0, maxit: int = 500, reltol: float = R_SQRT_EPS, alpha: float = 1.0, beta: float = 0.5,
                  gamma: float = 2.0) -> Dict:
    """R's optim(method = "Nelder-Mead") line by line (nmmin in R's src/appl/optim.c, after Nash 1990): the same
    initial simplex (step 10% of the largest |parameter|, 0.1 if all are 0), reflection / extension / contraction /
    shrink rules, convergence when the spread of the simplex values is <= reltol x (|f(x0)| + reltol), and at most
    maxit function evaluations. Non-finite values count as 1e35. Reproducing it (rather than using a better optimiser)
    makes Descriptron's OU fits and SURFACE paths the same as ouch::hansen's and surface::runSurface's."""
    big = 1.0e35
    x0 = np.array(x0, float)
    n = len(x0)
    f = fn(x0)
    if not np.isfinite(f):
        raise ValueError("function cannot be evaluated at initial parameters")
    funcount = 1
    convtol = reltol * (abs(f) + reltol)
    n1 = n + 1
    P = np.zeros((n + 1, n + 2))                               # rows 0..n-1 parameters, row n values; column n+1 = centroid
    P[n, 0] = f
    P[:n, 0] = x0
    L = 1
    size = 0.0
    step = max([0.1 * abs(v) for v in x0] + [0.0])
    if step == 0.0:
        step = 0.1
    for j in range(2, n1 + 1):
        P[:n, j - 1] = x0
        trystep = step
        while P[j - 2, j - 1] == x0[j - 2]:
            P[j - 2, j - 1] = x0[j - 2] + trystep
            trystep *= 10
        size += trystep
    oldsize = size
    calcvert = True
    fail = 0
    bvec = x0.copy()
    while True:
        if calcvert:
            for j in range(n1):
                if j + 1 != L:
                    bvec = P[:n, j].copy()
                    f = fn(bvec)
                    if not np.isfinite(f):
                        f = big
                    funcount += 1
                    P[n, j] = f
            calcvert = False
        VL = P[n, L - 1]; VH = VL; H = L
        for j in range(1, n1 + 1):
            if j != L:
                f = P[n, j - 1]
                if f < VL:
                    L = j; VL = f
                if f > VH:
                    H = j; VH = f
        if VH <= VL + convtol:
            break
        C = n + 2
        for i in range(n):
            temp = -P[i, H - 1]
            for j in range(n1):
                temp += P[i, j]
            P[i, C - 1] = temp / n
        bvec = (1.0 + alpha) * P[:n, C - 1] - alpha * P[:n, H - 1]
        f = fn(bvec)
        if not np.isfinite(f):
            f = big
        funcount += 1
        VR = f
        if VR < VL:
            P[n, C - 1] = f
            newb = gamma * bvec + (1 - gamma) * P[:n, C - 1]
            P[:n, C - 1] = bvec
            bvec = newb
            f = fn(bvec)
            if not np.isfinite(f):
                f = big
            funcount += 1
            if f < VR:
                P[:n, H - 1] = bvec; P[n, H - 1] = f
            else:
                P[:n, H - 1] = P[:n, C - 1]; P[n, H - 1] = VR
        else:
            if VR < VH:
                P[:n, H - 1] = bvec; P[n, H - 1] = VR
            bvec = (1 - beta) * P[:n, H - 1] + beta * P[:n, C - 1]
            f = fn(bvec)
            if not np.isfinite(f):
                f = big
            funcount += 1
            if f < P[n, H - 1]:
                P[:n, H - 1] = bvec; P[n, H - 1] = f
            elif VR >= VH:
                calcvert = True
                size = 0.0
                for j in range(n1):
                    if j + 1 != L:
                        P[:n, j] = beta * (P[:n, j] - P[:n, L - 1]) + P[:n, L - 1]
                        size += np.abs(P[:n, j] - P[:n, L - 1]).sum()
                if size < oldsize:
                    oldsize = size
                else:
                    fail = 10
                    break
        if funcount > maxit:
            break
    if funcount > maxit:
        fail = 1
    return {"x": P[:n, L - 1].copy(), "fun": float(P[n, L - 1]), "fail": fail, "fncount": funcount}


class OUTree:
    """a tree as ouch::ape2ouch(tree, scale = FALSE) numbers it, with the arrays the Hansen likelihood needs.
    Node order: preorder (as tree_arrays); `ouch` lists the preorder indices in ouch's node order (root, then ape's
    node numbers decreasing: internal nodes, then tips), which is the order SURFACE tries shift positions in."""

    def __init__(self, root: Node):
        self.nodes, self.parent, self.length, B = tree_arrays(root)
        N = len(self.nodes)
        self.N = N
        self.height = np.zeros(N)
        for i in range(1, N):
            self.height[i] = self.height[self.parent[i]] + self.length[i]
        self.tip_ix = np.array([i for i in range(N) if not self.nodes[i].kids])
        inner = [i for i in range(N) if self.nodes[i].kids]
        nt = len(self.tip_ix)
        ape = {int(t): k + 1 for k, t in enumerate(self.tip_ix)}
        ape.update({i: nt + 1 + k for k, i in enumerate(inner)})
        self.ape = ape
        self.ouch = [0] + sorted(range(1, N), key=lambda i: -ape[i])
        self.names = [self.nodes[i].name for i in self.tip_ix]
        self.A = B[self.tip_ix]                                # tips x nodes: node on the tip's path (root column 0)
        self.T = self.height[self.tip_ix]
        self.C = (self.A * self.length) @ self.A.T             # shared time from the root
        self.desc_incl = [np.flatnonzero(B[:, v] > 0) if v else np.arange(N) for v in range(N)]
        self.tipset = [frozenset(t.name for t in tips_of(self.nodes[v])) for v in range(N)]
        self.depth = float(self.height.max())

    def paint(self, shifts: List[Tuple[int, str]]) -> List[str]:
        """surface::repaint: each shift paints its node's branch and every descendant, older shifts first; the root
        takes the first shift's regime."""
        reg = [""] * self.N
        order = sorted(range(len(shifts)), key=lambda k: self.height[shifts[k][0]])
        for k in order:
            v, lab = shifts[k]
            for d in self.desc_incl[v]:
                reg[d] = lab
        reg[0] = shifts[0][1]
        return reg


def hansen_deviance(T: OUTree, y: np.ndarray, reg_codes: np.ndarray, nreg: int, alpha: float, s2: float):
    """-2 log-likelihood of one trait under the Ornstein-Uhlenbeck model with one optimum per regime (Hansen 1997,
    Butler & King 2004) as ouch 2.20 ou_lik_fn computes it: tip i's expectation is sum over regimes of theta_r times
    its weight (exp(-alpha T_i) for the root's regime plus, for every branch on its path, exp(-alpha (T_i - t_end)) -
    exp(-alpha (T_i - t_start)) for the branch's regime), covariance sigma^2 exp(-alpha (T_i + T_j - 2 s_ij)) /
    (2 alpha) (ouch 2.x's form: the root state drawn from the stationary distribution, not fixed), optima by generalised least squares with ouch's glssoln (singular values
    below sqrt(eps) x the largest dropped). Returns (deviance, optima) or (inf, None)."""
    from scipy.linalg import solve_triangular
    h, par = T.height, T.parent
    Ti = T.T[:, None]
    E = np.zeros_like(T.A)
    E[:, 1:] = T.A[:, 1:] * (np.exp(-alpha * (Ti - h[None, 1:])) - np.exp(-alpha * (Ti - h[None, par[1:]])))
    oh = np.zeros((T.N, nreg)); oh[np.arange(T.N), reg_codes] = 1.0
    W = E @ oh
    W[:, reg_codes[0]] += np.exp(-alpha * T.T)
    V = s2 * np.exp(-alpha * (Ti + T.T[None, :] - 2 * T.C)) / (2 * alpha)    # ouch 2.x: stationary root
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return np.inf, None
    a_ = solve_triangular(L, W, lower=True)
    x_ = solve_triangular(L, y, lower=True)
    U, sv, Vt = np.linalg.svd(a_, full_matrices=False)
    ok = sv > R_SQRT_EPS * sv.max()
    theta = Vt[ok].T @ ((U[:, ok].T @ x_) / sv[ok])
    r = x_ - a_ @ theta
    dev = len(y) * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + float(r @ r)
    return (dev, theta) if np.isfinite(dev) else (np.inf, None)


def hansen_fit(T: OUTree, y: np.ndarray, reg: List[str], sqrt_alpha0: float, sigma0: float, maxit: int = 500) -> Dict:
    """ouch::hansen(y, tree, regimes, sqrt.alpha, sigma) for one trait: Nelder-Mead (R's, reproduced) over
    (sqrt(alpha), sigma) from the given start. Returns loglik, alpha, sigma2, theta {regime: optimum}."""
    labels = sorted(set(reg))                                  # ouch: sort(unique(regimes))
    codes = np.array([labels.index(r) for r in reg])
    k = len(labels)
    fn = lambda p: hansen_deviance(T, y, codes, k, p[0] ** 2, p[1] ** 2)[0]
    opt = r_nelder_mead(fn, [sqrt_alpha0, sigma0], maxit=maxit)
    sa, sg = opt["x"]
    dev, theta = hansen_deviance(T, y, codes, k, sa ** 2, sg ** 2)
    return {"loglik": -0.5 * dev, "sqrt_alpha": float(sa), "sigma": float(sg), "alpha": float(sa ** 2),
            "sigma2": float(sg ** 2), "theta": dict(zip(labels, [float(v) for v in theta])) if theta is not None else {},
            "convergence": opt["fail"], "fncount": opt["fncount"]}


def surface_aicc(L: float, npar: int, n: int) -> float:
    """surface::getAIC(L, np, n, AICc = TRUE)."""
    r = -2 * L + 2 * npar
    return r + 2 * npar * ((npar + 1) / (n - npar - 1))


SURFACE_LETTERS = list("abcdefghijklmnopqrstuvwxyz") + ["z" + c for c in "abcdefghijklmnopqrstuvwxyz"] + \
    ["zz" + c for c in "abcdefghijklmnopqrstuvwxyz"]


def _n_regimes(shifts) -> Dict[str, int]:
    labs = [l for _, l in shifts]
    cnt = {l: labs.count(l) for l in dict.fromkeys(labs)}
    kk = len(cnt)
    return {"k": len(labs), "kprime": kk, "deltak": len(labs) - kk, "c": sum(v for v in cnt.values() if v > 1),
            "kprime_conv": sum(1 for v in cnt.values() if v > 1), "kprime_nonconv": sum(1 for v in cnt.values() if v == 1)}


def _fit_all(T: OUTree, Y: np.ndarray, shifts, starts) -> Tuple[List[Dict], float]:
    reg = T.paint(shifts)
    fits = [hansen_fit(T, Y[:, i], reg, s[0], s[1]) for i, s in enumerate(starts)]
    return fits, float(sum(f["loglik"] for f in fits))


def _starts_from(fits) -> List[Tuple[float, float]]:
    """surface passes the previous fit's alpha and sigma^2 back as sqrt.alpha = sqrt(alpha), sigma = sqrt(sigma^2)."""
    return [(np.sqrt(f["alpha"]), np.sqrt(f["sigma2"])) for f in fits]


def surface_start(T: OUTree, Y: np.ndarray) -> Dict:
    """surface::startingModel: one regime 'a' from the root; per trait sqrt.alpha = sqrt(2 / root age), sigma =
    sqrt(var / root age); AICc with k + nt (2 + k') parameters (k shifts counting the root, k' regimes) and n x nt
    observations."""
    n, nt = Y.shape
    shifts = [(0, "a")]
    starts = [(np.sqrt(2 / T.depth), np.sqrt(np.var(Y[:, i], ddof=1) / T.depth)) for i in range(nt)]
    fits, L = _fit_all(T, Y, shifts, starts)
    return {"fit": fits, "lnL": L, "aic": surface_aicc(L, 1 + nt * 3, n * nt), "shifts": shifts, "all_aic": None,
            "n_regimes": _n_regimes(shifts)}


def surface_add_regime(T: OUTree, Y: np.ndarray, old: Dict, exclude: float = 0.0, aic_threshold: float = 0.0) -> Dict:
    """surface::addRegime: try a new regime on every node (ouch order) without a shift and not excluded, keep the
    lowest AICc."""
    n, nt = Y.shape
    used = {l for _, l in old["shifts"]}
    new_lab = next(l for l in SURFACE_LETTERS if l not in used)
    k = len(old["shifts"]) + 1
    kk = len(used) + 1
    starts = _starts_from(old["fit"])
    excluded = set()
    if exclude and old.get("all_aic"):
        o = sorted([(v, nd) for nd, v in old["all_aic"].items() if np.isfinite(v)], key=lambda x: -x[0])
        excluded = {nd for _, nd in o[:int(np.floor(exclude * len(o)))]}
    skip = excluded | {v for v, _ in old["shifts"]}
    aics, fits, lnls = {}, {}, {}
    for v in T.ouch[1:]:
        if v in skip:
            continue
        sh = old["shifts"] + [(v, new_lab)]
        f, L = _fit_all(T, Y, sh, starts)
        aics[v] = surface_aicc(L, k + nt * (2 + kk), n * nt); fits[v] = f; lnls[v] = L
    best = min(aics, key=lambda v: (aics[v], T.ouch.index(v)))  # R: names(sort(aics))[1], ties in node order
    sh = old["shifts"] + [(best, new_lab)]
    return {"fit": fits[best], "lnL": lnls[best], "aic": aics[best], "shifts": sh, "all_aic": aics,
            "n_regimes": _n_regimes(sh)}


def surface_collapse(T: OUTree, Y: np.ndarray, old: Dict, aic_threshold: float = 0.0, only_best: bool = False) -> Dict:
    """surface::collapseRegimes: try merging every pair of regimes; if some merges lower AICc, merge (one pair if
    only one improves or only_best, else every connected cluster of improving pairs whose pairs all improve, otherwise
    that cluster's best pair) and refit."""
    n, nt = Y.shape
    shifts = list(old["shifts"])
    uniq = list(dict.fromkeys(l for _, l in shifts))
    k, kk = len(shifts), len(uniq)
    starts = _starts_from(old["fit"])
    rows = []
    for i in range(kk):
        for j in range(kk):
            if i > j:
                tmp = [(v, uniq[i] if l == uniq[j] else l) for v, l in shifts]
                f, L = _fit_all(T, Y, tmp, starts)
                a_ = surface_aicc(L, k + nt * (2 + kk - 1), n * nt)
                rows.append({"X": uniq[i], "Y": uniq[j], "aic": a_, "daic": a_ - old["aic"]})
    newshifts, best_aic, fit, lnl = shifts, old["aic"], old["fit"], old["lnL"]
    good = [r for r in rows if r["daic"] < aic_threshold]
    if good:
        if only_best or len(good) == 1:
            z = min(range(len(good)), key=lambda q: good[q]["daic"])
            x_, y_ = good[z]["X"], good[z]["Y"]
            target = next(l for _, l in newshifts if l == y_)
            newshifts = [(v, target if l == x_ else l) for v, l in newshifts]
        else:
            vx = list(dict.fromkeys([r["X"] for r in good] + [r["Y"] for r in good]))
            adj = {v: set() for v in vx}
            for r in good:
                adj[r["X"]].add(r["Y"]); adj[r["Y"]].add(r["X"])
            memb, comp = {}, 0
            for v in vx:                                       # igraph::clusters(weak): ids by first vertex
                if v in memb:
                    continue
                comp += 1; stack = [v]
                while stack:
                    u = stack.pop()
                    if u in memb:
                        continue
                    memb[u] = comp; stack += list(adj[u])
            for m in range(1, comp + 1):
                cl = [v for v in vx if memb[v] == m]
                temp = [r for r in rows if r["X"] in cl and r["Y"] in cl]
                if all(r["daic"] < aic_threshold for r in temp if np.isfinite(r["daic"])):
                    v_ = sorted(cl)
                else:
                    b = min(temp, key=lambda r: r["daic"])
                    v_ = sorted([b["X"], b["Y"]])
                newshifts = [(v, v_[0] if l in v_[1:] else l) for v, l in newshifts]
        fit, lnl = _fit_all(T, Y, newshifts, starts)
        nreg = len(set(l for _, l in newshifts))
        best_aic = surface_aicc(lnl, k + nt * (2 + nreg), n * nt)
    return {"fit": fit, "lnL": lnl, "aic": best_aic, "shifts": newshifts, "all_aics": rows,
            "n_regimes": _n_regimes(newshifts)}


def run_surface(T: OUTree, Y: np.ndarray, exclude: float = 0.0, aic_threshold: float = 0.0,
                max_steps: Optional[int] = None, verbose: bool = False) -> Dict:
    """surface::runSurface (Ingram & Mahler 2013): forward phase adding regime shifts while AICc falls by more than
    -aic_threshold, then backward phase merging regimes into convergent ones. Y: tips (in T.names order) x traits,
    each trait its own alpha, sigma^2 and optima, the regimes shared (traits independent, as surface)."""
    max_steps = max_steps or T.N
    fwd = [surface_start(T, Y)]
    for j in range(1, max_steps):
        nxt = surface_add_regime(T, Y, fwd[-1], exclude, aic_threshold)
        if verbose:
            print(f"    SURFACE forward step {j}: AICc {nxt['aic']:.3f} (previous {fwd[-1]['aic']:.3f})", flush=True)
        if nxt["aic"] - fwd[-1]["aic"] >= aic_threshold:
            break
        fwd.append(nxt)
    bwd = [{**fwd[-1]}]
    kmax = fwd[-1]["n_regimes"]["kprime"]
    for j in range(1, max(kmax, 1)):
        nxt = surface_collapse(T, Y, bwd[-1], aic_threshold)
        if nxt["aic"] == bwd[-1]["aic"]:
            break
        bwd.append(nxt)
    return {"fwd": fwd, "bwd": bwd, "final": bwd[-1]}


def surface_regimes(T: OUTree, final: Dict) -> Dict:
    """regime per node, the shifts (node -> regime), each regime's tips and whether it is convergent (painted by
    two or more separate shifts)."""
    reg = T.paint(final["shifts"])
    cnt = {}
    for _, l in final["shifts"]:
        cnt[l] = cnt.get(l, 0) + 1
    tips = {}
    for k, i in enumerate(T.tip_ix):
        tips.setdefault(reg[i], []).append(T.names[k])
    return {"node_regime": reg, "count": cnt, "tips": tips,
            "convergent": [l for l, c in cnt.items() if c > 1]}


# ============================================================================ colour analyses (driver)
def load_convergent_groups(path: str) -> Dict[str, List[str]]:
    """CSV species,group -> {group: [species, ...]}."""
    rows = list(csv.DictReader(open(path, newline="")))
    lc = {c.strip().lower(): c for c in rows[0]}
    sc, gc = lc.get("species") or list(rows[0])[0], lc.get("group") or list(rows[0])[1]
    out: Dict[str, List[str]] = {}
    for r in rows:
        out.setdefault(r[gc].strip(), []).append(_norm_name(r[sc]))
    return out


def _write_csv(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def _fmt_p(p: float) -> str:
    return f"{p:.3f}"


DIST_LABEL = {"de2000": "ΔE00", "de76": "ΔE76", "palette_emd": "palette EMD (ΔE76)", "pca": "colour-pattern PC distance"}


def _euclid(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0))


def _pca_scores(Xs: np.ndarray, k: int) -> np.ndarray:
    Z = Xs - Xs.mean(0)
    U, sv, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :k] * sv[:k]


def colour_engine(dist: str, name: str, S: Dict, names: List[str], a, gmap) -> Tuple[Optional[Dict], str]:
    """one colour distance for one trait set: the species distance matrix, how to rebuild it from a bootstrap
    replicate, the traits the convergence tests use, and the texts for the report. (None, reason) if not possible."""
    n = len(names)
    rows_sp = [S["species"].index(nm) for nm in names]
    cells, cix = colour_cell_index(S["full_cols"], a.colour_cells)
    lab = S["full_raw"][rows_sp][:, cix] if len(cells) else None
    thumbs = None
    if lab is not None:
        th = pattern_thumbnails(lab, cells)
        thumbs = dict(zip(names, th)) if th is not None else None
    spc = [S["counts"].get(nm, 0) for nm in names]
    e = {"dist": dist, "metric": DIST_LABEL[dist], "thumbs": thumbs, "lab": lab, "cells": cells}
    if dist in ("de2000", "de76"):
        if lab is None:
            return None, f"no columns match --colour_cells {a.colour_cells!r}"
        Dc = per_cell_distances(lab, dist)
        D = colour_distance(Dc)
        if not np.isfinite(D).all():
            bad = [(names[i], names[j]) for i, j in zip(*np.where(~np.isfinite(D))) if i < j][:5]
            return None, f"some species share no grid cell (e.g. {bad})"
        ok = np.isfinite(lab).all(axis=(0, 2))
        Xc = lab[:, ok, :].reshape(n, -1)
        what = f"colour of {int(ok.sum())} grid cells (L*, a*, b*)"
        if a.conv_pcs == "all":
            X = scale(Xc, "standardize"); tdesc = f"{X.shape[1]} standardised variables ({what})"
        else:
            Z = Xc - Xc.mean(0)
            U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
            ev = sv ** 2 / (sv ** 2).sum()
            k = n_axes(ev, a.conv_pcs, a.conv_pcs_max, n)
            X = Z @ Vt[:k].T
            tdesc = f"first {k} principal components ({100 * ev[:k].sum():.1f}% of the variance) of the {what}"
        nc = Dc.shape[0]
        e.update(D=D, X=X, tdesc=tdesc, ncols=nc, Dc=Dc,
                 dist_fn=lambda row: colour_distance(Dc, row),
                 resample=lambda B, rng: rng.integers(0, nc, size=(B, nc)),
                 desc=f"{len(cells)} homologous grid cells (L*, a*, b* species means; {min(spc)}-{max(spc)} specimens per "
                      f"species); species distance = mean per-cell {DIST_LABEL[dist]}",
                 boot_desc="grid cells resampled with replacement")
        return e, ""
    if dist == "pca":
        Xs = scale(S["raw"][rows_sp], "standardize")
        if Xs.shape[1] < 2:
            return None, "fewer than two varying features"
        sc, ev = pattern_pca(Xs)
        k = n_axes(ev, a.pca_pcs, a.conv_pcs_max, n)
        D = _euclid(sc[:, :k])
        X = Xs if a.conv_pcs == "all" else sc[:, :k]
        p = Xs.shape[1]
        e.update(D=D, X=X, k=k, ev=ev, scores=sc,
                 tdesc=(f"{p} standardised colour-pattern features" if a.conv_pcs == "all" else
                        f"first {k} principal components ({100 * ev[:k].sum():.1f}% of the variance) of the {p} "
                        "standardised colour-pattern features (the phylomorphospace's)"),
                 dist_fn=lambda row: _euclid(_pca_scores(Xs[:, row], k)),
                 resample=lambda B, rng: rng.integers(0, p, size=(B, p)),
                 desc=f"{p} standardised colour-pattern features (all numeric columns no species lacks, as the "
                      f"phylomorphospace with --scale standardize); species distance = Euclidean distance on the first {k} "
                      f"principal components ({100 * ev[:k].sum():.1f}% of the variance; --pca_pcs {a.pca_pcs})",
                 boot_desc=f"features resampled with replacement, standardised features re-analysed by PCA and the first "
                           f"{k} PCs kept")
        return e, ""
    if dist == "palette_emd":
        spec = dict(x.split("=", 1) for x in (a.palettes or [])).get(name)
        if not spec:
            return None, f"no --palettes {name}=... given"
        pals = load_palettes(spec, gmap, names)
        have = {p["species"] for p in pals}
        miss = [s for s in names if s not in have]
        if miss:
            return None, f"{len(miss)} species have no palette ({', '.join(miss[:6])}{' ...' if len(miss) > 6 else ''})"
        M = specimen_emd_matrix(pals)
        slot = np.array([names.index(p["species"]) for p in pals])
        D = species_mean_distance(M, slot, n)
        Xp, evp = pcoa(D)
        pos = evp[evp > 1e-10 * evp[0]]
        share = pos / pos.sum()
        k = n_axes(share, a.conv_pcs, a.conv_pcs_max, n) if a.conv_pcs != "all" else len(pos)
        neg = float(-evp[evp < 0].sum() / np.abs(evp).sum())
        nmulti = int(sum(1 for s in names if (slot == names.index(s)).sum() > 1))
        e.update(D=D, X=Xp[:, :k], M=M, pals=pals, slot=slot, k=k, pcoa_ev=evp,
                 tdesc=f"first {k} principal coordinates (PCoA, classical scaling of the species palette-EMD matrix; "
                       f"{100 * share[:k].sum():.1f}% of the positive eigenvalues; negative eigenvalues are "
                       f"{100 * neg:.1f}% of the sum of |eigenvalues|: EMD is not exactly Euclidean)",
                 dist_fn=lambda row: species_mean_distance(M, slot, n, row),
                 resample=lambda B, rng: specimen_resamples(slot, B, rng),
                 desc=f"{len(pals)} recolorize palettes of {n} species ({nmulti} with more than one specimen); specimen "
                      "distance = earth mover's distance between palettes (pixel-share weights, Euclidean CIELAB ground "
                      "distance, as emdist::emd); species distance = mean over all pairs of their specimens (as "
                      "validation/recolorize_v22/scripts/trees.R)",
                 boot_desc=f"specimens resampled with replacement within each species. Support here measures how stable "
                           f"a clade is to WHICH specimens represent each species: {n - nmulti} of {n} species have a "
                           "single specimen and are fixed, so a clade of such species always gets 100%. It is not "
                           "support from independent characters")
        return e, ""
    return None, f"unknown distance {dist}"


def support_counts_fn(dist_fn, D0: np.ndarray, names: List[str], method: str, idx: np.ndarray, tree: Node) -> Dict:
    """how many trees built from the replicate distances dist_fn(row) (rows of idx) contain each internal node of
    `tree` (UPGMA: the same clade; NJ: the same bipartition). Pairs with no distance in a replicate keep D0's."""
    rooted = method == "upgma"
    allnames = frozenset(names)
    ref = [(nd, s if rooted else split_key(s, allnames)) for nd, s in clade_sets(tree)]
    counts = {id(nd): 0 for nd, _ in ref}
    filled = 0
    for row in idx:
        D = dist_fn(row)
        bad = ~np.isfinite(D)
        if bad.any():
            filled += int(bad.sum()) // 2; D = np.where(bad, D0, D)
        keys = tree_keys(build_colour_tree(D, names, method), rooted)
        for nd, k in ref:
            if k is None or k in keys:
                counts[id(nd)] += 1
    counts["__filled__"] = filled
    return counts


def colour_analyses(name: str, S: Dict, tree: Node, tips: List[Node], a, out: Path, rng,
                    given_groups: Optional[Dict[str, List[str]]], gmap=None, chrono: Optional[Dict] = None) -> List[str]:
    """colour_tree, cophylo, heatmap and convergence for one trait set and every --colour_distance; with several
    distances the files are named <set>__<distance>_...; a comparison table <set>_distance_comparison.csv."""
    rep: List[str] = []
    dists = list(dict.fromkeys(a.colour_distance))
    summ = []
    for dist in dists:
        tag = name if len(dists) == 1 else f"{name}__{dist}"
        if len(dists) > 1:
            rep.append(f"### Colour distance: {dist} ({DIST_LABEL[dist]})")
        eng, why = colour_engine(dist, name, S, [t.name for t in tips], a, gmap)
        if eng is None:
            rep.append(f"Colour distance {dist}: {why} - skipped")
            continue
        r_, s_ = colour_one(name, tag, eng, S, tree, tips, a, out, rng, given_groups, chrono)
        rep += r_
        summ.append({"trait_set": name, "distance": dist, **s_})
    if len(summ) > 1:
        _write_csv(out / f"{name}_distance_comparison.csv", summ)
        rep += [f"### {name}: the colour distances compared ({name}_distance_comparison.csv)",
                "| distance | Mantel r (Pearson) vs patristic, P | RF (normalised) | clades with support >= "
                f"{a.min_support:g} | candidate groups | groups passing C1 and Ct1 (P < {a.conv_alpha:g}) |",
                "|---|---|---|---|---|---|"]
        for s in summ:
            rep.append(f"| {s['distance']} | {s.get('mantel_pearson_r', np.nan):.3f}, {s.get('mantel_pearson_P', np.nan):.3f} | "
                       f"{s.get('RF', '')} ({s.get('RF_normalised', np.nan):.3f}) | {s.get('n_supported', '')} | "
                       f"{s.get('n_candidates', '')} | {s.get('n_passing', '')} |")
        if len(summ) >= 2:
            rep.append("Mantel r between the colour distances (Pearson, lower triangles):")
            for i in range(len(summ)):
                for j in range(i + 1, len(summ)):
                    Di, Dj = summ[i]["_D"], summ[j]["_D"]
                    iu = np.triu_indices(len(Di), 1)
                    rep.append(f"  {summ[i]['distance']} vs {summ[j]['distance']}: r = "
                               f"{np.corrcoef(Di[iu], Dj[iu])[0, 1]:.3f}")
        for s in summ:
            s.pop("_D", None)
        _write_csv(out / f"{name}_distance_comparison.csv", summ)
    return rep


def colour_one(name, tag, eng, S, tree, tips, a, out, rng, given_groups, chrono) -> Tuple[List[str], Dict]:
    rep: List[str] = []
    names = [t.name for t in tips]
    n = len(names)
    D = eng["D"]
    metric = eng["metric"]
    summ: Dict = {"_D": D}
    need_tree = any(x in a.analyses for x in ("colour_tree", "cophylo", "heatmap")) or \
        ("convergence" in a.analyses and given_groups is None)
    ctree, sup, tangle = None, {}, None
    if need_tree:
        rep += ["### Colour-pattern tree",
                f"{eng['desc']}; {'UPGMA' if a.colour_tree_method == 'upgma' else 'neighbour joining'}; "
                f"{a.bootstrap} bootstrap replicates: {eng['boot_desc']} (seed {a.seed})"]
        if a.bootstrap > 0:
            ctree = build_colour_tree(D, names, a.colour_tree_method)
            idx = eng["resample"](a.bootstrap, rng)
            cnt = support_counts_fn(eng["dist_fn"], D, names, a.colour_tree_method, idx, ctree)
            nfill = cnt.pop("__filled__")
            sup = {k: v / a.bootstrap for k, v in cnt.items()}
            if nfill:
                rep.append(f"  {nfill} species pairs had no distance in a replicate and kept their full-data distance")
        else:
            ctree = build_colour_tree(D, names, a.colour_tree_method)
        with open(out / f"{tag}_colour_distance.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["species"] + names)
            for i, nm in enumerate(names):
                w.writerow([nm] + [f"{v:.6f}" for v in D[i]])
        if "M" in eng:
            with open(out / f"{tag}_specimen_emd.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["photo", "species"] + [p["photo"] for p in eng["pals"]])
                for i, p in enumerate(eng["pals"]):
                    w.writerow([p["photo"], p["species"]] + [f"{v:.6f}" for v in eng["M"][i]])
        (out / f"{tag}_colour_tree.nwk").write_text(
            to_newick(ctree, {id(k): _support_text(v) for k, v in sup.items()}) + ";\n")
        moltips = {t.name: t for t in tips_of(tree)}
        srows = []
        for nd, s in clade_sets(ctree):
            anc = moltips[min(s)]
            while not s <= {t.name for t in tips_of(anc)}:
                anc = anc.parent
            srows.append({"clade": clade_label(nd), "n_species": len(s), "support": sup.get(id(nd), np.nan),
                          "monophyletic_on_input_tree": len(tips_of(anc)) == len(s),
                          "members": "; ".join(sorted(s))})
        srows.sort(key=lambda r: (-np.nan_to_num(r["support"], nan=-1), r["n_species"]))
        _write_csv(out / f"{tag}_colour_tree_support.csv", srows)
        xlab = (f"{metric} / 2 (UPGMA height)" if a.colour_tree_method == "upgma" else f"{metric} (NJ branch length)")
        plot_colour_tree(ctree, sup, eng["thumbs"], out / f"{tag}_colour_tree.png",
                         f"{tag}: colour-pattern tree ({n} species; bootstrap support %)", xlab)
        photos = load_tip_images(a, name, names)
        if "pals" in eng:
            by = {}
            for p in eng["pals"]:
                by.setdefault(p["species"], []).append(p)
            plot_palette_tree(ctree, sup, {s: mean_palette(v) for s, v in by.items()},
                              {s: len(v) for s, v in by.items()}, photos, out / f"{tag}_colour_bars",
                              f"{tag}: palette-EMD colour tree and each species' mean recolorize palette", xlab)
            rep.append(f"Colour bars: {tag}_colour_bars.png/.pdf (each species' mean palette, block width = pixel share)")
        hi = [r for r in srows if r["support"] >= a.min_support]
        summ["n_supported"] = len(hi)
        summ["n_supported_not_input_clades"] = sum(1 for r in hi if not r["monophyletic_on_input_tree"])
        rep.append(f"Clades with support >= {a.min_support:g}: {len(hi)} of {len(srows)} "
                   f"({summ['n_supported_not_input_clades']} not clades of the input tree); "
                   f"files {tag}_colour_tree.nwk / .png, {tag}_colour_tree_support.csv, {tag}_colour_distance.csv")
        for r in hi:
            rep.append(f"  {r['support']:.2f}  {r['members']}"
                       + ("" if r["monophyletic_on_input_tree"] else "  [not a clade of the input tree]"))
        mol = _copy_tree(tree); _set_depths(mol)
        ncross = None
        if "cophylo" in a.analyses or "heatmap" in a.analyses:
            ncross = untangle(mol, ctree)
        if "cophylo" in a.analyses:
            P = patristic(tips)
            mp = mantel(P, D, "pearson", a.iterations, rng)
            ms = mantel(P, D, "spearman", a.iterations, rng)
            rf = rf_distance(tree, ctree)
            cand = [c["members"] for c in candidate_groups(ctree, sup, tree, a.min_support,
                                                           a.conv_max_size or n // 2)] if sup else []
            cand = [g for g in cand if not any(set(g) < set(h) for h in cand)]
            tangle = dict(mol=mol, ctree=ctree, sup=sup, stem=out / f"{tag}_tanglegram",
                          title=f"{tag}: input tree vs colour-pattern tree", largest=cand, ncross=ncross,
                          thumbs=photos or eng["thumbs"], photo=bool(photos))
            row = {"trait_set": name, "distance": eng["dist"], "n_species": n, "crossings": ncross,
                   "mantel_pearson_r": mp["r"], "mantel_pearson_P": mp["P"], "mantel_spearman_r": ms["r"],
                   "mantel_spearman_P": ms["P"], "permutations": a.iterations, "RF": rf["RF"],
                   "RF_normalised": rf["RF_normalised"], "shared_splits": rf["shared_splits"]}
            _write_csv(out / f"{tag}_cophylo.csv", [row])
            summ.update({k: row[k] for k in ("mantel_pearson_r", "mantel_pearson_P", "mantel_spearman_r",
                                             "mantel_spearman_P", "RF", "RF_normalised", "crossings")})
            rep += ["### Input tree vs colour tree",
                    f"Tanglegram {tag}_tanglegram.png/.pdf: {ncross} line crossings after node rotation",
                    f"Mantel test, patristic distance vs colour distance ({a.iterations} permutations): Pearson r = "
                    f"{mp['r']:.3f}, P = {_fmt_p(mp['P'])}; Spearman r = {ms['r']:.3f}, P = {_fmt_p(ms['P'])}",
                    f"Robinson-Foulds distance (unrooted): {rf['RF']} of a possible {2 * (n - 3)} "
                    f"(normalised {rf['RF_normalised']:.3f}; {rf['shared_splits']} bipartitions shared)"]
        if "heatmap" in a.analyses:
            plot_colour_heatmap(mol, ctree, sup, D, names, out / f"{tag}_colour_heatmap",
                                f"{tag}: colour distance between species ({metric})", metric, thumbs=photos)
            rep.append(f"Heatmap: {tag}_colour_heatmap.png/.pdf (rows: input tree order; columns: colour tree)")
    conv_rows = []
    if "convergence" in a.analyses:
        r_, conv_rows = convergence_section(name, tag, eng["X"], eng["tdesc"], tree, tips, ctree, sup, a, out, rng,
                                            given_groups)
        rep += r_
    summ["n_candidates"] = len(conv_rows)
    passing = [r for r in conv_rows
               if r["P_C1"] < a.conv_alpha and (r["n_lineages"] < 2 or (r["Ct1"] > 0 and r["P_Ct1"] < a.conv_alpha))]
    summ["n_passing"] = len(passing)
    summ["passing_groups"] = " | ".join(r["members"] for r in passing)
    if tangle is not None:
        hl, how = [], ""
        if a.tangle_highlight == "convergent" and conv_rows:
            for r in passing:
                hl.append({"members": r["members"].split("; "), "label": f"{r['group']}: C1 {r['C1']:.2f}"
                           + (f", Ct1 {r['Ct1']:.2f}" if r["n_lineages"] > 1 else "")})
            hl = [g for g in hl if not any(set(g["members"]) < set(h["members"]) for h in hl)]
            how = (f"coloured: candidate convergent groups with P(C1) < {a.conv_alpha:g} and Ct1 > 0 with "
                   f"P(Ct1) < {a.conv_alpha:g} (largest such group where they nest)")
        elif a.tangle_highlight in ("convergent", "largest") and tangle["largest"]:
            hl = [{"members": g, "label": f"colour clade ({len(g)} spp.)"} for g in tangle["largest"]]
            how = f"coloured: the largest colour-tree clades with support >= {a.min_support:g} that are not input-tree clades"
        lc = left_tree_colouring(chrono, a.tangle_tree_colour)
        plot_tanglegram(tangle["mol"], tangle["ctree"], tangle["sup"], tangle["stem"], tangle["title"], highlight=hl,
                        thumbs=tangle["thumbs"], ncross=tangle["ncross"], highlight_note=how,
                        thumb_note=("thumbnails: " + a.tip_images_note) if tangle["photo"] else
                        "thumbnails: species-mean colour of each grid cell", left_colour=lc, links=a.tangle_links,
                        right_title=f"colour tree ({eng['dist']})")
        rep.append(f"Tanglegram drawn{'; ' + how if how else ''}"
                   + (f"; molecular-tree branches coloured by {lc['what']}" if lc else ""))
    return rep, summ


def convergence_section(name, tag, X, tdesc, tree, tips, ctree, sup, a, out, rng, given_groups):
    rep = ["### Convergence"]
    names = [t.name for t in tips]
    n = len(names)
    if given_groups is not None:
        groups = []
        for g, mem in given_groups.items():
            m = sorted({s for s in mem if s in names})
            if len(m) >= 2:
                groups.append({"group": g, "members": m, "support": np.nan, "source": "given"})
            else:
                rep.append(f"  group {g}: fewer than two of its species on the tree - skipped")
    elif ctree is None or not sup:
        rep.append("No colour tree with bootstrap support: give --convergent_groups or --bootstrap > 0 - skipped")
        return rep, []
    else:
        groups = [{"group": f"G{i + 1}", **c, "source": "colour tree"}
                  for i, c in enumerate(candidate_groups(ctree, sup, tree, a.min_support, a.conv_max_size or n // 2))]
    rep.append(f"Traits: {tdesc}; {a.conv_sims} Brownian-motion simulations / resamplings")
    if not groups:
        rep.append(f"No candidate convergent group (colour-tree clades with support >= {a.min_support:g} that are not "
                   "clades of the input tree)")
        return rep, []
    T = ConvTree(tree)
    depth = np.diag(T.C)
    ultra = depth.max() - depth.min() <= 1e-6 * depth.max()
    cand_rows, rows = [], []
    for g in groups:
        mem = g["members"]
        lin = molecular_lineages(tree, mem)
        lins = sorted(set(lin.values()))
        anc = {t.name: t for t in tips_of(tree)}[mem[0]]
        while not set(mem) <= {t.name for t in tips_of(anc)}:
            anc = anc.parent
        cand_rows.append({"group": g["group"], "source": g["source"], "n_species": len(mem),
                          "colour_support": g.get("support", np.nan), "n_molecular_lineages": len(lins),
                          "molecular_lineages": " | ".join(lins),
                          "lineage_sister_groups": " | ".join(f"{l}: {lineage_sisters(tree, l)}" for l in lins),
                          "molecular_mrca_n_tips": len(tips_of(anc)), "members": "; ".join(mem)})
        r = {"trait_set": name, "group": g["group"], "n_species": len(mem), "n_lineages": len(lins),
             "colour_support": g.get("support", np.nan), "members": "; ".join(mem), "traits": tdesc}
        ct = convergence_test(T, X, mem, lin if len(lins) > 1 else None, a.conv_sims, rng)
        for i in range(4):
            r[f"C{i + 1}"], r[f"P_C{i + 1}"] = ct["C"][i], ct["P_C"][i]
        r["pairs_C"] = ct["n_pairs_C"]
        for i in range(4):
            r[f"Ct{i + 1}"], r[f"P_Ct{i + 1}"] = (ct["Ct"][i], ct["P_Ct"][i]) if len(lins) > 1 else (np.nan, np.nan)
        r["pairs_Ct"] = ct["n_pairs_Ct"] if len(lins) > 1 else 0
        if ultra:
            wi = test_wheatsheaf(X, T.C, np.array([s in mem for s in names]), a.conv_sims, rng)
            r.update(wheatsheaf=wi["w"], wheatsheaf_low95=wi["low95"], wheatsheaf_up95=wi["up95"], P_wheatsheaf=wi["P"])
        else:
            r.update(wheatsheaf=np.nan, wheatsheaf_low95=np.nan, wheatsheaf_up95=np.nan, P_wheatsheaf=np.nan)
        r["simulations"] = a.conv_sims
        rows.append(r)
        rep.append(f"  {g['group']} ({len(mem)} species in {len(lins)} molecular lineages"
                   + (f", colour support {g['support']:.2f}" if np.isfinite(g.get('support', np.nan)) else "")
                   + f"): {'; '.join(mem)}")
        rep.append("    C1-C4 = " + ", ".join(f"{r[f'C{i}']:.3f} (P {_fmt_p(r[f'P_C{i}'])})" for i in range(1, 5))
                   + f" over {r['pairs_C']} pairs")
        if len(lins) > 1:
            rep.append("    Ct1-Ct4 = " + ", ".join(f"{r[f'Ct{i}']:.3f} (P {_fmt_p(r[f'P_Ct{i}'])})" for i in range(1, 5))
                       + f" over {r['pairs_Ct']} between-lineage pairs")
        if ultra:
            rep.append(f"    Wheatsheaf index = {r['wheatsheaf']:.3f} (jackknife 95% interval {r['wheatsheaf_low95']:.3f}-"
                       f"{r['wheatsheaf_up95']:.3f}), P = {_fmt_p(r['P_wheatsheaf'])}")
    if not ultra:
        rep.append("  Wheatsheaf index not computed: the tree is not ultrametric (windex's fossil option is not implemented)")
    _write_csv(out / f"{tag}_convergent_candidates.csv", cand_rows)
    _write_csv(out / f"{tag}_convergence.csv", rows)
    rep += [f"  files {tag}_convergent_candidates.csv, {tag}_convergence.csv",
            "  P values are as convevol and windex compute them (#{simulated >= observed} / simulations), one per "
            "statistic and group, not corrected for multiple tests. Groups found from the colour tree are chosen "
            "because they look alike, which the Brownian-motion null does not model: read their P values as "
            "descriptive."]
    return rep, rows


# ---------------------------------------------------------------------------- colour-pattern PCs, models, SURFACE
def pattern_pc_matrix(S: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
    """PCA of the standardised colour-pattern features of the species on the tree (tip order): scores, variance
    shares, number of features."""
    Xs = scale(S["raw"][S["idx"]], "standardize")
    sc, ev = pattern_pca(Xs)
    return sc, ev, Xs.shape[1]


def branch_key(n: Node) -> frozenset:
    return frozenset(t.name for t in tips_of(n))


def surface_section(name: str, S: Dict, tree: Node, tips: List[Node], a, out: Path) -> Tuple[List[str], Dict]:
    """SURFACE on the first colour-pattern PCs, the model comparison (BM, OU1, EB, lambda, SURFACE by AICc on the same
    PCs), variable-rate BM on those PCs, and the two chronogram figures."""
    sc, ev, p = pattern_pc_matrix(S)
    n = len(tips)
    if str(a.surface_pcs) == "auto":
        k = max(2, min(broken_stick_count(ev, n), a.surface_pcs_max))
        how = f"auto: PCs above the broken-stick expectation, kept within 2-{a.surface_pcs_max}"
    else:
        k = int(a.surface_pcs); how = "given"
    k = max(1, min(k, len(ev)))
    Y = sc[:, :k]
    T = OUTree(tree)
    order = [[t.name for t in tips].index(nm) for nm in T.names]
    Yt = Y[order]
    rep = ["### SURFACE: convergent OU regimes on the colour pattern (Ingram & Mahler 2013)",
           f"Traits: the first {k} principal components ({100 * ev[:k].sum():.1f}% of the variance; {how}) of the {p} "
           f"standardised colour-pattern features; each PC its own alpha, sigma^2 and optima, regimes shared (as "
           f"surface 0.6); {n} species x {k} PCs = {n * k} observations for AICc; forward phase then backward phase, "
           f"aic_threshold {a.surface_aic_threshold:g}, exclude {a.surface_exclude:g}"]
    res = run_surface(T, Yt, exclude=a.surface_exclude, aic_threshold=a.surface_aic_threshold, verbose=False)
    S["surface_res"] = {"k": k, "tips": frozenset(T.names), "res": res}
    fin = res["final"]
    reg = surface_regimes(T, fin)
    path_rows = []
    for ph, steps in (("forward", res["fwd"]), ("backward", res["bwd"][1:])):
        for i, s in enumerate(steps):
            last = s["shifts"][-1]
            path_rows.append({"trait_set": name, "phase": ph, "step": i + 1, "AICc": s["aic"], "logLik": s["lnL"],
                              "n_shifts": s["n_regimes"]["k"], "n_regimes": s["n_regimes"]["kprime"],
                              "n_convergent_regimes": s["n_regimes"]["kprime_conv"],
                              "shift_added": (clade_label(T.nodes[last[0]]) + f" -> {last[1]}") if ph == "forward" and i else "",
                              "shifts": "; ".join(f"{clade_label(T.nodes[v])} -> {l}" for v, l in s["shifts"])})
    _write_csv(out / f"{name}_surface_path.csv", path_rows)
    prow = []
    labels = sorted(set(reg["node_regime"]))
    for j, f in enumerate(fin["fit"]):
        r = {"trait_set": name, "PC": f"PC{j + 1}", "alpha": f["alpha"], "sigma2": f["sigma2"],
             "half_life": np.log(2) / f["alpha"] if f["alpha"] > 0 else np.inf,
             "half_life_over_depth": (np.log(2) / f["alpha"]) / T.depth if f["alpha"] > 0 else np.inf,
             "logLik": f["loglik"], "optimiser_code": f["convergence"]}
        r.update({f"theta_{l}": f["theta"].get(l, np.nan) for l in labels})
        prow.append(r)
    _write_csv(out / f"{name}_surface_parameters.csv", prow)
    rrow = []
    for i in range(1, T.N):
        rrow.append({"trait_set": name, "branch_to": clade_label(T.nodes[i]), "is_tip": not T.nodes[i].kids,
                     "regime": reg["node_regime"][i], "shift_here": any(v == i for v, _ in fin["shifts"]),
                     "regime_convergent": reg["node_regime"][i] in reg["convergent"],
                     "tips": "; ".join(sorted(T.tipset[i]))})
    _write_csv(out / f"{name}_surface_regimes.csv", rrow)
    # model comparison on the same PCs, one AICc definition (surface's: n = species x PCs)
    nobs = n * k
    fits = [fit_models(Yt[:, j], T.C, ("BM", "OU", "EB", "lambda"), {}) for j in range(k)]
    tot = lambda m: (sum(next(f for f in fl if f["model"] == m)["logLik"] for fl in fits))
    nreg = fin["n_regimes"]
    cmp = [{"model": "BM", "logLik": tot("BM"), "n_par": 2 * k, "note": "sigma^2 and root state per PC"},
           {"model": "OU1", "logLik": res["fwd"][0]["lnL"], "n_par": 1 + 3 * k,
            "note": "SURFACE's start: one optimum (ouch, stationary root); surface counts the root shift"},
           {"model": "EB", "logLik": tot("EB"), "n_par": 3 * k, "note": "geiger form, a per PC"},
           {"model": "lambda", "logLik": tot("lambda"), "n_par": 3 * k, "note": "geiger form, lambda per PC"},
           {"model": "SURFACE", "logLik": fin["lnL"], "n_par": nreg["k"] + k * (2 + nreg["kprime"]),
            "note": f"{nreg['k']} shifts (root included), {nreg['kprime']} regimes, {nreg['kprime_conv']} convergent"}]
    for c in cmp:
        c["AICc"] = surface_aicc(c["logLik"], c["n_par"], nobs) if nobs - c["n_par"] - 1 > 0 else np.inf
    best = min(c["AICc"] for c in cmp)
    w = np.array([np.exp(-0.5 * (c["AICc"] - best)) for c in cmp])
    for c, wi in zip(cmp, w / w.sum()):
        c["dAICc"], c["weight"], c["trait_set"], c["PCs"] = c["AICc"] - best, float(wi), name, k
    _write_csv(out / f"{name}_model_comparison.csv", cmp)
    winner = min(cmp, key=lambda c: c["AICc"])["model"]
    surface_wins = winner == "SURFACE" and nreg["kprime"] > 1
    # variable-rate BM on the same PCs: the branch rate = mean of the per-PC rates (trace of the rate matrix / k)
    rates = []
    for j in range(k):
        R = multirate_bm(tree, tips, Y[:, j], a.rate_lambda)
        rates.append(R["edge_rate"])
    nodes = R["nodes"]
    mrate = np.vstack(rates).mean(axis=0)                       # NaN at the root (no branch)
    rel = np.log10(mrate / np.nanmedian(mrate[1:]))
    chrono = {"winner": winner, "surface_wins": surface_wins, "k": k,
              "regimes": {T.tipset[i]: reg["node_regime"][i] for i in range(T.N)},
              "convergent": reg["convergent"], "regime_tips": reg["tips"], "regime_count": reg["count"],
              "rates": {branch_key(nodes[i]): float(rel[i]) for i in range(1, len(nodes))}}
    rep += ["Model comparison on these PCs (AICc with n = species x PCs, as surface; "
            f"{name}_model_comparison.csv):"]
    for c in sorted(cmp, key=lambda c: c["AICc"]):
        rep.append(f"  {c['model']:8s} logLik {c['logLik']:9.3f}  parameters {c['n_par']:3d}  AICc {c['AICc']:9.3f}  "
                   f"dAICc {c['dAICc']:7.3f}  weight {c['weight']:.3f}")
    rep.append(f"Lowest AICc: {winner}" + ("" if surface_wins or winner != "SURFACE" else
                                           " (with a single regime: the same as OU1)"))
    rep.append(f"SURFACE path: {len(res['fwd']) - 1} shifts added in the forward phase (AICc "
               f"{res['fwd'][0]['aic']:.2f} -> {res['fwd'][-1]['aic']:.2f}), {len(res['bwd']) - 1} backward collapse "
               f"step(s) -> AICc {fin['aic']:.2f}; {nreg['k']} shifts, {nreg['kprime']} regimes, "
               f"{nreg['kprime_conv']} convergent ({name}_surface_path.csv)")
    for l in sorted(reg["tips"], key=lambda l: (l not in reg["convergent"], l)):
        sh = ["the root" if v == 0 else clade_label(T.nodes[v]) for v, ll in fin["shifts"] if ll == l]
        rep.append(f"  regime {l}{' [CONVERGENT: ' + str(reg['count'][l]) + ' separate shifts]' if l in reg['convergent'] else ''}"
                   f": shifts at {'; '.join(sh)}; species {', '.join(reg['tips'][l])}")
    for r in prow:
        rep.append(f"  {r['PC']}: alpha {r['alpha']:.4g} (half-life {r['half_life']:.3g}, {r['half_life_over_depth']:.3g} "
                   f"of the tree depth), sigma^2 {r['sigma2']:.4g}, optima "
                   + ", ".join(f"{l} {r['theta_' + l]:.3g}" for l in labels))
    rep.append(f"  files {name}_surface_regimes.csv (regime of every branch), {name}_surface_parameters.csv")
    rep.append("  PCA of species data can bias model choice towards OU / early burst (Uyeda, Caetano & Pennell 2015, "
               "Syst. Biol. 64: 677): treat the SURFACE regimes as hypotheses; per-PC single-model fits are in the "
               "models output. SURFACE's forward phase is greedy and its Nelder-Mead fits stop at a tolerance, as in "
               "surface 0.6, which Descriptron reproduces step by step.")
    rel_txt = f"mean variable-rate BM rate of PC1-PC{k}"
    plot_chronogram(tree, {branch_key(nodes[i]): float(rel[i]) for i in range(1, len(nodes))}, "rates",
                    out / f"{name}_chronogram_rates", f"{name}: colour-pattern evolutionary rate ({rel_txt}, "
                    f"lambda = {a.rate_lambda:g})")
    plot_chronogram(tree, chrono["regimes"], "regimes", out / f"{name}_chronogram_regimes",
                    f"{name}: SURFACE regimes on PC1-PC{k} (AICc winner: {winner})", regime_info=chrono)
    rep.append(f"Chronograms: {name}_chronogram_rates.png/.pdf, {name}_chronogram_regimes.png/.pdf")
    return rep, chrono


def left_tree_colouring(chrono: Optional[Dict], how: str) -> Optional[Dict]:
    """how the tanglegram's molecular tree is coloured: SURFACE regimes, variable rates, or nothing."""
    if not chrono or how == "none":
        return None
    use = how
    if how == "model":
        use = "regimes" if chrono.get("surface_wins") else "rates"
    if use == "regimes" and chrono.get("regimes"):
        return {"kind": "regimes", "branch": chrono["regimes"], "info": chrono,
                "what": f"SURFACE regimes (AICc winner {chrono.get('winner')})"}
    if use == "rates" and chrono.get("rates"):
        return {"kind": "rates", "branch": chrono["rates"],
                "what": f"variable-rate BM rate of the colour-pattern PCs (AICc winner {chrono.get('winner', '-')})"}
    return None


# ============================================================================ scoring against reference groups
def load_reference_groups(path: str, columns: List[str]) -> Dict[str, Dict[str, str]]:
    """CSV with a species column and one or more grouping columns -> {column: {species: label}} (empty labels
    skipped; species names normalised as tree tips are)."""
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        sys.exit(f"{path}: no rows")
    lc = {c.strip().lower(): c for c in rows[0]}
    sc = lc.get("species") or list(rows[0])[0]
    out = {}
    for col in columns:
        if col not in rows[0]:
            sys.exit(f"--reference_columns: {col!r} is not a column of {path} ({', '.join(rows[0])})")
        out[col] = {_norm_name(r[sc]): r[col].strip() for r in rows if r[col].strip() and r[sc].strip()}
    return out


def nn_agreement(D: np.ndarray, labels: List[str]) -> Tuple[float, int, np.ndarray]:
    """share of species whose nearest neighbour (smallest distance, itself excluded) has the same label, among the
    species whose label is carried by at least two species. Tied nearest neighbours share the species' vote equally.
    Returns (score, species counted, per-species share of same-label nearest neighbours; NaN if not counted)."""
    lab = np.asarray(labels, dtype=object)
    n = len(lab)
    W = np.array(D, float).copy()
    np.fill_diagonal(W, np.inf)
    mn = W.min(1)
    tol = 1e-12 * max(float(np.nanmax(np.where(np.isfinite(W), W, 0))), 1e-300)
    near = W <= mn[:, None] + tol
    same = lab[:, None] == lab[None, :]
    share = (near & same).sum(1) / near.sum(1)
    cnt = {v: int((lab == v).sum()) for v in set(lab)}
    ok = np.array([cnt[v] >= 2 for v in lab])
    per = np.where(ok, share, np.nan)
    return (float(share[ok].mean()) if ok.any() else float("nan")), int(ok.sum()), per


def adjusted_rand(a, b) -> float:
    """adjusted Rand index (Hubert & Arabie 1985) between two partitions of the same objects."""
    from scipy.special import comb
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    ua, ia = np.unique(a.astype(str), return_inverse=True)
    ub, ib = np.unique(b.astype(str), return_inverse=True)
    M = np.zeros((len(ua), len(ub)))
    np.add.at(M, (ia, ib), 1)
    s_ij = comb(M, 2).sum()
    s_a, s_b = comb(M.sum(1), 2).sum(), comb(M.sum(0), 2).sum()
    tot = comb(len(a), 2)
    exp = s_a * s_b / tot
    den = 0.5 * (s_a + s_b) - exp
    return float((s_ij - exp) / den) if den != 0 else float("nan")


def upgma_merges(D: np.ndarray) -> List[Tuple[int, int, float]]:
    """the merges of UPGMA (average linkage) in order: (cluster i, cluster j, height), clusters numbered by their
    lowest original index, as upgma() builds its tree."""
    n = len(D)
    W = np.array(D, float).copy()
    np.fill_diagonal(W, np.inf)
    size = np.ones(n)
    active = np.ones(n, bool)
    out = []
    for _ in range(n - 1):
        M = np.where(active[:, None] & active[None, :], W, np.inf)
        i, j = np.unravel_index(int(np.argmin(M)), M.shape)
        i, j = min(i, j), max(i, j)
        out.append((int(i), int(j), float(W[i, j])))
        new = (size[i] * W[i] + size[j] * W[j]) / (size[i] + size[j])
        W[i, :], W[:, i] = new, new
        W[i, i] = np.inf
        active[j] = False
        W[j, :], W[:, j] = np.inf, np.inf
        size[i] += size[j]
    return out


def cut_upgma(D: np.ndarray, k: int) -> np.ndarray:
    """UPGMA cut into k clusters (the first n - k merges), as cutree(hclust(as.dist(D), "average"), k); clusters
    numbered 1.. in order of their first member."""
    n = len(D)
    lab = np.arange(n)
    for i, j, _ in upgma_merges(D)[:max(0, n - k)]:
        lab[lab == lab[j]] = lab[i]
    _, first = np.unique(lab, return_index=True)
    order = {lab[f]: r + 1 for r, f in enumerate(sorted(first))}
    return np.array([order[v] for v in lab])


def pam(D: np.ndarray, k: int, max_swaps: int = 1000) -> Tuple[np.ndarray, np.ndarray, float]:
    """k-medoids by Kaufman & Rousseeuw's PAM (BUILD then SWAP), as cluster::pam(diss = TRUE) with its defaults
    (pamonce = FALSE): BUILD adds the object that most lowers the total distance to the nearest medoid; SWAP applies
    the best medoid / non-medoid exchange while it lowers that total. Returns (medoid indices, cluster of each object
    numbered 1.. in medoid order of first appearance, objective = mean distance to the nearest medoid)."""
    D = np.asarray(D, float)
    n = len(D)
    k = max(1, min(k, n))
    med = [int(np.argmin(D.sum(1)))]
    while len(med) < k:
        dn = D[:, med].min(1)
        gain = np.array([np.maximum(dn - D[:, i], 0).sum() if i not in med else -np.inf for i in range(n)])
        med.append(int(np.argmax(gain)))
    tot = D[:, med].min(1).sum()
    for _ in range(max_swaps):
        best = (0.0, None, None)
        for mi in range(k):
            for h in range(n):
                if h in med:
                    continue
                trial = med[:mi] + [h] + med[mi + 1:]
                d = D[:, trial].min(1).sum() - tot
                if d < best[0] - 1e-12 * max(tot, 1.0):
                    best = (d, mi, h)
        if best[1] is None:
            break
        med[best[1]] = best[2]
        tot = D[:, med].min(1).sum()
    near = np.argmin(D[:, med], axis=1)
    _, first = np.unique(near, return_index=True)
    order = {near[f]: r + 1 for r, f in enumerate(sorted(first))}
    return np.array(med), np.array([order[v] for v in near]), float(tot / n)


def score_reference(D: np.ndarray, labels: List[str], iters: int, rng) -> Dict:
    """nearest-neighbour agreement, adjusted Rand index of the UPGMA cut and of PAM k-medoids at k = number of
    labels, and their permutation P values (labels shuffled among the species, the clusters kept)."""
    lab = np.asarray(labels, dtype=object)
    k = len(set(lab))
    nn, ncount, per = nn_agreement(D, list(lab))
    cu = cut_upgma(D, k)
    _, cp, obj = pam(D, k)
    ari_u, ari_p = adjusted_rand(cu, lab), adjusted_rand(cp, lab)
    pn, pu, pp = np.empty(iters), np.empty(iters), np.empty(iters)
    for it in range(iters):
        sh = lab[rng.permutation(len(lab))]
        pn[it] = nn_agreement(D, list(sh))[0]
        pu[it] = adjusted_rand(cu, sh); pp[it] = adjusted_rand(cp, sh)
    P = lambda obs, r: float((1 + (r >= obs - 1e-12).sum()) / (iters + 1))
    return {"n_species": len(lab), "n_labels": k, "n_counted_for_NN": ncount, "NN_agreement": nn,
            "NN_chance_mean": float(pn.mean()), "P_NN": P(nn, pn), "ARI_UPGMA": ari_u, "ARI_UPGMA_chance_mean": float(pu.mean()),
            "P_ARI_UPGMA": P(ari_u, pu), "ARI_PAM": ari_p, "ARI_PAM_chance_mean": float(pp.mean()), "P_ARI_PAM": P(ari_p, pp),
            "PAM_mean_distance_to_medoid": obj, "permutations": iters, "_per": per, "_upgma": cu, "_pam": cp}


def reference_scoring(name: str, S: Dict, a, out: Path, gmap, tree_tip_names: set, rng) -> Tuple[List[str], List[Dict]]:
    """--analyses reference_scores: for every --colour_distance, the colour distance between the species (all species
    with data and a reference label, and / or only those on the tree: --score_species) scored against each
    --reference_columns grouping. Writes <set>__<distance>_reference_scores.csv and _reference_nearest.csv."""
    refs = load_reference_groups(a.reference_groups, [c.strip() for c in a.reference_columns.split(",") if c.strip()])
    labelled = set.intersection(*[set(v) for v in refs.values()])
    rep = [f"### {name}: colour distances scored against reference groups ({Path(a.reference_groups).name}: "
           f"{', '.join(refs)})",
           "NN = share of species whose nearest neighbour (smallest colour distance) carries the same label, among species "
           "whose label has at least two members in the species set (ties share the vote); chance = mean over the label "
           f"permutations; P = share of {a.iterations} permutations at least as high (+1 / +1). ARI = adjusted Rand "
           "index between the reference labels and the UPGMA (average-linkage) tree cut into as many clusters as labels "
           "(as cutree(hclust(average), k)), and PAM k-medoids at the same k (as cluster::pam)",
           "| species set | distance | grouping | n | k | NN (counted) | chance | P | ARI UPGMA (P) | ARI PAM (P) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    rows_all = []
    for dist in dict.fromkeys(a.colour_distance):
        rows, near_rows = [], []
        for sset in a.score_species:
            pool = [s for s in S["species"] if s in labelled]
            if sset == "tree":
                pool = [s for s in pool if s in tree_tip_names]
            if len(pool) < 4:
                rep.append(f"| {sset} | {dist} | - | {len(pool)} | | fewer than 4 species - skipped | | | | |")
                continue
            eng, why = colour_engine(dist, name, S, pool, a, gmap)
            if eng is None:
                rep.append(f"| {sset} | {dist} | - | {len(pool)} | | {why} - skipped | | | | |")
                continue
            D = eng["D"]
            if getattr(a, "_ref_D", None) is not None:
                a._ref_D[(name, dist, sset)] = (list(pool), np.array(D, float), eng["desc"])
            for col, m in refs.items():
                lab = [m[s] for s in pool]
                r = score_reference(D, lab, a.iterations, rng)
                row = {"trait_set": name, "distance": dist, "species_set": sset, "grouping": col,
                       **{k_: v for k_, v in r.items() if not k_.startswith("_")}, "distance_detail": eng["desc"]}
                rows.append(row)
                W = np.array(D, float); np.fill_diagonal(W, np.inf)
                for i, s in enumerate(pool):
                    j = int(np.argmin(W[i]))
                    near_rows.append({"trait_set": name, "distance": dist, "species_set": sset, "grouping": col,
                                      "species": s, "label": lab[i], "nearest": pool[j], "nearest_label": lab[j],
                                      "distance_to_nearest": float(W[i, j]), "share_same_label": r["_per"][i],
                                      "upgma_cluster": int(r["_upgma"][i]), "pam_cluster": int(r["_pam"][i])})
                rep.append(f"| {sset} | {dist} | {col} | {r['n_species']} | {r['n_labels']} | {r['NN_agreement']:.3f} "
                           f"({r['n_counted_for_NN']}) | {r['NN_chance_mean']:.3f} | {_fmt_p(r['P_NN'])} | "
                           f"{r['ARI_UPGMA']:.3f} ({_fmt_p(r['P_ARI_UPGMA'])}) | {r['ARI_PAM']:.3f} "
                           f"({_fmt_p(r['P_ARI_PAM'])}) |")
        _write_csv(out / f"{name}__{dist}_reference_scores.csv", rows)
        _write_csv(out / f"{name}__{dist}_reference_nearest.csv", near_rows)
        rows_all += rows
    rep.append(f"Files: {name}__<distance>_reference_scores.csv, {name}__<distance>_reference_nearest.csv (each "
               "species' nearest neighbour and clusters); reference_scores_summary.csv (all sets)")
    return rep, rows_all


def parse_combine_sets(specs: List[str]) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """--combine_sets NEW=SET1:W1,SET2:W2 -> [(NEW, [(SET1, W1), (SET2, W2)])] (a missing weight is 1)."""
    out = []
    for x in specs:
        new, _, rest = x.partition("=")
        parts = []
        for y in rest.split(","):
            nm, _, w = y.strip().partition(":")
            parts.append((nm.strip(), float(w) if w else 1.0))
        out.append((new.strip(), parts))
    return out


def combine_distances(mats: List[Tuple[List[str], np.ndarray]], weights: List[float]) -> Tuple[List[str], np.ndarray]:
    """weighted sum of species distance matrices, each divided by its median off-diagonal distance over the species
    all matrices have, divided by the sum of the weights: D = sum_i w_i D_i / median(D_i) / sum_i w_i (species in the
    first matrix's order). One matrix is returned as it is (normalising it changes nothing but its unit)."""
    names = [s for s in mats[0][0] if all(s in m[0] for m in mats[1:])]
    subs = [D[np.ix_([nm.index(s) for s in names], [nm.index(s) for s in names])] for nm, D in mats]
    if len(subs) == 1:
        return names, subs[0]
    med = lambda X: float(np.median(X[np.triu_indices_from(X, 1)]))
    return names, sum(w * X / med(X) for w, X in zip(weights, subs)) / sum(weights)


def combined_reference_scoring(a, out: Path) -> Tuple[List[str], List[Dict]]:
    """--combine_sets: each combination of trait sets' colour distances (combine_distances) scored against the
    reference groups like a trait set (score_reference), for every --colour_distance and --score_species the
    components share. Every scoring starts a new random generator seeded with --seed, so a combination's P values do
    not depend on the order of the analyses. Writes <NEW>__<distance>_reference_scores.csv, _reference_nearest.csv and
    _distance.csv (the combined species x species matrix)."""
    refs = load_reference_groups(a.reference_groups, [c.strip() for c in a.reference_columns.split(",") if c.strip()])
    rep, rows_all = [], []
    for new, parts in parse_combine_sets(a.combine_sets):
        rep += [f"### {new}: " + " + ".join(f"{w:g} x {nm}" for nm, w in parts) + " (each distance / its median; "
                "sum / sum of weights), scored against the reference groups",
                "| species set | distance | grouping | n | k | NN (counted) | chance | P | ARI UPGMA (P) | ARI PAM (P) |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for dist in dict.fromkeys(a.colour_distance):
            rows, near_rows = [], []
            for sset in a.score_species:
                got = [a._ref_D.get((nm, dist, sset)) for nm, _ in parts]
                if any(g is None for g in got):
                    rep.append(f"| {sset} | {dist} | - | | | not every component has this distance - skipped | | | | |")
                    continue
                names, D = combine_distances([(g[0], g[1]) for g in got], [w for _, w in parts])
                with open(out / f"{new}__{dist}_{sset}_distance.csv", "w", newline="") as fh:
                    wr = csv.writer(fh); wr.writerow(["species"] + names)
                    for i, s_ in enumerate(names):
                        wr.writerow([s_] + [f"{v:.10g}" for v in D[i]])
                for col, m in refs.items():
                    lab = [m[s_] for s_ in names]
                    r = score_reference(D, lab, a.iterations, np.random.default_rng(a.seed))
                    rows.append({"trait_set": new, "distance": dist, "species_set": sset, "grouping": col,
                                 **{k_: v for k_, v in r.items() if not k_.startswith("_")},
                                 "distance_detail": " + ".join(f"{w:g} x {nm} ({g[2]})" for (nm, w), g in zip(parts, got))})
                    W = np.array(D, float); np.fill_diagonal(W, np.inf)
                    for i, s_ in enumerate(names):
                        j = int(np.argmin(W[i]))
                        near_rows.append({"trait_set": new, "distance": dist, "species_set": sset, "grouping": col,
                                          "species": s_, "label": lab[i], "nearest": names[j], "nearest_label": lab[j],
                                          "distance_to_nearest": float(W[i, j]), "share_same_label": r["_per"][i],
                                          "upgma_cluster": int(r["_upgma"][i]), "pam_cluster": int(r["_pam"][i])})
                    rep.append(f"| {sset} | {dist} | {col} | {r['n_species']} | {r['n_labels']} | {r['NN_agreement']:.3f} "
                               f"({r['n_counted_for_NN']}) | {r['NN_chance_mean']:.3f} | {_fmt_p(r['P_NN'])} | "
                               f"{r['ARI_UPGMA']:.3f} ({_fmt_p(r['P_ARI_UPGMA'])}) | {r['ARI_PAM']:.3f} "
                               f"({_fmt_p(r['P_ARI_PAM'])}) |")
            _write_csv(out / f"{new}__{dist}_reference_scores.csv", rows)
            _write_csv(out / f"{new}__{dist}_reference_nearest.csv", near_rows)
            rows_all += rows
        rep.append("")
    return rep, rows_all


# ============================================================================ Mk model, ancestral states, stochastic maps
class MkTree:
    """a tree in preorder with each node's children, for the discrete-character (Mk) likelihood."""

    def __init__(self, root: Node):
        self.nodes, self.parent, self.length, _ = tree_arrays(root)
        self.N = len(self.nodes)
        self.kids = [[] for _ in range(self.N)]
        for i in range(1, self.N):
            self.kids[self.parent[i]].append(i)
        self.tip_ix = np.array([i for i in range(self.N) if not self.kids[i]])
        self.inner_ix = np.array([i for i in range(self.N) if self.kids[i]])
        self.names = [self.nodes[i].name for i in self.tip_ix]
        self.height = np.zeros(self.N)
        for i in range(1, self.N):
            self.height[i] = self.height[self.parent[i]] + self.length[i]


def mk_index(model: str, k: int) -> np.ndarray:
    """k x k matrix of parameter numbers (0..) for the off-diagonal rates, -1 on the diagonal: ER one rate, SYM
    q_ij = q_ji (pairs in row order of the upper triangle), ARD every rate its own (row order)."""
    I = -np.ones((k, k), int)
    c = 0
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            if model == "ER":
                I[i, j] = 0
            elif model == "SYM":
                if j > i:
                    I[i, j] = I[j, i] = c; c += 1
            else:
                I[i, j] = c; c += 1
    return I


def mk_Q(q: np.ndarray, index: np.ndarray) -> np.ndarray:
    Q = np.where(index >= 0, np.asarray(q, float)[np.maximum(index, 0)], 0.0)
    np.fill_diagonal(Q, -Q.sum(1))
    return Q


def _expm_branches(Q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """exp(Q t) for every branch length t (eigen-decomposition when well conditioned, else scipy expm)."""
    from scipy.linalg import expm
    try:
        ev, V = np.linalg.eig(Q)
        if np.linalg.cond(V) < 1e8:
            Vi = np.linalg.inv(V)
            P = np.real(np.einsum("ij,tj,jk->tik", V, np.exp(np.outer(t, ev)), Vi))
            if np.isfinite(P).all():
                return np.clip(P, 0.0, None)
    except np.linalg.LinAlgError:
        pass
    return np.array([np.clip(expm(Q * ti), 0.0, None) for ti in t])


def mk_root_prior(Lroot: np.ndarray, k: int, root: str) -> np.ndarray:
    """root state weights: 'equal' 1/k each (phytools::fitMk pi = "equal", make.simmap's default), 'fitzjohn' the
    conditional likelihoods normalised (FitzJohn et al. 2009; fitMk pi = "fitzjohn"), 'flat' 1 each (ape::ace, whose
    log-likelihood is therefore 'equal' + log(k))."""
    if root == "fitzjohn":
        return Lroot / Lroot.sum()
    if root == "flat":
        return np.ones(k)
    return np.full(k, 1.0 / k)


def mk_downpass(T: MkTree, X: np.ndarray, Q: np.ndarray):
    """conditional likelihoods (each node's normalised to sum 1), their log scale factors, the branch transition
    matrices and each node's message to its parent. X: tips x states (1 for the observed state)."""
    k = Q.shape[0]
    P = _expm_branches(Q, T.length)
    L = np.zeros((T.N, k))
    logc = np.zeros(T.N)
    msg = np.zeros((T.N, k))
    L[T.tip_ix] = X
    for v in range(T.N - 1, -1, -1):
        if T.kids[v]:
            prod = np.ones(k)
            for c in T.kids[v]:
                prod = prod * msg[c]
                logc[v] += logc[c]
            s = prod.sum()
            if not s > 0:
                return None
            L[v] = prod / s
            logc[v] += np.log(s)
        if v:
            msg[v] = P[v] @ L[v]
    return L, logc, P, msg


def mk_loglik(T: MkTree, X: np.ndarray, Q: np.ndarray, root: str = "equal") -> float:
    r = mk_downpass(T, X, Q)
    if r is None:
        return -np.inf
    L, logc, _, _ = r
    pi = mk_root_prior(L[0], Q.shape[0], root)
    return float(logc[0] + np.log(max((pi * L[0]).sum(), 1e-300)))


def fit_mk(T: MkTree, X: np.ndarray, model: str, root: str = "equal", starts: Optional[List[np.ndarray]] = None) -> Dict:
    """maximum-likelihood Mk fit: log rates by L-BFGS-B from several starts (phytools::fitMk's q.init = states / total
    branch length, x0.1 and x10, and any given starts), the best polished by Nelder-Mead. Rates bounded to
    [q.init x 1e-10, q.init x 1e6] (fitMk: [0, 1e50]; at those bounds the likelihood is flat to < 1e-8)."""
    from scipy.optimize import minimize
    k = X.shape[1]
    I = mk_index(model, k)
    npar = int(I.max()) + 1
    q0 = k / T.length.sum()
    lo, hi = np.log(q0) + np.log(1e-10), np.log(q0) + np.log(1e6)
    f = lambda th: -mk_loglik(T, X, mk_Q(np.exp(th), I), root)
    cand = [np.full(npar, np.log(q0) + d) for d in (0.0, np.log(0.1), np.log(10.0))] + \
        [np.clip(np.log(np.maximum(s, 1e-300)), lo, hi) for s in (starts or []) if len(s) == npar]
    best = None
    for th0 in cand:
        r = minimize(f, th0, method="L-BFGS-B", bounds=[(lo, hi)] * npar, options={"maxiter": 5000, "ftol": 1e-14,
                                                                                    "gtol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    r2 = minimize(f, best.x, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 40000,
                                                           "maxfev": 80000})
    th = np.clip(r2.x, lo, hi) if r2.fun < best.fun else best.x
    q = np.exp(th)
    ll = -f(th)
    n = len(T.tip_ix)
    aicc = -2 * ll + 2 * npar + (2 * npar * (npar + 1) / (n - npar - 1) if n - npar - 1 > 0 else np.inf)
    return {"model": model, "logLik": ll, "k": npar, "AIC": -2 * ll + 2 * npar, "AICc": aicc, "rates": q,
            "Q": mk_Q(q, I), "index": I, "root": root,
            "at_bound": bool(((th - lo) < 1e-6).any() or ((hi - th) < 1e-6).any())}


def mk_marginal(T: MkTree, X: np.ndarray, Q: np.ndarray, root: str = "equal") -> np.ndarray:
    """marginal posterior probability of each state at every node (preorder; tips give their observed state) given
    Q and the root prior: down pass of conditional likelihoods, up pass of the likelihood of the rest of the tree,
    as phytools::ancr(fitMk) and ape::ace(type = "discrete", marginal = FALSE)."""
    L, _, P, msg = mk_downpass(T, X, Q)
    k = Q.shape[0]
    out = np.zeros((T.N, k))
    up = np.zeros((T.N, k))                                  # likelihood of everything outside v's subtree, by v's state
    up[0] = mk_root_prior(L[0], k, root)
    for v in range(T.N):
        if v:
            p = T.parent[v]
            o = up[p].copy()
            for w in T.kids[p]:
                if w != v:
                    o = o * msg[w]
            up[v] = o @ P[v]
            up[v] /= up[v].sum()
        post = up[v] * L[v]
        out[v] = post / post.sum()
    return out


def _sample_branch(Q: np.ndarray, a: int, b: int, t: float, Pab: float, rng) -> List[Tuple[int, float]]:
    """a history of the chain from state a to state b over time t, by uniformization (Hobolth & Stone 2009, an
    exact sampler of the same conditional distribution as make.simmap's rejection sampling). Returns [(state,
    duration), ...] from the parent end."""
    if t <= 0:
        return [(b, 0.0)]
    mu = float(-np.diag(Q).min())
    if mu <= 0:
        return [(a, t)]
    k = Q.shape[0]
    R = np.eye(k) + Q / mu
    pw = [np.eye(k)]
    u = rng.random() * Pab
    cum, n, lp = 0.0, 0, -mu * t
    while True:
        cum += np.exp(lp) * pw[n][a, b]
        if cum >= u or n > 5000:
            break
        n += 1
        lp += np.log(mu * t) - np.log(n)
        pw.append(pw[-1] @ R)
    if n == 0:
        return [(a, t)]
    times = np.sort(rng.random(n)) * t
    seg, cur, last_t = [], a, 0.0
    for i in range(1, n + 1):
        w = R[cur] * pw[n - i][:, b]
        s = int(rng.choice(k, p=w / w.sum()))
        if s != cur:
            seg.append((cur, times[i - 1] - last_t)); last_t = times[i - 1]; cur = s
    seg.append((cur, t - last_t))
    return seg


def simmaps(T: MkTree, X: np.ndarray, Q: np.ndarray, nsim: int, rng, root: str = "equal") -> List[Dict]:
    """stochastic character maps (Huelsenbeck et al. 2003; Bollback 2006) as phytools::make.simmap(Q = fitted Q):
    the root state drawn from root prior x conditional likelihoods, each node's state given its parent's from
    P(t)[parent, .] x its conditional likelihoods (tips fixed at their observed state), then each branch's history
    given its two end states. Returns [{'state': node states, 'segments': per node [(state, duration)] along the
    branch above it}]."""
    L, _, P, _ = mk_downpass(T, X, Q)
    k = Q.shape[0]
    pr = mk_root_prior(L[0], k, root) * L[0]
    maps = []
    for _ in range(nsim):
        st = np.zeros(T.N, int)
        st[0] = int(rng.choice(k, p=pr / pr.sum()))
        segs = [[] for _ in range(T.N)]
        for v in range(1, T.N):
            w = P[v][st[T.parent[v]]] * L[v]
            st[v] = int(rng.choice(k, p=w / w.sum()))
            segs[v] = _sample_branch(Q, st[T.parent[v]], st[v], T.length[v], P[v][st[T.parent[v]], st[v]], rng)
        maps.append({"state": st, "segments": segs})
    return maps


def simmap_counts(T: MkTree, m: Dict, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """transitions i -> j (k x k) and time spent in each state, for one map (describe.simmap's count and times)."""
    C = np.zeros((k, k))
    tm = np.zeros(k)
    for v in range(1, T.N):
        sg = m["segments"][v]
        for s, d in sg:
            tm[s] += d
        for (s1, _), (s2, _) in zip(sg[:-1], sg[1:]):
            C[s1, s2] += 1
    return C, tm


def simmap_origins(T: MkTree, m: Dict, k: int) -> np.ndarray:
    """per state, the number of separate origins that living species of that state descend from: walking up from each
    tip, the most recent transition into its state (or the root, if none on the way); distinct events counted."""
    ev = [set() for _ in range(k)]
    tipstate = m["state"][T.tip_ix]
    for ti, s in zip(T.tip_ix, tipstate):
        v, idx = int(ti), len(m["segments"][ti]) - 1
        while True:
            if v == 0:
                ev[s].add(("root",)); break
            if idx > 0:
                ev[s].add((v, idx)); break
            v = int(T.parent[v])
            idx = len(m["segments"][v]) - 1 if v else 0
    return np.array([len(e) for e in ev])


def discrete_states(spec: str) -> Tuple[Dict[str, str], str, str]:
    """--discrete_trait CSV:COLUMN -> ({species: state}, path, column)."""
    path, col = spec.rsplit(":", 1)
    rows = list(csv.DictReader(open(path, newline="")))
    lc = {c.strip().lower(): c for c in rows[0]}
    sc = lc.get("species") or list(rows[0])[0]
    if col not in rows[0]:
        sys.exit(f"--discrete_trait: no column {col!r} in {path}")
    return {_norm_name(r[sc]): r[col].strip() for r in rows if r[col].strip()}, path, col


MK_COLOURS = ["#E69F00", "#882255", "#009E73", "#56B4E9", "#332288", "#CC79A7", "#D55E00", "#999933", "#8c6d31",
              "#999999"]                    # states in sorted order; no two blues or two pinks among the first five


def plot_mk_tree(T: MkTree, maps: List[Dict], post: np.ndarray, states: List[str], path_stem, title: str,
                 origins_txt: List[str]):
    """the dated tree, each branch painted along its length by the state most stochastic maps are in at that point,
    pies of the marginal ancestral-state probabilities at the internal nodes, tip squares in the observed state."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Wedge
    k = len(states)
    col = [MK_COLOURS[i % len(MK_COLOURS)] for i in range(k)]
    x = T.height
    ntip = len(T.tip_ix)
    y = np.zeros(T.N)
    order = [int(i) for i in T.tip_ix]
    y[order] = np.arange(ntip)
    for v in range(T.N - 1, -1, -1):
        if T.kids[v]:
            y[v] = np.mean([y[c] for c in T.kids[v]])
    H = x.max()
    key_rows = 1 + len(origins_txt)
    fig_h = max(4.5, 0.26 * ntip + 1.5) + 0.17 * key_rows
    fig = plt.figure(figsize=(9.0, fig_h))
    kh = (0.17 * key_rows + 0.3) / fig_h
    ax = fig.add_axes([0.03, kh + 0.04, 0.94, 1 - kh - 0.09])
    nseg = 40
    for v in range(1, T.N):
        p = T.parent[v]
        ax.plot([x[p], x[p]], [y[p], y[v]], color="#8a9399", lw=0.7, zorder=1)
        tt = (np.arange(nseg) + 0.5) / nseg * T.length[v]
        cnt = np.zeros((nseg, k))
        for m in maps:
            edges = np.cumsum([d for _, d in m["segments"][v]])
            ss = [s for s, _ in m["segments"][v]]
            ix = np.minimum(np.searchsorted(edges, tt, side="right"), len(ss) - 1)
            cnt[np.arange(nseg), np.array(ss)[ix]] += 1
        mode = cnt.argmax(1)
        s0 = 0
        while s0 < nseg:                                     # one line per run of the same modal state
            s1 = s0
            while s1 + 1 < nseg and mode[s1 + 1] == mode[s0]:
                s1 += 1
            a0 = x[p] + s0 / nseg * T.length[v]; a1 = x[p] + (s1 + 1) / nseg * T.length[v]
            ax.plot([a0, a1], [y[v], y[v]], color=col[mode[s0]], lw=2.4, solid_capstyle="butt", zorder=2)
            s0 = s1 + 1
    fig.canvas.draw()                                        # pies: circles in inches, whatever the axis scales
    bb = ax.get_position()
    ax.set_xlim(-0.02 * H, H * 1.42); ax.set_ylim(ntip - 0.4, -0.8)
    xu = (H * 1.44) / (bb.width * fig.get_figwidth())          # x units per inch
    yu = (ntip + 0.4) / (bb.height * fig.get_figheight())     # y units per inch
    for v in T.inner_ix:
        a0 = 90.0
        for s in range(k):
            if post[v, s] <= 1e-3:
                continue
            th = 360 * post[v, s]
            w = Wedge((0, 0), 1.0, a0, a0 + th, facecolor=col[s], edgecolor="white", lw=0.3)
            tr = matplotlib.transforms.Affine2D().scale(0.085 * xu, 0.085 * yu).translate(x[v], y[v]) + ax.transData
            w.set_transform(tr); ax.add_patch(w); a0 += th
    for v in T.tip_ix:
        s = int(maps[0]["state"][v])
        ax.scatter([x[v] + 0.012 * H], [y[v]], marker="s", s=22, color=col[s], zorder=4, edgecolor="none")
        ax.text(x[v] + 0.03 * H, y[v], T.nodes[v].name.replace("_", " "), va="center", fontsize=6.5, style="italic")
    from matplotlib.ticker import MaxNLocator
    ages = [v_ for v_ in MaxNLocator(6).tick_values(0, H) if 0 <= v_ <= H * 1.0001]
    ax.set_xticks([H - v_ for v_ in ages]); ax.set_xticklabels([f"{v_:g}" for v_ in ages], fontsize=6.5)
    ax.spines["bottom"].set_bounds(0, H)
    ax.set_xlabel("time before present (tree units; Ma on a chronogram)", fontsize=7.5)
    ax.set_yticks([])
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.set_title(title, fontsize=9)
    lax = fig.add_axes([0.03, 0.005, 0.94, kh])
    lax.axis("off")
    l1 = lax.legend(handles=[Patch(color=col[s], label=states[s].replace("_", " ")) for s in range(k)],
                    loc="upper left", fontsize=6.8, frameon=False, ncol=min(k, 5),
                    title="branches: state of most stochastic maps at each point; pies: marginal ancestral-state "
                          "probabilities; squares: observed state", title_fontsize=6.5, alignment="left")
    lax.add_artist(l1)
    lax.text(0.0, 0.62, "\n".join(origins_txt), fontsize=6.3, va="top", transform=lax.transAxes, family="monospace")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200)
    plt.close(fig)


def mk_section(tree: Node, states_of: Dict[str, str], col: str, a, out: Path, rng, tag: str = "") -> Tuple[List[str], Dict]:
    """--analyses mk_maps: Mk models of a discrete trait on the tree (ER, SYM, ARD by AICc), marginal ancestral
    states, stochastic maps (--simmaps) with transitions and origins per state, and the painted tree."""
    names = [t.name for t in tips_of(tree)]
    have = [s for s in names if s in states_of]
    tr = prune(_copy_tree(tree), set(have)) if len(have) < len(names) else _copy_tree(tree)
    _set_depths(tr)
    T = MkTree(tr)
    states = sorted({states_of[s] for s in T.names})
    k = len(states)
    X = np.zeros((len(T.names), k))
    for i, s in enumerate(T.names):
        X[i, states.index(states_of[s])] = 1.0
    pre = f"mk_{_safe(col)}{tag}"
    rep = [f"### Mk model of {col} ({len(T.names)} species, {k} states: " + ", ".join(
        f"{s} {int(X[:, i].sum())}" for i, s in enumerate(states)) + ")"]
    if len(have) < len(names):
        rep.append(f"  {len(names) - len(have)} tree species without a state dropped")
    fits = []
    for m in a.mk_models:
        starts = []
        if m in ("SYM", "ARD"):
            for f0 in fits:
                Qf = f0["Q"]; I = mk_index(m, k)
                q = np.zeros(int(I.max()) + 1)
                for i in range(k):
                    for j in range(k):
                        if I[i, j] >= 0:
                            q[I[i, j]] = Qf[i, j] if m == "ARD" else 0.5 * (Qf[i, j] + Qf[j, i])
                starts.append(q)
        fits.append(fit_mk(T, X, m, a.mk_root, starts))
    best_aicc = min(f["AICc"] for f in fits)
    w = np.array([np.exp(-0.5 * (f["AICc"] - best_aicc)) for f in fits]); w /= w.sum()
    rows = []
    for f, wi in zip(fits, w):
        f["dAICc"], f["weight"] = f["AICc"] - best_aicc, float(wi)
        r = {"model": f["model"], "logLik": f["logLik"], "n_par": f["k"], "AIC": f["AIC"], "AICc": f["AICc"],
             "dAICc": f["dAICc"], "weight": f["weight"], "rate_at_bound": f["at_bound"], "root_prior": a.mk_root}
        for i in range(k):
            for j in range(k):
                if i != j:
                    r[f"q_{states[i]}->{states[j]}"] = f["Q"][i, j]
        rows.append(r)
    _write_csv(out / f"{pre}_fits.csv", rows)
    best = min(fits, key=lambda f: f["AICc"])
    rep.append(f"  Root prior: {a.mk_root} (phytools::fitMk pi = \"{a.mk_root}\"; ape::ace's log-likelihood sums the "
               "root's conditional likelihoods without the 1/k and is this + log(k) under 'equal'); AICc with n = species")
    for f in fits:
        rep.append(f"  {f['model']:3s} logLik {f['logLik']:9.4f}  parameters {f['k']:2d}  AICc {f['AICc']:9.3f}  "
                   f"dAICc {f['dAICc']:7.3f}  weight {f['weight']:.3f}" + ("  (a rate at its bound)" if f["at_bound"] else ""))
    rep.append(f"  Lowest AICc: {best['model']}" + (f" (rate {best['rates'][0]:.4g} per unit branch length)"
                                                     if best["model"] == "ER" else ""))
    post = mk_marginal(T, X, best["Q"], a.mk_root)
    arows = []
    for v in T.inner_ix:
        arows.append({"node": clade_label(T.nodes[v]), "n_tips": len(tips_of(T.nodes[v])),
                      "age_before_present": float(T.height.max() - T.height[v]),
                      **{f"P_{s}": post[v, i] for i, s in enumerate(states)},
                      "most_probable": states[int(post[v].argmax())], "tips": "; ".join(t.name for t in tips_of(T.nodes[v]))})
    _write_csv(out / f"{pre}_ancestral_states.csv", arows)
    maps = simmaps(T, X, best["Q"], a.simmaps, rng, a.mk_root)
    cnts = np.array([simmap_counts(T, m, k)[0] for m in maps])
    tms = np.array([simmap_counts(T, m, k)[1] for m in maps])
    orig = np.array([simmap_origins(T, m, k) for m in maps])
    q = lambda x, p: float(np.quantile(x, p))
    srows, ptxt = [], []
    for j, s in enumerate(states):
        into = cnts[:, :, j].sum(1)
        srows.append({"state": s, "n_tips": int(X[:, j].sum()), "transitions_into_mean": float(into.mean()),
                      "transitions_into_median": float(np.median(into)), "transitions_into_low95": q(into, 0.025),
                      "transitions_into_up95": q(into, 0.975), "origins_with_living_descendants_mean": float(orig[:, j].mean()),
                      "origins_median": float(np.median(orig[:, j])), "origins_low95": q(orig[:, j], 0.025),
                      "origins_up95": q(orig[:, j], 0.975), "transitions_out_mean": float(cnts[:, j, :].sum(1).mean()),
                      "time_in_state_mean": float(tms[:, j].mean()), "share_of_maps_root_in_state":
                          float(np.mean([m["state"][0] == j for m in maps]))})
        ptxt.append(f"{s.replace('_', ' '):18s} tips {int(X[:, j].sum()):2d}  origins {np.median(orig[:, j]):.0f} "
                    f"({q(orig[:, j], 0.025):.0f}-{q(orig[:, j], 0.975):.0f})  transitions in "
                    f"{into.mean():.1f} ({q(into, 0.025):.0f}-{q(into, 0.975):.0f})")
    _write_csv(out / f"{pre}_origins.csv", srows)
    prow = []
    for i in range(k):
        for j in range(k):
            if i != j:
                c = cnts[:, i, j]
                prow.append({"from": states[i], "to": states[j], "mean": float(c.mean()), "low95": q(c, 0.025),
                             "up95": q(c, 0.975)})
    tot = cnts.sum((1, 2))
    prow.append({"from": "any", "to": "any", "mean": float(tot.mean()), "low95": q(tot, 0.025), "up95": q(tot, 0.975)})
    _write_csv(out / f"{pre}_simmap_transitions.csv", prow)
    plot_mk_tree(T, maps, post, states, out / f"{pre}_simmap_tree",
                 f"{col} on the tree: {best['model']} Mk model, {a.simmaps} stochastic maps",
                 ["origins = separate origins living species descend from (median, 95% interval over maps)"] + ptxt)
    rep += [f"  {a.simmaps} stochastic maps (Q fixed at the {best['model']} estimate, as make.simmap(Q = fitted); root "
            f"from the {a.mk_root} prior x conditional likelihoods): total changes {tot.mean():.1f} "
            f"(95% {q(tot, 0.025):.0f}-{q(tot, 0.975):.0f})",
            "  per state: origins = separate transitions (or the root) that the living species of that state descend "
            "from; transitions in = all changes into the state (some leave no living descendant); median or mean "
            "(95% interval over maps):"]
    rep += ["    " + t for t in ptxt]
    rep.append(f"  Files: {pre}_fits.csv, {pre}_ancestral_states.csv, {pre}_simmap_transitions.csv, {pre}_origins.csv, "
               f"{pre}_simmap_tree.png/.pdf")
    return rep, {"T": T, "X": X, "states": states, "best": best, "fits": fits, "post": post, "maps": maps,
                 "tipset": frozenset(T.names), "prefix": pre}


# ============================================================================ OUM: one optimum per discrete state
def bms_fit(T: OUTree, y: np.ndarray, reg_codes: np.ndarray, nreg: int) -> Dict:
    """Brownian motion with one rate per regime (O'Meara et al. 2006; OUwie BMS, phytools::brownie.lite), each
    branch in the regime of the node it leads to, one root state (GLS): covariance sum_r sigma^2_r x the time two
    tips share in regime r. Maximum likelihood over log sigma^2_r (L-BFGS-B from several starts)."""
    from scipy.optimize import minimize
    oh = np.zeros((T.N, nreg)); oh[np.arange(T.N), reg_codes] = 1.0
    Cr = np.einsum("in,jn,n,nr->rij", T.A, T.A, T.length, oh)          # regimes x tips x tips
    present = np.array([Cr[r].trace() > 0 for r in range(nreg)])
    s2bm = profile_loglik(y, T.C)[1]

    def nll(th):
        s = np.zeros(nreg); s[present] = np.exp(th)
        return -profile_loglik_fixed(y, np.tensordot(s, Cr, 1))[0]
    best = None
    for f in (1.0, 0.3, 3.0):
        r = minimize(nll, np.full(int(present.sum()), np.log(s2bm * f)), method="L-BFGS-B",
                     bounds=[(np.log(s2bm) - 25, np.log(s2bm) + 25)] * int(present.sum()))
        if best is None or r.fun < best.fun:
            best = r
    s = np.full(nreg, np.nan); s[present] = np.exp(best.x)
    ll, z0 = profile_loglik_fixed(y, np.tensordot(np.nan_to_num(s), Cr, 1))
    return {"loglik": ll, "sigma2": s, "root": z0}


def profile_loglik_fixed(y: np.ndarray, V: np.ndarray) -> Tuple[float, float]:
    """log-likelihood of y ~ N(z0, V) with the GLS mean z0 (V fully specified)."""
    from scipy.linalg import solve_triangular
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return -np.inf, np.nan
    ty = solve_triangular(L, y, lower=True); t1 = solve_triangular(L, np.ones(len(y)), lower=True)
    z0 = float(t1 @ ty / (t1 @ t1))
    r = ty - z0 * t1
    return float(-0.5 * (len(y) * np.log(2 * np.pi) + r @ r) - np.log(np.diag(L)).sum()), z0


def oum_fit(T: OUTree, Y: np.ndarray, reg: List[str], extra_starts: Optional[List[List[Tuple[float, float]]]] = None) -> Dict:
    """OU with one optimum per regime (Butler & King 2004) on every column of Y, each its own alpha and sigma^2, the
    regimes shared; each column fitted with ouch::hansen's likelihood (hansen_fit, R's Nelder-Mead) from SURFACE's
    start (sqrt(2 / depth), sqrt(var / depth)) and from any extra starts, keeping the best."""
    fits = []
    for j in range(Y.shape[1]):
        cands = [(np.sqrt(2 / T.depth), np.sqrt(np.var(Y[:, j], ddof=1) / T.depth))]
        cands += [s[j] for s in (extra_starts or [])]
        best = None
        for sa, sg in cands:
            f = hansen_fit(T, Y[:, j], reg, sa, sg)
            if best is None or f["loglik"] > best["loglik"]:
                best = f
        fits.append(best)
    return {"fit": fits, "lnL": float(sum(f["loglik"] for f in fits))}


def node_painting(T: OUTree, mk: Dict, node_states: np.ndarray) -> List[str]:
    """regime of every OUTree node (the branch leading to it) = the discrete state at that node, matching nodes of
    the Mk tree and the OU tree by their tip sets."""
    MT = mk["T"]
    key = {frozenset(t.name for t in tips_of(MT.nodes[v])): v for v in range(MT.N)}
    return [mk["states"][int(node_states[key[T.tipset[i]]])] for i in range(T.N)]


def plot_oum_optima(Y: np.ndarray, tip_states: List[str], states: List[str], fit: Dict, path_stem, title: str,
                    map_theta: Optional[np.ndarray] = None):
    """for each PC: the species' scores by state (dots) and the OUM optimum of each state (bar; with the 95%
    interval over the stochastic maps when given)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kpc = Y.shape[1]
    fig, axs = plt.subplots(1, kpc, figsize=(2.6 * kpc + 0.6, 3.6), squeeze=False)
    rng = np.random.default_rng(0)
    col = {s: MK_COLOURS[i % len(MK_COLOURS)] for i, s in enumerate(states)}
    for j in range(kpc):
        ax = axs[0, j]
        for i, s in enumerate(states):
            v = Y[[t == s for t in tip_states], j]
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=col[s], alpha=0.85, zorder=3, edgecolor="none")
            th = fit["fit"][j]["theta"].get(s, np.nan)
            ax.plot([i - 0.3, i + 0.3], [th, th], color="#222222", lw=1.6, zorder=4)
            if map_theta is not None:
                lo, hi = np.nanquantile(map_theta[:, j, i], [0.025, 0.975])
                ax.plot([i, i], [lo, hi], color="#222222", lw=0.8, zorder=4)
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels([s.replace("_", "\n") for s in states], fontsize=6.3)
        al = fit["fit"][j]["alpha"]
        ax.set_title(f"PC{j + 1}: alpha {al:.3g}, half-life {np.log(2) / al:.3g}", fontsize=8)
        ax.tick_params(labelsize=6.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        lo_, hi_ = ax.get_ylim()
        span = np.ptp(Y[:, j])
        ax.set_ylim(max(lo_, Y[:, j].min() - 1.5 * span), min(hi_, Y[:, j].max() + 1.5 * span))
    fig.suptitle(title, fontsize=8.5)
    fig.text(0.01, 0.005, "dots: species; black bar: OUM optimum on the most probable painting; whisker: 95% over "
             "stochastic-map paintings (optima far outside the data are clipped)", fontsize=6)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200)
    plt.close(fig)


def oum_section(name: str, S: Dict, tree: Node, tips: List[Node], a, out: Path, mk: Dict) -> List[str]:
    """--analyses oum: BM, OU1, OUM (one optimum per discrete state, painted from the Mk reconstruction), BMS (one
    rate per state) and SURFACE on the first colour-pattern PCs, by AICc with n = species x PCs (as surface). OUM and
    BMS are fitted on the painting by the most probable marginal state at each node and on each stochastic map
    (each branch in the state at the node it leads to, as ouch paints regimes)."""
    sc, ev, p = pattern_pc_matrix(S)
    n = len(tips)
    if str(a.surface_pcs) == "auto":
        k = max(2, min(broken_stick_count(ev, n), a.surface_pcs_max)); how = "broken-stick, as --surface_pcs auto"
    else:
        k = int(a.surface_pcs); how = "--surface_pcs"
    k = max(1, min(k, len(ev)))
    T = OUTree(tree)
    order = [[t.name for t in tips].index(nm) for nm in T.names]
    Yt = sc[:, :k][order]
    nobs = n * k
    states = mk["states"]
    rep = [f"### OUM: one colour-pattern optimum per {a.discrete_trait.rsplit(':', 1)[1]} state",
           f"Traits: the first {k} PCs ({100 * ev[:k].sum():.1f}% of the variance; {how}) of the {p} standardised "
           f"colour-pattern features; {n} species x {k} PCs = {nobs} observations for AICc (as surface). Each PC its own "
           "alpha, sigma^2 (and optima); regimes shared. Regimes: each branch in the state at the node it leads to "
           "(ouch's painting), from the most probable marginal Mk state (MAP painting) and from each of "
           f"{min(a.oum_maps, len(mk['maps']))} stochastic maps"]
    # BM, OU1
    bm = [fit_model(Yt[:, j], T.C, "BM", default_bounds(T.C)) for j in range(k)]
    ou1 = oum_fit(T, Yt, ["a"] * T.N)
    cmp = [{"model": "BM", "logLik": sum(f["logLik"] for f in bm), "n_par": 2 * k, "note": "sigma^2 and root per PC"},
           {"model": "OU1", "logLik": ou1["lnL"], "n_par": 3 * k, "note": "one optimum (ouch hansen, stationary root)"}]
    ou1_starts = [[(f["sqrt_alpha"], f["sigma"]) for f in ou1["fit"]]]
    tip_states = [mk_states_of_tip(mk, nm) for nm in T.names]
    cmp += [{"model": "white noise", "logLik": float(sum(white_noise_loglik(Yt[:, j]) for j in range(k))), "n_par": 2 * k,
             "note": "no phylogeny: one mean and variance per PC"},
            {"model": "white noise by state", "logLik": float(sum(white_noise_loglik(Yt[:, j], tip_states) for j in range(k))),
             "n_par": k * (1 + len(set(tip_states))),
             "note": "no phylogeny: one mean per state and one variance per PC (OUM's limit as alpha grows; an ANOVA)"}]
    # MAP painting
    post_states = mk["post"].argmax(1)
    MT = mk["T"]
    post_states[MT.tip_ix] = mk["X"].argmax(1)
    reg_map = node_painting(T, mk, post_states)
    used = sorted(set(reg_map))
    oum = oum_fit(T, Yt, reg_map, ou1_starts)
    codes = np.array([used.index(r) for r in reg_map])
    bms = [bms_fit(T, Yt[:, j], codes, len(used)) for j in range(k)]
    cmp += [{"model": "OUM (MAP painting)", "logLik": oum["lnL"], "n_par": k * (2 + len(used)),
             "note": f"{len(used)} regimes = states; alpha, sigma^2 and one optimum per state per PC"},
            {"model": "BMS (MAP painting)", "logLik": float(sum(b["loglik"] for b in bms)), "n_par": k * (1 + len(used)),
             "note": "one rate per state, one root state per PC"}]
    # SURFACE on the same PCs
    sres = S.get("surface_res")
    if sres is None or sres.get("k") != k or sres.get("tips") != frozenset(T.names):
        res = run_surface(T, Yt, exclude=a.surface_exclude, aic_threshold=a.surface_aic_threshold)
        sres = {"k": k, "tips": frozenset(T.names), "res": res}
        S["surface_res"] = sres
    fin = sres["res"]["final"]; nr = fin["n_regimes"]
    cmp.append({"model": "SURFACE", "logLik": fin["lnL"], "n_par": nr["k"] + k * (2 + nr["kprime"]),
                "note": f"{nr['k']} shifts (root included; each a parameter, as surface), {nr['kprime']} regimes, "
                        f"{nr['kprime_conv']} convergent"})
    for c in cmp:
        c["AICc"] = surface_aicc(c["logLik"], c["n_par"], nobs) if nobs - c["n_par"] - 1 > 0 else np.inf
    best = min(c["AICc"] for c in cmp)
    w = np.array([np.exp(-0.5 * (c["AICc"] - best)) for c in cmp]); w /= w.sum()
    for c, wi in zip(cmp, w):
        c.update(dAICc=c["AICc"] - best, weight=float(wi), trait_set=name, PCs=k)
    # over the stochastic maps
    nm_ = min(a.oum_maps, len(mk["maps"]))
    mrows, theta_maps = [], np.full((nm_, k, len(states)), np.nan)
    aic_of = {c["model"]: c["AICc"] for c in cmp}
    for mi in range(nm_):
        reg = node_painting(T, mk, mk["maps"][mi]["state"])
        u = sorted(set(reg))
        f = oum_fit(T, Yt, reg, ou1_starts)
        cds = np.array([u.index(r) for r in reg])
        b = [bms_fit(T, Yt[:, j], cds, len(u)) for j in range(k)]
        a_oum = surface_aicc(f["lnL"], k * (2 + len(u)), nobs)
        lb = float(sum(x["loglik"] for x in b))
        a_bms = surface_aicc(lb, k * (1 + len(u)), nobs)
        for j in range(k):
            for si, s in enumerate(states):
                theta_maps[mi, j, si] = f["fit"][j]["theta"].get(s, np.nan)
        mrows.append({"map": mi + 1, "n_regimes": len(u), "OUM_logLik": f["lnL"], "OUM_AICc": a_oum,
                      "BMS_logLik": lb, "BMS_AICc": a_bms,
                      "OUM_minus_BM": a_oum - aic_of["BM"], "OUM_minus_OU1": a_oum - aic_of["OU1"],
                      "OUM_minus_SURFACE": a_oum - aic_of["SURFACE"],
                      **{f"alpha_PC{j + 1}": f["fit"][j]["alpha"] for j in range(k)},
                      **{f"theta_PC{j + 1}_{s}": theta_maps[mi, j, si] for j in range(k) for si, s in enumerate(states)}})
    _write_csv(out / f"{name}_oum_maps.csv", mrows)
    if mrows:
        ao = np.array([r["OUM_AICc"] for r in mrows]); ab = np.array([r["BMS_AICc"] for r in mrows])
        q = lambda x: (float(np.median(x)), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))
        cmp.append({"model": f"OUM (over {nm_} maps)", "logLik": float(np.median([r["OUM_logLik"] for r in mrows])),
                    "n_par": "", "AICc": q(ao)[0], "AICc_low95": q(ao)[1], "AICc_up95": q(ao)[2],
                    "dAICc": q(ao)[0] - best, "weight": "",
                    "note": f"median over maps; OUM beats BM in {100 * np.mean(ao < aic_of['BM']):.0f}%, OU1 in "
                            f"{100 * np.mean(ao < aic_of['OU1']):.0f}%, SURFACE in {100 * np.mean(ao < aic_of['SURFACE']):.0f}% "
                            "of maps", "trait_set": name, "PCs": k})
        cmp.append({"model": f"BMS (over {nm_} maps)", "logLik": float(np.median([r["BMS_logLik"] for r in mrows])),
                    "n_par": "", "AICc": q(ab)[0], "AICc_low95": q(ab)[1], "AICc_up95": q(ab)[2],
                    "dAICc": q(ab)[0] - best, "weight": "", "note": "median over maps", "trait_set": name, "PCs": k})
    _write_csv(out / f"{name}_oum_model_comparison.csv", cmp)
    prow = []
    for j in range(k):
        f = oum["fit"][j]
        r = {"trait_set": name, "PC": f"PC{j + 1}", "variance_share": float(ev[j]), "alpha": f["alpha"],
             "sigma2": f["sigma2"], "half_life": np.log(2) / f["alpha"] if f["alpha"] > 0 else np.inf,
             "half_life_over_depth": np.log(2) / f["alpha"] / T.depth if f["alpha"] > 0 else np.inf,
             "stationary_variance": f["sigma2"] / (2 * f["alpha"]) if f["alpha"] > 0 else np.inf,
             "logLik": f["loglik"], "optimiser_code": f["convergence"]}
        for si, s in enumerate(states):
            r[f"theta_{s}"] = f["theta"].get(s, np.nan)
            if mrows:
                r[f"theta_{s}_maps_median"] = float(np.nanmedian(theta_maps[:, j, si]))
                r[f"theta_{s}_maps_low95"] = float(np.nanquantile(theta_maps[:, j, si], 0.025))
                r[f"theta_{s}_maps_up95"] = float(np.nanquantile(theta_maps[:, j, si], 0.975))
            r[f"tip_mean_{s}"] = float(Yt[[mk_states_of_tip(mk, nm) == s for nm in T.names], j].mean())
        r.update({f"BMS_sigma2_{s}": bms[j]["sigma2"][used.index(s)] if s in used else np.nan for s in states})
        prow.append(r)
    _write_csv(out / f"{name}_oum_parameters.csv", prow)
    plot_oum_optima(Yt, tip_states, states, oum, out / f"{name}_oum_optima",
                    f"{name}: colour-pattern PCs by state, OUM optima", theta_maps if mrows else None)
    rep.append(f"Model comparison ({name}_oum_model_comparison.csv):")
    for c in cmp:
        if c["n_par"] == "":
            rep.append(f"  {c['model']:22s} AICc median {c['AICc']:9.3f} (95% {c['AICc_low95']:.3f}-{c['AICc_up95']:.3f})  "
                       f"{c['note']}")
        else:
            rep.append(f"  {c['model']:22s} logLik {c['logLik']:9.3f}  parameters {c['n_par']:3d}  AICc {c['AICc']:9.3f}  "
                       f"dAICc {c['dAICc']:7.3f}  weight {c['weight']:.3f}")
    fixed = [c for c in cmp if c["n_par"] != ""]
    winner = min(fixed, key=lambda c: c["AICc"])["model"]
    om = next(c for c in cmp if c["model"] == "OUM (MAP painting)")
    rep.append(f"  Lowest AICc: {winner}. OUM (MAP painting) vs BM: dAICc {om['AICc'] - aic_of['BM']:+.2f}; vs OU1: "
               f"{om['AICc'] - aic_of['OU1']:+.2f}; vs SURFACE: {om['AICc'] - aic_of['SURFACE']:+.2f} (negative = OUM better)")
    for r in prow:
        rep.append(f"  {r['PC']}: alpha {r['alpha']:.4g} (half-life {r['half_life']:.3g} = {r['half_life_over_depth']:.3g} "
                   f"of the tree depth), sigma^2 {r['sigma2']:.4g}; optima " +
                   ", ".join(f"{s} {r['theta_' + s]:.3g}" for s in states))
    rep += [f"  Files: {name}_oum_model_comparison.csv, {name}_oum_parameters.csv, {name}_oum_maps.csv, "
            f"{name}_oum_optima.png/.pdf",
            "  Reading: OUM below BM, OU1 and SURFACE means the states have distinct colour-pattern optima that "
            "lineages moved towards on the tree. A half-life much shorter than the tree depth means species sit at their "
            "state's optimum whatever their ancestry, close to 'white noise by state' (no phylogeny); a half-life near or "
            "above the depth makes the optima extrapolations (OUM then approaches BM with trends). If the states were "
            "assigned by looking at the same colour patterns, a better fit of OUM is partly built in: it shows that the "
            "states are colour-distinct and how fast lineages track them, not that the states were reached independently "
            "of how they were defined."]
    return rep


def white_noise_loglik(y: np.ndarray, groups: Optional[List[str]] = None) -> float:
    """maximum log-likelihood of independent normal values with one mean (or one mean per group) and one variance."""
    y = np.asarray(y, float)
    r = y - y.mean() if groups is None else y - np.array([y[[g == h for h in groups]].mean() for g in groups])
    s2 = float((r ** 2).mean())
    return float(-0.5 * len(y) * (np.log(2 * np.pi * s2) + 1))


def mk_states_of_tip(mk: Dict, name: str) -> str:
    MT = mk["T"]
    i = MT.names.index(name)
    return mk["states"][int(mk["X"][i].argmax())]


# ============================================================================ ancestral colour pattern
ANC_Z95 = 1.96                       # phytools::fastAnc(CI = TRUE) uses 1.96 for its 95% interval


def patristic_between(a_nodes: List[Node], b_nodes: List[Node]) -> np.ndarray:
    """path length between every node of a_nodes and every node of b_nodes (tips or internal nodes)."""
    return np.array([[a.depth + b.depth - 2 * mrca_depth(a, b) for b in b_nodes] for a in a_nodes])


def bm_ancestral(tips: List[Node], nodes: List[Node], Y: np.ndarray, min_obs: int = 3) -> Dict:
    """Maximum-likelihood ancestral states under Brownian motion for every column of Y (tips x variables; NaN = the
    species lacks the value), with their variances, as phytools::fastAnc(vars = TRUE, CI = TRUE):
      estimate  = mu + C_nS C_SS^-1 (y_S - 1 mu), S the species with the value, mu their GLS root state; the same as
                  the maximum-likelihood ape::ace(method = "ML") and fastAnc (each node in turn made the root, its state the
                  contrast-weighted mean); on the species that have the value it is fastAnc on the tree pruned to them,
                  and a node that pruning would remove (all its species but one lack the value, or all of them) gets
                  the conditional expectation, which lies on the pruned branch it falls on
      variance  = sigma^2_REML / (1' C_(i)^-1 1), C_(i) the covariance of the species in S on the tree rerooted at
                  node i (= fastAnc's (1/v1 + 1/v2)^-1 x mean(pic^2) from the rescaled root branches), sigma^2_REML the
                  mean squared independent contrast on S
    Columns observed in fewer than min_obs species are left NaN. Returns est, var (nodes x p), sigsq, root, n_obs (p)."""
    Y = np.asarray(Y, float)
    n, p = Y.shape
    obs = np.isfinite(Y)
    est = np.full((len(nodes), p), np.nan); var = np.full((len(nodes), p), np.nan)
    sig = np.full(p, np.nan); mu = np.full(p, np.nan)
    C = vcv(tips, tips); Cnt = vcv(nodes, tips)
    D = patristic_between(tips, tips); Dnt = patristic_between(nodes, tips)
    pats: Dict[bytes, List[int]] = {}
    for j in range(p):
        pats.setdefault(obs[:, j].tobytes(), []).append(j)
    for key, cols in pats.items():
        si = np.where(np.frombuffer(key, dtype=bool))[0]
        if len(si) < min_obs:
            continue
        cols = np.array(cols)
        Ci = np.linalg.inv(C[np.ix_(si, si)])
        one = np.ones(len(si))
        Ys = Y[np.ix_(si, cols)]
        w = Ci @ one
        m = (w @ Ys) / (w @ one)
        R = Ys - m
        CiR = Ci @ R
        est[:, cols] = m + Cnt[:, si] @ CiR
        sig[cols] = (R * CiR).sum(0) / (len(si) - 1)
        Dss = D[np.ix_(si, si)]
        v = np.array([1.0 / (one @ np.linalg.solve(0.5 * (d[:, None] + d[None, :] - Dss), one)) for d in Dnt[:, si]])
        var[:, cols] = v[:, None] * sig[cols][None, :]
        mu[cols] = m
    return {"est": est, "var": var, "sigsq": sig, "root": mu, "n_obs": obs.sum(0)}


def k_by_column(Y: np.ndarray, C: np.ndarray, groups: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Blomberg's K of every column of Y (complete data; Kmult of one column, as phytools::phylosig(method = "K")) and,
    for each group of column indices, Kmult of the group (Adams 2014): K = [R'R / R'C^-1 R] / [(tr C - n / 1'C^-1 1) /
    (n - 1)], R the deviations from the GLS mean."""
    n = len(Y)
    Ci = np.linalg.inv(C)
    w = Ci.sum(1)
    R = Y - (w @ Y) / w.sum()
    num, den = (R * R).sum(0), (R * (Ci @ R)).sum(0)
    ex = (np.trace(C) - n / Ci.sum()) / (n - 1)
    K = num / den / ex
    Kg = np.array([num[g].sum() / den[g].sum() / ex for g in groups]) if groups else np.zeros(0)
    return K, Kg


def cell_signal(tips: List[Node], lab: np.ndarray, iters: int, rng) -> Dict:
    """per grid cell (lab: species x cells x 3, NaN where a species lacks the cell): K of L*, a*, b*, Kmult of the
    three, permutation P (species shuffled, as phytools::phylosig(test = TRUE) / geomorph::physignal) on the species
    that have the cell, and the REML Brownian-motion rate of each channel (mean squared contrast)."""
    ns, nc, _ = lab.shape
    out = {k: np.full((nc, 3), np.nan) for k in ("K", "P_K", "sigsq")}
    out.update(Kmult=np.full(nc, np.nan), P_Kmult=np.full(nc, np.nan), n_obs=np.isfinite(lab[..., 0]).sum(0))
    C_all = vcv(tips, tips)
    pats: Dict[bytes, List[int]] = {}
    for c in range(nc):
        pats.setdefault(np.isfinite(lab[:, c, 0]).tobytes(), []).append(c)
    for key, cells in pats.items():
        si = np.where(np.frombuffer(key, dtype=bool))[0]
        if len(si) < 4:
            continue
        cells = np.array(cells)
        C = C_all[np.ix_(si, si)]
        Y = lab[si][:, cells, :].reshape(len(si), -1)                   # columns cell-major: c0L c0a c0b c1L ...
        groups = [np.arange(3 * i, 3 * i + 3) for i in range(len(cells))]
        K, Kg = k_by_column(Y, C, groups)
        ge_k, ge_g = np.ones_like(K), np.ones_like(Kg)
        for _ in range(iters):
            Kp, Kgp = k_by_column(Y[rng.permutation(len(si))], C, groups)
            ge_k += Kp >= K - 1e-12; ge_g += Kgp >= Kg - 1e-12
        out["K"][cells] = K.reshape(-1, 3); out["P_K"][cells] = (ge_k / (iters + 1)).reshape(-1, 3)
        out["Kmult"][cells] = Kg; out["P_Kmult"][cells] = ge_g / (iters + 1)
        Ci = np.linalg.inv(C); w = Ci.sum(1)
        R = Y - (w @ Y) / w.sum()
        out["sigsq"][cells] = ((R * (Ci @ R)).sum(0) / (len(si) - 1)).reshape(-1, 3)
    return out


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted P values (stats::p.adjust(method = "BH")); NaN stays NaN."""
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    v = p[ok]; m = len(v)
    if m:
        o = np.argsort(v)[::-1]
        adj = np.minimum.accumulate(v[o] * m / np.arange(m, 0, -1))
        r = np.empty(m); r[o] = np.minimum(adj, 1.0); q[ok] = r
    return q


def load_cell_geometry(path: str) -> Dict:
    """color_phenomics_homology_v2_1.py --save_cell_geometry JSON -> grid lines, outline, cell -> (row, col, overlap)
    and the cells each specimen left empty."""
    J = json.load(open(path))
    J["_path"] = str(path)
    J["outline"] = np.asarray(J["consensus_outline"], float)
    J["cell_rc"] = {c["cell"]: (int(c["row"]), int(c["col"])) for c in J["cells"]}
    J["cell_overlap"] = {c["cell"]: c.get("overlap") for c in J["cells"]}
    J["blank"] = {fn: set(v.get("missing", [])) for fn, v in J.get("specimens", {}).items()}
    return J


_RASTER_CACHE: Dict[Tuple, Dict] = {}


def _pattern_grid(geom: Dict, cells: List[str], width: int) -> Dict:
    """pixel -> cell index (-1 outside the outline or the grid) for drawing on the consensus outline, and the outline
    pixels. Pixels follow the consensus coordinates (y down, as the images)."""
    key = (id(geom), tuple(cells), width)
    if key in _RASTER_CACHE:
        return _RASTER_CACHE[key]
    from matplotlib.path import Path as MPath
    ol = geom["outline"]
    (x0, y0), (x1, y1) = ol.min(0), ol.max(0)
    pad = 0.02 * (x1 - x0)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    s = width / (x1 - x0)
    H = int(np.ceil((y1 - y0) * s))
    xs = x0 + (np.arange(width) + 0.5) / s; ys = y0 + (np.arange(H) + 0.5) / s
    gx, gy = np.meshgrid(xs, ys)
    inside = MPath(ol).contains_points(np.c_[gx.ravel(), gy.ravel()]).reshape(H, width)
    xl, yl = np.asarray(geom["x_lines"]), np.asarray(geom["y_lines"])
    c = np.searchsorted(xl, gx) - 1; r = np.searchsorted(yl, gy) - 1
    lut = -np.ones((len(yl) + 1, len(xl) + 1), int)
    for i, nm in enumerate(cells):
        if nm in geom["cell_rc"]:
            rr, cc = geom["cell_rc"][nm]; lut[rr, cc] = i
    idx = np.where(inside & (r >= 0) & (c >= 0), lut[np.clip(r, 0, len(yl)), np.clip(c, 0, len(xl))], -1)
    er = inside.copy()
    er[1:] &= inside[:-1]; er[:-1] &= inside[1:]; er[:, 1:] &= inside[:, :-1]; er[:, :-1] &= inside[:, 1:]
    edge = inside & ~er
    edge2 = edge.copy()                                     # 2-pixel outline
    edge2[1:] |= edge[:-1]; edge2[:, 1:] |= edge[:, :-1]
    _RASTER_CACHE[key] = {"idx": idx, "inside": inside, "edge": edge2 & (inside | edge)}
    return _RASTER_CACHE[key]


def pattern_image(geom: Dict, cells: List[str], rgb: np.ndarray, drawn: Optional[np.ndarray] = None, width: int = 120,
                  rotate: int = 0, missing_rgb=(0.85, 0.85, 0.85), outline_rgb=(0.15, 0.15, 0.15)) -> np.ndarray:
    """RGBA image of one colour pattern on the consensus outline: every grid cell filled with its colour (rgb: cells x
    3 in [0, 1]), clipped to the outline; cells not drawn (drawn False or NaN) light grey; the outline dark; outside
    transparent. rotate = degrees anticlockwise (90: the top of the consensus, the head, to the left)."""
    g = _pattern_grid(geom, cells, width)
    idx, inside = g["idx"], g["inside"]
    img = np.zeros(idx.shape + (4,))
    rgb = np.asarray(rgb, float)
    ok = np.isfinite(rgb).all(1) if drawn is None else (np.asarray(drawn, bool) & np.isfinite(rgb).all(1))
    has = idx >= 0
    good = has.copy(); good[has] = ok[idx[has]]
    img[inside, :3] = missing_rgb; img[inside, 3] = 1
    img[good, :3] = rgb[idx[good]]
    img[g["edge"], :3] = outline_rgb; img[g["edge"], 3] = 1
    return np.rot90(img, rotate // 90)


def values_to_rgb(v: np.ndarray, vmin: float, vmax: float, cmap: str = "viridis") -> np.ndarray:
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    t = np.clip((np.asarray(v, float) - vmin) / ((vmax - vmin) or 1.0), 0, 1)
    out = cm(t)[:, :3]
    out[~np.isfinite(v)] = np.nan
    return out


def pattern_distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """mean CIEDE2000 over the cells both patterns have (cells x 3 L*a*b*)."""
    d = ciede2000(lab1, lab2)
    return float(np.nanmean(d)) if np.isfinite(d).any() else np.nan


def branch_pattern_changes(tree: Node, tips: List[Node], nodes: List[Node], est: np.ndarray, lab: np.ndarray,
                           desc_obs: np.ndarray) -> List[Dict]:
    """the colour-pattern change along every branch: CIEDE2000 per grid cell between the parent's reconstructed
    pattern (est: internal nodes x cells x L*a*b*, in the order of `nodes`) and the child's (an internal node's
    reconstruction, or a species' mean pattern, lab: tips x cells x 3, NaN = cell not measured), over the cells the
    child's clade has (desc_obs: nodes x cells; a species' own cells). Returns, in preorder, one dict per branch:
    child, parent, length, de (per-cell CIEDE2000, NaN outside the compared cells), mean (mean over the compared
    cells), rate (mean / branch length: Delta E00 per unit of time on a chronogram; NaN for a zero-length branch)."""
    ni = {id(n): i for i, n in enumerate(nodes)}
    ti = {id(t): i for i, t in enumerate(tips)}
    order, _, _, _ = tree_arrays(tree)
    out = []
    for c in order[1:]:
        P = est[ni[id(c.parent)]]
        if id(c) in ni:
            Q = est[ni[id(c)]]; ok = np.asarray(desc_obs[ni[id(c)]], bool) & np.isfinite(Q).all(1)
        else:
            Q = lab[ti[id(c)]]; ok = np.isfinite(Q).all(1)
        ok &= np.isfinite(P).all(1)
        d = np.full(len(P), np.nan)
        if ok.any():
            d[ok] = ciede2000(P[ok], Q[ok])
        m = float(np.mean(d[ok])) if ok.any() else np.nan
        L = float(c.length)
        out.append({"child": c, "parent": c.parent, "length": L, "de": d, "mean": m, "n_cells": int(ok.sum()),
                    "rate": m / L if L > 0 else np.nan})
    return out


def pattern_bm_branch_rates(S: Dict, tree: Node, tips: List[Node], spec, kmax: int, lam: float) -> Dict:
    """variable-rate Brownian motion (multirate_bm, as phytools::multirateBM) on each of the first k principal
    components of the set's standardised colour-pattern features (pattern_pc_matrix; k from n_axes(spec)); each
    branch's rate = the mean of the PCs' branch rates (as the 'rates' chronogram). Returns rate {branch_key: rate},
    per_pc (k x branches, preorder), nodes, k, ev (variance shares of the k PCs), converged."""
    sc, ev, _ = pattern_pc_matrix(S)
    k = n_axes(ev, spec, kmax, len(tips))
    fits = [multirate_bm(tree, tips, sc[:, i], lam) for i in range(k)]
    per = np.vstack([f["edge_rate"] for f in fits])
    mr = per.mean(0)
    nodes_ = fits[0]["nodes"]
    return {"rate": {branch_key(nodes_[i]): float(mr[i]) for i in range(1, len(nodes_))}, "per_pc": per,
            "nodes": nodes_, "k": k, "ev": ev[:k], "converged": all(f["converged"] for f in fits)}


def _trunc_cmap(name: str, lo: float = 0.0, hi: float = 0.88):
    """a perceptually uniform map without its lightest end (yellow branches vanish on white)."""
    import matplotlib
    from matplotlib.colors import ListedColormap
    return ListedColormap(matplotlib.colormaps[name](np.linspace(lo, hi, 256)), name=f"{name}_{lo:g}_{hi:g}")


def simmap_origin_branches(T: "MkTree", maps: List[Dict]) -> List[Dict[int, int]]:
    """per tip (T.tip_ix order): {branch: number of maps} for the branch carrying the most recent change into the tip's
    state (the node index below that branch; 0 = no change since the root), the walk of simmap_origins."""
    out = []
    for ti in T.tip_ix:
        cnt: Dict[int, int] = {}
        for m in maps:
            v, idx = int(ti), len(m["segments"][ti]) - 1
            while True:
                if v == 0:
                    o = 0; break
                if idx > 0:
                    o = v; break
                v = int(T.parent[v])
                idx = len(m["segments"][v]) - 1 if v else 0
            cnt[o] = cnt.get(o, 0) + 1
        out.append(cnt)
    return out


def state_styles(spec: List[str], states: List[str]) -> Dict[str, Tuple[str, str]]:
    """--state_codes STATE=CODE:#colour ... -> {state: (code, colour)}; states not given get their first letter and the
    Mk palette."""
    given = {}
    for x in spec or []:
        st, v = x.split("=", 1)
        code, _, colr = v.partition(":")
        given[st.strip()] = (code.strip() or st[:1], colr.strip() or None)
    out = {}
    for i, s in enumerate(states):
        code, colr = given.get(s, (s[:1], None))
        out[s] = (code, colr or MK_COLOURS[i % len(MK_COLOURS)])
    return out


def _short_species(s: str) -> str:
    p = s.replace("_", " ").split()
    return f"{p[0][0]}. {' '.join(p[1:])}" if len(p) > 1 else s


def _place_node_pictures(tree: Node, x: Dict, y: Dict, w: float, h: float, n_tips: int,
                         label_h: float = 0.12) -> Dict[Node, Tuple[float, float, float]]:
    """centre and scale of each internal node's picture (w x h data units at scale 1; label_h above it for the node
    number): on the node at full size unless that overlaps a picture already placed (deeper nodes first); then on the
    node at 85% or 70%; then, at 70%, the nearest vertical offset (steps of h / 3, within the tips) that is free. The
    caller draws a leader line to a moved picture."""
    placed: List[Tuple[float, float, float, float]] = []          # cx, cy, half width, half height (label included)
    out = {}
    free = lambda cx, cy, hw, hh: all(abs(cx - px) >= hw + pw or abs(cy - py) >= hh + ph for px, py, pw, ph in placed)
    for nd in sorted(internal_nodes(tree), key=lambda q: (x[id(q)], y[id(q)])):
        cx, cy = x[id(nd)], y[id(nd)]
        choice = None
        for sc in (1.0, 0.85, 0.7):
            if free(cx, cy - label_h / 2, sc * w / 2, (sc * h + label_h) / 2):
                choice = (cy, sc); break
        k = 1
        while choice is None and k < 40:
            for c in (cy + k * h / 3, cy - k * h / 3):
                if -0.5 + h / 2 <= c <= n_tips - 0.5 - h / 2 and free(cx, c - label_h / 2, 0.35 * w, (0.7 * h + label_h) / 2):
                    choice = (c, 0.7); break
            k += 1
        choice = choice or (cy, 0.7)
        sc = choice[1]
        placed.append((cx, choice[0] - label_h / 2, sc * w / 2, (sc * h + label_h) / 2)); out[nd] = (cx, choice[0], sc)
    return out


def plot_pattern_tree(tree: Node, imgs: Dict[int, np.ndarray], node_id: Dict[int, str], tip_state: Dict[str, str],
                      styles: Dict[str, Tuple[str, str]], path_stem, title: str, note: str,
                      branch_colour: Optional[Dict] = None):
    """the dated tree with each internal node's reconstructed pattern at the node and each species' pattern at its tip,
    the discrete state as a coded square, node numbers for the tables. Images are horizontal (rotated).
    branch_colour {'value': {id(node): value of the branch leading to it}, 'label', 'cmap', 'median'}: branches drawn
    thick in a perceptually uniform sequential map on a log scale (positive values; others grey), with a colour bar
    in the values' own units and the median marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    tips = tips_of(tree); n = len(tips)
    x, y = _layout(tree)
    H = max(x[id(t)] for t in tips)
    any_img = next(iter(imgs.values()))
    asp = any_img.shape[1] / any_img.shape[0]                  # width / height of a (rotated) pattern image
    row_in = 0.50 if asp > 1.6 else 0.66                        # compact parts (head, a petal): taller rows
    fig_h = n * row_in + 1.9
    tip_w_in = 0.86 * row_in * asp
    nh = min(0.66, 1.1 / (asp * row_in))                       # node image height in rows (at most 1.1 in wide)
    left_in = max(0.45, 0.5 * nh * asp * row_in + 0.12)        # room for the root's picture
    fig_w = 7.2 + tip_w_in + 3.1 + (left_in - 0.45)
    bar_h = 0.62 if branch_colour else 0.0                      # room for the colour bar below the time axis
    fig_h += bar_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_l, ax_w = 0.25 / fig_w, (fig_w - 0.35) / fig_w
    ax_b, ax_h = (0.85 + bar_h) / fig_h, (n * row_in) / fig_h  # 1.05 in above for the title and the note
    ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h])
    tree_in = 6.9                                              # inches of the axis given to the tree (root to tips)
    xu = H / tree_in                                           # x data units per inch
    yu = 1.0 / row_in                                          # y data units per inch
    bcol, norm, cmap = None, None, None
    if branch_colour:
        from matplotlib.colors import LogNorm
        vals = np.array([v for v in branch_colour["value"].values() if np.isfinite(v) and v > 0])
        cmap = _trunc_cmap(branch_colour.get("cmap", "viridis"))
        if len(vals):
            lo, hi = float(vals.min()), float(vals.max())
            norm = LogNorm(lo, hi if hi > lo else lo * 1.0001)
            bv = branch_colour["value"]
            bcol = lambda nd: (cmap(norm(bv[id(nd)])) if np.isfinite(bv.get(id(nd), np.nan)) and bv[id(nd)] > 0
                               else "#c4c9cd")
    _draw_tree(ax, tree, x, y, lw=1.0, color="#7a848c", bcol=bcol, blw=2.6 if bcol else None)
    hh0 = nh; ww0 = nh * asp * xu / yu
    for nd, (cx, cy, sc) in _place_node_pictures(tree, x, y, ww0, hh0, n).items():
        im = imgs.get(id(nd))
        if im is None:
            continue
        hh, ww = sc * hh0, sc * ww0
        nx_, ny_ = x[id(nd)], y[id(nd)]
        if abs(cy - ny_) > 1e-9:                                # moved off the node: a leader line and a dot
            ax.plot([nx_, cx], [ny_, cy + (hh / 2 if cy < ny_ else -hh / 2)], color="#555", lw=0.6, zorder=4)
            ax.plot([nx_], [ny_], "o", ms=2.2, color="#333", zorder=6)
        ax.imshow(im, extent=(cx - ww / 2, cx + ww / 2, cy + hh / 2, cy - hh / 2), interpolation="nearest", zorder=5,
                  aspect="auto")
        ax.text(cx - ww / 2, cy - hh / 2 - 0.02, node_id[id(nd)], fontsize=5.5, color="#333", ha="left", va="bottom",
                zorder=7, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.75))
    xi = H + 0.05 * xu
    for t in tips:
        ty = y[id(t)]
        ww = 0.86 * asp * xu / yu
        im = imgs.get(id(t))
        if im is not None:
            ax.imshow(im, extent=(xi, xi + ww, ty + 0.43, ty - 0.43), interpolation="nearest", zorder=5, aspect="auto")
        xs = xi + ww + 0.08 * xu
        st = tip_state.get(t.name)
        if st is not None:
            code, colr = styles[st]
            ax.add_patch(plt.Rectangle((xs, ty - 0.33), 0.27 * xu, 0.66, color=colr, zorder=5, lw=0))
            ax.text(xs + 0.135 * xu, ty, code, color="white", fontsize=7.5, fontweight="bold", ha="center",
                    va="center", zorder=6)
        ax.text(xs + 0.36 * xu, ty, t.name.replace("_", " "), fontsize=7.5, style="italic", va="center")
    ax.set_xlim(-left_in * xu, H + (fig_w - 0.35 - tree_in - left_in) * xu)
    ax.set_ylim(n - 0.5, -0.5)
    from matplotlib.ticker import MaxNLocator
    ages = [v for v in MaxNLocator(6).tick_values(0, H) if 0 <= v <= H * 1.0001]
    ax.set_xticks([H - v for v in ages]); ax.set_xticklabels([f"{v:g}" for v in ages], fontsize=7)
    ax.spines["bottom"].set_bounds(0, H)
    ax.set_xlabel("time before present (tree units; Ma on a chronogram)", fontsize=7.5)
    ax.set_yticks([])
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    fig.text(ax_l, 1 - 0.18 / fig_h, title, fontsize=10, va="top")
    import textwrap
    fig.text(ax_l, 1 - 0.42 / fig_h, "\n".join(textwrap.wrap(note, int(fig_w * 15))), fontsize=7, va="top",
             color="#444")
    if styles:
        fig.legend(handles=[Patch(color=c, label=f"{k}  {s.replace('_', ' ')}") for s, (k, c) in styles.items()],
                   loc="lower right", ncol=len(styles), fontsize=7, frameon=False, bbox_to_anchor=(0.99, 0.0))
    if norm is not None:
        import matplotlib as mpl
        from matplotlib.ticker import LogLocator, FuncFormatter
        cax = fig.add_axes([(left_in + 0.25) / fig_w, 0.36 / fig_h, 3.6 / fig_w, 0.13 / fig_h])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
        cax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        cax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        cax.tick_params(labelsize=6.5)
        med = branch_colour.get("median")
        if med is not None and np.isfinite(med) and norm.vmin <= med <= norm.vmax:
            cax.axvline(med, color="white", lw=1.4); cax.axvline(med, color="black", lw=0.6)
        cax.set_title(branch_colour["label"] + (f"  (log scale; line = median {med:.3g})" if med is not None else ""),
                      fontsize=7, loc="left", pad=3)
        cb.outline.set_linewidth(0.4)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=220)
    plt.close(fig)


def plot_pattern_gallery(items: List[Dict], path_stem, title: str, note: str, cbar: Optional[Dict] = None,
                         per_row: int = 10):
    """a grid of upright patterns: items [{'img': RGBA, 'img2': optional second image (uncertainty), 'label': str}]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(items)
    asp = items[0]["img"].shape[1] / items[0]["img"].shape[0]
    two = any("img2" in it for it in items)
    ph = 1.9 if asp < 0.35 else 1.5
    cw = max(1.4, ph * asp * (2.1 if two else 1.05) + 0.15)        # room for the labels
    nr = int(np.ceil(k / per_row))
    fig_w = per_row * cw + 0.4; fig_h = nr * (ph + 0.55) + 1.0 + (0.8 if cbar else 0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    top = 0.75
    for i, it in enumerate(items):
        r, c = divmod(i, per_row)
        x0 = 0.2 + c * cw; y0 = fig_h - top - (r + 1) * (ph + 0.55) + 0.45
        wi = ph * asp
        a1 = fig.add_axes([x0 / fig_w, y0 / fig_h, wi / fig_w, ph / fig_h]); a1.imshow(it["img"], interpolation="nearest")
        a1.axis("off")
        if "img2" in it:
            a2 = fig.add_axes([(x0 + wi * 1.05) / fig_w, y0 / fig_h, wi / fig_w, ph / fig_h])
            a2.imshow(it["img2"], interpolation="nearest"); a2.axis("off")
        fig.text((x0 + 0.02) / fig_w, (y0 - 0.05) / fig_h, it["label"], fontsize=6, va="top", ha="left")
    fig.text(0.2 / fig_w, 1 - 0.15 / fig_h, title, fontsize=10, va="top")
    fig.text(0.2 / fig_w, 1 - 0.42 / fig_h, note, fontsize=7, va="top", color="#444")
    if cbar:
        import matplotlib as mpl
        cax = fig.add_axes([0.2 / fig_w, 0.55 / fig_h, min(4.0, fig_w - 0.4) / fig_w, 0.12 / fig_h])
        mpl.colorbar.ColorbarBase(cax, cmap=mpl.colormaps[cbar["cmap"]], orientation="horizontal",
                                  norm=mpl.colors.Normalize(cbar["vmin"], cbar["vmax"]))
        cax.tick_params(labelsize=6); cax.set_xlabel(cbar["label"], fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=220)
    plt.close(fig)


def plot_convergence_panel(rows: List[Dict], styles, path_stem, title: str, note: str,
                           summary: Optional[Dict[str, str]] = None):
    """one row per state, its origins side by side; each origin: the reconstructed ancestor before the change (and
    the lineage's crown when it is a clade), an arrow, the descendant species' patterns. rows: [{'state', 'head',
    'anc': [(img, label)], 'desc': [(img, label)]}] (one per origin). Images upright."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    asp = rows[0]["anc"][0][0].shape[1] / rows[0]["anc"][0][0].shape[0]
    ph = 1.45 if asp < 0.35 else 1.15
    iw = ph * asp
    cw = max(iw + 0.1, 0.78)
    aw = 0.32                                           # arrow column
    gap = 0.35                                          # between origins
    lab_w = 1.25
    states = list(dict.fromkeys(r["state"] for r in rows))
    blocks = {st: [r for r in rows if r["state"] == st] for st in states}
    bw = lambda r: (len(r["anc"]) + len(r["desc"])) * cw + aw
    row_w = {st: sum(bw(r) for r in b) + gap * (len(b) - 1) for st, b in blocks.items()}
    fig_w = lab_w + max(row_w.values()) + 0.3
    rh = ph + 1.05
    head_h = 0.55 + 0.13 * len(textwrap.wrap(note, int(fig_w * 17)))
    fig_h = len(states) * rh + head_h + 0.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, st in enumerate(states):
        y0 = fig_h - head_h - (i + 1) * rh + 0.62
        code, colr = styles[st]
        fig.add_artist(plt.Line2D([0.1 / fig_w, 1 - 0.1 / fig_w], [(y0 + ph + 0.36) / fig_h] * 2, color=colr, lw=1.2))
        fig.patches.append(plt.Rectangle((0.12 / fig_w, (y0 + ph - 0.24) / fig_h), 0.24 / fig_w, 0.24 / fig_h,
                                         transform=fig.transFigure, color=colr, lw=0))
        fig.text(0.24 / fig_w, (y0 + ph - 0.12) / fig_h, code, fontsize=7.5, color="white", fontweight="bold",
                 ha="center", va="center")
        fig.text(0.12 / fig_w, (y0 + ph - 0.34) / fig_h, f"{st.replace('_', ' ')}\n{len(blocks[st])} origin"
                 + ("s" if len(blocks[st]) != 1 else "") + ((("\n" + summary[st]) if summary and st in summary else "")),
                 fontsize=6.5, va="top", ha="left", color="#222", linespacing=1.3)
        x = lab_w
        for r in blocks[st]:
            fig.text((x + 0.02) / fig_w, (y0 + ph + 0.2) / fig_h, r["head"], fontsize=6, va="center", ha="left",
                     color="#333")
            items = [(im, lb, True) for im, lb in r["anc"]] + [(im, lb, False) for im, lb in r["desc"]]
            for j, (im, lb, is_anc) in enumerate(items):
                a = fig.add_axes([x / fig_w, y0 / fig_h, iw / fig_w, ph / fig_h]); a.imshow(im, interpolation="nearest")
                a.axis("off")
                fig.text((x + iw / 2) / fig_w, (y0 - 0.04) / fig_h, lb, fontsize=5.4, ha="center", va="top",
                         style="normal" if is_anc else "italic", color="#555" if is_anc else "#000", linespacing=1.15)
                x += cw
                if is_anc and j + 1 == len(r["anc"]):
                    fig.text((x - 0.08 + aw / 2) / fig_w, (y0 + ph / 2) / fig_h, "\u2192", fontsize=13, ha="center",
                             va="center", color="#777")
                    x += aw
            x += gap
            fig.add_artist(plt.Line2D([(x - gap / 2) / fig_w] * 2, [(y0 - 0.3) / fig_h, (y0 + ph + 0.1) / fig_h],
                                      color="#dddddd", lw=0.6))
    fig.text(0.12 / fig_w, 1 - 0.12 / fig_h, title, fontsize=10, va="top")
    fig.text(0.12 / fig_w, 1 - 0.38 / fig_h, "\n".join(textwrap.wrap(note, int(fig_w * 17))), fontsize=6.8, va="top",
             color="#444")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=220)
    plt.close(fig)


def plot_cell_maps(geom, cells, panels: List[Dict], path_stem, title: str, note: str):
    """per-cell statistics on the consensus outline: panels [{'v': cells values, 'label', 'vmin', 'vmax', 'cmap',
    'mark': optional bool cells (dotted)}]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    k = len(panels)
    im0 = pattern_image(geom, cells, np.zeros((len(cells), 3)), width=160)
    asp = im0.shape[1] / im0.shape[0]
    ph = 3.6 if asp < 0.35 else 2.6
    pw = ph * asp
    cw = max(pw + 0.35, 1.25)
    fig_w = k * cw + 0.4; fig_h = ph + 2.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, P in enumerate(panels):
        x0 = 0.25 + i * cw
        a = fig.add_axes([x0 / fig_w, 1.05 / fig_h, pw / fig_w, ph / fig_h])
        rgb = values_to_rgb(P["v"], P["vmin"], P["vmax"], P["cmap"])
        a.imshow(pattern_image(geom, cells, rgb, width=160), interpolation="nearest"); a.axis("off")
        if P.get("mark") is not None:
            g = _pattern_grid(geom, cells, 160)
            rr, cc = [], []
            for ci in np.where(P["mark"])[0]:
                yy, xx = np.nonzero(g["idx"] == ci)
                if len(yy):
                    rr.append(yy.mean()); cc.append(xx.mean())
            a.scatter(cc, rr, s=1.2, color="white", lw=0)
        fig.text((x0 + pw / 2) / fig_w, (1.05 + ph + 0.08) / fig_h, P["label"], fontsize=7, ha="center", va="bottom")
        cax = fig.add_axes([x0 / fig_w, 0.62 / fig_h, pw / fig_w, 0.09 / fig_h])
        mpl.colorbar.ColorbarBase(cax, cmap=mpl.colormaps[P["cmap"]], orientation="horizontal",
                                  norm=mpl.colors.Normalize(P["vmin"], P["vmax"]))
        cax.tick_params(labelsize=5.5)
        cax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    fig.text(0.25 / fig_w, 1 - 0.12 / fig_h, title, fontsize=9.5, va="top")
    fig.text(0.25 / fig_w, 0.32 / fig_h, note, fontsize=6.5, va="top", color="#444")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=220)
    plt.close(fig)


def ancestral_pattern_section(name: str, S: Dict, tree: Node, tips: List[Node], a, out: Path, rng,
                              mk: Optional[Dict]) -> List[str]:
    """--analyses ancestral_pattern: L*, a*, b* of every homology grid cell reconstructed at every internal node under
    Brownian motion (bm_ancestral, as phytools::fastAnc with CI), drawn as pictures on the consensus outline."""
    geom = S.get("geometry")
    cells, cix = colour_cell_index(S["full_cols"], a.colour_cells)
    if not cells:
        return [f"Ancestral pattern: no colour cells in {name} (--colour_cells) - skipped"]
    sp_idx = [S["species"].index(t.name) for t in tips]
    lab = S["full_raw"][sp_idx][:, cix]                         # species x cells x 3
    ns = len(tips)
    nobs = np.isfinite(lab[..., 0]).sum(0)
    keep = nobs == ns if a.anc_missing == "complete" else nobs >= max(3, int(np.ceil(a.anc_min_share * ns)))
    rep = [f"### Ancestral colour pattern ({name}): {len(cells)} grid cells x L*, a*, b*; {int((nobs == ns).sum())} "
           f"measured in all {ns} species, {int(keep.sum())} reconstructed (--anc_missing {a.anc_missing}"
           + (f", cells in >= {a.anc_min_share:.0%} of the species" if a.anc_missing == "observed" else "") + ")"]
    cells = [c for c, k in zip(cells, keep) if k]
    lab = lab[:, keep]; nobs = nobs[keep]
    nodes = internal_nodes(tree)
    Y = lab.reshape(ns, -1)
    R = bm_ancestral(tips, nodes, Y)
    est = R["est"].reshape(len(nodes), len(cells), 3)
    se = np.sqrt(R["var"]).reshape(len(nodes), len(cells), 3)
    from scipy.stats import norm
    z = ANC_Z95 if abs(a.anc_ci - 0.95) < 1e-12 else float(norm.ppf(0.5 + a.anc_ci / 2))
    tipset = {id(t): i for i, t in enumerate(tips)}
    desc_obs = np.array([np.isfinite(lab[[tipset[id(t)] for t in tips_of(nd)], :, 0]).any(0) for nd in nodes])
    H = max(t.depth for t in tips)
    node_id = {id(nd): f"n{i + 1}" for i, nd in enumerate(nodes)}
    pre = f"{name}_ancestral_pattern"
    # ---- tables
    post = None; states = []
    if mk is not None:
        MT = mk["T"]; states = mk["states"]
        mk_ix = {frozenset(t.name for t in tips_of(MT.nodes[v])): v for v in range(MT.N)}
        post = {id(nd): mk["post"][mk_ix[frozenset(t.name for t in tips_of(nd))]] for nd in nodes
                if frozenset(t.name for t in tips_of(nd)) in mk_ix}
    nrows = []
    for i, nd in enumerate(nodes):
        r = {"node": node_id[id(nd)], "parent": node_id.get(id(nd.parent), "") if nd.parent else "",
             "n_tips": len(tips_of(nd)), "age_before_present": H - nd.depth, "clade": clade_label(nd),
             "tips": "; ".join(t.name for t in tips_of(nd)),
             "mean_ci_radius_lab": float(np.nanmean(np.sqrt(((z * se[i]) ** 2).sum(-1))))}
        if post is not None and id(nd) in post:
            r.update({f"P_{s}": float(post[id(nd)][k]) for k, s in enumerate(states)})
        nrows.append(r)
    _write_csv(out / f"{pre}_nodes.csv", nrows)
    rc = [re.match(r"^r(\d+)c(\d+)$", c) for c in cells]
    with open(out / f"{pre}_states.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node", "is_tip", "cell", "row", "col", "channel", "estimate", "se", "ci_low", "ci_high",
                    "n_species_with_cell", "descendants_with_cell"])
        for i, nd in enumerate(nodes):
            for j, c in enumerate(cells):
                for k, ch in enumerate("Lab"):
                    e, s_ = est[i, j, k], se[i, j, k]
                    w.writerow([node_id[id(nd)], 0, c, rc[j].group(1) if rc[j] else "", rc[j].group(2) if rc[j] else "",
                                ch, f"{e:.10g}", f"{s_:.10g}", f"{e - z * s_:.10g}", f"{e + z * s_:.10g}", int(nobs[j]),
                                int(desc_obs[i, j])])
        for ti, t in enumerate(tips):
            for j, c in enumerate(cells):
                for k, ch in enumerate("Lab"):
                    e = lab[ti, j, k]
                    w.writerow([t.name, 1, c, rc[j].group(1) if rc[j] else "", rc[j].group(2) if rc[j] else "", ch,
                                "" if not np.isfinite(e) else f"{e:.10g}", "", "", "", int(nobs[j]),
                                int(np.isfinite(e))])
    sig = cell_signal(tips, lab, a.iterations, rng)
    q = bh_adjust(sig["P_Kmult"])
    crow = []
    for j, c in enumerate(cells):
        r = {"cell": c, "row": rc[j].group(1) if rc[j] else "", "col": rc[j].group(2) if rc[j] else "",
             "overlap_with_outline": (geom or {}).get("cell_overlap", {}).get(c, "") if geom else "",
             "n_species_with_cell": int(nobs[j])}
        for k, ch in enumerate("Lab"):
            r[f"K_{ch}"] = sig["K"][j, k]; r[f"P_K_{ch}"] = sig["P_K"][j, k]
        r.update(Kmult_Lab=sig["Kmult"][j], P_Kmult_Lab=sig["P_Kmult"][j], P_Kmult_Lab_BH=q[j])
        for k, ch in enumerate("Lab"):
            r[f"sigsq_{ch}"] = sig["sigsq"][j, k]
        r["sigsq_Lab_sum"] = float(np.nansum(sig["sigsq"][j]))
        crow.append(r)
    _write_csv(out / f"{pre}_cells.csv", crow)
    nsig = int((sig["P_Kmult"] < 0.05).sum()); nq = int((q < 0.05).sum())
    rep += [f"  Brownian motion, maximum-likelihood ancestral states and {100 * a.anc_ci:g}% intervals as "
            "phytools::fastAnc(CI = TRUE) (the maximum-likelihood estimates; ape::ace's numerical optimiser reaches them on standardised data); each cell and channel on its "
            "own, on the species that have the cell; a node's cell is drawn when a descendant species has it",
            f"  Per-cell signal: Kmult(L*, a*, b*) median {np.nanmedian(sig['Kmult']):.3f} (range "
            f"{np.nanmin(sig['Kmult']):.3f}-{np.nanmax(sig['Kmult']):.3f}); P < 0.05 in {nsig} of {len(cells)} cells "
            f"({nq} after Benjamini-Hochberg), {a.iterations} permutations",
            f"  BM rate (REML sigma^2, sum of the three channels, CIELAB units^2 per tree unit): median "
            f"{np.nanmedian(np.nansum(sig['sigsq'], 1)):.3g}"]
    # ---- colour-pattern change along every branch
    branches = branch_pattern_changes(tree, tips, nodes, est, lab, desc_obs)
    rates = np.array([b["rate"] for b in branches])
    med_rate = float(np.nanmedian(rates)) if np.isfinite(rates).any() else np.nan
    bmr = None
    if "bm_rate" in (a.anc_branch_colour or []):
        bmr = pattern_bm_branch_rates(S, tree, tips, a.model_pcs or "auto", a.conv_pcs_max, a.rate_lambda)
    bm_med = float(np.nanmedian(list(bmr["rate"].values()))) if bmr else np.nan
    brows = []
    for b in branches:
        c = b["child"]
        top = np.argsort(-np.nan_to_num(b["de"], nan=-1))[:a.anc_top_cells]
        r = {"parent": node_id[id(b["parent"])], "child": node_id.get(id(c), c.name), "is_tip": int(not c.kids),
             "child_clade": clade_label(c), "child_age_before_present": H - c.depth, "length": b["length"],
             "n_cells_compared": b["n_cells"], "dE00_mean": b["mean"], "dE00_per_Myr": b["rate"],
             "rate_over_median": b["rate"] / med_rate if np.isfinite(med_rate) and med_rate > 0 else np.nan,
             "dE00_per_sqrt_Myr": b["mean"] / np.sqrt(b["length"]) if b["length"] > 0 else np.nan}
        if bmr is not None:
            v = bmr["rate"].get(branch_key(c), np.nan)
            r.update(bm_rate=v, bm_rate_over_median=v / bm_med if bm_med > 0 else np.nan)
        r["top_changing_cells"] = "; ".join(f"{cells[j]} {b['de'][j]:.1f}" for j in top if np.isfinite(b["de"][j]))
        brows.append(r)
    _write_csv(out / f"{pre}_branches.csv", brows)
    tl = float(sum(b["length"] for b in branches if np.isfinite(b["mean"])))
    tree_rate = float(np.nansum([b["mean"] for b in branches]) / tl) if tl > 0 else np.nan
    ordr = sorted([r for r in brows if np.isfinite(r["dE00_per_Myr"])], key=lambda r: -r["dE00_per_Myr"])
    lab_of = lambda r: r["child"] if r["is_tip"] else f"{r['child']} ({r['child_clade']})"
    rep += [f"  Pattern change per branch (mean CIEDE2000 over the cells, parent reconstruction to child; species = "
            f"its mean): median {np.nanmedian([b['mean'] for b in branches]):.2f}, median rate {med_rate:.3g} "
            f"Delta E00 per tree unit (Myr on a chronogram); whole tree (sum of changes / tree length) {tree_rate:.3g}",
            "    fastest: " + "; ".join(f"{lab_of(r)} {r['dE00_per_Myr']:.3g} (x{r['rate_over_median']:.1f})"
                                       for r in ordr[:5]),
            "    slowest: " + "; ".join(f"{lab_of(r)} {r['dE00_per_Myr']:.3g} (x{r['rate_over_median']:.2f})"
                                       for r in ordr[::-1][:5])]
    if bmr is not None:
        o2 = sorted(brows, key=lambda r: -np.nan_to_num(r["bm_rate"], nan=-1))[:5]
        rep.append(f"    variable-rate BM on PC1-{bmr['k']} ({100 * bmr['ev'].sum():.1f}% of the variance"
                   + ("" if bmr["converged"] else "; optimiser did not converge on every PC") + "), fastest: "
                   + "; ".join(f"{lab_of(r)} x{r['bm_rate_over_median']:.1f}" for r in o2))
    S["anc_summary"] = {"branches": branches, "brows": brows, "tree_rate": tree_rate, "median_rate": med_rate,
                        "mean_rate": float(np.nanmean(rates)), "cells": cells, "lab": lab, "nobs": nobs,
                        "geometry": geom, "bm": bmr, "tips": tips}
    # ---- images
    if geom is None:
        rep.append("  No --cell_geometry for this set: pictures need the grid geometry "
                   "(color_phenomics_homology_v2_1.py --save_cell_geometry) - tables only")
        rep.append(f"  Files: {pre}_states.csv, {pre}_nodes.csv, {pre}_cells.csv, {pre}_branches.csv")
        return rep
    W = a.pattern_width
    rot = a.pattern_rotate
    imgs_h, imgs_u = {}, {}
    for i, nd in enumerate(nodes):
        rgb = lab_to_rgb(est[i]); rgb[~np.isfinite(est[i]).all(1)] = np.nan
        imgs_h[id(nd)] = pattern_image(geom, cells, rgb, desc_obs[i], W, rot)
        imgs_u[id(nd)] = pattern_image(geom, cells, rgb, desc_obs[i], W, 0)
    tip_lab = {}
    for ti, t in enumerate(tips):
        rgb = lab_to_rgb(lab[ti]); rgb[~np.isfinite(lab[ti]).all(1)] = np.nan
        imgs_h[id(t)] = pattern_image(geom, cells, rgb, None, W, rot)
        imgs_u[id(t)] = pattern_image(geom, cells, rgb, None, W, 0)
        tip_lab[t.name] = lab[ti]
    tip_state = {t.name: mk_states_of_tip(mk, t.name) for t in tips if mk is not None and t.name in mk["T"].names}
    styles = state_styles(a.state_codes, states) if states else {}
    disc_col = a.discrete_trait.rsplit(":", 1)[1] if a.discrete_trait else ""
    note = (f"Brownian-motion ML ancestral states per grid cell and channel (L*, a*, b*), as phytools::fastAnc; {ns} "
            f"species means ({sum(S['counts'].get(t.name, 0) for t in tips)} specimens); CIELAB from the photographs "
            f"(sRGB, D65), drawn in sRGB on the consensus outline; grey = cell not measured in the clade. Node numbers "
            f"as in {pre}_nodes.csv" + (f"; squares: {disc_col}" if tip_state else ""))
    # ---- branch colours: rate of colour-pattern change
    modes = list(dict.fromkeys(a.anc_branch_colour or ["none"]))
    bc_data = {}
    if "rate_de00" in modes:
        bc_data["rate_de00"] = {"value": {id(b["child"]): b["rate"] for b in branches}, "label": "\u0394E00 per Myr",
                                "cmap": "viridis", "median": med_rate}
    if "bm_rate" in modes and bmr is not None:
        bv = {id(b["child"]): bmr["rate"].get(branch_key(b["child"]), np.nan) for b in branches}
        bc_data["bm_rate"] = {"value": bv, "cmap": "cividis", "median": float(np.nanmedian(list(bv.values()))),
                              "label": f"variable-rate BM rate (mean of PC1-{bmr['k']}, standardised features)"}
    what = {"rate_de00": "branches coloured by the rate of colour-pattern change, mean CIEDE2000 over the grid cells "
                         "between the reconstructed parent and the child (a species: its mean pattern) divided by the "
                         "branch length; under Brownian motion change grows with the square root of time, so short "
                         "branches read faster, and tip branches also carry differences between specimens",
            "bm_rate": f"branches coloured by the variable-rate Brownian-motion rate (phytools::multirateBM, lambda "
                       f"{a.rate_lambda:g}), mean over the first {bmr['k'] if bmr else 0} PCs of the standardised "
                       f"pattern features", "none": ""}
    tree_files = []
    for mi, mode in enumerate(modes):
        if mode != "none" and mode not in bc_data:
            continue
        stem = f"{pre}_tree" if mi == 0 else f"{pre}_tree_{mode}"
        plot_pattern_tree(tree, imgs_h, node_id, tip_state, styles, out / stem,
                          f"{name}: reconstructed colour pattern at every node"
                          + {"rate_de00": " and the rate of pattern change on every branch",
                             "bm_rate": " and the variable-rate BM rate on every branch", "none": ""}[mode],
                          note + (". " + what[mode][:1].upper() + what[mode][1:] if what[mode] else ""),
                          bc_data.get(mode))
        tree_files.append(f"{stem}.png/.pdf")
    hw = np.sqrt(((z * se) ** 2).sum(-1))                       # CI half-width radius in CIELAB units per cell
    vmax = float(np.nanpercentile(hw[desc_obs], 99)) if np.isfinite(hw).any() else 1.0
    items = []
    for i, nd in enumerate(nodes):
        u = values_to_rgb(hw[i], 0.0, vmax, "magma_r")
        lbl = f"{node_id[id(nd)]}  {H - nd.depth:.1f} Ma\n{len(tips_of(nd))} tips: " \
              f"{_short_species(tips_of(nd)[0].name)}..\nmean CI ±{np.nanmean(hw[i][desc_obs[i]]):.1f}"
        items.append({"img": imgs_u[id(nd)], "img2": pattern_image(geom, cells, u, desc_obs[i], W, 0), "label": lbl})
    plot_pattern_gallery(items, out / f"{pre}_nodes",
                         f"{name}: reconstructed pattern (left) and its {100 * a.anc_ci:g}% interval (right) at each node",
                         "interval = radius of the per-channel interval box, sqrt(sum over L*, a*, b* of the half-widths^2), "
                         "CIELAB units (about Delta E76); darker = less certain",
                         {"cmap": "magma_r", "vmin": 0.0, "vmax": vmax,
                          "label": f"{100 * a.anc_ci:g}% interval radius (CIELAB units)"}, per_row=min(10, len(items)))
    files = tree_files + [f"{pre}_nodes.png/.pdf"]
    # ---- per-cell maps
    Ks = sig["K"]; kmax = float(np.nanpercentile(np.r_[Ks.ravel(), sig["Kmult"]], 99))
    s2 = sig["sigsq"]; s2t = np.nansum(s2, 1)
    panels = [{"v": Ks[:, k], "label": f"K  {ch}*", "vmin": 0.0, "vmax": kmax, "cmap": "viridis",
               "mark": sig["P_K"][:, k] < 0.05} for k, ch in enumerate("Lab")]
    panels.append({"v": sig["Kmult"], "label": "Kmult (L*, a*, b*)", "vmin": 0.0, "vmax": kmax, "cmap": "viridis",
                   "mark": q < 0.05})
    for k, ch in enumerate("Lab"):
        panels.append({"v": np.log10(s2[:, k]), "label": f"log10 rate  {ch}*", "cmap": "cividis",
                       "vmin": float(np.nanpercentile(np.log10(s2), 1)), "vmax": float(np.nanpercentile(np.log10(s2), 99))})
    panels.append({"v": np.log10(s2t), "label": "log10 rate  (sum)", "cmap": "cividis",
                   "vmin": float(np.nanpercentile(np.log10(s2t), 1)), "vmax": float(np.nanpercentile(np.log10(s2t), 99))})
    plot_cell_maps(geom, cells, panels, out / f"{pre}_cell_maps",
                   f"{name}: phylogenetic signal and Brownian-motion rate per grid cell ({ns} species)",
                   f"K = Blomberg's K (Kmult for the three channels); white dots: permutation P < 0.05 ({a.iterations} "
                   "permutations; Kmult after Benjamini-Hochberg). Rate = REML sigma^2 (CIELAB units^2 per tree unit), "
                   "log10")
    files.append(f"{pre}_cell_maps.png/.pdf")
    # ---- convergence: origins from the stochastic maps
    if mk is not None and tip_state:
        MT = mk["T"]
        by_set = {frozenset(t.name for t in tips_of(nd)): nd for nd in nodes}
        by_set.update({frozenset([t.name]): t for t in tips})
        ob = simmap_origin_branches(MT, mk["maps"])
        nmaps = len(mk["maps"])
        orows, panel_rows, crows = [], [], []
        def node_lab(nd):
            return lab[tipset[id(nd)]] if id(nd) in tipset else est[nodes.index(nd)]
        for s in states:
            groups: Dict[int, List[Tuple[str, float]]] = {}
            for k_, ti in enumerate(MT.tip_ix):
                nm = MT.nodes[ti].name
                if tip_state.get(nm) != s:
                    continue
                v, c_ = max(ob[k_].items(), key=lambda kv: (kv[1], -kv[0]))
                groups.setdefault(v, []).append((nm, c_ / nmaps))
            origins = []
            for v, mem in sorted(groups.items(), key=lambda kv: -MT.height[kv[0]]):
                below = by_set[frozenset(t.name for t in tips_of(MT.nodes[v]))]
                before = by_set[frozenset(t.name for t in tips_of(MT.nodes[MT.parent[v]]))] if v else below
                origins.append({"v": v, "below": below, "before": before, "members": mem})
            for oi, o in enumerate(origins):
                bf, bl = o["before"], o["below"]
                pS = post.get(id(bf))[states.index(s)] if (post is not None and id(bf) in post) else np.nan
                share = float(np.mean([m[1] for m in o["members"]]))
                desc = [m[0] for m in o["members"]]
                d_anc_desc = float(np.mean([pattern_distance(node_lab(bf), tip_lab[d]) for d in desc]))
                orows.append({"state": s, "origin": oi + 1, "origin_branch_to": clade_label(bl) if o["v"] else "root",
                              "node_before_change": node_id.get(id(bf), bf.name), "age_before_change":
                                  H - bf.depth, "P_state_at_node_before": pS,
                              "share_of_maps_with_this_origin": share, "n_descendant_species": len(desc),
                              "descendants": "; ".join(desc), "dE00_ancestor_to_descendants": d_anc_desc})
                anc_items = [(imgs_u[id(bf)], f"{node_id.get(id(bf), '')} {H - bf.depth:.1f} Ma\nbefore change"
                                              f"\nP({styles[s][0]}) {pS:.2f}")]
                if bl.kids and bl is not bf and len(desc) > 1:
                    anc_items.append((imgs_u[id(bl)], f"{node_id[id(bl)]} {H - bl.depth:.1f} Ma\nlineage crown"))
                panel_rows.append({"state": s, "anc": anc_items,
                                   "desc": [(imgs_u[id(by_set[frozenset([d])])], _short_species(d).replace(" ", "\n", 1))
                                            for d in desc],
                                   "head": f"origin {oi + 1} ({100 * share:.0f}% of maps)"})
            if len(origins) >= 2:
                anc_nodes = list({id(o["before"]): o["before"] for o in origins}.values())
                da = [pattern_distance(node_lab(x_), node_lab(y_)) for i_, x_ in enumerate(anc_nodes)
                      for y_ in anc_nodes[i_ + 1:]]
                dd = [pattern_distance(tip_lab[m1[0]], tip_lab[m2[0]]) for i_, o1 in enumerate(origins)
                      for o2 in origins[i_ + 1:] for m1 in o1["members"] for m2 in o2["members"]]
                crows.append({"state": s, "n_origins": len(origins), "n_distinct_ancestors": len(anc_nodes),
                              "dE00_between_ancestors_mean": float(np.mean(da)) if da else np.nan,
                              "dE00_between_descendants_of_different_origins_mean": float(np.mean(dd)),
                              "descendants_minus_ancestors": float(np.mean(dd) - np.mean(da)) if da else np.nan})
        _write_csv(out / f"{pre}_origins.csv", orows)
        tipD = [pattern_distance(tip_lab[x_.name], tip_lab[y_.name]) for i_, x_ in enumerate(tips) for y_ in tips[i_ + 1:]]
        if crows:
            for r in crows:
                r["dE00_between_all_species_mean"] = float(np.mean(tipD))
            _write_csv(out / f"{pre}_convergence.csv", crows)
        if panel_rows:
            summ = {r["state"]: f"mean dE00 between\nancestors {r['dE00_between_ancestors_mean']:.1f}\n"
                                f"descendants {r['dE00_between_descendants_of_different_origins_mean']:.1f}\n"
                                f"(all species {np.mean(tipD):.1f})" for r in crows}
            plot_convergence_panel(panel_rows, styles, out / f"{pre}_convergence",
                                   f"{name}: reconstructed ancestors at each origin of every {disc_col} and their "
                                   "descendants",
                                   f"origins: the branch carrying the most recent change into the state, per species, in "
                                   f"most of {nmaps} stochastic maps ({mk['best']['model']} Mk); 'before change' = the "
                                   "node above that branch (Brownian-motion ML pattern); P = its marginal probability "
                                   "of already being in the state. dE00 = mean CIEDE2000 over the grid cells: between the "
                                   "ancestors before the separate origins, and between descendants of different "
                                   "origins. Ancestral estimates are weighted means of the species: they look smoother "
                                   "and more alike than real beetles, which works against seeing convergence",
                                   summ)
            files.append(f"{pre}_convergence.png/.pdf")
        rep.append(f"  Origins (most frequent origin branch per species over {nmaps} maps): " + "; ".join(
            f"{s} {sum(1 for r in orows if r['state'] == s)}" for s in states))
        for r in crows:
            rep.append(f"    {r['state']}: mean CIEDE2000 between the ancestors before the {r['n_origins']} origins "
                       f"{r['dE00_between_ancestors_mean']:.2f}, between descendants of different origins "
                       f"{r['dE00_between_descendants_of_different_origins_mean']:.2f} (all species pairs "
                       f"{np.mean(tipD):.2f})")
    rep.append(f"  Files: {pre}_states.csv (node x cell x channel), {pre}_nodes.csv, {pre}_cells.csv, {pre}_branches.csv"
               + (f", {pre}_origins.csv, {pre}_convergence.csv" if mk is not None else "") + ", " + ", ".join(files))
    return rep


# ============================================================================ a discrete trait drawn on the tree (discrete_tree)
def fitch(tree: Node, state_of: Dict[str, str]) -> Tuple[int, Dict[int, set]]:
    """Fitch (1971) parsimony on a (possibly multifurcating) tree: down pass (a node's set = the intersection of its
    children's sets, or their union at the cost of one change), up pass (a node keeps the part of its set shared with
    its parent's final set, else its own set). Returns the minimum number of changes and each node's final set
    (id(node) -> set of states)."""
    S, F = {}, {}
    score = 0

    def down(n):
        nonlocal score
        if not n.kids:
            S[id(n)] = {state_of[n.name]}; return S[id(n)]
        sets = [down(k) for k in n.kids]
        inter = set.intersection(*sets)
        if inter:
            S[id(n)] = inter
        else:
            S[id(n)] = set.union(*sets); score += 1
        return S[id(n)]

    def up(n, pf):
        F[id(n)] = set(S[id(n)]) if pf is None else ((S[id(n)] & pf) or set(S[id(n)]))
        for k in n.kids:
            up(k, F[id(n)])
    down(tree); up(tree, None)
    return score, F


def state_clades(tree: Node, state_of: Dict[str, str], state: str) -> List[List[str]]:
    """the largest clades whose species all have the state (a single species counts as a clade)."""
    acc = []

    def rec(n):
        ts = tips_of(n)
        if all(state_of[t.name] == state for t in ts):
            acc.append([t.name for t in ts]); return
        for k in n.kids:
            rec(k)
    rec(tree)
    return acc


def plot_discrete_tree(tree: Node, state_of: Dict[str, str], styles: Dict[str, Tuple[str, str]], path_stem, title: str,
                       note: str, labels: Optional[Dict[str, str]] = None, imgs: Optional[Dict[str, np.ndarray]] = None,
                       summary: Optional[List[str]] = None, scale_bar: Optional[float] = None):
    """the dated tree, branches coloured by the Fitch reconstruction (unambiguous state; ambiguous: grey dashed),
    each species with its state code in its colour, its name (and labels[species] in brackets) and a picture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    score, F = fitch(tree, state_of)
    tips = tips_of(tree); n = len(tips)
    x, y = _layout(tree)
    Hd = max(x[id(t)] for t in tips) or 1.0
    row = 0.42
    head = 0.75 + 0.15 * len(textwrap.wrap(note, 150))
    foot = 0.35 + 0.17 * len(summary or [])
    fig_w = 12.0; fig_h = n * row + head + foot
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.03, foot / fig_h, 0.42, n * row / fig_h])
    grey = "#c9c9c9"
    bc = lambda nd: styles[next(iter(F[id(nd)]))][1] if len(F[id(nd)]) == 1 else grey
    for nd in internal_nodes(tree):
        for k in nd.kids:
            c = bc(k)
            ax.plot([x[id(nd)], x[id(k)]], [y[id(k)]] * 2, color=c, lw=2.4 if c != grey else 1.4, solid_capstyle="butt",
                    ls="-" if c != grey else (0, (3, 2)))
        ax.plot([x[id(nd)]] * 2, [y[id(nd.kids[0])], y[id(nd.kids[-1])]], color=bc(nd) if nd is not tree else grey, lw=1.4)
    ax.set_xlim(0, Hd * 1.01); ax.set_ylim(n - 0.5 + 0.9, -0.5); ax.axis("off")
    if scale_bar is None:
        from matplotlib.ticker import MaxNLocator
        scale_bar = float(MaxNLocator(3).tick_values(0, Hd / 3)[1])
    ax.plot([0, scale_bar], [n - 0.2] * 2, color="#333", lw=1.4)
    ax.text(0, n + 0.05, f"{scale_bar:g} tree units (Ma on a chronogram)", ha="left", va="top", fontsize=8)
    for t in tips:
        yf = (foot + (n + 0.4 - y[id(t)]) / (n + 0.9) * n * row) / fig_h      # the tip's height in the figure
        st = state_of[t.name]; code, colr = styles[st]
        fig.text(0.455, yf, code, fontsize=9, fontweight="bold", color="white", va="center", ha="center",
                 bbox=dict(boxstyle="round,pad=0.25", fc=colr, ec="none"))
        fig.text(0.475, yf, t.name.replace("_", " ") + (f"   ({labels[t.name]})" if labels and t.name in labels else ""),
                 fontsize=9, style="italic", va="center")
        if imgs and t.name in imgs:
            im = imgs[t.name]; h_in = 0.36; w_in = h_in * im.shape[1] / im.shape[0]
            iax = fig.add_axes([0.70, yf - h_in / 2 / fig_h, w_in / fig_w, h_in / fig_h]); iax.imshow(im); iax.axis("off")
    fig.text(0.03, 1 - 0.35 / fig_h, title, fontsize=13, fontweight="bold")
    fig.text(0.03, 1 - 0.62 / fig_h, "\n".join(textwrap.wrap(note, 150)), fontsize=9, color="#444", va="top")
    for i, s_ in enumerate(summary or []):
        fig.text(0.03, (foot - 0.3 - i * 0.17) / fig_h, s_, fontsize=8.5, color="#222")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=150)
    plt.close(fig)
    return score


def discrete_tree_section(root: Node, a, out: Path) -> List[str]:
    """--analyses discrete_tree: the --discrete_trait states on the tree (pruned to the species with a state), Fitch
    parsimony branch colours, the number of separate clades per state, codes and colours from --state_codes (their
    order is the legend's), optional --discrete_label_column printed beside each name and --tip_images pictures."""
    states_of, path, col = discrete_states(a.discrete_trait)
    names = {t.name for t in tips_of(root)}
    keep = {s for s in states_of if s in names}
    if len(keep) < 3:
        return [f"discrete_tree: only {len(keep)} tree species have a {col} state - skipped"]
    tree = prune(_copy_tree(root), keep)
    given = [x.split("=", 1)[0].strip() for x in a.state_codes]
    present = sorted({states_of[s] for s in keep})
    order = [s for s in given if s in present] + [s for s in present if s not in given]
    styles = state_styles(a.state_codes, order)
    labels = None
    if a.discrete_label_column:
        rows = list(csv.DictReader(open(path, newline="")))
        sc = {c.strip().lower(): c for c in rows[0]}.get("species") or list(rows[0])[0]
        labels = {_norm_name(r[sc]): r[a.discrete_label_column] for r in rows if a.discrete_label_column in r}
    imgs = None
    if a.tip_images:
        imgs = load_tip_images(a, "", [t.name for t in tips_of(tree)])
    summ = []
    for s in order:
        cl = state_clades(tree, states_of, s)
        nt = sum(1 for t in tips_of(tree) if states_of[t.name] == s)
        summ.append(f"{styles[s][0]} = {s.replace('_', ' ')}: {nt} species in {len(cl)} separate clade(s) on the tree")
    stem = out / f"{_safe(col)}_on_tree"
    score = plot_discrete_tree(tree, states_of, styles, stem, f"{col.replace('_', ' ')} on the tree, {len(keep)} species",
                               f"Branch colours: Fitch parsimony reconstruction ({fitch(tree, states_of)[0]} changes at "
                               f"minimum); grey dashed = ambiguous. Descriptive only (model-based: --analyses mk_maps).",
                               labels, imgs, summ, a.discrete_scale_bar)
    return [f"## {col} on the tree (discrete_tree)", f"Fitch parsimony: at least {score} changes; " + "; ".join(summ),
            f"Files: {stem.name}.png/.pdf", ""]


# ============================================================================ rates of colour-pattern change between body parts
def make_pd(x: np.ndarray, eig_tol: float = 1e-6, posd_tol: float = 1e-8, maxit: int = 50) -> np.ndarray:
    """geomorph's makePD (4.0.7 / 4.1.1), transcribed: alternating projection onto the positive semidefinite matrices
    (eigenvalues <= eig_tol x the largest dropped) with Dykstra's correction, then the smallest eigenvalues raised to
    posd_tol x the largest and the matrix rescaled to keep its diagonal."""
    X = np.array(x, float); n = X.shape[0]; D = np.zeros_like(X)
    for _ in range(maxit):
        Y = X; R = Y - D
        d, Q = np.linalg.eigh(R); d, Q = d[::-1], Q[:, ::-1]
        keep = d > eig_tol * d[0]
        if not keep.any():
            raise ValueError("matrix seems negative semi-definite")
        Q = Q[:, keep]
        X = (Q * d[keep]) @ Q.T
        D = X - R
        if np.abs(Y - X).sum(1).max() / np.abs(Y).sum(1).max() <= 1e-6:
            break
    d, Q = np.linalg.eigh(X); d, Q = d[::-1], Q[:, ::-1]
    eps = posd_tol * abs(d[0])
    if d[-1] < eps:
        d = np.where(d < eps, eps, d)
        o = np.diag(X).copy()
        X = (Q * d) @ Q.T
        Dd = np.sqrt(np.maximum(eps, o) / np.diag(X))
        X = Dd[:, None] * X * Dd[None, :]
    return X


def _subset_rates(X: np.ndarray, Ci: np.ndarray, groups: List[np.ndarray], denom: Optional[int] = None) -> np.ndarray:
    """per group of columns: sum over the columns of the GLS-centred quadratic form x' C^-1 x / (N p_group), Adams'
    (2014) sigma^2_mult (denom: divide every group by this p instead, as geomorph's simulations do)."""
    N = X.shape[0]
    w = Ci.sum(1)
    Xc = X - (w @ X) / w.sum()
    q = ((Ci @ Xc) * Xc).sum(0)
    return np.array([q[g].sum() / N / (denom or len(g)) for g in groups])


def multi_evol_rates(Ys: List[np.ndarray], C: np.ndarray, sims: int, rng, geomorph_null: bool = False) -> Dict:
    """Adams (2014, Syst. Biol. 63: 166) comparison of net rates of multivariate Brownian-motion evolution between
    trait sets measured on the same species, as geomorph::compare.multi.evol.rates(Subset = TRUE):
      sigma^2_mult of a set = trace((X - 1a)' C^-1 (X - 1a)) / (N p), a the GLS mean (ML, divided by N), the mean
      per-variable rate (for colour cells: CIELAB units^2 per unit of branch length, per cell channel);
      statistic = largest / smallest rate, and every pairwise ratio;
      null = `sims` data sets simulated under multivariate BM with the observed rate matrix (X'C^-1X / N) whose
      diagonal is set to the rate of all the variables together (equal rates, covariances kept), made positive
      definite (make_pd); P = share of the observed and simulated ratios >= the observed (as geomorph's pval).
    geomorph (4.0.7 to 4.1.1) divides each set's sum by the total p, not the set's p, in the simulations only
    (fast.sigma.d.multi); with sets of different sizes that shifts its null ratios by the ratio of the sizes.
    Descriptron divides by the set's p (geomorph_null = True reproduces geomorph's). Returns rates, ratio, P,
    pairwise ratios and P, the global rate."""
    X = np.hstack(Ys); N, p = X.shape
    sizes = [Y.shape[1] for Y in Ys]
    groups = np.split(np.arange(p), np.cumsum(sizes)[:-1])
    Ci = np.linalg.inv(C)
    rates = _subset_rates(X, Ci, groups)
    glob = float(_subset_rates(X, Ci, [np.arange(p)])[0])
    pairs = [(i, j) for i in range(len(Ys)) for j in range(i + 1, len(Ys))]
    pr = lambda r: np.array([max(r[i], r[j]) / min(r[i], r[j]) for i, j in pairs])
    obs_ratio, obs_pair = float(rates.max() / rates.min()), pr(rates)
    w = Ci.sum(1); Xc = X - (w @ X) / w.sum()
    R = Xc.T @ Ci @ Xc / N
    np.fill_diagonal(R, glob)
    R0 = make_pd(R)
    ev, V = np.linalg.eigh(R0); ev, V = ev[::-1], V[:, ::-1]
    k = ev >= -1e-6 * abs(ev[0])
    Vs = V[:, k] * np.sqrt(np.maximum(ev[k], 0))
    Lc = np.linalg.cholesky(C)
    ge, gep = 1, np.ones(len(pairs))
    for _ in range(sims):
        Xs = Lc @ rng.standard_normal((N, Vs.shape[1])) @ Vs.T
        r = _subset_rates(Xs, Ci, groups, p if geomorph_null else None)
        pp = pr(r)
        ge += pp.max() >= obs_ratio - 1e-12 * obs_ratio
        gep += pp >= obs_pair - 1e-12 * obs_pair
    return {"rates": rates, "global_rate": glob, "ratio": obs_ratio, "P": ge / (sims + 1), "pairs": pairs,
            "pair_ratio": obs_pair, "pair_P": gep / (sims + 1), "sizes": sizes, "R0": R0}


def _part_scale(g: Dict) -> Optional[float]:
    """pixels per consensus unit of a part: sqrt(median quad area of a grid cell in the photographs / its area in
    consensus units), from color_homology_cell_pixels_<category>.csv beside the geometry JSON (None if missing)."""
    pth = g.get("_path")
    if not pth:
        return None
    cp = Path(pth).with_name(Path(pth).name.replace("color_homology_cells_", "color_homology_cell_pixels_")
                             .replace(".json", ".csv"))
    if not cp.exists():
        return None
    q = [float(r["quad_area_px"]) for r in csv.DictReader(open(cp)) if r.get("quad_area_px")]
    dx = float(np.median(np.diff(g["x_lines"]))); dy = float(np.median(np.diff(g["y_lines"])))
    return float(np.sqrt(np.median(q) / (dx * dy))) if q and dx > 0 and dy > 0 else None


def _poly_centroid_area(P: np.ndarray) -> Tuple[np.ndarray, float]:
    x, y = P[:, 0], P[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = cr.sum() / 2
    if abs(A) < 1e-12:
        return P.mean(0), 0.0
    return np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * A), abs(A)


def coco_part_offsets(coco_path: str, categories: List[str]) -> Optional[Dict[str, np.ndarray]]:
    """median position of each category's mask centroid relative to the centroid of all of them together, in pixels
    of a median-sized specimen (offsets divided by sqrt(total area) per image, times the median sqrt(total area)),
    over the images that have every category (polygon segmentations; RLE needs pycocotools)."""
    J = json.load(open(coco_path))
    cid = {c["name"]: c["id"] for c in J["categories"]}
    if any(c not in cid for c in categories):
        return None
    per: Dict[int, Dict[str, List]] = {}
    for an in J["annotations"]:
        nm = next((c for c in categories if cid[c] == an["category_id"]), None)
        if nm is None:
            continue
        seg = an["segmentation"]
        if isinstance(seg, list):
            for poly in seg:
                P = np.asarray(poly, float).reshape(-1, 2)
                if len(P) >= 3:
                    c, A = _poly_centroid_area(P); per.setdefault(an["image_id"], {}).setdefault(nm, []).append((c, A))
        else:
            from pycocotools import mask as mu
            rle = mu.frPyObjects(seg, *seg["size"]) if isinstance(seg.get("counts"), list) else seg
            m = mu.decode(rle).astype(bool)
            yy, xx = np.nonzero(m)
            if len(xx):
                per.setdefault(an["image_id"], {}).setdefault(nm, []).append((np.array([xx.mean(), yy.mean()]), float(len(xx))))
    offs, sizes, areas = {c: [] for c in categories}, [], {c: [] for c in categories}
    for d in per.values():
        if not all(c in d for c in categories):
            continue
        cen = {c: sum(cc * A for cc, A in d[c]) / sum(A for _, A in d[c]) for c in categories}
        are = {c: sum(A for _, A in d[c]) for c in categories}
        tot = sum(are.values()); g = sum(cen[c] * are[c] for c in categories) / tot
        sizes.append(np.sqrt(tot))
        for c in categories:
            offs[c].append((cen[c] - g) / np.sqrt(tot)); areas[c].append(are[c] / tot)
    if not sizes:
        return None
    S = float(np.median(sizes))
    return {c: np.median(np.array(offs[c]), 0) * S for c in categories} | \
        {f"_area_{c}": float(np.median(areas[c])) * S * S for c in categories}


def _part_layout(geoms: List[Dict], layout: str = "stack_y", offsets: Optional[List[np.ndarray]] = None,
                 rotate: int = 0, areas: Optional[List[float]] = None) -> List[np.ndarray]:
    """each part's consensus outline (y down, as the images) scaled to its median size in the photographs
    (_part_scale; equal widths if unknown) and placed: stack_y / stack_x = one after the other along the image's y / x
    axis in the order given (an elongate animal: head to tail), centred on the other axis; coco = each part's area
    centroid at its median offset from a COCO file (offsets, coco_part_offsets: any arrangement, e.g. petals around
    a flower). rotate = degrees anticlockwise for the drawing."""
    out = []
    pos = 0.0
    for i, g in enumerate(geoms):
        ol = np.asarray(g["outline"], float)
        sc = _part_scale(g) or 200.0 / np.ptp(ol[:, 0])
        if layout == "coco" and areas is not None and areas[i]:
            sc = float(np.sqrt(areas[i] / max(_poly_centroid_area(ol)[1], 1e-12)))   # the part's median area there
        P = ol * sc
        if layout == "coco" and offsets is not None:
            c, _ = _poly_centroid_area(P)
            P = P - c + offsets[i]
        elif layout == "stack_x":
            P = P - [P[:, 0].min(), 0.5 * (P[:, 1].min() + P[:, 1].max())] + [pos, 0]
            pos = P[:, 0].max()
        else:
            P = P - [0.5 * (P[:, 0].min() + P[:, 0].max()), P[:, 1].min()] + [0, pos]
            pos = P[:, 1].max()
        out.append(P)
    if rotate:
        t = np.radians(rotate)                                   # anticlockwise on screen with y down
        Rm = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        out = [P @ Rm.T for P in out]
    return out


def plot_body_map(geoms: List[Dict], names: List[str], panels: List[Dict], path_stem, title: str, note: str,
                  layout: str = "stack_y", offsets: Optional[List[np.ndarray]] = None, rotate: int = 0,
                  areas: Optional[List[float]] = None):
    """the organism assembled from the parts' consensus outlines (_part_layout: whatever parts are given), one copy
    per panel, each part filled by its value: panels [{'v': values per part, 'label', 'fmt', 'cmap'}]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.patches import Polygon
    import textwrap
    lay = _part_layout(geoms, layout, offsets, rotate, areas)
    allp = np.vstack(lay)
    W, Hh = np.ptp(allp[:, 0]), np.ptp(allp[:, 1])
    ph = 5.2; pw = ph * W / Hh
    cw = max(pw + 0.4, 2.2)
    k = len(panels)
    fig_w = k * cw + 0.3
    wrapped = textwrap.wrap(note, int(fig_w * 16))
    head = 0.55 + 0.13 * len(wrapped)
    fig_h = ph + head + 1.25
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, Pn in enumerate(panels):
        x0 = 0.2 + i * cw + (cw - pw) / 2
        ax = fig.add_axes([x0 / fig_w, 0.75 / fig_h, pw / fig_w, ph / fig_h])
        v = np.asarray(Pn["v"], float)
        lo = Pn.get("vmin", float(np.nanmin(v))); hi = Pn.get("vmax", float(np.nanmax(v)))
        norm = mpl.colors.Normalize(lo, hi if hi > lo else lo + 1e-9)
        cm = _trunc_cmap(Pn.get("cmap", "viridis"), 0.0, 0.95)
        for j, P in enumerate(lay):
            c = cm(norm(v[j])) if np.isfinite(v[j]) else (0.85, 0.85, 0.85, 1)
            ax.add_patch(Polygon(P, closed=True, fc=c, ec="#222", lw=0.8))
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            (cx, cy), _ = _poly_centroid_area(P)
            ax.text(cx, cy - 0.025 * Hh, names[j].replace("_", " "), fontsize=6.5, va="center", ha="center",
                    color="white" if lum < 0.55 else "black")
            ax.text(cx, cy + 0.02 * Hh, Pn["fmt"].format(v[j]) if np.isfinite(v[j]) else "-", fontsize=8,
                    ha="center", va="center", color="white" if lum < 0.55 else "black", fontweight="bold")
            if Pn.get("sub"):
                ax.text(cx, cy + 0.05 * Hh, Pn["sub"][j], fontsize=6, ha="center", va="center",
                        color="white" if lum < 0.55 else "black")
        ax.set_xlim(allp[:, 0].min() - 0.02 * W, allp[:, 0].max() + 0.02 * W)
        ax.set_ylim(allp[:, 1].max() + 0.01 * Hh, allp[:, 1].min() - 0.01 * Hh)
        ax.set_aspect("equal"); ax.axis("off")
        fig.text((x0 + pw / 2) / fig_w, (0.75 + ph + 0.1) / fig_h, "\n".join(textwrap.wrap(Pn["label"], int(cw * 11))),
                 fontsize=7.5, ha="center", va="bottom")
        cax = fig.add_axes([(0.2 + i * cw + 0.25) / fig_w, 0.42 / fig_h, (cw - 0.5) / fig_w, 0.09 / fig_h])
        mpl.colorbar.ColorbarBase(cax, cmap=cm, norm=norm, orientation="horizontal")
        cax.tick_params(labelsize=6); cax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    fig.text(0.2 / fig_w, 1 - 0.15 / fig_h, title, fontsize=10, va="top")
    fig.text(0.2 / fig_w, 1 - 0.42 / fig_h, "\n".join(wrapped), fontsize=6.8, va="top", color="#444")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=220)
    plt.close(fig)


def part_rates_section(sets: Dict[str, Dict], a, out: Path, rng) -> List[str]:
    """--part_sets: compare the rate of colour-pattern change between body parts (trait sets run through
    ancestral_pattern on the same species): per part the whole-tree rate of pattern change (sum over branches of the
    mean CIEDE2000 between parent and child / tree length) and the mean and median branch rate; Kmult of the part's
    L*a*b* cells (complete cells, raw CIELAB) with permutation P; Adams' sigma^2_mult per part and the test of equal
    rates (multi_evol_rates); the Spearman correlation between parts of each branch's change / sqrt(branch length)
    (do the same lineages change more than Brownian motion expects in every part?). Writes part_rates.csv, part_rate_pairs.csv, part_branch_rate_correlation.csv and the body map
    (body_map_part_rates.png/.pdf)."""
    from scipy.stats import spearmanr
    names = [n for n in a.part_sets if n in sets and "anc_summary" in sets[n]]
    rep = ["## Rates of colour-pattern change between body parts (" + ", ".join(names) + ")"]
    miss = [n for n in a.part_sets if n not in names]
    if miss:
        rep.append(f"  not analysed (no ancestral_pattern result): {', '.join(miss)}")
    if len(names) < 2:
        return rep + ["  fewer than two parts - skipped", ""]
    tipsets = [frozenset(t.name for t in sets[n]["anc_summary"]["tips"]) for n in names]
    common = frozenset.intersection(*tipsets)
    if any(ts != common for ts in tipsets):
        rep.append(f"  parts differ in species: the test uses the {len(common)} they share")
    ref = sets[names[0]]
    tree = prune(_copy_tree(ref["tree"]), set(common)); tips = tips_of(tree)
    C = vcv(tips, tips)
    Ys, rows = [], []
    for n in names:
        A = sets[n]["anc_summary"]
        order = [[t.name for t in A["tips"]].index(t.name) for t in tips]
        lab = A["lab"][order]
        comp = np.isfinite(lab[..., 0]).all(0)
        Y = lab[:, comp].reshape(len(tips), -1)
        Ys.append(Y)
        sig = phylo_signal(Y, C, a.iterations, rng)
        rows.append({"part": n, "n_species": len(tips), "n_cells": len(A["cells"]), "n_cells_complete": int(comp.sum()),
                     "dE00_per_Myr_tree": A["tree_rate"], "dE00_per_Myr_branch_mean": A["mean_rate"],
                     "dE00_per_Myr_branch_median": A["median_rate"], "Kmult_Lab": sig["Kmult"], "P_Kmult": sig["P"],
                     "Z_Kmult": sig["Z"]})
    T = multi_evol_rates(Ys, C, a.part_test_sims, rng)
    Tg = multi_evol_rates(Ys, C, a.part_test_sims, rng, geomorph_null=True)
    for r, v in zip(rows, T["rates"]):
        r["sigma2_mult_Lab"] = float(v); r["sigma2_mult_over_global"] = float(v / T["global_rate"])
    _write_csv(out / "part_rates.csv", rows)
    prow = [{"part_1": names[i], "part_2": names[j], "rate_ratio": float(T["pair_ratio"][k]),
             "P": float(T["pair_P"][k]), "P_geomorph_null": float(Tg["pair_P"][k])} for k, (i, j) in enumerate(T["pairs"])]
    _write_csv(out / "part_rate_pairs.csv", prow)
    # branch rates between parts
    # Delta E00 / sqrt(branch length), not / length: rates sharing the 1 / length factor correlate whatever happens
    # (short branches are 'fast' in every part); under BM the change grows with sqrt(time), so this has the same
    # expectation on every branch and a correlation means the same lineages change more than expected in both parts
    bk = {n: {branch_key(b["child"]): (b["mean"] / np.sqrt(b["length"]) if b["length"] > 0 else np.nan)
              for b in sets[n]["anc_summary"]["branches"]} for n in names}
    keys = [k for k in bk[names[0]] if all(k in bk[n] for n in names)]
    M = np.array([[bk[n][k] for k in keys] for n in names])
    crow = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ok = np.isfinite(M[i]) & np.isfinite(M[j])
            rho, pv = spearmanr(M[i, ok], M[j, ok])
            crow.append({"part_1": names[i], "part_2": names[j], "n_branches": int(ok.sum()), "spearman_rho": rho,
                         "P_uncorrected": pv})
    _write_csv(out / "part_branch_rate_correlation.csv", crow)
    order = np.argsort(-T["rates"])
    rep += [f"  {len(tips)} species; complete cells per part: " + ", ".join(f"{r['part']} {r['n_cells_complete']}"
                                                                          for r in rows),
            "  part: Delta E00 per Myr (whole tree; mean / median branch) | Kmult (P) | sigma^2_mult (CIELAB^2 per "
            "Myr per cell channel)"]
    rep += [f"    {r['part']}: {r['dE00_per_Myr_tree']:.3g} ({r['dE00_per_Myr_branch_mean']:.3g} / "
            f"{r['dE00_per_Myr_branch_median']:.3g}) | {r['Kmult_Lab']:.3f} ({r['P_Kmult']:.3f}) | "
            f"{r['sigma2_mult_Lab']:.4g}" for r in rows]
    rep += [f"  Adams (2014) test of equal rates (geomorph::compare.multi.evol.rates): fastest {names[order[0]]}, slowest "
            f"{names[order[-1]]}, ratio {T['ratio']:.2f}, P = {T['P']:.4f} ({a.part_test_sims} simulations; with "
            f"geomorph's null, which divides by the total number of variables, P = {Tg['P']:.4f})",
            "  pairwise (ratio, P): " + "; ".join(f"{r['part_1']}-{r['part_2']} {r['rate_ratio']:.2f} ({r['P']:.3f})"
                                                for r in prow),
            "  Spearman correlation between parts of the branch changes scaled for time (Delta E00 / sqrt(Myr)): "
            + "; ".join(f"{r['part_1']}-{r['part_2']} {r['spearman_rho']:.2f}" for r in crow)]
    geoms = [sets[n]["anc_summary"]["geometry"] for n in names]
    offs, areas_ = None, None
    if a.body_map_layout == "coco":
        cmap_ = dict(x.split("=", 1) for x in a.part_categories)
        cats = [cmap_.get(n, n) for n in names]
        d = coco_part_offsets(a.body_map_coco, cats) if a.body_map_coco else None
        if d is None:
            rep.append("  body map: --body_map_layout coco needs --body_map_coco with every part's category "
                       "(--part_categories SET=CATEGORY) - stacked along y instead")
        else:
            offs = [d[c] for c in cats]; areas_ = [d[f"_area_{c}"] for c in cats]
    if all(g is not None for g in geoms):
        plot_body_map(geoms, names, [
            {"v": [r["dE00_per_Myr_tree"] for r in rows], "label": "ΔE00 per Myr (whole tree: sum of branch "
             "changes / tree length)", "fmt": "{:.3f}", "cmap": "viridis", "vmin": 0.0},
            {"v": [r["sigma2_mult_Lab"] for r in rows], "label": "BM rate σ²mult (CIELAB² per Myr per "
             "cell channel)", "fmt": "{:.2f}", "cmap": "viridis", "vmin": 0.0},
            {"v": [r["Kmult_Lab"] for r in rows], "label": "phylogenetic signal Kmult (L*a*b* of the cells)",
             "fmt": "{:.2f}", "cmap": "cividis", "vmin": 0.0, "vmax": max(1.0, max(r["Kmult_Lab"] for r in rows)),
             "sub": [f"P {r['P_Kmult']:.3f}" for r in rows]}],
            out / "body_map_part_rates", "Rate of colour-pattern change and phylogenetic signal per part",
            f"{len(tips)} species; each part's consensus outline at its median size"
            + (" and median position in the COCO masks" if offs is not None else "") + "; colour scales start at 0. ΔE00 per "
            f"Myr: mean CIEDE2000 over the part's grid cells between the Brownian-motion reconstruction at each node and "
            f"its child (a species: its mean), summed over branches and divided by the tree length. σ²mult: "
            f"Adams (2014) mean per-variable BM rate on the cells all species have; test of equal rates P = "
            f"{T['P']:.3f} (largest / smallest {T['ratio']:.2f}, {a.part_test_sims} simulations). Kmult: Adams (2014), "
            f"{a.iterations} permutations",
            "coco" if offs is not None else (a.body_map_layout if a.body_map_layout != "coco" else "stack_y"), offs,
            a.body_map_rotate, areas_)
        rep.append("  Files: part_rates.csv, part_rate_pairs.csv, part_branch_rate_correlation.csv, "
                   "body_map_part_rates.png/.pdf")
    return rep + [""]


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, help="Newick tree whose tips are species names")
    ap.add_argument("--traits", action="append", default=[], metavar="NAME=CSV",
                    help="a trait table; repeat for several sets (shape, measurements, colour, texture, ...)")
    ap.add_argument("--species_col", default=None, help="column holding the species (default: use --species_regex)")
    ap.add_argument("--species_regex", default=None, help="regex with one group extracting the species from the id column")
    ap.add_argument("--groups", default=None, help="group labels CSV (filename + group_label/species) giving each "
                    "specimen's species, as for the rest of the pipeline (instead of --species_col/--species_regex)")
    ap.add_argument("--id_col", default=None, help="id column for --species_regex (default: filename/specimen_id/...)")
    ap.add_argument("--columns", action="append", default=[], metavar="NAME=REGEX",
                    help="keep only the columns of trait set NAME that match REGEX")
    ap.add_argument("--scale", action="append", default=[], metavar="NAME=none|standardize|log")
    ap.add_argument("--size_col", default=r"centroid|csize",
                    help="regex for a size column kept out of the trait set and used as the logCS predictor")
    ap.add_argument("--pgls", action="append", default=[], metavar="SET~PREDICTOR",
                    help="PREDICTOR is logCS (from a set's centroid-size column) or SET2:COLUMN")
    ap.add_argument("--analyses", nargs="+", default=["signal", "morphospace"],
                    choices=["signal", "pgls", "morphospace", "models", "rates", "colour_tree", "cophylo", "heatmap",
                             "convergence", "surface", "reference_scores", "mk_maps", "oum", "ancestral_pattern",
                             "discrete_tree"])
    ap.add_argument("--tip_regex", default=None,
                    help='regex with one group extracting the species from each tree tip (e.g. "_([A-Z][a-z]+ .+)$" '
                         'for tips such as "QH001-S1_Abemus chloropterus")')
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS),
                    help="models compared by 'models' (default all)")
    ap.add_argument("--model_bounds", action="append", default=[], metavar="PARAM=LO,HI",
                    help="bounds for alpha (OU), a (EB) or lambda; default geiger::fitContinuous's")
    ap.add_argument("--model_columns", action="append", default=[], metavar="NAME=REGEX",
                    help="variables of trait set NAME analysed by 'models' and 'rates' (default all its variables)")
    ap.add_argument("--rate_lambda", type=float, default=1.0,
                    help="smoothing of the variable rates (phytools::multirateBM lambda; larger = smoother)")
    ap.add_argument("--rate_top", type=int, default=10, help="fastest branches listed in the report per variable")
    ap.add_argument("--model_pcs", default=None, metavar="K|auto",
                    help="run 'models' and 'rates' on the first K principal components of the set's standardised "
                         "features (the colour pattern, not mean colour) instead of on its original variables; 'auto' = "
                         "the PCs reaching 90%% of the variance, at most --conv_pcs_max. PCA of species data can bias "
                         "model choice (Uyeda et al. 2015): results are reported per PC")
    ap.add_argument("--surface_pcs", default="auto", metavar="K|auto",
                    help="'surface': PCs of the standardised features analysed (default auto: the PCs above the "
                         "broken-stick expectation, kept within 2..--surface_pcs_max)")
    ap.add_argument("--surface_pcs_max", type=int, default=4,
                    help="largest automatic number of SURFACE PCs (each PC adds 2 + regimes parameters)")
    ap.add_argument("--surface_aic_threshold", type=float, default=0.0,
                    help="SURFACE: a shift or a merge is kept when it changes AICc by less than this (surface's "
                         "aic_threshold; 0 = any improvement)")
    ap.add_argument("--surface_exclude", type=float, default=0.0,
                    help="SURFACE: share of the worst candidate shift positions of the previous step not tried again "
                         "(surface's exclude)")
    g = ap.add_argument_group("colour-pattern trees and convergence (colour_tree, cophylo, heatmap, convergence)")
    g.add_argument("--colour_cells", default=COLOUR_CELLS_DEFAULT,
                   help="regex picking the colour columns, with named groups 'cell' (the homologous grid cell) and 'ch' "
                        "(L, a or b); default: the colour-homology table's r<i>c<j>_L/a/b_mean_abs columns")
    g.add_argument("--colour_distance", nargs="+", choices=["de2000", "de76", "palette_emd", "pca"], default=["de2000"],
                   help="one or more species colour distances (several: files named <set>__<distance>_...): de2000 / "
                        "de76 = mean per-cell CIEDE2000 / CIE76 between the species-mean L*a*b* of the homologous grid "
                        "cells; palette_emd = mean earth mover's distance between the specimens' recolorize palettes "
                        "(--palettes; convergence on a PCoA of it); pca = Euclidean distance on the first --pca_pcs PCs "
                        "of the standardised colour-pattern features (the phylomorphospace's)")
    g.add_argument("--palettes", action="append", default=[], metavar="NAME=GLOB_OR_CSV",
                   help="recolorize palettes of trait set NAME, for --colour_distance palette_emd: a file pattern (e.g. "
                        "'color_extraction/soma/*_palette.csv'; each file is matched to the photo whose file-name stem it "
                        "contains, the species from --groups) or a CSV with columns filename and palette [, species]")
    g.add_argument("--pca_pcs", default="auto",
                   help="PCs for --colour_distance pca: a number or 'auto' (reaching 90%% of the variance, at most "
                        "--conv_pcs_max)")
    g.add_argument("--colour_tree_method", choices=["upgma", "nj"], default="upgma",
                   help="clustering of the colour distances (UPGMA, rooted; or neighbour joining, unrooted)")
    g.add_argument("--bootstrap", type=int, default=1000,
                   help="replicates resampling the grid cells with replacement, for clade support of the colour tree")
    g.add_argument("--convergent_groups", default=None,
                   help="CSV with columns species,group: candidate convergent groups to test (default: found "
                        "automatically, colour-tree clades with support >= --min_support that are not clades of the tree)")
    g.add_argument("--min_support", type=float, default=0.7, help="bootstrap support for automatic candidate groups")
    g.add_argument("--conv_max_size", type=int, default=0,
                   help="largest automatic candidate group (default: half the species)")
    g.add_argument("--conv_pcs", default="auto",
                   help="traits for the convergence tests: 'auto' (the first colour PCs reaching 90%% of the variance, "
                        "at most --conv_pcs_max), a number of PCs, or 'all' (every colour variable, standardised)")
    g.add_argument("--conv_pcs_max", type=int, default=10)
    g.add_argument("--conv_alpha", type=float, default=0.05,
                   help="--tangle_highlight convergent: P threshold for C1 and Ct1")
    g.add_argument("--tangle_highlight", choices=["convergent", "largest", "none"], default="convergent",
                   help="tanglegram lines coloured by group: convergent = candidate groups passing C1 and Ct1 (needs "
                        "--analyses convergence, else largest); largest = the largest supported colour clades that are "
                        "not input-tree clades")
    g.add_argument("--tip_images", default=None,
                   help="CSV with columns species, image [, trait_set]: one picture per species drawn beside its name "
                        "on both sides of the tanglegram (e.g. the recolorized structure); rows with a trait_set apply "
                        "to that set only. Default: the species-mean colour of each grid cell")
    g.add_argument("--tip_image_rotate", type=int, choices=[0, 90, 180, 270], default=0,
                   help="rotate the --tip_images counter-clockwise (90 lays a head-up beetle on its side, head left)")
    g.add_argument("--tip_images_note", default="one specimen per species",
                   help="what the --tip_images show, printed in the tanglegram key")
    g.add_argument("--tangle_tree_colour", choices=["model", "rates", "regimes", "none"], default="model",
                   help="colour of the tanglegram's molecular-tree branches: model = the SURFACE regimes when SURFACE "
                        "has the lowest AICc on the colour-pattern PCs (needs --analyses surface), otherwise the "
                        "variable-rate BM rate; rates; regimes; none")
    g.add_argument("--tangle_links", choices=["curved", "straight"], default="curved",
                   help="tanglegram links: S-shaped Bezier curves or straight segments")
    g.add_argument("--conv_sims", type=int, default=1000,
                   help="Brownian-motion simulations for C1-C4 / Ct1-Ct4 and resamplings for the Wheatsheaf index")
    g = ap.add_argument_group("scoring colour distances against reference groups (reference_scores)")
    g.add_argument("--reference_groups", default=None,
                   help="CSV with a species column and one or more grouping columns (e.g. look-alike groups and "
                        "mimicry types): every --colour_distance of every set is scored against each --reference_columns "
                        "grouping (implies --analyses reference_scores)")
    g.add_argument("--reference_columns", default="group",
                   help="comma-separated grouping columns of --reference_groups (e.g. group,mimicry_type)")
    g.add_argument("--combine_sets", action="append", default=[], metavar="NEW=SET1:W1,SET2:W2",
                   help="a combination of trait sets scored like a set (reference_scores): the sets' species colour "
                        "distances, each divided by its median over the species they share, weighted, summed and "
                        "divided by the sum of the weights (e.g. body=soma:2,abdomen:1; one set alone is used "
                        "unscaled); repeat for several. Each scoring uses a new generator seeded with --seed")
    g.add_argument("--score_species", nargs="+", choices=["all", "tree"], default=["all", "tree"],
                   help="species scored: all = every species with colour data and a reference label, tree species or "
                        "not (the distances need no tree); tree = only those on the tree. The tree analyses always use "
                        "the tree species")
    g = ap.add_argument_group("discrete trait: Mk model, stochastic maps (mk_maps) and one OU optimum per state (oum)")
    g.add_argument("--discrete_trait", default=None, metavar="CSV:COLUMN",
                   help="a discrete trait per species (species column + COLUMN), e.g. the mimicry type")
    g.add_argument("--mk_models", nargs="+", choices=["ER", "SYM", "ARD"], default=["ER", "SYM", "ARD"],
                   help="Mk models compared by AICc (n = species); the best one is used for the maps")
    g.add_argument("--mk_root", choices=["equal", "fitzjohn"], default="equal",
                   help="root prior of the Mk models, ancestral states and maps (phytools::fitMk pi; make.simmap's "
                        "default is equal)")
    g.add_argument("--discrete_label_column", default=None,
                   help="discrete_tree: a column of the --discrete_trait CSV printed in brackets after each species "
                        "(e.g. the look-alike group)")
    g.add_argument("--discrete_scale_bar", type=float, default=None,
                   help="discrete_tree: length of the scale bar in tree units (default: a round third of the height)")
    g.add_argument("--simmaps", type=int, default=200, help="stochastic character maps (make.simmap nsim)")
    g.add_argument("--oum_maps", type=int, default=200,
                   help="'oum': stochastic maps whose paintings OUM and BMS are also fitted on (at most --simmaps)")
    g = ap.add_argument_group("ancestral colour pattern drawn as pictures (ancestral_pattern)")
    g.add_argument("--cell_geometry", action="append", default=[], metavar="NAME=JSON",
                   help="color_homology_cells_<category>.json from color_phenomics_homology_v2_1.py --save_cell_geometry "
                        "for trait set NAME: the grid cells in consensus space (to draw patterns on the outline) and the "
                        "cells each specimen left empty (set missing instead of the table's column means; the other "
                        "analyses then use only the cells no species lacks)")
    g.add_argument("--anc_missing", choices=["observed", "complete"], default="observed",
                   help="cells some species lack: reconstruct each cell on the species that have it (observed, default; "
                        "cells in fewer than --anc_min_share of the species are dropped) or use only the cells every "
                        "species has (complete)")
    g.add_argument("--anc_min_share", type=float, default=0.5,
                   help="with --anc_missing observed: least share of the species that must have a cell (default 0.5)")
    g.add_argument("--anc_ci", type=float, default=0.95, help="interval of the ancestral states (default 0.95: "
                   "estimate +- 1.96 SE, as phytools::fastAnc)")
    g.add_argument("--pattern_width", type=int, default=120, help="pixels across the consensus outline in the pictures")
    g.add_argument("--pattern_rotate", type=int, choices=[0, 90, 180, 270], default=90,
                   help="rotation of the pictures on the tree, degrees anticlockwise (90: the top of the consensus to "
                        "the left; the convergence panel and node gallery are upright)")
    g.add_argument("--anc_branch_colour", nargs="+", choices=["rate_de00", "bm_rate", "none"], default=["rate_de00"],
                   help="branches of the ancestral-pattern tree coloured by: rate_de00 = the rate of colour-pattern "
                        "change, mean CIEDE2000 over the grid cells between the reconstructed parent pattern and the "
                        "child's (a species: its mean) divided by the branch length (Delta E00 per Myr on a "
                        "chronogram; log scale, viridis); bm_rate = the variable-rate Brownian-motion rate "
                        "(multirateBM, --rate_lambda) on the first --model_pcs (default auto) pattern PCs; none. "
                        "Several: the first is <set>_ancestral_pattern_tree, the others "
                        "<set>_ancestral_pattern_tree_<mode>. The branch table <set>_ancestral_pattern_branches.csv "
                        "is always written")
    g.add_argument("--anc_top_cells", type=int, default=5,
                   help="cells with the largest CIEDE2000 listed per branch in the branch table")
    g.add_argument("--part_sets", nargs="+", default=[], metavar="SET",
                   help="trait sets that are parts of the same specimens (any names: head pronotum elytra abdomen; "
                        "forewing hindwing; petal sepal), each run through ancestral_pattern: their rates of "
                        "colour-pattern change compared (Delta E00 per Myr, Kmult, Adams' sigma^2_mult with the test "
                        "of equal rates, as geomorph::compare.multi.evol.rates) and drawn on the organism assembled "
                        "from their consensus outlines (body_map_part_rates; --body_map_layout)")
    g.add_argument("--body_map_layout", choices=["stack_y", "stack_x", "coco"], default="stack_y",
                   help="body map: parts placed one after the other along the image y (stack_y, e.g. head to tail) "
                        "or x axis, in the --part_sets order, or each at its median position in --body_map_coco "
                        "(coco: any arrangement)")
    g.add_argument("--body_map_coco", default=None,
                   help="COCO file with the parts as categories, for --body_map_layout coco")
    g.add_argument("--part_categories", nargs="+", default=[], metavar="SET=CATEGORY",
                   help="COCO category of a part set when its name differs (body map coco layout)")
    g.add_argument("--body_map_rotate", type=int, choices=[0, 90, 180, 270], default=0,
                   help="rotation of the body map drawing, degrees anticlockwise")
    g.add_argument("--part_test_sims", type=int, default=999,
                   help="simulations for the test of equal rates between --part_sets")
    g.add_argument("--state_codes", nargs="+", default=[], metavar="STATE=CODE:#COLOUR",
                   help="code and colour of each --discrete_trait state at the tips, e.g. non_mimic=n:#8a8a8a")
    ap.add_argument("--iterations", type=int, default=999, help="permutations (Kmult, PGLS, Mantel, reference scores)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    if a.reference_groups and "reference_scores" not in a.analyses:
        a.analyses = list(a.analyses) + ["reference_scores"]
    if "reference_scores" in a.analyses and not a.reference_groups:
        ap.error("--analyses reference_scores needs --reference_groups")
    if "oum" in a.analyses and "mk_maps" not in a.analyses:
        a.analyses = list(a.analyses) + ["mk_maps"]
    if a.part_sets and "ancestral_pattern" not in a.analyses:
        a.analyses = list(a.analyses) + ["ancestral_pattern"]
    if "ancestral_pattern" in a.analyses and a.discrete_trait and "mk_maps" not in a.analyses:
        a.analyses = list(a.analyses) + ["mk_maps"]
    if ("mk_maps" in a.analyses or "discrete_tree" in a.analyses) and not a.discrete_trait:
        ap.error("--analyses mk_maps / oum / discrete_tree need --discrete_trait CSV:COLUMN")
    if not a.traits and [x for x in a.analyses if x != "discrete_tree"] != []:
        ap.error("--traits is required (only --analyses discrete_tree runs without trait tables)")
    if a.combine_sets and "reference_scores" not in a.analyses:
        ap.error("--combine_sets is scored by reference_scores: give --reference_groups")
    a._ref_D = {} if a.combine_sets else None
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    root, tree_notes = load_tree(a.tree, a.tip_regex)
    colsel = dict(x.split("=", 1) for x in a.columns)
    modsel = dict(x.split("=", 1) for x in a.model_columns)
    mbounds = {k: tuple(float(v) for v in b.split(",")) for k, b in (x.split("=", 1) for x in a.model_bounds)}
    gmap = load_group_map(a.groups) if a.groups else None
    scales = dict(x.split("=", 1) for x in a.scale)
    geoms = dict(x.split("=", 1) for x in a.cell_geometry)

    sets = {}
    for spec in a.traits:
        name, path = spec.split("=", 1)
        geom = load_cell_geometry(geoms[name]) if name in geoms else None
        sp, full_cols, full_M, counts = load_traits(path, a.species_col, a.species_regex, a.id_col, gmap,
                                                    keep_missing=True, blank=geom["blank"] if geom else None)
        complete = ~np.isnan(full_M).any(axis=0)          # the other analyses use the columns no species lacks
        cols, M = [c for c, k in zip(full_cols, complete) if k], full_M[:, complete]
        sp = [_norm_name(s) for s in sp]
        counts = {_norm_name(k): v for k, v in counts.items()}
        if name in colsel:
            rx = re.compile(colsel[name]); k = [i for i, c in enumerate(cols) if rx.search(c)]
            cols, M = [cols[i] for i in k], M[:, k]
        # a size column (centroid size) is kept aside: it is a predictor (logCS), not part of the trait set
        szi = [i for i, c in enumerate(cols) if re.search(a.size_col, c, re.I)]
        size = M[:, szi[0]] if szi else None
        keep = [i for i in range(len(cols)) if i not in szi]
        sets[name] = {"species": sp, "cols": [cols[i] for i in keep], "raw": M[:, keep], "size": size,
                      "counts": counts, "full_cols": full_cols, "full_raw": full_M, "geometry": geom}

    report = ["# Phylogenetic comparative analyses", "", f"Tree: {a.tree}", ""]
    if tree_notes:
        report += [f"- {x}" for x in tree_notes[:20]] + ([f"- ... {len(tree_notes) - 20} more"] if len(tree_notes) > 20 else []) + [""]
    signal_rows, pgls_rows, model_rows, rate_summary = [], [], [], []
    colour_req = [x for x in ("colour_tree", "cophylo", "heatmap", "convergence") if x in a.analyses]
    given_groups = load_convergent_groups(a.convergent_groups) if a.convergent_groups else None
    ref_rows, mk_cache = [], {}
    disc = discrete_states(a.discrete_trait) if a.discrete_trait else None
    for name, S in sets.items():
        tipnames = {t.name for t in tips_of(root)}
        common = [s for s in S["species"] if s in tipnames]
        if "reference_scores" in a.analyses:
            r_, rows_ = reference_scoring(name, S, a, out, gmap, tipnames, rng)
            report += r_ + [""]
            ref_rows += rows_
        if len(common) < 4:
            report.append(f"{name}: only {len(common)} species match the tree - skipped"); continue
        tree = prune(_copy_tree(root), set(common))
        tips = tips_of(tree)
        idx = [S["species"].index(t.name) for t in tips]
        how = scales.get(name, "none")
        Y = scale(S["raw"][idx], how)
        if Y.shape[1] == 0 or not np.isfinite(Y).all() or float(Y.var(axis=0).sum()) == 0:
            report += [f"## {name}", "no variable differs among the species (for example empty homology cells) - skipped", ""]
            continue
        C = vcv(tips, tips)
        S.update(tree=tree, tips=tips, Y=Y, C=C, idx=idx)
        dropped = sorted(set(S["species"]) - set(common))
        report += [f"## {name}", f"{len(tips)} species, {Y.shape[1]} variables (scale: {how})"
                   + (f"; not in the tree: {', '.join(dropped)}" if dropped else ""), ""]
        if "signal" in a.analyses:
            r = phylo_signal(Y, C, a.iterations, rng)
            signal_rows.append({"trait_set": name, "n_species": len(tips), "n_variables": Y.shape[1], "scale": how, **r})
            report.append(f"Phylogenetic signal: Kmult = {r['Kmult']:.4f}, P = {r['P']:.4f}, Z = {r['Z']:.2f}")
        if "morphospace" in a.analyses and Y.shape[1] < 2:
            report.append("Phylomorphospace: fewer than two variables - not drawn")
        elif "morphospace" in a.analyses:
            nodes = internal_nodes(tree)
            anc = ancestral_states(Y, tips, nodes)
            mu = Y.mean(0)
            U, sv, Vt = np.linalg.svd(Y - mu, full_matrices=False)
            ev = sv ** 2 / (sv ** 2).sum()
            pc_t, pc_n = (Y - mu) @ Vt.T, (anc - mu) @ Vt.T
            np.savetxt(out / f"{name}_morphospace_tips.csv", pc_t[:, :min(5, pc_t.shape[1])], delimiter=",",
                       header=",".join(f"PC{i + 1}" for i in range(min(5, pc_t.shape[1]))), comments="")
            plot_morphospace(pc_t, pc_n, tips, nodes, ev, out / f"{name}_phylomorphospace.png",
                             f"{name}: phylomorphospace ({len(tips)} species)")
            report.append(f"Phylomorphospace: {name}_phylomorphospace.png (PC1 {100 * ev[0]:.1f}%, PC2 {100 * ev[1]:.1f}%)")
        if "models" in a.analyses or "rates" in a.analyses:
            if a.model_pcs:
                sc_, ev_, p_ = pattern_pc_matrix(S)
                kk_ = n_axes(ev_, a.model_pcs, a.conv_pcs_max, len(tips))
                cols = [f"PC{i + 1}" for i in range(kk_)]
                var_of = {c: sc_[:, i] for i, c in enumerate(cols)}
                report.append(f"Models and rates on the first {kk_} principal components ({100 * ev_[:kk_].sum():.1f}% of "
                              f"the variance: " + ", ".join(f"PC{i + 1} {100 * ev_[i]:.1f}%" for i in range(kk_))
                              + f") of the {p_} standardised features, one PC at a time. PCA of species data can bias "
                              "model choice towards OU and early burst (Uyeda, Caetano & Pennell 2015, Syst. Biol. 64: "
                              "677): results are given per PC, not only for PC1.")
            else:
                cols = [c for c in S["cols"] if name not in modsel or re.search(modsel[name], c)]
                Yraw = np.log(S["raw"][idx]) if how == "log" else S["raw"][idx]   # models are scale-free otherwise
                var_of = {c: Yraw[:, S["cols"].index(c)] for c in cols}
            depth = np.diag(C)
            if depth.max() - depth.min() > 1e-6 * depth.max():
                report.append(f"The tree is not ultrametric (root-to-tip {depth.min():.4g}-{depth.max():.4g}): rates, "
                              "alpha and a are per unit of branch length, not per unit of time.")
        if "models" in a.analyses:
            best = {}
            for c in cols:
                y = var_of[c]
                if np.ptp(y) == 0:
                    continue
                fits = fit_models(y, C, a.models, mbounds)
                for f in fits:
                    model_rows.append({"trait_set": name, "variable": c, "n_species": len(tips), **f})
                b = min(fits, key=lambda f: f["AICc"])
                best[b["model"]] = best.get(b["model"], 0) + 1
            report.append(f"Evolutionary models ({len(cols)} variables, lowest AICc): "
                          + ", ".join(f"{m} {best.get(m, 0)}" for m in a.models) + f"  (see {name}_model_fits.csv)")
            if a.model_pcs:
                for c in cols:
                    fr = sorted([r for r in model_rows if r["trait_set"] == name and r["variable"] == c],
                                key=lambda r: r["AICc"])
                    if fr:
                        report.append(f"  {c}: " + ", ".join(f"{r['model']} {r['dAICc']:.2f} (w {r['weight']:.2f})"
                                                           for r in fr) + "  [model dAICc (Akaike weight)]")
            nb = sum(1 for r in model_rows if r["trait_set"] == name and r["at_bound"])
            if nb:
                report.append(f"  {nb} fits have a parameter at its bound (column at_bound): widen --model_bounds or "
                              "read those estimates as limits")
        if "rates" in a.analyses:
            rows = []
            pc_rates = []
            for c in cols:
                y = var_of[c]
                if np.ptp(y) == 0:
                    continue
                R = multirate_bm(tree, tips, y, a.rate_lambda)
                pc_rates.append(R)
                delta, z = branch_changes(tree, tips, y, R["nodes"], R["parent"], R["length"])
                med = float(np.nanmedian(R["edge_rate"][1:]))
                for i in range(1, len(R["nodes"])):
                    rows.append({"trait_set": name, "variable": c, "branch_to": clade_label(R["nodes"][i]),
                                 "is_tip": not R["nodes"][i].kids, "length": R["length"][i],
                                 "rate": R["edge_rate"][i], "rate_over_median": R["edge_rate"][i] / med,
                                 "change": delta[i], "change_over_sd": z[i]})
                tag = _safe(f"{name}__{c}")
                plot_rate_tree(R["nodes"], R["parent"], R["length"], R["edge_rate"], out / f"{tag}_rate_tree.png",
                               f"{name}: {c} - variable-rate Brownian motion (lambda = {a.rate_lambda:g})")
                order = np.argsort(-R["edge_rate"][1:])[:a.rate_top] + 1
                rate_summary.append({"trait_set": name, "variable": c, "penalised_logLik": R["penalised_logLik"],
                                     "logLik": R["logLik"], "k": R["k"], "lambda": R["lambda"],
                                     "converged": R["converged"],
                                     "fastest": "; ".join(f"{clade_label(R['nodes'][i])} x{R['edge_rate'][i] / med:.1f}"
                                                          for i in order)})
                if not R["converged"]:
                    report.append(f"  rates {c}: optimiser did not converge ({R['message']})")
            if a.model_pcs and pc_rates:
                mr = np.vstack([R["edge_rate"] for R in pc_rates]).mean(axis=0)
                rl = np.log10(mr / np.nanmedian(mr[1:]))
                S["chrono"] = {"rates": {branch_key(pc_rates[0]["nodes"][i]): float(rl[i])
                                         for i in range(1, len(rl))}, "winner": "-"}
            if rows:
                with open(out / f"{name}_branch_rates.csv", "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                report.append(f"Variable rates: {name}_branch_rates.csv and one <set>__<variable>_rate_tree.png per variable")
                for r in rate_summary:
                    if r["trait_set"] == name:
                        report.append(f"  {r['variable']}: fastest branches (rate / median) {r['fastest']}")
        if "surface" in a.analyses:
            r_, chrono = surface_section(name, S, tree, tips, a, out)
            report += r_
            S["chrono"] = chrono
        if colour_req:
            report += colour_analyses(name, S, tree, tips, a, out, rng, given_groups, gmap, S.get("chrono"))
        if "mk_maps" in a.analyses:
            have = frozenset(t.name for t in tips if t.name in disc[0])
            if len(have) < 4:
                report.append(f"Mk model: only {len(have)} species have a {disc[2]} state - skipped")
            else:
                if have not in mk_cache:
                    tag = "" if not mk_cache else f"_{name}"
                    r_, mk_cache[have] = mk_section(tree, disc[0], disc[2], a, out, rng, tag)
                    report += r_
                else:
                    report.append(f"Mk model of {disc[2]}: same species as before - see {mk_cache[have]['prefix']}_*")
                if "oum" in a.analyses:
                    otree = tree if len(have) == len(tips) else prune(_copy_tree(tree), set(have))
                    otips = tips_of(otree)
                    So = S if len(have) == len(tips) else {**S, "idx": [S["species"].index(t.name) for t in otips]}
                    report += oum_section(name, So, otree, otips, a, out, mk_cache[have])
        if "ancestral_pattern" in a.analyses:
            mk_ = None
            if disc is not None:
                have_ = frozenset(t.name for t in tips if t.name in disc[0])
                mk_ = mk_cache.get(have_) if len(have_) == len(tips) else None
            report += ancestral_pattern_section(name, S, tree, tips, a, out, rng, mk_)
        report.append("")

    if "discrete_tree" in a.analyses:
        report += discrete_tree_section(root, a, out)
    if a.part_sets:
        report += part_rates_section(sets, a, out, rng)
    if a.combine_sets and "reference_scores" in a.analyses:
        r_, rows_ = combined_reference_scoring(a, out)
        report += r_
        ref_rows += rows_

    if "pgls" in a.analyses:
        for f in a.pgls:
            ys, pred = [x.strip() for x in f.split("~")]
            S = sets.get(ys)
            if S is None or "Y" not in S:
                report.append(f"PGLS {f}: trait set {ys} not analysed - skipped"); continue
            if pred == "logCS":
                if S["size"] is None:
                    report.append(f"PGLS {f}: no size column in {ys} (--size_col) - skipped"); continue
                xv = np.log(np.array([S["size"][S["species"].index(t.name)] for t in S["tips"]]))
            else:
                s2, cc = pred.split(":", 1); src = sets.get(s2)
                if src is None or cc not in src["cols"]:
                    report.append(f"PGLS {f}: predictor column not found - skipped"); continue
                xv = np.array([src["raw"][src["species"].index(t.name), src["cols"].index(cc)] for t in S["tips"]])
            r = pgls(S["Y"], xv, S["C"], a.iterations, rng)
            pgls_rows.append({"model": f, **r})
            report.append(f"PGLS {f}: R2 = {r['Rsq']:.4f}, F = {r['F']:.3f}, P = {r['P']:.4f}")

    if ref_rows:
        _write_csv(out / "reference_scores_summary.csv", ref_rows)
    for name in sets:
        rows = [r for r in model_rows if r["trait_set"] == name]
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(out / f"{name}_model_fits.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for fn, rows in (("phylogenetic_signal.csv", signal_rows), ("pgls.csv", pgls_rows), ("rate_fits.csv", rate_summary)):
        if rows:
            with open(out / fn, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "phylo_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
