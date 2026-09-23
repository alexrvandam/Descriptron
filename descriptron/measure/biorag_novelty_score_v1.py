#!/usr/bin/env python3
"""
biorag_novelty_score_v1.py — is this specimen outside every described species?
==============================================================================

A key that fails to place a specimen is a trigger, not evidence: it can fail
because a structure is missing, the specimen is damaged, or the sex is wrong.
This script asks the quantitative question instead, calibrated on the species
that are already described, and never with a universal cut-off:

  1  calibrate on the known species — for every character set, the distances
     between specimens of the SAME species and between DIFFERENT species;
  2  score a candidate as its distance to the nearest species in units of that
     species' own spread (g = d_nearest / median within-species distance);
  3  choose the threshold by error rate, not by a magic number: leave one
     SPECIMEN out over the known species gives the false-positive rate (a
     genuine member called novel) at any threshold; leave one SPECIES out gives
     the true-positive rate (a species the reference set has never seen);
  4  require congruence — outside the threshold in at least two independent
     character sets, not one;
  5  require >= 3 specimens (and report the candidate's own spread, so an
     aberrant individual or a mixed series is visible);
  6  print the key path: where the specimen fails, how many couplets it
     contradicts, and by how much it falls outside the terminal species.

Character sets (each calibrated separately):
  size          lengths, widths, areas, perimeters, landmark distances (mm)
  ratio         length/width and the standard proportions (scale-free)
  landmark      landmark distances / longest span (scale-free shape)
  colour        CIE L*a*b* of whole structures and of their thirds
  subjective    the image-read words turned into MULTISTATE characters (surface
                sculpture, lustre, margin, outline, curvature, apex, colour
                pattern, vestiture) per structure. A state change — striate vs
                crenulate, mottled vs uniform, concave vs convex — is a discrete
                difference and counts far more than a small shift in a
                measurement; the report names every state that differs.

Modes:
  --holdout_species CODE...   treat these species as unknown (a test)
  --holdout_all               do that for every species in turn (validation)
  --candidate_specimens ID... score specimens that are in the matrix
  --candidate_coco FILE       score an outside specimen set from a COCO file;
                              only the scale-free sets are used unless the file
                              carries a scale (mm) for its images

Usage:
  python biorag_novelty_score_v1.py --matrix_dir "$M/compiled_key_tier" \\
     --taxon_profile <profile.yaml> --key_tree "$M/key/key_tree.json" \\
     --subjective "$M/descriptions/subjective_character_flags.tsv" \\
     --holdout_species sp5 --out_dir "$M/novelty"
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import biorag_feature_policy as pol  # noqa: E402

VERSION = "1.0"
SETS = {
    "size": {"families": {"length", "area", "landmark_mm"}, "scale_free": False,
             "label": "size (mm: lengths, widths, areas, perimeters)"},
    "ratio": {"families": {"aspect_ratio", "ratio"}, "scale_free": True,
              "label": "proportions (length/width and standard ratios)"},
    "landmark": {"families": {"landmark_ratio"}, "scale_free": True,
                 "label": "landmark shape (distances / longest span)"},
    "colour": {"families": {"colour"}, "scale_free": True,
               "label": "colour (CIE L*a*b*, whole structure and thirds)"},
}
MIN_SHARED = 4          # features two specimens must share before they are compared
TOPK = 3                # how many of the largest character jumps the "jump" statistic averages
STATS = ("mean", "jump")


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


# ─────────────────────────────────────────────────────────────────────────────
# Reference data
# ─────────────────────────────────────────────────────────────────────────────

# Computed stand-ins for the descriptive characters a vision model does not
# repeat reliably: shape and margin from the outline, colour pattern from the
# colour-extraction metrics, sculpture from the texture features. All of these
# are measured by the pipeline already (texture_phenomics_homology.py,
# color_extraction / colour-homology, the measurement step).
COMPUTED_GROUPS = {
    "outline": [r'^meas_solidity$', r'^meas_extent$', r'^shape_PC[1-5]$'],
    "colour pattern": [r'^color_(adaptive|median)_(n_markings|pattern_complexity|boundary_strength|'
                       r'n_boundaries_(same|diff)|hue_std|sat_std|bri_std|mean_aspect_ratio|'
                       r'mean_shape_complexity|total_marking_area_mm2)$'],
    "sculpture": [r'^tex_phylo_PC([1-9]|10)$'],
}
# the colour-homology grid is condensed rather than used cell by cell: the mean
# and the spread of cell entropy, and the spread of cell lightness, are what a
# taxonomist means by "uniform" versus "mottled"
GRID_SUMMARIES = {
    "grid_entropy_mean": (r'^colhom_r\d+c\d+_color_entropy$', "mean"),
    "grid_entropy_sd": (r'^colhom_r\d+c\d+_color_entropy$', "std"),
    "grid_lightness_sd": (r'^colhom_r\d+c\d+_L_mean_abs$', "std"),
    "grid_a_sd": (r'^colhom_r\d+c\d+_a_mean_abs$', "std"),
    "grid_b_sd": (r'^colhom_r\d+c\d+_b_mean_abs$', "std"),
}


def computed_table(compiled_dir: Path, profile: Dict) -> pd.DataFrame:
    """Per specimen x structure table of the measured shape, colour-pattern and
    texture metrics that stand in for the unreliable descriptive characters."""
    src = next(iter(sorted(Path(compiled_dir).glob("*_full_features.csv"))), None)
    if src is None:
        raise FileNotFoundError(f"no *_full_features.csv in {compiled_dir}")
    f = pd.read_csv(src, low_memory=False)
    if "specimen_id" not in f.columns:
        f["specimen_id"] = [pol.specimen_id(ib, sp, profile)
                            for ib, sp in zip(f["image_base"], f["group_label"])]
    for name, (pat, how) in GRID_SUMMARIES.items():
        grid = [c for c in f.columns if re.match(pat, c)]
        if grid:
            f[name] = f[grid].mean(axis=1) if how == "mean" else f[grid].std(axis=1)
    pats = [re.compile(p) for group in COMPUTED_GROUPS.values() for p in group]
    cols = [c for c in f.columns if any(p.match(c) for p in pats)] + \
           [n for n in GRID_SUMMARIES if n in f.columns]
    if not cols:
        return pd.DataFrame()
    g = f.groupby(["specimen_id", "category"])[cols].mean()
    wide = g.unstack("category")
    wide.columns = [f"{cat}.{col}" for col, cat in wide.columns]
    return wide.dropna(axis=1, how="all")


def descriptive_table(path: Path, characters: Dict, coarse: bool = False,
                      allowed: Optional[set] = None) -> pd.DataFrame:
    """Per-specimen descriptive states -> a numeric table the same machinery can
    use: an ordinal character becomes one column scaled 0-1 along its scale, a
    nominal character becomes one 0/1 column per state. "not assessable" stays
    missing, so it is simply not compared.

    coarse=True uses the merged bands (smooth / finely sculptured / coarsely
    sculptured ...) instead of the full scale — the level a second observer
    reproduces. allowed restricts the table to named characters (the reliability
    gate)."""
    df = pd.read_csv(path, sep="\t")
    df = df[df["state"].astype(str).str.lower() != "not assessable"]
    cols: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in df.itertuples():
        cfg = characters.get(r.character)
        if cfg is None or (allowed is not None and r.character not in allowed):
            continue
        state = str(r.state).lower()
        cmap, cranks = (coarse_map(r.character), coarse_ranks(r.character)) if coarse else ({}, {})
        if coarse and cmap:
            if state not in cranks:
                continue
            band_names = list(dict.fromkeys(cmap[w] for w in cmap))
            if cfg["type"] == "ordinal":
                span = max(1.0, len(set(cranks.values())) - 1)
                cols[f"{r.category}|{r.character}"][r.specimen_id] = cranks[state] / span
            else:
                for band in band_names:
                    cols[f"{r.category}|{r.character}={band}"][r.specimen_id] = \
                        1.0 if cmap[state] == band else 0.0
            continue
        canon = canonical_states(cfg)
        ranks = state_ranks(cfg)
        if state not in ranks:
            continue
        if cfg["type"] == "ordinal":
            span = max(1.0, len(canon) - 1)
            cols[f"{r.category}|{r.character}"][r.specimen_id] = ranks[state] / span
        else:
            for st in canon:
                cols[f"{r.category}|{r.character}={st}"][r.specimen_id] = \
                    1.0 if st.lower() == canon[int(ranks[state])].lower() else 0.0
    t = pd.DataFrame(cols)
    return t.sort_index()


class Reference:
    """Specimen × feature tables per character set, standardised in units of the
    pooled within-species SD, plus the species each specimen belongs to."""

    def __init__(self, matrix_dir: Path, profile: Dict, descriptive_matrix: Optional[Path] = None,
                 characters: Optional[Dict] = None, computed_dir: Optional[Path] = None,
                 coarse: bool = False, allowed_characters: Optional[set] = None,
                 holdout_scale: bool = True):
        self.profile = profile
        # The pooled within-species SD is the unit every distance is expressed in, so it is a
        # statistic the specimen under test is scored against. With holdout_scale (the default)
        # it is re-derived without whatever is withheld (scale_for); False restores the single
        # scale computed once over everything, which is what runs before 2026-09-20 used.
        self.holdout_scale = holdout_scale
        self._scache = {}
        long = pd.read_csv(matrix_dir / "specimen_matrix_long.csv")
        self.fdict = pd.read_csv(matrix_dir / "feature_dictionary.tsv", sep="\t").set_index("feature_id")
        self.species_of = dict(zip(long["specimen_id"], long["species"]))
        self.sex_of = dict(zip(long["specimen_id"], long["sex"]))
        wide = long.pivot_table(index="specimen_id", columns="feature_id", values="value", aggfunc="mean")
        self.raw = wide                      # untransformed values (key path, report)
        self.logged = set()
        self.tables, self.scales = {}, {}
        for name, cfg in SETS.items():
            cols = [c for c in wide.columns
                    if c in self.fdict.index and self.fdict.loc[c, "family"] in cfg["families"]]
            t = wide[cols].copy()
            if cfg["families"] & {"length", "area", "landmark_mm"}:
                t = t.apply(lambda s: np.log(s.where(s > 0)))        # sizes compare on a log scale
                self.logged.update(cols)
            self.tables[name] = t
            self.scales[name] = self._within_sd(t)
        if descriptive_matrix is not None:
            # the descriptive characters become an ordinary character set, so they
            # are weighted and calibrated exactly like the measured ones
            dt = descriptive_table(Path(descriptive_matrix), characters or DESCRIPTIVE_CHARACTERS,
                                   coarse=coarse, allowed=allowed_characters)
            dt = dt.reindex(wide.index)
            SETS["descriptive"] = {"families": set(), "scale_free": True,
                                   "label": "descriptive characters (scored per specimen)"}
            self.tables["descriptive"] = dt
            self.scales["descriptive"] = self._within_sd(dt)
        if computed_dir is not None:
            ct = computed_table(Path(computed_dir), profile).reindex(wide.index)
            if len(ct.columns):
                SETS["computed"] = {"families": set(), "scale_free": True,
                                    "label": "computed shape, colour-pattern and texture metrics"}
                self.tables["computed"] = ct
                self.scales["computed"] = self._within_sd(ct)
        self.species = sorted({s for s in self.species_of.values()}, key=natural_key)
        self._wcache = {}

    def _within_sd(self, t: pd.DataFrame) -> pd.Series:
        """Pooled within-species SD per feature (robust, never zero)."""
        sp = pd.Series({i: self.species_of.get(i, "?") for i in t.index})
        centred = t.sub(t.groupby(sp).transform("median"))
        sd = centred.abs().median() * 1.4826
        med = t.abs().median()
        fallback = (med.replace(0, np.nan) * 0.05).fillna(1.0)
        return sd.where(sd > 1e-12, fallback).fillna(1.0)

    def z(self, name: str, row: pd.Series) -> pd.Series:
        return row / self.scales[name]

    def scale_for(self, name: str, exclude_species=(), exclude_specimens=()) -> pd.Series:
        """The scale of one character set, computed WITHOUT the material under test.
        Withholding a specimen from the distances while leaving it inside the spread those
        distances are divided by is not a hold-out."""
        if not self.holdout_scale or not (exclude_species or exclude_specimens) \
                or name in getattr(self, "unit_scale", ()):      # present/absent sets: scale is 1
            return self.scales[name]
        t = self.tables[name]
        drop = frozenset(i for i in t.index
                         if self.species_of.get(i) in exclude_species or i in exclude_specimens)
        if not drop:
            return self.scales[name]
        key = (name, drop)
        if key not in self._scache:
            self._scache[key] = self._within_sd(t.drop(index=list(drop)))
        return self._scache[key]

    def weights(self, name: str, exclude_species=(), exclude_specimens=()) -> pd.Series:
        """How much each character separates the species that are IN the reference
        set (between-species spread / within-species spread). Characters that do
        not separate anything get little weight; the candidate's own species is
        excluded so the weights cannot be tuned on it."""
        held = tuple(sorted(exclude_specimens)) if self.holdout_scale else ()
        key = (name, tuple(sorted(exclude_species)), held)
        if key in self._wcache:
            return self._wcache[key]
        t = self.tables[name]
        keep = [i for i in t.index if self.species_of.get(i) not in exclude_species
                and i not in held]
        sp = pd.Series({i: self.species_of[i] for i in keep})
        med = t.loc[keep].groupby(sp).median()
        between = med.std()
        w = (between / self.scale_for(name, exclude_species, held)).clip(lower=0.0, upper=5.0).fillna(0.0)
        w = w.where(w > 0.05, 0.05)                     # never fully ignore a character
        self._wcache[key] = w
        return w

    def specimens(self, exclude_species=(), exclude_specimens=()) -> List[str]:
        return [s for s in self.tables["ratio"].index
                if self.species_of.get(s) not in exclude_species and s not in exclude_specimens]


def distance(a: pd.Series, b: pd.Series, scale: pd.Series, min_shared=MIN_SHARED,
             weights: Optional[pd.Series] = None, stat: str = "mean") -> Tuple[float, int]:
    """Difference between two specimens in within-species SD units.

    stat="mean": weighted mean over the shared characters (weights = how much
                 each character separates the known species).
    stat="jump": the mean of the TOPK largest single-character differences —
                 a new species usually differs strongly in a few characters and
                 is unremarkable in the rest.
    """
    both = a.notna() & b.notna()
    n = int(both.sum())
    if n < min_shared:
        return math.nan, n
    z = (a[both] - b[both]).abs() / scale[both]
    if stat == "jump":
        k = min(TOPK, len(z))
        return float(np.sort(z.values)[-k:].mean()), n
    w = (weights[both] if weights is not None else pd.Series(1.0, index=z.index))
    if w.sum() <= 0:
        return float(z.mean()), n
    return float((z * w).sum() / w.sum()), n


def species_distance(ref: Reference, name: str, cand: pd.Series, species: str,
                     exclude_specimens=(), weights=None, stat="mean", scale=None) -> Tuple[float, int]:
    """Median distance from the candidate to the specimens of one species."""
    t = ref.tables[name]
    scale = ref.scales[name] if scale is None else scale
    ids = [i for i in t.index if ref.species_of.get(i) == species and i not in exclude_specimens]
    ds = [distance(cand, t.loc[i], scale, weights=weights, stat=stat) for i in ids]
    ok = [d for d, n in ds if d == d]
    return (float(np.median(ok)) if ok else math.nan), len(ok)


def within_spread(ref: Reference, name: str, species: str, exclude_specimens=(),
                  weights=None, stat="mean", scale=None) -> float:
    """Median distance between specimens of one species (its own spread)."""
    t = ref.tables[name]
    scale = ref.scales[name] if scale is None else scale
    ids = [i for i in t.index if ref.species_of.get(i) == species and i not in exclude_specimens]
    ds = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d, _ = distance(t.loc[ids[i]], t.loc[ids[j]], scale, weights=weights, stat=stat)
            if d == d:
                ds.append(d)
    return float(np.median(ds)) if ds else math.nan


def pooled_spread(ref: Reference, name: str, exclude_species=(), weights=None, stat="mean",
                  scale=None, exclude_specimens=()) -> float:
    vals = [within_spread(ref, name, s, exclude_specimens, weights=weights, stat=stat, scale=scale)
            for s in ref.species if s not in exclude_species]
    vals = [v for v in vals if v == v]
    return float(np.median(vals)) if vals else math.nan


def score_candidate(ref: Reference, name: str, cand: pd.Series, exclude_species=(),
                    exclude_specimens=(), stat="mean") -> Dict:
    """g = distance to the nearest species / that species' own spread."""
    weights = ref.weights(name, exclude_species, exclude_specimens)
    scale = ref.scale_for(name, exclude_species, exclude_specimens)
    rows = []
    for s in ref.species:
        if s in exclude_species:
            continue
        d, n = species_distance(ref, name, cand, s, exclude_specimens, weights, stat, scale=scale)
        if d == d:
            rows.append((s, d, n))
    if not rows:
        return {"g": math.nan, "nearest": None, "d": math.nan, "second": None}
    rows.sort(key=lambda r: r[1])
    nearest, d, _ = rows[0]
    w = within_spread(ref, name, nearest, exclude_specimens, weights, stat, scale=scale)
    if not (w == w):
        w = pooled_spread(ref, name, exclude_species, weights, stat, scale=scale,
                          exclude_specimens=exclude_specimens if ref.holdout_scale else ())
    return {"g": d / w if w and w == w else math.nan, "nearest": nearest, "d": d, "stat": stat,
            "spread_nearest": w, "second": rows[1][0] if len(rows) > 1 else None,
            "d_second": rows[1][1] if len(rows) > 1 else math.nan,
            "ranking": rows[:5]}


def group_spread(ref: Reference, name: str, ids, weights=None, stat="mean") -> float:
    """Median distance between the candidate's own specimens."""
    t = ref.tables[name]
    ds = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if ids[i] in t.index and ids[j] in t.index:
                d, _ = distance(t.loc[ids[i]], t.loc[ids[j]], ref.scales[name], weights=weights, stat=stat)
                if d == d:
                    ds.append(d)
    return float(np.median(ds)) if ds else math.nan


def group_score(ref: Reference, name: str, ids, exclude_species=(), stat="mean") -> Dict:
    """Barcode-gap logic at series level: how far the candidate series sits from
    the nearest described species, in units of its OWN spread. A real species is
    tight internally and far from everything else; an aberrant individual or a
    mixed series is not."""
    weights = ref.weights(name, exclude_species)
    t = ref.tables[name]
    ids = [i for i in ids if i in t.index]
    if not ids:
        return {"G": math.nan, "nearest": None}
    within = group_spread(ref, name, ids, weights, stat)
    pooled = pooled_spread(ref, name, exclude_species, weights, stat)
    floor = 0.5 * pooled if pooled == pooled else math.nan
    w_eff = max(within, floor) if within == within and floor == floor else (
        within if within == within else pooled)
    rows = []
    for sp in ref.species:
        if sp in exclude_species:
            continue
        others = [i for i in t.index if ref.species_of.get(i) == sp and i not in ids]
        ds = []
        for c in ids:
            for o in others:
                d, _ = distance(t.loc[c], t.loc[o], ref.scales[name], weights=weights, stat=stat)
                if d == d:
                    ds.append(d)
        if ds:
            rows.append((sp, float(np.median(ds))))
    if not rows or not (w_eff == w_eff) or w_eff <= 0:
        return {"G": math.nan, "nearest": None, "within": within}
    rows.sort(key=lambda r: r[1])
    return {"G": rows[0][1] / w_eff, "nearest": rows[0][0], "d_nearest": rows[0][1],
            "within": within, "stat": stat, "ranking": rows[:5],
            "second": rows[1][0] if len(rows) > 1 else None}


def calibrate_groups(ref: Reference, fpr: float, min_group=3) -> Dict:
    """Same two experiments, but for a SERIES of specimens: a genuine sub-series
    of a known species (its species stays in the reference) versus a species the
    reference set has never seen."""
    cal = {}
    for name in SETS:
        for stat in STATS:
            ins, outs = [], []
            for sp in ref.species:
                ids = [i for i in ref.tables[name].index if ref.species_of.get(i) == sp]
                if len(ids) < min_group + 1:          # need some specimens left behind
                    continue
                C = ids[:min_group]
                r_in = group_score(ref, name, C, stat=stat)
                r_out = group_score(ref, name, C, exclude_species=(sp,), stat=stat)
                if r_in.get("G") == r_in.get("G"):
                    ins.append(r_in["G"])
                if r_out.get("G") == r_out.get("G"):
                    outs.append(r_out["G"])
            if len(ins) < 5:
                continue
            thr = float(np.quantile(ins, 1 - fpr))
            v = {"stat": stat, "threshold": thr, "target_fpr": fpr, "n_in": len(ins), "n_out": len(outs),
                 "observed_fpr": float(np.mean([g > thr for g in ins])),
                 "tpr": float(np.mean([g > thr for g in outs])) if outs else math.nan,
                 "in_median": float(np.median(ins)), "out_median": float(np.median(outs)) if outs else math.nan}
            cal.setdefault(name, {"variants": {}})["variants"][stat] = v
        if name in cal and cal[name]["variants"]:
            best = max(cal[name]["variants"].values(),
                       key=lambda v: (v["tpr"] if v["tpr"] == v["tpr"] else -1))
            cal[name].update(best)
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# Descriptive (multistate) characters — species-level
# ─────────────────────────────────────────────────────────────────────────────

# Descriptive characters are read as MULTISTATE characters, and most of them are
# ORDINAL: the states form a degree scale, so "smooth -> punctate -> pitted ->
# rugose" is three steps apart, not simply "different". Intensity modifiers
# (weakly, finely, deeply, coarsely) move a state half a step along its scale.
# Characters whose states are kinds rather than degrees (striate vs reticulate,
# falcate vs clavate) stay nominal. A taxon profile may replace or extend this
# vocabulary under `descriptive_characters`.
DESCRIPTIVE_CHARACTERS = {
    # Each entry of "states" is one position on the scale. A list gives synonyms
    # that mean the same thing (they are NOT a step apart); the first is the
    # canonical term shown to a scorer.
    "surface roughness": {"type": "ordinal", "states": [
        "polished", "smooth", ["alutaceous", "shagreened"], "coriaceous", "punctulate", "punctate",
        ["foveate", "pitted"], "rugulose", "rugose", ["tuberculate", "wrinkled"]]},
    "surface pattern": {"type": "nominal", "states": [
        ["striate", "striated"], ["costate", "costulate"], "crenulate", "reticulate",
        ["granulate", "granular"]]},
    "lustre": {"type": "ordinal", "states": [
        ["matte", "dull"], ["subopaque", "opaque"], ["shining", "shiny"], ["glossy", "lustrous"]]},
    "transparency": {"type": "ordinal", "states": [
        "opaque", "translucent", "semitransparent", ["hyaline", "transparent"]]},
    "margin incision": {"type": "ordinal", "states": [
        "entire", ["undulate", "sinuate", "sinuous"], "crenulate", ["denticulate", "serrate"], "lobate",
        ["notched", "emarginate"], "bilobed"]},
    "elongation": {"type": "ordinal", "states": [
        "globular", ["subquadrate", "quadrate"], ["ovate", "oval", "obovate"], "cuneate", "fusiform",
        "elongate", "slender", "linear"]},
    "outline shape": {"type": "nominal", "states": [
        ["falcate", "falciform"], "clavate", ["digitiform", "finger-shaped"], "spatulate", "conical",
        ["cylindrical", "subcylindrical"], "lanceolate", ["triangular", "deltoid"],
        ["strap-shaped", "blade-shaped"]]},
    "curvature": {"type": "ordinal", "states": [
        "concave", ["flattened", "compressed"], "straight", ["curved", "arcuate", "bowed"], "convex",
        ["inflated", "swollen"]]},
    "apex": {"type": "ordinal", "states": [
        ["acuminate", "attenuate"], ["acute", "pointed"], ["obtuse", "blunt"], "rounded", "truncate",
        "emarginate", "bilobed"]},
    "colour heterogeneity": {"type": "ordinal", "states": [
        ["uniform", "unicolorous", "concolorous"], ["suffused", "infuscate", "infuscated"],
        ["bicolored", "bicoloured"], ["banded", "striped"], ["spotted", "maculate"],
        ["marbled", "clouded", "variegated"], "mottled"]},
    "vestiture": {"type": "ordinal", "states": [
        "glabrous", "setulose", "setose", "pubescent", "pilose", ["hairy", "bristly"]]},
}


def canonical_states(cfg) -> List[str]:
    """The term shown to a scorer for each position on the scale."""
    return [s[0] if isinstance(s, list) else s for s in cfg["states"]]


def state_ranks(cfg) -> Dict[str, float]:
    """Every accepted word (synonyms included) -> its position on the scale."""
    out = {}
    for i, s in enumerate(cfg["states"]):
        for w in (s if isinstance(s, list) else [s]):
            out[w.lower()] = float(i)
    return out


# Coarse bands: adjacent states merged into the 2-4 distinctions a second
# observer reproduces. Scoring stays fine-grained; the coarse view is what the
# reliability gate and (with --coarse_descriptive) the delimitation score use.
COARSE_BANDS = {
    "surface roughness": {"smooth": ["polished", "smooth", "alutaceous", "shagreened"],
                          "finely sculptured": ["coriaceous", "punctulate", "punctate"],
                          "coarsely sculptured": ["foveate", "pitted", "rugulose", "rugose",
                                                  "tuberculate", "wrinkled"]},
    "lustre": {"dull": ["matte", "dull", "subopaque", "opaque"],
               "shining": ["shining", "shiny", "glossy", "lustrous"]},
    "transparency": {"opaque": ["opaque"], "translucent": ["translucent", "semitransparent"],
                     "transparent": ["hyaline", "transparent"]},
    "margin incision": {"entire": ["entire", "undulate"],
                        "weakly incised": ["sinuate", "sinuous", "crenulate", "denticulate"],
                        "deeply incised": ["serrate", "lobate", "notched", "emarginate", "bilobed"]},
    "elongation": {"compact": ["globular", "subquadrate", "quadrate", "ovate", "oval", "obovate"],
                   "moderate": ["cuneate", "fusiform"],
                   "elongate": ["elongate", "slender", "linear"]},
    "curvature": {"concave": ["concave"],
                  "flat": ["flattened", "compressed", "straight"],
                  "convex": ["curved", "arcuate", "bowed", "convex", "inflated", "swollen"]},
    "apex": {"pointed": ["acuminate", "attenuate", "acute", "pointed"],
             "blunt": ["obtuse", "blunt", "rounded"],
             "truncate or incised": ["truncate", "emarginate", "bilobed"]},
    "colour heterogeneity": {"uniform": ["uniform", "unicolorous", "concolorous", "suffused",
                                         "infuscate", "infuscated"],
                             "two-toned": ["bicolored", "bicoloured", "banded", "striped"],
                             "patterned": ["spotted", "maculate", "marbled", "clouded", "variegated",
                                           "mottled"]},
    "vestiture": {"glabrous": ["glabrous"], "sparsely setose": ["setulose", "setose"],
                  "densely setose": ["pubescent", "pilose", "hairy", "bristly"]},
}


def coarse_map(character: str) -> Dict[str, str]:
    """state (any synonym) -> coarse band name; empty when the character has none."""
    bands = COARSE_BANDS.get(character, {})
    return {w.lower(): band for band, words in bands.items() for w in words}


def coarse_ranks(character: str) -> Dict[str, float]:
    """state -> position of its coarse band (for ordinal distances)."""
    bands = COARSE_BANDS.get(character, {})
    return {w.lower(): float(i) for i, (band, words) in enumerate(bands.items()) for w in words}


MODIFIERS = {"weakly": -0.5, "faintly": -0.5, "finely": -0.5, "slightly": -0.5, "sparsely": -0.5,
             "shallowly": -0.5, "moderately": 0.0, "strongly": 0.5, "deeply": 0.5, "coarsely": 0.5,
             "densely": 0.5, "heavily": 0.5, "markedly": 0.5, "distinctly": 0.5, "very": 0.5}


class Descriptive:
    """Species x (structure, character) -> a position on an ordinal scale, or a
    set of states for nominal characters, read from the image-read words that
    biorag_subjective_checks_v1.py recorded."""

    KINDS = ("texture", "shape", "setation", "colour")

    def __init__(self, path: Optional[Path], profile: Optional[Dict] = None):
        chars = {k: dict(v) for k, v in DESCRIPTIVE_CHARACTERS.items()}
        chars.update((profile or {}).get("descriptive_characters") or {})
        self.characters = chars
        self.rank_of, self.char_of, self.canonical = {}, {}, {}
        for c, cfg in chars.items():
            self.canonical[c] = canonical_states(cfg)
            for w, rank in state_ranks(cfg).items():
                self.char_of.setdefault(w, c)
                self.rank_of[(c, w)] = rank
        self.cells: Dict[str, Dict[Tuple[str, str], List]] = defaultdict(lambda: defaultdict(list))
        self.free: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        self.ok = False
        if not path or not Path(path).exists():
            return
        df = pd.read_csv(path, sep="\t").fillna("")
        for _, r in df.iterrows():
            if str(r.get("kind")) not in self.KINDS:
                continue
            sp, struct = str(r["species"]), str(r.get("structure") or "?")
            words = [w.strip().lower() for w in str(r.get("words", "")).split(",") if w.strip()]
            shift = sum(MODIFIERS.get(w, 0.0) for w in words)
            shift = max(-1.0, min(1.0, shift))
            for w in words:
                ch = self.char_of.get(w)
                if ch is None:
                    if w not in MODIFIERS:
                        self.free[sp][struct].add(w)
                    continue
                self.cells[sp][(struct, ch)].append((w, self.rank_of[(ch, w)] + shift))
        self.ok = bool(self.cells)

    def _value(self, cell_entries, char):
        """Ordinal: mean position on the scale. Nominal: the set of states."""
        if self.characters[char]["type"] == "ordinal":
            return float(np.mean([r for _, r in cell_entries]))
        return {w for w, _ in cell_entries}

    def states(self, cells, cell):
        return sorted({w for w, _ in cells[cell]})

    def compare(self, a: str, b: str, cells_a=None) -> Dict:
        """Cell by cell: for ordinal characters the difference in steps on the
        scale (scaled by the length of the scale); for nominal characters 1 when
        the states are disjoint."""
        A = cells_a if cells_a is not None else self.cells.get(a, {})
        B = self.cells.get(b, {})
        shared = set(A) & set(B)
        if not shared:
            return {"d": math.nan, "n": 0, "n_diff": 0, "differences": []}
        diffs, score = [], 0.0
        for cell in sorted(shared):
            struct, char = cell
            cfg = self.characters[char]
            va, vb = self._value(A[cell], char), self._value(B[cell], char)
            if cfg["type"] == "ordinal":
                span = max(1.0, len(canonical_states(cfg)) - 1)
                steps = abs(va - vb)
                score += min(1.0, steps / span)
                if steps >= 1.0:
                    diffs.append(f"{struct} {char}: {', '.join(self.states(A, cell))} vs "
                                 f"{', '.join(self.states(B, cell))} ({steps:.1f} steps on a "
                                 f"{len(canonical_states(cfg))}-state scale)")
            else:
                if not (va & vb):
                    score += 1.0
                    diffs.append(f"{struct} {char}: {', '.join(sorted(va))} vs {', '.join(sorted(vb))}")
                else:
                    score += 1 - len(va & vb) / len(va | vb)
        return {"d": score / len(shared), "n": len(shared), "n_diff": len(diffs), "differences": diffs}

    def score(self, cand: str, species: List[str], exclude=(), cells_a=None) -> Dict:
        rows = []
        for sp in species:
            if sp in exclude or sp == cand:
                continue
            r = self.compare(cand, sp, cells_a)
            if r["d"] == r["d"]:
                rows.append((sp, r["d"], r["n"], r["n_diff"], r["differences"]))
        if not rows:
            return {"g": math.nan, "nearest": None}
        rows.sort(key=lambda r: r[1])
        sp, d, n, ndiff, diffs = rows[0]
        return {"g": d, "d": d, "nearest": sp, "n_characters": n, "n_state_differences": ndiff,
                "n_features": n, "differences": diffs, "ranking": [(r[0], r[1]) for r in rows[:5]],
                "second": rows[1][0] if len(rows) > 1 else None}

    def differences(self, cand: str, other: str, cells_a=None, limit=8) -> List[str]:
        return self.compare(cand, other, cells_a)["differences"][:limit]


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(ref: Reference, subj: Descriptive, fpr: float) -> Dict:
    """Thresholds from the known species. For every character set both statistics
    are calibrated — the weighted mean over all characters and the mean of the
    largest character jumps — by leaving one SPECIMEN out (how often a genuine
    member is called novel: the false-positive rate, fixed at --fpr) and by
    leaving one SPECIES out (how often a species the reference set has never seen
    is caught: the true-positive rate). The statistic that catches more unseen
    species at the same false-positive rate is the one used."""
    cal = {}
    for name in SETS:
        variants = {}
        t = ref.tables[name]
        for stat in STATS:
            conspecific, novel = [], []
            w_all = ref.weights(name)
            for sid in t.index:
                sp = ref.species_of.get(sid)
                own = [i for i in t.index if ref.species_of.get(i) == sp and i != sid]
                if len(own) < 2:
                    continue
                d, _ = species_distance(ref, name, t.loc[sid], sp, (sid,), w_all, stat)
                wsp = within_spread(ref, name, sp, (sid,), w_all, stat)
                if d == d and wsp and wsp == wsp:
                    conspecific.append(d / wsp)
            for sp in ref.species:
                for sid in [i for i in t.index if ref.species_of.get(i) == sp]:
                    r = score_candidate(ref, name, t.loc[sid], exclude_species=(sp,), stat=stat)
                    if r["g"] == r["g"]:
                        novel.append(r["g"])
            if not conspecific:
                continue
            thr = float(np.quantile(conspecific, 1 - fpr))
            variants[stat] = {
                "stat": stat, "threshold": thr, "target_fpr": fpr,
                "n_conspecific": len(conspecific), "n_novel": len(novel),
                "observed_fpr": float(np.mean([g > thr for g in conspecific])),
                "tpr": float(np.mean([g > thr for g in novel])) if novel else math.nan,
                "conspecific_median": float(np.median(conspecific)),
                "novel_median": float(np.median(novel)) if novel else math.nan,
            }
        if not variants:
            continue
        best = max(variants.values(), key=lambda v: (v["tpr"] if v["tpr"] == v["tpr"] else -1))
        cal[name] = dict(best)
        cal[name]["variants"] = variants
    if subj.ok:
        # Descriptive states are recorded ONCE PER SPECIES, so there is no
        # within-species replicate and no false-positive rate can be estimated
        # from them. The reference distribution is therefore how different the
        # known species are from their own nearest neighbour; a candidate counts
        # as outside when it is more different than (1 - fpr) of those pairs.
        nn = []
        for sp in ref.species:
            r = subj.score(sp, ref.species)
            if r["g"] == r["g"]:
                nn.append(r["g"])
        if nn:
            thr = float(np.quantile(nn, 1 - fpr))
            cal["subjective"] = {
                "stat": "states", "threshold": thr, "target_fpr": math.nan,
                "n_conspecific": len(nn), "n_novel": 0,
                "observed_fpr": math.nan, "tpr": math.nan,
                "conspecific_median": float(np.median(nn)), "novel_median": math.nan,
                "reference": "nearest-neighbour state distance among the known species",
                "note": "one observation per species: no within-species replicate, so no error rate can be "
                        "estimated. The threshold is the {:.0f}th percentile of how different the known "
                        "species are from their nearest neighbour. Score descriptive characters per SPECIMEN "
                        "to make this set testable.".format(100 * (1 - fpr)),
            }
    return cal


def rank_descriptive(subj, species: List[str]) -> pd.DataFrame:
    """Which descriptive cells actually separate species: the share of species
    pairs that differ by at least one state (ordinal) or completely (nominal)."""
    rows = []
    cells = {c for sp in species for c in subj.cells.get(sp, {})}
    for cell in sorted(cells):
        have = [sp for sp in species if cell in subj.cells.get(sp, {})]
        if len(have) < 4:
            continue
        diff = tot = 0
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                a, b = subj.cells[have[i]], subj.cells[have[j]]
                r = subj.compare(have[i], have[j], {cell: a[cell]})
                tot += 1
                diff += int(r["n_diff"] > 0)
        rows.append({"structure": cell[0], "character": cell[1], "type": subj.characters[cell[1]]["type"],
                     "n_species": len(have), "pairs_differing": diff / tot if tot else math.nan,
                     "states_seen": ", ".join(sorted({w for sp in have for w, _ in subj.cells[sp][cell]}))})
    return pd.DataFrame(rows).sort_values("pairs_differing", ascending=False)


def rank_characters(ref: Reference, top=15) -> pd.DataFrame:
    """Which single characters actually separate the known species: leave-one-
    specimen-out nearest-species assignment using that character alone."""
    rows = []
    for name, t in ref.tables.items():
        for col in t.columns:
            v = t[col].dropna()
            if len(v) < 8:
                continue
            sp = pd.Series({i: ref.species_of[i] for i in v.index})
            if sp.nunique() < 3:
                continue
            scale = ref.scales[name][col]
            hit = tot = 0
            for sid in v.index:
                others = v.drop(sid)
                osp = sp.drop(sid)
                med = others.groupby(osp).median()
                if len(med) < 2:
                    continue
                pred = (med - v[sid]).abs().idxmin()
                hit += int(pred == sp[sid])
                tot += 1
            between = v.groupby(sp).median().std()
            rows.append({"set": name, "feature": col,
                         "label": ref.fdict.loc[col, "label"] if col in ref.fdict.index else col,
                         "n_specimens": int(len(v)), "n_species": int(sp.nunique()),
                         "loo_accuracy": hit / tot if tot else math.nan,
                         "between_over_within": float(between / scale) if scale else math.nan})
    df = pd.DataFrame(rows).sort_values(["loo_accuracy", "between_over_within"], ascending=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Key path
# ─────────────────────────────────────────────────────────────────────────────

def key_path(key: Dict, cand: pd.Series, alias=lambda f: f) -> Dict:
    if not key:
        return {}
    by_no = {c["number"]: c for c in key["couplets"]}
    node, steps, conflicts, missing = key["couplets"][0]["number"], [], 0, 0
    seen = set()
    while node in by_no and node not in seen:
        seen.add(node)
        c = by_no[node]
        ch = c["characters"][0]
        fid = alias(ch["feature_id"])
        val = cand.get(fid, math.nan)
        if not (val == val):
            missing += 1
            steps.append(f"couplet {c['number']}: {ch['label']} not measured — path stops")
            return {"terminal": None, "steps": steps, "conflicts": conflicts, "missing": missing,
                    "resolved": False}
        side = "A" if (val <= ch["threshold"] if ch["A_operator"] == "<=" else val > ch["threshold"]) else "B"
        rng = ch[f"{side}_range"]
        out = val < min(rng) or val > max(rng)
        conflicts += int(out)
        steps.append(f"couplet {c['number']}{side.lower()}: {ch['label']} = {val:.4g} "
                     f"({'outside' if out else 'within'} the observed {min(rng):.4g}–{max(rng):.4g})")
        nxt = c[f"{side}_goto"]
        if nxt in by_no:
            node = nxt
        else:
            return {"terminal": nxt, "steps": steps, "conflicts": conflicts, "missing": missing,
                    "resolved": True}
    return {"terminal": None, "steps": steps, "conflicts": conflicts, "missing": missing, "resolved": False}


# ─────────────────────────────────────────────────────────────────────────────
# Candidate scoring and report
# ─────────────────────────────────────────────────────────────────────────────

def deviating_characters(ref: Reference, cand: pd.Series, nearest: str, exclude_specimens=(), top=8,
                         raw=None, ids=()):
    """The characters in which the candidate is furthest from its nearest species,
    reported in the original units (not the log scale used for the distances)."""
    rows = []
    near_ids_all = [i for i in ref.raw.index if ref.species_of.get(i) == nearest
                    and i not in exclude_specimens] if raw is not None else []
    for name, t in ref.tables.items():
        near_ids = [i for i in t.index if ref.species_of.get(i) == nearest and i not in exclude_specimens]
        if not near_ids:
            continue
        med = t.loc[near_ids].median()
        for col in t.columns:
            if col in cand.index and cand[col] == cand[col] and med.get(col) == med.get(col):
                z = abs(cand[col] - med[col]) / ref.scales[name][col]
                if raw is not None and col in raw.columns:
                    cand_ids = [i for i in ids if i in raw.index]
                    cv = (raw.loc[cand_ids, col].median() if cand_ids
                          else (raw.loc[cand.name, col] if cand.name in raw.index else math.nan))
                    nv = raw.loc[near_ids_all, col].median() if near_ids_all else math.nan
                else:
                    cv, nv = cand[col], med[col]
                rows.append({"set": name, "feature": col,
                             "label": ref.fdict.loc[col, "label"] if col in ref.fdict.index else col,
                             "candidate": cv, "nearest_median": nv, "z": z,
                             "unit": ref.fdict.loc[col, "unit"] if col in ref.fdict.index else ""})
    return pd.DataFrame(rows).sort_values("z", ascending=False).head(top)


def score_all_sets(ref, subj, cand_row, cand_code, exclude_species, exclude_specimens, cal):
    out = {}
    for name in SETS:
        stat = cal.get(name, {}).get("stat", "mean")
        r = score_candidate(ref, name, cand_row, exclude_species, exclude_specimens, stat=stat)
        r["threshold"] = cal.get(name, {}).get("threshold", math.nan)
        r["outside"] = bool(r["g"] == r["g"] and r["g"] > r["threshold"])
        out[name] = r
    if subj.ok and cand_code:
        r = subj.score(cand_code, ref.species, exclude=tuple(exclude_species))
        r["threshold"] = cal.get("subjective", {}).get("threshold", math.nan)
        r["outside"] = bool(r["g"] == r["g"] and r["g"] > r["threshold"])
        out["subjective"] = r
    return out


def verdict(scored, n_specimens, min_sets=2, min_specimens=3):
    outside = [k for k, v in scored.items() if v.get("outside")]
    tested = [k for k, v in scored.items() if v.get("g") == v.get("g")]
    if len(outside) >= min_sets:
        head = (f"OUTSIDE the described species in {len(outside)} of {len(tested)} comparable character "
                f"set{'s' if len(tested) != 1 else ''} ({', '.join(outside)})")
        note = ("candidate new morphospecies" if n_specimens >= min_specimens else
                f"only {n_specimens} specimen(s): treat as a flag, not as evidence "
                f"(>= {min_specimens} needed)")
    elif outside:
        head = f"outside in ONE character set only ({outside[0]})"
        note = "not congruent: expected for an aberrant individual or a damaged structure"
    else:
        head = "within the range of the described species"
        note = "consistent with a known species"
    return head, note


def one_page(path: Path, title, cand_label, n_spec, scored, cal, keyres, devs, subj_diffs, ref, extra=(),
             per_specimen=None, series=None, key_note=None, min_sets=2):
    L = [f"# Novelty report — {cand_label}", "",
         f"*{title}*  ", f"generated {datetime.now():%Y-%m-%d %H:%M} by biorag_novelty_score_v1 {VERSION}", ""]
    head, note = verdict(scored, n_spec, min_sets=min_sets)
    L += [f"## Verdict: {head}", f"{note}", "",
          "## Character sets", "",
          "| character set | features compared | nearest species | distance g (within-species SD units) | "
          "threshold | outside? |", "|---|---|---|---|---|---|"]
    for name, r in scored.items():
        if r.get("g") != r.get("g"):
            L.append(f"| {SETS.get(name, {}).get('label', name)} | — | — | not comparable | — | — |")
            continue
        L.append(f"| {SETS.get(name, {}).get('label', name)} | {r.get('n_features', '')} | "
                 f"{r.get('nearest')} | {r['g']:.2f} | {r['threshold']:.2f} | "
                 f"{'**yes**' if r['outside'] else 'no'} |")
    if series:
        L += ["", "### The candidate series against its own spread", "",
              "| character set | distance to nearest species | candidate's own spread | ratio G | "
              "threshold | outside? |", "|---|---|---|---|---|---|"]
        for name, r in series.items():
            if r.get("G") != r.get("G"):
                continue
            L.append(f"| {SETS.get(name, {}).get('label', name)} | {r.get('d_nearest', float('nan')):.2f} | "
                     f"{r.get('within', float('nan')):.2f} | {r['G']:.2f} | {r['threshold']:.2f} | "
                     f"{'**yes**' if r['outside'] else 'no'} |")
        L += ["", "G = 1 means the series sits no further from the nearest described species than its own "
                  "members sit from each other."]
    L += ["", "Thresholds are set on the known species at a chosen false-positive rate; the table below gives "
              "the rate actually achieved and how often a species the reference set has never seen is caught.", "",
          "| character set | threshold | false positives (known specimen called novel) | "
          "true positives (unseen species caught) |", "|---|---|---|---|"]
    for name, c in cal.items():
        base = name.split(":")[-1]
        level = "series" if name.startswith("series:") else "specimen"
        n_in = c.get("n_conspecific", c.get("n_in", 0))
        n_out = c.get("n_novel", c.get("n_out", 0))
        tpr = f"{100 * c['tpr']:.0f}% (n={n_out})" if c.get("tpr") == c.get("tpr") else "—"
        L.append(f"| {SETS.get(base, {}).get('label', base)} ({level}, {c.get('stat', '')}) | "
                 f"{c['threshold']:.2f} | {100 * c['observed_fpr']:.0f}% (n={n_in}) | {tpr} |")
    if keyres:
        L += ["", "## Identification key", ""]
        if key_note:
            L += [f"*{key_note}*", ""]
        L += [f"- {s}" for s in keyres.get("steps", [])]
        L += ["", f"Terminal: **{keyres.get('terminal') or 'not reached'}**; "
                  f"{keyres.get('conflicts', 0)} couplet(s) where the specimen falls outside the observed "
                  f"range; {keyres.get('missing', 0)} character(s) not measured."]
    if devs is not None and len(devs):
        L += ["", "## Where it differs most from its nearest species", "",
              "| character | candidate | nearest species (median) | difference (within-species SD) |",
              "|---|---|---|---|"]
        for _, r in devs.iterrows():
            u = f" {r['unit']}" if r.get("unit") and str(r.get("unit")) != "nan" else ""
            L.append(f"| {r['label']} | {r['candidate']:.4g}{u} | {r['nearest_median']:.4g}{u} | "
                     f"{r['z']:.1f} |")
    if subj_diffs:
        L += ["", "## Descriptive characters that differ in state", "",
              "(image-read; a state change counts far more than a small shift in a measurement)", ""]
        L += [f"- {d}" for d in subj_diffs]
    L += ["", "## How to read this", "",
          "- `g` is the distance to the nearest described species divided by that species' own spread; "
          "g = 1 means “as far away as two specimens of that species usually are from each other”.",
          "- A single set above threshold is weak: damage, sex or a missing structure can do that. Two or more "
          "independent sets is the signal to follow up.",
          "- Morphological distinctness is evidence, not a species. Confirm with more specimens, both sexes, "
          "host and locality data, and ideally sequence data before any nomenclatural act."]
    L += list(extra)
    path.write_text("\n".join(L), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# External COCO candidate (scale-free characters only)
# ─────────────────────────────────────────────────────────────────────────────

def features_from_coco(path: Path, profile: Dict, category_map: Dict[str, str]) -> pd.DataFrame:
    """Per-image, per-structure scale-free shape features (length/width) from
    polygons, using the same PCA convention as the measurement step."""
    j = json.loads(Path(path).read_text())
    cats = {c["id"]: c["name"] for c in j.get("categories", [])}
    imgs = {i["id"]: i["file_name"] for i in j.get("images", [])}
    rows = []
    for a in j.get("annotations", []):
        seg = a.get("segmentation")
        if not seg or not isinstance(seg, list) or not seg[0]:
            continue
        pts = np.array(seg[0], dtype=float).reshape(-1, 2)
        if len(pts) < 5:
            continue
        cat = cats.get(a["category_id"], "?")
        cat = category_map.get(cat, category_map.get(re.sub(r'[^a-z0-9]', '', cat.lower()), cat))
        c = pts - pts.mean(axis=0)
        u, s, vt = np.linalg.svd(c, full_matrices=False)
        proj = c @ vt.T
        length = proj[:, 0].max() - proj[:, 0].min()
        width = proj[:, 1].max() - proj[:, 1].min()
        if width <= 0:
            continue
        rows.append({"image": imgs.get(a["image_id"], str(a["image_id"])), "category": cat,
                     "aspect_ratio": length / width})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    wide = df.pivot_table(index="image", columns="category", values="aspect_ratio", aggfunc="mean")
    wide.columns = [f"{c}.aspect_ratio" for c in wide.columns]
    return wide


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Score a candidate against the described species")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--key_tree", default=None)
    ap.add_argument("--subjective", default=None, help="subjective_character_flags.tsv (species level)")
    ap.add_argument("--coarse_descriptive", action="store_true",
                    help="score the descriptive characters at their coarse bands instead of the full scale. "
                         "NOT the default: on Diaphorina banding cut the detection rate from 33%% to 15%%, "
                         "because the fine distinctions carry the between-species signal even though an "
                         "individual call reproduces only 65-75%% of the time - averaging over many "
                         "structures recovers it, banding discards it for good")
    ap.add_argument("--reliability", default=None,
                    help="character_reliability.json from biorag_character_reliability_v1.py: drops the "
                         "characters it flags as not reproducible and keeps the rest at their full scale "
                         "(add --coarse_descriptive to band them as well)")
    ap.add_argument("--legacy_scale", action="store_true",
                    help="use one pooled within-species SD computed over every specimen, as runs "
                         "before 2026-09-20 did. The default re-derives it without the specimen or "
                         "species under test, because a spread is a statistic the test item must "
                         "not have helped compute")
    ap.add_argument("--computed_dir", default=None,
                    help="compiled feature directory: adds a character set of measured stand-ins for the "
                         "descriptive characters (solidity/extent, shape PCs, colour-pattern metrics, "
                         "texture PCs and colour entropy)")
    ap.add_argument("--descriptive_matrix", default=None,
                    help="descriptive_states_by_specimen.tsv from biorag_descriptive_scoring_v1.py; "
                         "when given, the descriptive characters are treated as a per-specimen character "
                         "set and calibrated like the others")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--holdout_species", nargs="*", default=[])
    ap.add_argument("--holdout_all", action="store_true")
    ap.add_argument("--candidate_specimens", nargs="*", default=[])
    ap.add_argument("--candidate_coco", default=None)
    ap.add_argument("--candidate_label", default="external candidate")
    ap.add_argument("--category_map", default=None,
                    help="JSON {coco category: profile category}, e.g. '{\"entire_forewing\": \"whole_wing\"}'")
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--min_sets", type=int, default=2)
    args = ap.parse_args()

    profile = pol.load_taxon_profile(args.taxon_profile)
    characters = {k: dict(v) for k, v in DESCRIPTIVE_CHARACTERS.items()}
    characters.update(profile.get("descriptive_characters") or {})
    allowed, coarse = None, args.coarse_descriptive
    if args.reliability and Path(args.reliability).exists():
        rel = json.loads(Path(args.reliability).read_text())
        allowed = set(rel.get("use", [])) | set(rel.get("use_coarse", []))
        print(f"reliability gate: using {len(allowed)} characters "
              f"({', '.join(sorted(allowed))}); dropped {rel.get('flag', [])}"
              + ("; scored at the coarse level" if coarse else ""))
    ref = Reference(Path(args.matrix_dir), profile,
                    Path(args.descriptive_matrix) if args.descriptive_matrix else None, characters,
                    Path(args.computed_dir) if args.computed_dir else None, coarse, allowed,
                    holdout_scale=not args.legacy_scale)
    # the species-level word comparison is only used when no per-specimen matrix exists
    subj = Descriptive(Path(args.subjective) if args.subjective and not args.descriptive_matrix else None,
                       profile)
    key = json.loads(Path(args.key_tree).read_text()) if args.key_tree and Path(args.key_tree).exists() else None
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"reference: {len(ref.species)} species, {len(ref.tables['ratio'].index)} specimens; "
          f"sets: " + ", ".join(f"{k} ({ref.tables[k].shape[1]})" for k in SETS) +
          (f", descriptive states ({len(subj.cells)} species, "
          f"{len({c for v in subj.cells.values() for c in v})} structure x character cells)" if subj.ok else ""))
    cal = calibrate(ref, subj, args.fpr)
    (out / "calibration.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk not in ("conspecific", "novel")} for k, v in cal.items()},
        indent=2))
    print("\ncalibration on the known species (FPR = a known specimen called novel; "
          "TPR = an unseen species caught):")
    for k, c in cal.items():
        extra = ""
        if c.get("variants"):
            extra = "   [" + ", ".join(f"{v['stat']}: TPR {100 * v['tpr']:.0f}%" for v in c["variants"].values()
                                       if v["tpr"] == v["tpr"]) + "]"
        tpr = f"{100 * c['tpr']:.0f}%" if c["tpr"] == c["tpr"] else "—"
        print(f"  {k:11s} {c['stat']:5s} threshold g={c['threshold']:.2f}  "
              f"FPR {100 * c['observed_fpr']:.0f}%  TPR {tpr}{extra}")

    gcal = calibrate_groups(ref, args.fpr)
    (out / "calibration_series.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "variants"} for k, v in gcal.items()}, indent=2))
    print("\ncalibration for a SERIES of specimens (candidate's distance to the nearest species "
          "in units of its own spread):")
    for k, c in gcal.items():
        tpr = f"{100 * c['tpr']:.0f}%" if c["tpr"] == c["tpr"] else "—"
        print(f"  {k:11s} {c['stat']:5s} threshold G={c['threshold']:.2f}  "
              f"FPR {100 * c['observed_fpr']:.0f}% (n={c['n_in']})  TPR {tpr} (n={c['n_out']})")

    ranks = rank_characters(ref)
    ranks.to_csv(out / "character_discrimination.tsv", sep="\t", index=False)
    if subj.ok:
        dranks = rank_descriptive(subj, ref.species)
        dranks.to_csv(out / "character_discrimination_descriptive.tsv", sep="\t", index=False)
        print("\nbest descriptive characters (share of species pairs that differ in state):")
        for _, r in dranks.head(8).iterrows():
            print(f"  {100 * r['pairs_differing']:5.1f}%  {r['structure']} {r['character']} "
                  f"({r['type']}, {r['n_species']} spp.): {r['states_seen'][:60]}")
    print(f"\nbest characters (leave-one-specimen-out assignment with that character alone):")
    for _, r in ranks.head(10).iterrows():
        print(f"  {100 * r['loo_accuracy']:5.1f}%  {r['set']:9s} {r['label']}")

    jobs = []
    if args.holdout_all:
        jobs += [("species", s) for s in ref.species]
    jobs += [("species", s) for s in args.holdout_species]
    jobs += [("specimen", s) for s in args.candidate_specimens]
    results = []
    for kind, code in jobs:
        if kind == "species":
            ids = [i for i in ref.tables["ratio"].index if ref.species_of.get(i) == code]
            excl_sp, excl_id = (code,), tuple(ids)
        else:
            ids = [code]
            excl_sp, excl_id = (), (code,)
        if not ids:
            print(f"  {code}: no specimens")
            continue
        per_spec = []
        for sid in ids:
            row = pd.concat([ref.tables[n].loc[sid] for n in SETS])
            row = row[~row.index.duplicated()]
            sc = score_all_sets(ref, subj, row, code if kind == "species" else None,
                                excl_sp, excl_id, cal)
            for n, r in sc.items():
                if n in SETS:
                    r["n_features"] = int(ref.tables[n].loc[sid].notna().sum())
            per_spec.append((sid, sc))
        # series-level score (the candidate's own spread vs the nearest species)
        gscored = {}
        if len(ids) >= 2:
            for n in SETS:
                st = gcal.get(n, {}).get("stat", "mean")
                r = group_score(ref, n, ids, exclude_species=excl_sp, stat=st)
                r["threshold"] = gcal.get(n, {}).get("threshold", math.nan)
                r["outside"] = bool(r.get("G") == r.get("G") and r["G"] > r["threshold"])
                gscored[n] = r
        # per-specimen summary = median g per set
        scored = {}
        for n in list(SETS) + (["subjective"] if subj.ok else []):
            gs = [sc[n]["g"] for _, sc in per_spec if n in sc and sc[n]["g"] == sc[n]["g"]]
            if not gs:
                scored[n] = {"g": math.nan}
                continue
            near = [sc[n].get("nearest") for _, sc in per_spec if n in sc]
            scored[n] = {"g": float(np.median(gs)), "nearest": max(set(near), key=near.count),
                         "threshold": cal.get(n, {}).get("threshold", math.nan),
                         "n_features": int(np.median([sc[n].get("n_features", 0) for _, sc in per_spec
                                                      if n in sc]))}
            scored[n]["outside"] = bool(scored[n]["g"] > scored[n]["threshold"])
        combined = {n: ({"g": gscored[n].get("G"), "nearest": gscored[n].get("nearest"),
                         "threshold": gscored[n].get("threshold"), "outside": gscored[n].get("outside"),
                         "n_features": scored.get(n, {}).get("n_features"), "level": "series"}
                        if n in gscored and gscored[n].get("G") == gscored[n].get("G")
                        else {**scored.get(n, {}), "level": "specimen"})
                    for n in scored}
        head, note = verdict(combined, len(ids), args.min_sets)
        nearest_overall = scored.get("ratio", {}).get("nearest") or next(
            (v.get("nearest") for v in scored.values() if v.get("nearest")), None)
        row0 = pd.concat([ref.tables[n].loc[ids[0]] for n in SETS])
        row0 = row0[~row0.index.duplicated()]
        raw0 = ref.raw.loc[ids[0]]
        kp = key_path(key, raw0) if key else {}
        devs = deviating_characters(ref, row0, nearest_overall, excl_id, raw=ref.raw, ids=ids) \
            if nearest_overall else None
        diffs = subj.differences(code, scored.get("subjective", {}).get("nearest") or "") \
            if subj.ok and kind == "species" else []
        label = (f"{pol.species_display_name(code, profile)} (held out)" if kind == "species"
                 else f"specimen {code}")
        one_page(out / f"novelty_{code}.md",
                 f"scored against the other {len(ref.species) - (1 if kind == 'species' else 0)} species",
                 label, len(ids), combined, {**cal, **{f"series:{k}": v for k, v in gcal.items()}},
                 kp, devs, diffs, ref, per_specimen=scored, series=gscored, min_sets=args.min_sets,
                 key_note=("this species was NOT removed from the key, which was built from all species; "
                           "the path below is therefore not a test of novelty, only a record of where the "
                           "specimen runs" if kind == "species" else None))
        results.append({"candidate": code, "kind": kind, "n_specimens": len(ids), "verdict": head,
                        "sets_outside": sum(1 for v in combined.values() if v.get("outside")),
                        **{f"G_{n}": gscored[n].get("G") for n in gscored},
                        **{f"g_{n}": scored[n].get("g") for n in scored},
                        **{f"out_{n}": combined[n].get("outside") for n in combined},
                        "key_terminal": kp.get("terminal"), "key_conflicts": kp.get("conflicts")})
        print(f"  {code:12s} {head}")

    if args.candidate_coco:
        cmap = json.loads(args.category_map) if args.category_map else {}
        ext = features_from_coco(Path(args.candidate_coco), profile, cmap)
        if ext.empty:
            print("no usable polygons in the COCO file")
        else:
            usable = [c for c in ext.columns if c in ref.tables["ratio"].columns]
            print(f"\nexternal candidate: {len(ext)} images, {len(usable)} scale-free characters shared "
                  f"with the reference ({', '.join(usable[:8])}{' ...' if len(usable) > 8 else ''})")
            per = []
            for img, row in ext[usable].iterrows():
                r = score_candidate(ref, "ratio", row)
                r["n_features"] = int(row.notna().sum())
                per.append((img, r))
                print(f"   {img[:48]:50s} g={r['g']:.2f}  nearest {r['nearest']}")
            # the per-image scores are the evidence for the summary line, so they are written out
            # rather than only printed: a detection rate cannot be checked from a median
            thr = cal["ratio"]["threshold"]
            pd.DataFrame([{"image": img, "g": r["g"], "nearest_species": r["nearest"],
                           "n_characters": r["n_features"], "threshold": thr,
                           "above_threshold": bool(r["g"] == r["g"] and r["g"] > thr)}
                          for img, r in per]).to_csv(out / "external_scores.tsv", sep="\t", index=False)
            gs = [r["g"] for _, r in per if r["g"] == r["g"]]
            scored = {"ratio": {"g": float(np.median(gs)), "nearest": per[0][1]["nearest"],
                                "threshold": cal["ratio"]["threshold"],
                                "n_features": int(np.median([r["n_features"] for _, r in per])),
                                "outside": bool(np.median(gs) > cal["ratio"]["threshold"])}}
            head, note = verdict(scored, len(per), min_sets=1)
            one_page(out / "novelty_external.md",  # only one set is comparable here
                     f"scored on scale-free shape ratios only ({len(usable)} characters shared with the "
                     f"reference set)", args.candidate_label, len(per), scored,
                     {"ratio": cal["ratio"]}, {}, None, [], ref, min_sets=1,
                     extra=["", "## Note", "", "Only the scale-free proportions could be used: the external "
                            "images carry no scale bar, so sizes in mm, colour and the landmark set were not "
                            "comparable. One character set alone cannot support a new species; here it is used "
                            "to test whether the score behaves as expected on material from another genus."])
            results.append({"candidate": args.candidate_label, "kind": "external", "n_specimens": len(per),
                            "verdict": head, "sets_outside": int(scored["ratio"]["outside"]),
                            "g_ratio": scored["ratio"]["g"]})
            print(f"   -> {head}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(out / "novelty_summary.tsv", sep="\t", index=False)
        print(f"\nreports -> {out}")


if __name__ == "__main__":
    main()
