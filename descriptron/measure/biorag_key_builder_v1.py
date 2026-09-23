#!/usr/bin/env python3
"""
biorag_key_builder_v1.py — data-driven dichotomous key (human + machine readable)
================================================================================

Replaces LLM-authored keys. The key TREE is computed from the Tier-1 data
matrix produced by biorag_key_feature_filter_v2.py; an LLM (optional) only
rewords each couplet, and its wording is accepted only if it keeps every
number and adds none. Therefore, by construction:

  * every species is reachable and appears at exactly one terminal;
  * every character is human-measurable (tier 'key' in biorag_feature_policy);
  * every threshold separates the observed specimen values of the two leads
    (or the couplet is explicitly flagged as overlapping, with its accuracy);
  * every number in the key is traceable to a data-matrix feature_id.

Splitting rule (per node = set of species S):
  1. For each key feature measured in every species of S, sort species by
     median value and evaluate every cut. A cut is PERFECT when the smallest
     value on one side exceeds the largest value on the other by at least the
     measurement resolution (mm 0.005, ratios 0.02, CIE 2 units) and 0.25
     pooled within-species SD.
  2. Score = feature-type weight (taxon profile) x structure priority x
     separation x sqrt(balance) x sqrt(sample-size factor).
  3. Preference order: perfect split on a character present in both sexes >
     perfect split by a male character AND a female character together >
     best overlapping split on a both-sex character (flagged) > single-sex split.
  4. Up to N secondary characters that also separate the same two groups are
     added to each lead.

Validation (written to key_validation_report.*):
  reachability, one terminal per species, re-check of every threshold against
  the specimen matrix, forbidden-term scan, number audit of the wording,
  E_Dicho (Pielou evenness of path lengths, as in biorag_key_qualitative_checker),
  resubstitution and leave-one-specimen-out identification accuracy.

Outputs (--output_dir):
  key_tree.json              structured key (the machine-readable master)
  taxonomic_key.txt / .md    human-readable key
  taxonomic_key.jsonld       JSON-LD (schema.org DefinedTermSet + Descriptron terms)
  taxonomic_key.sdd.xml      TDWG SDD 1.1 (Structured Descriptive Data) — key,
                             characters and per-species summary data
  character_matrix.tsv       species x key characters (min-max, n)
  identification_test.tsv    per specimen: resubstitution and LOO results
  key_validation_report.json / .txt

Usage:
  python biorag_key_builder_v1.py \
     --matrix_dir  ".../Diaphorina_monograph/compiled_key_tier" \
     --taxon-profile biorag_prompts/taxon_profiles/diaphorina_taxon_profile.yaml \
     --output_dir  ".../Diaphorina_monograph/key" \
     --llm-backend claude-code            # or api | none
"""

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import biorag_feature_policy as pol  # noqa: E402
from biorag_llm_backend import (add_backend_args, load_prompt_library,  # noqa: E402
                                make_llm_client, parse_json_response)

BUILDER_VERSION = "1.0"

RESOLUTION = {"mm": 0.005, "mm2": 0.0005, "x": 0.02, "L*": 2.0, "a*": 2.0,
              "b*": 2.0, "C*": 2.0, "deg": 5.0}
PRIORITY_WEIGHT = {1: 1.0, 2: 0.85, 3: 0.7}
DEFAULT_TYPE_WEIGHTS = {"ratio": 1.0, "landmark_ratio": 0.95, "aspect_ratio": 0.9,
                        "length": 0.8, "landmark_mm": 0.75, "colour": 0.6}
SEX_SYMBOL = {"male": "♂", "female": "♀"}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Feature:
    fid: str
    label: str
    definition: str
    unit: str
    family: str
    sex: str               # both | male | female
    priority: int
    category: str
    column: str
    equiv: str = ""        # features with the same equiv key measure the same thing


class Matrix:
    def __init__(self, long: pd.DataFrame, fdict: pd.DataFrame, profile: Dict):
        self.profile = profile
        key = fdict[fdict["tier"] == pol.TIER_KEY]
        self.features: Dict[str, Feature] = {}
        for r in key.itertuples():
            self.features[r.feature_id] = Feature(
                r.feature_id, r.label, r.definition, r.unit if isinstance(r.unit, str) else "",
                r.family, r.structure_sex if isinstance(r.structure_sex, str) else "both",
                int(r.key_priority) if r.key_priority == r.key_priority else 2,
                r.category, r.column)
        # equivalence: a ratio whose numerator and denominator are length and width
        # of ONE structure is that structure's aspect ratio
        abbr = profile.get("measurement_abbreviations", {})
        same = {}
        for rd in profile.get("ratios", []):
            n, d = abbr.get(rd.get("num")), abbr.get(rd.get("den"))
            if n and d and n["category"] == d["category"] and \
                    {n["measurement"], d["measurement"]} == {"length_mm", "height_mm"}:
                same[f"ratio_{rd['id']}"] = n["category"]
        for f in self.features.values():
            base = pol.split_category_name(f.category)[0]
            if f.column == "meas_aspect_ratio":
                f.equiv = f"aspect:{f.category}"
            elif f.column in same:
                f.equiv = f"aspect:{same[f.column]}"
            else:
                f.equiv = f.fid
        sub = long[long["feature_id"].isin(self.features)]
        self.long = sub
        self.spec_species = dict(zip(long["specimen_id"], long["species"]))
        sx = long[long["sex"].isin(["male", "female"])].groupby("specimen_id")["sex"].first()
        self.spec_sex = sx.to_dict()
        self.species = sorted(long["species"].unique(), key=natural_key)
        self.values: Dict[str, Dict[str, float]] = defaultdict(dict)   # specimen -> fid -> value
        for sid, fid, v in zip(sub["specimen_id"], sub["feature_id"], sub["value"]):
            self.values[sid][fid] = float(v)
        self.rebuild()

    def rebuild(self, exclude_specimen: Optional[str] = None):
        obs: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for sid, fv in self.values.items():
            if sid == exclude_specimen:
                continue
            sp = self.spec_species[sid]
            for fid, v in fv.items():
                obs[fid][sp].append(v)
        self.obs = {f: {s: np.asarray(v) for s, v in d.items()} for f, d in obs.items()}
        # present/absent characters: every recorded value is 0 or 1 (see binary_quality)
        self.binary = {f for f, d in self.obs.items()
                       if all(((v == 0) | (v == 1)).all() for v in d.values())
                       and len({float(x) for v in d.values() for x in v}) == 2}
        self.n_spec = Counter(self.spec_species[sid] for sid in self.values
                              if sid != exclude_specimen)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def natural_str(s):
    """Hashable natural-sort key (digits zero-padded)."""
    return re.sub(r'\d+', lambda m: m.group().zfill(8), str(s).lower())


# ─────────────────────────────────────────────────────────────────────────────
# Split search
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Split:
    fid: str
    low: Tuple[str, ...]          # species with LOWER values (lead A: <= t)
    high: Tuple[str, ...]         # species with HIGHER values (lead B: > t)
    threshold: float
    threshold_text: str
    perfect: bool
    accuracy: float
    gap: float
    score: float
    low_range: Tuple[float, float] = (0, 0)
    high_range: Tuple[float, float] = (0, 0)

    def partition(self):
        return frozenset(self.low)


def nice_threshold(lo_max: float, hi_min: float, unit: str) -> Tuple[float, str]:
    """Shortest decimal in the CENTRAL half of the gap (generalises better to
    new specimens than a value at the edge of the gap); falls back to any value
    strictly inside the gap, then to the midpoint."""
    mid = (lo_max + hi_min) / 2.0
    q = (hi_min - lo_max) / 4.0
    base = pol.decimals_for(unit, mid)
    for lo_ok, hi_ok in ((lo_max + q, hi_min - q), (lo_max, hi_min)):
        for d in range(0, max(base, 4) + 3):
            t = round(mid, d)
            if lo_ok <= t <= hi_ok and lo_max < t < hi_min:
                return t, f"{t:.{d}f}"
    d = max(base, 4)
    return mid, f"{mid:.{d}f}"


def binary_quality(groups: List[np.ndarray]) -> float:
    """How far a present/absent character can be trusted to split these species.

    A measured character is graded by its gap in pooled SDs. That cannot grade a state: a
    character fixed in every specimen has no variance, the ratio runs to its cap, and the
    builder would prefer a state seen three times to a measurement separated by four SDs.
    What limits a state is the length of the series it was seen in. If all n specimens of a
    species share it, the other state may still occur in up to about 3/n of the species (the
    rule of three, the 95% bound when none was observed in n), so the chance that a further
    specimen keys correctly is at least 1 - 3/n. The quality is the mean of that bound over
    the species being split, zero where the series is too short to say anything (n <= 3).
    This is the counterpart, for states, of the 2/(n+1) rule for observed ranges."""
    return float(np.mean([max(0.0, 1.0 - 3.0 / len(g)) for g in groups])) if groups else 0.0


def pooled_sd(groups: List[np.ndarray]) -> float:
    vs = [np.var(g, ddof=1) for g in groups if len(g) >= 2]
    if vs:
        return float(math.sqrt(np.mean(vs)))
    allv = np.concatenate(groups)
    return float(0.05 * (allv.max() - allv.min())) if len(allv) else 0.0


class KeyBuilder:
    def __init__(self, M: Matrix, profile: Dict, max_secondary: int = 2,
                 min_n_primary: int = 2):
        self.M = M
        self.profile = profile
        kp = profile.get("key", {})
        self.type_w = dict(DEFAULT_TYPE_WEIGHTS, **(kp.get("feature_type_weights") or {}))
        self.max_secondary = int(kp.get("max_secondary_characters", max_secondary))
        self.min_n = int(kp.get("min_specimens_primary", min_n_primary))

    # -- feature-level ---------------------------------------------------------
    def weight(self, f: Feature) -> float:
        return self.type_w.get(f.family, 0.5) * PRIORITY_WEIGHT.get(f.priority, 0.7)

    def splits_for(self, fid: str, S: List[str], imperfect: bool = False) -> List[Split]:
        """All cuts of feature fid over species S (perfect ones; overlapping ones
        too when imperfect=True)."""
        f = self.M.features[fid]
        d = self.M.obs.get(fid, {})
        if any(sp not in d for sp in S):
            return []
        groups = [d[sp] for sp in S]
        med = [float(np.median(g)) for g in groups]
        order = np.argsort(med, kind="mergesort")
        sd = pooled_sd(groups) or 1e-9
        res = RESOLUTION.get(f.unit, 0.0)
        nmin = min(len(g) for g in groups)
        nfac = min(1.0, nmin / max(1, self.min_n))
        cov = sum(len(g) for g in groups) / max(1, sum(self.M.n_spec[sp] for sp in S))
        w = self.weight(f) * cov
        m = len(S)
        out = []
        los = [float(groups[i].min()) for i in order]
        his = [float(groups[i].max()) for i in order]
        pre_max = np.maximum.accumulate(his)
        suf_min = np.minimum.accumulate(los[::-1])[::-1]
        for k in range(1, m):
            gap = float(suf_min[k] - pre_max[k - 1])
            bal = math.sqrt(min(k, m - k) / (m / 2.0))
            low = tuple(S[i] for i in order[:k])
            high = tuple(S[i] for i in order[k:])
            if gap > 0 and gap >= res and gap >= 0.25 * sd:
                t, tt = nice_threshold(float(pre_max[k - 1]), float(suf_min[k]), f.unit)
                q = min(gap / sd, 4.0) / 4.0
                if fid in self.M.binary:
                    q = binary_quality(groups)
                score = w * (0.25 + 0.75 * q) * bal * nfac
                out.append(Split(fid, low, high, t, tt, True, 1.0, gap, score,
                                 (min(los[:k]), float(pre_max[k - 1])),
                                 (float(suf_min[k]), max(his[k:]))))
            elif imperfect:
                s_ = self._overlap_split(fid, f, low, high,
                                         [groups[i] for i in order[:k]],
                                         [groups[i] for i in order[k:]],
                                         sd, res, 0.85, w * bal * nfac)
                if s_ is not None:
                    out.append(s_)
        return out

    def _overlap_split(self, fid, f, low, high, lgroups, hgroups, sd, res,
                       min_bal_acc, weight) -> Optional[Split]:
        """Best threshold for an overlapping split, judged by BALANCED accuracy
        (mean of the two sides' accuracies) so a large group cannot hide a
        useless threshold. Requires the group medians to differ by at least
        2x measurement resolution and 0.5 pooled SD."""
        a = np.concatenate(lgroups)
        b = np.concatenate(hgroups)
        if abs(float(np.median(b)) - float(np.median(a))) < max(2 * res, 0.5 * sd):
            return None
        best = None
        for aa, bb, lo_side, hi_side in ((a, b, low, high), (b, a, high, low)):
            if np.median(bb) <= np.median(aa):
                continue
            vals = np.concatenate([aa, bb])
            lab = np.concatenate([np.zeros(len(aa)), np.ones(len(bb))])
            idx = np.argsort(vals, kind="mergesort")
            sv, sl = vals[idx], lab[idx]
            err_lo = (len(aa) - np.cumsum(1 - sl)) / len(aa)      # low values above t
            err_hi = np.cumsum(sl) / len(bb)                      # high values <= t
            bacc = 1.0 - 0.5 * (err_lo + err_hi)
            valid = np.append(sv[1:] > sv[:-1], True)
            bacc = np.where(valid, bacc, -1)
            i = int(np.argmax(bacc))
            acc = float(bacc[i])
            if acc < min_bal_acc:
                continue
            c = float(sv[i])
            nxt = float(sv[i + 1]) if i + 1 < len(sv) else c
            t, tt = nice_threshold(c, nxt, f.unit) if nxt > c else (c, pol.fmt(c, f.unit))
            cand = Split(fid, lo_side, hi_side, t, tt, False, acc, float(bb.min() - aa.max()),
                         weight * 0.25 * acc ** 8,
                         (float(aa.min()), float(aa.max())), (float(bb.min()), float(bb.max())))
            if best is None or cand.accuracy > best.accuracy:
                best = cand
        return best

    def best_split(self, fid: str, S: List[str]) -> Optional[Split]:
        sp = self.splits_for(fid, S)
        return max(sp, key=lambda s: (s.score, s.perfect)) if sp else None

    def separates(self, fid: str, low: Tuple[str, ...], high: Tuple[str, ...],
                  min_accuracy: Optional[float] = None) -> Optional[Split]:
        """Does feature fid separate the given partition (either direction)?
        Perfect separation by default; with min_accuracy, the best overlapping
        threshold is returned if its specimen accuracy reaches min_accuracy."""
        f = self.M.features[fid]
        d = self.M.obs.get(fid, {})
        if any(s not in d for s in low + high):
            return None
        lv = np.concatenate([d[s] for s in low])
        hv = np.concatenate([d[s] for s in high])
        sd = pooled_sd([d[s] for s in low + high]) or 1e-9
        res = RESOLUTION.get(f.unit, 0.0)
        nmin = min(len(d[s]) for s in low + high)
        nfac = min(1.0, nmin / max(1, self.min_n))
        cov = (len(lv) + len(hv)) / max(1, sum(self.M.n_spec[s] for s in low + high))
        w = self.weight(f) * cov
        for a, b, lo_side, hi_side in ((lv, hv, low, high), (hv, lv, high, low)):
            gap = float(b.min() - a.max())
            if gap > 0 and gap >= res and gap >= 0.25 * sd:
                t, tt = nice_threshold(float(a.max()), float(b.min()), f.unit)
                q = min(gap / sd, 4.0) / 4.0
                if fid in self.M.binary:
                    q = binary_quality([d[s] for s in low + high])
                return Split(fid, lo_side, hi_side, t, tt, True, 1.0, gap,
                             w * (0.25 + 0.75 * q) * nfac,
                             (float(a.min()), float(a.max())), (float(b.min()), float(b.max())))
        if min_accuracy is None:
            return None
        return self._overlap_split(fid, f, low, high, [d[x] for x in low], [d[x] for x in high],
                                   sd, res, min_accuracy, w * nfac)

    # -- node-level ---------------------------------------------------------
    def choose(self, S: List[str]) -> Dict:
        feats = self.M.features
        both, male, female = [], [], []
        for fid, f in feats.items():
            for s in self.splits_for(fid, S):
                (male if f.sex == "male" else female if f.sex == "female" else both).append(s)
        if both:
            best = max(both, key=lambda s: (s.score, s.fid))
            return {"primary": [best], "kind": "perfect"}
        # no clean split on a both-sex character: also consider overlapping cuts
        for fid, f in feats.items():
            if f.sex not in ("male", "female"):
                both.extend(s for s in self.splits_for(fid, S, imperfect=True) if not s.perfect)
        # male + female pair separating the same partition
        best_m = {}
        for s in (x for x in male if x.perfect):
            p = s.partition() if min(s.low, key=natural_key) < min(s.high, key=natural_key) else frozenset(s.high)
            if p not in best_m or s.score > best_m[p].score:
                best_m[p] = s
        pair, pair_score = None, -1
        for s in (x for x in female if x.perfect):
            p = s.partition() if min(s.low, key=natural_key) < min(s.high, key=natural_key) else frozenset(s.high)
            if p in best_m:
                sc = 0.9 * (s.score + best_m[p].score) / 2
                if sc > pair_score:
                    pair, pair_score = (best_m[p], s), sc
        imperfect = max(both, key=lambda s: (s.score, s.fid)) if both else None
        single = max([s for s in male + female if s.perfect], key=lambda s: (s.score, s.fid), default=None)
        options = []
        if pair:
            options.append((pair_score, {"primary": list(pair), "kind": "paired_sexes"}))
        if imperfect:
            options.append((imperfect.score, {"primary": [imperfect], "kind": "overlap"}))
        if single:
            options.append((single.score * 0.4, {"primary": [single], "kind": "single_sex"}))
        if not options:
            return {}
        return max(options, key=lambda o: o[0])[1]

    def secondaries(self, primary: List[Split]) -> List[Split]:
        p0 = primary[0]
        low, high = p0.low, p0.high
        used_fids = {p.fid for p in primary}
        used_equiv = {self.M.features[p.fid].equiv for p in primary}
        used_groups = {(self.M.features[p.fid].category, self.M.features[p.fid].family) for p in primary}
        cands = []
        for fid, f in self.M.features.items():
            if fid in used_fids:
                continue
            s = self.separates(fid, low, high, min_accuracy=0.9)
            if s:
                cands.append(s)
        # perfect characters first, then overlapping ones; both-sex before single-sex
        cands.sort(key=lambda s: (not s.perfect,
                                  self.M.features[s.fid].sex in ("male", "female"),
                                  -s.score, s.fid))
        out = []
        prim_meds = self._medians(p0.fid, low + high)
        for s in cands:
            f = self.M.features[s.fid]
            grp = (f.category, f.family)
            if grp in used_groups or f.equiv in used_equiv:
                continue
            med = self._medians(s.fid, low + high)
            if prim_meds is not None and med is not None and len(med) > 2:
                r = np.corrcoef(prim_meds, med)[0, 1]
                if abs(r) > 0.95:
                    continue
            out.append(s)
            used_groups.add(grp)
            used_equiv.add(f.equiv)
            if len(out) >= self.max_secondary:
                break
        return out

    def _medians(self, fid, S):
        d = self.M.obs.get(fid, {})
        if any(s not in d for s in S):
            return None
        return np.array([np.median(d[s]) for s in S])

    def build(self) -> List[Dict]:
        couplets: List[Dict] = []

        def grow(S: List[str]) -> object:
            if len(S) == 1:
                return S[0]
            num = len(couplets) + 1
            node = {"number": num}
            couplets.append(node)
            ch = self.choose(S)
            if not ch:
                # no usable character: split alphabetically, flagged
                h = len(S) // 2
                node.update({"kind": "unresolved", "characters": [],
                             "A_species": S[:h], "B_species": S[h:]})
            else:
                prim = ch["primary"]
                sec = self.secondaries(prim) if ch["kind"] != "unresolved" else []
                p0 = prim[0]
                chars = []
                for role, s in [("primary", x) for x in prim] + [("secondary", x) for x in sec]:
                    # orient every character to the primary's lead A = p0.low
                    a_is_low = set(s.low) == set(p0.low)
                    chars.append({"role": role, "split": s, "A_is_low": a_is_low})
                node.update({"kind": ch["kind"], "characters": chars,
                             "A_species": sorted(p0.low, key=natural_key),
                             "B_species": sorted(p0.high, key=natural_key)})
            node["A_goto"] = grow(node["A_species"])
            node["B_goto"] = grow(node["B_species"])
            return num

        grow(list(self.M.species))
        return couplets


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def char_record(c: Dict, M: Matrix) -> Dict:
    s: Split = c["split"]
    f = M.features[s.fid]
    a_low = c["A_is_low"]
    rec = {
        "role": c["role"], "feature_id": s.fid, "label": f.label, "definition": f.definition,
        "unit": f.unit, "family": f.family, "column": f.column, "structure_sex": f.sex,
        "threshold": s.threshold, "threshold_text": s.threshold_text,
        "A_operator": "<=" if a_low else ">", "B_operator": ">" if a_low else "<=",
        "A_range": s.low_range if a_low else s.high_range,
        "B_range": s.high_range if a_low else s.low_range,
        "perfect_separation": s.perfect, "balanced_accuracy": round(s.accuracy, 4),
        "binary": s.fid in M.binary,
        "A_n": side_n(M, s.fid, s.low if a_low else s.high),
        "B_n": side_n(M, s.fid, s.high if a_low else s.low),
        "gap": s.gap,
    }
    if f.family == "colour" and f.column == "cie_L":
        rec["A_colour_term"] = group_colour(M, f.category, s.low if a_low else s.high)
        rec["B_colour_term"] = group_colour(M, f.category, s.high if a_low else s.low)
    elif f.family == "colour" and f.column in ("cie_L_darkest_third", "cie_L_palest_third"):
        rec["A_colour_term"] = pol.lightness_term(float(np.mean(rec["A_range"])))
        rec["B_colour_term"] = pol.lightness_term(float(np.mean(rec["B_range"])))
    return rec


def side_n(M: Matrix, fid: str, species) -> int:
    d = M.obs.get(fid, {})
    return int(sum(len(d[s]) for s in species if s in d))


def group_colour(M: Matrix, category: str, species) -> str:
    vals = {}
    for ch in ("L", "a", "b"):
        fid = pol.feature_id(category, f"cie_{ch}")
        d = M.obs.get(fid, {})
        arr = [d[s] for s in species if s in d]
        if not arr:
            return ""
        vals[ch] = float(np.median(np.concatenate(arr)))
    return pol.colour_name(vals["L"], vals["a"], vals["b"])


OP_TXT = {"<=": "≤", ">": ">"}


def template_lead(chars: List[Dict], side: str) -> str:
    parts = []
    for c in chars:
        op = OP_TXT[c[f"{side}_operator"]]
        lo, hi = c[f"{side}_range"]
        unit = c["unit"]
        u = "" if unit == "x" else (" mm" if unit == "mm" else f" {unit}" if unit not in ("L*", "a*", "b*", "C*", "deg") else "")
        pre = (SEX_SYMBOL[c["structure_sex"]] + " ") if c["structure_sex"] in SEX_SYMBOL else ""
        lab = c["label"]
        def span(fmt_lo, fmt_hi, suffix=""):
            return f"{fmt_lo}{suffix}" if fmt_lo == fmt_hi else f"{fmt_lo}–{fmt_hi}{suffix}"
        if c.get("binary"):
            # a state, not a measurement: say which, and in how many specimens it was seen
            state = "present" if c[f"{side}_operator"] == ">" else "absent"
            usually = "" if c["perfect_separation"] else "usually "
            parts.append(f"{pre}{lab}: {usually}{state} (n = {c[f'{side}_n']})")
            continue
        if unit in ("L*", "a*", "b*", "C*"):
            val = f"{unit} {op} {c['threshold_text']}"
            rng = f"{unit} " + span(pol.fmt(lo, unit), pol.fmt(hi, unit))
        elif unit == "deg":
            val = f"{op} {c['threshold_text']}°"
            rng = span(pol.fmt(lo, unit), pol.fmt(hi, unit), "°")
        else:
            xs = "×" if unit == "x" else ""
            val = f"{op} {c['threshold_text']}{xs}{u}"
            rng = span(pol.fmt(lo, unit, c['threshold']), pol.fmt(hi, unit, c['threshold']), f"{xs}{u}")
        rng += f"; n = {c[f'{side}_n']}"
        term = c.get(f"{side}_colour_term")
        term_txt = f"; ≈ {term}" if term else ""
        usually = "" if c["perfect_separation"] else "usually "
        parts.append(f"{pre}{usually}{lab} {val} (observed {rng}{term_txt})")
    return "; ".join(parts)


def allowed_numbers(chars: List[Dict]) -> List[float]:
    nums = []
    for c in chars:
        nums.append(c["threshold"])
        for side in ("A", "B"):
            nums.extend(c[f"{side}_range"])
            nums.append(float(c.get(f"{side}_n", 0)))
        nums.extend(float(x) for x in re.findall(r'\d+(?:\.\d+)?', c["label"]))
        nums.extend(float(x) for x in re.findall(r'\d+(?:\.\d+)?', c.get("definition", "")))
    return nums


DIRECTION_WORDS = {
    # (family or column, operator) -> words that contradict the operator
    ("aspect_ratio", "<="): r"\b(slender|elongate|narrow|narrower)\b",
    ("aspect_ratio", ">"): r"\b(stout|broad|broader|squat)\b",
    ("meas_length_mm", "<="): r"\b(long|longer|large|larger)\b",
    ("meas_length_mm", ">"): r"\b(short|shorter|small|smaller)\b",
    ("lmkmm", "<="): r"\b(long|longer|large|larger)\b",
    ("lmkmm", ">"): r"\b(short|shorter|small|smaller)\b",
    ("meas_height_mm", "<="): r"\b(wide|wider|broad|broader)\b",
    ("meas_height_mm", ">"): r"\b(narrow|narrower|slim|slimmer)\b",
    ("colour", "<="): r"\b(paler|lighter)\b",
    ("colour", ">"): r"\b(darker)\b",
    ("cie_L_end_contrast", "<="): r"\b(marked|pronounced|strong|strongly|contrasting|contrastingly)\b",
    ("cie_L_end_contrast", ">"): r"\b(slight|little|weak|weakly|concolorous|uniform|uniformly)\b",
}


def _direction_problems(text: str, c: Dict, side: str) -> List[str]:
    """Check descriptive words in the clause that carries this character's threshold."""
    op = c[f"{side}_operator"]
    col = c.get("column") or ""
    keys = [(col, op)] if (col, op) in DIRECTION_WORDS else []
    if not keys:
        fam = "lmkmm" if col.startswith("lmkmm_") else c["family"]
        keys = [(fam, op)] if (fam, op) in DIRECTION_WORDS else []
    if not keys:
        return []
    d = len(c["threshold_text"].split(".")[1]) if "." in c["threshold_text"] else 0
    out = []
    for clause in re.split(r';', text):
        vals = [float(n) for n in pol.numbers_in(clause)]
        if any(round(v, d) == round(c["threshold"], d) for v in vals):
            for m in re.finditer(DIRECTION_WORDS[keys[0]], clause, re.I):
                before = clause[max(0, m.start() - 25):m.start()].lower()
                if re.search(r"\b(not|no|without|hardly|never)\b(\s+\w+){0,2}\s*$", before):
                    continue
                out.append(f"word '{m.group(0)}' contradicts {op} for {c['label']}")
                break
            break
    return out


GLOSSARY_ABBR: set = set()


def check_wording(text: str, chars: List[Dict], species_tokens: List[str], side: str = "A") -> List[str]:
    problems = []
    allowed_abbr = {"CIE", "L"} | GLOSSARY_ABBR
    for c in chars:
        allowed_abbr |= set(re.findall(r'\b[A-Z]{2,}[0-9]*\b', c["label"] + " " + c.get("definition", "")))
        problems += _direction_problems(text, c, side)
    for tok in set(re.findall(r'\b[A-Z]{2,}[0-9]*\b', text)) - allowed_abbr:
        problems.append(f"abbreviation not in the character labels: {tok}")
    allowed = allowed_numbers(chars)
    for n in pol.numbers_in(text):
        if not pol.number_is_allowed(n, allowed):
            problems.append(f"number not in couplet data: {n}")
    found = [float(n) for n in pol.numbers_in(text)]
    for c in chars:
        d = len(c["threshold_text"].split(".")[1]) if "." in c["threshold_text"] else 0
        if not any(round(v, d) == round(c["threshold"], d) and abs(v - c["threshold"]) < 10 ** (-d)
                   for v in found):
            problems.append(f"threshold {c['threshold_text']} missing")
    t2 = pol.find_tier2_terms(text)
    if t2:
        problems.append(f"tier-2 terms: {t2}")
    for tok in species_tokens:
        if re.search(rf'(?<![\w.]){re.escape(tok)}(?![\w])', text):
            problems.append(f"species name in lead: {tok}")
    return problems


def word_couplets(recs: List[Dict], client, prompts, profile, model, workers, log):
    ctx = pol.build_taxon_context(profile)
    GLOSSARY_ABBR.update(profile.get("measurement_abbreviations", {}).keys())
    for rd in profile.get("ratios", []):
        GLOSSARY_ABBR.update(re.findall(r'[A-Z]{2,}[0-9]*', rd.get("id", "")))
    system = prompts.get("taxonomist_persona") + "\n\n" + prompts.get("key_couplet_wording", taxon_context=ctx)
    species_tokens = []
    for code in profile.get("species", {}):
        species_tokens.append(code)
        species_tokens.append(pol.species_display_name(code, profile))

    def one(rec):
        if not rec["characters"]:
            return rec["number"], None, ["no characters"]
        payload = {"couplet": rec["number"], "characters": [
            {k: c[k] for k in ("label", "definition", "unit", "structure_sex", "threshold_text",
                               "A_operator", "B_operator")} |
            {"A_observed_range": [pol.fmt(x, c["unit"], c["threshold"]) for x in c["A_range"]],
             "B_observed_range": [pol.fmt(x, c["unit"], c["threshold"]) for x in c["B_range"]],
             "A_n_specimens": c["A_n"], "B_n_specimens": c["B_n"],
             "usually": not c["perfect_separation"],
             "A_colour_term": c.get("A_colour_term", ""), "B_colour_term": c.get("B_colour_term", "")}
            for c in rec["characters"]],
            "template_lead_a": rec["A_template"], "template_lead_b": rec["B_template"]}
        msg = ("Word this couplet. Use ≤ and > as given. Male-only characters start with ♂, "
               "female-only with ♀. Keep each observed range and its n (write a single value "
               "when min = max). Characters with \"usually\": true overlap between the leads; "
               "start them with 'usually'.\n" + json.dumps(payload, ensure_ascii=False, indent=1))
        ab, probs = None, []
        for attempt in (1, 2):
            try:
                r = client.messages.create(model=model, max_tokens=1500, system=system,
                                           messages=[{"role": "user", "content": msg}])
                out = parse_json_response(r.content[0].text)
                a, b = out.get("lead_a", ""), out.get("lead_b", "")
                probs = check_wording(a, rec["characters"], species_tokens, "A") + \
                    check_wording(b, rec["characters"], species_tokens, "B")
                ab = (a, b)
            except Exception as e:  # noqa: BLE001
                ab, probs = None, [f"LLM error: {e}"]
            if ab and not probs:
                break
            msg = (msg + "\n\nYOUR PREVIOUS ANSWER WAS REJECTED by the automatic checker:\n- "
                   + "\n- ".join(dict.fromkeys(probs)) +
                   "\nReturn corrected JSON {\"lead_a\": ..., \"lead_b\": ...} that fixes every problem.")
        return rec["number"], ab, probs

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for num, ab, probs in ex.map(one, recs):
            results[num] = (ab, probs)
            log(f"  couplet {num}: {'wording accepted' if ab and not probs else 'template kept: ' + '; '.join(probs)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def paths_to_terminals(recs: List[Dict]) -> Dict[str, List[int]]:
    by_num = {r["number"]: r for r in recs}
    out: Dict[str, List[int]] = defaultdict(list)

    def walk(n, path, seen):
        if n in seen:
            return
        r = by_num[n]
        for side in ("A", "B"):
            g = r[f"{side}_goto"]
            if isinstance(g, int):
                walk(g, path + [n], seen | {n})
            else:
                out[g].append(len(path) + 1)
    walk(1, [], set())
    return out


def e_dicho(steps: Dict[str, int]) -> Optional[float]:
    S = len(steps)
    if S <= 1:
        return 1.0
    tot = sum(steps.values())
    H = -sum((s / tot) * math.log(s / tot) for s in steps.values() if s > 0)
    return round(H / math.log(S), 4)


# Hewlett-Packard d2 constants: E[range] = d2(n) * sigma for a normal sample of size n.
# Used to turn an observed range back into a spread, so a range measured on few specimens
# can be widened into an interval a NEW specimen is actually expected to fall inside.
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970,
       10: 3.078, 11: 3.173, 12: 3.258, 13: 3.336, 14: 3.407, 15: 3.472}


def tolerance_range(rng, n, coverage: float = 0.95):
    """Widen an observed min/max into a prediction interval for one further specimen.

    The observed range of n specimens is not the species' range: a new draw falls outside it
    with probability about 2/(n+1) no matter how good the character is, so at n = 5 a third of
    perfectly ordinary conspecifics look 'outside'. Treating min/max as the species limit turns
    a novelty test into a sample-size test. Here the range is converted back to a spread via the
    d2 constant and re-expanded as mean +/- t * s * sqrt(1 + 1/n).
    """
    if not rng or n is None or n < 2:
        return (float("-inf"), float("inf")) if rng else rng
    lo, hi = float(min(rng)), float(max(rng))
    if hi <= lo:
        return (lo, hi)
    n = int(n)
    d2 = _D2.get(min(n, 15), 3.472 + 0.03 * (n - 15))
    sigma = (hi - lo) / d2
    mid = 0.5 * (lo + hi)
    try:
        from scipy.stats import t as _t
        crit = float(_t.ppf(0.5 + coverage / 2, max(1, n - 1)))
    except Exception:
        crit = 2.5
    half = crit * sigma * math.sqrt(1.0 + 1.0 / n)
    # never tighter than what was actually seen
    return (min(lo, mid - half), max(hi, mid + half))


def conflicts_on_path(values: Dict[str, float], sex: str, recs: List[Dict],
                      _path=None, coverage: float = 0.95) -> Dict:
    """Walk the key exactly as biorag_novelty_score_v1.key_path does — the leading character
    of each couplet, path stops if that character was not measured — and count the couplets
    where the value fell outside the observed range of the side it was sent down.

    It deliberately does NOT reuse identify(), which tries each character in turn and skips
    the ones that are missing: that is the right rule for naming a specimen but a different
    traversal, and a false-alarm rate counted one way cannot be compared with a detection rate
    counted the other. Both arms of the calibration must walk the key the same way.
    """
    by_no = {r["number"]: r for r in recs}
    node, conflicts, tol_conflicts, seen = recs[0]["number"], 0, 0, set()
    exc = []          # per couplet: how far outside the range, as a fraction of its width
    while node in by_no and node not in seen:
        seen.add(node)
        c = by_no[node]
        if not c.get("characters"):
            return {"conflicts": conflicts, "tol_conflicts": tol_conflicts, "resolved": False,
                    "excursions": exc}
        ch = c["characters"][0]
        v = values.get(ch["feature_id"])
        if v is None or v != v:
            return {"conflicts": conflicts, "tol_conflicts": tol_conflicts, "resolved": False,
                    "excursions": exc}
        is_A = (v <= ch["threshold"]) if ch["A_operator"] == "<=" else (v > ch["threshold"])
        rng = ch["A_range"] if is_A else ch["B_range"]
        if rng:
            lo_, hi_ = min(rng), max(rng)
            conflicts += int(v < lo_ or v > hi_)
            w = (hi_ - lo_) or abs(hi_) or 1.0
            exc.append(round(max(0.0, lo_ - v, v - hi_) / w, 4))
            tl, th = tolerance_range(rng, ch.get("A_n" if is_A else "B_n"), coverage)
            tol_conflicts += int(v < tl or v > th)
        nxt = c["A_goto" if is_A else "B_goto"]
        if nxt in by_no:
            node = nxt
        else:
            return {"conflicts": conflicts, "tol_conflicts": tol_conflicts, "resolved": True,
                    "excursions": exc}
    return {"conflicts": conflicts, "tol_conflicts": tol_conflicts, "resolved": False,
                    "excursions": exc}


def identify(values: Dict[str, float], sex: str, recs: List[Dict], vote: bool = False):
    by_num = {r["number"]: r for r in recs}
    n, path = 1, []
    while True:
        r = by_num[n]
        decisions = []
        for c in r["characters"]:
            if c["structure_sex"] in ("male", "female") and sex != c["structure_sex"]:
                continue
            v = values.get(c["feature_id"])
            if v is None:
                continue
            is_A = (v <= c["threshold"]) if c["A_operator"] == "<=" else (v > c["threshold"])
            decisions.append("A" if is_A else "B")
            if not vote:
                break
        if not decisions:
            return None, path + [n], "unresolved"
        side = Counter(decisions).most_common(1)[0][0]
        path.append(n)
        g = r[f"{side}_goto"]
        if not isinstance(g, int):
            return g, path, "ok"
        n = g


def recheck_thresholds(recs: List[Dict], M: Matrix) -> List[Dict]:
    issues = []
    for r in recs:
        for c in r["characters"]:
            d = M.obs.get(c["feature_id"], {})
            for side in ("A", "B"):
                spp = r[f"{side}_species"]
                vals = np.concatenate([d[s] for s in spp if s in d]) if any(s in d for s in spp) else np.array([])
                if c[f"{side}_operator"] == "<=":
                    bad = vals[vals > c["threshold"]]
                else:
                    bad = vals[vals <= c["threshold"]]
                if len(bad) and c["perfect_separation"]:
                    issues.append({"couplet": r["number"], "feature_id": c["feature_id"],
                                   "side": side, "n_violations": int(len(bad))})
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Exporters
# ─────────────────────────────────────────────────────────────────────────────

def dest_text(g, profile):
    return str(g) if isinstance(g, int) else pol.species_display_name(g, profile)


def write_text_key(recs, profile, path_txt, path_md, title):
    lines = [title, "=" * len(title), ""]
    md = [f"# {title}", ""]
    for r in recs:
        a, b = r["A_text"], r["B_text"]
        da, db = dest_text(r["A_goto"], profile), dest_text(r["B_goto"], profile)
        flag = {"overlap": "  [ranges overlap slightly: use all listed characters]",
                "single_sex": "  [separating characters known for one sex only]",
                "unresolved": "  [no measured character separates these species]"}.get(r["kind"], "")
        lines.append(f"{r['number']}{flag}")
        lines.append(f"   {r['number']}a  {a} .......... {da}")
        lines.append(f"   {r['number']}b  {b} .......... {db}")
        lines.append("")
        ita = lambda g, d: d if isinstance(g, int) else f"*{d}*"  # noqa: E731
        md.append(f"**{r['number']}a.** {a} … {ita(r['A_goto'], da)}  ")
        md.append(f"**{r['number']}b.** {b} … {ita(r['B_goto'], db)}")
        md.append("")
    Path(path_txt).write_text("\n".join(lines), encoding="utf-8")
    Path(path_md).write_text("\n".join(md), encoding="utf-8")


def write_jsonld(recs, profile, M, path, meta):
    genus = profile.get("taxon", {}).get("genus", "")
    doc = {
        "@context": {"@vocab": "https://schema.org/", "dsc": "https://descriptron.org/ontology/",
                     "dwc": "http://rs.tdwg.org/dwc/terms/"},
        "@type": ["DefinedTermSet", "dsc:DichotomousKey"],
        "name": meta["title"], "dateCreated": meta["generated"],
        "dsc:generator": f"biorag_key_builder_v1 {BUILDER_VERSION}",
        "dsc:dataMatrix": meta["matrix_dir"],
        "dsc:taxa": [{"@type": "Taxon", "@id": f"#taxon-{code}", "identifier": code,
                      "name": pol.species_display_name(code, profile),
                      "taxonRank": "species", "parentTaxon": genus,
                      "dsc:status": pol.species_status(code, profile)} for code in M.species],
        "dsc:couplets": [],
    }
    for r in recs:
        leads = []
        for side in ("A", "B"):
            g = r[f"{side}_goto"]
            lead = {"@type": "dsc:Lead", "@id": f"#lead-{r['number']}{side.lower()}",
                    "text": r[f"{side}_text"],
                    "dsc:conditions": [{"dsc:featureId": c["feature_id"], "dsc:label": c["label"],
                                        "dsc:operator": c[f"{side}_operator"],
                                        "dsc:threshold": c["threshold"], "unitText": c["unit"],
                                        "dsc:observedRange": list(c[f"{side}_range"]),
                                        "dsc:structureSex": c["structure_sex"],
                                        "dsc:role": c["role"]} for c in r["characters"]],
                    "dsc:species": r[f"{side}_species"]}
            if isinstance(g, int):
                lead["dsc:goToCouplet"] = {"@id": f"#couplet-{g}"}
            else:
                lead["dsc:identifiesTaxon"] = {"@id": f"#taxon-{g}"}
            leads.append(lead)
        doc["dsc:couplets"].append({"@type": "dsc:Couplet", "@id": f"#couplet-{r['number']}",
                                    "position": r["number"], "dsc:separation": r["kind"],
                                    "dsc:leads": leads})
    Path(path).write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")


def write_sdd(recs, profile, M, path, meta, schema_dir: Optional[Path]):
    from lxml import etree
    NS = "http://rs.tdwg.org/UBIF/2006/"
    XSI = "http://www.w3.org/2001/XMLSchema-instance"
    E = lambda tag, parent=None, **attrs: (etree.SubElement(parent, f"{{{NS}}}{tag}", **attrs)  # noqa: E731
                                           if parent is not None else etree.Element(f"{{{NS}}}{tag}", nsmap={None: NS, "xsi": XSI}, **attrs))

    def rep(parent, label, detail=None):
        r = E("Representation", parent)
        E("Label", r).text = label
        if detail:
            E("Detail", r).text = detail
        return r

    root = E("Datasets")
    root.set(f"{{{XSI}}}schemaLocation", f"{NS} http://rs.tdwg.org/UBIF/2006/Schema/1.1/SDD.xsd")
    tm = E("TechnicalMetadata", root, created=meta["generated"][:19])
    E("Generator", tm, name="Descriptron BioRAG key builder", version=BUILDER_VERSION)
    ds = E("Dataset", root)
    ds.set("{http://www.w3.org/XML/1998/namespace}lang", "en")
    rep(ds, meta["title"], "Characters are Tier-1 (hand-measurable) features of the Descriptron "
                           "data matrix; summary data are per-species specimen statistics.")
    tn = E("TaxonNames", ds)
    for code in M.species:
        t = E("TaxonName", tn, id=f"t_{safe_id(code)}")
        rep(t, pol.species_display_name(code, profile))
    used = []
    for r in recs:
        for c in r["characters"]:
            if c["feature_id"] not in used:
                used.append(c["feature_id"])
    chars = E("Characters", ds)
    cid = {}
    for i, fid in enumerate(used, 1):
        f = M.features[fid]
        cid[fid] = f"c{i}"
        qc = E("QuantitativeCharacter", chars, id=f"c{i}")
        rep(qc, f.label, f"{f.definition} [feature_id {fid}]")
        if f.unit:
            mu = E("MeasurementUnit", qc)
            E("Label", mu, role="Abbrev").text = f.unit
    cds = E("CodedDescriptions", ds)
    for code in M.species:
        cd = E("CodedDescription", cds, id=f"d_{safe_id(code)}")
        rep(cd, pol.species_display_name(code, profile))
        sc = E("Scope", cd)
        E("TaxonName", sc, ref=f"t_{safe_id(code)}")
        sd = E("SummaryData", cd)
        for fid in used:
            arr = M.obs.get(fid, {}).get(code)
            if arr is None or not len(arr):
                continue
            q = E("Quantitative", sd, ref=cid[fid])
            for typ, val in (("Min", arr.min()), ("Max", arr.max()), ("Mean", arr.mean()),
                             ("SD", arr.std(ddof=1) if len(arr) > 1 else None), ("N", len(arr))):
                if val is not None:
                    E("Measure", q, type=typ, value=repr(float(val)) if typ != "N" else str(int(val)))
    iks = E("IdentificationKeys", ds)
    ik = E("IdentificationKey", iks, id="key1")
    rep(ik, meta["title"])
    leads = E("Leads", ik)
    for r in recs:
        for side in ("A", "B"):
            lid = f"L{r['number']}{side.lower()}"
            ld = E("Lead", leads, id=lid)
            if r["number"] != 1:
                parent = next(x for x in recs for s in ("A", "B") if x[f"{s}_goto"] == r["number"])
                ps = "A" if parent["A_goto"] == r["number"] else "B"
                E("Parent", ld, ref=f"L{parent['number']}{ps.lower()}")
            E("Statement", ld).text = r[f"{side}_text"]
            g = r[f"{side}_goto"]
            if not isinstance(g, int):
                E("TaxonName", ld, ref=f"t_{safe_id(g)}")
    tree = etree.ElementTree(root)
    tree.write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True)
    if schema_dir and (schema_dir / "SDD.xsd").exists():
        schema = etree.XMLSchema(etree.parse(str(schema_dir / "SDD.xsd")))
        ok = schema.validate(etree.parse(str(path)))
        return ok, [str(e) for e in schema.error_log][:20]
    return None, ["schema not found — not validated"]


def safe_id(s):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(s))


def write_matrix(recs, M, profile, path):
    used = []
    for r in recs:
        for c in r["characters"]:
            if c["feature_id"] not in used:
                used.append(c["feature_id"])
    rows = []
    for code in M.species:
        row = {"species_code": code, "species": pol.species_display_name(code, profile)}
        for fid in used:
            arr = M.obs.get(fid, {}).get(code)
            unit = M.features[fid].unit
            row[fid] = "" if arr is None else \
                f"{pol.fmt(arr.min(), unit, np.median(arr))}–{pol.fmt(arr.max(), unit, np.median(arr))} (n={len(arr)})"
        rows.append(row)
    df = pd.DataFrame(rows)
    hdr = pd.DataFrame([{"species_code": "label", "species": "",
                         **{fid: M.features[fid].label for fid in used}},
                        {"species_code": "unit", "species": "",
                         **{fid: M.features[fid].unit for fid in used}}])
    pd.concat([hdr, df]).to_csv(path, sep="\t", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_records(couplets, M):
    recs = []
    for node in couplets:
        chars = [char_record(c, M) for c in node["characters"]]
        rec = {"number": node["number"], "kind": node["kind"], "characters": chars,
               "A_species": node["A_species"], "B_species": node["B_species"],
               "A_goto": node["A_goto"], "B_goto": node["B_goto"]}
        rec["A_template"] = template_lead(chars, "A") if chars else "(no character available)"
        rec["B_template"] = template_lead(chars, "B") if chars else "(no character available)"
        rec["A_text"], rec["B_text"] = rec["A_template"], rec["B_template"]
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser(description="Data-driven dichotomous key builder (BioRAG v2)")
    ap.add_argument("--matrix_dir", required=True, help="Output of biorag_key_feature_filter_v2.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-loo", action="store_true", help="Skip leave-one-specimen-out test")
    ap.add_argument("--species", nargs="*", default=None, help="Restrict to these species codes")
    ap.add_argument("--exclude_specimens", nargs="*", default=None,
                    help="Build the key as if these specimens had never been collected. Used by the "
                         "evaluation scripts to score a specimen against a key that owes nothing to "
                         "it — withholding it from prediction alone is not a hold-out, because it "
                         "would still have helped define the ranges it is tested against")
    add_backend_args(ap)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "key_builder.log", "a")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    profile = pol.load_taxon_profile(args.taxon_profile)
    mdir = Path(args.matrix_dir)
    long = pd.read_csv(mdir / "specimen_matrix_long.csv")
    fdict = pd.read_csv(mdir / "feature_dictionary.tsv", sep="\t")
    if args.species:
        long = long[long["species"].isin(args.species)]
    if args.exclude_specimens:
        long = long[~long["specimen_id"].isin(set(args.exclude_specimens))]
    M = Matrix(long, fdict, profile)
    genus = profile.get("taxon", {}).get("genus", "")
    title = args.title or f"Key to the {len(M.species)} species of {genus} treated here"
    meta = {"title": title, "generated": datetime.now().isoformat(), "matrix_dir": str(mdir),
            "builder_version": BUILDER_VERSION, "policy_version": pol.POLICY_VERSION,
            "taxon_profile": profile.get("_path"), "llm_backend": args.llm_backend}
    log(f"[{meta['generated']}] building key: {len(M.species)} species, "
        f"{len(M.features)} key features, {len(M.values)} specimens")

    t0 = time.time()
    kb = KeyBuilder(M, profile)
    couplets = kb.build()
    recs = build_records(couplets, M)
    log(f"tree built in {time.time() - t0:.1f}s: {len(recs)} couplets; kinds "
        f"{dict(Counter(r['kind'] for r in recs))}")

    wording = {}
    if args.llm_backend != "none":
        prompts = load_prompt_library(args.system_prompts)
        client = make_llm_client(args.llm_backend, claude_bin=args.claude_bin,
                                 cc_model=args.cc_model,
                                 log_path=args.llm_log or str(out / "llm_calls.jsonl"))
        log(f"wording couplets with {args.llm_backend} ({args.model})")
        wording = word_couplets(recs, client, prompts, profile, args.model, args.workers, log)
        for r in recs:
            ab, probs = wording.get(r["number"], (None, ["not run"]))
            r["wording_problems"] = probs
            if ab and not probs:
                r["A_text"], r["B_text"] = ab
                r["wording_source"] = "llm"
            else:
                r["wording_source"] = "template"
                r["llm_rejected"] = ab
    else:
        for r in recs:
            r["wording_source"] = "template"

    # ---- validation
    steps = paths_to_terminals(recs)
    terminals = Counter({sp: len(v) for sp, v in steps.items()})
    missing = [s for s in M.species if s not in steps]
    multi = [s for s, n in terminals.items() if n > 1]
    step1 = {s: v[0] for s, v in steps.items()}
    thr_issues = recheck_thresholds(recs, M)
    t2 = [(r["number"], side, pol.find_tier2_terms(r[f"{side}_text"]))
          for r in recs for side in ("A", "B") if pol.find_tier2_terms(r[f"{side}_text"])]

    id_rows = []
    for sid, vals in M.values.items():
        sp = M.spec_species[sid]
        sex = M.spec_sex.get(sid, "unknown")
        got1, path1, st1 = identify(vals, sex, recs)
        gotv, _p, stv = identify(vals, sex, recs, vote=True)
        id_rows.append({"specimen_id": sid, "species": sp, "sex": sex,
                        "resub_first_char": got1 or st1, "resub_first_char_ok": got1 == sp,
                        "resub_vote": gotv or stv, "resub_vote_ok": gotv == sp,
                        "path": "-".join(map(str, path1))})
    loo_t0 = time.time()
    if not args.no_loo:
        log("leave-one-specimen-out: rebuilding the key once per specimen ...")
        for i, row in enumerate(id_rows):
            sid = row["specimen_id"]
            M.rebuild(exclude_specimen=sid)
            sp_left = {M.spec_species[s] for s in M.values if s != sid}
            if row["species"] not in sp_left:
                row["loo"] = "species has no other specimen"
                row["loo_ok"] = None
                continue
            kb2 = KeyBuilder(M, profile)
            recs2 = build_records(kb2.build(), M)
            got, _p, st = identify(M.values[sid], row["sex"], recs2)
            row["loo"] = got or st
            row["loo_ok"] = got == row["species"]
            # How often does a genuinely new specimen of a species already in the key fall outside
            # the range its branch was built from? Counted here, under the rebuild that excludes it,
            # because counting it on the published key asks a specimen whether it lies outside a
            # range it helped define. This is the honest false-alarm arm for the novelty rules.
            _cf = conflicts_on_path(M.values[sid], row["sex"], recs2)
            row["loo_conflicts"] = _cf["conflicts"]
            row["loo_conflicts_tol"] = _cf["tol_conflicts"]
            row["loo_excursions"] = ",".join(str(x) for x in _cf.get("excursions", []))
            row["loo_path_resolved"] = _cf["resolved"]
            if (i + 1) % 25 == 0:
                log(f"  LOO {i + 1}/{len(id_rows)} ({time.time() - loo_t0:.0f}s)")
        M.rebuild()
    idf = pd.DataFrame(id_rows).sort_values(["species", "specimen_id"], key=lambda s: s.map(natural_str))
    idf.to_csv(out / "identification_test.tsv", sep="\t", index=False)

    def rate(col):
        s = idf[col].dropna() if col in idf else pd.Series(dtype=bool)
        return None if not len(s) else round(float(s.astype(bool).mean()), 4)

    unresolved = int((idf["resub_first_char"] == "unresolved").sum())
    wording_ok = sum(1 for r in recs if r.get("wording_source") == "llm")
    report = {
        **meta,
        "n_species": len(M.species), "n_couplets": len(recs),
        "couplet_kinds": dict(Counter(r["kind"] for r in recs)),
        "all_species_reachable": not missing, "unreachable_species": missing,
        "species_with_multiple_terminals": multi,
        "e_dicho": e_dicho(step1), "e_dicho_species_counted": len(step1),
        "mean_steps": round(float(np.mean(list(step1.values()))), 2),
        "min_steps": min(step1.values()), "max_steps": max(step1.values()),
        "steps_per_species": dict(sorted(step1.items(), key=lambda kv: natural_key(kv[0]))),
        "threshold_violations_on_perfect_characters": thr_issues,
        "tier2_terms_in_leads": t2,
        "n_characters": sum(len(r["characters"]) for r in recs),
        "n_distinct_features": len({c["feature_id"] for r in recs for c in r["characters"]}),
        "feature_families_used": dict(Counter(c["family"] for r in recs for c in r["characters"])),
        "llm_wording_accepted": wording_ok, "llm_wording_rejected": len(recs) - wording_ok
        if args.llm_backend != "none" else None,
        "identification_resubstitution_first_character": rate("resub_first_char_ok"),
        "identification_resubstitution_vote": rate("resub_vote_ok"),
        "identification_leave_one_out": rate("loo_ok"),
        "identification_resubstitution_resolved_only": None if not len(idf[idf.resub_first_char != "unresolved"]) else
        round(float(idf[idf.resub_first_char != "unresolved"]["resub_first_char_ok"].mean()), 4),
        "identification_leave_one_out_resolved_only": None if "loo" not in idf else
        round(float(idf[idf["loo"].notna() & ~idf["loo"].isin(["unresolved", "species has no other specimen"])]["loo_ok"].astype(bool).mean()), 4),
        "specimens_unresolved_missing_data": unresolved,
        "n_specimens_tested": len(idf),
    }

    # ---- write outputs
    tree = {"meta": meta, "species": {c: {"name": pol.species_display_name(c, profile),
                                          "status": pol.species_status(c, profile)} for c in M.species},
            "couplets": recs, "validation": report}
    (out / "key_tree.json").write_text(json.dumps(tree, indent=1, ensure_ascii=False, default=float),
                                       encoding="utf-8")
    write_text_key(recs, profile, out / "taxonomic_key.txt", out / "taxonomic_key.md", title)
    write_jsonld(recs, profile, M, out / "taxonomic_key.jsonld", meta)
    schema_dir = Path(__file__).parent / "biorag_prompts" / "schemas" / "sdd_1.1"
    ok, errs = write_sdd(recs, profile, M, out / "taxonomic_key.sdd.xml", meta, schema_dir)
    report["sdd_schema_valid"] = ok
    report["sdd_schema_errors"] = errs
    write_matrix(recs, M, profile, out / "character_matrix.tsv")
    (out / "key_validation_report.json").write_text(json.dumps(report, indent=2, default=str))

    txt = [f"KEY VALIDATION REPORT — {title}", f"generated {meta['generated']}", ""]
    for k in ("n_species", "n_couplets", "couplet_kinds", "all_species_reachable",
              "unreachable_species", "species_with_multiple_terminals", "e_dicho",
              "mean_steps", "min_steps", "max_steps", "n_characters", "n_distinct_features",
              "feature_families_used", "threshold_violations_on_perfect_characters",
              "tier2_terms_in_leads", "llm_wording_accepted", "llm_wording_rejected",
              "identification_resubstitution_first_character", "identification_resubstitution_vote",
              "identification_leave_one_out", "identification_resubstitution_resolved_only",
              "identification_leave_one_out_resolved_only", "specimens_unresolved_missing_data",
              "sdd_schema_valid", "sdd_schema_errors"):
        txt.append(f"{k}: {report.get(k)}")
    (out / "key_validation_report.txt").write_text("\n".join(txt))
    log("\n".join(txt))
    logf.close()


if __name__ == "__main__":
    main()
