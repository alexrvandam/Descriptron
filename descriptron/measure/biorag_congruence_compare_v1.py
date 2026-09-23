#!/usr/bin/env python3
"""
biorag_congruence_compare_v1.py — does an extra character set add to the matrix?
================================================================================

The character matrix calls a specimen novel when it lies outside the described
species in at least `--min_sets` independent character sets. This script asks what
happens to that instrument when a further set is added — here the present/absent
characters proposed and scored by a vision-language model — and asks it under the
same two hold-outs as `biorag_instrument_compare_v2.py`:

  detection    the whole species is withheld, its specimens are scored
  false alarm  one specimen is withheld, its species stays in place

Three things are measured that a single "best margin" does not show.

1. CHARACTER-SELECTION LEAK. Model-proposed characters are drawn up from contrast
   sets "species X beside other species". When X is later withheld to play the
   part of an undescribed species, the character list still holds the characters
   that were proposed *in order to separate X* — a list no real undescribed
   species could have contributed to. `--proposals` closes this: with X withheld,
   every character whose `proposed_from` is X is removed before X is scored. The
   false-alarm arm keeps the full list, because a described species' characters
   legitimately exist when a further specimen of it arrives. Both variants are
   reported side by side so the size of the leak is a number, not a worry.

2. IN-SAMPLE OPERATING POINT. The best margin over a threshold sweep is chosen on
   the species it is then reported for. The nested estimate chooses threshold and
   rule on the other species and applies them to the one left out, which is what
   a user who calibrates on their own reference set will actually get.

3. PAIRED DIFFERENCE. Two instruments scored on the same 29 species are compared
   by the species on which they disagree (exact McNemar), not by two proportions.

It also scores the matrix as an IDENTIFIER under leave-one-specimen-out (nearest
species, per set and combined) and cross-tabulates it against the key's own
leave-one-out answer, since agreement between two instruments is a usable
confidence signal where neither is reliable alone.

  python biorag_congruence_compare_v1.py --matrix_dir "$M/compiled_key_tier" \\
      --taxon_profile <profile.yaml> \\
      --extra_states "$M/vlm_combined/vlm_character_states.tsv" \\
      --proposals "$M/vlm_combined/vlm_proposed_characters.tsv" \\
      --key_loo "$M/key/identification_test.tsv" --out_dir "$M/congruence_compare"
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                      # noqa: E402
from biorag_novelty_score_v1 import (Reference, SETS, species_distance,   # noqa: E402
                                     within_spread, pooled_spread)

VERSION = "1.1"
RULES = {"any specimen": lambda f, n: f > 0,
         "a majority of the series": lambda f, n: f > n / 2,
         "the whole series": lambda f, n: f == n}


# ─────────────────────────────────────────────────────────────────────────────
# the extra set
# ─────────────────────────────────────────────────────────────────────────────
def load_states(path: Path, present=("present", "1", "true", "yes")) -> pd.DataFrame:
    """specimen × character, 1 = present, 0 = absent, NaN = not scored."""
    s = pd.read_csv(path, sep="\t")
    s["v"] = s["state"].astype(str).str.lower().isin(present).astype(float)
    return s.pivot_table(index="specimen_id", columns="character", values="v", aggfunc="mean").round()


def register(ref: Reference, name: str, table: pd.DataFrame):
    """A present/absent set compares by the share of shared characters that differ, so its
    scale is 1 for every character. The pooled-MAD scale used for measurements is wrong
    here: it is zero for any mostly-fixed character and falls back to a value that depends
    on which state happens to be the majority."""
    t = table.reindex(ref.raw.index)
    ref.tables[name] = t
    ref.scales[name] = pd.Series(1.0, index=t.columns)
    ref.unit_scale = getattr(ref, "unit_scale", set()) | {name}


# ─────────────────────────────────────────────────────────────────────────────
# added sets that are MEASUREMENTS, not states (outline shape, colour pattern,
# texture): they must go through the same pooled within-species MAD as size,
# ratio, landmark and colour — the one the Reference re-derives without whatever
# is withheld (scale_for). Putting them in unit_scale would score them on a
# footing the built-in sets are not scored on.
# ─────────────────────────────────────────────────────────────────────────────
def register_continuous(ref: Reference, name: str, table: pd.DataFrame):
    t = table.reindex(ref.raw.index).astype(float)
    ref.tables[name] = t
    ref.scales[name] = ref._within_sd(t)       # the machinery the built-in sets use
    if name in getattr(ref, "unit_scale", set()):
        ref.unit_scale.discard(name)


def unregister(ref: Reference, name: str):
    """Drop a table and every cached statistic derived from it (the caches are keyed
    by set name, so a stale entry would be silently reused)."""
    ref.tables.pop(name, None)
    ref.scales.pop(name, None)
    for cache in (ref._scache, ref._wcache):
        for k in [k for k in cache if k[0] == name]:
            cache.pop(k, None)


def load_continuous(path: Path) -> pd.DataFrame:
    """specimen_id x feature, numeric. Columns that are empty everywhere are dropped."""
    t = pd.read_csv(path, sep="\t")
    idc = "specimen_id" if "specimen_id" in t.columns else t.columns[0]
    t = t.set_index(idc)
    t = t.apply(pd.to_numeric, errors="coerce")
    return t.dropna(axis=1, how="all")


class FoldSet:
    """A continuous set whose TABLE depends on the fold.

    Outline shape is Procrustes-aligned to a consensus, and a consensus is a statistic of
    the sample: leaving a specimen out of the distances while leaving it inside the shape
    everything is aligned to is not a hold-out. `FoldSet` therefore registers a different
    table per fold — the coordinates aligned to a consensus built from the training
    outlines alone — and hands `score_arms` its name. Tables are built lazily and the
    oldest are unregistered, so 29 + 149 folds cost a few tables' memory, not 178."""

    def __init__(self, ref: Reference, name: str, folds, columns, max_registered: int = 3):
        self.ref, self.name, self.folds = ref, name, folds
        self.columns = list(columns)
        self.max_registered = int(max_registered)
        self._tag: Dict[frozenset, str] = {}
        self._live: List[str] = []
        self.n_features = len(self.columns)

    def table(self, exclude) -> str:
        key = frozenset(exclude)
        tag = self._tag.get(key)
        if tag is None:
            tag = self._tag[key] = f"{self.name} #{len(self._tag)}"
        if tag in self._live:
            self._live.remove(tag)
        else:
            register_continuous(self.ref, tag, self.folds.get(key)[self.columns])
        self._live.append(tag)
        while len(self._live) > self.max_registered:
            unregister(self.ref, self._live.pop(0))
        return tag


# ─────────────────────────────────────────────────────────────────────────────
# scoring: one pass gives g (novelty) and the full ranking (identification)
# ─────────────────────────────────────────────────────────────────────────────
def score_full(ref, name, cand, exclude_species=(), exclude_specimens=(), stat="mean"):
    """Same arithmetic as biorag_novelty_score_v1.score_candidate, keeping every species."""
    weights = ref.weights(name, exclude_species, exclude_specimens)
    scale = ref.scale_for(name, exclude_species, exclude_specimens)
    rows = []
    for s in ref.species:
        if s in exclude_species:
            continue
        d, n = species_distance(ref, name, cand, s, exclude_specimens, weights, stat, scale=scale)
        if d == d:
            rows.append((s, d))
    if not rows:
        return math.nan, {}
    rows.sort(key=lambda r: r[1])
    nearest, d = rows[0]
    w = within_spread(ref, name, nearest, exclude_specimens, weights, stat, scale=scale)
    if not (w == w):
        w = pooled_spread(ref, name, exclude_species, weights, stat, scale=scale,
                          exclude_specimens=exclude_specimens if ref.holdout_scale else ())
    g = d / w if w and w == w else math.nan
    return g, dict(rows)


def score_arms(ref, names, table_for=None, table_for_specimen=None):
    """table_for(name, held_out_species) -> the set name to score the DETECTION arm with
    (lets a set drop the characters a withheld species contributed to).
    table_for_specimen(name, specimen_id) -> the same for the FALSE-ALARM arm, for a set
    whose table itself has to be rebuilt without the withheld specimen (outline shape:
    the consensus the coordinates are aligned to). The scores are always reported under
    the set's own name, so everything downstream is unchanged."""
    det, fal, ranks = [], [], []
    ids = list(ref.raw.index)
    for k, sid in enumerate(ids):
        sp = ref.species_of.get(sid)
        rd = {"specimen_id": sid, "species": sp}
        rf = {"specimen_id": sid, "species": sp}
        for name in names:
            dname = table_for(name, sp) if table_for else name
            if dname in ref.tables and sid in ref.tables[dname].index:
                g, _ = score_full(ref, dname, ref.tables[dname].loc[sid],
                                  exclude_species=(sp,), exclude_specimens=(sid,))
                rd[name] = g
            fname = table_for_specimen(name, sid) if table_for_specimen else name
            if fname in ref.tables and sid in ref.tables[fname].index:
                g, dist = score_full(ref, fname, ref.tables[fname].loc[sid],
                                     exclude_species=(), exclude_specimens=(sid,))
                rf[name] = g
                for s2, d in dist.items():
                    ranks.append({"specimen_id": sid, "species": sp, "set": name,
                                  "candidate_species": s2, "d": d})
        det.append(rd)
        fal.append(rf)
        if (k + 1) % 25 == 0:
            print(f"  scored {k + 1}/{len(ids)}")
    return pd.DataFrame(det), pd.DataFrame(fal), pd.DataFrame(ranks)


# ─────────────────────────────────────────────────────────────────────────────
# sweep, nested choice, paired comparison
# ─────────────────────────────────────────────────────────────────────────────
def species_counts(frame, cols, t, min_sets):
    n_out = (frame[cols] > t).sum(axis=1)
    g = pd.DataFrame({"species": frame["species"], "flag": n_out >= min_sets})
    return g.groupby("species")["flag"].agg(["sum", "size"])


def sweep(det, fal, cols, min_sets, grid=None):
    allg = pd.concat([det[cols].stack(), fal[cols].stack()]).dropna()
    if grid is None:
        grid = np.unique(allg.values)                       # every distinct operating point
    rows, calls = [], {}
    for t in grid:
        D = species_counts(det, cols, t, min_sets)
        F = species_counts(fal, cols, t, min_sets)
        for rule, fn in RULES.items():
            d_hit = D.apply(lambda r: bool(fn(r["sum"], r["size"])), axis=1)
            f_hit = F.apply(lambda r: bool(fn(r["sum"], r["size"])), axis=1)
            rows.append({"threshold": float(t), "rule": rule,
                         "caught": int(d_hit.sum()), "of_unseen": len(D),
                         "false_alarms": int(f_hit.sum()), "of_described": len(F),
                         "margin": d_hit.mean() - f_hit.mean()})
            calls[(float(t), rule)] = (d_hit, f_hit)
    return pd.DataFrame(rows), calls


def at_ceiling(sw, max_fa):
    ok = sw[sw["false_alarms"] <= max_fa]
    if not len(ok):
        return None
    return ok.sort_values(["caught", "false_alarms"], ascending=[False, True]).iloc[0]


def nested(sw, calls, species=None):
    """Choose (threshold, rule) on the other species, apply to the one left out.

    `species` fixes the denominator. An instrument that cannot score a species at all (a key
    whose every path stops on an unmeasured character) has not recognised it, so a species
    missing from an arm counts as not flagged rather than being dropped from the total."""
    if species is None:
        species = sorted(set().union(*[set(d.index) | set(f.index) for d, f in calls.values()]))
    keys = list(calls.keys())
    Dm = np.array([calls[k][0].reindex(species, fill_value=False).astype(float).values for k in keys])
    Fm = np.array([calls[k][1].reindex(species, fill_value=False).astype(float).values for k in keys])
    caught = false = 0
    for j in range(len(species)):
        keep = np.arange(len(species)) != j
        m = Dm[:, keep].mean(axis=1) - Fm[:, keep].mean(axis=1)
        best = int(np.argmax(m))                            # first best = lowest threshold
        caught += int(Dm[best, j])
        false += int(Fm[best, j])
    n = len(species)
    return {"caught": caught, "false_alarms": false, "of": n,
            "margin": round(caught / n - false / n, 3)}


def wilson(k, n, z=1.96):
    if not n:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0.0, c - h), 1), round(100 * min(1.0, c + h), 1))


def mcnemar(a_hit: pd.Series, b_hit: pd.Series):
    a_hit, b_hit = a_hit.align(b_hit, join="inner")
    only_a = sorted(a_hit.index[a_hit & ~b_hit])
    only_b = sorted(b_hit.index[b_hit & ~a_hit])
    n, k = len(only_a) + len(only_b), min(len(only_a), len(only_b))
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p


# ─────────────────────────────────────────────────────────────────────────────
# identification
# ─────────────────────────────────────────────────────────────────────────────
def identify(ranks: pd.DataFrame, sets):
    """Nearest species per set, and combined over sets by mean rank (rank is unit-free,
    so a set cannot dominate by having larger distances)."""
    r = ranks[ranks["set"].isin(sets)].copy()
    r["rank"] = r.groupby(["specimen_id", "set"])["d"].rank(method="average")
    comb = (r.groupby(["specimen_id", "species", "candidate_species"])["rank"].mean().reset_index())
    idx = comb.groupby("specimen_id")["rank"].idxmin()
    out = comb.loc[idx, ["specimen_id", "species", "candidate_species"]]
    return out.rename(columns={"candidate_species": "named"}).set_index("specimen_id")


def main():
    ap = argparse.ArgumentParser(description="Does an extra character set add to the matrix?")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--extra_states", default=None,
                    help="long table: specimen_id, character, state (present/absent)")
    ap.add_argument("--extra_name", default="vlm")
    ap.add_argument("--proposals", default=None,
                    help="vlm_proposed_characters.tsv (structure, name, proposed_from): with a "
                         "species withheld, the characters it was used to propose are removed")
    ap.add_argument("--extra_continuous", action="append", default=[], metavar="NAME=TABLE.tsv",
                    help="a MEASURED extra set (specimen_id x feature TSV). Repeatable. It goes "
                         "through the same pooled within-species MAD as size/ratio/landmark/"
                         "colour, re-derived without whatever is withheld")
    ap.add_argument("--shape_npz", default=None,
                    help="outline_registered.npz from biorag_outline_shape_v1.py")
    ap.add_argument("--shape_set", action="append", default=[], metavar="NAME[=struct,struct]",
                    help="a set built from --shape_npz; no structure list = every structure. "
                         "Its table is rebuilt per fold (the consensus must not contain the "
                         "withheld material). Repeatable")
    ap.add_argument("--shape_basis", choices=["coords", "pca"], default="coords")
    ap.add_argument("--shape_n_pcs", type=int, default=10)
    ap.add_argument("--key_loo", default=None, help="key/identification_test.tsv")
    ap.add_argument("--min_sets", type=int, default=2)
    ap.add_argument("--ceilings", nargs="*", type=int, default=[1, 3, 4, 6, 9])
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    ref = Reference(Path(a.matrix_dir), profile)
    base = [s for s in SETS if s in ref.tables and len(ref.tables[s].columns)]
    X, XL = a.extra_name, a.extra_name + "_leakfree"

    states = None
    if a.extra_states:
        states = load_states(Path(a.extra_states))
        register(ref, X, states)
        print(f"{X}: {states.shape[1]} characters on {states.shape[0]} specimens")

    contributed = {}
    if a.proposals and states is not None:
        p = pd.read_csv(a.proposals, sep="\t")
        p["character"] = p["structure"].astype(str) + ":" + p["name"].astype(str)
        # seen_species (every species on screen at proposal) when the run recorded it,
        # otherwise proposed_from (the species the contrast set was built around)
        who = "seen_species" if "seen_species" in p.columns else "proposed_from"
        by_sp = {}
        for ch, v in zip(p["character"], p[who].astype(str)):
            for sp in [x for x in v.split(";") if x and x != "nan"]:
                by_sp.setdefault(sp, []).append(ch)
        for sp, chs in by_sp.items():
            cols = sorted({c for c in chs if c in states.columns})
            if cols:
                contributed[sp] = cols
                register(ref, f"{XL}__{sp}", states.drop(columns=cols))
        print(f"selection leak closed on: {who}")
        print(f"characters traced to a proposing species: "
              f"{sum(len(v) for v in contributed.values())} of {states.shape[1]} "
              f"({len(contributed)} species)")
    pd.DataFrame([{"species": k, "characters_removed_when_withheld": len(v)}
                  for k, v in contributed.items()]).to_csv(
        out / "characters_removed_per_withheld_species.tsv", sep="\t", index=False)

    # ── added sets ──────────────────────────────────────────────────────────
    # A set the Reference silently ignores looks exactly like a set that adds nothing,
    # so every added set says here how many of its features and specimens were read.
    added, fold_sets, added_report = [], {}, []
    for spec in a.extra_continuous:
        if "=" not in spec:
            sys.exit(f"--extra_continuous wants NAME=TABLE.tsv, got {spec!r}")
        nm, path = spec.split("=", 1)
        t = load_continuous(Path(path))
        register_continuous(ref, nm, t)
        read = ref.tables[nm]
        n_spec = int(read.notna().any(axis=1).sum())
        added.append(nm)
        added_report.append({"set": nm, "kind": "continuous", "source": path,
                             "features_in_file": int(t.shape[1]),
                             "features_read": int(read.shape[1]),
                             "specimens_in_file": int(t.shape[0]),
                             "specimens_read": n_spec, "of_matrix_specimens": len(ref.raw.index)})
        print(f"{nm}: {read.shape[1]} features read on {n_spec}/{len(ref.raw.index)} "
              f"matrix specimens")
    if a.shape_set:
        if not a.shape_npz:
            sys.exit("--shape_set needs --shape_npz")
        import biorag_outline_shape_v1 as osh
        reg, _meta = osh.load_registered(Path(a.shape_npz))
        aligner = osh.FoldAligner(reg, basis=a.shape_basis, n_pcs=a.shape_n_pcs)
        folds = osh.FoldTables(aligner, max_cached=4)
        # runtime hold-out guard: the fold frame must be built from the training outlines
        # only, so replacing the withheld outlines with noise may not move any other row
        probe_sp = next((sp for sp in ref.species
                         if any(ref.species_of.get(s) == sp for s in aligner.specimens)), None)
        probe = [s for s in aligner.specimens if ref.species_of.get(s) == probe_sp]
        if not probe:
            sys.exit("outline shape: no specimen of the registered outlines is in the matrix")
        rng = np.random.default_rng(0)
        dirty = {st: dict(R) for st, R in reg.items()}
        for st in dirty:
            S = dirty[st]["shapes"].copy()
            for i, sid in enumerate(dirty[st]["ids"]):
                if sid in probe:
                    S[i] = rng.normal(0, 1, S[i].shape)
            dirty[st] = {**dirty[st], "shapes": S}
        clean_f = aligner.frame(probe)
        dirty_f = osh.FoldAligner(dirty, basis=a.shape_basis, n_pcs=a.shape_n_pcs).frame(probe)
        keep = [s for s in clean_f.index if s not in probe]
        if not clean_f.loc[keep].equals(dirty_f.loc[keep]):
            sys.exit("outline shape: the fold frame moved when the WITHHELD outlines were "
                     "replaced by noise — the consensus is not training-only")
        print(f"outline shape hold-out guard passed on {probe_sp} "
              f"({len(probe)} withheld outlines, {len(keep)} rows unchanged)")
        allcols = list(clean_f.columns)
        for spec in a.shape_set:
            nm, _, structs = spec.partition("=")
            want = [s.strip() for s in structs.split(",") if s.strip()] if structs else None
            cols = [c for c in allcols if want is None or c.rsplit(".", 1)[0] in want]
            if not cols:
                sys.exit(f"--shape_set {nm}: no structure matched {structs!r}")
            fold_sets[nm] = FoldSet(ref, nm, folds, cols)
            added.append(nm)
            sts = sorted({c.rsplit('.', 1)[0] for c in cols})
            n_spec = int(clean_f[cols].notna().any(axis=1).sum())
            added_report.append({"set": nm, "kind": "outline shape (per-fold table)",
                                 "source": a.shape_npz, "features_in_file": len(allcols),
                                 "features_read": len(cols), "structures": len(sts),
                                 "specimens_in_file": len(clean_f),
                                 "specimens_read": n_spec,
                                 "of_matrix_specimens": len(ref.raw.index),
                                 "structure_list": ", ".join(sts)})
            print(f"{nm}: {len(cols)} coordinates over {len(sts)} structures, "
                  f"{n_spec}/{len(ref.raw.index)} matrix specimens")
    if added_report:
        pd.DataFrame(added_report).to_csv(out / "added_sets_read.tsv", sep="\t", index=False)

    def table_for(name, sp):
        if name == XL:
            return f"{XL}__{sp}" if sp in contributed else X
        if name in fold_sets:
            return fold_sets[name].table(
                [s for s in ref.raw.index if ref.species_of.get(s) == sp])
        return name

    def table_for_specimen(name, sid):
        if name in fold_sets:
            return fold_sets[name].table([sid])
        return name

    if states is not None:
        register(ref, XL, states)                                     # false-alarm arm: full list
    names = base + ([X] if states is not None else []) \
        + ([XL] if contributed else []) + added
    print(f"scoring both arms for: {', '.join(names)}")
    det, fal, ranks = score_arms(ref, names, table_for,
                                 table_for_specimen if fold_sets else None)
    for nm in added:
        # the two arms must be scored on the same material: a set that reaches more
        # specimens in one arm than in the other is not comparable between them
        nd, nf = int(det[nm].notna().sum()), int(fal[nm].notna().sum())
        if nd != nf:
            sys.exit(f"{nm}: scored {nd} specimens in the detection arm and {nf} in the "
                     f"false-alarm arm — the two arms are not on the same footing")
        print(f"  {nm}: both arms scored {nd}/{len(ref.raw.index)} specimens")
    if contributed:
        # The false-alarm arm keeps the full character list, so it must be the SAME numbers as
        # the unfiltered set. If it is not, the two arms of the leak-closed instrument were
        # scored on different footings and any margin between them is an artefact.
        bad = ~((fal[X] - fal[XL]).abs().fillna(0) < 1e-12)
        if bad.any():
            sys.exit(f"false-alarm arm of {XL} differs from {X} for {int(bad.sum())} specimens: "
                     f"the two arms are not on the same scale")
        same = [sp for sp in ref.species if sp not in contributed]
        m = det["species"].isin(same)
        if len(same) and not ((det.loc[m, X] - det.loc[m, XL]).abs().fillna(0) < 1e-12).all():
            sys.exit(f"detection arm of {XL} differs from {X} for species that contributed no "
                     f"characters, where the two must be identical")
    det.to_csv(out / "detection_scores.tsv", sep="\t", index=False)
    fal.to_csv(out / "false_alarm_scores.tsv", sep="\t", index=False)

    combos = {"matrix": base}
    if states is not None:
        combos[f"matrix + {X}"] = base + [X]
        combos[f"{X} alone"] = [X]
    if contributed:
        combos[f"matrix + {X}, selection leak closed"] = base + [XL]
        combos[f"{X} alone, selection leak closed"] = [XL]
    for nm in added:
        combos[f"matrix + {nm}"] = base + [nm]
    if len(added) > 1:
        combos["matrix + every added set"] = base + added
    for nm in added:
        combos[f"{nm} alone"] = [nm]

    # the published grid, to confirm this script reproduces biorag_instrument_compare_v2
    allg = pd.concat([det[base].stack(), fal[base].stack()]).dropna()
    legacy, _ = sweep(det, fal, base, a.min_sets, np.quantile(allg, np.linspace(0.50, 0.999, 60)))
    lb = legacy.sort_values(["margin", "caught"], ascending=False).iloc[0]
    print(f"\nfor reference, the 60-quantile grid of instrument_compare_v2 (4 sets): margin {lb['margin']:.3f}, "
          f"{int(lb['caught'])}/{int(lb['of_unseen'])} caught, "
          f"{int(lb['false_alarms'])}/{int(lb['of_described'])} false")

    table, sweeps, best_calls = [], [], {}
    for label, cols in combos.items():
        ms = 1 if len(cols) == 1 else a.min_sets
        sw, calls = sweep(det, fal, cols, ms)
        sw["instrument"] = label
        sweeps.append(sw)
        b = sw.sort_values(["margin", "caught"], ascending=False).iloc[0]
        row = {"instrument": label, "sets": len(cols), "min_sets": ms,
               "best_margin_in_sample": round(float(b["margin"]), 3),
               "best_point": f"{int(b['caught'])}/{int(b['of_unseen'])} caught, "
                             f"{int(b['false_alarms'])}/{int(b['of_described'])} false "
                             f"({b['rule']})"}
        ne = nested(sw, calls)
        row["nested_margin"] = ne["margin"]
        row["nested_point"] = f"{ne['caught']}/{ne['of']} caught, {ne['false_alarms']}/{ne['of']} false"
        for c in a.ceilings:
            pnt = at_ceiling(sw, c)
            row[f"caught_at_max_{c}_false"] = int(pnt["caught"]) if pnt is not None else None
            if pnt is not None:
                best_calls[(label, c)] = calls[(float(pnt["threshold"]), pnt["rule"])][0]
        table.append(row)
    tab = pd.DataFrame(table)
    tab.to_csv(out / "congruence_comparison.tsv", sep="\t", index=False)
    pd.concat(sweeps).to_csv(out / "congruence_sweeps.tsv", sep="\t", index=False)

    paired = []
    ref_label = "matrix"
    for label in combos:
        if label == ref_label or "alone" in label:
            continue
        for c in a.ceilings:
            if (label, c) in best_calls and (ref_label, c) in best_calls:
                gained, lost, p = mcnemar(best_calls[(label, c)], best_calls[(ref_label, c)])
                paired.append({"comparison": f"{label} vs {ref_label}", "max_false_alarms": c,
                               "species_gained": ", ".join(gained), "species_lost": ", ".join(lost),
                               "n_gained": len(gained), "n_lost": len(lost),
                               "mcnemar_exact_p": round(p, 3),
                               "gained_species_that_proposed_characters":
                                   ", ".join(s for s in gained if s in contributed)})
    pair = pd.DataFrame(paired)
    pair.to_csv(out / "paired_species_differences.tsv", sep="\t", index=False)

    # ── identification under leave-one-specimen-out ──────────────────────────
    ident = []
    ranks = ranks[ranks["set"] != XL]
    ranks.to_csv(out / "identification_distances.tsv", sep="\t", index=False)
    named = {}
    id_combos = [(s, [s]) for s in base + ([X] if states is not None else []) + added] \
        + [("matrix (all measured sets)", base)] \
        + ([(f"matrix + {X}", base + [X])] if states is not None else []) \
        + [(f"matrix + {nm}", base + [nm]) for nm in added] \
        + ([("matrix + every added set", base + added)] if len(added) > 1 else [])
    for label, sets in id_combos:
        r = identify(ranks, sets)
        named[label] = r
        ident.append({"identifier": label, "specimens_named": len(r),
                      "correct": int((r["named"] == r["species"]).sum()),
                      "accuracy_of_named": round(float((r["named"] == r["species"]).mean()), 3),
                      "accuracy_of_all": round(float((r["named"] == r["species"]).sum()
                                                     / len(ref.raw.index)), 3)})
    # paired difference in NAMING, specimen by specimen, against the matrix alone
    id_paired = []
    if "matrix (all measured sets)" in named:
        m0 = named["matrix (all measured sets)"]
        hit0 = (m0["named"] == m0["species"]).reindex(ref.raw.index, fill_value=False)
        for label, _sets in id_combos:
            if label == "matrix (all measured sets)":
                continue
            mm = named[label]
            hit = (mm["named"] == mm["species"]).reindex(ref.raw.index, fill_value=False)
            gained, lost, p = mcnemar(hit, hit0)
            id_paired.append({"comparison": f"{label} vs matrix (all measured sets)",
                              "specimens_gained": len(gained), "specimens_lost": len(lost),
                              "correct": int(hit.sum()), "matrix_correct": int(hit0.sum()),
                              "of": len(ref.raw.index), "mcnemar_exact_p": round(p, 3)})
        pd.DataFrame(id_paired).to_csv(out / "paired_identification_differences.tsv",
                                       sep="\t", index=False)
    agree = None
    if a.key_loo and Path(a.key_loo).exists():
        k = pd.read_csv(a.key_loo, sep="\t").set_index("specimen_id")
        kr = k["loo"].where(k["loo"].isin(ref.species))
        ident.append({"identifier": "key (leave one specimen out)",
                      "specimens_named": int(kr.notna().sum()),
                      "correct": int((kr == k["species"]).sum()),
                      "accuracy_of_named": round(float((kr == k["species"]).sum()
                                                       / max(1, kr.notna().sum())), 3),
                      "accuracy_of_all": round(float((kr == k["species"]).sum() / len(k)), 3)})
        rows = []
        for label in [lab for lab, _ in id_combos if lab.startswith("matrix")]:
            m = named[label]["named"].reindex(k.index)
            both = kr.notna() & m.notna()
            same = both & (kr == m)
            diff = both & (kr != m)
            only_m = kr.isna() & m.notna()
            for what, mask, pick in (("key and matrix agree", same, m),
                                     ("they disagree: the key's name", diff, kr),
                                     ("they disagree: the matrix's name", diff, m),
                                     ("key reaches no name: the matrix's name", only_m, m)):
                n = int(mask.sum())
                ok = int((pick[mask] == k["species"][mask]).sum())
                rows.append({"matrix": label, "case": what, "specimens": n, "correct": ok,
                             "accuracy": round(ok / n, 3) if n else None})
            # the rule that follows: the agreed name, else the matrix's
            final = m.where(~(both & (kr == m)), kr).fillna(kr)
            ok = int((final == k["species"]).sum())
            rows.append({"matrix": label, "case": "combined: every specimen named (matrix, "
                         "flagged as confirmed where the key agrees)", "specimens": int(final.notna().sum()),
                         "correct": ok, "accuracy": round(ok / max(1, int(final.notna().sum())), 3)})
        agree = pd.DataFrame(rows)
        agree.to_csv(out / "identification_key_vs_matrix.tsv", sep="\t", index=False)
    idt = pd.DataFrame(ident)
    idt.to_csv(out / "identification_loo.tsv", sep="\t", index=False)

    (out / "congruence_summary.json").write_text(json.dumps({
        "version": VERSION, "min_sets": a.min_sets,
        "extra_set": X if states is not None else None,
        "extra_characters": int(states.shape[1]) if states is not None else 0,
        "added_sets": added_report,
        "characters_traced_to_a_proposing_species": int(sum(len(v) for v in contributed.values())),
        "matrix_margin_on_the_v2_quantile_grid": round(float(lb["margin"]), 3),
        "comparison": table, "paired": paired, "identification": ident,
        "paired_identification": id_paired,
        "identification_key_vs_matrix": agree.to_dict("records") if agree is not None else None,
        "limits": ["the false-alarm arm keeps the full character list; a specimen that was itself "
                   "shown to the model at the proposal stage is not removed from it",
                   "only `proposed_from` (the species in Group A) is traced; a withheld species "
                   "may also have served in Group B of another species' contrast set"]}, indent=2))

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70)
    print("\n" + tab.to_string(index=False))
    if len(pair):
        print("\n" + pair.to_string(index=False))
    print("\n" + idt.to_string(index=False))
    if agree is not None:
        print("\n" + agree.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
