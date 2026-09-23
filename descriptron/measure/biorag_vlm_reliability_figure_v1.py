#!/usr/bin/env python3
"""
biorag_vlm_reliability_figure_v1.py — when a vision-language model reads a
structure off an image, how much of what it says comes back a second time, how
much of it is pinned to a place, and how much of it is worth anything?
===============================================================================

Two experiments were run on the same images, and they ask different questions.
They are kept visibly apart in the figure because their failure modes differ.

  EXPERIMENT 1 — PROMPTED DESCRIPTIONS (panel a). The model was handed a fixed
  list of 11 descriptive characters (surface roughness, surface pattern, lustre,
  transparency, margin incision, elongation, outline shape, curvature, apex,
  colour heterogeneity, vestiture) and, for each one, a fixed list of states —
  most ordinal scales, two nominal — or "not assessable". It chose one state per
  specimen x structure x character. It was never asked WHERE it saw anything.
  A sample of the images was read a second time, independently, and the two
  readings compared cell by cell.

  EXPERIMENT 2 — FREE-FORM CHARACTER SEARCH (panels b-d). The model PROPOSED
  present/absent characters of its own from contrast sets, then scored every
  specimen alone and blind, and for every report said WHERE on a common frame
  it saw the character (x, y as fractions of a 768-px frame, origin top left).
  States were re-read twice: an original retest (states only) and a later retest
  of 30 specimens that recorded locations as well, so the same character on the
  same image can be asked to point twice.

THREE PROPERTIES, DELIBERATELY NOT CONFLATED

  REPEATABLE — the same call comes back on a second reading of the same image.
  Measured as raw agreement and, for the binary free-form characters, as Cohen's
  kappa, because a character that is "absent" on 95% of specimens agrees with
  itself 90% of the time by chance alone and raw agreement flatters it.

  LOCATED — the reports land in one place on the frame, that place is on the
  specimen, and a second reading of the SAME image puts the character back where
  the first one did. A character can be perfectly repeatable as a state and
  still wander over the whole structure, in which case nobody can be shown what
  was scored.

  USEFUL — the character carries information about species identity (adjusted
  mutual information), is fixed within at least one species and rare outside it,
  and survives as a confirmed autapomorphy. Repeatable and located say the
  observation is real; only this says it is worth putting in a treatment.

CHANCE CORRECTION IN EXPERIMENT 1 IS AN ESTIMATE, AND IS LABELLED AS ONE.
The paired second readings of the prompted retest were written to a temporary
folder that no longer exists; only the first reading survives. A true Cohen's
kappa therefore cannot be computed. What is drawn instead is the agreement
expected by chance from the FIRST-READING state frequencies of that character,
p_e = sum_k p_k^2 over its observed states ("not assessable" and blanks
excluded), and the chance-corrected agreement estimated from it,
(exact - p_e) / (1 - p_e). Supply --prompted_retest with a TSV of paired second
readings in the same column layout as the first reading and the script replaces
the estimate with a true Cohen's kappa everywhere it appears.

THE RETEST HAS SINCE BEEN REPEATED, AND PANEL a NOW SHOWS IT. Three readings of
the same images exist: a first reading (18 Sep) and two readings made on one
later day (A and B, 20 Sep), compared cell by cell by biorag_retest_compare_v1.py.
When --retest_compare points at that comparison (it defaults to
<monograph>/descriptive_states/retest_compare when the directory is there),
panel a is drawn from it instead: exact agreement WITHIN a session (A vs B) and
BETWEEN sessions (first vs A), with a true Cohen's kappa for each and the
direction in which the second reading moved. Without those files nothing
changes and panel a keeps the estimate described above.

ORIENTATION. Homology frames can lie end for end. Comparing locations BETWEEN
specimens therefore maps (x, y) -> (1 - x, 1 - y) for frames flagged flipped.
Comparing two readings of the SAME image needs no mapping and gets none.

  python biorag_vlm_reliability_figure_v1.py \\
      --monograph "$M" --out_dir "$M/vlm_reliability"

Writes prompted_character_repeatability.tsv, freeform_character_reliability.tsv,
relocation_pairs.tsv, attrition.tsv, vlm_reliability_summary.json and
fig_vlm_reliability.png / .pdf.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe                                      # noqa: E402
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec        # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Patch, Rectangle                          # noqa: E402

VERSION = "1.0"

# ── colour ──────────────────────────────────────────────────────────────────
# Experiment 1 verdicts: dark / mid / light, so the three survive greyscale,
# and a different marker shape each, so they survive it even in a bad print.
VERDICT_COLOUR = {"use": "#12436D", "use coarse": "#E69F00", "flag": "#BFBFBF"}
VERDICT_EDGE = {"use": "#12436D", "use coarse": "#9A6B00", "flag": "#7A7A7A"}
VERDICT_MARKER = {"use": "o", "use coarse": "s", "flag": "^"}
VERDICT_LABEL = {"use": "used as scored", "use coarse": "used at band level",
                 "flag": "set aside"}

# Experiment 1, sessions view of panel a: the same image read again the same day
# (dark blue, filled circle) and two days later (warm orange, filled diamond).
# The two differ in shape as well as colour, so the drop survives greyscale.
WITHIN_COLOUR, WITHIN_EDGE = "#12436D", "#0B2C48"
BETWEEN_COLOUR, BETWEEN_EDGE = "#E69F00", "#9A6B00"
# a third series — one day apart — in Okabe-Ito bluish green, a square: the three
# sit at three grey levels (dark blue 57, green 106, orange 162) and three shapes.
DAY_COLOUR, DAY_EDGE = "#009E73", "#00684C"

# Experiment 2: the teal family used for model-proposed characters elsewhere.
TEAL = "#00838f"
TEAL_DARK = "#005662"
TEAL_LIGHT = "#8ecfd6"
TEAL_PALE = "#d6eef0"
GREY = "#8a8a8a"
INK = "#222222"


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.6, width=0.7)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.7)


# ─────────────────────────────────────────────────────────────────────────────
# small statistics
# ─────────────────────────────────────────────────────────────────────────────

def cohen_kappa(a, b):
    """Cohen's kappa for two aligned label vectors. NaN when it is undefined."""
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    if len(a) == 0:
        return np.nan
    states = sorted(set(a) | set(b), key=str)
    if len(states) < 2:                       # only one state was ever used
        return np.nan
    po = float((a == b).mean())
    pe = float(sum((a == s).mean() * (b == s).mean() for s in states))
    if pe >= 1.0 - 1e-12:
        return np.nan
    return (po - pe) / (1.0 - pe)


def chance_agreement(states):
    """p_e = sum_k p_k^2 from one reading's state frequencies."""
    s = pd.Series(states).dropna()
    if len(s) == 0:
        return np.nan
    p = s.value_counts(normalize=True).values
    return float((p ** 2).sum())


def ecdf(x):
    x = np.sort(np.asarray(x, dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x)


def coordinate_step(values, candidates=(0.001, 0.005, 0.01, 0.02, 0.025, 0.05)):
    """Coarsest grid that at least 99% of the reported coordinates fall on."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    best = candidates[0]
    for g in candidates:
        on = float((np.abs(v / g - np.round(v / g)) < 1e-6).mean())
        if on >= 0.99:
            best = g
    return best


# ─────────────────────────────────────────────────────────────────────────────
# experiment 1 — prompted descriptions
# ─────────────────────────────────────────────────────────────────────────────

def experiment_one(first_tsv, reliability_tsv, repeatability_json,
                   prompted_retest, not_assessable):
    first = pd.read_csv(first_tsv, sep="\t")
    rel = pd.read_csv(reliability_tsv, sep="\t")
    rep = json.loads(Path(repeatability_json).read_text())

    scored = first[first["state"].notna() & (first["state"] != not_assessable)]

    retest = None
    if prompted_retest:
        retest = pd.read_csv(prompted_retest, sep="\t")
        keys = [k for k in ("specimen_id", "image", "structure", "character")
                if k in retest.columns and k in first.columns]
        retest = first.merge(retest[keys + ["state"]].drop_duplicates(keys),
                             on=keys, suffixes=("_1", "_2"))
        retest = retest[(retest["state_1"] != not_assessable)
                        & (retest["state_2"] != not_assessable)]

    rows = []
    for _, r in rel.iterrows():
        ch = r["character"]
        st = scored.loc[scored["character"] == ch, "state"]
        pe = chance_agreement(st)
        exact = float(r["repeat_exact"])
        coarse = r.get("repeat_coarse")
        coarse = float(coarse) if pd.notna(coarse) and str(coarse).strip() != "" else np.nan
        est = (exact - pe) / (1.0 - pe) if np.isfinite(pe) and pe < 1 else np.nan
        true_kappa, kappa_pairs, pe_pairs, exact_pairs = np.nan, 0, np.nan, np.nan
        if retest is not None:
            g = retest[retest["character"] == ch]
            if len(g):
                aa, bb = g["state_1"].values, g["state_2"].values
                true_kappa = cohen_kappa(aa, bb)
                kappa_pairs = int(len(g))
                exact_pairs = float((aa == bb).mean())
                sts = sorted(set(aa) | set(bb), key=str)
                pe_pairs = float(sum((aa == s).mean() * (bb == s).mean() for s in sts))
        # what the panel draws as "expected by chance": the true marginal p_e when the
        # paired readings exist, otherwise the first-reading estimate
        drawn_chance = pe_pairs if np.isfinite(pe_pairs) else pe
        drawn_corrected = true_kappa if np.isfinite(true_kappa) else est
        per = rep.get("per_character", {}).get(ch, {})
        rows.append({
            "character": ch,
            "type": r["type"],
            "states_in_scale": int(r["states"]),
            "states_observed": int(st.nunique()),
            "cells_scored": int(len(st)),
            "cells_not_assessable": int((first["character"] == ch).sum()) - int(len(st)),
            "n_compared": int(r["n_compared"]),
            "exact_agreement": exact,
            "within_one_step": float(r["repeat_within_one_step"]),
            "mean_steps": per.get("mean_steps"),
            "band_agreement": coarse,
            "chance_agreement_expected": round(drawn_chance, 4),
            "chance_corrected": round(drawn_corrected, 4) if np.isfinite(drawn_corrected) else np.nan,
            "chance_corrected_is_estimate": not np.isfinite(true_kappa),
            "chance_agreement_first_reading": round(pe, 4),
            "chance_corrected_estimate": round(est, 4) if np.isfinite(est) else np.nan,
            "cohen_kappa_true": round(true_kappa, 4) if np.isfinite(true_kappa) else np.nan,
            "exact_agreement_paired": round(exact_pairs, 4) if np.isfinite(exact_pairs) else np.nan,
            "kappa_pairs": kappa_pairs,
            "verdict": r["verdict"],
            "reason": r["reason"],
        })
    out = pd.DataFrame(rows).sort_values("exact_agreement", ascending=False)
    header = {
        "specimens_retested": rep.get("specimens"),
        "cells_compared": rep.get("cells_compared"),
        "cells_scored_total": int(len(scored)),
        "cells_not_assessable_total": int(len(first) - len(scored)),
        "pooled_exact_agreement": rep.get("exact_agreement"),
        "paired_readings_available": retest is not None,
    }
    return out, header


# ─────────────────────────────────────────────────────────────────────────────
# experiment 1, continued — the retest repeated properly: three readings
# ─────────────────────────────────────────────────────────────────────────────
# The paired second readings that were once lost have been re-run. There are now
# three readings of the same images: a first reading (18 Sep) and two readings
# made on the same later day (A and B, 20 Sep). That splits repeatability in two:
# WITHIN a session (A vs B, same day) and BETWEEN sessions (first vs A, two days
# apart). Both are true Cohen's kappa, so the old chance ESTIMATE is not needed
# whenever these files are present.

WITHIN_PAIR = "A_20Sep ~ B_20Sep"
BETWEEN_PAIR = "first_18Sep ~ A_20Sep"


def drift_arrow(share, kind):
    """Which way the second reading moved, among the disagreements.

    share = share of disagreements in which the second reading sat HIGHER on the
    ordinal scale. 0.5 is noise; near 0 or 1 is a shift. Nominal scales have no
    direction and get nothing.
    """
    if str(kind).strip().lower() == "nominal":
        return ""
    try:
        s = float(share)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(s):
        return ""
    if s <= 0.25:
        return "↓"
    if s >= 0.75:
        return "↑"
    return "↔"


def load_retest_sessions(retest_dir, within_pair=WITHIN_PAIR, between_pair=BETWEEN_PAIR):
    """Per-character and pooled within-/between-session agreement, or None.

    Returns None when the directory or either TSV is missing, or when the two
    pairs are not both in it — panel a then falls back, untouched, to the older
    single-retest view with its estimated chance correction.
    """
    if not retest_dir:
        return None
    d = Path(retest_dir)
    f_char, f_pool = (d / "retest_agreement_by_character.tsv",
                      d / "retest_agreement_pooled.tsv")
    if not (f_char.exists() and f_pool.exists()):
        return None
    ch = pd.read_csv(f_char, sep="\t")
    po = pd.read_csv(f_pool, sep="\t")
    if not {"pair", "character", "exact_agreement"} <= set(ch.columns):
        return None
    pairs = set(ch["pair"].astype(str))
    if within_pair not in pairs or between_pair not in pairs:
        return None

    w = ch[ch["pair"] == within_pair].drop_duplicates("character").set_index("character")
    b = ch[ch["pair"] == between_pair].drop_duplicates("character").set_index("character")
    rows = []
    for c in b.index:
        if c not in w.index:
            continue
        rw, rb = w.loc[c], b.loc[c]
        kind = str(rb.get("type", ""))
        share = rb.get("share_second_reading_higher", np.nan)
        rows.append({
            "character": c,
            "type": kind,
            "exact_within": float(rw["exact_agreement"]),
            "exact_between": float(rb["exact_agreement"]),
            "kappa_within": float(rw.get("cohen_kappa", np.nan)),
            "kappa_between": float(rb.get("cohen_kappa", np.nan)),
            "weighted_kappa_between": float(rb.get("weighted_kappa", np.nan)),
            "within_one_step_between": float(rb.get("within_one_step", np.nan)),
            "share_second_reading_higher_between": float(share) if pd.notna(share) else np.nan,
            "cells_within": int(rw.get("cells", 0)),
            "cells_between": int(rb.get("cells", 0)),
            "drift": drift_arrow(share, kind),
        })
    if not rows:
        return None
    per = pd.DataFrame(rows).sort_values("exact_between", ascending=False).reset_index(drop=True)

    def _pooled(pair):
        g = po[po["pair"] == pair]
        if not len(g):
            return {}
        r = g.iloc[0]
        return {"pair": pair,
                "cells": int(r["cells"]),
                "specimens": int(r["specimens"]) if pd.notna(r.get("specimens")) else None,
                "exact_agreement": float(r["exact_agreement"]),
                "mean_cohen_kappa": float(r["mean_cohen_kappa"])}

    return {"mode": "two",
            "per_character": per,
            "within": _pooled(within_pair),
            "between": _pooled(between_pair),
            "within_pair": within_pair,
            "between_pair": between_pair,
            "source": str(d)}


# The retest has since been run FIVE times: a first reading (18 Sep), two samples
# read on 20 Sep (A, B) and a complete second reading on 21 Sep with a sample
# re-read inside that same session. The 18 Sep reading stands alone; every later
# pair agrees, a day apart as well as within one session. Three series therefore
# say it: same session, one day apart, and across the step between 18 and 20 Sep.

THREE_WITHIN_PAIR = "second_21Sep ~ second_21Sep_again"
THREE_DAY_PAIR = "second_21Sep ~ A_20Sep"
THREE_STEP_PAIR = "first_18Sep ~ second_21Sep"


def load_retest_sessions_three(retest_dir, within_pair=THREE_WITHIN_PAIR,
                               day_pair=THREE_DAY_PAIR, step_pair=THREE_STEP_PAIR):
    """Three series — same session, one day apart, across the step — or None.

    Returns None whenever the files or any of the three pairs are missing, so the
    caller can fall back to the two-series reading of the older folder.
    """
    if not retest_dir:
        return None
    d = Path(retest_dir)
    f_char, f_pool = (d / "retest_agreement_by_character.tsv",
                      d / "retest_agreement_pooled.tsv")
    if not (f_char.exists() and f_pool.exists()):
        return None
    ch = pd.read_csv(f_char, sep="\t")
    po = pd.read_csv(f_pool, sep="\t")
    if not {"pair", "character", "exact_agreement"} <= set(ch.columns):
        return None
    pairs = set(ch["pair"].astype(str))
    if not {within_pair, day_pair, step_pair} <= pairs:
        return None

    def _by_char(pair):
        return ch[ch["pair"] == pair].drop_duplicates("character").set_index("character")

    w, dy, st = _by_char(within_pair), _by_char(day_pair), _by_char(step_pair)
    rows = []
    for c in st.index:
        if c not in w.index or c not in dy.index:
            continue
        rw, rd, rs = w.loc[c], dy.loc[c], st.loc[c]
        kind = str(rs.get("type", ""))
        share = rs.get("share_second_reading_higher", np.nan)
        rows.append({
            "character": c,
            "type": kind,
            "exact_within": float(rw["exact_agreement"]),
            "exact_day": float(rd["exact_agreement"]),
            "exact_step": float(rs["exact_agreement"]),
            "kappa_within": float(rw.get("cohen_kappa", np.nan)),
            "kappa_day": float(rd.get("cohen_kappa", np.nan)),
            "kappa_step": float(rs.get("cohen_kappa", np.nan)),
            "weighted_kappa_step": float(rs.get("weighted_kappa", np.nan)),
            "within_one_step_step": float(rs.get("within_one_step", np.nan)),
            "share_second_reading_higher_step": float(share) if pd.notna(share) else np.nan,
            "cells_within": int(rw.get("cells", 0)),
            "cells_day": int(rd.get("cells", 0)),
            "cells_step": int(rs.get("cells", 0)),
            "drift": drift_arrow(share, kind),
        })
    if not rows:
        return None
    # ordered by the agreement that matters: one day apart
    per = pd.DataFrame(rows).sort_values("exact_day", ascending=False).reset_index(drop=True)

    def _pooled(pair):
        g = po[po["pair"] == pair]
        if not len(g):
            return {}
        r = g.iloc[0]
        return {"pair": pair,
                "cells": int(r["cells"]),
                "specimens": int(r["specimens"]) if pd.notna(r.get("specimens")) else None,
                "exact_agreement": float(r["exact_agreement"]),
                "mean_cohen_kappa": float(r["mean_cohen_kappa"])}

    return {"mode": "three",
            "per_character": per,
            "within": _pooled(within_pair),
            "day": _pooled(day_pair),
            "step": _pooled(step_pair),
            "between": _pooled(step_pair),          # legacy alias: across the step
            "within_pair": within_pair,
            "day_pair": day_pair,
            "step_pair": step_pair,
            "between_pair": step_pair,
            "source": str(d)}


# ─────────────────────────────────────────────────────────────────────────────
# experiment 2 — free-form character search
# ─────────────────────────────────────────────────────────────────────────────

def load_freeform(a):
    first = pd.read_csv(a.freeform_first, sep="\t")
    first["structure"] = first["character"].str.split(":").str[0]

    orig = []
    for p in a.state_retest:
        d = pd.read_csv(p, sep="\t")[["specimen_id", "character", "state"]]
        d["retest"] = Path(p).parent.name
        orig.append(d)
    orig = pd.concat(orig, ignore_index=True) if orig else pd.DataFrame(
        columns=["specimen_id", "character", "state", "retest"])

    loc = pd.read_csv(a.location_retest, sep="\t")
    loc_states = loc[["specimen_id", "character", "state"]].copy()
    loc_states["retest"] = "location"

    pairs = pd.concat([orig, loc_states], ignore_index=True)
    paired = first[["specimen_id", "character", "state"]].merge(
        pairs, on=["specimen_id", "character"], suffixes=("_1", "_2"))

    tab = []
    for p in a.state_reliability:
        tab.append(pd.read_csv(p, sep="\t"))
    tab = pd.concat(tab, ignore_index=True) if tab else pd.DataFrame(
        columns=["character", "retest_agreement"])
    return first, loc, paired, tab


def relocation_pairs(first, loc, present):
    """Two readings of the SAME image: no orientation mapping is applied."""
    m = first.merge(loc, on=["specimen_id", "character"], suffixes=("_1", "_2"))
    both = m[(m["state_1"] == present) & (m["state_2"] == present)].dropna(
        subset=["x_1", "y_1", "x_2", "y_2"]).copy()
    both["distance"] = np.hypot(both["x_1"] - both["x_2"], both["y_1"] - both["y_2"])
    cols = ["specimen_id", "species", "structure", "character",
            "x_1", "y_1", "x_2", "y_2", "distance"]
    return both[[c for c in cols if c in both.columns]].sort_values(
        ["character", "specimen_id"]).reset_index(drop=True)


def oriented_reports(first, orientation_tsv, present):
    """Present, located first-reading reports with flipped frames turned round."""
    fo = pd.read_csv(orientation_tsv, sep="\t")
    flip = dict(zip(zip(fo["structure"], fo["specimen_id"]), fo["flipped"].astype(int)))
    p = first[first["state"] == present].dropna(subset=["x", "y"]).copy()
    f = np.array([flip.get((s, i), 0) for s, i in zip(p["structure"], p["specimen_id"])])
    p["flipped"] = f
    p["xo"] = np.where(f == 1, 1.0 - p["x"], p["x"])
    p["yo"] = np.where(f == 1, 1.0 - p["y"], p["y"])
    return p


def between_specimen_distances(p, rng, max_pairs_per_character=60):
    """Same character, two different specimens, orientation applied."""
    out = []
    for _, g in p.groupby("character"):
        g = g.drop_duplicates("specimen_id")
        n = len(g)
        if n < 2:
            continue
        k = int(min(max_pairs_per_character, n * (n - 1) // 2))
        idx = rng.integers(0, n, size=(k * 6, 2))
        idx = idx[idx[:, 0] != idx[:, 1]][:k]
        if not len(idx):
            continue
        aa, bb = g.iloc[idx[:, 0]], g.iloc[idx[:, 1]]
        out.append(np.hypot(aa["xo"].values - bb["xo"].values,
                            aa["yo"].values - bb["yo"].values))
    return np.concatenate(out) if out else np.array([])


def null_distances(p, rng, max_pairs_per_structure=600):
    """Null: two reports of DIFFERENT characters on the same structure."""
    out = []
    for _, g in p.groupby("structure"):
        n = len(g)
        if n < 2:
            continue
        k = int(min(max_pairs_per_structure, n))
        idx = rng.integers(0, n, size=(k * 3, 2))
        ch = g["character"].values
        idx = idx[(idx[:, 0] != idx[:, 1]) & (ch[idx[:, 0]] != ch[idx[:, 1]])][:k]
        if not len(idx):
            continue
        xo, yo = g["xo"].values, g["yo"].values
        out.append(np.hypot(xo[idx[:, 0]] - xo[idx[:, 1]], yo[idx[:, 0]] - yo[idx[:, 1]]))
    return np.concatenate(out) if out else np.array([])


def freeform_table(a, first, paired, tab, reloc):
    hi = pd.read_csv(a.heatmap_index, sep="\t")
    hi = hi[hi["category"] != "pooled"].copy()
    hi["key"] = hi["structure"] + ":" + hi["character"]
    hi = hi.set_index("key")
    ami = pd.read_csv(a.informativeness, sep="\t").set_index("character")
    fx = pd.read_csv(a.fixed_within_species, sep="\t")
    fixed_ok = set(fx.loc[fx["outside_rate"] <= a.max_outside_rate, "character"])
    fixed_any = set(fx["character"])
    tabmap = tab.set_index("character")["retest_agreement"].to_dict()

    rn = reloc.groupby("character")["distance"].agg(["size", "median"])
    n_states = first.groupby("character")["state"].nunique()
    n_spec = first.groupby("character")["specimen_id"].nunique()
    present_share = first.groupby("character")["state"].apply(
        lambda s: float((s == a.present_label).mean()))

    rows = []
    for ch in sorted(first["character"].unique()):
        g = paired[paired["character"] == ch]
        npair = int(len(g))
        agr = float((g["state_1"] == g["state_2"]).mean()) if npair else np.nan
        kap = cohen_kappa(g["state_1"].values, g["state_2"].values) if npair else np.nan
        h = hi.loc[ch] if ch in hi.index else None
        nrel = int(rn.loc[ch, "size"]) if ch in rn.index else 0
        mrel = float(rn.loc[ch, "median"]) if ch in rn.index else np.nan
        rows.append({
            "character": ch,
            "structure": ch.split(":")[0],
            "n_specimens": int(n_spec.get(ch, 0)),
            "present_share": round(float(present_share.get(ch, np.nan)), 4),
            "n_states_observed": int(n_states.get(ch, 0)),
            "n_pairs": npair,
            "agreement": round(agr, 4) if np.isfinite(agr) else np.nan,
            "retest_agreement_tabulated": tabmap.get(ch, np.nan),
            "cohen_kappa": round(kap, 4) if np.isfinite(kap) else np.nan,
            "kappa_defined": bool(np.isfinite(kap)),
            "n_located_reports": int(h["n_reports"]) if h is not None else 0,
            "heatmap_category": h["category"] if h is not None else "no_heatmap",
            "spread": float(h["spread"]) if h is not None else np.nan,
            "offset": float(h["offset"]) if h is not None else np.nan,
            "share_far_outside": float(h["share_far_outside"]) if h is not None else np.nan,
            "n_relocated_pairs": nrel,
            "median_relocation_distance": round(mrel, 4) if np.isfinite(mrel) else np.nan,
            "adjusted_mutual_information": float(ami["adjusted_mutual_information"].get(ch, np.nan)),
            "fixed_in_a_species": ch in fixed_any,
            "fixed_and_rare_outside": ch in fixed_ok,
        })
    df = pd.DataFrame(rows)

    # funnel flags — each one is the condition of its own step, not cumulative
    df["f_both_states"] = df["n_states_observed"] >= 2
    df["f_agreement"] = df["agreement"] >= a.min_agreement
    df["f_kappa"] = (df["cohen_kappa"] >= a.min_kappa) | (~df["kappa_defined"])
    df["f_kappa_undefined"] = ~df["kappa_defined"]
    df["f_enough_located"] = (df["heatmap_category"] != "too_few_reports") & \
                             (df["heatmap_category"] != "no_heatmap")
    df["f_tight"] = df["spread"] < a.max_spread
    df["f_on_specimen"] = df["share_far_outside"] <= a.max_far_outside
    df["f_relocation_testable"] = df["n_relocated_pairs"] >= a.min_relocated_pairs
    df["f_relocates"] = np.where(df["f_relocation_testable"],
                                 df["median_relocation_distance"] <= a.max_relocation,
                                 True)
    df["f_informative"] = df["adjusted_mutual_information"] >= a.min_ami
    df["f_fixed"] = df["fixed_and_rare_outside"]
    return df


def build_attrition(df, a, confirmed):
    steps = [
        ("proposed", None, None, None),
        ("both states occur among specimens", "f_both_states", None, None),
        (f"state repeats (agreement ≥ {a.min_agreement:.2f})", "f_agreement", None, None),
        (f"chance-corrected (κ ≥ {a.min_kappa:.2f})", "f_kappa",
         "f_kappa_undefined", "κ undefined"),
        ("enough located reports (category ≠ too few)", "f_enough_located", None, None),
        (f"tight cluster (spread < {a.max_spread:.2f})", "f_tight", None, None),
        (f"on the specimen (far-outside ≤ {a.max_far_outside:.2f})",
         "f_on_specimen", None, None),
        (f"relocates on the same image (median ≤ {a.max_relocation:.2f})",
         "f_relocates", "f_relocation_untestable", "not testable"),
        (f"informative about species (AMI ≥ {a.min_ami:.2f})", "f_informative", None, None),
        (f"fixed in a species, ≤ {a.max_outside_rate:.0%} outside", "f_fixed", None, None),
        ("confirmed autapomorphy", "__autapomorphy__", None, None),
    ]
    df = df.copy()
    df["f_relocation_untestable"] = ~df["f_relocation_testable"]
    alive = pd.Series(True, index=df.index)
    rows = []
    for label, flag, hatch_flag, hatch_label in steps:
        if flag is None:
            keep = alive.copy()
        elif flag == "__autapomorphy__":
            keep = pd.Series(False, index=df.index)
            n_keep = int(confirmed)
        else:
            keep = alive & df[flag].fillna(False)
        lost = int(alive.sum() - keep.sum())
        n_keep = int(confirmed) if flag == "__autapomorphy__" else int(keep.sum())
        n_hatch = int((keep & df[hatch_flag]).sum()) if hatch_flag else 0
        rows.append({
            "step": label, "surviving": n_keep, "lost_here": lost,
            "carried_untested": n_hatch,
            "untested_label": hatch_label or "",
            "characters": "; ".join(sorted(df.loc[keep, "character"])) if 0 < n_keep <= 12 else "",
        })
        alive = keep
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────────

def band(fig, x0, x1, y, text, colour, textcolour="white"):
    fig.add_artist(Rectangle((x0, y), x1 - x0, 0.0112, transform=fig.transFigure,
                             facecolor=colour, edgecolor="none", zorder=5, clip_on=False))
    fig.text((x0 + x1) / 2, y + 0.0056, text, transform=fig.transFigure, ha="center",
             va="center", fontsize=7.2, color=textcolour, zorder=6, fontweight="bold")


def capped_hist(ax, values, bins, colour, cap_factor=1.45, min_cap=12):
    """Histogram whose tallest bar is allowed to run off the top, marked and labelled."""
    counts, edges = np.histogram(values, bins=bins)
    ax.hist(values, bins=bins, color=colour, edgecolor="white", linewidth=0.5)
    if not len(counts):
        return
    order = np.sort(counts)[::-1]
    second = order[1] if len(order) > 1 else order[0]
    cap = max(second * cap_factor, min_cap)
    if order[0] <= cap:
        ax.set_ylim(0, order[0] * 1.30)
        return
    ax.set_ylim(0, cap)
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        if c <= cap:
            continue
        mid = (lo + hi) / 2
        w = (hi - lo)
        ax.add_patch(Rectangle((lo, cap * 0.755), w, cap * 0.070, facecolor="white",
                               edgecolor="none", zorder=6))
        for dx in (-w * 0.22, w * 0.22):
            ax.plot([mid + dx - w * 0.12, mid + dx + w * 0.12],
                    [cap * 0.762, cap * 0.828], color="0.45", lw=0.8, zorder=7,
                    solid_capstyle="round")
        ax.text(mid, cap * 0.862, f"{int(c)}", ha="center", va="bottom", fontsize=6.5,
                color="0.22", fontweight="bold", zorder=7)


def panel_a_sessions(axa, s):
    """Panel a drawn from the three readings: within a session vs between sessions.

    One row per character, ordered by between-session exact agreement. The line
    between the two markers IS the drop, and the small table on the right gives
    Cohen's kappa for each comparison and which way the second reading moved.
    """
    def _k(v):
        return "–" if not np.isfinite(v) else f"{v:+.2f}"

    per = s["per_character"].sort_values("exact_between", ascending=True).reset_index(drop=True)
    n = len(per)
    X_N, X_KW, X_KB, X_AR = 1.045, 1.160, 1.272, 1.355
    for i, r in per.iterrows():
        axa.plot([r["exact_between"], r["exact_within"]], [i, i], color="0.60", lw=1.0,
                 zorder=1, solid_capstyle="round")
        axa.plot(r["exact_between"], i, marker="D", ms=5.0, mfc=BETWEEN_COLOUR,
                 mec=BETWEEN_EDGE, mew=0.8, zorder=4)
        axa.plot(r["exact_within"], i, marker="o", ms=5.8, mfc=WITHIN_COLOUR,
                 mec=WITHIN_EDGE, mew=0.8, zorder=5)
        axa.text(X_N, i, f"{int(r['cells_within'])} / {int(r['cells_between'])}",
                 va="center", ha="center", fontsize=6.2, color="0.40")
        axa.text(X_KW, i, _k(r["kappa_within"]), va="center", ha="center",
                 fontsize=6.4, color=WITHIN_COLOUR)
        axa.text(X_KB, i, _k(r["kappa_between"]), va="center", ha="center",
                 fontsize=6.4, color=BETWEEN_EDGE)
        axa.text(X_AR, i, r["drift"], va="center", ha="center", fontsize=8.0,
                 color="0.30")
    for x, lab in ((X_N, "cells"), (X_KW, "κ same day"),
                   (X_KB, "κ 2 days"), (X_AR, "drift")):
        axa.text(x, n - 0.42, lab, va="bottom", ha="center", fontsize=6.1,
                 color="0.35", fontstyle="italic")
    axa.axvline(0.85, color="0.25", lw=0.85, ls="--", zorder=0)
    axa.text(0.845, -0.62, "85% exact", fontsize=6.2, color="0.25", va="center", ha="right")
    axa.set_yticks(np.arange(n))
    axa.set_yticklabels(per["character"])
    axa.set_ylim(-1.05, n + 0.05)
    axa.set_xlim(0.30, 1.405)
    axa.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axa.spines["bottom"].set_bounds(0.30, 1.0)
    axa.set_xlabel("exact agreement between two readings of the same image")
    axa.set_title("a  Descriptive categorical characters: the same image read again — "
                  "the same day, and two days later", loc="left", pad=8)

    w, b = s["within"], s["between"]
    handles = [Line2D([], [], marker="o", ls="none", mfc=WITHIN_COLOUR, mec=WITHIN_EDGE,
                      ms=5.8, label="the same day (two readings, 20 Sep)"),
               Line2D([], [], marker="D", ls="none", mfc=BETWEEN_COLOUR, mec=BETWEEN_EDGE,
                      ms=5.0, label="two days apart (18 Sep vs 20 Sep)")]
    axa.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.155),
               frameon=False, fontsize=6.4, ncol=2, handletextpad=0.55,
               columnspacing=2.4, labelspacing=0.36)
    axa.text(0.0, -0.265,
             f"within a session:   {w['cells']:,} cells on {w['specimens']} specimens, "
             f"{w['exact_agreement']:.1%} exact, mean Cohen's κ {w['mean_cohen_kappa']:.3f}\n"
             f"between sessions:  {b['cells']:,} cells on {b['specimens']} specimens, "
             f"{b['exact_agreement']:.1%} exact, mean Cohen's κ {b['mean_cohen_kappa']:.3f}\n"
             "drift, among the disagreements on an ordinal scale: ↑ the second reading "
             "usually sat higher (≥ 75% of them), ↓ usually lower (≤ 25%), "
             "↔ both ways; nominal scales blank.",
             transform=axa.transAxes, ha="left", va="top", fontsize=6.1,
             color="0.30", linespacing=1.55)


def panel_a_sessions_three(axa, s):
    """Panel a from five readings: same session, one day apart, across the step.

    Three markers per character joined by one line. Ordered by the middle series,
    one day apart, because that is the reproducibility that matters. The right-hand
    table gives Cohen's kappa for each of the three and, from the across-the-step
    comparison alone, which way the later reading moved on the scale.
    """
    def _k(v):
        return "–" if not np.isfinite(v) else f"{v:+.2f}"

    per = s["per_character"].sort_values("exact_day", ascending=True).reset_index(drop=True)
    n = len(per)
    X_N, X_K1, X_K2, X_K3, X_AR = 1.048, 1.152, 1.257, 1.362, 1.448
    for i, r in per.iterrows():
        xs = [r["exact_step"], r["exact_day"], r["exact_within"]]
        axa.plot([min(xs), max(xs)], [i, i], color="0.60", lw=1.0, zorder=1,
                 solid_capstyle="round")
        axa.plot(r["exact_step"], i, marker="D", ms=5.0, mfc=BETWEEN_COLOUR,
                 mec=BETWEEN_EDGE, mew=0.8, zorder=4)
        axa.plot(r["exact_day"], i, marker="s", ms=5.0, mfc=DAY_COLOUR,
                 mec=DAY_EDGE, mew=0.8, zorder=5)
        axa.plot(r["exact_within"], i, marker="o", ms=5.6, mfc=WITHIN_COLOUR,
                 mec=WITHIN_EDGE, mew=0.8, zorder=6)
        axa.text(X_N, i, f"{int(r['cells_step']):,}", va="center", ha="center",
                 fontsize=6.2, color="0.40")
        for x, v, col in ((X_K1, r["kappa_within"], WITHIN_COLOUR),
                          (X_K2, r["kappa_day"], DAY_EDGE),
                          (X_K3, r["kappa_step"], BETWEEN_EDGE)):
            axa.text(x, i, _k(v), va="center", ha="center", fontsize=6.4, color=col)
        axa.text(X_AR, i, r["drift"], va="center", ha="center", fontsize=8.0, color="0.30")
    for x, lab in ((X_N, "cells"), (X_K1, "κ session"), (X_K2, "κ a day"),
                   (X_K3, "κ the step"), (X_AR, "drift")):
        axa.text(x, n - 0.42, lab, va="bottom", ha="center", fontsize=6.1,
                 color="0.35", fontstyle="italic")
    axa.axvline(0.85, color="0.25", lw=0.85, ls="--", zorder=0)
    axa.text(0.845, -0.62, "85% exact", fontsize=6.2, color="0.25", va="center", ha="right")
    axa.set_yticks(np.arange(n))
    axa.set_yticklabels(per["character"])
    axa.set_ylim(-1.05, n + 0.05)
    axa.set_xlim(0.36, 1.492)
    axa.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axa.spines["bottom"].set_bounds(0.40, 1.0)
    axa.set_xlabel("exact agreement between two readings of the same image")
    axa.set_title("a  Descriptive categorical characters: the same image read again — in the "
                  "same session, a day apart, and across a change of the model's client",
                  loc="left", pad=8, fontsize=7.2)

    w, dy, st = s["within"], s["day"], s["step"]
    handles = [Line2D([], [], marker="o", ls="none", mfc=WITHIN_COLOUR, mec=WITHIN_EDGE,
                      ms=5.6, label="same session (21 Sep, read and re-read)"),
               Line2D([], [], marker="s", ls="none", mfc=DAY_COLOUR, mec=DAY_EDGE,
                      ms=5.0, label="one day apart (20 vs 21 Sep)"),
               Line2D([], [], marker="D", ls="none", mfc=BETWEEN_COLOUR, mec=BETWEEN_EDGE,
                      ms=5.0, label="across the change of client (18 vs 21 Sep)")]
    axa.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.145),
               frameon=False, fontsize=6.4, ncol=3, handletextpad=0.55,
               columnspacing=1.9, labelspacing=0.36)
    axa.text(0.0, -0.240,
             f"same session:              {w['cells']:,} cells on {w['specimens']} specimens, "
             f"{w['exact_agreement']:.1%} exact, mean Cohen's κ {w['mean_cohen_kappa']:.3f}\n"
             f"one day apart:             {dy['cells']:,} cells on {dy['specimens']} specimens, "
             f"{dy['exact_agreement']:.1%} exact, mean Cohen's κ {dy['mean_cohen_kappa']:.3f}\n"
             f"across the step:         {st['cells']:,} cells on {st['specimens']} specimens, "
             f"{st['exact_agreement']:.1%} exact, mean Cohen's κ {st['mean_cohen_kappa']:.3f}\n"
             "drift (across the step, ordinal characters only): ↑ the later reading usually "
             "sat higher (≥ 75% of the disagreements), ↓ usually lower (≤ 25%), "
             "↔ both ways; nominal blank.",
             transform=axa.transAxes, ha="left", va="top", fontsize=5.9,
             color="0.30", linespacing=1.46)


def _ami_note(attr):
    """The example sentence of panel d, read from the attrition table so that it cannot go stale."""
    try:
        st = attr["step"].astype(str)
        i = attr.index[st.str.startswith("informative about species")][0]
        lost, before = int(attr.loc[i, "lost_here"]), int(attr.loc[i - 1, "surviving"])
        return (f"{lost} lost to AMI are {lost} of the {before} that got that far, "
                f"not of the {int(attr['surviving'].iloc[0])} proposed")
    except Exception:
        return "characters lost at a late step are lost from the few that got that far, not from all proposed"


def draw_figure(a, e1, e1h, ff, reloc, attr, dists, res_step, kstats, out_png, out_pdf,
                sessions=None):
    plt.rcParams.update({"font.size": 7.4, "axes.titlesize": 9.0, "axes.labelsize": 7.4,
                         "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                         "axes.titleweight": "bold", "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(10.2, 12.4))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.92, 1.00, 1.22],
                  width_ratios=[1.00, 1.12],
                  left=0.155, right=0.958, top=0.912, bottom=0.072,
                  wspace=0.26, hspace=0.62)

    # ---- panel a ------------------------------------------------------------
    axa = fig.add_subplot(gs[0, :])
    tidy(axa)
    if sessions is not None and sessions.get("mode") == "three":
        panel_a_sessions_three(axa, sessions)
    elif sessions is not None:
        panel_a_sessions(axa, sessions)
    else:
        d = e1.sort_values("exact_agreement", ascending=True).reset_index(drop=True)
        for i, r in d.iterrows():
            v = r["verdict"]
            c, ec, mk = VERDICT_COLOUR[v], VERDICT_EDGE[v], VERDICT_MARKER[v]
            axa.plot([r["chance_agreement_expected"], r["exact_agreement"]], [i, i],
                     color=c, lw=1.0, alpha=0.5, zorder=1, solid_capstyle="round")
            if np.isfinite(r["band_agreement"]):
                axa.plot(r["band_agreement"], i, marker=mk, ms=6.4, mfc="white",
                         mec=ec, mew=1.3, zorder=3)
            axa.plot(r["exact_agreement"], i, marker=mk, ms=5.6, mfc=c, mec=ec,
                     mew=0.8, zorder=4)
            axa.plot([r["chance_agreement_expected"]] * 2, [i - 0.28, i + 0.28],
                     color=INK, lw=1.6, zorder=5)
            axa.text(1.018, i, f"n = {int(r['n_compared'])}", va="center", ha="left",
                     fontsize=6.4, color="0.35")
        axa.axvline(0.85, color="0.25", lw=0.85, ls="--", zorder=0)
        axa.axvline(0.80, color="0.55", lw=0.85, ls=":", zorder=0)
        axa.text(0.852, -0.62, "85% exact", fontsize=6.2, color="0.25", va="center", ha="left")
        axa.text(0.796, -0.62, "80% band", fontsize=6.2, color="0.45", va="center", ha="right")
        axa.set_yticks(np.arange(len(d)))
        axa.set_yticklabels(d["character"])
        axa.set_ylim(-1.05, len(d) - 0.35)
        axa.set_xlim(0.05, 1.135)
        axa.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        axa.spines["bottom"].set_bounds(0.05, 1.0)
        axa.set_xlabel("agreement between two readings of the same image")
        axa.set_title("a  Descriptive categorical characters: does the same state come back?", loc="left",
                      pad=8)
        estimated = bool(e1["chance_corrected_is_estimate"].all())
        caveat = ("the chance value is an ESTIMATE, Σp² from the first-reading state "
                  "frequencies:\nthe paired second readings were not kept, so no true "
                  "Cohen's κ can be computed here"
                  if estimated else
                  "the chance value is the marginal Σp₁p₂ of the paired readings, so the "
                  "chance-corrected\nagreement reported alongside it is a true Cohen's κ")
        axa.text(0.010, 0.985,
                 f"{e1h['cells_scored_total']:,} cells scored over "
                 f"{len(e1)} characters; {e1h['cells_compared']:,} of them read twice on "
                 f"{e1h['specimens_retested']} specimens "
                 f"(pooled exact agreement {e1h['pooled_exact_agreement']:.3f})\n" + caveat,
                 transform=axa.transAxes, ha="left", va="top", fontsize=6.1,
                 color="0.30", linespacing=1.5)
        handles = [Line2D([], [], marker="o", ls="none", mfc=INK, mec=INK, ms=5.6,
                          label="exact state repeats"),
                   Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, mew=1.3,
                          ms=6.4, label="band of 2–3 states repeats"),
                   Line2D([], [], marker="|", ls="none", color=INK, ms=8, mew=1.6,
                          label="expected by chance" + (" (estimate)" if estimated else ""))]
        handles += [Line2D([], [], marker=VERDICT_MARKER[v], ls="none",
                           mfc=VERDICT_COLOUR[v], mec=VERDICT_EDGE[v], ms=5.6,
                           label=VERDICT_LABEL[v]) for v in ("use", "use coarse", "flag")]
        axa.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.215),
                   frameon=False, fontsize=6.4, ncol=3, handletextpad=0.55,
                   columnspacing=2.2, labelspacing=0.36)

    # ---- panel b ------------------------------------------------------------
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.34)
    axb1 = fig.add_subplot(sub[0])
    axb2 = fig.add_subplot(sub[1], sharex=axb1)
    tidy(axb1); tidy(axb2)
    agr = ff["agreement"].dropna().values
    kap = ff["cohen_kappa"].dropna().values
    n_und = int((~ff["kappa_defined"]).sum())
    n_chr = int(len(ff))
    slot = -0.335                                    # parking slot for undefined kappa
    bins = np.round(np.arange(-0.25, 1.001, 0.05), 6)
    bins[-1] += 1e-9                                 # keep values of exactly 1.0 in

    capped_hist(axb1, agr, bins, TEAL)
    axb1.axvline(a.min_agreement, color=INK, lw=0.9, ls="--")
    axb1.set_ylabel("characters")
    axb1.text(0.015, 0.97, "raw agreement", transform=axb1.transAxes, fontsize=7.2,
              va="top", fontweight="bold", color=TEAL_DARK)
    axb1.text(0.015, 0.80,
              f"median {np.median(agr):.3f}\n"
              f"{int((ff['agreement'] >= a.min_agreement).sum())} of {n_chr} "
              f"≥ {a.min_agreement:.2f}",
              transform=axb1.transAxes, fontsize=6.4, va="top", color="0.28",
              linespacing=1.5)

    capped_hist(axb2, kap, bins, TEAL_LIGHT)
    ycap = axb2.get_ylim()[1]
    axb2.bar([slot], [min(n_und, ycap)], width=0.085, color=TEAL_PALE,
             edgecolor=TEAL_DARK, linewidth=0.7, hatch="////")
    axb2.axvline(a.min_kappa, color=INK, lw=0.9, ls="--")
    axb2.vlines(slot + 0.072, 0, ycap * 0.40, color="0.72", lw=0.7, ls=(0, (2, 2)))
    axb2.set_ylabel("characters")
    axb2.set_xlabel("value over the 194 model-proposed characters")
    axb2.text(0.015, 0.97, "Cohen's κ — the same pairs, chance removed",
              transform=axb2.transAxes, fontsize=7.2, va="top", fontweight="bold",
              color=TEAL_DARK)
    axb2.text(0.015, 0.80,
              f"median {np.median(kap):.3f}\n"
              f"{int((ff['cohen_kappa'] >= a.min_kappa).sum())} of {n_chr} "
              f"≥ {a.min_kappa:.2f}",
              transform=axb2.transAxes, fontsize=6.4, va="top", color="0.28",
              linespacing=1.5)
    axb2.annotate(f"κ undefined: only one\nstate ever used, n = {n_und}",
                  xy=(slot + 0.048, min(n_und, ycap) * 0.62),
                  xytext=(slot + 0.150, ycap * 0.44), fontsize=6.0, color=TEAL_DARK,
                  ha="left", va="top", linespacing=1.4,
                  arrowprops=dict(arrowstyle="-", lw=0.6, color=TEAL_DARK))
    axb1.set_xlim(slot - 0.085, 1.03)
    axb1.set_title("b  Model-proposed binary characters: state calls", loc="left", pad=8)
    n_lo_agr = int((ff["agreement"] < a.min_agreement).sum())
    n_lo_kap = int((ff["cohen_kappa"] < a.min_kappa).sum())
    axb2.text(0.0, -0.40,
              f"{kstats['n_pairs']:,} paired readings over {n_chr} characters;\n"
              f"pooled agreement {kstats['pooled_agreement']:.3f}, pooled κ "
              f"{kstats['pooled_kappa']:.3f}. Present/absent calls on\n"
              "rare states agree with themselves by chance, so raw\n"
              f"agreement flatters them: only {n_lo_agr} characters fall below\n"
              f"{a.min_agreement:.2f} agreement, but {n_lo_kap} fall below κ = {a.min_kappa:.2f} "
              f"and {n_und} more\nnever produced a second state to disagree about.",
              transform=axb2.transAxes, ha="left", va="top", fontsize=6.1,
              color="0.30", linespacing=1.55)

    # ---- panel c ------------------------------------------------------------
    axc = fig.add_subplot(gs[1, 1])
    tidy(axc)
    floor = res_step / 2.0
    series = [("same image, two readings", dists["same_image"], TEAL_DARK, "-", 1.8, (7, 8)),
              ("same character, two specimens", dists["between_specimen"], TEAL, "-", 1.5,
               (7, -12)),
              ("different characters, same structure", dists["null"], GREY, "--", 1.4, (7, 8))]
    for label, v, colour, ls, lw, off in series:
        x, f = ecdf(np.maximum(np.asarray(v, dtype=float), floor))
        axc.step(x, f, where="post", color=colour, lw=lw, ls=ls,
                 label=f"{label}  (n = {len(v):,})")
        med = max(float(np.median(v)), floor)
        axc.plot([med], [0.5], marker="o", ms=4.2, color=colour, mec="white",
                 mew=0.7, zorder=6)
        axc.annotate(f"{float(np.median(v)):.3f}", xy=(med, 0.5), xytext=off,
                     textcoords="offset points", ha="left", va="center", fontsize=6.6,
                     color=colour, fontweight="bold", zorder=7,
                     path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    axc.axvline(res_step, color="0.72", lw=1.0, zorder=0)
    axc.axhline(0.5, color="0.88", lw=0.7, zorder=0)
    axc.set_xscale("log")
    axc.set_xlim(floor * 0.85, 1.35)
    axc.set_ylim(0, 1.02)
    axc.set_xlabel("distance between two reported locations (frame units)")
    axc.set_ylabel("cumulative share of pairs")
    axc.set_title("c  Model-proposed binary characters: same place twice?", loc="left",
                  pad=8)
    leg = axc.legend(loc="lower right", bbox_to_anchor=(1.015, -0.022), frameon=True,
                     fontsize=6.3, handlelength=2.0, labelspacing=0.40, borderpad=0.35)
    leg.set_zorder(8)
    leg.get_frame().set(facecolor="white", edgecolor="none", alpha=1.0)
    axc.text(res_step * 0.86, 0.995,
             f"coordinate resolution {res_step:g} ≈ {res_step * 768:.0f} px of 768",
             fontsize=6.0, color="0.45", va="top", ha="right", rotation=90)
    axc.text(0.0, -0.215,
             f"A median at the grid step is the same answer twice: {np.mean(np.asarray(dists['same_image']) == 0):.0%} of same-image pairs\n"
             "repeat the coordinate exactly and are drawn at half a step. Frames that lie end for\n"
             "end are turned round before two specimens are compared; two readings of the same\n"
             "image are compared unmapped.",
             transform=axc.transAxes, ha="left", va="top", fontsize=6.1,
             color="0.30", linespacing=1.55)

    # ---- panel d ------------------------------------------------------------
    axd = fig.add_subplot(gs[2, :])
    tidy(axd)
    n = len(attr)
    ypos = np.arange(n)[::-1]
    top = float(attr["surviving"].iloc[0])
    for i, r in attr.iterrows():
        yy = ypos[i]
        solid = r["surviving"] - r["carried_untested"]
        axd.barh(yy, solid, height=0.60, color=TEAL, edgecolor=TEAL_DARK, linewidth=0.6)
        if r["carried_untested"]:
            axd.barh(yy, r["carried_untested"], left=solid, height=0.60,
                     color=TEAL_PALE, edgecolor=TEAL_DARK, linewidth=0.6, hatch="////")
            axd.text(r["surviving"] + 2.5, yy,
                     f"{int(r['carried_untested'])} {r['untested_label']}",
                     va="center", ha="left", fontsize=6.0, color=TEAL_DARK)
        if solid > top * 0.10:
            axd.text(solid - 2.5, yy, f"{int(r['surviving'])}", va="center",
                     ha="right", fontsize=7.0, color="white", fontweight="bold")
        else:
            axd.text(r["surviving"] + 2.5, yy, f"{int(r['surviving'])}", va="center",
                     ha="left", fontsize=7.0, color=INK, fontweight="bold")
        if i > 0 and r["lost_here"]:
            axd.text(top * 1.055, yy, f"−{int(r['lost_here'])}", va="center",
                     ha="left", fontsize=6.6, color="#a03030")
    axd.set_yticks(ypos)
    axd.set_yticklabels(attr["step"])
    axd.set_xlim(0, top * 1.14)
    axd.set_xticks(np.arange(0, top, 25))
    axd.spines["bottom"].set_bounds(0, top)
    axd.set_ylim(-0.75, n - 0.30)
    axd.set_xlabel("characters surviving")
    axd.set_title("d  What is left of 194 proposed characters", loc="left", pad=8)
    axd.text(top * 1.055, ypos[0], "lost here", va="center", ha="left", fontsize=6.4,
             color="#a03030", fontstyle="italic")
    axd.legend(handles=[Patch(facecolor=TEAL, edgecolor=TEAL_DARK, label="survives the step"),
                        Patch(facecolor=TEAL_PALE, edgecolor=TEAL_DARK, hatch="////",
                              label="carried forward, not testable at this step")],
               loc="upper left", bbox_to_anchor=(0.185, 0.275), frameon=False,
               fontsize=6.4, borderpad=0.1, labelspacing=0.4)
    axd.text(0.185, 0.155,
             "The steps are CUMULATIVE and applied in this order: each one is tested only on the survivors of the step above it, so\n"
             "how much a step costs depends on where it sits in the sequence — a character dropped for a loose cluster is never\n"
             "tested for informativeness, and the " + _ami_note(attr) + ".",
             transform=axd.transAxes, ha="left", va="top", fontsize=6.1, color="0.30",
             linespacing=1.55)

    pdp = axd.get_position()                     # panel d carries the long row labels
    axd.set_position([0.245, pdp.y0, pdp.x1 - 0.245, pdp.height])

    # ---- header bands and note ---------------------------------------------
    fig.canvas.draw()
    pa = axa.get_position(); pb = axb1.get_position(); pc = axc.get_position()
    band(fig, pa.x0, pa.x1, pa.y1 + 0.0295,
         "EXPERIMENT 1   ·   DESCRIPTIVE CATEGORICAL CHARACTERS (PROMPTED FROM A FIXED LIST)", "#12436D")
    band(fig, pb.x0, pc.x1, pb.y1 + 0.0295,
         "EXPERIMENT 2   ·   MODEL-PROPOSED BINARY CHARACTERS (FREE-FORM SEARCH)", TEAL)

    sessions_note = ""
    if sessions is not None and sessions.get("mode") == "three":
        sessions_note = "\nRepeatable within a session is not the same as reproducible: here the prompted characters reproduce from one day to the next and did not across a change of the model's client."
    elif sessions is not None:
        sessions_note = "\nRepeatable within a session is not the same as reproducible between sessions: the prompted characters agree with themselves on the day and drift two days later."
    fig.text(0.5, 0.012 if sessions is None else 0.006,
             "Repeatable, located and useful are three different things. Experiment 1 asked for one state from a fixed list and never asked where; experiment 2 let the model\n"
             "propose its own present/absent characters and then made it point at each one. A state that comes back is not yet evidence: it has to come back above chance, land\n"
             "in one place on the specimen, and separate species — and every one of those requirements removes characters that the one before it had passed."
             + sessions_note,
             ha="center", va="bottom", fontsize=6.5 if sessions is None else 6.2,
             color="0.25", linespacing=1.65 if sessions is None else 1.55)

    fig.savefig(out_png, dpi=a.dpi, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="One figure: how reliable a vision-language model was when it read "
                    "structures off images (prompted descriptions vs free-form search).")
    ap.add_argument("--monograph", required=True,
                    help="monograph root; every input below defaults inside it")
    ap.add_argument("--out_dir", default=None, help="default <monograph>/vlm_reliability")
    # experiment 1
    ap.add_argument("--prompted_first", default=None,
                    help="default <monograph>/descriptive_states/descriptive_states_by_specimen.tsv")
    ap.add_argument("--prompted_reliability", default=None,
                    help="default <monograph>/descriptive_states/character_reliability.tsv")
    ap.add_argument("--prompted_repeatability", default=None,
                    help="default <monograph>/descriptive_states/repeatability.json")
    ap.add_argument("--prompted_retest", default=None,
                    help="paired second readings, same columns as the first reading; "
                         "supplying it replaces the estimated chance correction with a "
                         "true Cohen's kappa")
    ap.add_argument("--retest_compare", default=None,
                    help="directory holding retest_agreement_by_character.tsv and "
                         "retest_agreement_pooled.tsv from biorag_retest_compare_v1.py; "
                         "defaults to <monograph>/descriptive_states/retest_compare when "
                         "it exists. With it, panel a is drawn as within-session vs "
                         "between-session agreement instead of the older estimate")
    ap.add_argument("--retest_within_pair", default=THREE_WITHIN_PAIR,
                    help="pair name for series 1, two readings in one session")
    ap.add_argument("--retest_day_pair", default=THREE_DAY_PAIR,
                    help="pair name for series 2, one day apart; panel a is sorted by it")
    ap.add_argument("--retest_step_pair", default=THREE_STEP_PAIR,
                    help="pair name for series 3, across the change of the model's client; "
                         "the drift arrow is computed from this pair")
    ap.add_argument("--not_assessable", default="not assessable")
    # experiment 2
    ap.add_argument("--freeform_first", default=None,
                    help="default <monograph>/vlm_combined/vlm_character_states.tsv")
    ap.add_argument("--state_retest", nargs="*", default=None,
                    help="state-only retest TSVs; defaults to the two aligned retests")
    ap.add_argument("--state_reliability", nargs="*", default=None,
                    help="per-character retest_agreement TSVs beside the state retests")
    ap.add_argument("--location_retest", default=None,
                    help="default <monograph>/vlm_location_retest_20260920/"
                         "vlm_character_states_retest.tsv")
    ap.add_argument("--heatmap_index", default=None,
                    help="default <monograph>/vlm_combined/figures_heatmap/vlm_heatmap_index.tsv")
    ap.add_argument("--heatmap_summary", default=None)
    ap.add_argument("--frame_orientation", default=None,
                    help="default <monograph>/homology_frames/frame_orientation.tsv")
    ap.add_argument("--informativeness", default=None,
                    help="default <monograph>/autapomorphies_vlm_final/character_informativeness.tsv")
    ap.add_argument("--fixed_within_species", default=None,
                    help="default <monograph>/vlm_combined/fixed_within_species.tsv")
    ap.add_argument("--autapomorphy_summary", default=None,
                    help="default <monograph>/autapomorphies_vlm_final/autapomorphy_summary.json")
    ap.add_argument("--present_label", default="present")
    # thresholds
    ap.add_argument("--min_agreement", type=float, default=0.80)
    ap.add_argument("--min_kappa", type=float, default=0.60)
    ap.add_argument("--max_spread", type=float, default=0.10)
    ap.add_argument("--max_far_outside", type=float, default=0.25)
    ap.add_argument("--max_relocation", type=float, default=0.05)
    ap.add_argument("--min_relocated_pairs", type=int, default=3)
    ap.add_argument("--min_ami", type=float, default=0.20)
    ap.add_argument("--max_outside_rate", type=float, default=0.10)
    ap.add_argument("--max_pairs_per_character", type=int, default=60)
    ap.add_argument("--max_pairs_per_structure", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--copy_to", nargs="*", default=None,
                    help="extra directories to copy the finished figure into")
    a = ap.parse_args()

    M = Path(a.monograph)
    dflt = {
        "prompted_first": M / "descriptive_states/descriptive_states_by_specimen.tsv",
        "prompted_reliability": M / "descriptive_states/character_reliability.tsv",
        "prompted_repeatability": M / "descriptive_states/repeatability.json",
        "freeform_first": M / "vlm_combined/vlm_character_states.tsv",
        "location_retest": M / "vlm_location_retest_20260920/vlm_character_states_retest.tsv",
        "heatmap_index": M / "vlm_combined/figures_heatmap/vlm_heatmap_index.tsv",
        "heatmap_summary": M / "vlm_combined/figures_heatmap/vlm_heatmap_summary.json",
        "frame_orientation": M / "homology_frames/frame_orientation.tsv",
        "informativeness": M / "autapomorphies_vlm_final/character_informativeness.tsv",
        "fixed_within_species": M / "vlm_combined/fixed_within_species.tsv",
        "autapomorphy_summary": M / "autapomorphies_vlm_final/autapomorphy_summary.json",
    }
    for k, v in dflt.items():
        if getattr(a, k) is None:
            setattr(a, k, str(v))
    if a.state_retest is None:
        a.state_retest = [str(M / "vlm_characters_aligned/vlm_character_states_retest.tsv"),
                          str(M / "vlm_characters_aligned_rest/vlm_character_states_retest.tsv")]
    if a.state_reliability is None:
        a.state_reliability = [str(M / "vlm_characters_aligned/vlm_character_reliability.tsv"),
                               str(M / "vlm_characters_aligned_rest/vlm_character_reliability.tsv")]
    if a.retest_compare is None:
        for cand in (M / "descriptive_states/retest_compare_five_readings_20260921",
                     M / "descriptive_states/retest_compare"):
            if cand.is_dir():
                a.retest_compare = str(cand)
                break
    out = Path(a.out_dir) if a.out_dir else M / "vlm_reliability"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ── experiment 1 ────────────────────────────────────────────────────────
    e1, e1h = experiment_one(a.prompted_first, a.prompted_reliability,
                             a.prompted_repeatability, a.prompted_retest,
                             a.not_assessable)
    e1.to_csv(out / "prompted_character_repeatability.tsv", sep="\t", index=False)
    print(f"experiment 1: {e1h['cells_scored_total']:,} cells scored, "
          f"{e1h['cells_compared']:,} compared on {e1h['specimens_retested']} specimens, "
          f"pooled exact {e1h['pooled_exact_agreement']}")
    for _, r in e1.iterrows():
        print(f"  {r['character']:<22} exact {r['exact_agreement']:.3f}  "
              f"band {'   -  ' if not np.isfinite(r['band_agreement']) else f'{r.band_agreement:.3f}'}  "
              f"chance {r['chance_agreement_expected']:.3f}  "
              f"{'est ' if r['chance_corrected_is_estimate'] else ''}"
              f"\u03ba {r['chance_corrected']:+.3f}  {r['verdict']}")

    # \u2500\u2500 experiment 1, the retest repeated: within a session vs between sessions \u2500\u2500
    sessions = load_retest_sessions_three(a.retest_compare, a.retest_within_pair,
                                          a.retest_day_pair, a.retest_step_pair)
    if sessions is None:
        sessions = load_retest_sessions(a.retest_compare)
    if sessions is not None and sessions.get("mode") == "three":
        sessions["per_character"].to_csv(out / "prompted_character_sessions.tsv",
                                         sep="\t", index=False)
        w, dy, st = sessions["within"], sessions["day"], sessions["step"]
        print(f"\nsessions, three series ({sessions['source']}):")
        for lab, key, pk in (("same session ", w, "within_pair"),
                             ("one day apart", dy, "day_pair"),
                             ("across step  ", st, "step_pair")):
            print(f"  {lab}  [{sessions[pk]}]  {key['cells']:,} cells on "
                  f"{key['specimens']} specimens, exact {key['exact_agreement']:.4f}, "
                  f"mean κ {key['mean_cohen_kappa']:.4f}")
        for _, r in sessions["per_character"].iterrows():
            print(f"  {r['character']:<22} session {r['exact_within']:.3f} "
                  f"(κ {r['kappa_within']:+.3f})   day {r['exact_day']:.3f} "
                  f"(κ {r['kappa_day']:+.3f})   step {r['exact_step']:.3f} "
                  f"(κ {r['kappa_step']:+.3f})   {r['drift'] or ' '}")
    elif sessions is not None:
        sessions["per_character"].to_csv(out / "prompted_character_sessions.tsv",
                                         sep="\t", index=False)
        w, b = sessions["within"], sessions["between"]
        print(f"\nsessions ({sessions['source']}):")
        print(f"  within a session  [{sessions['within_pair']}]  {w['cells']:,} cells on "
              f"{w['specimens']} specimens, exact {w['exact_agreement']:.4f}, "
              f"mean \u03ba {w['mean_cohen_kappa']:.4f}")
        print(f"  between sessions  [{sessions['between_pair']}]  {b['cells']:,} cells on "
              f"{b['specimens']} specimens, exact {b['exact_agreement']:.4f}, "
              f"mean \u03ba {b['mean_cohen_kappa']:.4f}")
        for _, r in sessions["per_character"].iterrows():
            print(f"  {r['character']:<22} within {r['exact_within']:.3f} "
                  f"(\u03ba {r['kappa_within']:+.3f})   between {r['exact_between']:.3f} "
                  f"(\u03ba {r['kappa_between']:+.3f})   {r['drift'] or ' '}")
    else:
        print("\nsessions: no retest_compare TSVs found \u2014 panel a keeps the older estimate")

    # ── experiment 2 ────────────────────────────────────────────────────────
    first, loc, paired, tab = load_freeform(a)
    reloc = relocation_pairs(first, loc, a.present_label)
    reloc.to_csv(out / "relocation_pairs.tsv", sep="\t", index=False)

    ff = freeform_table(a, first, paired, tab, reloc)
    confirmed = json.loads(Path(a.autapomorphy_summary).read_text()).get(
        "confirmed_autapomorphies", 0)
    attr = build_attrition(ff, a, confirmed)
    ff.to_csv(out / "freeform_character_reliability.tsv", sep="\t", index=False)
    attr.to_csv(out / "attrition.tsv", sep="\t", index=False)

    p = oriented_reports(first, a.frame_orientation, a.present_label)
    dists = {
        "same_image": reloc["distance"].values,
        "between_specimen": between_specimen_distances(p, rng, a.max_pairs_per_character),
        "null": null_distances(p, rng, a.max_pairs_per_structure),
    }
    res_step = coordinate_step(np.r_[first["x"].dropna().values, first["y"].dropna().values])
    pooled_kappa = cohen_kappa(paired["state_1"].values, paired["state_2"].values)
    kstats = {"n_pairs": int(len(paired)),
              "pooled_agreement": float((paired["state_1"] == paired["state_2"]).mean()),
              "pooled_kappa": float(pooled_kappa)}

    print(f"\nexperiment 2: {first['character'].nunique()} characters, "
          f"{first['specimen_id'].nunique()} specimens, {first['structure'].nunique()} structures")
    print(f"  paired readings {kstats['n_pairs']:,}  agreement {kstats['pooled_agreement']:.4f}  "
          f"pooled κ {kstats['pooled_kappa']:.4f}")
    print(f"  agreement ≥ {a.min_agreement}: {int((ff['agreement'] >= a.min_agreement).sum())}"
          f"   κ ≥ {a.min_kappa}: {int((ff['cohen_kappa'] >= a.min_kappa).sum())}"
          f"   κ undefined: {int((~ff['kappa_defined']).sum())}")
    print(f"  relocation pairs {len(reloc):,}  median {reloc['distance'].median():.4f}  "
          f"within 0.05 {float((reloc['distance'] <= 0.05).mean()):.3f}  "
          f"within 0.10 {float((reloc['distance'] <= 0.10).mean()):.3f}")
    print(f"  medians  same image {np.median(dists['same_image']):.4f}  "
          f"two specimens {np.median(dists['between_specimen']):.4f}  "
          f"different characters {np.median(dists['null']):.4f}  "
          f"(coordinate step {res_step:g})")
    print("\nattrition")
    for _, r in attr.iterrows():
        extra = f"  [{r['carried_untested']} {r['untested_label']}]" if r["carried_untested"] else ""
        print(f"  {r['step']:<52} {r['surviving']:>4}   -{r['lost_here']}{extra}")

    # ── figure ──────────────────────────────────────────────────────────────
    png, pdf = out / "fig_vlm_reliability.png", out / "fig_vlm_reliability.pdf"
    draw_figure(a, e1, e1h, ff, reloc, attr, dists, res_step, kstats, png, pdf,
                sessions=sessions)
    print(f"\nwrote {png}")

    for dest in (a.copy_to or []):
        dd = Path(dest)
        dd.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, dd / "Fig_vlm_reliability.png")
        shutil.copy2(pdf, dd / "Fig_vlm_reliability.pdf")
        print(f"copied to {dd / 'Fig_vlm_reliability.png'}")

    # ── summary ─────────────────────────────────────────────────────────────
    hsum = json.loads(Path(a.heatmap_summary).read_text()) if Path(a.heatmap_summary).exists() else {}
    sessions_block = {}
    if sessions is not None:
        def _n(v):
            v = float(v)
            return round(v, 4) if np.isfinite(v) else None
        cols = ["exact_within", "exact_between", "kappa_within", "kappa_between",
                "weighted_kappa_between", "share_second_reading_higher_between",
                "cells_within", "cells_between"]
        if sessions.get("mode") == "three":
            # three series; the old key names stay populated, between_sessions =
            # across the step, so the manuscript builder's tokens keep resolving
            cols = ["exact_within", "exact_day", "exact_step",
                    "kappa_within", "kappa_day", "kappa_step",
                    "weighted_kappa_step", "share_second_reading_higher_step",
                    "cells_within", "cells_day", "cells_step"]
            alias = {"exact_between": "exact_step", "kappa_between": "kappa_step",
                     "weighted_kappa_between": "weighted_kappa_step",
                     "share_second_reading_higher_between": "share_second_reading_higher_step",
                     "cells_between": "cells_step"}
            sessions_block = {
                "paired_readings_available": True,
                "retest_compare_dir": sessions["source"],
                "readings_compared": 5,
                "within_session_pair": sessions["within_pair"],
                "one_day_apart_pair": sessions["day_pair"],
                "across_step_pair": sessions["step_pair"],
                "between_sessions_pair": sessions["step_pair"],
                "within_session": sessions["within"],
                "one_day_apart": sessions["day"],
                "across_step": sessions["step"],
                "between_sessions": sessions["step"],
                "per_character_sessions": {
                    r["character"]: dict(
                        {c: (int(r[c]) if c.startswith("cells") else _n(r[c])) for c in cols},
                        **{k: (int(r[v]) if k.startswith("cells") else _n(r[v]))
                           for k, v in alias.items()})
                    for _, r in sessions["per_character"].iterrows()},
            }
        else:
            sessions_block = {
                "paired_readings_available": True,
                "retest_compare_dir": sessions["source"],
                "within_session_pair": sessions["within_pair"],
                "between_sessions_pair": sessions["between_pair"],
                "within_session": sessions["within"],
                "between_sessions": sessions["between"],
                "per_character_sessions": {
                    r["character"]: {c: (int(r[c]) if c.startswith("cells") else _n(r[c]))
                                     for c in cols}
                    for _, r in sessions["per_character"].iterrows()},
            }
    summary = {
        "version": VERSION,
        "experiment_1_prompted_descriptions": {
            **e1h,
            "characters": int(len(e1)),
            "chance_correction": ("true Cohen's kappa from --prompted_retest"
                                  if a.prompted_retest else
                                  "ESTIMATE: (exact - sum p_k^2) / (1 - sum p_k^2) from the "
                                  "first-reading state frequencies; the paired second readings "
                                  "were not kept")
                                 + (" — superseded for panel a by within_session / "
                                    "between_sessions below, which are true Cohen's kappa "
                                    "from the repeated retest"
                                    if sessions is not None else ""),
            "verdicts": e1["verdict"].value_counts().to_dict(),
            "per_character": e1.set_index("character")[
                ["type", "states_in_scale", "n_compared", "exact_agreement",
                 "band_agreement", "chance_agreement_expected",
                 "chance_corrected", "chance_corrected_is_estimate",
                 "verdict"]].round(4).to_dict("index"),
            **sessions_block,
        },
        "experiment_2_freeform_characters": {
            "characters": int(first["character"].nunique()),
            "specimens": int(first["specimen_id"].nunique()),
            "structures": int(first["structure"].nunique()),
            "first_reading_reports": int(len(first)),
            "paired_readings_total": kstats["n_pairs"],
            "paired_readings_original_retest": int((paired["retest"] != "location").sum()),
            "paired_readings_location_retest": int((paired["retest"] == "location").sum()),
            "keys_read_twice_in_both_retests": int(
                paired.groupby(["specimen_id", "character"]).size().gt(1).sum()),
            "pooled_state_agreement": round(kstats["pooled_agreement"], 4),
            "pooled_cohen_kappa": round(kstats["pooled_kappa"], 4),
            "characters_agreement_at_least": {
                str(a.min_agreement): int((ff["agreement"] >= a.min_agreement).sum())},
            "characters_tabulated_agreement_at_least": {
                str(a.min_agreement): int(
                    (ff["retest_agreement_tabulated"] >= a.min_agreement).sum())},
            "characters_kappa_at_least": {
                str(a.min_kappa): int((ff["cohen_kappa"] >= a.min_kappa).sum())},
            "characters_kappa_undefined": int((~ff["kappa_defined"]).sum()),
            "median_agreement": round(float(ff["agreement"].median()), 4),
            "median_kappa": round(float(ff["cohen_kappa"].median()), 4),
            "heatmap_categories": ff["heatmap_category"].value_counts().to_dict(),
            "heatmap_thresholds": hsum.get("thresholds", {}),
        },
        "localisation": {
            "coordinate_step": res_step,
            "coordinate_step_px_of_768": round(res_step * 768, 2),
            "relocation_pairs_same_image": int(len(reloc)),
            "relocation_median": round(float(reloc["distance"].median()), 4),
            "relocation_within_0.05": round(float((reloc["distance"] <= 0.05).mean()), 4),
            "relocation_within_0.10": round(float((reloc["distance"] <= 0.10).mean()), 4),
            "relocation_exact": round(float((reloc["distance"] == 0).mean()), 4),
            "median_same_image": round(float(np.median(dists["same_image"])), 4),
            "median_between_specimen": round(float(np.median(dists["between_specimen"])), 4),
            "median_null_different_characters": round(float(np.median(dists["null"])), 4),
            "n_between_specimen_pairs": int(len(dists["between_specimen"])),
            "n_null_pairs": int(len(dists["null"])),
            "orientation": "between specimens (x, y) -> (1 - x, 1 - y) for flipped frames; "
                           "two readings of the same image are compared unmapped",
            "seed": a.seed,
        },
        "attrition": attr.to_dict("records"),
        "attrition_note": "steps are cumulative and applied in the order listed; the loss "
                          "attributed to a step depends on its position in the sequence",
        "thresholds": {
            "min_agreement": a.min_agreement, "min_kappa": a.min_kappa,
            "max_spread": a.max_spread, "max_far_outside": a.max_far_outside,
            "max_relocation": a.max_relocation,
            "min_relocated_pairs": a.min_relocated_pairs,
            "min_ami": a.min_ami, "max_outside_rate": a.max_outside_rate},
        "confirmed_autapomorphies": int(confirmed),
    }
    (out / "vlm_reliability_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'vlm_reliability_summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
