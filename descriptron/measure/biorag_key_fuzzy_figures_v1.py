#!/usr/bin/env python3
"""
biorag_key_fuzzy_figures_v1.py — crisp versus graded: one key, three ways of walking it
=======================================================================================

A computed dichotomous key is a tree of couplets, each with a leading character, a
printed threshold and the range observed on either side. Nothing about that tree says
how a specimen should be run down it when a character is missing, or when the two
observed ranges overlap so that the threshold is a convention rather than a boundary.
Three walkers already exist in this pipeline and they answer those two questions
differently, so the SAME key gives three different answers for the same specimen.
This script runs all three over the same hold-out and draws the comparison.

  walker 1  strict, as printed
            `biorag_novelty_score_v1.key_path`. The leading character of the couplet
            and nothing else; if it was not measured the path stops, unnamed. It
            counts a "conflict" whenever the value lies outside the range observed on
            the side it was sent down — the crisp key's only warning signal.

  walker 2  first measured character
            `biorag_key_builder_v1.identify(..., vote=False)`. At each couplet it
            takes the first character that was measured, so a missing leading
            character falls back to the supporting ones, and decides by the printed
            threshold. This is the walker the published identification test uses.

  walker 3  graded (fuzzy) membership
            `biorag_key_fuzzy_v1.fuzzy_key_path`. Also falls back to supporting
            characters, but chooses the side by trapezoidal membership between the
            two OBSERVED ranges rather than by the printed threshold, so where the
            ranges overlap it can route a specimen the other way. It carries two
            grades down the path with a t-norm: `decisiveness` (how clearly the value
            chose its side) and `fit` (how compatible it is with the range on that
            side); `score = min(decisiveness, fit)` is the weakest link of the path.

Everything is evaluated by hold-out, because a specimen graded against ranges it
helped define cannot fall outside them:

  known arm      the key is rebuilt without that one specimen and the specimen is run
                 down it (the rebuilt trees are cached, one JSON per specimen);
  held-out arm   the key is rebuilt without the whole species, so the specimen belongs
                 to a species the key has never seen. It is then unidentifiable by
                 construction, and the question is only whether the key admits it.

What the figure asks, panel by panel:

  a  How much does the choice of walker matter? Falling back to supporting characters
     buys names (walker 1 -> 2); grading the couplets buys a few more names again but
     re-routes some specimens, and not always for the better.
  b  Does the graded score know when the name it just produced is wrong? (Yes.)
  c  Does it know when the specimen belongs to no described species? (Barely — this is
     the negative result and is drawn as plainly as the positive one.)
  d  Used as a warning light, how does the graded score compare with the crisp key's
     own warning, the out-of-range conflict count?

Nothing here calls a language model: the trees are computed from the character matrix
and every walker is deterministic, so the whole figure re-runs in seconds.

  python biorag_key_fuzzy_figures_v1.py \\
      --matrix_dir "$M/compiled_key_tier" --taxon_profile <profile.yaml> \\
      --key_loo_dir "$M/key_fuzzy_loo/keys_loo" \\
      --fuzzy_specimens "$M/key_fuzzy_loo/fuzzy_key_specimens.tsv" \\
      --identification_test "$M/key/identification_test.tsv" \\
      --out_dir "$M/key_fuzzy_loo/figures"

Outputs: crisp_vs_graded.tsv (one row per specimen), crisp_vs_graded_summary.json,
fig_crisp_vs_graded.png / .pdf (220 dpi).
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.lines import Line2D                                  # noqa: E402
from matplotlib.patches import Patch                                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                  # noqa: E402
from biorag_novelty_score_v1 import Reference, key_path              # noqa: E402
from biorag_key_builder_v1 import identify                           # noqa: E402
from biorag_key_fuzzy_v1 import build_key, fuzzy_key_path            # noqa: E402

VERSION = "1.0-crisp-vs-graded"

# The two walking styles, plus neutral greys for the outcome that is neither right nor
# wrong. Right / wrong are separated by lightness AND hatching, so the panels survive
# greyscale printing and colour-vision deficiency without relying on hue.
CRISP = "#1565c0"
GRADED = "#6a1b9a"
CRISP_LIGHT = "#8fb6e3"
GRADED_LIGHT = "#bb93d2"
GREY_NONE = "#cfcfcf"
GREY_INK = "#4d4d4d"
HATCH = "////"

WALKERS = [
    ("w1", "walker1", "1  strict, as printed\nleading character only", CRISP, CRISP_LIGHT),
    ("w2", "walker2", "2  first measured character\nprinted threshold", CRISP, CRISP_LIGHT),
    ("w3", "walker3", "3  graded membership\nobserved ranges", GRADED, GRADED_LIGHT),
]


# ─────────────────────────────────────────────────────────────────────────────
# Walking
# ─────────────────────────────────────────────────────────────────────────────

def _safe(sid: str) -> str:
    """The cache file name biorag_key_fuzzy_v1 writes for a specimen."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(sid))


def _couplets_visited(steps) -> str:
    """The couplet numbers a fuzzy path went through, for comparing routes."""
    out = []
    for s in steps:
        m = re.match(r"couplet (\d+)", s)
        if m:
            out.append(m.group(1))
    return "-".join(out)


def load_or_build_key(cache: Path, sid, matrix_dir: Path, species, python: str, profile_path: str):
    """The key rebuilt without this one specimen. Cached by biorag_key_fuzzy_v1; rebuilt
    here (deterministically, no model) only if the cache is incomplete."""
    cf = cache / f"{_safe(sid)}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    with tempfile.TemporaryDirectory() as td:
        key = build_key(matrix_dir, list(species), Path(td), python, profile_path,
                        exclude_specimens=[str(sid)])
    if "error" not in key:
        cache.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(key))
    return key


def walk_all(ref: Reference, cache: Path, matrix_dir: Path, python: str, profile_path: str):
    """One row per specimen: the three walkers over the same leave-one-specimen-out key."""
    species = list(ref.species) if hasattr(ref, "species") else sorted(set(ref.species_of.values()))
    rows = []
    for n, sid in enumerate(ref.raw.index, 1):
        key = load_or_build_key(cache, sid, matrix_dir, species, python, profile_path)
        if "error" in key:
            print(f"  key without {sid} unavailable — skipped")
            continue
        row = ref.raw.loc[sid]
        vals = {f: float(v) for f, v in row.items() if v == v}
        sp = ref.species_of.get(sid)
        sex = ref.sex_of.get(sid, "unknown")

        k1 = key_path(key, row)
        name2, path2, status2 = identify(vals, sex, key["couplets"])
        k3 = fuzzy_key_path(key, row, "min")

        name1 = k1.get("terminal") if k1.get("resolved") else None
        name3 = k3.get("terminal") if k3.get("resolved") else None
        rows.append({
            "specimen_id": sid, "species": sp, "sex": sex,
            "w1_name": name1, "w1_correct": bool(name1 is not None and name1 == sp),
            "w1_resolved": bool(k1.get("resolved")), "w1_conflicts": int(k1.get("conflicts", 0)),
            "w2_name": name2, "w2_correct": bool(name2 is not None and name2 == sp),
            "w2_resolved": bool(name2 is not None), "w2_status": status2,
            "w2_path": "-".join(map(str, path2)),
            "w3_name": name3, "w3_correct": bool(name3 is not None and name3 == sp),
            "w3_resolved": bool(k3.get("resolved")),
            "w3_score": (round(min(k3["decisiveness"], k3["fit"]), 4) if k3.get("resolved") else np.nan),
            "w3_decisiveness": round(k3.get("decisiveness", np.nan), 4),
            "w3_fit": round(k3.get("fit", np.nan), 4),
            "w3_used_support": int(k3.get("used_support", 0)),
            "w3_weakest": k3.get("weakest"),
            "w3_path": _couplets_visited(k3.get("steps", [])),
        })
        if n % 50 == 0:
            print(f"  walked {n}/{len(ref.raw.index)}")
    d = pd.DataFrame(rows)
    d["route_differs_2_vs_3"] = (d["w2_path"] != d["w3_path"]) | \
                                (d["w2_name"].fillna("~") != d["w3_name"].fillna("~"))
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def auc(pos, neg):
    """P(a random member of `pos` scores above a random member of `neg`); 0.5 = nothing."""
    pos = [x for x in pos if x == x]
    neg = [x for x in neg if x == x]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def warning_curve(right, wrong):
    """Sweep the flagging threshold over the graded score. A name is flagged as doubtful
    when its score falls below the threshold. x = % of right names needlessly doubted,
    y = % of wrong names caught."""
    ts = np.unique(np.concatenate([[0.0], np.sort(np.concatenate([right, wrong])), [1.0 + 1e-9]]))
    x = np.array([100.0 * np.mean(right < t) for t in ts])
    y = np.array([100.0 * np.mean(wrong < t) for t in ts])
    return ts, x, y


def point_at(right, wrong, t):
    return 100.0 * float(np.mean(right < t)), 100.0 * float(np.mean(wrong < t))


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def _tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, labelsize=9, colors=GREY_INK)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a9a")


def _violin(ax, pos, vals, face, edge, hatch=None):
    if len(vals) < 2:
        return
    v = ax.violinplot([vals], positions=[pos], widths=0.72, showextrema=False, showmedians=False)
    for b in v["bodies"]:
        b.set_facecolor(face)
        b.set_edgecolor(edge)
        b.set_alpha(0.55)
        b.set_linewidth(1.2)
        if hatch:
            b.set_hatch(hatch)


def _strip(ax, pos, vals, colour, rng, marker="o"):
    j = rng.uniform(-0.13, 0.13, size=len(vals))
    ax.plot(pos + j, vals, marker, ms=4.0, mfc="none", mec=colour, mew=0.9, alpha=0.85,
            linestyle="none", zorder=3)


def _median_bar(ax, pos, vals, colour):
    m = float(np.median(vals))
    ax.plot([pos - 0.32, pos + 0.32], [m, m], "-", color=colour, lw=2.6, zorder=5,
            solid_capstyle="round")
    ax.annotate(f"{m:.2f}", (pos + 0.42, m), fontsize=9, color=colour, va="center",
                ha="left", fontweight="bold")
    return m


def _panel_title(ax, title, subtitle):
    ax.set_title(title, loc="left", fontweight="bold", pad=22)
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.7, color=GREY_INK,
            va="bottom", ha="left")


def draw(d, fz, idt, stats, out_png, out_pdf):
    rng = np.random.default_rng(7)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 12,
                         "axes.labelsize": 10, "text.color": "#1a1a1a",
                         "axes.labelcolor": GREY_INK, "axes.edgecolor": "#9a9a9a"})
    fig = plt.figure(figsize=(12.4, 9.2))
    ax_a = fig.add_axes([0.075, 0.620, 0.385, 0.228])
    ax_b = fig.add_axes([0.575, 0.620, 0.385, 0.228])
    ax_c = fig.add_axes([0.075, 0.190, 0.385, 0.262])
    ax_d = fig.add_axes([0.575, 0.190, 0.385, 0.262])

    n_tot = stats["n_specimens"]

    # ── a  three ways of walking the same key ────────────────────────────────
    ys = [2.5, 1.25, 0.0]
    for y, (_pfx, wk, lab, dark, light) in zip(ys, WALKERS):
        c = stats["walkers"][wk]
        ax_a.text(0.0, y + 0.34, lab, fontsize=9.0, color="#1a1a1a", va="bottom", ha="left")
        segs = [(c["right"], dark, None, "white"),
                (c["wrong"], light, HATCH, "#1a1a1a"),
                (c["no_name"], GREY_NONE, None, "#333333")]
        left = 0.0
        for n, face, hatch, txt in segs:
            if n <= 0:
                continue
            ax_a.barh(y, n, left=left, height=0.52, color=face, edgecolor="white",
                      linewidth=1.6, hatch=hatch, zorder=2)
            if hatch:
                ax_a.barh(y, n, left=left, height=0.52, color="none", edgecolor=dark,
                          linewidth=1.0, zorder=3)
            ax_a.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=10,
                      color=txt, fontweight="bold", zorder=4)
            left += n
    ax_a.set_yticks([])
    ax_a.spines["left"].set_visible(False)
    ax_a.set_xlim(0, n_tot)
    ax_a.set_ylim(-0.45, 4.05)
    ax_a.set_xlabel("specimens")
    _panel_title(ax_a, "a  Three ways of walking the same key",
                 f"each of {n_tot} specimens run down a key rebuilt without it")
    _tidy(ax_a)
    ax_a.legend(handles=[Patch(facecolor="#6f6f6f", label="right name"),
                         Patch(facecolor="#dcdcdc", hatch=HATCH, edgecolor="#6f6f6f",
                               label="wrong name"),
                         Patch(facecolor=GREY_NONE, label="no name (path stops)")],
                loc="upper left", bbox_to_anchor=(-0.01, 1.01), ncol=3, frameon=False,
                fontsize=8.8, handlelength=1.4, columnspacing=1.2, handletextpad=0.5)

    r = stats["route_2_vs_3"]
    fig.text(0.075, 0.548,
             f"Walkers 2 and 3 take a different route through the key on {r['n_differ']} of "
             f"{n_tot} specimens. On those, walker 2 reaches the right\nname "
             f"{r['walker2_right']} times, walker 3 {r['walker3_right']} times, and neither "
             f"of them {r['neither_right']} times \u2014 grading the couplets moves specimens, "
             "but not reliably towards the right name.",
             fontsize=9.0, color=GREY_INK, va="top", ha="left", linespacing=1.55)

    # ── b  score separates right from wrong names ────────────────────────────
    kn = fz[(fz["arm"] == "known") & (fz["resolved"] == True)]            # noqa: E712
    right = kn[kn["correct"] == True]["score"].to_numpy(float)            # noqa: E712
    wrong = kn[kn["correct"] != True]["score"].to_numpy(float)            # noqa: E712
    _violin(ax_b, 0, right, GRADED, GRADED)
    _violin(ax_b, 1, wrong, GRADED_LIGHT, GRADED, HATCH)
    _strip(ax_b, 0, right, GRADED, rng)
    _strip(ax_b, 1, wrong, "#8e5aa8", rng)
    _median_bar(ax_b, 0, right, GRADED)
    _median_bar(ax_b, 1, wrong, GRADED)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels([f"right name\nn = {len(right)}", f"wrong name\nn = {len(wrong)}"],
                         fontsize=9.5, linespacing=1.35)
    ax_b.set_xlim(-0.62, 1.92)
    ax_b.set_ylim(-0.05, 1.08)
    ax_b.set_ylabel("path score (graded key)")
    _panel_title(ax_b, "b  The graded score says how far to trust a name",
                 "walker 3, described specimens that reached a name; bars are medians")
    _tidy(ax_b)
    ax_b.text(0.99, 0.98, f"AUC = {stats['AUC_right_vs_wrong']:.3f}", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=11, color=GRADED, fontweight="bold")

    # ── c  the score does not separate species absent from the key ──────────
    hd = fz[(fz["arm"] == "held_out") & (fz["resolved"] == True)]         # noqa: E712
    known_s = kn["score"].to_numpy(float)
    held_s = hd["score"].to_numpy(float)
    _violin(ax_c, 0, known_s, GRADED, GRADED)
    _violin(ax_c, 1, held_s, GRADED_LIGHT, GRADED, HATCH)
    _strip(ax_c, 0, known_s, GRADED, rng)
    _strip(ax_c, 1, held_s, "#8e5aa8", rng)
    _median_bar(ax_c, 0, known_s, GRADED)
    _median_bar(ax_c, 1, held_s, GRADED)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels([f"described species\n(that specimen held out)\nn = {len(known_s)}",
                          f"species the key has\nnever seen\nn = {len(held_s)}"],
                         fontsize=9.5, linespacing=1.35)
    ax_c.set_xlim(-0.62, 1.92)
    ax_c.set_ylim(-0.05, 1.08)
    ax_c.set_ylabel("path score (graded key)")
    _panel_title(ax_c, "c  The score does not detect species absent from the key",
                 "specimens of species the key has never seen score almost as highly as "
                 "specimens of described species")
    _tidy(ax_c)
    ax_c.text(0.99, 0.98, f"AUC = {stats['AUC_known_vs_held_out']:.3f}",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=11,
              color=GREY_INK, fontweight="bold")

    # ── d  the score as a warning light ──────────────────────────────────────
    ts, cx, cy = warning_curve(right, wrong)
    ax_d.plot([0, 100], [0, 100], ls=(0, (4, 4)), color="#b0b0b0", lw=1.2, zorder=1)
    ax_d.plot(cx, cy, "-", color=GRADED, lw=2.4, zorder=4, solid_capstyle="round")
    for t, dx, dy, ha, va in [(0.3, 10, -13, "left", "top"), (0.6, 10, -2, "left", "top")]:
        op = stats["operating_points"][f"score_below_{t}"]
        px, py = op["pct_right_flagged"], op["pct_wrong_flagged"]
        ax_d.plot([px], [py], "o", ms=9, mfc="white", mec=GRADED, mew=2.4, zorder=6)
        ax_d.annotate(f"score < {t}\n{py:.0f}% caught, {px:.0f}% doubted", (px, py),
                      textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                      fontsize=9.0, color=GRADED, linespacing=1.45)
    cw = stats["crisp_warning"]
    ax_d.plot([cw["pct_right_flagged"]], [cw["pct_wrong_flagged"]], "D", ms=11, color=CRISP,
              mec="white", mew=1.4, zorder=7)
    ax_d.annotate(f"crisp warning\n{cw['pct_wrong_flagged']:.0f}% caught, "
                  f"{cw['pct_right_flagged']:.0f}% doubted",
                  (cw["pct_right_flagged"], cw["pct_wrong_flagged"]), textcoords="offset points",
                  xytext=(13, -1), ha="left", va="center", fontsize=9.0, color=CRISP,
                  fontweight="bold", linespacing=1.45)
    ax_d.set_xlim(-2, 103)
    ax_d.set_ylim(-2, 103)
    ax_d.set_xticks([0, 20, 40, 60, 80, 100])
    ax_d.set_yticks([0, 20, 40, 60, 80, 100])
    ax_d.set_xlabel("% of right names needlessly doubted")
    ax_d.set_ylabel("% of wrong names flagged")
    _panel_title(ax_d, "d  Using the score as a warning",
                 "the flagging threshold swept over the graded score")
    _tidy(ax_d)
    ax_d.legend(handles=[Line2D([], [], color=GRADED, lw=2.4,
                                label="graded key: flag a low path score"),
                         Line2D([], [], color=CRISP, marker="D", ms=8, ls="none",
                                label="crisp key: flag a value outside the branch range"),
                         Line2D([], [], color="#b0b0b0", lw=1.2, ls=(0, (4, 4)),
                                label="no information")],
                loc="lower right", bbox_to_anchor=(1.01, -0.02), frameon=False,
                fontsize=8.6, handlelength=1.8, handletextpad=0.6, labelspacing=0.35)

    fig.suptitle("Walking one computed key crisply or with graded membership",
                 x=0.075, ha="left", y=0.968, fontsize=14, fontweight="bold")
    fig.text(0.075, 0.082,
             "Both arms are hold-outs: a described specimen is run down a key rebuilt without "
             "it; a specimen of an unseen species, down a key rebuilt without its whole "
             "species.\nScore is the path's weakest link \u2014 the minimum along it of "
             "decisiveness (distance from the crossover of the two observed ranges) and fit "
             "(how far inside the range taken).",
             fontsize=9.0, color=GREY_INK, va="top", ha="left", linespacing=1.6)

    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)



# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Crisp versus graded walks of one computed key")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--key_loo_dir", required=True,
                    help="cache of keys rebuilt without one specimen each (keys_loo)")
    ap.add_argument("--fuzzy_specimens", required=True,
                    help="fuzzy_key_specimens.tsv — both arms of the graded walker")
    ap.add_argument("--identification_test", required=True,
                    help="identification_test.tsv — the crisp key's own conflict warning")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used if a leave-one-out key has to be rebuilt")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    ref = Reference(Path(a.matrix_dir), profile)
    print(f"{len(ref.raw.index)} specimens, {len(set(ref.species_of.values()))} species")

    d = walk_all(ref, Path(a.key_loo_dir), Path(a.matrix_dir), a.python, a.taxon_profile)
    fz = pd.read_csv(a.fuzzy_specimens, sep="\t")
    idt = pd.read_csv(a.identification_test, sep="\t")

    # ---- per-walker outcome, known arm
    walkers = {}
    for pfx, wk, *_ in WALKERS:
        res = d[f"{pfx}_resolved"].astype(bool)
        cor = d[f"{pfx}_correct"].astype(bool)
        walkers[wk] = {"reached_a_name": int(res.sum()), "right": int(cor.sum()),
                       "wrong": int((res & ~cor).sum()), "no_name": int((~res).sum())}

    # ---- where walkers 2 and 3 disagree
    dif = d[d["route_differs_2_vs_3"]]
    route = {"n_differ": int(len(dif)),
             "n_different_name": int((d["w2_name"].fillna("~") != d["w3_name"].fillna("~")).sum()),
             "walker2_right": int(dif["w2_correct"].sum()),
             "walker3_right": int(dif["w3_correct"].sum()),
             "neither_right": int((~dif["w2_correct"].astype(bool) &
                                   ~dif["w3_correct"].astype(bool)).sum())}

    # ---- graded score, from the reference run on disk
    kn = fz[(fz["arm"] == "known") & (fz["resolved"] == True)]            # noqa: E712
    hd = fz[(fz["arm"] == "held_out") & (fz["resolved"] == True)]         # noqa: E712
    right = kn[kn["correct"] == True]["score"].to_numpy(float)            # noqa: E712
    wrong = kn[kn["correct"] != True]["score"].to_numpy(float)            # noqa: E712
    ops = {}
    for t in (0.3, 0.6):
        px, py = point_at(right, wrong, t)
        ops[f"score_below_{t}"] = {"pct_right_flagged": round(px, 1),
                                   "pct_wrong_flagged": round(py, 1),
                                   "n_right_flagged": int((right < t).sum()),
                                   "n_wrong_flagged": int((wrong < t).sum())}

    # ---- the crisp key's own warning: a value outside the range of the branch taken
    named = idt[~idt["loo"].isin(["unresolved", "species has no other specimen"])]
    cr = named[named["loo_ok"] == True]                                   # noqa: E712
    cw = named[named["loo_ok"] != True]                                   # noqa: E712
    crisp = {"n_named": int(len(named)), "n_right": int(len(cr)), "n_wrong": int(len(cw)),
             "n_right_flagged": int((cr["loo_conflicts"] > 0).sum()),
             "n_wrong_flagged": int((cw["loo_conflicts"] > 0).sum()),
             "pct_right_flagged": round(100.0 * float((cr["loo_conflicts"] > 0).mean()), 1),
             "pct_wrong_flagged": round(100.0 * float((cw["loo_conflicts"] > 0).mean()), 1)}

    stats = {
        "version": VERSION,
        "n_specimens": int(len(d)),
        "n_species": int(len(set(ref.species_of.values()))),
        "walkers": walkers,
        "walker_labels": {"walker1": "strict, as printed (leading character only)",
                          "walker2": "first measured character, printed threshold",
                          "walker3": "graded membership between the observed ranges"},
        "route_2_vs_3": route,
        "graded_known_named": int(len(kn)), "graded_held_out_named": int(len(hd)),
        "graded_held_out_total": int((fz["arm"] == "held_out").sum()),
        "median_score_right": round(float(np.median(right)), 4),
        "median_score_wrong": round(float(np.median(wrong)), 4),
        "median_score_known": round(float(np.median(kn["score"])), 4),
        "median_score_held_out": round(float(np.median(hd["score"])), 4),
        "AUC_right_vs_wrong": round(auc(right, wrong), 3),
        "AUC_known_vs_held_out": round(auc(kn["score"].to_numpy(float),
                                           hd["score"].to_numpy(float)), 3),
        "operating_points": ops,
        "crisp_warning": crisp,
        # known arm only; the reference run counts both arms together
        "known_arm_paths_using_a_supporting_character": int((d["w3_used_support"] > 0).sum()),
    }

    d.to_csv(out / "crisp_vs_graded.tsv", sep="\t", index=False)
    (out / "crisp_vs_graded_summary.json").write_text(json.dumps(stats, indent=2))
    draw(d, fz, idt, stats, out / "fig_crisp_vs_graded.png", out / "fig_crisp_vs_graded.pdf")
    print(json.dumps(stats, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
