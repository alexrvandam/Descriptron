#!/usr/bin/env python3
"""One figure and one table summarising every instrument that was tried on the same
material, for the two questions a monograph has to answer.

The two questions
-----------------
Naming    a described specimen, with its own record withheld from every statistic it is
          scored against (leave one specimen out), is given a name. The score is the
          number of specimens named correctly out of ALL specimens: a specimen an
          instrument cannot resolve counts as not named correctly, not as an abstention.
          The pale bar behind each solid bar is the share that reached any name at all,
          so the reader can see how much of a failure is "no answer" and how much is
          "wrong answer".

Novelty   each species in turn is withheld entirely and its specimens scored (the
          detection arm: unseen species recognised); and each specimen is withheld with
          its own species left in (the false-alarm arm: described species wrongly called
          new). Both arms are hold-outs.

The two caveats the figure must carry
-------------------------------------
1. nested versus in sample. An operating point (a threshold and a rule for pooling a
   series) chosen on the same species it is scored on is optimistic. The honest estimate
   is nested: the point is chosen on the other species and applied to the one left out.
   Both are drawn — filled marker nested, hollow marker best in sample, joined by a line,
   so the optimism of in-sample tuning is visible as the length of that line.
2. upper-bound cells. Where an instrument's characters were proposed by a model that saw
   the species, and those characters are still in the matrix when that species is
   withheld, the selection leak is open and the novelty figure is an upper bound. Those
   cells are marked with a dagger in the table and the figure. The one instrument for
   which the leak was closed is also shown, with the leak-open version as a faint ghost
   marker, so the size of that leak can be read off the plot.

Nothing here re-derives a statistic. Every number is either read from a result file or
recomputed with the same helpers the original analyses used (`calls_from_scores` and
`summarise` from biorag_instrument_compare_v3, `key_calls` from biorag_calibrate_v1,
`wilson` from biorag_congruence_compare_v1), so the figure cannot drift from the text.

Taxon agnostic: every path is a command-line argument. `--monograph` sets the root and
each input can be overridden individually.

Outputs into --out_dir:
    methods_summary.tsv
    fig_methods_summary.png   (220 dpi)
    fig_methods_summary.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VERSION = "1.0"

# Colours: kept consistent with the paper's existing instrument figure.
C_MATRIX = "#2e7d32"
C_KEY = "#1565c0"
C_GRAPH = "#ef6c00"
C_FUZZY = "#6a1b9a"
C_MODEL = "#00838f"          # the model-proposed-character family gets its own hue
C_GHOST = "#90a4ae"

CEILINGS = [1, 3, 6]
DAGGER = "†"


# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────────────
_POINT = re.compile(r"(\d+)\s*/\s*(\d+)\s*caught\s*,\s*(\d+)\s*/\s*(\d+)\s*false")


def parse_point(s):
    """'19/29 caught, 2/29 false' (optionally followed by '(rule)') -> (19, 2, 29)."""
    m = _POINT.search(str(s))
    if not m:
        raise ValueError(f"cannot read an operating point from {s!r}")
    return int(m.group(1)), int(m.group(3)), int(m.group(2))


def pick(rows, key, value):
    for r in rows:
        if r.get(key) == value:
            return r
    raise KeyError(f"no entry with {key} == {value!r}; saw "
                   f"{sorted(str(r.get(key)) for r in rows)}")


def lighten(hex_colour, f=0.55):
    """Blend towards white by fraction f."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * f) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# ─────────────────────────────────────────────────────────────────────────────
# gathering the eight methods
# ─────────────────────────────────────────────────────────────────────────────
def gather(a, pd, calls_from_scores, summarise, key_calls, wilson):
    """Read every number. Returns (methods, ghost, species, n_specimens)."""
    species = sorted(pd.read_csv(a.species_tsv, sep="\t")["species"].dropna().unique())
    n_sp = len(species)

    icj = json.loads(Path(a.instrument_json).read_text())
    ident = icj["identification"]
    novel = icj["novelty"]
    con = json.loads(Path(a.congruence_json).read_text())
    cid, ccmp = con["identification"], con["comparison"]

    def from_icj(name):
        i, v = pick(ident, "instrument", name), pick(novel, "instrument", name)
        return (i["correct"], i["reached_a_name"],
                parse_point(v["nested_point"]), v["nested_margin"],
                parse_point(v["best_point_in_sample"]), v["best_margin_in_sample"])

    def novelty_from_con(name):
        v = pick(ccmp, "instrument", name)
        return (parse_point(v["nested_point"]), v["nested_margin"],
                parse_point(v["best_point"]), v["best_margin_in_sample"])

    M = []

    # 1 character matrix -----------------------------------------------------
    c, nmd, np_, nm, bp, bm = from_icj("character matrix")
    M.append(dict(label="Character matrix", correct=c, named=nmd, nested=np_,
                  nested_margin=nm, insample=bp, insample_margin=bm,
                  colour=C_MATRIX, hatch=None, note=""))

    # 2 dichotomous key, crisp -----------------------------------------------
    c, nmd, np_, nm, bp, bm = from_icj("dichotomous key")
    M.append(dict(label="Dichotomous key (crisp)", correct=c, named=nmd, nested=np_,
                  nested_margin=nm, insample=bp, insample_margin=bm,
                  colour=C_KEY, hatch=None, note=""))

    # 3 graded (fuzzy) key ---------------------------------------------------
    fz = pd.read_csv(a.fuzzy_key_tsv, sep="\t")
    kn = fz[fz["arm"] == "known"]
    v = pick(novel, "instrument", "graded key score")
    M.append(dict(label="Key, graded (fuzzy)", correct=int(kn["correct"].sum()),
                  named=int(kn["resolved"].sum()),
                  nested=parse_point(v["nested_point"]), nested_margin=v["nested_margin"],
                  insample=parse_point(v["best_point_in_sample"]),
                  insample_margin=v["best_margin_in_sample"],
                  colour=C_FUZZY, hatch=None, note=""))

    # 4 knowledge graph ------------------------------------------------------
    c, nmd, np_, nm, bp, bm = from_icj("knowledge graph")
    M.append(dict(label="Knowledge graph", correct=c, named=nmd, nested=np_,
                  nested_margin=nm, insample=bp, insample_margin=bm,
                  colour=C_GRAPH, hatch=None, note=""))

    # 5 model-proposed characters alone (selection leak closed) --------------
    i = pick(cid, "identifier", "vlm")
    np_, nm, bp, bm = novelty_from_con("vlm alone, selection leak closed")
    M.append(dict(label="Model-proposed characters alone", correct=i["correct"],
                  named=i["specimens_named"], nested=np_, nested_margin=nm,
                  insample=bp, insample_margin=bm, colour=C_MODEL, hatch=None,
                  note="selection leak closed"))

    # 6 matrix + model-proposed characters -----------------------------------
    i = pick(cid, "identifier", "matrix + vlm")
    np_, nm, bp, bm = novelty_from_con("matrix + vlm, selection leak closed")
    M.append(dict(label="Matrix + model characters", correct=i["correct"],
                  named=i["specimens_named"], nested=np_, nested_margin=nm,
                  insample=bp, insample_margin=bm, colour=C_MATRIX, hatch="///",
                  note="selection leak closed"))

    # 7 key + model-proposed characters, crisp -------------------------------
    it = pd.read_csv(a.key_plus_ident_tsv, sep="\t")
    unnamed = {"unresolved", "species has no other specimen"}
    arms = pd.read_csv(a.key_plus_novelty_tsv, sep="\t")
    calls = key_calls(arms[arms["arm"] == "species_withheld"],
                      arms[arms["arm"] == "specimen_withheld"])
    row, _, _, _ = summarise("key + model characters", calls, species, CEILINGS)
    M.append(dict(label=f"Key + model characters (crisp) {DAGGER}",
                  correct=int(it["loo_ok"].sum()),
                  named=int((~it["loo"].isin(unnamed)).sum()),
                  nested=parse_point(row["nested_point"]),
                  nested_margin=row["nested_margin"],
                  insample=parse_point(row["best_point_in_sample"]),
                  insample_margin=row["best_margin_in_sample"],
                  colour=C_KEY, hatch="///",
                  note="upper bound: selection leak NOT closed"))

    # 8 key + model-proposed characters, graded ------------------------------
    f8 = pd.read_csv(a.key_plus_fuzzy_tsv, sep="\t")
    kn8 = f8[f8["arm"] == "known"]
    r8 = f8[f8["resolved"] == True]                                     # noqa: E712
    h, k = r8[r8["arm"] == "held_out"], r8[r8["arm"] == "known"]
    calls8 = calls_from_scores(h["score"], k["score"], h["species"], k["species"],
                               higher_is_novel=False)
    row8, _, _, _ = summarise("key + model characters, graded", calls8, species, CEILINGS)
    M.append(dict(label=f"Key + model characters, graded {DAGGER}",
                  correct=int(kn8["correct"].sum()), named=int(kn8["resolved"].sum()),
                  nested=parse_point(row8["nested_point"]),
                  nested_margin=row8["nested_margin"],
                  insample=parse_point(row8["best_point_in_sample"]),
                  insample_margin=row8["best_margin_in_sample"],
                  colour=C_FUZZY, hatch="///",
                  note="upper bound: selection leak NOT closed"))

    # the ghost: the same model-proposed characters before the leak was closed
    gn, gnm, gb, gbm = novelty_from_con("vlm alone")
    ghost = dict(label="Model-proposed characters alone, before the selection leak "
                       "was closed", nested=gn, nested_margin=gnm,
                 insample=gb, insample_margin=gbm, colour=C_GHOST,
                 note="reference only: selection leak open, novelty inflated")

    n_spec = int(icj.get("specimens") or len(pd.read_csv(a.species_tsv, sep="\t")))
    for m in M:
        m["of"] = n_spec
        m["of_species"] = n_sp
        m["pct"] = 100.0 * m["correct"] / n_spec
        m["pct_named"] = 100.0 * m["named"] / n_spec
        m["ci"] = wilson(m["correct"], n_spec)
    return M, ghost, species, n_spec


# ─────────────────────────────────────────────────────────────────────────────
# table
# ─────────────────────────────────────────────────────────────────────────────
def number_methods(methods):
    """Panel a is ordered best to worst; that order is the numbered key the table and
    both panels share."""
    order = sorted(methods, key=lambda m: m["pct"], reverse=True)
    for n, m in enumerate(order, 1):
        m["n"] = n
    return order


def write_table(order, ghost, out_dir, n_spec, n_sp, pd):
    rows = []
    for m in order:
        c, f, _ = m["nested"]
        bc, bf, _ = m["insample"]
        rows.append({
            "n": m["n"], "method": m["label"].replace(f" {DAGGER}", ""),
            "upper_bound": bool(DAGGER in m["label"]),
            "naming_correct": m["correct"], "naming_of": n_spec,
            "naming_reached_a_name": m["named"],
            "naming_pct_of_all": round(m["pct"], 1),
            "naming_95CI_low_pct": m["ci"][0], "naming_95CI_high_pct": m["ci"][1],
            "novelty_of_species": n_sp,
            "nested_caught": c, "nested_false_alarms": f,
            "nested_margin": m["nested_margin"],
            "in_sample_caught": bc, "in_sample_false_alarms": bf,
            "in_sample_margin": m["insample_margin"],
            "note": m["note"]})
    c, f, _ = ghost["nested"]
    bc, bf, _ = ghost["insample"]
    rows.append({"n": "", "method": ghost["label"], "upper_bound": False,
                 "naming_correct": "", "naming_of": "", "naming_reached_a_name": "",
                 "naming_pct_of_all": "", "naming_95CI_low_pct": "",
                 "naming_95CI_high_pct": "", "novelty_of_species": n_sp,
                 "nested_caught": c, "nested_false_alarms": f,
                 "nested_margin": ghost["nested_margin"], "in_sample_caught": bc,
                 "in_sample_false_alarms": bf,
                 "in_sample_margin": ghost["insample_margin"], "note": ghost["note"]})
    df = pd.DataFrame(rows)
    p = out_dir / "methods_summary.tsv"
    df.to_csv(p, sep="\t", index=False)
    return df, p


# ─────────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────────
# Hand-tuned label offsets in points, keyed by the number a method takes once panel a
# has been ordered. Only cosmetic: a re-run on other data still draws, with the default.
B_OFFSETS = {1: (9, -13), 2: (2, 9), 3: (7, -14), 4: (9, -7),
             5: (9, -3), 6: (-7, -15), 7: (9, 3), 8: (8, -11)}
GHOST_TEXT_XY = (26, 68)         # data coordinates: clear space above the ghost


def draw(order, ghost, out_dir, n_spec, n_sp, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#555555", "axes.labelcolor": "#222222",
                         "text.color": "#222222", "xtick.color": "#555555",
                         "ytick.color": "#555555"})

    fig = plt.figure(figsize=(14.6, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], left=0.185, right=0.985,
                          top=0.852, bottom=0.205, wspace=0.26)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ── panel A ─────────────────────────────────────────────────────────────
    ys = list(range(len(order)))[::-1]
    for y, m in zip(ys, order):
        axA.barh(y, m["pct_named"], height=0.70, color=lighten(m["colour"], 0.80),
                 edgecolor=lighten(m["colour"], 0.55), linewidth=0.7, zorder=1)
        axA.barh(y, m["pct"], height=0.70, color=lighten(m["colour"], 0.18),
                 edgecolor=m["colour"], linewidth=1.0, hatch=m["hatch"], zorder=2)
        lo, hi = m["ci"]
        axA.plot([lo, hi], [y, y], color="#1a1a1a", lw=1.2, zorder=4,
                 solid_capstyle="butt")
        for x in (lo, hi):
            axA.plot([x, x], [y - 0.14, y + 0.14], color="#1a1a1a", lw=1.2, zorder=4)
        axA.text(max(hi, m["pct_named"]) + 1.6, y, f"{m['correct']}/{n_spec}",
                 va="center", ha="left", fontsize=9.2, color="#333333", zorder=5)

    axA.set_yticks(ys)
    axA.set_yticklabels([f"{m['n']}  {m['label']}" for m in order], fontsize=9.6)
    axA.set_xlim(0, 118)
    axA.set_xticks([0, 20, 40, 60, 80, 100])
    axA.set_ylim(-0.75, len(order) - 0.25)
    axA.set_xlabel(f"per cent of all {n_spec} specimens")
    axA.set_title("a   Naming a described specimen", loc="left", fontsize=12.5,
                  fontweight="bold", pad=30)
    axA.xaxis.grid(True, color="#e2e2e2", lw=0.7, zorder=0)
    axA.set_axisbelow(True)
    axA.legend(handles=[
        Patch(facecolor="#9e9e9e", edgecolor="#4d4d4d", label="named correctly"),
        Patch(facecolor="#e2e2e2", edgecolor="#bdbdbd", label="reached any name"),
        Line2D([0], [0], color="#1a1a1a", lw=1.2, label="Wilson 95% interval")],
        loc="lower right", bbox_to_anchor=(1.0, 1.012), ncol=3, fontsize=8.6,
        frameon=False, borderpad=0.4, handlelength=1.6, columnspacing=1.5)

    # ── panel B ─────────────────────────────────────────────────────────────
    axB.plot([0, 100], [0, 100], ls=(0, (5, 4)), color="#9e9e9e", lw=1.1, zorder=1)
    axB.text(97, 95, "no information", rotation=45, rotation_mode="anchor",
             ha="right", va="top", fontsize=8.8, color="#8a8a8a",
             transform_rotates_text=True)

    def pt(p):
        c, f, of = p
        return 100.0 * f / of, 100.0 * c / of

    # The ghost: the nested point of the same instrument before the selection leak was
    # closed, with a dotted line to where closing it moved the instrument. Drawn first,
    # so it sits behind everything.
    gx, gy = pt(ghost["nested"])
    leaky = next((m for m in order if m["note"] == "selection leak closed"
                  and m["colour"] == C_MODEL), None)
    if leaky is not None:
        lx, ly = pt(leaky["nested"])
        axB.annotate("", xy=(lx, ly), xytext=(gx, gy), zorder=2,
                     arrowprops=dict(arrowstyle="-|>", color=C_GHOST, lw=1.0,
                                     ls=(0, (2, 2)), shrinkA=7, shrinkB=9,
                                     mutation_scale=11))
    axB.scatter([gx], [gy], s=84, facecolor=C_GHOST, edgecolor=C_GHOST, alpha=0.55,
                zorder=3, linewidths=1.2)
    axB.annotate(f"{leaky['n'] if leaky else 5} before the selection\nleak was closed",
                 xy=(gx, gy), xytext=GHOST_TEXT_XY, textcoords="data", fontsize=8.4,
                 color="#78909c", ha="center", va="bottom", zorder=6,
                 arrowprops=dict(arrowstyle="-", color="#b8c4ca", lw=0.8,
                                 shrinkA=4, shrinkB=7))

    for m in order:
        nx, ny = pt(m["nested"])
        bx, by = pt(m["insample"])
        axB.plot([nx, bx], [ny, by], color=m["colour"], lw=1.1, alpha=0.85, zorder=4)
        axB.scatter([bx], [by], s=86, facecolor="white", edgecolor=m["colour"],
                    linewidths=1.6, zorder=5,
                    hatch=None, marker="o")
        axB.scatter([nx], [ny], s=86, facecolor=m["colour"], edgecolor=m["colour"],
                    linewidths=1.0, zorder=6, marker="o")
        dx, dy = B_OFFSETS.get(m["n"], (7, 7))
        lab = f"{m['n']}{DAGGER}" if DAGGER in m["label"] else f"{m['n']}"
        axB.annotate(lab, xy=(nx, ny), xytext=(dx, dy), textcoords="offset points",
                     fontsize=10.2, fontweight="bold", color=m["colour"], zorder=7)

    axB.set_xlim(-3, 103)
    axB.set_ylim(-3, 103)
    axB.set_xticks([0, 20, 40, 60, 80, 100])
    axB.set_yticks([0, 20, 40, 60, 80, 100])
    axB.set_xlabel(f"described species wrongly called new (% of {n_sp})")
    axB.set_ylabel(f"withheld species recognised (% of {n_sp})")
    axB.set_title("b   Recognising a species never seen", loc="left", fontsize=12.5,
                  fontweight="bold", pad=30)
    axB.grid(True, color="#ededed", lw=0.7, zorder=0)
    axB.set_axisbelow(True)
    axB.legend(handles=[
        Line2D([0], [0], marker="o", ls="none", markerfacecolor="#4d4d4d",
               markeredgecolor="#4d4d4d", markersize=8.2,
               label="nested estimate (honest)"),
        Line2D([0], [0], marker="o", ls="none", markerfacecolor="white",
               markeredgecolor="#4d4d4d", markersize=8.2,
               label="best point in sample (optimistic)"),
        Line2D([0], [0], color="#4d4d4d", lw=1.1, label="the same method, joined")],
        loc="lower right", fontsize=8.6, frameon=True, framealpha=0.95,
        edgecolor="#cccccc", borderpad=0.6, title="numbers as in panel a",
        title_fontsize=8.6)

    note = "\n".join([
        f"Nested: the threshold and the series rule are chosen on the other {n_sp - 1} "
        "species and applied to the one left out. The hollow marker is the best point "
        "obtainable in sample, so the length of each line is the optimism of in-sample "
        "tuning.",
        f"{DAGGER} Upper bound: for these two the selection leak is not closed — "
        "characters proposed from a species are still in the matrix when that species is "
        "withheld. The faint marker is the one instrument for which the leak could be",
        "closed, shown as it stood before closing. Both arms of both questions are "
        "hold-outs, and a specimen that reaches no name counts as not named correctly. "
        f"n = {n_spec} specimens, {n_sp} species."])
    fig.text(0.012, 0.132, note, fontsize=8.6, color="#4a4a4a", va="top", ha="left",
             linespacing=1.55)

    png, pdf = out_dir / "fig_methods_summary.png", out_dir / "fig_methods_summary.pdf"
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    import matplotlib.pyplot as _p
    _p.close(fig)
    return png, pdf


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Summary figure and table for every method tried on one data set.")
    ap.add_argument("--monograph", required=True,
                    help="root of the monograph results; every other path defaults "
                         "under it and can be overridden")
    ap.add_argument("--scripts_dir", default=str(Path(__file__).resolve().parent),
                    help="directory holding the biorag_* helper modules")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--instrument_json", default=None,
                    help="instrument_comparison_v3.json")
    ap.add_argument("--congruence_json", default=None,
                    help="congruence_summary.json (model-proposed characters)")
    ap.add_argument("--species_tsv", default=None,
                    help="any per-specimen table with a `species` column; its sorted "
                         "unique values are the species list")
    ap.add_argument("--fuzzy_key_tsv", default=None,
                    help="graded key, leave-one-out: fuzzy_key_specimens.tsv")
    ap.add_argument("--key_plus_ident_tsv", default=None,
                    help="key built with the model-proposed characters: "
                         "identification_test.tsv")
    ap.add_argument("--key_plus_novelty_tsv", default=None,
                    help="novelty_specimens.tsv from the calibration of that key")
    ap.add_argument("--key_plus_fuzzy_tsv", default=None,
                    help="graded version of that key: fuzzy_key_specimens.tsv")
    ap.add_argument("--dpi", type=int, default=220)
    a = ap.parse_args()

    M = Path(a.monograph)
    a.out_dir = Path(a.out_dir or M / "methods_summary")
    a.instrument_json = a.instrument_json or M / "instrument_comparison_v3" / \
        "instrument_comparison_v3.json"
    a.congruence_json = a.congruence_json or M / "congruence_compare" / \
        "congruence_summary.json"
    a.species_tsv = a.species_tsv or M / "calibration" / "matrix_detection_scores.tsv"
    a.fuzzy_key_tsv = a.fuzzy_key_tsv or M / "key_fuzzy_loo" / "fuzzy_key_specimens.tsv"
    a.key_plus_ident_tsv = a.key_plus_ident_tsv or M / "key_plus_vlm" / \
        "identification_test.tsv"
    a.key_plus_novelty_tsv = a.key_plus_novelty_tsv or M / "calibration_key_plus_vlm" / \
        "novelty_specimens.tsv"
    a.key_plus_fuzzy_tsv = a.key_plus_fuzzy_tsv or M / "key_plus_vlm" / "fuzzy_loo" / \
        "fuzzy_key_specimens.tsv"

    missing = [str(p) for p in (a.instrument_json, a.congruence_json, a.species_tsv,
                                a.fuzzy_key_tsv, a.key_plus_ident_tsv,
                                a.key_plus_novelty_tsv, a.key_plus_fuzzy_tsv)
               if not Path(p).exists()]
    if missing:
        sys.exit("missing input:\n  " + "\n  ".join(missing))

    sys.path.insert(0, str(Path(a.scripts_dir).resolve()))
    import pandas as pd
    from biorag_instrument_compare_v3 import calls_from_scores, summarise
    from biorag_calibrate_v1 import key_calls
    from biorag_congruence_compare_v1 import wilson

    a.out_dir.mkdir(parents=True, exist_ok=True)
    methods, ghost, species, n_spec = gather(a, pd, calls_from_scores, summarise,
                                             key_calls, wilson)
    order = number_methods(methods)
    df, tsv = write_table(order, ghost, a.out_dir, n_spec, len(species), pd)
    png, pdf = draw(order, ghost, a.out_dir, n_spec, len(species), a.dpi)

    print(f"biorag_methods_summary_figure v{VERSION}   "
          f"{n_spec} specimens, {len(species)} species")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df.to_string(index=False))
    for p in (tsv, png, pdf):
        print("->", p)


if __name__ == "__main__":
    main()
