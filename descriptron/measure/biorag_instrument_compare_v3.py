#!/usr/bin/env python3
"""
biorag_instrument_compare_v3.py — key, knowledge graph and character matrix, on both questions
==============================================================================================

The same data can be asked two questions by three instruments:

  is this specimen one of the described species, and which?     (identification)
  does this series belong to none of them?                       (novelty)

  dichotomous key      ordered couplets, each a threshold with the range observed either side
  knowledge graph      every range a treatment asserts, read as a constraint
  character matrix     distance to the nearest species in units of that species' own spread,
                       over independent character sets that must agree

v2 of this script put the three on the same footing for novelty: both arms are hold-outs, so
no specimen is scored against a statistic it helped compute. v3 adds what v2 left out.

1. EVERY OPERATING POINT. v2 swept 60 quantiles of the scores it was handed, so the thresholds
   tried depended on which sets were in the pot and the curve moved when a set was added. v3
   tries every distinct threshold, for every instrument.
2. A NESTED MARGIN. The best margin over a sweep is chosen on the species it is reported for.
   The nested figure chooses threshold and rule on the other species and applies them to the
   one left out — what a user who calibrates on their own material will get. It is reported
   beside the in-sample best, never instead of it, with a Wilson interval on the detection.
3. ONE DENOMINATOR. A species the key cannot reach at all (every path stops on an unmeasured
   character) has not been recognised. v2 dropped such species from the key's total; v3
   counts them as missed, so all three instruments are out of the same number of species.
4. IDENTIFICATION, for all three, under leave-one-specimen-out — v2's table said "n/a" for the
   matrix — by series size, and cross-tabulated: where two instruments give the same name,
   how often is it right? Agreement between instruments that fail in different ways is a
   usable confidence signal even where none is reliable alone.

Inputs are the per-specimen tables the three evaluation scripts already write:

  python biorag_instrument_compare_v3.py --calibration "$M/calibration" \\
      --key_loo "$M/key/identification_test.tsv" \\
      --graph_scores "$M/graph_test_loo/graph_specimen_scores.tsv" \\
      [--fuzzy "$M/key_fuzzy_loo/fuzzy_key_specimens.tsv"] --out_dir "$M/instrument_comparison_v3"

`--calibration` is the output of biorag_calibrate_v1.py (novelty_specimens.tsv and the
matrix_* score tables). Nothing is re-scored here, so the comparison cannot drift from the
calibration the user was shown.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biorag_congruence_compare_v1 import (sweep as g_sweep, nested, identify,   # noqa: E402
                                          wilson, mcnemar, RULES)
from biorag_calibrate_v1 import key_calls                                # noqa: E402

VERSION = "3.0"
COLOURS = {"dichotomous key": "#1565c0", "character matrix": "#2e7d32",
           "knowledge graph": "#ef6c00", "graded key score": "#6a1b9a"}
BASE_SETS = ("size", "ratio", "landmark", "colour")


def calls_from_scores(det: pd.Series, fal: pd.Series, species_det, species_fal, higher_is_novel=True):
    """Per-species calls at every distinct threshold of a one-number-per-specimen score."""
    vals = np.unique(np.concatenate([det.dropna().values, fal.dropna().values]))
    out = {}
    for t in vals:
        for rule, fn in RULES.items():
            pair = []
            for sc, sp in ((det, species_det), (fal, species_fal)):
                flag = (sc > t) if higher_is_novel else (sc < t)
                flag = flag & sc.notna()
                g = pd.DataFrame({"species": sp, "flag": flag}).groupby("species")["flag"] \
                    .agg(["sum", "size"])
                pair.append(g.apply(lambda r: bool(fn(r["sum"], r["size"])), axis=1))
            out[(float(t), rule)] = tuple(pair)
    return out


def summarise(name, calls, species, ceilings):
    n = len(species)
    rows = []
    for (t, rule), (d, f) in calls.items():
        d = d.reindex(species, fill_value=False).astype(bool)
        f = f.reindex(species, fill_value=False).astype(bool)
        rows.append({"instrument": name, "threshold": t, "rule": rule, "caught": int(d.sum()),
                     "false_alarms": int(f.sum()), "of": n,
                     "detection": d.mean(), "false_alarm": f.mean(), "margin": d.mean() - f.mean()})
    sw = pd.DataFrame(rows)
    b = sw.sort_values(["margin", "caught"], ascending=False).iloc[0]
    ne = nested(sw, calls, species)
    row = {"instrument": name, "operating_points": len(sw),
           "best_margin_in_sample": round(float(b["margin"]), 3),
           "best_point_in_sample": f"{int(b['caught'])}/{n} caught, {int(b['false_alarms'])}/{n} false",
           "best_rule_in_sample": b["rule"],
           "nested_margin": ne["margin"],
           "nested_point": f"{ne['caught']}/{n} caught, {ne['false_alarms']}/{n} false",
           "nested_detection_95CI": "{}–{}%".format(*wilson(ne["caught"], n)),
           "nested_false_alarm_95CI": "{}–{}%".format(*wilson(ne["false_alarms"], n))}
    at = {}
    for c in ceilings:
        ok = sw[sw["false_alarms"] <= c]
        if len(ok):
            p = ok.sort_values(["caught", "false_alarms"], ascending=[False, True]).iloc[0]
            row[f"caught_at_max_{c}_false"] = int(p["caught"])
            at[c] = calls[(p["threshold"], p["rule"])][0].reindex(species, fill_value=False).astype(bool)
        else:
            row[f"caught_at_max_{c}_false"] = None
    return row, sw, ne, at


def by_series(ok: pd.Series, species: pd.Series, n_spec: dict):
    n = species.map(n_spec)
    out = {}
    for lo, hi, lab in ((1, 3, "2-3"), (4, 5, "4-5"), (6, 999, "6+")):
        m = (n >= lo) & (n <= hi)
        out[lab] = (int(ok[m].sum()), int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description="Key, graph and matrix on identification and novelty")
    ap.add_argument("--calibration", required=True, help="output dir of biorag_calibrate_v1.py")
    ap.add_argument("--key_loo", required=True, help="key/identification_test.tsv (leave one out)")
    ap.add_argument("--graph_scores", default=None, help="graph_test_loo/graph_specimen_scores.tsv")
    ap.add_argument("--fuzzy", default=None, help="key_fuzzy_loo/fuzzy_key_specimens.tsv")
    ap.add_argument("--min_sets", type=int, default=2)
    ap.add_argument("--ceilings", nargs="*", type=int, default=[1, 3, 6])
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    cal, out = Path(a.calibration), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(cal / "matrix_detection_scores.tsv", sep="\t")
    fal = pd.read_csv(cal / "matrix_false_alarm_scores.tsv", sep="\t")
    ranks = pd.read_csv(cal / "matrix_identification_distances.tsv", sep="\t")
    arms = pd.read_csv(cal / "novelty_specimens.tsv", sep="\t")
    species = sorted(det["species"].unique())
    n_spec = det.groupby("species").size().to_dict()
    sets = [s for s in BASE_SETS if s in det.columns]

    # ── novelty ──────────────────────────────────────────────────────────────
    instruments = {}
    _, mcalls = g_sweep(det, fal, sets, a.min_sets)
    instruments["character matrix"] = mcalls
    instruments["dichotomous key"] = key_calls(arms[arms.arm == "species_withheld"],
                                               arms[arms.arm == "specimen_withheld"])
    graph = None
    if a.graph_scores and Path(a.graph_scores).exists():
        graph = pd.read_csv(a.graph_scores, sep="\t")
        instruments["knowledge graph"] = calls_from_scores(
            graph["novel_rate"], graph["false_alarm_rate"], graph["species"], graph["species"])
    fz = None
    if a.fuzzy and Path(a.fuzzy).exists():
        fz = pd.read_csv(a.fuzzy, sep="\t")
        fz = fz[fz["resolved"] == True]                                       # noqa: E712
        h, k = fz[fz.arm == "held_out"], fz[fz.arm == "known"]
        instruments["graded key score"] = calls_from_scores(
            h["score"], k["score"], h["species"], k["species"], higher_is_novel=False)

    table, sweeps, at_ceiling = [], [], {}
    for name, calls in instruments.items():
        row, sw, ne, at = summarise(name, calls, species, a.ceilings)
        table.append(row)
        sweeps.append(sw)
        at_ceiling[name] = at
    tab = pd.DataFrame(table).sort_values("nested_margin", ascending=False)
    tab.to_csv(out / "S2_novelty_by_instrument.tsv", sep="\t", index=False)
    allsw = pd.concat(sweeps, ignore_index=True)
    allsw.to_csv(out / "instrument_sweeps_exact.tsv", sep="\t", index=False)

    paired = []
    for name in instruments:
        if name == "character matrix":
            continue
        for c in a.ceilings:
            if c in at_ceiling[name] and c in at_ceiling["character matrix"]:
                only_m, only_o, p = mcnemar(at_ceiling["character matrix"][c], at_ceiling[name][c])
                paired.append({"comparison": f"character matrix vs {name}", "max_false_alarms": c,
                               "only_the_matrix_caught": len(only_m), "only_the_other_caught": len(only_o),
                               "mcnemar_exact_p": round(p, 4),
                               "species_only_the_matrix_caught": ", ".join(only_m),
                               "species_only_the_other_caught": ", ".join(only_o)})
    pd.DataFrame(paired).to_csv(out / "S2b_paired_novelty.tsv", sep="\t", index=False)

    # ── identification ───────────────────────────────────────────────────────
    k = pd.read_csv(a.key_loo, sep="\t").set_index("specimen_id")
    truth = k["species"]
    names = pd.DataFrame(index=k.index)
    names["key"] = k["loo"].where(k["loo"].isin(species))
    names["matrix"] = identify(ranks, sets)["named"].reindex(k.index)
    if graph is not None:
        names["graph"] = graph.set_index("specimen_id")["id_species"].reindex(k.index)
    ident = []
    for col, label in (("key", "dichotomous key"), ("graph", "knowledge graph"),
                       ("matrix", "character matrix")):
        if col not in names:
            continue
        ok = (names[col] == truth)
        bs = by_series(ok, truth, n_spec)
        ident.append({"instrument": label, "reached_a_name": int(names[col].notna().sum()),
                      "correct": int(ok.sum()), "of": len(k),
                      "correct_of_all_pct": round(100 * ok.mean(), 1),
                      "correct_of_all_95CI": "{}–{}%".format(*wilson(int(ok.sum()), len(k))),
                      "correct_of_named_pct": round(100 * ok.sum() / max(1, names[col].notna().sum()), 1),
                      **{f"series_{lab}": f"{c}/{n}" for lab, (c, n) in bs.items()}})
    for s in sets:                                   # each character set alone, for the supplement
        r = identify(ranks, [s])
        ok = (r["named"] == r["species"])
        ident.append({"instrument": f"character matrix — {s} only", "reached_a_name": len(r),
                      "correct": int(ok.sum()), "of": len(k),
                      "correct_of_all_pct": round(100 * ok.sum() / len(k), 1),
                      "correct_of_named_pct": round(100 * ok.mean(), 1)})
    if "colour" in sets and len(sets) > 1:
        r = identify(ranks, [s for s in sets if s != "colour"])
        ok = (r["named"] == r["species"])
        ident.append({"instrument": "character matrix — without colour", "reached_a_name": len(r),
                      "correct": int(ok.sum()), "of": len(k),
                      "correct_of_all_pct": round(100 * ok.sum() / len(k), 1),
                      "correct_of_named_pct": round(100 * ok.mean(), 1)})
    idt = pd.DataFrame(ident)
    idt.to_csv(out / "S3_identification_by_instrument.tsv", sep="\t", index=False)

    # agreement between instruments
    agree = []

    def case(label, mask, pick):
        n = int(mask.sum())
        okn = int((pick[mask] == truth[mask]).sum())
        agree.append({"case": label, "specimens": n, "correct": okn,
                      "accuracy_pct": round(100 * okn / n, 1) if n else None,
                      "95CI": "{}–{}%".format(*wilson(okn, n)) if n else ""})
    K, Mx = names["key"], names["matrix"]
    case("key and matrix give the same name", K.notna() & (K == Mx), Mx)
    case("key and matrix disagree — the key's name", K.notna() & Mx.notna() & (K != Mx), K)
    case("key and matrix disagree — the matrix's name", K.notna() & Mx.notna() & (K != Mx), Mx)
    case("key reaches no name — the matrix's name", K.isna() & Mx.notna(), Mx)
    if "graph" in names:
        G = names["graph"]
        case("graph and matrix give the same name", G.notna() & (G == Mx), Mx)
        case("graph and matrix disagree — the graph's name", G.notna() & Mx.notna() & (G != Mx), G)
        case("graph and matrix disagree — the matrix's name", G.notna() & Mx.notna() & (G != Mx), Mx)
        case("all three give the same name", K.notna() & (K == Mx) & (G == Mx), Mx)
        case("key and graph agree, matrix differs — their name", K.notna() & (K == G) & (K != Mx), K)
    ag = pd.DataFrame(agree)
    ag.to_csv(out / "S4_agreement_between_instruments.tsv", sep="\t", index=False)
    names.assign(species=truth).to_csv(out / "identification_by_specimen.tsv", sep="\t")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.9), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    ax.plot([0, 100], [0, 100], color="#bdbdbd", lw=1.0, ls="--", zorder=1)
    ax.text(72, 76, "no information", fontsize=7, color="#9e9e9e", rotation=40)
    for name, g in allsw.groupby("instrument"):
        e = (g.sort_values("detection", ascending=False).drop_duplicates("false_alarm")
             .sort_values("false_alarm"))
        e = e[e["detection"].cummax() == e["detection"]]
        ax.step(100 * e["false_alarm"], 100 * e["detection"], where="post", lw=1.7,
                color=COLOURS.get(name, "#555"), label=name)
        r = tab[tab.instrument == name].iloc[0]
        c_, f_ = [int(x.split("/")[0]) for x in r["nested_point"].replace(" caught", "")
                  .replace(" false", "").split(", ")]
        n = len(species)
        ax.plot(100 * f_ / n, 100 * c_ / n, marker="D", ms=8, mec="white", mew=1.2,
                color=COLOURS.get(name, "#555"), zorder=5)
    ax.plot([], [], marker="D", ms=7, ls="", color="#555", mec="white",
            label="nested estimate (operating point\nchosen without the species it is scored on)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("described species wrongly called new (%)", fontsize=9)
    ax.set_ylabel("species the instrument has never seen, recognised (%)", fontsize=9)
    ax.set_title("A  Recognising an unseen species\nbest detection at each false-alarm rate, "
                 "every operating point", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.4, frameon=False, loc="lower right")

    ax = axes[1]
    main3 = idt[idt.instrument.isin(["dichotomous key", "knowledge graph", "character matrix"])]
    y = np.arange(len(main3))[::-1]
    for yi, (_, r) in zip(y, main3.iterrows()):
        col = COLOURS.get(r["instrument"], "#555")
        ax.barh(yi, r["correct_of_all_pct"], color=col, height=0.55)
        lo, hi = [float(v) for v in r["correct_of_all_95CI"].rstrip("%").split("–")]
        ax.plot([lo, hi], [yi, yi], color="#263238", lw=1.2)
        ax.text(min(97, hi + 2), yi, f"{r['correct']}/{r['of']}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(main3["instrument"], fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xlabel("described specimens given the right name,\nown record withheld (% of all specimens)",
                  fontsize=9)
    ax.set_title("B  Naming a described specimen", fontsize=9.5, loc="left")

    ax = axes[2]
    labs = ["2-3", "4-5", "6+"]
    x = np.arange(len(labs))
    wid = 0.26
    for i, (_, r) in enumerate(main3.iterrows()):
        vals = []
        for lab in labs:
            c_, n_ = [int(v) for v in str(r.get(f"series_{lab}", "0/0")).split("/")]
            vals.append(100 * c_ / n_ if n_ else 0)
        ax.bar(x + (i - 1) * wid, vals, wid, color=COLOURS.get(r["instrument"], "#555"),
               label=r["instrument"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} specimens" for lab in labs], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("right name (%)", fontsize=9)
    ax.set_xlabel("specimens available for the species", fontsize=9)
    ax.set_title("C  The same, by length of series", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_instrument_comparison_v3.png", dpi=220)
    fig.savefig(out / "fig_instrument_comparison_v3.pdf")
    plt.close(fig)

    (out / "instrument_comparison_v3.json").write_text(json.dumps({
        "version": VERSION, "species": len(species), "specimens": int(len(k)),
        "min_sets": a.min_sets, "novelty": tab.to_dict("records"), "paired_novelty": paired,
        "identification": ident, "agreement": agree,
        "note": "both arms are hold-outs; nested = operating point chosen on the other species; "
                "a species an instrument cannot score counts as missed"}, indent=2))
    pd.set_option("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 60)
    print(tab.to_string(index=False), "\n")
    print(idt.to_string(index=False), "\n")
    print(ag.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
