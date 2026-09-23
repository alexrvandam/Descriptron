#!/usr/bin/env python3
"""
biorag_instrument_compare_v2.py — the three instruments, measured the same way
==============================================================================

A key, a knowledge graph and a character matrix can all be asked the same two
questions, and until they are asked under the same conditions their answers cannot
be compared. The condition that matters is that **the specimen being tested must
not have helped compute the thing it is tested against**. Withholding it from the
prediction while leaving it inside the range, the asserted interval or the spread
is not a hold-out, and it silently flatters whichever instrument is measured that
way — which is how a first pass made the key look far better than the matrix when
the matrix was the only one being measured honestly.

Both arms, for every instrument, at the species level:

  detection    hold the whole species out, rebuild/re-derive without it, score its
               specimens. How many of the species the instrument has never seen does
               it recognise as outside?
  false alarm  hold one specimen out, leave its species in place, score it against
               everything re-derived without it. How many described species does the
               instrument wrongly call new?

Each instrument's threshold is then swept across its whole range, so the comparison
is a curve rather than one arbitrary operating point, and the instruments are read
off at a matched false-alarm rate.

  python biorag_instrument_compare_v2.py --matrix_dir "$M/compiled_key_tier" \\
      --taxon_profile <profile.yaml> --key_sweep "$M/calibration/novelty_tradeoff.tsv" \\
      --graph_sweep "$M/graph_test_loo/graph_tradeoff.tsv" --out_dir "$M/instrument_comparison_v2"
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                      # noqa: E402
from biorag_novelty_score_v1 import Reference, SETS, score_candidate     # noqa: E402

VERSION = "2.0"
RULES = {"any specimen": lambda f, n: f > 0,
         "a majority of the series": lambda f, n: f > n / 2,
         "the whole series": lambda f, n: f == n}


def matrix_scores(ref, min_sets: int):
    """g for every specimen in both arms, then the per-species counts at any threshold.

    detection   specimen scored with its whole species removed from the reference
    false alarm specimen scored with only itself removed, its species still present
    """
    det, fal = [], []
    for sid in ref.raw.index:
        sp = ref.species_of.get(sid)
        row_d, row_f = {"specimen_id": sid, "species": sp}, {"specimen_id": sid, "species": sp}
        for name in SETS:
            if name not in ref.tables or not len(ref.tables[name].columns):
                continue
            cand = ref.tables[name].loc[sid]
            d = score_candidate(ref, name, cand, exclude_species=(sp,), exclude_specimens=(sid,))
            f = score_candidate(ref, name, cand, exclude_species=(), exclude_specimens=(sid,))
            row_d[name], row_f[name] = d.get("g", math.nan), f.get("g", math.nan)
        det.append(row_d)
        fal.append(row_f)
    return pd.DataFrame(det), pd.DataFrame(fal)


def matrix_sweep(det: pd.DataFrame, fal: pd.DataFrame, min_sets: int):
    """Sweep the g threshold. A specimen is called novel when it lies beyond the threshold
    in at least `min_sets` independent character sets — the congruence requirement."""
    cols = [c for c in det.columns if c in SETS]
    rows = []
    allg = pd.concat([det[cols].stack(), fal[cols].stack()]).dropna()
    if not len(allg):
        return pd.DataFrame()
    for t in np.quantile(allg, np.linspace(0.50, 0.999, 60)):
        out = []
        for frame in (det, fal):
            n_out = (frame[cols] > t).sum(axis=1)
            g = pd.DataFrame({"species": frame["species"], "flag": n_out >= min_sets})
            out.append(g.groupby("species")["flag"].agg(["sum", "size"]))
        D, F = out
        for rule, fn in RULES.items():
            d_n = int(sum(fn(r["sum"], r["size"]) for _, r in D.iterrows()))
            f_n = int(sum(fn(r["sum"], r["size"]) for _, r in F.iterrows()))
            rows.append({"threshold": round(float(t), 4), "rule": rule,
                         "caught": d_n, "of_unseen": len(D),
                         "false_alarms": f_n, "of_described": len(F),
                         "detection": round(d_n / max(1, len(D)), 3),
                         "false_alarm": round(f_n / max(1, len(F)), 3),
                         "margin": round(d_n / max(1, len(D)) - f_n / max(1, len(F)), 3)})
    return pd.DataFrame(rows)


def at_fa(sw: pd.DataFrame, target: float):
    """Best detection among operating points at or below a given false-alarm rate."""
    ok = sw[sw["false_alarm"] <= target + 1e-9]
    if not len(ok):
        return None
    return ok.sort_values(["detection", "false_alarm"], ascending=[False, True]).iloc[0]


def main():
    ap = argparse.ArgumentParser(description="Three instruments, one set of conditions")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--key_sweep", required=True, help="calibration/novelty_tradeoff.tsv")
    ap.add_argument("--graph_sweep", default=None, help="graph_test_loo/graph_tradeoff.tsv")
    ap.add_argument("--fuzzy", default=None, help="key_fuzzy_loo/fuzzy_key_specimens.tsv")
    ap.add_argument("--min_sets", type=int, default=2)
    ap.add_argument("--targets", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    ref = Reference(Path(a.matrix_dir), profile)

    print("matrix: scoring both arms (each specimen held out of its own reference) ...")
    det, fal = matrix_scores(ref, a.min_sets)
    det.to_csv(out / "matrix_detection_scores.tsv", sep="\t", index=False)
    fal.to_csv(out / "matrix_false_alarm_scores.tsv", sep="\t", index=False)
    sw_m = matrix_sweep(det, fal, a.min_sets)
    sw_m["instrument"] = "character matrix"

    sw_k = pd.read_csv(a.key_sweep, sep="\t")
    sw_k["instrument"] = "key"
    sweeps = [sw_k, sw_m]
    if a.graph_sweep and Path(a.graph_sweep).exists():
        sw_g = pd.read_csv(a.graph_sweep, sep="\t")
        sw_g["instrument"] = "knowledge graph"
        sweeps.append(sw_g)

    # the fuzzy path score, scored the same way: held-out species vs described specimens,
    # both graded on keys rebuilt without the specimen concerned
    if a.fuzzy and Path(a.fuzzy).exists():
        fz = pd.read_csv(a.fuzzy, sep="\t")
        fz = fz[fz["resolved"] == True]                                        # noqa: E712
        rows = []
        if len(fz) and "score" in fz:
            for t in np.quantile(fz["score"].dropna(), np.linspace(0.01, 0.99, 60)):
                g = (fz.assign(flag=fz["score"] < t)
                     .groupby(["arm", "species"])["flag"].agg(["sum", "size"]).reset_index())
                D = g[g.arm == "held_out"]
                F = g[g.arm == "known"]
                for rule, fn in RULES.items():
                    d_n = int(sum(fn(r["sum"], r["size"]) for _, r in D.iterrows()))
                    f_n = int(sum(fn(r["sum"], r["size"]) for _, r in F.iterrows()))
                    rows.append({"threshold": round(float(t), 4), "rule": rule,
                                 "caught": d_n, "of_unseen": len(D),
                                 "false_alarms": f_n, "of_described": len(F),
                                 "detection": round(d_n / max(1, len(D)), 3),
                                 "false_alarm": round(f_n / max(1, len(F)), 3),
                                 "margin": round(d_n / max(1, len(D)) - f_n / max(1, len(F)), 3),
                                 "instrument": "fuzzy key score"})
        if rows:
            sweeps.append(pd.DataFrame(rows))

    allsw = pd.concat(sweeps, ignore_index=True, sort=False)
    allsw.to_csv(out / "instrument_sweeps.tsv", sep="\t", index=False)

    # ── the head-to-head table ───────────────────────────────────────────────
    table = []
    for inst, g in allsw.groupby("instrument"):
        row = {"instrument": inst, "best_margin": float(g["margin"].max())}
        b = g.sort_values(["margin", "detection"], ascending=False).iloc[0]
        row["best_margin_rule"] = b["rule"]
        row["best_margin_point"] = f"{int(b['caught'])}/{int(b['of_unseen'])} caught, " \
                                   f"{int(b['false_alarms'])}/{int(b['of_described'])} false"
        for t in a.targets:
            p = at_fa(g, t)
            row[f"detection_at_{int(100 * t)}pct_false_alarms"] = (
                f"{int(p['caught'])}/{int(p['of_unseen'])}" if p is not None else "—")
            row[f"rule_at_{int(100 * t)}pct"] = p["rule"] if p is not None else "—"
        table.append(row)
    tab = pd.DataFrame(table).sort_values("best_margin", ascending=False)
    tab.to_csv(out / "instrument_comparison.tsv", sep="\t", index=False)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.plot([0, 100], [0, 100], color="#bdbdbd", lw=1.0, ls="--", zorder=1)
    ax.text(70, 74, "no information", fontsize=7, color="#9e9e9e", rotation=38)
    colours = {"key": "#1565c0", "character matrix": "#2e7d32",
               "knowledge graph": "#ef6c00", "fuzzy key score": "#6a1b9a"}
    for inst, g in allsw.groupby("instrument"):
        # upper envelope: best detection achievable at each false-alarm rate
        e = (g.sort_values("detection", ascending=False)
             .drop_duplicates("false_alarm").sort_values("false_alarm"))
        e = e[e["detection"].cummax() == e["detection"]]
        ax.plot(100 * e["false_alarm"], 100 * e["detection"], marker="o", ms=3.4, lw=1.6,
                color=colours.get(inst, "#555"), label=inst)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("described species wrongly called new (%)", fontsize=9)
    ax.set_ylabel("species the instrument has never seen, recognised (%)", fontsize=9)
    ax.set_title("Three instruments, both arms held out\n"
                 "best achievable detection at each false-alarm rate", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.axis("off")
    ax.text(0, 1.0, "Read off at a matched false-alarm rate", fontsize=10, weight="bold",
            color="#263238", transform=ax.transAxes)
    cols = ["instrument"] + [f"detection_at_{int(100 * t)}pct_false_alarms" for t in a.targets] \
           + ["best_margin"]
    head = ["instrument"] + [f"at {int(100 * t)}% false alarms" for t in a.targets] + ["best margin"]
    y = 0.90
    xs = [0.0, 0.40, 0.57, 0.74, 0.90]
    for x, h in zip(xs, head):
        ax.text(x, y, h, fontsize=7.6, weight="bold", color="#455a64", transform=ax.transAxes)
    y -= 0.075
    for _, r in tab.iterrows():
        for x, c in zip(xs, cols):
            v = r[c]
            ax.text(x, y, f"{v:.2f}" if isinstance(v, float) else str(v), fontsize=8.0,
                    weight="bold" if c == "instrument" else "normal",
                    color=colours.get(r["instrument"], "#263238") if c == "instrument" else "#263238",
                    transform=ax.transAxes)
        y -= 0.068
        ax.text(0.02, y, f"rule: {r.get('rule_at_10pct', '')}", fontsize=6.6, color="#78909c",
                transform=ax.transAxes)
        y -= 0.062
    y -= 0.04
    for line in ("Both arms are hold-outs: a specimen is never scored against a range,",
                 "an asserted interval or a spread that it helped compute. Measured any",
                 "other way an instrument flatters itself, and the three cannot be compared.",
                 "",
                 "A flag is a reason to examine the specimens. It is never a determination."):
        ax.text(0, y, line, fontsize=7.4, color="#37474f", style="italic" if not line else "normal",
                transform=ax.transAxes)
        y -= 0.058
    fig.suptitle("Recognising a species none of the instruments has seen", fontsize=12,
                 x=0.02, ha="left", color="#1a237e")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_instrument_comparison.png", dpi=200)
    fig.savefig(out / "fig_instrument_comparison.pdf")
    plt.close(fig)

    (out / "instrument_comparison.json").write_text(json.dumps(
        {"version": VERSION, "min_sets": a.min_sets,
         "note": "both arms held out; see module docstring",
         "table": table}, indent=2))
    print("\n" + tab.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
