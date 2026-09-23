#!/usr/bin/env python3
"""
biorag_reliability_figures_v1.py — the descriptive-character investigation, as
tables and figures
=============================================================================

Turns the outputs of `biorag_descriptive_scoring_v1.py`,
`biorag_character_reliability_v1.py` and the `biorag_novelty_score_v1.py` runs
into the tables and figures that show HOW the conclusion was reached, step by
step, rather than only the conclusion:

  Table 1 / Figure 1  how repeatable is each descriptive character, exactly and
                      by band, against the two acceptance lines
  Table 2 / Figure 2  does each character track a measured counterpart? median
                      over structures and the structure where it agrees best
  Table 3 / Figure 3  the gating ablation: all characters vs gated, full scales
                      vs bands — four runs, one detection rate each
  Table 4 / Figure 4  every character set side by side in the final run
  Figure 5            the investigation as a sequence: what each step changed

Nothing is hard-coded to a taxon: every path is an argument, and the figures
draw whatever characters and sets the files contain.

  python biorag_reliability_figures_v1.py \\
      --reliability   "$M/descriptive_states/character_reliability.tsv" \\
      --ablation_dir  "$M/descriptive_states/gating_ablation" \\
      --final_run     "$M/novelty_gated" \\
      --scoring_report "$M/descriptive_states/scoring_report.json" \\
      --repeatability "$M/descriptive_states/repeatability.json" \\
      --out_dir       "$M/descriptive_states/figures"
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

VERSION = "1.0"

# the four ablation cells, in the order they are argued
ABLATION = [
    ("ungated", "all characters,\nfull scales", "every character scored, every state kept"),
    ("coarse_only", "all characters,\nbanded", "every character scored, states merged into bands"),
    ("gate_and_coarse", "gated,\nbanded", "characters that did not repeat dropped, states banded"),
    ("gate_only", "gated,\nfull scales", "characters that did not repeat dropped, states kept"),
]
SET_LABEL = {"size": "size (mm)", "ratio": "proportions", "landmark": "landmark shape",
             "colour": "colour (CIE)", "descriptive": "descriptive characters",
             "computed": "measured stand-ins"}
GREY, INK, HL, WARN = "#9aa0a6", "#202124", "#1a73e8", "#c5221f"


def style(ax, title="", xlabel="", ylabel="", pad=10):
    ax.set_title(title, fontsize=11, loc="left", color=INK, pad=pad)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(labelsize=8, colors=INK, length=3, color=GREY)
    ax.grid(axis="x", color="#e8eaed", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str, written: list):
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
        if ext == "png":
            written.append(p)
    plt.close(fig)


# ── 1. repeatability ──────────────────────────────────────────────────────────
def fig_repeatability(rel: pd.DataFrame, out: Path, written: list, min_fine=0.85, min_coarse=0.80):
    d = rel.sort_values("repeat_exact")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(d) + 1.9))
    ax.barh(y + 0.19, d["repeat_exact"], height=0.36, color=HL, zorder=3, label="same state exactly")
    ax.barh(y - 0.19, d["repeat_coarse"].fillna(0), height=0.36, color="#a8c7fa", zorder=3,
            label="same band")
    for i, r in enumerate(d.itertuples()):
        if not (r.repeat_coarse == r.repeat_coarse):
            ax.text(0.01, i - 0.19, "  nominal — no bands", va="center", fontsize=7, color=WARN, zorder=4)
    ax.axvline(min_fine, color=INK, lw=1, ls="--", zorder=4)
    ax.axvline(min_coarse, color=GREY, lw=1, ls=":", zorder=4)
    ax.annotate(f" keep as scored ({min_fine:.0%})", xy=(min_fine, 1.005),
                xycoords=("data", "axes fraction"), fontsize=7.5, color=INK, va="bottom")
    ax.annotate(f"keep banded ({min_coarse:.0%}) ", xy=(min_coarse, 1.005),
                xycoords=("data", "axes fraction"), fontsize=7.5, color=GREY, va="bottom", ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}" + ("  ⚑" if v == "flag" else "")
                        for c, v in zip(d["character"], d["verdict"])], fontsize=8.5)
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(-0.75, len(d) - 0.25)
    style(ax, "Figure 1. How often the same image gets the same word twice",
          "states repeated when the same images were scored a second time", pad=22)
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.02), ncol=2)
    save(fig, out, "fig1_repeatability", written)


# ── 2. congruence ─────────────────────────────────────────────────────────────
def fig_congruence(rel: pd.DataFrame, out: Path, written: list):
    d = rel[rel["computed_counterpart"].notna()].copy()
    if d.empty:
        return
    d = d.sort_values("congruence_best_structure")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.6, 0.44 * len(d) + 2.0))
    for i, r in enumerate(d.itertuples()):
        ax.plot([r.congruence, r.congruence_best_structure], [i, i], color=GREY, lw=1.4, zorder=2)
    ax.scatter(d["congruence"], y, s=34, color=GREY, zorder=3, label="median over structures")
    ax.scatter(d["congruence_best_structure"], y, s=38, color=HL, zorder=3,
               label="structure where it agrees best")
    ax.axvline(0.5, color=INK, lw=1, ls="--", zorder=1)
    ax.annotate(" strong agreement", xy=(0.5, 1.005), xycoords=("data", "axes fraction"),
                fontsize=7.5, color=INK, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.character}\n  vs {r.computed_counterpart}" for r in d.itertuples()],
                       fontsize=8)
    ax.set_xlim(0, max(0.85, float(d["congruence_best_structure"].max()) + 0.08))
    ax.set_ylim(-0.9, len(d) - 0.3)
    style(ax, "Figure 2. Does the word track anything the pipeline measures?",
          "|Spearman ρ| (ordinal characters) or eta² (nominal)", pad=22)
    ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.02), ncol=2)
    save(fig, out, "fig2_congruence", written)


# ── 3. the gating ablation ────────────────────────────────────────────────────
def read_ablation(ab: Path) -> pd.DataFrame:
    rows = []
    for key, label, note in ABLATION:
        c, g = ab / key / "calibration.json", ab / key / "calibration_series.json"
        if not c.exists():
            continue
        cal = json.loads(c.read_text()).get("descriptive", {})
        gcal = json.loads(g.read_text()).get("descriptive", {}) if g.exists() else {}
        rows.append({"run": key, "label": label.replace("\n", " "), "what_changed": note,
                     "characters_used": cal.get("n_characters"),
                     "threshold_specimen": round(cal.get("threshold", float("nan")), 3),
                     "false_positive_rate": cal.get("observed_fpr"),
                     "unseen_species_caught_specimen": cal.get("tpr"),
                     "unseen_species_caught_series": gcal.get("tpr")})
    return pd.DataFrame(rows)


def fig_ablation(ab_df: pd.DataFrame, out: Path, written: list):
    if ab_df.empty:
        return
    order = [k for k, _, _ in ABLATION if k in set(ab_df["run"])]
    d = ab_df.set_index("run").loc[order].reset_index()
    labels = [dict((k, l) for k, l, _ in ABLATION)[k] for k in d["run"]]
    x = np.arange(len(d))
    best = int(np.nanargmax(d["unseen_species_caught_specimen"].values))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colours = [HL if i == best else GREY for i in range(len(d))]
    ax.bar(x, d["unseen_species_caught_specimen"], width=0.58, color=colours, zorder=3)
    for i, v in enumerate(d["unseen_species_caught_specimen"]):
        if v == v:
            ax.text(i, v + 0.008, f"{v:.1%}", ha="center", fontsize=9,
                    color=HL if i == best else INK, fontweight="bold" if i == best else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", color="#e8eaed", lw=0.6, zorder=0)
    ax.grid(axis="x", visible=False)
    style(ax, "Figure 3. What dropping characters, and what merging states, does",
          "", "unseen species caught (per specimen)")
    ax.set_axisbelow(True)
    fig.text(0.02, -0.03,
             "All four runs are calibrated to the same rate of calling a known specimen novel, so the bars "
             "are comparable.\nDropping the characters that do not repeat helps; merging states into bands "
             "costs roughly half of whatever it is applied to.",
             fontsize=8, color=INK)
    save(fig, out, "fig3_gating_ablation", written)


# ── 4. character sets side by side ────────────────────────────────────────────
def read_sets(run: Path) -> pd.DataFrame:
    cal = json.loads((run / "calibration.json").read_text())
    gp = run / "calibration_series.json"
    gcal = json.loads(gp.read_text()) if gp.exists() else {}
    rows = []
    for k, c in cal.items():
        g = gcal.get(k, {})
        rows.append({"character_set": SET_LABEL.get(k, k), "key": k,
                     "statistic": c.get("stat"),
                     "threshold_specimen": round(c.get("threshold", float("nan")), 3),
                     "threshold_series": round(g.get("threshold", float("nan")), 3)
                     if g else float("nan"),
                     "false_positive_rate": c.get("observed_fpr"),
                     "caught_per_specimen": c.get("tpr"), "caught_per_series": g.get("tpr"),
                     "n_known": c.get("n_conspecific"), "n_unseen": c.get("n_novel")})
    return pd.DataFrame(rows).sort_values("caught_per_specimen", ascending=False)


def fig_sets(d: pd.DataFrame, out: Path, written: list):
    if d.empty:
        return
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 0.52 * len(d) + 2.0))
    colours = [HL if k == "descriptive" else GREY for k in d["key"]]
    ax.barh(y + 0.18, d["caught_per_specimen"], height=0.34, color=colours, zorder=3,
            label="per specimen")
    ax.barh(y - 0.18, d["caught_per_series"].fillna(0), height=0.34,
            color=["#a8c7fa" if k == "descriptive" else "#dadce0" for k in d["key"]], zorder=3,
            label="per series")
    for i, r in zip(y, d.itertuples()):
        if r.caught_per_specimen == r.caught_per_specimen:
            ax.text(r.caught_per_specimen + 0.006, i + 0.18, f"{r.caught_per_specimen:.0%}",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(d["character_set"], fontsize=9)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "Figure 4. Each character set on its own, same false-positive rate",
          "species caught when that species is removed from the reference set")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    save(fig, out, "fig4_character_sets", written)


# ── 5. the investigation as a sequence ────────────────────────────────────────
def fig_story(ab_df: pd.DataFrame, rel: pd.DataFrame, rep: dict, out: Path, written: list):
    steps = [["scored once\nper species",
              "no within-species replicate:\nno error rate can be estimated", np.nan,
              "not\nmeasurable"],
             ["scored per specimen,\nall characters",
              f"{len(rel)} characters on every\nspecimen × structure image", np.nan, ""],
             ["repeatability\nmeasured",
              (f"{100 * rep.get('exact_agreement', 0):.0f}% of states repeat;\n"
               f"{int((rel['verdict'] == 'flag').sum())} characters do not"), np.nan,
              "diagnosis,\nnot a run"],
             ["characters that do not\nrepeat dropped", "the rest kept at full scale", np.nan, ""]]
    look = {r["run"]: r for _, r in ab_df.iterrows()} if not ab_df.empty else {}
    if "ungated" in look:
        steps[1][2] = look["ungated"]["unseen_species_caught_specimen"]
    if "gate_only" in look:
        steps[3][2] = look["gate_only"]["unseen_species_caught_specimen"]
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    x = np.arange(len(steps))
    vals = [s[2] for s in steps]
    ok = [i for i, v in enumerate(vals) if v == v]
    ax.plot([x[i] for i in ok], [vals[i] for i in ok], color=HL, lw=2, zorder=3, marker="o", ms=8)
    for i, (title, note, v, absent) in enumerate(steps):
        ax.text(i, -0.075, title, ha="center", fontsize=9, color=INK, fontweight="bold")
        ax.text(i, -0.135, note, ha="center", fontsize=7.6, color=GREY)
        if v == v:
            ax.text(i, v + 0.022, f"{v:.1%}", ha="center", fontsize=10, color=HL, fontweight="bold")
        elif absent:
            ax.text(i, 0.02, absent, ha="center", fontsize=8,
                    color=WARN if "measurable" in absent else GREY)
    ax.set_xlim(-0.55, len(steps) - 0.45)
    ax.set_ylim(-0.2, max([v for v in vals if v == v] + [0.4]) + 0.09)
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v >= 0 else "")
    ax.spines["bottom"].set_visible(False)
    style(ax, "Figure 5. What each step of the investigation changed", "",
          "unseen species caught (per specimen)")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#e8eaed", lw=0.6)
    save(fig, out, "fig5_investigation_steps", written)


def main():
    ap = argparse.ArgumentParser(description="Tables and figures for the descriptive-character investigation")
    ap.add_argument("--reliability", required=True, help="character_reliability.tsv")
    ap.add_argument("--ablation_dir", default=None, help="directory of the gating ablation runs")
    ap.add_argument("--final_run", default=None, help="the novelty run to report set by set")
    ap.add_argument("--scoring_report", default=None)
    ap.add_argument("--repeatability", default=None)
    ap.add_argument("--min_fine", type=float, default=0.85)
    ap.add_argument("--min_coarse", type=float, default=0.80)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list = []

    rel = pd.read_csv(a.reliability, sep="\t")
    rep = json.loads(Path(a.repeatability).read_text()) if a.repeatability and \
        Path(a.repeatability).exists() else {}

    # Table 1 — repeatability and the verdict
    t1 = rel[["character", "type", "states", "coarse_bands", "n_compared", "repeat_exact",
              "repeat_within_one_step", "repeat_coarse", "verdict", "reason"]]
    t1.to_csv(out / "table1_character_repeatability.tsv", sep="\t", index=False)
    fig_repeatability(rel, out, written, a.min_fine, a.min_coarse)

    # Table 2 — congruence with the measured counterparts
    cols = [c for c in ["character", "computed_counterpart", "congruence_statistic", "congruence",
                        "congruence_best_structure", "structures_congruent", "structures_compared"]
            if c in rel.columns]
    rel[cols].to_csv(out / "table2_congruence.tsv", sep="\t", index=False)
    if "congruence_best_structure" in rel.columns:
        fig_congruence(rel, out, written)

    # Table 3 — the gating ablation
    ab = pd.DataFrame()
    if a.ablation_dir and Path(a.ablation_dir).exists():
        ab = read_ablation(Path(a.ablation_dir))
        n_all, n_kept = len(rel), int((rel["verdict"] != "flag").sum())
        ab["characters_used"] = [n_kept if r.startswith("gate") else n_all for r in ab["run"]]
        ab.to_csv(out / "table3_gating_ablation.tsv", sep="\t", index=False)
        fig_ablation(ab, out, written)

    # Table 4 — the character sets in the final run
    if a.final_run and (Path(a.final_run) / "calibration.json").exists():
        sets = read_sets(Path(a.final_run))
        sets.drop(columns=["key"]).to_csv(out / "table4_character_sets.tsv", sep="\t", index=False)
        fig_sets(sets, out, written)

    # Figure 5 — the sequence
    if not ab.empty:
        fig_story(ab, rel, rep, out, written)

    (out / "figures_index.json").write_text(json.dumps({
        "version": VERSION, "inputs": {k: str(v) for k, v in vars(a).items()},
        "files": sorted(p.name for p in out.iterdir())}, indent=2))
    print(f"-> {out}")
    for p in sorted(out.glob("table*.tsv")):
        print(f"   {p.name}")
    for p in written:
        print(f"   {p.name} (+ .pdf)")


if __name__ == "__main__":
    main()
