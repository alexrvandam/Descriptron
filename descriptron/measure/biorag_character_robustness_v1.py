#!/usr/bin/env python3
"""
biorag_character_robustness_v1.py — what is ONE character worth, measured the way the instruments are
======================================================================================================

The key, the knowledge graph and the character matrix are judged by withholding a specimen (can it
be named?) or a whole species (is it recognised as new?). The same two hold-outs answer the question
for a single character, and nothing else is needed:

  naming      each specimen is withheld in turn; every species' mean for the character is recomputed
              without it; the species are ranked by how close their mean lies to the specimen's value.
              top1 = share of specimens whose own species is nearest, top3 = among the three nearest,
              rank = mean position of the own species scaled 0 (always first) to 1 (always last).
              chance_top1 is 1 / (species that have the character), averaged over the specimens tested.
  novelty     the matrix flags a specimen by its distance to the NEAREST species. With the specimen's
              species described that is min(d_own, d_other) — d_own to its own species' mean (computed
              without it), d_other to the nearest other species' mean; had its species never been
              described it would be d_other. novelty_auc is the probability that the second exceeds
              the first for two specimens drawn at random (0.5 = on this character alone a new
              species looks like a described one, which with many species in one range is the rule:
              novelty is a property of characters in combination, not of any one of them).
  support     the spread of those per-specimen results over species: a character that names every
              specimen of three species and none of the rest is not the same as one that names half
              of each. species_named = species with at least half of their specimens named.

A character is tested only on species with at least two specimens that have it (the specimen under
test needs a conspecific left behind). Every statistic is a hold-out: the specimen is never part of
the mean it is compared with.

  python biorag_character_robustness_v1.py --matrix_dir <compiled_key_tier> --out_dir <dir> \
      [--tiers key description] [--min_species 5] [--top 40]

Reads `specimen_matrix_long.csv` (species, specimen_id, feature_id, base_category, family, tier,
value). Writes character_robustness.tsv, structure_robustness.tsv, family_robustness.tsv,
character_robustness_summary.json and fig_character_robustness.png/.pdf. No model is called.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.0"


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(pos == neg), by ranks."""
    if not len(pos) or not len(neg):
        return math.nan
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def one_character(g: pd.DataFrame):
    """g: rows of one feature (species, specimen_id, value). Returns the per-specimen hold-out table."""
    g = g.dropna(subset=["value"])
    n_by = g.groupby("species")["value"].agg(["sum", "count"])
    species = n_by.index.to_numpy()
    if len(species) < 2:
        return pd.DataFrame()
    s_sum, s_cnt = n_by["sum"].to_numpy(float), n_by["count"].to_numpy(float)
    idx = {s: i for i, s in enumerate(species)}
    rows = []
    for sp, sid, x in zip(g["species"], g["specimen_id"], g["value"]):
        i = idx[sp]
        if s_cnt[i] < 2:
            continue                                   # no conspecific would be left behind
        means = s_sum / s_cnt
        means = means.copy()
        means[i] = (s_sum[i] - x) / (s_cnt[i] - 1)     # own species' mean WITHOUT this specimen
        d = np.abs(means - x)
        d_own = d[i]
        others = np.delete(d, i)
        rank = 1 + np.sum(others < d_own) + 0.5 * np.sum(others == d_own)     # 1 = nearest
        rows.append({"species": sp, "specimen_id": sid, "rank": rank, "n_species": len(species),
                     "d_own": d_own, "d_other": float(others.min())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Hold-out worth of each character of the matrix, one at a time")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tiers", nargs="*", default=None, help="tiers to include (default: all in the matrix)")
    ap.add_argument("--min_species", type=int, default=5, help="report a character only when tested on this many species")
    ap.add_argument("--top", type=int, default=40, help="characters drawn in the figure")
    ap.add_argument("--no_figure", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(Path(a.matrix_dir) / "specimen_matrix_long.csv", low_memory=False)
    m["value"] = pd.to_numeric(m["value"], errors="coerce")
    if a.tiers:
        m = m[m["tier"].isin(a.tiers)]
    labels = {}
    fd = Path(a.matrix_dir) / "feature_dictionary.tsv"
    if fd.exists():
        d = pd.read_csv(fd, sep="\t")
        lab_col = next((c for c in ("label", "description", "name") if c in d.columns), None)
        if lab_col and "feature_id" in d.columns:
            labels = dict(zip(d["feature_id"], d[lab_col].astype(str)))
    # one value per specimen x character (a specimen photographed twice contributes its mean)
    m = (m.groupby(["species", "specimen_id", "feature_id", "base_category", "family", "tier"], as_index=False)["value"].mean())

    rows, per_specimen = [], []
    for fid, g in m.groupby("feature_id"):
        t = one_character(g)
        if not len(t) or t["species"].nunique() < a.min_species:
            continue
        n = len(t)
        k1 = int((t["rank"] <= 1).sum())
        k3 = int((t["rank"] <= 3).sum())
        lo, hi = wilson(k1, n)
        by_sp = t.assign(hit=t["rank"] <= 1).groupby("species")["hit"].mean()
        meta = g.iloc[0]
        rows.append({"feature_id": fid, "label": labels.get(fid, fid), "structure": meta["base_category"],
                     "family": meta["family"], "tier": meta["tier"],
                     "specimens_tested": n, "species_tested": int(t["species"].nunique()),
                     "species_with_character": int(t["n_species"].iloc[0]),
                     "top1": round(k1 / n, 4), "top1_ci_low": round(lo, 4), "top1_ci_high": round(hi, 4),
                     "chance_top1": round(float((1.0 / t["n_species"]).mean()), 4),
                     "top3": round(k3 / n, 4),
                     "rank_scaled": round(float(((t["rank"] - 1) / (t["n_species"] - 1)).mean()), 4),
                     "novelty_auc": round(auc(t["d_other"].to_numpy(),
                                              np.minimum(t["d_own"].to_numpy(), t["d_other"].to_numpy())), 4),
                     "species_named": int((by_sp >= 0.5).sum()),
                     "species_never_named": int((by_sp == 0).sum())})
        per_specimen.append(t.assign(feature_id=fid))
    res = pd.DataFrame(rows)
    if not len(res):
        raise SystemExit("no character could be tested (every species needs two specimens with the character)")
    res["times_chance"] = (res["top1"] / res["chance_top1"]).round(2)
    res = res.sort_values(["top1", "novelty_auc"], ascending=False)
    res.to_csv(out / "character_robustness.tsv", sep="\t", index=False)

    def roll(col):
        return (res.groupby(col).agg(characters=("feature_id", "size"), median_top1=("top1", "median"),
                                     best_top1=("top1", "max"), median_top3=("top3", "median"),
                                     median_novelty_auc=("novelty_auc", "median"),
                                     median_times_chance=("times_chance", "median"),
                                     best_character=("label", "first"))
                .round(4).sort_values("median_top1", ascending=False).reset_index())
    roll("structure").to_csv(out / "structure_robustness.tsv", sep="\t", index=False)
    roll("family").to_csv(out / "family_robustness.tsv", sep="\t", index=False)
    roll("tier").to_csv(out / "tier_robustness.tsv", sep="\t", index=False)

    summary = {"version": VERSION, "characters_tested": int(len(res)),
               "median_top1": float(res["top1"].median()), "median_chance_top1": float(res["chance_top1"].median()),
               "characters_at_least_twice_chance": int((res["times_chance"] >= 2).sum()),
               "characters_interval_above_chance": int((res["top1_ci_low"] > res["chance_top1"]).sum()),
               "characters_novelty_auc_above_0.7": int((res["novelty_auc"] >= 0.7).sum()),
               "best_single_character": res.iloc[0][["label", "top1", "top3", "novelty_auc", "specimens_tested"]].to_dict(),
               "by_tier": roll("tier").to_dict("records"),
               "note": "every value is a hold-out: the specimen is never part of the species mean it is ranked "
                       "against; a character is tested only on species with two or more specimens that have it"}
    (out / "character_robustness_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_tier", "note")}, indent=1, default=float))
    print("\nby structure:\n" + roll("structure").head(30).to_string(index=False))
    print("\nby family:\n" + roll("family").to_string(index=False))

    if not a.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = sorted(res["family"].astype(str).unique())
        pal = dict(zip(fams, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#000000", "#56B4E9", "#F0E442",
                              "#999999", "#7B3294"][:len(fams)] + ["#555555"] * max(0, len(fams) - 10)))
        # one row per structure: its best character (bar, 95% interval) and the median of all its
        # characters (diamond). Forty near-identical landmark distances would say nothing a reader can use.
        best = res.sort_values("top1", ascending=False).groupby("structure", as_index=False).first()
        med = res.groupby("structure")["top1"].median()
        best["median_top1"] = best["structure"].map(med)
        top = best.sort_values("top1").tail(a.top)
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, max(6, 0.3 * len(top) + 2)), gridspec_kw={"width_ratios": [1.5, 1]})
        y = np.arange(len(top))
        ax.barh(y, 100 * top["top1"], color=[pal[str(f)] for f in top["family"]], height=0.7)
        ax.errorbar(100 * top["top1"], y, xerr=[100 * (top["top1"] - top["top1_ci_low"]), 100 * (top["top1_ci_high"] - top["top1"])],
                    fmt="none", ecolor="#555555", elinewidth=0.8, capsize=1.5)
        ax.scatter(100 * top["median_top1"], y, marker="D", s=26, facecolor="white", edgecolor="#212121", zorder=6,
                   label="median of the structure's characters")
        ax.scatter(100 * top["chance_top1"], y, marker="|", color="#b71c1c", s=90, zorder=5, label="chance")
        for i, (n_, k_) in enumerate(zip(top["specimens_tested"], top["species_tested"])):
            ax.text(100 * top["top1_ci_high"].iloc[i] + 1, i, f"{n_} specimens, {k_} species", va="center", fontsize=5.8, color="#546e7a")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{s_}: {l}"[:64] for s_, l in zip(top["structure"], top["label"])], fontsize=6.8)
        ax.set_xlabel("withheld specimens whose own species is the nearest on this character alone (%)", fontsize=8)
        ax.set_title("a  The best single character of each structure, and the structure's median\n"
                     "(each specimen withheld in turn; bars = 95% interval)", loc="left", fontsize=9.5)
        for f in fams:
            ax.barh([-5], [0], color=pal[f], label=f)
        ax.set_ylim(-0.7, len(top) - 0.3)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2)
        for f, g in res.groupby("family"):
            ax2.scatter(100 * g["top1"], g["novelty_auc"], s=14, color=pal[str(f)], alpha=0.75, label=str(f))
        ax2.axhline(0.5, color="#b71c1c", lw=0.8, ls="--")
        ax2.set_xlabel("names a withheld specimen (%)", fontsize=8)
        ax2.set_ylabel("tells a described species from an unseen one (AUC)", fontsize=8)
        ax2.set_title(f"b  All {len(res)} characters: naming against novelty", loc="left", fontsize=9.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
            ax2.spines[s_].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "fig_character_robustness.png", dpi=200)
        fig.savefig(out / "fig_character_robustness.pdf")
        plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
