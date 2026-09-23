#!/usr/bin/env python3
"""
biorag_autapomorphy_v1.py — states fixed within a species and absent outside it
===============================================================================

An autapomorphy, operationally: a character state carried by every specimen of one
species and by no specimen of any other. It is the one kind of evidence that does
not decay with small samples. A continuous character read against the range of six
specimens misclassifies a seventh more than half the time by construction; a state
present in six of six and none of the other hundred-odd is significant at the same
sample size, which is why descriptive taxonomy has always rested on qualitative
diagnostic characters rather than on measurements.

Two things are computed, and they must not be confused:

  the catalogue   which states are fixed within a species and absent outside it,
                  with an exact test and a false-discovery-rate correction over
                  every character x state x species combination examined. Proposed
                  on part of each series and CONFIRMED on the part held back, so a
                  state that merely happens to fit the specimens that suggested it
                  is not reported as diagnostic.

  the instrument  whether discrete states recognise a species the reference has
                  never seen, scored in the same two arms as everything else:
                  detection with the whole species withheld, false alarms with one
                  specimen withheld and its species left in place.

Singletons are handled separately and labelled as such throughout. Roughly half of
the new species described from the tropics are known from one specimen, so they
cannot be excluded; but a single specimen cannot show that a state is FIXED within
its species, only that the state is unique to it among everything else measured.
That is a weaker claim, it is reported under its own heading, and it is never
pooled with the species that support the stronger one.

  python biorag_autapomorphy_v1.py --states "$M/mask_characters/mask_character_states.tsv" \\
      --out_dir "$M/autapomorphies" [--min_fixed 1.0] [--fdr 0.05]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact
except ImportError:                                                   # pragma: no cover
    fisher_exact = None

VERSION = "1.0"
RULES = {"any specimen": lambda f, n: f > 0,
         "a majority of the series": lambda f, n: f > n / 2,
         "the whole series": lambda f, n: f == n}


def bh(p: np.ndarray, n_tests: int = None) -> np.ndarray:
    """Benjamini-Hochberg over every test ATTEMPTED, not every test that survived.

    The survivors were chosen using the same data that produced their p-values — a state
    reaches this function only because it was already seen to be fixed within one species
    and absent from all others. Correcting over the survivors alone would treat a handful
    of tests as the whole experiment and call noise significant; the denominator has to be
    the tens of thousands of character x species x state combinations examined.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if not n:
        return p
    m = max(int(n_tests or n), n)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def wide(states: pd.DataFrame) -> pd.DataFrame:
    return states.pivot_table(index="specimen_id", columns="character", values="state",
                              aggfunc="first")


def catalogue(states: pd.DataFrame, species_of: dict, min_fixed: float, fdr: float,
              propose_frac: float, seed: int, min_series: int = 4):
    """Propose on part of each series, confirm on the rest."""
    rng = np.random.default_rng(seed)
    per_sp = {}
    for sp in sorted(set(species_of.values())):
        ids = sorted([i for i, s in species_of.items() if s == sp])
        if len(ids) == 1:
            per_sp[sp] = (ids, [])                       # singleton: nothing to hold back
        else:
            k = max(1, int(round(propose_frac * len(ids))))
            k = min(k, len(ids) - 1)                     # always keep at least one back
            pick = list(rng.choice(ids, size=k, replace=False))
            per_sp[sp] = (pick, [i for i in ids if i not in pick])

    W = wide(states)
    rows, n_tests = [], 0
    for ch in W.columns:
        col = W[ch].dropna()
        if col.nunique() < 2:
            continue
        for sp, (prop, held) in per_sp.items():
            mem_all = [i for i in col.index if species_of.get(i) == sp]
            if not mem_all:
                continue
            others = col.drop(index=mem_all, errors="ignore")
            prop_in = [i for i in prop if i in col.index]
            if not prop_in:
                continue
            for st in col.loc[prop_in].unique():
                n_tests += 1
                k_prop = int((col.loc[prop_in] == st).sum())
                if k_prop / len(prop_in) < min_fixed:     # not fixed in the proposing half
                    continue
                k_out = int((others == st).sum())
                if k_out:                                 # present outside: not diagnostic
                    continue
                held_in = [i for i in held if i in col.index]
                k_held = int((col.loc[held_in] == st).sum()) if held_in else 0
                a_, b_ = len(mem_all), int(len(others))
                k_all = int((col.loc[[i for i in mem_all if i in col.index]] == st).sum())
                if fisher_exact is not None:
                    _o, p = fisher_exact([[k_all, a_ - k_all],
                                          [0, b_]], alternative="greater")
                else:                                     # hypergeometric tail by hand
                    p = 1.0 / max(1, int(np.prod([(a_ + b_ - i) / (k_all - i)
                                                  for i in range(min(k_all, 5))])))
                rows.append({"character": ch, "state": st, "species": sp,
                             "n_species": a_, "n_other": b_,
                             "proposed_on": len(prop_in), "fixed_in_proposal": k_prop,
                             "held_back": len(held_in), "confirmed_in_held_back": k_held,
                             "confirmed": (len(held_in) > 0 and k_held == len(held_in)),
                             "singleton": len(mem_all) == 1,
                             "fixed_in_whole_series": k_all == a_,
                             "p": float(p)})
    d = pd.DataFrame(rows)
    if len(d):
        d["q"] = bh(d["p"].values, n_tests)
        d["tests_attempted"] = n_tests
        d["verdict"] = np.where(
            d["singleton"], "singleton candidate — unique, but invariance untestable",
            np.where(d["confirmed"] & d["fixed_in_whole_series"] & (d["q"] <= fdr)
                     & (d["n_species"] >= min_series),
                     "autapomorphy — fixed within, absent outside, confirmed on held-back specimens",
                     np.where(d["confirmed"] & (d["n_species"] < min_series),
                              f"too few specimens — a series of under {min_series} cannot show a "
                              f"state is fixed",
                              np.where(d["confirmed"],
                                       "supported but not significant after correction",
                                       "proposed only — failed on the specimens held back"))))
        d = d.sort_values(["singleton", "q", "n_species"], ascending=[True, True, False])
    return d, per_sp


def informativeness(states: pd.DataFrame, species_of: dict) -> pd.DataFrame:
    """How much does knowing a character's state tell you about the species?

    No state here is fixed-and-unique, but they are not therefore uninformative, and a
    reader still needs to know which parts of which structure carry signal. Adjusted
    mutual information between state and species, corrected for chance so that a character
    with many states is not rewarded for that alone, plus the largest share any single
    species holds of any one state — the nearest approach to diagnostic this set offers.
    """
    from sklearn.metrics import adjusted_mutual_info_score
    W = wide(states)
    rows = []
    for ch in W.columns:
        col = W[ch].dropna()
        if col.nunique() < 2 or len(col) < 20:
            continue
        sp = [species_of.get(i) for i in col.index]
        ami = adjusted_mutual_info_score(sp, col.astype(str).tolist())
        best, bsp, bst = 0.0, None, None
        for st, g in pd.DataFrame({"sp": sp, "st": col.astype(str).values}).groupby("st"):
            top = g["sp"].value_counts()
            share = top.iloc[0] / len(g)
            if share > best:
                best, bsp, bst = float(share), top.index[0], st
        struct, quantity = ch.split(":", 1)
        rows.append({"character": ch, "structure": struct, "quantity": quantity,
                     "n_specimens": int(len(col)), "adjusted_mutual_information": round(ami, 4),
                     "purest_state": bst, "purest_state_species": bsp,
                     "purest_state_share": round(best, 3)})
    return pd.DataFrame(rows).sort_values("adjusted_mutual_information", ascending=False)


def discrete_novelty(states: pd.DataFrame, species_of: dict):
    """Do discrete states recognise a species the reference has never seen?

    A specimen is described by its profile of states. Its distance to a species is
    the share of shared characters on which it differs from that species' modal
    state; its score is the distance to the NEAREST species. Both arms withhold the
    specimen from everything it is compared against, exactly as elsewhere.
    """
    W = wide(states)
    sp_of = {i: species_of.get(i) for i in W.index}
    species = sorted({s for s in sp_of.values() if s})

    def modal(ids, exclude):
        sub = W.loc[[i for i in ids if i != exclude]]
        return sub.mode(axis=0).iloc[0] if len(sub) else None

    def dist(row, prof):
        if prof is None:
            return np.nan
        both = row.notna() & prof.notna()
        n = int(both.sum())
        return float((row[both] != prof[both]).sum()) / n if n >= 10 else np.nan

    members = {s: [i for i in W.index if sp_of.get(i) == s] for s in species}
    out = []
    for sid in W.index:
        own = sp_of.get(sid)
        row = W.loc[sid]
        d_det = [dist(row, modal(members[s], sid)) for s in species if s != own]
        d_fa = [dist(row, modal(members[s], sid)) for s in species]
        out.append({"specimen_id": sid, "species": own,
                    "detection_score": float(np.nanmin(d_det)) if np.any(~np.isnan(d_det)) else np.nan,
                    "false_alarm_score": float(np.nanmin(d_fa)) if np.any(~np.isnan(d_fa)) else np.nan})
    d = pd.DataFrame(out)

    sweep = []
    good = d.dropna(subset=["detection_score", "false_alarm_score"])
    if len(good):
        allv = np.r_[good["detection_score"], good["false_alarm_score"]]
        for t in np.quantile(allv, np.linspace(0.30, 0.999, 60)):
            g = good.assign(a_=good["detection_score"] > t, b_=good["false_alarm_score"] > t)
            gg = g.groupby("species").agg(n=("specimen_id", "size"), a_=("a_", "sum"),
                                          b_=("b_", "sum"))
            for rule, fn in RULES.items():
                det = int(sum(fn(r["a_"], r["n"]) for _, r in gg.iterrows()))
                fa = int(sum(fn(r["b_"], r["n"]) for _, r in gg.iterrows()))
                sweep.append({"threshold": round(float(t), 4), "rule": rule,
                              "caught": det, "of_unseen": len(gg),
                              "false_alarms": fa, "of_described": len(gg),
                              "detection": round(det / max(1, len(gg)), 3),
                              "false_alarm": round(fa / max(1, len(gg)), 3),
                              "margin": round((det - fa) / max(1, len(gg)), 3),
                              "instrument": "discrete mask characters"})
    return d, pd.DataFrame(sweep)


def main():
    ap = argparse.ArgumentParser(description="Fixed-within, absent-outside character states")
    ap.add_argument("--states", required=True, help="mask_character_states.tsv (or any states table)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_fixed", type=float, default=1.0,
                    help="share of the proposing specimens that must carry the state (1.0 = all)")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--propose_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_series", type=int, default=4,
                    help="specimens a species needs before a state can be called FIXED within it. "
                         "With two, one specimen proposes and one confirms, which is a match of "
                         "n=1 against n=1 and is not evidence of invariance")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(a.states, sep="\t")
    species_of = dict(zip(st["specimen_id"], st["species"]))

    cat, per_sp = catalogue(st, species_of, a.min_fixed, a.fdr, a.propose_frac, a.seed,
                            a.min_series)
    cat.to_csv(out / "autapomorphy_candidates.tsv", sep="\t", index=False)

    conf = cat[cat["verdict"].str.startswith("autapomorphy")] if len(cat) else cat
    sing = cat[cat["singleton"]] if len(cat) else cat
    print(f"characters examined: {st['character'].nunique()}; "
          f"candidates proposed: {len(cat)}")

    info = informativeness(st, species_of)
    info.to_csv(out / "character_informativeness.tsv", sep="\t", index=False)
    print(f"most informative character: {info.iloc[0]['character']} "
          f"(AMI {info.iloc[0]['adjusted_mutual_information']})" if len(info) else "")

    dn, sweep = discrete_novelty(st, species_of)
    dn.to_csv(out / "discrete_novelty_scores.tsv", sep="\t", index=False)
    sweep.to_csv(out / "discrete_novelty_tradeoff.tsv", sep="\t", index=False)
    best = (sweep.sort_values(["margin", "detection"], ascending=False).iloc[0]
            if len(sweep) else None)

    n_sing_sp = sum(1 for _, (p, h) in per_sp.items() if not h)
    summary = {
        "version": VERSION,
        "characters_examined": int(st["character"].nunique()),
        "specimens": int(st["specimen_id"].nunique()),
        "species": int(len(per_sp)),
        "singleton_species": int(n_sing_sp),
        "tests_attempted": int(cat["tests_attempted"].iloc[0]) if len(cat) else 0,
        "candidates_proposed": int(len(cat)),
        "top_characters_by_information": (info.head(10)[["character", "adjusted_mutual_information",
                                                          "purest_state_species", "purest_state_share"]]
                                          .to_dict("records") if len(info) else []),
        "min_series_for_a_fixed_claim": a.min_series,
        "confirmed_autapomorphies": int(len(conf)),
        "species_with_a_confirmed_autapomorphy": int(conf["species"].nunique()) if len(conf) else 0,
        "singleton_candidates": int(len(sing)),
        "species_with_singleton_candidates": int(sing["species"].nunique()) if len(sing) else 0,
        "proposal_protocol": (f"proposed on {a.propose_frac:.0%} of each series, confirmed on the "
                              f"specimens held back; singletons cannot be held out and are "
                              f"reported separately as candidates, never as autapomorphies"),
        "discrete_novelty_best_operating_point": (
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in best.to_dict().items()} if best is not None else None),
    }
    (out / "autapomorphy_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    if len(conf):
        print("\nconfirmed, by species:")
        for sp, g in conf.groupby("species"):
            print(f"  {sp:14s} {len(g):3d}  e.g. {g.iloc[0]['character']} = {g.iloc[0]['state']}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
