#!/usr/bin/env python3
"""
descriptron_character_signal_v1.py - which characters separate species, and which carry phylogenetic signal?
===========================================================================================================

For every character of the matrix (compiled_key_tier: specimen_matrix_long.csv + feature_dictionary.tsv):
  species separation   Kruskal-Wallis across species (species with >= 2 specimens), effect size
                       eta^2 = (H - k + 1) / (n - k), Benjamini-Hochberg q over all characters
  phylogenetic signal  with --tree: Blomberg's K of the species means (Adams' Kmult on one variable, as
                       geomorph::physignal), permutation P (--iterations), on the tree pruned to the species measured

Outputs (--out_dir):
  character_signal_all.tsv        every character
  character_signal_top.tsv        per structure, the --top characters with the largest eta^2 (all of them if the
                                  structure has fewer)
  character_signal_top.md         the same as a readable table, one block per structure
  character_phylo_signal.png/.pdf     with --tree and --robustness: the companion to fig_character_robustness
      (same family colours): a, every character's Blomberg's K by structure; b, naming against K
  character_signal_volcano.png/.pdf   only with --figure
      a  volcano plot: eta^2 against -log10 q, coloured by phylogenetic signal, the strongest characters labelled
      b  species separation against phylogenetic signal (eta^2 against K)
  character_signal_summary.json

  python descriptron_character_signal_v1.py --matrix_dir <out>/compiled_key_tier --tree tree.nwk --out_dir <out>/character_signal
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import descriptron_phylo as dp                       # noqa: E402

INK, MUTED, GRID = "#1d2327", "#6b7780", "#d0d7de"
C_SIG, C_NS = "#0072B2", "#9aa5ad"
C_ROBUST, C_WEAK = "#0b57d0", "#7d878f"          # label text: robust (named >= --robust_x times chance) / not


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = ~np.isnan(p)
    if ok.sum():
        pp = p[ok]; o = np.argsort(pp); r = pp[o] * ok.sum() / np.arange(1, ok.sum() + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; qq = np.empty_like(r); qq[o] = np.minimum(r, 1); q[ok] = qq
    return q


def label_column(ax, xs, ys, labels, side_x=1.02, fontsize=5.6, gap_frac=1 / 34, colors=None):
    """labels in a column right of the axes, spaced so they never overlap, each joined to its point by a thin line"""
    if not len(xs):
        return
    lo, hi = ax.get_ylim(); gap = (hi - lo) * gap_frac
    order = np.argsort(ys)[::-1]
    pos, last = [], hi + gap
    for i in order:                                        # top-down, never closer than `gap`
        y = min(ys[i], last - gap); pos.append((i, y)); last = y
    shift = max(0.0, lo - min(y for _, y in pos))          # pushed below the axis: move the column up
    to_axes = ax.transAxes.inverted()
    for i, y in pos:
        px, py = to_axes.transform(ax.transData.transform((xs[i], ys[i])))
        _, ty = to_axes.transform(ax.transData.transform((xs[i], y + shift)))
        # a leader line of (near) zero length breaks matplotlib's arrow drawing: no line when label sits on the point
        line = abs(px - side_x) > 0.01 or abs(py - ty) > 0.01
        ax.annotate(labels[i], (xs[i], ys[i]), xytext=(side_x, y + shift), textcoords=("axes fraction", "data"),
                    fontsize=fontsize, color=colors[i] if colors is not None else INK, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=0, shrinkB=2) if line else None)


def short_label(structure, label):
    """the label alone when it already names its structure (e.g. antenna area), else structure: label"""
    st = str(structure).lower().replace("_", " ")
    first = str(label).split()[0].rstrip(":").lower() if str(label).split() else ""
    return str(label) if first and (first in st or st.startswith(first)) else f"{structure}: {label}"


def blomberg_k_perm(y, C, iters, rng):
    """Blomberg's K (= Kmult for one variable) and its permutation P, vectorised over permutations"""
    n = len(y); Ci = np.linalg.inv(C); s1 = Ci.sum(); expct = (np.trace(C) - n / s1) / (n - 1)
    Y = np.column_stack([y] + [y[rng.permutation(n)] for _ in range(iters)])     # n x (1 + iters)
    a = (Ci.sum(0) @ Y) / s1
    R = Y - a
    obs = (R * R).sum(0) / (R * (Ci @ R)).sum(0)
    K = obs / expct
    return float(K[0]), float((1 + (K[1:] >= K[0] - 1e-12).sum()) / (iters + 1))


def _eta_figure(D, out, tree_txt, a):
    D = D.copy(); D["mlq"] = -np.log10(D.q.clip(lower=1e-300))

    sig = (D.K_P <= 0.05) if "K_P" in D else pd.Series(False, index=D.index)
    fig, axs = plt.subplots(1, 2 if tree_txt else 1, figsize=(13.5 if tree_txt else 7.5, 4.8), squeeze=False,
                            gridspec_kw={"wspace": 1.0})
    ax = axs[0, 0]
    ax.scatter(D.eta2[~sig], D.mlq[~sig], s=9, color=C_NS, alpha=0.7, lw=0, label="no phylogenetic signal (P > 0.05)" if tree_txt else "character")
    if tree_txt:
        ax.scatter(D.eta2[sig], D.mlq[sig], s=12, color=C_SIG, lw=0, label="phylogenetic signal (P ≤ 0.05)")
    ax.axhline(-np.log10(a.fdr), color=MUTED, lw=0.6, ls=":"); ax.text(0.01, -np.log10(a.fdr) + 0.1, f"q = {a.fdr}", fontsize=6.5, color=MUTED)
    top = D.sort_values("eta2", ascending=False).head(a.label_top)
    label_column(ax, top.eta2.values, top.mlq.values, list(top.label))
    ax.set_xlabel("species separation (Kruskal–Wallis η²)", fontsize=8); ax.set_ylabel("−log10 q (FDR)", fontsize=8)
    ax.set_title(f"a  {len(D)} characters: separation and significance", loc="left", fontsize=9, weight="bold", color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left"); ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    if tree_txt:
        ax = axs[0, 1]; E = D.dropna(subset=["K"])
        s2 = E.K_P <= 0.05
        ax.scatter(E.eta2[~s2], E.K[~s2], s=9, color=C_NS, alpha=0.7, lw=0)
        ax.scatter(E.eta2[s2], E.K[s2], s=12, color=C_SIG, lw=0)
        both = E[s2 & (E.q <= a.fdr)].sort_values("eta2", ascending=False).head(a.label_top)
        label_column(ax, both.eta2.values, both.K.values, list(both.label))
        ax.axhline(1, color=MUTED, lw=0.6, ls=":"); ax.text(0.99, 1.02, "K = 1 (Brownian motion)", fontsize=6.5, color=MUTED, ha="right", transform=ax.get_yaxis_transform())
        ax.set_xlabel("species separation (Kruskal–Wallis η²)", fontsize=8); ax.set_ylabel("phylogenetic signal (Blomberg's K)", fontsize=8)
        ax.set_title(f"b  separation and phylogenetic signal ({int(s2.sum())} with P ≤ 0.05)", loc="left", fontsize=9, weight="bold", color=INK)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"character_signal_volcano.{ext}", dpi=300, facecolor="white", bbox_inches="tight")


def _robust_figure(A, out, tree_txt, a):
    """a: hold-out naming against novelty AUC (the character-robustness plot); b: naming against Blomberg's K"""
    D = A.dropna(subset=["top1", "novelty_auc"]).copy()
    sig = (D.K_P <= 0.05) if "K_P" in D else pd.Series(False, index=D.index)
    fig, axs = plt.subplots(1, 2 if tree_txt else 1, figsize=(13.5 if tree_txt else 7.5, 4.8), squeeze=False,
                            gridspec_kw={"wspace": 1.0})
    ax = axs[0, 0]
    ax.scatter(100 * D.top1[~sig], D.novelty_auc[~sig], s=9, color=C_NS, alpha=0.7, lw=0,
               label="no phylogenetic signal (P > 0.05)" if tree_txt else "character")
    if tree_txt:
        ax.scatter(100 * D.top1[sig], D.novelty_auc[sig], s=12, color=C_SIG, lw=0, label="phylogenetic signal (P ≤ 0.05)")
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=":"); ax.text(0.99, 0.505, "AUC 0.5 = chance", fontsize=6.5, color=MUTED,
                                                          ha="right", transform=ax.get_yaxis_transform())
    top = D.assign(sc=D.top1 + D.novelty_auc).sort_values("sc", ascending=False).head(a.label_top)
    label_column(ax, 100 * top.top1.values, top.novelty_auc.values, [short_label(st, lb) for st, lb in zip(top.structure, top.label)])
    ax.set_xlabel("names a withheld specimen (%, this character alone)", fontsize=8)
    ax.set_ylabel("tells a described species from an unseen one (AUC)", fontsize=8)
    ax.set_title(f"a  {len(D)} characters: naming against novelty (hold-outs)", loc="left", fontsize=9, weight="bold", color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left"); ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    if tree_txt:
        ax = axs[0, 1]; E = D.dropna(subset=["K"]); s2 = E.K_P <= 0.05
        ax.scatter(100 * E.top1[~s2], E.K[~s2], s=9, color=C_NS, alpha=0.7, lw=0)
        ax.scatter(100 * E.top1[s2], E.K[s2], s=12, color=C_SIG, lw=0)
        both = E[s2].sort_values("top1", ascending=False).head(a.label_top)
        label_column(ax, 100 * both.top1.values, both.K.values, [short_label(st, lb) for st, lb in zip(both.structure, both.label)])
        ax.axhline(1, color=MUTED, lw=0.6, ls=":"); ax.text(0.99, 1.02, "K = 1 (Brownian motion)", fontsize=6.5, color=MUTED, ha="right", transform=ax.get_yaxis_transform())
        ax.set_xlabel("names a withheld specimen (%, this character alone)", fontsize=8); ax.set_ylabel("phylogenetic signal (Blomberg's K)", fontsize=8)
        ax.set_title(f"b  naming and phylogenetic signal ({int(s2.sum())} with P ≤ 0.05)", loc="left", fontsize=9, weight="bold", color=INK)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"character_signal_volcano.{ext}", dpi=300, facecolor="white", bbox_inches="tight")


def _phylo_figure(A, out, a):
    """companion to the character-robustness figure, same family colours: a, every character's phylogenetic signal per
    structure; b, naming a withheld specimen against Blomberg's K. Filled = P <= 0.05, open = not."""
    D = A.dropna(subset=["K", "top1"]).copy()
    fams = sorted(A["family"].dropna().astype(str).unique())          # same order and palette as fig_character_robustness
    pal = dict(zip(fams, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#000000", "#56B4E9", "#F0E442",
                          "#999999", "#7B3294"][:len(fams)] + ["#555555"] * max(0, len(fams) - 10)))
    sig = D.K_P <= 0.05
    share = D.groupby("structure").apply(lambda g: (g.K_P <= 0.05).mean())
    order = share.sort_values().index.tolist()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, max(6, 0.3 * len(order) + 2)), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.3})
    ypos = {st: i for i, st in enumerate(order)}
    jit = pd.Series(np.random.default_rng(0).uniform(-0.25, 0.25, len(D)), index=D.index)
    for (f, filled), g in D.assign(_s=sig, _j=jit).groupby(["family", "_s"]):
        ys = g.structure.map(ypos) + g._j
        ax.scatter(g.K, ys, s=16 if filled else 12, facecolor=pal[str(f)] if filled else "none", edgecolor=pal[str(f)],
                   lw=0.7, alpha=0.9 if filled else 0.55, zorder=3 if filled else 2)
    for st, i in ypos.items():
        g = D[D.structure == st]
        ax.text(1.0, i, f"  {int((g.K_P <= 0.05).sum())}/{len(g)}", transform=ax.get_yaxis_transform(), va="center", fontsize=6, color="#546e7a")
    ax.axvline(1, color="#b71c1c", lw=0.8, ls="--")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.8)
    ax.set_xlabel("phylogenetic signal of each character (Blomberg's K; 1 = Brownian motion)", fontsize=8)
    ax.set_title("a  Phylogenetic signal of every character, by structure\n(filled: P ≤ 0.05; ringed: P ≤ 0.05 and K > 1; "
                 "right: characters with signal / tested)",
                 loc="left", fontsize=9.5)
    for f in fams:
        ax.scatter([], [], s=16, color=pal[f], label=f)
    ax.scatter([], [], s=14, facecolor="none", edgecolor="#555555", label="P > 0.05 (open)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2)
    for (f, filled), g in D.assign(_s=sig).groupby(["family", "_s"]):
        ax2.scatter(100 * g.top1, g.K, s=16 if filled else 11, facecolor=pal[str(f)] if filled else "none", edgecolor=pal[str(f)],
                    lw=0.7, alpha=0.9 if filled else 0.5, zorder=3 if filled else 2)
    ax2.axhline(1, color="#b71c1c", lw=0.8, ls="--")
    # every character with signal (P <= 0.05) and K > 1: labelled inside the empty upper part, short labels
    above = D[sig & (D.K > 1)].sort_values("K", ascending=False)
    if len(above):
        ax2.scatter(100 * above.top1, above.K, s=40, facecolor="none", edgecolor=INK, lw=0.6, zorder=4)
        ax.scatter(above.K, above.structure.map(ypos) + jit.loc[above.index], s=40, facecolor="none", edgecolor=INK, lw=0.6, zorder=4)
        short = [short_label(st, lb).replace("forewing: distance ", "").replace("distance ", "")
                 for st, lb in zip(above.structure, above.label)]
        label_column(ax2, 100 * above.top1.values, above.K.values, short, side_x=0.56, fontsize=4.8, gap_frac=1 / 75,
                     colors=[C_ROBUST if x >= a.robust_x else C_WEAK for x in (above.top1 / above.chance_top1)])
    lab = D[sig & ~D.index.isin(above.index)].assign(sc=D.top1 / D.chance_top1).sort_values("sc", ascending=False).head(a.label_top)
    label_column(ax2, 100 * lab.top1.values, lab.K.values, [short_label(st, lb) for st, lb in zip(lab.structure, lab.label)],
                 colors=[C_ROBUST if x >= a.robust_x else C_WEAK for x in (lab.top1 / lab.chance_top1)])
    ax2.set_xlabel("names a withheld specimen (%)", fontsize=8); ax2.set_ylabel("phylogenetic signal (Blomberg's K)", fontsize=8)
    ax2.set_title(f"b  All {len(D)} characters: naming against phylogenetic signal ({int(sig.sum())} with P ≤ 0.05)",
                  loc="left", fontsize=9.5)
    # the key: which characters are labelled, and why (both sets require phylogenetic signal)
    unl = D[~sig]
    by_rel = unl.assign(sc=unl.top1 / unl.chance_top1).sort_values("sc", ascending=False).head(3)
    by_raw = unl.sort_values("top1", ascending=False).head(3)
    fmt = lambda g: "; ".join(f"{short_label(r.structure, r.label)} ({100 * r.top1:.0f}% named, chance {100 * r.chance_top1:.0f}%, "
                              f"K {r.K:.2f}, P {r.K_P:.2f})" for r in g.itertuples())          # noqa: E731
    nrob = int(((D.top1 / D.chance_top1) >= a.robust_x).sum())
    key = (f"Labels.  Blue text: the character names withheld specimens at least {a.robust_x:g}× chance ({nrob} of {len(D)} "
           "characters do); grey text: less.\n"
           f"  In the plot, ringed: every character with phylogenetic signal (P ≤ 0.05) and K > 1 ({len(above)}).\n"
           f"  Right column: the {len(lab)} other characters with signal that name withheld specimens best relative to chance\n"
           "    (% named ÷ % expected by chance, from the character-robustness analysis; fair across structures with few and many species).\n"
           "  Both sets require phylogenetic signal, so the most robust characters WITHOUT signal are not labelled:\n"
           f"    by % named (as ranked in the character-robustness figure): {fmt(by_raw)}\n"
           f"    by % named relative to chance: {fmt(by_rel)}\n"
           "P: permutation test of phylogenetic signal (K larger than with species shuffled across the tree); it does not test K against 1.")
    ax2.text(-1.25, -0.09, key, transform=ax2.transAxes, ha="left", va="top", fontsize=6.6, color=INK, linespacing=1.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False); ax2.spines[s_].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"character_phylo_signal.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--tree", default=None, help="Newick tree, tips = species codes (adds phylogenetic signal)")
    ap.add_argument("--robustness", default=None,
                    help="character_robustness.tsv from biorag_character_robustness_v1.py: rank characters by hold-out "
                         "naming (top-1) and novelty AUC instead of the in-sample eta^2 (recommended)")
    ap.add_argument("--top", type=int, default=25, help="characters per structure in the table (default 25)")
    ap.add_argument("--iterations", type=int, default=9999,
                    help="permutations for K (default 9999: with hundreds of characters, 999 cannot reach FDR q < 0.05)")
    ap.add_argument("--min_species", type=int, default=4, help="species needed to test a character (default 4)")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--figure", action="store_true",
                    help="also draw character_signal_volcano.png/.pdf (off by default: the pipeline's character figure "
                         "is biorag_character_robustness_v1's fig_character_robustness)")
    ap.add_argument("--label_top", type=int, default=12, help="characters labelled in the volcano plot")
    ap.add_argument("--robust_x", type=float, default=3.0,
                    help="in the phylogenetic figure, labels of characters that name withheld specimens at least this many "
                         "times chance are drawn in blue (default 3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    md = Path(a.matrix_dir); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    L = pd.read_csv(md / "specimen_matrix_long.csv")
    fd = pd.read_csv(md / "feature_dictionary.tsv", sep="\t").set_index("feature_id")
    tree_txt = Path(a.tree).read_text() if a.tree else None
    tips = {t.name for t in dp.tips_of(dp.parse_newick(tree_txt))} if tree_txt else set()
    rng = np.random.default_rng(a.seed); ccache = {}; rows = []
    for fid, g in L.groupby("feature_id"):
        g = g.dropna(subset=["value"])
        per = g.groupby("specimen_id").agg(species=("species", "first"), value=("value", "mean"))
        cnt = per.species.value_counts(); multi = cnt[cnt >= 2].index
        row = {"feature_id": fid, "structure": fd.loc[fid, "category"] if fid in fd.index else fid.split(".")[0],
               "label": fd.loc[fid, "label"] if fid in fd.index else fid, "family": fd.loc[fid, "family"] if fid in fd.index else "",
               "tier": fd.loc[fid, "tier"] if fid in fd.index else "", "specimens": len(per), "species": per.species.nunique()}
        sub = per[per.species.isin(multi)]
        if len(multi) >= max(2, a.min_species) and sub.value.nunique() > 1:
            groups = [v.values for _, v in sub.groupby("species").value]
            H, p = kruskal(*groups); n, k = len(sub), len(groups)
            row.update(H=H, p=p, eta2=max(0.0, (H - k + 1) / (n - k)) if n > k else np.nan, kw_species=k)
        if tree_txt:
            means = per.groupby("species").value.mean()
            sp = sorted(set(means.index) & tips)
            if len(sp) >= a.min_species and means.loc[sp].nunique() > 1:
                key = frozenset(sp)
                if key not in ccache:
                    t = dp.prune(dp.parse_newick(tree_txt), set(sp)); tp = dp.tips_of(t)
                    ccache[key] = ([x.name for x in tp], dp.vcv(tp, tp))
                order, C = ccache[key]
                K, P = blomberg_k_perm(means.loc[order].values.astype(float), C, a.iterations, rng)
                row.update(K=K, K_P=P, tree_species=len(order))
        rows.append(row)
    A = pd.DataFrame(rows)
    A["q"] = bh(A.get("p", pd.Series(np.nan, index=A.index)).values)
    if "K_P" in A:
        A["K_q"] = bh(A["K_P"].values)
    rob = a.robustness is not None
    if rob:
        Rb = pd.read_csv(a.robustness, sep="\t")
        keep = ["feature_id", "top1", "top1_ci_low", "top1_ci_high", "chance_top1", "times_chance", "novelty_auc", "species_tested"]
        A = A.merge(Rb[[c for c in keep if c in Rb.columns]], on="feature_id", how="left")
        A = A.sort_values(["structure", "top1", "novelty_auc"], ascending=[True, False, False])
        rank_col = "top1"
    else:
        A = A.sort_values(["structure", "eta2"], ascending=[True, False])
        rank_col = "eta2"
    A.to_csv(out / "character_signal_all.tsv", sep="\t", index=False)
    T = A.dropna(subset=[rank_col]).groupby("structure", group_keys=False).head(a.top)
    T.to_csv(out / "character_signal_top.tsv", sep="\t", index=False)

    # readable table
    how = ("hold-out naming (each specimen withheld; share whose own species is nearest on this character alone)"
           if rob else "species separation (Kruskal-Wallis eta^2)")
    lines = [f"# Most informative characters per structure (top {a.top} by {how})", "",
             ("named: % of withheld specimens named (95% interval), chance in brackets; novelty AUC: tells a described "
              "species from an unseen one (0.5 = no better than chance) - the measures drawn in the character-robustness "
              "figure (fig_character_robustness); " if rob else "")
             + "eta2: Kruskal-Wallis effect size across species; q: Benjamini-Hochberg over all characters"
             + ("; K: Blomberg's K on the tree (P: permutation, " + str(a.iterations) + ")" if tree_txt else ""), ""]
    for st, g in T.groupby("structure", sort=False):
        lines += [f"## {st} ({len(g)} of {int((A.structure == st).sum())} characters)", "",
                  "| character | species |" + (" named % (95% CI) [chance] | novelty AUC |" if rob else "") + " eta2 | q |"
                  + (" K | P |" if tree_txt else ""),
                  "|---|---|" + ("---|---|" if rob else "") + "---|---|" + ("---|---|" if tree_txt else "")]
        for r in g.itertuples():
            s = f"| {r.label} | {r.species} |"
            if rob:
                s += (f" {100 * r.top1:.0f} ({100 * r.top1_ci_low:.0f}–{100 * r.top1_ci_high:.0f}) [{100 * r.chance_top1:.0f}] |"
                      f" {r.novelty_auc:.2f} |") if pd.notna(r.top1) else " - | - |"
            s += f" {r.eta2:.2f} | {r.q:.1e} |" if pd.notna(r.eta2) else " - | - |"
            if tree_txt:
                s += f" {r.K:.2f} | {r.K_P:.3f} |" if pd.notna(getattr(r, "K", np.nan)) else " - | - |"
            lines.append(s)
        lines.append("")
    (out / "character_signal_top.md").write_text("\n".join(lines))

    # figure
    if rob and tree_txt:                    # the phylogenetic companion to the character-robustness figure
        _phylo_figure(A, out, a)
    if a.figure and rob:
        _robust_figure(A, out, tree_txt, a)
    D = A.dropna(subset=["eta2", "q"]).copy()
    if a.figure and not rob:
        _eta_figure(D, out, tree_txt, a)
    summ = {"characters": int(len(A)), "ranked_by": rank_col, "tested_for_separation": int(A.eta2.notna().sum()),
            "separate_species_q_le_fdr": int((A.q <= a.fdr).sum()), "top_per_structure": a.top,
            "structures": int(A.structure.nunique())}
    if tree_txt:
        summ.update(tested_for_signal=int(A.K.notna().sum()), signal_P_le_0_05=int((A.K_P <= 0.05).sum()),
                    signal_q_le_fdr=int((A.K_q <= a.fdr).sum()), iterations=a.iterations,
                    separate_and_signal=int(((A.q <= a.fdr) & (A.K_P <= 0.05)).sum()))
    json.dump(summ, open(out / "character_signal_summary.json", "w"), indent=2)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
