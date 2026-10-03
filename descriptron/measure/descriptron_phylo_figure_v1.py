#!/usr/bin/env python3
"""
descriptron_phylo_figure_v1.py - one summary figure for the phylogenetic comparative analyses (pipeline step 7.2)
================================================================================================================

  a  the species tree (tips = group labels); with --dna_dir, species not monophyletic in the gene tree are marked
  b  colour-pattern phylomorphospace of the structure whose colour pattern best identifies species
     (leave-one-out nearest-specimen accuracy, among structures with >= --min_species species), one specimen
     photograph per species when --image_dir is given
  c  colour-pattern phylomorphospace of the best such structure that covers every species (if different from b)
  d  multivariate phylogenetic signal (Kmult) of every structure x trait set (descriptron_phylo.py output)
  e  with --dna_dir: barcode gap per species (dna_vs_morphology_v1.py); otherwise the colour-pattern
     identification accuracy of every structure, i.e. the ranking that chose b and c

The structures in b and c are chosen by how well their colour pattern separates species, never by their
phylogenetic result, so the choice cannot favour a signal. The ranking is written to colour_identification_rank.tsv.

  python descriptron_phylo_figure_v1.py --tree tree.nwk --phylo_dir <out>/phylo --colour_dir <out>/color_homology \
      --groups group_labels.csv [--image_dir images/] [--dna_dir dna/] --out <out>/phylo/phylo_summary_figure
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import descriptron_phylo as dp                       # noqa: E402

INK, MUTED, GRID = "#1d2327", "#6b7780", "#d0d7de"
C_OK, C_BAD = "#0072B2", "#D55E00"
SETS = [("outline_shape", "outline"), ("measurements", "size"), ("colour_pattern", "colour"), ("texture", "texture")]
Image.MAX_IMAGE_PIXELS = None


def colour_table(colour_dir: Path, cat: str):
    f = sorted((colour_dir / cat).glob(f"color_homology_features_{cat}*.csv"))
    return f[0] if f else None


def identification_rank(colour_dir: Path, gmap) -> pd.DataFrame:
    """leave-one-out nearest-specimen species accuracy of each structure's colour pattern (standardised features)"""
    rows = []
    for d in sorted(p for p in colour_dir.iterdir() if p.is_dir()):
        f = colour_table(colour_dir, d.name)
        if f is None:
            continue
        t = pd.read_csv(f); idc = t.columns[0]
        sp = np.array([next((gmap[k] for k in dp._key_forms(v) if k in gmap), None) for v in t[idc]], dtype=object)
        X = t.drop(columns=[idc]).apply(pd.to_numeric, errors="coerce")
        X = X.loc[:, X.notna().mean() > 0.9]
        if X.shape[1] == 0:
            continue
        X = X.fillna(X.median()); X = ((X - X.mean()) / X.std().replace(0, 1)).fillna(0).values
        ok = sp != None                                                  # noqa: E711
        X, sp = X[ok], sp[ok]
        n = pd.Series(sp).value_counts(); keep = np.isin(sp, n[n >= 2].index); X, sp = X[keep], sp[keep]
        if len(set(sp)) < 2:
            continue
        D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
        rows.append({"structure": d.name, "species": len(set(sp)), "specimens": len(sp), "features": X.shape[1],
                     "colour_identification_accuracy": round(float((sp[D.argmin(1)] == sp).mean()), 3)})
    return pd.DataFrame(rows).sort_values("colour_identification_accuracy", ascending=False)


def crop_object(path, size=160):
    """crop a photograph to the largest blob that differs from the border colour"""
    from scipy import ndimage
    im = Image.open(path).convert("RGB"); sm = im.copy(); sm.thumbnail((800, 800)); a = np.asarray(sm).astype(float)
    border = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3), a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
    bg = np.median(border, 0); mask = ndimage.binary_opening(np.linalg.norm(a - bg, axis=2) > 45, iterations=2)
    lab, n = ndimage.label(mask)
    if n:
        big = 1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1)))); ys, xs = np.nonzero(lab == big); f = im.width / sm.width
        x0, x1, y0, y1 = xs.min() * f, xs.max() * f, ys.min() * f, ys.max() * f; p = 0.04 * max(x1 - x0, y1 - y0)
        im = im.crop((max(0, x0 - p), max(0, y0 - p), min(im.width, x1 + p), min(im.height, y1 + p)))
    im.thumbnail((size, size)); return np.asarray(im)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--phylo_dir", required=True, help="descriptron_phylo output: <structure>/phylogenetic_signal.csv")
    ap.add_argument("--colour_dir", required=True, help="colour homology output: <structure>/color_homology_features_*.csv")
    ap.add_argument("--groups", required=True)
    ap.add_argument("--image_dir", default=None, help="photographs, for one thumbnail per species in panel b")
    ap.add_argument("--dna_dir", default=None, help="dna_vs_morphology_v1 output (barcode_gap.tsv); "
                    "morphospecies_monophyly.tsv from coi_species_tree_v1 in it or next to the tree")
    ap.add_argument("--min_species", type=int, default=10)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True, help="output path without extension (.png and .pdf are written)")
    a = ap.parse_args(argv)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tree_txt = Path(a.tree).read_text(); tree = dp.parse_newick(tree_txt)
    gmap = dp.load_group_map(a.groups)
    colour_dir = Path(a.colour_dir)
    K = pd.concat([pd.read_csv(f).assign(structure=f.parent.name)
                   for f in sorted(Path(a.phylo_dir).glob("*/phylogenetic_signal.csv"))], ignore_index=True)
    K.to_csv(out.parent / "phylogenetic_signal_all_structures.tsv", sep="\t", index=False)
    R = identification_rank(colour_dir, gmap)
    R = R[R.structure.isin(set(K.loc[K.trait_set == "colour_pattern", "structure"]))]
    R.to_csv(out.parent / "colour_identification_rank.tsv", sep="\t", index=False)
    if R.empty:
        sys.exit("no colour-pattern table with phylogenetic results: nothing to draw")
    elig = R[R.species >= a.min_species]
    pick_b = (elig if len(elig) else R).iloc[0].structure
    full = R[R.species == R.species.max()]
    pick_c = full.iloc[0].structure if full.iloc[0].structure != pick_b else (full.iloc[1].structure if len(full) > 1 else None)

    notmono, gap = set(), None
    if a.dna_dir:
        dd = Path(a.dna_dir)
        mono = next((p for p in (dd / "morphospecies_monophyly.tsv", Path(a.tree).parent / "morphospecies_monophyly.tsv") if p.exists()), None)
        if mono is not None:
            M = pd.read_csv(mono, sep="\t"); notmono = set(M.loc[~M.monophyletic.astype(bool), "species"])
        if (dd / "barcode_gap.tsv").exists():
            gap = pd.read_csv(dd / "barcode_gap.tsv", sep="\t")

    fig = plt.figure(figsize=(7.2, 10.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 0.95], width_ratios=[0.8, 1.2], hspace=0.5, wspace=0.5)
    if a.title:
        fig.suptitle(a.title, fontsize=9, weight="bold", color=INK, y=0.995)

    # a: tree
    ax = fig.add_subplot(gs[0, 0]); tips = dp.tips_of(tree); y = {id(t): i for i, t in enumerate(tips)}

    def ypos(n):
        if id(n) not in y:
            y[id(n)] = np.mean([ypos(k) for k in n.kids])
        return y[id(n)]

    def draw(n):
        for k in n.kids:
            ax.plot([n.depth, n.depth], [ypos(n), ypos(k)], color=INK, lw=0.6)
            ax.plot([n.depth, k.depth], [ypos(k), ypos(k)], color=INK, lw=0.6); draw(k)
    draw(tree); xmax = max(t.depth for t in tips) or 1.0
    for t in tips:
        ax.text(t.depth + 0.012 * xmax, y[id(t)], t.name, va="center", fontsize=max(3.6, min(5.6, 160 / len(tips))),
                color=C_BAD if t.name in notmono else INK)
    bar = float(f"{0.2 * xmax:.1g}")
    ax.plot([0, bar], [len(tips) + 0.2] * 2, color=INK, lw=0.8); ax.text(bar / 2, len(tips) + 0.5, f"{bar:g}", ha="center", va="top", fontsize=5.2)
    ax.set_xlim(-0.02 * xmax, xmax * 1.3); ax.set_ylim(len(tips) + 2.8, -1); ax.axis("off")
    ax.set_title("a  Species tree", loc="left", fontsize=8, weight="bold", color=INK)
    if notmono:
        ax.text(0, len(tips) + 2.4, "orange: not monophyletic in the gene tree", fontsize=5.2, color=C_BAD)

    def morphospace(ax, cat, thumbs, letter):
        f = colour_table(colour_dir, cat)
        sp, cols, Mx, _ = dp.load_traits(str(f), None, None, None, gmap)
        Mx = dp.scale(Mx, "standardize")
        t = dp.prune(dp.parse_newick(tree_txt), set(sp))              # prune works in place: a fresh tree each time
        tp = dp.tips_of(t); nodes = dp.internal_nodes(t)
        Y = Mx[[sp.index(x.name) for x in tp]]; anc = dp.ancestral_states(Y, tp, nodes); mu = Y.mean(0)
        _, sv, Vt = np.linalg.svd(Y - mu, full_matrices=False); ev = sv ** 2 / (sv ** 2).sum()
        pt, pn = (Y - mu) @ Vt[:2].T, (anc - mu) @ Vt[:2].T
        pos = {id(x): pt[j] for j, x in enumerate(tp)}; pos.update({id(x): pn[j] for j, x in enumerate(nodes)})
        for nd in nodes:
            for k in nd.kids:
                p, q = pos[id(nd)], pos[id(k)]; ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, lw=0.5, zorder=1)
        ax.scatter(pn[:, 0], pn[:, 1], s=6, color="white", edgecolor=MUTED, lw=0.5, zorder=2)
        col = lambda nm: C_BAD if nm in notmono else C_OK                  # noqa: E731
        drawn = False
        if thumbs and a.image_dir:
            fn = pd.read_csv(f, usecols=[0]).iloc[:, 0]; first = {}
            for v in fn:
                s = next((gmap[k] for k in dp._key_forms(v) if k in gmap), None)
                first.setdefault(s, re.sub(r"(\.(?:tiff?|png|jpe?g))_\d+$", r"\1", str(v), flags=re.I))
            try:
                for j in np.argsort(np.hypot(pt[:, 0], pt[:, 1])):
                    img = crop_object(Path(a.image_dir) / first[tp[j].name])
                    ab = AnnotationBbox(OffsetImage(img, zoom=0.16), pt[j], frameon=True, pad=0.05,
                                        bboxprops=dict(edgecolor=col(tp[j].name), lw=0.8, facecolor="white"))
                    ab.set_zorder(3 + j); ax.add_artist(ab)
                    ax.annotate(tp[j].name, pt[j], xytext=(0, -15), textcoords="offset points", ha="center", fontsize=5, zorder=60, color=INK)
                ax.scatter(pt[:, 0], pt[:, 1], s=0); ax.margins(0.16); drawn = True
            except (OSError, KeyError) as e:                               # missing photograph: fall back to points
                print(f"  thumbnails skipped ({e})")
                for art in list(ax.artists):
                    art.remove()
        if not drawn:
            ax.scatter(pt[:, 0], pt[:, 1], s=15, c=[col(x.name) for x in tp], zorder=3, edgecolor="white", lw=0.4)
            for j, x in enumerate(tp):
                ax.annotate(x.name, pt[j], xytext=(2, 2), textcoords="offset points", fontsize=4.6, color=MUTED)
        r = K[(K.structure == cat) & (K.trait_set == "colour_pattern")].iloc[0]
        acc = R.set_index("structure").loc[cat, "colour_identification_accuracy"]
        ax.text(0.0, 1.0, f"{len(tp)} spp., {len(cols)} features; Kmult {r.Kmult:.3f}, P = {r.P:.3f}; "
                f"identifies {100 * acc:.0f}%", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.6, color=INK)
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}%)", fontsize=6.4); ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)", fontsize=6.4)
        ax.tick_params(labelsize=5.6); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{letter}  {cat.replace('_', ' ')}: colour pattern", loc="left", fontsize=8, weight="bold", color=INK, pad=11)

    morphospace(fig.add_subplot(gs[0, 1]), pick_b, True, "b")
    if pick_c:
        morphospace(fig.add_subplot(gs[1, 0]), pick_c, False, "c")

    # d: Kmult grid
    ax = fig.add_subplot(gs[1:, 1])
    st = K.groupby("structure")["n_species"].max().sort_values(ascending=False); rows = list(st.index)
    for i, s in enumerate(rows):
        for j, (ts, _) in enumerate(SETS):
            r = K[(K.structure == s) & (K.trait_set == ts)]
            if r.empty:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="white", edgecolor=GRID, lw=0.4)); continue
            r = r.iloc[0]; p = r.P
            c = "#08519c" if p <= 0.01 else "#6baed6" if p <= 0.05 else "#f2f4f6"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=c, edgecolor="white", lw=0.8))
            ax.text(j + 0.5, i + 0.5, f"{r.Kmult:.2f}", ha="center", va="center", fontsize=5, color="white" if p <= 0.01 else INK)
    ax.set_xlim(0, len(SETS)); ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(SETS)) + 0.5); ax.set_xticklabels([l for _, l in SETS], fontsize=6); ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels([f"{s} ({st[s]})" for s in rows], fontsize=max(4.2, min(5.4, 140 / len(rows))))
    ax.tick_params(length=0); [s_.set_visible(False) for s_ in ax.spines.values()]
    for c, lab in (("#08519c", "P ≤ 0.01"), ("#6baed6", "P ≤ 0.05"), ("#f2f4f6", "P > 0.05")):
        ax.plot([], [], "s", color=c, label=lab, mec=GRID, ms=5)
    ax.legend(fontsize=5.4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=3)
    nsig = int((K.P <= 0.05).sum())
    ax.set_title(f"d  Phylogenetic signal (Kmult)\n{nsig} of {len(K)} tests P ≤ 0.05, uncorrected", loc="left",
                 fontsize=8, weight="bold", color=INK, pad=18)

    # e: barcode gap, or the identification ranking
    ax = fig.add_subplot(gs[2, 0])
    if gap is not None and gap["max_intra"].notna().any():
        g2 = gap.dropna(subset=["max_intra"]); neg = g2.gap <= 0
        ax.scatter(100 * g2.max_intra[~neg], 100 * g2.min_inter[~neg], s=14, color=C_OK, zorder=3, edgecolor="white", lw=0.4, label=f"gap > 0 ({int((~neg).sum())})")
        ax.scatter(100 * g2.max_intra[neg], 100 * g2.min_inter[neg], s=14, color=C_BAD, zorder=3, edgecolor="white", lw=0.4, label=f"no gap ({int(neg.sum())})")
        for _, r in g2[neg].iterrows():
            ax.annotate(r.species, (100 * r.max_intra, 100 * r.min_inter), xytext=(3, -1), textcoords="offset points", fontsize=5, color=C_BAD)
        m = 100 * max(g2.max_intra.max(), g2.min_inter.max()) * 1.05
        ax.plot([0, m], [0, m], color=MUTED, lw=0.5, ls=":"); ax.set_xlim(0, m); ax.set_ylim(0, m)
        ax.set_xlabel("largest distance within the species (% K2P)", fontsize=6.2); ax.set_ylabel("smallest distance to another (% K2P)", fontsize=6.2)
        ax.legend(fontsize=5.4, frameon=False, loc="lower right")
        ax.set_title(f"e  Barcode gap ({len(g2)} spp.)", loc="left", fontsize=8, weight="bold", color=INK)
    else:
        top = R.head(15)
        cs = [C_OK if s in (pick_b, pick_c) else "#9aa5ad" for s in top.structure]
        ax.barh(range(len(top)), 100 * top.colour_identification_accuracy, color=cs, height=0.65)
        ax.set_yticks(range(len(top))); ax.set_yticklabels([f"{s} ({n})" for s, n in zip(top.structure, top.species)], fontsize=5.2)
        ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("specimens identified from colour pattern (%)", fontsize=6.2)
        ax.set_title("e  Colour pattern: identification", loc="left", fontsize=8, weight="bold", color=INK)
    ax.tick_params(labelsize=5.6); ax.spines[["top", "right"]].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=300, bbox_inches="tight", facecolor="white")
    vals = {"panel_b": pick_b, "panel_c": pick_c, "rule": f"best colour-pattern identification with >= {a.min_species} species; "
            "best covering all species", "n_tests": len(K), "n_P_le_0.05": nsig, "not_monophyletic": sorted(notmono)}
    json.dump(vals, open(out.with_suffix(".json"), "w"), indent=1)
    print(json.dumps(vals))


if __name__ == "__main__":
    main()
