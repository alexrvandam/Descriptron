#!/usr/bin/env python3
"""
biorag_graph_support_figure_v1.py — how far apart are the species, and what are
the characters that are supposed to keep them apart actually worth?
===============================================================================

A published treatment asserts, for every species, the range it occupies in each
character. That range is the minimum and maximum of a handful of specimens, so
it is a statement about the material collected, not about the species. Two
consequences follow, and this script measures both on the same page.

  * On paper, a character with a narrow asserted range separates many species
    pairs: their stated intervals do not overlap. That is the graph's support
    for keeping a pair apart.
  * In the field, the next conspecific specimen falls outside the observed range
    of n specimens with probability about 2/(n+1) — so for n = 3 the asserted
    range is broken half the time, whatever the pair separation says.

Everything here that involves a specimen is therefore computed with that
specimen withheld from the statistic it is tested against.

DEFINITIONS (computed exactly as stated)

  1. PAIR SEPARATION (graph), per character. Among the species whose treatment
     asserts that character, the share of species pairs whose ranges do not
     overlap. Per species PAIR: the number and the share of the characters both
     treatments assert whose ranges do not overlap — the graph's support for
     keeping that pair apart.

  2. HOLD-OUT RETENTION (graph), per character. Over all specimens that have a
     value for it and whose species asserts it, the share whose value falls
     INSIDE their own species' range once that range has been re-derived
     without them (`regraph_without`). Alongside it, for the same specimens,
     the retention EXPECTED from sample size alone, mean(1 - 2/(n+1)) with n the
     number of conspecifics left defining the range (n < 1 skipped).

     The two agree exactly, and that is the result, not a coincidence: in a
     series of n+1 specimens exactly two, the smallest and the largest, fall
     outside the min-max range re-derived without them, so the hold-out
     retention of an asserted range IS 1 - 2/(n+1). It is a property of the
     series length, not of the character. What then varies between characters
     is only how long a series each treatment could assert them from -- and the
     characters that separate the most species pairs on paper turn out to be
     the ones asserted from the shortest series.

  3. HOLD-OUT ROUTING SUPPORT (key), per leading character. Over all specimens
     and all couplets on their leave-one-specimen-out path where that feature
     decided the step, the share of decisions that sent the specimen to the side
     whose species list contains its true species. The analogue, for a
     character, of a branch support value.

  Species distances come from the hold-out identification distances already
  computed by the calibration step: for every specimen, with its own record
  withheld, its distance to every species in each of four independent character
  sets, in units of the pooled within-species spread of that set. Per set,
  D[A,B] = median over specimens of A of d(a -> B); symmetrised; averaged over
  the sets in which the pair is defined. D[A,A] (own record withheld) is that
  species' own spread, and the GAP, D[A,B] / mean(D[A,A], D[B,B]), is how many
  own-spreads apart two species sit — the morphological analogue of a barcode
  gap. A gap near 1 means two species are no further apart than the specimens
  of either one are from each other.

  python biorag_graph_support_figure_v1.py \\
      --matrix_dir "$M/compiled_key_tier" --taxon_profile <profile.yaml> \\
      --jsonld_dir "$M/treatments/machine_readable/jsonld" \\
      --distances "$M/calibration/matrix_identification_distances.tsv" \\
      --identification "$M/instrument_comparison_v3/identification_by_specimen.tsv" \\
      --key_loo_dir "$M/key_fuzzy_loo/keys_loo" --out_dir "$M/graph_support"

Writes species_distance_matrix.tsv, species_pair_support.tsv,
graph_character_support.tsv, key_character_support.tsv,
graph_support_summary.json and fig_graph_support.png / .pdf.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec        # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Patch                                     # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage                  # noqa: E402
from scipy.spatial.distance import squareform                            # noqa: E402
from scipy.stats import spearmanr                                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                      # noqa: E402
from biorag_novelty_score_v1 import Reference                            # noqa: E402
from biorag_graph_identify_v1 import load_graph, regraph_without         # noqa: E402
from biorag_congruence_compare_v1 import wilson                          # noqa: E402

VERSION = "1.0"
TOL = 1e-9

# Okabe-Ito, colour-blind safe; one colour per character family, shared by panels c and d
FAMILY_COLOUR = {
    "length":         "#0072B2",
    "area":           "#56B4E9",
    "landmark_mm":    "#009E73",
    "aspect_ratio":   "#E69F00",
    "ratio":          "#D55E00",
    "landmark_ratio": "#CC79A7",
    "colour":         "#000000",
}
OTHER_COLOUR = "#999999"
FAMILY_LABEL = {
    "length": "length (mm)", "area": "area (mm²)", "landmark_mm": "landmark span (mm)",
    "aspect_ratio": "aspect ratio", "ratio": "ratio", "landmark_ratio": "landmark ratio",
    "colour": "colour (CIE L*a*b*)",
}


def fam_colour(f):
    return FAMILY_COLOUR.get(f, OTHER_COLOUR)


# ─────────────────────────────────────────────────────────────────────────────
# species distances
# ─────────────────────────────────────────────────────────────────────────────

def species_distances(dist_tsv: Path, species: list):
    """Symmetrised species x species distance, averaged over the character sets."""
    d = pd.read_csv(dist_tsv, sep="\t")
    sets = sorted(d["set"].unique())
    per_set = {}
    for s in sets:
        sub = d[d["set"] == s]
        piv = (sub.groupby(["species", "candidate_species"])["d"].median()
                  .unstack().reindex(index=species, columns=species))
        per_set[s] = ((piv + piv.T) / 2).values           # symmetrise
    stack = np.dstack([per_set[s] for s in sets])
    with np.errstate(invalid="ignore"):
        D = np.nanmean(stack, axis=2)                     # average over the sets defined
    n_sets = np.sum(~np.isnan(stack), axis=2)
    diag = np.diag(D).copy()                              # own spread, own record withheld
    gap = D / ((diag[:, None] + diag[None, :]) / 2.0)
    return D, gap, n_sets, sets


# ─────────────────────────────────────────────────────────────────────────────
# the graph
# ─────────────────────────────────────────────────────────────────────────────

def overlap(a, b):
    return not (a[1] < b[0] - TOL or b[1] < a[0] - TOL)


def graph_character_support(graph: dict, ref) -> pd.DataFrame:
    """Definitions 1 and 2, per asserted character."""
    asserters = defaultdict(list)
    for sp, rngs in graph.items():
        for fid in rngs:
            asserters[fid].append(sp)

    # 1. pair separation
    sep = {}
    for fid, spp in asserters.items():
        pairs = sepd = 0
        for i in range(len(spp)):
            for j in range(i + 1, len(spp)):
                pairs += 1
                if not overlap(graph[spp[i]][fid], graph[spp[j]][fid]):
                    sepd += 1
        sep[fid] = (pairs, sepd, sepd / pairs if pairs else np.nan)

    # 2. hold-out retention, and the retention expected from sample size alone
    inside, tested = Counter(), Counter()
    expected = defaultdict(list)
    for sid in ref.raw.index:
        sp = ref.species_of.get(sid)
        if sp not in graph:
            continue
        held = regraph_without(graph, ref, sid, sp).get(sp, {})
        ids = [i for i in ref.raw.index if ref.species_of.get(i) == sp and i != sid]
        sub = ref.raw.loc[ids]
        for fid in graph[sp]:
            if fid not in ref.raw.columns or fid not in held:
                continue
            v = ref.raw.at[sid, fid]
            if v != v:
                continue
            n_left = int(sub[fid].notna().sum()) if fid in sub.columns else 0
            if n_left < 1:                                # nothing left to define a range
                continue
            lo, hi = held[fid]
            tested[fid] += 1
            if lo - TOL <= v <= hi + TOL:
                inside[fid] += 1
            expected[fid].append(1.0 - 2.0 / (n_left + 1))

    fd = ref.fdict
    rows = []
    for fid, spp in sorted(asserters.items()):
        pairs, sepd, share = sep[fid]
        n_t = tested.get(fid, 0)
        rows.append({
            "feature_id": fid,
            "label": str(fd.loc[fid, "label"]) if fid in fd.index else fid,
            "family": str(fd.loc[fid, "family"]) if fid in fd.index else "",
            "n_species_asserting": len(spp),
            "n_pairs": pairs,
            "n_pairs_separated": sepd,
            "pair_separation": round(share, 4) if share == share else np.nan,
            "n_specimens_tested": n_t,
            "holdout_retention": round(inside.get(fid, 0) / n_t, 4) if n_t else np.nan,
            "expected_retention": round(float(np.mean(expected[fid])), 4) if expected[fid] else np.nan,
        })
    return pd.DataFrame(rows), asserters


def graph_pair_support(graph: dict, species: list):
    """Per species pair: shared asserted characters and how many separate them."""
    shared = np.full((len(species), len(species)), np.nan)
    separating = np.full((len(species), len(species)), np.nan)
    share = np.full((len(species), len(species)), np.nan)
    for i, a in enumerate(species):
        for j, b in enumerate(species):
            if i == j or a not in graph or b not in graph:
                continue
            common = set(graph[a]) & set(graph[b])
            if not common:
                shared[i, j] = 0
                continue
            k = sum(1 for f in common if not overlap(graph[a][f], graph[b][f]))
            shared[i, j] = len(common)
            separating[i, j] = k
            share[i, j] = k / len(common)
    return shared, separating, share


# ─────────────────────────────────────────────────────────────────────────────
# the key
# ─────────────────────────────────────────────────────────────────────────────

def walk_key(values: dict, sex, recs: list):
    """biorag_key_builder_v1.identify(vote=False), recording every decision.

    Returns (answer, steps, status); steps = [(couplet, character, side), ...].
    """
    by_num = {r["number"]: r for r in recs}
    n, steps, seen = 1, [], set()
    while True:
        if n not in by_num or n in seen:
            return None, steps, "unresolved"
        seen.add(n)
        r = by_num[n]
        chosen = None
        for c in r["characters"]:
            if c.get("structure_sex") in ("male", "female") and sex != c["structure_sex"]:
                continue
            v = values.get(c["feature_id"])
            if v is None or v != v:
                continue
            is_A = (v <= c["threshold"]) if c["A_operator"] == "<=" else (v > c["threshold"])
            chosen = (c, "A" if is_A else "B")
            break                                         # first measured character decides
        if chosen is None:
            return None, steps, "unresolved"
        c, side = chosen
        steps.append((r, c, side))
        g = r[f"{side}_goto"]
        if not isinstance(g, int):
            return g, steps, "ok"
        n = g


def key_character_support(key_dir: Path, ref):
    """Definition 3, plus the walker's own anchor (named / correct)."""
    hit, tot = Counter(), Counter()
    label, family = {}, {}
    named = correct = walked = 0
    for sid in ref.raw.index:
        p = key_dir / f"{sid}.json"
        if not p.exists():
            continue
        recs = json.loads(p.read_text())["couplets"]
        row = ref.raw.loc[sid]
        values = {k: float(v) for k, v in row.items() if v == v}
        ans, steps, _ = walk_key(values, ref.sex_of.get(sid), recs)
        walked += 1
        truth = ref.species_of.get(sid)
        if ans is not None:
            named += 1
            correct += int(ans == truth)
        for r, c, side in steps:
            fid = c["feature_id"]
            tot[fid] += 1
            label[fid] = c.get("label", fid)
            family[fid] = c.get("family", "")
            if truth in r.get(f"{side}_species", []):
                hit[fid] += 1
    rows = []
    for fid, n in tot.items():
        lo, hi = wilson(hit[fid], n)                      # returned as percentages
        rows.append({"feature_id": fid, "label": label[fid], "family": family[fid],
                     "n_decisions": n, "n_correct_side": hit[fid],
                     "routing_support": round(hit[fid] / n, 4),
                     "ci_low": round(lo / 100.0, 4), "ci_high": round(hi / 100.0, 4)})
    df = pd.DataFrame(rows).sort_values("routing_support", ascending=False)
    return df, {"specimens_walked": walked, "named": named, "correct": correct,
                "decisions": int(sum(tot.values()))}


# ─────────────────────────────────────────────────────────────────────────────
# classical (metric) MDS
# ─────────────────────────────────────────────────────────────────────────────

def classical_mds(D: np.ndarray, k: int = 2):
    n = D.shape[0]
    Dz = D.copy()
    np.fill_diagonal(Dz, 0.0)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (Dz ** 2) @ J
    w, V = np.linalg.eigh((B + B.T) / 2)
    idx = np.argsort(w)[::-1][:k]
    L = np.clip(w[idx], 0, None)
    X = V[:, idx] * np.sqrt(L)
    dd = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    iu = np.triu_indices(n, 1)
    stress = float(np.sqrt(((Dz[iu] - dd[iu]) ** 2).sum() / (Dz[iu] ** 2).sum()))
    var = float(L.sum() / np.clip(w, 0, None).sum())
    return X, stress, var


def repel_labels(ax, xy, texts, fontsize=6.0, iters=500, pad=0.008, radius=0.13,
                 avoid=None, points=None, point_pad=0.012):
    """Label repulsion in axes-fraction space (adjustText is not installed here).

    Text sizes are measured with the renderer, so labels are kept inside the axes
    and off one another whatever the panel's shape. `avoid` is a list of extra
    axes-fraction (x, y, w, h) boxes the labels must stay out of.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    probe = ax.text(0.5, 0.5, "", fontsize=fontsize, transform=ax.transAxes)
    w, h = [], []
    for t in texts:
        probe.set_text(t)
        bb = probe.get_window_extent(renderer=rend)
        o = inv.transform((0.0, 0.0))
        p = inv.transform((bb.width, bb.height))
        w.append(abs(p[0] - o[0]) + 0.012)
        h.append(abs(p[1] - o[1]) + 0.010)
    probe.remove()
    w, h = np.asarray(w), np.asarray(h)

    anchor = np.array([inv.transform(ax.transData.transform((x, y))) for x, y in xy])
    # start the labels alternately above and below their point: two labels that would
    # otherwise begin on top of one another then separate instead of jamming
    lab = anchor + np.column_stack([np.zeros(len(anchor)),
                                    np.where(np.arange(len(anchor)) % 2 == 0, 0.032, -0.032)])
    boxes = [np.asarray(b, float) for b in (avoid or [])]
    if points is not None and len(points):
        pts = np.array([inv.transform(ax.transData.transform((x, y))) for x, y in points])
        boxes += [np.array([px - point_pad, py - point_pad, 2 * point_pad, 2 * point_pad])
                  for px, py in pts]
    for _ in range(iters):
        move = np.zeros_like(lab)
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                dx, dy = lab[i, 0] - lab[j, 0], lab[i, 1] - lab[j, 1]
                ox = (w[i] + w[j]) / 2 + pad - abs(dx)
                oy = (h[i] + h[j]) / 2 + pad - abs(dy)
                if ox > 0 and oy > 0:
                    if ox / (w[i] + w[j]) < oy / (h[i] + h[j]):
                        s = np.sign(dx) if dx else 1.0
                        move[i, 0] += 0.5 * s * ox
                        move[j, 0] -= 0.5 * s * ox
                    else:
                        s = np.sign(dy) if dy else 1.0
                        move[i, 1] += 0.5 * s * oy
                        move[j, 1] -= 0.5 * s * oy
            for bx, by, bw, bh in boxes:                    # keep out of reserved boxes
                cx, cy = bx + bw / 2, by + bh / 2
                dx, dy = lab[i, 0] - cx, lab[i, 1] - cy
                ox = (w[i] + bw) / 2 - abs(dx)
                oy = (h[i] + bh) / 2 - abs(dy)
                if ox > 0 and oy > 0:
                    if ox < oy:
                        move[i, 0] += (np.sign(dx) if dx else 1.0) * ox
                    else:
                        move[i, 1] += (np.sign(dy) if dy else 1.0) * oy
            d = lab[i] - anchor[i]                          # stay near its own point
            r = float(np.hypot(*d))
            if r > radius:
                move[i] -= 0.6 * d * (1 - radius / r)
            elif r < 0.020:
                move[i] += np.array([0.0, 0.014])
        lab += 0.55 * move
        lab[:, 0] = np.clip(lab[:, 0], w / 2 + 0.006, 1 - w / 2 - 0.006)
        lab[:, 1] = np.clip(lab[:, 1], h / 2 + 0.006, 1 - h / 2 - 0.006)

    # the point boxes can hold two labels together in a dense cloud, so finish with a
    # hard pass that resolves label-on-label overlap and nothing else
    for _ in range(400):
        worst = 0.0
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                dx, dy = lab[i, 0] - lab[j, 0], lab[i, 1] - lab[j, 1]
                ox = (w[i] + w[j]) / 2 + pad - abs(dx)
                oy = (h[i] + h[j]) / 2 + pad - abs(dy)
                if ox > 0 and oy > 0:
                    worst = max(worst, min(ox, oy))
                    if ox / (w[i] + w[j]) < oy / (h[i] + h[j]):
                        s = np.sign(dx) if dx else 1.0
                        lab[i, 0] += 0.55 * s * ox
                        lab[j, 0] -= 0.55 * s * ox
                    else:
                        s = np.sign(dy) if dy else 1.0
                        lab[i, 1] += 0.55 * s * oy
                        lab[j, 1] -= 0.55 * s * oy
        lab[:, 0] = np.clip(lab[:, 0], w / 2 + 0.006, 1 - w / 2 - 0.006)
        lab[:, 1] = np.clip(lab[:, 1], h / 2 + 0.006, 1 - h / 2 - 0.006)
        if worst < 1e-4:
            break

    for (lx, ly), (axx, ayy), t in zip(lab, anchor, texts):
        if np.hypot(lx - axx, ly - ayy) > 0.030:
            ax.plot([axx, lx], [ayy, ly], lw=0.35, color="0.55", zorder=2,
                    transform=ax.transAxes)
        ax.text(lx, ly, t, fontsize=fontsize, ha="center", va="center", zorder=7,
                color="0.12", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="none", alpha=0.78))


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Species distances, knowledge-graph character support and key routing support")
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--jsonld_dir", required=True)
    ap.add_argument("--distances", required=True)
    ap.add_argument("--identification", required=True)
    ap.add_argument("--key_loo_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_species_asserting", type=int, default=8,
                    help="panel c: characters asserted by at least this many species")
    ap.add_argument("--min_decisions", type=int, default=10,
                    help="panel d: key characters that decided at least this many hold-out steps")
    ap.add_argument("--dpi", type=int, default=220)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    profile = pol.load_taxon_profile(a.taxon_profile)
    ref = Reference(Path(a.matrix_dir), profile)
    species = list(ref.species)
    n_spec = {s: sum(1 for v in ref.species_of.values() if v == s) for s in species}

    # the published character lists, with every range recomputed from the current matrix
    graph = load_graph(Path(a.jsonld_dir))
    for sp in list(graph):
        graph = regraph_without(graph, ref, None, sp)
    ranges_per_sp = [len(v) for v in graph.values()]
    no_ranges = sorted(set(species) - set(graph))
    print(f"graph: {len(graph)}/{len(species)} species carry ranges, "
          f"median {np.median(ranges_per_sp):.1f} ranges each; no ranges: {', '.join(no_ranges)}")

    # distances
    D, gap, n_sets, set_names = species_distances(Path(a.distances), species)
    pd.DataFrame(D, index=species, columns=species).to_csv(
        out / "species_distance_matrix.tsv", sep="\t", float_format="%.5f")

    # graph support
    chars, asserters = graph_character_support(graph, ref)
    chars.to_csv(out / "graph_character_support.tsv", sep="\t", index=False)
    shared, separating, share = graph_pair_support(graph, species)

    pair_rows = []
    for i in range(len(species)):
        for j in range(i + 1, len(species)):
            pair_rows.append({
                "species_a": species[i], "species_b": species[j],
                "distance": round(float(D[i, j]), 5),
                "gap": round(float(gap[i, j]), 5),
                "sets_defined": int(n_sets[i, j]),
                "shared_characters": (int(shared[i, j]) if shared[i, j] == shared[i, j] else np.nan),
                "separating_characters": (int(separating[i, j])
                                          if separating[i, j] == separating[i, j] else np.nan),
                "separating_share": (round(float(share[i, j]), 4)
                                     if share[i, j] == share[i, j] else np.nan)})
    pairs = pd.DataFrame(pair_rows).sort_values("gap")
    pairs.to_csv(out / "species_pair_support.tsv", sep="\t", index=False)

    # key support
    keydf, anchor = key_character_support(Path(a.key_loo_dir), ref)
    keydf.to_csv(out / "key_character_support.tsv", sep="\t", index=False)
    print(f"key walker: {anchor['named']} of {anchor['specimens_walked']} specimens named, "
          f"{anchor['correct']} correct, {anchor['decisions']} hold-out decisions")

    # how well the matrix names each species under hold-out (panel a colour)
    ident = pd.read_csv(Path(a.identification), sep="\t")
    ident["ok"] = ident["matrix"].astype(str) == ident["species"].astype(str)
    acc = ident.groupby("species")["ok"].mean()
    node_acc = np.array([acc.get(s, np.nan) for s in species], dtype=float)

    # MDS
    X, stress, var_expl = classical_mds(D, 2)

    # ── figure ──────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 7.4, "axes.titlesize": 8.8, "axes.labelsize": 7.4,
                         "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
                         "axes.titleweight": "bold", "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(9.2, 11.8))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.20, 1.02, 0.90],
                  left=0.085, right=0.975, top=0.960, bottom=0.098,
                  wspace=0.34, hspace=0.36)

    # ---- panel a -----------------------------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    tidy(axa)
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    dmin = float(Dm.min())
    drawn = set()
    for i in range(len(species)):
        for j in np.argsort(Dm[i])[:2]:
            e = tuple(sorted((i, int(j))))
            if e in drawn:
                continue
            drawn.add(e)
            lw = 0.25 + 2.2 * (dmin / D[e[0], e[1]]) ** 3
            axa.plot(X[[e[0], e[1]], 0], X[[e[0], e[1]], 1], lw=lw, color="#7f8fa6",
                     alpha=0.8, zorder=1, solid_capstyle="round")
    sizes = np.array([26 + 24 * n_spec[s] for s in species])
    sc = axa.scatter(X[:, 0], X[:, 1], s=sizes, c=node_acc, cmap="viridis",
                     vmin=0, vmax=1, edgecolor="white", linewidth=0.7, zorder=4)
    axa.set_xlabel("MDS axis 1 (within-species spreads)")
    axa.set_ylabel("MDS axis 2 (within-species spreads)")
    axa.set_title("a  How far apart the species are", loc="left")
    axa.margins(0.15)
    axa.set_aspect("equal", adjustable="datalim")
    axa.text(0.992, 0.012,
             f"classical MDS, stress-1 = {stress:.3f}\n"
             f"{100*var_expl:.0f}% of the distance structure on 2 axes\n"
             f"lines join each species to its two nearest\n"
             f"node area \u221d specimens ({min(n_spec.values())}\u2013{max(n_spec.values())})",
             transform=axa.transAxes, va="bottom", ha="right", fontsize=5.9, color="0.32",
             linespacing=1.35)
    cba = fig.colorbar(sc, ax=axa, fraction=0.040, pad=0.025)
    cba.set_label("share of that species' specimens\nthe character matrix names\ncorrectly (hold-out)",
                  fontsize=6.0, labelpad=2)
    cba.ax.tick_params(labelsize=5.8, length=1.8, pad=1.2)
    repel_labels(axa, X, species, fontsize=5.9, radius=0.135, iters=800, pad=0.010,
                 avoid=[(0.50, 0.0, 0.50, 0.185)], points=X, point_pad=0.019)

    # ---- panel b -----------------------------------------------------------
    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1],
                                  width_ratios=[0.135, 1.0], height_ratios=[1.0, 0.30],
                                  wspace=0.035, hspace=0.06)
    axd = fig.add_subplot(sub[0, 0])
    axb = fig.add_subplot(sub[0, 1])
    axcb = fig.add_subplot(sub[1, :])
    axcb.axis("off")

    Dz = D.copy()
    np.fill_diagonal(Dz, 0.0)
    Z = linkage(squareform((Dz + Dz.T) / 2, checks=False), method="average")
    dn = dendrogram(Z, orientation="left", no_plot=True)
    order = [int(i) for i in dn["leaves"]][::-1]          # top-to-bottom, as imshow draws
    dendrogram(Z, orientation="left", ax=axd, color_threshold=0,
               above_threshold_color="0.45", link_color_func=lambda k: "0.45")
    axd.invert_yaxis()
    axd.set_xticks([])
    axd.set_yticks([])
    for s in axd.spines.values():
        s.set_visible(False)
    axd.text(0.0, 1.012, "average\nlinkage", transform=axd.transAxes, fontsize=5.6,
             color="0.4", va="bottom", ha="left", linespacing=1.2)

    n = len(species)
    lower = np.full((n, n), np.nan)
    upper = np.full((n, n), np.nan)
    for r in range(n):
        for c in range(n):
            i, j = order[r], order[c]
            if r > c:
                lower[r, c] = gap[i, j]
            elif r < c:
                upper[r, c] = share[i, j]
    keep = np.zeros((n, n), bool)
    keep[np.triu_indices(n, 1)] = True
    gmax = float(np.nanpercentile(lower, 95))
    imL = axb.imshow(np.ma.masked_invalid(lower), cmap="RdYlBu", vmin=0.8,
                     vmax=max(gmax, 1.3), interpolation="nearest")
    undefined = np.ma.masked_array(np.ones((n, n)), mask=~(keep & np.isnan(upper)))
    axb.imshow(undefined, cmap=matplotlib.colors.ListedColormap(["0.86"]), vmin=0, vmax=1,
               interpolation="nearest")
    imU = axb.imshow(np.ma.masked_invalid(np.where(keep, upper, np.nan)),
                     cmap="Purples", vmin=0, vmax=1, interpolation="nearest")
    for r in range(n):
        axb.add_patch(plt.Rectangle((r - .5, r - .5), 1, 1, fc="white", ec="0.75", lw=0.3))
    axb.set_xticks(range(n))
    axb.set_yticks(range(n))
    axb.set_xticklabels([species[i] for i in order], rotation=90, fontsize=5.0)
    axb.set_yticklabels([species[i] for i in order], fontsize=5.0)
    axb.tick_params(length=1.2, pad=1.0)
    for s in axb.spines.values():
        s.set_visible(False)
    axb.set_title("b  The gap between every pair", loc="left", x=-0.135)

    pos = axcb.get_position()
    cax1 = fig.add_axes([pos.x0 + 0.004, pos.y0 + 0.0035, pos.width * 0.42, 0.0068])
    cb1 = fig.colorbar(imL, cax=cax1, orientation="horizontal", extend="max")
    cb1.set_label("gap (lower triangle): how many own-\nspreads apart the pair sits (| marks 1.0)",
                  fontsize=5.7, labelpad=1.4)
    cb1.ax.tick_params(labelsize=5.3, length=1.5, pad=0.8)
    cb1.ax.axvline(1.0, color="k", lw=1.0)
    cax2 = fig.add_axes([pos.x0 + pos.width * 0.575, pos.y0 + 0.0035, pos.width * 0.42, 0.0068])
    cb2 = fig.colorbar(imU, cax=cax2, orientation="horizontal")
    cb2.set_label("graph support (upper triangle): share\nof shared asserted characters that\nseparate the pair \u00b7 grey: none asserted", fontsize=5.7, labelpad=1.4)
    cb2.ax.tick_params(labelsize=5.3, length=1.5, pad=0.8)

    # ---- panel c -----------------------------------------------------------
    axc = fig.add_subplot(gs[1, :])
    tidy(axc)
    cc = chars[(chars["n_species_asserting"] >= a.min_species_asserting)
               & chars["pair_separation"].notna() & chars["holdout_retention"].notna()].copy()
    cols = [fam_colour(f) for f in cc["family"]]
    axc.scatter(cc["pair_separation"], cc["holdout_retention"],
                s=6 + 4.4 * cc["n_species_asserting"], c=cols, alpha=0.85,
                edgecolor="white", linewidth=0.5, zorder=3)
    exp_mean = float(cc["expected_retention"].mean())
    axc.axhline(exp_mean, color="0.2", lw=1.0, ls="--", zorder=2)
    axc.text(0.995, exp_mean + 0.015,
             f"retention expected from sample size alone, mean of 1\u22122/(n+1) = {exp_mean:.2f}",
             transform=axc.get_yaxis_transform(), ha="right", va="bottom",
             fontsize=6.2, color="0.2")
    rho, pval = spearmanr(cc["pair_separation"], cc["holdout_retention"])
    axc.set_xlabel("pair separation: share of asserting species pairs whose stated ranges do not overlap")
    axc.set_ylabel("hold-out retention: share of specimens\ninside their own species' range,\nre-derived without them")
    axc.set_title("c  What the graph's characters are worth", loc="left")
    axc.set_xlim(-0.02, 1.02)
    axc.set_ylim(0.0, 1.0)
    axc.text(0.988, 0.965, f"Spearman \u03c1 = {rho:.2f} (p = {pval:.0e}), n = {len(cc)} characters",
             transform=axc.transAxes, fontsize=6.4, color="0.2", ha="right", va="top")

    axc.text(0.012, 0.225,
             "Hold-out retention is an identity, not a measurement:\n"
             "the two extreme specimens of any series fall outside\n"
             "the range re-derived without them, so retention =\n"
             f"1\u22122/(n+1) exactly \u2014 for all {len(chars)} asserted characters.",
             transform=axc.transAxes, va="top", ha="left", fontsize=6.0, color="0.2",
             linespacing=1.45)

    fams = [f for f in FAMILY_COLOUR if (cc["family"] == f).any() or (keydf["family"] == f).any()]
    handles = [Patch(fc=fam_colour(f), ec="white", label=FAMILY_LABEL.get(f, f)) for f in fams]
    handles += [Line2D([], [], ls="none", marker="o", mfc="0.72", mec="white",
                       ms=np.sqrt(6 + 4.4 * k) * 0.82, label=f"asserted by {k} species")
                for k in (10, 20)]
    leg = axc.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.005), ncol=3,
                     fontsize=6.1, frameon=True, framealpha=0.95, edgecolor="0.85",
                     handlelength=1.1, handletextpad=0.5, columnspacing=1.0, labelspacing=0.34,
                     title="character family (panels c and d); symbol area = species asserting it",
                     title_fontsize=6.1)
    leg.get_frame().set_linewidth(0.5)

    lab_idx = list(cc.sort_values("pair_separation", ascending=False).index[:5]) + \
              list(cc.sort_values("holdout_retention").index[:5])
    lab_idx = list(dict.fromkeys(lab_idx))
    lx = cc.loc[lab_idx, "pair_separation"].values
    ly = cc.loc[lab_idx, "holdout_retention"].values
    lt = [(str(t)[:33] + "\u2026") if len(str(t)) > 34 else str(t)
          for t in cc.loc[lab_idx, "label"].values]
    repel_labels(axc, np.column_stack([lx, ly]), lt, fontsize=6.0, radius=0.26,
                 iters=1200, pad=0.011,
                 avoid=[(0.0, 0.70, 0.66, 0.31), (0.60, 0.92, 0.40, 0.09),
                        (0.40, exp_mean, 0.60, 0.050),    # ylim is (0, 1): y == axes fraction
                        (0.0, 0.005, 0.365, 0.235)],
                 points=np.column_stack([cc["pair_separation"].values,
                                         cc["holdout_retention"].values]),
                 point_pad=0.009)

    # ---- panel d -----------------------------------------------------------
    axk = fig.add_subplot(gs[2, :])
    tidy(axk)
    kk = keydf[keydf["n_decisions"] >= a.min_decisions].sort_values("routing_support")
    kpos = axk.get_position()
    axk.set_position([0.275, kpos.y0, 0.975 - 0.275, kpos.height])
    y = np.arange(len(kk))
    axk.barh(y, kk["routing_support"], height=0.66,
             color=[fam_colour(f) for f in kk["family"]], edgecolor="white", linewidth=0.5,
             zorder=3)
    axk.errorbar(kk["routing_support"], y,
                 xerr=[kk["routing_support"] - kk["ci_low"], kk["ci_high"] - kk["routing_support"]],
                 fmt="none", ecolor="0.2", elinewidth=0.8, capsize=1.8, zorder=4)
    axk.axvline(0.5, color="#b00020", lw=1.0, ls="--", zorder=5)
    axk.text(0.505, len(kk) - 0.10, "chance for a two-way step", color="#b00020",
             fontsize=6.2, va="center", ha="left")
    axk.set_yticks(y)
    axk.set_yticklabels([(str(t)[:47] + "\u2026") if len(str(t)) > 48 else str(t)
                         for t in kk["label"]], fontsize=6.4)
    axk.set_ylim(-0.70, len(kk) + 0.15)
    axk.set_xlim(0, 1.13)
    axk.set_xticks(np.arange(0, 1.01, 0.2))
    axk.set_xlabel("hold-out routing support: share of decisions on that character that sent the "
                   "specimen to the side holding its true species")
    axk.set_title("d  Support for the characters the key uses", loc="left", x=-0.271)
    for yi, (hi, nd) in enumerate(zip(kk["ci_high"], kk["n_decisions"])):
        axk.text(min(hi + 0.012, 1.02), yi, f"n = {nd}", va="center", fontsize=6.1, color="0.3")

    # ---- note --------------------------------------------------------------
    fig.text(0.085, 0.056,
             "Pair separation \u2014 among the species whose treatment asserts a character, the share "
             "of species pairs whose stated ranges do not overlap. Hold-out retention \u2014 the share of\n"
             "specimens whose value still falls inside their own species' range once that range is "
             "re-derived without them; it equals 1\u22122/(n+1) exactly, because two specimens of any\n"
             "series are its extremes, so the characters that separate the most pairs on paper are "
             "simply those asserted from the shortest series. Routing support \u2014 the share of\n"
             "leave-one-out key decisions on a character that sent the specimen towards its own species. "
             "Every value that involves a specimen is a hold-out: the specimen is removed\n"
             "from the range, the spread or the key it is tested against.",
             fontsize=6.3, color="0.2", va="top", ha="left", linespacing=1.55)

    fig.savefig(out / "fig_graph_support.png", dpi=a.dpi)
    fig.savefig(out / "fig_graph_support.pdf")
    plt.close(fig)

    # ---- summary -----------------------------------------------------------
    iu = np.triu_indices(n, 1)
    offgap = gap[iu]
    nn_gap = []
    for i in range(n):
        j = int(np.argmin(Dm[i]))
        nn_gap.append(float(gap[i, j]))
    closest = pairs.head(10)[["species_a", "species_b", "distance", "gap",
                              "shared_characters", "separating_characters",
                              "separating_share"]].to_dict("records")
    summary = {
        "version": VERSION,
        "species": n, "specimens": int(len(ref.raw)),
        "character_sets": set_names,
        "species_with_asserted_ranges": len(graph),
        "species_without_asserted_ranges": no_ranges,
        "median_ranges_per_species": float(np.median(ranges_per_sp)),
        "distinct_asserted_characters": int(len(asserters)),
        "mds_stress1": round(stress, 4),
        "mds_variance_2_axes": round(var_expl, 4),
        "key_walker_anchor": anchor,
        "gap": {
            "median_all_pairs": round(float(np.nanmedian(offgap)), 4),
            "median_nearest_neighbour": round(float(np.median(nn_gap)), 4),
            "pairs_below_1_5": int((offgap < 1.5).sum()),
            "pairs_below_1_0": int((offgap < 1.0).sum()),
            "pairs_total": int(len(offgap)),
            "min": round(float(np.nanmin(offgap)), 4),
            "max": round(float(np.nanmax(offgap)), 4)},
        "graph_characters": {
            "n": int(len(chars)),
            "n_in_panel_c": int(len(cc)),
            "median_pair_separation": round(float(chars["pair_separation"].median()), 4),
            "median_holdout_retention": round(float(chars["holdout_retention"].median()), 4),
            "median_expected_retention": round(float(chars["expected_retention"].median()), 4),
            "median_pair_separation_panel_c": round(float(cc["pair_separation"].median()), 4),
            "median_holdout_retention_panel_c": round(float(cc["holdout_retention"].median()), 4),
            "mean_expected_retention_panel_c": round(exp_mean, 4),
            "spearman_separation_vs_retention": round(float(rho), 4),
            "spearman_p": float(pval),
            "characters_where_retention_equals_expected": int(
                (chars["holdout_retention"].sub(chars["expected_retention"]).abs() < 1e-9).sum()),
            "spearman_retention_vs_mean_series_size": round(float(spearmanr(
                cc["holdout_retention"],
                cc["n_specimens_tested"] / cc["n_species_asserting"]).statistic), 4),
            "spearman_separation_vs_mean_series_size": round(float(spearmanr(
                cc["pair_separation"],
                cc["n_specimens_tested"] / cc["n_species_asserting"]).statistic), 4),
            "retention_note": ("hold-out retention equals 1-2/(n+1) exactly: in a series of n+1 "
                               "specimens the two extremes fall outside the range re-derived "
                               "without them, so retention reports series length, not character "
                               "quality")},
        "pair_support": {
            "median_shared_characters": float(np.nanmedian(pairs["shared_characters"])),
            "median_separating_share": float(np.nanmedian(pairs["separating_share"])),
            "pairs_with_ranges_both": int(pairs["separating_share"].notna().sum())},
        "key_characters": {
            "n": int(len(keydf)), "n_in_panel_d": int(len(kk)),
            "median_routing_support_all": round(float(keydf["routing_support"].median()), 4),
            "median_routing_support_panel_d": round(float(kk["routing_support"].median()), 4),
            "total_decisions": int(keydf["n_decisions"].sum()),
            "best_panel_d": kk.tail(3).iloc[::-1][
                ["label", "family", "routing_support", "ci_low", "ci_high",
                 "n_decisions"]].to_dict("records"),
            "worst_panel_d": kk.head(3)[
                ["label", "family", "routing_support", "ci_low", "ci_high",
                 "n_decisions"]].to_dict("records")},
        "ten_closest_pairs": closest,
    }
    (out / "graph_support_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "ten_closest_pairs"}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
