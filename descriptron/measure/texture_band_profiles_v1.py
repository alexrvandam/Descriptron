#!/usr/bin/env python3
"""
texture_band_profiles_v1.py  (27 Oct 2026)
==========================================
Anterior -> posterior texture BAND PROFILES on a homologous (TPS) grid, tested against groups.

Input is a per-cell class-share array from a homology grid (specimens x grid rows x grid columns x texture classes),
e.g. the cell_class_share_<part>.npy written by the ant-vocabulary TPS grid step, or any per-cell class probabilities
on a grid whose row 0 is the anterior end. Each grid row becomes one band; a band's profile is the mean class share of
its cells (cells outside the part are NaN and ignored). Classes can be merged for display and testing.

Two subcommands:
  bands  cell shares -> bands.npz (X = specimens x bands x classes, ids, class names)
         --cells cell_class_share_elytron.npy --ids ids_elytron.txt --rows 10 --cols 8 --pad 3
         --classes smooth,superficial,fine_rugulose,longitudinal_rugose,longitudinal_crossmeshes,reticulate_rugose
         --merge "smooth/superficial=smooth+superficial;fine-rugulose=fine_rugulose;longitudinally rugose=longitudinal_rugose+longitudinal_crossmeshes;reticulate-rugose=reticulate_rugose"
         --out bands_elytron.npz
  test   bands.npz (+ more parts, concatenated) x a labels table -> per-group tests and figures
         --bands bands_elytron.npz[,bands_pronotum.npz] --labels labels.tsv --id_col file --group_col el_striae
         [--map "impressed=impressed;rows=rows / distinct;distinct=rows / distinct;weak=weak / absent;absent=weak / absent"]
         [--use_part 0] --out OUTDIR
     Distance = mean Jensen-Shannon distance between band profiles. Tests: PERMANOVA R^2 with label permutations, and
     nearest-neighbour agreement against a label-shuffle null (95th percentile). With one specimen per species the group
     can be a published character state (the species' own words); with several specimens per species it can be the species.
     Writes band_tests.tsv, BAND_PROFILES_by_group.png (mean stacked profile per group, anterior -> posterior) and
     DENDROGRAM_by_group.png (UPGMA on the same distance, a balanced subsample, leaves coloured by group).

First used on 342 Trigonopterus species from Riedel's revisions (weevil_texture_v1/riedel_bands_v1/). Requires numpy,
scipy, matplotlib.
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy.spatial.distance import jensenshannon, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

PAL = ['#e6e6e6', '#ffc800', '#3c5aff', '#e61e1e', '#8c3cc8', '#aaaaaa', '#2ca02c', '#ff7f0e']
GROUP_COL = ['#000000', '#2ca02c', '#b07aa1', '#d62728', '#1f77b4', '#8c564b', '#e377c2', '#7f7f7f']


def cmd_bands(a):
    S = np.load(a.cells)
    ids = [l.rstrip('\n') for l in open(a.ids) if l.strip()]
    if len(ids) != len(S): sys.exit(f'{len(S)} specimens in --cells but {len(ids)} ids')
    NR, NC = a.rows + 2 * a.pad, a.cols + 2 * a.pad
    if S.shape[1] != NR * NC: sys.exit(f'{S.shape[1]} cells per specimen, expected ({a.rows}+2*{a.pad}) x ({a.cols}+2*{a.pad}) = {NR * NC}')
    S = S.reshape(len(S), NR, NC, -1)[:, a.pad:a.pad + a.rows, a.pad:a.pad + a.cols]
    B = np.nanmean(S, axis=2)                                            # specimens x bands x classes
    cls = a.classes.split(',') if a.classes else [f'class{k}' for k in range(B.shape[-1])]
    if a.merge:
        names, idx = [], []
        for part in a.merge.split(';'):
            n, members = part.split('='); names.append(n.strip()); idx.append([cls.index(m.strip()) for m in members.split('+')])
        B = np.stack([B[..., ix].sum(-1) for ix in idx], -1); cls = names
    B = B / np.clip(B.sum(-1, keepdims=True), 1e-9, None)
    np.savez(a.out, X=B, ids=np.array(ids), classes=np.array(cls))
    print(f'{a.out}: {B.shape[0]} specimens x {B.shape[1]} bands x {B.shape[2]} classes')


def jsd_matrix(X):
    n = len(X); D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = np.nanmean([jensenshannon(X[i, b] + 1e-9, X[j, b] + 1e-9) for b in range(X.shape[1])])
    return D


def permanova(D, g, rng, nperm=999):
    g = np.asarray(g); n = len(g); D2 = D ** 2; sst = D2[np.triu_indices(n, 1)].sum() / n
    def ssw(gg): return sum(D2[np.ix_(gg == k, gg == k)][np.triu_indices((gg == k).sum(), 1)].sum() / (gg == k).sum() for k in np.unique(gg))
    r2 = 1 - ssw(g) / sst; perm = np.array([1 - ssw(rng.permutation(g)) / sst for _ in range(nperm)])
    return r2, (1 + (perm >= r2).sum()) / (nperm + 1)


def nn_agreement(D, g, rng, nperm=300):
    g = np.asarray(g); nn = (D + np.diag(np.full(len(g), np.inf))).argmin(1)
    null = [(p[nn] == p).mean() for p in (rng.permutation(g) for _ in range(nperm))]
    return (g[nn] == g).mean(), float(np.percentile(null, 95))


def cmd_test(a):
    rng = np.random.default_rng(a.seed); os.makedirs(a.out, exist_ok=True)
    parts = [np.load(f, allow_pickle=True) for f in a.bands.split(',')]
    common = sorted(set.intersection(*[set(map(str, p['ids'])) for p in parts]))
    X = np.concatenate([p['X'][[list(map(str, p['ids'])).index(i) for i in common]] for p in parts], 1)
    cls = [str(c) for c in parts[0]['classes']]; nb = [p['X'].shape[1] for p in parts]
    if a.use_part is not None:
        s0 = sum(nb[:a.use_part]); X = X[:, s0:s0 + nb[a.use_part]]
    mp = dict(kv.split('=', 1) for kv in a.map.split(';')) if a.map else None
    lab = {r[a.id_col]: r[a.group_col] for r in csv.DictReader(open(a.labels), delimiter='\t' if a.labels.endswith('.tsv') else ',')}
    keep, g = [], []
    for i, s in enumerate(common):
        v = lab.get(s)
        if v in (None, '', 'NA'): continue
        v = mp.get(v) if mp else v
        if v: keep.append(i); g.append(v)
    X = X[keep]; ids = [common[i] for i in keep]
    ok = ~np.isnan(X).any((1, 2)); X, ids, g = X[ok], [x for x, k in zip(ids, ok) if k], [x for x, k in zip(g, ok) if k]
    D = jsd_matrix(X); r2, p = permanova(D, g, rng, a.nperm); nn, null95 = nn_agreement(D, g, rng)
    groups = sorted(set(g), key=lambda s: -g.count(s))
    row = dict(group_col=a.group_col, n=len(g), groups=json.dumps({k: g.count(k) for k in groups}), bands=X.shape[1],
               permanova_R2=round(float(r2), 3), p=round(float(p), 3), nn_same_group=round(float(nn), 3), nn_null95=round(null95, 3))
    with open(os.path.join(a.out, 'band_tests.tsv'), 'w') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row), delimiter='\t'); w.writeheader(); w.writerow(row)
    print(row)
    pal = PAL[:len(cls)]; xb = np.arange(X.shape[1])
    fig, ax = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.6), sharey=True, squeeze=False); ax = ax[0]
    for a_, s in zip(ax, groups):
        M = X[[k for k, gg in enumerate(g) if gg == s]].mean(0)
        a_.stackplot(xb, M.T, colors=pal, labels=cls); a_.set_title(f'{a.title_prefix}{s} (n={g.count(s)})', fontsize=13)
        a_.set_xticks([0, xb[-1]]); a_.set_xticklabels(['ant.', 'post.']); a_.set_xlim(0, xb[-1]); a_.set_ylim(0, 1)
    ax[0].set_ylabel('share of band'); ax[0].legend(loc='lower left', fontsize=9)
    fig.suptitle(a.title or f'{X.shape[1]} bands anterior -> posterior on the homologous grid, by {a.group_col}', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'BAND_PROFILES_by_group.png'), dpi=150); plt.close(fig)
    sub = []
    for s in groups:
        ii = [k for k, gg in enumerate(g) if gg == s]; sub += list(rng.choice(ii, min(a.per_group, len(ii)), replace=False))
    Z = linkage(squareform(D[np.ix_(sub, sub)], checks=False), 'average'); gc = {s: GROUP_COL[k % len(GROUP_COL)] for k, s in enumerate(groups)}
    fig = plt.figure(figsize=(14, max(6, 0.26 * len(sub) + 1.5))); a0 = fig.add_axes([0.02, 0.07, 0.2, 0.86])
    dn = dendrogram(Z, orientation='left', no_labels=True, ax=a0, color_threshold=0, above_threshold_color='#1f77b4'); a0.axis('off')
    leaves = dn['leaves'][::-1]; a1 = fig.add_axes([0.24, 0.07, 0.12, 0.86]); a2 = fig.add_axes([0.40, 0.07, 0.5, 0.86]); n = len(leaves)
    for r, li in enumerate(leaves):
        k = sub[li]; y = n - 1 - r
        a1.add_patch(plt.Rectangle((0, y), 1, 0.9, color=gc[g[k]])); a1.text(1.05, y + 0.45, ids[k] if not a.label_fn else a.label_fn(ids[k]), va='center', fontsize=8)
        cum = np.zeros(X.shape[1])
        for c in range(len(cls)):
            for b in range(X.shape[1]): a2.add_patch(plt.Rectangle((b, y + cum[b] * 0.9), 1, X[k, b, c] * 0.9, color=pal[c]))
            cum += X[k, :, c]
    a1.set_xlim(0, 3.2); a1.set_ylim(0, n); a1.axis('off'); a2.set_xlim(0, X.shape[1]); a2.set_ylim(0, n); a2.set_yticks([])
    a2.set_xticks([0.5, X.shape[1] - 0.5]); a2.set_xticklabels(['ant.', 'post.']); a2.set_title(f'{X.shape[1]} bands (anterior -> posterior)')
    hs = [plt.Rectangle((0, 0), 1, 1, color=gc[s]) for s in groups] + [plt.Rectangle((0, 0), 1, 1, color=c) for c in pal]
    fig.legend(hs, [f'{a.title_prefix}{s}' for s in groups] + cls, loc='lower center', ncol=4, fontsize=9)
    fig.suptitle(f'Clustered by band profile (UPGMA, Jensen-Shannon); colour bar = {a.group_col}', fontsize=12)
    fig.savefig(os.path.join(a.out, 'DENDROGRAM_by_group.png'), dpi=150); plt.close(fig)
    print('wrote', a.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    b = sp.add_parser('bands'); b.add_argument('--cells', required=True); b.add_argument('--ids', required=True)
    b.add_argument('--rows', type=int, required=True); b.add_argument('--cols', type=int, required=True); b.add_argument('--pad', type=int, default=0)
    b.add_argument('--classes', default=''); b.add_argument('--merge', default=''); b.add_argument('--out', required=True)
    t = sp.add_parser('test'); t.add_argument('--bands', required=True); t.add_argument('--labels', required=True)
    t.add_argument('--id_col', required=True); t.add_argument('--group_col', required=True); t.add_argument('--map', default='')
    t.add_argument('--use_part', type=int, default=None, help='test only the n-th bands file (0-based); default: all concatenated')
    t.add_argument('--per_group', type=int, default=14, help='dendrogram: specimens drawn per group'); t.add_argument('--nperm', type=int, default=999)
    t.add_argument('--seed', type=int, default=0); t.add_argument('--title', default=''); t.add_argument('--title_prefix', default='')
    t.add_argument('--out', required=True)
    a = ap.parse_args(); a.label_fn = None
    cmd_bands(a) if a.cmd == 'bands' else cmd_test(a)
