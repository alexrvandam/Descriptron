#!/usr/bin/env python3
"""
descriptron_combine_characters_v1.py - which characters, alone and together, tell the species apart?
=====================================================================================================
(3 Nov 2026.) Each GM run of the semilandmark GPA (V42) analyses ONE structure. This step puts several of them - and any other
per-specimen numbers, e.g. BioRAG-Probe texture scores or measurements - side by side for the same specimens and asks which
blocks identify the species, alone and in every combination, and how strongly the blocks are correlated.

Blocks (--block NAME=PATH[:KIND], repeat):
  shape   a V42 shape_traits_phylo_<category>.csv (filename, PC1..PCn, centroid_size): the first --pcs_per_block shape PCs
          of that structure's own GPA. Centroid size is NOT used (it is pixels unless the images were scaled).
  table   any CSV/TSV with a specimen column (found by --id_regex in the first column that matches) and numeric columns,
          e.g. probe scores. Columns are z-scored; a table with more than --pcs_per_block columns is reduced by PCA.
Specimens are matched across blocks by --id_regex (default CASENT number); the group (species) of each specimen comes from
--group_labels (CSV filename,group_label - the same file V42 takes) or from --group_regex on the file name.

Tests (the same test as V42's CVA, so the numbers are comparable):
  1  every combination of blocks: LDA, leave-one-out species ID over the specimens of groups with n >= 2, chance from
     --permutations label shuffles (mean, 95th percentile, p = (k+1)/(N+1)), proportional chance. All combinations use the
     SAME specimens (those present in every block), so the rows can be compared.
  2  the best combination, honestly: picking the best of many combinations and reporting its accuracy is optimistic (the
     choice was made on the same specimens). Nested leave-one-out: for each held-out specimen the combination is chosen by
     an inner leave-one-out on the others only, then used to predict it. Reports the nested accuracy and how often each
     combination was chosen; --nested_permutations shuffles give its chance level (slow; 0 = off).
  3  integration: two-block PLS between every pair of blocks (r-PLS of the first singular pair, --permutations shuffles of
     the specimens in one block): correlated blocks carry overlapping information.
Outputs (--out_dir): combinations.tsv, nested_selection.tsv, integration.tsv, summary.txt, combinations.png

    python descriptron_combine_characters_v1.py \\
        --block mesosoma_lateral=M1/out/mesosoma/shape_traits_phylo_mesosoma.csv \\
        --block petiole=MT1/out/petiole/shape_traits_phylo_petiole.csv \\
        --block texture=probe/specimen_scores_primary.tsv:table \\
        --group_labels group_labels.csv --out_dir combined/ --permutations 999
"""
import argparse
import csv
import itertools
import os
import re
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_predict


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--block', action='append', required=True, help='NAME=PATH[:shape|table] (repeat; default kind shape)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--group_labels', default='', help='CSV filename,group_label (as for V42); matched on the specimen id')
    ap.add_argument('--group_regex', default=r'^[^_]+', help='group from the file name when --group_labels is not given (default: text before the first underscore)')
    ap.add_argument('--id_regex', default=r'CASENT\d+', help='specimen id inside file names / table rows')
    ap.add_argument('--pcs_per_block', type=int, default=5, help='shape PCs per block (fixed in advance; default 5)')
    ap.add_argument('--max_blocks', type=int, default=0, help='largest combination tested (0 = all blocks)')
    ap.add_argument('--permutations', type=int, default=999)
    ap.add_argument('--nested_permutations', type=int, default=0, help='chance level of the nested best-combination accuracy')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def _read_rows(path):
    with open(path, newline='') as f:
        sample = f.read(4096); f.seek(0)
        return list(csv.DictReader(f, delimiter='\t' if sample.count('\t') > sample.count(',') else ','))


def load_block(path, kind, id_re, k):
    """{specimen_id: vector}, column names, the file name seen for each specimen."""
    rows = _read_rows(path); out, seen = {}, {}
    if kind == 'shape':
        cols = [c for c in rows[0] if re.fullmatch(r'PC\d+', c)]
        cols = sorted(cols, key=lambda c: int(c[2:]))[:k]
        for r in rows:
            m = id_re.search(r['filename'])
            if m:
                out[m.group()] = np.array([float(r[c]) for c in cols]); seen[m.group()] = r['filename']
        return out, cols, seen
    # table: the id column is the first whose values match id_regex; numeric columns are the rest that parse as numbers
    idc = next((c for c in rows[0] if all(id_re.search(str(r[c] or '')) for r in rows[:20])), None)
    if idc is None:
        sys.exit(f'{path}: no column matches --id_regex {id_re.pattern}')
    num = []
    for c in rows[0]:
        if c == idc: continue
        try: [float(r[c]) for r in rows]; num.append(c)
        except (TypeError, ValueError): pass
    acc = {}
    for r in rows:
        acc.setdefault(id_re.search(r[idc]).group(), []).append([float(r[c]) for c in num])
    X = {s: np.mean(v, 0) for s, v in acc.items()}            # several rows per specimen (e.g. images) -> mean
    M = np.array(list(X.values())); mu, sd = M.mean(0), M.std(0); sd[sd == 0] = 1
    X = {s: (v - mu) / sd for s, v in X.items()}
    if len(num) > k:
        p = PCA(n_components=k).fit(np.array(list(X.values())))
        X = {s: p.transform(v[None])[0] for s, v in X.items()}; num = [f'PC{i+1}' for i in range(k)]
    return X, num, {s: s for s in X}


def loo_acc(X, y):
    return np.mean(cross_val_predict(LinearDiscriminantAnalysis(), X, y, cv=LeaveOneOut()) == y) * 100


def rpls(A, B):
    A = A - A.mean(0); B = B - B.mean(0); u, s, vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return float(np.corrcoef(A @ u[:, 0], B @ vt[0])[0, 1])


def nested_best(blocks, combos, y, idx_of):
    """Outer leave-one-out; the combination is chosen on the other specimens only (inner leave-one-out)."""
    n = len(y); correct, chosen = 0, []
    for i in range(n):
        tr = np.r_[0:i, i + 1:n]; best, best_a = None, -1
        for c in combos:
            X = np.hstack([blocks[b] for b in c])
            a = loo_acc(X[tr], y[tr])
            if a > best_a: best, best_a = c, a
        X = np.hstack([blocks[b] for b in best])
        pred = LinearDiscriminantAnalysis().fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
        correct += pred == y[i]; chosen.append(best)
    return correct / n * 100, chosen


def main():
    a = parse_args(); os.makedirs(a.out_dir, exist_ok=True); id_re = re.compile(a.id_regex); rng = np.random.default_rng(a.seed)
    B, cols, names = {}, {}, {}
    for spec in a.block:
        name, rest = spec.split('=', 1); path, kind = rest, 'shape'
        if rest.rsplit(':', 1)[-1] in ('shape', 'table'):
            path, kind = rest.rsplit(':', 1)
        B[name], cols[name], seen = load_block(path, kind, id_re, a.pcs_per_block)
        for k_, v_ in seen.items(): names.setdefault(k_, v_)   # keep a real file name (it carries the group) over a bare id
        print(f'block {name} ({kind}): {len(B[name])} specimens, {len(cols[name])} variables  <- {path}')
    gl = {}
    if a.group_labels:
        for r in _read_rows(a.group_labels):
            m = id_re.search(r['filename'])
            if m: gl[m.group()] = r['group_label']
    grp_re = re.compile(a.group_regex)
    def group(s):
        if s in gl: return gl[s]
        m = grp_re.search(names.get(s, s)); return m.group() if m else None
    common = sorted(set.intersection(*[set(v) for v in B.values()]))
    groups = np.array([group(s) for s in common]); ok = np.array([g is not None for g in groups])
    cnt = {g: int((groups == g).sum()) for g in set(groups[ok])}
    keep = ok & np.array([cnt.get(g, 0) >= 2 for g in groups]); y = groups[keep]
    if len(set(y)) < 2:
        sys.exit('fewer than 2 groups with n >= 2: check --group_labels / --group_regex against the file names')
    Xb = {b: np.array([B[b][s] for s in common])[keep] for b in B}
    names_b = list(B); kmax = a.max_blocks or len(names_b)
    combos = [c for n in range(1, kmax + 1) for c in itertools.combinations(names_b, n)]
    props = np.unique(y, return_counts=True)[1] / len(y)
    head = [f'{len(common)} specimens present in every block; {keep.sum()} in {len(set(y))} groups with n >= 2 are classified '
            f'(groups with one specimen cannot be tested); {a.pcs_per_block} variables per block; proportional chance '
            f'{np.sum(props ** 2) * 100:.1f}%']
    print(head[0])
    # 1 every combination
    rows = []
    for c in combos:
        X = np.hstack([Xb[b] for b in c]); acc = loo_acc(X, y)
        perm = np.array([loo_acc(X, rng.permutation(y)) for _ in range(a.permutations)]) if a.permutations else np.array([np.nan])
        p = (np.sum(perm >= acc) + 1) / (len(perm) + 1) if a.permutations else np.nan
        rows.append({'blocks': ' + '.join(c), 'n_blocks': len(c), 'n_variables': X.shape[1], 'loo_accuracy_pct': round(acc, 1),
                     'perm_mean_pct': round(float(np.nanmean(perm)), 1), 'perm_95th_pct': round(float(np.nanpercentile(perm, 95)), 1),
                     'p': round(p, 4)})
        print(f"  {rows[-1]['blocks']:<70} {acc:5.1f}%  (chance {rows[-1]['perm_mean_pct']}%, 95th {rows[-1]['perm_95th_pct']}%, p {p:.4f})", flush=True)
    with open(os.path.join(a.out_dir, 'combinations.tsv'), 'w') as f:
        f.write('\t'.join(rows[0]) + '\n'); [f.write('\t'.join(str(r[k]) for k in r) + '\n') for r in rows]
    # 2 nested best combination
    nacc, chosen = nested_best(Xb, combos, y, None)
    ch = {}
    for c in chosen: ch[' + '.join(c)] = ch.get(' + '.join(c), 0) + 1
    nperm = []
    for _ in range(a.nested_permutations):
        nperm.append(nested_best(Xb, combos, rng.permutation(y), None)[0])
    nline = f'nested leave-one-out accuracy of "choose the best combination" = {nacc:.1f}%'
    if nperm:
        nperm = np.array(nperm); nline += (f' (chance over {len(nperm)} shuffles: mean {nperm.mean():.1f}%, 95th '
                                           f'{np.percentile(nperm, 95):.1f}%, p {(np.sum(nperm >= nacc) + 1) / (len(nperm) + 1):.4f})')
    print(nline)
    with open(os.path.join(a.out_dir, 'nested_selection.tsv'), 'w') as f:
        f.write('combination\ttimes_chosen_of_' + str(len(y)) + '\n')
        for k, v in sorted(ch.items(), key=lambda t: -t[1]): f.write(f'{k}\t{v}\n')
    # 3 integration
    irows = []
    for b1, b2 in itertools.combinations(names_b, 2):
        sp = sorted(set(B[b1]) & set(B[b2])); A1 = np.array([B[b1][s] for s in sp]); A2 = np.array([B[b2][s] for s in sp])
        r = rpls(A1, A2); pr = np.array([rpls(A1, A2[rng.permutation(len(sp))]) for _ in range(a.permutations)]) if a.permutations else np.array([np.nan])
        irows.append((b1, b2, len(sp), r, float(np.nanmean(pr)), (np.sum(pr >= r) + 1) / (len(pr) + 1) if a.permutations else np.nan))
    with open(os.path.join(a.out_dir, 'integration.tsv'), 'w') as f:
        f.write('block_1\tblock_2\tn_specimens\tr_PLS\tperm_mean_r\tp\n')
        for t in irows: f.write(f'{t[0]}\t{t[1]}\t{t[2]}\t{t[3]:.3f}\t{t[4]:.3f}\t{t[5]:.4f}\n')
    best = max(rows, key=lambda r: r['loo_accuracy_pct'])
    summ = head + ['', 'single blocks: ' + '; '.join(f"{r['blocks']} {r['loo_accuracy_pct']}% (p {r['p']})" for r in rows if r['n_blocks'] == 1),
                   f"all blocks: {rows[-1]['loo_accuracy_pct']}% (p {rows[-1]['p']})" if kmax == len(names_b) else '',
                   f"highest single combination: {best['blocks']} {best['loo_accuracy_pct']}% - chosen after looking at all "
                   f"{len(rows)}, so optimistic; the honest figure is the nested one:", nline,
                   'most often chosen in the nested selection: ' + '; '.join(f'{k} ({v}/{len(y)})' for k, v in sorted(ch.items(), key=lambda t: -t[1])[:3]),
                   '', 'integration (two-block PLS): ' + '; '.join(f'{t[0]} x {t[1]} r {t[3]:.2f} p {t[5]:.3f}' for t in irows)]
    open(os.path.join(a.out_dir, 'summary.txt'), 'w').write('\n'.join(s for s in summ if s is not None) + '\n')
    # figure: one bar per combination, chance band behind it
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        R = sorted(rows, key=lambda r: r['loo_accuracy_pct']); h = 0.32 * len(R) + 1.6
        fig, ax = plt.subplots(figsize=(8.5, h)); yy = np.arange(len(R))
        ax.barh(yy, [r['perm_95th_pct'] for r in R], height=0.72, color='#d9d9d9', label='95th percentile of shuffled labels')
        ax.barh(yy, [r['loo_accuracy_pct'] for r in R], height=0.36, color='#2a6f97', label='leave-one-out species ID')
        for i, r in enumerate(R):
            ax.text(r['loo_accuracy_pct'] + 0.8, i, f"{r['loo_accuracy_pct']:.0f}%", va='center', fontsize=8, color='#333333')
        ax.axvline(nacc, color='#9b2226', lw=1.5, ls='--', label=f'nested best-combination accuracy {nacc:.0f}%')
        ax.set_yticks(yy); ax.set_yticklabels([r['blocks'] for r in R], fontsize=8); ax.set_xlabel('% of specimens assigned to the right species')
        ax.set_xlim(0, 100); ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=7, loc='lower right', frameon=False)
        ax.set_title(f'Characters combined: {keep.sum()} specimens, {len(set(y))} species', fontsize=10)
        plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, 'combinations.png'), dpi=200); plt.close()
    except Exception as e:
        print(f'figure skipped: {e}')
    print(f'-> {a.out_dir}')


if __name__ == '__main__':
    main()
