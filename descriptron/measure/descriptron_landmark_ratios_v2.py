#!/usr/bin/env python3
"""
descriptron_landmark_ratios_v2.py - measurements and taxonomic ratios (indices) from landmarks and masks, with species tests
=========================================================================================================================
v2 (17 Nov 2026) = v1 + statistics: Benjamini-Hochberg q values over all variables (Kruskal-Wallis and character agreement); Dunn's
pairwise species tests (pairwise_dunn.tsv: which species pairs a ratio separates); PERMANOVA (position) with PERMDISP (spread) on the
leave-one-out variables; --size_measure allometry check (log-log common within-species slope, species test on size-adjusted values; only
with real sizes). v1 is unchanged.
(17 Nov 2026.) Taxonomic keys and descriptions use linear measurements and their ratios (e.g. ants: HL, HW, EL, CI = HW/HL x 100,
OI = EL/HW x 100). The GM programs (V42 outlines, landmark_gpa_V2) test shape; this step gives the numbers a taxonomist reports, defined
once in a config and applied to every specimen of a COCO file (Descriptron/GUI save), plus the tests that say whether they separate species.

Definitions (repeat --measure / --ratio, or put them in --config TSV with columns kind, name, definition, description):

  landmark terms (1-based landmark numbers, the order of the keypoint set):
    A-B            distance between landmarks A and B
    A-B|C-D        the A-B distance measured ALONG the axis C->D (projection), e.g. head length on the midline
    A-B^C-D        the A-B distance measured ACROSS (perpendicular to) the axis C-D, e.g. a width across the midline
    P>A-B          perpendicular distance of landmark P from the line through A and B
    P>A-B@R        the same, signed: positive when P lies on the same side of the line as landmark R
  mask terms (COCO category name; masks as polygons or RLE):
    cat:length  cat:width      extent of the mask along / across its own principal axis
    cat:length|A-B             extent of the mask ALONG the landmark axis A-B;  cat:width^A-B  extent ACROSS it
    cat:area  cat:feret        area (px^2) / maximum caliper diameter
  mean(t1;t2;...)              mean of several terms (e.g. left and right eye)
  ratio:  NUM/DEN  or  NUM/DEN*100   (NUM, DEN = names of measures defined before)

  Example (ant head, frontal view; landmark numbers of the Descriptron frontal template):
    --measure HL=9-10 --measure HW=11-12^9-10 --measure EL="mean(1-2|9-10;3-4|9-10)" --ratio CI=HW/HL*100 --ratio OI=EL/HW*100
  Ready-made: --config ratio_configs/ant_head_frontal_v1.tsv

Specimens and groups: one row per image; group (species) from --group_labels (CSV filename,group_label, as for V42) or --group_regex on
the file name. Measures are in px; with --px_per_mm (CSV file,px_per_mm) also in mm. Ratios are unit-free (fine for un-scaled renders).

Tests (same as V42 / descriptron_combine_characters_v1, so numbers are comparable):
  per measure and ratio: Kruskal-Wallis across the groups with n >= 2;
  all ratios together (or --loo_on measures|both): LDA, leave-one-out species ID over the specimens of groups with n >= 2, chance from
  --permutations label shuffles (mean, 95th percentile, p = (k+1)/(N+1));
  --character TSV (taxon, label) [--positive WORD]: agreement of each measure/ratio (species means) with a coded character, e.g. a published
  matrix: binary -> AUC of positive vs rest + Mann-Whitney p (two-sided); more states -> Kruskal-Wallis on species means.
Outputs (--out_dir): specimen_measurements.tsv, species_summary.tsv (n, mean, sd, min, max and a 'min-max (mean)' text per species),
tests.tsv, character_agreement.tsv, summary.txt, ratios_by_group.png.

    python descriptron_landmark_ratios_v1.py --json heads.json --config ratio_configs/ant_head_frontal_v1.tsv \\
        --group_labels group_labels.csv --out_dir ratios/ --permutations 999
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np


# ----------------------------------------------------------------------------------------------------------------- definitions
def parse_term(s):
    s = s.strip()
    m = re.fullmatch(r'mean\((.+)\)', s)
    if m:
        return ('mean', [parse_term(t) for t in m.group(1).split(';')])
    m = re.fullmatch(r'(\d+)>(\d+)-(\d+)(?:@(\d+))?', s)
    if m:
        return ('pline', int(m[1]), int(m[2]), int(m[3]), int(m[4]) if m[4] else None)
    m = re.fullmatch(r'(\d+)-(\d+)(?:([|^])(\d+)-(\d+))?', s)
    if m:
        return ('dist', int(m[1]), int(m[2]), m[3], int(m[4]) if m[4] else None, int(m[5]) if m[5] else None)
    m = re.fullmatch(r'([^:]+):(length|width|area|feret)(?:([|^])(\d+)-(\d+))?', s)
    if m:
        return ('mask', m[1].strip(), m[2], m[3], int(m[4]) if m[4] else None, int(m[5]) if m[5] else None)
    raise ValueError(f'cannot read the term "{s}" (see --help for the syntax)')


def term_landmarks(t):
    if t[0] == 'mean':
        return set().union(*[term_landmarks(x) for x in t[1]])
    if t[0] == 'pline':
        return {t[1], t[2], t[3]} | ({t[4]} if t[4] else set())
    if t[0] == 'dist':
        return {t[1], t[2]} | ({t[4], t[5]} if t[3] else set())
    return {t[4], t[5]} if t[3] else set()


def term_categories(t):
    if t[0] == 'mean':
        return set().union(*[term_categories(x) for x in t[1]])
    return {t[1]} if t[0] == 'mask' else set()


def parse_ratio(s):
    m = re.fullmatch(r'\s*([A-Za-z_][\w.]*)\s*/\s*([A-Za-z_][\w.]*)\s*(?:\*\s*([0-9.]+))?\s*', s)
    if not m:
        raise ValueError(f'cannot read the ratio "{s}" (NUM/DEN or NUM/DEN*100)')
    return m[1], m[2], float(m[3]) if m[3] else 1.0


def read_definitions(a):
    measures, ratios, desc = [], [], {}
    if a.config:
        with open(a.config) as fh:
            for r in csv.DictReader((l for l in fh if l.strip() and not l.startswith('#')), delimiter='\t'):
                kind, name, d = r['kind'].strip().lower(), r['name'].strip(), r['definition'].strip()
                desc[name] = (r.get('description') or '').strip()
                (measures if kind == 'measure' else ratios).append((name, d))
    for item, store in ((a.measure, measures), (a.ratio, ratios)):
        for x in item or []:
            name, d = x.split('=', 1)
            store.append((name.strip(), d.strip()))
    M = [(n, parse_term(d), d) for n, d in measures]
    names = [n for n, _, _ in M]
    R = []
    for n, d in ratios:
        num, den, k = parse_ratio(d)
        for x in (num, den):
            if x not in names:
                raise ValueError(f'ratio {n}: "{x}" is not a defined measure ({", ".join(names)})')
        R.append((n, num, den, k, d))
    if not M:
        raise ValueError('no measures defined (--measure, or --config)')
    return M, R, desc


# ---------------------------------------------------------------------------------------------------------------- geometry
def _unit(v):
    n = float(np.hypot(*v))
    return v / n if n > 0 else np.array([np.nan, np.nan])


def evaluate(t, K, masks):
    """K: (n, 2) landmark array (NaN = missing); masks: dict category -> (N, 2) pixel coordinates (x, y) or None."""
    kind = t[0]
    if kind == 'mean':
        v = [evaluate(x, K, masks) for x in t[1]]
        return float(np.mean(v)) if all(np.isfinite(v)) else np.nan

    def P(i):
        return K[i - 1] if 0 < i <= len(K) else np.array([np.nan, np.nan])

    if kind == 'pline':
        p, a, b, r = P(t[1]), P(t[2]), P(t[3]), (P(t[4]) if t[4] else None)
        u = _unit(b - a)
        d = float(u[0] * (p - a)[1] - u[1] * (p - a)[0])            # signed: cross(u, p - a)
        if r is None:
            return abs(d)
        side = float(u[0] * (r - a)[1] - u[1] * (r - a)[0])
        return d * np.sign(side) if side != 0 else np.nan
    if kind == 'dist':
        v = P(t[2]) - P(t[1])
        if not t[3]:
            return float(np.hypot(*v))
        u = _unit(P(t[5]) - P(t[4]))
        along = float(v @ u)
        return abs(along) if t[3] == '|' else float(np.hypot(*(v - along * u)))
    # mask terms
    cat, what, op, c, d = t[1], t[2], t[3], t[4], t[5]
    X = masks.get(cat)
    if X is None or len(X) < 3:
        return np.nan
    if what == 'area':
        return float(len(X))
    if what == 'feret':
        import cv2
        h = cv2.convexHull(X.astype(np.float32)).reshape(-1, 2)
        dd = np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1))
        return float(dd.max())
    if op:
        u = _unit(P(d) - P(c))
        if not np.all(np.isfinite(u)):
            return np.nan
    else:
        C = np.cov((X - X.mean(0)).T)
        w, V = np.linalg.eigh(C)
        u = V[:, np.argmax(w)]
    s_along, s_across = X @ u, X @ np.array([-u[1], u[0]])
    if (what == 'length' and op != '^') or (what == 'width' and op == '|'):
        return float(np.ptp(s_along) + 1)
    return float(np.ptp(s_across) + 1)


# ---------------------------------------------------------------------------------------------------------------- COCO input
def read_coco(path, keypoint_category, need_cats):
    d = json.load(open(path))
    cat = {c['id']: c['name'] for c in d.get('categories', [])}
    ims = {}
    for im in d['images']:
        ims[im['id']] = im
        ims.setdefault(str(im['id']), im)
    by_name = {os.path.splitext(os.path.basename(str(im['file_name'])))[0]: im for im in d['images']}
    kps, segs = {}, {}
    for a in d['annotations']:
        im = ims.get(a['image_id']) or ims.get(str(a['image_id'])) or by_name.get(os.path.basename(str(a['image_id'])))
        if im is None:
            continue
        f = os.path.basename(str(im['file_name']))
        name = cat.get(a['category_id'], str(a['category_id']))
        if a.get('keypoints') and not a.get('source_line_category'):
            if keypoint_category and name != keypoint_category:
                continue
            if f not in kps:                                            # first keypoint set of an image = the landmarks
                k = np.array(a['keypoints'], float).reshape(-1, 3)
                xy = k[:, :2].copy()
                xy[k[:, 2] <= 0] = np.nan
                kps[f] = xy
        elif name in need_cats and a.get('segmentation'):
            segs.setdefault(f, {}).setdefault(name, []).append(a['segmentation'])
    return d['images'], kps, segs


def mask_pixels(seg_list, h, w):
    from pycocotools import mask as M
    m = np.zeros((h, w), bool)
    for s in seg_list:
        if isinstance(s, list):
            m |= M.decode(M.merge(M.frPyObjects(s, h, w))).astype(bool)
        else:
            if isinstance(s.get('counts'), list):
                s = M.frPyObjects(s, h, w)
            m |= M.decode(s).astype(bool)
    ys, xs = np.nonzero(m)
    return np.c_[xs, ys].astype(float)


# ---------------------------------------------------------------------------------------------------------------- statistics
def loo_lda(X, y, permutations, seed):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    def acc(lab):
        hit = 0
        for i in range(len(lab)):
            tr = np.ones(len(lab), bool)
            tr[i] = False
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
            try:
                m = LinearDiscriminantAnalysis().fit((X[tr] - mu) / sd, lab[tr])
            except (ValueError, IndexError, np.linalg.LinAlgError):     # no within-species spread (e.g. constant ratios): LDA undefined
                from sklearn.neighbors import NearestCentroid
                m = NearestCentroid().fit((X[tr] - mu) / sd, lab[tr])
            hit += m.predict(((X[i] - mu) / sd)[None])[0] == lab[i]
        return hit / len(lab)
    obs = acc(y)
    rng = np.random.default_rng(seed)
    null = np.array([acc(rng.permutation(y)) for _ in range(permutations)]) if permutations else np.array([])
    p = (np.sum(null >= obs) + 1) / (len(null) + 1) if len(null) else np.nan
    _, cnt = np.unique(y, return_counts=True)
    return obs, null, p, float(((cnt / cnt.sum()) ** 2).sum())


def bh(p):
    """Benjamini-Hochberg false discovery rate (q values) for a list of p values (NaN kept)."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        out = np.empty(len(pv))
        out[o] = np.minimum(r, 1)
        q[ok] = out
    return q


def dunn(xs, labels):
    """Dunn's pairwise test after Kruskal-Wallis (rank-based, tie-corrected); BH q within the variable."""
    from scipy.stats import norm, rankdata
    allx = np.concatenate(xs)
    N = len(allx)
    rk = rankdata(allx)
    _, t = np.unique(allx, return_counts=True)
    tie = (t ** 3 - t).sum() / (12 * (N - 1)) if N > 1 else 0
    mr, ns, s = [], [], 0
    for x in xs:
        mr.append(rk[s:s + len(x)].mean()); ns.append(len(x)); s += len(x)
    out = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            se = np.sqrt((N * (N + 1) / 12 - tie) * (1 / ns[i] + 1 / ns[j]))
            z = (mr[i] - mr[j]) / se if se > 0 else 0.0
            out.append([labels[i], labels[j], z, 2 * norm.sf(abs(z))])
    q = bh([r[3] for r in out])
    return [r + [qq] for r, qq in zip(out, q)]


def _ss_within(D2, g):
    s = 0.0
    for u in np.unique(g):
        i = np.where(g == u)[0]
        s += D2[np.ix_(i, i)].sum() / (2 * len(i))
    return s


def permanova(X, g, permutations, seed):
    """One-way PERMANOVA (Anderson 2001) on Euclidean distances of the z-scored variables: pseudo-F, R2, permutation p."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    D2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    N, k = len(g), len(np.unique(g))
    sst = D2.sum() / (2 * N)
    ssw = _ss_within(D2, g)
    F = ((sst - ssw) / (k - 1)) / (ssw / (N - k))
    rng = np.random.default_rng(seed)
    null = np.array([(lambda w: ((sst - w) / (k - 1)) / (w / (N - k)))(_ss_within(D2, rng.permutation(g))) for _ in range(permutations)])
    return F, (sst - ssw) / sst, (np.sum(null >= F) + 1) / (permutations + 1)


def permdisp(X, g, permutations, seed):
    """Homogeneity of multivariate dispersion (Anderson 2006, PERMDISP): ANOVA F on distances to the group centroids, permutation p.
    PERMANOVA can be significant because groups differ in SPREAD rather than position; a non-significant PERMDISP rules that out."""
    from scipy.stats import f_oneway
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)

    def F_of(lab):
        d = np.empty(len(lab))
        for u in np.unique(lab):
            i = lab == u
            d[i] = np.sqrt(((Z[i] - Z[i].mean(0)) ** 2).sum(1))
        return d, f_oneway(*[d[lab == u] for u in np.unique(lab)]).statistic
    d, F = F_of(g)
    rng = np.random.default_rng(seed)
    resid = d - np.array([d[g == u].mean() for u in g])                  # permute residuals of the distances (Anderson 2006)
    base = np.array([d[g == u].mean() for u in g])
    from scipy.stats import f_oneway as fo
    null = np.array([fo(*[(base + rng.permutation(resid))[g == u] for u in np.unique(g)]).statistic for _ in range(permutations)])
    return F, (np.sum(null >= F) + 1) / (permutations + 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--json', required=True, help='COCO file with the landmarks (one keypoint set per image) and/or masks')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--config', default='', help='TSV: kind (measure|ratio), name, definition, description')
    ap.add_argument('--measure', action='append', help='NAME=TERM (repeat); see the syntax above')
    ap.add_argument('--ratio', action='append', help='NAME=NUM/DEN[*100] (repeat)')
    ap.add_argument('--keypoint_category', default='', help='only keypoint sets of this category (default: the first set per image)')
    ap.add_argument('--group_labels', default='', help='CSV filename,group_label (as for V42)')
    ap.add_argument('--group_regex', default=r'^([^_]+)_', help='group from the file name when --group_labels is not given')
    ap.add_argument('--px_per_mm', default='', help='CSV file,px_per_mm: measures also in mm')
    ap.add_argument('--loo_on', default='ratios', choices=['ratios', 'measures', 'both'],
                    help='variables of the leave-one-out species test (ratios: unit-free, safe for un-scaled images)')
    ap.add_argument('--permutations', type=int, default=999)
    ap.add_argument('--character', default='', help='TSV taxon,label: a coded character to compare every measure/ratio with')
    ap.add_argument('--positive', default='', help='binary --character: the label counted as positive')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--size_measure', default='', help='v2: allometry check - a measure used as body size (e.g. HL); each ratio is regressed '
                    'on it (log-log, common within-species slope) and the species test repeated on size-adjusted values. Needs real sizes: '
                    '--px_per_mm, or --assume_same_scale when every image has the same magnification')
    ap.add_argument('--assume_same_scale', action='store_true', help='v2: all images at one magnification (pixel sizes comparable)')
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    M, R, desc = read_definitions(a)
    need_cats = set().union(*[term_categories(t) for _, t, _ in M])
    images, kps, segs = read_coco(a.json, a.keypoint_category, need_cats)

    groups = {}
    if a.group_labels:
        for r in csv.DictReader(open(a.group_labels)):
            groups[os.path.splitext(os.path.basename(r['filename']))[0]] = r['group_label']
    scale = {}
    if a.px_per_mm:
        for r in csv.DictReader(open(a.px_per_mm)):
            scale[os.path.splitext(os.path.basename(r['file']))[0]] = float(r['px_per_mm'])

    rows = []
    for im in images:
        f = os.path.basename(str(im['file_name']))
        stem = os.path.splitext(f)[0]
        K = kps.get(f, np.zeros((0, 2)))
        masks = {c: mask_pixels(s, im['height'], im['width']) for c, s in segs.get(f, {}).items()}
        if not len(K) and not masks:
            continue
        g = groups.get(stem)
        if g is None:
            m = re.search(a.group_regex, f)
            g = m.group(1) if m else ''
        row = {'file': f, 'group': g}
        for n, t, _ in M:
            row[n] = evaluate(t, K, masks)
            if stem in scale:
                row[n + '_mm'] = row[n] / scale[stem] ** (2 if t[0] == 'mask' and t[2] == 'area' else 1)
        for n, num, den, k, _ in R:
            row[n] = row[num] / row[den] * k if np.isfinite(row[den]) and row[den] != 0 else np.nan
        rows.append(row)
    if not rows:
        sys.exit('no image has landmarks or the masks named in the definitions')
    cols = list(rows[0].keys())
    with open(os.path.join(a.out_dir, 'specimen_measurements.tsv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter='\t')
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])

    names = [n for n, _, _ in M] + [n for n, *_ in R]
    G = np.array([r['group'] for r in rows])
    V = {n: np.array([r[n] for r in rows], float) for n in names}
    ug = sorted(set(G))
    with open(os.path.join(a.out_dir, 'species_summary.tsv'), 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['group', 'variable', 'n', 'mean', 'sd', 'min', 'max', 'text'])
        for g in ug:
            for n in names:
                x = V[n][(G == g) & np.isfinite(V[n])]
                if len(x):
                    dec = 1 if n in [r[0] for r in R] else 2
                    w.writerow([g, n, len(x), round(x.mean(), 4), round(x.std(ddof=1), 4) if len(x) > 1 else '', round(x.min(), 4), round(x.max(), 4),
                                f'{x.min():.{dec}f}-{x.max():.{dec}f} ({x.mean():.{dec}f})' if len(x) > 1 else f'{x[0]:.{dec}f}'])

    from scipy.stats import kruskal, mannwhitneyu
    multi = [g for g in ug if (G == g).sum() >= 2]
    tests, lines = [], []
    for n in names:
        xs = [V[n][(G == g) & np.isfinite(V[n])] for g in multi]
        xs = [x for x in xs if len(x)]
        if len(xs) >= 2 and np.ptp(np.concatenate(xs)) > 0:
            H, p = kruskal(*xs)
            tests.append((n, 'Kruskal-Wallis across groups (n>=2)', len(xs), round(H, 3), p))
    q_kw = bh([t[4] for t in tests])                                    # v2: FDR over all variables tested
    tests = [t + (q,) for t, q in zip(tests, q_kw)]
    pair_rows = []                                                      # v2: which species pairs differ (Dunn, BH within variable)
    for n in names:
        lab = [g for g in multi if np.isfinite(V[n][G == g]).any()]
        xs = [V[n][(G == g) & np.isfinite(V[n])] for g in lab]
        if len(xs) >= 2 and np.ptp(np.concatenate(xs)) > 0:
            for g1, g2, z, p, q in dunn(xs, lab):
                pair_rows.append([n, g1, g2, round(np.nanmean(V[n][G == g1]), 4), round(np.nanmean(V[n][G == g2]), 4), round(z, 3), p, q])
    with open(os.path.join(a.out_dir, 'pairwise_dunn.tsv'), 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['variable', 'group_a', 'group_b', 'mean_a', 'mean_b', 'z', 'p', 'q_BH_within_variable'])
        w.writerows(pair_rows)
    use = [n for n, *_ in R] if a.loo_on == 'ratios' else ([n for n, _, _ in M] if a.loo_on == 'measures' else names)
    if not use:
        use = [n for n, _, _ in M]
    X = np.column_stack([V[n] for n in use])
    keep = np.isin(G, multi) & np.all(np.isfinite(X), 1)
    if keep.sum() >= 3 and len(set(G[keep])) >= 2:
        acc, null, p, chance = loo_lda(X[keep], G[keep], a.permutations, a.seed)
        tests.append(('+'.join(use), f'LDA leave-one-out species ID ({keep.sum()} specimens, {len(set(G[keep]))} groups)', len(use), round(100 * acc, 1), p))
        lines.append(f'leave-one-out species ID on {", ".join(use)}: {100 * acc:.1f}% ({keep.sum()} specimens in {len(set(G[keep]))} groups with n >= 2); '
                     f'permutation chance mean {100 * null.mean():.1f}%, 95th percentile {100 * np.percentile(null, 95):.1f}%, p = {p:.4f}; '
                     f'proportional chance {100 * chance:.1f}%' if len(null) else f'leave-one-out species ID: {100 * acc:.1f}%')
        if a.permutations:                                              # v2: multivariate position (PERMANOVA) and spread (PERMDISP)
            F, R2, pF = permanova(X[keep], G[keep], a.permutations, a.seed)
            Fd, pd_ = permdisp(X[keep], G[keep], a.permutations, a.seed)
            tests.append(('+'.join(use), 'PERMANOVA (Euclidean, z-scored), pseudo-F', len(use), round(F, 3), pF, np.nan))
            tests.append(('+'.join(use), 'PERMDISP (dispersion), F', len(use), round(Fd, 3), pd_, np.nan))
            lines.append(f'PERMANOVA on the same variables: pseudo-F {F:.2f}, R2 {R2:.2f} (share of variation between species), p = {pF:.4f}; '
                         f'PERMDISP (do species differ in spread?) F {Fd:.2f}, p = {pd_:.4f}'
                         + (' -> spreads differ, so part of the PERMANOVA result may be spread, not position' if pd_ < 0.05 else ' -> spreads similar, PERMANOVA reflects position'))
    allo = []
    if a.size_measure:                                                  # v2: allometry check
        if a.size_measure not in V:
            sys.exit(f'--size_measure {a.size_measure} is not a defined measure')
        if not (scale or a.assume_same_scale):
            lines.append(f'allometry check skipped: {a.size_measure} is in pixels and the images may differ in magnification '
                         '(give --px_per_mm, or --assume_same_scale if they do not)')
        else:
            sz = np.array([r.get(a.size_measure + '_mm', r[a.size_measure]) for r in rows], float)
            for n, *_ in R:
                y = V[n]
                ok = np.isin(G, multi) & np.isfinite(y) & np.isfinite(sz) & (y > 0) & (sz > 0)
                ly, ls = np.log(y[ok]), np.log(sz[ok]); gg = G[ok]
                cy = ly - np.array([ly[gg == u].mean() for u in gg]); cs = ls - np.array([ls[gg == u].mean() for u in gg])
                b = (cs @ cy) / (cs @ cs) if (cs @ cs) > 0 else 0.0                    # common within-species slope
                r2 = 1 - ((cy - b * cs) ** 2).sum() / (cy ** 2).sum() if (cy ** 2).sum() > 0 else 0.0
                adj = ly - b * (ls - ls.mean())
                H, p = kruskal(*[adj[gg == u] for u in np.unique(gg)])
                allo.append((n, round(b, 3), round(r2, 3), round(H, 3), p))
            lines.append(f'allometry ({a.size_measure} as size, log-log, common within-species slope; slope 0 = ratio independent of size): '
                         + '; '.join(f'{n} slope {b} R2 {r2}, species after size adjustment p {p:.2g}' for n, b, r2, H, p in allo))
    with open(os.path.join(a.out_dir, 'tests.tsv'), 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['variable', 'test', 'k', 'statistic_or_accuracy_pct', 'p', 'q_BH'])
        w.writerows([t if len(t) == 6 else t + (np.nan,) for t in tests])

    agree = []
    if a.character:
        ch = {}
        for r in csv.DictReader(open(a.character), delimiter='\t'):
            ch.setdefault(r['taxon'].strip(), r['label'].strip())
        means = {n: {g: np.nanmean(V[n][G == g]) for g in ug} for n in names}
        labelled = [g for g in ug if g in ch]
        states = sorted({ch[g] for g in labelled})
        for n in names:
            if a.positive and len(states) >= 2:
                pos = [means[n][g] for g in labelled if ch[g] == a.positive and np.isfinite(means[n][g])]
                neg = [means[n][g] for g in labelled if ch[g] != a.positive and np.isfinite(means[n][g])]
                if pos and neg:
                    u = mannwhitneyu(pos, neg, alternative='two-sided')
                    agree.append((n, f'{a.positive} vs rest (species means)', len(pos), len(neg), round(u.statistic / (len(pos) * len(neg)), 3), u.pvalue))
            elif len(states) >= 2:
                xs = [[means[n][g] for g in labelled if ch[g] == s and np.isfinite(means[n][g])] for s in states]
                xs = [x for x in xs if x]
                if len(xs) >= 2:
                    H, p = kruskal(*xs)
                    agree.append((n, 'Kruskal-Wallis on species means across ' + '/'.join(states), sum(map(len, xs)), len(xs), round(H, 3), p))
        agree = [r + (q,) for r, q in zip(agree, bh([r[5] for r in agree]))]           # v2: FDR over the variables
        with open(os.path.join(a.out_dir, 'character_agreement.tsv'), 'w', newline='') as fh:
            w = csv.writer(fh, delimiter='\t')
            w.writerow(['variable', 'comparison', 'n_pos_or_species', 'n_neg_or_states', 'AUC_or_H', 'p', 'q_BH'])
            w.writerows(agree)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        show = [n for n, *_ in R] or [n for n, _, _ in M]
        fig, axs = plt.subplots(len(show), 1, figsize=(max(6, 0.45 * len(ug)), 2.4 * len(show)), squeeze=False)
        for ax, n in zip(axs[:, 0], show):
            for i, g in enumerate(ug):
                x = V[n][(G == g) & np.isfinite(V[n])]
                ax.plot(np.full(len(x), i) + np.linspace(-0.12, 0.12, len(x)) if len(x) > 1 else [i], x, 'o', ms=4, alpha=0.8, color='#3b6ea5')
                if len(x):
                    ax.hlines(x.mean(), i - 0.25, i + 0.25, color='#222', lw=1)
            ax.set_ylabel(n)
            ax.set_xticks(range(len(ug)))
            ax.set_xticklabels(ug, rotation=60, fontsize=7)
            ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, 'ratios_by_group.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f'figure skipped: {e}')

    with open(os.path.join(a.out_dir, 'summary.txt'), 'w') as fh:
        fh.write(f'{len(rows)} specimens, {len(ug)} groups ({len(multi)} with n >= 2) from {a.json}\n\nDefinitions:\n')
        for n, _, d in M:
            fh.write(f'  {n} = {d}{"   " + desc[n] if desc.get(n) else ""}\n')
        for n, num, den, k, d in R:
            fh.write(f'  {n} = {d}{"   " + desc[n] if desc.get(n) else ""}\n')
        fh.write('\nDo the species differ on average? Kruskal-Wallis across groups with n >= 2 (q = Benjamini-Hochberg over all variables):\n')
        for t in tests:
            if t[1].startswith('Kruskal'):
                fh.write(f'  {t[0]:24s} H {t[3]:8.2f}  p {t[4]:.2g}  q {t[5]:.2g}\n')
        npair = sum(1 for r in pair_rows if r[7] < 0.05)
        fh.write(f'  pairwise (Dunn, BH within each variable): {npair} of {len(pair_rows)} variable x species-pair comparisons q < 0.05 '
                 '-> pairwise_dunn.tsv (which species a ratio separates, for keys and diagnoses)\n')
        fh.write('\nCan single specimens be told apart? (and multivariate tests)\n  ' + '\n  '.join(lines) + '\n')
        if agree:
            fh.write(f'\nAgreement with {a.character} (one mean per species, so specimens of a species are not counted as independent):\n')
            for r in agree:
                fh.write(f'  {r[0]:24s} {r[1]}: {"AUC" if a.positive else "H"} {r[4]}  p {r[5]:.2g}  q {r[6]:.2g}  ({r[2]} / {r[3]})\n')
        fh.write('\nNotes: a ratio can differ between species only because the species differ in size (allometry); with real sizes '
                 '(--px_per_mm) use --size_measure to check. Groups with one specimen enter the summaries but not the tests.\n')
    print(open(os.path.join(a.out_dir, 'summary.txt')).read())


if __name__ == '__main__':
    main()
