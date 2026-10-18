#!/usr/bin/env python3
"""Which estimate of a hidden margin is closer to the truth: the convex-hull chord (outline repair) or the TPS estimate
from the reference shape (estimate_hidden_semilandmarks_v1.py, Gunz et al. 2009)? Added 12 Oct 2026.

Stretches of k consecutive semilandmarks that ARE visible are hidden on purpose, at random positions, in every specimen;
each stretch is rebuilt (a) by the straight chord between its two visible neighbours (what the convex-hull bridge gives
across a short gap) and (b) by TPS from the Procrustes mean of the OTHER specimens (the specimen never contributes to its
own reference), fitted on the specimen's remaining visible points. Error = mean distance to the true points, as a share
of the specimen's centroid size. Points already flagged hidden in the real data are never used as truth.

Usage: validate_hidden_semilandmarks_v1.py --gpa_dir semilandmarks/<cat> --hidden_csv <tps_dir>/hidden_semilandmarks.csv
           --output_dir out [--lengths 3,6,9,12 --draws 3]
"""
import argparse, json, os, sys
import numpy as np
from scipy.interpolate import RBFInterpolator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # flat pip layout: siblings in the same folder
from estimate_hidden_semilandmarks_v1 import norm, gpa, fit_any


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpa_dir', required=True); ap.add_argument('--hidden_csv', default='')
    ap.add_argument('--output_dir', required=True); ap.add_argument('--lengths', default='3,6,9,12')
    ap.add_argument('--draws', type=int, default=3); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.output_dir, exist_ok=True); rng = np.random.default_rng(a.seed)
    bt = json.load(open(os.path.join(a.gpa_dir, 'back_transformed_coco.json'))); im = {i['id']: i['file_name'] for i in bt['images']}
    S, names = [], []
    for x in bt['annotations']:
        P = np.array(x['segmentation'][0], float).reshape(-1, 2)
        S.append(P[:-1] if np.allclose(P[0], P[-1]) else P); names.append(im[x['image_id']])
    n = len(S[0]); flagged = {nm: set() for nm in names}
    if a.hidden_csv and os.path.exists(a.hidden_csv):
        import csv
        for r in csv.DictReader(open(a.hidden_csv)): flagged.setdefault(r['specimen'], set()).add(int(r['landmark']))
    rows = []
    for i, P in enumerate(S):
        others = [S[j] for j in range(len(S)) if j != i]; M = gpa(others); cs = np.sqrt(((P - P.mean(0)) ** 2).sum(1).sum())
        for k in [int(v) for v in a.lengths.split(',')]:
            for _ in range(a.draws):
                for _try in range(50):
                    s0 = int(rng.integers(0, n)); idx = [(s0 + t) % n for t in range(k)]
                    if not (set(idx) | {(s0 - 1) % n, (s0 + k) % n}) & flagged[names[i]]: break
                else:
                    continue
                h = np.zeros(n, bool); h[idx] = True; vis = ~h & ~np.isin(np.arange(n), list(flagged[names[i]]))
                p0, p1 = P[(s0 - 1) % n], P[(s0 + k) % n]; t = np.arange(1, k + 1) / (k + 1)
                chord = p0 + t[:, None] * (p1 - p0)
                Mi = fit_any(norm(M), norm(P), vis)
                tps = RBFInterpolator(Mi[vis], P[vis], kernel='thin_plate_spline', smoothing=1e-8)(Mi[h])
                truth = P[idx]
                rows.append((names[i], k, np.linalg.norm(chord - truth, axis=1).mean() / cs, np.linalg.norm(tps - truth, axis=1).mean() / cs))
    import csv
    with open(os.path.join(a.output_dir, 'hidden_semilandmark_validation_rows.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['specimen', 'k_hidden', 'chord_error_cs', 'tps_error_cs']); w.writerows(rows)
    R = np.array([(r[1], r[2], r[3]) for r in rows])
    lines = [f'{len(S)} specimens, {n} semilandmarks; error = mean distance to the true points / centroid size (x1000); '
             f'stretches never include points flagged hidden in the real data; the reference never includes the specimen itself.',
             'k hidden | n | chord median | TPS median | TPS better in']
    for k in sorted(set(R[:, 0])):
        r = R[R[:, 0] == k]
        lines.append(f'{int(k):>8} | {len(r):>3} | {1000 * np.median(r[:, 1]):12.2f} | {1000 * np.median(r[:, 2]):10.2f} | {np.mean(r[:, 2] < r[:, 1]):.0%}')
    txt = '\n'.join(lines) + '\n'; open(os.path.join(a.output_dir, 'hidden_semilandmark_validation.txt'), 'w').write(txt); print(txt)


if __name__ == '__main__':
    main()
