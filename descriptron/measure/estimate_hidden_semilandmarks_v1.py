#!/usr/bin/env python3
"""Hidden semilandmarks estimated by thin-plate spline from the reference shape (added 12 Oct 2026; opt-in).

When another structure (antenna, leg) lies over a margin, the semilandmark scripts (V34/V42 --repair_outline_with)
bridge the hidden stretch along the convex hull: a straight chord. This step treats those bridged semilandmarks as
MISSING and estimates them from the reference shape, following Gunz et al. (2009, J. Hum. Evol. 57: 48-62;
geomorph estimate.missing(method = "TPS")):
  1. a semilandmark is "hidden" when it does not lie on the structure's own (unrepaired) mask, dilated by a tolerance
     (default 1.5% of the structure size, because the repaired outline is smoothed by ~1%);
  2. the reference = Procrustes mean of the specimens with no hidden semilandmarks (all specimens if fewer than 3);
  3. for each specimen, a TPS is fitted from the reference's VISIBLE points to the specimen's visible points and
     applied to the reference's hidden points; the estimate replaces the chord;
  4. steps 2-3 are repeated (--iterations) with the updated mean.
Writes a GPA folder with the same two files the colour/texture homology scripts read (gpa_mean.csv,
back_transformed_coco.json), so it can be passed to them as --gpa_dir, plus hidden_semilandmarks.csv (every hidden
point: chord position, TPS estimate, shift in px) and hidden_semilandmarks_summary.txt. Keep the original folder:
running the analysis on both is the sensitivity analysis.

Usage: estimate_hidden_semilandmarks_v1.py --gpa_dir semilandmarks/<cat> --json annotations.json --category_name <cat>
           --output_dir semilandmarks_tps/<cat> [--tol_px 3 --iterations 3]
"""
import argparse, json, os, re, logging
import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator
from pycocotools import mask as mu
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def norm(X):
    X = X - X.mean(0); return X / np.sqrt((X ** 2).sum())


def rot(X, Y):
    U, _, Vt = np.linalg.svd(X.T @ Y); D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(U @ Vt)); return X @ (U @ D @ Vt)


def fit_any(X, Y, w):
    """Orthogonal Procrustes of X onto Y using the visible points w, reflection allowed (whichever fits better)."""
    U, _, Vt = np.linalg.svd(X[w].T @ Y[w]); R = U @ Vt
    return X @ R


def gpa(shapes, iters=20):
    Z = [norm(s) for s in shapes]; M = Z[0]
    for _ in range(iters):
        Z = [rot(z, M) for z in Z]; M2 = norm(np.mean(Z, 0))
        if np.abs(M2 - M).max() < 1e-8: break
        M = M2
    return M


def decode(ann, h, w):
    sg = ann['segmentation']
    r = mu.merge(mu.frPyObjects(sg, h, w)) if isinstance(sg, list) else (mu.frPyObjects(sg, h, w) if isinstance(sg.get('counts'), list) else sg)
    return mu.decode(r).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpa_dir', required=True); ap.add_argument('--json', required=True); ap.add_argument('--category_name', required=True)
    ap.add_argument('--output_dir', required=True); ap.add_argument('--tol_px', type=float, default=0,
                    help='a point further than this from the structure mask is hidden; 0 (default) = --tol_frac of the structure size')
    ap.add_argument('--tol_frac', type=float, default=0.015, help='tolerance as a share of the structure size (the outline repair smooths by ~1%%)')
    ap.add_argument('--also_visible', default='',
                    help='categories that are part of the outline but not of the structure mask (e.g. compound_eye on a '
                         'head outline): points lying on them are visible, not hidden')
    ap.add_argument('--figure', type=int, default=6, help='overlay figure of chord vs TPS for this many specimens (0 = none)')
    ap.add_argument('--image_dir', default='', help='images for the figure (optional; outlines only without it)')
    ap.add_argument('--iterations', type=int, default=3); ap.add_argument('--landmark_step', type=int, default=1,
                    help='use every Nth visible point as TPS control (1 = all visible; raise if the fit is ill-conditioned)')
    a = ap.parse_args(); os.makedirs(a.output_dir, exist_ok=True)
    bt = json.load(open(os.path.join(a.gpa_dir, 'back_transformed_coco.json')))
    coco = json.load(open(a.json)); cid = {c['name'].lower(): c['id'] for c in coco['categories']}[a.category_name.lower()]
    imgs = {os.path.basename(i['file_name']): i for i in coco['images']}
    vis_ids = {c['id'] for c in coco['categories'] if c['name'].lower() in {n.strip().lower() for n in a.also_visible.split(',') if n.strip()}}
    extra = {}
    for x in coco['annotations']:
        if x['category_id'] in vis_ids and x.get('segmentation'): extra.setdefault(x['image_id'], []).append(x)
    best = {}
    for x in coco['annotations']:
        if x['category_id'] == cid and x.get('segmentation'):
            if x['image_id'] not in best or x.get('area', 0) > best[x['image_id']].get('area', 0): best[x['image_id']] = x
    bimg = {i['id']: i['file_name'] for i in bt['images']}
    S, hidden, names, anns = [], [], [], []
    for x in bt['annotations']:
        fn = bimg[x['image_id']]; orig = re.sub(r'_\d+$', '', os.path.basename(fn))
        P = np.array(x['segmentation'][0], float).reshape(-1, 2)
        closed = np.allclose(P[0], P[-1]); P = P[:-1] if closed else P
        im = imgs.get(orig)
        if im is None or im['id'] not in best:
            logging.warning(f'no original mask for {fn}; kept as is'); hid = np.zeros(len(P), bool)
        else:
            m = decode(best[im['id']], im['height'], im['width'])
            for e_ in extra.get(im['id'], []): m |= decode(e_, im['height'], im['width'])
            tol = a.tol_px or a.tol_frac * np.ptp(P, 0).max()
            k = int(2 * tol + 1); m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            xy = np.round(P).astype(int); xy[:, 0] = xy[:, 0].clip(0, im['width'] - 1); xy[:, 1] = xy[:, 1].clip(0, im['height'] - 1)
            hid = m[xy[:, 1], xy[:, 0]] == 0
        S.append(P); hidden.append(hid); names.append(fn); anns.append((x, closed))
    n_h = np.array([h.sum() for h in hidden]); logging.info(f'{len(S)} specimens; {int((n_h > 0).sum())} with hidden semilandmarks '
                                                            f'(median {np.median(n_h[n_h > 0]) if (n_h > 0).any() else 0:.0f} of {len(S[0])})')
    E = [s.copy() for s in S]; rows = []
    for it in range(a.iterations):
        ref_idx = [i for i in range(len(S)) if not hidden[i].any()]
        if len(ref_idx) < 3: ref_idx = list(range(len(S)))
        M = gpa([E[i] for i in ref_idx])
        for i in range(len(S)):
            h = hidden[i]
            if not h.any() or h.all() or (~h).sum() < 6: continue
            vis = np.where(~h)[0][::a.landmark_step]
            Mi = fit_any(norm(M), norm(E[i]), ~h)   # reference in the specimen's orientation (reflection allowed: the semilandmark
                                                    # script may have reflected 'mirror image' outlines); scale/translation -> TPS
            f = RBFInterpolator(Mi[vis], E[i][vis], kernel='thin_plate_spline', smoothing=1e-8)
            E[i][h] = f(Mi[h])
    for i in range(len(S)):
        for j in np.where(hidden[i])[0]:
            rows.append((names[i], j, *S[i][j], *E[i][j], float(np.hypot(*(E[i][j] - S[i][j])))))
    M = gpa(E)
    with open(os.path.join(a.output_dir, 'gpa_mean.csv'), 'w') as fh:
        fh.write('x,y\n'); [fh.write(f'{p[0]},{p[1]}\n') for p in M]
    out = json.loads(json.dumps(bt))
    for (x, closed), e, o in zip(anns, E, out['annotations']):
        pts = np.vstack([e, e[:1]]) if closed else e; o['segmentation'] = [pts.ravel().tolist()]
    json.dump(out, open(os.path.join(a.output_dir, 'back_transformed_coco.json'), 'w'))
    with open(os.path.join(a.output_dir, 'hidden_semilandmarks.csv'), 'w') as fh:
        fh.write('specimen,landmark,x_chord,y_chord,x_tps,y_tps,shift_px\n')
        for r in rows: fh.write(f'{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.2f}\n')
    sh = np.array([r[6] for r in rows]) if rows else np.array([0.0])
    txt = (f'{len(S)} specimens, {int((n_h > 0).sum())} with hidden semilandmarks, {len(rows)} hidden points in total.\n'
           f'TPS estimate vs convex-hull chord: median shift {np.median(sh):.1f} px, 90th percentile {np.percentile(sh, 90):.1f} px, max {sh.max():.1f} px.\n'
           f'Reference: Procrustes mean of specimens without hidden points; {a.iterations} iterations.\n')
    open(os.path.join(a.output_dir, 'hidden_semilandmarks_summary.txt'), 'w').write(txt); print(txt)
    if a.figure:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        order = np.argsort([-h.sum() for h in hidden])[:a.figure // 2].tolist() + np.argsort([h.sum() if h.any() else 1e9 for h in hidden])[:a.figure - a.figure // 2].tolist()
        fig, ax = plt.subplots(1, len(order), figsize=(4 * len(order), 4.4)); ax = np.atleast_1d(ax)
        for k, i in enumerate(order):
            orig = re.sub(r'_\d+$', '', os.path.basename(names[i])); ip = os.path.join(a.image_dir, orig) if a.image_dir else ''
            if ip and os.path.exists(ip):
                img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB); ax[k].imshow(img)
            c_ = np.vstack([S[i], S[i][:1]]); e_ = np.vstack([E[i], E[i][:1]])
            ax[k].plot(*c_.T, '-', c='cyan', lw=1); ax[k].plot(*e_.T, '-', c='magenta', lw=1)
            ax[k].plot(*S[i][hidden[i]].T, 'o', c='cyan', ms=3); ax[k].plot(*E[i][hidden[i]].T, 'o', c='magenta', ms=3)
            pad = 0.08 * np.ptp(S[i], 0).max(); lo, hi = S[i].min(0) - pad, S[i].max(0) + pad
            ax[k].set_xlim(lo[0], hi[0]); ax[k].set_ylim(hi[1], lo[1]); ax[k].axis('off')
            ax[k].set_title(f'{orig[:28]}\n{int(hidden[i].sum())} hidden', fontsize=8)
        fig.suptitle('Hidden semilandmarks: cyan = convex-hull chord (outline repair), magenta = TPS estimate from the reference (Gunz et al. 2009)', fontsize=9)
        plt.tight_layout(); plt.savefig(os.path.join(a.output_dir, 'hidden_semilandmarks_examples.png'), dpi=90); plt.close()


if __name__ == '__main__':
    main()
