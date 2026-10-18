"""Wing margin fringe as a character (17 Oct 2026, AVD: 'how much of the posterior margin is fringed with setae').
No annotation needed: membrane edge at NATIVE resolution (opening removes the 2-3 px hairs), outline split at the base (leftmost)
and apex (rightmost) points into the anterior and the posterior chain (wing base on the LEFT). Every --step px along a chain:
outward normal; the hairs are thin dark lines on the plain background (grey black-top-hat > --thr outside the membrane);
fringe length = farthest distance along the normal (+- --side px sideways) that still has hair pixels, capped at --max_len.
Per wing and chain: fringed fraction (length >= --min_len), median length, length profile in 10 bins base->apex, hair density
(dark runs crossing a line --probe px outside the edge, per 100 px of margin). px_per_mm from --scales TSV (image<TAB>px_per_mm) if given.
Writes fringe_per_wing.tsv, fringe_profiles.tsv and native x2 check crops for --sheet images.
Usage: script --images DIR --files a.jpg b.jpg ... | --list FILE  --out DIR"""
import argparse, os, csv
import numpy as np, cv2
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--images', required=True); ap.add_argument('--files', nargs='*', default=[]); ap.add_argument('--list', default='')
ap.add_argument('--out', required=True); ap.add_argument('--step', type=int, default=10); ap.add_argument('--thr', type=float, default=6); ap.add_argument('--side', type=int, default=4)
ap.add_argument('--max_len', type=int, default=150); ap.add_argument('--min_len', type=int, default=12); ap.add_argument('--probe', type=int, default=10)
ap.add_argument('--scales', default=''); ap.add_argument('--sheet', type=int, default=4); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); files = a.files + ([l.strip() for l in open(a.list) if l.strip()] if a.list else [])
SC = {r[0]: float(r[1]) for r in csv.reader(open(a.scales), delimiter='\t') if len(r) > 1 and r[1].replace('.', '', 1).isdigit()} if a.scales else {}
def membrane(I):
    g = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY); bg = np.median(np.concatenate([g[:20].ravel(), g[-20:].ravel(), g[:, :20].ravel(), g[:, -20:].ravel()]))
    f = (np.abs(cv2.GaussianBlur(g, (0, 0), 1.5).astype(int) - bg) > 10).astype(np.uint8)
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    n, lb, st, _ = cv2.connectedComponentsWithStats(f); f = (lb == 1 + st[1:, 4].argmax()).astype(np.uint8)
    cs, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); m = np.zeros_like(f); cv2.drawContours(m, cs, -1, 1, -1)
    C = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32); K = 31   # smoothed outline (circular moving average) for stable normals
    Cs = np.stack([np.convolve(np.r_[C[-K:, j], C[:, j], C[:K, j]], np.ones(K) / K, 'same')[K:-K] for j in range(2)], 1)
    return g, m, Cs
rows, prof, shown = [], [], 0
for fn in files:
    I = np.asarray(Image.open(os.path.join(a.images, fn)).convert('RGB')); g, m, C = membrane(I); H, W = g.shape
    hair = (cv2.morphologyEx(cv2.GaussianBlur(g, (0, 0), 0.7), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > a.thr) & (m == 0)
    hair = hair.astype(np.uint8)
    ib, ia = int(np.argmin(C[:, 0])), int(np.argmax(C[:, 0]))   # base (leftmost), apex (rightmost)
    seq = lambda i, j: C[i:j + 1] if i <= j else np.concatenate([C[i:], C[:j + 1]])
    c1, c2 = seq(ib, ia), seq(ia, ib)[::-1]
    post, ant = (c1, c2) if c1[:, 1].mean() > c2[:, 1].mean() else (c2, c1)   # posterior chain = lower in the image
    res = {'image': fn, 'px_per_mm': SC.get(fn, '')}
    for name, ch in (('posterior', post), ('anterior', ant)):
        ch = ch.astype(np.float32); k = max(15, a.step); L = []; X = []; dens_hits = 0
        for i in range(k, len(ch) - k, a.step):
            p = ch[i]; t = ch[i + k] - ch[i - k]; t /= np.linalg.norm(t) + 1e-6; nrm = np.array([-t[1], t[0]])
            q = p + 6 * nrm
            if 0 <= int(q[1]) < H and 0 <= int(q[0]) < W and m[int(q[1]), int(q[0])]: nrm = -nrm   # point outward
            far = 0
            for dist in range(3, a.max_len):
                hit = False
                for s in range(-a.side, a.side + 1):
                    x, y = (p + dist * nrm + s * t).astype(int)
                    if 0 <= x < W and 0 <= y < H and hair[y, x]: hit = True; break
                if hit: far = dist
                elif dist - far > 12: break
            L.append(far); X.append(p[0])
        L = np.array(L); X = np.array(X)
        # density: dark-run crossings along the probe line a.probe px outside the edge
        line = []
        for i in range(k, len(ch) - k, 2):
            p = ch[i]; t = ch[i + k] - ch[i - k]; t /= np.linalg.norm(t) + 1e-6; nrm = np.array([-t[1], t[0]]); q = p + 6 * nrm
            if 0 <= int(q[1]) < H and 0 <= int(q[0]) < W and m[int(q[1]), int(q[0])]: nrm = -nrm
            x, y = (p + a.probe * nrm).astype(int); line.append(hair[y, x] if 0 <= x < W and 0 <= y < H else 0)
        line = np.array(line); runs = int(((line[1:] == 1) & (line[:-1] == 0)).sum()); mlen = len(ch)
        fr = L >= a.min_len; res[f'{name}_fringed_frac'] = round(float(fr.mean()), 3) if len(L) else ''
        res[f'{name}_median_len_px'] = round(float(np.median(L[fr])), 1) if fr.any() else 0
        res[f'{name}_hairs_per_100px'] = round(runs * 100 / max(mlen, 1), 2); res[f'{name}_margin_len_px'] = mlen
        if fr.any():   # where the fringe stops: last fringed point as a fraction of the base->apex x range
            res[f'{name}_fringe_end_frac'] = round(float((X[fr].max() - C[ib, 0]) / max(C[ia, 0] - C[ib, 0], 1)), 3)
        bins = np.clip(((X - C[ib, 0]) / max(C[ia, 0] - C[ib, 0], 1) * 10).astype(int), 0, 9)
        prof.append([fn, name] + [round(float(np.median(L[bins == b])), 1) if (bins == b).any() else '' for b in range(10)])
        if name == 'posterior' and shown < a.sheet:   # native x2 crop around the middle of the posterior chain, measured tips in red
            mid = ch[len(ch) // 2].astype(int); y0, x0 = max(0, mid[1] - 120), max(0, mid[0] - 300); cr = I[y0:y0 + 240, x0:x0 + 600].copy()
            for i, (Lv) in zip(range(k, len(ch) - k, a.step), L):
                p = ch[i]; t = ch[i + k] - ch[i - k]; t /= np.linalg.norm(t) + 1e-6; nrm = np.array([-t[1], t[0]]); q = p + 6 * nrm
                if 0 <= int(q[1]) < H and 0 <= int(q[0]) < W and m[int(q[1]), int(q[0])]: nrm = -nrm
                e = p + Lv * nrm; cv2.line(cr, (int(p[0] - x0), int(p[1] - y0)), (int(e[0] - x0), int(e[1] - y0)), (255, 0, 0), 1)
            cs_, _ = cv2.findContours(m[y0:y0 + 240, x0:x0 + 600].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(cr, cs_, -1, (0, 160, 0), 1)
            Image.fromarray(cv2.resize(cr, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)).save(os.path.join(a.out, f'check_{os.path.splitext(fn)[0]}.jpg'), quality=92); shown += 1
    rows.append(res); print(fn, {k: v for k, v in res.items() if 'posterior' in k}, flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k != 'image', k))
with open(os.path.join(a.out, 'fringe_per_wing.tsv'), 'w') as fh:
    fh.write('\t'.join(keys) + '\n'); [fh.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n') for r in rows]
with open(os.path.join(a.out, 'fringe_profiles.tsv'), 'w') as fh:
    fh.write('image\tchain\t' + '\t'.join(f'bin{b}_len_px' for b in range(10)) + '\n'); [fh.write('\t'.join(map(str, r)) + '\n') for r in prof]
print('done', len(rows))
