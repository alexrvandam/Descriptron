"""Wing spot pattern, no training (17 Oct 2026). Training scale is enough for spots (AVD), but we keep native resolution anyway.
Per wing: membrane = native outline minus veins (grown --vein_grow px) minus the basal --base_frac of the wing length.
Clear-membrane reference = the --ref_pct percentile of membrane L*; spot = smoothed L* darker than the reference by >= --dl,
opened, components >= --min_frac of the wing. Per wing: spot fraction of the membrane in 10 bins along the wing (base -> apex)
and in named zones (central = --central, preapical = --preapical, as fractions of base->apex length).
Optional: --spot_coco (hand spot outlines) -> IoU per wing for each --dl. Optional: --labels TSV (taxon, label) + --positive ->
species-level AUC of the zone fraction with a permutation null.
Usage: script --images DIR --veins_coco VEINS.json --out DIR [--spot_coco S.json] [--labels L.tsv --positive present --zone preapical]"""
import argparse, json, os, re, csv
import numpy as np, cv2
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--images', required=True); ap.add_argument('--veins_coco', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--spot_coco', default=''); ap.add_argument('--labels', default=''); ap.add_argument('--positive', default='present'); ap.add_argument('--zone', default='preapical')
ap.add_argument('--dls', default='8,12,16'); ap.add_argument('--dl', type=float, default=12); ap.add_argument('--ref_pct', type=float, default=80); ap.add_argument('--vein_grow', type=int, default=12)
ap.add_argument('--base_frac', type=float, default=0.15); ap.add_argument('--min_frac', type=float, default=0.003)
ap.add_argument('--central', default='0.35,0.58'); ap.add_argument('--preapical', default='0.62,0.92'); ap.add_argument('--taxon_regex', default=r'^([A-Za-z]+?)_')
ap.add_argument('--max_elong', type=float, default=5, help='v1b: drop components longer than this x their width'); ap.add_argument('--sheet', type=int, default=8); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); DLS = [float(x) for x in a.dls.split(',')]; ZONES = {'central': [float(x) for x in a.central.split(',')], 'preapical': [float(x) for x in a.preapical.split(',')]}
LOG = open(os.path.join(a.out, 'summary.txt'), 'w')
def log(*s): print(*s); print(*s, file=LOG); LOG.flush()
def membrane(I):
    g = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY); bg = np.median(np.concatenate([g[:20].ravel(), g[-20:].ravel(), g[:, :20].ravel(), g[:, -20:].ravel()]))
    f = (np.abs(cv2.GaussianBlur(g, (0, 0), 1.5).astype(int) - bg) > 10).astype(np.uint8)
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    n, lb, st, _ = cv2.connectedComponentsWithStats(f); f = (lb == 1 + st[1:, 4].argmax()).astype(np.uint8)
    cs, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); m = np.zeros_like(f); cv2.drawContours(m, cs, -1, 1, -1); return m
def masks_of(path, cat_filter=None):
    d = json.load(open(path)); cat = {c['id']: c['name'] for c in d['categories']}; by = {}
    for an in d['annotations']:
        if cat_filter is None or cat[an['category_id']] in cat_filter: by.setdefault(an['image_id'], []).append(an)
    return {im['file_name']: (im, by.get(im['id'], [])) for im in d['images']}
def raster(anns, H, W):
    m = np.zeros((H, W), np.uint8)
    for an in anns:
        for p in an['segmentation']: cv2.fillPoly(m, [np.array(p).reshape(-1, 2).astype(np.int32)], 1)
    return m
VE = masks_of(a.veins_coco); SP = masks_of(a.spot_coco) if a.spot_coco else {}
taxon = lambda f: (re.search(a.taxon_regex, f) or re.search('(.*)', f)).group(1)
rows, ious, shown = [], {dl: [] for dl in DLS}, 0
for fn, (im, vanns) in VE.items():
    p = os.path.join(a.images, fn)
    if not os.path.exists(p): continue
    I = np.asarray(Image.open(p).convert('RGB')); H, W = I.shape[:2]; wing = membrane(I); V = raster(vanns, H, W)
    xs = np.nonzero(wing.any(0))[0]; x0, x1 = xs.min(), xs.max(); xr = (np.arange(W) - x0) / max(x1 - x0, 1)
    memb = wing & (1 - cv2.dilate(V, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.vein_grow + 1,) * 2))); memb[:, xr < a.base_frac] = 0
    L = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * 100 / 255; Ls = cv2.GaussianBlur(L, (0, 0), 6)
    ref = np.percentile(L[memb > 0], a.ref_pct)
    def spots(dl):
        s = ((Ls < ref - dl) & (memb > 0)).astype(np.uint8); s = cv2.morphologyEx(s, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        n, lb, st, _ = cv2.connectedComponentsWithStats(s); k = np.zeros(n, np.uint8); k[1:] = st[1:, 4] >= a.min_frac * wing.sum()
        for i in range(1, n):   # v1b: long thin strips (tinted costal cell, anal fold) are not spots
            if k[i] and a.max_elong:
                cs_, _ = cv2.findContours((lb == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); (w_, h_) = cv2.minAreaRect(max(cs_, key=cv2.contourArea))[1]
                if max(w_, h_) / max(min(w_, h_), 1) > a.max_elong: k[i] = 0
        return k[lb]
    S = spots(a.dl)
    r = {'image': fn, 'taxon': taxon(fn), 'spot_frac': round(float(S.sum() / max(memb.sum(), 1)), 4), 'ref_L': round(float(ref), 1)}
    for b in range(10):
        sel = (xr >= b / 10) & (xr < (b + 1) / 10); mm = memb[:, sel]; r[f'bin{b}'] = round(float(S[:, sel].sum() / max(mm.sum(), 1)), 4)
    for z, (lo, hi) in ZONES.items():
        sel = (xr >= lo) & (xr < hi); r[z] = round(float(S[:, sel].sum() / max(memb[:, sel].sum(), 1)), 4)
    if fn in SP:
        G = raster(SP[fn][1], H, W) & wing
        for dl in DLS:
            P = spots(dl)   # scored on membrane only (veins inside a hand spot are not membrane)
            Gm, Pm = G & memb, P & memb; ious[dl].append((Gm & Pm).sum() / max((Gm | Pm).sum(), 1))
        r['hand_iou'] = round(float(ious[a.dl][-1]), 3)
    rows.append(r)
    if shown < a.sheet and (fn in SP or not SP):
        s = 900 / W; O = cv2.resize(I, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); m = cv2.resize(S, (O.shape[1], O.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        O2 = O.copy(); O2[m] = (0.45 * O2[m] + [140, 0, 140]).astype(np.uint8)
        if fn in SP:
            g_ = cv2.resize(raster(SP[fn][1], H, W), (O.shape[1], O.shape[0]), interpolation=cv2.INTER_NEAREST); cs, _ = cv2.findContours(g_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(O2, cs, -1, (0, 170, 0), 2)
        cv2.putText(O2, f'{fn[:36]} spot (magenta) vs hand (green)' if fn in SP else fn[:40], (6, 22), 0, 0.6, (0, 0, 0), 2)
        Image.fromarray(np.hstack([O, O2])).save(os.path.join(a.out, f'spots_{os.path.splitext(fn)[0]}.jpg'), quality=88); shown += 1
keys = list(rows[0].keys()) + (['hand_iou'] if 'hand_iou' not in rows[0] else [])
with open(os.path.join(a.out, 'per_wing.tsv'), 'w') as fh:
    fh.write('\t'.join(keys) + '\n'); [fh.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n') for r in rows]
log(f'wing_spot_pattern_v1 | {len(rows)} wings | dl {a.dl} | ref pct {a.ref_pct} | zones {ZONES}')
if SP:
    for dl in DLS: log(f'  spot mask vs hand outlines (membrane only), dl {dl}: mean IoU {np.mean(ious[dl]):.3f} (n={len(ious[dl])})')
if a.labels:
    from sklearn.metrics import roc_auc_score
    LAB = {r_[0]: r_[1] for r_ in csv.reader(open(a.labels), delimiter='\t') if len(r_) > 1 and not r_[0].startswith('#')}
    for z in list(ZONES) + ['spot_frac']:
        tm = {}
        for r in rows: tm.setdefault(r['taxon'], []).append(r[z])
        lt = [t for t in tm if t in LAB]; y = np.array([LAB[t] == a.positive for t in lt]); sc = np.array([np.mean(tm[t]) for t in lt])
        if len(set(y)) < 2: continue
        A = roc_auc_score(y, sc); rng = np.random.default_rng(0); nl = [roc_auc_score(rng.permutation(y), sc) for _ in range(5000)]
        log(f'  {z:10s} vs literature ({a.positive}: {int(y.sum())} of {len(y)} species): species AUC {A:.3f}, permutation null95 {np.percentile(nl, 95):.3f}')
        if z == a.zone:
            for t in sorted(lt, key=lambda t: -np.mean(tm[t])): log(f'     {t:16s} {LAB[t]:8s} {np.mean(tm[t]):.3f} (n={len(tm[t])})')
