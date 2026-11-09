"""Vein pilosity PROFILES: setulae score in homologous chunks along every wing vein (2 Nov 2026).

Same features and classifier as biorag_probe_parts_v1 PRIMARY B (DINOv3 L18 raw + high-pass patch tokens at native scale,
StandardScaler -> PCA 64 -> logistic C 0.05, patches of a vein carry that vein's literature label), but every patch inside a vein
is scored and placed at its position along the vein, so each vein becomes a base -> apex profile instead of one number.

Homology: each vein's centreline = longest path through its skeleton, oriented from the end nearest the wing base (centroid of the
humeral vein; the humeral itself from its subcosta end). Position t = arc length / vein length (0 base, 1 apex). Each vein type gets
ONE chunk count for all specimens, K = clip(round(median length / --chunk_px), 1, --kmax), so chunk j of R5 is the same stretch of R5
in every wing.

Hold-out: a labelled vein is scored by models that saw neither that vein nor that taxon (leave-one-vein-out x taxon folds, as the
probe's B); an unlabelled vein (not in --part_labels) by models trained on all labelled veins of the other taxon folds.
Transmitted-light photos superimpose the dorsal and ventral surfaces, so a score is for the vein as seen, not per surface.

Outputs (--out): profiles_long.tsv (image, taxon, vein, source, K, chunk, t_mid, n_cells, P), per_taxon_profile.tsv (taxon x vein_chunk),
report.txt (vein-level held-out AUC on labelled veins + median profile per vein), heatmap_taxon_x_chunk.png, maps/chunks_<image>.jpg.
"""
import argparse, csv, json, os, re, time, numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser()
ap.add_argument('--images', required=True); ap.add_argument('--coco', required=True); ap.add_argument('--part_labels', required=True)
ap.add_argument('--positive', required=True); ap.add_argument('--out', required=True); ap.add_argument('--taxon_regex', default=r'^([A-Za-z]+?)_')
ap.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m'); ap.add_argument('--layer', type=int, default=18)
ap.add_argument('--dilate_px', type=int, default=6); ap.add_argument('--tile', type=int, default=1024); ap.add_argument('--patches', type=int, default=48)
ap.add_argument('--folds', type=int, default=3); ap.add_argument('--chunk_px', type=float, default=120); ap.add_argument('--kmax', type=int, default=10)
ap.add_argument('--base_part', default='humeral'); ap.add_argument('--base_part_ref', default='subcosta')
ap.add_argument('--order', default='costa,subcosta,humeral,r1,r,Rs,R5,R-M,bm-m,M1-2,M1,M2,M4,Cu,CuA,A1', help='vein order for tables/heatmap')
ap.add_argument('--max_images', type=int, default=0); ap.add_argument('--maps_per_taxon', type=int, default=1)
ap.add_argument('--smooth_sigma_px', type=float, default=40, help='smooth profile: Gaussian sigma along arc length (0 = off)')
ap.add_argument('--smooth_n', type=int, default=20, help='smooth profile sampled at this many fixed t positions per vein')
a = ap.parse_args(); os.makedirs(f'{a.out}/maps', exist_ok=True); LOG = open(f'{a.out}/report.txt', 'w')
def log(*s): print(*s, flush=True); print(*s, file=LOG); LOG.flush()
PL = {r[0]: r[1] for r in csv.reader(open(a.part_labels), delimiter='\t') if r and not r[0].startswith('#') and len(r) >= 2}
d = json.load(open(a.coco)); cat = {c['id']: c['name'] for c in d['categories']}; by = {}
for an in d['annotations']: by.setdefault(an['image_id'], []).append(an)
ims = [im for im in d['images'] if im['id'] in by]; ims = ims[:a.max_images] if a.max_images else ims
taxon = lambda f: (re.search(a.taxon_regex, f) or re.search('(.*)', f)).group(1)
log(f'vein_pilosity_profile v1 {time.strftime("%Y-%m-%d %H:%M")} | {len(ims)} images | labelled {PL} | positive {a.positive} | L{a.layer} | chunk_px {a.chunk_px} kmax {a.kmax}')
from transformers import AutoModel
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; net = AutoModel.from_pretrained(a.model).to(dev).eval(); DIM = net.config.hidden_size; NREG = getattr(net.config, 'num_register_tokens', 4)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
def grid(x):   # same tiling as biorag_probe_parts_v1
    H, W = x.shape[:2]; OV = 128; STEP = a.tile - 2 * OV
    def one(t):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            hs = net(pixel_values=torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1)))[None].float().to(dev), output_hidden_states=True).hidden_states
        return hs[a.layer + 1][0, 1 + NREG:, :].float().reshape(t.shape[0] // 16, t.shape[1] // 16, DIM)
    if H <= a.tile and W <= a.tile: return one(x)
    out = torch.zeros((H // 16, W // 16, DIM), device=dev)
    for y0 in range(0, H, STEP):
        for x0 in range(0, W, STEP):
            ya, xa = max(0, y0 - OV), max(0, x0 - OV); t = x[ya:min(H, y0 + STEP + OV), xa:min(W, x0 + STEP + OV)]; t = t[:t.shape[0] // 16 * 16, :t.shape[1] // 16 * 16]
            T = one(t); ky, kx = (y0 - ya) // 16, (x0 - xa) // 16; ny, nx = min(STEP, H - y0) // 16, min(STEP, W - x0) // 16
            out[y0 // 16:y0 // 16 + ny, x0 // 16:x0 // 16 + nx] = T[ky:ky + ny, kx:kx + nx]
    return out
def centreline(m):
    """longest skeleton path of the largest skeleton component -> (N,2) xy points in order, cumulative arc length."""
    sk = skeletonize(m > 0); ys, xs = np.nonzero(sk)
    if len(ys) < 2: return None, None
    idx = -np.ones(sk.shape, np.int64); idx[ys, xs] = np.arange(len(ys)); r, c, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx; ok = (y2 < sk.shape[0]) & (x2 >= 0) & (x2 < sk.shape[1]); ok[ok] = sk[y2[ok], x2[ok]]
        r.append(idx[ys[ok], xs[ok]]); c.append(idx[y2[ok], x2[ok]]); w.append(np.full(ok.sum(), np.hypot(dy, dx)))
    r, c, w = np.concatenate(r), np.concatenate(c), np.concatenate(w); G = coo_matrix((w, (r, c)), shape=(len(ys),) * 2).tocsr(); G = G + G.T
    _, lab = connected_components(G, directed=False); big = np.flatnonzero(lab == np.bincount(lab).argmax())
    d0 = dijkstra(G, indices=big[0]); A = big[np.argmax(np.where(np.isfinite(d0[big]), d0[big], -1))]
    dA, pred = dijkstra(G, indices=A, return_predecessors=True); B = big[np.argmax(np.where(np.isfinite(dA[big]), dA[big], -1))]
    path = [B]
    while path[-1] != A: path.append(pred[path[-1]])
    P = np.stack([xs[path], ys[path]], 1).astype(np.float32); s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    return P, s
rng = np.random.default_rng(0); REC = []   # one record per (image, vein): cells, t per cell, features, length, train sample
for n, im in enumerate(ims):
    f = im['file_name']; I = np.asarray(Image.open(os.path.join(a.images, f)).convert('RGB')); H, W = I.shape[:2]; H16, W16 = H // 16 * 16, W // 16 * 16
    G = grid(((I[:H16, :W16].astype(np.float32) / 255 - MEAN) / STD).astype(np.float32)); gh, gw = H16 // 16, W16 // 16
    masks = {}
    for an in by[im['id']]:
        m = np.zeros((H, W), np.uint8)
        for p in an['segmentation']: cv2.fillPoly(m, [np.array(p).reshape(-1, 2).astype(np.int32)], 1)
        nm = cat[an['category_id']]; masks[nm] = masks.get(nm, 0) | m
    allm = np.zeros((H, W), np.uint8)
    for m in masks.values(): allm |= m
    wmask = torch.from_numpy(cv2.resize(cv2.dilate(allm, np.ones((41, 41), np.uint8))[:H16, :W16], (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)).to(dev).clamp(max=1)
    num = F.avg_pool2d((G * wmask[..., None]).permute(2, 0, 1)[None], 7, 1, 3, count_include_pad=False)[0].permute(1, 2, 0)   # high-pass as the probe
    den = F.avg_pool2d(wmask[None, None], 7, 1, 3, count_include_pad=False)[0, 0]; HP = G - num / den.clamp_min(1e-6)[..., None]
    cen = lambda nm: np.argwhere(masks[nm] > 0)[:, ::-1].mean(0) if nm in masks and masks[nm].any() else None
    base = cen(a.base_part)
    for nm, m in masks.items():
        P, s = centreline(m)
        if P is None: continue
        ref = cen(a.base_part_ref) if nm == a.base_part else base
        if ref is not None and np.hypot(*(P[-1] - ref)) < np.hypot(*(P[0] - ref)): P, s = P[::-1], s[-1] - s[::-1]
        md = cv2.dilate(m, np.ones((2 * a.dilate_px + 1,) * 2, np.uint8)); cov = cv2.resize(md[:H16, :W16].astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
        cells = np.flatnonzero(cov.ravel() >= 0.3)
        if len(cells) < 3: continue
        cxy = np.stack([(cells % gw + .5) * 16, (cells // gw + .5) * 16], 1); _, j = cKDTree(P).query(cxy); t = s[j] / max(s[-1], 1)
        ci = torch.from_numpy(cells).to(dev); X = torch.cat([G.reshape(-1, DIM)[ci], HP.reshape(-1, DIM)[ci]], 1).cpu().numpy().astype(np.float16)
        tr = np.sort(rng.choice(len(cells), min(a.patches, len(cells)), replace=False))
        REC.append(dict(image=f, taxon=taxon(f), vein=nm, label=PL.get(nm), cells=cells, t=t, X=X, tr=tr, length=float(s[-1]), grid=(gh, gw), hw=(H, W), cl=P, s=s))
    print(f'{n + 1}/{len(ims)} {f} veins {len(masks)}', flush=True); del G, HP, num; torch.cuda.empty_cache()
del net; torch.cuda.empty_cache()
veins = [v for v in a.order.split(',') if any(r['vein'] == v for r in REC)] + sorted({r['vein'] for r in REC} - set(a.order.split(',')))
K = {v: int(np.clip(round(np.median([r['length'] for r in REC if r['vein'] == v]) / a.chunk_px), 1, a.kmax)) for v in veins}
log('chunks per vein (median length px): ' + ', '.join(f'{v} {K[v]} ({np.median([r["length"] for r in REC if r["vein"] == v]):.0f})' for v in veins))
taxa = sorted({r['taxon'] for r in REC}); tf = {t: i % a.folds for i, t in enumerate(rng.permutation(taxa))}
lab_veins = [v for v in veins if v in PL]
def fit(train):
    X = np.concatenate([r['X'][r['tr']] for r in train]).astype(np.float32); y = np.concatenate([[int(r['label'] == a.positive)] * len(r['tr']) for r in train])
    return make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.05, max_iter=2000)).fit(X, y)
for k in range(a.folds):
    lab_tr = [r for r in REC if r['label'] is not None and tf[r['taxon']] != k]
    for v in lab_veins:   # leave-one-vein-out x taxon fold
        tr = [r for r in lab_tr if r['vein'] != v]
        if len({r['label'] == a.positive for r in tr}) < 2: continue
        m = fit(tr)
        for r in REC:
            if r['vein'] == v and tf[r['taxon']] == k: r['P'] = m.predict_proba(r['X'].astype(np.float32))[:, 1]; r['source'] = 'labelled'
    m = fit(lab_tr)
    for r in REC:
        if r['label'] is None and tf[r['taxon']] == k: r['P'] = m.predict_proba(r['X'].astype(np.float32))[:, 1]; r['source'] = 'predicted'
    log(f'fold {k + 1}/{a.folds} done ({sum(tf[t] == k for t in taxa)} taxa)')
REC = [r for r in REC if 'P' in r]
for r in REC:
    ch = np.minimum((r['t'] * K[r['vein']]).astype(int), K[r['vein']] - 1); r['ch'] = ch
    r['prof'] = [float(r['P'][ch == j].mean()) if (ch == j).any() else np.nan for j in range(K[r['vein']])]
with open(f'{a.out}/profiles_long.tsv', 'w') as fh:
    fh.write('image\ttaxon\tvein\tsource\tK\tchunk\tt_mid\tn_cells\tP\n')
    for r in REC:
        for j, p in enumerate(r['prof']): fh.write(f'{r["image"]}\t{r["taxon"]}\t{r["vein"]}\t{r["source"]}\t{K[r["vein"]]}\t{j + 1}\t{(j + .5) / K[r["vein"]]:.3f}\t{int((r["ch"] == j).sum())}\t{p:.3f}\n')
cols = [(v, j) for v in veins for j in range(K[v])]; M = np.full((len(taxa), len(cols)), np.nan)
for i, t in enumerate(taxa):
    for c, (v, j) in enumerate(cols):
        vals = [r['prof'][j] for r in REC if r['taxon'] == t and r['vein'] == v and np.isfinite(r['prof'][j])]
        if vals: M[i, c] = np.mean(vals)
src = {v: next(r['source'] for r in REC if r['vein'] == v) for v in veins}
with open(f'{a.out}/per_taxon_profile.tsv', 'w') as fh:
    fh.write('taxon\tn_images\t' + '\t'.join(f'{v}_{j + 1}of{K[v]}{"*" if src[v] == "predicted" else ""}' for v, j in cols) + '\n')
    for i, t in enumerate(taxa): fh.write(f'{t}\t{len({r["image"] for r in REC if r["taxon"] == t})}\t' + '\t'.join('' if np.isnan(x) else f'{x:.2f}' for x in M[i]) + '\n')
L = [r for r in REC if r['source'] == 'labelled']; yv = np.array([int(r['label'] == a.positive) for r in L]); sv = np.array([r['P'].mean() for r in L])
log(f'vein-level held-out AUC (labelled veins, vein AND taxon unseen): {roc_auc_score(yv, sv):.3f}  (n {len(L)}; probe B on the same veins: 1.000)')
log('median profile per vein over all images, base -> apex (* = vein never in training):')
for v in veins:
    pr = np.nanmedian(np.array([r['prof'] for r in REC if r['vein'] == v]), 0)
    log(f'  {v:9s}{"*" if src[v] == "predicted" else " "} {PL.get(v, "-"):15s} ' + ' '.join(f'{x:.2f}' for x in pr))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(max(10, len(cols) * .16), max(5, len(taxa) * .22)))
im_ = ax.imshow(M, aspect='auto', cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
x0 = 0
for v in veins:
    ax.axvline(x0 - .5, color='w', lw=1.5); ax.text(x0 + K[v] / 2 - .5, -1, v + ('*' if src[v] == 'predicted' else ''), ha='center', va='bottom', fontsize=7, rotation=90); x0 += K[v]
ax.set_yticks(range(len(taxa))); ax.set_yticklabels(taxa, fontsize=7); ax.set_xticks([]); ax.set_xlabel('veins, each base -> apex in homologous chunks (* = never trained; prediction only)')
fig.colorbar(im_, ax=ax, fraction=.02, label='P(setulose), taxon held out'); fig.tight_layout(); fig.savefig(f'{a.out}/heatmap_taxon_x_chunk.png', dpi=150); plt.close(fig)
done = {}
for r in REC:   # maps: each chunk filled with its own mean score
    t = r['taxon']
    if r['image'] not in done and len([1 for x in done.values() if x == t]) >= a.maps_per_taxon: continue
    done[r['image']] = t
for f in done:
    rs = [r for r in REC if r['image'] == f]; (gh, gw), (H, W) = rs[0]['grid'], rs[0]['hw']
    I = np.asarray(Image.open(os.path.join(a.images, f)).convert('RGB')); O = I.copy(); Pm = np.full(gh * gw, np.nan)
    for r in rs: Pm[r['cells']] = np.array(r['prof'])[r['ch']]
    Pu = cv2.resize(np.nan_to_num(Pm.reshape(gh, gw), nan=-1).astype(np.float32), (gw * 16, gh * 16), interpolation=cv2.INTER_NEAREST)
    full = np.full((H, W), -1, np.float32); full[:gh * 16, :gw * 16] = Pu; sel = full >= 0
    col = cv2.applyColorMap((np.clip(full, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)[..., ::-1]; O[sel] = (0.25 * O[sel] + 0.75 * col[sel]).astype(np.uint8)
    for r in rs:
        c = r['cells'][len(r['cells']) // 2]; cv2.putText(O, r['vein'] + ('*' if r['source'] == 'predicted' else ''), (int((c % gw) * 16), int((c // gw) * 16)), cv2.FONT_HERSHEY_SIMPLEX, max(.45, W / 3500), (0, 0, 0), 1, cv2.LINE_AA)
    Image.fromarray(np.vstack([I, O])).save(f'{a.out}/maps/chunks_{os.path.splitext(f)[0]}.jpg', quality=88)
log(f'maps {len(done)} -> {a.out}/maps (each chunk filled with its mean held-out P; yellow = setulose)')
if a.smooth_sigma_px > 0:   # smooth profile from the same held-out patch scores: Gaussian along arc length, sampled at fixed t
    tq = (np.arange(a.smooth_n) + .5) / a.smooth_n
    for r in REC:
        sc = r['t'] * r['length']; w = np.exp(-0.5 * ((r['s'][:, None] - sc[None]) / a.smooth_sigma_px) ** 2)
        r['Ps'] = (w * r['P'][None]).sum(1) / np.maximum(w.sum(1), 1e-9); r['sprof'] = np.interp(tq * r['length'], r['s'], r['Ps'])
    with open(f'{a.out}/smooth_profiles_long.tsv', 'w') as fh:
        fh.write('image\ttaxon\tvein\tsource\tt\tP_smooth\n')
        for r in REC:
            for t_, p in zip(tq, r['sprof']): fh.write(f'{r["image"]}\t{r["taxon"]}\t{r["vein"]}\t{r["source"]}\t{t_:.3f}\t{p:.3f}\n')
    with open(f'{a.out}/per_taxon_smooth_profile.tsv', 'w') as fh:
        fh.write('taxon\tn_images\t' + '\t'.join(f'{v}{"*" if src[v] == "predicted" else ""}_t{t_:.3f}' for v in veins for t_ in tq) + '\n')
        for t in taxa:
            vals = [np.nanmean([r['sprof'][i] for r in REC if r['taxon'] == t and r['vein'] == v]) if any(r['taxon'] == t and r['vein'] == v for r in REC) else np.nan for v in veins for i in range(a.smooth_n)]
            fh.write(f'{t}\t{len({r["image"] for r in REC if r["taxon"] == t})}\t' + '\t'.join('' if np.isnan(x) else f'{x:.2f}' for x in vals) + '\n')
    for f in done:   # same images as the chunk maps
        rs = [r for r in REC if r['image'] == f]; I = np.asarray(Image.open(os.path.join(a.images, f)).convert('RGB')); O = (I * 0.6 + 255 * 0.4).astype(np.uint8)
        lw = max(5, int(I.shape[1] / 400))
        for r in rs:
            C = r['cl']
            for k in range(len(C) - 1):
                col = cv2.applyColorMap(np.uint8([[int(np.clip(r['Ps'][k], 0, 1) * 255)]]), cv2.COLORMAP_PLASMA)[0, 0]
                cv2.line(O, tuple(C[k].astype(int)), tuple(C[k + 1].astype(int)), tuple(int(c) for c in col[::-1]), lw)
            mid = C[len(C) // 2].astype(int); cv2.putText(O, r['vein'] + ('*' if r['source'] == 'predicted' else ''), (int(mid[0]) + 8, int(mid[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, max(.45, I.shape[1] / 3500), (0, 0, 0), 1, cv2.LINE_AA)
        Image.fromarray(np.vstack([I, O])).save(f'{a.out}/maps/smooth_{os.path.splitext(f)[0]}.jpg', quality=88)
    log(f'smooth profiles (sigma {a.smooth_sigma_px} px, {a.smooth_n} fixed t per vein) -> smooth_profiles_long.tsv, per_taxon_smooth_profile.tsv, maps/smooth_*.jpg')
log('done')
