"""Vein MACROTRICHIA detector from a few example clicks (2 Nov 2026; the taxonomist clicked example setae, not every seta).

Recipe = the BioRAG-Probe idea at point level: frozen DINOv3 ViT-L/16 tokens on a x--up grid (grey, z-scored per tile), raw + high-pass
(token minus its 7x7 mean) at --layer, StandardScaler -> PCA64 -> logistic regression. Positives = the token under each click on a vein;
background = tokens >= --gap_px from every click, half ON the vein mask and half beside it (positive-unlabelled: unclicked setae
are in there, so the model is conservative). Clicked wings are scored by a model that never saw that wing (leave-one-wing-out); all other wings by the model fitted on
every clicked wing. Detections = local maxima of the smoothed score map within the dilated vein masks, >= --nms_px apart, at each of
--thresholds. Counts are lower bounds (setae fall off in life; dorsal + ventral rows superimpose in transmitted light, so a close pair
can be one detection).

Outputs (--out): report.txt (held-out recall of the clicks per threshold), detections.tsv, counts_per_vein.tsv, counts_per_chunk.tsv
(homologous chunks, same K rule as vein_pilosity_profile_v1), per_taxon_counts.tsv, maps/det_<image>.jpg (+ zoom sheets).
"""
import argparse, csv, json, os, re, time, numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from skimage.morphology import skeletonize
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
ap = argparse.ArgumentParser()
ap.add_argument('--images', required=True); ap.add_argument('--coco', required=True, help='vein masks (COCO)')
ap.add_argument('--clicks', required=True, help='Descriptron final JSON with keypoint annotations (one per image)'); ap.add_argument('--out', required=True)
ap.add_argument('--taxon_regex', default=r'^([A-Za-z]+?)_'); ap.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m')
ap.add_argument('--layer', type=int, default=18); ap.add_argument('--up', type=int, default=4); ap.add_argument('--tile', type=int, default=1024)
ap.add_argument('--vein_dilate_px', type=int, default=10); ap.add_argument('--gap_px', type=float, default=15); ap.add_argument('--n_bg', type=int, default=400)
ap.add_argument('--nms_px', type=float, default=10); ap.add_argument('--thresholds', default='0.5,0.7,0.9'); ap.add_argument('--hit_px', type=float, default=8)
ap.add_argument('--chunk_px', type=float, default=120); ap.add_argument('--kmax', type=int, default=10)
ap.add_argument('--base_part', default='humeral'); ap.add_argument('--base_part_ref', default='subcosta')
ap.add_argument('--veins', default='', help='comma list to restrict (default: all)'); ap.add_argument('--max_images', type=int, default=0)
ap.add_argument('--maps_per_taxon', type=int, default=1)
ap.add_argument('--smooth_tokens', type=float, default=0, help='Gaussian sigma (tokens) on the score map before peak finding; 0 = raw')
ap.add_argument('--nms_vein', default='', help='per-vein peak spacing overriding --nms_px, e.g. costa:7,subcosta:7 (parallel rows of setae on the costa)')
ap.add_argument('--resume', action='store_true', help='wings that already have <out>/scores/<stem>.npz reuse that score map (no DINOv3); models are refitted from the clicks (fixed seed). '
                'Clicked wings are always rescored and compared with their saved map as a check.')
ap.add_argument('--merge_click_px', type=float, default=0, help='merge clicks closer than this on one image (accidental double clicks); 0 = off')
a = ap.parse_args(); os.makedirs(f'{a.out}/maps', exist_ok=True); THR = [float(x) for x in a.thresholds.split(',')]
LOG = open(f'{a.out}/report.txt', 'w')
def log(*s): print(*s, flush=True); print(*s, file=LOG); LOG.flush()
rng = np.random.default_rng(0); taxon = lambda f: re.match(a.taxon_regex, f).group(1)
V = json.load(open(a.coco)); cat = {c['id']: c['name'] for c in V['categories']}; van = {}
for an in V['annotations']: van.setdefault(an['image_id'], []).append(an)
vim = {os.path.splitext(i['file_name'])[0]: i for i in V['images']}
CK = {}
for an in json.load(open(a.clicks))['annotations']:
    stem = os.path.splitext(os.path.basename(str(an['image_id'])))[0]
    if stem in vim and an.get('keypoints'): CK.setdefault(stem, []).append(np.array(an['keypoints'], float).reshape(-1, 3)[:, :2])
CK = {k: np.concatenate(v) for k, v in CK.items()}
if a.merge_click_px > 0:   # accidental double clicks on one seta -> one click
    for k, P in CK.items():
        kept = []
        for q in P:
            if all(np.hypot(*(q - r)) >= a.merge_click_px for r in kept): kept.append(q)
        CK[k] = np.array(kept)
NV = {kv.split(':')[0]: float(kv.split(':')[1]) for kv in a.nms_vein.split(',') if kv}
keep = set(a.veins.split(',')) if a.veins else None
ims = [vim[s] for s in sorted(CK)] + [i for i in sorted(V['images'], key=lambda i: i['file_name']) if os.path.splitext(i['file_name'])[0] not in CK]
if a.max_images: ims = ims[:a.max_images]
log(f'vein_macrotrichia_detect v1 {time.strftime("%Y-%m-%d %H:%M")} | {len(ims)} wings, {sum(os.path.splitext(i["file_name"])[0] in CK for i in ims)} with clicks '
    f'({sum(len(CK[os.path.splitext(i["file_name"])[0]]) for i in ims if os.path.splitext(i["file_name"])[0] in CK)} clicks) | L{a.layer} x{a.up} | thresholds {THR}')
from transformers import AutoModel
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net = AutoModel.from_pretrained(a.model).to(dev).eval(); DIM = net.config.hidden_size; NREG = getattr(net.config, 'num_register_tokens', 4)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]); PX = 16 / a.up   # native px per token
def centreline(m):   # same as vein_pilosity_profile_v1
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
def tokens(G, region):
    """token features on a PX-native-px grid for every token whose centre is inside region; returns (rows, cols, feats float16)."""
    H, W = G.shape; Tn = a.tile // a.up; OVn = 8 * 16 // a.up; STEP = Tn - 2 * OVn; out_rc, out_f = [], []
    for y0 in range(0, H, STEP):
        for x0 in range(0, W, STEP):
            if not region[y0:y0 + STEP, x0:x0 + STEP].any(): continue
            ya, xa = max(0, y0 - OVn), max(0, x0 - OVn); c = G[ya:min(H, y0 + STEP + OVn), xa:min(W, x0 + STEP + OVn)]
            u = cv2.resize(c, None, fx=a.up, fy=a.up, interpolation=cv2.INTER_CUBIC); u = u[:u.shape[0] // 16 * 16, :u.shape[1] // 16 * 16]
            if min(u.shape) < 16 * 8: continue
            u = np.clip((u - u.mean()) / (u.std() + 1e-6) * .18 + .5, 0, 1)
            x = torch.from_numpy(((np.stack([u] * 3, -1) - MEAN) / STD).astype(np.float32).transpose(2, 0, 1).copy())[None].to(dev)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
                T = net(pixel_values=x, output_hidden_states=True).hidden_states[a.layer + 1][0, 1 + NREG:].float().reshape(u.shape[0] // 16, u.shape[1] // 16, DIM)
            mu = F.avg_pool2d(T.permute(2, 0, 1)[None], 7, 1, 3, count_include_pad=False)[0].permute(1, 2, 0); Hp = T - mu
            gy, gx = np.mgrid[0:T.shape[0], 0:T.shape[1]]; cy, cx = ya + (gy + .5) * PX, xa + (gx + .5) * PX
            core = (cy >= y0) & (cy < min(H, y0 + STEP)) & (cx >= x0) & (cx < min(W, x0 + STEP))
            ok = core & region[cy.astype(int).clip(0, H - 1), cx.astype(int).clip(0, W - 1)]
            if not ok.any(): continue
            ok_t = torch.from_numpy(ok).to(dev)
            out_f.append(torch.cat([T[ok_t], Hp[ok_t]], 1).cpu().numpy().astype(np.float16))
            out_rc.append(np.stack([np.round(cy[ok] / PX - .5), np.round(cx[ok] / PX - .5)], 1).astype(np.int32))
    return np.concatenate(out_rc), np.concatenate(out_f)
def wing(im, feats=True):
    f = im['file_name']; G = np.asarray(Image.open(os.path.join(a.images, f)).convert('L'), np.float32) / 255; H, W = G.shape; masks = {}
    for an in van[im['id']]:
        nm = cat[an['category_id']]
        if keep and nm not in keep and nm not in (a.base_part, a.base_part_ref): continue
        m = np.zeros((H, W), np.uint8)
        for p in an['segmentation']: cv2.fillPoly(m, [np.array(p).reshape(-1, 2).astype(np.int32)], 1)
        masks[nm] = masks.get(nm, 0) | m
    det_veins = [v for v in masks if not keep or v in keep]; allv = np.zeros((H, W), np.uint8)
    for v in det_veins: allv |= masks[v]
    region = cv2.dilate(allv, np.ones((2 * a.vein_dilate_px + 1,) * 2, np.uint8)).astype(bool)
    if not feats: return G, masks, det_veins, region, None, None
    rc, X = tokens(G, region); return G, masks, det_veins, region, rc, X
def fit(X, y): return make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced')).fit(X.astype(np.float32), y)
# pass 1: clicked wings -> training tokens (+ keep their full token sets for held-out scoring)
TR, KEEP, POS = {}, {}, {}
for im in ims:
    stem = os.path.splitext(im['file_name'])[0]
    if stem not in CK: continue
    t0 = time.time(); G, masks, dv, region, rc, X = wing(im); P = CK[stem]
    inreg = region[P[:, 1].astype(int).clip(0, G.shape[0] - 1), P[:, 0].astype(int).clip(0, G.shape[1] - 1)]
    tree = cKDTree(rc[:, ::-1] * PX + PX / 2); d, j = tree.query(P[inreg]); pos = np.unique(j[d <= PX])
    dc, _ = cKDTree(P).query(rc[:, ::-1] * PX + PX / 2); allv = np.zeros(G.shape, np.uint8)
    for v in dv: allv |= masks[v]
    onv = allv[(rc[:, 0] * PX + PX / 2).astype(int).clip(0, G.shape[0] - 1), (rc[:, 1] * PX + PX / 2).astype(int).clip(0, G.shape[1] - 1)] > 0
    b1, b2 = np.flatnonzero((dc >= a.gap_px) & onv), np.flatnonzero((dc >= a.gap_px) & ~onv)   # half ON the vein (bare vein / unclicked), half beside it
    bgi = np.r_[rng.choice(b1, min(a.n_bg // 2, len(b1)), replace=False), rng.choice(b2, min(a.n_bg // 2, len(b2)), replace=False)].astype(int)
    TR[stem] = (np.concatenate([X[pos], X[bgi]]), np.r_[np.ones(len(pos)), np.zeros(len(bgi))]); KEEP[stem] = (G, masks, dv, region, rc, X); POS[stem] = (pos, bgi)
    log(f'  clicked {stem}: {len(P)} clicks, {inreg.sum()} within {a.vein_dilate_px} px of a scored vein -> {len(pos)} positive tokens, {len(bgi)} background  ({time.time() - t0:.0f}s)')
models = {s: fit(np.concatenate([TR[k][0] for k in TR if k != s]), np.concatenate([TR[k][1] for k in TR if k != s])) for s in TR}
full = fit(np.concatenate([v[0] for v in TR.values()]), np.concatenate([v[1] for v in TR.values()])); log(f'models: {len(models)} leave-one-wing-out + 1 on all clicked wings')
# pass 2: score every wing, detect peaks, assign to veins / chunks
DET, VREC, HIT, HITV = [], [], {t: [0, 0] for t in THR}, {}
for n, im in enumerate(ims):
    f = im['file_name']; stem = os.path.splitext(f)[0]; t0 = time.time()
    RS = f'{a.out}/scores/{stem}.npz'
    if a.resume and stem not in KEEP and os.path.exists(RS):   # --resume: reuse the saved score map
        G, masks, dv, region, _, _ = wing(im, feats=False); Z_ = np.load(RS); S, M = Z_['S'].astype(np.float32), Z_['M']; NRES = globals().get('NRES', 0) + 1
    else:
        G, masks, dv, region, rc, X = KEEP.pop(stem) if stem in KEEP else wing(im)
        sc = (models[stem] if stem in models else full).predict_proba(X.astype(np.float32))[:, 1]; del X
        if stem in POS: log(f'  held-out raw score {stem}: clicks median {np.median(sc[POS[stem][0]]):.2f} (IQR {np.percentile(sc[POS[stem][0]], 25):.2f}-{np.percentile(sc[POS[stem][0]], 75):.2f}), background median {np.median(sc[POS[stem][1]]):.2f}, 95% {np.percentile(sc[POS[stem][1]], 95):.2f}')
        gh, gw = int(np.ceil(G.shape[0] / PX)), int(np.ceil(G.shape[1] / PX)); S = np.zeros((gh, gw), np.float32); M = np.zeros((gh, gw), bool)
        S[rc[:, 0].clip(0, gh - 1), rc[:, 1].clip(0, gw - 1)] = sc; M[rc[:, 0].clip(0, gh - 1), rc[:, 1].clip(0, gw - 1)] = True
        if a.resume and os.path.exists(RS):   # determinism check: refitted model vs the saved map of the stopped run
            _o = np.load(RS)['S'].astype(np.float32); log(f'  resume check {stem}: max |new - saved| score {np.abs(_o - S.astype(np.float16).astype(np.float32)).max():.4f}' if _o.shape == S.shape else f'  resume check {stem}: shape differs')
        os.makedirs(f'{a.out}/scores', exist_ok=True); np.savez_compressed(f'{a.out}/scores/{stem}.npz', S=S.astype(np.float16), M=M, px=PX, model='held_out' if stem in CK else 'all_clicked')   # redo peaks offline
    if a.resume: S = S.astype(np.float16).astype(np.float32)   # same precision for reused and new maps (peak ties behave alike)
    Ss = gaussian_filter(S, a.smooth_tokens) if a.smooth_tokens > 0 else S; pk = (Ss == maximum_filter(Ss, size=int(2 * round(min([a.nms_px] + list(NV.values())) / PX) + 1))) & M & (Ss >= min(THR))
    py, px = np.nonzero(pk); xy = np.stack([(px + .5) * PX, (py + .5) * PX], 1); ps = Ss[py, px]
    H, W = G.shape; dist = {v: cv2.distanceTransform(1 - masks[v], cv2.DIST_L2, 5) for v in dv}
    vi = np.argmin(np.stack([dist[v][xy[:, 1].astype(int).clip(0, H - 1), xy[:, 0].astype(int).clip(0, W - 1)] for v in dv], 1), 1) if len(xy) else np.array([], int)
    if NV and len(xy):   # each peak must be a maximum at ITS vein's spacing (--nms_vein; other veins keep --nms_px)
        MF = {r_: maximum_filter(Ss, size=int(2 * round(r_ / PX) + 1)) for r_ in {NV.get(v, a.nms_px) for v in dv}}
        ok = np.array([Ss[y_, x_] == MF[NV.get(dv[k_], a.nms_px)][y_, x_] for y_, x_, k_ in zip(py, px, vi)], bool)
        py, px, xy, ps, vi = py[ok], px[ok], xy[ok], ps[ok], vi[ok]
    cen = lambda nm: np.argwhere(masks[nm] > 0)[:, ::-1].mean(0) if nm in masks and masks[nm].any() else None; base = cen(a.base_part)
    for k, v in enumerate(dv):
        Pc, s = centreline(masks[v])
        if Pc is None: continue
        ref = cen(a.base_part_ref) if v == a.base_part else base
        if ref is not None and np.hypot(*(Pc[-1] - ref)) < np.hypot(*(Pc[0] - ref)): Pc, s = Pc[::-1], s[-1] - s[::-1]
        sel = np.flatnonzero(vi == k); t = s[cKDTree(Pc).query(xy[sel])[1]] / max(s[-1], 1) if len(sel) else np.array([])
        VREC.append(dict(image=f, taxon=taxon(f), vein=v, length=float(s[-1]), t=t, score=ps[sel], clicked=stem in CK))
        for i, tt in zip(sel, t): DET.append((f, taxon(f), v, round(float(xy[i, 0]), 1), round(float(xy[i, 1]), 1), round(float(ps[i]), 3), round(float(tt), 3), 'held_out' if stem in CK else 'all_clicked'))
    if stem in CK and len(CK[stem]):   # held-out recall of the example clicks (precision is not measurable: clicks are examples, not counts)
        P = CK[stem]; inreg = region[P[:, 1].astype(int).clip(0, H - 1), P[:, 0].astype(int).clip(0, W - 1)]
        for th in THR:
            Q = xy[ps >= th]; d = cKDTree(Q).query(P[inreg])[0] if len(Q) else np.full(inreg.sum(), np.inf)
            HIT[th][0] += int((d <= a.hit_px).sum()); HIT[th][1] += int(inreg.sum())
            cv_ = np.argmin(np.stack([dist[v][P[inreg][:, 1].astype(int).clip(0, H - 1), P[inreg][:, 0].astype(int).clip(0, W - 1)] for v in dv], 1), 1)
            for k_, hit_ in zip(cv_, d <= a.hit_px): HITV.setdefault((dv[k_], th), [0, 0]); HITV[(dv[k_], th)][0] += int(hit_); HITV[(dv[k_], th)][1] += 1
    print(f'{n + 1}/{len(ims)} {f}: {int((ps >= THR[len(THR) // 2]).sum())} detections @ {THR[len(THR) // 2]}  ({time.time() - t0:.0f}s)', flush=True)
    if stem in CK or sum(r[1] == taxon(f) and r[0] != f for r in {(d_[0], d_[1]) for d_ in DET}) < a.maps_per_taxon:   # map
        O = (np.stack([G] * 3, -1) * 255).astype(np.uint8)
        for (x, y), p in zip(xy, ps):
            col = cv2.applyColorMap(np.uint8([[int(p * 255)]]), cv2.COLORMAP_PLASMA)[0, 0][::-1]; cv2.circle(O, (int(x), int(y)), 7, tuple(int(c) for c in col), 2)
        if stem in CK:
            for x, y in CK[stem]: cv2.drawMarker(O, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
        cv2.putText(O, f'{stem}: circles = detections (colour = score, plasma), green + = example clicks ({"held out" if stem in CK else "model on all clicked wings"})',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
        Image.fromarray(O).save(f'{a.out}/maps/det_{stem}.jpg', quality=90)
if a.resume: log(f'--resume: {globals().get("NRES", 0)} wings reused their saved score map')
log('held-out recall of the example clicks (a detection within %g px), clicked wings scored by models that never saw them:' % a.hit_px)
for th in THR: log(f'  threshold {th}: {HIT[th][0]}/{HIT[th][1]} = {HIT[th][0] / max(1, HIT[th][1]):.2f}')
log('  per vein (nearest vein mask to the click): ' + '; '.join(f'{v} ' + ' '.join(f'{HITV[(v, th)][0]}/{HITV[(v, th)][1]}@{th}' for th in THR) for v in sorted({v for v, _ in HITV})))
with open(f'{a.out}/detections.tsv', 'w') as fh:
    fh.write('image\ttaxon\tvein\tx\ty\tscore\tt\tmodel\n'); [fh.write('\t'.join(map(str, r)) + '\n') for r in DET]
veins = sorted({r['vein'] for r in VREC}); K = {v: int(np.clip(round(np.median([r['length'] for r in VREC if r['vein'] == v]) / a.chunk_px), 1, a.kmax)) for v in veins}
with open(f'{a.out}/counts_per_vein.tsv', 'w') as fh:
    fh.write('image\ttaxon\tvein\tlength_px\t' + '\t'.join(f'n@{t}\tper100px@{t}' for t in THR) + '\n')
    for r in VREC: fh.write(f'{r["image"]}\t{r["taxon"]}\t{r["vein"]}\t{r["length"]:.0f}\t' + '\t'.join(f'{(r["score"] >= t).sum()}\t{100 * (r["score"] >= t).sum() / max(r["length"], 1):.2f}' for t in THR) + '\n')
mid = THR[len(THR) // 2]
with open(f'{a.out}/counts_per_chunk.tsv', 'w') as fh:
    fh.write(f'image\ttaxon\tvein\tK\tchunk\tt_mid\tn@{mid}\n')
    for r in VREC:
        ch = np.minimum((r['t'][r['score'] >= mid] * K[r['vein']]).astype(int), K[r['vein']] - 1)
        for j in range(K[r['vein']]): fh.write(f'{r["image"]}\t{r["taxon"]}\t{r["vein"]}\t{K[r["vein"]]}\t{j + 1}\t{(j + .5) / K[r["vein"]]:.3f}\t{int((ch == j).sum())}\n')
taxa = sorted({r['taxon'] for r in VREC})
with open(f'{a.out}/per_taxon_counts.tsv', 'w') as fh:
    fh.write(f'taxon\tn_images\t' + '\t'.join(f'{v}_per100px@{mid}' for v in veins) + '\n')
    for tx in taxa:
        fh.write(f'{tx}\t{len({r["image"] for r in VREC if r["taxon"] == tx})}\t' + '\t'.join(
            (lambda L: f'{np.mean(L):.2f}' if L else '')([100 * (r['score'] >= mid).sum() / max(r['length'], 1) for r in VREC if r['taxon'] == tx and r['vein'] == v]) for v in veins) + '\n')
log(f'median detections per 100 px of vein @ {mid} (all wings):')
for v in veins:
    L = [100 * (r['score'] >= mid).sum() / max(r['length'], 1) for r in VREC if r['vein'] == v]; log(f'  {v:9s} {np.median(L):.2f}  (IQR {np.percentile(L, 25):.2f}-{np.percentile(L, 75):.2f}, n {len(L)})')
log('done')
