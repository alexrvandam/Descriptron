"""Wing spot pattern from DINOv3 patch tokens, no training (17 Oct 2026, AVD: 'a general rule from the DINO patch embeddings using
PCA ... I can see where the edges of the spots are in the PCA coloured wings'). The OpenCV L* threshold (wing_spot_pattern_v1.py)
is kept as the baseline.
Per wing: tokens on an 8-px grid (4 passes shifted by half a patch, native resolution, overlapping tiles), layer(s) --layers,
RAW tokens (spots are broad colour regions; high-pass removes them on purpose) inside the membrane (outline minus veins grown
--vein_grow px minus the basal --base_frac). PCA (16) fitted on that wing -> k-means K -> spot = the cluster(s) whose mean L* is
darker than the lightest cluster by >= --dl (rule, no annotation). Smooth upsampling (bilinear one-hot), scored at native resolution.
Outputs per wing: spot fraction per zone (central / preapical / 10 bins); IoU vs hand spot outlines (--spot_coco); species AUC vs
literature (--labels) with a permutation null. Sheet: photo | L18 PCA (RGB) | spot (magenta) + hand (green)."""
import argparse, json, os, re, csv, time
import numpy as np, cv2, torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
ap = argparse.ArgumentParser(); ap.add_argument('--images', required=True); ap.add_argument('--veins_coco', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--spot_coco', default=''); ap.add_argument('--labels', default=''); ap.add_argument('--positive', default='present')
ap.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m'); ap.add_argument('--layers', default='12,18'); ap.add_argument('--ks', default='3,4')
ap.add_argument('--dl', type=float, default=6); ap.add_argument('--vein_grow', type=int, default=12); ap.add_argument('--base_frac', type=float, default=0.15)
ap.add_argument('--central', default='0.35,0.58'); ap.add_argument('--preapical', default='0.62,0.92'); ap.add_argument('--taxon_regex', default=r'^([A-Za-z]+?)_')
ap.add_argument('--lowpass', type=int, default=0, help='v1b: radius in 8-px cells of the in-membrane token average before clustering (0 = off)'); ap.add_argument('--tile', type=int, default=1024); ap.add_argument('--max_images', type=int, default=0); ap.add_argument('--sheet', type=int, default=8); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); LAYERS = [int(x) for x in a.layers.split(',')]; KS = [int(x) for x in a.ks.split(',')]
ZONES = {'central': [float(x) for x in a.central.split(',')], 'preapical': [float(x) for x in a.preapical.split(',')]}
LOG = open(os.path.join(a.out, 'summary.txt'), 'w')
def log(*s): print(*s, flush=True); print(*s, file=LOG); LOG.flush()
from transformers import AutoModel
dev = 'cuda'; net = AutoModel.from_pretrained(a.model).to(dev).eval(); DIM = net.config.hidden_size; NREG = getattr(net.config, 'num_register_tokens', 4)
def grids(x):
    H, W = x.shape[:2]; OV = 128; STEP = a.tile - 2 * OV
    def one(t):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            hs = net(pixel_values=torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1)))[None].to(dev), output_hidden_states=True).hidden_states
        return {L: hs[L + 1][0, 1 + NREG:, :].float().reshape(t.shape[0] // 16, t.shape[1] // 16, DIM) for L in LAYERS}
    if H <= a.tile and W <= a.tile: return one(x)
    out = {L: torch.zeros((H // 16, W // 16, DIM), device=dev) for L in LAYERS}
    for y0 in range(0, H, STEP):
        for x0 in range(0, W, STEP):
            ya, xa = max(0, y0 - OV), max(0, x0 - OV); t = x[ya:min(H, y0 + STEP + OV), xa:min(W, x0 + STEP + OV)]; t = t[:t.shape[0] // 16 * 16, :t.shape[1] // 16 * 16]
            T = one(t); ky, kx = (y0 - ya) // 16, (x0 - xa) // 16; ny, nx = min(STEP, H - y0) // 16, min(STEP, W - x0) // 16
            for L in LAYERS: out[L][y0 // 16:y0 // 16 + ny, x0 // 16:x0 // 16 + nx] = T[L][ky:ky + ny, kx:kx + nx]
    return out
def grid8(I):
    x = (I.astype(np.float32) / 255 - MEAN) / STD; H, W = x.shape[:2]; H16, W16 = (H - 8) // 16 * 16, (W - 8) // 16 * 16; G = {}
    for dy in (0, 8):
        for dx in (0, 8):
            g = grids(x[dy:dy + H16, dx:dx + W16])
            for L in LAYERS:
                if L not in G: G[L] = torch.zeros((2 * g[L].shape[0], 2 * g[L].shape[1], DIM), device=dev)
                G[L][dy // 8::2, dx // 8::2] = g[L]
    return G
def membrane(I):
    g = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY); bg = np.median(np.concatenate([g[:20].ravel(), g[-20:].ravel(), g[:, :20].ravel(), g[:, -20:].ravel()]))
    f = (np.abs(cv2.GaussianBlur(g, (0, 0), 1.5).astype(int) - bg) > 10).astype(np.uint8)
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    n, lb, st, _ = cv2.connectedComponentsWithStats(f); f = (lb == 1 + st[1:, 4].argmax()).astype(np.uint8)
    cs, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); m = np.zeros_like(f); cv2.drawContours(m, cs, -1, 1, -1); return m
def coco_masks(path):
    d = json.load(open(path)); by = {}
    for an in d['annotations']: by.setdefault(an['image_id'], []).append(an)
    return {im['file_name']: by.get(im['id'], []) for im in d['images']}
def raster(anns, H, W):
    m = np.zeros((H, W), np.uint8)
    for an in anns:
        for p in an['segmentation']: cv2.fillPoly(m, [np.array(p).reshape(-1, 2).astype(np.int32)], 1)
    return m
VE = coco_masks(a.veins_coco); SP = coco_masks(a.spot_coco) if a.spot_coco else {}
taxon = lambda f: (re.search(a.taxon_regex, f) or re.search('(.*)', f)).group(1)
files = [f for f in VE if os.path.exists(os.path.join(a.images, f))]
files = sorted(files, key=lambda f: (f not in SP, f))[:a.max_images or None]   # hand-outlined wings first (sheet)
rows = []; shown = 0
log(f'wing_spot_tokens_v1 {time.strftime("%Y-%m-%d %H:%M")} | {len(files)} wings | layers {LAYERS} | K {KS} | dl {a.dl} | raw tokens, 8-px grid')
for n, fn in enumerate(files):
    t0 = time.time(); I = np.asarray(Image.open(os.path.join(a.images, fn)).convert('RGB')); H, W = I.shape[:2]; wing = membrane(I); V = raster(VE[fn], H, W)
    xs = np.nonzero(wing.any(0))[0]; x0, x1 = xs.min(), xs.max(); xr = (np.arange(W) - x0) / max(x1 - x0, 1)
    memb = wing & (1 - cv2.dilate(V, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.vein_grow + 1,) * 2))); memb[:, xr < a.base_frac] = 0
    Lst = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * 100 / 255
    G = grid8(I); h8, w8 = next(iter(G.values())).shape[:2]
    ys_, xs_ = (np.arange(h8) * 8 + 8).clip(0, H - 1), (np.arange(w8) * 8 + 8).clip(0, W - 1); mg = memb[np.ix_(ys_, xs_)]; idx = np.flatnonzero(mg.ravel() > 0)
    G_ = cv2.resize(memb, (W, H)); r = {'image': fn, 'taxon': taxon(fn)}; Gt = raster(SP[fn], H, W) & wing if fn in SP else None
    M = np.float32([[8, 0, 8], [0, 8, 8]]); keep_sheet = None
    for L in LAYERS:
        T = G[L]
        if a.lowpass:   # v1b: broad characters - average each token over its in-membrane neighbourhood (removes microtrichia speckle)
            import torch.nn.functional as F
            wt = torch.from_numpy(mg.astype(np.float32)).to(dev); k_ = 2 * a.lowpass + 1
            num = F.avg_pool2d((T * wt[..., None]).permute(2, 0, 1)[None], k_, 1, a.lowpass, count_include_pad=False)[0].permute(1, 2, 0)
            den = F.avg_pool2d(wt[None, None], k_, 1, a.lowpass, count_include_pad=False)[0, 0]; T = num / den.clamp_min(1e-6)[..., None]
        Z = PCA(16, random_state=0).fit_transform(T.reshape(-1, DIM)[torch.from_numpy(idx).to(dev)].cpu().numpy())
        for K in KS:
            lab = np.full(h8 * w8, -1); lab[idx] = KMeans(K, n_init=4, random_state=0).fit_predict(Z); lab = lab.reshape(h8, w8)
            up = np.stack([cv2.warpAffine((lab == k).astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for k in range(K)], -1)
            pix = up.argmax(-1); mL = [Lst[(pix == k) & (memb > 0)].mean() if ((pix == k) & (memb > 0)).any() else 0 for k in range(K)]
            light = max(mL); S = np.isin(pix, [k for k in range(K) if mL[k] <= light - a.dl]).astype(np.uint8) & memb
            tag = f'L{L}_K{K}'; r[f'{tag}_spot_frac'] = round(float(S.sum() / max(memb.sum(), 1)), 4)
            for z, (lo, hi) in ZONES.items(): sel = (xr >= lo) & (xr < hi); r[f'{tag}_{z}'] = round(float(S[:, sel].sum() / max(memb[:, sel].sum(), 1)), 4)
            if Gt is not None: Gm = Gt & memb; r[f'{tag}_iou'] = round(float((Gm & S).sum() / max((Gm | S).sum(), 1)), 3)
            if L == LAYERS[-1] and K == KS[-1]: keep_sheet = S
    if shown < a.sheet and keep_sheet is not None:   # photo | PCA RGB of the last layer | spots + hand
        Z3 = PCA(3, random_state=0).fit_transform(G[LAYERS[-1]].reshape(-1, DIM)[torch.from_numpy(idx).to(dev)].cpu().numpy()); Z3 = (Z3 - np.percentile(Z3, 2, 0)) / (np.percentile(Z3, 98, 0) - np.percentile(Z3, 2, 0) + 1e-6)
        rgb = np.ones((h8 * w8, 3), np.float32); rgb[idx] = Z3.clip(0, 1); rgb = rgb.reshape(h8, w8, 3)
        P = (np.stack([cv2.warpAffine(rgb[..., c], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for c in range(3)], -1) * 255).astype(np.uint8); P[wing == 0] = 255
        s = 700 / W; f_ = lambda x: cv2.resize(x, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        O = f_(I).copy(); m = cv2.resize(keep_sheet, (O.shape[1], O.shape[0]), interpolation=cv2.INTER_NEAREST) > 0; O[m] = (0.45 * O[m] + [140, 0, 140]).astype(np.uint8)
        if Gt is not None: cs, _ = cv2.findContours(cv2.resize(Gt, (O.shape[1], O.shape[0]), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(O, cs, -1, (0, 170, 0), 2)
        row = np.hstack([f_(I), f_(P), O]); cv2.putText(row, fn[:40], (6, 22), 0, 0.6, (0, 0, 0), 2)
        Image.fromarray(row).save(os.path.join(a.out, f'spots_tokens_{os.path.splitext(fn)[0]}.jpg'), quality=88); shown += 1
    rows.append(r); del G; torch.cuda.empty_cache(); print(f'{n + 1}/{len(files)} {fn} {time.time() - t0:.0f}s', flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('image', 'taxon'), k))
with open(os.path.join(a.out, 'per_wing.tsv'), 'w') as fh:
    fh.write('\t'.join(keys) + '\n'); [fh.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n') for r in rows]
LAB = {r_[0]: r_[1] for r_ in csv.reader(open(a.labels), delimiter='\t') if len(r_) > 1 and not r_[0].startswith('#')} if a.labels else {}
from sklearn.metrics import roc_auc_score
for L in LAYERS:
    for K in KS:
        tag = f'L{L}_K{K}'; line = f'{tag:8s}'
        iv = [r[f'{tag}_iou'] for r in rows if f'{tag}_iou' in r]
        if iv: line += f' | IoU vs hand {np.mean(iv):.3f} (n={len(iv)})'
        for z in ['preapical', 'central', 'spot_frac']:
            tm = {}
            for r in rows: tm.setdefault(r['taxon'], []).append(r[f'{tag}_{z}'])
            lt = [t for t in tm if t in LAB]; y = np.array([LAB[t] == a.positive for t in lt])
            if len(set(y)) == 2:
                sc = np.array([np.mean(tm[t]) for t in lt]); A = roc_auc_score(y, sc); rng = np.random.default_rng(0); nl = np.percentile([roc_auc_score(rng.permutation(y), sc) for _ in range(5000)], 95)
                line += f' | {z} AUC {A:.3f} (null95 {nl:.3f})'
        log(line)
log('rule: spot = clusters darker (mean L*) than the lightest membrane cluster by >= dl; no annotation used. Labels: preapical band present/absent (literature).')
