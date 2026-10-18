"""Training-free wing veins from DINOv3 patch tokens, scored against hand outlines (17 Oct 2026, AVD: 'segment the veins with the
PCA / high-pass PCA as a first attempt; test it on the spotted wings too').

Per image, inside the wing outline (plain-background segmentation, no annotation used):
  tokens on an 8-px grid (4 passes shifted by half a patch, native resolution, overlapping tiles)
  -> variant 'raw' = token, 'hp' = token minus its in-wing neighbourhood mean (13x13 cells of 8 px = the 7x7-patch window)
  -> PCA (16) fitted on that wing -> k-means K -> the vein cluster is chosen WITHOUT the hand outlines: the cluster whose pixels
     have the highest mean dark-ridge response (grey black-top-hat). 'oracle' = the best cluster by IoU (upper bound only: it uses
     the hand outlines, it says whether a vein cluster exists at all).
Optional comparison: DINOMask held-out probability files (--probs_dirs name=dir,...): vein = argmax not background.
Scores at native resolution vs the union of all hand-drawn veins: IoU, recall@2px, precision@2px, and recall of hand vein pixels
INSIDE dark wing spots vs outside. Spots are measured, not taken from the literature: membrane pixels (wing minus dilated hand
veins) at least --spot_dl L* darker than that wing's median membrane, smoothed, components >= 0.3% of the wing.
Usage: see --help. Writes per_image.tsv + summary.txt + a sheet of the first --sheet images."""
import argparse, json, os, sys, time
import numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
ap = argparse.ArgumentParser()
ap.add_argument('--coco', required=True, help='hand vein outlines (all categories = vein)'); ap.add_argument('--images', required=True)
ap.add_argument('--image_list_dir', default='', help='score only images that have a file <stem>.npz here (e.g. a DINOMask held-out probs dir)')
ap.add_argument('--probs_dirs', default='', help='name=dir,...  DINOMask held-out probs/<stem>.npz to score the same way')
ap.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m'); ap.add_argument('--layers', default='12,18')
ap.add_argument('--ks', default='3,4,5'); ap.add_argument('--tile', type=int, default=1024); ap.add_argument('--spot_dl', type=float, default=12)
ap.add_argument('--drop_posterior_margin', type=int, default=0, help='v1b: px band along the POSTERIOR outline chain (base->apex, lower side) removed from the rule prediction: the fringe is not a vein; the anterior margin (costa) is kept'); ap.add_argument('--min_comp_frac', type=float, default=0, help='v1b: drop predicted vein components smaller than this fraction of the wing area'); ap.add_argument('--sheet', type=int, default=8); ap.add_argument('--out', required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); LAYERS = [int(x) for x in a.layers.split(',')]; KS = [int(x) for x in a.ks.split(',')]
LOG = open(os.path.join(a.out, 'summary.txt'), 'w')
def log(*s): print(*s); print(*s, file=LOG); LOG.flush()
d = json.load(open(a.coco)); by = {}
for an in d['annotations']: by.setdefault(an['image_id'], []).append(an)
ims = [im for im in d['images'] if not a.image_list_dir or os.path.exists(os.path.join(a.image_list_dir, os.path.splitext(im['file_name'])[0] + '.npz'))]
PD = dict(x.split('=', 1) for x in a.probs_dirs.split(',') if x)
log(f'wing_vein_unsupervised_eval_v1 {time.strftime("%Y-%m-%d %H:%M")} | {len(ims)} images | layers {LAYERS} | K {KS} | model {a.model} | compared: {list(PD)}')
from transformers import AutoModel
dev = 'cuda'; net = AutoModel.from_pretrained(a.model).to(dev).eval(); DIM = net.config.hidden_size; NREG = getattr(net.config, 'num_register_tokens', 4)
def grids(x):
    """x normalised HxWx3 (H,W multiples of 16) -> {L: [H/16, W/16, DIM]} with overlapping tiles."""
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
    """4 passes shifted by 8 px, interleaved -> {L: [H8, W8, DIM]} on an 8-px grid (cell (i,j) centred at (8i+8, 8j+8))."""
    x = (I.astype(np.float32) / 255 - MEAN) / STD; H, W = x.shape[:2]; H16, W16 = (H - 8) // 16 * 16, (W - 8) // 16 * 16
    G = {}
    for dy in (0, 8):
        for dx in (0, 8):
            g = grids(x[dy:dy + H16, dx:dx + W16])
            for L in LAYERS:
                if L not in G: G[L] = torch.zeros((2 * g[L].shape[0], 2 * g[L].shape[1], DIM), device=dev)
                G[L][dy // 8::2, dx // 8::2] = g[L]
    return G
def wing_mask(I):
    g = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY); s = 800 / max(g.shape); gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bg = np.median(np.concatenate([gs[:5].ravel(), gs[-5:].ravel(), gs[:, :5].ravel(), gs[:, -5:].ravel()]))
    fg = (np.abs(gs.astype(int) - bg) > 12).astype(np.uint8); fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(fg); fg = (lb == 1 + st[1:, 4].argmax()).astype(np.uint8)
    cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); fg = np.zeros_like(fg); cv2.drawContours(fg, cs, -1, 1, -1)
    return cv2.erode(cv2.resize(fg, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST), np.ones((13, 13), np.uint8))
K5 = np.ones((5, 5), np.uint8)
def score(p, g, wing, spot):
    p = p & wing; i_, u_ = (p & g).sum(), (p | g).sum(); pd, gd = cv2.dilate(p, K5), cv2.dilate(g, K5)
    gi, go = g & spot, g & (1 - spot)
    return dict(iou=i_ / max(u_, 1), rec=(g & pd).sum() / max(g.sum(), 1), prec=(p & gd).sum() / max(p.sum(), 1),
                rec_spot=(gi & pd).sum() / gi.sum() if gi.sum() > 200 else np.nan, rec_clear=(go & pd).sum() / max(go.sum(), 1))
rows = []; sheet = []
for n, im in enumerate(ims):
    t0 = time.time(); I = np.asarray(Image.open(os.path.join(a.images, im['file_name'])).convert('RGB')); H, W = I.shape[:2]
    g = np.zeros((H, W), np.uint8)
    for an in by.get(im['id'], []):
        for p in an['segmentation']: cv2.fillPoly(g, [np.array(p).reshape(-1, 2).astype(np.int32)], 1)
    wing = cv2.dilate(wing_mask(I), np.ones((13, 13), np.uint8));   # full outline incl. costa; hand veins NOT used here
    Lab = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * 100 / 255
    memb = wing & (1 - cv2.dilate(g, np.ones((15, 15), np.uint8))); med = np.median(Lab[memb > 0])
    sp = ((cv2.GaussianBlur(Lab, (0, 0), 6) < med - a.spot_dl) & (memb > 0)).astype(np.uint8); sp = cv2.morphologyEx(sp, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    nl, lb, st, _ = cv2.connectedComponentsWithStats(sp); keep = np.zeros(nl, np.uint8); keep[1:] = st[1:, 4] >= 0.003 * wing.sum(); sp = keep[lb]
    xs_ = np.nonzero(wing.any(0))[0]; sp[:, :int(xs_.min() + 0.15 * (xs_.max() - xs_.min()))] = 0   # v1b: wing-base sclerites are not spots
    spot = cv2.dilate(sp, np.ones((25, 25), np.uint8)) & wing   # veins running through a spot
    pband = np.zeros((H, W), np.uint8)
    if a.drop_posterior_margin:
        cs_, _ = cv2.findContours(wing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); Cc = max(cs_, key=cv2.contourArea).reshape(-1, 2); ib, ia = int(np.argmin(Cc[:, 0])), int(np.argmax(Cc[:, 0]))
        c1 = Cc[ib:ia + 1] if ib <= ia else np.concatenate([Cc[ib:], Cc[:ia + 1]]); c2 = Cc[ia:ib + 1] if ia <= ib else np.concatenate([Cc[ia:], Cc[:ib + 1]])
        post = c1 if c1[:, 1].mean() > c2[:, 1].mean() else c2
        cv2.polylines(pband, [post.astype(np.int32)], False, 1, thickness=2 * a.drop_posterior_margin + 1)
    spot_frac = sp.sum() / max(memb.sum(), 1)
    grey = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY); ridge = cv2.morphologyEx(cv2.GaussianBlur(grey, (0, 0), 1.5), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(np.float32)
    G = grid8(I); h8, w8 = next(iter(G.values())).shape[:2]
    ys, xs = (np.arange(h8) * 8 + 8).clip(0, H - 1), (np.arange(w8) * 8 + 8).clip(0, W - 1); wg = wing[np.ix_(ys, xs)].astype(np.float32)
    base = dict(image=im['file_name'], spot_frac=round(float(spot_frac), 4), vein_px_in_spots=int((g & spot).sum()))
    for L in LAYERS:
        T = G[L]; wt = torch.from_numpy(wg).to(dev)
        num = F.avg_pool2d((T * wt[..., None]).permute(2, 0, 1)[None], 13, 1, 6, count_include_pad=False)[0].permute(1, 2, 0)
        den = F.avg_pool2d(wt[None, None], 13, 1, 6, count_include_pad=False)[0, 0]; hp = T - num / den.clamp_min(1e-6)[..., None]
        idx = np.flatnonzero(wg.ravel() > 0)
        for var, X in (('raw', T), ('hp', hp)):
            Z = PCA(16, random_state=0).fit_transform(X.reshape(-1, DIM)[torch.from_numpy(idx).to(dev)].cpu().numpy())
            for K in KS:
                lab = np.full(h8 * w8, -1); lab[idx] = KMeans(K, n_init=4, random_state=0).fit_predict(Z); lab = lab.reshape(h8, w8)
                # cluster map to pixels: one-hot, bilinear (cell centres at 8i+8), argmax
                oh = np.stack([(lab == k).astype(np.float32) for k in range(K)], -1)
                M = np.float32([[8, 0, 8], [0, 8, 8]]); up = np.stack([cv2.warpAffine(oh[..., k], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for k in range(K)], -1)
                pix = up.argmax(-1)
                rk = [ridge[(pix == k) & (wing > 0)].mean() if ((pix == k) & (wing > 0)).any() else -1 for k in range(K)]
                rule = (pix == int(np.argmax(rk))).astype(np.uint8)
                if a.drop_posterior_margin: rule &= 1 - pband
                if a.min_comp_frac:
                    nl_, lb_, st_, _ = cv2.connectedComponentsWithStats(rule & wing); kp = np.zeros(nl_, np.uint8); kp[1:] = st_[1:, 4] >= a.min_comp_frac * wing.sum(); rule = kp[lb_]
                s_rule = score(rule, g, wing, spot); s_or = max((score((pix == k).astype(np.uint8), g, wing, spot) for k in range(K)), key=lambda s: s['iou'])
                rows.append({**base, 'method': f'{var}_L{L}_K{K}_rule', **s_rule}); rows.append({**base, 'method': f'{var}_L{L}_K{K}_oracle', **s_or})
                if var == 'hp' and L == LAYERS[-1] and K == KS[len(KS) // 2] and n < a.sheet: sheet.append((im['file_name'], I, g, rule, spot))
    stem = os.path.splitext(im['file_name'])[0]
    for name, dr in PD.items():
        f = os.path.join(dr, stem + '.npz')
        if os.path.exists(f):
            P = np.load(f)['probs'].astype(np.float32); v = (P.argmax(-1) > 0).astype(np.uint8); v = cv2.resize(v, (W, H), interpolation=cv2.INTER_NEAREST)
            rows.append({**base, 'method': f'DINOMask_{name}', **score(v, g, wing, spot)})
    print(f'{n + 1}/{len(ims)} {im["file_name"]} spot {spot_frac:.3f} {time.time() - t0:.0f}s', flush=True)
    del G; torch.cuda.empty_cache()
cols = ['image', 'method', 'spot_frac', 'vein_px_in_spots', 'iou', 'rec', 'prec', 'rec_spot', 'rec_clear']
with open(os.path.join(a.out, 'per_image.tsv'), 'w') as fh:
    fh.write('\t'.join(cols) + '\n'); [fh.write('\t'.join(str(round(r[c], 4)) if isinstance(r[c], float) else str(r[c]) for c in cols) + '\n') for r in rows]
spotted = sorted({r['image'] for r in rows if r['vein_px_in_spots'] > 2000}); log(f'images with veins running through a dark spot (>2000 hand vein px inside a spot): {len(spotted)} of {len(ims)}')
log(f'{"method":26s} {"IoU":>5s} {"rec@2":>6s} {"prec@2":>6s} | spotted wings: {"rec in spot":>11s} {"rec outside":>11s}  (means over images)')
for m in dict.fromkeys(r['method'] for r in rows):
    R = [r for r in rows if r['method'] == m]; S = [r for r in R if r['image'] in spotted]
    log(f'{m:26s} {np.mean([r["iou"] for r in R]):5.3f} {np.mean([r["rec"] for r in R]):6.3f} {np.mean([r["prec"] for r in R]):6.3f} | '
        f'{np.nanmean([r["rec_spot"] for r in S]) if S else np.nan:11.3f} {np.mean([r["rec_clear"] for r in S]) if S else np.nan:11.3f}')
log('rule = vein cluster picked by dark-ridge response (no annotation used); oracle = best cluster by IoU against the hand outlines (upper bound, NOT a usable method).')
if sheet:   # photo | hand veins (green) + spots (blue outline) | rule cluster (orange)
    tiles = []
    for fn, I, g, r, sp in sheet:
        s = 900 / I.shape[1]; f = lambda m: cv2.resize(m, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        A = cv2.resize(I, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); B = A.copy(); B[f(g) > 0] = (0.4 * B[f(g) > 0] + 0.6 * np.array([40, 200, 40])).astype(np.uint8)
        cs, _ = cv2.findContours(f(sp), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(B, cs, -1, (40, 90, 255), 2)
        C = A.copy(); C[f(r) > 0] = (0.4 * C[f(r) > 0] + 0.6 * np.array([255, 140, 0])).astype(np.uint8)
        row = np.hstack([A, B, C]); cv2.putText(row, fn[:45], (8, 28), 0, 0.8, (0, 0, 0), 2); tiles.append(row)
    w = max(t.shape[1] for t in tiles); tiles = [cv2.copyMakeBorder(t, 0, 8, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    Image.fromarray(np.vstack(tiles)).save(os.path.join(a.out, 'sheet_photo_hand+spots_hpPCA-rule.jpg'), quality=88)
log('done')
