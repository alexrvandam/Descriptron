#!/usr/bin/env python3
"""High-resolution texture maps on ant head montages.
 1. TEACHER: 6-class + punctate classifier trained on literature-labelled data (unified_tokens_v2 npz, 1000 px/mm, z-scored grey).
    Dense teacher maps on each ant head at 1000 px/mm: 4 half-patch-shifted passes, bilinear upsampling, guided filter.
 2. STUDENT: tokens at --hi px/mm (montages are natively ~2000-2700 px/mm) with the teacher's smoothed probabilities as soft
    targets; MLP head trained leave-one-SPECIES-out (folds by species) -> held-out species predicted at --hi px/mm, same smoothing.
 3. Per head: regions dorsum (8 bands posterior->anterior), sides, occiput, occipital corners -> class probabilities (6 fine +
    merged), punctate, and ridge geometry from a Sato ridge mask at --hi px/mm (density, longitudinal share, coherence, cell size).
Usage: texture_hires_v3.py teacher_tokens.npz images_dir predictions_v2.json scales.csv out_dir [--hi 2000]
25 Oct 2026: --teacher_only (teacher maps + tables only), --cap_category/--dorsum_category (other taxa; '' = whole capsule),
--width_mm, polygon COCO masks. Oxytelus: --teacher_only --cap_category head --dorsum_category ''"""
import argparse, sys, json, csv, re, os, numpy as np, torch, cv2, torch.nn as nn
from PIL import Image
from pycocotools import mask as mu
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.filters import sato
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser(); [ap.add_argument(x) for x in ('teacher', 'images', 'preds', 'scales', 'out')]
ap.add_argument('--lo', type=float, default=1000); ap.add_argument('--hi', type=float, default=2000); ap.add_argument('--bands', type=int, default=8)
ap.add_argument('--folds', type=int, default=5); ap.add_argument('--teacher_only', action='store_true', help='teacher maps only (no student; for photos below ~1000 px/mm)')
ap.add_argument('--cap_category', default='head_capsule'); ap.add_argument('--dorsum_category', default='cephalic_dorsum', help="'' = whole capsule")
ap.add_argument('--width_mm', type=float, nargs=2, default=(0.4, 1.6), help='capsule width range kept (mm)'); ap.add_argument('--per_head', type=int, default=1500); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
os.makedirs(f'{a.out}/maps', exist_ok=True); dev = 'cuda'; rng = np.random.default_rng(0)
z = np.load(a.teacher); CL = list(z['classes']); K = len(CL); Xt = z['X'].astype(np.float32)
sc = StandardScaler().fit(Xt); pc = PCA(64, random_state=0).fit(sc.transform(Xt)); Zt = pc.transform(sc.transform(Xt))
T_cls = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(Zt, z['y'])
mp = z['punct'] >= 0; T_pun = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(Zt[mp], z['punct'][mp])
def teach(T):
    Zq = pc.transform(sc.transform(T)); P = np.zeros((len(T), K)); P[:, T_cls.classes_] = T_cls.predict_proba(Zq); return np.concatenate([P, T_pun.predict_proba(Zq)[:, 1:]], 1)
net = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m').to(dev).eval(); NREG = net.config.num_register_tokens; D = net.config.hidden_size
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
def toks(g):
    x = torch.tensor(((np.stack([g] * 3, -1) - MEAN) / STD).transpose(2, 0, 1)[None], dtype=torch.float32).to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        return net(pixel_values=x, output_hidden_states=True).hidden_states[19][0, 1 + NREG:].float().cpu().numpy()
def rz(a_, size, interp):
    """cv2.resize for any number of channels (INTER_AREA/LINEAR assert on >4 channels in OpenCV 5)."""
    return np.stack([cv2.resize(np.ascontiguousarray(a_[..., c]), size, interpolation=interp) for c in range(a_.shape[-1])], -1)
def gfilter(I, p, r=8, eps=1e-3):
    mean = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1)); mI = mean(I); vI = mean(I * I) - mI * mI; out = np.empty_like(p)
    for k in range(p.shape[2]):
        q = p[..., k]; mq = mean(q); a_ = (mean(I * q) - mI * mq) / (vI + eps); b_ = mq - a_ * mI; out[..., k] = mean(a_) * I + mean(b_)
    return out
def dense(g, fn, C):
    """4 half-patch-shifted passes of fn(tokens)->(N,C), bilinear to pixels, averaged, guided-filtered with the grey image."""
    H, W = g.shape; acc = np.zeros((H, W, C), np.float32)
    for dx, dy in ((0, 0), (8, 0), (0, 8), (8, 8)):
        x = np.pad(g, ((dy, 16 - dy if dy else 0), (dx, 16 - dx if dx else 0)), mode='edge'); gh, gw = x.shape[0] // 16, x.shape[1] // 16
        pr = fn(toks(x[:gh * 16, :gw * 16])).reshape(gh, gw, C).astype(np.float32)
        acc += rz(pr, (gw * 16, gh * 16), cv2.INTER_LINEAR)[dy:dy + H, dx:dx + W]
    return gfilter(g.astype(np.float32), acc / 4)
def dec(x, hw=None):
    if isinstance(x['segmentation'], list):   # polygons (25 Oct 2026)
        m = np.zeros(hw, np.uint8)
        for q in x['segmentation']: cv2.fillPoly(m, [np.array(q).reshape(-1, 2).round().astype(np.int32)], 1)
        return m.astype(bool)
    s = dict(x['segmentation']); s['counts'] = s['counts'].encode() if isinstance(s['counts'], str) else s['counts']; return mu.decode(s).astype(bool)
d = json.load(open(a.preds)); cid = {c['name']: c['id'] for c in d['categories']}; by = {}
for x in d['annotations']: by.setdefault(x['image_id'], {})[x['category_id']] = x
S = {r['file']: float(r['px_per_mm']) for r in csv.DictReader(open(a.scales)) if r['px_per_mm']}
heads = []
for im in d['images']:
    f = im['file_name']; A = by.get(im['id'], {})
    hw = (im['height'], im['width']); dec_ = lambda k: dec(A[cid[k]], hw)
    if f not in S or cid.get(a.cap_category) not in A or (a.dorsum_category and cid.get(a.dorsum_category) not in A): continue
    cap = dec_(a.cap_category); ys, xs = np.where(cap); x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if not a.width_mm[0] <= (x1 - x0) / S[f] <= a.width_mm[1]: continue
    R = {'dorsum': (dec_(a.dorsum_category) & cap) if a.dorsum_category else cap.copy()}
    for k, nm in (('occiput', 'occiput'), ('occipital_corner', 'corners')):
        R[nm] = (dec_(k) & cap & ~R['dorsum']) if cid.get(k) in A else np.zeros_like(cap)
    sid = cap & ~R['dorsum'] & ~R['occiput'] & ~R['corners']
    for k in ('clypeus', 'frontal_triangle'):
        if cid.get(k) in A: sid &= ~dec_(k)
    R['sides'] = sid; m_ = re.search(r'FHG\d+', f)
    heads.append(dict(f=f, sp=m_.group() if m_ else '?', ppm=S[f], box=(x0, x1, y0, y1), cap=cap[y0:y1, x0:x1], R={k: v[y0:y1, x0:x1] for k, v in R.items()}))
    if a.limit and len(heads) >= a.limit: break
print('heads', len(heads), flush=True)
def grey_at(h, ppmm):
    x0, x1, y0, y1 = h['box']; g = np.asarray(Image.open(os.path.join(a.images, h['f'])).convert('L'), np.float32)[y0:y1, x0:x1] / 255
    s = ppmm / h['ppm']; nh, nw = int(g.shape[0] * s) // 16 * 16, int(g.shape[1] * s) // 16 * 16
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    m = cv2.resize(h['cap'].astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
    g = np.clip((g - g[m].mean()) / (g[m].std() + 1e-6) * 0.18 + 0.5, 0, 1).astype(np.float32); return g, m
# ---- 1. teacher maps (lo) and student training samples (hi)
samples = {}
for i, h in enumerate(heads):
    g, m = grey_at(h, a.lo); Pt = dense(g, teach, K + 1)
    if a.teacher_only: h['teacher'] = Pt; h['g_lo'] = g; print(f'teacher {i + 1}/{len(heads)}', flush=True); continue
    gh_, m_hi = grey_at(h, a.hi); gh2, gw2 = gh_.shape[0] // 16, gh_.shape[1] // 16
    T = toks(gh_[:gh2 * 16, :gw2 * 16]).reshape(gh2, gw2, -1)
    tgt = rz(Pt, (gw2, gh2), cv2.INTER_AREA)
    mg = cv2.resize(m_hi[:gh2 * 16, :gw2 * 16].astype(np.float32), (gw2, gh2), interpolation=cv2.INTER_AREA) >= 0.6
    sel = rng.permutation(np.where(mg.ravel())[0])[:a.per_head]
    samples[h['f']] = (T.reshape(-1, D)[sel].astype(np.float16), tgt.reshape(-1, K + 1)[sel].astype(np.float32))
    h['teacher'] = Pt; h['g_lo'] = g
    print(f'teacher {i + 1}/{len(heads)} {h["sp"]}', flush=True)
# ---- 3. profiles + ridge geometry at hi + figures + COCO
MERGE = {'smooth_or_superficial': ['smooth', 'superficial'], 'longitudinal_any': ['longitudinal_rugose', 'longitudinal_crossmeshes']}
PAL = np.array([[235, 235, 235], [170, 170, 170], [255, 200, 0], [60, 90, 255], [140, 60, 200], [230, 30, 30]], np.uint8)
coco = {'images': [], 'annotations': [], 'categories': [{'id': i + 1, 'name': 'texture_' + c} for i, c in enumerate(CL)]}; rows = []
def ridge_geom(g, reg):
    if reg.sum() < 500: return {}
    r = sato(g, sigmas=[6, 9, 12], black_ridges=True); thr = np.percentile(r[reg], 70); rid = remove_small_objects((r > thr) & reg, 100); sk = skeletonize(rid)
    mb = cv2.GaussianBlur(sk.astype(np.float32), (0, 0), 2.0); Ix, Iy = cv2.Sobel(mb, cv2.CV_32F, 1, 0), cv2.Sobel(mb, cv2.CV_32F, 0, 1)
    J = [cv2.GaussianBlur(v, (0, 0), 5.0) for v in (Ix * Ix, Iy * Iy, Ix * Iy)]; ang = 0.5 * np.arctan2(2 * J[2], J[0] - J[1])[sk & reg]
    cells = label(reg & ~rid, connectivity=1); props = [p for p in regionprops(cells) if p.area >= 40]
    um2 = (1000 / a.hi) ** 2
    return dict(ridge_density_per_mm=round(float(sk[reg].sum() / reg.sum() * a.hi), 2), longit_share=round(float((np.abs(np.cos(ang)) > np.cos(np.deg2rad(25))).mean()), 3) if len(ang) else '',
                coherence=round(float(np.hypot(np.cos(2 * ang).mean(), np.sin(2 * ang).mean())), 3) if len(ang) else '',
                cell_area_um2_median=round(float(np.median([p.area for p in props]) * um2), 1) if props else '')
def finish(h):
        P = h['student']; g = h['g_hi']; H, W = g.shape; rs = lambda m: cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        Rr = {k: rs(v) for k, v in h['R'].items()}; capr = rs(h['cap']); r = dict(file=h['f'], species=h['sp'], px_per_mm=h['ppm'])
        for nm, reg in Rr.items():
            if reg.sum() < 50: continue
            for c in range(K): r[f'{nm}_{CL[c]}'] = round(float(P[..., c][reg].mean()), 3)
            for mn, parts in MERGE.items(): r[f'{nm}_{mn}'] = round(float(sum(P[..., CL.index(p)][reg].mean() for p in parts)), 3)
            r[f'{nm}_punctate'] = round(float(P[..., K][reg].mean()), 3)
            for k_, v in ridge_geom(g, reg).items(): r[f'{nm}_{k_}'] = v
        dm = Rr['dorsum']; rows_ = np.where(dm.any(1))[0]; e = np.linspace(rows_.min(), rows_.max() + 1, a.bands + 1).astype(int); dom = []
        for b in range(a.bands):
            bm = np.zeros_like(dm); bm[e[b]:e[b + 1]] = dm[e[b]:e[b + 1]]
            if bm.sum() < 20: dom.append(-1); continue
            pr = P[bm][:, :K].mean(0); dom.append(int(pr.argmax()))
            for c in range(K): r[f'band{b + 1}_{CL[c]}'] = round(float(pr[c]), 3)
            for k_, v in ridge_geom(g, bm).items(): r[f'band{b + 1}_{k_}'] = v
        rows.append(r)
        cls = P[..., :K].argmax(-1); cm = PAL[cls]; cm[~capr] = 0
        for nm, col in (('dorsum', (0, 255, 255)), ('occiput', (255, 0, 255)), ('corners', (0, 255, 0))):
            cn, _ = cv2.findContours(Rr[nm].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(cm, cn, -1, col, 3)
        for y_ in e: cv2.line(cm, (0, int(y_)), (W, int(y_)), (0, 255, 255), 1)
        tl = cv2.resize(h['teacher'][..., :K].argmax(-1).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST); tm = PAL[tl]; tm[~capr] = 0
        x0, x1, y0, y1 = h['box']; rgb = cv2.resize(np.asarray(Image.open(os.path.join(a.images, h['f'])).convert('RGB'))[y0:y1, x0:x1], (W, H))
        sheet = np.hstack([rgb, tm, cm]); s_ = 520 / H; sheet = cv2.resize(sheet, (int(sheet.shape[1] * s_), 520), interpolation=cv2.INTER_AREA)
        Image.fromarray(sheet).save(f'{a.out}/maps/{os.path.splitext(h["f"])[0]}.jpg', quality=85)
        full = np.zeros((h['cap'].shape[0], h['cap'].shape[1]), np.uint8); clsf = cv2.resize(cls.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        im_ = [i for i in d['images'] if i['file_name'] == h['f']][0]; ii = len(coco['images']) + 1; coco['images'].append({'id': ii, 'file_name': h['f'], 'width': im_['width'], 'height': im_['height']})
        for c in range(K):
            m = np.zeros((im_['height'], im_['width']), np.uint8); m[y0:y1, x0:x1] = (clsf == c) & h['cap']
            if not m.any(): continue
            e_ = mu.encode(np.asfortranarray(m)); e_['counts'] = e_['counts'].decode()
            coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': ii, 'category_id': c + 1, 'segmentation': e_, 'area': int(m.sum()), 'bbox': [int(v) for v in mu.toBbox(e_)], 'iscrowd': 0})

# ---- 2. student, leave-species-out folds
if a.teacher_only:   # 25 Oct 2026: maps and tables from the teacher at --lo px/mm
    a.hi = a.lo
    for h in heads: h['student'] = h['teacher']; h['g_hi'] = h['g_lo']; finish(h)
    json.dump(coco, open(f'{a.out}/texture_masks_hires.json', 'w'))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f'{a.out}/texture_hires_table.tsv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter='\t'); w.writeheader(); w.writerows(rows)
    np.savez_compressed(f'{a.out}/teacher_maps.npz', **{os.path.splitext(h['f'])[0]: h['teacher'].astype(np.float16) for h in heads})
    json.dump({'mode': 'teacher_only', 'px_per_mm': a.lo, 'heads': len(rows), 'classes': CL}, open(f'{a.out}/summary.json', 'w'), indent=1)
    print('done (teacher only)', len(rows), 'heads'); sys.exit(0)
spp = sorted({h['sp'] for h in heads}); fold = {s: k % a.folds for k, s in enumerate(rng.permutation(spp))}
class Head(nn.Module):
    def __init__(s): super().__init__(); s.net = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, K + 1))
    def forward(s, t): return s.net(t)
def train(files):
    X = torch.from_numpy(np.concatenate([samples[f][0] for f in files]).astype(np.float32)); Yt = torch.from_numpy(np.concatenate([samples[f][1] for f in files]))
    yc = Yt[:, :K] / Yt[:, :K].sum(1, keepdim=True).clamp_min(1e-6); yp = Yt[:, K]
    hd = Head().to(dev); opt = torch.optim.AdamW(hd.parameters(), lr=1e-3, weight_decay=0.01)
    for ep in range(25):
        perm = torch.randperm(len(X))
        for j in range(0, len(X), 4096):
            idx = perm[j:j + 4096]; o = hd(X[idx].to(dev))
            loss = -(yc[idx].to(dev) * torch.log_softmax(o[:, :K], -1)).sum(1).mean() + nn.functional.binary_cross_entropy_with_logits(o[:, K], yp[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    hd.eval()
    def fn(T):
        with torch.no_grad(): o = hd(torch.from_numpy(T).to(dev)); return torch.cat([torch.softmax(o[:, :K], -1), torch.sigmoid(o[:, K:])], 1).cpu().numpy()
    return fn
agree = []
for k in range(a.folds):
    te = [h for h in heads if fold[h['sp']] == k]
    if not te: continue
    fn = train([h['f'] for h in heads if fold[h['sp']] != k])
    for h in te:
        g, m = grey_at(h, a.hi); h['student'] = dense(g, fn, K + 1); h['g_hi'] = g
        tlo = rz(h['teacher'], (g.shape[1], g.shape[0]), cv2.INTER_LINEAR)
        agree.append(float((tlo[..., :K].argmax(-1) == h['student'][..., :K].argmax(-1))[m].mean())); del tlo
        finish(h); h.pop('student'); h.pop('g_hi'); h.pop('teacher'); h.pop('g_lo', None)   # stream: free the big maps
        print(f'  done {h["sp"]} {h["f"][:40]}', flush=True)
    print(f'fold {k}: {len(te)} held-out heads', flush=True)
print('student vs teacher per-pixel class agreement (held-out species): median %.2f' % np.median(agree))
json.dump(coco, open(f'{a.out}/texture_masks_hires.json', 'w'))
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(f'{a.out}/texture_hires_table.tsv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys, delimiter='\t'); w.writeheader(); w.writerows(rows)
json.dump({'student_teacher_agreement_median': float(np.median(agree)), 'heads': len(rows), 'classes': CL}, open(f'{a.out}/summary.json', 'w'), indent=1)
print('done', len(rows), 'heads; columns', len(keys))
