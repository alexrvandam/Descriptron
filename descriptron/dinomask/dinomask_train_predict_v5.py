#!/usr/bin/env python3
"""
DINOMask training + prediction, v5 (Descriptron packaged copy of patch_classifier_coco_v5.py, 23 Oct 2026; see README.md
beside this file). Code identical to the research copy. Runs saved with the default --hidden 512 can also be read by
dinomask_predict_rot4_v1.py. Original header:

patch_classifier_coco_v1.py - per-patch DINOv3 part classifier on a hand-annotated polygon set (weevils)
v5 (21 Oct 2026, copy of v4; v4 unchanged): --cache_dir (tokens computed once per image/orientation and reused),
training array built in place (v4 held every token twice), --bg_keep (keep only this share of background
patches far from the specimen; background near it is always kept), --lr / --hidden / --wd, --curve_json
(per-epoch train loss, held-out loss and per-class held-out patch IoU; ignored cells excluded), --grid
(several lr:hidden settings trained on the same loaded tokens, curves only, then exit). Defaults = v4.
======================================================================================================
Input: a VIA project (polygons with a 'sclerite' attribute) or COCO json, an image dir, a category map
(raw name -> merged name; unmapped names kept as-is), a held-out image list (by species, from the filename
prefix). Each image -> DINOv3 ViT-S tokens at --size (patch 16); each patch labelled by the majority class
of its pixels (0 = background). Head: LayerNorm -> MLP on the token (+ positional bias). Reports held-out
per-patch accuracy, per-class IoU (patch level and after nearest upsampling), and writes overlay PNGs
(ground truth | prediction) for the held-out images so the result can be judged by eye.
"""
import argparse, json, os, re, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, cv2
from PIL import Image
from transformers import AutoModel
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
ap = argparse.ArgumentParser(); ap.add_argument('--via', default=''); ap.add_argument('--coco', default=''); ap.add_argument('--fold_lr', action='store_true'); ap.add_argument('--drop', default='Select Label,line_keypoints'); ap.add_argument('--species_regex', default=r'([A-Za-z]+?)_'); ap.add_argument('--test_list', default='', help='comma list of image filenames to hold out (overrides the species split)'); ap.add_argument('--exclude_regex', default='', help='drop images whose name matches'); ap.add_argument('--model', default='facebook/dinov3-vits16-pretrain-lvd1689m'); ap.add_argument('--predict_dir', default='', help='unlabelled images to predict after training on ALL labelled images'); ap.add_argument('--predict_list', default='', help='v3: text file, one image path per line (used instead of --predict_dir)'); ap.add_argument('--aug', default='', help='training-only copies: comma list of rot90,rot180,rot270,hflip'); ap.add_argument('--n_train', type=int, default=0, help='use only the first N training images (learning curve)'); ap.add_argument('--min_area_frac', type=float, default=0.0005, help='a class covering less of the image than this is reported absent'); ap.add_argument('--save_probs', action='store_true', help='held-out images: per-pixel class probabilities from 4 half-patch-shifted passes, smooth upsampling -> probs/<stem>.npz'); ap.add_argument('--images', required=True); ap.add_argument('--catmap', default=''); ap.add_argument('--containers', default='auto', help="v2: classes that ENCLOSE other classes (e.g. head_capsule around cephalic_dorsum) get their own yes/no head so their mask keeps its full extent; 'auto' = detect from annotations, 'none' = v1 behaviour, or a comma list"); ap.add_argument('--contain_frac', type=float, default=0.9); ap.add_argument('--fg_mask', default='none', choices=['none', 'border'], help='prediction only: border = specimen foreground from the background colour (fg_mask_v1.py); every foreground pixel gets a body class, everything else background'); ap.add_argument('--out', required=True)
ap.add_argument('--partial_bg', default='', help='v4b: comma list of the full vocabulary; on an image where one of these is NOT drawn, background cells near the specimen (drawn-parts box grown 40%%) are IGNORED in the loss instead of taught as background'); ap.add_argument('--size', type=int, default=560, help='longest side in px; 0 = the image native resolution'); ap.add_argument('--test_frac', type=float, default=0.33); ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--min_count', type=int, default=4)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--hidden', type=int, default=512); ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--cache_dir', default=''); ap.add_argument('--bg_keep', type=float, default=1.0); ap.add_argument('--curve_json', default=''); ap.add_argument('--grid', default='', help='v5: comma list of lr:hidden, e.g. 3e-4:512,1e-3:1024 -> curves only')
CM_FN = None
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); G = a.size // 16; dev = 'cuda' if torch.cuda.is_available() else 'cpu'; rng = np.random.default_rng(a.seed)
cm = {}
if a.via:
    d = json.load(open(a.via)); meta = d.get('_via_img_metadata', d)
else:   # COCO -> the same (filename, regions) structure, polygons only (RLE decoded to the largest contour)
    from pycocotools import mask as mu
    d = json.load(open(a.coco)); cid = {c['id']: c['name'] for c in d['categories']}; byimg = {}
    for an in d['annotations']:
        if 'segmentation' not in an or not an['segmentation']: continue      # keypoint / line annotations have no polygon
        nm = cid.get(an['category_id'], ''); seg = an['segmentation']; polys = []
        if isinstance(seg, dict):
            s_ = dict(seg); s_['counts'] = s_['counts'].encode() if isinstance(s_['counts'], str) else s_['counts']; m_ = mu.decode(s_ if isinstance(s_['counts'], bytes) else mu.frPyObjects(s_, s_['size'][0], s_['size'][1]))
            cs, _ = cv2.findContours(m_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); polys = [c_.reshape(-1, 2) for c_ in cs if len(c_) >= 3]
        else: polys = [np.asarray(p_, float).reshape(-1, 2) for p_ in (seg if isinstance(seg[0], list) else [seg]) if len(p_) >= 6]
        for p_ in polys: byimg.setdefault(an['image_id'], []).append({'shape_attributes': {'name': 'polygon', 'all_points_x': p_[:, 0].tolist(), 'all_points_y': p_[:, 1].tolist()}, 'region_attributes': {'sclerite': nm}})
    meta = {str(im['id']): {'filename': im['file_name'], 'regions': byimg.get(im['id'], [])} for im in d['images']}
drop = set(x for x in a.drop.split(',') if x)
if a.catmap:
    m = json.load(open(a.catmap))
    if isinstance(m, dict) and 'categories' in m:          # a COCO category list of MERGED names (left/right folded): map raw -> merged
        merged = {c['name'] for c in m['categories']}; fix = {'elytra': 'elytron', 'metasterunum': 'metasternum', 'mouth_parts': 'mouthparts'}
        def to_merged(raw):
            b = re.sub(r'^(left|right)_', '', raw); b = fix.get(b, b)
            if b in merged: return b
            for c in merged:
                if b.rstrip('s') == c.rstrip('s'): return c
            return b
        cm = {}; CM_FN = to_merged
    else:
        cm = m if isinstance(m, dict) and all(isinstance(v, str) for v in m.values()) else {raw: merged for merged, raws in m.items() for raw in (raws if isinstance(raws, list) else [raws])} if isinstance(m, dict) else {}
items = []
for k, v in meta.items():
    fn = v.get('filename', ''); p = Path(a.images) / fn
    if not p.exists(): continue
    def cname(raw):
        if raw in drop: return 'background'
        n = CM_FN(raw) if CM_FN else cm.get(raw, raw)
        return re.sub(r'^(left|right)_', '', n) if a.fold_lr else n
    polys = [(cname(r['region_attributes'].get('sclerite', '')), np.stack([r['shape_attributes']['all_points_x'], r['shape_attributes']['all_points_y']], 1)) for r in v.get('regions', []) if r['shape_attributes'].get('name') == 'polygon' and len(r['shape_attributes']['all_points_x']) >= 3]
    items.append((fn, p, polys))
from collections import Counter
cnt = Counter(c for _, _, polys in items for c, _ in polys if c != 'background'); classes = ['background'] + sorted(c for c, n in cnt.items() if n >= a.min_count); cidx = {c: i for i, c in enumerate(classes)}
def sp_of(fn):
    m_ = re.match(a.species_regex, fn); return m_.group(1) if m_ else fn
if a.exclude_regex: items = [it for it in items if not re.search(a.exclude_regex, it[0])]
species = sorted(set(sp_of(fn) for fn, _, _ in items)); rng.shuffle(species); test_sp = set(species[:max(1, int(round(a.test_frac * len(species))))])
if a.test_list: tl = set(a.test_list.split(',')); test_sp = set(sp_of(fn) for fn, _, _ in items if fn in tl)
if (a.predict_dir or a.predict_list) and not a.test_list: test_sp = set()
# v2: container detection - class A encloses class B if, on images with both, >= contain_frac of B's area lies inside A (median over images)
def _raster(polys, names, H, W, sc):
    m = np.zeros((int(H * sc) + 1, int(W * sc) + 1), np.uint8)
    for c, pts in polys:
        if c in names: cv2.fillPoly(m, [(pts * sc).astype(np.int32)], 1)
    return m.astype(bool)
CONT = []
if a.containers == 'auto':
    frac = {}
    for fn, p, polys in items:
        if not polys: continue
        allp = np.concatenate([pt for _, pt in polys]); H_, W_ = allp[:, 1].max() + 2, allp[:, 0].max() + 2; sc = 256 / max(H_, W_)
        present = sorted({c for c, _ in polys if c in cidx and c != 'background'}); masks = {c: _raster(polys, {c}, H_, W_, sc) for c in present}
        for A in present:
            for B in present:
                if A != B and masks[B].sum() and masks[A].sum() > masks[B].sum(): frac.setdefault((A, B), []).append((masks[A] & masks[B]).sum() / masks[B].sum())
    CONT = sorted({A for (A, B), v in frac.items() if len(v) >= 2 and np.median(v) >= a.contain_frac})
    for A in CONT: print(f"  container: {A} encloses {sorted(B for (A2, B), v in frac.items() if A2 == A and len(v) >= 2 and np.median(v) >= a.contain_frac)}", flush=True)
elif a.containers != 'none': CONT = [c for c in a.containers.split(',') if c in cidx]
print(f"{len(items)} images, {len(classes)-1} classes with >= {a.min_count} polygons, {len(species)} species, held out: {sorted(test_sp)}", flush=True)
net = AutoModel.from_pretrained(a.model).to(dev).eval(); DIM = net.config.hidden_size; NREG = getattr(net.config, 'num_register_tokens', 4)
TILE, OV = 1536, 256; STEP = TILE - 2 * OV
def fwd(x):
    """v4: last-layer patch tokens of a normalised HxWx3 image (multiples of 16); overlapping TILE-px tiles, tile centres kept,
    so native-resolution images fit the GPU. Returns a float32 tensor [gh*gw, DIM] on the GPU (raster order)."""
    H, W = x.shape[:2]; gh, gw = H // 16, W // 16
    def one(t):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            return net(pixel_values=torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1)))[None].to(dev)).last_hidden_state[0, 1 + NREG:, :].float()
    if H <= TILE and W <= TILE: return one(x)
    out = torch.zeros((gh, gw, DIM), dtype=torch.float32, device=dev)
    for y0 in range(0, H, STEP):
        for x0 in range(0, W, STEP):
            a_, b_ = max(0, y0 - OV), max(0, x0 - OV); t = x[a_:min(H, y0 + STEP + OV), b_:min(W, x0 + STEP + OV)]; th, tw = t.shape[0] // 16, t.shape[1] // 16
            T = one(t[:th * 16, :tw * 16]).reshape(th, tw, DIM); ky, kx = (y0 - a_) // 16, (x0 - b_) // 16; ny, nx = min(STEP, H - y0) // 16, min(STEP, W - x0) // 16
            out[y0 // 16:y0 // 16 + ny, x0 // 16:x0 // 16 + nx] = T[ky:ky + ny, kx:kx + nx]
    return out.reshape(gh * gw, DIM)
(out / 'gt_fullres').mkdir(exist_ok=True)
def prep(p, polys, tf=''):
    if tf == 'hflip':   # v4 fix (15 Oct): a mirrored copy must swap left_/right_ names, else the model cannot learn the side
        polys = [((('right_' + c[5:]) if c.startswith('left_') else ('left_' + c[6:]) if c.startswith('right_') else c), pts) for c, pts in polys]
    im = Image.open(p).convert('RGB'); W, H = im.size; lab = np.zeros((H, W), np.int32)
    for c, pts in sorted(polys, key=lambda t: -cv2.contourArea(t[1].astype(np.float32))):     # big first, small parts drawn on top
        if c in cidx: cv2.fillPoly(lab, [pts.astype(np.int32)], cidx[c])
    if not tf: cv2.imwrite(str(out / 'gt_fullres' / (Path(p).stem + '.png')), lab.astype(np.uint8))
    _voc = [v for v in a.partial_bg.split(',') if v]
    if _voc and any(v not in {c for c, _ in polys} for v in _voc) and (lab > 0).any():   # v4b: incomplete image -> ignore background near the specimen
        _ys, _xs = np.nonzero(lab); _gy, _gx = int(0.4 * (np.ptp(_ys) + 1)), int(0.4 * (np.ptp(_xs) + 1)); _box = np.zeros_like(lab, bool)
        _box[max(0, _ys.min() - _gy):_ys.max() + _gy + 1, max(0, _xs.min() - _gx):_xs.max() + _gx + 1] = True; lab[_box & (lab == 0)] = 65535
    cont = np.zeros((H, W, max(1, len(CONT))), np.uint8)     # v2: each container's FULL filled mask (not overwritten by its parts)
    for j, c_ in enumerate(CONT):
        _cj = np.zeros((H, W), np.uint8)   # contiguous buffer: cont[..., j] is a strided view OpenCV 5 rejects (>1 container)
        for c, pts in polys:
            if c == c_: cv2.fillPoly(_cj, [pts.astype(np.int32)], 1)
        cont[..., j] = _cj
        if not tf and CONT: cv2.imwrite(str(out / 'gt_fullres' / (Path(p).stem + f'__{c_}.png')), cont[..., j] * 255)
    if tf:
        f_ = {'rot90': lambda x: np.rot90(x, 1), 'rot180': lambda x: np.rot90(x, 2), 'rot270': lambda x: np.rot90(x, 3), 'hflip': lambda x: x[:, ::-1]}[tf]
        im = Image.fromarray(np.ascontiguousarray(f_(np.asarray(im)))); lab = np.ascontiguousarray(f_(lab)); cont = np.ascontiguousarray(f_(cont)); W, H = im.size
    s = (a.size or max(W, H)) / max(W, H); nw, nh = max(16, int(round(W * s / 16)) * 16), max(16, int(round(H * s / 16)) * 16)   # keep aspect, multiple of 16
    x = (np.asarray(im.resize((nw, nh), Image.BILINEAR), np.float32) / 255 - MEAN) / STD; labg = cv2.resize(lab.astype(np.uint16), (nw // 16, nh // 16), interpolation=cv2.INTER_NEAREST).astype(np.int64); labg[labg == 65535] = -100   # v4b: ignored cells
    _cf = None
    if a.cache_dir:   # v5: tokens depend only on image, orientation, size and model
        import hashlib; _k = hashlib.md5(f"{Path(p).resolve()}|{tf}|{a.size}|{a.model}|{nw}x{nh}".encode()).hexdigest()[:16]
        _cf = Path(a.cache_dir) / f"{Path(p).stem[:60]}__{tf or 'orig'}__{_k}.npy"
    if _cf is not None and _cf.exists(): T = np.load(_cf)
    else:
        T = fwd(x).cpu().numpy().astype(np.float16)   # v4: tiled, stored half precision
        if _cf is not None: _cf.parent.mkdir(parents=True, exist_ok=True); np.save(_cf, T)
    prep.cont = (cv2.resize(cont.astype(np.float32), (nw // 16, nh // 16), interpolation=cv2.INTER_AREA).reshape(nh // 16, nw // 16, -1) >= 0.5)
    return T.reshape(nh // 16, nw // 16, -1), labg, (nw // 16, nh // 16)
def _keep(L):   # v5: all labelled / ignored cells, all background within the grown specimen box, bg_keep of the rest
    L = L.reshape(L.shape[0], -1) if L.ndim > 2 else L; k = np.ones(L.shape, bool)
    if a.bg_keep < 1.0 and (L > 0).any():
        ys, xs = np.nonzero(L > 0); gy, gx = int(0.4 * (np.ptp(ys) + 1)), int(0.4 * (np.ptp(xs) + 1)); box = np.zeros(L.shape, bool)
        box[max(0, ys.min() - gy):ys.max() + gy + 1, max(0, xs.min() - gx):xs.max() + gx + 1] = True
        far = (L == 0) & ~box; k[far] = rng.random(int(far.sum())) < a.bg_keep
    return k.reshape(-1) & (L.reshape(-1) != -100)
KEEP = {}
def _tr_entry(fn, T, L, grid):   # v5: training copies keep only the selected patches (2-D tokens, 1-D labels)
    k = _keep(L); KEEP[fn] = k; return (fn, T.reshape(-1, T.shape[-1])[k], L.reshape(-1)[k], grid, False)
data = []
ntr = 0; CONTG = {}
for fn, p, polys in items:
    is_te = sp_of(fn) in test_sp
    if not is_te and a.n_train and ntr >= a.n_train: continue
    T, L, grid = prep(p, polys); CONTG[fn] = prep.cont; data.append((fn, T, L, grid, True) if is_te else _tr_entry(fn, T, L, grid)); ntr += (not is_te); print(f"  {fn}: grid {grid}, {len(polys)} polygons", flush=True)
    if not is_te:
        for tf in [t for t in a.aug.split(',') if t]: T2, L2, g2 = prep(p, polys, tf); CONTG[fn + '@' + tf] = prep.cont; data.append(_tr_entry(fn + '@' + tf, T2, L2, g2))
tr = [x for x in data if not x[4]]; te = [x for x in data if x[4]]; K = len(classes)
_n = sum(len(l) for _, _, l, _, _ in tr); Xtr = np.empty((_n, DIM), np.float16); Ytr = np.empty(_n, np.int64); _o = 0
data = [x for x in data if x[4]]                     # keep only the held-out entries; training chunks are freed as copied
for j in range(len(tr)):
    fn, t, l, g, f = tr[j]; m = len(l); Xtr[_o:_o + m] = t; Ytr[_o:_o + m] = l; _o += m; tr[j] = (fn, None, None, g, f); del t
data = None
print(f"  training patches: {_n} (bg_keep {a.bg_keep}); {Xtr.nbytes / 1e9:.1f} GB", flush=True)
class Head(nn.Module):
    def __init__(s, d=None, h=None): d = d or DIM; h = h or a.hidden; super().__init__(); s.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Dropout(0.2), nn.Linear(h, K))
    def forward(s, t): return s.net(t)
Xt = torch.from_numpy(Xtr); Yt = torch.from_numpy(Ytr)
freq = np.bincount(Ytr[Ytr >= 0], minlength=K) / max(1, (Ytr >= 0).sum()); w = torch.tensor(1.0 / np.maximum(freq, 1e-4) ** 0.5, dtype=torch.float32).to(dev); w = w / w.mean()
_Xte = [torch.from_numpy(T.reshape(-1, T.shape[-1])) for _, T, _, _, _ in te]; _Yte = np.concatenate([L.reshape(-1) for _, _, L, _, _ in te]) if te else None
def _eval(head):   # v5: held-out loss (same class weights) and per-class patch IoU; ignored (-100) cells excluded
    head.eval(); P_, lo, n_ = [], 0.0, 0
    with torch.no_grad():
        for X_, (_, _, L, _, _) in zip(_Xte, te):
            z = head(X_.to(dev).float()); y = torch.from_numpy(L.reshape(-1)).to(dev); P_.append(z.argmax(-1).cpu().numpy())
            v = y != -100
            if v.any(): lo += nn.functional.cross_entropy(z[v], y[v], weight=w, reduction='sum').item(); n_ += int(v.sum())
    P_ = np.concatenate(P_); v = _Yte != -100; iou = {}
    for k in range(1, K):
        g = (_Yte == k) & v; pk = (P_ == k) & v; u = (g | pk).sum()
        if g.sum(): iou[classes[k]] = float((g & pk).sum() / u)
    return lo / max(1, n_), iou, P_
def _train(lr, hidden, curve=None):
    torch.manual_seed(a.seed); head = Head(h=hidden).to(dev); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=a.wd)
    for ep in range(a.epochs):
        head.train(); perm = torch.randperm(len(Xt)); tot = 0
        for i in range(0, len(perm), 2048): idx = perm[i:i + 2048]; loss = nn.functional.cross_entropy(head(Xt[idx].to(dev).float()), Yt[idx].to(dev), weight=w); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        trl = tot / (len(perm) // 2048 + 1)
        if te and (curve is not None or ep % 10 == 9 or ep == a.epochs - 1):
            vl, iou, P_ = _eval(head)
            if curve is not None: curve.append({'epoch': ep + 1, 'train_loss': trl, 'heldout_loss': vl, 'iou': iou, 'mean_iou': float(np.mean(list(iou.values()))) if iou else None})
            if ep % 10 == 9 or ep == a.epochs - 1: print(f"  ep {ep+1}: lr {lr:g} hidden {hidden}  train loss {trl:.3f}  held-out loss {vl:.3f}  held-out mean patch IoU {np.mean(list(iou.values())):.3f}", flush=True)
    return head
if a.grid:   # v5: curves for several settings on the same tokens, then stop
    res = []
    for g in a.grid.split(','):
        lr_, h_ = g.split(':'); cur = []; t0 = time.time(); _train(float(lr_), int(h_), cur)
        res.append({'lr': float(lr_), 'hidden': int(h_), 'wd': a.wd, 'epochs': a.epochs, 'bg_keep': a.bg_keep, 'test': sorted(test_sp), 'curve': cur, 'seconds': time.time() - t0})
        json.dump(res, open(a.curve_json or out / 'grid_curves.json', 'w'), indent=0)
    print("GRID_DONE"); raise SystemExit(0)
_curve = [] if a.curve_json else None
head = _train(a.lr, a.hidden, _curve)
if _curve is not None: json.dump({'lr': a.lr, 'hidden': a.hidden, 'wd': a.wd, 'curve': _curve}, open(a.curve_json, 'w'), indent=0)
if te:
    head.eval(); P = []; Y = []
    with torch.no_grad():
        for fn, T, L, grid, _ in te: P.append(head(torch.from_numpy(T.reshape(-1, T.shape[-1])).to(dev).float()).argmax(-1).cpu().numpy()); Y.append(L.reshape(-1))
    P = np.concatenate(P); Y = np.concatenate(Y)
head.eval(); chead = None
if CONT:   # v2: one sigmoid output per container, BCE on full container masks
    Ct = torch.from_numpy(np.concatenate([CONTG[fn].reshape(-1, len(CONT))[KEEP[fn]] for fn, _, _, _, _ in tr]).astype(np.float32))
    chead = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, len(CONT))).to(dev)
    copt = torch.optim.AdamW(chead.parameters(), lr=1e-3, weight_decay=0.01)
    for ep in range(a.epochs):
        chead.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 2048): idx = perm[i:i + 2048]; l_ = nn.functional.binary_cross_entropy_with_logits(chead(Xt[idx].to(dev).float()), Ct[idx].to(dev)); copt.zero_grad(); l_.backward(); copt.step()
    chead.eval(); torch.save(chead.state_dict(), out / 'container_head.pt'); json.dump(CONT, open(out / 'containers.json', 'w'))
    for j, c_ in enumerate(CONT):
        if not te: break
        with torch.no_grad(): Pc = np.concatenate([(torch.sigmoid(chead(torch.from_numpy(T.reshape(-1, T.shape[-1])).to(dev).float()))[:, j] > 0.5).cpu().numpy() for _, T, _, _, _ in te])
        Yc = np.concatenate([CONTG[fn][..., j].reshape(-1) for fn, _, _, _, _ in te]); u = (Pc | Yc).sum()
        print(f"  container {c_:<20} full-extent patch IoU {(Pc & Yc).sum() / u if u else float('nan'):.2f}  n {int(Yc.sum())}", flush=True)
ious = {}
for k in (range(1, K) if te else []):
    inter = ((P == k) & (Y == k)).sum(); union = ((P == k) | (Y == k)).sum(); ious[classes[k]] = (inter / union if union else np.nan, int((Y == k).sum()))
rows = sorted(((c, v[0], v[1]) for c, v in ious.items() if v[1] > 0), key=lambda r: -r[2])
print("held-out species, per-class patch IoU (n = ground-truth patches):"); [print(f"  {c:<24} IoU {iou:.2f}  n {n}") for c, iou, n in rows]
vals = np.array([iou for _, iou, _ in rows]) if rows else np.array([np.nan]); print(f"mean IoU over {len(rows)} classes present: {np.nanmean(vals):.3f}; classes with IoU >= 0.5: {(vals >= 0.5).sum()}; >= 0.3: {(vals >= 0.3).sum()}")
json.dump({'classes': classes, 'test_species': sorted(test_sp), 'ious': {c: [float(i), n] for c, i, n in rows}}, open(out / 'results.json', 'w'), indent=1); torch.save(head.state_dict(), out / 'head.pt')
# predicted label maps for the SAM2 refinement step (step 2)
maps = {}
for fn, T, L, grid, _ in te:
    with torch.no_grad(): maps[fn] = head(torch.from_numpy(T.reshape(-1, T.shape[-1])).to(dev).float()).argmax(-1).cpu().numpy().reshape(L.shape).astype(np.uint8)
np.savez_compressed(out / 'pred_labelmaps.npz', **{k: v for k, v in maps.items()}); json.dump({'classes': classes, 'gt_labelmaps': {fn: L.tolist() for fn, _, L, _, _ in te}}, open(out / 'labelmaps_meta.json', 'w'))
if a.save_probs:
    (out / 'probs').mkdir(exist_ok=True)
    for fn, p, polys in [it for it in items if sp_of(it[0]) in test_sp]:
        im = Image.open(p).convert('RGB'); W0, H0 = im.size; s_ = (a.size or max(W0, H0)) / max(W0, H0); nw, nh = max(16, int(round(W0 * s_ / 16)) * 16), max(16, int(round(H0 * s_ / 16)) * 16)
        base = (np.asarray(im.resize((nw, nh), Image.BILINEAR), np.float32) / 255 - MEAN) / STD; acc = np.zeros((nh, nw, K), np.float32)
        for dx, dy in ((0, 0), (8, 0), (0, 8), (8, 8)):
            x = np.pad(base, ((dy, 16 - dy if dy else 0), (dx, 16 - dx if dx else 0), (0, 0)), mode='edge'); gh, gw = x.shape[0] // 16, x.shape[1] // 16
            T_ = fwd(x[:gh * 16, :gw * 16])
            with torch.no_grad():
                pr = torch.softmax(head(T_), -1).cpu().numpy().reshape(gh, gw, K)
            up = cv2.resize(pr, (gw * 16, gh * 16), interpolation=cv2.INTER_LINEAR)      # patch centres -> pixels, smooth
            acc += up[dy:dy + nh, dx:dx + nw]
        np.savez_compressed(out / 'probs' / f"{Path(fn).stem}.npz", probs=(acc / 4).astype(np.float16), size=np.array([nw, nh]))
    print(f"saved shifted-pass probabilities for {len(list((out / 'probs').glob('*.npz')))} held-out images", flush=True)
# overlays for the held-out images: GT | prediction, label colours by class index
pal = (np.array([[0, 0, 0]] + [[(37 * k) % 255, (91 * k) % 255, (151 * k) % 255] for k in range(1, K)])).astype(np.uint8)
for fn, T, L, grid, _ in te:
    with torch.no_grad(): P1 = head(torch.from_numpy(T.reshape(-1, T.shape[-1])).to(dev).float()).argmax(-1).cpu().numpy().reshape(L.shape)
    im = cv2.imread(str(Path(a.images) / fn)); H, W = im.shape[:2]; sc = 900 / max(H, W); im = cv2.resize(im, (int(W * sc), int(H * sc)))
    def ov(lab): m = cv2.resize(pal[np.maximum(lab, 0)], (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST); return cv2.addWeighted(im, 0.45, m, 0.55, 0)
    sheet = np.hstack([ov(L), ov(P1)]); cv2.putText(sheet, f"{fn}  ground truth | DINOv3 per-patch prediction (held-out species)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2); cv2.imwrite(str(out / f"overlay_{Path(fn).stem}.png"), sheet)
# legend
leg = np.zeros((20 * K, 420, 3), np.uint8)
for k in range(K): leg[20 * k:20 * k + 20, :40] = pal[k]; cv2.putText(leg, classes[k], (48, 20 * k + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
cv2.imwrite(str(out / 'legend.png'), leg); print("ALL_DONE")

def gfilter(I, p, r=8, eps=1e-3):
    mean = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1)); mI = mean(I); vI = mean(I * I) - mI * mI; out = np.empty_like(p)
    for k in range(p.shape[2]):
        q = p[..., k]; mp = mean(q); a_ = (mean(I * q) - mI * mp) / (vI + eps); b_ = mp - a_ * mI; out[..., k] = mean(a_) * I + mean(b_)
    return out
if a.predict_dir or a.predict_list:
    from pycocotools import mask as mu
    exts = ('.tif', '.tiff', '.jpg', '.jpeg', '.png'); files = [Path(l.strip()) for l in open(a.predict_list) if l.strip()] if a.predict_list else sorted(f for f in Path(a.predict_dir).iterdir() if f.suffix.lower() in exts and not f.name.startswith('._'))
    coco = {'images': [], 'annotations': [], 'categories': [{'id': k, 'name': c} for k, c in enumerate(classes) if k]}; absent = {}
    (out / 'pred_overlays').mkdir(exist_ok=True); head.eval(); t0 = time.time()
    for ii, f in enumerate(files, 1):
        im = Image.open(f).convert('RGB'); W0, H0 = im.size; s_ = (a.size or max(W0, H0)) / max(W0, H0); nw, nh = max(16, int(round(W0 * s_ / 16)) * 16), max(16, int(round(H0 * s_ / 16)) * 16)
        rgb = np.asarray(im.resize((nw, nh), Image.BILINEAR)); base = (rgb.astype(np.float32) / 255 - MEAN) / STD; acc = np.zeros((nh, nw, K), np.float32)
        for dx, dy in ((0, 0), (8, 0), (0, 8), (8, 8)):
            x = np.pad(base, ((dy, 16 - dy if dy else 0), (dx, 16 - dx if dx else 0), (0, 0)), mode='edge'); gh, gw = x.shape[0] // 16, x.shape[1] // 16
            T_ = fwd(x[:gh * 16, :gw * 16])
            with torch.no_grad(): pr = torch.softmax(head(T_), -1).cpu().numpy().reshape(gh, gw, K)
            acc += cv2.resize(pr, (gw * 16, gh * 16), interpolation=cv2.INTER_LINEAR)[dy:dy + nh, dx:dx + nw]
            if chead is not None:
                with torch.no_grad(): pc = torch.sigmoid(chead(T_)).cpu().numpy().reshape(gh, gw, len(CONT))
                cacc = (cacc if dx or dy else 0) + cv2.resize(pc, (gw * 16, gh * 16), interpolation=cv2.INTER_LINEAR).reshape(gh * 16, gw * 16, len(CONT))[dy:dy + nh, dx:dx + nw]
        if a.fg_mask == 'border':   # 12 Oct 2026: background removal - no body pixel may be called background, nothing outside is a part
            from fg_mask_v1 import fg_mask as _fgm
            _fg = _fgm(rgb); acc[_fg, 0] = 0; acc[~_fg, 1:] = 0; acc[~_fg, 0] = 4
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255; lab = gfilter(grey, acc / 4).argmax(-1).astype(np.uint8)
        labf = cv2.resize(lab, (W0, H0), interpolation=cv2.INTER_NEAREST); coco['images'].append({'id': ii, 'file_name': f.name, 'width': W0, 'height': H0}); absent[f.name] = []
        cmask = {}
        if chead is not None:
            cg_ = gfilter(grey, cacc / 4) > 0.5
            if a.fg_mask == 'border': cg_ &= _fg[..., None]
            for j, c_ in enumerate(CONT): cmask[cidx[c_]] = cv2.resize(cg_[..., j].astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST)
        for k in range(1, K):
            m = cmask[k] if k in cmask else (labf == k).astype(np.uint8)
            if m.sum() < a.min_area_frac * W0 * H0: absent[f.name].append(classes[k]); continue
            rle = mu.encode(np.asfortranarray(m)); rle['counts'] = rle['counts'].decode(); x_, y_, w_, h_ = cv2.boundingRect(m)
            coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': ii, 'category_id': k, 'segmentation': rle, 'area': int(m.sum()), 'bbox': [x_, y_, w_, h_], 'iscrowd': 0})
        pal_ = (np.array([[0, 0, 0]] + [[(37 * k) % 255, (91 * k) % 255, (151 * k) % 255] for k in range(1, K)])).astype(np.uint8)
        sm = cv2.resize(rgb[:, :, ::-1], (900, int(900 * nh / nw))); ov = cv2.addWeighted(sm, 0.45, cv2.resize(pal_[lab], (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST), 0.55, 0)
        cv2.imwrite(str(out / 'pred_overlays' / (f.stem + '.png')), np.hstack([sm, ov]))
        if ii % 10 == 0: print(f"  predicted {ii}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    json.dump(coco, open(out / 'predictions.json', 'w')); json.dump(absent, open(out / 'absent_parts.json', 'w'), indent=1)
    print(f"predictions.json: {len(files)} images, {len(coco['annotations'])} masks; parts reported absent on {sum(bool(v) for v in absent.values())} images", flush=True)
    # v4: ALSO write both Descriptron formats (AVD): predictions_polygons.json = polygon COCO that the Descriptron editor opens and edits
    # (largest piece + pieces >= 10% of it, approxPolyDP 1.5 px); predictions.rlejson = the same RLE masks as predictions.json
    json.dump(coco, open(out / 'predictions.rlejson', 'w')); pc = {'images': coco['images'], 'categories': coco['categories'], 'annotations': []}
    for a_ in coco['annotations']:
        r_ = dict(a_['segmentation']); r_['counts'] = r_['counts'].encode() if isinstance(r_['counts'], str) else r_['counts']
        m_ = cv2.morphologyEx(mu.decode(r_).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); cs_, _ = cv2.findContours(m_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs_: continue
        big_ = max(cv2.contourArea(c) for c in cs_); ps_ = [cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2) for c in cs_ if cv2.contourArea(c) >= max(0.1 * big_, 200)]
        ps_ = [q.astype(float).ravel().tolist() for q in ps_ if len(q) >= 3]
        if not ps_: continue
        x_, y_, w_, h_ = cv2.boundingRect(np.vstack([np.array(q).reshape(-1, 2) for q in ps_]).astype(np.int32))
        pc['annotations'].append({'id': len(pc['annotations']) + 1, 'image_id': a_['image_id'], 'category_id': a_['category_id'], 'segmentation': ps_, 'area': float(a_.get('area', 0)), 'bbox': [x_, y_, w_, h_], 'iscrowd': 0})
    json.dump(pc, open(out / 'predictions_polygons.json', 'w')); print(f"also wrote predictions_polygons.json ({len(pc['annotations'])} polygon masks, opens in the Descriptron editor) and predictions.rlejson", flush=True)
