#!/usr/bin/env python3
"""Texture teacher -> student for ANY region and ANY literature labels (general form of texture_hires_v3.py, 29 Oct 2026).

Same recipe as the ant/weevil sculpture work (texture_hires_v3.py, region_tokens_v2.py --ppmm):
 0. Every region is brought to the SAME resolution before tokenising: px per mm from --scales (--size_mode ppmm), or, when the
    images have no scale bar, the region's short side set to a fixed number of pixels (--size_mode px). Grey, z-scored in the region.
 1. TEACHER: PCA(64) + logistic on DINOv3 patch tokens (layer --layer) of the LABELLED images at --lo, labels = literature words
    per taxon (--labels taxon<TAB>label). Validated leave-one-TAXON-out against --null label permutations.
    Dense teacher maps on every image: 4 half-patch-shifted passes, bilinear to pixels, guided filter (labelled images get the
    map of the teacher fitted WITHOUT their taxon, so the student check below is honest).
 2. STUDENT: MLP on tokens at --hi with the teacher maps as soft targets, folds by --group_regex (e.g. species), so every image
    is predicted by a student that never saw its group. Held-out student scores on the labelled taxa are reported like the teacher's.
 3. Outputs: report.txt, image_scores.tsv (teacher / student mean P per class in the region), sheets/ (photo + teacher overlay |
    photo + student overlay | raw PCA | high-pass PCA, all at --hi), maps.npz (student P maps at --hi, float16).
Example (Mycetophilidae wing membrane, archive without scale bars):
  texture_teacher_student_v1.py --images imgs --coco membrane_coco.json --region membrane --labels labels_microtrichia_rows.tsv
      --positive rows --taxon_regex '^([A-Za-z]+)__' --group_regex '^([A-Za-z]+__[A-Za-z]+)' --size_mode px --lo 1024 --hi 2048 --out out
"""
import argparse, os, re, csv, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--images', required=True); ap.add_argument('--coco', nargs='+', required=True, help='COCO files with the region (polygons or RLE); first file with the region wins')
ap.add_argument('--region', required=True, help='COCO category name of the region'); ap.add_argument('--labels', required=True, help='TSV with columns taxon, label (literature word per taxon)')
ap.add_argument('--positive', default='', help='binary task: label counted as positive (AUC reported); empty = multi-class (balanced accuracy)')
ap.add_argument('--taxon_regex', default=r'^([^_]+)_', help='taxon of a file name, matched against --labels')
ap.add_argument('--group_regex', default='', help='student folds (e.g. species); default = --taxon_regex')
ap.add_argument('--size_mode', default='px', choices=['px', 'ppmm'], help='px = region short side set to --lo/--hi pixels; ppmm = --lo/--hi px per mm from --scales')
ap.add_argument('--lo', type=float, default=1024, help='teacher resolution'); ap.add_argument('--hi', type=float, default=2048, help='student resolution')
ap.add_argument('--max_up', type=float, default=8.0, help='largest upsampling allowed (px mode)')
ap.add_argument('--scales', default='', help='CSV with file + px_per_mm (also accepts image_filename + pixels_per_unit) for --size_mode ppmm')
ap.add_argument('--layer', type=int, default=18); ap.add_argument('--features', default='raw', choices=['raw', 'raw+hp'], help='raw tokens, or raw + high-pass (token minus 7x7 in-region mean)')
ap.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m'); ap.add_argument('--tile', type=int, default=1024)
ap.add_argument('--per_image', type=int, default=1500, help='patches sampled per image (teacher and student)'); ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--epochs', type=int, default=25); ap.add_argument('--null', type=int, default=100, help='label permutations for the teacher')
ap.add_argument('--sheets', type=int, default=24, help='images drawn as sheets (spread over taxa)'); ap.add_argument('--max_images', type=int, default=0)
ap.add_argument('--teacher_files', default='', help='optional text file, one file name per line: only these labelled images train/score the teacher; other images of a labelled taxon are mapped as unlabelled')
ap.add_argument('--out', required=True)
a = ap.parse_args(); os.makedirs(f'{a.out}/sheets', exist_ok=True); rng = np.random.default_rng(0); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
log_f = open(f'{a.out}/report.txt', 'w')
def log(*s):
    t = ' '.join(str(x) for x in s); print(t, flush=True); log_f.write(t + '\n'); log_f.flush()
log('texture_teacher_student v1', time.strftime('%Y-%m-%d %H:%M'), '|', ' '.join(f'{k}={v}' for k, v in vars(a).items()))
# ---------- regions ----------
def dec(x, hw):
    if isinstance(x['segmentation'], list):
        m = np.zeros(hw, np.uint8)
        for q in x['segmentation']: cv2.fillPoly(m, [np.array(q).reshape(-1, 2).round().astype(np.int32)], 1)
        return m.astype(bool)
    from pycocotools import mask as mu
    s = dict(x['segmentation']); s['counts'] = s['counts'].encode() if isinstance(s['counts'], str) else s['counts']; return mu.decode(s).astype(bool)
REG = {}
for cf in a.coco:
    d = json.load(open(cf)); cid = [c['id'] for c in d['categories'] if c['name'] == a.region]; ims = {i['id']: i for i in d['images']}; got = {}
    for x in d['annotations']:
        if x['category_id'] in cid: got.setdefault(ims[x['image_id']]['file_name'], (ims[x['image_id']], []))[1].append(x)
    n = 0
    for f, v in got.items():
        if f not in REG and os.path.exists(os.path.join(a.images, f)): REG[f] = v; n += 1
    log(f'  {cf}: region in {len(got)} images, {n} used')
files = sorted(REG)[:a.max_images or None]
LAB = {r['taxon']: r['label'] for r in csv.DictReader(open(a.labels), delimiter='\t')}
CL = sorted(set(LAB.values())); K = len(CL); POS = CL.index(a.positive) if a.positive else -1
assert K >= 2, 'need at least two label words'; assert not a.positive or a.positive in CL, f'--positive {a.positive} not in labels {CL}'
tax = lambda f: (re.search(a.taxon_regex, f) or [None, ''])[1]
grp = lambda f: (re.search(a.group_regex or a.taxon_regex, f) or [None, f])[1]
SC = {}
if a.scales:
    for r in csv.DictReader(open(a.scales)):
        k = r.get('file') or r.get('image_filename'); v = r.get('px_per_mm') or r.get('pixels_per_unit')
        if k and v: SC[os.path.basename(k)] = float(v)
if a.size_mode == 'ppmm':
    def key(f): return f if f in SC else f.split('__', 1)[-1]   # archive copies are named <taxon>__<original name>
    miss = [f for f in files if key(f) not in SC]; files = [f for f in files if key(f) in SC]
    if miss: log(f'  {len(miss)} images without a scale skipped, e.g. {miss[:3]}')
TF = {l.strip() for l in open(a.teacher_files) if l.strip()} if a.teacher_files else None
lab_files = [f for f in files if tax(f) in LAB and (TF is None or f in TF)]; TAXA = sorted({tax(f) for f in lab_files})
log(f'images {len(files)}; labelled {len(lab_files)} in {len(TAXA)} taxa; classes {CL}; per class (taxa):',
    {c: sum(LAB[t] == c for t in TAXA) for c in CL}, '| unused label rows:', sorted(set(LAB) - set(TAXA)))
# ---------- resolution + grey z-score (texture_hires_v3 grey_at) ----------
def grey_at(f, res):
    im, anns = REG[f]; hw = (im['height'], im['width']); m = np.zeros(hw, bool)
    for x in anns: m |= dec(x, hw)
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g = np.asarray(Image.open(os.path.join(a.images, f)).convert('L'), np.float32)[y0:y1, x0:x1] / 255; m = m[y0:y1, x0:x1]
    s = res / SC[key(f)] if a.size_mode == 'ppmm' else min(a.max_up, res / min(g.shape))
    nh, nw = max(16, int(g.shape[0] * s) // 16 * 16), max(16, int(g.shape[1] * s) // 16 * 16)
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC); m = cv2.resize(m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
    g = np.clip((g - g[m].mean()) / (g[m].std() + 1e-6) * 0.18 + 0.5, 0, 1).astype(np.float32); return g, m, s
# ---------- DINOv3 tokens, tiled (tile centres kept) ----------
from transformers import AutoModel
net = AutoModel.from_pretrained(a.model).to(dev).eval(); NREG = getattr(net.config, 'num_register_tokens', 4); D = net.config.hidden_size
FD = D * (2 if a.features == 'raw+hp' else 1)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
def toks(g):
    """grey HxW (multiples of 16) -> [H/16, W/16, D] float16 numpy, layer --layer."""
    x = ((np.stack([g] * 3, -1) - MEAN) / STD).astype(np.float32); H, W = g.shape; OV = 128; STEP = a.tile - 2 * OV
    def one(t):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            hs = net(pixel_values=torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1)))[None].to(dev), output_hidden_states=True).hidden_states
        return hs[a.layer + 1][0, 1 + NREG:].float().reshape(t.shape[0] // 16, t.shape[1] // 16, D).half().cpu().numpy()
    if H <= a.tile and W <= a.tile: return one(x)
    out = np.zeros((H // 16, W // 16, D), np.float16)
    for y0 in range(0, H, STEP):
        for x0 in range(0, W, STEP):
            ya, xa = max(0, y0 - OV), max(0, x0 - OV); t = x[ya:min(H, y0 + STEP + OV), xa:min(W, x0 + STEP + OV)]; t = t[:t.shape[0] // 16 * 16, :t.shape[1] // 16 * 16]
            T = one(t); ky, kx = (y0 - ya) // 16, (x0 - xa) // 16; ny, nx = min(STEP, H - y0) // 16, min(STEP, W - x0) // 16
            out[y0 // 16:y0 // 16 + ny, x0 // 16:x0 // 16 + nx] = T[ky:ky + ny, kx:kx + nx]
    return out
def feats(T, mg):
    """[gh,gw,D] tokens + [gh,gw] region coverage -> [gh*gw, FD] features (raw, or raw + high-pass inside the region)."""
    if a.features == 'raw': return T.reshape(-1, D)
    G = torch.from_numpy(T.astype(np.float32)); w = torch.from_numpy((mg >= 0.5).astype(np.float32))
    num = F.avg_pool2d((G * w[..., None]).permute(2, 0, 1)[None], 7, 1, 3, count_include_pad=False)[0].permute(1, 2, 0)
    den = F.avg_pool2d(w[None, None], 7, 1, 3, count_include_pad=False)[0, 0]
    return np.concatenate([T, (G - num / den.clamp_min(1e-3)[..., None]).numpy().astype(np.float16)], -1).reshape(-1, FD)
def cover(m, gh, gw): return cv2.resize(m[:gh * 16, :gw * 16].astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
def rz(a_, size, interp): return np.stack([cv2.resize(np.ascontiguousarray(a_[..., c]), size, interpolation=interp) for c in range(a_.shape[-1])], -1)
def gfilter(I, p, r=8, eps=1e-3):
    mean = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1)); mI = mean(I); vI = mean(I * I) - mI * mI; out = np.empty_like(p)
    for k in range(p.shape[2]):
        q = p[..., k]; mq = mean(q); a_ = (mean(I * q) - mI * mq) / (vI + eps); b_ = mq - a_ * mI; out[..., k] = mean(a_) * I + mean(b_)
    return out
def dense(g, m, fn):
    """4 half-patch-shifted passes of fn(features)->(N,K), bilinear to pixels, averaged, guided-filtered with the grey image (texture_hires_v3)."""
    H, W = g.shape; acc = np.zeros((H, W, K), np.float32)
    for dx, dy in ((0, 0), (8, 0), (0, 8), (8, 8)):
        x = np.pad(g, ((dy, 16 - dy if dy else 0), (dx, 16 - dx if dx else 0)), mode='edge'); mx = np.pad(m, ((dy, 16 - dy if dy else 0), (dx, 16 - dx if dx else 0)))
        gh, gw = x.shape[0] // 16, x.shape[1] // 16; T = toks(x[:gh * 16, :gw * 16])
        pr = fn(feats(T, cover(mx, gh, gw))).reshape(gh, gw, K).astype(np.float32)
        acc += rz(pr, (gw * 16, gh * 16), cv2.INTER_LINEAR)[dy:dy + H, dx:dx + W]
    return np.clip(gfilter(g, acc / 4), 0, 1)
def sample(g, m, n):
    gh, gw = g.shape[0] // 16, g.shape[1] // 16; T = toks(g[:gh * 16, :gw * 16]); mg = cover(m, gh, gw); Fe = feats(T, mg)
    idx = np.where(mg.ravel() >= 0.6)[0]; idx = rng.permutation(idx)[:n] if len(idx) else np.array([], int); return Fe[idx].astype(np.float16), idx, T, mg
def score(P, m): return P[m].mean(0) if m.any() else np.full(K, np.nan)
# ---------- 1. teacher ----------
t0 = time.time(); TX, TY, TT = [], [], []
for i, f in enumerate(lab_files):
    g, m, s = grey_at(f, a.lo); X, _, _, _ = sample(g, m, a.per_image); TX.append(X); TY.append(np.full(len(X), CL.index(LAB[tax(f)]))); TT.append(np.full(len(X), TAXA.index(tax(f))))
    if i % 20 == 0: log(f'  teacher tokens {i + 1}/{len(lab_files)} x{s:.2f}  {time.time() - t0:.0f}s')
TX, TY, TT = np.concatenate(TX).astype(np.float32), np.concatenate(TY), np.concatenate(TT)
def fit(X, y):
    sc = StandardScaler().fit(X); pc = PCA(64, random_state=0).fit(sc.transform(X)); lr = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(pc.transform(sc.transform(X)), y)
    def fn(Fe):
        P = np.zeros((len(Fe), K)); P[:, lr.classes_] = lr.predict_proba(pc.transform(sc.transform(Fe.astype(np.float32)))); return P
    return fn
def evaluate(lab_of_taxon):
    """leave-one-taxon-out: image-level mean P(class) of held-out taxa. lab_of_taxon: taxon index -> class index."""
    y = lab_of_taxon[TT]; S = np.zeros((len(TT), K))
    for t in range(len(TAXA)):
        tr = TT != t
        if len(np.unique(y[tr])) < K: return None
        Ztr, Zte = ZLOTO[t]; lr = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(Ztr, y[tr])
        P = np.zeros((len(Zte), K)); P[:, lr.classes_] = lr.predict_proba(Zte); S[~tr] = P
    ti = np.array([lab_of_taxon[t] for t in range(len(TAXA))]); St = np.array([S[TT == t].mean(0) for t in range(len(TAXA))])
    if POS >= 0: return roc_auc_score(ti == POS, St[:, POS]), S
    return balanced_accuracy_score(ti, St.argmax(1)), S
ZLOTO = {}   # scaler + PCA fitted without the held-out taxon, once; permutations only refit the logistic
for t in range(len(TAXA)):
    tr = TT != t; sc = StandardScaler().fit(TX[tr]); pc = PCA(64, random_state=0).fit(sc.transform(TX[tr]))
    ZLOTO[t] = (pc.transform(sc.transform(TX[tr])), pc.transform(sc.transform(TX[~tr])))
lab_t = np.array([CL.index(LAB[t]) for t in TAXA]); res = evaluate(lab_t); obs = res[0] if res else float('nan')
nulls = []
for p in range(a.null):
    r_ = evaluate(rng.permutation(lab_t))
    if r_: nulls.append(r_[0])
metric = 'AUC' if POS >= 0 else 'balanced accuracy'
log(f'TEACHER leave-one-taxon-out {metric} (taxon level): {obs:.3f}   null95 {np.percentile(nulls, 95) if nulls else float("nan"):.3f}  (n null {len(nulls)})  {time.time() - t0:.0f}s')
T_all = fit(TX, TY); T_wo = {t: fit(TX[TT != TAXA.index(t)], TY[TT != TAXA.index(t)]) for t in TAXA}
del TX
# dense teacher maps at lo for every image -> student samples at hi
SAMP, rows, SH = {}, {}, set()
by_tax = {}
for f in files: by_tax.setdefault(tax(f), []).append(f)
for t, fs in by_tax.items(): SH.update(fs[:max(1, a.sheets // max(1, len(by_tax)))])
SH = set(list(SH)[:a.sheets]) if len(SH) > a.sheets else SH
TMAP = {}
for i, f in enumerate(files):
    g, m, _ = grey_at(f, a.lo); fn = T_wo.get(tax(f), T_all); Pt = dense(g, m, fn)
    gh_, mh, _ = grey_at(f, a.hi); X, idx, T, mg = sample(gh_, mh, a.per_image); gh2, gw2 = gh_.shape[0] // 16, gh_.shape[1] // 16
    tgt = rz(Pt, (gw2, gh2), cv2.INTER_AREA).reshape(-1, K)[idx]; SAMP[f] = (X, tgt.astype(np.float32))
    rows[f] = dict(file=f, taxon=tax(f), group=grp(f), labelled=int(f in lab_files), label=LAB.get(tax(f), ''), lo_px=f'{g.shape[0]}x{g.shape[1]}', hi_px=f'{gh_.shape[0]}x{gh_.shape[1]}',
                   **{f'teacher_{c}': round(float(v), 4) for c, v in zip(CL, score(Pt, m))})
    if f in SH:
        TMAP[f] = cv2.resize(Pt, (gh_.shape[1], gh_.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(gh_.shape[0], gh_.shape[1], K)
        def pca_rgb(Fe):
            v = Fe[mg.ravel() >= 0.5].astype(np.float32); pc = PCA(3, random_state=0).fit(v); z = pc.transform(Fe.astype(np.float32)); lo_, hi_ = np.percentile(z[mg.ravel() >= 0.5], [2, 98], 0)
            z = np.clip((z - lo_) / (hi_ - lo_ + 1e-6), 0, 1).reshape(gh2, gw2, 3); z[mg < 0.5] = 0
            return cv2.resize((z * 255).astype(np.uint8), (gh_.shape[1], gh_.shape[0]), interpolation=cv2.INTER_NEAREST)
        Fr = T.reshape(-1, D)
        G = torch.from_numpy(T.astype(np.float32)); w = torch.from_numpy((mg >= 0.5).astype(np.float32))
        num = F.avg_pool2d((G * w[..., None]).permute(2, 0, 1)[None], 7, 1, 3, count_include_pad=False)[0].permute(1, 2, 0); den = F.avg_pool2d(w[None, None], 7, 1, 3, count_include_pad=False)[0, 0]
        Fh = (G - num / den.clamp_min(1e-3)[..., None]).numpy().reshape(-1, D)
        rows[f]['_pca'] = (pca_rgb(Fr), pca_rgb(Fh))
    if i % 20 == 0: log(f'  teacher maps + student tokens {i + 1}/{len(files)}  {time.time() - t0:.0f}s')
# ---------- 2. student, folds by group ----------
class Head(nn.Module):
    def __init__(s): super().__init__(); s.net = nn.Sequential(nn.LayerNorm(FD), nn.Linear(FD, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, K))
    def forward(s, t): return s.net(t)
def train(fs):
    X = torch.from_numpy(np.concatenate([SAMP[f][0] for f in fs]).astype(np.float32)); Y = torch.from_numpy(np.concatenate([SAMP[f][1] for f in fs]))
    Y = Y / Y.sum(1, keepdim=True).clamp_min(1e-6); hd = Head().to(dev); opt = torch.optim.AdamW(hd.parameters(), lr=1e-3, weight_decay=0.01)
    for ep in range(a.epochs):
        perm = torch.randperm(len(X))
        for j in range(0, len(X), 4096):
            idx = perm[j:j + 4096]; loss = -(Y[idx].to(dev) * torch.log_softmax(hd(X[idx].to(dev)), -1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    hd.eval()
    def fn(Fe):
        with torch.no_grad(): return torch.softmax(hd(torch.from_numpy(Fe.astype(np.float32)).to(dev)), -1).cpu().numpy()
    return fn
G_ = sorted({grp(f) for f in files}); fold = {g_: k % a.folds for k, g_ in enumerate(rng.permutation(G_))}
# a labelled taxon must sit in ONE fold, so the student check on labelled taxa is leave-taxon-out as well
for t in TAXA:
    k = fold[grp(by_tax[t][0])]
    for f in by_tax[t]: fold[grp(f)] = k
MAPS, agree = {}, []
PAL = np.array([[60, 90, 255], [255, 140, 0], [40, 200, 80], [200, 40, 200], [230, 30, 30], [0, 200, 220]], np.uint8)
def overlay(rgb, P, m):
    c = PAL[P.argmax(-1) % len(PAL)] if POS < 0 else (np.array(__import__('matplotlib').cm.coolwarm(P[..., POS])[..., :3]) * 255).astype(np.uint8)
    o = rgb.copy(); o[m] = (0.45 * rgb[m] + 0.55 * c[m]).astype(np.uint8); return o
for k in range(a.folds):
    te = [f for f in files if fold[grp(f)] == k]
    if not te: continue
    fn = train([f for f in files if fold[grp(f)] != k])
    for f in te:
        g, m, _ = grey_at(f, a.hi); Ps = dense(g, m, fn); rows[f].update({f'student_{c}': round(float(v), 4) for c, v in zip(CL, score(Ps, m))})
        MAPS[os.path.splitext(f)[0]] = cv2.resize(Ps, (max(1, Ps.shape[1] // 4), max(1, Ps.shape[0] // 4)), interpolation=cv2.INTER_AREA).astype(np.float16)
        if f in TMAP:
            im, anns = REG[f]; hw = (im['height'], im['width']); mm = np.zeros(hw, bool)
            for x in anns: mm |= dec(x, hw)
            ys, xs = np.where(mm); rgb = np.asarray(Image.open(os.path.join(a.images, f)).convert('RGB'))[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            rgb = cv2.resize(rgb, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_CUBIC); pr, ph = rows[f].pop('_pca')
            sheet = np.vstack([np.hstack([overlay(rgb, TMAP[f], m), overlay(rgb, Ps, m)]), np.hstack([pr, ph])])
            s_ = 1400 / sheet.shape[1]; sheet = cv2.resize(sheet, (1400, int(sheet.shape[0] * s_)), interpolation=cv2.INTER_AREA)
            cv2.putText(sheet, f'{f[:60]} | teacher {rows[f]["teacher_" + CL[max(POS, 0)]]:.2f} student {rows[f]["student_" + CL[max(POS, 0)]]:.2f} ({CL[max(POS, 0)]})', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            Image.fromarray(sheet).save(f'{a.out}/sheets/{os.path.splitext(f)[0]}.jpg', quality=88); del TMAP[f]
    log(f'  student fold {k}: {len(te)} held-out images  {time.time() - t0:.0f}s')
for r in rows.values(): r.pop('_pca', None)
# student check on labelled taxa (held out by fold)
by_tax_all = by_tax; by_tax = {t: [f for f in fs if f in lab_files] or fs for t, fs in by_tax_all.items()}   # labelled-taxon checks use teacher images only
St = np.array([[np.mean([rows[f][f'student_{c}'] for f in by_tax[t]]) for c in CL] for t in TAXA])
smet = roc_auc_score(lab_t == POS, St[:, POS]) if POS >= 0 else balanced_accuracy_score(lab_t, St.argmax(1))
log(f'STUDENT held-out-fold {metric} on the labelled taxa (taxon level): {smet:.3f}')
for t in TAXA: log(f'    {t:22s} {LAB[t]:12s} teacher(without taxon) ' + ' '.join(f'{np.mean([rows[f]["teacher_" + c] for f in by_tax[t]]):.2f}' for c in CL) + '   student ' + ' '.join(f'{x:.2f}' for x in St[TAXA.index(t)]))
un = [f for f in files if f not in lab_files]
if un:
    c0 = CL[max(POS, 0)]
    for nm, col in (('teacher', f'teacher_{c0}'), ('student', f'student_{c0}')):
        v = np.array([rows[f][col] for f in un]); log(f'UNLABELLED images ({len(un)}): {nm} P({c0}) median {np.median(v):.2f}; >0.5 in {(v > .5).sum()}/{len(v)}')
keys = list(dict.fromkeys(k for r in rows.values() for k in r))
with open(f'{a.out}/image_scores.tsv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys, delimiter='\t'); w.writeheader(); w.writerows(rows.values())
np.savez_compressed(f'{a.out}/maps.npz', **MAPS); log('wrote image_scores.tsv, maps.npz (student P at hi / 4), sheets/', len(os.listdir(f'{a.out}/sheets')))
log('DONE', f'{time.time() - t0:.0f}s')
