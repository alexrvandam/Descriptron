#!/usr/bin/env python3
"""
biorag_probe_v1.py - BioRAG-Probe: test a literature-stated character against DINOv3 tokens inside one annotated region
=====================================================================================================================
(16 Oct 2026; the pattern used for ant sculpture, weevil vestiture and Mycetophila wing veins, written once, taxon-agnostic.)

Inputs
  --images DIR          images (file names carry the taxon, see --taxon_regex)
  --coco A.json [B.json ...]   region masks; for each image the FIRST file that has --region for it is used (e.g. hand outlines
                        first, model predictions second)
  --region NAME         COCO category to probe (all pieces of it in an image are joined)
  Sub-region, two routes (can be combined):
  --exclude A,B         SUBTRACT given masks (any --coco file), dilated --exclude_dilate_px (e.g. veins from a wing outline)
  --keep_majority_cluster K   no masks needed: k-means the region's tokens into K groups per image, keep the largest
                        (e.g. wing membrane without veins; elytral intervals without striae)
  --labels TSV          columns taxon, label (+ anything else, kept). label = a class word for a binary task (give --positive)
                        or a number for a numeric task. Rows whose taxon matches no image are reported.
Method (fixed in advance; every number is computed with whole taxa held out, never on a taxon used for fitting)
  For each image: region mask (dilated by --dilate_px so structures standing out of the region are included), crop with margin,
  for each --scales factor and --layers layer: DINOv3 patch tokens (overlapping tiles, tile centres kept), patches >= 50% inside
  the mask; 'hp' = high-pass token (token minus the mean of in-mask neighbours in a 7x7-patch window).
  A  specimen-level: [mean, 90th percentile] of the region's tokens -> PCA -> logistic / ridge.
  B  patch-level: every patch carries its specimen's label; logistic on PCA tokens; specimen score = mean patch probability.
  Baseline (no training): dark-dot density = local maxima of a black-top-hat filter inside the mask per 100 px of region length.
  Leave-one-taxon-out; binary: AUC over specimens and over taxa (taxon mean score); numeric: Spearman over taxa.
  Null: labels permuted between taxa (taxa keep their specimens), --null times; reports the 95th percentile.
  PRIMARY result = method B, layer --primary_layer, variant raw+hp, scale 1 (declared before running); the rest is exploratory.
Outputs (--out): report.txt, scores_<config>.tsv, specimen_scores_primary.tsv (incl. unlabelled images, predicted from all
  labelled taxa), sheet_primary.jpg (one image per taxon: native crop x2 | patch probability map | dark dots).
"""
import argparse, json, os, re, csv, sys, time
import numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--images', required=True); ap.add_argument('--coco', nargs='+', required=True); ap.add_argument('--region', required=True)
ap.add_argument('--labels', required=True); ap.add_argument('--positive', default='', help='binary task: the label word counted as positive')
ap.add_argument('--taxon_regex', default=r'^([^_]+)_'); ap.add_argument('--out', required=True)
ap.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m'); ap.add_argument('--layers', default='12,18')
ap.add_argument('--scales', default='1,2'); ap.add_argument('--primary_layer', type=int, default=18)
ap.add_argument('--dilate_px', type=int, default=8); ap.add_argument('--margin_px', type=int, default=48)
ap.add_argument('--patches_per_image', type=int, default=256); ap.add_argument('--null', type=int, default=100, help='label permutations for method A'); ap.add_argument('--null_b', type=int, default=30, help='label permutations for the PRIMARY method-B result (slow)')
ap.add_argument('--region_px', type=int, default=0, help='>0: resize each region so its SHORT side is this many px (x --scales); 0 = native'); ap.add_argument('--region_max_up', type=float, default=8.0)
ap.add_argument('--tile', type=int, default=1024); ap.add_argument('--max_images', type=int, default=0)
ap.add_argument('--exclude', default='', help='comma list of COCO categories (from any --coco file) to SUBTRACT from the region, e.g. veins from a wing'); ap.add_argument('--exclude_dilate_px', type=int, default=10)
ap.add_argument('--input', default='colour', choices=['colour', 'grey'], help='v1d: grey = luminance copied to 3 channels (grey beat colour for ant sculpture and weevil pilosity)')
ap.add_argument('--norm', default='none', choices=['none', 'zscore'], help='v1d: zscore = per-image, per-channel mean/sd inside the region set to 128/48 (brightness normalisation)')
ap.add_argument('--clahe', type=float, default=0, help='v1d: CLAHE clip limit on L* before the above (0 = off)')
ap.add_argument('--fold_groups', default='', help='v1c: TSV taxon<TAB>group; cross-validation holds out whole groups (e.g. source paper) instead of single taxa'); ap.add_argument('--keep_majority_cluster', type=int, default=0, help='K>=2: inside the region, k-means the tokens (layer --primary_layer) into K groups per image and keep only the LARGEST group (e.g. wing membrane without veins, no vein model needed); 0 = off')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
LAYERS = [int(x) for x in a.layers.split(',')]; SCALES = [float(x) for x in a.scales.split(',')]
log_f = open(os.path.join(a.out, 'report.txt'), 'w')
def log(*s):
    t = ' '.join(str(x) for x in s); print(t, flush=True); log_f.write(t + '\n'); log_f.flush()
log('BioRAG-Probe v1', time.strftime('%Y-%m-%d %H:%M'), '| region', a.region, '| exclude', a.exclude or '-', '| keep_majority_cluster', a.keep_majority_cluster, '| layers', LAYERS, '| scales', SCALES, '| model', a.model)
# ---------- region masks (first COCO file that has the region wins) ----------
regions = {}; EXCL = {}
for cf in a.coco:
    d = json.load(open(cf)); cn = {c['id']: c['name'] for c in d['categories']}; fn = {i['id']: i['file_name'] for i in d['images']}; got = {}
    for an in d['annotations']:
        if cn.get(an['category_id']) == a.region and isinstance(an.get('segmentation'), list) and an['segmentation']:
            got.setdefault(fn[an['image_id']], []).extend(an['segmentation'])
    new = {k: (v, os.path.basename(cf)) for k, v in got.items() if k not in regions}; regions.update(new); log(f'  {cf}: region in {len(got)} images, {len(new)} used')
    for an in d['annotations']:   # v1c: polygons to subtract (--exclude); every file contributes
        if cn.get(an['category_id']) in set(filter(None, a.exclude.split(','))) and isinstance(an.get('segmentation'), list): EXCL.setdefault(fn[an['image_id']], []).extend(an['segmentation'])
# ---------- labels ----------
lab_rows = list(csv.DictReader(open(a.labels), delimiter='\t')); LAB = {r['taxon']: r['label'] for r in lab_rows}
numeric = not a.positive
if numeric: LAB = {k: float(v) for k, v in LAB.items()}
else: LAB = {k: int(v == a.positive) for k, v in LAB.items()}
taxon = lambda f: (re.search(a.taxon_regex, f) or [None, ''])[1]
files = sorted(f for f in regions if os.path.exists(os.path.join(a.images, f)))
if a.max_images: files = files[:a.max_images]
lab_files = [f for f in files if taxon(f) in LAB]
log(f'images with region: {len(files)}; labelled: {len(lab_files)} in {len({taxon(f) for f in lab_files})} taxa; unused label rows: {sorted(set(LAB) - {taxon(f) for f in files})}')
# ---------- DINOv3 ----------
from transformers import AutoModel
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net = AutoModel.from_pretrained(a.model).to(dev).eval(); DIM = net.config.hidden_size; NREG = getattr(net.config, 'num_register_tokens', 4)
def layer_grids(x):
    """x: normalised HxWx3 (multiples of 16) -> {L: [gh, gw, DIM] float32 on dev}; overlapping tiles, tile centres kept."""
    H, W = x.shape[:2]; gh, gw = H // 16, W // 16; OV = 128; STEP = a.tile - 2 * OV
    def one(t):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            hs = net(pixel_values=torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1)))[None].to(dev), output_hidden_states=True).hidden_states
        th, tw = t.shape[0] // 16, t.shape[1] // 16
        return {L: hs[L + 1][0, 1 + NREG:, :].float().reshape(th, tw, DIM) for L in LAYERS}
    if H <= a.tile and W <= a.tile: return one(x)
    out = {L: torch.zeros((gh, gw, DIM), device=dev) for L in LAYERS}
    for y0 in range(0, H, STEP):
        for x0 in range(0, W, STEP):
            ya, xa = max(0, y0 - OV), max(0, x0 - OV); t = x[ya:min(H, y0 + STEP + OV), xa:min(W, x0 + STEP + OV)]; t = t[:t.shape[0] // 16 * 16, :t.shape[1] // 16 * 16]
            T = one(t); ky, kx = (y0 - ya) // 16, (x0 - xa) // 16; ny, nx = min(STEP, H - y0) // 16, min(STEP, W - x0) // 16
            for L in LAYERS: out[L][y0 // 16:y0 // 16 + ny, x0 // 16:x0 // 16 + nx] = T[L][ky:ky + ny, kx:kx + nx]
    return out
def highpass(G, w):
    """G [gh,gw,D], w [gh,gw] in-mask weights -> token minus weighted 7x7 neighbourhood mean (in-mask patches only)."""
    Gw = (G * w[..., None]).permute(2, 0, 1)[None]; num = F.avg_pool2d(Gw, 7, 1, 3, count_include_pad=False)[0].permute(1, 2, 0)
    den = F.avg_pool2d(w[None, None], 7, 1, 3, count_include_pad=False)[0, 0]; return G - num / den.clamp_min(1e-3)[..., None]
def region_crop(f, scale):
    I = np.asarray(Image.open(os.path.join(a.images, f)).convert('RGB')); H, W = I.shape[:2]; m = np.zeros((H, W), np.uint8)
    for s in regions[f][0]: cv2.fillPoly(m, [np.array(s, np.float32).reshape(-1, 2).astype(np.int32)], 1)
    if a.dilate_px: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.dilate_px + 1,) * 2))
    if EXCL.get(f):   # v1c: subtract given masks (dilated)
        e = np.zeros((H, W), np.uint8)
        for s_ in EXCL[f]: cv2.fillPoly(e, [np.array(s_, np.float32).reshape(-1, 2).astype(np.int32)], 1)
        m = m & (1 - cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.exclude_dilate_px + 1,) * 2)))
    ys, xs = np.nonzero(m); y0, y1 = max(0, ys.min() - a.margin_px), min(H, ys.max() + a.margin_px + 1); x0, x1 = max(0, xs.min() - a.margin_px), min(W, xs.max() + a.margin_px + 1)
    C, M = I[y0:y1, x0:x1], m[y0:y1, x0:x1]
    if a.region_px: scale = scale * min(a.region_max_up, a.region_px / max(1, min(np.ptp(ys), np.ptp(xs))))   # 18 Oct: every region upsampled to the same size (Riedel lesson)
    if scale != 1: C = cv2.resize(C, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC); M = cv2.resize(M, (C.shape[1], C.shape[0]), interpolation=cv2.INTER_NEAREST)
    h16, w16 = max(16, C.shape[0] // 16 * 16), max(16, C.shape[1] // 16 * 16); C, M = C[:h16, :w16], M[:h16, :w16]
    return prep(C, M), M, (x0, y0)
def prep(C, M):
    """v1d input options; defaults (colour, none, clahe 0) return C unchanged."""
    if a.clahe > 0:
        L_ = cv2.cvtColor(C, cv2.COLOR_RGB2LAB); L_[..., 0] = cv2.createCLAHE(clipLimit=a.clahe, tileGridSize=(8, 8)).apply(L_[..., 0]); C = cv2.cvtColor(L_, cv2.COLOR_LAB2RGB)
    if a.input == 'grey': C = np.repeat(cv2.cvtColor(C, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)
    if a.norm == 'zscore' and (M > 0).sum() > 10:
        Cf = C.astype(np.float32); v = Cf[M > 0]; C = np.clip((Cf - v.mean(0)) / (v.std(0) + 1e-3) * 48 + 128, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(C)
def dark_dots(C, M):
    """baseline: black-top-hat local maxima inside the (undilated-equivalent) mask, per 100 px of region length (skeleton)."""
    g = cv2.cvtColor(C, cv2.COLOR_RGB2GRAY).astype(np.float32); bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    inside = M > 0; v = bh[inside]; thr = max(6.0, np.median(v) + 4 * 1.4826 * np.median(np.abs(v - np.median(v))))
    pk = (bh == cv2.dilate(bh, np.ones((7, 7), np.uint8))) & (bh > thr) & inside; ys, xs = np.nonzero(pk)
    sk = cv2.ximgproc.thinning(M * 255) if hasattr(cv2, 'ximgproc') else None
    length = (sk > 0).sum() if sk is not None else max(M.shape) ; return len(ys) / max(1, length) * 100, np.stack([xs, ys], 1)
# ---------- extraction ----------
CFG = [(s, L, v) for s in SCALES for L in LAYERS for v in ('raw', 'hp')]
AGG = {c: {} for c in CFG}; PAT = {c: {} for c in CFG}; DOTS = {}; KEPT = {}; t0 = time.time()
for i, f in enumerate(files):
    for s in SCALES:
        C, M, _ = region_crop(f, s)
        if s == 1: DOTS[f] = dark_dots(C, M)[0]
        x = (C.astype(np.float32) / 255 - MEAN) / STD; Gs = layer_grids(x); gh, gw = Gs[LAYERS[0]].shape[:2]
        w = cv2.resize(M.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA); inside = w >= 0.5
        if inside.sum() < 3: continue
        if a.keep_majority_cluster >= 2:   # v1b (16 Oct, AVD): auto sub-region = largest token cluster inside the region
            from sklearn.cluster import KMeans
            Lr = a.primary_layer if a.primary_layer in LAYERS else LAYERS[-1]; Xc = Gs[Lr].reshape(-1, DIM)[torch.from_numpy(np.flatnonzero(inside)).to(dev)].cpu().numpy()
            Xc = PCA(min(16, len(Xc) - 1), random_state=0).fit_transform(Xc); kl = KMeans(a.keep_majority_cluster, n_init=4, random_state=0).fit_predict(Xc)
            sub = np.zeros(inside.sum(), bool); sub[kl == np.bincount(kl).argmax()] = True; inside = inside.copy(); inside[inside] = sub
            if s == 1: KEPT[f] = float(sub.mean())
        wt = torch.from_numpy(inside.astype(np.float32)).to(dev); ii = torch.from_numpy(np.flatnonzero(inside)).to(dev)
        for L in LAYERS:
            for v, G in (('raw', Gs[L]), ('hp', highpass(Gs[L], wt))):
                T = G.reshape(-1, DIM)[ii].cpu().numpy(); AGG[(s, L, v)][f] = np.concatenate([T.mean(0), np.percentile(T, 90, axis=0)]).astype(np.float32)
                k = rng.choice(len(T), min(len(T), a.patches_per_image), replace=False); PAT[(s, L, v)][f] = T[k].astype(np.float16)
        del Gs
    if (i + 1) % 25 == 0 or i + 1 == len(files): log(f'  extracted {i + 1}/{len(files)}  {time.time() - t0:.0f}s')
np.savez_compressed(os.path.join(a.out, 'aggregates.npz'), files=np.array(files), **{f'{s}_{L}_{v}': np.stack([AGG[(s, L, v)].get(f, np.zeros(2 * DIM, np.float32)) for f in files]) for (s, L, v) in CFG})
# ---------- evaluation ----------
yv = lambda fs: np.array([LAB[taxon(f)] for f in fs]); grp = lambda fs: np.array([taxon(f) for f in fs])
def score(fs, sc):
    y, g = yv(fs), grp(fs); tg = sorted(set(g)); ym = np.array([y[g == t].mean() for t in tg]); sm = np.array([sc[g == t].mean() for t in tg])
    if numeric: return spearmanr(ym, sm)[0], spearmanr(y, sc)[0]
    return roc_auc_score(ym, sm), roc_auc_score(y, sc)
FG = {r[0]: r[1] for r in csv.reader(open(a.fold_groups), delimiter='\t') if len(r) > 1 and not r[0].startswith('#')} if a.fold_groups else {}
fold = lambda fs: np.array([FG.get(taxon(f), taxon(f)) for f in fs])   # v1c: hold out whole GROUPS (e.g. papers); scores stay per taxon
if FG: log(f'fold groups: {len(set(FG.values()))} groups from {a.fold_groups} (whole groups held out)')
def logo(fit_predict, fs, y):
    g = fold(fs); out = np.zeros(len(fs))
    for t in sorted(set(g)):
        te = g == t
        if len(set(y[~te])) < 2: out[te] = np.nan; continue
        out[te] = fit_predict(~te, te, y)
    return out
def method_A(c, fs):
    X = np.stack([AGG[c][f] for f in fs])
    def fp(tr, te, y):
        k = min(32, tr.sum() - 1); mdl = make_pipeline(StandardScaler(), PCA(k, random_state=0), Ridge(1.0) if numeric else LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000))
        mdl.fit(X[tr], y[tr]); return mdl.predict(X[te]) if numeric else mdl.predict_proba(X[te])[:, 1]
    return fp
def method_B(c, fs):
    P_ = [PAT[c][f] for f in fs]
    def fp(tr, te, y):
        Xt = np.vstack([P_[j] for j in np.flatnonzero(tr)]).astype(np.float32); Yt = np.concatenate([np.full(len(P_[j]), y[j]) for j in np.flatnonzero(tr)])
        mdl = make_pipeline(StandardScaler(), PCA(64, random_state=0), Ridge(1.0) if numeric else LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000)); mdl.fit(Xt, Yt)
        return np.array([(mdl.predict(P_[j].astype(np.float32)) if numeric else mdl.predict_proba(P_[j].astype(np.float32))[:, 1]).mean() for j in np.flatnonzero(te)])
    return fp
def null_scores(fp, fs, n):
    global LAB
    y = yv(fs); g = grp(fs); tg = sorted(set(g)); tv = {t: LAB[t] for t in tg}; vals = []; keep = dict(LAB)
    for _ in range(n):
        perm = dict(zip(tg, rng.permutation([tv[t] for t in tg]))); LAB = {**keep, **perm}; yp = yv(fs); sc = logo(fp, fs, yp); ok = ~np.isnan(sc)
        try: vals.append(score(list(np.array(fs)[ok]), sc[ok])[0])
        except ValueError: pass
    LAB = keep; return np.nanpercentile(vals, 95) if vals else np.nan
fs = [f for f in lab_files if all(f in AGG[c] for c in CFG)]; y = yv(fs)
if KEPT: log(f'keep_majority_cluster {a.keep_majority_cluster}: kept share of region patches median {np.median(list(KEPT.values())):.2f} (min {min(KEPT.values()):.2f}, max {max(KEPT.values()):.2f})')
log(f'\nEVALUATION on {len(fs)} labelled images, {len(set(grp(fs)))} taxa; task = {"numeric (Spearman)" if numeric else "binary (AUC), positive = " + a.positive}')
if not numeric: log('  taxa:', {t: (int(LAB[t]), int((grp(fs) == t).sum())) for t in sorted(set(grp(fs)))})
d_ = np.array([DOTS[f] for f in fs]); log(f'BASELINE dark-dot density (no training): taxon-level {score(fs, d_)[0]:.3f}, image-level {score(fs, d_)[1]:.3f}')
rows = []
for c in CFG + [(s, L, 'raw+hp') for s in SCALES for L in LAYERS]:
    if c[2] == 'raw+hp':
        for f in fs: AGG[c] = AGG.get(c, {}); AGG[c][f] = np.concatenate([AGG[(c[0], c[1], 'raw')][f], AGG[(c[0], c[1], 'hp')][f]])
        PAT[c] = {f: np.concatenate([PAT[(c[0], c[1], 'raw')][f], PAT[(c[0], c[1], 'hp')][f]], 1) for f in PAT[(c[0], c[1], 'raw')]}
    for mname, mk in (('A', method_A), ('B', method_B)):
        fp = mk(c, fs); sc = logo(fp, fs, y); ok = ~np.isnan(sc); s_t, s_i = score(list(np.array(fs)[ok]), sc[ok])
        prim = (mname == 'B' and c == (1.0, a.primary_layer, 'raw+hp'))
        nl = null_scores(fp, fs, a.null if mname == 'A' else a.null_b) if (mname == 'A' or prim) else np.nan
        rows.append((mname, c, s_t, s_i, nl)); log(f'  {"PRIMARY " if prim else ""}{mname} scale {c[0]:g} L{c[1]} {c[2]:7s}: taxon {s_t:.3f}  image {s_i:.3f}  null95 {nl:.3f}')
        with open(os.path.join(a.out, f'scores_{mname}_s{c[0]:g}_L{c[1]}_{c[2]}.tsv'), 'w') as fh:
            fh.write('file\ttaxon\tlabel\tscore\n'); [fh.write(f'{f}\t{taxon(f)}\t{LAB[taxon(f)]}\t{v:.4f}\n') for f, v in zip(fs, sc)]
# ---------- primary model on all labelled taxa -> unlabelled images + sheet ----------
c = (1.0, a.primary_layer, 'raw+hp'); allf = [f for f in files if f in PAT[(1.0, a.primary_layer, 'raw')]]
for f in allf:
    if f not in PAT[c]: PAT[c][f] = np.concatenate([PAT[(1.0, a.primary_layer, 'raw')][f], PAT[(1.0, a.primary_layer, 'hp')][f]], 1)
Xt = np.vstack([PAT[c][f] for f in fs]).astype(np.float32); Yt = np.concatenate([np.full(len(PAT[c][f]), LAB[taxon(f)]) for f in fs])
mdl = make_pipeline(StandardScaler(), PCA(64, random_state=0), Ridge(1.0) if numeric else LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000)).fit(Xt, Yt)
pred = lambda T: mdl.predict(T) if numeric else mdl.predict_proba(T)[:, 1]
with open(os.path.join(a.out, 'specimen_scores_primary.tsv'), 'w') as fh:
    fh.write('file\ttaxon\tlabelled\tlabel\tscore_all_taxa_model\tdark_dots_per_100px\tmask_source\n')
    for f in allf: fh.write(f'{f}\t{taxon(f)}\t{int(taxon(f) in LAB)}\t{LAB.get(taxon(f), "")}\t{pred(PAT[c][f].astype(np.float32)).mean():.4f}\t{DOTS.get(f, np.nan):.2f}\t{regions[f][1]}\n')
log('wrote specimen_scores_primary.tsv (unlabelled images scored by the model fitted on all labelled taxa)')
# sheet: one image per taxon, held-out patch map for labelled taxa
sc_prim = dict(zip(fs, logo(method_B(c, fs), fs, y))); tiles = []
for t in sorted({taxon(f) for f in allf}):
    f = sorted(x for x in allf if taxon(x) == t)[0]; C, M, _ = region_crop(f, 1.0); x = (C.astype(np.float32) / 255 - MEAN) / STD; Gs = layer_grids(x); gh, gw = Gs[a.primary_layer].shape[:2]
    w = cv2.resize(M.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA); inside = w >= 0.5; wt = torch.from_numpy(inside.astype(np.float32)).to(dev)
    G = Gs[a.primary_layer]; T = torch.cat([G, highpass(G, wt)], -1).reshape(-1, 2 * DIM).cpu().numpy()
    if t in LAB:   # held-out map: refit without this taxon
        tr = grp(fs) != t; Xt_ = np.vstack([PAT[c][g] for g, k in zip(fs, tr) if k]).astype(np.float32); Yt_ = np.concatenate([np.full(len(PAT[c][g]), LAB[taxon(g)]) for g, k in zip(fs, tr) if k])
        m_ = make_pipeline(StandardScaler(), PCA(64, random_state=0), Ridge(1.0) if numeric else LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000)).fit(Xt_, Yt_)
        p = m_.predict(T) if numeric else m_.predict_proba(T)[:, 1]
    else: p = pred(T)
    pm = np.where(inside.ravel(), p, np.nan).reshape(gh, gw); lo, hi = (0, 1) if not numeric else (np.nanmin(list(LAB.values())), np.nanmax(list(LAB.values())))
    col = cv2.applyColorMap(np.uint8(255 * np.clip((np.nan_to_num(pm, nan=lo) - lo) / max(1e-6, hi - lo), 0, 1)), cv2.COLORMAP_MAGMA)
    col = cv2.resize(col, (C.shape[1], C.shape[0]), interpolation=cv2.INTER_NEAREST); col[cv2.resize(inside.astype(np.uint8), (C.shape[1], C.shape[0]), interpolation=cv2.INTER_NEAREST) == 0] = 255
    _, pts = dark_dots(C, M); dd = C[:, :, ::-1].copy()
    for (px, py) in pts: cv2.circle(dd, (int(px), int(py)), 4, (0, 255, 0), 1)
    Z = 2; blk = np.vstack([cv2.resize(im, None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC) for im in (C[:, :, ::-1], col, dd)])[:, :1800]
    hdr = np.full((28, blk.shape[1], 3), 255, np.uint8); s_ = sc_prim.get(f, pred(PAT[c][f].astype(np.float32)).mean())
    cv2.putText(hdr, f'{t} | label {LAB.get(t, "unlabelled")} | score {s_:.2f} ({"held out" if t in LAB else "all-taxa model"}) | {f[:40]}', (4, 20), 0, 0.55, (0, 0, 0), 1); tiles.append(np.vstack([hdr, blk]))
    del Gs
Wd = max(x.shape[1] for x in tiles); S = np.vstack([cv2.copyMakeBorder(x, 0, 10, 0, Wd - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in tiles])
cv2.imwrite(os.path.join(a.out, 'sheet_primary.jpg'), S, [cv2.IMWRITE_JPEG_QUALITY, 85]); log('wrote sheet_primary.jpg (native crop x2 | held-out P map | dark dots)'); log('DONE')
