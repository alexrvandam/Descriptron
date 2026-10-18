"""BioRAG-Probe, per-PART version (17 Oct 2026). A literature statement that labels PARTS rather than taxa (e.g. Laffoon 1957 for
Mycetophila: 'C, Sc, R1 and Rs always with setulae above and below ... branches of M and Cu bare below, nearly always sparsely
setulose above') -> are the DINOv3 tokens inside each part's mask consistent with it, on a part the probe never saw?

Validation (declared before running):
  every (image, part) is one sample; folds = leave-one-PART-out x species 3-fold (the test part AND the test species are never in
  training), so the probe cannot learn 'which vein is this' or 'which species is this'.
  A = per (image, part): mean + p90 of tokens (L raw and L high-pass) -> PCA(32) -> logistic.
  B = patch probe: logistic on patch tokens (raw+hp), sample score = mean patch probability.  PRIMARY = B, --primary_layer, raw+hp.
  Baselines with the same folds: dark-dot density (black-top-hat spots / mask area) and darkness+width (mean grey, mask area /
  skeleton length) -> a probe that only learns 'thick dark vein' is caught here.
  Null: part labels permuted among the parts (same number positive), --null times, 95th percentile.
Usage: script --images DIR --coco GT.json --part_labels TSV(part<TAB>label) --positive LABEL --out DIR [--taxon_regex RE]."""
import argparse, json, os, re, time, csv
import numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
ap = argparse.ArgumentParser()
ap.add_argument('--images', required=True); ap.add_argument('--coco', required=True); ap.add_argument('--part_labels', required=True)
ap.add_argument('--positive', required=True); ap.add_argument('--taxon_regex', default=r'^([A-Za-z]+?)_'); ap.add_argument('--out', required=True)
ap.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m'); ap.add_argument('--layers', default='12,18'); ap.add_argument('--primary_layer', type=int, default=18)
ap.add_argument('--dilate_px', type=int, default=6); ap.add_argument('--tile', type=int, default=1024); ap.add_argument('--patches', type=int, default=48)
ap.add_argument('--folds', type=int, default=3); ap.add_argument('--null', type=int, default=200); ap.add_argument('--null_a', type=int, default=30); ap.add_argument('--null_b', type=int, default=10); ap.add_argument('--max_images', type=int, default=0)
ap.add_argument('--apply_parts', default='', help='v1b: parts NOT used in training (e.g. bm-m); the PRIMARY B model of each taxon fold scores them in held-out taxa')
ap.add_argument('--apply_labels', default='', help='v1b: TSV taxon<TAB>label for the apply parts (e.g. literature bM setose/bare), positive = --positive_apply')
ap.add_argument('--positive_apply', default='setose')
ap.add_argument('--maps', type=int, default=6, help='images shown with held-out patch probabilities on every labelled part'); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); LAYERS = [int(x) for x in a.layers.split(',')]; LOG = open(os.path.join(a.out, 'report.txt'), 'w')
def log(*s): print(*s, flush=True); print(*s, file=LOG); LOG.flush()
PL = {r[0]: r[1] for r in csv.reader(open(a.part_labels), delimiter='\t') if r and not r[0].startswith('#') and len(r) >= 2}
APPLY = [x for x in a.apply_parts.split(',') if x]; SA = []
d = json.load(open(a.coco)); cat = {c['id']: c['name'] for c in d['categories']}; by = {}
for an in d['annotations']:
    if cat[an['category_id']] in PL or cat[an['category_id']] in APPLY: by.setdefault(an['image_id'], []).append(an)
ims = [im for im in d['images'] if im['id'] in by]; ims = ims[:a.max_images] if a.max_images else ims
taxon = lambda f: (re.search(a.taxon_regex, f) or re.search('(.*)', f)).group(1)
log(f'BioRAG-Probe parts v1 {time.strftime("%Y-%m-%d %H:%M")} | {len(ims)} images, {len({taxon(i["file_name"]) for i in ims})} taxa | parts {PL} | positive {a.positive} | layers {LAYERS} | model {a.model}')
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
rng = np.random.default_rng(0); S = []; PATCH = []; MAPS = {}
for n, im in enumerate(ims):
    I = np.asarray(Image.open(os.path.join(a.images, im['file_name'])).convert('RGB')); H, W = I.shape[:2]; H16, W16 = H // 16 * 16, W // 16 * 16
    x = (I[:H16, :W16].astype(np.float32) / 255 - MEAN) / STD; G = grids(x); gh, gw = H16 // 16, W16 // 16
    grey = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY); bh = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    allm = np.zeros((H, W), np.uint8)
    masks = {}
    for an in by[im['id']]:
        m = np.zeros((H, W), np.uint8)
        for p in an['segmentation']: cv2.fillPoly(m, [np.array(p).reshape(-1, 2).astype(np.int32)], 1)
        nm = cat[an['category_id']]; masks[nm] = masks.get(nm, 0) | m; allm |= m
    wmask = torch.from_numpy(cv2.resize(cv2.dilate(allm, np.ones((41, 41), np.uint8)), (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)).to(dev).clamp(max=1)
    HP = {}
    for L in LAYERS:   # high-pass: token minus the mean of its 7x7 neighbourhood near the veins
        T = G[L]; num = F.avg_pool2d((T * wmask[..., None]).permute(2, 0, 1)[None], 7, 1, 3, count_include_pad=False)[0].permute(1, 2, 0)
        den = F.avg_pool2d(wmask[None, None], 7, 1, 3, count_include_pad=False)[0, 0]; HP[L] = T - num / den.clamp_min(1e-6)[..., None]
    for nm, m in masks.items():
        md = cv2.dilate(m, np.ones((2 * a.dilate_px + 1,) * 2, np.uint8)); cov = cv2.resize(md[:H16, :W16].astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
        cells = np.flatnonzero(cov.ravel() >= 0.3)
        if len(cells) < 3: continue
        cs_, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); length = sum(cv2.arcLength(c, True) for c in cs_) / 2   # half perimeter ~ vein length
        lab_ = PL.get(nm, 'APPLY')
        rec = dict(image=im['file_name'], taxon=taxon(im['file_name']), part=nm, label=lab_, y=int(lab_ == a.positive),
                   dark_dots=float((bh[md > 0] > 25).mean()), grey=float(grey[m > 0].mean()), width=float(m.sum() / max(length, 1)), n_cells=len(cells))
        ci = torch.from_numpy(cells).to(dev)
        for L in LAYERS:
            R = G[L].reshape(-1, DIM)[ci]; P = HP[L].reshape(-1, DIM)[ci]
            rec[f'A{L}'] = torch.cat([R.mean(0), torch.quantile(R, 0.9, 0), P.mean(0), torch.quantile(P, 0.9, 0)]).cpu().numpy()
            if L == a.primary_layer:
                sel = cells if len(cells) <= a.patches else rng.choice(cells, a.patches, replace=False); si = torch.from_numpy(np.sort(sel)).to(dev)
                PATCH.append(torch.cat([G[L].reshape(-1, DIM)[si], HP[L].reshape(-1, DIM)[si]], 1).cpu().numpy().astype(np.float16)); rec['patch_idx'] = len(PATCH) - 1
                if n < a.maps: MAPS.setdefault(im['file_name'], {})[nm] = (cells, torch.cat([G[L].reshape(-1, DIM)[ci], HP[L].reshape(-1, DIM)[ci]], 1).cpu().numpy().astype(np.float16), (gh, gw))
        (SA if nm in APPLY else S).append(rec)
    print(f'{n + 1}/{len(ims)} {im["file_name"]} parts {len(masks)}', flush=True); del G, HP; torch.cuda.empty_cache()
parts = sorted({r['part'] for r in S}); taxa = sorted({r['taxon'] for r in S}); tf = {t: i % a.folds for i, t in enumerate(rng.permutation(taxa))}
log(f'samples {len(S)} (image x part); parts: ' + ', '.join(f'{p}={PL[p]}({sum(r["part"] == p for r in S)})' for p in parts))
def cv(feat, y_of, clf_fn):
    """leave-one-part-out x taxon fold; returns held-out scores aligned with S."""
    sc = np.full(len(S), np.nan)
    for p in parts:
        for k in range(a.folds):
            te = [i for i, r in enumerate(S) if r['part'] == p and tf[r['taxon']] == k]; tr = [i for i, r in enumerate(S) if r['part'] != p and tf[r['taxon']] != k]
            if not te or len({y_of[i] for i in tr}) < 2: continue
            sc[te] = clf_fn(feat, tr, te, y_of)
    return sc
def clfA(Xs):
    def f(_, tr, te, y):
        m = make_pipeline(StandardScaler(), PCA(min(32, len(tr) - 1), random_state=0), LogisticRegression(C=0.1, max_iter=2000)).fit(Xs[tr], y[tr]); return m.predict_proba(Xs[te])[:, 1]
    return f
def clfB(_, tr, te, y):
    X = np.concatenate([PATCH[S[i]['patch_idx']] for i in tr]).astype(np.float32); yy = np.concatenate([[y[i]] * len(PATCH[S[i]['patch_idx']]) for i in tr])
    m = make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.05, max_iter=2000)).fit(X, yy)
    return np.array([m.predict_proba(PATCH[S[i]['patch_idx']].astype(np.float32))[:, 1].mean() for i in te])
def auc(sc, y): ok = ~np.isnan(sc); return roc_auc_score(y[ok], sc[ok]) if len(set(y[ok])) == 2 else np.nan
def part_auc(sc, y):   # AUC over PART means (each part one point): the honest n
    pm = {p: np.nanmean([sc[i] for i, r in enumerate(S) if r['part'] == p]) if np.isfinite([sc[i] for i, r in enumerate(S) if r['part'] == p]).any() else np.nan for p in parts}
    ok = [p for p in parts if np.isfinite(pm[p])]   # v1b: a part with no held-out score (e.g. the only positive part) is skipped
    yy = [int(PL[p] == a.positive) for p in ok]
    return (roc_auc_score(yy, [pm[p] for p in ok]) if len(set(yy)) == 2 else np.nan), pm
y = np.array([r['y'] for r in S]); pos_n = int(sum(PL[p] == a.positive for p in parts))
def nulls(fn, reps):
    out = []
    for _ in range(reps):
        lab = dict(zip(parts, rng.permutation([PL[p] == a.positive for p in parts]))); yp = np.array([int(lab[r['part']]) for r in S]); out.append(auc(cv(None, yp, fn), yp))
    return np.nanpercentile(out, 95)
res = {}
XD = np.array([[r['dark_dots']] for r in S]); XG = np.array([[r['grey'], r['width']] for r in S])
def clf_small(X):
    def f(_, tr, te, y): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return f
for name, fn in [('BASELINE dark dots', clf_small(XD)), ('BASELINE darkness+width', clf_small(XG))] + [(f'A L{L} raw+hp mean+p90', clfA(np.stack([r[f'A{L}'] for r in S]))) for L in LAYERS]:
    sc = cv(None, y, fn); res[name] = sc; pa, pm = part_auc(sc, y)
    log(f'{name:28s}: sample AUC {auc(sc, y):.3f}  part-mean AUC {pa:.3f}  null95 {nulls(fn, a.null if not name.startswith("A") else a.null_a):.3f}')
sc = cv(None, y, clfB); res['B'] = sc; pa, pm = part_auc(sc, y)
log(f'PRIMARY B L{a.primary_layer} raw+hp patches   : sample AUC {auc(sc, y):.3f}  part-mean AUC {pa:.3f}  null95 {nulls(clfB, a.null_b):.3f}  (null from {a.null_b} permutations)')
log('held-out mean score per part (PRIMARY B): ' + ', '.join(f'{p}[{PL[p]}] {pm[p]:.2f}' for p in sorted(parts, key=lambda p: -pm[p])))
with open(os.path.join(a.out, 'scores.tsv'), 'w') as fh:
    ks = list(res); fh.write('image\ttaxon\tpart\tlabel\tdark_dots\tgrey\twidth\t' + '\t'.join(ks) + '\n')
    for i, r in enumerate(S): fh.write(f'{r["image"]}\t{r["taxon"]}\t{r["part"]}\t{r["label"]}\t{r["dark_dots"]:.4f}\t{r["grey"]:.1f}\t{r["width"]:.1f}\t' + '\t'.join(f'{res[k][i]:.3f}' for k in ks) + '\n')
# maps: patch probabilities from a B model that never saw that image's taxon fold (all parts in training -> shows WHERE on each vein)
for fn_, D in MAPS.items():
    k = tf[taxon(fn_)]; tr = [i for i, r in enumerate(S) if tf[r['taxon']] != k]
    X = np.concatenate([PATCH[S[i]['patch_idx']] for i in tr]).astype(np.float32); yy = np.concatenate([[y[i]] * len(PATCH[S[i]['patch_idx']]) for i in tr])
    m = make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.05, max_iter=2000)).fit(X, yy)
    I = np.asarray(Image.open(os.path.join(a.images, fn_)).convert('RGB')); gh, gw = next(iter(D.values()))[2]; Pm = np.full(gh * gw, np.nan)
    for nm, (cells, Xc, _) in D.items(): Pm[cells] = m.predict_proba(Xc.astype(np.float32))[:, 1]
    Pm = Pm.reshape(gh, gw); val = ~np.isnan(Pm); Mx = np.float32([[16, 0, 8], [0, 16, 8]]); H, W = I.shape[:2]
    pu = cv2.warpAffine(np.nan_to_num(Pm).astype(np.float32), Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    vu = cv2.warpAffine(val.astype(np.float32), Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    p = np.where(vu > 0.3, pu / np.maximum(vu, 1e-6), np.nan); col = cv2.applyColorMap((np.nan_to_num(p) * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)[..., ::-1]
    O = I.copy(); sel = ~np.isnan(p); O[sel] = (0.25 * O[sel] + 0.75 * col[sel]).astype(np.uint8)
    Image.fromarray(np.vstack([I, O])).save(os.path.join(a.out, f'map_{os.path.splitext(fn_)[0]}.jpg'), quality=88)
log('maps: map_<image>.jpg = photo above, held-out-taxon patch probability of the POSITIVE class on every labelled part below (plasma: yellow = positive).')
if SA:   # v1b: apply the B model (trained on the labelled parts of the other taxon folds) to parts it never saw
    LA = {r[0]: r[1] for r in csv.reader(open(a.apply_labels), delimiter='\t') if r and len(r) >= 2} if a.apply_labels else {}
    out = []
    for k in range(a.folds):
        tr = [i for i, r in enumerate(S) if tf.get(r['taxon'], -1) != k]
        te = [r for r in SA if tf.get(r['taxon'], hash(r['taxon']) % a.folds) == k]
        if not te: continue
        X = np.concatenate([PATCH[S[i]['patch_idx']] for i in tr]).astype(np.float32); yy = np.concatenate([[y[i]] * len(PATCH[S[i]['patch_idx']]) for i in tr])
        m = make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.05, max_iter=2000)).fit(X, yy)
        for r in te: out.append((r['image'], r['taxon'], r['part'], float(m.predict_proba(PATCH[r['patch_idx']].astype(np.float32))[:, 1].mean())))
    with open(os.path.join(a.out, 'apply_scores.tsv'), 'w') as fh:
        fh.write('image\ttaxon\tpart\tP_positive\tliterature\n'); [fh.write(f'{i}\t{t}\t{p}\t{v:.3f}\t{LA.get(t, "-")}\n') for i, t, p, v in out]
    for part in APPLY:
        tm = {}
        for i, t, p, v in out:
            if p == part: tm.setdefault(t, []).append(v)
        log(f'APPLY {part} (never in training; taxon held out): mean P({a.positive}) per taxon, literature label:')
        for t in sorted(tm, key=lambda t: -np.mean(tm[t])): log(f'   {t:22s} {np.mean(tm[t]):.2f} (n={len(tm[t])})  {LA.get(t, "-")}')
        lt = [t for t in tm if t in LA]
        if len({LA[t] for t in lt}) == 2:
            yt = np.array([LA[t] == a.positive_apply for t in lt]); st = np.array([np.mean(tm[t]) for t in lt]); A_ = roc_auc_score(yt, st)
            nl = [roc_auc_score(rng.permutation(yt), st) for _ in range(2000)]
            log(f'APPLY {part}: taxon AUC {A_:.3f} vs literature ({len(lt)} taxa, {int(yt.sum())} {a.positive_apply}); permutation null95 {np.percentile(nl, 95):.3f}')
log('done')
