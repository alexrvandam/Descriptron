#!/usr/bin/env python3
"""
DINOMask hypercolumn probe v1 (25 Oct 2026): find a small, repeated structure (setae, scales, punctures, tubercles, pores ...)
on every specimen from a few clicks.

Method (A. R. Van Dam's idea of reading the same spot at several depths of DINOv3): the frozen DINOv3 ViT-L/16 backbone
reads each image once; the patch tokens of layers 6, 12, 18 and 23 (shallow = fine detail, deep = what the thing is) are
reduced to 64 PCA components each and concatenated into a 256-number "hypercolumn" per 16-px patch (Hariharan et al. 2015,
CVPR, for the hypercolumn idea). A logistic regression on the hypercolumns separates the clicked structure from everything
else. Four half-patch-shifted passes give ~8-px precision. On Tri-20 (Trigonopterus), taught on the pronotum and tested on
the unseen elytra, cream scale-like setae: AUC .997, AP .972 (layer 23 alone .962 / .867).

Clicks come from the Descriptron GUI keypoint tool: POSITIVE points (left click) = the structure; NEGATIVE points (marked
not-this, saved with visibility 0) = look-alikes it must ignore (glare, debris, sockets). More negatives are sampled
automatically from the specimen (fg_mask_v1.py) away from the positive clicks. Input files: the GUI's COCO save or the
per-image auto-save <image>_keypoints.json (both carry 'keypoints_clicked' = x, y, v for every click), several allowed.

  train (+ predict): --points a.json[,b_keypoints.json,...|folder] --images DIR --name scale_seta --out RUN
                     [--predict_dir DIR | --predict_list list.txt] [--ablate]
  predict only:      --run RUN --predict_dir DIR --out OUT

Writes (RUN or OUT): probe.joblib (PCA bases, scaler, classifier, settings), results.json (clicks used, cross-validated AUC /
AP: leave-one-image-out with >= 2 clicked images, else a 2-fold left/right split of one image; per-layer ablation with
--ablate), predictions_keypoints.json (one keypoint annotation per image in the GUI's own format: opens as editable points,
so wrong points can be deleted, missed ones added, and the probe retrained), counts.json (per image; per part with
--within_coco), overlay_<image>.jpg.
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fg_mask_v1 import fg_mask

Image.MAX_IMAGE_PIXELS = None
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
SHIFTS = ((0, 0), (8, 0), (0, 8), (8, 8))                       # half-patch shifts (pixels of the resized image)
ap = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--points', default='', help='comma list of COCO / <image>_keypoints.json files or folders holding them')
ap.add_argument('--images', default='', help='folder of the clicked images (file_name in the COCO is looked up here)')
ap.add_argument('--name', default='structure', help='what was clicked, e.g. scale_seta (used in the outputs)')
ap.add_argument('--run', default='', help='predict with an already trained probe (folder with probe.joblib)')
ap.add_argument('--predict_dir', default=''); ap.add_argument('--predict_list', default='', help='text file, one image path per line')
ap.add_argument('--out', required=True)
ap.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m')
ap.add_argument('--layers', default='6,12,18,23'); ap.add_argument('--pca', type=int, default=64)
ap.add_argument('--size', type=int, default=2048, help='longest side in px fed to DINOv3 (multiple of 16)')
ap.add_argument('--auto_neg', type=int, default=1500, help='automatic negative patches per training image')
ap.add_argument('--neg_dist', type=float, default=2.0, help='automatic negatives at least this many patches from a positive click')
ap.add_argument('--neg_far', type=float, default=8.0, help='... and at most this many (local negatives: the cuticle around each click, so the '
                'probe learns the structure, not the body region the clicks happen to be on); 0 = anywhere on the specimen')
ap.add_argument('--highpass', type=float, default=3.0, help='subtract each feature\'s local mean over this many patches (sigma) inside the '
                'specimen, removing where-on-the-body signal; 0 = off')
ap.add_argument('--C', type=float, default=0.05, help='logistic regression inverse regularisation')
ap.add_argument('--threshold', type=float, default=0.5); ap.add_argument('--min_dist', type=float, default=0.75,
                help='minimum distance between two detections, in patches (16 px of the resized image)')
ap.add_argument('--within_coco', default='', help='DINOMask predictions (RLE or polygons) to count detections per part')
ap.add_argument('--within', default='', help='comma list of parts: keep only detections inside these (needs --within_coco)')
ap.add_argument('--ablate', action='store_true', help='also cross-validate each layer alone and layers 6+23')
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


# ------------------------------------------------------------------ clicks
def click_files(spec):
    fs = []
    for s in [x for x in spec.split(',') if x.strip()]:
        s = s.strip()
        fs += sorted(glob.glob(os.path.join(s, '*.json'))) if os.path.isdir(s) else [s]
    return fs


def read_clicks(files, images_dir):
    """{image path: (positives Nx2, negatives Mx2)} in original image pixels, from 'keypoints_clicked' (or 'keypoints')."""
    clicks = {}
    for f in files:
        d = json.load(open(f)); ims = {im['id']: im for im in d.get('images', [])}
        for an in d.get('annotations', []):
            kp = an.get('keypoints_clicked') or an.get('keypoints')
            if not kp or an.get('image_id') not in ims: continue
            im = ims[an['image_id']]; fn = im.get('file_name', '')
            cands = [fn if os.path.isabs(fn) else os.path.join(images_dir or os.path.dirname(os.path.abspath(f)), fn),
                     os.path.join(os.path.dirname(os.path.abspath(f)), os.path.basename(fn))]
            path = next((c for c in cands if os.path.isfile(c)), None)
            if path is None: print(f'[probe] image not found for {fn} ({f}); skipped'); continue
            k = np.asarray(kp, float).reshape(-1, 3); pos = k[k[:, 2] > 0, :2]; neg = k[(k[:, 2] <= 0) & (k[:, :2].sum(1) > 0), :2]
            p0, n0 = clicks.get(path, (np.zeros((0, 2)), np.zeros((0, 2))))
            clicks[path] = (np.vstack([p0, pos]), np.vstack([n0, neg]))
    return clicks


# ------------------------------------------------------------------ DINOv3
net = None
def load_net(model):
    global net
    if net is None:
        from transformers import AutoModel
        net = AutoModel.from_pretrained(model).to(dev).eval()
    return net


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def resized(rgb, side):
    h, w = rgb.shape[:2]; s = side / max(h, w); W, H = int(round(w * s)), int(round(h * s))
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA), s


def tokens(rgb_s, layers, shift):
    """rgb_s: resized image; returns {layer: (gh, gw, D)} for the grid starting at pixel (sy, sx)."""
    sy, sx = shift; im = rgb_s[sy:, sx:]; H, W = im.shape[0] // 16 * 16, im.shape[1] // 16 * 16; im = im[:H, :W]
    x = torch.tensor(((im.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1)[None]).to(dev)
    m = load_net(a.model); nreg = m.config.num_register_tokens
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
        hs = m(pixel_values=x, output_hidden_states=True).hidden_states
    return {L: hs[L + 1][0, 1 + nreg:].float().cpu().numpy().reshape(H // 16, W // 16, -1) for L in layers}


def fg_grid(fgs, shift, gh, gw):
    sy, sx = shift; m = fgs[sy:sy + gh * 16, sx:sx + gw * 16].astype(np.float32)
    return cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA) > 0.5


def hyper(T, pcas, layers, fg=None, hp=0.0):
    """Concatenated per-layer PCA features; hp > 0: minus their local mean inside the specimen (normalised convolution)."""
    gh, gw = T[layers[0]].shape[:2]
    F = np.concatenate([pcas[L].transform(T[L].reshape(gh * gw, -1)).astype(np.float32) for L in layers], 1).reshape(gh, gw, -1)
    if hp and fg is not None:
        M = fg.astype(np.float32); den = cv2.GaussianBlur(M, (0, 0), hp) + 1e-6
        F = F - np.stack([cv2.GaussianBlur(F[..., k] * M, (0, 0), hp) for k in range(F.shape[-1])], -1) / den[..., None]
    return F


# ------------------------------------------------------------------ probability map + peaks
def prob_map(rgb, probe):
    """Mean of the 4 shifted passes, bilinear, on the resized image grid. Returns (prob HxW, fg HxW, scale)."""
    rgb_s, s = resized(rgb, probe['size']); H, W = rgb_s.shape[:2]; fgs = fg_mask(rgb_s)
    acc = np.zeros((H, W), np.float32); wt = np.zeros((H, W), np.float32)
    for sh in SHIFTS:
        T = tokens(rgb_s, probe['layers'], sh); gh, gw = T[probe['layers'][0]].shape[:2]
        F = hyper(T, probe['pcas'], probe['layers'], fg_grid(fgs, sh, gh, gw), probe.get('highpass', 0)).reshape(gh * gw, -1)
        p = probe['clf'].predict_proba(probe['scaler'].transform(F))[:, 1].reshape(gh, gw).astype(np.float32)
        up = cv2.resize(p, (gw * 16, gh * 16), interpolation=cv2.INTER_LINEAR); sy, sx = sh
        acc[sy:sy + gh * 16, sx:sx + gw * 16] += up; wt[sy:sy + gh * 16, sx:sx + gw * 16] += 1
    return acc / np.maximum(wt, 1e-6), fgs, s


def peaks(prob, fgs, thr, min_dist_px):
    inner = cv2.erode(fgs.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    k = max(3, int(round(min_dist_px)) | 1)
    pk = np.argwhere((prob == maximum_filter(prob, size=k)) & (prob > thr) & inner)
    return pk[np.argsort(-prob[pk[:, 0], pk[:, 1]])]


# ------------------------------------------------------------------ train
def train(clicks):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    layers = [int(x) for x in a.layers.split(',')]
    # 1) PCA per layer on foreground tokens of the training images (unshifted pass)
    pool = {L: [] for L in layers}; cache = {}
    for path in clicks:
        rgb_s, s = resized(load_rgb(path), a.size); fgs = fg_mask(rgb_s); cache[path] = (rgb_s, s, fgs)
        T = tokens(rgb_s, layers, (0, 0)); gh, gw = T[layers[0]].shape[:2]; fg = fg_grid(fgs, (0, 0), gh, gw)
        idx = np.flatnonzero(fg.ravel()); idx = rng.choice(idx, min(len(idx), 20000), replace=False)
        for L in layers: pool[L].append(T[L].reshape(gh * gw, -1)[idx])
    pcas = {L: PCA(a.pca, random_state=a.seed).fit(np.vstack(pool[L])) for L in layers}; del pool
    # 2) labelled patches from every shifted pass: click patch = label; automatic negatives far from positives
    X, Y, G, LX = [], [], [], []          # features, label, image index, left/right half (for the 1-image split)
    n_pos = n_neg = 0
    for gi, (path, (pos, neg)) in enumerate(clicks.items()):
        rgb_s, s, fgs = cache[path]; n_pos += len(pos); n_neg += len(neg)
        for sh in SHIFTS:
            T = tokens(rgb_s, layers, sh); gh, gw = T[layers[0]].shape[:2]; fg = fg_grid(fgs, sh, gh, gw); F = hyper(T, pcas, layers, fg, a.highpass)
            def cell(p):   # original px -> patch (row, col) of this pass
                y, x = p[1] * s - sh[0], p[0] * s - sh[1]; return int(y // 16), int(x // 16)
            lab = {}
            for p in neg:
                r, c = cell(p)
                if 0 <= r < gh and 0 <= c < gw: lab[(r, c)] = 0
            for p in pos:
                r, c = cell(p)
                if 0 <= r < gh and 0 <= c < gw: lab[(r, c)] = 1
            pc = np.array([cell(p) for p in pos], float).reshape(-1, 2)
            cand = np.argwhere(fg)
            if len(pc):
                dmin = np.min(np.linalg.norm(cand[:, None, :] - pc[None], axis=2), 1)
                cand = cand[(dmin >= a.neg_dist) & ((dmin <= a.neg_far) if a.neg_far > 0 else True)]
            for r, c in cand[rng.choice(len(cand), min(len(cand), a.auto_neg // len(SHIFTS)), replace=False)]:
                lab.setdefault((int(r), int(c)), 0)
            midc = np.median(pc[:, 1]) if len(pc) else gw / 2
            for (r, c), y in lab.items():
                X.append(F[r, c]); Y.append(y); G.append(gi); LX.append(int(c >= midc))
    X = np.array(X, np.float32); Y = np.array(Y); G = np.array(G); LX = np.array(LX)
    print(f'[probe] {len(clicks)} image(s), {n_pos} positive and {n_neg} negative clicks -> {Y.sum()} positive and {(Y == 0).sum()} negative patches (4 shifted passes)')
    if Y.sum() < 3: sys.exit('[probe] need at least a few positive clicks')
    D = a.pca; slices = {L: slice(i * D, (i + 1) * D) for i, L in enumerate(layers)}
    def fit(Xt, Yt):
        sc = StandardScaler().fit(Xt); return sc, LogisticRegression(C=a.C, class_weight='balanced', max_iter=4000).fit(sc.transform(Xt), Yt)
    def cv(cols):
        folds = ([(G != g, G == g) for g in np.unique(G)] if len(np.unique(G)) >= 2 else [(LX == 0, LX == 1), (LX == 1, LX == 0)])
        ys, ps = [], []
        for tr, te in folds:
            if len(np.unique(Y[tr])) < 2 or len(np.unique(Y[te])) < 2: continue
            sc, clf = fit(X[tr][:, cols], Y[tr]); ys.append(Y[te]); ps.append(clf.predict_proba(sc.transform(X[te][:, cols]))[:, 1])
        if not ys: return None
        y_, p_ = np.concatenate(ys), np.concatenate(ps)
        return dict(auc=round(float(roc_auc_score(y_, p_)), 3), ap=round(float(average_precision_score(y_, p_)), 3), ap_chance=round(float(y_.mean()), 3))
    allc = np.arange(X.shape[1]); res = {'hypercolumn ' + '+'.join(f'L{L}' for L in layers): cv(allc)}
    if a.ablate:
        for L in layers: res[f'L{L}'] = cv(np.arange(X.shape[1])[slices[L]])
        if 6 in layers and 23 in layers: res['L6+L23'] = cv(np.r_[np.arange(X.shape[1])[slices[6]], np.arange(X.shape[1])[slices[23]]])
    split = 'leave-one-image-out' if len(np.unique(G)) >= 2 else '2-fold left/right split of the clicked image'
    for k, v in res.items(): print(f'[probe] {split}: {k}: {v}')
    sc, clf = fit(X, Y)
    probe = dict(pcas=pcas, scaler=sc, clf=clf, layers=layers, size=a.size, name=a.name, model=a.model,
                 threshold=a.threshold, min_dist=a.min_dist, highpass=a.highpass, version='dinomask_hypercolumn_probe_v1')
    import joblib; joblib.dump(probe, out / 'probe.joblib')
    json.dump(dict(name=a.name, images=list(clicks), positive_clicks=n_pos, negative_clicks=n_neg,
                   patches=dict(positive=int(Y.sum()), negative=int((Y == 0).sum())), validation=split, results=res,
                   settings=dict(layers=layers, pca=a.pca, size=a.size, auto_neg=a.auto_neg, neg_dist=a.neg_dist, neg_far=a.neg_far, highpass=a.highpass, C=a.C,
                                 threshold=a.threshold, min_dist=a.min_dist, model=a.model)), open(out / 'results.json', 'w'), indent=1)
    return probe


# ------------------------------------------------------------------ parts (optional)
def part_masks(coco_path, file_name, shape):
    from pycocotools import mask as mu
    d = json.load(open(coco_path)); im = next((i for i in d['images'] if os.path.basename(i['file_name']) == os.path.basename(file_name)), None)
    if im is None: return {}
    names = {c['id']: c['name'] for c in d['categories']}; M = {}
    for an in d['annotations']:
        if an.get('image_id') != im['id'] or not an.get('segmentation'): continue
        seg = an['segmentation']
        if isinstance(seg, dict):
            s_ = dict(seg); s_['counts'] = s_['counts'].encode() if isinstance(s_['counts'], str) else s_['counts']; m = mu.decode(s_).astype(bool)
        else:
            m = np.zeros(shape, np.uint8)
            for p in (seg if isinstance(seg[0], list) else [seg]): cv2.fillPoly(m, [np.asarray(p, np.int32).reshape(-1, 2)], 1)
            m = m.astype(bool)
        nm = names.get(an['category_id'], ''); M[nm] = M.get(nm, np.zeros(shape, bool)) | m
    return M


# ------------------------------------------------------------------ predict
def predict(probe, paths):
    within = [x for x in a.within.split(',') if x]
    coco = dict(images=[], categories=[{'id': 1, 'name': 'keypoints', 'supercategory': 'keypoints'}], annotations=[])
    counts = {}
    for i, path in enumerate(paths):
        rgb = load_rgb(path); prob, fgs, s = prob_map(rgb, probe)
        pk = peaks(prob, fgs, probe['threshold'], probe['min_dist'] * 16)
        pts = [(float((c + 0.5) / s), float((r + 0.5) / s), float(prob[r, c])) for r, c in pk]
        parts = part_masks(a.within_coco, path, rgb.shape[:2]) if a.within_coco else {}
        rec = {'total': 0}
        keep = []
        for x, y, p in pts:
            hit = [nm for nm, m in parts.items() if 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1] and m[int(y), int(x)]]
            if within and not any(nm in within for nm in hit): continue
            keep.append((x, y, p)); rec['total'] += 1
            for nm in hit: rec[nm] = rec.get(nm, 0) + 1
        counts[os.path.basename(path)] = rec
        fb = os.path.basename(path); coco['images'].append({'id': fb, 'file_name': fb, 'height': rgb.shape[0], 'width': rgb.shape[1]})
        if keep:
            kp = [v for x, y, _ in keep for v in (round(x, 1), round(y, 1), 2)]; xs, ys = [k[0] for k in keep], [k[1] for k in keep]
            coco['annotations'].append({'id': i + 1, 'image_id': fb, 'category_id': 1, 'keypoints': kp, 'keypoints_clicked': kp,
                                        'num_keypoints': len(keep), 'point_order': list(range(1, len(keep) + 1)),
                                        'keypoint_status': ['present'] * len(keep), 'area': 1, 'iscrowd': 0,
                                        'bbox': [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                                        'probe_name': probe['name'], 'probe_scores': [round(p, 3) for _, _, p in keep]})
        ov = (rgb * 0.8).astype(np.uint8).copy(); r_ = max(4, int(round(6 / s)))
        for x, y, _ in keep: cv2.circle(ov, (int(x), int(y)), r_, (255, 225, 0), -1); cv2.circle(ov, (int(x), int(y)), r_, (90, 60, 0), max(1, r_ // 4))
        f = 1600 / max(ov.shape[:2]); Image.fromarray(cv2.resize(ov, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else ov).save(out / f'overlay_{Path(path).stem}.jpg', quality=88)
        print(f'[probe] {fb}: {rec["total"]} {probe["name"]}' + (f'  {dict((k, v) for k, v in rec.items() if k != "total")}' if len(rec) > 1 else ''))
    json.dump(coco, open(out / 'predictions_keypoints.json', 'w')); json.dump(counts, open(out / 'counts.json', 'w'), indent=1)
    print(f'[probe] -> {out / "predictions_keypoints.json"} (opens in the GUI as points), counts.json, overlay_*.jpg')


EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
if a.run:
    import joblib; probe = joblib.load(Path(a.run) / 'probe.joblib')
else:
    cl = read_clicks(click_files(a.points), a.images)
    if not cl: sys.exit('[probe] no clicks found: pass --points with the GUI keypoint save(s)')
    probe = train(cl)
targets = []
if a.predict_list: targets = [l.strip() for l in open(a.predict_list) if l.strip()]
elif a.predict_dir: targets = sorted(p for p in glob.glob(os.path.join(a.predict_dir, '*')) if p.lower().endswith(EXT) and 'mask' not in Path(p).stem.lower())
if targets: predict(probe, targets)
