#!/usr/bin/env python3
"""
biorag_probe_panels_v2.py - BioRAG-Probe standard figure: FOUR PANELS per specimen (16 Oct 2026)
v2 (AVD: v1 PCA panels too pixelated): --shifts 2 = four passes shifted by half a patch -> 8-px token grid, smooth (bilinear)
upsampling; zoom windows run at --zoom_upscale 2 (8 native px per patch) with their OWN shared PCA bases.
==============================================================================================
  1 photo | 2 DINOv3 layer-L PCA (3 components as RGB) | 3 high-pass layer-L PCA | 4 segmentation overlay
Panel 4 draws every COCO category found for the image (outlines, one colour per category, legend), plus optional points
(e.g. setae / dark dots from a probe) from a TSV. A second figure per specimen ('ZOOM') repeats the four panels on a
native-resolution window (default: around one named region) enlarged x2, so single setae / microtrichia can be judged.
The PCA bases are fitted ONCE on foreground tokens pooled over all specimens in the run, so a colour means the same thing on
every specimen; components are scaled by the pooled 1st-99th percentiles; background is WHITE (no black maps).
High-pass = token minus the mean of its foreground neighbours in a 7x7-patch window (local texture, not position on the body).
Foreground: union of all drawn masks + the specimen against a plain background (border-colour difference), so the PCA is not
fitted on the background.
Usage:
  biorag_probe_panels_v1.py --images DIR --files a.jpg b.jpg ... --coco hand.json pred.json spots.json --out DIR
        [--layer 18] [--zoom_region bm-m] [--zoom_px 700] [--points pts.tsv (file, x, y)] [--title 'Mycetophila wings']
"""
import argparse, json, os, csv, time, numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--images', required=True); ap.add_argument('--files', nargs='+', required=True); ap.add_argument('--coco', nargs='*', default=[])
ap.add_argument('--out', required=True); ap.add_argument('--layer', type=int, default=18); ap.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m')
ap.add_argument('--zoom_region', default=''); ap.add_argument('--zoom_px', type=int, default=700); ap.add_argument('--points', default='')
ap.add_argument('--title', default=''); ap.add_argument('--max_side', type=int, default=0, help='0 = native resolution (default)'); ap.add_argument('--tile', type=int, default=1024); ap.add_argument('--shifts', type=int, default=2, help='passes per axis shifted by 16/shifts px (2 = 8-px grid)'); ap.add_argument('--zoom_upscale', type=float, default=2.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
# ---------- annotations: per file, category -> polygons (first COCO file with that category for that image wins) ----------
ANN = {}; CATS = []
for cf in a.coco:
    d = json.load(open(cf)); cn = {c['id']: c['name'] for c in d['categories']}; fn = {i['id']: os.path.basename(i['file_name']) for i in d['images']}; got = {}
    for an in d['annotations']:
        if isinstance(an.get('segmentation'), list) and an['segmentation'] and fn.get(an['image_id']) in a.files:
            got.setdefault(fn[an['image_id']], {}).setdefault(cn[an['category_id']], []).extend(an['segmentation'])
    for f, cats in got.items():
        for c, s in cats.items():
            if c not in ANN.setdefault(f, {}): ANN[f][c] = s
            if c not in CATS: CATS.append(c)
PAL = {c: tuple(int(v) for v in cv2.cvtColor(np.uint8([[[int(179 * i / max(1, len(CATS))), 230, 220]]]), cv2.COLOR_HSV2RGB)[0, 0]) for i, c in enumerate(CATS)}
if 'spot_pattern' in PAL: PAL['spot_pattern'] = (20, 20, 20)
PTS = {}
if a.points:
    for r in csv.DictReader(open(a.points), delimiter='\t'): PTS.setdefault(r['file'], []).append((float(r['x']), float(r['y'])))
# ---------- DINOv3 ----------
from transformers import AutoModel
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net = AutoModel.from_pretrained(a.model).to(dev).eval(); DIM = net.config.hidden_size; NREG = getattr(net.config, 'num_register_tokens', 4)
def grid(x):
    H, W = x.shape[:2]; gh, gw = H // 16, W // 16; OV = 128; STEP = a.tile - 2 * OV
    def one(t):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            hs = net(pixel_values=torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1)))[None].to(dev), output_hidden_states=True).hidden_states
        return hs[a.layer + 1][0, 1 + NREG:, :].float().reshape(t.shape[0] // 16, t.shape[1] // 16, DIM)
    if H <= a.tile and W <= a.tile: return one(x)
    out = torch.zeros((gh, gw, DIM), device=dev)
    for y0 in range(0, H, STEP):
        for x0 in range(0, W, STEP):
            ya, xa = max(0, y0 - OV), max(0, x0 - OV); t = x[ya:min(H, y0 + STEP + OV), xa:min(W, x0 + STEP + OV)]; t = t[:t.shape[0] // 16 * 16, :t.shape[1] // 16 * 16]
            T = one(t); ky, kx = (y0 - ya) // 16, (x0 - xa) // 16; ny, nx = min(STEP, H - y0) // 16, min(STEP, W - x0) // 16; out[y0 // 16:y0 // 16 + ny, x0 // 16:x0 // 16 + nx] = T[ky:ky + ny, kx:kx + nx]
    return out
def highpass(G, w):
    Gw = (G * w[..., None]).permute(2, 0, 1)[None]; num = F.avg_pool2d(Gw, 7, 1, 3, count_include_pad=False)[0].permute(1, 2, 0)
    den = F.avg_pool2d(w[None, None], 7, 1, 3, count_include_pad=False)[0, 0]; return G - num / den.clamp_min(1e-3)[..., None]
def foreground(I, f):
    g = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY); s = 800 / max(g.shape); gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bg = np.median(np.concatenate([gs[:5].ravel(), gs[-5:].ravel(), gs[:, :5].ravel(), gs[:, -5:].ravel()]))
    fg = (np.abs(gs.astype(int) - bg) > 12).astype(np.uint8); fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(fg); fg = (lb == 1 + st[1:, 4].argmax()).astype(np.uint8) if n > 1 else fg
    cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); fg = np.zeros_like(fg); cv2.drawContours(fg, cs, -1, 1, -1)
    fg = cv2.resize(fg, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)
    for c, segs in ANN.get(f, {}).items():
        for sg in segs: cv2.fillPoly(fg, [np.array(sg, np.float32).reshape(-1, 2).astype(np.int32)], 1)
    return fg
def load(f, window=None):
    I = np.asarray(Image.open(os.path.join(a.images, f)).convert('RGB')); fg = foreground(I, f)
    if window:
        x0, y0, x1, y1 = window; I, fg = I[y0:y1, x0:x1], fg[y0:y1, x0:x1]
        if a.zoom_upscale != 1: I = cv2.resize(I, None, fx=a.zoom_upscale, fy=a.zoom_upscale, interpolation=cv2.INTER_CUBIC); fg = cv2.resize(fg, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)
    s = 1.0 if not a.max_side else min(1.0, a.max_side / max(I.shape[:2]))
    if s < 1: I = cv2.resize(I, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); fg = cv2.resize(fg, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)
    h, w = I.shape[0] // 16 * 16, I.shape[1] // 16 * 16; return I[:h, :w], fg[:h, :w], s
def tokens(I, fg):
    """v2: --shifts^2 passes, image shifted by st = 16/shifts px, interleaved into one st-px grid (raw + high-pass per pass)."""
    S = max(1, a.shifts); st = 16 // S; H, W = I.shape[:2]; fh, fw = H // st, W // st; R = Hp = None
    for sy in range(S):
        for sx in range(S):
            J = I[sy * st:, sx * st:]; J = J[:J.shape[0] // 16 * 16, :J.shape[1] // 16 * 16]; G = grid((J.astype(np.float32) / 255 - MEAN) / STD); gh, gw = G.shape[:2]
            w = cv2.resize(fg[sy * st:sy * st + gh * 16, sx * st:sx * st + gw * 16].astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) >= 0.5
            hp = highpass(G, torch.from_numpy(w.astype(np.float32)).to(dev)).cpu().numpy().astype(np.float16); G = G.cpu().numpy().astype(np.float16)
            if R is None: R = np.zeros((fh, fw, G.shape[2]), np.float16); Hp = np.zeros_like(R)
            ys, xs = slice(sy, sy + S * gh, S), slice(sx, sx + S * gw, S); R[ys, xs] = G[:len(range(fh)[ys]), :len(range(fw)[xs])]; Hp[ys, xs] = hp[:len(range(fh)[ys]), :len(range(fw)[xs])]
    w = cv2.resize(fg.astype(np.float32), (fw, fh), interpolation=cv2.INTER_AREA) >= 0.5; return R, Hp, w
def zoom_window(f):
    I = Image.open(os.path.join(a.images, f)); W, H = I.size; segs = ANN.get(f, {}).get(a.zoom_region)
    if segs: pts = np.vstack([np.array(s).reshape(-1, 2) for s in segs]); cx, cy = pts.mean(0)
    else: cx, cy = W / 2, H / 2
    h = a.zoom_px // 2; x0, y0 = int(np.clip(cx - 1.5 * h, 0, max(0, W - 3 * h))), int(np.clip(cy - h, 0, max(0, H - 2 * h))); return (x0, y0, min(W, x0 + 3 * h), min(H, y0 + 2 * h))
# ---------- pass 1: tokens for whole specimens and zoom windows ----------
t0 = time.time(); DATA = {}
for f in a.files:
    for kind, win in (('whole', None), ('zoom', zoom_window(f))):
        I, fg, s = load(f, win); R, Hp, w = tokens(I, fg); DATA[(f, kind)] = (I, fg, s, R, Hp, w, win)
    print(f'  tokens {f}  {time.time() - t0:.0f}s', flush=True)
def fit(key, kind='whole'):
    X = np.vstack([d[3 if key == 'raw' else 4][d[5]] for k, d in DATA.items() if k[1] == kind]).astype(np.float32)
    X = X[rng.choice(len(X), min(len(X), 60000), replace=False)]; p = PCA(3, random_state=0).fit(X); Z = p.transform(X); return p, np.percentile(Z, 1, 0), np.percentile(Z, 99, 0)
BASES = {(k, kind): fit(k, kind) for k in ('raw', 'hp') for kind in ('whole', 'zoom')}
def pca_img(T, w, key, shape, kind, fgpx):
    """v2: PCA RGB on the fine grid, bilinear upsampling (cell centres aligned), background white at PIXEL level."""
    p, lo, hi = BASES[(key, kind)]; gh, gw = w.shape; Z = p.transform(T.reshape(-1, T.shape[-1]).astype(np.float32)).reshape(gh, gw, 3)
    rgb = np.clip((Z - lo) / np.maximum(hi - lo, 1e-6), 0, 1).astype(np.float32); st = 16 // max(1, a.shifts)
    up = cv2.resize(rgb, (gw * st, gh * st), interpolation=cv2.INTER_LINEAR); M = np.float32([[1, 0, (16 - st) / 2], [0, 1, (16 - st) / 2]])
    up = cv2.warpAffine(up, M, (shape[1], shape[0]), borderMode=cv2.BORDER_REPLICATE); out = np.uint8(255 * up); out[fgpx[:shape[0], :shape[1]] == 0] = 255; return out
def seg_img(f, I, s, win):
    o = I.copy(); off = np.array([win[0], win[1]]) if win else np.zeros(2); lw = max(2, int(round(I.shape[1] / 700)))
    for c, segs in ANN.get(f, {}).items():
        for sg in segs:
            q = ((np.array(sg, np.float32).reshape(-1, 2) - off) * s * (a.zoom_upscale if win else 1)).astype(np.int32)
            if c == 'spot_pattern':
                ov = o.copy(); cv2.fillPoly(ov, [q], (60, 60, 60)); o = cv2.addWeighted(ov, 0.35, o, 0.65, 0)
            cv2.polylines(o, [q], True, PAL[c], lw)
    for (x, y) in PTS.get(f, []):
        x, y = (x - off[0]) * s * (a.zoom_upscale if win else 1), (y - off[1]) * s * (a.zoom_upscale if win else 1)
        if 0 <= x < I.shape[1] and 0 <= y < I.shape[0]: cv2.circle(o, (int(x), int(y)), max(3, lw * 2), (0, 200, 0), max(1, lw // 2))
    return o
def label(img, text):
    b = np.full((34, img.shape[1], 3), 255, np.uint8); cv2.putText(b, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2); return np.vstack([b, img])
def legend(cats, width):
    b = np.full((30, width, 3), 255, np.uint8); x = 6
    for c in cats:
        cv2.rectangle(b, (x, 8), (x + 16, 22), PAL[c][::-1] if False else PAL[c], -1); cv2.putText(b, c, (x + 20, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1); x += 30 + 9 * len(c)
        if x > width - 80: break
    return b
for (f, kind), (I, fg, s, R, Hp, w, win) in DATA.items():
    sh = I.shape[:2]; Z = 1   # v2: zoom windows are already upscaled before DINOv3
    P = [label(I, '1 photo' + (f' (native window, x{a.zoom_upscale:g})' if kind == 'zoom' else ' (native)')), label(pca_img(R, w, 'raw', sh, kind, fg), f'2 layer {a.layer} PCA (RGB = PC1-3, shared bases, {16 // max(1, a.shifts)}-px grid)'),
         label(pca_img(Hp, w, 'hp', sh, kind, fg), f'3 high-pass layer {a.layer} PCA'), label(seg_img(f, I, s, win), '4 segmentation' + (' + probe points' if PTS.get(f) else ''))]
    if Z != 1: P = [cv2.resize(p, None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC) for p in P]
    top = np.hstack([P[0], np.full((P[0].shape[0], 12, 3), 255, np.uint8), P[1]]); bot = np.hstack([P[2], np.full((P[2].shape[0], 12, 3), 255, np.uint8), P[3]])
    S = np.vstack([top, np.full((12, top.shape[1], 3), 255, np.uint8), bot]); cats = sorted(ANN.get(f, {}))
    hdr = np.full((40, S.shape[1], 3), 255, np.uint8); cv2.putText(hdr, f'{a.title}  {f}  ({kind})', (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    S = np.vstack([hdr, S, legend(cats, S.shape[1])]); mx = 3600
    if S.shape[1] > mx: S = cv2.resize(S, (mx, int(S.shape[0] * mx / S.shape[1])), interpolation=cv2.INTER_AREA)
    Image.fromarray(S).save(os.path.join(a.out, f'{os.path.splitext(f)[0]}_4PANEL_{kind}.jpg'), quality=90)
print(f'wrote {len(DATA)} four-panel figures to {a.out}  ({time.time() - t0:.0f}s)')
