#!/usr/bin/env python3
"""DINOMask prediction with test-time rot4 (Descriptron packaged copy: falls back to the CPU when there is no GPU) (17 Oct 2026; same probs()/guided filter as rotation_test_v1.py). For each image: predict at
0/90/180/270 deg, rotate the probabilities back, keep the rotation with the highest mean max-probability over non-background pixels
(catches figures with the head on the right / sideways). Writes predictions.rlejson (RLE, keeps holes) + predictions_polygons.json for
--classes (default: all non-background), picked_rotation.tsv and overlays.
Usage: dinomask_predict_rot4_v1.py --run RUN_DIR (head.pt + results.json) --images DIR [--list FILE] --out DIR [--classes a,b] [--size 2048]"""
import argparse, json, os, numpy as np, torch, torch.nn as nn, cv2
from PIL import Image
from transformers import AutoModel
ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--images', required=True); ap.add_argument('--list', default='')
ap.add_argument('--out', required=True); ap.add_argument('--classes', default=''); ap.add_argument('--size', type=int, default=2048)
ap.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-lvd1689m'); ap.add_argument('--min_frac', type=float, default=0.002); a = ap.parse_args()
os.makedirs(os.path.join(a.out, 'overlays'), exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
classes = json.load(open(os.path.join(a.run, 'results.json')))['classes']; K = len(classes)
want = [c for c in (a.classes.split(',') if a.classes else classes[1:]) if c in classes]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'   # packaged copy: runs without a GPU too (slowly)
net = AutoModel.from_pretrained(a.model).to(dev).eval(); NREG = net.config.num_register_tokens; D = net.config.hidden_size
head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, K)).to(dev)
sd = torch.load(os.path.join(a.run, 'head.pt'), map_location=dev); head.load_state_dict({k.replace('net.', ''): v for k, v in sd.items()}); head.eval()
def gfilter(I, p, r=8, eps=1e-3):
    mean = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1)); mI = mean(I); vI = mean(I * I) - mI * mI; out = np.empty_like(p)
    for k in range(p.shape[2]): q = p[..., k]; mp = mean(q); aa = (mean(I * q) - mI * mp) / (vI + eps); b = mp - aa * mI; out[..., k] = mean(aa) * I + mean(b)
    return out
def probs(rgb):
    H, W = rgb.shape[:2]; s = a.size / max(H, W); nh, nw = int(H * s) // 16 * 16, int(W * s) // 16 * 16; rgb_ = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    base = (rgb_.astype(np.float32) / 255 - MEAN) / STD; acc = np.zeros((nh, nw, K), np.float32)
    for dx, dy in ((0, 0), (8, 0), (0, 8), (8, 8)):
        x = np.pad(base, ((dy, 16 - dy if dy else 0), (dx, 16 - dx if dx else 0), (0, 0)), mode='edge'); gh, gw = x.shape[0] // 16, x.shape[1] // 16
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16): T = net(pixel_values=torch.from_numpy(x[:gh * 16, :gw * 16].transpose(2, 0, 1).copy())[None].to(dev)).last_hidden_state[0, 1 + NREG:].float()
        with torch.no_grad(): pr = torch.softmax(head(T), -1).cpu().numpy().reshape(gh, gw, K)
        acc += cv2.resize(pr, (gw * 16, gh * 16), interpolation=cv2.INTER_LINEAR)[dy:dy + nh, dx:dx + nw]
    P = gfilter(cv2.cvtColor(rgb_, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255, acc / 4); return cv2.resize(P, (W, H), interpolation=cv2.INTER_LINEAR)
def rle(m):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(m.astype(np.uint8))); r['counts'] = r['counts'].decode('ascii'); return r
files = [l.strip() for l in open(a.list) if l.strip()] if a.list else sorted(f for f in os.listdir(a.images) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')))
cats = [{'id': i + 1, 'name': c} for i, c in enumerate(want)]; R = {'images': [], 'annotations': [], 'categories': cats}; Pj = json.loads(json.dumps(R)); rot = []
cols = [tuple(int(v) for v in cv2.applyColorMap(np.uint8([[int(255 * i / max(K - 1, 1))]]), cv2.COLORMAP_TURBO)[0, 0][::-1]) for i in range(K)]
for n, fn in enumerate(files):
    rgb = np.asarray(Image.open(os.path.join(a.images, os.path.basename(fn))).convert('RGB')); H, W = rgb.shape[:2]; best = None
    for k in range(4):
        P = np.rot90(probs(np.ascontiguousarray(np.rot90(rgb, k))), -k); L = P.argmax(-1); fg = L > 0
        conf = float(P.max(-1)[fg].mean()) if fg.any() else 0.0
        if best is None or conf > best[0]: best = (conf, L, k)
    conf, L, k = best; rot.append((os.path.basename(fn), k * 90, conf)); iid = n + 1
    R['images'].append({'id': iid, 'file_name': os.path.basename(fn), 'width': W, 'height': H}); Pj['images'].append(dict(R['images'][-1]))
    O = rgb.copy()
    for c in cats:
        m = (L == classes.index(c['name'])).astype(np.uint8)
        if m.sum() < a.min_frac * H * W: continue
        R['annotations'].append({'id': len(R['annotations']) + 1, 'image_id': iid, 'category_id': c['id'], 'segmentation': rle(m), 'area': float(m.sum()), 'bbox': list(cv2.boundingRect(m)), 'iscrowd': 0})
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Pj['annotations'].append({'id': len(Pj['annotations']) + 1, 'image_id': iid, 'category_id': c['id'], 'segmentation': [q.reshape(-1).astype(float).tolist() for q in cs if len(q) >= 3], 'area': float(m.sum()), 'bbox': list(cv2.boundingRect(m)), 'iscrowd': 0})
        cv2.drawContours(O, cs, -1, cols[classes.index(c['name'])], max(2, W // 400))
        if len(cs): x_, y_, _, _ = cv2.boundingRect(max(cs, key=cv2.contourArea)); cv2.putText(O, c['name'], (x_, y_ + 20), 0, max(0.5, W / 1500), cols[classes.index(c['name'])], 2)
    s = min(1.0, 900 / max(H, W)); Image.fromarray(cv2.resize(O, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)).save(os.path.join(a.out, 'overlays', os.path.splitext(os.path.basename(fn))[0] + '.jpg'), quality=85)
    print(f'{n + 1}/{len(files)} {fn} rot {k * 90} conf {conf:.2f}', flush=True)
json.dump(R, open(os.path.join(a.out, 'predictions.rlejson'), 'w')); json.dump(Pj, open(os.path.join(a.out, 'predictions_polygons.json'), 'w'))
with open(os.path.join(a.out, 'picked_rotation.tsv'), 'w') as f: f.write('image\trotation_deg\tconfidence\n'); [f.write(f'{x}\t{y}\t{z:.3f}\n') for x, y, z in rot]
print('done', len(R['annotations']), 'masks')
