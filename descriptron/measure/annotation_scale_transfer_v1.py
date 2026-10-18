#!/usr/bin/env python3
"""Annotate huge images on smaller copies, then put the annotations back on the originals (17 Oct 2026; general, any taxon).

  shrink    --images DIR [--list FILE] --out DIR [--max_side 2700 | --scale 0.5] [--quality 95]
            writes a JPG copy of every image (aspect kept) + scale_map.tsv (small file -> original path and both sizes)
  to_small  --coco FULL.json --scale_map scale_map.tsv --out SMALL.json
            existing full-resolution annotations (e.g. predictions to correct) -> coordinates of the small copies
  to_full   --coco SMALL.json(.rlejson) --scale_map scale_map.tsv --out FULL.json
            annotations made on the small copies -> original image names, sizes and coordinates; writes FULL.json (polygons)
            AND FULL.rlejson (RLE, keeps holes), as the GUI does

Masks are scaled as CONTOURS (outer + holes, sub-pixel scaled, refilled at the target size), not by nearest-neighbour upsampling,
so edges stay smooth. Keypoints, keypoints_clicked, bbox, area, line_points and line_thickness are scaled; every other field is copied.
A mask made at scale s can only be as precise as 1/s original pixels: at 0.5 that is about 2 px on the original."""
import argparse, csv, json, os, sys, numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
EXT = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')


def shrink(a):
    os.makedirs(a.out, exist_ok=True)
    files = [l.strip() for l in open(a.list) if l.strip()] if a.list else sorted(f for f in os.listdir(a.images) if f.lower().endswith(EXT) and not f.startswith('._') and '_mask' not in f)
    rows = []
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(a.images, f); im = Image.open(p); W, H = im.size
        s = a.scale if a.scale else min(1.0, a.max_side / max(W, H)); w, h = max(1, round(W * s)), max(1, round(H * s))
        small = os.path.splitext(os.path.basename(p))[0] + '.jpg'
        im.convert('RGB').resize((w, h), Image.LANCZOS).save(os.path.join(a.out, small), quality=a.quality)
        rows.append((small, os.path.abspath(p), W, H, w, h)); print(f'{small}: {W}x{H} -> {w}x{h}', flush=True)
    with open(os.path.join(a.out, 'scale_map.tsv'), 'w') as fh:
        fh.write('small_file\toriginal_path\torig_width\torig_height\tsmall_width\tsmall_height\n')
        for r in rows: fh.write('\t'.join(map(str, r)) + '\n')
    print(len(rows), 'images ->', a.out)


def load_map(p):
    M = {}
    for r in csv.DictReader(open(p), delimiter='\t'):
        d = dict(small=r['small_file'], orig=os.path.basename(r['original_path']), W=int(r['orig_width']), H=int(r['orig_height']), w=int(r['small_width']), h=int(r['small_height']))
        for k in (d['small'], d['orig'], os.path.splitext(d['small'])[0], os.path.splitext(d['orig'])[0]): M[k] = d
    return M


def seg_to_mask(seg, h, w):
    from pycocotools import mask as mu
    if isinstance(seg, dict):
        r = dict(seg)
        if isinstance(r['counts'], list): r = mu.frPyObjects(r, h, w)
        elif isinstance(r['counts'], str): r['counts'] = r['counts'].encode('ascii')
        return mu.decode(r).astype(np.uint8)
    m = np.zeros((h, w), np.uint8)
    polys = [np.asarray(p, np.float64).reshape(-1, 2) for p in seg if len(p) >= 6]
    if polys: cv2.fillPoly(m, [np.round(p).astype(np.int32) for p in polys], 1)
    return m


def scale_mask(m, sx, sy, W, H):
    """contours (with holes) scaled about pixel centres, refilled at W x H with 8x sub-pixel precision"""
    cs, hier = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE); out = np.zeros((H, W), np.uint8)
    if not cs: return out
    SH = 3; f = 1 << SH
    def sc(c): q = c.reshape(-1, 2).astype(np.float64); q[:, 0] = (q[:, 0] + 0.5) * sx - 0.5; q[:, 1] = (q[:, 1] + 0.5) * sy - 0.5; return np.round(q * f).astype(np.int32)
    for i, c in enumerate(cs):
        if hier[0][i][3] < 0: cv2.fillPoly(out, [sc(c)], 1, lineType=cv2.LINE_8, shift=SH)
    for i, c in enumerate(cs):
        if hier[0][i][3] >= 0: cv2.fillPoly(out, [sc(c)], 0, lineType=cv2.LINE_8, shift=SH)
    return out


def rle(m):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(m.astype(np.uint8))); r['counts'] = r['counts'].decode('ascii'); return r


def polys(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1).astype(float).tolist() for c in cs if len(c) >= 3]


def scale_kp(v, sx, sy):
    if isinstance(v, str):
        try: v = json.loads(v)
        except Exception: return v
    if not isinstance(v, list): return v
    if v and isinstance(v[0], (list, tuple)): return [[p[0] * sx, p[1] * sy] + list(p[2:]) for p in v]
    out = list(v)
    for i in range(0, len(out) - 2, 3): out[i] = out[i] * sx; out[i + 1] = out[i + 1] * sy
    return out


def convert(a, direction):
    M = load_map(a.scale_map); d = json.load(open(a.coco)); img_new = {}; key = {}
    for im in d['images']:
        m = M.get(os.path.basename(im['file_name'])) or M.get(os.path.splitext(os.path.basename(im['file_name']))[0])
        if m is None: sys.exit(f"image {im['file_name']} not in {a.scale_map}")
        new = dict(im)
        if direction == 'to_full': new.update(file_name=m['orig'], width=m['W'], height=m['H']); sx, sy, tw, th = m['W'] / m['w'], m['H'] / m['h'], m['W'], m['H']; sh, sw = im.get('height', m['h']), im.get('width', m['w'])
        else: new.update(file_name=m['small'], width=m['w'], height=m['h']); sx, sy, tw, th = m['w'] / m['W'], m['h'] / m['H'], m['w'], m['h']; sh, sw = im.get('height', m['H']), im.get('width', m['W'])
        if isinstance(im['id'], str) and os.path.splitext(os.path.basename(im['file_name']))[0] in im['id']:
            new['id'] = im['id'].replace(os.path.splitext(os.path.basename(im['file_name']))[0], os.path.splitext(new['file_name'])[0])
        key[im['id']] = (new['id'], sx, sy, tw, th, sh, sw); img_new[im['id']] = new
    P = {k: v for k, v in d.items() if k not in ('images', 'annotations')}; R = dict(P)
    P['images'] = R['images'] = list(img_new.values()); P['annotations'] = []; R['annotations'] = []
    for an in d['annotations']:
        iid, sx, sy, tw, th, sh, sw = key[an['image_id']]; b = dict(an, image_id=iid)
        for k in ('keypoints', 'keypoints_clicked', 'line_points'):
            if k in b: b[k] = scale_kp(b[k], sx, sy)
        if 'line_thickness' in b and isinstance(b['line_thickness'], (int, float)): b['line_thickness'] = max(1, round(b['line_thickness'] * (sx + sy) / 2))
        if isinstance(b.get('bbox'), list) and len(b['bbox']) == 4: x, y, w, h = b['bbox']; b['bbox'] = [x * sx, y * sy, w * sx, h * sy]
        seg = an.get('segmentation')
        if seg:
            m = scale_mask(seg_to_mask(seg, sh, sw), sx, sy, tw, th)
            b['area'] = float(m.sum());
            if m.any(): b['bbox'] = [float(v) for v in cv2.boundingRect(m)]
            P['annotations'].append(dict(b, segmentation=polys(m))); R['annotations'].append(dict(b, segmentation=rle(m)))
        else:
            if isinstance(b.get('area'), (int, float)) and b['area'] > 1: b['area'] = b['area'] * sx * sy
            P['annotations'].append(b); R['annotations'].append(b)
    base = os.path.splitext(a.out)[0]
    json.dump(P, open(base + '.json', 'w')); json.dump(R, open(base + '.rlejson', 'w'))
    print(f"{len(d['images'])} images, {len(d['annotations'])} annotations -> {base}.json (polygons) + {base}.rlejson (RLE)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('shrink'); s.add_argument('--images', required=True); s.add_argument('--list', default=''); s.add_argument('--out', required=True)
    s.add_argument('--max_side', type=int, default=2700); s.add_argument('--scale', type=float, default=0); s.add_argument('--quality', type=int, default=95)
    for c in ('to_small', 'to_full'):
        s = sp.add_parser(c); s.add_argument('--coco', required=True); s.add_argument('--scale_map', required=True); s.add_argument('--out', required=True)
    a = ap.parse_args()
    shrink(a) if a.cmd == 'shrink' else convert(a, a.cmd)
