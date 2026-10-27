#!/usr/bin/env python3
"""line_drawing_v1.py - sutures, carinae, striae or any other LINE structures as skeleton lines, plus a line drawing per specimen
(2.8.3, 25 Oct 2026; first used on Oxytelus head and pronotum sutures).
1. The masks of the --line_categories (e.g. suture) are skeletonized and measured by V35 (--method skeleton): one row per mask in
   line_metrics.csv with its length in px (and mm with --mm_scale).
2. Line drawing per image: outlines of the --outline_categories from the V42 contour semilandmarks (smooth, 100 points; the
   pipeline's semilandmarks/<category>/pre_alignment.TPS), the skeleton lines drawn on top. Where a wide mask gave two parallel
   skeleton branches, gaps up to --merge_px are closed and the drawing is skeletonized again (drawing only; lengths unchanged).
   Lines are rendered at --upsample x and lightly smoothed (Gaussian 0.5 px).
Outputs in --out_dir: line_metrics.csv, line_coco.json (the line masks), <image>_line_drawing.png/.pdf
Usage: python line_drawing_v1.py --coco parts.json --image_dir images --semilandmarks_dir out/semilandmarks --out_dir out/line_drawing \
           --line_categories suture --outline_categories head,pronotum [--mm_scale 172] [--merge_px 3] [--upsample 8] [--margin 8]"""
import argparse, ast, json, os, subprocess, sys
import numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

ap = argparse.ArgumentParser()
ap.add_argument('--coco', required=True); ap.add_argument('--image_dir', required=True); ap.add_argument('--out_dir', required=True)
ap.add_argument('--semilandmarks_dir', default='', help='pipeline semilandmarks folder (V42); without it the V35-free outline is the mask contour')
ap.add_argument('--line_categories', required=True, help='comma list of line structures, e.g. suture,carina')
ap.add_argument('--outline_categories', default='', help='comma list of parts drawn as outlines, e.g. head,pronotum')
ap.add_argument('--mm_scale', type=float, default=None, help='pixels per mm (passed to V35)')
ap.add_argument('--merge_px', type=float, default=3.0, help='merge parallel skeleton branches closer than this (image px); 0 = off')
ap.add_argument('--upsample', type=int, default=8); ap.add_argument('--margin', type=int, default=8)
ap.add_argument('--lw', type=float, default=0.8, help='outline line width (pt)'); ap.add_argument('--dpi', type=int, default=600)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
LINES = [c.strip() for c in a.line_categories.split(',') if c.strip()]
OUTL = [c.strip() for c in a.outline_categories.split(',') if c.strip()]

# 1. V35 skeleton measurements of the line masks
d = json.load(open(a.coco)); cid = {c['id']: c['name'] for c in d['categories']}
sub = dict(d); sub['annotations'] = [x for x in d['annotations'] if cid[x['category_id']] in LINES]
if not sub['annotations']: raise SystemExit(f'no annotations of {LINES} in {a.coco}')
lc = os.path.join(a.out_dir, 'line_coco.json'); json.dump(sub, open(lc, 'w'))
v35 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'measurement_script_to_try_after_kpts_prediction_measure_kpts_V35.py')
met = os.path.join(a.out_dir, 'line_metrics.csv')
cmd = [sys.executable, v35, '--json', lc, '--image_dir', a.image_dir, '--output_dir', os.path.join(a.out_dir, 'v35_skeleton'),
       '--method', 'skeleton', '--save_results', '--output_file', met] + (['--mm_scale', str(a.mm_scale)] if a.mm_scale else [])
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode or not os.path.exists(met): sys.exit('V35 skeleton step failed:\n' + r.stderr[-2000:])
M = pd.read_csv(met)
print(M[[c for c in ('image_filename', 'category_name', 'length_pixels', 'length_mm') if c in M.columns]].to_string(index=False))

# 2. line drawing per image
def tps_outline(cat, image):
    """the V42 pre-alignment contour of <cat> for <image> (TPS blocks: LM=n, n lines, IMAGE=<file>_<k>)"""
    f = os.path.join(a.semilandmarks_dir, cat, 'pre_alignment.TPS')
    if not a.semilandmarks_dir or not os.path.exists(f): return None
    L = open(f).read().splitlines(); i = 0
    while i < len(L):
        if L[i].startswith('LM='):
            n = int(L[i][3:]); pts = np.array([[float(v) for v in L[i + 1 + k].split()[:2]] for k in range(n)]); j = i + 1 + n
            img = next((L[k][6:] for k in range(j, min(j + 4, len(L))) if L[k].startswith('IMAGE=')), '')
            if os.path.basename(img).startswith(os.path.basename(image)): return pts
            i = j
        else: i += 1
    return None
def mask_outline(cat, im):
    m = np.zeros((im['height'], im['width']), np.uint8)
    for x in d['annotations']:
        if x['image_id'] == im['id'] and cid[x['category_id']] == cat and isinstance(x['segmentation'], list):
            for q in x['segmentation']: cv2.fillPoly(m, [np.array(q).reshape(-1, 2).round().astype(np.int32)], 1)
    cs = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    return max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float) if cs else None
done = set()
for im in d['images']:
    fn = os.path.basename(im['file_name'])
    if fn in done: continue
    done.add(fn); rows = M[M['image_filename'].astype(str).str.contains(os.path.splitext(fn)[0], regex=False)] if 'image_filename' in M else M
    if rows.empty: continue
    outl = {c: (tps_outline(c, fn) if tps_outline(c, fn) is not None else mask_outline(c, im)) for c in OUTL}
    pts = [np.array(ast.literal_eval(s), float) for s in rows['segmentation_coords']]
    allp = np.vstack([p for p in list(outl.values()) + pts if p is not None and len(p)])
    X0, Y0 = np.floor(allp.min(0)) - a.margin; X1, Y1 = np.ceil(allp.max(0)) + a.margin; U = a.upsample
    canvas = np.zeros((int(Y1 - Y0) * U, int(X1 - X0) * U), np.float32)
    for p in pts:   # every 8-neighbour link of the skeleton
        dd = np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1)); i_, j_ = np.where(np.triu(dd <= 1.5, 1))
        q = ((p - [X0, Y0]) * U + U / 2).round().astype(int)
        for s_, t_ in zip(i_, j_): cv2.line(canvas, tuple(int(v) for v in q[s_]), tuple(int(v) for v in q[t_]), 1.0, 3, cv2.LINE_AA)
    if a.merge_px > 0:
        bw = cv2.morphologyEx((canvas > 0.3).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * a.merge_px * U) + 1,) * 2))
        canvas = cv2.dilate(skeletonize(bw > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
    canvas = np.clip(cv2.GaussianBlur(canvas, (0, 0), 0.5 * U) * 2.2, 0, 1)
    w, h = X1 - X0, Y1 - Y0; fig, ax = plt.subplots(figsize=(3.0 * w / max(w, h), 3.0 * h / max(w, h)))
    for c, o in outl.items():
        if o is not None: ax.plot(*np.vstack([o, o[:1]]).T, '-', color='k', lw=a.lw)
    ax.imshow(np.dstack([np.zeros_like(canvas)] * 3 + [canvas]), extent=(X0, X1, Y1, Y0), interpolation='bilinear', zorder=3)
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_aspect('equal'); ax.axis('off')
    stem = os.path.join(a.out_dir, os.path.splitext(fn)[0] + '_line_drawing')
    fig.savefig(stem + '.png', dpi=a.dpi, bbox_inches='tight', facecolor='white'); fig.savefig(stem + '.pdf', bbox_inches='tight'); plt.close(fig)
    print('wrote', stem + '.png')
