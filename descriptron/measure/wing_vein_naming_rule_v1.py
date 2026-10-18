"""Name wing veins by a topological rule, no training (17 Oct 2026, AVD: 'work from the costal vein ... proximal to distal, as in
entomology texts; lost veins the user re-labels').
Input: a binary vein mask per wing (hand outlines merged, or a DINOMask probs dir, or any COCO of predicted veins). Wing base on the LEFT.
Rule v1 (a taxon PROFILE gives the names; default = Mycetophila as annotated here):
  1. membrane outline at native resolution; split at base (leftmost) and apex (rightmost) into anterior and posterior chains.
  2. COSTA = vein pixels within --costa_band px of the anterior chain.
  3. skeleton of the remaining veins; TERMINALS = skeleton end points within --term_dist px of the outline;
     each terminal branch = skeleton path from the terminal back to the first junction.
  4. terminals reaching the ANTERIOR chain, ordered from the apex backwards, get --anterior names (default R5, r1, subcosta);
     terminals reaching the POSTERIOR chain, ordered from the apex towards the base, get --posterior names (default M1, M2, M4, CuA, A1).
  Stems and cross-veins (r, Rs, R-M, bm-m, M1-2, Cu, humeral) are NOT named by v1: the user names them (or a later rule by junctions).
Scoring against the hand-named outlines: every named branch -> the hand vein it overlaps most (pixels within 3 px); per name: correct /
wrong / missing. Writes per_wing.tsv, summary.txt and sheets (branches coloured by assigned name, label at the terminal).
Usage: script --coco HAND.json --images DIR --out DIR [--probs_dir DIR (predicted veins: argmax>0)] [--image_list_dir DIR]"""
import argparse, json, os
import numpy as np, cv2
from PIL import Image
from skimage.morphology import skeletonize
ap = argparse.ArgumentParser(); ap.add_argument('--coco', required=True); ap.add_argument('--images', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--probs_dir', default=''); ap.add_argument('--image_list_dir', default=''); ap.add_argument('--costa_band', type=int, default=30)
ap.add_argument('--term_dist', type=int, default=60); ap.add_argument('--anterior', default='R5,r1'); ap.add_argument('--posterior', default='M1,M2,M4,CuA'); ap.add_argument('--free_anterior', default='subcosta', help='v1e: vein ending FREE in the membrane, nearest the costa, basal half (Sc in Mycetophila)'); ap.add_argument('--free_posterior', default='A1', help='v1e: vein ending free nearest the posterior margin, basal half'); ap.add_argument('--humeral_frac', type=float, default=0.15, help='v1e: costa-touching terminals in this basal fraction are cross-veins (humeral), not named')
ap.add_argument('--min_branch', type=int, default=40); ap.add_argument('--prune', type=int, default=30, help='v1c: skeleton spurs shorter than this are removed first'); ap.add_argument('--profiles', default='', help='v1l: TSV genus<TAB>anterior<TAB>posterior<TAB>free_anterior<TAB>free_posterior (comma lists, "-" = none); genus from --genus_regex on the file name; unknown genera use the command-line names'); ap.add_argument('--genus_regex', default=r'^([A-Z][a-z]+)'); ap.add_argument('--texture_outline', type=int, default=0, help='v1j: add local texture to the outline (pale membranes)'); ap.add_argument('--term_dist_distal', type=int, default=0, help='v1j: posterior tips in the distal 55%% may stop this far from the margin'); ap.add_argument('--skeleton_costa', type=int, default=40, help='v1i: >0 = costa = skeleton pixels within this many px of the anterior outline; cut only those'); ap.add_argument('--costa_piece', type=int, default=5, help='v1h: margin over the costa band for dropping costa pieces (was 15)'); ap.add_argument('--costa_touch', type=int, default=15, help='v1b: terminal within this many px of the costa = anterior vein'); ap.add_argument('--sheet', type=int, default=6); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); ANT, POST = a.anterior.split(','), a.posterior.split(','); ANT0, POST0 = list(ANT), list(POST); FREE = [x for x in (a.free_anterior, a.free_posterior) if x]; NAMES = ['costa'] + ANT + POST + FREE
LOG = open(os.path.join(a.out, 'summary.txt'), 'w')
def log(*s): print(*s); print(*s, file=LOG); LOG.flush()
d = json.load(open(a.coco)); cat = {c['id']: c['name'] for c in d['categories']}; by = {}
for an in d['annotations']: by.setdefault(an['image_id'], []).append(an)
ims = [im for im in d['images'] if not a.image_list_dir or os.path.exists(os.path.join(a.image_list_dir, os.path.splitext(im['file_name'])[0] + '.npz'))]
def outline(I):
    g = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY); bg = np.median(np.concatenate([g[:20].ravel(), g[-20:].ravel(), g[:, :20].ravel(), g[:, -20:].ravel()]))
    f = (np.abs(cv2.GaussianBlur(g, (0, 0), 1.5).astype(int) - bg) > 10).astype(np.uint8)
    if a.texture_outline:   # v1j: pale membrane on a pale background: add local texture (microtrichia) - background is smooth
        gf = g.astype(np.float32); sd = np.sqrt(np.maximum(cv2.blur(gf * gf, (15, 15)) - cv2.blur(gf, (15, 15)) ** 2, 0))
        bsd = np.median(np.concatenate([sd[:30].ravel(), sd[-30:].ravel(), sd[:, :30].ravel(), sd[:, -30:].ravel()]))
        f |= (cv2.GaussianBlur(sd, (0, 0), 3) > bsd * 2 + 1.5).astype(np.uint8)
        f = cv2.morphologyEx(f, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    n, lb, st, _ = cv2.connectedComponentsWithStats(f); f = (lb == 1 + st[1:, 4].argmax()).astype(np.uint8)
    cs, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); C = max(cs, key=cv2.contourArea).reshape(-1, 2)
    ib, ia = int(np.argmin(C[:, 0])), int(np.argmax(C[:, 0]))
    c1 = C[ib:ia + 1] if ib <= ia else np.concatenate([C[ib:], C[:ia + 1]]); c2 = C[ia:ib + 1] if ia <= ib else np.concatenate([C[ia:], C[:ib + 1]])
    if c1[:, 1].mean() > c2[:, 1].mean(): post, ant = c1, c2[::-1]
    else: post, ant = c2[::-1], c1          # both chains run base -> apex
    return f, ant, post
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
def branches(sk):
    """v1d: terminal -> path back to the first TRUE junction. Crossing number (0->1 transitions around the 8-ring) separates real
    junctions (>= 3 arms) from staircase corners of an 8-connected skeleton, which stopped the v1c tracer after 1 px."""
    ys, xs = np.nonzero(sk); S = set(zip(ys.tolist(), xs.tolist()))
    def cn(p):
        r = [(p[0] + dy, p[1] + dx) in S for dy, dx in RING]; return sum((not r[k]) and r[(k + 1) % 8] for k in range(8))
    out = []
    for p in S:
        if cn(p) != 1: continue
        path = [p]; vis = {p}; cur = p
        while True:
            cand = [(cur[0] + dy, cur[1] + dx) for dy, dx in NB if (cur[0] + dy, cur[1] + dx) in S and (cur[0] + dy, cur[1] + dx) not in vis]
            if not cand: break
            if len(path) > 1 and cn(cur) >= 3: break
            if len(cand) > 1 and not all(max(abs(u[0] - v[0]), abs(u[1] - v[1])) <= 1 for u in cand for v in cand): break
            vis.update(cand)
            nxt = max(cand, key=lambda q: (q[0] - path[max(0, len(path) - 4)][0]) ** 2 + (q[1] - path[max(0, len(path) - 4)][1]) ** 2)
            path.append(nxt); cur = nxt
            if len(path) > 20000: break
        out.append(path)
    return out
def prune(sk, L, it=3):
    """v1c: remove skeleton spurs shorter than L px (end -> junction), a few rounds; vein tips often split into 2-3 short spurs."""
    sk = sk.copy()
    for _ in range(it):
        ch = False
        for br in branches(sk):
            if len(br) < L:
                for (y, x) in br[:-1]: sk[y, x] = 0
                ch = True
        if not ch: break
    return skeletonize(sk > 0).astype(np.uint8)
PROF = {}
if a.profiles:
    import csv as _csv, re as _re
    for r_ in _csv.reader(open(a.profiles), delimiter='\t'):
        if r_ and not r_[0].startswith('#') and len(r_) >= 5: PROF[r_[0]] = [[x for x in c.split(',') if x and x != '-'] for c in r_[1:5]]
    for v in PROF.values():
        for n in v[0] + v[1] + v[2] + v[3]:
            if n not in NAMES: NAMES.append(n)
rows = []; tally = {n: [0, 0, 0] for n in NAMES}; shown = 0
for im in ims:
    I = np.asarray(Image.open(os.path.join(a.images, im['file_name'])).convert('RGB')); H, W = I.shape[:2]; memb, ant, post = outline(I)
    gt = {}
    for an in by.get(im['id'], []):
        m = np.zeros((H, W), np.uint8); sg = an['segmentation']
        if isinstance(sg, dict):   # v1k: RLE keeps the holes (wing cells) that polygons of a vein NETWORK lose
            from pycocotools import mask as mu
            m = mu.decode(sg if isinstance(sg['counts'], str) else mu.frPyObjects(sg, H, W)).astype(np.uint8)
        else:
            for p in sg: cv2.fillPoly(m, [np.array(p).reshape(-1, 2).astype(np.int32)], 1)
        gt[cat[an['category_id']]] = gt.get(cat[an['category_id']], 0) | m
    if a.probs_dir:
        P = np.load(os.path.join(a.probs_dir, os.path.splitext(im['file_name'])[0] + '.npz'))['probs'].astype(np.float32)
        V = cv2.resize((P.argmax(-1) > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        V = np.zeros((H, W), np.uint8)
        for m in gt.values(): V |= m
    aband = np.zeros((H, W), np.uint8); cv2.polylines(aband, [ant.astype(np.int32)], False, 1, thickness=2 * a.costa_band + 1)
    costa = V & aband; rest = V & (1 - cv2.dilate(costa, np.ones((5, 5), np.uint8)))
    if a.skeleton_costa:   # v1i: skeletonise ALL veins first, then cut only the costa's own skeleton line (narrow band) -> veins joining C get a tip at C
        skf = skeletonize(cv2.morphologyEx(V, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0).astype(np.uint8)
        nb = np.zeros((H, W), np.uint8); cv2.polylines(nb, [ant.astype(np.int32)], False, 1, thickness=2 * a.skeleton_costa + 1)
        csk = skf & nb; costa = cv2.dilate(csk, np.ones((9, 9), np.uint8)) & V if csk.any() else costa
        sk = prune(skf & (1 - cv2.dilate(csk, np.ones((5, 5), np.uint8))), a.prune)
    else:
        sk = prune(skeletonize(cv2.morphologyEx(rest, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0).astype(np.uint8), a.prune)
    dist_ant = cv2.distanceTransform((1 - cv2.polylines(np.zeros((H, W), np.uint8), [ant.astype(np.int32)], False, 1, 3)).astype(np.uint8), cv2.DIST_L2, 3)
    dist_post = cv2.distanceTransform((1 - cv2.polylines(np.zeros((H, W), np.uint8), [post.astype(np.int32)], False, 1, 3)).astype(np.uint8), cv2.DIST_L2, 3)
    A_, P_, FR = [], [], []
    dist_costa = cv2.distanceTransform((1 - costa).astype(np.uint8), cv2.DIST_L2, 3) if costa.any() else np.full((H, W), 1e9, np.float32)
    for br in branches(sk):
        if len(br) < a.min_branch: continue
        bb = np.array(br)
        if np.median(dist_ant[bb[:, 0], bb[:, 1]]) < a.costa_band + a.costa_piece: continue   # v1b/v1h: a piece running INSIDE the costa band is not a vein of its own (R1 runs close to C)
        y, x = br[0]; da, dp = dist_ant[y, x], dist_post[y, x]
        xb = (x - ant[0][0]) / max(ant[-1][0] - ant[0][0], 1)
        # v1f: three explicit cases. touches the costa -> anterior vein; reaches the posterior margin -> posterior vein; else ends FREE
        if dist_costa[y, x] <= a.costa_touch: side_ant = True
        elif dp <= a.term_dist or (xb >= 0.45 and dp <= a.term_dist_distal): side_ant = False   # v1j: predicted veins often fade before the margin distally
        else:
            if xb < 0.6: FR.append((da, dp, br))
            continue
        if side_ant and xb < a.humeral_frac: continue   # humeral cross-vein
        ch = ant if side_ant else post; k = int(np.argmin(((ch - [x, y]) ** 2).sum(1))); pos = k / max(len(ch) - 1, 1)
        (A_ if side_ant else P_).append((pos, br))
    A_.sort(key=lambda t: -t[0]); P_.sort(key=lambda t: -t[0])   # anterior: from the apex backwards; posterior: from the apex towards the base
    ANT_, POST_, FA_, FP_ = ANT0, POST0, a.free_anterior, a.free_posterior
    if PROF:
        import re as _re; gm = _re.search(a.genus_regex, im['file_name']); pr = PROF.get(gm.group(1) if gm else '')
        if pr: ANT_, POST_, FA_, FP_ = pr[0], pr[1], (pr[2][0] if pr[2] else ''), (pr[3][0] if pr[3] else '')
    ANT, POST = ANT_, POST_
    named = [('costa', None)] + [(ANT[i], br) for i, (_, br) in enumerate(A_[:len(ANT)])] + [(POST[i], br) for i, (_, br) in enumerate(P_[:len(POST)])]
    if FR and FA_ and any(t[0] < t[1] for t in FR):   # v1g: Sc = the free anterior vein ending FARTHEST from the base (humeral is shorter, more basal)
        fa = max([t for t in FR if t[0] < t[1]], key=lambda t: t[2][0][1]); named.append((FA_, fa[2])); FR = [f for f in FR if f is not fa]
    if FR and FP_: named.append((FP_, min(FR, key=lambda t: t[1])[2]))
    res = {'image': im['file_name'], 'n_ant_terminals': len(A_), 'n_post_terminals': len(P_)}
    gtd = {n: cv2.dilate(m, np.ones((7, 7), np.uint8)) for n, m in gt.items()}
    for nm, br in named:
        if br is None: pix = np.column_stack(np.nonzero(costa)) if costa.any() else np.zeros((0, 2), int)
        else: pix = np.array(br[:max(3, len(br) // 2)])   # v1g: score the DISTAL half (the part that reaches the margin / ends free)
        if len(pix) == 0: res[nm] = 'missing'; continue
        votes = {n: int(m[pix[:, 0], pix[:, 1]].sum()) for n, m in gtd.items()}; best = max(votes, key=votes.get)
        res[nm] = 'correct' if best == nm and votes[best] > 0 else f'wrong({best if votes[best] else "no vein"})'
    for nm in NAMES:
        if nm not in res: res[nm] = 'missing'
        tally[nm][0 if res[nm] == 'correct' else 1 if res[nm].startswith('wrong') else 2] += 1
    rows.append(res)
    if shown < a.sheet:
        s = 1000 / W; O = cv2.resize(I, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); cols = {n: tuple(int(c) for c in cv2.applyColorMap(np.uint8([[int(255 * i / len(NAMES))]]), cv2.COLORMAP_TURBO)[0, 0][::-1]) for i, n in enumerate(NAMES)}
        cm = cv2.resize(costa, (O.shape[1], O.shape[0]), interpolation=cv2.INTER_NEAREST) > 0; O[cm] = cols['costa']
        for nm, br in named[1:]:
            for (y, x) in br[::2]: cv2.circle(O, (int(x * s), int(y * s)), 2, cols[nm], -1)
            y, x = br[0]; ok = res[nm] == 'correct'
            cv2.putText(O, nm, (int(x * s) + 4, int(y * s) - 4), 0, 0.6, (0, 0, 0), 3); cv2.putText(O, nm, (int(x * s) + 4, int(y * s) - 4), 0, 0.6, (0, 150, 0) if ok else (220, 0, 0), 2)
        cv2.putText(O, im['file_name'][:40] + ('  [predicted veins]' if a.probs_dir else '  [veins from COCO]'), (8, 24), 0, 0.7, (0, 0, 0), 2)
        Image.fromarray(O).save(os.path.join(a.out, f'named_{os.path.splitext(im["file_name"])[0]}.jpg'), quality=88); shown += 1
keys = ['image', 'n_ant_terminals', 'n_post_terminals'] + NAMES
with open(os.path.join(a.out, 'per_wing.tsv'), 'w') as fh:
    fh.write('\t'.join(keys) + '\n'); [fh.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n') for r in rows]
log(f'vein naming rule v1 | {len(rows)} wings | veins from {"predicted probs " + a.probs_dir if a.probs_dir else "hand outlines (segmentation perfect)"}')
log(f'{"name":10s} {"correct":>8s} {"wrong":>6s} {"missing":>8s}')
for n in NAMES: c, w, m = tally[n]; log(f'{n:10s} {c / len(rows):8.2f} {w / len(rows):6.2f} {m / len(rows):8.2f}')
tot = np.array([tally[n] for n in NAMES]).sum(0); log(f'ALL       {tot[0] / tot.sum():8.2f} {tot[1] / tot.sum():6.2f} {tot[2] / tot.sum():8.2f}')
from collections import Counter
for n in NAMES:
    errs = Counter(r[n] for r in rows if r[n].startswith('wrong'))
    if errs: log(f'  {n} confused with: ' + ', '.join(f'{k[6:-1]} x{v}' for k, v in errs.most_common(3)))
