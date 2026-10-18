#!/usr/bin/env python3
"""Shared additions for the TPS homology grids (texture_phenomics_homology.py, color_phenomics_homology_v2_1.py).
Added 12 Oct 2026. All of it is opt-in; with the defaults (--grid_ext 0, no occluders) the calling scripts behave as before.

1. Extended grid (--grid_ext N): the regular grid over the consensus bounding box is extended by N cells on every side, and a
   cell is kept when at least --min_cell_overlap of its area lies inside the consensus outline (instead of requiring the
   cell CENTRE inside), so cells along the margins are no longer dropped. The TPS warp itself is unchanged: it is defined by
   the semilandmarks only.
2. Occluders (--occluder_categories scape,antenna,...): annotations of those categories are removed from the foreground
   mask, so features are computed from VISIBLE pixels only, and every cell gets a visible fraction
   = visible pixels / (visible + occluded pixels inside the outline). Tested on 65 Austromyrma heads by transplanting real
   scape coverage between specimens through the TPS warps: sampling only visible pixels gave the lowest error (total
   variation 0.050 per damaged cell, band-level bias 0.0004); dropping cells under 60% visible, coverage weighting and
   mirror filling were no better or worse (ant_dorsal/texture_v1/PUBLICATION_texture_sculpture_20261012, result 7).
"""
import os, json, logging
import numpy as np
import cv2


def create_grid_extended(contour, n_rows=8, n_cols=10, ext=2, min_overlap=0.02):
    """Grid of n_rows x n_cols cells over the contour's bounding box, extended by `ext` cells on each side.
    Returns the same grid_info keys as create_grid_inside_contour (x_lines, y_lines, n_rows, n_cols, cell_inside),
    with n_rows/n_cols including the extension, plus 'ext' and 'cell_overlap' (fraction of each cell inside the outline)."""
    pts = contour[:-1] if len(contour) > 1 and np.array_equal(contour[0], contour[-1]) else contour
    (xmin, ymin), (xmax, ymax) = pts.min(axis=0), pts.max(axis=0)
    dx, dy = (xmax - xmin) / n_cols, (ymax - ymin) / n_rows
    x_lines = xmin + dx * np.arange(-ext, n_cols + ext + 1)
    y_lines = ymin + dy * np.arange(-ext, n_rows + ext + 1)
    NR, NC = len(y_lines) - 1, len(x_lines) - 1
    # rasterise the outline and the cells on a fine canvas to measure overlap
    S = 3000.0 / max(x_lines[-1] - x_lines[0], y_lines[-1] - y_lines[0])   # fine canvas: a touching edge is <1% of a cell
    W, H = int((x_lines[-1] - x_lines[0]) * S) + 2, int((y_lines[-1] - y_lines[0]) * S) + 2
    to_px = lambda p: np.round((np.asarray(p) - [x_lines[0], y_lines[0]]) * S).astype(np.int32)
    poly = np.zeros((H, W), np.uint8); cv2.fillPoly(poly, [to_px(pts)], 1)
    overlap = np.zeros((NR, NC))
    for r in range(NR):
        for c in range(NC):
            a0, b0 = to_px([x_lines[c], y_lines[r]]); a1, b1 = to_px([x_lines[c + 1], y_lines[r + 1]])
            blk = poly[b0:b1, a0:a1]; overlap[r, c] = blk.mean() if blk.size else 0.0
    cell_inside = overlap >= min_overlap
    logging.info(f"Extended grid: {n_rows}x{n_cols} + {ext} cells each side = {NR}x{NC}; "
                 f"{cell_inside.sum()} cells overlap the outline by >= {min_overlap:.0%}")
    return {'x_lines': x_lines, 'y_lines': y_lines, 'n_rows': NR, 'n_cols': NC, 'cell_inside': cell_inside,
            'ext': ext, 'cell_overlap': overlap}


def build_occluder_lookup(coco_json_path, names, target=None):
    """basename -> list of annotations of the occluder categories. Unknown names are reported and IGNORED
    (no fallback to another category). 'auto' = categories flagged "occluder": true in the COCO file. With `target`,
    the target itself and categories nested with it (part/container) are dropped."""
    names = [n.strip().lower() for n in names if n.strip()]
    if not names:
        return {}
    coco = json.load(open(coco_json_path))
    if 'auto' in names or target or any('>' in n for n in names):
        occ = resolve_occluders(','.join(names), coco)
        names = sorted(occluders_for(target, occ, containment_related(coco)) if target else occ)
        if not names:
            return {}
    ids = {c['id']: c['name'] for c in coco.get('categories', []) if c['name'].lower() in names}
    missing = set(names) - {v.lower() for v in ids.values()}
    if missing:
        logging.warning(f"Occluder categories not in the COCO file (ignored): {sorted(missing)}")
    img = {i['id']: os.path.basename(i.get('file_name', '')) for i in coco.get('images', [])}
    look = {}
    for a in coco.get('annotations', []):
        if a.get('category_id') in ids and a['image_id'] in img:
            look.setdefault(img[a['image_id']], []).append(a)
    logging.info(f"Occluders {sorted(ids.values())}: annotations on {len(look)} images")
    return look


def visible_and_expected(fg_mask, occ_anns, decode_fn):
    """fg_mask (uint8 0/1) and a list of occluder annotations -> (visible uint8, expected uint8).
    expected = fg plus the occluded part lying within the convex hull of fg (what would be visible without the occluder)."""
    h, w = fg_mask.shape
    occ = np.zeros((h, w), np.uint8)
    for a in occ_anns or []:
        occ |= decode_fn(a, h, w)
    if not occ.any():
        return fg_mask, fg_mask
    cn, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = np.zeros((h, w), np.uint8)
    if cn:
        cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cn))], 1)
    visible = fg_mask & (1 - occ)
    expected = visible | (occ & hull)
    return visible.astype(np.uint8), expected.astype(np.uint8)


def cell_visible_fraction(cell_quad_mask_fn, visible, expected):
    """cell_quad_mask_fn(mask) -> cell mask restricted to `mask` (the scripts' build_cell_mask with a bound cell)."""
    e = int(cell_quad_mask_fn(expected).sum())
    return (int(cell_quad_mask_fn(visible).sum()) / e) if e else np.nan


def write_coverage_csv(path, coverage, grid_info):
    """coverage: {filename: array (n_rows*n_cols)} -> long CSV of the cells kept by the grid."""
    ci = grid_info['cell_inside']; NC = grid_info['n_cols']
    with open(path, 'w') as fh:
        fh.write('filename,cell,row,col,visible_fraction\n')
        for fn, cov in coverage.items():
            for r in range(ci.shape[0]):
                for c in range(ci.shape[1]):
                    if ci[r, c]:
                        v = cov[r * NC + c]
                        fh.write(f'{fn},r{r}c{c},{r},{c},{"" if np.isnan(v) else f"{v:.4f}"}\n')


def repair_outline(mask, occ_anns, h, w, decode_fn, close_frac=0.01, touch_px=3):
    """Outline for semilandmarks when parts are hidden: mask + the occluders lying within its convex hull, closed and
    hole-filled, largest component. Used by the semilandmark scripts (V34/V42) --repair_outline_with.
    Keeps antennae/legs crossing the margin from denting the contour (which bends the TPS warp).
    Only occluder pieces that TOUCH the structure (within touch_px) are added, so an occluder lying in a concavity of the
    convex hull without covering the structure cannot fill that concavity. Where the margin is hidden the outline follows
    the convex hull (a straight chord); estimate_hidden_semilandmarks_v1.py replaces those points by a TPS estimate."""
    from scipy.ndimage import binary_fill_holes
    if mask.sum() == 0:
        return mask
    occ = np.zeros((h, w), np.uint8)
    for a in occ_anns or []:
        occ |= (decode_fn(a, h, w) > 0).astype(np.uint8)
    cn, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = np.zeros((h, w), np.uint8); cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cn))], 1)
    add = (occ & hull & (1 - mask)).astype(np.uint8)
    if add.any() and touch_px > 0:
        near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch_px + 1, 2 * touch_px + 1)))
        n_, lab_ = cv2.connectedComponents(add)
        keep = np.unique(lab_[(near > 0) & (add > 0)]); keep = keep[keep > 0]
        add = np.isin(lab_, keep).astype(np.uint8)
    m = (mask | add).astype(np.uint8)
    ys, xs = np.where(mask); k = max(3, int(close_frac * max(np.ptp(xs), np.ptp(ys))) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = binary_fill_holes(m).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    if n > 2:
        m = (lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1).astype(np.uint8)
    return m


def containment_related(coco, frac=0.9, min_images=2):
    """Pairs of categories where one lies inside the other (median share of the smaller inside the larger >= frac over
    >= min_images images), e.g. cephalic_dorsum inside head_capsule. Such a pair is a part and its container, never an
    occluder and the structure it hides. Returns {category name: set of related names}. Same rule as DINOMask v2."""
    from pycocotools import mask as mu
    name = {c['id']: c['name'] for c in coco.get('categories', [])}
    size = {i['id']: (i.get('height'), i.get('width')) for i in coco.get('images', [])}
    by = {}
    for a in coco.get('annotations', []):
        if a.get('segmentation') and not a.get('is_line') and a['image_id'] in size and None not in size[a['image_id']]:
            by.setdefault(a['image_id'], {}).setdefault(name.get(a['category_id']), []).append(a)
    share = {}
    for iid, cats in by.items():
        h, w = size[iid]
        def dec(lst):
            m = np.zeros(((h + 3) // 4, (w + 3) // 4), bool)
            for a in lst:
                sg = a['segmentation']
                r = mu.merge(mu.frPyObjects(sg, h, w)) if isinstance(sg, list) else (mu.frPyObjects(sg, h, w) if isinstance(sg.get('counts'), list) else sg)
                m |= mu.decode(r)[::4, ::4].astype(bool)   # 4x coarser: enough to tell 'inside' from 'overlapping'
            return m
        M = {k: dec(v) for k, v in cats.items() if k}
        for A in M:
            for B in M:
                if A != B and M[B].sum() and M[B].sum() <= M[A].sum():
                    share.setdefault((A, B), []).append((M[A] & M[B]).sum() / M[B].sum())
    rel = {}
    for (A, B), v in share.items():
        if len(v) >= min_images and np.median(v) >= frac:
            rel.setdefault(A, set()).add(B); rel.setdefault(B, set()).add(A)
    return rel


def resolve_occluders(spec, coco):
    """Which category can hide which. Returns {occluder: set of structures it can hide, or {'*'} = any}.
    spec: comma list of entries; each entry is a category name (hides anything), 'name>a|b|c' (hides only a, b, c),
    or 'auto' = what the COCO categories say: "occludes": [names] (set in the GUI) or "occluder": true (anything).
    Pairs matter: a pronotum hides the head and eye but only ABUTS the elytron, so it must not be allowed to repair it.
    Unknown names are ignored with a warning; no fallback."""
    cats = {c['name'].lower(): c for c in coco.get('categories', [])}
    out = {}
    def add(o, targets):
        if o not in cats:
            logging.warning(f"Occluder category not in the COCO file (ignored): {o}"); return
        t = {x for x in targets if x == '*' or x in cats}
        bad = set(targets) - t
        if bad: logging.warning(f"'{o}' hides unknown categories (ignored): {sorted(bad)}")
        out.setdefault(o, set()).update(t or {'*'})
    for e in [x.strip() for x in (spec or '').split(',') if x.strip()]:
        if e.lower() == 'auto':
            for n, c in cats.items():
                if c.get('occludes'): add(n, {x.lower() for x in c['occludes']})
                elif c.get('occluder'): add(n, {'*'})
        elif '>' in e:
            o, t = e.split('>', 1); add(o.strip().lower(), {x.strip().lower() for x in t.split('|') if x.strip()})
        else:
            add(e.lower(), {'*'})
    return out


def occluders_for(target, occluders, related):
    """The occluders that can hide `target`: those allowed to hide it ('*' or listed), never the target itself and never
    a category nested with it (part/container). Occluders hide each other too (legs crossing legs)."""
    t = target.lower(); nested = {r.lower() for r in related.get(target, set()) | related.get(t, set())}
    occ = occluders if isinstance(occluders, dict) else {o: {'*'} for o in occluders}
    return {o for o, tg in occ.items() if o != t and o not in nested and ('*' in tg or t in tg)}

_SETUP_CACHE = {}
def occluder_setup(spec, coco_json_path):
    """(occluder names, containment relations) for a COCO file, computed once per run."""
    key = (spec, coco_json_path)
    if key not in _SETUP_CACHE:
        coco = json.load(open(coco_json_path))
        occ = resolve_occluders(spec, coco)
        rel = containment_related(coco) if occ else {}
        nested = sorted({o for o in occ if rel.get(o)})
        logging.info(f"Occluders: { {o: sorted(t) for o, t in occ.items()} }" + (f"; nested pairs never treated as occlusion for: {nested}" if nested else ''))
        _SETUP_CACHE[key] = (occ, rel)
    return _SETUP_CACHE[key]
