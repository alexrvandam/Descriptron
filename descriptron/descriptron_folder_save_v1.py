"""Folder-session save for the Descriptron GUI (v89, 18 Oct 2026). Pure functions, no Tk, so they can be tested.

Three faults in the marmot save when stepping through a folder:
  1. the keypoints of the image on screen were attached to the FIRST image in the file;
  2. the save emptied the session, so a second save held only the images touched after the first one;
  3. images never opened in the session were left out, and keypoints auto-saved for other images went in
     with category_id -1 on a second, duplicate image entry.
build_folder_save() returns new image and annotation lists for one save: one image entry per file, the image
on screen first (the save attaches its live keypoints to images[0]), every visited image's keypoints from the
folder cache, and the untouched images copied from the annotation file the session was loaded from.
The session's own lists are not changed; the caller restores them after the save."""
import json, os, copy

IMAGE_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif')


def stem(name):
    n = os.path.basename(str(name))
    for e in IMAGE_EXTS + tuple(x.upper() for x in IMAGE_EXTS):
        if n.endswith(e):
            return n[:-len(e)]
    return n


def kp_annotation(st, image_id, cat_id, ann_id, status_names=None):
    """COCO keypoint annotation from a folder-cache entry {'coords','vis','orders'} (same fields as the GUI writes)."""
    coords, vis = st['coords'], list(st.get('vis') or [2] * len(st['coords']))
    vis += [2] * (len(coords) - len(vis))
    kps, clicked, shown = [], [], []
    for (x, y), v in zip(coords, vis):
        kps += [float(x), float(y), int(v)] if v > 0 else [0, 0, 0]
        clicked += [float(x), float(y), max(int(v), 0)]
        if v > 0:
            shown.append((x, y))
    shown = shown or [(0, 0)]
    xs, ys = [p[0] for p in shown], [p[1] for p in shown]
    names = status_names or {2: 'present', 1: 'present', 0: 'absent', -1: 'unknown'}
    return {'id': ann_id, 'image_id': image_id, 'category_id': cat_id, 'keypoints': kps,
            'num_keypoints': sum(1 for v in vis[:len(coords)] if v > 0),
            'point_order': list(st.get('orders') or range(1, len(coords) + 1)), 'keypoints_clicked': clicked,
            'keypoint_status': [names.get(v, 'unknown') for v in vis[:len(coords)]],
            'bbox': [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], 'area': 1, 'iscrowd': 0}


def _is_autosaved_kp(a):
    return bool(a.get('keypoints')) and 'keypoints_clicked' in a and not a.get('source_line_category')


def build_folder_save(acc, current, folder_images, kp_cache, mask_cache, source_path=None, kp_cat_name='line_keypoints'):
    """acc: the session accumulator {'images','annotations','categories'} (categories may be extended in place).
    current: dict(id, file_name, height, width) of the image on screen, or None.
    folder_images: paths of the folder; kp_cache / mask_cache: the GUI's per-image caches (keys = image paths).
    Returns (images, annotations)."""
    cats = acc.setdefault('categories', [])
    def cat_id(name):
        for c in cats:
            if c['name'] == name:
                return c['id']
        nid = max([c['id'] for c in cats if isinstance(c.get('id'), int)] or [0]) + 1
        cats.append({'id': nid, 'name': name, 'supercategory': 'keypoints' if name == kp_cat_name else ''})
        return nid
    kpc = cat_id(kp_cat_name)
    # one entry per file; prefer the entry the masks use (id != bare stem) over the keypoint auto-save duplicate
    imgs = [copy.deepcopy(i) for i in acc.get('images', [])]
    if current and stem(current['file_name']) not in {stem(i.get('file_name') or i.get('id')) for i in imgs}:
        imgs.append(dict(current))
    canon, order = {}, []
    for im in imgs:
        s = stem(im.get('file_name') or im.get('id'))
        if s not in canon:
            canon[s] = im; order.append(s)
        elif str(canon[s]['id']) == s and str(im['id']) != s:
            for k, v in canon[s].items():
                im.setdefault(k, v)
            canon[s] = im
    idmap = {}
    for im in acc.get('images', []) + ([current] if current else []):
        idmap[im['id']] = canon[stem(im.get('file_name') or im.get('id'))]['id']
    id2stem = {canon[s]['id']: s for s in order}
    cur_s = stem(current['file_name']) if current else None
    cached = {stem(p) for p, st in kp_cache.items()} | ({cur_s} if cur_s else set())
    anns, used = [], set()
    for a in acc.get('annotations', []):
        iid = a.get('image_id')
        iid = idmap.get(iid, canon[stem(iid)]['id'] if stem(iid) in canon else iid)   # unknown id: match by file name
        if _is_autosaved_kp(a) and id2stem.get(iid) in cached:
            continue                                   # replaced by the live points / the folder cache below
        fix_cat = _is_autosaved_kp(a) and a.get('category_id') in (-1, None)
        # the same dict when nothing changes, so in-place edits by the save (brush-edited masks) reach the session
        b = a if (iid == a.get('image_id') and not fix_cat) else dict(a, image_id=iid)
        if fix_cat:
            b['category_id'] = kpc
        anns.append(b); used.add(b.get('id'))
    nxt = max([x for x in used if isinstance(x, int)] or [0]) + 1
    for p, st in kp_cache.items():                     # visited images other than the one on screen
        s = stem(p)
        if s == cur_s or not st or not st.get('coords') or s not in canon:
            continue
        anns.append(kp_annotation(st, canon[s]['id'], kpc, nxt)); used.add(nxt); nxt += 1
    # images never opened in this session: copy them from the file the session was loaded from
    visited = {stem(p) for p in list(kp_cache) + list(mask_cache)} | ({cur_s} if cur_s else set())
    folder_stems = {stem(p) for p in folder_images}
    if source_path and os.path.exists(source_path):
        src = json.load(open(source_path))
        rle = {}
        sib = os.path.splitext(source_path)[0] + '.rlejson'
        if source_path.endswith('.json') and os.path.exists(sib):
            try:
                rle = {a['id']: a['segmentation'] for a in json.load(open(sib)).get('annotations', []) if isinstance(a.get('segmentation'), dict)}
            except Exception:
                rle = {}
        scat = {c['id']: c['name'] for c in src.get('categories', [])}
        sim = {}
        for im in src.get('images', []):
            s = stem(im.get('file_name') or im.get('id'))
            if s in folder_stems and s not in visited and s not in canon:
                sim[im['id']] = s
        for iid, s in sim.items():
            if s not in canon:
                im = copy.deepcopy(next(i for i in src['images'] if i['id'] == iid)); canon[s] = im; order.append(s)
        for a in src.get('annotations', []):
            if a.get('image_id') not in sim:
                continue
            s = sim[a['image_id']]; b = copy.deepcopy(a); b['image_id'] = canon[s]['id']
            name = scat.get(a.get('category_id'))
            b['category_id'] = kpc if (_is_autosaved_kp(a) and (name in (None, kp_cat_name, 'keypoints'))) else (cat_id(name) if name else a.get('category_id'))
            if a.get('id') in rle:
                b['segmentation_rle'] = rle[a['id']]
            if b.get('id') in used:
                b['id'] = nxt; nxt += 1
            used.add(b.get('id')); anns.append(b)
    images = [canon[s] for s in order]
    if cur_s in canon:                                 # the save attaches the live keypoints to images[0]
        images.remove(canon[cur_s]); images.insert(0, canon[cur_s])
    return images, anns
