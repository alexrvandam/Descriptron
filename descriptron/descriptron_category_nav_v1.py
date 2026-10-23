"""Category navigator logic for the Descriptron GUI (v91): pure functions, no Tk.

The GUI keeps its masks in `segmentation_masks` (list of dicts with 'category_id', sometimes 'label') and
their names in `labels` (mask index -> name). These helpers turn that into "name (n)" entries and find the
mask to jump to, so the GUI can make it the current mask exactly as Next Mask does.
"""
import json
import os

UNLABELED = "unlabeled"


def mask_category_names(masks, labels=None, id_to_name=None, prefer_category_id=False):
    """One category name per mask, in mask order.

    Normal mode: labels[i], else mask['label'], else id_to_name[category_id], else 'unlabeled'.
    prefer_category_id (prediction view, as Next Mask does): id_to_name[category_id] first.
    """
    labels = labels or {}
    id_to_name = id_to_name or {}
    out = []
    for i, m in enumerate(masks or []):
        m = m or {}
        by_id = id_to_name.get(m.get("category_id"))
        cands = [by_id, labels.get(i), m.get("label")] if prefer_category_id else \
                [labels.get(i), m.get("label"), by_id]
        name = next((str(c) for c in cands if c not in (None, "")), UNLABELED)
        out.append(name)
    return out


def category_counts(names):
    """[(name, n), ...] in order of first appearance (stable while masks keep their order)."""
    counts, order = {}, []
    for n in names:
        if n not in counts:
            counts[n] = 0
            order.append(n)
        counts[n] += 1
    return [(n, counts[n]) for n in order]


def format_entry(name, n):
    return f"{name} ({n})"


def parse_entry(entry):
    """'seta (12)' -> 'seta'; a name that itself ends in '(x)' keeps it (only a trailing ' (<int>)' is cut)."""
    s = str(entry or "")
    if s.endswith(")") and " (" in s:
        head, tail = s.rsplit(" (", 1)
        if tail[:-1].isdigit():
            return head
    return s


def first_index(names, name):
    for i, n in enumerate(names):
        if n == name:
            return i
    return None


def next_index(names, name, current):
    """The next mask of `name` after `current`, wrapping round; None if the image has none."""
    k = len(names)
    if not k:
        return None
    start = current if isinstance(current, int) and 0 <= current < k else -1
    for step in range(1, k + 1):
        i = (start + step) % k
        if names[i] == name:
            return i
    return None


def target_index(names, name, current):
    """Where a jump to `name` goes: the next one of that category if the current mask is already one
    (selecting it again cycles), otherwise the first one."""
    if isinstance(current, int) and 0 <= current < len(names) and names[current] == name:
        return next_index(names, name, current)
    return first_index(names, name)


def folder_category_counts(coco_path, image_paths):
    """Counts per category over the folder's images in a COCO file as saved on disk:
    [(name, n_masks, n_images), ...] sorted by name. Only annotations whose image file name is one of
    `image_paths` count. Returns [] if the file cannot be read."""
    try:
        with open(coco_path) as fh:
            coco = json.load(fh)
    except Exception:
        return []
    wanted = {os.path.basename(str(p)) for p in (image_paths or [])}
    wanted_stems = {os.path.splitext(w)[0] for w in wanted}
    ok_ids = set()
    for im in coco.get("images", []):
        fn = os.path.basename(str(im.get("file_name", "")))
        if fn in wanted or os.path.splitext(fn)[0] in wanted_stems:
            ok_ids.add(im.get("id"))
    id_to_name = {c.get("id"): str(c.get("name")) for c in coco.get("categories", [])}
    n_masks, imgs = {}, {}
    for a in coco.get("annotations", []):
        if a.get("image_id") not in ok_ids or not a.get("segmentation"):
            continue
        nm = id_to_name.get(a.get("category_id"), UNLABELED)
        n_masks[nm] = n_masks.get(nm, 0) + 1
        imgs.setdefault(nm, set()).add(a.get("image_id"))
    return sorted(((nm, n_masks[nm], len(imgs[nm])) for nm in n_masks), key=lambda t: t[0].lower())
