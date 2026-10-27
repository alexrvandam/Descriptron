"""Point types for the Descriptron GUI keypoint tool (v98): pure functions, no Tk.

The keypoint tool marks both ordered homologous landmarks and many unordered small structures (setae, pits, pores,
scales ...). Each placed point now records a TYPE. "landmark" is the default and means exactly what every point
meant before v98; other types are free names the user adds (seta, pit, ...).

The GUI keeps one string per point in `keypoint_types`, parallel to point_coords / point_labels / point_orders /
keypoint_labels (and keypoint_visibility). Saved files get one extra field on each keypoint annotation:

    "keypoint_types": ["landmark", "landmark", "seta", "seta", "pit"]

one entry per point, in the same order as "keypoints" / "keypoints_clicked". Nothing else in the file changes.
A file without the field is read as all "landmark".

What is here:
  * the type list and its persistence (~/.config/descriptron/point_types.json, or $DESCRIPTRON_POINT_TYPES)
  * a deterministic colour and a short tag per type (for drawing)
  * TypeTracker: keeps the GUI's keypoint_types aligned with point_coords through add / delete / reorder / move /
    clear / hide-show / undo, by following the point objects (identity) and, for copies, their coordinates
  * the JSON field: read it from an annotation, find the annotation that holds a given set of points, and add the
    field to a COCO dict or file for the point sets whose types are known
"""
import hashlib
import json
import os
import tempfile

DEFAULT = "landmark"
FIELD = "keypoint_types"

# distinct, readable on photographs; landmark points keep their usual green / red / goldenrod / purple crosses
PALETTE = ("#ff7f0e", "#1f77b4", "#e377c2", "#17becf", "#bcbd22", "#8c564b",
           "#ff1493", "#00ced1", "#ffd700", "#7f7fff", "#ff4500", "#2e8b57")


# --------------------------------------------------------------------------------------------- the type list
def config_path():
    p = os.environ.get("DESCRIPTRON_POINT_TYPES")
    if p:
        return p
    return os.path.join(os.path.expanduser("~"), ".config", "descriptron", "point_types.json")


def normalize_name(name):
    """'  scale seta ' -> 'scale_seta'; '' / None -> None. Case is kept."""
    if name is None:
        return None
    s = "_".join(str(name).split())
    return s or None


def merge_types(types):
    """Deduplicated (case-insensitive), 'landmark' first."""
    out, seen = [DEFAULT], {DEFAULT.lower()}
    for t in types or []:
        t = normalize_name(t)
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def load_types(path=None):
    """The saved list (always starts with 'landmark'); ['landmark'] if there is no file or it cannot be read."""
    path = path or config_path()
    try:
        with open(path) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = data.get("point_types", [])
        if not isinstance(data, list):
            return [DEFAULT]
        return merge_types([t for t in data if isinstance(t, str)])
    except Exception:
        return [DEFAULT]


def save_types(types, path=None):
    """Write the list; True on success (never raises)."""
    path = path or config_path()
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"point_types": merge_types(types)}, fh, indent=1)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def add_type(types, name):
    """(new list, canonical name) - an existing name (any case) is returned as it is spelled in the list."""
    n = normalize_name(name)
    types = merge_types(types)
    if not n:
        return types, None
    for t in types:
        if t.lower() == n.lower():
            return types, t
    return types + [n], n


# --------------------------------------------------------------------------------------------- drawing
def is_landmark(t):
    return t in (None, "", DEFAULT)


def colour_for(t):
    """None for landmark (drawn as before); a fixed palette colour per name otherwise (same name -> same colour)."""
    if is_landmark(t):
        return None
    h = int(hashlib.md5(str(t).encode("utf-8")).hexdigest(), 16)
    return PALETTE[h % len(PALETTE)]


def tag_for(t, known=()):
    """Shortest prefix (min. 1 letter) that tells this type apart from the other non-landmark types in `known`."""
    if is_landmark(t):
        return ""
    t = str(t)
    others = [str(k).lower() for k in known if not is_landmark(k) and str(k).lower() != t.lower()]
    for n in range(1, len(t) + 1):
        p = t[:n].lower()
        if not any(o.startswith(p) for o in others):
            return t[:n]
    return t


# --------------------------------------------------------------------------------------------- alignment
def fit(types, n, default=DEFAULT):
    """types padded with default / cut to n entries (None -> all default)."""
    types = [t if isinstance(t, str) and t else default for t in (types or [])][:n]
    return types + [default] * (n - len(types))


def _vkey(c):
    try:
        return (round(float(c[0]), 4), round(float(c[1]), 4))
    except Exception:
        return None


class TypeTracker(object):
    """Keeps `types` (the GUI's keypoint_types list, always changed in place) aligned with point_coords.

    The GUI's operations move the point objects themselves (remove = del, reorder = pop / insert), so after any of
    them every point that survived is still the same Python list: its type follows it. A point that is a new object
    is looked up by its coordinates among the points known before (hide / show copies the lists) and the hidden
    sets (bank snapshots carry 'types'); otherwise it gets `new_type` (a click) or 'landmark'."""

    def __init__(self, types):
        self.types = types
        self._objs = []
        self._memo = {}                             # coordinates -> type of points seen in hidden sets

    def _remember_bank(self, bank):
        for snap in bank or ():
            try:
                cs = snap.get("coords") or []
                for c, t in zip(cs, fit(snap.get("types"), len(cs))):
                    self._memo[_vkey(c)] = t
            except Exception:
                pass

    def baseline(self, coords):
        self._objs = list(coords)

    def sync(self, coords, bank=(), new_type=None, trust=False):
        n = len(coords)
        self._remember_bank(bank)                   # Show All empties the bank before the points come back
        if trust:                                   # the types list was put back by someone who knows (undo / redo)
            self.types[:] = fit(self.types, n)
            self._objs = list(coords)
            return self.types
        if len(self._objs) == n == len(self.types) and all(a is b for a, b in zip(self._objs, coords)):
            return self.types
        old_t = fit(self.types, len(self._objs))
        by_id = {id(o): t for o, t in zip(self._objs, old_t)}
        by_val = dict(self._memo)
        for o, t in zip(self._objs, old_t):
            by_val[_vkey(o)] = t
        new = []
        for c in coords:
            if id(c) in by_id:
                new.append(by_id[id(c)])
            elif new_type is not None:
                new.append(new_type)
            else:
                new.append(by_val.get(_vkey(c), DEFAULT))
        self.types[:] = new
        self._objs = list(coords)
        return self.types

    def set(self, coords, types):
        self._memo = {}                             # a load starts afresh
        self.types[:] = fit(types, len(coords))
        self._objs = list(coords)
        return self.types

    def set_one(self, coords, i, t):
        self.sync(coords)
        if 0 <= i < len(self.types):
            self.types[i] = t if t else DEFAULT
        return self.types


# --------------------------------------------------------------------------------------------- JSON field
def types_from_annotation(ann, n=None):
    """The annotation's keypoint_types, fitted to n points (n defaults to its number of keypoints);
    all 'landmark' when the field is missing or malformed."""
    if n is None:
        n = len((ann or {}).get("keypoints") or []) // 3
    t = (ann or {}).get(FIELD)
    if not isinstance(t, list) or (t and len(t) != n):
        return [DEFAULT] * n
    return fit(t, n)


def annotation_coords(ann):
    """Per point (x, y) as the GUI loads it: the visible position, else the clicked one, else (0, 0);
    plus the alternative (x, y) for slots where the two loaders could differ."""
    kp = ann.get("keypoints") or []
    cl = ann.get("keypoints_clicked") or []
    out = []
    for i in range(len(kp) // 3):
        x, y, v = kp[3 * i], kp[3 * i + 1], kp[3 * i + 2]
        try:
            v = int(v)
        except Exception:
            v = 0
        if v > 0:
            out.append(((x, y),))
        else:
            alts = []
            if len(cl) >= 3 * i + 2:
                alts.append((cl[3 * i], cl[3 * i + 1]))
            alts.append((x, y))
            out.append(tuple(alts))
    return out


def clicked_coords(ann):
    """(x, y) per point from keypoints_clicked (else keypoints)."""
    src = ann.get("keypoints_clicked") or ann.get("keypoints") or []
    return [(src[3 * i], src[3 * i + 1]) for i in range(len(src) // 3)]


def _close(a, b, tol):
    try:
        return abs(float(a[0]) - float(b[0])) <= tol and abs(float(a[1]) - float(b[1])) <= tol
    except Exception:
        return False


def coords_match_annotation(ann, coords, tol=1e-6):
    ac = annotation_coords(ann)
    if len(ac) != len(coords):
        return False
    return all(any(_close(a, c, tol) for a in alts) for alts, c in zip(ac, coords))


def is_point_annotation(ann):
    return bool(ann.get("keypoints")) and not ann.get("source_line_category")


def find_types_for_coords(annotations, coords, tol=1e-6):
    """keypoint_types of the (first) keypoint annotation that holds exactly these points; all 'landmark' if that
    annotation has no field or no annotation matches."""
    n = len(coords)
    for a in annotations or []:
        if is_point_annotation(a) and coords_match_annotation(a, coords, tol):
            return types_from_annotation(a, n)
    return [DEFAULT] * n


def add_types_to_coco(data, typed_sets, tol=1e-6):
    """Add keypoint_types to every keypoint annotation (not line-derived) that has none and whose clicked points
    are one of typed_sets [(coords, types), ...]. Returns the number of annotations changed."""
    sets = [(list(c), fit(t, len(c))) for c, t in typed_sets or [] if c]
    k = 0
    for a in (data or {}).get("annotations", []) or []:
        if not is_point_annotation(a) or FIELD in a:
            continue
        cc = clicked_coords(a)
        for c, t in sets:
            if len(c) == len(cc) and all(_close(p, q, tol) for p, q in zip(cc, c)):
                a[FIELD] = list(t)
                k += 1
                break
    return k


def patch_json_file(path, typed_sets, indent=4):
    """add_types_to_coco on a saved file, rewritten atomically with the same indent; returns the count (0: untouched)."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception:
        return 0
    k = add_types_to_coco(data, typed_sets)
    if not k:
        return 0
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".pt_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=indent)
        try:
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)      # keep the saved file's permissions
        except Exception:
            pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        return 0
    return k
