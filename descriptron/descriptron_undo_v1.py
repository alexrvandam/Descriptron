"""descriptron_undo_v1 - general undo / redo and unsaved-changes tracking for the Descriptron annotation GUI.

Pure Python (no Tk). The GUI keeps its state in module globals; this module works on any mapping `ns` (the GUI
passes globals()) and on named fields of it.

A state snapshot = copies of the named fields:
  * containers (dict / list / tuple) are copied recursively, numeric lists with a fast path;
  * numpy arrays (mask pixels) and any other non-container object are SHARED, not copied. Pixel edits done in
    place by the brush / eraser / lasso are undone by the GUI's own pixel stack (each entry stamped here with a
    sequence number, so one undo always takes whichever happened last).

Undo is strictly last-in-first-out over both kinds of entry. An entry made on another image is never restored:
undo refuses (the GUI shows why) until you are back on that image.

Unsaved changes ("dirty"): set by any recorded action that changed something, by any pixel edit, and by a change
of the session-wide fields (the COCO accumulator, the absent-structure records) against the copy taken at the
last save / load. Cleared by save / load.
"""
import numbers

try:                                      # numpy is optional for the pure tests
    import numpy as _np
    _NDARRAY = _np.ndarray
except Exception:                         # pragma: no cover
    _np = None
    _NDARRAY = ()

KP_FIELDS = ("point_coords", "point_labels", "point_orders", "keypoint_labels", "keypoint_visibility",
             "keypoint_bank")
ALL_FIELDS = KP_FIELDS + (
    "segmentation_masks", "labels", "current_mask_index", "structure_status",
    "coco_output_accumulate", "coco_output_mask", "instance_counters",
    "categories", "category_name_to_id", "category_id_to_name", "category_id_to_supercategory",
    "label_options", "category_label_options", "scale_bar_info", "bounding_boxes",
    "_UNDO_STACK")
# fields saved with the session file (changes here are unsaved work whatever image is on screen)
SESSION_FIELDS = ("coco_output_accumulate", "structure_status")
GROUPS = {"kp": KP_FIELDS, "all": ALL_FIELDS}

_SCALARS = (numbers.Number, str, bytes, type(None), bool)


# --------------------------------------------------------------------------------------------- copy / compare
def copy_value(v):
    """Copy containers recursively; share arrays and all other objects."""
    if isinstance(v, dict):
        return type(v)((k, copy_value(x)) for k, x in v.items()) if type(v) is dict else \
            _copy_dict_subclass(v)
    if isinstance(v, list):
        if v and isinstance(v[0], _SCALARS) and isinstance(v[-1], _SCALARS) and \
                all(isinstance(x, _SCALARS) for x in v):
            return list(v)
        return [copy_value(x) for x in v]
    if isinstance(v, tuple):
        return tuple(copy_value(x) for x in v)
    if isinstance(v, set):
        return set(v)
    return v                                           # ndarray, scalars, objects: shared


def _copy_dict_subclass(v):
    try:
        out = type(v)()
        for k, x in v.items():
            out[k] = copy_value(x)
        return out
    except Exception:
        return {k: copy_value(x) for k, x in v.items()}


def same(a, b):
    """Structural equality; numpy arrays compare by identity (in-place pixel edits are not seen here)."""
    if a is b:
        return True
    if isinstance(a, _NDARRAY) or isinstance(b, _NDARRAY):
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        if len(a) != len(b):
            return False
        for k, x in a.items():
            if k not in b or not same(x, b[k]):
                return False
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) and not (isinstance(a, list) and isinstance(b, list)):
            return False
        if len(a) != len(b):
            return False
        return all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (dict, list, tuple)) or isinstance(b, (dict, list, tuple)):
        return False
    try:
        r = a == b
        return bool(r) if isinstance(r, bool) else False
    except Exception:
        return False


_MISSING = object()


def capture(ns, fields):
    """Snapshot of ns[f] for f in fields (missing names are recorded as missing and left alone on restore)."""
    return {f: (copy_value(ns[f]) if f in ns else _MISSING) for f in fields}


def restore(ns, data):
    """Put a snapshot back. Lists / dicts already in ns are refilled in place (other references stay valid);
    everything else is rebound."""
    for f, v in data.items():
        if v is _MISSING:
            continue
        cur = ns.get(f, _MISSING)
        new = copy_value(v)                            # the stored snapshot stays pristine for redo
        if isinstance(cur, list) and isinstance(new, list):
            cur[:] = new
        elif isinstance(cur, dict) and isinstance(new, dict) and type(cur) is dict:
            cur.clear()
            cur.update(new)
        else:
            ns[f] = new


def differs(ns, data):
    """True if the live state differs from the snapshot in any of its fields."""
    for f, v in data.items():
        if v is _MISSING:
            if f in ns:
                return True
            continue
        if f not in ns or not same(v, ns[f]):
            return True
    return False


# --------------------------------------------------------------------------------------------- manager
class Entry(object):
    __slots__ = ("seq", "label", "key", "data", "group", "pending")

    def __init__(self, seq, label, key, data, group):
        self.seq, self.label, self.key, self.data, self.group = seq, label, key, data, group
        self.pending = False

    @property
    def fields(self):
        return tuple(self.data.keys())


class UndoManager(object):
    def __init__(self, max_states=30, session_fields=SESSION_FIELDS):
        self.max_states = int(max_states)
        self.session_fields = tuple(session_fields)
        self.states = []                  # Entry, oldest first
        self.redo = []                    # Entry, most recent last
        self._seq = 0
        self._stamps = {}                 # id(pixel payload) -> (weak/strong ref, seq, key)
        self.dirty_groups = set()         # 'kp' / 'all' changes since the last save / load
        self._session_baseline = None

    @property
    def dirty(self):
        return bool(self.dirty_groups)

    # ---- sequence ----
    def next_seq(self):
        self._seq += 1
        return self._seq

    # ---- state entries ----
    def begin(self, ns, label, key, group="all"):
        fields = GROUPS.get(group, group) if isinstance(group, str) else tuple(group)
        return Entry(self.next_seq(), label, key, capture(ns, fields), group if isinstance(group, str) else "custom")

    def commit(self, entry, ns, keep_if_noop=False):
        """Put a begun entry on the stack. A no-op is dropped unless keep_if_noop (a dialog that may still change
        things after it returned): then it is kept as 'pending' and becomes real when check_pending sees a change.
        Returns True if the entry was kept."""
        noop = not differs(ns, entry.data)
        if noop and not keep_if_noop:
            return False
        if self.states:
            top = self.states[-1]
            if top.key == entry.key and top.fields == entry.fields and _same_data(top.data, entry.data):
                self.states.pop()         # the previous entry turned out to be a no-op
        self.states.append(entry)
        del self.states[:-self.max_states]
        self.redo.clear()
        if noop:
            entry.pending = True
        else:
            self.dirty_groups.add(entry.group)
        return True

    def check_pending(self, ns, key):
        """Pending entries of this image that now differ from the live state become real (and mark dirty)."""
        hit = False
        for e in reversed(self.states):
            if e.pending and e.key == key and differs(ns, e.data):
                e.pending = False
                self.dirty_groups.add(e.group)
                hit = True
        return hit

    def has_pending(self):
        return any(e.pending for e in self.states)

    def pop_state(self):
        return self.states.pop() if self.states else None

    def push_redo(self, entry):
        self.redo.append(entry)

    def pop_redo(self):
        return self.redo.pop() if self.redo else None

    def push_after_redo(self, entry):
        """Undo entry for a redo: on the stack, the rest of the redo stack is kept."""
        entry.seq = self.next_seq()
        self.states.append(entry)
        del self.states[:-self.max_states]
        self.dirty_groups.add(entry.group)

    def clear(self):
        self.states.clear()
        self.redo.clear()
        self._stamps.clear()

    # ---- pixel entries (the GUI's own stack; stamped here) ----
    def stamp_pixel(self, obj, key, label="mask edit"):
        seq = self.next_seq()
        self._stamps[id(obj)] = (obj, seq, key, label)
        self.redo.clear()
        self.dirty_groups.add("all")
        return seq

    def pixel_stamp(self, obj):
        s = self._stamps.get(id(obj))
        if s is None or s[0] is not obj:
            return None
        return s[1], s[2], s[3]

    def prune_stamps(self, live_objects):
        keep = {id(o) for o in live_objects}
        for e in self.states + self.redo:
            st = e.data.get("_UNDO_STACK")
            if isinstance(st, (list, tuple)):
                for item in st:
                    if isinstance(item, tuple) and len(item) > 1:
                        keep.add(id(item[1]))
        for k in list(self._stamps):
            if k not in keep:
                del self._stamps[k]

    # ---- the decision for one undo ----
    def decide(self, ns, key, pixel_top=_MISSING):
        """What one undo should do now.
        pixel_top: the payload of the top entry of the GUI's pixel stack (or _MISSING / None when empty).
        Returns (kind, info): kind in 'none', 'state', 'pixel', 'refuse'. info: dict(label, key, seq).
        No-op state entries on top (of this image) are dropped on the way."""
        while True:
            st = self.states[-1] if self.states else None
            px = None
            if pixel_top is not _MISSING and pixel_top is not None:
                s = self.pixel_stamp(pixel_top)
                px = s if s is not None else (0, key, "mask edit")      # unstamped = older than any state
            if st is None and px is None:
                return "none", {}
            if st is not None and (px is None or st.seq > px[0]):
                if st.key != key:
                    return "refuse", {"label": st.label, "key": st.key, "seq": st.seq}
                if not differs(ns, st.data):
                    self.states.pop()                 # nothing to undo in it
                    continue
                return "state", {"label": st.label, "key": st.key, "seq": st.seq}
            if px[1] != key:
                return "refuse", {"label": px[2], "key": px[1], "seq": px[0]}
            return "pixel", {"label": px[2], "key": px[1], "seq": px[0]}

    # ---- unsaved changes ----
    def mark_clean(self, ns):
        self.dirty_groups.clear()
        for e in self.states:
            e.pending = False
        self._session_baseline = capture(ns, self.session_fields)

    def mark_keypoints_saved(self, ns, session_was_changed):
        """Save Keypoints wrote the keypoints only: keypoint changes are saved, mask / session changes are not.
        The keypoint save itself adds to the accumulator; that is absorbed unless the session had other changes."""
        self.dirty_groups.discard("kp")
        for e in self.states:
            if e.group == "kp":
                e.pending = False
        if not session_was_changed:
            self._session_baseline = capture(ns, self.session_fields)

    def refresh_session_baseline(self, ns):
        self._session_baseline = capture(ns, self.session_fields)

    def session_differs(self, ns):
        if self._session_baseline is None:
            return False
        return differs(ns, self._session_baseline)

    def is_dirty(self, ns, key=None):
        if key is not None:
            self.check_pending(ns, key)
        return bool(self.dirty or self.session_differs(ns))


def _same_data(a, b):
    if a.keys() != b.keys():
        return False
    return all(same(a[k], b[k]) for k in a)
