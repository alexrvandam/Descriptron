"""
Tests for remove_keypoint and rename_keypoint logic.
Runs fully headless — no GUI, no imports from the main script.
"""
import sys

# ── Replicate only what we need ──────────────────────────────────────────────
point_coords    = []
point_labels    = []   # 1=positive, -1=negative (SAM click type)
point_orders    = []   # display order / COCO array position
keypoint_labels = []   # parallel display labels (kept == point_orders after fixes)

def _redraw_points():  pass  # stub
def _update_superscripts(): pass  # stub

def remove_keypoint(index):
    global point_coords, keypoint_labels, point_labels, point_orders
    total = len(point_coords)
    if len(keypoint_labels) != total:
        keypoint_labels[:] = list(range(1, total + 1))
    if index < 0 or index >= total:
        return
    del point_coords[index]
    if index < len(point_labels):
        del point_labels[index]
    if index < len(point_orders):
        del point_orders[index]
    if index < len(keypoint_labels):
        del keypoint_labels[index]
    keypoint_labels[:] = list(range(1, len(keypoint_labels) + 1))
    point_orders[:]    = list(range(1, len(point_orders)    + 1))
    _redraw_points()
    _update_superscripts()


def rename_keypoint(index):
    global keypoint_labels, point_coords, point_labels, point_orders
    total = len(point_coords)
    if len(keypoint_labels) != total:
        keypoint_labels[:] = list(range(1, total + 1))
    if index < 0 or index >= total:
        return False, "invalid index"
    # (In the real app the user types into a dialog; here caller sets new_val directly)
    return True, None   # placeholder; actual reorder done by rename_keypoint_to below

def rename_keypoint_to(index, new_val):
    """Move keypoint at array-index `index` to 1-based position `new_val`."""
    global keypoint_labels, point_coords, point_labels, point_orders
    total = len(point_coords)
    if len(keypoint_labels) != total:
        keypoint_labels[:] = list(range(1, total + 1))
    if index < 0 or index >= total:
        return False, "invalid index"
    if new_val < 1 or new_val > total:
        return False, f"out of range (1–{total})"
    if new_val == keypoint_labels[index]:
        return True, "no-op"

    # Reorder all parallel lists the same way
    def _move(lst, idx, target_1based):
        if idx < len(lst):
            item = lst.pop(idx)
            lst.insert(target_1based - 1, item)

    _move(keypoint_labels, index, new_val)
    keypoint_labels[:] = list(range(1, len(keypoint_labels) + 1))

    _move(point_coords, index, new_val)
    _move(point_labels, index, new_val)
    if index < len(point_orders):
        _move(point_orders, index, new_val)
        point_orders[:] = list(range(1, len(point_orders) + 1))

    _redraw_points()
    _update_superscripts()
    return True, None


# ── Helpers ──────────────────────────────────────────────────────────────────
def setup(n):
    """Initialise n keypoints with coords (i,i), all positive, labels 1..n."""
    global point_coords, point_labels, point_orders, keypoint_labels
    point_coords    = [[i, i] for i in range(1, n + 1)]
    point_labels    = [1] * n
    point_orders    = list(range(1, n + 1))
    keypoint_labels = list(range(1, n + 1))

passed = failed = 0

def check(condition, name, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
        failed += 1


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== remove_keypoint tests ===")

# T1: Remove from middle — numbers must be sequential with no gap
setup(10)
remove_keypoint(6)   # remove #7 (0-based index 6)
check(len(point_coords) == 9, "T1a: 9 coords remain")
check(len(keypoint_labels) == 9, "T1b: keypoint_labels length = 9")
check(len(point_orders) == 9, "T1c: point_orders length = 9")
check(len(point_labels) == 9, "T1d: point_labels length = 9")
check(keypoint_labels == list(range(1, 10)), "T1e: keypoint_labels sequential 1–9",
      str(keypoint_labels))
check(point_orders == list(range(1, 10)), "T1f: point_orders sequential 1–9",
      str(point_orders))
# The coord that was at index 7 (value [8,8]) is now at index 6
check(point_coords[6] == [8, 8], "T1g: coord after removed slot is the next original",
      str(point_coords[6]))

# T2: Remove first keypoint
setup(5)
remove_keypoint(0)
check(keypoint_labels == [1, 2, 3, 4], "T2a: remove first → labels 1–4", str(keypoint_labels))
check(point_coords[0] == [2, 2], "T2b: coord[0] is old coord[1]")

# T3: Remove last keypoint
setup(5)
remove_keypoint(4)
check(keypoint_labels == [1, 2, 3, 4], "T3a: remove last → labels 1–4", str(keypoint_labels))
check(point_coords[-1] == [4, 4], "T3b: last coord is old 4th")

# T4: Remove with empty / out-of-sync keypoint_labels (simulates the bug)
setup(5)
keypoint_labels.clear()   # simulate keypoint_labels never initialised
remove_keypoint(2)
check(len(point_coords) == 4, "T4a: coord removed despite empty keypoint_labels")
check(keypoint_labels == [1, 2, 3, 4], "T4b: keypoint_labels rebuilt correctly",
      str(keypoint_labels))

# T5: Remove all one by one → always sequential
setup(4)
for _ in range(4):
    remove_keypoint(0)
check(point_coords == [], "T5a: all removed")
check(keypoint_labels == [], "T5b: labels empty")
check(point_orders == [], "T5c: orders empty")


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== rename_keypoint (move-to-position) tests ===")

# T6: Move keypoint #8 to position 11 (with 25 keypoints)
setup(25)
original_coord_8  = point_coords[7][:]   # coord of original #8
original_coord_11 = point_coords[10][:]  # coord of original #11
ok, err = rename_keypoint_to(7, 11)      # index 7 = keypoint #8
check(ok, "T6a: rename succeeded", err)
check(len(point_coords) == 25, "T6b: still 25 coords")
check(keypoint_labels == list(range(1, 26)), "T6c: labels sequential 1–25",
      str(keypoint_labels[:15]))
# Old #8 should now be at index 10 (position 11)
check(point_coords[10] == original_coord_8,
      "T6d: old coord #8 is now at array position 11",
      f"got {point_coords[10]}, expected {original_coord_8}")
# Old #9 should now be at index 7 (position 8) — shifted left
# After removing index 7 and inserting at 10, indices 7–9 shift:
# old 9→idx8, old 10→idx9, old 11→idx10 then #8 inserted, so old 9 is now at idx 7
check(point_orders == list(range(1, 26)), "T6e: point_orders sequential 1–25",
      str(point_orders[:15]))

# T7: Move first to last
setup(5)
coord_1 = point_coords[0][:]
rename_keypoint_to(0, 5)
check(point_coords[-1] == coord_1, "T7a: first coord moved to last position",
      str(point_coords))
check(keypoint_labels == [1, 2, 3, 4, 5], "T7b: labels still 1–5")

# T8: Move last to first
setup(5)
coord_5 = point_coords[4][:]
rename_keypoint_to(4, 1)
check(point_coords[0] == coord_5, "T8a: last coord moved to first position",
      str(point_coords))
check(keypoint_labels == [1, 2, 3, 4, 5], "T8b: labels still 1–5")

# T9: No-op (rename to same position)
setup(5)
orig = [c[:] for c in point_coords]
rename_keypoint_to(2, 3)   # keypoint #3 stays at position 3
check(point_coords == orig, "T9: no-op leaves coords unchanged")

# T10: Out-of-range rejected
setup(5)
ok, _ = rename_keypoint_to(0, 0)
check(not ok, "T10a: position 0 rejected")
ok, _ = rename_keypoint_to(0, 6)
check(not ok, "T10b: position 6 rejected (only 5 keypoints)")

# T11: Remove then rename — combined workflow
setup(10)
remove_keypoint(6)          # remove #7 → 9 keypoints remain, renumbered 1–9
ok, _ = rename_keypoint_to(3, 7)   # move #4 to position 7
check(ok, "T11a: rename after remove succeeded")
check(len(point_coords) == 9, "T11b: still 9 coords")
check(keypoint_labels == list(range(1, 10)), "T11c: labels sequential 1–9",
      str(keypoint_labels))


# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
