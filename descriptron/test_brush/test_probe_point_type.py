"""Headless test of --point_type in dinomask/dinomask_hypercolumn_probe_v1.py (v98 point types).
read_clicks is taken VERBATIM from the probe (parsed with ast; the probe itself loads torch / DINOv3 at import) and run on
small GUI-style keypoint files. Without --point_type the result must equal the pre-change probe's (the .bak copy).
Run: <python of the GUI env> test_brush/test_probe_point_type.py"""
import ast, glob, json, os, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(os.path.dirname(HERE), "dinomask", "dinomask_hypercolumn_probe_v1.py")
OLD = sorted(glob.glob(PROBE + ".bak_pre_pointtype_*"))


def read_clicks_from(path):
    src = open(path).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "read_clicks")
    ns = {"np": np, "json": json, "os": os}
    exec(ast.get_source_segment(src, fn), ns)
    return ns["read_clicks"]


def ann(iid, pts, types=None):
    kp = []
    for x, y, v in pts:
        kp += [x, y, v]
    a = {"id": iid, "image_id": iid, "category_id": 1, "keypoints": [c if p[2] > 0 else 0 for p in pts for c in p],
         "keypoints_clicked": kp}
    if types is not None:
        a["keypoint_types"] = types
    return a


def main():
    d = tempfile.mkdtemp()
    for n in ("a.jpg", "b.jpg", "c.jpg"):
        open(os.path.join(d, n), "w").close()
    typed = {"images": [{"id": "a", "file_name": "a.jpg"}, {"id": "b", "file_name": "b.jpg"}],
             "annotations": [ann("a", [(10, 10, 2), (20, 20, 2), (30, 30, 0), (40, 40, 2), (50, 50, 0)],
                                 ["landmark", "seta", "seta", "pit", "pit"]),
                             ann("b", [(5, 5, 2), (6, 6, 2)], ["landmark", "landmark"])]}
    untyped = {"images": [{"id": "c", "file_name": "c.jpg"}],
               "annotations": [ann("c", [(7, 7, 2), (8, 8, 0)])]}
    f1, f2 = os.path.join(d, "typed.json"), os.path.join(d, "c_keypoints.json")
    json.dump(typed, open(f1, "w")); json.dump(untyped, open(f2, "w"))
    rc = read_clicks_from(PROBE)
    A, B, C = (os.path.join(d, x) for x in ("a.jpg", "b.jpg", "c.jpg"))
    ok = total = 0

    def check(cond, name):
        nonlocal ok, total
        total += 1; ok += bool(cond)
        print(("ok  " if cond else "FAIL") + " " + name)

    def pts(arr):
        return [tuple(map(float, r)) for r in arr]

    allc = rc([f1, f2], d)
    check(set(allc) == {A, B, C}, "no --point_type: every image used (typed and untyped files)")
    check(pts(allc[A][0]) == [(10, 10), (20, 20), (40, 40)] and pts(allc[A][1]) == [(30, 30), (50, 50)],
          "no --point_type: every click (all types) as before")
    s = rc([f1, f2], d, "seta")
    check(set(s) == {A}, "seta: only the image with seta clicks; untyped file not used")
    check(pts(s[A][0]) == [(20, 20)] and pts(s[A][1]) == [(30, 30)], "seta: its positive AND its negative")
    p = rc([f1, f2], d, "pit")
    check(pts(p[A][0]) == [(40, 40)] and pts(p[A][1]) == [(50, 50)], "pit")
    lm = rc([f1, f2], d, "landmark")
    check(set(lm) == {A, B} and pts(lm[B][0]) == [(5, 5), (6, 6)] and pts(lm[A][0]) == [(10, 10)],
          "landmark: typed files only")
    check(rc([f1, f2], d, "pore") == {}, "a type nobody clicked -> no clicks")
    bad = {"images": typed["images"][:1], "annotations": [dict(typed["annotations"][0], keypoint_types=["seta"])]}
    f3 = os.path.join(d, "bad.json"); json.dump(bad, open(f3, "w"))
    check(rc([f3], d, "seta") == {}, "malformed keypoint_types (wrong length) -> skipped, not misaligned")
    if OLD:
        old = read_clicks_from(OLD[-1])(([f1, f2]), d)
        same = set(old) == set(allc) and all(np.array_equal(old[k][0], allc[k][0]) and np.array_equal(old[k][1], allc[k][1])
                                              for k in old)
        check(same, "default identical to the pre-change probe (" + os.path.basename(OLD[-1]) + ")")
    src = open(PROBE).read()
    check("ap.add_argument('--point_type', default=''" in src and "read_clicks(click_files(a.points), a.images, a.point_type or None)" in src,
          "--point_type option wired (default empty = all clicks)")
    print(f"{ok} / {total} probe point-type checks passed")
    if ok != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
