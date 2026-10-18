"""Headless tests for the pure parts of BioRAG Annotate (biorag_annotate_v1.py, biorag_annotate_tokens_v1.py).
Run:  <python of the GUI env> test_brush/test_biorag_annotate.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import biorag_annotate_v1 as ba      # noqa: E402
import biorag_annotate_tokens_v1 as bat  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    info = str(info)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info and not cond else ""))


# ---- token high-pass
G = np.random.default_rng(0).normal(size=(6, 7, 5)).astype(np.float32)
H = ba.highpass_grid(G, 3)
ref = np.zeros_like(G)
for y in range(6):
    for x in range(7):
        ref[y, x] = G[y, x] - G[max(0, y - 1):y + 2, max(0, x - 1):x + 2].mean((0, 1))
check("highpass = token minus 3x3 mean (border: existing patches only)", np.allclose(H, ref, atol=1e-5))
check("highpass of a constant grid is zero", np.allclose(ba.highpass_grid(np.ones((4, 4, 3)), 3), 0))
T = G.reshape(42, 5).astype(np.float16)
check("token_features raw / hp / both shapes",
      ba.token_features(T, (6, 7), "raw").shape == (42, 5) and ba.token_features(T, (6, 7), "high-pass").shape == (42, 5)
      and ba.token_features(T, (6, 7), "raw+high-pass").shape == (42, 10))

# ---- PCA
rng = np.random.default_rng(1)
X = rng.normal(size=(2000, 8)).astype(np.float32) * np.array([10, 5, 2, 1, 1, 1, 1, 1], np.float32)
mu, V = ba.fit_pca(X)
check("PCA recovers the main axes in order", np.allclose(np.abs(V[:3, :3]), np.eye(3), atol=0.05))
U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
check("PCA = SVD of centred data (up to sign)", np.allclose(np.abs(Vt[:3].T), np.abs(V), atol=1e-3))
P = ba.project(X, mu, V)
rgb = ba.minmax_rgb(P)
check("PCA->RGB min-max in [0,1], extremes hit", rgb.min() == 0 and rgb.max() == 1 and np.allclose(rgb.min(0), 0)
      and np.allclose(rgb.max(0), 1))

# ---- Otsu foreground
v = np.r_[rng.normal(-3, 0.5, 700), rng.normal(3, 0.5, 300)]
t = ba.otsu_threshold(v)
check("Otsu threshold separates two modes", (v > t).sum() == 300 and -3 < t < 3, t)
grid = (10, 10)
pc1 = np.full(100, -3.0)
pc1.reshape(10, 10)[3:7, 3:7] = 3.0
fg, _ = ba.foreground_from_pc1(pc1, ba.border_index(grid))
check("foreground = side without the border patches", fg.sum() == 16 and fg.reshape(10, 10)[5, 5])
fg2, _ = ba.foreground_from_pc1(-pc1, ba.border_index(grid))
check("foreground independent of PC1 sign", fg2.sum() == 16)
fg3, _ = ba.foreground_from_pc1(pc1, ba.border_index(grid), flip=True)
check("flip swaps fg / bg", fg3.sum() == 84)

# ---- working size / resolution rule
check("small image is UPSAMPLED to the long side", bat.working_size(500, 300, 1536) == (1536, 928))
check("large image downsampled, multiples of 16", all(x % 16 == 0 for x in bat.working_size(3586, 5389, 1536))
      and bat.working_size(3586, 5389, 1536)[1] == 1536)

# ---- patch selection -> mask at original size
Hh, Ww = 300, 450
work = bat.working_size(Hh, Ww, 1024)        # (688?, 1024)
gh, gw = work[0] // 16, work[1] // 16
img = np.zeros((Hh, Ww, 3), np.uint8)
cv2.circle(img, (225, 150), 80, (255, 255, 255), -1)
grey = ba.grey_working_image(img, work)
yy, xx = np.mgrid[0:gh, 0:gw]
cy, cx = (yy + 0.5) * Hh / gh, (xx + 0.5) * Ww / gw
sel = (cx - 225) ** 2 + (cy - 150) ** 2 < 80 ** 2
m = ba.selection_to_mask(sel, (Hh, Ww), grey)
truth = (img[..., 0] > 0)
iou = (m.astype(bool) & truth).sum() / (m.astype(bool) | truth).sum()
m_block = ba.selection_to_mask(sel, (Hh, Ww), None)
iou_b = (m_block.astype(bool) & truth).sum() / (m_block.astype(bool) | truth).sum()
check("mask has the ORIGINAL image size, 0/1 uint8", m.shape == (Hh, Ww) and m.dtype == np.uint8 and set(np.unique(m)) <= {0, 1})
check("guided filter follows the edge (IoU > 0.97 and >= blocks)", iou > 0.97 and iou >= iou_b - 1e-9, f"{iou:.4f} vs {iou_b:.4f}")
sel2 = sel.copy()
sel2[0, 0] = True                               # one stray patch in the dark corner
m2 = ba.selection_to_mask(sel2, (Hh, Ww), None, min_frac=0.001)
check("tiny components (< 0.1% of image) are removed", m2[:3, :3].sum() == 0)
check("empty selection -> empty mask", ba.selection_to_mask(np.zeros((gh, gw), bool), (Hh, Ww), grey).sum() == 0)

# ---- RLE / polygons
mm = np.zeros((37, 53), np.uint8)
mm[5:20, 10:30] = 1
mm[0, 0] = 1
mm[-1, -1] = 1
r = ba.mask_to_rle(mm)
check("uncompressed RLE round trip", np.array_equal(ba.rle_to_mask(r), mm))
r0 = ba.mask_to_rle(np.zeros((4, 5), np.uint8))
check("RLE of an empty mask", r0["counts"] == [20] and ba.rle_to_mask(r0).sum() == 0)
try:
    from pycocotools import mask as mu_
    enc = mu_.encode(np.asfortranarray(mm))
    s = enc["counts"].decode("ascii")
    check("compressed RLE string decoded like pycocotools", np.array_equal(ba.rle_to_mask({"size": enc["size"], "counts": s}), mm))
    check("our RLE counts equal pycocotools counts", ba.rle_string_to_counts(s) == r["counts"])
except ImportError:
    print("SKIP pycocotools not installed")
sq = np.zeros((50, 60), np.uint8)
sq[10:30, 20:40] = 1
pl = ba.mask_to_polygons(sq)
back = ba.segmentation_to_mask(pl, (50, 60))
check("polygon round trip (square)", (back.astype(bool) ^ sq.astype(bool)).sum() <= 40)
check("string-encoded polygons", np.array_equal(ba.segmentation_to_mask(json.dumps(pl), (50, 60)), back))
check("flat polygon", np.array_equal(ba.segmentation_to_mask(pl[0], (50, 60)), back))

# ---- category grid (smallest mask wins)
big = np.zeros((100, 100), np.uint8); big[10:90, 10:90] = 1
small = np.zeros((100, 100), np.uint8); small[40:60, 40:60] = 1
anns = [{"category_id": 7, "segmentation": ba.mask_to_rle(big)},
        {"category_id": 9, "segmentation": ba.mask_to_polygons(small)}]
cg = ba.category_grid(anns, {7: 0, 9: 1}, (100, 100), (10, 10))
check("category per patch centre; smallest mask wins; none = -1",
      cg[5, 5] == 1 and cg[2, 2] == 0 and cg[0, 0] == -1 and (cg == 1).sum() == 4, str(cg))

# ---- COCO writing + merging by category name
m_a = np.zeros((40, 60), np.uint8); m_a[5:15, 5:25] = 1
m_b = np.zeros((30, 30), np.uint8); m_b[2:12, 2:12] = 1
asg = [{"category": "femur", "params": {"layer": 18}, "masks": {"a.tif": ba.mask_to_rle(m_a), "b.jpg": ba.mask_to_rle(m_b)}},
       {"category": "tibia", "params": {"layer": 18}, "masks": {"a.tif": ba.mask_to_rle(np.pad(m_b, ((0, 10), (0, 30))))}}]
poly, rle = ba.build_coco(asg, {"a.tif": (40, 60), "b.jpg": (30, 30)}, {"femur": 4})
check("COCO: editor category id reused, new name gets a new id",
      {c["name"]: c["id"] for c in poly["categories"]} == {"femur": 4, "tibia": 5})
check("COCO: 3 annotations, source + params on each",
      len(poly["annotations"]) == 3 and all(a["source"] == "biorag_annotate" and a["biorag_annotate"]["layer"] == 18
                                           for a in poly["annotations"]))
check("COCO: polygon file has polygons, rle file has RLE",
      isinstance(poly["annotations"][0]["segmentation"], list) and isinstance(rle["annotations"][0]["segmentation"], dict))
check("COCO: area/bbox right", poly["annotations"][0]["area"] == 200 and poly["annotations"][0]["bbox"] == [5, 5, 20, 10])
base = {"images": [{"id": 1, "file_name": "a.tif", "height": 40, "width": 60}],
        "annotations": [{"id": 10, "image_id": 1, "category_id": 2, "segmentation": [[0, 0, 5, 0, 5, 5]], "area": 1, "bbox": [0, 0, 5, 5], "iscrowd": 0}],
        "categories": [{"id": 2, "name": "tibia"}]}
base_copy = json.loads(json.dumps(base))
mg = ba.merge_coco(base, poly)
check("merge: base untouched", base == base_copy)
check("merge: old annotation kept as is", mg["annotations"][0] == base["annotations"][0])
cn = {c["name"]: c["id"] for c in mg["categories"]}
check("merge: categories matched by NAME (tibia keeps id 2), femur added", cn == {"tibia": 2, "femur": 3})
new = mg["annotations"][1:]
check("merge: new ids unique and above old", len({a["id"] for a in mg["annotations"]}) == 4 and min(a["id"] for a in new) == 11)
names = {i["id"]: i["file_name"] for i in mg["images"]}
check("merge: images matched by file name; missing image added",
      sorted(names.values()) == ["a.tif", "b.jpg"] and all(names[a["image_id"]] in ("a.tif", "b.jpg") for a in new))
tib = [a for a in new if a["category_id"] == 2]
check("merge: tibia annotation points to image a.tif and category 2", len(tib) == 1 and names[tib[0]["image_id"]] == "a.tif")
per = ba.coco_annotations_by_image({"images": [{"id": "x/y/a", "file_name": "a.tif"}],
                                    "annotations": [{"image_id": "x/y/a", "category_id": 1}]}, ["/data/a.tif", "/d/c.png"])
check("annotations by image: string image ids, missing images empty", len(per["a.tif"]) == 1 and per["c.png"] == [])

# ---- cache round trip + validity
with tempfile.TemporaryDirectory() as d:
    imp = os.path.join(d, "img one.png")
    cv2.imwrite(imp, np.zeros((20, 30, 3), np.uint8))
    cp = bat.cache_path(d, imp, 18, 1024)
    Tt = np.random.default_rng(3).normal(size=(12, 1024)).astype(np.float16)
    bat.save_cache(cp, Tt, (3, 4), (20, 30), (48, 64), 18, 1024, bat.DEFAULT_MODEL, imp)
    z = bat.load_cache(cp)
    check("cache round trip (tokens float16, grid, sizes, scale)",
          z["tokens"].dtype == np.float16 and np.array_equal(z["tokens"], Tt) and tuple(z["grid"]) == (3, 4)
          and tuple(z["image_hw"]) == (20, 30) and np.allclose(z["scale"], [48 / 20, 64 / 30]) and int(z["layer"]) == 18)
    check("cache valid for same settings", bat.cache_is_valid(cp, imp, 18, 1024, bat.DEFAULT_MODEL))
    check("cache invalid for another layer / model", not bat.cache_is_valid(cp, imp, 12, 1024, bat.DEFAULT_MODEL)
          and not bat.cache_is_valid(cp, imp, 18, 1024, "other/model"))
    os.utime(imp, (1, 1))
    check("cache invalid after the image changed", not bat.cache_is_valid(cp, imp, 18, 1024, bat.DEFAULT_MODEL))
    check("cache name keeps the extension", os.path.basename(cp) == "img one.png__L18_s1024.npz")
    rec = ba.ImageRec(imp, z)
    check("ImageRec reads the cache", rec.n == 12 and rec.grid == (3, 4) and rec.image_hw == (20, 30))
    # image loading: RGBA, 16-bit grey
    from PIL import Image
    p4 = os.path.join(d, "rgba.png")
    Image.fromarray(np.dstack([np.full((5, 6, 3), 100, np.uint8), np.zeros((5, 6), np.uint8)]), "RGBA").save(p4)
    a4 = bat.load_rgb8(p4)
    check("RGBA -> RGB (alpha dropped, as the GUI)", a4.shape == (5, 6, 3) and a4.dtype == np.uint8 and a4[0, 0, 0] == 100)
    p16 = os.path.join(d, "g16.tif")
    g16 = (np.arange(30).reshape(5, 6) * 2000).astype(np.uint16)
    Image.fromarray(g16).save(p16)
    a16 = bat.load_rgb8(p16)
    check("16-bit grey tif -> 8-bit RGB scaled by max", a16.shape == (5, 6, 3) and a16.max() == 255 and a16[0, 0, 0] == 0)
    p48 = os.path.join(d, "rgb16.tif")
    cv2.imwrite(p48, (np.random.default_rng(4).integers(0, 65535, (5, 6, 3))).astype(np.uint16))
    a48 = bat.load_rgb8(p48)
    check("16-bit RGB tif -> 8-bit RGB", a48.shape == (5, 6, 3) and a48.dtype == np.uint8)

# ---- crops / zoomed region
check("crop working size: small crop upsampled but at most 8x", bat.crop_working_size(100, 60, 2048) == (800, 480))
check("crop working size: larger crop -> long side = size", bat.crop_working_size(600, 300, 2048)[0] == 2048)
check("list line with crop box", bat.parse_list_line("/a b/c.tif\t10,20,110,220\n") == ("/a b/c.tif", (10, 20, 110, 220))
      and bat.parse_list_line("/x.png") == ("/x.png", None))
check("crop cache name differs from whole image", bat.cache_path("/c", "/i/a.tif", 18, 2048, (1, 2, 3, 4)) != bat.cache_path("/c", "/i/a.tif", 18, 2048))
box = (100, 50, 300, 250)                      # crop 200 x 200 in a 400 x 500 (H x W) image
selc = np.zeros((10, 10), bool); selc[2:5, 3:7] = True
mc = ba.selection_to_mask(selc, (400, 500), None, box=box, min_frac=0)
ys, xs = np.nonzero(mc)
check("crop selection -> mask maps back into the image (box offset, crop scale)",
      mc.shape == (400, 500) and xs.min() == 100 + 60 and xs.max() == 100 + 139 and ys.min() == 50 + 40 and ys.max() == 50 + 99,
      (xs.min(), xs.max(), ys.min(), ys.max()))
gw_ = ba.grey_working_image(np.zeros((200, 200, 3), np.uint8), (160, 160))
mg = ba.selection_to_mask(selc, (400, 500), gw_, box=box, min_frac=0)
check("crop mask with guided filter stays inside the box", mg[:50].sum() == 0 and mg[:, :100].sum() == 0 and mg.sum() > 0)
check("paste_mask", ba.paste_mask(np.ones((2, 3)), (5, 6, 8, 8), (10, 10)).sum() == 6)
cx, cy = ba.patch_centres((10, 10), box)
check("patch centres in a crop", np.isclose(cx[0], 110) and np.isclose(cy[-1], 240))
check("point -> patch (inside / outside the box)", ba.point_to_patch(111, 51, (10, 10), box) == (0, 0)
      and ba.point_to_patch(299, 249, (10, 10), box) == (9, 9) and ba.point_to_patch(10, 10, (10, 10), box) is None)
cgc = ba.category_grid([{"category_id": 1, "segmentation": ba.mask_to_rle(ba.paste_mask(np.ones((100, 100)), (100, 50, 200, 150), (400, 500)))}],
                       {1: 0}, (400, 500), (10, 10), box)
check("category grid of a crop uses the crop's patch centres", (cgc == 0).sum() == 25 and cgc[0, 0] == 0 and cgc[9, 9] == -1)

# ---- brush pixel masks
pm = np.zeros((32, 48), np.uint8); pm[0:16, 0:8] = 1; pm[16:32, 32:48] = 1
ps = ba.pix_to_patch_sel(pm, (2, 3))
check("pixel mask -> patch selection at >= 50% of a patch", ps.tolist() == [[True, False, False], [False, False, True]], ps)
wm = ba.selection_to_work_mask(selc, np.zeros((160, 160), np.float32) + 0.5, min_frac=0)
check("selection -> working-size pixel mask (brush start)", wm.shape == (160, 160) and wm.dtype == np.uint8 and wm[50, 60] == 1 and wm[0, 0] == 0)
wi = ba.work_mask_to_image(wm, (400, 500), box)
check("brush mask -> image as is (box offset)", wi.shape == (400, 500) and wi[50 + 63, 100 + 75] == 1 and wi[:50].sum() == 0)
check("brush mask whole image", ba.work_mask_to_image(np.ones((16, 16), np.uint8), (40, 30)).sum() == 1200)

# ---- similarity
Fs = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [-1, 0, 0]], np.float32)
sims = ba.cosine_to(ba.normalize_rows(Fs), np.array([2, 0, 0], np.float32))
check("cosine similarity", np.isclose(sims[0], 1) and sims[1] > 0.99 and np.isclose(sims[2], 0) and np.isclose(sims[3], -1))
check("normalize rows (zero row safe)", np.allclose(np.linalg.norm(ba.normalize_rows(np.array([[3, 4], [0, 0]])), axis=1), [1, 0]))

# ---- zoom / pan
zb = ba.zoom_box((0, 100, 0, 50), 25, 10, 0.5, (100, 50))
check("zoom about the cursor keeps the cursor point fixed", np.allclose(zb, (12.5, 62.5, 5, 30)), zb)
check("zoom out never exceeds the image", np.allclose(ba.zoom_box((10, 60, 5, 30), 30, 20, 4, (100, 50)), (0, 100, 0, 50)))
check("pan clamps to the image", np.allclose(ba.pan_box((50, 80, 10, 20), 100, -50, (100, 50)), (70, 100, 0, 10)))

# ---- colour window (vertical rainbow), 18 Oct
_rgb = np.array([[0, 0, 1], [0, 0, 0.4], [1, 0, 0], [0.98, 0, 0.1], [0, 1, 0]], np.float32)
_h, _s, _v = ba.rgb_to_hsv(_rgb)
check("hsv hue in degrees (blue 240, red 0, green 120)", np.allclose(_h[[0, 2, 4]], [240, 0, 120], atol=1), _h)
check("hue window selects blues, both bright and dark", list(ba.hue_window_mask(_h, _v, 240, 20)) == [True, True, False, False, False])
check("brightness range keeps only dark blue", list(ba.hue_window_mask(_h, _v, 240, 20, 0.0, 0.6)) == [False, True, False, False, False])
check("hue window wraps around red (0/360)", list(ba.hue_window_mask(_h, _v, 355, 20)) == [False, False, True, True, False])

_a = np.zeros((5, 7), bool); _a[2, 1] = True
_b = np.zeros((5, 7), bool); _b[2, 2] = _b[2, 3] = _b[2, 4] = True; _b[0, 6] = True
check("colour 2 near colour 1: k=1 keeps only the touching patch", np.argwhere(ba.grow_near(_a, _b, 1)).tolist() == [[2, 2]])
check("colour 2 near colour 1: k=3 follows the chain, not the far patch", np.argwhere(ba.grow_near(_a, _b, 3)).tolist() == [[2, 2], [2, 3], [2, 4]])
check("colour 2 near colour 1: nothing near -> nothing", not ba.grow_near(np.zeros((5, 7), bool), _b, 3).any())

# ---- selection rules (18 Oct): apply a zoomed-region selection elsewhere
def _old_pca(items, variant, fg, flip):
    """The window's PCA as it was before it moved into pca_fit_tokens (copied), to prove nothing changed."""
    N = sum(g[0] * g[1] for _, g in items)
    offs = np.cumsum([0] + [g[0] * g[1] for _, g in items])
    rng = np.random.default_rng(0)
    frac = min(1.0, ba.MAX_FIT / max(1, N))
    samp = [np.sort(rng.choice(g[0] * g[1], size=max(1, int(round(g[0] * g[1] * frac))), replace=False)) for _, g in items]
    def fp(keep):
        Xs = []
        for (T, g), s, o in zip(items, samp, offs):
            F = ba.token_features(T, g, variant)
            Xs.append(F[s if keep is None else s[keep[o + s]]])
        mu, V = ba.fit_pca(np.concatenate(Xs, 0))
        P = np.zeros((N, 3), np.float32)
        for (T, g), o in zip(items, offs):
            P[o:o + g[0] * g[1]] = ba.project(ba.token_features(T, g, variant), mu, V)
        return P
    P = fp(None)
    vis = np.ones(N, bool)
    if fg:
        bidx = np.concatenate([o + ba.border_index(g) for (_, g), o in zip(items, offs)])
        hi = P[:, 0] > ba.otsu_threshold(P[:, 0])
        vis = ~hi if hi[bidx].mean() > 0.5 else hi
        if flip:
            vis = ~vis
        if vis.sum() >= 10:
            P = fp(vis)
    return P, ba.minmax_rgb(P, P[vis]), vis

_rng = np.random.default_rng(7)
def _crop_tokens(gh, gw, D=12):
    T = _rng.normal(size=(gh, gw, D)).astype(np.float32)
    T[1:gh - 1, 1:gw - 1, :3] += 4.0                     # an "object" away from the border
    T[2:4, 2:5, 3:6] += 3.0                              # a "structure" on it
    return T.reshape(gh * gw, D).astype(np.float16)
_items = [(_crop_tokens(9, 11), (9, 11)), (_crop_tokens(8, 10), (8, 10))]
_grids = [(0, (9, 11)), (99, (8, 10))]
for _variant in ba.VARIANTS:
    for _fg, _flip in ((False, False), (True, False), (True, True)):
        P0, rgb0, vis0 = _old_pca(_items, _variant, _fg, _flip)
        P1, rgb1, vis1, prm = ba.pca_fit_tokens(_items, _variant, _fg, _flip)
        same = np.array_equal(P0, P1) and np.array_equal(rgb0, rgb1) and np.array_equal(vis0, vis1)
        hue, _, _ = ba.rgb_to_hsv(rgb1)
        c1 = float(np.median(hue[vis1]))
        wins = [{"on": True, "c": c1, "w": 60, "lo": 0, "hi": 100},
                {"on": True, "c": (c1 + 90) % 360, "w": 80, "lo": 0, "hi": 100, "near": 2}]
        new, _, _ = ba.colour_window_selection(rgb1, _grids, wins)
        crop_sel = new & vis1                              # what the window shows ("replace")
        rule = {"kind": "colour windows", "variant": _variant, "fg": prm["fg"], "mu1": prm["mu1"], "V1": prm["V1"],
                "mu": prm["mu"], "V": prm["V"], "lo": prm["lo"], "hi": prm["hi"], "windows": wins, "layer": 18,
                "scale": [1.5, 1.5]}
        if _fg:
            rule.update({"t": prm["t"], "fg_hi": prm["fg_hi"]})
        with tempfile.TemporaryDirectory() as d:
            rp = os.path.join(d, "rule.npz")
            rid = ba.save_rule(rp, rule)
            rule2 = ba.load_rule(rp)
        got = np.concatenate([ba.rule_scores(T, g, rule2)[0] for T, g in _items])
        check(f"colour rule on the crop's own tokens reproduces the crop selection exactly ({_variant}, fg={_fg}, flip={_flip})",
              same and np.array_equal(got, crop_sel) and 0 < crop_sel.sum() < len(crop_sel),
              f"pca same {same}, {int(got.sum())} vs {int(crop_sel.sum())}")
check("rule id is stable for the same rule", ba.save_rule(os.path.join(tempfile.mkdtemp(), "r.npz"), rule) == rid)
# similar rule
P1, rgb1, vis1, prm = ba.pca_fit_tokens(_items, "raw+high-pass", True, False)
Fn = [ba.normalize_rows(ba.token_features(T, g, "raw+high-pass")) for T, g in _items]
q = Fn[0][3 * 11 + 3]
win_sel = np.concatenate([ba.cosine_to(F, q) >= 0.6 for F in Fn]) & vis1
rule_s = {"kind": "similar", "variant": "raw+high-pass", "fg": True, "mu1": prm["mu1"], "V1": prm["V1"], "t": prm["t"],
          "fg_hi": prm["fg_hi"], "proto": q, "threshold": 0.6}
got = np.concatenate([ba.rule_scores(T, g, rule_s)[0] for T, g in _items])
check("similar rule reproduces the in-window similarity selection", np.array_equal(got, win_sel) and win_sel.sum() > 1)
# learned rule
Xl = np.concatenate([ba.token_features(T, g, "raw") for T, g in _items])
yl = np.zeros(len(Xl), bool)
yl[[3 * 11 + 3, 3 * 11 + 4, 2 * 11 + 2, 2 * 11 + 3]] = True                     # 4 positives (rare class)
m = ba.train_logreg(Xl, yl)
pr = ba.logreg_proba(Xl, m)
check("logistic regression (balanced) separates a rare class", ((pr >= 0.5) == yl).mean() > 0.95 and pr[yl].min() >= 0.5,
      f"acc {((pr >= 0.5) == yl).mean():.3f}, min pos {pr[yl].min():.3f}")
rule_l = {"kind": "learned", "variant": "raw", "fg": False, "mu1": prm["mu1"], "V1": prm["V1"], "threshold": 0.5,
          "lr_mean": m["mean"], "lr_std": m["std"], "lr_w": m["w"], "lr_b": m["b"]}
with tempfile.TemporaryDirectory() as d:
    ba.save_rule(os.path.join(d, "l.npz"), rule_l)
    rule_l2 = ba.load_rule(os.path.join(d, "l.npz"))
got = np.concatenate([ba.rule_scores(T, g, rule_l2)[0] for T, g in _items])
check("learned rule after save/load = the trained model's prediction", np.array_equal(got, pr >= 0.5))
check("rule summary has no arrays (JSON-able)", json.dumps(ba.rule_summary(rule_l)) is not None)
# colour-2 growth stays inside each image grid
rgbx = np.zeros((2 * 12, 3), np.float32)
rgbx[:] = (0.5, 0.5, 0.5)
rgbx[11] = (0, 0, 1)                                     # colour 1 at the last patch of grid A (3x4)
rgbx[12] = (1, 1, 0)                                     # colour 2 at the first patch of grid B (3x4)
rgbx[10] = (1, 1, 0)                                     # colour 2 next to colour 1 inside grid A
h1, _, _ = ba.rgb_to_hsv(rgbx[[11]]); h2, _, _ = ba.rgb_to_hsv(rgbx[[12]])
selx, m1x, m2x = ba.colour_window_selection(rgbx, [(0, (3, 4)), (12, (3, 4))],
                                            [{"on": True, "c": float(h1[0]), "w": 10, "lo": 0, "hi": 100},
                                             {"on": True, "c": float(h2[0]), "w": 10, "lo": 0, "hi": 100, "near": 1}])
check("colour 2 grows only inside its own image grid", np.flatnonzero(selx).tolist() == [10, 11])
# fine grids
big_a = np.zeros((325, 488), bool); big_a[100, 200] = True
big_b = np.zeros((325, 488), bool); big_b[100, 201:210] = True; big_b[300, 400] = True
gn = ba.grow_near(big_a, big_b, 9)
check("grow_near on a fine (325 x 488) grid follows the chain only", gn.sum() == 9 and not gn[300, 400])
check("fine working size = whole image at the region's scale", bat.fine_working_size(3586, 5389, 1.4481, 1.4486) == (5200, 7808))
check("applied result cache name", os.path.basename(bat.applied_cache_path("/c", "/i/a b.tif", "abc123")) == "a b.tif__applied_abc123.npz")

check("tiles laid like the region: one tile starts at the anchor, partial tiles at the ends",
      bat.tile_edges(100 * 16, 30 * 16, 45 * 16) == [(0, 240), (240, 720), (720, 1200), (1200, 1600)])
check("tiles without anchor start at 0", bat.tile_edges(64, 32, 0) == [(0, 32), (32, 64)])

check("list line: optional per-region size", bat.parse_list_size("/a.tif\t1,2,3,4\t2961\n") == 2961
      and bat.parse_list_size("/a.tif\t1,2,3,4") is None and bat.parse_list_line("/a.tif\t1,2,3,4\t2961") == ("/a.tif", (1, 2, 3, 4)))

_sp = np.zeros((40, 40), bool); _sp[5, 5] = True; _sp[20:30, 20:30] = True       # one lone patch + a block
_m_frac = ba.selection_to_mask(_sp, (640, 640), None)                             # default: 0.1% of the image
_m_px = ba.selection_to_mask(_sp, (640, 640), None, min_px=256)                   # 1 patch (16 x 16 px here)
check("min_px = 1 patch keeps a lone selected patch (setae), 0.1% rule drops it", _m_px[5 * 16 + 8, 5 * 16 + 8] == 1 and _m_frac[5 * 16 + 8, 5 * 16 + 8] == 0)
check("brush start with min_frac=0 keeps a lone patch", ba.selection_to_work_mask(_sp, np.full((640, 640), 0.5, np.float32), min_frac=0)[88, 88] == 1)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
