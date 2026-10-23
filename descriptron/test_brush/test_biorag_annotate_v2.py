"""Headless tests for the v2 additions of BioRAG Annotate (biorag_annotate_v2.py, biorag_annotate_tokens_v2.py):
masked normalised Gaussian high-pass, percentile colours, PCA fitted on the specimen mask, v1 reproduction,
mask resampling, refine-on-zoom geometry, 'native' size.
Run:  <python of the GUI env> test_brush/test_biorag_annotate_v2.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import importlib  # noqa: E402
b2 = importlib.import_module("biorag_annotate_v2")
t2 = importlib.import_module("biorag_annotate_tokens_v2")
b1 = importlib.import_module("biorag_annotate_v1")
print("testing", os.path.basename(b2.__file__), "+", os.path.basename(t2.__file__), "(v1 for reference)")

PASS, FAIL = [], []


def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    info = str(info)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info and not cond else ""))


# ---- direct numpy reference of the masked normalised Gaussian high-pass
def np_gauss_blur(A, sigma):
    """Separable Gaussian, radius ceil(4 sigma), reflect-101 border (numpy 'reflect'), per channel."""
    r = int(np.ceil(4 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    A = np.asarray(A, np.float64)
    squeeze = A.ndim == 2
    if squeeze:
        A = A[..., None]
    P = np.pad(A, ((r, r), (0, 0), (0, 0)), mode="reflect")
    B = sum(k[i] * P[i:i + A.shape[0]] for i in range(2 * r + 1))
    P = np.pad(B, ((0, 0), (r, r), (0, 0)), mode="reflect")
    B = sum(k[i] * P[:, i:i + A.shape[1]] for i in range(2 * r + 1))
    return B[..., 0] if squeeze else B


def np_masked_hp(G, M, sigma):
    num = np_gauss_blur(G * M[..., None], sigma)
    den = np_gauss_blur(M.astype(np.float64), sigma)[..., None] + 1e-6
    return G - num / den


rng = np.random.default_rng(0)
G = rng.normal(size=(40, 50, 7)).astype(np.float32)
yy, xx = np.mgrid[:40, :50]
M = ((yy - 20) ** 2 / 15 ** 2 + (xx - 25) ** 2 / 20 ** 2) < 1
for s in (1.0, 2.0, 3.0):
    H = b2.highpass_gauss_grid(G, M, s)
    R = np_masked_hp(G.astype(np.float64), M.astype(np.float64), s)
    err = float(np.abs(H - R)[M].max())
    check(f"masked Gaussian high-pass (sigma {s:g}) = numpy reference inside the mask", err < 1e-4, err)
H0 = b2.highpass_gauss_grid(G, None, 3.0)
R0 = G - np_gauss_blur(G, 3.0)
check("unmasked Gaussian high-pass = G - Gaussian blur (numpy reference)", np.abs(H0 - R0).max() < 1e-4,
      np.abs(H0 - R0).max())
Gc = np.ones((30, 30, 4), np.float32) * 5
Gc[~M[:30, :30]] = 100.0                     # background very different: must not leak into the mask
Hc = b2.highpass_gauss_grid(Gc, M[:30, :30], 3.0)
check("masked high-pass of a field constant inside the mask is 0 there (background does not leak)",
      np.abs(Hc[M[:30, :30]]).max() < 1e-3, np.abs(Hc[M[:30, :30]]).max())
check("Gaussian kernel radius = ceil(4 sigma)", b2.gauss_ksize(3) == 25 and b2.gauss_ksize(1.5) == 13)
Gwide = rng.normal(size=(12, 10, 600)).astype(np.float32)   # > 256 channels -> chunked blur
check("channel-chunked blur (600 channels) = numpy reference",
      np.abs(b2.highpass_gauss_grid(Gwide, None, 2.0) - (Gwide - np_gauss_blur(Gwide, 2.0))).max() < 1e-4)
Gs = rng.normal(size=(5, 4, 3)).astype(np.float32)          # grid smaller than the kernel
check("tiny grid (smaller than the kernel) = numpy reference",
      np.abs(b2.highpass_gauss_grid(Gs, None, 3.0) - (Gs - np_gauss_blur(Gs, 3.0))).max() < 1e-4)

# ---- token_features: v2 Gaussian, v1 box kept
T = G.reshape(-1, 7).astype(np.float16)
F = b2.token_features(T, (40, 50), "high-pass", 3.0, M)
check("token_features 'high-pass' = masked Gaussian high-pass",
      np.allclose(F, b2.highpass_gauss_grid(T.astype(np.float32).reshape(40, 50, 7), M, 3.0).reshape(-1, 7)))
check("token_features flat mask = 2-D mask", np.allclose(F, b2.token_features(T, (40, 50), "high-pass", 3.0, M.ravel())))
check("token_features 'raw+high-pass' shape", b2.token_features(T, (40, 50), "raw+high-pass").shape == (2000, 14))
check("'high-pass (3x3 box, v1)' = v1 token_features 'high-pass'",
      np.array_equal(b2.token_features(T, (40, 50), "high-pass (3x3 box, v1)"), b1.token_features(T, (40, 50), "high-pass")))
check("'raw+high-pass (3x3 box, v1)' = v1 'raw+high-pass'",
      np.array_equal(b2.token_features(T, (40, 50), "raw+high-pass (3x3 box, v1)"),
                     b1.token_features(T, (40, 50), "raw+high-pass")))
check("raw unchanged", np.array_equal(b2.token_features(T, (40, 50), "raw"), b1.token_features(T, (40, 50), "raw")))
check("all v1 variant names still offered", all(v in b2.VARIANTS for v in b1.VARIANTS))

# ---- percentile colours
P = rng.normal(size=(5000, 3)).astype(np.float32)
P[:10] *= 50                                 # a few extreme patches
rgb, lo, hi = b2.scale_rgb(P, None, "2-98 percentile")
check("percentile scaling: lo/hi = 2nd / 98th percentile", np.allclose(lo, np.percentile(P, 2, 0), atol=1e-5)
      and np.allclose(hi, np.percentile(P, 98, 0), atol=1e-5))
frac0, frac1 = (rgb == 0).mean(0), (rgb == 1).mean(0)
check("percentile scaling: ~2% clipped at each end per PC", np.all(np.abs(frac0 - 0.02) < 0.003)
      and np.all(np.abs(frac1 - 0.02) < 0.003), (frac0, frac1))
mm = b1.minmax_rgb(P)
check("percentile scaling uses the range: median spread > 3x min-max's", np.std(rgb[:, 0]) > 3 * np.std(mm[:, 0]),
      (np.std(rgb[:, 0]), np.std(mm[:, 0])))
check("'min-max' option = v1 minmax_rgb", np.allclose(b2.scale_rgb(P, None, "min-max")[0], mm))
check("percentile_rgb = scale_rgb percentile", np.allclose(b2.percentile_rgb(P), rgb, atol=1e-6))
ref = P[100:]
check("percentile scaling over a reference subset", np.allclose(b2.scale_rgb(P, ref)[1], np.percentile(ref, 2, 0), atol=1e-5))

# ---- PCA fitted on the specimen mask
gh, gw, D = 30, 40, 6
Tm = rng.normal(size=(gh, gw, D)).astype(np.float32) * 0.1
spec = np.zeros((gh, gw), bool)
spec[8:22, 10:30] = True
Tm[~spec, 0] += rng.normal(size=(~spec).sum()) * 20      # background varies along dim 0
Tm[spec, 3] += rng.normal(size=spec.sum()) * 2           # specimen varies along dim 3
items = [(Tm.reshape(-1, D), (gh, gw))]
P1, rgb1, vis1, prm1 = b2.pca_fit_tokens_v2(items, "raw", masks=[spec.ravel()], fit_on_mask=True)
P0, rgb0, vis0, prm0 = b2.pca_fit_tokens_v2(items, "raw", masks=[spec.ravel()], fit_on_mask=False)
check("PCA on the mask: PC1 = the specimen's axis (dim 3)", np.argmax(np.abs(prm1["V"][:, 0])) == 3, prm1["V"][:, 0])
check("PCA on all tokens: PC1 = the background's axis (dim 0)", np.argmax(np.abs(prm0["V"][:, 0])) == 0)
check("PCA on the mask: mean = the specimen tokens' mean",
      np.allclose(prm1["mu"], Tm[spec].mean(0), atol=1e-4))
check("background patches dark grey, not PCA colours", np.allclose(rgb1[~spec.ravel()], b2.BG_GREY))
check("percentile colours over the specimen tokens", np.allclose(prm1["lo"], np.percentile(P1[spec.ravel()], 2, 0), atol=1e-4))
check("specimen fit: background neither plotted nor selectable (vis = mask)", np.array_equal(vis1, spec.ravel()))
check("no specimen fit, foreground only off: all patches visible", vis0.all())
_, _, vis2, _ = b2.pca_fit_tokens_v2(items, "raw", fg=True, masks=[spec.ravel()])
check("foreground only on: visible = the mask", np.array_equal(vis2, spec.ravel()))
_, _, _, prm3 = b2.pca_fit_tokens_v2(items, "high-pass", masks=[spec.ravel()], sigma=2.0)
check("prm records the Gaussian high-pass (kind, sigma, mask, fit)", prm3["hp_kind"] == "gauss" and prm3["hp_sigma"] == 2.0
      and prm3["hp_masked"] and prm3["pca_fit"] == "mask" and prm3["engine"] == "v2")
# the mask feeds the high-pass too: masked-feature PCA = PCA of masked features computed by hand
F_hp = b2.highpass_gauss_grid(Tm, spec, 2.0).reshape(-1, D)
mu_h, V_h = b2.fit_pca(F_hp[spec.ravel()])
check("high-pass PCA = fit_pca of masked high-pass features on the mask", np.allclose(np.abs(prm3["V"]), np.abs(V_h), atol=1e-4))

# ---- v1 results reproduce
Tv = rng.normal(size=(20 * 25, 8)).astype(np.float16)
for var1, var2 in (("high-pass", "high-pass (3x3 box, v1)"), ("raw", "raw"), ("raw+high-pass", "raw+high-pass (3x3 box, v1)")):
    for fg in (False, True):
        a = b1.pca_fit_tokens([(Tv, (20, 25))], var1, fg=fg)
        b = b2.pca_fit_tokens_v2([(Tv, (20, 25))], var2, fg=fg, masks=None, fit_on_mask=False, colour="min-max")
        check(f"v1 reproduces: {var2}, fg={fg} (no mask, no specimen fit, min-max)",
              np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]))

# ---- foreground per image from raw tokens
Tf = rng.normal(size=(24, 24, 5)).astype(np.float32) * 0.2
blob = np.zeros((24, 24), bool)
blob[6:18, 5:19] = True
Tf[blob] += 3.0
fgm = b2.foreground_grid(Tf.reshape(-1, 5), (24, 24))
check("foreground_grid finds the specimen (border side = background)", np.array_equal(fgm, blob.ravel()))
check("foreground_grid flip swaps", np.array_equal(b2.foreground_grid(Tf.reshape(-1, 5), (24, 24), True), ~blob.ravel()))

# ---- whole-image mask -> zoomed region grid
m = np.zeros((10, 20), bool)
m[2:6, 4:12] = True
check("resample_mask_grid: same grid + box = identity",
      np.array_equal(b2.resample_mask_grid(m, (10, 20), (0, 0, 200, 100), (10, 20), (0, 0, 200, 100)), m.ravel()))
r = b2.resample_mask_grid(m, (10, 20), (0, 0, 200, 100), (40, 40), (40, 20, 120, 60)).reshape(40, 40)
cx, cy = b2.patch_centres((40, 40), (40, 20, 120, 60))
expect = m[(cy // 10).astype(int)][:, (cx // 10).astype(int)]
check("resample_mask_grid: region grid takes the patch under its centre", np.array_equal(r, expect))
r2 = b2.resample_mask_grid(m, (10, 20), (0, 0, 200, 100), (4, 4), (180, 80, 260, 140)).reshape(4, 4)
check("resample_mask_grid: outside the source box = False", not r2.any())

# ---- refine-on-zoom geometry
check("patch_screen_px: 16-px patches at 2x = 32 screen px",
      abs(b2.patch_screen_px((0, 500, 0, 300), (1000, 600), (0, 0, 1600, 1600), (100, 100)) - 32.0) < 1e-6)
check("patch_screen_px: equal aspect uses the limiting axis",
      abs(b2.patch_screen_px((0, 500, 0, 600), (1000, 600), (0, 0, 1600, 1600), (100, 100)) - 16.0) < 1e-6)
check("region_patch_image_px: 512-px view at 2048 -> 4 px/patch", abs(b2.region_patch_image_px((0, 0, 512, 512), 2048) - 4.0) < 1e-6)
check("region_patch_image_px: capped at MAX_CROP_SCALE (8x)",
      abs(b2.region_patch_image_px((0, 0, 100, 100), 2048) - 100 / (t2.crop_working_size(100, 100, 2048)[1] / 16)) < 1e-6
      and t2.crop_working_size(100, 100, 2048)[1] == 800)
check("view_box clips to the image", b2.view_box((-5.2, 30.5, 10.1, 900.0), (100, 200)) == (0, 10, 31, 200))
check("box_overlap_frac", abs(b2.box_overlap_frac((0, 0, 10, 10), (5, 0, 20, 10)) - 0.5) < 1e-9
      and b2.box_overlap_frac((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0)

# ---- native size (tokens v2)
check("size_code('native') = 0, '2048' = 2048", b2.size_code("native") == 0 and b2.size_code("2048") == 2048)
check("parse_size('native') = 0", t2.parse_size("native") == 0 and t2.parse_size("1536") == 1536)
check("working_size native keeps the long side (multiple of 16)",
      t2.working_size(2160, 2880, 0) == (2160, 2880) and t2.working_size(1001, 1499, 0) == (1008, 1504))
check("working_size with a size is unchanged from v1",
      all(t2.working_size(h, w, s) == importlib.import_module("biorag_annotate_tokens_v1").working_size(h, w, s)
          for h, w, s in ((2160, 2880, 2048), (300, 500, 1024), (1001, 1499, 1536))))
check("cache file of native size is distinct", t2.cache_path("/c", "/a/x.tif", 12, 0).endswith("x.tif__L12_s0.npz"))
check("token CLI v2 uses the v2 UI module", t2._ui_module().__name__ == "biorag_annotate_v2")
check("UI v2 runs the v2 token script", os.path.basename(b2.TOKENS_SCRIPT) == "biorag_annotate_tokens_v2.py")

# ---- rules: v1 rule = box; v2 rule redoes the specimen mask per target
Tr = rng.normal(size=(18 * 22, 6)).astype(np.float16)
items_r = [(Tr, (18, 22))]
_, rgb_r, vis_r, prm_r = b1.pca_fit_tokens(items_r, "high-pass")
rule1 = {"kind": "colour windows", "variant": "high-pass", "fg": False, "mu1": prm_r["mu1"], "V1": prm_r["V1"],
         "mu": prm_r["mu"], "V": prm_r["V"], "lo": prm_r["lo"], "hi": prm_r["hi"],
         "windows": [{"on": True, "c": 120.0, "w": 90.0, "lo": 0, "hi": 100}, {"on": False, "c": 0, "w": 60, "lo": 0, "hi": 100, "near": 0}]}
s1, sc1 = b1.rule_scores(Tr, (18, 22), rule1)
s2, sc2 = b2.rule_scores(Tr, (18, 22), rule1)
check("a rule saved by v1 ('high-pass', no hp_kind) scores identically in v2 (box)", np.array_equal(s1, s2)
      and np.allclose(sc1, sc2))
Tb = Tf.reshape(-1, 5).astype(np.float16)
_, _, _, pv = b2.pca_fit_tokens_v2([(Tb, (24, 24))], "high-pass", masks=[blob.ravel()])
rule2 = {"kind": "similar", "variant": "high-pass", "fg": False, "mu1": pv["mu1"], "V1": pv["V1"], "hp_kind": "gauss",
         "hp_sigma": 3.0, "hp_mask": "foreground", "fg_mode": "per_image_raw_otsu", "flip_fg": False,
         "proto": np.ones(5, np.float32) / np.sqrt(5), "threshold": -2.0}
s3, _ = b2.rule_scores(Tb, (24, 24), rule2)
check("a v2 rule selects only inside the target's own specimen mask", np.array_equal(s3, blob.ravel()))

# the window's own path (pca_fit_tokens_v2 + the prm keys apply_rule copies) reproduces the crop selection
_items = [(Tf.reshape(-1, 5).astype(np.float16) + rng.normal(size=(576, 5)).astype(np.float16) * 0.5, (24, 24))]
for _var in ("high-pass", "raw+high-pass", "raw"):
    _masks = [b2.foreground_grid(T_, g_) for T_, g_ in _items]
    Pw, rgbw, visw, prmw = b2.pca_fit_tokens_v2(_items, _var, masks=_masks, sigma=3.0)
    prmw.update({"hp_mask": "foreground", "flip_fg": False, "fg_mode": "per_image_raw_otsu"})
    hue, _, _ = b2.rgb_to_hsv(rgbw)
    c1 = float(np.median(hue[visw]))
    wins = [{"on": True, "c": c1, "w": 120, "lo": 0, "hi": 100}, {"on": False, "c": 0, "w": 60, "lo": 0, "hi": 100, "near": 0}]
    new, _, _ = b2.colour_window_selection(rgbw, [(0, (24, 24))], wins)
    crop_sel = new & visw
    rule = {"kind": "colour windows", "variant": prmw["variant"], "fg": prmw["fg"], "mu1": prmw["mu1"], "V1": prmw["V1"],
            "mu": prmw["mu"], "V": prmw["V"], "lo": prmw["lo"], "hi": prmw["hi"], "windows": wins}
    for _k in ("hp_kind", "hp_sigma", "hp_mask", "fg_mode", "flip_fg"):
        rule[_k] = prmw[_k]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        b2.save_rule(os.path.join(d, "r.npz"), rule)
        rule_l = b2.load_rule(os.path.join(d, "r.npz"))
    got = b2.rule_scores(_items[0][0], (24, 24), rule_l)[0]
    check(f"v2 colour rule on the crop's own tokens reproduces the window selection ({_var}, specimen mask)",
          np.array_equal(got, crop_sel) and 0 < crop_sel.sum() < len(crop_sel), (int(got.sum()), int(crop_sel.sum())))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
