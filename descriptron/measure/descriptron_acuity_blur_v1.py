#!/usr/bin/env python3
"""descriptron_acuity_blur_v1.py - what a viewer with a given visual acuity resolves of a specimen at a given distance
=====================================================================================================================

Blurs every specimen photograph of a COCO file to the spatial detail an animal with visual acuity A (cycles per
degree) can resolve from a viewing distance D, so that colour pattern can be measured "through the predator's eye"
by the rest of Descriptron (the blurred images keep the original file names and sizes: the COCO masks still apply).

Method: AcuityView (Caves, E.M. & Johnsen, S. 2018. AcuityView: an r package for portraying the effects of visual
acuity on scenes observed by an animal. Methods in Ecology and Evolution 9: 793-797, doi:10.1111/2041-210X.12911),
the R package AcuityView 1.1.1, re-implemented in Python:

  1. the minimum resolvable angle MRA = 1 / acuity (degrees per cycle; or --mra_deg directly)
  2. the image is a square of n x n pixels (n a power of two) covering realWidth at distance D:
         widthInDegrees  = 57.2958 * 2 * atan(realWidth / D / 2)
         pixelsPerDegree = n / widthInDegrees
  3. on the centred (fft-shifted) frequency grid, centre at index round(n/2) (0-based; round(n/2)+1 in R), a
     frequency-domain pixel at radius r (pixels) from the centre has spatial frequency
         f = round(r) / n * pixelsPerDegree     (cycles per degree)
     and the eye's modulation transfer function there is the Gaussian
         MTF(f) = exp(-3.56 * (MRA * f)^2)
     (MTF(0) = 1; MTF falls to exp(-3.56) = 0.028 at f = acuity; a Gaussian point spread with full width at half
     maximum = MRA). With different horizontal and vertical MRA (X, Y) the effective MRA in direction theta is
     MRA_X MRA_Y / sqrt((MRA_Y cos theta)^2 + (MRA_X sin theta)^2), as AcuityView.
  4. each channel is converted from sRGB to linear light (IEC 61966-2-1), Fourier transformed, multiplied by the
     MTF, transformed back, and converted back to sRGB.

This file is validated against R AcuityView 1.1.1 (validation/validate_acuity_blur_vs_r.py): --acuityview_compat
reproduces the package's floating-point output array (the matrix it rasterises into its PNG) on synthetic gratings,
checkerboards, noise and real padded specimen crops, at several distances, widths, acuities and one anisotropic
setting.

Deviations from AcuityView in the default (science) mode, and why
-----------------------------------------------------------------
--acuityview_compat switches every one of these off and reproduces AcuityView exactly (for validation).

  a. No per-channel min-max stretch. AcuityView rescales each colour channel of the input to [0, 1] before
     linearising, and again after blurring when a channel exceeds 1. That changes hue, chroma and lightness by a
     different amount in every photograph (a dark beetle on a white plate has min ~0 and max ~1 anyway, but a red
     channel spanning 0.1-0.9 is stretched to 0-1), so colours measured afterwards would no longer be the
     specimen's. Science mode linearises the 8-bit sRGB values as they are and clips the result to [0, 1].
  b. Real part instead of modulus. AcuityView multiplies the fft-shifted spectrum by the MTF and inverts it
     without un-shifting, which multiplies the image by (-1)^(x+y); it removes that sign by taking Mod(). The MTF
     grid is real and even, so the inverse is real and Mod() equals |blurred image|. Science mode un-shifts the MTF
     and keeps the (signed) real result, then clips negatives to 0: the two differ only where the sampled
     (round(r)) Gaussian rings below zero, which Mod() would fold back as positive light.
  c. Background (--background). AcuityView blurs the whole picture, so the white plate around a specimen washes
     into the specimen's edge colours. The default here, normalized, is normalized convolution (Knutsson & Westin
     1993, Proc. IEEE CVPR: 515-523): blurred = blur(image x mask) / blur(mask) inside the specimen mask (the union
     of the COCO annotations of --mask_category), computed in linear light; outside the mask the output is white.
     Inside the specimen this is the MTF-weighted mean of specimen pixels only - what the eye would integrate over
     the specimen if it were seen against a background of its own average colour near each edge. Pixels within
     --edge_px of the mask (rasterisation differences between programs) also get the normalized value, or the
     original pixel where blur(mask) < 1e-3. white = the photograph as it is (plate included) on a white pad, as
     AcuityView; value = everything outside the mask replaced by one sRGB colour (e.g. a dung or soil colour) before
     blurring.
  d. Non-square photographs. AcuityView needs a square power-of-two image. Each photograph is centred on the
     smallest square power-of-two canvas that holds it with a margin on every side of at least --margin_frac of its
     longer side and 3 MTF full widths (MRA) of the most blurring requested setting (so the circular FFT does not
     wrap blur from one edge onto the other); offsets (n - H) // 2 and (n - W) // 2; the padding is white (white,
     compat), the --background value colour (value) or outside the mask (normalized). realWidth = n / px_per_mm
     (mm) from images[].scale_px_per_mm. The output is cropped back to the photograph's own size and position.
  e. Pixels per degree (--angle_mode). AcuityView spreads the angle of the whole canvas evenly over its pixels
     (canvas mode), so the scale depends on how much padding the canvas has; at 5 cm a 40 mm canvas gives 4.8%
     fewer pixels per degree than the line of sight's own scale. Science mode (centre) uses the scale at the centre
     of the view, pixelsPerDegree = px_per_mm x D / 57.2958 (the limit of AcuityView's formula for a small canvas),
     independent of padding; at 20 cm and beyond the two differ by < 0.5% for these photographs. Both are flat-image
     approximations (the viewer looks perpendicular at the centre of the specimen).
  f. The degree constant 57.2958 (AcuityView's rounding of 180/pi) is kept in both modes.

Colour space: the photographs are sRGB and the blur is a spatial filter only. It says what detail is resolved, not
which colours the viewer distinguishes: a bird's colour vision (UV-sensitive, tetrachromatic) needs reflectance
spectra and a receptor-noise model, which RGB photographs cannot supply. Colours measured afterwards are human-vision
CIELAB of the blurred image.

Usage
-----
  python descriptron_acuity_blur_v1.py --coco soma_v5b.json --image_dir photos/ --out_dir bird_blur_v1/ \\
      --acuity_cpd 6.29 22.5 33 --distance_cm 5 20 50 --mask_category soma_abdomen --view dorsal \\
      --acuity_labels 6.29=starling 22.5=blackbird 33=magpie \\
      --qc_species newtoni collaris coeruleipennis cooteri basicornis felis marmorellus aerugineus

  writes  bird_blur_v1/cpd6.29_d5cm/<file_name>  ...  one folder per acuity x distance (or mraX_dYcm with --mra_deg),
          blur_parameters.csv (one row per image x setting: canvas, realWidth, widthInDegrees, pixels per degree, MTF
          full width in px and mm), blur_settings.json, qc_sheet.jpg
"""
import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import scipy.fft as _fft
    _FFT_KW = {"workers": 1}
except ImportError:  # pragma: no cover
    import numpy.fft as _fft
    _FFT_KW = {}
try:
    from scipy.ndimage import binary_dilation as _binary_dilation
except ImportError:  # pragma: no cover
    _binary_dilation = None

DEG = 57.2958   # AcuityView's degrees per radian
MTF_K = 3.56    # AcuityView's Gaussian MTF constant


# ------------------------------------------------------------------ colour transfer (IEC 61966-2-1, as AcuityView)
def srgb_to_linear(v):
    v = np.asarray(v, dtype=float)
    return np.where(v <= 0.04045, v / 12.92, ((np.maximum(v, 0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.asarray(v, dtype=float)
    return np.where(v < 0.0031308, v * 12.92, 1.055 * np.power(np.maximum(v, 0), 1 / 2.4) - 0.055)


def plotrix_rescale(x, lo, hi):
    """plotrix::rescale(x, c(lo, hi)): linear map of range(x) onto [lo, hi]; unchanged when x is constant."""
    mn, mx = float(np.min(x)), float(np.max(x))
    if mn == mx:
        return x
    return lo + (x - mn) * ((hi - lo) / (mx - mn))


# ------------------------------------------------------------------ geometry and MTF
def next_pow2(k):
    return 1 << max(1, int(math.ceil(math.log2(max(int(k), 2)))))


def width_in_degrees(real_width, distance):
    return DEG * (2 * math.atan(real_width / distance / 2))


def pixels_per_degree(n, real_width=None, distance=None, angle_mode="canvas", px_per_unit=None):
    """canvas: AcuityView (n / widthInDegrees of the canvas). centre: px_per_unit * distance / 57.2958 (the scale on
    the line of sight; real_width is not used). Units of real_width, distance and 1/px_per_unit must agree."""
    if angle_mode == "canvas":
        return n / width_in_degrees(real_width, distance)
    if angle_mode == "centre":
        return px_per_unit * distance / DEG
    raise ValueError(angle_mode)


def mtf(freq_cpd, mra_deg):
    """Gaussian modulation transfer function of AcuityView: exp(-3.56 (MRA f)^2)."""
    return np.exp(-MTF_K * (mra_deg * np.asarray(freq_cpd, dtype=float)) ** 2)


def acuityview_mtf_grid(n, ppd, mra_x, mra_y=None):
    """The MTF on AcuityView's fft-shifted grid, array[row = image y, col = image x], DC at [n//2, n//2]
    (R: blur[i, j] with i = image x, centre round(w/2)+1)."""
    if mra_y is None:
        mra_y = mra_x
    c = int(round(n / 2))
    idx = np.arange(n, dtype=float) - c
    X = idx[None, :]
    Y = idx[:, None]
    r = np.sqrt(X ** 2 + Y ** 2)
    freq = np.round(r) / n * ppd
    if mra_x == mra_y:
        eff = np.full_like(r, float(mra_x))
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            sin, cos = Y / r, X / r
            eff = mra_x * mra_y / np.sqrt((mra_y * cos) ** 2 + (mra_x * sin) ** 2)
    H = np.exp(-MTF_K * (eff * freq) ** 2)
    H[c, c] = 1.0
    return H


def mtf_fwhm_px(mra_deg, ppd):
    """Full width at half maximum of the point spread (= the MRA) in pixels."""
    return mra_deg * ppd


# ------------------------------------------------------------------ AcuityView 1.1.1, exactly
def acuityview_compat(img, distance, real_width, mra_x, mra_y=None):
    """AcuityView::AcuityView step by step on a square power-of-two sRGB image (float, [0,1], H x W x 3, rows = y).
    Returns the float array AcuityView rasterises (its rgbmatrix, already in image orientation)."""
    img = np.asarray(img, dtype=float)
    n = img.shape[0]
    if img.shape[0] != img.shape[1] or n & (n - 1):
        raise ValueError("AcuityView needs a square image with a power-of-two size")
    ppd = n / width_in_degrees(real_width, distance)
    H = acuityview_mtf_grid(n, ppd, mra_x, mra_y)
    out = np.empty((n, n, 3))
    for k in range(3):
        lin = srgb_to_linear(plotrix_rescale(img[..., k], 0.0, 1.0))
        F = np.fft.fftshift(np.fft.fft2(lin)) / n            # (1/w) * fft_matrix_shift(fftw2d(m))
        inv = np.fft.ifft2(F * H) * n                        # (1/w) * unnormalised inverse
        s = linear_to_srgb(np.abs(inv))                      # Mod, then linear -> sRGB
        if s.max() > 1:
            s = plotrix_rescale(s, float(s.min()), 1.0)
        out[..., k] = s
    return out


# ------------------------------------------------------------------ science mode
def _rfft(a):
    return _fft.rfft2(a, **_FFT_KW)


def _irfft(A, n):
    return _fft.irfft2(A, s=(n, n), **_FFT_KW)


def half_mtf(n, ppd, mra_x, mra_y=None):
    """AcuityView's MTF un-shifted to FFT order, half spectrum for rfft2 (the grid is real and even)."""
    return np.fft.ifftshift(acuityview_mtf_grid(n, ppd, mra_x, mra_y))[:, : n // 2 + 1]


def place_on_canvas(arr, n, fill):
    H, W = arr.shape[:2]
    oy, ox = (n - H) // 2, (n - W) // 2
    shape = (n, n) + arr.shape[2:]
    c = np.empty(shape, dtype=float)
    c[...] = fill
    c[oy:oy + H, ox:ox + W] = arr
    return c, (oy, ox)


def crop_from_canvas(c, offset, H, W):
    oy, ox = offset
    return c[oy:oy + H, ox:ox + W]


def dilate(mask, px):
    if px <= 0:
        return mask.copy()
    if _binary_dilation is not None:
        yy, xx = np.mgrid[-px:px + 1, -px:px + 1]
        return _binary_dilation(mask, structure=(xx ** 2 + yy ** 2) <= px * px)
    out = mask.copy()   # pragma: no cover - square fallback
    for dy in range(-px, px + 1):
        for dx in range(-px, px + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def canvas_size(H, W, margin_frac=0.1, min_margin_px=0):
    m = max(int(math.ceil(margin_frac * max(H, W))), int(math.ceil(min_margin_px)))
    return next_pow2(max(H, W) + 2 * m)


def blur_image(img8, settings, px_per_mm, mask=None, background="normalized", bg_value=(255, 255, 255),
               angle_mode="centre", compat=False, margin_frac=0.1, edge_px=3, n=None):
    """Blur one photograph for every setting.

    img8: H x W x 3 uint8 sRGB; settings: list of dicts with 'mra_deg' (and optional 'mra_y_deg') and 'distance_mm';
    mask: H x W bool specimen mask (needed for normalized / value). Returns ({name: H x W x 3 uint8}, info dict)."""
    img = np.asarray(img8, dtype=float)[..., :3] / 255.0
    Hh, Ww = img.shape[:2]
    if n is None:
        # margin: 3 MRA of the most blurring setting (computed with the centre scale, padding-independent)
        worst = max(3 * s["mra_deg"] * px_per_mm * s["distance_mm"] / DEG for s in settings) if settings else 0
        n = canvas_size(Hh, Ww, margin_frac, worst)
    real_width = n / px_per_mm
    info = {"canvas_px": n, "real_width_mm": real_width, "settings": {}}
    out = {}
    if compat:
        canvas, off = place_on_canvas(img, n, 1.0)
        for s in settings:
            r = acuityview_compat(canvas, s["distance_mm"], real_width, s["mra_deg"], s.get("mra_y_deg"))
            out[s["name"]] = _to8(crop_from_canvas(r, off, Hh, Ww))
            ppd = n / width_in_degrees(real_width, s["distance_mm"])
            info["settings"][s["name"]] = _setting_info(n, real_width, s, ppd, px_per_mm)
        return out, info

    lin = srgb_to_linear(img)
    if background == "normalized":
        if mask is None or not mask.any():
            raise ValueError("background 'normalized' needs a non-empty specimen mask")
        m = mask.astype(float)
        cm, off = place_on_canvas(m, n, 0.0)
        cl, _ = place_on_canvas(lin * m[..., None], n, 0.0)
        FM = _rfft(cm)
        FL = [_rfft(cl[..., k]) for k in range(3)]
        region = dilate(mask, edge_px)
    else:
        if background == "white":
            src = lin
            fill = 1.0
        elif background == "value":
            if mask is None:
                raise ValueError("background 'value' needs a specimen mask")
            fill = srgb_to_linear(np.asarray(bg_value, dtype=float) / 255.0)
            src = np.where(mask[..., None], lin, fill)
        else:
            raise ValueError(background)
        cl, off = place_on_canvas(src, n, fill)
        FL = [_rfft(cl[..., k]) for k in range(3)]
    for s in settings:
        if angle_mode == "canvas":
            ppd = pixels_per_degree(n, real_width, s["distance_mm"], "canvas")
        else:
            ppd = pixels_per_degree(n, None, s["distance_mm"], "centre", px_per_unit=px_per_mm)
        Hf = half_mtf(n, ppd, s["mra_deg"], s.get("mra_y_deg"))
        res = np.empty((Hh, Ww, 3))
        if background == "normalized":
            den = crop_from_canvas(_irfft(FM * Hf, n), off, Hh, Ww)
            ok = region & (den > 1e-3)
            for k in range(3):
                num = crop_from_canvas(_irfft(FL[k] * Hf, n), off, Hh, Ww)
                ch = np.ones((Hh, Ww))                       # white outside the specimen
                ch[region] = lin[..., k][region]             # fallback: the original pixel
                ch[ok] = num[ok] / den[ok]
                res[..., k] = ch
        else:
            for k in range(3):
                res[..., k] = crop_from_canvas(_irfft(FL[k] * Hf, n), off, Hh, Ww)
        out[s["name"]] = _to8(linear_to_srgb(np.clip(res, 0.0, 1.0)))
        info["settings"][s["name"]] = _setting_info(n, real_width, s, ppd, px_per_mm)
    return out, info


def _setting_info(n, real_width, s, ppd, px_per_mm):
    fwhm = mtf_fwhm_px(s["mra_deg"], ppd)
    return {"distance_mm": s["distance_mm"], "mra_deg": s["mra_deg"],
            "width_in_degrees": width_in_degrees(real_width, s["distance_mm"]), "pixels_per_degree": ppd,
            "fwhm_px": fwhm, "fwhm_mm": fwhm / px_per_mm}


def _to8(a):
    """[0,1] float -> uint8 as R's rgb(): floor(255 v + 0.5)."""
    return np.clip(np.floor(np.asarray(a) * 255 + 0.5), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ COCO masks
def rasterise_segmentation(seg, H, W):
    """COCO polygons (list of flat lists) or RLE (uncompressed counts; compressed via pycocotools) -> bool mask."""
    if isinstance(seg, dict):
        counts = seg.get("counts")
        if isinstance(counts, list):
            flat = np.zeros(H * W, dtype=bool)
            pos, val = 0, False
            for c in counts:
                flat[pos:pos + c] = val
                pos += c
                val = not val
            return flat.reshape((W, H)).T     # column-major
        from pycocotools import mask as mu   # compressed RLE
        return mu.decode(seg).astype(bool)
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for poly in seg:
        if len(poly) >= 6:
            d.polygon([(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)], fill=1, outline=1)
    return np.asarray(im, dtype=bool)


def specimen_mask(coco, image, category="soma_abdomen"):
    cats = {c["name"]: c["id"] for c in coco.get("categories", [])}
    if category not in ("all", None, "") and category not in cats:
        raise SystemExit(f"--mask_category {category!r} not in the COCO categories {sorted(cats)}")
    cid = None if category in ("all", None, "") else cats[category]
    H, W = int(image["height"]), int(image["width"])
    m = np.zeros((H, W), dtype=bool)
    for a in coco.get("annotations", []):
        if a["image_id"] == image["id"] and (cid is None or a["category_id"] == cid) and a.get("segmentation"):
            m |= rasterise_segmentation(a["segmentation"], H, W)
    return m


# ------------------------------------------------------------------ settings
def fmt(x):
    return f"{x:g}"


def build_settings(acuity_cpd=None, mra_deg=None, distance_cm=(20,)):
    settings = []
    if acuity_cpd:
        for a in acuity_cpd:
            for d in distance_cm:
                settings.append({"name": f"cpd{fmt(a)}_d{fmt(d)}cm", "acuity_cpd": a, "mra_deg": 1.0 / a,
                                 "distance_cm": d, "distance_mm": 10.0 * d})
    for m in (mra_deg or []):
        for d in distance_cm:
            settings.append({"name": f"mra{fmt(m)}_d{fmt(d)}cm", "acuity_cpd": 1.0 / m, "mra_deg": m,
                             "distance_cm": d, "distance_mm": 10.0 * d})
    return settings


# ------------------------------------------------------------------ worker
def _work(job):
    (img_path, rec, mask, settings, args_d, out_dir) = job
    t0 = time.time()
    img8 = np.asarray(Image.open(img_path).convert("RGB"))
    if img8.shape[0] != int(rec["height"]) or img8.shape[1] != int(rec["width"]):
        return {"file": rec["file_name"], "error": f"image is {img8.shape[1]}x{img8.shape[0]}, COCO says "
                                                   f"{rec['width']}x{rec['height']}"}
    outs, info = blur_image(img8, settings, float(rec["scale_px_per_mm"]), mask=mask,
                            background=args_d["background"], bg_value=args_d["bg_value"],
                            angle_mode=args_d["angle_mode"], compat=args_d["compat"],
                            margin_frac=args_d["margin_frac"], edge_px=args_d["edge_px"])
    name = Path(rec["file_name"]).name
    for sname, arr in outs.items():
        p = Path(out_dir) / sname / name
        if p.suffix.lower() in (".jpg", ".jpeg"):
            Image.fromarray(arr).save(p, quality=97, subsampling=0)
        else:
            Image.fromarray(arr).save(p)
    info.update({"file": rec["file_name"], "species": rec.get("species", ""), "px_per_mm": rec["scale_px_per_mm"],
                 "height": img8.shape[0], "width": img8.shape[1], "seconds": time.time() - t0,
                 "mask_px": int(mask.sum()) if mask is not None else None})
    return info


# ------------------------------------------------------------------ QC sheet
def _font(size):
    for p in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def qc_sheet(rows, settings, image_dir, out_dir, out_path, labels=None, tile_h=300, title=""):
    """rows: list of (species, file_name). One row per species: original, then every setting."""
    labels = labels or {}
    cols = [("original", None)] + [(s["name"], s) for s in settings]
    tiles, widths = [], []
    for sp, fn in rows:
        r = []
        for cname, s in cols:
            p = (Path(image_dir) / fn) if s is None else (Path(out_dir) / cname / Path(fn).name)
            im = Image.open(p).convert("RGB")
            w = max(1, int(round(im.width * tile_h / im.height)))
            r.append(im.resize((w, tile_h), Image.LANCZOS))
        tiles.append(r)
        widths.append(max(t.width for t in r))
    tw = max(widths) + 10
    f1, f2 = _font(15), _font(13)
    head, left = 62, 190
    sheet = Image.new("RGB", (left + tw * len(cols), head + (tile_h + 10) * len(rows) + 30), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 4), title, fill="black", font=f1)
    for j, (cname, s) in enumerate(cols):
        x = left + j * tw
        if s is None:
            l1, l2 = "original", ""
        else:
            l1 = labels.get(fmt(s["acuity_cpd"]), f"{fmt(s['acuity_cpd'])} cpd")
            l2 = f"{fmt(s['acuity_cpd'])} cpd, {fmt(s['distance_cm'])} cm"
        d.text((x + 4, 26), l1, fill="black", font=f1)
        d.text((x + 4, 44), l2, fill="black", font=f2)
    for i, (sp, fn) in enumerate(rows):
        y = head + i * (tile_h + 10)
        d.text((6, y + tile_h // 2 - 8), sp.replace("Platydracus ", "P. "), fill="black", font=f1)
        for j, t in enumerate(tiles[i]):
            sheet.paste(t, (left + j * tw + (tw - t.width) // 2, y))
    sheet.save(out_path, quality=92)
    return out_path


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, help="COCO json; images[] need scale_px_per_mm (px per mm)")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    g = ap.add_argument_group("viewer")
    g.add_argument("--acuity_cpd", type=float, nargs="+", help="visual acuity(s), cycles per degree (MRA = 1/cpd)")
    g.add_argument("--mra_deg", type=float, nargs="+", help="minimum resolvable angle(s) in degrees, instead of "
                                                            "or beside --acuity_cpd")
    g.add_argument("--distance_cm", type=float, nargs="+", default=[20.0], help="viewing distance(s), cm")
    g.add_argument("--acuity_labels", nargs="*", default=[], help="labels for the QC sheet, e.g. 6.29=starling")
    g = ap.add_argument_group("method")
    g.add_argument("--background", default="normalized", help="normalized (default) | white | an sRGB colour "
                   "R,G,B (e.g. 92,64,51): pixels outside the mask set to it before blurring")
    g.add_argument("--mask_category", default="soma_abdomen", help="COCO category whose annotations form the "
                   "specimen mask, or 'all' (every annotation of the image)")
    g.add_argument("--view", default="dorsal", help="only images with images[].view equal to this ('all' = every image)")
    g.add_argument("--angle_mode", choices=["centre", "canvas"], default="centre",
                   help="pixels per degree: centre (scale on the line of sight, default) or canvas (AcuityView)")
    g.add_argument("--acuityview_compat", action="store_true", help="reproduce AcuityView 1.1.1 exactly (channel "
                   "min-max stretch, Mod, final rescale, whole photo on a white pad, canvas angle) - for validation")
    g.add_argument("--margin_frac", type=float, default=0.1, help="least padding on each side, share of the longer side")
    g.add_argument("--edge_px", type=int, default=3, help="normalized: also fill pixels this close to the mask")
    ap.add_argument("--images", nargs="*", help="only these file names")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--qc_species", nargs="*", default=[], help="species (substring of images[].species) for the QC sheet")
    ap.add_argument("--qc_sheet", default=None, help="QC sheet path (default <out_dir>/qc_sheet.jpg)")
    args = ap.parse_args(argv)

    if not args.acuity_cpd and not args.mra_deg:
        ap.error("give --acuity_cpd and/or --mra_deg")
    settings = build_settings(args.acuity_cpd, args.mra_deg, args.distance_cm)
    if args.acuityview_compat:
        bg, bgv = "white", (255, 255, 255)
    elif args.background in ("normalized", "white"):
        bg, bgv = args.background, (255, 255, 255)
    else:
        try:
            bgv = tuple(int(x) for x in args.background.split(","))
            assert len(bgv) == 3
        except Exception:
            ap.error("--background must be normalized, white or R,G,B")
        bg = "value"
    coco = json.load(open(args.coco))
    imgs = [im for im in coco["images"] if args.view in ("all", "") or im.get("view") == args.view]
    if args.images:
        keep = {Path(x).name for x in args.images}
        imgs = [im for im in imgs if Path(im["file_name"]).name in keep]
    imgs = [im for im in imgs if im.get("scale_px_per_mm")]
    if not imgs:
        raise SystemExit("no images (check --view and images[].scale_px_per_mm)")
    out = Path(args.out_dir)
    for s in settings:
        (out / s["name"]).mkdir(parents=True, exist_ok=True)
    args_d = {"background": bg, "bg_value": bgv, "angle_mode": "canvas" if args.acuityview_compat else args.angle_mode,
              "compat": args.acuityview_compat, "margin_frac": args.margin_frac, "edge_px": args.edge_px}
    jobs = []
    for im in imgs:
        p = Path(args.image_dir) / im["file_name"]
        if not p.exists():
            p = Path(args.image_dir) / Path(im["file_name"]).name
        mask = specimen_mask(coco, im, args.mask_category) if bg != "white" else None
        if mask is not None and not mask.any():
            print(f"  skip {im['file_name']}: no {args.mask_category} annotation", file=sys.stderr)
            continue
        jobs.append((str(p), im, mask, settings, args_d, str(out)))
    t0 = time.time()
    print(f"{len(jobs)} images x {len(settings)} settings -> {out}  (background {bg}, angle {args_d['angle_mode']}"
          f"{', AcuityView compat' if args.acuityview_compat else ''})", flush=True)
    infos = []
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            for info in ex.map(_work, jobs):
                infos.append(info)
    else:
        infos = [_work(j) for j in jobs]
    errors = [i for i in infos if "error" in i]
    for e in errors:
        print(f"  ERROR {e['file']}: {e['error']}", file=sys.stderr)
    infos = [i for i in infos if "error" not in i]
    with open(out / "blur_parameters.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "species", "setting", "acuity_cpd", "mra_deg", "distance_cm", "px_per_mm", "width_px",
                    "height_px", "canvas_px", "real_width_mm", "width_in_degrees", "pixels_per_degree", "fwhm_px",
                    "fwhm_mm", "mask_px"])
        for i in infos:
            for s in settings:
                si = i["settings"][s["name"]]
                w.writerow([i["file"], i["species"], s["name"], f"{s['acuity_cpd']:.6g}", f"{s['mra_deg']:.6g}",
                            fmt(s["distance_cm"]), i["px_per_mm"], i["width"], i["height"], i["canvas_px"],
                            f"{i['real_width_mm']:.4f}", f"{si['width_in_degrees']:.4f}",
                            f"{si['pixels_per_degree']:.4f}", f"{si['fwhm_px']:.3f}", f"{si['fwhm_mm']:.5f}",
                            i["mask_px"]])
    meta = {"program": "descriptron_acuity_blur_v1.py", "command": " ".join(sys.argv), "coco": args.coco,
            "image_dir": args.image_dir, "background": bg, "background_value": bgv, "mask_category": args.mask_category,
            "view": args.view, "angle_mode": args_d["angle_mode"], "acuityview_compat": args.acuityview_compat,
            "margin_frac": args.margin_frac, "edge_px": args.edge_px, "settings": settings,
            "n_images": len(infos), "n_errors": len(errors), "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(out / "blur_settings.json", "w"), indent=1)
    print(f"done: {len(infos)} images in {meta['seconds']} s; {len(errors)} errors", flush=True)
    if args.qc_species:
        rows = []
        for q in args.qc_species:
            hit = next((im for im in imgs if q.lower() in str(im.get("species", "")).lower()
                        and any(i["file"] == im["file_name"] for i in infos)), None)
            if hit is None:
                print(f"  QC: no image for {q}", file=sys.stderr)
                continue
            rows.append((hit.get("species", q), hit["file_name"]))
        labels = dict(kv.split("=", 1) for kv in args.acuity_labels)
        labels = {fmt(float(k)): v for k, v in labels.items()}
        title = (f"Predator's-eye blur (AcuityView MTF; background {bg}, angle {args_d['angle_mode']}"
                 f"{', AcuityView compat' if args.acuityview_compat else ''}): original, then acuity x distance")
        p = qc_sheet(rows, settings, args.image_dir, out, args.qc_sheet or out / "qc_sheet.jpg", labels, title=title)
        print(f"QC sheet: {p}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
