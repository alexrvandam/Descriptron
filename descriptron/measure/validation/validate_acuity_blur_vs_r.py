#!/usr/bin/env python3
"""descriptron_acuity_blur_v1 --acuityview_compat against R AcuityView 1.1.1, on the floating-point output.

Python writes square power-of-two 8-bit RGB test images: synthetic (sine gratings, a checkerboard, colour rings,
noise, an image with one constant channel) and real specimen crops (Platydracus dorsal photographs centred on a
white power-of-two canvas, as descriptron_acuity_blur_v1 pads them; one at full resolution, one halved). R loads each
with imager::load.image and runs the unmodified AcuityView::AcuityView(); the matrix AcuityView rasterises into its
PNG (rgbmatrix) is captured by replacing grid.raster in AcuityView's import environment with a function that stores
its argument and then calls grid::grid.raster - the package code itself is not touched. The floats are written with
writeBin (double precision) and compared with descriptron_acuity_blur_v1.acuityview_compat() on the same PNG.

Reported per image x setting: max and mean absolute difference (0-1 units) of the float arrays; the 8-bit PNG
AcuityView wrote against Descriptron's 8-bit quantisation (floor(255 v + 0.5)); and, for information only, how far
the science core (no channel stretch, real part, clip; --background white, --angle_mode canvas) lies from AcuityView.

  python validate_acuity_blur_vs_r.py --out_dir val_acuity/ --coco soma_v5b.json --image_dir photos/
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _sibling(name):
    here = Path(__file__).resolve()
    return next((c for c in (here.parents[1] / name, here.parent / name) if c.exists()), here.parents[1] / name)


def load_ab():
    spec = importlib.util.spec_from_file_location("descriptron_acuity_blur_v1", _sibling("descriptron_acuity_blur_v1.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["descriptron_acuity_blur_v1"] = m
    spec.loader.exec_module(m)
    return m


R_CODE = r'''
args <- commandArgs(TRUE); d <- args[1]
suppressMessages({library(AcuityView); library(imager)})
imp <- parent.env(asNamespace("AcuityView"))
captured <- NULL
unlockBinding("grid.raster", imp)
assign("grid.raster", function(image, ...) { captured <<- image; grid::grid.raster(image, ...) }, envir = imp)
jobs <- read.csv(file.path(d, "jobs.csv"), stringsAsFactors = FALSE)
times <- numeric(nrow(jobs))
for (k in seq_len(nrow(jobs))) {
  j <- jobs[k, ]
  img <- load.image(file.path(d, j$image))
  ry <- if (is.na(j$eyeY)) NULL else j$eyeY
  t0 <- Sys.time()
  suppressMessages(AcuityView(photo = img, distance = j$distance, realWidth = j$realWidth,
                              eyeResolutionX = j$eyeX, eyeResolutionY = ry, plot = FALSE,
                              output = file.path(d, paste0(j$id, "_R.png"))))
  times[k] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  con <- file(file.path(d, paste0(j$id, "_R.bin")), "wb"); writeBin(as.vector(captured), con, size = 8); close(con)
  writeLines(paste(dim(captured), collapse = " "), file.path(d, paste0(j$id, "_R.dim")))
}
writeLines(c(as.character(packageVersion("AcuityView")), as.character(packageVersion("imager")),
             as.character(packageVersion("fftwtools")), as.character(packageVersion("plotrix")), R.version.string),
           file.path(d, "versions.txt"))
writeLines(as.character(times), file.path(d, "times.txt"))
'''


def synthetic(n_list=(64, 128, 256), seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    n = 64
    x = np.arange(n)
    g = 0.5 + 0.5 * np.sin(2 * np.pi * x / 8)
    out["grating_v_64"] = np.stack([np.tile(g, (n, 1))] * 3, -1)                          # vertical bars
    gh = 0.5 + 0.5 * np.sin(2 * np.pi * x / 5)
    out["grating_h_col_64"] = np.stack([np.tile(gh[:, None], (1, n)), np.full((n, n), 0.3) + 0.4 * np.tile(g, (n, 1)),
                                        np.tile(g[::-1], (n, 1))], -1)
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    cb = (((yy // 8) + (xx // 8)) % 2).astype(float)
    out["checker_128"] = np.stack([cb, 0.2 + 0.6 * cb, 1 - cb], -1)
    r = np.hypot(yy - n / 2, xx - n / 3)
    out["rings_128"] = np.stack([0.5 + 0.5 * np.cos(r / 3), 0.5 + 0.5 * np.sin(r / 5), (xx / n)], -1)
    n = 256
    out["noise_256"] = rng.random((n, n, 3))
    const = rng.random((n, n, 3)) * 0.6 + 0.2
    const[..., 0] = 0.5                                                           # constant channel: no rescale
    out["constchan_256"] = const
    return {k: np.clip(np.floor(v * 255 + 0.5), 0, 255).astype(np.uint8) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--coco", required=True, help="COCO file holding the --real specimens")
    ap.add_argument("--image_dir", required=True, help="folder with the images of --coco")
    ap.add_argument("--real", nargs="*", default=["Platydracus newtoni", "Platydracus collaris"])
    ap.add_argument("--rscript", default="/usr/bin/Rscript")
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    ab = load_ab()
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)

    images = synthetic()
    settings_syn = [(2.0, 2.0, 0.2, None), (10.0, 1.0, 0.05, None), (5.0, 2.0, 0.1, 0.3)]
    jobs = []
    for name, im in images.items():
        Image.fromarray(im).save(d / f"{name}.png")
        for k, (dist, rw, ex, ey) in enumerate(settings_syn):
            jobs.append(dict(id=f"{name}_s{k}", image=f"{name}.png", distance=dist, realWidth=rw, eyeX=ex, eyeY=ey))
    # real crops, padded on a white power-of-two canvas as the tool does
    coco = json.load(open(a.coco))
    for i, sp in enumerate(a.real):
        rec = next(im for im in coco["images"] if im.get("species") == sp and im.get("view") == "dorsal")
        img = np.asarray(Image.open(Path(a.image_dir) / rec["file_name"]).convert("RGB"))
        ppm = float(rec["scale_px_per_mm"])
        if i % 2 == 1:   # halved, so a 1024 canvas
            img = np.asarray(Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2), Image.LANCZOS))
            ppm /= 2
        n = ab.next_pow2(max(img.shape[:2]))
        canvas, _ = ab.place_on_canvas(img.astype(float), n, 255.0)
        name = f"real_{sp.split()[-1]}_{n}"
        Image.fromarray(canvas.astype(np.uint8)).save(d / f"{name}.png")
        rw = n / ppm
        for k, (cpd, dist_mm) in enumerate([(6.29, 500.0), (22.5, 200.0), (33.0, 50.0)]):
            jobs.append(dict(id=f"{name}_s{k}", image=f"{name}.png", distance=dist_mm, realWidth=rw, eyeX=1 / cpd,
                             eyeY=None, cpd=cpd, px_per_mm=ppm))
    with open(d / "jobs.csv", "w") as fh:
        fh.write("id,image,distance,realWidth,eyeX,eyeY\n")
        for j in jobs:
            fh.write(f"{j['id']},{j['image']},{j['distance']!r},{j['realWidth']!r},{j['eyeX']!r},"
                     f"{'NA' if j['eyeY'] is None else repr(j['eyeY'])}\n")
    (d / "acuityview_capture.R").write_text(R_CODE)
    t0 = time.time()
    r = subprocess.run([a.rscript, str(d / "acuityview_capture.R"), str(d)], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr)
        raise SystemExit("R failed")
    r_secs = time.time() - t0
    r_times = [float(x) for x in (d / "times.txt").read_text().split()]

    rows = []
    for j, rt in zip(jobs, r_times):
        dims = [int(x) for x in (d / f"{j['id']}_R.dim").read_text().split()]
        R = np.fromfile(d / f"{j['id']}_R.bin", dtype=np.float64).reshape(dims, order="F")
        src = np.asarray(Image.open(d / j["image"]).convert("RGB"), dtype=float) / 255.0
        t1 = time.time()
        P = ab.acuityview_compat(src, j["distance"], j["realWidth"], j["eyeX"], j["eyeY"])
        py_t = time.time() - t1
        diff = np.abs(P - R)
        Rpng = np.asarray(Image.open(d / f"{j['id']}_R.png").convert("RGB")).astype(int)
        P8 = ab._to8(P).astype(int)
        # science core: no stretch, real part + clip, white canvas, AcuityView's canvas angle (info only)
        n = src.shape[0]
        ppd = ab.pixels_per_degree(n, j["realWidth"], j["distance"], "canvas")
        Hf = ab.half_mtf(n, ppd, j["eyeX"], j["eyeY"])
        lin = ab.srgb_to_linear(src)
        sci = np.stack([ab._irfft(ab._rfft(lin[..., k]) * Hf, n) for k in range(3)], -1)
        sci = ab.linear_to_srgb(np.clip(sci, 0, 1))
        rows.append({
            "id": j["id"], "image": j["image"], "n": n, "distance": j["distance"], "realWidth": j["realWidth"],
            "eyeResolutionX": j["eyeX"], "eyeResolutionY": j["eyeY"],
            "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "png8_max_abs_diff_levels": int(np.abs(Rpng - P8).max()),
            "png8_share_pixels_differing": float((Rpng != P8).any(-1).mean()),
            "r_float_range": [float(R.min()), float(R.max())],
            "science_core_vs_R_max_abs_diff": float(np.abs(sci - R).max()),
            "science_core_vs_R_mean_abs_diff": float(np.abs(sci - R).mean()),
            "r_seconds": rt, "python_seconds": py_t,
            "pass": bool(diff.max() <= a.tol)})
        print(f"{j['id']:28s} n={n:5d} max|d|={diff.max():.2e} mean|d|={diff.mean():.2e} "
              f"png8 max={rows[-1]['png8_max_abs_diff_levels']} ({rows[-1]['png8_share_pixels_differing']:.4f} px) "
              f"science-core vs R mean={rows[-1]['science_core_vs_R_mean_abs_diff']:.3e}  R {rt:.1f}s py {py_t:.2f}s")
    summ = {
        "what": "descriptron_acuity_blur_v1.acuityview_compat vs R AcuityView::AcuityView (unmodified; rgbmatrix "
                "captured from grid.raster), same 8-bit PNG input",
        "versions": (d / "versions.txt").read_text().split("\n"),
        "tolerance": a.tol, "n_comparisons": len(rows), "n_pass": sum(r["pass"] for r in rows),
        "max_abs_diff_overall": max(r["max_abs_diff"] for r in rows),
        "mean_abs_diff_overall": float(np.mean([r["mean_abs_diff"] for r in rows])),
        "png8_max_level_diff_overall": max(r["png8_max_abs_diff_levels"] for r in rows),
        "r_total_seconds": r_secs,
        "note_science_core": "science_core_vs_R_* are NOT pass/fail: the science mode deliberately does not stretch "
                             "channels, takes the real part and clips (see the program docstring)",
        "rows": rows}
    json.dump(summ, open(d / "validation_summary.json", "w"), indent=1)
    print(f"\n{summ['n_pass']}/{summ['n_comparisons']} within {a.tol:g}; overall max |diff| "
          f"{summ['max_abs_diff_overall']:.3e}, mean {summ['mean_abs_diff_overall']:.3e}; 8-bit PNG max "
          f"{summ['png8_max_level_diff_overall']} level(s)")
    return 0 if summ["n_pass"] == summ["n_comparisons"] else 1


if __name__ == "__main__":
    sys.exit(main())
