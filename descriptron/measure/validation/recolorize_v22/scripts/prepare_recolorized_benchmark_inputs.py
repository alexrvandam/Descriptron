"""Copy a colour_benchmark_v1 `prepare` folder, swapping each wing's pixels for its Descriptron recolorize image, so
patternize and Colormesh read the SAME recolorized wings as Descriptron.

Everything the R scripts read (specimens.csv, landmarks/, outlines/, colormesh_* files, prepare_info.json) is copied
unchanged. Each image in images/ is the original padded PNG with the pixels inside the wing replaced by the recolorize
output (recolorize writes the wing on a pure-black background at the original image size; the padding added by
`prepare` is right/bottom only, so pixel coordinates and landmarks stay valid). Outside the wing the original image is
kept, so the only change any method sees is the recolorized wing.

Usage: prepare_recolorized_benchmark_inputs.py --inputs <prepare outdir> --recolor_dir <FHS_and_CLAHE_V22 --recolorize
       output folder> [--prefix recolorize_dE20_normalized_] --outdir <new folder>"""
import argparse, glob, json, os, shutil
import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", required=True)
ap.add_argument("--recolor_dir", required=True)
ap.add_argument("--prefix", default="recolorize_dE20_normalized_")
ap.add_argument("--suffix", default="_fg_whole_wing.png")
ap.add_argument("--outdir", required=True)
a = ap.parse_args()
if os.path.exists(a.outdir):
    raise SystemExit(f"{a.outdir} exists; choose a new folder")
os.makedirs(os.path.join(a.outdir, "images")); os.makedirs(os.path.join(a.outdir, "qc"))
for sub in ("landmarks", "outlines"):
    shutil.copytree(os.path.join(a.inputs, sub), os.path.join(a.outdir, sub))
for f in os.listdir(a.inputs):
    p = os.path.join(a.inputs, f)
    if os.path.isfile(p):
        shutil.copy2(p, os.path.join(a.outdir, f))

spec = pd.read_csv(os.path.join(a.inputs, "specimens.csv"))
rows = []
for k, r in spec.iterrows():
    key = r["stem"].replace(".tif", "tif")
    hits = [h for h in glob.glob(os.path.join(a.recolor_dir, a.prefix + key + "_*" + a.suffix))
            if os.path.basename(h)[len(a.prefix) + len(key):].count("_") == 4]  # _<annid>_fg_whole_wing.png
    assert len(hits) == 1, (r["id"], hits)
    rec = cv2.imread(hits[0], cv2.IMREAD_COLOR)
    base = cv2.imread(os.path.join(a.inputs, "images", r["id"] + ".png"), cv2.IMREAD_COLOR)
    h, w = rec.shape[:2]
    assert h <= base.shape[0] and w <= base.shape[1], r["id"]
    wing = rec.any(axis=2)
    out = base.copy()
    out[:h, :w][wing] = rec[wing]
    cv2.imwrite(os.path.join(a.outdir, "images", r["id"] + ".png"), out)
    rows.append({"id": r["id"], "recolorize_file": os.path.basename(hits[0]), "wing_pixels": int(wing.sum())})
    if k < 4:
        cv2.imwrite(os.path.join(a.outdir, "qc", f"before_after_{r['id']}.jpg"),
                    cv2.resize(np.hstack([base, out]), None, fx=0.35, fy=0.35))
pd.DataFrame(rows).to_csv(os.path.join(a.outdir, "recolorize_source_files.csv"), index=False)
json.dump({"inputs": a.inputs, "recolor_dir": a.recolor_dir, "prefix": a.prefix, "n": len(rows)},
          open(os.path.join(a.outdir, "recolorize_swap_info.json"), "w"), indent=1)
print(f"wrote {len(rows)} recolorized images to {a.outdir}/images")
