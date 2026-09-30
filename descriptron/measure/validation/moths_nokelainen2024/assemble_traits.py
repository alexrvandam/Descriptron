"""Per-specimen trait tables for descriptron_phylo: shape (aligned semilandmarks), proportions (scale-free),
colour pattern and texture (homology-cell features). Species column = 'species'."""
import csv, json, re
from pathlib import Path
import numpy as np, pandas as pd
cat = "moth_dorsal"
sp_of = lambda fn: fn.split("__")[0]
key = lambda fn: re.sub(r"\.png_\d+$", ".png", fn)
al = json.load(open(f"semilandmarks/{cat}/aligned_coco.json"))
name = {im["id"]: im["file_name"] for im in al["images"]}
rows = []
for a in al["annotations"]:
    xy = np.array(a["segmentation"][0]).reshape(-1, 2)
    if np.allclose(xy[0], xy[-1]): xy = xy[:-1]
    rows.append([sp_of(name[a["image_id"]]), key(name[a["image_id"]])] + xy.flatten().tolist())
k = (len(rows[0]) - 2) // 2
pd.DataFrame(rows, columns=["species", "file"] + [f"{c}{i+1}" for i in range(k) for c in "xy"]).to_csv("shape.csv", index=False)
m = pd.read_csv("measurements/all_metrics.csv")
m["file"] = m.image_filename.map(lambda s: re.sub(r"\.png.*$", ".png", str(s)))
m["circularity"] = 4 * np.pi * m.area_pixels / m.perimeter_pixels ** 2
m["axis_ratio"] = m.major_axis_length_pixels / m.minor_axis_length_pixels
m["species"] = m.file.map(sp_of)
m[["species", "file", "aspect_ratio", "extent", "solidity", "circularity", "axis_ratio"]].to_csv("meas.csv", index=False)
for src, dst in ((f"color_homology/color_homology_features_{cat}_combined_hw1.csv", "colour.csv"),
                 (f"texture_homology/texture_homology_features_{cat}.csv", "texture.csv")):
    d = pd.read_csv(src); d.insert(0, "species", d.filename.map(sp_of)); d["filename"] = d.filename.map(key)
    d = d.rename(columns={"filename": "file"})
    num = d.columns[2:]; d = d[["species", "file"] + [c for c in num if d[c].notna().all() and d[c].std() > 0]]
    d.to_csv(dst, index=False); print(dst, d.shape)
print("shape", k, "points; meas", len(m))
