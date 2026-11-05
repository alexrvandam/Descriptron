"""How closely does the TPS frame of color_phenomics_homology_v2_1 follow each specimen's outline, with few or many
control points? For each specimen and each --steps value (every Nth of the sliding semilandmarks used as a TPS control
point, as --landmark_step does), the consensus outline (resampled to 1000 points) is warped onto the specimen with the
same TPS as the colour grid (build_tps_warp, mean -> specimen) and compared with the specimen's own mask from the COCO
file (the AI segmentation the semilandmarks were placed on):
  IoU            warped consensus polygon vs mask
  edge_error     mean distance of the warped outline to the mask edge, in % of the square root of the mask area
  outside_share  share of the warped consensus area that falls outside the mask (pixels a grid cell would sample
                 from the background or from the wrong part)
Usage: tps_control_density_fit.py --gpa_dir <semilandmarks/cat> --json <coco> --category_name <cat>
       --measure_dir <gui/measure> --out_csv <csv> [--steps 25,20,10,5,2,1]"""
import argparse, os, sys
import numpy as np, pandas as pd, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--gpa_dir", required=True); ap.add_argument("--json", required=True)
ap.add_argument("--category_name", required=True); ap.add_argument("--measure_dir", required=True)
ap.add_argument("--out_csv", required=True); ap.add_argument("--steps", default="25,20,10,5,2,1")
ap.add_argument("--limit", type=int, default=0, help="first N specimens only (testing)")
a = ap.parse_args()
sys.path.insert(0, a.measure_dir)
import color_phenomics_homology_v2_1 as ch

mean_shape, specimens = ch.load_gpa_outputs(a.gpa_dir)
lookup, _ = ch.build_mask_lookup(a.json, a.category_name)
steps = [int(s) for s in a.steps.split(",")]


def resample(c, n):
    c = np.vstack([c, c[:1]]); d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    t = np.linspace(0, d[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])])


dense_mean = resample(mean_shape, 1000)
rows = []
for k, (fn, spec) in enumerate(specimens.items()):
    if a.limit and k >= a.limit:
        break
    ml = lookup.get(ch.get_original_filename(fn)) or lookup.get(os.path.basename(ch.get_original_filename(fn)))
    if not ml:
        continue
    h, w = int(ml["img"]["height"]), int(ml["img"]["width"])
    m = ch.decode_mask(ml["ann"], h, w).astype(bool)
    if m.sum() < 100:
        continue
    edge = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    edge = np.vstack([e.reshape(-1, 2) for e in edge]).astype(float)
    n = min(len(spec), len(mean_shape)); sp, mn = spec[:n], mean_shape[:n]
    scale = np.sqrt(m.sum())
    for st in steps:
        idx = list(range(0, n, st))
        warp = ch.build_tps_warp(mn[idx], sp[idx])
        poly = warp(dense_mean)
        pm = np.zeros((h, w), np.uint8); cv2.fillPoly(pm, [np.round(poly).astype(np.int32)], 1); pm = pm.astype(bool)
        inter = (pm & m).sum(); union = (pm | m).sum()
        # nearest mask-edge pixel for every warped outline point (chunked)
        dmin = np.concatenate([np.sqrt(((poly[i:i + 200, None] - edge[None]) ** 2).sum(-1)).min(1) for i in range(0, len(poly), 200)])
        rows.append(dict(specimen=fn, species=fn.split("__")[0], n_control=len(idx), step=st, IoU=inter / union,
                         edge_error_pct=100 * dmin.mean() / scale, outside_share=(pm & ~m).sum() / max(pm.sum(), 1)))
d = pd.DataFrame(rows); d.to_csv(a.out_csv, index=False)
print(d.groupby("n_control")[["IoU", "edge_error_pct", "outside_share"]].median().round(4).to_string())
