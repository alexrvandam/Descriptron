"""Figure: the colour-homology grid warped onto specimens with few or many TPS control points (as --landmark_step).
One row per specimen, one column per control-point density: the image the grid samples (e.g. the recolorize output),
the specimen's mask edge (white), the warped consensus outline (yellow), the warped grid cells kept by the grid
(cyan; --grid_ext as in color_phenomics_homology_v2_1) and the control points (red).
Usage: tps_density_overlay.py --gpa_dir <semilandmarks/cat> --json <coco> --category_name <cat> --image_dir <dir>
       --measure_dir <gui/measure> --specimens a.png,b.png --out <png> [--steps 25,5,1] [--grid_ext 2]"""
import argparse, os, sys
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
for k in ("--gpa_dir", "--json", "--category_name", "--image_dir", "--measure_dir", "--specimens", "--out"):
    ap.add_argument(k, required=True)
ap.add_argument("--steps", default="25,5,1"); ap.add_argument("--grid_ext", type=int, default=2)
a = ap.parse_args()
sys.path.insert(0, a.measure_dir)
import color_phenomics_homology_v2_1 as ch
from homology_grid_ext import create_grid_extended

mean_shape, specimens = ch.load_gpa_outputs(a.gpa_dir)
lookup, _ = ch.build_mask_lookup(a.json, a.category_name)
grid = create_grid_extended(mean_shape, n_rows=8, n_cols=10, ext=a.grid_ext, min_overlap=0.02) if a.grid_ext > 0 \
    else ch.create_grid_inside_contour(mean_shape, n_rows=8, n_cols=10)
steps = [int(s) for s in a.steps.split(",")]
wanted = a.specimens.split(",")
keys = [k for w in wanted for k in specimens if ch.get_original_filename(k) == w]
fig, axs = plt.subplots(len(keys), len(steps), figsize=(4.2 * len(steps), 3.2 * len(keys)), squeeze=False)
for r, fn in enumerate(keys):
    spec = specimens[fn]; n = min(len(spec), len(mean_shape)); sp, mn = spec[:n], mean_shape[:n]
    ip = ch.resolve_image_path(fn, a.image_dir); img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
    ml = lookup[ch.get_original_filename(fn)]; m = ch.decode_mask(ml["ann"], img.shape[0], img.shape[1])
    edge = [e.reshape(-1, 2) for e in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]]
    ys, xs = np.where(m > 0); pad = 15
    for c, st in enumerate(steps):
        ax = axs[r, c]; idx = list(range(0, n, st)); warp = ch.build_tps_warp(mn[idx], sp[idx])
        ax.imshow(np.where(m[..., None] > 0, img, (0.35 * img + 0.65 * 255).astype(np.uint8)))
        for e in edge:
            ax.plot(e[:, 0], e[:, 1], color="white", lw=1.6)
        wi = ch.warp_grid_to_specimen(grid, warp)
        for i in range(grid["n_rows"]):
            for j in range(grid["n_cols"]):
                if grid["cell_inside"][i, j]:
                    q = np.array([wi[i, j], wi[i, j + 1], wi[i + 1, j + 1], wi[i + 1, j], wi[i, j]])
                    ax.plot(q[:, 0], q[:, 1], color="#22d3ee", lw=0.6)
        o = warp(np.vstack([mean_shape, mean_shape[:1]])); ax.plot(o[:, 0], o[:, 1], color="#facc15", lw=1.2)
        ax.scatter(sp[idx, 0], sp[idx, 1], s=10 if len(idx) > 20 else 28, color="#ef4444", zorder=5, edgecolor="white", lw=0.4)
        ax.set_xlim(xs.min() - pad, xs.max() + pad); ax.set_ylim(ys.max() + pad, ys.min() - pad); ax.axis("off")
        if r == 0:
            ax.set_title(f"{len(idx)} TPS control points", fontsize=10)
    axs[r, 0].text(-0.02, 0.5, fn.split("__")[0].replace("_", " "), transform=axs[r, 0].transAxes, rotation=90,
                   va="center", ha="right", fontsize=9, style="italic")
fig.text(0.5, 0.005, "white: specimen mask (AI segmentation)   yellow: consensus outline warped by the TPS   cyan: grid cells   red: control points",
         ha="center", fontsize=8)
plt.tight_layout(rect=(0.02, 0.02, 1, 1)); plt.savefig(a.out, dpi=160); print(a.out)
