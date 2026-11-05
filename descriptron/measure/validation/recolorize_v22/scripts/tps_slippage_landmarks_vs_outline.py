"""Slippage of a TPS homology frame with three control sets, on the same specimens and the same TPS
(color_phenomics_homology_v2_1.build_tps_warp, consensus -> specimen):
  landmarks   the fixed landmarks only (the information patternize's patLan* functions warp with)
  outline     every --step-th outline semilandmark (Descriptron before 4 Nov 2026)
  both        outline semilandmarks + landmarks (homology_landmarks_v1, Descriptron from 4 Nov 2026)
Margin slippage: the consensus outline (1000 points) warped onto the specimen vs the specimen's mask from the COCO file:
  IoU, share of the warped area outside the mask, mean edge distance (% of sqrt(mask area)).
Interior slippage: each landmark left out in turn (from 'landmarks' and 'both'; 'outline' never uses it), its consensus
position warped onto the specimen and compared with where it really is (% of sqrt(mask area)).
Consensus landmark positions and the per-specimen landmarks come from homology_landmarks_v1.LandmarkControls (median,
mirror images by outline winding; specimens whose landmarks disagree with their outline are left out and listed).
Usage: tps_slippage_landmarks_vs_outline.py --gpa_dir <semilandmarks/cat> --json <coco with the masks>
       --landmark_json <coco with keypoints> --category_name <cat> --measure_dir <gui/measure> --out_dir <dir> [--step 5]"""
import argparse, os, sys, json
import numpy as np, pandas as pd, cv2

ap = argparse.ArgumentParser()
for k in ("--gpa_dir", "--json", "--landmark_json", "--category_name", "--measure_dir", "--out_dir"):
    ap.add_argument(k, required=True)
ap.add_argument("--step", type=int, default=5)
ap.add_argument("--both_steps", default="", help="extra outline spacings for the combined set, e.g. 10,20")
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
sys.path.insert(0, a.measure_dir)
import color_phenomics_homology_v2_1 as ch
from homology_landmarks_v1 import LandmarkControls

mean_shape, specimens = ch.load_gpa_outputs(a.gpa_dir)
lookup, _ = ch.build_mask_lookup(a.json, a.category_name)
lc = LandmarkControls.from_coco(a.landmark_json, specimens, mean_shape)
assert lc.n_used, "no landmarks on this structure"
n = len(mean_shape); oidx = list(range(0, n, a.step)); extra_steps = [int(x) for x in a.both_steps.split(",") if x.strip()]


def resample(c, m):
    c = np.vstack([c, c[:1]]); d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    t = np.linspace(0, d[-1], m, endpoint=False)
    return np.column_stack([np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])])


dense = resample(mean_shape, 1000); rows, lrows = [], []
for fn, sp in specimens.items():
    k = lc.n_for(fn)
    ml = lookup.get(ch.get_original_filename(fn))
    if not k or not ml:
        continue
    h, w = int(ml["img"]["height"]), int(ml["img"]["width"]); m = ch.decode_mask(ml["ann"], h, w).astype(bool)
    if m.sum() < 100:
        continue
    edge = np.vstack([e.reshape(-1, 2) for e in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]]).astype(float)
    scale = np.sqrt(m.sum()); sp = sp[:n]
    lm = lc.spec_lm[fn]; ok = np.where(np.isfinite(lm[:, 0]) & np.isfinite(lc.mean_lm[:, 0]))[0]
    sets = {"landmarks": (lc.mean_lm[ok], lm[ok]), "outline": (mean_shape[oidx], sp[oidx]),
            "both": (np.vstack([mean_shape[oidx], lc.mean_lm[ok]]), np.vstack([sp[oidx], lm[ok]]))}
    for bs in extra_steps:
        bi = list(range(0, n, bs)); sets[f"both_step{bs}"] = (np.vstack([mean_shape[bi], lc.mean_lm[ok]]), np.vstack([sp[bi], lm[ok]]))
    for name, (cm, cs) in sets.items():
        poly = ch.build_tps_warp(cm, cs)(dense)
        pm = np.zeros((h, w), np.uint8); cv2.fillPoly(pm, [np.round(poly).astype(np.int32)], 1); pm = pm.astype(bool)
        dmin = np.concatenate([np.sqrt(((poly[i:i + 200, None] - edge[None]) ** 2).sum(-1)).min(1) for i in range(0, len(poly), 200)])
        rows.append(dict(specimen=fn, controls=name, n_controls=len(cm), IoU=(pm & m).sum() / (pm | m).sum(),
                         outside_share=(pm & ~m).sum() / max(pm.sum(), 1), edge_error_pct=100 * dmin.mean() / scale))
    for j in ok:                                           # leave landmark j out
        rest = [i for i in ok if i != j]
        lo = {"landmarks": (lc.mean_lm[rest], lm[rest]), "outline": (mean_shape[oidx], sp[oidx]),
              "both": (np.vstack([mean_shape[oidx], lc.mean_lm[rest]]), np.vstack([sp[oidx], lm[rest]]))}
        for bs in extra_steps:
            bi = list(range(0, n, bs)); lo[f"both_step{bs}"] = (np.vstack([mean_shape[bi], lc.mean_lm[rest]]), np.vstack([sp[bi], lm[rest]]))
        for name, (cm, cs) in lo.items():
            pred = ch.build_tps_warp(cm, cs)(lc.mean_lm[[j]])[0]
            lrows.append(dict(specimen=fn, controls=name, landmark=lc.names[j], error_pct=100 * np.linalg.norm(pred - lm[j]) / scale))
R = pd.DataFrame(rows); L = pd.DataFrame(lrows)
R.to_csv(os.path.join(a.out_dir, "margin_slippage.csv"), index=False); L.to_csv(os.path.join(a.out_dir, "interior_slippage_leave_one_landmark_out.csv"), index=False)
S = R.groupby("controls")[["IoU", "outside_share", "edge_error_pct"]].median().join(L.groupby("controls").error_pct.median().rename("interior_error_pct_median"))
S["n_specimens"] = R.groupby("controls").specimen.nunique(); S = S.loc[["landmarks", "outline", "both"] + [f"both_step{b}" for b in extra_steps]]
S.to_csv(os.path.join(a.out_dir, "slippage_summary.csv"))
json.dump({"left_out_disagreeing": lc.info.get("outline_landmark_disagree", [])}, open(os.path.join(a.out_dir, "slippage_info.json"), "w"), indent=1)
print(S.round(4).to_string())
