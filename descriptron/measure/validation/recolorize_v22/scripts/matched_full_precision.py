"""matched_colours_<V>__<R>.csv and per_image_pair_<V>__<R>.csv (format of compare_palettes.py) from V22's own
_palette.csv (5-decimal sRGB) and _labels.png, i.e. without the 8-bit rounding of the recoloured PNG.
Usage: matched_full_precision.py <R dir> <V22 dir> <out_dir> <Vname> <Rname>"""
import sys, os, glob, numpy as np, pandas as pd, cv2
from skimage.color import rgb2lab, deltaE_ciede2000, deltaE_cie76
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
Rd, Vd, out, VN, RN = sys.argv[1:6]
P = pd.read_csv(os.path.join(Rd, "palettes.csv")); M, Q = [], []
for img, g in P.groupby("image"):
    v = glob.glob(os.path.join(Vd, f"*_{img}_palette.csv"))
    if not v: continue
    V = pd.read_csv(v[0]); V = V[V.n_pixels > 0]; g = g[g["size"] > 0]
    lR = rgb2lab(g[["R", "G", "B"]].values.reshape(-1, 1, 3)).reshape(-1, 3)
    lV = rgb2lab(V[["R01", "G01", "B01"]].values.reshape(-1, 1, 3)).reshape(-1, 3)
    D = deltaE_ciede2000(lV[:, None, :], lR[None, :, :]); vi, ri = linear_sum_assignment(D)
    fV = V.n_pixels.values / V.n_pixels.sum(); fR = g["size"].values / g["size"].sum()
    for i, j in zip(vi, ri):
        M.append(dict(image=img, species=img.split("__")[0], clusterA=int(V.colour.values[i]), clusterB=int(g.cluster.values[j]),
                      dE00=D[i, j], dE76=float(deltaE_cie76(lV[i], lR[j])), fracA=fV[i], fracB=fR[j]))
    aR = cv2.imread(os.path.join(Rd, f"{img}_assign.png"), cv2.IMREAD_UNCHANGED); aR = aR[..., 0] if aR.ndim == 3 else aR
    aV = cv2.imread(v[0].replace("_palette.csv", "_labels.png"), cv2.IMREAD_UNCHANGED); fg = aV > 0
    mp = {int(V.colour.values[i]): int(g.cluster.values[j]) for i, j in zip(vi, ri)}
    Q.append(dict(image=img, species=img.split("__")[0], kA=len(V), kB=len(g),
                  pixel_agreement=float(np.mean(np.vectorize(lambda x: mp.get(int(x), -1))(aV[fg]) == aR[fg])),
                  ARI=adjusted_rand_score(aR[fg], aV[fg]), matched_dE00_median=float(np.median(D[vi, ri]))))
pd.DataFrame(M).to_csv(os.path.join(out, f"matched_colours_{VN}__{RN}.csv"), index=False)
pd.DataFrame(Q).to_csv(os.path.join(out, f"per_image_pair_{VN}__{RN}.csv"), index=False)
d = pd.DataFrame(M); print(VN, RN, len(Q), "images; matched dE00 median %.4f max %.4f" % (d.dE00.median(), d.dE00.max()))
