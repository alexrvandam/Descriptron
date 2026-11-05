#!/usr/bin/env python3
"""Per-image comparison of colour maps from Descriptron (FHS_and_CLAHE_V21.py labavg_* output) and recolorize.

Both programs' outputs are read as they were written; nothing is re-clustered here.
  Descriptron: the labavg_* PNG is a flat-colour image; within the _bin_ mask each distinct RGB value is one cluster.
  recolorize : <stem>_assign.png (cluster index per pixel, 0 = background) + palettes.csv (sRGB 0-1 centres).
All colours are converted to CIELAB (D65) with skimage for both programs (the skimage vs grDevices difference is
<= 0.45 dE76, see compare/conversion/).

--method NAME=descriptron:<dir>:<glob with {fg}>[:<glob of the image that was clustered, with {fg}>]
--method NAME=recolorize:<dir>
--pair A,B   (repeatable)

Writes to --out_dir:
  palettes_<NAME>.csv        image, species, cluster, L, a, b, n_px, frac          (one row per palette colour)
  per_image_<NAME>.csv       image, k, fidelity_dE00_mean (input pixel vs its palette colour) [, centre_vs_pixelmean]
  matched_colours_<A>__<B>.csv  one row per Hungarian-matched colour pair (dE00, dE76, sizes)
  per_image_pair_<A>__<B>.csv   kA, kB, pixel agreement, ARI, per-pixel dE00 between the two recoloured images
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2lab, deltaE_ciede2000, deltaE_cie76
from sklearn.metrics import adjusted_rand_score


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"))


def lab_of(rgb_u8_or_float):
    x = np.asarray(rgb_u8_or_float, dtype=float)
    if x.max() > 1.0:
        x = x / 255.0
    return rgb2lab(x.reshape(-1, 1, 3)).reshape(-1, 3)


def wmedian(values, weights):
    o = np.argsort(values); v = values[o]; w = weights[o]; c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def load_descriptron(d, pattern, fg, m):
    hits = glob.glob(os.path.join(d, pattern.replace("{fg}", fg)))
    if len(hits) != 1:
        return None
    img = load_rgb(hits[0])
    px = img[m]
    uniq, inv = np.unique(px, axis=0, return_inverse=True)
    lab = np.zeros(img.shape[:2], np.int32); lab[m] = inv.ravel() + 1
    return dict(labels=lab, pal_lab=lab_of(uniq), file=hits[0])


def load_recolorize(d, stem, m, pal_df):
    p = os.path.join(d, stem + "_assign.png")
    if not os.path.isfile(p) or stem not in pal_df.groups:
        return None
    a = np.array(Image.open(p)).astype(np.int32)
    if a.ndim == 3:
        a = a[:, :, 0]
    g = pal_df.get_group(stem).sort_values("cluster")
    present = np.unique(a[m]); present = present[present > 0]
    # relabel to 1..k over the clusters actually used inside the mask
    newlab = np.zeros_like(a); pal = []
    for i, c in enumerate(present, 1):
        newlab[(a == c) & m] = i
        row = g[g.cluster == c].iloc[0]; pal.append([row.R, row.G, row.B])
    if ((a > 0) != m).any():
        print("WARN recolorize background differs from mask", stem, int(((a > 0) != m).sum()), file=sys.stderr)
    return dict(labels=newlab, pal_lab=lab_of(np.array(pal, float)), file=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--method", action="append", required=True); ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    methods = {}
    for spec in a.method:
        name, rest = spec.split("=", 1); parts = rest.split(":")
        if parts[0] == "descriptron":
            methods[name] = dict(kind="d", dir=parts[1], pattern=parts[2], clustered=parts[3] if len(parts) > 3 else None)
        else:
            pal = pd.read_csv(os.path.join(parts[1], "palettes.csv"))
            methods[name] = dict(kind="r", dir=parts[1], pal=pal.groupby("image"))
    pairs = [tuple(p.split(",")) for p in a.pair]
    fgs = sorted(f for f in os.listdir(a.pairs_dir) if "_fg_" in f and f.endswith(".png"))
    if a.limit:
        fgs = fgs[:a.limit]
    pal_rows = {n: [] for n in methods}; img_rows = {n: [] for n in methods}
    match_rows = {p: [] for p in pairs}; pair_rows = {p: [] for p in pairs}; raw_rows = []
    for ii, fg in enumerate(fgs):
        stem = os.path.splitext(fg)[0]; species = fg.split("__")[0]
        raw = load_rgb(os.path.join(a.pairs_dir, fg))
        mk = np.array(Image.open(os.path.join(a.pairs_dir, fg.replace("_fg_", "_bin_", 1))))
        m = (mk[:, :, 0] if mk.ndim == 3 else mk) > 0
        raw_lab = lab_of(raw[m])
        raw_rows.append((stem, species, *raw_lab.mean(0), int(m.sum())))
        res = {}
        for n, md in methods.items():
            r = load_descriptron(md["dir"], md["pattern"], fg, m) if md["kind"] == "d" else load_recolorize(md["dir"], stem, m, md["pal"])
            if r is None:
                continue
            L = r["labels"][m]; k = len(r["pal_lab"])
            n_px = np.bincount(L, minlength=k + 1)[1:]
            r["n_px"] = n_px; r["Lfg"] = L; res[n] = r
            for c in range(k):
                pal_rows[n].append((stem, species, c + 1, *r["pal_lab"][c], int(n_px[c]), n_px[c] / m.sum()))
            fid = deltaE_ciede2000(raw_lab, r["pal_lab"][L - 1])
            row = dict(image=stem, species=species, k=k, fidelity_dE00_mean=float(fid.mean()),
                       fidelity_dE00_median=float(np.median(fid)))
            if md.get("clustered"):
                cp = glob.glob(os.path.join(md["dir"], md["clustered"].replace("{fg}", fg)))
                if len(cp) == 1:
                    cl = lab_of(load_rgb(cp[0])[m])
                    pm = np.array([cl[L == c + 1].mean(0) for c in range(k)])
                    dcent = deltaE_ciede2000(r["pal_lab"], pm)
                    row["centre_vs_pixelmean_dE00_max"] = float(dcent.max())
                    row["centre_vs_pixelmean_dE00_wmean"] = float(np.average(dcent, weights=n_px))
                    row["centre_vs_pixelmean_dE00_list"] = ";".join(f"{x:.2f}" for x in dcent)
                    row["cluster_frac_list"] = ";".join(f"{x:.4f}" for x in n_px / m.sum())
            img_rows[n].append(row)
        for (A, B) in pairs:
            if A not in res or B not in res:
                continue
            pa, pb = res[A]["pal_lab"], res[B]["pal_lab"]
            C00 = deltaE_ciede2000(np.repeat(pa, len(pb), 0), np.tile(pb, (len(pa), 1))).reshape(len(pa), len(pb))
            C76 = deltaE_cie76(np.repeat(pa, len(pb), 0), np.tile(pb, (len(pa), 1))).reshape(len(pa), len(pb))
            ri, ci = linear_sum_assignment(C00)
            for i, j in zip(ri, ci):
                match_rows[(A, B)].append((stem, species, i + 1, j + 1, C00[i, j], C76[i, j],
                                           res[A]["n_px"][i] / m.sum(), res[B]["n_px"][j] / m.sum()))
            LA, LB = res[A]["Lfg"], res[B]["Lfg"]
            mp = -np.ones(len(pa) + 1, int); mp[ri + 1] = ci + 1
            agree = float((mp[LA] == LB).mean())
            ct = np.zeros((len(pa) + 1, len(pb) + 1)); np.add.at(ct, (LA, LB), 1); ct = ct[1:, 1:]
            nz = ct > 0
            pix_de = C00[nz]; w = ct[nz]
            # nearest-colour (not 1:1) distance, pixel weighted, both directions
            nnA = float(np.average(C00.min(1), weights=res[A]["n_px"])); nnB = float(np.average(C00.min(0), weights=res[B]["n_px"]))
            pair_rows[(A, B)].append(dict(image=stem, species=species, kA=len(pa), kB=len(pb), pixel_agreement=agree,
                                          ARI=float(adjusted_rand_score(LA, LB)),
                                          pixel_dE00_mean=float(np.average(pix_de, weights=w)), pixel_dE00_median=wmedian(pix_de, w),
                                          matched_dE00_median=float(np.median(C00[ri, ci])), nearest_dE00_A_to_B=nnA, nearest_dE00_B_to_A=nnB))
        if ii % 25 == 0:
            print(ii, stem, {n: len(res[n]["pal_lab"]) for n in res}, flush=True)
    for n in methods:
        pd.DataFrame(pal_rows[n], columns=["image", "species", "cluster", "L", "a", "b", "n_px", "frac"]).to_csv(os.path.join(a.out_dir, f"palettes_{n}.csv"), index=False)
        pd.DataFrame(img_rows[n]).to_csv(os.path.join(a.out_dir, f"per_image_{n}.csv"), index=False)
    for p in pairs:
        pd.DataFrame(match_rows[p], columns=["image", "species", "clusterA", "clusterB", "dE00", "dE76", "fracA", "fracB"]).to_csv(
            os.path.join(a.out_dir, f"matched_colours_{p[0]}__{p[1]}.csv"), index=False)
        pd.DataFrame(pair_rows[p]).to_csv(os.path.join(a.out_dir, f"per_image_pair_{p[0]}__{p[1]}.csv"), index=False)
    pd.DataFrame(raw_rows, columns=["image", "species", "L", "a", "b", "n_fg"]).to_csv(os.path.join(a.out_dir, "raw_mean_lab.csv"), index=False)
    print("done", len(fgs), "images")


if __name__ == "__main__":
    main()
