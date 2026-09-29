#!/usr/bin/env python3
"""Colour-pattern benchmark: Descriptron homology-cell colour vs patternize vs Colormesh.

Two sub-commands, no hard-coded paths:

  prepare  -- choose the common wing set (Descriptron colour row AND 17 landmarks),
              attach species labels, flag mirror images, impute a missing landmark
              (if any), write the inputs the two R packages need:
                * images/<id>.png         square, padded (right/bottom, edge-replicate)
                                          copies of the TIFFs, NOT rotated or flipped
                * landmarks/<id>_landmarks.txt   17 landmarks, "x y" pixel coords (y down)
                * colormesh_landmarks.csv  17 landmarks + outline semilandmarks (long format)
                * colormesh_perimeter.json perimeter map / main landmarks
                * outlines/<id>_outline.txt      whole_wing outline, pixel coords (y down)
                * specimens.csv            the fixed order used by every method
                * descriptron_features.csv L*a*b* means of the homology cells, same order

  analyse  -- read the per-method feature tables (same order is asserted), run a
              covariance PCA per method, compute LOO 1-NN species accuracy,
              between-species share of variance, Procrustes agreement with a
              permutation test, Spearman on PC1, and draw the 3-panel figure.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- helpers
def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def procrustes_fit(src, dst, allow_reflection=True):
    """Similarity transform mapping src -> dst (least squares). Returns function."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, s, vt = np.linalg.svd(a.T @ b)
    r = u @ vt
    if not allow_reflection and np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
        s[-1] *= -1
    scale = s.sum() / (a ** 2).sum()
    return lambda q: (q - mu_s) @ r * scale + mu_d


def gpa(shapes, n_iter=20):
    """Plain GPA with reflection allowed (used only to impute a missing landmark and to
    pick the patternize cartoon specimen)."""
    X = [s - s.mean(0) for s in shapes]
    X = [s / np.sqrt((s ** 2).sum()) for s in X]
    mean = X[0].copy()
    for _ in range(n_iter):
        X = [procrustes_fit(s, mean)(s) for s in X]
        new = np.mean(X, 0)
        new -= new.mean(0)
        new /= np.sqrt((new ** 2).sum())
        if np.abs(new - mean).max() < 1e-10:
            mean = new
            break
        mean = new
    return mean, X


def densify(poly, step=1.0):
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        L = np.linalg.norm(b - a)
        k = max(1, int(np.ceil(L / step)))
        for t in np.arange(k) / k:
            out.append(a + t * (b - a))
    return np.array(out)


# ----------------------------------------------------------------------------- prepare
def prepare(args):
    import cv2

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    for sub in ("images", "landmarks", "outlines", "qc"):
        os.makedirs(os.path.join(args.outdir, sub), exist_ok=True)

    kpj = json.load(open(args.landmarks))
    img_by_name = {i["file_name"]: i for i in kpj["images"]}
    kp = {a["image_id"]: np.array(a["keypoints"], float).reshape(-1, 3) for a in kpj["annotations"]}

    col = pd.read_csv(args.descriptron_csv)
    col["stem"] = col["filename"].str.replace(r"_\d+$", "", regex=True)
    assert not col["stem"].duplicated().any(), "duplicate Descriptron rows per image"
    common = col[col["stem"].isin(img_by_name)].copy()

    lab = pd.read_csv(args.labels)
    lab = dict(zip(lab["filename"], lab["group_label"]))
    common["species"] = common["stem"].map(lab)
    miss = common["species"].isna().sum()
    if miss:
        sys.exit(f"{miss} wings have no species label in {args.labels}")
    common["id"] = common["stem"].str.replace(r"\.tif$", "", regex=True)
    assert common["id"].str.match(r"^[A-Za-z0-9_]+$").all(), "ids must be plain for Colormesh"
    common = common.sort_values(["species", "id"]).reset_index(drop=True)
    n = len(common)
    print(f"common wings: {n}")

    # --- landmarks, missing-landmark imputation
    L = np.stack([kp[img_by_name[s]["id"]] for s in common["stem"]])  # n x 17 x 3
    vis = L[:, :, 2] > 0
    complete = vis.all(1)
    mean, _ = gpa([L[i, :, :2] for i in np.where(complete)[0]])
    imputed = []
    for i in np.where(~complete)[0]:
        ok = vis[i]
        f = procrustes_fit(mean[ok], L[i, ok, :2], allow_reflection=True)
        est = f(mean)
        for j in np.where(~ok)[0]:
            imputed.append((str(common.at[i, "id"]), int(j + 1), float(est[j, 0]), float(est[j, 1])))
            L[i, j, :2] = est[j]
    lm = L[:, :, :2]

    # --- mirror flags (sign of signed area vs majority), image coords (y down)
    sa = np.array([signed_area(p) for p in lm])
    maj = np.sign(np.sign(sa).sum())
    common["signed_area_sign"] = np.sign(sa).astype(int)
    common["mirror_vs_majority"] = (np.sign(sa) != maj).astype(int)

    # --- specimen closest to the reflection-allowed consensus (patternize cartoon)
    mean_all, aligned = gpa([p for p in lm])
    pd_to_mean = np.array([np.sqrt(((a - mean_all) ** 2).sum()) for a in aligned])
    cartoon_id = common.at[int(np.argmin(pd_to_mean)), "id"]

    # --- images: pad to a common square canvas (right/bottom), write PNG
    sizes = []
    for s in common["stem"]:
        im = img_by_name[s]
        sizes.append((im["width"], im["height"]))
    S = int(max(max(w, h) for w, h in sizes))
    for k, row in common.iterrows():
        img = cv2.imread(os.path.join(args.images, row["stem"]), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        assert (w, h) == sizes[k], (row["stem"], (w, h), sizes[k])
        pad = cv2.copyMakeBorder(img, 0, S - h, 0, S - w, cv2.BORDER_REPLICATE)
        cv2.imwrite(os.path.join(args.outdir, "images", row["id"] + ".png"), pad)
        np.savetxt(os.path.join(args.outdir, "landmarks", row["id"] + "_landmarks.txt"),
                   lm[k], fmt="%.3f")

    # --- whole_wing outlines and semilandmarks for Colormesh
    u = json.load(open(args.outline_coco))
    cid = [c["id"] for c in u["categories"] if c["name"] == args.outline_category]
    assert len(cid) == 1
    uim = {i["id"]: i["file_name"] for i in u["images"]}
    outl = {}
    for a in u["annotations"]:
        if a["category_id"] == cid[0] and uim[a["image_id"]] in set(common["stem"]):
            polys = [np.array(p, float).reshape(-1, 2) for p in a["segmentation"]]
            big = max(polys, key=lambda p: abs(signed_area(p)))
            prev = outl.get(uim[a["image_id"]])
            if prev is None or abs(signed_area(big)) > abs(signed_area(prev)):
                outl[uim[a["image_id"]]] = big
    missing_outline = [s for s in common["stem"] if s not in outl]
    if missing_outline:
        sys.exit(f"no {args.outline_category} outline for {missing_outline}")

    margin = [i - 1 for i in args.perimeter_landmarks]  # 0-based, in perimeter order
    nm = len(margin)
    dens, idxs = {}, {}
    seg_len = np.zeros((n, nm))
    bad_order = []
    for k, row in common.iterrows():
        P = densify(outl[row["stem"]], 1.0)
        pos = [int(np.argmin(((P - lm[k, j]) ** 2).sum(1))) for j in margin]
        # direction in which the landmarks appear in perimeter order
        chosen = None
        for d in (1, -1):
            steps = [((pos[(t + 1) % nm] - pos[t]) * d) % len(P) for t in range(nm)]
            if sum(steps) == len(P):  # one full turn -> cyclic order respected
                chosen = d
                break
        if chosen is None:
            bad_order.append(row["id"])
            chosen = 1
        dens[k], idxs[k] = (P, chosen), pos
        for t in range(nm):
            steps = ((pos[(t + 1) % nm] - pos[t]) * chosen) % len(P)
            seg_len[k, t] = steps  # 1 px spacing -> arc length in px
    # allocate args.n_semilandmarks proportional to mean arc length, >=1 per segment
    rel = seg_len.mean(0) / seg_len.mean(0).sum()
    alloc = np.maximum(1, np.floor(rel * args.n_semilandmarks)).astype(int)
    while alloc.sum() < args.n_semilandmarks:
        alloc[np.argmax(rel * args.n_semilandmarks - alloc)] += 1
    while alloc.sum() > args.n_semilandmarks:
        alloc[np.argmax(alloc)] -= 1
    rows = []
    semi_rows = []
    for k, row in common.iterrows():
        P, d = dens[k]
        pos = idxs[k]
        pts = [lm[k, j] for j in range(lm.shape[1])]  # rows 1..17 = landmarks
        perim_map = []
        nxt = lm.shape[1]
        for t in range(nm):
            perim_map.append(margin[t] + 1)
            steps = ((pos[(t + 1) % nm] - pos[t]) * d) % len(P)
            for q in range(1, alloc[t] + 1):
                off = int(round(q * steps / (alloc[t] + 1)))
                pts.append(P[(pos[t] + d * off) % len(P)])
                nxt += 1
                perim_map.append(nxt)
        pts = np.array(pts)
        for j, (x, y) in enumerate(pts):
            rows.append({"id": row["id"], "point": j + 1, "x": x, "y_img": y, "y_tps": S - y})
        np.savetxt(os.path.join(args.outdir, "outlines", row["id"] + "_outline.txt"),
                   outl[row["stem"]], fmt="%.2f")
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "colormesh_landmarks.csv"), index=False)
    json.dump({"perimeter_map": perim_map, "main_landmarks": list(range(1, lm.shape[1] + 1)),
               "n_landmarks": int(lm.shape[1]), "n_semilandmarks": int(alloc.sum()),
               "semilandmarks_per_segment": alloc.tolist(),
               "perimeter_landmarks": args.perimeter_landmarks, "canvas": S},
              open(os.path.join(args.outdir, "colormesh_perimeter.json"), "w"), indent=1)

    # --- Descriptron features in the same order
    cells = sorted({m.group(1) for c in col.columns for m in [re.match(r"(r\d+c\d+)_L_mean_abs$", c)] if m})
    feats = [f"{c}_{ch}_mean_abs" for c in cells for ch in ("L", "a", "b")]
    D = common[["id"] + feats].copy()
    D.to_csv(os.path.join(args.outdir, "descriptron_features.csv"), index=False)
    if args.descriptron_pca_csv:  # sensitivity: the pipeline's own PC1-5 (all wings, own features)
        own = pd.read_csv(args.descriptron_pca_csv).set_index("filename").loc[common["filename"]]
        pcs = [f"PC{i}" for i in range(1, 6)]
        E = own[pcs].reset_index(drop=True)
        E.insert(0, "id", common["id"].values)
        E.to_csv(os.path.join(args.outdir, "descriptron_ownPCA_PC1to5.csv"), index=False)

    spec = common[["id", "stem", "filename", "species", "signed_area_sign", "mirror_vs_majority"]].copy()
    spec["order"] = np.arange(1, n + 1)
    spec["patternize_cartoon"] = (spec["id"] == cartoon_id).astype(int)
    spec["n_descriptron_nan"] = common[feats].isna().sum(1).values
    spec.to_csv(os.path.join(args.outdir, "specimens.csv"), index=False)
    info = {"n_wings": n, "n_cells": len(cells), "n_descriptron_features": len(feats),
            "descriptron_nan_total": int(common[feats].isna().sum().sum()),
            "mirror_count": int(spec["mirror_vs_majority"].sum()),
            "majority_sign": int(maj), "imputed_landmarks": imputed, "canvas_px": S,
            "image_sizes": {f"{w}x{h}": sizes.count((w, h)) for (w, h) in set(sizes)},
            "cartoon_id": cartoon_id, "outline_order_problems": bad_order,
            "species_counts": {k: int(v) for k, v in common["species"].value_counts().items()},
            "prepare_seconds": round(time.time() - t0, 1)}
    json.dump(info, open(os.path.join(args.outdir, "prepare_info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))

    # QC overlay of landmarks + semilandmarks on 3 wings (one mirrored)
    lmdf = pd.DataFrame(rows)
    qc_ids = list(spec["id"][:2]) + list(spec.loc[spec.mirror_vs_majority == 1, "id"][:1])
    for i in qc_ids:
        img = cv2.imread(os.path.join(args.outdir, "images", i + ".png"))
        sub = lmdf[lmdf.id == i]
        for _, r in sub.iterrows():
            c = (0, 0, 255) if r.point <= lm.shape[1] else (0, 200, 0)
            cv2.circle(img, (int(r.x), int(r.y_img)), 6 if r.point <= lm.shape[1] else 4, c, -1)
            if r.point <= lm.shape[1]:
                cv2.putText(img, str(int(r.point)), (int(r.x) + 6, int(r.y_img) - 6), 0, 0.9, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(args.outdir, "qc", f"prep_landmarks_{i}.jpg"), img)


# ----------------------------------------------------------------------------- analyse
def pca_scores(X):
    Xc = X - X.mean(0)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    scores = u * s
    var = s ** 2 / (len(X) - 1)
    keep = var > 1e-12 * var.sum()
    return scores[:, keep], var[keep] / var.sum()


def loo_1nn(Z, y, eligible):
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    idx = np.where(eligible)[0]
    # neighbours may be any wing (incl. singleton species); only eligible wings are scored
    pred = y[np.argmin(D[idx], 1)]
    return float((pred == y[idx]).mean()), int(len(idx))


def between_share(Z, y):
    tot = ((Z - Z.mean(0)) ** 2).sum()
    btw = 0.0
    for g in np.unique(y):
        m = y == g
        btw += m.sum() * ((Z[m].mean(0) - Z.mean(0)) ** 2).sum()
    return float(btw / tot)


def procrustes_r(A, B):
    """Symmetric Procrustes correlation (PROTEST): sqrt(1 - m12^2)."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    A = A / np.sqrt((A ** 2).sum())
    B = B / np.sqrt((B ** 2).sum())
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return float(s.sum())


def analyse(args):
    from scipy.stats import spearmanr

    spec = pd.read_csv(args.specimens)
    ids = spec["id"].tolist()
    y = spec["species"].to_numpy()
    counts = pd.Series(y).value_counts()
    eligible = np.array([counts[s] >= 2 for s in y])

    methods = {}
    for spec_str in args.method:
        name, path = spec_str.split("=", 1)
        df = pd.read_csv(path)
        assert df.columns[0] == "id", f"{path}: first column must be id"
        assert df["id"].tolist() == ids, f"{name}: wings not in the same order as specimens.csv"
        X = df.drop(columns=["id"]).to_numpy(float)
        if np.isnan(X).any():
            nn = int(np.isnan(X).sum())
            print(f"{name}: {nn} NaN values -> column mean imputation")
            cm = np.nanmean(X, 0)
            X = np.where(np.isnan(X), cm, X)
        X = X[:, X.std(0) > 0]
        methods[name] = X

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary, scores = [], {}
    for name, X in methods.items():
        Z, ve = pca_scores(X)
        scores[name] = Z
        k = min(args.n_pcs, Z.shape[1])
        acc5, ne = loo_1nn(Z[:, :k], y, eligible)
        accall, _ = loo_1nn(Z, y, eligible)
        accraw, _ = loo_1nn(X, y, eligible)
        summary.append({"method": name, "n_wings": len(X), "n_features": X.shape[1],
                        "n_pcs": Z.shape[1], "var_PC1": ve[0], "var_PC2": ve[1],
                        "var_PC1to5": ve[:k].sum(),
                        "loo1nn_acc_PC1to5": acc5, "loo1nn_acc_allPCs": accall,
                        "loo1nn_scored_wings": ne,
                        "between_species_share_PC1to5": between_share(Z[:, :k], y),
                        "between_species_share_allPCs": between_share(Z, y)})
        out = pd.DataFrame(Z[:, :min(10, Z.shape[1])],
                           columns=[f"PC{i + 1}" for i in range(min(10, Z.shape[1]))])
        out.insert(0, "species", y)
        out.insert(0, "id", ids)
        out.to_csv(os.path.join(args.outdir, f"pc_scores_{name}.csv"), index=False)
    if args.brightness_csv:
        # diagnostic: how much of each PC1 is overall wing brightness, and a trivial
        # baseline = LOO 1-NN on the whole-wing mean L*, a*, b* (3 numbers per wing)
        B = pd.read_csv(args.brightness_csv)
        assert B["id"].tolist() == ids
        meanL = B.filter(regex=r"_L_mean_abs$").mean(1).to_numpy()
        base = np.c_[meanL, B.filter(regex=r"_a_mean_abs$").mean(1), B.filter(regex=r"_b_mean_abs$").mean(1)]
        for srow in summary:
            srow["spearman_PC1_vs_mean_Lstar"] = spearmanr(scores[srow["method"]][:, 0], meanL)[0]
        bacc, _ = loo_1nn(base - base.mean(0), y, eligible)
        summary.append({"method": "baseline_wing_mean_Lab", "n_wings": len(base), "n_features": 3,
                        "loo1nn_acc_PC1to5": bacc, "loo1nn_acc_allPCs": bacc,
                        "loo1nn_scored_wings": int(eligible.sum()),
                        "between_species_share_PC1to5": between_share(base - base.mean(0), y),
                        "between_species_share_allPCs": between_share(base - base.mean(0), y)})
    pd.DataFrame(summary).to_csv(os.path.join(args.outdir, "summary_per_method.csv"), index=False)

    pairs = []
    names = list(methods)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A = scores[names[i]][:, :args.n_pcs]
            B = scores[names[j]][:, :args.n_pcs]
            r = procrustes_r(A, B)
            perm = np.array([procrustes_r(A, B[rng.permutation(len(B))]) for _ in range(args.n_perm)])
            p = (1 + (perm >= r).sum()) / (args.n_perm + 1)
            rho, prho = spearmanr(scores[names[i]][:, 0], scores[names[j]][:, 0])
            pairs.append({"method_a": names[i], "method_b": names[j], "procrustes_r_PC1to5": r,
                          "procrustes_perm_p": p, "n_perm": args.n_perm,
                          "spearman_PC1": rho, "spearman_PC1_p": prho,
                          "abs_spearman_PC1": abs(rho)})
    pd.DataFrame(pairs).to_csv(os.path.join(args.outdir, "summary_agreement.csv"), index=False)
    print(pd.DataFrame(summary).to_string())
    print(pd.DataFrame(pairs).to_string())

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sp = sorted(set(y), key=lambda s: int(re.sub(r"\D", "", s) or 0))
        pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
               "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        marks = "osD^v<>Ph*"
        fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.4))
        for ax, name in zip(np.atleast_1d(axes), names):
            Z = scores[name]
            ve = next(r for r in summary if r["method"] == name)
            for q, s in enumerate(sp):
                m = y == s
                ax.scatter(Z[m, 0], Z[m, 1], s=34, c=pal[q % 10], marker=marks[q % 10],
                           edgecolor="k", linewidth=0.4, label=s)
            ax.set_xlabel(f"PC1 ({100 * ve['var_PC1']:.1f}%)")
            ax.set_ylabel(f"PC2 ({100 * ve['var_PC2']:.1f}%)")
            ax.set_title(f"{args.labels.get(name, name) if isinstance(args.labels, dict) else name}\n"
                         f"LOO 1-NN (PC1-5) = {ve['loo1nn_acc_PC1to5']:.2f}", fontsize=10)
            ax.axhline(0, c="0.85", lw=0.6, zorder=0)
            ax.axvline(0, c="0.85", lw=0.6, zorder=0)
        np.atleast_1d(axes)[-1].legend(title="species", fontsize=8, bbox_to_anchor=(1.02, 1),
                                       loc="upper left", frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.outdir, f"colour_benchmark_pca.{ext}"), dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--images", required=True)
    p.add_argument("--landmarks", required=True, help="COCO keypoints JSON (17 landmarks)")
    p.add_argument("--descriptron_csv", required=True)
    p.add_argument("--labels", required=True, help="CSV filename,group_label")
    p.add_argument("--outline_coco", required=True)
    p.add_argument("--descriptron_pca_csv", default=None,
                   help="optional: Descriptron's own PCA table (sensitivity analysis)")
    p.add_argument("--outline_category", default="whole_wing")
    p.add_argument("--perimeter_landmarks", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   help="landmarks on the wing margin, in perimeter order (1-based)")
    p.add_argument("--n_semilandmarks", type=int, default=55)
    p.add_argument("--outdir", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--specimens", required=True)
    a.add_argument("--method", action="append", required=True, help="name=features.csv (first col id)")
    a.add_argument("--n_pcs", type=int, default=5)
    a.add_argument("--n_perm", type=int, default=999)
    a.add_argument("--seed", type=int, default=20260928)
    a.add_argument("--figure", action="store_true")
    a.add_argument("--brightness_csv", default=None,
                   help="optional Descriptron L*a*b* table: PC1-vs-brightness diagnostic + trivial baseline")
    a.add_argument("--outdir", required=True)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    else:
        args.labels = None
        analyse(args)


if __name__ == "__main__":
    main()
