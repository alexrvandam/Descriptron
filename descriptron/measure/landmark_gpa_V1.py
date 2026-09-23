#!/usr/bin/env python
"""
landmark_gpa_V1.py — Generalized Procrustes Analysis on COCO keypoints

Reads a COCO keypoints JSON, performs full GPA (Bookstein/Procrustes),
and produces the same suite of downstream analyses as V34 does for
semi-landmarks but working directly on discrete homologous landmarks.

Analyses:
  - GPA alignment (translate → scale → rotate, iterated to convergence)
  - PCA of Procrustes shape coordinates
  - UMAP of shape coordinates
  - Centroid size statistics & ANOVA (if groups)
  - MANOVA on Procrustes coordinates (if groups)
  - CVA (Canonical Variate Analysis) with pairwise Mahalanobis heatmap (if ≥3 groups)
  - Form space PCA (shape + log centroid size)
  - Allometry regression (shape on log centroid size)
  - Shape-difference wireframes (mean ± PC deformations)
  - Hierarchical dendrogram

Usage:
  python landmark_gpa_V1.py \
    --json forewing_keypoints.json \
    --output_dir gpa_output/ \
    --group_labels group_labels.csv \
    --category head_keypoints
"""

import argparse
import json
import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial import procrustes as scipy_procrustes
from sklearn.decomposition import PCA
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─── Procrustes helpers ─────────────────────────────────────────────

def centroid_size(cfg):
    c = cfg.mean(axis=0)
    return np.sqrt(np.sum((cfg - c) ** 2))


def center(cfg):
    return cfg - cfg.mean(axis=0)


def scale_to_unit(cfg):
    cs = centroid_size(cfg)
    if cs < 1e-12:
        return cfg, cs
    return cfg / cs, cs


def rotate_to_target(src, tgt):
    U, _, Vt = np.linalg.svd(tgt.T @ src)
    R = (U @ Vt)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return src @ R.T


def procrustes_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def gpa(configs, max_iter=200, tol=1e-8):
    n = len(configs)
    centered = [center(c) for c in configs]
    scaled = []
    sizes = []
    for c in centered:
        s, cs = scale_to_unit(c)
        scaled.append(s)
        sizes.append(cs)

    mean_shape = scaled[0].copy()
    for iteration in range(max_iter):
        aligned = []
        for s in scaled:
            aligned.append(rotate_to_target(s, mean_shape))
        new_mean = np.mean(aligned, axis=0)
        new_mean, _ = scale_to_unit(new_mean)
        diff = procrustes_distance(new_mean, mean_shape)
        mean_shape = new_mean
        if diff < tol:
            log.info(f"GPA converged at iteration {iteration + 1} (diff={diff:.2e})")
            break
    else:
        log.warning(f"GPA did not converge after {max_iter} iterations (diff={diff:.2e})")

    final_aligned = []
    for s in scaled:
        final_aligned.append(rotate_to_target(s, mean_shape))

    return np.array(final_aligned), mean_shape, np.array(sizes)


# ─── Data loading ────────────────────────────────────────────────────

def load_keypoints(json_path, category_name=None):
    with open(json_path) as f:
        data = json.load(f)

    cats = {c["id"]: c["name"] for c in data["categories"]}
    imgs = {im["id"]: im["file_name"] for im in data["images"]}

    target_cats = {}
    for cid, cname in cats.items():
        if category_name is None or cname == category_name:
            target_cats[cid] = cname

    if not target_cats:
        log.error(f"No matching categories found for '{category_name}'. Available: {list(cats.values())}")
        sys.exit(1)

    results = {}
    for cid, cname in target_cats.items():
        configs = []
        filenames = []
        for ann in data["annotations"]:
            if ann["category_id"] != cid:
                continue
            kps = ann.get("keypoints", [])
            if not kps:
                continue
            n_kp = len(kps) // 3
            arr = np.array(kps).reshape(n_kp, 3)
            vis = arr[:, 2]
            if np.all(vis > 0):
                configs.append(arr[:, :2].astype(np.float64))
                fn = imgs.get(ann["image_id"], f"image_{ann['image_id']}")
                filenames.append(os.path.basename(fn))
            else:
                fn = imgs.get(ann["image_id"], f"image_{ann['image_id']}")
                missing = np.sum(vis == 0)
                log.warning(f"Skipping {fn} — {missing}/{n_kp} landmarks not visible")

        results[cname] = (configs, filenames)
        log.info(f"Category '{cname}': {len(configs)} complete specimens, {arr.shape[0]} landmarks each")

    return results


def load_groups(group_file, filenames):
    if not group_file or not os.path.exists(group_file):
        return None
    df = pd.read_csv(group_file)
    col_fn = df.columns[0]
    col_grp = df.columns[1]
    gmap = dict(zip(df[col_fn], df[col_grp]))
    groups = []
    for fn in filenames:
        g = gmap.get(fn)
        if g is None:
            bn = os.path.splitext(fn)[0]
            for k, v in gmap.items():
                if os.path.splitext(k)[0] == bn:
                    g = v
                    break
        groups.append(g if g is not None else "unknown")
    return groups


# ─── Analyses ────────────────────────────────────────────────────────

def run_pca(aligned, filenames, groups, output_dir, prefix, title_extra=""):
    k = aligned.shape[1]
    X = aligned.reshape(len(aligned), -1)
    n_components = min(X.shape[0] - 1, X.shape[1], 10)
    if n_components < 2:
        log.warning(f"Not enough specimens for PCA (n={X.shape[0]})")
        return None
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    if groups:
        unique_groups = sorted(set(groups))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_groups), 1)))
        for i, g in enumerate(unique_groups):
            idx = [j for j, gg in enumerate(groups) if gg == g]
            ax.scatter(scores[idx, 0], scores[idx, 1], label=g, color=colors[i], s=60, alpha=0.8)
        ax.legend(title="Group")
    else:
        ax.scatter(scores[:, 0], scores[:, 1], s=60, alpha=0.8)

    for i, fn in enumerate(filenames):
        ax.annotate(fn, (scores[i, 0], scores[i, 1]), fontsize=5, alpha=0.6)

    var_exp = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    ax.set_title(f"PCA of Procrustes Coordinates{title_extra}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_pca.png"), dpi=200)
    plt.close(fig)

    pc_df = pd.DataFrame(scores[:, :min(5, n_components)],
                         columns=[f"PC{i+1}" for i in range(min(5, n_components))])
    pc_df.insert(0, "filename", filenames)
    if groups:
        pc_df.insert(1, "group", groups)
    pc_df.to_csv(os.path.join(output_dir, f"{prefix}_pc_scores.csv"), index=False)

    var_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(n_components)],
        "variance_explained": var_exp[:n_components],
        "cumulative": np.cumsum(var_exp[:n_components])
    })
    var_df.to_csv(os.path.join(output_dir, f"{prefix}_variance.csv"), index=False)

    log.info(f"PCA: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%, cumulative top-5={np.sum(var_exp[:5]):.1f}%")
    return pca, scores


def run_umap(aligned, filenames, groups, output_dir, prefix):
    try:
        import umap
    except ImportError:
        log.warning("umap-learn not installed — skipping UMAP")
        return

    X = aligned.reshape(len(aligned), -1)
    if X.shape[0] < 5:
        n_neighbors = max(2, X.shape[0] - 1)
    else:
        n_neighbors = min(15, X.shape[0] - 1)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    if groups:
        unique_groups = sorted(set(groups))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_groups), 1)))
        for i, g in enumerate(unique_groups):
            idx = [j for j, gg in enumerate(groups) if gg == g]
            ax.scatter(embedding[idx, 0], embedding[idx, 1], label=g, color=colors[i], s=60, alpha=0.8)
        ax.legend(title="Group")
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=60, alpha=0.8)

    for i, fn in enumerate(filenames):
        ax.annotate(fn, (embedding[i, 0], embedding[i, 1]), fontsize=5, alpha=0.6)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of Procrustes Coordinates")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_umap.png"), dpi=200)
    plt.close(fig)
    log.info("UMAP plot saved")


def run_manova(aligned, groups, output_dir, prefix):
    try:
        from statsmodels.multivariate.manova import MANOVA
    except ImportError:
        log.warning("statsmodels not available — skipping MANOVA")
        return

    unique = sorted(set(groups))
    if len(unique) < 2:
        log.warning("Need ≥2 groups for MANOVA")
        return

    X = aligned.reshape(len(aligned), -1)
    n_vars = X.shape[1]
    n_obs = X.shape[0]

    if n_vars >= n_obs:
        pca_dim = max(2, n_obs - len(unique) - 1)
        log.info(f"MANOVA: reducing {n_vars} shape vars to {pca_dim} PCs (n={n_obs}, groups={len(unique)})")
        pca = PCA(n_components=pca_dim)
        X = pca.fit_transform(X)
        n_vars = pca_dim

    cols = [f"v{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["group"] = groups

    formula = " + ".join(cols) + " ~ group"
    try:
        m = MANOVA.from_formula(formula, data=df)
        result = m.mv_test()
        out_path = os.path.join(output_dir, f"{prefix}_manova.txt")
        with open(out_path, "w") as f:
            f.write(str(result))
        log.info(f"MANOVA results saved to {out_path}")

        lines = str(result).split("\n")
        for line in lines:
            if "Wilks" in line and "Pr" not in line:
                parts = line.split()
                if len(parts) >= 5:
                    log.info(f"  Wilks' lambda = {parts[1]}, F = {parts[3]}, p = {parts[4]}")
    except Exception as e:
        log.warning(f"MANOVA failed: {e}")


def run_cva(aligned, groups, filenames, output_dir, prefix):
    unique = sorted(set(groups))
    if len(unique) < 3:
        log.info(f"CVA requires ≥3 groups (have {len(unique)}) — skipping")
        return

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X = aligned.reshape(len(aligned), -1)
    n_classes = len(unique)

    if X.shape[1] >= X.shape[0]:
        pca_dim = max(n_classes, X.shape[0] - n_classes - 1)
        pca_dim = min(pca_dim, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=pca_dim)
        X = pca.fit_transform(X)

    lda = LinearDiscriminantAnalysis()
    try:
        cv_scores = lda.fit_transform(X, groups)
    except Exception as e:
        log.warning(f"CVA failed: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 1)))
    for i, g in enumerate(unique):
        idx = [j for j, gg in enumerate(groups) if gg == g]
        if cv_scores.shape[1] >= 2:
            ax.scatter(cv_scores[idx, 0], cv_scores[idx, 1], label=g, color=colors[i], s=60, alpha=0.8)
        else:
            ax.scatter(cv_scores[idx, 0], np.zeros(len(idx)), label=g, color=colors[i], s=60, alpha=0.8)

    ax.legend(title="Group")
    ax.set_xlabel("CV1")
    ax.set_ylabel("CV2" if cv_scores.shape[1] >= 2 else "")
    ax.set_title("Canonical Variate Analysis")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_cva.png"), dpi=200)
    plt.close(fig)

    group_means = {}
    for g in unique:
        idx = [j for j, gg in enumerate(groups) if gg == g]
        group_means[g] = X[idx].mean(axis=0)

    n_g = len(unique)
    mahal = np.zeros((n_g, n_g))
    pooled_cov = np.cov(X.T)
    try:
        inv_cov = np.linalg.pinv(pooled_cov)
    except Exception:
        log.warning("Mahalanobis distance computation failed — singular covariance")
        return

    for i in range(n_g):
        for j in range(i + 1, n_g):
            diff = group_means[unique[i]] - group_means[unique[j]]
            d = np.sqrt(diff @ inv_cov @ diff)
            mahal[i, j] = d
            mahal[j, i] = d

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(mahal, xticklabels=unique, yticklabels=unique, annot=True, fmt=".2f",
                cmap="YlOrRd", ax=ax)
    ax.set_title("Pairwise Mahalanobis Distances")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_mahalanobis.png"), dpi=200)
    plt.close(fig)

    mahal_df = pd.DataFrame(mahal, index=unique, columns=unique)
    mahal_df.to_csv(os.path.join(output_dir, f"{prefix}_mahalanobis.csv"))
    log.info("CVA + Mahalanobis heatmap saved")


def run_centroid_size_analysis(sizes, groups, filenames, output_dir, prefix):
    cs_df = pd.DataFrame({"filename": filenames, "centroid_size": sizes})
    if groups:
        cs_df["group"] = groups
    cs_df.to_csv(os.path.join(output_dir, f"{prefix}_centroid_sizes.csv"), index=False)

    if groups:
        unique = sorted(set(groups))
        if len(unique) >= 2:
            from scipy.stats import f_oneway
            group_data = [sizes[[j for j, g in enumerate(groups) if g == ug]] for ug in unique]
            group_data = [gd for gd in group_data if len(gd) > 0]
            if len(group_data) >= 2 and all(len(gd) >= 1 for gd in group_data):
                try:
                    F, p = f_oneway(*group_data)
                    log.info(f"Centroid size ANOVA: F={F:.3f}, p={p:.4f}")
                    with open(os.path.join(output_dir, f"{prefix}_size_anova.txt"), "w") as f:
                        f.write(f"One-way ANOVA on centroid size\n")
                        f.write(f"Groups: {unique}\n")
                        f.write(f"Group sizes: {[len(gd) for gd in group_data]}\n")
                        f.write(f"Group means: {[gd.mean() for gd in group_data]}\n")
                        f.write(f"F-statistic: {F:.4f}\n")
                        f.write(f"p-value: {p:.6f}\n")
                except Exception as e:
                    log.warning(f"Centroid size ANOVA failed: {e}")

        fig, ax = plt.subplots(figsize=(8, 5))
        cs_df.boxplot(column="centroid_size", by="group", ax=ax)
        ax.set_title("Centroid Size by Group")
        ax.set_xlabel("Group")
        ax.set_ylabel("Centroid Size (pixels)")
        plt.suptitle("")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}_centroid_size_boxplot.png"), dpi=200)
        plt.close(fig)


def run_form_space(aligned, sizes, filenames, groups, output_dir, prefix):
    X = aligned.reshape(len(aligned), -1)
    log_cs = np.log(sizes).reshape(-1, 1)
    form = np.hstack([X, log_cs])

    n_components = min(form.shape[0] - 1, form.shape[1], 10)
    if n_components < 2:
        log.warning("Not enough specimens for form space PCA")
        return

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(form)

    fig, ax = plt.subplots(figsize=(10, 8))
    if groups:
        unique_groups = sorted(set(groups))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_groups), 1)))
        for i, g in enumerate(unique_groups):
            idx = [j for j, gg in enumerate(groups) if gg == g]
            ax.scatter(scores[idx, 0], scores[idx, 1], label=g, color=colors[i], s=60, alpha=0.8)
        ax.legend(title="Group")
    else:
        ax.scatter(scores[:, 0], scores[:, 1], s=60, alpha=0.8)

    var_exp = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"Form PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"Form PC2 ({var_exp[1]:.1f}%)")
    ax.set_title("Form Space PCA (Shape + Size)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_form_space.png"), dpi=200)
    plt.close(fig)
    log.info(f"Form space PCA: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%")


def run_allometry(aligned, sizes, groups, output_dir, prefix):
    X = aligned.reshape(len(aligned), -1)
    log_cs = np.log(sizes)

    if X.shape[1] >= X.shape[0]:
        pca_dim = max(2, X.shape[0] - 2)
        pca = PCA(n_components=pca_dim)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    from numpy.linalg import lstsq
    A = np.column_stack([log_cs, np.ones(len(log_cs))])
    coeffs, residuals, _, _ = lstsq(A, X_reduced, rcond=None)

    predicted = A @ coeffs
    ss_res = np.sum((X_reduced - predicted) ** 2)
    ss_tot = np.sum((X_reduced - X_reduced.mean(axis=0)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    pc1_pred = PCA(n_components=min(2, X_reduced.shape[1])).fit_transform(predicted)
    pc1_res = PCA(n_components=min(2, X_reduced.shape[1])).fit_transform(X_reduced)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    if groups:
        unique_groups = sorted(set(groups))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_groups), 1)))
        for i, g in enumerate(unique_groups):
            idx = [j for j, gg in enumerate(groups) if gg == g]
            ax.scatter(log_cs[idx], pc1_res[idx, 0], label=g, color=colors[i], s=60, alpha=0.8)
        ax.legend(title="Group", fontsize=7)
    else:
        ax.scatter(log_cs, pc1_res[:, 0], s=60, alpha=0.8)
    z = np.polyfit(log_cs, pc1_res[:, 0], 1)
    x_line = np.linspace(log_cs.min(), log_cs.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)
    ax.set_xlabel("log(Centroid Size)")
    ax.set_ylabel("Shape PC1")
    ax.set_title(f"Allometry (R²={r_squared:.3f})")

    ax = axes[1]
    residuals_shape = X_reduced - predicted
    if residuals_shape.shape[1] >= 2 and residuals_shape.shape[0] >= 3:
        pca_resid = PCA(n_components=2)
        resid_scores = pca_resid.fit_transform(residuals_shape)
        if groups:
            for i, g in enumerate(unique_groups):
                idx = [j for j, gg in enumerate(groups) if gg == g]
                ax.scatter(resid_scores[idx, 0], resid_scores[idx, 1], label=g, color=colors[i], s=60, alpha=0.8)
        else:
            ax.scatter(resid_scores[:, 0], resid_scores[:, 1], s=60, alpha=0.8)
        ax.set_xlabel("Residual PC1")
        ax.set_ylabel("Residual PC2")
        ax.set_title("Size-Corrected Shape (Residuals)")
    else:
        ax.text(0.5, 0.5, "Insufficient dimensions\nfor residual PCA", ha="center", va="center",
                transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_allometry.png"), dpi=200)
    plt.close(fig)

    with open(os.path.join(output_dir, f"{prefix}_allometry.txt"), "w") as f:
        f.write(f"Multivariate allometry: shape ~ log(centroid size)\n")
        f.write(f"R² = {r_squared:.4f}\n")
        f.write(f"n = {len(sizes)} specimens\n")
        f.write(f"Shape variables = {X_reduced.shape[1]}\n")
    log.info(f"Allometry: R²={r_squared:.3f}")


def run_dendrogram(aligned, filenames, groups, output_dir, prefix):
    X = aligned.reshape(len(aligned), -1)
    dists = pdist(X, metric="euclidean")
    Z = linkage(dists, method="ward")

    fig, ax = plt.subplots(figsize=(max(12, len(filenames) * 0.4), 6))

    leaf_labels = filenames
    if groups:
        leaf_labels = [f"{fn} [{g}]" for fn, g in zip(filenames, groups)]

    dendrogram(Z, labels=leaf_labels, ax=ax, leaf_rotation=90, leaf_font_size=7)
    ax.set_title("Hierarchical Clustering of Procrustes Shape Coordinates (Ward)")
    ax.set_ylabel("Procrustes Distance")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_dendrogram.png"), dpi=200)
    plt.close(fig)
    log.info("Dendrogram saved")


def plot_mean_shape(mean_shape, output_dir, prefix, cat_name):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(mean_shape[:, 0], mean_shape[:, 1], "ko", markersize=8)
    for i, (x, y) in enumerate(mean_shape):
        ax.annotate(str(i + 1), (x, y), fontsize=8, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points", color="blue")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"Mean Landmark Configuration — {cat_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_mean_shape.png"), dpi=200)
    plt.close(fig)


def plot_wireframe_deformations(aligned, mean_shape, pca_obj, output_dir, prefix):
    if pca_obj is None:
        return
    k = mean_shape.shape[0]
    mean_flat = mean_shape.reshape(1, -1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].plot(mean_shape[:, 0], mean_shape[:, 1], "ko-", markersize=6)
    for i, (x, y) in enumerate(mean_shape):
        axes[0].annotate(str(i + 1), (x, y), fontsize=7, color="blue")
    axes[0].set_title("Mean Shape")
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()

    for pc_idx, ax in zip([0, 1], axes[1:]):
        if pc_idx >= pca_obj.n_components_:
            break
        pc_vec = pca_obj.components_[pc_idx]
        sd = np.sqrt(pca_obj.explained_variance_[pc_idx])
        for sign, color, label in [(-2, "blue", "-2σ"), (2, "red", "+2σ")]:
            deformed = (mean_flat + sign * sd * pc_vec).reshape(k, 2)
            ax.plot(deformed[:, 0], deformed[:, 1], "o-", color=color, markersize=4,
                    alpha=0.6, label=label)
        ax.plot(mean_shape[:, 0], mean_shape[:, 1], "ko-", markersize=3, alpha=0.3, label="mean")
        ax.set_title(f"PC{pc_idx + 1} Deformation ({pca_obj.explained_variance_ratio_[pc_idx]*100:.1f}%)")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_pc_deformations.png"), dpi=200)
    plt.close(fig)
    log.info("PC deformation wireframes saved")


def save_tps(aligned, sizes, filenames, output_dir, prefix):
    path = os.path.join(output_dir, f"{prefix}_aligned.tps")
    with open(path, "w") as f:
        for i, (cfg, cs, fn) in enumerate(zip(aligned, sizes, filenames)):
            f.write(f"LM={cfg.shape[0]}\n")
            for x, y in cfg:
                f.write(f"{x:.6f} {y:.6f}\n")
            f.write(f"IMAGE={fn}\n")
            f.write(f"SCALE={cs:.6f}\n")
            f.write(f"ID={i}\n\n")
    log.info(f"TPS file saved: {path}")


def save_procrustes_coords(aligned, filenames, groups, output_dir, prefix):
    k = aligned.shape[1]
    cols = []
    for i in range(k):
        cols.extend([f"lm{i+1}_x", f"lm{i+1}_y"])
    X = aligned.reshape(len(aligned), -1)
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "filename", filenames)
    if groups:
        df.insert(1, "group", groups)
    path = os.path.join(output_dir, f"{prefix}_procrustes_coords.csv")
    df.to_csv(path, index=False)
    log.info(f"Procrustes coordinates saved: {path}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Landmark GPA with full morphometric suite")
    parser.add_argument("--json", required=True, help="COCO keypoints JSON")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_labels", default=None, help="CSV with filename,group columns")
    parser.add_argument("--category", default=None, help="Process only this category name (default: all)")
    parser.add_argument("--perform_manova", action="store_true", default=True)
    parser.add_argument("--skip_umap", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = load_keypoints(args.json, args.category)

    for cat_name, (configs, filenames) in all_results.items():
        if len(configs) < 3:
            log.warning(f"Category '{cat_name}': only {len(configs)} specimens — need ≥3 for meaningful GPA")
            if len(configs) < 2:
                continue

        log.info(f"\n{'='*60}")
        log.info(f"Processing: {cat_name} ({len(configs)} specimens, {configs[0].shape[0]} landmarks)")
        log.info(f"{'='*60}")

        cat_dir = os.path.join(args.output_dir, cat_name.replace(" ", "_"))
        os.makedirs(cat_dir, exist_ok=True)
        prefix = cat_name.replace(" ", "_")

        groups = load_groups(args.group_labels, filenames)

        # GPA
        aligned, mean_shape, sizes = gpa(configs)

        # Save outputs
        save_tps(aligned, sizes, filenames, cat_dir, prefix)
        save_procrustes_coords(aligned, filenames, groups, cat_dir, prefix)
        plot_mean_shape(mean_shape, cat_dir, prefix, cat_name)

        # PCA
        pca_result = run_pca(aligned, filenames, groups, cat_dir, prefix)
        pca_obj = pca_result[0] if pca_result else None
        plot_wireframe_deformations(aligned, mean_shape, pca_obj, cat_dir, prefix)

        # UMAP
        if not args.skip_umap:
            run_umap(aligned, filenames, groups, cat_dir, prefix)

        # Dendrogram
        run_dendrogram(aligned, filenames, groups, cat_dir, prefix)

        # Centroid size
        run_centroid_size_analysis(sizes, groups, filenames, cat_dir, prefix)

        # Form space
        run_form_space(aligned, sizes, filenames, groups, cat_dir, prefix)

        # Allometry
        run_allometry(aligned, sizes, groups, cat_dir, prefix)

        # Group-dependent analyses
        if groups and len(set(groups)) >= 2 and args.perform_manova:
            run_manova(aligned, groups, cat_dir, prefix)
            run_cva(aligned, groups, filenames, cat_dir, prefix)

        log.info(f"Done: {cat_name}")

    log.info("\nAll categories processed.")


if __name__ == "__main__":
    main()
