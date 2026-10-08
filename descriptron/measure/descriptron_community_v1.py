#!/usr/bin/env python3
"""
descriptron_community_v1.py - is the morphology of a community explained by phylogeny, by habitat, or by both?
=============================================================================================================

For specimens collected at many sites that differ in a treatment (for example primary vs secondary forest), with
colour, colour-pattern, texture or any other Descriptron traits, a phylogeny of the species and a specimen metadata
table (Darwin Core), this program asks:

  varpart     how much of the trait variation is explained by the treatment alone, by phylogeny alone, by the two
              together (they cannot be told apart), and how much by neither. Units are species x treatment-level
              means; phylogeny enters as phylogenetic eigenvectors (principal coordinates of the patristic distances,
              Diniz-Filho et al. 1998), and the fractions are adjusted R2 of redundancy analysis (Peres-Neto et al.
              2006), as in Desdevises et al. (2003). Fractions [a] (treatment | phylogeny) and [c] (phylogeny |
              treatment) are tested by permutation of residuals (Freedman & Lane). Labels follow Desdevises et al.:
              [b] is the shared fraction (vegan::varpart labels the same numbers [a] X1|X2, [b] X2|X1, [c] shared).
  paired      species found under both levels of the treatment, compared with themselves: phylogeny is held
              constant, so a consistent shift is the treatment's effect. Multivariate (sign-flip permutation of the
              species' difference vectors) and per feature (Benjamini-Hochberg).
  signal      multivariate phylogenetic signal Kmult (Adams 2014) of the species means, overall and within each
              treatment level.
  community   per site: species richness, Faith's PD, MPD and MNTD with standardised effect sizes against random
              draws from the species pool (Webb et al. 2002; picante's ses.mpd / ses.mntd with the taxa.labels-like
              null), and trait dispersion (FDis, Laliberte & Legendre 2010) and trait MPD of the site's species;
              then the treatment levels are compared with SITES as the replicates (permuting whole sites).
  morphospace PCA of the species x level means, the species tree drawn through the species means, and an arrow
              from each species' mean under one level to its mean under the other.

Inputs
  --metadata   one row per specimen: CSV, TSV, Excel, or Darwin Core occurrence.txt. Default columns are Darwin
               Core: occurrenceID (specimen), scientificName (species), locationID (site), habitat (treatment),
               associatedMedia (image file names, '|' separated), associatedSequences. Any column can be named with
               --specimen_col/--species_col/--site_col/--media_col. Further categorical variables: --factor (the
               first is the treatment); site-level numbers (elevation, canopy cover...): --covariate.
  traits       either Descriptron trait tables, --traits NAME=table.csv (one row per image or annotation, e.g. the
               colour/texture homology features), or --coco annotations.json --image_dir images/ to compute basic
               colour (CIE L*a*b*) and texture (grey-level co-occurrence) features from the masks here
               (--structure picks one category). Rows are linked to specimens through associatedMedia, or by the
               specimen id appearing in the image file name.
  phylogeny    --tree species.nwk (tips = species; each site's community is the tree pruned to the species recorded
               there), or --tree tips.nwk --tip_map tip_map.csv (columns tip, species, site: one sequence per species
               per site; a site's community is its own tips), and/or --site_tree SITE=tree.nwk for sites with their
               own tree. Tip names match species with spaces or underscores.

  python descriptron_community_v1.py --metadata occurrences.csv --tree ants.nwk \\
      --traits colour=color_homology_features.csv --traits texture=texture_homology_features.csv \\
      --factor habitat --covariate minimumElevationInMeters --out_dir community/

A Descriptron colour-homology table is split automatically into two trait sets, <name>_absolute (each cell's
colour, which a camera or light change shifts) and <name>_pattern (each cell relative to the specimen's own mean,
plus within-cell variation), so colour, colour pattern and texture are compared separately (--no_split_colour
keeps the table whole). With --coco the computed sets are colour_absolute, colour_pattern and texture.

Writes, in --out_dir: units.tsv, varpart.tsv + varpart.png/.pdf, paired.tsv, paired_features_<set>.tsv,
signal.tsv, community_sites.tsv, community_comparison.tsv + community_by_<treatment>.png/.pdf,
morphospace_<set>.png/.pdf (dots) and, whenever --image_dir is given, morphospace_<set>_thumbnails.png/.pdf
(the specimen nearest each species x level mean, cut out with COCO masks if available), report.md and summary.json.
If the site lists do not look like community samples (median < 3 specimens per site, or > 40% single-species sites),
the run says so: community metrics are then about collecting, not co-occurrence.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import descriptron_phylo as ph                 # noqa: E402  (tree, VCV, Kmult and ancestral states, checked vs geomorph)

VERSION = "1.0"
DWC = {"specimen": "occurrenceID", "species": "scientificName", "site": "locationID", "media": "associatedMedia",
       "sequence": "associatedSequences", "factor": "habitat"}


# ============================================================================ names, metadata, linking
def norm(s) -> str:
    """species / tip names compared with spaces, underscores and case ignored"""
    return re.sub(r"[\s_]+", "_", str(s).strip()).strip("_").lower()


def read_table(path: str) -> pd.DataFrame:
    p = str(path)
    if p.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(p, dtype=str)
    sep = "\t" if p.lower().endswith((".tsv", ".txt")) else ","
    return pd.read_csv(p, sep=sep, dtype=str, keep_default_na=False)


def image_keys(name) -> list:
    return ph._key_forms(os.path.basename(str(name)))


def link_rows(ids, meta, a) -> list:
    """trait-row id (image file / annotation) -> metadata row index, via associatedMedia or the specimen id
    appearing in the file name"""
    by_media = {}
    if a.media_col in meta.columns:
        for i, v in meta[a.media_col].items():
            for f in re.split(r"[|;,]", str(v)):
                if f.strip():
                    for k in image_keys(f.strip()):
                        by_media.setdefault(k, i)
    specs = sorted(((str(s), i) for i, s in meta[a.specimen_col].items() if str(s)), key=lambda x: -len(x[0]))
    out = []
    for x in ids:
        hit = next((by_media[k] for k in image_keys(x) if k in by_media), None)
        if hit is None:
            base = os.path.basename(str(x))
            hit = next((i for s, i in specs if re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", base)), None)
        out.append(hit)
    return out


# ============================================================================ traits
def traits_from_table(path, meta, a):
    d = pd.read_csv(path)
    idc = next((c for c in ("filename", "file", "image", "image_filename", "specimen", "id") if c in d.columns), d.columns[0])
    num = [c for c in d.columns if c != idc and pd.api.types.is_numeric_dtype(d[c]) and d[c].notna().any()]
    d = d[[idc] + num]
    d["_meta"] = link_rows(d[idc].tolist(), meta, a)
    d["_image"] = d[idc].astype(str).map(lambda x: re.sub(r"\.(png|jpe?g|tiff?|bmp)_\d+$", r".\1", x, flags=re.I))
    return d.dropna(subset=["_meta"]), num


def split_colour(name, d, cols):
    """A Descriptron colour-homology table holds two different things: absolute colour of each cell (*_mean_abs and
    the global chroma/hue/means), which a brighter or warmer camera shifts, and colour PATTERN - each cell relative
    to the specimen's own mean (*_dL_rel, *_da_rel, *_db_rel) plus within-cell variation (std, entropy, dominant
    colour proportion) - which a uniform change of light cancels out of. They are analysed as separate trait sets.
    Returns {name: (d, cols)} with one or two entries."""
    rel = [c for c in cols if re.search(r"_d[Lab]_rel$", c)]
    ab = [c for c in cols if re.search(r"_[Lab]_mean_abs$", c)] + \
        [c for c in cols if c in ("chroma_ab", "hue_ab_cos", "hue_ab_sin", "L_mean", "a_mean", "b_mean")]
    if not rel or not ab:
        return {name: (d, cols)}
    within = [c for c in cols if re.search(r"_([Lab]_std|color_entropy|dominant_proportion)$", c)]
    return {f"{name}_absolute": (d, ab), f"{name}_pattern": (d, rel + within)}


def traits_from_coco(coco_paths, image_dir, structure, meta, a):
    """basic colour and texture per annotation mask (used when no Descriptron trait tables are given)"""
    import cv2
    from PIL import Image
    from skimage.color import rgb2lab
    from skimage.feature import graycomatrix, graycoprops
    rows = []
    for cp in coco_paths:
        d = json.load(open(cp))
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        imgs = {im["id"]: im for im in d.get("images", [])}
        for ann in d.get("annotations", []):
            seg = ann.get("segmentation")
            if not seg or (structure and cats.get(ann.get("category_id")) != structure):
                continue
            im = imgs.get(ann.get("image_id"), {})
            path = Path(image_dir) / os.path.basename(str(im.get("file_name", "")))
            if not path.exists():
                continue
            rgb = np.asarray(Image.open(path).convert("RGB"))
            m = np.zeros(rgb.shape[:2], np.uint8)
            if isinstance(seg, list):
                for poly in seg:
                    cv2.fillPoly(m, [np.asarray(poly, float).reshape(-1, 2).round().astype(np.int32)], 1)
            else:
                from pycocotools import mask as mu
                m = mu.decode(seg if isinstance(seg.get("counts"), str) else mu.frPyObjects(seg, *seg["size"])).astype(np.uint8)
            if m.sum() < 50:
                continue
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop, mk = rgb[y0:y1, x0:x1], m[y0:y1, x0:x1].astype(bool)
            lab = rgb2lab(crop)[mk]
            L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]
            hue = np.arctan2(B, A)
            g = (crop.mean(2) * mk).astype(np.uint8)
            P = graycomatrix(g, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True)
            P[0, :, :, :] = 0; P[:, 0, :, :] = 0                       # ignore the blanked background
            P = P / np.maximum(P.sum(axis=(0, 1), keepdims=True), 1)
            f = {"L_mean": L.mean(), "L_sd": L.std(), "a_mean": A.mean(), "a_sd": A.std(), "b_mean": B.mean(),
                 "b_sd": B.std(), "chroma_mean": np.hypot(A, B).mean(), "hue_sin": np.sin(hue).mean(),
                 "hue_cos": np.cos(hue).mean(), "L_q10_rel": np.quantile(L, .1) - np.median(L),
                 "L_q90_rel": np.quantile(L, .9) - np.median(L)}
            for prop in ("contrast", "homogeneity", "energy", "correlation"):
                f[f"glcm_{prop}"] = float(np.nan_to_num(graycoprops(P, prop)).mean())
            rows.append({"file": f"{im.get('file_name')}_{ann.get('id')}", **f})
    if not rows:
        sys.exit("no annotation masks could be read from --coco / --image_dir")
    d = pd.DataFrame(rows)
    d["_meta"] = link_rows([r.rsplit("_", 1)[0] for r in d["file"]], meta, a)
    d["_image"] = [r.rsplit("_", 1)[0] for r in d["file"]]
    absolute = ["L_mean", "a_mean", "b_mean", "chroma_mean", "hue_sin", "hue_cos"]
    pattern = ["L_sd", "a_sd", "b_sd", "L_q10_rel", "L_q90_rel"]       # spread within the structure, light-independent
    texture = [c for c in d.columns if c.startswith("glcm_")]
    d = d.dropna(subset=["_meta"])
    return {"colour_absolute": (d, absolute), "colour_pattern": (d, pattern), "texture": (d, texture)}


# ============================================================================ phylogeny
class Phylogeny:
    """one tree (species tips, or per-site tips with a tip map) and optional per-site trees"""

    def __init__(self, a, species_all):
        self.site_trees = {}
        for st in a.site_tree or []:
            site, _, path = st.partition("=")
            self.site_trees[site] = ph.parse_newick(open(path).read())
        self.tipmap = None
        self.root = ph.parse_newick(open(a.tree).read()) if a.tree else None
        sp_norm = {norm(s): s for s in species_all}
        if self.root is not None and a.tip_map:
            tm = read_table(a.tip_map)
            self.tipmap = {r["tip"]: (sp_norm.get(norm(r["species"]), r["species"]), str(r["site"])) for _, r in tm.iterrows()}
        # species-level tree: tips renamed to species; with a tip map, one tip (the first) per species
        self.sp_root = None
        if self.root is not None:
            import copy
            r = copy.deepcopy(self.root)
            keep, seen = set(), set()
            for t in ph.tips_of(r):
                sp = (self.tipmap.get(t.name, (None, None))[0] if self.tipmap else sp_norm.get(norm(t.name)))
                if sp and sp not in seen:
                    seen.add(sp); t.name = sp; keep.add(sp)
                else:
                    t.name = f"__drop__{id(t)}"
            self.sp_root = ph.prune(r, keep) if keep else None
        self.species = sorted(t.name for t in ph.tips_of(self.sp_root)) if self.sp_root else []

    def patristic(self, root, names):
        tips = {t.name: t for t in ph.tips_of(root)}
        T = [tips[n] for n in names]
        return np.array([[0.0 if a is b else a.depth + b.depth - 2 * ph.mrca_depth(a, b) for b in T] for a in T])

    def species_distance(self, names):
        return self.patristic(self.sp_root, names)

    def species_vcv(self, names):
        import copy
        r = ph.prune(copy.deepcopy(self.sp_root), set(names))
        tips = {t.name: t for t in ph.tips_of(r)}
        return ph.vcv([tips[n] for n in names], [tips[n] for n in names]), r

    def site_community(self, site, species_here):
        """(tip names, distance matrix, PD) of one site's community, or None"""
        import copy
        if site in self.site_trees:
            r = self.site_trees[site]
            names = [t.name for t in ph.tips_of(r)]
            return names, self.patristic(r, names), faith_pd(r, set(names))
        if self.root is None:
            return None
        if self.tipmap:
            names = [t.name for t in ph.tips_of(self.root) if self.tipmap.get(t.name, (None, None))[1] == str(site)]
            if not names:
                return None
            return names, self.patristic(self.root, names), faith_pd(self.root, set(names))
        names = [s for s in species_here if s in self.species]
        if not names:
            return None
        return names, self.species_distance(names), faith_pd(self.sp_root, set(names))


def _u(name) -> str:
    """names as written to the validation files: spaces -> underscores (Newick-safe)"""
    return str(name).replace(" ", "_")


def to_newick(n) -> str:
    """Newick of a (pruned) tree, branch lengths kept"""
    def rec(x):
        lab = _u(x.name) if not x.kids else "(" + ",".join(f"{rec(k)}:{k.length!r}" for k in x.kids) + ")"
        return lab
    return rec(n) + ";"


def faith_pd(root, keep) -> float:
    """Faith's PD including the root: total length of the branches joining the kept tips to the root"""
    total = 0.0

    def rec(n):
        nonlocal total
        hit = (n.name in keep) if not n.kids else any([rec(k) for k in n.kids])
        if hit and n.parent is not None:
            total += n.length
        return hit
    rec(root)
    return total


def phylo_eigenvectors(D, max_k):
    """principal coordinates of a distance matrix; eigenvectors above the broken-stick expectation (at most max_k)"""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    pos = w > 1e-10 * max(w.max(), 1e-12)
    w, V = w[pos], V[:, pos]
    m = len(w)
    stick = np.array([sum(1.0 / j for j in range(i, m + 1)) / m for i in range(1, m + 1)])
    k = int(np.sum(np.cumprod(w / w.sum() > stick)))
    k = max(1, min(k, max_k, m))
    return V[:, :k] * np.sqrt(w[:k]), w / w.sum()


# ============================================================================ statistics
def _rss(X, Y):
    X1 = np.column_stack([np.ones(len(Y)), X]) if X is not None and X.size else np.ones((len(Y), 1))
    B, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    R = Y - X1 @ B
    return float((R ** 2).sum()), X1.shape[1] - 1, X1 @ B


def r2adj(X, Y):
    n = len(Y)
    tss = float(((Y - Y.mean(0)) ** 2).sum())
    rss, p, _ = _rss(X, Y)
    r2 = 1 - rss / tss
    return r2, 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else float("nan"), p


def partial_F_test(Y, X, Z, iters, rng):
    """F of X given Z (Z may be None), permuting residuals of the reduced model (Freedman & Lane)"""
    XZ = X if Z is None else np.column_stack([X, Z])
    rss_red, p_red, fit_red = _rss(Z, Y)
    rss_full, p_full, _ = _rss(XZ, Y)
    n = len(Y)
    df1, df2 = p_full - p_red, n - p_full - 1
    if df1 <= 0 or df2 <= 0:
        return float("nan"), float("nan")
    F = ((rss_red - rss_full) / df1) / (rss_full / df2)
    res = Y - fit_red
    ge = 0
    for _ in range(iters):
        Yp = fit_red + res[rng.permutation(n)]
        r_red, _, _ = _rss(Z, Yp)
        r_full, _, _ = _rss(XZ, Yp)
        ge += ((r_red - r_full) / df1) / (r_full / df2) >= F - 1e-12
    return float(F), float((ge + 1) / (iters + 1))


def varpart(Y, E, P, iters, rng):
    """Desdevises et al. 2003 / vegan::varpart with two explanatory tables: adjusted-R2 fractions"""
    _, ab, _ = r2adj(E, Y)
    _, bc, _ = r2adj(P, Y)
    r2_all, abc, _ = r2adj(np.column_stack([E, P]), Y)
    Fa, pa = partial_F_test(Y, E, P, iters, rng)
    Fc, pc = partial_F_test(Y, P, E, iters, rng)
    FE, pE = partial_F_test(Y, E, None, iters, rng)
    FP, pP = partial_F_test(Y, P, None, iters, rng)
    return {"treatment_total_[a+b]": ab, "phylogeny_total_[b+c]": bc, "both_[a+b+c]": abc,
            "treatment_only_[a]": abc - bc, "shared_[b]": ab + bc - abc, "phylogeny_only_[c]": abc - ab,
            "unexplained_[d]": 1 - abc, "F_treatment_given_phylogeny": Fa, "P_treatment_given_phylogeny": pa,
            "F_phylogeny_given_treatment": Fc, "P_phylogeny_given_treatment": pc,
            "F_treatment": FE, "P_treatment": pE, "F_phylogeny": FP, "P_phylogeny": pP}


def bh(p):
    p = np.asarray(p, float); n = len(p)
    o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def sign_flip(Dm, iters, rng):
    """Dm: species x features differences; returns (statistic, P) for the mean difference vector"""
    obs = float((Dm.mean(0) ** 2).sum())
    ge = 0
    for _ in range(iters):
        s = rng.choice([-1.0, 1.0], size=(len(Dm), 1))
        ge += float(((Dm * s).mean(0) ** 2).sum()) >= obs - 1e-12
    return obs, (ge + 1) / (iters + 1)


def site_label_test(values, labels, A, B, iters, rng):
    v = np.asarray(values, float); lab = np.asarray(labels)
    ok = ~np.isnan(v) & np.isin(lab, [A, B])
    v, lab = v[ok], lab[ok]
    if (lab == A).sum() < 2 or (lab == B).sum() < 2:
        return dict(n_A=int((lab == A).sum()), n_B=int((lab == B).sum()), mean_A=np.nan, mean_B=np.nan, diff=np.nan, P=np.nan)
    obs = v[lab == B].mean() - v[lab == A].mean()
    ge = 0
    for _ in range(iters):
        p = rng.permutation(lab)
        ge += abs(v[p == B].mean() - v[p == A].mean()) >= abs(obs) - 1e-12
    return dict(n_A=int((lab == A).sum()), n_B=int((lab == B).sum()), mean_A=float(v[lab == A].mean()),
                mean_B=float(v[lab == B].mean()), diff=float(obs), P=float((ge + 1) / (iters + 1)))


def pca(M, max_k=10, var=0.95):
    Z = (M - M.mean(0)) / np.where(M.std(0, ddof=1) > 0, M.std(0, ddof=1), 1)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    k = int(min(max_k, max(2, np.searchsorted(np.cumsum(ev), var) + 1), len(S)))
    return Z @ Vt[:k].T, ev[:k], Vt[:k], Z.mean(0), M.mean(0), np.where(M.std(0, ddof=1) > 0, M.std(0, ddof=1), 1)


def mpd_mntd(D):
    n = len(D)
    if n < 2:
        return np.nan, np.nan
    iu = np.triu_indices(n, 1)
    Dm = D + np.diag(np.full(n, np.inf))
    return float(D[iu].mean()), float(Dm.min(1).mean())


def ses(obs, null):
    null = np.asarray([x for x in null if x == x])
    if not len(null) or null.std(ddof=1) == 0 or obs != obs:
        return np.nan, np.nan
    return float((obs - null.mean()) / null.std(ddof=1)), float((np.sum(null <= obs) + 1) / (len(null) + 1))


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--specimen_col", default=DWC["specimen"])
    ap.add_argument("--species_col", default=DWC["species"])
    ap.add_argument("--site_col", default=DWC["site"])
    ap.add_argument("--media_col", default=DWC["media"])
    ap.add_argument("--factor", action="append", default=None,
                    help="categorical variable(s); the first is the treatment (default: habitat)")
    ap.add_argument("--contrast", nargs=2, default=None, metavar=("A", "B"),
                    help="the two levels of the treatment to compare (default: the two most common)")
    ap.add_argument("--covariate", action="append", default=[], help="site-level numeric column(s)")
    ap.add_argument("--traits", action="append", default=[], metavar="NAME=table.csv")
    ap.add_argument("--no_split_colour", action="store_true",
                    help="do not split a colour-homology table into colour (absolute) and colour pattern (relative)")
    ap.add_argument("--coco", nargs="*", default=None)
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--structure", default=None, help="with --coco: use only this category")
    ap.add_argument("--thumbnails", action="store_true",
                    help="draw each morphospace also with a specimen thumbnail at every species x level point (the "
                         "specimen nearest that mean); on by default whenever --image_dir is given; cut out with "
                         "--thumb_coco (or --coco) masks if given")
    ap.add_argument("--no_thumbnails", action="store_true", help="do not draw the thumbnail morphospaces")
    ap.add_argument("--thumb_coco", nargs="*", default=None,
                    help="COCO file(s) whose masks cut the specimen out of its image (default: --coco)")
    ap.add_argument("--thumb_size", type=float, default=0.55, help="thumbnail size in inches")
    ap.add_argument("--tree", default=None)
    ap.add_argument("--tip_map", default=None, help="CSV: tip, species, site (one sequence per species per site)")
    ap.add_argument("--site_tree", action="append", default=[], metavar="SITE=tree.nwk")
    ap.add_argument("--analyses", nargs="+", default=["varpart", "paired", "signal", "community", "morphospace"])
    ap.add_argument("--save_matrices", action="store_true",
                    help="also write the exact inputs behind the results (varpart matrices, site species lists, trees, "
                         "phylomorphospace tips and ancestral states) to <out_dir>/validation_inputs/, for checking in R "
                         "(validation/validate_community_run_vs_r.py)")
    ap.add_argument("--iters", type=int, default=999)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    a.factor = a.factor or [DWC["factor"]]
    a.thumbnails = (a.thumbnails or bool(a.image_dir)) and not a.no_thumbnails
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    vdir = out / "validation_inputs"
    if a.save_matrices:
        vdir.mkdir(exist_ok=True)
    log = []

    def say(s):
        print(s, flush=True); log.append(s)

    meta = read_table(a.metadata)
    for c in [a.specimen_col, a.species_col, a.site_col, *a.factor, *a.covariate]:
        if c not in meta.columns:
            sys.exit(f"--metadata has no column '{c}' (columns: {', '.join(meta.columns)})")
    meta = meta[meta[a.species_col].astype(str).str.strip() != ""].reset_index(drop=True)
    focal = a.factor[0]
    levels = meta[focal].value_counts()
    A, B = a.contrast or (sorted(levels.index[:2]) if len(levels) >= 2 else (None, None))
    say(f"metadata: {len(meta)} specimens, {meta[a.species_col].nunique()} species, {meta[a.site_col].nunique()} sites; "
        f"treatment '{focal}': {dict(levels)}; contrast {A} vs {B}")
    site_level = meta.groupby(a.site_col)[focal].agg(lambda s: s.mode().iloc[0])
    mixed = meta.groupby(a.site_col)[focal].nunique()
    if (mixed > 1).any():
        say(f"  note: {int((mixed > 1).sum())} site(s) hold more than one '{focal}' level; each site takes its commonest")

    # ---- traits
    sets = {}
    for t in a.traits:
        name, _, path = t.partition("=")
        d, cols = traits_from_table(path, meta, a)
        sets.update(split_colour(name, d, cols) if not a.no_split_colour else {name: (d, cols)})
    if a.coco:
        if not a.image_dir:
            sys.exit("--coco needs --image_dir")
        sets.update(traits_from_coco(a.coco, a.image_dir, a.structure, meta, a))
    for name, (d, cols) in sets.items():
        say(f"traits '{name}': {len(cols)} features, {len(d)} rows linked to {d['_meta'].nunique()} specimens")

    phy = Phylogeny(a, sorted(meta[a.species_col].unique())) if (a.tree or a.site_tree) else None
    if phy is not None:
        phy._vdir = vdir if a.save_matrices else None
    if phy and phy.species:
        missing = sorted(set(meta[a.species_col]) - set(phy.species))
        say(f"tree: {len(phy.species)} species{' (tip map: one tip per species per site)' if phy.tipmap else ''}"
            + (f"; {len(missing)} species in the metadata are not in the tree: {missing[:5]}" if missing else ""))
    summary = {"version": VERSION, "treatment": focal, "contrast": [A, B], "sets": {}}
    rows_var, rows_pair, rows_sig = [], [], []

    # ---- per trait set
    unit_tables = []
    for name, (d, cols) in sets.items():
        m = meta.loc[d["_meta"].astype(int)].reset_index(drop=True)
        X = d[cols].to_numpy(float)
        keep = ~np.isnan(X).any(1)
        X, m = X[keep], m[keep.tolist()].reset_index(drop=True)
        # specimen means first (several images per specimen), then species x level units
        spec = pd.DataFrame(X, columns=cols)
        spec["_specimen"] = m[a.specimen_col].values; spec["_species"] = m[a.species_col].values
        spec["_site"] = m[a.site_col].values
        spec["_image"] = d["_image"].to_numpy()[keep] if "_image" in d else ""
        for f in a.factor:
            spec[f] = m[f].values
        for c in a.covariate:
            spec[c] = pd.to_numeric(m[c], errors="coerce").values
        sp_mean = spec.groupby(["_specimen", "_species", "_site", *a.factor], dropna=False).agg(
            {**{c: "mean" for c in cols}, **{c: "mean" for c in a.covariate}, "_image": "first"}).reset_index()
        unit_keys = ["_species", *a.factor]
        units = sp_mean.groupby(unit_keys).agg({**{c: "mean" for c in cols}, **{c: "mean" for c in a.covariate},
                                                "_specimen": "count", "_site": "nunique"}).reset_index()
        units = units.rename(columns={"_specimen": "n_specimens", "_site": "n_sites"})
        if phy and phy.species:
            units = units[units["_species"].isin(phy.species)].reset_index(drop=True)
        if len(units) < 4:
            say(f"  '{name}': only {len(units)} species x level units - skipped"); continue
        S, ev, comps, zmean, mmean, msd = pca(units[cols].to_numpy(float), max_k=min(10, len(units) - 2))
        Y = S - S.mean(0)
        ut = units[unit_keys + ["n_specimens", "n_sites"]].copy(); ut.insert(0, "set", name)
        for i in range(S.shape[1]):
            ut[f"PC{i+1}"] = S[:, i]
        unit_tables.append(ut)
        say(f"  '{name}': {len(units)} species x level units, {S.shape[1]} PCs ({100*ev.sum():.0f}% of variance)")
        summary["sets"][name] = {"units": int(len(units)), "pcs": int(S.shape[1]), "pc_variance": float(ev.sum())}

        # varpart
        if "varpart" in a.analyses and phy and phy.species:
            E = pd.get_dummies(units[a.factor].astype(str), drop_first=True).to_numpy(float)
            if a.covariate:
                Cv = units[a.covariate].to_numpy(float)
                Cv = np.where(np.isnan(Cv), np.nanmean(Cv, 0), Cv)
                E = np.column_stack([E, (Cv - Cv.mean(0)) / np.where(Cv.std(0) > 0, Cv.std(0), 1)])
            spp = sorted(units["_species"].unique())
            D = phy.species_distance(spp)
            max_k = max(1, len(units) - E.shape[1] - 3)
            Vp, _ = phylo_eigenvectors(D, max_k)
            idx = {s: i for i, s in enumerate(spp)}
            P = Vp[[idx[s] for s in units["_species"]]]
            r = varpart(Y, E, P, a.iters, rng)
            if a.save_matrices:
                for nm_, M_ in (("Y", Y), ("E", E), ("P", P)):
                    pd.DataFrame(M_).to_csv(vdir / f"varpart_{name}_{nm_}.csv", index=False)
            r.update(set=name, n_units=len(units), n_species=len(spp), n_treatment_terms=E.shape[1],
                     n_phylo_eigenvectors=P.shape[1])
            rows_var.append(r)
            say(f"    varpart: treatment only {r['treatment_only_[a]']:.3f} (P={r['P_treatment_given_phylogeny']:.3g}), "
                f"shared {r['shared_[b]']:.3f}, phylogeny only {r['phylogeny_only_[c]']:.3f} "
                f"(P={r['P_phylogeny_given_treatment']:.3g}), unexplained {r['unexplained_[d]']:.3f}")

        # paired within-species contrast
        if "paired" in a.analyses and A is not None:
            u = units.copy(); u[[f"PC{i+1}" for i in range(S.shape[1])]] = S
            pa = u[u[focal] == A].groupby("_species").mean(numeric_only=True)
            pb = u[u[focal] == B].groupby("_species").mean(numeric_only=True)
            both = sorted(set(pa.index) & set(pb.index))
            if len(both) >= 3:
                pcs = [f"PC{i+1}" for i in range(S.shape[1])]
                Dm = pb.loc[both, pcs].to_numpy() - pa.loc[both, pcs].to_numpy()
                stat, p = sign_flip(Dm, a.iters, rng)
                Df = pb.loc[both, cols].to_numpy() - pa.loc[both, cols].to_numpy()
                feat = []
                for j, c in enumerate(cols):
                    sd = np.std(Df[:, j], ddof=1)
                    s_, p_ = sign_flip(Df[:, [j]] / (sd if sd > 0 else 1), a.iters, rng)
                    feat.append({"feature": c, "mean_difference": float(Df[:, j].mean()),
                                 "n_species_up": int((Df[:, j] > 0).sum()), "n_species": len(both), "P": p_})
                fq = pd.DataFrame(feat); fq["q_BH"] = bh(fq["P"]); fq.sort_values("P").to_csv(
                    out / f"paired_features_{name}.tsv", sep="\t", index=False)
                rows_pair.append({"set": name, "level_A": A, "level_B": B, "n_species_in_both": len(both),
                                  "mean_shift_squared_length": stat, "P": p,
                                  "n_features_q<0.05": int((fq["q_BH"] < 0.05).sum())})
                say(f"    paired: {len(both)} species under both {A} and {B}; mean shift P={p:.3g}; "
                    f"{int((fq['q_BH'] < 0.05).sum())} feature(s) with q<0.05")
            else:
                say(f"    paired: only {len(both)} species under both levels - skipped (needs 3)")

        # phylogenetic signal
        if "signal" in a.analyses and phy and phy.species:
            u = units.copy(); u[[f"PC{i+1}" for i in range(S.shape[1])]] = S
            pcs = [f"PC{i+1}" for i in range(S.shape[1])]
            for lev, g in [("all", u)] + [(L, u[u[focal] == L]) for L in (A, B) if L is not None]:
                mm = g.groupby("_species")[pcs].mean()
                if len(mm) < 4:
                    continue
                C, _ = phy.species_vcv(list(mm.index))
                res = ph.phylo_signal(mm.to_numpy(), C, a.iters, rng)
                rows_sig.append({"set": name, "level": lev, "n_species": len(mm), **res})
                say(f"    Kmult ({lev}): {res['Kmult']:.3f} (P={res['P']:.3g}, n={len(mm)})")

        # morphospace with arrows
        if "morphospace" in a.analyses and phy and phy.species:
            try:
                plot_morphospace(units, S, ev, phy, focal, A, B, name, out)
                if a.thumbnails:
                    # the specimen closest to each species x level mean, in the same PC space
                    Zs = (((sp_mean[cols].to_numpy(float) - mmean) / msd) - zmean) @ comps.T
                    reps = []
                    for i, ur in units.iterrows():
                        sel = (sp_mean["_species"] == ur["_species"]).to_numpy()
                        for f in a.factor:
                            sel &= (sp_mean[f].astype(str) == str(ur[f])).to_numpy()
                        idx = np.nonzero(sel)[0]
                        if len(idx):
                            j = idx[np.argmin(((Zs[idx] - S[i]) ** 2).sum(1))]
                            reps.append(sp_mean.iloc[j]["_image"])
                        else:
                            reps.append(None)
                    n_ok = plot_morphospace(units, S, ev, phy, focal, A, B, name, out, thumbs=(reps, a))
                    say(f"    thumbnails: {n_ok} of {len(units)} species x level points drawn as specimens")
            except Exception as e:  # noqa: BLE001
                say(f"    morphospace: not drawn ({e})")

    if unit_tables:
        pd.concat(unit_tables).to_csv(out / "units.tsv", sep="\t", index=False)
    if rows_var:
        pd.DataFrame(rows_var).to_csv(out / "varpart.tsv", sep="\t", index=False)
        try:
            plot_varpart(rows_var, focal, out)
        except Exception as e:  # noqa: BLE001
            say(f"varpart figure not drawn ({e})")
    if rows_pair:
        pd.DataFrame(rows_pair).to_csv(out / "paired.tsv", sep="\t", index=False)
    if rows_sig:
        pd.DataFrame(rows_sig).to_csv(out / "signal.tsv", sep="\t", index=False)

    # ---- community, sites as replicates
    if "community" in a.analyses and phy:
        site_sp = meta.groupby(a.site_col)[a.species_col].agg(lambda s: sorted(set(s)))
        pool = phy.species
        tips_of_species = defaultdict(list)                  # tip-map mode: a random community draws distinct
        if phy.tipmap:                                       # species, each as one of its own sequences
            for t in ph.tips_of(phy.root):
                if t.name in phy.tipmap:
                    tips_of_species[phy.tipmap[t.name][0]].append(t.name)
            pool = sorted(tips_of_species)
        trait_sp = {}
        for name, (d, cols) in sets.items():
            m = meta.loc[d["_meta"].astype(int)]
            X = pd.DataFrame(d[cols].to_numpy(float), columns=cols)
            X["_sp"] = m[a.species_col].values
            mm = X.groupby("_sp").mean().dropna()
            if len(mm) >= 3:
                Z, _, _, _, _, _ = pca(mm.to_numpy(), max_k=min(10, len(mm) - 1))
                trait_sp[name] = pd.DataFrame(Z, index=mm.index)
        rows, site_rows = [], []
        for site, spp in site_sp.items():
            r = {"site": site, focal: site_level.get(site), "richness": len(spp)}
            for c in a.covariate:
                r[c] = pd.to_numeric(meta.loc[meta[a.site_col] == site, c], errors="coerce").mean()
            com = phy.site_community(str(site), spp)
            if com is not None:
                names, Dd, pd_ = com
                if a.save_matrices:
                    site_rows.append({"site": site, "tips": ",".join(_u(n_) for n_ in names), "PD": pd_})
                mpd, mntd = mpd_mntd(Dd)
                r.update(n_in_tree=len(names), PD=pd_, MPD=mpd, MNTD=mntd)
                if site not in phy.site_trees and len(names) >= 2 and len(pool) > len(names):
                    nm, nn, npd = [], [], []
                    for _ in range(a.iters):
                        draw = list(rng.choice(pool, len(names), replace=False))
                        if phy.tipmap:
                            draw = [str(rng.choice(tips_of_species[sp])) for sp in draw]
                        Dn = (phy.patristic(phy.root, draw) if phy.tipmap else phy.species_distance(draw))
                        x, y = mpd_mntd(Dn); nm.append(x); nn.append(y)
                        npd.append(faith_pd(phy.root if phy.tipmap else phy.sp_root, set(draw)))
                    r["SES_PD"], r["P_PD"] = ses(pd_, npd)
                    r["SES_MPD"], r["P_MPD"] = ses(mpd, nm)
                    r["SES_MNTD"], r["P_MNTD"] = ses(mntd, nn)
            for name, T in trait_sp.items():
                here = [s for s in spp if s in T.index]
                if len(here) >= 2:
                    Zs = T.loc[here].to_numpy()
                    r[f"FDis_{name}"] = float(np.linalg.norm(Zs - Zs.mean(0), axis=1).mean())
                    Dz = np.linalg.norm(Zs[:, None] - Zs[None], axis=2)
                    r[f"traitMPD_{name}"] = mpd_mntd(Dz)[0]
                    null = []
                    for _ in range(a.iters):
                        dr = T.to_numpy()[rng.choice(len(T), len(here), replace=False)]
                        null.append(np.linalg.norm(dr - dr.mean(0), axis=1).mean())
                    r[f"SES_FDis_{name}"], r[f"P_FDis_{name}"] = ses(r[f"FDis_{name}"], null)
            rows.append(r)
        cs = pd.DataFrame(rows)
        cs.to_csv(out / "community_sites.tsv", sep="\t", index=False)
        if a.save_matrices and site_rows:
            pd.DataFrame(site_rows).merge(cs[["site", "MPD", "MNTD"]], on="site", how="left").to_csv(
                vdir / "site_species.tsv", sep="\t", index=False)
            tree_for_sites = phy.root if phy.tipmap else phy.sp_root
            if tree_for_sites is not None:
                (vdir / "tree_sites.nwk").write_text(to_newick(tree_for_sites))
        # are these site lists community samples at all? museum or opportunistic records usually are not
        per_site = meta.groupby(a.site_col)[a.specimen_col].nunique()
        single = float((cs["richness"] <= 1).mean())
        if per_site.median() < 3 or single > 0.4:
            say(f"  WARNING: the site lists do not look like community samples (median {per_site.median():.0f} "
                f"specimen(s) per site; {100*single:.0f}% of sites hold a single species). Richness, PD, MPD, MNTD, "
                f"their SES and trait dispersion then measure collecting effort and selection, not co-occurrence; "
                f"interpret them only for standardised surveys (same method and effort at every site).")
            summary["community_warning"] = True
        comp = []
        if A is not None:
            metrics = [c for c in cs.columns if c not in ("site", focal, *a.covariate) and not c.startswith("P_")
                       and pd.api.types.is_numeric_dtype(cs[c])]
            for mtr in metrics:
                comp.append({"metric": mtr, **site_label_test(cs[mtr], cs[focal], A, B, a.iters, rng)})
            cdf = pd.DataFrame(comp)
            cdf.to_csv(out / "community_comparison.tsv", sep="\t", index=False)
            try:
                plot_community(cs, cdf, focal, A, B, out)
            except Exception as e:  # noqa: BLE001
                say(f"community figure not drawn ({e})")
            for _, r in cdf.iterrows():
                if r["metric"] in ("richness", "PD", "MPD", "MNTD", "SES_MPD", "SES_MNTD") or r["metric"].startswith("FDis"):
                    say(f"  community {r['metric']}: {A} {r['mean_A']:.3g} (n={r['n_A']}) vs {B} {r['mean_B']:.3g} "
                        f"(n={r['n_B']}) sites, permutation P={r['P']:.3g}")
        summary["sites"] = int(len(cs))

    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=str)
    (out / "report.md").write_text("# descriptron_community_v1\n\n" + "\n".join(f"- {s}" for s in log) + "\n")
    print(f"-> {out}")


def plot_varpart(rows, focal, out):
    """one bar per trait set: treatment only, shared, phylogeny only, unexplained (negative adjusted R2 shown as 0)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parts = [("treatment_only_[a]", f"{focal} only", "#D55E00"), ("shared_[b]", "shared", "#CC79A7"),
             ("phylogeny_only_[c]", "phylogeny only", "#0072B2"), ("unexplained_[d]", "unexplained", "#DDDDDD")]
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(rows) + 1.4))
    for i, r in enumerate(rows):
        left = 0.0
        for key, lab, col in parts:
            v = max(0.0, float(r[key]))
            ax.barh(i, v, left=left, color=col, edgecolor="white", label=lab if i == 0 else None)
            if v >= 0.06:
                ax.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if col != "#DDDDDD" else "#333")
            left += v
        p_a, p_c = r["P_treatment_given_phylogeny"], r["P_phylogeny_given_treatment"]
        ax.text(1.01, i, f"P {focal}={p_a:.3g}\nP phylo={p_c:.3g}", va="center", fontsize=7, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["set"] for r in rows])
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("share of trait variation (adjusted R2)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"varpart.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_community(cs, cdf, focal, A, B, out):
    """site-level boxplots by treatment level for the main community measures, with the site-permutation P"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [m for m in ["richness", "PD", "SES_PD", "MPD", "SES_MPD", "MNTD", "SES_MNTD"] +
               [c for c in cs.columns if c.startswith("FDis_")] if m in cs.columns and cs[m].notna().sum() >= 4]
    if not metrics:
        return
    pv = dict(zip(cdf["metric"], cdf["P"])) if cdf is not None and len(cdf) else {}
    n = len(metrics); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, m in zip(axs.ravel(), metrics):
        data = [cs.loc[cs[focal] == L, m].dropna().to_numpy() for L in (A, B)]
        ax.boxplot(data, widths=0.5, showfliers=False)
        for j, dd in enumerate(data, 1):
            ax.scatter(j + rng.uniform(-0.12, 0.12, len(dd)), dd, s=14, color=["#0072B2", "#D55E00"][j - 1], zorder=3)
        ax.set_xticks([1, 2]); ax.set_xticklabels([f"{A}\n(n={len(data[0])})", f"{B}\n(n={len(data[1])})"], fontsize=7)
        if m.startswith("SES"):
            ax.axhline(0, color="#999", lw=0.6, ls="--")
        ax.set_title(f"{m}   P={pv.get(m, float('nan')):.3g}", fontsize=8)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Sites as replicates: {A} vs {B} (P from permuting whole sites)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"community_by_{focal}.{ext}", dpi=200)
    plt.close(fig)


_MASKS = {}


def _thumb(image_name, a, max_px=180):
    """an RGB thumbnail of one specimen: cut out with its mask on white when a COCO mask exists, else the image"""
    from PIL import Image, ImageDraw
    base = os.path.basename(str(image_name))
    cands = [Path(a.image_dir) / base] if a.image_dir else []
    if a.image_dir and not cands[0].exists():
        stem = os.path.splitext(base)[0]
        cands = sorted(Path(a.image_dir).glob(f"{glob_escape(stem)}.*"))
    path = next((c for c in cands if c.exists()), None)
    if path is None:
        return None
    im = Image.open(path).convert("RGB")
    if not _MASKS and (a.thumb_coco or a.coco):
        for cp in (a.thumb_coco or a.coco):
            d = json.load(open(cp))
            cats = {c["id"]: c["name"] for c in d.get("categories", [])}
            imgs = {x["id"]: os.path.basename(str(x.get("file_name", ""))) for x in d.get("images", [])}
            for ann in d.get("annotations", []):
                seg = ann.get("segmentation")
                if isinstance(seg, list) and seg and (not a.structure or cats.get(ann.get("category_id")) == a.structure):
                    k = imgs.get(ann.get("image_id"))
                    if k and (k not in _MASKS or ann.get("area", 0) > _MASKS[k][1]):
                        _MASKS[k] = (seg, ann.get("area", 0))
        _MASKS.setdefault("__loaded__", None)
    seg = (_MASKS.get(path.name) or (None,))[0]
    if seg:
        m = Image.new("L", im.size, 0); dr = ImageDraw.Draw(m)
        for poly in seg:
            dr.polygon([tuple(q) for q in np.asarray(poly, float).reshape(-1, 2)], fill=255)
        bb = m.getbbox()
        white = Image.new("RGB", im.size, (255, 255, 255))
        im = Image.composite(im, white, m).crop(bb) if bb else im
    im.thumbnail((max_px, max_px))
    return np.asarray(im)


def glob_escape(s):
    import glob
    return glob.escape(s)


def plot_morphospace(units, S, ev, phy, focal, A, B, name, out, thumbs=None):
    import copy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = units.copy(); u["PC1"], u["PC2"] = S[:, 0], S[:, 1]
    spm = u.groupby("_species")[["PC1", "PC2"]].mean()
    r = ph.prune(copy.deepcopy(phy.sp_root), set(spm.index))
    tips, nodes = ph.tips_of(r), ph.internal_nodes(r)
    Yt = spm.loc[[t.name for t in tips]].to_numpy()
    anc = ph.ancestral_states(Yt, tips, nodes)
    pos = {id(t): Yt[i] for i, t in enumerate(tips)}
    pos.update({id(n): anc[i] for i, n in enumerate(nodes)})
    fig, ax = plt.subplots(figsize=(7, 6))
    for n in nodes:
        for k in n.kids:
            ax.plot(*zip(pos[id(n)], pos[id(k)]), color="#9a9a9a", lw=0.8, zorder=1)
    cols = {A: "#0072B2", B: "#D55E00"}
    for lev, g in u.groupby(focal):
        ax.scatter(g["PC1"], g["PC2"], s=22, color=cols.get(lev, "#666"), label=str(lev), zorder=3)
    n_ok = 0
    if thumbs is not None:
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        reps, a = thumbs
        fig.set_size_inches(10, 8.5)
        for i, img in enumerate(reps):
            t = _thumb(img, a) if img else None
            if t is None:
                continue
            ob = OffsetImage(t, zoom=a.thumb_size * fig.dpi / max(t.shape[:2]))
            ab = AnnotationBbox(ob, (u["PC1"].iloc[i], u["PC2"].iloc[i]), frameon=True, pad=0.15, zorder=4,
                                bboxprops=dict(edgecolor=cols.get(u[focal].iloc[i], "#666"), lw=1.6))
            ax.add_artist(ab); n_ok += 1
    if A is not None:
        pa = u[u[focal] == A].groupby("_species")[["PC1", "PC2"]].mean()
        pb = u[u[focal] == B].groupby("_species")[["PC1", "PC2"]].mean()
        for s in sorted(set(pa.index) & set(pb.index)):
            ax.annotate("", xy=pb.loc[s], xytext=pa.loc[s], arrowprops=dict(arrowstyle="->", color="#444", lw=0.8), zorder=2)
    if thumbs is None and getattr(phy, "_vdir", None) is not None:
        (phy._vdir / f"morpho_{name}_tree.nwk").write_text(to_newick(r))
        pd.DataFrame(Yt, index=[_u(t.name) for t in tips], columns=["PC1", "PC2"]).to_csv(phy._vdir / f"morpho_{name}_tips.csv")
        pd.DataFrame({"descendants": [",".join(sorted(_u(t.name) for t in ph.tips_of(n_))) for n_ in nodes],
                      "PC1": anc[:, 0], "PC2": anc[:, 1]}).to_csv(phy._vdir / f"morpho_{name}_nodes.csv", index=False)
    ax.set_xlabel(f"PC1 ({100*ev[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({100*ev[1]:.0f}%)")
    ax.set_title(f"{name}: species x {focal} means; tree through the species means;\narrows {A} -> {B} for species under both",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    if thumbs is not None:
        ax.set_title(ax.get_title() + "\nthumbnail = the specimen nearest each mean (border = level)", fontsize=9)
        ax.margins(0.12)
    fig.tight_layout()
    suffix = "_thumbnails" if thumbs is not None else ""
    for ext in ("png", "pdf"):
        fig.savefig(out / f"morphospace_{name}{suffix}.{ext}", dpi=200)
    plt.close(fig)
    return n_ok


if __name__ == "__main__":
    main()
