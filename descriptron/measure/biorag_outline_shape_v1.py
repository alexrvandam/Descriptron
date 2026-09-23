#!/usr/bin/env python3
"""
biorag_outline_shape_v1.py — outline shape as a character set the matrix can read
=================================================================================

The pipeline already measures the outline of every annotated structure, but the
identification matrix never sees it: the legacy GPA step writes `shape_PC*`
columns that (a) come from two separate GPAs (the 25-species batch and the
4-species batch) so they are not comparable, and (b) cannot align a MIRRORED
closed outline — its `reflect()` negates the coordinates without reversing the
point order, so a left forewing never lands on a right one. About a fifth of the
forewings here are mirror images of the rest and a quarter to two fifths of the
tibiae and labial segments lie end for end, so the legacy PCs mix real shape
with which way round the slide was mounted.

This script builds outline shape again, from the polygons, in a form the
hold-out harness can use honestly:

  1  one closed outline per specimen x structure (the largest polygon),
     resampled to n equally spaced points by arc length, centred, scaled to
     unit centroid size and forced counter-clockwise;

  2  REGISTRATION — done ONCE on the full sample, and label-free. Each outline
     is tried as it is and MIRRORED (x -> -x with the point order reversed, so
     it is counter-clockwise again), at every cyclic starting point, and fitted
     by rotation only (no reflection inside the Procrustes step) to the running
     consensus; the best candidate is kept and the consensus is recomputed until
     it stops moving. This only fixes CORRESPONDENCE — which point of one
     outline answers to which point of another — exactly as a morphometrician
     fixes it by digitising landmarks in an agreed order. It uses no species
     labels, so it cannot leak them. Outlines whose two best candidates are
     nearly tied (near-symmetric structures) are flagged `ambiguous`;

  3  PER-FOLD COORDINATES — the hold-out invariant. The consensus a specimen is
     aligned to is a statistic of the sample, so it must be rebuilt without
     whatever is withheld. `FoldAligner.frame(exclude)` computes the consensus
     from the TRAINING outlines only (rotation-only GPA, seeded on the first
     training outline, which cannot depend on the withheld data) and returns
     every specimen's coordinates aligned to that consensus. Changing a withheld
     specimen's outline therefore cannot change its own score.

No PCA by default: the Reference compares character sets by per-feature
standardised differences with a pooled within-species spread it re-derives under
hold-out, so 2n aligned coordinates per structure behave as a scaled Procrustes
distance. `--basis pca --n_pcs k` fits the basis on the training outlines only
and projects everyone, for comparison.

Usage:
  python biorag_outline_shape_v1.py \\
      --coco "<unified COCO with every structure>" \\
      --identity_dir "<compiled dir whose *_full_features.csv carries specimen_id>" \\
      --matrix_dir "<compiled_key_tier>" \\
      --taxon_profile biorag_prompts/taxon_profiles/diaphorina_taxon_profile.yaml \\
      --n_points 64 --out_dir "<out>"

Outputs
  outline_registration.tsv        per outline: specimen, structure, mirrored, shift, both
                                  Procrustes distances, ambiguity margin
  outline_registered.npz          registered outlines + ids + structures (input to the harness)
  outline_shape_fullsample.tsv    wide table aligned to the FULL-SAMPLE consensus — for
                                  inspection only; never score with it (it is not a hold-out)
  outline_shape_summary.json      per structure: specimens, mirrored, ambiguous, matrix match
  outline_wing_handedness.tsv     mirror calls against an independent handedness ground truth
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_feature_policy as pol                                       # noqa: E402

VERSION = "1.0"
DEFAULT_N_POINTS = 64
# An outline is `ambiguous` when its best as-is fit and its best mirrored fit are
# within this relative gap. 0.25 is not arbitrary: checked against the forewing
# handedness read from the cell geometry (see wing_handedness), every structure
# whose outlines register with a median gap above it recovers handedness at
# 98-100%, and every structure below it at chance.
DEFAULT_AMBIGUOUS_MARGIN = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# names
# ─────────────────────────────────────────────────────────────────────────────
def norm_image(x) -> str:
    """Image keys differ between the COCO ('Scan 001') and the compiled table
    ('Scan_001') only in whitespace and the extension. Punctuation is KEPT: two
    files here differ only in a dot ('cf.carrisae' vs 'cfcarrisae')."""
    s = str(x).strip().lower()
    s = re.sub(r'\.(tif|tiff|jpg|jpeg|png|bmp)$', '', s)
    return re.sub(r'\s+', '_', s)


def norm_category(x) -> str:
    """The rule biorag_key_feature_filter_v2.merge_category_aliases groups
    aliases by: letters and digits only, so 'cell-c+sc' and 'cell-csc' agree."""
    return re.sub(r'[^a-z0-9]', '', str(x).lower())


def strip_sex(cat: str) -> str:
    return re.sub(r'__(male|female)$', '', str(cat))


def load_identity(identity_dir: Path, profile: Dict) -> Tuple[Dict, Dict]:
    """(image, structure) -> the (specimen_id, species, matrix category) rows the
    compiled matrix built from it.

    A list, not a single value: in this data set a few images are filed under two
    species codes at once (sp1 / sp1A), and the matrix carries BOTH specimens
    built from the same picture. Replicating that keeps outline shape on exactly
    the same specimens as the four measured sets."""
    src = sorted(Path(identity_dir).glob("*_full_features.csv"))
    if not src:
        raise FileNotFoundError(f"no *_full_features.csv in {identity_dir}")
    f = pd.read_csv(src[0], low_memory=False)
    if "specimen_id" not in f.columns:
        f["specimen_id"] = [pol.specimen_id(ib, sp, profile)
                            for ib, sp in zip(f["image_base"], f["group_label"])]
    aliases = {norm_category(k): v for k, v in (profile.get("category_aliases") or {}).items()}
    idmap: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = defaultdict(list)
    for r in f.itertuples():
        base = strip_sex(r.category)
        key = (norm_image(r.image_base), norm_category(aliases.get(norm_category(base), base)))
        tgt = (r.specimen_id, r.group_label, r.category)
        if tgt not in idmap[key]:
            idmap[key].append(tgt)
    return dict(idmap), aliases


# ─────────────────────────────────────────────────────────────────────────────
# outlines
# ─────────────────────────────────────────────────────────────────────────────
def signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def resample_closed(pts: np.ndarray, n: int) -> Optional[np.ndarray]:
    """n points equally spaced along the closed perimeter."""
    keep = np.r_[True, (np.abs(np.diff(pts, axis=0)).sum(axis=1) > 1e-9)]
    p = pts[keep]
    if len(p) < 3:
        return None
    p = np.vstack([p, p[:1]])
    seg = np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return None
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def normalise_shape(p: np.ndarray) -> Optional[np.ndarray]:
    """Force counter-clockwise, then centre and scale to unit centroid size."""
    q = p[::-1].copy() if signed_area(p) < 0 else p
    c = q - q.mean(axis=0)
    cs = math.sqrt(float((c ** 2).sum()))
    if not (cs > 0):
        return None
    return c / cs


def mirror_shape(p: np.ndarray) -> np.ndarray:
    """x -> -x AND the point order reversed, so the outline is counter-clockwise
    again. Negating x alone (what the legacy GPA does) leaves the traversal
    running backwards, and no rotation can then bring it onto the others."""
    q = p[::-1].copy()
    q[:, 0] = -q[:, 0]
    return q


def outlines_from_coco(coco_path: Path, idmap: Dict, aliases: Dict, n_points: int,
                       log=print) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict]:
    """structure -> specimen_id -> normalised outline (n_points, 2)."""
    j = json.loads(Path(coco_path).read_text())
    cats = {c["id"]: c["name"] for c in j.get("categories", [])}
    imgs = {i["id"]: i["file_name"] for i in j.get("images", [])}
    best: Dict[Tuple[str, str], Tuple[float, np.ndarray, str]] = {}
    stats = {"annotations": len(j.get("annotations", [])), "unmapped": 0,
             "unusable_polygon": 0, "used": 0}
    unmapped_by_cat: Dict[str, int] = defaultdict(int)
    for a in j.get("annotations", []):
        seg = a.get("segmentation")
        if not isinstance(seg, list) or not seg:
            stats["unusable_polygon"] += 1
            continue
        if isinstance(seg[0], (int, float)):                  # a bare flat list
            seg = [seg]
        rings = []
        for s in seg:
            if not isinstance(s, (list, tuple)) or len(s) < 8:
                continue
            q = np.asarray(s, dtype=float).reshape(-1, 2)
            if len(q) >= 4:
                rings.append(q)
        if not rings:
            stats["unusable_polygon"] += 1
            continue
        ring = max(rings, key=lambda q: abs(signed_area(q)))
        raw_cat = cats.get(a["category_id"], "?")
        ckey = norm_category(raw_cat)
        ckey = norm_category(aliases.get(ckey, raw_cat)) if ckey in aliases else ckey
        key = (norm_image(imgs.get(a["image_id"], "")), ckey)
        targets = idmap.get(key)
        if not targets:
            stats["unmapped"] += 1
            unmapped_by_cat[raw_cat] += 1
            continue
        area = abs(signed_area(ring))
        res = resample_closed(ring, n_points)
        if res is None:
            stats["unusable_polygon"] += 1
            continue
        shp = normalise_shape(res)
        if shp is None:
            stats["unusable_polygon"] += 1
            continue
        stats["used"] += 1
        for sid, _species, matrix_cat in targets:
            k = (matrix_cat, sid)
            if k not in best or area > best[k][0]:
                best[k] = (area, shp, imgs.get(a["image_id"], ""))
    out: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    prov: Dict[Tuple[str, str], str] = {}
    for (cat, sid), (_area, shp, img) in best.items():
        out[cat][sid] = shp
        prov[(cat, sid)] = img
    stats["unmapped_by_category"] = dict(sorted(unmapped_by_cat.items(),
                                                key=lambda kv: -kv[1]))
    log(f"COCO: {stats['annotations']} annotations, {stats['used']} usable and matched, "
        f"{stats['unmapped']} on image x structure rows the compiled matrix does not carry "
        f"(dropped upstream by the exclusion list / annotation screen), "
        f"{stats['unusable_polygon']} unusable polygons")
    return {k: v for k, v in sorted(out.items())}, {"coco": stats, "provenance": prov}


# ─────────────────────────────────────────────────────────────────────────────
# Procrustes (rotation only — never a reflection inside the fit)
# ─────────────────────────────────────────────────────────────────────────────
def _rot_terms(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For A = Y^T C, the best proper rotation R maximises trace(R^T A); the
    maximum is hypot(A00+A11, A10-A01) and R is read straight off it."""
    u = A[..., 0, 0] + A[..., 1, 1]
    v = A[..., 1, 0] - A[..., 0, 1]
    return u, v


def align_to(Y: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, float]:
    """Rotate Y (unit size, centred) onto C. Returns (aligned, Procrustes distance)."""
    A = Y.T @ C
    u, v = _rot_terms(A)
    m = math.hypot(float(u), float(v))
    if m <= 0:
        return Y.copy(), math.sqrt(2.0)
    c, s = float(u) / m, float(v) / m
    R = np.array([[c, -s], [s, c]])
    return Y @ R, math.sqrt(max(0.0, 2.0 - 2.0 * m))


def align_batch(S: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate every shape in S (N, n, 2) onto C (n, 2)."""
    A = np.einsum('nkp,kq->npq', S, C)
    u, v = _rot_terms(A)
    m = np.hypot(u, v)
    safe = np.where(m > 0, m, 1.0)
    c, s = u / safe, v / safe
    R = np.stack([np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)], axis=-2)
    return np.einsum('nkp,npq->nkq', S, R), np.sqrt(np.maximum(0.0, 2.0 - 2.0 * m))


def _roll_index(n: int) -> np.ndarray:
    k = np.arange(n)
    return (k[None, :] + k[:, None]) % n            # IDX[s, k] = (k + s) % n


def _best_over_shifts(S: np.ndarray, C: np.ndarray, IDX: np.ndarray):
    """For each shape: the best cyclic starting point against C.
    Returns (best shift, Procrustes distance at it)."""
    Ys = S[:, IDX]                                   # (N, n_shifts, n, 2)
    u = np.einsum('nskp,kp->ns', Ys, C)
    v = (np.einsum('nsk,k->ns', Ys[..., 1], C[:, 0])
         - np.einsum('nsk,k->ns', Ys[..., 0], C[:, 1]))
    m = np.hypot(u, v)
    best = np.argmax(m, axis=1)
    mm = m[np.arange(len(S)), best]
    return best, np.sqrt(np.maximum(0.0, 2.0 - 2.0 * mm))


def _elongation(p: np.ndarray) -> float:
    sv = np.linalg.svd(p, compute_uv=False)
    return float(sv[0] / sv[1]) if sv[1] > 0 else math.inf


def register_structure(ids: Sequence[str], shapes: np.ndarray, n_iter: int = 5,
                       tol: float = 1e-10, margin: float = DEFAULT_AMBIGUOUS_MARGIN):
    """Label-free registration of one structure: mirror x cyclic shift x rotation,
    iterated against the consensus. Returns registered shapes and the calls made."""
    N, n, _ = shapes.shape
    IDX = _roll_index(n)
    mirrored_shapes = np.stack([mirror_shape(shapes[i]) for i in range(N)])
    seed = int(np.argsort([_elongation(shapes[i]) for i in range(N)])[N // 2])
    C = shapes[seed].copy()
    mirror_flag = np.zeros(N, dtype=bool)
    shift = np.zeros(N, dtype=int)
    d_as, d_mi = np.zeros(N), np.zeros(N)
    reg = shapes.copy()
    rounds = 0
    for it in range(n_iter):
        rounds = it + 1
        s_a, d_a = _best_over_shifts(shapes, C, IDX)
        s_m, d_m = _best_over_shifts(mirrored_shapes, C, IDX)
        use_mirror = d_m < d_a
        for i in range(N):
            src = mirrored_shapes[i] if use_mirror[i] else shapes[i]
            sh = int(s_m[i] if use_mirror[i] else s_a[i])
            reg[i] = src[IDX[sh]]
        reg, _ = align_batch(reg, C)
        mirror_flag, shift, d_as, d_mi = use_mirror, np.where(use_mirror, s_m, s_a), d_a, d_m
        newC = reg.mean(axis=0)
        newC = newC - newC.mean(axis=0)
        cs = math.sqrt(float((newC ** 2).sum()))
        if cs <= 0:
            break
        newC = newC / cs
        _, moved = align_to(newC, C)
        C = newC
        if moved < tol:
            break
    gap = np.abs(d_as - d_mi)
    mean_d = (d_as + d_mi) / 2.0
    ambiguous = gap < margin * np.where(mean_d > 0, mean_d, 1.0)
    return {"ids": list(ids), "shapes": reg, "consensus": C, "mirrored": mirror_flag,
            "shift": shift, "d_as_is": d_as, "d_mirrored": d_mi, "ambiguous": ambiguous,
            "rounds": rounds, "seed_specimen": ids[seed]}


# ─────────────────────────────────────────────────────────────────────────────
# per-fold coordinates — the hold-out part
# ─────────────────────────────────────────────────────────────────────────────
class FoldAligner:
    """Registered outlines -> the coordinate table of one fold.

    The consensus every specimen is aligned to is rebuilt from the TRAINING
    outlines alone, seeded on the first training outline in sorted order (never
    on the withheld one, and never on a full-sample mean, which the withheld one
    would have helped make). Registration — mirror and starting point — is fixed
    once on the full sample and is NOT recomputed per fold: it only decides which
    point answers to which, uses no labels, and re-deciding it per fold would
    make 178 tables that cannot be compared."""

    def __init__(self, reg: Dict[str, Dict], structures: Optional[Sequence[str]] = None,
                 basis: str = "coords", n_pcs: int = 10, gpa_rounds: int = 5,
                 tol: float = 1e-12):
        self.reg = reg
        self.structures = list(structures) if structures else sorted(reg)
        self.basis, self.n_pcs = basis, int(n_pcs)
        self.gpa_rounds, self.tol = int(gpa_rounds), float(tol)
        self.specimens = sorted({s for st in self.structures for s in reg[st]["ids"]})
        self._index = {st: {sid: i for i, sid in enumerate(reg[st]["ids"])}
                       for st in self.structures}
        self.columns: List[str] = []
        for st in self.structures:
            n = reg[st]["shapes"].shape[1]
            if basis == "pca":
                self.columns += [f"{st}.pc{k:02d}" for k in range(self.n_pcs)]
            else:
                self.columns += [f"{st}.x{i:02d}" for i in range(n)] \
                                + [f"{st}.y{i:02d}" for i in range(n)]

    # ── the consensus of one fold ────────────────────────────────────────────
    def _consensus(self, S: np.ndarray) -> np.ndarray:
        C = S[0].copy()
        for _ in range(self.gpa_rounds):
            A, _ = align_batch(S, C)
            newC = A.mean(axis=0)
            newC = newC - newC.mean(axis=0)
            cs = math.sqrt(float((newC ** 2).sum()))
            if cs <= 0:
                break
            newC = newC / cs
            _, moved = align_to(newC, C)
            C = newC
            if moved < self.tol:
                break
        return C

    def frame(self, exclude: Sequence[str] = ()) -> pd.DataFrame:
        """Every specimen's coordinates aligned to the consensus of the training
        outlines (everything except `exclude`)."""
        drop = set(exclude)
        out = pd.DataFrame(np.nan, index=self.specimens, columns=self.columns, dtype=float)
        for st in self.structures:
            R = self.reg[st]
            ids = R["ids"]
            S = R["shapes"]
            train = [i for i, sid in enumerate(ids) if sid not in drop]
            if len(train) < 1:
                continue
            first = min(train, key=lambda i: ids[i])
            order = [first] + [i for i in train if i != first]
            C = self._consensus(S[order])
            A, _ = align_batch(S, C)
            if self.basis == "pca":
                X = A.reshape(len(ids), -1)
                Xt = X[train]
                mu = Xt.mean(axis=0)
                k = min(self.n_pcs, len(train) - 1, Xt.shape[1])
                if k < 1:
                    continue
                _, _, Vt = np.linalg.svd(Xt - mu, full_matrices=False)
                P = (X - mu) @ Vt[:k].T
                cols = [f"{st}.pc{j:02d}" for j in range(k)]
                out.loc[ids, cols] = P
            else:
                n = S.shape[1]
                cols = [f"{st}.x{i:02d}" for i in range(n)] + [f"{st}.y{i:02d}" for i in range(n)]
                out.loc[ids, cols] = np.concatenate([A[:, :, 0], A[:, :, 1]], axis=1)
        return out


class FoldTables:
    """`frame()` with a small cache. The harness walks specimens in index order,
    so consecutive specimens of one species share a detection fold; four frames
    is plenty and keeps the memory flat."""

    def __init__(self, aligner: FoldAligner, max_cached: int = 4):
        self.aligner = aligner
        self.max_cached = int(max_cached)
        self._cache: Dict[frozenset, pd.DataFrame] = {}
        self._order: List[frozenset] = []
        self.builds = 0

    def get(self, exclude: Sequence[str]) -> pd.DataFrame:
        key = frozenset(exclude)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        t = self.aligner.frame(sorted(key))
        self.builds += 1
        self._cache[key] = t
        self._order.append(key)
        while len(self._order) > self.max_cached:
            self._cache.pop(self._order.pop(0), None)
        return t


# ─────────────────────────────────────────────────────────────────────────────
# save / load
# ─────────────────────────────────────────────────────────────────────────────
def save_registered(path: Path, reg: Dict[str, Dict], meta: Dict):
    blob = {"__meta__": json.dumps(meta), "__structures__": np.array(sorted(reg), dtype=object)}
    for st, R in reg.items():
        blob[f"{st}::shapes"] = R["shapes"].astype(np.float64)
        blob[f"{st}::ids"] = np.array(R["ids"], dtype=object)
        blob[f"{st}::consensus"] = R["consensus"]
        blob[f"{st}::mirrored"] = R["mirrored"]
        blob[f"{st}::shift"] = R["shift"]
        blob[f"{st}::ambiguous"] = R["ambiguous"]
        blob[f"{st}::d_as_is"] = R["d_as_is"]
        blob[f"{st}::d_mirrored"] = R["d_mirrored"]
    np.savez_compressed(path, **blob)


def load_registered(path: Path) -> Tuple[Dict[str, Dict], Dict]:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["__meta__"]))
    reg = {}
    for st in [str(s) for s in z["__structures__"]]:
        reg[st] = {"ids": [str(s) for s in z[f"{st}::ids"]],
                   "shapes": z[f"{st}::shapes"],
                   "consensus": z[f"{st}::consensus"],
                   "mirrored": z[f"{st}::mirrored"],
                   "shift": z[f"{st}::shift"],
                   "ambiguous": z[f"{st}::ambiguous"],
                   "d_as_is": z[f"{st}::d_as_is"],
                   "d_mirrored": z[f"{st}::d_mirrored"]}
    return reg, meta


# ─────────────────────────────────────────────────────────────────────────────
# an independent check on the mirror calls: forewing handedness
# ─────────────────────────────────────────────────────────────────────────────
def wing_handedness(coco_path: Path, idmap: Dict, aliases: Dict,
                    base=("cell-cu2", "cell-a"), apex=("cell-r2", "cell-Rs"),
                    third="pterostigma", wing="whole_wing") -> pd.DataFrame:
    """Which way round a forewing lies, read from the cells and not from the
    outline: the sign of cross(base -> apex, base -> pterostigma) using polygon
    centroids splits the wings into two handedness classes. Nothing in it comes
    from the registration, so it is a real check on the `mirrored` flag."""
    j = json.loads(Path(coco_path).read_text())
    cats = {c["id"]: c["name"] for c in j.get("categories", [])}
    imgs = {i["id"]: i["file_name"] for i in j.get("images", [])}
    cent: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for a in j.get("annotations", []):
        seg = a.get("segmentation")
        if not isinstance(seg, list) or not seg:
            continue
        if isinstance(seg[0], (int, float)):
            seg = [seg]
        rings = [np.asarray(s, float).reshape(-1, 2) for s in seg
                 if isinstance(s, (list, tuple)) and len(s) >= 8]
        if not rings:
            continue
        ring = max(rings, key=lambda q: abs(signed_area(q)))
        cent[norm_image(imgs.get(a["image_id"], ""))][cats.get(a["category_id"], "?")] = \
            ring.mean(axis=0)
    rows = []
    for img, cs in cent.items():
        b = next((cs[c] for c in base if c in cs), None)
        t = next((cs[c] for c in apex if c in cs), None)
        p = cs.get(third)
        if b is None or t is None or p is None:
            continue
        v1, v2 = t - b, p - b
        cr = float(v1[0] * v2[1] - v1[1] * v2[0])
        targets = idmap.get((img, norm_category(wing)), [])
        for sid, species, cat in targets:
            rows.append({"image": img, "specimen_id": sid, "species": species,
                         "structure": cat, "cross": cr, "handedness": "A" if cr > 0 else "B"})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Outline shape as a character set (hold-out safe)")
    ap.add_argument("--coco", required=True)
    ap.add_argument("--identity_dir", required=True,
                    help="compiled dir whose *_full_features.csv maps image x structure to "
                         "specimen_id (use the same dir the matrix was built from, so the "
                         "exclusion list and the category aliases are already applied)")
    ap.add_argument("--matrix_dir", default=None,
                    help="compiled_key_tier — only to report how many matrix specimens matched")
    ap.add_argument("--taxon_profile", default=None)
    ap.add_argument("--n_points", type=int, default=DEFAULT_N_POINTS)
    ap.add_argument("--gpa_rounds", type=int, default=5)
    ap.add_argument("--ambiguous_margin", type=float, default=DEFAULT_AMBIGUOUS_MARGIN)
    ap.add_argument("--min_specimens", type=int, default=4,
                    help="structures with fewer outlines than this are dropped")
    ap.add_argument("--basis", choices=["coords", "pca"], default="coords")
    ap.add_argument("--n_pcs", type=int, default=10)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = pol.load_taxon_profile(a.taxon_profile)
    idmap, aliases = load_identity(Path(a.identity_dir), profile)
    print(f"identity map: {len(idmap)} image x structure rows, "
          f"{len({t[0] for v in idmap.values() for t in v})} specimens")

    shapes_by_struct, info = outlines_from_coco(Path(a.coco), idmap, aliases, a.n_points)

    matrix_ids = None
    if a.matrix_dir:
        L = pd.read_csv(Path(a.matrix_dir) / "specimen_matrix_long.csv")
        matrix_ids = set(L["specimen_id"])
        print(f"matrix: {len(matrix_ids)} specimens, {L['species'].nunique()} species")

    reg, rows, summary = {}, [], {}
    for st, d in shapes_by_struct.items():
        if len(d) < a.min_specimens:
            print(f"  {st}: only {len(d)} outlines — dropped")
            continue
        ids = sorted(d)
        S = np.stack([d[i] for i in ids])
        R = register_structure(ids, S, n_iter=a.gpa_rounds, margin=a.ambiguous_margin)
        reg[st] = R
        matched = len([i for i in ids if matrix_ids is None or i in matrix_ids])
        gap = np.abs(R["d_as_is"] - R["d_mirrored"]) / np.maximum(
            1e-12, (R["d_as_is"] + R["d_mirrored"]) / 2.0)
        summary[st] = {"specimens": len(ids), "in_matrix": matched,
                       "mirrored": int(R["mirrored"].sum()),
                       "ambiguous": int(R["ambiguous"].sum()),
                       "ambiguous_fraction": round(float(R["ambiguous"].mean()), 3),
                       "median_relative_gap": round(float(np.median(gap)), 3),
                       "gpa_rounds": R["rounds"], "seed_specimen": R["seed_specimen"],
                       "median_procrustes_to_consensus":
                           round(float(np.median(np.minimum(R["d_as_is"], R["d_mirrored"]))), 4)}
        print(f"  {st}: {len(ids)} outlines ({matched} in the matrix), "
              f"{int(R['mirrored'].sum())} mirrored, {int(R['ambiguous'].sum())} ambiguous")
        for k, sid in enumerate(ids):
            rows.append({"structure": st, "specimen_id": sid,
                         "image": info["provenance"].get((st, sid), ""),
                         "mirrored": bool(R["mirrored"][k]), "shift": int(R["shift"][k]),
                         "d_as_is_best": round(float(R["d_as_is"][k]), 6),
                         "d_mirrored_best": round(float(R["d_mirrored"][k]), 6),
                         "relative_gap": round(float(abs(R["d_as_is"][k] - R["d_mirrored"][k]) /
                                                     max(1e-12, (R["d_as_is"][k] + R["d_mirrored"][k]) / 2)), 4),
                         "ambiguous": bool(R["ambiguous"][k])})
    if not reg:
        sys.exit("no structure had enough outlines")
    regtab = pd.DataFrame(rows)
    regtab.to_csv(out / "outline_registration.tsv", sep="\t", index=False)

    meta = {"version": VERSION, "n_points": a.n_points, "coco": str(a.coco),
            "identity_dir": str(a.identity_dir), "ambiguous_margin": a.ambiguous_margin,
            "coco_stats": info["coco"]}
    save_registered(out / "outline_registered.npz", reg, meta)

    aligner = FoldAligner(reg, basis=a.basis, n_pcs=a.n_pcs, gpa_rounds=a.gpa_rounds)
    full = aligner.frame(())
    full.index.name = "specimen_id"
    full.to_csv(out / "outline_shape_fullsample.tsv", sep="\t")

    # the independent handedness check — for every structure of the forewing, not
    # only the membrane: the mirror call is only as good as the outline is chiral
    hand = wing_handedness(Path(a.coco), idmap, aliases)
    agreement = {}
    if len(hand):
        hsp = hand.drop_duplicates("specimen_id").set_index("specimen_id")["handedness"]
        himg = set(hand["image"])
        print(f"forewing handedness from the cell geometry: "
              f"{int((hsp == 'A').sum())} vs {int((hsp == 'B').sum())} specimens")
        for st, R in reg.items():
            # only structures cut from the forewing pictures: the handedness of a
            # specimen's wing says nothing about how its rostrum slide was laid down
            on_wing = [norm_image(info["provenance"].get((st, s), "")) in himg for s in R["ids"]]
            if np.mean(on_wing) < 0.8:
                continue
            flag = pd.Series(R["mirrored"], index=R["ids"]).reindex(hsp.index).dropna()
            if len(flag) < 10:
                continue
            same = float((flag.astype(bool) == (hsp.loc[flag.index] == "A")).mean())
            agreement[st] = {"specimens": int(len(flag)),
                             "mirrored_called": int(flag.astype(bool).sum()),
                             "agreement_up_to_a_global_swap": round(max(same, 1 - same), 3),
                             "median_relative_gap": summary[st]["median_relative_gap"],
                             "ambiguous_fraction": summary[st]["ambiguous_fraction"]}
            print(f"  {st:26s} agreement {agreement[st]['agreement_up_to_a_global_swap']:.3f} "
                  f"(median gap {agreement[st]['median_relative_gap']:.3f}, "
                  f"ambiguous {100 * agreement[st]['ambiguous_fraction']:.0f}%)")
        pd.DataFrame([dict(structure=k, **v) for k, v in agreement.items()]).to_csv(
            out / "outline_handedness_agreement.tsv", sep="\t", index=False)
    hand.to_csv(out / "outline_wing_handedness.tsv", sep="\t", index=False)

    js = {"version": VERSION, "n_points": a.n_points, "basis": a.basis,
          "structures": summary, "coco": info["coco"],
          "specimens_with_any_outline": len(aligner.specimens),
          "matrix_specimens": len(matrix_ids) if matrix_ids else None,
          "matrix_specimens_with_any_outline":
              len([s for s in aligner.specimens if matrix_ids is None or s in matrix_ids]),
          "wing_handedness_check": agreement,
          "note": "outline_shape_fullsample.tsv is aligned to the full-sample consensus and is "
                  "for inspection only; scoring uses FoldAligner.frame(exclude), which rebuilds "
                  "the consensus without whatever is withheld."}
    (out / "outline_shape_summary.json").write_text(json.dumps(js, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
