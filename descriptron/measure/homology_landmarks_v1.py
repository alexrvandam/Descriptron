#!/usr/bin/env python3
"""
homology_landmarks_v1.py  (4 Nov 2026)
======================================
Landmarks AND semilandmarks as TPS control points for the homology grids.

The colour, texture and probe homology grids are drawn once on the GPA consensus and thin-plate-spline warped onto
each specimen. Until now the warp used only the outline semilandmarks (every --landmark_step-th), so interior
structures - vein junctions, eyespot centres, sutures, setal bases - were placed only by interpolation from the
outline. With this module the specimen's fixed landmarks (COCO keypoints, as the GUI saves them) are added to the
control points, so the grid is pinned at the outline AND at every landmark the specimen has.

  1  landmarks per specimen  keypoint annotations of the same image (any category with keypoints, or the ones named);
                             only points on or within --landmark_tol_px of the structure's own polygon are kept, so
                             landmarks of other structures in the same photograph are ignored; v = 0 is missing
  2  consensus landmarks     each specimen's landmarks carried into the consensus frame by the similarity transform
                             (Procrustes) that maps its outline onto the consensus outline - reflected when the outline
                             runs the other way round (a mirror image, as V42's reflect_mirrored writes it); the
                             consensus position is their MEDIAN. A landmark is used only when at least
                             --landmark_min_share of the specimens have it
     check                   a specimen whose landmarks sit far from the consensus once its outline is aligned
                             (median distance > 3 x the typical one and > 10 % of the outline radius) has an outline
                             that does not correspond to the others (an old GPA without mirror handling, a mis-ordered
                             outline) or wrong landmarks: it is named in the log and warped from its outline alone
  3  control points          outline semilandmarks (as before) + every landmark this specimen has and the consensus
                             has, same TPS (scipy RBFInterpolator, thin plate spline, smoothing 1e-5)

With no keypoints for the structure, nothing changes (the controls are the outline semilandmarks only).
Used by color_phenomics_homology_v2_1.py, texture_phenomics_homology.py and biorag_homology_frame_v1.py.
"""
import json
import logging
import os
import re

import numpy as np


def _stem(name):
    """'sp23_7902.jpg_240' -> 'sp23_7902'; 'dir/sp23_7902.tif' -> 'sp23_7902'."""
    b = os.path.basename(str(name))
    m = re.match(r'^(.+\.\w+)_\d+$', b)
    b = m.group(1) if m else b
    return os.path.splitext(b)[0]


def _poly_dist(pts, poly):
    """Signed distance of points to a closed polygon (negative inside), via cv2.pointPolygonTest."""
    import cv2
    c = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    return np.array([-cv2.pointPolygonTest(c, (float(x), float(y)), True) for x, y in pts])


def _signed_area(p):
    p = np.asarray(p, float); x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def similarity_fit(src, dst, allow_reflection=True):
    """Least-squares similarity transform src -> dst (both n x 2, same point order). Returns f(points) -> points.
    allow_reflection: True = reflect when the outline runs the other way round (opposite signed area: a mirrored
    specimen, as the GPA's reflect_mirrored treats it); False = never. The choice is NOT made by the better fit, which
    could swap left and right landmarks on a symmetric structure."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    ms, md = src.mean(0), dst.mean(0); a, b = src - ms, dst - md
    mirrored = allow_reflection and _signed_area(src) * _signed_area(dst) < 0
    best = None
    for refl in ((True,) if mirrored else (False,)):
        aa = a * np.array([1, -1]) if refl else a
        u, s, vt = np.linalg.svd(aa.T @ b)
        r = u @ vt
        if np.linalg.det(r) < 0:                      # keep a proper rotation; reflection only through aa
            u[:, -1] *= -1; s[-1] *= -1; r = u @ vt
        sc = s.sum() / (aa ** 2).sum()
        res = ((sc * aa @ r - b) ** 2).sum()
        if best is None or res < best[0]:
            best = (res, refl, sc, r)
    _, refl, sc, r = best

    def f(p):
        p = np.asarray(p, float) - ms
        if refl:
            p = p * np.array([1, -1])
        return sc * p @ r + md
    return f


class LandmarkControls:
    """Fixed landmarks to add to the outline semilandmarks of a homology-grid TPS.

    lc = LandmarkControls.from_coco(json_path, specimens, mean_shape, ...)
    cm, cs = lc.controls(fn_gpa, ctrl_mean, ctrl_spec)   # appended control points (mean frame, specimen image)
    """

    def __init__(self, spec_lm, mean_lm, names=None, info=None):
        self.spec_lm = spec_lm            # fn_gpa -> (K, 2) image coords, NaN = missing
        self.mean_lm = mean_lm            # (K, 2) consensus frame, NaN = not used
        self.names = names or [f"lm{i + 1}" for i in range(len(mean_lm))]
        self.info = info or {}

    @property
    def n_used(self):
        return int(np.isfinite(self.mean_lm[:, 0]).sum()) if self.mean_lm is not None and len(self.mean_lm) else 0

    def controls(self, fn_gpa, ctrl_mean, ctrl_spec):
        if not self.n_used or fn_gpa not in self.spec_lm:
            return ctrl_mean, ctrl_spec
        s = self.spec_lm[fn_gpa]
        ok = np.isfinite(s[:, 0]) & np.isfinite(self.mean_lm[:, 0])
        if not ok.any():
            return ctrl_mean, ctrl_spec
        return np.vstack([ctrl_mean, self.mean_lm[ok]]), np.vstack([ctrl_spec, s[ok]])

    def n_for(self, fn_gpa):
        s = self.spec_lm.get(fn_gpa)
        if s is None or not self.n_used:
            return 0
        return int((np.isfinite(s[:, 0]) & np.isfinite(self.mean_lm[:, 0])).sum())

    @classmethod
    def from_coco(cls, json_path, specimens, mean_shape, categories="", tol_px=10.0, indices="",
                  min_share=0.5, allow_reflection=True):
        """specimens: fn_gpa -> outline points in image coordinates (as load_gpa_outputs returns them)."""
        empty = cls({}, np.zeros((0, 2)), info={"reason": "no keypoints"})
        if not json_path or not os.path.isfile(json_path):
            return empty
        coco = json.load(open(json_path))
        want = {c.strip().lower() for c in categories.split(",") if c.strip()}
        cat_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
        kp_names = {}
        for c in coco.get("categories", []):
            if c.get("keypoints"):
                kp_names[c["id"]] = list(c["keypoints"])
        by_stem = {}
        img_stem = {i["id"]: _stem(i["file_name"]) for i in coco.get("images", [])}
        for a in coco.get("annotations", []):
            kp = a.get("keypoints")
            if not kp or len(kp) < 3:
                continue
            if want and cat_name.get(a.get("category_id"), "").lower() not in want:
                continue
            arr = np.array(kp, float).reshape(-1, 3)
            xy = arr[:, :2].copy(); xy[arr[:, 2] <= 0] = np.nan
            by_stem.setdefault(img_stem.get(a["image_id"]), []).append((xy, a.get("category_id")))
        if not by_stem:
            return empty
        sel = [int(x) - 1 for x in re.split(r"[,\s]+", indices.strip()) if x] if indices.strip() else None
        spec_lm, K, cat_used = {}, 0, None
        for fn, poly in specimens.items():
            cands = by_stem.get(_stem(fn))
            if not cands:
                continue
            best = None
            for xy, cid in cands:                       # the keypoint set lying on this structure
                if sel is not None:
                    xy = xy[[i for i in sel if i < len(xy)]]
                ok = np.isfinite(xy[:, 0])
                if not ok.any():
                    continue
                d = np.full(len(xy), np.inf); d[ok] = _poly_dist(xy[ok], poly)
                keep = d <= tol_px
                if best is None or keep.sum() > best[0]:
                    x2 = xy.copy(); x2[~keep] = np.nan; best = (keep.sum(), x2, cid)
            if best and best[0] > 0:
                spec_lm[fn] = best[1]; K = max(K, len(best[1])); cat_used = best[2]
        if not spec_lm:
            return cls({}, np.zeros((0, 2)), info={"reason": "keypoints found, none on this structure"})
        for fn in spec_lm:                              # pad to a common length
            if len(spec_lm[fn]) < K:
                spec_lm[fn] = np.vstack([spec_lm[fn], np.full((K - len(spec_lm[fn]), 2), np.nan)])
        mean_shape = np.asarray(mean_shape, float)
        acc = np.full((len(spec_lm), K, 2), np.nan)
        for j, (fn, lm) in enumerate(spec_lm.items()):
            sp = np.asarray(specimens[fn], float); n = min(len(sp), len(mean_shape))
            f = similarity_fit(sp[:n], mean_shape[:n], allow_reflection=allow_reflection)
            ok = np.isfinite(lm[:, 0]); acc[j, ok] = f(lm[ok])
        share = np.isfinite(acc[:, :, 0]).mean(0) * len(spec_lm) / max(len(specimens), 1)
        cs = np.sqrt(((mean_shape - mean_shape.mean(0)) ** 2).sum(1).mean())   # consensus RMS radius
        mean_lm = np.nanmedian(acc, axis=0)
        dev = np.nanmedian(np.sqrt(((acc - mean_lm[None]) ** 2).sum(-1)), axis=1) / cs     # per specimen
        bad = dev > max(3 * np.nanmedian(dev), 0.10)
        if bad.any() and (~bad).sum() >= 3:
            mean_lm = np.nanmedian(acc[~bad], axis=0)
        flagged = [fn for fn, b in zip(spec_lm, bad) if b]
        for fn in flagged:
            spec_lm[fn] = np.full_like(spec_lm[fn], np.nan)
        good = acc[~bad] if (~bad).any() else acc
        spread = np.sqrt(np.nanmean(((good - mean_lm[None]) ** 2).sum(-1), axis=0)) / cs if cs else np.full(K, np.nan)
        drop = share < min_share
        mean_lm[drop] = np.nan
        names = kp_names.get(cat_used) or [f"lm{i + 1}" for i in range(K)]
        if sel is not None and len(names) >= K:
            names = [names[i] for i in sel][:K]
        info = {"n_specimens_with_landmarks": len(spec_lm), "n_specimens": len(specimens), "n_landmarks": K,
                "n_used": int((~drop).sum()), "share_present": share.round(3).tolist(),
                "spread_pct_of_radius": (100 * spread).round(2).tolist(), "dropped": [names[i] for i in np.where(drop)[0]],
                "outline_landmark_disagree": flagged,
                "specimen_deviation_pct_of_radius": {fn: round(100 * float(d), 2) for fn, d in zip(spec_lm, dev)}}
        return cls(spec_lm, mean_lm, names=names[:K], info=info)


def add_cli(p):
    """The same options in every homology program."""
    p.add_argument('--landmarks', choices=['auto', 'none'], default='auto',
                   help="auto (default): fixed landmarks (COCO keypoints on this structure) are added to the outline "
                        "semilandmarks as TPS control points of the grid; none = outline semilandmarks only (before 4 Nov 2026)")
    p.add_argument('--landmark_json', default='',
                   help='COCO file with the keypoints (default: --json). Matched to the specimens by image name')
    p.add_argument('--landmark_categories', default='',
                   help='comma list of categories holding the keypoints (default: any category with keypoints)')
    p.add_argument('--landmark_tol_px', type=float, default=10.0,
                   help='a keypoint belongs to the structure when it lies inside its outline or this close to it (px)')
    p.add_argument('--landmark_indices', default='',
                   help='1-based keypoint numbers to use (default all)')
    p.add_argument('--landmark_min_share', type=float, default=0.5,
                   help='use a landmark only when at least this share of the specimens has it (default 0.5)')
    return p


def from_args(args, specimens, mean_shape, allow_reflection=True):
    if getattr(args, 'landmarks', 'none') == 'none':
        return None
    lc = LandmarkControls.from_coco(args.landmark_json or args.json, specimens, mean_shape,
                                    categories=args.landmark_categories, tol_px=args.landmark_tol_px,
                                    indices=args.landmark_indices, min_share=args.landmark_min_share,
                                    allow_reflection=allow_reflection)
    if lc.n_used:
        i = lc.info
        logging.info(f"Landmarks: {i['n_used']} of {i['n_landmarks']} used as extra TPS control points "
                     f"({i['n_specimens_with_landmarks']}/{i['n_specimens']} specimens have landmarks on this structure); "
                     f"spread around the consensus position, % of outline radius: {i['spread_pct_of_radius']}")
        if i["dropped"]:
            logging.info(f"Landmarks present in too few specimens, not used: {i['dropped']}")
        if i["outline_landmark_disagree"]:
            logging.warning(f"{len(i['outline_landmark_disagree'])} specimen(s) whose landmarks do not fit their aligned "
                            f"outline (mirror image in a GPA without mirror handling, a mis-ordered outline, or wrong "
                            f"landmarks); warped from the outline alone: {', '.join(i['outline_landmark_disagree'])}")
        return lc
    logging.info(f"Landmarks: none used ({lc.info.get('reason', 'too few specimens')}); outline semilandmarks only")
    return None
