#!/usr/bin/env python3
"""
biorag_frame_orientation_calibrate_v1.py — is the half-turn detector right, and how right?
==========================================================================================

`biorag_frame_orientation_v1.py` decides which homology frames lie end for end from the
width profile along the long axis, and keeps a structure's decisions only where picture
content corroborates them. Six numbers govern that (`--min_confidence`, `--min_ncc_margin`,
`--min_corroboration`, `--min_corroborated_flips`, and the two reporting-only risk flags),
and not one of them was ever measured against anything. This script measures them.

The ground truth, and why it is not circular
--------------------------------------------
It comes from the photographs, not from the frames. A metaleg image carries the femur AND
the tibia; the end of the tibia that touches the femur is its proximal end. A rostrum image
carries both labial segments; the end of LAB2 that touches LAB1 is its base. A male
terminalia image carries the two halves of the aedeagus. None of that is a statistic over
the sample, none of it looks at a width profile or at a correlation with a mean frame, and
none of it has a threshold: it is an observation of the individual specimen.

Carrying it into the frame is exact rather than estimated. `biorag_homology_frame_v1.py`
run with `--gpa_dir` places the semilandmark step's own back-transformed contour in the
frame, point for point, so the contour in `back_transformed_coco.json` and the array in
`<specimen>_landmarks.npy` are the same landmarks in two coordinate systems and matching
them by index recovers the frame's rotation to machine precision (the residual is ~1e-15
on this data set, and a frame whose residual is not is dropped rather than guessed at).

Within a structure the majority defines "as is" — the same convention the detector uses —
and a frame that puts the joint at the other end is truly turned.

What is reported
----------------
  ground_truth.tsv          per frame: the cue used, where along the frame the anatomy
                            says the base lies, the margin, and truly_turned (0/1)
  existing_vs_truth.tsv     the table as it stands, scored frame by frame
  scored_2x2.tsv            per structure: truly turned x called turned, with sensitivity,
                            false-turn rate and the share left undecided
  threshold_sweep.tsv       the operating curve: every combination of --min_confidence,
                            --min_ncc_margin and --min_corroboration, scored on the frames
                            that have a ground truth
  cross_validation.tsv      thresholds chosen on four fifths of the SPECIMENS and scored on
                            the fifth, five times, globally and per structure
  mirror_calls.tsv          per image: the handedness vote, and per frame the verdict
  mirror_validation.tsv     the vote against the independent cell-centroid handedness
  read_time_transform.tsv   which of the six candidate read-time transforms brings each
                            frame's own outline onto the consensus, and by how much
  calibration_summary.json  the numbers above in one place

Nothing here writes to the frames or to the existing orientation table.

  python biorag_frame_orientation_calibrate_v1.py \\
      --frames_dir "$M/homology_frames" --gpa_dir "<semilandmarks dir>" \\
      --coco "<unified coco.json>" --taxon_profile <p.yaml> \\
      --matrix_dir "$M/compiled_key_tier" \\
      --orientation "$M/homology_frames/frame_orientation.tsv" \\
      --outline_registration "<outline_registration.tsv>" \\
      --neighbours tibia=femura femura=tibia LAB2=LAB1 LAB1=LAB2 \\
      --axis_pairs "whole_wing=cell-cu2,cell-a>cell-r2,cell-Rs" \\
      --exclude_specimens sp1_1 sp4_5 sp4_6 sp5_2 --exclude_structures_for_excluded wing \\
      --out_dir "<results dir>"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biorag_frame_orientation_v1 as fo                                   # noqa: E402
import biorag_feature_policy as pol                                        # noqa: E402
from biorag_specimen_id import specimen_of                                 # noqa: E402

VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# the ground truth
# ─────────────────────────────────────────────────────────────────────────────
def ground_truth(frames: Path, gpa_dir: str, polys: dict, pairs: dict, axes: dict,
                 sid_of, min_end_margin: float, min_axis_cosine: float,
                 drop: set, structs) -> pd.DataFrame:
    """One row per frame the photographs can orient, with `truly_turned` relative to the
    majority of that structure."""
    rows = []
    for st in structs:
        sdir = frames / st
        cons_full = np.load(sdir / "consensus_px.npy")
        cons = fo.ring(cons_full)
        u, _v = fo.principal_axis(cons)
        contours = fo.load_gpa_contours(gpa_dir, st)
        by_sid = {}
        for fn, g in contours.items():
            s = sid_of(fo.image_key(fn))
            if s:
                by_sid.setdefault(s, (fn, g))
        got_any = []
        for f in sorted(sdir.glob("*_landmarks.npy")):
            sid = f.name[: -len("_landmarks.npy")]
            if (st, sid) in drop or sid in drop:
                continue
            hit = by_sid.get(sid)
            if hit is None:
                continue
            fn, g = hit
            lmf = np.load(f)
            if len(fo._ring_np(g)) != len(fo._ring_np(lmf)):
                continue
            _R, res = fo.kabsch_rotation(fo._ring_np(g), fo._ring_np(lmf))
            if res > 1e-3:
                continue
            img = fo.image_key(fn)
            near = polys.get(img, {})
            for nb in pairs.get(st, []):
                if nb not in near:
                    continue
                t = fo.neighbour_end(g, lmf, fo.densify(near[nb]), cons, u)
                got_any.append(dict(structure=st, specimen_id=sid, image=img,
                                    cue=f"neighbour:{nb}", reading=t,
                                    margin=abs(t - 0.5) * 2.0, side=1 if t > 0.5 else -1,
                                    fit_residual=res))
            if st in axes:
                lo, hi = axes[st]
                a = [near[c].mean(0) for c in lo if c in near and c != st]
                b = [near[c].mean(0) for c in hi if c in near and c != st]
                if a and b:
                    c = fo.axis_direction_cos(g, lmf, np.mean(b, 0) - np.mean(a, 0), u)
                    got_any.append(dict(structure=st, specimen_id=sid, image=img,
                                        cue="axis", reading=c, margin=abs(c),
                                        side=1 if c > 0 else -1, fit_residual=res))
        rows.extend(got_any)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    need = np.where(df["cue"].str.startswith("neighbour"), min_end_margin, min_axis_cosine)
    df["usable"] = (df["margin"] >= need).astype(int)
    # one reading per frame: the clearest usable one
    best = (df[df["usable"] == 1].sort_values("margin", ascending=False)
            .drop_duplicates(["structure", "specimen_id"]).copy())
    out = []
    for st, g in best.groupby("structure"):
        maj = 1 if (g["side"] > 0).sum() >= (g["side"] < 0).sum() else -1
        g = g.copy()
        g["majority_side"] = maj
        g["truly_turned"] = (g["side"] != maj).astype(int)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else best


# ─────────────────────────────────────────────────────────────────────────────
# the detector, with its scores computed once and its thresholds applied many times
# ─────────────────────────────────────────────────────────────────────────────
def structure_cache(sdir: Path, cons: np.ndarray, bins: int, ncc_px: int, no_ncc: bool):
    """Everything the decision rule needs, computed once: the profile scores, and BOTH
    possible image-content score sets.

    `orient_by_image` anchors the sign of its leading singular vector on the frames the
    profile pass decided, so its scores depend on --min_confidence. They depend on it only
    through one global sign, however, so both outcomes are precomputed here and the sweep
    simply chooses between them — which makes a sweep over thousands of threshold
    combinations affordable without changing a single number it would otherwise produce."""
    u, v = fo.principal_axis(fo.ring(cons))
    prof = {}
    for f in sorted(sdir.glob("*_landmarks.npy")):
        sid = f.name[: -len("_landmarks.npy")]
        w = fo.width_profile(np.load(f), u, v, bins)
        if w is None:
            continue
        wu = fo.unit(w)
        if wu is not None:
            prof[sid] = wu
    if not prof:
        return None
    scores, _ref = fo.orient_by_profile(prof)
    cache = {"sids": sorted(prof), "profile": scores, "img": None}
    if no_ncc:
        return cache
    keep = [s for s in sorted(prof) if (sdir / f"{s}.png").exists()]
    if len(keep) < 4:
        return cache
    G = np.stack([fo.grey(sdir / f"{s}.png", ncc_px) for s in keep])
    A = (G - G[:, ::-1, ::-1]).reshape(len(G), -1) / 2.0
    A -= A.mean(1, keepdims=True)
    if not np.isfinite(A).all() or np.linalg.norm(A) <= 0:
        return cache
    v1 = np.linalg.svd(A, full_matrices=False)[2][0]
    proj = A @ v1
    both = {}
    for sign in (+1, -1):
        p = proj * sign
        m = np.where(p[:, None, None] >= 0, G, G[:, ::-1, ::-1]).mean(0)
        both[sign] = {s: (fo.ncc(G[i], m), fo.ncc(G[i, ::-1, ::-1], m))
                      for i, s in enumerate(keep)}
    cache["img"] = {"keep": keep, "proj": proj, "both": both}
    return cache


def decide(cache: dict, min_confidence: float, min_ncc_margin: float,
           min_corroboration: float, min_corroborated_flips: int, no_ncc: bool = False):
    """The decision rule of biorag_frame_orientation_v1.main, exactly, from the cache."""
    rec = {}
    dec, undec = {}, []
    for sid, (s_as, s_rot) in cache["profile"].items():
        conf = abs(s_as - s_rot) / 2.0
        if conf >= min_confidence:
            flip = bool(s_rot > s_as)
            dec[sid] = flip
            rec[sid] = [int(flip), "profile"]
        else:
            undec.append(sid)
            rec[sid] = [0, "undecided"]
    if no_ncc or cache["img"] is None:
        return rec
    keep, proj, both = cache["img"]["keep"], cache["img"]["proj"], cache["img"]["both"]
    at = np.array([1.0 if not dec.get(s, False) else -1.0 for s in keep if s in dec])
    ap = np.array([proj[i] for i, s in enumerate(keep) if s in dec])
    sign = +1
    if len(at) >= 4:
        if float(np.mean(np.sign(ap) == at)) < 0.5:
            sign = -1
    elif (proj < 0).sum() > (proj > 0).sum():
        sign = -1
    img = both[sign]
    d = {sid: (c_rot - c_as) for sid, (c_as, c_rot) in img.items()}
    for sid in undec:
        if abs(d.get(sid, 0.0)) >= min_ncc_margin:
            rec[sid] = [int(d[sid] > 0), "ncc"]
    for sid, flip in dec.items():
        if flip and d.get(sid, 0.0) <= -min_ncc_margin:
            rec[sid][:] = [0, "conflict"]
    want = [s for s, f in dec.items() if f and s in d]
    if len(want) >= min_corroborated_flips:
        corrob = float(np.mean([d[s] >= min_ncc_margin for s in want]))
        if corrob < min_corroboration:
            for sid, r in rec.items():
                if r[0]:
                    r[:] = [0, "uncorroborated"]
    return {k: (v[0], v[1]) for k, v in rec.items()}


def align_convention(called: dict, truth: pd.DataFrame):
    """A structure's "as is" is a majority, and the detector and the anatomy each take
    their own. When they disagree the whole structure is turned, which changes nothing
    about any heat map (a half turn applied to every frame alike), but it would invert the
    2 x 2 table and make a detector that is right look wrong. So the convention is aligned
    before scoring, and every alignment is reported."""
    ov = [(called[s][0], t) for s, t in zip(truth["specimen_id"], truth["truly_turned"])
          if s in called and called[s][1] in ("profile", "ncc", "neighbour")]
    if len(ov) < 4:
        return False
    agree = np.mean([c == t for c, t in ov])
    return bool(agree < 0.5)


def score(called: dict, truth: pd.DataFrame, swap: bool):
    tp = fp = tn = fn = undec = 0
    for s, t in zip(truth["specimen_id"], truth["truly_turned"]):
        r = called.get(s)
        if r is None:
            continue
        c = r[0]
        if swap and r[1] in ("profile", "ncc", "neighbour"):
            c = 1 - c
        if r[1] in ("undecided", "uncorroborated", "conflict"):
            undec += 1
        if t == 1 and c == 1:
            tp += 1
        elif t == 1 and c == 0:
            fn += 1
        elif t == 0 and c == 1:
            fp += 1
        else:
            tn += 1
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, undecided=undec, n=tp + fp + tn + fn)


def rates(s: dict) -> dict:
    pos, neg = s["tp"] + s["fn"], s["fp"] + s["tn"]
    sens = s["tp"] / pos if pos else float("nan")
    fpr = s["fp"] / neg if neg else float("nan")
    acc = (s["tp"] + s["tn"]) / s["n"] if s["n"] else float("nan")
    return dict(sensitivity=sens, false_turn_rate=fpr, accuracy=acc,
                youden=(sens - fpr) if pos and neg else float("nan"),
                undecided_share=(s["undecided"] / s["n"] if s["n"] else float("nan")))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Calibrate the half-turn detector against the "
                                             "anatomy in the photographs")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--gpa_dir", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--orientation", required=True, help="the table as it stands, to be scored")
    ap.add_argument("--outline_registration", default=None)
    ap.add_argument("--mirror_reliable", nargs="*", default=None,
                    help="structures allowed to vote on an image's handedness; chosen from "
                         "mirror_within_image_consistency.tsv. Default: all unambiguous")
    ap.add_argument("--handedness", default=None,
                    help="outline_wing_handedness.tsv — the independent check on the mirror vote")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--neighbours", nargs="*", default=None)
    ap.add_argument("--axis_pairs", nargs="*", default=None)
    ap.add_argument("--exclude_specimens", nargs="*", default=[],
                    help="specimen ids whose polygons were in a stale orientation when the "
                         "frames were built; excluded from the ground truth only")
    ap.add_argument("--exclude_structures", nargs="*", default=[],
                    help="structures on which --exclude_specimens applies (default: all)")
    ap.add_argument("--min_end_margin", type=float, default=0.30)
    ap.add_argument("--min_axis_cosine", type=float, default=0.50)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--ncc_px", type=int, default=192)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = Path(a.frames_dir)
    profile = pol.load_taxon_profile(a.taxon_profile)
    codes = sorted(set(pd.read_csv(Path(a.matrix_dir) / "specimen_matrix_long.csv")
                       ["species"].astype(str)))
    sid_of = lambda fn: specimen_of(fn, profile, codes)[1]                  # noqa: E731
    pairs = fo.parse_pairs(a.neighbours) or fo.parse_pairs(
        [f"{k}={','.join(v)}" for k, v in (profile.get("orientation_neighbours") or {}).items()])
    axes = fo.parse_axes(a.axis_pairs) or fo.parse_axes(
        [f"{k}={','.join(v['from'])}>{','.join(v['to'])}"
         for k, v in (profile.get("orientation_axes") or {}).items()])
    polys = fo.load_coco_polygons(a.coco)
    structs = sorted(d.name for d in frames.iterdir()
                     if d.is_dir() and (d / "consensus_px.npy").exists())

    drop = set()
    for s in a.exclude_specimens:
        if a.exclude_structures:
            drop |= {(st, s) for st in a.exclude_structures}
        else:
            drop.add(s)

    # ── 1. ground truth ------------------------------------------------------
    gt = ground_truth(frames, a.gpa_dir, polys, pairs, axes, sid_of,
                      a.min_end_margin, a.min_axis_cosine, drop, structs)
    gt.to_csv(out / "ground_truth.tsv", sep="\t", index=False)
    print(f"ground truth: {len(gt)} frames of {sum(len(list((frames/s).glob('*_landmarks.npy'))) for s in structs)}")
    cov = gt.groupby(["structure", "cue"]).agg(
        n=("specimen_id", "size"), turned=("truly_turned", "sum"),
        margin_med=("margin", "median")).reset_index()
    print(cov.to_string(index=False))
    cov.to_csv(out / "ground_truth_coverage.tsv", sep="\t", index=False)

    # ── 2. score the table as it stands --------------------------------------
    tab = pd.read_csv(a.orientation, sep="\t")
    existing = {(r.structure, r.specimen_id): (int(r.flipped), str(r.method))
                for r in tab.itertuples()}
    rows2, det = [], []
    for st, g in gt.groupby("structure"):
        called = {s: existing[(st, s)] for s in g["specimen_id"] if (st, s) in existing}
        sw = align_convention(called, g)
        s = score(called, g, sw)
        r = rates(s)
        rows2.append(dict(structure=st, convention_turned_round=int(sw), **s, **r,
                          called_turned_all=int(tab[tab["structure"] == st]["flipped"].sum()),
                          frames_all=int((tab["structure"] == st).sum())))
        for sid, t in zip(g["specimen_id"], g["truly_turned"]):
            if (st, sid) in existing:
                e = existing[(st, sid)]
                c = e[0]
                if sw and e[1] in ("profile", "ncc", "neighbour"):
                    c = 1 - c
                det.append(dict(structure=st, specimen_id=sid, truly_turned=int(t),
                                called_turned=int(c), method=e[1]))
    sc = pd.DataFrame(rows2)
    sc.to_csv(out / "scored_2x2.tsv", sep="\t", index=False)
    pd.DataFrame(det).to_csv(out / "existing_vs_truth.tsv", sep="\t", index=False)
    print("\n=== the existing table against the anatomy ===")
    print(sc[["structure", "n", "tp", "fp", "fn", "tn", "sensitivity", "false_turn_rate",
              "undecided_share", "convention_turned_round"]].to_string(index=False))

    # ── 3. the operating curve -----------------------------------------------
    print("\ncaching profile and picture scores …")
    caches = {}
    for st in structs:
        if st not in set(gt["structure"]):
            continue
        c = structure_cache(frames / st, np.load(frames / st / "consensus_px.npy"),
                            a.bins, a.ncc_px, no_ncc=False)
        if c:
            caches[st] = c
    conf_grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50]
    ncc_grid = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30]
    corr_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    sweep = []
    decisions = {}
    for mc in conf_grid:
        for mn in ncc_grid:
            for cr in corr_grid:
                key = (mc, mn, cr)
                tot = dict(tp=0, fp=0, tn=0, fn=0, undecided=0, n=0)
                per = {}
                for st, cache in caches.items():
                    called = decide(cache, mc, mn, cr, 3)
                    decisions[(key, st)] = called
                    g = gt[gt["structure"] == st]
                    sw = align_convention(called, g)
                    s = score(called, g, sw)
                    per[st] = s
                    for k in tot:
                        tot[k] += s[k]
                sweep.append(dict(min_confidence=mc, min_ncc_margin=mn, min_corroboration=cr,
                                  **tot, **rates(tot),
                                  **{f"{st}_youden": rates(per[st])["youden"] for st in per}))
    sw_df = pd.DataFrame(sweep)
    sw_df.to_csv(out / "threshold_sweep.tsv", sep="\t", index=False)
    best = sw_df.sort_values("youden", ascending=False).head(8)
    print("\n=== operating curve, best eight on the full sample (NOT a hold-out) ===")
    print(best[["min_confidence", "min_ncc_margin", "min_corroboration", "tp", "fp", "fn",
                "tn", "sensitivity", "false_turn_rate", "youden"]].to_string(index=False))
    cur = sw_df[(sw_df["min_confidence"] == 0.05) & (sw_df["min_ncc_margin"] == 0.02)
                & (sw_df["min_corroboration"] == 0.6)]
    if len(cur):
        print("current defaults:")
        print(cur[["tp", "fp", "fn", "tn", "sensitivity", "false_turn_rate",
                   "youden"]].to_string(index=False))

    # ── 4. cross-validation over specimens -----------------------------------
    # The thresholds are the only thing the labels choose, so they are the only thing that
    # has to be held out. The profile singular vector and the mean frame are unsupervised
    # statistics of the sample and see no ground truth at any point.
    spec = sorted(set(gt["specimen_id"]))
    rng = np.random.default_rng(a.seed)
    fold_of = {s: int(i) for s, i in zip(spec, rng.integers(0, a.folds, len(spec)))}
    gt = gt.copy()
    gt["fold"] = gt["specimen_id"].map(fold_of)
    cv_rows = []
    for mode in ("global", "per_structure"):
        for k in range(a.folds):
            tr = gt[gt["fold"] != k]
            te = gt[gt["fold"] == k]
            if te.empty:
                continue
            if mode == "global":
                bestkey, bestv = None, -np.inf
                for key in {kk for kk, _st in decisions}:
                    tot = dict(tp=0, fp=0, tn=0, fn=0, undecided=0, n=0)
                    for st in caches:
                        called = decisions[(key, st)]
                        g = tr[tr["structure"] == st]
                        if g.empty:
                            continue
                        s = score(called, g, align_convention(called, gt[gt["structure"] == st]))
                        for kk in tot:
                            tot[kk] += s[kk]
                    y = rates(tot)["youden"]
                    if y == y and y > bestv:
                        bestkey, bestv = key, y
                tot = dict(tp=0, fp=0, tn=0, fn=0, undecided=0, n=0)
                for st in caches:
                    called = decisions[(bestkey, st)]
                    g = te[te["structure"] == st]
                    if g.empty:
                        continue
                    s = score(called, g, align_convention(called, gt[gt["structure"] == st]))
                    for kk in tot:
                        tot[kk] += s[kk]
                cv_rows.append(dict(mode=mode, fold=k, structure="ALL",
                                    chosen=str(bestkey), **tot, **rates(tot)))
            else:
                for st in caches:
                    gtr, gte = tr[tr["structure"] == st], te[te["structure"] == st]
                    if gtr.empty or gte.empty:
                        continue
                    bestkey, bestv = None, -np.inf
                    for key in {kk for kk, s2 in decisions if s2 == st}:
                        called = decisions[(key, st)]
                        y = rates(score(called, gtr,
                                        align_convention(called, gt[gt["structure"] == st])))["youden"]
                        if y != y:
                            y = rates(score(called, gtr, False))["accuracy"]
                        if y == y and y > bestv:
                            bestkey, bestv = key, y
                    called = decisions[(bestkey, st)]
                    s = score(called, gte, align_convention(called, gt[gt["structure"] == st]))
                    cv_rows.append(dict(mode=mode, fold=k, structure=st,
                                        chosen=str(bestkey), **s, **rates(s)))
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out / "cross_validation.tsv", sep="\t", index=False)
    print("\n=== cross-validated (thresholds chosen on the other folds) ===")
    for mode, g in cv.groupby("mode"):
        tot = {k: int(g[k].sum()) for k in ("tp", "fp", "tn", "fn", "undecided", "n")}
        r = rates(tot)
        print(f"{mode:<14} tp={tot['tp']:>3} fp={tot['fp']:>3} fn={tot['fn']:>3} tn={tot['tn']:>4}"
              f"  sens={r['sensitivity']:.3f}  false-turn={r['false_turn_rate']:.3f}"
              f"  youden={r['youden']:.3f}")

    # the anatomy itself, as a rule: coverage and (by construction) perfect agreement
    n_frames = sum(len(list((frames / s).glob("*_landmarks.npy"))) for s in structs)
    adj_cov = len(gt) / n_frames

    summary = dict(version=VERSION, frames_total=int(n_frames),
                   ground_truth_frames=int(len(gt)),
                   ground_truth_coverage=round(adj_cov, 4),
                   truly_turned=int(gt["truly_turned"].sum()),
                   existing_table=dict(
                       tp=int(sc["tp"].sum()), fp=int(sc["fp"].sum()),
                       fn=int(sc["fn"].sum()), tn=int(sc["tn"].sum()),
                       **{k: (None if v != v else round(float(v), 4)) for k, v in
                          rates({k: int(sc[k].sum()) for k in
                                 ("tp", "fp", "tn", "fn", "undecided", "n")}).items()}),
                   cross_validated={
                       mode: {k: (None if v != v else round(float(v), 4)) for k, v in
                              rates({k: int(g[k].sum()) for k in
                                     ("tp", "fp", "tn", "fn", "undecided", "n")}).items()}
                       for mode, g in cv.groupby("mode")})

    # ── 5. mirrors ------------------------------------------------------------
    if a.outline_registration:
        reg = fo.load_outline_registration(a.outline_registration)
        image_of = {}
        for st in structs:
            for fn in fo.load_gpa_contours(a.gpa_dir, st):
                s = sid_of(fo.image_key(fn))
                if s:
                    image_of[(st, s)] = fo.image_key(fn)
        # only frames that exist are allowed to vote or to be voted on
        have = {(st, f.name[: -len("_landmarks.npy")])
                for st in structs for f in (frames / st).glob("*_landmarks.npy")}
        reg = {k: v for k, v in reg.items() if k in have}
        verdict, per_image = fo.mirror_vote(reg, image_of, a.mirror_reliable, 1)
        mrows = [dict(image=img, mirrored=int(m), votes_mirrored=y, votes=n)
                 for img, (m, y, n) in sorted(per_image.items())]
        pd.DataFrame(mrows).to_csv(out / "mirror_calls.tsv", sep="\t", index=False)
        frows = [dict(structure=st, specimen_id=sid, image=image_of.get((st, sid), ""),
                      mirrored=v) for (st, sid), v in sorted(verdict.items())]
        pd.DataFrame(frows).to_csv(out / "mirror_frames.tsv", sep="\t", index=False)
        print(f"\nmirror vote: {len(per_image)} images decided, "
              f"{sum(1 for v in per_image.values() if v[0])} mirrored; "
              f"{len(verdict)} frames, {sum(verdict.values())} mirrored")
        summary["mirror"] = dict(images=len(per_image),
                                 images_mirrored=int(sum(1 for v in per_image.values() if v[0])),
                                 frames=len(verdict), frames_mirrored=int(sum(verdict.values())))

        # ── 5b. does mirroring happen off the wing at all? ------------------
        # Every structure cut out of ONE photograph has ONE handedness, so two structures
        # from the same image must agree. Where the registration's calls for a pair of
        # structures agree far above half the time, the mirroring it is reading is a real
        # property of the slide; where they agree at chance, it is reading noise off a
        # near-symmetric outline and nothing should be corrected on that evidence. This
        # needs no external ground truth, which is the point: the forewing has one and
        # nothing else does.
        pairwise = {}
        by_img = {}
        for (st, sid), (mir, amb, gap) in reg.items():
            img = image_of.get((st, sid))
            if img is None or amb:
                continue
            by_img.setdefault(img, {})[st] = (1 if mir else 0, gap)
        for img, d in by_img.items():
            ks = sorted(d)
            for i in range(len(ks)):
                for jx in range(i + 1, len(ks)):
                    k = (ks[i], ks[jx])
                    pairwise.setdefault(k, []).append(int(d[ks[i]][0] == d[ks[jx]][0]))
        # An agreement has to be read against the rate at which the two calls agree by
        # accident. Where a structure is almost never called mirrored (the aedeagus: 0 of
        # 50), two such structures agree perfectly without either of them having read
        # anything, so the raw agreement is uninformative and only the excess over chance
        # says whether mirroring was detected.
        rate = {}
        for (st, sid), (mir, amb, _g) in reg.items():
            if not amb:
                rate.setdefault(st, []).append(1 if mir else 0)
        rate = {k: float(np.mean(v)) for k, v in rate.items()}
        prow = []
        for k, v in sorted(pairwise.items()):
            if len(v) < 8:
                continue
            obs = float(np.mean(v))
            pa, pb = rate.get(k[0], 0.5), rate.get(k[1], 0.5)
            exp = pa * pb + (1 - pa) * (1 - pb)
            cons_ = max(obs, 1 - obs)
            prow.append(dict(structure_a=k[0], structure_b=k[1], images=len(v),
                             agreement=round(obs, 3), consistency=round(cons_, 3),
                             p_mirror_a=round(pa, 3), p_mirror_b=round(pb, 3),
                             expected_by_chance=round(exp, 3),
                             kappa=round((cons_ - exp) / (1 - exp) if exp < 1 else float("nan"), 3)))
        pdf = pd.DataFrame(prow)
        pdf.to_csv(out / "mirror_within_image_consistency.tsv", sep="\t", index=False)
        if len(pdf):
            print("\n=== do two structures from the same photograph agree on handedness? ===")
            print(pdf.sort_values("kappa", ascending=False).to_string(index=False))
            summary["mirror"]["within_image_consistency"] = {
                f"{r.structure_a}|{r.structure_b}": [int(r.images), float(r.consistency),
                                                     float(r.kappa)]
                for r in pdf.itertuples()}

        if a.handedness and Path(a.handedness).exists():
            h = pd.read_csv(a.handedness, sep="\t")
            # join on the IMAGE. The handedness is a property of one photograph, and the
            # forewing slide of a specimen says nothing about how its metaleg was mounted;
            # joining on specimen_id would score the wing's answer against a leg.
            def _norm(x):
                s = str(x).strip().lower()
                for e in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
                    if s.endswith(e):
                        s = s[: -len(e)]
                return s.replace(" ", "_")
            hand = {_norm(i): hh for i, hh in zip(h["image"], h["handedness"])}
            vr = []
            for (st, sid), m in verdict.items():
                key = _norm(image_of.get((st, sid), ""))
                if key in hand:
                    vr.append(dict(structure=st, specimen_id=sid, image=key, mirrored=m,
                                   handedness=hand[key]))
            vdf = pd.DataFrame(vr)
            vdf.to_csv(out / "mirror_validation.tsv", sep="\t", index=False)
            if len(vdf):
                ag = []
                for st, g in vdf.groupby("structure"):
                    x = (g["handedness"] == "A").astype(int).to_numpy()
                    y = g["mirrored"].to_numpy()
                    acc = max((x == y).mean(), (x != y).mean())
                    ag.append(dict(structure=st, n=len(g), agreement=round(float(acc), 3)))
                adf = pd.DataFrame(ag)
                adf.to_csv(out / "mirror_handedness_agreement.tsv", sep="\t", index=False)
                print(adf.to_string(index=False))
                summary["mirror"]["handedness_agreement"] = {
                    r.structure: r.agreement for r in adf.itertuples()}

        # ── 6. which read-time transform ------------------------------------
        canvas = 768
        sjf = frames / "homology_frames_summary.json"
        if sjf.exists():
            canvas = int(json.loads(sjf.read_text()).get("canvas", canvas))
        flip_now = {(r.structure, r.specimen_id): int(r.flipped) for r in tab.itertuples()}
        trows = []
        for st in structs:
            cons = np.load(frames / st / "consensus_px.npy")
            axis = fo.consensus_axis_degrees(cons)
            for f in sorted((frames / st).glob("*_landmarks.npy")):
                sid = f.name[: -len("_landmarks.npy")]
                if (st, sid) not in verdict:
                    continue
                lm = np.load(f)
                name, d0, gap = fo.best_read_time_transform(lm, cons, canvas, axis)
                per = {}
                for cand in fo.TRANSFORMS:
                    q = fo.apply_frame_transform(fo._ring_np(lm) / canvas, cand, axis)
                    c = fo.ring(cons) / canvas
                    dd = np.sqrt(((q[:, None, :] - c[None, :, :]) ** 2).sum(-1))
                    per[cand] = 0.5 * (dd.min(1).mean() + dd.min(0).mean())
                trows.append(dict(structure=st, specimen_id=sid, axis_deg=round(axis, 2),
                                  mirrored=verdict[(st, sid)],
                                  flipped_now=flip_now.get((st, sid), 0),
                                  best=name, best_distance=round(d0, 5),
                                  margin_over_runner_up=round(gap, 5),
                                  **{f"d_{k}": round(float(v), 5) for k, v in per.items()}))
        tdf = pd.DataFrame(trows)
        tdf.to_csv(out / "read_time_transform_outline.tsv", sep="\t", index=False)
        if len(tdf):
            print("\n=== outline test (a negative control, see below) ===")
            print(pd.crosstab(tdf["mirrored"], tdf["best"]).to_string())
            summary["read_time_transform_outline"] = {
                str(k): {kk: int(vv) for kk, vv in v.items()}
                for k, v in pd.crosstab(tdf["mirrored"], tdf["best"]).to_dict("index").items()}

        # ── 6b. the test that can actually answer -----------------------------
        # The outline test above CANNOT choose a reflection and it is worth saying why:
        # biorag_homology_frame_v1 placed the frame by minimising exactly that distance
        # over rotations, so `identity` wins by construction whether or not the specimen
        # is a mirror image. Deciding needs points whose ARRANGEMENT is chiral, and the
        # photograph has them: the centroids of the other annotated structures on the same
        # image, carried into the frame by the recovered similarity. Three or more such
        # points have a handedness that no rotation can change and a reflection must.
        arows, srows = [], []
        for st in structs:
            cons = np.load(frames / st / "consensus_px.npy")
            axis = fo.consensus_axis_degrees(cons)
            u, _v2 = fo.principal_axis(fo.ring(cons))
            contours = fo.load_gpa_contours(a.gpa_dir, st)
            by_sid = {}
            for fn, g in contours.items():
                s = sid_of(fo.image_key(fn))
                if s:
                    by_sid.setdefault(s, (fn, g))
            ref = {}
            for f in sorted((frames / st).glob("*_landmarks.npy")):
                sid = f.name[: -len("_landmarks.npy")]
                hit = by_sid.get(sid)
                if hit is None or (st, sid) not in verdict:
                    continue
                fn, g = hit
                lmf = np.load(f)
                if len(fo._ring_np(g)) != len(fo._ring_np(lmf)):
                    continue
                R, sc_, t_, res = fo.recover_similarity(g, lmf)
                if res > 1e-3:
                    continue
                near = polys.get(fo.image_key(fn), {})
                # three points per neighbour — its centroid and the two ends of its own
                # long axis — so that even a photograph with a SINGLE neighbour (a rostrum
                # has only LAB1 and LAB2, a metaleg only femur and tibia) still gives a
                # configuration with a handedness. Two points and a centroid cannot be
                # mapped onto each other by a rotation if they are a reflection apart.
                pts = {}
                for k, v in near.items():
                    if k == st:
                        continue
                    q = fo._ring_np(v)
                    uu, ww = fo.principal_axis(q)
                    s_ = (q - q.mean(0)) @ uu
                    w_ = (q - q.mean(0)) @ ww
                    # the two ends of the neighbour's long axis AND the two of its short
                    # axis: three points strung out along one line have no handedness, and
                    # a rostrum's two segments lie end to end, so the long axis alone
                    # leaves the reflection undecidable exactly where it matters
                    for tag, pp in (("c", q.mean(0)), ("lo", q[int(np.argmin(s_))]),
                                    ("hi", q[int(np.argmax(s_))]),
                                    ("left", q[int(np.argmin(w_))]),
                                    ("right", q[int(np.argmax(w_))])):
                        pts[f"{k}:{tag}"] = ((pp @ R) * sc_ + t_) / canvas
                if len(pts) >= 3:
                    ref[sid] = pts
            if len(ref) < 8:
                continue
            names = sorted({k for p in ref.values() for k in p})
            base = [s for s in ref if not verdict[(st, s)] and not flip_now.get((st, s), 0)]
            if len(base) < 4:
                base = [s for s in ref if not verdict[(st, s)]]
            cons_pts = {}
            for nm in names:
                v = [ref[s][nm] for s in base if nm in ref[s]]
                if len(v) >= 3:
                    cons_pts[nm] = np.mean(v, 0)
            if len(cons_pts) < 2:
                continue
            for sid, pts in ref.items():
                common = [nm for nm in cons_pts if nm in pts]
                if len(common) < 3:
                    continue
                P = np.array([pts[nm] for nm in common])
                Q = np.array([cons_pts[nm] for nm in common])
                d = {}
                for cand in fo.TRANSFORMS:
                    d[cand] = float(np.mean(np.linalg.norm(
                        fo.apply_frame_transform(P, cand, axis) - Q, axis=1)))
                order = sorted(d, key=d.get)
                # how chiral the reference configuration is at all: the distance the best
                # reflection is from the best rotation. Near zero, the points are strung
                # out along a line and no reflection can be told from a rotation.
                rot = min(d["identity"], d["halfturn"])
                ref_ = min(d["mirror"], d["mirror_halfturn"], d["flipx"], d["flipy"])
                chirality = abs(rot - ref_)
                arows.append(dict(structure=st, specimen_id=sid, axis_deg=round(axis, 2),
                                  mirrored=verdict[(st, sid)],
                                  flipped_now=flip_now.get((st, sid), 0),
                                  n_reference_points=len(common), best=order[0],
                                  best_distance=round(d[order[0]], 5),
                                  margin_over_runner_up=round(d[order[1]] - d[order[0]], 5),
                                  chirality=round(chirality, 5),
                                  **{f"d_{k}": round(v, 5) for k, v in d.items()}))
                # the side test the reader can check: which side of the long axis does a
                # named reference structure sit on, before and after the correction?
                for nm in ("pterostigma:c", "cell-cu1:c", "femura:c", "tibia:c",
                           "LAB1:c", "LAB2:c", "male_proctiger:c"):
                    if nm not in pts:
                        continue
                    b4 = float((pts[nm] - 0.5) @ np.array([-u[1], u[0]]))
                    af = float((fo.apply_frame_transform(pts[nm], order[0], axis).ravel() - 0.5)
                               @ np.array([-u[1], u[0]]))
                    srows.append(dict(structure=st, specimen_id=sid, reference=nm,
                                      mirrored=verdict[(st, sid)],
                                      side_before=int(np.sign(b4)), side_after=int(np.sign(af))))
        adf = pd.DataFrame(arows)
        adf.to_csv(out / "read_time_transform.tsv", sep="\t", index=False)
        sdf = pd.DataFrame(srows)
        sdf.to_csv(out / "read_time_side_test.tsv", sep="\t", index=False)
        if len(adf):
            print("\n=== which transform maps the neighbouring anatomy onto the consensus ===")
            ct = pd.crosstab(adf["mirrored"], adf["best"])
            print(ct.to_string())
            print(f"margin over the runner-up: median {adf['margin_over_runner_up'].median():.4f}")
            summary["read_time_transform"] = {
                str(k): {kk: int(vv) for kk, vv in v.items()}
                for k, v in ct.to_dict("index").items()}
        if len(sdf):
            agr = []
            for (st, nm), g in sdf.groupby(["structure", "reference"]):
                if len(g) < 8:
                    continue
                mb = g["side_before"].mode().iat[0]
                ma = g["side_after"].mode().iat[0]
                agr.append(dict(structure=st, reference=nm, n=len(g),
                                same_side_before=round(float((g["side_before"] == mb).mean()), 3),
                                same_side_after=round(float((g["side_after"] == ma).mean()), 3)))
            sag = pd.DataFrame(agr)
            sag.to_csv(out / "read_time_side_agreement.tsv", sep="\t", index=False)
            if len(sag):
                print("\n=== does the correction put a named neighbour on the majority side? ===")
                print(sag.to_string(index=False))
                summary["side_test"] = {f"{r.structure}|{r.reference}":
                                        [int(r.n), float(r.same_side_before), float(r.same_side_after)]
                                        for r in sag.itertuples()}

    (out / "calibration_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
