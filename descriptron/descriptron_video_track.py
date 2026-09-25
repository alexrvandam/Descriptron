#!/usr/bin/env python3
"""
descriptron_video_track.py — follow individual specimens and their pose through a video
========================================================================================

An addition to the GUI's "Load Video for SAM2" tool, for behaviour and movement studies in the style of
DeepLabCut / SLEAP, working from a few annotated frames:

  frames      extract frames from a video (every Nth), named 00000.jpg, 00001.jpg, ... (what SAM2 reads)
  track       follow each individual through the frames with SAM2's video mode, starting from its mask or
              box on one or more annotated frames; identities are kept (track_id), giving a mask and a box
              per individual per frame
  pose        carry the pose keypoints (e.g. a pose skeleton placed in the GUI on some frames) through the
              video inside each individual's mask: pyramidal Lucas-Kanade tracking, forwards and backwards
              from every labelled frame, with a forward-backward round-trip check giving each point a
              confidence (like DeepLabCut's likelihood); points leaving their individual's mask are down-weighted
  export-dlc  write the poses as a DeepLabCut CSV (scorer / individuals / bodyparts / x, y, likelihood)
  overlay     draw masks, boxes, keypoints and skeleton sticks on the frames (PNG frames or an .mp4)

Annotated frames are ordinary Descriptron COCO: images named like the frames, masks (or boxes) with
"track_id" for the individuals, keypoints with the same "track_id" for the poses (a keypoint set inside an
individual's mask is assigned to it if track_id is missing).

    python descriptron_video_track.py frames clip.mp4 --out-dir clip_frames --every 2
    python descriptron_video_track.py track clip_frames --init first_frame.json --out tracks.json \\
        --sam2-checkpoint ../checkpoints/sam2_hiera_large.pt --sam2-config sam2_hiera_l.yaml
    python descriptron_video_track.py pose clip_frames --tracks tracks.json --labels poses.json --out pose.json
    python descriptron_video_track.py export-dlc pose.json --out CollectedData_descriptron.csv
    python descriptron_video_track.py overlay clip_frames --tracks tracks.json --pose pose.json --out overlay.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PALETTE = [(214, 120, 42), (59, 116, 232), (121, 169, 25), (123, 74, 237), (207, 94, 148), (180, 164, 19),
           (0, 180, 240), (80, 80, 200), (60, 160, 90), (200, 100, 60)]


def frame_list(folder) -> List[Path]:
    fs = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not fs:
        sys.exit(f"no frames in {folder}")
    return fs


def mask_from_seg(seg, h, w) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    if isinstance(seg, dict):
        from pycocotools import mask as mu
        rle = seg if isinstance(seg.get("counts"), (str, bytes)) else mu.frPyObjects(seg, h, w)
        return mu.decode(rle).astype(bool)
    if seg and not isinstance(seg[0], list):
        seg = [seg]
    for p in seg or []:
        if len(p) >= 6:
            cv2.fillPoly(m, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], 1)
    return m.astype(bool)


def polygon_of(mask: np.ndarray):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return [], [0, 0, 0, 0], 0.0
    c = max(cs, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return [c.reshape(-1).astype(float).tolist()], [float(x), float(y), float(w), float(h)], float(mask.sum())


# ------------------------------------------------------------------ frames
def cmd_frames(a):
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.video}")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    i = k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % a.every == 0 and (a.max_frames is None or k < a.max_frames):
            if a.max_side and max(fr.shape[:2]) > a.max_side:
                s = a.max_side / max(fr.shape[:2]); fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"{k:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95]); k += 1
        i += 1
    json.dump({"video": str(a.video), "fps": fps, "every": a.every, "frames": k}, open(out / "frames_info.json", "w"), indent=1)
    print(f"wrote {k} frames to {out} (every {a.every} of {i}; source {fps:.2f} fps)")


# ------------------------------------------------------------------ track
def _init_objects(coco: dict, frames: List[Path]):
    """-> {frame_index: [(track_id, mask or None, box or None)]} from the annotated frames."""
    by_name = {p.name: i for i, p in enumerate(frames)}
    by_stem = {p.stem: i for i, p in enumerate(frames)}
    ims = {im["id"]: im for im in coco["images"]}
    out, auto = defaultdict(list), defaultdict(int)
    for a in coco["annotations"]:
        if "keypoints" in a and not a.get("segmentation") and not a.get("bbox"):
            continue
        if "keypoints" in a and not a.get("segmentation"):
            continue                                        # keypoint sets are poses, not individuals
        im = ims.get(a["image_id"])
        if not im:
            continue
        fi = by_name.get(Path(im["file_name"]).name, by_stem.get(Path(im["file_name"]).stem))
        if fi is None:
            continue
        tid = a.get("track_id")
        if tid is None:
            auto[fi] += 1; tid = auto[fi]
        h, w = im.get("height"), im.get("width")
        m = mask_from_seg(a["segmentation"], h, w) if a.get("segmentation") and h and w else None
        box = a.get("bbox")
        out[fi].append((int(tid), m, box))
    if not out:
        sys.exit("no individuals (masks or boxes) on frames that exist in the frame folder")
    return out


def cmd_track(a):
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    frames = frame_list(a.frames)
    init = _init_objects(json.load(open(a.init)), frames)
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    pred = build_sam2_video_predictor(a.sam2_config, a.sam2_checkpoint, device=dev)
    h, w = cv2.imread(str(frames[0])).shape[:2]
    results = defaultdict(dict)                               # frame -> track -> mask
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        state = pred.init_state(video_path=str(a.frames), offload_video_to_cpu=True, offload_state_to_cpu=a.offload)
        for fi, objs in sorted(init.items()):
            for tid, m, box in objs:
                if m is not None:
                    pred.add_new_mask(state, frame_idx=fi, obj_id=tid, mask=m)
                elif box:
                    x, y, bw, bh = box
                    pred.add_new_points_or_box(state, frame_idx=fi, obj_id=tid, box=np.array([x, y, x + bw, y + bh], np.float32))
        start = min(init)
        for reverse in (False, True):
            if reverse and start == 0:
                continue
            for fi, ids, logits in pred.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
                for k, tid in enumerate(ids):
                    results[fi][int(tid)] = (logits[k, 0] > 0).cpu().numpy()
    images, anns = [], []
    for fi, p in enumerate(frames):
        images.append({"id": fi + 1, "file_name": p.name, "width": w, "height": h, "frame_index": fi})
        for tid, m in sorted(results.get(fi, {}).items()):
            seg, bbox, area = polygon_of(m)
            if area < a.min_area:
                continue
            anns.append({"id": len(anns) + 1, "image_id": fi + 1, "category_id": 1, "track_id": tid,
                         "segmentation": seg, "bbox": bbox, "area": area, "iscrowd": 0})
    json.dump({"images": images, "annotations": anns, "categories": [{"id": 1, "name": a.category}],
               "info": {"descriptron_video_tracks": {"frames_dir": str(a.frames), "init": str(a.init)}}},
              open(a.out, "w"))
    n_tracks = len({x["track_id"] for x in anns})
    per = defaultdict(int)
    for x in anns:
        per[x["track_id"]] += 1
    print(f"{n_tracks} individuals tracked over {len(frames)} frames -> {a.out}  "
          f"(frames present per individual: {dict(per)})")


# ------------------------------------------------------------------ pose
def _load_labels(coco: dict, frames: List[Path], track_masks):
    """-> {(frame, track): np.array (K,2) with nan for missing}, keypoint names, skeleton."""
    by_name = {p.name: i for i, p in enumerate(frames)}
    by_stem = {p.stem: i for i, p in enumerate(frames)}
    ims = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c for c in coco["categories"]}
    labels, names, skel = {}, None, []
    for a in coco["annotations"]:
        if "keypoints" not in a:
            continue
        im = ims.get(a["image_id"])
        fi = by_name.get(Path(im["file_name"]).name, by_stem.get(Path(im["file_name"]).stem)) if im else None
        if fi is None:
            continue
        kp = np.array(a["keypoints"], float).reshape(-1, 3)
        order = a.get("point_order") or list(range(1, len(kp) + 1))
        K = max(order)
        pts = np.full((K, 2), np.nan)
        for (x, y, v), o in zip(kp, order):
            if v > 0:
                pts[int(o) - 1] = (x, y)
        tid = a.get("track_id")
        if tid is None:                                          # the individual whose mask holds the points
            best, votes = None, 0
            for t, m in track_masks.get(fi, {}).items():
                inside = sum(1 for x, y in pts if not np.isnan(x) and 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1]
                             and m[int(y), int(x)])
                if inside > votes:
                    best, votes = t, inside
            tid = best
        if tid is None:
            continue
        labels[(fi, int(tid))] = pts
        c = cats.get(a["category_id"], {})
        names = names or c.get("keypoints"); skel = skel or c.get("skeleton") or []
    return labels, names, skel


def _lk(prev_gray, gray, pts):
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p0 = pts.astype(np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk)
    pb, stb, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk)
    fb = np.linalg.norm(p0 - pb, axis=2).ravel()
    ok = (st.ravel() == 1) & (stb.ravel() == 1)
    return p1.reshape(-1, 2), np.where(ok, fb, np.inf)


def cmd_pose(a):
    frames = frame_list(a.frames)
    tracks = json.load(open(a.tracks))
    h, w = tracks["images"][0]["height"], tracks["images"][0]["width"]
    tmask = defaultdict(dict)
    for x in tracks["annotations"]:
        tmask[x["image_id"] - 1][x["track_id"]] = mask_from_seg(x["segmentation"], h, w)
    labels, names, skel = _load_labels(json.load(open(a.labels)), frames, tmask)
    if not labels:
        sys.exit("no labelled pose frames matched the frames / individuals")
    K = max(v.shape[0] for v in labels.values())
    names = names or [str(i) for i in range(1, K + 1)]
    gray = {}
    def G(i):
        if i not in gray:
            if len(gray) > 60:
                gray.pop(next(iter(gray)))
            gray[i] = cv2.cvtColor(cv2.imread(str(frames[i])), cv2.COLOR_BGR2GRAY)
        return gray[i]
    # estimates[(frame, track)] = list of (points (K,2), confidence (K,), temporal distance)
    est = defaultdict(list)
    sigma = a.fb_sigma
    kernel = np.ones((2 * a.mask_margin + 1, 2 * a.mask_margin + 1), np.uint8)
    for (f0, tid), P0 in sorted(labels.items()):
        est[(f0, tid)].append((P0.copy(), np.where(np.isnan(P0[:, 0]), 0.0, 1.0), 0))
        for step in (1, -1):
            P, conf, worst = P0.copy(), np.where(np.isnan(P0[:, 0]), 0.0, 1.0), np.zeros(K)
            f = f0
            while 0 <= f + step < len(frames) and abs(f + step - f0) <= a.max_gap:
                nf = f + step
                if tid not in tmask.get(nf, {}) and tmask:                # individual not in this frame: stop
                    break
                good = ~np.isnan(P[:, 0]) & (conf > 0)
                if not good.any():
                    break
                P1, fb = _lk(G(f), G(nf), np.nan_to_num(P))
                worst = np.maximum(worst, np.where(good, fb, np.inf))
                conf = np.where(good, np.exp(-worst / sigma), 0.0)
                P = np.where(good[:, None], P1, np.nan)
                m = tmask.get(nf, {}).get(tid)
                if m is not None:
                    md = cv2.dilate(m.astype(np.uint8), kernel).astype(bool)
                    for k in range(K):
                        if good[k]:
                            x, y = int(round(P[k, 0])), int(round(P[k, 1]))
                            if not (0 <= y < h and 0 <= x < w and md[y, x]):
                                conf[k] *= 0.5
                est[(nf, tid)].append((P.copy(), conf.copy(), abs(nf - f0)))
                f = nf
    images, anns = [], []
    track_ids = sorted({t for (_, t) in est})
    for fi, p in enumerate(frames):
        images.append({"id": fi + 1, "file_name": p.name, "width": w, "height": h, "frame_index": fi})
        for tid in track_ids:
            cand = est.get((fi, tid))
            if not cand:
                continue
            Pm, Cm = np.zeros((K, 2)), np.zeros(K)
            for P, C, dist in cand:                                  # confidence- and nearness-weighted blend
                wgt = C / (1.0 + dist)
                ok = ~np.isnan(P[:, 0])
                Pm[ok] += (P[ok] * wgt[ok, None]); Cm[ok] += wgt[ok]
            conf = np.array([max(c[1][k] for c in cand) for k in range(K)])
            kp, sc = [], []
            for k in range(K):
                if Cm[k] > 0 and conf[k] >= a.min_conf:
                    x, y = Pm[k] / Cm[k]
                    kp += [round(float(x), 2), round(float(y), 2), 2 if conf[k] >= 0.6 else 1]
                else:
                    kp += [0, 0, 0]
                sc.append(round(float(conf[k]), 4))
            vis = [(kp[3 * k], kp[3 * k + 1]) for k in range(K) if kp[3 * k + 2] > 0]
            bbox = ([min(x for x, _ in vis), min(y for _, y in vis), max(x for x, _ in vis) - min(x for x, _ in vis),
                     max(y for _, y in vis) - min(y for _, y in vis)] if vis else [0, 0, 0, 0])
            anns.append({"id": len(anns) + 1, "image_id": fi + 1, "category_id": 1, "track_id": tid, "keypoints": kp,
                         "num_keypoints": len(vis), "keypoint_scores": sc, "point_order": list(range(1, K + 1)),
                         "bbox": bbox, "area": max(1.0, bbox[2] * bbox[3]), "iscrowd": 0,
                         "labelled": any(d == 0 for _, _, d in cand)})
    json.dump({"images": images, "annotations": anns,
               "categories": [{"id": 1, "name": "keypoints", "keypoints": names, "skeleton": skel}]},
              open(a.out, "w"))
    n_lab = sum(1 for x in anns if x["labelled"])
    mean_conf = np.mean([s for x in anns for s in x["keypoint_scores"]]) if anns else 0
    print(f"poses for {len(track_ids)} individuals on {len({x['image_id'] for x in anns})} frames "
          f"({n_lab} labelled, the rest carried); mean confidence {mean_conf:.2f} -> {a.out}")


# ------------------------------------------------------------------ export and overlay
def cmd_export_dlc(a):
    d = json.load(open(a.pose))
    cat = d["categories"][0]; names = cat["keypoints"]
    ims = {im["id"]: im for im in d["images"]}
    tracks = sorted({x["track_id"] for x in d["annotations"]})
    by = defaultdict(dict)
    for x in d["annotations"]:
        by[x["image_id"]][x["track_id"]] = x
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        cols = [(f"individual{t}", n, c) for t in tracks for n in names for c in ("x", "y", "likelihood")]
        w.writerow(["scorer"] + [a.scorer] * len(cols))
        w.writerow(["individuals"] + [c[0] for c in cols])
        w.writerow(["bodyparts"] + [c[1] for c in cols])
        w.writerow(["coords"] + [c[2] for c in cols])
        for iid in sorted(ims):
            row = [f"labeled-data/{a.video_name}/{ims[iid]['file_name']}"]
            for t in tracks:
                x = by[iid].get(t)
                for k in range(len(names)):
                    if x and x["keypoints"][3 * k + 2] > 0:
                        row += [x["keypoints"][3 * k], x["keypoints"][3 * k + 1], x["keypoint_scores"][k]]
                    else:
                        row += ["", "", ""]
            w.writerow(row)
    print(f"wrote DeepLabCut-style CSV {a.out}: {len(tracks)} individuals x {len(names)} bodyparts x {len(ims)} frames")


def cmd_overlay(a):
    frames = frame_list(a.frames)
    tr = json.load(open(a.tracks)) if a.tracks else None
    po = json.load(open(a.pose)) if a.pose else None
    h, w = cv2.imread(str(frames[0])).shape[:2]
    tby, pby = defaultdict(list), defaultdict(list)
    for x in (tr or {}).get("annotations", []):
        tby[x["image_id"] - 1].append(x)
    for x in (po or {}).get("annotations", []):
        pby[x["image_id"] - 1].append(x)
    skel = (po or {}).get("categories", [{}])[0].get("skeleton", []) if po else []
    vw = None
    if a.out.lower().endswith(".mp4"):
        vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    else:
        Path(a.out).mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(frames):
        im = cv2.imread(str(p))
        for x in tby.get(i, []):
            col = PALETTE[(x["track_id"] - 1) % len(PALETTE)]
            m = mask_from_seg(x["segmentation"], h, w)
            im[m] = (0.55 * im[m] + 0.45 * np.array(col)).astype(np.uint8)
            bx, by_, bw, bh = map(int, x["bbox"])
            cv2.rectangle(im, (bx, by_), (bx + bw, by_ + bh), col, 1)
            cv2.putText(im, f"#{x['track_id']}", (bx, max(12, by_ - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        for x in pby.get(i, []):
            kp = np.array(x["keypoints"]).reshape(-1, 3)
            col = PALETTE[(x["track_id"] - 1) % len(PALETTE)]
            for s, t in skel:
                if kp[s - 1, 2] > 0 and kp[t - 1, 2] > 0:
                    cv2.line(im, tuple(map(int, kp[s - 1, :2])), tuple(map(int, kp[t - 1, :2])), (255, 255, 255), 2)
            for (x_, y_, v) in kp:
                if v > 0:
                    cv2.circle(im, (int(x_), int(y_)), 4, col if v == 2 else (0, 200, 255), -1)
        if vw is not None:
            vw.write(im)
        else:
            cv2.imwrite(str(Path(a.out) / p.name), im)
    if vw is not None:
        vw.release()
    print(f"overlay -> {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("Annotated frames")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("frames"); p.add_argument("video"); p.add_argument("--out-dir", required=True)
    p.add_argument("--every", type=int, default=1); p.add_argument("--max-frames", type=int)
    p.add_argument("--max-side", type=int, default=1280, help="shrink frames whose longer side is larger (0 = keep)")
    p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("track"); p.add_argument("frames"); p.add_argument("--init", required=True); p.add_argument("--out", required=True)
    p.add_argument("--sam2-checkpoint", required=True); p.add_argument("--sam2-config", default="sam2_hiera_l.yaml")
    p.add_argument("--category", default="individual"); p.add_argument("--min-area", type=float, default=20)
    p.add_argument("--offload", action="store_true", help="keep SAM2's state on the CPU (long videos, small GPUs)")
    p.add_argument("--cpu", action="store_true"); p.set_defaults(fn=cmd_track)
    p = sub.add_parser("pose"); p.add_argument("frames"); p.add_argument("--tracks", required=True)
    p.add_argument("--labels", required=True); p.add_argument("--out", required=True)
    p.add_argument("--max-gap", type=int, default=100000, help="carry labels at most this many frames")
    p.add_argument("--fb-sigma", type=float, default=2.0, help="px: round-trip error at which confidence drops to 1/e")
    p.add_argument("--min-conf", type=float, default=0.05); p.add_argument("--mask-margin", type=int, default=6)
    p.set_defaults(fn=cmd_pose)
    p = sub.add_parser("export-dlc"); p.add_argument("pose"); p.add_argument("--out", required=True)
    p.add_argument("--scorer", default="descriptron"); p.add_argument("--video-name", default="video")
    p.set_defaults(fn=cmd_export_dlc)
    p = sub.add_parser("overlay"); p.add_argument("frames"); p.add_argument("--tracks"); p.add_argument("--pose")
    p.add_argument("--out", required=True); p.add_argument("--fps", type=float, default=15); p.set_defaults(fn=cmd_overlay)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
