#!/usr/bin/env python3
"""
Can Claude find a homologous landmark if it is shown one?  (grounding test, v1, 2026-09-26)
===========================================================================================
Claude cannot be fine-tuned, so this tests what can be given to it at test time instead:
a labelled example (a reference wing with the landmark circled) and a check (cycle
consistency), against DINOLand (DINOv3 patch-token landmark transfer) on the same wings,
scored against hand-placed ground truth.

Arms, each scored as error in % of wing length against the hand-placed landmark:
  claude_fwd     Claude sees the reference with ONE landmark circled and the unmarked target,
                 and places the homologous point on the target.
  claude_cycle   claude_fwd, kept only if a SECOND, independent Claude session, shown the
                 target with Claude's point circled and the unmarked reference, maps it back
                 to within --cycle_tol of the true reference landmark (SAM2-PAL's round trip).
  dinoland       dinov3_landmark_transfer_v52.py from the same single reference.
  agree          claude_fwd, kept only where it lands within --agree_tol of DINOLand.

Every Claude call is a separate `claude -p` process (biorag_llm_backend claude-code backend:
no session persistence, no tools, parent-session variables stripped), so no call sees another.
Calls are cached in <out_dir>/claude_calls.jsonl, so an interrupted run resumes.

Wings are first brought to one orientation (mirror images flipped, upside-down ones turned),
as the homology frames are, so the test measures matching and not orientation.

Stages (run in order; each reads the previous one's output):
  prepare   pick the reference + targets, write canonical crops and ground truth
  claude    forward and back calls
  dinoland  write run_dinoland.sh for these crops (run it in the biorag env: DINOv3 needs transformers >= 4.56)
  score     tables + figure
  choose    set-of-marks arm: Claude picks among lettered candidates (dinoland | truth)
  score_choose  tables for the choose arm
  choose_back   exact round trip on the choose arm (independent session, lettered reference)

Example:
  python claude_landmark_grounding_test_v1.py prepare --coco <forewing_keypoints.json> \
      --image_dir <images> --out_dir <out> --n_targets 20 --landmarks 3,6,9,13
  python claude_landmark_grounding_test_v1.py claude --out_dir <out> --model sonnet
  python claude_landmark_grounding_test_v1.py dinoland --out_dir <out> --dinoland <dinov3_landmark_transfer_v52.py>
  python claude_landmark_grounding_test_v1.py score --out_dir <out>
"""
import argparse
import base64
import io
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LONG_SIDE = 800          # px, long side of every crop Claude and DINOLand see
MARGIN = 0.12            # crop margin around the landmarks, fraction of the landmark bbox

SYSTEM = ("You are an expert insect morphologist who knows psyllid (Hemiptera: Psylloidea) "
          "forewing venation. You compare specimens point by point. Reply with JSON only.")

FWD_TEXT = ("Image 1 is a reference psyllid forewing. A red circle marks one landmark: a point "
            "on the venation (a vein junction, vein ending or cell corner).\n"
            "Image 2 is the forewing of another specimen, possibly another species, shown in "
            "the same orientation.\n"
            "Find the homologous point on image 2: the same junction of the same veins, not "
            "simply the same position in the frame.\n"
            'Reply with JSON only: {"x": <0-1, from the left edge of image 2>, '
            '"y": <0-1, from the top edge of image 2>, "confidence": "high|medium|low", '
            '"reason": "<one short sentence naming the veins>"}')

BACK_TEXT = FWD_TEXT          # the same task, run in the other direction


# ------------------------------------------------------------------ helpers
def species_of(fn):
    return fn.split("_forewing")[0].rsplit("_", 1)[0]


def canonical(img, pts, flip, rot180):
    """Apply the orientation fix to an image and its landmarks (pixel coords)."""
    W, H = img.size
    pts = pts.copy()
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT); pts[:, 0] = W - 1 - pts[:, 0]
    if rot180:
        img = img.transpose(Image.ROTATE_180)
        pts[:, 0] = W - 1 - pts[:, 0]; pts[:, 1] = H - 1 - pts[:, 1]
    return img, pts


def handed(p):
    x, y = p[:, 0], p[:, 1]
    return 1 if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) >= 0 else -1


def crop_resize(img, pts, long_side=None):
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    m = MARGIN * max(x1 - x0, y1 - y0)
    box = (max(0, int(x0 - m)), max(0, int(y0 - m)),
           min(img.size[0], int(x1 + m)), min(img.size[1], int(y1 + m)))
    c = img.crop(box)
    s = (long_side or LONG_SIDE) / max(c.size)
    c = c.resize((round(c.size[0] * s), round(c.size[1] * s)), Image.LANCZOS)
    q = (pts - np.array(box[:2])) * s
    return c, q


def wing_length(q):
    d = np.linalg.norm(q[:, None] - q[None], axis=-1)
    return float(d.max())


def b64(img):
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG", quality=92)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(buf.getvalue()).decode()}}


def mark(img, xy, r=14):
    im = img.convert("RGB").copy(); d = ImageDraw.Draw(im)
    x, y = xy
    for w, col in ((5, (255, 255, 255)), (3, (230, 0, 0))):
        d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=w)
    return im


# ------------------------------------------------------------------ prepare
def prepare(a):
    coco = json.load(open(a.coco))
    imgs = {i["id"]: i for i in coco["images"]}
    excl = set(x.strip() for x in (a.exclude or "").split(",") if x.strip())
    wings = []
    for ann in coco["annotations"]:
        fn = imgs[ann["image_id"]]["file_name"]
        if any(e in fn for e in excl):
            continue
        k = np.array(ann["keypoints"], float).reshape(-1, 3)
        if (k[:, 2] == 0).any() or not (Path(a.image_dir) / fn).exists():
            continue
        wings.append((fn, k[:, :2]))
    # orientation: majority handedness kept, the rest flipped; then landmark 1 to the right
    hs = [handed(p) for _, p in wings]
    maj = 1 if sum(hs) >= 0 else -1
    out = Path(a.out_dir); (out / "crops").mkdir(parents=True, exist_ok=True)
    prepped = []
    for (fn, p), h in zip(wings, hs):
        img = Image.open(Path(a.image_dir) / fn)
        img = img.convert("RGB")
        flip = h != maj
        _, p1 = canonical(img, p, flip, False)
        rot = p1[0, 0] < p1[:, 0].mean()
        img2, p2 = canonical(img, p, flip, rot)
        prepped.append(dict(file=fn, species=species_of(fn), flip=bool(flip), rot180=bool(rot),
                            img=img2, pts=p2))
    # reference = the wing closest to the mean shape (Procrustes-free: centred + scaled)
    def norm(p):
        q = p - p.mean(0); return q / np.linalg.norm(q)
    M = np.mean([norm(w["pts"]) for w in prepped], 0)
    ref = min(prepped, key=lambda w: np.linalg.norm(norm(w["pts"]) - M))
    rng = random.Random(a.seed)
    pool = [w for w in prepped if w is not ref]
    rng.shuffle(pool)
    by_sp, targets = {}, []
    for w in pool:                              # one per species first, then fill
        if w["species"] not in by_sp and w["species"] != ref["species"]:
            by_sp[w["species"]] = w; targets.append(w)
    for w in pool:
        if len(targets) >= a.n_targets: break
        if w not in targets: targets.append(w)
    targets = targets[:a.n_targets]
    # extra references (multi-reference arm): the next wings closest to the mean shape, other species
    extra = []
    for w in sorted(pool, key=lambda w: np.linalg.norm(norm(w["pts"]) - M)):
        if len(extra) >= a.n_extra_refs: break
        if w in targets or w["species"] == ref["species"] or w["species"] in {e["species"] for e in extra}:
            continue
        extra.append(w)
    rows = []
    for role, w in [("reference", ref)] + [("target", t) for t in targets] + [("extra_reference", e) for e in extra]:
        c, q = crop_resize(w["img"], w["pts"], a.long_side)
        name = Path(w["file"]).stem + ".png"
        c.save(out / "crops" / name)
        rows.append(dict(role=role, crop=name, file=w["file"], species=w["species"],
                         flip=w["flip"], rot180=w["rot180"], size=list(c.size),
                         wing_length=wing_length(q), landmarks=q.round(2).tolist()))
    lms = [int(x) for x in a.landmarks.split(",")]
    json.dump(dict(landmarks=lms, reference=rows[0], targets=[r for r in rows if r["role"] == "target"],
                   extra_references=[r for r in rows if r["role"] == "extra_reference"], seed=a.seed,
                   long_side=a.long_side, source_coco=str(a.coco)), open(out / "design.json", "w"), indent=1)
    # the reference as a one-image COCO for DINOLand (all 17 landmarks)
    r = rows[0]
    kp = []
    for x, y in r["landmarks"]:
        kp += [x, y, 2]
    cat = [c for c in coco["categories"]][0]
    json.dump({"images": [{"id": 1, "file_name": r["crop"], "width": r["size"][0], "height": r["size"][1]}],
               "annotations": [{"id": 1, "image_id": 1, "category_id": cat["id"], "keypoints": kp,
                                "num_keypoints": len(kp) // 3}],
               "categories": [cat]}, open(out / "reference_keypoints.json", "w"), indent=1)
    print(f"reference {r['file']} ({r['species']}); {len(targets)} targets from "
          f"{len({t['species'] for t in targets})} species; {len(extra)} extra references; landmarks {lms}; "
          f"{sum(w['flip'] for w in prepped)} of {len(prepped)} wings flipped, "
          f"{sum(w['rot180'] for w in prepped)} turned 180")


# ------------------------------------------------------------------ claude
_LOCK = threading.Lock()


def _cached(path):
    done = {}
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line); done[r["key"]] = r
            except Exception:
                pass
    return done


def _ask(client, model, img1, img2, text):
    from biorag_llm_backend import parse_json_response
    content = [{"type": "text", "text": "Image 1:"}, b64(img1),
               {"type": "text", "text": "Image 2:"}, b64(img2),
               {"type": "text", "text": text}]
    r = client.messages.create(model=model, max_tokens=400, system=SYSTEM,
                               messages=[{"role": "user", "content": content}])
    txt = "".join(getattr(b, "text", "") for b in r.content)
    try:
        j = parse_json_response(txt)
        return float(j["x"]), float(j["y"]), j.get("confidence"), j.get("reason"), txt, r.model
    except Exception:
        return None, None, None, None, txt, r.model


def run_claude(a):
    from biorag_llm_backend import make_llm_client
    out = Path(a.out_dir); d = json.load(open(out / "design.json"))
    client = make_llm_client("claude-code", log_path=str(out / "llm_calls.jsonl"))
    cache_p = out / "claude_calls.jsonl"; done = _cached(cache_p)
    ref = d["reference"]
    # load every image fully before the threads start: PIL's lazy reads are not thread-safe
    refimg = Image.open(out / "crops" / ref["crop"]).convert("RGB")
    timgs = {t["crop"]: Image.open(out / "crops" / t["crop"]).convert("RGB") for t in d["targets"]}
    jobs = [(t, lm) for t in d["targets"] for lm in d["landmarks"]]

    def one(job):
        t, lm = job
        timg = timgs[t["crop"]]; W, H = timg.size
        rx, ry = ref["landmarks"][lm - 1]
        kf = f"fwd|{t['crop']}|{lm}"
        if kf in done:
            f = done[kf]
        else:
            x, y, conf, why, txt, model = _ask(client, a.model, mark(refimg, (rx, ry)), timg, FWD_TEXT)
            f = dict(key=kf, arm="fwd", target=t["crop"], lm=lm, x=x, y=y, confidence=conf,
                     reason=why, raw=txt, model=model)
            with _LOCK:
                open(cache_p, "a").write(json.dumps(f) + "\n")
        if f["x"] is None:
            return
        kb = f"back|{t['crop']}|{lm}"
        if kb not in done:
            px, py = f["x"] * W, f["y"] * H
            x, y, conf, why, txt, model = _ask(client, a.model, mark(timg, (px, py)), refimg, BACK_TEXT)
            b = dict(key=kb, arm="back", target=t["crop"], lm=lm, x=x, y=y, confidence=conf,
                     reason=why, raw=txt, model=model)
            with _LOCK:
                open(cache_p, "a").write(json.dumps(b) + "\n")
        print(f"  {t['crop'][:40]:40s} lm{lm:>2} done", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs))


# ------------------------------------------------------------------ dinoland
def dinoland(a):
    out = Path(a.out_dir)
    cmd = (f'python "{a.dinoland}" --imgA "{out}/crops/{json.load(open(out/"design.json"))["reference"]["crop"]}" '
           f'--landmarks "{out}/reference_keypoints.json" --ref_dir "{out}/crops" '
           f'--batch_glob "{out}/crops/*.png" --batch_n 1000 --align feature --outdir "{out}/dinoland"')
    print(cmd)
    (out / "run_dinoland.sh").write_text("#!/bin/bash\nset -e\n" + cmd + "\n")


# ------------------------------------------------------------------ score
def score(a):
    out = Path(a.out_dir); d = json.load(open(out / "design.json"))
    calls = _cached(out / "claude_calls.jsonl")
    ref = d["reference"]; RL = ref["wing_length"]
    dl = {}
    dp = out / "dinoland" / "predictions_keypoints.json"
    if dp.exists():
        dj = json.load(open(dp)); names = {i["id"]: i["file_name"] for i in dj["images"]}
        for an in dj["annotations"]:
            k = np.array(an["keypoints"], float).reshape(-1, 3)
            dl[Path(names[an["image_id"]]).name] = k
    rows = []
    for t in d["targets"]:
        W, H = t["size"]; L = t["wing_length"]
        for lm in d["landmarks"]:
            gx, gy = t["landmarks"][lm - 1]
            r = dict(target=t["crop"], species=t["species"], lm=lm)
            f = calls.get(f"fwd|{t['crop']}|{lm}"); b = calls.get(f"back|{t['crop']}|{lm}")
            if f and f["x"] is not None:
                r["claude_err"] = 100 * np.hypot(f["x"] * W - gx, f["y"] * H - gy) / L
                r["confidence"] = f["confidence"]
            if b and b["x"] is not None:
                rw, rh = ref["size"]; rx, ry = ref["landmarks"][lm - 1]
                r["cycle_err"] = 100 * np.hypot(b["x"] * rw - rx, b["y"] * rh - ry) / RL
            k = dl.get(t["crop"])
            if k is not None:
                dx, dy, v = k[lm - 1]
                if v > 0:
                    r["dino_err"] = 100 * np.hypot(dx - gx, dy - gy) / L
                    r["dino_v"] = int(v)
                    if f and f["x"] is not None:
                        r["claude_dino_gap"] = 100 * np.hypot(f["x"] * W - dx, f["y"] * H - dy) / L
            rows.append(r)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(out / "per_landmark_errors.csv", index=False)

    def summ(name, e, n_all):
        e = np.asarray([x for x in e if x == x])
        if not len(e):
            return dict(arm=name, n=0, coverage=0)
        return dict(arm=name, n=len(e), coverage=round(len(e) / n_all, 3),
                    median_err_pct=round(float(np.median(e)), 2),
                    within_5pct=round(float((e <= 5).mean()), 3),
                    within_10pct=round(float((e <= 10).mean()), 3))
    N = len(df)
    S = [summ("claude_fwd", df.get("claude_err", []), N)]
    if "cycle_err" in df:
        keep = df[df["cycle_err"] <= a.cycle_tol]
        S.append(summ(f"claude_cycle (round trip <= {a.cycle_tol}%)", keep.get("claude_err", []), N))
        rej = df[df["cycle_err"] > a.cycle_tol]
        S.append(summ("  rejected by round trip", rej.get("claude_err", []), N))
    if "dino_err" in df:
        S.append(summ("dinoland", df["dino_err"], N))
        if "dino_v" in df:
            S.append(summ("dinoland (v=2, confirmed only)", df[df["dino_v"] == 2]["dino_err"], N))
        if "claude_dino_gap" in df:
            ag = df[df["claude_dino_gap"] <= a.agree_tol]
            S.append(summ(f"claude where it agrees with dinoland (<= {a.agree_tol}%)", ag["claude_err"], N))
    for c in ("high", "medium", "low"):
        if "confidence" in df:
            S.append(summ(f"  claude, self-rated {c}", df[df["confidence"] == c]["claude_err"], N))
    sm = pd.DataFrame(S); sm.to_csv(out / "summary.csv", index=False)
    per_lm = df.groupby("lm").agg(claude_median=("claude_err", "median"),
                                  **({"dino_median": ("dino_err", "median")} if "dino_err" in df else {}))
    per_lm.to_csv(out / "per_landmark_medians.csv")
    print(sm.to_string(index=False)); print(); print(per_lm.round(2).to_string())
    if "cycle_err" in df and df["cycle_err"].notna().sum() > 3:
        from scipy.stats import spearmanr
        m = df[["cycle_err", "claude_err"]].dropna()
        rho, p = spearmanr(m["cycle_err"], m["claude_err"])
        print(f"\nround-trip error vs true error: Spearman rho={rho:.2f} (p={p:.3g}, n={len(m)})")
    _figure(out, d, calls, dl)


def _figure(out, d, calls, dl, n=6):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = d["targets"][:n]; lm = d["landmarks"][0]
    fig, axs = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.2))
    ref = d["reference"]
    axs[0].imshow(Image.open(out / "crops" / ref["crop"]))
    for k, l in enumerate(d["landmarks"]):
        x, y = ref["landmarks"][l - 1]; axs[0].scatter([x], [y], s=60, facecolor="none", edgecolor="red", lw=2)
        axs[0].text(x + 12, y, str(l), color="red", fontsize=9)
    axs[0].set_title("reference", fontsize=9)
    for ax, t in zip(axs[1:], ts):
        W, H = t["size"]; ax.imshow(Image.open(out / "crops" / t["crop"]))
        for l in d["landmarks"]:
            gx, gy = t["landmarks"][l - 1]; ax.scatter([gx], [gy], s=40, c="lime", marker="+", lw=2)
            f = calls.get(f"fwd|{t['crop']}|{l}")
            if f and f["x"] is not None:
                ax.scatter([f["x"] * W], [f["y"] * H], s=40, facecolor="none", edgecolor="red", lw=1.5)
            k = dl.get(t["crop"])
            if k is not None and k[l - 1, 2] > 0:
                ax.scatter([k[l - 1, 0]], [k[l - 1, 1]], s=30, c="cyan", marker="x", lw=1.5)
        ax.set_title(t["species"][:22], fontsize=8)
    for ax in axs: ax.axis("off")
    fig.suptitle("green +: hand-placed   red o: Claude (one reference)   cyan x: DINOLand", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "examples.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------ choose (set-of-marks)
CHOOSE_TEXT = ("Image 1 is a reference psyllid forewing. A red circle marks one landmark on the "
               "venation (a vein junction, vein ending or cell corner).\n"
               "Image 2 is the forewing of another specimen, possibly another species, in the same "
               "orientation. Candidate points on image 2 are marked with letters.\n"
               "Which lettered point on image 2 is homologous to the circled landmark: the same "
               "junction of the same veins, not simply the same position in the frame? Exactly one "
               "letter is intended; if none fits exactly, choose the closest homologue.\n"
               'Reply with JSON only: {"letter": "<one letter>", "confidence": "high|medium|low", '
               '"reason": "<one short sentence naming the veins>"}')
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"          # no I/O, which read as 1/0


def mark_candidates(img, pts, letters, r=9):
    """Small numbered-style markers: a dot with a letter tag, white halo for legibility."""
    from PIL import ImageFont
    im = img.convert("RGB").copy(); d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for (x, y), L in zip(pts, letters):
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255), width=4)
        d.ellipse([x - r, y - r, x + r, y + r], outline=(0, 90, 255), width=2)
        tx, ty = x + r + 2, y - r - 14
        d.rectangle([tx - 2, ty, tx + 14, ty + 19], fill=(255, 255, 255))
        d.text((tx, ty), L, fill=(0, 60, 200), font=font)
    return im


ZOOM_TEXT = ("Image 1 is a reference psyllid forewing; a red circle marks one landmark on the venation "
             "(a vein junction, vein ending or cell corner). Image 2 is a close-up of the same reference "
             "around that landmark, circled again.\n"
             "Image 3 is a close-up of the forewing of another specimen, possibly another species, in the "
             "same orientation. Candidate points on image 3 are marked with letters.\n"
             "Which lettered point on image 3 is homologous to the circled landmark: the same junction of "
             "the same veins? Exactly one letter is intended; if none fits exactly, choose the closest "
             "homologue.\n"
             'Reply with JSON only: {"letter": "<one letter>", "confidence": "high|medium|low", '
             '"reason": "<one short sentence naming the veins>"}')

MULTI_TEXT = ("Images 1 to {n} are reference psyllid forewings from {n} different specimens. On each, a red "
              "circle marks the SAME homologous landmark on the venation (a vein junction, vein ending or "
              "cell corner).\n"
              "Image {t} is the forewing of another specimen, possibly another species, in the same "
              "orientation. Candidate points on image {t} are marked with letters.\n"
              "Which lettered point on image {t} is homologous to the circled landmark: the same junction "
              "of the same veins, not simply the same position in the frame? Exactly one letter is intended; "
              "if none fits exactly, choose the closest homologue.\n"
              'Reply with JSON only: {{"letter": "<one letter>", "confidence": "high|medium|low", '
              '"reason": "<one short sentence naming the veins>"}}')


ANCHOR_TEXT = ("Image 1 is a reference psyllid forewing. A red circle marks one landmark on the venation "
               "(a vein junction, vein ending or cell corner). Three other landmarks are labelled with Greek "
               "letters in orange.\n"
               "Image 2 is the forewing of another specimen, possibly another species, in the same "
               "orientation. The SAME three landmarks are labelled with the same Greek letters in orange; use "
               "them to orient yourself. Candidate points on image 2 are marked with Latin letters in blue.\n"
               "Which Latin-lettered point on image 2 is homologous to the circled landmark: the same junction "
               "of the same veins? Exactly one letter is intended; if none fits exactly, choose the closest "
               "homologue.\n"
               'Reply with JSON only: {"letter": "<one Latin letter>", "confidence": "high|medium|low", '
               '"reason": "<one short sentence naming the veins and the Greek-labelled landmarks you used>"}')
GREEK = ["\u03b1", "\u03b2", "\u03b3"]


def mark_anchors(img, pts, labels, r=9):
    from PIL import ImageFont
    im = img.convert("RGB").copy(); d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for (x, y), L in zip(pts, labels):
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 140, 0), outline=(255, 255, 255), width=2)
        tx, ty = x - r - 22, y - r - 22
        d.rectangle([tx - 2, ty, tx + 18, ty + 25], fill=(255, 255, 255))
        d.text((tx, ty), L, fill=(220, 100, 0), font=font)
    return im


def pick_anchors(ref_pts, lm, k=3, n_exclude_near=3):
    """k anchors spread over the wing: drop the query and its nearest neighbours (so the anchors orient
    without pointing at the answer), then farthest-point sampling from the query."""
    P = np.asarray(ref_pts, float); q = P[lm - 1]
    d = np.linalg.norm(P - q, axis=1)
    near = set(np.argsort(d)[: n_exclude_near + 1])        # includes the query itself
    pool = [i for i in range(len(P)) if i not in near]
    chosen = [max(pool, key=lambda i: d[i])]
    while len(chosen) < k:
        chosen.append(max((i for i in pool if i not in chosen),
                          key=lambda i: min(np.linalg.norm(P[i] - P[j]) for j in chosen + [lm - 1])))
    return [i + 1 for i in chosen]


def _window(img, cx, cy, side, out_px=800):
    """Square close-up of side `side` px centred on (cx, cy), clamped to the image, resized to out_px."""
    W, H = img.size
    side = min(side, W, H)
    x0 = int(min(max(cx - side / 2, 0), W - side)); y0 = int(min(max(cy - side / 2, 0), H - side))
    box = (x0, y0, x0 + int(side), y0 + int(side))
    s = out_px / (box[2] - box[0])
    return img.crop(box).resize((out_px, out_px), Image.LANCZOS), box, s


def _small(img, long_side=800):
    s = long_side / max(img.size)
    return img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS), s


def run_choose(a):
    """Set-of-marks arm: Claude chooses among lettered candidates instead of pointing.
    --candidates dinoland: the 17 points DINOLand predicted on the target (the practical pipeline);
    --candidates truth: the 17 hand-placed points (control: pure homology recognition)."""
    from biorag_llm_backend import make_llm_client, parse_json_response
    out = Path(a.out_dir); d = json.load(open(out / "design.json"))
    client = make_llm_client("claude-code", log_path=str(out / "llm_calls.jsonl"))
    tag = a.candidates + ("" if a.variant == "single" else f"_{a.variant}")
    cache_p = out / f"choose_{tag}.jsonl"; done = _cached(cache_p)
    ref = d["reference"]
    refimg = Image.open(out / "crops" / ref["crop"]).convert("RGB")
    timgs = {t["crop"]: Image.open(out / "crops" / t["crop"]).convert("RGB") for t in d["targets"]}
    extras = [(e, Image.open(out / "crops" / e["crop"]).convert("RGB")) for e in d.get("extra_references", [])]
    if a.variant == "multiref" and not extras:
        raise SystemExit("multiref needs extra references: prepare with --n_extra_refs")
    centre = {}
    if a.variant == "zoom":          # window centre = DINOLand's prediction, rescaled from the run it came from
        src = Path(a.dinoland_from); sd = json.load(open(src / "design.json"))
        ssz = {t["crop"]: t["size"] for t in sd["targets"]}
        dj = json.load(open(src / "dinoland" / "predictions_keypoints.json"))
        names = {i["id"]: i["file_name"] for i in dj["images"]}
        for an in dj["annotations"]:
            nm = Path(names[an["image_id"]]).name
            t = next((t for t in d["targets"] if t["crop"] == nm), None)
            if t is None: continue
            f_ = t["size"][0] / ssz[nm][0]
            centre[nm] = np.array(an["keypoints"], float).reshape(-1, 3)[:, :2] * f_
    dl = {}
    if a.candidates == "dinoland":
        dj = json.load(open(out / "dinoland" / "predictions_keypoints.json"))
        names = {i["id"]: i["file_name"] for i in dj["images"]}
        for an in dj["annotations"]:
            dl[Path(names[an["image_id"]]).name] = np.array(an["keypoints"], float).reshape(-1, 3)
    jobs = [(t, lm) for t in d["targets"] for lm in d["landmarks"]]

    def one(job):
        t, lm = job
        key = f"choose-{tag}|{t['crop']}|{lm}"
        if key in done:
            return
        if a.candidates == "truth":
            cand = [(i + 1, x, y) for i, (x, y) in enumerate(t["landmarks"])]
        else:
            k = dl[t["crop"]]
            cand = [(i + 1, x, y) for i, (x, y, v) in enumerate(k) if v > 0]
        timg = timgs[t["crop"]]
        rx, ry = ref["landmarks"][lm - 1]
        in_window = True
        anchors = []
        if a.variant == "anchors":
            anchors = pick_anchors(ref["landmarks"], lm)
            cand = [c for c in cand if c[0] not in anchors]
        if a.variant == "zoom":
            cx, cy = centre[t["crop"]][lm - 1]
            tw, tbox, ts = _window(timg, cx, cy, a.zoom_frac * t["wing_length"])
            cand = [(i, (x - tbox[0]) * ts, (y - tbox[1]) * ts) for i, x, y in cand
                    if tbox[0] <= x < tbox[2] and tbox[1] <= y < tbox[3]]
            in_window = any(i == lm for i, _, _ in cand)
            if not cand:
                rec = dict(key=key, candidates=a.candidates, variant=a.variant, target=t["crop"], lm=lm,
                           letter="", chosen_lm=None, x=None, y=None, n_candidates=0, in_window=False)
                with _LOCK:
                    open(cache_p, "a").write(json.dumps(rec) + "\n")
                return
        rng = random.Random(f"{a.seed}|{key}")         # letters shuffled per call, reproducibly
        order = cand[:]; rng.shuffle(order)
        letters = LETTERS[:len(order)]
        if a.variant == "zoom":
            rsmall, rs = _small(refimg)
            rw, rbox, rsz = _window(refimg, rx, ry, a.zoom_frac * ref["wing_length"])
            content = [{"type": "text", "text": "Image 1:"}, b64(mark(rsmall, (rx * rs, ry * rs))),
                       {"type": "text", "text": "Image 2:"},
                       b64(mark(rw, ((rx - rbox[0]) * rsz, (ry - rbox[1]) * rsz), r=22)),
                       {"type": "text", "text": "Image 3:"},
                       b64(mark_candidates(tw, [(x, y) for _, x, y in order], letters)),
                       {"type": "text", "text": ZOOM_TEXT}]
        elif a.variant == "anchors":
            ri = mark_anchors(mark(refimg, (rx, ry)), [ref["landmarks"][i - 1] for i in anchors], GREEK)
            ti = mark_anchors(mark_candidates(timg, [(x, y) for _, x, y in order], letters),
                              [t["landmarks"][i - 1] for i in anchors], GREEK)
            content = [{"type": "text", "text": "Image 1:"}, b64(ri), {"type": "text", "text": "Image 2:"}, b64(ti),
                       {"type": "text", "text": ANCHOR_TEXT}]
        elif a.variant == "multiref":
            refs = [(ref, refimg)] + extras
            content = []
            for n_, (rr, ri) in enumerate(refs, 1):
                content += [{"type": "text", "text": f"Image {n_}:"},
                            b64(mark(ri, tuple(rr["landmarks"][lm - 1])))]
            content += [{"type": "text", "text": f"Image {len(refs) + 1}:"},
                        b64(mark_candidates(timg, [(x, y) for _, x, y in order], letters)),
                        {"type": "text", "text": MULTI_TEXT.format(n=len(refs), t=len(refs) + 1)}]
        else:
            content = [{"type": "text", "text": "Image 1:"}, b64(mark(refimg, (rx, ry))),
                       {"type": "text", "text": "Image 2:"},
                       b64(mark_candidates(timg, [(x, y) for _, x, y in order], letters)),
                       {"type": "text", "text": CHOOSE_TEXT}]
        r = client.messages.create(model=a.model, max_tokens=400, system=SYSTEM,
                                   messages=[{"role": "user", "content": content}])
        txt = "".join(getattr(b, "text", "") for b in r.content)
        try:
            j = parse_json_response(txt); L = str(j.get("letter", "")).strip().upper()[:1]
        except Exception:
            j, L = {}, ""
        pick = dict(zip(letters, order)).get(L)
        # x, y are stored in the target crop's own pixels, whatever the variant showed
        if pick and a.variant == "zoom":
            px, py = tbox[0] + pick[1] / ts, tbox[1] + pick[2] / ts
        else:
            px, py = (pick[1], pick[2]) if pick else (None, None)
        rec = dict(key=key, candidates=a.candidates, variant=a.variant, in_window=in_window, anchors=anchors,
                   target=t["crop"], lm=lm, letter=L,
                   chosen_lm=pick[0] if pick else None,
                   x=px if pick else None, y=py if pick else None,
                   n_candidates=len(order), letters={L_: c[0] for L_, c in zip(letters, order)},
                   confidence=j.get("confidence"), reason=j.get("reason"), raw=txt, model=r.model)
        with _LOCK:
            open(cache_p, "a").write(json.dumps(rec) + "\n")
        print(f"  {t['crop'][:40]:40s} lm{lm:>2} -> {L} (lm {rec['chosen_lm']})", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs))


def score_choose(a):
    out = Path(a.out_dir); d = json.load(open(out / "design.json"))
    import pandas as pd
    rows = []
    for p in sorted(out.glob("choose_*.jsonl")):
        if p.name.startswith("choose_back"):
            continue
        cand = p.stem[len("choose_"):]
        recs = _cached(p)
        tmap = {t["crop"]: t for t in d["targets"]}
        for r in recs.values():
            t = tmap[r["target"]]; gx, gy = t["landmarks"][r["lm"] - 1]
            err = (100 * np.hypot(r["x"] - gx, r["y"] - gy) / t["wing_length"]) if r["x"] is not None else np.nan
            rows.append(dict(candidates=cand, target=r["target"], lm=r["lm"], chosen_lm=r["chosen_lm"],
                             correct=r["chosen_lm"] == r["lm"], err_pct=err,
                             n_candidates=max(r["n_candidates"], 1), confidence=r.get("confidence"),
                             in_window=r.get("in_window", True)))
    df = pd.DataFrame(rows); df.to_csv(out / "choose_per_landmark.csv", index=False)
    for cand, g in df.groupby("candidates"):
        if (~g["in_window"]).any():
            print(f"{cand}: true landmark outside the close-up in {(~g['in_window']).sum()} of {len(g)} "
                  f"(counted as wrong); correct when inside: {g[g.in_window]['correct'].mean():.3f}")
    S = []
    for cand, g in df.groupby("candidates"):
        e = g["err_pct"].dropna()
        S.append(dict(arm=f"claude chooses among {cand} candidates", n=len(g),
                      correct_homologue=round(g["correct"].mean(), 3),
                      chance=round(float((1 / g["n_candidates"]).mean()), 3),
                      median_err_pct=round(float(e.median()), 2),
                      within_5pct=round(float((e <= 5).mean()), 3),
                      within_10pct=round(float((e <= 10).mean()), 3)))
        for c in ("high", "medium", "low"):
            gc = g[g["confidence"] == c]
            if len(gc):
                S.append(dict(arm=f"  self-rated {c}", n=len(gc), correct_homologue=round(gc["correct"].mean(), 3)))
    sm = pd.DataFrame(S); sm.to_csv(out / "choose_summary.csv", index=False)
    print(sm.to_string(index=False))
    print(df.groupby(["candidates", "lm"])["correct"].mean().unstack(0).round(2).to_string())


def run_choose_back(a):
    """Exact round trip for the choose arm: a second, independent session sees the target with
    the point chosen in choose_<candidates> circled, and the reference with its 17 hand-placed
    landmarks lettered (shuffled), and picks the homologue. The round trip passes only if it
    lands on the landmark the forward question was about."""
    from biorag_llm_backend import make_llm_client, parse_json_response
    out = Path(a.out_dir); d = json.load(open(out / "design.json"))
    client = make_llm_client("claude-code", log_path=str(out / "llm_calls.jsonl"))
    fwd = _cached(out / f"choose_{a.candidates}.jsonl")
    cache_p = out / f"choose_back_{a.candidates}.jsonl"; done = _cached(cache_p)
    ref = d["reference"]
    refimg = Image.open(out / "crops" / ref["crop"]).convert("RGB")
    timgs = {t["crop"]: Image.open(out / "crops" / t["crop"]).convert("RGB") for t in d["targets"]}

    def one(r):
        key = "back-" + r["key"]
        if key in done or r["x"] is None:
            return
        cand = [(i + 1, x, y) for i, (x, y) in enumerate(ref["landmarks"])]
        rng = random.Random(f"{a.seed}|{key}")
        order = cand[:]; rng.shuffle(order)
        letters = LETTERS[:len(order)]
        content = [{"type": "text", "text": "Image 1:"}, b64(mark(timgs[r["target"]], (r["x"], r["y"]))),
                   {"type": "text", "text": "Image 2:"},
                   b64(mark_candidates(refimg, [(x, y) for _, x, y in order], letters)),
                   {"type": "text", "text": CHOOSE_TEXT}]
        resp = client.messages.create(model=a.model, max_tokens=400, system=SYSTEM,
                                      messages=[{"role": "user", "content": content}])
        txt = "".join(getattr(b, "text", "") for b in resp.content)
        try:
            j = parse_json_response(txt); L = str(j.get("letter", "")).strip().upper()[:1]
        except Exception:
            j, L = {}, ""
        pick = dict(zip(letters, order)).get(L)
        rec = dict(key=key, fwd_key=r["key"], target=r["target"], lm=r["lm"], fwd_chosen_lm=r["chosen_lm"],
                   back_lm=pick[0] if pick else None, round_trip_ok=(pick is not None and pick[0] == r["lm"]),
                   letter=L, confidence=j.get("confidence"), reason=j.get("reason"), raw=txt, model=resp.model)
        with _LOCK:
            open(cache_p, "a").write(json.dumps(rec) + "\n")
        print(f"  {r['target'][:40]:40s} lm{r['lm']:>2} fwd->{r['chosen_lm']} back->{rec['back_lm']}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, list(fwd.values())))


def score_choose_back(a):
    out = Path(a.out_dir); d = json.load(open(out / "design.json"))
    import pandas as pd
    tmap = {t["crop"]: t for t in d["targets"]}
    fwd = _cached(out / f"choose_{a.candidates}.jsonl"); back = _cached(out / f"choose_back_{a.candidates}.jsonl")
    rows = []
    for k, r in fwd.items():
        b = back.get("back-" + k); t = tmap[r["target"]]; gx, gy = t["landmarks"][r["lm"] - 1]
        err = 100 * np.hypot(r["x"] - gx, r["y"] - gy) / t["wing_length"] if r["x"] is not None else np.nan
        rows.append(dict(target=r["target"], lm=r["lm"], fwd_correct=r["chosen_lm"] == r["lm"], err_pct=err,
                         round_trip_ok=bool(b and b["round_trip_ok"]), back_lm=b["back_lm"] if b else None))
    df = pd.DataFrame(rows); df.to_csv(out / f"choose_round_trip_{a.candidates}.csv", index=False)
    S = []
    for name, g in [("all forward choices", df), ("round trip passed", df[df.round_trip_ok]),
                    ("round trip failed", df[~df.round_trip_ok])]:
        e = g["err_pct"].dropna()
        S.append(dict(subset=name, n=len(g), coverage=round(len(g) / len(df), 3),
                      correct_homologue=round(g["fwd_correct"].mean(), 3) if len(g) else np.nan,
                      median_err_pct=round(float(e.median()), 2) if len(e) else np.nan,
                      within_5pct=round(float((e <= 5).mean()), 3) if len(e) else np.nan))
    sm = pd.DataFrame(S); sm.to_csv(out / f"choose_round_trip_summary_{a.candidates}.csv", index=False)
    print(sm.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="stage", required=True)
    p = sp.add_parser("prepare"); p.add_argument("--coco", required=True); p.add_argument("--image_dir", required=True)
    p.add_argument("--out_dir", required=True); p.add_argument("--n_targets", type=int, default=20)
    p.add_argument("--landmarks", default="3,6,9,13"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", default="", help="comma-separated file-name fragments to leave out (e.g. flagged wings)")
    p.add_argument("--long_side", type=int, default=LONG_SIDE, help="px, long side of every crop (800; 1568 ~ the most the model takes in)")
    p.add_argument("--n_extra_refs", type=int, default=0, help="extra references for the multi-reference arm")
    p = sp.add_parser("claude"); p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="sonnet"); p.add_argument("--workers", type=int, default=4)
    p = sp.add_parser("dinoland"); p.add_argument("--out_dir", required=True); p.add_argument("--dinoland", required=True)
    p = sp.add_parser("choose", help="set-of-marks: Claude picks the homologue among lettered candidates")
    p.add_argument("--out_dir", required=True); p.add_argument("--candidates", choices=["dinoland", "truth"], required=True)
    p.add_argument("--variant", choices=["single", "zoom", "multiref", "anchors"], default="single")
    p.add_argument("--zoom_frac", type=float, default=0.30, help="close-up side, fraction of wing length")
    p.add_argument("--dinoland_from", default=None, help="run dir whose dinoland/ predictions centre the close-up")
    p.add_argument("--model", default="sonnet"); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("score_choose"); p.add_argument("--out_dir", required=True)
    p = sp.add_parser("choose_back", help="exact round trip for the choose arm")
    p.add_argument("--out_dir", required=True); p.add_argument("--candidates", choices=["dinoland", "truth"], required=True)
    p.add_argument("--model", default="sonnet"); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("score_choose_back"); p.add_argument("--out_dir", required=True)
    p.add_argument("--candidates", choices=["dinoland", "truth"], required=True)
    p = sp.add_parser("score"); p.add_argument("--out_dir", required=True)
    p.add_argument("--cycle_tol", type=float, default=5.0, help="round-trip tolerance, %% of wing length")
    p.add_argument("--agree_tol", type=float, default=5.0, help="Claude-DINOLand agreement, %% of wing length")
    a = ap.parse_args()
    {"prepare": prepare, "claude": run_claude, "dinoland": dinoland, "score": score,
     "choose": run_choose, "score_choose": score_choose,
     "choose_back": run_choose_back, "score_choose_back": score_choose_back}[a.stage](a)


if __name__ == "__main__":
    main()
