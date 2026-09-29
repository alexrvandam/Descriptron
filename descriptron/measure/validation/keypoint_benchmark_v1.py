#!/usr/bin/env python3
"""
Keypoint benchmark: DINOLand vs a trained keypoint detector (torchvision Keypoint R-CNN) vs Claude, on the SAME
held-out Diaphorina forewings, scored against hand-placed landmarks as % of wing length (largest landmark span).

Every wing is brought to one orientation first (mirror images flipped, upside-down ones turned) and cropped
to its landmarks, exactly as in claude_landmark_grounding_test_v1.py, so all methods see identical images. The
test wings are the 20 targets of that grounding test (read from its design.json), so Claude's existing answers
are scored on the same wings. Training / reference sets are nested: k = 1 (the grounding test's reference, the
wing closest to the mean shape), 5, 20 and all remaining wings.

Stages:
  prepare   crops + COCO for every wing, test/train split, nested training sets     (measure_env)
            --raw: wings as photographed (whole image, no orientation fix, no crop) for the realistic test
  dinoland  write run_dinoland_k<k>.sh per k (run them in the biorag env)
  kprcnn    write run_kprcnn_k<k>.sh per k (train + predict with torchvision_det; GPU, measure_env/samm)
  score     one table: method x k -> median error, share within 5% and 10%, failing wings

  python keypoint_benchmark_v1.py prepare --coco <forewing_keypoints.json> --image_dir <images> \
      --grounding_dir <claude_grounding_test/run1_...> --out_dir <out> --exclude sp4_5
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import claude_landmark_grounding_test_v1 as G  # noqa: E402  (same orientation fix and crop)

KS = (1, 5, 20, "all")


def prepare(a):
    coco = json.load(open(a.coco))
    imgs = {i["id"]: i for i in coco["images"]}
    excl = [e for e in (a.exclude or "").split(",") if e]
    wings = []
    for ann in coco["annotations"]:
        fn = imgs[ann["image_id"]]["file_name"]
        if any(e in fn for e in excl) or not (Path(a.image_dir) / fn).exists():
            continue
        k = np.array(ann["keypoints"], float).reshape(-1, 3)
        if (k[:, 2] == 0).any():
            continue
        wings.append((fn, k[:, :2]))
    hs = [G.handed(p) for _, p in wings]
    maj = 1 if sum(hs) >= 0 else -1
    out = Path(a.out_dir); (out / "crops").mkdir(parents=True, exist_ok=True)
    cat = dict(coco["categories"][0]); cat["id"] = 1
    images, anns, meta = [], [], {}
    for n, ((fn, p), h) in enumerate(zip(wings, hs), 1):
        img = Image.open(Path(a.image_dir) / fn).convert("RGB")
        if a.raw:           # as photographed: whole image, no orientation fix, no crop to the landmarks
            s_ = a.raw_long_side / max(img.size)
            c = img.resize((round(img.size[0] * s_), round(img.size[1] * s_)), Image.LANCZOS)
            q = p * s_
        else:
            flip = h != maj
            _, p1 = G.canonical(img, p, flip, False)
            rot = p1[0, 0] < p1[:, 0].mean()
            img2, p2 = G.canonical(img, p, flip, rot)
            c, q = G.crop_resize(img2, p2)
        name = Path(fn).stem + ".png"
        c.save(out / "crops" / name)
        images.append({"id": n, "file_name": name, "width": c.size[0], "height": c.size[1]})
        kp = []
        for x, y in q:
            kp += [float(x), float(y), 2]
        x0, y0 = q.min(0); x1, y1 = q.max(0)
        anns.append({"id": n, "image_id": n, "category_id": 1, "keypoints": kp, "num_keypoints": len(q),
                     "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], "iscrowd": 0,
                     "area": float((x1 - x0) * (y1 - y0))})
        meta[name] = dict(id=n, file=fn, species=G.species_of(fn), wing_length=G.wing_length(q))
    all_coco = {"images": images, "annotations": anns, "categories": [cat]}
    json.dump(all_coco, open(out / "all_wings.json", "w"))
    gd = json.load(open(Path(a.grounding_dir) / "design.json"))
    test = [t["crop"] for t in gd["targets"]]
    ref = gd["reference"]["crop"]
    missing = [t for t in test + [ref] if t not in meta]
    if missing:
        raise SystemExit(f"grounding-test wings not found here: {missing}")
    pool = [nm for nm in meta if nm not in test and nm != ref]
    random.Random(a.seed).shuffle(pool)
    order = [ref] + pool                               # nested: every larger set contains the smaller
    split = {"test": test, "train_order": order,
             "k": {str(k): order[: (len(order) if k == "all" else k)] for k in KS}}
    json.dump(split, open(out / "split.json", "w"), indent=1)
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    # one single-image COCO per training wing (DINOLand references)
    (out / "refs").mkdir(exist_ok=True)
    byname = {i["file_name"]: i for i in images}
    for nm in order:
        i = byname[nm]; an = next(x for x in anns if x["image_id"] == i["id"])
        json.dump({"images": [dict(i, id=1)], "annotations": [dict(an, id=1, image_id=1)], "categories": [cat]},
                  open(out / "refs" / (Path(nm).stem + "_keypoints.json"), "w"))
    # the test images alone, for batch prediction
    (out / "test_crops").mkdir(exist_ok=True)
    for nm in test:
        (out / "test_crops" / nm).write_bytes((out / "crops" / nm).read_bytes())
    print(f"{len(images)} wings; test {len(test)}; training pool {len(order)} "
          f"(k = {', '.join(str(len(v)) for v in split['k'].values())}); reference {ref}")


def dinoland(a):
    out = Path(a.out_dir); split = json.load(open(out / "split.json"))
    for k, names in split["k"].items():
        refs = ",".join(str(out / "refs" / (Path(n).stem + "_keypoints.json")) for n in names)
        cmd = (f'"{a.python}" "{a.dinoland}" --imgA "{out}/crops/{names[0]}" --landmarks "{refs}" '
               f'--ref_dir "{out}/crops" --batch_glob "{out}/test_crops/*.png" --batch_n 1000 '
               f'--align feature --outdir "{out}/dinoland_k{k}"'
               + (" --orientation_search rot4 --mirror_refs" if a.orient else ""))
        (out / f"run_dinoland_k{k}.sh").write_text("#!/bin/bash\nset -e\n" + cmd + "\n")
    print("wrote", ", ".join(f"run_dinoland_k{k}.sh" for k in split["k"]))


def kprcnn(a):
    out = Path(a.out_dir); split = json.load(open(out / "split.json")); meta = json.load(open(out / "meta.json"))
    test_ids = [meta[n]["id"] for n in split["test"]]
    for k, names in split["k"].items():
        ids = [meta[n]["id"] for n in names]
        (out / f"train_ids_k{k}.json").write_text(json.dumps(ids))
        (out / "test_ids.json").write_text(json.dumps(test_ids))
        d = out / f"kprcnn_k{k}"
        cmd = (f'"{a.python}" "{a.tvdir}/tv_train_v1.py" --task keypoints --coco-json "{out}/all_wings.json" '
               f'--img-dir "{out}/crops" --output-dir "{d}" --train-ids "{out}/train_ids_k{k}.json" '
               f'--total-iters {a.iters} --checkpoint-period {a.iters} --seed 0\n'
               f'"{a.python}" "{a.tvdir}/tv_predict_v1.py" --checkpoint "{d}/model_final_your_taxon.pth" '
               f'--coco-json "{out}/all_wings.json" --img-dir "{out}/crops" --image_ids "{out}/test_ids.json" '
               f'--annotations_out "{d}/predictions_keypoints.json" --score_threshold 0.05')
        (out / f"run_kprcnn_k{k}.sh").write_text("#!/bin/bash\nset -e\n" + cmd + "\n")
    print("wrote", ", ".join(f"run_kprcnn_k{k}.sh" for k in split["k"]))


def _errors_from_coco(pred_path, meta, test, truth, by_name=True):
    """One prediction per test wing (highest-scoring if several); returns {wing: [err% x 17]} with
    missing wings / points counted as a full wing length (100%)."""
    res = {}
    if not Path(pred_path).exists():
        return None
    pj = json.load(open(pred_path))
    names = {i["id"]: Path(i["file_name"]).name for i in pj.get("images", [])}
    best = {}
    for an in pj["annotations"]:
        nm = names.get(an["image_id"]) if names else None
        if nm is None:
            nm = next((n for n in test if meta[n]["id"] == an["image_id"]), None)
        if nm not in test or "keypoints" not in an:
            continue
        s = an.get("score", 1.0)
        if nm not in best or s > best[nm][0]:
            best[nm] = (s, np.array(an["keypoints"], float).reshape(-1, 3))
    for nm in test:
        g = truth[nm]; L = meta[nm]["wing_length"]
        if nm not in best:
            res[nm] = [100.0] * len(g); continue
        k = best[nm][1]
        res[nm] = [100 * np.hypot(*(k[j, :2] - g[j])) / L if k[j, 2] > 0 else 100.0 for j in range(len(g))]
    return res


def score(a):
    import pandas as pd
    out = Path(a.out_dir); split = json.load(open(out / "split.json")); meta = json.load(open(out / "meta.json"))
    allc = json.load(open(out / "all_wings.json")); test = split["test"]
    idname = {i["id"]: i["file_name"] for i in allc["images"]}
    truth = {idname[an["image_id"]]: np.array(an["keypoints"], float).reshape(-1, 3)[:, :2]
             for an in allc["annotations"] if idname[an["image_id"]] in test}
    rows = []
    for method, pat in (("DINOLand", "dinoland_k{}/predictions_keypoints.json"),
                        ("Keypoint R-CNN", "kprcnn_k{}/predictions_keypoints.json")):
        for k in split["k"]:
            e = _errors_from_coco(out / pat.format(k), meta, test, truth)
            if e is None:
                continue
            E = np.concatenate([np.array(v) for v in e.values()])
            E4 = np.concatenate([np.array(v)[[l - 1 for l in a.claude_landmarks]] for v in e.values()])
            fails = sum(np.median(v) > 20 for v in e.values())
            rows.append(dict(method=method, k=k, n_train=len(split["k"][k]), median_pct=round(float(np.median(E)), 2),
                             within_5=round(float(np.mean(E <= 5)), 3), within_10=round(float(np.mean(E <= 10)), 3),
                             failing_wings=int(fails), median_pct_4lm=round(float(np.median(E4)), 2),
                             within_5_4lm=round(float(np.mean(E4 <= 5)), 3)))
    # Claude, from the grounding test (4 landmarks, one reference = k 1)
    gdir = Path(a.grounding_dir)
    if (gdir / "per_landmark_errors.csv").exists():
        c = pd.read_csv(gdir / "per_landmark_errors.csv")
        for col, lab in (("claude_err", "Claude, pointing"),):
            e = c[col].dropna().values
            rows.append(dict(method=lab, k="1", n_train=1, median_pct_4lm=round(float(np.median(e)), 2),
                             within_5_4lm=round(float(np.mean(e <= 5)), 3)))
    if (gdir / "choose_per_landmark.csv").exists():
        c = pd.read_csv(gdir / "choose_per_landmark.csv")
        g = c[c.candidates == "dinoland"]["err_pct"].dropna().values
        if len(g):
            rows.append(dict(method="Claude, choosing among DINOLand candidates", k="1", n_train=1,
                             median_pct_4lm=round(float(np.median(g)), 2), within_5_4lm=round(float(np.mean(g <= 5)), 3)))
    df = pd.DataFrame(rows); df.to_csv(out / "keypoint_benchmark_summary.csv", index=False)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="stage", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("--coco", required=True); p.add_argument("--image_dir", required=True)
    p.add_argument("--grounding_dir", required=True); p.add_argument("--out_dir", required=True)
    p.add_argument("--exclude", default=""); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw", action="store_true", help="wings as photographed: no orientation fix, no crop")
    p.add_argument("--raw_long_side", type=int, default=1024)
    p = sp.add_parser("dinoland"); p.add_argument("--out_dir", required=True)
    p.add_argument("--orient", action="store_true", help="add --orientation_search rot4 --mirror_refs")
    p.add_argument("--dinoland", required=True); p.add_argument("--python", required=True)
    p = sp.add_parser("kprcnn"); p.add_argument("--out_dir", required=True)
    p.add_argument("--tvdir", required=True); p.add_argument("--python", required=True)
    p.add_argument("--iters", type=int, default=2000)
    p = sp.add_parser("score"); p.add_argument("--out_dir", required=True); p.add_argument("--grounding_dir", required=True)
    p.add_argument("--claude_landmarks", default="3,6,9,13",
                   type=lambda s: [int(x) for x in s.split(",")])
    a = ap.parse_args()
    {"prepare": prepare, "dinoland": dinoland, "kprcnn": kprcnn, "score": score}[a.stage](a)


if __name__ == "__main__":
    main()
