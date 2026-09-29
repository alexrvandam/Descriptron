#!/usr/bin/env python3
"""
Segmentation benchmark: SAM2-PAL (Descriptron 2.1.1, sam2_pal_batch_v21) against SST, the official code of the
method it builds on (Feng et al. 2025, "Static segmentation by tracking", github.com/Imageomics/SST), on
hand-annotated Diaphorina structures, scored by IoU against the repaired annotations.

Like for like (one annotated template each):
  sst_oneshot      SST segment.py pipeline as distributed (support frame + all queries as ONE video, no
                   training), SST's own functions, run per structure
  sst_perimage     SST one-shot with one short video per test image (support + that image), all structures
  pal_zeroshot     SAM2-PAL v21, one template, no fine-tuning
  pal_ft1          SAM2-PAL v21, one template, palindrome fine-tuning (the SOP recipe)
  pal_ft1_orient   pal_ft1's checkpoint, predicting with SAM2-PAL's built-in --orientation_search rot4
                   (still one template; a documented feature of the released tool)
What the SAM2-PAL workflow adds:
  pal_ft5          five templates (equivalently: one template plus four corrected predictions)
  pal_ft5_orient   pal_ft5's checkpoint with --orientation_search rot4

Stages:
  prepare    split, templates, test-image folders, run scripts          (any env with numpy/PIL/cv2)
  sst        run SST one-shot per structure                              (biorag python + SST on PYTHONPATH)
  score      IoU per structure, per image; summary table                 (measure_env)

  python segmentation_benchmark_v1.py prepare --coco <unified.json> --image_dir <images> --out_dir <out>
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

GROUPS = {"forewing": {"match": "_forewing", "n_test": None},
          "rostrum": {"match": "_rostrum", "n_test": 40}}
N_TEMPLATES = 5


def species_of(fn):
    return fn.split("_ch00")[0].rsplit("_", 2)[0]


def ann_mask(ann, h, w):
    import cv2
    m = np.zeros((h, w), np.uint8)
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        for poly in seg:
            pts = np.array(poly, float).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    elif isinstance(seg, dict):
        from pycocotools import mask as mu
        rle = seg if isinstance(seg.get("counts"), str) else mu.frPyObjects(seg, h, w)
        m = mu.decode(rle).astype(np.uint8)
    return m.astype(bool)


def prepare(a):
    coco = json.load(open(a.coco))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    split = {}
    for g, spec in GROUPS.items():
        excl = [e for e in (a.exclude or "").split(",") if e]
        imgs = [i for i in coco["images"] if spec["match"] in i["file_name"]
                and (Path(a.image_dir) / i["file_name"]).exists()
                and not any(e in i["file_name"] for e in excl)]
        ids = {i["id"] for i in imgs}
        anns = [x for x in coco["annotations"] if x["image_id"] in ids]
        per_img = {i["id"]: {x["category_id"] for x in anns if x["image_id"] == i["id"]} for i in imgs}
        # structures annotated on at least 80% of the group's images (rare extras would decide the templates)
        cnt = {c: sum(c in v for v in per_img.values()) for c in {x["category_id"] for x in anns}}
        cat_ids = sorted(c for c, n in cnt.items() if n >= 0.8 * len(imgs))
        full = [i for i in imgs if per_img[i["id"]] >= set(cat_ids)]     # templates must carry every structure
        rng = random.Random(a.seed)
        rng.shuffle(full)
        templates, seen = [], set()
        for i in full:                                                  # distinct species first
            if species_of(i["file_name"]) not in seen:
                templates.append(i); seen.add(species_of(i["file_name"]))
            if len(templates) == N_TEMPLATES:
                break
        tnames = {t["file_name"] for t in templates}
        test = [i for i in imgs if i["file_name"] not in tnames]
        rng.shuffle(test)
        if spec["n_test"]:
            test = test[: spec["n_test"]]
        gdir = out / g
        for sub in ("test_images", "templates"):
            (gdir / sub).mkdir(parents=True, exist_ok=True)
        for i in test:
            dst = gdir / "test_images" / i["file_name"]
            if not dst.exists():
                dst.write_bytes((Path(a.image_dir) / i["file_name"]).read_bytes())
        for t in templates:
            dst = gdir / "templates" / t["file_name"]
            if not dst.exists():
                dst.write_bytes((Path(a.image_dir) / t["file_name"]).read_bytes())
        catlist = [{"id": c, "name": cats[c]} for c in cat_ids]

        def sub_coco(images):
            iid = {i["id"] for i in images}
            return {"images": images, "categories": catlist,
                    "annotations": [x for x in anns if x["image_id"] in iid and x["category_id"] in cat_ids]}
        json.dump(sub_coco(templates[:1]), open(gdir / "templates_1.json", "w"))
        json.dump(sub_coco(templates), open(gdir / f"templates_{N_TEMPLATES}.json", "w"))
        json.dump(sub_coco(test), open(gdir / "test_gt.json", "w"))
        split[g] = {"categories": [cats[c] for c in cat_ids], "templates": [t["file_name"] for t in templates],
                    "test": [i["file_name"] for i in test]}
        print(f"{g}: {len(imgs)} images; templates {len(templates)} "
              f"({', '.join(species_of(t['file_name']) for t in templates)}); test {len(test)}; "
              f"structures {len(cat_ids)}")
    json.dump(split, open(out / "split.json", "w"), indent=1)
    write_pal_scripts(a, out, split)


def write_pal_scripts(a, out, split):
    common = (f'--sam2_checkpoint "{a.sam2_checkpoint}" --sam2_config {a.sam2_config} '
              f'--num_points 30 --chunk_size 1 --interleave_template --save_masks')
    ft = '--pal_finetuning --num_epochs 40 --learning_rate 1e-5 --max_images_per_epoch 60 --cycle_consistency'
    lines = ["#!/bin/bash", "# SAM2-PAL arms (SOP recipe); one log per run", "set -u"]
    for g, s in split.items():
        gd = out / g
        t1 = gd / "templates" / s["templates"][0]
        arms = {
            "pal_zeroshot": f'--template_json "{gd}/templates_1.json"',
            "pal_ft1": f'--template_json "{gd}/templates_1.json" --training_json "{gd}/templates_1.json" '
                       f'--training_images_dir "{gd}/templates" {ft}',
            f"pal_ft{N_TEMPLATES}": f'--template_json "{gd}/templates_{N_TEMPLATES}.json" '
                                    f'--training_json "{gd}/templates_{N_TEMPLATES}.json" '
                                    f'--training_images_dir "{gd}/templates" {ft}',
        }
        for arm, extra in arms.items():
            o = gd / arm
            lines.append(f'echo "{g} {arm} start $(date +%T)" >> "{out}/pal_status.txt"')
            lines.append(f'"{a.python}" "{a.sam2pal}" --template_image "{t1}" {extra} --image_dir "{gd}/test_images" '
                         f'--output_dir "{o}" {common} > "{gd}/log_{arm}.txt" 2>&1 '
                         f'|| echo "{g} {arm} FAILED" >> "{out}/pal_status.txt"')
    lines.append(f'echo ALL_DONE >> "{out}/pal_status.txt"')
    (out / "run_sam2pal.sh").write_text("\n".join(lines) + "\n")
    print("wrote run_sam2pal.sh")


def run_sst(a):
    """SST one-shot, exactly as segment.py does it, but saving the raw masks: one video per structure
    (support frame first, then every test image), support mask = that structure on template 1."""
    import cv2
    import sam_utils                                        # from SST's src/sst on PYTHONPATH
    out = Path(a.out_dir); split = json.load(open(out / "split.json"))
    pred = sam_utils.build_sam2_predictor(checkpoint=a.sam2_checkpoint, model_cfg="sam2_hiera_l")
    for g, s in split.items():
        gd = out / g; od = gd / "sst_oneshot" / "masks"; od.mkdir(parents=True, exist_ok=True)
        tj = json.load(open(gd / "templates_1.json"))
        t = tj["images"][0]; cid = {c["name"]: c["id"] for c in tj["categories"]}
        sup = cv2.imread(str(gd / "templates" / t["file_name"]))[..., ::-1]
        queries = [cv2.imread(str(gd / "test_images" / f))[..., ::-1] for f in s["test"]]
        for cname in s["categories"]:
            anns = [x for x in tj["annotations"] if x["category_id"] == cid[cname]]
            m = np.zeros(sup.shape[:2], bool)
            for x in anns:
                m |= ann_mask(x, t["height"], t["width"])
            state = sam_utils.load_masks(pred, [q.copy() for q in queries], sup, [m])
            frames = sam_utils.propagate_masks(pred, state)
            for f, q, fr in zip(s["test"], queries, frames[1:]):          # frame 0 is the support image
                mk = fr["segmentation"][0].astype(np.uint8) if len(fr["segmentation"]) else np.zeros((1024, 1024), np.uint8)
                mk = cv2.resize(mk, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(od / f"{Path(f).stem}__{cname}.png"), mk * 255)
            print(f"{g} {cname}: {len(queries)} masks", flush=True)
            del state
            import torch; torch.cuda.empty_cache()


def run_sst_perimage(a):
    """SST one-shot, one short video per test image (support frame + that image only), so the tracker cannot
    lose the object along a long sequence; SST's own functions, same support masks as run_sst."""
    import cv2
    import torch
    import sam_utils
    out = Path(a.out_dir); split = json.load(open(out / "split.json"))
    pred = sam_utils.build_sam2_predictor(checkpoint=a.sam2_checkpoint, model_cfg="sam2_hiera_l")
    for g, s in split.items():
        gd = out / g; od = gd / "sst_perimage" / "masks"; od.mkdir(parents=True, exist_ok=True)
        tj = json.load(open(gd / "templates_1.json"))
        t = tj["images"][0]; cid = {c["name"]: c["id"] for c in tj["categories"]}
        sup = cv2.imread(str(gd / "templates" / t["file_name"]))[..., ::-1]
        sup_masks = []
        for cname in s["categories"]:
            m = np.zeros(sup.shape[:2], bool)
            for x in tj["annotations"]:
                if x["category_id"] == cid[cname]:
                    m |= ann_mask(x, t["height"], t["width"])
            sup_masks.append(m)
        for f in s["test"]:
            q = cv2.imread(str(gd / "test_images" / f))[..., ::-1]
            state = sam_utils.load_masks(pred, [q.copy()], sup, sup_masks)       # all structures, one image
            frames = sam_utils.propagate_masks(pred, state)
            fr = frames[1]
            ids = list(fr["obj_ids"])
            for k, cname in enumerate(s["categories"]):
                mk = fr["segmentation"][ids.index(k)].astype(np.uint8) if k in ids else np.zeros((1024, 1024), np.uint8)
                mk = cv2.resize(mk, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(od / f"{Path(f).stem}__{cname}.png"), mk * 255)
            del state; torch.cuda.empty_cache()
        print(f"{g}: {len(s['test'])} images done", flush=True)


def _pal_masks(pred_json, test_names, cats):
    pj = json.load(open(pred_json))
    names = {i["id"]: Path(i["file_name"]).name for i in pj["images"]}
    cname = {c["id"]: c["name"] for c in pj.get("categories", [])}
    out = {}
    for x in pj["annotations"]:
        nm = names.get(x["image_id"]); cn = cname.get(x["category_id"], x.get("category_name"))
        if nm in test_names and cn in cats:
            out.setdefault((nm, cn), []).append(x)
    return out, {Path(i["file_name"]).name: i for i in pj["images"]}


def score(a):
    import cv2
    import pandas as pd
    out = Path(a.out_dir); split = json.load(open(out / "split.json"))
    rows = []
    for g, s in split.items():
        gd = out / g; gt = json.load(open(gd / "test_gt.json"))
        gcat = {c["id"]: c["name"] for c in gt["categories"]}
        gimg = {i["id"]: i for i in gt["images"]}
        truth = {}
        for x in gt["annotations"]:
            i = gimg[x["image_id"]]; k = (i["file_name"], gcat[x["category_id"]])
            truth[k] = truth.get(k, np.zeros((i["height"], i["width"]), bool)) | ann_mask(x, i["height"], i["width"])
        arms = {}
        for arm in ("sst_oneshot", "sst_perimage"):
            if (gd / arm / "masks").exists():
                arms[arm] = ("png", gd / arm / "masks")
        for arm in ("pal_zeroshot", "pal_ft1", f"pal_ft{N_TEMPLATES}", "pal_ft1_orient", f"pal_ft{N_TEMPLATES}_orient"):
            pj = gd / arm / "pal_predictions.json"
            if pj.exists():
                arms[arm] = ("coco", pj)
        for arm, (kind, src) in arms.items():
            if kind == "coco":
                pm, pimg = _pal_masks(src, set(s["test"]), set(s["categories"]))
            for (fn, cn), tm in truth.items():
                h, w = tm.shape
                if kind == "png":
                    p = cv2.imread(str(src / f"{Path(fn).stem}__{cn}.png"), cv2.IMREAD_GRAYSCALE)
                    pr = (p > 127) if p is not None else np.zeros_like(tm)
                else:
                    pr = np.zeros_like(tm)
                    for x in pm.get((fn, cn), []):
                        pr |= ann_mask(x, h, w)
                inter = np.logical_and(pr, tm).sum(); union = np.logical_or(pr, tm).sum()
                rows.append(dict(group=g, arm=arm, image=fn, structure=cn,
                                 iou=float(inter / union) if union else np.nan))
    df = pd.DataFrame(rows); df.to_csv(out / "segmentation_benchmark_per_mask.csv", index=False)
    summ = (df.groupby(["group", "arm"])["iou"]
            .agg(n="count", mean="mean", median="median", over_0_7=lambda v: float((v > 0.7).mean()))
            .round(3).reset_index())
    summ.to_csv(out / "segmentation_benchmark_summary.csv", index=False)
    per = df.pivot_table(index=["group", "structure"], columns="arm", values="iou", aggfunc="mean").round(3)
    per.to_csv(out / "segmentation_benchmark_per_structure.csv")
    print(summ.to_string(index=False)); print(); print(per.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="stage", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("--coco", required=True); p.add_argument("--image_dir", required=True)
    p.add_argument("--out_dir", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", default="", help="comma-separated file-name fragments to leave out "
                   "(e.g. an image whose content does not match its body part)")
    p.add_argument("--sam2pal", required=True); p.add_argument("--python", required=True)
    p.add_argument("--sam2_checkpoint", required=True); p.add_argument("--sam2_config", default="sam2_hiera_l.yaml")
    p = sp.add_parser("sst"); p.add_argument("--out_dir", required=True); p.add_argument("--sam2_checkpoint", required=True)
    p = sp.add_parser("sst_perimage"); p.add_argument("--out_dir", required=True)
    p.add_argument("--sam2_checkpoint", required=True)
    p = sp.add_parser("score"); p.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    {"prepare": prepare, "sst": run_sst, "sst_perimage": run_sst_perimage, "score": score}[a.stage](a)


if __name__ == "__main__":
    main()
