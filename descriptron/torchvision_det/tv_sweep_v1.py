#!/usr/bin/env python3
"""
tv_sweep_v1.py — does torchvision match Detectron2 on held-out species?
=======================================================================

Grouped k-fold over species: every fold trains on some species and is scored on
species it has never seen, which is the question a taxonomist actually asks — a
new species arrives, are its sclerites segmented? Categories are anatomical
structures, so the label set is constant across folds and only the specimens change.

Three things make the comparison fair, and each one had to be deliberate:

1. **The recipe is shared.** Both arms use the Detectron2 schedule (batch 2,
   lr 5e-4, warmup 8.5%, steps 40/80%, wd 1e-4, clip 1.0, 128 RoIs, the same
   augmentations, no flip). `tv_train_v1.RECIPE` is the single copy of it.
2. **The budget is equal.** Fixed iterations, no early stopping. Keep
   `--iters` at or below 5000 and the Detectron2 script's own EarlyStoppingHook
   (patience 5, evaluated every 1000) cannot fire, so its original code needs no
   modification.
3. **One evaluator.** Both arms emit COCO detections and both are scored by
   `tv_predict_v1.evaluate`. Note that the Detectron2 training script evaluates
   on a *copy of its training set* (it writes the same JSON to train.json and
   val.json), so its own reported AP is in-sample; the held-out score here is
   computed from outside and is the one to quote.

    python tv_sweep_v1.py \
        --coco_json unified.json --img_dir images/ --group_labels group_labels.csv \
        --output_dir sweep_out/ --folds 3 --iters 3000

Resumable: a run whose metrics file already exists is skipped, so a GPU hang
costs one run rather than the sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tv_predict_v1 import evaluate                       # noqa: E402

# What each arm IS. Stated explicitly because a comparison is only as trustworthy as
# the guarantee that the arms differ in exactly one declared way.
TORCHVISION_ARMS = {
    "tv_v1":        {"arch": "v1", "mosaic": False},
    "tv_v2":        {"arch": "v2", "mosaic": False},
    "tv_v1_mosaic": {"arch": "v1", "mosaic": True},
    "tv_v2_mosaic": {"arch": "v2", "mosaic": True},
}

HERE = Path(__file__).resolve().parent
DEFAULT_D2_TRAIN = (HERE.parent / "detectron2" /
                    "detectron2_training_and_filterV10_and_kpts-17.py")


def species_of_images(coco, group_labels):
    """filename -> species, then image id -> species. Unlabelled images are dropped."""
    lab = {}
    with open(group_labels) as fh:
        for row in csv.DictReader(fh):
            keys = list(row)
            lab[Path(row[keys[0]]).name.strip()] = str(row[keys[1]]).strip()
    out = {}
    for im in coco["images"]:
        sp = lab.get(Path(im["file_name"]).name.strip())
        if sp:
            out[im["id"]] = sp
    return out


def grouped_folds(id_to_species, k, seed=0):
    """
    k folds, whole species in one fold, balanced by image count. Greedy
    largest-first assignment keeps the folds even without shuffling species
    across them, which would leak.
    """
    import random
    by_species = defaultdict(list)
    for image_id, sp in id_to_species.items():
        by_species[sp].append(image_id)
    species = sorted(by_species, key=lambda s: (-len(by_species[s]), s))
    rng = random.Random(seed)
    rng.shuffle(species)
    species.sort(key=lambda s: -len(by_species[s]))
    folds = [[] for _ in range(k)]
    sizes = [0] * k
    fold_species = [[] for _ in range(k)]
    for sp in species:
        j = sizes.index(min(sizes))
        folds[j].extend(by_species[sp])
        fold_species[j].append(sp)
        sizes[j] += len(by_species[sp])
    return folds, fold_species


def write_subset(coco, image_ids, path):
    """A COCO file holding only these images — what the Detectron2 script needs."""
    keep = set(image_ids)
    sub = {"images": [im for im in coco["images"] if im["id"] in keep],
           "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
           "categories": coco["categories"]}
    Path(path).write_text(json.dumps(sub))
    return sub


def run(cmd, log_path):
    """Run a training/prediction subprocess, tee-ing to a log so a crash is diagnosable."""
    print("   $ " + " ".join(str(c) for c in cmd[:6]) + " …", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.run([str(c) for c in cmd], stdout=log,
                              stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"   ! exit {proc.returncode}; see {log_path}")
    return proc.returncode


def arm_torchvision(a, fold, train_ids, test_ids, out, mosaic_p, arch):
    ids_dir = out / "ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    (ids_dir / f"train_fold{fold}.json").write_text(json.dumps(train_ids))
    (ids_dir / f"test_fold{fold}.json").write_text(json.dumps(test_ids))
    ck = out / f"model_final_fold{fold}.pth"
    if not ck.exists():
        rc = run([a.python, HERE / "tv_train_v1.py", "--task", "masks", "--arch", arch,
                  "--coco-json", a.coco_json, "--img-dir", a.img_dir,
                  "--output-dir", out, "--dataset-name", f"fold{fold}",
                  "--total-iters", a.iters, "--checkpoint-period", 0,
                  "--train-ids", ids_dir / f"train_fold{fold}.json",
                  "--mosaic_p", mosaic_p, "--mosaic-size", a.mosaic_size,
                  "--num-workers", a.num_workers, "--seed", a.seed,
                  "--log-period", 200], out / f"train_fold{fold}.log")
        if rc != 0:
            return None
    det = out / f"detections_fold{fold}.json"
    rc = run([a.python, HERE / "tv_predict_v1.py", "--checkpoint", ck,
              "--coco-json", a.coco_json, "--img-dir", a.img_dir,
              "--image_ids", ids_dir / f"test_fold{fold}.json", "--detections_out", det,
              "--score_threshold", a.score_threshold,
              "--num-workers", a.num_workers], out / f"predict_fold{fold}.log")
    return det if rc == 0 and det.exists() else None


def stage_images_for_d2(a, coco, out_root):
    """
    The Detectron2 training script creates train/ and val/ INSIDE --img-dir and
    copies every image into both. Pointed at the master image folder that writes
    two full copies of the dataset next to the originals, so it is given a staging
    directory of its own instead: one copy here, the script's two copies inside
    it, and the whole thing is deletable afterwards. Prediction still reads the
    master folder — only training needs the staging copy.
    """
    import shutil
    stage = Path(a.d2_img_dir) if a.d2_img_dir else out_root / "d2_stage" / "images"
    stage.mkdir(parents=True, exist_ok=True)
    src_dir = Path(a.img_dir)
    wanted = {Path(im["file_name"]).name for im in coco["images"]}
    copied = 0
    for name in sorted(wanted):
        dst = stage / name
        if dst.exists():
            continue
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"[sweep] detectron2 staging dir {stage} ({copied} copied, {len(wanted)} wanted)")
    return str(stage)


def arm_detectron2(a, fold, train_ids, test_ids, out, coco, img_dir):
    ids_dir = out / "ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    (ids_dir / f"test_fold{fold}.json").write_text(json.dumps(test_ids))
    train_json = out / f"train_fold{fold}.json"
    write_subset(coco, train_ids, train_json)
    names = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    (out / "thing_classes.json").write_text(json.dumps(names))

    cfg = out / f"config_fold{fold}.yaml"
    weights = out / f"model_final_fold{fold}.pth"
    if not (cfg.exists() and weights.exists()):
        rc = run([a.d2_python, a.d2_train_script,
                  "--coco-json", train_json, "--img-dir", img_dir,
                  "--output-dir", out, "--dataset-name", f"fold{fold}",
                  "--total-iters", a.iters, "--checkpoint-period", 10 ** 9,
                  "--train-segmentation-only"], out / f"train_fold{fold}.log")
        if rc != 0:
            return None
        # the script names its outputs after the dataset; find them
        for c in sorted(out.rglob("config_fold%d*.yaml" % fold)):
            cfg = c
            break
        # prefer model_final.pth: DefaultTrainer writes it in checkpointer format,
        # whereas model_final_<name>.pth is a bare state_dict that
        # DetectionCheckpointer cannot load (it pops a "model" key that is absent)
        found = (sorted(out.rglob("model_final.pth"))
                 or sorted(out.rglob("model_final*fold%d*.pth" % fold))
                 or sorted(out.rglob("model_final*.pth")))
        if found:
            weights = found[0]
    if not (Path(cfg).exists() and Path(weights).exists()):
        print(f"   ! detectron2 fold {fold}: no config/weights found under {out}")
        return None
    det = out / f"detections_fold{fold}.json"
    rc = run([a.d2_python, HERE / "d2_predict_fold_v1.py",
              "--config_file", cfg, "--model_weights", weights,
              "--coco_json", a.coco_json, "--img_dir", a.img_dir,
              "--image_ids", ids_dir / f"test_fold{fold}.json", "--detections_out", det,
              "--score_threshold", a.score_threshold,
              "--thing_classes", out / "thing_classes.json"],
             out / f"predict_fold{fold}.log")
    return det if rc == 0 and det.exists() else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco_json", required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--group_labels", required=True,
                   help="CSV: filename,group_label — the species each image belongs to")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--iters", type=int, default=3000,
                   help="identical for every arm; keep <=5000 so the Detectron2 "
                        "script's early stopping (patience 5, every 1000) cannot fire")
    p.add_argument("--arms", default="tv_v2,tv_v2_mosaic,d2",
                   help="any of: d2, " + ", ".join(sorted(TORCHVISION_ARMS)))
    p.add_argument("--mosaic_p", type=float, default=0.5)
    p.add_argument("--mosaic-size", type=int, default=512)
    p.add_argument("--score_threshold", type=float, default=0.05,
                   help="low on purpose: COCO AP integrates over the whole curve")
    p.add_argument("--python", default=sys.executable, help="interpreter for the torchvision arms")
    p.add_argument("--d2_python", default="/home/localuser/Desktop/Descriptron/conda/envs/"
                                          "detectron2_env/bin/python")
    p.add_argument("--d2_train_script", default=str(DEFAULT_D2_TRAIN))
    p.add_argument("--d2_img_dir", default=None,
                   help="staging directory for the Detectron2 arm (default: "
                        "<output_dir>/d2_stage/images). It must not be the master image "
                        "folder: that script writes two full copies of the dataset into "
                        "--img-dir as train/ and val/")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry_run", action="store_true", help="print the plan and stop")
    a = p.parse_args()

    out_root = Path(a.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    coco = json.loads(Path(a.coco_json).read_text())
    id_to_species = species_of_images(coco, a.group_labels)
    folds, fold_species = grouped_folds(id_to_species, a.folds, a.seed)
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    unknown = [x for x in arms if x != "d2" and x not in TORCHVISION_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from "
                         f"{['d2'] + sorted(TORCHVISION_ARMS)}")

    plan = {"folds": a.folds, "iters": a.iters, "arms": arms,
            "images_with_species": len(id_to_species),
            "fold_sizes": [len(f) for f in folds],
            "fold_species": fold_species}
    (out_root / "sweep_plan.json").write_text(json.dumps(plan, indent=2))
    print(f"[sweep] {len(id_to_species)} labelled images, {len(set(id_to_species.values()))} species")
    for i, (f, sp) in enumerate(zip(folds, fold_species)):
        print(f"  fold {i}: {len(f):4d} images, {len(sp):2d} species — {', '.join(sorted(sp))}")
    print(f"[sweep] arms={arms} iters={a.iters} -> {len(arms)*a.folds} runs")
    if a.dry_run:
        return

    d2_img_dir = (stage_images_for_d2(a, coco, out_root) if "d2" in arms else a.img_dir)
    rows, t0 = [], time.time()
    for arm in arms:
        for fold in range(a.folds):
            out = out_root / arm
            out.mkdir(parents=True, exist_ok=True)
            metrics_path = out / f"metrics_fold{fold}.json"
            if metrics_path.exists():
                print(f"[sweep] {arm} fold {fold}: already done, skipping")
                rows.append({"arm": arm, "fold": fold,
                             **json.loads(metrics_path.read_text())})
                continue
            test_ids = folds[fold]
            train_ids = [i for j, f in enumerate(folds) if j != fold for i in f]
            print(f"\n[sweep] {arm} fold {fold}: train {len(train_ids)} / test {len(test_ids)} "
                  f"({(time.time()-t0)/60:.0f} min elapsed)")
            if arm == "d2":
                det = arm_detectron2(a, fold, train_ids, test_ids, out, coco, d2_img_dir)
            else:
                spec = TORCHVISION_ARMS[arm]
                det = arm_torchvision(a, fold, train_ids, test_ids, out,
                                      a.mosaic_p if spec["mosaic"] else 0.0,
                                      spec["arch"])
            if det is None:
                print(f"[sweep] {arm} fold {fold}: FAILED, continuing")
                rows.append({"arm": arm, "fold": fold, "AP": None})
                continue
            try:
                m = evaluate(a.coco_json, json.loads(Path(det).read_text()), "masks",
                             image_ids=test_ids, out_json=metrics_path)
            except Exception as exc:                      # noqa: BLE001
                # a scoring failure must cost one fold, not the eight runs after it
                print(f"[sweep] {arm} fold {fold}: evaluation FAILED ({type(exc).__name__}: "
                      f"{exc}); detections kept at {det}")
                rows.append({"arm": arm, "fold": fold, "AP": None})
                continue
            m.update(n_train=len(train_ids), n_test=len(test_ids),
                     species=len(fold_species[fold]))
            metrics_path.write_text(json.dumps(m, indent=2))
            rows.append({"arm": arm, "fold": fold, **m})

    cols = ["arm", "fold", "AP", "AP50", "AP75", "n_detections", "n_train", "n_test", "species"]
    with open(out_root / "sweep_results.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[sweep] {len(rows)} runs in {(time.time()-t0)/60:.1f} min "
          f"-> {out_root/'sweep_results.tsv'}")
    for arm in arms:
        got = [r["AP"] for r in rows if r["arm"] == arm and r.get("AP") is not None]
        if got:
            print(f"   {arm:14} mean AP {sum(got)/len(got):.4f} over {len(got)} folds")


if __name__ == "__main__":
    main()
