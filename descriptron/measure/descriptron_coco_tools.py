#!/usr/bin/env python3
"""
descriptron_coco_tools.py — housekeeping for Descriptron COCO files and image folders
======================================================================================

  split        train / validation split, by SPECIMEN when a pattern says which images belong together
               (several views or body parts of one individual must never be split across train and val)
  prepare-d2   everything Detectron2 needs, using the existing tools unchanged:
               [Combine COCO (coco_combiner_V13) if several inputs] -> Min-COCO (coco_converter_v24) -> split
  check        one report of the problems that silently break training or measurement
  rename       rename or merge categories (old=new; several old names may map to one new name)
  folders      tidy an image folder: macOS ._ files, mask PNGs mixed with images, 16-bit TIFF -> 8-bit PNG

Examples
    python descriptron_coco_tools.py split ann.json --out-dir d2/ --val-fraction 0.2 \\
        --group-regex "^(.+?)_(head|wing|dorsal|lateral)"
    python descriptron_coco_tools.py prepare-d2 ann.json --images photos/ --out-dir d2/ --group-regex "^([^_]+_[^_]+)"
    python descriptron_coco_tools.py check ann.json --images photos/
    python descriptron_coco_tools.py rename ann.json --out fixed.json --map "occipital margin=occipital_corner"
    python descriptron_coco_tools.py folders photos/ --remove-dot-underscore --move-masks        (add --apply to act)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

IMG_EXT = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")


def load(path):
    with open(path) as f:
        return json.load(f)


def save(d, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=1)


# ------------------------------------------------------------------ split
def group_of(file_name: str, regex: Optional[str]) -> str:
    stem = Path(file_name).stem
    if not regex:
        return stem
    m = re.search(regex, stem)
    if not m:
        return stem
    return m.group(1) if m.groups() else m.group(0)


def split_coco(coco: dict, val_fraction: float = 0.2, group_regex: Optional[str] = None, seed: int = 0):
    """Split whole groups (specimens) into train and val. Returns (train, val, manifest)."""
    groups = defaultdict(list)
    for im in coco["images"]:
        groups[group_of(im["file_name"], group_regex)].append(im["id"])
    keys = sorted(groups)
    rnd = random.Random(seed)
    rnd.shuffle(keys)
    n_val = int(round(len(keys) * val_fraction))
    if val_fraction > 0 and len(keys) >= 2:
        n_val = min(max(n_val, 1), len(keys) - 1)
    else:
        n_val = 0
    val_groups = set(keys[:n_val])
    val_ids = {i for g in val_groups for i in groups[g]}

    def subset(ids):
        return {**{k: v for k, v in coco.items() if k not in ("images", "annotations")},
                "images": [im for im in coco["images"] if im["id"] in ids],
                "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
                "categories": coco.get("categories", [])}
    all_ids = {im["id"] for im in coco["images"]}
    train, val = subset(all_ids - val_ids), subset(val_ids)
    manifest = {"val_fraction": val_fraction, "group_regex": group_regex, "seed": seed,
                "groups": len(keys), "images_per_group_max": max((len(v) for v in groups.values()), default=0),
                "train": {"groups": len(keys) - n_val, "images": len(train["images"]), "annotations": len(train["annotations"])},
                "val": {"groups": n_val, "images": len(val["images"]), "annotations": len(val["annotations"])},
                "val_groups": sorted(val_groups)}
    return train, val, manifest


def cmd_split(a):
    coco = load(a.input)
    train, val, man = split_coco(coco, a.val_fraction, a.group_regex, a.seed)
    out = Path(a.out_dir)
    save(train, out / "train.json"); save(val, out / "val.json"); save(man, out / "split_manifest.json")
    print(f"{man['groups']} groups ({'specimens by --group-regex' if a.group_regex else 'one per image'}): "
          f"train {man['train']['images']} images / val {man['val']['images']} images")
    if not a.group_regex and man["images_per_group_max"] == 1:
        print("note: without --group-regex every image is its own group; if one specimen has several images, "
              "give a pattern so they stay on the same side of the split")
    if man["val"]["images"] == 0:
        print("warning: validation set is empty (too few images); training will have no held-out check")
    print(f"wrote {out/'train.json'}, {out/'val.json'}, {out/'split_manifest.json'}")


# ------------------------------------------------------------------ prepare for Detectron2
def cmd_prepare_d2(a):
    here = Path(__file__).resolve().parent
    gui = here.parent if here.name == "measure" else here
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    src = a.inputs[0]
    if len(a.inputs) > 1:
        comb = gui / "coco_combiner_V13.py"
        combined = out / "combined.json"
        cmd = [sys.executable, str(comb), "--input-jsons", *a.inputs, "--output-json", str(combined)]
        if a.images:
            cmd += ["--images-folders", a.images]
        print("combine:", " ".join(cmd)); subprocess.run(cmd, check=True)
        src = str(combined)
    conv = gui / "coco_converter_v24.py"
    cmd = [sys.executable, str(conv), "--input-json", src, "--output-dir", str(out), "--output-base-name", "d2"]
    if a.images:
        cmd += ["--img-dir", a.images]
    if a.exclude_categories:
        cmd += ["--exclude-categories", *a.exclude_categories]
    print("min-coco:", " ".join(cmd)); subprocess.run(cmd, check=True)
    minimal = out / ("d2_keypoints.json" if a.keypoints_only else
                     "d2_segmentation.json" if a.segmentation_only else "d2_combined.json")
    coco = load(minimal)
    train, val, man = split_coco(coco, a.val_fraction, a.group_regex, a.seed)
    save(train, out / "train.json"); save(val, out / "val.json"); save(man, out / "split_manifest.json")
    print(f"\nsplit of {minimal.name}: train {man['train']['images']} images, val {man['val']['images']} images "
          f"({man['groups']} {'specimens' if a.group_regex else 'images'})")
    print("\nTrain with (Detectron2 v18):")
    print(f"  --coco-json {out/'train.json'} --val-json {out/'val.json'} --img-dir {a.images or '<images>'}")


# ------------------------------------------------------------------ check
def _poly_area(p):
    xs, ys = p[0::2], p[1::2]
    return 0.5 * abs(sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i] for i in range(len(xs))))


def check_coco(coco: dict, images: Optional[str] = None) -> Dict[str, List[str]]:
    issues = defaultdict(list)
    ims = {}
    for im in coco.get("images", []):
        if im["id"] in ims:
            issues["duplicate image ids"].append(str(im["id"]))
        ims[im["id"]] = im
    names = Counter(im["file_name"] for im in coco.get("images", []))
    issues["same file listed twice"] += [n for n, c in names.items() if c > 1]
    stems = defaultdict(set)
    for n in names:
        stems[Path(n).stem.lower()].add(n)
    issues["same name, different extension"] += [" / ".join(sorted(v)) for v in stems.values() if len(v) > 1]
    if images:
        on_disk = {p.name.lower(): p for p in Path(images).rglob("*") if p.suffix.lower() in IMG_EXT}
        disk_stems = {Path(k).stem: k for k in on_disk}
        for n in names:
            if n.lower() not in on_disk:
                alt = disk_stems.get(Path(n).stem.lower())
                issues["image file missing"].append(n + (f"  (a file '{on_disk[alt].name}' exists: extension differs)" if alt else ""))
    cats = {c["id"]: c for c in coco.get("categories", [])}
    norm = defaultdict(list)
    for c in cats.values():
        norm[re.sub(r"[\s_\-]+", "_", c["name"].strip().lower())].append(c["name"])
    issues["category names that differ only by spaces/underscores/case"] += [" | ".join(repr(x) for x in v)
                                                                             for v in norm.values() if len(v) > 1]
    issues["category names with leading/trailing spaces"] += [repr(c["name"]) for c in cats.values() if c["name"] != c["name"].strip()]
    used = Counter(a.get("category_id") for a in coco.get("annotations", []))
    issues["categories with no annotations"] += [c["name"] for cid, c in cats.items() if not used.get(cid)]
    kp_counts = defaultdict(Counter)
    for a in coco.get("annotations", []):
        im = ims.get(a.get("image_id"))
        where = f"annotation {a.get('id')} ({im['file_name'] if im else 'image ' + str(a.get('image_id'))})"
        if im is None:
            issues["annotation points to a missing image"].append(where); continue
        if a.get("category_id") not in cats:
            issues["annotation has an unknown category"].append(where)
        seg = a.get("segmentation")
        if isinstance(seg, list) and seg and not isinstance(seg[0], list):
            issues["flat (un-nested) polygon"].append(where); seg = [seg]
        if isinstance(seg, list):
            polys = [p for p in seg if isinstance(p, list)]
            if seg and not any(len(p) >= 6 for p in polys):
                issues["empty or degenerate mask"].append(where)
            elif polys and sum(_poly_area(p) for p in polys if len(p) >= 6) < 4 and not a.get("is_line"):
                issues["mask with (almost) zero area"].append(where)
            W, H = im.get("width") or 0, im.get("height") or 0
            if W and H and polys:
                xs = [v for p in polys for v in p[0::2]]; ys = [v for p in polys for v in p[1::2]]
                if xs and (min(xs) < -2 or min(ys) < -2 or max(xs) > W + 2 or max(ys) > H + 2):
                    issues["mask outside the image"].append(where)
        if "keypoints" in a:
            kp_counts[cats.get(a.get("category_id"), {}).get("name", "?")][len(a["keypoints"]) // 3] += 1
    for cname, cnt in kp_counts.items():
        if len(cnt) > 1:
            issues["landmark sets of different lengths in one category"].append(
                f"{cname}: " + ", ".join(f"{n} landmarks x{c}" for n, c in sorted(cnt.items())))
    zero = [im["file_name"] for im in ims.values() if not im.get("width") or not im.get("height")]
    issues["images without width/height"] += zero
    return {k: v for k, v in issues.items() if v}


def cmd_check(a):
    coco = load(a.input)
    issues = check_coco(coco, a.images)
    n_img, n_ann = len(coco.get("images", [])), len(coco.get("annotations", []))
    print(f"{a.input}: {n_img} images, {n_ann} annotations, {len(coco.get('categories', []))} categories")
    if a.template:
        if not any(Path(im["file_name"]).stem.lower() == Path(a.template).stem.lower() for im in coco.get("images", [])):
            issues["SAM2-PAL template image is not in this JSON"] = [Path(a.template).name]
    if not issues:
        print("no problems found"); return 0
    for k, v in issues.items():
        print(f"\n{k} ({len(v)}):")
        for x in v[:a.show]:
            print(f"  - {x}")
        if len(v) > a.show:
            print(f"  ... and {len(v) - a.show} more")
    if a.report:
        save(issues, a.report); print(f"\nfull report: {a.report}")
    return 1


# ------------------------------------------------------------------ rename / merge
def rename_categories(coco: dict, mapping: Dict[str, str]) -> dict:
    by_name, new_cats, remap = {}, [], {}
    for c in coco.get("categories", []):
        target = mapping.get(c["name"], mapping.get(c["name"].strip(), c["name"]))
        if target in by_name:
            remap[c["id"]] = by_name[target]["id"]                 # merge into the existing category
        else:
            nc = dict(c, name=target); by_name[target] = nc; new_cats.append(nc); remap[c["id"]] = c["id"]
    anns = [dict(a, category_id=remap.get(a.get("category_id"), a.get("category_id"))) for a in coco.get("annotations", [])]
    return {**coco, "categories": new_cats, "annotations": anns}


def cmd_rename(a):
    mapping = {}
    for m in a.map:
        old, _, new = m.partition("=")
        if not new:
            sys.exit(f"--map needs old=new, got {m!r}")
        mapping[old] = new.strip()
    coco = load(a.input)
    before = len(coco.get("categories", []))
    out = rename_categories(coco, mapping)
    save(out, a.out)
    print(f"wrote {a.out}: {before} -> {len(out['categories'])} categories")


# ------------------------------------------------------------------ folders
def cmd_folders(a):
    root = Path(a.folder)
    plan = []
    if a.remove_dot_underscore:
        plan += [("delete", p, None) for p in root.rglob("._*")]
    if a.move_masks:
        dest = root / "_masks_moved_out"
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() == ".png" and re.search(r"(_mask|_masks|mask_)", p.stem, re.I):
                plan.append(("move", p, dest / p.name))
    if a.tiff8:
        dest = root / "_png8"
        for p in root.iterdir():
            if p.suffix.lower() in (".tif", ".tiff"):
                plan.append(("tiff8", p, dest / (p.stem + ".png")))
    if not plan:
        print("nothing to do"); return
    for act, p, q in plan:
        print(f"{'would ' if not a.apply else ''}{act}: {p.name}" + (f" -> {q.relative_to(root)}" if q else ""))
    if not a.apply:
        print(f"\n{len(plan)} actions shown; nothing changed. Add --apply to do them."); return
    for act, p, q in plan:
        if act == "delete":
            p.unlink(missing_ok=True)
        elif act == "move":
            q.parent.mkdir(exist_ok=True); shutil.move(str(p), str(q))
        elif act == "tiff8":
            import numpy as np
            from PIL import Image
            q.parent.mkdir(exist_ok=True)
            im = Image.open(p); arr = np.asarray(im)
            if arr.dtype != np.uint8:
                lo, hi = np.percentile(arr, (0.1, 99.9))
                arr = np.clip((arr.astype(float) - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(q)
    print(f"done: {len(plan)} actions (originals of converted TIFFs are kept)")


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("Examples")[-1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("split"); p.add_argument("input"); p.add_argument("--out-dir", required=True)
    p.add_argument("--val-fraction", type=float, default=0.2); p.add_argument("--group-regex"); p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_split)
    p = sub.add_parser("prepare-d2"); p.add_argument("inputs", nargs="+"); p.add_argument("--images"); p.add_argument("--out-dir", required=True)
    p.add_argument("--val-fraction", type=float, default=0.2); p.add_argument("--group-regex"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-categories", nargs="*", default=[])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--segmentation-only", action="store_true"); g.add_argument("--keypoints-only", action="store_true")
    p.set_defaults(fn=cmd_prepare_d2)
    p = sub.add_parser("check"); p.add_argument("input"); p.add_argument("--images")
    p.add_argument("--template", help="SAM2-PAL template image that must be in the JSON")
    p.add_argument("--show", type=int, default=15); p.add_argument("--report", help="write every issue to this JSON")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("rename"); p.add_argument("input"); p.add_argument("--out", required=True)
    p.add_argument("--map", nargs="+", required=True, help='"old name=new name" (repeat; same new name merges)')
    p.set_defaults(fn=cmd_rename)
    p = sub.add_parser("folders"); p.add_argument("folder")
    p.add_argument("--remove-dot-underscore", action="store_true"); p.add_argument("--move-masks", action="store_true")
    p.add_argument("--tiff8", action="store_true", help="write 8-bit PNG copies of TIFFs into _png8/")
    p.add_argument("--apply", action="store_true", help="actually do it (default: only show what would happen)")
    p.set_defaults(fn=cmd_folders)
    a = ap.parse_args(argv)
    r = a.fn(a)
    return r or 0


if __name__ == "__main__":
    sys.exit(main())
