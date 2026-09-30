"""One accepted segmentation per specimen (first image of the record that passed QC), up to N per species;
copies the RGBA crops to images/ and writes coco.json (polygon = largest alpha contour, category moth_dorsal)."""
import csv, json, shutil, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2
from PIL import Image
N = 10
ok = Path("seg/ok"); out = Path("images"); out.mkdir(exist_ok=True)
by = defaultdict(dict)
excl = {l.strip() for l in open("exclude_qc.txt") if l.strip() and not l.startswith("#")} if Path("exclude_qc.txt").exists() else set()
for p in sorted(ok.glob("*.png"), key=lambda p: (p.stem.rsplit("_", 1)[0], int(p.stem.rsplit("_", 1)[1]))):
    sp, rest = p.stem.split("__"); gid = rest.rsplit("_", 1)[0]
    if f"{sp}__{gid}" in excl: continue
    by[sp].setdefault(gid, p)
coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "moth_dorsal"}]}
rows = []
for sp in sorted(by):
    for gid, p in list(by[sp].items())[:N]:
        fn = f"{sp}__{gid}.png"; shutil.copy(p, out / fn)
        a = np.array(Image.open(p))[:, :, 3] > 127
        cnts, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)[:, 0, :]
        iid = len(coco["images"]) + 1
        coco["images"].append({"id": iid, "file_name": fn, "width": a.shape[1], "height": a.shape[0]})
        x, y, w, h = cv2.boundingRect(c)
        coco["annotations"].append({"id": iid, "image_id": iid, "category_id": 1, "segmentation": [c.flatten().astype(float).tolist()],
                                    "area": float(a.sum()), "bbox": [x, y, w, h], "iscrowd": 0})
        rows.append((sp, gid, fn))
json.dump(coco, open("coco.json", "w"))
with open("specimens_used.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["species", "gbifID", "file"]); w.writerows(rows)
cnt = {sp: min(N, len(d)) for sp, d in by.items()}
print(len(rows), "specimens;", len(cnt), "species; per species:", sorted(cnt.values()))
