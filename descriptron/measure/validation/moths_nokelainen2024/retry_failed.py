"""Retry failed downloads one at a time, 5 s apart (Symbiota refused connections under 4 parallel requests)."""
import csv, re, time, io, urllib.request
from pathlib import Path
from PIL import Image
rows = list(csv.DictReader(open("manifest.csv"))); n = 0
refused = 0
for r in rows:
    if r["status"] in ("ok", "cached"): continue
    if "symbiota" in r["url"] and refused >= 10:
        r["status"] = "skipped: symbiota refusing connections"; continue
    fn = Path("images_raw") / f"{r['species'].replace(' ', '_')}__{r['gbifID']}_{r['img_k']}.jpg"
    for t in range(1 if "symbiota" in r["url"] else 3):
        try:
            req = urllib.request.Request(r["url"], headers={"User-Agent": "Descriptron-validation/2.1 (research use)"})
            im = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=90).read())).convert("RGB")
            im.thumbnail((3000, 3000)); im.save(fn, quality=95); r["status"] = "ok_retry"; r["file"] = fn.name; n += 1; refused = 0; break
        except Exception as e:
            r["status"] = f"failed: {e}"; refused += "symbiota" in r["url"]; time.sleep(20)
    time.sleep(5)
with open("manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("recovered", n)

# Top-up: species still short of 5 specimens get records from other hosts (Symbiota refused connections), up to 12 records
import random
from collections import defaultdict
random.seed(2)
have = defaultdict(set)
for r in rows:
    if r["status"] in ("ok", "cached", "ok_retry"): have[r["species"]].add(r["gbifID"])
cand = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("gbif_candidates.csv")):
    if any(h in r["url"] for h in ("symbiota.org", "data.nhm.ac.uk", "digitarium.fi")): continue
    if re.search(r"ventral|_V\.|_v\.|underside", r["url"] + " " + r["title"] + " " + r["description"], re.I): continue
    cand[r["species"]][r["gbifID"]].append(r)
added = []
for sp, d in cand.items():
    if len(have[sp]) >= 5: continue
    ids = [g for g in d if g not in have[sp]]; random.shuffle(ids)
    for gid in ids[:12]:
        for k, r in enumerate(d[gid][:3]):
            fn = Path("images_raw") / f"{sp.replace(' ', '_')}__{gid}_{k}.jpg"
            try:
                req = urllib.request.Request(r["url"], headers={"User-Agent": "Descriptron-validation/2.1 (research use)"})
                im = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=90).read())).convert("RGB")
                im.thumbnail((3000, 3000)); im.save(fn, quality=95); st = "ok_topup"
            except Exception as e:
                st = f"failed: {e}"
            added.append({"file": fn.name if st == "ok_topup" else "", "species": sp, "gbifID": gid, "img_k": k, "institution": r["institution"],
                          "catalog": r["catalog"], "license": r["license"], "url": r["url"], "status": st})
            time.sleep(1)
with open("manifest.csv", "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writerows(added)
print("top-up images:", sum(a["status"] == "ok_topup" for a in added))
