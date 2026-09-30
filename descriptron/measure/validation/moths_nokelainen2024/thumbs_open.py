"""Openly licensed (CC0 / CC BY, not NC or SA) museum photographs for figure thumbnails of species whose measured
specimens are all NC-licensed. Writes images_thumbs_raw/ and thumbs_manifest.csv."""
import csv, io, random, time, urllib.request
from pathlib import Path
from PIL import Image
NEED = ['Zale_lunata', 'Anarta_myrtilli', 'Lygephila_craccae', 'Hypena_crassalis', 'Lygephila_pastinum',
        'Macrochilo_cribrumalis', 'Herminia_grisealis', 'Rhyparia_purpurata']
open_lic = lambda l: l and "-nc" not in l.lower() and "-sa" not in l.lower() and ("/by/" in l or "zero" in l)
random.seed(3); out = Path("images_thumbs_raw"); out.mkdir(exist_ok=True); rows = []
cand = {}
for r in csv.DictReader(open("gbif_candidates.csv")):
    s = r["species"].replace(" ", "_")
    if s in NEED and open_lic(r["license"]) and "symbiota" not in r["url"]:
        cand.setdefault(s, {}).setdefault(r["gbifID"], r)
for s in NEED:
    ids = list(cand.get(s, {})); random.shuffle(ids)
    ids.sort(key=lambda g: cand[s][g]["institution"] not in ("ETHZ", "PU"))
    for gid in ids[:4]:
        r = cand[s][gid]; fn = out / f"{s}__{gid}_0.jpg"
        try:
            req = urllib.request.Request(r["url"], headers={"User-Agent": "Descriptron-validation/2.1 (research use)"})
            im = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=90).read())).convert("RGB"); im.thumbnail((3000, 3000)); im.save(fn, quality=95)
            rows.append({**r, "file": fn.name})
        except Exception as e:
            print(s, gid, "failed", e)
        time.sleep(1)
with open("thumbs_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "downloaded")
