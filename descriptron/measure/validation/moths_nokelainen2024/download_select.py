"""Pick up to N_REC specimen records per species from gbif_candidates.csv (NHMUK/MZH unreachable), paper's source
institution first, then ETHZ/TAM, then the rest; random order within a tier (seed 1). Downloads every image of each
record (long side capped at 3000 px); ventral views skipped by URL/title. Writes manifest.csv (licence per image)."""
import csv, random, re, time, io, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
N_REC = 22
random.seed(1)
SCAN = {"Arachnis picta": {"ASUHIC","CSU","MEM","MISSA","NMSU","YPM"}, "Gnophaela vermiculata": {"CSU","ASUHIC","NMSU"},
        "Grammia incorrupta": {"NAU"}, "Grammia nevadensis": {"SDNHM"}, "Grammia ornata": {"SDNHM"}, "Hypoprepia fucosa": {"YPM"},
        "Notarctia proxima": {"SDNHM","CSU"}, "Virbia ostenta": {"CSU","ASUHIC","NAU","MISSA","NMSU","ANSP"},
        "Zale lunata": {"PU"}, "Autographa californica": {"ASUHIC","CSU","MISSA","UMNH"}, "Orthosia hibisci": {"CSU"}}
recs = defaultdict(lambda: defaultdict(list)); meta = {}
for r in csv.DictReader(open("gbif_candidates.csv")):
    if r["institution"] in ("NHMUK", "MZH"): continue
    if re.search(r"ventral|_V\.|_v\.|underside", r["url"] + " " + r["title"] + " " + r["description"], re.I): continue
    recs[r["species"]][r["gbifID"]].append(r); meta[r["gbifID"]] = r
jobs = []
for sp, d in recs.items():
    pref = SCAN.get(sp, {"TAM"} if "Estonian" in d[next(iter(d))][0]["paper_source"] else set())
    ids = list(d); random.shuffle(ids)
    tier = lambda i: 0 if d[i][0]["institution"] in pref else 1 if d[i][0]["institution"] in ("ETHZ", "TAM") else 2
    for gid in sorted(ids, key=tier)[:N_REC]:
        for k, r in enumerate(d[gid][:3]):
            jobs.append((sp, gid, k, r))
out = Path("images_raw"); out.mkdir(exist_ok=True)
def fetch(j):
    sp, gid, k, r = j
    fn = out / f"{sp.replace(' ', '_')}__{gid}_{k}.jpg"
    if fn.exists(): return (j, fn, "cached")
    for t in range(3):
        try:
            req = urllib.request.Request(r["url"], headers={"User-Agent": "Descriptron-validation/2.1 (research use)"})
            im = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=90).read())).convert("RGB")
            im.thumbnail((3000, 3000)); im.save(fn, quality=95); return (j, fn, "ok")
        except Exception as e:
            err = str(e); time.sleep(2)
    return (j, None, err)
with ThreadPoolExecutor(4) as ex, open("manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "species", "gbifID", "img_k", "institution", "catalog", "license", "url", "status"])
    for n, (j, fn, st) in enumerate(ex.map(fetch, jobs)):
        sp, gid, k, r = j
        w.writerow([fn.name if fn else "", sp, gid, k, r["institution"], r["catalog"], r["license"], r["url"], st]); f.flush()
        if n % 50 == 0: print(n, "/", len(jobs), sp, st, flush=True)
print("done", len(jobs))
