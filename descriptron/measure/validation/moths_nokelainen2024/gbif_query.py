"""List GBIF preserved-specimen occurrences with images for each species (open API, CC licences recorded)."""
import csv, json, re, sys, time, urllib.parse, urllib.request
from collections import Counter
rows = [l.rstrip("\n").split("\t") for l in open("clade_sources.tsv")]
out = []
for fam, name, origin, src in rows:
    sci = re.sub(r"\s*\(.*$", "", name).strip()
    sci = " ".join(sci.split()[:2])            # species level (drop subspecies)
    recs, off = [], 0
    while off < 900:
        q = urllib.parse.urlencode({"scientificName": sci, "mediaType": "StillImage",
                                    "basisOfRecord": "PRESERVED_SPECIMEN", "limit": 300, "offset": off})
        d = json.load(urllib.request.urlopen("https://api.gbif.org/v1/occurrence/search?" + q, timeout=60))
        recs += d["results"]; off += 300
        if d["endOfRecords"]: break
    for r in recs:
        for m in r.get("media", []):
            if m.get("type") == "StillImage" and m.get("identifier"):
                out.append({"family": fam, "species": sci, "paper_source": src, "gbifID": r["key"],
                            "institution": r.get("institutionCode", ""), "catalog": r.get("catalogNumber", ""),
                            "sex": r.get("sex", ""), "country": r.get("countryCode", ""),
                            "license": m.get("license", r.get("license", "")), "url": m["identifier"],
                            "title": m.get("title", ""), "description": (m.get("description") or "")[:80]})
    c = Counter(r.get("institutionCode", "") for r in recs)
    print(f"{sci}: {len(recs)} recs; {dict(c.most_common(4))}", flush=True)
    time.sleep(0.3)
with open("gbif_candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
