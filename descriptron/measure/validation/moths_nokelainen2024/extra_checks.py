"""Inputs for the robustness checks (run in WORK after assemble_traits.py):
- robust_ethz_tam/*.csv: specimens photographed at ETH Zurich or Tartu only, species with >= 3 such specimens;
- strategy.csv: the authors' aposematic (1) / camouflaged (0) classification, for PGLS;
- tree_full82_renamed.nwk: the authors' full tree with the 7 tip names mapped (to show descriptron_phylo prunes it);
- specimen_hsv_institution.csv: per-specimen brightness/saturation (mean RGB) with the photographing institution."""
import re
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
FIX = {"Orthosia_hibisci_brucei": "Orthosia_hibisci", "Arctia_villica_britannica": "Arctia_villica", "Nudaria_mundane": "Nudaria_mundana",
       "Schranckia_taenialis": "Schrankia_taenialis", "Euplagia_quadripunctata": "Euplagia_quadripunctaria",
       "Calliteara_taiwana": "Calliteara_pudibunda", "Herminia_tarsicrinalis": "Herminia_grisealis"}
u = pd.read_csv("specimens_used.csv", dtype=str)
man = pd.read_csv("manifest.csv", dtype=str).drop_duplicates("gbifID").set_index("gbifID")
u["inst"] = u.gbifID.map(man.institution).fillna("NA")
rows = []
for _, r in u.iterrows():
    im = np.array(Image.open(Path("images") / r.file)); m = im[:, :, 3] > 127; rgb = im[:, :, :3][m].astype(float).mean(0)
    rows.append({"species": r.species, "inst": r.inst, "bright": rgb.max() / 255, "sat": (rgb.max() - rgb.min()) / rgb.max()})
pd.DataFrame(rows).to_csv("specimen_hsv_institution.csv", index=False)
keep = set(u.loc[u.inst.isin(["ETHZ", "TAM"]), "file"]); Path("robust_ethz_tam").mkdir(exist_ok=True)
for f in ("shape", "meas", "colour", "texture"):
    d = pd.read_csv(f + ".csv"); d = d[d.file.isin(keep)]; n = d.groupby("species").size()
    d[d.species.isin(n[n >= 3].index)].to_csv(f"robust_ethz_tam/{f}.csv", index=False)
p = pd.read_csv("SourceData/SourceData_phylanova_colourmetrics.txt", sep="\t", decimal=",", encoding="latin1")
p["species"] = p["tip.label"].replace(FIX); p["aposematic"] = (p.strategy == "aposematic").astype(int)
p[p.species.isin(set(u.species))][["species", "aposematic"]].to_csv("strategy.csv", index=False)
t = Path("SourceData/SourceData_Fig3_ultrametric-1.tre").read_text()
for a, b in FIX.items():
    t = re.sub(rf"\b{a}\b", b, t)
Path("tree_full82_renamed.nwk").write_text(t)
print("extra inputs written")
