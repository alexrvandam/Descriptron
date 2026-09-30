"""(1) Blomberg's K of the authors' deposited species means (colourmetrics_for_phylanova_data.txt; 43 Noctuoidea
species, their tree) computed by Descriptron and by phytools::phylosig - a like-for-like numerical check.
(2) Independent images: species-mean HSV saturation and brightness of the whole dorsal moth measured by Descriptron on
GBIF specimens vs the authors' fore/hindwing values (different specimens, different wing regions) - Spearman rho."""
import argparse, importlib.util, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("--measure_dir", required=True); ap.add_argument("--rscript", required=True)
ap.add_argument("--rlib", required=True); ap.add_argument("--published", required=True); ap.add_argument("--out", default="compare_published")
a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("dp", Path(a.measure_dir) / "descriptron_phylo.py"); dp = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp)
tree = dp.parse_newick(Path("tree_noctuoidea43.nwk").read_text()); tips = dp.tips_of(tree); tn = [t.name for t in tips]
fix = {"Orthosia_hibisci_brucei": "Orthosia_hibisci", "Arctia_villica_britannica": "Arctia_villica", "Nudaria_mundane": "Nudaria_mundana",
       "Schranckia_taenialis": "Schrankia_taenialis", "Euplagia_quadripunctata": "Euplagia_quadripunctaria",
       "Calliteara_taiwana": "Calliteara_pudibunda", "Herminia_tarsicrinalis": "Herminia_grisealis"}
P = pd.read_csv(a.published, sep="\t", decimal=",", encoding="latin1"); P["tip"] = P["tip.label"].replace(fix)
P = P.set_index("tip").loc[tn]
traits = [c for c in P.columns if c.endswith(".x") and P[c].dtype.kind in "fi"]
P[traits] = P[traits].astype(float)
P[traits].to_csv(out / "published_traits.csv"); (out / "tree.nwk").write_text(Path("tree_noctuoidea43.nwk").read_text())
R = r'''args <- commandArgs(TRUE); .libPaths(c(args[1], .libPaths())); suppressMessages({library(ape); library(phytools)})
phy <- read.tree(file.path(args[2], "tree.nwk")); X <- read.csv(file.path(args[2], "published_traits.csv"), row.names = 1, check.names = FALSE)
o <- data.frame(); for (v in colnames(X)) { x <- X[phy$tip.label, v]; names(x) <- phy$tip.label
  set.seed(1); k <- phylosig(phy, x, method = "K", test = TRUE, nsim = 999); o <- rbind(o, data.frame(trait = v, K = k$K, P = k$P)) }
write.csv(o, file.path(args[2], "R_K.csv"), row.names = FALSE)'''
(out / "k.R").write_text(R); subprocess.run([a.rscript, str(out / "k.R"), a.rlib, str(out)], check=True)
RK = pd.read_csv(out / "R_K.csv").set_index("trait"); C = dp.vcv(tips, tips); rows = []
for t in traits:
    r = dp.phylo_signal(P[t].values[:, None], C, 999, np.random.default_rng(1))
    rows.append({"trait": t, "K_descriptron": r["Kmult"], "K_phytools": RK.loc[t, "K"], "rel_diff": abs(r["Kmult"] - RK.loc[t, "K"]) / RK.loc[t, "K"],
                 "P_descriptron": r["P"], "P_phytools": RK.loc[t, "P"]})
K = pd.DataFrame(rows); K.to_csv(out / "published_traits_K_descriptron_vs_phytools.csv", index=False)
# (2) independent images: HSV of the masked dorsal moth (OpenCV HSV scaled to 0-1 as in the authors' table)
sp_rows = []
for p in sorted(Path("images").glob("*.png")):
    im = np.array(Image.open(p)); m = im[:, :, 3] > 127
    hsv = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2HSV)[m].astype(float)
    rgb = im[:, :, :3][m].astype(float).mean(0)          # the authors' definition: HSV of the region's mean RGB
    sp_rows.append({"species": p.stem.split("__")[0], "saturation": (rgb.max() - rgb.min()) / rgb.max(), "brightness": rgb.max() / 255,
                    "saturation_per_pixel": hsv[:, 1].mean() / 255, "brightness_per_pixel": hsv[:, 2].mean() / 255})
D = pd.DataFrame(sp_rows).groupby("species").mean().loc[tn]; D.to_csv(out / "descriptron_species_hsv.csv")
cor = []
for ours in ("saturation", "brightness", "saturation_per_pixel", "brightness_per_pixel"):
    for wing in ("fw", "hw"):
        rho, p = spearmanr(D[ours], P[f"{wing}.{ours.split('_')[0]}.x"]); cor.append({"measure": ours, "published_wing": wing, "spearman_rho": rho, "P": p, "n_species": len(tn)})
pd.DataFrame(cor).to_csv(out / "independent_images_vs_published.csv", index=False)
# (3) the authors' values rebuilt for the whole moth, like Descriptron's: per image, two forewings + two hindwings + body,
# area-weighted mean RGB (their per-region table, SourceData_Fig5_moth_image_data.txt), then HSV
img = pd.read_csv(Path(a.published).with_name("SourceData_Fig5_moth_image_data.txt"), sep="\t", decimal=",", encoding="latin1")
img["img"] = img["Label.x"].str.replace(r"_[fhb]\d*$", "", regex=True); wts = {"F": 2, "H": 2, "B": 1}; wm = []
for (spn, _), g in img.groupby(["Species", "img"]):
    if set(g.ROI) != {"F", "H", "B"}: continue
    wt = np.array([wts[r] * ar for r, ar in zip(g.ROI, g.area)]); rgb = (g[["r", "g", "b"]].values * wt[:, None]).sum(0) / wt.sum()
    wm.append({"species": spn, "bright": rgb.max() / 255, "sat": (rgb.max() - rgb.min()) / rgb.max()})
PW = pd.DataFrame(wm).groupby("species").mean(); PW.index = [{**fix, "Lypephila_craccae": "Lygephila_craccae"}.get(i, i) for i in PW.index]
PW.to_csv(out / "published_wholemoth_species_hsv.csv")
com = D.index.intersection(PW.index); whole = []
for ours, theirs in (("brightness", "bright"), ("saturation", "sat")):
    rho, pv = spearmanr(D.loc[com, ours], PW.loc[com, theirs])
    whole.append({"measure": ours, "published": "whole moth (2F+2H+B, area-weighted mean RGB, from their per-image table)", "spearman_rho": rho, "P": pv, "n_species": len(com)})
pd.DataFrame(whole).to_csv(out / "independent_images_vs_published_wholemoth.csv", index=False)
print(pd.DataFrame(whole).round(3).to_string())
json.dump({"K_max_rel_diff": float(K.rel_diff.max()), "n_traits": len(traits), "correlations": cor}, open(out / "summary.json", "w"), indent=1)
print(K.round(6).to_string()); print(pd.DataFrame(cor).round(3).to_string())
