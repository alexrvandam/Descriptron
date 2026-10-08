#!/usr/bin/env python3
"""
make_example.py - a small, synthetic data set for descriptron_community_v1.py, with known answers
==================================================================================================

Writes (beside this file, or into --out):

  metadata_darwin_core_example.csv   one row per specimen, Darwin Core column names:
        occurrenceID, catalogNumber, institutionCode, basisOfRecord, scientificName, family, genus,
        specificEpithet, locationID, locality, country, decimalLatitude, decimalLongitude, habitat,
        minimumElevationInMeters, eventDate, recordedBy, identifiedBy, associatedSequences, associatedMedia
     - locationID is the site; habitat is the treatment ("primary forest" / "secondary forest");
       minimumElevationInMeters is an example site-level covariate (--covariate);
     - associatedMedia holds the specimen's image file name(s), '|' separated: this links trait rows to specimens;
     - associatedSequences holds a sequence id for the ONE exemplar per species that was sequenced (strategy 1:
       one tree of all species, pruned per site) - every other specimen leaves it empty.
  tree_species.nwk                   strategy 1: one tip per species (tip names = scientificName, spaces -> _)
  tree_site_tips.nwk + tip_map.csv   strategy 2: one sequence per species PER SITE; tip_map.csv says which species
                                     and site each tip is (columns tip, species, site)
  traits_colour_example.csv          one row per image: filename + 12 colour features
  README.md

The traits are simulated so the analysis has something to find: each species' colour is Brownian motion on the
tree (phylogenetic signal), secondary-forest populations are shifted on two features (a treatment effect, the same
in every species), and the two habitats hold phylogenetically different subsets of species (so part of the
variation is shared between phylogeny and habitat).
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def random_tree(names, rng):
    """ultrametric coalescent-like tree as Newick, plus the tip-to-tip shared-path matrix"""
    nodes = [(n, 0.0) for n in names]           # (newick, height)
    h = 0.0
    while len(nodes) > 1:
        h += rng.exponential(1.0 / len(nodes))
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        (a, ha), (b, hb) = nodes[i], nodes[j]
        new = (f"({a}:{h - ha:.5f},{b}:{h - hb:.5f})", h)
        nodes = [x for k, x in enumerate(nodes) if k not in (i, j)] + [new]
    return nodes[0][0] + ";", h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    n_sp = 24
    genera = ["Pheidole", "Strumigenys", "Camponotus", "Crematogaster", "Tetramorium", "Odontomachus"]
    species = [f"{genera[i % len(genera)]} sp{i + 1:02d}" for i in range(n_sp)]
    tips = [s.replace(" ", "_") for s in species]
    nwk, height = random_tree(tips, rng)
    (out / "tree_species.nwk").write_text(nwk + "\n")

    # Brownian-motion colour on the tree: simulate by the shared-path covariance of a parsed copy of the tree
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import descriptron_phylo as ph
    root = ph.parse_newick(nwk)
    T = {t.name: t for t in ph.tips_of(root)}
    C = ph.vcv([T[t] for t in tips], [T[t] for t in tips])
    L = np.linalg.cholesky(C + 1e-9 * np.eye(n_sp))
    n_feat = 12
    sp_effect = L @ rng.normal(size=(n_sp, n_feat)) * 1.5
    # habitat preference follows phylogeny (feature 0 of a second BM draw): primary-only, secondary-only, both
    pref = (L @ rng.normal(size=n_sp))
    order = np.argsort(pref)
    where = {}
    for k, i in enumerate(order):
        where[species[i]] = "primary" if k < 8 else ("secondary" if k >= 16 else "both")
    sites = [(f"P{i + 1:02d}", "primary forest") for i in range(10)] + [(f"S{i + 1:02d}", "secondary forest") for i in range(12)]
    shift = np.zeros(n_feat); shift[[1, 4]] = [1.2, -0.9]        # the treatment effect, same in every species

    rows, trait_rows, tipmap, exemplar = [], [], [], set()
    k = 0
    for s_i, (site, hab) in enumerate(sites):
        lat, lon = 9.0 + 0.05 * s_i, -83.6 - 0.04 * s_i
        elev = int(200 + 60 * s_i + rng.integers(-30, 30))
        allowed = [s for s in species if where[s] == "both" or where[s] == hab.split()[0]]
        here = sorted(rng.choice(allowed, size=min(len(allowed), int(rng.integers(6, 11))), replace=False).tolist())
        for sp in here:
            i = species.index(sp)
            tipmap.append({"tip": f"{sp.replace(' ', '_')}__{site}", "species": sp, "site": site})
            for rep in range(int(rng.integers(1, 4))):
                k += 1
                occ = f"DESC-ANT-{k:05d}"
                img = f"{occ}_head_frontal.tif"
                seq = ""
                if sp not in exemplar:
                    exemplar.add(sp); seq = f"COI_{sp.replace(' ', '_')}"
                gen, ep = sp.split(" ")
                rows.append({"occurrenceID": occ, "catalogNumber": f"CAT{k:05d}", "institutionCode": "MFN",
                             "basisOfRecord": "PreservedSpecimen", "scientificName": sp, "family": "Formicidae",
                             "genus": gen, "specificEpithet": ep, "locationID": site,
                             "locality": f"Example plot {site}", "country": "Costa Rica",
                             "decimalLatitude": round(lat, 5), "decimalLongitude": round(lon, 5), "habitat": hab,
                             "minimumElevationInMeters": elev, "eventDate": "2027-03-15", "recordedBy": "Example collector",
                             "identifiedBy": "Example taxonomist", "associatedSequences": seq, "associatedMedia": img})
                x = sp_effect[i] + (shift if hab.startswith("secondary") else 0) + rng.normal(scale=0.35, size=n_feat)
                trait_rows.append({"filename": img, **{f"colour_f{j + 1:02d}": round(float(x[j]), 5) for j in range(n_feat)}})

    with open(out / "metadata_darwin_core_example.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "traits_colour_example.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trait_rows[0])); w.writeheader(); w.writerows(trait_rows)
    # strategy 2: one sequence per species per site -> a tree whose tips are species__site
    # each species' site sequences sit together where the species sits in the species tree (short branches)
    sp_newick = nwk.rstrip(";")
    for sp in species:
        tips_sp = [t["tip"] for t in tipmap if t["species"] == sp]
        tok = sp.replace(" ", "_")
        if tips_sp:
            sub = "(" + ",".join(f"{t}:0.00500" for t in tips_sp) + ")" if len(tips_sp) > 1 else tips_sp[0]
            sp_newick = sp_newick.replace(f"{tok}:", f"{sub}:", 1)
    (out / "tree_site_tips.nwk").write_text(sp_newick + ";\n")
    with open(out / "tip_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tip", "species", "site"]); w.writeheader(); w.writerows(tipmap)
    print(f"{len(rows)} specimens, {n_sp} species, {len(sites)} sites -> {out}")


if __name__ == "__main__":
    main()
