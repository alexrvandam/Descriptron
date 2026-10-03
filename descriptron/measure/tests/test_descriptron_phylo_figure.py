"""descriptron_phylo_figure_v1: panels b/c are chosen by colour-pattern identification (>= min_species; then the
best structure covering every species), never by the phylogenetic result; the figure renders without DNA."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import descriptron_phylo_figure_v1 as fig        # noqa: E402

SP = [f"s{i}" for i in range(12)]


def _setup(tmp):
    rng = np.random.default_rng(1)
    tree = "(" + ",".join(f"({SP[i]}:1,{SP[i+1]}:1):1" for i in range(0, 12, 2)) + ");"
    (tmp / "tree.nwk").write_text(tree)
    groups = []
    # 'good' separates all 12 species, 'all' covers 12 but noisier, 'few' perfect but only 4 species
    for cat, spp, noise in (("good", SP, 0.1), ("noisy", SP, 3.0), ("few", SP[:4], 0.01)):
        d = tmp / "colour" / cat; d.mkdir(parents=True)
        rows = []
        for s in spp:
            centre = rng.normal(0, 1, 20)
            for k in range(3):
                fn = f"img_{cat}_{s}_{k}.tif"; groups.append((fn, s))
                rows.append([f"{fn}_{len(rows)}"] + list(centre + rng.normal(0, noise, 20)))
        pd.DataFrame(rows, columns=["filename"] + [f"f{i}" for i in range(20)]).to_csv(
            d / f"color_homology_features_{cat}.csv", index=False)
        p = tmp / "phylo" / cat; p.mkdir(parents=True)
        # deliberately make the NOISY structure the strongest phylogenetic result: it must not be chosen for that
        pd.DataFrame([{"trait_set": "colour_pattern", "n_species": len(spp), "n_variables": 20, "scale": "standardize",
                       "Kmult": 0.9 if cat == "noisy" else 0.1, "P": 0.001 if cat == "noisy" else 0.5, "Z": 1}]).to_csv(
            p / "phylogenetic_signal.csv", index=False)
    pd.DataFrame(groups, columns=["filename", "group_label"]).to_csv(tmp / "groups.csv", index=False)


def test_rule_and_render(tmp_path):
    _setup(tmp_path)
    fig.main(["--tree", str(tmp_path / "tree.nwk"), "--phylo_dir", str(tmp_path / "phylo"),
              "--colour_dir", str(tmp_path / "colour"), "--groups", str(tmp_path / "groups.csv"),
              "--min_species", "10", "--out", str(tmp_path / "out" / "fig")])
    v = json.load(open(tmp_path / "out" / "fig.json"))
    assert v["panel_b"] == "good" and v["panel_c"] == "noisy"     # 'few' excluded by min_species, 'noisy' only as full cover
    assert (tmp_path / "out" / "fig.png").stat().st_size > 10000 and (tmp_path / "out" / "fig.pdf").exists()
    r = pd.read_csv(tmp_path / "out" / "colour_identification_rank.tsv", sep="\t")
    assert r.iloc[0].structure in ("few", "good")
