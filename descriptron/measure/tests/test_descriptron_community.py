"""descriptron_community_v1: runs on the synthetic example (both tree strategies, per-site trees, COCO input) and
recovers what the example was built with (phylogenetic signal, a within-species habitat shift, clustered
primary-forest communities)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent.parent
EX = HERE / "examples" / "community_example"
sys.path.insert(0, str(HERE))
import descriptron_community_v1 as cm      # noqa: E402
import descriptron_phylo as ph             # noqa: E402


def run(out, *extra, iters="99"):
    r = subprocess.run([sys.executable, str(HERE / "descriptron_community_v1.py"),
                        "--metadata", str(EX / "metadata_darwin_core_example.csv"),
                        "--traits", f"colour={EX / 'traits_colour_example.csv'}", "--factor", "habitat",
                        "--iters", iters, "--out_dir", str(out), *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_species_tree_strategy(tmp_path):
    run(tmp_path, "--tree", str(EX / "tree_species.nwk"), "--covariate", "minimumElevationInMeters")
    for f in ("units.tsv", "varpart.tsv", "paired.tsv", "signal.tsv", "community_sites.tsv",
              "community_comparison.tsv", "morphospace_colour.png", "report.md", "summary.json"):
        assert (tmp_path / f).exists(), f
    v = pd.read_csv(tmp_path / "varpart.tsv", sep="\t").iloc[0]
    assert v["phylogeny_only_[c]"] > 0.3 and v["P_phylogeny_given_treatment"] < 0.05
    fr = v[["treatment_only_[a]", "shared_[b]", "phylogeny_only_[c]", "unexplained_[d]"]].sum()
    assert abs(fr - 1) < 1e-9                                     # the four fractions add up to 1
    p = pd.read_csv(tmp_path / "paired.tsv", sep="\t").iloc[0]
    assert p["n_species_in_both"] >= 3 and p["P"] < 0.1           # the simulated within-species shift
    s = pd.read_csv(tmp_path / "signal.tsv", sep="\t")
    assert (s.loc[s.level == "all", "P"] < 0.05).all()
    c = pd.read_csv(tmp_path / "community_comparison.tsv", sep="\t").set_index("metric")
    assert c.loc["SES_MPD", "mean_A"] < c.loc["SES_MPD", "mean_B"]  # primary forest more clustered (simulated)
    sites = pd.read_csv(tmp_path / "community_sites.tsv", sep="\t")
    assert len(sites) == 22 and sites["minimumElevationInMeters"].notna().all()


def test_site_tips_strategy(tmp_path):
    run(tmp_path, "--tree", str(EX / "tree_site_tips.nwk"), "--tip_map", str(EX / "tip_map.csv"))
    sites = pd.read_csv(tmp_path / "community_sites.tsv", sep="\t")
    tm = pd.read_csv(EX / "tip_map.csv")
    assert (sites.set_index("site")["n_in_tree"] == tm.groupby("site").size()).all()   # each site uses its own tips


def test_per_site_tree_file(tmp_path):
    import copy
    root = ph.parse_newick((EX / "tree_species.nwk").read_text())
    meta = pd.read_csv(EX / "metadata_darwin_core_example.csv")
    spp = sorted({s.replace(" ", "_") for s in meta.loc[meta.locationID == "P01", "scientificName"]})
    r = ph.prune(copy.deepcopy(root), set(spp))
    def nwk(n):
        return n.name if not n.kids else "(" + ",".join(f"{nwk(k)}:{k.length}" for k in n.kids) + ")"
    (tmp_path / "P01.nwk").write_text(nwk(r) + ";")
    run(tmp_path / "o", "--site_tree", f"P01={tmp_path / 'P01.nwk'}", "--analyses", "community")
    s = pd.read_csv(tmp_path / "o/community_sites.tsv", sep="\t").set_index("site")
    assert s.loc["P01", "n_in_tree"] == len(spp) and s.loc["P01", "PD"] > 0


def test_faith_pd_and_mpd_on_a_known_tree():
    root = ph.parse_newick("((A:1,B:1):2,(C:2,D:2):1);")
    assert cm.faith_pd(root, {"A", "B"}) == pytest.approx(4.0)      # 1 + 1 + 2 (root edge included)
    assert cm.faith_pd(root, {"A", "C"}) == pytest.approx(6.0)
    T = {t.name: t for t in ph.tips_of(root)}
    D = np.array([[0.0 if x == y else T[x].depth + T[y].depth - 2 * ph.mrca_depth(T[x], T[y]) for y in "ABC"] for x in "ABC"])
    mpd, mntd = cm.mpd_mntd(D)
    assert mpd == pytest.approx((2 + 6 + 6) / 3) and mntd == pytest.approx((2 + 2 + 6) / 3)


def test_coco_input_computes_colour_and_texture(tmp_path):
    from PIL import Image
    meta = pd.read_csv(EX / "metadata_darwin_core_example.csv").head(40)
    meta.to_csv(tmp_path / "meta.csv", index=False)
    imgs, anns = [], []
    rng = np.random.default_rng(0)
    for k, (_, r) in enumerate(meta.iterrows(), 1):
        fn = r["associatedMedia"].replace(".tif", ".png")
        meta.loc[meta.index[k - 1], "associatedMedia"] = fn
        base = np.array([40, 60, 90]) + (30 if "secondary" in r["habitat"] else 0)
        Image.fromarray(np.clip(base + rng.normal(0, 15, (60, 60, 3)), 0, 255).astype(np.uint8)).save(tmp_path / fn)
        imgs.append({"id": k, "file_name": fn, "width": 60, "height": 60})
        anns.append({"id": k, "image_id": k, "category_id": 1, "segmentation": [[5, 5, 55, 5, 55, 55, 5, 55]]})
    meta.to_csv(tmp_path / "meta.csv", index=False)
    (tmp_path / "c.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": [{"id": 1, "name": "head"}]}))
    r = subprocess.run([sys.executable, str(HERE / "descriptron_community_v1.py"), "--metadata", str(tmp_path / "meta.csv"),
                        "--coco", str(tmp_path / "c.json"), "--image_dir", str(tmp_path), "--structure", "head",
                        "--tree", str(EX / "tree_species.nwk"), "--analyses", "signal", "--iters", "19",
                        "--out_dir", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert all(f"traits '{n}'" in r.stdout for n in ("colour_absolute", "colour_pattern", "texture"))
    assert "40 rows linked to 40 specimens" in r.stdout
    r2 = subprocess.run([sys.executable, str(HERE / "descriptron_community_v1.py"), "--metadata", str(tmp_path / "meta.csv"),
                         "--coco", str(tmp_path / "c.json"), "--image_dir", str(tmp_path), "--structure", "head",
                         "--tree", str(EX / "tree_species.nwk"), "--analyses", "morphospace", "--thumbnails",
                         "--iters", "9", "--out_dir", str(tmp_path / "o2")], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert (tmp_path / "o2/morphospace_colour_pattern_thumbnails.png").exists()
    import re as _re
    n = [tuple(map(int, x)) for x in _re.findall(r"thumbnails: (\d+) of (\d+)", r2.stdout)]
    assert n and all(a_ == b_ and a_ > 0 for a_, b_ in n)          # every point drawn as a specimen


def test_missing_column_is_a_clear_error(tmp_path):
    r = subprocess.run([sys.executable, str(HERE / "descriptron_community_v1.py"),
                        "--metadata", str(EX / "metadata_darwin_core_example.csv"), "--factor", "canopy",
                        "--out_dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode != 0 and "no column 'canopy'" in (r.stderr + r.stdout)


def test_colour_homology_table_is_split_into_colour_and_pattern():
    import pandas as pd
    d = pd.DataFrame({"file": ["a.png"], "r1c0_dL_rel": [1.0], "r1c0_da_rel": [0.1], "r1c0_db_rel": [0.2],
                      "r1c0_L_mean_abs": [50.0], "r1c0_a_mean_abs": [1.0], "r1c0_b_mean_abs": [2.0],
                      "r1c0_L_std": [3.0], "chroma_ab": [4.0]})
    cols = [c for c in d.columns if c != "file"]
    sets = cm.split_colour("colour", d, cols)
    assert set(sets) == {"colour_absolute", "colour_pattern"}
    assert set(sets["colour_absolute"][1]) == {"r1c0_L_mean_abs", "r1c0_a_mean_abs", "r1c0_b_mean_abs", "chroma_ab"}
    assert set(sets["colour_pattern"][1]) == {"r1c0_dL_rel", "r1c0_da_rel", "r1c0_db_rel", "r1c0_L_std"}
    assert set(cm.split_colour("texture", d[["file", "r1c0_L_std"]], ["r1c0_L_std"])) == {"texture"}   # not colour: whole


def test_survey_warning(tmp_path):
    # the example is survey-like: no warning
    run(tmp_path / "a", "--tree", str(EX / "tree_species.nwk"), "--analyses", "community", iters="9")
    assert "community_warning" not in json.loads((tmp_path / "a/summary.json").read_text())
    # museum-like: one specimen per site
    meta = pd.read_csv(EX / "metadata_darwin_core_example.csv")
    meta["locationID"] = [f"X{i:03d}" for i in range(len(meta))]
    meta.to_csv(tmp_path / "m.csv", index=False)
    r = subprocess.run([sys.executable, str(HERE / "descriptron_community_v1.py"), "--metadata", str(tmp_path / "m.csv"),
                        "--traits", f"colour={EX / 'traits_colour_example.csv'}", "--tree", str(EX / "tree_species.nwk"),
                        "--analyses", "community", "--iters", "9", "--out_dir", str(tmp_path / "b")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "do not look like community samples" in r.stdout
    assert json.loads((tmp_path / "b/summary.json").read_text())["community_warning"] is True


@pytest.mark.skipif(__import__("shutil").which("Rscript") is None, reason="needs R (vegan, ape, phytools)")
def test_run_matches_r_number_by_number(tmp_path):
    run(tmp_path, "--tree", str(EX / "tree_species.nwk"), "--save_matrices", iters="19")
    assert (tmp_path / "validation_inputs/site_species.tsv").exists()
    r = subprocess.run([sys.executable, str(HERE / "validation" / "validate_community_run_vs_r.py"),
                        "--run_dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    df = pd.read_csv(tmp_path / "validation_vs_R/community_vs_R.tsv", sep="\t")
    assert set(df.panel) >= {"variance partitioning", "community phylogenetics", "phylomorphospace ancestral states"}
    assert df.abs_difference.max() < 1e-8
