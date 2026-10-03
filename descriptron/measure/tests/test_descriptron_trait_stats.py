"""descriptron_trait_stats: PERMANOVA, residualising a covariate, leave-one-out identification, Wilson intervals,
Benjamini-Hochberg, Cramer's V, id matching and the command line.

Values against R (vegan::adonis2, class::knn.cv, prop.test, binom.test, chisq.test) are checked by
validation/validate_trait_stats.py; these tests pin the properties that hold without R.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import descriptron_trait_stats as ts        # noqa: E402


def test_permanova_one_variable_is_anova():
    from scipy.stats import f_oneway
    rng = np.random.default_rng(0)
    y = np.repeat(["a", "b", "c"], 6); x = rng.normal(size=18) + (y == "b") * 1.5
    r2, f, p = ts.permanova(x[:, None], y, 199, 1)
    assert abs(f - f_oneway(*(x[y == g] for g in "abc")).statistic) < 1e-9
    ss = lambda v: ((v - v.mean()) ** 2).sum()
    assert abs(r2 - (1 - sum(ss(x[y == g]) for g in "abc") / ss(x))) < 1e-12


def test_permanova_exact_two_groups():
    """two perfectly separated groups of 3: 2 of the 20 arrangements reach the observed F."""
    X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]]); y = np.array(list("aaabbb"))
    r2, f, p = ts.permanova(X, y, 999, 1)
    assert abs(p - 2 / 20) < 1e-12 and r2 > 0.99


def test_residualise_removes_a_covariate_effect():
    rng = np.random.default_rng(1)
    y = np.repeat(["a", "b"], 20); c = (y == "b").astype(float) + rng.normal(0, 0.01, 40)
    X = np.column_stack([3 * c, -2 * c]) + rng.normal(0, 0.01, (40, 2))
    assert ts.permanova(X, y, 99, 1)[0] > 0.99
    assert ts.permanova(ts.residualise(X, c[:, None]), y, 99, 1)[0] < 0.2


def test_loo_1nn_separated_clusters():
    rng = np.random.default_rng(2)
    y = np.repeat(["a", "b", "c", "solo"], [5, 5, 5, 1])
    X = np.vstack([rng.normal(m, 0.1, (n, 3)) for m, n in ((0, 5), (5, 5), (10, 5), (20, 1))])
    hits = ts.loo_1nn(X, y)
    assert sum(h[0] for h in hits) == 15 and all(h[1] for h in hits if h[0])
    assert hits[-1] == (False, False, None)          # a single specimen cannot be scored


def test_wilson_matches_r():
    lo, hi = ts.wilson(40, 48)                       # R: prop.test(40, 48, correct = FALSE)$conf.int
    assert abs(lo - 0.7042219) < 1e-6 and abs(hi - 0.9130453) < 1e-6


def test_bh_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.array([0.001, 0.04, 0.03, 0.5, 0.2, 0.01])
    assert np.allclose(ts.bh(p), multipletests(p, method="fdr_bh")[1])


def test_cramers_v_perfect_association():
    import pandas as pd
    v, p, n = ts.cramers_v_perm(pd.Series(list("xxxyyyzzz")), np.array(list("aaabbbccc")), 199, 1)
    assert abs(v - 1) < 1e-12 and n == 9 and p < 0.05


def test_id_forms_match_suffix_and_extension():
    m = {k: "sp1" for k in ts._key_forms("wing01.tif")}
    assert ts.group_of("wing01.tif_43", m) == "sp1" and ts.group_of("wing01", m) == "sp1"


def test_cli_end_to_end(tmp_path):
    rng = np.random.default_rng(3); rows, groups, cats = [], [], []
    for s, shift in (("sp1", 0), ("sp2", 3), ("sp3", 6)):
        for k in range(6):
            fn = f"{s}_{k}.png"
            base = rng.normal(shift, 1, 4)
            rows.append([f"{fn}_{k + 1}"] + list(base) + [100 + shift + rng.normal(), 128 + rng.normal(), 128 + rng.normal()])
            groups.append([fn, s]); cats.append([fn, "dark" if s == "sp3" else "pale"])
    hdr = ["filename", "r0c0_dL_rel", "r0c1_dL_rel", "r1c0_dL_rel", "r1c1_dL_rel", "r0c0_L_mean_abs", "r0c0_a_mean_abs", "r0c0_b_mean_abs"]
    for fn, h, r in (("colour.csv", hdr, rows), ("groups.csv", ["filename", "group_label"], groups),
                     ("states.csv", ["filename", "tone"], cats)):
        with open(tmp_path / fn, "w", newline="") as f:
            w = csv.writer(f); w.writerow(h); w.writerows(r)
    out = tmp_path / "out"
    ts.main(["--groups", str(tmp_path / "groups.csv"), "--traits", f"colour={tmp_path / 'colour.csv'}",
             "--categorical", f"states={tmp_path / 'states.csv'}", "--permutations", "99", "--out_dir", str(out)])
    summ = list(csv.DictReader(open(out / "summary.csv")))
    assert [r["trait_set"] for r in summ] == ["colour", "states"]
    assert int(summ[0]["n_specimens"]) == 18 and "permanova_R2_beyond_mean_colour" in summ[0]
    assert float(summ[0]["permanova_P"]) <= 0.02
    for fn in ("per_species.csv", "pairwise_permanova.csv", "species_validation.csv", "categorical_characters.csv",
               "separation_by_trait_set.csv", "mcnemar_between_trait_sets.csv", "trait_stats_report.md",
               "pca_colour.png", "identification_by_trait_set.png", "species_separation.png"):
        assert (out / fn).exists(), fn
    cat = list(csv.DictReader(open(out / "categorical_characters.csv")))
    assert cat[0]["character"] == "tone" and float(cat[0]["cramers_V"]) == pytest.approx(1.0)


def test_fit_ratio_marks_a_specimen_closer_to_another_species():
    X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [0.5]]); y = np.array(list("aaabbb"))    # last "b" sits next to the "a"s
    r = ts.fit_ratio(X, y)
    assert all(v < 1 for v in r[:5]) and r[5] > 1


def test_specimen_flags_find_a_mislabelled_specimen(tmp_path):
    """a hypothesis (group labels from DNA, field notes...) with one specimen put in the wrong species: that
    specimen comes out 're-examine', named as the species it really belongs to; the others 'fits'."""
    rng = np.random.default_rng(4); rows, groups = [], []
    for s, shift in (("sp1", 0), ("sp2", 6), ("sp3", 12)):
        for k in range(6):
            fn = f"{s}_{k}.png"
            rows.append([fn] + list(rng.normal(shift, 0.5, 4)))
            groups.append([fn, "sp2" if (s, k) == ("sp1", 0) else s])          # sp1_0 mislabelled as sp2
    for fn, h, r in (("traits.csv", ["filename", "f1", "f2", "f3", "f4"], rows),
                     ("groups.csv", ["filename", "group_label"], groups)):
        with open(tmp_path / fn, "w", newline="") as f:
            w = csv.writer(f); w.writerow(h); w.writerows(r)
    out = tmp_path / "out"
    ts.main(["--groups", str(tmp_path / "groups.csv"), "--traits", f"shape={tmp_path / 'traits.csv'}",
             "--permutations", "99", "--no_pairwise", "--out_dir", str(out)])
    flags = {r["specimen_id"]: r for r in csv.DictReader(open(out / "specimen_flags.csv"))}
    bad = flags["sp1_0.png"]
    assert bad["status"] == "re-examine" and bad["assigned_species"] == "sp2" and bad["named_elsewhere_as"] == "sp1"
    assert float(bad["fit_ratio_shape"]) > 1
    assert all(r["status"] == "fits" for k, r in flags.items() if k != "sp1_0.png")
