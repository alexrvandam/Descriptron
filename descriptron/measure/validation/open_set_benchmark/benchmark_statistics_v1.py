#!/usr/bin/env python3
"""
benchmark_statistics_v1.py - the statistics a reviewer asked for, on the open-set benchmark
==========================================================================================

Specimens are clustered in species, so specimen-level intervals overstate precision. This script adds:

1. SPECIES-CLUSTER BOOTSTRAP (resampling species, with all their specimens) for every instrument:
   naming accuracy, nested recognition (caught / false alarms / margin), and the paired difference to the matrix,
   with percentile 95% intervals; Holm correction over the family of paired tests.
2. SPECIMEN-LEVEL RATES at the nested operating point: of the specimens of a withheld species, how many are
   flagged; of the specimens of known species (each withheld alone), how many are falsely flagged.
3. NAMING BY GROUP: described vs undescribed test species; species present by name in a model's training catalogue
   vs not (does pre-training give a head start?).
4. A STANDARD FROZEN-EMBEDDING BASELINE: logistic regression on the embeddings (PCA inside each fold), refitted inside
   every hold-out - naming by leave-one-specimen-out; recognition by the maximum class probability, scored with
   the same sweep and nested choice as everything else.

Inputs are the benchmark's own outputs, so nothing here re-scores the published instruments.

  python benchmark_statistics_v1.py --bench <bench> --compare compare_all_548 --cosine cosine_check_548 \
      --embeddings embeddings_548 --taxon_profile <profile.yaml> --out <bench>/statistics_548
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent)); sys.path.insert(0, str(HERE))
from biorag_congruence_compare_v1 import sweep, identify, mcnemar          # noqa: E402
import biorag_feature_policy as pol                                          # noqa: E402
import cosine_knn_check_v1 as ck                                             # noqa: E402

MODELS = {"bioclip": "BioCLIP", "bioclip2": "BioCLIP 2", "clibd5m": "CLIBD (BIOSCAN-5M)"}


def nested_per_species(sw, calls, species):
    """as biorag_congruence_compare_v1.nested, but returning the per-species outcome and the chosen point"""
    keys = list(calls.keys())
    Dm = np.array([calls[k][0].reindex(species, fill_value=False).astype(float).values for k in keys])
    Fm = np.array([calls[k][1].reindex(species, fill_value=False).astype(float).values for k in keys])
    out = []
    for j in range(len(species)):
        keep = np.arange(len(species)) != j
        best = int(np.argmax(Dm[:, keep].mean(1) - Fm[:, keep].mean(1)))
        out.append({"species": species[j], "caught": int(Dm[best, j]), "false": int(Fm[best, j]), "point": keys[best]})
    return pd.DataFrame(out).set_index("species")


def specimen_rates(det, fal, cols, min_sets, nps):
    """specimen-level flag rates at each species' nested point"""
    dh = dn = fh = fn = 0
    for sp, r in nps.iterrows():
        t, _rule = r["point"]
        d = det[det["species"] == sp]; f = fal[fal["species"] == sp]
        dflag = ((d[cols] > t).sum(axis=1) >= min_sets); fflag = ((f[cols] > t).sum(axis=1) >= min_sets)
        dh += int(dflag.sum()); dn += len(d); fh += int(fflag.sum()); fn += len(f)
    return dh, dn, fh, fn


def boot(per_species_fn, species, B=4000, seed=1):
    rng = np.random.default_rng(seed)
    sp = np.array(species)
    return np.array([per_species_fn(rng.choice(sp, len(sp), replace=True)) for _ in range(B)])


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def logistic_baseline(emb_npz, ids, sp_of, species, n_pc=20, C=1.0):
    """leave-one-specimen-out naming + max-probability novelty scores (species-withheld / specimen-withheld)"""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    emb = ck.load_views(emb_npz, ids)
    views = sorted({v for s in ids for v in emb[s]})
    dim = len(next(iter(next(iter(emb.values())).values())))
    X = np.zeros((len(ids), len(views) * dim)); M = np.zeros((len(ids), len(views)))
    for i, s in enumerate(ids):
        for k, v in enumerate(views):
            if v in emb[s]:
                X[i, k * dim:(k + 1) * dim] = emb[s][v]; M[i, k] = 1
    X = np.hstack([X, M])                                  # which views are present, so a missing view is not "zero"
    y = np.array([sp_of[s] for s in ids])

    def fit_predict(tr, te):
        p = PCA(n_components=min(n_pc, tr.sum() - 1)).fit(X[tr])
        lr = LogisticRegression(C=C, max_iter=2000).fit(p.transform(X[tr]), y[tr])
        pr = lr.predict_proba(p.transform(X[te]))
        return lr.classes_[pr.argmax(1)], pr.max(1)
    named, fal = {}, {}
    for i, s in enumerate(ids):
        tr = np.ones(len(ids), bool); tr[i] = False
        nm, pm = fit_predict(tr, ~tr)
        named[s] = nm[0]; fal[s] = 1 - pm[0]                 # novelty score = 1 - max probability
    det = {}
    for sp in species:
        te = y == sp; tr = ~te
        _, pm = fit_predict(tr, te)
        for s, p in zip(np.array(ids)[te], pm):
            det[s] = 1 - p
    return named, det, fal


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--compare", default="compare_all_548")
    ap.add_argument("--cosine", default="cosine_check_548")
    ap.add_argument("--embeddings", default="embeddings_548")
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--training_overlap", nargs="*", default=["virg", "zebrana", "fab", "turn", "punctulata"],
                    help="test species present by name in TreeOfLife-200M (from training_data_check)")
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    B = Path(a.bench); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(B / a.compare / "detection_scores.tsv", sep="\t"); fal = pd.read_csv(B / a.compare / "false_alarm_scores.tsv", sep="\t")
    dist = pd.read_csv(B / a.compare / "identification_distances.tsv", sep="\t")
    sp_of = dict(zip(det.specimen_id, det.species)); ids = list(det.specimen_id); species = sorted(set(sp_of.values()))
    profile = pol.load_taxon_profile(a.taxon_profile)
    status = {s: pol.species_status(s, profile) for s in species}
    base = ["size", "ratio", "landmark", "colour"]

    # ── instruments: per-specimen naming correctness, per-species nested recognition ──
    inst_named, inst_nested, inst_spec = {}, {}, {}
    m = identify(dist, base); inst_named["matrix"] = (m["named"] == m["species"]).reindex(ids)
    sw, calls = sweep(det, fal, base, 2); nps = nested_per_species(sw, calls, species)
    assert (int(nps.caught.sum()), int(nps.false.sum())) == (20, 7), "control failed"
    inst_nested["matrix"] = nps; inst_spec["matrix"] = specimen_rates(det, fal, base, 2, nps)
    for k, lab in MODELS.items():
        emb = ck.load_views(B / a.embeddings / f"{k}_image_embeddings.npz", ids)
        protos = {s: ck.prototype([i for i in ids if sp_of[i] == s], emb) for s in species}
        named, dsc, fsc = {}, {}, {}
        for i in ids:
            pr = {}
            for s in species:
                mem = [j for j in ids if sp_of[j] == s and j != i]
                if mem:
                    pr[s] = ck.sim(emb[i], ck.prototype(mem, emb))
            named[i] = max(pr, key=pr.get); fsc[i] = 1 - max(pr.values())
            dsc[i] = 1 - max(ck.sim(emb[i], protos[s]) for s in species if s != sp_of[i])
        inst_named[lab] = pd.Series({i: named[i] == sp_of[i] for i in ids}).reindex(ids)
        D = pd.DataFrame({"specimen_id": ids, "species": [sp_of[i] for i in ids], "s": [dsc[i] for i in ids]})
        F = pd.DataFrame({"specimen_id": ids, "species": [sp_of[i] for i in ids], "s": [fsc[i] for i in ids]})
        sw_, calls_ = sweep(D, F, ["s"], 1); n_ = nested_per_species(sw_, calls_, species)
        inst_nested[lab] = n_; inst_spec[lab] = specimen_rates(D, F, ["s"], 1, n_)
        # logistic-regression baseline on the same embeddings
        nm, dd, ff = logistic_baseline(B / a.embeddings / f"{k}_image_embeddings.npz", ids, sp_of, species)
        lab2 = f"{lab} + logistic regression"
        inst_named[lab2] = pd.Series({i: nm[i] == sp_of[i] for i in ids}).reindex(ids)
        D = pd.DataFrame({"specimen_id": ids, "species": [sp_of[i] for i in ids], "s": [dd[i] for i in ids]})
        F = pd.DataFrame({"specimen_id": ids, "species": [sp_of[i] for i in ids], "s": [ff[i] for i in ids]})
        sw_, calls_ = sweep(D, F, ["s"], 1); n_ = nested_per_species(sw_, calls_, species)
        inst_nested[lab2] = n_; inst_spec[lab2] = specimen_rates(D, F, ["s"], 1, n_)
        print(lab, "done", flush=True)

    # ── bootstrap over species ──
    sp_idx = {s: [i for i in ids if sp_of[i] == s] for s in species}
    rows, pairs = [], []
    for lab in inst_named:
        ok = inst_named[lab]; ne = inst_nested[lab]
        acc_b = boot(lambda S: np.mean(np.concatenate([ok.loc[sp_idx[s]].values for s in S])), species, a.B)
        mar_b = boot(lambda S: ne.loc[S, "caught"].mean() - ne.loc[S, "false"].mean(), species, a.B)
        dh, dn, fh, fn = inst_spec[lab]
        rows.append({"instrument": lab, "named_right": int(ok.sum()), "of": len(ok), "naming_acc": round(ok.mean(), 3),
                     "naming_ci_species_boot": "%.3f-%.3f" % ci(acc_b),
                     "caught": int(ne.caught.sum()), "false_alarms": int(ne.false.sum()),
                     "nested_margin": round(ne.caught.mean() - ne.false.mean(), 3), "margin_ci_species_boot": "%.2f-%.2f" % ci(mar_b),
                     "specimens_of_withheld_species_flagged": f"{dh}/{dn} ({dh / dn:.0%})",
                     "known_specimens_falsely_flagged": f"{fh}/{fn} ({fh / fn:.0%})"})
        if lab != "matrix":
            mo = inst_named["matrix"]; mn = inst_nested["matrix"]
            dacc = boot(lambda S: np.mean(np.concatenate([mo.loc[sp_idx[s]].values for s in S]))
                        - np.mean(np.concatenate([ok.loc[sp_idx[s]].values for s in S])), species, a.B)
            dmar = boot(lambda S: (mn.loc[S, "caught"].mean() - mn.loc[S, "false"].mean())
                        - (ne.loc[S, "caught"].mean() - ne.loc[S, "false"].mean()), species, a.B)
            _, _, p_name = mcnemar(mo, ok)
            pairs.append({"vs matrix": lab, "naming_diff": round(mo.mean() - ok.mean(), 3), "naming_diff_ci": "%.3f-%.3f" % ci(dacc),
                          "naming_boot_p": float(2 * min((dacc <= 0).mean(), (dacc >= 0).mean())), "naming_mcnemar_p": p_name,
                          "margin_diff": round((mn.caught.mean() - mn.false.mean()) - (ne.caught.mean() - ne.false.mean()), 3),
                          "margin_diff_ci": "%.2f-%.2f" % ci(dmar),
                          "margin_boot_p": float(2 * min((dmar <= 0).mean(), (dmar >= 0).mean()))})
    R = pd.DataFrame(rows); P = pd.DataFrame(pairs)
    P["naming_p_holm"] = holm(P["naming_boot_p"].clip(lower=1 / a.B).values)
    P["margin_p_holm"] = holm(P["margin_boot_p"].clip(lower=1 / a.B).values)
    R.to_csv(out / "instruments_species_bootstrap.tsv", sep="\t", index=False)
    P.to_csv(out / "paired_vs_matrix_species_bootstrap.tsv", sep="\t", index=False)

    # ── naming by group ──
    grp = []
    groups = {"described (incl. cf.)": [s for s in species if status[s] in ("described", "cf")],
              "undescribed": [s for s in species if status[s] == "undescribed"],
              "in TreeOfLife-200M by name": [s for s in species if s in a.training_overlap],
              "not in training by name": [s for s in species if s not in a.training_overlap]}
    for lab, ok in inst_named.items():
        r = {"instrument": lab}
        for g, ss in groups.items():
            idx = [i for s in ss for i in sp_idx[s]]
            r[f"{g} ({len(ss)} spp, {len(idx)} specimens)"] = f"{int(ok.loc[idx].sum())}/{len(idx)} ({ok.loc[idx].mean():.0%})"
        grp.append(r)
    G = pd.DataFrame(grp); G.to_csv(out / "naming_by_group.tsv", sep="\t", index=False)
    json.dump({"B": a.B, "species": len(species), "specimens": len(ids), "groups": {g: v for g, v in groups.items()}},
              open(out / "statistics_meta.json", "w"), indent=2)
    pd.set_option("display.width", 250)
    print(R.to_string(index=False)); print(P.to_string(index=False)); print(G.to_string(index=False))


if __name__ == "__main__":
    main()
