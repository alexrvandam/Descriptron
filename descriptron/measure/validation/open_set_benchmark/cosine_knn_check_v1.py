#!/usr/bin/env python3
"""
cosine_knn_check_v1.py - score foundation-model embeddings the way they are normally used (cosine kNN)
=====================================================================================================

The main benchmark scored every embedding with Descriptron's standardised distance (distance to the nearest
species in units of that species' own spread). Foundation-model embeddings are normally compared by cosine
similarity, so a reviewer may ask whether they were judged with the wrong ruler. This script re-scores the same
embeddings with cosine distance under the SAME two hold-outs and the SAME threshold sweep / nested choice
(biorag_congruence_compare_v1.sweep / nested / at_ceiling / mcnemar), and compares them with the matrix scores the
benchmark already wrote.

Per specimen, the embedding of each view is the L2-normalised mean of its images; the similarity of two specimens is
the mean cosine over the views both have. Two novelty scores:
  1nn        1 - similarity to the nearest reference specimen
  prototype  1 - similarity to the nearest species mean (views averaged, re-normalised)
Detection arm: references = every specimen of the OTHER species (the whole species withheld).
False-alarm arm: references = every other specimen (only the specimen withheld).
Naming: leave one specimen out, species of the nearest specimen / nearest prototype.

  python cosine_knn_check_v1.py --embeddings_dir <bench>/embeddings --compare_dir <bench>/compare_all \
      --models bioclip bioclip2 clibd5m --out_dir <bench>/cosine_check
"""
import argparse
import re
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))                     # gui/measure
from biorag_congruence_compare_v1 import sweep, nested, at_ceiling, mcnemar   # noqa: E402


def load_views(npz, keep_ids):
    z = np.load(npz, allow_pickle=True)
    E, sid, view = z["embeddings"], z["specimen_id"].astype(str), z["view"].astype(str)
    views = sorted(set(view))
    out = {}
    for s in keep_ids:
        m = sid == s
        d = {}
        for v in views:
            mv = m & (view == v)
            if mv.any():
                e = E[mv].mean(0); d[v] = e / np.linalg.norm(e)
        out[s] = d
    return out


def sim(a, b):
    common = [v for v in a if v in b]
    return float(np.mean([a[v] @ b[v] for v in common])) if common else np.nan


def prototype(members, emb):
    d = {}
    for v in sorted({v for s in members for v in emb[s]}):
        vs = [emb[s][v] for s in members if v in emb[s]]
        if vs:
            e = np.mean(vs, 0); d[v] = e / np.linalg.norm(e)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings_dir", required=True)
    ap.add_argument("--compare_dir", required=True, help="the benchmark's biorag_congruence_compare_v1 output")
    ap.add_argument("--models", nargs="+", default=["bioclip", "bioclip2", "clibd5m"])
    ap.add_argument("--matrix_sets", nargs="+", default=["size", "ratio", "landmark", "colour"])
    ap.add_argument("--min_sets", type=int, default=2)
    ap.add_argument("--ceilings", nargs="*", type=int, default=[1, 3, 4, 6, 9])
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    cdir, out = Path(a.compare_dir), Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    mdet = pd.read_csv(cdir / "detection_scores.tsv", sep="\t"); mfal = pd.read_csv(cdir / "false_alarm_scores.tsv", sep="\t")
    sp_of = dict(zip(mdet.specimen_id, mdet.species)); ids = list(mdet.specimen_id)
    species = sorted(set(sp_of.values()))
    msw, mcalls = sweep(mdet, mfal, a.matrix_sets, a.min_sets)
    mn = nested(msw, mcalls)
    # control: recomputed from the same score files, the matrix must reproduce the comparison's own nested point
    cmp_ = pd.read_csv(cdir / "congruence_comparison.tsv", sep="\t").set_index("instrument")
    m_ = re.match(r"(\d+)/\d+ caught, (\d+)/\d+ false", str(cmp_.loc["matrix", "nested_point"]))
    assert m_ and (mn["caught"], mn["false_alarms"]) == (int(m_.group(1)), int(m_.group(2))), \
        f"control failed: matrix nested {mn} vs comparison {cmp_.loc['matrix', 'nested_point']}"
    rows, paired, naming = [{"instrument": "matrix (standardised distance)", **mn}], [], []
    for model in a.models:
        emb = load_views(Path(a.embeddings_dir) / f"{model}_image_embeddings.npz", ids)
        S = np.array([[sim(emb[i], emb[j]) for j in ids] for i in ids]); np.fill_diagonal(S, np.nan)
        lab = np.array([sp_of[i] for i in ids])
        det = {"specimen_id": ids, "species": list(lab)}; fal = dict(det)
        d1, f1, dp, fp, n1, npro = [], [], [], [], [], []
        protos_all = {s: prototype([i for i in ids if sp_of[i] == s], emb) for s in species}
        for k, i in enumerate(ids):
            oth = lab != lab[k]
            d1.append(1 - np.nanmax(S[k, oth]))
            f1.append(1 - np.nanmax(S[k]))
            # prototypes: detection = other species' full prototypes; false alarm = own species without the specimen
            dp.append(1 - max(sim(emb[i], protos_all[s]) for s in species if s != lab[k]))
            own_wo = [j for j in ids if sp_of[j] == lab[k] and j != i]
            pr = {s: (prototype(own_wo, emb) if s == lab[k] else protos_all[s]) for s in species}
            pr = {s: p for s, p in pr.items() if p}
            simp = {s: sim(emb[i], p) for s, p in pr.items()}
            fp.append(1 - max(simp.values()))
            n1.append(lab[int(np.nanargmax(S[k]))]); npro.append(max(simp, key=simp.get))
        for score, dv, fv, named in (("1nn", d1, f1, n1), ("prototype", dp, fp, npro)):
            nm = f"{model}_cos_{score}"
            D = pd.DataFrame({**det, nm: dv}); Fd = pd.DataFrame({**fal, nm: fv})
            sw, calls = sweep(D, Fd, [nm], 1)
            ne = nested(sw, calls); rows.append({"instrument": nm, **ne})
            right = int(sum(n == l for n, l in zip(named, lab)))
            naming.append({"identifier": nm, "correct": right, "of": len(ids)})
            for c in a.ceilings:
                pm, pk = at_ceiling(msw, c), at_ceiling(sw, c)
                if pm is None or pk is None:
                    continue
                cm = mcalls[(float(pm["threshold"]), pm["rule"])][0]; ck = calls[(float(pk["threshold"]), pk["rule"])][0]
                om, ok_, p = mcnemar(cm, ck)
                paired.append({"model": nm, "max_false_alarms": c, "matrix_caught": int(cm.sum()), "model_caught": int(ck.sum()),
                               "only_matrix": len(om), "only_model": len(ok_), "mcnemar_exact_p": round(p, 4)})
        print(model, "done", flush=True)
    R = pd.DataFrame(rows); R.to_csv(out / "cosine_novelty_nested.tsv", sep="\t", index=False)
    P = pd.DataFrame(paired); P.to_csv(out / "cosine_paired_vs_matrix.tsv", sep="\t", index=False)
    Nm = pd.DataFrame(naming); Nm.to_csv(out / "cosine_naming_loo.tsv", sep="\t", index=False)
    json.dump({"matrix_control": mn, "n_specimens": len(ids), "n_species": len(species)}, open(out / "cosine_check.json", "w"), indent=2)
    print(R.to_string(index=False)); print(Nm.to_string(index=False))
    print(P.to_string(index=False))


if __name__ == "__main__":
    main()
