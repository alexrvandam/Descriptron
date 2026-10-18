#!/usr/bin/env python3
"""Imaging-batch correction for homology / texture / colour feature tables (added 12 Oct 2026; opt-in).

Specimens photographed in different sessions (camera, magnification, lighting) can differ more by session than by
species. This shifts every non-reference batch onto the reference batch by the mean per-feature difference between the
two, estimated ONLY from species photographed in both batches (species-matched, so true species differences between
batches do not leak into the correction). Tested on 65 Austromyrma heads (two montage sessions, 8 shared species): the
same-species nearest neighbour across sessions rose from 0.04 to 0.36 on TPS grid cells, and the one specimen imaged in
both sessions found its own other photo first on band and whole-table features; rescaling the spread as well (location-
scale) was worse (ant_dorsal/texture_v1/PUBLICATION_texture_sculpture_20261012, result 8).

Usage:
  homology_batch_meanshift_v1.py --features texture_homology_features_head.csv --metadata meta.csv \
      --batch_column session --species_column species [--reference M2] [--evaluate] --output_dir out/
metadata: one row per image, a 'filename' column matching the features table (basename match), plus the batch and
species columns. Writes <features>_batchcorrected.csv, batch_shift_report.txt, batch_shift_per_feature.csv and, with
--evaluate, batch_shift_evaluation.csv (leave-one-species-out: the query's species never contributes to the shift).
"""
import argparse, os, logging
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def shift_vector(X, sp, bt, batch, ref, exclude=None):
    shared = sorted(set(sp[bt == batch]) & set(sp[bt == ref]) - ({exclude} if exclude is not None else set()))
    if not shared:
        return None, shared
    d = np.mean([np.nanmean(X[(sp == s) & (bt == ref)], 0) - np.nanmean(X[(sp == s) & (bt == batch)], 0) for s in shared], 0)
    return np.nan_to_num(d), shared


def correct(X, sp, bt, ref, exclude=None):
    Y = X.copy(); info = {}
    for b in sorted(set(bt) - {ref}):
        d, shared = shift_vector(X, sp, bt, b, ref, exclude)
        info[b] = shared
        if d is not None:
            Y[bt == b] += d
    return Y, info


def evaluate(X, sp, bt, ref):
    """Nearest other image is the same species? (a) no correction, (b) shift; neighbour forced from another batch."""
    rows = []
    for mode in ('none', 'mean shift'):
        ok = tot = 0
        for q in sorted(set(sp)):
            Y = X if mode == 'none' else correct(X, sp, bt, ref, exclude=q)[0]
            mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0) + 1e-9; Z = np.nan_to_num((Y - mu) / sd)
            for i in np.where(sp == q)[0]:
                m = (bt != bt[i]); m[i] = False
                if not (m & (sp == q)).any():
                    continue
                j = np.where(m)[0][np.argmin(np.linalg.norm(Z[m] - Z[i], axis=1))]; ok += sp[j] == q; tot += 1
        rows.append(dict(correction=mode, cross_batch_same_species_nn=round(ok / tot, 3) if tot else np.nan, n_queries=tot))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', required=True); ap.add_argument('--metadata', required=True)
    ap.add_argument('--batch_column', required=True); ap.add_argument('--species_column', default='species')
    ap.add_argument('--filename_column', default='filename'); ap.add_argument('--reference', default='',
                    help='Reference batch (default: the batch with most images)')
    ap.add_argument('--evaluate', action='store_true'); ap.add_argument('--output_dir', required=True)
    a = ap.parse_args(); os.makedirs(a.output_dir, exist_ok=True)
    F = pd.read_csv(a.features); M = pd.read_csv(a.metadata, sep='\t' if a.metadata.endswith('.tsv') else ',')
    M['_bn'] = M[a.filename_column].astype(str).map(os.path.basename); F['_bn'] = F[a.filename_column].astype(str).map(os.path.basename)
    D = F.merge(M[['_bn', a.batch_column, a.species_column]], on='_bn', how='left')
    miss = D[a.batch_column].isna() | D[a.species_column].isna()
    if miss.any():
        logging.warning(f'{miss.sum()} images without batch/species in the metadata are left uncorrected')
    num = [c for c in F.columns if c not in (a.filename_column, '_bn') and pd.api.types.is_numeric_dtype(F[c])]
    ok = ~miss.values; X = D.loc[ok, num].to_numpy(float); sp = D.loc[ok, a.species_column].astype(str).to_numpy(); bt = D.loc[ok, a.batch_column].astype(str).to_numpy()
    ref = a.reference or pd.Series(bt).value_counts().index[0]
    Y, info = correct(X, sp, bt, ref)
    out = F.drop(columns='_bn').copy(); out.loc[ok, num] = Y
    base = os.path.splitext(os.path.basename(a.features))[0]
    out.to_csv(os.path.join(a.output_dir, f'{base}_batchcorrected.csv'), index=False)
    per = pd.DataFrame({'feature': num})
    lines = [f'Reference batch: {ref} ({(bt == ref).sum()} images)']
    for b, shared in info.items():
        d, _ = shift_vector(X, sp, bt, b, ref)
        if d is None:
            lines.append(f'Batch {b}: NOT corrected - no species shared with {ref}')
            continue
        per[f'shift_{b}'] = d
        sdv = np.nanstd(X, 0) + 1e-9
        lines.append(f'Batch {b}: {(bt == b).sum()} images, shifted using {len(shared)} shared species ({", ".join(shared)}); '
                     f'mean |shift| = {np.mean(np.abs(d) / sdv):.2f} SD per feature (max {np.max(np.abs(d) / sdv):.2f})')
    per.to_csv(os.path.join(a.output_dir, 'batch_shift_per_feature.csv'), index=False)
    if a.evaluate:
        ev = evaluate(X, sp, bt, ref); ev.to_csv(os.path.join(a.output_dir, 'batch_shift_evaluation.csv'), index=False)
        lines += ['', 'Leave-one-species-out evaluation (nearest image from another batch is the same species):'] + \
                 [f'  {r.correction}: {r.cross_batch_same_species_nn} (n = {r.n_queries})' for r in ev.itertuples()]
    open(os.path.join(a.output_dir, 'batch_shift_report.txt'), 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
