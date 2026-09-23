#!/usr/bin/env python3
"""
biorag_key_feature_filter_v2.py — build the human-checkable data matrix
======================================================================

v2 of biorag_key_feature_filter.py (v1 kept unchanged). v1 only dropped
columns by name; v2 also REPAIRS the values that a taxonomist would read:

  1. Colour: OpenCV 8-bit LAB (L* x 2.55, a*/b* + 128) is converted to true
     CIE L*a*b*; chroma C*ab and hue h_ab are recomputed from the centred
     values (the compiled colhom_chroma_ab / hue columns were computed on
     uncentred values and are discarded). Orientation-free regional colour
     (darkest / palest third, contrast between the two end thirds) is derived
     from the TPS-aligned colour grid.
  2. Landmarks: Procrustes-scaled distances (unit centroid size) are turned
     into (a) scale-free ratios to the longest landmark span, measurable with
     an ocular micrometer, and (b) mm distances where a centroid size in mm
     exists (distance x centroid size is exact: the GPA only centres, scales
     and rotates).
  3. Ratios: the standard proportions are RECOMPUTED per specimen with the
     specimen-ID rule from the taxon profile. (compile_specimen_data.py's
     default specimen regex merged all specimens of names such as
     'morph_22_3' into one ID and missed head/wing matches.)
  4. Sex-dimorphic structures listed in the profile (split_by_sex) are split
     into male and female structures instead of being pooled.
  5. Every column is assigned an evidence tier by biorag_feature_policy.py.

Outputs (in --output_dir):
  specimen_matrix_long.csv       Tier-1 values per specimen (THE checkable matrix)
  feature_dictionary.tsv         feature_id -> structure, label, unit, tier, definition
  species_feature_summary.csv    per species x feature: n, min, max, mean, sd, median
  coverage_species_by_structure.tsv
  <prefix>_full_features.csv     image-level, Tier-1 + meta columns (for v1 tools)
  <prefix>_species_summary.csv   v1 long format rebuilt from the specimen matrix
  <prefix>_diagnostic_report.json   source report restricted to Tier-1 features
  filter_report_v2.json          provenance: encodings, ratio audit, reference spans
  biorag_cache_filtered/         (optional, --cache_dir) v1 diagnosis-text filter

Usage:
  python biorag_key_feature_filter_v2.py \
      --compiled_dir "/path/Diaphorina_compiled_29species" \
      --output_dir   "/path/Diaphorina_monograph/compiled_key_tier" \
      --taxon_profile biorag_prompts/taxon_profiles/diaphorina_taxon_profile.yaml
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import biorag_feature_policy as pol  # noqa: E402

FILTER_VERSION = "2.1"


def _find_prefix(compiled_dir: Path, prefix: str = None) -> str:
    if prefix:
        return prefix
    hits = sorted(compiled_dir.glob("*_full_features.csv"))
    if not hits:
        raise FileNotFoundError(f"No *_full_features.csv in {compiled_dir}")
    return hits[0].name[: -len("_full_features.csv")]


def merge_category_aliases(df: pd.DataFrame, profile: dict, report: dict) -> pd.DataFrame:
    """One structure stored under two category names (e.g. 'cell-c+sc' from the
    measurement step and 'cell-csc' from a step that stripped '+' from folder
    names). Aliases come from profile['category_aliases'] and from automatic
    detection: names equal after removing non-alphanumerics, rows on the same
    images, and (almost) no column filled in both. Rows are merged per image
    into the canonical name; values already present in the canonical row win."""
    aliases = dict(profile.get("category_aliases") or {})
    cats = sorted(df["category"].dropna().unique())
    groups = defaultdict(list)
    for c in cats:
        groups[re.sub(r'[^a-z0-9]', '', str(c).lower())].append(c)
    meta = {"image_base", "category", "group_label"}
    auto = {}
    for names in groups.values():
        if len(names) < 2:
            continue
        canon = max(names, key=lambda n: (len(n), n))
        for other in names:
            if other == canon or other in aliases:
                continue
            a = df[df["category"] == canon].set_index("image_base")
            b = df[df["category"] == other].set_index("image_base")
            shared = a.index.intersection(b.index)
            if len(shared) < 0.5 * min(len(a), len(b)):
                continue
            ca = {c for c in a.columns if c not in meta and not c.startswith("has_") and a[c].notna().any()}
            cb = {c for c in b.columns if c not in meta and not c.startswith("has_") and b[c].notna().any()}
            both = {c for c in ca & cb if not c.startswith(("ratio_", "scale_"))}
            if len(both) <= 0.05 * max(1, min(len(ca), len(cb))):
                auto[other] = canon
    aliases.update(auto)
    log = {}
    for other, canon in aliases.items():
        if other not in cats:
            continue
        rows_o = df[df["category"] == other]
        n_merged = n_renamed = 0
        for idx, r in rows_o.iterrows():
            tgt = df.index[(df["category"] == canon) & (df["image_base"] == r["image_base"])]
            if len(tgt):
                t = tgt[0]
                fill = [c for c in df.columns if c not in meta and pd.isna(df.at[t, c]) and pd.notna(r[c])]
                df.loc[t, fill] = r[fill].values
                for c in df.columns:
                    if c.startswith("has_") and r[c] and not df.at[t, c]:
                        df.at[t, c] = r[c]
                df = df.drop(index=idx)
                n_merged += 1
            else:
                df.at[idx, "category"] = canon
                n_renamed += 1
        log[f"{other} -> {canon}"] = {"rows_merged": n_merged, "rows_renamed": n_renamed,
                                      "source": "auto-detected" if other in auto else "taxon profile"}
    report["category_aliases_merged"] = log
    return df


def add_identity(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    df = df.copy()
    df["sex"] = df["image_base"].map(lambda s: pol.specimen_sex(s, profile))
    df["specimen_id"] = [pol.specimen_id(ib, sp, profile)
                         for ib, sp in zip(df["image_base"], df["group_label"])]
    excluded = [c for c, v in profile.get("structures", {}).items()
                if isinstance(v, dict) and v.get("exclude")]
    df = df[~df["category"].isin(excluded)].copy()
    df["base_category"] = df["category"]
    for cat, info in profile.get("structures", {}).items():
        if isinstance(info, dict) and info.get("split_by_sex"):
            m = (df["category"] == cat) & df["sex"].isin(["male", "female"])
            df.loc[m, "category"] = [pol.sex_split_category(cat, s) for s in df.loc[m, "sex"]]
    return df


def _norm_name(x) -> str:
    x = str(x).lower().replace('.tif', '').replace('+', 'plus')
    return re.sub(r'[^a-z0-9]', '', x)


def apply_exclusions(df: pd.DataFrame, path: str, report: dict) -> pd.DataFrame:
    """Drop image x structure rows listed in an exclusion CSV
    (columns: image_filename, category_name[, reason])."""
    ex = pd.read_csv(path)
    keys = {(_norm_name(i), _norm_name(c)): r for i, c, r in
            zip(ex["image_filename"], ex["category_name"],
                ex.get("reason", pd.Series([""] * len(ex))))}
    k = [(_norm_name(i), _norm_name(c)) for i, c in zip(df["image_base"], df["base_category"])]
    hit = pd.Series([x in keys for x in k], index=df.index)
    report["exclusions"] = {"list": str(path), "entries": len(ex), "rows_removed": int(hit.sum()),
                            "removed": [f"{i} | {c}" for i, c in
                                        df.loc[hit, ["image_base", "category"]].values.tolist()],
                            "entries_not_found (already absent from the compiled matrix)":
                                sorted(f"{a} | {b}" for a, b in set(keys) - set(k))}
    return df[~hit].copy()


def screen_outliers(df: pd.DataFrame, out_path: Path, report: dict,
                    z_max: float = 6.0, rel_max: float = 0.5) -> pd.DataFrame:
    """FLAG (never drop) size values far outside their species' range, and
    suspected label swaps between two structures of the same image.
    Robust z = |x - median| / (1.4826 * MAD) within species x structure (n >= 4)."""
    cols = [c for c in ("meas_length_mm", "meas_height_mm", "meas_area_mm2") if c in df.columns]
    rows = []
    stats = {}
    for (sp, cat), g in df.groupby(["group_label", "category"]):
        for c in cols:
            v = g[c].dropna()
            if len(v) < 4:
                continue
            med = float(v.median())
            mad = float((v - med).abs().median()) * 1.4826 or 1e-12
            stats[(sp, cat, c)] = (float(v.min()), float(v.max()), med, mad)
            for idx, x in v.items():
                z = abs(x - med) / mad
                if z > z_max and abs(x - med) > rel_max * abs(med):
                    rows.append({"image_base": df.at[idx, "image_base"], "species": sp,
                                 "category": cat, "feature": c, "value": x,
                                 "species_median": med, "robust_z": round(z, 1)})
    flags = pd.DataFrame(rows)
    if len(flags):
        # label-swap check: flagged value fits another structure of the same image
        notes = []
        for r in flags.itertuples():
            same = df[(df["image_base"] == r.image_base) & (df["category"] != r.category)]
            hint = ""
            for oc, ov in zip(same["category"], same[r.feature]):
                st_other = stats.get((r.species, oc, r.feature))
                st_self = stats.get((r.species, r.category, r.feature))
                if st_other and st_self and ov == ov and \
                        st_other[0] <= r.value <= st_other[1] and st_self[0] * 0.8 <= ov <= st_self[1] * 1.2:
                    hint = f"possible label swap with {oc}"
            notes.append(hint)
        flags["note"] = notes
    flags.to_csv(out_path, sep="\t", index=False)
    report["outlier_screen"] = {"rule": f"robust z > {z_max} and deviation > {int(rel_max*100)}% of species median",
                                "n_flagged": int(len(flags)), "file": str(out_path),
                                "note": "flagged values are removed only with --exclude_flagged; "
                                        "add confirmed errors to the exclusion list and fix them in the annotations"}
    return flags


def drop_flagged(df: pd.DataFrame, flags: pd.DataFrame, report: dict,
                 whole_image_min: int = 3) -> pd.DataFrame:
    """Remove flagged image x structure rows; remove the whole image when
    >= whole_image_min of its structures are flagged (wrong-scale signature)."""
    if flags is None or not len(flags):
        report["flagged_rows_removed"] = {"whole_images": [], "rows_removed": 0}
        return df
    per_img = flags.groupby("image_base")["category"].nunique()
    bad_imgs = set(per_img[per_img >= whole_image_min].index)
    pairs = set(zip(flags["image_base"], flags["category"]))
    m = df["image_base"].isin(bad_imgs) | pd.Series(
        [(i, c) in pairs for i, c in zip(df["image_base"], df["category"])], index=df.index)
    report["flagged_rows_removed"] = {"whole_images": sorted(bad_imgs), "rows_removed": int(m.sum())}
    return df[~m].copy()


def add_cie_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    if "colhom_L_mean" not in df.columns:
        report["colour"] = "no colhom_L_mean column — colour features skipped"
        return df
    enc = pol.detect_lab_encoding(df["colhom_L_mean"].values, df["colhom_a_mean"].values)
    report["colour_encoding"] = enc
    # Empty cells: some colour-homology runs write L=a=b=0 where the mask does not
    # reach a grid cell. With OpenCV encoding a*=b*=0 is not a real colour -> missing.
    if enc["ab_offset"]:
        n_empty = 0
        trip = [("colhom_L_mean", "colhom_a_mean", "colhom_b_mean")]
        for c in df.columns:
            m = re.match(r'^(colhom_r\d+c\d+)_L_mean_abs$', c)
            if m:
                trip.append((c, f"{m.group(1)}_a_mean_abs", f"{m.group(1)}_b_mean_abs"))
        for Lc, ac, bc in trip:
            if ac in df.columns and bc in df.columns:
                z = (df[Lc] == 0) & (df[ac] == 0) & (df[bc] == 0)
                n_empty += int(z.sum())
                df.loc[z, [Lc, ac, bc]] = np.nan
        report["colour_empty_cells_set_missing"] = n_empty
    # Copied colour: two structures of the same image with identical whole-structure
    # L*, a*, b* (a failed colour run can reuse another structure's values).
    # The copy is the structure whose colour grid is empty; if neither grid is
    # empty both are blanked (cannot tell which is right).
    grid_L = [c for c in df.columns if re.match(r'^colhom_r\d+c\d+_L_mean_abs$', c)]
    key_cols = ["colhom_L_mean", "colhom_a_mean", "colhom_b_mean"]
    dup_log = []
    sub = df.dropna(subset=key_cols)
    for img, g in sub.groupby("image_base"):
        if len(g) < 2:
            continue
        for _, gg in g.groupby(key_cols):
            if len(gg) < 2:
                continue
            empty = [i for i in gg.index if grid_L and df.loc[i, grid_L].isna().all()]
            bad = empty if 0 < len(empty) < len(gg) else list(gg.index)
            for i in bad:
                df.loc[i, key_cols + [c for c in df.columns if c.startswith("colhom_r")]] = np.nan
                dup_log.append(f"{img} | {df.loc[i, 'category']}")
    report["colour_copied_values_removed"] = dup_log
    L = df["colhom_L_mean"] * enc["L_scale"]
    a = df["colhom_a_mean"] - enc["ab_offset"]
    b = df["colhom_b_mean"] - enc["ab_offset"]
    df["cie_L"], df["cie_a"], df["cie_b"] = L, a, b
    df["cie_C"] = np.hypot(a, b)
    df["cie_h"] = np.degrees(np.arctan2(b, a)) % 360.0

    grid = defaultdict(list)
    for c in df.columns:
        m = re.match(r'^colhom_r(\d+)c(\d+)_L_mean_abs$', c)
        if m:
            grid[int(m.group(2))].append(c)
    if grid:
        ncol = max(grid) + 1
        k = max(1, ncol // 3)
        thirds = {
            "first": [c for j in range(0, k) for c in grid.get(j, [])],
            "middle": [c for j in range(k, ncol - k) for c in grid.get(j, [])],
            "last": [c for j in range(ncol - k, ncol) for c in grid.get(j, [])],
        }
        means = {t: df[cols].mean(axis=1) * enc["L_scale"] for t, cols in thirds.items() if cols}
        tm = pd.DataFrame(means)
        df["cie_L_end_contrast"] = (tm["first"] - tm["last"]).abs()
        df["cie_L_darkest_third"] = tm.min(axis=1)
        df["cie_L_palest_third"] = tm.max(axis=1)
        report["colour_grid"] = {"n_columns": ncol, "third_width_columns": k,
                                 "note": "orientation-free: grid is TPS-aligned but the "
                                         "proximal/distal direction is not asserted"}
    return df


def add_landmark_columns(df: pd.DataFrame, profile: dict, report: dict):
    dist_cols = [c for c in df.columns if re.match(r'^lmk_dist_\d+_\d+$', c)]
    refs = {}
    if not dist_cols:
        return df, refs
    new_cols = {}
    lm_rows = df[dist_cols].notna().any(axis=1)
    for cat in sorted(df.loc[lm_rows, "base_category"].unique()):
        mask = lm_rows & (df["base_category"] == cat)
        sub = df.loc[mask, dist_cols]
        present = [c for c in dist_cols if sub[c].notna().sum() >= max(3, 0.5 * len(sub))]
        if not present:
            continue
        ls = profile.get("landmark_sets", {}).get(cat, {})
        rp = ls.get("reference_pair", "auto")
        if isinstance(rp, (list, tuple)) and len(rp) == 2:
            ref_col = f"lmk_dist_{min(rp)}_{max(rp)}"
        else:
            ref_col = sub[present].median().idxmax()
        i0, j0 = ref_col.split("_")[2:4]
        refs[cat] = (int(i0), int(j0))
        ref = df.loc[mask, ref_col]
        cs = df.loc[mask, "lmk_centroid_size_mm"] if "lmk_centroid_size_mm" in df.columns else None
        for c in present:
            i, j = c.split("_")[2:4]
            if c != ref_col:
                new_cols.setdefault(f"lmkrel_{i}_{j}", pd.Series(np.nan, index=df.index))
                new_cols[f"lmkrel_{i}_{j}"].loc[mask] = df.loc[mask, c] / ref
            if cs is not None:
                new_cols.setdefault(f"lmkmm_{i}_{j}", pd.Series(np.nan, index=df.index))
                new_cols[f"lmkmm_{i}_{j}"].loc[mask] = df.loc[mask, c] * cs
        report.setdefault("landmarks", {})[cat] = {
            "reference_pair": [int(i0), int(j0)],
            "reference_rule": "fixed in profile" if isinstance(rp, (list, tuple)) else
                              "auto: largest median Procrustes distance (longest span)",
            "n_rows": int(mask.sum()),
            "n_rows_with_mm_scale": int(cs.notna().sum()) if cs is not None else 0,
            "n_distance_features": len(present),
        }
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    return df, refs


def recompute_ratios(df: pd.DataFrame, profile: dict, report: dict) -> pd.DataFrame:
    """Specimen-level standard proportions -> long rows (category 'proportions')."""
    abbr = profile.get("measurement_abbreviations", {})
    ratios = profile.get("ratios", [])
    if not abbr or not ratios:
        report["ratios"] = "no ratio definitions in taxon profile"
        return pd.DataFrame()
    spec = df.groupby(["group_label", "specimen_id", "base_category"]).agg(
        length_mm=("meas_length_mm", "mean"), height_mm=("meas_height_mm", "mean")).reset_index()
    spec_sex = (df[df["sex"] != "unknown"].groupby("specimen_id")["sex"]
                .agg(lambda s: s.value_counts().index[0]))
    lookup = {(r.specimen_id, r.base_category): r for r in spec.itertuples()}
    specimens = df[["group_label", "specimen_id"]].drop_duplicates()

    def val(sid, key, sex):
        cfg = abbr.get(key)
        if not cfg:
            return None
        if cfg.get("sex") and cfg["sex"] != sex:
            return None
        r = lookup.get((sid, cfg["category"]))
        if r is None:
            return None
        v = getattr(r, cfg["measurement"])
        return None if v is None or v != v or v <= 0 else float(v)

    rows = []
    for sp, sid in specimens.itertuples(index=False):
        sex = spec_sex.get(sid, "unknown")
        for rd in ratios:
            if rd.get("sex", "all") != "all" and rd["sex"] != sex:
                continue
            parts = [p.strip() for p in str(rd["num"]).split("+")]
            nums = [val(sid, p, sex) for p in parts]
            den = val(sid, rd["den"], sex)
            if den and all(v is not None for v in nums):
                rows.append({"group_label": sp, "specimen_id": sid, "sex": sex,
                             "category": "proportions", "base_category": "proportions",
                             "column": f"ratio_{rd['id']}", "value": sum(nums) / den,
                             "n_images": 1})
    out = pd.DataFrame(rows)

    # audit against the compiled ratio columns
    audit = {}
    old_cols = [c for c in df.columns if c.startswith("ratio_")]
    for c in old_cols:
        old = df.groupby("specimen_id")[c].first().dropna()
        new = out[out["column"] == c].set_index("specimen_id")["value"] if len(out) else pd.Series(dtype=float)
        both = old.index.intersection(new.index)
        diff = (old[both] - new[both]).abs() > 1e-4 * np.maximum(1, new[both].abs())
        audit[c] = {"specimens_old": int(len(old)), "specimens_new": int(len(new)),
                    "both": int(len(both)), "values_differ": int(diff.sum()),
                    "only_old": int(len(old.index.difference(new.index))),
                    "only_new": int(len(new.index.difference(old.index)))}
    report["ratio_audit_vs_compiled"] = audit
    return out


def main():
    ap = argparse.ArgumentParser(description="Build the human-checkable (Tier-1) data matrix")
    ap.add_argument("--compiled_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--taxon_profile", required=True)
    ap.add_argument("--file_prefix", default=None,
                    help="Prefix of <prefix>_full_features.csv (auto-detected)")
    ap.add_argument("--exclude_list", default=None,
                    help="CSV of annotations to drop (image_filename, category_name, reason)")
    ap.add_argument("--exclude_flagged", action="store_true",
                    help="Drop rows flagged by the outlier/label-swap screen (see outlier_flags.tsv)")
    ap.add_argument("--cache_dir", default=None,
                    help="Optional BioRAG cache to copy with computational sentences removed (v1 filter)")
    args = ap.parse_args()

    src, dst = Path(args.compiled_dir), Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    prefix = _find_prefix(src, args.file_prefix)
    profile = pol.load_taxon_profile(args.taxon_profile)
    report = {"filter_version": FILTER_VERSION, "policy_version": pol.POLICY_VERSION,
              "generated": datetime.now().isoformat(), "compiled_dir": str(src),
              "taxon_profile": profile.get("_path"), "file_prefix": prefix}

    df = pd.read_csv(src / f"{prefix}_full_features.csv", low_memory=False)
    report["input_shape"] = list(df.shape)
    df = merge_category_aliases(df, profile, report)
    if report["category_aliases_merged"]:
        print(f"Category aliases merged: {report['category_aliases_merged']}")
    df = add_identity(df, profile)
    if args.exclude_list:
        df = apply_exclusions(df, args.exclude_list, report)
        print(f"Exclusions: {report['exclusions']['rows_removed']} rows removed; "
              f"{len(report['exclusions']['entries_not_found (already absent from the compiled matrix)'])} "
              f"list entries already absent from the compiled matrix")
    flags = screen_outliers(df, dst / "outlier_flags.tsv", report)
    print(f"Outlier screen: {report['outlier_screen']['n_flagged']} values flagged -> outlier_flags.tsv")
    if args.exclude_flagged:
        df = drop_flagged(df, flags, report)
        print(f"  --exclude_flagged: {report['flagged_rows_removed']['rows_removed']} rows removed "
              f"(whole images: {len(report['flagged_rows_removed']['whole_images'])})")
    df = add_cie_columns(df, report)
    df, refs = add_landmark_columns(df, profile, report)

    tiers = {c: pol.classify_column(c) for c in df.columns}
    tier1_cols = [c for c, (t, _f) in tiers.items() if t in pol.TIER1 and not c.startswith("ratio_")]
    counts = defaultdict(int)
    for c, (t, _f) in tiers.items():
        counts[t] += 1
    report["columns_by_tier"] = dict(counts)
    report["excluded_columns"] = sorted(c for c, (t, _f) in tiers.items() if t == pol.TIER_EXCLUDE)

    # ---- long Tier-1 matrix: mean over images of the same specimen+structure
    g = df.groupby(["group_label", "specimen_id", "category", "base_category"])
    agg = g[tier1_cols].mean()
    nimg = g.size().rename("n_images")
    sex = g["sex"].agg(lambda s: s.value_counts().index[0])
    long = (agg.join(nimg).join(sex).reset_index()
            .melt(id_vars=["group_label", "specimen_id", "category", "base_category", "sex", "n_images"],
                  var_name="column", value_name="value").dropna(subset=["value"]))
    ratios = recompute_ratios(df, profile, report)
    long = pd.concat([long, ratios], ignore_index=True)
    long["feature_id"] = [pol.feature_id(c, col) for c, col in zip(long["category"], long["column"])]
    long["tier"] = long["column"].map(lambda c: pol.classify_column(c)[0])
    long["family"] = long["column"].map(lambda c: pol.classify_column(c)[1])
    long = long.rename(columns={"group_label": "species"})
    long = long[["species", "specimen_id", "sex", "category", "base_category", "column",
                 "feature_id", "tier", "family", "value", "n_images"]]
    long.to_csv(dst / "specimen_matrix_long.csv", index=False)

    # ---- feature dictionary
    fd_rows = []
    for (fid, cat, bcat, col), sub in long.groupby(["feature_id", "category", "base_category", "column"]):
        info = pol.structure_info(cat, profile)
        label, definition = pol.feature_label(cat, col, profile, refs)
        tier, fam = pol.classify_column(col)
        sexes = sorted(set(sub["sex"]) - {"unknown"})
        conv = ""
        if col.startswith("cie_"):
            conv = report.get("colour_encoding", {}).get("source_encoding", "")
        elif col.startswith("lmkrel_"):
            conv = f"Procrustes distance / reference span {refs.get(bcat)}"
        elif col.startswith("lmkmm_"):
            conv = "Procrustes distance x landmark centroid size (mm)"
        elif col.startswith("ratio_"):
            conv = "recomputed per specimen from mean structure lengths (taxon profile definitions)"
        fd_rows.append({
            "feature_id": fid, "category": cat, "base_category": bcat, "column": col,
            "tier": tier, "family": fam, "label": label, "definition": definition,
            "unit": pol.feature_unit(col),
            "structure_sex": info.get("sex") if cat != "proportions" else
            ("male" if sexes == ["male"] else "female" if sexes == ["female"] else "both"),
            "section": info.get("section") if cat != "proportions" else "Proportions",
            "key_priority": info.get("key_priority", 2) if cat != "proportions" else 1,
            "n_species": int(sub["species"].nunique()),
            "n_specimens": int(sub["specimen_id"].nunique()),
            "conversion": conv,
        })
    fdict = pd.DataFrame(fd_rows).sort_values(["section", "category", "feature_id"])
    fdict.to_csv(dst / "feature_dictionary.tsv", sep="\t", index=False)

    # ---- species summaries (specimen = unit of observation)
    ss = (long.groupby(["species", "feature_id"])["value"]
          .agg(n="count", min="min", max="max", mean="mean", sd="std", median="median")
          .reset_index())
    ss = ss.merge(fdict[["feature_id", "tier", "unit", "category", "column"]], on="feature_id", how="left")
    ss.to_csv(dst / "species_feature_summary.csv", index=False)

    cov = (long.groupby(["species", "category"])["specimen_id"].nunique()
           .unstack(fill_value=0))
    cov.to_csv(dst / "coverage_species_by_structure.tsv", sep="\t")

    # ---- backward-compatible files for v1 tools
    meta = ["image_base", "category", "group_label", "specimen_id", "sex"] + \
        [c for c in df.columns if c.startswith("has_")]
    df[meta + tier1_cols].to_csv(dst / f"{prefix}_full_features.csv", index=False)
    diag_src = src / f"{prefix}_diagnostic_report.json"
    diag_lookup = set()
    if diag_src.exists():
        with open(diag_src) as f:
            rep = json.load(f)
        keep = set(tier1_cols)
        for cat, cd in rep.get("categories", {}).items():
            for t in cd.get("tests", []):
                if t.get("feature") in keep and t.get("significant", True):
                    diag_lookup.add((cat, t.get("feature")))
            cd["tests"] = [t for t in cd.get("tests", []) if t.get("feature") in keep]
            cd["diagnostic_features"] = [x for x in cd.get("diagnostic_features", []) if x in keep]
            cd["pairwise_significant"] = [p for p in cd.get("pairwise_significant", [])
                                          if p.get("feature") in keep]
            cd["n_diagnostic"] = len(cd["diagnostic_features"])
        rep["_filtered"] = {"filter_version": FILTER_VERSION, "generated": report["generated"],
                            "note": "restricted to Tier-1 source columns; tests were run on the "
                                    "ORIGINAL (unconverted, sex-pooled) values"}
        with open(dst / f"{prefix}_diagnostic_report.json", "w") as f:
            json.dump(rep, f, indent=1)
    v1 = long.copy()
    v1["feature"] = v1["column"]
    v1s = (v1.groupby(["species", "category", "feature"])["value"]
           .agg(n="count", mean="mean", std="std", min="min", max="max").reset_index())
    v1s["cv_percent"] = (100 * v1s["std"] / v1s["mean"].abs()).round(2)
    v1s["is_diagnostic"] = [(c, f) in diag_lookup for c, f in zip(v1s["category"], v1s["feature"])]
    v1s.rename(columns={"species": "group_label"}).to_csv(dst / f"{prefix}_species_summary.csv", index=False)

    if args.cache_dir:
        from biorag_key_feature_filter import filter_cache_diagnosis_files
        stats = filter_cache_diagnosis_files(args.cache_dir, dst / "biorag_cache_filtered")
        report["cache_filter"] = stats

    report["output"] = {
        "specimen_matrix_rows": int(len(long)),
        "n_species": int(long["species"].nunique()),
        "n_specimens": int(long["specimen_id"].nunique()),
        "n_features_tier_key": int((fdict["tier"] == pol.TIER_KEY).sum()),
        "n_features_tier_description": int((fdict["tier"] == pol.TIER_DESC).sum()),
    }
    report["landmark_reference_pairs"] = {k: list(v) for k, v in refs.items()}
    with open(dst / "filter_report_v2.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Tier-1 matrix: {len(long)} values, {report['output']['n_species']} species, "
          f"{report['output']['n_specimens']} specimens")
    print(f"Features: key={report['output']['n_features_tier_key']}  "
          f"description={report['output']['n_features_tier_description']}")
    print(f"Colour encoding: {report.get('colour_encoding')}")
    print(f"Landmark reference spans: {report['landmark_reference_pairs']}")
    bad = {k: v for k, v in report.get("ratio_audit_vs_compiled", {}).items()
           if v["values_differ"] or v["only_old"] or v["only_new"]}
    print(f"Ratio audit (recomputed vs compiled): {len(bad)} ratio columns changed")
    for k, v in bad.items():
        print(f"  {k}: {v}")
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()
