"""
descriptron_mcp.data_tools — read the inputs and outputs of the programs
========================================================================

Small, dependency-light readers (csv/json from the standard library, Pillow
for images) so the calling model can look at a COCO file, a results table, a
figure or a specimen photograph without a program run.

    DESCRIPTRON_MCP_ALLOWED_ROOTS   if set, file tools refuse paths outside these
                                    directories (OS path separator between them)
"""
from __future__ import annotations

import csv
import fnmatch
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(min(2**31 - 1, 10**9))


def check_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    roots = [Path(r).expanduser().resolve()
             for r in os.environ.get("DESCRIPTRON_MCP_ALLOWED_ROOTS", "").split(os.pathsep) if r.strip()]
    if roots and not any(p == r or r in p.parents for r in roots):
        raise PermissionError(f"{p} is outside DESCRIPTRON_MCP_ALLOWED_ROOTS")
    return p


def list_files(directory: str, pattern: str = "*", recursive: bool = False, max_items: int = 200) -> dict:
    d = check_path(directory)
    if not d.is_dir():
        raise ValueError(f"not a directory: {d}")
    it = d.rglob("*") if recursive else d.iterdir()
    rows, total = [], 0
    for p in it:
        if not fnmatch.fnmatch(p.name, pattern):
            continue
        total += 1
        if len(rows) < max_items:
            try:
                size = p.stat().st_size if p.is_file() else None
            except OSError:
                size = None
            rows.append({"path": str(p.relative_to(d)), "dir": p.is_dir(), "bytes": size})
    rows.sort(key=lambda r: r["path"])
    return {"directory": str(d), "matches": total, "shown": len(rows), "items": rows}


def read_text(path: str, max_chars: int = 20000, start_char: int = 0) -> dict:
    p = check_path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    chunk = text[start_char:start_char + max_chars]
    return {"path": str(p), "total_chars": len(text), "start_char": start_char,
            "truncated": start_char + max_chars < len(text), "text": chunk}


def _delimiter(p: Path, head: str) -> str:
    if p.suffix.lower() in (".tsv", ".tab"):
        return "\t"
    return "\t" if head.count("\t") > head.count(",") else ","


def read_table(path: str, max_rows: int = 50, columns: list[str] | None = None,
               where: dict[str, str] | None = None) -> dict:
    """CSV/TSV preview. `where` keeps rows whose column equals the given value exactly."""
    p = check_path(path)
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        head = fh.read(4096)
        fh.seek(0)
        reader = csv.DictReader(fh, delimiter=_delimiter(p, head))
        header = reader.fieldnames or []
        missing = [c for c in (columns or []) + list((where or {}).keys()) if c not in header]
        if missing:
            raise ValueError(f"no such column(s) {missing}; columns are: {header}")
        rows, matched, total = [], 0, 0
        for row in reader:
            total += 1
            if where and any(row.get(k) != str(v) for k, v in where.items()):
                continue
            matched += 1
            if len(rows) < max_rows:
                rows.append({c: row.get(c) for c in columns} if columns else row)
    return {"path": str(p), "columns": header, "total_rows": total, "matching_rows": matched,
            "shown": len(rows), "rows": rows}


def coco_summary(path: str, max_categories: int = 200) -> dict:
    p = check_path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    cats = {c["id"]: c.get("name", str(c["id"])) for c in d.get("categories", [])}
    anns = d.get("annotations", [])
    per_cat = Counter(cats.get(a.get("category_id"), f"<unknown id {a.get('category_id')}>") for a in anns)
    per_img = Counter(a.get("image_id") for a in anns)
    kinds = Counter()
    for a in anns:
        seg = a.get("segmentation")
        if isinstance(seg, dict):
            kinds["rle"] += 1
        elif seg:
            kinds["polygon"] += 1
        if a.get("keypoints"):
            kinds["keypoints"] += 1
    imgs = d.get("images", [])
    zero_dims = [i.get("file_name") for i in imgs if not i.get("width") or not i.get("height")]
    unused = [i.get("file_name") for i in imgs if i.get("id") not in per_img]
    return {
        "path": str(p), "images": len(imgs), "annotations": len(anns), "categories": len(cats),
        "annotations_per_category": dict(per_cat.most_common(max_categories)),
        "annotation_kinds": dict(kinds),
        "images_without_annotations": {"count": len(unused), "examples": unused[:10]},
        # a zero width/height silently rescales every coordinate downstream
        "images_with_zero_dimensions": {"count": len(zero_dims), "examples": zero_dims[:10]},
        "example_images": [i.get("file_name") for i in imgs[:5]],
    }


def view_image(path: str, max_side: int = 1568):
    """Load any Pillow-readable image (incl. 16-bit TIFF), downscale, return JPEG bytes + info."""
    from PIL import Image as PILImage
    p = check_path(path)
    with PILImage.open(p) as im:
        info = {"path": str(p), "original_size": list(im.size), "mode": im.mode}
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            im = im.point(lambda v: v * (1 / 256)).convert("L")   # 16-bit micro-CT slices
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        info["shown_size"] = list(im.size)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), info


# ---------------------------------------------------------------- evidence --
def species_evidence(matrix_dir: str, species: str, tier: str | None = None,
                     structure: str | None = None, max_rows: int = 500) -> dict:
    """One species' per-feature n/min/max/mean from species_feature_summary.csv, with the
    range across all species beside each, which is what a comparison must be read against."""
    f = check_path(matrix_dir) / "species_feature_summary.csv"
    if not f.exists():
        raise ValueError(f"{f} not found — matrix_dir must be the output of biorag_key_feature_filter_v2")
    across: dict[str, list] = defaultdict(list)
    mine = []
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                across[r["feature_id"]].append((float(r["min"]), float(r["max"]), r["species"]))
            except (TypeError, ValueError):
                pass
            if r["species"] != species:
                continue
            if tier and r.get("tier") != tier:
                continue
            if structure and r.get("category") != structure:
                continue
            mine.append(r)
    if not mine:
        known = sorted({s for v in across.values() for *_, s in v})
        raise ValueError(f"no rows for species '{species}' (tier={tier}, structure={structure}); "
                         f"species codes in this matrix: {known}")
    rows = []
    for r in mine[:max_rows]:
        a = across.get(r["feature_id"], [])
        rows.append({
            "feature_id": r["feature_id"], "tier": r.get("tier"), "unit": r.get("unit"),
            "n": r.get("n"), "min": r.get("min"), "max": r.get("max"), "mean": r.get("mean"),
            "all_species_min": min((x[0] for x in a), default=None),
            "all_species_max": max((x[1] for x in a), default=None),
            "n_species": len(a),
        })
    return {"species": species, "features": len(mine), "shown": len(rows), "rows": rows,
            "note": ("Tiers: 'key' = usable in the key and every section; 'description' = description "
                     "sections; anything statistical belongs in Remarks only.")}
