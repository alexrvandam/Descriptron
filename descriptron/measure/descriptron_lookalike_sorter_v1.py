#!/usr/bin/env python3
"""
descriptron_lookalike_sorter_v1.py - sort specimens by eye into look-alike groups, and read the groups back
==========================================================================================================

  build  a self-contained HTML drag board (no server, works offline): one thumbnail per species (or per specimen,
         --per specimen) from a COCO file and its images, optionally cropped to one COCO category (--category, any
         name: a habitus, a forewing, a flower) and with the background outside that category's mask whitened
         (--mask_background); an optional second picture per item (--alt_image_dir, e.g. the recolorized images),
         toggled on the board; species on an optional tree (--tree) starred. The expert drags look-alikes together;
         'Save positions' downloads a JSON of every thumbnail's centre. Also writes the items JSON and, with
         --thumbs_dir, the thumbnails as PNG files plus tip_images.csv (species, image) for descriptron_phylo
         --tip_images.
  read   the saved positions -> groups: single-linkage clustering of the thumbnail centres cut at --cut board pixels
         (default: halfway across the largest jump between successive merge distances), the gap report (largest
         link inside a group, smallest distance between groups, so the range of cuts giving the same groups),
         groups numbered in reading order (rows of --row_height board pixels by the group centroid, top to bottom,
         left to right within a row), optional named merges of groups (--merge TYPE=G1,G2; column --merge_column),
         written as a CSV: group, <merge_column>, species, on_tree, x, y, note.

  python descriptron_lookalike_sorter_v1.py build --coco soma_v5.json --image_dir crops/ --image_filter view=dorsal \\
      --tree tree.nwk --alt_image_dir colour_soma_v5/tip_images_soma_abdomen --alt_pattern "{species}.png" \\
      --title "Platydracus look-alike sorter" --save_name platydracus_lookalike_positions.json --out sorter.html
  python descriptron_lookalike_sorter_v1.py read --positions platydracus_lookalike_positions.json --row_height 500 \\
      --merge bee_wasp_mimic=G1,G2 mutillid_mimic=G6,G7 pompilid_mimic=G8,G9,G11 weak_bee_mimic=G3,G4 \\
              non_mimic=G5,G10 --merge_column mimicry_type \\
      --note "G8=taxonomist: scoliid or pompilid wasp mimic, close to G9 and G11" --out lookalike_groups.csv

Reproduces lookalike_sorter/alex_lookalike_groups.csv (Platydracus, 6 Nov 2026) from the taxonomist's saved positions.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ============================================================================ build
def _species_of(im: Dict, attr: str, gmap: Optional[Dict[str, str]]) -> Optional[str]:
    if gmap is not None:
        fn = im["file_name"]
        for k in (fn, Path(fn).stem, Path(fn).name):
            if k in gmap:
                return gmap[k]
        return None
    v = im.get(attr)
    return re.sub(r"[\s_]+", "_", str(v).strip()) if v else None


def _tree_tips(path: Optional[str]) -> set:
    if not path:
        return set()
    txt = Path(path).read_text()
    return {re.sub(r"[\s_]+", "_", m.strip().strip("'\"")) for m in re.findall(r"[(,]\s*([^():,;\[\]]+?)\s*(?=[:,)])", txt)}


def short_name(sp: str) -> str:
    p = sp.replace("_", " ").split()
    return f"{p[0][0]}. {' '.join(p[1:])}" if len(p) > 1 else sp


def _data_url(img, fmt: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 88} if fmt == "JPEG" else {}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _segmentation_mask(seg, h: int, w: int) -> np.ndarray:
    from PIL import Image, ImageDraw
    if isinstance(seg, list):
        im = Image.new("L", (w, h), 0); d = ImageDraw.Draw(im)
        for p in seg:
            if len(p) >= 6:
                d.polygon([(p[i], p[i + 1]) for i in range(0, len(p), 2)], fill=1)
        return np.asarray(im, bool)
    from pycocotools import mask as mu
    rle = mu.frPyObjects(seg, h, w) if isinstance(seg.get("counts"), list) else seg
    return mu.decode(rle).astype(bool)


def build_items(a) -> List[Dict]:
    from PIL import Image
    D = json.load(open(a.coco))
    cat = {c["name"]: c["id"] for c in D["categories"]}
    if a.category and a.category not in cat:
        sys.exit(f"build: no category {a.category!r} in {a.coco} (has: {', '.join(cat)})")
    filt = dict(x.split("=", 1) for x in a.image_filter)
    gmap = None
    if a.groups:
        rows = list(csv.DictReader(open(a.groups)))
        lc = {c.lower(): c for c in rows[0]}
        fc = lc.get("filename") or lc.get("file_name") or list(rows[0])[0]
        gc = lc.get("group_label") or lc.get("species") or list(rows[0])[1]
        gmap = {r[fc]: re.sub(r"[\s_]+", "_", r[gc].strip()) for r in rows}
    anns: Dict[int, List] = {}
    for an in D["annotations"]:
        if a.category and an["category_id"] == cat[a.category]:
            anns.setdefault(an["image_id"], []).append(an)
    on_tree = _tree_tips(a.tree)
    items, seen = [], set()
    for im in D["images"]:
        if any(str(im.get(k)) != v for k, v in filt.items()):
            continue
        sp = _species_of(im, a.species_attr, gmap)
        if not sp:
            continue
        key = sp if a.per == "species" else im["file_name"]
        if key in seen:
            continue
        f = Path(a.image_dir) / im["file_name"]
        if not f.exists():
            continue
        if a.category and im["id"] not in anns:
            continue
        seen.add(key)
        img = Image.open(f).convert("RGB")
        if a.category:
            m = np.zeros((im["height"], im["width"]), bool)
            for an in anns[im["id"]]:
                m |= _segmentation_mask(an["segmentation"], im["height"], im["width"])
            if a.mask_background:
                arr = np.asarray(img).copy(); arr[~m] = 255; img = Image.fromarray(arr)
            ys, xs = np.nonzero(m)
            pad = int(a.pad * max(np.ptp(xs), np.ptp(ys)))
            img = img.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), min(img.width, xs.max() + 1 + pad),
                            min(img.height, ys.max() + 1 + pad)))
        elif a.fg_pattern:                                       # crop to the foreground (specimen) mask
            fp = Path(a.image_dir) / a.fg_pattern.format(stem=Path(im["file_name"]).stem)
            if fp.exists():
                m = np.asarray(Image.open(fp).convert("L")) > 127
                ys, xs = np.nonzero(m)
                if len(xs):
                    img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        th = img.resize((max(1, int(img.width * a.thumb_height / img.height)), a.thumb_height), Image.LANCZOS)
        it = {"sp": key, "name": short_name(sp) if a.per == "species" else f"{short_name(sp)} {Path(im['file_name']).stem[-12:]}",
              "tree": sp in on_tree, "photo": _data_url(th, "JPEG"), "src": im["file_name"]}
        if a.alt_image_dir:
            cands = sorted(Path(a.alt_image_dir).glob(a.alt_pattern.format(stem=Path(im["file_name"]).stem, species=sp)))
            if cands:
                al = Image.open(cands[0]).convert("RGBA")
                al = al.resize((max(1, int(al.width * a.alt_height / al.height)), a.alt_height), Image.LANCZOS)
                it["rec"] = _data_url(al, "PNG")
        it.setdefault("rec", it["photo"])
        if a.thumbs_dir:
            Path(a.thumbs_dir).mkdir(parents=True, exist_ok=True)
            p = Path(a.thumbs_dir) / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', key)}.png"
            th.save(p); it["_thumb"] = str(p)
        items.append(it)
    return items


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root{--bg:#f4f2ee;--ink:#1d1d1f;--muted:#6b6b70;--card:#ffffff;--line:#d8d4cc;--accent:#2f6f8f}
body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif}
#bar{position:sticky;top:0;z-index:10;background:var(--card);border-bottom:1px solid var(--line);padding:8px 12px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
#bar button{font:inherit;padding:5px 12px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
#bar button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
#help{color:var(--muted);font-size:12.5px}
#board{position:relative;width:__BW__px;height:__BH__px;background-image:radial-gradient(#d9d5cc 1px,transparent 1px);background-size:40px 40px}
.t{position:absolute;cursor:grab;user-select:none;text-align:center;padding:3px;border-radius:6px;border:1px solid transparent;background:rgba(255,255,255,.0)}
.t:hover{border-color:var(--line);background:#fff}
.t.drag{cursor:grabbing;z-index:5;box-shadow:0 4px 14px rgba(0,0,0,.18);background:#fff}
.t img{display:block;margin:0 auto;pointer-events:none}
.t .n{font-size:11px;font-style:italic;white-space:nowrap;color:var(--ink)}
.t .s{color:#b07d00;font-style:normal}
</style></head><body>
<div id="bar">
 <strong>__TITLE__</strong>
 <button id="view">Show: photo</button>
 <label>size <input id="size" type="range" min="50" max="220" value="110"></label>
 <button id="save" class="primary">Save positions</button>
 <button id="reset">Reset layout</button>
 <span id="help">Drag look-alikes together and pull the groups well apart; empty space between groups is what defines them.
 __STAR__Your layout is kept in this browser; Save downloads __SAVE__.</span>
</div>
<div id="board"></div>
<script>
const ITEMS = __ITEMS__;
const KEY = __KEY__;
const board = document.getElementById('board'); let mode = 'photo', H = 110; const pos = {};
function store(){ try{ localStorage.setItem(KEY, JSON.stringify({pos, mode, H})); }catch(e){} }
function load(){ try{ const s = JSON.parse(localStorage.getItem(KEY)||'null'); if(s){ Object.assign(pos, s.pos||{}); mode = s.mode||mode; H = s.H||H; } }catch(e){} }
function defaults(){ const cols = 13; ITEMS.forEach((it,i)=>{ if(!pos[it.sp]) pos[it.sp] = {x: 90 + (i%cols)*300, y: 120 + Math.floor(i/cols)*330}; }); }
function render(){
  board.innerHTML=''; document.getElementById('view').textContent = 'Show: ' + (mode==='photo'?'photo':'second view');
  document.getElementById('size').value = H;
  ITEMS.forEach(it=>{
    const d = document.createElement('div'); d.className='t'; d.dataset.sp = it.sp;
    const im = new Image(); im.src = mode==='photo'?it.photo:it.rec; im.style.height = H+'px'; d.appendChild(im);
    const n = document.createElement('div'); n.className='n'; n.innerHTML = it.name + (it.tree?' <span class="s">★</span>':''); d.appendChild(n);
    board.appendChild(d);
    im.onload = ()=>place(d); place(d); drag(d);
  });
}
function place(d){ const p = pos[d.dataset.sp]; d.style.left = (p.x - d.offsetWidth/2)+'px'; d.style.top = (p.y - d.offsetHeight/2)+'px'; }
function drag(d){
  d.addEventListener('pointerdown', e=>{
    e.preventDefault(); d.setPointerCapture(e.pointerId); d.classList.add('drag');
    const p = pos[d.dataset.sp], sx = e.pageX, sy = e.pageY, ox = p.x, oy = p.y;
    const mv = ev=>{ p.x = Math.max(20, ox + ev.pageX - sx); p.y = Math.max(20, oy + ev.pageY - sy); place(d); };
    const up = ()=>{ d.classList.remove('drag'); d.removeEventListener('pointermove', mv); d.removeEventListener('pointerup', up); store(); };
    d.addEventListener('pointermove', mv); d.addEventListener('pointerup', up);
  });
}
document.getElementById('view').onclick = ()=>{ mode = mode==='photo'?'rec':'photo'; store(); render(); };
document.getElementById('size').oninput = e=>{ H = +e.target.value; store(); render(); };
document.getElementById('reset').onclick = ()=>{ if(confirm('Put every thumbnail back on the starting grid?')){ for(const k in pos) delete pos[k]; defaults(); store(); render(); } };
document.getElementById('save').onclick = ()=>{
  const out = {saved: new Date().toISOString(), note: 'thumbnail centre positions in board pixels', thumbnail_height: H, view: mode,
               species: ITEMS.map(it=>({species: it.sp, on_tree: it.tree, x: Math.round(pos[it.sp].x), y: Math.round(pos[it.sp].y), photo: it.src}))};
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type:'application/json'}));
  a.download = __SAVEJS__; a.click();
};
load(); defaults(); render();
</script></body></html>"""


def render_html(items: List[Dict], title: str, save_name: str, board_w: int = 4200, board_h: int = 3000,
                any_tree: bool = True) -> str:
    clean = [{k: v for k, v in it.items() if not k.startswith("_")} for it in items]
    key = "descriptron_sorter_" + re.sub(r"[^A-Za-z0-9]+", "_", title)[:40]
    return (HTML.replace("__TITLE__", title).replace("__SAVEJS__", json.dumps(save_name)).replace("__SAVE__", save_name)
            .replace("__KEY__", json.dumps(key)).replace("__BW__", str(board_w)).replace("__BH__", str(board_h))
            .replace("__STAR__", "★ = on the tree. " if any_tree else "")
            .replace("__ITEMS__", json.dumps(clean)))


def cmd_build(a) -> int:
    items = build_items(a)
    if not items:
        sys.exit("build: no images matched (check --image_dir, --image_filter, --category, --species_attr/--groups)")
    Path(a.out).write_text(render_html(items, a.title, a.save_name, a.board_width, a.board_height, bool(a.tree)))
    if a.items_out:
        json.dump([{k: v for k, v in it.items() if not k.startswith("_")} for it in items], open(a.items_out, "w"))
    if a.thumbs_dir:
        with open(Path(a.thumbs_dir) / "tip_images.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["species", "image"])
            for it in items:
                w.writerow([it["sp"], it["_thumb"]])
    print(f"wrote {a.out}: {len(items)} items ({sum(it['tree'] for it in items)} on the tree)")
    return 0


# ============================================================================ read
def single_linkage(X: np.ndarray, cut: Optional[float]) -> Dict:
    """single-linkage clusters of the points X (n x 2) cut at `cut` (None: halfway across the largest jump between
    successive merge distances); with the largest merge distance inside a cluster and the smallest between clusters."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import cdist
    Z = linkage(X, "single")
    h = np.sort(Z[:, 2])
    if cut is None:
        j = int(np.argmax(np.diff(h))) if len(h) > 1 else 0
        cut = float((h[j] + h[j + 1]) / 2) if len(h) > 1 else float(h[0] + 1)
    lab = fcluster(Z, cut, "distance")
    within = max([d for d in h if d <= cut], default=0.0)
    between = min([float(cdist(X[lab == p], X[lab == q]).min()) for p in set(lab) for q in set(lab) if p < q],
                  default=np.inf)
    return {"labels": lab, "cut": cut, "within_max": float(within), "between_min": float(between)}


def number_groups(X: np.ndarray, lab: np.ndarray, row_height: float, prefix: str = "G") -> Dict[int, str]:
    """group names in reading order: rows of row_height board pixels by the group's centroid (top to bottom), then
    left to right."""
    cen = {k: X[lab == k].mean(0) for k in set(lab)}
    order = sorted(cen, key=lambda k: (int(cen[k][1] // row_height), cen[k][0]))
    return {k: f"{prefix}{i + 1}" for i, k in enumerate(order)}


def cmd_read(a) -> int:
    J = json.load(open(a.positions))
    P = J["species"] if isinstance(J, dict) else J
    X = np.array([[float(p["x"]), float(p["y"])] for p in P])
    sp = [p["species"] for p in P]
    R = single_linkage(X, a.cut)
    names = number_groups(X, R["labels"], a.row_height, a.prefix)
    merge = {}
    for m in a.merge:
        t, _, gs = m.partition("=")
        for g in gs.split(","):
            merge[g.strip()] = t.strip()
    notes = dict(x.split("=", 1) for x in a.note)
    rows = []
    for i, s in enumerate(sp):
        g = names[R["labels"][i]]
        rows.append({"group": g, **({a.merge_column: merge.get(g, "")} if a.merge else {}), "species": s,
                     "on_tree": str(bool(P[i].get("on_tree", False))), "x": int(round(X[i, 0])), "y": int(round(X[i, 1])),
                     "note": notes.get(s, notes.get(g, ""))})
    rows.sort(key=lambda r: (r["group"], r["species"]))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sizes = {g: sum(1 for r in rows if r["group"] == g) for g in sorted(set(names.values()), key=lambda g: int(g[len(a.prefix):]))}
    print(f"{len(sizes)} groups by single linkage cut at {R['cut']:.1f} px: " + ", ".join(f"{g} {n}" for g, n in sizes.items()))
    print(f"gap report: largest link inside a group {R['within_max']:.1f} px, smallest distance between groups "
          f"{R['between_min']:.1f} px: any cut between them gives the same groups")
    unm = sorted(g for g in sizes if a.merge and g not in merge)
    if unm:
        print("groups without a --merge name:", ", ".join(unm))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the HTML drag board")
    b.add_argument("--coco", required=True); b.add_argument("--image_dir", required=True)
    b.add_argument("--category", default=None, help="crop each thumbnail to this COCO category (any name)")
    b.add_argument("--mask_background", action="store_true", help="whiten the image outside --category's mask")
    b.add_argument("--pad", type=float, default=0.03, help="crop margin, share of the category's longer side")
    b.add_argument("--fg_pattern", default=None, help="without --category: crop to the bounding box of a foreground "
                   "mask beside each image, {stem} = image stem (e.g. '{stem}_mask.png')")
    b.add_argument("--image_filter", nargs="+", default=[], metavar="ATTR=VALUE", help="e.g. view=dorsal")
    b.add_argument("--species_attr", default="species", help="COCO image attribute holding the species")
    b.add_argument("--groups", default=None, help="instead: group labels CSV (filename, group_label)")
    b.add_argument("--per", choices=["species", "specimen"], default="species",
                   help="one item per species (its first image) or per image")
    b.add_argument("--tree", default=None, help="Newick tree: its species are starred")
    b.add_argument("--alt_image_dir", default=None, help="second picture per item (e.g. recolorized images)")
    b.add_argument("--alt_pattern", default="*{stem}*.png", help="glob in --alt_image_dir: {stem} = image stem, {species} = species")
    b.add_argument("--thumb_height", type=int, default=260); b.add_argument("--alt_height", type=int, default=200)
    b.add_argument("--board_width", type=int, default=4200); b.add_argument("--board_height", type=int, default=3000)
    b.add_argument("--title", default="Look-alike sorter")
    b.add_argument("--save_name", default="lookalike_positions.json", help="file name the board saves to")
    b.add_argument("--items_out", default=None, help="also write the items JSON")
    b.add_argument("--thumbs_dir", default=None, help="also write the thumbnails and tip_images.csv here")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)
    r = sub.add_parser("read", help="saved positions -> groups (single linkage) with the gap report")
    r.add_argument("--positions", required=True)
    r.add_argument("--cut", type=float, default=None, help="board pixels (default: in the largest gap)")
    r.add_argument("--row_height", type=float, default=500.0, help="reading-order rows for numbering the groups")
    r.add_argument("--prefix", default="G")
    r.add_argument("--merge", nargs="+", default=[], metavar="NAME=G1,G2", help="named unions of groups")
    r.add_argument("--merge_column", default="type")
    r.add_argument("--note", nargs="+", default=[], metavar="GROUP_OR_SPECIES=TEXT",
                   help="a note for the species of a group, or for one species (the species' note wins)")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_read)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
