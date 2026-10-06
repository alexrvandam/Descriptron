#!/usr/bin/env python3
"""
descriptron_labelled_plates_v1.py - species plates in the specimens' own colours, with every structure named
=============================================================================================================

The companion to generate_species_plates.py (which is left as it is): the same specimens, but with no mask
overlay. Each annotated structure gets a thin leader line from the edge of the structure to its name, set in a
column beside the image, the way a hand-labelled plate in a taxonomic paper looks.

Where the line ends: it starts from the deepest point of the part of the structure that no smaller structure
covers (for a structure with others inside it, such as the whole wing or whole head, from the deepest point of
its whole outline, since its uncovered remainder may be an unannotated part such as the eyes), heads straight
for the label, and the dot is put where it leaves the structure's outline. A wing cell's
dot is therefore on that cell's own edge, facing its label, and the whole wing's dot is on the wing margin.
With --anchor centre the line runs on to that deepest point instead, so the dot sits in the middle of the
structure (the deepest point, not the centroid, which can fall outside a curved or U-shaped structure). A
structure with smaller ones inside it (the whole wing around its cells, the whole head around the vertex) has
no middle of its own, so its dot goes to the centroid of its whole outline (or its deepest point, if the
centroid falls outside it).
Labels sit on the side of the image the structure is on, at the height of the structure, pushed apart only as
far as needed to stop them overlapping.

For every species (from --group_labels) writes into --plates_dir/<species>/:
  <image>_labelled.png              one labelled image per specimen photograph
  <species>_labelled_plate.png/pdf  those images together (up to --max_images_per_species)

  python descriptron_labelled_plates_v1.py --coco_json <coco.json> --image_dir <images> \\
      --group_labels <group_labels.csv> --plates_dir <out> [--species sp1 sp4] [--outline] [--anchor centre]
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_species_plates as gsp          # noqa: E402  (COCO, mask and group-label readers)

INK = "#1d2327"
HALO = [pe.withStroke(linewidth=1.8, foreground="white")]


def pretty(name):
    return name.replace("_", " ")


def structures_of(anns, id_to_cat, h, w, scale):
    """(name, mask) at display scale for every annotation with a segmentation (keypoint-only ones skipped)"""
    hs, ws = max(1, round(h * scale)), max(1, round(w * scale))
    out = []
    for a in anns:
        seg = a.get("segmentation")
        if not seg:
            continue
        name = id_to_cat.get(a["category_id"], "?")
        if isinstance(seg, list):
            m = np.zeros((hs, ws), np.uint8)
            for poly in seg:
                pts = (np.asarray(poly, float).reshape(-1, 2) * scale).round().astype(np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(m, [pts], 1)
            m = m.astype(bool)
        else:
            m = cv2.resize(gsp.seg_to_mask(seg, h, w).astype(np.uint8), (ws, hs),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        if m.sum() >= 4:
            out.append((name, m))
    return out


def own_regions(structs):
    """each structure minus every SMALLER structure on top of it (falls back to the whole mask if nothing is
    left), so a container such as the whole wing points at its margin, not at a cell inside it"""
    areas = [m.sum() for _, m in structs]
    own = []
    for i, (n, m) in enumerate(structs):
        cover = np.zeros_like(m)
        for j, (_, o) in enumerate(structs):
            if j != i and areas[j] < areas[i]:
                cover |= o
        r = m & ~cover
        own.append(r if r.sum() > 0.03 * m.sum() else m)
    return own


def is_container(structs, i, inside=0.8):
    """True when some smaller structure lies (mostly) inside structure i, e.g. the whole wing around its cells"""
    m = structs[i][1]
    a = m.sum()
    return any(o.sum() < a and (o & m).sum() >= inside * o.sum() for j, (_, o) in enumerate(structs) if j != i)


def centroid(mask):
    """centroid of the whole outline; if it falls outside the structure (a curved one), its deepest point"""
    ys, xs = np.nonzero(mask)
    x, y = int(round(xs.mean())), int(round(ys.mean()))
    if mask[y, x]:
        return x, y
    dt = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(dt), dt.shape)
    return x, y


def exit_point(mask, start, toward):
    """walk from `start` (inside the structure) straight toward the label; the last pixel still inside the
    structure is where the leader line meets its outline, on the side facing the label"""
    H, W = mask.shape
    x0, y0 = start
    dx, dy = toward[0] - x0, toward[1] - y0
    n = int(np.hypot(dx, dy)) + 1
    last = (x0, y0)
    for t in np.linspace(0, 1, n):
        x, y = int(round(x0 + t * dx)), int(round(y0 + t * dy))
        if not (0 <= x < W and 0 <= y < H) or not mask[y, x]:
            break
        last = (x, y)
    return last


def spread(targets, lo, hi, gap):
    """label heights as close to their targets as possible, at least `gap` apart, inside [lo, hi]"""
    if not targets:
        return []
    order = np.argsort(targets)
    n = len(targets)
    gap = min(gap, (hi - lo) / max(1, n - 1)) if n > 1 else gap
    ys = np.clip(np.asarray(targets, float)[order], lo, hi)
    for _ in range(200):                              # alternate push-down / push-up until stable
        moved = False
        for k in range(1, n):
            if ys[k] - ys[k - 1] < gap - 1e-9:
                ys[k] = ys[k - 1] + gap; moved = True
        if ys[-1] > hi:
            ys[-1] = hi
            for k in range(n - 2, -1, -1):
                if ys[k + 1] - ys[k] < gap - 1e-9:
                    ys[k] = ys[k + 1] - gap; moved = True
        if ys[0] < lo:
            ys += lo - ys[0]; moved = True
        if not moved:
            break
    out = np.empty(n)
    out[order] = ys
    return list(out)


def draw_labelled(img_rgb, structs, out_path, title=None, img_width_in=5.0, fontsize=9, outline=False, dpi=200,
                  anchor="edge"):
    H, W = img_rgb.shape[:2]
    own = own_regions(structs)
    poles = []
    for i, r in enumerate(own):
        if is_container(structs, i):     # its uncovered remainder can be an unannotated part (a head's eyes),
            r = structs[i][1]            # so start from the middle of the whole outline instead
        dt = cv2.distanceTransform(np.pad(r, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(dt), dt.shape)
        poles.append((x, y))
    names = [pretty(n) for n, _ in structs]
    left = [i for i in range(len(structs)) if poles[i][0] < W / 2]
    right = [i for i in range(len(structs)) if i not in left]
    per_in = W / img_width_in                                  # data units per inch
    char_in = fontsize * 0.62 / 72
    margin_l = (max((len(names[i]) for i in left), default=0) * char_in + 0.45) * per_in
    margin_r = (max((len(names[i]) for i in right), default=0) * char_in + 0.45) * per_in
    gap = fontsize * 1.45 / 72 * per_in
    fig_w = img_width_in + (margin_l + margin_r) / per_in
    fig_h = H / per_in + (0.35 if title else 0.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img_rgb)
    if outline:
        for (_, m) in structs:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                c = c[:, 0, :]
                ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color="white", lw=0.5, alpha=0.8)
    for side, idx in (("L", left), ("R", right)):
        ys = spread([poles[i][1] for i in idx], 0.02 * H, 0.98 * H, gap)
        tx = -0.25 * per_in if side == "L" else W + 0.25 * per_in
        for i, ty in zip(idx, ys):
            if anchor == "centre":
                ax_, ay = centroid(structs[i][1]) if is_container(structs, i) else poles[i]
            else:
                ax_, ay = exit_point(structs[i][1], poles[i], (tx, ty))
            ax.plot([tx, ax_], [ty, ay], color=INK, lw=0.6, path_effects=HALO, solid_capstyle="round", zorder=3)
            ax.plot(ax_, ay, "o", ms=2.6, mfc=INK, mec="white", mew=0.6, zorder=4)
            ax.text(tx + (-0.06 if side == "L" else 0.06) * per_in, ty, names[i], fontsize=fontsize, color=INK,
                    ha="right" if side == "L" else "left", va="center", zorder=5)
    ax.set_xlim(-margin_l, W + margin_r); ax.set_ylim(H, 0); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 1, color=INK, pad=4)
    fig.subplots_adjust(0, 0, 1, (fig_h - 0.35) / fig_h if title else 1)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def make_plate(species, files, out_base, cols=2, dpi=200):
    ims = [plt.imread(str(f)) for f in files]
    rows = int(np.ceil(len(ims) / cols))
    cell_w = 6.5
    heights = [max(im.shape[0] / im.shape[1] * cell_w for im in ims[r * cols:(r + 1) * cols]) for r in range(rows)]
    fig, axs = plt.subplots(rows, cols, figsize=(cell_w * cols, sum(heights) + 0.6),
                            gridspec_kw={"height_ratios": heights}, squeeze=False)
    for k, ax in enumerate(axs.ravel()):
        ax.axis("off")
        if k < len(ims):
            ax.imshow(ims[k])
            ax.text(0.0, 1.0, "abcdefghijklmnop"[k], transform=ax.transAxes, fontsize=13, weight="bold",
                    va="bottom", ha="left", color=INK)
    fig.suptitle(species, fontsize=15, style="italic" if not species[:2].lower() in ("sp", "cf") else "normal",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=dpi, facecolor="white")
    plt.close(fig)


def round_robin(imgs):
    """one image of each body part in turn (forewing, rostrum, metaleg, ...), so a plate capped at N images
    shows every body part instead of the first N files"""
    groups = defaultdict(list)
    for im in imgs:
        groups[gsp.extract_body_part(im["file_name"])].append(im)
    order = sorted(groups)
    out = []
    while any(groups[g] for g in order):
        for g in order:
            if groups[g]:
                out.append(groups[g].pop(0))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco_json", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--group_labels", required=True)
    p.add_argument("--plates_dir", required=True)
    p.add_argument("--species", nargs="*", default=None, help="only these species (default: all)")
    p.add_argument("--max_images_per_species", type=int, default=8)
    p.add_argument("--display_px", type=int, default=1400, help="longest image side as drawn")
    p.add_argument("--outline", action="store_true", help="also draw each structure's outline, thin and white")
    p.add_argument("--anchor", default="edge", choices=["edge", "centre", "center"],
                   help="where each leader line ends: on the structure's outline, facing its label (default), or "
                        "in the middle of the structure (its deepest point)")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()

    coco, id_to_img, id_to_cat, img_anns = gsp.load_coco(a.coco_json)
    fn_to_species = gsp.load_group_labels(a.group_labels)
    by_species = defaultdict(list)
    for img in coco["images"]:
        sp = fn_to_species.get(img["file_name"]) or fn_to_species.get(gsp._normalize_fn(img["file_name"]))
        if sp and (not a.species or sp in a.species):
            by_species[sp].append(img)
    out = Path(a.plates_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"Labelled plates for {len(by_species)} species -> {out}")
    for sp in sorted(by_species):
        sd = out / sp; sd.mkdir(exist_ok=True)
        done = []
        for img in round_robin(by_species[sp])[:a.max_images_per_species]:
            path = os.path.join(a.image_dir, img["file_name"])
            bgr = cv2.imread(path) if os.path.exists(path) else None
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            s = min(1.0, a.display_px / max(h, w))
            rgb = cv2.cvtColor(cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            structs = structures_of(img_anns.get(img["id"], []), id_to_cat, h, w, s)
            if not structs:
                continue
            stem = Path(img["file_name"]).stem
            f = sd / f"{stem}_labelled.png"
            draw_labelled(rgb, structs, f, title=gsp.extract_body_part(img["file_name"]), outline=a.outline,
                          dpi=a.dpi, anchor="centre" if a.anchor in ("centre", "center") else "edge")
            done.append(f)
        if done:
            make_plate(sp, done, sd / f"{sp}_labelled_plate", dpi=a.dpi)
            print(f"  {sp}: {len(done)} images")


if __name__ == "__main__":
    main()
