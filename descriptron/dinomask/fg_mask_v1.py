"""Specimen foreground from a plain photographic background (added 12 Oct 2026, for DINOMask --fg_mask border).
Background colour = median of a border frame in L*a*b*; foreground = colour distance above Otsu's threshold
(floor --min_dist), cleaned, holes filled; keeps the largest component and components near it (legs, antennae),
drops long thin pieces (scale bars) and specks."""
import numpy as np, cv2
from scipy.ndimage import binary_fill_holes


def fg_mask(rgb, frame=0.04, min_dist=10.0, near=0.08):
    h, w = rgb.shape[:2]; lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = max(2, int(frame * min(h, w))); border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    # background may shade across the frame: model it as a smooth surface from the border median per side
    med = np.median(border, 0); d = np.linalg.norm(lab - med, axis=-1)
    d8 = np.clip(d * 2, 0, 255).astype(np.uint8); t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (d8 > max(t, 2 * min_dist)).astype(np.uint8)
    k = max(3, int(0.004 * max(h, w)) | 1); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    filled = binary_fill_holes(m).astype(np.uint8); holes = filled & (1 - m)   # fill only SMALL holes: background seen between
    nh_, hl, hst, _ = cv2.connectedComponentsWithStats(holes)                     # legs and body is real background
    small = np.zeros(nh_, bool); small[1:] = hst[1:, cv2.CC_STAT_AREA] < 0.0015 * h * w; m = (m | small[hl]).astype(np.uint8)
    # scale bars touching a leg: long horizontal runs that are thin
    L = max(9, int(0.08 * w)); T = max(3, int(0.012 * h))
    horiz = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)); thick = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((T, 1), np.uint8))
    bar = cv2.dilate(horiz & (1 - thick), np.ones((3, 3), np.uint8)); m = (m & (1 - bar)).astype(np.uint8)
    n, lb, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1: return m.astype(bool)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); x, y, ww, hh = st[big, :4]; px, py = int(near * ww), int(near * hh)
    keep = np.zeros(n, bool); keep[big] = True
    for i in range(1, n):
        if i == big: continue
        xi, yi, wi, hi, ai = st[i]
        thin = max(wi, hi) > 8 * max(1, min(wi, hi))            # scale bar
        inside = xi + wi >= x - px and xi <= x + ww + px and yi + hi >= y - py and yi <= y + hh + py
        if ai >= 0.002 * st[big, cv2.CC_STAT_AREA] and inside and not thin: keep[i] = True
    return keep[lb]
