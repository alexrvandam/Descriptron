"""descriptron_acuity_blur_v1: the AcuityView MTF (1 at zero frequency, falling with frequency, 0.028 at the acuity),
identity at very high acuity, normalized convolution keeping a uniform specimen uniform and the white plate out of
it, padding / cropping that returns the photograph's own size, and the command line writing one folder per setting
with the original file names. Fast, no R (the comparison with R AcuityView is validation/validate_acuity_blur_vs_r.py)."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import descriptron_acuity_blur_v1 as ab        # noqa: E402


def _specimen(H=150, W=90, colour=(150, 40, 30), plate=(250, 250, 250)):
    img = np.empty((H, W, 3), np.uint8)
    img[:] = plate
    mask = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    mask[((yy - H / 2) / (0.4 * H)) ** 2 + ((xx - W / 2) / (0.3 * W)) ** 2 <= 1] = True
    img[mask] = colour
    return img, mask


def test_mtf_zero_monotonic_and_at_acuity():
    assert ab.mtf(0.0, 0.2) == 1.0
    f = np.linspace(0, 50, 501)
    v = ab.mtf(f, 1 / 6.29)
    assert np.all(np.diff(v) < 0) and v[0] == 1.0
    assert abs(ab.mtf(6.29, 1 / 6.29) - np.exp(-3.56)) < 1e-12
    H = ab.acuityview_mtf_grid(64, 10.0, 0.1)
    assert H[32, 32] == 1.0 and H.max() == 1.0 and H.min() > 0
    # radial profile falls away from the centre, and the grid is even (real filter)
    assert np.all(np.diff(H[32, 32:]) <= 0)
    U = np.fft.ifftshift(H)
    assert np.allclose(U, np.roll(U[::-1, ::-1], (1, 1), (0, 1)))


def test_geometry_constants():
    assert abs(ab.width_in_degrees(2.0, 2.0) - 57.2958 * 2 * np.arctan(0.5)) < 1e-12
    n, ppm, d = 2048, 50.0, 500.0
    canvas = ab.pixels_per_degree(n, n / ppm, d, "canvas")
    centre = ab.pixels_per_degree(n, None, d, "centre", px_per_unit=ppm)
    assert abs(centre - ppm * d / 57.2958) < 1e-12
    assert abs(canvas / centre - 1) < 0.005           # 41 mm canvas at 50 cm: < 0.5 %
    assert ab.next_pow2(1162) == 2048 and ab.next_pow2(1024) == 1024 and ab.next_pow2(3) == 4


def test_identity_at_huge_acuity():
    img, mask = _specimen()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, img.shape).astype(np.uint8)
    s = ab.build_settings([1e6], None, [20])
    for bg in ("white", "normalized"):
        out, _ = ab.blur_image(img, s, 40.0, mask=mask, background=bg)
        o = out[s[0]["name"]].astype(int)
        region = np.ones_like(mask) if bg == "white" else mask
        assert np.abs(o[region] - img.astype(int)[region]).max() <= 1


def test_normalized_keeps_uniform_specimen_uniform_and_plate_out():
    img, mask = _specimen()
    s = ab.build_settings([6.29], None, [50])          # strong blur: FWHM ~ 30 px at 40 px/mm
    out, info = ab.blur_image(img, s, 40.0, mask=mask, background="normalized")
    o = out[s[0]["name"]].astype(int)
    assert info["settings"][s[0]["name"]]["fwhm_px"] > 20
    assert np.abs(o[mask] - np.array([150, 40, 30])).max() <= 1      # still the specimen's colour, edges included
    assert (o[~ab.dilate(mask, 3)] == 255).all()                     # white outside
    # the plain blur washes the plate into the edge (what normalized avoids)
    w, _ = ab.blur_image(img, s, 40.0, mask=mask, background="white")
    interior = mask & ~ab.dilate(~mask, 5)
    assert np.abs(w[s[0]["name"]].astype(int)[mask & ~interior] - [150, 40, 30]).max() > 20   # 5-px edge band


def test_background_value_and_compat_run():
    img, mask = _specimen()
    s = ab.build_settings([22.5], None, [20])
    out, _ = ab.blur_image(img, s, 40.0, mask=mask, background="value", bg_value=(90, 60, 40))
    o = out[s[0]["name"]]
    assert tuple(o[0, 0]) == (90, 60, 40)
    c, info = ab.blur_image(img, s, 40.0, compat=True)
    assert c[s[0]["name"]].shape == img.shape


def test_padding_round_trip_sizes():
    for H, W in [(150, 90), (1162, 504), (64, 64), (1, 7)]:
        n = ab.canvas_size(H, W, 0.1)
        assert n & (n - 1) == 0 and n >= max(H, W) * 1.2
        a = np.arange(H * W * 3, dtype=float).reshape(H, W, 3)
        c, off = ab.place_on_canvas(a, n, -1.0)
        assert c.shape == (n, n, 3) and off == ((n - H) // 2, (n - W) // 2)
        assert np.array_equal(ab.crop_from_canvas(c, off, H, W), a)
        assert (c == -1).sum() == 3 * (n * n - H * W)


def test_compat_stretches_channels_science_does_not():
    rng = np.random.default_rng(3)
    img = (0.3 + 0.4 * rng.random((32, 32, 3)))           # channels span 0.3-0.7
    c = ab.acuityview_compat(img, 10.0, 1.0, 1e-6)
    assert abs(c.min() - 0) < 1e-9 and abs(c.max() - 1) < 1e-9   # AcuityView stretched them to 0-1
    s = ab.build_settings(None, [1e-6], [1])
    o, _ = ab.blur_image(np.floor(img * 255 + .5).astype(np.uint8), s, 32.0, background="white")
    v = o[s[0]["name"]] / 255
    assert v.min() > 0.29 and v.max() < 0.71


def test_coco_mask_and_cli(tmp_path):
    img, mask = _specimen()
    Image.fromarray(img).save(tmp_path / "a_dorsal.png")
    Image.fromarray(img).save(tmp_path / "b_ventral.png")
    ys, xs = np.nonzero(mask)
    poly = []
    for t in np.linspace(0, 2 * np.pi, 80, endpoint=False):
        poly += [45 + 0.3 * 90 * np.cos(t), 75 + 0.4 * 150 * np.sin(t)]
    coco = {"categories": [{"id": 4, "name": "soma_abdomen"}],
            "images": [{"id": 1, "file_name": "a_dorsal.png", "width": 90, "height": 150, "view": "dorsal",
                        "species": "Platydracus testi", "scale_px_per_mm": 40.0},
                       {"id": 2, "file_name": "b_ventral.png", "width": 90, "height": 150, "view": "ventral",
                        "species": "Platydracus testi", "scale_px_per_mm": 40.0}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 4, "segmentation": [poly]},
                            {"id": 2, "image_id": 2, "category_id": 4, "segmentation": [poly]}]}
    m = ab.specimen_mask(coco, coco["images"][0])
    assert (m ^ mask).sum() < 0.05 * mask.sum()
    (tmp_path / "c.json").write_text(json.dumps(coco))
    rc = ab.main(["--coco", str(tmp_path / "c.json"), "--image_dir", str(tmp_path), "--out_dir", str(tmp_path / "o"),
                  "--acuity_cpd", "6.29", "33", "--distance_cm", "5", "50", "--workers", "1",
                  "--qc_species", "testi", "--acuity_labels", "6.29=starling"])
    assert rc == 0
    for name in ("cpd6.29_d5cm", "cpd6.29_d50cm", "cpd33_d5cm", "cpd33_d50cm"):
        f = tmp_path / "o" / name / "a_dorsal.png"
        assert f.exists() and Image.open(f).size == (90, 150)
        assert not (tmp_path / "o" / name / "b_ventral.png").exists()     # dorsal only
    assert (tmp_path / "o" / "qc_sheet.jpg").exists() and (tmp_path / "o" / "blur_parameters.csv").exists()
