"""FHS_and_CLAHE_V22 --recolorize against R recolorize 0.2.0 (Weller et al. 2024), on recolorize's own five example
beetles (extdata, CC BY 4.0). The R reference was made with recolorize2() at its defaults, at cutoff 45 and at bins 3
(validation/recolorize_v22/scripts/run_R_beetles.R); the fixtures hold R's palettes and, for the defaults, its pixel map.

What has to match: the same number of colours per image, every colour within 0.05 CIEDE2000 of R's (measured: 0.046
at most) and, for the defaults, >= 99.9 % of pixels on the same colour (measured >= 99.99 %; the rest are pixels R
assigns at random among near-equal distances, max.col ties.method = "random").

V22 is found, in this order: $DESCRIPTRON_V22, the copy beside tests/ (repository), the copy installed with
descriptron-core."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pd = pytest.importorskip("pandas")
skc = pytest.importorskip("skimage.color")
linear_sum_assignment = pytest.importorskip("scipy.optimize").linear_sum_assignment

DATA = Path(__file__).parent / "data" / "recolorize_beetles"
BEETLES = ["chongi", "corbetti", "ephippigera", "fulgidissima", "ocellata"]
SETTINGS = {"default": dict(cutoff=20.0, bins=2), "cut45": dict(cutoff=45.0, bins=2), "bins3": dict(cutoff=20.0, bins=3)}


def _find_v22():
    cands = [os.environ.get("DESCRIPTRON_V22"), Path(__file__).parent.parent / "FHS_and_CLAHE_V22.py"]
    try:
        import descriptron.tools as _t   # pip layout (descriptron-core): flat tools/ folder
        cands.append(Path(_t.__file__).parent / "FHS_and_CLAHE_V22.py")
    except Exception:
        pass
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    pytest.skip("FHS_and_CLAHE_V22.py not found")


@pytest.fixture(scope="module")
def v22():
    p = _find_v22()
    spec = importlib.util.spec_from_file_location("fhs_v22", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _lab(rgb01):
    return skc.rgb2lab(np.asarray(rgb01, float).reshape(-1, 1, 3)).reshape(-1, 3)


@pytest.mark.parametrize("setting", list(SETTINGS))
def test_palettes_match_R(v22, tmp_path, setting):
    R = pd.read_csv(DATA / f"R_{setting}_palettes.csv")
    for b in BEETLES:
        out = tmp_path / f"{setting}_{b}.png"
        v22.recolorize_clustering(str(DATA / f"{b}_fg_beetle.png"), str(out), mask_path=str(DATA / f"{b}_bin_beetle.png"),
                                  linkage_method="complete", bin_space="sRGB", white="R", **SETTINGS[setting])
        V = pd.read_csv(str(out)[:-4] + "_palette.csv"); V = V[V.n_pixels > 0]
        g = R[(R.image == f"{b}_fg_beetle") & (R["size"] > 0)]
        assert len(V) == len(g), f"{setting} {b}: {len(V)} colours, R {len(g)}"
        D = skc.deltaE_ciede2000(_lab(V[["R01", "G01", "B01"]].values)[:, None], _lab(g[["R", "G", "B"]].values)[None])
        vi, ri = linear_sum_assignment(D)
        assert D[vi, ri].max() < 0.05, f"{setting} {b}: max dE00 {D[vi, ri].max():.4f}"
        if setting == "default":
            aR = cv2.imread(str(DATA / f"R_default_{b}_fg_beetle_assign.png"), cv2.IMREAD_UNCHANGED)
            aR = aR[..., 0] if aR.ndim == 3 else aR
            aV = cv2.imread(str(out)[:-4] + "_labels.png", cv2.IMREAD_UNCHANGED)
            fg = aV > 0
            mp = {int(V.colour.values[i]): int(g.cluster.values[j]) for i, j in zip(vi, ri)}
            agree = np.mean(np.vectorize(lambda x: mp.get(int(x), -1))(aV[fg]) == aR[fg])
            assert agree >= 0.999, f"{b}: pixel agreement {agree:.5f}"
