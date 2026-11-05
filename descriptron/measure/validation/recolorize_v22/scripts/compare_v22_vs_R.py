"""Compare FHS_and_CLAHE_V22 --recolorize output with R recolorize output on the same images.
Usage: compare_v22_vs_R.py <R_dir with palettes.csv + *_assign.png> <V22 dir> <out.csv>"""
import sys, os, glob, numpy as np, pandas as pd, cv2
from skimage import color
from scipy.optimize import linear_sum_assignment
Rd, Vd, out = sys.argv[1:4]
P = pd.read_csv(os.path.join(Rd, 'palettes.csv')); rows = []
for img, g in P.groupby('image'):
    v = glob.glob(os.path.join(Vd, f'*_{img}_palette.csv'))
    if not v: continue
    V = pd.read_csv(v[0]); g = g[g['size'] > 0]
    labR = color.rgb2lab(g[['R', 'G', 'B']].values.reshape(-1, 1, 3)).reshape(-1, 3)
    labV = color.rgb2lab((V[['R01', 'G01', 'B01']].values if 'R01' in V else V[['R', 'G', 'B']].values / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    D = np.array([[color.deltaE_ciede2000(a[None], b[None])[0] for b in labV] for a in labR])
    ri, vi = linear_sum_assignment(D)
    aR = cv2.imread(os.path.join(Rd, f'{img}_assign.png'), cv2.IMREAD_UNCHANGED)
    aV = cv2.imread(v[0].replace('_palette.csv', '_labels.png'), cv2.IMREAD_UNCHANGED)
    if aR.ndim == 3: aR = aR[:, :, 0]
    fg = aV > 0
    mapping = {int(g['cluster'].values[i]): int(V['colour'].values[j]) for i, j in zip(ri, vi)}
    agree = np.mean([mapping.get(int(r), -1) == int(q) for r, q in zip(aR[fg], aV[fg])]) if aR.shape == aV.shape else np.nan
    rows.append(dict(image=img, n_R=len(g), n_V22=len(V), dE00_median=np.median(D[ri, vi]), dE00_max=D[ri, vi].max(),
                     size_R=list(g['size'].values[ri]), size_V22=list(V['n_pixels'].values[vi]), pixel_agreement=agree,
                     fg_R=int((aR > 0).sum()), fg_V22=int(fg.sum())))
df = pd.DataFrame(rows); df.to_csv(out, index=False); pd.set_option('display.width', 250); print(df.to_string(index=False))
