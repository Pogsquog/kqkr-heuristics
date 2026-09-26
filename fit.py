# Fit an additive lookup-table model V ~ DTC over won white-to-move positions (8-fold symmetry).
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, sys
d = np.fromfile('dtc.bin', dtype=np.uint8)[:1 << 24]
i = np.nonzero(d != 255)[0]; y = d[i].astype(np.float64)
wk, wq, bk, br = i >> 18, (i >> 12) & 63, (i >> 6) & 63, i & 63
def tf(s, t):  # dihedral transform t of square s
    f, r = s & 7, s >> 3
    if t & 1: f = 7 - f
    if t & 2: r = 7 - r
    if t & 4: f, r = r, f
    return r * 8 + f
TRI = {}
for s in range(64):
    f, r = s & 7, s >> 3
    if f <= 3 and r <= f: TRI[s] = len(TRI)
# choose transform putting bk in triangle
tsel = np.full(64, -1)
for s in range(64):
    for t in range(8):
        if tf(s, t) in TRI: tsel[s] = t; break
t = tsel[bk]
def T(s):
    out = np.empty_like(s)
    for k in range(8):
        m = t == k; out[m] = tf(s[m], k)
    return out
bkc, wkc, wqc, brc = T(bk), T(wk), T(wq), T(br)
tri = np.array([TRI.get(s, 0) for s in range(64)])[bkc]
def rel(a, b): return ((a & 7) - (b & 7) + 7) * 15 + ((a >> 3) - (b >> 3) + 7)
groups = sys.argv[1].split(',') if len(sys.argv) > 1 else ['A', 'B', 'C', 'D', 'E', 'G']
cols, off = [], 0
spec = {'A': (tri * 64 + wkc, 640), 'B': (tri * 64 + wqc, 640), 'C': (tri * 64 + brc, 640),
        'D': (rel(wq, br), 225), 'E': (rel(wk, br), 225), 'G': (rel(wk, wq), 225)}
layout = []
for g in ['A', 'B', 'C', 'D', 'E', 'G']:
    c, n = spec[g]
    if g in groups: cols.append(c + off)
    layout.append((g, off, n)); off += n
nr = len(y)
X = sp.csr_matrix((np.ones(nr * len(cols)), (np.repeat(np.arange(nr), len(cols)), np.stack(cols, 1).ravel())), shape=(nr, off))
w = sla.lsqr(X, y, damp=1e-3, atol=1e-8, btol=1e-8, iter_lim=400)[0]
pred = X @ w
print('params used', sum(n for g, o, n in layout if g in groups), 'rmse', np.sqrt(np.mean((pred - y) ** 2)), 'corr', np.corrcoef(pred, y)[0, 1])
w.astype(np.float32).tofile('lut.bin')
