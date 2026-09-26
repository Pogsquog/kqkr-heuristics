import numpy as np
d = np.fromfile('dtc.bin', dtype=np.uint8)[:1 << 24]
i = np.nonzero(d != 255)[0]; v = d[i].astype(int)
wk, wq, bk, br = i >> 18, (i >> 12) & 63, (i >> 6) & 63, i & 63
F = lambda s: s & 7; R = lambda s: s >> 3
cheb = lambda a, b: np.maximum(abs(F(a) - F(b)), abs(R(a) - R(b)))
ed = lambda s: np.minimum(np.minimum(F(s), 7 - F(s)), np.minimum(R(s), 7 - R(s)))
q = cheb(br, bk) == 1
print("rook adjacent: count", q.sum())
for k in [3, 6, 10, 15, 20]:
    m = q & (v <= k)
    print(f"DTC<={k}: {m.sum()} ({100*m.sum()/q.sum():.1f}%)  ed(bk) dist:", np.bincount(ed(bk[m]), minlength=4), " kk dist:", np.bincount(cheb(wk[m], bk[m]), minlength=8)[2:])
