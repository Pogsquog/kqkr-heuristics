import numpy as np
d = np.fromfile('dtc.bin', dtype=np.uint8)[:1 << 24]
i = np.nonzero(d != 255)[0]; v = d[i].astype(int)
wk, wq, bk, br = i >> 18, (i >> 12) & 63, (i >> 6) & 63, i & 63
F = lambda s: s & 7; R = lambda s: s >> 3
cheb = lambda a, b: np.maximum(abs(F(a) - F(b)), abs(R(a) - R(b)))
ed = lambda s: np.minimum(np.minimum(F(s), 7 - F(s)), np.minimum(R(s), 7 - R(s)))
co = lambda s: np.minimum(np.minimum(cheb(s, 0), cheb(s, 7)), np.minimum(cheb(s, 56), cheb(s, 63)))
def grid(nx, x, ny, y):
    print(f"max DTC, rows={nx}, cols={ny}   (mean in brackets)")
    for a in np.unique(x):
        row = []
        for b in np.unique(y):
            m = (x == a) & (y == b)
            row.append(f"{v[m].max():2d}({v[m].mean():4.1f})" if m.any() else "   -     ")
        print(f"{a:2d}: " + ' '.join(row))
grid('ed(bk)', ed(bk), 'kk', cheb(wk, bk))
grid('co(bk)', co(bk), 'kk', cheb(wk, bk))
grid('ed(bk)', ed(bk), 'rk', cheb(br, bk))
print("percentiles of DTC (50/90/99/99.9/max) by rk:")
for r in range(1, 8):
    m = cheb(br, bk) == r
    print(r, np.percentile(v[m], [50, 90, 99, 99.9]).round(1), v[m].max(), m.sum())
