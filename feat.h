// Shared evaluation features. Needs dW/dB (tablebase) and TBQ declared before inclusion.
#pragma once
#include <math.h>
#include "sym.h"
// ---- features (all integers; eval = sum w_i f_i, White minimises) ----
static inline int cdist(int s) { return MAX(3 - F(s), F(s) - 4) + MAX(3 - R(s), R(s) - 4); }  // 0..6
static inline int edist(int s) { return MIN(MIN(F(s), 7 - F(s)), MIN(R(s), 7 - R(s))); }       // 0..3
static inline int bkmob(int wk, int wq, int bk, int br) {
  int n = 0;
  for (uint64_t m = KATT[bk]; m; m &= m - 1) {
    int s = __builtin_ctzll(m);
    if (s == br || adj(s, wk)) continue;
    if (s != wq && qatt(wq, s, BIT(wk) | BIT(br))) continue;
    n++;
  }
  return n;
}
// squares of the rectangle the queen's rank+file confine the black king to (KRK-style box)
static inline int qbox(int wq, int bk) {
  int fw = F(bk) < F(wq) ? F(wq) : F(bk) > F(wq) ? 7 - F(wq) : 8;
  int rw = R(bk) < R(wq) ? R(wq) : R(bk) > R(wq) ? 7 - R(wq) : 8;
  return fw * rw;
}
static const char *FNAMES[] = {"cd",  "kk",   "kkm",  "rk",  "qk",   "mob", "ed", "co",
                               "box", "rkm",  "qkm",  "wkc", "wkcd", "tb",  "rkk", "one", "lut", "kqk", 0};
enum { NF = 18 };
static int W8[NF], USE[NF], nuse;

static inline int lut(int wk, int wq, int bk, int br);
static inline int feat(int k, int wk, int wq, int bk, int br, int stm) {
  switch (k) {
    case 0: return cdist(bk);                  // black king away from centre (want big)
    case 1: return cheb(wk, bk);               // king distance
    case 2: return manh(wk, bk);
    case 3: return cheb(br, bk);               // rook-own king distance
    case 4: return cheb(wq, bk);
    case 5: return bkmob(wk, wq, bk, br);
    case 6: return edist(bk);
    case 7: return MIN(MIN(cheb(bk, 0), cheb(bk, 7)), MIN(cheb(bk, 56), cheb(bk, 63)));
    case 8: return qbox(wq, bk);
    case 9: return manh(br, bk);
    case 10: return manh(wq, bk);
    case 11: return cdist(wk);
    case 12: return stm == 0 && wk_in_check(wk, wq, bk, br);
    case 13: { int v = stm ? dB[IDX(wk, wq, bk, br)] : dW[IDX(wk, wq, bk, br)]; return v == 255 ? 99 : v / TBQ; }
    case 14: return cheb(br, wk);
    case 15: return 1;
    case 16: return lut(wk, wq, bk, br);
    case 17: {  // KQK distance-to-mate as if the rook were absent
      if (adj(wq, bk) && !adj(wq, wk)) return 12;  // queen en prise to the king
      int v = qatt(wq, bk, BIT(wk)) ? k3B[IDX3(wk, wq, bk)] : k3W[IDX3(wk, wq, bk)];
      return v == 255 ? 12 : v;
    }
  }
  return 0;
}
// additive lookup-table model written by fit.py (bk canonicalised into a1-d1-d4)
static float LUT[3 * 640 + 3 * 225];
static void init_lut(void) {
  init_sym();
  FILE *f = fopen("lut.bin", "rb");
  if (!f) return;
  if (fread(LUT, 4, 3 * 640 + 3 * 225, f)) {}
  fclose(f);
}
static inline int rel(int a, int b) { return (F(a) - F(b) + 7) * 15 + (R(a) - R(b) + 7); }
static inline int lut(int wk, int wq, int bk, int br) {
  int t = TSEL[bk], tr = TRI[tfs(bk, t)];
  wk = tfs(wk, t), wq = tfs(wq, t), br = tfs(br, t), bk = tfs(bk, t);
  float v = LUT[tr * 64 + wk] + LUT[640 + tr * 64 + wq] + LUT[1280 + tr * 64 + br] + LUT[1920 + rel(wq, br)] +
            LUT[2145 + rel(wk, br)] + LUT[2370 + rel(wk, wq)];
  return (int)lrintf(v * 100);
}
static inline int eval(int wk, int wq, int bk, int br, int stm) {
  int v = 0;
  for (int j = 0; j < nuse; j++) v += W8[USE[j]] * feat(USE[j], wk, wq, bk, br, stm);
  return v;
}

