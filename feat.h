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
// Defender's options with the rook. A destination t is "safe" if White cannot take the rook there
// for free: not attacked by the white king or queen, or defended by the black king.
// which: 0 = legal rook moves, 1 = safe moves, 2 = safe checks, 3 = safe moves attacking the queen, 4 = any checks
static inline int rookopt(int wk, int wq, int bk, int br, int which) {
  int n = 0;
  for (int d = 0; d < 4; d++) {
    int f = F(br), r = R(br);
    for (;;) {
      f += DF[d], r += DR[d];
      if (f < 0 || f > 7 || r < 0 || r > 7) break;
      int t = r * 8 + f;
      if (t == bk || t == wk) break;
      if (t == wq) { n += which < 2; break; }  // capturing the queen
      if (qatt(wq, bk, BIT(wk) | BIT(t))) continue;  // rook pinned against the king
      int safe = adj(bk, t) || (!adj(wk, t) && !qatt(wq, t, BIT(wk) | BIT(bk)));
      if (which == 0) n++;
      else if (safe && which == 1) n++;
      else if (safe && which == 2) n += ratt(t, wk, BIT(wq) | BIT(bk));
      else if (safe && which == 3) n += ratt(t, wq, BIT(wk) | BIT(bk));
      else if (which == 4) n += ratt(t, wk, BIT(wq) | BIT(bk));
    }
  }
  return n;
}
// squares the black king can reach by king moves through squares not attacked by White (rook blocks the queen)
static inline int kregion(int wk, int wq, int bk, int br) {
  uint64_t free = 0;
  for (int s = 0; s < 64; s++)
    if (s != wq && s != br && !adj(s, wk) && !qatt(wq, s, BIT(wk) | BIT(br))) free |= BIT(s);
  uint64_t reg = BIT(bk), prev = 0;
  while (reg != prev) {
    prev = reg;
    for (uint64_t m = reg; m; m &= m - 1) reg |= KATT[__builtin_ctzll(m)] & free;
  }
  return __builtin_popcountll(reg) - 1;
}
// same, but the rook does not block the queen: the box drawn by the queen and white king alone
static inline int kbox(int wk, int wq, int bk) {
  uint64_t free = 0;
  for (int s = 0; s < 64; s++)
    if (s != wq && !adj(s, wk) && !qatt(wq, s, BIT(wk))) free |= BIT(s);
  uint64_t reg = BIT(bk), prev = 0;
  while (reg != prev) {
    prev = reg;
    for (uint64_t m = reg; m; m &= m - 1) reg |= KATT[__builtin_ctzll(m)] & free;
  }
  return __builtin_popcountll(reg) - 1;
}
// Stricter desperado test (black to move, king boxed in): the rook has a check on a square where
// no White capture of it reaches a won KQK (capture impossible, or every capture draws).
static inline int desp2(int wk, int wq, int bk, int br) {
  if (bkmob(wk, wq, bk, br) || bk_in_check(wk, wq, bk, br)) return 0;
  for (int d = 0; d < 4; d++) {
    int f = F(br), r = R(br);
    for (;;) {
      f += DF[d], r += DR[d];
      if (f < 0 || f > 7 || r < 0 || r > 7) break;
      int t = r * 8 + f;
      if (t == bk || t == wk || t == wq) break;
      if (qatt(wq, bk, BIT(wk) | BIT(t))) continue;
      if (!ratt(t, wk, BIT(wq) | BIT(bk))) continue;
      int win = 0;  // can White take the checking rook into a won KQK?
      if (adj(wk, t) && !adj(bk, t) && k3B[IDX3(t, wq, bk)] != 255) win = 1;
      if (qatt(wq, t, BIT(wk) | BIT(bk)) && !adj(bk, t) && k3B[IDX3(wk, t, bk)] != 255) win = 1;
      if (!win) return 1;
    }
  }
  return 0;
}
static const char *FNAMES[] = {"cd",  "kk",   "kkm",  "rk",  "qk",   "mob", "ed", "co",
                               "box", "rkm",  "qkm",  "wkc", "wkcd", "tb",  "rkk", "one", "lut", "kqk",
                               "rmob", "rsafe", "rchk", "rqa", "kreg", "bkst", "desp", "desp2", "kbox", 0};
enum { NF = 27 };
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
    case 18: return rookopt(wk, wq, bk, br, 0);
    case 19: return rookopt(wk, wq, bk, br, 1);
    case 20: return rookopt(wk, wq, bk, br, 2);
    case 21: return rookopt(wk, wq, bk, br, 3);
    case 22: return kregion(wk, wq, bk, br);
    case 23: return bkmob(wk, wq, bk, br) == 0;  // black king boxed in
    case 24:  // desperado rook: king boxed in and the rook is checking or can check (capture = stalemate)
      return bkmob(wk, wq, bk, br) == 0 && !bk_in_check(wk, wq, bk, br) &&
             (wk_in_check(wk, wq, bk, br) || rookopt(wk, wq, bk, br, 4) > 0);
    case 25: return stm == 1 && desp2(wk, wq, bk, br);
    case 26: return kbox(wk, wq, bk);
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

