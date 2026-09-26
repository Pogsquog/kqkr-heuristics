// KQ vs KR core: board geometry, legality, move generation.
// Position index: (wk<<18)|(wq<<12)|(bk<<6)|br. W* = white to move, B* = black to move.
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N (1 << 24)
#define N3 (1 << 18)
#define IDX(wk, wq, bk, br) (((wk) << 18) | ((wq) << 12) | ((bk) << 6) | (br))
#define IDX3(wk, wq, bk) (((wk) << 12) | ((wq) << 6) | (bk))
#define BIT(s) (1ULL << (s))

// Successor codes for moves that leave KQKR.
#define S_CAPR (-1)  // white captured the rook (-> KQK, black to move); see kqk table
#define S_LOSS (-2)  // black captured the queen

static uint64_t KATT[64], BETWEEN[64][64];
static uint8_t LINE[64][64];  // 1 = rank/file, 2 = diagonal
static const int DF[8] = {1, -1, 0, 0, 1, 1, -1, -1}, DR[8] = {0, 0, 1, -1, 1, -1, 1, -1};

static inline int F(int s) { return s & 7; }
static inline int R(int s) { return s >> 3; }
static inline int ABS(int x) { return x < 0 ? -x : x; }
static inline int MAX(int a, int b) { return a > b ? a : b; }
static inline int MIN(int a, int b) { return a < b ? a : b; }
static inline int cheb(int a, int b) { return MAX(ABS(F(a) - F(b)), ABS(R(a) - R(b))); }
static inline int manh(int a, int b) { return ABS(F(a) - F(b)) + ABS(R(a) - R(b)); }
static inline int adj(int a, int b) { return cheb(a, b) <= 1; }

static void init_geom(void) {
  for (int a = 0; a < 64; a++) {
    KATT[a] = 0;
    for (int b = 0; b < 64; b++) {
      BETWEEN[a][b] = 0;
      LINE[a][b] = 0;
      if (a != b && cheb(a, b) == 1) KATT[a] |= BIT(b);
    }
    for (int d = 0; d < 8; d++) {
      int f = F(a), r = R(a);
      uint64_t path = 0;
      for (;;) {
        f += DF[d], r += DR[d];
        if (f < 0 || f > 7 || r < 0 || r > 7) break;
        int b = r * 8 + f;
        LINE[a][b] = d < 4 ? 1 : 2;
        BETWEEN[a][b] = path;
        path |= BIT(b);
      }
    }
  }
}

static inline int qatt(int q, int s, uint64_t occ) { return LINE[q][s] && !(BETWEEN[q][s] & occ); }
static inline int ratt(int r, int s, uint64_t occ) { return LINE[r][s] == 1 && !(BETWEEN[r][s] & occ); }

static inline int distinct4(int a, int b, int c, int d) {
  return a != b && a != c && a != d && b != c && b != d && c != d;
}
static inline int legalW(int wk, int wq, int bk, int br) {
  return distinct4(wk, wq, bk, br) && !adj(wk, bk) && !qatt(wq, bk, BIT(wk) | BIT(br));
}
static inline int legalB(int wk, int wq, int bk, int br) {
  return distinct4(wk, wq, bk, br) && !adj(wk, bk) && !ratt(br, wk, BIT(wq) | BIT(bk));
}
static inline int bk_in_check(int wk, int wq, int bk, int br) { return qatt(wq, bk, BIT(wk) | BIT(br)); }
static inline int wk_in_check(int wk, int wq, int bk, int br) { return ratt(br, wk, BIT(wq) | BIT(bk)); }

// White moves from W(wk,wq,bk,br). out[i] = B index, or S_CAPR with cap3[i] = KQK B index.
static inline int gen_white(int wk, int wq, int bk, int br, int *out, int *cap3) {
  int n = 0;
  for (uint64_t m = KATT[wk]; m; m &= m - 1) {
    int s = __builtin_ctzll(m);
    if (s == wq || adj(s, bk)) continue;
    if (s == br) { out[n] = S_CAPR; cap3[n++] = IDX3(s, wq, bk); continue; }
    if (ratt(br, s, BIT(wq) | BIT(bk))) continue;
    out[n++] = IDX(s, wq, bk, br);
  }
  for (int d = 0; d < 8; d++) {
    int f = F(wq), r = R(wq);
    for (;;) {
      f += DF[d], r += DR[d];
      if (f < 0 || f > 7 || r < 0 || r > 7) break;
      int t = r * 8 + f;
      if (t == wk || t == bk) break;
      if (t == br) { out[n] = S_CAPR; cap3[n++] = IDX3(wk, t, bk); break; }
      if (ratt(br, wk, BIT(t) | BIT(bk))) continue;
      out[n++] = IDX(wk, t, bk, br);
    }
  }
  return n;
}

// Black moves from B(wk,wq,bk,br). out[i] = W index or S_LOSS. Returns count.
static inline int gen_black(int wk, int wq, int bk, int br, int *out) {
  int n = 0;
  for (uint64_t m = KATT[bk]; m; m &= m - 1) {
    int s = __builtin_ctzll(m);
    if (s == br || adj(s, wk)) continue;
    if (s == wq) { out[n++] = S_LOSS; continue; }
    if (qatt(wq, s, BIT(wk) | BIT(br))) continue;
    out[n++] = IDX(wk, wq, s, br);
  }
  for (int d = 0; d < 4; d++) {
    int f = F(br), r = R(br);
    for (;;) {
      f += DF[d], r += DR[d];
      if (f < 0 || f > 7 || r < 0 || r > 7) break;
      int t = r * 8 + f;
      if (t == bk || t == wk) break;
      if (t == wq) { out[n++] = S_LOSS; break; }
      if (qatt(wq, bk, BIT(wk) | BIT(t))) continue;
      out[n++] = IDX(wk, wq, bk, t);
    }
  }
  return n;
}

#define UNPACK(i) int wk = (i) >> 18, wq = ((i) >> 12) & 63, bk = ((i) >> 6) & 63, br = (i) & 63
