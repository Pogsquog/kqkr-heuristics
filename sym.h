// 8-fold symmetry reduction and a precomputed successor graph over canonical positions.
// Compact index c = (TRI[bk] << 18) | (wk << 12) | (wq << 6) | br, with bk mapped into a1-d1-d4.
// When bk lies on the a1-h8 diagonal the transpose also fixes it; the smaller index is canonical.
// Needs k3B (KQK table) loaded before build_graph().
#pragma once
#include "kq.h"

#define NC (10 << 18)
static int TSEL[64], TRI[64], TRISQ[10];
static uint8_t TF[8][64];
static inline int tfs(int s, int t) {
  int f = F(s), r = R(s), x;
  if (t & 1) f = 7 - f;
  if (t & 2) r = 7 - r;
  if (t & 4) x = f, f = r, r = x;
  return r * 8 + f;
}
static void init_sym(void) {
  int n = 0;
  for (int s = 0; s < 64; s++) {
    TRI[s] = (F(s) <= 3 && R(s) <= F(s)) ? n++ : -1;
    if (TRI[s] >= 0) TRISQ[TRI[s]] = s;
  }
  for (int s = 0; s < 64; s++) {
    for (int t = 0; t < 8; t++) TF[t][s] = tfs(s, t);
    for (int t = 7; t >= 0; t--) if (TRI[tfs(s, t)] >= 0) TSEL[s] = t;
  }
}
static inline int canon(int wk, int wq, int bk, int br) {
  int t = TSEL[bk], b = TF[t][bk], k = TF[t][wk], q = TF[t][wq], r = TF[t][br];
  int c = (TRI[b] << 18) | (k << 12) | (q << 6) | r;
  if (F(b) == R(b)) {
    int c2 = (TRI[b] << 18) | (TF[4][k] << 12) | (TF[4][q] << 6) | TF[4][r];
    if (c2 < c) c = c2;
  }
  return c;
}
static inline int canon_full(int i) { UNPACK(i); return canon(wk, wq, bk, br); }
#define CUNPACK(c) int bk = TRISQ[(c) >> 18], wk = ((c) >> 12) & 63, wq = ((c) >> 6) & 63, br = (c) & 63
static inline int cfull(int c) { CUNPACK(c); return IDX(wk, wq, bk, br); }

// Children: W -> compact B index, G_CONV (safe rook capture) or G_BAD (capture into drawn KQK);
//           B -> compact W index or G_BAD (queen lost).
enum { G_CONV = -1, G_BAD = -2 };
enum { L_OK = 1, L_MATE = 2, L_STALE = 4 };
static uint8_t *legW, *legB;  // 0 = illegal or non-canonical; legB also carries L_MATE / L_STALE
static uint8_t *multW;        // number of full-board W positions this canonical position stands for
static uint32_t *offW, *offB;
static int32_t *chW, *chB;

__attribute__((unused)) static void build_graph(void) {
  legW = calloc(NC, 1), legB = calloc(NC, 1), multW = calloc(NC, 1);
  offW = calloc(NC + 1, 4), offB = calloc(NC + 1, 4);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int c = 0; c < NC; c++) {
    CUNPACK(c);
    if (canon(wk, wq, bk, br) != c) continue;
    int out[40], cap[40];
    if (legalW(wk, wq, bk, br)) {
      legW[c] = L_OK, offW[c + 1] = gen_white(wk, wq, bk, br, out, cap);
      int img[8], m = 0;
      for (int t = 0; t < 8; t++) {
        int v = IDX(TF[t][wk], TF[t][wq], TF[t][bk], TF[t][br]), dup = 0;
        for (int j = 0; j < m; j++) dup |= img[j] == v;
        if (!dup) img[m++] = v;
      }
      multW[c] = m;
    }
    if (legalB(wk, wq, bk, br)) {
      int k = gen_black(wk, wq, bk, br, out);
      legB[c] = L_OK | (k ? 0 : bk_in_check(wk, wq, bk, br) ? L_MATE : L_STALE), offB[c + 1] = k;
    }
  }
  for (int c = 0; c < NC; c++) offW[c + 1] += offW[c], offB[c + 1] += offB[c];
  chW = malloc(4L * offW[NC]), chB = malloc(4L * offB[NC]);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int c = 0; c < NC; c++) {
    CUNPACK(c);
    int out[40], cap[40];
    if (legW[c]) {
      int k = gen_white(wk, wq, bk, br, out, cap);
      for (int j = 0; j < k; j++)
        chW[offW[c] + j] = out[j] == S_CAPR ? (k3B[cap[j]] != 255 ? G_CONV : G_BAD) : canon_full(out[j]);
    }
    if (legB[c]) {
      int k = gen_black(wk, wq, bk, br, out);
      for (int j = 0; j < k; j++) chB[offB[c] + j] = out[j] == S_LOSS ? G_BAD : canon_full(out[j]);
    }
  }
}
