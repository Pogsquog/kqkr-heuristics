// Builds KQK (DTM) and KQKR (DTC and DTM) tables by iterated forward fixpoint.
// Values = number of white moves still needed; 255 = not a white win.
#include "kq.h"

uint8_t k3W[N3], k3B[N3];
uint8_t dW[N], dB[N];

static void build_kqk(void) {
  memset(k3W, 255, N3), memset(k3B, 255, N3);
  // mates (black to move)
  for (int i = 0; i < N3; i++) {
    int wk = i >> 12, wq = (i >> 6) & 63, bk = i & 63;
    if (wk == wq || wk == bk || wq == bk || adj(wk, bk)) continue;
    int moves = 0;
    for (uint64_t m = KATT[bk]; m; m &= m - 1) {
      int s = __builtin_ctzll(m);
      if (adj(s, wk)) continue;
      if (s != wq && qatt(wq, s, BIT(wk))) continue;
      moves++;
    }
    if (!moves && qatt(wq, bk, BIT(wk))) k3B[i] = 0;
  }
  for (int n = 1; n < 40; n++) {
    int ch = 0;
    for (int i = 0; i < N3; i++) {
      int wk = i >> 12, wq = (i >> 6) & 63, bk = i & 63;
      if (k3W[i] != 255 || wk == wq || wk == bk || wq == bk || adj(wk, bk) || qatt(wq, bk, BIT(wk))) continue;
      int ok = 0;
      for (uint64_t m = KATT[wk]; m && !ok; m &= m - 1) {
        int s = __builtin_ctzll(m);
        if (s == wq || adj(s, bk)) continue;
        if (k3B[IDX3(s, wq, bk)] <= n - 1) ok = 1;
      }
      for (int d = 0; d < 8 && !ok; d++) {
        int f = F(wq), r = R(wq);
        for (;;) {
          f += DF[d], r += DR[d];
          if (f < 0 || f > 7 || r < 0 || r > 7) break;
          int t = r * 8 + f;
          if (t == wk || t == bk) break;
          if (k3B[IDX3(wk, t, bk)] <= n - 1) { ok = 1; break; }
        }
      }
      if (ok) k3W[i] = n, ch++;
    }
    for (int i = 0; i < N3; i++) {
      int wk = i >> 12, wq = (i >> 6) & 63, bk = i & 63;
      if (k3B[i] != 255 || wk == wq || wk == bk || wq == bk || adj(wk, bk)) continue;
      int moves = 0, all = 1;
      for (uint64_t m = KATT[bk]; m; m &= m - 1) {
        int s = __builtin_ctzll(m);
        if (adj(s, wk)) continue;
        if (s == wq) { moves++, all = 0; continue; }
        if (qatt(wq, s, BIT(wk))) continue;
        moves++;
        if (k3W[IDX3(wk, wq, s)] > n) all = 0;
      }
      if (moves && all) k3B[i] = n, ch++;
    }
    if (!ch) break;
  }
}

// mode 0 = DTC (capture of rook into won KQK ends the count), 1 = DTM
static void build_kqkr(int mode) {
  memset(dW, 255, N), memset(dB, 255, N);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int i = 0; i < N; i++) {
    UNPACK(i);
    if (!legalB(wk, wq, bk, br)) continue;
    int out[32];
    if (gen_black(wk, wq, bk, br, out) == 0 && bk_in_check(wk, wq, bk, br)) dB[i] = 0;
  }
  for (int n = 1; n < 120; n++) {
    long ch = 0;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : ch)
    for (int i = 0; i < N; i++) {
      if (dW[i] != 255) continue;
      UNPACK(i);
      if (!legalW(wk, wq, bk, br)) continue;
      int out[40], cap[40];
      int k = gen_white(wk, wq, bk, br, out, cap);
      for (int j = 0; j < k; j++) {
        int v = out[j] == S_CAPR ? (k3B[cap[j]] == 255 ? 255 : mode ? k3B[cap[j]] : 0) : dB[out[j]];
        if (v <= n - 1) { dW[i] = n, ch++; break; }
      }
    }
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : ch)
    for (int i = 0; i < N; i++) {
      if (dB[i] != 255) continue;
      UNPACK(i);
      if (!legalB(wk, wq, bk, br)) continue;
      int out[32];
      int k = gen_black(wk, wq, bk, br, out), all = k > 0;
      for (int j = 0; j < k && all; j++)
        if (out[j] == S_LOSS || dW[out[j]] > n) all = 0;
      if (all) dB[i] = n, ch++;
    }
    fprintf(stderr, "mode %d n=%d changed=%ld\n", mode, n, ch);
    if (!ch) break;
  }
}

static void stats(const char *name) {
  long legal = 0, won = 0, hist[256] = {0};
  int mx = 0;
  for (int i = 0; i < N; i++) {
    UNPACK(i);
    if (!legalW(wk, wq, bk, br)) continue;
    legal++;
    if (dW[i] != 255) won++, hist[dW[i]]++, mx = MAX(mx, dW[i]);
  }
  printf("%s: WTM legal=%ld won=%ld (%.3f%%) max=%d\n", name, legal, won, 100.0 * won / legal, mx);
}

int main(void) {
  init_geom();
  build_kqk();
  int mx = 0;
  for (int i = 0; i < N3; i++) if (k3W[i] != 255) mx = MAX(mx, k3W[i]);
  printf("KQK max DTM (wtm) = %d\n", mx);
  FILE *f = fopen("kqk.bin", "wb");
  fwrite(k3W, 1, N3, f), fwrite(k3B, 1, N3, f), fclose(f);
  build_kqkr(0);
  stats("KQKR DTC");
  f = fopen("dtc.bin", "wb");
  fwrite(dW, 1, N, f), fwrite(dB, 1, N, f), fclose(f);
  build_kqkr(1);
  stats("KQKR DTM");
  f = fopen("dtm.bin", "wb");
  fwrite(dW, 1, N, f), fwrite(dB, 1, N, f), fclose(f);
  return 0;
}
