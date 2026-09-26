// Policy harness: White plays argmin of a depth-d minimax over a linear eval of features;
// then an exact fixpoint over all positions checks the policy wins against EVERY defence.
// usage: ./pol [-r] [-v] depth feat=weight ...   (-r: adversarial tie-breaking, -v: examples)
#include "kq.h"
#include <math.h>

uint8_t k3W[N3], k3B[N3], dW[N], dB[N];
int32_t *TW, *TB;
static int TBQ = 1;
#define INF 1000000000
#define MATE (-100000000)
#define CAPW (-50000000)
#define TERM (-10000000)

#include "feat.h"

static inline int adjterm(int v) { return v < TERM ? v + 1 : v; }

// value of a white move child
static inline int wchild(int s, int c3, const int32_t *tb) {
  if (s == S_CAPR) return k3B[c3] == 255 ? INF : CAPW;
  return tb[s];
}

static void sweepB(const int32_t *tw, int32_t *tb) {  // B value = max over black replies
#pragma omp parallel for schedule(dynamic, 4096)
  for (int i = 0; i < N; i++) {
    UNPACK(i);
    if (!legalB(wk, wq, bk, br)) continue;
    int out[32], k = gen_black(wk, wq, bk, br, out), v;
    if (!k) v = bk_in_check(wk, wq, bk, br) ? MATE : INF;
    else {
      v = -INF;
      for (int j = 0; j < k; j++) v = MAX(v, out[j] == S_LOSS ? INF : tw[out[j]]);
    }
    tb[i] = adjterm(v);
  }
}
static void sweepW(const int32_t *tb, int32_t *tw) {  // W value = min over white moves
#pragma omp parallel for schedule(dynamic, 4096)
  for (int i = 0; i < N; i++) {
    UNPACK(i);
    if (!legalW(wk, wq, bk, br)) continue;
    int out[40], cap[40], k = gen_white(wk, wq, bk, br, out, cap), v = INF;
    for (int j = 0; j < k; j++) v = MIN(v, wchild(out[j], cap[j], tb));
    tw[i] = adjterm(v);
  }
}
static void leaf(int32_t *t, int stm) {
#pragma omp parallel for schedule(dynamic, 4096)
  for (int i = 0; i < N; i++) {
    UNPACK(i);
    if (stm ? legalB(wk, wq, bk, br) : legalW(wk, wq, bk, br)) t[i] = eval(wk, wq, bk, br, stm);
  }
}

static const char *sq(int s) {
  static char b[8][3];
  static int k;
  char *p = b[k++ & 7];
  p[0] = 'a' + F(s), p[1] = '1' + R(s), p[2] = 0;
  return p;
}

int main(int argc, char **argv) {
  int robust = 0, verbose = 0, a = 1, trace = -1;
  while (a < argc && argv[a][0] == '-') {
    if (argv[a][1] == 'r') robust = 1;
    if (argv[a][1] == 'v') verbose = 1;
    if (argv[a][1] == 't') trace = atoi(argv[++a]);
    a++;
  }
  int depth = atoi(argv[a++]);
  for (; a < argc; a++) {
    char name[32];
    double w;
    if (sscanf(argv[a], "%31[^=]=%lf", name, &w) != 2) { fprintf(stderr, "bad arg %s\n", argv[a]); return 1; }
    int k = 0;
    while (FNAMES[k] && strcmp(FNAMES[k], name)) k++;
    if (!FNAMES[k]) { fprintf(stderr, "unknown feature %s\n", name); return 1; }
    W8[k] = (int)lround(w), USE[nuse++] = k;
  }
  if (getenv("TBQ")) TBQ = atoi(getenv("TBQ"));
  init_geom();
  init_lut();
  FILE *f = fopen("kqk.bin", "rb");
  if (!f || fread(k3W, 1, N3, f) != N3 || fread(k3B, 1, N3, f) != N3) return 1;
  fclose(f);
  f = fopen("dtc.bin", "rb");
  if (!f || fread(dW, 1, N, f) != N || fread(dB, 1, N, f) != N) return 1;
  fclose(f);
  TW = malloc(4L * N), TB = malloc(4L * N);

  // depth-d search tables; root children need TB at depth d-1
  if (depth & 1) leaf(TB, 1);
  else leaf(TW, 0), sweepB(TW, TB);
  for (int d = (depth & 1) ? 1 : 2; d < depth; d += 2) sweepW(TB, TW), sweepB(TW, TB);
  // root best value per W position (stored in TW)
  sweepW(TB, TW);

  // fixpoint: pW = white moves to conversion under the policy vs all defences
  uint8_t *pW = malloc(N), *pB = malloc(N);
  memset(pW, 255, N), memset(pB, 255, N);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int i = 0; i < N; i++) {
    UNPACK(i);
    int out[32];
    if (legalB(wk, wq, bk, br) && !gen_black(wk, wq, bk, br, out) && bk_in_check(wk, wq, bk, br)) pB[i] = 0;
  }
  for (int n = 1; n <= 100; n++) {
    long ch = 0;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : ch)
    for (int i = 0; i < N; i++) {
      if (pW[i] != 255 || dW[i] == 255) continue;  // only positions that are theoretically won
      UNPACK(i);
      int out[40], cap[40], k = gen_white(wk, wq, bk, br, out, cap);
      int best = TW[i], okc = 0, badc = 0;
      for (int j = 0; j < k; j++) {
        if (adjterm(wchild(out[j], cap[j], TB)) != best) continue;
        int good = out[j] == S_CAPR ? k3B[cap[j]] != 255 : pB[out[j]] <= n - 1;
        if (good) okc++; else badc++;
        if (!robust) break;  // deterministic: first best move in generation order
      }
      if (okc && !badc) pW[i] = n, ch++;
    }
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : ch)
    for (int i = 0; i < N; i++) {
      if (pB[i] != 255 || dB[i] == 255) continue;
      UNPACK(i);
      int out[32], k = gen_black(wk, wq, bk, br, out), all = k > 0;
      for (int j = 0; j < k && all; j++)
        if (out[j] == S_LOSS || pW[out[j]] > n) all = 0;
      if (all) pB[i] = n, ch++;
    }
    if (!ch) break;
  }
  long won = 0, ok50 = 0, okany = 0, hist[256] = {0};
  double sum = 0, sumopt = 0;
  int mx = 0;
  for (int i = 0; i < N; i++) {
    if (dW[i] == 255) continue;
    UNPACK(i);
    if (!legalW(wk, wq, bk, br)) continue;
    won++;
    if (pW[i] != 255) {
      okany++, hist[pW[i]]++, mx = MAX(mx, pW[i]);
      if (pW[i] <= 50) ok50++, sum += pW[i], sumopt += dW[i];
    }
  }
  printf("depth=%d%s won=%ld ok50=%ld (%.4f%%) fail=%ld okany=%ld max=%d mean=%.2f (opt %.2f)\n", depth,
         robust ? " robust" : "", won, ok50, 100.0 * ok50 / won, won - ok50, okany, mx, sum / MAX(ok50, 1),
         sumopt / MAX(ok50, 1));
  if (trace >= 0) {  // play policy vs a defender that avoids policy-won positions, max DTC
    int i = trace;
    for (int ply = 0; ply < 60; ply++) {
      UNPACK(i);
      printf("%2d. WK %s WQ %s BK %s BR %s  dtc=%d pol=%d val=%d\n", ply + 1, sq(wk), sq(wq), sq(bk), sq(br), dW[i], pW[i], TW[i]);
      int out[40], cap[40], k = gen_white(wk, wq, bk, br, out, cap), b = -3;
      for (int j = 0; j < k; j++)
        if (adjterm(wchild(out[j], cap[j], TB)) == TW[i]) { b = out[j]; break; }
      if (b < 0) { printf("   -> %s\n", b == S_CAPR ? "captures rook" : "??"); break; }
      int bo[32], bn = gen_black(b >> 18, (b >> 12) & 63, (b >> 6) & 63, b & 63, bo), best = -1, bs = -1;
      if (!bn) { printf("   -> mate/stalemate\n"); break; }
      for (int j = 0; j < bn; j++) {
        if (bo[j] == S_LOSS) { best = -2; break; }
        int sc = (pW[bo[j]] == 255 ? 1000 : pW[bo[j]]) * 256 + dW[bo[j]];
        if (sc > bs) bs = sc, best = bo[j];
      }
      if (best == -2) { printf("   -> black wins queen\n"); break; }
      i = best;
    }
  }
  if (verbose) {
    int shown = 0;
    for (int i = 0; i < N && shown < 12; i += 7919) {
      if (dW[i] == 255 || pW[i] != 255) continue;
      UNPACK(i);
      if (!legalW(wk, wq, bk, br)) continue;
      printf("  fail[%d]: WK %s WQ %s BK %s BR %s (dtc %d)\n", i, sq(wk), sq(wq), sq(bk), sq(br), dW[i]);
      shown++;
    }
  }
  return 0;
}
