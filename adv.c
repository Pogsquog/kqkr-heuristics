// "Advice" policy harness. Given a potential V (linear in features, on white-to-move positions),
// White plays the move that forces V to strictly drop (or converts: mate / safe rook capture)
// in the fewest moves <= K, never letting V rise at an intermediate white-to-move position.
// The policy uses no tablebase knowledge; the tablebase is only used to score it afterwards.
// An exact fixpoint then checks the policy against every defence.
// usage: ./adv [-t idx] [-s] K feat=weight ...
#include "kq.h"

uint8_t k3W[N3], k3B[N3], dW[N], dB[N];
static int TBQ = 1;
#include "feat.h"

static int16_t *VW;
static uint8_t *dist;   // moves needed to force a V drop (255 = cannot within K)
static int32_t *pmove;  // chosen successor: B index, S_CAPR, or S_LOSS if none

static const char *sq(int s) {
  static char b[8][3];
  static int k;
  char *p = b[k++ & 7];
  p[0] = 'a' + F(s), p[1] = '1' + R(s), p[2] = 0;
  return p;
}

// can White, after reaching B position b from a W position at level t, be sure the next
// W position is a drop or a same-level position with dist <= lim?
static inline int bgood(int b, int t, int lim) {
  UNPACK(b);
  int out[32], k = gen_black(wk, wq, bk, br, out);
  if (!k) return bk_in_check(wk, wq, bk, br);  // mate good, stalemate bad
  for (int j = 0; j < k; j++) {
    int q = out[j];
    if (q == S_LOSS) return 0;
    int v = VW[q];
    if (v < t) continue;
    if (v > t || dist[q] > lim) return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  int a = 1, trace = -1, quiet = 0;
  while (a < argc && argv[a][0] == '-') {
    if (argv[a][1] == 't') trace = atoi(argv[++a]);
    if (argv[a][1] == 's') quiet = 1;
    a++;
  }
  int K = atoi(argv[a++]);
  for (; a < argc; a++) {
    char name[32];
    double w;
    if (sscanf(argv[a], "%31[^=]=%lf", name, &w) != 2) { fprintf(stderr, "bad arg %s\n", argv[a]); return 1; }
    int k = 0;
    while (FNAMES[k] && strcmp(FNAMES[k], name)) k++;
    if (!FNAMES[k]) { fprintf(stderr, "unknown feature %s\n", name); return 1; }
    W8[k] = (int)lround(w), USE[nuse++] = k;
  }
  init_geom();
  init_lut();
  FILE *f = fopen("kqk.bin", "rb");
  if (!f || fread(k3W, 1, N3, f) != N3 || fread(k3B, 1, N3, f) != N3) return 1;
  fclose(f);
  f = fopen("dtc.bin", "rb");
  if (!f || fread(dW, 1, N, f) != N || fread(dB, 1, N, f) != N) return 1;
  fclose(f);
  VW = malloc(2L * N), dist = malloc(N), pmove = malloc(4L * N);
  memset(dist, 255, N);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int i = 0; i < N; i++) {
    UNPACK(i);
    VW[i] = legalW(wk, wq, bk, br) ? eval(wk, wq, bk, br, 0) : 0;
    pmove[i] = S_LOSS;
  }
  for (int n = 1; n <= K; n++) {
    long ch = 0;
#pragma omp parallel for schedule(dynamic, 1024) reduction(+ : ch)
    for (int i = 0; i < N; i++) {
      if (dist[i] != 255) continue;
      UNPACK(i);
      if (!legalW(wk, wq, bk, br)) continue;
      int out[40], cap[40], k = gen_white(wk, wq, bk, br, out, cap), t = VW[i];
      int pick = -1;
      for (int j = 0; j < k && pick < 0; j++)  // conversion first: safe rook capture or mate
        if (out[j] == S_CAPR ? k3B[cap[j]] != 255 : bgood(out[j], -(1 << 30), 0)) pick = j;
      for (int j = 0; j < k && pick < 0; j++)
        if (out[j] != S_CAPR && bgood(out[j], t, n - 1)) pick = j;
      if (pick >= 0) dist[i] = n, pmove[i] = out[pick], ch++;
    }
    if (!ch) break;
  }

  // exact verification: moves to conversion under the policy against all defences
  uint8_t *pW = malloc(N), *pB = malloc(N);
  memset(pW, 255, N), memset(pB, 255, N);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int i = 0; i < N; i++) {
    UNPACK(i);
    int out[32];
    if (legalB(wk, wq, bk, br) && !gen_black(wk, wq, bk, br, out) && bk_in_check(wk, wq, bk, br)) pB[i] = 0;
  }
  for (int n = 1; n <= 250; n++) {
    long ch = 0;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : ch)
    for (int i = 0; i < N; i++) {
      if (pW[i] != 255 || dist[i] == 255) continue;
      int m = pmove[i];
      if (m == S_CAPR || (m >= 0 && pB[m] <= n - 1)) pW[i] = n, ch++;
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
  long won = 0, cov = 0, okall = 0, ok50 = 0, dh[256] = {0};
  double sum = 0, sumopt = 0;
  int mx = 0, vmin = 1 << 30, vmax = -(1 << 30);
  for (int i = 0; i < N; i++) {
    if (dW[i] == 255) continue;
    UNPACK(i);
    if (!legalW(wk, wq, bk, br)) continue;
    won++, dh[dist[i]]++;
    vmin = MIN(vmin, VW[i]), vmax = MAX(vmax, VW[i]);
    if (dist[i] != 255) cov++;
    if (pW[i] != 255) {
      okall++, mx = MAX(mx, pW[i]);
      if (pW[i] <= 50) ok50++, sum += pW[i], sumopt += dW[i];
    }
  }
  printf("K=%d V-range=[%d,%d] won=%ld covered=%.4f%% verified=%.4f%% within50=%.4f%% fail=%ld max=%d mean=%.2f (opt %.2f)\n",
         K, vmin, vmax, won, 100.0 * cov / won, 100.0 * okall / won, 100.0 * ok50 / won, won - ok50, mx,
         sum / MAX(ok50, 1), sumopt / MAX(ok50, 1));
  if (!quiet) {
    static long lw[65536], lc[65536], lv[65536];
    for (int i = 0; i < N; i++) {
      if (dW[i] == 255) continue;
      UNPACK(i);
      if (!legalW(wk, wq, bk, br)) continue;
      int l = VW[i] + 32768;
      lw[l]++, lc[l] += dist[i] != 255, lv[l] += pW[i] != 255 && pW[i] <= 50;
    }
    printf("  level: won / uncovered / covered-but-failing\n");
    for (int l = 0; l < 65536; l++)
      if (lw[l]) printf("  V=%d: %ld / %ld / %ld\n", l - 32768, lw[l], lw[l] - lc[l], lc[l] - lv[l]);
    printf("  dist histogram (won positions):");
    for (int k = 1; k <= K; k++) printf(" %d:%ld", k, dh[k]);
    printf(" none:%ld\n", dh[255]);
    int shown = 0;
    for (int i = 0; i < N && shown < 6; i += 7919) {
      if (dW[i] == 255 || pW[i] != 255) continue;
      UNPACK(i);
      if (!legalW(wk, wq, bk, br)) continue;
      printf("  fail[%d]: WK %s WQ %s BK %s BR %s (dtc %d, V %d, dist %d)\n", i, sq(wk), sq(wq), sq(bk), sq(br), dW[i],
             VW[i], dist[i]);
      shown++;
    }
  }
  if (trace >= 0) {  // policy vs defender that prefers unresolved, then longest
    int i = trace;
    for (int ply = 0; ply < 80; ply++) {
      UNPACK(i);
      printf("%2d. WK %s WQ %s BK %s BR %s  V=%d dist=%d dtc=%d pol=%d\n", ply + 1, sq(wk), sq(wq), sq(bk), sq(br),
             VW[i], dist[i], dW[i], pW[i]);
      int b = pmove[i];
      if (b < 0) { printf("   -> %s\n", b == S_CAPR ? "captures rook" : "no plan"); break; }
      int bo[32], bn = gen_black(b >> 18, (b >> 12) & 63, (b >> 6) & 63, b & 63, bo), best = -1, bs = -1;
      if (!bn) { printf("   -> mate\n"); break; }
      for (int j = 0; j < bn; j++) {
        if (bo[j] == S_LOSS) { best = -2; break; }
        int sc = (pW[bo[j]] == 255 ? 1000 : pW[bo[j]]) * 256 + dW[bo[j]];
        if (sc > bs) bs = sc, best = bo[j];
      }
      if (best == -2) { printf("   -> black wins queen\n"); break; }
      i = best;
    }
  }
  return 0;
}
