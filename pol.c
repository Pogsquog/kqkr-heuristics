// Policy harness: White plays argmin of a depth-d minimax over a linear eval of features;
// then an exact fixpoint over all positions checks the policy wins against EVERY defence.
// Runs on the 8-fold symmetry-reduced successor graph (sym.h); results are reported for the full board.
// usage: ./pol [-r] [-v] [-f] depth feat=weight ...
//   -r: adversarial tie-breaking, -v: examples (-v -v: all failures),
//   -f / -f2: forbid White moves that box the black king in while the rook can check (desp / desp2 rule)
//   -f3: forbid moves into a desperado "ladder" that White cannot escape (see ladder())
#include "kq.h"
#include <math.h>

uint8_t k3W[N3], k3B[N3], dW[N], dB[N];
int32_t *TW, *TB;  // indexed by canonical compact position (sym.h)
static int TBQ = 1;
#define INF 1000000000
#define MATE (-100000000)
#define CAPW (-50000000)
#define TERM (-10000000)

#include "feat.h"

static inline int adjterm(int v) { return v < TERM ? v + 1 : v; }

// value of a white move child (compact B index or G_CONV / G_BAD)
static int forbid_feat = 24;
static uint8_t *forbid;  // B positions White may not move into (-f), or NULL
static inline int wchild(int ch, const int32_t *tb) {
  return ch >= 0 ? (forbid && forbid[ch] ? INF : tb[ch]) : ch == G_CONV ? CAPW : INF;
}

static void sweepB(const int32_t *tw, int32_t *tb) {  // B value = max over black replies
#pragma omp parallel for schedule(dynamic, 4096)
  for (int c = 0; c < NC; c++) {
    int f = legB[c], v;
    if (!f) continue;
    if (f & L_MATE) v = MATE;
    else if (f & L_STALE) v = INF;
    else {
      v = -INF;
      for (uint32_t j = offB[c]; j < offB[c + 1]; j++) v = MAX(v, chB[j] < 0 ? INF : tw[chB[j]]);
    }
    tb[c] = adjterm(v);
  }
}
static void sweepW(const int32_t *tb, int32_t *tw) {  // W value = min over white moves
#pragma omp parallel for schedule(dynamic, 4096)
  for (int c = 0; c < NC; c++) {
    if (!legW[c]) continue;
    int v = INF;
    for (uint32_t j = offW[c]; j < offW[c + 1]; j++) v = MIN(v, wchild(chW[j], tb));
    tw[c] = adjterm(v);
  }
}
static void leaf(int32_t *t, int stm) {
#pragma omp parallel for schedule(dynamic, 4096)
  for (int c = 0; c < NC; c++) {
    if (!(stm ? legB[c] : legW[c])) continue;
    CUNPACK(c);
    t[c] = eval(wk, wq, bk, br, stm);
  }
}

static const char *sq(int s) {
  static char b[8][3];
  static int k;
  char *p = b[k++ & 7];
  p[0] = 'a' + F(s), p[1] = '1' + R(s), p[2] = 0;
  return p;
}

// Desperado ladder: once the black king has no moves, Black's only useful moves are rook checks
// and White's only moves are check evasions. Solve that narrow game exactly (like reading a go
// ladder): White escapes if Black runs out of checks, the black king is unboxed without the queen
// hanging, or White can take the rook into a won KQK. Positions from which Black can keep checking forever are marked in
// forbid[]. No KQKR tablebase is used, only the move graph and KQK.
static void ladder(void) {
  uint8_t *boxed = calloc(NC, 1), *incheck = calloc(NC, 1), *lb = malloc(NC), *lw = malloc(NC);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int c = 0; c < NC; c++) {
    CUNPACK(c);
    if (legB[c] && !(legB[c] & L_MATE)) boxed[c] = bkmob(wk, wq, bk, br) == 0;
    if (legW[c]) incheck[c] = wk_in_check(wk, wq, bk, br);
  }
  memset(lb, 255, NC), memset(lw, 255, NC);
  for (int c = 0; c < NC; c++) {
    if (legB[c] && !boxed[c]) {  // ladder over: normal play resumes, unless the rook can take the queen
      int hang = 0;
      for (uint32_t j = offB[c]; j < offB[c + 1]; j++) hang |= chB[j] == G_BAD;
      if (!hang) lb[c] = 0;
    }
    if (legW[c] && !incheck[c]) lw[c] = 0;  // Black gave up checking: White has a free move
  }
  int mx = 0;
  for (int n = 1; n < 250; n++) {
    long ch = 0;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : ch)
    for (int c = 0; c < NC; c++) {
      if (!incheck[c] || lw[c] != 255) continue;
      for (uint32_t j = offW[c]; j < offW[c + 1]; j++) {
        int m = chW[j];
        if (m == G_CONV || (m >= 0 && lb[m] <= n - 1)) { lw[c] = n, ch++; break; }
      }
    }
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : ch)
    for (int c = 0; c < NC; c++) {
      if (!boxed[c] || lb[c] != 255 || (legB[c] & L_STALE)) continue;
      int all = 1;
      for (uint32_t j = offB[c]; j < offB[c + 1] && all; j++)
        if (chB[j] < 0 || lw[chB[j]] > n) all = 0;
      if (all) lb[c] = n, ch++;
    }
    if (!ch) break;
    mx = n;
  }
  long cnt[2][2] = {{0}};  // [ladder says draw][tablebase says draw], boxed B positions
  for (int c = 0; c < NC; c++) {
    if (!boxed[c]) continue;
    forbid[c] = lb[c] == 255;
    cnt[forbid[c]][dB[cfull(c)] == 255]++;
  }
  fprintf(stderr, "ladder: boxed B positions: ladder-escape & won %ld, ladder-escape & drawn %ld, "
          "ladder-draw & won %ld, ladder-draw & drawn %ld; longest escape %d plies\n",
          cnt[0][0], cnt[0][1], cnt[1][0], cnt[1][1], mx);
  if (getenv("LADDBG")) {  // print the ladder line from a B position given as a full-board index
    int c = canon_full(atoi(getenv("LADDBG")));
    for (int ply = 0; ply < 20; ply++) {
      CUNPACK(c);
      fprintf(stderr, "  %c WK %s WQ %s BK %s BR %s  lb=%d lw=%d dB=%d dW=%d\n", ply & 1 ? 'W' : 'B', sq(wk), sq(wq), sq(bk),
              sq(br), lb[c], lw[c], dB[cfull(c)], dW[cfull(c)]);
      int nx = G_BAD;
      if (!(ply & 1)) {
        if (!boxed[c]) break;
        int bs = -1;
        for (uint32_t j = offB[c]; j < offB[c + 1]; j++)
          if (chB[j] >= 0 && lw[chB[j]] > bs) bs = lw[chB[j]], nx = chB[j];
      } else {
        if (!incheck[c]) break;
        for (uint32_t j = offW[c]; j < offW[c + 1]; j++)
          if (chW[j] == G_CONV || (chW[j] >= 0 && lb[chW[j]] == lw[c] - 1)) { nx = chW[j]; break; }
      }
      if (nx < 0) { fprintf(stderr, "  -> %s\n", nx == G_CONV ? "rook captured into won KQK" : "end"); break; }
      c = nx;
    }
  }
  free(boxed), free(incheck), free(lb), free(lw);
}


int main(int argc, char **argv) {
  int robust = 0, verbose = 0, a = 1, trace = -1;
  while (a < argc && argv[a][0] == '-') {
    if (argv[a][1] == 'r') robust = 1;
    if (argv[a][1] == 'f') forbid = calloc(NC, 1), forbid_feat = argv[a][2] == '3' ? -1 : argv[a][2] == '2' ? 25 : 24;
    if (argv[a][1] == 'v') verbose++;
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
  build_graph();
  uint8_t *dWc = malloc(NC), *dBc = malloc(NC);
#pragma omp parallel for schedule(static)
  for (int c = 0; c < NC; c++) {
    int i = cfull(c);
    dWc[c] = legW[c] ? dW[i] : 255, dBc[c] = legB[c] ? dB[i] : 255;
  }
  if (forbid && forbid_feat < 0) ladder();
  else if (forbid) {
#pragma omp parallel for schedule(dynamic, 4096)
    for (int c = 0; c < NC; c++) {
      if (!legB[c]) continue;
      CUNPACK(c);
      forbid[c] = feat(forbid_feat, wk, wq, bk, br, 1);
    }
  }
  TW = malloc(4L * NC), TB = malloc(4L * NC);

  // depth-d search tables; root children need TB at depth d-1
  if (depth & 1) leaf(TB, 1);
  else leaf(TW, 0), sweepB(TW, TB);
  for (int d = (depth & 1) ? 1 : 2; d < depth; d += 2) sweepW(TB, TW), sweepB(TW, TB);
  // root best value per W position (stored in TW)
  sweepW(TB, TW);

  // fixpoint: pW = white moves to conversion under the policy vs all defences.
  // Only still-unresolved positions are kept on the active lists.
  uint8_t *pW = malloc(NC), *pB = malloc(NC);
  memset(pW, 255, NC), memset(pB, 255, NC);
  int32_t *actW = malloc(4L * NC), *actB = malloc(4L * NC);
  int nW = 0, nB = 0;
  for (int c = 0; c < NC; c++) {
    if (legB[c] & L_MATE) pB[c] = 0;
    else if (dBc[c] != 255) actB[nB++] = c;
    if (dWc[c] != 255) actW[nW++] = c;
  }
  for (int n = 1; n <= 100; n++) {
    long ch = 0;
#pragma omp parallel for schedule(dynamic, 1024) reduction(+ : ch)
    for (int x = 0; x < nW; x++) {
      int c = actW[x], best = TW[c], okc = 0, badc = 0;
      for (uint32_t j = offW[c]; j < offW[c + 1]; j++) {
        int m = chW[j];
        if (adjterm(wchild(m, TB)) != best) continue;
        int good = m == G_CONV || (m >= 0 && pB[m] <= n - 1);
        if (good) okc++; else badc++;
        if (!robust) break;  // deterministic: first best move in generation order (canonical frame)
      }
      if (okc && !badc) pW[c] = n, ch++;
    }
#pragma omp parallel for schedule(dynamic, 1024) reduction(+ : ch)
    for (int x = 0; x < nB; x++) {
      int c = actB[x], all = offB[c + 1] > offB[c];
      for (uint32_t j = offB[c]; j < offB[c + 1] && all; j++)
        if (chB[j] < 0 || pW[chB[j]] > n) all = 0;
      if (all) pB[c] = n, ch++;
    }
    if (!ch) break;
    int k = 0;
    for (int x = 0; x < nW; x++) if (pW[actW[x]] == 255) actW[k++] = actW[x];
    nW = k, k = 0;
    for (int x = 0; x < nB; x++) if (pB[actB[x]] == 255) actB[k++] = actB[x];
    nB = k;
  }
  // statistics over the full board: each canonical position counts for its whole symmetry orbit
  long won = 0, ok50 = 0, okany = 0;
  double sum = 0, sumopt = 0;
  int mx = 0;
  for (int c = 0; c < NC; c++) {
    if (dWc[c] == 255) continue;
    int m = multW[c];
    won += m;
    if (pW[c] != 255) {
      okany += m, mx = MAX(mx, pW[c]);
      if (pW[c] <= 50) ok50 += m, sum += m * pW[c], sumopt += m * dWc[c];
    }
  }
  printf("depth=%d%s%s won=%ld ok50=%ld (%.4f%%) fail=%ld okany=%ld max=%d mean=%.2f (opt %.2f)\n", depth,
         robust ? " robust" : "", forbid ? " forbid" : "", won, ok50, 100.0 * ok50 / won, won - ok50, okany, mx, sum / MAX(ok50, 1),
         sumopt / MAX(ok50, 1));
  if (trace >= 0) {  // play policy vs a defender that avoids policy-won positions, max DTC (canonical frame)
    int c = canon_full(trace);
    for (int ply = 0; ply < 60; ply++) {
      CUNPACK(c);
      printf("%2d. WK %s WQ %s BK %s BR %s  dtc=%d pol=%d val=%d\n", ply + 1, sq(wk), sq(wq), sq(bk), sq(br), dWc[c],
             pW[c], TW[c]);
      int b = G_BAD;
      for (uint32_t j = offW[c]; j < offW[c + 1]; j++)
        if (adjterm(wchild(chW[j], TB)) == TW[c]) { b = chW[j]; break; }
      if (b < 0) { printf("   -> %s\n", b == G_CONV ? "captures rook" : "??"); break; }
      if (offB[b + 1] == offB[b]) { printf("   -> mate/stalemate\n"); break; }
      int best = -1, bs = -1;
      for (uint32_t j = offB[b]; j < offB[b + 1]; j++) {
        int w = chB[j];
        if (w < 0) { best = -2; break; }
        int sc = (pW[w] == 255 ? 1000 : pW[w]) * 256 + dWc[w];
        if (sc > bs) bs = sc, best = w;
      }
      if (best == -2) { printf("   -> black wins queen\n"); break; }
      c = best;
    }
  }
  if (verbose) {
    int shown = 0;
    for (int c = 0; c < NC && shown < 12; c += verbose > 1 ? 1 : 7919) {
      if (dWc[c] == 255 || pW[c] != 255) continue;
      CUNPACK(c);
      printf("  fail[%d]: WK %s WQ %s BK %s BR %s (dtc %d)\n", cfull(c), sq(wk), sq(wq), sq(bk), sq(br), dWc[c]);
      shown++;
    }
  }
  return 0;
}
