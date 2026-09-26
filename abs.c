// Optimal potential over an abstraction: find level sets L0, L1, ... of abstract states so that from
// every won position in level t White can force, within K moves and without rising above level t,
// a conversion or a position at a lower level. Greedy maximal peeling is optimal (monotone).
// Then the resulting advice policy is verified exactly against all defences.
// usage: ./abs abstraction K        abstraction: bk | bkwk | bkwq | bkbr | bkwkq
#include "kq.h"

uint8_t k3W[N3], k3B[N3], dW[N], dB[N];
static int32_t *A;      // abstract id per W position (-1 illegal)
static int16_t *LV;     // level per abstract id (-1 = not assigned yet)
static uint8_t *inC;    // candidate flag per abstract id
static uint8_t *dist;
static int32_t *pmove;
static int M;

static int TSEL[64], TRI[64];
static inline int tfs(int s, int t) {
  int f = F(s), r = R(s), x;
  if (t & 1) f = 7 - f;
  if (t & 2) r = 7 - r;
  if (t & 4) x = f, f = r, r = x;
  return r * 8 + f;
}
static int mode;
static int abstract(int wk, int wq, int bk, int br) {
  int t = TSEL[bk], b = TRI[tfs(bk, t)];
  wk = tfs(wk, t), wq = tfs(wq, t), br = tfs(br, t);
  switch (mode) {
    case 0: return b;
    case 1: return b * 64 + wk;
    case 2: return b * 64 + wq;
    case 3: return b * 64 + br;
    case 4: return (b * 64 + wk) * 3 + (qatt(wq, tfs(bk, t), 0) ? 0 : cheb(wq, tfs(bk, t)) <= 2 ? 1 : 2);
  }
  return 0;
}

// level of W position q relative to the level being built (t): -1 lower/goal, 0 same (candidate), 1 higher
static inline int rel_level(int q) {
  int a = A[q];
  if (LV[a] >= 0) return -1;
  return inC[a] ? 0 : 1;
}
static inline int bgood(int b, int lim) {
  UNPACK(b);
  int out[32], k = gen_black(wk, wq, bk, br, out);
  if (!k) return bk_in_check(wk, wq, bk, br);
  for (int j = 0; j < k; j++) {
    int q = out[j];
    if (q == S_LOSS) return 0;
    int r = rel_level(q);
    if (r < 0) continue;
    if (r > 0 || dist[q] > lim) return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  const char *names[] = {"bk", "bkwk", "bkwq", "bkbr", "bkwkq", 0};
  for (mode = 0; names[mode] && strcmp(names[mode], argv[1]); mode++);
  int K = atoi(argv[2]);
  init_geom();
  int n0 = 0;
  for (int s = 0; s < 64; s++) TRI[s] = (F(s) <= 3 && R(s) <= F(s)) ? n0++ : -1;
  for (int s = 0; s < 64; s++)
    for (int t = 7; t >= 0; t--) if (TRI[tfs(s, t)] >= 0) TSEL[s] = t;
  M = (int[]){10, 640, 640, 640, 1920}[mode];
  FILE *f = fopen("kqk.bin", "rb");
  if (!f || fread(k3W, 1, N3, f) != N3 || fread(k3B, 1, N3, f) != N3) return 1;
  fclose(f);
  f = fopen("dtc.bin", "rb");
  if (!f || fread(dW, 1, N, f) != N || fread(dB, 1, N, f) != N) return 1;
  fclose(f);
  A = malloc(4L * N), LV = malloc(2L * M), inC = malloc(M), dist = malloc(N), pmove = malloc(4L * N);
  for (int a = 0; a < M; a++) LV[a] = -1;
  long *wonc = calloc(M, sizeof(long));
#pragma omp parallel for schedule(dynamic, 4096)
  for (int i = 0; i < N; i++) {
    UNPACK(i);
    A[i] = legalW(wk, wq, bk, br) ? abstract(wk, wq, bk, br) : -1;
    pmove[i] = S_LOSS;
  }
  for (int i = 0; i < N; i++) if (A[i] >= 0 && dW[i] != 255) wonc[A[i]]++;
  int remaining = 0;
  for (int a = 0; a < M; a++) remaining += wonc[a] > 0;
  memset(dist, 255, N);
  for (int t = 0; remaining > 0; t++) {
    for (int a = 0; a < M; a++) inC[a] = LV[a] < 0 && wonc[a] > 0;
    int csize;
    for (int iter = 0;; iter++) {
      // attractor within K for positions whose state is a candidate
#pragma omp parallel for schedule(dynamic, 4096)
      for (int i = 0; i < N; i++) if (A[i] >= 0 && inC[A[i]]) dist[i] = 255;
      for (int n = 1; n <= K; n++) {
        long ch = 0;
#pragma omp parallel for schedule(dynamic, 1024) reduction(+ : ch)
        for (int i = 0; i < N; i++) {
          if (A[i] < 0 || !inC[A[i]] || dist[i] != 255) continue;
          UNPACK(i);
          int out[40], cap[40], k = gen_white(wk, wq, bk, br, out, cap), pick = -1;
          for (int j = 0; j < k && pick < 0; j++)
            if (out[j] == S_CAPR ? k3B[cap[j]] != 255 : bgood(out[j], -1)) pick = j;  // convert first
          for (int j = 0; j < k && pick < 0; j++)
            if (out[j] != S_CAPR && bgood(out[j], n - 1)) pick = j;
          if (pick >= 0) dist[i] = n, pmove[i] = out[pick], ch++;
        }
        if (!ch) break;
      }
      // drop candidate states that contain an uncovered won position
      uint8_t *bad = calloc(M, 1);
      for (int i = 0; i < N; i++)
        if (A[i] >= 0 && inC[A[i]] && dW[i] != 255 && dist[i] == 255) bad[A[i]] = 1;
      int removed = 0;
      csize = 0;
      for (int a = 0; a < M; a++) {
        if (inC[a] && bad[a]) inC[a] = 0, removed++;
        csize += inC[a];
      }
      free(bad);
      if (!removed || !csize) break;
    }
    if (!csize) { printf("K=%d: stuck at level %d with %d abstract states unassigned\n", K, t, remaining); break; }
    for (int a = 0; a < M; a++) if (inC[a]) LV[a] = t;
    remaining -= csize;
    printf("level %d: %d states\n", t, csize);
    fflush(stdout);
  }
  // exact verification of the resulting policy
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
      if (pW[i] != 255 || A[i] < 0 || LV[A[i]] < 0 || dist[i] == 255) continue;
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
  long won = 0, ok = 0, ok50 = 0;
  int mx = 0;
  double sum = 0;
  for (int i = 0; i < N; i++) {
    if (A[i] < 0 || dW[i] == 255) continue;
    won++;
    if (pW[i] != 255) ok++, mx = MAX(mx, pW[i]), ok50 += pW[i] <= 50, sum += pW[i];
  }
  printf("%s K=%d: verified %.4f%%, within 50: %.4f%%, max %d, mean %.2f\n", argv[1], K, 100.0 * ok / won,
         100.0 * ok50 / won, mx, sum / MAX(ok, 1));
  char fn[64];
  sprintf(fn, "lev_%s_K%d.bin", argv[1], K);
  f = fopen(fn, "wb");
  fwrite(LV, 2, M, f);
  fclose(f);
  return 0;
}
