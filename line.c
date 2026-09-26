// Print a DTC-optimal line (White fastest, Black slowest) from a given or the longest position.
#include "kq.h"
uint8_t k3W[N3], k3B[N3], dW[N], dB[N];
static void show(int wk, int wq, int bk, int br) {
  for (int r = 7; r >= 0; r--) {
    printf("   ");
    for (int f = 0; f < 8; f++) {
      int s = r * 8 + f;
      putchar(s == wk ? 'K' : s == wq ? 'Q' : s == bk ? 'k' : s == br ? 'r' : '.');
    }
    putchar('\n');
  }
}
static const char *sq(int s) { static char b[8][3]; static int k; char *p = b[k++ & 7]; p[0] = 'a' + F(s), p[1] = '1' + R(s), p[2] = 0; return p; }
int main(int argc, char **argv) {
  init_geom();
  FILE *f = fopen("kqk.bin", "rb"); if (fread(k3W, 1, N3, f)) {} if (fread(k3B, 1, N3, f)) {} fclose(f);
  f = fopen("dtc.bin", "rb"); if (fread(dW, 1, N, f)) {} if (fread(dB, 1, N, f)) {} fclose(f);
  int i = -1;
  if (argc > 1) i = atoi(argv[1]);
  else for (int j = 0; j < N; j++) if (dW[j] != 255) { UNPACK(j); if (legalW(wk, wq, bk, br) && dW[j] == 31) { i = j; break; } }
  int boards = argc > 2 ? atoi(argv[2]) : 0;
  for (int mv = 1; ; mv++) {
    UNPACK(i);
    printf("%2d. WK %s WQ %s BK %s BR %s dtc=%d\n", mv, sq(wk), sq(wq), sq(bk), sq(br), dW[i]);
    if (boards) show(wk, wq, bk, br);
    int out[40], cap[40], k = gen_white(wk, wq, bk, br, out, cap), b = -9;
    for (int j = 0; j < k; j++) {
      if (out[j] == S_CAPR && k3B[cap[j]] != 255 && dW[i] == 1) { b = S_CAPR; break; }
      if (out[j] >= 0 && dB[out[j]] == dW[i] - 1) { b = out[j]; break; }
    }
    if (b == S_CAPR) { printf("    Qx/Kx rook\n"); break; }
    if (dB[b] == 0) { printf("    mate\n"); break; }
    int bo[32], bn = gen_black(b >> 18, (b >> 12) & 63, (b >> 6) & 63, b & 63, bo), best = -1, bs = -1;
    for (int j = 0; j < bn; j++) if (bo[j] >= 0 && dW[bo[j]] != 255 && dW[bo[j]] > bs) bs = dW[bo[j]], best = bo[j];
    i = best;
  }
}
