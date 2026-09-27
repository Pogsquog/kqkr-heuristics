// Strategy analysis over DTC-optimal play: which kinds of White move are optimal, and does optimal
// play monotonically restrict the black king's region (kreg) against every defence?
// usage: ./strat
#include "kq.h"

uint8_t k3W[N3], k3B[N3], dW[N], dB[N];
static int TBQ = 1;
#include "feat.h"

enum { C_TACT, C_KING, C_QCHK, C_QROOK, C_QHOLD, NCAT };
static const char *CN[] = {"tactic", "king", "Qcheck", "Qattack-rook", "Qhold"};

int main(void) {
  init_geom();
  init_lut();
  FILE *f = fopen("kqk.bin", "rb");
  if (!f || fread(k3W, 1, N3, f) != N3 || fread(k3B, 1, N3, f) != N3) return 1;
  fclose(f);
  f = fopen("dtc.bin", "rb");
  if (!f || fread(dW, 1, N, f) != N || fread(dB, 1, N, f) != N) return 1;
  fclose(f);
  build_graph();
  // region measures at W positions: 0 kreg (rook blocks queen lines), 1 box ignoring the rook, 2 king mobility
  static const char *MN[] = {"kreg", "kreg-norook", "bkmob"};
  int8_t *kr[3];
  for (int v = 0; v < 3; v++) kr[v] = malloc(NC);
#pragma omp parallel for schedule(static)
  for (int c = 0; c < NC; c++) {
    CUNPACK(c);
    if (!legW[c]) continue;
    kr[0][c] = kregion(wk, wq, bk, br);
    uint64_t fr = 0;
    for (int s = 0; s < 64; s++)
      if (s != wq && !adj(s, wk) && !qatt(wq, s, BIT(wk))) fr |= BIT(s);
    uint64_t reg = BIT(bk), prev = 0;
    while (reg != prev) { prev = reg; for (uint64_t m = reg; m; m &= m - 1) reg |= KATT[__builtin_ctzll(m)] & fr; }
    kr[1][c] = __builtin_popcountll(reg) - 1;
    kr[2][c] = bkmob(wk, wq, bk, br);
  }
  // for each measure: best worst-case change over (a) optimal moves, (b) any winning move
  long mono[4][3][2][3] = {{{{0}}}};
  // per DTC band: positions, positions whose optimal set contains each category, "only" each category,
  // and the best worst-case kreg change over optimal moves (<0, 0, >0)
  enum { NB = 4 };
  static const char *BN[] = {"dtc 1-2", "dtc 3-9", "dtc 10-19", "dtc 20-31"};
  long tot[NB] = {0}, has[NB][NCAT] = {{0}}, only[NB][NCAT] = {{0}}, dk[NB][3] = {{0}}, nopt[NB] = {0};
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : mono[:4][:3][:2][:3], tot[:NB], has[:NB][:NCAT], only[:NB][:NCAT], dk[:NB][:3], nopt[:NB])
  for (int c = 0; c < NC; c++) {
    if (!legW[c]) continue;
    int i = cfull(c), d = dW[i];
    if (d == 255) continue;
    CUNPACK(c);
    int b = d <= 2 ? 0 : d <= 9 ? 1 : d <= 19 ? 2 : 3, m = multW[c];
    int out[40], cap[40], k = gen_white(wk, wq, bk, br, out, cap), mask = 0, bestd = 99, no = 0;
    int bd[3][2] = {{99, 99}, {99, 99}, {99, 99}};
    for (int j = 0; j < k; j++) {
      int ch = chW[offW[c] + j];
      int v = ch == G_CONV ? 0 : ch == G_BAD ? 255 : dB[cfull(ch)];
      if (v == 255) continue;
      for (int ms = 0; ms < 3; ms++) {
        int worst = -99;
        if (ch >= 0 && !(legB[ch] & L_MATE))
          for (uint32_t t = offB[ch]; t < offB[ch + 1]; t++)
            if (chB[t] >= 0) worst = MAX(worst, kr[ms][chB[t]] - kr[ms][c]);
        if (v == d - 1) bd[ms][0] = MIN(bd[ms][0], worst);
        bd[ms][1] = MIN(bd[ms][1], worst);
      }
      if (v != d - 1) continue;
      no++;
      int cat;
      if (ch == G_CONV || (ch >= 0 && (legB[ch] & L_MATE))) cat = C_TACT;
      else {
        UNPACK(out[j]);
        (void)bk;
        if (wk != ((c >> 12) & 63)) cat = C_KING;
        else if (bk_in_check(wk, wq, bk, br)) cat = C_QCHK;
        else if (qatt(wq, br, BIT(wk) | BIT(bk))) cat = C_QROOK;
        else cat = C_QHOLD;
      }
      mask |= 1 << cat;
      if (ch < 0 || cat == C_TACT) { bestd = -99; continue; }
      int worst = -99;  // worst case over black replies of kreg change
      for (uint32_t t = offB[ch]; t < offB[ch + 1]; t++) {
        int w = chB[t];
        if (w >= 0) worst = MAX(worst, kr[0][w] - kr[0][c]);
      }
      bestd = MIN(bestd, worst);
    }
    tot[b] += m, nopt[b] += (long)m * no;
    for (int t = 0; t < NCAT; t++) {
      if (mask >> t & 1) has[b][t] += m;
      if (mask == 1 << t) only[b][t] += m;
    }
    dk[b][bestd < 0 ? 0 : bestd == 0 ? 1 : 2] += m;
    for (int ms = 0; ms < 3; ms++)
      for (int o = 0; o < 2; o++) mono[b][ms][o][bd[ms][o] < 0 ? 0 : bd[ms][o] == 0 ? 1 : 2] += m;
  }
  for (int b = 0; b < NB; b++) {
    printf("%-10s %9ld positions, %.2f optimal moves each\n", BN[b], tot[b], (double)nopt[b] / tot[b]);
    printf("   optimal set contains:");
    for (int t = 0; t < NCAT; t++) printf("  %s %.1f%%", CN[t], 100.0 * has[b][t] / tot[b]);
    printf("\n   only this kind:      ");
    for (int t = 0; t < NCAT; t++) printf("  %s %.1f%%", CN[t], 100.0 * only[b][t] / tot[b]);
    printf("\n   best optimal move vs worst reply: kreg shrinks %.1f%%, same %.1f%%, must grow %.1f%%\n",
           100.0 * dk[b][0] / tot[b], 100.0 * dk[b][1] / tot[b], 100.0 * dk[b][2] / tot[b]);
    for (int ms = 0; ms < 3; ms++)
      for (int o = 0; o < 2; o++)
        printf("   %-12s %-8s moves: shrinks %5.1f%%  same %5.1f%%  must grow %5.1f%%\n", MN[ms], o ? "winning" : "optimal",
               100.0 * mono[b][ms][o][0] / tot[b], 100.0 * mono[b][ms][o][1] / tot[b], 100.0 * mono[b][ms][o][2] / tot[b]);
  }
  return 0;
}
