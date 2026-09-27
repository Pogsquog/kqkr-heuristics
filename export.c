// Writes viewer/tb.bin: canonical White-to-move DTM and DTC tables (NC bytes each, see sym.h)
// followed by the full KQK tables k3W and k3B (N3 bytes each). The viewer derives Black-to-move
// values with one ply of search.
#include "kq.h"

uint8_t k3W[N3], k3B[N3];
#include "sym.h"

static uint8_t buf[N], out[NC];

static void canon_table(const char *fn, FILE *o) {
  FILE *f = fopen(fn, "rb");
  if (!f || fread(buf, 1, N, f) != N) exit(1);
  fclose(f);
  memset(out, 255, NC);
  for (int c = 0; c < NC; c++) {
    CUNPACK(c);
    if (canon(wk, wq, bk, br) == c && legalW(wk, wq, bk, br)) out[c] = buf[IDX(wk, wq, bk, br)];
  }
  fwrite(out, 1, NC, o);
}

int main(void) {
  init_geom();
  init_sym();
  FILE *f = fopen("kqk.bin", "rb");
  if (!f || fread(k3W, 1, N3, f) != N3 || fread(k3B, 1, N3, f) != N3) return 1;
  fclose(f);
  FILE *o = fopen("viewer/tb.bin", "wb");
  canon_table("dtm.bin", o);
  canon_table("dtc.bin", o);
  fwrite(k3W, 1, N3, o), fwrite(k3B, 1, N3, o), fclose(o);
  return 0;
}
