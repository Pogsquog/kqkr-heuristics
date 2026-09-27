CFLAGS = -O3 -march=native -fopenmp -Wall
PROGS = tb pol adv abs line export strat

all: $(PROGS)
%: %.c kq.h feat.h sym.h
	$(CC) $(CFLAGS) -o $@ $< -lm
# viewer/tb.js: canonical DTM/DTC tables + KQK, gzipped and base64-embedded (needs dtm.bin, dtc.bin, kqk.bin)
viewer: export
	./export && gzip -9 -f viewer/tb.bin
	{ printf 'window.KQKR_TB="'; base64 -w0 viewer/tb.bin.gz; printf '";\n'; } > viewer/tb.js
	rm -f viewer/tb.bin.gz
.PHONY: viewer
clean:
	rm -f $(PROGS)
