CFLAGS = -O3 -march=native -fopenmp -Wall
PROGS = tb pol adv abs line

all: $(PROGS)
%: %.c kq.h feat.h sym.h
	$(CC) $(CFLAGS) -o $@ $< -lm
clean:
	rm -f $(PROGS)
