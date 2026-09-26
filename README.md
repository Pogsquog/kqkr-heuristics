# Compact winning heuristics for KQ vs KR

Goal: the smallest function of the four piece squares that, when followed, wins every won
KQKR position (White to move) within 50 moves against *any* defence.

Every claim below is **exhaustively verified** over all 8,952,608 legal White-to-move positions
(8,863,768 of them are theoretical wins). The verifier fixes White's moves to the policy and
lets Black choose freely, then computes an exact fixpoint. A position counts as won only if
every Black defence ends in mate or a safe rook capture within 50 moves. Cycles count as
failures. The policies are memoryless: no repetition history.

## Files

| file | purpose |
|---|---|
| `kq.h` | geometry, legality, move generation |
| `sym.h` | 8-fold symmetry reduction and precomputed successor graph over canonical positions |
| `tb.c` | KQK DTM and KQKR DTC/DTM tablebases (`./tb`, ~20 s; gives max DTC 31, max DTM 35) |
| `feat.h` | evaluation features |
| `pol.c` | `./pol [-r] [-f3] [-v [-v]] depth feat=w ...`: depth-d minimax on a linear eval, then exact verification (`-r` = every tied best move must also win; `-f3` = desperado ladder rule; `-v -v` = list all failures); runs on the symmetry-reduced graph |
| `adv.c` | "advice" policy: force a strict drop of a potential V within K moves |
| `abs.c` | optimal potential over an abstraction (e.g. two-king squares) via greedy level peeling |
| `tune.py` | coordinate descent on eval weights; objective = exact failure count |
| `fit.py`, `line.c`, `an*.py` | LUT regression, optimal-line printer, analysis |

Build with `make`. `pol` works on canonical positions: the black king is mapped into a1–d1–d4,
which leaves 2.6M slots instead of 16.7M. It precomputes every successor once, so each search ply
becomes a gather over the graph (≈8 ms). A 44-ply search plus exact verification takes ≈1 s,
down from 17 s over the full board. Statistics are still reported over the full board: each
canonical position is weighted by the size of its symmetry orbit. Robust (`-r`) results are
identical to the full-board version. Without `-r`, ties are broken by move order in the canonical
frame. That makes the policy symmetric, but it shifts non-robust counts slightly relative to the
old full-board tie-break. `adv` and `abs` still sweep the full board.

## Features (White minimises)

`cd` = black king's distance from the centre (0–6), `ed` = its distance to the edge (0–3),
`kk` = king–king Chebyshev distance, `qk`/`qkm` = queen to black king (Chebyshev/Manhattan), `rk`/`rkm` = rook to its own
king (Chebyshev/Manhattan), `mob` = legal black king moves.

Defender-option features:
- `rmob`: legal rook moves.
- `rsafe`: rook moves to squares where White can't take the rook for free.
- `rchk`: safe rook checks.
- `rqa`: safe rook moves that attack the queen.
- `kreg`: squares the black king can reach through unattacked squares.
- `bkst`: the black king has no moves.
- `desp`: the black king is boxed in and the rook is checking or can check.

## Results: search depth needed for 100% (all wins within 50 moves)

| eval | terms | plies for 100% | worst game |
|---|---|---|---|
| none (search only for mate / safe rook capture) | 0 | 62 (i.e. tablebase depth) | 37 |
| mop-up `-10cd + 4kk - 3rk` | 3 | 56 | 38 |
| `-7cd + 6kk + 4qk - 3rk + mob + 4ed` | 6 | 48 | 42 (robust) |
| `-7cd + 8kk + 4qk - 3rk - 3rkm + mob + ed` | 7 | **44** | 42 (robust) |
| `-7cd + 8kk + 4qk - 4qkm - 4rk - 2rkm + mob + ed` | 8 | **40** (canonical tie-break; not robust) | 38 |
| 14-term defender eval (below) + desperado ladder rule | 14 + rule | **36** (canonical tie-break; not robust) | 40 |
| additive LUT fit to DTC | 2,595 | >44 (16 failures at 44) | – |

Close misses (non-robust, canonical tie-break): the 6-term eval at 44 plies has 160 failures.
The 7-term eval has 376 / 168 / 56 failures at 38 / 40 / 42 plies. With `rk=-4` it has 216
failures at 40 plies; the old full-board tie-break gave 146.

A rerun of the depth-40 tuner (`tune40.log`, ~1 s per evaluation) added `qkm` and reached 0
failures at 40 plies with the 8-term eval above. `4qk - 4qkm` = -4·min(|Δfile|, |Δrank|) between
the queen and the black king, so it rewards the queen sharing a rank or file with the king. At
38 plies it has 80 failures. It is not robust to tie-breaking: with `-r` it has 912 failures at
40 plies and 1,632 at 42.

### Defender-option features (`tune32d.log`, `tune36.log`)

These features help most at shallower depth. On the 8-term eval (scaled ×4), `kreg=-3` alone
cuts failures at 30 plies from 97,904 to 11,036. The tuner at 32 plies brought failures there
from 42,176 down to 1,136. Continuing at 36 plies gave this 14-term eval:

`-26cd + 29kk + 16qk - 8qkm - 16rk - 8rkm + 4mob + 3ed + co + box + 3kqk - wkc + rmob - 3kreg`

| plies | 34 | 35 | 36 | 37 | 38 | 40 |
|---|---|---|---|---|---|---|
| failures | 1,232 | 222,548 | **8** | 1,160 | 88 | 0 |

Odd depths are much worse. The 8 failures at 36 plies are one symmetry orbit, `WK h8 WQ h7 BK
a1 BR a4` (DTC 24). There White plays Qc2, which boxes the black king in with no moves. The
rook then becomes a desperado and checks forever, because taking it is stalemate. No depth
catches this, since the draw is a repetition, not a terminal. Penalising "king boxed in" (`bkst`)
or "boxed in and the rook can check" (`desp`) backfires. `desp` fires in 56k won positions for
every 3k drawn ones, and boxing the king in is also how most mates happen. Any weight large
enough to matter adds hundreds of thousands of failures. Coordinate descent with both in the
pool stays at 8.

Banning those moves outright (`pol -f`) is also a disaster: 266,604 failures at 36 plies. A
stricter ban (`-f2`) only flags a rook check if White can't take the rook into a won KQK. It
fixes the trap but still fails 1,968 positions at 36 plies. It fires on 18k won and 22k drawn
Black-to-move positions. Whether the checks eventually run out depends on geometry several moves
ahead, which no test of the current position can see.

### Desperado ladder (`pol -f3`, `tune36f3.log`, `tune34f3.log`)

This works like reading a ladder in go: a narrow, forcing search that can go deep cheaply. Once
the black king has no moves, Black's only useful moves are rook checks, and White's only moves are
check evasions. `ladder()` in `pol.c` solves that restricted game exactly, as a fixpoint over the
move graph. It uses no KQKR tablebase, only KQK. White escapes if one of these happens:

* Black runs out of checks.
* White takes the rook into a won KQK.
* The king gets unboxed without the queen hanging to the rook.

Positions from which Black can check forever are banned as White move targets.

The queen-hanging condition matters. Without it, blocking with the queen (Qc2 … Rh4+ Qh7 Rxh7)
counts as an escape, and the rule misses the original trap. Boxed-in Black-to-move positions,
canonical counts:

| | tablebase won | tablebase drawn |
|---|---|---|
| ladder: White escapes | 3,477 | 424 |
| ladder: perpetual | **0** | 3,056 |

The rule never bans a winning move, and it catches 88% of the desperado draws. The longest
escape is 8 plies. With the ban on, the 14-term eval above reaches 0 failures at 38 plies.
Re-tuning at 36 plies with `-f3` gives:

`-26cd + 21kk + 16qk - 8qkm - 15rk - 8rkm + 4mob + 3ed + co + box + 3kqk - wkc + rmob - 3kreg` + ladder rule

| plies | 30 | 32 | 34 | 36 | 38 | 40 |
|---|---|---|---|---|---|---|
| failures | 120,084 | 78,208 | 1,184 | **0** | 16 | 0 |

Without the ladder rule the same weights have 224 failures at 36 plies. With `-r` there are 576,
so the result depends on tie-breaking. A tuner pass at 34 plies stalled at 232 failures.

## What does *not* work (and why)

* **Shallow search + any static eval fails.** Mop-up evals, the 2,595-parameter LUT, and the
  "rook-less KQK distance" all top out at 57–71% for depth ≤ 16. Wins are all-or-nothing: DTC is
  bimodal (≈5M positions win by tactics in ≤ 3 moves; the rest form a hump at 10–31). Black can
  steer every hump position into the policy's worst region, where the rook shadows the queen or
  harasses the king with checks and the policy cycles.
* **Plateaus are fatal for memoryless policies.** Even the true DTC, quantised to steps of 2,
  fails 35% at depth 2, because ties let the defender force repetitions.
* **Two-piece potentials need long lookahead.** A potential depending only on (bk, wk) needs
  ≥ 18 moves of lookahead (lower bound: every king pair has a placement of Q and R with
  DTC ≥ 18). (bk, br) needs ≥ 20. (bk, wq) cannot close even a first level at K = 4, because
  any black king move leaves the state. Progress in KQKR is intrinsically a four-piece property.
