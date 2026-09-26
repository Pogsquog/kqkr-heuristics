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
| `tb.c` | KQK DTM and KQKR DTC/DTM tablebases (`./tb`, ~20 s; gives max DTC 31, max DTM 35) |
| `feat.h` | evaluation features |
| `pol.c` | `./pol [-r] depth feat=w ...`: depth-d minimax on a linear eval, then exact verification (`-r` = every tied best move must also win) |
| `adv.c` | "advice" policy: force a strict drop of a potential V within K moves |
| `abs.c` | optimal potential over an abstraction (e.g. two-king squares) via greedy level peeling |
| `tune.py` | coordinate descent on eval weights; objective = exact failure count |
| `fit.py`, `line.c`, `an*.py` | LUT regression, optimal-line printer, analysis |

Depth-d search over all 16.7M positions costs only d table sweeps (≈0.5 s each), so even
60-ply searches can be verified over the whole space.

## Features (White minimises)

`cd` = black king's distance from the centre (0–6), `ed` = its distance to the edge (0–3),
`kk` = king–king Chebyshev distance, `qk` = queen to black king, `rk`/`rkm` = rook to its own
king (Chebyshev/Manhattan), `mob` = legal black king moves.

## Results: search depth needed for 100% (all wins within 50 moves)

| eval | terms | plies for 100% | worst game |
|---|---|---|---|
| none (search only for mate / safe rook capture) | 0 | 62 (i.e. tablebase depth) | 37 |
| mop-up `-10cd + 4kk - 3rk` | 3 | 56 | 38 |
| `-7cd + 6kk + 4qk - 3rk + mob + 4ed` | 6 | 48 | 42 (robust) |
| `-7cd + 8kk + 4qk - 3rk - 3rkm + mob + ed` | 7 | **44** | 42 (robust) |
| additive LUT fit to DTC | 2,595 | >44 (16 failures at 44) | – |

Close misses: 6-term at 44 plies: 140 failures. 7-term at 40 plies: 146 failures with `rk=-4` (99.998%); tuning was stopped there.

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
