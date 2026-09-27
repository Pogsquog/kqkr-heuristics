// KQ vs KR rules and tablebase lookup for the viewer. Mirrors kq.h / sym.h.
// Position: {wk, wq, bk, br, stm: 'w' | 'b'}; br = -1 once the rook is taken, wq = -1 once the queen is lost.
// Values are White moves still needed (mate for 'dtm', mate or safe rook capture for 'dtc'); 255 = not a win.
(function (G) {
  const NC = 10 << 18, N3 = 1 << 18;
  const F = s => s & 7, R = s => s >> 3;
  const cheb = (a, b) => Math.max(Math.abs(F(a) - F(b)), Math.abs(R(a) - R(b)));
  const adj = (a, b) => cheb(a, b) <= 1;

  function tfs(s, t) {
    let f = F(s), r = R(s), x;
    if (t & 1) f = 7 - f;
    if (t & 2) r = 7 - r;
    if (t & 4) x = f, f = r, r = x;
    return r * 8 + f;
  }
  const TRI = new Int8Array(64), TRISQ = [], TSEL = new Uint8Array(64);
  const TF = Array.from({length: 8}, () => new Uint8Array(64));
  {
    let n = 0;
    for (let s = 0; s < 64; s++) {
      TRI[s] = F(s) <= 3 && R(s) <= F(s) ? n++ : -1;
      if (TRI[s] >= 0) TRISQ[TRI[s]] = s;
    }
    for (let s = 0; s < 64; s++) {
      for (let t = 0; t < 8; t++) TF[t][s] = tfs(s, t);
      for (let t = 7; t >= 0; t--) if (TRI[tfs(s, t)] >= 0) TSEL[s] = t;
    }
  }
  function canon(wk, wq, bk, br) {
    const t = TSEL[bk], b = TF[t][bk], k = TF[t][wk], q = TF[t][wq], r = TF[t][br];
    let c = (TRI[b] << 18) | (k << 12) | (q << 6) | r;
    if (F(b) === R(b)) {
      const c2 = (TRI[b] << 18) | (TF[4][k] << 12) | (TF[4][q] << 6) | TF[4][r];
      if (c2 < c) c = c2;
    }
    return c;
  }
  function uncanon(c) {
    return {bk: TRISQ[c >> 18], wk: (c >> 12) & 63, wq: (c >> 6) & 63, br: c & 63};
  }

  // slider on a attacks s (straight lines, plus diagonals if diag) with the squares in occ blocking
  function slide(a, s, diag, occ) {
    if (a < 0 || s < 0 || a === s) return false;
    const df = F(s) - F(a), dr = R(s) - R(a);
    if (!(df === 0 || dr === 0 || (diag && Math.abs(df) === Math.abs(dr)))) return false;
    const sf = Math.sign(df), sr = Math.sign(dr);
    for (let f = F(a) + sf, r = R(a) + sr; f !== F(s) || r !== R(s); f += sf, r += sr)
      if (occ.includes(r * 8 + f)) return false;
    return true;
  }
  const qatt = (q, s, occ) => slide(q, s, true, occ);
  const ratt = (r, s, occ) => slide(r, s, false, occ);

  function legal(p) {
    const sq = [p.wk, p.wq, p.bk, p.br].filter(s => s >= 0);
    if (new Set(sq).size !== sq.length || adj(p.wk, p.bk)) return false;
    return p.stm === 'w' ? !qatt(p.wq, p.bk, [p.wk, p.br]) : !ratt(p.br, p.wk, [p.wq, p.bk]);
  }
  const inCheck = p => (p.stm === 'w' ? ratt(p.br, p.wk, [p.wq, p.bk]) : qatt(p.wq, p.bk, [p.wk, p.br]));
  const KD = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
  function* ray(s, dirs) {
    for (const [df, dr] of dirs) {
      const line = [];
      for (let f = F(s) + df, r = R(s) + dr; f >= 0 && f < 8 && r >= 0 && r < 8; f += df, r += dr) line.push(r * 8 + f);
      yield line;
    }
  }
  const kingSteps = s => [...ray(s, KD)].filter(l => l.length).map(l => l[0]);

  function genW(p) {
    if (p.wq < 0) return [];
    const out = [], mk = (piece, from, to, cap, q) => out.push({piece, from, to, cap, next: {...q, stm: 'b'}});
    for (const s of kingSteps(p.wk)) {
      if (s === p.wq || adj(s, p.bk)) continue;
      if (s === p.br) { mk('K', p.wk, s, true, {...p, wk: s, br: -1}); continue; }
      if (ratt(p.br, s, [p.wq, p.bk])) continue;
      mk('K', p.wk, s, false, {...p, wk: s});
    }
    for (const line of ray(p.wq, KD))
      for (const t of line) {
        if (t === p.wk || t === p.bk) break;
        if (t === p.br) { mk('Q', p.wq, t, true, {...p, wq: t, br: -1}); break; }
        if (ratt(p.br, p.wk, [t, p.bk])) continue;
        mk('Q', p.wq, t, false, {...p, wq: t});
      }
    return out;
  }
  function genB(p) {
    if (p.wq < 0) return [];
    const out = [], mk = (piece, from, to, cap, q) => out.push({piece, from, to, cap, next: {...q, stm: 'w'}});
    for (const s of kingSteps(p.bk)) {
      if (s === p.br || adj(s, p.wk)) continue;
      if (s === p.wq) { mk('K', p.bk, s, true, {...p, bk: s, wq: -1}); continue; }
      if (qatt(p.wq, s, [p.wk, p.br])) continue;
      mk('K', p.bk, s, false, {...p, bk: s});
    }
    if (p.br >= 0)
      for (const line of ray(p.br, KD.slice(0, 4)))
        for (const t of line) {
          if (t === p.bk || t === p.wk) break;
          if (t === p.wq) { mk('R', p.br, t, true, {...p, br: t, wq: -1}); break; }
          if (qatt(p.wq, p.bk, [p.wk, t])) continue;
          mk('R', p.br, t, false, {...p, br: t});
        }
    return out;
  }
  const gen = p => (p.stm === 'w' ? genW(p) : genB(p));

  let TB = null;
  function setTables(buf) {
    TB = {dtm: buf.subarray(0, NC), dtc: buf.subarray(NC, 2 * NC),
          k3W: buf.subarray(2 * NC, 2 * NC + N3), k3B: buf.subarray(2 * NC + N3, 2 * NC + 2 * N3)};
  }

  // White moves still needed from p with best play; 255 = draw
  function value(p, m) {
    if (p.wq < 0) return 255;
    if (p.br < 0) {
      const v = (p.stm === 'w' ? TB.k3W : TB.k3B)[(p.wk << 12) | (p.wq << 6) | p.bk];
      return m === 'dtc' && v !== 255 ? 0 : v;
    }
    if (p.stm === 'w') return TB[m][canon(p.wk, p.wq, p.bk, p.br)];
    const ms = genB(p);
    if (!ms.length) return inCheck(p) ? 0 : 255;
    let mx = 0;
    for (const mv of ms) if ((mx = Math.max(mx, value(mv.next, m))) === 255) break;
    return mx;
  }
  // value after a move, counted from the position before it
  function moveValue(p, mv, m) {
    const v = value(mv.next, m);
    return p.stm === 'w' && v !== 255 ? v + 1 : v;
  }

  // squares the black king can reach through squares White does not attack
  function blackZone(p) {
    const safe = s => s !== p.br && !adj(s, p.wk) && s !== p.wk && (p.wq < 0 || s === p.wq || !qatt(p.wq, s, [p.wk, p.br]));
    const seen = new Set([p.bk]), todo = [p.bk];
    while (todo.length)
      for (const s of kingSteps(todo.pop())) if (!seen.has(s) && safe(s)) seen.add(s), todo.push(s);
    return seen;
  }
  function queenControl(p) {
    const out = new Set();
    if (p.wq < 0) return out;
    for (const line of ray(p.wq, KD))
      for (const t of line) {
        out.add(t);
        if (t === p.wk || t === p.br) break;
      }
    return out;
  }
  function transform(p, t) {
    const m = s => (s < 0 ? s : TF[t][s]);
    return {wk: m(p.wk), wq: m(p.wq), bk: m(p.bk), br: m(p.br), stm: p.stm};
  }

  G.KQKR = {NC, F, R, cheb, canon, uncanon, legal, inCheck, gen, genW, genB, setTables, value, moveValue,
            blackZone, queenControl, transform, tables: () => TB};
})(typeof window !== 'undefined' ? window : globalThis);
