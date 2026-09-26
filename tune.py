# Coordinate-descent over integer eval weights; objective = exact number of failing won positions.
import subprocess, sys, re, json
depth = int(sys.argv[1]); prog = sys.argv[2] if len(sys.argv) > 2 else './pol'
flags = sys.argv[4].split() if len(sys.argv) > 4 else []  # e.g. '-r'
pool = ['cd', 'kk', 'rk', 'qk', 'mob', 'ed', 'co', 'box', 'kqk', 'wkc', 'rkk', 'kkm', 'qkm', 'rkm',
        'rmob', 'rsafe', 'rchk', 'rqa', 'kreg', 'bkst', 'desp']
w = dict.fromkeys(pool, 0); w.update(json.loads(sys.argv[3]) if len(sys.argv) > 3 else dict(cd=-10, kk=4, rk=-3))
cache = {}
def score(w):
    key = tuple(sorted((k, v) for k, v in w.items() if v))
    if key not in cache:
        out = subprocess.run([prog] + flags + [str(depth)] + [f"{k}={v}" for k, v in key], capture_output=True, text=True).stdout
        cache[key] = int(re.search(r'fail=(\d+)', out).group(1))
        print(cache[key], dict(key), flush=True)
    return cache[key]
best = score(w)
for rnd in range(10):
    improved = False
    for k in pool:
        for step in (1, -1, 3, -3, 8, -8):
            t = dict(w); t[k] += step
            s = score(t)
            if s < best: best, w, improved = s, t, True
    print('ROUND', rnd, best, {k: v for k, v in w.items() if v}, flush=True)
    if not improved: break
