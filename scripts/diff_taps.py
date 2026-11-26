#!/usr/bin/env python3
"""Diff the kernel's per-layer taps (cmdline taps=1, qwen35 kernel only) against reference taps.
usage: diff_taps.py <serial.log> <reference_taps.json> [--tol 5e-3]
reference_taps.json: {"taps": [{"pos": int, "embed": S, "layers": [S x 24], "final_norm": S}, ...]} with
S = {"sum": f, "abs": f, "sq": f, "first8": [f x 8]} (sum, sum|x|, sum x^2 and the first 8 values of the activation
vector at that point, from a float32 reference run over the same token sequence).
asm line: TAP l=<E|F|n> p=<pos> t=<tok> sum=<hex f64 bits> abs=<..> sq=<..> f8=<8 hex f32 bits, comma sep>
Reports, per (pos, layer), max relative error over sum/abs/sq and first8; prints the first (pos, layer) over tol
and a per-layer summary (worst over positions)."""
import re, sys, json, struct
f64 = lambda h: struct.unpack('<d', struct.pack('<Q', int(h, 16)))[0]
f32 = lambda h: struct.unpack('<f', struct.pack('<I', int(h, 16)))[0]
a = [x for x in sys.argv[1:] if not x.startswith('--')]; tol = 5e-3
if '--tol' in sys.argv: tol = float(sys.argv[sys.argv.index('--tol') + 1])
log = open(a[0], errors='replace').read().replace('\r', '')
asm = {}
for m in re.finditer(r'TAP l=(E|F|\d+) p=(\d+) t=(\d+) sum=([0-9A-F]+) abs=([0-9A-F]+) sq=([0-9A-F]+) f8=((?:[0-9A-F]{8},?){8})', log):
    l = m[1] if m[1] in 'EF' else int(m[1])
    asm[(int(m[2]), l)] = dict(sum=f64(m[4]), abs=f64(m[5]), sq=f64(m[6]), first8=[f32(h) for h in m[7].split(',')])
gold = {}
for t in json.load(open(a[1]))['taps']:
    gold[(t['pos'], 'E')] = t['embed']
    for i, L in enumerate(t['layers']): gold[(t['pos'], i)] = L
    if 'final_norm' in t: gold[(t['pos'], 'F')] = t['final_norm']
def err(x, y): return abs(x - y) / max(abs(y), 1e-6)
def cmp(p, g):
    e = [err(p['abs'], g['abs']), err(p['sq'], g['sq']), abs(p['sum'] - g['sum']) / max(g['abs'], 1e-6)]  # sum normalised by abs (sum may be ~0)
    s = max(abs(v) for v in g['first8']) or 1.0
    e.append(max(abs(x - y) for x, y in zip(p['first8'], g['first8'])) / s)
    return max(e)
keys = sorted([k for k in gold if k in asm], key=lambda k: (k[0], -2 if k[1] == 'E' else (99 if k[1] == "F" else k[1])))
print(f'asm taps {len(asm)}  golden {len(gold)}  common {len(keys)}')
first = None; worst = {}
for k in keys:
    e = cmp(asm[k], gold[k]); worst[k[1]] = max(worst.get(k[1], 0), e)
    if e > tol and first is None: first = (k, e)
print('first over tol %g:' % tol, first)
for l in ['E'] + list(range(24)) + ['F']:
    if l in worst: print(f'  layer {l!s:>2} worst rel err {worst[l]:.2e}')
