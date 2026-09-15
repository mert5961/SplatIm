"""H3: repair the extracted person PLY (despill + boundary shrink).

Reads H2-extract/keep.ply and applies:
  1. Despill: for each splat's base color (DC SH), suppress green spill:
     g' = g - DESPILL * max(0, g - max(r, b))
  2. Boundary shrink: splats whose mean 3D scale exceeds MAX_S (world m)
     are uniformly scaled down so smean == MAX_S (kills the soft halo
     caused by oversized boundary splats).
Writes H2-extract/keep-repaired.ply + a JSON report.

Usage: python h3_repair.py [keep.ply]
"""
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
DESPILL = 1.0      # 1.0 = hard despill
MAX_S = 0.010      # 10 mm cap on mean 3D scale


def main():
    keep = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / 'work/research/H2-extract/keep.ply'
    out = keep.parent / 'keep-repaired.ply'

    v = PlyData.read(str(keep))['vertex']
    n = len(v)
    f_dc = np.stack([np.asarray(v[f'f_dc_{i}']) for i in range(3)], 1).astype(np.float32)
    sh0 = np.clip(0.5 + f_dc / np.pi, 0.0, 1.0)

    # 1. despill (operate on base color, then convert back to f_dc)
    g = sh0[:, 1]
    spill = np.maximum(0.0, g - np.maximum(sh0[:, 0], sh0[:, 2]))
    n_spilled = int((spill > 1e-4).sum())
    g2 = g - DESPILL * spill
    sh0[:, 1] = np.clip(g2, 0.0, 1.0)
    f_dc_new = (sh0 - 0.5) * np.pi

    # 2. boundary shrink
    sc = np.stack([np.asarray(v[f'scale_{i}']) for i in range(3)], 1).astype(np.float32)
    s = np.exp(sc)
    smean = s.mean(1)
    over = smean > MAX_S
    n_over = int(over.sum())
    if n_over:
        f = np.where(over[:, None], (MAX_S / smean)[:, None], 1.0)
        sc = np.log(np.exp(sc) * f)

    # write: structured-array slice of original + replaced color/scale fields
    data = np.ascontiguousarray(v.data)
    for i in range(3):
        data[f'f_dc_{i}'] = f_dc_new[:, i]
        data[f'scale_{i}'] = sc[:, i]
    el = PlyElement.describe(data, 'vertex')
    PlyData([el], text=False).write(str(out))

    rep = {'in': str(keep), 'out': str(out), 'n': n,
           'despill': {'n_spilled': n_spilled,
                       'mean_spill': round(float(spill.mean()), 5),
                       'max_spill': round(float(spill.max()), 5)},
           'shrink': {'n_over_cap': n_over, 'cap_m': MAX_S,
                      'smean_p95_before': round(float(np.quantile(smean, 0.95)), 5),
                      'smean_p95_after': round(float(np.quantile(np.minimum(smean, MAX_S), 0.95)), 5)}}
    (ROOT / 'work/research/H3-repair.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
