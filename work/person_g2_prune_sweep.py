"""G2: single-variable prune-opacity sweep on the winner held-gate PLY.

Takes the aw0.65 held-gate winner (48/12 positional split, 30k iters,
L_outside mean 2.53%), post-hoc prunes sigmoid-opacity < tau, and scores
the 12 honest held views with the G1 CUDA renderer (calibrated vs brush,
NOTE-g1-renderer.md). tau = 0.0078 is brush's built-in MIN_OPACITY (2/255).

Primary: mean L_outside (RVM matte < 0.5). Guard: certain-FG recall
(matte > 0.99, alpha > 0.5). Single variable: tau only.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from splat_alpha_render import load_cameras, load_ply, gauss_cov3d
from splat_alpha_torch import project, render_alpha_gpu

ROOT = Path(__file__).resolve().parent.parent
PLY = ROOT / 'outputs/multi-view-scenes/20260915-043006-person-eval-held-aw0.65/scene.ply'
SPARSE = ROOT / 'outputs/multi-view-scenes/20260914-151822-person-evalsplit/sparse_txt_full'
MASKS = ROOT / 'outputs/multi-view-scenes/20260914-084036-person-57b59c/dataset/masks'
HELD = ['00001', '00009', '00016', '00026', '00039', '00048', '00056',
        '00067', '00079', '00088', '00100', '00109']
TAUS = [0.0, 2.0 / 255.0, 0.02, 0.05, 0.10, 0.15, 0.20]


def main() -> int:
    cams, views = load_cameras(SPARSE)
    viewmap = {x['stem']: x for x in views}
    xyz, scale, rot, opacity = load_ply(PLY)
    cov3d = gauss_cov3d(scale, rot)
    print(f'winner: {len(xyz)} gaussians; opacity med '
          f'{np.median(opacity):.4f} p10 {np.percentile(opacity, 10):.4f} '
          f'p5 {np.percentile(opacity, 5):.4f}')
    msk = {s: np.asarray(Image.open(MASKS / f'{s}.png').convert('L')).astype(np.float32) / 255.0
           for s in HELD}
    out = {'ply': str(PLY), 'n_total': len(xyz), 'taus': {}, 'views': HELD}
    t0 = time.time()
    for tau in TAUS:
        keep = np.nonzero(opacity >= tau)[0] if tau > 0 else np.arange(len(xyz))
        L, rec = [], []
        for s in HELD:
            vw = viewmap[s]
            cam = cams[vw['cam_id']]
            g = project(xyz[keep], cov3d[keep], opacity[keep],
                        cam, vw['q'], vw['t'], cam['w'], cam['h'])
            a = render_alpha_gpu(g, cam['w'], cam['h'])
            m = msk[s]
            outside = m < 0.5
            L.append(float(a[outside].sum()) / outside.sum())
            fg = m > 0.99
            rec.append(float((a[fg] > 0.5).sum()) / fg.sum())
        out['taus'][f'{tau:.4f}'] = {
            'n_gaussians': int(len(keep)),
            'L_outside_mean': float(np.mean(L)),
            'L_outside_min': float(np.min(L)),
            'L_outside_max': float(np.max(L)),
            'fg_recall_mean': float(np.mean(rec)),
            'per_view_L': {s: round(l, 5) for s, l in zip(HELD, L)},
        }
        r = out['taus'][f'{tau:.4f}']
        print(f"tau={tau:.4f}: n={r['n_gaussians']:>7}  L={r['L_outside_mean']*100:.3f}% "
              f"(min {r['L_outside_min']*100:.2f} max {r['L_outside_max']*100:.2f})  "
              f"FG-recall={r['fg_recall_mean']:.4f}")
    out['sec'] = round(time.time() - t0, 1)
    dst = ROOT / 'work/research/G2-prune-sweep.json'
    json.dump(out, open(dst, 'w'), indent=2)
    print(f'wrote {dst} ({out["sec"]}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
