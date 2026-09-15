"""H3: evaluate a person-only PLY against GT on 12 held views.

Regions identical to H0 (interior: erode 20 px, band: 12 px,
outside: dilate 4 px). Optional matte clip (compositing):
  --clip N   alpha *= dilate(binary mask, N px)  (default 0 = raw)

Usage: python h3_eval.py <person.ply> [--clip N]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import colour

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
sys.path.insert(0, str(ROOT / 'work'))
from splat_alpha_render import load_cameras, load_ply, gauss_cov3d
from splat_rgb_render import render_rgb_alpha

SPARSE = ROOT / 'outputs/multi-view-scenes/20260914-151822-person-evalsplit/sparse_txt_full'
IMAGES = ROOT / 'outputs/multi-view-scenes/20260914-151822-person-evalsplit/full/images'
MASKS = ROOT / 'outputs/multi-view-scenes/20260914-084036-person-57b59c/dataset/masks'
STEMS = ['00001', '00009', '00016', '00026', '00039', '00048',
         '00056', '00067', '00079', '00088', '00100', '00109']
ERODE = 20
BAND = 12


def de00(a, b):
    """a, b: (n,3) float [0,1] sRGB -> mean dE00 (identical to h0_compare)."""
    la = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(a, 0, 1)))
    lb = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(b, 0, 1)))
    return float(colour.delta_E(la, lb, method='CIE 2000').mean())


def main():
    from scipy import ndimage
    from plyfile import PlyData

    args = list(sys.argv[1:])
    def take(flag, cast, default):
        if flag in args:
            i = args.index(flag)
            val = cast(args[i + 1])
            del args[i:i + 2]
            return val
        return default
    def _g(name):
        return globals()[name]
    clip = take('--clip', int, 0)
    SPARSE = Path(take('--sparse', str, str(_g('SPARSE'))))
    IMAGES = Path(take('--images', str, str(_g('IMAGES'))))
    MASKS = Path(take('--masks', str, str(_g('MASKS'))))
    stems_arg = take('--stems', str, None)
    STEMS = stems_arg.split(',') if stems_arg else _g('STEMS')
    ply = Path(args[0])

    v = PlyData.read(str(ply))['vertex']
    xyz, scale, rot, opacity = load_ply(ply)
    colors = np.stack([np.asarray(v[f'f_dc_{i}']) for i in range(3)], axis=1).astype(np.float32)
    sh0 = 0.5 + colors / np.pi
    np.clip(sh0, 0.0, 1.0, out=sh0)
    cov3d = gauss_cov3d(scale, rot)
    print(f'ply: {len(xyz)} gaussians, clip={clip}', flush=True)

    cams, views = load_cameras(SPARSE)
    vwmap = {vw['stem']: vw for vw in views}
    res = {}
    for stem in STEMS:
        vw = vwmap[stem]
        cam = cams[vw['cam_id']]
        w, h = cam['w'], cam['h']
        gt = np.asarray(Image.open(IMAGES / f'{stem}.png').convert('RGB'))
        m = np.asarray(Image.open(MASKS / f'{stem}.png').convert('L')).astype(np.float32) / 255.0
        alpha, rgb = render_rgb_alpha(xyz, cov3d, opacity, sh0, cam, vw['q'], vw['t'], w, h)
        fg = m > 0.5
        if clip:
             alpha = alpha * ndimage.binary_dilation(fg, iterations=clip).astype(np.float32)
        outd = ndimage.binary_dilation(fg, iterations=8) & \
             ~ndimage.binary_dilation(fg, iterations=4)
        interior = ndimage.binary_erosion(fg, iterations=ERODE)
        band = fg & ~ndimage.binary_erosion(fg, iterations=BAND)
        d = de00(np.clip(rgb, 0, 1)[interior].reshape(-1, 3),
                 np.clip(gt / 255.0, 0, 1)[interior].reshape(-1, 3)) if interior.any() else 0.0
        db = de00(np.clip(rgb, 0, 1)[band].reshape(-1, 3),
                  np.clip(gt / 255.0, 0, 1)[band].reshape(-1, 3)) if band.any() else 0.0
        sil = alpha > 0.5
        iou = float((sil & fg).sum()) / max(float((sil | fg).sum()), 1)
        leak = float(alpha[outd].mean()) if outd.any() else 0.0
        res[stem] = {'interior_dE00': round(float(d), 3), 'band_dE00': round(float(db), 3),
                     'silhouette_iou': round(iou, 4), 'alpha_outside': round(leak, 5)}
        print(stem, res[stem], flush=True)
    keys = ['interior_dE00', 'band_dE00', 'silhouette_iou', 'alpha_outside']
    agg = {k: round(float(np.mean([res[s][k] for s in STEMS])), 4) for k in keys}
    agg['n_splat'] = int(len(xyz))
    out = ROOT / f'work/research/H3-eval-{ply.stem}-clip{clip}.json'
    out.write_text(json.dumps({'ply': str(ply), 'agg': agg, 'per_view': res}, indent=2))
    print('AGG', agg)
    print('wrote', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
