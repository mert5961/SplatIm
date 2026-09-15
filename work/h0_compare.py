"""H0: three-way loss measurement at identical held views.

Compares full-scene (B0) render vs masked-winner (E3x aw0.65) render against
the GT photo, per region: interior FG (eroded), boundary band, outside.
Also writes side-by-side + face/hair crops for visual judgment.

Usage: python h0_compare.py
Inputs (fixed paths):
  work/research/H0-compare/b0-full/<stem>.png      (RGBA render)
  work/research/H0-compare/masked-e3x/<stem>.png   (RGBA render)
  outputs/.../20260914-151822-person-evalsplit/full/images/<stem>.png  (GT)
  outputs/.../20260914-084036-person-57b59c/dataset/masks/<stem>.png   (RVM mask)
Output: work/research/H0-compare.json + side-by-side PNGs in
  work/research/H0-compare/sidebyside/
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import colour

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
HELD = ['00001', '00009', '00016', '00026', '00039', '00048', '00056',
        '00067', '00079', '00088', '00100', '00109']
GT_DIR = ROOT / 'outputs/multi-view-scenes/20260914-151822-person-evalsplit/full/images'
MASKS = ROOT / 'outputs/multi-view-scenes/20260914-084036-person-57b59c/dataset/masks'
OUT = ROOT / 'work/research/H0-compare'
BAND = 12  # px boundary band width
ERODE = 20  # px interior erosion


def regions(mask):
    from scipy import ndimage
    fg = mask > 0.5
    interior = ndimage.binary_erosion(fg, iterations=ERODE)
    band = fg & ~ndimage.binary_erosion(fg, iterations=BAND)
    return interior, band, ~fg


def de00(a, b):
    """a, b: (n,3) float [0,1] sRGB -> mean dE00."""
    la = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(a, 0, 1)))
    lb = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(b, 0, 1)))
    return float(colour.delta_E(la, lb, method='CIE 2000').mean())


def main():
    (OUT / 'sidebyside').mkdir(parents=True, exist_ok=True)
    per = {}
    agg = {k: [] for k in ('b0_int', 'mx_int', 'b0_band', 'mx_band',
                           'b0_out', 'mx_out', 'cross_int')}
    for stem in HELD:
        gt = np.asarray(Image.open(GT_DIR / f'{stem}.png').convert('RGB')).astype(np.float32) / 255.0
        m = np.asarray(Image.open(MASKS / f'{stem}.png').convert('L')).astype(np.float32) / 255.0
        if m.shape != gt.shape[:2]:
            m = np.asarray(Image.open(MASKS / f'{stem}.png').convert('L')
                           .resize((gt.shape[1], gt.shape[0]))).astype(np.float32) / 255.0
        interior, band, outside = regions(m)

        def load(p):
            r = np.asarray(Image.open(p).convert('RGBA')).astype(np.float32) / 255.0
            if r.shape[:2] != gt.shape[:2]:
                r = np.asarray(Image.open(p).convert('RGBA')
                               .resize((gt.shape[1], gt.shape[0]))).astype(np.float32) / 255.0
            return r[..., :3], r[..., 3]

        b0, b0a = load(OUT / 'b0-full' / f'{stem}.png')
        mx, mxa = load(OUT / 'masked-e3x' / f'{stem}.png')

        d = {}
        for name, reg in (('int', interior), ('band', band), ('out', outside)):
            if reg.sum() < 100:
                continue
            d[f'b0_{name}'] = round(de00(b0[reg], gt[reg]), 3)
            d[f'mx_{name}'] = round(de00(mx[reg], gt[reg]), 3)
        d['cross_int'] = round(de00(b0[interior], mx[interior]), 3)
        # alpha stats
        d['b0_alpha_out'] = round(float(b0a[outside].mean()), 5)
        d['mx_alpha_out'] = round(float(mxa[outside].mean()), 5)
        per[stem] = d
        for k in agg:
            if k in d:
                agg[k].append(d[k])
        print(stem, {k: d[k] for k in ('b0_int', 'mx_int', 'b0_band', 'mx_band',
                                       'cross_int', 'b0_alpha_out', 'mx_alpha_out')}, flush=True)

        # side-by-side: GT | B0 | masked (downscaled) + crops (face top-third, hair top-fifth)
        scale = 800 / max(gt.shape)
        def small(r):
            h, w = int(r.shape[0] * scale), int(r.shape[1] * scale)
            return np.asarray(Image.fromarray((r * 255).astype(np.uint8)).resize((w, h)))
        row = np.concatenate([small(gt), small(b0), small(mx)], axis=1)
        Image.fromarray(row).save(OUT / 'sidebyside' / f'{stem}.png')
        h, w = gt.shape[:2]
        crop = slice(int(h * 0.10), int(h * 0.55))
        rowc = np.concatenate([small(gt[crop, :]), small(b0[crop, :]), small(mx[crop, :])], axis=1)
        Image.fromarray(rowc).save(OUT / 'sidebyside' / f'{stem}_face.png')

    rep = {
        'note': 'dE00 vs GT per region. int=interior FG (20px eroded), '
                'band=12px boundary, out=outside mask. cross_int = B0 vs masked '
                'direct (what detail differs). lower=better.',
        'per_view': per,
        'means': {k: round(float(np.mean(v)), 3) if v else None for k, v in agg.items()},
    }
    (ROOT / 'work/research/H0-compare.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep['means'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
