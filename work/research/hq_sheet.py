"""Render 12 held views of a person PLY; build GT/raw/clip4 contact sheets.

Usage: python hq_sheet.py [person.ply]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy import ndimage

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
sys.path.insert(0, str(ROOT / 'work'))
from splat_alpha_render import load_cameras, load_ply, gauss_cov3d
from splat_rgb_render import render_rgb_alpha

JOB = ROOT / 'outputs/multi-view-scenes/20260915-111609-person-hq-vid174253'
OUT = ROOT / 'work/research/HQ-renders'
STEMS = [f'{i:05d}' for i in range(1, 112, 10)]
CLIP = 4


def main():
    ply = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / 'work/research/H2-extract-hq/keep-unc-despill-hq.ply'
    tag = ply.stem
    OUT.mkdir(exist_ok=True)
    cams, views = load_cameras(JOB / 'eval' / 'sparse_txt')
    vws = sorted((v for v in views if v['stem'] in set(STEMS)), key=lambda v: v['stem'])
    assert len(vws) == 12, len(vws)

    v = PlyData.read(str(ply))['vertex']
    xyz, scale, rot, opacity = load_ply(ply)
    colors = np.stack([np.asarray(v[f'f_dc_{i}']) for i in range(3)], axis=1).astype(np.float32)
    sh0 = 0.5 + colors / np.pi
    np.clip(sh0, 0.0, 1.0, out=sh0)
    cov3d = gauss_cov3d(scale, rot)
    print(f'{tag}: {len(xyz)} gaussians', flush=True)

    sheets = {'gt': [], 'raw': [], 'clip4': []}
    for vw in vws:
        stem = vw['stem']
        cam = cams[vw['cam_id']]
        w, h = cam['w'], cam['h']
        a, r = render_rgb_alpha(xyz, cov3d, opacity, sh0, cam, vw['q'], vw['t'], w, h)
        gt = np.asarray(Image.open(JOB / 'eval/images' / f'{stem}.png').convert('RGB'))
        mask = np.asarray(Image.open(JOB / 'eval/masks' / f'{stem}.png').convert('L'))
        a4 = a * ndimage.binary_dilation(mask > 127, iterations=CLIP).astype(np.float32)
        sheets['gt'].append(np.dstack([gt, np.full_like(gt[..., :1], 255)]))
        sheets['raw'].append(np.clip(np.dstack([r, a]) * 255, 0, 255).astype(np.uint8))
        sheets['clip4'].append(np.clip(np.dstack([r, a4]) * 255, 0, 255).astype(np.uint8))
        print(stem, 'alpha_mean', round(float(a.mean()), 4), flush=True)

    def montage(tiles, path, w=450):
        h = int(tiles[0].shape[0] * w / tiles[0].shape[1])
        cols, rows = 4, 3
        sheet = Image.new('RGB', (cols * w, rows * h), (30, 30, 30))
        for i, t in enumerate(tiles):
            sheet.paste(Image.fromarray(t).resize((w, h), Image.LANCZOS),
                        ((i % cols) * w, (i // cols) * h))
        sheet.save(path)
        print('wrote', path, flush=True)

    montage(sheets['gt'], OUT / f'sheet_{tag}_gt.png')
    montage(sheets['raw'], OUT / f'sheet_{tag}_raw.png')
    montage(sheets['clip4'], OUT / f'sheet_{tag}_clip4.png')
    print('done')


if __name__ == '__main__':
    main()
