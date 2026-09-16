"""Clean the keep set: drop splats that project OUTSIDE the person mask in
most of the 120 orbit views (wall splats leaked through thin limbs).

Writes work/research/H2-extract-hq/keep-clean-despill-hq.ply
(keep-only, despilled; uncertain set excluded).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
sys.path.insert(0, str(ROOT / 'work'))
from splat_alpha_render import load_cameras, load_ply, quat_to_mat

JOB = ROOT / 'outputs/multi-view-scenes/20260915-111609-person-hq-vid174253'
EXTRACT = ROOT / 'work/research/H2-extract-hq'
THR = 0.5


def inside_frac(xyz, cams, views, masks):
    n = len(xyz)
    inside = np.zeros(n, np.int32)
    outside = np.zeros(n, np.int32)
    for vw in views:
        cam = cams[vw['cam_id']]
        R = quat_to_mat(vw['q'][None, :])[0]
        cp = xyz @ R.T + vw['t'][None, :]
        z = cp[:, 2]
        ok = z > 0.2
        u = cam['fx'] * cp[ok, 0] / z[ok] + cam['cx']
        vv = cam['fy'] * cp[ok, 1] / z[ok] + cam['cy']
        inb = (u >= 0) & (u < cam['w']) & (vv >= 0) & (vv < cam['h'])
        idx = np.nonzero(ok)[0][inb]
        hit = masks[vw['stem']][vv[inb].astype(int), u[inb].astype(int)] > 127
        inside[idx[hit]] += 1
        outside[idx[~hit]] += 1
    tot = (inside + outside).astype(np.float64)
    return np.where(tot > 0, inside / np.maximum(tot, 1), 0.0)


def main():
    cams, views = load_cameras(JOB / 'eval' / 'sparse_txt')
    masks = {p.stem: np.asarray(Image.open(p).convert('L'))
             for p in sorted((JOB / 'eval' / 'masks').glob('*.png'))}
    assert len(masks) == len(views) == 120, (len(masks), len(views))

    v = PlyData.read(str(JOB / 'scene.ply'))['vertex']
    xyz_all, _, _, _ = load_ply(JOB / 'scene.ply')
    ev = np.load(EXTRACT / 'evidence.npz')
    fg, ffrac = ev['fg'], ev['ffrac']
    keep = (fg >= 5.0) & (ffrac >= 0.5)
    sel = np.nonzero(keep)[0]
    print(f'keep set: {len(sel)} splats', flush=True)

    fr = inside_frac(xyz_all[sel], cams, views, masks)
    clean = fr >= THR
    print(f'inside_frac: mean={fr.mean():.3f} median={np.median(fr):.3f} '
          f'p10={np.percentile(fr,10):.3f} p90={np.percentile(fr,90):.3f}')
    for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        print(f'  frac in [{lo:.1f},{hi:.1f}): {int(((fr >= lo) & (fr < hi)).sum())}')
    print(f'clean (>= {THR}): {int(clean.sum())} kept, {int((~clean).sum())} dropped',
          flush=True)

    idx = sel[clean]
    data = np.ascontiguousarray(v.data[idx])
    f_dc = np.stack([data[f'f_dc_{i}'].astype(np.float32) for i in range(3)], 1)
    sh0 = np.clip(0.5 + f_dc / np.pi, 0.0, 1.0)
    spill = np.maximum(0.0, sh0[:, 1] - np.maximum(sh0[:, 0], sh0[:, 2]))
    sh0[:, 1] = np.clip(sh0[:, 1] - spill, 0.0, 1.0)
    data['f_dc_0'] = ((sh0[:, 0] - 0.5) * np.pi).astype(np.float32)
    data['f_dc_1'] = ((sh0[:, 1] - 0.5) * np.pi).astype(np.float32)
    data['f_dc_2'] = ((sh0[:, 2] - 0.5) * np.pi).astype(np.float32)
    out = EXTRACT / 'keep-clean-despill-hq.ply'
    PlyData([PlyElement.describe(data, 'vertex')], text=False).write(str(out))
    np.save(EXTRACT / 'inside_frac.npy', fr)
    print(f'despilled {int((spill > 1e-4).sum())}, wrote {out} ({len(data)} splats)')


if __name__ == '__main__':
    main()
