"""Split the uncertain set: person-edge splats (inside mask in most views,
keep at halved opacity) vs environment residue (drop).

Writes keep-edges-despill-hq.ply = keep + unc(inside_frac >= 0.5).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
sys.path.insert(0, str(ROOT / 'work'))
from splat_alpha_render import load_cameras, load_ply, quat_to_mat
from hq_clean import inside_frac, JOB, EXTRACT, THR


def main():
    cams, views = load_cameras(JOB / 'eval' / 'sparse_txt')
    masks = {p.stem: np.asarray(Image.open(p).convert('L'))
             for p in sorted((JOB / 'eval' / 'masks').glob('*.png'))}

    v = PlyData.read(str(JOB / 'scene.ply'))['vertex']
    xyz_all, _, _, _ = load_ply(JOB / 'scene.ply')
    ev = np.load(EXTRACT / 'evidence.npz')
    fg, ffrac = ev['fg'], ev['ffrac']
    keep = (fg >= 5.0) & (ffrac >= 0.5)
    hide = fg < 2.0
    unc = ~(keep | hide)
    sel = np.nonzero(unc)[0]
    print(f'unc set: {len(sel)} splats', flush=True)

    fr = inside_frac(xyz_all[sel], cams, views, masks)
    print(f'unc inside_frac: mean={fr.mean():.3f} median={np.median(fr):.3f} '
          f'p10={np.percentile(fr,10):.3f}')
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        print(f'  frac in [{lo:.1f},{hi:.1f}): {int(((fr >= lo) & (fr < hi)).sum())}')
    edge = fr >= THR
    print(f'unc edges (>= {THR}): {int(edge.sum())}, residue dropped: '
          f'{int((~edge).sum())}', flush=True)

    keep_idx = np.nonzero(keep)[0]
    unc_edge_idx = sel[edge]
    idx = np.concatenate([keep_idx, unc_edge_idx])
    data = np.ascontiguousarray(v.data[idx])
    n_keep = len(keep_idx)
    f_dc = np.stack([data[f'f_dc_{i}'].astype(np.float32) for i in range(3)], 1)
    sh0 = np.clip(0.5 + f_dc / np.pi, 0.0, 1.0)
    spill = np.maximum(0.0, sh0[:, 1] - np.maximum(sh0[:, 0], sh0[:, 2]))
    data['f_dc_0'] = ((sh0[:, 0] - 0.5) * np.pi).astype(np.float32)
    data['f_dc_1'] = ((sh0[:, 1] - 0.5) * np.pi).astype(np.float32)
    data['f_dc_2'] = ((sh0[:, 2] - 0.5) * np.pi).astype(np.float32)
    logit = data['opacity'].astype(np.float32)[n_keep:]
    data['opacity'][n_keep:] = np.log(np.exp(logit) * 0.5)
    out = EXTRACT / 'keep-edges-despill-hq.ply'
    PlyData([PlyElement.describe(data, 'vertex')], text=False).write(str(out))
    np.save(EXTRACT / 'unc_inside_frac.npy', fr)
    print(f'wrote {out} ({len(data)} splats: keep {n_keep} + '
          f'unc-edges {len(unc_edge_idx)})')


if __name__ == '__main__':
    main()
