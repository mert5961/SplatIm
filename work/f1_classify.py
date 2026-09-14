"""F1: classify residual defects — matte/edge vs anatomical motion.

Loads the winner (E3 aw0.50) and B0 PLYs, projects Gaussian centers into
25 train views, and splits outside-mask opacity mass into:
  edge_band: within 8px of the matte boundary (alpha edge -> matte cause)
  far: >8px from any foreground (floaters -> seed/prune cause)
  ghost: opacity mass INSIDE mask but in high-uncertainty regions
         (U>0.5 from E1) -> motion/pose-smear suspect.
Anatomical verdict needs the user viewer check; this quantifies the split.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
WIN = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-145455-person-e3-aw0.50' / 'scene.ply'
B0 = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-095135-person-clean-3fbc0a' / 'scene.ply'
MASKS = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c' / 'dataset' / 'masks'
HYB = ROOT / 'work' / 'research' / 'HYBRID-60views'
OUT = ROOT / 'work' / 'research' / 'F1-defect-split.json'


def main() -> int:
    sys.path.insert(0, str(ROOT / 'work'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('pd', str(ROOT / 'work' / 'person_diag.py'))
    pd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pd)
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    cams = pd.read_cameras(Path(r'/tmp/person_diag/sparse_txt/cameras.txt'))
    images, order = pd.read_images(Path(r'/tmp/person_diag/sparse_txt/images.txt'))
    masks = {}
    for p in sorted(MASKS.glob('*.png')):
        masks[p.stem] = np.asarray(Image.open(p).convert('L')).astype(np.float32) / 255.0

    out = {}
    for tag, ply in (('B0', B0), ('AW050', WIN)):
        xs, ys, zs, ops = pd.read_ply_xyz_opacity(ply)
        sig = np.array([pd.sigmoid(o) for o in ops])
        edge_m = far_m = ghost_m = tot_m = 0.0
        n = 0
        for img_id in order[:60:3][:25]:
            info = images[img_id]
            m = masks[Path(info['name']).stem]
            cam = cams[info['cam']]
            R = pd.qvec_to_rot(info['qvec'])
            dt = distance_transform_edt(m > 0.5)
            unc_p = HYB / f'unc_{Path(info["name"]).stem}.npy'
            U = np.load(str(unc_p)) if unc_p.exists() else np.zeros_like(m)
            if U.shape != m.shape:
                U = np.asarray(Image.fromarray(U).resize((m.shape[1], m.shape[0]))).astype(np.float32)
            for x, y, z, a in zip(xs, ys, zs, sig):
                X = R[0][0] * x + R[0][1] * y + R[0][2] * z + info['tvec'][0]
                Y = R[1][0] * x + R[1][1] * y + R[1][2] * z + info['tvec'][1]
                Z = R[2][0] * x + R[2][1] * y + R[2][2] * z + info['tvec'][2]
                if Z <= 0:
                    continue
                fx, fy, cx, cy = cam['params'][:4]
                u = int(round(fx * X / Z + cx))
                v = int(round(fy * Y / Z + cy))
                if 0 <= v < m.shape[0] and 0 <= u < m.shape[1]:
                    if m[v, u] <= 0.5:
                        if dt[v, u] <= 8:
                            edge_m += a
                        else:
                            far_m += a
                    else:
                        tot_m += a
                        if U[v, u] > 0.5:
                            ghost_m += a
            n += 1
        tot = edge_m + far_m + tot_m
        out[tag] = {'views': n,
                    'edge_band_frac': round(float(edge_m / max(tot, 1e-9)), 4),
                    'far_floater_frac': round(float(far_m / max(tot, 1e-9)), 4),
                    'highU_inside_frac': round(float(ghost_m / max(tot_m, 1e-9)), 4)}
        print(tag, out[tag], flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
