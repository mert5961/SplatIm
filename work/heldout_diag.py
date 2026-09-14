"""Held-out leakage on the EVAL-split 48-train PLY, measured ONLY on the
12 held-out views (honest protocol; person_diag defaults to train views).

Usage: heldout_diag.py <ply> ; masks fixed to RVM soft mattes, views fixed
to EVAL-60-split held_out. Prints min/mean/max outside-alpha opacity.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality/Documents/Codex/2026-09-11/i-w'.replace('/', '\\'))
MASKS = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c' / 'dataset' / 'masks'
SPLIT = json.loads((ROOT / 'work' / 'research' / 'EVAL-60-split.json').read_text())
HELD = set(SPLIT['held_out'])


def main() -> int:
    sys.path.insert(0, str(ROOT / 'work'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('pd', str(ROOT / 'work' / 'person_diag.py'))
    pd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pd)
    from PIL import Image
    ply = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / 'outputs' / 'multi-view-scenes' / '20260914-151822-person-evalsplit' / 'train_scene.ply')
    masks = {}
    for p in sorted(MASKS.glob('*.png')):
        if p.stem in HELD:
            masks[p.stem] = np.asarray(Image.open(p).convert('L')).astype(np.float32) / 255.0
    assert len(masks) == 12, len(masks)
    cams = pd.read_cameras(Path(r'/tmp/person_diag/sparse_txt/cameras.txt'))
    images, order = pd.read_images(Path(r'/tmp/person_diag/sparse_txt/images.txt'))
    xs, ys, zs, ops = pd.read_ply_xyz_opacity(ply)
    per_view = []
    for img_id in order:
        info = images[img_id]
        stem = Path(info['name']).stem
        if stem not in masks:
            continue
        m = masks[stem]
        cam = cams[info['cam']]
        R = pd.qvec_to_rot(info['qvec'])
        tot = out = 0.0
        for x, y, z, o in zip(xs, ys, zs, ops):
            X = R[0][0] * x + R[0][1] * y + R[0][2] * z + info['tvec'][0]
            Y = R[1][0] * x + R[1][1] * y + R[1][2] * z + info['tvec'][1]
            Z = R[2][0] * x + R[2][1] * y + R[2][2] * z + info['tvec'][2]
            if Z <= 0:
                continue
            fx, fy, cx, cy = cam['params'][:4]
            u = int(round(fx * X / Z + cx))
            v = int(round(fy * Y / Z + cy))
            if 0 <= v < m.shape[0] and 0 <= u < m.shape[1]:
                a = pd.sigmoid(o)
                tot += a
                if m[v, u] <= 0.5:
                    out += a
        if tot > 0:
            per_view.append(out / tot)
    print(f'held-out ({len(per_view)} views): min={min(per_view):.4f} '
          f'mean={float(np.mean(per_view)):.4f} max={max(per_view):.4f}', flush=True)
    with open(ROOT / 'work' / 'research' / 'HELDOUT-48train.json', 'w') as f:
        json.dump({'ply': str(ply), 'views': len(per_view),
                   'min': min(per_view), 'mean': float(np.mean(per_view)),
                   'max': max(per_view)}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
