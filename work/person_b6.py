"""B6: corrected alpha-aware green despill (v2) + retrain.

B4 bug: correction was excess^2 (strength already contained excess),
so edge fringe (excess~0.2) got only ~20% correction -> user still sees
cyclorama spill. v2 applies G' = G - k*excess with k matched to the
physics of background blend (stronger at low alpha):
  k = min(1.0, 0.15 + 1.7*(1-a)) for a > 0.02
a=0.1 -> 1.0 (full), a=0.5 -> 1.0, a=1.0 -> 0.15 (interior-safe).
R/B untouched. Same B3 sparse + 30k config otherwise.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
B3_JOB = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-125903-person-b3-filt30k'
BRUSH = ROOT / 'work' / 'multiview-tools' / 'brush' / 'brush_app.exe'


def despill_v2(rgba: np.ndarray) -> np.ndarray:
    img = rgba.astype(np.float32) / 255.0
    r, g, b, a = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
    excess = np.clip(g - np.maximum(r, b), 0.0, 1.0)
    k = np.minimum(1.0, 0.15 + 1.7 * (1.0 - a))
    k = np.where(a > 0.02, k, 0.0)
    out = rgba.copy()
    out[..., 1] = np.clip((g - k * excess) * 255.0, 0, 255).astype(np.uint8)
    return out


def verify() -> None:
    from PIL import Image
    for f in ('00000', '00119'):
        base = np.asarray(Image.open(
            B3_JOB / 'dataset' / 'images' / f'{f}.png').convert('RGBA')).astype(np.float32)
        for tag, arr in (('before', base), ('after', despill_v2(base).astype(np.float32))):
            rgb, al = arr[..., :3] / 255.0, arr[..., 3] / 255.0
            edge = (al > 0.05) & (al < 0.95)
            ex = np.clip(rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]), 0, 1)[edge]
            print(f'{f} {tag}: edge excess mean={ex.mean():.3f} p90={np.percentile(ex,90):.3f}',
                  flush=True)


def main() -> int:
    from PIL import Image
    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-b6-despillv2'
    job.mkdir(parents=True)
    print('JOB', job, flush=True)
    dataset = job / 'dataset'
    shutil.copytree(B3_JOB / 'dataset' / 'sparse', dataset / 'sparse')
    images_dst = dataset / 'images'
    images_dst.mkdir(parents=True)
    n = 0
    for png in sorted((B3_JOB / 'dataset' / 'images').glob('*.png')):
        base = np.asarray(Image.open(png).convert('RGBA'))
        Image.fromarray(despill_v2(base), 'RGBA').save(images_dst / png.name)
        n += 1
    print(f'{n} despill-v2 RGBA PNGs written', flush=True)

    cmd = [str(BRUSH), str(dataset),
           '--total-steps', '30000', '--growth-stop-iter', '12000',
           '--match-alpha-weight', '0.2', '--max-resolution', '1600',
           '--max-splats', '1000000', '--export-every', '30000',
           '--export-path', str(job), '--export-name', 'scene.ply']
    print('CMD', ' '.join(cmd), flush=True)
    t0 = time.time()
    with open(job / 'brush.log', 'w') as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f'exit={r.returncode} elapsed={time.time() - t0:.0f}s', flush=True)
    return r.returncode


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        verify()
    else:
        sys.exit(main())
