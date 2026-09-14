"""B4: B3 pipeline + conservative alpha-aware green despill.

B3 reached the leakage floor attributable to matte edges (1.4%) but user
still reports green spill on skin/arms. Despill must be conservative:
only where green dominates AND alpha says foreground, strongest at
semi-transparent edges (alpha in 0.05..0.95), nearly zero in the solid
interior. Both RGB versions are retrained identically otherwise.

Despill operator (per pixel, RGB in [0,1], alpha a):
  excess = G - max(R, B); excess clipped at 0
  edge = 1 - |2a - 1|            # 0 in solid fg/bg, 1 at a=0.5
  strength = clip(excess,0,1) * (0.15 + 0.85*edge) * clip((a-0.02)/0.2,0,1)
  G' = G - strength * excess
Only the G channel is touched; R/B unchanged.
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


def despill(rgba: np.ndarray) -> np.ndarray:
    img = rgba.astype(np.float32) / 255.0
    r, g, b, a = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
    excess = np.clip(g - np.maximum(r, b), 0.0, 1.0)
    edge = 1.0 - np.abs(2.0 * a - 1.0)
    strength = excess * (0.15 + 0.85 * edge) * np.clip((a - 0.02) / 0.2, 0.0, 1.0)
    out = rgba.copy()
    out[..., 1] = np.clip((g - strength * excess) * 255.0, 0, 255).astype(np.uint8)
    return out


def main() -> int:
    from PIL import Image
    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-b4-despill30k'
    job.mkdir(parents=True)
    print('JOB', job, flush=True)
    dataset = job / 'dataset'
    shutil.copytree(B3_JOB / 'dataset' / 'sparse', dataset / 'sparse')
    images_dst = dataset / 'images'
    images_dst.mkdir(parents=True)
    n = 0
    for png in sorted((B3_JOB / 'dataset' / 'images').glob('*.png')):
        base = np.asarray(Image.open(png).convert('RGBA'))
        Image.fromarray(despill(base), 'RGBA').save(images_dst / png.name)
        n += 1
    print(f'{n} despilled RGBA PNGs written', flush=True)

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
    sys.exit(main())
