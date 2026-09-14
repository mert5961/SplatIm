"""E4: 1600 vs 1920 on the winner (alpha-weight 0.50, B2 base).

Undistorted source is 899x1600; 1920 training upsamples (no new geometry,
tests whether extra pixels help the alpha edge). Only resolution changes:
--max-resolution 1920, same 30k/growth_stop 12k/aw0.50/cap 1M.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
BASE = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-124953-person-b2-rgba30k' / 'dataset'
BRUSH = ROOT / 'work' / 'multiview-tools' / 'brush' / 'brush_app.exe'


def main() -> int:
    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-e4-res1920'
    job.mkdir(parents=True)
    cmd = [str(BRUSH), str(BASE),
           '--total-steps', '30000', '--growth-stop-iter', '12000',
           '--match-alpha-weight', '0.50', '--max-resolution', '1920',
           '--max-splats', '1000000', '--export-every', '30000',
           '--export-path', str(job), '--export-name', 'scene.ply']
    print('JOB', job, flush=True)
    print('CMD', ' '.join(cmd), flush=True)
    t0 = time.time()
    with open(job / 'brush.log', 'w') as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f'exit={r.returncode} elapsed={time.time()-t0:.0f}s', flush=True)
    return r.returncode


if __name__ == '__main__':
    sys.exit(main())
