"""E3x: extend alpha-weight ablation upward {0.65, 0.80} on the B2 base.

E3 showed monotone leakage improvement 0.10->0.50 (1.3/1.2/1.0/0.9%).
Single variable changes: --match-alpha-weight only. Same B2 RGBA dataset,
30k, growth_stop 12k, cap 1M, 1600px.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
BASE = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-124953-person-b2-rgba30k' / 'dataset'
BRUSH = ROOT / 'work' / 'multiview-tools' / 'brush' / 'brush_app.exe'
WEIGHTS = ['0.65', '0.80']


def main() -> int:
    for w in WEIGHTS:
        ts = time.strftime('%Y%m%d-%H%M%S')
        job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-e3x-aw{w}'
        job.mkdir(parents=True)
        cmd = [str(BRUSH), str(BASE),
               '--total-steps', '30000', '--growth-stop-iter', '12000',
               '--match-alpha-weight', w, '--max-resolution', '1600',
               '--max-splats', '1000000', '--export-every', '30000',
               '--export-path', str(job), '--export-name', 'scene.ply']
        print('JOB', job, 'W', w, flush=True)
        t0 = time.time()
        with open(job / 'brush.log', 'w') as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f'w={w} exit={r.returncode} elapsed={time.time()-t0:.0f}s', flush=True)
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
