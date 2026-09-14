"""E3: match_alpha_weight ablation {0.10, 0.20, 0.35, 0.50} — one variable.

Base: B2 RGBA dataset (120 views, full sparse, 1600px, 30k, growth_stop
12k, cap 1M). ONLY --match-alpha-weight changes. Each run exports
scene.ply into its own job dir; leakage measured with person_diag.py
against RVM soft mattes. Held-out decision needs EVAL split renders;
train-view leakage is reported as the diagnostic proxy.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
BASE = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-124953-person-b2-rgba30k' / 'dataset'
BRUSH = ROOT / 'work' / 'multiview-tools' / 'brush' / 'brush_app.exe'
WEIGHTS = ['0.10', '0.35', '0.50']  # 0.20 already exists as B2


def main() -> int:
    for w in WEIGHTS:
        ts = time.strftime('%Y%m%d-%H%M%S')
        job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-e3-aw{w}'
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
