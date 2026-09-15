"""EVAL-split rerun at aw0.65 with alpha-capable eval_save_to_disk.

Uses the rebuilt RGBA binary. Flags needed:
- --eval-split-every N: hold out every Nth view as eval set
- --eval-every N: run eval pass every N steps (writes PNG to export-path)
- --eval-save-to-disk: actually write rendered eval images to disk
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
DATASET = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-151822-person-evalsplit' / 'train'
BRUSH = ROOT / 'work' / 'multiview-tools' / 'brush' / 'brush_app.exe'


def main() -> int:
    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-eval-aw065-rgba'
    job.mkdir(parents=True, exist_ok=True)
    cmd = [str(BRUSH), str(DATASET),
           '--total-steps', '30000', '--growth-stop-iter', '12000',
           '--match-alpha-weight', '0.65', '--max-resolution', '1600',
           '--max-splats', '1000000', '--export-every', '30000',
           '--export-path', str(job), '--export-name', 'scene.ply',
           '--eval-split-every', '5', '--eval-every', '5000',
           '--eval-save-to-disk']
    print('JOB', job, flush=True)
    t0 = time.time()
    log = open(job / 'brush.log', 'w', buffering=1)
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f'eval split aw0.65 RGBA exit={r.returncode} elapsed={time.time()-t0:.0f}s', flush=True)
    return r.returncode


if __name__ == '__main__':
    sys.exit(main())
