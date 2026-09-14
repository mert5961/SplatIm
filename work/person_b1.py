"""B1: isolate the training-duration effect.

Same binary-mask dataset as the 7k baseline
(outputs/multi-view-scenes/20260914-095135-person-clean-3fbc0a/dataset),
retrained at 30000 steps with identical optics (1600px, 3M cap).
Compares directly against report.json gaussians=88062.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
BASE = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-095135-person-clean-3fbc0a'
BRUSH = ROOT / 'work' / 'multiview-tools' / 'brush' / 'brush_app.exe'


def main() -> int:
    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-b1-30k'
    job.mkdir(parents=True)
    log_path = job / 'brush.log'
    cmd = [
        str(BRUSH), str(BASE / 'dataset'),
        '--total-steps', '30000',
        '--growth-stop-iter', '15000',
        '--max-resolution', '1600',
        '--max-splats', '3000000',
        '--export-every', '30000',
        '--export-path', str(job),
        '--export-name', 'scene.ply',
    ]
    print('CMD', ' '.join(cmd), flush=True)
    t0 = time.time()
    with open(log_path, 'w') as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f'exit={r.returncode} elapsed={time.time() - t0:.0f}s log={log_path}', flush=True)
    return r.returncode



if __name__ == '__main__':
    sys.exit(main())
