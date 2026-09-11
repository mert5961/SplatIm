"""Regression check: tuned COLMAP settings must not hurt the easy sets."""
import json, sys, time
from pathlib import Path
import numpy as np
from plyfile import PlyData

ROOT = Path(r'C:/Users/reality/Documents/Codex/2026-09-11/i-w').resolve()
sys.path.insert(0, str(ROOT / 'outputs'))
from MultiViewSplat import train_scene

def run_case(name, files, expect_aligned, allow_partial=False):
    final, last = None, ''
    start = time.monotonic()
    for result in train_scene(files, 'Preview', True, allow_partial):
        label = result[1].split('\n')[0]
        if label != last:
            last = label
        final = result
    elapsed = time.monotonic() - start
    scene_ply, status, log_path = final[0], final[1], final[2]
    job = Path(log_path).parent
    alignment = json.loads((job / 'alignment.json').read_text(encoding='utf-8'))
    report = json.loads((job / 'report.json').read_text(encoding='utf-8'))
    ok = alignment['aligned_photos'] == expect_aligned and alignment['weak_alignment'] is False and scene_ply
    print(f'{name}: aligned={alignment["aligned_photos"]}/{alignment["unique_photos"]} '
          f'points={alignment["sparse_points"]} gaussians={report["gaussians"]} '
          f'weak={alignment["weak_alignment"]} elapsed={elapsed:.0f}s -> {"OK" if ok else "FAIL"}', flush=True)
    return ok

stills = sorted((ROOT / 'work' / 'multiview-test' / 'south-building' / 'images').glob('*.JPG'))[:20]
clip = ROOT / 'work' / 'multiview-test' / 'synthetic-walk.mp4'

ok1 = run_case('south-building', [str(p) for p in stills], 20)
ok2 = run_case('synthetic-clip', [str(clip)], 20)
print('REGRESSION', 'OK' if (ok1 and ok2) else 'FAIL', flush=True)
if not (ok1 and ok2):
    sys.exit(1)
