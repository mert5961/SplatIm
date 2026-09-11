"""Run a real video through the video-input pipeline (COLMAP + Brush)."""
import json, sys, time
from pathlib import Path
import numpy as np
from plyfile import PlyData

ROOT = Path(r'C:/Users/reality/Documents/Codex/2026-09-11/i-w').resolve()
sys.path.insert(0, str(ROOT / 'outputs'))
from MultiViewSplat import train_scene

clip = Path(sys.argv[1])
quality = sys.argv[2] if len(sys.argv) > 2 else 'Preview'
allow_partial = (sys.argv[3].lower() == 'true') if len(sys.argv) > 3 else False
print(f'input: {clip} quality={quality} allow_partial={allow_partial}', flush=True)

final = None
last = ''
start = time.monotonic()
for result in train_scene([str(clip)], quality, True, allow_partial):
    label = result[1].split('\n')[0]
    if label != last:
        print(label, flush=True)
        last = label
    final = result
print(f'elapsed: {time.monotonic() - start:.1f}s', flush=True)

scene_ply, status, log_path = final[0], final[1], final[2]
job = Path(log_path).parent
print(f'job: {job.name}', flush=True)
print(f'status: {status}', flush=True)
alignment = json.loads((job / 'alignment.json').read_text(encoding='utf-8'))
print('alignment:', alignment, flush=True)
if scene_ply:
    report = json.loads((job / 'report.json').read_text(encoding='utf-8'))
    print('report:', report, flush=True)
    ply = PlyData.read(scene_ply)
    v = ply['vertex'].data
    assert len(v) > 0 and all(np.isfinite(v[name]).all() for name in v.dtype.names)
    print(f'scene.ply OK: {len(v):,} Gaussians', flush=True)
