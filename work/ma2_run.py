"""C1 (EPIC C): MatAnyone2 alphas on the B5 60-view set (isolated py3.10 env).

Runs the repo inference (warmup=10, r_erode/dilate=10, save_image) via the
ma2-env interpreter as a subprocess (file protocol: frames dir + first-frame
SAM mask in, pha/*.png out). Reports mean |alpha_ma2 - alpha_rvm| on the
trimap unknown band + certain-FG/BG behavior. No fusion yet.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
MA2_PY = ROOT / 'work' / 'ma2-env' / 'Scripts' / 'python.exe'
REPO = ROOT / 'work' / 'matanyone2repo'
CKPT = ROOT / 'work' / 'models' / 'matanyone2' / 'matanyone2.pth'
STAGE = ROOT / 'work' / 'research' / 'SAM21-60views' / 'frames_jpg'
SAM0 = ROOT / 'work' / 'research' / 'SAM21-60views' / 'mask_00000.npy'
RVM_MASKS = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c' / 'dataset' / 'masks'
B5_SEL = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60' / 'selection.json'
OUT = ROOT / 'work' / 'research' / 'MA2-60views'


def main() -> int:
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    kept = json.loads(B5_SEL.read_text())['kept']
    frames = sorted(STAGE.glob('*.jpg'))
    assert len(frames) == 60, len(frames)
    # first-frame SAM mask -> repo input mask (uint8 L, target=255).
    m0 = np.load(SAM0)
    m0 = m0[0] if m0.ndim == 3 else m0
    Image.fromarray(((m0 > 0).astype(np.uint8)) * 255).save(OUT / 'mask0.png')

    import os as _os
    ckpt_link = REPO / 'pretrained_models' / 'matanyone2.pth'
    ckpt_link.parent.mkdir(exist_ok=True)
    if ckpt_link.is_symlink() or ckpt_link.exists():
        ckpt_link.unlink()
    _os.link(CKPT, ckpt_link)

    t0 = time.time()
    r = subprocess.run(
        [str(MA2_PY), str(REPO / 'inference_matanyone2.py'),
         '-i', str(STAGE), '-m', str(OUT / 'mask0.png'),
         '-o', str(OUT), '-c', str(ckpt_link),
         '-w', '10', '-e', '10', '-d', '10',
         '--suffix', 'person60', '--save_image'],
        cwd=str(REPO), capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
    video_name = f'{STAGE.name}_person60'
    pha_dir = OUT / video_name / 'pha'
    phas = sorted(pha_dir.glob('*.png'))
    print(f'ma2 alphas: {len(phas)} in {time.time()-t0:.0f}s', flush=True)
    assert len(phas) == 60, len(phas)

    # Compare vs RVM on trimap unknown band + certain regions.
    tri_dir = ROOT / 'work' / 'research' / 'trimap-set'
    diffs_u, fg_rvm, fg_ma2, bg_rvm, bg_ma2 = [], [], [], [], []
    for p in sorted(tri_dir.glob('*.png')):
        if p.name == 'manifest.json':
            continue
        stem = p.stem
        if stem not in kept:
            continue
        tri = np.asarray(Image.open(p))
        ma2 = np.asarray(Image.open(pha_dir / f'{kept.index(stem):04d}.png').convert('L')).astype(np.float32) / 255.0
        rvm = np.asarray(Image.open(RVM_MASKS / f'{stem}.png').convert('L')).astype(np.float32) / 255.0
        if ma2.shape != tri.shape:
            ma2 = np.asarray(Image.open(pha_dir / f'{kept.index(stem):04d}.png').convert('L').resize(
                (tri.shape[1], tri.shape[0]))).astype(np.float32) / 255.0
        unk = tri == 128
        diffs_u.append(float(np.abs(ma2[unk] - rvm[unk]).mean()) if unk.any() else 0.0)
        fg = tri == 255
        bg = tri == 0
        fg_rvm.append(float(rvm[fg].mean())); fg_ma2.append(float(ma2[fg].mean()))
        bg_rvm.append(float(rvm[bg].mean())); bg_ma2.append(float(ma2[bg].mean()))
    rep = {'frames': len(phas), 'seconds': round(time.time() - t0, 1),
           'trimap_frames': len(diffs_u),
           'mean_abs_diff_unknown': float(np.mean(diffs_u)) if diffs_u else None,
           'fg_recall_rvm': float(np.mean(fg_rvm)) if fg_rvm else None,
           'fg_recall_ma2': float(np.mean(fg_ma2)) if fg_ma2 else None,
           'bg_leak_rvm': float(np.mean(bg_rvm)) if bg_rvm else None,
           'bg_leak_ma2': float(np.mean(bg_ma2)) if bg_ma2 else None}
    (OUT / 'report.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
