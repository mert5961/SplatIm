"""C2 (EPIC C): Sapiens2-1B per-frame alpha + premultiplied foreground.

Uses the repo vis tool (tools/vis/vis_matting.py + gss_p3m_metasim 1B
config) on the B5 60-view jpgs in the MAIN venv (torch 2.8 cu128).
Saves raw float32 alpha .npy per frame (--save_pred). Compares vs RVM
and MatAnyone2 on the trimap set: FG recall, BG leak, unknown-band MAE.
Temporal authority = NO (image model); boundary/foreground = YES.
CKPT env: SAPIENS_CHECKPOINT_ROOT not needed, checkpoint path passed.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
PY = ROOT / 'work' / 'sharp-env' / 'Scripts' / 'python.exe'
REPO = ROOT / 'work' / 'sapiens2repo'
DENSE = REPO / 'sapiens' / 'dense'
CFG = DENSE / 'configs' / 'matting' / 'gss_p3m_metasim' / 'sapiens2_1b_matting_gss_p3m_metasim-1024x768.py'
CKPT = Path(r'C:\Users\reality\.cache\huggingface\hub\models--facebook--sapiens2-matting-1b'
            r'\snapshots\88b79074a58a0e2bb5befc00a237488c43cd94da\sapiens2_1b_matting.safetensors')
STAGE = ROOT / 'work' / 'research' / 'SAM21-60views' / 'frames_jpg'
RVM_MASKS = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c' / 'dataset' / 'masks'
MA2_PHA = ROOT / 'work' / 'research' / 'MA2-60views' / 'frames_jpg_person60' / 'pha'
B5_SEL = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60' / 'selection.json'
OUT = ROOT / 'work' / 'research' / 'SAPIENS2-60views'


def main() -> int:
    from PIL import Image
    assert CFG.exists(), CFG
    assert CKPT.exists(), CKPT
    OUT.mkdir(parents=True, exist_ok=True)
    kept = json.loads(B5_SEL.read_text())['kept']

    t0 = time.time()
    r = subprocess.run(
        [str(PY), str(DENSE / 'tools' / 'vis' / 'vis_matting.py'), str(CFG), str(CKPT),
         '--input', str(STAGE), '--output', str(OUT / 'vis'), '--save_pred',
         '--device', 'cuda:0'],
        cwd=str(DENSE), capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
        return r.returncode
    npy = sorted((OUT / 'vis').rglob('*_alpha.npy'))
    print(f'sapiens alphas: {len(npy)} in {time.time()-t0:.0f}s', flush=True)
    assert len(npy) == 60, len(npy)

    # Compare vs RVM + MA2 on trimap set. vis tool names outputs 00000..
    # by sorted-input order: map positionally (sorted jpg names).
    jpg_order = sorted(p.stem for p in STAGE.glob('*.jpg'))
    kept_sorted = sorted(kept)
    # STAGE jpgs were staged in B5_RGBA order == sorted(kept) order
    pos_of = {stem: kept_sorted.index(stem) for stem in kept_sorted}
    du_r, du_m, fg, bg, bgr, bgm = [], [], [], [], [], []
    for p in sorted(tri_dir.glob('*.png')):
        if p.name == 'manifest.json':
            continue
        stem = p.stem
        if stem not in kept:
            continue
        tri = np.asarray(Image.open(p))
        pos = pos_of[stem]
        sp = np.load(str(npy[pos])).astype(np.float32)
        sp = np.clip(sp, 0, 1)
        if sp.shape != tri.shape:
            sp = np.asarray(Image.fromarray(sp).resize((tri.shape[1], tri.shape[0]))).astype(np.float32)
        rvm = np.asarray(Image.open(RVM_MASKS / f'{stem}.png').convert('L')).astype(np.float32) / 255.0
        ma2 = np.asarray(Image.open(ma2_files[kept.index(stem)]).convert('L')).astype(np.float32) / 255.0
        unk = tri == 128
        du_r.append(float(np.abs(sp[unk] - rvm[unk]).mean()))
        du_m.append(float(np.abs(sp[unk] - ma2[unk]).mean()))
        fgmask, bgmask = tri == 255, tri == 0
        fg.append(float(sp[fgmask].mean()))
        bg.append(float(sp[bgmask].mean()))
        bgr.append(float(rvm[bgmask].mean()))
        bgm.append(float(ma2[bgmask].mean()))
    rep = {'frames': len(npy), 'seconds': round(time.time() - t0, 1),
           'trimap_frames': len(du_r),
           'unknown_mae_vs_rvm': float(np.mean(du_r)),
           'unknown_mae_vs_ma2': float(np.mean(du_m)),
           'fg_recall_sapiens': float(np.mean(fg)),
           'bg_leak_sapiens': float(np.mean(bg)),
           'bg_leak_rvm': float(np.mean(bgr)),
           'bg_leak_ma2': float(np.mean(bgm))}
    (OUT / 'report.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
