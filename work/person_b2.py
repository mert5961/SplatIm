"""B2: soft-alpha RGBA supervision + 30k steps.

B1 showed duration alone densifies (88k -> 153k Gaussians) but raises
outside-mask opacity (6.7% -> 9.9%): masks/ only weights the loss, never
forces rendered alpha to zero. B2 instead writes soft RVM alpha into RGBA
PNGs in dataset/images/ with NO masks/ folder, so Brush runs the
alpha-match branch (train.rs:170): total_err + |alpha_in - alpha_pred| *
match_alpha_weight. Verified in scene.rs:233-254 that Brush premultiplies
the input itself, so the PNG must carry UN-premultiplied RGB + raw alpha.

Reuses poses + undistorted frames from 20260914-080603-07ed0a and RVM
soft mattes from 20260914-084036-person-57b59c (NOT the hardened binary
masks: soft edges are required for hair/silhouette supervision).
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
SRC_JOB = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-080603-07ed0a'
MATTE_JOB = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c'
BRUSH = ROOT / 'work' / 'multiview-tools' / 'brush' / 'brush_app.exe'
COLMAP = ROOT / 'work' / 'multiview-tools' / 'colmap' / 'bin' / 'colmap.exe'


def build_rgba_dataset(job: Path) -> None:
    from PIL import Image

    dataset = job / 'dataset'
    images_dst = dataset / 'images'
    images_dst.mkdir(parents=True)
    images_src = SRC_JOB / 'dataset' / 'images'
    masks_src = MATTE_JOB / 'dataset' / 'masks'

    n = 0
    for jpg in sorted(images_src.glob('*.jpg')):
        matte = masks_src / f'{jpg.stem}.png'
        if not matte.exists():
            print(f'WARN missing matte for {jpg.name}', flush=True)
            continue
        rgb = np.asarray(Image.open(jpg).convert('RGB'))
        a = np.asarray(Image.open(matte).convert('L'))
        rgba = np.dstack([rgb, a]).astype(np.uint8)
        Image.fromarray(rgba, 'RGBA').save(images_dst / f'{jpg.stem}.png')
        n += 1
    print(f'{n} RGBA PNGs (un-premultiplied) written', flush=True)

    # Rewrite the sparse model's image names .jpg -> .png via TXT roundtrip.
    txt_dir = dataset / 'sparse_txt'
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir(parents=True)
    r = subprocess.run([str(COLMAP), 'model_converter', '--input_path',
                        str(SRC_JOB / 'dataset' / 'sparse'),
                        '--output_path', str(txt_dir), '--output_type', 'TXT'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(r.returncode)
    images_txt = txt_dir / 'images.txt'
    text = images_txt.read_text()
    images_txt.write_text(text.replace('.jpg', '.png'))
    print(f'renamed {text.count(".jpg")} image entries -> .png', flush=True)
    sparse_bin = dataset / 'sparse'
    sparse_bin.mkdir(exist_ok=True)
    r = subprocess.run([str(COLMAP), 'model_converter', '--input_path', str(txt_dir),
                        '--output_path', str(sparse_bin), '--output_type', 'BIN'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(r.returncode)
    for p in ('cameras.bin', 'images.bin', 'points3D.bin'):
        assert (sparse_bin / p).exists(), p
    print(f'sparse BIN rewritten to {sparse_bin}', flush=True)


def main() -> int:
    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-b2-rgba30k'
    job.mkdir(parents=True)
    print('JOB', job, flush=True)
    build_rgba_dataset(job)

    cmd = [
        str(BRUSH), str(job / 'dataset'),
        '--total-steps', '30000',
        '--growth-stop-iter', '12000',
        '--match-alpha-weight', '0.2',
        '--max-resolution', '1600',
        '--max-splats', '1000000',
        '--export-every', '30000',
        '--export-path', str(job),
        '--export-name', 'scene.ply',
    ]
    print('CMD', ' '.join(cmd), flush=True)
    t0 = time.time()
    with open(job / 'brush.log', 'w') as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f'exit={r.returncode} elapsed={time.time() - t0:.0f}s', flush=True)
    return r.returncode


if __name__ == '__main__':
    sys.exit(main())
