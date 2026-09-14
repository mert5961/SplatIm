"""B5 step 2: dense static-window dataset with its own COLMAP poses.

Rationale: the orbit frames are 0.62s apart at 29.6fps, so a 24-frame
orbit window spans ~15s of subject motion. Re-extracting dense 30fps
frames for a short static segment (~4s, 120 frames) keeps the subject
near-rigid while COLMAP re-solves poses from the static studio.

Pipeline (fresh job dir, no reuse of orbit poses):
  1. Read the chosen window from /tmp/person_diag/b5_window.json; map
     orbit frame stems to video timestamps (frame i -> i * 74.11/120 s,
     duration 74.11s from ffprobe).
  2. ffmpeg: extract that segment at 30fps to job/input/*.jpg.
  3. RVM resnet50 (same as matte_person.py) over the dense frames ->
     job/mattes/*.png soft alpha.
  4. COLMAP from scratch on job/input (GPU SIFT, guided matching,
     mapper) -> job/dataset/sparse (+ database.db); abort unless
     >=70% frames aligned and >=200 sparse points (same gate as pipeline).
  5. Undistort at 1600px -> job/dataset/images/*.jpg.
  6. Compose RGBA (un-premultiplied) -> job/dataset/images/*.png,
     rewrite images.bin names .jpg->.png, delete jpgs, NO masks/.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
VIDEO = Path(r'C:\Users\reality\Downloads\VID20260914174253.mp4')
DUR = 74.11
FPS = 30
SEG = 4.0
COLMAP = ROOT / 'work' / 'multiview-tools' / 'colmap' / 'bin' / 'colmap.exe'
FFMPEG = (ROOT / 'work' / 'sharp-env' / 'Lib' / 'site-packages' / 'imageio_ffmpeg'
          / 'binaries' / 'ffmpeg-win-x86_64-v7.1.exe')


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(r.returncode)
    return r


def main() -> int:
    sys.path.insert(0, str(ROOT / 'work' / 'rvm'))
    import torch
    from PIL import Image
    from model import MattingNetwork

    win = json.loads(Path(r'/tmp/person_diag/b5_window.json').read_text())
    start_idx = win['best']['start']
    t0 = start_idx * (DUR / 120.0)
    print(f'window start orbit-frame={start_idx} t0={t0:.2f}s seg={SEG}s', flush=True)

    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-b5-window'
    (job / 'input').mkdir(parents=True)
    print('JOB', job, flush=True)

    run([str(FFMPEG), '-y', '-ss', f'{t0:.3f}', '-i', str(VIDEO),
         '-t', f'{SEG:.1f}', '-r', str(FPS), '-q:v', '2',
         str(job / 'input' / 'frame_%04d.jpg')])
    frames = sorted((job / 'input').glob('*.jpg'))
    print(f'{len(frames)} dense frames extracted', flush=True)
    if len(frames) < 30:
        raise SystemExit('too few frames extracted')

    # RVM mattes (recurrent, same settings as matte_person.py).
    device = 'cuda'
    model = MattingNetwork('resnet50').eval().to(device)
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth',
        map_location='cpu', progress=True))
    mattes = job / 'mattes'
    mattes.mkdir()
    rec = [None] * 4
    ds = None
    for i, p in enumerate(frames):
        img = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
        src = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
        if ds is None:
            ds = min(512 / max(src.shape[3], src.shape[4]), 1.0)
        with torch.no_grad():
            _, pha, *rec = model(src, *rec, ds)
        a = (pha[0, 0, 0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(a, 'L').save(mattes / f'{p.stem}.png')
        if i % 30 == 0:
            print(f'matte {i}/{len(frames)}', flush=True)
    print(f'{len(frames)} mattes written', flush=True)

    # COLMAP from scratch.
    db = job / 'database.db'
    if db.exists():
        db.unlink()
    run([str(COLMAP), 'feature_extractor', '--database_path', str(db),
         '--image_path', str(job / 'input'), '--ImageReader.single_camera', '1',
         '--ImageReader.camera_model', 'PINHOLE', '--SiftExtraction.use_gpu', '1'])
    run([str(COLMAP), 'exhaustive_matcher', '--database_path', str(db),
         '--SiftMatching.use_gpu', '1', '--SiftMatching.guided_matching', '1'])
    sparse = job / 'dataset' / 'sparse'
    sparse.mkdir(parents=True)
    run([str(COLMAP), 'mapper', '--database_path', str(db),
         '--image_path', str(job / 'input'), '--output_path', str(sparse)])

    # Gate + report.
    txt = job / 'sparse_txt'
    txt.mkdir()
    run([str(COLMAP), 'model_converter', '--input_path', str(sparse),
         '--output_path', str(txt), '--output_type', 'TXT'])
    n_img = sum(1 for ln in open(txt / 'images.txt') if not ln.startswith('#') and ln.strip()) // 2
    n_pts = sum(1 for ln in open(txt / 'points3D.txt') if not ln.startswith('#') and ln.strip())
    frac = n_img / len(frames)
    print(f'aligned {n_img}/{len(frames)} ({frac:.0%}), sparse {n_pts}', flush=True)
    if frac < 0.70 or n_pts < 200:
        raise SystemExit(f'weak alignment: {n_img}/{len(frames)}, {n_pts} pts')

    # Undistort + RGBA.
    run([str(COLMAP), 'image_undistorter', '--image_path', str(job / 'input'),
         '--input_path', str(sparse), '--output_path', str(job / 'dataset'),
         '--output_type', 'COLMAP', '--max_image_size', '1600'])
    imgs = job / 'dataset' / 'images'
    n = 0
    for jpg in sorted(imgs.glob('*.jpg')):
        m = mattes / f'{jpg.stem}.png'
        if not m.exists():
            jpg.unlink()
            continue
        rgb = np.asarray(Image.open(jpg).convert('RGB'))
        a = np.asarray(Image.open(m).convert('L'))
        if a.shape[:2] != rgb.shape[:2]:
            a = np.asarray(Image.open(m).convert('L').resize((rgb.shape[1], rgb.shape[0])))
        Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), 'RGBA').save(
            imgs / f'{jpg.stem}.png')
        jpg.unlink()
        n += 1
    print(f'{n} undistorted RGBA PNGs', flush=True)
    run([str(COLMAP), 'model_converter', '--input_path', str(sparse),
         '--output_path', str(txt), '--output_type', 'TXT'])
    t = (txt / 'images.txt').read_text()
    (txt / 'images.txt').write_text(t.replace('.jpg', '.png'))
    (sparse / 'images.bin').unlink(missing_ok=True)
    run([str(COLMAP), 'model_converter', '--input_path', str(txt),
         '--output_path', str(sparse), '--output_type', 'BIN'])
    print('JOB READY', job, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
