"""Headless HQ pipeline for the VID20260914174253 orbit (30k steps, 1920 px).

Stages (idempotent; each skips when its output already exists):
  frames    extract 120 frames from the video (same sampling as the UI pipeline)
  colmap    SIFT affine features -> guided exhaustive matching -> mapper
  undistort COLMAP image_undistorter @ 1920 px
  masks     green-screen chroma key -> dataset/masks (brush ignores black)
  brush     30,000 steps, 1920 px, 3M splat cap -> job/scene.ply (B0-HQ)
  rvm       Robust Video Matting resnet50 over the 120 undistorted frames
  eval      eval layout: sparse TXT + held-view GT PNGs + RVM masks
  extract   h2_extract (mask-guided keep/uncertain/hide) + despill +
            keep-unc-despill-hq.ply (uncertain opacity halved)
  evalh3    h3_eval raw + clip4 on 12 held views (+ B0-HQ raw for baseline)

Usage: python person_hq_pipeline.py [stage ...]   (default: all, in order)
"""
import hashlib
import json
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
VIDEO = Path(r'C:\Users\reality\Downloads\VID20260914174253.mp4')
TOOLS = ROOT / 'work' / 'multiview-tools'
COLMAP = TOOLS / 'colmap' / 'bin' / 'colmap.exe'
BRUSH = TOOLS / 'brush' / 'brush_app.exe'
JOBS = ROOT / 'outputs' / 'multi-view-scenes'
STATE = ROOT / 'work' / 'research' / 'HQ-JOB.json'

MAX_FRAMES = 120
STEPS = 30000
RES = 1920
SPLAT_CAP = 3000000
HELD = ['00001', '00011', '00021', '00031', '00041', '00051',
        '00061', '00071', '00081', '00091', '00101', '00111']
EXTRACT_OUT = ROOT / 'work' / 'research' / 'H2-extract-hq'


def job_dir() -> Path:
    if STATE.exists():
        return Path(json.loads(STATE.read_text())['job'])
    job = JOBS / (time.strftime('%Y%m%d-%H%M%S') + '-person-hq-vid174253')
    job.mkdir(parents=True)
    STATE.write_text(json.dumps({'job': str(job)}, indent=2))
    return job


def log(job: Path, msg: str):
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    with (job / 'pipeline.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def run_cmd(job: Path, cmd, label: str):
    log(job, f'### {label}\n### {subprocess.list2cmdline([str(c) for c in cmd])}')
    with (job / 'pipeline.log').open('a', encoding='utf-8') as f:
        f.flush()
        proc = subprocess.Popen([str(c) for c in cmd], stdout=f,
                                stderr=subprocess.STDOUT, cwd=str(job),
                                creationflags=subprocess.CREATE_NO_WINDOW)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f'{label} failed (exit {rc})')
    log(job, f'{label} done')


def stage_frames(job: Path):
    images = job / 'input'
    images.mkdir(exist_ok=True)
    existing = sorted(images.glob('*.jpg'))
    if len(existing) >= MAX_FRAMES:
        log(job, f'frames: {len(existing)} already present, skip')
        return
    import math
    import imageio
    import imageio_ffmpeg
    from PIL import Image

    total_frames, total_secs = imageio_ffmpeg.count_frames_and_secs(str(VIDEO))
    effective = min(MAX_FRAMES, max(1, math.ceil(total_secs * 8)))
    step = max(1, round(total_frames / effective))
    picks = list(range(0, total_frames, step))[:MAX_FRAMES]
    hashes = set()
    for name in existing:
        with open(name, 'rb') as f:
            hashes.add(hashlib.sha256(f.read()).hexdigest())
    reader = imageio.get_reader(str(VIDEO), format='ffmpeg')
    counter = 0
    saved = 0
    try:
        for pick in picks:
            frame = Image.fromarray(reader.get_data(pick))
            digest = hashlib.sha256(frame.tobytes()).hexdigest()
            if digest in hashes:
                continue
            hashes.add(digest)
            name = f'{counter:05d}.jpg'
            frame.save(images / name, quality=95)
            counter += 1
            saved += 1
            if counter % 20 == 0:
                log(job, f'frames: {counter}/{MAX_FRAMES}')
    finally:
        reader.close()
    log(job, f'frames: saved {saved} (total {counter})')


def stage_colmap(job: Path):
    db = job / 'database.db'
    sparse = job / 'sparse'
    if any(sparse.rglob('images.bin')):
        log(job, 'colmap: sparse model present, skip')
        return
    run_cmd(job, [COLMAP, 'feature_extractor',
                  '--database_path', db, '--image_path', job / 'input',
                  '--ImageReader.camera_model', 'SIMPLE_RADIAL',
                  '--ImageReader.single_camera', '1',
                  '--FeatureExtraction.max_image_size', '2400',
                  '--SiftExtraction.estimate_affine_shape', '1'],
            'colmap 1/3 features')
    run_cmd(job, [COLMAP, 'exhaustive_matcher',
                  '--database_path', db,
                  '--FeatureMatching.guided_matching', '1'],
            'colmap 2/3 matching')
    (job / 'sparse').mkdir(exist_ok=True)
    run_cmd(job, [COLMAP, 'mapper',
                  '--database_path', db, '--image_path', job / 'input',
                  '--output_path', sparse],
            'colmap 3/3 mapper')
    best = 0
    for p in sparse.rglob('images.bin'):
        with open(p, 'rb') as f:
            best = max(best, struct.unpack('<Q', f.read(8))[0])
    log(job, f'colmap: {best} registered images')
    if best < 3:
        raise RuntimeError(f'COLMAP registered only {best} images')


def stage_undistort(job: Path):
    dataset = job / 'dataset'
    if (dataset / 'images').exists():
        log(job, 'undistort: dataset present, skip')
        return
    model = None
    for p in sorted((job / 'sparse').rglob('images.bin')):
        with open(p, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
        if model is None or n > model[0]:
            model = (n, p.parent)
    run_cmd(job, [COLMAP, 'image_undistorter',
                  '--image_path', job / 'input', '--input_path', model[1],
                  '--output_path', dataset, '--output_type', 'COLMAP',
                  '--max_image_size', str(RES)],
            f'undistort @ {RES} px')


def stage_masks(job: Path):
    dataset = job / 'dataset'
    masks = dataset / 'masks'
    if masks.exists():
        log(job, 'masks: present, skip')
        return
    import numpy as np
    from PIL import Image, ImageFilter

    images_dir = dataset / 'images'
    paths = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    coverage = []
    for p in paths:
        a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
        coverage.append(float(((a[..., 1] - np.maximum(a[..., 0], a[..., 2]) > 0.12)
                               & (a[..., 1] > 0.12)).mean()))
    mean_cov = float(np.mean(coverage))
    if mean_cov < 0.05:
        log(job, 'masks: no green backdrop detected; training without masks')
        return
    masks.mkdir()
    for p in paths:
        a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
        background = ((a[..., 1] - np.maximum(a[..., 0], a[..., 2]) > 0.12)
                      & (a[..., 1] > 0.12))
        mask = Image.fromarray(np.where(background, 0, 255).astype('uint8'), 'L') \
            .filter(ImageFilter.GaussianBlur(2))
        mask.save(masks / (p.stem + '.png'))
    log(job, f'masks: keyed {mean_cov:.0%} mean background out of training')


def stage_brush(job: Path):
    result = job / 'scene.ply'
    if result.exists():
        log(job, 'brush: scene.ply present, skip')
        return
    log(job, f'### brush {STEPS} steps @ {RES} px, cap {SPLAT_CAP}')
    cmd = [BRUSH, job / 'dataset', '--total-steps', str(STEPS),
           '--max-resolution', str(RES), '--max-splats', str(SPLAT_CAP),
           '--export-every', str(STEPS), '--export-path', job,
           '--export-name', 'scene.ply']
    with (job / 'brush.log').open('a', encoding='utf-8') as f:
        f.write('\n### ' + subprocess.list2cmdline([str(c) for c in cmd]) + '\n')
        f.flush()
        proc = subprocess.Popen([str(c) for c in cmd], stdout=f,
                                stderr=subprocess.STDOUT, cwd=str(job),
                                creationflags=subprocess.CREATE_NO_WINDOW)
        rc = proc.wait()
    if rc != 0 or not result.exists():
        raise RuntimeError(f'brush failed (exit {rc}, scene.ply={result.exists()})')
    from plyfile import PlyData
    n = len(PlyData.read(str(result))['vertex'])
    log(job, f'brush: scene.ply {n} gaussians')


def stage_rvm(job: Path):
    eval_dir = job / 'eval'
    masks = eval_dir / 'masks'
    done = len(list(masks.glob('*.png'))) if masks.exists() else 0
    if done >= MAX_FRAMES:
        log(job, f'rvm: {done} masks present, skip')
        return
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(ROOT / 'work' / 'rvm'))
    from model import MattingNetwork

    masks.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    model = MattingNetwork('resnet50').eval().to(device)
    model.load_state_dict(torch.hub.load_state_dict_from_url(
        'https://github.com/PeterL1n/RobustVideoMatting/releases/download/'
        'v1.0.0/rvm_resnet50.pth', map_location='cpu'))
    images = sorted((job / 'dataset' / 'images').glob('*.jpg'))
    rec = [None] * 4
    t0 = time.time()
    for i, p in enumerate(images):
        out = masks / (p.stem + '.png')
        if out.exists():
            continue
        img = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
        src = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
        downsample = min(512 / max(src.shape[3], src.shape[4]), 1.0)
        with torch.no_grad():
            _fgr, pha, *rec = model(src, *rec, downsample)
        alpha = (pha[0, 0, 0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(alpha, 'L').save(out)
        if i % 20 == 0:
            log(job, f'rvm: {i}/{len(images)} ({time.time() - t0:.0f}s)')
    log(job, f'rvm: {len(list(masks.glob("*.png")))} masks in {time.time() - t0:.0f}s')


def stage_eval_layout(job: Path):
    eval_dir = job / 'eval'
    sparse_txt = eval_dir / 'sparse_txt'
    gt = eval_dir / 'images'
    if (sparse_txt / 'images.txt').exists() and \
            len(list(gt.glob('*.png'))) >= len(HELD):
       log(job, 'eval layout: present, skip')
       return
    from PIL import Image
    sparse_txt.mkdir(parents=True, exist_ok=True)
    run_cmd(job, [COLMAP, 'model_converter',
                  '--input_path', job / 'dataset' / 'sparse',
                  '--output_path', sparse_txt, '--output_type', 'TXT'],
            'model_converter BIN->TXT')
    gt.mkdir(exist_ok=True)
    for stem in HELD:
        src = job / 'dataset' / 'images' / f'{stem}.jpg'
        dst = gt / f'{stem}.png'
        if not dst.exists():
            Image.open(src).convert('RGB').save(dst)
    log(job, f'eval layout: sparse_txt + {len(list(gt.glob("*.png")))} held GT pngs')


def build_keep_unc(job: Path):
    """despill keep∪uncertain, halve uncertain opacity (validated recipe)."""
    import numpy as np
    from plyfile import PlyData, PlyElement

    ev = np.load(EXTRACT_OUT / 'evidence.npz')
    fg, ffrac = ev['fg'], ev['ffrac']
    keep = (fg >= 5.0) & (ffrac >= 0.5)
    hide = fg < 2.0
    unc = ~(keep | hide)
    sel = np.nonzero(keep | unc)[0]

    ply = job / 'scene.ply'
    v = PlyData.read(str(ply))['vertex']
    data = np.ascontiguousarray(v.data[sel])
    f_dc = np.stack([data[f'f_dc_{i}'].astype(np.float32) for i in range(3)], 1)
    sh0 = np.clip(0.5 + f_dc / np.pi, 0.0, 1.0)
    spill = np.maximum(0.0, sh0[:, 1] - np.maximum(sh0[:, 0], sh0[:, 2]))
    sh0[:, 1] = np.clip(sh0[:, 1] - spill, 0.0, 1.0)
    data['f_dc_0'] = ((sh0[:, 0] - 0.5) * np.pi).astype(np.float32)
    data['f_dc_1'] = ((sh0[:, 1] - 0.5) * np.pi).astype(np.float32)
    data['f_dc_2'] = ((sh0[:, 2] - 0.5) * np.pi).astype(np.float32)
    unc_sel = unc[sel]
    logit = data['opacity'].astype(np.float32)[unc_sel]
    data['opacity'][unc_sel] = np.log(np.exp(logit) * 0.5)
    out = EXTRACT_OUT / 'keep-unc-despill-hq.ply'
    PlyData([PlyElement.describe(data, 'vertex')], text=False).write(str(out))
    log(job, f'keep-unc: {len(data)} splats (keep {int(keep.sum())} + '
             f'unc {int(unc.sum())}), despilled {int((spill > 1e-4).sum())}')
    return out


def stage_extract(job: Path):
    if (EXTRACT_OUT / 'evidence.npz').exists():
        log(job, 'extract: evidence present, skip h2_extract')
    else:
        run_cmd(job, [sys.executable, str(ROOT / 'work' / 'h2_extract.py'),
                      job / 'scene.ply',
                      '--sparse', job / 'eval' / 'sparse_txt',
                      '--masks', job / 'eval' / 'masks',
                      '--out', EXTRACT_OUT,
                      '--report', ROOT / 'work' / 'research' / 'H2-extract-hq.json'],
               'h2_extract (B0-HQ)')
    if not (EXTRACT_OUT / 'keep-unc-despill-hq.ply').exists():
        build_keep_unc(job)


def stage_evalh3(job: Path):
    import subprocess as sp

    stems = ','.join(HELD)
    common = ['--sparse', str(job / 'eval' / 'sparse_txt'),
              '--images', str(job / 'eval' / 'images'),
              '--masks', str(job / 'eval' / 'masks'), '--stems', stems]
    py = sys.executable
    runs = [
        (job / 'scene.ply', 0),
        (EXTRACT_OUT / 'keep-unc-despill-hq.ply', 0),
        (EXTRACT_OUT / 'keep-unc-despill-hq.ply', 4),
    ]
    for ply, clip in runs:
        tag = ply.stem + f'-clip{clip}'
        out_json = ROOT / 'work' / 'research' / f'H3-eval-{tag}.json'
        if out_json.exists():
            log(job, f'evalh3 {tag}: present, skip')
            continue
        cmd = [py, str(ROOT / 'work' / 'h3_eval.py'), str(ply)]
        if clip:
            cmd += ['--clip', str(clip)]
        cmd += common
        run_cmd(job, cmd, f'h3_eval {tag}')


STAGES = {
    'frames': stage_frames,
    'colmap': stage_colmap,
    'undistort': stage_undistort,
    'masks': stage_masks,
    'brush': stage_brush,
    'rvm': stage_rvm,
    'eval': stage_eval_layout,
    'extract': stage_extract,
    'evalh3': stage_evalh3,
}

ORDER = list(STAGES)


def main():
    wanted = sys.argv[1:] or ORDER
    job = job_dir()
    log(job, f'pipeline start: {wanted} (job {job.name})')
    for name in wanted:
        t0 = time.time()
        STAGES[name](job)
        log(job, f'stage {name}: {time.time() - t0:.0f}s')
    log(job, 'pipeline complete')


if __name__ == '__main__':
    main()
