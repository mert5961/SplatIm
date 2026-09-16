"""Headless object pipeline for VID20260916150820 (hanging PCB + beacon + pin).

Object: black PCB board ("SPARL/COAL TRACK", CO logo) with a white-dome beacon
on the left and a brass connection pin at the bottom; hangs from a talent-track
loop with a cable coil below. Body is BLACK (green tints in frames = green
screen spill -> despilled after extraction).

Stages (idempotent; each skips when its output already exists):
  frames    extract 120 frames from the video
  colmap    SIFT affine features -> guided exhaustive matching -> mapper
  undistort COLMAP image_undistorter @ 1920 px
  sam2      SAM 2.1 box prompt on undistorted frame 0 -> 120 raw masks
  ma2       MatAnyone2 refinement over the 120 frames -> dataset/masks
  brush     30,000 steps, 1920 px, 3M cap, masked training -> scene.ply
  eval      sparse TXT + held GT pngs + eval masks
  extract   h2 mask-guided extraction -> keep-only + despill
            -> keep-despill-obj.ply (uncertain set excluded: env residue)
  evalh3    h3_eval raw + clip4 on 12 held views

Usage: python object_pipeline.py [stage ...]   (default: all, in order)
"""
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
VIDEO = Path(r'C:\Users\reality\Downloads\VID20260916150820.mp4')
TOOLS = ROOT / 'work' / 'multiview-tools'
COLMAP = TOOLS / 'colmap' / 'bin' / 'colmap.exe'
BRUSH = TOOLS / 'brush' / 'brush_app.exe'
JOBS = ROOT / 'outputs' / 'multi-view-scenes'
STATE = ROOT / 'work' / 'research' / 'OBJ-JOB.json'

CKPT_SAM2 = ROOT / 'work' / 'models' / 'sam2' / 'sam2.1_hiera_small.pt'
SAM2_CFG = ROOT / 'work' / 'sam2repo' / 'sam2' / 'configs' / 'sam2.1' / 'sam2.1_hiera_s.yaml'
MA2_PY = ROOT / 'work' / 'ma2-env' / 'Scripts' / 'python.exe'
MA2_REPO = ROOT / 'work' / 'matanyone2repo'
MA2_CKPT = ROOT / 'work' / 'models' / 'matanyone2' / 'matanyone2.pth'

MAX_FRAMES = 120
STEPS = 30000
RES = 1920
SPLAT_CAP = 3000000
HELD = [f'{i:05d}' for i in range(1, 120, 10)]
EXTRACT_OUT = ROOT / 'work' / 'research' / 'O2-extract-obj'
# Undistorted frame 0 (1076x1914) prompts
BOX = [420, 660, 660, 1120]      # x0,y0,x1,y1: board + pin
BEACON_BOX = [425, 865, 505, 940]  # x0,y0,x1,y1: white-dome beacon


def job_dir() -> Path:
    if STATE.exists():
        return Path(json.loads(STATE.read_text())['job'])
    job = JOBS / (time.strftime('%Y%m%d-%H%M%S') + '-object-vid150820')
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


def stage_sam2(job: Path):
    raw = job / 'sam2_raw'
    if len(list(raw.glob('*.png'))) >= MAX_FRAMES:
        log(job, 'sam2: masks present, skip')
        return
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    stage = job / 'sam2_frames'
    stage.mkdir(exist_ok=True)
    srcs = sorted((job / 'dataset' / 'images').glob('*.jpg'))
    for i, p in enumerate(srcs):
        dst = stage / f'{i:05d}.jpg'
        if not dst.exists():
            shutil.copyfile(p, dst)
    raw.mkdir(exist_ok=True)

    def _bool_mask(vm):
        # vm: tensor [B,1,H,W] (one entry per SAM2 object) -> union bool [H,W]
        return (vm.amax(dim=0) > 0).cpu().numpy()[0]

    log(job, f'sam2: box={BOX} beacon_box={BEACON_BOX}')
    predictor = build_sam2_video_predictor(str(SAM2_CFG), str(CKPT_SAM2), device='cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(stage))
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1, box=np.array(BOX))
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=2, box=np.array(BEACON_BOX))
        t0 = time.time()
        n = 0
        for fi, _, vm in predictor.propagate_in_video(state):
            m = _bool_mask(vm)
            Image.fromarray(m.astype(np.uint8) * 255).save(raw / f'{fi:05d}.png')
            n += 1
            if n % 20 == 0:
                log(job, f'sam2: {n}/{len(srcs)} ({time.time() - t0:.0f}s)')
    log(job, f'sam2: {n} masks in {time.time() - t0:.0f}s')


def stage_ma2(job: Path):
    masks = job / 'dataset' / 'masks'
    if len(list(masks.glob('*.png'))) >= MAX_FRAMES:
        log(job, 'ma2: masks present, skip')
        return
    import numpy as np
    from PIL import Image

    srcs = sorted((job / 'sam2_frames').glob('*.jpg'))
    assert len(srcs) == MAX_FRAMES, len(srcs)
    ckpt_link = MA2_REPO / 'pretrained_models' / 'matanyone2.pth'
    ckpt_link.parent.mkdir(exist_ok=True)
    if ckpt_link.is_symlink() or ckpt_link.exists():
        ckpt_link.unlink()
    if not ckpt_link.exists():
        os.link(MA2_CKPT, ckpt_link)
    out_root = job / 'ma2_out'
    t0 = time.time()
    r = subprocess.run(
        [str(MA2_PY), str(MA2_REPO / 'inference_matanyone2.py'),
         '-i', str(job / 'sam2_frames'), '-m', str(job / 'sam2_raw' / '00000.png'),
         '-o', str(out_root), '-c', str(ckpt_link),
         '-w', '10', '-e', '10', '-d', '10',
         '--suffix', 'obj120', '--save_image'],
        cwd=str(MA2_REPO), capture_output=True, text=True)
    tail = (r.stdout or '')[-800:] + (r.stderr or '')[-800:]
    with (job / 'pipeline.log').open('a', encoding='utf-8') as f:
        f.write(tail + '\n')
    if r.returncode != 0:
        raise RuntimeError(f'ma2 failed (exit {r.returncode})\n{tail}')
    pha_dir = out_root / 'sam2_frames_obj120' / 'pha'
    phas = sorted(pha_dir.glob('*.png'))
    if len(phas) < MAX_FRAMES:
        raise RuntimeError(f'ma2 produced only {len(phas)} alphas')
    masks.mkdir(parents=True, exist_ok=True)
    coves = []
    for i, p in enumerate(phas[:MAX_FRAMES]):
        a = np.asarray(Image.open(p).convert('L'))
        img = Image.open(job / 'dataset' / 'images' / f'{i:05d}.jpg')
        if a.shape[1:] != img.size[::-1]:
            a = np.asarray(Image.fromarray(a).resize(img.size[::-1]))
        Image.fromarray(a).save(masks / f'{i:05d}.png')
        coves.append(float((a > 127).mean()))
    good = int(sum(c > 0.002 for c in coves))
    if good < MAX_FRAMES // 2:
        log(job, f'ma2: temporal collapse ({MAX_FRAMES - good}/{MAX_FRAMES} empty) '
                f'-> SAM2 fallback')
        for i in range(MAX_FRAMES):
            shutil.copyfile(job / 'sam2_raw' / f'{i:05d}.png', masks / f'{i:05d}.png')
        coves = [float((np.asarray(Image.open(masks / f'{i:05d}.png')) > 0).mean())
                 for i in range(MAX_FRAMES)]
    log(job, f'ma2: {MAX_FRAMES} masks, mean object coverage '
             f'{float(np.mean(coves)):.3f} ({time.time() - t0:.0f}s)')


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


def stage_eval_layout(job: Path):
    eval_dir = job / 'eval'
    sparse_txt = eval_dir / 'sparse_txt'
    gt = eval_dir / 'images'
    emasks = eval_dir / 'masks'
    if (sparse_txt / 'images.txt').exists() and \
            len(list(gt.glob('*.png'))) >= len(HELD) and \
            len(list(emasks.glob('*.png'))) >= MAX_FRAMES:
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
    emasks.mkdir(exist_ok=True)
    for p in (job / 'dataset' / 'masks').glob('*.png'):
        dst = emasks / p.name
        if not dst.exists():
            shutil.copyfile(p, dst)
    log(job, f'eval layout: sparse_txt + {len(list(gt.glob("*.png")))} held GT '
             f'+ {len(list(emasks.glob("*.png")))} masks')


def build_keep_obj(job: Path):
    """keep-only + despill (uncertain excluded: env residue lesson from H2-hq)."""
    import numpy as np
    from plyfile import PlyData, PlyElement

    ev = np.load(EXTRACT_OUT / 'evidence.npz')
    fg, ffrac = ev['fg'], ev['ffrac']
    keep = (fg >= 5.0) & (ffrac >= 0.5)
    sel = np.nonzero(keep)[0]

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
    out = EXTRACT_OUT / 'keep-despill-obj.ply'
    EXTRACT_OUT.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, 'vertex')], text=False).write(str(out))
    log(job, f'keep-only: {len(data)} splats (of {len(v.data)}), '
             f'despilled {int((spill > 1e-4).sum())} -> {out.name}')
    return out


def stage_extract(job: Path):
    if (EXTRACT_OUT / 'evidence.npz').exists():
        log(job, 'extract: evidence present, skip h2_extract')
    else:
        EXTRACT_OUT.mkdir(parents=True, exist_ok=True)
        run_cmd(job, [sys.executable, str(ROOT / 'work' / 'h2_extract.py'),
                      job / 'scene.ply',
                      '--sparse', job / 'eval' / 'sparse_txt',
                      '--masks', job / 'eval' / 'masks',
                      '--out', EXTRACT_OUT,
                      '--report', EXTRACT_OUT / 'report.json'],
                'h2_extract (object)')
    if not (EXTRACT_OUT / 'keep-despill-obj.ply').exists():
        build_keep_obj(job)


def stage_evalh3(job: Path):
    stems = ','.join(HELD)
    for clip in (0, 4):
        ply = EXTRACT_OUT / 'keep-despill-obj.ply'
        tag = ply.stem + f'-clip{clip}'
        if (ROOT / 'work/research' / f'H3-eval-{tag}.json').exists():
            log(job, f'evalh3 {tag}: present, skip')
            continue
        cmd = [sys.executable, str(ROOT / 'work' / 'h3_eval.py'), ply,
               '--sparse', job / 'eval' / 'sparse_txt',
               '--images', job / 'eval' / 'images',
               '--masks', job / 'eval' / 'masks', '--stems', stems]
        if clip:
            cmd += ['--clip', str(clip)]
        run_cmd(job, cmd, f'h3_eval {tag}')


STAGES = {n: f for n, f in list(globals().items()) if n.startswith('stage_')}
ORDER = ['frames', 'colmap', 'undistort', 'sam2', 'ma2', 'brush',
         'eval_layout', 'extract', 'evalh3']


def main():
    stages = sys.argv[1:] or ORDER
    job = job_dir()
    log(job, f'pipeline start: {stages} (job {job.name})')
    for name in stages:
        STAGES[f'stage_{name}'](job)
    log(job, 'pipeline complete')


if __name__ == '__main__':
    main()
