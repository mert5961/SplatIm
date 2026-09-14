"""B5: motion-aware angular frame selection (~60 views, full 360 coverage).

Revision: fresh dense-window COLMAP fails (narrow baseline, no model),
and a fixed person crop is impossible (person fills 99-100% of frame on
a 360 orbit). So B5 keeps the proven orbit poses and selects ~60 views:
60 angular bins over unwrapped yaw, best frame per bin by masked flow
magnitude (tiebreak: sharpness). Then prunes images.txt/points3D.txt to
the subset, copies the 60 RGBA PNGs from the B2 job, retrains 30k.
"""
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
B2_JOB = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-124953-person-b2-rgba30k'
TXT_SRC = B2_JOB / 'dataset' / 'sparse_txt'
BRUSH = ROOT / 'work' / 'multiview-tools' / 'brush' / 'brush_app.exe'
COLMAP = ROOT / 'work' / 'multiview-tools' / 'colmap' / 'bin' / 'colmap.exe'
NBINS = 60


def per_frame_scores():
    import cv2
    from PIL import Image
    imgs = sorted((B2_JOB / 'dataset' / 'images').glob('*.png'))
    gray = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in imgs]
    motion, sharp = [], []
    for i, p in enumerate(imgs):
        a = np.asarray(Image.open(p).convert('RGBA'))[..., 3].astype(np.float32) / 255.0
        sharp.append(float(cv2.Laplacian(gray[i], cv2.CV_64F).var()))
        if i < len(imgs) - 1:
            flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i + 1], None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            m = (a > 0.05)
            motion.append(float(mag[m].mean()) if m.any() else 0.0)
    motion.append(motion[-1])
    return [p.stem for p in imgs], motion, sharp


def yaws():
    out = {}
    with open(TXT_SRC / 'images.txt') as f:
        lines = [ln for ln in f if not ln.startswith('#') and ln.strip()]
    for i in range(0, len(lines), 2):
        h = lines[i].split()
        qw, qx, qy, qz = (float(x) for x in h[1:5])
        yaw = math.atan2(2 * (qw * qy + qx * qz), 1 - 2 * (qy * qy + qx * qx))
        out[Path(h[9]).stem] = (int(h[0]), yaw)
    return out


def select(stems, motion, sharp, yawmap):
    ys = np.array([yawmap[s][1] for s in stems])
    uw = np.unwrap(ys)
    lo, hi = uw.min(), uw.max()
    bins = np.linspace(lo, hi, NBINS + 1)
    chosen = []
    for b in range(NBINS):
        idx = [i for i in range(len(stems)) if bins[b] <= uw[i] <= bins[b + 1]]
        if not idx:
            continue
        idx.sort(key=lambda i: (motion[i], -sharp[i]))
        chosen.append(stems[idx[0]])
    return sorted(chosen)


def prune_txt(src_txt, dst_txt, keep_stems, keep_ids):
    shutil.copy(src_txt / 'cameras.txt', dst_txt / 'cameras.txt')
    with open(src_txt / 'images.txt') as f:
        lines = [ln for ln in f if not ln.startswith('#') and ln.strip()]
    with open(dst_txt / 'images.txt', 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        for i in range(0, len(lines), 2):
            stem = Path(lines[i].split()[9]).stem
            if stem in keep_stems:
                f.write(lines[i] if lines[i].endswith('\n') else lines[i] + '\n')
                f.write(lines[i + 1] if lines[i + 1].endswith('\n') else lines[i + 1] + '\n')
    kept_pts = dropped_pts = 0
    with open(src_txt / 'points3D.txt') as fin, open(dst_txt / 'points3D.txt', 'w') as fout:
        for line in fin:
            if line.startswith('#') or not line.strip():
                fout.write(line)
                continue
            p = line.split()
            track = [(int(p[i]), int(p[i + 1])) for i in range(8, len(p), 2)]
            track = [(a, b) for a, b in track if a in keep_ids]
            if len(track) >= 2:
                fout.write(' '.join(p[:8] + [str(x) for pair in track for x in pair]) + '\n')
                kept_pts += 1
            else:
                dropped_pts += 1
    print(f'pruned: images {len(keep_stems)}, points kept={kept_pts} dropped={dropped_pts}', flush=True)
    return kept_pts


def main() -> int:
    t0 = time.time()
    stems, motion, sharp = per_frame_scores()
    yawmap = yaws()
    chosen = select(stems, motion, sharp, yawmap)
    keep_ids = {yawmap[s][0] for s in chosen}
    print(f'selected {len(chosen)}/{len(stems)} views in {time.time()-t0:.0f}s', flush=True)

    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-b5-sel60'
    job.mkdir(parents=True)
    print('JOB', job, flush=True)
    dataset = job / 'dataset'
    (dataset / 'images').mkdir(parents=True)
    for s in chosen:
        shutil.copy(B2_JOB / 'dataset' / 'images' / f'{s}.png', dataset / 'images' / f'{s}.png')
    txt = dataset / 'sparse_txt'
    txt.mkdir()
    npts = prune_txt(TXT_SRC, txt, set(chosen), keep_ids)
    (job / 'selection.json').write_text(json.dumps(
        {'kept': chosen, 'points': npts, 'nbins': NBINS}, indent=2))

    sparse = dataset / 'sparse'
    sparse.mkdir()
    r = subprocess.run([str(COLMAP), 'model_converter', '--input_path', str(txt),
                        '--output_path', str(sparse), '--output_type', 'BIN'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(r.returncode)

    cmd = [str(BRUSH), str(dataset),
           '--total-steps', '30000', '--growth-stop-iter', '12000',
           '--match-alpha-weight', '0.2', '--max-resolution', '1600',
           '--max-splats', '1000000', '--export-every', '30000',
           '--export-path', str(job), '--export-name', 'scene.ply']
    print('CMD', ' '.join(cmd), flush=True)
    t1 = time.time()
    with open(job / 'brush.log', 'w') as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f'exit={r.returncode} elapsed={time.time() - t1:.0f}s', flush=True)
    return r.returncode


if __name__ == '__main__':
    sys.exit(main())
