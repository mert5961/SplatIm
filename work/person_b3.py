"""B3: B2 RGBA pipeline + foreground-filtered sparse init.

B2 proved alpha supervision is the dominant fix (leakage 6.7% -> 1.2%).
Remaining noise source: 55% of COLMAP seeds sit <50% inside the person.
B3 filters points3D by multi-view mask support (track>=3, fg>=0.60,
median alpha>=0.50) and retrains the identical B2 config.

Reuses: B2 job's RGBA images + sparse_txt as the starting point, soft RVM
mattes for the support test. Output is a fresh job dir with its own
dataset (images copied, sparse filtered+renamed to .png, NO masks/).
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
B2_JOB = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-124953-person-b2-rgba30k'
MATTE_JOB = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c'
BRUSH = ROOT / 'work' / 'multiview-tools' / 'brush' / 'brush_app.exe'
COLMAP = ROOT / 'work' / 'multiview-tools' / 'colmap' / 'bin' / 'colmap.exe'


def load_txt_models(txt_dir):
    cams, images, order, points = {}, {}, [], {}
    with open(txt_dir / 'cameras.txt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            cams[int(p[0])] = {'model': p[1], 'params': [float(x) for x in p[4:]]}
    with open(txt_dir / 'images.txt') as f:
        lines = [ln for ln in f if not ln.startswith('#') and ln.strip()]
    for i in range(0, len(lines), 2):
        h = lines[i].split()
        images[int(h[0])] = {
            'qvec': [float(x) for x in h[1:5]], 'tvec': [float(x) for x in h[5:8]],
            'cam': int(h[8]), 'name': h[9]}
        order.append(int(h[0]))
    with open(txt_dir / 'points3D.txt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            points[int(p[0])] = {
                'xyz': [float(x) for x in p[1:4]], 'rgb': p[4:7], 'err': p[7],
                'track': ' '.join(p[8:])}
    return cams, images, order, points


def qvec_to_rot(q):
    qw, qx, qy, qz = q
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ]


def filter_points(txt_dir, masks):
    """Rewrite points3D.txt in place; returns (kept, dropped)."""
    from PIL import Image
    cams, images, order, points = load_txt_models(txt_dir)
    by_id = {i: images[i] for i in order}
    alpha = {}
    for p in sorted((MATTE_JOB / 'dataset' / 'masks').glob('*.png')):
        alpha[p.stem] = np.asarray(Image.open(p).convert('L')).astype(np.float32) / 255.0

    kept_lines, kept, dropped = [], 0, 0
    with open(txt_dir / 'points3D.txt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                kept_lines.append(line)
                continue
            p = line.split()
            pid = int(p[0])
            xyz = [float(x) for x in p[1:4]]
            track = [(int(p[i]), int(p[i + 1])) for i in range(8, len(p), 2)]
            vals = []
            for img_id, _ in track:
                info = by_id.get(img_id)
                if not info:
                    continue
                m = alpha.get(Path(info['name']).stem)
                if m is None:
                    continue
                R = qvec_to_rot(info['qvec'])
                X = R[0][0] * xyz[0] + R[0][1] * xyz[1] + R[0][2] * xyz[2] + info['tvec'][0]
                Y = R[1][0] * xyz[0] + R[1][1] * xyz[1] + R[1][2] * xyz[2] + info['tvec'][1]
                Z = R[2][0] * xyz[0] + R[2][1] * xyz[1] + R[2][2] * xyz[2] + info['tvec'][2]
                if Z <= 0:
                    continue
                fx, fy, cx, cy = cams[info['cam']]['params'][:4]
                u = int(round(fx * X / Z + cx))
                v = int(round(fy * Y / Z + cy))
                if 0 <= v < m.shape[0] and 0 <= u < m.shape[1]:
                    vals.append(m[v, u])
            if len(vals) >= 3 and (sum(1 for v in vals if v > 0.5) / len(vals)) >= 0.60 \
                    and float(np.median(vals)) >= 0.50:
                kept_lines.append(line)
                kept += 1
            else:
                dropped += 1
    with open(txt_dir / 'points3D.txt', 'w') as f:
        f.writelines(kept_lines)
        f.write(f'# FILTERED b3 keep={kept} drop={dropped}\n')
    print(f'sparse filter: kept={kept} dropped={dropped}', flush=True)
    return kept, dropped


def main() -> int:
    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-b3-filt30k'
    job.mkdir(parents=True)
    print('JOB', job, flush=True)
    dataset = job / 'dataset'
    shutil.copytree(B2_JOB / 'dataset' / 'images', dataset / 'images')
    txt_dir = dataset / 'sparse_txt'
    shutil.copytree(B2_JOB / 'dataset' / 'sparse_txt', txt_dir)
    # B2's sparse_txt already references .png; only filter points now.
    kept, dropped = filter_points(txt_dir, None)

    sparse_bin = dataset / 'sparse'
    sparse_bin.mkdir(exist_ok=True)
    r = subprocess.run([str(COLMAP), 'model_converter', '--input_path', str(txt_dir),
                        '--output_path', str(sparse_bin), '--output_type', 'BIN'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(r.returncode)
    print(f'sparse BIN rewritten (kept={kept})', flush=True)

    cmd = [str(BRUSH), str(dataset),
           '--total-steps', '30000', '--growth-stop-iter', '12000',
           '--match-alpha-weight', '0.2', '--max-resolution', '1600',
           '--max-splats', '1000000', '--export-every', '30000',
           '--export-path', str(job), '--export-name', 'scene.ply']
    print('CMD', ' '.join(cmd), flush=True)
    t0 = time.time()
    with open(job / 'brush.log', 'w') as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f'exit={r.returncode} elapsed={time.time() - t0:.0f}s', flush=True)
    return r.returncode


if __name__ == '__main__':
    sys.exit(main())
