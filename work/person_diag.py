"""Diagnostics for person-only 3DGS noise: sparse support + outside-alpha opacity.

Usage: person_diag.py [PLY_PATH] [MASK_DIR]
Defaults measure the 7k binary-mask baseline (B0).
"""
import argparse
import re
import struct
import sys
from pathlib import Path

ROOT = Path(r'C:/Users/reality/Documents/Codex/2026-09-11/i-w')
B0_JOB = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-095135-person-clean-3fbc0a'

def read_cameras(path):
    cams = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            cams[int(parts[0])] = {
                'model': parts[1], 'w': int(parts[2]), 'h': int(parts[3]),
                'params': [float(x) for x in parts[4:]],
            }
    return cams


def read_images(path):
    images, order = {}, []
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith('#') and ln.strip()]
    for i in range(0, len(lines), 2):
        head = lines[i].split()
        images[int(head[0])] = {
            'qvec': [float(x) for x in head[1:5]],
            'tvec': [float(x) for x in head[5:8]],
            'cam': int(head[8]), 'name': head[9],
        }
        order.append(int(head[0]))
    return images, order


def read_points(path):
    pts = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            pts[int(p[0])] = {
                'xyz': [float(x) for x in p[1:4]],
                'err': float(p[7]),
                'track': [(int(p[i]), int(p[i + 1])) for i in range(8, len(p), 2)],
            }
    return pts


def qvec_to_rot(q):
    qw, qx, qy, qz = q
    R = [[0.0] * 3 for _ in range(3)]
    R[0][0] = 1 - 2 * (qy * qy + qz * qz)
    R[0][1] = 2 * (qx * qy - qw * qz)
    R[0][2] = 2 * (qx * qz + qw * qy)
    R[1][0] = 2 * (qx * qy + qw * qz)
    R[1][1] = 1 - 2 * (qx * qx + qz * qz)
    R[1][2] = 2 * (qy * qz - qw * qx)
    R[2][0] = 2 * (qx * qz - qw * qy)
    R[2][1] = 2 * (qy * qz + qw * qx)
    R[2][2] = 1 - 2 * (qx * qx + qy * qy)
    return R


def project(xyz, info, cam):
    R = qvec_to_rot(info['qvec'])
    x, y, z = xyz
    X = R[0][0] * x + R[0][1] * y + R[0][2] * z + info['tvec'][0]
    Y = R[1][0] * x + R[1][1] * y + R[1][2] * z + info['tvec'][1]
    Z = R[2][0] * x + R[2][1] * y + R[2][2] * z + info['tvec'][2]
    if Z <= 0:
        return None
    if cam['model'] == 'PINHOLE':
        fx, fy, cx, cy = cam['params'][:4]
        return (fx * X / Z + cx, fy * Y / Z + cy)
    return None


def read_masks(mask_dir):
    sys.path.insert(0, str(ROOT / 'work' / 'sharp-env' / 'Lib' / 'site-packages'))
    from PIL import Image
    import numpy as np
    masks = {}
    for p in sorted(Path(mask_dir).glob('*.png')):
        a = np.asarray(Image.open(p).convert('L')).astype(np.float32) / 255.0
        masks[p.stem] = a
    return masks


def sparse_support(masks):
    cams = read_cameras(Path(r'/tmp/person_diag/sparse_txt/cameras.txt'))
    images, order = read_images(Path(r'/tmp/person_diag/sparse_txt/images.txt'))
    pts = read_points(Path(r'/tmp/person_diag/sparse_txt/points3D.txt'))

    n_pts = len(pts)
    full = mid = low = 0
    for pid, pt in pts.items():
        ratios = []
        for img_id, _ in pt['track']:
            info = images.get(img_id)
            if not info:
                continue
            m = masks.get(Path(info['name']).stem)
            if m is None or len(m) == 0:
                continue
            uv = project(pt['xyz'], info, cams[info['cam']])
            if uv is None:
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))
            if 0 <= v < m.shape[0] and 0 <= u < m.shape[1]:
                ratios.append(1.0 if m[v, u] > 0.5 else 0.0)
        if not ratios:
            continue
        r = sum(ratios) / len(ratios)
        if r >= 0.8:
            full += 1
        elif r >= 0.5:
            mid += 1
        else:
            low += 1
    total = full + mid + low
    print(f'sparse points: {n_pts}, observed in masks: {total}', flush=True)
    print(f'  >80% foreground: {full} ({full/max(total,1):.1%})', flush=True)
    print(f'  50-80% foreground: {mid} ({mid/max(total,1):.1%})', flush=True)
    print(f'  <50% foreground: {low} ({low/max(total,1):.1%})', flush=True)
    return full, mid, low


def read_ply_xyz_opacity(path):
    names, fmt, count = [], '', 0
    with open(path, 'rb') as f:
        line = f.readline().decode('ascii', 'replace')
        while line.strip() != 'end_header':
            m = re.match(r'element vertex (\d+)', line)
            if m:
                count = int(m.group(1))
            m = re.match(r'property (\w+) (\S+)', line)
            if m:
                fmt += m.group(1)[0]
                names.append(m.group(2))
            line = f.readline().decode('ascii', 'replace')
        data = f.read()
    row = struct.calcsize('<' + fmt)
    need = {'x': 0, 'y': 1, 'z': 2}
    xi, yi, zi = names.index('x'), names.index('y'), names.index('z')
    oi = names.index('opacity') if 'opacity' in names else None
    xs, ys, zs, ops = [], [], [], []
    for i in range(count):
        vals = struct.unpack_from('<' + fmt, data, i * row)
        xs.append(vals[xi]); ys.append(vals[yi]); zs.append(vals[zi])
        ops.append(vals[oi] if oi is not None else 1.0)
    return xs, ys, zs, ops


def sigmoid(x):
    import math
    return 1.0 / (1.0 + math.exp(-x))


def outside_mask_opacity(ply_path, masks):
    import numpy as np
    cams = read_cameras(Path(r'/tmp/person_diag/sparse_txt/cameras.txt'))
    images, order = read_images(Path(r'/tmp/person_diag/sparse_txt/images.txt'))
    xs, ys, zs, ops = read_ply_xyz_opacity(Path(ply_path))

    per_view_out = []
    use = [i for i in order[:60] if images[i]['name'].rsplit('.', 1)[0] in masks]
    for img_id in use[:25]:
        info = images[img_id]
        m = masks[Path(info['name']).stem]
        cam = cams[info['cam']]
        R = qvec_to_rot(info['qvec'])
        tot = out = 0.0
        for x, y, z, o in zip(xs, ys, zs, ops):
            X = R[0][0] * x + R[0][1] * y + R[0][2] * z + info['tvec'][0]
            Y = R[1][0] * x + R[1][1] * y + R[1][2] * z + info['tvec'][1]
            Z = R[2][0] * x + R[2][1] * y + R[2][2] * z + info['tvec'][2]
            if Z <= 0:
                continue
            fx, fy, cx, cy = cam['params'][:4]
            u = int(round(fx * X / Z + cx)); v = int(round(fy * Y / Z + cy))
            if 0 <= v < m.shape[0] and 0 <= u < m.shape[1]:
                a = sigmoid(o)
                tot += a
                if m[v, u] <= 0.5:
                    out += a
        if tot > 0:
            per_view_out.append(out / tot)
    if per_view_out:
        print(f'outside-alpha opacity over {len(per_view_out)} views: '
              f'min={min(per_view_out):.3f} mean={float(np.mean(per_view_out)):.3f} '
              f'max={max(per_view_out):.3f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ply', nargs='?', default=str(B0_JOB / 'scene.ply'))
    ap.add_argument('masks', nargs='?', default=str(B0_JOB / 'dataset' / 'masks'))
    args = ap.parse_args()
    masks = read_masks(args.masks)
    full, mid, low = sparse_support(masks)
    outside_mask_opacity(args.ply, masks)
