"""G1: minimal 3DGS alpha renderer — order-independent alpha accumulation.

alpha(px) = 1 - prod_i (1 - o_i * G_i(px)),  G_i = exp(-0.5 d^T conic_i d)
(d computed on the 2D-projected covariance; accumulation is order-free).

Purpose: fast held-view alpha maps for the prune T-sweep (EPIC G).
Validated against brush's rasterizer on the aw0.80 held renders
(metrics.json per_view L_outside). Tolerance: mean |dL| < 0.3pp and
per-view rank agreement.

Usage: python splat_alpha_render.py <scene.ply> <sparse_txt_dir> <masks_dir> <out_json> [stems...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
TILE = 32


def load_cameras(txt_dir: Path):
    cams = {}
    for ln in (txt_dir / 'cameras.txt').read_text().splitlines():
        if ln.startswith('#') or not ln.strip():
            continue
        p = ln.split()
        cams[int(p[0])] = {'model': p[1], 'w': int(p[2]), 'h': int(p[3]),
                           'fx': float(p[4]), 'fy': float(p[5]),
                           'cx': float(p[6]), 'cy': float(p[7])}
    views = []
    lines = [ln for ln in (txt_dir / 'images.txt').read_text().splitlines()
             if not ln.startswith('#') and ln.strip()]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        q = np.array([float(x) for x in p[1:5]])  # w x y z
        t = np.array([float(x) for x in p[5:8]])
        stem = Path(p[9]).stem
        views.append({'stem': stem, 'cam_id': int(p[8]), 'q': q, 't': t})
    return cams, views


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Vectorized w,x,y,z quaternion -> (N,3,3) rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((len(q), 3, 3), dtype=np.float32)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - w * z)
    m[:, 0, 2] = 2 * (x * z + w * y)
    m[:, 1, 0] = 2 * (x * y + w * z)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - w * x)
    m[:, 2, 0] = 2 * (x * z - w * y)
    m[:, 2, 1] = 2 * (y * z + w * x)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def load_ply(path: Path):
    from plyfile import PlyData
    v = PlyData.read(str(path))['vertex']
    xyz = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], 1).astype(np.float32)
    scale = np.exp(np.stack([np.asarray(v[f'scale_{i}']) for i in range(3)], 1).astype(np.float32))
    rot = np.stack([np.asarray(v[f'rot_{i}']) for i in range(4)], 1).astype(np.float32)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v['opacity']).astype(np.float32)))
    return xyz, scale, rot, opacity


def gauss_cov3d(scale, rot):
    R = quat_to_mat(rot)
    M = R * scale[:, None, :]  # R @ diag(scale)
    return M @ np.transpose(M, (0, 2, 1))  # (N,3,3)


def render_alpha(xyz, cov3d, opacity, cam, q, t, w, h):
    """Order-independent alpha map (h, w) float32."""
    R = quat_to_mat(q[None, :])[0]
    cam_pts = xyz @ R.T + t[None, :]
    z = cam_pts[:, 2]
    front = z > 0.2
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    u = fx * cam_pts[:, 0] / np.maximum(z, 1e-8) + cx
    v = fy * cam_pts[:, 1] / np.maximum(z, 1e-8) + cy

    idx = np.nonzero(front)[0]
    # Jacobian J = [[fx/z,0,-fx*x/z^2],[0,fy/z,-fy*y/z^2]]
    J0 = np.stack([fx / z[idx], np.zeros(len(idx)),
                   -fx * cam_pts[idx, 0] / (z[idx] ** 2)], 1).astype(np.float32)
    J1 = np.stack([np.zeros(len(idx)), fy / z[idx],
                   -fy * cam_pts[idx, 1] / (z[idx] ** 2)], 1).astype(np.float32)
    Sig = cov3d[idx]
    SigC = R[None, :, :] @ Sig @ R.T[None, :, :]  # cam space (k,3,3)
    tmp0 = (SigC @ J0[:, :, None])[:, :, 0]  # (k,3)
    tmp1 = (SigC @ J1[:, :, None])[:, :, 0]
    c00 = np.einsum('ki,ki->k', tmp0, J0)
    c01 = np.einsum('ki,ki->k', tmp0, J1)
    c11 = np.einsum('ki,ki->k', tmp1, J1)
    # low-rank regularizer (3DGS standard 0.3 blur in px^2)
    c00 += 0.3
    c11 += 0.3
    det = c00 * c11 - c01 * c01
    ok = det > 1e-9
    idx, c00, c01, c11, det = idx[ok], c00[ok], c01[ok], c11[ok], det[ok]
    conic00 = c11 / det
    conic01 = -c01 / det
    conic11 = c00 / det

    # radius: 3 sigma of larger eigenvalue
    mid = 0.5 * (c00 + c11)
    disc = np.sqrt(np.maximum(0.25 * (c00 - c11) ** 2 + c01 ** 2, 0))
    lam = mid + disc
    radius = np.ceil(3.0 * np.sqrt(lam)).astype(np.int32)
    radius = np.clip(radius, 2, 256)

    uu, vv = u[idx], v[idx]
    o = opacity[idx]

    logbuf = np.zeros((h, w), dtype=np.float64)
    # tile binning
    # vectorized tile binning: (gaussian, tile) pairs, sorted by tile
    nt_x = (w + TILE - 1) // TILE
    nt_y = (h + TILE - 1) // TILE
    t0x = np.clip(((uu - radius) // TILE).astype(np.int32), 0, nt_x - 1)
    t1x = np.clip(((uu + radius) // TILE).astype(np.int32), 0, nt_x - 1)
    t0y = np.clip(((vv - radius) // TILE).astype(np.int32), 0, nt_y - 1)
    t1y = np.clip(((vv + radius) // TILE).astype(np.int32), 0, nt_y - 1)
    gy_counts = (t1y - t0y + 1)
    gx_counts = (t1x - t0x + 1)
    pair_counts = gy_counts * gx_counts
    total_pairs = int(pair_counts.sum())
    if total_pairs == 0:
        return np.zeros((h, w), dtype=np.float32)
    # expand per-gaussian tile ranges
    gidx = np.repeat(np.arange(len(idx)), pair_counts)
    offs = np.repeat(np.cumsum(pair_counts) - pair_counts, pair_counts)
    steps = np.arange(total_pairs)
    within = steps - offs  # 0..pair_count-1 within each gaussian
    ny = gy_counts[gidx]
    wy = within % ny
    wx = within // ny[gidx]
    ty = (t0y[gidx] + wy).astype(np.int32)
    tx = (t0x[gidx] + wx).astype(np.int32)
    tkey = ty * nt_x + tx
    order = np.argsort(tkey, kind='stable')
    tkey_sorted = tkey[order]
    ks_all = gidx[order]
    boundaries = np.nonzero(np.diff(tkey_sorted))[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(tkey_sorted)]])
    for s, e in zip(starts, ends):
        ks = ks_all[s:e]
        tyx = divmod(int(tkey_sorted[s]), nt_x)
        ty, tx = tyx
        px0, py0 = tx * TILE, ty * TILE
        px1, py1 = min(px0 + TILE, w), min(py0 + TILE, h)
        gx = np.arange(px0, px1, dtype=np.float32)[None, :]
        gy = np.arange(py0, py1, dtype=np.float32)[:, None]
        dx = gx - uu[ks][:, None, None]
        dy = gy - vv[ks][:, None, None]
        a = conic00[ks][:, None, None] * dx * dx + \
            2 * conic01[ks][:, None, None] * dx * dy + \
            conic11[ks][:, None, None] * dy * dy
        alpha = o[ks][:, None, None] * np.exp(-0.5 * np.maximum(a, 0))
        np.clip(alpha, 0, 0.999999, out=alpha)
        contrib = np.log1p(-alpha).sum(axis=0)
        logbuf[py0:py1, px0:px1] += contrib
    return (1.0 - np.exp(logbuf)).astype(np.float32)


def main() -> int:
    ply = Path(sys.argv[1])
    txt_dir = Path(sys.argv[2])
    masks = Path(sys.argv[3])
    out_json = Path(sys.argv[4])
    stems = sys.argv[5:] or None

    cams, views = load_cameras(txt_dir)
    xyz, scale, rot, opacity = load_ply(ply)
    cov3d = gauss_cov3d(scale, rot)
    print(f'ply: {len(xyz)} gaussians', flush=True)

    if stems:
        views = [vw for vw in views if vw['stem'] in set(stems)]
    per = {}
    for vw in views:
        cam = cams[vw['cam_id']]
        w, h = cam['w'], cam['h']
        amap = render_alpha(xyz, cov3d, opacity, cam, vw['q'], vw['t'], w, h)
        m = np.asarray(Image.open(masks / f"{vw['stem']}.png").convert('L')).astype(np.float32) / 255.0
        if m.shape != (h, w):
            m = np.asarray(Image.open(masks / f"{vw['stem']}.png").convert('L')
                           .resize((w, h))).astype(np.float32) / 255.0
        outside = amap[m < 0.5]
        per[vw['stem']] = {
            'L_outside': round(float(outside.mean()), 5),
            'alpha_max_out': round(float(outside.max()), 4),
        }
        print(vw['stem'], per[vw['stem']], flush=True)
    Ls = [d['L_outside'] for d in per.values()]
    rep = {'ply': str(ply), 'n': len(xyz), 'n_views': len(per),
           'L_outside_mean': round(float(np.mean(Ls)), 5),
           'per_view': per}
    out_json.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != 'per_view'}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
