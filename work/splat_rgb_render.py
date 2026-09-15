"""H0: RGB+alpha renderer — exact per-splat TBB, depth-ordered.

TBB: process splats front-to-back; each splat i contributes
  tau_i(px) = T_i(px) * W_i(px),  T_i = transmittance reaching the splat
  alpha = sum_i tau_i  (telescoping: exactly <= 1)
  color = sum_i tau_i c_i / alpha
W_i(px) = o_i * exp(-0.5 d^T conic_i d) (same 2D projection math, tiles,
0.3 regularizer and 3-sigma radius as G1/splat_alpha_render).

Per-splat sequential loop (exact); ~30 s/view for 400k splats.

Usage: python splat_rgb_render.py <scene.ply> <sparse_txt_dir> <out_dir> [stems...]
Writes <out_dir>/<stem>.png (RGBA) per view.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

from splat_alpha_render import load_cameras, load_ply, quat_to_mat, gauss_cov3d

TILE = 32
OFF = np.arange(32, dtype=np.float32)


def render_rgb_alpha(xyz, cov3d, opacity, colors, cam, q, t, w, h):
    """Exact TBB. Returns (alpha (h,w) f32, rgb (h,w,3) f32 in [0,1])."""
    R = quat_to_mat(q[None, :])[0]
    cam_pts = xyz @ R.T + t[None, :]
    z = cam_pts[:, 2]
    front = z > 0.2
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    u = fx * cam_pts[:, 0] / np.maximum(z, 1e-8) + cx
    v = fy * cam_pts[:, 1] / np.maximum(z, 1e-8) + cy

    idx = np.nonzero(front)[0]
    inb = (u[idx] >= 0) & (u[idx] < w) & (v[idx] >= 0) & (v[idx] < h)
    idx = idx[inb]
    alive = opacity[idx] > 1e-4
    idx = idx[alive]
    if len(idx) == 0:
        return np.zeros((h, w), np.float32), np.zeros((h, w, 3), np.float32)

    J0 = np.stack([fx / z[idx], np.zeros(len(idx)),
                   -fx * cam_pts[idx, 0] / (z[idx] ** 2)], 1).astype(np.float32)
    J1 = np.stack([np.zeros(len(idx)), fy / z[idx],
                   -fy * cam_pts[idx, 1] / (z[idx] ** 2)], 1).astype(np.float32)
    Sig = cov3d[idx]
    SigC = R[None, :, :] @ Sig @ R.T[None, :, :]
    tmp0 = (SigC @ J0[:, :, None])[:, :, 0]
    tmp1 = (SigC @ J1[:, :, None])[:, :, 0]
    c00 = np.einsum('ki,ki->k', tmp0, J0) + 0.3
    c01 = np.einsum('ki,ki->k', tmp0, J1)
    c11 = np.einsum('ki,ki->k', tmp1, J1) + 0.3
    det = c00 * c11 - c01 * c01
    ok = det > 1e-9
    idx, c00, c01, c11, det = idx[ok], c00[ok], c01[ok], c11[ok], det[ok]
    if len(idx) == 0:
        return np.zeros((h, w), np.float32), np.zeros((h, w, 3), np.float32)
    conic00 = c11 / det
    conic01 = -c01 / det
    conic11 = c00 / det
    mid = 0.5 * (c00 + c11)
    disc = np.sqrt(np.maximum(0.25 * (c00 - c11) ** 2 + c01 * c01, 0))
    radius = np.clip(np.ceil(3.0 * np.sqrt(mid + disc)).astype(np.int32), 2, 256)

    uu = u[idx].astype(np.float64)
    vv = v[idx].astype(np.float64)
    oo = opacity[idx]
    cc = colors[idx]
    rad = radius
    k00, k01, k11 = conic00, conic01, conic11
    zz = z[idx]
    order = np.argsort(zz, kind='stable')

    A = np.zeros((h, w), np.float32)
    tbuf = np.zeros((h, w), np.float64)
    rgbbuf = np.zeros((h, w, 3), np.float64)
    nt_x = (w + TILE - 1) // TILE
    nt_y = (h + TILE - 1) // TILE

    for gi in order:
        i = int(gi)
        oi = float(oo[i])
        ux, vx = uu[i], vv[i]
        r = int(rad[i])
        a00, a01, a11 = float(k00[i]), float(k01[i]), float(k11[i])
        t0x = max(int((ux - r) // TILE), 0)
        t1x = min(int((ux + r) // TILE), nt_x - 1)
        t0y = max(int((vx - r) // TILE), 0)
        t1y = min(int((vx + r) // TILE), nt_y - 1)
        c3 = cc[i]
        for ty in range(t0y, t1y + 1):
            py0 = ty * TILE
            py1 = min(py0 + TILE, h)
            row = OFF[:py1 - py0] + py0 - vx
            for tx in range(t0x, t1x + 1):
                px0 = tx * TILE
                px1 = min(px0 + TILE, w)
                col = OFF[:px1 - px0] + px0 - ux
                dx = col[None, :]
                dy = row[:, None]
                a = a00 * dx * dx + 2.0 * a01 * dx * dy + a11 * dy * dy
                W = np.clip(oi * np.exp(-0.5 * np.maximum(a, 0.0)), 0.0, 0.999999)
                sl = (slice(py0, py1), slice(px0, px1))
                T = 1.0 - A[sl]
                tau = (T * W).astype(np.float64)
                tbuf[sl] += tau
                rgbbuf[sl] += tau[..., None] * c3
                A[sl] = (1.0 - T * (1.0 - W)).astype(np.float32)

    alpha = tbuf.astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        rgb = np.where(tbuf[..., None] > 1e-6,
                       rgbbuf / np.maximum(tbuf, 1e-9)[..., None], 0.0).astype(np.float32)
    return alpha, rgb


def main() -> int:
    ply = Path(sys.argv[1])
    txt_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    stems = sys.argv[4:] or None
    out_dir.mkdir(parents=True, exist_ok=True)

    cams, views = load_cameras(txt_dir)
    v = PlyData.read(str(ply))['vertex']
    xyz, scale, rot, opacity = load_ply(ply)
    colors = np.stack([np.asarray(v[f'f_dc_{i}']) for i in range(3)], axis=1).astype(np.float32)
    sh0 = 0.5 + colors / np.pi  # DC SH term -> base color
    np.clip(sh0, 0.0, 1.0, out=sh0)
    cov3d = gauss_cov3d(scale, rot)
    print(f'ply: {len(xyz)} gaussians', flush=True)

    if stems:
        views = [vw for vw in views if vw['stem'] in set(stems)]
    for vw in views:
        cam = cams[vw['cam_id']]
        w, h = cam['w'], cam['h']
        alpha, rgb = render_rgb_alpha(xyz, cov3d, opacity, sh0, cam, vw['q'], vw['t'], w, h)
        rgba = np.dstack([rgb, alpha]).astype(np.float32) * 255.0
        Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8)).save(out_dir / f"{vw['stem']}.png")
        print(vw['stem'], 'alpha_mean', round(float(alpha.mean()), 4),
              'alpha_max', round(float(alpha.max()), 4), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
