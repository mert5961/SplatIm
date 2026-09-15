"""G1 GPU core: torch/CUDA rasterization of the order-independent alpha map.

alpha(px) = 1 - prod_i (1 - o_i * G_i(px));  accumulate log1p(-alpha)
via index_add_ (atomicAdd) over ragged per-gaussian bboxes.

Reuses loaders from splat_alpha_render. CLI matches splat_alpha_render.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from splat_alpha_render import load_cameras, load_ply, gauss_cov3d, quat_to_mat

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
DEV = 'cuda'
RADIUS_CAP = 512
PX_BUDGET = 400_000_000  # max bbox pixels per chunk


def project(xyz, cov3d, opacity, cam, q, t, w, h):
    """numpy -> per-gaussian torch tensors: u, v, conic00/01/11, radius, o."""
    R = quat_to_mat(q[None, :])[0]
    cp = xyz @ R.T + t[None, :]
    z = cp[:, 2]
    idx = np.nonzero(z > 0.2)[0]
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    J0 = np.stack([fx / z[idx], np.zeros(len(idx)),
                   -fx * cp[idx, 0] / (z[idx] ** 2)], 1).astype(np.float32)
    J1 = np.stack([np.zeros(len(idx), ), fy / z[idx],
                   -fy * cp[idx, 1] / (z[idx] ** 2)], 1).astype(np.float32)
    SigC = R[None, :, :] @ cov3d[idx] @ R.T[None, :, :]
    tmp0 = (SigC @ J0[:, :, None])[:, :, 0]
    tmp1 = (SigC @ J1[:, :, None])[:, :, 0]
    c00 = np.einsum('ki,ki->k', tmp0, J0) + 0.3
    c01 = np.einsum('ki,ki->k', tmp0, J1)
    c11 = np.einsum('ki,ki->k', tmp1, J1) + 0.3
    det = c00 * c11 - c01 * c01
    ok = det > 1e-9
    idx, c00, c01, c11, det = idx[ok], c00[ok], c01[ok], c11[ok], det[ok]
    # conic = inverse of cov2d (px^2 -> px^-2)
    conic00 = c11 / det
    conic01 = -c01 / det
    conic11 = c00 / det
    # NOTE: brush render path applies no cov_compensation (training only)
    mid = 0.5 * (c00 + c11)
    disc = np.sqrt(np.maximum(0.25 * (c00 - c11) ** 2 + c01 ** 2, 0))
    radius = np.clip(np.ceil(3.32 * np.sqrt(mid + disc)), 2, RADIUS_CAP).astype(np.int64)
    u = (fx * cp[idx, 0] / z[idx] + cx).astype(np.float32)
    v = (fy * cp[idx, 1] / z[idx] + cy).astype(np.float32)
    o = opacity[idx].astype(np.float32)
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
    return dict(idx=idx, u=to(u), v=to(v), c00=to(conic00), c01=to(conic01),
                c11=to(conic11), radius=to(radius), o=to(o))


def render_alpha_gpu(g, w, h):
    """g: dict from project() -> alpha map (h, w) numpy float32."""
    logbuf = torch.zeros(h * w, dtype=torch.float32, device=DEV)
    rx = g['radius']
    ry = g['radius']
    bw = 2 * rx + 1
    bh = 2 * ry + 1
    npx = bw * bh
    total = int(npx.sum().item())
    csum = torch.cumsum(npx, 0)
    # chunk over gaussians by pixel budget
    starts = [0]
    acc = 0
    for gi in range(len(rx)):
        acc += int(npx[gi].item())
        if acc >= PX_BUDGET:
            starts.append(gi + 1)
            acc = 0
    if starts[-1] != len(rx):
        starts.append(len(rx))
    for s0, s1 in zip(starts[:-1], starts[1:]):
        off = (csum[s0 - 1] if s0 > 0 else 0)
        n = int(npx[s0:s1].sum().item())
        gsel = torch.arange(s0, s1, device=DEV)
        within = torch.arange(n, device=DEV)
        within = within - torch.repeat_interleave((csum[s0:s1] - npx[s0:s1] - off), npx[s0:s1])
        # within is now 0..npx_i-1 per gaussian
        bwi = bw[s0:s1]
        bhi = bh[s0:s1]
        rep = torch.repeat_interleave(torch.arange(s1 - s0, device=DEV), npx[s0:s1])
        wy = within // bwi[rep]
        wx = within % bwi[rep]
        uu = g['u'][s0:s1][rep] - rx[s0:s1][rep] + wx
        vv = g['v'][s0:s1][rep] - ry[s0:s1][rep] + wy
        u_i = torch.floor(uu).long()
        v_i = torch.floor(vv).long()
        inb = (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)
        px = v_i * w + u_i
        dx = uu - g['u'][s0:s1][rep]
        dy = vv - g['v'][s0:s1][rep]
        a = (g['c00'][s0:s1][rep] * dx * dx
             + 2 * g['c01'][s0:s1][rep] * dx * dy
             + g['c11'][s0:s1][rep] * dy * dy)
        alpha = g['o'][s0:s1][rep] * torch.exp(-0.5 * torch.clamp(a, min=0))
        alpha = torch.clamp(alpha, 0, 0.999999)
        contrib = torch.log1p(-alpha)
        logbuf.index_add_(0, px[inb], contrib[inb])
    amap = 1.0 - torch.exp(logbuf)
    return amap.view(h, w).cpu().numpy().astype(np.float32)


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
        g = project(xyz, cov3d, opacity, cam, vw['q'], vw['t'], w, h)
        amap = render_alpha_gpu(g, w, h)
        m = np.asarray(Image_open(masks, vw['stem']).convert('L')).astype(np.float32) / 255.0
        if m.shape != (h, w):
            m = np.asarray(Image_open(masks, vw['stem']).convert('L')
                           .resize((w, h))).astype(np.float32) / 255.0
        outside = amap[m < 0.5]
        per[vw['stem']] = {'L_outside': round(float(outside.mean()), 5),
                           'alpha_max_out': round(float(outside.max()), 4)}
        print(vw['stem'], per[vw['stem']], flush=True)
    Ls = [d['L_outside'] for d in per.values()]
    rep = {'ply': str(ply), 'n': len(xyz), 'n_views': len(per),
           'L_outside_mean': round(float(np.mean(Ls)), 5), 'per_view': per}
    out_json.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != 'per_view'}, indent=2), flush=True)
    return 0


def Image_open(masks: Path, stem: str):
    from PIL import Image
    return Image.open(masks / f'{stem}.png')


if __name__ == '__main__':
    sys.exit(main())
