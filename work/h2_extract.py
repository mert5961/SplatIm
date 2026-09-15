"""H2: SAGA-style mask-guided object extraction from the full-scene PLY.

For every Gaussian, in every view (depth-ordered TBB), compute the
transmittance T_i reaching the splat at its projected pixel. Visible
splats (T_i > t_vis) accumulate evidence from the RVM mask:

  fg_i = sum_v T_i * mask(u_i,v_i)
  bg_i = sum_v T_i * (1 - mask(u_i,v_i))

Occluded background splats (behind the person) have T_i ~ 0 and get no
evidence — they do NOT inherit the person's silhouette.

Classification (user spec):
  keep      : fg_i >= 5 and fg_i/(fg_i+bg_i) >= 0.5   (object material,
              even at low opacity)
  hide      : fg_i < 2                                 (clearly background)
  uncertain : otherwise                                (boundary, kept for
              review)

Exports keep.ply / hide.ply / uncertain.ply + H2-extract.json.

Usage: python h2_extract.py [ply]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
import splat_alpha_render as g1

SPARSE = ROOT / 'outputs/multi-view-scenes/20260914-151822-person-evalsplit/sparse_txt_full'
MASKS = ROOT / 'outputs/multi-view-scenes/20260914-084036-person-57b59c/dataset/masks'
OUT = ROOT / 'work/research/H2-extract'
T_VIS = 0.15
CHUNK = 4096
FG_MIN = 5.0
FG_KEEP = 2.0


def render_geometry(xyz, cov3d, cam, q, t, w, h):
    """Per-splat projection + 2D weight, depth-sorted. Returns (order, u, v, o, conics)."""
    R = g1.quat_to_mat(q[None, :])[0]
    cam_pts = xyz @ R.T + t[None, :]
    z = cam_pts[:, 2]
    front = z > 0.2
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    u = fx * cam_pts[:, 0] / np.maximum(z, 1e-8) + cx
    v = fy * cam_pts[:, 1] / np.maximum(z, 1e-8) + cy

    idx = np.nonzero(front)[0]
    inb = (u[idx] >= 0) & (u[idx] < w) & (v[idx] >= 0) & (v[idx] < h)
    idx = idx[inb]
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
    conic00 = c11 / det
    conic01 = -c01 / det
    conic11 = c00 / det
    mid = 0.5 * (c00 + c11)
    disc = np.sqrt(np.maximum(0.25 * (c00 - c11) ** 2 + c01 * c01, 0))
    radius = np.clip(np.ceil(3.0 * np.sqrt(mid + disc)).astype(np.int32), 2, 256)
    srt = np.argsort(z[idx], kind='stable')
    return (idx[srt], u[idx][srt], v[idx][srt], radius[srt],
            (conic00[srt], conic01[srt], conic11[srt]))


def main():
    args = list(sys.argv[1:])
    def take(flag, default):
       if flag in args:
           i = args.index(flag)
           val = args[i + 1]
           del args[i:i + 2]
           return val
    def _g(name):
        return globals()[name]
    REPORT = Path(take('--report', str(ROOT / 'work/research/H2-extract.json')))
    SPARSE = Path(take('--sparse', str(_g('SPARSE'))))
    MASKS = Path(take('--masks', str(_g('MASKS'))))
    OUT = Path(take('--out', str(_g('OUT'))))
    ply = Path(args[0]) if args else \
       ROOT / 'outputs/multi-view-scenes/20260914-080603-07ed0a/scene.ply'
    OUT.mkdir(parents=True, exist_ok=True)

    v = PlyData.read(str(ply))['vertex']
    xyz, scale, rot, opacity = g1.load_ply(ply)
    cov3d = g1.gauss_cov3d(scale, rot)
    n = len(xyz)
    print(f'ply: {n} gaussians', flush=True)

    cams, views = g1.load_cameras(SPARSE)
    fg = np.zeros(n, np.float32)
    bg = np.zeros(n, np.float32)

    for vi, vw in enumerate(views):
        cam = cams[vw['cam_id']]
        w, h = cam['w'], cam['h']
        m = np.asarray(Image.open(MASKS / f"{vw['stem']}.png").convert('L')).astype(np.float32) / 255.0
        if m.shape != (h, w):
            m = np.asarray(Image.open(MASKS / f"{vw['stem']}.png").convert('L')
                           .resize((w, h))).astype(np.float32) / 255.0
        order, u, v_, radius, (c00, c01, c11) = \
            render_geometry(xyz, cov3d, cam, vw['q'], vw['t'], w, h)
        o = opacity[order]
        uu, vv = u, v_  # already depth-sorted, aligned to order
        A = np.zeros((h, w), np.float32)  # accumulated alpha (1 - transmittance)
        for s in range(0, len(order), CHUNK):
            ks = slice(s, min(s + CHUNK, len(order)))
            n_c = min(s + CHUNK, len(order)) - s
            si = order[ks]
            # T before this chunk at each splat's center pixel
            pu = np.clip(uu[ks].round().astype(np.int32), 0, w - 1)
            pv = np.clip(vv[ks].round().astype(np.int32), 0, h - 1)
            T = (1.0 - A[pv, pu]).astype(np.float32)
            vis = T > T_VIS
            if vis.any():
                mval = m[pv[vis], pu[vis]]
                fg[si[vis]] += (T[vis] * mval).astype(np.float32)
                bg[si[vis]] += (T[vis] * (1.0 - mval)).astype(np.float32)
            # update A: splats in this chunk
            c00c, c01c, c11c, oc, uuc, vvc, r = c00[ks], c01[ks], c11[ks], o[ks], uu[ks], vv[ks], radius[ks]
            TILE = 32
            nt_x = (w + TILE - 1) // TILE
            nt_y = (h + TILE - 1) // TILE
            t0x = np.clip(((uuc - r) // TILE).astype(np.int32), 0, nt_x - 1)
            t1x = np.clip(((uuc + r) // TILE).astype(np.int32), 0, nt_x - 1)
            t0y = np.clip(((vvc - r) // TILE).astype(np.int32), 0, nt_y - 1)
            t1y = np.clip(((vvc + r) // TILE).astype(np.int32), 0, nt_y - 1)
            gy = t1y - t0y + 1
            gx = t1x - t0x + 1
            pc = gy * gx
            total = int(pc.sum())
            if total == 0:
                continue
            gidx = np.repeat(np.arange(n_c), pc)
            offs = np.repeat(np.cumsum(pc) - pc, pc)
            within = np.arange(total) - offs
            ny = gy[gidx]
            ty = (t0y[gidx] + within % ny).astype(np.int32)
            tx = (t0x[gidx] + within // ny).astype(np.int32)
            tkey = ty * nt_x + tx
            ord2 = np.argsort(tkey, kind='stable')
            ks_all = gidx[ord2]
            tkey_s = tkey[ord2]
            bounds = np.nonzero(np.diff(tkey_s))[0] + 1
            starts = np.concatenate([[0], bounds])
            ends = np.concatenate([bounds, [total]])
            for s2, e2 in zip(starts, ends):
                kk = ks_all[s2:e2]
                tyx, txx = divmod(int(tkey_s[s2]), nt_x)
                px0, py0 = txx * TILE, tyx * TILE
                px1, py1 = min(px0 + TILE, w), min(py0 + TILE, h)
                gxg = np.arange(px0, px1, dtype=np.float32)[None, :]
                gyg = np.arange(py0, py1, dtype=np.float32)[:, None]
                dx = gxg - uuc[kk][:, None, None]
                dy = gyg - vvc[kk][:, None, None]
                a = (c00c[kk][:, None, None] * dx * dx
                     + 2 * c01c[kk][:, None, None] * dx * dy
                     + c11c[kk][:, None, None] * dy * dy)
                W = oc[kk][:, None, None] * np.exp(-0.5 * np.maximum(a, 0))
                W = np.clip(W, 0, 0.999999)
                A[py0:py1, px0:px1] = 1.0 - (1.0 - A[py0:py1, px0:px1]) * np.exp(np.log1p(-W).sum(axis=0))
        print(f'{vi+1}/{len(views)} {vw["stem"]} '
              f'fg>5: {int((fg > 5).sum())}', flush=True)

    denom = np.maximum(fg + bg, 1e-6)
    ffrac = fg / denom
    keep = (fg >= FG_MIN) & (ffrac >= 0.5)
    hide = fg < FG_KEEP
    unc = ~(keep | hide)
    np.savez_compressed(str(OUT / 'evidence.npz'), fg=fg, bg=bg,
                     ffrac=ffrac, opacity=opacity)

    def write_ply(mask_sel, name):
         from plyfile import PlyData as PD, PlyElement
         sel = np.nonzero(mask_sel)[0]
         if len(sel) == 0:
             return
         sub = np.ascontiguousarray(v.data[sel])
         el2 = PlyElement.describe(sub, 'vertex')
         PD([el2], text=False).write(str(OUT / name))
         print(name, len(sel), flush=True)

    write_ply(keep, 'keep.ply')
    write_ply(hide, 'hide.ply')
    write_ply(unc, 'uncertain.ply')

    rep = {'ply': str(ply), 'n': n, 't_vis': T_VIS,
           'n_keep': int(keep.sum()), 'n_hide': int(hide.sum()),
           'n_uncertain': int(unc.sum()),
           'fg_median': {
               'keep': round(float(np.median(fg[keep])) if keep.any() else 0, 2),
               'hide': round(float(np.median(fg[hide])) if hide.any() else 0, 2),
               'uncertain': round(float(np.median(fg[unc])) if unc.any() else 0, 2),
           },
           'opacity_mean': {
               'keep': round(float(opacity[keep].mean()) if keep.any() else 0, 3),
               'hide': round(float(opacity[hide].mean()) if hide.any() else 0, 3),
               'uncertain': round(float(opacity[unc].mean()) if unc.any() else 0, 3),
           }}
    (REPORT).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
