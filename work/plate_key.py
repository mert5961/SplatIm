"""D1+D2: local green-screen plate + chroma-distance key (independent evidence).

D1: certain-screen pixels = outside dilated SAM mask AND RVM alpha<0.02
    AND green-chroma likely (G>max(R,B)). Fit per-channel degree-2
    polynomial surface -> plate B(x,y) + validity + fit error.
D2: CbCr Mahalanobis vs local screen stats from the same certain pixels,
    soft key via smoothstep(d0,d1) from measured quantiles; low confidence
    when Y<Ymin (black polo). Output is EVIDENCE, not the matte.

Runs on the B5 60-view RGBA frames (RGB used, alpha ignored except for
the certain-screen gate). Color metadata: h264 yuv420p tv bt709, no
transfer/primaries tags -> BT.709 assumption logged in manifest.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
B5_IMG = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60' / 'dataset' / 'images'
SAM_DIR = ROOT / 'work' / 'research' / 'SAM21-60views'
RVM_MASKS = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c' / 'dataset' / 'masks'
B5_SEL = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60' / 'selection.json'
OUT = ROOT / 'work' / 'research' / 'PLATE-KEY-60views'
YMIN = 0.06


def poly2_fit(H, W, ys, xs, vals):
    xn = (xs / max(W - 1, 1)) * 2 - 1
    yn = (ys / max(H - 1, 1)) * 2 - 1
    A = np.stack([np.ones_like(xn), xn, yn, xn ** 2, xn * yn, yn ** 2], 1)
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    B = (np.stack([np.ones_like(gx), gx, gy, gx ** 2, gx * gy, gy ** 2], -1) * coef).sum(-1)
    pred = (A * coef).sum(-1)
    return B, float(np.sqrt(np.mean((pred - vals) ** 2)))


def main() -> int:
    from PIL import Image
    from scipy.ndimage import binary_dilation
    OUT.mkdir(parents=True, exist_ok=True)
    kept = json.loads(B5_SEL.read_text())['kept']
    sam_files = sorted(SAM_DIR.glob('mask_*.npy'))
    assert len(sam_files) == 60, len(sam_files)

    rows = []
    t0 = time.time()
    for i, stem in enumerate(sorted(kept)):
        rgb = np.asarray(Image.open(B5_IMG / f'{stem}.png').convert('RGB')).astype(np.float32) / 255.0
        H, W, _ = rgb.shape
        sam = np.load(sam_files[i])
        sam = sam[0] if sam.ndim == 3 else sam
        if sam.shape != (H, W):
            sam = np.asarray(Image.fromarray((sam > 0).astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
        else:
            sam = sam > 0
        rvm = np.asarray(Image.open(RVM_MASKS / f'{stem}.png').convert('L')).astype(np.float32) / 255.0
        if rvm.shape != (H, W):
            rvm = np.asarray(Image.open(RVM_MASKS / f'{stem}.png').convert('L').resize((W, H))).astype(np.float32) / 255.0
        R, G, Bch = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        green_likely = (G > np.maximum(R, Bch)) & (G > 0.15)
        certain = (~binary_dilation(sam, iterations=7)) & (rvm < 0.02) & green_likely
        frac = float(certain.mean())
        plate, err, key = None, None, None
        if certain.sum() > 500:
            ys, xs = np.nonzero(certain)
            step = max(1, len(xs) // 20000)
            ys, xs = ys[::step], xs[::step]
            ch, errs = [], []
            for c in range(3):
                Bc, e = poly2_fit(H, W, ys, xs, rgb[ys, xs, c])
                ch.append(Bc)
                errs.append(e)
            plate = np.stack(ch, -1).clip(0, 1)
            err = float(np.mean(errs))
            # chroma stats on certain pixels (CbCr approx via 709 matrix)
            Cb = -0.168736 * R + -0.331264 * G + 0.5 * Bch
            Cr = 0.5 * R + -0.418688 * G + -0.081312 * Bch
            z = np.stack([Cb[certain], Cr[certain]], 1)
            mu = z.mean(0)
            cov = np.cov(z.T) + 1e-6 * np.eye(2)
            inv = np.linalg.inv(cov)
            Z = np.stack([Cb.ravel(), Cr.ravel()], 1) - mu
            d = np.sqrt(np.einsum('ij,jk,ik->i', Z, inv, Z)).reshape(H, W)
            d_bg = d[certain]
            d_fg = d[(rvm > 0.95)]
            d0 = float(np.percentile(d_bg, 95)) if d_bg.size else 1.0
            d1 = float(np.percentile(d_fg, 5)) if d_fg.size else 5.0
            if d1 <= d0:
                d1 = d0 + 1.0
            t = np.clip((d - d0) / max(d1 - d0, 1e-6), 0, 1)
            key = 1.0 - (t * t * (3 - 2 * t))  # 1=screen, smoothstep
            Y = 0.2126 * R + 0.7152 * G + 0.0722 * Bch
            conf = np.clip((Y - YMIN) / 0.1, 0, 1)
            key = key * conf + 0.5 * (1 - conf)  # uncertain in darks
            Image.fromarray((key * 255).astype(np.uint8)).save(OUT / f'key_{stem}.png')
        rows.append({'frame': stem, 'certain_frac': round(frac, 4),
                     'fit_rmse': round(err, 4) if err is not None else None,
                     'key_mean': round(float(key.mean()), 4) if key is not None else None})
        if i % 20 == 0:
            print(f'{i}/60 certain={frac:.2f} rmse={err}', flush=True)
    rep = {'frames': len(rows), 'seconds': round(time.time() - t0, 1),
           'color_assumption': 'BT.709 (h264 yuv420p tv bt709, no transfer tags)',
           'frames_ok': sum(1 for r in rows if r['fit_rmse'] is not None),
           'mean_certain_frac': float(np.mean([r['certain_frac'] for r in rows])),
           'mean_fit_rmse': float(np.mean([r['fit_rmse'] for r in rows if r['fit_rmse'] is not None])),
           'rows': rows}
    (OUT / 'report.json').write_text(json.dumps(rep, indent=2))
    print(f"ok={rep['frames_ok']}/60 certain={rep['mean_certain_frac']:.2f} rmse={rep['mean_fit_rmse']:.4f}",
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
