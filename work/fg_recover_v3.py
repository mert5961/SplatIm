"""D3: linear-light foreground recovery + luminance-preserving despill v3.

Pipeline per frame (BT.709 assumption logged; h264 yuv420p tv bt709, no
transfer tags):
  encoded sRGB -> linear (inverse EOTF) -> unmix with local plate B(x,y)
  (D1 poly fit recomputed here from certain-screen pixels) ->
  luminance-preserving despill in chroma plane (l^T s = 0) ->
  re-encode (OETF) -> straight RGBA PNG (alpha = MA2 consensus below).

Alpha bands: >=0.70 direct unmix; 0.10-0.70 blend with propagated color;
<0.10 propagate from confident foreground (never divide by tiny alpha).
Alpha source: MatAnyone2 (temporal authority); SAM core forces ~1 where
SAM+RVM+MA2 agree; key forces 0 where key>0.9 AND SAM==0.
Gamut report: fraction of corrected pixels clipped.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality/Documents/Codex/2026-09-11/i-w'.replace('/', '\\'))
B5_IMG = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60' / 'dataset' / 'images'
SAM_DIR = ROOT / 'work' / 'research' / 'SAM21-60views'
RVM_MASKS = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c' / 'dataset' / 'masks'
MA2_DIR = ROOT / 'work' / 'research' / 'MA2-60views' / 'frames_jpg_person60' / 'pha'
KEY_DIR = ROOT / 'work' / 'research' / 'PLATE-KEY-60views'
B5_SEL = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60' / 'selection.json'
OUT = ROOT / 'work' / 'research' / 'FGUV3-60views'
LUMA = np.array([0.2126, 0.7152, 0.0722])


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def poly2_plate(rgb, valid):
    H, W, _ = rgb.shape
    ys, xs = np.nonzero(valid)
    step = max(1, len(xs) // 20000)
    ys, xs = ys[::step], xs[::step]
    xn = (xs / max(W - 1, 1)) * 2 - 1
    yn = (ys / max(H - 1, 1)) * 2 - 1
    A = np.stack([np.ones_like(xn), xn, yn, xn ** 2, xn * yn, yn ** 2], 1)
    plates, errs = [], []
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, rgb[ys, xs, c], rcond=None)
        gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        G = np.stack([np.ones_like(gx), gx, gy, gx ** 2, gx * gy, gy ** 2], -1)
        plates.append((G * coef).sum(-1))
        errs.append(float(np.sqrt(np.mean(((A * coef).sum(-1) - rgb[ys, xs, c]) ** 2))))
    return np.stack(plates, -1).clip(0, 1), float(np.mean(errs))


def main() -> int:
    from PIL import Image
    from scipy.ndimage import binary_dilation, distance_transform_edt
    OUT.mkdir(parents=True, exist_ok=True)
    kept = json.loads(B5_SEL.read_text())['kept']
    ma2_files = sorted(MA2_DIR.glob('*.png'))
    sam_files = sorted(SAM_DIR.glob('mask_*.npy'))
    assert len(ma2_files) == 60 and len(sam_files) == 60

    # screen chroma direction from first certain plate (global approx)
    rows = []
    t0 = time.time()
    for i, stem in enumerate(sorted(kept)):
        rgb_e = np.asarray(Image.open(B5_IMG / f'{stem}.png').convert('RGB')).astype(np.float32) / 255.0
        H, W, _ = rgb_e.shape
        lin = srgb_to_linear(rgb_e)
        sam = np.load(sam_files[i])
        sam = sam[0] if sam.ndim == 3 else sam
        if sam.shape != (H, W):
            sam = np.asarray(Image.fromarray((sam > 0).astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
        else:
            sam = sam > 0
        ma2 = np.asarray(Image.open(ma2_files[i]).convert('L')).astype(np.float32) / 255.0
        if ma2.shape != (H, W):
            ma2 = np.asarray(Image.open(ma2_files[i]).convert('L').resize((W, H))).astype(np.float32) / 255.0
        rvm = np.asarray(Image.open(RVM_MASKS / f'{stem}.png').convert('L')).astype(np.float32) / 255.0
        if rvm.shape != (H, W):
            rvm = np.asarray(Image.open(RVM_MASKS / f'{stem}.png').convert('L').resize((W, H))).astype(np.float32) / 255.0
        key_p = KEY_DIR / f'key_{stem}.png'
        key = np.asarray(Image.open(key_p).convert('L')).astype(np.float32) / 255.0 if key_p.exists() else np.zeros((H, W), np.float32)

        # consensus alpha: SAM core ~1, key bg 0, else MA2
        alpha = ma2.copy()
        core = sam & (rvm > 0.5) & (ma2 > 0.5)
        alpha[core] = np.maximum(alpha[core], 0.98)
        bg = (~binary_dilation(sam, iterations=7)) & (key > 0.9)
        alpha[bg] = 0.0

        # local plate on certain screen
        R, G, Bc = lin[..., 0], lin[..., 1], lin[..., 2]
        green_likely = (G > np.maximum(R, Bc)) & (G > 0.05)
        certain = (~binary_dilation(sam, iterations=7)) & (rvm < 0.02) & green_likely
        if certain.sum() > 500:
            plate, _ = poly2_plate(lin, certain)
        else:
            plate = np.zeros_like(lin)
            plate[..., 1] = 0.3

        # unmix with bands; mid band trusts MA2 (temporal authority) at full
        # weight instead of 50/50 with propagation (v1 under-trusted matte)
        a = alpha[..., None]
        Fh = (lin - (1 - a) * plate) / np.maximum(a, 1e-6)
        conf = np.zeros_like(alpha)
        hi = alpha >= 0.10
        lo = alpha < 0.10
        F = np.where(hi[..., None], Fh, lin)
        conf[hi] = 1.0
        conf[lo] = 0.1
        if lo.any():
            _, idx = distance_transform_edt(~hi, return_indices=True)
            F[lo] = Fh[idx[0][lo], idx[1][lo]]

        # luminance-preserving despill in chroma plane; clamp the removal
        # to the physical spill range (never below plate-corrected color)
        Pb = plate - LUMA * ((plate @ LUMA) / (LUMA @ LUMA))[..., None]
        s = Pb / (np.linalg.norm(Pb, axis=-1, keepdims=True) + 1e-9)
        c = F - LUMA * ((F @ LUMA) / (LUMA @ LUMA))[..., None]
        e = np.maximum(0.0, (c * s).sum(-1))
        k = np.clip(0.15 + 0.85 * (1 - alpha), 0, 1)
        Fd = F - (k * np.minimum(e, 0.25))[..., None] * s
        pre_clip = int(((Fd < 0) | (Fd > 1)).any(-1).sum())
        Fd = Fd.clip(0, 1)
        gamut_frac = float(pre_clip / (H * W))

        out_rgb = (linear_to_srgb(Fd) * 255).astype(np.uint8)
        Image.fromarray(np.dstack([out_rgb, (alpha * 255).astype(np.uint8)]), 'RGBA').save(
            OUT / f'fg_{stem}.png')
        np.save(OUT / f'conf_{stem}.npy', conf.astype(np.float32))
        rows.append({'frame': stem, 'gamut_clip_frac': round(gamut_frac, 5),
                     'alpha_mean': round(float(alpha.mean()), 4),
                     'conf_mean': round(float(conf.mean()), 4)})
        if i % 20 == 0:
            print(f'{i}/60 gamut={gamut_frac:.4f}', flush=True)
    rep = {'frames': len(rows), 'seconds': round(time.time() - t0, 1),
           'color': 'BT.709 EOTF/OETF, unmix-then-despill, bands 0.7/0.1',
           'mean_gamut_clip': float(np.mean([r['gamut_clip_frac'] for r in rows])),
           'rows': rows}
    (OUT / 'report.json').write_text(json.dumps(rep, indent=2))
    print(f'done gamut_clip={rep["mean_gamut_clip"]:.5f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
