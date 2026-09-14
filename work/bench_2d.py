"""E2: 2D benchmark RVM vs MatAnyone2 vs Sapiens2 vs chroma-key.

Metrics per §14 on the trimap set (12 frames): core FG miss, certain BG
leakage, boundary temporal MAE (Farneback-warped alpha residual on the
edge band), semantic IoU, boundary F-score, pairwise disagreement,
green-screen contradiction (alpha where key>0.9 & SAM==0), edge spill
score, matte entropy. Picks the primary matte by gate order, no averaging.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
RVM_MASKS = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c' / 'dataset' / 'masks'
MA2_DIR = ROOT / 'work' / 'research' / 'MA2-60views' / 'frames_jpg_person60' / 'pha'
SAP_DIR = ROOT / 'work' / 'research' / 'SAPIENS2-60views' / 'vis'
KEY_DIR = ROOT / 'work' / 'research' / 'PLATE-KEY-60views'
SAM_DIR = ROOT / 'work' / 'research' / 'SAM21-60views'
HYB_DIR = ROOT / 'work' / 'research' / 'HYBRID-60views'
B5_SEL = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60' / 'selection.json'
B5_IMG = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60' / 'dataset' / 'images'
OUT = ROOT / 'work' / 'research' / 'BENCH-2D.json'


def edge_band(a, lo=0.05, hi=0.95):
    return (a > lo) & (a < hi)


def main() -> int:
    from PIL import Image
    from scipy.ndimage import binary_erosion, binary_dilation
    import cv2
    kept = json.loads(B5_SEL.read_text())['kept']
    kept_sorted = sorted(kept)
    tri_files = [p for p in sorted((ROOT / 'work' / 'research' / 'trimap-set').glob('*.png'))
                 if p.name != 'manifest.json' and p.stem in kept]
    ma2_files = sorted(MA2_DIR.glob('*.png'))
    sap_all = sorted(SAP_DIR.rglob('*_alpha.npy'))
    sam_files = sorted(SAM_DIR.glob('mask_*.npy'))

    def load_all(stem, idx):
        tri = np.asarray(Image.open(ROOT / 'work' / 'research' / 'trimap-set' / f'{stem}.png'))
        H, W = tri.shape
        def fit(a):
            a = np.asarray(a, dtype=np.float32)
            if a.max() > 1.0:
                a = a / 255.0
            if a.shape != (H, W):
                a = np.asarray(Image.fromarray(a).resize((W, H))).astype(np.float32)
                if a.max() > 1.0:
                    a = a / 255.0
            return np.clip(a, 0, 1)
        rvm = fit(Image.open(RVM_MASKS / f'{stem}.png').convert('L'))
        ma2 = fit(Image.open(ma2_files[idx]).convert('L'))
        sap = fit(np.load(str(sap_all[idx])))
        key_p = KEY_DIR / f'key_{stem}.png'
        key = fit(Image.open(key_p).convert('L')) if key_p.exists() else np.zeros((H, W), np.float32)
        hyb = fit(Image.open(HYB_DIR / f'hybrid_{stem}.png').convert('L'))
        sam = np.load(sam_files[idx])
        sam = sam[0] if sam.ndim == 3 else sam
        if sam.shape != (H, W):
            sam = np.asarray(Image.fromarray((sam > 0).astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
        else:
            sam = sam > 0
        return tri, {'rvm': rvm, 'ma2': ma2, 'sapiens': sap, 'key': key, 'hybrid': hyb}, sam

    acc = {k: {'miss': [], 'leak': [], 'iou': [], 'bf': [], 'contra': [], 'spill': [], 'ent': []}
           for k in ('rvm', 'ma2', 'sapiens', 'key', 'hybrid')}
    disag, tma = [], []
    for stem in [p.stem for p in tri_files]:
        idx = kept_sorted.index(stem)
        tri, C, sam = load_all(stem, idx)
        fg, bg = tri == 255, tri == 0
        ea_tri = np.logical_xor(tri > 0, binary_erosion(tri > 0)) if (tri > 0).any() else np.zeros_like(fg)
        for k, a in C.items():
            aa = 1.0 - a if k == 'key' else a  # key=1 means screen
            acc[k]['miss'].append(float(1 - aa[fg].mean()) if fg.any() else 0.0)
            acc[k]['leak'].append(float(aa[bg].mean()) if bg.any() else 0.0)
            acc[k]['iou'].append(float(((aa > 0.5) & (tri > 0)).sum() / max(((aa > 0.5) | (tri > 0)).sum(), 1)))
            ea = np.logical_xor(aa > 0.5, binary_erosion(aa > 0.5)) if (aa > 0.5).any() else np.zeros_like(fg)
            if ea.any() and ea_tri.any():
                pa = np.logical_and(ea, binary_dilation(ea_tri, iterations=2)).sum() / ea.sum()
                ra = np.logical_and(ea_tri, binary_dilation(ea, iterations=2)).sum() / ea_tri.sum()
                acc[k]['bf'].append(float(2 * pa * ra / max(pa + ra, 1e-9)))
            acc[k]['contra'].append(float(aa[(key_gt(C, sam))].mean()) if (key_gt(C, sam)).any() else 0.0)
            eb = edge_band(aa)
            rgb = np.asarray(Image.open(B5_IMG / f'{stem}.png').convert('RGB')).astype(np.float32) / 255.0
            if rgb.shape[:2] != aa.shape:
                rgb = np.asarray(Image.open(B5_IMG / f'{stem}.png').convert('RGB').resize((aa.shape[1], aa.shape[0]))).astype(np.float32) / 255.0
            ex = np.clip(rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]), 0, 1)[eb]
            acc[k]['spill'].append(float(ex.mean()) if eb.any() else 0.0)
            h, _ = np.histogram(aa, bins=16, range=(0, 1), density=True)
            h = h[h > 0] / h[h > 0].sum()
            acc[k]['ent'].append(float(-(h * np.log(h)).sum()))
        disag.append(float(np.abs(C['ma2'] - C['sapiens']).mean()))
        # temporal MAE on edge band via flow warp to next kept frame
        j = kept_sorted.index(stem)
        if j + 1 < len(kept_sorted):
            g0 = cv2.imread(str(B5_IMG / f'{stem}.png'), cv2.IMREAD_GRAYSCALE)
            g1 = cv2.imread(str(B5_IMG / f'{kept_sorted[j+1]}.png'), cv2.IMREAD_GRAYSCALE)
            if g0 is not None and g1 is not None and g0.shape == g1.shape:
                flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                H, W = g0.shape
                gx, gy = np.meshgrid(np.arange(W), np.arange(H))
                mx = np.clip(gx + flow[..., 0], 0, W - 1).astype(np.float32)
                my = np.clip(gy + flow[..., 1], 0, H - 1).astype(np.float32)
                warped = cv2.remap(C['ma2'].astype(np.float32), mx, my, cv2.INTER_LINEAR)
                ma2n = C['ma2']
                eb = edge_band(ma2n)
                tma.append(float(np.abs(warped[eb] - ma2n[eb]).mean()) if eb.any() else 0.0)
    rep = {'frames': len(tri_files), 'models': {}}
    for k, v in acc.items():
        rep['models'][k] = {m: round(float(np.mean(x)), 5) if x else None for m, x in v.items()}
    rep['ma2_sapiens_disagreement'] = round(float(np.mean(disag)), 5)
    rep['ma2_temporal_edge_mae'] = round(float(np.mean(tma)), 5) if tma else None
    OUT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)
    return 0


def key_gt(C, sam):
    from scipy.ndimage import binary_dilation
    return (C['key'] > 0.9) & (~binary_dilation(sam, iterations=7))


if __name__ == '__main__':
    sys.exit(main())
