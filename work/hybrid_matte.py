"""E1: regional hybrid matte + uncertainty (no blind averaging).

Rules (Research02 §17):
  certain FG = SAM core & RVM>0.5 & MA2>0.5            -> alpha = max(MA2, RVM), ~1
  certain BG = ~dilate(SAM) & key>0.9 & MA2<0.1       -> alpha = 0
  unknown band = MA2 fractional                        -> alpha = MA2
  disagreement (SAM!=RVM core level)                   -> keep MA2, U=1
U = 0.5*|MA2-Sapiens| + 0.3*|MA2-RVM| + 0.2*key_conflict, clipped [0,1],
key_conflict = key*(1-SAM_dilated) outside person.
Outputs per frame: hybrid_<stem>.png (L), unc_<stem>.npy (float32).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
SAM_DIR = ROOT / 'work' / 'research' / 'SAM21-60views'
RVM_MASKS = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c' / 'dataset' / 'masks'
MA2_DIR = ROOT / 'work' / 'research' / 'MA2-60views' / 'frames_jpg_person60' / 'pha'
SAP_DIR = ROOT / 'work' / 'research' / 'SAPIENS2-60views' / 'vis'
KEY_DIR = ROOT / 'work' / 'research' / 'PLATE-KEY-60views'
B5_SEL = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60' / 'selection.json'
OUT = ROOT / 'work' / 'research' / 'HYBRID-60views'


def main() -> int:
    from PIL import Image
    from scipy.ndimage import binary_dilation
    OUT.mkdir(parents=True, exist_ok=True)
    kept = json.loads(B5_SEL.read_text())['kept']
    kept_sorted = sorted(kept)
    sam_files = sorted(SAM_DIR.glob('mask_*.npy'))
    ma2_files = sorted(MA2_DIR.glob('*.png'))
    sap_files = {p.name.split('_alpha')[0]: p for p in SAP_DIR.rglob('*_alpha.npy')}
    assert len(sam_files) == 60 and len(ma2_files) == 60, (len(sam_files), len(ma2_files))

    rows = []
    t0 = time.time()
    for i, stem in enumerate(kept_sorted):
        H_W = None
        sam = np.load(sam_files[i])
        sam = sam[0] if sam.ndim == 3 else sam
        sam = sam > 0
        rvm = np.asarray(Image.open(RVM_MASKS / f'{stem}.png').convert('L')).astype(np.float32) / 255.0
        H, W = rvm.shape
        if sam.shape != (H, W):
            sam = np.asarray(Image.fromarray(sam.astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
        ma2 = np.asarray(Image.open(ma2_files[i]).convert('L')).astype(np.float32) / 255.0
        if ma2.shape != (H, W):
            ma2 = np.asarray(Image.open(ma2_files[i]).convert('L').resize((W, H))).astype(np.float32) / 255.0
        sap_cands = [p for k, p in sap_files.items() if True]
        # positional map: vis outputs follow sorted-jpg order == kept_sorted order
        sap = np.load(str(sorted(SAP_DIR.rglob('*_alpha.npy'))[i])).astype(np.float32)
        sap = np.clip(sap, 0, 1)
        if sap.shape != (H, W):
            sap = np.asarray(Image.fromarray(sap).resize((W, H))).astype(np.float32)
        key_p = KEY_DIR / f'key_{stem}.png'
        key = np.asarray(Image.open(key_p).convert('L')).astype(np.float32) / 255.0 if key_p.exists() else np.zeros((H, W), np.float32)

        hybrid = ma2.copy()
        core = sam & (rvm > 0.5) & (ma2 > 0.5)
        hybrid[core] = np.maximum(ma2[core], rvm[core])
        bg = (~binary_dilation(sam, iterations=7)) & (key > 0.9) & (ma2 < 0.1)
        hybrid[bg] = 0.0
        disag = (sam != (rvm > 0.5))
        key_conflict = key * (~binary_dilation(sam, iterations=3)).astype(np.float32)
        U = (0.5 * np.abs(ma2 - sap) + 0.3 * np.abs(ma2 - rvm) + 0.2 * key_conflict).clip(0, 1)
        U[disag] = np.maximum(U[disag], 0.75)
        Image.fromarray((hybrid * 255).astype(np.uint8)).save(OUT / f'hybrid_{stem}.png')
        np.save(OUT / f'unc_{stem}.npy', U.astype(np.float32))
        rows.append({'frame': stem, 'u_mean': round(float(U.mean()), 4),
                     'u_p90': round(float(np.percentile(U, 90)), 4),
                     'core_frac': round(float(core.mean()), 4),
                     'bg_frac': round(float(bg.mean()), 4)})
        if i % 20 == 0:
            print(f'{i}/60 Umean={U.mean():.3f}', flush=True)
    rep = {'frames': len(rows), 'seconds': round(time.time() - t0, 1),
           'mean_U': float(np.mean([r['u_mean'] for r in rows])),
           'mean_U_p90': float(np.mean([r['u_p90'] for r in rows])),
           'rows': rows}
    (OUT / 'report.json').write_text(json.dumps(rep, indent=2))
    print(f'done U={rep["mean_U"]:.3f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
