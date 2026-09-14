"""Core dE00: winner vs B2-source RGB on certain foreground (alpha>0.95).

Compares the EVAL-split train PLY's supervision RGB (B2 RGBA dataset) and
the despill-v3 foreground (FGUV3) against the B6-source RGB on confident
foreground pixels: median + p95 CIEDE2000. Gate: median<1, p95<3.
Uses colour-science (sRGB D65 -> Lab). No training involved.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality/Documents/Codex/2026-09-11/i-w'.replace('/', '\\'))
B2_IMG = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-124953-person-b2-rgba30k' / 'dataset' / 'images'
FGU_DIR = ROOT / 'work' / 'research' / 'FGUV3-60views'
B6_IMG = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-132825-person-b6-despillv2' / 'dataset' / 'images'
B5_SEL = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60' / 'selection.json'
OUT = ROOT / 'work' / 'research' / 'DE00-core.json'


def main() -> int:
    from PIL import Image
    import colour
    kept = json.loads(B5_SEL.read_text())['kept']
    d_v3, d_b2 = [], []
    n = 0
    for stem in sorted(kept):
        b6p = B6_IMG / f'{stem}.png'
        if not b6p.exists():
            continue
        ref = np.asarray(Image.open(b6p).convert('RGB')).astype(np.float32) / 255.0
        a = np.asarray(Image.open(B2_IMG / f'{stem}.png').convert('RGBA')).astype(np.float32) / 255.0
        v3p = FGU_DIR / f'fg_{stem}.png'
        if not v3p.exists():
            continue
        v3 = np.asarray(Image.open(v3p).convert('RGBA')).astype(np.float32) / 255.0
        core = a[..., 3] > 0.95
        if core.sum() < 1000:
            continue
        lab_ref = colour.sRGB_to_XYZ(ref[core]) if False else None
        for tag, arr in (('b2', a[..., :3]), ('v3', v3[..., :3])):
            L1 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(ref[core], 0, 1)))
            L2 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(arr[core], 0, 1)))
            d = colour.delta_E(L1, L2, method='CIE 2000')
            (d_b2 if tag == 'b2' else d_v3).extend(float(x) for x in d[::37])
        n += 1
    rep = {'frames': n}
    for tag, d in (('b2_source_vs_b6', d_b2), ('despillv3_vs_b6', d_v3)):
        a = np.array(d)
        rep[tag] = {'n': len(a), 'median': round(float(np.median(a)), 4),
                    'p95': round(float(np.percentile(a, 95)), 4),
                    'mean': round(float(a.mean()), 4)}
    OUT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
