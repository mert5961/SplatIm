"""Harden RVM soft mattes into binary person masks, then retrain.

Brush's mask mode multiplies per-pixel loss by the mask value, so RVM's
soft alpha (non-zero background floor, semi-transparent edges) leaks the
environment into the splat. This script copies the person job's dataset,
replaces each mask with a binary version:
  threshold (>= 96) -> largest connected component -> 2px gaussian feather
then trains Brush on the cleaned dataset.
"""
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
SOURCE_JOB = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c'


def main():
    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-clean-{uuid.uuid4().hex[:6]}'
    (job).mkdir(parents=True)
    shutil.copytree(SOURCE_JOB / 'dataset', job / 'dataset')

    masks = sorted((job / 'dataset' / 'masks').glob('*.png'))
    removed_total = 0.0
    for i, m in enumerate(masks):
        a = np.asarray(Image.open(m))
        binm = a >= 96
        pre = int(binm.sum())
        lab, n = ndimage.label(binm)
        if n > 1:
            sizes = ndimage.sum(binm, lab, index=list(range(1, n + 1)))
            binm = lab == (int(np.argmax(sizes)) + 1)
            removed_total += float(1 - binm.sum() / max(pre, 1))
        Image.fromarray((binm * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(2)).save(m)
        if i % 20 == 0:
            print(f'{i}/{len(masks)} hardened', flush=True)

    print(f'{len(masks)} masks hardened, stray foreground removed: {removed_total:.4f} (mean frac)', flush=True)
    print('JOB', job, flush=True)


if __name__ == '__main__':
    main()
