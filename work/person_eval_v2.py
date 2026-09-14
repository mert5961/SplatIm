"""A2: person_eval_v2 — stratified 48/12 split + rendered-alpha metrics.

Split: B5 60-view set in orbit (name) order, every 5th -> held-out
(12 views, evenly stratified over 360 deg), rest train (48).
Metrics operate on RENDERED RGBA (Brush --eval-save-to-disk output) vs
the soft matte: L_outside, IoU(A>0.5, M>0.5), boundary F-score.
Synthetic self-test proves the math before real renders exist.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
B5_JOB = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60'
SPLIT_OUT = ROOT / 'work' / 'research' / 'EVAL-60-split.json'


def build_split():
    sel = json.loads((B5_JOB / 'selection.json').read_text())['kept']
    ordered = sorted(sel)  # zero-padded orbit stems sort in orbit order
    held = set(ordered[4::5][:12])
    train = [s for s in ordered if s not in held]
    assert len(held) == 12 and len(train) == 48, (len(held), len(train))
    split = {'train': train, 'held_out': sorted(held), 'n_train': 48, 'n_held': 12}
    SPLIT_OUT.write_text(json.dumps(split, indent=2))
    return split


def l_outside(alpha_r, mask):
    a = np.asarray(alpha_r, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    return float((a * (1.0 - m)).sum() / (a.sum() + 1e-9))


def iou(alpha_r, mask, tau=0.5):
    a = np.asarray(alpha_r) > tau
    m = np.asarray(mask) > tau
    inter = np.logical_and(a, m).sum()
    union = np.logical_or(a, m).sum()
    return float(inter / max(union, 1))


def boundary_f(alpha_r, mask, tau=0.5, tol=2):
    from scipy.ndimage import binary_dilation, binary_erosion
    a = np.asarray(alpha_r) > tau
    m = np.asarray(mask) > tau
    ea = np.logical_xor(a, binary_erosion(a)) if a.any() else np.zeros_like(a)
    em = np.logical_xor(m, binary_erosion(m)) if m.any() else np.zeros_like(m)
    if not ea.any() or not em.any():
        return 0.0
    pa = np.logical_and(ea, binary_dilation(em, iterations=tol)).sum() / max(ea.sum(), 1)
    ra = np.logical_and(em, binary_dilation(ea, iterations=tol)).sum() / max(em.sum(), 1)
    return float(2 * pa * ra / max(pa + ra, 1e-9))


def synthetic_test():
    rng = np.random.default_rng(0)
    H, W = 200, 200
    yy, xx = np.mgrid[:H, :W]
    gt = (((xx - 100) ** 2 + (yy - 100) ** 2) < 60 ** 2).astype(float)
    bg = gt < 0.5
    leak = gt.copy()
    sel = np.zeros_like(gt, bool)
    sel[bg] = rng.random(bg.sum()) < 0.30
    leak[sel] = 0.9
    lo = l_outside(leak, gt)
    assert 0.05 < lo < 0.6, lo
    assert iou(leak, gt) < 0.999, (iou(leak, gt), 'leak must hurt IoU')
    f = boundary_f(leak, gt)
    assert 0.0 <= f <= 1.0, f
    # shrunken render: leakage 0 but recall lost -> IoU drops
    small = (((xx - 100) ** 2 + (yy - 100) ** 2) < 40 ** 2).astype(float)
    assert l_outside(small, gt) == 0.0
    assert iou(small, gt) < 0.6, 'shrinking must hurt IoU (anti leakage-gaming)'
    print(f'synthetic OK: leak L={lo:.3f} IoU={iou(leak,gt):.3f} F={f:.3f} '
          f'shrunk IoU={iou(small,gt):.3f}', flush=True)


def main():
    split = build_split()
    print(f'split: train={len(split["train"])} held={len(split["held_out"])} -> {SPLIT_OUT.name}',
          flush=True)
    synthetic_test()


if __name__ == '__main__':
    main()
