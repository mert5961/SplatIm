"""Inspect per-frame masks: how many blobs, is the background woman included?"""
from pathlib import Path

import numpy as np
from PIL import Image

masks = sorted(Path(r'C:/Users/reality/Documents/Codex/2026-09-11/i-w/outputs/multi-view-scenes/20260914-084036-person-57b59c/dataset/masks').glob('*.png'))
print(len(masks), 'masks')

rows = []
for p in masks:
    a = np.asarray(Image.open(p))
    binm = a >= 128
    small = binm[::8, ::8]
    H, W = small.shape
    lab = np.zeros((H, W), int)
    n = 0
    sizes = []
    for y in range(H):
        for x in range(W):
            if small[y, x] and not lab[y, x]:
                n += 1
                stack = [(y, x)]
                lab[y, x] = n
                cnt = 0
                while stack:
                    cy, cx = stack.pop()
                    cnt += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and small[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = n
                            stack.append((ny, nx))
                sizes.append(cnt)
    if sizes:
        tot = sum(sizes)
        big = max(sizes) / tot
        sig = sum(1 for s in sizes if s > 0.002 * tot)
    else:
        big, sig = 0.0, 0
    rows.append((p.stem, float(a.mean()) / 255, sig, float(big)))

rows.sort(key=lambda r: r[3])
print('stem  mean_alpha  n_sig_blobs  largest_frac')
for r in rows[:15]:
    print(f'{r[0]}  {r[1]:.2f}  {r[2]}  {r[3]:.3f}')
multi = sum(1 for r in rows if r[2] > 1)
print('frames with >1 significant blob:', multi)
lowbig = [r[0] for r in rows if r[3] < 0.9]
print('frames where largest blob <90% of foreground:', lowbig)
