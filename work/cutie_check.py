"""B3 (EPIC B): Cutie cross-check — forward+reverse VOS, swap detection.

Seeds frame 0 from the SAM2.1 mask (target identity established by B1),
propagates forward AND reverse over the B5 60-view jpgs. Reports
disagreement frames (IoU<0.9) and centroid jumps. Cutie weights:
work/models/cutie/cutie-base-mega.pth (public release, MIT code).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
sys.path.insert(0, str(ROOT / 'work' / 'cutierepo'))
STAGE = ROOT / 'work' / 'research' / 'SAM21-60views' / 'frames_jpg'
SAM0 = ROOT / 'work' / 'research' / 'SAM21-60views' / 'mask_00000.npy'
WEIGHTS = ROOT / 'work' / 'models' / 'cutie' / 'cutie-base-mega.pth'
OUT = ROOT / 'work' / 'research' / 'CUTIE-60views'


def load_frames():
    from PIL import Image
    files = sorted(STAGE.glob('*.jpg'))
    assert len(files) == 60, len(files)
    imgs = []
    for p in files:
        a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(a).permute(2, 0, 1))
    return imgs


def build_core():
    import os
    import torch
    os.chdir(ROOT / 'work' / 'cutierepo')
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    from cutie.utils.get_default_model import get_default_model
    from cutie.inference.inference_core import InferenceCore
    cutie = get_default_model()
    cutie.load_weights(torch.load(str(WEIGHTS), map_location='cpu'))
    cutie = cutie.cuda().float()
    return InferenceCore(cutie, cutie.cfg)


def run_pass(core, imgs, seed, reverse=False):
    order = range(len(imgs) - 1, -1, -1) if reverse else range(len(imgs))
    masks = {}
    with torch.no_grad():
        for k, i in enumerate(order):
            if (not reverse and i == 0) or (reverse and i == len(imgs) - 1):
                m = torch.from_numpy(seed.astype(np.int64)).cuda()
                out = core.step(imgs[i].cuda(), m, objects=[1])
            else:
                out = core.step(imgs[i].cuda())
            prob = core.output_prob_to_mask(out).cpu().numpy()
            prob = prob[0] if prob.ndim == 3 else prob
            masks[i] = (prob > 0.5).astype(np.uint8)
    return masks


def main() -> int:
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = load_frames()
    seed = np.load(SAM0)
    seed = seed[0] if seed.ndim == 3 else seed
    seed = (seed > 0).astype(np.uint8)
    _, H, W = imgs[0].shape
    if seed.shape != (H, W):
        seed = np.asarray(Image.fromarray(seed).resize((W, H), Image.NEAREST)).astype(np.int64)
    else:
        seed = seed.astype(np.int64)
    t0 = time.time()
    fwd = run_pass(build_core(), imgs, seed, reverse=False)
    rev = run_pass(build_core(), imgs, seed, reverse=True)
    dt = time.time() - t0
    ious, jumps, prev_cx = [], [], None
    for i in range(len(imgs)):
        a, b = fwd[i] > 0, rev[i] > 0
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        ious.append(float(inter / max(union, 1)))
        ys, xs = np.nonzero(a)
        cx = float(xs.mean()) if len(xs) else -1.0
        jumps.append(prev_cx is not None and abs(cx - prev_cx) > 150)
        prev_cx = cx
        if i % 20 == 0:
            np.save(OUT / f'cutie_fwd_{i:05d}.npy', fwd[i])
    rep = {'frames': len(imgs), 'seconds': round(dt, 1),
           'mean_fwd_rev_iou': float(np.mean(ious)),
           'min_iou': float(np.min(ious)),
           'disagree_frames': [i for i, v in enumerate(ious) if v < 0.9],
           'centroid_jumps': [i for i, j in enumerate(jumps) if j]}
    (OUT / 'report.json').write_text(json.dumps(rep, indent=2))
    print(f'done {dt:.0f}s mean_iou={rep["mean_fwd_rev_iou"]:.3f} '
          f'disagree={rep["disagree_frames"]} jumps={rep["centroid_jumps"]}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
