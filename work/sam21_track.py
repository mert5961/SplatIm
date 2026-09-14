"""B1 (EPIC B): SAM 2.1 target-identity tracking baseline.

Box prompt on the target (NOT text 'person': early distractor woman),
forward video propagation over the B5 60-view set. Saves raw bool masks
+ float32 logits where exposed. Identity-swap check: mask centroid jump
and area discontinuity flags per frame.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
B5_IMG = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-131958-person-b5-sel60' / 'dataset' / 'images'
CKPT = ROOT / 'work' / 'models' / 'sam2' / 'sam2.1_hiera_small.pt'
CFG = 'configs/sam2.1/sam2.1_hiera_s.yaml'
OUT = ROOT / 'work' / 'research' / 'SAM21-60views'
STAGE = OUT / 'frames_jpg'  # SAM2 video predictor only reads .jpg/.jpeg
# Box around the target in frame 0 (899x1600): person torso.
BOX = [10, 233, 776, 1599]

def main() -> int:
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor
    OUT.mkdir(parents=True, exist_ok=True)
    frames = sorted(B5_IMG.glob('*.png'))
    assert len(frames) == 60, len(frames)
    STAGE.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(frames):
        Image.open(p).convert('RGB').save(STAGE / f'{i:05d}.jpg', quality=95)
    print(f'{len(frames)} frames staged, box={BOX}', flush=True)

    predictor = build_sam2_video_predictor(CFG, str(CKPT), device='cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(STAGE))
        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1, box=np.array(BOX))
        np.save(OUT / 'logits_f00000.npy', out_mask_logits[0].cpu().numpy())
        areas, cx = [], []
        t0 = time.time()
        for fi, _, out_logits in predictor.propagate_in_video(state):
            m = (out_logits[0] > 0.0).cpu().numpy()
            np.save(OUT / f'mask_{fi:05d}.npy', m.astype(np.uint8))
            areas.append(float(m.sum()))
            ys, xs = np.nonzero(m[0] if m.ndim == 3 else m)
            cx.append(float(xs.mean()) if len(xs) else -1.0)
        dt = time.time() - t0
    areas = np.array(areas)
    med = float(np.median(areas))
    flags = [int(abs(a - med) / max(med, 1) > 0.35) for a in areas]
    jumps = [0] + [int(abs(cx[i] - cx[i - 1]) > 150) for i in range(1, len(cx))]
    rep = {'frames': len(frames), 'box': BOX, 'ckpt': CKPT.name,
           'seconds': round(dt, 1), 'median_area': med,
           'area_flagged': [i for i, f in enumerate(flags) if f],
           'centroid_jumps': [i for i, j in enumerate(jumps) if j]}
    (OUT / 'report.json').write_text(json.dumps(rep, indent=2))
    print(f'done {dt:.0f}s flagged_area={rep["area_flagged"]} jumps={rep["centroid_jumps"]}',
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
