"""B5 step 1: scan the source video for the most static dense window.

Scores every candidate window (default 4s @ 30fps = 120 frames, stride
0.5s) on the 120 already-extracted orbit frames by:
  motion  = mean Farneback flow magnitude inside the RVM person mask
            between consecutive frames (lower is better),
  sharp   = mean Laplacian variance inside the mask (higher is better),
  cover   = COLMAP-aligned camera yaw span inside the window (higher better;
            from undistorted sparse_txt images.txt).
Picks the window maximizing cover/(1+motion) with sharpness as tiebreak,
and writes the chosen frame id list for the B5 build step.

Inputs: B0 undistorted images + soft RVM mattes + sparse_txt poses.
Output: /tmp/person_diag/b5_window.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
WINDOW = 24  # frames after dense re-extraction (~0.8s at 30fps); the scan
             # below only ranks orbit segments, the build re-extracts dense.
SRC_IMG = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-080603-07ed0a' / 'dataset' / 'images'
MASKS = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c' / 'dataset' / 'masks'
TXT = Path(r'/tmp/person_diag/sparse_txt/images.txt')
OUT = Path(r'/tmp/person_diag/b5_window.json')
WINDOW = 24  # ~15s of the 0.62s-spaced orbit frames
STRIDE = 2


def load_gray(p):
    import cv2
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def yaw_of(line):
    h = line.split()
    qw, qx, qy, qz = (float(x) for x in h[1:5])
    return math.atan2(2 * (qw * qy + qx * qz), 1 - 2 * (qy * qy + qx * qx))


def main() -> int:
    import cv2
    t0 = time.time()
    frames = sorted(SRC_IMG.glob('*.jpg'))
    assert len(frames) == 120, len(frames)
    gray = [load_gray(p) for p in frames]
    masks = []
    for p in frames:
        m = np.asarray(__import__('PIL.Image', fromlist=['open']).open(
            MASKS / f'{p.stem}.png').convert('L')).astype(np.float32) / 255.0
        masks.append((m > 0.5).astype(np.uint8))
    motion = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i + 1], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        m = ((masks[i] | masks[i + 1]) > 0).astype(bool)
        motion.append(float(mag[m].mean()) if m.any() else 0.0)
    sharp = [float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in gray]
    yaws = {}
    with open(TXT) as f:
        lines = [ln for ln in f if not ln.startswith('#') and ln.strip()]
    for i in range(0, len(lines), 2):
        h = lines[i].split()
        yaws[Path(h[9]).stem] = yaw_of(lines[i])

    best, rows = None, []
    for s in range(0, len(frames) - WINDOW + 1, STRIDE):
        seg = list(range(s, s + WINDOW))
        mo = float(np.mean([motion[i] for i in seg[:-1]]))
        sh = float(np.mean([sharp[i] for i in seg]))
        ys = [yaws[frames[i].stem] for i in seg if frames[i].stem in yaws]
        span = 0.0
        if len(ys) > 2:
            a = np.unwrap(np.array(ys))
            span = float(a.max() - a.min())
        score = span / (1.0 + mo)
        rows.append({'start': s, 'motion': mo, 'sharp': sh, 'span_deg': span * 180 / math.pi,
                     'score': score})
        if best is None or (score, sh) > (best['score'], best['sharp']):
            best = rows[-1]
    rows.sort(key=lambda r: (r['score'], r['sharp']), reverse=True)
    out = {'window': WINDOW, 'stride': STRIDE, 'n_frames': len(frames),
           'best': best, 'top5': rows[:5],
           'chosen': [frames[i].stem for i in range(best['start'], best['start'] + WINDOW)]}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"scanned {len(rows)} windows in {time.time()-t0:.0f}s", flush=True)
    for r in rows[:5]:
        print(f"start={r['start']:3d} motion={r['motion']:.3f} sharp={r['sharp']:.0f} "
              f"span={r['span_deg']:.1f}deg score={r['score']:.4f}", flush=True)
    print('chosen:', out['chosen'][0], '...', out['chosen'][-1], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
