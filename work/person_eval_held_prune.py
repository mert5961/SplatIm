"""G3: full 30k brush retrain of the aw0.65 winner recipe with the G2-winning
prune opacity level. Single variable vs the held-gate baseline
(20260915-043006-person-eval-held-aw0.65, L_outside 2.53%): only
--prune-opacity changes (brush's built-in default is 2/255 = 0.0078).

Requires the patched brush_app (work/brush-source, flag --prune-opacity).
Scoring is identical to person_eval_held.py (true RGBA alpha vs RVM matte,
12 honest held views of brush's positional 48/12 split).
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-151822-person-evalsplit' / 'full'
BRUSH = ROOT / 'work' / 'multiview-tools' / 'brush' / 'brush_app.exe'
MASKS = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c' / 'dataset' / 'masks'
SPLIT = json.loads((ROOT / 'work' / 'research' / 'EVAL-60-split.json').read_text())
STEMS = sorted(set(SPLIT['train']) | set(SPLIT['held_out']))
HELD = [s for i, s in enumerate(STEMS) if i % 5 == 0]  # brush's eval phase


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run(aw: str, prune: str) -> Path:
    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-eval-held-aw{aw}-prune{prune}'
    job.mkdir(parents=True, exist_ok=True)
    cmd = [str(BRUSH), str(FULL),
           '--total-steps', '30000', '--growth-stop-iter', '12000',
           '--match-alpha-weight', aw, '--max-resolution', '1600',
           '--max-splats', '1000000', '--export-every', '30000',
           '--export-path', str(job), '--export-name', 'scene.ply',
           '--eval-split-every', '5', '--eval-every', '5000',
           '--eval-save-to-disk', '--prune-opacity', prune]
    print('CMD', ' '.join(cmd), flush=True)
    t0 = time.time()
    log = open(job / 'brush.log', 'w', buffering=1)
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    log.close()
    (job / 'timing.json').write_text(json.dumps(
        {'run': f'aw{aw}-prune{prune}', 'start': t0, 'elapsed_s': round(elapsed, 1),
         'exit': r.returncode}, indent=2))
    print(f'aw={aw} prune={prune} exit={r.returncode} elapsed={elapsed:.0f}s', flush=True)
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    return job


def score(job: Path, aw: str, prune: str) -> dict:
    from PIL import ImageFilter
    import colour
    from PIL import Image

    evdir = job / 'eval_30000'
    per, Ls, cbl, cfr, ious, f1s, proxies, de = {}, [], [], [], [], [], [], []
    for stem in HELD:
        rp = evdir / f'{stem}.png'
        if not rp.exists():
            per[stem] = {'error': 'missing png'}
            continue
        im = Image.open(rp)
        r = np.asarray(im.convert('RGBA')).astype(np.float32) / 255.0
        m = np.asarray(Image.open(MASKS / f'{stem}.png').convert('L')).astype(np.float32) / 255.0
        if r.shape[:2] != m.shape:
            im2 = im.convert('RGBA').resize((m.shape[1], m.shape[0]))
            r = np.asarray(im2).astype(np.float32) / 255.0
        a = r[..., 3]
        l_out = float(a[m < 0.5].mean()) if (m < 0.5).any() else float('nan')
        c_bg = float(a[m == 0].mean()) if (m == 0).any() else float('nan')
        fg99 = m > 0.99
        fg_rec = float((a[fg99] > 0.5).mean()) if fg99.any() else float('nan')
        A = a > 0.5
        B = m > 0.5
        iou = float((A & B).sum() / max((A | B).sum(), 1))
        Aimg = Image.fromarray((A * 255).astype(np.uint8))
        Bimg = Image.fromarray((B * 255).astype(np.uint8))
        Ad = Aimg.filter(ImageFilter.MaxFilter(7)).point(lambda v: 255 if v > 127 else 0)
        Ae = Aimg.filter(ImageFilter.MinFilter(7)).point(lambda v: 255 if v > 127 else 0)
        Bd = Bimg.filter(ImageFilter.MaxFilter(7)).point(lambda v: 255 if v > 127 else 0)
        Be = Bimg.filter(ImageFilter.MinFilter(7)).point(lambda v: 255 if v > 127 else 0)
        eA = (np.asarray(Ad) > 127) & ~(np.asarray(Ae) > 127)
        eB = (np.asarray(Bd) > 127) & ~(np.asarray(Be) > 127)
        P = float((eA & eB).sum() / max(eA.sum(), 1))
        R = float((eB & eA).sum() / max(eB.sum(), 1))
        f1 = 2 * P * R / max(P + R, 1e-9)
        energy = r.mean(-1)
        proxy = float((energy * (1 - m)).sum() / max(energy.sum(), 1e-9))
        per[stem] = {'mode': im.mode, 'L_outside': l_out, 'certain_bg_leak': c_bg,
                     'certain_fg_recall': fg_rec, 'iou': iou, 'boundary_f1': f1,
                     'rgb_proxy': proxy}
        Ls.append(l_out)
        cbl.append(c_bg)
        cfr.append(fg_rec)
        ious.append(iou)
        f1s.append(f1)
        proxies.append(proxy)
        core = m > 0.9
        if core.sum() >= 1000:
            gt = np.asarray(Image.open(FULL / 'images' / f'{stem}.png').convert('RGB')).astype(np.float32) / 255.0
            L1 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(gt[core], 0, 1)))
            L2 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(r[..., :3][core], 0, 1)))
            d = colour.delta_E(L1, L2, method='CIE 2000')
            de.extend(float(x) for x in d[::37])

    n = len(Ls)
    rep = {
        'job': str(job), 'aw': aw, 'prune_opacity': float(prune),
        'n_held_scored': n, 'held_views': HELD,
        'L_outside': {'mean': float(np.mean(Ls)), 'min': float(np.min(Ls)),
                      'max': float(np.max(Ls))},
        'certain_bg_leak_mean': float(np.mean(cbl)),
        'certain_fg_recall_mean': float(np.mean(cfr)),
        'iou_mean': float(np.mean(ious)),
        'boundary_f1_mean': float(np.mean(f1s)),
        'rgb_proxy_mean': float(np.mean(proxies)),
        'de00_fg_core': {'n': len(de),
                         'median': round(float(np.median(de)), 4) if de else None,
                         'p95': round(float(np.percentile(de, 95)), 4) if de else None},
        'gate': {'L_outside_lt_1pct': bool(np.mean(Ls) < 0.01) if n else False},
        'baseline_held_gate_aw065_L_outside_mean': 0.025347159632171195,
        'per_view': per,
    }
    (job / 'metrics.json').write_text(json.dumps(rep, indent=2))
    hashes = {
        'inputs': {
            'images': {s: sha256(FULL / 'images' / f'{s}.png') for s in STEMS},
            'sparse_images_bin': sha256(FULL / 'sparse' / 'images.bin'),
        },
        'outputs': {
            'scene.ply': sha256(job / 'scene.ply'),
            'eval_30000': {s: sha256(evdir / f'{s}.png') for s in HELD
                           if (evdir / f'{s}.png').exists()},
        },
        'config': {'total_steps': 30000, 'growth_stop_iter': 12000,
                   'match_alpha_weight': float(aw), 'max_resolution': 1600,
                   'max_splats': 1000000, 'eval_split_every': 5,
                   'eval_every': 5000, 'eval_save_to_disk': True,
                   'prune_opacity': float(prune),
                   'train_views': len(STEMS) - len(HELD), 'held_views': len(HELD),
                   'dataset': str(FULL)},
    }
    (job / 'report.json').write_text(json.dumps(hashes, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != 'per_view'}, indent=2), flush=True)
    return rep


def main() -> int:
    aw = os.environ.get('AW', '0.65')
    prune = os.environ.get('PRUNE_TAU', '0.15')
    job = run(aw, prune)
    rep = score(job, aw, prune)
    base = rep['baseline_held_gate_aw065_L_outside_mean']
    print(f"GATE: baseline {base*100:.2f}% -> {rep['L_outside']['mean']*100:.2f}% "
          f"({'MET' if rep['gate']['L_outside_lt_1pct'] else 'NOT MET'} <1%)", flush=True)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
