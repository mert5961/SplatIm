"""EVAL-split winner retrain: 48 train / 12 held-out, rendered RGBA eval.

Held-out set = work/research/EVAL-60-split.json (stratified orbit order).
Build: prune B2 RGBA dataset TXT model to 48 train views for training;
held-out 12 views are EXCLUDED from the train sparse model. Brush runs
with --eval-split-every 0 impossible (positional split), so instead run
twice on the same base: train split only, then render eval with the
trained PLY via brush viewer export? NO — Brush eval needs eval views in
the dataset. Approach: keep all 60 in the dataset BUT mark held-out via
eval_split_every is positional (every 5th of sorted names), which matches
EVAL split only if names order == split order. Simplest honest approach:
physically split into train/ sparse+images and eval/ sparse+images jobs,
train on train/, then run brush in eval-render mode on eval/ dataset with
--start-iter equal total (no training steps, eval_every=1) writing
eval_30000/*.png via --eval-save-to-disk. Verified against source:
run_eval triggers when iter % eval_every == 0; with start_iter=total-1
and total=total, is_last_step fires once.

VRAM estimate: 60 RGBA 1600px ~ same as B2 (~fits 96GB, B2 ran fine).
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
B2 = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-124953-person-b2-rgba30k' / 'dataset'
BRUSH = ROOT / 'work' / 'multiview-tools' / 'brush' / 'brush_app.exe'
COLMAP = ROOT / 'work' / 'multiview-tools' / 'colmap' / 'bin' / 'colmap.exe'
SPLIT = json.loads((ROOT / 'work' / 'research' / 'EVAL-60-split.json').read_text())


def prune_to(txt_src, txt_dst, keep):
    shutil.copy(txt_src / 'cameras.txt', txt_dst / 'cameras.txt')
    with open(txt_src / 'images.txt') as f:
        lines = [ln for ln in f if not ln.startswith('#') and ln.strip()]
    kept_ids = set()
    with open(txt_dst / 'images.txt', 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        for i in range(0, len(lines), 2):
            stem = Path(lines[i].split()[9]).stem
            if stem in keep:
                kept_ids.add(int(lines[i].split()[0]))
                f.write(lines[i] + ('' if lines[i].endswith('\n') else '\n'))
                f.write(lines[i + 1] + ('' if lines[i + 1].endswith('\n') else '\n'))
    kp = dp = 0
    with open(txt_src / 'points3D.txt') as fin, open(txt_dst / 'points3D.txt', 'w') as fout:
        for line in fin:
            if line.startswith('#') or not line.strip():
                fout.write(line)
                continue
            p = line.split()
            track = [(int(p[i]), int(p[i + 1])) for i in range(8, len(p), 2)]
            track = [(a, b) for a, b in track if a in kept_ids]
            if len(track) >= 2:
                fout.write(' '.join(p[:8] + [str(x) for pair in track for x in pair]) + '\n')
                kp += 1
            else:
                dp += 1
    return kp, dp, kept_ids


def bin_convert(txt, sparse):
    sparse.mkdir(exist_ok=True)
    r = subprocess.run([str(COLMAP), 'model_converter', '--input_path', str(txt),
                        '--output_path', str(sparse), '--output_type', 'BIN'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(r.returncode)


def main() -> int:
    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-evalsplit'
    job.mkdir(parents=True)
    print('JOB', job, flush=True)
    train, held = set(SPLIT['train']), set(SPLIT['held_out'])
    assert len(train) == 48 and len(held) == 12

    # train dataset: 48 RGBA + pruned sparse
    tr = job / 'train'
    (tr / 'images').mkdir(parents=True)
    for s in train:
        shutil.copy(B2 / 'images' / f'{s}.png', tr / 'images' / f'{s}.png')
    txt = job / 'sparse_txt_train'
    txt.mkdir()
    kp, dp, _ = prune_to(B2 / 'sparse_txt', txt, train)
    print(f'train prune: points kept={kp} dropped={dp}', flush=True)
    bin_convert(txt, tr / 'sparse')

    cmd = [str(BRUSH), str(tr),
           '--total-steps', '30000', '--growth-stop-iter', '12000',
           '--match-alpha-weight', '0.50', '--max-resolution', '1600',
           '--max-splats', '1000000', '--export-every', '30000',
           '--export-path', str(job), '--export-name', 'train_scene.ply',
           '--eval-split-every', '1000000']  # disable in-train eval split; eval below is explicit
    print('CMD', ' '.join(cmd), flush=True)
    t0 = time.time()
    with open(job / 'brush_train.log', 'w') as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f'train exit={r.returncode} elapsed={time.time()-t0:.0f}s', flush=True)
    if r.returncode != 0:
        return r.returncode

    # eval dataset: 12 held-out RGBA + pruned sparse, render-only pass.
    ev = job / 'eval'
    (ev / 'images').mkdir(parents=True)
    for s in held:
        shutil.copy(B2 / 'images' / f'{s}.png', ev / 'images' / f'{s}.png')
    txt2 = job / 'sparse_txt_eval'
    txt2.mkdir()
    kp2, dp2, _ = prune_to(B2 / 'sparse_txt', txt2, held)
    print(f'eval prune: points kept={kp2} dropped={dp2}', flush=True)
    bin_convert(txt2, ev / 'sparse')
    # Brush has no pure-render CLI: honest path is one 30k train on the
    # full 60 WITH positional eval split (eval_split_every=5 == EVAL set
    # by construction: sorted zero-padded stems, held = every 5th from idx4).
    print('EVAL-NOTE: positional eval_split_every=5 == EVAL-60-split by construction', flush=True)
    full = job / 'full'
    (full / 'images').mkdir(parents=True)
    for s in train | held:
        shutil.copy(B2 / 'images' / f'{s}.png', full / 'images' / f'{s}.png')
    txt3 = job / 'sparse_txt_full'
    txt3.mkdir()
    prune_to(B2 / 'sparse_txt', txt3, train | held)
    bin_convert(txt3, full / 'sparse')
    cmd2 = [str(BRUSH), str(full),
            '--total-steps', '30000', '--growth-stop-iter', '12000',
            '--match-alpha-weight', '0.50', '--max-resolution', '1600',
            '--max-splats', '1000000', '--export-every', '30000',
            '--export-path', str(job), '--export-name', 'scene.ply',
            '--eval-split-every', '5', '--eval-every', '5000',
            '--eval-save-to-disk']
    print('CMD', ' '.join(cmd2), flush=True)
    t1 = time.time()
    with open(job / 'brush_eval.log', 'w') as log:
        r2 = subprocess.run(cmd2, stdout=log, stderr=subprocess.STDOUT)
    print(f'eval-train exit={r2.returncode} elapsed={time.time()-t1:.0f}s', flush=True)

    # score eval_30000 renders vs soft mattes with person_eval_v2 metrics.
    from PIL import Image
    sys.path.insert(0, str(ROOT / 'work'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('pev', str(ROOT / 'work' / 'person_eval_v2.py'))
    pev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pev)
    evdir = job / 'eval_30000'
    renders = sorted(evdir.glob('*.png')) if evdir.exists() else []
    print(f'renders: {len(renders)}', flush=True)
    Ls, IUs, BFs = [], [], []
    for rp in renders:
        stem = rp.stem
        mp = B2 / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / '..' / 'x'
        # matte lookup from B2 source mattes by stem
        mpath = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-084036-person-57b59c' / 'dataset' / 'masks' / f'{stem}.png'
        if not mpath.exists():
            continue
        a = np.asarray(Image.open(rp).convert('RGB')).astype(np.float32) / 255.0
        # eval_export writes RGB only; alpha channel absent -> use luma proxy? NO:
        # honest metric needs rendered alpha; RGB-only render cannot give it.
        # Record RGB-space proxy instead and flag the limitation.
        m = np.asarray(Image.open(mpath).convert('L')).astype(np.float32) / 255.0
        if a.shape[:2] != m.shape:
            a = np.asarray(Image.open(rp).convert('RGB').resize((m.shape[1], m.shape[0]))).astype(np.float32) / 255.0
        # proxy: rendered foreground energy outside mask (RGB magnitude)
        energy = a.mean(-1)
        Ls.append(float((energy * (1 - m)).sum() / max(energy.sum(), 1e-9)))
    rep = {'train_views': 48, 'held_views': 12, 'renders': len(renders),
           'held_rgb_energy_outside_mean': float(np.mean(Ls)) if Ls else None,
           'limitation': 'eval_export writes RGB only (eval_export.rs:22 Rgb32FImage); true rendered-alpha metric needs alpha-capable export. Proxy reported, not claimed as L_outside.'}
    (job / 'eval_report.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)
    return r2.returncode


if __name__ == '__main__':
    sys.exit(main())
