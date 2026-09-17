"""Person-only extraction (YUNUS, purple shirt) from the existing full-scene job.

Job: outputs/multi-view-scenes/20260917-051016-7fdc72 (120 views, 542,749 G,
30k steps, no masks). The user wants ONLY the purple-shirt person, not the
office.

Approach (H-plan evidence: full-scene extraction beats masked training):
  sam2        SAM 2.1 box prompt on undistorted frame 0 -> 120 hard masks
  masks       copy masks into dataset/masks
  eval_layout model_converter BIN->TXT + 12 held GT + masks
  prefilter   drop splats outside the person silhouette in ALL 120 views:
              they never overlap a person pixel, so they contribute nothing
              to fg evidence and nothing to transmittance on person pixels
              -> classification is bit-identical to a full run, much faster
  extract     h2_extract mask-guided evidence on the prefiltered PLY
  evalh3      h3_eval raw + clip4 on 12 held views (output stem keep-yunus)
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
JOB = ROOT / 'outputs/multi-view-scenes/20260917-051016-7fdc72'
TOOLS = ROOT / 'work' / 'multiview-tools'
COLMAP = TOOLS / 'colmap' / 'bin' / 'colmap.exe'
PY = ROOT / 'work' / 'sharp-env' / 'Scripts' / 'python.exe'
CKPT_SAM2 = ROOT / 'work' / 'models' / 'sam2' / 'sam2.1_hiera_small.pt'
SAM2_CFG = ROOT / 'work' / 'sam2repo' / 'sam2' / 'configs' / 'sam2.1' / 'sam2.1_hiera_s.yaml'
EXTRACT_OUT = ROOT / 'work' / 'research' / 'YUNUS-extract'

MAX_FRAMES = 120
HELD = [f'{i:05d}' for i in range(1, 120, 10)]
BOX = [95, 80, 465, 943]  # x0,y0,x1,y1 on undistorted frame 0 (530x943)


def log(msg: str):
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    with (JOB / 'person_extract.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def run_cmd(cmd, label: str):
    log(f'### {label}\n### {subprocess.list2cmdline([str(c) for c in cmd])}')
    with (JOB / 'person_extract.log').open('a', encoding='utf-8') as f:
        r = subprocess.run([str(c) for c in cmd], stdout=f,
                           stderr=subprocess.STDOUT, cwd=str(JOB),
                           creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:
        raise RuntimeError(f'{label} failed (exit {r.returncode})')
    log(f'{label} done')


def stage_sam2():
    raw = JOB / 'sam2_raw'
    if len(list(raw.glob('*.png'))) >= MAX_FRAMES:
        log('sam2: masks present, skip')
        return
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(ROOT / 'work' / 'sam2repo'))
    from sam2.build_sam import build_sam2_video_predictor

    stage = JOB / 'sam2_frames'
    stage.mkdir(exist_ok=True)
    srcs = sorted((JOB / 'dataset' / 'images').glob('*.jpg'))
    assert len(srcs) == MAX_FRAMES, len(srcs)
    for i, p in enumerate(srcs):
        dst = stage / f'{i:05d}.jpg'
        if not dst.exists():
            shutil.copyfile(p, dst)
    raw.mkdir(exist_ok=True)

    def _bool_mask(vm):
        return (vm.amax(dim=0) > 0).cpu().numpy()[0]

    log(f'sam2: box={BOX}')
    predictor = build_sam2_video_predictor(str(SAM2_CFG), str(CKPT_SAM2), device='cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(stage))
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1, box=np.array(BOX))
        t0 = time.time()
        n = 0
        for fi, _, vm in predictor.propagate_in_video(state):
            m = _bool_mask(vm)
            Image.fromarray(m.astype(np.uint8) * 255).save(raw / f'{fi:05d}.png')
            n += 1
            if n % 20 == 0:
                log(f'sam2: {n}/{len(srcs)} ({time.time() - t0:.0f}s)')
    coves = [float((np.asarray(Image.open(raw / f'{i:05d}.png')) > 0).mean())
             for i in range(MAX_FRAMES)]
    bad = [i for i, c in enumerate(coves) if c < 0.05]
    if bad:
        raise RuntimeError(f'sam2 masks too small in frames {bad} '
                           f'(coverage {coves[bad[0]]:.3f})')
    log(f'sam2: {n} masks, mean coverage {float(np.mean(coves)):.3f} '
        f'min {min(coves):.3f} ({time.time() - t0:.0f}s)')


def stage_masks():
    masks = JOB / 'dataset' / 'masks'
    if len(list(masks.glob('*.png'))) >= MAX_FRAMES:
        log('masks: present, skip')
        return
    masks.mkdir(parents=True, exist_ok=True)
    for i in range(MAX_FRAMES):
        src = JOB / 'sam2_raw' / f'{i:05d}.png'
        dst = masks / f'{i:05d}.png'
        if not dst.exists():
            shutil.copyfile(src, dst)
    log(f'masks: {len(list(masks.glob("*.png")))} in dataset/masks')


def stage_eval_layout():
    eval_dir = JOB / 'eval'
    sparse_txt = eval_dir / 'sparse_txt'
    gt = eval_dir / 'images'
    emasks = eval_dir / 'masks'
    if (sparse_txt / 'images.txt').exists() and \
            len(list(gt.glob('*.png'))) >= len(HELD) and \
            len(list(emasks.glob('*.png'))) >= MAX_FRAMES:
        log('eval layout: present, skip')
        return
    from PIL import Image
    sparse_txt.mkdir(parents=True, exist_ok=True)
    run_cmd([COLMAP, 'model_converter',
             '--input_path', JOB / 'dataset' / 'sparse',
             '--output_path', sparse_txt, '--output_type', 'TXT'],
            'model_converter BIN->TXT')
    gt.mkdir(exist_ok=True)
    for stem in HELD:
        src = JOB / 'dataset' / 'images' / f'{stem}.jpg'
        dst = gt / f'{stem}.png'
        if not dst.exists():
            Image.open(src).convert('RGB').save(dst)
    emasks.mkdir(exist_ok=True)
    for p in (JOB / 'dataset' / 'masks').glob('*.png'):
        dst = emasks / p.name
        if not dst.exists():
            shutil.copyfile(p, dst)
    log(f'eval layout: sparse_txt + {len(list(gt.glob("*.png")))} held GT '
        f'+ {len(list(emasks.glob("*.png")))} masks')


def stage_prefilter():
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT / 'work'))
    import splat_alpha_render as g1
    import h2_extract
    from plyfile import PlyData, PlyElement

    out_ply = EXTRACT_OUT / 'prefilter.ply'
    report = EXTRACT_OUT / 'prefilter.json'
    if out_ply.exists():
        log(f'prefilter: present ({out_ply.stat().st_size // 1024} KB), skip')
        return
    EXTRACT_OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ply = JOB / 'scene.ply'
    v = PlyData.read(str(ply))['vertex']
    xyz, scale, rot, opacity = g1.load_ply(ply)
    cov3d = g1.gauss_cov3d(scale, rot)
    n = len(xyz)
    log(f'prefilter: {n} gaussians')

    cams, views = g1.load_cameras(JOB / 'eval' / 'sparse_txt')
    # keep-candidate = in-frame AND inside mask in at least one view -> only
    # splats with possible fg evidence. Everything else is provably fg=0 ->
    # hide in the full run (never-in-frame: 0/0 evidence; outside-mask: bg only).
    # On a 360-degree orbit this set is large (nearly everything is behind the
    # person from some angle), so the reduction is small; kept for correctness
    # and for partial-orbit jobs where it is large.
    keepcand = np.zeros(n, bool)
    inframe_ever = np.zeros(n, bool)
    for vi, vw in enumerate(views):
        cam = cams[vw['cam_id']]
        w, h = cam['w'], cam['h']
        m = np.asarray(Image.open(JOB / 'eval' / 'masks' / f"{vw['stem']}.png").convert('L')) > 0
        if m.shape != (h, w):
            m = np.asarray(Image.fromarray(m).resize((w, h))) > 0
        order, u, v_, radius, _ = h2_extract.render_geometry(xyz, cov3d, cam, vw['q'], vw['t'], w, h)
        inb = np.zeros(n, bool)
        inb[order] = True
        inside = np.zeros(n, bool)
        pu = np.clip(u.round().astype(np.int32), 0, w - 1)
        pv = np.clip(v_.round().astype(np.int32), 0, h - 1)
        inside[order] = m[pv, pu]
        np.logical_or(keepcand, inb & inside, out=keepcand)
        np.logical_or(inframe_ever, inb, out=inframe_ever)
        if vi % 20 == 0:
            log(f'prefilter: {vi + 1}/{len(views)} keep-candidates so far {int(keepcand.sum())}')

    sel = np.nonzero(keepcand)[0]
    data = np.ascontiguousarray(v.data[sel])
    PlyData([PlyElement.describe(data, 'vertex')], text=False).write(str(out_ply))
    rep = {'n_full': n, 'n_prefilter': len(sel),
           'n_dropped': int((~keepcand).sum()),
           'n_never_inframe': int((~inframe_ever).sum()),
           'secs': round(time.time() - t0)}
    report.write_text(json.dumps(rep, indent=2))
    log(f'prefilter: kept {len(sel)}/{n} (dropped {int((~keepcand).sum())} provable bg) '
        f'in {time.time() - t0:.0f}s')


def stage_extract():
    if (EXTRACT_OUT / 'evidence.npz').exists() and (EXTRACT_OUT / 'keep.ply').exists():
        log('extract: present, skip')
        return
    EXTRACT_OUT.mkdir(parents=True, exist_ok=True)
    run_cmd([PY, ROOT / 'work' / 'h2_extract.py', EXTRACT_OUT / 'prefilter.ply',
             '--sparse', JOB / 'eval' / 'sparse_txt',
             '--masks', JOB / 'eval' / 'masks',
             '--out', EXTRACT_OUT,
             '--report', EXTRACT_OUT / 'report.json'],
            'h2_extract (person)')


def stage_evalh3():
    keep = EXTRACT_OUT / 'keep.ply'
    if not keep.exists():
        raise RuntimeError('keep.ply missing')
    named = EXTRACT_OUT / 'keep-yunus.ply'
    if not named.exists():
        shutil.copyfile(keep, named)
    for clip in (0, 4):
        tag = f'keep-yunus-clip{clip}'
        if (ROOT / 'work' / 'research' / f'H3-eval-{tag}.json').exists():
            log(f'evalh3 {tag}: present, skip')
            continue
        run_cmd([PY, ROOT / 'work' / 'h3_eval.py', str(named),
                 '--sparse', str(JOB / 'eval' / 'sparse_txt'),
                 '--images', str(JOB / 'eval' / 'images'),
                 '--masks', str(JOB / 'eval' / 'masks'),
                 '--stems', ','.join(HELD), '--clip', str(clip)],
                f'h3_eval {tag}')


STAGES = {n: f for n, f in list(globals().items()) if n.startswith('stage_')}
ORDER = ['sam2', 'masks', 'eval_layout', 'prefilter', 'extract', 'evalh3']


def main():
    stages = sys.argv[1:] or ORDER
    log(f'person-extract start: {stages}')
    for s in stages:
        STAGES[f'stage_{s}']()
    log('person-extract complete')


if __name__ == '__main__':
    main()
