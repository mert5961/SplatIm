"""Align overlapping photographs with COLMAP, then train one scene with Brush."""
from __future__ import annotations

import json
import hashlib
import os
import re
import struct
import subprocess
import threading
import time
import uuid
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT.parent / 'work' / 'multiview-tools'
BRUSH = TOOLS / 'brush' / 'brush_app.exe'
JOBS = ROOT / 'multi-view-scenes'
_cancel = threading.Event()
_active = threading.Event()


def stop_training():
    if _active.is_set():
        _cancel.set()
        return 'Stopping the current multi-photo job. Its files will be kept.'
    return 'No multi-photo job is running.'


def train_scene(files, quality, same_camera, allow_partial=False, progress=gr.Progress()):
    if not files or len(files) < 3:
        raise gr.Error('Choose at least 3 overlapping photos. For useful coverage, aim for 20–100 or more.')
    colmaps = list((TOOLS / 'colmap').rglob('colmap.exe'))
    if not colmaps or not BRUSH.is_file():
        raise gr.Error('COLMAP or Brush is missing from the local installation.')
    if _active.is_set():
        raise gr.Error('A multi-photo job is already running.')
    _active.set()
    _cancel.clear()
    job = JOBS / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    images = job / 'input'
    sparse = job / 'sparse'
    dataset = job / 'dataset'
    for directory in (images, sparse, dataset):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = job / 'pipeline.log'
    env = os.environ.copy()
    colmap = colmaps[0]
    env['PATH'] = str(colmap.parent) + os.pathsep + env.get('PATH', '')
    env['QT_QPA_PLATFORM'] = 'offscreen'
    presets = {'Preview': (1000, 1024), 'Standard': (7000, 1600), 'High quality': (30000, 1920)}
    steps, resolution = presets[quality]
    process = None
    terminal_process = False

    def tail_log():
        if not log_path.exists():
            return ''
        with log_path.open('rb') as stream:
            stream.seek(max(0, log_path.stat().st_size - 18000))
            return stream.read().decode('utf-8', errors='replace')

    def is_running():
        return process.isalive() if terminal_process else process.poll() is None

    def terminate():
        if terminal_process:
            process.terminate(force=True)
        else:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def stage(label, command, fraction):
        nonlocal process, terminal_process
        if _cancel.is_set():
            raise InterruptedError('Stopped by you. The partial project is saved.')
        with log_path.open('a', encoding='utf-8') as log:
            log.write('\n### ' + label + '\n' + subprocess.list2cmdline([str(x) for x in command]) + '\n')
            log.flush()
            terminal_process = Path(command[0]) == BRUSH
            reader_thread = None
            screen = None
            screen_lock = threading.Lock()
            previous_screen = ''
            if terminal_process:
                from winpty import PtyProcess
                import pyte
                screen = pyte.Screen(160, 24)
                terminal = pyte.Stream(screen)
                process = PtyProcess.spawn(subprocess.list2cmdline([str(x) for x in command]),
                    cwd=str(job), env=env, dimensions=(24, 160))
                def read_terminal():
                    while True:
                        try:
                            chunk = process.read(8192)
                            if not chunk:
                                break
                            with screen_lock:
                                terminal.feed(chunk)
                        except (EOFError, OSError):
                            break
                reader_thread = threading.Thread(target=read_terminal, daemon=True)
                reader_thread.start()
            else:
                process = subprocess.Popen([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT,
                                           env=env, cwd=job, creationflags=subprocess.CREATE_NO_WINDOW)
            started = time.monotonic()
            while is_running():
                if _cancel.wait(1):
                    terminate()
                    raise InterruptedError('Stopped by you. The partial project is saved.')
                live_text = ''
                if screen is not None:
                    with screen_lock:
                        live_text = '\n'.join(line.rstrip() for line in screen.display).strip()
                    if live_text and live_text != previous_screen:
                        log.write('\n' + live_text + '\n')
                        log.flush()
                        previous_screen = live_text
                match = re.search(r'(\d+)\s*/\s*(\d+)\s+Steps', live_text)
                detail = label
                if match:
                    current, total = map(int, match.groups())
                    if total > 0:
                        progress((current, total), desc='Training steps')
                        detail += f' · {current:,}/{total:,} ({current/total:.1%})'
                else:
                    progress(None, desc=label)
                elapsed = int(time.monotonic() - started)
                yield None, f'**{detail}** · {elapsed // 60}m {elapsed % 60:02d}s', None, tail_log()
            if reader_thread:
                reader_thread.join(timeout=3)
                with screen_lock:
                    log.write('\n' + '\n'.join(line.rstrip() for line in screen.display).strip() + '\n')
                log.flush()
            returncode = process.exitstatus if terminal_process else process.returncode
            if terminal_process:
                process.close()
                process = None
            if returncode not in (None, 0):
                raise RuntimeError(f'{label} failed (exit {returncode}). Download the run log for details.')

    try:
        from PIL import Image, ImageOps
        import pillow_heif
        pillow_heif.register_heif_opener()
        names = {}
        hashes = set()
        dimensions = set()
        for index, filename in enumerate(files):
            source = Path(filename)
            name = f'{index:05d}.jpg'
            with Image.open(source) as photo:
                photo = ImageOps.exif_transpose(photo)
                photo = photo.convert('RGB')
                digest = hashlib.sha256(photo.tobytes()).hexdigest()
                if digest in hashes:
                    continue
                hashes.add(digest)
                dimensions.add(photo.size)
                exif = photo.getexif().tobytes()
                photo.convert('RGB').save(images / name, quality=95, exif=exif)
            names[name] = source.name
        unique_count = len(names)
        if unique_count < 3:
            raise RuntimeError('Fewer than 3 unique images remain after removing duplicates.')
        if same_camera and len(dimensions) != 1:
            raise RuntimeError('Image dimensions differ. Uncheck the same-camera option and try again.')
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'Prepared {unique_count} unique images from {len(files)} uploads.\n')
        (job / 'input-names.json').write_text(json.dumps(names, indent=2), encoding='utf-8')
        db = job / 'database.db'
        yield from stage('1/4 · Finding visual features', [colmap, 'feature_extractor',
            '--database_path', db, '--image_path', images, '--ImageReader.camera_model', 'SIMPLE_RADIAL',
            '--ImageReader.single_camera', str(int(same_camera)),
            '--FeatureExtraction.max_image_size', '2400'], .1)
        yield from stage('2/4 · Matching overlapping photos', [colmap, 'exhaustive_matcher',
            '--database_path', db], .25)
        yield from stage('3/4 · Estimating camera positions', [colmap, 'mapper',
            '--database_path', db, '--image_path', images, '--output_path', sparse], .4)
        models = []
        for path in sparse.rglob('images.bin'):
            with path.open('rb') as stream:
                count = struct.unpack('<Q', stream.read(8))[0]
            models.append((count, path.parent))
        if not models:
            raise RuntimeError('No connected scene could be reconstructed. Use more overlapping, sharp photos taken from different positions.')
        registered, model = max(models, key=lambda value: value[0])
        if registered < 3:
            raise RuntimeError('Fewer than 3 cameras aligned. Add more overlapping photos.')
        with (model / 'points3D.bin').open('rb') as stream:
            point_count = struct.unpack('<Q', stream.read(8))[0]
        weak = registered / unique_count < .7 or point_count < 200
        alignment = {'uploaded_photos': len(files), 'unique_photos': unique_count,
                     'aligned_photos': registered, 'sparse_points': point_count,
                     'weak_alignment': weak, 'partial_training_allowed': bool(allow_partial)}
        (job / 'alignment.json').write_text(json.dumps(alignment, indent=2), encoding='utf-8')
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'\nALIGNMENT: {registered}/{unique_count} unique photos, {point_count} triangulated points.\n')
        if weak and not allow_partial:
            raise RuntimeError(f'Alignment is too weak for a reliable scene: {registered}/{unique_count} unique photos and {point_count} 3D points. Training was skipped. Add more overlapping views of a motionless subject. Enable experimental partial training only if you want this incomplete reconstruction.')
        yield from stage(f'3/4 · Correcting lenses ({registered}/{len(files)} photos aligned)',
            [colmap, 'image_undistorter', '--image_path', images, '--input_path', model,
             '--output_path', dataset, '--output_type', 'COLMAP', '--max_image_size', str(resolution)], .55)
        yield from stage(f'4/4 · Training one scene · {steps:,} steps',
            [BRUSH, dataset, '--total-steps', str(steps), '--max-resolution', str(resolution),
             '--max-splats', '3000000', '--export-every', str(steps), '--export-path', job,
             '--export-name', 'scene.ply'], .65)
        result = job / 'scene.ply'
        if not result.is_file():
            raise RuntimeError('The trainer finished without exporting a splat.')
        import numpy as np
        from plyfile import PlyData
        vertices = PlyData.read(result)['vertex'].data
        if not len(vertices) or not all(np.isfinite(vertices[name]).all() for name in vertices.dtype.names):
            raise RuntimeError('The exported splat is empty or has invalid numeric data.')
        report = {'input_photos': len(files), 'unique_photos': unique_count, 'sparse_points': point_count, 'aligned_photos': registered, 'components': len(models),
                  'steps': steps, 'gaussians': len(vertices), 'output': str(result)}
        (job / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        warning = (f' **{len(files)-registered} photos were not included in the largest connected scene.**'
                   if registered < len(files) else '')
        progress(1, desc='Scene ready')
        yield str(result), f'Finished: **{registered}/{len(files)} photos** combined into one scene with **{len(vertices):,} Gaussians**.{warning}\n\nOpen the downloaded PLY in the **View splat** tab.', str(log_path), tail_log()
    except (RuntimeError, InterruptedError, OSError, ValueError) as error:
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'\nSTOP: {error}\n')
        yield None, f'**{error}**\n\nProject files: `{job.name}`', str(log_path), tail_log()
    finally:
        if process is not None and is_running():
            terminate()
        _active.clear()
