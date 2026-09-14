"""Person-only 3DGS: matte the person out of the video-orbit dataset.

Runs Robust Video Matting (RVM, resnet50) over the 120 undistorted frames
of the 2026-09-14 orbit (job 20260914-080603-07ed0a) and writes one
grayscale PNG per frame into a fresh dataset's ``masks`` folder
(white = person, black = environment). Brush ignores black mask regions,
so the subsequent 3DGS training keeps only the person — no environment.

Camera poses are reused from the source job (COLMAP is not re-run).
"""
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'work' / 'rvm'))

import numpy as np
import torch
from PIL import Image

from model import MattingNetwork

SOURCE_JOB = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-080603-07ed0a'
CHECKPOINT_URL = 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth'


def main():
    device = 'cuda'
    model = MattingNetwork('resnet50').eval().to(device)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location='cpu', progress=True))

    ts = time.strftime('%Y%m%d-%H%M%S')
    job = ROOT / 'outputs' / 'multi-view-scenes' / f'{ts}-person-{uuid.uuid4().hex[:6]}'
    dataset = job / 'dataset'
    (dataset / 'images').mkdir(parents=True)
    shutil.copytree(SOURCE_JOB / 'dataset' / 'images', dataset / 'images' / 'src')
    shutil.copytree(SOURCE_JOB / 'dataset' / 'sparse', dataset / 'sparse')
    masks = dataset / 'masks'
    masks.mkdir()

    images = sorted((dataset / 'images' / 'src').glob('*.jpg'))
    print(f'{len(images)} frames from {SOURCE_JOB.name}', flush=True)

    # Brush expects images directly under dataset/images (not a src subdir)
    for p in images:
        shutil.move(str(p), str(dataset / 'images' / p.name))
    shutil.rmtree(dataset / 'images' / 'src')

    rec = [None] * 4
    downsample = None
    t0 = time.time()
    for i, src_path in enumerate(sorted((dataset / 'images').glob('*.jpg'))):
        img = np.asarray(Image.open(src_path).convert('RGB'), dtype=np.float32) / 255.0
        src = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
        if downsample is None:
            downsample = min(512 / max(src.shape[3], src.shape[4]), 1.0)
        with torch.no_grad():
            fgr, pha, *rec = model(src, *rec, downsample)
        alpha = (pha[0, 0, 0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(alpha, 'L').save(masks / (src_path.stem + '.png'))
        if i in (0, 60, 119):
            f = fgr[0, 0].cpu().numpy().transpose(1, 2, 0)
            a = alpha[..., None] / 255.0
            rgba = np.concatenate([f * a, a], axis=-1)
            Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA').save(
                job / f'matte_preview_{src_path.stem}.png')
        if i % 20 == 0:
            print(f'{i}/{len(images)} done ({time.time() - t0:.0f}s)', flush=True)

    print(f'masks written: {len(list(masks.glob("*.png")))} in {time.time() - t0:.0f}s', flush=True)
    print('JOB', job, flush=True)


if __name__ == '__main__':
    main()
