"""A1: freeze B6 with manifest + hashes. Read-only wrt the B6 job dir."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
B6 = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-132825-person-b6-despillv2'
B3 = ROOT / 'outputs' / 'multi-view-scenes' / '20260914-125903-person-b3-filt30k'
OUT = ROOT / 'work' / 'research' / 'FREEZE-B6-manifest.json'


def sha256(p: Path, n: int = 1 << 26) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    images = sorted((B6 / 'dataset' / 'images').glob('*.png'))
    img_hashes = [sha256(p) for p in images]
    manifest = {
        'experiment_id': 'B6-despillv2',
        'branch': 'refactor/c-plan-rgba-30k',
        'job_dir': str(B6),
        'scene_ply': {'path': str(B6 / 'scene.ply'), 'sha256': sha256(B6 / 'scene.ply'),
                      'bytes': (B6 / 'scene.ply').stat().st_size},
        'report': json.loads((B6 / 'report.json').read_text()),
        'alignment': json.loads((B6 / 'alignment.json').read_text()),
        'frames': {'count': len(images),
                   'names': [p.name for p in images],
                   'combined_sha256': hashlib.sha256(''.join(img_hashes).encode()).hexdigest()},
        'sparse': {p: sha256(B6 / 'dataset' / 'sparse' / p)
                   for p in ('cameras.bin', 'images.bin', 'points3D.bin')},
        'mattes_src': str(ROOT / 'outputs' / 'multi-view-scenes' /
                          '20260914-084036-person-57b59c' / 'dataset' / 'masks'),
        'provenance': {'b3_job': str(B3), 'poses_from': '20260914-080603-07ed0a'},
        'metrics': {'gaussians': 57997, 'outside_alpha_opacity_mean': 0.013,
                    'edge_excess_pretrain': 0.04},
        'timing': {'train_elapsed_s': 203},
    }
    try:
        import torch
        manifest['env'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                           'python': __import__('sys').version.split()[0]}
    except Exception as e:  # noqa: BLE001
        manifest['env'] = {'error': str(e)}
    OUT.write_text(json.dumps(manifest, indent=2))
    print(f'freeze manifest -> {OUT} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
