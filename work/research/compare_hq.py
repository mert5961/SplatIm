"""Compare 7k B0 pipeline vs 30k/1920px HQ pipeline (aggregate level).

Held-view sets differ between runs (old: 60-frame evalsplit stems; new:
120-frame set, every 10th), so comparison is on per-run means.

Reads:
  H3-eval-keep-unc-despill-clip4.json      (7k winner, old held set)
  H3-eval-scene-clip0.json                 (B0-HQ raw, new held set)
  H3-eval-keep-unc-despill-hq-clip0.json   (HQ winner raw)
  H3-eval-keep-unc-despill-hq-clip4.json   (HQ winner + matte clip)
  H2-extract-hq.json / H2-extract.json     (extraction counts)
Writes: work/research/HQ-comparison.md
"""
import json
from pathlib import Path

ROOT = Path(r'C:\Users\reality\Documents\Codex\2026-09-11\i-w')
R = ROOT / 'work' / 'research'


def load(name):
    p = R / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def row(label, d):
    if d is None:
        return f'| {label} | — |'
    a = d['agg']
    return (f"| {label} | {a.get('interior_dE00', '—')} | "
            f"{a.get('band_dE00', '—')} | {a.get('silhouette_iou', '—')} | "
            f"{a.get('alpha_outside', '—')} |")


old_winner = load('H3-eval-keep-unc-despill-clip4.json')
new_b0 = load('H3-eval-scene-clip0.json')
new_raw = load('H3-eval-keep-unc-despill-hq-clip0.json')
new_clip = load('H3-eval-keep-unc-despill-hq-clip4.json')
old_h2 = load('H2-extract.json')
new_h2 = load('H2-extract-hq.json')

lines = ['# HQ re-run comparison (30k steps / 1920 px / 3M cap)',
         '',
         'Held sets differ (old: 60-frame evalsplit, 12 stems; new: 120-frame',
         'orbit, stems 00001..00111 every 10th). Aggregate means only.',
         '',
         '| model | interior dE00 | band dE00 | silhouette IoU | alpha 4-8 px |',
         '|---|---|---|---|---|']
lines.append(row('7k keep-unc-despill + clip4 (old winner)', old_winner))
lines.append(row('B0-HQ raw (30k/1920px, full scene)', new_b0))
lines.append(row('HQ keep-unc-despill raw', new_raw))
lines.append(row('**HQ keep-unc-despill + clip4**', new_clip))
lines += ['', '## Extraction counts', '']
if old_h2 and new_h2:
    lines += [f"- 7k: keep {old_h2['n_keep']:,} / unc {old_h2['n_uncertain']:,} / "
              f"hide {old_h2['n_hide']:,} (of {old_h2['n']:,} splats)",
              f"- HQ: keep {new_h2['n_keep']:,} / unc {new_h2['n_uncertain']:,} / "
              f"hide {new_h2['n_hide']:,} (of {new_h2['n']:,} splats)"]
else:
    lines.append('(pending)')
lines.append('')

(R / 'HQ-comparison.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
