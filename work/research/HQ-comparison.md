# HQ re-run comparison (30k steps / 1920 px / 3M cap)

Held sets differ (old: 60-frame evalsplit, 12 stems; new: 120-frame
orbit, stems 00001..00111 every 10th). Aggregate means only.

| model | interior dE00 | band dE00 | silhouette IoU | alpha 4-8 px |
|---|---|---|---|---|
| 7k keep-unc-despill + clip4 (old winner) | 5.9114 | 11.387 | 0.9811 | 0.0 |
| B0-HQ raw (30k/1920px, full scene) | 5.2868 | 8.7766 | 0.4772 | 0.8883 |
| HQ keep-unc-despill raw | 5.8416 | 11.9858 | 0.9672 | 0.34 |
| **HQ keep-unc-despill + clip4** | 5.8416 | 11.9858 | 0.9819 | 0.0 |

## Extraction counts

- 7k: keep 48,838 / unc 10,896 / hide 368,571 (of 428,305 splats)
- HQ: keep 107,648 / unc 21,674 / hide 1,010,266 (of 1,139,588 splats)
