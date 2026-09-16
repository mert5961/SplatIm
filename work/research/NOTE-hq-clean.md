# NOTE: H2-extract-hq environment-residue cleaning (2026-09-16)

User reported heavy environment residue in the H2-extract-hq person asset.

## Root cause
- `keep-unc-despill-hq.ply` (129,322) = keep (107,648) + uncertain (21,674,
  opacity halved). The haze seen in renders/viewer is the **uncertain set**:
  environment splats (wall/ceiling/floor) with 2-4 weak fg-evidence hits.
- The **keep set itself is clean**: per-splat inside-mask fraction over all 120
  orbit views has mean 0.967, median 1.000, p10 0.891; zero splats below 0.5.
  No wall leak into keep (h2's transmittance evidence works).

## Why the mask-consistency filter alone cannot clean unc
The studio wall surrounds the orbit, so wall splats project INSIDE the person
mask in most views (they are behind the person). inside_frac >= 0.5 keeps
14,594 unc splats that are mostly "wall behind person", not hair -> haze
remains in keep-edges.

## Assets built (work/research/H2-extract-hq/)
| asset | splats | interior | band | IoU(c4) | halo |
|---|---|---|---|---|---|
| keep-unc-despill-hq (old) | 129,322 | 5.84 | 11.99 | 0.9819 | 0.0 |
| keep-clean-despill-hq (keep only) | 107,648 | 6.12 | 12.54 | 0.9803 | 0.0 |
| keep-edges-despill-hq (keep + unc frac>=0.5) | 122,242 | 5.77 | 10.79 | 0.9818 | 0.0 |

## Recommendation
- **3D viewer / compositing base: keep-clean-despill-hq.ply** (haze-free).
- keep-edges wins metrics slightly (band -1.15 dE00: the dropped 7,080
  environment splats also polluted the band) but still hazy in raw renders.
- Remaining artifact: green rim on silhouette edges (despill under-corrects
  the 4-8 px band); next step if needed = stronger despill on edge band only.

Scripts: hq_clean.py (keep filter), hq_clean2.py (unc split),
hq_sheet.py (12-view GT/raw/clip4 sheets -> work/research/HQ-renders/).
