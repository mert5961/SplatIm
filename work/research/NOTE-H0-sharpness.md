# NOTE-H0-sharpness — Full scene vs masked winner: detail comparison

Date: 2026-09-15
Question (user): the masked pipeline (E3x aw0.65) looks sharp, but does it
preserve **all** the detail of the original full-scene reconstruction (B0)?
If not, build extraction around the **full** scene instead of retraining
masked.

## Models compared
- **B0 (full scene)**: `outputs/multi-view-scenes/20260914-080603-07ed0a/scene.ply`
  — 428,305 Gaussians, original 7-step green-screen reconstruction.
- **MX (masked winner)**: `outputs/multi-view-scenes/20260914-154428-person-e3x-aw0.65/scene.ply`
  — 115,886 Gaussians, E3x premultiplied-alpha training on RVM masks.
- **GT**: `outputs/multi-view-scenes/20260914-151822-person-evalsplit/full/images/*.png`
  (60 original photos, 12 held views sampled: 00001, 00009, 00016, 00026,
  00039, 00048, 00056, 00067, 00079, 00088, 00100, 00109).

## Method
- New exact TBB renderer `work/splat_alpha_render.py` (per-splat sequential
  depth-order composition, tiled, float32). Verified on a synthetic 3-splat
  case (alpha exact, color exact) and on view 00001 (alpha_max = 1.0,
  person colors clean — no green bleed-through).
 - `work/h0_compare.py` renders both models on the 12 held views, then
   computes CIEDE2000 against GT in three regions per view:
   - `interior`: RVM mask eroded by 12 px
   - `band`: |mask - eroded| (boundary region)
   - `cross`: B0 vs MX directly (interior region)
 - RVM masks: `outputs/multi-view-scenes/20260914-084036-person-57b59c/dataset/masks/*.png`

## Results (dE00, 12 held views, mean)

| Region | B0 full | MX masked | Verdict |
|---|---|---|---|
| interior FG | **5.13** | 6.24 | masked 22% worse |
| boundary band | **8.52** | 11.16 | masked 31% worse |
| outside (vs green GT) | 11.93 | 46.56 | MX has **no background at all** (renders black outside matte) |
| cross B0↔MX (interior) | 3.77 | — | the two models disagree on the object itself |

Per-view data: `work/research/H0-compare.json`.
Side-by-side strips (GT | B0 | MX): `work/research/H0-compare/sidebyside/*.png`.

## Visual findings (side-by-side)
- B0 preserves hair, face, tattoo, logo at near-GT sharpness; the cost is the
  blocky green environment (a property of the 7-step environment, not of the
  person).
- MX: silhouette is clean (black background), but hair is visibly softer and
  jagged; boundary detail degraded. The dE00 numbers confirm what is visible.

## Conclusion
The user requirement "foreground detail at least as good as the original" is
**NOT met by the masked pipeline**. The masked reconstruction lost interior
(+22%) and boundary (+31%) color/geometry fidelity relative to the full
scene.

**Decision: proceed with extraction from the full scene (H2), not with the
masked model.** The full scene keeps B0's detail; H2 must separate the
person's Gaussians from the environment. The masked model's value is
retained as: (a) its matte-consistent silhouette as a reference for H3
boundary repair, (b) its 0.004-0.035 alpha-outside leakage as the leakage
target for the extracted person.

## H2 design (in progress)
Cross-view evidence per Gaussian:
- `fg[s]`: accumulated transmittance-weighted in-mask coverage over all 60
  views (visibility T computed sequentially in depth order per chunk).
- `bg[s]`: same with (1 - mask).
- keep if `fg >= 5` and `fg/(fg+bg) >= 0.5`; hide if `fg < 2`; else
  uncertain.
- Output: `work/research/H2-extract/{keep,hide,uncertain}.ply` +
  `evidence.npz` (fg, bg, ffrac, opacity per Gaussian) +
  `work/research/H2-extract.json`.
