# NOTE-H2H3 — Full-scene person extraction (H2 evidence + H3 repair)

Date: 2026-09-15
Follows `NOTE-H0-sharpness.md`: the masked pipeline (E3x) lost 22% interior /
31% boundary dE00 vs the full scene (B0), so extraction is built on the
**full-scene PLY** instead of the masked model.

## H2 — cross-view visibility evidence (`work/h2_extract.py`)

For every Gaussian in `20260914-080603-07ed0a/scene.ply` (428,305),
accumulate over all 60 views (exact TBB transmittance, depth-ordered):

- `fg[s]`: sum of T·mask at the splat center (in-mask visibility)
- `bg[s]`: sum of T·(1−mask)
- `ffrac = fg/(fg+bg)`

Decision (T_VIS=0.15, FG_MIN=5, FG_KEEP=2):

| class | rule | count | fg median | opacity mean |
|---|---|---|---|---|
| keep | fg≥5 ∧ ffrac≥0.5 | **48,838** | 20.5 | 0.237 |
| uncertain | else, fg≥2 | 10,896 | 4.2 | 0.075 |
| hide | fg<2 | 368,571 | 0.0 | 0.125 |

Outputs: `work/research/H2-extract/{keep,hide,uncertain}.ply`,
`evidence.npz` (per-Gaussian fg/bg/ffrac/opacity), `H2-extract.json`.

## H3 — evaluation (`work/h3_eval.py`) + repair (`work/h3_repair.py`)

Same regions as H0 (interior erode 20, band 12, outside dilate 4), 12 held
 | model (12 held views) | interior dE00 | band dE00 | IoU | halo 0–4 px | halo 4–8 px |
 |---|---|---|---|---|---|
 | B0 full scene (vs green GT) | **5.13** | **8.52** | — | n/a (has env) | n/a |
 | MX masked E3x aw0.65 | 6.24 | 11.16 | — | 0.004–0.035 | ~0 |
 | H2 keep.ply (raw) | 6.03 | 10.85 | 0.966 | 0.37 | 0.22 |
 | H2 keep.ply + scale shrink 10 mm cap | 9.26 | 12.55 | **0.455 (broken)** | 0.16 | — |
 | H2 keep-despill.ply (raw) | 6.20 | 11.81 | 0.966 | 0.37 | 0.22 |
 | **H2 keep-despill.ply + matte clip 4 px** | 6.20 | 11.81 | **0.976** | 0 (clipped) | 0 (clipped) |

### Findings
1. **Interior detail ≈ B0** (6.03 vs 5.13): extraction preserves the
   original reconstruction's detail (user requirement met, modulo boundary).
2. **Halo**: the keep set's boundary splats (median 7 mm, p95 34 mm,
   p99 62 mm) spill a soft green-tinted fringe outside the silhouette
   (α-outside 0.456). Root cause: full-scene boundary splats were trained
   against the green background → green bleed + oversized footprints.
   Max fg = 32.6 across all splats ⇒ no always-visible environment splats
   survived; the blobs in raw renders are the person's own edge splats.
3. **Despill** (g′ = g − max(0, g − max(r,b)) on DC-SH base color): 15,005
   of 48,838 splats (31%) had spill, mean 0.015, max 0.63. Removes the
   green hair/edge tint.
4. **Global scale shrink is REJECTED**: capping smean at 10 mm shrinks 36%
   of splats, destroying alpha coverage (IoU 0.966→0.455, interior
   6.03→9.26). Shrinking is only valid as a targeted per-splat boundary
   operation, not global.
5. **Matte clip at composite time** (α ×= dilate(RVM mask, 4 px)) is the
   clean halo fix: standard CG compositing; geometry untouched.

### Deliverable shape (proposed)
- `keep-despill.ply` — the 3D person asset (B0 detail, despilled).
- Compositing rule: render person, clip alpha with per-view RVM matte
  (dilated 4 px), premultiply.
- Optional next (only if clip edge is unsatisfactory): targeted boundary
  splat shrink using per-view silhouette distance, or short refine pass on
  uncertain-band splats with GT color+alpha supervision.
