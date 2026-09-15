# G1 — Python/CUDA held-view alpha renderer (calibrated)

Status: COMPLETE (calibrated 12/12 held views).

## What
`work/splat_alpha_torch.py` — GPU (torch) order-independent alpha compositor:
`alpha(px) = 1 - prod_i (1 - o_i * G_i(px))`, accumulated as `sum log1p(-alpha_i)`
via `index_add_` over ragged per-Gaussian bboxes, chunked by a 4e8-pixel budget.
Per-Gaussian 2D covariance from the standard 3DGS projection
(`cov2d = J Sigma_cam J^T + 0.3 I`), conic = inverse, pixel evaluated at
`pixel-left + frac(center-left)` (matches brush sampling within <1px).

Loaders shared with `work/splat_alpha_render.py` (COLMAP PINHOLE wxyz
quaternions, PLY with `+0.3` blur in `gauss_cov3d`).

## Bug found and fixed (this note)
First renderer build under-reported held-view L_outside by 2.4-4.8x
(ratio mine/brush 0.31-0.65). Root cause: in `render_alpha_gpu`, the
chunk-local index was computed as `within = arange(n) - off` where `off`
is the chunk's global start offset — double subtraction. Chunk 0 (off=0)
was correct; every later chunk's `within` shifted negative by `off`, so
its pixels fell out of bounds and were silently dropped. Full-image
renders use 6 chunks (2.16e9 bbox px for 101k splats, 9974 splats with
radius>100) → most splats lost. Fix: `within = torch.arange(n, device=DEV)`
(one-line). Verified on a 3-Gaussian synthetic case with forced 3-chunk
budgeting (maxdiff 0.0 before/after fix) and on real data.

## Calibration vs brush (the reference)
Reference: brush's own eval renders, job
`outputs/multi-view-scenes/20260915-043336-person-eval-held-aw0.80`
(final 30k splats, 12 honest held views, RGBA PNGs from the quantized
4-channel eval path; metrics.json verified derived from those PNGs).

Per-view L_outside (RVM-mask outside), mine/brush ratio:

| view  | mine    | brush   | ratio |
|-------|---------|---------|-------|
| 00001 | 0.02623 | 0.02355 | 1.114 |
| 00009 | 0.03057 | 0.02719 | 1.124 |
| 00016 | 0.03267 | 0.02723 | 1.200 |
| 00026 | 0.01982 | 0.01745 | 1.136 |
| 00039 | 0.02562 | 0.02170 | 1.181 |
| 00048 | 0.04792 | 0.04346 | 1.103 |
| 00056 | 0.02899 | 0.02436 | 1.190 |
| 00067 | 0.01528 | 0.01418 | 1.077 |
| 00079 | 0.01738 | 0.01525 | 1.139 |
| 00088 | 0.02612 | 0.02430 | 1.075 |
| 00100 | 0.02933 | 0.02681 | 1.094 |
| 00109 | 0.01487 | 0.01358 | 1.095 |

Mean ratio 1.127, range 1.075-1.200, all 12 views. Per-pixel map
correlation 0.9997 (00001).

## Bias direction and cause
Mine is consistently ~8-20% HIGHER than brush. That is expected and
documented, not a residual bug:
- brush rasterizer skips per-splat contributions with alpha < 1/255
  (rasterize.wgsl) — I include all;
- brush early-terminates a pixel when accumulated T < 4e-4;
- brush rasterizes in f32 with atomic add.
I am the faithful full-composition reference; brush is slightly below it.
For G2/G3 the metric is relative (before/after prune, same renderer),
so a constant ~1.1x bias against brush cancels. Absolute comparison to
brush numbers: divide by ~1.13.

## Speed
~4.3 s/view at 899x1600 (12 views in 51 s, RTX 5090, f32, 6 chunks).

## Superseded artifacts (buggy renderer, do not use)
- `work/research/G1-12view-compare.json` (ratios 0.31-0.46 — pre-fix)
- `work/research/G1-render-calib-2view.json` (pre-fix)
Kept for the record of the debugging arc.

## Verified during debugging (all pass)
- per-splat data (u, v, cov2d, o) matches standard 3DGS formulas in f64
  (cov3d from PLY scale/quat, camera from sparse_txt_full, identical to
  the dataset BIN model used by brush);
- accumulation matches float64 brute force (tiny 3-Gaussian case,
  maxdiff <= 0.0005 pre-fix, 0.0 post-fix at all budget settings);
- culling identical to brush (no det<0.01 splats exist; near-plane
  difference touches 16 out-of-frame splats only);
- cameras.txt (TXT) == dataset full/sparse BIN model (focal, principal
  point, quats, tvecs identical);
- metrics.json of the reference job is derived from its eval_30000 PNGs
  (per-view L matches to 5 decimals).
