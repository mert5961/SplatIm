# Response to advisor — B0..B6 experimental results and feedback request

Branch: `refactor/c-plan-rgba-30k` (master untouched at `3a3e33e`;
safety tag/branch `pre-c-plan-2026-09-14` pushed).
Scripts: `work/person_b{1,2,3,4,5_select,6}.py`, diagnostics `work/person_diag.py`.
Metric: outside-mask opacity = fraction of sigmoid(opacity) mass whose
projection falls on mask<=0.5, mean over 25 train views.

## Results

| Exp | Change vs previous | Gaussians | Leakage |
|-----|--------------------|-----------|---------|
| B0 | binary mask, 7k (baseline) | 88,062 | 6.7% |
| B1 | same dataset, 30k, growth_stop 15k | 152,994 | 9.9% |
| B2 | soft-alpha RGBA, NO masks/, match_alpha 0.2, growth_stop 12k, cap 1M | 100,033 | 1.2% |
| B3 | B2 + sparse filtered (kept 18,266 / dropped 50,438; track>=3, fg>=0.6, med>=0.5) | 58,725 | 1.4% |
| B4 | B3 + conservative despill (G-only, edge-weighted) | 58,658 | 1.3% |
| B5 | B2 config on 60/120 views (60 yaw bins, masked-flow best/bin) | 76,149 | 1.2% |
| B6 | B3 + corrected despill v2 (see §3) | 57,997 | 1.3% |

User viewed B2 ("masking successful, spill aside looks good") and B6
(`20260914-132825-person-b6-despillv2/scene.ply`, current best).

## 1. Advisor was right: supervision semantics, not duration

B1 isolates duration: 30k steps densifies (88k -> 153k) but leakage
RISES 6.7% -> 9.9%. Longer mask-mode training grows more background
floaters, it does not clean them. B2 (RGBA + alpha-match branch,
`train.rs:170`) drops leakage to 1.2% at 100k Gaussians despite a 1M cap.
Verified in source: with `masks/` present Brush silently takes the
`total_err * alpha` branch even if the image has its own alpha
(`scene.rs:129-148` copies mask into pixel[3]; `is_masked()` routes the
loss). B2's dataset has NO `masks/` dir — this was load-bearing.

## 2. Sparse filtering halves the model, floor is matte-edge

B3: 55% of the 68,704 COLMAP seeds sit <50% inside the person. Filtering
to 18,266 foreground seeds halves Gaussians (100k -> 59k) with leakage
1.2% -> 1.4% (noise: same matte set, fewer seeds, edge band unchanged).
Conclusion: remaining ~1.3% leakage is the RVM soft-edge band, not seed
contamination. Matches the advisor's P1-not-P0 ranking.

## 3. Despill: found and fixed a real bug, still color-only

Measured: solid interior is clean (green excess ~0.005); spill lives ONLY
in the alpha edge band (orbit end frames: edge excess 0.19-0.20, 93-95%
of edge pixels affected). B4 applied `strength = excess*(...)` then
`G -= strength*excess`, i.e. excess^2 — at fringe excess~0.2 that is only
~20% correction. User confirmed "meh". B6 uses `G -= k*excess` with
`k = min(1, 0.15+1.7*(1-a))` (full correction at a<=0.5, 15% in solid
interior, R/B untouched): pre-train edge excess 0.19 -> 0.04 verified.
Post-train leakage 1.3%, geometry unchanged — as expected, spill was
never geometric. Residual risk: despill happens pre-Brush-premultiply
(RGB stored un-premultiplied, Brush premultiplies in `view_to_sample_image`);
a wrongly-despuilled fringe pixel still supervises color through alpha-match.

## 4. Two roadmap items proven infeasible on this capture

- Fresh dense-window COLMAP: 4s@30fps segment has too little parallax;
  mapper exits 0 with NO model (verified twice). Narrow-baseline studio
  orbit cannot self-calibrate on a short clip.
- Fixed person-centric crop: person fills 99-100% of the 899x1600 frame
  (median box 637x1486, union == full frame on a 360 orbit). There is
  nothing to crop; raising global resolution is the only optical lever.
B5 (angular selection) was the feasible substitute: 60 views, same 1.2%.

## 5. Questions for the advisor

1. Leakage floor ~1.2-1.4% is all matte-edge band. Is SAM2->MatAnyone2
   still the highest-value next step, or would you first try
   `match_alpha_weight` sweep (0.1/0.2/0.5) since B2 used 0.2 untested?
2. B5 shows 60 views == 120 views quality. Does that change your frame
   recommendation, or push harder toward canonicalization (motion, not
   views, is now the binding constraint)?
3. For root canonicalization on this capture (subject rotates ~360° while
   camera orbits): do you recommend HMR2/GVHMR root-yaw absorption into
   COLMAP cameras, or is the residual articulation (arms/hair/clothing)
   large enough to skip straight to a deformable avatar backend?
4. `eval_split_every` exists in Brush loader but was never used; all
   metrics above are train-view leakage. Should the next round add a
   held-out eval split (e.g. every 12th view) before further tuning?
5. Any concern that despill-v2 pre-premultiply interaction (Q above)
   biases the alpha-match term at fringes, or is un-premultiplied +
   Brush-internal-premultiply the correct convention (it matches
   `scene.rs:233` comment "assumes un-premultiplied alpha")?
