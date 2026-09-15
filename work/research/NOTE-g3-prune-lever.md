# G3 — in-training prune_opacity lever (single-variable, documented dead end at <1%)

Objective gate: held-out true-alpha L_outside < 1% (12 honest held views,
brush positional eval split, aw0.65 base, brush's own eval).

Baseline (held-gate winner, no prune): L = 2.535% (96,538 G).

## G2 — post-hoc prune sweep (already-pruned PLY, G1 renderer)

tau 0 -> 0.20: L 2.831% -> 2.477% (96,538 -> 67,048 G). Monotonic, ~12%
relative, insufficient. Post-hoc cannot reach 1%.

## G3 — in-training prune sweep (brush retrained with --prune-opacity tau)

Brush patch (work/brush-source, gitignored; /tmp/brush-prune.patch):
TrainConfig.prune_opacity f32 (default 0.0078 = built-in 2/255) replaces
the hard-coded MIN_OPACITY in the refine prune mask. Binary installed at
work/multiview-tools/brush/brush_app.exe (backups: .pre-prune-flag,
.pre-c-plan-rgba).

| tau   | L_outside | n_G   | certain-BG | FG-recall | IoU    | dE00 med/p95 |
|-------|-----------|-------|------------|-----------|--------|--------------|
| 0.0078| 2.535%    | 96,538| 1.86%      | 0.9926    | 0.9628 | 3.16 / 10.63 |
| 0.20  | 1.44%     | 52,589| 0.77%      | 0.9921    | 0.9707 | 3.12 / 10.69 |
| 0.30  | 1.42%     | 33,057| 0.75%      | 0.9917    | 0.9710 | 3.10 / 10.68 |
| 0.40  | 1.43%     | 20,556| 0.76%      | 0.9915    | 0.9706 | 3.11 / 10.63 |

All jobs: 30k steps, growth-stop 12k, aw 0.65, 1600px, same dataset
(20260914-151822-person-evalsplit/full). Runner: work/person_eval_held_prune.py
(env PRUNE_TAU, AW). ~3 min each.

## Findings

1. In-training pruning amplifies the post-hoc effect ~3.5x: same tau=0.20
   gives 1.44% (in-training) vs 2.477% (post-hoc). Mechanism: low-opacity
   splats are continuously pruned and re-sampled from high-opacity parents
   during growth/refine, so the ghost population never accumulates.
2. Plateau at tau >= 0.20: 1.42-1.44% across tau 0.20/0.30/0.40. The
   residual is NOT low-opacity ghosts (o<0.5 fraction: 23.6% / 9.6% / 2.4%
   at 0.20/0.30/0.40) — it is high-opacity splats that legitimately belong
   to the person but protrude past the 2D RVM matte in unobserved held
   viewpoints (reconstruction/occlusion error, e.g. arm behind body).
3. Guards hold across the sweep: FG-recall >= 0.9915, certain-BG leak
   drops 1.86% -> 0.75%, IoU improves 0.963 -> 0.971, dE00 FG-core
   unchanged (~3.1 / ~10.7). Boundary F1 drops 0.583 -> 0.53 (edges get
   crisper matte, softer gradient).
4. Dead end: the gate <1% is unreachable with prune_opacity alone.
   Extrapolation from the plateau (flat 0.20->0.40) rules out higher tau;
   the remaining ~1.4% comes from high-opacity splats that pruning cannot
   touch without destroying the person (FG-recall would collapse).

## Recommendation

tau=0.30 (33,057 G, L 1.42%): best measured L and dE00, 66% fewer
Gaussians than baseline, all guards >= baseline except boundary F1
(0.583 -> 0.533, cosmetic). tau=0.40 (20,556 G) is within noise on L and
dE00 but prunes 78% — less robust to dataset/scene changes.

Best artifact: outputs/multi-view-scenes/20260915-072116-person-eval-held-aw0.65-prune0.30/scene.ply
(job dir gitignored; metrics copied to work/research/HELD-aw065-prune0.30-metrics.json).

## Next lever (out of scope for this goal)

The residual needs a structural prior, not opacity pruning: depth/matte
supervision quality (E2 MA2 matte vs RVM on held views), or per-view
matte-consistency regularization. That is a second variable — excluded
by the single-variable stop rule.
