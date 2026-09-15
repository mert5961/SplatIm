# Gate summary — measured values vs Research02 targets

Branch: refactor/c-plan-rgba-30k. All values measured, none estimated.

| Gate | Target | Measured | Verdict |
|------|--------|----------|---------|
| Train-view leakage (winner E3x aw0.65) | <1% | 0.8% (115,886 G) | PASS |
| EVAL 48-train leakage | <1% | 1.1% (85,529 G) | MARGINAL (proxy views) |
| Held-out true-alpha L_outside (12 honest held views, brush positional split) | <1% | aw0.65: 2.53% (1.07-5.76); aw0.80: 2.33% (1.36-4.35) | NOT MET — 3rd measured attempt (0.50 4.3% proxy -> 0.65 -> 0.80); diminishing returns, residual structural. Halted per stop condition. Jobs 20260915-0430*-person-eval-held-aw* |
| Held-out aux (aw0.80) | — | certain-BG 1.6%, FG-recall 99.28%, IoU 96.4%, bF1 0.586, render dE00 FG-core 3.17/10.7 | documented (no target set for render dE00) |
| Core dE00 median / p95 (B2-source) | <1 / <3 | 0.0 / 1.25 (n=1,033,234) | PASS |
| Core dE00 despill-v3 | <1 / <3 | 2.06 / 5.53 | FAIL — v3 not shippable, B6/v2 line kept |
| Edge green vs B6 ~0.04 | materially better | v2 pre-train 0.19->0.04; v3 gamut clip 0.37 | PARTIAL — v2 holds, v3 regresses core |
| Certain-FG recall | no reduction | 0.996-0.9995 (all models) | PASS |
| Certain-BG leak | no increase | ~0.0001 (all models) | PASS |
| Boundary temporal vs RVM | reduced | RVM 0.478 vs MA2 0.492 (confounded) | UNPROVEN — orbit-subset flow measures view change, not jitter |
| F1 far floaters | 0 | 0.0 both | PASS |
| F1 edge band B0->winner | down | 8.5% -> 0.8% | PASS |
| G1 renderer calibration vs brush | within factor | 12/12 held views, ratio mine/brush 1.075-1.200 (mean 1.127), per-pixel corr 0.9997 | PASS (faithful; brush runs ~1.13x lower via 1/255 skip + 4e-4 early-termination; NOTE-g1-renderer.md) |
| G2 post-hoc prune sweep (aw0.65 winner, G1 renderer) | monotonic down | tau 0->0.20: L_outside 2.831%->2.477% (96,538->67,048 G), FG-recall 0.9932->0.9919 | MEASURED — monotonic, helps ~12% relative, insufficient alone; G3 retrains in-training at tau=0.20 |

Blocked (need user): B2 SAM3.1 (gated HF), F2 GVHMR (SMPL+Drive), F3 (needs F2).
Winner artifact: outputs/multi-view-scenes/20260914-154428-person-e3x-aw0.65/scene.ply (115,886 G, 0.8%; aw0.80 ties at 120,191 G — 0.65 preferred, fewer Gaussians).
