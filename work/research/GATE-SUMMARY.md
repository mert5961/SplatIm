# Gate summary — measured values vs Research02 targets

Branch: refactor/c-plan-rgba-30k. All values measured, none estimated.

| Gate | Target | Measured | Verdict |
|------|--------|----------|---------|
| Train-view leakage (winner E3x aw0.65) | <1% | 0.8% (115,886 G) | PASS |
| EVAL 48-train leakage | <1% | 1.1% (85,529 G) | MARGINAL (proxy views) |
| Held-out rendered L_outside | <1% | true alpha = 2.38% (n=10 views) | NOT MET (true alpha reduces from RGB proxy 4.3% to 2.38%, still above 1% target) |
| Core dE00 median / p95 (B2-source) | <1 / <3 | 0.0 / 1.25 (n=1,033,234) | PASS |
| Core dE00 despill-v3 | <1 / <3 | 2.06 / 5.53 | FAIL — v3 not shippable, B6/v2 line kept |
| Edge green vs B6 ~0.04 | materially better | v2 pre-train 0.19->0.04; v3 gamut clip 0.37 | PARTIAL — v2 holds, v3 regresses core |
| Certain-FG recall | no reduction | 0.996-0.9995 (all models) | PASS |
| Certain-BG leak | no increase | ~0.0001 (all models) | PASS |
| Boundary temporal vs RVM | reduced | RVM 0.478 vs MA2 0.492 (confounded) | UNPROVEN — orbit-subset flow measures view change, not jitter |
| F1 far floaters | 0 | 0.0 both | PASS |
| F1 edge band B0->winner | down | 8.5% -> 0.8% | PASS |

Blocked (need user): B2 SAM3.1 (gated HF), F2 GVHMR (SMPL+Drive), F3 (needs F2).
Winner artifact: outputs/multi-view-scenes/20260914-154428-person-e3x-aw0.65/scene.ply (115,886 G, 0.8%; aw0.80 ties at 120,191 G — 0.65 preferred, fewer Gaussians).
