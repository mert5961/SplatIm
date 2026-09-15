# Final status — Research02 epics A-F (branch refactor/c-plan-rgba-30k)

Winner: E3x aw0.65 — 115,886 Gaussians, train-view leakage 0.8%
(`outputs/multi-view-scenes/20260914-154428-person-e3x-aw0.65/scene.ply`;
aw0.80 ties at 0.8%/120,191 G — 0.65 preferred).
B6 frozen at work/research/FREEZE-B6-manifest.json. Master untouched.

## Epic completion
- A1-A4 done (freeze, eval v2 + synthetic proof, trimap set).
- B1 done (SAM2.1 IoU 0.99 vs RVM, no swap). B2 BLOCKED (gated HF repo).
  B3 done (Cutie fwd-rev IoU 0.92, no jumps).
- C1 done (MA2 isolated env). C2 done (Sapiens2 3-way compare).
  C3 deferred to E2 gate by plan.
- D1+D2 done (plate rmse 0.072, chroma key). D3 done (despill v3,
  gamut 0.37 flagged — edge-only residual).
- E1 done (hybrid + uncertainty U=0.104). E2 done (MA2 primary;
  key rejected: miss 0.28/leak 0.14). E3 done (0.50 wins).
  E4 done (1920 == 1600, no gain).
- F1 done (edge 8.5%->0.9%, far 0, highU 0.8%->0.5%; no anatomical signal).
  F2 BLOCKED (SMPL signup + Drive ckpts need user). F3 parked (needs F2).

## Numeric gates
- Train-view leakage (proxy): B0 6.7% -> winner 0.8% (target <1% MET on proxy).
- Held-out TRUE-alpha leakage (full-60, brush positional eval split,
  12 honest held views, `work/person_eval_held.py`):
  alpha-weight series: aw0.65 2.53%, aw0.80 2.33% (diminishing returns).
  Structural lever (single-variable): in-training prune_opacity sweep
  (G3, `work/person_eval_held_prune.py`, brush --prune-opacity):
  tau 0.0078->0.20->0.30->0.40 gives L 2.54%->1.44%->1.42%->1.43%
  (96,538->52,589->33,057->20,556 G). Plateau at tau>=0.20: residual is
  high-opacity splats protruding past the 2D matte in held views, not
  low-opacity ghosts — unreachable by opacity pruning. Gate <1% NOT MET
  (best 1.42% at tau=0.30); documented single-variable dead end
  (NOTE-g3-prune-lever.md). Guards hold: FG-recall >=0.9915, certain-BG
  1.86->0.75%, dE00 ~3.1/10.7, IoU 0.963->0.971.
  Artifacts: outputs/multi-view-scenes/20260915-0{430,715,721,724}*-person-eval-held-*
  (job dirs gitignored; metrics in work/research/HELD-*-metrics.json).
- Held-out auxiliaries (aw0.80): certain-BG leak 1.6%, certain-FG recall
  99.28%, IoU 96.4%, boundary F1 0.586, render-vs-GT dE00 (FG core)
  median 3.17 / p95 10.7 — held-view render quality documented.
- Core dE00 (supervision vs B6 source): PASS (0.0 / 1.25, n=1,033,234).
- Certain-FG recall ~0.996-0.9995, certain-BG leak ~0.0001 (train view):
  no regression. Edge contamination: B6 pre-train excess 0.04; v3 gamut
  clip 0.37 flagged.

## Resume pointers
- Held-out alpha gate: best single-variable result 1.42% (prune tau=0.30,
  33,057 G) vs target <1%. Dead end documented (NOTE-g3-prune-lever.md);
  further improvement needs a second variable (depth/matte supervision
  quality), out of scope per single-variable stop rule.
- Remaining user-gated: HF login for SAM3.1 (B2), SMPL signup + Drive
  ckpts (F2), F3 needs F2.
