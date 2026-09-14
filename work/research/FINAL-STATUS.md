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

## Numeric gates (train-view proxy; held-out renders need EVAL-split train)
- Leakage: B0 6.7% -> winner 0.9% (target <1% MET on proxy).
- Core dE00 / held-out renders: NOT YET (needs user viewer verdict +
  EVAL-split retrain, the honest remaining work).
- Certain-FG recall ~0.996-0.9995, certain-BG leak ~0.0001: no regression.
- Edge contamination: B6 pre-train excess 0.04; v3 gamut clip 0.37 flagged.

## Resume pointers
- Blocked items need user: HF login for SAM3.1, SMPL signup + Drive ckpts.
- Next honest step: EVAL-split (48/12) retrain of winner + held-out
  rendered-alpha metrics, then user viewer verdict on anatomical ghosting.
