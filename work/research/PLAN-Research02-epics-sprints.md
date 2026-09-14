# Plan-board — Research02 multi-evidence front end

Source: `work/research/Research02-chatgpt.md` (§23 order, §24 safety, §25 gates).
Branch: `refactor/c-plan-rgba-30k`. Never touch master. One variable per ablation.
Commit + push every green task; checkpoint = push.

## Sprint 1 — Freeze + evaluation foundation (EPIC A)

- [ ] **A1 Freeze B6.** Manifest: branch, config, PLY hash, frame manifest
  (B3 120-view set), COLMAP model id, RVM ckpt hash, input/output hashes,
  torch/CUDA versions, metrics.json (leakage 1.3%, 57,997 Gaussians),
  timing.json. Do not mutate B6 after.
- [ ] **A2 person_eval_v2.** 48-train/12-held-out angular split (stratified
  orbit, not naive every-12th); rendered accumulated-alpha metric
  `L_outside`, IoU(A_r>τ,M), boundary F-score; JSON/CSV reports.
- [ ] **A3 Synthetic tests.** C=αF+(1−α)B generator (green gradients, soft
  edges, hair strands, black cloth, spill on F); MAE(F̂,F)<1e-4 on
  noise-free α-sufficient pixels; |ΔY|<1e-6 unclipped despill. Prove color
  space handling before touching real frames.
- [ ] **A4 Trimap set.** 12–16 frames over 360° (front/back/profile/hair/
  arms/worst-spill/early distractor-woman frames). Certain FG / certain BG /
  unknown only — no fake fractional ground truth.

Gate: A2+A3 green before any model integration.

## Sprint 2 — Semantic identity (EPIC B)

- [ ] **B1 SAM 2.1 baseline.** Promptable video tracking, box on target
  (NOT text "person" — distractor woman). Save raw masks/logits. Apache-2.0.
- [ ] **B2 SAM 3.1 identity.** Exemplar/box target select + track. Custom SAM
  license — registry only, no ship decision.
- [ ] **B3 Cutie cross-check.** Independent VOS memory; forward+reverse pass
  disagreement = swap detector. Report identity-swap frames.

Gate: identity stable on early distractor frames, else stop (no matting yet).

## Sprint 3 — Temporal alpha (EPIC C)

- [ ] **C1 MatAnyone2.** Isolated py3.10 env (subprocess/file protocol, main
  venv stays 3.12). First-frame SAM mask in, per-frame float32 alpha out
  on the B5 60-view set. No MQE dependency (unreleased) — own quality layer.
  NTU S-Lab = research-only until cleared.
- [ ] **C2 Sapiens2-1B matting.** Per-frame alpha + premultiplied foreground
  (verify tensor convention in code first). Temporal authority = NO;
  boundary/foreground authority = YES. Disagreement vs MA2 = confidence.
- [ ] **C3 μMatting (optional).** Benchmark only if C1/C2 leave edge quality
  open. 1080×1920 source fits its high-res design.

Rule: run each independently, measure, no averaging. Trimap from B:
erode(M,r)=FG, outside dilate(M,2r)=BG, r=round(0.0025·min(H,W)).

## Sprint 4 — Green-screen physics (EPIC D)

- [ ] **D1 screen_plate.py.** Certain-screen pixels (outside dilated SAM AND
  matte α<0.02 AND green-chroma likely) → smooth local plate B(x,y)
  (poly → TPS). Outputs plate_rgb, plate_validity, plate_fit_error.
- [ ] **D2 chroma_key.py.** CbCr Mahalanobis vs local screen stats,
  smoothstep(d0,d1) from measured quantiles; low confidence when Y<Ymin
  (black polo). Independent evidence, NOT the matte.
- [ ] **D3 foreground_recover.py + despill_v3.py.** Linear light (ffprobe
  transfer metadata first, log assumption if missing). Unmix THEN despill:
  F̂=(C−(1−α)B)/max(α,ε) only α≥0.7; blend 0.1–0.7; propagate <0.1.
  Despill in luminance-preserving chroma subspace (lᵀs=0). Gamut report.

Gate §25: core ΔE00 median <1, p95 <3; edge contamination beats B6 ~0.04.

## Sprint 5 — Consensus + production 3D (EPIC E)

- [ ] **E1 Regional hybrid.** Certain FG/BG from SAM+key agreement (α→1/0);
  unknown band = MA2/μMatting fractional; disagreement = keep primary +
  high uncertainty (never blind-average). Emit alpha+confidence per frame.
- [ ] **E2 2D benchmark.** RVM vs MA2 vs Sapiens2 vs key on §14 metrics
  (core miss, BG leak, temporal MAE, IoU, boundary F, disagreement,
  key-contradiction, spill, ΔE, clipping, entropy). Pick primary matte.
- [ ] **E3 Alpha-weight ablation.** 60 views, filtered sparse, 30k:
  match_alpha ∈ {0.10, 0.20, 0.35, 0.50}. Held-out views decide.
- [ ] **E4 1600 vs 1920.** Only on winner. No upsampling above source.

## Sprint 6 — Motion gate (EPIC F)

- [ ] **F1 Classify.** Double shoulders / arm trails / hair duplication =
  anatomical → F2. Random floaters → stay in EPIC E loop.
- [ ] **F2 GVHMR canonicalization.** World-grounded root motion → absorb
  yaw into cameras → static Brush retry.
- [ ] **F3 Deformable branch.** HUGS/GauHuman ONLY if F2 fails, separate
  branch, never mixed into Brush production path.

## Standing rules (§24, condensed)

One variable per ablation. After every edit: diff → syntax check →
smallest test → synthetic test if math. NaN/Inf, dim mismatch, or
name mismatch = hard fail. Never silently premultiply, retime color,
resize masks apart from RGB, touch intrinsics, threshold soft alpha,
swap checkpoints, or renormalize alpha. Log every assumption.
Visual "looks good" ≠ pass without metric gates.

## License registry (needs product review before ship)

SAM2 Apache-2.0 · SAM3 custom · MatAnyone2 S-Lab research-only ·
Sapiens2 custom · GauHuman S-Lab · HUGS check. Track code vs weight
licenses separately in model registry.
