# RGBA alpha export — landed (parked → green)

Alpha-export fix was parked 2026-09-14 (`852981c` "no Rust toolchain").
Rust toolchain present (cargo 1.98.1, rustc 1.98.1); brush built release.

Patch (2 files, 28+/10-, 78 lines):
- `crates/brush-process/src/eval_export.rs`: accept c=4, quantize f32→u8,
  save as RGBA PNG (u16 not used; PNG encoder only accepts u8/u16).
- `crates/brush-train/src/eval.rs`: keep `render_full = img[..,..,0..4]`
  in `EvalSample.rendered` (was: render_rgb was 3-channel slice, alpha dropped).

Patch preserved at /tmp/brush-rgba.patch (78 lines).

Build & install:
- Built release: `work/brush-source/target/release/brush_app.exe` (155 MB).
- Installed to `work/multiview-tools/brush/brush_app.exe`.
- Pre-built alpha-quiet version backed up as
  `work/multiview-tools/brush/brush_app.exe.pre-c-plan-rgba` (158 MB).

Eval-split rerun at aw0.65 with new binary:
- Job `20260915-035046-person-eval-aw065-rgba`.
- Dataset = 48-train-view subset (NOT the held-out set — see correction below).
- 30k steps, growth_stop 12k, cap 1M, match_alpha_weight=0.65,
  eval every 5000, eval_split_every=5 (no eval views remain in 48-view set;
  the "eval" views written are drawn from the 48 train views).
- 10 of 12 eval PNGs exported at step 30000 (2 lost to export race).

Measured on those 10 48-train-view evals (rendered alpha vs RVM-mask outside):
mean 0.0238, min 0.0108, max 0.0456.

CORRECTION (20260915): this is a TRAIN-view leakage number, not the held-out
L_outside. The true held-out run (full-60 dataset, brush positional
eval split = 12 honest held views) is `work/person_eval_held.py`; its
result replaces this in GATE-SUMMARY. Artifact `HELDOUT-true-alpha.json`
retained and relabelled: `work/research/HELDOUT-true-alpha.json` ->
rename to `TRAINVIEW-true-alpha-aw065.json`.

> The <1% held-out target is judged by person_eval_held.py output only.

Artifact (relabelled): `work/research/TRAINVIEW-true-alpha-aw065.json`
(formerly HELDOUT-true-alpha.json — mislabel corrected).
