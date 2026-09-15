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
- 48 train + 12 brush-held views, 30k steps, growth_stop 12k, cap 1M,
  match_alpha_weight=0.65, eval every 5000, eval_split_every=5.
- 10 of 12 eval PNGs exported at step 30000 (2 lost to export race).

TRUE held-out L_outside (rendered alpha vs RVM-mask outside region):
| Stat | Value |
|------|-------|
| n_views | 10 |
| mean | 0.0238 |
| min | 0.0108 |
| max | 0.0456 |

> <1% target NOT MET on true alpha (mean 2.38%). The earlier 4.3% RGB-energy
> proxy was overestimating leakage; truth is ~half of that (better) but still
> above target.

Artifact: `work/research/HELDOUT-true-alpha.json`.

GATE-SUMMARY update: L_outside row swaps "RGB proxy 4.3%" → "true alpha 2.38%".
Status changes from "NOT MET" to "TRUE METRIC > 1% TARGET".
