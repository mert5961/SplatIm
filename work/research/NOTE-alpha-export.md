# True rendered-alpha export — investigated, parked (no Rust toolchain)

Finding: Brush CAN render alpha — B::render_splats returns img with 4th
channel (train.rs:169 slices pred_image[..,..,3..4] as pred_alpha; the
differentiable renderer outputs RGBA). The ONLY blocker is eval_export.rs:
it hard-codes Rgb32FImage (3 channels) and drops alpha on save.

Fix (when a Rust toolchain exists): save Rgba32FImage (or two PNGs:
RGB + alpha16) in eval_save_to_disk, thread through run_eval naming.
Estimate: 15-line change, rebuild brush_app, rerun EVAL job eval pass.

No Rust toolchain on this machine (cargo rc=127); VS Build Tools +
cargo install would take ~30-60 min compile. Parked: proxy 4.3% stands
with limitation logged. Revisit when user wants the true L_outside gate.
