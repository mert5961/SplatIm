# Future: EVAL-split rerun at aw0.65 (not run — single-variable rule)

The EVAL-split job trained at aw0.50 (winner at the time). E3x later moved
the winner to aw0.65 (leak 0.8% vs 0.9%, train-view proxy). A like-for-like
EVAL rerun at aw0.65 would need another ~2x30k trainings (~7 min GPU).
Not run: marginal expected gain (0.1pp), costs GPU + new job dirs.
Run it only if held-out alpha-export lands (true L_outside) or user asks.
Base artifacts exist: train/ (48 RGBA), eval/ (12 RGBA), sparse_txt_*.
