# Optimization branch

## Progress display

`outputs/ProgressUI.py` is a presentation adapter around the existing pipeline functions. It forwards their results unchanged, observes existing progress callbacks and run logs, and sends status updates to the active stage's panel.

- SAM 2 displays the reported tracked-frame count.
- Extraction reads the current run's camera counters from its existing log.
- Long phases without detailed telemetry display an animated activity bar and elapsed time. This is intentionally not an estimated completion percentage.
- Logs are read as they grow. Old run counters are excluded.
- Completed or failed jobs clear the activity bar. A disconnected browser cannot cause two adapter workers to run backend stages concurrently.

The progress-display change does not alter model commands or extraction. The later accumulated-keyframe fix below changes how the selection workflow stores and supplies SAM prompts.

## Accumulated selection corrections

In **2 · Select + track**, edited frames save when you switch frames or propagate. **Save keyframe** also pins an unchanged prediction. The panel lists saved frames and the frames used by the last successful propagation. Simply viewing an unedited frame does not add a constraint.

All saved prompts are registered in one fresh SAM inference state before tracking. Interior strokes remain positive-point prompts; edits to a predicted silhouette become full-mask prompts. Full-mask corrections are preserved exactly on their authored frames, including empty masks marking an absent/occluded target. This does not guarantee perfect tracking between keyframes.

Keyframes persist in `subject-keyframes/manifest.json` with separate PNG revisions, independently of generated `sam2_raw` masks. Removing a keyframe removes its active manifest entry but keeps its previous PNG revisions on disk. `sam2-keyframes-used.json` records the frames and revision used. Pending saved changes block training/refinement until propagation runs again.

The editor uses the source frame's aspect ratio and native canvas size, with PNG transport. Frame changes work through mouse, keyboard, and number input. The displayed frame identity travels with the editor payload so queued actions cannot silently relabel a prior frame's brush strokes.

After a page reload, use **1 · Prepare → Resume a prepared video → Resume saved frames + keyframes** to continue without rerunning camera alignment. Legacy `prompt-frame-*.png` debug files are not automatically imported: older runs did not reliably record accumulated keyframes or recover unsaved strokes.

**MatAnyone2 limitation:** the optional refinement integration still seeds from frame zero only; it does not enforce later corrected keyframes. To train using the corrected SAM mask sequence, skip refinement. This change does not add SeC or change the matting model.

Checks include persistence across six frame edits, prompt-mode retention, absent-target masks, wrong-frame rejection, exact authored-mask preservation, stale-sequence protection, and resuming. `tests/keyframe_smoke.py <job>` runs actual SAM on a temporary eight-frame copy with six keyframes; `--serve` exposes the full app as an isolated browser test on port 7872. Neither modifies the source job.

## Optional uncertain-splat recovery

Open **Video → selected subject → 5 · Extract + results → Recover useful uncertain splats**. Select a completed job, or choose the current session after extracting it.

The minimum score controls how conservative recovery is. Start at **0.90**. The maximum-added setting limits the number of additions; it does not force lower-scoring splats into the output. Set it to zero to export only the original accepted splats.

The score combines:

- 55% support from the Gaussian's projected footprint across all registered mask views, using 25 weighted samples per footprint;
- 20% proximity to reliable accepted geometry;
- 15% color agreement with nearby accepted splats;
- 10% accumulated foreground evidence from the existing extraction.

Candidates must also pass minimum mask-support, neighborhood, and observation gates. Added splats have their long axes capped against nearby reliable geometry to reduce large streaks. Scores are heuristic support values, not calibrated probabilities. They do not replace a visibility-aware reconstruction or repair geometry that was never captured.

Only the uncertain partition is eligible. Original accepted splats are retained byte-for-byte, and rejected background splats are not recovered. The tool verifies that the original kept PLY matches the saved scene/evidence before proceeding. It never changes the original scene, masks, extraction, or selected-subject PLY.

Each settings combination writes under `outputs/video-subject-jobs/<job>/subject-recovery/`:

- a new `*-recovered-subject.ply`;
- `recovered-only.ply`, showing just the additions;
- a report with counts, parameters and source hash;
- CSV scores and a detailed NPZ evidence file.

The UI shows the top 50 candidates and whether they were added. Full scores are available in the CSV.

## Yunus comparison

The inspected job was `20260918-063615-YUNUS-4a7848`. Its scene had 79,190 splats: 27,470 accepted, 10,396 uncertain and 41,324 rejected. The uncertain pass contains purple shirt material together with extensive background haze. Adding the entire pass is not supported by the comparison.

At a score threshold of 0.90, recovery adds 66 splats. Six registered views were rendered at 960 pixels high without silhouette clipping, using the same renderer for both files:

| Measurement | Original | Recovery |
| --- | ---: | ---: |
| Splat count | 27,470 | 27,536 |
| Mean interior opacity | 98.9278% | 98.9922% |
| Mean opacity outside the expanded mask | 1.7092% | 1.7186% |
| Silhouette IoU | 0.957549 | 0.957430 |

This is a small opacity increase with a small spill increase, not proof that the back-view transparency is fixed. These are sampled training views and DC-color renders, not held-out ground truth or a perceptual-quality benchmark. Lower thresholds recover more material but increased haze in the initial comparison. The original export remains the reference.

To reproduce the comparisons from the repository root:

```powershell
& .\work\sharp-env\Scripts\python.exe work\compare_subject_recovery.py `
  outputs\video-subject-jobs\20260918-063615-YUNUS-4a7848 --score .9 --quick
& .\work\sharp-env\Scripts\python.exe work\compare_subject_recovery.py `
  outputs\video-subject-jobs\20260918-063615-YUNUS-4a7848 --score .9 --viewer
```

The second command compares the viewer's default and opposite-orbit camera angles without claiming a ground-truth metric for those novel views. In that comparison, the conservative recovery looks very similar to the original; the opposite view still has sparse, wispy leg geometry. It should not be treated as a completed fix for novel-view transparency.

## Checks

```powershell
& .\work\sharp-env\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover streamed progress, error completion, stale-log exclusion, disconnected workers, score thresholds, count caps, partition boundaries and preservation of accepted attributes. `tests/progress_preview.py` provides a browser fixture that runs no models; it was used to verify visible frame-count updates and an idle unrelated panel.
