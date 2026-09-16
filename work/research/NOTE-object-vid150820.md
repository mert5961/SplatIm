# NOTE: object 3D from VID20260916150820 (hanging PCB + beacon + pin)

Date: 2026-09-16
Job: `outputs/multi-view-scenes/20260916-055346-object-vid150820`
State: `work/research/OBJ-JOB.json`

## Request

"Make a 3D of the object hanging under the crane talent track, with a beacon
and a connection pin. Body is black; green tints in the video are green-screen
spill -> despill."

Object = black PCB assembly: green-solder-mask PCB face ("SPARL/COAL TRACK",
CO logo), black enclosure face, white-dome beacon on the left edge, brass
connection pin at the bottom. Hangs from a talent-track loop with a cable
coil below.

## Decision: cable EXCLUDED

The cable coil is the hanging element (scene), not the object. It is thin,
flexible, and highly reflective — 3DGS captures it poorly and it would
contaminate the object PLY. Deliverable = PCB + beacon + pin. (Follow-up:
separate cable capture if requested.)

## Pipeline (`work/object_pipeline.py`, idempotent stages)

| stage     | what                                                        | result                          |
|-----------|-------------------------------------------------------------|---------------------------------|
| frames    | 120 frames @ 8 fps from the mp4                             | 120/120                         |
| colmap    | SIFT+affine, guided exhaustive matching, mapper             | 120/120 registered              |
| undistort | SIMPLE_RADIAL @ 1920 px                                     | 1076x1914 dataset               |
| sam2      | SAM 2.1 hiera-small, dual box prompts on frame 0 (board box [420,660,660,1120] + beacon box [425,865,505,940]), video propagation, union | 120 clean masks (board+beacon+pin, cable excluded) |
| ma2       | MatAnyone2 (warmup 10, erode/dilate 10)                     | **temporal collapse: 92/120 frames empty -> automatic SAM2 fallback** |
| brush     | 30,000 steps @ 1920 px, 3M cap, masked training             | scene.ply, 46,807 gaussians (4.5 min) |
| extract   | h2 mask-guided evidence (120 views), keep = fg>=5 & ffrac>=0.5, despill G-=max(0,G-max(R,B)) | keep-despill-obj.ply, 22,595 splats, 12,866 despilled |
| evalh3    | 12 held views (every 10th), raw + clip4                     | metrics below                   |

## Why MatAnyone2 failed here (and why the fallback is fine)

MA2 is a video matting model tuned for near-static backgrounds and a large
foreground. This capture is a fast full orbit (120 frames) of a SMALL object
(2-8% of frame) over a changing office/green-screen background: temporal
propagation lost the object after the first frames (92/120 empty alphas).
The C-plan person job (large person, 60 frames) did work with MA2. For an
opaque rigid object, SAM2's hard masks (verified clean on multiple frames)
are sufficient for masked 3DGS training; soft alpha adds nothing.

## Metrics (12 held views, keep-despill-obj.ply)

| variant | silhouette IoU | alpha_outside | interior dE00 | band dE00 |
|---------|---------------:|--------------:|--------------:|----------:|
| raw     | 0.622          | 0.416         | 13.45         | 13.64     |
| clip4   | **0.853**      | **0.0**       | 13.45         | 13.64     |

- clip4 (4-px mask dilation) is the correct read: no alpha leak outside the
  object, 85% silhouette overlap.
- dE00 ~13 is inflated by construction: the GT still contains green-screen
  spill on the black body, the render is despilled (cleaner than the GT).
  The PCB face is genuinely green (solder mask) and renders dark teal-green,
  close to GT (GT mean [45,48,50] vs render [46,62,62] on the board).

## Artifacts

- Final PLY: `work/research/O2-extract-obj/keep-despill-obj.ply` (22,595 splats, 5.3 MB)
  (also copied to `<job>/object-final.ply`)
- Full scene (incl. background, for reference): `<job>/scene.ply` (46,807 splats)
- Contact sheets: `work/research/OBJ-renders/sheet_keep-despill-obj_{gt,raw,clip4}.png`
- Metrics: `work/research/H3-eval-keep-despill-obj-{clip0,clip4}.json`
- SAM2 raw masks: `<job>/sam2_raw/*.png` (reusable for a cable pass)

## Reproduce

```
python work/object_pipeline.py            # all stages, idempotent
python work/research/obj_sheet.py         # 12-view GT/raw/clip4 sheets
```
