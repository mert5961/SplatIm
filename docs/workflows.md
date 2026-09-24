# SplatIm workflow guide

[Back to the project overview](../README.md)

> [!NOTE]
> These instructions describe the configured workstation. It uses Python 3.12, PyTorch 2.8.0 with CUDA 12.8, and Apple's pinned SHARP dependencies. The local `work/` directory contains the environment, upstream tools, and model checkpoint; those machine-specific dependencies are not stored in Git.

## Launch the application

From the repository root, double-click `outputs\Start-SharpSplatUI.ps1`, or run:

```powershell
& '.\outputs\Start-SharpSplatUI.ps1'
```

The browser interface opens at [http://127.0.0.1:7869](http://127.0.0.1:7869).

## Create a splat from one photo

Use **Create splat** for a fast single-image estimate.

1. Upload a photo. A video is also accepted, but this workflow uses only its middle frame.
2. Leave **Person only** enabled to remove the surroundings and crop around the detected person. Disable it to keep the complete image as a scene.
3. Review the cutout preview and click **Create Gaussian Splat**.
4. Download the resulting `.ply` file or open it in **View splat**.

Generated files remain local under `outputs\my-splats`.

### PowerShell command

```powershell
& '.\outputs\Convert-ImageToSplat.ps1' -InputPath 'C:\path\to\photo.jpg'
```

The command also accepts a directory of images. Use `-OutputPath` to choose a destination other than the default `outputs\my-splats` directory.

## Extract one subject from an orbit video

Use **Video → selected subject** when the camera moves around a mostly stationary person or object. Unlike the quick single-image workflow, this pipeline reconstructs camera geometry from multiple aligned views.

1. Upload the video, choose a quality preset, and click **Prepare and align video**. The app samples up to 120 frames, reconstructs camera positions with COLMAP, and displays the registered undistorted frames.
2. Choose any registered frame and paint a rough stroke inside the subject. Click **Propagate selection with SAM 2.1**.
3. Inspect every overlay and raw mask. The galleries report coverage, centroid movement, and large jumps. To correct a frame, erase false areas or paint missing areas on its red mask layer, then propagate again.
4. Optionally run **MatAnyone2** and compare its separate overlay and mask galleries with the SAM 2.1 result. The most recently accepted mask source is used for training.
5. Train the masked scene with Brush, then run mask-guided Gaussian extraction. Keep the intermediate scene for diagnosis; the final `*-selected-subject.ply` contains the accepted subject Gaussians.

Every job keeps its frames, prompts, masks, metrics, COLMAP data, training inputs, intermediate scene, extraction evidence, reports, and command log under `outputs\video-subject-jobs`. That entire directory is runtime data and is ignored by Git.

### Recover uncertain splats

After extraction, **Recover useful uncertain splats** can score uncertain Gaussians without retraining.

1. Choose a saved job.
2. Start with a minimum support score of **0.90**.
3. Set the maximum number of additions.
4. Compare the recovery result from the same camera angle as the original.

Lower thresholds can restore more subject material but can also add haze. The recovery operation creates a separate PLY plus a score table, CSV, report, and added-only PLY. See the [optimization notes](optimization.md) for measurements and limitations.

### Capture requirement

Camera motion is mandatory. A fixed camera recording a moving subject—or a turntable where only the object rotates—does not provide the geometry this pipeline needs. Use a slow orbit with strong overlap and keep the subject as still as practical.

## Build one connected multi-view scene

Use **Multiple photos → one scene** for overlapping images of the same stationary scene.

1. Select photos taken from different camera positions.
2. Use Preview quality to check alignment.
3. Move to Standard or High quality for longer training once the alignment is sound.
4. Keep **Same camera** enabled only when camera, focal length, zoom, and image dimensions are consistent.

The pipeline uses COLMAP 4.2.0 for camera estimation and Brush 0.3.0 for Gaussian training. Images outside the largest reconstructed component are omitted. This is new scene training, not a merge of independent SHARP outputs.

Portraits against plain backgrounds are particularly difficult to align. The pipeline uses affine-shape SIFT with guided matching; when fewer than 70% of views align, **Train even if alignment is weak** can train from the largest connected subset.

For green-screen captures, enable **Ignore green screen background**. SplatIm creates one mask per undistorted image for Brush. If no green background is detected, training proceeds without those masks.

### Use video clips as multi-view input

The same tab accepts `.mp4`, `.mov`, `.m4v`, `.avi`, `.webm`, and `.mkv` files. It samples up to 120 frames per clip and treats them like uploaded photos. Mixing photos and clips is supported; disable **Same camera** when their dimensions or camera properties differ.

A clip that jumps between unrelated places creates disconnected components and is rejected unless **Train even if alignment is weak** is enabled.

### Command line for one video

```powershell
& '.\work\sharp-env\Scripts\python.exe' `
  '.\work\run_real_video.py' 'C:\path\to\clip.mp4' Preview
```

Pass `true` as the third argument to train even when alignment is weak. The sampled frames and scene stay under `outputs\multi-view-scenes\<job>`.

### Quality presets

| Preset | Training steps | Maximum image size |
| --- | ---: | ---: |
| Preview | 1,000 | 1,024 px |
| Standard | 7,000 | 1,600 px |
| High quality | 30,000 | 1,920 px |

Training is capped at three million Gaussians. The stop button preserves partial work; automatic resume is not provided.

## View a splat

The **View splat** tab opens `.ply`, `.splat`, and `.ksplat` files in a local interactive viewer. Its browser modules are stored in `outputs\viewer-assets`, so viewing does not require an internet connection.

## Workstation verification

The configured machine has passed the following checks:

- NVIDIA RTX PRO 6000 Blackwell Workstation Edition with 96 GB VRAM.
- PyTorch CUDA matrix computation.
- Full SHARP inference and finite-attribute PLY export.
- `pip check` and local viewer startup.
- A COLMAP South Building smoke test with all 20 input cameras aligned.
- SHARP trajectory rendering through gsplat's CUDA extension.

The trajectory renderer uses CMake 4.4.3, ninja 1.13.0, MSVC 14.44, and CUDA 12.8. On this machine, gsplat 1.5.3 requires removing the unsupported `-Wno-attributes` flag from its installed `_backend.py` before the first JIT build.

See [Apple SHARP](https://github.com/apple/ml-sharp) for upstream source and documentation. Code and model licenses are available in the local `work/ml-sharp/` checkout.

## Runtime artifact policy

Generated images, masks, videos, models, arrays, logs, caches, and 3D outputs must remain in ignored runtime directories. Only intentional documentation assets belong under `docs/assets/`; see the [documentation asset policy](assets/README.md).
