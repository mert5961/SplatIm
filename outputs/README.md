# SHARP is installed and tested

Installed locally in this task's work folder using Python 3.12, PyTorch 2.8.0 with CUDA 12.8, and Apple's pinned SHARP dependencies. The 2.81 GB model is downloaded. Keep the work folder: it contains the environment, source, and checkpoint used by the launcher.

## Convert your photo

### Browser UI

Double-click `Start-SharpSplatUI.ps1`, or run:

```powershell
& 'C:\Users\reality\Documents\Codex\2026-09-11\i-w\outputs\Start-SharpSplatUI.ps1'
```

It opens a local page at `http://127.0.0.1:7860`. Upload a photo, click **Create Gaussian Splat**, and download the `.ply` output. The **View splat** tab loads `.ply`, `.splat`, and `.ksplat` files in a local interactive viewer. Select generated files directly from `outputs\my-splats`; the viewer code is stored locally in `outputs\viewer-assets`, so viewing does not need an Internet connection. Converted files are kept in `outputs\my-splats`.

### Multiple photos into one connected scene

Use the **Multiple photos → one scene** tab. Select overlapping photos of the same stationary scene taken from different positions, then click **Build one connected scene**. Start with Preview to check alignment, then use Standard or High quality for longer training. The same-camera option assumes the camera, focal length/zoom, and image dimensions are consistent; uncheck it for mixed cameras or zoom settings.
#### Using video clips

The same tab accepts video files (`.mp4`, `.mov`, `.m4v`, `.avi`, `.webm`, `.mkv`). Up to 120 frames are sampled uniformly per clip, then treated exactly like uploaded photos: deduplicated, aligned with COLMAP, and trained into one shared scene. For best alignment keep the camera moving slowly and steadily; a handheld walkthrough around a building or object works well. Mixing photos and clips in one job is allowed — uncheck the same-camera option if resolutions differ.

One capture session per job: a clip that jumps between unrelated locations produces disconnected components, and the weak-alignment guard will skip training unless *Train even if alignment is weak* is checked.
#### Command line for a single video

```powershell
& 'C:\Users\reality\Documents\Codex\2026-09-11\i-w\work\sharp-env\Scripts\python.exe' `
  'C:\Users\reality\Documents\Codex\2026-09-11\i-w\work\run_real_video.py' 'C:\path\to\clip.mp4' Preview
```

Pass `true` as the third argument to keep training even when alignment is weak. Frames land in `outputs\multi-view-scenes\<job>\input`, and the finished scene is written next to them as `scene.ply`.

Not every video can become a scene: reconstruction needs a **moving camera around a stationary subject**. A fixed camera filming a moving subject (a person dancing, a robot demo) cannot be reconstructed and now fails with an explicit message.

The workflow uses local COLMAP 4.2.0 to estimate camera positions and Brush 0.3.0 to train a shared Gaussian splat. Photos that cannot connect to the largest reconstructed component are omitted, and the result reports how many were used. This mode performs new scene training, not a merge of independent SHARP outputs.
Hard subjects: portraits of a person (especially against a plain backdrop) are the hardest case for alignment. The pipeline therefore uses affine-shape SIFT with guided matching, which registered 6–8 of 10 such photos where plain SIFT managed only 3. When fewer than 70% of views align, check **Train even if alignment is weak** to build the scene from the largest connected subset.

Quality presets: Preview = 1,000 steps at up to 1,024 px; Standard = 7,000 steps at up to 1,600 px; High quality = 30,000 steps at up to 1,920 px. Training is capped at 3 million Gaussians. Converted images, camera reconstruction, run log, report, and final scene.ply are kept in outputs/multi-view-scenes. The stop button preserves partial work; automatic resume is not provided.

Verified on 20 images from COLMAP's public South Building dataset: 20/20 cameras aligned and a valid 5,447-Gaussian preview PLY was exported after 1,000 training steps. This is a functional smoke test, not a final-quality benchmark. The dataset source is https://demuc.de/colmap/datasets/.

### Single-image PowerShell command

Open PowerShell and run (replace the example image path):

```powershell
& 'C:\Users\reality\Documents\Codex\2026-09-11\i-w\outputs\Convert-ImageToSplat.ps1' -InputPath 'C:\path\to\photo.jpg'
```

You can also pass a folder of images. Results default to the my-splats folder beside the launcher. Use -OutputPath to choose another output folder. Inference uses the local GPU and downloaded checkpoint.

## Verification

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB.
- CUDA matrix computation passed.
- Full inference and PLY export passed using Apple's included teaser.jpg.
- Output: sample-splat/teaser.ply, 1,179,648 Gaussians, 66,061,086 bytes.
- All exported numeric attributes are finite.
- pip check passed.
- The local interactive viewer starts successfully in Chrome.
- SHARP's trajectory-video renderer (`sharp render`) compiles gsplat's CUDA
  extension via ninja + MSVC + the local CUDA 12.8 toolkit. The required
  compilers are already installed on this host (CMake 4.4.3, ninja 1.13.0
  inside `work\sharp-env\Scripts`, MSVC 14.44 via the VS 2022 BuildTools at
  `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`, nvcc
  12.8). Because gsplat 1.5.3 builds with `-Wno-attributes` (a GCC/clang
  flag that MSVC rejects with D8021), drop that flag by editing the
  installed copy of `work\sharp-env\Lib\site-packages\gsplat\cuda\_backend.py`
  line 177:
  `extra_cflags = [opt_level, "-Wno-attributes"]` -> `extra_cflags = [opt_level]`.
  After the one-time JIT build, the trajectory video renders to `*.mp4`
  beside the input PLY.
- Sample photo lacks focal-length EXIF, so SHARP used its default 30 mm estimate.

Source and documentation: https://github.com/apple/ml-sharp
Code/model licenses are in work/ml-sharp/LICENSE and LICENSE_MODEL.
