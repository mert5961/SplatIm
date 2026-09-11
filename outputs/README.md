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

The workflow uses local COLMAP 4.2.0 to estimate camera positions and Brush 0.3.0 to train a shared Gaussian splat. Photos that cannot connect to the largest reconstructed component are omitted, and the result reports how many were used. This mode performs new scene training, not a merge of independent SHARP outputs.

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
- The local interactive viewer starts successfully in Chrome. SHARP's optional trajectory-video renderer requires additional CUDA build tooling and was not installed.
- Sample photo lacks focal-length EXIF, so SHARP used its default 30 mm estimate.

Source and documentation: https://github.com/apple/ml-sharp
Code/model licenses are in work/ml-sharp/LICENSE and LICENSE_MODEL.
