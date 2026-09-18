# SplatIm

SplatIm is a Windows desktop tool for turning photos and moving-camera videos into 3D Gaussian splats. It brings quick single-photo generation, multi-view scene reconstruction, subject selection from video, and an interactive splat viewer into one local browser interface.

> This is an active prototype. The current setup is tied to a Windows workstation with an NVIDIA GPU and locally installed model/tool environments. A clean-machine installer is not available yet.

## What it can do

| Workflow | Input | Result |
| --- | --- | --- |
| Create splat | One photo, or one frame from a video | A fast SHARP Gaussian splat estimate |
| Video → selected subject | An orbit video around a mostly stationary subject | A subject-focused splat, with inspectable masks and extraction stages |
| Multiple photos → one scene | Overlapping photos or video frames of one scene | One connected multi-view Gaussian scene |
| View splat | `.ply`, `.splat`, or `.ksplat` | An interactive 3D viewer |

The video-subject workflow exposes the processing stages: COLMAP camera alignment, brush-based target selection, SAM 2.1 mask tracking, optional MatAnyone2 alpha refinement, Brush training, and mask-guided Gaussian extraction. Each stage has its own previews, status, and logs so mask problems can be found before spending time on training.

## Launch on the configured workstation

Run `outputs/Start-SharpSplatUI.ps1` from PowerShell or double-click it in File Explorer. The local interface opens at [http://127.0.0.1:7869](http://127.0.0.1:7869).

The launcher currently uses the Python environment and model/tool installations under `work/`. Those large dependencies and model weights are intentionally not included in this Git repository, so cloning the source alone is not enough to run the full application.

## Capture tips

- For a person or object splat, move the camera around a mostly stationary subject. A fixed camera filming a moving person does not provide the camera motion needed for 3D reconstruction.
- Keep neighboring views overlapping and move smoothly. Avoid motion blur, abrupt cuts, and large lighting changes.
- In the subject workflow, inspect the propagated masks before training. Missing foreground can remove parts of the subject; background included in the mask can create stray Gaussians.
- The quick single-photo workflow estimates unseen viewpoints. Use the multi-view workflows when you need actual camera coverage around the subject or scene.

## Project layout

- `outputs/SharpSplatUI.py` — Gradio application and workflow panels
- `outputs/VideoSubjectSplat.py` — video subject reconstruction pipeline
- `outputs/MultiViewSplat.py` — multi-view scene reconstruction
- `outputs/SplatViewer.html` and `outputs/viewer-assets/` — local 3D splat viewer
- `outputs/README.md` — fuller workflow notes and local command examples
- `work/` — research, helper scripts, and locally installed dependencies; not a portable installation bundle

## Upstream projects

SplatIm builds on [Apple SHARP](https://github.com/apple/ml-sharp) for single-image splat generation, [COLMAP](https://github.com/colmap/colmap) for camera reconstruction, [SAM 2](https://github.com/facebookresearch/sam2) for video mask tracking, [MatAnyone2](https://github.com/pq-yang/MatAnyone2) for optional alpha refinement, and [Brush](https://github.com/VAST-AI-Research/Brush) for Gaussian scene training. Refer to each upstream project for its license and model terms.
