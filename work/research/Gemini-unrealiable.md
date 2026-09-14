To resolve the artifacts in the person-only 3DGS pipeline outlined in `session-summary-2026-09-14.md`, here is the expanded, step-by-step implementation guide to overhaul the point-cloud initialization, temporal selection, and matting stages.

## 1. Eradicating "Dead Noise" from Background Seeds

Because Brush weights the loss by the mask, background seed points initialized from the 68,704 sparse COLMAP points receive zero gradient and remain permanently frozen. You must either filter them before training or penalize them during training.

### Method A: Pre-Filtering the COLMAP Point Cloud (Immediate Python Fix)

Use COLMAP's provided `read_write_model.py` to project and prune the sparse cloud before it reaches Brush.

1. **Load Camera Data:** Read `cameras.bin`, `images.bin`, and `points3D.bin` from `outputs/multi-view-scenes/<job>/dataset/sparse`.


2. **Project & Test:** For each 3D point $(X, Y, Z)$, project it into the 2D image space $(u, v)$ for every image where it is registered.
3. **Mask Validation:** Check the pixel value at $(u, v)$ in the corresponding hardened mask in `dataset/masks/`.


4. **Prune:** If the point does not land on a white pixel (value > 0) in at least 2 distinct camera views, delete it.
5. **Re-save:** Write the filtered points back to `points3D.bin`. Brush will now initialize with zero background seeds.

### Method B: Active Opacity Penalization in Brush (Rust Library Modification)

If you prefer to fix this at the renderer level, modify `work/brush-source/crates/brush-train/src/train.rs`. Around line 166, instead of purely ignoring masked pixels, actively penalize the opacity of splats that render outside the mask.

In the Rust training loop, introduce an inverse mask loss applied to the rendered alpha channel:

```rust
// Current logic (approximate):
// let color_loss = (L1 + SSIM) * mask;

// Grounded Fix:
let mask_inv = 1.0 - mask; // Black areas become 1.0
let alpha_render = render_result.alpha; 

// Penalize any opacity that appears in the inverted mask region
let background_penalty = mask_inv * alpha_render;
let penalty_weight = 0.05; // Tune based on step count

let total_loss = color_loss + (background_penalty * penalty_weight);

```

## 2. Managing Non-Rigid Motion and Signal Quality

A 74-second full 360° orbit with a human subject guarantees non-rigid micro-movements. 3DGS assumes a perfectly static scene; motion creates smearing.

### Surgical Temporal Cropping

Do not sample 120 frames uniformly over 74 seconds (1 frame per 0.62s). Find the most static continuous 3-to-5 second window in `VID20260914174253.mp4`.

1. Identify a phase where the subject is perfectly still (e.g., from 12.0s to 15.0s).
2. Extract dense frames for only this segment using `imageio-ffmpeg` or a raw FFmpeg CLI:


```bash
ffmpeg -ss 00:00:12.000 -i C:\Users\reality\Downloads\VID20260914174253.mp4 -t 00:00:03.000 -r 30 -q:v 2 frames/frame_%04d.jpg

```


*(Note: This yields ~90 highly correlated frames. The orbit angle will be narrow, but the geometry will be razor-sharp).*

### Shift to Dynamic Frameworks (For Full Orbit Captures)

If capturing the entire 360° orbit is non-negotiable, standard 3DGS will fail. Given your NVIDIA RTX PRO 6000 Blackwell with 96 GB VRAM, you have immense headroom to run dynamic, memory-intensive state-of-the-art models.

* **Implement Scaffold-GS:** This model utilizes voxel grids to anchor Gaussians, mitigating floating artifacts caused by slight subject motion. Clone the repository and replace Brush as the training backend for `person-only` jobs.



## 3. Upgrading Matting & Color Space

The transition to a binary mask using a threshold ($\geq96$) causes boundary noise due to the lack of temporal consistency in RVM and green spill from the studio stage.

### Implementing SAM2 for Temporal Consistency

Replace the resnet50 RVM implementation in `work/matte_person.py` with Meta's **Segment Anything 2 (SAM2)** to guarantee jitter-free silhouettes.

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

# Initialize SAM2 Predictor (uses a fraction of your 96GB VRAM)
predictor = build_sam2_video_predictor(
    "sam2_hiera_large.yaml", 
    "sam2_hiera_large.pt", 
    device="cuda"
)

# Initialize video state on the extracted frames directory
inference_state = predictor.init_state(video_path="outputs/frames/")

# Provide a positive point prompt on the subject (e.g., the black Zero Density polo) on frame 0
predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=[[1000, 500]], # approximate center of subject
    labels=[1],
)

# Propagate temporally across all frames to get perfectly consistent masks
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    # Threshold logits to binary mask and save
    video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()

```

### Mathematical Green Despill Pre-processing

Green spill on the subject's skin and arms confuses the Spherical Harmonics (SH) calculation because the color changes depending on the camera's angle to the green screen. Normalize the RGB values before feature extraction (COLMAP).

Inject this rapid OpenCV despill pass into your Python pipeline orchestration before Step 2:

```python
import cv2
import numpy as np
import glob

def despill_green(image_path):
    img = cv2.imread(image_path)
    B, G, R = cv2.split(img)
    
    # Cap the Green channel at the maximum of Red or Blue
    max_RB = np.maximum(R, B)
    G_clean = np.minimum(G, max_RB)
    
    clean_img = cv2.merge([B, G_clean, R])
    cv2.imwrite(image_path, clean_img)

for img_file in glob.glob("outputs/frames/*.jpg"):
    despill_green(img_file)

```