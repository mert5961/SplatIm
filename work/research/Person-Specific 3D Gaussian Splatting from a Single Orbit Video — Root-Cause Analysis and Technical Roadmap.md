# Person-Specific 3D Gaussian Splatting from a Single Orbit Video  
## Root-Cause Analysis, Solution Architecture, and Experimental Roadmap

## Executive Summary

The current system attempts to reconstruct a person-specific 3D Gaussian Splat from a 74.1-second monocular video containing an approximately 360° camera orbit. The pipeline first reconstructs the complete scene and then attempts to isolate the person as a standalone foreground asset.

The existing pipeline is technically solid:

- COLMAP successfully registers all 120 sampled frames.
- The reconstruction contains approximately 68,704 sparse 3D points.
- Brush 0.3.0 is used as the 3D Gaussian Splatting backend.
- RVM-based video matting has already been integrated.
- The workstation uses an NVIDIA RTX PRO 6000 Blackwell with 96 GB VRAM.

The current failure mode is visible noise, floaters, and/or unstable geometry around the person in the third person-only reconstruction.

The most important conclusion is that the original hypothesis—

> Background COLMAP seed points receive zero gradients in masked regions, remain permanently frozen, and therefore can never be pruned.

—is only partially correct.

Background sparse points can certainly degrade initialization and consume model capacity. However, Brush does not simply leave all invisible Gaussians permanently untouched. Its training system includes opacity and scale regularization, refinement passes, low-opacity pruning, and MCMC-style redistribution mechanisms.

A more important problem is the training schedule.

The current reconstruction terminates after only **7,000 optimization steps**, while the standard Brush training regime is approximately **30,000 steps**, with Gaussian growth/densification continuing until roughly the middle of training.

The current training therefore ends while the representation is still in its growth phase rather than after a proper consolidation and cleanup phase.

The second major problem is the distinction between **masking** and **alpha supervision**.

The current `masks/` mechanism primarily tells Brush:

> Ignore this region when computing the photometric reconstruction loss.

It does not necessarily tell the model:

> The rendered opacity must be zero here.

For a standalone foreground asset, the latter is what is actually required.

The highest-priority change is therefore:

**Replace mask-only training with true soft-alpha RGBA supervision and train through a complete densification + stabilization schedule.**

The recommended order of intervention is:

| Priority | Intervention | Expected Value | Engineering Cost |
|---|---|---|---|
| P0 | Replace `masks/` training with true **soft-alpha RGBA input** | Directly supervises foreground transparency rather than merely ignoring the background | Very low |
| P0 | Increase training to approximately **25k–30k steps**, stopping growth around 10k–15k | Allows post-growth refinement and pruning | Very low |
| P1 | Replace or supplement RVM with **MatAnyone 2**, potentially initialized using SAM 2 | Better temporal consistency and target-person assignment | Low–medium |
| P1 | Filter COLMAP sparse points using foreground-mask reprojection support | Prevents environment geometry from entering person-only initialization | Low |
| P1 | Introduce a fixed person-centric crop and update camera intrinsics | Allocates substantially more effective resolution to the person | Medium |
| P2 | Use motion-aware, angular-coverage-aware frame selection | Reduces human-motion inconsistency while maintaining 360° coverage | Medium |
| P2 | Introduce **subject-centric root-motion canonicalization** | Converts much of the person's global rotation into camera motion | Medium–high |
| P3 | Move to GauHuman, 3DGS-Avatar, HUGS, or a 4DGS family | Correct model class if articulated human motion remains dominant | High |

The recommended immediate experiment is therefore:

**Soft target alpha → RGBA frames → Brush → 30k optimization steps**

using the existing COLMAP solution.

Only after this experiment should sparse-cloud filtering and more complex dynamic-human methods be introduced.

---

# 1. Root-Cause Analysis

## 1.1 Separate Camera Pose Estimation from Person Reconstruction

The fact that COLMAP registers 120/120 frames is a strong indication that the static studio environment provides sufficient features for reliable camera calibration.

However, this introduces an important architectural distinction:

### Static environment

Useful for:

- feature matching,
- camera pose estimation,
- bundle adjustment,
- global scale consistency.

### Human subject

Useful for:

- foreground reconstruction,
- alpha supervision,
- subject-specific Gaussian initialization.

These two tasks should not use identical geometric support.

The preferred architecture is therefore:

```text
Static studio/background
        │
        ▼
COLMAP camera reconstruction
        │
        ├──────── Camera intrinsics/extrinsics
        │
Person segmentation / alpha matting
        │
        ▼
Foreground-only Gaussian initialization
        │
        ▼
Person-only 3DGS training
```

This is preferable to performing COLMAP entirely on foreground-only images, because the background currently provides reliable static geometry for camera calibration.

A future refinement would be to explicitly mask the human and other moving people during COLMAP feature extraction, allowing COLMAP to solve poses exclusively from the static environment.

---

# 2. Training Schedule Is Probably a Major Failure Source

The current Brush run uses:

```text
total_steps = 7000
```

This is substantially shorter than the typical 3DGS optimization schedule.

If Gaussian growth is still enabled when optimization terminates, the representation never receives a sufficiently long post-densification period to:

- stabilize Gaussian locations,
- reduce opacity of incorrect primitives,
- prune weak Gaussians,
- consolidate duplicate geometry,
- improve anisotropic covariance,
- settle SH/color parameters.

In other words:

```text
growth
████████████████████████████
        training stops here ↑
```

instead of:

```text
growth
██████████████
              │
              ▼
refinement / settling / pruning
████████████████████
```

For the next experiment, use approximately:

```text
total_steps       = 30000
growth_stop_iter  = 12000–15000
refine_every      = 200
max_resolution    = 1600
max_splats        = 1,000,000 initially
ssim_weight       = 0.2
lpips_weight      = 0
```

`growth_stop_iter = 12000` should be treated as an experimental starting point rather than a universal optimum.

A second run can use the default value around 15,000.

The key requirement is that the training continues for a significant period **after densification has stopped**.

---

# 3. Masking and Alpha Supervision Are Not Equivalent

This distinction is critical.

The existing person-only workflow uses:

```text
dataset/
    images/
    masks/
```

This approach primarily suppresses image-space reconstruction loss outside the foreground mask.

Conceptually:

\[
L =
M(x,y)
\cdot
L_{RGB}
\]

Therefore, when:

\[
M(x,y)=0
\]

the model receives little or no direct photometric evidence that the rendered opacity itself must also become zero.

For foreground asset reconstruction, the correct target is instead:

\[
\alpha_{rendered}
\approx
\alpha_{target}
\]

including:

\[
\alpha_{target}=0
\]

outside the person.

This is a fundamentally stronger constraint.

The recommended dataset representation is therefore:

```text
dataset/
    images/
        frame0001.png   # RGBA
        frame0002.png
        ...
```

with **no separate `masks/` directory for the final person-only optimization**.

The alpha channel should contain the soft person matte.

The RGB channels should contain the foreground color, preferably after conservative green-spill removal.

---

# 4. Foreground Loss Normalization

The current masked reconstruction loss is effectively similar to:

\[
L_{\text{current}}
=
\frac{1}{3HW}
\sum_{x,y,c}
M_{xy}E_{xyc}
\]

This means that when the person occupies only 30% of an image, the useful photometric gradient is substantially smaller than when the person occupies 80%.

A foreground-normalized formulation would instead use:

\[
L_{\text{fg}}
=
\frac{
\sum_{x,y,c} M_{xy} E_{xyc}
}{
3\sum_{x,y}M_{xy}+\epsilon
}
\]

This removes foreground-area dependency.

If Brush mask mode is retained for other applications, modifying its masked loss normalization would therefore be reasonable.

Conceptually:

```rust
let weighted = total_err * alpha_input.clone();

let mask_mass =
    alpha_input.sum() * channels as f32 + EPSILON;

let loss =
    weighted.sum() / mask_mass;
```

However, for the current person-only reconstruction, real RGBA supervision should be tested **before modifying Brush source code**.

Changing the loss normalization also changes the relative magnitude between photometric and auxiliary losses, so opacity/scale regularization weights may subsequently need retuning.

---

# 5. Revised Interpretation of the Background Seed Hypothesis

The original theory was:

> Environment seeds are in masked areas, receive zero gradient, remain frozen, and never disappear.

The corrected interpretation is:

> Environment COLMAP points are undesirable person-only initialization samples and can consume optimization capacity, but they are not necessarily permanently frozen. Brush has mechanisms capable of removing low-opacity or otherwise undesirable Gaussians. The larger issue is that mask-only supervision does not explicitly force the background opacity to zero, and the current 7k run terminates too early for the complete cleanup process.

Sparse-point filtering is therefore valuable, but it should be considered a **P1 optimization**, not the first fix.

True alpha supervision operates throughout training, whereas sparse filtering only improves initialization.

---

# 6. Sparse COLMAP Foreground Filtering

A robust person-only initialization can still be constructed from the existing COLMAP reconstruction.

For every COLMAP `Point3D`, inspect every image observation belonging to its track.

For each observation:

1. Obtain the associated image.
2. Obtain the corresponding 2D feature coordinates.
3. Sample the foreground alpha/mask at that location.
4. Compute how consistently the track lies inside the person.

Recommended initial rule:

```python
for point in reconstruction.points3D.values():

    alpha_values = []

    for obs in point.track.elements:

        image = reconstruction.images[obs.image_id]

        xy = image.points2D[
            obs.point2D_idx
        ].xy

        alpha = sample_alpha(
            alpha_map[image.name],
            xy
        )

        alpha_values.append(alpha)

    track_length = len(alpha_values)

    foreground_ratio = mean(
        a > 0.50
        for a in alpha_values
    )

    median_alpha = median(alpha_values)

    keep = (
        track_length >= 3
        and foreground_ratio >= 0.60
        and median_alpha >= 0.50
    )

    if not keep:
        reconstruction.delete_point3D(point.id)
```

Initial thresholds:

```text
minimum track length : 3
foreground support   : 0.60
median alpha         : 0.50
```

These are engineering starting points, not literature-standard values.

A useful ablation is:

```text
foreground support ∈ {0.40, 0.60, 0.80}
```

A more conservative version should sample an **eroded foreground matte** rather than the raw matte.

For example:

```text
core foreground     alpha > 0.8
uncertain boundary  0.2–0.8
background          alpha < 0.2
```

Keep points only when their multi-view track is strongly supported by core foreground pixels.

This avoids initializing Gaussians from unstable hair/matte boundaries.

### Important diagnostic

The number of points surviving this filter is itself informative.

If only a very small fraction of COLMAP points survive, that suggests that the moving person was not reliably triangulated under the static-scene SfM assumption.

That would support moving toward:

- alpha-supervised free Gaussian optimization,
- body-prior initialization,
- canonicalized human reconstruction.

---

# 7. Matting Upgrade

## Current RVM approach

RVM provides:

- recurrent temporal processing,
- soft alpha,
- good general human matting.

It remains a useful baseline.

However, the capture contains another person during the early part of the video.

This creates a subject-identity problem.

### Better target-person workflow

Recommended:

```text
Selected clean frame
      │
      ▼
SAM 2 / manual target-person mask
      │
      ▼
MatAnyone 2
      │
      ▼
Target-specific temporally coherent alpha sequence
```

SAM 2 should not be treated as the final matting algorithm because segmentation masks do not provide high-quality fractional alpha around:

- hair,
- fingers,
- clothing edges,
- motion-blurred boundaries.

Its primary role here is **target initialization and tracking identity**.

MatAnyone / MatAnyone 2 are more directly aligned with the required problem because they produce temporally consistent soft alpha mattes.

---

# 8. Avoid Hard Binary Mattes for Final Training

The current hardened mask uses approximately:

```text
threshold >= 96
largest connected component
2-pixel feather
```

This may suppress some background leakage, but it destroys fractional coverage around thin structures.

Examples:

- hair,
- ears,
- fingers,
- shirt edges,
- anti-aliased silhouettes.

For alpha-supervised Gaussian reconstruction, preserve soft alpha whenever possible.

Use hard/eroded masks for:

- seed filtering,
- diagnostic support,
- COLMAP foreground/background classification.

Use soft alpha for:

- final RGBA training,
- silhouette supervision,
- edge-quality reconstruction.

---

# 9. Green-Screen Despill

Even with perfect geometry, green spill produces cross-view appearance inconsistency.

This is particularly visible on:

- skin,
- hair,
- shoulders,
- reflective materials,
- black clothing edges.

The preprocessing should therefore include conservative alpha-aware despill.

Conceptually:

```python
fg = image.copy()

spill = detect_green_dominance(fg)

strength = (
    spill
    * alpha
    * boundary_confidence
)

fg = remove_green_component(
    fg,
    strength
)
```

Despill should not aggressively modify all foreground pixels.

The strongest correction should occur near semi-transparent matte boundaries.

A useful implementation principle is:

```text
high alpha + low green dominance
    → almost no correction

edge alpha + strong green dominance
    → stronger correction
```

The despilled RGB and the original RGB should be retained so the impact can be evaluated independently.

---

# 10. Person-Centric Cropping

Increasing the global training resolution from 1600 to 2048 or 4K is not the first optimization to make.

The person occupies only part of each frame.

A more efficient solution is to crop the dataset around a fixed subject-centered region while maintaining correct camera intrinsics.

If a crop begins at:

\[
(x_0,y_0)
\]

then:

\[
c'_x = c_x - x_0
\]

\[
c'_y = c_y - y_0
\]

while:

\[
f'_x=f_x
\]

\[
f'_y=f_y
\]

before resizing.

If the cropped image is subsequently scaled by factor \(s\):

\[
f'_x = sf_x
\]

\[
f'_y = sf_y
\]

\[
c'_x = s(c_x-x_0)
\]

\[
c'_y = s(c_y-y_0)
\]

This effectively increases the subject's pixels-per-meter without wasting resolution on studio walls and ceiling.

A fixed crop across views is preferable to independently moving crops because it preserves a simpler and more stable camera model.

---

# 11. Motion-Aware Frame Selection

Simply reducing 120 frames to a contiguous 20–40 frame time segment is not necessarily correct.

Doing so may destroy angular coverage.

Instead, frame selection should optimize both:

- camera viewpoint coverage,
- human-motion stability.

A useful score is:

\[
S_i =
w_a C_{\text{angular}}
-
w_m M_{\text{motion}}
-
w_b B_{\text{blur}}
-
w_e E_{\text{matte}}
\]

where:

- \(C_{\text{angular}}\) = contribution to camera-angle coverage,
- \(M_{\text{motion}}\) = estimated body-motion magnitude,
- \(B_{\text{blur}}\) = motion/defocus blur,
- \(E_{\text{matte}}\) = matting uncertainty or temporal inconsistency.

For example:

```text
360° / 6° ≈ 60 angular bins
```

Select the best frame from each angular bin.

This produces approximately 60 views while preserving full 360° coverage.

### Human-motion score

Possible sources:

- HMR2 / 4DHumans joint motion,
- GVHMR,
- optical flow restricted to the human mask,
- keypoint displacement,
- mask-shape change.

### Blur score

Possible metrics:

- Laplacian variance,
- high-frequency energy,
- edge-strength statistics.

### Matte quality

Possible metrics:

- boundary instability,
- frame-to-frame alpha residual after optical-flow warping,
- segmentation confidence.

---

# 12. The Fundamental Static-Scene Limitation

Standard 3DGS assumes a static scene.

The current capture violates this assumption because both:

- the camera moves,
- the person slowly rotates and changes pose.

Camera motion is valid.

Person motion is not.

If the person's world-space position at time \(t\) is:

\[
X_t
\]

standard static 3DGS implicitly assumes:

\[
X_t = X
\]

for every frame.

For articulated human motion:

\[
X_t \neq X
\]

Therefore the optimizer attempts to explain inconsistent observations by:

- duplicating Gaussians,
- increasing covariance,
- creating semi-transparent geometry,
- producing ghost contours,
- generating floaters.

This is especially visible around:

- arms,
- shoulders,
- hair,
- hands,
- shirt edges.

If these artifacts remain after correct RGBA supervision and full training, additional optimization steps will not fundamentally solve the problem.

At that point the problem should be classified as **dynamic-human reconstruction**, not insufficient static 3DGS training.

---

# 13. Subject-Centric Root Canonicalization

Before replacing the entire system with a deformable avatar pipeline, there is a lighter intermediate approach.

Estimate the global/root transform of the person in each frame and absorb it into the camera trajectory.

Let:

\[
T_{cw,t}
\]

be the COLMAP world-to-camera transform.

Let:

\[
T_{wp,t}
\]

be the canonical-person-to-world transform.

Then define:

\[
T_{cp,t}
=
T_{cw,t}
T_{wp,t}
\]

The resulting camera trajectory describes the scene relative to a canonicalized person.

In practical terms:

```text
Original interpretation

camera moves
person also rotates

        ↓

Canonical interpretation

person remains approximately fixed
camera incorporates both motions
```

This can substantially reduce errors caused by global body yaw and translation.

Residual articulation remains:

- arms,
- head,
- clothing,
- posture.

But the largest rigid component of human motion can be removed.

### Proposed pipeline

```text
RGB frames
   │
   ├──────────────► COLMAP
   │                    │
   │                    ▼
   │                 T_cw[t]
   │
   └──────────────► HMR2 / GVHMR
                        │
                        ├─ root rotation
                        ├─ root translation
                        └─ body pose
                               │
                               ▼
                         T_wp[t]
                               │
                               ▼
               T_cp[t] = T_cw[t] @ T_wp[t]
                               │
                               ▼
                   canonical subject cameras
                               │
                               ▼
                     static RGBA 3DGS
```

HMR coordinates should not simply be assumed to match COLMAP coordinates.

A robust implementation should fit the estimated body to each COLMAP camera using:

- 2D keypoints,
- PnP,
- reprojection optimization,
- temporal smoothing.

This proposal is a lightweight engineering bridge between standard static 3DGS and fully deformable human Gaussian methods.

---

# 14. Human-Specific Gaussian Alternatives

If root canonicalization is insufficient, the representation itself must model deformation.

Strong relevant approaches include:

## GauHuman

Uses:

- canonical Gaussian representation,
- SMPL/body priors,
- Linear Blend Skinning,
- articulated deformation.

Best suited when the objective becomes a controllable human avatar rather than merely a static isolated splat.

## 3DGS-Avatar

Adds non-rigid deformation to Gaussian primitives.

Relevant because it explicitly targets monocular videos containing moving humans.

## HUGS

Separates:

- static environment,
- animatable human representation.

Conceptually very relevant to the current problem because the original video contains exactly this combination.

## GaussianAvatar / SplattingAvatar

Useful when a mesh/body representation is acceptable and animation is a requirement.

## Modern 4DGS methods

2025–2026 work increasingly represents:

\[
G = G(x,y,z,t)
\]

rather than:

\[
G = G(x,y,z)
\]

These are appropriate when the desired output is a dynamic performance capture.

They are less attractive if the production requirement remains:

> Export one conventional static person `.ply`.

---

# 15. Recommended Backend Strategy

## Primary production backend

Continue using **Brush** initially.

Reasons:

- already integrated,
- working on Windows,
- existing UI,
- existing COLMAP pipeline,
- RGBA support,
- acceptable deployment output,
- low engineering cost.

## Research / secondary backend

Add:

**Nerfstudio Splatfacto + gsplat**

if further experimentation becomes necessary.

Python makes it substantially easier to prototype:

- new losses,
- foreground-normalized losses,
- alpha regularization,
- custom pruning rules,
- diagnostics,
- visibility tests,
- custom initialization.

A second backend is particularly useful for determining whether remaining artifacts originate from:

```text
dataset / capture limitations
```

or:

```text
Brush-specific training behavior
```

---

# 16. Recommended Production-Candidate Pipeline

```text
Input Orbit Video
        │
        ▼
Frame Extraction
        │
        ├───────────────────────┐
        │                       │
        ▼                       ▼
COLMAP camera reconstruction   Target-person segmentation
        │                       │
        │                   SAM 2 / manual seed
        │                       │
        │                       ▼
        │                 MatAnyone 2
        │                       │
        │                       ▼
        │                  Soft alpha
        │                       │
        │                       ▼
        │                 Temporal QC
        │                       │
        │                       ▼
        │                Green despill
        │                       │
        │                       ▼
        │                   RGBA PNG
        │                       │
        ├──────────────┐        │
        │              │        │
        ▼              │        │
Sparse-point           │        │
foreground filter      │        │
        │              │        │
        └──────────────┴────────┘
                       │
                       ▼
             Person-centric dataset
                       │
                       ▼
                    Brush
               ~30k optimization
                       │
                       ▼
               Person-only 3DGS
                       │
                       ▼
           Quantitative evaluation
```

The separate `masks/` directory should not be used for the final alpha-supervised model.

---

# 17. Initial Brush Configuration

Recommended first high-quality reconstruction:

```text
total_steps         = 30000
growth_stop_iter    = 12000
refine_every        = 200
max_resolution      = 1600
max_splats          = 1000000
ssim_weight         = 0.2
lpips_loss_weight   = 0
match_alpha_weight  = 0.2
```

Treat:

```text
match_alpha_weight = 0.2
```

as an experimental value.

Recommended sweep:

```text
0.1
0.2
0.5
```

Do not immediately increase to 2048+ resolution.

First determine whether:

- correct alpha semantics,
- sufficient optimization length,
- correct initialization,

remove the structural noise.

Resolution should be optimized afterwards.

---

# 18. Experimental Matrix

A controlled ablation sequence is essential.

| Experiment | Foreground Representation | Steps | Sparse Init | Frames | Canonicalization | Purpose |
|---|---|---:|---|---|---|---|
| B0 | Binary `masks/` | 7k | Full | 120 | No | Existing baseline |
| B1 | Same binary mask | 30k | Full | 120 | No | Isolate training-duration effect |
| B2 | **Soft RGBA alpha** | 30k | Full | 120 | No | Measure alpha-supervision effect |
| B3 | Soft RGBA | 30k | **Foreground-filtered** | 120 | No | Measure initialization contamination |
| B4 | MatAnyone 2 RGBA | 30k | Filtered | 120 | No | Measure matte-quality effect |
| B5 | MatAnyone 2 RGBA | 30k | Filtered | ~60 optimized | No | Measure motion/frame-selection effect |
| B6 | Same | 30k | Filtered | ~60 | **Root canonicalized** | Measure global human-motion effect |
| B7 | B6 dataset | Equivalent | Same | Same | Same | Cross-check using gsplat/Splatfacto |

The most informative immediate experiments are:

```text
B1
B2
B3
```

Do not change all variables simultaneously.

---

# 19. Diagnostics and Metrics

Conventional PSNR alone is insufficient because the most important failure is **foreground isolation**.

Recommended metrics:

| Metric | Purpose |
|---|---|
| Foreground PSNR | Appearance reconstruction |
| Foreground SSIM | Structural fidelity |
| Foreground LPIPS | Perceptual detail |
| Alpha IoU | Coarse silhouette accuracy |
| Alpha SAD / MAD | Soft-alpha reconstruction |
| Boundary-band error | Hair/clothing/edge quality |
| Outside-alpha leakage | Measures background/floater opacity |
| Detached component count | Measures visible floating geometry |
| Gaussian count | Representation complexity |
| PLY size | Deployment cost |
| Render FPS | Runtime suitability |

### Outside-alpha leakage

A particularly useful metric is:

\[
L_{out}
=
\frac{
\sum_p
\alpha_{render}(p)
[1-M_{GT}(p)]
}{
\sum_p
\alpha_{render}(p)+\epsilon
}
\]

This answers:

> What fraction of rendered opacity exists outside the expected human silhouette?

This metric directly targets the current failure mode.

---

# 20. Temporal Matte Evaluation

Compare RVM and MatAnyone 2 using optical-flow-compensated temporal consistency:

\[
E_{temp}
=
\frac{1}{N-1}
\sum_t
\left|
\alpha_{t+1}
-
W(
\alpha_t,
F_{t\rightarrow t+1}
)
\right|
\]

where:

- \(F\) is optical flow,
- \(W\) warps the previous matte to the next frame.

A lower value indicates better temporal stability.

This should be evaluated separately in:

```text
foreground interior
boundary band
background
```

because boundary stability matters considerably more than uniform interior alpha.

---

# 21. Suggested Release Gates

These values are project-specific engineering targets rather than established academic thresholds.

| Metric | Suggested Target |
|---|---:|
| Outside alpha leakage | ideally <1%; acceptable <2% |
| Held-out silhouette IoU | >0.97 |
| Boundary alpha MAD | ≥25% improvement over B0 |
| Foreground LPIPS | ≥15% improvement over B0 |
| Major detached floaters | 0 |
| Final asset size | preferably <50 MB |
| Gaussian count | preferably <500k when quality permits |
| 360° visual audit | no persistent hair/arm/foot floaters |

---

# 22. Recommended Software Architecture

Avoid continuing to place every operation inside `MultiViewSplat.py`.

Refactor incrementally:

```text
outputs/
├── MultiViewSplat.py
├── SharpSplatUI.py
│
work/
├── person_capture/
│   ├── config.py
│   ├── extract_frames.py
│   ├── select_frames.py
│   │
│   ├── matte/
│   │   ├── rvm_backend.py
│   │   ├── sam2_seed.py
│   │   ├── matanyone2_backend.py
│   │   └── temporal_qc.py
│   │
│   ├── preprocess/
│   │   ├── make_rgba.py
│   │   ├── despill.py
│   │   └── crop_intrinsics.py
│   │
│   ├── colmap/
│   │   ├── pose_pipeline.py
│   │   ├── filter_sparse.py
│   │   └── diagnostics.py
│   │
│   ├── human_pose/
│   │   ├── hmr2.py
│   │   ├── root_fit.py
│   │   └── canonicalize.py
│   │
│   ├── train/
│   │   ├── brush.py
│   │   └── splatfacto.py
│   │
│   └── eval/
│       ├── render_views.py
│       ├── image_metrics.py
│       ├── alpha_metrics.py
│       └── report.py
│
└── configs/
    ├── baseline.yaml
    ├── rgba_30k.yaml
    ├── filtered_init.yaml
    └── canonical.yaml
```

Each experiment should save:

```text
Git commit
configuration
input frame manifest
matting model/version
COLMAP reconstruction identifier/hash
Brush CLI arguments
training statistics
final Gaussian count
evaluation report
```

This will turn the current experimentation into a reproducible reconstruction pipeline.

---

# 23. Example Experiment Interface

```python
@dataclass
class ExperimentConfig:
    name: str

    frame_manifest: Path

    matte_backend: Literal[
        "rvm",
        "matanyone2"
    ]

    use_rgba: bool
    filter_sparse: bool
    canonicalize_subject: bool

    total_steps: int = 30_000
    growth_stop_iter: int = 12_000
    max_resolution: int = 1600
    max_splats: int = 1_000_000
    alpha_weight: float = 0.2


def run_experiment(
    cfg: ExperimentConfig
) -> Metrics:

    frames = load_manifest(
        cfg.frame_manifest
    )

    reconstruction = (
        ensure_colmap_reconstruction(
            frames
        )
    )

    alpha = run_matting(
        frames,
        backend=cfg.matte_backend,
    )

    rgba = build_rgba_dataset(
        frames=frames,
        alpha=alpha,
        despill=True,
    )

    if cfg.filter_sparse:

        sparse = filter_sparse_cloud(
            reconstruction=
                reconstruction,
            alpha_maps=alpha,
        )

    else:

        sparse = reconstruction

    cameras = (
        reconstruction.cameras
    )

    if cfg.canonicalize_subject:

        body_motion = (
            estimate_human_motion(
                frames
            )
        )

        cameras = (
            canonicalize_cameras(
                cameras,
                body_motion,
            )
        )

    dataset = write_brush_dataset(
        images=rgba,
        sparse=sparse,
        cameras=cameras,

        # Important:
        masks_folder=None,
    )

    model = train_brush(
        dataset,
        total_steps=
            cfg.total_steps,
        growth_stop_iter=
            cfg.growth_stop_iter,
        max_resolution=
            cfg.max_resolution,
        max_splats=
            cfg.max_splats,
        match_alpha_weight=
            cfg.alpha_weight,
    )

    return evaluate(
        model,
        dataset
    )
```

---

# 24. Implementation Order

## Phase 0 — Diagnostics

Before further architectural changes:

1. Record the final Gaussian count of Run 3.
2. Visualize the sparse COLMAP cloud separately.
3. Color sparse points according to foreground-mask support.
4. Measure how many points have:
   - >80% foreground support,
   - 50–80% support,
   - <50% support.
5. Render alpha from the existing Run 3 model.
6. Measure opacity outside the person mask.

These diagnostics can validate the background-seed hypothesis without expensive retraining.

---

## Phase 1 — Training Schedule Fix

Run:

```text
B1
binary mask
120 frames
full sparse cloud
30k steps
```

If noise decreases dramatically:

> premature termination was a major contributor.

If not:

> proceed immediately to alpha supervision.

---

## Phase 2 — Correct Foreground Semantics

Generate soft RGBA images.

Remove the separate `masks/` training path.

Run:

```text
B2
soft RGBA
120 frames
full sparse cloud
30k
```

This is the most important experiment.

Expected improvement:

- fewer background floaters,
- better silhouette,
- cleaner transparency,
- more stable hair/clothing boundaries.

---

## Phase 3 — Foreground Sparse Initialization

Apply multi-view mask-support filtering.

Run:

```text
B3
soft RGBA
filtered sparse model
120 frames
30k
```

Compare directly against B2.

If B3 is only marginally better than B2, sparse contamination is not the dominant problem.

If B3 improves substantially:

> retain sparse filtering permanently.

---

## Phase 4 — Matting Upgrade

Integrate:

```text
SAM 2 target seed
        ↓
MatAnyone 2
        ↓
soft alpha
```

Run the same training configuration.

Compare:

```text
RVM RGBA
vs
MatAnyone 2 RGBA
```

using both:

- temporal matte metrics,
- final 3D reconstruction metrics.

---

## Phase 5 — Frame Optimization

Select approximately 45–72 views using:

```text
camera angle coverage
+ low human motion
+ sharpness
+ matte confidence
```

Do not optimize frame count blindly.

The target is:

> minimum temporal inconsistency while maintaining sufficient viewpoint diversity.

---

## Phase 6 — Person-Centric Crop

Introduce a fixed crop and correctly update camera intrinsics.

This should be preferred over immediately raising global image resolution.

---

## Phase 7 — Motion Canonicalization

If residual artifacts are now mostly:

- double contours,
- arm ghosting,
- shoulder ghosting,
- hair duplication,

then the primary problem is performer motion.

Integrate:

```text
HMR2 / GVHMR
        ↓
root motion
        ↓
subject-centric cameras
        ↓
static Brush reconstruction
```

---

## Phase 8 — Alternative Human Reconstruction

Only if the canonicalized static model is still insufficient should the project move to:

- GauHuman,
- 3DGS-Avatar,
- HUGS,
- GaussianAvatar,
- SplattingAvatar,
- modern monocular 4DGS approaches.

---

# 25. Recommended Technical Decision

Do **not** replace the current system with a 4D Gaussian architecture yet.

There are still several high-value corrections available within the existing pipeline.

The immediate target configuration should be:

```text
COLMAP poses from static environment
        +
target-person soft alpha
        +
RGBA Brush training
        +
30k optimization
        +
growth stopped around 12k–15k
```

Then add:

```text
foreground sparse filtering
```

and evaluate again.

If the remaining errors are predominantly disconnected floaters and unwanted opacity, continue improving:

- alpha supervision,
- sparse initialization,
- pruning,
- loss behavior.

If the remaining errors instead become anatomically structured artifacts such as:

- duplicated shoulders,
- arm trails,
- double silhouettes,
- facial/hair ghosting,

then the root cause is no longer Gaussian cleanup.

It is the violation of the static-scene assumption caused by human motion.

At that point, the next engineering milestone should be:

**subject-centric human-motion canonicalization**, followed only if necessary by a fully deformable human Gaussian representation.

---

# Final Recommended Target Pipeline

```text
Video
 │
 ├─► full-resolution frame extraction
 │
 ├─► static-background COLMAP reconstruction
 │       └─ camera intrinsics + extrinsics
 │
 ├─► target-person initialization
 │       └─ SAM 2 / manual mask
 │
 ├─► MatAnyone 2
 │       └─ temporally stable soft alpha
 │
 ├─► conservative green despill
 │
 ├─► fixed person-centric crop
 │       └─ corrected camera intrinsics
 │
 ├─► RGBA training dataset
 │
 ├─► foreground-support filtering of COLMAP points
 │
 ├─► motion-aware angular frame selection
 │
 ├─► Brush training
 │       ├─ 30k steps
 │       ├─ growth stop 12k–15k
 │       ├─ alpha supervision
 │       └─ controlled Gaussian budget
 │
 ├─► alpha / leakage / boundary evaluation
 │
 └─► IF anatomical ghosting remains:
         │
         ├─► HMR2 / GVHMR
         ├─► root-motion canonicalization
         └─► repeat static RGBA 3DGS

             IF still insufficient:
                 ↓
         GauHuman / 3DGS-Avatar /
         HUGS / deformable 4DGS
```

## Bottom Line

The strongest next move is **not a new 4DGS architecture**.

The current person-only reconstruction is not yet testing the best version of static foreground 3DGS.

First correct three issues:

1. **Use true soft-alpha RGBA supervision instead of background-ignore masks.**
2. **Train for approximately 30k steps so optimization continues well beyond the Gaussian growth phase.**
3. **Filter foreground initialization points and evaluate whether background COLMAP geometry materially affects the result.**

Only after these controlled experiments should the remaining artifacts be attributed to the subject's non-rigid motion and escalated to canonical-human or deformable Gaussian reconstruction.