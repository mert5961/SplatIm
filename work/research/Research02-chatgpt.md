Below is the implementation specification I would give to the coding agent. It updates the earlier plan using the actual B0–B6 measurements, and it deliberately separates **semantic segmentation, alpha matting, chroma-key/color recovery, 3DGS supervision, and human-motion modeling** so that each stage can be tested independently.

Your B0–B6 work already establishes several facts rather than hypotheses: longer mask-mode training made leakage worse; true RGBA supervision dropped leakage from 6.7–9.9% to about 1.2%; sparse foreground filtering cut the model almost in half without materially changing the leakage floor; the remaining green contamination is concentrated in the alpha boundary; and 60 selected views perform essentially the same as 120 for leakage.   The current work should therefore stop treating Brush training duration, view count, or background COLMAP seeds as the primary unsolved problem.

# Scientific implementation plan: robust person-only 3DGS from green-screen orbit video

## 1. Target architecture

The new pipeline should have **three independent vision signals** before anything is passed to Brush:

```text
                         ┌──────────────────────────┐
                         │  ORIGINAL VIDEO FRAMES   │
                         └────────────┬─────────────┘
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
             ▼                        ▼                        ▼
   SEMANTIC / IDENTITY         SOFT ALPHA MATTING       GREEN-SCREEN ANALYSIS
   SEGMENTATION                / FOREGROUND EST.        / IMAGE-BASED KEY
             │                        │                        │
 SAM 3.1 / SAM 2.1            MatAnyone2              local clean plate
 Cutie / SAMURAI              μMatting                 chroma-distance key
             │                Sapiens2                screen-color model
             │                RVM baseline                    │
             └──────────────┬─────────┴───────────────────────┘
                            │
                            ▼
                  QUALITY / CONSENSUS LAYER
                            │
                       alpha + confidence
                            │
                            ▼
                 foreground color recovery
                            │
                       despill v3
                            │
                            ▼
                    straight RGBA PNG
                            │
                            ▼
               foreground-filtered COLMAP
                            │
                            ▼
                     Brush 3DGS
                            │
                            ▼
                    held-out metrics
                            │
             ┌──────────────┴──────────────┐
             │ artifacts random?          │ anatomical ghosting?
             ▼                            ▼
       matte/pruning fixes        root canonicalization
                                      GVHMR
                                         │
                                         ▼
                               deformable human GS
                                  only if necessary
```

The crucial architectural rule is:

> **Segmentation is not matting. Matting is not chroma keying. Despill is not alpha estimation.**

They provide complementary evidence and must remain independently measurable.

---

# 2. Models worth integrating now

| Component                    | Candidate                         | Role in this project                     | Recommendation                             |
| ---------------------------- | --------------------------------- | ---------------------------------------- | ------------------------------------------ |
| Target/person tracking       | **SAM 3.1**                       | Select and track the exact target person | Primary semantic candidate                 |
| Target/person tracking       | **SAM 2.1**                       | Prompted video segmentation              | Required baseline/fallback                 |
| Motion-aware VOS             | **SAMURAI**                       | SAM2 + motion-aware memory               | Optional difficult-track fallback          |
| Robust VOS                   | **Cutie**                         | Object-level video memory                | Excellent independent cross-check          |
| Video matting                | **MatAnyone 2**                   | Temporally consistent soft alpha         | Primary candidate                          |
| High-res video matting       | **μMatting**                      | Fine detail + temporal refinement        | Benchmark against MatAnyone2               |
| Human-specific image matting | **Sapiens2-1B Matting**           | Independent high-detail alpha/foreground | Very important secondary teacher           |
| Multi-person guided matting  | **MaGGIe**                        | Mask-guided instance matting             | Useful if distractor woman causes failures |
| Existing baseline            | **RVM**                           | Current reference                        | Keep permanently as control                |
| High-res image matting       | BiRefNet-HR / MattePro / ViTMatte | Frame-wise boundary oracle candidates    | Secondary only                             |
| Classical keying             | IBK-style clean-plate key         | Exploit known green screen               | **Must implement**                         |
| Temporal QC                  | **SEA-RAFT**                      | Flow + temporal matte measurement        | Recommended evaluator                      |

### SAM 3.1 versus SAM 2.1

SAM 3 added concept/exemplar prompting and video tracking, and SAM 3.1, released in March 2026, added Object Multiplex and further tracking optimizations. SAM 3's architecture combines detection with a memory-based video tracker and supports text, exemplars, points, boxes, and masks. ([Meta AI][1])

But **SAM must not produce the final 3DGS alpha**. It is a segmentation model, so its boundary is fundamentally more binary than a real matte. MatAnyone 2's paper explicitly discusses the problem of segmentation supervision producing segmentation-like boundaries instead of fractional hair/edge alpha. ([arXiv][2])

For this particular video, SAM 3.1's most useful role is:

```text
"Which exact human is the target?"
```

not:

```text
"What is the exact fractional alpha of every hair?"
```

Use a visual exemplar/box around the target rather than the generic text prompt `"person"`, because the early background woman is a real distractor.

SAM 2.1 remains an important baseline because it already provides promptable video tracking, multi-object handling, and independent per-object processing. Its repository is mature and Apache-2.0 licensed. ([GitHub][3])

### MatAnyone 2

This is currently one of the strongest matches to the problem. It was a CVPR 2026 Highlight, explicitly targets human video matting, takes a video plus a first-frame segmentation mask, and outputs per-frame alpha/foreground results. ([GitHub][4])

The underlying research is highly relevant to your failure mode: MatAnyone 2 introduces a learned quality evaluator and a large real-world video-matting dataset to improve both semantic correctness and fine boundary quality. It specifically addresses long videos and large appearance variation using long-range reference frames. ([arXiv][2])

There is one important implementation constraint: the public repository lists the quality-evaluator checkpoint/training release as unfinished. **Do not build your pipeline around MQE being available.** Implement our own measurable disagreement/quality layer instead. ([GitHub][4])

### μMatting

This is sufficiently recent and relevant that I would benchmark it, not merely mention it. CVPR 2026's μMatting is designed around high-resolution human video matting, uses sparse 3D convolution for critical-region refinement, and introduces temporal modulation. Its associated αMatte4K work explicitly targets fractional alpha and temporal consistency at high resolution. ([Open Access][5])

Your source is roughly 1080×1920 after orientation, so this is almost exactly the kind of case where high-resolution boundary treatment matters.

### Sapiens2 is particularly interesting here

Meta released a Sapiens2-1B human-matting checkpoint in May 2026. Sapiens2 is human-specific and its matting task predicts **soft alpha plus foreground RGB**, which is extremely interesting for the current green-spill problem. ([GitHub][6])

It is image-based rather than temporally recurrent, so I would **not replace MatAnyone2 with it**. Instead:

```text
MatAnyone2 / μMatting = temporal alpha authority
Sapiens2               = independent per-frame boundary/foreground authority
```

Disagreement between those systems is itself a useful confidence signal.

Sapiens2 also has segmentation, pose, normals and pointmaps in the same human-centric model family, which may become useful later for canonicalization. ([GitHub][6])

---

# 3. Do not blindly fuse models

The agent must not immediately average:

```text
SAM + RVM + MatAnyone2 + Sapiens2
```

That is not scientifically justified.

First run every candidate independently and measure it.

Only introduce fusion **after** determining where each model is superior.

Use the semantic models to define a trimap:

$$
T(p)=
\begin{cases}
FG & p\in\operatorname{erode}(M,r)\\
BG & p\notin\operatorname{dilate}(M,2r)\\
UNK & \text{otherwise}
\end{cases}
$$

where a reasonable initial normalized radius is:

$$
r=\operatorname{round}(0.0025\min(H,W))
$$

approximately 3 px at 1080 pixels.

The key insight is that semantic segmentation is highly trustworthy in the **core**, while matting models should dominate the **unknown boundary band**.

---

# 4. Implement a real image-based green-screen keyer

Do not rely only on AI.

The existence of a green cyclorama is valuable measurement information.

Foundry's IBK keyer is built around this exact concept: derive/use a clean-plate image representing the spatial variation of the screen rather than assuming one global picked green color. That is why IBK handles uneven screens better than a single-color key. ([Foundry Learn][7])

Implement an equivalent deterministic signal.

## Screen plate estimation

For each frame estimate:

$$
B_t(x,y)
$$

the local screen color.

Do **not** simply use:

```python
B = median(all_green_pixels)
```

because illumination across the cyclorama is spatially non-uniform.

Start by finding certain screen pixels:

```text
outside dilated SAM mask
AND
alpha_matting < 0.02
AND
green-chroma likelihood > threshold
```

Then fit a smooth local background field.

Good options, from simplest to strongest:

```text
degree-2/3 robust polynomial surface
→ thin-plate spline
→ Poisson/biharmonic interpolation
```

Fit each channel independently but use the same validity mask.

The result must include:

```text
plate_rgb.exr/png
plate_validity.png
plate_fit_error.npy
```

Do not hide failed regions.

---

# 5. Use a chroma-distance key as independent evidence

Instead of the simplistic:

$$
G > R,\;G>B
$$

model the screen chromaticity statistically.

For each pixel obtain a chroma vector:

$$
z_p=[C_b,C_r]^T
$$

and estimate local/background statistics:

$$
\mu_B,\Sigma_B
$$

from known green pixels.

Then compute Mahalanobis distance:

$$
d_B(p)=
\sqrt{
(z_p-\mu_B)^T
(\Sigma_B+\epsilon I)^{-1}
(z_p-\mu_B)
}
$$

Convert to a soft key:

$$
\alpha_{key}
=
\operatorname{smoothstep}(d_0,d_1,d_B)
$$

where `d0` and `d1` must be derived from measured quantiles of confirmed background/foreground samples rather than hard-coded forever.

For extremely dark pixels, chroma becomes unstable. Therefore lower the confidence of the classical key for:

$$
Y < Y_{\min}
$$

and defer to semantic/matting models.

That is important for the black polo.

---

# 6. Color science: perform compositing math in linear light

This is an easy place for a coding agent to make a subtle but serious error.

Before doing:

$$
C=\alpha F+(1-\alpha)B
$$

the image must be in an appropriate **linear-light RGB representation**.

Therefore first inspect the source using `ffprobe`:

```text
color_primaries
color_transfer
colorspace
color_range
```

Never silently assume BT.709/sRGB if metadata exists.

If metadata is missing, log the assumption in the experiment manifest.

Workflow:

```text
encoded RGB
    ↓ EOTF
linear RGB
    ↓ compositing/decontamination calculations
corrected linear RGB
    ↓ OETF
encoded straight RGB
    + alpha
```

Do not change Brush's training color convention silently. Re-encode the corrected foreground to the expected input representation.

---

# 7. Replace despill-v2 with physically cleaner despill-v3

B6 already fixed the `excess²` bug and reduced edge green excess from roughly 0.19 to 0.04. That proves the basic diagnosis. 

But:

```text
G -= k * excess
```

is still a channel heuristic.

Implement the next version in a **luminance-preserving screen chroma subspace**.

Let linear RGB be:

$$
F=[R,G,B]^T
$$

and Rec.709 linear luminance vector:

$$
l=
[0.2126,\;0.7152,\;0.0722]^T
$$

Define projection into the chroma plane:

$$
P=
I-
\frac{ll^T}{l^Tl}
$$

Measure mean screen color \(\bar B\), then obtain the normalized green-screen chroma direction:

$$
s=
\frac{P\bar B}
{\|P\bar B\|+\epsilon}
$$

For foreground:

$$
c=PF
$$

Screen-like spill magnitude:

$$
e=
\max(0,c^Ts-\tau)
$$

Then:

$$
F'=
F-k(\alpha)e\,s
$$

Because:

$$
l^Ts=0
$$

this operation preserves luminance by construction:

$$
l^TF'=l^TF
$$

before gamut clipping.

Use an alpha-dependent strength such as:

$$
k(\alpha)
=
\operatorname{clip}
(k_{core}+k_{edge}(1-\alpha),0,1)
$$

but tune these experimentally.

This is substantially preferable to reducing the G channel directly.

---

# 8. Separate two kinds of green contamination

The agent must explicitly distinguish:

### Background mixing

A semi-transparent edge pixel obeys:

$$
C=\alpha F+(1-\alpha)B
$$

Given alpha and local plate \(B\):

$$
\hat F=
\frac{C-(1-\alpha)B}
{\max(\alpha,\epsilon)}
$$

This removes **compositing contamination**.

### Physical green spill

Green light actually illuminated:

```text
skin
hair
shirt
arms
```

That green is part of \(F\), so the compositing inversion cannot remove it.

This requires the screen-chroma despill above.

Therefore:

```text
foreground reconstruction
        THEN
physical despill
```

not one heuristic pretending to do both.

---

# 9. Do not invert alpha near zero

The equation becomes ill-conditioned as:

$$
\alpha\rightarrow0
$$

Therefore use three bands initially:

```text
alpha >= 0.70
    direct foreground recovery allowed

0.10 <= alpha < 0.70
    recovered color blended with propagated confident foreground color

alpha < 0.10
    never trust division result
    propagate color from neighboring foreground
```

Exact thresholds are starting points and must be configurable.

For low-alpha hair fringes, RGB should be extended from valid foreground instead of amplified through division by tiny alpha.

FBA Matting specifically treats foreground estimation as an important component rather than treating alpha alone as sufficient. ([GitHub][8])

---

# 10. Add a deterministic confidence map

Every frame should now produce:

```text
alpha.png / .npy
foreground_rgb
semantic_mask
key_alpha
confidence
uncertainty
screen_plate
despill_amount
```

Do not throw away intermediate data.

A useful uncertainty model is:

$$
U =
w_1|\alpha_{MA2}-\alpha_{Sapiens}|
+
w_2|\alpha_{MA2}-\alpha_{key}|
+
w_3E_{temporal}
+
w_4E_{semantic}
$$

normalized to `[0,1]`.

High disagreement is not something to hide by averaging.

It is useful information saying:

> this pixel/frame is unreliable.

Eventually this confidence can also be incorporated into Brush loss weighting.

---

# 11. Temporal measurement must use optical-flow occlusion handling

Use SEA-RAFT rather than raw frame-to-frame alpha differences.

SEA-RAFT is a modern RAFT variant with strong accuracy/generalization and provides uncertainty information; its implementation is BSD-3-Clause. ([GitHub][9])

For alpha:

$$
E_{temp}
=
\frac{
\sum_p
V_t(p)
\left|
\alpha_{t+1}(p)
-
W(\alpha_t,F_{t\rightarrow t+1})(p)
\right|
}{
\sum_pV_t(p)+\epsilon
}
$$

where \(V_t\) excludes:

* flow-occluded pixels,
* disocclusions,
* unreliable flow,
* frame boundaries.

Report it separately for:

```text
core foreground
boundary band
background
```

A matte can have a good whole-image temporal score simply because 90% of the image is static background. That metric would be misleading.

---

# 12. Build a small human-verified evaluation set

There is no true alpha ground truth for the existing video.

Do not pretend there is.

Create approximately:

```text
12–16 frames
```

distributed across the full 360° orbit, deliberately including:

* front,
* back,
* profile,
* hair-heavy views,
* arm boundaries,
* worst spill views,
* early frames containing the second person.

For these frames, manually verify a **trimap**, not fractional hair alpha:

```text
certain foreground
certain background
unknown boundary
```

That gives genuine binary ground truth for semantic errors without pretending that manually painted hair alpha is physically exact.

For soft-alpha mathematics, use synthetic data tests.

---

# 13. Mandatory synthetic mathematical tests

Before processing the real video, the coding agent must create synthetic unit tests.

Generate:

$$
C=\alpha F+(1-\alpha)B
$$

from known:

```text
F
B
alpha
```

including:

* smooth green gradients,
* different green illumination,
* soft alpha edges,
* thin hair-like structures,
* black clothing,
* noisy compression,
* physical green spill added to F.

The foreground recovery test should verify, for example:

$$
MAE(\hat F,F)<10^{-4}
$$

for noise-free synthetic pixels with sufficiently large alpha.

The test must prove that the equation is implemented in the correct color space.

Also verify:

$$
|\Delta Y|<10^{-6}
$$

for the unclipped luminance-preserving despill operation.

These are unit tests, not subjective screenshots.

---

# 14. Required 2D metrics

Before running Brush, every matte candidate must get a report containing at least:

| Metric                     | Definition / purpose                            |
| -------------------------- | ----------------------------------------------- |
| Core FG miss               | \(mean(1-\alpha)\) on human-verified certain FG |
| Certain BG leakage         | \(mean(\alpha)\) on certain BG                  |
| Boundary temporal MAE      | flow-warped alpha residual                      |
| Semantic IoU               | thresholded alpha vs verified semantic mask     |
| Boundary F-score           | silhouette edge agreement                       |
| Model disagreement         | MatAnyone2 vs Sapiens2/μMatting                 |
| Green-screen contradiction | alpha where chroma says certain screen          |
| Edge spill score           | projected screen-chroma contamination           |
| Core color change          | ΔE before/after despill on alpha>0.95           |
| Gamut clipping             | fraction of corrected pixels clipped            |
| Matte entropy              | detects overly binary / overly fuzzy mattes     |

Do not optimize only one metric.

In particular:

```text
lower leakage
```

can be achieved simply by shrinking the person.

So leakage must always be paired with:

```text
foreground recall
boundary F-score
alpha/boundary quality
```

---

# 15. Important metric trap: compositing residual

If the pipeline calculates:

$$
F=
\frac{C-(1-\alpha)B}{\alpha}
$$

and then evaluates:

$$
\|C-[\alpha F+(1-\alpha)B]\|
$$

that metric is circular and nearly guaranteed to look excellent.

**The coding agent must not present this as evidence of matte quality.**

It is useful as:

* a numerical implementation sanity test,
* a synthetic-test metric,

but not an independent quality metric on real footage.

---

# 16. Recommended matting experiment

Do this **before another expensive set of 3DGS runs**.

Run the same 60 selected frames through:

```text
RVM
MatAnyone2
μMatting
Sapiens2
```

and additionally generate semantic tracks from:

```text
SAM 3.1
SAM 2.1
```

The 60-view choice is justified by your B5 result: 60 selected views retained the same 1.2% leakage as 120 views while producing fewer Gaussians. 

Do not initially fuse.

Generate a benchmark report first.

My expectation is:

```text
MatAnyone2 / μMatting
    strongest temporal matte

Sapiens2
    strong independent human boundary/foreground evidence

SAM3.1
    strongest semantic target identity

classical key
    strongest direct evidence that a pixel is green screen
```

The final pipeline can then exploit each where it is strongest.

---

# 17. A particularly strong hybrid strategy

If the measurements confirm the expected behavior, the production matte should be built regionally:

```text
CERTAIN FOREGROUND
SAM core + matting agreement
→ preserve MatAnyone/μMatting alpha, usually close to 1

CERTAIN BACKGROUND
SAM exterior + chroma screen agreement
→ alpha = 0

UNKNOWN EDGE
MatAnyone2 or μMatting
→ preserve fractional alpha

MODEL DISAGREEMENT
→ retain primary matte
→ set high uncertainty
→ do NOT blindly average

GREEN EDGE
→ alpha remains from matte
→ RGB corrected by foreground recovery + despill
```

This is much safer than a single end-to-end black-box model.

---

# 18. Brush experiment design after 2D selection

Once the best matte/color pipeline is selected, use the already-proven Brush semantics:

```text
RGBA images
NO masks/ directory
foreground-filtered COLMAP
30k steps
growth stop ~12k
```

Your own source inspection established that the presence of `masks/` routes Brush back into mask-weighted loss semantics even when image alpha exists; removing `masks/` was therefore essential to the successful B2 run. 

Then perform:

```text
match_alpha_weight:
0.10
0.20
0.35
0.50
```

Only this parameter changes in that ablation.

Selection must be made from **held-out rendered views**, not train leakage.

---

# 19. Fix the evaluation split before doing more optimization

Your current leakage metric uses projected Gaussian opacity on 25 training views. 

That was useful diagnostically but it is no longer sufficient.

For the 60-view configuration use approximately:

```text
48 train
12 held out
```

stratified over the orbit.

Do not select every twelfth chronologically without checking angular distribution.

For final 3D evaluation, render actual accumulated alpha:

$$
A_r(p)
$$

and measure:

$$
L_{outside}
=
\frac{
\sum_pA_r(p)[1-M(p)]
}{
\sum_pA_r(p)+\epsilon
}
$$

Also report:

$$
IoU(A_r>\tau,M)
$$

and boundary F-score.

---

# 20. Keep sparse filtering permanently

B3 is already conclusive enough operationally.

Filtering:

```text
68,704 COLMAP points
       ↓
18,266 foreground points
```

reduced the final representation from roughly 100k Gaussians to 59k while leaving quality/leakage essentially unchanged. 

That is valuable even if it was not the primary noise cause.

Keep:

```text
track >= 3
foreground support >= 0.60
median alpha >= 0.50
```

as the current baseline and only retune if a measured regression appears.

---

# 21. Do not revisit the failed crop/dense-window ideas

Two previous paths are now experimentally closed for this capture:

A four-second dense COLMAP window failed because there was insufficient parallax, and the person already occupies effectively the entire 899×1600 frame, so a fixed person crop yields no meaningful gain. 

Do not let the agent waste time reopening these ideas unless new data is collected.

Instead, once the matte pipeline is correct, benchmark:

```text
1600 undistortion/training
vs
1920 native-height equivalent
```

using the same calibrated reconstruction.

Do not upsample above source resolution and claim additional geometric information.

---

# 22. Motion becomes the next problem only after this

If the best 2D matte + despill pipeline produces clean RGBA but the 3D result still shows:

```text
double shoulders
two arm surfaces
hair trails
duplicated shirt edges
anatomical smearing
```

stop tuning alpha.

That is evidence of the static-scene model being violated.

HUGS is particularly relevant: it was designed to reconstruct a static scene plus an animatable human from a monocular video and reports operation with roughly 50–100 frames—the same range as your 60-view dataset. It initializes human Gaussians using SMPL and allows Gaussians to deviate to represent clothing and hair. ([Open Access][10])

GauHuman similarly addresses articulated humans from monocular video with a human/body prior. ([GitHub][11])

Before replacing Brush completely, however, I would first try **root-motion canonicalization** using GVHMR. GVHMR specifically estimates world-grounded human motion from monocular video and has an available implementation. ([GitHub][12])

Only if root canonicalization fails should the project escalate to an articulated/deformable Gaussian backend.

---

# 23. Exact coding-agent execution order

The agent should execute the project in this order and **may not skip a gate because a result "looks good"**:

1. **Freeze B6.** Preserve branch, config, PLY, frame manifest, COLMAP model, matte and hashes. Do not mutate B6.

2. **Create `person_eval_v2`.** Implement train/eval angular split, actual rendered-alpha metrics, boundary metrics, and reproducible JSON/CSV reports. Prove the metric code on synthetic masks.

3. **Create the verified trimap set.** Select 12–16 representative frames around the full orbit. Store FG/BG/unknown separately.

4. **Integrate SAM 3.1 and SAM 2.1 behind one interface.** Input must support box/mask/exemplar. Save raw logits/masks where APIs expose them. Do not convert immediately to 8-bit binary PNG and throw away precision.

5. **Run target-identity tracking.** Detect identity swaps against the second person. Generate forward and reverse tracking passes where practical and report their disagreement.

6. **Integrate MatAnyone2 in an isolated environment.** Its documented environment is Python 3.10, whereas SAM3 and the existing project are naturally compatible with Python 3.12-era PyTorch. Use a subprocess/file protocol rather than contaminating the main venv. ([GitHub][4])

7. **Benchmark μMatting and Sapiens2.** Do not replace MatAnyone2 until quantitative results prove superiority. Sapiens2 must export raw `float32` alpha; inspect its code before assuming foreground-output tensor conventions. Its documentation specifies raw alpha output and premultiplied foreground prediction. ([GitHub][13])

8. **Implement `screen_plate.py`.** Robustly identify green-screen samples, create local plate \(B(x,y)\), report fit residuals and coverage.

9. **Implement `chroma_key.py`.** Use the statistical chroma-distance key. This output is an independent measurement, not the final matte.

10. **Implement `foreground_recover.py` and `despill_v3.py`.** Work in linear light, use stable-alpha ranges, luminance-preserving chroma projection, gamut checks and synthetic tests.

11. **Generate the 2D benchmark.** Compare RVM, MatAnyone2, μMatting and Sapiens2. No 3D training until this report exists.

12. **Choose a primary matte.** If fusion is attempted, run it as another candidate and prove it beats the best single model. Never assume ensemble = better.

13. **Run Brush alpha-weight ablation.** Use 60 selected views, filtered sparse cloud, 30k steps and held-out views.

14. **Run 1600 vs 1920 only on the winning pipeline.**

15. **Classify remaining defects.** If they are alpha/color defects, remain in the current pipeline. If they are anatomically structured ghosting, begin GVHMR root canonicalization.

16. **Only then create an alternative HUGS/GauHuman/deformable-GS branch.** Do not mix this research branch into the validated Brush production path.

---

# 24. Mandatory repository safety rules

The coding agent must obey these constraints throughout the work:

```text
Never modify master directly.

Every experiment gets:
- unique experiment ID
- Git commit
- full config
- model/checkpoint ID + hash
- input frame manifest
- input hashes
- output hashes
- Python/PyTorch/CUDA versions
- metrics.json
- timing.json

Never change two independent experimental variables
and call the result an ablation.

After every source edit:
1. inspect git diff
2. run syntax/type checks
3. run the smallest relevant test
4. run mathematical synthetic tests when applicable

If an edit breaks syntax/tests:
revert that edit before attempting another approach.

Never silently:
- premultiply RGBA
- alter color transfer functions
- resize masks separately from RGB
- change COLMAP camera intrinsics
- threshold a soft alpha
- switch model checkpoints
- normalize alpha differently
- convert float alpha to uint8 before evaluation

Any assumption must be written to the manifest.

Any NaN/Inf is a hard failure.

Any image/mask dimension mismatch is a hard failure.

Any camera/image frame-name mismatch is a hard failure.

A visually pleasing output is NOT a passed experiment
unless the metrics and validation gates also pass.
```

---

# 25. Acceptance gates

I would set the first engineering targets relative to B6 rather than inventing arbitrary perfect numbers.

A new 2D matte/color pipeline is accepted only if it:

```text
reduces boundary temporal error vs RVM
AND
does not increase certain-background leakage
AND
does not reduce certain-foreground recall
AND
reduces screen-chroma edge contamination
AND
does not materially alter core foreground color.
```

For despill specifically:

```text
median ΔE00 in alpha > 0.95 core:
target < 1 if possible

95th percentile core ΔE00:
target < 3

edge green-screen-axis contamination:
must improve materially over B6's ~0.04

gamut-clipped corrected pixels:
report explicitly; target near zero
```

For the final 3D result:

```text
held-out alpha leakage
held-out silhouette IoU
held-out boundary F-score
foreground LPIPS
Gaussian count
detached floater count
```

must all be reported together.

Do not select a model solely because it has the lowest leakage.

---

# 26. Licensing must be tracked from day one

This matters because the eventual use is professional broadcast software.

SAM 2 is Apache-2.0, while SAM 3 uses Meta's newer custom SAM License. ([Meta AI][14])

MatAnyone/MatAnyone2 use the NTU S-Lab license; the S-Lab license text permits redistribution/use for non-commercial purposes and requires contacting the contributors for commercial use. Treat MatAnyone2 as a **research/evaluation dependency until commercial permission is clarified**. ([GitHub][15])

GauHuman has the same commercial-use issue under S-Lab licensing. ([GitHub][16])

Sapiens2 uses Meta's custom Sapiens2 License and includes specific use restrictions, so it also needs product/legal review before shipping rather than being treated like an Apache/MIT dependency. ([GitHub][17])

Therefore add this to the model registry:

```yaml
license:
  name:
  url:
  commercial_status:
    - allowed
    - research_only
    - requires_review
  checkpoint_license:
  code_license:
```

Do not assume the code license and model-weight license are identical.

---

# 27. My recommended near-term stack

For **research quality**, I would currently bet on:

```text
SAM 3.1
    ↓ exact target identity

MatAnyone2
vs
μMatting
    ↓ choose by measured temporal/boundary quality

Sapiens2
    ↓ independent human-matte / foreground cross-check

IBK-style local screen model
    ↓ deterministic green-screen evidence

linear-light foreground recovery
    ↓

luminance-preserving despill-v3
    ↓

straight RGBA
    ↓

filtered COLMAP seeds
    ↓

60 selected views
    ↓

Brush
30k / growth-stop 12k
    ↓

held-out alpha rendering metrics
```

The most important new insight from the research is that **the strongest system is probably not “find one better matting AI model.”**

It is a **multi-evidence reconstruction front end**:

$$
\text{semantic identity}
+
\text{temporal alpha}
+
\text{human-specific image detail}
+
\text{known green-screen physics}
$$

followed by explicit uncertainty measurement.

That gives you something much closer to a production/broadcast capture pipeline than simply swapping RVM for another neural network.

And the current B0–B6 results are actually encouraging: the large 3DGS failure has already been corrected by RGBA alpha supervision; the sparse cloud problem is controlled; the model is compact; and 60 frames are sufficient.  The next work should therefore focus heavily on **matte fidelity, foreground color recovery and objective validation**, and only move to deformable-human Gaussians if the residual failure is demonstrably anatomical motion rather than alpha/color error.

[1]: https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/?utm_source=chatgpt.com "SAM 3: Segment Anything with Concepts | Research - AI at Meta"
[2]: https://arxiv.org/abs/2512.11782 "MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator"
[3]: https://github.com/facebookresearch/sam2?utm_source=chatgpt.com "GitHub - facebookresearch/sam2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model. · GitHub"
[4]: https://github.com/pq-yang/MatAnyone2 "GitHub - pq-yang/MatAnyone2: [CVPR 2026 Highlight] MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator · GitHub"
[5]: https://openaccess.thecvf.com/content/CVPR2026/html/Chen_alphaMatte4K__muMatting_Dataset_and_Model_for_Ultra-Micro_Precision_Alpha_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[6]: https://github.com/facebookresearch/sapiens2?utm_source=chatgpt.com "GitHub - facebookresearch/sapiens2: 1K resolution vision transformers pretrained on 1B human images. · GitHub"
[7]: https://learn.foundry.com/nuke/14.0/content/reference_guide/keyer_nodes/ibkcolor.html?utm_source=chatgpt.com "IBKColor"
[8]: https://github.com/MarcoForte/FBA_Matting?utm_source=chatgpt.com "GitHub - MarcoForte/FBA_Matting: Official repository for the paper F, B, Alpha Matting · GitHub"
[9]: https://github.com/princeton-vl/SEA-RAFT?utm_source=chatgpt.com "GitHub - princeton-vl/SEA-RAFT: [ECCV2024 - Oral, Best Paper Award Candidate] SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow · GitHub"
[10]: https://openaccess.thecvf.com/content/CVPR2024/html/Kocabas_HUGS_Human_Gaussian_Splats_CVPR_2024_paper.html?utm_source=chatgpt.com "CVPR 2024 Open Access Repository"
[11]: https://github.com/skhu101/GauHuman?utm_source=chatgpt.com "GitHub - skhu101/GauHuman: Code for our CVPR'2024 paper \"GauHuman: Articulated Gaussian Splatting from Monocular Human Videos\" · GitHub"
[12]: https://github.com/czw90130/GVHMR?utm_source=chatgpt.com "GitHub - czw90130/GVHMR · GitHub"
[13]: https://github.com/facebookresearch/sapiens2/blob/main/docs/MATTING.md "sapiens2/docs/MATTING.md at main · facebookresearch/sapiens2 · GitHub"
[14]: https://ai.meta.com/blog/segment-anything-2-video/?utm_source=chatgpt.com "Update: Expanding access to Meta Segment Anything 2.1 on Amazon SageMaker JumpStart"
[15]: https://github.com/pq-yang/MatAnyone/blob/main/LICENSE?utm_source=chatgpt.com "MatAnyone/LICENSE at main · pq-yang/MatAnyone · GitHub"
[16]: https://github.com/skhu101/GauHuman/blob/main/LICENSE.md?utm_source=chatgpt.com "GauHuman/LICENSE.md at main · skhu101/GauHuman · GitHub"
[17]: https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md "sapiens2/LICENSE.md at main · facebookresearch/sapiens2 · GitHub"
