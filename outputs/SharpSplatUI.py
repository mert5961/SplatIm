"""Local browser UI for Apple SHARP single-image Gaussian splatting."""

from __future__ import annotations

import re
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps
from plyfile import PlyData, PlyElement
from MultiViewSplat import VIDEO_SUFFIXES, train_scene, stop_training
from VideoSubjectSplat import (
    EXTRACT_PANEL,
    PREPARE_PANEL,
    REFINE_PANEL,
    SAM_PANEL,
    TRAIN_PANEL,
    extract_subject_splat,
    load_target_frame,
    switch_target_frame,
    save_target_keyframe,
    forget_target_keyframe,
    resume_prepared_video,
    prepare_video,
    refine_matanyone2,
    run_sam2,
    train_masked_splat,
)
from ProgressUI import CSS as PROGRESS_CSS, present_job
from SubjectRecovery import export_recovery
from SubjectKeyframes import preprocess_editor_safely


APP_DIR = Path(__file__).resolve().parent
TASK_DIR = APP_DIR.parent
SHARP = TASK_DIR / "work" / "sharp-env" / "Scripts" / "sharp.exe"
CHECKPOINT = TASK_DIR / "work" / "models" / "sharp_2572gikvuh.pt"
RESULTS_DIR = APP_DIR / "my-splats"
VIEWER_PATH = APP_DIR / "SplatViewer.html"
RVM_DIR = TASK_DIR / "work" / "rvm"
RVM_CHECKPOINT = Path(torch.hub.get_dir()) / "checkpoints" / "rvm_resnet50.pth"

_matting_model = None


def _person_matte(source: Path) -> tuple[Image.Image, np.ndarray]:
    """Return an upright RGB image and an RVM person alpha matte."""
    global _matting_model

    if not RVM_DIR.is_dir() or not RVM_CHECKPOINT.is_file():
        raise gr.Error("The local person-removal model is missing.")
    if str(RVM_DIR) not in sys.path:
        sys.path.insert(0, str(RVM_DIR))
    if _matting_model is None:
        from model import MattingNetwork

        model = MattingNetwork("resnet50").eval().to("cuda")
        model.load_state_dict(torch.load(RVM_CHECKPOINT, map_location="cpu", weights_only=True))
        _matting_model = model

    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to("cuda")
    downsample = min(512 / max(tensor.shape[-2:]), 1.0)
    with torch.inference_mode():
        _foreground, alpha, *_ = _matting_model(tensor, *([None] * 4), downsample)
    matte = alpha[0, 0, 0].float().cpu().numpy()
    return image, np.clip(matte, 0.0, 1.0)


def _crop_person(image: Image.Image, matte: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """Crop around detected people with enough context for stable depth inference."""
    ys, xs = np.where(matte >= 0.20)
    if len(xs) < 256:
        raise gr.Error("I could not find a clear person in this photo. Try a sharper image or turn off Person only.")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    pad_x = max(24, round(width * 0.12))
    pad_y = max(24, round(height * 0.08))
    x0, x1 = max(0, x0 - pad_x), min(image.width, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(image.height, y1 + pad_y)
    return image.crop((x0, y0, x1, y1)), matte[y0:y1, x0:x1]


def _video_middle_frame(video: Path, run_dir: Path) -> Path:
    """Extract the clip's middle frame as the single photo SHARP consumes.

    The decode uses the ffmpeg backend shipped with imageio-ffmpeg (same as the
    multi-photo pipeline), so no system ffmpeg install is needed.
    """
    import imageio.v2 as imageio
    import imageio_ffmpeg

    total_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(video))
    if total_frames < 1:
        raise gr.Error("I could not read any frames from that video.")
    reader = imageio.get_reader(str(video), format="ffmpeg")
    try:
        frame = reader.get_data(total_frames // 2)
    finally:
        reader.close()
    out = run_dir / f"{video.stem}-frame.jpg"
    Image.fromarray(frame).save(out, quality=95)
    return out


def _prepare_person_input(source: Path, run_dir: Path) -> tuple[Path, Path, np.ndarray]:
    """Create a cropped neutral-background input and a transparent preview."""
    image, matte = _person_matte(source)
    image, matte = _crop_person(image, matte)

    rgb = np.asarray(image, dtype=np.float32)
    alpha = matte[..., None]
    # A quiet gray background avoids teaching SHARP the original room while still
    # giving its RGB-only model a conventional image to infer depth from.
    composite = np.clip(rgb * alpha + 128.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    sharp_input = run_dir / f"{source.stem}-person-input.png"
    Image.fromarray(composite, "RGB").save(sharp_input)

    preview = np.dstack((rgb.astype(np.uint8), (matte * 255).astype(np.uint8)))
    preview_path = run_dir / f"{source.stem}-person-preview.png"
    Image.fromarray(preview, "RGBA").save(preview_path)
    return sharp_input, preview_path, matte


def _filter_ply_to_person(ply_path: Path, matte: np.ndarray) -> tuple[int, int]:
    """Keep SHARP's two Gaussian layers only where the person matte is present."""
    ply = PlyData.read(str(ply_path), mmap=False)
    vertices = ply["vertex"].data
    total = len(vertices)

    grid_side = 768  # SHARP's public checkpoint emits two 768x768 layers.
    grid_pixels = grid_side * grid_side
    if total % grid_pixels:
        raise gr.Error("This SHARP output has an unexpected layout, so person isolation was not applied.")
    layers = total // grid_pixels

    matte_grid = np.asarray(
        Image.fromarray((matte * 255).astype(np.uint8), "L").resize(
            (grid_side, grid_side), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    alpha = np.tile(matte_grid.reshape(-1), layers)
    keep = alpha >= 0.12
    kept = np.ascontiguousarray(vertices[keep])
    kept_alpha = alpha[keep]

    # Feather hair and clothing edges by scaling each retained Gaussian's opacity.
    opacity = 1.0 / (1.0 + np.exp(-kept["opacity"].astype(np.float64)))
    opacity = np.clip(opacity * kept_alpha, 1e-5, 1.0 - 1e-5)
    kept["opacity"] = np.log(opacity / (1.0 - opacity)).astype(np.float32)

    elements = [PlyElement.describe(kept, "vertex")]
    for element in ply.elements:
        if element.name == "vertex":
            continue
        data = element.data.copy()
        if element.name == "frame" and len(data) >= 2:
            data[1]["frame"] = len(kept)
        elements.append(PlyElement.describe(data, element.name))
    # Windows keeps the source PLY mapped while it is being read, so write beside
    # it and atomically replace only after serialization finishes.
    filtered_path = ply_path.with_name(f"{ply_path.stem}.filtered{ply_path.suffix}")
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(filtered_path))
    filtered_path.replace(ply_path)
    return total, len(kept)


def create_splat(input_file: str | None, person_only: bool, progress=gr.Progress()):
    """Run the installed SHARP model and return its exported PLY file."""
    if not input_file:
        raise gr.Error("Choose a photo or video first.")
    if not SHARP.is_file() or not CHECKPOINT.is_file():
        raise gr.Error("The SHARP installation or its model checkpoint is missing.")

    source = Path(input_file)
    run_dir = RESULTS_DIR / f"{source.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    copied_source = run_dir / source.name
    shutil.copy2(source, copied_source)
    if source.suffix.lower() in VIDEO_SUFFIXES:
        progress(0.05, desc="Reading a frame from the video")
        copied_source = _video_middle_frame(copied_source, run_dir)

    preview_path = None
    sharp_input = copied_source
    matte = None
    if person_only:
        progress(0.08, desc="Finding the person and removing the background")
        sharp_input, preview_path, matte = _prepare_person_input(copied_source, run_dir)

    progress(0.1, desc="Preparing image")
    command = [
        str(SHARP),
        "predict",
        "-i",
        str(sharp_input),
        "-o",
        str(run_dir),
        "-c",
        str(CHECKPOINT),
        "--device",
        "cuda",
    ]
    progress(0.3, desc="Generating Gaussian splat on GPU")

    finished = subprocess.run(command, capture_output=True, text=True)
    console_text = "\n".join(part for part in (finished.stdout, finished.stderr) if part).strip()
    if finished.returncode:
        details = console_text
        raise gr.Error(f"SHARP could not create the splat. {details[-700:]}")

    generated_path = run_dir / f"{sharp_input.stem}.ply"
    splat_path = run_dir / f"{source.stem}-person.ply" if person_only else generated_path
    if person_only and generated_path.is_file():
        generated_path.replace(splat_path)
    if not splat_path.is_file():
        raise gr.Error("SHARP finished without creating the expected .ply file.")

    isolation_note = ""
    if person_only and matte is not None:
        progress(0.9, desc="Removing background Gaussians")
        before, after = _filter_ply_to_person(splat_path, matte)
        isolation_note = f" Kept {after:,} of {before:,} Gaussians around the detected person."

    size_mb = splat_path.stat().st_size / 1_000_000
    progress(1, desc="Complete")
    tail = console_text[-4000:] or "(no console output)"
    return (
        str(splat_path),
        str(preview_path) if preview_path else None,
        f"Done. Created `{splat_path.name}` ({size_mb:.1f} MB).{isolation_note}",
        tail,
    )


# The multi-view pipeline stages, in the exact order the pipeline runs them.
# Index 5 (green-screen keying) only runs when the option is enabled.
PIPELINE_STAGES = [
    "Prepare views (photos / video frames)",
    "1/4 · Finding visual features (COLMAP)",
    "2/4 · Matching overlapping photos (COLMAP)",
    "3/4 · Estimating camera positions (COLMAP)",
    "3/4 · Correcting lenses — undistort (COLMAP)",
    "Green-screen keying (background masks)",
    "4/4 · Training one scene (Brush)",
    "Export & verify (scene.ply)",
]


def _stage_index(detail: str) -> int | None:
    """Map a stage label yielded by train_scene() onto the board row index."""
    if detail.startswith("1/4"):
        return 1
    if detail.startswith("2/4"):
        return 2
    if detail.startswith("3/4"):
        return 3 if "Estimating" in detail else 4
    if detail.startswith("4/4"):
        return 6
    return None


def _board_markdown(state: dict, mask_background: bool, current_detail: str) -> str:
    """Render the stage board: done / running / pending / failed rows."""
    rows = []
    for index, label in enumerate(PIPELINE_STAGES):
        if index == 5 and not mask_background:
            rows.append("- ~~Green-screen keying~~ — not requested")
            continue
        if state["done"]:
            rows.append(f"- ✅ {label}")
            continue
        if state["failed"] is not None:
            if index < state["stage"]:
                rows.append(f"- ✅ {label}")
            elif index == state["stage"]:
                rows.append(f"- ❌ **{label}** — failed")
            else:
                rows.append(f"- ⬜ {label}")
            continue
        if index < state["stage"]:
            rows.append(f"- ✅ {label}")
        elif index == state["stage"]:
            suffix = ""
            match = re.search(r"· (\d[\d,]*)/(\d[\d,]*)", current_detail)
            if match:
                suffix = f" · {match.group(1)}/{match.group(2)}"
            rows.append(f"- 🔄 **{label}**{suffix}")
        else:
            rows.append(f"- ⬜ {label}")
    return "**Process**\n\n" + "\n".join(rows)


def train_scene_live(files, quality, same_camera, allow_partial, mask_background, progress=gr.Progress()):
    """Wrap train_scene() to also stream the live log and stage board.

    The pipeline itself is untouched: every stage, command, and job file is
    exactly what train_scene() produces. This wrapper only re-sends the
    values the pipeline already yields, plus the derived board markdown.
    """
    state = {"stage": 0, "failed": None, "done": False}
    for result, status, log_file, live_log in train_scene(
        files, quality, same_camera, allow_partial, mask_background, progress=progress
    ):
        if status.startswith("Finished:"):
            state["done"] = True
            detail = ""
        else:
            detail = status.split("**")[1] if status.startswith("**") else status
            index = _stage_index(detail)
            if index is not None:
                state["stage"] = index
            else:
                state["failed"] = detail
        board = _board_markdown(state, mask_background, detail)
        yield result, status, log_file, board, live_log


def prepare_subject_live(video, quality):
    yield from present_job(prepare_video, [video, quality], label="Prepare + camera alignment",
                           count=8, status_index=4, board_index=5, log_index=7)


def track_subject_live(state, frame, editor, context):
    yield from present_job(run_sam2, [state, frame, editor, context], label="SAM 2.1 identity tracking",
                           count=7, status_index=3, board_index=4, log_index=6)


def refine_subject_live(state):
    yield from present_job(refine_matanyone2, [state], label="MatAnyone2 alpha refinement",
                           count=7, status_index=3, board_index=4, log_index=6)


def train_subject_live(state, quality):
    yield from present_job(train_masked_splat, [state, quality], label="Brush masked training",
                           count=6, status_index=2, board_index=3, log_index=5)


def extract_subject_live(state):
    yield from present_job(extract_subject_splat, [state], label="Mask-guided Gaussian extraction",
                           count=9, status_index=4, board_index=5, log_index=8)


def recovery_job_choices():
    jobs = APP_DIR / "video-subject-jobs"
    completed = [p for p in jobs.glob("*/job.json")
                 if (p.parent / "subject-extraction/evidence.npz").is_file()
                 and (p.parent / "subject-extraction/keep.ply").is_file()]
    return [("Current session", "")] + [(p.parent.name, p.parent.name)
                                          for p in sorted(completed, reverse=True)]


def prepared_job_choices():
    jobs = APP_DIR / "video-subject-jobs"
    return [p.parent.name for p in sorted(jobs.glob("*/job.json"), reverse=True)
            if (p.parent / "sam2_frames" / "00000.jpg").is_file()]


LOCK_SELECTION = "(...args) => { document.getElementById('subject-selection').inert = true; return args; }"
UNLOCK_SELECTION = "() => { document.getElementById('subject-selection').inert = false; }"


def recover_subject(state, saved_job, minimum_score, maximum_added):
    jobs = (APP_DIR / "video-subject-jobs").resolve()
    if saved_job:
        job = (jobs / saved_job).resolve()
    elif state and state.get("job"):
        job = Path(state["job"]).resolve()
    else:
        raise gr.Error("Choose a completed extraction or finish the current extraction first.")
    if job.parent != jobs:
        raise gr.Error("Choose a job from this app's saved extractions.")
    result = export_recovery(job, float(minimum_score), int(maximum_added))
    return (result["final"], result["recovered"], result["report"], result["scores"],
            f"**Recovered {result['recovered_count']:,} uncertain splats.** "
            f"Original: {result['original_count']:,} → new export: {result['final_count']:,}. "
            "Compare both files from the same viewing angle. The support score is a heuristic, "
            "not a probability; lower thresholds may introduce haze or background fragments.", result["preview"])


with gr.Blocks(title="SplatIm — Photo and Video to Gaussian Splat") as demo:
    gr.Markdown(
        "# Photo / video → Gaussian Splat\n"
        "Create a quick estimate from one frame, reconstruct a selected video subject, "
        "or combine overlapping views into one connected scene."
    )
    with gr.Tabs():
        with gr.Tab("Video → selected subject"):
            video_subject_state = gr.State({})
            # Submit the displayed frame identity with its editor payload, rather
            # than resolving a potentially newer server-side State on a queued event.
            target_context = gr.JSON(value={}, visible=False)
            gr.Markdown(
                "Use this for an **orbit video of a mostly stationary person or object**. The workflow is "
                "intentionally staged: you choose the target with a brush, inspect every propagated mask, "
                "optionally refine it, and only then spend time training the splat."
            )
            with gr.Tabs():
                with gr.Tab("1 · Prepare"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            subject_video = gr.File(
                                label="Orbit video",
                                file_types=["video", ".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"],
                                type="filepath",
                            )
                            subject_quality = gr.Radio(
                                ["Preview", "Standard", "High quality"], value="Preview",
                                label="Reconstruction / training quality",
                            )
                            prepare_subject = gr.Button("Prepare and align video", variant="primary")
                            prepared_jobs = gr.Dropdown(prepared_job_choices(), label="Resume a prepared video")
                            with gr.Row():
                                refresh_prepared = gr.Button("Refresh prepared videos")
                                resume_subject = gr.Button("Resume saved frames + keyframes")
                        with gr.Column(scale=2):
                            prepare_board = gr.Markdown(PREPARE_PANEL)
                    prepare_status = gr.Markdown(
                        "Camera alignment happens before masking so the brush and every debug mask use the "
                        "exact undistorted frames consumed by the splat trainer."
                    )
                    aligned_gallery = gr.Gallery(
                        label="Registered frames (all views used by the pipeline)", columns=6,
                        height=430, object_fit="contain", allow_preview=True,
                    )
                    with gr.Row():
                        prepare_run_log = gr.File(label="Preparation run log", interactive=False)
                        prepare_live_log = gr.Textbox(
                            label="Preparation output", lines=14, max_lines=14, interactive=False,
                        )

                with gr.Tab("2 · Select + track"):
                    gr.Markdown(
                        "SAM 2.1 produces a **binary identity mask**: it decides which object to follow. "
                        "It is expected to look hard around hair and fabric; the next stage creates the soft alpha matte."
                    )
                    with gr.Row():
                        with gr.Column(scale=3, elem_id="subject-selection"):
                            target_frame = gr.Slider(
                                0, 1, value=0, step=1, label="Prompt / correction frame",
                                info="Edits save when you switch frames. All saved keyframes guide the next propagation.",
                            )
                            target_editor = gr.ImageEditor(
                                label="Roto-style target brush (red overlay = selected)",
                                type="numpy", image_mode="RGBA", height=620,
                                brush=gr.Brush(default_size=32, colors=["#ff3355"], default_color="#ff3355", color_mode="fixed"),
                                eraser=gr.Eraser(default_size=40),
                                layers=gr.LayerOptions(allow_additional_layers=False, layers=["Target mask"]),
                                transforms=(), fixed_canvas=False, format="png", sources=(),
                            )
                            target_editor.preprocess_original = target_editor.preprocess
                            target_editor.preprocess = lambda payload: preprocess_editor_safely(target_editor, payload)
                            target_note = gr.Markdown("Prepare the video to load a selectable frame.")
                            with gr.Row():
                                save_target = gr.Button("Save keyframe")
                                forget_target = gr.Button("Remove this keyframe")
                            keyframe_status = gr.Markdown("No saved keyframes.")
                            track_subject = gr.Button("Propagate all saved keyframes with SAM 2.1", variant="primary")
                        with gr.Column(scale=2):
                            sam_board = gr.Markdown(SAM_PANEL)
                            gr.Markdown(
                                "Paint inside the wanted person or object. After tracking, choose a bad frame "
                                "with the slider, erase false areas or paint missing areas. Edits are saved when "
                                "switching frames or propagating. Save keyframe also pins an unchanged prediction. "
                                "All saved keyframes are used together; visiting a frame alone does not pin it."
                            )
                            sam_run_log = gr.File(label="SAM 2.1 run log", interactive=False)
                            sam_live_log = gr.Textbox(
                                label="SAM 2.1 output", lines=14, max_lines=14, interactive=False,
                            )
                    sam_status = gr.Markdown("Waiting for a prepared video and brush prompt.")
                    with gr.Tabs():
                        with gr.Tab("Tracking overlays"):
                            sam_overlay_gallery = gr.Gallery(
                                label="Every tracked frame — green is selected; captions show coverage and centroid jumps",
                                columns=5, height=650, object_fit="contain", allow_preview=True,
                            )
                        with gr.Tab("Binary masks"):
                            sam_mask_gallery = gr.Gallery(
                                label="SAM 2.1 binary identity masks", columns=6, height=650,
                                object_fit="contain", allow_preview=True,
                            )

                with gr.Tab("3 · Refine matte"):
                    gr.Markdown(
                        "MatAnyone2 creates soft alpha from **frame zero's mask only** in the current integration. "
                        "It does **not** enforce your later saved keyframes. To keep the corrected SAM sequence, "
                        "skip this optional stage and train after tracking. If you refine, inspect every result; "
                        "refinement switches the active training masks to MatAnyone2."
                    )
                    with gr.Row():
                        refine_subject = gr.Button("Refine masks with MatAnyone2", variant="primary")
                        refine_board = gr.Markdown(REFINE_PANEL)
                    refine_status = gr.Markdown("Run SAM 2.1 first, then refine and review the alpha mattes here.")
                    with gr.Tabs():
                        with gr.Tab("Matte overlays"):
                            refined_overlay_gallery = gr.Gallery(
                                label="Every refined frame", columns=5, height=650,
                                object_fit="contain", allow_preview=True,
                            )
                        with gr.Tab("Alpha mattes"):
                            refined_mask_gallery = gr.Gallery(
                                label="MatAnyone2 soft alpha mattes", columns=6, height=650,
                                object_fit="contain", allow_preview=True,
                            )
                    with gr.Row():
                        refine_run_log = gr.File(label="MatAnyone2 run log", interactive=False)
                        refine_live_log = gr.Textbox(
                            label="MatAnyone2 output", lines=14, max_lines=14, interactive=False,
                        )

                with gr.Tab("4 · Train"):
                    gr.Markdown(
                        "Brush trains the Gaussian scene using the reviewed masks. False foreground can create "
                        "background Gaussians; missing foreground can remove thin body parts and hair."
                    )
                    with gr.Row():
                        train_subject = gr.Button("Train masked Gaussian scene", variant="primary")
                        train_board = gr.Markdown(TRAIN_PANEL)
                    train_status = gr.Markdown("Review the masks before starting the GPU training stage.")
                    raw_subject_splat = gr.File(
                        label="Intermediate masked scene (.ply)", interactive=False
                    )
                    with gr.Row():
                        train_run_log = gr.File(label="Brush training log", interactive=False)
                        train_live_log = gr.Textbox(
                            label="Brush output", lines=16, max_lines=16, interactive=False,
                        )

                with gr.Tab("5 · Extract + results"):
                    gr.Markdown(
                        "The final pass checks every trained Gaussian against the reviewed masks from all cameras, "
                        "then separates accepted subject, uncertain boundary, and rejected background Gaussians."
                    )
                    with gr.Row():
                        extract_subject = gr.Button("Extract subject-only Gaussians", variant="primary")
                        extract_board = gr.Markdown(EXTRACT_PANEL)
                    extract_status = gr.Markdown("Train the masked scene first, then create the subject-only PLY.")
                    with gr.Row():
                        final_subject_splat = gr.File(
                            label="Final selected subject (.ply)", interactive=False
                        )
                        uncertain_subject_splat = gr.File(
                            label="Debug: uncertain boundary Gaussians (.ply)", interactive=False
                        )
                        hidden_subject_splat = gr.File(
                            label="Debug: rejected background Gaussians (.ply)", interactive=False
                        )
                    extraction_report = gr.File(label="Extraction report (.json)", interactive=False)
                    with gr.Row():
                        extract_run_log = gr.File(label="Extraction run log", interactive=False)
                        extract_live_log = gr.Textbox(
                            label="Extraction output", lines=16, max_lines=16, interactive=False,
                        )
                    gr.Markdown(
                        "### Recover useful uncertain splats\n\n"
                        "Score the existing uncertain splats using mask support across views, proximity "
                        "to accepted geometry, and nearby subject colors. Higher thresholds recover fewer splats. "
                        "The maximum is a cap, not a target. Added splats are limited to local body-scale sizes; "
                        "your original export is preserved. This does not rerun training."
                    )
                    saved_choices = recovery_job_choices()
                    recovery_job = gr.Dropdown(
                        saved_choices, value=saved_choices[1][1] if len(saved_choices) > 1 else "",
                        label="Extraction to optimize",
                    )
                    refresh_recovery = gr.Button("Refresh saved extractions")
                    with gr.Row():
                        recovery_score = gr.Slider(.5, 1, value=.9, step=.01,
                            label="Minimum recovery support score",
                            info="Conservative start: 0.90. Lower it to explore the completeness / spill tradeoff.")
                        recovery_cap = gr.Number(value=5000, minimum=0, maximum=1000000,
                            precision=0, label="Maximum added splats (0 = original only)")
                    recover_button = gr.Button("Score uncertain splats and export recovery", variant="primary")
                    recovery_status = gr.Markdown("Choose a completed extraction to recover from.")
                    with gr.Row():
                        recovery_result = gr.File(label="Recovered subject (.ply)", interactive=False)
                        recovery_added = gr.File(label="Debug: added splats only (.ply)", interactive=False)
                    with gr.Row():
                        recovery_report = gr.File(label="Recovery report (.json)", interactive=False)
                        recovery_scores = gr.File(label="Per-splat scores (.csv)", interactive=False)
                    recovery_table = gr.Dataframe(
                        headers=["Scene index", "Support score", "Mask support", "Near body", "Color match", "Added"],
                        datatype=["number", "number", "number", "number", "number", "bool"],
                        label="Top 50 recovery candidates", interactive=False,
                    )
                    refresh_recovery.click(lambda: gr.update(choices=recovery_job_choices()),
                                           outputs=recovery_job, queue=False, show_progress="hidden")
                    recover_button.click(recover_subject,
                        [video_subject_state, recovery_job, recovery_score, recovery_cap],
                        [recovery_result, recovery_added, recovery_report, recovery_scores, recovery_status, recovery_table],
                        show_progress_on=recovery_status, concurrency_id="gpu", concurrency_limit=1)

            prepare_subject.click(
                prepare_subject_live,
                [subject_video, subject_quality],
                [video_subject_state, target_frame, target_editor, aligned_gallery,
                 prepare_status, prepare_board, prepare_run_log, prepare_live_log],
                concurrency_id="gpu", concurrency_limit=1, show_progress="minimal", show_progress_on=prepare_board,
            ).then(load_target_frame, [video_subject_state, target_frame],
                   [target_editor, target_note, target_context, keyframe_status],
                   concurrency_id="gpu", concurrency_limit=1, show_progress="hidden")
            refresh_prepared.click(lambda: gr.update(choices=prepared_job_choices()),
                                   outputs=prepared_jobs, queue=False, show_progress="hidden")
            resume_subject.click(resume_prepared_video, [prepared_jobs],
                [video_subject_state, target_frame, target_editor, aligned_gallery,
                 prepare_status, prepare_board, prepare_run_log, prepare_live_log, subject_quality],
                concurrency_id="gpu", concurrency_limit=1, show_progress="minimal", show_progress_on=prepare_board,
            ).then(load_target_frame, [video_subject_state, target_frame],
                   [target_editor, target_note, target_context, keyframe_status],
                   concurrency_id="gpu", concurrency_limit=1, show_progress="hidden")
            target_frame.input(
                switch_target_frame, [video_subject_state, target_frame, target_editor, target_context],
                [target_editor, target_note, target_context, keyframe_status],
                concurrency_id="gpu", concurrency_limit=1, show_progress="hidden", js=LOCK_SELECTION,
            ).then(lambda: None, js=UNLOCK_SELECTION, queue=False, show_progress="hidden")
            save_target.click(save_target_keyframe, [video_subject_state, target_editor, target_context],
                              [target_context, keyframe_status], concurrency_id="gpu",
                              concurrency_limit=1, show_progress="hidden", js=LOCK_SELECTION
            ).then(lambda: None, js=UNLOCK_SELECTION, queue=False, show_progress="hidden")
            forget_target.click(forget_target_keyframe,
                                [video_subject_state, target_frame, target_editor, target_context],
                                [target_editor, target_note, target_context, keyframe_status],
                                concurrency_id="gpu", concurrency_limit=1, show_progress="hidden", js=LOCK_SELECTION
            ).then(lambda: None, js=UNLOCK_SELECTION, queue=False, show_progress="hidden")
            track_subject.click(
                track_subject_live, [video_subject_state, target_frame, target_editor, target_context],
                [video_subject_state, sam_overlay_gallery, sam_mask_gallery, sam_status,
                 sam_board, sam_run_log, sam_live_log],
                concurrency_id="gpu", concurrency_limit=1, show_progress="minimal", show_progress_on=sam_board,
                js=LOCK_SELECTION,
            ).then(load_target_frame, [video_subject_state, target_frame],
                   [target_editor, target_note, target_context, keyframe_status],
                   concurrency_id="gpu", concurrency_limit=1, show_progress="hidden"
            ).then(lambda: None, js=UNLOCK_SELECTION,
                   queue=False, show_progress="hidden")
            refine_subject.click(
                refine_subject_live, [video_subject_state],
                [video_subject_state, refined_overlay_gallery, refined_mask_gallery,
                 refine_status, refine_board, refine_run_log, refine_live_log],
                concurrency_id="gpu", concurrency_limit=1, show_progress="minimal", show_progress_on=refine_board,
            )
            train_subject.click(
                train_subject_live, [video_subject_state, subject_quality],
                [video_subject_state, raw_subject_splat, train_status, train_board,
                 train_run_log, train_live_log],
                concurrency_id="gpu", concurrency_limit=1, show_progress="minimal", show_progress_on=train_board,
            )
            extract_subject.click(
                extract_subject_live, [video_subject_state],
                [video_subject_state, final_subject_splat, uncertain_subject_splat,
                 hidden_subject_splat, extract_status, extract_board, extraction_report,
                 extract_run_log, extract_live_log],
                concurrency_id="gpu", concurrency_limit=1, show_progress="minimal", show_progress_on=extract_board,
            )
        with gr.Tab("Create splat"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.File(
                        label="Photo or video (quick mode — one frame is used)",
                        file_types=["image", "video", ".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"],
                        type="filepath", height=260,
                    )
                    person_only = gr.Checkbox(
                        value=True,
                        label="Person only — remove the background and crop around the subject",
                    )
                    generate = gr.Button("Create Gaussian Splat", variant="primary", size="lg")
                with gr.Column(scale=1):
                    output_splat = gr.File(label="Gaussian splat (.ply)", interactive=False)
                    person_preview = gr.Image(
                        label="Detected person", type="filepath", height=260, interactive=False
                    )
                    status = gr.Markdown("Ready when you are.")
                    sharp_console = gr.Textbox(label="SHARP console output", lines=6, max_lines=6,
                                               interactive=False)
            gr.Markdown(
                "This is the **quick single-frame route**, not video reconstruction. Person only is on by "
                "default. The preview shows what will be kept; turn it off when you want the entire room or scene. "
                "Video clips are reduced to their middle frame. Use **Video → selected subject** when you want "
                "to brush a target and use all aligned video views. "
                "SHARP infers nearby viewpoints from one image, so unseen sides are still an estimate."
            )
            generate.click(
                create_splat,
                [input_image, person_only],
                [output_splat, person_preview, status, sharp_console],
            )
        with gr.Tab("Multiple photos → one scene"):
            gr.Markdown('Upload overlapping photos **or video clips** of the **same stationary place or object**, '
                        'captured from different positions. Aim for 20–100+ photos (videos are sampled automatically, '
                        'up to 120 frames per clip) with roughly 70–80% overlap. Alignment and training can take several minutes or longer.')
            multi_photos = gr.File(label='Photos or videos of one scene', file_count='multiple',
                                   file_types=['image', 'video', '.mp4', '.mov', '.m4v', '.avi', '.webm', '.mkv'],
                                   type='filepath')
            with gr.Row():
                quality = gr.Radio(['Preview', 'Standard', 'High quality'], value='Standard', label='Training quality')
                same_camera = gr.Checkbox(value=True, label='All photos use the same camera, zoom, and resolution')
                allow_partial = gr.Checkbox(value=False, label='Train even if alignment is weak (experimental)')
                mask_bg = gr.Checkbox(value=False, label='Ignore green screen background')
            with gr.Row():
                train = gr.Button('Build one connected scene', variant='primary')
                stop = gr.Button('Stop multi-photo job')
            multi_board = gr.Markdown(_board_markdown({"stage": 0, "failed": None, "done": False}, False, ""))
            multi_live_log = gr.Textbox(label='Live process log', lines=16, max_lines=16,
                                        interactive=False)
            multi_status = gr.Markdown('Ready. Only photos that can be aligned into the largest connected scene will be used.')
            multi_result = gr.File(label='Connected scene (.ply)', interactive=False)
            multi_log = gr.File(label='Run log', interactive=False)
            train.click(train_scene_live, [multi_photos, quality, same_camera, allow_partial, mask_bg],
                        [multi_result, multi_status, multi_log, multi_board, multi_live_log],
                        concurrency_id='gpu', concurrency_limit=1)
            stop.click(stop_training, outputs=multi_status, queue=False)
        with gr.Tab("View splat"):
            viewer_src = (APP_DIR / 'SplatViewer.html').as_posix()
            gr.HTML(
                f'<iframe src="/gradio_api/file={viewer_src}" '
                'title="Gaussian splat viewer" style="border: 0; height: 680px; width: 100%"></iframe>'
            )


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    demo.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7869")),
        show_error=True,
        theme=gr.themes.Soft(),
        css=PROGRESS_CSS,
        allowed_paths=[str(APP_DIR)],
    )
