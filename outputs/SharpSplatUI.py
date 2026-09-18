"""Local browser UI for Apple SHARP single-image Gaussian splatting."""

from __future__ import annotations

import re
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


with gr.Blocks(title="SHARP — Image to Gaussian Splat") as demo:
    gr.Markdown(
        "# Photo → Gaussian Splat\n"
        "Create a quick splat from one photo, or combine overlapping photos into one connected scene."
    )
    with gr.Tabs():
        with gr.Tab("Create splat"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.File(
                        label="Photo or video (one frame is used)",
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
                "Person only is on by default. The preview shows what will be kept; turn it off when "
                "you want the entire room or scene. Video clips are reduced to their middle frame. "
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
        show_error=True,
        theme=gr.themes.Soft(),
        allowed_paths=[str(APP_DIR)],
    )
