"""Local browser UI for Apple SHARP single-image Gaussian splatting."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import gradio as gr
from MultiViewSplat import train_scene, stop_training


APP_DIR = Path(__file__).resolve().parent
TASK_DIR = APP_DIR.parent
SHARP = TASK_DIR / "work" / "sharp-env" / "Scripts" / "sharp.exe"
CHECKPOINT = TASK_DIR / "work" / "models" / "sharp_2572gikvuh.pt"
RESULTS_DIR = APP_DIR / "my-splats"
VIEWER_PATH = APP_DIR / "SplatViewer.html"


def create_splat(input_file: str | None, progress=gr.Progress()):
    """Run the installed SHARP model and return its exported PLY file."""
    if not input_file:
        raise gr.Error("Choose an image first.")
    if not SHARP.is_file() or not CHECKPOINT.is_file():
        raise gr.Error("The SHARP installation or its model checkpoint is missing.")

    source = Path(input_file)
    run_dir = RESULTS_DIR / f"{source.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    copied_source = run_dir / source.name
    shutil.copy2(source, copied_source)

    progress(0.1, desc="Preparing image")
    command = [
        str(SHARP),
        "predict",
        "-i",
        str(copied_source),
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

    splat_path = run_dir / f"{source.stem}.ply"
    if not splat_path.is_file():
        raise gr.Error("SHARP finished without creating the expected .ply file.")

    size_mb = splat_path.stat().st_size / 1_000_000
    progress(1, desc="Complete")
    tail = console_text[-4000:] or "(no console output)"
    return str(splat_path), f"Done. Created `{splat_path.name}` ({size_mb:.1f} MB).", tail


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
                        label="Photo", file_types=["image"], type="filepath", height=260
                    )
                    generate = gr.Button("Create Gaussian Splat", variant="primary", size="lg")
                with gr.Column(scale=1):
                    output_splat = gr.File(label="Gaussian splat (.ply)", interactive=False)
                    status = gr.Markdown("Ready when you are.")
                    sharp_console = gr.Textbox(label="SHARP console output", lines=6, max_lines=6,
                                               interactive=False)
            gr.Markdown(
                "SHARP infers nearby viewpoints from one image. For the most convincing result, use a "
                "sharp photo with clear depth and avoid expecting unseen sides of an object or scene."
            )
            generate.click(create_splat, [input_image], [output_splat, status, sharp_console])
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
