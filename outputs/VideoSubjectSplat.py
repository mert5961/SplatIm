"""Inspectable video -> selected subject -> Gaussian splat workflow.

This module deliberately keeps the expensive stages separate.  A user can
inspect and correct the mask sequence before any splat training begins, and
can compare the trained full PLY with the final mask-guided extraction.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from MultiViewSplat import extract_video_frames
from SubjectKeyframes import (read_keyframes, save_keyframe, remove_keyframe, mask_digest,
                              load_keyframe_prompts, keyframe_summary, keyframe_revision,
                              register_keyframes, require_applied_keyframes)


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
JOBS = APP_DIR / "video-subject-jobs"
TOOLS = ROOT / "work" / "multiview-tools"
COLMAP = next(iter((TOOLS / "colmap").rglob("colmap.exe")), None)
BRUSH = TOOLS / "brush" / "brush_app.exe"
SAM2_REPO = ROOT / "work" / "sam2repo"
SAM2_CONFIG = SAM2_REPO / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = ROOT / "work" / "models" / "sam2" / "sam2.1_hiera_small.pt"
MA2_PYTHON = ROOT / "work" / "ma2-env" / "Scripts" / "python.exe"
MA2_REPO = ROOT / "work" / "matanyone2repo"
MA2_CHECKPOINT = ROOT / "work" / "models" / "matanyone2" / "matanyone2.pth"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
QUALITY = {
    "Preview": (1_000, 1_024),
    "Standard": (7_000, 1_600),
    "High quality": (30_000, 1_920),
}


def _log_path(state: dict) -> Path:
    return Path(state["job"]) / "pipeline.log"


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text.rstrip() + "\n")


def _tail(path: Path, limit: int = 24_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


def _board(active: int, *, failed: int | None = None, complete: int = -1) -> str:
    stages = [
        "Decode representative video frames",
        "COLMAP: features, overlap, and camera poses",
        "Undistort the registered views",
        "User brush / target prompt",
        "SAM 2.1: propagate the target mask",
        "MatAnyone2: optional alpha refinement",
        "Brush: train the masked Gaussian scene",
        "Mask-guided Gaussian extraction",
    ]
    rows = []
    for index, label in enumerate(stages):
        if failed == index:
            mark = "❌"
            suffix = " — failed"
        elif index <= complete:
            mark = "✅"
            suffix = ""
        elif index == active:
            mark = "🔄"
            suffix = " — current"
        else:
            mark = "⬜"
            suffix = ""
        rows.append(f"- {mark} **{index + 1}. {label}**{suffix}")
    return "### Pipeline inspector\n\n" + "\n".join(rows)


def _stage_panel(title: str, state: str, detail: str = "") -> str:
    icons = {"waiting": "⬜", "running": "🔄", "complete": "✅", "failed": "❌"}
    suffix = f"\n\n{detail}" if detail else ""
    return f"### {title}\n\n{icons[state]} **{state.capitalize()}**{suffix}"


def _editor_value(frame: Path, mask: Path | None = None) -> dict:
    background = Image.open(frame).convert("RGB")
    empty = np.zeros((background.height, background.width, 4), dtype=np.uint8)
    layers: list[Image.Image] = [Image.fromarray(empty, "RGBA")]
    if mask is not None and mask.is_file():
        matte = np.asarray(Image.open(mask).convert("L"))
        rgba = np.zeros((*matte.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 3] = np.where(matte > 127, 115, 0).astype(np.uint8)
        layers = [Image.fromarray(rgba, "RGBA")]
    return {"background": background, "layers": layers, "composite": None}


def _frame_paths(state: dict) -> list[Path]:
    return [Path(value) for value in state.get("frames", [])]


def _gallery(paths: list[Path], maximum: int = 120) -> list[tuple[str, str]]:
    if not paths:
        return []
    if len(paths) <= maximum:
        picks = range(len(paths))
    else:
        picks = np.linspace(0, len(paths) - 1, maximum).round().astype(int)
    return [(str(paths[index]), f"Frame {index:03d}") for index in picks]


def _run_logged(command: list[object], cwd: Path, log_path: Path, label: str) -> None:
    rendered = subprocess.list2cmdline([str(value) for value in command])
    _append_log(log_path, f"\n### {label}\n{rendered}")
    env = os.environ.copy()
    if COLMAP:
        env["PATH"] = str(COLMAP.parent) + os.pathsep + env.get("PATH", "")
    env["QT_QPA_PLATFORM"] = "offscreen"
    with log_path.open("a", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(value) for value in command], cwd=str(cwd), env=env,
            stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    if result.returncode:
        raise RuntimeError(f"{label} failed (exit {result.returncode}).")


def _largest_model(sparse: Path) -> tuple[int, Path, int]:
    models: list[tuple[int, Path, int]] = []
    for images_bin in sparse.rglob("images.bin"):
        with images_bin.open("rb") as stream:
            registered = struct.unpack("<Q", stream.read(8))[0]
        points_bin = images_bin.parent / "points3D.bin"
        points = 0
        if points_bin.is_file():
            with points_bin.open("rb") as stream:
                points = struct.unpack("<Q", stream.read(8))[0]
        models.append((registered, images_bin.parent, points))
    if not models:
        raise RuntimeError(
            "COLMAP could not create a connected camera model. The camera must move "
            "around a mostly stationary subject with strong overlap."
        )
    return max(models, key=lambda value: value[0])


def prepare_video(video: str | None, quality: str, progress=gr.Progress()):
    """Decode, align, and undistort a clip before interactive selection."""
    if not video:
        raise gr.Error("Choose a video first.")
    if COLMAP is None or not BRUSH.is_file():
        raise gr.Error("The local COLMAP or Brush installation is missing.")

    source = Path(video)
    job = JOBS / f"{time.strftime('%Y%m%d-%H%M%S')}-{source.stem}-{uuid.uuid4().hex[:6]}"
    input_dir = job / "input"
    sparse = job / "sparse"
    dataset = job / "dataset"
    for directory in (input_dir, sparse, dataset):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = job / "pipeline.log"
    state = {"job": str(job), "source": str(source), "quality": quality}

    try:
        progress(0.02, desc="Decoding representative frames")
        _append_log(log_path, f"SOURCE: {source}\nQUALITY: {quality}")
        count = 0
        for count, (frame, label) in enumerate(extract_video_frames(source), start=1):
            frame.convert("RGB").save(input_dir / f"{count - 1:05d}.jpg", quality=95)
            _append_log(log_path, f"frame {count - 1:05d}: {label}")
        if count < 3:
            raise RuntimeError("The video produced fewer than three usable frames.")

        _append_log(log_path, f"Decoded {count} representative frames.")
        progress(0.12, desc="COLMAP: finding visual features")
        db = job / "database.db"
        _run_logged([
            COLMAP, "feature_extractor", "--database_path", db,
            "--image_path", input_dir, "--ImageReader.camera_model", "SIMPLE_RADIAL",
            "--ImageReader.single_camera", "1", "--FeatureExtraction.max_image_size", "2400",
            "--SiftExtraction.estimate_affine_shape", "1",
        ], job, log_path, "COLMAP 1/3: feature extraction")
        progress(0.28, desc="COLMAP: matching overlapping views")
        _run_logged([
            COLMAP, "exhaustive_matcher", "--database_path", db,
            "--FeatureMatching.guided_matching", "1",
        ], job, log_path, "COLMAP 2/3: exhaustive matching")
        progress(0.42, desc="COLMAP: estimating camera positions")
        _run_logged([
            COLMAP, "mapper", "--database_path", db, "--image_path", input_dir,
            "--output_path", sparse,
        ], job, log_path, "COLMAP 3/3: camera reconstruction")
        registered, model, sparse_points = _largest_model(sparse)
        if registered < 3:
            raise RuntimeError(f"Only {registered} camera views aligned; at least three are required.")

        resolution = QUALITY[quality][1]
        progress(0.62, desc=f"Undistorting {registered} registered views")
        _run_logged([
            COLMAP, "image_undistorter", "--image_path", input_dir,
            "--input_path", model, "--output_path", dataset,
            "--output_type", "COLMAP", "--max_image_size", str(resolution),
        ], job, log_path, "COLMAP: undistort registered views")

        # SAM's video predictor requires consecutive JPEG names.  Keep a map to
        # the real COLMAP/Brush image names so masks can be copied back exactly.
        source_frames = sorted(
            path for path in (dataset / "images").iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        sam_frames = job / "sam2_frames"
        sam_frames.mkdir(exist_ok=True)
        mapping = []
        for index, path in enumerate(source_frames):
            staged = sam_frames / f"{index:05d}.jpg"
            Image.open(path).convert("RGB").save(staged, quality=96)
            mapping.append({"index": index, "dataset_name": path.name, "sam_frame": staged.name})
        (job / "frame-map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        state.update({
            "frames": [str(sam_frames / item["sam_frame"]) for item in mapping],
            "mapping": mapping,
            "registered": registered,
            "decoded": count,
            "sparse_points": sparse_points,
            "complete_stage": 2,
        })
        (job / "job.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        progress(1, desc="Ready for subject selection")
        first = Path(state["frames"][0])
        status = (
            f"**Ready to select a target.** COLMAP aligned **{registered}/{count}** sampled "
            f"views with **{sparse_points:,}** sparse 3D points. Brush over the object or person "
            "in one frame; a rough interior stroke is enough for SAM 2.1."
        )
        yield (
            state, gr.update(minimum=0, maximum=len(mapping) - 1, value=0, step=1),
            _editor_value(first), _gallery(_frame_paths(state)), status,
            _stage_panel(
                "Prepare + camera alignment", "complete",
                f"{registered}/{count} sampled views registered; {sparse_points:,} sparse points.",
            ), str(log_path), _tail(log_path),
        )
    except Exception as error:
        _append_log(log_path, f"STOP: {error}")
        yield (
            state, gr.update(), None, _gallery(sorted(input_dir.glob("*.jpg"))),
            f"**Preparation stopped:** {error}",
            _stage_panel("Prepare + camera alignment", "failed", str(error)),
            str(log_path), _tail(log_path),
        )


def load_target_frame(state: dict | None, frame_index: int):
    if not state or not state.get("frames"):
        return None, "Prepare a video first.", {}, "No saved keyframes."
    frames = _frame_paths(state)
    index = max(0, min(int(frame_index), len(frames) - 1))
    raw_mask = Path(state["job"]) / "sam2_raw" / f"{index:05d}.png"
    saved = read_keyframes(state["job"]).get(str(index))
    selected = Path(state["job"]) / "subject-keyframes" / saved["file"] if saved else raw_mask
    editor = _editor_value(frames[index], selected if selected.is_file() else None)
    mode = saved["mode"] if saved else ("mask" if raw_mask.is_file() else "points")
    baseline = np.asarray(editor["layers"][0])[..., 3] > 8
    context = dict(job=state["job"], index=index, mode=mode, digest=mask_digest(baseline))
    note = f"Editing registered frame **{index}/{len(frames) - 1}** (`{frames[index].name}`)."
    if saved:
        note += " Restored your saved keyframe (" + ("full mask" if mode == "mask" else "interior brush hint") + ")."
    elif raw_mask.is_file():
        note += " The red layer is SAM 2.1's current mask; erase mistakes or brush missing areas, then propagate again."
    else:
        note += " Paint a rough stroke inside the target."
    note += " Edits save when you switch frames or propagate; use Save keyframe to pin this selection."
    return editor, note, context, keyframe_summary(state["job"])


def resume_prepared_video(job_name):
    job = (JOBS / str(job_name)).resolve()
    if job.parent != JOBS.resolve() or not (job / "job.json").is_file():
        raise gr.Error("Choose a prepared video from this app.")
    state = json.loads((job / "job.json").read_text(encoding="utf-8"))
    frames = _frame_paths(state)
    if not frames or not all(frame.is_file() for frame in frames):
        raise gr.Error("The saved job's prepared frames are missing. Prepare the video again.")
    state["job"] = str(job)
    log = _log_path(state)
    return (state, gr.update(minimum=0, maximum=len(frames)-1, value=0, step=1),
            _editor_value(frames[0]), _gallery(frames),
            f"**Resumed {job.name}.** Open Select + track to continue. No alignment or tracking was rerun.",
            _stage_panel("Prepare + camera alignment", "complete", f"Restored {len(frames)} prepared frames."),
            str(log) if log.is_file() else None, _tail(log), state.get("quality", "Preview"))


def _background_matches(editor, frame):
    if not editor:
        return False
    # SubjectKeyframes' decoder fallback intentionally drops only a corrupt
    # browser-cache background. The frame identity still arrives in context.
    if editor.get("background") is None:
        return True
    received = np.asarray(editor["background"])[..., :3]
    with Image.open(frame) as image:
        expected = np.asarray(image.convert("RGB"))
    # Browser canvas round-trips can alter a few edge pixels even with PNG.
    return received.shape == expected.shape and float(np.abs(
        received.astype(np.float32) - expected).mean()) < .5


def save_target_keyframe(state, editor, context, force=True):
    if not state or not state.get("frames") or not context:
        raise gr.Error("Prepare a video and load a target frame first.")
    if context["job"] != state["job"] or not _background_matches(editor, _frame_paths(state)[context["index"]]):
        raise gr.Error("The editor is still changing frames. Wait for the frame to load, then try again.")
    index = context["index"]
    mask = _mask_from_editor(editor, _frame_paths(state)[index], allow_empty=True)
    digest = mask_digest(mask)
    changed = digest != context["digest"]
    if changed or force:
        if context["mode"] == "points" and not mask.any():
            if changed:
                remove_keyframe(state["job"], index)
            elif force:
                raise gr.Error("Paint inside the target before saving a keyframe.")
        elif context["mode"] == "points" and mask.sum() < 16:
            raise gr.Error("The brush selection is too small. Add a clearer stroke before switching frames.")
        else:
            save_keyframe(state["job"], index, mask, context["mode"])
    return dict(context, digest=digest), keyframe_summary(state["job"])


def switch_target_frame(state, frame_index, editor, context):
    if context and state and context.get("job") == state.get("job"):
        save_target_keyframe(state, editor, context, force=False)
    return load_target_frame(state, frame_index)


def forget_target_keyframe(state, frame_index, editor, context):
    if not context or not state or context.get("job") != state.get("job"):
        raise gr.Error("Load a target frame first.")
    if int(frame_index) != context["index"] or not _background_matches(editor, _frame_paths(state)[context["index"]]):
        raise gr.Error("Wait for the selected frame to load before removing its keyframe.")
    remove_keyframe(state["job"], context["index"])
    return load_target_frame(state, context["index"])


def _mask_from_editor(editor: dict | None, frame: Path, allow_empty=False) -> np.ndarray:
    if not editor:
        raise gr.Error("Paint over the target before running SAM 2.1.")
    layers = editor.get("layers") or []
    if not layers:
        if allow_empty:
            with Image.open(frame) as image:
                return np.zeros((image.height, image.width), dtype=bool)
        raise gr.Error("No brush stroke was found. Paint over the target first.")
    combined = None
    for layer in layers:
        array = np.asarray(layer)
        if array.ndim == 3 and array.shape[2] >= 4:
            alpha = array[..., 3]
        elif array.ndim == 3:
            alpha = np.max(array[..., :3], axis=2)
        else:
            alpha = array
        layer_mask = alpha > 8
        combined = layer_mask if combined is None else combined | layer_mask
    if combined is None or (not allow_empty and int(combined.sum()) < 16):
        raise gr.Error("The brush selection is too small. Add a clearer stroke inside the target.")
    with Image.open(frame) as image:
        target_size = image.size
    if combined.shape != target_size[::-1]:
        combined = np.asarray(
            Image.fromarray(combined.astype(np.uint8) * 255).resize(target_size, Image.Resampling.NEAREST)
        ) > 127
    return combined


def _positive_points_from_brush(mask: np.ndarray, maximum: int = 10) -> np.ndarray:
    """Turn an interior brush stroke into spatially distributed SAM point prompts."""
    ys, xs = np.where(mask)
    coordinates = np.column_stack((xs, ys)).astype(np.float32)
    if len(coordinates) <= maximum:
        return coordinates
    # Keep farthest points across the stroke so a long head-to-foot brush gives
    # SAM evidence over the whole person rather than many near-duplicate clicks.
    if len(coordinates) > 20_000:
        coordinates = coordinates[::max(1, len(coordinates) // 20_000)]
    center = coordinates.mean(axis=0)
    first = int(np.argmin(np.square(coordinates - center).sum(axis=1)))
    selected = [first]
    distance = np.square(coordinates - coordinates[first]).sum(axis=1)
    while len(selected) < maximum:
        next_index = int(np.argmax(distance))
        selected.append(next_index)
        candidate_distance = np.square(coordinates - coordinates[next_index]).sum(axis=1)
        distance = np.minimum(distance, candidate_distance)
    return coordinates[selected]


def _overlay(frame: Path, mask: Path, output: Path, color=(88, 255, 140)) -> tuple[float, float, float]:
    image = np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32)
    matte = np.asarray(Image.open(mask).convert("L")) > 127
    if matte.shape != image.shape[:2]:
        matte = np.asarray(Image.fromarray(matte).resize((image.shape[1], image.shape[0]))) > 0
    display = image.copy()
    tint = np.asarray(color, dtype=np.float32)
    display[matte] = display[matte] * 0.48 + tint * 0.52
    Image.fromarray(np.clip(display, 0, 255).astype(np.uint8)).save(output)
    coverage = float(matte.mean())
    ys, xs = np.where(matte)
    cx = float(xs.mean() / max(1, matte.shape[1])) if len(xs) else 0.0
    cy = float(ys.mean() / max(1, matte.shape[0])) if len(ys) else 0.0
    return coverage, cx, cy


def _mask_debug_outputs(state: dict, mask_dir: Path, tag: str) -> tuple[list, list, dict]:
    frames = _frame_paths(state)
    output_dir = Path(state["job"]) / f"{tag}_overlays"
    output_dir.mkdir(exist_ok=True)
    metrics = []
    overlays = []
    masks = []
    previous = None
    for index, frame in enumerate(frames):
        mask = mask_dir / f"{index:05d}.png"
        if not mask.is_file():
            continue
        overlay = output_dir / f"{index:05d}.jpg"
        coverage, cx, cy = _overlay(frame, mask, overlay)
        jump = 0.0 if previous is None else float(((cx - previous[0]) ** 2 + (cy - previous[1]) ** 2) ** 0.5)
        previous = (cx, cy)
        warning = " ⚠ jump" if jump > 0.18 else ""
        caption = f"F{index:03d} · area {coverage:.1%} · jump {jump:.3f}{warning}"
        overlays.append((str(overlay), caption))
        masks.append((str(mask), f"F{index:03d} mask · {coverage:.1%}"))
        metrics.append({"frame": index, "coverage": coverage, "centroid_x": cx,
                        "centroid_y": cy, "centroid_jump": jump, "warning": jump > 0.18})
    report = {
        "model": tag,
        "frames": len(metrics),
        "mean_coverage": float(np.mean([row["coverage"] for row in metrics])) if metrics else 0.0,
        "large_centroid_jumps": sum(bool(row["warning"]) for row in metrics),
        "per_frame": metrics,
    }
    (Path(state["job"]) / f"{tag}-mask-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (Path(state["job"]) / f"{tag}-mask-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]) if metrics else ["frame"])
        writer.writeheader()
        writer.writerows(metrics)
    return overlays, masks, report


def run_sam2(state: dict | None, frame_index: int, editor: dict | None, context: dict, progress=gr.Progress()):
    """Save current edits and use every persistent keyframe in one tracking session."""
    if not state or not state.get("frames"):
        raise gr.Error("Prepare a video first.")
    if not SAM2_CHECKPOINT.is_file() or not SAM2_CONFIG.is_file():
        raise gr.Error("The SAM 2.1 checkpoint or configuration is missing.")
    frames = _frame_paths(state)
    if not context or int(frame_index) != context["index"]:
        raise gr.Error("Wait for the selected frame to load before propagating.")
    save_target_keyframe(state, editor, context, force=False)
    job = Path(state["job"])
    prompts = load_keyframe_prompts(job)
    if not prompts:
        raise gr.Error("Paint a target or save a keyframe before propagating.")
    revision = keyframe_revision(job)
    indices = [index for index, _, _ in prompts]
    raw = job / "sam2_raw"
    _append_log(
        _log_path(state),
        f"\n### SAM 2.1\nUsing {len(prompts)} saved keyframes: {indices}\n"
        + "\n".join(f"prompt frame={i}, mode={mode}, painted pixels={int(mask.sum())}"
                     for i, mode, mask in prompts),
    )

    if str(SAM2_REPO) not in sys.path:
        sys.path.insert(0, str(SAM2_REPO))
    from sam2.build_sam import build_sam2_video_predictor
    import torch

    progress(0.04, desc="Loading SAM 2.1")
    predictor = build_sam2_video_predictor(str(SAM2_CONFIG), str(SAM2_CHECKPOINT), device="cuda")
    tracked: dict[int, np.ndarray] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference = predictor.init_state(video_path=str(job / "sam2_frames"))
        register_keyframes(predictor, inference, prompts, _positive_points_from_brush)
        for reverse in (False, True):
            direction = "backward" if reverse else "forward"
            for frame_id, _object_ids, logits in predictor.propagate_in_video(
                inference, start_frame_idx=min(indices), reverse=reverse
            ):
                tracked[int(frame_id)] = (logits[0, 0] > 0).cpu().numpy()
                progress(len(tracked) / max(1, len(frames)), desc=f"SAM 2.1 {direction}: {len(tracked)}/{len(frames)}")
    del predictor
    torch.cuda.empty_cache()

    if len(tracked) != len(frames):
        raise gr.Error(f"SAM 2.1 returned {len(tracked)}/{len(frames)} frame masks.")
    # Explicit full-mask corrections remain exact at their authored frames.
    for index, mode, mask in prompts:
        if mode == "mask":
            tracked[index] = mask
    raw.mkdir(exist_ok=True)
    for frame_id, mask in tracked.items():
        Image.fromarray(mask.astype(np.uint8) * 255).save(raw / f"{frame_id:05d}.png")
    (job / "sam2-keyframes-used.json").write_text(
        json.dumps(dict(frames=indices, revision=revision), indent=2), encoding="utf-8")
    overlays, masks, report = _mask_debug_outputs(state, raw, "sam2")
    state = dict(state)
    state.update({"mask_source": "sam2", "mask_dir": str(raw), "complete_stage": 4})
    (job / "job.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    _append_log(_log_path(state), json.dumps({key: value for key, value in report.items() if key != "per_frame"}))
    status = (
        f"**SAM 2.1 tracked {len(tracked)} frames using all {len(prompts)} saved keyframes.** Mean selected area: "
        f"**{report['mean_coverage']:.1%}**; large centroid jumps: "
        f"**{report['large_centroid_jumps']}**. Inspect every overlay. To fix a frame, "
        "move the frame slider, erase/add paint on the red layer, and run SAM 2.1 again."
    )
    return state, overlays, masks, status, _stage_panel(
        "SAM 2.1 identity tracking", "complete",
        f"{len(tracked)} masks; {report['large_centroid_jumps']} large centroid jumps.",
    ), str(_log_path(state)), _tail(_log_path(state))


def refine_matanyone2(state: dict | None, progress=gr.Progress()):
    """Run optional MatAnyone2 refinement from SAM's first-frame mask."""
    if not state or not state.get("mask_dir"):
        raise gr.Error("Run SAM 2.1 tracking first.")
    require_applied_keyframes(state["job"])
    for required in (MA2_PYTHON, MA2_REPO, MA2_CHECKPOINT):
        if not required.exists():
            raise gr.Error(f"MatAnyone2 dependency is missing: {required}")
    job = Path(state["job"])
    raw = Path(state["mask_dir"])
    seed = raw / "00000.png"
    if not seed.is_file():
        raise gr.Error("SAM 2.1 did not produce a frame-zero mask for MatAnyone2.")
    checkpoint = MA2_REPO / "pretrained_models" / "matanyone2.pth"
    checkpoint.parent.mkdir(exist_ok=True)
    if not checkpoint.exists():
        try:
            os.link(MA2_CHECKPOINT, checkpoint)
        except OSError:
            shutil.copy2(MA2_CHECKPOINT, checkpoint)
    output = job / "matanyone2"
    progress(0.1, desc="MatAnyone2: refining temporal alpha")
    _run_logged([
        MA2_PYTHON, MA2_REPO / "inference_matanyone2.py", "-i", job / "sam2_frames",
        "-m", seed, "-o", output, "-c", checkpoint,
        "-w", "10", "-e", "10", "-d", "10", "--suffix", "refined", "--save_image",
    ], MA2_REPO, _log_path(state), "MatAnyone2 alpha refinement")
    candidates = sorted(output.rglob("pha"))
    if not candidates:
        raise gr.Error("MatAnyone2 finished without an alpha directory.")
    alphas = sorted(candidates[-1].glob("*.png"))
    frames = _frame_paths(state)
    if len(alphas) < len(frames):
        raise gr.Error(f"MatAnyone2 returned {len(alphas)}/{len(frames)} alpha mattes.")
    refined = job / "matanyone2_masks"
    refined.mkdir(exist_ok=True)
    for index, (alpha, frame) in enumerate(zip(alphas, frames)):
        with Image.open(frame) as image:
            size = image.size
        matte = Image.open(alpha).convert("L")
        if matte.size != size:
            matte = matte.resize(size, Image.Resampling.BILINEAR)
        matte.save(refined / f"{index:05d}.png")
    overlays, masks, report = _mask_debug_outputs(state, refined, "matanyone2")
    state = dict(state)
    state.update({"mask_source": "matanyone2", "mask_dir": str(refined), "complete_stage": 5})
    (job / "job.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    progress(1, desc="MatAnyone2 masks ready")
    status = (
        f"**MatAnyone2 refined {len(frames)} masks.** Mean selected area: "
        f"**{report['mean_coverage']:.1%}**; large centroid jumps: "
        f"**{report['large_centroid_jumps']}**. Compare these overlays with the SAM 2.1 tab; "
        "the active training-mask source is now MatAnyone2."
    )
    return state, overlays, masks, status, _stage_panel(
        "MatAnyone2 alpha refinement", "complete",
        f"{len(frames)} soft alpha mattes generated.",
    ), str(_log_path(state)), _tail(_log_path(state))


def _install_training_masks(state: dict) -> None:
    job = Path(state["job"])
    source = Path(state["mask_dir"])
    target = job / "dataset" / "masks"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir()
    for item in state["mapping"]:
        src = source / f"{item['index']:05d}.png"
        if not src.is_file():
            raise RuntimeError(f"Missing mask for frame {item['index']:05d}.")
        image_path = job / "dataset" / "images" / item["dataset_name"]
        with Image.open(image_path) as image:
            size = image.size
        matte = Image.open(src).convert("L")
        if matte.size != size:
            matte = matte.resize(size, Image.Resampling.BILINEAR)
        matte.save(target / f"{Path(item['dataset_name']).stem}.png")


def train_masked_splat(state: dict | None, quality: str, progress=gr.Progress()):
    """Train Brush with the reviewed per-view masks."""
    if not state or not state.get("mask_dir"):
        raise gr.Error("Review a SAM 2.1 or MatAnyone2 mask sequence first.")
    require_applied_keyframes(state["job"])
    if quality != state.get("quality"):
        raise gr.Error(
            f"This job was prepared at {state.get('quality')} quality. Change the quality back, "
            "or prepare the video again so alignment, masks, and training use one resolution."
        )
    job = Path(state["job"])
    result = job / "scene.ply"
    try:
        _install_training_masks(state)
        steps, resolution = QUALITY[quality]
        progress(0.05, desc=f"Brush: training {steps:,} steps")
        _run_logged([
            BRUSH, job / "dataset", "--total-steps", str(steps),
            "--max-resolution", str(resolution), "--max-splats", "3000000",
            "--export-every", str(steps), "--export-path", job, "--export-name", "scene.ply",
        ], job, _log_path(state), f"Brush masked training ({steps:,} steps @ {resolution}px)")
        if not result.is_file():
            raise RuntimeError("Brush finished without exporting scene.ply.")
        from plyfile import PlyData
        count = len(PlyData.read(str(result), mmap=True)["vertex"])
        state = dict(state)
        state.update({"raw_splat": str(result), "complete_stage": 6, "quality": quality})
        (job / "job.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        progress(1, desc="Masked splat ready")
        status = (
            f"**Masked Brush training finished:** `{result.name}` contains **{count:,} Gaussians**. "
            "This is an intermediate full scene; masked training can still leave hidden background "
            "Gaussians. Run the extraction stage to create the subject-only PLY."
        )
        return state, str(result), status, _stage_panel(
            "Brush masked training", "complete", f"{count:,} Gaussians exported."
        ), str(_log_path(state)), _tail(_log_path(state))
    except Exception as error:
        _append_log(_log_path(state), f"STOP: {error}")
        return state, None, f"**Training stopped:** {error}", _stage_panel(
            "Brush masked training", "failed", str(error)
        ), str(_log_path(state)), _tail(_log_path(state))


def extract_subject_splat(state: dict | None, progress=gr.Progress()):
    """Classify trained Gaussians using mask evidence from every camera view."""
    if not state or not state.get("raw_splat"):
        raise gr.Error("Train the masked splat first.")
    job = Path(state["job"])
    raw_splat = Path(state["raw_splat"])
    output = job / "subject-extraction"
    output.mkdir(exist_ok=True)
    sparse_txt = job / "sparse_txt"
    sparse_txt.mkdir(exist_ok=True)
    report = output / "report.json"
    try:
        progress(0.05, desc="Preparing camera data for extraction")
        _run_logged([
            COLMAP, "model_converter", "--input_path", job / "dataset" / "sparse",
            "--output_path", sparse_txt, "--output_type", "TXT",
        ], job, _log_path(state), "COLMAP model converter for extraction")
        progress(0.15, desc="Classifying Gaussians against every reviewed mask")
        _run_logged([
            sys.executable, ROOT / "work" / "h2_extract.py", raw_splat,
            "--sparse", sparse_txt, "--masks", job / "dataset" / "masks",
            "--out", output, "--report", report,
        ], ROOT, _log_path(state), "Mask-guided Gaussian extraction")
        final = output / "keep.ply"
        if not final.is_file():
            raise RuntimeError("Extraction finished without producing keep.ply.")
        final_named = job / f"{Path(state['source']).stem}-selected-subject.ply"
        shutil.copy2(final, final_named)
        data = json.loads(report.read_text(encoding="utf-8"))
        state = dict(state)
        state.update({"final_splat": str(final_named), "complete_stage": 7})
        (job / "job.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        progress(1, desc="Subject-only splat ready")
        status = (
            f"**Subject-only splat complete.** Kept **{data['n_keep']:,}** of "
            f"**{data['n']:,}** Gaussians; **{data['n_uncertain']:,}** boundary Gaussians were "
            f"saved separately for debugging. Final file: `{final_named.name}`."
        )
        uncertain = output / "uncertain.ply"
        hidden = output / "hide.ply"
        return state, str(final_named), (str(uncertain) if uncertain.is_file() else None), (str(hidden) if hidden.is_file() else None), status, _stage_panel(
            "Mask-guided Gaussian extraction", "complete",
            f"Kept {data['n_keep']:,} of {data['n']:,} Gaussians.",
        ), str(report), str(_log_path(state)), _tail(_log_path(state))
    except Exception as error:
        _append_log(_log_path(state), f"STOP: {error}")
        return state, None, None, None, f"**Extraction stopped:** {error}", _stage_panel(
            "Mask-guided Gaussian extraction", "failed", str(error)
        ), None, str(_log_path(state)), _tail(_log_path(state))


PREPARE_PANEL = _stage_panel("Prepare + camera alignment", "waiting")
SAM_PANEL = _stage_panel("SAM 2.1 identity tracking", "waiting")
REFINE_PANEL = _stage_panel("MatAnyone2 alpha refinement", "waiting")
TRAIN_PANEL = _stage_panel("Brush masked training", "waiting")
EXTRACT_PANEL = _stage_panel("Mask-guided Gaussian extraction", "waiting")
