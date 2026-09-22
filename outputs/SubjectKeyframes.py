"""Persistent user prompts, separate from disposable tracking predictions."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from gradio.components.image_editor import ImageEditor
from gradio.data_classes import FileData


IMAGE_DECODE_ERRORS = (OSError, SyntaxError)


def preprocess_editor_safely(component, payload, preprocess):
    # Gradio may read an image while a concurrent canvas upload replaces the
    # same cache file. Retry only image decoding/I/O failures, with a bounded wait.
    for attempt in range(5):
        try:
            return preprocess(payload)
        except IMAGE_DECODE_ERRORS as error:
            if attempt == 4:
                original_error = error
                break
            time.sleep(.1)
    # The canvas uploads background and paint layers separately. A corrupt
    # transient background must not discard a valid brush layer: the source
    # frame is already identified by the submitted keyframe context.
    source = component.blob_storage.get(payload.id) if payload.id else payload
    source = source or payload
    if source is None:
        raise original_error
    for attempt in range(5):
        try:
            layers = [component.convert_and_format_image(layer) for layer in source.layers]
            break
        except IMAGE_DECODE_ERRORS:
            if attempt == 4:
                raise
            time.sleep(.1)
    if payload.id:
        component.blob_storage.pop(payload.id, None)
    return {"background": None, "layers": layers, "composite": None}


def install_resilient_image_editor():
    """Patch Gradio's decoder class before events capture their input components."""
    if getattr(ImageEditor.preprocess, "_splatim_resilient", False):
        return
    original_preprocess = ImageEditor.preprocess
    original_convert = ImageEditor.convert_and_format_image

    def convert_from_snapshot(component, file):
        # Decode a memory snapshot instead of leaving Pillow attached to the
        # browser cache path. On Windows a failed PNG decode otherwise keeps
        # the still-growing file locked and prevents the upload from finishing.
        if isinstance(file, FileData):
            file = Path(file.path).read_bytes()
        return original_convert(component, file)

    def preprocess_resiliently(component, payload):
        return preprocess_editor_safely(
            component, payload, lambda submitted: original_preprocess(component, submitted)
        )

    preprocess_resiliently._splatim_resilient = True
    ImageEditor.convert_and_format_image = convert_from_snapshot
    ImageEditor.preprocess = preprocess_resiliently


def mask_digest(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def read_keyframes(job):
    path = Path(job) / "subject-keyframes" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def keyframe_revision(job):
    return hashlib.sha256(json.dumps(read_keyframes(job), sort_keys=True).encode()).hexdigest()


def _write_manifest(directory, entries):
    temporary = directory / f"manifest-{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    os.replace(temporary, directory / "manifest.json")


def save_keyframe(job, index, mask, mode):
    if mode not in ("points", "mask"):
        raise ValueError("Unknown keyframe prompt mode.")
    directory = Path(job) / "subject-keyframes"
    directory.mkdir(exist_ok=True)
    entries = read_keyframes(job)
    digest = mask_digest(mask)
    old = entries.get(str(index))
    if old and old["digest"] == digest and old["mode"] == mode:
        return
    # Keep previous revisions on disk so replacing/removing a keyframe is recoverable.
    name = f"{index:05d}-{uuid.uuid4().hex}.png"
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(directory / name)
    entries[str(index)] = dict(file=name, mode=mode, digest=digest)
    _write_manifest(directory, entries)


def remove_keyframe(job, index):
    directory = Path(job) / "subject-keyframes"
    entries = read_keyframes(job)
    if entries.pop(str(index), None) is not None:
        _write_manifest(directory, entries)


def load_keyframe_prompts(job):
    entries = read_keyframes(job)
    prompts = []
    for key in sorted(entries, key=int):
        entry = entries[key]
        with Image.open(Path(job) / "subject-keyframes" / entry["file"]) as image:
            mask = np.asarray(image.convert("L")) > 127
        if mask_digest(mask) != entry["digest"]:
            raise ValueError(f"Saved keyframe {key} failed its integrity check.")
        prompts.append((int(key), entry["mode"], mask))
    return prompts


def keyframe_summary(job):
    entries = read_keyframes(job)
    frames = ", ".join(f"F{int(key):03d}" for key in sorted(entries, key=int)) or "none"
    last = Path(job) / "sam2-keyframes-used.json"
    used = json.loads(last.read_text(encoding="utf-8")) if last.is_file() else {}
    used_frames = ", ".join(f"F{index:03d}" for index in used.get("frames", [])) or "none"
    current = used.get("revision") == keyframe_revision(job)
    suffix = "Tracking is up to date with saved keyframes." if current else "Propagate to apply the saved keyframes."
    return (f"**{len(entries)} saved keyframes:** {frames}.\n\n"
            f"**Last successful tracking used {len(used.get('frames', []))}:** {used_frames}. {suffix}")


def require_applied_keyframes(job):
    """Never silently train/refine an old sequence after saved prompts change."""
    if not (Path(job) / "subject-keyframes" / "manifest.json").is_file():
        return  # Existing single-frame jobs remain usable.
    path = Path(job) / "sam2-keyframes-used.json"
    used = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if used.get("revision") != keyframe_revision(job):
        raise ValueError("Saved keyframes changed. Propagate all saved keyframes in Select + track before continuing.")


def register_keyframes(predictor, inference, prompts, points_from_brush):
    """Register every prompt in the same fresh inference state, before propagation."""
    for index, mode, mask in prompts:
        if mode == "mask":
            predictor.add_new_mask(inference_state=inference, frame_idx=index, obj_id=1, mask=mask)
        else:
            points = points_from_brush(mask)
            predictor.add_new_points_or_box(inference_state=inference, frame_idx=index, obj_id=1,
                                            points=points, labels=np.ones(len(points), dtype=np.int32))
