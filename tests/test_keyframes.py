"""Keyframe persistence and pipeline integration without GPU/model downloads."""
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
from PIL import Image
import gradio as gr
from gradio.components.image_editor import EditorData
from gradio.data_classes import FileData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
import VideoSubjectSplat as pipeline
from SubjectKeyframes import (read_keyframes, load_keyframe_prompts, keyframe_summary,
                              save_keyframe, remove_keyframe, require_applied_keyframes,
                              preprocess_editor_safely)


class KeyframeTests(unittest.TestCase):
    def test_editor_cache_retry_is_bounded_and_does_not_swallow_failure(self):
        component = Mock()
        component.preprocess_original = Mock(side_effect=[SyntaxError("broken PNG chunk"), {"layers": []}])
        with patch("SubjectKeyframes.time.sleep"):
            self.assertEqual(preprocess_editor_safely(component, Mock(id=None)), {"layers": []})
            broken = Mock()
            broken.preprocess_original = Mock(side_effect=SyntaxError("invalid image"))
            broken.blob_storage = {}
            broken.convert_and_format_image = Mock(side_effect=SyntaxError("invalid layer"))
            with self.assertRaises(SyntaxError):
                preprocess_editor_safely(broken, Mock(id=None, layers=[object()]))
            self.assertEqual(broken.preprocess_original.call_count, 5)

    def test_corrupt_canvas_background_keeps_valid_brush_layer(self):
        with tempfile.TemporaryDirectory() as folder, patch("SubjectKeyframes.time.sleep"):
            folder = Path(folder)
            background = folder / "broken.png"
            background.write_bytes(b"not a png")
            layer = folder / "layer.png"
            rgba = np.zeros((12, 9, 4), dtype=np.uint8)
            rgba[2:8, 3:7, 3] = 255
            Image.fromarray(rgba).save(layer)
            component = gr.ImageEditor(type="numpy", image_mode="RGBA", format="png")
            component.preprocess_original = component.preprocess
            payload = EditorData(background=FileData(path=str(background)),
                                 layers=[FileData(path=str(layer))])
            result = preprocess_editor_safely(component, payload)
            self.assertIsNone(result["background"])
            np.testing.assert_array_equal(result["layers"][0][..., 3] > 0, rgba[..., 3] > 0)
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.job = Path(self.temporary.name)
        frames = self.job / "sam2_frames"
        frames.mkdir()
        for index in range(8):
            Image.new("RGB", (24, 32), (index * 20, 90, 80)).save(frames / f"{index:05d}.jpg")
        self.state = dict(job=str(self.job), frames=[str(p) for p in sorted(frames.glob("*.jpg"))])
        self.mask = np.zeros((32, 24), dtype=bool)
        self.mask[8:24, 8:16] = True

    def paint(self, editor, mask=None):
        rgba = np.zeros((32, 24, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 3] = np.asarray(self.mask if mask is None else mask) * 115
        return dict(editor, layers=[rgba])

    def test_six_frame_edits_autosave_restore_and_keep_prompt_modes(self):
        editor, _, context, _ = pipeline.load_target_frame(self.state, 0)
        for index in range(6):
            editor = self.paint(editor)
            editor, _, context, summary = pipeline.switch_target_frame(self.state, index+1, editor, context)
        self.assertIn("6 saved keyframes", summary)
        self.assertEqual(len(read_keyframes(self.job)), 6)
        # Visiting an unedited frame does not create an accidental constraint.
        editor, _, context, _ = pipeline.switch_target_frame(self.state, 2, editor, context)
        np.testing.assert_array_equal(np.asarray(editor["layers"][0])[..., 3] > 8, self.mask)
        raw = self.job / "sam2_raw"
        raw.mkdir()
        Image.fromarray(self.mask.astype("uint8") * 255).save(raw / "00002.png")
        _, _, restored, _ = pipeline.load_target_frame(self.state, 2)
        self.assertEqual(restored["mode"], "points")
        self.assertEqual(len(read_keyframes(self.job)), 6)

    def test_full_mask_correction_empty_occlusion_and_remove_are_persistent(self):
        raw = self.job / "sam2_raw"
        raw.mkdir()
        Image.fromarray(self.mask.astype("uint8") * 255).save(raw / "00003.png")
        editor, _, context, _ = pipeline.load_target_frame(self.state, 3)
        self.assertEqual(context["mode"], "mask")
        editor = self.paint(editor, np.zeros_like(self.mask))
        context, _ = pipeline.save_target_keyframe(self.state, editor, context)
        prompt = load_keyframe_prompts(self.job)[0]
        self.assertEqual(prompt[:2], (3, "mask"))
        self.assertFalse(prompt[2].any())
        remove_keyframe(self.job, 3)
        self.assertEqual(read_keyframes(self.job), {})
        self.assertEqual(len(list((self.job / "subject-keyframes").glob("*.png"))), 1)

    def test_wrong_frame_editor_is_rejected_and_pending_edits_block_training(self):
        editor, _, context, _ = pipeline.load_target_frame(self.state, 0)
        wrong, _, _, _ = pipeline.load_target_frame(self.state, 1)
        with self.assertRaises(Exception):
            pipeline.save_target_keyframe(self.state, self.paint(wrong), context)
        pipeline.save_target_keyframe(self.state, self.paint(editor), context)
        with self.assertRaisesRegex(ValueError, "Propagate all"):
            require_applied_keyframes(self.job)

    def test_background_allows_tiny_canvas_roundtrip_changes_but_not_padding(self):
        editor, _, context, _ = pipeline.load_target_frame(self.state, 0)
        background = np.asarray(editor["background"]).copy()
        background[0, 0, 0] = 40
        editor = dict(self.paint(editor), background=background)
        pipeline.save_target_keyframe(self.state, editor, context)
        with self.assertRaises(Exception):
            pipeline.save_target_keyframe(self.state,
                dict(editor, background=np.zeros((32, 32, 3), dtype=np.uint8)), context)

    def test_resume_restores_existing_frames_and_keyframes_without_processing(self):
        (self.job / "job.json").write_text(json.dumps(dict(self.state, quality="High quality")))
        save_keyframe(self.job, 4, self.mask, "mask")
        with patch.object(pipeline, "JOBS", self.job.parent):
            result = pipeline.resume_prepared_video(self.job.name)
            self.assertEqual(result[0]["frames"], self.state["frames"])
            self.assertEqual(result[-1], "High quality")
            self.assertIn("1 saved keyframes", pipeline.load_target_frame(result[0], 4)[-1])
            with self.assertRaises(Exception):
                pipeline.resume_prepared_video("../outside")

    def test_pipeline_registers_all_six_before_propagation_and_keeps_corrections(self):
        import torch
        for index in range(1, 7):
            save_keyframe(self.job, index, self.mask, "points" if index % 2 else "mask")
        editor, _, context, _ = pipeline.load_target_frame(self.state, 3)
        calls = []

        class Predictor:
            def init_state(inner, **kwargs):
                return {}

            def add_new_points_or_box(inner, **kwargs):
                calls.append((kwargs["frame_idx"], "points"))

            def add_new_mask(inner, **kwargs):
                calls.append((kwargs["frame_idx"], "mask"))

            def propagate_in_video(inner, inference, start_frame_idx, reverse):
                self.assertEqual(len(calls), 6)
                self.assertEqual(start_frame_idx, 1)
                indices = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, 8)
                for index in indices:
                    yield index, [1], torch.ones((1, 1, 32, 24))

        fake_module = types.ModuleType("sam2.build_sam")
        fake_module.build_sam2_video_predictor = lambda *a, **k: Predictor()
        checkpoint = self.job / "checkpoint"
        checkpoint.touch()
        with patch.dict(sys.modules, {"sam2.build_sam": fake_module}), \
             patch.object(pipeline, "SAM2_CONFIG", checkpoint), \
             patch.object(pipeline, "SAM2_CHECKPOINT", checkpoint), \
             patch.object(torch, "autocast", lambda *a, **k: nullcontext()), \
             patch.object(torch.cuda, "empty_cache"):
            result = pipeline.run_sam2(self.state, 3, editor, context, progress=lambda *a, **k: None)
        self.assertIn("all 6 saved keyframes", result[3])
        self.assertEqual(calls, [(i, "points" if i % 2 else "mask") for i in range(1, 7)])
        np.testing.assert_array_equal(np.asarray(Image.open(self.job / "sam2_raw/00002.png")) > 0, self.mask)
        self.assertEqual(len(list((self.job / "sam2_raw").glob("*.png"))), 8)
        self.assertIn("Last successful tracking used 6", keyframe_summary(self.job))
        require_applied_keyframes(self.job)
        remove_keyframe(self.job, 2)
        with self.assertRaises(ValueError):
            require_applied_keyframes(self.job)


if __name__ == "__main__":
    unittest.main()
