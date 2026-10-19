"""Keyframe persistence and pipeline integration without GPU/model downloads."""
import json
import sys
import tempfile
import types
import unittest
from io import BytesIO
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
from PIL import Image
from gradio.components.image_editor import EditorData, ImageEditor
from gradio.data_classes import FileData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
import VideoSubjectSplat as pipeline
from SubjectKeyframes import (read_keyframes, load_keyframe_prompts, keyframe_summary,
                              save_keyframe, remove_keyframe, require_applied_keyframes,
                              preprocess_editor_safely, install_resilient_image_editor,
                              register_keyframes)


class KeyframeTests(unittest.TestCase):
    def test_editor_cache_retry_is_bounded_and_does_not_swallow_failure(self):
        component = Mock()
        decoder = Mock(side_effect=[SyntaxError("broken PNG chunk"), {"layers": []}])
        with patch("SubjectKeyframes.time.sleep"):
            self.assertEqual(preprocess_editor_safely(component, Mock(id=None), decoder), {"layers": []})
            broken = Mock()
            broken_decoder = Mock(side_effect=SyntaxError("invalid image"))
            broken.blob_storage = {}
            broken.convert_and_format_image = Mock(side_effect=SyntaxError("invalid layer"))
            with self.assertRaises(SyntaxError):
                preprocess_editor_safely(broken, Mock(id=None, layers=[object()]), broken_decoder)
            self.assertEqual(broken_decoder.call_count, 5)

    def test_corrupt_canvas_background_keeps_valid_brush_layer(self):
        with tempfile.TemporaryDirectory() as folder, patch("SubjectKeyframes.time.sleep"):
            folder = Path(folder)
            background = folder / "broken.png"
            Image.new("RGB", (120, 90), "green").save(background)
            complete_png = background.read_bytes()
            background.write_bytes(complete_png[:len(complete_png) // 2])
            layer = folder / "layer.png"
            rgba = np.zeros((12, 9, 4), dtype=np.uint8)
            rgba[2:8, 3:7, 3] = 255
            Image.fromarray(rgba).save(layer)
            install_resilient_image_editor()
            component = ImageEditor(type="numpy", image_mode="RGBA", format="png")
            payload = EditorData(background=FileData(path=str(background)),
                                 layers=[FileData(path=str(layer))])
            with self.assertRaisesRegex(OSError, "truncated"):
                with Image.open(BytesIO(background.read_bytes())) as truncated:
                    truncated.convert("RGBA")
            result = component.preprocess(payload)
            self.assertIsNone(result["background"])
            self.assertTrue(getattr(component.preprocess, "_splatim_resilient", False))
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
                logits = torch.from_numpy(self.mask.astype(np.float32) * 2 - 1)[None, None]
                return kwargs["frame_idx"], [1], logits

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

    def test_inverted_dense_prompt_falls_back_to_saved_mask_before_propagation(self):
        import torch
        calls = []

        class Predictor:
            def add_new_points_or_box(inner, **kwargs):
                calls.append(("points", kwargs["frame_idx"]))
                inverted = np.logical_not(self.mask)
                logits = torch.from_numpy(inverted.astype(np.float32) * 2 - 1)[None, None]
                return kwargs["frame_idx"], [1], logits

            def add_new_mask(inner, **kwargs):
                calls.append(("mask", kwargs["frame_idx"]))

        report = register_keyframes(
            Predictor(), {}, [(4, "points", self.mask)], pipeline._positive_points_from_brush
        )
        self.assertEqual(calls, [("points", 4), ("mask", 4)])
        self.assertEqual(report["fallback_frames"], [4])
        self.assertEqual(report["frames"][0]["decision"], "exact-mask fallback")
        self.assertGreater(report["frames"][0]["complement_iou"], report["frames"][0]["iou"])

    def test_matanyone_quality_gate_rejects_collapse_and_accepts_stable_refinement(self):
        source = {
            "mean_coverage": 0.09,
            "large_centroid_jumps": 0,
            "per_frame": [{"frame": index, "coverage": 0.09} for index in range(20)],
        }
        collapsed = {
            "mean_coverage": 0.011,
            "large_centroid_jumps": 9,
            "per_frame": [{"frame": index, "coverage": 0.001} for index in range(20)],
        }
        rejected = pipeline._matanyone_quality_gate(source, collapsed)
        self.assertFalse(rejected["accepted"])
        self.assertEqual(len(rejected["collapsed_frames"]), 20)
        self.assertGreaterEqual(len(rejected["reasons"]), 2)

        stable = {
            "mean_coverage": 0.082,
            "large_centroid_jumps": 2,
            "per_frame": [{"frame": index, "coverage": 0.082} for index in range(20)],
        }
        accepted = pipeline._matanyone_quality_gate(source, stable)
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["reasons"], [])

    def test_resumed_collapsed_matanyone_job_falls_back_to_sam_before_training(self):
        sam_report = {
            "mean_coverage": 0.1,
            "per_frame": [{"frame": index, "coverage": 0.1} for index in range(8)],
        }
        refined_report = {
            "mean_coverage": 0.005,
            "large_centroid_jumps": 6,
            "per_frame": [{"frame": index, "coverage": 0.001} for index in range(8)],
        }
        (self.job / "sam2_raw").mkdir()
        (self.job / "sam2-mask-report.json").write_text(json.dumps(sam_report))
        (self.job / "matanyone2-mask-report.json").write_text(json.dumps(refined_report))
        state = dict(self.state, mask_source="matanyone2",
                     mask_dir=str(self.job / "matanyone2_masks"), raw_splat="old.ply")
        fallback, quality = pipeline._reject_stale_matanyone_source(state)
        self.assertFalse(quality["accepted"])
        self.assertEqual(fallback["mask_source"], "sam2")
        self.assertEqual(Path(fallback["mask_dir"]), self.job / "sam2_raw")
        self.assertNotIn("raw_splat", fallback)


if __name__ == "__main__":
    unittest.main()
