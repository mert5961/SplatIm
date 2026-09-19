"""Run with the configured sharp environment: python -m unittest discover -s tests."""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
from ProgressUI import present_job
from SubjectRecovery import export_recovery


class ProgressTests(unittest.TestCase):
    def test_disconnect_does_not_overlap_backend_workers(self):
        finished = threading.Event()
        def first(progress):
            time.sleep(1.2)
            finished.set()
            return ("done", "done", "done")
        def second(progress):
            if not finished.is_set():
                raise RuntimeError("Backend workers overlapped")
            return ("second done", "done", "done")
        params = dict(label="Test", count=3, status_index=0, board_index=1, log_index=2)
        disconnected = present_job(first, [], **params)
        next(disconnected)
        disconnected.close()
        updates = list(present_job(second, [], **params))
        self.assertEqual(updates[-1][0], "second done")

    def test_generator_backend_outputs_survive_and_activity_moves(self):
        result = ({"job": "unchanged"}, "done", "complete", "log")

        def backend(progress):
            progress(.05, desc="A long phase without a percentage")
            time.sleep(.65)
            yield result

        updates = list(present_job(backend, [], label="Test", count=4,
                                  status_index=1, board_index=2, log_index=3))
        self.assertIn("indeterminate", updates[0][2])
        self.assertIn("elapsed", updates[0][2])
        self.assertEqual(updates[-1], result)

    def test_actual_counts_and_failure_clear_busy_state(self):
        def run_sam2(progress):
            progress(.5, desc="SAM 2.1 forward: 3/6")
            time.sleep(.65)
            raise ValueError("test failure")

        updates = list(present_job(run_sam2, [], label="SAM", count=3,
                                  status_index=0, board_index=1, log_index=2))
        self.assertIn('aria-valuenow="50"', updates[0][1])
        self.assertIn("Stopped", updates[-1][1])
        self.assertNotIn("stage-meter", updates[-1][1])

    def test_extraction_does_not_reuse_previous_run_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "pipeline.log"
            log.write_text("120/120 old fg>5: 20\n")

            def extract_subject_splat(state, progress):
                progress(.15, desc="Classifying")
                time.sleep(.6)
                with log.open("a") as stream:
                    stream.write("1/4 new fg>5: 2\n")
                time.sleep(.6)
                return ("done", "complete", "log")

            updates = list(present_job(extract_subject_splat, [{"job": folder}],
                           label="Extract", count=3, status_index=0, board_index=1, log_index=2))
            self.assertIn("indeterminate", updates[0][1])
            self.assertIn('aria-valuenow="25"', updates[1][1])
            self.assertNotIn("old", updates[1][2])


class RecoveryTests(unittest.TestCase):
    def test_budget_threshold_partition_and_original_attributes(self):
        dtype = [("x", "f4"), ("opacity", "f4")] + [(f"scale_{i}", "f4") for i in range(3)]
        data = np.zeros(6, dtype=dtype)
        data["x"] = np.arange(6)
        data["opacity"] = np.arange(6) + .1
        scored = dict(data=data, keep=np.array([True, True, False, False, False, False]),
                      indices=np.array([2, 3, 4, 5]), score=np.array([.9, .95, .99, .6]),
                      eligible=np.array([True, True, False, True]), scales=np.ones((4, 3))*.02,
                      scene_sha256="test")
        for name in ("mask_support", "proximity", "color_support", "fg", "fraction", "observed"):
            scored[name] = np.ones(4)
        with tempfile.TemporaryDirectory() as folder:
            job = Path(folder)
            (job / "job.json").write_text(json.dumps({"source": "test.mov"}))
            result = export_recovery(job, .8, 1, scored)
            exported = PlyData.read(result["final"], mmap=False)["vertex"].data
            self.assertEqual(result["recovered_count"], 1)
            self.assertEqual(exported[:2].tobytes(), data[:2].tobytes())
            self.assertEqual(exported[2]["x"], 3)
            self.assertAlmostEqual(float(exported[2]["scale_0"]), float(np.log(.02)), places=6)
            zero = export_recovery(job, .5, 0, scored)
            self.assertEqual(PlyData.read(zero["final"], mmap=False)["vertex"].data.tobytes(), data[:2].tobytes())
            self.assertEqual(zero["recovered_count"], 0)
            strict = export_recovery(job, 1, 100, scored)
            self.assertEqual(strict["recovered_count"], 0)
            with self.assertRaises(ValueError):
                export_recovery(job, -1, 1, scored)


if __name__ == "__main__":
    unittest.main()
