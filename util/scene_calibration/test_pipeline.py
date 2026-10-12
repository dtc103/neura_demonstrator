import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_pipeline
from trajectory_tools import atomic_json


class Child:
    def __init__(self, code=None):
        self.returncode = code
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, signal):
        self.signals.append(signal)
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode


def fixture(root):
    config = root / "config.json"
    path = root / "trajectory.json"
    atomic_json(config, {"session": "old", "board": "board.json", "cameras": {}, "robot": {},
                         "printed_square_size_verified": True})
    waypoints = [{"name": f"pose_{i}", "capture": "train" if i < 10 else "validation", "positions": [0.] * 7}
                 for i in range(13)]
    atomic_json(path, {"waypoints": waypoints})
    return ["run_pipeline.py", "--config", str(config), "--path", str(path),
            "--session", str(root / "new"), "--output", str(root / "results"),
            "--speed-scale", "8", "--goal-tolerance-rad", "0.03", "--skip-settle",
            "--disable-path-tolerance", "--disable-goal-time-tolerance", "--wait-seconds", "2.5"]


class PipelineTests(unittest.TestCase):
    def test_startup_accepts_all_boards_occluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "panda.urdf").write_text("<robot/>")
            with patch.object(run_pipeline, "latest_state", return_value={}), \
                 patch.object(run_pipeline, "capture_ready", return_value={
                     "ready": True, "board_detected": {"wrist": False, "side": False, "opposite": False}}):
                run_pipeline.wait_ready(root, [], 1, True)

    def test_camera_python_ignores_inherited_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cv2.py").write_text('raise RuntimeError("Imported wrong OpenCV")\n')
            probe = root / "probe.py"
            probe.write_text('import cv2\nprint(cv2.__file__)\n')
            # Put the injected module in a separate directory from the script.
            script_dir = root / "scripts"
            script_dir.mkdir()
            probe = probe.rename(script_dir / "probe.py")
            command = run_pipeline.camera_command(run_pipeline.ROOT / ".venv/bin/python", probe)
            result = subprocess.run(command, env={**os.environ, "PYTHONPATH": str(root)},
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(run_pipeline.ROOT / ".venv/"), result.stdout)

    def test_review_creates_no_folders_or_live_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(sys, "argv", fixture(root)), patch.object(run_pipeline.subprocess, "Popen") as popen, \
                 patch.object(run_pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                self.assertEqual(run_pipeline.main(), 0)
            popen.assert_not_called()
            self.assertNotIn("--execute", run.call_args.args[0])
            self.assertFalse((root / "new").exists())

    def test_capture_then_solver_and_failure_cleanup(self):
        for runner_code, solver_code in ((0, 0), (0, 1), (1, 0)):
            with self.subTest(runner=runner_code, solver=solver_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = fixture(root) + ["--execute"]
                bridge, camera, runner = Child(), Child(), Child(runner_code)

                def ready(session, processes, timeout, cameras):
                    for i in range(13):
                        sample = session / f"sample_{i:04d}" / "sample.json"
                        atomic_json(sample, {"split": "train" if i < 10 else "validation",
                                             "automation": {"waypoint": f"pose_{i}"}})

                def solve(command):
                    self.assertIsNotNone(bridge.returncode)
                    self.assertIsNotNone(camera.returncode)
                    if solver_code == 0:
                        atomic_json(root / "results" / "scene.json", {"validated": True})
                    return SimpleNamespace(returncode=solver_code)

                with patch.object(sys, "argv", argv), patch.object(run_pipeline.importlib.util, "find_spec", return_value=object()), \
                     patch.object(run_pipeline, "wait_ready", side_effect=ready), \
                     patch.object(run_pipeline.subprocess, "Popen", side_effect=[bridge, camera, runner]) as popen, \
                     patch.object(run_pipeline.subprocess, "run", side_effect=solve) as run:
                    code = run_pipeline.main()
                self.assertEqual(code, 1 if runner_code or solver_code else 0)
                self.assertTrue(bridge.signals)
                self.assertTrue(camera.signals)
                self.assertEqual(run.call_count, 0 if runner_code else 1)
                command = popen.call_args.args[0]
                self.assertIn("--execute", command)
                self.assertIn("--disable-goal-time-tolerance", command)
                self.assertNotIn("--motion-only", command)
                self.assertEqual(command[command.index("--speed-scale") + 1], "8.0")
                saved = json.loads((root / "new" / "pipeline_config.json").read_text())
                self.assertEqual(saved["session"], str(root / "new"))
                self.assertEqual(saved["board"], str(root / "board.json"))
                status = json.loads((root / "new" / "pipeline_status.json").read_text())
                self.assertEqual(status["stage"], "failed" if runner_code else "calibration_failed" if solver_code else "complete")

    def test_missing_or_duplicate_captures_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            waypoints = [{"name": "one", "capture": "train"}, {"name": "transit", "capture": "none"}]
            with self.assertRaisesRegex(ValueError, "incomplete"):
                run_pipeline.verify_captures(root, waypoints)
            for i in range(2):
                atomic_json(root / f"sample_{i}" / "sample.json", {"split": "train", "automation": {"waypoint": "one"}})
            with self.assertRaisesRegex(ValueError, "extra"):
                run_pipeline.verify_captures(root, waypoints)

    def test_startup_failure_starts_no_runner_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bridge, camera = Child(), Child()
            with patch.object(sys, "argv", fixture(root) + ["--execute"]), \
                 patch.object(run_pipeline.importlib.util, "find_spec", return_value=object()), \
                 patch.object(run_pipeline, "wait_ready", side_effect=RuntimeError("No board detected")), \
                 patch.object(run_pipeline.subprocess, "Popen", side_effect=[bridge, camera]) as popen, \
                 patch.object(run_pipeline.subprocess, "run") as run:
                self.assertEqual(run_pipeline.main(), 1)
            self.assertEqual(popen.call_count, 2)
            run.assert_not_called()
            self.assertTrue(bridge.signals)
            self.assertTrue(camera.signals)



if __name__ == "__main__":
    unittest.main()
