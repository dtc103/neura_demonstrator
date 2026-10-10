import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from calibration_core import (average_transform, check_transform, difference, franka_flange_pose,
                              inverse, solve_dataset, stable_robot_sample, transform)
from capture_scene import observe
import capture_scene
from generate_board import definition, make_board
from solve_scene import export_scene


def pose(rvec, position):
    return transform(cv2.Rodrigues(np.array(rvec, dtype=float))[0], position)


def synthetic_dataset():
    rng = np.random.default_rng(17)
    flange_wrist = pose([0.12, -0.08, 0.05], [0.03, -0.02, 0.08])
    base_board = pose([0.05, 0.1, -0.03], [0.15, 0.1, 1.2])
    base_cameras = {"side": pose([0.04, 0.2, -0.1], [-0.2, 0, 0.4]),
                    "opposite": pose([-0.08, -0.2, 0.1], [0.3, 0, 0.5])}
    intrinsic = {"width": 1280, "height": 720, "fx": 900, "fy": 900, "cx": 640, "cy": 360,
                 "distortion_coefficients": [0, 0, 0, 0, 0]}
    metadata = {"robot": {"model": "Franka Emika Panda", "base_frame": "panda_link0", "flange_frame": "panda_link8"},
                "cameras": {role: {"serial": role, "intrinsics": copy.deepcopy(intrinsic)}
                            for role in ("wrist", "side", "opposite")}, "board": definition()}
    objects = make_board(definition()).getChessboardCorners()
    matrix = np.array([[900., 0, 640], [0, 900, 360], [0, 0, 1]])
    samples = []
    for i in range(24):
        base_flange = pose(rng.uniform(-0.65, 0.65, 3), rng.uniform(-0.1, 0.1, 3) + [0, 0, 0.3])
        cameras = {"wrist": base_flange @ flange_wrist, **base_cameras}
        observations = {}
        for role, base_camera in cameras.items():
            camera_board = inverse(base_camera) @ base_board
            pixels, _ = cv2.projectPoints(objects, cv2.Rodrigues(camera_board[:3, :3])[0], camera_board[:3, 3], matrix, None)
            observations[role] = {"T_camera_from_board": camera_board.tolist(), "object_points_m": objects.tolist(),
                                  "image_points_px": pixels.reshape(-1, 2).tolist()}
        samples.append({"id": i + 1, "split": "train" if i < 20 else "validation",
                        "T_base_from_flange": base_flange.tolist(), "observations": observations,
                        "robot": {"q": [0.] * 7}})
    return metadata, samples, flange_wrist, base_board, base_cameras


class CalibrationTests(unittest.TestCase):
    def test_recovers_known_hand_eye_and_scene_with_held_out_validation(self):
        metadata, samples, expected_x, expected_board, expected_cameras = synthetic_dataset()
        x, board, cameras, report = solve_dataset(metadata, samples)
        self.assertTrue(report["passed"])
        np.testing.assert_allclose(x, expected_x, atol=1e-8)
        np.testing.assert_allclose(board, expected_board, atol=1e-8)
        for role in cameras:
            np.testing.assert_allclose(cameras[role], expected_cameras[role], atol=1e-8)
        scene = export_scene(metadata, samples, x, board, cameras, report)
        self.assertEqual(scene["cameras"]["wrist"]["parent"], "panda_link8")
        np.testing.assert_allclose(scene["side_from_opposite"], inverse(expected_cameras["side"]) @ expected_cameras["opposite"], atol=1e-8)
        self.assertEqual(scene["world_frame"], "panda_link0")
        self.assertIn("table boundaries and thickness", scene["unmeasured_geometry"])

    def test_bad_held_out_observation_fails_validation(self):
        metadata, samples, *_ = synthetic_dataset()
        samples[-1]["observations"]["wrist"]["T_camera_from_board"][0][3] += 0.04
        *_, report = solve_dataset(metadata, samples)
        self.assertFalse(report["passed"])

    def test_translation_only_motion_is_rejected(self):
        metadata, samples, *_ = synthetic_dataset()
        for sample in samples:
            matrix = np.array(sample["T_base_from_flange"])
            matrix[:3, :3] = np.eye(3)
            sample["T_base_from_flange"] = matrix.tolist()
        with self.assertRaisesRegex(ValueError, "rotation diversity"):
            solve_dataset(metadata, samples)

    def test_requires_validation_for_each_fixed_camera(self):
        metadata, samples, *_ = synthetic_dataset()
        for sample in samples:
            if sample["split"] == "validation":
                del sample["observations"]["side"]
        with self.assertRaisesRegex(ValueError, "held-out observation for side"):
            solve_dataset(metadata, samples)

    def test_column_major_franka_pose_removes_tcp_offset(self):
        base_flange = pose([0.2, 0.3, -0.1], [0.4, -0.2, 0.6])
        flange_ee = pose([0, 0, -0.7], [0, 0, 0.1034])
        sample = {"o_t_ee": (base_flange @ flange_ee).ravel(order="F").tolist(),
                  "f_t_ee": flange_ee.ravel(order="F").tolist()}
        np.testing.assert_allclose(franka_flange_pose(sample), base_flange, atol=1e-10)

    def test_stationary_interval_and_movement_detection(self):
        pose_matrix = pose([0.2, 0, 0], [0.3, 0.2, 0.4])
        history = [{"received_monotonic_ns": 1_000_000_000 + i * 30_000_000,
                    "o_t_ee": pose_matrix.ravel(order="F").tolist(), "f_t_ee": np.eye(4).ravel(order="F").tolist(),
                    "dq": [0.] * 7, "robot_mode": 1} for i in range(36)]
        _, recovered = stable_robot_sample(history, 1_100_000_000, 2_000_000_000)
        np.testing.assert_allclose(recovered, pose_matrix, atol=1e-10)
        history[20]["o_t_ee"][12] += 0.02
        with self.assertRaisesRegex(ValueError, "Arm moved"):
            stable_robot_sample(history, 1_100_000_000, 2_000_000_000)
        with self.assertRaises(ValueError):
            stable_robot_sample(history[:2], 1_100_000_000, 2_000_000_000)

    def test_reflections_and_nan_transforms_are_rejected(self):
        reflected = np.eye(4)
        reflected[0, 0] = -1
        with self.assertRaises(ValueError):
            check_transform(reflected)
        with self.assertRaises(ValueError):
            check_transform(np.full((4, 4), np.nan))

    def test_printed_board_detects_in_capture_quality_gate(self):
        board = make_board(definition())
        image = np.pad(board.generateImage((1000, 1400), marginSize=0), 80, constant_values=255)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        intr = {"fx": 1200., "fy": 1200., "cx": image.shape[1] / 2, "cy": image.shape[0] / 2,
                "distortion_coefficients": [0, 0, 0, 0, 0]}
        observation, _, _ = observe(image, cv2.aruco.CharucoDetector(board), board, intr)
        self.assertIsNotNone(observation)
        self.assertEqual(len(observation["corner_ids"]), 24)
        observation, _, _ = observe(np.full_like(image, 255), cv2.aruco.CharucoDetector(board), board, intr)
        self.assertIsNone(observation)


class WorkflowTests(unittest.TestCase):
    def make_session(self, directory, bad=False):
        metadata, samples, *_ = synthetic_dataset()
        metadata["printed_square_size_verified"] = True
        directory.mkdir()
        (directory / "session.json").write_text(json.dumps(metadata))
        (directory / "panda.urdf").write_text('<robot name="panda"><link name="panda_link0"/><link name="panda_link8"/></robot>')
        if bad:
            samples[-1]["observations"]["wrist"]["T_camera_from_board"][0][3] += 0.04
        for sample in samples:
            folder = directory / f"sample_{sample['id']:04d}"
            folder.mkdir()
            (folder / "sample.json").write_text(json.dumps(sample))

    def test_cli_exports_files_and_preserves_accepted_result(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            self.make_session(root / "session")
            command = [sys.executable, str(Path(__file__).parent / "solve_scene.py"),
                       "--session", str(root / "session"), "--output", str(root / "results")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            scene_path = root / "results/scene.json"
            before = scene_path.read_bytes()
            scene = json.loads(before)
            self.assertTrue(scene["calibration"]["passed"])
            self.assertTrue((root / "results/panda.urdf").exists())
            report = json.loads((root / "results/calibration_report.json").read_text())
            self.assertEqual(len(report["input_sha256"]), 26)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(scene_path.read_bytes(), before)

    def test_failed_cli_validation_writes_report_without_scene(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            self.make_session(root / "session", bad=True)
            result = subprocess.run([sys.executable, str(Path(__file__).parent / "solve_scene.py"),
                                     "--session", str(root / "session"), "--output", str(root / "results")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "results/scene.json").exists())
            report = json.loads((root / "results/calibration_report.json").read_text())
            self.assertFalse(report["passed"])

    def test_capture_loop_saves_three_labeled_camera_observations_without_ros_or_gui(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            metadata, samples, *_ = synthetic_dataset()
            config = {"cameras": {role: role for role in ("wrist", "side", "opposite")}, "robot": metadata["robot"],
                      "board": "board.json", "printed_square_size_verified": True, "session": "session"}
            (root / "config.json").write_text(json.dumps(config))
            (root / "board.json").write_text(json.dumps(definition()))
            (root / "session").mkdir()
            (root / "session/panda.urdf").write_text("<robot/>")

            class FakeCamera:
                def __init__(self, context, serial):
                    self.metadata = metadata["cameras"][serial]

                def image(self):
                    return np.full((80, 80, 3), 255, dtype=np.uint8), {"received_monotonic_ns": time.monotonic_ns()}

                def stop(self):
                    pass

            devices = []
            for serial in ("wrist", "side", "opposite"):
                device = Mock()
                device.get_info.side_effect = lambda field, value=serial: value
                devices.append(device)
            context = Mock()
            context.query_devices.return_value = devices
            image = np.full((80, 80, 3), 255, dtype=np.uint8)
            robot = {"bridge_id": "test", "samples": []}
            with patch.object(sys, "argv", ["capture_scene.py", "--config", str(root / "config.json")]), \
                 patch.object(capture_scene.rs, "context", return_value=context), \
                 patch.object(capture_scene, "Camera", FakeCamera), \
                 patch.object(capture_scene, "observe", return_value=(samples[0]["observations"]["wrist"], image, "OK")), \
                 patch.object(capture_scene, "read_robot", return_value=robot), \
                 patch.object(capture_scene, "stable_robot_sample", return_value=(samples[0]["robot"], np.eye(4))), \
                 patch.object(cv2, "namedWindow"), patch.object(cv2, "imshow"), \
                 patch.object(cv2, "destroyAllWindows"), patch.object(cv2, "waitKey", side_effect=[32, ord("q")]), \
                 patch.object(cv2, "getWindowProperty", return_value=1):
                capture_scene.main()
            saved = json.loads((root / "session/sample_0001/sample.json").read_text())
            self.assertEqual(saved["split"], "train")
            self.assertEqual(set(saved["observations"]), {"wrist", "side", "opposite"})
            self.assertTrue((root / "session/sample_0001/wrist.png").exists())
            self.assertFalse(list((root / "session").glob(".sample_*.tmp")))


if __name__ == "__main__":
    unittest.main()
