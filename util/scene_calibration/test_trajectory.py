import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import capture_scene
import run_trajectory
import teach_waypoints
from run_trajectory import Runner, capture_ready
from test_calibration import synthetic_dataset
from trajectory_tools import (atomic_json, check_position, joint_limits, latest_state,
                              new_plan, scaled_plan, smooth_segment, validate_plan)


def make_path(root):
    urdf = root / "panda.urdf"
    urdf.write_text('<robot name="panda">' + ''.join(
        f'<joint name="panda_joint{i}" type="revolute"><limit lower="-3.0" upper="3.0" velocity="2.0"/></joint>'
        for i in range(1, 8)) + '</robot>')
    plan = new_plan(urdf)
    plan["waypoints"] = [{"name": "front", "positions": [0.] * 7, "capture": "train"},
                         {"name": "tilt", "positions": [0.15] + [0.] * 6, "capture": "validation"}]
    return urdf, plan


class TrajectoryTests(unittest.TestCase):
    def test_capture_retries_three_times_with_monitored_pause(self):
        runner = Runner.__new__(Runner)
        runner.plan = {"capture_retries": 3, "retake_pause": 2.0}
        runner.capture_once = Mock(side_effect=[run_trajectory.CaptureFailure("No board")] * 3 + [None])
        runner.check_capture_state = Mock()
        waypoint = {"name": "one"}
        with patch.object(run_trajectory.time, "monotonic", side_effect=[0, 0, 1, 2] * 3):
            runner.capture(waypoint)
        self.assertEqual(runner.capture_once.call_count, 4)
        self.assertEqual(runner.check_capture_state.call_count, 6)
        runner.plan["retake_pause"] = 0
        runner.capture_once = Mock(side_effect=run_trajectory.CaptureFailure("No board"))
        with patch('builtins.print') as output:
            runner.capture(waypoint)
        self.assertIn("Skipping capture at one after 4 attempts", output.call_args.args[0])
        self.assertEqual(runner.capture_once.call_count, 4)

    def test_capture_timeout_reuses_request_and_robot_errors_do_not_retry(self):
        runner = Runner.__new__(Runner)
        runner.plan = {"capture_retries": 3, "retake_pause": 0}
        request = {"id": "pending"}
        runner.capture_once = Mock(side_effect=[run_trajectory.CaptureFailure("timeout", request), None])
        runner.capture({"name": "one"})
        self.assertIs(runner.capture_once.call_args.args[1], request)
        runner.capture_once = Mock(side_effect=ValueError("Robot reports an error"))
        with self.assertRaisesRegex(ValueError, "Robot reports an error"):
            runner.capture({"name": "one"})
        runner.capture_once.assert_called_once()

    def test_delete_waypoint_needs_no_robot_and_preserves_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plan = make_path(root)
            atomic_json(root / "path.json", plan)
            argv = ["teach_waypoints.py", "--output", str(root / "path.json"), "--delete", "front",
                    "--config", str(root / "nonexistent.json")]
            with patch.object(sys, "argv", argv), \
                 patch.object(teach_waypoints, "latest_state", side_effect=AssertionError("No robot reads")):
                teach_waypoints.main()
            self.assertEqual(json.loads((root / "path.json").read_text())["waypoints"], plan["waypoints"][1:])
            backups = list(root.glob("path.before_delete_*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text()), plan)
            with self.assertRaises(ValueError):
                teach_waypoints.delete_waypoint(root / "path.json", "missing")
            self.assertEqual(len(list(root.glob("path.before_delete_*.json"))), 1)

    def test_insert_transit_points_preserves_existing_path_and_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plan = make_path(root)
            atomic_json(root / "config.json", {"session": "."})
            atomic_json(root / "path.json", plan)
            state = {"q": [0.2] * 7, "dq": [0.] * 7, "ros_stamp_ns": 42}
            argv = ["teach_waypoints.py", "--config", str(root / "config.json"),
                    "--output", str(root / "path.json"), "--insert-after", "front"]
            with patch.object(sys, "argv", argv), patch("builtins.input", side_effect=["m", "m", "q"]), \
                 patch.object(teach_waypoints, "latest_state", return_value=state), \
                 patch.object(teach_waypoints.time, "monotonic", side_effect=[0, 0, 2, 0, 0, 2]), \
                 patch.object(teach_waypoints.time, "sleep"):
                teach_waypoints.main()
            saved = json.loads((root / "path.json").read_text())
            self.assertEqual([w["name"] for w in saved["waypoints"]],
                             ["front", "transit_001", "transit_002", "tilt"])
            self.assertEqual(saved["waypoints"][0], plan["waypoints"][0])
            self.assertEqual(saved["waypoints"][-1], plan["waypoints"][-1])
            self.assertTrue(all(w["capture"] == "none" for w in saved["waypoints"][1:3]))
            backups = list(root.glob("path.before_insert_*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text()), plan)
            self.assertEqual(teach_waypoints.insertion_index(saved, "3"), 3)
            with self.assertRaises(ValueError):
                teach_waypoints.insertion_index(saved, "missing")

    def test_speed_scaling_doubles_duration_without_changing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan = make_path(Path(directory))
            slower = scaled_plan(plan, 0.5)
            start, end = [0.] * 7, [0.8] * 7
            original_time, _ = smooth_segment(start, end, plan["motion"])
            slower_time, points = smooth_segment(start, end, slower["motion"])
            self.assertAlmostEqual(slower_time, original_time * 2)
            np.testing.assert_allclose(points[-1]["positions"], end)
            self.assertEqual(plan["motion"]["max_velocity_rad_s"], 0.15)
            self.assertLessEqual(max(max(map(abs, p["velocities"])) for p in points), 0.075 + 1e-9)
            faster = scaled_plan(plan, 2, joint_limits(Path(directory) / "panda.urdf"))
            faster_time, faster_points = smooth_segment(start, end, faster["motion"])
            self.assertAlmostEqual(faster_time, original_time / 2)
            np.testing.assert_allclose(faster_points[-1]["positions"], end)
            self.assertAlmostEqual(faster["motion"]["max_acceleration_rad_s2"], 1.2)
            self.assertAlmostEqual(faster["motion"]["max_jerk_rad_s3"], 8)
            with self.assertRaisesRegex(ValueError, "URDF velocity"):
                scaled_plan(plan, 20, joint_limits(Path(directory) / "panda.urdf"))
            for scale in (0, -1, float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    scaled_plan(plan, scale)

    def test_goal_tolerance_has_no_upper_cap_but_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf, plan = make_path(Path(directory))
            for value in (0.0001, 0.05, 5.0):
                plan["goal_tolerance_rad"] = value
                validate_plan(plan, urdf)
            for value in (0, -1, float("inf"), float("nan")):
                plan["goal_tolerance_rad"] = value
                with self.assertRaisesRegex(ValueError, "goal_tolerance_rad"):
                    validate_plan(plan, urdf)

    def test_skip_settle_waits_fixed_time_and_still_monitors_errors(self):
        runner = Runner.__new__(Runner)
        runner.plan = {"skip_settle": True, "wait_seconds": 2.0}
        runner.tick = Mock(return_value={"q": [1.] * 7, "dq": [0.01] * 7})
        with patch.object(run_trajectory.time, "monotonic", side_effect=[0, 0, 1, 2]):
            runner.settle([0.] * 7)
        self.assertEqual(runner.tick.call_count, 2)
        runner.tick.side_effect = ValueError("Robot state is stale")
        with patch.object(run_trajectory.time, "monotonic", side_effect=[0, 0]):
            with self.assertRaisesRegex(ValueError, "stale"):
                runner.settle([0.] * 7)

    def test_feedback_records_named_joint_tracking_errors(self):
        runner = Runner.__new__(Runner)
        runner.max_tracking_errors = {}
        feedback = SimpleNamespace(joint_names=["panda_joint2"], error=SimpleNamespace(positions=[-0.051]),
                                   desired=SimpleNamespace(positions=[-0.6]), actual=SimpleNamespace(positions=[-0.549]))
        runner.receive_feedback(SimpleNamespace(feedback=feedback))
        self.assertAlmostEqual(runner.max_tracking_errors["panda_joint2"], 0.051)
        self.assertIs(runner.last_feedback, feedback)

    def test_smooth_path_respects_limits_and_stops_at_both_ends(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf, plan = make_path(root)
            validate_plan(plan, urdf)
            start, end = [0.] * 7, [1.2, -1., 0.7, -0.2, 0., 0.4, -0.6]
            duration, points = smooth_segment(start, end, plan["motion"])
            self.assertGreaterEqual(duration, 6)
            np.testing.assert_allclose(points[0]["positions"], start)
            np.testing.assert_allclose(points[-1]["positions"], end)
            np.testing.assert_allclose(points[0]["velocities"], 0)
            np.testing.assert_allclose(points[-1]["velocities"], 0)
            for point in points:
                self.assertLessEqual(max(map(abs, point["velocities"])), 0.15 + 1e-9)
                self.assertLessEqual(max(map(abs, point["accelerations"])), 0.3 + 1e-9)
                check_position(point["positions"], joint_limits(urdf))
            max_delta = max(abs(a - b) for a, b in zip(start, end))
            self.assertLessEqual(60 * max_delta / duration**3, 1.0 + 1e-9)

    def test_invalid_joint_positions_hash_and_motion_settings_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf, plan = make_path(Path(directory))
            for value in (float("nan"), 3., -3.):
                bad = copy.deepcopy(plan)
                bad["waypoints"][0]["positions"][0] = value
                with self.assertRaises(ValueError):
                    validate_plan(bad, urdf)
            bad = copy.deepcopy(plan)
            bad["motion"]["max_velocity_rad_s"] = 1
            with self.assertRaises(ValueError):
                validate_plan(bad, urdf)
            bad = copy.deepcopy(plan)
            bad["urdf_sha256"] = "other_robot"
            with self.assertRaises(ValueError):
                validate_plan(bad, urdf)

    def test_default_review_never_imports_ros_or_sends_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plan = make_path(root)
            atomic_json(root / "path.json", plan)
            atomic_json(root / "config.json", {"session": "."})
            result = subprocess.run([sys.executable, str(Path(__file__).parent / "run_trajectory.py"),
                                     "--config", str(root / "config.json"), "--path", str(root / "path.json")],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("REVIEW ONLY", result.stdout)
            self.assertFalse((root / "capture_request.json").exists())

    def test_stale_states_and_camera_restart_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {"received_monotonic_ns": time.monotonic_ns(), "q": [0.] * 7, "dq": [0.] * 7, "robot_mode": 1}
            atomic_json(root / "state.json", {"host": os.uname().nodename, "samples": [state]})
            self.assertEqual(latest_state(root / "state.json")["q"], [0.] * 7)
            state["received_monotonic_ns"] -= 10**9
            atomic_json(root / "state.json", {"host": os.uname().nodename, "samples": [state]})
            with self.assertRaisesRegex(ValueError, "stale"):
                latest_state(root / "state.json")
            atomic_json(root / "capture_ready.json", {"ready": True, "capture_token": "new", "received_monotonic_ns": time.monotonic_ns()})
            with self.assertRaisesRegex(ValueError, "restarted"):
                capture_ready(root, "old")

    def test_canceled_action_with_zero_error_code_is_not_success(self):
        runner = Runner.__new__(Runner)
        with tempfile.TemporaryDirectory() as directory:
            _, runner.plan = make_path(Path(directory))
            runner.plan["disable_path_tolerance"] = True
            runner.plan["goal_tolerance_rad"] = 0.02
            runner.plan["skip_settle"] = True
            runner.tick = Mock(return_value={"q": [0.] * 7, "dq": [0.01] * 7})
            runner.action_type = SimpleNamespace(Goal=lambda: SimpleNamespace(trajectory=SimpleNamespace(points=[])),
                                                Result=SimpleNamespace(SUCCESSFUL=0))
            handle = Mock(accepted=True)
            future = Mock()
            future.result.return_value = handle
            future.add_done_callback.side_effect = lambda callback: callback(future)
            runner.client = Mock()
            runner.client.send_goal_async.return_value = future
            runner.cancel_requested = False
            runner.wait_future = Mock(side_effect=[handle, SimpleNamespace(status=5,
                result=SimpleNamespace(error_code=0, error_string="canceled"))])
            fake_modules = {"builtin_interfaces.msg": SimpleNamespace(Duration=lambda **kw: SimpleNamespace(**kw)),
                            "control_msgs.msg": SimpleNamespace(JointTolerance=lambda **kw: SimpleNamespace(**kw)),
                            "trajectory_msgs.msg": SimpleNamespace(JointTrajectoryPoint=SimpleNamespace),
                            "action_msgs.msg": SimpleNamespace(GoalStatus=SimpleNamespace(STATUS_SUCCEEDED=4))}
            with patch.dict(sys.modules, fake_modules):
                with self.assertRaisesRegex(ValueError, "status 5"):
                    runner.move([0.1] + [0.] * 6)
            sent = runner.client.send_goal_async.call_args.args[0]
            self.assertTrue(all(t.position == -1 and t.velocity == -1 and t.acceleration == -1
                                for t in sent.path_tolerance))
            self.assertTrue(all(t.position == 0.02 for t in sent.goal_tolerance))

    def test_goal_deadline_encoding_and_client_timeout_follow_setting(self):
        for allowance in (3.0, 10.25, None):
            with self.subTest(allowance=allowance), tempfile.TemporaryDirectory() as directory:
                runner = Runner.__new__(Runner)
                _, runner.plan = make_path(Path(directory))
                runner.plan["goal_time_tolerance_seconds"] = allowance
                runner.tick = Mock(return_value={"q": [0.] * 7, "dq": [0.] * 7})
                runner.action_type = SimpleNamespace(Goal=lambda: SimpleNamespace(trajectory=SimpleNamespace(points=[])),
                                                    Result=SimpleNamespace(SUCCESSFUL=0))
                handle = Mock(accepted=True)
                future = Mock()
                future.result.return_value = handle
                future.add_done_callback.side_effect = lambda callback: callback(future)
                runner.client = Mock()
                runner.client.send_goal_async.return_value = future
                runner.cancel_requested = False
                runner.wait_future = Mock(side_effect=[handle, SimpleNamespace(status=4,
                    result=SimpleNamespace(error_code=0, error_string=""))])
                fake_modules = {"builtin_interfaces.msg": SimpleNamespace(Duration=lambda **kw: SimpleNamespace(**kw)),
                                "control_msgs.msg": SimpleNamespace(JointTolerance=lambda **kw: SimpleNamespace(**kw)),
                                "trajectory_msgs.msg": SimpleNamespace(JointTrajectoryPoint=SimpleNamespace),
                                "action_msgs.msg": SimpleNamespace(GoalStatus=SimpleNamespace(STATUS_SUCCEEDED=4))}
                with patch.dict(sys.modules, fake_modules):
                    runner.move([0.1] + [0.] * 6)
                sent = runner.client.send_goal_async.call_args.args[0]
                encoded = sent.goal_time_tolerance.sec + sent.goal_time_tolerance.nanosec / 1e9
                self.assertEqual(encoded, -1 if allowance is None else allowance)
                timeout = runner.wait_future.call_args.args[1]
                self.assertEqual(timeout, None if allowance is None else 6 + allowance + 5)

    def test_wait_without_deadline_keeps_monitoring_robot_errors(self):
        runner = Runner.__new__(Runner)
        runner.tick = Mock(side_effect=ValueError("Robot reports an error"))
        future = Mock()
        future.done.return_value = False
        with self.assertRaisesRegex(ValueError, "Robot reports an error"):
            runner.wait_future(future, None)


    def test_capture_failure_prevents_second_move_and_requests_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plan = make_path(root)
            atomic_json(root / "config.json", {"session": "."})
            atomic_json(root / "path.json", plan)
            fake_runner = Mock()
            fake_runner.tick.return_value = {"q": [0.] * 7}
            fake_runner.capture.side_effect = ValueError("missing side detection")
            argv = ["run_trajectory.py", "--execute", "--config", str(root / "config.json"), "--path", str(root / "path.json")]
            with patch.object(sys, "argv", argv), patch.object(run_trajectory, "Runner", return_value=fake_runner), \
                 patch.object(run_trajectory, "latest_state", return_value={"q": [0.] * 7}), \
                 patch.object(run_trajectory, "capture_ready", return_value={"board_detected": {r: False for r in ("wrist", "side", "opposite")}}):
                with self.assertRaisesRegex(SystemExit, "missing side detection"):
                    run_trajectory.main()
            fake_runner.move.assert_called_once()
            fake_runner.cancel.assert_called_once()
            fake_runner.close.assert_called_once()

    def test_motion_only_does_not_touch_camera_interface_or_save_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plan = make_path(root)
            # Legacy plans may retain this field; it no longer restricts startup.
            plan["start_tolerance_rad"] = 0.03
            atomic_json(root / "config.json", {"session": "."})
            atomic_json(root / "path.json", plan)
            fake_runner = Mock()
            current = [0.6] * 7
            fake_runner.tick.return_value = {"q": current}
            argv = ["run_trajectory.py", "--execute", "--motion-only", "--max-waypoints", "1",
                    "--speed-scale", "2", "--goal-tolerance-rad", "0.1", "--skip-settle", "--wait-seconds", "3",
                    "--config", str(root / "config.json"), "--path", str(root / "path.json")]
            with patch.object(sys, "argv", argv), patch.object(run_trajectory, "Runner", return_value=fake_runner) as factory, \
                 patch.object(run_trajectory, "latest_state", return_value={"q": current}), \
                 patch.object(run_trajectory, "capture_ready", side_effect=AssertionError("Camera API must not be called")):
                run_trajectory.main()
            self.assertFalse(factory.call_args.kwargs["capture_enabled"])
            runtime_plan = factory.call_args.args[1]
            self.assertTrue(runtime_plan["skip_settle"])
            self.assertEqual(runtime_plan["wait_seconds"], 3)
            self.assertEqual(runtime_plan["goal_tolerance_rad"], 0.1)
            self.assertAlmostEqual(runtime_plan["motion"]["max_velocity_rad_s"], 0.3)
            fake_runner.move.assert_called_once_with(plan["waypoints"][0]["positions"])
            fake_runner.settle.assert_called_once()
            fake_runner.capture.assert_not_called()
            self.assertFalse((root / "capture_request.json").exists())


class AutomaticCaptureTests(unittest.TestCase):
    def test_capture_request_acknowledgement_and_provenance(self):
        self.capture_scenario(None)

    def test_opposite_board_occlusion_does_not_block_capture(self):
        self.capture_scenario("opposite")

    def test_wrist_board_occlusion_blocks_capture(self):
        self.capture_scenario("wrist")

    def capture_scenario(self, missing_role):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, samples, *_ = synthetic_dataset()
            atomic_json(root / "config.json", {"cameras": {r: r for r in ("wrist", "side", "opposite")},
                                              "robot": metadata["robot"], "board": "board.json", "session": ".",
                                              "printed_square_size_verified": True})
            atomic_json(root / "board.json", metadata["board"])
            (root / "panda.urdf").write_text("<robot/>")
            atomic_json(root / "capture_request.json", {"id": "request1", "capture_token": "token1", "split": "validation",
                        "goal_tolerance_rad": 0.1,
                        "requested_monotonic_ns": time.monotonic_ns(), "expected_positions": [0.] * 7,
                        "waypoint": "pose_001", "trajectory_sha256": "planhash"})
            image = np.full((80, 80, 3), 255, dtype=np.uint8)

            class FakeCamera:
                def __init__(self, context, serial):
                    self.metadata = metadata["cameras"][serial]

                def image(self):
                    return image, {"received_monotonic_ns": time.monotonic_ns()}

                def stop(self):
                    pass

            devices = []
            for serial in ("wrist", "side", "opposite"):
                device = Mock()
                device.get_info.side_effect = lambda field, value=serial: value
                devices.append(device)
            context = Mock()
            context.query_devices.return_value = devices
            observe_count = 0

            def observe(*args):
                nonlocal observe_count
                role = ("wrist", "side", "opposite")[observe_count % 3]
                observe_count += 1
                return (None if role == missing_role else samples[0]["observations"]["wrist"], image, "OK")
            with patch.object(sys, "argv", ["capture_scene.py", "--config", str(root / "config.json"), "--accept-requests"]), \
                 patch.object(capture_scene.rs, "context", return_value=context), \
                 patch.object(capture_scene, "Camera", FakeCamera), \
                 patch.object(capture_scene, "uuid4", return_value=SimpleNamespace(hex="token1")), \
                 patch.object(capture_scene, "observe", side_effect=observe), \
                 patch.object(capture_scene, "read_robot", return_value={"bridge_id": "test", "samples": []}), \
                 patch.object(capture_scene, "stable_robot_sample", return_value=(samples[0]["robot"], np.eye(4))), \
                 patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), patch.object(cv2, "imshow"), patch.object(cv2, "destroyAllWindows"), \
                 patch.object(cv2, "waitKey", side_effect=[255, ord("q")]), patch.object(cv2, "getWindowProperty", return_value=1):
                capture_scene.main()
            response = json.loads((root / "capture_response.json").read_text())
            if missing_role == "wrist":
                self.assertEqual(response["status"], "failed")
                self.assertFalse((root / "sample_0001").exists())
                return
            self.assertEqual(response["status"], "saved")
            self.assertEqual(response["id"], "request1")
            self.assertEqual(set(response["observations"]), {"wrist", "side", "opposite"} - {missing_role})
            sample = json.loads((root / "sample_0001/sample.json").read_text())
            self.assertEqual(sample["split"], "validation")
            self.assertEqual(sample["automation"]["trajectory_sha256"], "planhash")
            self.assertEqual(set(sample["images"]), {"wrist", "side", "opposite"})
            self.assertTrue((root / "sample_0001/opposite.png").exists())
            self.assertEqual(sample["images"]["opposite"]["board_detected"], missing_role != "opposite")
            self.assertFalse(json.loads((root / "capture_ready.json").read_text())["ready"])


if __name__ == "__main__":
    unittest.main()
