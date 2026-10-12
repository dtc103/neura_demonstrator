"""Review a taught path, or explicitly execute stop-and-capture using Panda ROS 2."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from itertools import count
from uuid import uuid4

from trajectory_tools import JOINTS, atomic_json, check_position, latest_state, scaled_plan, segment_duration, smooth_segment, validate_plan

ROOT = Path(__file__).parent


class CaptureFailure(ValueError):
    def __init__(self, message, pending_request=None):
        super().__init__(message)
        self.pending_request = pending_request


def capture_ready(session, token=None):
    try:
        data = json.loads((session / "capture_ready.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Start capture_scene.py with --accept-requests before executing") from error
    if not data.get("ready") or not 0 <= (time.monotonic_ns() - data["received_monotonic_ns"]) / 1e9 <= 2:
        raise ValueError("Camera capture is not ready or its heartbeat is stale")
    if token is not None and token != data["capture_token"]:
        raise ValueError("Camera capture restarted during the run")
    return data


class Runner:
    def __init__(self, session, plan, trajectory_hash, action_name, capture_enabled=True):
        import rclpy
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        from controller_manager_msgs.srv import ListControllers
        self.rclpy = rclpy
        self.action_type = FollowJointTrajectory
        self.session, self.plan, self.trajectory_hash = session, plan, trajectory_hash
        self.capture_enabled = capture_enabled
        rclpy.init()
        self.node = rclpy.create_node("scene_calibration_trajectory_runner")
        self.client = ActionClient(self.node, FollowJointTrajectory, action_name)
        self.controllers = self.node.create_client(ListControllers, "/controller_manager/list_controllers")
        self.goal = None
        self.submission = None
        self.cancel_requested = False
        self.token = capture_ready(session)["capture_token"] if capture_enabled else None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.feedback_path = session / f"motion_feedback_{stamp}.csv"
        self.feedback_file = self.feedback_path.open("x", newline="")
        self.feedback_writer = csv.writer(self.feedback_file)
        self.feedback_writer.writerow(["received_monotonic_ns", "waypoint", "joint", "desired_rad", "actual_rad", "error_rad"])
        self.current_waypoint = ""
        self.last_feedback_write_ns = 0
        print(f"Motion feedback log: {self.feedback_path}", flush=True)

    def receive_feedback(self, message):
        feedback = message.feedback
        self.last_feedback = feedback
        received = time.monotonic_ns()
        write_sample = received - getattr(self, "last_feedback_write_ns", 0) >= 50_000_000
        for index, name in enumerate(feedback.joint_names):
            if index >= len(feedback.error.positions):
                continue
            error = abs(feedback.error.positions[index])
            self.max_tracking_errors[name] = max(error, self.max_tracking_errors.get(name, 0.0))
            writer = getattr(self, "feedback_writer", None)
            if writer is not None and write_sample:
                desired = feedback.desired.positions[index] if index < len(feedback.desired.positions) else ""
                actual = feedback.actual.positions[index] if index < len(feedback.actual.positions) else ""
                writer.writerow([received, self.current_waypoint, name, desired, actual, feedback.error.positions[index]])
        stream = getattr(self, "feedback_file", None)
        if stream is not None and write_sample:
            stream.flush()
            self.last_feedback_write_ns = received

    def print_tracking(self, target, failed=False):
        print("Joint tracking overview (radians):", flush=True)
        print("Joint             Peak path error   Last desired error   Last target error", flush=True)
        feedback = getattr(self, "last_feedback", None)
        lookup = {} if feedback is None else {name: i for i, name in enumerate(feedback.joint_names)}
        for name, position in zip(JOINTS, target):
            peak = getattr(self, "max_tracking_errors", {}).get(name)
            index = lookup.get(name)
            path_error = None if index is None or index >= len(feedback.error.positions) else feedback.error.positions[index]
            actual = None if index is None or index >= len(feedback.actual.positions) else feedback.actual.positions[index]
            values = ["n/a" if v is None else f"{v:.6f}" for v in (peak, path_error, None if actual is None else position - actual)]
            print(f"{name:<17} {values[0]:>15} {values[1]:>20} {values[2]:>19}", flush=True)
        if failed:
            path_limit = "disabled" if self.plan.get("disable_path_tolerance", False) else "0.050 rad"
            print(f"Path limit: {path_limit}; final-pose limit: {self.plan['goal_tolerance_rad']} rad.", flush=True)
            print("Feedback is sampled; the controller log contains the exact failing cycle.", flush=True)

    def tick(self):
        self.rclpy.spin_once(self.node, timeout_sec=0.02)
        if self.capture_enabled:
            capture_ready(self.session, self.token)
        return latest_state(self.session / "robot_state.json")

    def wait_future(self, future, timeout):
        start = time.monotonic()
        while not future.done():
            self.tick()
            if timeout is not None and time.monotonic() - start > timeout:
                raise ValueError("ROS action/service timed out")
        result = future.result()
        if result is None:
            raise ValueError("ROS action/service returned no result")
        return result

    def check_controllers(self):
        from controller_manager_msgs.srv import ListControllers
        if not self.controllers.wait_for_service(timeout_sec=3):
            raise ValueError("Controller-manager service unavailable")
        response = self.wait_future(self.controllers.call_async(ListControllers.Request()), 3)
        active = {c.name for c in response.controller if c.state == "active"}
        if "joint_trajectory_controller" not in active:
            raise ValueError("Activate joint_trajectory_controller before replay; leave it inactive while teaching")
        other = {c.name for c in response.controller if c.state == "active" and c.name != "joint_trajectory_controller"
                 and any(interface.endswith("/effort") or interface.endswith("/position") or interface.endswith("/velocity")
                         for interface in c.claimed_interfaces)}
        if other:
            raise ValueError(f"Other motion controllers are active: {sorted(other)}")
        if not self.client.wait_for_server(timeout_sec=3):
            raise ValueError("Joint trajectory action server unavailable")

    def move(self, positions):
        from builtin_interfaces.msg import Duration
        from action_msgs.msg import GoalStatus
        from control_msgs.msg import JointTolerance
        from trajectory_msgs.msg import JointTrajectoryPoint
        state = self.tick()
        if not self.plan.get("skip_settle", False) and max(map(abs, state["dq"])) > 0.002:
            raise ValueError("Robot is moving before a trajectory segment")
        duration, points = smooth_segment(state["q"], positions, self.plan["motion"])
        self.last_feedback = None
        self.max_tracking_errors = {}
        goal = self.action_type.Goal()
        goal.trajectory.joint_names = JOINTS
        for data in points:
            point = JointTrajectoryPoint()
            point.positions, point.velocities, point.accelerations = data["positions"], data["velocities"], data["accelerations"]
            nanoseconds = round(data["seconds"] * 1e9)
            point.time_from_start = Duration(sec=nanoseconds // 10**9, nanosec=nanoseconds % 10**9)
            goal.trajectory.points.append(point)
        goal_seconds = self.plan.get("goal_time_tolerance_seconds", 3.0)
        # A negative-one ROS tolerance erases the controller's default deadline.
        # Zero would inherit the default rather than explicitly disabling it.
        goal_ns = -10**9 if goal_seconds is None else round(goal_seconds * 10**9)
        goal.goal_time_tolerance = Duration(sec=goal_ns // 10**9, nanosec=goal_ns % 10**9)
        # ROS uses -1 to erase a tolerance; zero would retain controller defaults.
        path_limit = -1.0 if self.plan.get("disable_path_tolerance", False) else 0.05
        other_path_limit = -1.0 if self.plan.get("disable_path_tolerance", False) else 0.0
        goal.path_tolerance = [JointTolerance(name=name, position=path_limit,
                                             velocity=other_path_limit, acceleration=other_path_limit)
                               for name in JOINTS]
        goal.goal_tolerance = [JointTolerance(name=name, position=self.plan["goal_tolerance_rad"]) for name in JOINTS]
        self.submission = self.client.send_goal_async(goal, feedback_callback=self.receive_feedback)

        def submitted(future):
            handle = future.result()
            if handle is not None and handle.accepted:
                self.goal = handle
                if self.cancel_requested:
                    handle.cancel_goal_async()

        self.submission.add_done_callback(submitted)
        self.goal = self.wait_future(self.submission, 3)
        if not self.goal.accepted:
            self.goal = None
            raise ValueError("Trajectory goal was rejected")
        result_timeout = None if goal_seconds is None else duration + goal_seconds + 5
        wrapped = self.wait_future(self.goal.get_result_async(), result_timeout)
        self.goal = None
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or wrapped.result.error_code != self.action_type.Result.SUCCESSFUL:
            self.print_tracking(positions, failed=True)
            raise ValueError(f"Trajectory failed: {wrapped.result.error_string} (status {wrapped.status}, code {wrapped.result.error_code})")

    def settle(self, target):
        if self.plan.get("skip_settle", False):
            seconds = self.plan["wait_seconds"]
            print(f"Goal succeeded; waiting {seconds:g}s before continuing (settling check skipped).", flush=True)
            start = time.monotonic()
            while time.monotonic() - start < seconds:
                # Keep checking fresh robot state and errors during the pause.
                self.tick()
            return
        stable_since = None
        start = time.monotonic()
        while time.monotonic() - start < 15:
            state = self.tick()
            at_target = max(abs(a - b) for a, b in zip(state["q"], target)) <= self.plan["goal_tolerance_rad"]
            stopped = max(map(abs, state["dq"])) <= 0.002
            if at_target and stopped:
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= self.plan["settle_seconds"]:
                    return
            else:
                stable_since = None
        position_errors = [abs(a - b) for a, b in zip(state["q"], target)]
        velocities = list(map(abs, state["dq"]))
        position_joint = max(range(len(JOINTS)), key=position_errors.__getitem__)
        velocity_joint = max(range(len(JOINTS)), key=velocities.__getitem__)
        raise ValueError("Arm did not settle at the requested pose: "
                         f"largest position error {JOINTS[position_joint]}={position_errors[position_joint]:.6f} rad "
                         f"(limit {self.plan['goal_tolerance_rad']:g}); "
                         f"largest speed {JOINTS[velocity_joint]}={velocities[velocity_joint]:.6f} rad/s "
                         "(limit 0.002), required continuously for "
                         f"{self.plan['settle_seconds']:g}s")

    def capture(self, waypoint):
        pending = None
        retries = self.plan.get("capture_retries", 3)
        for attempt in range(retries + 1):
            try:
                return self.capture_once(waypoint, pending)
            except CaptureFailure as error:
                pending = error.pending_request
                if attempt == retries:
                   print (f"Capture failed after {retries + 1} attempts: Continue with next position.")
                   break
                pause = self.plan.get("retake_pause", 2.0)
                print(f"Capture attempt {attempt + 1} failed: {error}. Retrying {attempt + 1}/{retries} after {pause:g}s.", flush=True)
                start = time.monotonic()
                while time.monotonic() - start < pause:
                    self.check_capture_state(waypoint)

    def check_capture_state(self, waypoint):
        state = self.tick()
        if max(abs(a - b) for a, b in zip(state["q"], waypoint["positions"])) > self.plan["goal_tolerance_rad"] or (
                not self.plan.get("skip_settle", False) and max(map(abs, state["dq"])) > 0.002):
            raise ValueError("Arm moved while waiting for capture")

    def capture_once(self, waypoint, pending_request=None):
        request = pending_request or {
            "id": uuid4().hex,
            "capture_token": self.token, "requested_monotonic_ns": time.monotonic_ns(),
            "waypoint": waypoint["name"], "split": waypoint["capture"],
            "expected_positions": waypoint["positions"], "goal_tolerance_rad": self.plan["goal_tolerance_rad"],
            "trajectory_sha256": self.trajectory_hash}
        request_id = request["id"]
        if pending_request is None:
            atomic_json(self.session / "capture_request.json", request)
        start = time.monotonic()
        while time.monotonic() - start < 10:
            self.check_capture_state(waypoint)
            response_path = self.session / "capture_response.json"
            if response_path.exists():
                response = json.loads(response_path.read_text())
                if response.get("id") == request_id and response.get("capture_token") == self.token:
                    if response.get("status") != "saved":
                        raise CaptureFailure(f"Capture rejected: {response.get('message', 'unknown reason')}")
                    print(f"  Saved sample {response['sample_id']}: {', '.join(response['observations'])}", flush=True)
                    return
        # Keep the same ID while the outcome is unknown: a late saved response
        # must not cause a second image/pose to be saved for this waypoint.
        raise CaptureFailure("Camera capture acknowledgement timed out", pending_request=request)

    def cancel(self):
        self.cancel_requested = True
        if self.submission is not None and not self.submission.done():
            start = time.monotonic()
            while not self.submission.done() and time.monotonic() - start < 2:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.goal is not None and self.goal.accepted:
            future = self.goal.cancel_goal_async()
            start = time.monotonic()
            while not future.done() and time.monotonic() - start < 2:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
            print("Requested cancellation of the active trajectory. Use the physical stop if motion does not stop.", flush=True)
        elif self.submission is not None and not self.submission.done():
            print("Could not confirm the pending goal state. Use the physical stop; do not assume motion has stopped.", flush=True)

    def close(self):
        self.feedback_file.close()
        self.node.destroy_node()
        self.rclpy.shutdown()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scene_config_auto.json")
    parser.add_argument("--path", type=Path, default=ROOT / "trajectories/calibration.json")
    parser.add_argument("--execute", action="store_true", help="Actually send the reviewed joint trajectory to the robot")
    parser.add_argument("--motion-only", action="store_true", help="Replay and settle without camera capture or a camera heartbeat")
    parser.add_argument("--max-waypoints", type=int, help="Only review/run the first N waypoints for an initial test")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Positive timing scale: 0.5 doubles duration; 2 halves it. Scaled velocity must fit URDF limits")
    parser.add_argument("--disable-path-tolerance", action="store_true", help="Disable ROS action path tracking tolerance checks; retain final-pose and stationary checks")
    parser.add_argument("--goal-tolerance-rad", type=float, help="Positive finite final waypoint tolerance per joint in radians, with no upper cap")
    parser.add_argument("--skip-settle", action="store_true", help="After action success, use a fixed pause instead of runner joint-position/velocity settling checks")
    parser.add_argument("--wait-seconds", type=float, default=2.0, help="Pause after action success with --skip-settle (default: 2 seconds)")
    parser.add_argument("--retake-pause", type=float, default=2.0, help="Seconds between failed capture attempts (default: 2)")
    parser.add_argument("--capture-retries", type=int, default=3, help="Retries after the initial failed capture (default: 3, for four attempts total)")
    goal_time = parser.add_mutually_exclusive_group()
    goal_time.add_argument("--goal-time-tolerance-seconds", type=float,
                           help="Extra seconds allowed to reach the goal after trajectory end (default: 3)")
    goal_time.add_argument("--disable-goal-time-tolerance", action="store_true",
                           help="Wait without a goal deadline for action success; final-position tolerance remains active")
    parser.add_argument("--action", default="/joint_trajectory_controller/follow_joint_trajectory")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.max_waypoints is not None and args.max_waypoints < 1:
        parser.error("--max-waypoints must be positive")
    if not math.isfinite(args.wait_seconds) or args.wait_seconds < 0:
        parser.error("--wait-seconds must be finite and nonnegative")
    if not math.isfinite(args.retake_pause) or args.retake_pause < 0:
        parser.error("--retake-pause must be finite and nonnegative")
    if args.capture_retries < 0:
        parser.error("--capture-retries must be nonnegative")
    if args.goal_time_tolerance_seconds is not None and (
            not math.isfinite(args.goal_time_tolerance_seconds) or
            not 0 < args.goal_time_tolerance_seconds <= 2147483647 or
            round(args.goal_time_tolerance_seconds * 10**9) == 0):
        parser.error("--goal-time-tolerance-seconds must be positive, finite, and representable as a ROS duration; use --disable-goal-time-tolerance to disable")
    runner = None
    execution_lock = None
    try:
        config = json.loads(args.config.read_text())
        session = args.config.resolve().parent / config["session"]
        plan = json.loads(args.path.read_text())
        if args.goal_tolerance_rad is not None:
            plan["goal_tolerance_rad"] = args.goal_tolerance_rad
        plan["disable_path_tolerance"] = args.disable_path_tolerance
        plan["skip_settle"] = args.skip_settle
        plan["wait_seconds"] = args.wait_seconds
        plan["retake_pause"] = args.retake_pause
        plan["capture_retries"] = args.capture_retries
        plan["goal_time_tolerance_seconds"] = (None if args.disable_goal_time_tolerance else
                                               args.goal_time_tolerance_seconds if args.goal_time_tolerance_seconds is not None else 3.0)
        limits = validate_plan(plan, session / "panda.urdf")
        plan = scaled_plan(plan, args.speed_scale, limits)
        waypoints = plan["waypoints"][:args.max_waypoints] if args.max_waypoints is not None else plan["waypoints"]
        counts = Counter(w["capture"] for w in waypoints)
        print(f"Path: {args.path.resolve()}; {counts['train']} fitting, {counts['validation']} validation, {counts['none']} transit points")
        print(f"Speed scale: {args.speed_scale:g}; peak speed limit: {plan['motion']['max_velocity_rad_s']:.3f} rad/s")
        print(f"Path tolerance: {'disabled' if args.disable_path_tolerance else '0.05 rad'}; final waypoint tolerance: {plan['goal_tolerance_rad']:g} rad")
        if args.disable_goal_time_tolerance:
            print("Goal deadline: disabled; waiting for action success until interrupted or a robot error occurs.")
        else:
            print(f"Goal time allowance after trajectory end: {plan['goal_time_tolerance_seconds']:g}s")
        if args.skip_settle:
            print(f"After action success: fixed {args.wait_seconds:g}s pause; runner settling checks disabled.")
            if not args.motion_only:
                print("Camera capture still checks pose stability during image acquisition.")
        if args.motion_only:
            print("MOTION-ONLY TEST: no camera recorder required, no images or calibration captures will be saved.")
        previous = plan["waypoints"][0]["positions"]
        for i, waypoint in enumerate(waypoints):
            seconds = segment_duration(previous, waypoint["positions"], plan["motion"])
            print(f"  {i + 1:02d}. {waypoint['name']}: {waypoint['capture']}, >= {seconds:.1f}s movement; q={waypoint['positions']}")
            previous = waypoint["positions"]
        print("Transitions are smooth joint-space interpolation, NOT collision-checked paths.")
        if not args.execute:
            print("REVIEW ONLY: no ROS node started and no robot commands sent. Use --execute only after checking every transition.")
            return
        import fcntl
        lock_path = ROOT / "trajectories/.runner.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        execution_lock = lock_path.open("w")
        try:
            fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another automatic trajectory runner is already active") from error
        state = latest_state(session / "robot_state.json")
        check_position(state["q"], limits)
        first_seconds = segment_duration(state["q"], waypoints[0]["positions"], plan["motion"])
        print(f"Starting from measured current joints; first move to {waypoints[0]['name']} takes at least {first_seconds:.1f}s.", flush=True)
        if not args.motion_only:
            capture_ready(session)
        try:
            runner = Runner(session, plan, hashlib.sha256(args.path.read_bytes()).hexdigest(), args.action,
                            capture_enabled=not args.motion_only)
        except ImportError as error:
            raise ValueError("Execute with the Panda Pixi Python, not the camera uv environment") from error
        runner.check_controllers()
        for index, waypoint in enumerate(waypoints):
            print(f"Moving {index + 1}/{len(waypoints)}: {waypoint['name']}", flush=True)
            check_position(runner.tick()["q"], limits)
            runner.current_waypoint = waypoint["name"]
            runner.move(waypoint["positions"])
            runner.settle(waypoint["positions"])
            if not args.motion_only and waypoint["capture"] != "none":
                runner.capture(waypoint)
        print("Sequence complete. The trajectory controller remains active and holds the final pose.")
    except (Exception, KeyboardInterrupt) as error:
        if runner is not None:
            try:
                runner.cancel()
            except Exception as cancel_error:
                print(f"Cancellation could not be confirmed: {cancel_error}. Use the physical stop.", flush=True)
        raise SystemExit(f"Sequence stopped: {error}") from error
    finally:
        if runner is not None:
            runner.close()
        if execution_lock is not None:
            execution_lock.close()


if __name__ == "__main__":
    main()
