"""Capture a taught trajectory, fit calibration, validate it, and export the scene."""

from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from run_trajectory import ROOT, build_parser, capture_ready
from trajectory_tools import atomic_json, latest_state


def camera_command(python, script, *arguments):
    # Pixi/ROS exports PYTHONPATH. Without -E it overrides the uv environment
    # and can import Pixi's GUI-less OpenCV. -s also excludes user site packages.
    # Unlike -I, these flags preserve local imports next to our scripts.
    return [str(python), "-E", "-s", "-u", str(script), *map(str, arguments)]


def motion_arguments(args, config, path, execute=False):
    result = ["--config", str(config), "--path", str(path), "--speed-scale", str(args.speed_scale),
              "--wait-seconds", str(args.wait_seconds), "--action", args.action,
              "--retake-pause", str(args.retake_pause), "--capture-retries", str(args.capture_retries)]
    for name in ("motion_only", "disable_path_tolerance", "skip_settle", "disable_goal_time_tolerance"):
        if getattr(args, name):
            result.append("--" + name.replace("_", "-"))
    for name in ("max_waypoints", "goal_tolerance_rad", "goal_time_tolerance_seconds"):
        if getattr(args, name) is not None:
            result.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if execute:
        result.append("--execute")
    return result


def stop_process(process, label):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(f"{label} did not exit after interruption; terminating process.", flush=True)
        if label == "Trajectory runner":
            print("Robot cancellation was not confirmed. Use the physical stop before proceeding.", flush=True)
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def ensure_alive(processes):
    for label, process in processes:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"{label} exited unexpectedly (code {code}); inspect its log in the capture folder")


def wait_ready(session, processes, timeout, cameras):
    deadline = time.monotonic() + timeout
    reason = "waiting for robot state"
    last_reason = None
    while time.monotonic() < deadline:
        ensure_alive(processes)
        try:
            latest_state(session / "robot_state.json")
            if not (session / "panda.urdf").exists():
                raise ValueError("waiting for robot_description")
            if cameras:
                capture_ready(session)
            return
        except (OSError, ValueError, KeyError) as error:
            reason = str(error)
        if reason != last_reason:
            print(f"Startup: {reason}", flush=True)
            last_reason = reason
        time.sleep(0.1)
    raise RuntimeError(f"Startup timed out: {reason}")


def verify_captures(session, waypoints):
    expected = Counter(w["name"] for w in waypoints if w["capture"] != "none")
    actual = Counter()
    splits = {w["name"]: w["capture"] for w in waypoints}
    for path in session.glob("sample_*/sample.json"):
        sample = json.loads(path.read_text())
        name = sample.get("automation", {}).get("waypoint")
        if name not in splits or sample["split"] != splits[name]:
            raise ValueError(f"Unexpected capture or split: {path}")
        actual[name] += 1
    if actual != expected:
        raise ValueError(f"Capture sequence incomplete: missing {dict(expected - actual)}, extra {dict(actual - expected)}")


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--session", type=Path, help="New capture folder (default: captures/pipeline_TIMESTAMP)")
    parser.add_argument("--output", type=Path, help="Calibration results folder (default: results/pipeline_TIMESTAMP)")
    parser.add_argument("--square-mm", type=float, help="Measured printed square size; current large board uses 35 mm")
    parser.add_argument("--startup-timeout-seconds", type=float, default=120,
                        help="Seconds to wait for fresh robot state and camera/board readiness")
    args = parser.parse_args()
    if not math.isfinite(args.startup_timeout_seconds) or args.startup_timeout_seconds <= 0:
        parser.error("--startup-timeout-seconds must be finite and positive")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    trajectory_path = args.path.resolve()
    plan = json.loads(trajectory_path.read_text())
    waypoints = plan["waypoints"][:args.max_waypoints] if args.max_waypoints is not None else plan["waypoints"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session = (args.session or ROOT / "captures" / f"pipeline_{stamp}").resolve()
    output = (args.output or ROOT / "results" / f"pipeline_{stamp}").resolve()
    camera_python = ROOT / ".venv/bin/python"
    print(f"Captures: {session}\nCalibration results: {output}", flush=True)
    if not args.execute:
        print("REVIEW ONLY: no cameras, robot commands, or output folders will be created.", flush=True)
        code = subprocess.run([sys.executable, str(ROOT / "run_trajectory.py"),
                               *motion_arguments(args, config_path, trajectory_path)]).returncode
        print("With --execute: state bridge -> cameras -> replay/capture -> calibration fitting -> held-out evaluation -> scene export.")
        return code
    if importlib.util.find_spec("rclpy") is None:
        parser.error("Run the pipeline with the Panda Pixi Python, as used for run_trajectory.py")
    if not args.motion_only:
        if not camera_python.is_file():
            parser.error("Camera uv environment is missing: util/scene_calibration/.venv")
        if args.square_mm is None and not config.get("printed_square_size_verified"):
            parser.error("Pass --square-mm with your measured printed square size")
        if args.square_mm is not None and (not math.isfinite(args.square_mm) or args.square_mm <= 0):
            parser.error("--square-mm must be finite and positive")
        counts = Counter(w["capture"] for w in waypoints)
        if counts["train"] < 10 or counts["validation"] < 3:
            parser.error("Full calibration requires at least 10 fitting and 3 validation waypoints; increase --max-waypoints or teach more poses")
    if session.exists():
        parser.error("--session must be a new folder, to keep each calibration dataset separate")
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be new or empty; previous results will be preserved")
    if session == output:
        parser.error("--session and --output must be different folders")

    bridge = camera = runner = None
    logs = []
    status = {"created_utc": stamp, "session": str(session), "output": str(output),
              "arguments": vars(args).copy(), "stage": "starting"}
    status["arguments"] = {key: str(value) if isinstance(value, Path) else value
                            for key, value in status["arguments"].items()}
    session.mkdir(parents=True, exist_ok=False)
    runtime_config = {**config, "session": str(session),
                      "board": str((config_path.parent / config["board"]).resolve())}
    atomic_json(session / "pipeline_config.json", runtime_config)
    atomic_json(session / "trajectory.json", plan)
    atomic_json(session / "pipeline_status.json", status)
    try:
        bridge_log = (session / "bridge.log").open("w")
        logs.append(bridge_log)
        bridge = subprocess.Popen([sys.executable, "-u", str(ROOT / "robot_state_bridge.py"),
                                   "--output", str(session / "robot_state.json")], stdout=bridge_log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        processes = [("Robot state bridge", bridge)]
        if not args.motion_only:
            camera_log = (session / "camera.log").open("w")
            logs.append(camera_log)
            command = camera_command(camera_python, ROOT / "capture_scene.py",
                                     "--config", session / "pipeline_config.json", "--accept-requests")
            if args.square_mm is not None:
                command.extend(["--square-mm", str(args.square_mm)])
            camera = subprocess.Popen(command, stdout=camera_log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(("Camera recorder", camera))
        print("Waiting for fresh robot state" + (" and all camera streams (board detection is checked at capture poses)…" if camera else "…"), flush=True)
        print(f"Startup logs: {session / 'bridge.log'}" + (f", {session / 'camera.log'}" if camera else ""), flush=True)
        wait_ready(session, processes, args.startup_timeout_seconds, camera is not None)
        status["stage"] = "replaying"
        command = [sys.executable, "-u", str(ROOT / "run_trajectory.py"),
                   *motion_arguments(args, session / "pipeline_config.json", session / "trajectory.json", execute=True)]
        status["runner_command"] = shlex.join(command)
        atomic_json(session / "pipeline_status.json", status)
        runner = subprocess.Popen(command, start_new_session=True)
        while runner.poll() is None:
            ensure_alive(processes)
            time.sleep(0.1)
        if runner.returncode != 0:
            raise RuntimeError("Trajectory/capture sequence failed; partial data preserved. Calibration solver was not run.")
        ensure_alive(processes)
        if args.motion_only:
            status["stage"] = "motion_test_complete"
            print("Motion-only run complete; no calibration was fitted.", flush=True)
            return 0
        verify_captures(session, waypoints)
        stop_process(camera, "Camera recorder")
        stop_process(bridge, "Robot state bridge")
        status["stage"] = "solving_and_validating"
        atomic_json(session / "pipeline_status.json", status)
        print("Captures complete. Fitting calibration and evaluating held-out poses…", flush=True)
        code = subprocess.run(camera_command(camera_python, ROOT / "solve_scene.py",
                                              "--session", session, "--output", output)).returncode
        status["solver_exit_code"] = code
        if code != 0:
            status["stage"] = "calibration_failed"
            print(f"Calibration failed; captures and diagnostics preserved. Inspect {output / 'calibration_report.json'} if available.", flush=True)
            return code
        if not (output / "scene.json").is_file():
            raise RuntimeError("Solver completed without exporting scene.json")
        status["stage"] = "complete"
        print(f"Pipeline complete. Scene calibration: {output / 'scene.json'}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        status["stage"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        status["error"] = str(error) or "Interrupted by user"
        print(f"Pipeline stopped: {status['error']}", flush=True)
        return 1
    finally:
        # Keep state/camera processes alive until runner cancellation has completed.
        stop_process(runner, "Trajectory runner")
        stop_process(camera, "Camera recorder")
        stop_process(bridge, "Robot state bridge")
        for stream in logs:
            stream.close()
        atomic_json(session / "pipeline_status.json", status)


if __name__ == "__main__":
    raise SystemExit(main())
