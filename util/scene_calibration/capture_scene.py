"""Capture stationary robot/ChArUco observations; never sends robot commands."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

import cv2
import numpy as np
import pyrealsense2 as rs

from calibration_core import ROLES, difference, stable_robot_sample, transform
from camera_geometry import draw_axes, estimate_pose, project_points, supported_model
from generate_board import make_board

ROOT = Path(__file__).parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_robot(path):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Waiting for robot_state_bridge.py to write robot states") from error
    if data.get("host") != os.uname().nodename:
        raise ValueError("Robot bridge and camera capture must run on the same PC")
    history = data.get("samples", [])
    if not history:
        raise ValueError("No robot states received")
    age = (time.monotonic_ns() - history[-1]["received_monotonic_ns"]) / 1e9
    if not 0 <= age <= 0.3:
        raise ValueError("Robot state is stale; check ROS_DOMAIN_ID=100 and the bridge")
    return data


def observe(image, detector, board, intrinsics, min_corners=8, max_rms_px=1.5):
    corners, ids, markers, marker_ids = detector.detectBoard(image)
    display = image.copy()
    if marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(display, markers, marker_ids)
    count = 0 if ids is None else len(ids)
    if count:
        corners = np.asarray(corners).reshape(-1, 1, 2)
        ids = np.asarray(ids).reshape(-1, 1)
        cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
    if count < min_corners or board.checkCharucoCornersCollinear(ids):
        return None, display, f"{count} corners; need {min_corners} spread across board"
    objects, pixels = board.matchImagePoints(corners, ids)
    pixels = pixels.reshape(-1, 2)
    # A tiny visible patch may have a low reprojection error but poor pose accuracy.
    if np.prod(np.ptp(pixels, axis=0)) < image.shape[0] * image.shape[1] * 0.01:
        return None, display, "Visible board area is too small"
    valid, rotation, translation = estimate_pose(objects, pixels, intrinsics)
    if not valid:
        return None, display, "Pose estimation failed"
    orientation = cv2.Rodrigues(rotation)[0]
    if np.min((orientation @ objects.reshape(-1, 3).T + translation) [2]) <= 0:
        return None, display, "Invalid board depth"
    projected = project_points(objects, rotation, translation, intrinsics)
    rms = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
    if rms > max_rms_px:
        return None, display, f"Reprojection error {rms:.2f}px exceeds {max_rms_px:.2f}px"
    draw_axes(display, rotation, translation, intrinsics)
    return {"T_camera_from_board": transform(orientation, translation).tolist(),
            "corner_ids": ids.ravel().tolist(), "object_points_m": objects.reshape(-1, 3).tolist(),
            "image_points_px": pixels.tolist(), "reprojection_rms_px": rms}, display, f"{count} corners; RMS {rms:.2f}px"


class Camera:
    def __init__(self, context, serial):
        self.pipeline = rs.pipeline(context)
        self.started = False
        try:
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            profile = self.pipeline.start(config)
            self.started = True
            device = profile.get_device()
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.metadata = {"model": device.get_info(rs.camera_info.name), "serial": serial, "stream": "color",
                             "intrinsics": {"width": intr.width, "height": intr.height, "fps": 30,
                                            "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                                            "distortion_model": str(intr.model),
                                            "distortion_coefficients": [0.0] * 5 if intr.model == rs.distortion.none else list(intr.coeffs)}}
            if not supported_model(self.metadata["intrinsics"]):
                raise ValueError(f"{serial}: unsupported PnP distortion model {intr.model}; rectify or calibrate this stream first")
        except Exception:
            self.stop()
            raise

    def stop(self):
        if self.started:
            self.pipeline.stop()
            self.started = False

    def image(self):
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return None
        # Drain queued frames to keep this receipt near the actual observation.
        for _ in range(4):
            newer = self.pipeline.poll_for_frames()
            if not newer:
                break
            frames = newer
        frame = frames.get_color_frame()
        if not frame:
            return None
        received = time.monotonic_ns()
        return np.asanyarray(frame.get_data()).copy(), {
            "received_monotonic_ns": received, "device_timestamp_ms": frame.get_timestamp(),
            "device_timestamp_domain": str(frame.get_frame_timestamp_domain()), "frame_number": frame.get_frame_number()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scene_config.json")
    parser.add_argument("--square-mm", type=float, help="Actual measured printed square size; confirms metric scale")
    parser.add_argument("--list", action="store_true", help="List serial numbers without opening streams")
    parser.add_argument("--accept-requests", action="store_true", help="Accept automatic captures from run_trajectory.py")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    home = args.config.resolve().parent
    board_config = json.loads((home / config["board"]).read_text())
    if args.square_mm is not None:
        if not np.isfinite(args.square_mm) or args.square_mm <= 0:
            parser.error("--square-mm must be a positive finite measured length")
        ratio = args.square_mm / (board_config["square_length_m"] * 1000)
        board_config["square_length_m"] *= ratio
        board_config["marker_length_m"] *= ratio
    elif not config.get("printed_square_size_verified") and not args.list:
        parser.error("Measure a printed square in both directions, then pass --square-mm ACTUAL_SIZE. Do not assume printer scaling is correct.")
    if set(config["cameras"]) != set(ROLES) or len(set(config["cameras"].values())) != 3:
        parser.error("Configure three distinct serial numbers for wrist, side, and opposite")
    try:
        context = rs.context()
        devices = list(context.query_devices())
        detected = {d.get_info(rs.camera_info.serial_number): d.get_info(rs.camera_info.name) for d in devices}
        for serial, name in detected.items():
            print(f"{serial}: {name}", flush=True)
        if args.list:
            return
        missing = set(config["cameras"].values()) - set(detected)
        if missing:
            raise ValueError(f"Configured cameras not connected: {sorted(missing)}")
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    session = home / config["session"]
    session.mkdir(parents=True, exist_ok=True)
    lock = (session / ".capture.lock").open("w")
    import fcntl
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("Another capture process is already using this session") from error
    cameras = {}
    capture_token = uuid4().hex
    processed_request = None
    last_heartbeat = 0
    try:
        for role in ROLES:
            print(f"Starting {role} camera {config['cameras'][role]}: RGB 1280x720 at 30 FPS…", flush=True)
            cameras[role] = Camera(context, config["cameras"][role])
            intr = cameras[role].metadata["intrinsics"]
            print(f"Ready: {role} camera; distortion model: {intr.get('distortion_model', 'brown_conrady')}", flush=True)
        metadata = {"schema_version": 1, "robot": config["robot"], "board": board_config,
                    "cameras": {role: camera.metadata for role, camera in cameras.items()},
                    "printed_square_size_verified": True,
                    "capture_contract": "Stationary observations; local monotonic receipts on one PC. Not hardware-synchronized motion capture."}
        metadata_path = session / "session.json"
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Existing session has different board, cameras, or stream intrinsics. Choose a new session folder.")
        write_json(metadata_path, metadata)
        board = make_board(board_config)
        detector = cv2.aruco.CharucoDetector(board)
        samples = list(session.glob("sample_*/sample.json"))
        saved_samples = [json.loads(p.read_text()) for p in samples]
        saved_poses = [s["T_base_from_flange"] for s in saved_samples]
        coverage = {split: {role: sum(s["split"] == split and role in s["observations"] for s in saved_samples)
                            for role in ROLES} for split in ("train", "validation")}
        next_id = max((int(p.parent.name.split("_")[1]) for p in samples), default=0) + 1
        print("All three streams started. Opening the preview window…", flush=True)
        print("SPACE: save training pose | V: save held-out validation pose | Q/Escape: quit", flush=True)
        print("Leave the board and fixed cameras stationary. Position the robot yourself; this tool sends no commands.")
        print(f"Writing {session.resolve()}; existing captures: {len(samples)}")
        print(f"Accepted camera observations: {coverage}", flush=True)
        cached = {}
        status = "Waiting for all three color streams and robot state bridge"
        window = "Scene calibration — SPACE=train, V=validation"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 840)
        preview_started = time.monotonic()
        while True:
            for role, camera in cameras.items():
                frame = camera.image()
                if frame is not None:
                    image, stamp = frame
                    observation, display, message = observe(image, detector, board, camera.metadata["intrinsics"])
                    cached[role] = (image, stamp, observation, display, message)
            tiles = {}
            now = time.monotonic_ns()
            if args.accept_requests and now - last_heartbeat > 250_000_000:
                write_json(session / "capture_ready.json", {
                    "capture_token": capture_token, "received_monotonic_ns": now,
                    "ready": len(cached) == 3 and all(now - e[1]["received_monotonic_ns"] <= 200_000_000 for e in cached.values()),
                    "board_detected": {role: role in cached and cached[role][2] is not None for role in ROLES}})
                last_heartbeat = now
            for role in ROLES:
                tile_width, tile_height = (1280, 720) if role == "wrist" else (640, 360)
                if role not in cached:
                    tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
                    message = "Waiting for frames"
                else:
                    _, stamp, _, display, message = cached[role]
                    tile = cv2.resize(display, (tile_width, tile_height))
                    if now - stamp["received_monotonic_ns"] > 200_000_000:
                        message = "STALE CAMERA FRAME"
                cv2.rectangle(tile, (0, 0), (tile_width, 45), (0, 0, 0), -1)
                cv2.putText(tile, f"{role}: {config['cameras'][role]}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(tile, message, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 255), 1)
                counts = f"Saved: {coverage['train'][role]} fitting / {coverage['validation'][role]} validation"
                cv2.putText(tile, counts, (8, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 0), 1)
                tiles[role] = tile
            # Preserve the 16:9 views: the full-width wrist view is twice
            # the height of either top tile, rather than stretched sideways.
            footer = np.zeros((40, 1280, 3), dtype=np.uint8)
            cv2.putText(footer, status[:180], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            grid = np.vstack((np.hstack((tiles["side"], tiles["opposite"])), tiles["wrist"], footer))
            cv2.imshow(window, grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Camera preview stopped: Q or Escape pressed.", flush=True)
                break
            # Some window backends report not-visible until the first window
            # is mapped, or -1 when visibility querying is unsupported.
            if time.monotonic() - preview_started > 1 and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) == 0:
                print("Camera preview stopped: window reports closed.", flush=True)
                break
            request = None
            if args.accept_requests and (session / "capture_request.json").exists():
                candidate = json.loads((session / "capture_request.json").read_text())
                if candidate.get("capture_token") == capture_token and candidate.get("id") != processed_request:
                    request = candidate
                    processed_request = request.get("id")
                    key = ord("v") if request.get("split") == "validation" else ord(" ")
            if key not in (ord(" "), ord("v"), ord("V")):
                continue
            destination = None
            try:
                if request is not None:
                    from trajectory_tools import vector
                    if request.get("split") not in ("train", "validation") or not isinstance(request.get("id"), str):
                        raise ValueError("Invalid automatic capture request")
                    if not 0 <= (now - request["requested_monotonic_ns"]) / 1e9 <= 10:
                        raise ValueError("Automatic capture request is stale")
                    vector(request.get("expected_positions"))
                    goal_tolerance = request.get("goal_tolerance_rad", 0.005)
                    if type(goal_tolerance) not in (int, float) or not np.isfinite(goal_tolerance) or goal_tolerance <= 0:
                        raise ValueError("Requested goal tolerance must be finite and greater than zero")
                if len(cached) != 3 or any(now - entry[1]["received_monotonic_ns"] > 200_000_000 for entry in cached.values()):
                    raise ValueError("Waiting for fresh frames from all cameras")
                if cached["wrist"][2] is None:
                    raise ValueError("Wrist camera needs a reliable board detection to save this pose")
                first_receipt = min(entry[1]["received_monotonic_ns"] for entry in cached.values())
                start_ns, end_ns = first_receipt - 500_000_000, now + 350_000_000
                before = read_robot(session / "robot_state.json")
                if not (session / "panda.urdf").exists():
                    raise ValueError("Waiting for robot_description; bridge must save panda.urdf for the scene model")
                # Wait for a post-observation state interval too: don't accept a
                # frame at the start of movement just because earlier states were still.
                while time.monotonic_ns() < end_ns:
                    time.sleep(0.02)
                robot = read_robot(session / "robot_state.json")
                if before["bridge_id"] != robot["bridge_id"]:
                    raise ValueError("Robot bridge restarted during capture; retry")
                representative, pose = stable_robot_sample(robot["samples"], start_ns, end_ns)
                if request is not None and max(abs(a - b) for a, b in zip(representative["q"], request["expected_positions"])) > goal_tolerance:
                    raise ValueError("Robot is not at the requested joint pose")
                if any((error := difference(pose, old))["translation_mm"] < 2 and error["rotation_deg"] < 2 for old in saved_poses):
                    raise ValueError("This wrist pose was already saved; move/tilt to a different pose")
                destination = session / f".sample_{next_id:04d}_{os.getpid()}.tmp"
                destination.mkdir(exist_ok=False)
                observations = {}
                images = {}
                for role, (image, stamp, observation, _, _) in cached.items():
                    image_name = f"{role}.png"
                    if not cv2.imwrite(str(destination / image_name), image):
                        raise OSError(f"Cannot save {image_name}")
                    images[role] = {**stamp, "image": image_name, "board_detected": observation is not None}
                    if observation is not None:
                        observations[role] = {**observation, **stamp, "image": image_name}
                sample = {"id": next_id, "split": "train" if key == ord(" ") else "validation",
                          "captured_utc": datetime.now(timezone.utc).isoformat(), "bridge_id": robot["bridge_id"],
                          "T_base_from_flange": pose.tolist(), "robot": representative,
                          "stability_interval_monotonic_ns": [start_ns, end_ns], "observations": observations,
                          "images": images}
                if request is not None:
                    sample["automation"] = {"request_id": request["id"], "waypoint": request.get("waypoint"),
                                            "trajectory_sha256": request.get("trajectory_sha256"),
                                            "goal_tolerance_rad": goal_tolerance,
                                            "expected_joint_positions": request["expected_positions"]}
                write_json(destination / "sample.json", sample)
                destination.rename(session / f"sample_{next_id:04d}")
                destination = None
                saved_poses.append(pose.tolist())
                for role in observations:
                    coverage[sample["split"]][role] += 1
                status = f"Saved {next_id}: {sample['split']} — board observations: {', '.join(observations)}"
                print(status)
                if request is not None:
                    write_json(session / "capture_response.json", {"id": request["id"], "capture_token": capture_token,
                               "status": "saved", "sample_id": next_id, "observations": list(observations)})
                print(f"Accepted camera observations: {coverage}", flush=True)
                if sample["split"] == "validation":
                    missing = [r for r in ROLES if r not in observations]
                    if missing:
                        print(f"Validation saved without usable board detection for: {', '.join(missing)}. "
                              "Those cameras still need separate accepted validation observations.", flush=True)
                next_id += 1
            except (ValueError, OSError) as error:
                if destination is not None and destination.exists():
                    shutil.rmtree(destination)
                status = f"Not saved: {error}"
                print(status)
                if request is not None:
                    write_json(session / "capture_response.json", {"id": request["id"], "capture_token": capture_token,
                               "status": "failed", "message": str(error)})
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    finally:
        if args.accept_requests:
            write_json(session / "capture_ready.json", {"capture_token": capture_token,
                       "received_monotonic_ns": time.monotonic_ns(), "ready": False})
        for camera in cameras.values():
            camera.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Preserve the original startup error if GUI support is unavailable.
            pass
        lock.close()


if __name__ == "__main__":
    main()
