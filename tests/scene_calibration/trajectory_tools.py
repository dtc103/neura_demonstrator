"""Shared offline validation and smooth joint-path generation; no ROS imports."""

import hashlib
import copy
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def scaled_plan(plan, scale, limits=None):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Speed scale must be finite and greater than zero")
    result = copy.deepcopy(plan)
    motion = result["motion"]
    motion["min_segment_seconds"] /= scale
    motion["max_velocity_rad_s"] *= scale
    motion["max_acceleration_rad_s2"] *= scale**2
    motion["max_jerk_rad_s3"] *= scale**3
    if any(not math.isfinite(value) or value <= 0 for value in motion.values()):
        raise ValueError("Scaled motion settings must be finite and positive")
    if limits is not None and motion["max_velocity_rad_s"] > min(bound["velocity"] for bound in limits.values()):
        raise ValueError("Scaled velocity exceeds a joint's URDF velocity limit; reduce --speed-scale")
    return result


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def vector(values, size=7):
    if not isinstance(values, list) or len(values) != size or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError(f"Expected {size} finite numeric joint values")
    return values


def latest_state(path):
    try:
        history = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Cannot read robot-state bridge file") from error
    if history.get("host") != os.uname().nodename or not history.get("samples"):
        raise ValueError("Robot-state bridge must be running on this PC")
    state = history["samples"][-1]
    age = (time.monotonic_ns() - state["received_monotonic_ns"]) / 1e9
    if not 0 <= age <= 0.3:
        raise ValueError("Robot state is stale")
    vector(state["q"])
    vector(state["dq"])
    if state.get("current_error_flags") or state.get("robot_mode") in (4, 5, 6):
        raise ValueError("Robot is stopped, in reflex, recovering, or reports an error")
    return state


def joint_limits(urdf_path):
    limits = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        if joint.attrib["name"] in JOINTS:
            entry = joint.find("limit")
            limits[joint.attrib["name"]] = {key: float(entry.attrib[key]) for key in ("lower", "upper", "velocity")}
    if set(limits) != set(JOINTS):
        raise ValueError("URDF must contain limits for all seven Panda joints")
    return limits


def check_position(positions, limits):
    vector(positions)
    for name, position in zip(JOINTS, positions):
        bound = limits[name]
        if not bound["lower"] + 0.02 <= position <= bound["upper"] - 0.02:
            raise ValueError(f"{name} waypoint is outside the joint limits or within 0.02 rad of a limit")


def new_plan(urdf_path):
    return {"schema_version": 1, "robot": "Franka Emika Panda", "joint_names": JOINTS,
            "urdf_sha256": hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),
            "motion": {"min_segment_seconds": 6.0, "max_velocity_rad_s": 0.15,
                       "max_acceleration_rad_s2": 0.3, "max_jerk_rad_s3": 1.0},
            "settle_seconds": 2.0, "goal_tolerance_rad": 0.005,
            "note": "Joint-space transitions are not collision-checked. Review every transition, including transit points.",
            "waypoints": []}


def validate_plan(plan, urdf_path, require_capture=True):
    if plan.get("schema_version") != 1 or plan.get("joint_names") != JOINTS:
        raise ValueError("Unsupported plan schema or joint order")
    if plan.get("urdf_sha256") != hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest():
        raise ValueError("The live robot description differs from the one used to teach this path")
    motion = plan["motion"]
    for key in ("min_segment_seconds", "max_velocity_rad_s", "max_acceleration_rad_s2", "max_jerk_rad_s3"):
        if type(motion.get(key)) not in (int, float) or not math.isfinite(motion[key]) or motion[key] <= 0:
            raise ValueError(f"Invalid motion setting: {key}")
    if motion["max_velocity_rad_s"] > 0.15 or motion["max_acceleration_rad_s2"] > 0.3 or motion["max_jerk_rad_s3"] > 1:
        raise ValueError("This calibration runner is limited to 0.15 rad/s, 0.3 rad/s², and 1 rad/s³")
    for key, low, high in (("settle_seconds", 1, 10),):
        value = plan.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
    tolerance = plan.get("goal_tolerance_rad")
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("goal_tolerance_rad must be finite and greater than zero")
    limits = joint_limits(urdf_path)
    if motion["max_velocity_rad_s"] > min(bound["velocity"] for bound in limits.values()):
        raise ValueError("Requested velocity exceeds a joint's URDF velocity limit")
    waypoints = plan.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("Teach at least one waypoint before reviewing or running the path")
    names = set()
    for waypoint in waypoints:
        name = waypoint.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Waypoint names must be nonempty and unique")
        names.add(name)
        check_position(waypoint.get("positions"), limits)
        if waypoint.get("capture") not in ("train", "validation", "none"):
            raise ValueError(f"{name}: capture must be train, validation, or none")
    if require_capture and not any(w["capture"] != "none" for w in waypoints):
        raise ValueError("Path has no capture poses")
    return limits


def segment_duration(start, end, motion):
    distance = max(abs(b - a) for a, b in zip(vector(start), vector(end)))
    # Quintic rest-to-rest interpolation: analytic derivative maxima.
    return max(motion["min_segment_seconds"], 1.875 * distance / motion["max_velocity_rad_s"],
               math.sqrt((10 / math.sqrt(3)) * distance / motion["max_acceleration_rad_s2"]),
               (60 * distance / motion["max_jerk_rad_s3"]) ** (1 / 3))


def smooth_segment(start, end, motion):
    duration = segment_duration(start, end, motion)
    steps = math.ceil(duration / 0.1)
    points = []
    for i in range(steps + 1):
        s = i / steps
        scale = 10 * s**3 - 15 * s**4 + 6 * s**5
        velocity = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
        acceleration = (60 * s - 180 * s**2 + 120 * s**3) / duration**2
        delta = [b - a for a, b in zip(start, end)]
        points.append({"seconds": duration * s, "positions": [a + scale * d for a, d in zip(start, delta)],
                       "velocities": [velocity * d for d in delta], "accelerations": [acceleration * d for d in delta]})
    return duration, points
