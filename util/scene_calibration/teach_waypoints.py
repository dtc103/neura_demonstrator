"""Teach joint waypoints from the state bridge; never sends robot commands."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

from trajectory_tools import atomic_json, check_position, joint_limits, latest_state, new_plan, validate_plan

ROOT = Path(__file__).parent


def insertion_index(plan, anchor):
    """Resolve a stable waypoint name or a displayed one-based path index."""
    for index, waypoint in enumerate(plan["waypoints"]):
        if waypoint["name"] == anchor:
            return index + 1
    if anchor.isdigit() and 1 <= int(anchor) <= len(plan["waypoints"]):
        return int(anchor)
    raise ValueError(f"Unknown insertion waypoint: {anchor}; use its name or one-based path index")


def unused_name(plan, prefix):
    names = {waypoint["name"] for waypoint in plan["waypoints"]}
    number = 1
    while f"{prefix}_{number:03d}" in names:
        number += 1
    return f"{prefix}_{number:03d}"


def delete_waypoint(path, selector):
    path = Path(path)
    plan = json.loads(path.read_text())
    index = insertion_index(plan, selector) - 1
    removed = plan["waypoints"][index]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.stem}.before_delete_{stamp}.json")
    shutil.copy2(path, backup)
    del plan["waypoints"][index]
    atomic_json(path, plan)
    print(f"Deleted {removed['name']} ({removed['capture']}); {len(plan['waypoints'])} waypoints remain.")
    print(f"Original path backed up to {backup}")
    print("Review the new transition between the remaining neighbours before replay.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scene_config_auto.json")
    parser.add_argument("--output", type=Path, default=ROOT / "trajectories/calibration.json")
    edit = parser.add_mutually_exclusive_group()
    edit.add_argument("--insert-after", help="Insert transit points after an existing waypoint name (e.g. pose_024) or one-based path index")
    edit.add_argument("--delete", help="Delete one waypoint by name or one-based path index, back up the path, and exit; no robot connection needed")
    args = parser.parse_args()
    if args.delete is not None:
        try:
            delete_waypoint(args.output, args.delete)
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit(str(error)) from error
        return
    config = json.loads(args.config.read_text())
    session = args.config.resolve().parent / config["session"]
    urdf = session / "panda.urdf"
    try:
        limits = joint_limits(urdf)
        plan = json.loads(args.output.read_text()) if args.output.exists() else new_plan(urdf)
        if plan["waypoints"]:
            validate_plan(plan, urdf, require_capture=False)
        insert_at = insertion_index(plan, args.insert_after) if args.insert_after is not None else None
        backup_created = False
        print(f"Path: {args.output.resolve()}; recorded waypoints: {len(plan['waypoints'])}")
        print("This tool only reads states. Use native hand-guiding with motion controllers inactive.")
        if insert_at is not None:
            previous = plan['waypoints'][insert_at - 1]['name']
            following = plan['waypoints'][insert_at]['name'] if insert_at < len(plan['waypoints']) else 'end of path'
            print(f"INSERT MODE: {previous} -> new transit points -> {following}; existing names are preserved.")
            print("M: insert measured clearance/transit point without capture | Q: finish")
        else:
            print("T: fitting pose | V: validation pose | M: clearance/transit point without capture | Q: finish")
        print("Replay approaches the first waypoint from the measured current pose. Include clearance points between views.")
        while True:
            choice = input("After positioning and settling, enter T/V/M/Q: ").strip().lower()
            if choice == "q":
                break
            if choice not in ("t", "v", "m"):
                continue
            if insert_at is not None and choice != "m":
                print("Insert mode accepts M for transit points or Q to finish.")
                continue
            try:
                # Require stability over an interval, not just one zero-velocity sample.
                initial = latest_state(session / "robot_state.json")
                start = time.monotonic()
                while time.monotonic() - start < 1:
                    state = latest_state(session / "robot_state.json")
                    if max(abs(a - b) for a, b in zip(state["q"], initial["q"])) > 0.002 or max(map(abs, state["dq"])) > 0.002:
                        raise ValueError("Arm is moving; release guiding controls and let it settle")
                    time.sleep(0.05)
                check_position(state["q"], limits)
                if choice != "m" and any(w["capture"] != "none" and
                        max(abs(a - b) for a, b in zip(w["positions"], state["q"])) < 0.005 for w in plan["waypoints"]):
                    raise ValueError("This joint pose is already in the path")
                waypoint = {"name": unused_name(plan, "transit" if insert_at is not None else "pose"), "positions": state["q"],
                            "capture": {"t": "train", "v": "validation", "m": "none"}[choice],
                            "taught_ros_stamp_ns": state["ros_stamp_ns"]}
                if insert_at is not None:
                    if not backup_created:
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        backup = args.output.with_name(f"{args.output.stem}.before_insert_{stamp}.json")
                        shutil.copy2(args.output, backup)
                        backup_created = True
                        print(f"Original path backed up to {backup}")
                    plan["waypoints"].insert(insert_at, waypoint)
                    insert_at += 1
                else:
                    plan["waypoints"].append(waypoint)
                atomic_json(args.output, plan)
                print(f"Saved {waypoint['name']} ({waypoint['capture']}): {waypoint['positions']}")
            except ValueError as error:
                print(f"Not saved: {error}")
        if plan["waypoints"]:
            print("Path saved. Review the joint-space transitions before executing; teaching endpoints does not check collisions between them.")
        else:
            print("No waypoints were recorded.")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
