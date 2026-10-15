"""Print Quest controller coordinates; never connects to or commands a robot."""

import argparse
from datetime import datetime, timezone
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).parent


def load_reader(source=None):
    local = Path(source) if source is not None else ROOT / "vendor/meta_quest_teleop"
    if source is not None or local.exists():
        local = local.resolve()
        if not (local / "meta_quest_teleop/reader.py").is_file():
            raise RuntimeError(f"Reader source not found in {local}")
        sys.path.insert(0, str(local))
        importlib.invalidate_caches()
        print(f"Using downloaded reader directly: {local} (no package build needed)", flush=True)
    check_apk()
    try:
        from meta_quest_teleop.reader import MetaQuestReader
    except ImportError as error:
        raise RuntimeError("Reader or its dependencies are missing. Run `uv sync` in this folder.") from error
    return MetaQuestReader


def format_pose(hand, transform, orientation=False):
    if transform is None:
        return f"{hand}: no tracking data yet"
    values = [float(transform[row][3]) for row in range(3)]
    if not all(math.isfinite(value) for value in values):
        return f"{hand}: invalid position"
    text = f"{hand}: x={values[0]:+.4f} m  y={values[1]:+.4f} m  z={values[2]:+.4f} m"
    if orientation:
        from scipy.spatial.transform import Rotation
        matrix = [[float(transform[row][column]) for column in range(3)] for row in range(3)]
        if not all(math.isfinite(value) for row in matrix for value in row):
            return text + "  orientation=invalid"
        # Extrinsic XYZ: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            roll, pitch, yaw = Rotation.from_matrix(matrix).as_euler("xyz", degrees=True)
        text += f"  roll={roll:+.2f} deg  pitch={pitch:+.2f} deg  yaw={yaw:+.2f} deg"
        if any("Gimbal lock" in str(w.message) for w in caught):
            text += "  (gimbal lock: roll/yaw are not uniquely determined)"
    return text


def check_apk():
    spec = importlib.util.find_spec("meta_quest_teleop")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Reader dependency missing. Run `uv sync` in this test folder first.")
    package = Path(next(iter(spec.submodule_search_locations)))
    apk = package / "APK/teleop-pointer-frame-relative.apk"
    if not apk.is_file():
        raise RuntimeError("Reader APK is missing. Fetch the repository's Git LFS files and reinstall the dependency; see README.md.")
    with apk.open("rb") as stream:
        header = stream.read(128)
    if header.startswith(b"version https://git-lfs.github.com/spec/"):
        raise RuntimeError("The APK is a Git LFS pointer, not the app. Fetch Git LFS files; see README.md.")
    if not header.startswith(b"PK"):
        raise RuntimeError("The bundled APK is not a ZIP/APK file. Check the dependency installation.")


def run(args):
    adb = shutil.which("adb")
    if adb is None:
        raise RuntimeError("ADB is missing. Install it with `sudo apt install adb`, then try again.")
    subprocess.run([adb, "start-server"], check=True)
    if args.list:
        subprocess.run([adb, "devices", "-l"], check=True)
        return
    MetaQuestReader = load_reader(getattr(args, "reader_source", None))
    hands = ("left", "right") if args.hand == "both" else (args.hand,)
    reader = None
    try:
        print("Connecting to the Quest headset. The reader installs its APK if it is not already installed.", flush=True)
        reader = MetaQuestReader(ip_address=args.ip)
        getter = (reader.get_hand_controller_transform_ros if args.frame == "ros"
                  else reader.get_hand_controller_transform_openxr)
        axes = "x forward, y left, z up" if args.frame == "ros" else "x right, y up, z backward"
        print(f"Frame: {args.frame} ({axes}); positions in meters. Origin is the Quest tracking reference, not the robot base.", flush=True)
        print("Keep the headset/app active and the controllers tracked. Ctrl+C stops printing.", flush=True)
        print("trigger: index-finger trigger value, 0=released and 1=fully pressed.", flush=True)
        started = time.monotonic()
        while args.duration is None or time.monotonic() - started < args.duration:
            stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            print(stamp + " | " + " | ".join(
                format_pose(hand, getter(hand), args.orientation)
                + f"  trigger={reader.get_trigger_value(hand):.3f}"
                for hand in hands), flush=True)
            time.sleep(1 / args.rate)
    finally:
        if reader is not None:
            reader.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--frame", choices=("ros", "openxr"), default="ros", help="Axis convention only; does not require ROS")
    parser.add_argument("--rate", type=float, default=5, help="Console print frequency in Hz, not headset capture rate (default: 5)")
    parser.add_argument("--orientation", action="store_true", help="Also print roll, pitch, and yaw in degrees (extrinsic XYZ convention)")
    parser.add_argument("--ip", help="Quest IP for wireless ADB; omit for USB")
    parser.add_argument("--duration", type=float, help="Stop printing after this many seconds")
    parser.add_argument("--list", action="store_true", help="List ADB devices without installing or launching the Quest app")
    parser.add_argument("--reader-source", type=Path, help="Reader checkout directory; defaults to vendor/meta_quest_teleop if present, otherwise the installed package")
    args = parser.parse_args()
    for name in ("rate", "duration"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name} must be finite and positive")
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nTracking stopped.")
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Tracking could not start: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
