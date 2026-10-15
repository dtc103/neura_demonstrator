"""Print controller-only WebXR local-floor poses; no robot connection."""
import argparse
from datetime import datetime, timezone
import math
import time
import warnings
from webxr_reader import WebXRReader
from native_reader import add_backend_arguments, start_reader

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



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--frame", choices=("ros", "openxr"), default="ros")
    parser.add_argument("--rate", type=float, default=5)
    parser.add_argument("--orientation", action="store_true", help="Print roll, pitch and yaw in degrees")
    parser.add_argument("--duration", type=float)
    add_backend_arguments(parser)
    args = parser.parse_args()
    for name in ("rate", "duration"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name} must be finite and positive")
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    reader = None
    try:
        reader = start_reader(args)
        hands = ("left", "right") if args.hand == "both" else (args.hand,)
        getter = reader.get_hand_controller_transform_ros if args.frame == "ros" else reader.get_hand_controller_transform_openxr
        print(f"Controller grip poses in a fixed tracking space, {args.frame} axes. Positions in meters; trigger 0..1.", flush=True)
        started, previous_epoch = time.monotonic(), None
        while args.duration is None or time.monotonic() - started < args.duration:
            epoch = reader.get_frame_epoch()
            if epoch is not None and epoch != previous_epoch:
                print(f"Tracking reference: {epoch}", flush=True)
            previous_epoch = epoch
            stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            print(stamp + " | " + " | ".join(format_pose(hand, getter(hand), args.orientation)
                  + f"  trigger={reader.get_trigger_value(hand):.3f}" for hand in hands), flush=True)
            time.sleep(1 / args.rate)
    except KeyboardInterrupt:
        print("Tracking stopped.")
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Cannot start tracking relay: {error}. Close other viewers using this port.") from error
    finally:
        if reader is not None:
            reader.stop()


if __name__ == "__main__":
    main()
