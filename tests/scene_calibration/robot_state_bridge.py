"""Run with the Panda's Pixi Python; read ROS states into an atomic local file."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import time


def atomic_write(path, text):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "captures/current/robot_state.json")
    parser.add_argument("--topic", default="/franka_robot_state_broadcaster/robot_state")
    parser.add_argument("--description-topic", default="/robot_description")
    args = parser.parse_args()
    try:
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from franka_msgs.msg import FrankaState
        from std_msgs.msg import String
    except ImportError as error:
        raise SystemExit("Run this bridge with `pixi run -e jazzy python ...` from the Panda project, not the camera uv environment.") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # An exclusive lock prevents two bridge processes from interleaving states.
    import fcntl
    lock = args.output.with_suffix(".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("Another bridge is already writing this output file") from error
    rclpy.init()
    node = rclpy.create_node("scene_calibration_state_reader")
    history = deque(maxlen=150)
    bridge_id = f"{os.getpid()}-{time.monotonic_ns()}"
    last_write = 0

    def receive(message):
        nonlocal last_write
        now = time.monotonic_ns()
        sample = {"received_monotonic_ns": now, "received_unix_ns": time.time_ns(),
                  "ros_stamp_ns": message.header.stamp.sec * 10**9 + message.header.stamp.nanosec,
                  "robot_time_s": message.time, "o_t_ee": list(message.o_t_ee), "f_t_ee": list(message.f_t_ee),
                  "q": list(message.q), "dq": list(message.dq), "robot_mode": int(message.robot_mode),
                  "current_error_flags": [name for name in message.current_errors.get_fields_and_field_types()
                                          if getattr(message.current_errors, name)],
                  "control_command_success_rate": message.control_command_success_rate}
        history.append(sample)
        if now - last_write >= 25_000_000:
            atomic_write(args.output, json.dumps({"schema_version": 1, "bridge_id": bridge_id, "topic": args.topic,
                                                  "host": os.uname().nodename, "samples": list(history)}, allow_nan=False))
            last_write = now

    def description(message):
        if message.data:
            atomic_write(args.output.parent / "panda.urdf", message.data)

    node.create_subscription(FrankaState, args.topic, receive,
                             QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
    node.create_subscription(String, args.description_topic, description,
                             QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                        durability=DurabilityPolicy.TRANSIENT_LOCAL))
    print(f"Reading {args.topic}; writing {args.output.resolve()}", flush=True)
    print("This node only subscribes. It does not activate controllers or command the robot.", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        lock.close()


if __name__ == "__main__":
    main()
