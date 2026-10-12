#!/usr/bin/env python3
"""Open or close the Panda gripper using the running ROS 2 gripper server."""

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


def operate(args):
    import rclpy
    from rclpy.action import ActionClient
    from franka_msgs.action import Move
    from action_msgs.msg import GoalStatus

    rclpy.init(args=[])
    node = rclpy.create_node("panda_gripper_command")
    client = ActionClient(node, Move, args.action)
    handle = None

    def wait(future, timeout):
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("Gripper action timed out")
            rclpy.spin_once(node, timeout_sec=0.05)
        result = future.result()
        if result is None:
            raise RuntimeError("Gripper action returned no result")
        return result

    try:
        if not client.wait_for_server(timeout_sec=5):
            raise RuntimeError(f"{args.action} unavailable. Start the Panda launch first.")
        goal = Move.Goal()
        goal.width = 0.08 if args.open else 0.0
        goal.speed = args.speed
        print(f"{'Opening' if args.open else 'Closing'} gripper to {goal.width * 1000:g} mm…", flush=True)
        submission = client.send_goal_async(goal)
        # Also cancel a late-accepted goal if submission times out.
        cancel_pending = False

        def accepted(future):
            accepted_handle = future.result()
            if cancel_pending and accepted_handle is not None and accepted_handle.accepted:
                accepted_handle.cancel_goal_async()

        submission.add_done_callback(accepted)
        try:
            handle = wait(submission, 5)
        except (Exception, KeyboardInterrupt):
            cancel_pending = True
            rclpy.spin_once(node, timeout_sec=0.1)
            raise
        if not handle.accepted:
            raise RuntimeError("Gripper command rejected")
        result = wait(handle.get_result_async(), args.timeout)
        if result.status != GoalStatus.STATUS_SUCCEEDED or not result.result.success:
            raise RuntimeError(f"Gripper command failed: {result.result.error or 'unsuccessful action result'}")
        print("Gripper opened." if args.open else "Gripper closed.")
    except (Exception, KeyboardInterrupt):
        if handle is not None and handle.accepted:
            try:
                wait(handle.cancel_goal_async(), 2)
            except Exception:
                pass
        raise
    finally:
        client.destroy()
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--open", action="store_true", help="Open to 80 mm")
    command.add_argument("--close", action="store_true", help="Close to 0 mm using Move, without a force-grasp command")
    parser.add_argument("--speed", type=float, default=0.05, help="Finger separation speed in m/s (default: 0.05)")
    parser.add_argument("--timeout", type=float, default=15, help="Command completion timeout in seconds")
    parser.add_argument("--action", default="/panda_gripper/move")
    parser.add_argument("--pixi-project", type=Path, default=Path.home() / "install/pixi_panda_ros2")
    parser.add_argument("--ros-runtime", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("speed", "timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    if not args.ros_runtime:
        pixi = shutil.which("pixi")
        if pixi is None:
            parser.error("pixi is not on PATH; run this script with the Panda Pixi Python and --ros-runtime")
        if not args.pixi_project.is_dir():
            parser.error(f"Pixi project not found: {args.pixi_project}")
        result = subprocess.run([pixi, "run", "-e", "jazzy", "python", str(Path(__file__).resolve()),
                                 *sys.argv[1:], "--ros-runtime"], cwd=args.pixi_project)
        return result.returncode
    try:
        operate(args)
    except (ImportError, RuntimeError, TimeoutError, KeyboardInterrupt) as error:
        print(f"Gripper command stopped: {error or 'interrupted'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
