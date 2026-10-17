#!/usr/bin/env python3
"""Read-only reset teaching. Save intermediate poses first, start pose last."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from panda_controller import JOINTS
from recording import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output exists; choose a new name to preserve the old reset sequence')
    import rclpy
    from franka_msgs.msg import FrankaState
    from rclpy.qos import qos_profile_sensor_data
    rclpy.init()
    node = rclpy.create_node('teach_demonstration_reset')
    samples = []
    node.create_subscription(FrankaState, '/franka_robot_state_broadcaster/robot_state',
                             lambda msg: samples.append((time.monotonic(), msg)), qos_profile_sensor_data)
    data = {'version': 1, 'joints': JOINTS, 'poses': []}
    print('READ ONLY: use your normal hand-guiding setup. This script sends no commands.')
    print('Place the arm at each intermediate pose, then the final start pose.')
    print('Enter: save stationary pose | u: remove last pose | q: finish')
    try:
        while True:
            choice = input('Pose> ').strip().lower()
            if choice == 'q':
                break
            if choice == 'u':
                if data['poses']:
                    data['poses'].pop()
            elif choice == '':
                samples.clear()
                end = time.monotonic()+1.
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=.02)
                recent = [s for t, s in samples if t > end-.5]
                if len(recent) < 3 or time.monotonic()-samples[-1][0] > .2:
                    print('No fresh robot state; pose not saved')
                    continue
                q = np.array([s.q for s in recent])
                if (not np.isfinite(q).all() or np.max(np.ptp(q, axis=0)) > .003
                        or max(np.max(np.abs(s.dq)) for s in recent) > .02):
                    print('Arm is moving; pose not saved')
                    continue
                data['poses'].append({'name': f'pose_{len(data["poses"])+1:03d}',
                                      'q': list(recent[-1].q)})
            else:
                continue
            a.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(a.output, data)
            print(f'Saved {len(data["poses"])} poses. Last pose is the start pose.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
