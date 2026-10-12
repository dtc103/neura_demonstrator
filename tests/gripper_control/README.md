# Panda gripper control

Keep the existing Panda launch running:

```bash
cd ~/install/pixi_panda_ros2
pixi run -e jazzy franka robot_ip:=172.16.0.2
```

In another terminal:

```bash
cd ~/dev/neura_demonstrator/tests/gripper_control
.venv/bin/python gripper.py --close
.venv/bin/python gripper.py --open
```

This folder has its own uv environment with no additional Python dependencies.
The script invokes the existing Panda Pixi environment for ROS 2 and its
`franka_msgs` types, inheriting that environment's ROS domain configuration.
It sends one Move action to `/panda_gripper/move`, waits for the result, and exits.
It does not start the robot driver or command the arm. Closing requests 0 mm;
it is not a force-controlled grasp, and obstruction can produce a failed move.

Optional settings: `--speed 0.05` (m/s), `--timeout 15` (seconds),
`--action /panda_gripper/move`, and `--pixi-project PATH`.

To recreate only this folder's environment: `uv sync --offline`.
