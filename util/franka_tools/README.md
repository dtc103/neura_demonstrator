# Collision threshold CLI

Keep the Panda driver running, with teleoperation/replay stopped, then run:

```bash
cd ~/install/pixi_panda_ros2
pixi run -e jazzy python ~/dev/neura_demonstrator/util/franka_tools/configure_collision.py
```

The editor prompts for all eight collision threshold arrays. Press Enter to
keep an array, enter its replacement numbers separated by spaces or commas, or
enter `q` to cancel. It displays the complete request before asking you to type
`APPLY`. It checks that the robot reports a fresh stationary state without errors
before calling the driver service.

**The driver has no threshold getter.** Values displayed on the first run are
a snapshot of the installed driver's source defaults, not verified active values.
Subsequent runs display the last successfully applied values saved by this tool.
Restarting the driver or changing settings elsewhere can make that record stale.

Torque arrays contain seven joint thresholds in Nm. Force arrays contain
`Fx Fy Fz Mx My Mz`, with forces in N and moments in Nm. Lower thresholds detect
contact; upper thresholds trigger collision stops. Nominal and acceleration
thresholds are edited separately. These settings change collision detection,
not the controller's commanded contact force.

Options:

- `--dry-run`: edit and review without connecting to ROS or changing anything.
- `--defaults`: use source defaults as the editing baseline.
- `--state-file PATH`: use a different last-applied record.
- `--service NAME`: override the Panda service name.
- `--timeout SECONDS`: service availability/state/response timeout (default 10).

A response timeout leaves the application outcome unknown: the tool does not
retry or update the last-applied record. No settings are applied automatically
at startup.
