# Quest → Panda teleoperation

`teleoperate.py` uses Quest browser tracking and, by default, the same
`joint_trajectory_controller` used for calibration. Pink/Pinocchio converts relative
TCP targets into joint targets. Robot masses/inertias are not used by this IK layer.
The installed joint controller still uses effort PID control; this does not bypass
robot dynamics or guarantee stability. No collision checking is implemented.

## Install the joint-backend dependencies

The existing Jazzy environment already contains Pinocchio, NumPy, SciPy and ROS.
Install Pink (the PyPI package is **pin-pink**, not `pink`) and its QP solver:

```bash
cd ~/install/pixi_panda_ros2
uv pip install --python .pixi/envs/jazzy/bin/python \
  -r ~/dev/neura_demonstrator/util/demonstration_collection/requirements-joint.txt
```

This modifies the existing environment; a future Pixi reinstall may require repeating
it. Package downloads failed with a DNS error in the development session. Installation
and the real Pink integration test remain necessary on your machine. Run the tests:

```bash
pixi run -e jazzy python -m unittest discover \
  -s ~/dev/neura_demonstrator/util/demonstration_collection -p 'test_*.py' -v
```

The Pink test should **pass**, not be skipped, before hardware validation.

## Start and preview

Terminal 1, existing driver (FCI enabled):

```bash
cd ~/install/pixi_panda_ros2
pixi run -e jazzy franka robot_ip:=172.16.0.2
```

A fresh driver launch activates `joint_trajectory_controller`. Do not switch to
Cartesian impedance for the default backend. If a previous session left Cartesian
impedance active, stop its command publishers and switch back explicitly:

```bash
pixi run -e jazzy ros2 control switch_controllers \
  --deactivate cartesian_impedance_controller \
  --activate joint_trajectory_controller --strict
```

Terminal 2, preview (no arm/gripper commands):

```bash
cd ~/install/pixi_panda_ros2
pixi run -e jazzy python \
  ~/dev/neura_demonstrator/util/demonstration_collection/teleoperate.py \
  --backend joint --scale 1.0 --frame-alignment world --yaw-deg 0
```

Terminal 3:

```bash
adb reverse tcp:8443 tcp:8443
```

Open `http://localhost:8443/` inside the **Quest browser**, then Start controller
tracking. Stop any separate Quest viewer first because it uses the same port.
No robot bridge, native APK, camera recorder or separate relay is needed.

Preview computes Pink targets and bounded trajectories using live robot state.
Because the robot does not follow preview commands, large preview movements can
trigger the tracking-error guard; use small motions to check axes. Logs include
requested TCP pose, measured pose, joint endpoint, solver/planner time and inputs.

After dependencies and offline tests pass, hardware validation is enabled with
`--execute` on the same command. This implementation has not been hardware-tested.
Validate stationary hold and small translation-only movements before full rotation;
`--translation-only` holds TCP orientation for each clutch engagement. Do not run
calibration trajectories or other action clients concurrently with teleoperation.
Topic-publisher checks do not detect competing trajectory action clients.

## Controls and coordinate mapping

- **Right thumbstick click:** initialize/reset the reference and workspace center.
  Startup waits indefinitely. Release grip and trigger afterward.
- **Right grip:** clutch. Each engagement records current robot and controller poses;
  subsequent controller displacement is mapped relative to those poses. Release
  replaces the outstanding joint target with a finite braking trajectory. Control
  stays running; re-engagement uses fresh references.
- **Right trigger click:** first press closes, next press opens (80 mm).
  Holding the trigger does not repeat commands. Release below 20% before
  pressing above 80% for another click. Release once after startup, reference
  reset, or tracking loss to re-arm; releasing alone never opens the gripper.
  Closing uses a Grasp action (default 20 N); opening uses Move.
  Each click sends one complete action. Clicks while busy are ignored with a
  message, never queued. Grip release does not cancel a gripper movement.
  Action failures/timeouts still stop the program. First click after restarting
  the program always requests close.
- **Hold right B:** precision translation and rotation (`--precision-scale`, default 0.5).
  Release B for normal translation (`--scale`, default 1.0). Both edges rebase
  references without jumping the target; rotation scaling is unaffected.
- **Tracking loss/recenter:** brake and require release/re-engagement. A changed
  browser tracking reference additionally requires another thumbstick reset.
- **Ctrl+C:** best-effort braking, leaves the selected controller active holding
  the endpoint. No automatic robot-error recovery or controller switching.

`--scale 1.0` means 10 cm of controller displacement requests 10 cm of robot
translation; `--scale 0.5` requests 5 cm. Rotation is relative and one-to-one,
subject to configured command speed limits. Euler angles are not subtracted.

World alignment at yaw zero maps Quest forward (-Z) to robot-base +X, right (+X)
to base -Y, and up (+Y) to base +Z. `--yaw-deg` adjusts the fixed horizontal
alignment. This is a manually chosen correspondence, not an automatically measured
Quest-to-robot calibration.

Alternatively, `--frame-alignment tool` sets `B = R_tcp @ R_controller.T` on
thumbstick reset. Controller-local axes at that instant map to TCP-local axes;
B stays fixed through clutch re-engagement. The same B maps translation and relative
rotation. Yaw is ignored in tool mode. A downward-pointing TCP can map a horizontal
hand direction vertically: use world mode for room-up = robot-up.

The live URDF root is used by Pinocchio, matching the installed pose broadcaster's
`base` label. TCP is `panda_hand_tcp`. Startup compares model FK against measured
TCP and rejects mismatched frames. Joint names/order are checked too.

## Parameters

| Flag | Default | Meaning |
|---|---:|---|
| `--backend` | joint | `joint` (Pink + JTC) or existing `cartesian` torque-controller target backend |
| `--scale` | 1.0 | Translation ratio |
| `--smoothing-seconds` | 0.05 | Pose low-pass time constant; 0 disables filtering |
| `--max-speed` | 10.0 | TCP target speed cap, m/s |
| `--max-angular-speed` | 10.0 | TCP target angular speed cap, rad/s |
| `--trajectory-horizon` | 0.35 | Joint interpolation/braking horizon, seconds; minimum 0.1 |
| `--joint-speed` | 2.0 | Joint command speed cap, rad/s, also constrained by URDF |
| `--joint-acceleration` | 5.0 | Joint command acceleration cap, rad/s² |
| `--joint-jerk` | 50.0 | Joint command jerk cap, rad/s³ |
| `--workspace-radius` | 0.35 | Sphere around last reset, meters; not collision geometry |
| `--frame-alignment` | world | Fixed world/yaw mapping or reset-time tool alignment |
| `--yaw-deg` | 0 | World mapping yaw in degrees |
| `--translation-only` | off | Ignore controller rotation |
| `--stale-seconds` | 0.25 | State receipt freshness threshold |
| `--grasp-force` | 20 | Closing grasp force, N |
| `--gripper-speed` | 0.1 | Finger separation speed, m/s |
| `--port` | 8443 | Browser relay port |
| `--log` | automatic | Diagnostic JSONL file |
| `--execute` | off | Send arm and gripper commands |

## Trajectory behavior and limitations

At a nominal 50 Hz, Pink solves a local differential IK problem using the previous
commanded joint configuration, a TCP frame task, and weak posture regularization.
Joint limits include a 0.02 rad margin. New inputs replace old targets; there is no
queue of controller poses. Unreachable targets can be approximated by differential
IK rather than producing an exact solution; tracking-error guards stop accumulation.

Each replacement is a quintic joint segment starting at the previously planned
position, velocity and acceleration and ending at rest. Polynomial extrema are
checked for joint position, velocity, acceleration and jerk limits. If necessary,
progress toward the latest target is reduced. Clutch release requests a braking
endpoint based on commanded velocity/acceleration, rather than snapping velocity
instantaneously to zero. Holding does not repeatedly restart that braking segment.

ROS messages start on receipt. Command continuity is based on the local monotonic
clock; transport/controller scheduling can add timing mismatch. Python is not a
hard realtime servo. Every transmitted segment ends at rest, limiting continued
commanded motion on process loss to its remaining horizon, but it is not a physical
stop or a guarantee against controller instability. Actual joint motion can differ
from the commanded derivatives. Hardware protections remain in place.

More than 0.15 rad joint tracking error aborts planning; more than 8 cm / 0.5 rad
TCP execution error disarms the clutch. For the joint backend this compares measured
TCP with the current planned joint trajectory, not the requested hand target. Input
lag is reported separately and is not itself a robot execution fault. State/controller loss and robot errors stop
the program. No collision model for table, cables, or robot links is used.

The older backend is available explicitly with `--backend cartesian` and requires
the Cartesian controller active. It is retained for comparison, not presented as a
fix for the previously observed oscillations.

## Responsive profile and precision mode

Defaults now allow input targets up to 10 m/s and 10 rad/s, with commanded joint
limits of 2 rad/s (also capped by each URDF limit), 5 rad/s² acceleration, and
50 rad/s³ jerk. Input caps do not describe actual TCP or joint velocity. The
latest user run with 50 rad/s³ jerk reached about 0.48 rad/s measured joint speed;
this is evidence from that run, not validation of all postures or these new defaults.
The acceleration cap is lower than the user's 50 rad/s² override. Workspace and
execution-error guards remain enabled.

Right **B** selects precision translation and rotation only while held.
The default 0.5 ratio halves translation and rotational increments, and halves
the angular-speed cap. Releasing B restores the normal translation scale and
1:1 rotation. Press/release rebases the references without a target jump. Its current state is
used again after tracking reconnects, rather than retaining a toggled mode.
The active scale, trigger clicks, and gripper state are printed or logged.
Reload the Quest browser page to receive the B-button field; older pages cannot
request precision mode. Use `--grasp-force` for closing force (default 20 N)
and `--gripper-speed` for movement speed (default 0.1 m/s).

To retain the user's physical world alignment and workspace choice, pass
`--yaw-deg 180 --workspace-radius 1.0`; these are not changed globally.

The gripper speed parameter sets the Move/Grasp speed.
A full 80 mm move at 0.1 m/s takes at least 0.8 s plus action/communication latency;
Trigger clicks request a full movement, with no incremental stepping or retargeting.

If terminal output says `B=missing`, stop the Quest VR session and reload
`http://localhost:8443/`, then Start controller tracking again. The page diagnostics
should say `Client: 20261014-b-hold` and display the right B state. The terminal
prints B state alongside scale; missing fields produce an explicit warning.

## Demonstration collection

See [COLLECTION.md](COLLECTION.md) for the desktop GUI, reset teaching, lossless RGB-D recording, timestamps and clutch-segment export. Standalone teleoperation remains available.
