# Stationary scene calibration procedure

Goal: export the Panda model and the three RGB camera poses for a digital twin,
with the robot base as the world origin. All observations use the same stationary
ChArUco board. No camera-to-camera tape measurements are required.

## 1. Prepare the scene and board

- Fix the board flat on rigid backing at the center of the workplace. It must
  not slide, flex, or change orientation during fitting or validation.
- Leave the robot base and both scene-camera mounts fixed. Secure the wrist mount.
- Confirm the printed square pitch in both directions. Five square widths should
  be 125 mm and seven square heights 175 mm for the original 25 mm layout. If the
  print is uniformly scaled, use the measured square pitch in the capture command.
  If its horizontal and vertical pitches disagree, reprint.
- Close all other D405 viewers/recorders so the calibration process can open the
  color streams. Use steady illumination with minimal glare on the board.

The configured mapping is wrist `130322272081`, side `130322271876`, and opposite
side `130322271346`. Verify the physical cameras using the labelled preview.

## 2. Start the Panda interface

Prepare FCI using the procedure already established for this robot. In terminal A:

```bash
cd /home/jan/install/pixi_panda_ros2
mkdir -p /home/jan/dev/neura_demonstrator/tests/scene_calibration/captures
pixi run -e jazzy franka robot_ip:=172.16.0.2 2>&1 | \
  tee /home/jan/dev/neura_demonstrator/tests/scene_calibration/captures/panda_launch.log
```

This launch opens the hand and activates the arm's joint trajectory torque
controller. It stays running during capture.

The command records the driver output in `captures/panda_launch.log` so messages
are available even if they scroll out of the terminal. It replaces that log on
each launch; preserve it before starting another run if diagnosing an error.

## 3. Switch to state reading before hand-guiding

In terminal B:

```bash
cd /home/jan/install/pixi_panda_ros2
pixi run -e jazzy ros2 control switch_controllers \
  --deactivate joint_trajectory_controller --strict
pixi run -e jazzy ros2 control list_controllers
```

Confirm the trajectory controller is inactive and no other motion/effort
controller is active. State/pose/twist broadcasters should remain active. The
driver switches to continuous reading when no command controller is running.
Do this again if you restart terminal A, since the launch activates the trajectory
controller each time.

Use the Panda's normal Pilot-Grip enabling/guiding controls to reposition the arm.
Do not force it against an active holding controller. If normal guiding is
refused or causes an error, stop and inspect the mode before continuing.

The separate cable-connected black device is the external activation/user-stop
device, not a wrist guiding button. Use its normal operating sequence according
to this Panda's manual; controller deactivation does not replace it. Activating
that device interrupts external arm control. After repositioning, release the
wrist guiding controls and restore the external device to its released state for
FCI observation. Confirm fresh robot states resume before saving. The recorder
rejects user-stop/reflex states or stale data. If reading does not resume, inspect
the robot/driver status before proceeding; do not bypass the external device.

First test one repositioning and one capture before collecting the whole set.
Verify the deactivation succeeded and the launch reports continuous reading.
An `Overrun detected` / `missed cycles` controller-manager warning concerns its
scheduling deadline; it does not by itself establish a failed robot command.
An FCI control exception, `communication_constraints_violation`, stale robot
states, or active robot error requires inspection before further capture. The
standard kernel may miss scheduling deadlines, but do not attribute every error
to it or assume messages caused by pressing the external device are harmless.

## 4. Start the read-only robot-state bridge

In terminal C:

```bash
cd /home/jan/install/pixi_panda_ros2
pixi run -e jazzy python \
  /home/jan/dev/neura_demonstrator/tests/scene_calibration/robot_state_bridge.py
```

The bridge saves recent measured robot states and the robot URDF. It sends no
commands. Its default state topic is
`/franka_robot_state_broadcaster/robot_state`. If the capture reports missing
states, check that topic and domain 100:

```bash
pixi run -e jazzy ros2 topic list -t
```

## 5. Open all three camera previews

In terminal D, after verifying a 25 mm print:

```bash
cd /home/jan/dev/neura_demonstrator/tests/scene_calibration
uv run capture_scene.py --list
uv run capture_scene.py --square-mm 25
```

Replace `25` with the actual verified pitch if different. Confirm serial/role
mapping and reliable board detection. A usable observation requires at least
eight detected corners, useful image coverage, and acceptable reprojection error.
Prefer most or all of the 24 internal corners visible and spread across the image.

**The wrist camera must see enough of the board for every saved capture.** It can
lose sight of it while you move between poses. The whole board is not required.
Some gripper occlusion is acceptable if enough identifiable corners remain.

Both fixed cameras need several usable views of the same unmoved board. They
need not succeed in every wrist pose: the recorder saves whichever fixed-camera
observations are usable. Each needs at least three fitting observations and one
validation observation. Prefer clear views in many captures; move the arm out of
their sight lines when possible rather than moving their mounts or the board.

## 6. Collect deliberate, varied fitting poses

Aim for **20–30 captures using Space**. The solver calls these `train` captures;
they fit calibration parameters, not a neural network.

At each pose, release the guiding controls, let the robot settle for at least one
second, click the preview window to give it keyboard focus, and press Space once.
Keep the arm still through the short post-capture stability check. Verify the
terminal says `Saved ...: train`; a rejected capture does not count.

This 24-pose plan is a starting guide, not exact robot coordinates or commanded
movements:

| Group | Captures | Aim |
| --- | --- | --- |
| Near frontal views | 4 | Camera pointing toward the board from different lateral positions; avoid identical orientations for all captures |
| Left/right viewing tilts | 6 | Several clearly different left and right oblique views, aiming toward the board |
| Forward/back viewing tilts | 6 | Tilt about a second axis; include views from both directions |
| Combined diagonal tilts | 4 | Combine the two tilts, with the board still detectable |
| Different viewing distances | 4 | Nearer/farther views with different tilts and lateral positions |

Moderate view tilts around 20–35 degrees are a useful starting aim if visibility
and reach allow; no angle needs to be measured. An initial camera-to-board
distance around 20–40 cm often makes the prepared board practical, but let the
actual sharpness and corner detection determine usable positions. These are
viewing suggestions, not validated robot paths.

Do not keep the camera pointing straight down with an unchanged wrist orientation
for every capture. Translation alone cannot constrain the hand-eye mounting.
Rotations about at least two distinct axes are necessary. Rotating only around
the board normal is also insufficient. Cover different parts of the image while
keeping the board identifiable, and avoid near edge-on views, heavy occlusion,
blur, joint limits, or collisions with the work area.

## 7. Collect separate validation poses

Save **3–5 additional poses with V**, not Space. Choose a mix such as a left
oblique view, a forward/back oblique view, and a diagonal view at another distance.
Do not reuse the fitting poses. These captures are withheld from parameter
fitting. Ensure each scene camera has at least one clear board observation among
them. The board must still be unmoved.

The solver requires at least ten fitting poses and three validation poses, but
that minimum is not an accuracy guarantee. Spread and diversity matter more than
collecting many nearly identical images. Duplicate arm poses are rejected.

## 8. Solve, inspect, and retain the output

Quit capture with Q/Escape. The camera capture and bridge can now be stopped;
the offline solver does not need a live robot connection. In terminal D:

```bash
uv run solve_scene.py
```

An accepted result contains:

```text
results/scene.json
results/panda.urdf
results/calibration_report.json
```

Inspect the report for fit and held-out errors. The default limits are 5 mm
translation, 2 degrees rotation, and 3 pixels RMS reprojection error. They measure
consistency and do not certify absolute accuracy. Failed checks write a report
without an accepted scene file. Do not loosen limits to conceal poor captures.

The scene contains camera optical-frame transforms and RGB intrinsics, the
board pose, and the robot's reference joint configuration and URDF. The wrist
camera is attached to the flange, so it follows the arm in the digital twin.
URDF mesh assets, table dimensions, fixtures, and other objects are separate.

Keep raw captures and the report for reproducibility. Use a new output directory
for another solution; accepted `scene.json` files are not overwritten. Once the
result is validated, the board can be removed. Recalibrate affected transforms
when camera mounts or the robot base are moved.

Reference: [OpenCV hand-eye observation requirements](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
