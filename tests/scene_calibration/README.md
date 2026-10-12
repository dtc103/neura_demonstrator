# Scene calibration: wrist camera and two fixed cameras

This folder holds the calibration workflow for a Franka Panda and three RealSense
D405 cameras. It has its own uv project and `.venv`, board tools, a stationary
camera/robot capture program, an offline hand-eye solver, and a scene exporter.
The Panda is connected through the community Pixi ROS 2 Jazzy driver with
libfranka 0.9.2. Calibration tools only subscribe to robot state; they never send
motion, gripper, controller-switching, or recovery commands.

## Record and export the scene

For teaching and replaying a predefined robot path with automatic stopped-pose
captures, use [AUTOMATIC_CAPTURE.md](AUTOMATIC_CAPTURE.md). The default path
review sends no robot commands; replay requires the explicit `--execute` option.

An enlarged A4 target is available in [board_large/](board_large/README.md):
35 mm squares and a 175 × 245 mm patterned area. Use `scene_config_take3.json`
for that board, `captures/take3` for its bridge/captures, and `results/take3` for
its solver output. Earlier board definitions and datasets remain separate.

For the complete terminal sequence and a suggested capture plan, follow
[PROCEDURE.md](PROCEDURE.md).

The camera mapping in `scene_config.json` is:

| Role | Serial |
| --- | --- |
| Wrist | `130322272081` |
| Side | `130322271876` |
| Opposite side | `130322271346` |

Capture verifies these serials against connected devices. Confirm the physical
mapping using the preview labels. The board must remain fixed in the center of
the workplace for the entire session, including validation. Fix both scene camera
mounts and the wrist mount, too.

1. **Confirm the printed dimensions.** The default is 25 mm square pitch. Measure
   several squares horizontally and vertically and divide by the number of
   squares. Uniform printer scaling can be accommodated with `--square-mm` using
   the measured value. If horizontal and vertical pitches differ noticeably,
   reprint instead of supplying one compromise size. This is necessary because
   the previous print dialog had automatic scaling enabled. Do not regenerate or
   move the board during capture.
2. **Start your existing Panda ROS launch** using your established procedure.
   The supplied driver launch activates the joint trajectory torque controller
   and opens the hand automatically; the calibration programs themselves do not.
   Manual arm positioning requires a supported guiding/control procedure. Do not
   push the arm against its active holding controller. The recorder can remain
   open while you reposition, but it will only accept stationary observations.
   Before using native hand-guiding, deactivate the startup trajectory controller
   and check that only broadcasters remain active:

   ```bash
   cd /home/jan/install/pixi_panda_ros2
   pixi run -e jazzy ros2 control switch_controllers --deactivate joint_trajectory_controller --strict
   pixi run -e jazzy ros2 control list_controllers
   ```

   The hardware driver switches to continuous reading with no command controller.
   If normal guiding is refused, stop and check the robot mode rather than pushing
   it or activating a different controller blindly.
3. **Start the read-only ROS bridge in a second terminal.** Use the Panda's Pixi
   Python, which has `rclpy` and `franka_msgs`. The camera uv environment does not
   need ROS Python bindings. Both programs must run on the same PC.

   ```bash
   cd /home/jan/install/pixi_panda_ros2
   pixi run -e jazzy python /home/jan/dev/neura_demonstrator/tests/scene_calibration/robot_state_bridge.py
   ```

   Default topic: `/franka_robot_state_broadcaster/robot_state`. Confirm it with
   `ros2 topic list -t` in domain 100. If your topic differs, pass `--topic NAME`.
   The bridge also saves `/robot_description` as `captures/current/panda.urdf`.
   It converts no matrices: raw column-major `o_t_ee` and `f_t_ee` are preserved.
   Camera capture derives the measured flange pose by removing the configured
   end-effector offset. ROS and camera environments communicate through an atomic
   local state-history JSON file, rather than mixing Pixi and uv Python packages.
4. **Stop other programs using the D405 streams and start capture.** After
   verifying a 25 mm print, run:

   ```bash
   cd /home/jan/dev/neura_demonstrator/tests/scene_calibration
   uv run capture_scene.py --list
   uv run capture_scene.py --square-mm 25
   ```

   Replace `25` with the verified square pitch if different. The program displays
   all three cameras with their role/serial, detected corners, and pose quality.
   Verify that the board is detectable before recording poses.
5. **Collect training captures.** Position the arm so the wrist camera sees the
   board. Let it settle for at least a second, then press **Space** to save a
   training capture. Change both position and orientation. Aim for 20–30 varied
   poses, with rotations about at least two axes; the solver requires at least
   ten training captures. The wrist view is required for every saved pose. The
   scene cameras are recorded whenever their board detection is usable; each
   needs at least three training observations. Partial board visibility is fine
   when enough corners are spread across the image. The capture gate requires
   eight corners, sufficient image coverage, and RMS reprojection error ≤1.5 px.
6. **Collect separate validation captures.** At least three additional arm poses
   are saved with **V**. These are withheld from fitting. Each fixed camera must
   have at least one usable board observation in the validation set. Repeated arm
   poses are rejected, including validation poses duplicated from training.
7. **Quit capture with Q/Escape and solve offline.** The robot need not be running
   for this step:

   ```bash
   uv run solve_scene.py
   ```

   An accepted calibration produces:

   ```text
   results/scene.json
   results/panda.urdf
   results/calibration_report.json
   results/calibration_summary.txt
   ```

   The default gates are maximum fixed-board consistency error of 5 mm and 2°,
   and RMS reprojection error ≤3 px, for both training and validation captures.
   These are initial acceptance limits, not a certified accuracy specification.
   Failure produces a report but no accepted `scene.json`. Poor rotation
   diversity or missing captures stops fitting with an explicit message.
   Do not loosen limits just to obtain a file without investigating the errors.

   The solver prints per-camera training/validation tables with observation
   counts, average/maximum position, rotation, and image reprojection errors,
   pass counts, and the worst failing capture IDs. `!` marks a maximum exceeding
   its configured limit. The same overview is saved as `calibration_summary.txt`,
   including when validation fails and no accepted scene is exported.

The saved session includes raw PNG images, corner correspondences, factory
intrinsics/distortion, robot joint positions and velocities, native robot
matrices, and ROS, robot, device, and local receipt timestamps. Capture checks
stationarity around the observation, freshness of the bridge and camera frames,
and robot error flags. This is a **stopped-pose procedure**, not hardware time
synchronization suitable for moving demonstrations. Its local receipt timestamps
cannot establish the exact camera exposure time relative to robot state.

D405 color streams can report `distortion.inverse_brown_conrady`. This is a
stream lens-distortion model, not a different camera product. The capture and
preview support it through the RealSense SDK: detected color pixels are converted
to bearing rays for PnP, and the SDK projects points back into the original color
image for quality checks and overlays. No measured depth or replacement camera
is required. The native coefficients remain in the exported scene file; they
must not be treated as ordinary OpenCV Brown coefficients by the simulator.

For a new session, change `session` in a separate copy of `scene_config.json`,
and start the bridge with `--output PATH_TO_SESSION/robot_state.json`. Capture
refuses to mix different boards, serials, or intrinsics in an existing session.
Export preserves existing accepted scenes; use `--output results/NEW_NAME` for
another result.

## What scene.json describes

- Robot base at the world origin, the captured URDF, and a reference joint pose.
- Fixed side/opposite camera poses relative to the robot base.
- Wrist optical-camera pose relative to `panda_link8` (the flange), plus a pose at
  the reference arm configuration. Its pose changes with the arm during use.
- Each RGB stream's intrinsics, distortion, resolution, and serial number.
- Board geometry and board pose relative to the robot base.
- Camera-origin separations at the reference arm configuration and validation
  provenance.

Matrices use explicit `T_A_from_B` names with row-major nested JSON arrays and
column vectors. Camera coordinates use RealSense/OpenCV optical axes: X right,
Y down, Z forward. A simulator may require a coordinate conversion. The board
origin is a corner of the patterned rectangle, not the center of the paper.

The URDF references robot mesh assets through `package://franka_description/...`;
those assets remain external. This is not a complete simulator project or a scan
of the room. Table bounds/thickness, fixtures, objects, camera housings, and mount
geometry need separate measurements/CAD. Robot/camera placement can be imported
into your chosen simulator from this neutral scene description.

OpenCV 5.0.0.93 in the current environment omits the Python hand-eye binding
([upstream issue](https://github.com/opencv/opencv/issues/29565)). The solver uses
OpenCV's Park method when available and an equivalent log-rotation/SVD plus
least-squares implementation otherwise. The report identifies which was used.
Synthetic tests verify known mounting transforms, scene-camera transforms,
column-major Franka conversion, stationarity rejection, and held-out failure.

## Current step: prepare the board and check the cameras

```bash
cd tests/scene_calibration
uv sync
uv run generate_board.py
uv run check_board.py --list
uv run check_board.py --serial YOUR_CAMERA_SERIAL
```

Stop other viewers that are using the selected camera first. Repeat the preview
for the wrist camera and both scene cameras. The window marks detected corners
and, for supported camera distortion models, overlays an estimated board frame.
This checks target visibility; it is not a finished robot calibration.

The generated target is `board/charuco_a4.svg`, with its matching definition in
`board/board.json`. Print the SVG on A4 portrait at **100% / actual size**, without
fit-to-page or printer/browser headers and footers. The patterned area must be
**125 × 175 mm**, made of **25 mm squares**. Check the dimensions in both directions
and mount the print flat on a rigid backing. The default is a starting target;
if it is too small in the scene views or too large in the wrist view, adjust the
layout before collecting calibration data. Do not scale a print without updating
the definition used for detection.

The D405's ideal depth range is **7–50 cm**. Check actual depth quality if the
scene cameras are farther away. ChArUco pose estimation in the preview uses color
corners and the known board geometry, not measured depth, so color detection and
useful depth coverage should be assessed separately.

For the Panda's future ROS 2 connection, first record the installed robot system
version and select compatible driver/libfranka versions. We need a measured
end-effector pose, not just a commanded pose. In libfranka this is `O_T_EE`;
`F_T_EE` describes the configured end-effector frame relative to the flange. These
are column-major 4×4 matrices. The eventual ROS adapter must preserve that frame
meaning and use matching image/robot observations. Do not install a driver solely
on the assumption that the latest release supports this older Panda.

Board-generation checks (no robot or cameras required):

```bash
uv run python -m unittest -v
```

## What needs to be recovered

Use the robot base as the common coordinate frame. Recover full position and
orientation (a rigid 3D transform), not just a distance between camera housings.

| Relationship | How to obtain it | Does it change during operation? |
| --- | --- | --- |
| Wrist camera → robot flange/tool frame | Eye-in-hand calibration | Constant while mounting is unchanged |
| Robot flange/tool frame → robot base | Robot's measured pose / forward kinematics | Changes when the arm moves |
| Scene camera 1 → robot base | Shared target anchored through wrist calibration, or eye-to-hand calibration | Constant while mounting is unchanged |
| Scene camera 2 → robot base | Same procedure for camera 2 | Constant while mounting is unchanged |

Distinguish the robot flange from the configured tool center point (TCP). Record
exactly which frame the robot API returns and use it consistently. A camera's
optical frame is also different from its housing or mounting bracket.

Calibrating cameras does not automatically recover the pose of every object in
the scene. Object positions require an additional measurement, such as depth
plus detection, a marker on an object, or multi-view reconstruction.

## Recommended common-target workflow

The following assumes the wrist camera is rigidly mounted to a functioning robot
arm, the robot exposes its measured Cartesian pose, and all cameras can observe
the same stationary calibration target. They do not have to observe it at the
same instant if it remains fixed throughout the procedure.

1. **Identify cameras by serial number.** Assign `wrist`, `scene_1`, and `scene_2`.
   Keep each camera's resolution, stream type, intrinsics, and distortion model
   with its captures. RealSense intrinsics can be read through the SDK; verify
   their suitability rather than assuming an image is distortion-free. If using
   raw depth points, also handle the device's depth-to-color extrinsics.
2. **Prepare a rigid, flat calibration board.** A ChArUco board combines
   checkerboard corners and identifiable markers. Select square sizes and overall
   dimensions that are resolvable by both the close wrist camera and the distant
   scene cameras. Print at actual size, mount it flat, and verify the physical
   square pitch in both directions with a ruler or calipers. The supplied square
   size establishes metric scale.
3. **Fix the board in the workspace.** Choose a location visible from all three
   cameras. Its distance to the robot base does not need to be measured. Leave
   the board and fixed camera mounts untouched during calibration.
4. **Collect wrist-camera observations at varied robot poses.** A practical
   starting target is 20–30 usable poses, with translations and substantial
   rotations about different axes. This is a planning target, not a guarantee of
   accuracy. Keep the board visible and in focus. For initial capture, stop and
   let the arm settle, then pair an image with the robot's measured pose. Record
   timestamps and avoid pairing an old buffered image with a new pose. Pure
   translation or rotation about only one axis does not adequately constrain the
   hand-eye solution.
5. **Solve the wrist mounting transform.** Detect board corners, estimate the
   board's pose in the wrist camera with PnP, and pass the paired robot and board
   poses to OpenCV's `calibrateHandEye`. Use explicit transform directions and
   one unit convention (meters for translations). Its eye-in-hand output maps
   the camera frame into the robot's chosen flange/tool frame.
6. **Anchor the board to the robot base.** Combine each robot pose, the solved
   wrist mounting transform, and the wrist-camera board observation. These
   estimates should agree on one stationary board pose. Fit a representative
   rigid transform and examine the spread; do not blindly average matrix entries.
7. **Anchor both scene cameras.** Estimate the same fixed board's pose in each
   scene camera. Combine that with the board-to-base transform from step 6.
   Collect repeated observations and reject bad detections. This connects the
   two fixed cameras and the moving wrist camera through the robot base.
8. **Validate on observations withheld from fitting.** Check image reprojection
   errors and how consistently held-out wrist observations recover the fixed
   board in robot-base coordinates. Cross-check a common target point across
   cameras and report errors in millimeters and degrees. Choose acceptance limits
   based on the task rather than promising a specific accuracy in advance.

If a common fixed target is not visible to all cameras, an alternative is to
attach a target rigidly to the robot tool and perform **eye-to-hand calibration**
for the scene cameras. The board then moves with the arm. This is a separate
capture phase: do not move the board during the fixed-target wrist procedure.

## Transform convention for implementation

`T_A_from_B` converts coordinates expressed in B into coordinates expressed in A.
Each transform is a 4×4 matrix, with its translation expressed in meters.

Define `base`, `tool`, `wrist`, `board`, `scene_1`, and `scene_2` explicitly.
`tool` below means whichever rigid robot frame was consistently used for the
recorded poses, not necessarily the TCP.

```text
T_base_from_wrist(t) = T_base_from_tool(t) @ T_tool_from_wrist

T_base_from_board(i) = T_base_from_tool(i)
                       @ T_tool_from_wrist
                       @ T_wrist_from_board(i)

T_base_from_scene_1 = T_base_from_board @ inverse(T_scene_1_from_board)
T_base_from_scene_2 = T_base_from_board @ inverse(T_scene_2_from_board)

T_scene_1_from_scene_2 = inverse(T_base_from_scene_1) @ T_base_from_scene_2
T_scene_1_from_wrist(t) = inverse(T_base_from_scene_1) @ T_base_from_wrist(t)
```

Thus the wrist-to-scene relationship changes with arm pose even though the
wrist mounting calibration is constant. The length of a transform's translation
vector gives the separation between those coordinate-frame origins.

## Automatic computation versus manual work

The numerical solution can be automatic once images, robot poses, camera
intrinsics, and the board's actual dimensions are available. Initially, manually
position the board and move/teach suitable arm poses. Automatic capture at each
stopped pose can follow; automatic robot motion needs the actual robot interface
and a trajectory appropriate for this physical setup.

A measuring tape is useful for rough setup checks. It is not necessary to
measure every camera-to-camera distance, nor does it recover rotations or exact
optical centers. Measuring the board's true dimensions is necessary unless using
a target with already certified dimensions.

Recalibrate affected relationships when a camera mount, board attachment used
for eye-to-hand capture, or robot tool definition changes. Also verify the
intrinsics if stream settings change. Keep serial numbers, stream profiles,
frame names, units, raw captures, timestamps, and validation results with every
calibration output. Synchronizing cameras during moving demonstrations remains
a separate task from this spatial calibration.

## References

- [OpenCV hand-eye calibration and transform directions](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [OpenCV ChArUco detection and board pose estimation](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)
- [RealSense SDK projection, intrinsics, and extrinsics](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)
- [D405 specifications and ideal range](https://www.realsenseai.com/product-family/d405-series/)
- [Franka measured pose and frame definitions](https://frankarobotics.github.io/libfranka/latest/structfranka_1_1RobotState.html)
- [Franka robot system and library compatibility](https://frankarobotics.github.io/docs/compatibility.html)
# Automated pipeline

See [PIPELINE.md](PIPELINE.md) for the single-command capture, calibration fit,
held-out evaluation, and scene export workflow in `run_pipeline.py`.

