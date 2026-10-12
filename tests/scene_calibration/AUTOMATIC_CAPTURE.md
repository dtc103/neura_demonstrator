# Taught stop-and-capture sequence

This workflow records a fresh sequence of measured joint positions, then replays
it through the existing ROS `FollowJointTrajectory` action. At each capture pose
it waits for the action to finish, measured joint motion to settle, and the camera
recorder to acknowledge a saved observation before advancing.

The new session is `captures/automatic`; its configuration is
`scene_config_auto.json`, using the 35 mm target. Earlier sessions are preserved.
The scripts have been tested offline and with mocked ROS/cameras. No actual robot
trajectory has been executed or collision-checked by the assistant.

## Phase A: teach a fresh path

Delete a mistaken waypoint without a robot connection:

```bash
.venv/bin/python teach_waypoints.py --delete transit_001
```

Use a waypoint name or one-based replay index (`--delete 25`). Add `--output`
for a different trajectory file. The original is backed up before deletion;
remaining names and capture splits are preserved. Stop replay before editing
the path, and review the new transition between the remaining neighbours.

To repair an existing transition, start the read-only teacher with
`--insert-after pose_024` (or the displayed one-based index `24`). Use the same
config and output path as the original path. With motion controllers inactive,
hand-guide to a clearance position, release guiding controls, and type `M`.
Each additional `M` inserts another transit point in teaching order, before the
original next waypoint. Type `Q` to finish. Existing waypoint names and capture
splits remain unchanged, and an original-path backup is saved on the first
insertion. The inserted points have unique `transit_NNN` names and do not take
pictures. Review both sides of every new transition; insertion does not perform
collision checking. Replay indices change when points are inserted.

1. Keep the 35 mm board and camera mounts stationary. Use the Panda launch and
   physical hand-guiding sequence established in `PROCEDURE.md`. During teaching,
   deactivate `joint_trajectory_controller` and confirm only broadcasters are
   active. The teacher does not activate controllers or issue robot commands.
2. Stop the old bridge/camera capture. Start the bridge for this session from the
   Panda project:

   ```bash
   cd /home/jan/install/pixi_panda_ros2
   pixi run -e jazzy python \
     /home/jan/dev/neura_demonstrator/tests/scene_calibration/robot_state_bridge.py \
     --output /home/jan/dev/neura_demonstrator/tests/scene_calibration/captures/automatic/robot_state.json
   ```

3. Open the three camera views from the calibration folder:

   ```bash
   cd /home/jan/dev/neura_demonstrator/tests/scene_calibration
   .venv/bin/python capture_scene.py --config scene_config_auto.json \
     --square-mm 35 --accept-requests
   ```

   Verify printed dimensions first. During teaching, use this as a preview; do
   not press Space/V in the camera window, which would save manual captures into
   the new session. All three cameras should detect the board at capture poses.

4. In another terminal, start the teacher:

   ```bash
   cd /home/jan/dev/neura_demonstrator/tests/scene_calibration
   .venv/bin/python teach_waypoints.py --config scene_config_auto.json
   ```

   Move the arm yourself using its supported hand-guiding controls, release them,
   release the external device for FCI observation, and let it settle. Type a
   command in the teacher terminal and press Enter:

   | Key | Records |
   | --- | --- |
   | T | Joint pose for a fitting capture |
   | V | Joint pose for a held-out validation capture |
   | M | Clearance/transit joint pose, without a picture |
   | Q | Finish teaching |

   The teacher requires fresh error-free states, low measured joint velocities,
   and one second of joint stability. It saves each waypoint immediately into
   `trajectories/calibration.json`. Restarting the teacher resumes that file.

   Choose a first capture pose with the board visible in the wrist view. You
   will review the approach from your current pose to that first waypoint before replay. Teach the path
   in the order you want it executed. Include clearance points between poses
   where necessary; endpoints alone cannot establish a collision-free transition.
   Transit points may repeat, allowing you to teach a return through familiar
   clearance poses. Replay starts from the measured current joint pose; the
   approach to the first waypoint must also have adequate clearance.
   Aim for about 20–30 fitting poses with multi-axis tilts, plus 5–6 validation
   poses. Plan how to get back through clearance points between these groups too.

## Phase B: review without motion

```bash
.venv/bin/python run_trajectory.py --config scene_config_auto.json
```

This default is **review-only**. It imports no ROS packages, creates no ROS node,
and sends no robot commands. It validates the seven joint names, finite values,
URDF limits with a margin, the robot-description hash, and conservative motion
settings. It prints every waypoint, capture split, and segment duration.

Review the JSON and every transition, ideally in your simulator. Motion is smooth
quintic interpolation in joint space. It is **not MoveIt collision planning**:
the robot can sweep through an obstacle even when both endpoints are clear.
Add transit points as needed. The camera mounting, board, tool mass, and physical
workspace must still correspond to the taught setup.

Default motion limits are 0.15 rad/s, 0.3 rad/s², and 1 rad/s³, with at least
six seconds per segment. Duration increases automatically for larger joint
changes. Slow trajectories do not remove the FCI 1 kHz timing requirement. Your
frequent scheduling overruns remain relevant to active control. Test a short
reviewed sequence first; do not assume that successful state reading proves a
full motion sequence reliable on the standard kernel.

## Phase C: execute a reviewed sequence

For an initial movement-only test, keep the state bridge running and omit camera
capture. Pass `--motion-only` to `run_trajectory.py`; without `--execute` this
still only prints a review. Adding `--execute` moves the robot, waits for settling,
and saves no images. `--max-waypoints 2` limits a first test to the first two poses.
All robot-state, joint-limit, controller, and cancellation checks still
apply. The trajectory controller must be active for this test.

Before starting, choose a current position with a clear joint-space approach to
the first taught waypoint. Release native guiding controls and the external activation
device. Keep the physical stop accessible and monitor the robot. Ensure the
bridge and camera capture remain running with the same automatic-session paths.

For automatic replay, activate the trajectory controller from the Panda project:

```bash
cd /home/jan/install/pixi_panda_ros2
pixi run -e jazzy ros2 control switch_controllers \
  --activate joint_trajectory_controller --strict
pixi run -e jazzy ros2 control list_controllers
```

Only that motion controller should be active; broadcasters remain active. Do not
use the native guiding buttons during automatic movement. Unlike manual capture,
this stage needs the motion controller active.

The following command **moves the real robot**. Run it only after reviewing the
transitions and testing the setup:

```bash
pixi run -e jazzy python \
  /home/jan/dev/neura_demonstrator/tests/scene_calibration/run_trajectory.py \
  --config /home/jan/dev/neura_demonstrator/tests/scene_calibration/scene_config_auto.json \
  --execute
```

The runner approaches the first waypoint from the measured current joints using
the same smooth interpolation and motion limits as later segments. There is no
0.03 rad start-position requirement. Older plans' `start_tolerance_rad` field is
ignored; existing waypoint files do not need to be re-taught. The approach is not
collision-checked. At each capture it requires the measured joints within 0.005 rad of the
target, measured velocities ≤0.002 rad/s, and at least two seconds of settling.
The camera recorder additionally checks its existing stationary interval around
the image receipts. All three board detections are required for each automatic
capture; a missing view stops the run rather than silently losing validation
coverage. It stores the waypoint name, plan hash, request ID, images, and measured
robot state with each sample.

Motion rejection/cancellation, timeouts, stale states, robot errors, camera
restart/heartbeat loss, or failed captures stop the sequence. The runner requests
cancellation if a trajectory is still active. Cancellation over ROS is not an
emergency stop; use the physical device if motion does not stop. There is no
automatic error recovery or continuation after failures.

On successful completion, the trajectory controller remains active holding the
final pose. Deactivate it again before native hand-guiding. For another complete
run, use a new session/config and matching bridge path; repeated captures in the
same session are rejected as duplicate poses.

## Diagnosing motion aborts

`--speed-scale 0.5` doubles every segment's movement duration and reduces the
velocity/acceleration/jerk limits consistently. It changes timing, not the taught
waypoints. Use `--max-waypoints 1 --motion-only` for an isolated approach test;
add `--execute` only when ready to move the robot.

The runner retains a 0.05 rad path-position tolerance and the plan's final-pose
tolerance (normally 0.005 rad). A path failure means a joint exceeded its allowed
error against the time-varying commanded reference, even if the robot looks
close to the endpoint. A goal-time failure means the endpoint tolerance was not
met within the three-second deadline. Neither is a Pixi lock-format issue.

Each executed run writes `captures/automatic/motion_feedback_TIMESTAMP.csv`
with named joint desired/actual/error positions, recorded at up to 20 Hz.
Failures print peak tracking errors and latest errors relative to both the
commanded trajectory and the final target. The controller's own log records
the exact violating cycle, which may fall between feedback samples.

Slowdown is a diagnostic step, not a guarantee. Persistent endpoint offsets can
require checking controller gains, load configuration, and timing. Do not remove
tracking checks just to label an aborted movement successful.

## Phase D: solve offline

```bash
cd /home/jan/dev/neura_demonstrator/tests/scene_calibration
.venv/bin/python solve_scene.py --session captures/automatic --output results/automatic
```

Automatic positioning improves repeatability but does not guarantee calibration
accuracy. The existing fitting and held-out validation checks still determine
whether an accepted scene can be exported.
## Optional tracking tolerances and timing

Capture failures are retried three times after the initial attempt by default.
Use `--retake-pause SECONDS` (default 2) for the retry delay, and optionally
`--capture-retries COUNT`. Replay remains at the waypoint until a capture is
saved, or stops after all attempts fail. Robot/state errors still stop
immediately. A timeout retains the pending request ID to avoid double-saving.

`--goal-time-tolerance-seconds 10` gives the controller ten additional seconds
after the trajectory's scheduled end to satisfy the goal tolerances (default:
three seconds). Fractional positive seconds are supported. This also extends
the runner's action-result timeout to match.

Alternatively, `--disable-goal-time-tolerance` explicitly erases the ROS goal
deadline and disables the runner's corresponding result timeout. These two
options are mutually exclusive. The runner still waits for action success,
checks fresh state and robot errors, and cancels on Ctrl+C. If the requested
goal tolerance can never be reached, it can wait indefinitely; it will not
advance simply because the trajectory's scheduled time has elapsed.
This allowance is separate from `--wait-seconds`, which applies only after
action success.

Replay accepts `--disable-path-tolerance` to erase ROS action path tracking
tolerances, and `--goal-tolerance-rad VALUE` to select any positive finite final
joint tolerance in radians, without an upper cap. For example, 0.02 rad is
1.15 degrees. These options must be supplied on each run; defaults remain
0.05 rad for path tracking and the taught plan's final tolerance.
The robot's own protection remains active, but does not guarantee accurate
waypoint tracking or collision-free transitions. Action failures still stop
the sequence. Automatic capture receives the same final tolerance
and records the measured pose, rather than substituting the requested pose.

`--speed-scale 2` halves movement durations, doubles the velocity limits,
quadruples acceleration limits, and multiplies jerk limits by eight. Values
above one are supported; the scaled velocity must still fit the robot's URDF
velocity limits. Increasing speed also affects the initial approach.

`--skip-settle --wait-seconds 2` replaces the runner's settling check with a
fixed two-second pause after a successful action result. The pause is
configurable and may be zero. The runner continues to check state freshness
and robot errors during the pause, but does not require continuous low joint
velocity or target agreement. It also skips its velocity check before the next
move and while awaiting a capture acknowledgement. The ROS action still needs
to succeed within the requested goal tolerance. The camera recorder retains
its independent acquisition stability checks, so a moving camera can still
cause a capture rejection. Without `--skip-settle`, normal settling checks apply.

