# Demonstration collection

run_collection.py adds an episode recorder and PySide6 desktop GUI to the existing
joint teleoperation backend. It accepts the same teleoperation parameters.
Preview is the default; --execute enables arm and gripper commands.
The new pipeline has offline tests, but has not been validated on the physical robot.

## Dependencies

Use the same ROS/Pink environment as teleoperation. In the current Jazzy
environment, PySide6 and pyrealsense2 are missing; OpenCV is already present:

~~~bash
uv pip install --python ~/install/pixi_panda_ros2/.pixi/envs/jazzy/bin/python \
  PySide6 pyrealsense2
~~~

requirements-collection.txt lists the added dependencies for a fresh setup.
The reader needs RealSense device access/udev rules already used by calibration.
No native Quest app, separate robot bridge, camera recorder, or separate relay is needed.

## Teach a reset sequence

This script only reads the published robot state. Position the robot with your
normal hand-guiding setup, with teleoperation stopped. Record intermediate clearance
poses first, then the desired episode start pose last:

~~~bash
cd ~/install/pixi_panda_ros2
pixi run -e jazzy python \
  ~/dev/neura_demonstrator/util/demonstration_collection/teach_reset.py \
  --output ~/dev/neura_demonstrator/util/demonstration_collection/reset_sequences/default.json
~~~

Press Enter at each stationary pose. u removes the last pose; q exits.
Every edit is saved. An existing file is never overwritten; use a new filename
to teach a revised path. The script requires fresh, stationary robot state and
never switches controllers or enables hand guiding for you.

A reset sequence may contain **one, two, or more poses**. Every pose contains
seven joint angles in q, one for each Panda joint; this is not a seven-pose
requirement. The joints header may be omitted to use standard Panda joint order.

Reset execution opens the fingers first and visits every saved joint pose.
Joint interpolation between poses is **not collision checked**. Test the full
reset path from representative task-end configurations. Reset speed has its own
--reset-speed parameter (default 0.25 rad/s); teleoperation speed overrides do
not accelerate resets. Reset goals retain finite position and time tolerances.

Add `--disable-reset-path-tolerance` (alias `--disable-path-tolerance`) to
`run_collection.py` to disable the ROS trajectory action's path tolerances
for reset movements. By default the reset position path tolerance is 0.15 rad.
The option applies to both the initial reset and resets after episodes; it is
saved in session settings. Final goal tolerance (0.03 rad), goal time allowance
(5 seconds), reset timeout, robot fault checks, and teleoperation tracking checks
remain active. It does not change force/collision thresholds or replay settings.

## Start a session

The GUI's optional **Session name** field chooses the recording folder name
under the output directory (also available as `--session-name NAME`). Names are
independent of the task prompts; spaces and Unicode are supported, leading and
trailing whitespace is trimmed, and path separators are rejected. Reusing a name
creates `_002`, `_003`, etc., without overwriting recordings. Blank names retain
the timestamp naming scheme. Replay discovers both named and timestamped sessions.

Terminal 1, your normal driver (joint trajectory controller active):

~~~bash
cd ~/install/pixi_panda_ros2
pixi run -e jazzy franka robot_ip:=172.16.0.2
~~~

Terminal 2, initial preview:

~~~bash
cd ~/install/pixi_panda_ros2
pixi run -e jazzy python \
  ~/dev/neura_demonstrator/util/demonstration_collection/run_collection.py \
  --backend joint --scale 1.0 --frame-alignment world --yaw-deg 180 \
  --workspace-radius 1.0 \
  --reset-sequence ~/dev/neura_demonstrator/util/demonstration_collection/reset_sequences/default.json \
  --scene ~/dev/neura_demonstrator/util/scene_calibration/results/pipeline_take3_exported/scene.json \
  --output ~/dev/neura_demonstrator/datasets/demonstrations
~~~

Use your desired teleoperation parameters on this command, as before. Add
--execute for live collection. Preview does not move or reset the robot and
records preview=true in session metadata; exclude preview sessions from training.
Preview still requires robot state and cameras.

The GUI automatically calls `setup_quest.sh` on startup with the selected `--port`
(default 8443). This sets up ADB USB forwarding and requests the Quest proximity
override, so the headset behaves as worn. A disconnected or unauthorized headset
produces a terminal warning without preventing the GUI from opening.

At GUI startup, a background thread also hardware-resets the three cameras in
`--camera-config`. Close other camera programs before launching collection.
The Quest server and task fields remain usable during the reset. **Prepare
session** becomes available after a ten-second reboot pause and reconnection
of all configured serials (up to thirty additional seconds). A reset failure is
shown in the GUI and terminal, with a **Retry camera reset** button. Streaming
is still checked during session preparation. Cameras are reset once per program
launch, not between sessions or episodes.

After reconnecting the headset, retry manually (use your selected port):

~~~bash
bash ~/dev/neura_demonstrator/util/demonstration_collection/setup_quest.sh 8443
~~~

The override is not automatically restored when the GUI closes. To restore normal
proximity behavior afterward:

~~~bash
adb shell am broadcast -a com.oculus.vrpowermanager.automation_disable
~~~

The Quest relay starts immediately when the GUI opens, before preparing a session
and without requiring robot or camera initialization. Open http://localhost:8443/
in the Quest browser and start controller tracking. It stays running between
sessions until the GUI closes; changing the prompt does not require reconnecting
the Quest. In the GUI, enter the task prompt, select a reset JSON and click
**Prepare session** to initialize the robot connection and cameras. The browser client should say
20261014-recording; older pages lack the timestamps required to start an episode.
Stop other viewers/teleoperators using port 8443 first.

The camera serial mapping defaults to util/scene_calibration/scene_config_auto.json.
Override it with --camera-config. The configuration needs a cameras object with
distinct wrist, side, and opposite serials. Scene calibration is optional
(--scene embeds its JSON); camera intrinsics and depth-to-color extrinsics are
always obtained from the active devices. ChArUco detection is never required.

## Operator workflow

1. **A:** open gripper and execute the selected reset path.
2. **Right thumbstick:** establish the teleoperation reference frame.
3. **A:** start an episode. Release and re-engage grip before moving.
4. Grip remains clutch; trigger clicks close/open; hold B for precision.
5. **A:** end successfully. **Left X:** end unsuccessfully. **Left Y:** interrupted.
6. Input is gated immediately, even with grip held. The bounded stop completes,
   a minimum 0.5-second observation tail is saved, and pending gripper movement
   finishes. After recording flushes, the fingers open and the reset path runs.
7. Reset ends in the reference-waiting state: thumbstick, then A starts another
   episode under the same prompt.

Button actions use rising edges and require release after reconnect. A/X/Y during
reset are ignored, not queued. GUI buttons offer the same events. B is never reset.
Preparing a session does not itself move the arm. Close the session to change
prompt/reset sequence and prepare a new session.

Clutch-released intervals are fully recorded, including measured robot state.
clutch_pressed is the raw thresholded input; engaged identifies actual permitted
teleoperation. The latter is preferable for training segment selection.

Camera/recording failure, robot fault, or stale robot/gripper state interrupts the
episode and ends the session **without automatic reset or error recovery**.
Temporary right-controller tracking loss or stale Quest frames instead pause motion
like a clutch release while the same episode continues recording. On return, release
and re-engage grip to resume. Quest clock-sync loss also pauses motion until timing
recovers. Reconnection/recentering keeps the recording but requires a fresh thumbstick
reference before resuming. Tracking pause/restore/reference-change events are saved;
`control.jsonl` retains raw `clutch_pressed`, `tracking_available`, `input_allowed`,
and effective `engaged` separately. Use `engaged` for clutch-active exports. The GUI
A/X/Y controls can still finish an episode when Quest tracking is unavailable. GUI close/Ctrl+C also stops control and flushes recording.
Close the window again after shutdown completes. Ordinary scene occlusion is fine.

## Files and timing

Each session gets a unique folder, never overwriting earlier recordings:

~~~text
session_<UTC>/
  session.json               prompt, settings, reset poses, controller parameters,
                             code hashes/revision, clock anchor, optional scene
  robot.urdf                 live robot description
  reset_*.jsonl              reset commands/results, outside demonstration episodes
  episode_000001/
    episode.json             outcome, start/end/stop-request times, camera calibration, counts
    quest.jsonl              every accepted browser frame and receipt time
    robot_state.jsonl        complete received FrankaState (including q/dq, torques,
                             desired states, wrench estimates, payload, modes/errors)
    robot_pose.jsonl         received TCP pose messages
    gripper_state.jsonl      received gripper JointState messages
    control.jsonl            mapped target, clutch/engaged/scale, tracking error
    sent_command.jsonl       actual published joint trajectories after IK/limiting
    gripper_command.jsonl    issued Move/Grasp goals
    gripper_result.jsonl     action results
    clock_samples.jsonl      host monotonic, Unix and ROS clock pairs
    events.jsonl             episode boundary events
    camera_<role>.jsonl      per-stream timestamps/frame IDs, gaps, image paths
    cameras/<role>/          lossless RGB/depth files
~~~

Streams without samples may have no file. Robot observations are saved at the
subscription's received rate, not duplicated to an invented 1 kHz. ROS
sensor-data QoS does not guarantee that every hardware sample reaches the process.
The complete original ROS message is retained, including its header and robot time.
received_ns is host monotonic time at callback entry (before serialization), not
claimed to be the device acquisition time. Different streams are **not** forced into
frame-index alignment.

Quest frames contain browser sample time, WebXR animation-frame time, performance
time origin, sequence/epoch, button/axis values, and a recurring clock-offset
estimate. Clock exchanges retain round-trip duration and sample age. Approximate
host frame time is frame_time_ms * 1e6 + host_offset_ns. WebXR frame times can
refer to predicted tracking poses, not raw IMU capture times. Neither this estimate
nor half the round-trip duration guarantees acquisition-time accuracy.

RealSense RGB and depth timestamps/domains are stored independently, with frame
numbers and available sensor/exposure metadata. estimated_host_ns uses the
minimum observed receipt-minus-device offset. It includes unknown transport delay;
uncertainty_ns=null explicitly means unmeasured. This initial mapping is not
hardware synchronization or a calibrated drift-compensation solution. Raw timing
and periodic ROS/host clock pairs allow better offline alignment later.
Camera clock regressions stop recording rather than silently joining incompatible time.

Raw streams distinguish user intention, commands actually sent, and physical
response. Training conversion can then choose command labels and measure latency.
Do not shift all commands to measured execution time or blindly concatenate
clutch-active intervals. A LeRobot/training resampler is intentionally separate.

## Storage and quality

Default --image-format npy saves exact uint8 **BGR** arrays and native uint16
depth arrays (allow_pickle=False). Read with numpy.load; convert BGR to RGB
for training. Depth is not resampled/aligned; use saved intrinsics/extrinsics and
multiply values by depth_scale_m for metres.

Optional --image-format png uses lossless PNG (16-bit depth preserved).
It costs substantially more encoding CPU; measure whether it can keep up.
No lossy RGB compression or depth conversion is performed in either mode.

Defaults explicitly request RGB + depth at 1280×720, 30 FPS on all three cameras.
Unsupported combinations fail visibly; there is no silent resolution fallback.
Use --camera-width, --camera-height, --camera-fps to change both streams.
RGB-D acquisition increases USB bandwidth over the former RGB-only preview.
At defaults, raw arrays require about 415 MB/s, or 1.49 TB/hour plus overhead.
Actual usable disk space and write throughput must be checked on the recording disk.

Acquisition, disk writing, and the GUI run independently of the ROS loop. Queues
are bounded; overflow, disk errors, less than 2 GiB free, or camera timeout stop
collection and mark the episode interrupted. Frame-counter gaps remain visible
in camera metadata. Successfully finalized episodes have counts and outcome;
unfinished/crashed episodes remain identifiable rather than masquerading as success.
Keep failed/interrupted recordings; filter them during dataset export.

## Export clutch-active windows

This creates an index of separate time windows without modifying the originals:

~~~bash
pixi run -e jazzy python \
  ~/dev/neura_demonstrator/util/demonstration_collection/export_clutch_segments.py \
  /path/to/session_<UTC> --output /path/to/clutch_segments.json
~~~

The index preserves episode outcomes and host timestamps. It does not yet convert
to LeRobot or stitch pauses away.

## Offline verification

~~~bash
/home/jan/install/pixi_panda_ros2/.pixi/envs/jazzy/bin/python \
  -m unittest discover \
  -s ~/dev/neura_demonstrator/util/demonstration_collection -p 'test_*.py'
~~~

Tests cover control transitions, held buttons, reset goal serialization/bounds,
fault handling, lossless arrays, incomplete recording labels and clutch windows.
Hardware acquisition, GUI rendering, disk throughput and complete robot reset
paths still require validation with installed dependencies and attached devices.

## Replay a recorded episode

In the collection GUI choose **Demonstrations → Replay recordings…**. Close any
active collection session first; its Quest server stays available. The replay
window lists episodes by prompt, session, outcome and duration. Select one and
click **Start playback** to view the three recorded cameras. Drag the timeline
to seek, choose a slower playback speed, or select **Show depth**. Use the visible
−/+ buttons or type a speed from 0.10 to 1.00 and press Enter. Speed remains
editable during camera playback without jumping the timeline; it is locked during
live robot replay. Enter in this field never starts motion. Playback reads
images lazily on a background thread and never alters the dataset. Camera-only
playback does not require the Panda driver, cameras, or headset.

You can also open the replay window directly:

~~~bash
cd ~/install/pixi_panda_ros2
pixi run -e jazzy python \
  ~/dev/neura_demonstrator/util/demonstration_collection/replay_demonstration.py \
  --root ~/dev/neura_demonstrator/datasets/demonstrations
~~~

Optional `--episode /path/to/session_.../episode_000001` selects an episode at
startup. Add **--execute** to unlock the **Replay on the live robot** checkbox.
The same checkbox is available from the collection GUI if it was launched with
--execute. It is always unchecked initially. Start the usual Panda driver with
joint_trajectory_controller active, stop other teleoperation/trajectory publishers,
arrange the objects and clear the recorded reset/replay path, then select the
checkbox and click **Start live replay**. This sends robot commands:

1. Validate the completed recording, the live robot model and controller state.
2. Open the gripper and follow the reset sequence embedded in that session.
3. Make a short, bounded alignment to the recorded initial joints (only if the
   saved reset endpoint is within 0.05 rad), then verify actual error <=0.03 rad.
4. Restore the recorded initial gripper width.
5. Replay the measured joint motion, with recorded Move/Grasp commands at their
   recorded offsets, while showing the corresponding recorded cameras.
6. Hold the final pose. There is no automatic reset or restart at replay end.

**Stop** cancels arm and gripper actions. Closing the window or pressing Escape
while live replay is running requests cancellation and keeps the window open until
the worker has finished. Use the physical stop if action cancellation does not
stop motion. A robot fault, stale state, gripper failure, controller change or
competing trajectory publisher aborts replay; no automatic fault recovery occurs.

Live replay reconstructs **measured q**, not overlapping teleoperation commands,
with a rest-to-rest quintic spline. Position extrema are checked against the live
URDF limits. The entire timeline is slowed as necessary to keep joint speed below
min(URDF limit, 1 rad/s), acceleration below 3 rad/s², and jerk below 30 rad/s³.
The selected speed (0.1–1×) is an upper bound; actual speed is shown during replay.
Camera playback and gripper-event offsets use the same adjusted timeline. The
recorded gripper Move/Grasp speed remains unchanged. If a gripper action has not
finished before the next event, replay stops rather than queueing it late.
Preview, interrupted, incomplete, invalid or discontinuous robot recordings cannot
be replayed live; their available camera data can still be inspected.

The camera timeline uses recorded host receipt times. In live mode it follows the
scheduled arm trajectory, not measured progress or new camera images. It does not
claim hardware synchronization, reproduce contact forces, guarantee identical
object interactions, or collision-check the workspace. The complete live sequence
still requires supervised hardware validation; development tests do not move the
robot.


## Precision control in recordings

Holding right B applies `--precision-scale` (default 0.5) to translation and
rotational increments, and to the angular-speed cap. The recorded raw Quest poses
and buttons remain unchanged. `control.jsonl` stores `precision`, `scale`
(translation), `rotation_scale`, `angular_speed_limit_rad_s`, and the mapped target
after scaling/filtering/limiting. `sent_command.jsonl` stores the actual joint
trajectories sent after IK and joint limiting. `robot_state.jsonl` and
`robot_pose.jsonl` store measured motion without rescaling. No samples are divided
by ten or removed during precision mode. Older episodes retain their original
translation-only precision behavior and do not gain these new fields retroactively.


## Multiple task descriptions

Enter equivalent descriptions separated by semicolons in the task field, e.g.
`Place the gear on the plate; Move the gear onto the plate; Put the gear on the plate`.
Leading/trailing whitespace is removed from each description and empty entries
are ignored. At least one nonempty description is required. The ordered list is
saved as `prompts` in both `session.json` and every `episode.json`; the first
description remains in `prompt` for existing readers. All descriptions label the
same episode; recordings are not duplicated. A later training exporter can sample
a description from the list. Existing recordings are unchanged.

Replay provides **Reset playback to 0**, which pauses and rewinds the selected
episode, displaying its first available camera frames. It does not reset/move the
robot and is disabled while live replay runs. The speed readout shows the requested
video speed and, during live replay, the actual retimed speed.
