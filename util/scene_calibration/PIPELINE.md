# Capture, fit, validate, and export

The pipeline uses the existing camera recorder, ROS waypoint runner, and scene
solver. Calibration fitting estimates rigid transforms; it does not train a
robot-learning neural network.
Camera capture and solving run in the camera uv environment with inherited
Python path/home settings and user site packages disabled, so Pixi's OpenCV
cannot override the GUI-enabled version. The bridge and runner retain ROS's
Python environment.

Keep the Panda launch running, with FCI active and the joint trajectory
controller active. Stop any old camera preview and robot state bridge; the
pipeline starts and stops its own copies. Keep the board, scene cameras, and
wrist mount fixed. The starting robot pose does not need board visibility:
the pipeline moves to the first taught waypoint before checking board detection.
All three cameras must supply fresh images. Scene-camera board detection is
optional at each pose, so arm occlusion does not prevent a wrist capture. Images
from all three cameras are saved; only accepted detections enter calibration.

```bash
cd ~/install/pixi_panda_ros2
pixi run -e jazzy franka robot_ip:=172.16.0.2
```

In another terminal:

```bash
cd ~/install/pixi_panda_ros2
pixi run -e jazzy python \
  ~/dev/neura_demonstrator/util/scene_calibration/run_pipeline.py \
  --config ~/dev/neura_demonstrator/util/scene_calibration/scene_config_auto.json \
  --session ~/dev/neura_demonstrator/util/scene_calibration/captures/pipeline_take1 \
  --output ~/dev/neura_demonstrator/util/scene_calibration/results/pipeline_take1 \
  --square-mm 35 \
  --speed-scale 8 \
  --disable-path-tolerance \
  --goal-tolerance-rad 0.03 \
  --goal-time-tolerance-seconds 10 \
  --skip-settle --wait-seconds 2.5 \
  --execute
```

Use your tested motion parameters; alternatively specify
`--disable-goal-time-tolerance` instead of a numeric goal time allowance.
Omitting `--execute` reviews the trajectory without starting hardware processes
or creating output folders. `--motion-only` starts only the state bridge and
runner, then skips cameras and fitting. The pipeline accepts all runner flags,
including `--path`, `--max-waypoints`, and `--action`.

Capture and results paths default to matching timestamped folders under
`captures/` and `results/`. Explicit relative paths resolve from the terminal's
current working directory. Capture folders must be new; results folders must be
new or empty. At least 10 fitting and 3 held-out validation poses are required.
Paths are not collision checked; transit poses are replayed without capturing.
The camera recorder retains its acquisition stability checks even with
`--skip-settle`. A fixed pause does not override those checks.

Startup waits up to 120 seconds for fresh robot state, robot description, fresh
camera frames. Board detection is not required at startup; wrist detection is
required when capturing at a waypoint. Override the startup timeout with
`--startup-timeout-seconds`. Camera startup messages and errors go to `camera.log`
in the capture folder; state bridge messages go to `bridge.log`. The preview
window remains available. Let the pipeline save images; avoid pressing Space/V
for extra manual captures during a run.

Failed capture acknowledgements trigger three retries after the initial attempt
(four attempts total). Set `--retake-pause 2` for the delay between attempts,
independently of the post-movement `--wait-seconds` pause. Optionally override
the retry count with `--capture-retries`. The robot stays at the same waypoint
and advances after successful saving or exhausted retries. Exhausted retries
skip that capture with a warning and continue replay. Missing captures are listed
in pipeline_status.json under capture_summary and do not block the solver.
Duplicate, unknown, or incorrectly labelled captures are still rejected. An acknowledgement timeout reuses the pending
request ID to avoid duplicate captures if a success arrives late. Robot errors,
stale state, a stopped/restarted recorder, and unexpected robot motion still
stop immediately rather than being treated as a rejected camera observation.

Each run stores `pipeline_config.json`, `trajectory.json`, and
`pipeline_status.json` alongside images, robot states, and motion feedback.
After replay finishes, cameras/bridge stop and
`solve_scene.py` prints fitting and held-out evaluation results. A computed
calibration always exports `scene.json`, `panda.urdf`, `calibration_report.json`,
and `calibration_summary.txt`, even if error limits are exceeded. Failed
validation leaves `calibration.passed=false`, `status=validation_failed`, and
warnings in the scene file. The pipeline records
`stage=complete_with_validation_warnings` and `validation_passed=false`.
Export completion returns exit code zero; inspect the validation flag to decide
whether the estimate meets your accuracy requirements. Missing or unusable
input still prevents solving and no transforms are invented.
Failed motion or a recorder/state failure preserves partial data and
does not run the solver. Individual rejected captures do not prevent solving;
the solver checks whether enough usable observations remain for each camera. Ctrl+C requests runner cancellation before stopping
the bridge and cameras. The existing Panda launch remains running.

To investigate or refit a saved dataset without moving the robot:

```bash
cd ~/dev/neura_demonstrator/util/scene_calibration
.venv/bin/python solve_scene.py --session captures/pipeline_take1 --output results/refit_take1
```
