# Larger ChArUco target for take3

Print `charuco_a4.svg` on **A4 portrait at 100% / actual size**. Disable automatic
scaling. Keep the white margin and mount the sheet flat on a rigid backing.

- Pattern: 5 × 7 squares, same marker dictionary/IDs as the original.
- Actual square pitch: **35 mm**.
- Patterned rectangle: **175 mm wide × 245 mm high**.
- Marker side: 24.5 mm.
- Scale increase: 40% in each direction; 96% greater area than the 25 mm board.

Check printed dimensions horizontally and vertically before collecting data.
The old board and definition remain in `../board/` and are used by earlier
sessions. Do not mix new-board observations into those sessions.

First check the three camera views with the larger board. Increase wrist viewing
distance if needed so enough corners remain visible during tilted views. Partial
views are allowed, but capture requires at least eight spread-out corners.
Larger markers may improve detection; they do not establish that calibration
will pass, and do not address robot/image timing or a moving camera mount.

Use `../scene_config_take3.json`, which selects this definition and
`captures/take3`. After verifying the print:

```bash
cd ~/dev/neura_demonstrator/util/scene_calibration
.venv/bin/python capture_scene.py --config scene_config_take3.json --square-mm 35
```

Start the robot bridge from the Panda Pixi project using the matching output:

```bash
pixi run -e jazzy python \
  /home/jan/dev/neura_demonstrator/util/scene_calibration/robot_state_bridge.py \
  --output /home/jan/dev/neura_demonstrator/util/scene_calibration/captures/take3/robot_state.json
```

After collecting fitting and held-out validation captures, solve with:

```bash
.venv/bin/python solve_scene.py --session captures/take3 --output results/take3
```
