# Quest controller coordinate test

## Check the installed APK

Run `.venv/bin/python diagnose_headset.py` from this folder. It queries the
installed package version, hashes the installed APK on the headset, and compares
it with the local APKs. It does not install, uninstall, launch, or stop the app.
The report is saved to `headset_diagnostics.json` for inspection.

The bundled `teleop-pointer-frame-relative.apk` with SHA256
`a6e02ee027d90d2a794e971646600ceb2b5be6dde89aef3cb703d3e7b6fbb689`
contains compiled headset-relative pointer-pose calculation. Its native render
routine calls GetHeadPose, inverts its matrix, and multiplies it by the pointer
matrix. The C++ source currently checked into the repository uses world poses
and differs from this APK. No public relative/absolute switch was found in the
reader or app source. Neither `--frame ros` nor the viewer's zero-reference
control changes the APK's moving reference frame.

## Pygame orientation viewer

Install Pygame into this test's environment if it is not present:

```bash
cd ~/dev/neura_demonstrator/tests/quest_controller_tracking
uv pip install --python .venv/bin/python 'pygame>=2.6,<3'
.venv/bin/python visualize_controller.py --hand right
```

The viewer uses Pygame to draw a shaded cuboid with perspective projection and
rotating controller axes: X red, Y green, Z blue. Position remains centred to
isolate orientation. The fixed grid and tracking-frame axes provide a visual
reference; roll/pitch/yaw and trigger pressure are displayed above/below it.
The cuboid is a display shape, not an exact controller model.

Controller buttons: press `X` on the left controller to show its cuboid, or `A`
on the right to show the right controller. The left controller has X/Y buttons,
not A/B. Switching controllers clears the previous zero reference.
Keyboard controls: `Tab` switches left/right; `R` makes the current orientation the zero
reference; `A` restores tracking-frame orientation; `Esc` or window close exits.
`--frame openxr` selects native axes; default `ros` matches the console script.
`--ip` and `--reader-source` work as in the console script. A zero reference is
only a fixed rotation offset; it does not correct headset-relative tracking or
change which device the APK tracks. Disable hand tracking on the headset when
testing controllers only. The reader currently returns cached poses, so the
viewer cannot certify freshness or tracking validity from those poses alone.

To preview without the headset:

```bash
.venv/bin/python visualize_controller.py --demo
```

This does not connect to ADB or install/start the headset app. The display rate
defaults to 60 FPS; change it with `--fps`. Re-lock dependencies with `uv lock`
when internet access is available if you added Pygame without updating uv.lock.

Each printed controller pose also includes `trigger=VALUE`, the index-finger
trigger value reported by the reader: 0 is released and 1 is fully pressed.
This is printed by default, including when a controller pose is unavailable.

Prints left/right controller position in meters using
[Neuracore's Meta Quest reader](https://github.com/NeuracoreAI/meta_quest_teleop).
`--orientation` prints roll, pitch, and yaw in degrees, converted from the
reported rotation matrix using extrinsic XYZ: `R = Rz(yaw) Ry(pitch) Rx(roll)`.
Angles describe orientation in the selected output frame. Euler angles can
wrap at 180 degrees; near pitch ±90 degrees, roll/yaw become ambiguous and the
output marks gimbal lock. This conversion does not change the tracking origin.
No ROS, Franka launch,
robot-state bridge, or robot connection is required.

This test has its own uv `.venv`. The reader also imports SciPy, which is listed
explicitly here because upstream's setup.py does not declare it.

## Host setup (once)

Install ADB and Git LFS if missing:

```bash
sudo apt update
sudo apt install adb git-lfs
git lfs install
```

Git LFS is needed to obtain the actual APK instead of a small pointer file.
Install dependencies while the computer has internet access:

```bash
cd ~/dev/neura_demonstrator/tests/quest_controller_tracking
uv sync
```

The first successful sync creates uv.lock, locking the reader's resolved Git
commit and its dependencies. Subsequent installs can use `uv sync --locked`.
If the APK is missing or still a pointer, fetch a local copy with Git LFS and
install that copy into this environment:

```bash
git clone https://github.com/NeuracoreAI/meta_quest_teleop.git vendor/meta_quest_teleop
git -C vendor/meta_quest_teleop lfs pull
.venv/bin/python track_controllers.py --hand both --orientation
```

The script automatically prefers this local reader checkout. No `uv pip install`
or setuptools build is needed for the fallback; the existing environment still
supplies NumPy, SciPy, and pure-python-adb. Use `.venv/bin/python` directly after
cloning. `--reader-source PATH` selects a different checkout. The test checks
the APK before creating the reader, to prevent installation of a Git LFS pointer.

## Headset setup

Enable Quest developer mode, connect the headset through a USB data cable,
and accept the USB debugging prompt inside the headset. Follow the upstream
README for developer-mode setup. Keep the headset awake and its app active;
ensure the controllers are paired and tracked. The first tracking run installs
and launches the reader's APK if it is absent. `--list` does neither.

```bash
cd ~/dev/neura_demonstrator/tests/quest_controller_tracking
.venv/bin/python track_controllers.py --list
.venv/bin/python track_controllers.py
```

Only right controller, with orientation:

```bash
.venv/bin/python track_controllers.py --hand right --orientation --rate 5
```

Example console output:

```text
… | right: x=+0.1000 m  y=-0.2000 m  z=+0.3000 m
```

Default axes are ROS convention (x forward, y left, z up); select `--frame
openxr` for native axes (x right, y up, z backward). These remain relative to
the Quest tracking reference, not an aligned Panda base. This is a coordinate
print test, not a calibrated tracking accuracy or latency measurement.
For USB troubleshooting run `adb devices -l`: `unauthorized` means the headset
debugging prompt still needs approval; an empty list means no device is found.

Optional `--duration 10` stops after ten seconds. Wireless `--ip ADDRESS`
requires wireless ADB setup first, as documented upstream. Start with USB.
Ctrl+C stops reading. The script does not force-stop the headset app.
