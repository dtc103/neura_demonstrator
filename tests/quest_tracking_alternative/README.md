# Quest tracking alternative: WebXR + Python

## Native app: launch tracking entirely from the PC

Native mode is now the default. App source is in `android/` and uses package
`org.neura.questtracking`, leaving the old Neuracore app installed. It starts
Touch controller pose/trigger/button streaming automatically when launched.
There is no browser and no in-app Start button. Reference space is STAGE when
available, otherwise LOCAL; neither follows the headset. Bare-hand interaction
profiles/extensions are not enabled. Metadata signals reference changes.

**Build status:** Python protocol/launcher tests pass. The Android SDK/NDK are
not installed here; no APK has been compiled or physically tested yet. This
source is an implementation to build and verify, not a certified working APK.

One-time setup/build (internet needed; review SDK license prompts):

```bash
cd ~/dev/neura_demonstrator/tests/quest_tracking_alternative
.venv/bin/python build_native_app.py --setup
```

Bootstrap downloads have a 120-second read/connect timeout and two retries.
SDK tooling and Gradle downloads display a progress bar, transferred bytes,
speed, and estimated remaining time when a server provides the total size.
Unknown-size responses show a byte counter and speed. Connection/retry messages
are printed before data arrives. No additional progress-bar dependency is needed.
For a slow connection, use `--download-timeout 300 --download-retries 3`.
These options cover the Python downloader for SDK command-line tools and
Gradle; sdkmanager and Gradle handle their own subsequent network requests.

The helper installs SDK tooling locally in `.android-sdk`, downloads a verified
Gradle 8.9 distribution, and builds with API 34, NDK 27.2.12479018, Android CMake
3.22.1, AGP 8.7.3, and OpenXR loader 1.1.53. Java 17+ is required (Java 21 is
already installed on this PC). Existing tooling can be selected with `--sdk`
and `--gradle`. Check without downloading: `build_native_app.py --check`.

Once the build succeeds, install and launch in one command:

```bash
.venv/bin/python visualize_controller.py \
  --install-apk android/app/build/outputs/apk/debug/app-debug.apk
```

For subsequent runs:

```bash
.venv/bin/python visualize_controller.py
# or console output:
.venv/bin/python track_controllers.py --hand both --orientation
```

The Python launcher checks the app is installed, applies
`com.oculus.vrpowermanager.prox_close`, forwards USB TCP, and launches the app.
On Ctrl+C/window close, it stops this app, removes its forwarding, and restores
normal wear detection with `com.oculus.vrpowermanager.automation_disable`.
`--no-keep-awake` disables the override; `--serial SERIAL` selects a headset.
First-time permissions/system dialogs can still require headset interaction.
Off-head tracking and the sensor override must be verified on your firmware.
Keep the tracking cameras facing the controller workspace. The stream clears
missing/untracked data rather than reusing the last known controller pose.

If startup receives no frames within 30 seconds, it stops and restores the
sensor. Inspect `adb logcat -s NeuraQuest` for native runtime errors. If USB
disconnects during cleanup, restore the sensor manually after reconnecting:

```bash
adb shell am broadcast -a com.oculus.vrpowermanager.automation_disable
```

The app exports data only. It does not connect to or command the Panda.

## Browser backend (retained)

Use `--backend webxr` on either Python script for the original browser workflow
below. Native mode does not use port 8443 or the browser page.

A separate test with its own uv environment and Pygame viewer. It uses the
WebXR `local-floor` approach discussed for
[vr-teleop-kit](https://github.com/Dream-Machines-Robotics/vr-teleop-kit), with a
small standalone relay instead of installing that repository's robot/IK stack.
The browser acquires poses; processing and visualization run in Python.
This is not a wrapper around vr-teleop-kit or the old Neuracore APK.

## What it preserves

- Left/right position in meters, roll/pitch/yaw in degrees, and trigger value.
- The same shaded Pygame cuboid and controller axes.
- Left controller X selects left; right A selects right.
- Keyboard R zeros orientation, A restores tracking-frame orientation, Tab
  switches controllers, and Escape closes the window.
- A synthetic `--demo` preview that never accesses the headset or ADB.

It requests **local-floor**, never viewer/head space. The browser reads physical
controller `gripSpace` poses and explicitly excludes bare hands. It does not
disable hand tracking in the system settings: if the headset switches to hands,
the test shows missing controller data rather than substituting palm poses.
Tracking is cleared after 0.5 seconds without packets. Recentring/reference
resets clear the viewer's zero reference. No robot commands are sent.

## Run over USB

Close the old headset teleop app and return to the Quest browser. You do not
need to uninstall the old APK. Keep USB debugging authorized.

Start one Python script (console or viewer, not both on the same port):

```bash
cd ~/dev/neura_demonstrator/tests/quest_tracking_alternative
.venv/bin/python visualize_controller.py --backend webxr --hand right
```

In another PC terminal, forward the local page through USB:

```bash
adb reverse tcp:8443 tcp:8443
```

In the **Quest browser**, open **http://localhost:8443/**, press **Start controller
tracking**, and allow the VR session. Keep the headset awake and controllers
tracked. The desktop viewer will update. The browser requires starting the
session by a user action; Python cannot start it remotely.

For console output instead:

```bash
.venv/bin/python track_controllers.py --backend webxr --hand both --orientation --rate 5
```

Close the viewer first, then restart tracking from the page. For an offline
rendering preview: `.venv/bin/python visualize_controller.py --demo`.

Default axis convention is `ros`; `--frame openxr` selects native WebXR axes.
Both use the same local-floor reference. `--host` and `--port` customize the
relay; for USB keep the default 127.0.0.1 binding. Change both ports in `adb
reverse` and the browser URL if you change `--port`. Remote Wi-Fi HTTP is not
a secure WebXR context; this implementation documents the USB route only.

## Environment

The environment is prepared with the already-installed NumPy, SciPy, and Pygame
packages copied into this test's separate `.venv`; it has no dependency on the
old reader. Network restrictions prevented generating a fresh uv.lock here.
To resolve/lock and recreate its dependencies with internet access:

```bash
cd ~/dev/neura_demonstrator/tests/quest_tracking_alternative
uv sync
```

Python 3.12 is selected. No new APK, Android build tools, ROS, or robot bridge
is required. The project currently resolves for Linux x86_64, matching this PC.

## Verify the reference frame

Leave a controller on the table, move/rotate your headset, and check that the
cuboid's orientation stays approximately fixed. Rotate the controller and check
that the cuboid follows it. The room reference can change on recentering or
tracking recovery, and it has not been calibrated to the Panda base. Minor
tracking drift is possible. This path still needs a live Quest browser check;
offline geometry/protocol tests do not verify physical tracking.
