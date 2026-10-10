# Quest 3 controller visualization

A 12 cm box follows the controller's tracked **position and rotation**. The Quest
browser obtains the controller's fused pose using WebXR `gripSpace`; a local
Python server relays it to a desktop browser at up to 30 updates per second.
The headset also renders boxes at both controllers' actual poses in VR.
All assets are local. No Unity application, JavaScript packages, CDN, or Python
dependencies are needed.

You need the **Quest 3 headset and a controller paired to it**. A controller by
itself connected to the PC does not supply this headset-tracked pose. Keep the
headset awake and the WebXR session active while using the demo.

## Environment and server

This test has its own `pyproject.toml`, `uv.lock`, and `.venv`, independent of
`tests/realsense_streaming`.

```bash
cd tests/quest3_visualization
uv sync
uv run server.py
```

Open **http://localhost:8000/** on the PC. Check **Simulation** to preview
translation and rotation without hardware. Simulation is explicitly labelled
and overrides incoming tracking only in that viewer.

## Connect the headset over USB

1. Enable Developer Mode on the Quest and install Android platform-tools (`adb`)
   on your PC. Connect the headset with a data-capable USB cable, put it on, and
   accept the USB debugging prompt.
2. In another terminal, check the connection and forward the server port:

   ```bash
   adb devices
   adb reverse tcp:8000 tcp:8000
   ```

   The device must show `device`, rather than `unauthorized`. If multiple devices
   are connected, use `adb -s QUEST_SERIAL reverse tcp:8000 tcp:8000`.
3. In **Meta Quest Browser**, open **http://localhost:8000/?source=1**. This is
   the headset's localhost; `adb reverse` routes it to the PC server.
4. Click **Start Quest tracking**, accept the immersive session permission, and
   hold the controller inside the headset's tracking area. Switch to controller
   input if the Quest is using hand tracking.
5. On the PC, turn off Simulation, select **Right** or **Left**, and move the
   controller. Click **Recenter box** in a comfortable starting pose.

The box centers itself on the first valid pose. Recenter makes the current
position and orientation the new zero. Translation remains in the Quest's fixed
world axes: X right, Y up, and −Z forward, with one unit equal to one meter.
It does not use a head-relative space, so moving your head alone does not move
a stationary controller's box. The axes are established by the headset's
reference space and can change when the Quest resets its origin; a reference
reset automatically recenters the desktop view.

The desktop camera is initially angled so depth motion is visible. Drag to
orbit and scroll to zoom. The controller axes rotate with the box. The ground
grid spacing is 0.25 m. After more than one second without fresh poses, or when
the selected controller loses tracking, the box disappears and the status
indicates missing tracking. An emulated position is labelled as an estimate.

Leave the immersive session using the Quest system UI to stop tracking. Close
the server with Ctrl+C. Port forwarding can be removed with:

```bash
adb reverse --remove tcp:8000
```

## Wi-Fi instead of USB

Use a certificate trusted by the Quest browser and valid for your PC's LAN
hostname/IP. Start the server listening on your network:

```bash
uv run server.py --host 0.0.0.0 --cert certificate.pem --key private-key.pem
```

Open `https://YOUR_PC_ADDRESS:8000/?source=1` on the Quest and
`https://YOUR_PC_ADDRESS:8000/` on the PC. Both devices need network connectivity
to this port. A plain HTTP LAN IP is not a secure WebXR origin. A self-signed
certificate warning bypass is not a reliable substitute for a trusted
certificate; the USB localhost setup avoids certificate configuration.
Use the LAN listener on a trusted demo network: the relay has no authentication.

## Checks

```bash
uv run python -m unittest -v
```

Tests cover invalid samples, quaternion normalization, sender ownership, origin
resets, stale poses, HTTP routes, and event stream delivery. Live socket tests
skip with an explicit reason if the execution sandbox prohibits sockets.

With the server running, open **http://localhost:8000/static/test_math.html**
to check the browser's quaternion, matrix, translation, and recenter math.
Physical headset tracking needs to be checked on the actual Quest.

API references: [WebXR controller grip space](https://developer.mozilla.org/en-US/docs/Web/API/XRInputSource/gripSpace),
[WebXR session and secure-origin setup](https://developer.mozilla.org/en-US/docs/Web/API/WebXR_Device_API/Startup_and_shutdown),
[Meta browser debugging and device setup](https://developers.meta.com/vr/documentation/web/browser-remote-debugging/).
