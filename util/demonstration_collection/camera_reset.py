"""Startup-only RealSense restart; no Qt or robot commands."""
import json
from pathlib import Path
import time


def reset_cameras(config, cancel, report, *, sdk=None, reboot_seconds=10, timeout=30):
    """Reset configured cameras, then wait for their serials to reappear.

    report receives (phase, message); the caller owns the background thread.
    Enumeration after reset confirms connection, not successful streaming.
    """
    try:
        serials = json.loads(Path(config).read_text())['cameras']
        if set(serials) != {'wrist', 'side', 'opposite'} or len(set(serials.values())) != 3:
            raise ValueError('Camera config must specify three distinct wrist/side/opposite serials')
        if sdk is None:
            import pyrealsense2 as sdk
        context = sdk.context()

        def devices():
            return {device.get_info(sdk.camera_info.serial_number): device
                    for device in context.query_devices()}

        connected = devices()
        missing = set(serials.values()) - connected.keys()
        if missing:
            raise RuntimeError(f'Configured cameras not connected: {sorted(missing)}')
        for role, serial in serials.items():
            if cancel.is_set():
                return
            report('resetting', f'Restarting {role} camera ({serial})…')
            connected[serial].hardware_reset()
        connected.clear()  # Old device handles are invalid after a reset.
        report('resetting', 'Waiting for cameras to reconnect; you can prepare the Quest and task fields.')
        if cancel.wait(reboot_seconds):
            return
        deadline = time.monotonic() + timeout
        while not cancel.is_set():
            missing = set(serials.values()) - devices().keys()
            if not missing:
                report('ready', 'Camera reset complete. Prepare a session when ready.')
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Cameras did not reconnect: {sorted(missing)}')
            if cancel.wait(.5):
                return
    except Exception as error:
        report('failed', f'Camera reset failed: {error}. Close other camera programs/check USB, then retry.')
