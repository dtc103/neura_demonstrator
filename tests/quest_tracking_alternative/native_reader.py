"""Launch the native Quest tracking app over ADB and consume controller poses."""

import json
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time

from webxr_reader import TrackingState, WebXRReader

PACKAGE = "org.neura.questtracking"
ACTIVITY = PACKAGE + "/.TrackingActivity"
ROOT = Path(__file__).parent
DEFAULT_APK = ROOT / "android/app/build/outputs/apk/debug/app-debug.apk"


class NativeReader(WebXRReader):
    def __init__(self, host="127.0.0.1", port=9201, serial=None, keep_awake=True, install_apk=None):
        self.state = TrackingState(allowed_spaces=("stage", "local"))
        self.stop_event = threading.Event()
        self.connection = None
        self.thread = None
        self.forwarded = self.proximity_overridden = self.launched = False
        self.host, self.port = host, port
        self.last_error = None
        self.adb = shutil.which("adb")
        if not self.adb:
            raise RuntimeError("ADB is missing; install adb first")
        self.prefix = [self.adb] + (["-s", serial] if serial else [])
        if host != "127.0.0.1":
            raise ValueError("Native USB tracking requires --host 127.0.0.1")
        try:
            self.command("get-state")
            if install_apk is not None:
                apk = Path(install_apk)
                if not apk.is_file():
                    raise RuntimeError(f"APK not found: {apk}; build it with build_native_app.py")
                self.command("install", "-r", str(apk.resolve()), timeout=90)
            if not self.command("shell", "pm", "path", PACKAGE).startswith("package:"):
                raise RuntimeError("Native app is not installed. Build it first, then pass --install-apk " + str(DEFAULT_APK))
            if keep_awake:
                # Mark before issuing the command so a timeout still triggers undo.
                self.proximity_overridden = True
                self.command("shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.prox_close")
                print("Applied proximity override: headset is treated as worn. It will be restored on exit.", flush=True)
            mappings = self.command("forward", "--list")
            if any(f"tcp:{port}" in line.split()[1:2] for line in mappings.splitlines()):
                raise RuntimeError(f"ADB host port {port} is already forwarded; close the other reader or choose another --port")
            self.command("forward", f"tcp:{port}", "tcp:9201")
            self.forwarded = True
            self.launched = True
            output = self.command("shell", "am", "start", "-W", "-n", ACTIVITY)
            if "Error:" in output or "Exception" in output:
                raise RuntimeError(output)
            self.thread = threading.Thread(target=self.receive, daemon=True)
            self.thread.start()
            print("Native Quest app launched; waiting for fixed-frame controller data. No browser or Start button is needed.", flush=True)
            deadline = time.monotonic() + 30
            while self.state.snapshot() is None:
                if time.monotonic() > deadline:
                    raise RuntimeError("No native pose frames received after 30 seconds. "
                                       f"Last connection error: {self.last_error}. Inspect `adb logcat -s NeuraQuest`. "
                                       "One-time headset permissions may still require acceptance.")
                time.sleep(.05)
        except BaseException:
            self.stop()
            raise

    def command(self, *args, timeout=15):
        result = subprocess.run([*self.prefix, *args], text=True, capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ADB command failed")
        return result.stdout.strip()

    def receive(self):
        while not self.stop_event.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=1) as connection:
                    self.connection = connection
                    connection.settimeout(.25)
                    buffer = b""
                    while not self.stop_event.is_set():
                        try:
                            chunk = connection.recv(16384)
                        except socket.timeout:
                            continue
                        if not chunk:
                            break
                        buffer += chunk
                        if len(buffer) > 131072:
                            raise ValueError("Native pose buffer exceeded limit")
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            if len(line) > 32768:
                                raise ValueError("Native pose frame exceeded limit")
                            self.accept_frame(json.loads(line))
                            self.last_error = None
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.last_error = str(error)
            finally:
                self.connection = None
            self.stop_event.wait(.25)

    def accept_frame(self, data):
        if not isinstance(data, dict) or data.get("schema") != "neura.quest.controllers.v1":
            raise ValueError("Wrong native protocol schema")
        stamp = data.get("sample_time_ns")
        if type(stamp) is not int or stamp < 0:
            raise ValueError("Invalid native sample timestamp")
        previous = getattr(self, "last_sample", None)
        if previous is not None and previous[0] == data.get("source") and stamp <= previous[1]:
            raise ValueError("Repeated or out-of-order native sample timestamp")
        self.state.publish(data)
        self.last_sample = (data["source"], stamp)

    def stop(self):
        self.stop_event.set()
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(timeout=2)
        # Cleanup is best effort; report if the headset cannot be reached.
        commands = []
        if self.launched:
            commands.append(("stop native app", ("shell", "am", "force-stop", PACKAGE)))
        if self.forwarded:
            commands.append(("remove forwarding", ("forward", "--remove", f"tcp:{self.port}")))
        if self.proximity_overridden:
            commands.append(("restore proximity sensor", ("shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.automation_disable")))
        self.launched = self.forwarded = self.proximity_overridden = False
        for description, arguments in commands:
            try:
                self.command(*arguments, timeout=3)
            except (RuntimeError, subprocess.SubprocessError) as error:
                print(f"Could not {description}: {error}", flush=True)


def add_backend_arguments(parser):
    parser.add_argument("--backend", choices=("native", "webxr"), default="native")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="Host port; default 9201 native, 8443 WebXR")
    parser.add_argument("--serial", help="ADB device serial for native mode")
    parser.add_argument("--no-keep-awake", action="store_true", help="Native mode: leave physical proximity behaviour unchanged")
    parser.add_argument("--install-apk", type=Path, help="Native mode: install this APK before launching")


def start_reader(args):
    port = args.port if args.port is not None else (9201 if args.backend == "native" else 8443)
    if not 1 <= port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.backend == "webxr":
        if args.install_apk is not None:
            raise ValueError("--install-apk only applies to native mode")
        return WebXRReader(args.host, port)
    return NativeReader(args.host, port, args.serial, not args.no_keep_awake, args.install_apk)
