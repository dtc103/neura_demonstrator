"""Run with `uv run server.py`; relay WebXR poses to desktop viewers locally."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import ssl
import threading
import time

STATIC = Path(__file__).parent / "static"
STALE_SECONDS = 1.0


def empty_pose():
    return {"type": "pose", "controllers": {"left": None, "right": None}}


def validate_pose(payload):
    """Accept complete finite position/quaternion samples, or lost tracking."""
    if not isinstance(payload, dict) or payload.get("type") != "pose":
        raise ValueError("Expected a pose message")
    controllers = payload.get("controllers")
    if not isinstance(controllers, dict) or set(controllers) != {"left", "right"}:
        raise ValueError("Expected left and right controller entries")
    result = {"type": "pose", "controllers": {}, "reset": payload.get("reset") is True}
    for hand, pose in controllers.items():
        if pose is None:
            result["controllers"][hand] = None
            continue
        if not isinstance(pose, dict):
            raise ValueError("Invalid controller")
        values = {}
        for name, size in (("position", 3), ("orientation", 4)):
            vector = pose.get(name)
            if not isinstance(vector, list) or len(vector) != size:
                raise ValueError(f"Invalid {name}")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                raise ValueError("Pose components must be finite numbers")
            values[name] = vector.copy()
        norm = math.hypot(*values["orientation"])
        if not 0.9 <= norm <= 1.1:
            raise ValueError("Expected a unit quaternion")
        values["orientation"] = [v / norm for v in values["orientation"]]
        values["emulatedPosition"] = pose.get("emulatedPosition") is True
        result["controllers"][hand] = values
    return result


class Relay:
    def __init__(self):
        self.condition = threading.Condition()
        self.latest = empty_pose()
        self.sequence = 0
        self.updated = 0.0
        self.source = None
        self.epoch = 0

    def publish(self, source, payload):
        with self.condition:
            if self.source != source:
                if self.source is not None and time.monotonic() - self.updated < STALE_SECONDS:
                    return False
                self.source = source
                self.epoch += 1
            if payload.get("reset"):
                self.epoch += 1
            self.latest = {**payload, "epoch": self.epoch}
            self.updated = time.monotonic()
            self.sequence += 1
            self.condition.notify_all()
            return True

    def snapshot(self, previous):
        with self.condition:
            if previous == self.sequence:
                self.condition.wait(timeout=0.5)
            if self.updated and time.monotonic() - self.updated > STALE_SECONDS:
                self.latest = empty_pose()
                self.updated = 0.0
                self.source = None
                self.sequence += 1
            return self.sequence, self.latest


class PoseServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        self.relay = Relay()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # A pose arrives ~30 times per second; avoid flooding the terminal.
        if len(args) >= 2 and str(args[1]) not in ("200", "204"):
            super().log_message(format, *args)

    def respond(self, status, body=b"", content_type="text/plain"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            previous = -1
            try:
                while True:
                    sequence, payload = self.server.relay.snapshot(previous)
                    data = f"data: {json.dumps(payload)}\n\n" if sequence != previous else ": heartbeat\n\n"
                    self.wfile.write(data.encode())
                    self.wfile.flush()
                    previous = sequence
            except (ConnectionError, OSError):
                return
        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/static/style.css": ("style.css", "text/css"),
            "/static/app.js": ("app.js", "text/javascript"),
            "/static/math.js": ("math.js", "text/javascript"),
            "/static/renderer.js": ("renderer.js", "text/javascript"),
            "/static/test_math.html": ("test_math.html", "text/html; charset=utf-8"),
        }
        if path not in assets:
            self.respond(404, b"Not found")
            return
        name, mime = assets[path]
        self.respond(200, (STATIC / name).read_bytes(), mime)

    def do_POST(self):
        if self.path != "/api/pose":
            self.close_connection = True
            self.respond(404, b"Not found")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                self.close_connection = True
                self.respond(413, b"Expected a pose smaller than 8192 bytes")
                return
            payload = json.loads(self.rfile.read(size))
            pose = validate_pose(payload)
            source = payload.get("source")
            if not isinstance(source, str) or not 1 <= len(source) <= 64:
                raise ValueError("Expected a sender ID")
        except (ValueError, TypeError, OverflowError) as error:
            self.respond(400, str(error).encode())
            return
        if not self.server.relay.publish(source, pose):
            self.respond(409, b"Another Quest sender is active. Stop its tracking first.")
            return
        self.respond(204)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cert", type=Path, help="Trusted HTTPS certificate (PEM)")
    parser.add_argument("--key", type=Path, help="HTTPS private key (PEM)")
    args = parser.parse_args()
    if bool(args.cert) != bool(args.key):
        parser.error("--cert and --key must be supplied together")
    server = PoseServer((args.host, args.port))
    if args.cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert, args.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    scheme = "https" if args.cert else "http"
    print(f"Desktop viewer: {scheme}://localhost:{args.port}/", flush=True)
    print(f"Quest sender:   {scheme}://localhost:{args.port}/?source=1", flush=True)
    if not args.cert:
        print(f"USB forwarding: adb reverse tcp:{args.port} tcp:{args.port}", flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"For LAN access, replace localhost with the PC address (listening on {args.host}).", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
