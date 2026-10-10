import errno
from http.client import HTTPConnection
import json
import io
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from server import Handler, PoseServer, Relay, STALE_SECONDS, empty_pose, validate_pose


def sample():
    return {"type": "pose", "source": "test-headset", "controllers": {
        "left": None, "right": {"position": [0.25, 1.2, -0.5], "orientation": [0, 0, 0, 1]}}}


class PoseTests(unittest.TestCase):
    def test_preserves_translation_and_normalizes_rotation(self):
        data = sample()
        data["controllers"]["right"]["orientation"][-1] = 1.01
        result = validate_pose(data)
        self.assertEqual(result["controllers"]["right"]["position"], [0.25, 1.2, -0.5])
        self.assertEqual(result["controllers"]["right"]["orientation"], [0, 0, 0, 1])

    def test_rejects_invalid_tracking_samples(self):
        for value in (float("nan"), float("inf"), True, "0"):
            with self.subTest(value=value):
                data = sample()
                data["controllers"]["right"]["position"][0] = value
                with self.assertRaises(ValueError):
                    validate_pose(data)
        for field, value in (("orientation", [0, 0, 0, 0]), ("position", [0, 0])):
            data = sample()
            data["controllers"]["right"][field] = value
            with self.assertRaises(ValueError):
                validate_pose(data)

    def test_tracking_loss_is_explicit(self):
        self.assertEqual(validate_pose(empty_pose())["controllers"], {"left": None, "right": None})

    def test_sender_ownership_and_reference_reset(self):
        relay = Relay()
        pose = validate_pose(sample())
        self.assertTrue(relay.publish("first", pose))
        first_epoch = relay.latest["epoch"]
        self.assertFalse(relay.publish("second", pose))
        self.assertTrue(relay.publish("first", {**pose, "reset": True}))
        self.assertGreater(relay.latest["epoch"], first_epoch)
        relay.updated = time.monotonic() - STALE_SECONDS - 0.1
        self.assertEqual(relay.snapshot(-1)[1], empty_pose())
        self.assertTrue(relay.publish("second", pose))


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.server = PoseServer(("127.0.0.1", 0))
        except PermissionError as error:
            if error.errno == errno.EPERM:
                raise unittest.SkipTest("This sandbox prohibits opening local sockets") from error
            raise
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def connection(self):
        return HTTPConnection(*self.server.server_address, timeout=3)

    def test_static_assets_and_no_path_traversal(self):
        with self.connection() as connection:
            for path in ("/", "/?source=1", "/static/app.js", "/static/math.js", "/static/renderer.js"):
                connection.request("GET", path)
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())
            connection.request("GET", "/static/../server.py")
            response = connection.getresponse()
            self.assertEqual(response.status, 404)
            response.read()

    def test_pose_reaches_event_stream_and_disconnect_clears_it(self):
        self.server.relay = Relay()
        with self.connection() as viewer, self.connection() as sender:
            viewer.request("GET", "/api/events")
            stream = viewer.getresponse()
            self.assertEqual(stream.getheader("Content-Type"), "text/event-stream")

            def read_event():
                while True:
                    line = stream.readline()
                    if line.startswith(b"data: "):
                        return json.loads(line[6:])

            self.assertEqual(read_event()["controllers"], empty_pose()["controllers"])
            data = sample()
            sender.request("POST", "/api/pose", json.dumps(data), {"Content-Type": "application/json"})
            response = sender.getresponse()
            self.assertEqual(response.status, 204)
            response.read()
            event = read_event()
            self.assertEqual(event["controllers"]["right"]["position"], [0.25, 1.2, -0.5])
            self.assertEqual(read_event()["controllers"], empty_pose()["controllers"])

    def test_invalid_json_and_oversized_body(self):
        with self.connection() as connection:
            connection.request("POST", "/api/pose", "broken json")
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
            connection.request("POST", "/api/pose", "x" * 8193)
            response = connection.getresponse()
            self.assertEqual(response.status, 413)
            response.read()


class HandlerTests(unittest.TestCase):
    """Exercise actual HTTP parsing/serialization even without socket permissions."""

    def request(self, method, path, body=b"", relay=None, event_limit=None):
        class Connection:
            def __init__(self):
                self.output = bytearray()
                self.event_count = 0

            def makefile(self, *args):
                return io.BytesIO(
                    f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)

            def sendall(self, data):
                if event_limit and bytes(data).startswith(b"data: "):
                    self.event_count += 1
                    if self.event_count > event_limit:
                        raise ConnectionResetError("Viewer closed")
                self.output.extend(data)

        connection = Connection()
        server = SimpleNamespace(relay=relay or Relay())
        Handler(connection, ("127.0.0.1", 12345), server)
        return bytes(connection.output), server.relay

    def test_http_pose_validation_and_relay(self):
        data = sample()
        response, relay = self.request("POST", "/api/pose", json.dumps(data).encode())
        self.assertTrue(response.startswith(b"HTTP/1.1 204"))
        self.assertEqual(relay.latest["controllers"]["right"]["position"], [0.25, 1.2, -0.5])
        response, _ = self.request("POST", "/api/pose", b"invalid")
        self.assertTrue(response.startswith(b"HTTP/1.1 400"))
        response, _ = self.request("POST", "/api/pose", b"x" * 8193)
        self.assertTrue(response.startswith(b"HTTP/1.1 413"))
        data["source"] = "second-headset"
        response, _ = self.request("POST", "/api/pose", json.dumps(data).encode(), relay)
        self.assertTrue(response.startswith(b"HTTP/1.1 409"))

    def test_static_routes(self):
        for path in ("/", "/?source=1", "/static/app.js", "/static/math.js", "/static/renderer.js"):
            response, _ = self.request("GET", path)
            self.assertTrue(response.startswith(b"HTTP/1.1 200"))
            self.assertIn(b"Content-Length:", response)
        response, _ = self.request("GET", "/static/../server.py")
        self.assertTrue(response.startswith(b"HTTP/1.1 404"))

    def test_sse_delivers_tracking_and_loss(self):
        relay = Relay()
        tracked = validate_pose(sample())
        relay.snapshot = Mock(side_effect=[(1, tracked), (2, empty_pose()), (3, empty_pose())])
        response, _ = self.request("GET", "/api/events", relay=relay, event_limit=2)
        self.assertIn(b"Content-Type: text/event-stream", response)
        events = [json.loads(line[6:]) for line in response.splitlines() if line.startswith(b"data: ")]
        self.assertEqual(events[0]["controllers"], tracked["controllers"])
        self.assertEqual(events[1]["controllers"], empty_pose()["controllers"])


if __name__ == "__main__":
    unittest.main()
