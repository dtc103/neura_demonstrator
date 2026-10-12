"""ManiFeel-compatible policy that queries a running DBP policy server (eval/run_dbp_server.py).

Runs inside ManiFeel's Python 3.8 environment (numpy + pyzmq + msgpack + msgpack-numpy, no DBP import).
DBP was trained on ManiFeel's own observations (dbp/prepare_manifeel_dbp.py: wrist camera as uint8, raw
7-D state, 2 observation steps), so they are passed through unchanged; the server returns the 8 executed
steps of ManiFeel's delta-action chunk.
"""

import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
import zmq


class DBPClientPolicy:
    def __init__(self, host, port, timeout_s=300):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_s * 1000)
        self._socket.connect("tcp://{}:{}".format(host, port))

    def _call(self, endpoint, data=None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self._socket.send(msgpack.packb(request, default=mnp.encode))
        response = msgpack.unpackb(self._socket.recv(), object_hook=mnp.decode, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError("DBP server error: {}".format(response["error"]))
        return response

    def ping(self):
        return self._call("ping")

    def reset(self):
        pass

    def predict_action(self, obs_dict):
        wrist = obs_dict["wrist"].cpu().numpy()  # (B, T, 3, H, W) float in [0, 1]
        data = {
            # uint8 like the training data (and 4x less to send)
            "wrist": np.ascontiguousarray((np.clip(wrist, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)),
            "state": np.ascontiguousarray(obs_dict["state"].cpu().numpy().astype(np.float32)),  # (B, T, 7)
        }
        response = self._call("get_action", data)
        return {k: torch.from_numpy(np.array(response[k], dtype=np.float32)) for k in ("action", "action_pred")}
