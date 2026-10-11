"""ManiFeel-compatible policy that queries a running LeRobot policy server (eval/run_lerobot_server.py).

Runs inside ManiFeel's Python 3.8 environment (numpy + pyzmq + msgpack + msgpack-numpy, no LeRobot import).
The policy was trained on the ManiFeel LeRobot dataset from convert_to_lerobot_v2.py (delta format), so the
observations are converted the same way as for GR00T's delta models (gr00t_client_policy.py): current wrist
frame as uint8 and the 7-D state with the quaternion sign-canonicalized; the first n_action_steps of the
returned delta-action chunk are executed.
"""

import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
import zmq


class LeRobotClientPolicy:
    def __init__(self, host, port, task, n_action_steps=8, timeout_s=300):
        self.task = task
        self.n_action_steps = n_action_steps
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_s * 1000)
        self._socket.connect("tcp://{}:{}".format(host, port))
        self.reset()

    def _call(self, endpoint, data=None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self._socket.send(msgpack.packb(request, default=mnp.encode))
        response = msgpack.unpackb(self._socket.recv(), object_hook=mnp.decode, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError("LeRobot server error: {}".format(response["error"]))
        return response

    def ping(self):
        return self._call("ping")

    def reset(self):
        self._prev_quat = None

    def _canonicalize(self, quat):
        """Same rule as training: first step -> largest component positive, then stay continuous."""
        quat = quat.copy()
        if self._prev_quat is None:
            flip = quat[np.arange(len(quat)), np.abs(quat).argmax(1)] < 0
        else:
            flip = (quat * self._prev_quat).sum(1) < 0
        quat[flip] *= -1
        self._prev_quat = quat
        return quat

    def predict_action(self, obs_dict):
        # ManiFeel stacks n_obs_steps frames; the policy was trained on the current frame only
        wrist = obs_dict["wrist"][:, -1].cpu().numpy()  # (B, 3, H, W) float in [0, 1]
        state = obs_dict["state"][:, -1].cpu().numpy().astype(np.float32)  # (B, 7) pos + quat xyzw
        state = np.concatenate([state[:, 0:3], self._canonicalize(state[:, 3:7])], axis=1)
        data = {
            "wrist": np.ascontiguousarray((np.clip(wrist, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)),
            "state": np.ascontiguousarray(state, dtype=np.float32),
            "task": self.task,
        }
        chunk = np.array(self._call("get_action", data)["action_pred"], dtype=np.float32)  # (B, T, 6)
        return {"action": torch.from_numpy(chunk[:, : self.n_action_steps]), "action_pred": torch.from_numpy(chunk)}
