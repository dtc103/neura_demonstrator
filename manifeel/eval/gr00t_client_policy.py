"""ManiFeel-compatible policy that queries a running GR00T policy server (gr00t/eval/run_gr00t_server.py).

Runs inside ManiFeel's Python 3.8 environment, so it only needs numpy + pyzmq + msgpack + msgpack-numpy
(no gr00t import). Observations are converted exactly like convert_to_lerobot_v2.py did for training.

action_format="delta":    model trained on ManiFeel's delta actions; returns n_action_steps of the chunk
                          as-is (wrist image, eef_position, sign-canonicalized eef_quaternion).
action_format="eef_pose": model trained on EEF poses (xyz + rot6d); the server returns absolute target
                          poses. Each env step, the next target is turned into a ManiFeel command with the
                          inverse controller model fit on the demos (fit_inverse_controller.py): desired
                          motion (next target - actual current pose), the motion of the previous step, and
                          the previous command. Expects the runner to step one action at a time; the model
                          is re-queried every `replan_steps` steps. With `ensemble_m`, it is queried every
                          step and the target is a temporal ensemble of all overlapping plans (see ACT).
"""

import json

import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
import zmq


def quat_xyzw_to_matrix(q):
    x, y, z, w = (q / np.linalg.norm(q, axis=-1, keepdims=True)).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rot6d_to_matrix(r6):
    """GR00T's rot6d: first two rows of the rotation matrix, re-orthonormalized (Gram-Schmidt)."""
    row1 = r6[..., 0:3] / np.linalg.norm(r6[..., 0:3], axis=-1, keepdims=True)
    row2 = r6[..., 3:6] - (row1 * r6[..., 3:6]).sum(-1, keepdims=True) * row1
    row2 = row2 / np.linalg.norm(row2, axis=-1, keepdims=True)
    return np.stack([row1, row2, np.cross(row1, row2)], -2)


def matrix_to_rotvec(m):
    angle = np.arccos(np.clip((np.trace(m, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0))
    vee = np.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1)
    sin = np.sin(angle)
    scale = np.where(sin > 1e-8, angle / (2 * np.where(sin > 1e-8, sin, 1.0)), 0.5)
    return vee * scale[..., None]


class Gr00tClientPolicy:
    def __init__(self, host, port, task, action_format="delta", n_action_steps=8, replan_steps=8,
                 inverse_controller=None, ensemble_m=None, gripper_width=None, timeout_s=300):
        assert action_format in ("delta", "eef_pose")
        self.task = task
        self.action_format = action_format
        self.n_action_steps = n_action_steps
        self.replan_steps = replan_steps
        self.ensemble_m = ensemble_m  # eef_pose: temporal ensembling (query every step), None = off
        # eef_pose models trained with a gripper column (co-training with REASSEMBLE): ManiFeel holds the
        # gear throughout, so the state gets this constant opening and the predicted gripper is ignored
        self.gripper_width = gripper_width
        if action_format == "eef_pose":
            # (6, 4) per axis: desired motion, previous motion, previous action, bias
            self.inverse_weights = np.asarray(json.loads(open(inverse_controller).read())["weights"])
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
            raise RuntimeError("GR00T server error: {}".format(response["error"]))
        return response

    def ping(self):
        return self._call("ping")

    def reset(self):
        self._prev_quat = None
        self._plan = None
        self._plan_step = 0
        self._prev_pos = None  # eef_pose: pose and command of the previous step
        self._prev_rot = None
        self._prev_action = None
        self._plans = []  # eef_pose + ensembling: past plans, newest last

    def _ensemble_target(self, plan):
        """Temporal ensembling: weighted mean of all past plans' predictions for the current step.

        A plan made `age` steps ago predicts this step at index `age`; weight exp(-m * age), so the
        newest plan (made from the latest image) counts most. Averaging independent flow-matching
        samples cancels their noise. rot6d is averaged element-wise and re-orthonormalized.
        """
        self._plans = (self._plans + [plan])[-plan.shape[1]:]
        n = len(self._plans)
        preds = np.stack([p[:, n - 1 - i] for i, p in enumerate(self._plans)])  # (n, B, 9), oldest first
        ages = np.arange(n - 1, -1, -1)
        w = np.exp(-self.ensemble_m * ages)[:, None, None]
        mean = (w * preds).sum(0) / w.sum()
        rot = rot6d_to_matrix(mean[:, 3:9])
        return np.concatenate([mean[:, 0:3], rot[:, :2, :].reshape(len(mean), 6)], axis=1)

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

    def _query(self, image, state):
        batch = len(image)
        observation = {
            "video": {"wrist": np.ascontiguousarray(image[:, None])},  # (B, 1, H, W, 3)
            "state": state,
            "language": {"annotation.human.task_description": [[self.task]] * batch},
        }
        action, _ = self._call("get_action", {"observation": observation, "options": None})
        return action

    def predict_action(self, obs_dict):
        # ManiFeel stacks n_obs_steps frames; GR00T was trained on the current frame only
        wrist = obs_dict["wrist"][:, -1].cpu().numpy()  # (B, 3, H, W) float in [0, 1]
        state = obs_dict["state"][:, -1].cpu().numpy().astype(np.float64)  # (B, 7) pos + quat xyzw
        image = (np.clip(wrist, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).transpose(0, 2, 3, 1)

        if self.action_format == "delta":
            gr00t_state = {
                "eef_position": np.ascontiguousarray(state[:, None, 0:3], dtype=np.float32),
                "eef_quaternion": np.ascontiguousarray(
                    self._canonicalize(state[:, 3:7])[:, None], dtype=np.float32),
            }
            action = self._query(image, gr00t_state)
            chunk = np.concatenate([action["delta_translation"], action["delta_rotation"]], axis=-1)
            return {"action": torch.from_numpy(chunk[:, : self.n_action_steps].astype(np.float32))}

        # --- eef_pose: track the predicted pose trajectory from the actual current pose ---
        cur_pos = state[:, 0:3]
        cur_rot = quat_xyzw_to_matrix(state[:, 3:7])
        if self.ensemble_m is not None or self._plan is None or self._plan_step >= self.replan_steps:
            rot6d = cur_rot[:, :2, :].reshape(len(state), 6)
            pose = np.concatenate([cur_pos, rot6d], axis=1)[:, None].astype(np.float32)  # (B, 1, 9)
            gr00t_state = {"eef_pose": pose}
            if self.gripper_width is not None:
                gr00t_state["gripper"] = np.full((len(state), 1, 1), self.gripper_width, dtype=np.float32)
            self._plan = np.asarray(self._query(image, gr00t_state)["eef_pose"], dtype=np.float64)
            self._plan_step = 0
        if self.ensemble_m is None:
            target = self._plan[:, self._plan_step]  # (B, 9) absolute pose to reach after this step
            self._plan_step += 1
        else:
            target = self._ensemble_target(self._plan)

        # World-frame pose changes, as in fit_inverse_controller.py (position, rotvec of R_to * R_from^-1)
        desired = np.concatenate([
            target[:, 0:3] - cur_pos,
            matrix_to_rotvec(rot6d_to_matrix(target[:, 3:9]) @ np.swapaxes(cur_rot, -1, -2)),
        ], axis=1)
        if self._prev_pos is None:  # episode start: robot at rest, no previous command
            prev_motion = np.zeros_like(desired)
            self._prev_action = np.zeros_like(desired)
        else:
            prev_motion = np.concatenate([
                cur_pos - self._prev_pos,
                matrix_to_rotvec(cur_rot @ np.swapaxes(self._prev_rot, -1, -2)),
            ], axis=1)
        w = self.inverse_weights
        action = desired * w[:, 0] + prev_motion * w[:, 1] + self._prev_action * w[:, 2] + w[:, 3]
        action = np.clip(action, -1.0, 1.0)
        self._prev_pos, self._prev_rot, self._prev_action = cur_pos, cur_rot, action
        return {"action": torch.from_numpy(action[:, None].astype(np.float32))}  # (B, 1, 6)
