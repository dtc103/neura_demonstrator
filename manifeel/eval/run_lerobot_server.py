"""Policy server for a LeRobot policy checkpoint (e.g. the GR00T N1.7 drifting head from
slurm_train_gr00t_drifting_manifeel_h200.sbatch), for ManiFeel's Python 3.8 client (eval/lerobot_client_policy.py).

Runs in the LeRobot environment of the training job and speaks the same ZeroMQ + msgpack protocol as
eval/run_dbp_server.py. The policy's own pre/post-processors (normalization, GR00T tokenization) are loaded
from the checkpoint, so observations go in exactly as LeRobot's dataset provided them during training.

Request {"endpoint": "get_action", "data": {"wrist": (B, 3, H, W) uint8, "state": (B, 7) float, "task": str}}
-> {"action_pred": (B, chunk_size, action_dim)}, unnormalized.

    python run_lerobot_server.py --checkpoint runs_lerobot/<run>/checkpoints/010000/pretrained_model --port 5555
"""

import argparse

import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
import zmq

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

IMAGE_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="<run>/checkpoints/<step>/pretrained_model")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    cfg.device = args.device
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint, config=cfg)
    policy.to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    print(f"Loaded {cfg.type} policy (chunk_size {cfg.chunk_size}) from {args.checkpoint}", flush=True)

    socket = zmq.Context.instance().socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"Server is ready on port {args.port}", flush=True)

    while True:
        request = msgpack.unpackb(socket.recv(), object_hook=mnp.decode, raw=False)
        try:
            if request["endpoint"] == "ping":
                response = {"status": "ok", "policy": cfg.type, "checkpoint": args.checkpoint}
            elif request["endpoint"] == "get_action":
                data = request["data"]
                batch_size = len(data["wrist"])
                batch = preprocessor({
                    IMAGE_KEY: torch.from_numpy(np.asarray(data["wrist"])).float() / 255.0,
                    STATE_KEY: torch.from_numpy(np.asarray(data["state"], dtype=np.float32)),
                    "task": [data["task"]] * batch_size,
                })
                with torch.inference_mode():
                    chunk = policy.predict_action_chunk(batch)  # (B, T, D), normalized
                # The postprocessor works on single-step actions, as in LeRobot's rollout loop
                steps = [postprocessor(chunk[:, t]) for t in range(chunk.shape[1])]
                response = {"action_pred": torch.stack(steps, dim=1).float().cpu().numpy()}
            else:
                response = {"error": f"unknown endpoint {request['endpoint']}"}
        except Exception as e:  # report to the client instead of leaving it waiting
            response = {"error": repr(e)}
        socket.send(msgpack.packb(response, default=mnp.encode))


if __name__ == "__main__":
    main()
