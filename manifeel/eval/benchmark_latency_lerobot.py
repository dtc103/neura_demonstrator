"""Inference latency of a LeRobot policy checkpoint (GR00T N1.7 VLM + one-step drifting head, arXiv 2609.18108),
for comparison with standard GR00T (benchmark_latency_gr00t.py). Same observations, same call as the ManiFeel eval
server (eval/run_lerobot_server.py): preprocessor -> predict_action_chunk -> postprocessor per step.

Run inside the drifting LeRobot fork's environment:
    python benchmark_latency_lerobot.py --checkpoint runs_lerobot/<run>/checkpoints/020000/pretrained_model \
        --npz manifeel_gear_dbp_256.npz --out latency/drifting.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from latency_utils import StageTimer, benchmark, load_observations, write

IMAGE_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", default="assemble the gear onto the shaft")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 50])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    cfg.device = args.device
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint, config=cfg)
    policy.to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    model = policy._groot_model
    timer = StageTimer(model)
    wrist, state = load_observations(args.npz, 2 * max(args.batch_sizes) * 4, args.seed)

    def call(w, s):  # exactly what eval/run_lerobot_server.py does per request
        batch = preprocessor({
            IMAGE_KEY: torch.from_numpy(np.ascontiguousarray(w.transpose(0, 3, 1, 2))).float() / 255.0,
            STATE_KEY: torch.from_numpy(s),
            "task": [args.task] * len(s),
        })
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
        steps = [postprocessor(chunk[:, t]) for t in range(chunk.shape[1])]
        return torch.stack(steps, dim=1).float().cpu().numpy()

    print(f"{cfg.type} policy {args.checkpoint}: one network evaluation in the action head", flush=True)
    results = benchmark(call, timer, wrist, state, args.batch_sizes, args.repeats, args.warmup)
    write(args.out, f"gr00t_n1.7_vlm_{cfg.type}", args.checkpoint,
          {"params_backbone_M": sum(p.numel() for p in model.backbone.parameters()) / 1e6,
           "params_action_head_M": sum(p.numel() for p in model.action_head.parameters()) / 1e6}, results)


if __name__ == "__main__":
    main()
