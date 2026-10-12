"""Inference latency of a standard GR00T N1.7 checkpoint (Isaac-GR00T), for comparison with the drifting head
(benchmark_latency_lerobot.py). Same observations, same call as the ManiFeel eval server: Gr00tPolicy.get_action
on the wrist frame + EEF state + task text -> 16-step action chunk.

Run inside the Isaac-GR00T environment:
    uv run python benchmark_latency_gr00t.py --model-path runs/<run>/checkpoint-10000 \
        --npz manifeel_gear_dbp_256.npz --out latency/gr00t.json
"""

from dataclasses import dataclass, field
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

from latency_utils import StageTimer, benchmark, load_observations, write


@dataclass
class Args:
    model_path: str
    npz: str
    out: str
    task: str = "assemble the gear onto the shaft"
    embodiment_tag: str = "new_embodiment"
    device: str = "cuda:0"
    batch_sizes: list[int] = field(default_factory=lambda: [1, 50])
    """1 = one real robot; 50 = ManiFeel's parallel eval environments"""
    repeats: int = 100
    warmup: int = 10
    seed: int = 0


def main(args: Args):
    torch.manual_seed(args.seed)
    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag), model_path=args.model_path,
                         device=args.device, strict=True)
    timer = StageTimer(policy.model)
    wrist, state = load_observations(args.npz, 2 * max(args.batch_sizes) * 4, args.seed)

    def call(w, s):  # exactly what eval/gr00t_client_policy.py sends for delta actions
        b = len(s)
        observation = {
            "video": {"wrist": np.ascontiguousarray(w[:, None])},
            "state": {"eef_position": np.ascontiguousarray(s[:, None, 0:3]),
                      "eef_quaternion": np.ascontiguousarray(s[:, None, 3:7])},
            "language": {"annotation.human.task_description": [[args.task]] * b},
        }
        action, _ = policy.get_action(observation)
        return np.concatenate([action["delta_translation"], action["delta_rotation"]], axis=-1)

    head = policy.model.action_head
    print(f"GR00T N1.7 {args.model_path}: flow matching, {head.num_inference_timesteps} denoising steps", flush=True)
    results = benchmark(call, timer, wrist, state, args.batch_sizes, args.repeats, args.warmup)
    write(args.out, "gr00t_n1.7_flow_matching", args.model_path,
          {"num_inference_timesteps": head.num_inference_timesteps,
           "params_backbone_M": sum(p.numel() for p in policy.model.backbone.parameters()) / 1e6,
           "params_action_head_M": sum(p.numel() for p in head.parameters()) / 1e6}, results)


if __name__ == "__main__":
    main(tyro.cli(Args))
