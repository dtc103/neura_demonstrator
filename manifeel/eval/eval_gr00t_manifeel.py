"""Closed-loop ManiFeel evaluation of a GR00T policy served by gr00t/eval/run_gr00t_server.py.

Uses ManiFeel's own env runner (vision_wrist task config), so success rate and videos are measured
exactly like ManiFeel's vision-only Diffusion Policy baseline. Run inside ManiFeel's Python 3.8
environment / Apptainer container, from the manifeel repo root.
"""

import isaacgym  # noqa: F401  (must be imported before torch)

import argparse
import json
import os
import pathlib
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gr00t_client_policy import Gr00tClientPolicy  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifeel-root", required=True, help="clone of purdue-mars/manifeel")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isaacgym-cfg", default="isaacgym_config_gear.yaml")
    parser.add_argument("--task", default="assemble the gear onto the shaft")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--n-test", type=int, default=50, help="parallel test episodes (ManiFeel default 50)")
    parser.add_argument("--n-test-vis", type=int, default=10, help="episodes to record as video")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=100000, help="ManiFeel test_start_seed")
    parser.add_argument("--action-format", choices=["delta", "eef_pose"], default="delta",
                        help="must match the dataset the model was trained on")
    parser.add_argument("--replan-steps", type=int, default=8,
                        help="eef_pose: env steps per model query (the delta format executes 8 per query)")
    parser.add_argument("--inverse-controller",
                        default=str(pathlib.Path(__file__).resolve().parent / "manifeel_gear_inverse_controller.json"),
                        help="eef_pose: fit_inverse_controller.py output (pose change -> ManiFeel command)")
    parser.add_argument("--ensemble-m", type=float, default=None,
                        help="eef_pose: temporal ensembling with weights exp(-m * age); queries every step")
    parser.add_argument("--gripper-width", type=float, default=None,
                        help="eef_pose models with a gripper column: constant opening sent as state (e.g. 0.033)")
    args = parser.parse_args()
    eef = args.action_format == "eef_pose"

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config_dir = pathlib.Path(args.manifeel_root) / "manifeel" / "config"
    hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None)
    cfg = hydra.compose(
        config_name="train_diffusion_workspace",
        overrides=[
            "task=vision_wrist",
            "isaacgym_cfg_name={}".format(args.isaacgym_cfg),
            "task.env_runner.n_test={}".format(args.n_test),
            "task.env_runner.n_test_vis={}".format(args.n_test_vis),
            "task.env_runner.max_steps={}".format(args.max_steps),
            "task.env_runner.test_start_seed={}".format(args.seed),
        ] + (["n_action_steps=1"] if eef else []),  # eef_pose: the client tracks the plan step by step
    )

    policy = Gr00tClientPolicy(args.host, args.port, args.task, action_format=args.action_format,
                               n_action_steps=cfg.n_action_steps, replan_steps=args.replan_steps,
                               inverse_controller=args.inverse_controller, ensemble_m=args.ensemble_m,
                               gripper_width=args.gripper_width)
    print("GR00T server:", policy.ping(), flush=True)

    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(out))
    log = runner.run(policy)

    videos = sorted(str(v._path) for k, v in log.items() if k.startswith("test/sim_video"))
    result = {
        "success_rate": float(log["test/mean_score"]),
        "n_test": args.n_test,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "task": args.task,
        "action_format": args.action_format,
        "replan_steps": args.replan_steps if eef else cfg.n_action_steps,
        "inverse_controller": os.path.basename(args.inverse_controller) if eef else None,
        "ensemble_m": args.ensemble_m if eef else None,
        "gripper_width": args.gripper_width if eef else None,
        "videos": [os.path.relpath(v, out) for v in videos],
    }
    (out / "eval_result.json").write_text(json.dumps(result, indent=2))
    print("\nSuccess rate: {:.1%} over {} episodes".format(result["success_rate"], args.n_test), flush=True)
    print("Results: {}".format(out), flush=True)


if __name__ == "__main__":
    main()
