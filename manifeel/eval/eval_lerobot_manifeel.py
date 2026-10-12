"""Closed-loop ManiFeel evaluation of a LeRobot policy served by eval/run_lerobot_server.py
(e.g. GR00T N1.7 with the one-step drifting action head).

Same as eval_gr00t_manifeel.py (ManiFeel's own env runner, vision_wrist task config, so success rate and
videos are measured exactly like ManiFeel's vision-only Diffusion Policy baseline), with the LeRobot client.
Run inside ManiFeel's Python 3.8 environment, from the manifeel repo root.
"""

import isaacgym  # noqa: F401  (must be imported before torch)

import argparse
import json
import os
import pathlib
import sys

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lerobot_client_policy import LeRobotClientPolicy  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifeel-root", required=True, help="clone of purdue-mars/manifeel")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isaacgym-cfg", default="isaacgym_config_gear.yaml")
    parser.add_argument("--task", default="assemble the gear onto the shaft", help="must match training")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--n-test", type=int, default=50, help="parallel test episodes (ManiFeel default 50)")
    parser.add_argument("--n-test-vis", type=int, default=10, help="episodes to record as video")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=100000, help="ManiFeel test_start_seed")
    args = parser.parse_args()

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
        ],  # n_action_steps 8: executed steps per call, as in the GR00T delta evals
    )

    policy = LeRobotClientPolicy(args.host, args.port, args.task, n_action_steps=cfg.n_action_steps)
    server = policy.ping()
    print("LeRobot server:", server, flush=True)

    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(out))
    log = runner.run(policy)

    videos = sorted(str(v._path) for k, v in log.items() if k.startswith("test/sim_video"))
    result = {
        "success_rate": float(log["test/mean_score"]),
        "n_test": args.n_test,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "task": args.task,
        "policy": server.get("policy"),
        "checkpoint": server.get("checkpoint"),
        "n_action_steps": cfg.n_action_steps,
        "videos": [os.path.relpath(v, out) for v in videos],
    }
    (out / "eval_result.json").write_text(json.dumps(result, indent=2))
    print("\nSuccess rate: {:.1%} over {} episodes".format(result["success_rate"], args.n_test), flush=True)
    print("Results: {}".format(out), flush=True)


if __name__ == "__main__":
    main()
