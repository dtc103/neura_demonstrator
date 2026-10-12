"""Fit the mapping from desired end-effector motion to ManiFeel actions on the demos.

Used by eval/gr00t_client_policy.py (action_format="eef_pose") to turn GR00T's predicted EEF poses
back into ManiFeel's delta commands. Per action axis i, a linear model fit on all demo steps:

    action_t[i] = w0 * desired_t[i] + w1 * motion_{t-1}[i] + w2 * action_{t-1}[i] + b

desired_t   = pose change the robot makes from step t to t+1 (in eval: next planned pose - current pose)
motion_{t-1} = pose change it made in the previous step
Pose changes are world-frame: position difference and rotation vector of R_next * R_current^-1.

A plain per-axis gain (the first eval version) left the commands jerky: it ignores that consecutive
commands are smooth and that the robot keeps pushing down during insertion while barely moving.

    uv run python fit_inverse_controller.py data/gear_quan_Sep15 --out eval/manifeel_gear_inverse_controller.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import zarr

AXES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("eval/manifeel_gear_inverse_controller.json"))
    args = parser.parse_args()

    store = zarr.open(str(args.zarr_path), mode="r")
    state = store["data/state"][:].astype(np.float64)
    action = store["data/action"][:].astype(np.float64)
    ends = store["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])

    rot = Rotation.from_quat(state[:, 3:7])
    motion = np.zeros((len(state), 6))  # motion[t] = pose change from t to t+1 (within an episode)
    motion[:-1, :3] = state[1:, :3] - state[:-1, :3]
    motion[:-1, 3:] = (rot[1:] * rot[:-1].inv()).as_rotvec()

    # Steps t with t-1 and t+1 in the same episode; the episode start (zero motion/action) is covered
    # by the intercept and by the eval client starting from zero history.
    idx = np.concatenate([np.arange(s + 1, e - 1) for s, e in zip(starts, ends)])
    holdout = np.isin(np.searchsorted(ends, idx, side="right"), np.arange(0, len(ends), 5))

    weights, r2 = [], []
    for i in range(6):
        X = np.column_stack([motion[idx, i], motion[idx - 1, i], action[idx - 1, i], np.ones(len(idx))])
        y = action[idx, i]
        w_val, *_ = np.linalg.lstsq(X[~holdout], y[~holdout], rcond=None)
        pred = X[holdout] @ w_val
        r2.append(1 - ((y[holdout] - pred) ** 2).sum() / ((y[holdout] - y[holdout].mean()) ** 2).sum())
        w, *_ = np.linalg.lstsq(X, y, rcond=None)  # final fit on all episodes
        weights.append(w.tolist())

    result = {
        "features": ["desired_motion", "previous_motion", "previous_action", "bias"],
        "axes": AXES,
        "weights": weights,
        "heldout_r2": [round(float(v), 4) for v in r2],
        "dataset": args.zarr_path.name,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for name, w, v in zip(AXES, weights, r2):
        print(f"{name:7s} w_desired {w[0]:9.2f}  w_prev_motion {w[1]:9.2f}  w_prev_action {w[2]:6.3f}"
              f"  bias {w[3]:7.4f}  held-out R2 {v:.2f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
