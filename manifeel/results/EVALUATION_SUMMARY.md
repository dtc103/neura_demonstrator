# Evaluation Summary — ManiFeel Gear Assembly (+ LIBERO)

*Status: 2026-10-09. All ManiFeel numbers: closed loop in ManiFeel's Isaac Gym (TacSL) simulator with ManiFeel's own
env runner, 50 episodes, one seed (100000), ≤ 500 steps, vision only (wrist camera + EEF state, no tactile).
Uncertainty of a single 50-episode run: roughly ± 7 percentage points.*

## At a glance

| Approach | Best ManiFeel success |
|---|---|
| ManiFeel Diffusion Policy, vision only *(paper, 3 seeds)* | *63 % ± 2 %* |
| ManiFeel Diffusion Policy + tactile *(paper, 3 seeds)* | *66 % ± 1 %* |
| **GR00T N1.7, delta actions, ManiFeel data** | **60 %** |
| GR00T N1.7, delta actions, ManiFeel + REASSEMBLE | 52 % |
| GR00T N1.7 VLM + one-step drifting head (arXiv 2609.18108), delta actions | 50 % |
| GR00T N1.7, EEF poses, ManiFeel data | 36 % |
| GR00T N1.7, EEF poses, ManiFeel + REASSEMBLE | 34 % |
| Drift-Based Policy (DBP, 1-step), delta actions | 32 % |
| GR00T N1.7, EEF poses, REASSEMBLE only (zero-shot) | 0 % |
| *LIBERO-Spatial, GR00T N1.7 (for reference)* | *96.5 % (NVIDIA: 97.5 %)* |

## 1. GR00T N1.7 — delta actions (ManiFeel's own action space)

Actions = normalized EEF delta commands (Δpos ≤ 1 cm, Δrot ≤ 0.05 rad), 16-step chunk, 8 steps executed per call.

| Training data | Train / render res. | Run | 5k | 10k | 15k | 20k |
|---|---|---|---|---|---|---|
| ManiFeel | 256 / 256 | `gr00t_manifeel_gear` (10k schedule) | – | 54 % | – | – |
| ManiFeel | 256 / 256 | `gr00t_manifeel_gear_20k` | **60 %** | **60 %** | 52 % | 52 % |
| ManiFeel | 480 / 480 | `gr00t_manifeel_gear_20k_img480_v2` | 58 % | 54 % | 50 % | 52 % |
| ManiFeel | 512 / 512 | `gr00t_manifeel_gear_20k_img512_v2` | 56 % | **60 %** | 56 % | 52 % |
| ManiFeel + REASSEMBLE (delta pseudo-actions) | 256 / 256 | `gr00t_manifeel_reassemble_delta_20k` | 46 % | 50 % | 52 % | 46 % |
| ManiFeel + REASSEMBLE (delta pseudo-actions) | 480 / 480 | `gr00t_manifeel_reassemble_delta_20k_img480_v2` | 22 % | 42 % | 50 % | 42 % |

- Earlier checkpoints (5k–10k) are best; later ones overfit the 50 demos.
- Higher image resolution brings no gain.
- Co-training with REASSEMBLE costs ~8–10 points (real vs. sim images, reconstructed pseudo-actions, other controller).

## 2. GR00T N1.7 — EEF poses (xyz + rot6d, converted back to ManiFeel commands)

The client turns predicted poses into ManiFeel delta commands (fixed gains or an inverse controller fitted on the demos).

| Training data | Run | Checkpoint | Conversion | New plan every | Ensembling | Success |
|---|---|---|---|---|---|---|
| ManiFeel | `gr00t_manifeel_gear_eef_10k` | 5k | fixed gains | 8 steps | – | 0 % |
| ManiFeel | `gr00t_manifeel_gear_eef_10k` | 10k | fixed gains | 8 steps | – | 0 % |
| ManiFeel | `gr00t_manifeel_gear_eef_10k` | 10k | inverse controller | 8 steps | – | 0 % |
| ManiFeel | `gr00t_manifeel_gear_eef_10k` | 10k | inverse controller | 4 steps | – | 0 % |
| ManiFeel | `gr00t_manifeel_gear_eef_10k` | 10k | inverse controller | **every step** | – | **36 %** |
| ManiFeel | `gr00t_manifeel_gear_eef_10k` | 10k | inverse controller | every step | m = 1.0 | 8 % |
| ManiFeel | `gr00t_manifeel_gear_eef_10k` | 10k | inverse controller | every step | m = 0.3 | 2 % |
| ManiFeel | `gr00t_manifeel_gear_eef_10k` | 10k | inverse controller | every step | m = 0.1 | 8 % |
| ManiFeel + REASSEMBLE (+ gripper) | `gr00t_gear_eef_cotrain_10k` | 5k | inverse controller | every step | – | 8 % |
| ManiFeel + REASSEMBLE (+ gripper) | `gr00t_gear_eef_cotrain_10k` | 10k | inverse controller | every step | – | 34 % |
| REASSEMBLE only, 480 px (+ gripper) | `gr00t_reassemble_eef_20k_img480` | 5k / 10k / 15k / 20k | inverse controller | every step | – | 0 / 0 / 0 / 0 % |

- Pose actions only work when replanning every step: the impedance controller lags behind the plan, so older targets point to the wrong place.
- Pose trajectories cannot express the downward pushing force of the insertion; the inverse controller recovers it only partly.
- REASSEMBLE-only (task text "insert the medium gear onto the shaft") does not transfer zero-shot to the simulator.

## 3. Drift-Based Policy (DBP) — one-step generation, delta actions

`runs_dbp/dbp_manifeel_256`: DBPO's ImageDBPPolicy (ResNet18 + 1-D U-Net, 263 M params), wrist 256×256 + state,
2 observation frames, 16-step chunk, 8 executed, one network evaluation per chunk. Trained on 45 episodes
(5 held out), 4 × H200, batch 64, 500 epochs (34 min). EMA weights.

| Checkpoint | val MSE (open loop) | val MSE best of 8 | Success |
|---|---|---|---|
| epoch 50 | 0.0285 | 0.0282 | 26 % |
| **epoch 100** | 0.0298 | 0.0296 | **32 %** |
| epoch 250 | 0.0312 | 0.0310 | 18 % |
| epoch 499 | 0.0314 | 0.0309 | 18 % |

- Best validation error at epoch 10 (0.0234, not saved); overfits afterwards, success drops from epoch 100 on.
- Open-loop validation error does not predict closed-loop success well (epoch 100 beats epoch 50).
- The 8 proposals per observation barely differ (best-of-8 only 2–4 % better), which limits proposal ranking (GPC-Rank).

## 4. GR00T N1.7 VLM + one-step drifting action head (arXiv 2609.18108)

`runs_lerobot/gr00t_drifting_manifeel_20k`: the paper's LeRobot code (RealManShao/lerobot, feat/drif-ov), policy
`drifting`: frozen Cosmos-Reason2-2B VLM (no GR00T robot pretraining) + a one-step transformer head trained from
scratch with the Implicit Drifting Policy objective, deterministic, 1 network evaluation per chunk. Same data
and schedule as the GR00T delta runs (wrist 256 + state, 4 GPUs x 32 = 128, 20k steps; 1 h 20 min).

| Checkpoint | 5k | 10k | 15k | 20k |
|---|---|---|---|---|
| Success | 22 % | 44 % | **50 %** | **50 %** |

- Clearly better than DBP (32 %), ~10 points below standard GR00T (60 %), comparable to the paper's LIBERO gap.
- Improves until 15–20k and plateaus, unlike standard GR00T, which peaks at 5–10k and then overfits.

## 5. DriftWorld world model (frame prediction, no success rate)

`runs_driftworld/driftworld_manifeel_256_mn` (32 x H200, stopped at epoch ~113 / step 16k): action-conditioned
world model, front + wrist 256x256, 4 history frames + 6-D delta actions -> next 2 frames. It predicts images,
not actions, so it is evaluated offline on the 5 held-out episodes against "copy the last real frame"
(`driftworld/eval_manifeel.py`; images in `results/driftworld_eval/`).

| Mode | View | PSNR model / copy | SSIM model / copy | LPIPS model / copy (lower = better) |
|---|---|---|---|---|
| one step, t+1 | front | **31.3** / 29.0 | 0.935 / **0.966** | 0.022 / **0.010** |
| one step, t+1 | wrist | **30.5** / 29.4 | 0.943 / **0.986** | 0.042 / **0.007** |
| autoregressive, whole episode | front | **18.1** / 16.9 | 0.771 / **0.866** | 0.166 / **0.138** |
| autoregressive, whole episode | wrist | **19.2** / 16.9 | 0.840 / **0.911** | 0.177 / **0.135** |

- It has learned the action-conditioned motion: the gear and gripper move in the right direction (PSNR beats copy).
- But frames get noisy and the robot dissolves after ~20-30 autoregressive steps (SSIM / LPIPS worse than copy).
- Consistent with the flat training loss: a pixel-space drifting loss gives little signal; not usable as a ranker yet.

## 6. Inference latency: standard GR00T vs. drifting head

Same H200, same job, 100 calls with different ManiFeel observations, median, full policy call as in the eval
servers (`slurm_benchmark_latency_h200.sbatch`; raw numbers in `results/latency/`). Checkpoints: GR00T 10k (60 %),
drifting 20k (50 %).

| Batch | Model | End to end | Backbone | Action head | Other |
|---|---|---|---|---|---|
| 1 (one robot) | GR00T N1.7 (flow matching, 4 steps) | 85.4 ms | 29.2 ms | 53.7 ms | 2.6 ms |
| 1 (one robot) | GR00T VLM + drifting head (1 step) | **40.3 ms (2.1x)** | 24.9 ms | **8.5 ms (6.3x)** | 6.8 ms |
| 50 (ManiFeel eval) | GR00T N1.7 | 213.0 ms | 104.4 ms | 55.2 ms | 53.3 ms |
| 50 (ManiFeel eval) | GR00T VLM + drifting head | 184.1 ms (1.16x) | 122.1 ms | 9.9 ms (5.6x) | 52.2 ms |

- Action head: 1,621 M parameters (GR00T DiT) vs. 386 M (drifting); same 1,524 M backbone.
- For a single robot the call is 2x faster end to end (~25 Hz instead of ~12 Hz): enough to replan every 100 ms step.
- With large batches the shared backbone and pre-processing dominate and the advantage mostly disappears.
