#!/bin/bash -l
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:4   # Helma never shares nodes: always use all 4 GPUs
#SBATCH --time=12:00:00
#SBATCH --job-name=gr00t_drifting
#SBATCH --export=NONE

# GR00T N1.7 with a one-step drifting action head on ManiFeel gear demos, as in
# "Technical Report: One-Step Drifting Action Heads for GR00T N1.7" (arXiv 2609.18108).
# Code: the paper's LeRobot fork (github.com/RealManShao/lerobot, branch feat/drif-ov), cloned unchanged:
#   policy "drifting": GR00T N1.7's VLM (Cosmos-Reason2-2B, frozen) + one-step transformer head trained with the
#                      Implicit Drifting Policy objective (arXiv 2606.01098), deterministic, 1 network evaluation
#   policy "drif_ov":  the paper's overlap-conditioned extension for asynchronous chunk replacement; in
#                      synchronous execution like ManiFeel's runner it gets no prefix and acts like "drifting"
# As in the paper, the head starts from scratch and the VLM is the plain Cosmos-Reason2-2B (no GR00T robot
# pretraining). Data: the same ManiFeel LeRobot dataset as the GR00T delta runs (wrist 256 + 7-D state, 6-D
# delta actions), converted to LeRobot v3.0 inside the job. Same schedule as gr00t_manifeel_gear_20k
# (4 GPUs x 32 = 128, 20k steps, checkpoints every 5k).
#
#   cd <WORKSPACE>/manifeel && sbatch scripts/train/slurm_train_gr00t_drifting_manifeel_h200.sbatch
#   overlap variant:  sbatch --export=NONE,POLICY_TYPE=drif_ov,RUN_NAME=gr00t_drif_ov_manifeel_20k scripts/train/slurm_train_gr00t_drifting_manifeel_h200.sbatch
# Resubmitting with the same RUN_NAME resumes from the last checkpoint.
# Evaluate: slurm_eval_lerobot_manifeel_h200.sbatch

unset SLURM_EXPORT_ENV
set -euo pipefail

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

POLICY_TYPE="${POLICY_TYPE:-drifting}"
RUN_NAME="${RUN_NAME:-gr00t_drifting_manifeel_20k}"
STEPS="${STEPS:-20000}"
BATCH_PER_GPU="${BATCH_PER_GPU:-32}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"            # action horizon, as GR00T on ManiFeel
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"     # executed per call, as ManiFeel's runner
USE_WANDB="${USE_WANDB:-1}"
SEED="${SEED:-1000}"                      # the paper's seed

WORKSPACE="/hnvme/workspace/y300bb11-isaag-groot"
MANIFEEL_DIR="$WORKSPACE/manifeel"
DATA_ZIP="${DATA_ZIP:-$MANIFEEL_DIR/gear_quan_Sep15_lerobot_v2.1.zip}"
OUT_DIR="$MANIFEEL_DIR/runs_lerobot/$RUN_NAME"
LEROBOT_COMMIT="c36a91abf5af28a3954a15474e2508dd842e121f"   # feat/drif-ov, 2026-10-08
UV="$HOME/.local/bin/uv"
NUM_GPUS="$(nvidia-smi -L | wc -l)"

export UV_CACHE_DIR="$TMPDIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TMPDIR/uv-python"
export HF_HOME="$TMPDIR/hf-home"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"   # Cosmos-Reason2-2B download
export HF_LEROBOT_HOME="$TMPDIR/lerobot-home"
export PYTHONUNBUFFERED=1

log() { echo "[$(date +%H:%M:%S)] $*"; }

[ -r "$DATA_ZIP" ] || { echo "Missing dataset $DATA_ZIP" >&2; exit 1; }
mkdir -p "$MANIFEEL_DIR/runs_lerobot"

log "[setup] LeRobot drifting fork @ ${LEROBOT_COMMIT:0:7}"
LEROBOT="$TMPDIR/lerobot"
git init -q "$LEROBOT"
git -C "$LEROBOT" fetch -q --depth 1 https://github.com/RealManShao/lerobot "$LEROBOT_COMMIT"
git -C "$LEROBOT" checkout -q FETCH_HEAD
VENV="$TMPDIR/lerobot-venv"
"$UV" venv -q --python 3.12 "$VENV"
(cd "$LEROBOT" && "$UV" pip install -q --python "$VENV/bin/python" -e ".[groot,training]")

log "[data] $DATA_ZIP -> LeRobot v3.0"
DATA_DIR="$TMPDIR/data/manifeel_gear_v30"
mkdir -p "$TMPDIR/data/unzip"
unzip -q "$DATA_ZIP" -d "$TMPDIR/data/unzip"
mv "$(dirname "$(dirname "$(find "$TMPDIR/data/unzip" -path '*/meta/info.json' | head -1)")")" "$DATA_DIR"
"$VENV/bin/python" "$LEROBOT/src/lerobot/scripts/convert_dataset_v21_to_v30.py" \
    --repo-id=local/manifeel_gear --root="$DATA_DIR" --push-to-hub=false
grep -q '"codebase_version": "v3.0"' "$DATA_DIR/meta/info.json" || { echo "Dataset conversion failed" >&2; exit 1; }

if [ -d "$OUT_DIR/checkpoints/last" ]; then
    log "[train] resuming $RUN_NAME from $(readlink -f "$OUT_DIR/checkpoints/last")"
    ARGS=(--config_path="$OUT_DIR/checkpoints/last/pretrained_model/train_config.json" --resume=true
          --dataset.root="$DATA_DIR")
else
    log "[train] $RUN_NAME: $POLICY_TYPE, $NUM_GPUS GPUs x $BATCH_PER_GPU, $STEPS steps -> $OUT_DIR"
    ARGS=(--dataset.repo_id=local/manifeel_gear --dataset.root="$DATA_DIR" --dataset.video_backend=pyav
          --policy.type="$POLICY_TYPE" --policy.use_bf16=true --policy.device=cuda --policy.push_to_hub=false
          --policy.chunk_size="$CHUNK_SIZE" --policy.n_action_steps="$N_ACTION_STEPS"
          --batch_size="$BATCH_PER_GPU" --steps="$STEPS" --save_freq="$SAVE_FREQ" --log_freq=100
          --num_workers=8 --seed="$SEED" --output_dir="$OUT_DIR" --job_name="$RUN_NAME"
          --wandb.enable="$([ "$USE_WANDB" = 1 ] && echo true || echo false)"
          --wandb.project=gr00t-drifting-manifeel --wandb.entity=jan_kerner-university-of-tuebingen)
fi
cd "$LEROBOT"
"$VENV/bin/torchrun" --standalone --nproc_per_node="$NUM_GPUS" -m lerobot.scripts.lerobot_train "${ARGS[@]}"
log "[done] checkpoints in $OUT_DIR/checkpoints"
