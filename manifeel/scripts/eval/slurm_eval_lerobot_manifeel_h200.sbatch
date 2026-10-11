#!/bin/bash -l
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --job-name=lerobot_manifeel_eval
#SBATCH --export=NONE
#SBATCH --exclude=h24-04   # Isaac Gym segfaults on this node

# Closed-loop evaluation of a LeRobot policy checkpoint (e.g. GR00T N1.7 + one-step drifting head from
# slurm_train_gr00t_drifting_manifeel_h200.sbatch) in ManiFeel's Isaac Gym (TacSL) simulator, with videos.
# LeRobot policy server (Python 3.12, the drifting LeRobot fork, eval/run_lerobot_server.py) + ManiFeel client
# (Python 3.8, eval/eval_lerobot_manifeel.py) using ManiFeel's own env runner; the ManiFeel side is set up
# exactly like slurm_eval_manifeel_h200.sbatch (GR00T eval).
#
# Submit:      cd <WORKSPACE>/manifeel && sbatch --export=NONE,STEP=010000 scripts/eval/slurm_eval_lerobot_manifeel_h200.sbatch
# Other run:   sbatch --export=NONE,RUN_NAME=gr00t_drif_ov_manifeel_20k,STEP=010000 scripts/eval/slurm_eval_lerobot_manifeel_h200.sbatch
# Quick test:  sbatch --export=NONE,STEP=005000,N_TEST=2,N_TEST_VIS=2,MAX_STEPS=50 scripts/eval/slurm_eval_lerobot_manifeel_h200.sbatch
# Results:     runs_lerobot/<RUN_NAME>/eval_manifeel/<STEP>/eval_result.json

unset SLURM_EXPORT_ENV
set -euo pipefail

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

RUN_NAME="${RUN_NAME:-gr00t_drifting_manifeel_20k}"
STEP="${STEP:-last}"                      # checkpoint folder: 005000, 010000, ... or last
N_TEST="${N_TEST:-50}"                    # ManiFeel's default number of test episodes
N_TEST_VIS="${N_TEST_VIS:-10}"            # how many of them get a video
MAX_STEPS="${MAX_STEPS:-500}"             # ManiFeel's vision_wrist default
TASK_TEXT="${TASK_TEXT:-assemble the gear onto the shaft}"   # must match training
ISAACGYM_CFG="${ISAACGYM_CFG:-isaacgym_config_gear.yaml}"
# Unique per job: several single-GPU eval jobs can land on the same node
PORT="${PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"

WORKSPACE="/hnvme/workspace/y300bb11-isaag-groot"
MANIFEEL_DIR="$WORKSPACE/manifeel"
CKPT_PATH="$MANIFEEL_DIR/runs_lerobot/$RUN_NAME/checkpoints/$STEP/pretrained_model"
OUT_DIR="$MANIFEEL_DIR/runs_lerobot/$RUN_NAME/eval_manifeel/$STEP"
UV="$HOME/.local/bin/uv"

LEROBOT_COMMIT="c36a91abf5af28a3954a15474e2508dd842e121f"   # same as training
MANIFEEL_COMMIT="ebfa9e1784848903f642ba1b5233f42511a6de62"
ISAACGYMENVS_COMMIT="1a61da683cf1485f3307684c740771ea5e842b39"
DIFFUSION_POLICY_COMMIT="5ba07ac6661db573af695b419a7947ecb704690f"

export UV_CACHE_DIR="$TMPDIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TMPDIR/uv-python"
export HF_HOME="$TMPDIR/hf-home"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"   # Cosmos-Reason2-2B download
export PIP_CACHE_DIR="$TMPDIR/pip-cache"
export PYTHONUNBUFFERED=1

log() { echo "[$(date +%H:%M:%S)] $*"; }
clone() { git init -q "$3"; git -C "$3" fetch -q --depth 1 "$1" "$2"; git -C "$3" checkout -q FETCH_HEAD; }

[ -f "$CKPT_PATH/config.json" ] || { echo "No checkpoint at $CKPT_PATH" >&2; exit 1; }
mkdir -p "$OUT_DIR"

# --- 1. LeRobot policy server (same environment as the training job)
log "[setup] LeRobot drifting fork @ ${LEROBOT_COMMIT:0:7}"
LEROBOT="$TMPDIR/lerobot"
clone https://github.com/RealManShao/lerobot "$LEROBOT_COMMIT" "$LEROBOT"
VENV="$TMPDIR/lerobot-venv"
"$UV" venv -q --python 3.12 "$VENV"
(cd "$LEROBOT" && "$UV" pip install -q --python "$VENV/bin/python" -e ".[groot]" pyzmq msgpack msgpack-numpy)
log "[server] starting on port $PORT with $CKPT_PATH"
"$VENV/bin/python" "$MANIFEEL_DIR/eval/run_lerobot_server.py" --checkpoint "$CKPT_PATH" --device cuda:0 --port "$PORT" \
    > "$OUT_DIR/lerobot_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# --- 2. ManiFeel environment (Python 3.8)
log "[setup] ManiFeel env"
SRC="$TMPDIR/manifeel_src"
ENV="$TMPDIR/manifeel-env"
mkdir -p "$SRC" "$TMPDIR/micromamba"
tar -xzf "$MANIFEEL_DIR/assets/IsaacGym_Preview_TacSL_Package.tar.gz" -C "$SRC"
clone https://github.com/purdue-mars/manifeel "$MANIFEEL_COMMIT" "$SRC/manifeel"
clone https://github.com/purdue-mars/manifeel-isaacgymenvs "$ISAACGYMENVS_COMMIT" "$SRC/manifeel-isaacgymenvs"
clone https://github.com/real-stanford/diffusion_policy "$DIFFUSION_POLICY_COMMIT" "$SRC/diffusion_policy"
# -f: fail on HTTP errors instead of piping an error page into tar (seen with parallel jobs)
curl -fLs --retry 5 --retry-all-errors -o "$TMPDIR/micromamba.tar.bz2" \
    https://micro.mamba.pm/api/micromamba/linux-64/latest
tar -xjf "$TMPDIR/micromamba.tar.bz2" -C "$TMPDIR/micromamba" bin/micromamba
MAMBA_ROOT_PREFIX="$TMPDIR/micromamba" "$TMPDIR/micromamba/bin/micromamba" create -q -y \
    -p "$ENV" -c conda-forge python=3.8 pip libglu
PIP="$ENV/bin/python -m pip"
$PIP install -q -e "$SRC/IsaacGym_Preview_TacSL_Package/isaacgym/python"
$PIP install -q -e "$SRC/manifeel-isaacgymenvs" -e "$SRC/diffusion_policy" -e "$SRC/manifeel"
$PIP install -q wandb==0.12.21 dill==0.3.9 tqdm==4.67.1 av==12.3.0 numpy==1.23.3 \
    opencv-python==4.10.0.84 zarr==2.16.1 einops==0.4.1 huggingface-hub==0.25.0 \
    diffusers==0.11.1 pandas==2.0.3 numba==0.56.4 rtree==1.3.0 \
    pyzmq msgpack msgpack-numpy ninja

# --- 3. Wait for the server
for _ in $(seq 1 120); do
    grep -q "Server is ready" "$OUT_DIR/lerobot_server.log" && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "LeRobot server died, see $OUT_DIR/lerobot_server.log" >&2; exit 1; }
    sleep 5
done
grep -q "Server is ready" "$OUT_DIR/lerobot_server.log" || { echo "LeRobot server not ready after 10 min" >&2; exit 1; }
log "[server] ready: $(grep "Loaded" "$OUT_DIR/lerobot_server.log" | tail -1)"

# --- 4. Evaluate in Isaac Gym
log "[eval] n_test=$N_TEST n_test_vis=$N_TEST_VIS max_steps=$MAX_STEPS -> $OUT_DIR"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions" WANDB_MODE=disabled
cd "$SRC/manifeel"
python "$MANIFEEL_DIR/eval/eval_lerobot_manifeel.py" \
    --manifeel-root "$SRC/manifeel" --output-dir "$OUT_DIR" \
    --isaacgym-cfg "$ISAACGYM_CFG" --task "$TASK_TEXT" --port "$PORT" \
    --n-test "$N_TEST" --n-test-vis "$N_TEST_VIS" --max-steps "$MAX_STEPS"
log "[done] results in $OUT_DIR"
