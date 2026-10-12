#!/bin/bash -l
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=gr00t_manifeel_eval
#SBATCH --export=NONE

# Closed-loop evaluation of a fine-tuned GR00T model in ManiFeel's Isaac Gym (TacSL) simulator,
# with videos. GR00T policy server (Python 3.12) + ManiFeel client (Python 3.8) using ManiFeel's
# own env runner, comparable to their vision-only baseline.
#
# Isaac Gym runs directly on the node, not in ManiFeel's manifeel.sif: that image ships NVIDIA GL
# libraries for driver 535, which clash with Helma's 595 driver and segfault on sim creation.
# The nodes provide Vulkan/GL for 595; only libGLU comes from conda-forge.
#
# Submit:      cd <WORKSPACE>/manifeel && sbatch scripts/eval/slurm_eval_manifeel_h200.sbatch
# Quick test:  sbatch --export=NONE,N_TEST=2,N_TEST_VIS=2,MAX_STEPS=50 scripts/eval/slurm_eval_manifeel_h200.sbatch
# Other model: sbatch --export=NONE,MODEL_SUBDIR=checkpoint-5000 scripts/eval/slurm_eval_manifeel_h200.sbatch
# EEF model:   sbatch --export=NONE,RUN_NAME=gr00t_manifeel_gear_eef,ACTION_FORMAT=eef_pose scripts/eval/slurm_eval_manifeel_h200.sbatch
# Render 480:  sbatch --export=NONE,RENDER_SIZE=480,... scripts/eval/slurm_eval_manifeel_h200.sbatch   (results: <model>_render480)

unset SLURM_EXPORT_ENV
set -euo pipefail

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

RUN_NAME="${RUN_NAME:-gr00t_manifeel_gear}"
MODEL_SUBDIR="${MODEL_SUBDIR:-}"          # empty = final model in the run folder
N_TEST="${N_TEST:-50}"                    # ManiFeel's default number of test episodes
N_TEST_VIS="${N_TEST_VIS:-10}"            # how many of them get a video
MAX_STEPS="${MAX_STEPS:-500}"             # ManiFeel's vision_wrist default
TASK_TEXT="${TASK_TEXT:-assemble the gear onto the shaft}"   # must match training
ISAACGYM_CFG="${ISAACGYM_CFG:-isaacgym_config_gear.yaml}"
ACTION_FORMAT="${ACTION_FORMAT:-delta}"   # eef_pose for models trained on the *_eef dataset
REPLAN_STEPS="${REPLAN_STEPS:-8}"         # eef_pose: env steps per model query (1 = replan every step)
ENSEMBLE_M="${ENSEMBLE_M:-}"              # eef_pose: temporal ensembling weight decay, e.g. 0.3 (empty = off)
GRIPPER_WIDTH="${GRIPPER_WIDTH:-}"        # eef_pose models with a gripper column (co-training): 0.033
RENDER_SIZE="${RENDER_SIZE:-256}"         # camera resolution of ManiFeel's renderer (ManiFeel default: 256)
# Unique per job: several single-GPU eval jobs can land on the same node
PORT="${PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"

WORKSPACE="/hnvme/workspace/y300bb11-isaag-groot"
MANIFEEL_DIR="$WORKSPACE/manifeel"
MODEL_DIR="$MANIFEEL_DIR/runs/$RUN_NAME${MODEL_SUBDIR:+/$MODEL_SUBDIR}"
# eef_pose results get the controller setting in the folder name, so variants don't overwrite each other
if [ "$ACTION_FORMAT" = "eef_pose" ]; then
    [ -n "$ENSEMBLE_M" ] && DEFAULT_TAG="ens${ENSEMBLE_M}" || DEFAULT_TAG="replan${REPLAN_STEPS}"
    EVAL_TAG="${EVAL_TAG:-$DEFAULT_TAG}"
fi
# Non-default render resolutions get their own result folder, e.g. checkpoint-10000_render480
[ "$RENDER_SIZE" != "256" ] && EVAL_TAG="${EVAL_TAG:+${EVAL_TAG}_}render${RENDER_SIZE}"
OUT_DIR="$MANIFEEL_DIR/runs/$RUN_NAME/eval_manifeel/${MODEL_SUBDIR:-final}${EVAL_TAG:+_$EVAL_TAG}"
UV="$HOME/.local/bin/uv"

GR00T_COMMIT="51d4c89f72fda44cbf77285c6a8114b52676b8a1"
MANIFEEL_COMMIT="ebfa9e1784848903f642ba1b5233f42511a6de62"
ISAACGYMENVS_COMMIT="1a61da683cf1485f3307684c740771ea5e842b39"
DIFFUSION_POLICY_COMMIT="5ba07ac6661db573af695b419a7947ecb704690f"

export UV_CACHE_DIR="$TMPDIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TMPDIR/uv-python"
export HF_HOME="$TMPDIR/hf-home"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export PIP_CACHE_DIR="$TMPDIR/pip-cache"
export PYTHONUNBUFFERED=1

log() { echo "[$(date +%H:%M:%S)] $*"; }
clone() { git init -q "$3"; git -C "$3" fetch -q --depth 1 "$1" "$2"; git -C "$3" checkout -q FETCH_HEAD; }

[ -f "$MODEL_DIR/config.json" ] || { echo "No model at $MODEL_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

# --- 1. GR00T policy server
log "[setup] GR00T @ ${GR00T_COMMIT:0:7}"
clone https://github.com/NVIDIA/Isaac-GR00T "$GR00T_COMMIT" "$TMPDIR/Isaac-GR00T"
module load cuda/12.8.0
(cd "$TMPDIR/Isaac-GR00T" && "$UV" sync --python 3.12 --frozen)
log "[server] starting on port $PORT with $MODEL_DIR"
(cd "$TMPDIR/Isaac-GR00T" && "$UV" run --frozen python gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_DIR" --embodiment-tag NEW_EMBODIMENT --device cuda:0 --port "$PORT") \
    > "$OUT_DIR/gr00t_server.log" 2>&1 &
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

# Render resolution: camera image_size + observation buffer sizes (TacSLTaskGear.yaml) and the
# observation shape metadata (vision_wrist.yaml) of the job's ManiFeel copy
if [ "$RENDER_SIZE" != "256" ]; then
    CFG="$SRC/manifeel/manifeel/config/task"
    sed -i -e "s/\[ 256, 256, 3 \]/[ $RENDER_SIZE, $RENDER_SIZE, 3 ]/" \
           -e "s/image_size: \[256, 256\]/image_size: [$RENDER_SIZE, $RENDER_SIZE]/" "$CFG/TacSLTaskGear.yaml"
    sed -i -e "s/\[3, 256, 256\]/[3, $RENDER_SIZE, $RENDER_SIZE]/" "$CFG/vision_wrist.yaml"
    log "[setup] render size $RENDER_SIZE: $(grep -c "$RENDER_SIZE, $RENDER_SIZE" "$CFG/TacSLTaskGear.yaml") camera settings patched"
fi

# --- 3. Wait for the server
for _ in $(seq 1 120); do
    grep -q "Server is ready" "$OUT_DIR/gr00t_server.log" && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "GR00T server died, see $OUT_DIR/gr00t_server.log" >&2; exit 1; }
    sleep 5
done
grep -q "Server is ready" "$OUT_DIR/gr00t_server.log" || { echo "GR00T server not ready after 10 min" >&2; exit 1; }
log "[server] ready"

# --- 4. Evaluate in Isaac Gym
log "[eval] n_test=$N_TEST n_test_vis=$N_TEST_VIS max_steps=$MAX_STEPS -> $OUT_DIR"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions" WANDB_MODE=disabled
cd "$SRC/manifeel"
python "$MANIFEEL_DIR/eval/eval_gr00t_manifeel.py" \
    --manifeel-root "$SRC/manifeel" --output-dir "$OUT_DIR" \
    --isaacgym-cfg "$ISAACGYM_CFG" --task "$TASK_TEXT" --port "$PORT" \
    --n-test "$N_TEST" --n-test-vis "$N_TEST_VIS" --max-steps "$MAX_STEPS" \
    --action-format "$ACTION_FORMAT" --replan-steps "$REPLAN_STEPS" \
    ${ENSEMBLE_M:+--ensemble-m "$ENSEMBLE_M"} ${GRIPPER_WIDTH:+--gripper-width "$GRIPPER_WIDTH"}
log "[done] results in $OUT_DIR"
