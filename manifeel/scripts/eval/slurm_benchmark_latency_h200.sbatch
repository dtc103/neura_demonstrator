#!/bin/bash -l
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --job-name=latency_gr00t_vs_drifting
#SBATCH --export=NONE

# Inference latency: standard GR00T N1.7 (flow-matching DiT, 4 steps) vs. GR00T N1.7 VLM + one-step drifting
# head (arXiv 2609.18108), on the same GPU in the same job, with the same ManiFeel observations and the same policy
# calls as the eval servers. Batch size 1 (one robot) and 50 (ManiFeel's parallel eval). Splits each call into
# backbone / action head / rest. Uses new files only (eval/benchmark_latency_*.py, eval/latency_utils.py).
#
#   cd <WORKSPACE>/manifeel && sbatch scripts/eval/slurm_benchmark_latency_h200.sbatch
#   other checkpoints: sbatch --export=NONE,GR00T_MODEL=runs/<run>/checkpoint-5000,DRIFTING_CKPT=runs_lerobot/<run>/checkpoints/015000/pretrained_model ...
# Results: latency/<JOB_ID>/{gr00t.json, drifting.json, comparison.md}

unset SLURM_EXPORT_ENV
set -euo pipefail

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

WORKSPACE="/hnvme/workspace/y300bb11-isaag-groot"
MANIFEEL_DIR="$WORKSPACE/manifeel"
GR00T_MODEL="$MANIFEEL_DIR/${GR00T_MODEL:-runs/gr00t_manifeel_gear_20k/checkpoint-10000}"      # 60 %
DRIFTING_CKPT="$MANIFEEL_DIR/${DRIFTING_CKPT:-runs_lerobot/gr00t_drifting_manifeel_20k/checkpoints/020000/pretrained_model}"  # 50 %
NPZ="$MANIFEEL_DIR/manifeel_gear_dbp_256.npz"    # wrist frames + EEF states of the demos
REPEATS="${REPEATS:-100}"
OUT_DIR="$MANIFEEL_DIR/latency/${SLURM_JOB_ID:-local}"
UV="$HOME/.local/bin/uv"
GR00T_COMMIT="51d4c89f72fda44cbf77285c6a8114b52676b8a1"      # same as training / eval
LEROBOT_COMMIT="c36a91abf5af28a3954a15474e2508dd842e121f"    # same as training / eval

export UV_CACHE_DIR="$TMPDIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TMPDIR/uv-python"
export HF_HOME="$TMPDIR/hf-home"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export PYTHONUNBUFFERED=1

log() { echo "[$(date +%H:%M:%S)] $*"; }
clone() { git init -q "$3"; git -C "$3" fetch -q --depth 1 "$1" "$2"; git -C "$3" checkout -q FETCH_HEAD; }

[ -f "$GR00T_MODEL/config.json" ] || { echo "No GR00T model at $GR00T_MODEL" >&2; exit 1; }
[ -f "$DRIFTING_CKPT/config.json" ] || { echo "No drifting checkpoint at $DRIFTING_CKPT" >&2; exit 1; }
[ -r "$NPZ" ] || { echo "Missing $NPZ" >&2; exit 1; }
mkdir -p "$OUT_DIR"
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv | tee "$OUT_DIR/gpu.txt"

# --- 1. Standard GR00T N1.7 (Isaac-GR00T env)
log "[setup] Isaac-GR00T @ ${GR00T_COMMIT:0:7}"
clone https://github.com/NVIDIA/Isaac-GR00T "$GR00T_COMMIT" "$TMPDIR/Isaac-GR00T"
module load cuda/12.8.0
(cd "$TMPDIR/Isaac-GR00T" && "$UV" sync -q --python 3.12 --frozen)
log "[bench] GR00T N1.7: $GR00T_MODEL"
(cd "$TMPDIR/Isaac-GR00T" && "$UV" run --frozen python "$MANIFEEL_DIR/eval/benchmark_latency_gr00t.py" \
    --model-path "$GR00T_MODEL" --npz "$NPZ" --out "$OUT_DIR/gr00t.json" --repeats "$REPEATS")

# --- 2. GR00T N1.7 VLM + drifting head (LeRobot fork env)
log "[setup] LeRobot drifting fork @ ${LEROBOT_COMMIT:0:7}"
clone https://github.com/RealManShao/lerobot "$LEROBOT_COMMIT" "$TMPDIR/lerobot"
VENV="$TMPDIR/lerobot-venv"
"$UV" venv -q --python 3.12 "$VENV"
(cd "$TMPDIR/lerobot" && "$UV" pip install -q --python "$VENV/bin/python" -e ".[groot]")
log "[bench] drifting: $DRIFTING_CKPT"
"$VENV/bin/python" "$MANIFEEL_DIR/eval/benchmark_latency_lerobot.py" \
    --checkpoint "$DRIFTING_CKPT" --npz "$NPZ" --out "$OUT_DIR/drifting.json" --repeats "$REPEATS"

# --- 3. Table
python3 "$MANIFEEL_DIR/eval/compare_latency.py" "$OUT_DIR/gr00t.json" "$OUT_DIR/drifting.json" \
    --out "$OUT_DIR/comparison.md"
log "[done] $OUT_DIR/comparison.md"
