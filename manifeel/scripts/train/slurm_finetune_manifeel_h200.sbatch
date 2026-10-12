#!/bin/bash -l
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:4   # Helma never shares nodes: always use all 4 GPUs
#SBATCH --time=24:00:00
#SBATCH --job-name=gr00t_manifeel
#SBATCH --export=NONE
# Warn us 10 minutes before the time limit so we can stop cleanly and sync outputs
#SBATCH --signal=B:USR1@600

# Fine-tune GR00T N1.7 on the ManiFeel gear assembly demos (vision-only: wrist camera + EEF state,
# no tactile), converted to GR00T-flavored LeRobot v2.1 by convert_to_lerobot_v2.py.
#
# Everything heavy (repo, Python, venv, uv/HF caches, model weights, dataset, checkpoints)
# lives in node-local $TMPDIR. Only logs, configs and the latest complete checkpoint are
# synced back to <WORKSPACE>/manifeel/runs/<RUN_NAME>.
#
# Expected in <WORKSPACE>/manifeel/:
#   gear_quan_Sep15_lerobot_v2.1.zip   (top-level folder = gear_quan_Sep15_lerobot_v2.1)
#   gr00t_manifeel_gear_config.py      (GR00T modality config, registers NEW_EMBODIMENT)
#
# Submit:           cd <WORKSPACE>/manifeel && sbatch scripts/train/slurm_finetune_manifeel_h200.sbatch
# Short test:       sbatch --export=NONE,MAX_STEPS=200 scripts/train/slurm_finetune_manifeel_h200.sbatch
# Resubmitting the same RUN_NAME resumes from the latest checkpoint in the workspace.
# End-effector pose actions (relative EEF, see convert_to_lerobot_v2.py --action-format eef_pose):
#   sbatch --export=NONE,DATASET_NAME=gear_quan_Sep15_lerobot_v2.1_eef,MODALITY_CONFIG_NAME=gr00t_manifeel_gear_eef_config.py,RUN_NAME=gr00t_manifeel_gear_eef ...
# Co-training ManiFeel + REASSEMBLE (EEF + gripper), equal share per dataset:
#   sbatch --export=NONE,DATASET_NAME=gear_quan_Sep15_lerobot_v2.1_eef_gripper,EXTRA_DATASET_ZIPS=/hnvme/workspace/y300bb11-isaag-groot/manifeel/reassemble_gear_lerobot_v2.1_eef_gripper.zip,DS_WEIGHTS_ALPHA=0,MODALITY_CONFIG_NAME=gr00t_gear_eef_gripper_config.py,RUN_NAME=... ...
# Several checkpoints to compare in the simulator:
#   sbatch --export=NONE,MAX_STEPS=20000,SAVE_STEPS=5000,KEEP_ALL_CHECKPOINTS=1,RUN_NAME=gr00t_manifeel_gear_20k ...

unset SLURM_EXPORT_ENV
set -euo pipefail

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# --- Settings ------------------------------------------------------------------------------
DATASET_NAME="${DATASET_NAME:-gear_quan_Sep15_lerobot_v2.1}"
# Co-training: extra dataset zips (absolute paths, space-separated; top-level folder = zip name), trained
# together with DATASET_NAME under the same embodiment. DS_WEIGHTS_ALPHA (read by finetune.sh) sets the
# sampling weights: len^alpha per dataset, 0 = equal share per dataset.
EXTRA_DATASET_ZIPS="${EXTRA_DATASET_ZIPS:-}"
# Image size GR00T resizes every frame to (shortest edge, after a CROP_FRACTION crop); stored in the
# checkpoint, so evaluation uses it too. Empty = GR00T default (256 -> 8x8 image tokens); 512 -> 16x16.
SHORTEST_IMAGE_EDGE="${SHORTEST_IMAGE_EDGE:-}"
CROP_FRACTION="${CROP_FRACTION:-0.95}"
RUN_NAME="${RUN_NAME:-gr00t_manifeel_gear}"
WORKSPACE="/hnvme/workspace/y300bb11-isaag-groot"
MANIFEEL_DIR="$WORKSPACE/manifeel"
MODALITY_CONFIG="$MANIFEEL_DIR/${MODALITY_CONFIG_NAME:-gr00t_manifeel_gear_config.py}"

# Fail fast if a dataset is not reachable. GPU nodes do not mount /home/vault: copy vault datasets
# into the workspace first.
for zip_path in "$MANIFEEL_DIR/$DATASET_NAME.zip" $EXTRA_DATASET_ZIPS; do
    [ -r "$zip_path" ] || { echo "Dataset not readable on this node: $zip_path (vault is not mounted on GPU nodes; copy it to the workspace)" >&2; exit 1; }
done
UV="$HOME/.local/bin/uv"

GR00T_REPO_URL="https://github.com/NVIDIA/Isaac-GR00T"
GR00T_COMMIT="${GR00T_COMMIT:-51d4c89f72fda44cbf77285c6a8114b52676b8a1}"  # N1.7, same as the LIBERO runs
BASE_MODEL="nvidia/GR00T-N1.7-3B"

# Helma never shares a node, so train on every GPU of the allocation (data parallel, DeepSpeed ZeRO-2).
# The global batch stays 128 (32 per GPU on 4 GPUs), so runs are comparable to the earlier 1-GPU ones.
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MAX_STEPS="${MAX_STEPS:-10000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
(( GLOBAL_BATCH_SIZE % NUM_GPUS == 0 )) || { echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE not divisible by $NUM_GPUS GPUs" >&2; exit 1; }
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
# 1 = keep every checkpoint in the workspace (weights only, no optimizer state) so several
# training lengths can be evaluated; such runs cannot be resumed. 0 = keep only the newest.
KEEP_ALL_CHECKPOINTS="${KEEP_ALL_CHECKPOINTS:-0}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-gr00t-manifeel}"
export WANDB_ENTITY="${WANDB_ENTITY:-jan_kerner-university-of-tuebingen}"
SYNC_EVERY_S=600

# --- Paths: everything in $TMPDIR --------------------------------------------------------------
GR00T_DIR="$TMPDIR/Isaac-GR00T"
DATASET_DIR="$TMPDIR/data/$DATASET_NAME"
OUTPUT_DIR="$TMPDIR/outputs/$RUN_NAME"
WS_OUT="$MANIFEEL_DIR/runs/$RUN_NAME"

export UV_CACHE_DIR="$TMPDIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TMPDIR/uv-python"
export HF_HOME="$TMPDIR/hf-home"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR" "$WS_OUT" "$TMPDIR/data"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 1. Code + environment --------------------------------------------------------------------
log "[setup] cloning Isaac-GR00T @ ${GR00T_COMMIT:0:7} into $GR00T_DIR"
git init -q "$GR00T_DIR"
git -C "$GR00T_DIR" remote add origin "$GR00T_REPO_URL"
git -C "$GR00T_DIR" fetch -q --depth 1 origin "$GR00T_COMMIT"
git -C "$GR00T_DIR" checkout -q FETCH_HEAD
cd "$GR00T_DIR"

# GR00T bug: when fine-tuning from the base checkpoint, Gr00tN1d7Processor.from_pretrained only applies
# a fixed list of override keys, which lacks shortest_image_edge/crop_fraction. --shortest-image-edge
# then lands in the experiment config but the processor silently keeps the checkpoint's 256.
PROC_PY=gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py
perl -i -pe 'if (/override_keys = \[/../\]/) { s/^( *)"max_action_dim",$/$&\n$1"shortest_image_edge",\n$1"crop_fraction",/ }' "$PROC_PY"
sed -n '/override_keys = \[/,/\]/p' "$PROC_PY" | grep -q '"shortest_image_edge"' \
    || { echo "Patching the image-size override into $PROC_PY failed" >&2; exit 1; }

# DeepSpeed runs $CUDA_HOME/bin/nvcc on import; /usr/local/cuda on the nodes has no nvcc.
# 12.8 matches torch's CUDA build. The module also sets CUDA_HOME.
module load cuda/12.8.0

log "[setup] uv sync (Python 3.12 + GR00T deps)"
"$UV" sync --python 3.12 --frozen

# --- 2. Dataset ------------------------------------------------------------------------------
log "[setup] unzipping $MANIFEEL_DIR/$DATASET_NAME.zip"
unzip -q "$MANIFEEL_DIR/$DATASET_NAME.zip" -d "$TMPDIR/data"
DATASET_PATHS="$DATASET_DIR"
for zip_path in $EXTRA_DATASET_ZIPS; do
    log "[setup] unzipping $zip_path"
    unzip -q "$zip_path" -d "$TMPDIR/data"
    DATASET_PATHS="$DATASET_PATHS:$TMPDIR/data/$(basename "$zip_path" .zip)"
done

# --- 3. Video decoding: torchcodec needs FFmpeg 4-7 shared libs ---------------------------------
TEST_VIDEO="$DATASET_DIR/videos/chunk-000/observation.images.wrist/episode_000000.mp4"
can_decode() {
    "$UV" run --frozen python -c \
        "from torchcodec.decoders import VideoDecoder; VideoDecoder('$TEST_VIDEO')[0]" >/dev/null 2>&1
}
if ! can_decode; then
    log "[setup] no usable system FFmpeg, installing ffmpeg<8 from conda-forge into \$TMPDIR"
    mkdir -p "$TMPDIR/micromamba"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C "$TMPDIR/micromamba" bin/micromamba
    MAMBA_ROOT_PREFIX="$TMPDIR/micromamba" "$TMPDIR/micromamba/bin/micromamba" create -q -y \
        -p "$TMPDIR/ffmpeg" -c conda-forge "ffmpeg<8"
    export LD_LIBRARY_PATH="$TMPDIR/ffmpeg/lib:${LD_LIBRARY_PATH:-}"
    can_decode || { echo "torchcodec still cannot decode $TEST_VIDEO" >&2; exit 1; }
fi
log "[setup] video decoding OK"

# --- 4. Model weights (pre-fetch so training starts from a warm cache) ----------------------------
log "[setup] downloading $BASE_MODEL and nvidia/Cosmos-Reason2-2B"
"$UV" run --frozen hf download "$BASE_MODEL" >/dev/null
"$UV" run --frozen hf download nvidia/Cosmos-Reason2-2B >/dev/null

# --- 5. Resume + syncing back to the workspace ------------------------------------------------------
# A checkpoint is complete once GR00T's on_save callback has copied wandb_config.json into it.
latest_complete_ckpt() {
    local ckpt latest=""
    for ckpt in $(ls -d "$1"/checkpoint-* 2>/dev/null | sort -V); do
        [ -f "$ckpt/wandb_config.json" ] && latest="$ckpt"
    done
    echo "$latest"
}

RESUME_ARGS=()
SAVE_ARGS=()
WS_CKPT="$(latest_complete_ckpt "$WS_OUT")"
if [ "$KEEP_ALL_CHECKPOINTS" = "1" ]; then
    SAVE_ARGS=(--save-only-model)
    [ -z "$WS_CKPT" ] || { echo "$WS_OUT already has checkpoints; weights-only runs cannot resume, use a new RUN_NAME" >&2; exit 1; }
elif [ -n "$WS_CKPT" ]; then
    log "[setup] resuming from $WS_CKPT"
    rsync -a --exclude='checkpoint-*' --exclude='.partial' --exclude='open_loop_eval' \
        "$WS_OUT/" "$OUTPUT_DIR/"
    rsync -a "$WS_CKPT" "$OUTPUT_DIR/"
    RESUME_ARGS=(--resume-from-checkpoint)
fi

# Keep only the newest complete checkpoint in the workspace (each is tens of GB),
# or every complete one with KEEP_ALL_CHECKPOINTS=1.
sync_outputs() {
    rsync -a --exclude='checkpoint-*' "$OUTPUT_DIR/" "$WS_OUT/" || true
    local ckpt
    if [ "$KEEP_ALL_CHECKPOINTS" = "1" ]; then
        for ckpt in $(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null); do
            [ -f "$ckpt/wandb_config.json" ] && [ ! -d "$WS_OUT/$(basename "$ckpt")" ] || continue
            rsync -a "$ckpt" "$WS_OUT/.partial/" && mv "$WS_OUT/.partial/$(basename "$ckpt")" "$WS_OUT/" \
                && log "[sync] $(basename "$ckpt") -> $WS_OUT"
        done
        return
    fi
    ckpt="$(latest_complete_ckpt "$OUTPUT_DIR")"
    if [ -n "$ckpt" ] && [ ! -d "$WS_OUT/$(basename "$ckpt")" ]; then
        rsync -a "$ckpt" "$WS_OUT/.partial/" && mv "$WS_OUT/.partial/$(basename "$ckpt")" "$WS_OUT/" \
            && find "$WS_OUT" -maxdepth 1 -name 'checkpoint-*' ! -name "$(basename "$ckpt")" \
                -exec rm -rf {} + \
            && log "[sync] $(basename "$ckpt") -> $WS_OUT"
    fi
}
( while sleep "$SYNC_EVERY_S"; do sync_outputs; done ) &
SYNC_LOOP_PID=$!

# --- 6. Train ------------------------------------------------------------------------------------
# Runs in the background so the USR1 trap can interrupt it before the time limit.
TRAIN_PID=""
on_time_limit_approaching() {
    log "[signal] time limit approaching -- stopping training and syncing"
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID" 2>/dev/null
        for _ in $(seq 1 60); do
            kill -0 "$TRAIN_PID" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$TRAIN_PID" 2>/dev/null && kill -KILL "$TRAIN_PID" 2>/dev/null
    fi
}
trap on_time_limit_approaching USR1

log "[train] datasets=$DATASET_PATHS ds_weights_alpha=${DS_WEIGHTS_ALPHA:-none} image_edge=${SHORTEST_IMAGE_EDGE:-default} gpus=$NUM_GPUS batch=$GLOBAL_BATCH_SIZE accum=$GRAD_ACCUM steps=$MAX_STEPS"
set +e
NUM_GPUS="$NUM_GPUS" MAX_STEPS="$MAX_STEPS" GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
SAVE_STEPS="$SAVE_STEPS" USE_WANDB="$USE_WANDB" \
"$UV" run --frozen bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_PATHS" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --state-dropout-prob 0.2 \
    --wandb-project "$WANDB_PROJECT" \
    "${RESUME_ARGS[@]}" "${SAVE_ARGS[@]}" \
    ${SHORTEST_IMAGE_EDGE:+--shortest-image-edge "$SHORTEST_IMAGE_EDGE" --crop-fraction "$CROP_FRACTION"} \
    -- --gradient_accumulation_steps "$GRAD_ACCUM" &
TRAIN_PID=$!
wait "$TRAIN_PID"
TRAIN_EXIT=$?
# wait returns early when USR1 arrives; wait again for the training process to be gone
kill -0 "$TRAIN_PID" 2>/dev/null && wait "$TRAIN_PID"
set -e
log "[train] finished with exit code $TRAIN_EXIT"

kill "$SYNC_LOOP_PID" 2>/dev/null || true
sync_outputs

# --- 7. Quick open-loop eval on the latest checkpoint --------------------------------------------
EVAL_CKPT="$(latest_complete_ckpt "$OUTPUT_DIR")"
if [ -z "$EVAL_CKPT" ]; then
    log "[eval] no complete checkpoint yet -- skipping"
else
    log "[eval] open-loop eval of $EVAL_CKPT"
    mkdir -p "$WS_OUT/open_loop_eval"
    for traj in 0 1 2; do
        "$UV" run --frozen python gr00t/eval/open_loop_eval.py \
            --model-path "$EVAL_CKPT" \
            --dataset-path "$DATASET_DIR" \
            --embodiment-tag NEW_EMBODIMENT \
            --traj-ids "$traj" \
            --save-plot-path "$WS_OUT/open_loop_eval/$(basename "$EVAL_CKPT")_traj_${traj}.jpeg" \
            2>&1 | tee -a "$WS_OUT/open_loop_eval/results.txt" || true
    done
fi

log "[done] outputs in $WS_OUT"
exit "$TRAIN_EXIT"
