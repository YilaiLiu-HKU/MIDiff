#!/usr/bin/env bash
set -euo pipefail

# MIDiff training parameters for the midiff checkpoint family.
# Evidence:
# - ckpt/midiff contains model/ema/opt checkpoints
#   every 2000 steps, including ema_0.9999_048000.pt.
# - log-rank001..005 indicate an MPI multi-rank run.
# - data/our.csv is real/reference data, not a checkpoint output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${ROOT_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}
PYTHON=${PYTHON:-python}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5}
MPI_N=${MPI_N:-6}

DATA_DIR=${DATA_DIR:-${ROOT_DIR}/cgasf}
SAVE_DIR=${SAVE_DIR:-${ROOT_DIR}/ckpt/midiff}

IMAGE_SIZE=${IMAGE_SIZE:-256}
NUM_CHANNELS=${NUM_CHANNELS:-128}
NUM_RES_BLOCKS=${NUM_RES_BLOCKS:-3}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-1000}
NOISE_SCHEDULE=${NOISE_SCHEDULE:-cosine}
LR=${LR:-5e-4}

# Per-rank batch size. With MPI_N=6 this gives global_batch=24.
BATCH_SIZE=${BATCH_SIZE:-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
LOG_INTERVAL=${LOG_INTERVAL:-10}

cd "${ROOT_DIR}"

CMD=(
  "${PYTHON}" train_midiff.py
  --data_dir "${DATA_DIR}"
  --image_size "${IMAGE_SIZE}"
  --num_channels "${NUM_CHANNELS}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --diffusion_steps "${DIFFUSION_STEPS}"
  --noise_schedule "${NOISE_SCHEDULE}"
  --learn_sigma True
  --class_cond False
  --rescale_learned_sigmas False
  --rescale_timesteps False
  --lr "${LR}"
  --batch_size "${BATCH_SIZE}"
  --save_dir "${SAVE_DIR}"
  --special_weight 1.0
  --attention_type triple
  --save_interval "${SAVE_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
)

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
if [[ "${MPI_N}" -gt 1 ]]; then
  exec mpiexec -n "${MPI_N}" "${CMD[@]}"
else
  exec "${CMD[@]}"
fi
