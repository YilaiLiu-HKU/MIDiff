#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_NPZ="${INPUT_NPZ:-${REPO_DIR}/ckpt/midiff/ema_0.9999_048000.pt_samples_3000x256x160x1.npz}"
DATASET_NPZ="${DATASET_NPZ:-${REPO_DIR}/data/dataset_original_npz/all_users_data_with6cluster.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/exp/results}"
RECOVERED_NPZ="${RECOVERED_NPZ:-${OUTPUT_DIR}/recovered_midiff.npz}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/MIDiff.csv}"

NPZ_KEY="${NPZ_KEY:-arr_0}"
MEAN_FACTOR="${MEAN_FACTOR:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TARGET_H="${TARGET_H:-192}"
TARGET_W="${TARGET_W:-140}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${REPO_DIR}/gasf_cross_conversion_inverse.py" \
  --input-npz "${INPUT_NPZ}" \
  --output-npz "${RECOVERED_NPZ}" \
  --dataset-npz "${DATASET_NPZ}" \
  --npz-key "${NPZ_KEY}" \
  --mean-factor "${MEAN_FACTOR}" \
  --num-workers "${NUM_WORKERS}" \
  --target-h "${TARGET_H}" \
  --target-w "${TARGET_W}"

"${PYTHON_BIN}" "${REPO_DIR}/changetocsv.py" \
  --input-npz "${RECOVERED_NPZ}" \
  --output-csv "${OUTPUT_CSV}"

echo "Done: ${OUTPUT_CSV}"
