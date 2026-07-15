#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

REAL_DATA="${REAL_DATA:-${REPO_DIR}/data/our.csv}"
MIDIFF_DATA="${MIDIFF_DATA:-${REPO_DIR}/exp/results/MIDiff.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/exp/results/cross_variable}"
GPU="${GPU:-0}"

SOURCES=("real" "MIDiff")
TARGETS=("ch1" "ch2" "ch3")
MODEL_TYPES=("MLP" "LSTM" "Transformer" "Mamba")

for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
  for SOURCE in "${SOURCES[@]}"; do
    for TARGET in "${TARGETS[@]}"; do
      "${PYTHON_BIN}" "${REPO_DIR}/exp/run_downstream_cross_variable.py" \
        --real-data "${REAL_DATA}" \
        --midiff-data "${MIDIFF_DATA}" \
        --train-source "${SOURCE}" \
        --target-name "${TARGET}" \
        --model-type "${MODEL_TYPE}" \
        --epochs 100 \
        --batch-size 256 \
        --learning-rate 1e-3 \
        --hidden-dim 128 \
        --num-layers 2 \
        --dropout 0.2 \
        --nhead 4 \
        --seed 42 \
        --gpu "${GPU}" \
        --output-dir "${OUTPUT_DIR}"
    done
  done
done
