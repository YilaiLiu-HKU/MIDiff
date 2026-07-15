#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

REAL_DATA="${REAL_DATA:-${REPO_DIR}/data/our.csv}"
MIDIFF_DATA="${MIDIFF_DATA:-${REPO_DIR}/exp/results/MIDiff.csv}"
OUTPUT_XLSX="${OUTPUT_XLSX:-${REPO_DIR}/exp/results/eval_midiff_real.xlsx}"

"${PYTHON_BIN}" "${REPO_DIR}/exp/evaluate_generation_metrics.py" \
  --real-file "${REAL_DATA}" \
  --synth-files "${MIDIFF_DATA}" \
  --baseline-synth-file "${MIDIFF_DATA}" \
  --out-xlsx "${OUTPUT_XLSX}" \
  --js-bins 50 \
  --fdds-bins 50 \
  --ac-max-lag 20 \
  --da-test-size 0.3 \
  --da-random-state 0 \
  --da-n-runs 5 \
  --pred-lag 5 \
  --pred-n-runs 5
