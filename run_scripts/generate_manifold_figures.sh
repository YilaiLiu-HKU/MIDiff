#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

REAL_DATA="${REAL_DATA:-${REPO_DIR}/data/our.csv}"
MIDIFF_DATA="${MIDIFF_DATA:-${REPO_DIR}/exp/results/MIDiff.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/exp/results/manifold}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEED="${SEED:-42}"
STYLEB_ONLY="${STYLEB_ONLY:-1}"

if [[ -n "${SYNTH_FILES:-}" ]]; then
  read -r -a SYNTH_FILE_ARGS <<< "${SYNTH_FILES}"
else
  SYNTH_FILE_ARGS=(
    "${MIDIFF_DATA}"
  )
fi

"${PYTHON_BIN}" "${REPO_DIR}/exp/visualize_data_quality.py" \
  --real-file "${REAL_DATA}" \
  --synth-files "${SYNTH_FILE_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --seed "${SEED}" \
  $([[ "${STYLEB_ONLY}" == "1" ]] && printf '%s' "--styleb-only")
