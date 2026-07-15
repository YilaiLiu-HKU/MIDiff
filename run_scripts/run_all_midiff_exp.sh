#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/exp/results}"

# ---------------------------------------------------------------------------
# Section 1: Convert MIDiff inference NPZ to eval-format CSV.
# Set RUN_INFER_TO_CSV=0 to skip this step.
# Inputs can be overridden with INPUT_NPZ, DATASET_NPZ, OUTPUT_CSV.
# ---------------------------------------------------------------------------
if [[ "${RUN_INFER_TO_CSV:-1}" == "1" ]]; then
  echo "[1/4] Converting inference NPZ to eval CSV"
  OUTPUT_DIR="${INFER_OUTPUT_DIR:-${RESULTS_DIR}}" \
    "${SCRIPT_DIR}/infer_npz_to_eval_csv.sh"
else
  echo "[1/4] Skipping inference NPZ to eval CSV"
fi

# ---------------------------------------------------------------------------
# Section 2: Compute generation metrics for real data vs MIDiff.
# Set RUN_METRICS=0 to skip this step.
# Uses REAL_DATA, MIDIFF_DATA, OUTPUT_XLSX if provided.
# ---------------------------------------------------------------------------
if [[ "${RUN_METRICS:-1}" == "1" ]]; then
  echo "[2/4] Computing generation metrics"
  "${SCRIPT_DIR}/evaluate_midiff_real.sh"
else
  echo "[2/4] Skipping generation metrics"
fi

# ---------------------------------------------------------------------------
# Section 3: Run downstream cross-variable tasks on real and MIDiff.
# Set RUN_DOWNSTREAM=1 to enable; default is off because this is costly.
# Uses GPU, REAL_DATA, MIDIFF_DATA, OUTPUT_DIR if provided.
# ---------------------------------------------------------------------------
if [[ "${RUN_DOWNSTREAM:-0}" == "1" ]]; then
  echo "[3/4] Running downstream cross-variable tasks"
  OUTPUT_DIR="${DOWNSTREAM_OUTPUT_DIR:-${RESULTS_DIR}/cross_variable}" \
    "${SCRIPT_DIR}/run_downstream_midiff_real.sh"
else
  echo "[3/4] Skipping downstream tasks"
fi

# ---------------------------------------------------------------------------
# Section 4: Generate manifold figures, including styleB_base t-SNE and UMAP.
# Set RUN_MANIFOLD=0 to skip this step.
# Uses REAL_DATA, MIDIFF_DATA, OUTPUT_DIR, MAX_SAMPLES, SEED if provided.
# ---------------------------------------------------------------------------
if [[ "${RUN_MANIFOLD:-1}" == "1" ]]; then
  echo "[4/4] Generating manifold figures"
  OUTPUT_DIR="${MANIFOLD_OUTPUT_DIR:-${RESULTS_DIR}/manifold}" \
    "${SCRIPT_DIR}/generate_manifold_figures.sh"
else
  echo "[4/4] Skipping manifold figures"
fi

echo "MIDiff experiment pipeline finished."
