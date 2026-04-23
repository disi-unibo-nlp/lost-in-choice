#!/usr/bin/env bash
# =============================================================================
# retrieve_job.sh — Retrieve results from a completed Together AI batch job
#
# Usage:
#   bash scripts/retrieve_job.sh
#
# Set OUTPUT_DIR to the timestamped folder printed by submit_job.sh.
# JOB_ID is optional: if left empty the script reads job_id.txt from
# OUTPUT_DIR automatically.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# PARAMETERS — edit these
# -----------------------------------------------------------------------------

# Directory created by submit_job.sh (contains job_id.txt and run_config.json)
OUTPUT_DIR="out/completions/gpt-oss-120b/mmlu-pro-augmented-1000__window_0.50__sparse/2026-04-16_12-00-00"

# Together AI batch job ID — leave empty to auto-read from job_id.txt
JOB_ID=""

# -----------------------------------------------------------------------------
# (no edits needed below this line)
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Validate output dir
if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "ERROR: OUTPUT_DIR not found: ${OUTPUT_DIR}"
    echo "Set OUTPUT_DIR to the folder printed by submit_job.sh."
    exit 1
fi

# Resolve job ID
if [[ -z "${JOB_ID}" ]]; then
    ID_FILE="${OUTPUT_DIR}/job_id.txt"
    if [[ ! -f "${ID_FILE}" ]]; then
        echo "ERROR: JOB_ID is empty and job_id.txt not found in OUTPUT_DIR."
        echo "Either set JOB_ID above or re-check OUTPUT_DIR."
        exit 1
    fi
    JOB_ID="$(cat "${ID_FILE}")"
    echo "Job ID loaded from ${ID_FILE}: ${JOB_ID}"
fi

# Build optional --job-id flag
JOB_ID_FLAG=""
if [[ -n "${JOB_ID}" ]]; then
    JOB_ID_FLAG="--job-id ${JOB_ID}"
fi

echo "============================================================"
echo "  retrieve_job.sh"
echo "  output dir : ${OUTPUT_DIR}"
echo "  job id     : ${JOB_ID}"
echo "============================================================"
echo ""

python src/retrieve_inference.py \
    --output-dir "${OUTPUT_DIR}" \
    ${JOB_ID_FLAG}
