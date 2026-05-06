#!/usr/bin/env bash
# =============================================================================
# submit_job_vertex.sh — Submit batch inference jobs to Vertex AI (Gemini)
#
# Loops over all requested WINDOWS × MODES combinations, prints a recap
# banner, and asks for confirmation before submitting anything.
#
# Usage:
#   bash scripts/submit_job_vertex.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# PARAMETERS — edit these
# -----------------------------------------------------------------------------

# Gemini model to run inference with
VERTEX_MODEL="gemini-3-flash-preview"

# GCP project ID (or set VERTEX_PROJECT in your .env)
PROJECT=""

# Vertex AI API location
LOCATION="global"

# GCS bucket name (without gs://) (or set GCS_BUCKET in your .env)
BUCKET="bucket-lic"

# Window positions to submit (space-separated subset of: 0.0 0.25 0.5 0.75 1.0)
WINDOWS=(0.0 0.25 0.5 0.75 1.0)

# Placement modes to submit (space-separated subset of: block sparse)
MODES=(sparse)

# Thinking level: MINIMAL | LOW | MEDIUM | HIGH (default for Gemini 3)
THINKING_LEVEL="HIGH"

# Distractor sampling seed — must match a seed_<N>/ folder produced by augment_final_dataset.py
SEED=42

# Number of options in the augmented dataset
N_OPTIONS=500

# Run on the original 10-option baseline (ignores WINDOWS / MODES / SEED / N_OPTIONS)
BASELINE=false

# Limit number of questions per job (leave empty for the full dataset)
LIMIT=""

# Base output directory (a timestamped subfolder is created per job)
OUTPUT_DIR="out/completions"

# -----------------------------------------------------------------------------
# (no edits needed below this line)
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Load .env so VERTEX_PROJECT / GCS_BUCKET are available if not set above
if [[ -f ".env" ]]; then
    set -o allexport
    source .env
    set +o allexport
fi

[[ -n "${PROJECT}" ]] && export VERTEX_PROJECT="${PROJECT}"
[[ -n "${BUCKET}" ]]  && export GCS_BUCKET="${BUCKET}"

LIMIT_FLAG=""
[[ -n "${LIMIT}" ]] && LIMIT_FLAG="--limit ${LIMIT}"

BASELINE_FLAG=""
[[ "${BASELINE}" == "true" ]] && BASELINE_FLAG="--baseline"

PROJECT_FLAG=""
[[ -n "${PROJECT}" ]] && PROJECT_FLAG="--project ${PROJECT}"

# Build the list of jobs to submit
declare -a JOBS=()
if [[ "${BASELINE}" == "true" ]]; then
    JOBS+=("baseline")
else
    for W in "${WINDOWS[@]}"; do
        for M in "${MODES[@]}"; do
            JOBS+=("window=${W}  mode=${M}")
        done
    done
fi

N_JOBS=${#JOBS[@]}

# ── Banner ────────────────────────────────────────────────────────────────────
BOLD='\033[1m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RESET='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║          VERTEX AI BATCH INFERENCE SUBMISSION PLAN           ║${RESET}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "model"          "${VERTEX_MODEL}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "project"        "${VERTEX_PROJECT:-from .env}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "location"       "${LOCATION}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "bucket"         "gs://${GCS_BUCKET:-from .env}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "thinking_level" "${THINKING_LEVEL}"
if [[ "${BASELINE}" == "true" ]]; then
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "dataset"        "baseline (10 options)"
else
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "seed"           "${SEED}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "n_options"      "${N_OPTIONS}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "windows"        "${WINDOWS[*]}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "modes"          "${MODES[*]}"
fi
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "limit"          "${LIMIT:-none (full dataset)}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "output_dir"     "${OUTPUT_DIR}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
echo -e "${CYAN}║${RESET}  ${BOLD}Jobs to submit: ${N_JOBS}${RESET}"
echo -e "${CYAN}║${RESET}"
for i in "${!JOBS[@]}"; do
    printf "${CYAN}║${RESET}    %2d.  %s\n" "$((i+1))" "${JOBS[$i]}"
done
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Confirmation ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}Submit all ${N_JOBS} job(s) above? [y/N]${RESET} \c"
read -r CONFIRM
if [[ ! "${CONFIRM}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""

# ── Submission loop ───────────────────────────────────────────────────────────
SUBMITTED=0
FAILED=0

if [[ "${BASELINE}" == "true" ]]; then
    echo -e "${BOLD}── Submitting: baseline ──────────────────────────────────────────${RESET}"
    if python3 src/run_inference_vertex_batch.py \
        --vertex-model      "${VERTEX_MODEL}" \
        --location          "${LOCATION}" \
        --thinking-level    "${THINKING_LEVEL}" \
        --output-dir        "${OUTPUT_DIR}" \
        ${PROJECT_FLAG} \
        ${LIMIT_FLAG} \
        ${BASELINE_FLAG}; then
        SUBMITTED=$((SUBMITTED + 1))
        echo -e "${GREEN}  ✓ submitted${RESET}"
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ failed"
    fi
else
    JOB_NUM=0
    for W in "${WINDOWS[@]}"; do
        for M in "${MODES[@]}"; do
            JOB_NUM=$((JOB_NUM + 1))
            echo -e "${BOLD}── [${JOB_NUM}/${N_JOBS}] window=${W}  mode=${M} ────────────────────────────────${RESET}"
            if python3 src/run_inference_vertex_batch.py \
                --vertex-model      "${VERTEX_MODEL}" \
                --location          "${LOCATION}" \
                --thinking-level    "${THINKING_LEVEL}" \
                --seed              "${SEED}" \
                --n-options         "${N_OPTIONS}" \
                --window            "${W}" \
                --mode              "${M}" \
                --output-dir        "${OUTPUT_DIR}" \
                ${PROJECT_FLAG} \
                ${LIMIT_FLAG}; then
                SUBMITTED=$((SUBMITTED + 1))
                echo -e "${GREEN}  ✓ submitted${RESET}"
            else
                FAILED=$((FAILED + 1))
                echo "  ✗ failed (continuing to next job)"
            fi
            echo ""
        done
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo -e "  Submitted : ${GREEN}${SUBMITTED}${RESET} / ${N_JOBS}"
[[ ${FAILED} -gt 0 ]] && echo "  Failed    : ${FAILED}"
echo -e "${CYAN}══════════════════════════════════════════════════════════════${RESET}"
