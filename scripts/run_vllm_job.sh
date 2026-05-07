#!/usr/bin/env bash
# =============================================================================
# run_vllm_job.sh — Run local vLLM inference jobs on augmented MMLU-Pro
#
# Loops over all requested WINDOWS × MODES combinations, prints a recap
# banner, and asks for confirmation before starting anything.
#
# Usage:
#   bash scripts/run_vllm_job.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# PARAMETERS — edit these
# -----------------------------------------------------------------------------

# HuggingFace model ID
MODEL="Qwen/Qwen3-27B-FP8"

# Window positions to run (space-separated subset of: 0.0 0.25 0.5 0.75 1.0)
WINDOWS=(0.0 0.25 0.5 0.75 1.0)

# Placement modes to run (space-separated subset of: block sparse)
MODES=(block sparse)

# Distractor sampling seed — must match a seed_<N>/ folder produced by augment_final_dataset.py
SEED=42

# Number of options in the augmented dataset
N_OPTIONS=1000

# Run on the original 10-option baseline (ignores WINDOWS / MODES / SEED / N_OPTIONS)
BASELINE=false

# Limit number of questions per job (leave empty for the full dataset)
LIMIT=""

# Maximum tokens to generate per response
MAX_TOKENS=32000

# Number of GPUs for tensor parallelism (1 = single GPU)
TENSOR_PARALLEL_SIZE=1

# Override vLLM context window (leave empty to use model default)
# Increase if you get OOM or context-too-long errors with large option counts.
MAX_MODEL_LEN=""

# Disable Qwen3 thinking mode (pass --no-thinking; thinking is on by default)
ENABLE_THINKING=true

# Base output directory (a timestamped subfolder is created per job)
OUTPUT_DIR="out/completions"

# -----------------------------------------------------------------------------
# (no edits needed below this line)
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LIMIT_FLAG=""
[[ -n "${LIMIT}" ]] && LIMIT_FLAG="--limit ${LIMIT}"

BASELINE_FLAG=""
[[ "${BASELINE}" == "true" ]] && BASELINE_FLAG="--baseline"

MAX_MODEL_LEN_FLAG=""
[[ -n "${MAX_MODEL_LEN}" ]] && MAX_MODEL_LEN_FLAG="--max-model-len ${MAX_MODEL_LEN}"

THINKING_FLAG=""
[[ "${ENABLE_THINKING}" == "false" ]] && THINKING_FLAG="--no-thinking"

# Build the list of jobs
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
echo -e "${CYAN}║               LOCAL vLLM INFERENCE PLAN                      ║${RESET}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "model"              "${MODEL}"
if [[ "${BASELINE}" == "true" ]]; then
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "dataset"            "baseline (10 options)"
else
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "seed"               "${SEED}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "n_options"          "${N_OPTIONS}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "windows"            "${WINDOWS[*]}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "modes"              "${MODES[*]}"
fi
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "max_tokens"         "${MAX_TOKENS}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "tensor_parallel"    "${TENSOR_PARALLEL_SIZE}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "max_model_len"      "${MAX_MODEL_LEN:-model default}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "enable_thinking"    "${ENABLE_THINKING}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "limit"              "${LIMIT:-none (full dataset)}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "output_dir"         "${OUTPUT_DIR}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
echo -e "${CYAN}║${RESET}  ${BOLD}Jobs to run: ${N_JOBS}${RESET}"
echo -e "${CYAN}║${RESET}"
for i in "${!JOBS[@]}"; do
    printf "${CYAN}║${RESET}    %2d.  %s\n" "$((i+1))" "${JOBS[$i]}"
done
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Confirmation ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}Run all ${N_JOBS} job(s) above? [y/N]${RESET} \c"
read -r CONFIRM
if [[ ! "${CONFIRM}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""

# ── Inference loop ────────────────────────────────────────────────────────────
SUCCEEDED=0
FAILED=0

if [[ "${BASELINE}" == "true" ]]; then
    echo -e "${BOLD}── Running: baseline ─────────────────────────────────────────────${RESET}"
    if python3 src/run_inference_vllm.py \
        --model-name            "${MODEL}" \
        --max-tokens            "${MAX_TOKENS}" \
        --tensor-parallel-size  "${TENSOR_PARALLEL_SIZE}" \
        --output-dir            "${OUTPUT_DIR}" \
        ${MAX_MODEL_LEN_FLAG} \
        ${THINKING_FLAG} \
        ${LIMIT_FLAG} \
        ${BASELINE_FLAG}; then
        SUCCEEDED=$((SUCCEEDED + 1))
        echo -e "${GREEN}  ✓ done${RESET}"
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
            if python3 src/run_inference_vllm.py \
                --model-name            "${MODEL}" \
                --seed                  "${SEED}" \
                --n-options             "${N_OPTIONS}" \
                --window                "${W}" \
                --mode                  "${M}" \
                --max-tokens            "${MAX_TOKENS}" \
                --tensor-parallel-size  "${TENSOR_PARALLEL_SIZE}" \
                --output-dir            "${OUTPUT_DIR}" \
                ${MAX_MODEL_LEN_FLAG} \
                ${THINKING_FLAG} \
                ${LIMIT_FLAG}; then
                SUCCEEDED=$((SUCCEEDED + 1))
                echo -e "${GREEN}  ✓ done${RESET}"
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
echo -e "  Completed : ${GREEN}${SUCCEEDED}${RESET} / ${N_JOBS}"
[[ ${FAILED} -gt 0 ]] && echo "  Failed    : ${FAILED}"
echo -e "${CYAN}══════════════════════════════════════════════════════════════${RESET}"
