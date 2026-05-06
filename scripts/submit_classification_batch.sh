#!/usr/bin/env bash
# =============================================================================
# submit_classification_batch.sh — Classify reasoning traces against taxonomy
#
# Wraps preprocess/analysis/submit_classification_batch.py.
# Edit the PARAMETERS section, then run:
#
#   bash scripts/submit_classification_batch.sh
#   bash scripts/submit_classification_batch.sh --dry-run
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# PARAMETERS — edit these
# -----------------------------------------------------------------------------

# Path to the frozen taxonomy produced by synthesize_taxonomy.py
TAXONOMY_PATH="out/open_coding_batches/2026-04-30_11-10-35/taxonomy_draft.json"

# Path to open_codes.jsonl (used for few-shot exemplar auto-selection).
# Set to "" to run zero-shot.
OPEN_CODES_PATH="out/open_coding_batches/2026-04-30_11-10-35/open_codes.jsonl"

# Number of few-shot exemplars per taxonomy code (0 = zero-shot)
N_EXEMPLARS=0

# Samples per (model × n_options × window × mode) condition
N_PER_CONDITION=50

# Together AI judge model
JUDGE_MODEL="qwen/Qwen3.5-9B"

# Base output directory (a timestamped subfolder is created automatically)
OUTPUT_DIR="out/classification_batches"

# Random seed for stratified sampling
SEED=42

# Optional filters (leave empty to use all models / all tiers from MODEL_POOL)
# Space-separated, e.g.: MODELS="gpt-oss-120b Qwen3.5-9B"
MODELS=""
# Space-separated, e.g.: N_OPTIONS="1000"
N_OPTIONS=""

# -----------------------------------------------------------------------------
# (no edits needed below this line)
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Pass --dry-run through if given as argument to this script
DRY_RUN_FLAG=""
for arg in "$@"; do
    [[ "${arg}" == "--dry-run" ]] && DRY_RUN_FLAG="--dry-run"
done

# Validate required files exist
if [[ ! -f "${TAXONOMY_PATH}" ]]; then
    echo "ERROR: taxonomy not found: ${TAXONOMY_PATH}"
    exit 1
fi

BOLD='\033[1m'
CYAN='\033[1;36m'
RESET='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║           STRATEGY CLASSIFICATION BATCH                      ║${RESET}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
printf "${CYAN}║${RESET}  %-24s %-35s ${CYAN}║${RESET}\n" "judge model"      "${JUDGE_MODEL}"
printf "${CYAN}║${RESET}  %-24s %-35s ${CYAN}║${RESET}\n" "taxonomy"         "${TAXONOMY_PATH}"
printf "${CYAN}║${RESET}  %-24s %-35s ${CYAN}║${RESET}\n" "open codes"       "${OPEN_CODES_PATH:-none (zero-shot)}"
printf "${CYAN}║${RESET}  %-24s %-35s ${CYAN}║${RESET}\n" "exemplars/code"   "${N_EXEMPLARS}"
printf "${CYAN}║${RESET}  %-24s %-35s ${CYAN}║${RESET}\n" "samples/condition" "${N_PER_CONDITION}"
printf "${CYAN}║${RESET}  %-24s %-35s ${CYAN}║${RESET}\n" "seed"             "${SEED}"
printf "${CYAN}║${RESET}  %-24s %-35s ${CYAN}║${RESET}\n" "models filter"    "${MODELS:-all from MODEL_POOL}"
printf "${CYAN}║${RESET}  %-24s %-35s ${CYAN}║${RESET}\n" "n_options filter" "${N_OPTIONS:-all from MODEL_POOL}"
printf "${CYAN}║${RESET}  %-24s %-35s ${CYAN}║${RESET}\n" "output dir"       "${OUTPUT_DIR}/<timestamp>"
[[ -n "${DRY_RUN_FLAG}" ]] && \
printf "${CYAN}║${RESET}  %-24s %-35s ${CYAN}║${RESET}\n" "mode"             "DRY RUN — no submission"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# Build optional flags
OPEN_CODES_FLAG=""
[[ -n "${OPEN_CODES_PATH}" && -f "${OPEN_CODES_PATH}" ]] && \
    OPEN_CODES_FLAG="--open-codes-path ${OPEN_CODES_PATH}"

MODELS_FLAG=""
[[ -n "${MODELS}" ]] && MODELS_FLAG="--models ${MODELS}"

N_OPTIONS_FLAG=""
[[ -n "${N_OPTIONS}" ]] && N_OPTIONS_FLAG="--n-options ${N_OPTIONS}"

# Run
python3 src/analysis/submit_classification_batch.py \
    --taxonomy-path     "${TAXONOMY_PATH}" \
    --n-exemplars       "${N_EXEMPLARS}" \
    --n-per-condition   "${N_PER_CONDITION}" \
    --judge-model       "${JUDGE_MODEL}" \
    --output-dir        "${OUTPUT_DIR}" \
    --seed              "${SEED}" \
    ${OPEN_CODES_FLAG} \
    ${MODELS_FLAG} \
    ${N_OPTIONS_FLAG} \
    ${DRY_RUN_FLAG}
