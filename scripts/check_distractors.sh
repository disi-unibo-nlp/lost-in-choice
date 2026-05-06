#!/usr/bin/env bash
# =============================================================================
# check_distractors.sh — Submit a Together AI batch job to judge distractor
#                        safety for one (seed, window, n) configuration.
#
# Run this once per seed.  Because distractors are window-invariant and
# block/sparse share the same distractor set, the single file
#   data/bench/mmlu-pro-augmented-<N_OPTIONS>/seed_<SEED>/window_<WINDOW>/block.jsonl
# covers all 990 unique distractors for that seed.  The resulting
# flagged_distractors.jsonl can then be applied to every (n, window, mode)
# variant via apply_distractor_fixes.sh.
#
# Usage:
#   bash scripts/check_distractors.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# PARAMETERS — edit these
# -----------------------------------------------------------------------------

# Seed to check (run this script once per seed)
SEED=42

# Window position of the file to submit (any window is equivalent;
# 0.00 is the default)
WINDOW=0.00

# Number of options in the augmented dataset (use the largest N to cover all
# distractors; smaller N variants are strict subsets)
N_OPTIONS=1000

# Model used to judge distractors
MODEL="openai/gpt-oss-120b"

# Limit number of questions (leave empty for the full dataset)
LIMIT=""

# -----------------------------------------------------------------------------
# (no edits needed below this line)
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

WINDOW_LABEL=$(printf "window_%.2f" "${WINDOW}")
DATA="data/bench/mmlu-pro-augmented-${N_OPTIONS}/seed_${SEED}/${WINDOW_LABEL}/block.jsonl"

LIMIT_FLAG=""
[[ -n "${LIMIT}" ]] && LIMIT_FLAG="--limit ${LIMIT}"

BOLD='\033[1m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RESET='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║              DISTRACTOR CHECK SUBMISSION                     ║${RESET}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "seed"       "${SEED}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "n_options"  "${N_OPTIONS}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "window"     "${WINDOW}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "mode"       "block"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "model"      "${MODEL}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "limit"      "${LIMIT:-none (full dataset)}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "data"       "${DATA}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

if [[ ! -f "${DATA}" ]]; then
    echo "ERROR: data file not found: ${DATA}"
    echo "Run augment_final_dataset_v4.py first."
    exit 1
fi

echo -e "${YELLOW}Submit distractor check job? [y/N]${RESET} \c"
read -r CONFIRM
if [[ ! "${CONFIRM}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${BOLD}── Submitting distractor check ──────────────────────────────────${RESET}"

python3 preprocess/check_distractors.py \
    --data  "${DATA}"  \
    --model "${MODEL}" \
    ${LIMIT_FLAG}

echo ""
echo -e "${GREEN}✓ Job submitted.${RESET}"
echo "Once the job completes, retrieve results with:"
echo "  python preprocess/retrieve_distractor_check.py \\"
echo "    --output-dir out/distractor_check/... \\"
echo "    --data ${DATA}"
echo ""
echo "Then apply fixes to all variants with:"
echo "  bash scripts/apply_distractor_fixes.sh"
