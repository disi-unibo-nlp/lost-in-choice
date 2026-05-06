#!/usr/bin/env bash
# =============================================================================
# apply_distractor_fixes.sh — Apply distractor replacements from a completed
#                             check_distractors run to all (n, window, mode)
#                             variants, for every seed.
#
# Point FLAGGED_DIR_SEED_* to the output directories produced by
# retrieve_distractor_check.py (one per seed).  Each directory must contain
# a flagged_distractors.jsonl file.
#
# Writes cleaned variants to OUTPUT_BASE_OUT-<N>/ (separate from the original
# augmented data).
#
# Usage:
#   bash scripts/apply_distractor_fixes.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# PARAMETERS — edit these
# -----------------------------------------------------------------------------

# Output directory of retrieve_distractor_check.py, one per seed.
# Set to "" to skip that seed.
FLAGGED_DIR_SEED_42="out/distractor_check/merged/block"
FLAGGED_DIR_SEED_123=""
FLAGGED_DIR_SEED_456=""

# Seeds to process (must match the directories above)
SEEDS=(42)

# N-option sizes to update
N_OPTIONS=(1000)

# Source augmented data base (as passed to augment_final_dataset_v4.py)
OUTPUT_BASE="data/bench/mmlu-pro-augmented"

# Destination base for the cleaned data (separate from the originals)
OUTPUT_BASE_OUT="data/bench/mmlu-pro-augmented-clean"

# Pass --dry-run to preview without writing files
DRY_RUN=""
# DRY_RUN="--dry-run"

# -----------------------------------------------------------------------------
# (no edits needed below this line)
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BOLD='\033[1m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RESET='\033[0m'

# Map seed → flagged dir
declare -A FLAGGED_DIR
FLAGGED_DIR[42]="${FLAGGED_DIR_SEED_42}"
FLAGGED_DIR[123]="${FLAGGED_DIR_SEED_123}"
FLAGGED_DIR[456]="${FLAGGED_DIR_SEED_456}"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║              APPLY DISTRACTOR FIXES                         ║${RESET}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "seeds"           "${SEEDS[*]}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "n_options"       "${N_OPTIONS[*]}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "source base"     "${OUTPUT_BASE}-<N>"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "dest base"       "${OUTPUT_BASE_OUT}-<N>"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "dry_run"         "${DRY_RUN:-(no, files will be written)}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
echo -e "${CYAN}║${RESET}  Flagged distractor files:"
for S in "${SEEDS[@]}"; do
    DIR="${FLAGGED_DIR[$S]}"
    JSONL="${DIR}/flagged_distractors.jsonl"
    if [[ -z "${DIR}" ]]; then
        printf "${CYAN}║${RESET}    seed %-5s  %s\n" "${S}" "[skipped]"
    elif [[ -f "${JSONL}" ]]; then
        printf "${CYAN}║${RESET}    seed %-5s  %s\n" "${S}" "${JSONL}"
    else
        printf "${CYAN}║${RESET}    seed %-5s  %s\n" "${S}" "NOT FOUND: ${JSONL}"
    fi
done
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# Validate all flagged files exist before asking for confirmation
MISSING=0
for S in "${SEEDS[@]}"; do
    DIR="${FLAGGED_DIR[$S]}"
    [[ -z "${DIR}" ]] && continue
    JSONL="${DIR}/flagged_distractors.jsonl"
    if [[ ! -f "${JSONL}" ]]; then
        echo "ERROR: flagged_distractors.jsonl not found for seed ${S}: ${JSONL}"
        MISSING=$((MISSING + 1))
    fi
done
if [[ ${MISSING} -gt 0 ]]; then
    echo "Fix the paths above and re-run."
    exit 1
fi

echo -e "${YELLOW}Apply fixes to all variants? [y/N]${RESET} \c"
read -r CONFIRM
if [[ ! "${CONFIRM}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""

# -----------------------------------------------------------------------------
# Apply fixes per seed
# -----------------------------------------------------------------------------
SUCCEEDED=0
FAILED=0

for S in "${SEEDS[@]}"; do
    DIR="${FLAGGED_DIR[$S]}"
    [[ -z "${DIR}" ]] && echo "  seed ${S}: skipped" && continue

    JSONL="${DIR}/flagged_distractors.jsonl"
    echo -e "${BOLD}── Seed ${S} ──────────────────────────────────────────────────────${RESET}"

    if python3 preprocess/apply_distractor_fixes.py \
        --flagged-jsonl     "${JSONL}" \
        --seed              "${S}" \
        --n-options         "${N_OPTIONS[@]}" \
        --output-base       "${OUTPUT_BASE}" \
        --output-base-out   "${OUTPUT_BASE_OUT}" \
        ${DRY_RUN}; then
        SUCCEEDED=$((SUCCEEDED + 1))
        echo -e "${GREEN}  ✓ seed ${S} done${RESET}"
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ seed ${S} failed (continuing)"
    fi
    echo ""
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "${CYAN}══════════════════════════════════════════════════════════════${RESET}"
echo -e "  Seeds completed : ${GREEN}${SUCCEEDED}${RESET} / ${#SEEDS[@]}"
[[ ${FAILED} -gt 0 ]] && echo "  Failed          : ${FAILED}"
if [[ -z "${DRY_RUN}" && ${SUCCEEDED} -gt 0 ]]; then
    echo ""
    echo "  Cleaned data written to:"
    for N in "${N_OPTIONS[@]}"; do
        echo "    ${OUTPUT_BASE_OUT}-${N}/"
    done
fi
echo -e "${CYAN}══════════════════════════════════════════════════════════════${RESET}"
