#!/usr/bin/env bash
# =============================================================================
# retrieve_distractor_check.sh — Retrieve results from completed distractor
#                                check batch jobs (one per seed).
#
# Set OUTPUT_DIR_SEED_* to the timestamped directories printed by
# check_distractors.sh.  job_id.txt is read automatically from each directory.
#
# Usage:
#   bash scripts/retrieve_distractor_check.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# PARAMETERS — edit these
# -----------------------------------------------------------------------------

# Timestamped output directories created by check_distractors.sh, one per seed.
# Set to "" to skip that seed.
OUTPUT_DIR_SEED_42="out/distractor_check/openai_gpt-oss-120b/block/2026-04-24_15-04-12"
OUTPUT_DIR_SEED_123="out/distractor_check/openai_gpt-oss-120b/block/REPLACE_WITH_TIMESTAMP_123"
OUTPUT_DIR_SEED_456="out/distractor_check/openai_gpt-oss-120b/block/REPLACE_WITH_TIMESTAMP_456"

# Seeds to retrieve (must match the directories above)
SEEDS=(42) # 123 456)

# Parameters matching check_distractors.sh (used to reconstruct the --data path)
N_OPTIONS=1000
WINDOW=0.00

# -----------------------------------------------------------------------------
# (no edits needed below this line)
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

WINDOW_LABEL=$(printf "window_%.2f" "${WINDOW}")

declare -A OUTPUT_DIR
OUTPUT_DIR[42]="${OUTPUT_DIR_SEED_42}"
OUTPUT_DIR[123]="${OUTPUT_DIR_SEED_123}"
OUTPUT_DIR[456]="${OUTPUT_DIR_SEED_456}"

BOLD='\033[1m'
CYAN='\033[1;36m'
GREEN='\033[1;32m'
RESET='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║           RETRIEVE DISTRACTOR CHECK RESULTS                  ║${RESET}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "seeds"      "${SEEDS[*]}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "n_options"  "${N_OPTIONS}"
printf "${CYAN}║${RESET}  %-28s %-31s ${CYAN}║${RESET}\n" "window"     "${WINDOW}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
echo -e "${CYAN}║${RESET}  Output directories:"
for S in "${SEEDS[@]}"; do
    DIR="${OUTPUT_DIR[$S]}"
    if [[ -z "${DIR}" ]]; then
        printf "${CYAN}║${RESET}    seed %-5s  %s\n" "${S}" "[skipped]"
    elif [[ -d "${DIR}" ]]; then
        printf "${CYAN}║${RESET}    seed %-5s  %s\n" "${S}" "${DIR}"
    else
        printf "${CYAN}║${RESET}    seed %-5s  %s\n" "${S}" "NOT FOUND: ${DIR}"
    fi
done
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# Validate all directories exist before running anything
MISSING=0
for S in "${SEEDS[@]}"; do
    DIR="${OUTPUT_DIR[$S]}"
    [[ -z "${DIR}" ]] && continue
    if [[ ! -d "${DIR}" ]]; then
        echo "ERROR: output dir not found for seed ${S}: ${DIR}"
        MISSING=$((MISSING + 1))
    fi
done
if [[ ${MISSING} -gt 0 ]]; then
    echo "Fix the paths above and re-run."
    exit 1
fi

# -----------------------------------------------------------------------------
# Retrieve per seed
# -----------------------------------------------------------------------------
SUCCEEDED=0
FAILED=0

for S in "${SEEDS[@]}"; do
    DIR="${OUTPUT_DIR[$S]}"
    [[ -z "${DIR}" ]] && echo "  seed ${S}: skipped" && continue

    DATA="data/bench/mmlu-pro-augmented-${N_OPTIONS}/seed_${S}/${WINDOW_LABEL}/block.jsonl"

    echo -e "${BOLD}── Seed ${S} ──────────────────────────────────────────────────────${RESET}"
    echo "  output-dir : ${DIR}"
    echo "  data       : ${DATA}"

    if [[ ! -f "${DATA}" ]]; then
        echo "  ERROR: data file not found: ${DATA}"
        FAILED=$((FAILED + 1))
        continue
    fi

    if python3 preprocess/retrieve_distractor_check.py \
        --output-dir "${DIR}" \
        --data       "${DATA}"; then
        SUCCEEDED=$((SUCCEEDED + 1))
        echo -e "${GREEN}  ✓ seed ${S} done — results in ${DIR}/flagged_distractors.jsonl${RESET}"
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
if [[ ${SUCCEEDED} -gt 0 ]]; then
    echo ""
    echo "  Next step — apply fixes to all variants:"
    echo "    Fill in FLAGGED_DIR_SEED_* in scripts/apply_distractor_fixes.sh"
    echo "    using the output directories above, then run:"
    echo "      bash scripts/apply_distractor_fixes.sh"
fi
echo -e "${CYAN}══════════════════════════════════════════════════════════════${RESET}"
