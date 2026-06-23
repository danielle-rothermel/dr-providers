#!/usr/bin/env bash
# Live-test OpenRouter configs used by nl_latents experiment profiles.
#
# Prerequisites:
#   OPENROUTER_API_KEY must be set.
#
# Usage:
#   scripts/live_test_llm_configs.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MESSAGE="Say hello in one short sentence."
TEMPERATURE="0.7"
TOP_P="0.95"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    printf 'ERROR: OPENROUTER_API_KEY is not set.\n' >&2
    exit 1
fi

run_config() {
    local label="$1"
    shift

    printf '\n==> %s\n' "${label}"
    uv run python -m dr_providers.cli \
        --message "${MESSAGE}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        "$@"
}

failures=0

run_config "openrouter/xiaomi/mimo-v2.5/off/v1" \
    --model "xiaomi/mimo-v2.5" \
    --reasoning-disabled \
    || failures=$((failures + 1))

run_config "openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5/off/v1" \
    --model "nvidia/llama-3.3-nemotron-super-49b-v1.5" \
    --reasoning-disabled \
    || failures=$((failures + 1))

run_config "openrouter/google/gemini-2.5-flash/off/v1" \
    --model "google/gemini-2.5-flash" \
    --reasoning-disabled \
    || failures=$((failures + 1))

run_config "openrouter/google/gemini-3-flash-preview/off/v1" \
    --model "google/gemini-3-flash-preview" \
    --reasoning-disabled \
    || failures=$((failures + 1))

run_config "openrouter/google/gemini-3.1-flash-lite/off/v1" \
    --model "google/gemini-3.1-flash-lite" \
    --reasoning-disabled \
    || failures=$((failures + 1))

run_config "openrouter/google/gemini-2.5-flash-lite/off/v1" \
    --model "google/gemini-2.5-flash-lite" \
    --reasoning-disabled \
    || failures=$((failures + 1))

run_config "openrouter/openai/gpt-5-nano/low/v1" \
    --model "openai/gpt-5-nano" \
    --effort low \
    || failures=$((failures + 1))

run_config "openrouter/openai/gpt-oss-20b/low/v1" \
    --model "openai/gpt-oss-20b" \
    --effort low \
    || failures=$((failures + 1))


printf '\n'
if [[ "${failures}" -eq 0 ]]; then
    printf 'All %s OpenRouter configs passed.\n' "9"
    exit 0
fi

printf '%s of 9 OpenRouter configs failed.\n' "${failures}" >&2
exit 1
