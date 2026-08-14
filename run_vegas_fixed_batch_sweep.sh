#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/benchmark_results/vegas_fixed_batch_sweep_${RUN_ID}}"
SPECULATIVE_TOKENS=(6 12 18 24)

mkdir -p -- "${LOG_DIR}"
cd -- "${ROOT_DIR}"

printf 'Logs: %s\n' "${LOG_DIR}"

for gamma in "${SPECULATIVE_TOKENS[@]}"; do
    log_file="${LOG_DIR}/longbench_v2_gamma_${gamma}.log"
    printf '\nRunning LongBench v2 Vegas with num_speculative_tokens=%s\n' "${gamma}"

    "${PYTHON_BIN}" benchmarks/benchmark_vegas_fixed_batch.py \
        --algorithm vegas \
        --dataset longbench-v2 \
        --longbench-length any \
        --max-model-len 40960 \
        --target-input-tokens 32768 \
        --max-tokens 1024 \
        --batch-size 4 \
        --num-samples 128 \
        --num-speculative-tokens "${gamma}" \
        "$@" 2>&1 | tee "${log_file}"
done

for gamma in "${SPECULATIVE_TOKENS[@]}"; do
    log_file="${LOG_DIR}/aime25_gamma_${gamma}.log"
    printf '\nRunning AIME25 Vegas with num_speculative_tokens=%s\n' "${gamma}"

    "${PYTHON_BIN}" benchmarks/benchmark_vegas_fixed_batch.py \
        --algorithm vegas \
        --dataset aime25 \
        --max-model-len 32768 \
        --max-tokens 24576 \
        --batch-size 2 \
        --num-samples 30 \
        --num-speculative-tokens "${gamma}" \
        "$@" 2>&1 | tee "${log_file}"
done

printf '\nSweep complete. Logs: %s\n' "${LOG_DIR}"
