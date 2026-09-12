#!/usr/bin/env bash
set -euo pipefail

experiment_root=$(cd "$(dirname "$0")/.." && pwd)
trace_dir="$experiment_root/data/traces"
generator="$experiment_root/scripts/generate_longctx_poisson.py"

for spec in 02:0.2 05:0.5; do
  label=${spec%%:*}
  qps=${spec#*:}
  python3 "$generator" \
    --output "$trace_dir/longctx_27k_${label}qps.jsonl" \
    --qps "$qps" \
    --duration-s 1800 \
    --lower-input-tokens 27000 \
    --upper-input-tokens 27001 \
    --output-tokens 64 \
    --max-context-tokens 32768 \
    --seed 27000
done

sha256sum \
  "$trace_dir/longctx_27k_02qps.jsonl" \
  "$trace_dir/longctx_27k_05qps.jsonl"
