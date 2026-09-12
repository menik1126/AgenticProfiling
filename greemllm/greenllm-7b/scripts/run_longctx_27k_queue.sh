#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
green_root="$workspace_root/greenllm-7b"

export QUEUE_LABEL=longctx-27k-three-way
export WORKLOAD_LABEL="Long-context 27K at 0.2/0.5 QPS"
export PREFILL_SPLIT_ROUTING_CONFIG_FILE="$workspace_root/prefillsplit-7b/config/routing_longctx_27k.json"
export ORIGINAL_GREENLLM_CONFIG="$green_root/config/greenllm_longctx_27k.yaml"
export IDLE600_GREENLLM_CONFIG="$green_root/config/greenllm_longctx_27k_idle600.yaml"

exec "$green_root/scripts/run_azure_code5_baseline_green_pair.sh" \
  longctx_27k_02qps \
  longctx_27k_05qps
