#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
green_root="$workspace_root/greenllm-7b"

export QUEUE_LABEL=azure-code10-continuous-batching-three-way
export WORKLOAD_LABEL="Azure Code 10 QPS continuous batching"
export ORIGINAL_GREENLLM_CONFIG="$green_root/config/greenllm_azure_code10.yaml"
export IDLE600_GREENLLM_CONFIG="$green_root/config/greenllm_azure_code10_idle600.yaml"

exec "$green_root/scripts/run_azure_code5_baseline_green_pair.sh" azure_code10
