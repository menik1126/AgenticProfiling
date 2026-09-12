#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
experiment_root="$workspace_root/prefillsplit-7b"
dynamo_root="$workspace_root/vendor/dynamo-v0.3.1"
hf_cache=/home/shenhaojia/.cache/huggingface
dynamo_run_patch="$workspace_root/.build-cache/dynamo-run-a1e1077/dynamo-run"
container_name=greenllm-prefillsplit-7b
baseline_container=greenllm-defaultnv-7b

if docker ps --format '{{.Names}}' | grep -qx "$baseline_container"; then
  echo "Baseline container $baseline_container is still running; leaving it untouched" >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$container_name"; then
  echo "Container $container_name already exists" >&2
  exit 1
fi

expected_gpu_count=8
mapfile -t gpu_caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/[[:space:]]//g')
if [[ "${#gpu_caps[@]}" -ne "$expected_gpu_count" ]]; then
  echo "Expected $expected_gpu_count visible GPUs, found ${#gpu_caps[@]}" >&2
  exit 1
fi
for cap in "${gpu_caps[@]}"; do
  if [[ "$cap" != "${gpu_caps[0]}" ]]; then
    echo "Visible GPUs have mixed compute capabilities: ${gpu_caps[*]}" >&2
    exit 1
  fi
done
arch_label="sm${gpu_caps[0]//./}"
image="greenllm-dynamo:0.3.1-trtllm-${arch_label}"

if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "Required architecture-specific image is not available: $image" >&2
  exit 1
fi
if [[ ! -x "$dynamo_run_patch" ]]; then
  echo "Patched Dynamo frontend is not available: $dynamo_run_patch" >&2
  exit 1
fi

mkdir -p "$experiment_root/logs"
docker run --rm -d \
  --name "$container_name" \
  --gpus all \
  --runtime nvidia \
  --network host \
  --privileged \
  --shm-size=10G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  -e PREFILL_SPLIT_ROUTING_CONFIG=/experiment/config/routing.json \
  -v "$dynamo_root:/workspace" \
  -v "$experiment_root:/experiment" \
  -v "$hf_cache:/root/.cache/huggingface" \
  -v "$dynamo_run_patch:/usr/local/lib/python3.12/dist-packages/dynamo/sdk/cli/bin/dynamo-run:ro" \
  -w /workspace \
  "$image" \
  bash /experiment/scripts/container_entrypoint.sh

echo "Started $container_name. Follow $experiment_root/logs/dynamo.log for readiness."

