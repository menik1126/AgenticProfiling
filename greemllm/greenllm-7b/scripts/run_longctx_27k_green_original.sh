#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
green_root="$workspace_root/greenllm-7b"
service_root="$workspace_root/prefillsplit-7b"
container=greenllm-prefillsplit-7b
baseline_container=greenllm-defaultnv-7b
image=greenllm-dynamo:0.3.1-trtllm-sm86
routing="$service_root/config/routing_longctx_27k.json"
config="$green_root/config/greenllm_longctx_27k.yaml"
trace=${1:-longctx_27k_02qps}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
queue_dir="$green_root/queue-runs/$stamp-longctx-27k-green-original"
mkdir -p "$queue_dir"
exec > >(tee -a "$queue_dir/queue.log") 2>&1

reset_clocks() {
  docker run --rm --pull never --gpus all --runtime nvidia --privileged \
    -v /usr/sbin/nvidia-smi:/usr/local/bin/nvidia-smi:ro \
    --entrypoint /lib64/ld-linux-x86-64.so.2 "$image" \
    /usr/local/bin/nvidia-smi -i 0,1,2,3,4,5,6,7 -rgc >/dev/null 2>&1 || true
}

stop_remove() {
  local name=$1
  docker stop --timeout 30 "$name" >/dev/null 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do
    if ! docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
      return 0
    fi
    sleep 1
  done
  echo "Container $name still exists after cleanup" >&2
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  stop_remove "$container"
  reset_clocks
  exit "$status"
}
trap cleanup EXIT INT TERM HUP

registered_count() {
  local log=$1 component=$2
  [[ -f "$log" ]] || { echo 0; return; }
  sed -n "s/.*\[$component:\([0-9][0-9]*\)\] Starting $component instance with all registered endpoints.*/\1/p" \
    "$log" | sort -u | wc -l
}

if docker ps --format '{{.Names}}' | grep -qx "$baseline_container"; then
  echo "Baseline service is still running" >&2
  exit 1
fi
if [[ ! -s "$workspace_root/defaultnv-7b/data/traces/$trace.jsonl" ]]; then
  echo "Missing trace: $trace" >&2
  exit 1
fi

echo "Original GreenLLM $trace started at $(date -Is)"
echo "GreenLLM commit: $(git -C "$green_root" rev-parse HEAD)"
echo "Service commit: $(git -C "$service_root" rev-parse HEAD)"
echo "Trace SHA-256: $(sha256sum "$workspace_root/defaultnv-7b/data/traces/$trace.jsonl" | awk '{print $1}')"

ready=false
for attempt in 1 2 3; do
  stop_remove "$container"
  mkdir -p "$service_root/logs"
  if [[ -e "$service_root/logs/dynamo.log" ]]; then
    mv "$service_root/logs/dynamo.log" "$queue_dir/attempt-$attempt.previous-dynamo.log"
  fi
  PREFILL_SPLIT_ROUTING_CONFIG_FILE="$routing" "$service_root/scripts/launch.sh"
  for _ in $(seq 1 240); do
    models=$(curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null || true)
    decode=$(registered_count "$service_root/logs/dynamo.log" PrefillSplitTensorRTLLMWorker)
    prefill=$(registered_count "$service_root/logs/dynamo.log" TensorRTLLMPrefillWorker)
    if grep -Fq 'Qwen/Qwen2.5-Coder-7B-Instruct' <<<"$models" \
        && [[ "$decode" -eq 4 && "$prefill" -eq 2 ]]; then
      ready=true
      break
    fi
    if grep -Fq 'Race condition, retry the call' "$service_root/logs/dynamo.log" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  [[ "$ready" == true ]] && break
done
[[ "$ready" == true ]] || { echo "Service failed readiness" >&2; exit 1; }

echo "Service ready with Decode 4/4 and Prefill 2/2 at $(date -Is)"
for _ in 1 2; do
  curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen2.5-Coder-7B-Instruct","messages":[{"role":"user","content":"Warm up the inference service."}],"stream":false,"max_tokens":32,"nvext":{"ignore_eos":true},"temperature":0}' \
    >/dev/null
done

GREENLLM_CONFIG="$config" "$green_root/scripts/run_suite.sh" "$trace"
echo "Original GreenLLM $trace completed at $(date -Is)"
