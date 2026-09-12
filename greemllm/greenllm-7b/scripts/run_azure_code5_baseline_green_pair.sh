#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
baseline_root="$workspace_root/defaultnv-7b"
service_root="$workspace_root/prefillsplit-7b"
green_root="$workspace_root/greenllm-7b"
traces=("$@")
if (( ${#traces[@]} == 0 )); then
  traces=(azure_code5)
fi
queue_label=${QUEUE_LABEL:-azure-code5-three-way}
workload_label=${WORKLOAD_LABEL:-"${traces[*]}"}
original_config=${ORIGINAL_GREENLLM_CONFIG:-"$green_root/config/greenllm.yaml"}
idle600_config=${IDLE600_GREENLLM_CONFIG:-"$green_root/config/greenllm_idle600.yaml"}
baseline_container=greenllm-defaultnv-7b
green_container=greenllm-prefillsplit-7b
image=greenllm-dynamo:0.3.1-trtllm-sm86
stamp=$(date -u +%Y%m%dT%H%M%SZ)
queue_dir="$green_root/queue-runs/$stamp-$queue_label"
mkdir -p "$queue_dir"
exec > >(tee -a "$queue_dir/queue.log") 2>&1

reset_clocks() {
  local running_container=
  for candidate in "$baseline_container" "$green_container"; do
    if docker ps --format '{{.Names}}' | grep -qx "$candidate"; then
      running_container=$candidate
      break
    fi
  done
  if [[ -n "$running_container" ]]; then
    docker cp /usr/sbin/nvidia-smi "$running_container:/usr/local/bin/nvidia-smi" >/dev/null 2>&1 || true
    if docker exec "$running_container" /lib64/ld-linux-x86-64.so.2 \
        /usr/local/bin/nvidia-smi -i 0,1,2,3,4,5,6,7 -rgc >/dev/null 2>&1; then
      return 0
    fi
  fi
  docker run --rm --pull never --gpus all --runtime nvidia --privileged \
    -v /usr/sbin/nvidia-smi:/usr/local/bin/nvidia-smi:ro \
    --entrypoint /lib64/ld-linux-x86-64.so.2 "$image" \
    /usr/local/bin/nvidia-smi -i 0,1,2,3,4,5,6,7 -rgc >/dev/null 2>&1 || true
}

stop_remove() {
  local container=$1
  if docker ps --format '{{.Names}}' | grep -qx "$container"; then
    docker stop --timeout 30 "$container" >/dev/null 2>&1 || true
  fi
  if docker ps -a --format '{{.Names}}' | grep -qx "$container"; then
    docker rm -f "$container" >/dev/null 2>&1 || true
  fi
  for _ in $(seq 1 60); do
    if ! docker ps -a --format '{{.Names}}' | grep -qx "$container"; then
      return 0
    fi
    sleep 1
  done
  echo "Container $container still exists after cleanup" >&2
  return 1
}

wait_for_service_port_release() {
  for _ in $(seq 1 60); do
    if ! ss -ltnH 'sport = :8000' | grep -q .; then
      return 0
    fi
    sleep 1
  done
  echo "TCP port 8000 is still in use after service cleanup" >&2
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  reset_clocks
  stop_remove "$baseline_container"
  stop_remove "$green_container"
  exit "$status"
}
trap cleanup EXIT INT TERM HUP

wait_for_idle_gpus() {
  while true; do
    mapfile -t users < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | sort -u)
    if (( ${#users[@]} == 0 )); then
      return 0
    fi
    echo "Waiting for GPUs to become idle; compute PIDs: ${users[*]}"
    sleep 30
  done
}

registered_count() {
  local log=$1
  local component=$2
  if [[ ! -f "$log" ]]; then
    echo 0
    return 0
  fi
  sed -n "s/.*\[$component:\([0-9][0-9]*\)\] Starting $component instance with all registered endpoints.*/\1/p" \
    "$log" 2>/dev/null | sort -u | wc -l
}

start_healthy_service() {
  local label=$1
  local root=$2
  local container=$3
  local decode_component=$4
  local log="$root/logs/dynamo.log"
  local healthy=false

  for attempt in 1 2 3; do
    stop_remove "$container"
    wait_for_service_port_release
    mkdir -p "$root/logs"
    if [[ -e "$log" ]]; then
      mv "$log" "$queue_dir/${label}-attempt-${attempt}.previous-dynamo.log"
    fi
    "$root/scripts/launch.sh"
    for _ in $(seq 1 240); do
      if ! docker ps --format '{{.Names}}' | grep -qx "$container"; then
        break
      fi
      local models decode_workers prefill_workers
      models=$(curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null || true)
      decode_workers=$(registered_count "$log" "$decode_component")
      prefill_workers=$(registered_count "$log" TensorRTLLMPrefillWorker)
      if grep -Fq 'Qwen/Qwen2.5-Coder-7B-Instruct' <<<"$models" \
          && [[ "$decode_workers" -eq 4 && "$prefill_workers" -eq 2 ]]; then
        echo "$label service ready with Decode 4/4 and Prefill 2/2 at $(date -Is)"
        healthy=true
        break
      fi
      if grep -Fq 'Race condition, retry the call' "$log" 2>/dev/null; then
        echo "$label service attempt $attempt hit worker registration race" >&2
        break
      fi
      sleep 2
    done
    if [[ "$healthy" == true ]]; then
      return 0
    fi
    reset_clocks
    stop_remove "$container"
  done
  echo "$label could not register all workers after three attempts" >&2
  return 1
}

warmup_service() {
  for _ in 1 2; do
    curl -fsS http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"Qwen/Qwen2.5-Coder-7B-Instruct","messages":[{"role":"user","content":"Warm up the inference service."}],"stream":false,"max_tokens":32,"nvext":{"ignore_eos":true},"temperature":0}' \
      >/dev/null
  done
}

latest_result_root() {
  find "$1/results" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-
}

validate_result() {
  local label=$1
  local root=$2
  shift 2
  python3 - "$label" "$root" "$@" <<'PY'
import json
import sys

label, root, *traces = sys.argv[1:]
for trace in traces:
    path = f"{root}/{trace}/summary.json"
    summary = json.load(open(path))
    if summary["successful_requests"] != summary["requests"]:
        raise SystemExit(f"{label} {trace} has failed requests: {summary}")
    print(f"{label} result validated: {path}")
PY
}

for trace in "${traces[@]}"; do
  if [[ ! "$trace" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid trace name: $trace" >&2
    exit 1
  fi
  if [[ ! -s "$baseline_root/data/traces/$trace.jsonl" ]]; then
    echo "Missing trace: $baseline_root/data/traces/$trace.jsonl" >&2
    exit 1
  fi
done

echo "$workload_label three-way queue started at $(date -Is)"
echo "Baseline commit: $(git -C "$baseline_root" rev-parse HEAD)"
echo "Service commit: $(git -C "$service_root" rev-parse HEAD)"
echo "GreenLLM commit: $(git -C "$green_root" rev-parse HEAD)"
echo "Original config: $original_config"
echo "Idle600 config: $idle600_config"
for trace in "${traces[@]}"; do
  echo "Trace $trace SHA-256: $(sha256sum "$baseline_root/data/traces/$trace.jsonl" | awk '{print $1}')"
done

wait_for_idle_gpus
reset_clocks

start_healthy_service Baseline "$baseline_root" "$baseline_container" TensorRTLLMWorker
echo "Starting Baseline $workload_label at $(date -Is)"
"$baseline_root/scripts/run_suite.sh" "${traces[@]}"
baseline_result=$(latest_result_root "$baseline_root")
validate_result Baseline "$baseline_result" "${traces[@]}"
stop_remove "$baseline_container"
reset_clocks

start_healthy_service GreenLLM-original "$service_root" "$green_container" PrefillSplitTensorRTLLMWorker
warmup_service
echo "Starting original GreenLLM $workload_label at $(date -Is)"
GREENLLM_CONFIG="$original_config" \
  "$green_root/scripts/run_suite.sh" "${traces[@]}"
original_result=$(latest_result_root "$green_root")
validate_result GreenLLM-original "$original_result" "${traces[@]}"
stop_remove "$green_container"
reset_clocks

start_healthy_service GreenLLM-idle600 "$service_root" "$green_container" PrefillSplitTensorRTLLMWorker
warmup_service
echo "Starting idle600 GreenLLM $workload_label at $(date -Is)"
GREENLLM_CONFIG="$idle600_config" \
  "$green_root/scripts/run_suite.sh" "${traces[@]}"
idle600_result=$(latest_result_root "$green_root")
validate_result GreenLLM-idle600 "$idle600_result" "${traces[@]}"

reset_clocks
stop_remove "$green_container"
echo "Baseline result: $baseline_result"
echo "Original GreenLLM result: $original_result"
echo "Idle600 GreenLLM result: $idle600_result"
echo "$workload_label three-way queue completed at $(date -Is)"
