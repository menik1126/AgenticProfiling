#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
experiment_root="$workspace_root/defaultnv-7b"
arch_label=${ARCH_LABEL:-}
if [[ -z "$arch_label" ]]; then
  mapfile -t _caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/[[:space:]]//g')
  if (( ${#_caps[@]} == 0 )); then
    echo "No visible GPUs while selecting the baseline image" >&2
    exit 1
  fi
  arch_label="sm${_caps[0]//./}"
fi
image=${IMAGE:-"greenllm-dynamo:0.3.1-trtllm-${arch_label}"}
build_session=${BUILD_SESSION:-"dynamo-${arch_label}"}
container_name="greenllm-defaultnv-7b"
expected_gpu_count=8
idle_memory_mib=500
poll_s=30
log_file="$experiment_root/logs/baseline-auto-run.log"

mkdir -p "$experiment_root/logs"
exec > >(tee -a "$log_file") 2>&1

echo "[$(date -Is)] waiting for $image"
while ! docker image inspect "$image" >/dev/null 2>&1; do
  if ! tmux has-session -t "$build_session" 2>/dev/null; then
    echo "[$(date -Is)] build session $build_session ended before $image appeared" >&2
    exit 1
  fi
  sleep "$poll_s"
done
echo "[$(date -Is)] image is available: $image"

gpu_state() {
  local -a caps memory
  mapfile -t caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/[[:space:]]//g')
  mapfile -t memory < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed 's/[[:space:]]//g')
  [[ "${#caps[@]}" -eq "$expected_gpu_count" ]] || return 1
  [[ "${#memory[@]}" -eq "$expected_gpu_count" ]] || return 1
  for cap in "${caps[@]}"; do
    [[ "$cap" == "${caps[0]}" ]] || return 1
    [[ "sm${cap//./}" == "$arch_label" ]] || return 1
  done
  for used in "${memory[@]}"; do
    [[ "$used" =~ ^[0-9]+$ ]] || return 1
    if (( used > idle_memory_mib )); then
      return 1
    fi
  done
  local compute_pids
  compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
  [[ -z "$compute_pids" ]]
}

while ! gpu_state; do
  gpu_summary=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr '\n' '; ' || true)
  echo "[$(date -Is)] GPUs are not idle; waiting (index, used MiB, total MiB): $gpu_summary"
  sleep "$poll_s"
done
echo "[$(date -Is)] all $expected_gpu_count GPUs are idle; launching baseline"

"$experiment_root/scripts/launch.sh"

echo "[$(date -Is)] waiting for Dynamo endpoint"
for _ in $(seq 1 240); do
  model_list=$(curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null || true)
  decode_endpoints=$(docker exec "$container_name" bash -lc \
    "ETCDCTL_API=3 etcdctl get --prefix instances/dynamo/TensorRTLLMWorker/generate --keys-only 2>/dev/null | sed '/^$/d' | wc -l" \
    2>/dev/null || echo 0)
  if grep -Fq 'Qwen/Qwen2.5-Coder-7B-Instruct' <<<"$model_list" \
      && [[ "$decode_endpoints" -eq 4 ]]; then
    echo "[$(date -Is)] Dynamo model workers are registered; running complete baseline suite"
    exec "$experiment_root/scripts/run_suite.sh"
  fi
  if ! docker ps --filter "name=^${container_name}$" --filter status=running --format '{{.ID}}' | grep -q .; then
    echo "[$(date -Is)] baseline container stopped before endpoint became ready" >&2
    docker logs "$container_name" 2>&1 | tail -100 >&2 || true
    exit 1
  fi
  sleep 5
done

echo "[$(date -Is)] endpoint did not become ready within 20 minutes" >&2
docker logs "$container_name" 2>&1 | tail -100 >&2 || true
exit 1
