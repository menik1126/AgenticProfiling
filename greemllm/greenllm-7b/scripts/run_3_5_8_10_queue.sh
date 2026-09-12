#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
baseline_root="$workspace_root/defaultnv-7b"
prefillsplit_root="$workspace_root/prefillsplit-7b"
greenllm_root="$workspace_root/greenllm-7b"
baseline_container=greenllm-defaultnv-7b
prefillsplit_container=greenllm-prefillsplit-7b
traces=(alibaba_3qps alibaba_5qps alibaba_8qps alibaba_10qps)
start_stage=${1:-baseline}

case "$start_stage" in
  baseline|prefillsplit|greenllm) ;;
  *)
    echo "Usage: $0 [baseline|prefillsplit|greenllm]" >&2
    exit 2
    ;;
esac

stop_container() {
  local name=$1
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    docker stop --timeout 30 "$name" >/dev/null
  fi
  for _ in $(seq 1 60); do
    if ! docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for container removal: $name" >&2
  return 1
}

reset_clocks() {
  local helper=
  if docker ps --format '{{.Names}}' | grep -qx "$prefillsplit_container"; then
    helper=$prefillsplit_container
  elif docker ps --format '{{.Names}}' | grep -qx "$baseline_container"; then
    helper=$baseline_container
  fi
  if [[ -n "$helper" ]]; then
    docker cp /usr/sbin/nvidia-smi "$helper:/usr/local/bin/nvidia-smi" >/dev/null 2>&1 || true
    docker exec "$helper" /lib64/ld-linux-x86-64.so.2 \
      /usr/local/bin/nvidia-smi -i 0,1,2,3,4,5,6,7 -rgc >/dev/null 2>&1 || true
  else
    nvidia-smi -i 0,1,2,3,4,5,6,7 -rgc >/dev/null 2>&1 || true
  fi
}

wait_ready() {
  local container=$1
  local waited=0
  local model=Qwen/Qwen2.5-Coder-7B-Instruct
  while (( waited < 1800 )); do
    if ! docker ps --format '{{.Names}}' | grep -qx "$container"; then
      echo "$container exited before becoming ready" >&2
      docker logs --tail 200 "$container" 2>&1 || true
      return 1
    fi
    if curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null | grep -Fq "$model" && \
        curl -fsS http://127.0.0.1:8000/v1/chat/completions \
          -H 'Content-Type: application/json' \
          -d '{"model":"Qwen/Qwen2.5-Coder-7B-Instruct","messages":[{"role":"user","content":"Readiness probe."}],"stream":false,"max_tokens":1,"nvext":{"ignore_eos":true},"temperature":0}' \
          >/dev/null 2>&1; then
      echo "$container ready after ${waited}s"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "Timed out waiting for $container" >&2
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  reset_clocks
  stop_container "$baseline_container" || true
  stop_container "$prefillsplit_container" || true
  exit "$status"
}
trap cleanup EXIT INT TERM HUP

echo "Queue started at $(date -Is)"
reset_clocks
stop_container "$baseline_container"
stop_container "$prefillsplit_container"

if [[ "$start_stage" == baseline ]]; then
  echo "Stage baseline started at $(date -Is)"
  "$baseline_root/scripts/launch.sh"
  wait_ready "$baseline_container"
  "$baseline_root/scripts/run_suite.sh" "${traces[@]}"
fi

if [[ "$start_stage" == baseline || "$start_stage" == prefillsplit ]]; then
  echo "Stage prefillsplit started at $(date -Is)"
  "$prefillsplit_root/scripts/launch.sh"
  wait_ready "$prefillsplit_container"
  "$prefillsplit_root/scripts/run_suite.sh" "${traces[@]}"
fi

echo "Stage greenllm started at $(date -Is)"
"$prefillsplit_root/scripts/launch.sh"
wait_ready "$prefillsplit_container"
"$greenllm_root/scripts/run_suite.sh" "${traces[@]}"

echo "Queue completed at $(date -Is)"
