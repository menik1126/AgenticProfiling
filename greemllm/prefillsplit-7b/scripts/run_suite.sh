#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
experiment_root="$workspace_root/prefillsplit-7b"
baseline_root="$workspace_root/defaultnv-7b"
trace_dir="$baseline_root/data/traces"
replayer="$baseline_root/scripts/replay.py"
container_name=greenllm-prefillsplit-7b
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
result_root="$experiment_root/results/$run_stamp"

cleanup() {
  local status=$?
  trap - EXIT
  if docker ps --format '{{.Names}}' | grep -qx "$container_name"; then
    echo "Stopping $container_name after benchmark"
    docker stop --timeout 30 "$container_name" >/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT

mkdir -p "$result_root"

model=Qwen/Qwen2.5-Coder-7B-Instruct
model_list=$(curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null || true)
if ! grep -Fq "$model" <<<"$model_list"; then
  echo "PrefillSplit Dynamo model workers are not registered" >&2
  exit 1
fi

traces=("$@")
if (( ${#traces[@]} == 0 )); then
  traces=(alibaba_1qps)
fi

for required in "${traces[@]}"; do
  if [[ ! -s "$trace_dir/$required.jsonl" ]]; then
    echo "Missing trace: $trace_dir/$required.jsonl" >&2
    exit 1
  fi
done

cp "$experiment_root/config/experiment.yaml" "$result_root/experiment.yaml"
git -C "$experiment_root" rev-parse HEAD > "$result_root/source_commit.txt"
git -C "$baseline_root" rev-parse HEAD > "$result_root/replayer_commit.txt"

for warmup in 1 2; do
  curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen2.5-Coder-7B-Instruct","messages":[{"role":"user","content":"Warm up the PrefillSplit inference service."}],"stream":false,"max_tokens":32,"nvext":{"ignore_eos":true},"temperature":0}' \
    >/dev/null
done

for trace in "${traces[@]}"; do
  echo "Starting $trace at $(date -Is)"
  python3 "$replayer" \
    --trace "$trace_dir/$trace.jsonl" \
    --output-dir "$result_root/$trace"
done

echo "$result_root"
