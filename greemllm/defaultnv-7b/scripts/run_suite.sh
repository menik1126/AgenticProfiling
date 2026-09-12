#!/usr/bin/env bash
set -euo pipefail

experiment_root=$(cd "$(dirname "$0")/.." && pwd)
trace_dir="$experiment_root/data/traces"
container_name=greenllm-defaultnv-7b
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
cp "$experiment_root/config/experiment.yaml" "$result_root/experiment.yaml"
git -C "$experiment_root" rev-parse HEAD > "$result_root/source_commit.txt"

model_list=$(curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null || true)
if ! grep -Fq 'Qwen/Qwen2.5-Coder-7B-Instruct' <<<"$model_list"; then
  echo "Dynamo model workers are not registered" >&2
  exit 1
fi

traces=("$@")
if (( ${#traces[@]} == 0 )); then
  traces=(
    alibaba_1qps alibaba_3qps alibaba_5qps alibaba_8qps alibaba_10qps
    azure_code5 azure_code8 azure_conv5 azure_conv8
  )
fi

for required in "${traces[@]}"; do
  if [[ ! -s "$trace_dir/$required.jsonl" ]]; then
    echo "Missing trace: $trace_dir/$required.jsonl" >&2
    exit 1
  fi
done

for warmup in 1 2; do
  curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen2.5-Coder-7B-Instruct","messages":[{"role":"user","content":"Warm up the inference service."}],"stream":false,"max_tokens":32,"nvext":{"ignore_eos":true},"temperature":0}' \
    >/dev/null
done

for trace in "${traces[@]}"; do
  echo "Starting $trace at $(date -Is)"
  python3 "$experiment_root/scripts/replay.py" \
    --trace "$trace_dir/$trace.jsonl" \
    --output-dir "$result_root/$trace"
done

echo "$result_root"
