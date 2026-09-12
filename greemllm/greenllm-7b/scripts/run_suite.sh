#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
experiment_root="$workspace_root/greenllm-7b"
trace_dir="$workspace_root/defaultnv-7b/data/traces"
container=greenllm-prefillsplit-7b
image=greenllm-dynamo:0.3.1-trtllm-sm86
config=${GREENLLM_CONFIG:-"$experiment_root/config/greenllm.yaml"}
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
result_root="$experiment_root/results/$run_stamp"

reset_clocks() {
  if docker exec "$container" /lib64/ld-linux-x86-64.so.2 \
      /usr/local/bin/nvidia-smi -i 0,1,2,3,4,5,6,7 -rgc >/dev/null 2>&1; then
    return 0
  fi
  docker run --rm --pull never --gpus all --runtime nvidia --privileged \
    -v /usr/sbin/nvidia-smi:/usr/local/bin/nvidia-smi:ro \
    --entrypoint /lib64/ld-linux-x86-64.so.2 "$image" \
    /usr/local/bin/nvidia-smi -i 0,1,2,3,4,5,6,7 -rgc >/dev/null 2>&1 || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  reset_clocks
  exit "$status"
}
trap cleanup EXIT INT TERM HUP

traces=("$@")
if (( ${#traces[@]} == 0 )); then
  traces=(alibaba_3qps alibaba_5qps alibaba_8qps alibaba_10qps)
fi
for trace in "${traces[@]}"; do
  if [[ ! -s "$trace_dir/$trace.jsonl" ]]; then
    echo "Missing trace: $trace_dir/$trace.jsonl" >&2
    exit 1
  fi
done

model_list=$(curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null || true)
if ! grep -Fq 'Qwen/Qwen2.5-Coder-7B-Instruct' <<<"$model_list"; then
  echo "PrefillSplit service is not ready" >&2
  exit 1
fi
docker cp /usr/sbin/nvidia-smi "$container:/usr/local/bin/nvidia-smi" >/dev/null
reset_clocks
mkdir -p "$result_root"
cp "$config" "$result_root/greenllm.yaml"
git -C "$experiment_root" rev-parse HEAD > "$result_root/source_commit.txt"
git -C "$workspace_root/prefillsplit-7b" rev-parse HEAD > "$result_root/service_commit.txt"
sha256sum \
  "$workspace_root/greenllm-profile-7b/results/20260902T125826Z/prefill_models.json" \
  "$workspace_root/greenllm-profile-7b/results/decode-20260902T133755Z/decode_models.json" \
  > "$result_root/profile_sha256.txt"

for trace in "${traces[@]}"; do
  echo "Starting $trace at $(date -Is)"
  python3 "$experiment_root/scripts/greenllm_replay.py" \
    --config "$config" \
    --trace "$trace_dir/$trace.jsonl" \
    --output-dir "$result_root/$trace"
  reset_clocks
done

sleep 5
nvidia-smi --query-gpu=index,clocks.current.sm,clocks.max.sm,pstate \
  --format=csv,noheader -i 0,1,2,3,4,5,6,7 \
  > "$result_root/gpu_clocks_after_reset.csv"
echo "$result_root"
