#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
experiment_root="$workspace_root/greenllm-7b"
container=greenllm-prefillsplit-7b
image=greenllm-dynamo:0.3.1-trtllm-sm86
config="$experiment_root/config/greenllm.yaml"
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
result="$experiment_root/results/$run_stamp/alibaba_1qps"

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

watch_parent=$$
( while kill -0 "$watch_parent" 2>/dev/null; do sleep 1; done; reset_clocks ) &
watchdog_pid=$!
cleanup() {
  trap - EXIT INT TERM HUP
  reset_clocks
  kill "$watchdog_pid" >/dev/null 2>&1 || true
  wait "$watchdog_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

model_list=$(curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null || true)
if ! grep -Fq 'Qwen/Qwen2.5-Coder-7B-Instruct' <<<"$model_list"; then
  echo "PrefillSplit service is not ready" >&2
  exit 1
fi
docker cp /usr/sbin/nvidia-smi "$container:/usr/local/bin/nvidia-smi" >/dev/null
reset_clocks
mkdir -p "$(dirname "$result")"
git -C "$experiment_root" rev-parse HEAD >"$(dirname "$result")/source_commit.txt"
sha256sum \
  "$workspace_root/greenllm-profile-7b/results/20260902T125826Z/prefill_models.json" \
  "$workspace_root/greenllm-profile-7b/results/decode-20260902T133755Z/decode_models.json" \
  >"$(dirname "$result")/profile_sha256.txt"
python3 "$experiment_root/scripts/greenllm_replay.py" \
  --config "$config" --output-dir "$result"
reset_clocks
sleep 5
nvidia-smi --query-gpu=index,clocks.current.sm,clocks.max.sm,pstate \
  --format=csv,noheader -i 0,1,2,3,4,5,6,7 \
  >"$result/gpu_clocks_after_reset.csv"
echo "$(dirname "$result")"
