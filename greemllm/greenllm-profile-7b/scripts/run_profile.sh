#!/usr/bin/env bash
set -euo pipefail

profile_root=$(cd "$(dirname "$0")/.." && pwd)
container=greenllm-prefillsplit-7b
image=greenllm-dynamo:0.3.1-trtllm-sm86
profile_gpus=4,5,6,7
config_path=${1:-$profile_root/config/profile.yaml}
config_path=$(realpath "$config_path")
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
output="$profile_root/results/$run_stamp"

reset_clocks() {
  if docker exec "$container" \
      /lib64/ld-linux-x86-64.so.2 /usr/local/bin/nvidia-smi \
      -i "$profile_gpus" -rgc >/dev/null 2>&1; then
    return 0
  fi

  # The serving container can itself fail under load. Clock cleanup must not
  # depend on that container still existing.
  docker run --rm --pull never --gpus all --runtime nvidia --privileged \
    -v /usr/sbin/nvidia-smi:/usr/local/bin/nvidia-smi:ro \
    --entrypoint /lib64/ld-linux-x86-64.so.2 \
    "$image" /usr/local/bin/nvidia-smi -i "$profile_gpus" -rgc \
    >/dev/null 2>&1 || true
}

watch_parent=$$
(
  while kill -0 "$watch_parent" 2>/dev/null; do
    sleep 1
  done
  reset_clocks
) &
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

# Profiling is only valid from the normal, unlocked governor state. This also
# makes a stale lock from an interrupted older run harmless.
reset_clocks

mkdir -p "$profile_root/results"
python3 "$profile_root/scripts/offline_profile.py" \
  --config "$config_path" \
  --output "$output"
reset_clocks
# Let the dynamic governor leave the last profiling P-state before recording
# the restoration evidence; an immediate query can still show a transient
# high clock even though the lock has already been removed.
sleep 5
nvidia-smi --query-gpu=index,clocks.current.sm,clocks.max.sm,pstate \
  --format=csv,noheader -i "$profile_gpus" >"$output/gpu_clocks_after_reset.csv"
echo "$output"
