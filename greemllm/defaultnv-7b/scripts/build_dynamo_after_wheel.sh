#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
experiment_root="$workspace_root/defaultnv-7b"
vendor_root="$workspace_root/vendor/dynamo-v0.3.1"
trt_root="$workspace_root/.build-cache/trtllm-v0.3.1"
trtllm_commit=137fe35539ea182f1495f5021bfda97c729e50c3
expected_gpu_count=8

mapfile -t gpu_caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/[[:space:]]//g')
if (( ${#gpu_caps[@]} != expected_gpu_count )); then
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
wheel_dir="$trt_root/$arch_label"
staging_dir="$wheel_dir/dynamo-wheel"
build_session="trtllm-${arch_label}-retry"
dynamo_session="dynamo-${arch_label}"
image="greenllm-dynamo:0.3.1-trtllm-${arch_label}"
build_log="$experiment_root/logs/build-dynamo-${arch_label}.log"

mkdir -p "$experiment_root/logs"
echo "Waiting for the $arch_label TensorRT-LLM wheel in $wheel_dir"
while :; do
  wheel=$(find "$wheel_dir" -maxdepth 1 -type f -name '*.whl' -print -quit 2>/dev/null || true)
  if [[ -n "$wheel" && -f "$wheel_dir/SHA256SUMS" && -f "$wheel_dir/commit.txt" ]]; then
    break
  fi
  if ! tmux has-session -t "$build_session" 2>/dev/null; then
    echo "TensorRT-LLM build session $build_session ended before a valid wheel appeared" >&2
    exit 1
  fi
  sleep 30
done

mkdir -p "$staging_dir"
ln -f "$wheel" "$staging_dir/$(basename "$wheel")"
printf 'amd64_%s\n' "$trtllm_commit" > "$staging_dir/commit.txt"
echo "Found wheel $(basename "$wheel"); building $image"

if ! (cd "$vendor_root" && ./container/build.sh \
  --framework TENSORRTLLM \
  --tensorrtllm-pip-wheel-dir "$staging_dir" \
  --tag "$image" \
  > "$build_log" 2>&1); then
  echo "Dynamo build failed; see $build_log" >&2
  exit 1
fi

echo "Dynamo image built: $image"
if tmux has-session -t "baseline-after-build-$arch_label" 2>/dev/null; then
  echo "baseline-after-build-$arch_label already exists"
else
  tmux new-session -d -s "baseline-after-build-$arch_label" \
    "cd $experiment_root && exec env ARCH_LABEL=$arch_label BUILD_SESSION=$dynamo_session ./scripts/wait_for_idle_then_run.sh"
  echo "Started baseline-after-build-$arch_label"
fi
