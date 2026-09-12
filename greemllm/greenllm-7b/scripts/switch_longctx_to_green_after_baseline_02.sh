#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
summary=${1:?usage: $0 BASELINE_02_SUMMARY [OLD_TMUX_SESSION]}
old_session=${2:-greenllm_longctx_27k}
green_session=greenllm_longctx_green02
green_runner="$workspace_root/greenllm-7b/scripts/run_longctx_27k_green_original.sh"

echo "Waiting for a complete Baseline 0.2 summary: $summary"
until jq -e '.requests > 0 and .successful_requests == .requests' "$summary" >/dev/null 2>&1; do
  sleep 1
done

echo "Baseline 0.2 is complete at $(date -Is); stopping the remaining queue"
tmux kill-session -t "$old_session" 2>/dev/null || true

for _ in $(seq 1 180); do
  if ! docker ps -a --format '{{.Names}}' | grep -qx 'greenllm-defaultnv-7b'; then
    break
  fi
  sleep 1
done
if docker ps -a --format '{{.Names}}' | grep -qx 'greenllm-defaultnv-7b'; then
  echo "Baseline container did not finish cleanup" >&2
  exit 1
fi

tmux kill-session -t "$green_session" 2>/dev/null || true
tmux new-session -d -s "$green_session" \
  "cd '$workspace_root/greenllm-7b' && exec '$green_runner' longctx_27k_02qps"
echo "Started original GreenLLM 0.2 in tmux session $green_session at $(date -Is)"
