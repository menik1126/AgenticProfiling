#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 PREFILLSPLIT_REPLAY_PID" >&2
  exit 2
fi
replay_pid=$1
profile_root=$(cd "$(dirname "$0")/.." && pwd)

echo "[$(date -Is)] waiting for PrefillSplit replay PID $replay_pid"
while kill -0 "$replay_pid" 2>/dev/null; do
  sleep 10
done
echo "[$(date -Is)] PrefillSplit replay finished; starting offline profiling"
tmux kill-session -t greenllm-prefillsplit-suite 2>/dev/null || true
exec "$profile_root/scripts/run_profile.sh"

