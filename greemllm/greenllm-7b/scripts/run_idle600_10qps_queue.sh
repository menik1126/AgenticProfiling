#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
green_root="$workspace_root/greenllm-7b"
service_root="$workspace_root/prefillsplit-7b"
container=greenllm-prefillsplit-7b
config="$green_root/config/greenllm_idle600.yaml"
trace="$workspace_root/defaultnv-7b/data/traces/alibaba_10qps.jsonl"
smoke_trace="$workspace_root/defaultnv-7b/data/traces/fixed_output_smoke.jsonl"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
queue_dir="$green_root/queue-runs/$stamp-idle600-10qps"
mkdir -p "$queue_dir"
exec > >(tee -a "$queue_dir/queue.log") 2>&1

reset_clocks() {
  if docker ps --format '{{.Names}}' | grep -qx "$container"; then
    docker cp /usr/sbin/nvidia-smi "$container:/usr/local/bin/nvidia-smi" >/dev/null 2>&1 || true
    docker exec "$container" /lib64/ld-linux-x86-64.so.2 \
      /usr/local/bin/nvidia-smi -i 0,1,2,3,4,5,6,7 -rgc >/dev/null 2>&1 || true
  fi
}

cleanup() {
  status=$?
  trap - EXIT INT TERM HUP
  reset_clocks
  if docker ps --format '{{.Names}}' | grep -qx "$container"; then
    docker stop --timeout 30 "$container" >/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM HUP

echo "Idle600 queue started at $(date -Is)"
service_ready=false
for attempt in 1 2 3; do
  if ! docker ps --format '{{.Names}}' | grep -qx "$container"; then
    if docker ps -a --format '{{.Names}}' | grep -qx "$container"; then
      docker rm "$container" >/dev/null
    fi
    "$service_root/scripts/launch.sh"
  fi
  endpoint_ready=false
  for i in $(seq 1 360); do
    if curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null \
        | grep -Fq 'Qwen/Qwen2.5-Coder-7B-Instruct'; then
      endpoint_ready=true
      break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$container"; then
      break
    fi
    sleep 5
  done
  decode_workers=$(sed -n \
    's/.*\[PrefillSplitTensorRTLLMWorker:\([0-9][0-9]*\)\] Starting PrefillSplitTensorRTLLMWorker instance with all registered endpoints.*/\1/p' \
    "$service_root/logs/dynamo.log" | sort -u | wc -l)
  if [[ "$endpoint_ready" == true && "$decode_workers" -eq 4 ]]; then
    echo "Service ready with 4/4 Decode workers at $(date -Is)"
    service_ready=true
    break
  fi
  echo "Service attempt $attempt invalid: endpoint_ready=$endpoint_ready decode_workers=$decode_workers" >&2
  reset_clocks
  if docker ps --format '{{.Names}}' | grep -qx "$container"; then
    docker stop --timeout 30 "$container" >/dev/null || true
  fi
done
if [[ "$service_ready" != true ]]; then
  echo "Could not start all four Decode workers after three attempts" >&2
  exit 1
fi

smoke_dir="$green_root/results/smoke-idle600-$stamp"
python3 "$green_root/scripts/greenllm_replay.py" \
  --config "$config" --trace "$smoke_trace" --output-dir "$smoke_dir"

python3 - "$smoke_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
events = [json.loads(line) for line in (root / "frequency_events.jsonl").open()]
if summary["successful_requests"] != summary["requests"]:
    raise SystemExit("idle600 smoke had failed requests")
for group in ("prefill_short", "prefill_long"):
    if not any(
        row["group"] == group
        and row["reason"] == "idle_locked"
        and row["frequency_mhz"] == 600
        for row in events
    ):
        raise SystemExit(f"missing 600 MHz idle event for {group}")
    if not any(
        row["group"] == group
        and row["reason"] == "final_reset"
        and row["frequency_mhz"] is None
        for row in events
    ):
        raise SystemExit(f"missing final clock reset for {group}")
print("Idle600 smoke validation passed")
PY

GREENLLM_CONFIG="$config" "$green_root/scripts/run_suite.sh" alibaba_10qps
echo "Idle600 10 QPS queue completed at $(date -Is)"
