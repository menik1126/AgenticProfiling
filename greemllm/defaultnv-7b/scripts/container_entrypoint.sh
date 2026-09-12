#!/usr/bin/env bash
set -euo pipefail

rm -rf /tmp/greenllm-nats /tmp/greenllm-etcd
mkdir -p /tmp/greenllm-nats /tmp/greenllm-etcd /experiment/logs

nats-server -js -sd /tmp/greenllm-nats > /experiment/logs/nats.log 2>&1 &
nats_pid=$!
etcd \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://0.0.0.0:2379 \
  --data-dir /tmp/greenllm-etcd > /experiment/logs/etcd.log 2>&1 &
etcd_pid=$!

cleanup() {
  kill "$nats_pid" "$etcd_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.disagg:Frontend -f /experiment/config/dynamo_disagg.yaml \
  > /experiment/logs/dynamo.log 2>&1 &
dynamo_pid=$!
wait "$dynamo_pid"
