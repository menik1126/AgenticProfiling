#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "$0")/../.." && pwd)
experiment_root="$workspace_root/defaultnv-7b"
source_dir="$experiment_root/data/source"
trace_dir="$experiment_root/data/traces"
mkdir -p "$source_dir" "$trace_dir"

code_csv="$source_dir/AzureLLMInferenceTrace_code_1week.csv"
conv_csv="$source_dir/AzureLLMInferenceTrace_conv_1week.csv"

if [[ ! -s "$code_csv" ]]; then
  curl -fL --retry 5 --retry-all-errors \
    -o "$code_csv.part" \
    https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv
  mv "$code_csv.part" "$code_csv"
fi
if [[ ! -s "$conv_csv" ]]; then
  curl -fL --retry 5 --retry-all-errors \
    -o "$conv_csv.part" \
    https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv
  mv "$conv_csv.part" "$conv_csv"
fi

for qps in 1 3 5 8 10; do
  python3 "$experiment_root/scripts/prepare_traces.py" alibaba \
    --servegen-root "$workspace_root/vendor/ServeGen" \
    --pool m-large --span-start 64800 --span-end 68400 \
    --duration 1800 --qps "$qps" --seed 0 \
    --output "$trace_dir/alibaba_${qps}qps.jsonl"
done

for kind in code conv; do
  input_csv="$source_dir/AzureLLMInferenceTrace_${kind}_1week.csv"
  for divisor in 5 8; do
    python3 "$experiment_root/scripts/prepare_traces.py" azure \
      --input "$input_csv" --kind "$kind" --rate-divisor "$divisor" \
      --duration 1800 --output "$trace_dir/azure_${kind}${divisor}.jsonl"
  done
done

sha256sum "$code_csv" "$conv_csv" "$trace_dir"/*.jsonl \
  > "$experiment_root/data/SHA256SUMS"
