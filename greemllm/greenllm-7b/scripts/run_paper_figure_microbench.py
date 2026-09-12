#!/usr/bin/env python3
"""Run the missing local micro-experiments for paper Figures 3c and 12.

The service must already be running.  Every fixed-clock point is reset in a
finally block; GreenLLM's own controller also resets clocks after every run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
GREEN = ROOT / "greenllm-7b"
BASE = ROOT / "defaultnv-7b"
SOURCE_TRACE = BASE / "data/traces/alibaba_10qps.jsonl"
BASE_CONFIG = GREEN / "config/greenllm.yaml"
REPLAY = BASE / "scripts/replay.py"
GREEN_REPLAY = GREEN / "scripts/greenllm_replay.py"
GPUS = tuple(range(8))


class Clocks:
    container = "greenllm-prefillsplit-7b"

    def __init__(self) -> None:
        subprocess.run(
            ["docker", "cp", "/usr/sbin/nvidia-smi", f"{self.container}:/usr/local/bin/nvidia-smi"],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    def _run(self, flag: str, value: str | None = None) -> None:
        command = [
            "docker", "exec", self.container, "/lib64/ld-linux-x86-64.so.2",
            "/usr/local/bin/nvidia-smi", "-i", ",".join(map(str, GPUS)), flag,
        ]
        if value is not None:
            command.append(value)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    def lock_all(self, frequency: int) -> None:
        self._run("-lgc", f"{frequency},{frequency}")

    def reset_all(self) -> None:
        self._run("-rgc")

    def close(self) -> None:
        pass


def make_trace(path: Path, duration_s: float, max_output_tokens: int) -> int:
    rows = []
    with SOURCE_TRACE.open() as stream:
        for line in stream:
            row = json.loads(line)
            if float(row["scheduled_offset_s"]) >= duration_s:
                continue
            row["output_tokens"] = min(max_output_tokens, max(1, int(row["output_tokens"])))
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def green_config(
    output: Path, trace: Path, *, prefill_margin: float, decode_margin: float
) -> Path:
    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["trace"] = str(trace)
    config["controller"]["prefill_slo_margin"] = prefill_margin
    config["controller"]["tbt_lower_margin"] = decode_margin
    path = output / "input_config.yaml"
    output.mkdir(parents=True, exist_ok=False)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    trace = root / "trace.jsonl"
    count = make_trace(trace, args.duration_s, args.max_output_tokens)
    metadata = {
        "source_trace": str(SOURCE_TRACE),
        "duration_s": args.duration_s,
        "max_output_tokens": args.max_output_tokens,
        "requests": count,
        "greenllm_commit": subprocess.check_output(
            ["git", "-C", str(GREEN), "rev-parse", "HEAD"], text=True
        ).strip(),
        "baseline_replayer_commit": subprocess.check_output(
            ["git", "-C", str(BASE), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"micro trace: {count} requests", flush=True)

    clocks = Clocks()
    try:
        fixed_root = root / "fixed_clock"
        fixed_root.mkdir()
        for frequency in (600, 900, 1200, 1500, 1800, 1950):
            point = fixed_root / f"{frequency}mhz"
            print(f"fixed-clock point: {frequency} MHz", flush=True)
            clocks.reset_all()
            try:
                clocks.lock_all(frequency)
                run(
                    ["python3", str(REPLAY), "--trace", str(trace), "--output-dir", str(point)],
                    fixed_root / f"{frequency}mhz.log",
                )
            finally:
                clocks.reset_all()
            time.sleep(2)

        prefill_root = root / "prefill_margin"
        prefill_root.mkdir()
        for margin in (0.2, 0.4, 0.6, 0.85, 0.95, 1.2, 2.0):
            label = str(margin).replace(".", "p")
            point = prefill_root / label
            print(f"prefill-margin point: {margin}", flush=True)
            config = green_config(
                point, trace, prefill_margin=margin, decode_margin=0.95
            )
            run(
                ["python3", str(GREEN_REPLAY), "--config", str(config), "--output-dir", str(point / "run")],
                prefill_root / f"{label}.log",
            )
            clocks.reset_all()
            time.sleep(2)

        decode_root = root / "decode_margin"
        decode_root.mkdir()
        for margin in (0.2, 0.4, 0.6, 0.85, 0.95, 1.2, 2.0):
            label = str(margin).replace(".", "p")
            point = decode_root / label
            print(f"decode-margin point: {margin}", flush=True)
            config = green_config(
                point, trace, prefill_margin=0.95, decode_margin=margin
            )
            run(
                ["python3", str(GREEN_REPLAY), "--config", str(config), "--output-dir", str(point / "run")],
                decode_root / f"{label}.log",
            )
            clocks.reset_all()
            time.sleep(2)
    finally:
        clocks.reset_all()
        clocks.close()

    after = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=index,clocks.current.sm,pstate",
            "--format=csv,noheader",
        ],
        text=True,
    )
    (root / "gpu_clocks_after_reset.csv").write_text(after)
    print(root, flush=True)


if __name__ == "__main__":
    main()
