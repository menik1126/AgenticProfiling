#!/usr/bin/env python3
"""Run a paper-Figure-1-style sinusoidal workload under two DVFS policies."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import yaml

from run_paper_figure_microbench import BASE, BASE_CONFIG, GREEN, Clocks, run


def make_trace(path: Path, duration_s: float) -> int:
    rows = []
    credit = 0.0
    request_id = 0
    dt = 0.05
    steps = int(duration_s / dt)
    for step in range(steps):
        t = step * dt
        qps = 5.5 - 4.5 * math.cos(2 * math.pi * t / duration_s)
        credit += qps * dt
        while credit >= 1.0:
            rows.append({
                "request_id": request_id,
                "scheduled_offset_s": t,
                "input_tokens": 65,
                "output_tokens": 64,
                "source": "local-sinusoidal-microbenchmark",
            })
            request_id += 1
            credit -= 1.0
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=180.0)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    trace = root / "sinusoidal_trace.jsonl"
    count = make_trace(trace, args.duration_s)
    (root / "metadata.json").write_text(json.dumps({
        "duration_s": args.duration_s,
        "qps_min": 1.0,
        "qps_max": 10.0,
        "input_tokens": 65,
        "output_tokens": 64,
        "requests": count,
        "greenllm_commit": subprocess.check_output(
            ["git", "-C", str(GREEN), "rev-parse", "HEAD"], text=True
        ).strip(),
        "baseline_replayer_commit": subprocess.check_output(
            ["git", "-C", str(BASE), "rev-parse", "HEAD"], text=True
        ).strip(),
    }, indent=2) + "\n")
    clocks = Clocks()
    try:
        clocks.reset_all()
        print("sinusoid DefaultNV", flush=True)
        run([
            "python3", str(BASE / "scripts/replay.py"), "--trace", str(trace),
            "--output-dir", str(root / "defaultnv"),
        ], root / "defaultnv.log")
        clocks.reset_all()

        config = yaml.safe_load(BASE_CONFIG.read_text())
        config["trace"] = str(trace)
        config_path = root / "greenllm_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print("sinusoid GreenLLM", flush=True)
        run([
            "python3", str(GREEN / "scripts/greenllm_replay.py"),
            "--config", str(config_path), "--output-dir", str(root / "greenllm"),
        ], root / "greenllm.log")
    finally:
        clocks.reset_all()
        clocks.close()
    after = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,clocks.current.sm,pstate",
        "--format=csv,noheader",
    ], text=True)
    (root / "gpu_clocks_after_reset.csv").write_text(after)
    print(root, flush=True)


if __name__ == "__main__":
    main()
