#!/usr/bin/env python3
"""Generate a deterministic, two-pool long-context Poisson replay trace."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qps", type=float, required=True)
    parser.add_argument("--duration-s", type=float, default=1800)
    parser.add_argument("--lower-input-tokens", type=int, default=27000)
    parser.add_argument("--upper-input-tokens", type=int, default=27001)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=27000)
    args = parser.parse_args()

    if args.qps <= 0 or args.duration_s <= 0:
        raise ValueError("qps and duration-s must be positive")
    if args.lower_input_tokens < 16:
        raise ValueError("lower-input-tokens must be at least 16")
    if args.upper_input_tokens != args.lower_input_tokens + 1:
        raise ValueError("upper-input-tokens must be exactly one token above lower")
    if args.output_tokens < 1:
        raise ValueError("output-tokens must be positive")
    if args.upper_input_tokens + args.output_tokens > args.max_context_tokens:
        raise ValueError("input plus output exceeds max-context-tokens")

    rng = random.Random(args.seed)
    timestamp = 0.0
    request_id = 0
    rows = []
    while True:
        timestamp += rng.expovariate(args.qps)
        if timestamp >= args.duration_s:
            break
        rows.append(
            {
                "request_id": request_id,
                "scheduled_offset_s": timestamp,
                "input_tokens": (
                    args.lower_input_tokens
                    if request_id % 2 == 0
                    else args.upper_input_tokens
                ),
                "output_tokens": args.output_tokens,
                "source": "synthetic-longctx-poisson-two-pool",
            }
        )
        request_id += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "requests": len(rows),
                "qps": args.qps,
                "duration_s": args.duration_s,
                "input_tokens": [
                    args.lower_input_tokens,
                    args.upper_input_tokens,
                ],
                "output_tokens": args.output_tokens,
                "seed": args.seed,
            }
        )
    )


if __name__ == "__main__":
    main()
