#!/usr/bin/env python3
"""Convert the disclosed Alibaba/Azure sources into one replay JSONL schema."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def write_jsonl(rows, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(output)
    print(json.dumps({"output": str(output), "requests": count}))


def prepare_alibaba(args) -> None:
    servegen_root = args.servegen_root.resolve()
    sys.path.insert(0, str(servegen_root))
    previous = Path.cwd()
    os.chdir(servegen_root)
    try:
        from servegen import Category, ClientPool, generate_workload
        from servegen.utils import get_constant_rate_fn

        pool = ClientPool(Category.LANGUAGE, args.pool)
        view = pool.span(args.span_start, args.span_end)
        requests = generate_workload(
            view,
            get_constant_rate_fn(view, args.qps),
            duration=args.duration,
            seed=args.seed,
        )
    finally:
        os.chdir(previous)

    rows = (
        {
            "request_id": request.request_id,
            "scheduled_offset_s": float(request.timestamp),
            "input_tokens": int(request.data["input_tokens"]),
            "output_tokens": int(request.data["output_tokens"]),
            "source": "alibaba-servegen",
        }
        for request in requests
    )
    write_jsonl(rows, args.output)


def parse_timestamp(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()


def prepare_azure(args) -> None:
    if args.target_qps is not None:
        if args.target_qps <= 0:
            raise ValueError("Azure target QPS must be positive")
        target_requests = round(args.duration * args.target_qps)
        source_rows = []
        with args.input.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Azure CSV is missing fields: {sorted(missing)}")
            for source_row in reader:
                source_rows.append(source_row)
                if len(source_rows) == target_requests:
                    break
        if len(source_rows) != target_requests:
            raise ValueError(
                f"Azure CSV has {len(source_rows)} usable rows; expected {target_requests}"
            )

        first_timestamp = parse_timestamp(source_rows[0]["TIMESTAMP"])
        last_timestamp = parse_timestamp(source_rows[-1]["TIMESTAMP"])
        source_span = last_timestamp - first_timestamp
        if source_span <= 0:
            raise ValueError("Azure target-QPS source span must be positive")
        target_span = (target_requests - 1) / args.target_qps
        scale = target_span / source_span

        rows = (
            {
                "request_id": request_id,
                "scheduled_offset_s": (
                    parse_timestamp(source_row["TIMESTAMP"]) - first_timestamp
                ) * scale,
                "input_tokens": int(source_row["ContextTokens"]),
                "output_tokens": int(source_row["GeneratedTokens"]),
                "source": f"azure-{args.kind}-target-{args.target_qps:g}qps",
            }
            for request_id, source_row in enumerate(source_rows)
        )
        write_jsonl(rows, args.output)
        return

    def rows():
        first_timestamp = None
        request_id = 0
        with args.input.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Azure CSV is missing fields: {sorted(missing)}")
            for source_row in reader:
                timestamp = parse_timestamp(source_row["TIMESTAMP"])
                if first_timestamp is None:
                    first_timestamp = timestamp
                offset = (timestamp - first_timestamp) * args.rate_divisor
                if offset < 0:
                    continue
                if offset >= args.duration:
                    break
                yield {
                    "request_id": request_id,
                    "scheduled_offset_s": offset,
                    "input_tokens": int(source_row["ContextTokens"]),
                    "output_tokens": int(source_row["GeneratedTokens"]),
                    "source": f"azure-{args.kind}-1of{args.rate_divisor:g}",
                }
                request_id += 1

    write_jsonl(rows(), args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="source", required=True)

    alibaba = subparsers.add_parser("alibaba")
    alibaba.add_argument("--servegen-root", type=Path, required=True)
    alibaba.add_argument("--pool", default="m-large")
    alibaba.add_argument("--span-start", type=int, default=64800)
    alibaba.add_argument("--span-end", type=int, default=68400)
    alibaba.add_argument("--duration", type=int, default=1800)
    alibaba.add_argument("--qps", type=float, required=True)
    alibaba.add_argument("--seed", type=int, default=0)
    alibaba.add_argument("--output", type=Path, required=True)
    alibaba.set_defaults(func=prepare_alibaba)

    azure = subparsers.add_parser("azure")
    azure.add_argument("--input", type=Path, required=True)
    azure.add_argument("--kind", choices=("code", "conv"), required=True)
    azure_rate = azure.add_mutually_exclusive_group(required=True)
    azure_rate.add_argument("--rate-divisor", type=float)
    azure_rate.add_argument("--target-qps", type=float)
    azure.add_argument("--duration", type=int, default=1800)
    azure.add_argument("--output", type=Path, required=True)
    azure.set_defaults(func=prepare_azure)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
