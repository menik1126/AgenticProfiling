#!/usr/bin/env python3
"""Replay a token-length trace against Dynamo while sampling all GPU boards."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import aiohttp
import pynvml
from transformers import AutoTokenizer


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * pct / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


class PowerSampler:
    def __init__(self, interval_s: float):
        pynvml.nvmlInit()
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        ]
        self.interval_s = interval_s
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.stop_event.is_set():
            timestamp = time.perf_counter()
            gpu_rows = []
            for index, handle in enumerate(self.handles):
                gpu_rows.append(
                    {
                        "gpu": index,
                        "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                        "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                        "utilization_pct": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                    }
                )
            self.samples.append({"timestamp_s": timestamp, "gpus": gpu_rows})
            self.stop_event.wait(self.interval_s)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        pynvml.nvmlShutdown()


class PromptFactory:
    def __init__(self, model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.cache = {}

    def make(self, requested_tokens: int):
        requested_tokens = max(int(requested_tokens), 16)
        if requested_tokens in self.cache:
            return self.cache[requested_tokens]
        empty = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=True,
            add_generation_prompt=True,
        )
        empty_tokens = empty["input_ids"] if isinstance(empty, Mapping) else empty
        requested_tokens = max(requested_tokens, len(empty_tokens))
        low, high = 0, requested_tokens + 32
        best = ""
        best_count = 0
        while low <= high:
            count = (low + high) // 2
            content = "x " * count
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
            )
            tokens = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            actual = len(tokens)
            if actual <= requested_tokens and actual >= best_count:
                best, best_count = content, actual
            if actual < requested_tokens:
                low = count + 1
            elif actual > requested_tokens:
                high = count - 1
            else:
                break
        result = (best, best_count)
        self.cache[requested_tokens] = result
        return result


async def request_once(session, semaphore, endpoint, model, prompt_factory, request, start_time):
    scheduled = start_time + request["scheduled_offset_s"]
    await asyncio.sleep(max(0, scheduled - time.perf_counter()))
    async with semaphore:
        dispatched = time.perf_counter()
        prompt, actual_input_tokens = prompt_factory.make(request["input_tokens"])
        desired_output = max(1, int(request["output_tokens"]))
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": desired_output,
            "nvext": {"ignore_eos": True},
            "temperature": 0,
        }
        token_times = []
        stream_events = 0
        observed_output_tokens = 0
        raw_observed_output_tokens = 0
        handoff_token_adjusted = False
        error = None
        status = None
        try:
            async with session.post(endpoint, json=payload) as response:
                status = response.status
                if response.status != 200:
                    error = (await response.text())[:1000]
                else:
                    buffer = b""
                    async for chunk in response.content.iter_any():
                        buffer += chunk
                        while b"\n\n" in buffer:
                            event, buffer = buffer.split(b"\n\n", 1)
                            if not event.startswith(b"data:") or b"[DONE]" in event:
                                continue
                            try:
                                decoded = json.loads(event[5:].strip())
                                stream_events += 1
                                usage = decoded.get("usage") or {}
                                completion_tokens = usage.get("completion_tokens")
                                if (
                                    isinstance(completion_tokens, int)
                                    and completion_tokens > raw_observed_output_tokens
                                ):
                                    raw_observed_output_tokens = completion_tokens
                                    # Dynamo's disaggregated one-token path counts
                                    # its context-only handoff token in usage.  It
                                    # is transport accounting, not a second token
                                    # returned to the workload.
                                    effective_tokens = completion_tokens
                                    if desired_output == 1 and completion_tokens == 2:
                                        effective_tokens = 1
                                        handoff_token_adjusted = True
                                    timestamp = time.perf_counter()
                                    if effective_tokens > observed_output_tokens:
                                        token_times.extend(
                                            [timestamp]
                                            * (effective_tokens - observed_output_tokens)
                                        )
                                        observed_output_tokens = effective_tokens
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass
        except Exception as exc:
            error = repr(exc)
        completed = time.perf_counter()
        if (
            status == 200
            and error is None
            and observed_output_tokens != desired_output
        ):
            error = (
                f"completion token mismatch: expected {desired_output}, "
                f"observed {observed_output_tokens}"
            )
        tbts_ms = [(b - a) * 1000 for a, b in zip(token_times, token_times[1:])]
        ttft_ms = (token_times[0] - dispatched) * 1000 if token_times else math.nan
        return {
            **request,
            "actual_input_tokens": actual_input_tokens,
            "observed_stream_events": stream_events,
            "observed_output_tokens": observed_output_tokens,
            "raw_observed_output_tokens": raw_observed_output_tokens,
            "handoff_token_adjusted": handoff_token_adjusted,
            "scheduled_lateness_ms": (dispatched - scheduled) * 1000,
            "ttft_ms": ttft_ms,
            "tbt_p95_ms": percentile(tbts_ms, 95),
            "latency_ms": (completed - dispatched) * 1000,
            "http_status": status,
            "error": error,
        }


def summarize(results, samples, start_time, end_time):
    successful = [row for row in results if row["http_status"] == 200 and not row["error"]]
    short_medium = [row for row in successful if row["input_tokens"] <= 1024]
    long_rows = [row for row in successful if row["input_tokens"] > 1024]
    energy_j = 0.0
    for left, right in zip(samples, samples[1:]):
        interval = right["timestamp_s"] - left["timestamp_s"]
        left_power = sum(gpu["power_w"] for gpu in left["gpus"])
        right_power = sum(gpu["power_w"] for gpu in right["gpus"])
        energy_j += (left_power + right_power) * 0.5 * interval
    tbt_eligible = [
        row for row in successful
        if math.isfinite(row["tbt_p95_ms"])
    ]
    tbt_pass = [row for row in tbt_eligible if row["tbt_p95_ms"] <= 100]
    ttft_pass = [
        row for row in successful
        if row["ttft_ms"] <= (400 if row["input_tokens"] <= 1024 else 2000)
    ]
    return {
        "requests": len(results),
        "successful_requests": len(successful),
        "duration_s": end_time - start_time,
        "throughput_requests_s": len(successful) / (end_time - start_time),
        "ttft_p50_ms": percentile([row["ttft_ms"] for row in successful], 50),
        "ttft_p95_ms": percentile([row["ttft_ms"] for row in successful], 95),
        "short_medium_ttft_pass_pct": 100 * sum(row["ttft_ms"] <= 400 for row in short_medium) / max(1, len(short_medium)),
        "long_ttft_pass_pct": 100 * sum(row["ttft_ms"] <= 2000 for row in long_rows) / max(1, len(long_rows)),
        "overall_ttft_pass_pct": 100 * len(ttft_pass) / max(1, len(successful)),
        "tbt_eligible_requests": len(tbt_eligible),
        "tbt_pass_pct": 100 * len(tbt_pass) / max(1, len(tbt_eligible)),
        "energy_j": energy_j,
        "energy_wh": energy_j / 3600,
        "energy_j_per_request": energy_j / max(1, len(successful)),
        "power_samples": len(samples),
    }


async def run(args):
    requests = [json.loads(line) for line in args.trace.read_text().splitlines() if line]
    prompt_factory = PromptFactory(args.model)
    sampler = PowerSampler(args.power_interval_ms / 1000)
    semaphore = asyncio.Semaphore(args.max_inflight)
    timeout = aiohttp.ClientTimeout(total=None, connect=60)
    start_time = time.perf_counter() + 2
    sampler.start()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                request_once(session, semaphore, args.endpoint, args.model, prompt_factory, request, start_time)
            )
            for request in requests
        ]
        results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    sampler.stop()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "requests.jsonl").open("w") as stream:
        for row in results:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (args.output_dir / "power.jsonl").open("w") as stream:
        for row in sampler.samples:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = summarize(results, sampler.samples, start_time, end_time)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--power-interval-ms", type=int, default=20)
    parser.add_argument("--max-inflight", type=int, default=2048)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
