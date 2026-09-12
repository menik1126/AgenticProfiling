#!/usr/bin/env python3
"""Coarse offline latency, power, and TPS profiling for GreenLLM."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import aiohttp
import numpy as np
import pynvml
import yaml
from transformers import AutoTokenizer


def percentile(values, pct):
    values = sorted(v for v in values if not math.isnan(v))
    if not values:
        return math.nan
    pos = (len(values) - 1) * pct / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


class PromptFactory:
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.cache = {}

    def make(self, requested_tokens, nonce=""):
        requested_tokens = int(requested_tokens)
        cache_key = (requested_tokens, nonce)
        if cache_key in self.cache:
            return self.cache[cache_key]
        low, high, best, best_n = 0, requested_tokens + 32, "", 0
        while low <= high:
            n = (low + high) // 2
            text = f"profiling request {nonce} " + "x " * n
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            tokens = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            actual = len(tokens)
            if actual <= requested_tokens and actual >= best_n:
                best, best_n = text, actual
            if actual < requested_tokens:
                low = n + 1
            elif actual > requested_tokens:
                high = n - 1
            else:
                break
        self.cache[cache_key] = (best, best_n)
        return self.cache[cache_key]


class PowerSampler:
    def __init__(self, interval_s):
        pynvml.nvmlInit()
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
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
            row = {"timestamp_s": time.perf_counter(), "gpus": []}
            for i, handle in enumerate(self.handles):
                row["gpus"].append({
                    "gpu": i,
                    "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                    "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                    "memory_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
                    "utilization_pct": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                })
            self.samples.append(row)
            self.stop_event.wait(self.interval_s)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        pynvml.nvmlShutdown()


class ClockController:
    def __init__(self, container):
        self.container = container

    def _run(self, *args):
        subprocess.run(
            [
                "docker", "exec", self.container,
                "/lib64/ld-linux-x86-64.so.2",
                "/usr/local/bin/nvidia-smi",
                *args,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    def lock(self, gpus, frequency):
        self._run("-i", ",".join(map(str, gpus)), "-lgc", f"{frequency},{frequency}")

    def reset(self, gpus):
        self._run("-i", ",".join(map(str, gpus)), "-rgc")


def integrate_energy(samples, gpu_ids):
    gpu_ids = set(gpu_ids)
    energy_j = 0.0
    for left, right in zip(samples, samples[1:]):
        dt = right["timestamp_s"] - left["timestamp_s"]
        lp = sum(g["power_w"] for g in left["gpus"] if g["gpu"] in gpu_ids)
        rp = sum(g["power_w"] for g in right["gpus"] if g["gpu"] in gpu_ids)
        energy_j += (lp + rp) * 0.5 * dt
    return energy_j


def summarize_telemetry(samples, gpu_ids):
    gpu_ids = set(gpu_ids)
    gpu_rows = [
        gpu
        for sample in samples
        for gpu in sample["gpus"]
        if gpu["gpu"] in gpu_ids
    ]
    if not gpu_rows:
        return {}
    return {
        "power_w_mean": statistics.fmean(row["power_w"] for row in gpu_rows),
        "sm_clock_mhz_mean": statistics.fmean(
            row["sm_clock_mhz"] for row in gpu_rows
        ),
        "sm_clock_mhz_min": min(row["sm_clock_mhz"] for row in gpu_rows),
        "sm_clock_mhz_max": max(row["sm_clock_mhz"] for row in gpu_rows),
        "utilization_pct_mean": statistics.fmean(
            row["utilization_pct"] for row in gpu_rows
        ),
        "samples_per_gpu": len(gpu_rows) / len(gpu_ids),
    }


async def request_once(session, endpoint, model, prompt, output_tokens, delay_s=0):
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": output_tokens,
        "nvext": {"ignore_eos": True},
        "temperature": 0,
    }
    started = time.perf_counter()
    token_times, observed, error = [], 0, None
    try:
        async with session.post(endpoint, json=payload) as response:
            if response.status != 200:
                error = (await response.text())[:500]
            else:
                buffer = b""
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    while b"\n\n" in buffer:
                        event, buffer = buffer.split(b"\n\n", 1)
                        if not event.startswith(b"data:") or b"[DONE]" in event:
                            continue
                        item = json.loads(event[5:].strip())
                        completion = (item.get("usage") or {}).get("completion_tokens")
                        if isinstance(completion, int) and completion > observed:
                            now = time.perf_counter()
                            token_times.extend([now] * (completion - observed))
                            observed = completion
    except Exception as exc:
        error = repr(exc)
    ended = time.perf_counter()
    tbts = [(b - a) * 1000 for a, b in zip(token_times, token_times[1:])]
    return {
        "ttft_ms": (token_times[0] - started) * 1000 if token_times else math.nan,
        "tbt_mean_ms": statistics.fmean(tbts) if tbts else math.nan,
        "tbt_median_ms": statistics.median(tbts) if tbts else math.nan,
        "tbt_p95_ms": percentile(tbts, 95),
        "tbt_values_ms": tbts,
        "decode_duration_ms": (
            (token_times[-1] - token_times[0]) * 1000
            if len(token_times) > 1 else math.nan
        ),
        # The disaggregated endpoint reports the context-only handoff token in
        # completion_tokens. Every subsequent timestamp is one decode token.
        "observed_decode_tokens": max(0, observed - 1),
        "observed_output_tokens": observed,
        "latency_ms": (ended - started) * 1000,
        "error": error,
    }


async def measured_requests(config, prompt_factory, specs, gpu_ids):
    sampler = PowerSampler(config["power_sample_interval_ms"] / 1000)
    timeout = aiohttp.ClientTimeout(total=None, connect=60)
    sampler.start()
    started = time.perf_counter()
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            rows = await asyncio.gather(*[
                request_once(
                    session,
                    config["endpoint"],
                    config["model"],
                    prompt_factory.make(
                        spec["input_tokens"], spec.get("nonce", "")
                    )[0],
                    spec["output_tokens"],
                    spec.get("delay_s", 0),
                )
                for spec in specs
            ])
    finally:
        ended = time.perf_counter()
        sampler.stop()
    return rows, sampler.samples, ended - started, integrate_energy(sampler.samples, gpu_ids)


async def measured_steady_load(
    config,
    prompt_factory,
    input_tokens,
    output_tokens,
    concurrency,
    duration_s,
    nonce_prefix,
    gpu_ids,
):
    """Run a closed-loop load without building an unbounded arrival queue."""
    sampler = PowerSampler(config["power_sample_interval_ms"] / 1000)
    timeout = aiohttp.ClientTimeout(total=None, connect=60)
    results = []
    request_index = 0
    sampler.start()
    started = time.perf_counter()
    deadline = started + duration_s

    async def worker(session, worker_index):
        nonlocal request_index
        while time.perf_counter() < deadline:
            current_index = request_index
            request_index += 1
            prompt = prompt_factory.make(
                input_tokens,
                f"{nonce_prefix}-worker-{worker_index}-request-{current_index}",
            )[0]
            results.append(await request_once(
                session,
                config["endpoint"],
                config["model"],
                prompt,
                output_tokens,
            ))

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(*[
                worker(session, worker_index)
                for worker_index in range(concurrency)
            ])
    finally:
        ended = time.perf_counter()
        sampler.stop()
    return (
        results,
        sampler.samples,
        ended - started,
        integrate_energy(sampler.samples, gpu_ids),
    )


async def warmup_requests(
    config, prompt_factory, input_tokens, output_tokens, count, nonce_prefix
):
    if count <= 0:
        return
    timeout = aiohttp.ClientTimeout(total=None, connect=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for warmup_index in range(count):
            prompt = prompt_factory.make(
                input_tokens, f"{nonce_prefix}-warmup-{warmup_index}"
            )[0]
            for attempt in range(config["request_retries"] + 1):
                result = await request_once(
                    session, config["endpoint"], config["model"], prompt, output_tokens
                )
                if not result["error"]:
                    break
                if attempt < config["request_retries"]:
                    await asyncio.sleep(config["request_retry_delay_s"])
            else:
                raise RuntimeError(f"warmup request failed: {result['error']}")


async def measure_idle(config, gpu_ids):
    sampler = PowerSampler(config["power_sample_interval_ms"] / 1000)
    sampler.start()
    try:
        await asyncio.sleep(config["idle_sample_duration_s"])
    finally:
        sampler.stop()
    elapsed = sampler.samples[-1]["timestamp_s"] - sampler.samples[0]["timestamp_s"]
    return sampler.samples, elapsed, integrate_energy(sampler.samples, gpu_ids)


def r_squared(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    return 1.0 if total == 0 and residual == 0 else 1 - residual / total


def fit_prefill_models(rows, config):
    latency_rows = [row for row in rows if row["phase"] == "prefill_latency"]
    reference_frequency = config["reference_sm_frequency_mhz"]
    reference_rows = [
        row for row in latency_rows if row["frequency_mhz"] == reference_frequency
    ]
    reference_medians = []
    for length in sorted({row["actual_input_tokens"] for row in reference_rows}):
        values = [
            row["ttft_ms"]
            for row in reference_rows
            if row["actual_input_tokens"] == length and row["error"] is None
        ]
        if values:
            reference_medians.append((length, statistics.median(values)))
    if len(reference_medians) < 3:
        raise RuntimeError("at least three successful prompt lengths are needed")
    lengths = np.asarray([item[0] for item in reference_medians], dtype=float)
    ttfts = np.asarray([item[1] for item in reference_medians], dtype=float)
    latency_coefficients = np.polyfit(lengths, ttfts, 2)

    latency_matrix = []
    reference_by_length = dict(reference_medians)
    for frequency in sorted({row["frequency_mhz"] for row in latency_rows}):
        for length in sorted({
            row["actual_input_tokens"]
            for row in latency_rows
            if row["frequency_mhz"] == frequency
        }):
            values = [
                row["ttft_ms"]
                for row in latency_rows
                if row["frequency_mhz"] == frequency
                and row["actual_input_tokens"] == length
                and row["error"] is None
            ]
            if not values or length not in reference_by_length:
                continue
            observed = statistics.median(values)
            predicted = reference_by_length[length] * reference_frequency / frequency
            latency_matrix.append({
                "frequency_mhz": frequency,
                "input_tokens": length,
                "median_ttft_ms": observed,
                "inverse_frequency_prediction_ms": predicted,
                "absolute_percentage_error": abs(observed - predicted) / observed,
            })
    non_reference_errors = [
        point["absolute_percentage_error"]
        for point in latency_matrix
        if point["frequency_mhz"] != reference_frequency
    ]

    power_models = {}
    for pool in config["prefill_power"]["pools"]:
        power_rows = [
            row
            for row in rows
            if row["phase"] == "prefill_power" and row["pool"] == pool["name"]
        ]
        by_concurrency = {}
        for concurrency in sorted({row["concurrency"] for row in power_rows}):
            concurrency_rows = [
                row for row in power_rows if row["concurrency"] == concurrency
            ]
            if len(concurrency_rows) < 4:
                raise RuntimeError(
                    f"at least four power points are needed for "
                    f"{pool['name']} concurrency {concurrency}"
                )
            frequencies = np.asarray(
                [row["frequency_mhz"] for row in concurrency_rows], dtype=float
            )
            powers = np.asarray(
                [row["average_power_w"] for row in concurrency_rows], dtype=float
            )
            coefficients = np.polyfit(frequencies, powers, 3)
            points = []
            for row in concurrency_rows:
                requests_per_second = row["successful_requests_per_second"]
                points.append({
                    "frequency_mhz": row["frequency_mhz"],
                    "average_power_w": row["average_power_w"],
                    "requests_per_second": requests_per_second,
                    "energy_j_per_request": (
                        row["average_power_w"] / requests_per_second
                    ),
                    "utilization_pct_mean": row["telemetry"][
                        "utilization_pct_mean"
                    ],
                })
            optimum = min(points, key=lambda point: point["energy_j_per_request"])
            by_concurrency[str(concurrency)] = {
                "coefficients_high_to_low": coefficients.tolist(),
                "power_fit_r_squared": r_squared(
                    powers, np.polyval(coefficients, frequencies)
                ),
                "measured_energy_optimum": optimum,
                "points": points,
            }
        power_models[pool["name"]] = {
            "gpu_ids": pool["gpu_ids"],
            "prompt_length": pool["prompt_length"],
            "by_concurrency": by_concurrency,
        }

    idle_rows = [row for row in rows if row["phase"] == "idle_power"]
    return {
        "latency_model": {
            "reference_frequency_mhz": reference_frequency,
            "formula": "ttft_ms = a*input_tokens^2 + b*input_tokens + c",
            "coefficients": {
                "a": float(latency_coefficients[0]),
                "b": float(latency_coefficients[1]),
                "c": float(latency_coefficients[2]),
            },
            "r_squared": r_squared(
                ttfts, np.polyval(latency_coefficients, lengths)
            ),
            "median_points": [
                {"input_tokens": int(length), "median_ttft_ms": float(ttft)}
                for length, ttft in reference_medians
            ],
        },
        "inverse_frequency_validation": {
            "formula": "t(L,f) = t(L,f_ref) * f_ref / f",
            "mean_absolute_percentage_error": (
                statistics.fmean(non_reference_errors)
                if non_reference_errors else 0.0
            ),
            "p95_absolute_percentage_error": (
                percentile(non_reference_errors, 95)
                if non_reference_errors else 0.0
            ),
            "points": latency_matrix,
        },
        "idle_power": {
            row.get("pool", "all"): row for row in idle_rows
        },
        "power_models": power_models,
    }


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


async def main_async(args):
    config = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "profile.yaml").write_text(args.config.read_text())
    prompt_factory = PromptFactory(config["model"])
    clocks = ClockController(config["service_container"])
    frequencies = config["sm_frequencies_mhz"]
    groups = config["gpu_groups"]
    rows_path = args.output / "measurements.jsonl"
    all_rows = []
    run_nonce = args.output.name

    def record(row):
        all_rows.append(row)
        with rows_path.open("a") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    controlled_gpus = groups["prefill"]
    clocks.reset(controlled_gpus)
    try:
        idle_samples, idle_elapsed, idle_energy = await measure_idle(
            config, controlled_gpus
        )
        record({
            "phase": "idle_power",
            "pool": "all",
            "gpu_ids": controlled_gpus,
            "elapsed_s": idle_elapsed,
            "energy_j": idle_energy,
            "average_power_w": idle_energy / idle_elapsed,
            "telemetry": summarize_telemetry(idle_samples, controlled_gpus),
        })
        for pool in config["prefill_power"]["pools"]:
            pool_energy = integrate_energy(idle_samples, pool["gpu_ids"])
            record({
                "phase": "idle_power",
                "pool": pool["name"],
                "gpu_ids": pool["gpu_ids"],
                "elapsed_s": idle_elapsed,
                "energy_j": pool_energy,
                "average_power_w": pool_energy / idle_elapsed,
                "telemetry": summarize_telemetry(
                    idle_samples, pool["gpu_ids"]
                ),
            })
        print("idle power profiling complete", flush=True)

        latency = config["prefill_latency"]
        reference_frequency = config["reference_sm_frequency_mhz"]
        latency_frequencies = latency.get(
            "frequencies_mhz", [reference_frequency]
        )
        for frequency in latency_frequencies:
            clocks.lock(controlled_gpus, frequency)
            await asyncio.sleep(config["settle_after_clock_change_s"])
            for length in latency["prompt_lengths"]:
                await warmup_requests(
                    config,
                    prompt_factory,
                    length,
                    latency["output_tokens"],
                    latency["warmup_repetitions"],
                    f"{run_nonce}-latency-{frequency}-{length}",
                )
                for repetition in range(latency["repetitions"]):
                    nonce = (
                        f"{run_nonce}-latency-{frequency}-{length}-{repetition}"
                    )
                    actual_tokens = prompt_factory.make(length, nonce)[1]
                    specs = [{
                        "input_tokens": length,
                        "output_tokens": latency["output_tokens"],
                        "nonce": nonce,
                    }]
                    results, samples, elapsed, energy = await measured_requests(
                        config, prompt_factory, specs, controlled_gpus
                    )
                    record({
                        "phase": "prefill_latency",
                        "frequency_mhz": frequency,
                        "requested_input_tokens": length,
                        "actual_input_tokens": actual_tokens,
                        "repetition": repetition,
                        "elapsed_s": elapsed,
                        "energy_j": energy,
                        "telemetry": summarize_telemetry(
                            samples, controlled_gpus
                        ),
                        **results[0],
                    })
                print(
                    f"latency complete: {frequency} MHz, {length} tokens",
                    flush=True,
                )
            clocks.reset(controlled_gpus)

        pp = config["prefill_power"]
        concurrency_levels = pp.get("concurrency_levels", [pp["concurrency"]])
        for pool in pp["pools"]:
            for concurrency in concurrency_levels:
                for frequency in frequencies:
                    clocks.lock(controlled_gpus, frequency)
                    await asyncio.sleep(config["settle_after_clock_change_s"])
                    await warmup_requests(
                        config,
                        prompt_factory,
                        pool["prompt_length"],
                        pp["output_tokens"],
                        pp["warmup_repetitions"],
                        (
                            f"{run_nonce}-power-{pool['name']}-"
                            f"{concurrency}-{frequency}"
                        ),
                    )
                    results, samples, elapsed, energy = await measured_steady_load(
                        config,
                        prompt_factory,
                        pool["prompt_length"],
                        pp["output_tokens"],
                        concurrency,
                        pp["duration_s"],
                        (
                            f"{run_nonce}-power-{pool['name']}-"
                            f"{concurrency}-{frequency}"
                        ),
                        pool["gpu_ids"],
                    )
                    error_count = sum(bool(result["error"]) for result in results)
                    record({
                        "phase": "prefill_power", "pool": pool["name"],
                        "gpu_ids": pool["gpu_ids"],
                        "concurrency": concurrency,
                        "frequency_mhz": frequency, "requests": len(results),
                        "errors": error_count,
                        "successful_requests_per_second": (
                            (len(results) - error_count) / elapsed
                        ),
                        "elapsed_s": elapsed, "energy_j": energy,
                        "average_power_w": energy / elapsed,
                        "telemetry": summarize_telemetry(
                            samples, pool["gpu_ids"]
                        ),
                    })
                    if error_count:
                        raise RuntimeError(
                            f"{pool['name']} concurrency {concurrency} at "
                            f"{frequency} MHz had {error_count}/"
                            f"{len(results)} request errors"
                        )
                    print(
                        f"power complete: {pool['name']}, concurrency "
                        f"{concurrency}, {frequency} MHz",
                        flush=True,
                    )
                    clocks.reset(controlled_gpus)

        write_csv(args.output / "measurements.csv", all_rows)
        models = fit_prefill_models(all_rows, config)
        (args.output / "prefill_models.json").write_text(
            json.dumps(models, indent=2) + "\n"
        )
    finally:
        try:
            clocks.reset(controlled_gpus)
        except subprocess.CalledProcessError:
            # The outer runner has an independent privileged-container fallback.
            # Do not hide the original profiling failure if the serving container
            # disappeared at the same time.
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
