#!/usr/bin/env python3
"""Trace-derived offline Decode latency, power, and energy profiling."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import subprocess
from pathlib import Path

import numpy as np
import yaml

from offline_profile import (
    ClockController,
    PromptFactory,
    measure_idle,
    measured_requests,
    measured_steady_load,
    percentile,
    r_squared,
    summarize_telemetry,
    warmup_requests,
    write_csv,
)


def successful(rows):
    return [row for row in rows if not row["error"] and row["observed_decode_tokens"]]


def fit_decode_models(rows, config):
    latency_rows = [row for row in rows if row["phase"] == "decode_latency"]
    reference_frequency = config["reference_sm_frequency_mhz"]
    latency_models = {}
    for length in sorted({row["actual_input_tokens"] for row in latency_rows}):
        points = []
        for frequency in sorted({row["frequency_mhz"] for row in latency_rows}):
            subset = successful([
                row for row in latency_rows
                if row["actual_input_tokens"] == length
                and row["frequency_mhz"] == frequency
            ])
            if subset:
                points.append({
                    "frequency_mhz": frequency,
                    "median_tbt_ms": statistics.median(row["tbt_median_ms"] for row in subset),
                    "p95_tbt_ms": statistics.median(row["tbt_p95_ms"] for row in subset),
                    "median_decode_tps": statistics.median(
                        row["observed_decode_tokens"] / (row["decode_duration_ms"] / 1000)
                        for row in subset
                    ),
                })
        reference = next(point for point in points if point["frequency_mhz"] == reference_frequency)
        errors = []
        for point in points:
            predicted = reference["median_tbt_ms"] * reference_frequency / point["frequency_mhz"]
            point["inverse_frequency_prediction_ms"] = predicted
            point["inverse_frequency_absolute_percentage_error"] = abs(
                point["median_tbt_ms"] - predicted
            ) / point["median_tbt_ms"]
            if point["frequency_mhz"] != reference_frequency:
                errors.append(point["inverse_frequency_absolute_percentage_error"])
        latency_models[str(length)] = {
            "points": points,
            "inverse_frequency_mape": statistics.fmean(errors),
            "inverse_frequency_p95_ape": percentile(errors, 95),
        }

    power_models = {}
    power_rows = [row for row in rows if row["phase"] == "decode_power"]
    slo = config["decode_power"]["tbt_slo_ms"]
    for length in sorted({row["actual_input_tokens"] for row in power_rows}):
        by_concurrency = {}
        for concurrency in sorted({row["concurrency"] for row in power_rows}):
            subset = sorted([
                row for row in power_rows
                if row["actual_input_tokens"] == length
                and row["concurrency"] == concurrency
            ], key=lambda row: row["frequency_mhz"])
            frequencies = np.asarray([row["frequency_mhz"] for row in subset], dtype=float)
            powers = np.asarray([row["average_power_w"] for row in subset], dtype=float)
            coefficients = np.polyfit(frequencies, powers, 3)
            points = [{
                "frequency_mhz": row["frequency_mhz"],
                "average_power_w": row["average_power_w"],
                "decode_tokens_per_second": row["decode_tokens_per_second"],
                "energy_j_per_decode_token": row["energy_j_per_decode_token"],
                "dynamic_energy_j_per_decode_token": row["dynamic_energy_j_per_decode_token"],
                "tbt_p95_ms": row["tbt_p95_ms"],
                "slo_feasible": row["tbt_p95_ms"] <= slo,
            } for row in subset]
            feasible = [point for point in points if point["slo_feasible"]]
            by_concurrency[str(concurrency)] = {
                "power_fit_coefficients_high_to_low": coefficients.tolist(),
                "power_fit_r_squared": r_squared(powers, np.polyval(coefficients, frequencies)),
                "measured_total_energy_optimum": min(points, key=lambda point: point["energy_j_per_decode_token"]),
                "measured_slo_feasible_energy_optimum": (
                    min(feasible, key=lambda point: point["energy_j_per_decode_token"])
                    if feasible else None
                ),
                "points": points,
            }
        power_models[str(length)] = {"by_concurrency": by_concurrency}

    idle = next(row for row in rows if row["phase"] == "idle_power")
    return {
        "decode_token_definition": "completion tokens after the disaggregated context-only handoff token",
        "tbt_slo_ms": slo,
        "idle_power": idle,
        "latency_models_by_input_tokens": latency_models,
        "power_models_by_input_tokens": power_models,
    }


async def main_async(args):
    config = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "measurements.jsonl"
    all_rows = []
    if rows_path.exists():
        all_rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        print(f"resuming from {len(all_rows)} recorded rows", flush=True)
    (args.output / "profile.yaml").write_text(args.config.read_text())
    prompts = PromptFactory(config["model"])
    clocks = ClockController(config["service_container"])
    gpu_ids = config["gpu_groups"]["decode"]
    run_nonce = args.output.name

    def record(row):
        all_rows.append(row)
        with rows_path.open("a") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    clocks.reset(gpu_ids)
    try:
        idle_rows = [row for row in all_rows if row["phase"] == "idle_power"]
        if idle_rows:
            idle_power = idle_rows[0]["average_power_w"]
        else:
            samples, elapsed, energy = await measure_idle(config, gpu_ids)
            idle_power = energy / elapsed
            record({
                "phase": "idle_power", "gpu_ids": gpu_ids, "elapsed_s": elapsed,
                "energy_j": energy, "average_power_w": idle_power,
                "telemetry": summarize_telemetry(samples, gpu_ids),
            })
            print("decode idle power profiling complete", flush=True)

        latency = config["decode_latency"]
        for frequency in latency["frequencies_mhz"]:
            clocks.lock(gpu_ids, frequency)
            await asyncio.sleep(config["settle_after_clock_change_s"])
            for length in latency["prompt_lengths"]:
                completed_repetitions = {
                    row["repetition"] for row in all_rows
                    if row["phase"] == "decode_latency"
                    and row["frequency_mhz"] == frequency
                    and row["requested_input_tokens"] == length
                }
                if len(completed_repetitions) >= latency["repetitions"]:
                    continue
                await warmup_requests(
                    config, prompts, length, latency["output_tokens"],
                    latency["warmup_repetitions"],
                    f"{run_nonce}-decode-latency-{frequency}-{length}",
                )
                for repetition in range(latency["repetitions"]):
                    if repetition in completed_repetitions:
                        continue
                    nonce = f"{run_nonce}-decode-latency-{frequency}-{length}-{repetition}"
                    actual = prompts.make(length, nonce)[1]
                    results, telemetry, request_elapsed, request_energy = await measured_requests(
                        config, prompts, [{"input_tokens": length, "output_tokens": latency["output_tokens"], "nonce": nonce}], gpu_ids
                    )
                    record({
                        "phase": "decode_latency", "gpu_ids": gpu_ids,
                        "frequency_mhz": frequency, "requested_input_tokens": length,
                        "actual_input_tokens": actual, "repetition": repetition,
                        "elapsed_s": request_elapsed, "energy_j": request_energy,
                        "telemetry": summarize_telemetry(telemetry, gpu_ids), **results[0],
                    })
                print(f"decode latency complete: {frequency} MHz, {length} tokens", flush=True)
            clocks.reset(gpu_ids)

        power = config["decode_power"]
        for length in power["prompt_lengths"]:
            for concurrency in power["concurrency_levels"]:
                for frequency in config["sm_frequencies_mhz"]:
                    if any(
                        row["phase"] == "decode_power"
                        and row["requested_input_tokens"] == length
                        and row["concurrency"] == concurrency
                        and row["frequency_mhz"] == frequency
                        for row in all_rows
                    ):
                        continue
                    clocks.lock(gpu_ids, frequency)
                    await asyncio.sleep(config["settle_after_clock_change_s"])
                    await warmup_requests(
                        config, prompts, length, power["output_tokens"],
                        power["warmup_repetitions"],
                        f"{run_nonce}-decode-power-{length}-{concurrency}-{frequency}",
                    )
                    results, telemetry, load_elapsed, load_energy = await measured_steady_load(
                        config, prompts, length, power["output_tokens"], concurrency,
                        power["duration_s"],
                        f"{run_nonce}-decode-power-{length}-{concurrency}-{frequency}", gpu_ids,
                    )
                    good = successful(results)
                    errors = len(results) - len(good)
                    decode_tokens = sum(row["observed_decode_tokens"] for row in good)
                    tbts = [value for row in good for value in row["tbt_values_ms"]]
                    dynamic_energy = max(0.0, load_energy - idle_power * load_elapsed)
                    actual = prompts.make(length, f"{run_nonce}-actual-{length}")[1]
                    record({
                        "phase": "decode_power", "gpu_ids": gpu_ids,
                        "frequency_mhz": frequency, "requested_input_tokens": length,
                        "actual_input_tokens": actual, "concurrency": concurrency,
                        "requests": len(results), "errors": errors,
                        "decode_tokens": decode_tokens, "elapsed_s": load_elapsed,
                        "energy_j": load_energy, "dynamic_energy_j": dynamic_energy,
                        "average_power_w": load_energy / load_elapsed,
                        "decode_tokens_per_second": decode_tokens / load_elapsed,
                        "energy_j_per_decode_token": load_energy / decode_tokens,
                        "dynamic_energy_j_per_decode_token": dynamic_energy / decode_tokens,
                        "tbt_mean_ms": statistics.fmean(tbts),
                        "tbt_median_ms": statistics.median(tbts),
                        "tbt_p95_ms": percentile(tbts, 95),
                        "telemetry": summarize_telemetry(telemetry, gpu_ids),
                    })
                    if errors:
                        raise RuntimeError(
                            f"decode length {length}, concurrency {concurrency}, frequency {frequency} had {errors}/{len(results)} errors"
                        )
                    print(f"decode power complete: {length} tokens, concurrency {concurrency}, {frequency} MHz", flush=True)
                    clocks.reset(gpu_ids)

        write_csv(args.output / "measurements.csv", all_rows)
        (args.output / "decode_models.json").write_text(
            json.dumps(fit_decode_models(all_rows, config), indent=2) + "\n"
        )
    finally:
        try:
            clocks.reset(gpu_ids)
        except subprocess.CalledProcessError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
