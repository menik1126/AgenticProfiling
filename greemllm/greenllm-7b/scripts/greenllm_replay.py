#!/usr/bin/env python3
"""Replay one trace with profile-driven Prefill and Decode DVFS control."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import statistics
import time
from bisect import bisect_left
from pathlib import Path

import aiohttp
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
BASE_REPLAY = WORKSPACE / "defaultnv-7b/scripts/replay.py"
spec = importlib.util.spec_from_file_location("baseline_replay", BASE_REPLAY)
baseline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baseline)


CLOCK_DAEMON_CODE = r'''import json, sys, time, pynvml
pynvml.nvmlInit()
handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
try:
    for line in sys.stdin:
        command = json.loads(line)
        started = time.perf_counter()
        for gpu in command["gpus"]:
            if command["frequency"] is None:
                pynvml.nvmlDeviceResetGpuLockedClocks(handles[gpu])
            else:
                pynvml.nvmlDeviceSetGpuLockedClocks(
                    handles[gpu], command["frequency"], command["frequency"]
                )
        print(json.dumps({"elapsed_ms": (time.perf_counter() - started) * 1000}), flush=True)
finally:
    pynvml.nvmlShutdown()
'''


def nearest_ceiling(values, value):
    values = sorted(values)
    position = bisect_left(values, value)
    return values[min(position, len(values) - 1)]


def interpolate(points, x):
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise AssertionError("unreachable interpolation interval")


class GreenLLMController:
    def __init__(self, config, output_dir):
        self.config = config
        self.control = config["controller"]
        self.frequencies = self.control["frequencies_mhz"]
        self.prefill_idle_frequency = self.control.get("prefill_idle_frequency_mhz")
        if (
            self.prefill_idle_frequency is not None
            and self.prefill_idle_frequency not in self.frequencies
        ):
            raise ValueError(
                "prefill_idle_frequency_mhz must be null or one of frequencies_mhz"
            )
        self.container = config["service_container"]
        self.output_dir = output_dir
        self.prefill_model = json.loads(Path(config["prefill_profile"]).read_text())
        self.decode_model = json.loads(Path(config["decode_profile"]).read_text())
        self.groups = {
            "prefill_short": self.control["prefill_short_gpus"],
            "prefill_long": self.control["prefill_long_gpus"],
            "decode": self.control["decode_gpus"],
        }
        self.prefill_active = {"prefill_short": {}, "prefill_long": {}}
        self.decode_active = {}
        self.token_events = []
        self.desired = {
            name: self.prefill_idle_frequency if name.startswith("prefill_") else None
            for name in self.groups
        }
        # Unknown sentinels force a real reset at startup instead of assuming
        # that clocks were left unlocked by a previous process.
        self.applied = {name: object() for name in self.groups}
        self.frequency_events = []
        self.control_samples = []
        self.task = None
        self.clock_daemon = None
        self.stop_event = asyncio.Event()

    async def _start_clock_daemon(self):
        self.clock_daemon = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i", self.container,
            "python3", "-u", "-c", CLOCK_DAEMON_CODE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _stop_clock_daemon(self):
        if self.clock_daemon is None:
            return
        self.clock_daemon.stdin.close()
        await self.clock_daemon.wait()
        self.clock_daemon = None

    async def _apply(self, group, frequency, reason):
        if self.applied[group] == frequency:
            return
        started = time.perf_counter()
        command = json.dumps({"gpus": self.groups[group], "frequency": frequency}) + "\n"
        self.clock_daemon.stdin.write(command.encode())
        await self.clock_daemon.stdin.drain()
        response = await self.clock_daemon.stdout.readline()
        if not response:
            error = (await self.clock_daemon.stderr.read()).decode(errors="replace")
            raise RuntimeError(f"NVML clock daemon exited: {error}")
        daemon_elapsed_ms = json.loads(response)["elapsed_ms"]
        ended = time.perf_counter()
        self.applied[group] = frequency
        self.frequency_events.append({
            "timestamp_s": ended,
            "group": group,
            "gpu_ids": self.groups[group],
            "frequency_mhz": frequency,
            "reason": reason,
            "apply_latency_ms": (ended - started) * 1000,
            "nvml_latency_ms": daemon_elapsed_ms,
        })

    def _prefill_latency(self, length, frequency):
        matrix = self.prefill_model["inverse_frequency_validation"]["points"]
        by_frequency = {}
        for row in matrix:
            by_frequency.setdefault(row["frequency_mhz"], []).append(
                (row["input_tokens"], row["median_ttft_ms"])
            )
        frequency_points = [
            (measured_frequency, interpolate(length_points, length))
            for measured_frequency, length_points in by_frequency.items()
        ]
        return interpolate(frequency_points, frequency)

    def _prefill_target_for_request(self, group, length, active_count):
        pool = "short_medium" if group == "prefill_short" else "long"
        concurrency_levels = [
            int(value) for value in
            self.prefill_model["power_models"][pool]["by_concurrency"]
        ]
        concurrency = nearest_ceiling(concurrency_levels, active_count)
        optimum = self.prefill_model["power_models"][pool]["by_concurrency"][str(concurrency)]["measured_energy_optimum"]["frequency_mhz"]
        slo = self.control["short_ttft_slo_ms"] if group == "prefill_short" else self.control["long_ttft_slo_ms"]
        budget = slo * self.control["prefill_slo_margin"]
        feasible = [
            frequency for frequency in self.frequencies
            if self._prefill_latency(length, frequency) <= budget
        ]
        minimum_safe = min(feasible) if feasible else max(self.frequencies)
        if active_count > max(concurrency_levels):
            return max(self.frequencies)
        return nearest_ceiling(self.frequencies, max(optimum, minimum_safe))

    def _refresh_prefill(self, group):
        active = self.prefill_active[group]
        if not active:
            self.desired[group] = self.prefill_idle_frequency
            return
        count = len(active)
        self.desired[group] = max(
            self._prefill_target_for_request(group, length, count)
            for length in active.values()
        )

    def prefill_start(self, request_key, length):
        group = "prefill_short" if length <= self.control["prefill_threshold_tokens"] else "prefill_long"
        self.prefill_active[group][request_key] = length
        self._refresh_prefill(group)
        return group

    def prefill_end(self, request_key, group):
        self.prefill_active[group].pop(request_key, None)
        self._refresh_prefill(group)

    def _decode_coarse_target(self):
        if not self.decode_active:
            return None
        active_count = len(self.decode_active)
        if active_count > 16:
            return max(self.frequencies)
        length_keys = [int(value) for value in self.decode_model["power_models_by_input_tokens"]]
        representative_length = statistics.median(self.decode_active.values())
        length = min(length_keys, key=lambda value: abs(value - representative_length))
        table = self.decode_model["power_models_by_input_tokens"][str(length)]["by_concurrency"]
        concurrency = nearest_ceiling([int(value) for value in table], active_count)
        optimum = table[str(concurrency)]["measured_slo_feasible_energy_optimum"]
        return optimum["frequency_mhz"] if optimum else max(self.frequencies)

    def decode_start(self, request_key, length):
        self.decode_active[request_key] = length

    def decode_end(self, request_key):
        self.decode_active.pop(request_key, None)

    def decode_token(self, timestamp, tbt_ms):
        self.token_events.append((timestamp, tbt_ms))

    def _decode_feedback_target(self, now):
        coarse = self._decode_coarse_target()
        if coarse is None:
            return None, 0.0, math.nan
        window = self.control["decode_window_ms"] / 1000
        recent = [(timestamp, tbt) for timestamp, tbt in self.token_events if timestamp >= now - window]
        self.token_events = [(timestamp, tbt) for timestamp, tbt in self.token_events if timestamp >= now - 6]
        tps = len(recent) / window
        valid_tbts = [tbt for _, tbt in recent if not math.isnan(tbt)]
        p95 = baseline.percentile(valid_tbts, 95)
        index = self.frequencies.index(coarse)
        if len(valid_tbts) >= 5 and p95 > self.control["tbt_slo_ms"]:
            index = min(index + 1, len(self.frequencies) - 1)
        elif len(valid_tbts) >= 5 and p95 < self.control["tbt_slo_ms"] * self.control["tbt_lower_margin"]:
            index = max(index - 1, 0)
        return self.frequencies[index], tps, p95

    async def _run(self):
        interval = self.control["control_interval_ms"] / 1000
        while not self.stop_event.is_set():
            now = time.perf_counter()
            decode_target, tps, tbt_p95 = self._decode_feedback_target(now)
            self.desired["decode"] = decode_target
            self.control_samples.append({
                "timestamp_s": now,
                "prefill_short_active": len(self.prefill_active["prefill_short"]),
                "prefill_long_active": len(self.prefill_active["prefill_long"]),
                "decode_active": len(self.decode_active),
                "decode_tps_200ms": tps,
                "decode_tbt_p95_ms_200ms": tbt_p95,
                "desired_frequencies_mhz": dict(self.desired),
            })
            for group in self.groups:
                desired = self.desired[group]
                if group.startswith("prefill_") and not self.prefill_active[group]:
                    reason = "idle_locked" if desired is not None else "idle"
                else:
                    reason = "idle" if desired is None else "profile_and_slo"
                if self.applied[group] != desired:
                    await self._apply(group, desired, reason)
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def start(self):
        await self._start_clock_daemon()
        for group in self.groups:
            await self._apply(group, None, "initial_reset")
        self.task = asyncio.create_task(self._run())

    async def stop(self):
        self.stop_event.set()
        task_error = None
        if self.task:
            try:
                await self.task
            except Exception as exc:
                task_error = exc
        try:
            for group in self.groups:
                try:
                    await self._apply(group, None, "final_reset")
                except Exception:
                    pass
        finally:
            await self._stop_clock_daemon()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, rows in (
            ("frequency_events.jsonl", self.frequency_events),
            ("controller_samples.jsonl", self.control_samples),
        ):
            with (self.output_dir / filename).open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        if task_error is not None:
            raise task_error


async def request_once(session, semaphore, config, prompt_factory, request, start_time, controller):
    scheduled = start_time + request["scheduled_offset_s"]
    await asyncio.sleep(max(0, scheduled - time.perf_counter()))
    async with semaphore:
        dispatched = time.perf_counter()
        prompt, actual_input_tokens = prompt_factory.make(request["input_tokens"])
        desired_output = max(1, int(request["output_tokens"]))
        request_key = str(request["request_id"])
        prefill_group = controller.prefill_start(request_key, actual_input_tokens)
        decode_started = False
        previous_token_time = None
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": True, "max_tokens": desired_output,
            "nvext": {"ignore_eos": True}, "temperature": 0,
        }
        token_times, stream_events, observed_output_tokens = [], 0, 0
        raw_observed_output_tokens = 0
        handoff_token_adjusted = False
        error, status = None, None
        try:
            async with session.post(config["endpoint"], json=payload) as response:
                status = response.status
                if status != 200:
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
                            except json.JSONDecodeError:
                                continue
                            stream_events += 1
                            completion = (decoded.get("usage") or {}).get("completion_tokens")
                            if not isinstance(completion, int) or completion <= raw_observed_output_tokens:
                                continue
                            raw_observed_output_tokens = completion
                            effective_completion = completion
                            if desired_output == 1 and completion == 2:
                                effective_completion = 1
                                handoff_token_adjusted = True
                            if effective_completion <= observed_output_tokens:
                                continue
                            now = time.perf_counter()
                            increment = effective_completion - observed_output_tokens
                            token_times.extend([now] * increment)
                            if not decode_started:
                                controller.prefill_end(request_key, prefill_group)
                                controller.decode_start(request_key, actual_input_tokens)
                                decode_started = True
                            for _ in range(increment):
                                tbt = math.nan if previous_token_time is None else (now - previous_token_time) * 1000
                                controller.decode_token(now, tbt)
                                previous_token_time = now
                            observed_output_tokens = effective_completion
        except Exception as exc:
            error = repr(exc)
        finally:
            if decode_started:
                controller.decode_end(request_key)
            else:
                controller.prefill_end(request_key, prefill_group)
        completed = time.perf_counter()
        if status == 200 and error is None and observed_output_tokens != desired_output:
            error = f"completion token mismatch: expected {desired_output}, observed {observed_output_tokens}"
        tbts_ms = [(b - a) * 1000 for a, b in zip(token_times, token_times[1:])]
        return {
            **request, "actual_input_tokens": actual_input_tokens,
            "observed_stream_events": stream_events,
            "observed_output_tokens": observed_output_tokens,
            "raw_observed_output_tokens": raw_observed_output_tokens,
            "handoff_token_adjusted": handoff_token_adjusted,
            "scheduled_lateness_ms": (dispatched - scheduled) * 1000,
            "ttft_ms": (token_times[0] - dispatched) * 1000 if token_times else math.nan,
            "tbt_p95_ms": baseline.percentile(tbts_ms, 95),
            "latency_ms": (completed - dispatched) * 1000,
            "http_status": status, "error": error,
        }


async def run(args):
    config = yaml.safe_load(args.config.read_text())
    if args.trace is not None:
        config["trace"] = str(args.trace.resolve())
    requests = [json.loads(line) for line in Path(config["trace"]).read_text().splitlines() if line]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "greenllm.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    prompt_factory = baseline.PromptFactory(config["model"])
    sampler = baseline.PowerSampler(config["power_interval_ms"] / 1000)
    semaphore = asyncio.Semaphore(config["max_inflight"])
    controller = GreenLLMController(config, args.output_dir)
    timeout = aiohttp.ClientTimeout(total=None, connect=60)
    start_time = time.perf_counter() + 2
    sampler.start()
    try:
        await controller.start()
        async with aiohttp.ClientSession(timeout=timeout) as session:
            results = await asyncio.gather(*[
                asyncio.create_task(request_once(
                    session, semaphore, config, prompt_factory, request,
                    start_time, controller,
                )) for request in requests
            ])
        end_time = time.perf_counter()
    finally:
        sampler.stop()
        if controller.clock_daemon is not None:
            await controller.stop()
    with (args.output_dir / "requests.jsonl").open("w") as stream:
        for row in results:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (args.output_dir / "power.jsonl").open("w") as stream:
        for row in sampler.samples:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = baseline.summarize(results, sampler.samples, start_time, end_time)
    summary["frequency_changes"] = len(controller.frequency_events)
    summary["controller_samples"] = len(controller.control_samples)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
