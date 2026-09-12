#!/usr/bin/env python3
"""Draw paper-style decode and runtime figures from the completed local runs.

The Figure 11 analogue deliberately labels the orange bars "profile-selected"
rather than GreenLLM: the available data are an offline fixed-clock sweep, not
an online defaultNV/GreenLLM TPS sweep like the paper's experiment.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "greenllm-profile-7b/results/decode-20260902T133755Z/measurements.csv"
GREEN = ROOT / "greenllm-7b/results/20260906T075348Z/alibaba_1qps"
BASELINE = ROOT / "defaultnv-7b/results/20260902T080638Z/alibaba_1qps"
OUT = ROOT / "greenllm-7b/figures/paper_style_1qps/decode"

BLUE = "#58A5E1"
ORANGE = "#F28E5B"
GREEN_LINE = "#009E73"
CYAN = "#00A6D6"
GRID = "#D8D8D8"
INK = "#222222"


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.75,
            "grid.color": GRID,
            "grid.linestyle": "--",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def request_bins(path: Path, bin_s: float, duration_s: float):
    """Approximate realized TPS by spreading each request's tokens over decode.

    The final replayer did not retain token timestamps.  This reconstruction is
    therefore an interval-average, not the paper's 20 ms emitted-token counter.
    """
    n = int(math.ceil(duration_s / bin_s))
    tokens = np.zeros(n)
    tbt_values: list[list[float]] = [[] for _ in range(n)]
    for row in jsonl(path):
        if row.get("http_status") != 200 or row.get("latency_ms") is None:
            continue
        start = float(row["scheduled_offset_s"]) + float(row.get("scheduled_lateness_ms", 0.0)) / 1000
        decode_start = start + float(row.get("ttft_ms") or 0.0) / 1000
        decode_end = start + float(row["latency_ms"]) / 1000
        count = float(row.get("output_tokens") or row.get("observed_output_tokens") or 0)
        span = max(decode_end - decode_start, 1e-6)
        rate = count / span
        first = max(0, int(decode_start // bin_s))
        last = min(n - 1, int(decode_end // bin_s))
        for idx in range(first, last + 1):
            overlap = max(0.0, min(decode_end, (idx + 1) * bin_s) - max(decode_start, idx * bin_s))
            tokens[idx] += rate * overlap
        # A one-token completion has no inter-token interval.  The legacy
        # disaggregated endpoint exposed its context-only handoff as a second
        # stream event, but that must not enter the TBT population.
        tbt = row.get("tbt_p95_ms")
        if tbt is not None and float(row.get("output_tokens") or 0) > 1:
            idx = min(n - 1, max(0, int(start // bin_s)))
            tbt_values[idx].append(float(tbt))
    x = (np.arange(n) + 0.5) * bin_s
    tps = tokens / bin_s
    tbt = np.array([percentile(v, 95) for v in tbt_values])
    return x, tps, tbt


def power_bins(path: Path, groups: dict[str, set[int]], bin_s: float, duration_s: float):
    rows = jsonl(path)
    first = next(rows)
    t0 = float(first["timestamp_s"])
    n = int(math.ceil(duration_s / bin_s))
    sums = {name: {"clock": np.zeros(n), "power": np.zeros(n), "count": np.zeros(n)} for name in groups}

    def consume(row):
        idx = int((float(row["timestamp_s"]) - t0) // bin_s)
        if not (0 <= idx < n):
            return
        by_id = {int(g["gpu"]): g for g in row["gpus"]}
        for name, ids in groups.items():
            selected = [by_id[i] for i in ids if i in by_id]
            if not selected:
                continue
            sums[name]["clock"][idx] += np.mean([float(g["sm_clock_mhz"]) for g in selected])
            sums[name]["power"][idx] += np.sum([float(g["power_w"]) for g in selected])
            sums[name]["count"][idx] += 1

    consume(first)
    for row in rows:
        consume(row)
    result = {}
    for name, metrics in sums.items():
        count = metrics.pop("count")
        result[name] = {
            key: np.divide(value, count, out=np.full(n, np.nan), where=count > 0)
            for key, value in metrics.items()
        }
    return (np.arange(n) + 0.5) * bin_s, result


def controller_bins(path: Path, bin_s: float, duration_s: float):
    rows = jsonl(path)
    first = next(rows)
    t0 = float(first["timestamp_s"])
    n = int(math.ceil(duration_s / bin_s))
    values = {"short": np.zeros(n), "long": np.zeros(n), "count": np.zeros(n)}

    def consume(row):
        idx = int((float(row["timestamp_s"]) - t0) // bin_s)
        if 0 <= idx < n:
            values["short"][idx] += float(row["prefill_short_active"])
            values["long"][idx] += float(row["prefill_long_active"])
            values["count"][idx] += 1

    consume(first)
    for row in rows:
        consume(row)
    count = values["count"]
    return (np.arange(n) + 0.5) * bin_s, {
        key: np.divide(value, count, out=np.full(n, np.nan), where=count > 0)
        for key, value in values.items()
        if key != "count"
    }


def plot_figure1_local() -> None:
    duration, bin_s = 1810.0, 10.0
    x_b, tps_b, tbt_b = request_bins(BASELINE / "requests.jsonl", bin_s, duration)
    x_g, tps_g, tbt_g = request_bins(GREEN / "requests.jsonl", bin_s, duration)
    xp_b, p_b = power_bins(BASELINE / "power.jsonl", {"decode": {0, 1, 2, 3}}, bin_s, duration)
    xp_g, p_g = power_bins(GREEN / "power.jsonl", {"decode": {0, 1, 2, 3}}, bin_s, duration)

    fig, axes = plt.subplots(3, 1, figsize=(7.15, 5.25), sharex=True)
    axes[0].plot(x_b / 60, tps_b, color=BLUE, lw=1.2, label="defaultNV")
    axes[0].plot(x_g / 60, tps_g, color=ORANGE, lw=1.2, label="GreenLLM")
    axes[0].set_ylabel("Decode TPS")
    axes[0].legend(ncol=2, frameon=False, loc="upper right")

    axes[1].plot(xp_b / 60, p_b["decode"]["clock"], color=BLUE, lw=1.15, label="defaultNV")
    axes[1].plot(xp_g / 60, p_g["decode"]["clock"], color=ORANGE, lw=1.15, label="GreenLLM")
    axes[1].set_ylabel("Mean decode GPU\nSM clock (MHz)")
    axes[1].set_ylim(150, 2050)

    axes[2].plot(x_b / 60, tbt_b, color=BLUE, lw=1.05, marker="o", ms=1.7, label="defaultNV")
    axes[2].plot(x_g / 60, tbt_g, color=ORANGE, lw=1.05, marker="o", ms=1.7, label="GreenLLM")
    axes[2].axhline(100, color=CYAN, lw=1.6, label="100 ms SLO")
    axes[2].set_ylabel("Per-bin P95 TBT (ms)")
    axes[2].set_xlabel("Time since trace start (min)")
    axes[2].set_ylim(0, 300)
    clipped = int(np.sum(tbt_b > 300) + np.sum(tbt_g > 300))
    axes[2].text(
        0.01,
        0.96,
        f"{clipped} startup bin(s) above 300 ms clipped for readability",
        transform=axes[2].transAxes,
        va="top",
        fontsize=6.5,
        color="#555555",
    )
    axes[2].legend(ncol=3, frameon=False, loc="upper right")
    for ax in axes:
        ax.grid(True)
        ax.set_xlim(0, 30.2)
    axes[0].set_title("Local Figure 1 analogue — Alibaba 1 QPS trace (not a sinusoidal workload)")
    fig.text(
        0.5,
        0.005,
        "TPS is reconstructed as a 10 s interval-average because token emission timestamps were not retained.",
        ha="center",
        fontsize=6.7,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1), h_pad=0.55)
    save(fig, "fig1_local_alibaba_decode_dynamics")


def profile_rows():
    with PROFILE.open() as handle:
        return [row for row in csv.DictReader(handle) if row["phase"] == "decode_power"]


def plot_figure3b_local() -> None:
    rows = profile_rows()
    by_config: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_config[(int(row["concurrency"]), int(row["requested_input_tokens"]))].append(row)

    curves: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (concurrency, _), group in by_config.items():
        energies = np.array([float(r["energy_j_per_decode_token"]) for r in group])
        minimum = float(np.min(energies))
        for row in group:
            curves[concurrency][int(row["frequency_mhz"])].append(
                float(row["energy_j_per_decode_token"]) / minimum
            )

    fig, ax = plt.subplots(figsize=(5.05, 3.25))
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(curves)))
    for color, concurrency in zip(colors, sorted(curves)):
        freqs = np.array(sorted(curves[concurrency]))
        means = np.array([np.mean(curves[concurrency][f]) for f in freqs])
        lows = np.array([np.min(curves[concurrency][f]) for f in freqs])
        highs = np.array([np.max(curves[concurrency][f]) for f in freqs])
        ax.fill_between(freqs, lows, highs, color=color, alpha=0.10, linewidth=0)
        ax.plot(freqs, means, color=color, lw=1.35, marker="o", ms=2.7, label=f"Concurrency {concurrency}")
        best = int(np.argmin(means))
        ax.scatter(freqs[best], means[best], s=28, facecolor="white", edgecolor=color, linewidth=1.1, zorder=5)
    target_c = 8
    target_freqs = np.array(sorted(curves[target_c]))
    target_means = np.array([np.mean(curves[target_c][f]) for f in target_freqs])
    target_i = int(np.argmin(target_means))
    ax.annotate(
        "Energy-minimum band",
        xy=(target_freqs[target_i], target_means[target_i]),
        xytext=(target_freqs[target_i], target_means[target_i] + 0.17),
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=INK,
        arrowprops={"arrowstyle": "-|>", "color": CYAN, "lw": 1.4},
    )
    ax.set_xlabel("SM frequency (MHz)")
    ax.set_ylabel(r"Normalized decode energy  $E/E_{min}$")
    ax.set_title("Local Figure 3(b) analogue — fixed-clock decode profiling")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)
    ax.set_xlim(560, 1990)
    ax.text(
        0.015,
        0.985,
        "Line: mean across 4 context lengths\nShade: min–max across contexts",
        transform=ax.transAxes,
        va="top",
        fontsize=6.7,
        color="#555555",
    )
    fig.tight_layout()
    save(fig, "fig3b_local_normalized_decode_energy")


def plot_figure11_local() -> None:
    rows = profile_rows()
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["concurrency"]), int(row["requested_input_tokens"]))].append(row)

    aggregated = defaultdict(lambda: {"tps": [], "base_tbt": [], "selected_tbt": [], "saving": [], "freq": []})
    for (concurrency, _), group in groups.items():
        baseline = max(group, key=lambda row: int(row["frequency_mhz"]))
        feasible = [row for row in group if float(row["tbt_p95_ms"]) <= 100.0]
        selected = min(feasible, key=lambda row: float(row["energy_j_per_decode_token"]))
        dst = aggregated[concurrency]
        dst["tps"].append(float(baseline["decode_tokens_per_second"]))
        dst["base_tbt"].append(float(baseline["tbt_p95_ms"]))
        dst["selected_tbt"].append(float(selected["tbt_p95_ms"]))
        base_energy = float(baseline["energy_j_per_decode_token"])
        dst["saving"].append(100 * (base_energy - float(selected["energy_j_per_decode_token"])) / base_energy)
        dst["freq"].append(int(selected["frequency_mhz"]))

    concurrencies = sorted(aggregated)
    tps = np.array([np.mean(aggregated[c]["tps"]) for c in concurrencies])
    base_tbt = np.array([np.mean(aggregated[c]["base_tbt"]) for c in concurrencies])
    selected_tbt = np.array([np.mean(aggregated[c]["selected_tbt"]) for c in concurrencies])
    savings = np.array([np.mean(aggregated[c]["saving"]) for c in concurrencies])
    x = np.arange(len(concurrencies))
    width = 0.34

    fig, ax = plt.subplots(figsize=(5.35, 3.25))
    ax.bar(x - width / 2, base_tbt, width, color=BLUE, edgecolor=INK, linewidth=0.55, hatch="///", label="defaultNV (1950 MHz)")
    ax.bar(x + width / 2, selected_tbt, width, color=ORANGE, edgecolor=INK, linewidth=0.55, hatch="\\\\", label="Profile-selected clock")
    ax.axhline(100, color=CYAN, lw=1.6, label="100 ms SLO")
    ax.set_ylabel("Mean P95 TBT (ms)")
    ax.set_xlabel("Measured decode TPS at 1950 MHz [concurrency]")
    ax.set_xticks(x, [f"{value:.0f}\n[{c}]" for value, c in zip(tps, concurrencies)])
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y")

    ax2 = ax.twinx()
    ax2.plot(x, savings, color=GREEN_LINE, marker="o", ms=3.5, lw=1.3, label="Energy saving")
    ax2.set_ylabel("Energy saving (%)", color=GREEN_LINE)
    ax2.tick_params(axis="y", colors=GREEN_LINE)
    ax2.spines["right"].set_color(GREEN_LINE)
    ax2.set_ylim(0, max(32, math.ceil(float(np.max(savings)) / 5) * 5 + 2))
    for idx, c in enumerate(concurrencies):
        freqs = aggregated[c]["freq"]
        label = f"{min(freqs)}" if min(freqs) == max(freqs) else f"{min(freqs)}–{max(freqs)}"
        ax.text(idx + width / 2, selected_tbt[idx] + 2.0, f"{label} MHz", ha="center", va="bottom", fontsize=5.9, rotation=35)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    order = [l1.index("defaultNV (1950 MHz)"), l1.index("Profile-selected clock"), l1.index("100 ms SLO")]
    handles = [h1[i] for i in order] + h2
    labels = [l1[i] for i in order] + l2
    ax.legend(handles, labels, ncol=2, frameon=False, loc="upper left")
    ax.set_title("Local Figure 11 analogue — offline profile selection, not an online TPS sweep")
    fig.tight_layout()
    save(fig, "fig11_local_decode_profile_selection")


def plot_idle_prefill_diagnostic() -> None:
    duration, bin_s = 1810.0, 5.0
    x_c, active = controller_bins(GREEN / "controller_samples.jsonl", bin_s, duration)
    x_p, power = power_bins(
        GREEN / "power.jsonl",
        {"short": {4, 5}, "long": {6, 7}},
        bin_s,
        duration,
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.15, 5.25), sharex=True)
    axes[0].plot(x_c / 60, active["short"], color=BLUE, lw=0.95, label="Short prefill (GPU 4–5)")
    axes[0].plot(x_c / 60, active["long"], color=ORANGE, lw=0.95, label="Long prefill (GPU 6–7)")
    axes[0].set_ylabel("Mean active\nrequests")
    axes[0].legend(ncol=2, frameon=False, loc="upper right")

    axes[1].plot(x_p / 60, power["short"]["clock"], color=BLUE, lw=0.95)
    axes[1].plot(x_p / 60, power["long"]["clock"], color=ORANGE, lw=0.95)
    axes[1].axhline(210, color=CYAN, lw=1.1, ls="--", label="Measured idle-profile clock (210 MHz)")
    axes[1].set_ylabel("Mean SM clock\nper GPU (MHz)")
    axes[1].set_ylim(150, 2050)
    axes[1].legend(frameon=False, loc="lower right")

    axes[2].plot(x_p / 60, power["short"]["power"], color=BLUE, lw=0.95, label="Short pool")
    axes[2].plot(x_p / 60, power["long"]["power"], color=ORANGE, lw=0.95, label="Long pool")
    axes[2].set_ylabel("Pool power (W)")
    axes[2].set_xlabel("Time since trace start (min)")
    axes[2].set_ylim(bottom=0)

    idle_short = active["short"] == 0
    idle_long = active["long"] == 0
    short_idle_clock = np.nanmean(power["short"]["clock"][idle_short])
    long_idle_clock = np.nanmean(power["long"]["clock"][idle_long])
    short_idle_power = np.nanmean(power["short"]["power"][idle_short])
    long_idle_power = np.nanmean(power["long"]["power"][idle_long])
    axes[2].text(
        0.01,
        0.96,
        f"Idle-bin means: short {short_idle_clock:.0f} MHz / {short_idle_power:.0f} W; "
        f"long {long_idle_clock:.0f} MHz / {long_idle_power:.0f} W",
        transform=axes[2].transAxes,
        va="top",
        fontsize=6.6,
        color="#555555",
    )
    for ax in axes:
        ax.grid(True)
        ax.set_xlim(0, 30.2)
    axes[0].set_title("Prefill idle-state diagnostic — GreenLLM Alibaba 1 QPS")
    fig.tight_layout(h_pad=0.55)
    save(fig, "diagnostic_prefill_idle_clock_power")


def main() -> None:
    paper_style()
    plot_figure1_local()
    plot_figure3b_local()
    plot_figure11_local()
    plot_idle_prefill_diagnostic()
    print(f"Wrote 8 files to {OUT}")


if __name__ == "__main__":
    main()
