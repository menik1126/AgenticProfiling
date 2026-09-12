#!/usr/bin/env python3
"""Paper-style plots for the local Alibaba 1 QPS end-to-end experiment.

The plots are explicitly labelled as a local reproduction.  The three input
directories are intentionally fixed so rerunning this script cannot silently
mix experiment generations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/shenhaojia/energy")
OUT = ROOT / "greenllm-7b/figures/paper_style_1qps/e2e"
RUNS = {
    "DefaultNV": ROOT / "defaultnv-7b/results/20260902T080638Z/alibaba_1qps",
    "PrefillSplit": ROOT / "prefillsplit-7b/results/20260902T090234Z/alibaba_1qps",
    "GreenLLM": ROOT / "greenllm-7b/results/20260906T075348Z/alibaba_1qps",
}

# Matched to the paper's blue/orange palette. PrefillSplit is neutral because
# the paper's Figure 5 only compares two series.
COLORS = {
    "DefaultNV": "#58A5E1",
    "PrefillSplit": "#8C8C8C",
    "GreenLLM": "#F28E5B",
    "saving": "#009E73",
    "slo": "#00A6D6",
    "prefill": "#F28E5B",
    "decode": "#58A5E1",
}
HANDOFF_ERROR = "completion token mismatch: expected 1, observed 2"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_valid_request(row: dict) -> bool:
    if row.get("http_status") != 200:
        return False
    error = row.get("error")
    return error is None or (
        error == HANDOFF_ERROR
        and row.get("output_tokens") == 1
        and row.get("observed_output_tokens") == 2
    )


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x)


def integrate_power(rows: list[dict], gpu_ids: set[int]) -> float:
    """Integrate the selected GPUs' sampled power and return Wh."""
    timestamps = np.asarray([r["timestamp_s"] for r in rows], dtype=float)
    watts = np.asarray(
        [sum(g["power_w"] for g in r["gpus"] if g["gpu"] in gpu_ids) for r in rows],
        dtype=float,
    )
    order = np.argsort(timestamps)
    return float(np.trapezoid(watts[order], timestamps[order]) / 3600.0)


def binned_power(rows: list[dict], width_s: float = 10.0) -> tuple[np.ndarray, ...]:
    t = np.asarray([r["timestamp_s"] for r in rows], dtype=float)
    t -= t.min()
    total = np.asarray([sum(g["power_w"] for g in r["gpus"]) for r in rows])
    decode = np.asarray([sum(g["power_w"] for g in r["gpus"] if g["gpu"] < 4) for r in rows])
    prefill = total - decode
    bins = np.floor(t / width_s).astype(int)
    unique = np.unique(bins)
    centers = (unique + 0.5) * width_s / 60.0
    aggregate = lambda a: np.asarray([a[bins == b].mean() for b in unique])
    return centers, aggregate(total), aggregate(decode), aggregate(prefill)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ttft_distribution(requests: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.0))
    groups = [
        ("(a) Short/Medium prompts ($L \\leq 1024$)", lambda r: r["input_tokens"] <= 1024, 0.4),
        ("(b) Long prompts ($L > 1024$)", lambda r: r["input_tokens"] > 1024, 2.0),
    ]
    for ax, (title, select, slo_s) in zip(axes, groups):
        series = {
            method: [r["ttft_ms"] / 1000 for r in rows if is_valid_request(r) and select(r)]
            for method, rows in requests.items()
        }
        upper = max(slo_s * 1.12, max(percentile(v, 99.5) for v in series.values()))
        bins = np.linspace(0, upper, 48)
        for method, values in series.items():
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.28,
                color=COLORS[method],
                edgecolor=COLORS[method],
                linewidth=1.0,
                label=method,
            )
        ax.axvline(slo_s, color=COLORS["slo"], linestyle=":", linewidth=1.7, label="SLO")
        ax.text(
            slo_s,
            ax.get_ylim()[1] * 0.72,
            "SLO",
            color=COLORS["slo"],
            ha="left",
            va="top",
        )
        ax.set(title=title, xlabel="TTFT (s)", ylabel="Density", xlim=(0, upper))
        ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle("Local reproduction — Alibaba 1 QPS", y=1.01, fontsize=10)
    fig.tight_layout()
    save(fig, "ttft_distribution_by_prompt_class")


def plot_ttft_cdf(requests: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    groups = [
        ("(a) Short/Medium", lambda r: r["input_tokens"] <= 1024, 400),
        ("(b) Long", lambda r: r["input_tokens"] > 1024, 2000),
    ]
    for ax, (title, select, slo_ms) in zip(axes, groups):
        all_values = []
        for method, rows in requests.items():
            values = [r["ttft_ms"] for r in rows if is_valid_request(r) and select(r)]
            all_values.extend(values)
            x, y = ecdf(values)
            ax.plot(x, y * 100, color=COLORS[method], linewidth=1.6, label=method)
        ax.axvline(slo_ms, color=COLORS["slo"], linestyle=":", linewidth=1.7, label="SLO")
        ax.set_xlim(0, max(slo_ms * 1.12, percentile(all_values, 99.5)))
        ax.set_ylim(0, 101)
        ax.set(title=title, xlabel="TTFT (ms)")
    axes[0].set_ylabel("Requests completed (%)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("TTFT CDF — Local reproduction / Alibaba 1 QPS", y=1.03, fontsize=10)
    fig.tight_layout()
    save(fig, "ttft_cdf_by_prompt_class")


def derive_metrics(requests: dict[str, list[dict]], power: dict[str, list[dict]]) -> dict:
    metrics = {}
    for method in RUNS:
        valid = [r for r in requests[method] if is_valid_request(r)]
        sm = [r for r in valid if r["input_tokens"] <= 1024]
        long = [r for r in valid if r["input_tokens"] > 1024]
        # One-token requests have no true inter-token interval. The old replay's
        # extra context-only handoff event must not be treated as decode TBT.
        tbt = [r for r in valid if r["output_tokens"] > 1 and r.get("tbt_p95_ms") is not None]
        decode_wh = integrate_power(power[method], set(range(4)))
        prefill_wh = integrate_power(power[method], set(range(4, 8)))
        metrics[method] = {
            "requests": len(requests[method]),
            "successful_requests_corrected": len(valid),
            "handoff_token_adjustments": sum(r.get("error") == HANDOFF_ERROR for r in valid),
            "ttft_pass_pct": 100
            * (sum(r["ttft_ms"] <= 400 for r in sm) + sum(r["ttft_ms"] <= 2000 for r in long))
            / len(valid),
            "tbt_pass_pct": 100 * sum(r["tbt_p95_ms"] <= 100 for r in tbt) / len(tbt),
            "tbt_eligible_requests": len(tbt),
            "ttft_p95_ms": percentile([r["ttft_ms"] for r in valid], 95),
            "tbt_p95_of_request_p95_ms": percentile([r["tbt_p95_ms"] for r in tbt], 95),
            "decode_energy_wh": decode_wh,
            "prefill_energy_wh": prefill_wh,
            "total_energy_wh": decode_wh + prefill_wh,
        }
    baseline = metrics["DefaultNV"]["total_energy_wh"]
    for values in metrics.values():
        values["relative_total_energy"] = values["total_energy_wh"] / baseline
        values["energy_saving_pct"] = 100 * (1 - values["total_energy_wh"] / baseline)
    return metrics


def plot_table3(metrics: dict) -> None:
    methods = list(RUNS)
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))

    decode = np.asarray([metrics[m]["decode_energy_wh"] for m in methods])
    prefill = np.asarray([metrics[m]["prefill_energy_wh"] for m in methods])
    axes[0].bar(x, decode, color=COLORS["decode"], edgecolor="white", hatch="//", label="Decode")
    axes[0].bar(x, prefill, bottom=decode, color=COLORS["prefill"], edgecolor="white", hatch="..", label="Prefill")
    for i, total in enumerate(decode + prefill):
        axes[0].text(i, total + 8, f"{total:.1f}", ha="center", va="bottom", fontsize=8)
    axes[0].set(title="(a) GPU energy", ylabel="Energy (Wh)", xticks=x, xticklabels=methods)
    axes[0].legend(frameon=False, loc="lower left")

    saving = [metrics[m]["energy_saving_pct"] for m in methods]
    bars = axes[1].bar(x, saving, color=[COLORS[m] for m in methods], edgecolor="white")
    for bar, value in zip(bars, saving):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}%", ha="center", fontsize=8)
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].set(title="(b) Energy saving", ylabel="vs. DefaultNV (%)", xticks=x, xticklabels=methods)

    width = 0.34
    ttft = [metrics[m]["ttft_pass_pct"] for m in methods]
    tbt = [metrics[m]["tbt_pass_pct"] for m in methods]
    axes[2].bar(x - width / 2, ttft, width, color=COLORS["slo"], edgecolor="white", hatch="//", label="TTFT")
    axes[2].bar(x + width / 2, tbt, width, color=COLORS["saving"], edgecolor="white", hatch="..", label="TBT")
    for xpos, value in zip(np.r_[x - width / 2, x + width / 2], np.r_[ttft, tbt]):
        axes[2].text(xpos, value - 2.0, f"{value:.1f}", ha="center", va="top", rotation=90, color="white", fontsize=7)
    axes[2].set(title="(c) SLO compliance", ylabel="Pass rate (%)", ylim=(90, 100.8), xticks=x, xticklabels=methods)
    axes[2].legend(frameon=False, loc="lower left")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Local Table 3 view — Alibaba 1 QPS", y=1.04, fontsize=10)
    fig.tight_layout()
    save(fig, "table3_local_chat_1qps")


def plot_power_timeseries(power: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.3, 5.7), sharex=True, sharey=True)
    for ax, method in zip(axes, RUNS):
        minutes, total, decode, prefill = binned_power(power[method])
        ax.plot(minutes, total, color="#333333", linewidth=1.2, label="Total")
        ax.fill_between(minutes, 0, decode, color=COLORS["decode"], alpha=0.45, label="Decode GPUs 0–3")
        ax.fill_between(minutes, decode, decode + prefill, color=COLORS["prefill"], alpha=0.42, label="Prefill GPUs 4–7")
        ax.set_title(method, loc="left", color=COLORS[method], fontweight="bold")
        ax.set_ylabel("Power (W)")
    axes[-1].set_xlabel("Elapsed time (min)")
    axes[0].legend(frameon=False, ncol=3, loc="upper center")
    fig.suptitle("10 s mean GPU power — Local reproduction / Alibaba 1 QPS", y=1.01, fontsize=10)
    fig.tight_layout()
    save(fig, "power_timeseries_all_methods")


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    requests = {method: load_jsonl(path / "requests.jsonl") for method, path in RUNS.items()}
    power = {method: load_jsonl(path / "power.jsonl") for method, path in RUNS.items()}
    metrics = derive_metrics(requests, power)
    with (OUT / "metrics.json").open("w") as handle:
        json.dump(
            {
                "scope": "Local reproduction / Alibaba 1 QPS",
                "input_runs": {method: str(path) for method, path in RUNS.items()},
                "accounting_note": (
                    "The 92 output_tokens=1 / observed_output_tokens=2 context-handoff records "
                    "per method are treated as successful; they are excluded from TBT because a "
                    "one-token completion has no true inter-token interval."
                ),
                "methods": metrics,
            },
            handle,
            indent=2,
        )
    plot_ttft_distribution(requests)
    plot_ttft_cdf(requests)
    plot_table3(metrics)
    plot_power_timeseries(power)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
