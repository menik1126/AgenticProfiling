#!/usr/bin/env python3
"""Render paper-style figures from the local Prefill profiling sweep.

This script intentionally plots only measured local data.  In particular, the
Figure 10 counterpart is labelled as an approximation: the profiling sweep is
closed-loop and records aggregate throughput, not per-request TTFT samples, so
mean request time is estimated with Little's-law proxy C / throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SOURCE_DEFAULT = Path(
    "/home/shenhaojia/energy/greenllm-profile-7b/results/20260902T125826Z"
)
OUTPUT_DEFAULT = Path(
    "/home/shenhaojia/energy/greenllm-7b/figures/paper_style_1qps/prefill"
)

BLUE = "#58A5E1"
ORANGE = "#F28E5B"
GREEN = "#009E73"
CYAN = "#00A6D6"
PURPLE = "#5E3C99"
CONCURRENCIES = (1, 4, 8, 16)
CONC_COLORS = {
    1: "#482878",
    4: "#31688E",
    8: "#35B779",
    16: "#FDE725",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D8D8D8",
            "grid.linestyle": "--",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: mpl.figure.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("pdf", None), ("png", 300)):
        fig.savefig(output / f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_data(source: Path) -> tuple[dict, list[dict[str, str]]]:
    with (source / "prefill_models.json").open() as handle:
        models = json.load(handle)
    with (source / "measurements.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return models, rows


def plot_fig3a(models: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharey=False)
    pool_titles = (("short_medium", "Short (65-token prompt)"), ("long", "Long (6,131-token prompt)"))

    for panel, (ax, (pool, title)) in enumerate(zip(axes, pool_titles)):
        pool_model = models["power_models"][pool]
        for concurrency in CONCURRENCIES:
            points = pool_model["by_concurrency"][str(concurrency)]["points"]
            freq = np.asarray([p["frequency_mhz"] for p in points], dtype=float)
            energy = np.asarray([p["energy_j_per_request"] for p in points], dtype=float)
            normalized = energy / energy.min()
            ax.plot(
                freq,
                normalized,
                marker="o",
                markersize=3.2,
                linewidth=1.15,
                color=CONC_COLORS[concurrency],
                markerfacecolor="white",
                markeredgewidth=0.8,
                label=f"Concurrency {concurrency}",
            )
            optimum = int(np.argmin(energy))
            ax.scatter(
                [freq[optimum]],
                [normalized[optimum]],
                marker="v",
                s=30,
                facecolor=CONC_COLORS[concurrency],
                edgecolor="black",
                linewidth=0.35,
                zorder=4,
            )
            if concurrency == 8:
                arrow_x = freq[optimum]
                arrow_y = normalized[optimum]
        ax.set_title(f"({chr(97 + panel)}) {title}", loc="left", fontweight="bold")
        ax.set_xlabel("SM frequency (MHz)")
        ax.set_ylabel(r"Normalized prefill energy ($E/E_{min}$)")
        ax.set_xticks([600, 900, 1200, 1500, 1800, 1950])
        ax.set_xlim(560, 1990)
        ax.set_ylim(bottom=0.96)
        ax.annotate(
            "Optimal frequency\nregion",
            xy=(arrow_x, arrow_y + 0.006),
            xytext=(arrow_x, arrow_y + 0.15),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#222222",
            arrowprops={"arrowstyle": "-|>", "color": "#0072B2", "lw": 1.5},
        )
        if panel == 0:
            ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Local counterpart of Fig. 3a — Prefill energy/frequency profiling", y=1.02)
    fig.tight_layout()
    save_figure(fig, output, "fig3a_prefill_energy_frequency")


def plot_fig7(models: dict, rows: list[dict[str, str]], output: Path) -> None:
    latency = models["latency_model"]
    ref = int(latency["reference_frequency_mhz"])
    measured = [
        row
        for row in rows
        if row["phase"] == "prefill_latency" and int(row["frequency_mhz"]) == ref
    ]
    lengths = np.asarray([float(row["actual_input_tokens"]) for row in measured])
    times = np.asarray([float(row["ttft_ms"]) / 1000.0 for row in measured])
    median_points = latency["median_points"]
    median_x = np.asarray([p["input_tokens"] for p in median_points], dtype=float)
    median_y = np.asarray([p["median_ttft_ms"] / 1000.0 for p in median_points])
    a, b, c = (
        latency["coefficients"]["a"] / 1000.0,
        latency["coefficients"]["b"] / 1000.0,
        latency["coefficients"]["c"] / 1000.0,
    )
    curve_x = np.linspace(0, lengths.max() * 1.015, 500)
    curve_y = a * curve_x**2 + b * curve_x + c
    fitted_measured = a * lengths**2 + b * lengths + c
    rmse = float(np.sqrt(np.mean((times - fitted_measured) ** 2)))

    fig, ax = plt.subplots(figsize=(4.5, 3.15))
    ax.scatter(
        lengths,
        times,
        marker="o",
        s=18,
        facecolors="none",
        edgecolors=BLUE,
        linewidths=0.9,
        label="Measurements",
    )
    ax.scatter(median_x, median_y, s=18, color=BLUE, zorder=3, label="Median by length")
    ax.plot(curve_x, curve_y, color=PURPLE, linewidth=1.7, label=r"Quadratic fit: $aL^2+bL+c$")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("TTFT (s)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.05,
        (
            rf"$a={a:.2e}$ s/token$^2$" "\n"
            rf"$b={b:.2e}$ s/token, $c={c:.3f}$ s" "\n"
            rf"$R^2={latency['r_squared']:.5f}$, RMSE={rmse:.3f} s"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
    )
    ax.set_title(f"Prefill latency model at {ref} MHz")
    fig.tight_layout()
    save_figure(fig, output, "fig7_prefill_latency_model")


def plot_fig8(models: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), sharex=True)
    pool_titles = (("short_medium", "Short pool (2 GPUs)"), ("long", "Long pool (2 GPUs)"))

    for panel, (ax, (pool, title)) in enumerate(zip(axes, pool_titles)):
        pool_model = models["power_models"][pool]
        r2_values = []
        for concurrency in CONCURRENCIES:
            profile = pool_model["by_concurrency"][str(concurrency)]
            points = profile["points"]
            freq = np.asarray([p["frequency_mhz"] for p in points], dtype=float)
            power = np.asarray([p["average_power_w"] for p in points], dtype=float)
            coeff = np.asarray(profile["coefficients_high_to_low"], dtype=float)
            fit_x = np.linspace(freq.min(), freq.max(), 300)
            ax.scatter(
                freq,
                power,
                s=13,
                facecolors="none",
                edgecolors=CONC_COLORS[concurrency],
                linewidths=0.75,
            )
            ax.plot(
                fit_x,
                np.polyval(coeff, fit_x),
                color=CONC_COLORS[concurrency],
                linewidth=1.25,
                label=f"C={concurrency}",
            )
            r2_values.append(float(profile["power_fit_r_squared"]))
        ax.set_title(f"({chr(97 + panel)}) {title}", loc="left", fontweight="bold")
        ax.set_xlabel("SM frequency (MHz)")
        ax.set_ylabel("Average pool power (W)")
        ax.set_xticks([600, 900, 1200, 1500, 1800, 1950])
        ax.text(
            0.03,
            0.95,
            rf"cubic fits, $R^2={min(r2_values):.3f}$–{max(r2_values):.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
        )
        ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 0.83))
    fig.suptitle("Local counterpart of Fig. 8 — Prefill power models", y=1.02)
    fig.tight_layout()
    save_figure(fig, output, "fig8_prefill_power_model")


def plot_fig10_approx(models: dict, output: Path) -> None:
    """Plot the closest defensible Figure 10 view available in this sweep.

    The sweep has no paired online defaultNV/GreenLLM runs and no request-level
    TTFT distribution.  We therefore compare the measured 1950-MHz point with
    each concurrency's measured energy-optimal point.  For the closed-loop load,
    C / requests-per-second is a mean response-time proxy, not P90 TTFT.
    """

    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.55), sharex=True)
    pool_specs = (
        ("short_medium", "Short — 65-token prompt", 0.4),
        ("long", "Long — 6,131-token prompt", 2.0),
    )
    x = np.arange(len(CONCURRENCIES), dtype=float)
    width = 0.33

    for panel, (ax, (pool, title, slo)) in enumerate(zip(axes, pool_specs)):
        profiles = models["power_models"][pool]["by_concurrency"]
        default_time = []
        optimized_time = []
        saving = []
        optimum_freqs = []
        for concurrency in CONCURRENCIES:
            profile = profiles[str(concurrency)]
            points = profile["points"]
            default = next(p for p in points if int(p["frequency_mhz"]) == 1950)
            optimum = profile["measured_energy_optimum"]
            default_time.append(concurrency / float(default["requests_per_second"]))
            optimized_time.append(concurrency / float(optimum["requests_per_second"]))
            saving.append(
                100.0
                * (1.0 - float(optimum["energy_j_per_request"]) / float(default["energy_j_per_request"]))
            )
            optimum_freqs.append(int(optimum["frequency_mhz"]))

        bars_default = ax.bar(
            x - width / 2,
            default_time,
            width,
            color=BLUE,
            edgecolor="#333333",
            linewidth=0.55,
            hatch="///",
            label="1950 MHz reference",
        )
        bars_opt = ax.bar(
            x + width / 2,
            optimized_time,
            width,
            color=ORANGE,
            edgecolor="#333333",
            linewidth=0.55,
            hatch="\\\\\\",
            label="Measured energy optimum",
        )
        ax.axhline(slo, color=CYAN, linewidth=2.0, label=f"{int(slo * 1000)} ms SLO")
        ax.set_ylabel("Mean latency proxy (s)")
        ax.set_title(f"({chr(97 + panel)}) {title}", loc="left", fontweight="bold")
        ax2 = ax.twinx()
        ax2.plot(x, saving, color=GREEN, marker="o", markersize=3.2, linewidth=1.25, label="Energy saving")
        ax2.set_ylabel("Energy saving (%)", color=GREEN)
        ax2.tick_params(axis="y", colors=GREEN)
        ax2.grid(False)
        ax2.set_ylim(bottom=0)
        for rect, freq in zip(bars_opt, optimum_freqs):
            ax.annotate(
                f"{freq}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )
        if panel == 0:
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(
                handles1 + handles2,
                labels1 + labels2,
                frameon=False,
                ncol=2,
                loc="upper left",
                fontsize=7,
            )

    axes[-1].set_xticks(x, [str(c) for c in CONCURRENCIES])
    axes[-1].set_xlabel("Closed-loop concurrency")
    fig.suptitle("Fig. 10 profiling counterpart (approximation)", y=0.995)
    fig.text(
        0.5,
        0.005,
        "Latency proxy = concurrency / measured throughput; bars are not P90 TTFT. "
        "No paired online defaultNV/GreenLLM data are implied.",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.98))
    save_figure(fig, output, "fig10_prefill_local_approx")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    configure_style()
    models, rows = load_data(args.source)
    plot_fig3a(models, args.output)
    plot_fig7(models, rows, args.output)
    plot_fig8(models, args.output)
    plot_fig10_approx(models, args.output)
    print(f"Wrote 8 figure files to {args.output}")


if __name__ == "__main__":
    main()
