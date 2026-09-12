#!/usr/bin/env python3
"""Render a complete local counterpart for GreenLLM Figures 1--12.

All quantitative panels use local measurements.  Conceptual figures are
redrawn with the actual 8x RTX 3090 topology and the behavior implemented by
the local controller.  Figure 12 requires --micro-root from
run_paper_figure_microbench.py; Figure 3c uses the same run when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "greenllm-7b/figures/paper_reproduction_latest"
PREFILL_PROFILE = ROOT / "greenllm-profile-7b/results/20260902T125826Z"
DECODE_PROFILE = ROOT / "greenllm-profile-7b/results/decode-20260902T133755Z"
RUN_ROOTS = {
    "DefaultNV": ROOT / "defaultnv-7b/results/20260908T032631Z",
    "PrefillSplit": ROOT / "prefillsplit-7b/results/20260908T053856Z",
    "GreenLLM": ROOT / "greenllm-7b/results/20260908T104658Z",
}
QPS_LEVELS = (3, 5, 8, 10)
BLUE = "#58A5E1"
ORANGE = "#F28E5B"
GREEN = "#009E73"
CYAN = "#00A6D6"
PURPLE = "#5E3C99"
GREY = "#777777"
INK = "#202020"
COLORS = {"DefaultNV": BLUE, "PrefillSplit": GREY, "GreenLLM": ORANGE}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 9.2, "legend.fontsize": 7.1, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.linewidth": .75, "axes.grid": True,
        "grid.color": "#D8D8D8", "grid.linestyle": "--", "grid.linewidth": .5,
        "grid.alpha": .7, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def pct(values, q):
    return float(np.percentile(values, q)) if values else math.nan


def integrate(rows: list[dict], ids: set[int]) -> float:
    t = np.asarray([r["timestamp_s"] for r in rows])
    w = np.asarray([sum(g["power_w"] for g in r["gpus"] if g["gpu"] in ids) for r in rows])
    return float(np.trapezoid(w, t) / 3600)


def valid(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("http_status") == 200 and r.get("error") is None]


def run_dir(method: str, qps: int) -> Path:
    return RUN_ROOTS[method] / f"alibaba_{qps}qps"


def request_bins(rows: list[dict], width: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(math.ceil(duration / width))
    tokens = np.zeros(n)
    for r in valid(rows):
        start = float(r["scheduled_offset_s"]) + float(r.get("scheduled_lateness_ms", 0)) / 1000
        ds = start + float(r["ttft_ms"]) / 1000
        de = start + float(r["latency_ms"]) / 1000
        if de <= ds:
            continue
        rate = float(r["output_tokens"]) / (de - ds)
        for i in range(max(0, int(ds // width)), min(n - 1, int(de // width)) + 1):
            overlap = max(0., min(de, (i + 1) * width) - max(ds, i * width))
            tokens[i] += rate * overlap
    return (np.arange(n) + .5) * width, tokens / width


def power_bins(rows: list[dict], ids: set[int], width: float, duration: float, field: str) -> tuple[np.ndarray, np.ndarray]:
    t0 = rows[0]["timestamp_s"]
    n = int(math.ceil(duration / width))
    sums, counts = np.zeros(n), np.zeros(n)
    for r in rows:
        i = int((r["timestamp_s"] - t0) // width)
        if 0 <= i < n:
            selected = [g for g in r["gpus"] if g["gpu"] in ids]
            value = np.mean([g[field] for g in selected]) if field != "power_w" else sum(g[field] for g in selected)
            sums[i] += value; counts[i] += 1
    values = np.divide(sums, counts, out=np.full(n, np.nan), where=counts > 0)
    return (np.arange(n) + .5) * width, values


def box(ax, xy, wh, text, color="#F3F3F3", fontsize=8):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.025",
                           fc=color, ec=INK, lw=.8)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, a, b, text=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1, color=INK))
    if text:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.025, text, ha="center", fontsize=7)


def fig1(out: Path, dynamic: Path | None):
    if dynamic and (dynamic / "defaultnv").exists():
        duration, width = 180., 2.
        br = jsonl(dynamic / "defaultnv/requests.jsonl")
        gr = jsonl(dynamic / "greenllm/requests.jsonl")
        bp = jsonl(dynamic / "defaultnv/power.jsonl")
        gp = jsonl(dynamic / "greenllm/power.jsonl")
        workload = "Sinusoidal 1–10–1 QPS"
    else:
        duration, width = 1800., 10.
        br = jsonl(run_dir("DefaultNV", 10) / "requests.jsonl")
        gr = jsonl(run_dir("GreenLLM", 10) / "requests.jsonl")
        bp = jsonl(run_dir("DefaultNV", 10) / "power.jsonl")
        gp = jsonl(run_dir("GreenLLM", 10) / "power.jsonl")
        workload = "Alibaba 10 QPS"
    xb, tb = request_bins(br, width, duration); xg, tg = request_bins(gr, width, duration)
    xcb, cb = power_bins(bp, set(range(4)), width, duration, "sm_clock_mhz")
    xcg, cg = power_bins(gp, set(range(4)), width, duration, "sm_clock_mhz")
    fig, ax = plt.subplots(2, 1, figsize=(7.1, 4.0), sharex=True)
    for panel, (a, xt, tokens, xc, clocks, name) in enumerate((
        (ax[0], xb, tb, xcb, cb, "DefaultNV"),
        (ax[1], xg, tg, xcg, cg, "GreenLLM"),
    )):
        token_line = a.plot(xt/60, tokens, color="#C85A32", lw=1, label="Decoded TPS")
        twin = a.twinx()
        clock_line = twin.plot(xc/60, clocks, color=BLUE, lw=1, label=f"{name} clock")
        a.set_ylabel("Decoded TPS"); twin.set_ylabel("Mean SM clock (MHz)", color=BLUE)
        twin.grid(False); twin.set_ylim(500, 2000)
        a.set_title(f"({chr(97+panel)}) {name} — {workload}", loc="left")
        a.legend(token_line + clock_line, [x.get_label() for x in token_line + clock_line],
                 frameon=False, ncol=2, loc="upper right")
        a.set_xlim(0, duration / 60)
    ax[1].set_xlabel("Elapsed time (min)")
    fig.suptitle("Figure 1 reproduction — Decode workload and SM-frequency tracking")
    fig.tight_layout(); save(fig, out, "fig01_decode_frequency_tracking")


def fig2(out: Path):
    fig, ax = plt.subplots(figsize=(7.1, 2.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, (.02,.35),(.14,.3),"Request\nprompt", "#FFFFFF")
    box(ax, (.22,.28),(.25,.44),"Prefill\nprocess all input tokens\nproduce first token + KV", "#FCE2D5")
    box(ax, (.57,.28),(.25,.44),"Decode\none token per iteration\nreuse KV cache", "#DDEFFC")
    box(ax, (.88,.35),(.10,.3),"EOS /\nresponse", "#FFFFFF")
    arrow(ax,(.16,.5),(.22,.5)); arrow(ax,(.47,.5),(.57,.5)); arrow(ax,(.82,.5),(.88,.5))
    ax.text(.52,.55,"KV handoff",ha="center",fontsize=7)
    ax.text(.345,.14,"2 workers × TP2\n(GPUs 4–7)",ha="center",fontsize=7.5)
    ax.text(.695,.14,"4 workers × TP1\n(GPUs 0–3)",ha="center",fontsize=7.5)
    ax.set_title("Figure 2 reproduction — Local disaggregated inference path")
    save(fig,out,"fig02_prefill_decode_serving")


def profile_data():
    pm=json.loads((PREFILL_PROFILE/"prefill_models.json").read_text())
    dm=json.loads((DECODE_PROFILE/"decode_models.json").read_text())
    return pm,dm


def fig3(out: Path, micro: Path | None):
    pm,dm=profile_data(); fig,axes=plt.subplots(1,3,figsize=(10.1,3.05))
    for c,color in zip((1,4,8,16),plt.cm.viridis(np.linspace(.1,.9,4))):
        pts=pm["power_models"]["short_medium"]["by_concurrency"][str(c)]["points"]
        f=np.array([p["frequency_mhz"] for p in pts]); e=np.array([p["energy_j_per_request"] for p in pts]); e/=e.min()
        axes[0].plot(f,e,"o-",ms=2.5,lw=1,color=color,label=f"C={c}")
    axes[0].set(title="(a) Prefill",xlabel="SM frequency (MHz)",ylabel=r"Normalized energy ($E/E_{min}$)")
    axes[0].legend(frameon=False,ncol=2)
    curves=defaultdict(lambda:defaultdict(list))
    for length,model in dm["power_models_by_input_tokens"].items():
        for c,prof in model["by_concurrency"].items():
            vals=np.array([p["energy_j_per_decode_token"] for p in prof["points"]]); mn=vals.min()
            for p in prof["points"]: curves[int(c)][p["frequency_mhz"]].append(p["energy_j_per_decode_token"]/mn)
    for c,color in zip(sorted(curves),plt.cm.viridis(np.linspace(.1,.9,len(curves)))):
        f=sorted(curves[c]); axes[1].plot(f,[np.mean(curves[c][x]) for x in f],"o-",ms=2.5,lw=1,color=color,label=f"C={c}")
    axes[1].set(title="(b) Decode",xlabel="SM frequency (MHz)",ylabel=r"Normalized energy ($E/E_{min}$)")
    axes[1].legend(frameon=False,ncol=2)
    if micro and (micro/"fixed_clock").exists():
        fs=[]; es=[]; feasible=[]
        for d in sorted((micro/"fixed_clock").glob("*mhz"),key=lambda x:int(x.name[:-3])):
            s=json.loads((d/"summary.json").read_text()); fs.append(int(d.name[:-3])); es.append(s["energy_j"])
            feasible.append(s["overall_ttft_pass_pct"] >= 95 and s["tbt_pass_pct"] >= 95)
        es=np.asarray(es)/min(es); axes[2].plot(fs,es,"-",color=PURPLE,lw=1.4)
        for f,e,ok in zip(fs,es,feasible):
            axes[2].scatter(f,e,s=25,marker="o" if ok else "x",color=GREEN if ok else "#D62728",zorder=3)
        best=int(np.argmin(es)); axes[2].annotate("Measured minimum",(fs[best],es[best]),xytext=(fs[best],es[best]+.15),ha="center",arrowprops={"arrowstyle":"->"})
        axes[2].text(.98,.04,"● ≥95% TTFT & TBT pass\n× infeasible",transform=axes[2].transAxes,ha="right",va="bottom",fontsize=6.8)
    else:
        axes[2].text(.5,.5,"Fixed-clock microbenchmark pending",ha="center",va="center",transform=axes[2].transAxes)
    axes[2].set(title="(c) 10 QPS trace, 60 s slice",xlabel="All-GPU SM frequency (MHz)",ylabel="Normalized total energy")
    fig.suptitle("Figure 3 reproduction — Local RTX 3090 energy/frequency profiles")
    fig.tight_layout(); save(fig,out,"fig03_energy_frequency_profiles")


def fig4(out: Path):
    fig,ax=plt.subplots(figsize=(7.2,3.0));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    box(ax,(.02,.38),(.13,.24),"HTTP\nrequests","#FFFFFF")
    box(ax,(.20,.34),(.14,.32),"Length router\n≤1024 / >1024","#E9E9E9")
    box(ax,(.41,.62),(.18,.23),"Short Prefill\nGPUs 4–5","#FCE2D5")
    box(ax,(.41,.15),(.18,.23),"Long Prefill\nGPUs 6–7","#F6C6AF")
    box(ax,(.68,.34),(.16,.32),"Decode pool\nGPUs 0–3","#DDEFFC")
    box(ax,(.88,.34),(.10,.32),"Stream\noutput","#FFFFFF")
    arrow(ax,(.15,.5),(.20,.5));arrow(ax,(.34,.5),(.41,.735));arrow(ax,(.34,.5),(.41,.265))
    arrow(ax,(.59,.735),(.68,.55),"KV");arrow(ax,(.59,.265),(.68,.45),"KV");arrow(ax,(.84,.5),(.88,.5))
    ax.text(.50,.91,"Prefill profile + TTFT SLO → per-pool clock",ha="center",color=ORANGE)
    ax.text(.76,.065,"Decode profile + TPS/TBT feedback → shared decode clock",ha="center",color=BLUE,fontsize=7.5)
    ax.set_title("Figure 4 reproduction — Actual local GreenLLM topology")
    save(fig,out,"fig04_system_overview")


def fig5(out: Path):
    a=valid(jsonl(run_dir("DefaultNV",8)/"requests.jsonl")); b=valid(jsonl(run_dir("PrefillSplit",8)/"requests.jsonl"))
    groups=[("Short/medium",lambda r:r["input_tokens"]<=1024,400),("Long",lambda r:r["input_tokens"]>1024,2000)]
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
    for ax,(title,sel,slo) in zip(axes,groups):
        av=[r["ttft_ms"] for r in a if sel(r)];bv=[r["ttft_ms"] for r in b if sel(r)]
        hi=max(slo*1.1,pct(av+bv,99.5));bins=np.linspace(0,hi,45)
        ax.hist(av,bins=bins,density=True,alpha=.45,color=BLUE,label="Before routing")
        ax.hist(bv,bins=bins,density=True,alpha=.45,color=ORANGE,label="After routing")
        ax.axvline(slo,color=CYAN,lw=1.5,label="SLO");ax.set(title=title,xlabel="TTFT (ms)",ylabel="Density")
    axes[0].legend(frameon=False);fig.suptitle("Figure 5 reproduction — Alibaba 8 QPS TTFT distribution")
    fig.tight_layout();save(fig,out,"fig05_ttft_before_after_routing")


def fig6(out: Path):
    fig,ax=plt.subplots(figsize=(7.1,2.45));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    box(ax,(.03,.34),(.16,.32),"Per-class queue\nactive lengths $L_k$","#FFFFFF")
    box(ax,(.26,.34),(.18,.32),"Measured latency\ntable $t(L,f)$","#FCE2D5")
    box(ax,(.51,.34),(.18,.32),"Energy optimum\n+ SLO feasibility","#E5F5E9")
    box(ax,(.76,.34),(.20,.32),"Apply max safe clock\nto worker GPU pair","#DDEFFC")
    arrow(ax,(.19,.5),(.26,.5));arrow(ax,(.44,.5),(.51,.5));arrow(ax,(.69,.5),(.76,.5))
    ax.text(.60,.14,"Local implementation uses measured tables; idle → unlock clocks",ha="center",fontsize=8)
    ax.set_title("Figure 6 reproduction — Implemented Prefill queue and control")
    save(fig,out,"fig06_prefill_control")


def fig7(out: Path):
    pm,_=profile_data(); lat=pm["latency_model"]; ref=lat["reference_frequency_mhz"]
    rows=list(csv.DictReader((PREFILL_PROFILE/"measurements.csv").open()))
    rows=[r for r in rows if r["phase"]=="prefill_latency" and int(r["frequency_mhz"])==ref]
    x=np.array([float(r["actual_input_tokens"]) for r in rows]);y=np.array([float(r["ttft_ms"])/1000 for r in rows])
    co=lat["coefficients"];xx=np.linspace(0,x.max()*1.02,400);yy=(co["a"]*xx**2+co["b"]*xx+co["c"])/1000
    fig,ax=plt.subplots(figsize=(4.6,3.1));ax.scatter(x,y,s=17,facecolors="none",edgecolors=BLUE,label="Measurements")
    ax.plot(xx,yy,color=PURPLE,lw=1.6,label=r"Quadratic fit $aL^2+bL+c$")
    ax.set(xlabel="Prompt length (tokens)",ylabel="TTFT (s)",title=f"Figure 7 reproduction — Prefill latency at {ref} MHz")
    ax.text(.97,.06,rf"$R^2={lat['r_squared']:.5f}$",transform=ax.transAxes,ha="right");ax.legend(frameon=False)
    fig.tight_layout();save(fig,out,"fig07_prefill_latency_model")


def fig8(out: Path):
    pm,_=profile_data();fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
    for ax,(pool,title) in zip(axes,[("short_medium","Short/medium pool"),("long","Long pool")]):
        for c,color in zip((1,4,8,16),plt.cm.viridis(np.linspace(.1,.9,4))):
            prof=pm["power_models"][pool]["by_concurrency"][str(c)];pts=prof["points"]
            x=np.array([p["frequency_mhz"] for p in pts]);y=np.array([p["average_power_w"] for p in pts]);xx=np.linspace(x.min(),x.max(),200)
            ax.scatter(x,y,s=10,facecolors="none",edgecolors=color);ax.plot(xx,np.polyval(prof["coefficients_high_to_low"],xx),color=color,lw=1,label=f"C={c}")
        ax.set(title=title,xlabel="SM frequency (MHz)",ylabel="2-GPU power (W)");ax.legend(frameon=False,ncol=2)
    fig.suptitle("Figure 8 reproduction — Measured Prefill power with cubic fits");fig.tight_layout();save(fig,out,"fig08_prefill_power_model")


def fig9(out: Path):
    fig,ax=plt.subplots(figsize=(7.1,2.6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    box(ax,(.02,.34),(.14,.32),"Decode\ntoken events","#FFFFFF")
    box(ax,(.23,.57),(.18,.24),"200 ms TPS\nwindow","#DDEFFC")
    box(ax,(.23,.18),(.18,.24),"200 ms P95 TBT\nwindow","#DDEFFC")
    box(ax,(.49,.57),(.19,.24),"Profile-selected\ncoarse target","#E5F5E9")
    box(ax,(.49,.18),(.19,.24),"Feedback: one\nfrequency-grid step","#E5F5E9")
    box(ax,(.77,.34),(.20,.32),"Apply shared clock\nto GPUs 0–3","#FCE2D5")
    arrow(ax,(.16,.5),(.23,.69));arrow(ax,(.16,.5),(.23,.30));arrow(ax,(.41,.69),(.49,.69));arrow(ax,(.41,.30),(.49,.30));arrow(ax,(.68,.69),(.77,.56));arrow(ax,(.68,.30),(.77,.44))
    ax.text(.50,.04,"Control tick: 20 ms; local grid: 600–1950 MHz (not the paper's 15 MHz grid)",ha="center",fontsize=7.5)
    ax.set_title("Figure 9 reproduction — Implemented Decode feedback controller")
    save(fig,out,"fig09_decode_controller")


def energy_by_pool(method,qps,ids):
    return integrate(jsonl(run_dir(method,qps)/"power.jsonl"),set(ids))


def fig10(out: Path):
    classes=[("Short",lambda r:r["input_tokens"]<=256,400),("Medium",lambda r:256<r["input_tokens"]<=1024,400),("Long",lambda r:r["input_tokens"]>1024,2000)]
    savings=[]
    for q in QPS_LEVELS:
        b=energy_by_pool("DefaultNV",q,range(4,8));g=energy_by_pool("GreenLLM",q,range(4,8));savings.append(100*(1-g/b))
    fig,axes=plt.subplots(3,1,figsize=(5.1,6.7),sharex=True)
    x=np.arange(len(QPS_LEVELS));w=.34
    for ax,(title,select,slo) in zip(axes,classes):
        for j,m in enumerate(("DefaultNV","GreenLLM")):
            vals=[]
            for q in QPS_LEVELS:
                rs=[r["ttft_ms"] for r in valid(jsonl(run_dir(m,q)/"requests.jsonl")) if select(r)]
                vals.append(pct(rs,90))
            ax.bar(x+(j-.5)*w,vals,w,color=COLORS[m],edgecolor=INK,lw=.4,hatch="///" if j==0 else "\\\\",label=m)
        ax.axhline(slo,color=CYAN,lw=1.4,label="SLO");ax.set_ylabel("P90 TTFT (ms)");ax.set_title(title,loc="left")
        twin=ax.twinx();twin.plot(x,savings,"o-",color=GREEN,lw=1,ms=3,label="Prefill energy saving");twin.set_ylabel("Saving (%)",color=GREEN);twin.grid(False)
    axes[-1].set_xticks(x,QPS_LEVELS);axes[-1].set_xlabel("Alibaba trace request rate (QPS)");axes[0].legend(frameon=False,ncol=3)
    fig.suptitle("Figure 10 reproduction — Actual trace Prefill results");fig.tight_layout();save(fig,out,"fig10_prefill_ttft_vs_load")


def fig11(out: Path):
    x=np.arange(len(QPS_LEVELS));w=.34;base=[];green=[];tps=[];saving=[]
    for q in QPS_LEVELS:
        br=valid(jsonl(run_dir("DefaultNV",q)/"requests.jsonl"));gr=valid(jsonl(run_dir("GreenLLM",q)/"requests.jsonl"))
        base.append(pct([r["tbt_p95_ms"] for r in br if r["output_tokens"]>1],90));green.append(pct([r["tbt_p95_ms"] for r in gr if r["output_tokens"]>1],90))
        tps.append(sum(r["output_tokens"] for r in br)/json.loads((run_dir("DefaultNV",q)/"summary.json").read_text())["duration_s"])
        be=energy_by_pool("DefaultNV",q,range(4));ge=energy_by_pool("GreenLLM",q,range(4));saving.append(100*(1-ge/be))
    fig,ax=plt.subplots(figsize=(5.5,3.25));ax.bar(x-w/2,base,w,color=BLUE,hatch="///",edgecolor=INK,lw=.4,label="DefaultNV")
    ax.bar(x+w/2,green,w,color=ORANGE,hatch="\\\\",edgecolor=INK,lw=.4,label="GreenLLM");ax.axhline(100,color=CYAN,lw=1.4,label="100 ms SLO")
    ax.set(xticks=x,xticklabels=[f"{v:.0f}\n({q} QPS)" for v,q in zip(tps,QPS_LEVELS)],xlabel="Measured output tokens/s",ylabel="P90 of request P95 TBT (ms)")
    twin=ax.twinx();twin.plot(x,saving,"o-",color=GREEN,lw=1.2,label="Decode energy saving");twin.set_ylabel("Energy saving (%)",color=GREEN);twin.grid(False)
    h,l=ax.get_legend_handles_labels();h2,l2=twin.get_legend_handles_labels();ax.legend(h+h2,l+l2,frameon=False,ncol=2)
    ax.set_title("Figure 11 reproduction — Actual trace Decode results");fig.tight_layout();save(fig,out,"fig11_decode_tbt_vs_tps")


def fig12(out: Path, micro: Path | None):
    fig,axes=plt.subplots(2,1,figsize=(5.1,5.4))
    specs=[("prefill_margin","Prefill margin",set(range(4,8)),"ttft_ms","P90 TTFT (ms)"),("decode_margin","Decode margin",set(range(4)),"tbt_p95_ms","P90 TBT (ms)")]
    if not micro:
        for ax in axes: ax.text(.5,.5,"Margin microbenchmark pending",transform=ax.transAxes,ha="center",va="center")
    else:
        for ax,(folder,label,ids,metric,ylabel) in zip(axes,specs):
            points=[]
            for d in (micro/folder).iterdir():
                if not d.is_dir(): continue
                margin=float(d.name.replace("p",".")); run=d/"run"; rows=valid(jsonl(run/"requests.jsonl")); power=jsonl(run/"power.jsonl")
                vals=[r[metric] for r in rows if not (metric.startswith("tbt") and r["output_tokens"]<=1)]
                points.append((margin,integrate(power,ids),pct(vals,90)))
            points.sort(); margins=[x[0] for x in points];energy=np.array([x[1] for x in points]);lat=[x[2] for x in points];energy/=energy.min()
            ax.bar(range(len(points)),energy,color=ORANGE if folder.startswith("prefill") else BLUE,hatch="//",edgecolor=INK,lw=.4,label="Normalized energy")
            ax.set_ylabel("Normalized energy");ax.set_xticks(range(len(points)),margins);ax.set_xlabel(label)
            twin=ax.twinx();twin.plot(range(len(points)),lat,"ko-",lw=1,ms=3,label=ylabel);twin.set_ylabel(ylabel);twin.grid(False)
            ax.set_title(f"({'a' if folder.startswith('prefill') else 'b'}) {label} sensitivity",loc="left")
    fig.suptitle("Figure 12 reproduction — Measured energy–latency trade-off");fig.tight_layout();save(fig,out,"fig12_margin_sensitivity")


def write_manifest(out: Path, micro: Path | None, dynamic: Path | None):
    manifest={"scope":"Local reproduction of paper Figures 1–12","hardware":"8× NVIDIA GeForce RTX 3090","model":"Qwen/Qwen2.5-Coder-7B-Instruct","full_trace_runs":{k:str(v) for k,v in RUN_ROOTS.items()},"prefill_profile":str(PREFILL_PROFILE),"decode_profile":str(DECODE_PROFILE),"micro_root":str(micro) if micro else None,"dynamic_root":str(dynamic) if dynamic else None,"notes":{"fig1":"Measured sinusoidal 1–10–1 QPS microbenchmark when dynamic_root is set.","fig3c":"60 s Alibaba 10 QPS slice, output capped at 64 tokens.","fig10":"Actual trace request QPS rather than homogeneous synthetic prompt TPS.","fig12":"60 s trace slice; local controller margin semantics."}}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (out/"README.md").write_text("""# GreenLLM local paper-figure reproduction

These are measured local counterparts, not digitizations of paper data.

| Figure | Local source |
|---|---|
| 1 | Measured 180 s sinusoidal 1–10–1 QPS DefaultNV/GreenLLM runs |
| 2 | Implemented disaggregated Prefill/Decode execution path |
| 3 | Completed Prefill/Decode profiles plus fixed-clock trace microbenchmark |
| 4 | Actual 8-GPU topology, router, pools, and controllers |
| 5 | Latest Alibaba 8 QPS DefaultNV vs PrefillSplit request records |
| 6 | Implemented table-driven Prefill controller |
| 7 | Measured 1950 MHz Prefill latency sweep and quadratic fit |
| 8 | Measured Prefill power sweep and cubic fits |
| 9 | Implemented 20 ms Decode TPS/TBT feedback controller |
| 10 | Latest full-length Alibaba 3/5/8/10 QPS request and power records |
| 11 | Latest full-length Alibaba 3/5/8/10 QPS Decode results |
| 12 | Measured 60 s Prefill/Decode margin sweep |

Key deviations from the paper are intentional and labelled: RTX 3090 and
Qwen2.5-Coder-7B replace A100 and Qwen3-14B; Figure 1 uses 65 input/64 output
tokens and a 1–10 QPS sinusoid; Figure 3c and Figure 12 use a 60 s Alibaba
10-QPS slice capped at 64 output tokens; Figure 10 uses real trace QPS rather
than homogeneous synthetic prompt TPS.  Figure 9 documents the local discrete
frequency grid, which is coarser than the paper's 15 MHz steps.  See
`manifest.json` and the original result directories for provenance.
""")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=OUT_DEFAULT);parser.add_argument("--micro-root",type=Path);parser.add_argument("--dynamic-root",type=Path)
    args=parser.parse_args();out=args.output.resolve();micro=args.micro_root.resolve() if args.micro_root else None;dynamic=args.dynamic_root.resolve() if args.dynamic_root else None;style()
    fig1(out,dynamic);fig2(out);fig3(out,micro);fig4(out);fig5(out);fig6(out);fig7(out);fig8(out);fig9(out);fig10(out);fig11(out);fig12(out,micro);write_manifest(out,micro,dynamic)
    print(out)


if __name__=="__main__": main()
