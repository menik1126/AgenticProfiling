# GreenLLM 7B end-to-end adaptation

This directory retains the `PrefillSplit` service and adds profile-driven
Prefill and Decode DVFS control for the first Alibaba 1-QPS trace without
changing or deleting the DefaultNV and PrefillSplit experiment results.

The local controller deliberately uses measured lookup tables instead of the
paper's inaccurate inverse-frequency Prefill approximation. It records every
frequency transition and 20-ms control sample and resets all GPU clocks on
normal completion, failure, signal, or parent-process loss.

Run the end-to-end experiment with:

```bash
scripts/run_first_trace.sh
```

The underlying routing service is the already validated `PrefillSplit`
ablation; the DVFS controller runs alongside the trace replayer and leaves the
working `defaultnv-7b` baseline unchanged.

The serving topology is unchanged: four single-GPU decode workers and two
TP=2 prefill workers:

- requests with at most 1024 input tokens always use one prefill worker;
- requests with more than 1024 input tokens always use the other worker;
- GPU 4-5 and 6-7 receive independent Prefill frequency decisions;
- GPU 0-3 receive a shared Decode frequency selected from the local profile
  and corrected by live TBT feedback.

The two discovered prefill instance IDs are sorted and pinned to the two
roles when each decode worker starts. The process fails instead of silently
changing roles if the prefill membership changes during a run.

The existing `greenllm-dynamo:0.3.1-trtllm-sm86` or `sm89` image is reused.
Python components are loaded from `python/prefillsplit` through `PYTHONPATH`,
so no image rebuild is required.

Do not launch this service while `greenllm-defaultnv-7b` is running. The
launch script checks this condition and exits without stopping the baseline.
