# GreenLLM PrefillSplit 7B adaptation

This directory adds the `PrefillSplit` ablation from GreenLLM without
changing the working `defaultnv-7b` baseline.

The serving topology is unchanged: four single-GPU decode workers and two
TP=2 prefill workers. The only behavioral change is prefill routing:

- requests with at most 1024 input tokens always use one prefill worker;
- requests with more than 1024 input tokens always use the other worker;
- all GPUs continue to use NVIDIA's default DVFS policy.

The two discovered prefill instance IDs are sorted and pinned to the two
roles when each decode worker starts. The process fails instead of silently
changing roles if the prefill membership changes during a run.

The existing `greenllm-dynamo:0.3.1-trtllm-sm86` or `sm89` image is reused.
Python components are loaded from `python/prefillsplit` through `PYTHONPATH`,
so no image rebuild is required.

Do not launch this service while `greenllm-defaultnv-7b` is running. The
launch script checks this condition and exits without stopping the baseline.

