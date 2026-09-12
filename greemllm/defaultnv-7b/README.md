# GreenLLM DefaultNV 7B baseline

This repository adapts the DefaultNV baseline from GreenLLM
(arXiv:2508.16449v1) to one local eight-GPU node and a 7B model. The GPU
model is detected at build and run time.

## Fixed substitutions

- GPU: the local 8-GPU node (currently RTX 3090 or RTX 4090) instead of
  8x A100-SXM4 40 GiB.
- TensorRT-LLM wheels and images are stored separately by compute capability
  (`sm86` for RTX 3090 and `sm89` for RTX 4090), so node drift does not replace
  the other build.
- Model: Qwen/Qwen2.5-Coder-7B-Instruct instead of Qwen3-14B/30B.

Everything else follows the disclosed paper configuration where possible:
Dynamo v0.3.1, its pinned TensorRT-LLM commit, BF16, two prefill workers with
TP=2, four single-GPU decode workers, round-robin routing, remote prefill,
NVIDIA default DVFS, and at least 30 minutes per trace workload.

The paper does not disclose the exact ServeGen model-size pool, time slice,
random seed, prompt construction, power-sampling interval, or full request
replayer. This adaptation fixes those choices in `config/experiment.yaml` and
records them in every result manifest.
