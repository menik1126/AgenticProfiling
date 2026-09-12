# GreenLLM offline profiling for the local 7B service

This profiles the running `greenllm-prefillsplit-7b` service on the local
8-GPU node. The Prefill-only workflow follows the GreenLLM paper:

1. measure idle power with the normal NVIDIA governor;
2. at a reference SM clock, fit prefill latency versus prompt length;
3. sweep SM clocks and fit active prefill power for the short and long pools.

GPU clocks are changed through `nvidia-smi` inside the privileged serving
container. Both Python `finally` cleanup and a parent-process watchdog reset
GPUs 4-7 to the normal unlocked governor on success, ordinary failure,
Ctrl-C, TERM/HUP, and even an externally killed runner. The result directory
contains raw JSONL/CSV measurements, `prefill_models.json`, and the post-reset
clock query. Decode profiling is deliberately excluded from this run.

Clock cleanup first uses the serving container and falls back to a fresh,
privileged, short-lived container if the service itself has exited.
Every warm-up and measured request has a run-unique prompt nonce while keeping
the configured token length, preventing prefix/KV reuse from biasing latency.
The power sweep uses a fixed-concurrency, closed-loop load so slow frequencies
remain saturated without accumulating an unbounded open-loop request queue.
The post-reset clock evidence is sampled after a five-second governor settling
period and includes the P-state.

`config/alibaba_1qps_full_prefill.yaml` is the complete profile for the current
first trace. It measures a trace-derived length-by-frequency latency matrix,
validates the paper's inverse-frequency latency approximation, records idle
power per TP2 pool, and measures 13 frequencies at concurrency 1/4/8/16 for
both prompt-length pools. Run it with:

```bash
scripts/run_profile.sh config/alibaba_1qps_full_prefill.yaml
```

`config/alibaba_1qps_full_decode.yaml` profiles the four decode GPUs separately.
It sweeps trace-derived context lengths, concurrency, and SM frequency; records
TTFT/TBT, tokens/s, total and idle-subtracted J/token; and chooses both raw and
100-ms-TPOT-SLO-feasible energy optima. Run it with:

```bash
scripts/run_decode_profile.sh config/alibaba_1qps_full_decode.yaml
```

An interrupted run can be resumed without repeating completed grid points by
passing its existing result directory as the second argument.
