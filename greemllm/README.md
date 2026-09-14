# GreenLLM 7B 适配复现

本目录收录 GreenLLM 的本地 7B 适配复现，是 AgenticProfiling 仓库的 `greemllm/` 模块。以下路径均相对于本目录。

## 代码结构

| 目录 | 用途 |
| --- | --- |
| `defaultnv-7b/` | DefaultNV 基线、服务部署、负载生成与回放、指标采集 |
| `prefillsplit-7b/` | 按输入长度分流的 PrefillSplit 消融实验 |
| `greenllm-profile-7b/` | Prefill/Decode 离线测量、模型生成与分析绘图 |
| `greenllm-7b/` | 基于 profiling 的动态调频控制器与端到端对比实验 |

核心入口为 `prefillsplit-7b/python/prefillsplit/worker.py`、
`greenllm-profile-7b/scripts/offline_profile.py`、
`greenllm-profile-7b/scripts/decode_profile.py` 和
`greenllm-7b/scripts/greenllm_replay.py`。各子目录 README 提供实验说明。

## 环境与数据

这是针对本地八卡节点和 Qwen2.5-Coder-7B-Instruct 的适配实现，
并非论文原硬件、模型及全部算法细节的原样复刻。
控制器使用实测查找表替代论文的 Prefill 反频率近似。


运行前需要准备 Dynamo v0.3.1、ServeGen、TensorRT-LLM、模型与 profile 数据。

