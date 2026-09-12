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

仓库保留代码、配置及一个小型 smoke trace；完整负载、实验结果、日志、
生成图表、模型缓存、构建缓存和第三方源码不随仓库上传。
运行前需要准备 Dynamo v0.3.1、ServeGen、TensorRT-LLM、模型与 profile 数据。
脚本期望 `vendor/` 和 `.build-cache/` 中存在相关依赖。
配置与部分脚本仍包含原实验机的 `/home/shenhaojia/energy/` 绝对路径，
在其他机器运行前需按实际位置修改。

本地迁移保留了旧目录的兼容符号链接，以及指向原依赖目录的符号链接；
这些环境链接不上传。原四个仓库的 Git 元数据保存在原工作区的
`.repository-backups/AgenticProfiling-import/` 中，新仓库以统一快照开始跟踪。
