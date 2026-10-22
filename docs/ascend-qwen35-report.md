# Qwen3.5-0.8B 单 Ascend 910C chip 阶段报告

成员贡献：**待小组填写姓名与各自贡献**。本分支交付环境隔离与核验、NPU 算子适配、模型集成、正确性与性能实验及复现文档；不将自动化完成的工作归到未经确认的个人名下。

## 1. 结果与范围

完整 **24 层文本模型**已在单 chip 上完成 Prefill、连续 Decode、多请求槽位复用和正常 EOS 停止。验收采用 FP16 权重、激活、KV 与卷积历史，FP32 GDN 递推状态和累加。没有删层、近似递推或 CPU 主计算回退。视觉输入、MTP、投机解码、ACL Graph、前缀复用、KV offload 和缓存续接 Prefill 不在本次验收范围。

所有改动保持 TokenSpeed 的统一模型、执行和缓存路径：runtime 只经 `tokenspeed-kernel` 分派 NPU 实现。基线为 `b8137dd1e8b9b054e00d5bc9519e6f3dd0ca492a`，改动在 `codex/ascend-qwen35-08b`。安装、启动和测试的实测命令见 [setup.md](setup.md)，逐模块形状、来源及状态见 [算子矩阵](ascend-qwen35-operator-matrix.md)。

## 2. 环境、模型与单 chip 证据

| 项目 | 实测配置 |
| --- | --- |
| 设备 | Ascend 910C，设备名 Ascend910_9362；物理 7 = board 3 / chip 1，PCIe `0000:93:00.0` |
| 可见与并行 | `ASCEND_RT_VISIBLE_DEVICES=7` → 逻辑 `npu:0`；world size、batch、最大并发均为 1 |
| HBM | 物理 64 GiB；驱动报告可用容量约 61.28 GiB |
| 系统 | AArch64 Linux 5.10，Python 3.11.16 |
| 驱动 / 固件 / CANN | 25.5.2 / 查询为 NA（未知）/ 9.0.0，`V100R001C10SPC001B250` |
| 框架 | PyTorch 2.9.0+cpu + torch-npu 2.9.0，Triton-Ascend 3.2.1 |
| 依赖 | Transformers 5.12.0、NumPy 2.4.6；完整快照见 `results/ascend-environment.json` |

先验证 `/workspace/nanoGPT` 的原有 10.65M 参数 checkpoint，成功生成 16 Token，再建立 `.venv --system-site-packages` 覆盖层；没有修改原 nanoGPT 环境。原 Triton-Ascend 3.2.0 与 CANN 9 的驱动编译不兼容，覆盖层的 3.2.1 经真实 kernel 编译和执行验证。scheduler 使用本 checkout 的 editable 源码构建，因为发布 wheel 缺少 `StorageConfig`。

权重来自官方 ModelScope Qwen 仓库，逐文件 revision、大小和 SHA256 固定在 `results/qwen35-model-manifest.json`；获取脚本会校验摘要。权重在 `/tmp/hw1-qwen35-model`，不进入 Git。单个 safetensors 为 1,746,942,600 字节，SHA256 为 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`。

配置是 18 层 GDN + 6 层全注意力，隐藏维 1024、MLP 3584、词表 248320。全注意力 Q/KV 为 8/2 头、head dim 256，RoPE 64 维；GDN 为 16 头、K/V 维 128、6144 卷积通道、卷积宽 4。

加载审计检查了全部文本参数：752,393,024 个独立参数、230 个加载名称、231 个参数名称（embedding/LM head 共享），missing 和 unexpected 均为空；所有参数位于 `npu:0`。checkpoint 的 153 个视觉张量和 15 个 MTP 张量按 `language_model_only` 范围跳过。实验逐步检查输入、位置与 logits 都在 NPU；仅正确性模式在完整 forward 结束后复制 logits 到 CPU。

## 3. 复用、修改与关键障碍

| 类别 | 内容 |
| --- | --- |
| 直接复用 | Embedding、非量化 GEMM/MLP、部分 RoPE、KV 写入、因果卷积、贪心采样、LCM 缓存分配与生命周期、paged attention decode |
| 新增 | Ascend GDN 精确递推 kernel 及公共注册；FP32 中间计算的 Gemma residual-add RMSNorm kernel |
| 修改 | Gated RMSNorm 使用设备无关 context；dense Qwen 路径避免无用 CUDA stream；head dim 256 的 Prefill 采用 vendor FIA 的 BSH 布局；显式 FP16 KV 选择与对应卷积历史 dtype |

最大的计算适配缺口是 GDN：原公共注册没有 NPU 实现。新增 kernel 保留完整的 decay、prediction、delta correction 和 query read，采用 FP32 状态、按 head 的 16 个 value 行分块、时间维精确串行递推。其目的首先是建立可信基线，未宣称达到并行 chunk 算法的性能。

另一个阻塞来自 CANN 9 的 TND Prefill 接口不接受 head dim 256。适配把 packed 请求逐段视为 BSH，并提供显式 causal mask，仍调用原来的 `npu_fused_infer_attention_score`，没有把 attention 转到 CPU。已有 BSH paged decode 支持该维度。初始 Gemma norm 的 CUDA context 和融合入口缺口也分别修正。

关键调用链为 `Qwen3_5GatedDeltaNet.forward` → 投影 → `PagedAttention` → `GdnAttnBackend/MambaAttnBackend` → 公共卷积与 GDN 分派 → Ascend kernel → 现有状态提交。LCM 统一拥有源 checkpoint 页、live 目标页和释放过程。公共递推状态始终为 `[page,head,value,key]`，没有错误沿用 FLA 的 K/V 转置布局。卷积历史为 `[page,6144,3]`，保存最早到最新的三个输入；这与通用 Inkling ring 的 `[slot,R,D]` 是不同契约。Prefill 不修改源状态，Decode 只写指定 live 页，负页索引不写状态。模型层间不插入主机同步；实验在完整 forward 边界同步计时。

## 4. 正确性

### 分层测试

目标 NPU 上的关键算子套件 **126 passed、1 deselected**：Gemma norm 12、Gated RMSNorm 12、GDN 30、卷积 38、MHA 14、既有 layernorm 4、RoPE 16。配置和缓存回归 **195 passed、2 skipped、74 subtests passed**。合计 321 个通过测试及 74 个子测试；两个 CUDA 专用分配测试跳过，独立 NPUGraph 测试未启用，不计为通过。

CPU FP64 算子参考覆盖真实 head 几何、长度 1、卷积窗口边界、63/64/65、129/2048、ragged/空请求、非整齐及分组 head、非零初始状态、分段等价、多步 Decode、源/目标页重映射与未选中页不变。卷积历史和源页要求精确一致。GDN FP32 state 使用 `atol=3e-6, rtol=3e-5`；FP16 输出 `1e-4/5e-4`，BF16 输出 `1e-3/4e-3`，分别容纳最终 dtype 舍入并对长期状态使用更严阈值。测试文件中的其他逐算子阈值按 FP64 参考与目标精度单独指定。

### 完整模型

参考是未经修改的 Transformers 5.12.0 `Qwen3_5ForCausalLM`，CPU FP32、eager attention、严格加载同一 checkpoint 的全部文本权重和共享 head。使用 NPU 实际生成 Token 作为逐步上下文，每步比较**全部 248320 个 logits**，避免自由生成分叉掩盖误差。CPU 正常构造也确保非持久 RoPE buffer 被初始化。

预先固定的整模型门槛为 RMS ≤0.10 logit、最大绝对误差 ≤0.75、cosine ≥0.999。算子门槛更严；整模型绝对门槛允许 24 层低精度舍入累积，同时检查分布、相似度、最大值和 top-1 regret，Token 一致率仅作辅助。

| 输入 | 比较步数 | 最大绝对误差 | 最大 RMS | 最小 cosine | 贪心一致率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 Token | 32 | 0.120226 | 0.021602 | 0.999973 | 100% |
| 128 Token | 32 | 0.040472 | 0.009491 | 0.999992 | 100% |
| 2048 Token | 32 | 0.054121 | 0.010401 | 0.999991 | 100% |
| 中文对话 | 64 | 0.053310 | 0.008371 | 0.999993 | 100% |
| 算术对话 | 3 | 0.051054 | 0.010807 | 0.999988 | 100% |

共 163 个参考比较步骤全部通过；独立干净进程重跑的全部 227 步输入、位置与 logits 又与已验证进程逐位一致（`results/qwen35-reproduction.json`）。原始逐步 max/RMS/p99/稳定化相对误差/cosine/top-1 数据见 `results/qwen35-correctness.json`。另两次长度 1、128 的 32 步请求在无关请求之后复用槽位，要求每步全部 logits **逐位一致**，均通过。包括重复请求的 227 个 forward 与输出、实际 Token 数、EOS 原因保存在 `results/qwen35-generation.json`；算术回答为 `42`，3 Token 后正常 EOS。最终 FP16 HTTP 启动也由另一干净进程复测，结果见 `results/qwen35-http-smoke.json`。

BF16 候选未通过同一门槛，保留失败证据 `results/qwen35-bf16-diagnostic.json`，没有放宽容差。采用 FP16 后通过全部门槛。FP16 转换不是无损转换：审计发现 268428 个元素变化，最大权重误差 0.00048828125，15948 个微小值下溢为零；`A_log/dt_bias` 保持 FP32、无溢出，见 `results/qwen35-fp16-weight-cast.json`。端到端参考仍使用原 checkpoint 的 FP32 值，因而上述比较包含转换的影响。

## 5. 性能基线

| 输入 / 输出 Token | Prefill forward 中位数 ms | TTFT 中位数 [min,max] ms | TPOT 中位数 [min,max] ms | TPOT 标准差 ms | 总时长中位数 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 32 | 109.63 | 159.64 [156.61, 189.97] | 45.84 [44.66, 46.04] | 0.44 | 1.581 |
| 128 / 128 | 107.32 | 156.91 [151.83, 158.24] | 44.78 [43.57, 45.09] | 0.50 | 5.844 |
| 2048 / 32 | 980.91 | 1031.98 [1030.48, 1033.94] | 44.95 [44.67, 45.90] | 0.37 | 2.426 |
| 2048 / 128 | 980.65 | 1031.08 [1025.47, 1042.52] | 44.74 [43.54, 71.20] | 8.44 | 6.719 |

原始 52 次请求（12 次预热、40 次正式）的时延、峰值占用、完整输出和实际步数保留在 `results/qwen35-benchmark.json`；每步计时与设备分配器占用保存在各请求的 `steps_file`（仓库外），紧凑表为 `results/qwen35-benchmark.csv`。2048/128 组包含一次 10.077 秒的请求（TPOT 71.197 ms），保留该样本，没有剔除异常值；共享主机调度等因素的具体归因未经独立验证。

长度以预先构造的实际 Token IDs 计数。每组独立预热 3 次、正式测量 10 次，单请求串行，固定长度试验使用 `ignore_eos=True`，正常 EOS 已单独验证。编译、加载和预热不计入正式值。Prefill/Decode forward 在调用前后同步；NPU event span 包含主机发射间隙，不能解释为纯 kernel 时间。

TTFT 从本地 Engine 提交到第一个流式帧，包含 IPC、调度、首个 forward、采样和输出送达，不含分词和 HTTP/网络。TPOT =（最后帧时间 − 第一帧时间）/（实际输出数 − 1）。性能模式不复制 logits 到 CPU，但边界同步与逐 forward JSON 记录会增加开销；这是可复现的 **eager、带观测基线**，不是优化后的服务吞吐。所有流式 delta Token IDs 累计后与 usage、forward 次数交叉检查。

加载后 NPU 分配器占用 1,574,239,232 字节（1.466 GiB）；LCM 预分配缓存 arena 113,246,208 字节（108 MiB，无 host offload）。启动摘要 allocated 约 1.82 GiB、reserved 2.00 GiB。

| 输入 Token | 推理峰值 allocated GiB | allocator reserved GiB |
| --- | ---: | ---: |
| 128 | 1.899 | 2.207 |
| 2048 | 2.019 | 2.207 |

这些值来自 `torch.npu` 分配器，不包括驱动上下文的全部开销。测量期间 `npu-smi info -i 3 -c 1 -t proc-mem` 只在目标 chip 观察到本实验 scheduler 2462 MB 与父进程 112 MB（工具原始单位）；两者均位于同一 chip。其他用户服务没有停止。

## 6. Profiler 与优化优先级

在独立进程中，128 和 2048 输入各先预热 3 次，再采集 Prefill 与首个 Decode 的 CPU/NPU trace。每段均有 `ProfilerStep#0`，并确认包含全部 18 个 GDN scan 和 6 个 FIA；按设备 task 的区间并集计算 busy，避免重叠任务重复计时。逐 kernel、类型及原 trace 摘要见 `results/qwen35-profile.json`，可用 `scripts/qwen35_profile_summary.py` 重新生成。

| 输入 / 阶段 | kernel 数 | kernel busy 并集 ms | 首末 kernel 跨度 ms | 因果卷积累计 ms | GDN scan 累计 ms | 显式 memcpy 累计 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 / Prefill | 1139 | 70.615 | 131.348 | 49.550 | 9.557 | 0.0050 |
| 128 / Decode | 526 | 7.059 | 55.468 | 2.886 | 0.394 | 0.0076 |
| 2048 / Prefill | 1139 | 970.594 | 988.352 | 756.623 | 146.881 | 0.0000 |
| 2048 / Decode | 526 | 7.418 | 53.088 | 2.847 | 0.386 | 0.0073 |

2048 Token Prefill 中，因果卷积 756.623 ms，占 kernel busy 约 **78.0%**；GDN scan 146.881 ms，约 **15.1%**。全 attention 仅 1.914 ms，所有 dense matmul 累计 11.679 ms。dtype/layout/device-copy 类为 3.757 ms（约 0.4%），没有显式 memcpy 事件。长 Prefill 约 98.2% 的首末 kernel 时间跨度有设备 kernel 在运行，因此当前优先级是**优化因果卷积**，其次是 GDN，不能把主要开销归到 attention 或主机传输。

源码显示复用的卷积 kernel 在每个 chunk 内按 Token 串行循环，做宽 4 的卷积与 SiLU；128→2048 时其累计耗时从 49.550→756.623 ms，约 15.3 倍，接近长度的 16 倍。后续可尝试 Ascend 友好的 Token/通道并行分块与更合适的内存访问，并重跑卷积状态与完整模型检查。这里的访存效率属于基于源码的优化假设：本次没有采集 HBM 带宽/算术强度计数器，不能据此断言已达到带宽或计算吞吐上限。

Decode 则有 526 个设备 kernel，总 busy 7.059/7.418 ms，首末跨度 55.468/53.088 ms，设备空隙约 86%/86%。显式 memcpy 不到 0.008 ms，dtype/layout/device-copy 约 0.16 ms。结合 CPU enqueue 时间线，eager 小算子发射和主机调度是另一优化方向，可先减少分散调用，再独立验证图捕获。profiler 会增加主机开销，因此不把这 53–55 ms 当作正式基线，也不将空隙全部解释成 CPU 数值计算；正式 decode forward 中位数约 40 ms、TPOT 约 45 ms。

本阶段完成正确性与瓶颈定位，没有混入未经验证的融合优化或性能提升声明。

## 7. 交付与局限

大文件均在仓库外：权重 `/tmp/hw1-qwen35-model`，完整 logits `/tmp/hw1-qwen35-final-validation`，测量中间文件 `/tmp/hw1-qwen35-benchmark-v2`，原始 profiler 数据 `/tmp/hw1-qwen35-profile-v2`。`/tmp` 可能被清理，可用仓库脚本、manifest 和文档重新生成。

已验证范围为 batch 1、单 chip、最大上下文配置 4096、输入测试 1/128/2048、单次完整 Prefill 后连续 Decode。关闭前缀复用和图捕获；head dim 256 的缓存续接 TND Prefill、并发 batch、视觉与多 chip 需要独立适配和测试。本报告不将未测特性列为支持项。

代码与实验记录按阶段提交并推送至远端分支。报告开头仍需小组填写真实成员贡献，教学网提交由小组完成；没有自动执行教学网提交。
