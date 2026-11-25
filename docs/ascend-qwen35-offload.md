# Qwen3.5 Ascend HBM–Host 分层缓存

本文记录 CKPT-3 的实现边界、安全性约束和可复现实验入口。实验报告及表格见仓库根目录的 `作业3实验报告.md`，原始结果见 `results/hw3-*.json` 和 `results/hw3-evidence/`。

## 实现边界

本阶段直接复用 TokenSpeed 的通用两级缓存框架：C++ 调度器管理 Device/Host 块、前缀标识、引用计数和生命周期，Python `HostCacheExecutor` 管理 D2H/H2D 流与完成事件，Triton Host transfer kernel 执行真实 DMA。Qwen3.5 的 Ascend 端主要适配和补强为：

- 在 Triton-Ascend 3.2.0 上使用 NPU 专用 causal-conv launch config，避免 910C 预填充越界；
- 消除静态循环中 Python `bool` 与 Triton mask 混合的编译错误；
- 将可选 `viztracer` 改为仅在对应 profiling 模式下延迟导入；
- 向 Python 绑定暴露 Host 空闲块和累计淘汰数，并增加 C++/绑定测试；
- 增加实验侧观测器，记录每个 TP rank 的真实本地传输行、字节数、完成延迟与块级 gauge。

## 缓存闭环

1. 调度器以链式前缀 hash 和 cache group 标识状态，分配 Device LCM 块。
2. Qwen3.5 的 full-attention KV 与 GDN 卷积/递归状态共用同一 cache arena；状态块是“消费到某一 token 位置后的完整快照”。
3. 普通冷前缀在不再活跃后写回 Host L2。主动 retraction 时，hash 完整的块进 Host L2，未对齐尾块进请求私有 snapshot pool，cache group 外的 slot state 进固定宽度 blob arena。
4. D2H 源页要么保持 pinned 直到 ACK，要么依靠 stream-ordered fence 在源页重用前完成。传输中的 Host 块不可淘汰。
5. 命中时从 Host L2 回载；retraction 恢复同时从 L2 和 snapshot pool 导入到新 Device 块，并恢复 slot-state blob。调度器只在 `RestoreDone` 后重新调度请求。
6. Host 不命中不会使请求失败：该前缀作为 miss 重做 prefill。Host 满时先淘汰未被请求证明有用、且最久未访问的已提交冷块。

TP=2 下，每一行在 Device 和 Host 两端都经过 ownership translation：分片 group 只在所有者 rank 传输，复制 group 在各 rank 保持一致。所有 rank 对同一 scheduler op 发送 ACK，因此恢复的 token 边界与生命周期一致。

## 实验入口

Ascend 环境中先设置可见设备并加载 CANN：

```bash
export ASCEND_RT_VISIBLE_DEVICES=6
source scripts/ascend_env.sh
```

单卡正确性基线：

```bash
python scripts/qwen35_offload_experiment.py \
  --model /path/to/Qwen3.5-0.8B --tp 1 --offload off \
  --workload correctness --host-gb 0.1 --snapshot-host-gb 0.25 \
  --device-tokens 8192 --max-seqs 3 --prompt-tokens 384 \
  --output-tokens 8 --repeats 2 --debug-retraction-interval 0 \
  --work-dir /tmp/hw3-off --output /tmp/hw3-off.json
```

强制经过快照/恢复的对照：

```bash
python scripts/qwen35_offload_experiment.py \
  --model /path/to/Qwen3.5-0.8B --tp 1 --offload on \
  --workload correctness --host-gb 0.1 --snapshot-host-gb 0.25 \
  --device-tokens 8192 --max-seqs 3 --prompt-tokens 384 \
  --output-tokens 8 --repeats 2 --debug-retraction-interval 2 \
  --work-dir /tmp/hw3-on --output /tmp/hw3-on.json

python scripts/qwen35_offload_compare.py --require-snapshot-restore \
  --baseline /tmp/hw3-off.json --offload /tmp/hw3-on.json \
  --output /tmp/hw3-compare.json
```

TP=2 需暴露同一板卡的两个 chip：

```bash
export ASCEND_RT_VISIBLE_DEVICES=6,7
```

`--workload reuse` 用于高复用工作集，`--workload churn` 用于低复用及 Host 淘汰。Qwen3.5 混合状态需跨过完整 LCM 提交边界才能跨请求复用；本次 0.8B 复用实验使用 1536-token prompt。

## 验证

```bash
python -m pytest -q tokenspeed-scheduler/python/tests/test_cache_binding.py

cmake -S tokenspeed-scheduler -B /tmp/tokenspeed-scheduler-tests \
  -DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON \
  -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=OFF
cmake --build /tmp/tokenspeed-scheduler-tests -j 8
/tmp/tokenspeed-scheduler-tests/tokenspeed_scheduler_tests
```
