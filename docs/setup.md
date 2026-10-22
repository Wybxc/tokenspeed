# Single-chip Ascend development environment

Work starts from `b8137dd1` on branch `codex/ascend-qwen35-08b`.
The host has four boards with two chips each. Physical device 7 is board 3,
chip 1, PCIe `0000:93:00.0`, with 64 GiB HBM. Exposing only device 7 maps it
to logical `npu:0`. Other users' serving processes must remain running.

Observed stack: AArch64 Linux, driver 25.5.2, CANN 9.0.0
(`V100R001C10SPC001B250`), Python 3.11.16, PyTorch 2.9.0+cpu with
torch-npu 2.9.0. The board query reports firmware as `NA`; it has not been
independently identified. The existing `/opt/miniconda3/envs/aics` environment
has Transformers 5.17.0, NumPy 2.4.6 and Triton-Ascend 3.2.0.

Before installing anything, the existing nanoGPT completed this check:

```bash
cd /workspace/nanoGPT
source /usr/local/Ascend/cann-9.0.0/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  /opt/miniconda3/envs/aics/bin/python sample.py \
  --out_dir=out-shakespeare-char --num_samples=1 --max_new_tokens=16 \
  --device=npu:0 --compile=False
```

It loaded the existing 10.65M-parameter checkpoint and generated
`DUKE VINCENTIO:`. The explicit loading override is needed by this existing
trusted checkpoint's older serialization format. No source or dependencies in
that environment were changed.

Create an isolated overlay that reuses the installed framework:

```bash
cd /home/u2501112122/hw1/tokenspeed
/opt/miniconda3/envs/aics/bin/python -m venv --system-site-packages .venv
```

The initial combined dependency installation was
rejected by pip: Triton-Ascend 3.2.1 pins NumPy 1.26.4, whereas the existing
environment uses NumPy 2.4.6. The overlay keeps the original framework intact;
extension compatibility must be demonstrated by executing NPU kernels.

The following installation completed in the isolated overlay:

```bash
source .venv/bin/activate
python -m pip install -i https://mirrors.huaweicloud.com/repository/pypi/simple \
  pytest pre-commit uv apache-tvm-ffi==0.1.13 transformers==5.12.0
python -m pip install --no-deps triton-ascend==3.2.1 \
  --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
UV_PYTHON_INSTALL_DIR=/tmp/hw1-python uv python install 3.12
```

Triton-Ascend 3.2.0 in the original environment could not compile its driver
against CANN 9.0.0 (`RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` missing). The overlay's
3.2.1 compiled and executed the GDN scan successfully with NumPy 2.4.6.
`apache-tvm-ffi==0.1.13` is the upstream recipe's pin; PyPI marks it yanked.
We retain it for this fixed reproduction and record that installation fact.

From the repository root, explicitly select the chip and source the helper:

```bash
export ASCEND_RT_VISIBLE_DEVICES=7
source scripts/ascend_env.sh
pytest -q tokenspeed-kernel-npu/test/test_gdn.py -x
```

Result: **20 passed** on the target NPU. This covers FP64 recurrence comparison
at lengths 1, 3, 4, 5, 63, 64, 65, 129 and 2048, BF16/FP32 outputs, actual
16×128×128 head geometry, grouped/non-square heads, paged multistep decode,
immutable source pages, ragged and empty requests, and split-prefill equality.
BF16 output tolerance is `atol=1e-3, rtol=4e-3` (one final BF16 rounding);
FP32 state tolerance is `atol=3e-6, rtol=3e-5` against a CPU FP64 recurrence.

Model files reside at `/tmp/hw1-qwen35-model`, outside Git. The official
ModelScope Qwen repository is reachable from this host; direct Hugging Face
connections reset. `results/qwen35-model-manifest.json` fixes each file's
revision, size and SHA256 digest. The complete checkpoint is 1,746,942,600 bytes.
The acquisition script validates that manifest, including the weight checksum.

```bash
python scripts/download_qwen35.py \
  --manifest results/qwen35-model-manifest.json --output /tmp/hw1-qwen35-model
OMP_NUM_THREADS=4 pytest -q tokenspeed-kernel-npu/test/test_causal_conv.py -x
```

Both commands passed: all files verified, and **19 convolution tests passed**.
The convolution tests cover lengths 1, 2, 3, 4, 5, 63, 64, 65 and 129 with
17 and 6144 channels, nonzero history, multiple decode updates, distinct
read/write pages and reinitialized request slots. Outputs use
`atol=4e-3, rtol=4e-3` against CPU FP64 convolution plus SiLU; stored BF16
input history must match exactly.

Additional runtime dependencies were installed with the framework pinned:

```bash
python -m pip install -i https://mirrors.huaweicloud.com/repository/pypi/simple \
  torch==2.9.0 transformers==5.12.0 compressed-tensors==0.15.0 \
  tokenspeed-scheduler==0.1.41 msgspec orjson einops fastapi uvloop pyzmq \
  grpcio==1.81.1 grpcio-health-checking==1.81.1 grpcio-reflection==1.81.1 \
  ninja openai setproctitle viztracer partial-json-parser pybase64 python-multipart \
  tokenspeed-smg-grpc-proto==0.4.27.post20261019 \
  tokenspeed-smg-grpc-servicer==0.14.0.post20261019 \
  tokenspeed-smg==1.13.0.post20261019
PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
  CMAKE_BUILD_PARALLEL_LEVEL=4 \
  CMAKE_ARGS=-DOPENSSL_ROOT_DIR=/opt/miniconda3/envs/aics \
  python -m pip install --no-deps -e tokenspeed-scheduler
```

The final source build is necessary: the released 0.1.41 wheel lacks
`StorageConfig`, which this checkout imports. CMake reuses existing Conda
OpenSSL headers/libraries because the system development headers are absent.
No scheduler version bump or system installation is needed.
`compressed-tensors==0.15.0` supports PyTorch 2.9; newer releases require 2.10.
The measured package snapshot is `results/ascend-environment.json`.
Install the local NPU package and the HTTP control dependency:

```bash
PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
  python -m pip install --no-deps -e tokenspeed-kernel-npu
python -m pip install -i https://mirrors.huaweicloud.com/repository/pypi/simple uvicorn
python test/ci_system/verify_triton_ascend.py
OMP_NUM_THREADS=4 pytest -q test/runtime/layers/test_ascend_gemma_norm.py
OMP_NUM_THREADS=4 pytest -q tokenspeed-kernel-npu/test/test_mha.py -k 'not npugraph'
ASCEND_RT_VISIBLE_DEVICES=7 scripts/serve_qwen35.sh /tmp/hw1-qwen35-model 31957
```

The initial BF16 checks passed: Triton verification, 14 norm tests and 8 eager MHA tests
(the separate NPUGraph test is deliberately deselected). The server reached
`ts serve ready` on localhost port 31957. Two sequential OpenAI chat requests
worked: `17 + 25` returned `42` and EOS after 3 tokens, and a Chinese explanation
generated 64 tokens. That preliminary launch used all 24 text layers, BF16 weights and
activations, FP32 recurrent state, eager execution and greedy sampling.
Prefix reuse, KV offload and retraction snapshots are explicitly disabled.
The small single-request cache otherwise rounds its default snapshot pool
down to zero whole blocks and refuses startup.

The final acceptance launcher selects **FP16** compute and cache, keeping
FP32 GDN state. The BF16 candidate exceeded the initially fixed end-to-end
logit tolerances on a length-one prompt. Its diagnostic summary is retained in
`results/qwen35-bf16-diagnostic.json`; tolerances were not relaxed. The explicit
FP16 cache option avoids an unsupported FP16-query/BF16-cache combination in
the vendor attention operator. Conv history follows FP16 compute in the Qwen
cache recipe. The original BF16 `auto` cache default is unchanged.

The expanded operator suite passed **126 tests** (one NPUGraph test
deselected), including FP16, BF16 and FP32 where applicable:

```bash
export ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4
source scripts/ascend_env.sh
pytest -q test/runtime/layers/test_ascend_gemma_norm.py \
  tokenspeed-kernel-npu/test/test_gdn.py \
  tokenspeed-kernel-npu/test/test_causal_conv.py \
  tokenspeed-kernel-npu/test/test_mha.py \
  tokenspeed-kernel-npu/test/test_layernorm.py \
  tokenspeed-kernel-npu/test/test_rotary_embedding.py -k 'not npugraph'
pytest -q test/runtime/test_cache_setup.py test/runtime/test_cli_config_compat.py
python scripts/qwen35_experiment.py validate --model /tmp/hw1-qwen35-model \
  --work-dir /tmp/hw1-qwen35-final-validation --output results/qwen35-generation.json
python scripts/qwen35_experiment.py compare --model /tmp/hw1-qwen35-model \
  --work-dir /tmp/hw1-qwen35-final-validation --output results/qwen35-correctness.json
```

Use a fresh work directory for each `validate` run; `compare` reads that run's
saved inputs/logits. The configuration/cache regressions passed **195 tests
and 74 subtests**; two existing allocation tests require CUDA and were skipped.
Full-model comparison passed 163 forward steps against Transformers 5.12.0 CPU
FP32, using the same checkpoint values and the exact NPU-generated contexts.
Each step compares all 248320 logits, not only sampled tokens. The fixed gates
are RMS error ≤0.10 logit, maximum absolute error ≤0.75 and cosine ≥0.999.
Two repeated requests (1 and 128 input tokens) additionally require bitwise
equal logits at all 32 steps after unrelated requests. Normal EOS is checked
separately from fixed-length `ignore_eos` trials. Large logits are outside Git
in the specified work directory; the small summaries are under `results/`.

An independent fresh process repeated all 227 captured forwards (including
the two repeated requests). Inputs, positions and every logit were bitwise
equal to the previously accepted run; see `results/qwen35-reproduction.json`.
The experiment accumulates delta token IDs from every streaming frame and
checks the total against both completion usage and the number of forwards.

The following four-grid benchmark and separate profiler runs completed:

```bash
export ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4
source scripts/ascend_env.sh
python scripts/qwen35_experiment.py benchmark --model /tmp/hw1-qwen35-model \
  --work-dir /tmp/hw1-qwen35-benchmark-v2 --output results/qwen35-benchmark.json
python scripts/qwen35_experiment.py profile --model /tmp/hw1-qwen35-model \
  --work-dir /tmp/hw1-qwen35-profile-v2 --output results/qwen35-profile-requests.json
python scripts/qwen35_profile_summary.py --work-dir /tmp/hw1-qwen35-profile-v2 \
  --output results/qwen35-profile.json
```

Use fresh directories on subsequent runs. Run measurements alone on the
selected chip; do not overlap operator tests or profiler with benchmarks.
Each input/output pair (128/32, 128/128, 2048/32, 2048/128) has three warmups
and ten formal requests. The profiler separately warms each shape three times
before recording one prefill and one decode, with an explicit completed
profiler step. Its internal zero-warmup schedule warning refers to the
single recorded step, not the preceding model warmups.
All 18 GDN and six full-attention kernels are present in each trace.
Device transfer events, layout/cast kernels and device busy interval union
are reported separately; profiler times are not substituted for benchmark
latencies. See [the report](ascend-qwen35-report.md) for definitions and results.

For the final FP16 HTTP check, start the launcher in one terminal and wait for
`ts serve ready`. Run the smoke test in another terminal, then stop the
launcher with Ctrl-C:

```bash
ASCEND_RT_VISIBLE_DEVICES=7 scripts/serve_qwen35.sh /tmp/hw1-qwen35-model 31957
```

```bash
python3 scripts/qwen35_http_smoke.py --base-url http://127.0.0.1:31957/v1 \
  --output results/qwen35-http-smoke.json
```

This clean-process FP16 launch returned `42` with normal EOS after three
tokens, then generated 64 Chinese tokens in the next request. The large
profiler traces/logits stay outside Git. The upstream Engine can print pending
async task messages during context-manager shutdown; the experiments exited
successfully after writing their checked results.
