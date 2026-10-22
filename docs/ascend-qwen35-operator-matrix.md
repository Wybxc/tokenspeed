# Qwen3.5-0.8B Ascend operator inventory

The fixed official config has 24 text layers: 18 Gated DeltaNet (GDN) layers
and six full-attention layers, interleaved 3:1. Hidden width is 1024 and MLP
width 3584. Attention has eight query heads, two KV heads, width 256 and
64 rotary channels. GDN has 16 key/value heads of width 128, convolution
width 4 and 6144 convolution channels. Text vocabulary size is 248320.
This task validates text generation; the checkpoint also contains vision and
MTP weights, which are outside the selected execution scope.

`T` denotes packed tokens, `B` requests and `P` allocated state pages.
Activations/weights are BF16 and recurrent state is FP32. Status distinguishes
source availability, dispatch availability, and actual hardware validation.

| Module | Input → output / mutable state | Implementation source | Initial Ascend status | Validation |
| --- | --- | --- | --- | --- |
| Embedding / tied LM head | IDs `[T]` → `[T,1024]`; final logits `[B,248320]` | Existing TokenSpeed embedding, dense PyTorch GEMM | Existing NPU path | Fixed-weight model logits and generation |
| QKV and GDN projections | `[T,1024]` → packed head/gate rows | Existing unquantized linear layers | Existing NPU path | Weight shapes and model comparison |
| MLP / SwiGLU | `[T,1024]` → `[T,2×3584]` → `[T,3584]` → `[T,1024]` | Existing GEMM and registered Ascend activation | Existing NPU path | Numerical module and model comparison |
| Gemma RMSNorm | BF16 `[T,1024]`, FP32 variance, weight offset `1+w` | Reused grouped Gemma norm; new Ascend fused residual add norm | 8 NPU tests passed | CPU FP64 reference; empty and large rows; exact residual writes |
| Gated RMSNorm | `[T,16,128]` plus gate, SiLU then norm | Existing GDN Triton normalization with device-neutral context | 6 NPU tests passed | Target-shape CPU FP64 reference, both gate orders |
| QK norm / partial RoPE | Q `[T,8,256]`, K `[T,2,256]`; 64 rotary channels | Existing composite attention prologue | Existing NPU path | RoPE tests and model comparison |
| Causal GQA / KV write | Q as above; K/V cache `[slots,2,256]` | Existing Ascend FIA; added BSH prefill layout for D=256 | 8 eager MHA tests passed | FP64 attention at lengths 1, packed 3/5, 128, 2048; paged decode |
| Causal depthwise convolution | Packed `[6144,T]` → same; window `[P,6144,3]` | Existing common Triton convolution | 19 NPU tests passed | Window boundaries, segmented input, read/write remapping |
| GDN gate | `a,b [T,16]`, `A_log,dt_bias [16]` → log decay/beta | Existing prefill gate; fused into new Ascend decode scan | Added decode implementation | CPU FP64 gate/recurrence comparison |
| GDN scan | Q/K/V `[1,T,16,128]` → `[1,T,16,128]`; state `[P,16,128,128]` | New Ascend tiled serial recurrence, registered through common kernel package | 20 NPU tests passed | FP64 reference, irregular lengths, grouped/non-square heads, multi-step state and isolation |
| Greedy sampling | `[B,248320]` → `[B]` token IDs | Existing TokenSpeed greedy backend | Existing NPU path | Fixed-token logits and generated IDs |

## GDN call chain and state ownership

`Qwen3_5GatedDeltaNet.forward` projects the complete hidden state, then calls
`PagedAttention`. The existing `GdnAttnBackend`/`MambaAttnBackend` owns forward
metadata, while the LCM cache subsystem owns state pages and their lifetime.
The state group's input checkpoint and live destination are distinct pages;
decode can evolve the live page in place after the checkpoint is released.
No new request-private state owner is introduced.

Prefill gathers the immutable input state, seeds the destination convolution
window and executes causal convolution. Q/K/V split into packed head views;
the gate produces log decay and sigmoid beta. Public `gdn_chunk_prefill`
selects `ascend_gdn_chunk_prefill`, which executes the NPU scan and returns
final FP32 state. The existing backend commits that state to the live page.
Decode reads raw gates inside the scan and updates only the selected live
page. Padded negative destinations cannot write state.

The Ascend scan retains the public `[request,head,value,key]` layout throughout;
it does not transpose it to FLA's `[request,head,key,value]` layout. Each program
owns 16 value rows of one head and reads device sequence bounds. It applies
the full decay, prediction, delta correction and query read at every token.
The scan uses FP32 arithmetic and rounds output once to the activation dtype.
Host synchronization belongs at measurement/output boundaries, not between
layers. Prefill and decode retain the existing scheduler and execution path.

The serial recurrence is a correctness baseline. Its long-input cost must be
measured before considering a parallel/chunked optimization. ACL Graph,
prefix reuse and speculative execution require independent validation before
being advertised as supported by this adaptation.

CANN 9 rejects head dimension 256 in its TND prefill interface. The added
path views each packed sequence as BSH and calls the same vendor FIA operator
with an explicit causal mask. Query, key, value and all attention math remain
on NPU. The existing BSH paged decode already supports D=256. Cache-backed
TND extend at D=256 is not validated; this configuration disables prefix reuse
and uses one whole prefill (up to 4094 input tokens), followed by decode.

The Qwen GDN convolution window is `[page,channel,width-1]`, as declared by
`LinearAttnConfig.conv_state_shape`; the `[slot,R,D]` ring described by the
generic convolution package applies to its separate Inkling operator. The
window contains the last three *inputs*, with the oldest entry first.
