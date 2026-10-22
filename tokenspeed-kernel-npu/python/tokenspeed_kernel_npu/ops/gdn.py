# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Exact gated delta recurrence on Ascend, with FP32 K-last state.

Each program retains a tile of one head's state while traversing the sequence.
Sequence bounds and page indices are read on device. No model tensor is copied
to the host. The serial scan is an initial correctness/performance baseline.
"""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _gdn_scan(
    Q,
    Kp,
    Vp,
    G,
    Beta,
    S,
    Dst,
    Out,
    Bounds,
    Read,
    Write,
    ALog,
    Bias,
    qr,
    kr,
    vr,
    gr,
    br,
    s0,
    s1,
    s2,
    s3,
    d0,
    d1,
    d2,
    d3,
    qh,
    kh,
    vh,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DECODE: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    vi = tl.program_id(2) * BV + tl.arange(0, BV)
    ki = tl.arange(0, BK)
    qhead = head // (HV // H)
    if DECODE:
        start = row
        end = row + 1
        src = tl.load(Read + row).to(tl.int64)
        dst = tl.load(Write + row).to(tl.int64)
    else:
        start = tl.load(Bounds + row)
        end = tl.load(Bounds + row + 1)
        src = row.to(tl.int64)
        dst = src
    valid = (vi[:, None] < DV) & (ki[None, :] < DK)
    state = tl.load(
        S + src * s0 + head * s1 + vi[:, None] * s2 + ki[None, :] * s3,
        mask=valid & (src >= 0),
        other=0,
    ).to(tl.float32)
    for t in range(start, end):
        q = tl.load(Q + t * qr + qhead * qh + ki, ki < DK, 0).to(tl.float32)
        k = tl.load(Kp + t * kr + qhead * kh + ki, ki < DK, 0).to(tl.float32)
        v = tl.load(Vp + t * vr + head * vh + vi, vi < DV, 0).to(tl.float32)
        if NORMALIZE:
            q *= tl.rsqrt(tl.sum(q * q, 0) + 1e-6)
            k *= tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
        g = tl.load(G + t * gr + head).to(tl.float32)
        beta = tl.load(Beta + t * br + head).to(tl.float32)
        if DECODE:
            x = g + tl.load(Bias + head).to(tl.float32)
            softplus = tl.log(1.0 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0.0)
            g = -tl.exp(tl.load(ALog + head).to(tl.float32)) * softplus
            beta = tl.sigmoid(beta)
        state *= tl.exp(g)
        delta = beta * (v - tl.sum(state * k[None, :], 1))
        state += delta[:, None] * k[None, :]
        out = tl.sum(state * (q * scale)[None, :], 1)
        tl.store(Out + (t * HV + head) * DV + vi, out, vi < DV)
    tl.store(
        Dst + dst * d0 + head * d1 + vi[:, None] * d2 + ki[None, :] * d3,
        state,
        valid & (dst >= 0),
    )


def _validate(q, k, v, state):
    if q.device.type != "npu" or any(x.device != q.device for x in (k, v, state)):
        raise ValueError("GDN inputs and state must share one NPU")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("GDN expects q/k [B,T,H,K] and v [B,T,HV,V]")
    if q.shape[:2] != v.shape[:2] or v.shape[2] % q.shape[2]:
        raise ValueError("GDN token dimensions and grouped heads must agree")
    if any(x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError("GDN input head channels must be contiguous")
    if state.dtype != torch.float32 or state.shape[1:] != (
        v.shape[2],
        v.shape[3],
        q.shape[3],
    ):
        raise ValueError("Ascend GDN requires FP32 state [pages,HV,V,K]")


def gdn_prefill(
    q,
    k,
    v,
    g,
    beta,
    *,
    scale,
    initial_state,
    cu_seqlens,
    qk_l2norm,
    output_final_state,
    output_h,
):
    """Scan packed ``[1,T,H,K]`` Q/K and ``[1,T,HV,V]`` V on the NPU.

    Gates are log decay and post-sigmoid beta ``[1,T,HV]``. Bounds are
    ``[N+1]`` device integers. Input state ``[N,HV,V,K]`` is immutable;
    the returned FP32 state has the same layout. Checkpoints are unsupported.
    """
    _validate(q, k, v, initial_state)
    if output_h:
        raise ValueError("Ascend GDN does not produce intermediate checkpoints")
    if q.shape[0] != 1 or initial_state.shape[0] != cu_seqlens.numel() - 1:
        raise ValueError("Packed prefill needs one state per sequence")
    if g.shape != v.shape[:3] or beta.shape != g.shape:
        raise ValueError("GDN gates must have shape [1,T,HV]")
    out = torch.empty_like(v)
    final = torch.empty_like(initial_state)
    dk, dv, hv = q.shape[-1], v.shape[-1], v.shape[-2]
    _gdn_scan[(initial_state.shape[0], hv, triton.cdiv(dv, 16))](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        final,
        out,
        cu_seqlens,
        cu_seqlens,
        cu_seqlens,
        g,
        g,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        g.stride(1),
        beta.stride(1),
        *initial_state.stride(),
        *final.stride(),
        q.stride(2),
        k.stride(2),
        v.stride(2),
        dk**-0.5 if scale is None else scale,
        H=q.shape[-2],
        HV=hv,
        DK=dk,
        DV=dv,
        BK=triton.next_power_of_2(dk),
        BV=16,
        DECODE=False,
        NORMALIZE=qk_l2norm,
    )
    return out, final if output_final_state else None


def gdn_decode(
    q,
    k,
    v,
    *,
    A_log,
    a,
    dt_bias,
    b,
    initial_state,
    initial_state_indices,
    scale,
    output_state_indices,
    use_qk_l2norm,
):
    """One-token decode from/to device page indices ``[B]``.

    Raw gates ``a,b`` have shape ``[B,1,HV]``. FP32 K-last state is
    updated only at destination pages. Negative destination IDs skip writes;
    negative source IDs contribute a zero initial state. Distinct requests
    must have distinct destinations and must not overwrite another's source.
    """
    _validate(q, k, v, initial_state)
    if q.shape[1] != 1 or a.shape != v.shape[:3] or b.shape != a.shape:
        raise ValueError("Decode needs one token per request and [B,1,HV] gates")
    out = torch.empty_like(v)
    destination = (
        initial_state_indices if output_state_indices is None else output_state_indices
    )
    dk, dv, hv = q.shape[-1], v.shape[-1], v.shape[-2]
    _gdn_scan[(q.shape[0], hv, triton.cdiv(dv, 16))](
        q,
        k,
        v,
        a,
        b,
        initial_state,
        initial_state,
        out,
        initial_state_indices,
        initial_state_indices,
        destination,
        A_log,
        dt_bias,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        a.stride(0),
        b.stride(0),
        *initial_state.stride(),
        *initial_state.stride(),
        q.stride(2),
        k.stride(2),
        v.stride(2),
        dk**-0.5 if scale is None else scale,
        H=q.shape[-2],
        HV=hv,
        DK=dk,
        DV=dv,
        BK=triton.next_power_of_2(dk),
        BV=16,
        DECODE=True,
        NORMALIZE=use_qk_l2norm,
    )
    return out
