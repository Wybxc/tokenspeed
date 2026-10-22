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

"""Ascend normalization operators."""

from __future__ import annotations

import torch
import torch_npu
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _gemma_add_rmsnorm_kernel(
    X,
    Residual,
    Weight,
    Out,
    x_stride,
    r_stride,
    eps,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * x_stride + cols, cols < WIDTH, 0).to(tl.float32)
    residual = tl.load(Residual + row * r_stride + cols, cols < WIDTH, 0).to(tl.float32)
    value = x + residual
    weight = tl.load(Weight + cols, cols < WIDTH, 0).to(tl.float32)
    variance = tl.sum(value * value, 0) / WIDTH
    out = value * tl.rsqrt(variance + eps) * (1.0 + weight)
    tl.store(Residual + row * r_stride + cols, value, cols < WIDTH)
    tl.store(Out + row * WIDTH + cols, out, cols < WIDTH)


def gemma_add_rmsnorm(x, residual, weight, eps):
    """Add then normalize BF16/FP32 ``[T,D]`` on one NPU, using FP32 ``1+w``.

    The FP32 sum feeds the variance before rounding. Update the residual with
    that sum rounded to its dtype; return a new output and the same residual.
    """
    if x.ndim != 2 or x.shape != residual.shape or weight.shape != (x.shape[1],):
        raise ValueError("Gemma add RMSNorm expects matching [T,D] rows and [D] weight")
    if any(t.device != x.device or t.dtype != x.dtype for t in (residual, weight)):
        raise ValueError("Gemma add RMSNorm tensors must share dtype and device")
    if x.device.type != "npu" or any(t.stride(-1) != 1 for t in (x, residual, weight)):
        raise ValueError("Gemma add RMSNorm needs NPU tensors with contiguous channels")
    block = triton.next_power_of_2(x.shape[1])
    if block > 65536:
        raise ValueError("Gemma RMSNorm width exceeds one reduction block")
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if x.shape[0]:
        _gemma_add_rmsnorm_kernel[(x.shape[0],)](
            x,
            residual,
            weight,
            out,
            x.stride(0),
            residual.stride(0),
            eps,
            WIDTH=x.shape[1],
            BLOCK=block,
        )
    return out, residual


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm, optionally adding and updating a residual tensor."""
    if x.shape[0] == 0:
        result = x if out is None else out
        return (result, residual) if residual is not None else result
    if residual is not None:
        result, _, residual_out = torch_npu.npu_add_rms_norm(
            x, residual, weight, epsilon=eps
        )
        residual.copy_(residual_out)
        return result, residual
    result, _ = torch_npu.npu_rms_norm(x, weight, epsilon=eps)
    if out is not None:
        out.copy_(result)
        result = out
    return result


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm to query and key tensors."""
    del enable_pdl
    head_dim = q_weight.numel()
    if k_weight.numel() != head_dim:
        raise ValueError("q and k weights must have the same head dimension")
    if q.shape[-1] % head_dim or k.shape[-1] % head_dim:
        raise ValueError("q and k widths must be divisible by the head dimension")

    q_shape, k_shape = q.shape, k.shape
    q_heads = q.reshape(-1, q.shape[-1] // head_dim, head_dim)
    k_heads = k.reshape(-1, k.shape[-1] // head_dim, head_dim)
    q_scale = torch.rsqrt(q_heads.float().square().mean(-1, keepdim=True) + eps)
    k_scale = torch.rsqrt(k_heads.float().square().mean(-1, keepdim=True) + eps)
    q_out = (q_heads * q_scale.to(q.dtype) * q_weight).reshape(q_shape)
    k_out = (k_heads * k_scale.to(k.dtype) * k_weight).reshape(k_shape)
    return q_out, k_out


__all__ = ["qk_rmsnorm", "rmsnorm"]
