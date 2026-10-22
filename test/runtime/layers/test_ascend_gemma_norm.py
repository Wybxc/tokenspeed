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

"""Exercise real runtime Gemma norms and their NPU fused residual entry."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed.runtime.layers.attention.linear.layernorm_gated import RMSNorm
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Gemma normalization requires Ascend"
)


@pytest.mark.parametrize("rows", [0, 1, 7, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_gemma_norm_and_fused_add(rows, dtype):
    torch.manual_seed(43)
    norm = GemmaRMSNorm(1024, 1e-6).to(device="npu", dtype=dtype)
    norm.weight.copy_(torch.randn_like(norm.weight) * 0.1)
    x = torch.randn(rows, 1024, device="npu", dtype=dtype)
    residual = torch.randn_like(x)

    def reference(value):
        value = value.cpu().double()
        return (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            * (1.0 + norm.weight.cpu().double())
        )

    # Compare to unrounded FP64: BF16's nearest rounding has <= 1/256
    # relative error. Comparing two independently rounded BF16 references
    # would instead penalize a one-ULP tie caused by FP32 reduction order.
    rtol = 4e-3 if dtype == torch.bfloat16 else 1e-5

    expected = reference(x.float())
    torch.testing.assert_close(norm(x).cpu().double(), expected, atol=1e-6, rtol=rtol)
    total = x.float() + residual.float()
    expected = reference(total)
    out, updated = norm(x.clone(), residual.clone())
    torch.testing.assert_close(out.cpu().double(), expected, atol=1e-6, rtol=rtol)
    torch.testing.assert_close(updated, total.to(dtype), atol=0, rtol=0)
    out, fp8, updated = norm.add_norm_with_fp8(x.clone(), residual.clone(), None)
    assert fp8 is None
    torch.testing.assert_close(out.cpu().double(), expected, atol=1e-6, rtol=rtol)
    torch.testing.assert_close(updated, total.to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 7, 128])
@pytest.mark.parametrize("before_gate", [True, False])
@torch.no_grad()
def test_gdn_gated_norm(rows, before_gate):
    torch.manual_seed(47)
    norm = RMSNorm(
        128, eps=1e-6, norm_before_gate=before_gate, device="npu", dtype=torch.bfloat16
    )
    norm.weight.copy_(torch.randn_like(norm.weight) * 0.1 + 1)
    x, z = [torch.randn(rows, 16, 128, device="npu").bfloat16() for _ in range(2)]
    xd, zd, wd = [t.cpu().double() for t in (x, z, norm.weight)]
    gate = torch.nn.functional.silu(zd)
    value = xd if before_gate else xd * gate
    expected = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * wd
    if before_gate:
        expected *= gate
    torch.testing.assert_close(
        norm(x, z).cpu().double(), expected, atol=4e-3, rtol=4e-3
    )
