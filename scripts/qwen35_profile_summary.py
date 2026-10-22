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

"""Summarize Ascend profiler traces without counting overlapping kernels twice."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from decimal import Decimal
from pathlib import Path


def union_ms(events, origin):
    intervals = []
    for event in events:
        start = float(Decimal(str(event["ts"])) - origin)
        intervals.append((start, start + float(event["dur"])))
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total / 1000


def family(name):
    lower = name.lower()
    if "causal_conv" in lower:
        return "causal_convolution"
    if "gdn_scan" in lower:
        return "gdn_recurrence"
    if "matmul" in lower:
        return "dense_matmul"
    if "fusedinferattention" in lower:
        return "attention"
    if "rmsnorm" in lower or "layernorm" in lower:
        return "normalization"
    if any(part in lower for part in ("cast", "transpose", "copy", "transdata")):
        return "dtype_layout_device_copy"
    if "index" in lower or "gather" in lower:
        return "device_indexing"
    return "other_device_math"


def summarize(path):
    trace = json.loads(path.read_text())
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    complete = [e for e in events if e.get("ph") == "X"]
    kernels = [
        e
        for e in complete
        if e.get("args", {}).get("Task Type", "").startswith("KERNEL")
    ]
    transfers = [e for e in complete if "MEMCPY" in e["name"]]
    assert kernels, path
    origin = min(Decimal(str(e["ts"])) for e in kernels)
    last = max(Decimal(str(e["ts"])) + Decimal(str(e["dur"])) for e in kernels)
    by_name = collections.defaultdict(list)
    by_family = collections.defaultdict(list)
    for event in kernels:
        by_name[event["name"]].append(event)
        by_family[family(event["name"])].append(event)
    # The model must execute all 18 GDN and six full-attention layers.
    assert len(by_name["_gdn_scan"]) == 18
    assert len(by_name["FusedInferAttentionScore"]) == 6
    return dict(
        trace=str(path),
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        request=path.parts[-5],
        forward=path.parts[-4],
        profiler_step_ms=[
            e["dur"] / 1000 for e in complete if e["name"].startswith("ProfilerStep#")
        ],
        kernel_count=len(kernels),
        task_types=dict(collections.Counter(e["args"]["Task Type"] for e in kernels)),
        kernel_duration_sum_ms=sum(e["dur"] for e in kernels) / 1000,
        kernel_busy_union_ms=union_ms(kernels, origin),
        first_to_last_kernel_ms=float(last - origin) / 1000,
        transfers=dict(
            count=len(transfers),
            duration_sum_ms=sum(e["dur"] for e in transfers) / 1000,
        ),
        families={
            name: dict(
                count=len(group),
                duration_sum_ms=sum(e["dur"] for e in group) / 1000,
                busy_union_ms=union_ms(group, origin),
            )
            for name, group in sorted(by_family.items())
        },
        top_kernels=[
            dict(
                name=name,
                count=len(group),
                duration_sum_ms=sum(e["dur"] for e in group) / 1000,
            )
            for name, group in sorted(
                by_name.items(),
                key=lambda item: sum(e["dur"] for e in item[1]),
                reverse=True,
            )[:15]
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(
        Path(args.work_dir).glob(
            "profile-*/profile-*/*/ASCEND_PROFILER_OUTPUT/trace_view.json"
        )
    )
    assert (
        len(paths) == 4
    ), f"Expected two input lengths and prefill/decode traces, got {len(paths)}"
    result = dict(
        method="Chrome trace device task durations; interval union excludes overlapping device tasks; explicit memcpy events counted separately",
        limitations="Profiler perturbs host timings. Device task duration does not determine arithmetic intensity or achieved HBM bandwidth. No bandwidth counters collected.",
        traces=[summarize(path) for path in paths],
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
