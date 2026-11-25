# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Run reproducible Qwen3.5 HBM--Host cache experiments on Ascend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qwen35_offload_probe import install
from qwen35_probe import engine_kwargs

# Spawned TP workers execute this module before constructing their executor.
if os.environ.get("QWEN35_PROBE_DIR"):
    install()


def _write_control(root: Path, name: str) -> None:
    temporary = root / "active.tmp"
    temporary.write_text(json.dumps({"id": name, "mode": "benchmark"}))
    temporary.replace(root / "active.json")


def _request(engine, ids: list[int], output_tokens: int, *, session: str) -> dict:
    started = time.perf_counter()
    arrivals = []
    output_ids = []
    final = None
    for frame in engine.generate(
        input_ids=ids,
        sampling_params={
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        stream=True,
    ):
        arrivals.append(time.perf_counter())
        output_ids.extend(frame["output_ids"])
        final = frame
    assert final is not None and len(output_ids) == output_tokens
    return {
        "session": session,
        "input_tokens": len(ids),
        "output_ids": output_ids,
        "cached_tokens": int(final["meta_info"]["cached_tokens"]),
        "finish_reason": final["meta_info"]["finish_reason"],
        "ttft_ms": (arrivals[0] - started) * 1000,
        "wall_ms": (arrivals[-1] - started) * 1000,
        "tpot_ms": (arrivals[-1] - arrivals[0]) * 1000 / max(1, len(arrivals) - 1),
    }


def _wave(engine, requests, output_tokens: int) -> tuple[list[dict], float]:
    barrier = threading.Barrier(len(requests))

    def one(item):
        session, ids = item
        barrier.wait()
        return _request(engine, ids, output_tokens, session=session)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        results = list(pool.map(one, requests))
    return results, time.perf_counter() - started


def _tokens(pattern: list[int], length: int, salt: int) -> list[int]:
    # Keep most tokens common while giving every session a stable, distinct
    # first page.  This exercises both reuse and isolation.
    salted = [pattern[(index + salt * 17) % len(pattern)] for index in range(128)]
    tail = (pattern * (length // len(pattern) + 2))[: max(0, length - 128)]
    return (salted + tail)[:length]


def _summarize_trace(root: Path) -> dict:
    rank_dirs = sorted(root.glob("rank-*")) or [root]
    ranks = []
    for rank_dir in rank_dirs:
        capacity_path = rank_dir / "cache-capacity.json"
        events_path = rank_dir / "cache-events.jsonl"
        capacity = (
            json.loads(capacity_path.read_text()) if capacity_path.exists() else {}
        )
        events = (
            [json.loads(line) for line in events_path.read_text().splitlines()]
            if events_path.exists()
            else []
        )
        submitted = {}
        latency = {}
        gauges = [event for event in events if event["event"] == "gauge"]
        for event in events:
            kind = event.get("kind")
            if event["event"] == "submit":
                values = submitted.setdefault(
                    kind, {"batches": 0, "operations": 0, "blocks": 0, "bytes": 0}
                )
                values["batches"] += 1
                values["operations"] += len(event["op_ids"])
                values["blocks"] += event["blocks"]
                values["bytes"] += event["bytes"]
            elif event["event"] == "complete":
                latency.setdefault(kind, []).append(event["latency_ms"])
        ranks.append(
            {
                "rank": capacity.get("rank", len(ranks)),
                "capacity": capacity,
                "transfers": submitted,
                "completion_latency_ms": {
                    kind: {
                        "count": len(values),
                        "median": statistics.median(values),
                        "max": max(values),
                    }
                    for kind, values in latency.items()
                },
                "gauges": {
                    "samples": len(gauges),
                    "final": gauges[-1] if gauges else None,
                    "min_device_empty_blocks": min(
                        (g["device_empty_blocks"] for g in gauges), default=None
                    ),
                    "max_device_empty_blocks": max(
                        (g["device_empty_blocks"] for g in gauges), default=None
                    ),
                    "min_device_active_blocks": min(
                        (g["device_active_blocks"] for g in gauges), default=None
                    ),
                    "max_device_active_blocks": max(
                        (g["device_active_blocks"] for g in gauges), default=None
                    ),
                    "max_device_cached_blocks": max(
                        (g["device_cached_blocks"] for g in gauges), default=None
                    ),
                    "max_host_cached_blocks": max(
                        (g["host_cached_blocks"] for g in gauges), default=None
                    ),
                    "max_host_pinned_blocks": max(
                        (g["host_pinned_blocks"] for g in gauges), default=None
                    ),
                    "host_evictions": max(
                        (g["host_evictions"] for g in gauges), default=0
                    ),
                    "min_snapshot_free_blocks": min(
                        (g["snapshot_free_blocks"] for g in gauges), default=None
                    ),
                    "max_retracted_requests": max(
                        (g["retracted_requests"] for g in gauges), default=0
                    ),
                },
            }
        )
    return {"ranks": ranks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--offload", choices=("on", "off"), required=True)
    parser.add_argument(
        "--workload", choices=("correctness", "reuse", "churn"), required=True
    )
    parser.add_argument("--host-gb", type=float, required=True)
    parser.add_argument("--snapshot-host-gb", type=float, required=True)
    parser.add_argument("--device-tokens", type=int, required=True)
    parser.add_argument("--max-seqs", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--debug-retraction-interval", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.work_dir.exists():
        parser.error("Use a fresh work directory")
    if (
        min(
            args.device_tokens,
            args.max_seqs,
            args.prompt_tokens,
            args.output_tokens,
            args.repeats,
        )
        <= 0
    ):
        parser.error("Token counts, sequence count and repeats must be positive")
    if args.offload == "off" and args.debug_retraction_interval:
        parser.error("Forced retraction requires the Host snapshot pool")
    devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    if len(devices) != args.tp or len(set(devices)) != args.tp:
        parser.error("Expose exactly one distinct physical chip per TP rank")
    if args.tp == 2 and int(devices[0]) // 2 != int(devices[1]) // 2:
        parser.error("TP=2 must use the two chips of one board")

    args.work_dir.mkdir(parents=True)
    os.environ["QWEN35_PROBE_DIR"] = str(args.work_dir.resolve())
    install()

    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pattern = tokenizer.encode(
        "Hierarchical cache experiments must preserve every recurrent state and attention value. ",
        add_special_tokens=False,
    )
    config = engine_kwargs(args.model)
    config.update(
        world_size=args.tp,
        attn_tp_size=args.tp,
        dense_tp_size=args.tp,
        enable_allreduce_fusion=False,
        enable_prefix_caching=True,
        disable_kvstore=args.offload == "off",
        kvstore_size=args.host_gb if args.offload == "on" else 0,
        kvstore_io_backend="direct",
        retraction_snapshot_host_gb=(
            args.snapshot_host_gb if args.offload == "on" else 0
        ),
        retraction_snapshot_ratio=0,
        retraction_snapshot_max_requests=(args.max_seqs if args.offload == "on" else 0),
        debug_force_retraction_interval=args.debug_retraction_interval,
        max_num_seqs=args.max_seqs,
        max_total_tokens=args.device_tokens,
        max_model_len=max(4096, args.prompt_tokens + args.output_tokens + 256),
        chunked_prefill_size=min(4096, args.device_tokens),
        disable_overlap_schedule=True,
    )
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    result = {"arguments": arguments, "config": config, "waves": []}
    repository = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {
        name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
        for name in (
            "python/tokenspeed/runtime/cache/l2/executor.py",
            "python/tokenspeed/runtime/cache/transfer/ops.py",
            "tokenspeed-kernel/python/tokenspeed_kernel/ops/kvcache/host_transfer.py",
            "tokenspeed-scheduler/csrc/cache/coordinator/cache_coordinator.cpp",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run(name: str, requests) -> None:
        _write_control(args.work_dir, name)
        rows, elapsed = _wave(engine, requests, args.output_tokens)
        wave = {
            "name": name,
            "wall_seconds": elapsed,
            "output_tokens_per_second": len(rows) * args.output_tokens / elapsed,
            "requests": rows,
        }
        result["waves"].append(wave)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "name": name,
                    "throughput": wave["output_tokens_per_second"],
                    "cached": [r["cached_tokens"] for r in rows],
                }
            ),
            flush=True,
        )

    with Engine(**config) as engine:
        if args.workload == "correctness":
            for repeat in range(args.repeats):
                requests = [
                    (
                        f"session-{session}",
                        _tokens(pattern, args.prompt_tokens, session),
                    )
                    for session in range(args.max_seqs)
                ]
                run(f"correctness-{repeat}", requests)
        elif args.workload == "reuse":
            hot = _tokens(pattern, args.prompt_tokens, 0)
            for repeat in range(args.repeats):
                run(f"hot-{repeat}-before", [("hot", hot)])
                for salt in range(1, args.max_seqs + 2):
                    run(
                        f"hot-{repeat}-churn-{salt}",
                        [
                            (
                                f"churn-{salt}",
                                _tokens(
                                    pattern, args.prompt_tokens, 100 * repeat + salt
                                ),
                            )
                        ],
                    )
                run(f"hot-{repeat}-after", [("hot", hot)])
        else:
            first = _tokens(pattern, args.prompt_tokens, 1000)
            run("churn-first", [("first", first)])
            for repeat in range(args.repeats * (args.max_seqs + 2)):
                run(
                    f"churn-unique-{repeat}",
                    [
                        (
                            f"unique-{repeat}",
                            _tokens(pattern, args.prompt_tokens, 2000 + repeat),
                        )
                    ],
                )
            run("churn-revisit", [("first", first)])

    (args.work_dir / "active.json").unlink(missing_ok=True)
    result["trace"] = _summarize_trace(args.work_dir)
    all_requests = [request for wave in result["waves"] for request in wave["requests"]]
    result["summary"] = {
        "requests": len(all_requests),
        "cache_hits": sum(request["cached_tokens"] > 0 for request in all_requests),
        "cache_misses": sum(request["cached_tokens"] == 0 for request in all_requests),
        "median_ttft_ms": statistics.median(
            request["ttft_ms"] for request in all_requests
        ),
        "median_tpot_ms": statistics.median(
            request["tpot_ms"] for request in all_requests
        ),
        "output_tokens_per_second": sum(
            len(request["output_ids"]) for request in all_requests
        )
        / sum(wave["wall_seconds"] for wave in result["waves"]),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
