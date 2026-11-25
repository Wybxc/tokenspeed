# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Experiment-only cache lifecycle and transfer observers for homework 3.

The serving runtime does not import this module.  It records actual local-rank
DMA rows after ownership translation, scheduler block gauges, and completion
latency without synchronizing the NPU streams.
"""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path

import torch
from qwen35_probe import install as install_model_probe
from qwen35_probe import rank_directory


def _root() -> Path:
    return rank_directory(Path(os.environ["QWEN35_PROBE_DIR"]))


def _append(record: dict) -> None:
    record = {"time_ns": time.time_ns(), **record}
    with (_root() / "cache-events.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _transfer_bytes(storage, transfers) -> int:
    if storage is None:
        return 0
    return sum(
        int(storage.host_cache_block_bytes[int(group_index)])
        for group_index, _, _ in transfers
    )


def install() -> None:
    """Install model, Host-transfer and scheduler-gauge observers once."""
    install_model_probe()

    from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
    from tokenspeed.runtime.cache.transfer.ops import HostTier
    from tokenspeed.runtime.engine.event_loop import EventLoop

    if getattr(HostCacheExecutor.poll_results, "_hw3_probe", False):
        return

    original_init = HostCacheExecutor.__init__
    original_write = HostCacheExecutor._start_writing
    original_load = HostCacheExecutor._start_loading
    original_snapshot = HostCacheExecutor._start_snapshot_store
    original_restore = HostCacheExecutor._start_restore
    original_poll = HostCacheExecutor.poll_results
    original_publish = EventLoop._publish_scheduler_kv_events

    @functools.wraps(original_init)
    def observed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._hw3_pending = {}
        capacity = {
            "rank": (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            ),
            "device_lcm_blocks": int(self.layout.num_lcm_blocks),
            "host_lcm_blocks": (
                int(self.host_storage.num_host_lcm_blocks) if self.host_storage else 0
            ),
            "host_bytes": (
                int(self.host_storage.host_buffer_bytes) if self.host_storage else 0
            ),
            "snapshot_lcm_blocks": (
                int(self.snapshot_storage.num_host_lcm_blocks)
                if self.snapshot_storage
                else 0
            ),
            "snapshot_bytes": (
                int(self.snapshot_storage.host_buffer_bytes)
                if self.snapshot_storage
                else 0
            ),
            "blob_bytes_per_request": int(self.blob_bytes),
            "blob_arena_bytes": (
                int(self.blob_arena.numel()) if self.blob_arena is not None else 0
            ),
            "max_retracted_requests": int(self.max_retracted_requests),
            "transfer_backend": self.transfer_backend,
        }
        (_root() / "cache-capacity.json").write_text(
            json.dumps(capacity, indent=2) + "\n"
        )

    def submitted(self, kind: str, op_ids, transfers, storage, *, blob_bytes: int = 0):
        op_ids = list(dict.fromkeys(int(op_id) for op_id in op_ids))
        if not op_ids:
            return
        started = time.perf_counter()
        for op_id in op_ids:
            self._hw3_pending[op_id] = (kind, started)
        _append(
            {
                "event": "submit",
                "kind": kind,
                "op_ids": op_ids,
                "blocks": len(transfers),
                "bytes": _transfer_bytes(storage, transfers) + int(blob_bytes),
            }
        )

    @functools.wraps(original_write)
    def observed_write(self, op_ids, transfers, **kwargs):
        result = original_write(self, op_ids, transfers, **kwargs)
        submitted(self, "write_back", op_ids, transfers, self.host_storage)
        return result

    @functools.wraps(original_load)
    def observed_load(self, op_ids, transfers, **kwargs):
        result = original_load(self, op_ids, transfers, **kwargs)
        submitted(self, "load_back", op_ids, transfers, self.host_storage)
        return result

    @functools.wraps(original_snapshot)
    def observed_snapshot(self, ops, transfers, **kwargs):
        result = original_snapshot(self, ops, transfers, **kwargs)
        submitted(
            self,
            "snapshot",
            (op.op_id for op in ops),
            transfers,
            self.snapshot_storage,
            blob_bytes=len(ops) * self.blob_bytes,
        )
        return result

    @functools.wraps(original_restore)
    def observed_restore(self, ops, rows_by_tier, **kwargs):
        result = original_restore(self, ops, rows_by_tier, **kwargs)
        l2_rows = rows_by_tier[HostTier.L2]
        pool_rows = rows_by_tier[HostTier.SNAPSHOT_POOL]
        submitted(
            self,
            "restore",
            (op.op_id for op in ops),
            l2_rows,
            self.host_storage,
            blob_bytes=_transfer_bytes(self.snapshot_storage, pool_rows)
            + len(ops) * self.blob_bytes,
        )
        return result

    @functools.wraps(original_poll)
    def observed_poll(self):
        results = original_poll(self)
        finished = time.perf_counter()
        for result in results:
            op_id = int(result.op_id)
            pending = self._hw3_pending.pop(op_id, None)
            if pending is not None:
                kind, started = pending
                _append(
                    {
                        "event": "complete",
                        "kind": kind,
                        "op_id": op_id,
                        "latency_ms": (finished - started) * 1000,
                    }
                )
        return results

    @functools.wraps(original_publish)
    def observed_publish(self, *args, **kwargs):
        result = original_publish(self, *args, **kwargs)
        scheduler = self.scheduler
        total = int(self._scheduler_cache_geometry.num_usable_pages)
        empty = int(scheduler.empty_lcm_blocks())
        active = int(scheduler.active_lcm_blocks())
        gauge = {
            "event": "gauge",
            "device_total_blocks": total,
            "device_empty_blocks": empty,
            "device_active_blocks": active,
            "device_cached_blocks": total - empty - active,
            "host_cached_blocks": int(scheduler.host_pool_cached_blocks()),
            "host_free_blocks": int(scheduler.host_pool_free_blocks()),
            "host_pinned_blocks": int(scheduler.host_pool_pinned_blocks()),
            "host_evictions": int(scheduler.host_pool_evictions()),
            "snapshot_free_blocks": int(scheduler.snapshot_pool_free_blocks()),
            "retracted_requests": int(scheduler.retracted_size()),
        }
        state = tuple(gauge.values())
        if state != getattr(self, "_hw3_last_cache_gauge", None):
            self._hw3_last_cache_gauge = state
            _append(gauge)
        return result

    observed_poll._hw3_probe = True
    HostCacheExecutor.__init__ = observed_init
    HostCacheExecutor._start_writing = observed_write
    HostCacheExecutor._start_loading = observed_load
    HostCacheExecutor._start_snapshot_store = observed_snapshot
    HostCacheExecutor._start_restore = observed_restore
    HostCacheExecutor.poll_results = observed_poll
    EventLoop._publish_scheduler_kv_events = observed_publish
