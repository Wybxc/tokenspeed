# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Compare homework-3 runs and enforce cache correctness invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _outputs(run: dict) -> dict[tuple[str, str], list[int]]:
    return {
        (wave["name"], request["session"]): request["output_ids"]
        for wave in run["waves"]
        for request in wave["requests"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--offload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-snapshot-restore",
        action="store_true",
        help="Require every TP rank to exercise balanced request-image snapshot/restore.",
    )
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    offload = json.loads(args.offload.read_text())
    baseline_outputs = _outputs(baseline)
    offload_outputs = _outputs(offload)
    assert baseline_outputs.keys() == offload_outputs.keys()
    mismatches = [
        key for key in baseline_outputs if baseline_outputs[key] != offload_outputs[key]
    ]
    ranks = offload["trace"]["ranks"]
    per_rank = []
    for rank in ranks:
        transfers = rank["transfers"]
        snapshot_ops = transfers.get("snapshot", {}).get("operations", 0)
        restore_ops = transfers.get("restore", {}).get("operations", 0)
        per_rank.append(
            {
                "rank": rank["rank"],
                "snapshot_operations": snapshot_ops,
                "restore_operations": restore_ops,
                "snapshot_restore_balanced": snapshot_ops == restore_ops,
                "host_evictions": rank["gauges"]["host_evictions"],
            }
        )
    result = {
        "compared_requests": len(baseline_outputs),
        "exact_output_match": not mismatches,
        "mismatches": [
            {"wave": wave, "session": session} for wave, session in mismatches
        ],
        "offload_ranks": per_rank,
        "all_ranks_exercised_snapshot_restore": all(
            rank["snapshot_operations"] > 0 and rank["restore_operations"] > 0
            for rank in per_rank
        ),
        "all_ranks_balanced": all(
            rank["snapshot_restore_balanced"] for rank in per_rank
        ),
    }
    assert result["exact_output_match"], result
    if args.require_snapshot_restore:
        assert result["all_ranks_exercised_snapshot_restore"], result
        assert result["all_ranks_balanced"], result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
