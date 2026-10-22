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

"""Reproduce complete-model correctness and four serial performance baselines.

Large logits/profiler files stay in --work-dir, outside the repository. The
optional observers synchronize only at whole-forward measurement boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from qwen35_probe import engine_kwargs, install
from transformers import AutoTokenizer

if os.environ.get("QWEN35_PROBE_DIR"):
    install()  # multiprocessing spawn executes this module in the worker too.


def prompt_ids(tokenizer, length):
    text = (
        "The research group studies efficient language model inference. "
        "A correct recurrent state must preserve information across tokens. "
        "Explain the result clearly and check the arithmetic. "
    )
    pattern = tokenizer.encode(text, add_special_tokens=False)
    return (pattern * (length // len(pattern) + 1))[:length]


def request(engine, root, name, ids, count, mode="capture", ignore_eos=True):
    if (root / name).exists():
        raise FileExistsError("Use a fresh --work-dir for each experiment run")
    control = root / "active.json"
    control.write_text(json.dumps(dict(id=name, mode=mode)))
    started = time.perf_counter()
    frames = []
    timestamps = []
    for frame in engine.generate(
        input_ids=ids,
        sampling_params=dict(
            temperature=0, max_new_tokens=count, ignore_eos=ignore_eos
        ),
        stream=True,
    ):
        frames.append(frame)
        timestamps.append(time.perf_counter())
    control.unlink()
    final = frames[-1]
    elapsed = timestamps[-1] - started
    steps = [
        json.loads(line)
        for line in (root / name / "steps.jsonl").read_text().splitlines()
    ]
    assert final["meta_info"]["completion_tokens"] == len(final["output_ids"])
    assert len(steps) == len(final["output_ids"])
    result = dict(
        id=name,
        input_tokens=len(ids),
        requested_output_tokens=count,
        output=final,
        wall_seconds=elapsed,
        ttft_ms=(timestamps[0] - started) * 1000,
        steps=steps,
    )
    print(
        json.dumps(
            {k: result[k] for k in ("id", "input_tokens", "wall_seconds", "ttft_ms")}
        ),
        flush=True,
    )
    return result


def compare_reference(args, captures, tokenizer):
    from safetensors import safe_open
    from transformers import Qwen3_5Config, Qwen3_5ForCausalLM

    torch.set_num_threads(8)
    config = Qwen3_5Config.from_pretrained(
        args.model, local_files_only=True
    ).text_config
    config._attn_implementation = "eager"
    # Construct only the complete text model. Check every weight, including tied head.
    # CPU construction also initializes nonpersistent RoPE frequency buffers.
    with torch.device("cpu"):
        reference = Qwen3_5ForCausalLM(config)
    weights = {}
    for path in Path(args.model).glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name.startswith("model.language_model."):
                    weights[name.replace("model.language_model.", "model.", 1)] = (
                        checkpoint.get_tensor(name).float()
                    )
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    reference.load_state_dict(weights, strict=True, assign=True)
    reference.eval()
    del weights
    layer_values = {}

    def observe_layer(index):
        def hook(module, inputs, output):
            layer_values[str(index)] = (
                output.detach().float().cpu().reshape(-1, config.hidden_size)
            )

        return hook

    for index, layer in enumerate(reference.model.layers):
        layer.register_forward_hook(observe_layer(index))
    comparisons = []
    with torch.inference_mode():
        for capture in captures:
            if capture["id"].endswith("-repeat"):
                continue  # Already required to be bitwise identical to the first run.
            paths = sorted(
                (Path(args.work_dir) / capture["id"]).glob("[0-9][0-9][0-9][0-9].pt")
            )
            cache = None
            errors, ref_argmax, actual_argmax = [], [], []
            for path in paths:
                sample = torch.load(path, weights_only=True)
                ids = sample["input_ids"].reshape(1, -1)
                expected = reference(
                    ids, past_key_values=cache, use_cache=True, logits_to_keep=1
                )
                cache = expected.past_key_values
                expected_logits = expected.logits[0, -1].float()
                assert torch.isfinite(expected_logits).all()
                actual = sample["logits"][0]
                layer_path = path.with_name("layers-" + path.name)
                if layer_path.exists():
                    actual_layers = torch.load(layer_path, weights_only=True)
                    layer_errors = []
                    for index in range(config.num_hidden_layers):
                        a, b = actual_layers[str(index)], layer_values[str(index)]
                        layer_errors.append(
                            dict(
                                layer=index,
                                rms=(a - b).square().mean().sqrt().item(),
                                reference_rms=b.square().mean().sqrt().item(),
                            )
                        )
                    path.with_name("layer-errors-" + path.stem + ".json").write_text(
                        json.dumps(layer_errors)
                    )
                difference = (actual - expected_logits).abs()
                # A priori full-network bounds: RMS <= 0.10 logit,
                # max <= 0.75; cosine >= .999. Module tests use tighter bounds.
                rms = difference.square().mean().sqrt().item()
                maximum = difference.max().item()
                cosine = torch.nn.functional.cosine_similarity(
                    actual, expected_logits, dim=0
                ).item()
                top = expected_logits.topk(2)
                errors.append(
                    dict(
                        step=len(errors),
                        max_abs=maximum,
                        rms=rms,
                        p99=difference.quantile(0.99).item(),
                        cosine=cosine,
                        mean_signed=(actual - expected_logits).mean().item(),
                        reference_rms=expected_logits.square().mean().sqrt().item(),
                        top1_regret=(
                            expected_logits.max() - expected_logits[actual.argmax()]
                        ).item(),
                        stabilized_relative_p99=(
                            difference / expected_logits.abs().clamp_min(1)
                        )
                        .quantile(0.99)
                        .item(),
                        reference_top_margin=(top.values[0] - top.values[1]).item(),
                    )
                )
                ref_argmax.append(int(expected_logits.argmax()))
                actual_argmax.append(int(actual.argmax()))
                print(capture["id"], errors[-1], flush=True)
                if not getattr(args, "diagnostic", False):
                    assert rms <= 0.10 and maximum <= 0.75 and cosine >= 0.999, errors[
                        -1
                    ]
            comparisons.append(
                dict(
                    id=capture["id"],
                    steps=len(errors),
                    errors=errors,
                    greedy_agreement=sum(
                        a == b for a, b in zip(ref_argmax, actual_argmax)
                    )
                    / len(errors),
                    reference_argmax=ref_argmax,
                    actual_argmax=actual_argmax,
                )
            )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["validate", "compare", "benchmark", "profile"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Collect errors without claiming numerical validation",
    )
    args = parser.parse_args()
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "active.json").unlink(missing_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.mode == "compare":
        captures = json.loads((root / "captures.json").read_text())
        comparisons = compare_reference(args, captures, tokenizer)
        Path(args.output).write_text(
            json.dumps(
                dict(
                    reference="Transformers 5.12.0 CPU FP32, eager attention",
                    validated=not args.diagnostic,
                    thresholds=dict(rms_max=0.10, max_abs=0.75, cosine_min=0.999),
                    comparisons=comparisons,
                ),
                indent=2,
            )
            + "\n"
        )
        return
    os.environ["QWEN35_PROBE_DIR"] = str(root)
    install()
    from tokenspeed.runtime.entrypoints.engine import Engine

    captures = []
    with Engine(**engine_kwargs(args.model)) as engine:
        if args.mode == "validate":
            for length in (1, 128, 2048):
                captures.append(
                    request(
                        engine,
                        root,
                        f"length-{length}",
                        prompt_ids(tokenizer, length),
                        32,
                    )
                )
            # Reusing the same slot after unrelated histories must reproduce B exactly.
            repeated = request(
                engine, root, "length-128-repeat", prompt_ids(tokenizer, 128), 32
            )
            for first, second in zip(
                sorted((root / "length-128").glob("[0-9][0-9][0-9][0-9].pt")),
                sorted((root / "length-128-repeat").glob("[0-9][0-9][0-9][0-9].pt")),
            ):
                torch.testing.assert_close(
                    torch.load(first, weights_only=True)["logits"],
                    torch.load(second, weights_only=True)["logits"],
                    rtol=0,
                    atol=0,
                )
            captures.append(repeated)
            repeated = request(
                engine, root, "length-1-repeat", prompt_ids(tokenizer, 1), 32
            )
            for first, second in zip(
                sorted((root / "length-1").glob("[0-9][0-9][0-9][0-9].pt")),
                sorted((root / "length-1-repeat").glob("[0-9][0-9][0-9][0-9].pt")),
            ):
                torch.testing.assert_close(
                    torch.load(first, weights_only=True)["logits"],
                    torch.load(second, weights_only=True)["logits"],
                    rtol=0,
                    atol=0,
                )
            captures.append(repeated)
            for i, text in enumerate(
                ("用中文解释什么是递推状态。", "What is 17 + 25? Answer briefly.")
            ):
                ids = tokenizer.apply_chat_template(
                    [dict(role="user", content=text)],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
                captures.append(
                    request(engine, root, f"chat-{i}", ids, 64, ignore_eos=False)
                )
            assert captures[-1]["output"]["text"].strip() == "42"
            assert (
                captures[-1]["output"]["meta_info"]["finish_reason"]["type"] == "stop"
            )
            (root / "captures.json").write_text(json.dumps(captures, indent=2) + "\n")
            output = dict(
                weight_audit=json.loads((root / "weight-audit.json").read_text()),
                exact_request_isolation=True,
                captures=captures,
            )
        elif args.mode == "profile":
            output = [
                request(
                    engine,
                    root,
                    f"profile-{length}",
                    prompt_ids(tokenizer, length),
                    4,
                    mode="profile",
                )
                for length in (128, 2048)
            ]
        else:
            trials = []
            for length in (128, 2048):
                for count in (32, 128):
                    for repeat in range(13):
                        trial = request(
                            engine,
                            root,
                            f"bench-{length}-{count}-{repeat}",
                            prompt_ids(tokenizer, length),
                            count,
                            mode="timing",
                        )
                        assert len(trial["steps"]) == count, trial
                        trial["warmup"] = repeat < 3
                        trial["tpot_ms"] = (
                            trial["wall_seconds"] * 1000 - trial["ttft_ms"]
                        ) / (count - 1)
                        trials.append(trial)
            summaries = []
            for length in (128, 2048):
                for count in (32, 128):
                    group = [
                        t
                        for t in trials
                        if t["input_tokens"] == length
                        and t["requested_output_tokens"] == count
                        and not t["warmup"]
                    ]
                    metrics = {}
                    for key in ("ttft_ms", "tpot_ms", "wall_seconds"):
                        values = [t[key] for t in group]
                        metrics[key] = dict(
                            median=statistics.median(values),
                            min=min(values),
                            max=max(values),
                            stdev=statistics.stdev(values),
                        )
                    metrics["prefill_forward_ms"] = statistics.median(
                        t["steps"][0]["wall_ms"] for t in group
                    )
                    metrics["decode_forward_ms"] = statistics.median(
                        s["wall_ms"] for t in group for s in t["steps"][1:]
                    )
                    metrics["peak_allocated_bytes"] = max(
                        s["peak_allocated"] for t in group for s in t["steps"]
                    )
                    metrics["peak_reserved_bytes"] = max(
                        s["peak_reserved"] for t in group for s in t["steps"]
                    )
                    summaries.append(
                        dict(input_tokens=length, output_tokens=count, metrics=metrics)
                    )
            output = dict(
                config=engine_kwargs(args.model),
                warmups=3,
                repeats=10,
                weight_audit=json.loads((root / "weight-audit.json").read_text()),
                summaries=summaries,
                trials=trials,
            )
    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
