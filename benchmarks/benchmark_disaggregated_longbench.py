#!/usr/bin/env python3
"""LongBench-v2 prefill/decode-disaggregation benchmark.

The script launches separate NIXL-connected prefill and decode vLLM servers.
Available GPUs are split evenly; when the count is odd, decode receives the
extra GPU. Only decode-server counters and decode-side latency are measured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from benchmark_vegas import build_longbench_prompt
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vegas_online_benchmark_utils import (
    ManagedServer,
    RequestSample,
    compare_runs,
    make_payload,
    make_server_command,
    run_online_workload,
    speculative_config,
    stream_completion,
    write_json,
)

DEFAULT_ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AR with Vegas/StreamingLLM on the decode side "
        "of a NIXL-disaggregated LongBench-v2 deployment."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--algorithm", choices=("vegas", "streamingllm"), default="vegas"
    )
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--min-input-tokens", type=int, default=96 * 1024)
    parser.add_argument("--max-input-tokens", type=int, default=120 * 1024)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="LongBench output cap; inputs remain the dominant 96K-120K load.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="Decode maximum batch (paper default for Qwen3-4B/8B: 4).",
    )
    parser.add_argument(
        "--prefill-max-num-seqs",
        type=int,
        default=4,
        help="Maximum concurrent sequences on the prefill server.",
    )
    parser.add_argument("--warmup-completions", type=int, default=None)
    parser.add_argument("--tail-completions", type=int, default=None)
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=None,
        help="Draft length gamma (Qwen3-8B paper default: Vegas=9, StreamingLLM=5).",
    )
    parser.add_argument("--sparse-attn-ratio", type=float, default=0.07)
    parser.add_argument(
        "--sparse-attn-min-tokens",
        type=int,
        default=256,
        help="Minimum KV tokens kept by sparse draft attention.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU IDs. Defaults to CUDA_VISIBLE_DEVICES, then "
        "all GPUs reported by nvidia-smi.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--prefill-port", type=int, default=8100)
    parser.add_argument("--decode-port", type=int, default=8200)
    parser.add_argument("--prefill-side-channel-port", type=int, default=5600)
    parser.add_argument("--decode-side-channel-port", type=int, default=5700)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--metrics-poll-interval", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rope-scaling-json",
        default=json.dumps(DEFAULT_ROPE_SCALING),
        help="RoPE scaling JSON. Pass an empty string to disable it.",
    )
    parser.add_argument(
        "--show-progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--server-extra-arg",
        action="append",
        default=[],
        help="Additional single argument passed to both vLLM servers; repeatable.",
    )
    return parser.parse_args()


def discover_gpus(argument: str | None) -> list[str]:
    if argument:
        return [gpu.strip() for gpu in argument.split(",") if gpu.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Unable to discover GPUs; specify at least two with --gpus"
        ) from exc
    return [line.strip() for line in output.splitlines() if line.strip()]


def validate_args(args: argparse.Namespace, gpus: list[str]) -> None:
    for name in (
        "num_requests",
        "min_input_tokens",
        "max_input_tokens",
        "max_model_len",
        "max_tokens",
        "max_num_seqs",
        "prefill_max_num_seqs",
        "num_speculative_tokens",
        "sparse_attn_min_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if len(gpus) < 2:
        raise ValueError(
            f"Disaggregated prefill/decode requires at least 2 GPUs; found {gpus}"
        )
    if len(set(gpus)) != len(gpus):
        raise ValueError("--gpus contains duplicate GPU IDs")
    if args.min_input_tokens > args.max_input_tokens:
        raise ValueError("--min-input-tokens cannot exceed --max-input-tokens")
    if args.max_input_tokens + args.max_tokens > args.max_model_len:
        raise ValueError(
            "--max-input-tokens + --max-tokens must not exceed --max-model-len"
        )
    if not 0 < args.sparse_attn_ratio <= 1:
        raise ValueError("--sparse-attn-ratio must be in (0, 1]")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.prefill_port == args.decode_port:
        raise ValueError("Prefill and decode ports must differ")
    if args.rope_scaling_json:
        value = json.loads(args.rope_scaling_json)
        if not isinstance(value, dict):
            raise ValueError("--rope-scaling-json must decode to a JSON object")


def prepare_longbench_prompts(
    args: argparse.Namespace,
) -> tuple[list[str], list[int], list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rng = random.Random(args.seed)
    targets = [
        rng.randint(args.min_input_tokens, args.max_input_tokens)
        for _ in range(args.num_requests)
    ]
    dataset = load_dataset("THUDM/LongBench-v2", split="train", streaming=True)
    prompts: list[str] = []
    token_counts: list[int] = []
    used_targets: list[int] = []
    progress = tqdm(
        total=args.num_requests,
        desc="prepare-longbench-v2",
        unit="prompt",
        disable=not args.show_progress,
    )
    try:
        for item in dataset:
            target = targets[len(prompts)]
            try:
                prompt, prompt_tokens = build_longbench_prompt(
                    tokenizer,
                    item,
                    target,
                    args.enable_thinking,
                )
            except ValueError:
                continue
            if prompt_tokens < args.min_input_tokens:
                continue
            prompts.append(prompt)
            token_counts.append(prompt_tokens)
            used_targets.append(target)
            progress.update()
            progress.set_postfix(input_tokens=f"{prompt_tokens:,}")
            if len(prompts) == args.num_requests:
                break
    finally:
        progress.close()
    if len(prompts) != args.num_requests:
        raise ValueError(
            f"LongBench-v2 supplied only {len(prompts)} prompts in the requested "
            f"{args.min_input_tokens:,}-{args.max_input_tokens:,} token range"
        )
    return prompts, token_counts, used_targets


async def disaggregated_request_runner(
    session: aiohttp.ClientSession,
    index: int,
    _prompt: str,
    prompt_tokens: int,
    payload: dict[str, Any],
) -> RequestSample:
    prefill_url = payload.pop("_prefill_url")
    decode_url = payload.pop("_decode_url")
    request_id = uuid.uuid4().hex
    headers = {"X-Request-Id": request_id}
    prefill_payload = dict(payload)
    prefill_payload.update(
        {
            "max_tokens": 1,
            "stream": False,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            },
        }
    )
    prefill_payload.pop("stream_options", None)
    try:
        async with session.post(
            prefill_url, json=prefill_payload, headers=headers
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"prefill HTTP {response.status}: {body[:1000]}")
            prefill_response = await response.json()
        kv_transfer_params = prefill_response.get("kv_transfer_params")
        if not kv_transfer_params:
            raise RuntimeError("prefill response did not contain kv_transfer_params")
        payload["kv_transfer_params"] = kv_transfer_params
    except Exception as exc:
        now = time.perf_counter()
        return RequestSample(
            index=index,
            prompt_tokens=prompt_tokens,
            output_tokens=0,
            start_time=now,
            first_token_time=None,
            end_time=now,
            error=f"{type(exc).__name__}: {exc}",
        )

    # stream_completion starts its timer here. Consequently TTFT, decode
    # latency, E2E latency, and throughput are all decode-side measurements;
    # the potentially minutes-long remote prefill is deliberately excluded.
    return await stream_completion(
        session,
        url=decode_url,
        payload=payload,
        index=index,
        prompt_tokens=prompt_tokens,
        headers=headers,
    )


def run_one(
    args: argparse.Namespace,
    *,
    mode: str,
    prompts: list[str],
    token_counts: list[int],
    prefill_gpus: list[str],
    decode_gpus: list[str],
    warmup_completions: int,
    tail_completions: int,
    output_dir: Path,
) -> dict[str, Any]:
    rope_scaling = (
        json.loads(args.rope_scaling_json) if args.rope_scaling_json else None
    )
    kv_config = {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
    }
    prefill_command = make_server_command(
        model=args.model,
        port=args.prefill_port,
        tensor_parallel_size=len(prefill_gpus),
        max_model_len=args.max_model_len,
        max_num_seqs=args.prefill_max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        speculative_config=None,
        kv_transfer_config=kv_config,
        rope_scaling=rope_scaling,
        extra_args=args.server_extra_arg,
    )
    decode_spec = (
        None
        if mode == "ar"
        else speculative_config(
            args.algorithm,
            args.num_speculative_tokens,
            args.sparse_attn_ratio,
            args.sparse_attn_min_tokens,
        )
    )
    decode_command = make_server_command(
        model=args.model,
        port=args.decode_port,
        tensor_parallel_size=len(decode_gpus),
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        speculative_config=decode_spec,
        kv_transfer_config=kv_config,
        rope_scaling=rope_scaling,
        extra_args=args.server_extra_arg,
    )
    common_environment = {"UCX_NET_DEVICES": "all"}
    prefill_server = ManagedServer(
        command=prefill_command,
        gpus=prefill_gpus,
        port=args.prefill_port,
        log_path=output_dir / f"prefill_{mode}.log",
        startup_timeout=args.startup_timeout,
        environment=common_environment
        | {"VLLM_NIXL_SIDE_CHANNEL_PORT": str(args.prefill_side_channel_port)},
    )
    decode_server = ManagedServer(
        command=decode_command,
        gpus=decode_gpus,
        port=args.decode_port,
        log_path=output_dir / f"decode_{mode}.log",
        startup_timeout=args.startup_timeout,
        environment=common_environment
        | {"VLLM_NIXL_SIDE_CHANNEL_PORT": str(args.decode_side_channel_port)},
    )
    print(
        f"\nStarting {mode}: prefill GPUs={','.join(prefill_gpus)}, "
        f"decode GPUs={','.join(decode_gpus)}"
    )
    with prefill_server, decode_server:
        payload = make_payload(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            seed=args.seed,
            ignore_eos=args.ignore_eos,
        )
        payload["_prefill_url"] = f"http://127.0.0.1:{args.prefill_port}/v1/completions"
        payload["_decode_url"] = f"http://127.0.0.1:{args.decode_port}/v1/completions"
        run = asyncio.run(
            run_online_workload(
                prompts=prompts,
                prompt_token_counts=token_counts,
                payload_template=payload,
                request_runner=disaggregated_request_runner,
                metrics_url=f"http://127.0.0.1:{args.decode_port}/metrics",
                warmup_completions=warmup_completions,
                tail_completions=tail_completions,
                poll_interval=args.metrics_poll_interval,
                description=f"decode-{mode}",
                show_progress=args.show_progress,
            )
        )
    write_json(output_dir / f"{mode}.json", run)
    return run


def main() -> None:
    args = parse_args()
    if args.num_speculative_tokens is None:
        args.num_speculative_tokens = 9 if args.algorithm == "vegas" else 5
    gpus = discover_gpus(args.gpus)
    validate_args(args, gpus)
    split = len(gpus) // 2
    prefill_gpus = gpus[:split]
    decode_gpus = gpus[split:]
    warmup_completions = (
        args.warmup_completions
        if args.warmup_completions is not None
        else args.max_num_seqs
    )
    tail_completions = (
        args.tail_completions
        if args.tail_completions is not None
        else args.max_num_seqs
    )
    if warmup_completions <= 0 or tail_completions < 0:
        raise ValueError("warmup must be positive and tail must be non-negative")
    if warmup_completions + tail_completions >= args.num_requests:
        raise ValueError(
            "warmup + tail completions must leave a non-empty measurement window"
        )

    output_dir = args.output_dir or Path("benchmark_results") / (
        f"disaggregated_longbench_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts, token_counts, targets = prepare_longbench_prompts(args)
    print(
        f"Prepared {len(prompts)} LongBench-v2 requests; actual input tokens "
        f"mean={statistics.mean(token_counts):,.1f}, "
        f"range={min(token_counts):,}-{max(token_counts):,}"
    )
    print(
        f"GPU split: prefill={prefill_gpus} ({len(prefill_gpus)}), "
        f"decode={decode_gpus} ({len(decode_gpus)}); only decode is measured"
    )

    ar = run_one(
        args,
        mode="ar",
        prompts=prompts,
        token_counts=token_counts,
        prefill_gpus=prefill_gpus,
        decode_gpus=decode_gpus,
        warmup_completions=warmup_completions,
        tail_completions=tail_completions,
        output_dir=output_dir,
    )
    spec = run_one(
        args,
        mode=args.algorithm,
        prompts=prompts,
        token_counts=token_counts,
        prefill_gpus=prefill_gpus,
        decode_gpus=decode_gpus,
        warmup_completions=warmup_completions,
        tail_completions=tail_completions,
        output_dir=output_dir,
    )
    comparison = compare_runs(ar, spec, args.algorithm, args.num_speculative_tokens)
    result = {
        "configuration": vars(args) | {"output_dir": str(output_dir)},
        "gpu_split": {"prefill": prefill_gpus, "decode": decode_gpus},
        "workload": {
            "dataset": "longbench-v2",
            "requests": len(prompts),
            "target_input_tokens": targets,
            "actual_input_tokens_mean": statistics.mean(token_counts),
            "actual_input_tokens_min": min(token_counts),
            "actual_input_tokens_max": max(token_counts),
            "decode_only_measurement": True,
            "warmup_completions": warmup_completions,
            "tail_completions": tail_completions,
        },
        "comparison": comparison,
    }
    write_json(output_dir / "comparison.json", result)
    print(f"Results written to {output_dir.resolve()}")
    headline = {
        key: value for key, value in comparison.items() if key not in {"ar", "spec"}
    }
    print(f"RESULT_JSON {json.dumps(headline, sort_keys=True)}")


if __name__ == "__main__":
    main()
