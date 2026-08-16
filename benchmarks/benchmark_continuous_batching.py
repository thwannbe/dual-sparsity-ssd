#!/usr/bin/env python3
"""Paper-style continuous-batching benchmark for AIME'25 and CodeElo.

All requests are submitted concurrently. The stable window starts after a
dataset-specific number of completions and ends before the final scheduler
drain, so throughput is measured while continuous batching remains saturated.
AR and sparse self-speculative decoding run sequentially on the same workload.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmark_vegas import apply_chat_template
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vegas_online_benchmark_utils import (
    ManagedServer,
    compare_runs,
    make_payload,
    make_server_command,
    run_online_workload,
    speculative_config,
    stream_completion,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AR with Vegas/StreamingLLM under saturated "
        "continuous batching on long-reasoning workloads."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", choices=("aime25", "codeelo"), default="aime25")
    parser.add_argument(
        "--algorithm", choices=("vegas", "streamingllm"), default="vegas"
    )
    parser.add_argument(
        "--aime-repeats",
        type=int,
        default=32,
        help="Replicate all 30 AIME questions this many times (paper: 32).",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Optional workload limit. By default AIME uses 30*repeats and "
        "CodeElo uses every row.",
    )
    parser.add_argument(
        "--warmup-completions",
        type=int,
        default=None,
        help="Completions excluded before measurement (AIME default: 400; "
        "CodeElo default: one max batch).",
    )
    parser.add_argument(
        "--tail-completions",
        type=int,
        default=None,
        help="Final completions excluded to avoid scheduler-drain bias "
        "(default: --max-num-seqs).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=37632,
        help="Context capacity. The default preserves the 32K generation cap "
        "while fitting Qwen3-8B on a 24 GiB GPU.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Long generation cap matching Qwen3's native reasoning budget.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=96,
        help="Maximum active requests. The default saturates decoding while "
        "leaving sampler workspace on a 24 GiB GPU.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=None,
        help="Draft length gamma (Qwen3-8B paper default: Vegas=6, StreamingLLM=4).",
    )
    parser.add_argument("--sparse-attn-ratio", type=float, default=0.07)
    parser.add_argument(
        "--sparse-attn-min-tokens",
        type=int,
        default=256,
        help="Minimum KV tokens kept by sparse draft attention. Lower this "
        "for short generations where the default 256-token floor would "
        "otherwise dominate --sparse-attn-ratio.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force every request to max tokens. EOS is respected by default.",
    )
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU IDs used by each sequential server run.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--metrics-poll-interval", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show-progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Results directory (default: timestamped benchmark_results path).",
    )
    parser.add_argument(
        "--server-extra-arg",
        action="append",
        default=[],
        help="Additional single argument passed to both vLLM servers; repeat "
        "this option for multiple arguments.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--aime-repeats": args.aime_repeats,
        "--max-model-len": args.max_model_len,
        "--max-tokens": args.max_tokens,
        "--max-num-seqs": args.max_num_seqs,
        "--num-speculative-tokens": args.num_speculative_tokens,
        "--sparse-attn-min-tokens": args.sparse_attn_min_tokens,
        "--tensor-parallel-size": args.tensor_parallel_size,
        "--metrics-poll-interval": args.metrics_poll_interval,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_requests is not None and args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if args.max_tokens >= args.max_model_len:
        raise ValueError("--max-tokens must be smaller than --max-model-len")
    if not 0 < args.sparse_attn_ratio <= 1:
        raise ValueError("--sparse-attn-ratio must be in (0, 1]")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if len(gpus) != args.tensor_parallel_size:
        raise ValueError("--gpus must contain exactly --tensor-parallel-size GPU IDs")


def format_codeelo(item: Mapping[str, Any]) -> str:
    examples = []
    for example in item.get("examples", []):
        if len(example) >= 2:
            examples.append(f"Input:\n{example[0]}\n\nExpected output:\n{example[1]}")
    sections = [
        (
            "Solve the following competitive-programming problem. Explain your "
            "reasoning and provide a complete GNU C++17 solution."
        ),
        f"Title: {item.get('title', item.get('problem_id', ''))}",
        f"Problem:\n{item['description']}",
    ]
    for title, key in (
        ("Input", "input"),
        ("Output", "output"),
        ("Interaction", "interaction"),
        ("Note", "note"),
    ):
        value = item.get(key)
        if value:
            sections.append(f"{title}:\n{value}")
    if examples:
        sections.append("Examples:\n" + "\n\n".join(examples))
    return "\n\n".join(sections)


def prepare_prompts(
    args: argparse.Namespace,
) -> tuple[list[str], list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.dataset == "aime25":
        questions = list(load_dataset("math-ai/aime25", split="test")["problem"])
        raw_prompts = questions * args.aime_repeats
    else:
        dataset = load_dataset("Qwen/CodeElo", split="test")
        raw_prompts = [format_codeelo(item) for item in dataset]
    if args.num_requests is not None:
        if args.num_requests > len(raw_prompts):
            repeats = math.ceil(args.num_requests / len(raw_prompts))
            raw_prompts = (raw_prompts * repeats)[: args.num_requests]
        else:
            raw_prompts = raw_prompts[: args.num_requests]

    prompts = []
    token_counts = []
    for raw_prompt in tqdm(
        raw_prompts,
        desc=f"prepare-{args.dataset}",
        unit="prompt",
        disable=not args.show_progress,
    ):
        prompt = apply_chat_template(tokenizer, raw_prompt, args.enable_thinking)
        prompts.append(prompt)
        token_counts.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
    if max(token_counts) + args.max_tokens > args.max_model_len:
        raise ValueError(
            f"Longest prompt ({max(token_counts)} tokens) + --max-tokens "
            f"({args.max_tokens}) exceeds --max-model-len ({args.max_model_len})"
        )
    return prompts, token_counts


async def direct_request_runner(
    session: Any,
    index: int,
    _prompt: str,
    prompt_tokens: int,
    payload: dict[str, Any],
) -> Any:
    return await stream_completion(
        session,
        url=payload.pop("_url"),
        payload=payload,
        index=index,
        prompt_tokens=prompt_tokens,
    )


def run_one(
    args: argparse.Namespace,
    *,
    mode: str,
    prompts: list[str],
    token_counts: list[int],
    warmup_completions: int,
    tail_completions: int,
    output_dir: Path,
) -> dict[str, Any]:
    spec = (
        None
        if mode == "ar"
        else speculative_config(
            args.algorithm,
            args.num_speculative_tokens,
            args.sparse_attn_ratio,
            args.sparse_attn_min_tokens,
        )
    )
    command = make_server_command(
        model=args.model,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        speculative_config=spec,
        extra_args=args.server_extra_arg,
    )
    server = ManagedServer(
        command=command,
        gpus=[gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()],
        port=args.port,
        log_path=output_dir / f"server_{mode}.log",
        startup_timeout=args.startup_timeout,
    )
    print(f"\nStarting {mode} server on GPU(s) {args.gpus} ...")
    with server:
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
        payload["_url"] = f"http://127.0.0.1:{args.port}/v1/completions"
        run = asyncio.run(
            run_online_workload(
                prompts=prompts,
                prompt_token_counts=token_counts,
                payload_template=payload,
                request_runner=direct_request_runner,
                metrics_url=f"http://127.0.0.1:{args.port}/metrics",
                warmup_completions=warmup_completions,
                tail_completions=tail_completions,
                poll_interval=args.metrics_poll_interval,
                description=mode,
                show_progress=args.show_progress,
            )
        )
    write_json(output_dir / f"{mode}.json", run)
    return run


def main() -> None:
    args = parse_args()
    if args.num_speculative_tokens is None:
        args.num_speculative_tokens = 6 if args.algorithm == "vegas" else 4
    validate_args(args)
    output_dir = args.output_dir or Path("benchmark_results") / (
        f"continuous_batching_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts, token_counts = prepare_prompts(args)
    warmup_completions = (
        args.warmup_completions
        if args.warmup_completions is not None
        else 400
        if args.dataset == "aime25"
        else args.max_num_seqs
    )
    tail_completions = (
        args.tail_completions
        if args.tail_completions is not None
        else args.max_num_seqs
    )
    if warmup_completions <= 0:
        raise ValueError("--warmup-completions must be positive")
    if tail_completions < 0:
        raise ValueError("--tail-completions cannot be negative")
    if warmup_completions + tail_completions >= len(prompts):
        raise ValueError(
            "warmup + tail completions must leave a non-empty stable window; "
            f"got {warmup_completions} + {tail_completions} for {len(prompts)} requests"
        )

    print(
        f"Prepared {len(prompts)} {args.dataset} requests; input tokens/request "
        f"mean={statistics.mean(token_counts):,.1f}, "
        f"range={min(token_counts):,}-{max(token_counts):,}"
    )
    print(
        f"Stable window: after completion {warmup_completions} through "
        f"{len(prompts) - tail_completions}; max_num_seqs={args.max_num_seqs}"
    )

    ar = run_one(
        args,
        mode="ar",
        prompts=prompts,
        token_counts=token_counts,
        warmup_completions=warmup_completions,
        tail_completions=tail_completions,
        output_dir=output_dir,
    )
    spec = run_one(
        args,
        mode=args.algorithm,
        prompts=prompts,
        token_counts=token_counts,
        warmup_completions=warmup_completions,
        tail_completions=tail_completions,
        output_dir=output_dir,
    )
    comparison = compare_runs(ar, spec, args.algorithm, args.num_speculative_tokens)
    result = {
        "configuration": vars(args) | {"output_dir": str(output_dir)},
        "workload": {
            "requests": len(prompts),
            "input_tokens_mean": statistics.mean(token_counts),
            "input_tokens_min": min(token_counts),
            "input_tokens_max": max(token_counts),
            "continuous_batching": True,
            "all_requests_submitted_concurrently": True,
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
