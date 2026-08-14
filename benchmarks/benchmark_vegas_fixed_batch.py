"""Compare AR and sparse speculative decoding over multiple closed batches.

Each group of ``batch_size`` requests is admitted together and allowed to
finish before the next group is submitted. There are no queued requests to
refill a freed slot within a measured batch. EOS is respected, so the active
batch can naturally shrink as requests finish.
"""

import argparse
import gc
import json
import os
import statistics
import time
from functools import partial
from typing import Any

import torch
from benchmark_vegas import (
    mean_gpu_latency_ms,
    prepare_aime_prompts,
    prepare_longbench_prompts,
    read_spec_decode_stats,
    subtract_spec_stats,
    summarize_common_metrics,
)
from tqdm import tqdm
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an AR baseline and Vegas/StreamingLLM sequentially "
        "over the same closed batches, with EOS respected and no request refill."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--dataset",
        choices=("aime25", "longbench-v2"),
        default="aime25",
    )
    parser.add_argument(
        "--longbench-length",
        choices=("short", "medium", "long", "any"),
        default="long",
    )
    parser.add_argument(
        "--target-input-tokens",
        type=int,
        default=None,
        help="Target LongBench input length. The default is max model length "
        "minus max output tokens.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("vegas", "streamingllm"),
        default="vegas",
        help="Sparse speculative algorithm compared with AR.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of requests admitted in each closed batch.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Total measured requests. Must be a multiple of --batch-size; "
        "each group finishes before the next group is admitted.",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. The default 0 uses greedy decoding.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=16,
        help="Untimed tokens generated on a held-out closed batch for each engine.",
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=6)
    parser.add_argument("--sparse-attn-ratio", type=float, default=0.07)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show outer closed-batch progress and inner per-request progress.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.max_tokens >= args.max_model_len:
        raise ValueError("--max-tokens must be smaller than --max-model-len")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if args.num_samples % args.batch_size != 0:
        raise ValueError("--num-samples must be a multiple of --batch-size")
    if args.warmup_tokens < 0:
        raise ValueError("--warmup-tokens cannot be negative")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if args.target_input_tokens is not None and args.target_input_tokens < 1:
        raise ValueError("--target-input-tokens must be positive")


def prepare_prompts(
    args: argparse.Namespace,
) -> tuple[list[str], list[int], list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    warmup_count = args.batch_size if args.warmup_tokens else 0
    required_count = args.num_samples + warmup_count

    if args.dataset == "longbench-v2":
        maximum_input_tokens = args.max_model_len - args.max_tokens
        target_input_tokens = args.target_input_tokens or maximum_input_tokens
        if target_input_tokens > maximum_input_tokens:
            raise ValueError(
                "--target-input-tokens plus --max-tokens must not exceed "
                "--max-model-len"
            )
        prepared = prepare_longbench_prompts(
            tokenizer,
            required_count,
            target_input_tokens,
            args.longbench_length,
            args.enable_thinking,
        )
    else:
        prepared = prepare_aime_prompts(
            tokenizer,
            required_count,
            args.enable_thinking,
        )

    measured = prepared[: args.num_samples]
    warmup = prepared[args.num_samples :]
    return (
        [prompt for prompt, _ in measured],
        [count for _, count in measured],
        [prompt for prompt, _ in warmup],
    )


def make_speculative_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "sparse_attn",
        "num_speculative_tokens": args.num_speculative_tokens,
        "sparse_attn_algorithm": args.algorithm,
        "sparse_attn_ratio": args.sparse_attn_ratio,
    }


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def run_closed_batches(
    args: argparse.Namespace,
    algorithm: str,
    prompts: list[str],
    warmup_prompts: list[str],
) -> tuple[
    dict[str, Any],
    list[Any],
    dict[str, int | list[int]] | None,
    list[float],
    list[int],
]:
    speculative_config = None if algorithm == "ar" else make_speculative_config(args)
    llm = LLM(
        model=args.model,
        max_num_seqs=args.batch_size,
        max_model_len=args.max_model_len,
        seed=args.seed,
        speculative_config=speculative_config,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        disable_log_stats=False,
        # Sparse-attention speculation does not support async scheduling.
        # Disable it for AR as well so the comparison uses the same scheduler
        # execution mode and reduces one source of numerical-path differences.
        async_scheduling=False,
    )

    # EOS is intentionally respected in warmup and measurement. Each generate
    # call contains exactly batch_size requests and is allowed to finish before
    # the next call, so no waiting request can refill a freed slot.
    if args.warmup_tokens:
        warmup_params = SamplingParams(
            max_tokens=min(args.warmup_tokens, args.max_model_len - 1),
            temperature=args.temperature,
            ignore_eos=False,
        )
        print(f"  warming up {algorithm}: {len(warmup_prompts)} requests")
        warmup_tqdm = partial(tqdm, position=1, leave=False)
        llm.generate(
            warmup_prompts,
            warmup_params,
            use_tqdm=warmup_tqdm if args.show_progress else False,
        )

    stats_before = (
        read_spec_decode_stats(llm, args.num_speculative_tokens)
        if speculative_config is not None
        else None
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        ignore_eos=False,
    )

    prompt_batches = chunked(prompts, args.batch_size)
    outputs = []
    batch_elapsed_seconds = []
    batch_output_tokens = []
    batch_progress = tqdm(
        prompt_batches,
        desc=f"{algorithm} sample groups",
        unit="batch",
        position=0,
        disable=not args.show_progress,
    )
    inner_tqdm = partial(tqdm, position=1, leave=False)
    for batch_index, prompt_batch in enumerate(batch_progress, start=1):
        started = time.perf_counter()
        batch_outputs = llm.generate(
            prompt_batch,
            sampling_params,
            use_tqdm=inner_tqdm if args.show_progress else False,
        )
        batch_elapsed = time.perf_counter() - started
        outputs.extend(batch_outputs)
        batch_elapsed_seconds.append(batch_elapsed)
        batch_output_tokens.append(
            sum(
                len(completion.token_ids)
                for request in batch_outputs
                for completion in request.outputs
            )
        )
        batch_progress.set_postfix(
            batch=f"{batch_index}/{len(prompt_batches)}",
            latency=f"{batch_elapsed:.3f}s",
            output_tokens=batch_output_tokens[-1],
        )

    total_elapsed = sum(batch_elapsed_seconds)
    common = summarize_common_metrics(outputs, total_elapsed)

    spec_stats = None
    if stats_before is not None:
        stats_after = read_spec_decode_stats(llm, args.num_speculative_tokens)
        spec_stats = subtract_spec_stats(stats_after, stats_before)

    # Drop the engine inside this scope (rather than in a helper that would
    # only delete its own reference), then reset process-global distributed
    # state before constructing the next engine.
    del llm
    gc.collect()
    cleanup_dist_env_and_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        common,
        outputs,
        spec_stats,
        batch_elapsed_seconds,
        batch_output_tokens,
    )


def output_token_ids(outputs: list[Any]) -> list[list[int]]:
    return [
        [token for completion in request.outputs for token in completion.token_ids]
        for request in outputs
    ]


def diagnose_output_mismatches(
    ar_token_ids: list[list[int]],
    spec_token_ids: list[list[int]],
) -> list[dict[str, int | None]]:
    mismatches = []
    for sample_index, (ar_ids, spec_ids) in enumerate(
        zip(ar_token_ids, spec_token_ids, strict=True)
    ):
        if ar_ids == spec_ids:
            continue
        common_length = min(len(ar_ids), len(spec_ids))
        first_difference = next(
            (
                position
                for position in range(common_length)
                if ar_ids[position] != spec_ids[position]
            ),
            common_length,
        )
        mismatches.append(
            {
                "sample_index": sample_index,
                "first_difference": first_difference,
                "ar_length": len(ar_ids),
                "spec_length": len(spec_ids),
                "ar_token": (
                    ar_ids[first_difference] if first_difference < len(ar_ids) else None
                ),
                "spec_token": (
                    spec_ids[first_difference]
                    if first_difference < len(spec_ids)
                    else None
                ),
            }
        )
    return mismatches


def metric_mean_ms(run: dict[str, Any], metric: str) -> float | None:
    value = run[metric]
    return value[0] * 1000 if value is not None else None


def metric_ratio(
    numerator: dict[str, Any], denominator: dict[str, Any], metric: str
) -> str:
    numerator_ms = metric_mean_ms(numerator, metric)
    denominator_ms = metric_mean_ms(denominator, metric)
    if numerator_ms is None or denominator_ms in (None, 0):
        return "-"
    return f"{numerator_ms / denominator_ms:.3f}x"


def format_value(value: float | None, unit: str) -> str:
    return "n/a" if value is None else f"{value:.3f} {unit}"


def summarize_samples(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def format_distribution(summary: dict[str, float], unit: str) -> str:
    return (
        f"{summary['mean']:.3f} ± {summary['stdev']:.3f} {unit} "
        f"(p50 {summary['median']:.3f})"
    )


def print_comparison(
    args: argparse.Namespace,
    prompt_token_counts: list[int],
    ar: dict[str, Any],
    spec: dict[str, Any],
    num_exact_output_matches: int,
    spec_stats: dict[str, int | list[int]] | None,
    ar_batch_elapsed: list[float],
    spec_batch_elapsed: list[float],
    ar_batch_output_tokens: list[int],
    spec_batch_output_tokens: list[int],
    mismatch_diagnostics: list[dict[str, int | None]],
) -> dict[str, Any]:
    speedup = ar["elapsed_seconds"] / spec["elapsed_seconds"]
    num_batches = len(ar_batch_elapsed)
    ar_latency_dist = summarize_samples(ar_batch_elapsed)
    spec_latency_dist = summarize_samples(spec_batch_elapsed)
    paired_speedups = [
        ar_elapsed / spec_elapsed
        for ar_elapsed, spec_elapsed in zip(
            ar_batch_elapsed, spec_batch_elapsed, strict=True
        )
    ]
    paired_speedup_dist = summarize_samples(paired_speedups)
    ar_output_dist = summarize_samples(
        [float(value) for value in ar_batch_output_tokens]
    )
    spec_output_dist = summarize_samples(
        [float(value) for value in spec_batch_output_tokens]
    )
    exact_output_match = num_exact_output_matches == args.num_samples
    throughput_ratio = (
        spec["e2e_output_throughput"] / ar["e2e_output_throughput"]
        if ar["e2e_output_throughput"]
        else float("nan")
    )
    output_token_ratio = (
        spec["output_tokens"] / ar["output_tokens"]
        if ar["output_tokens"]
        else float("nan")
    )
    print(f"\n{'=' * 94}")
    print("  CLOSED-BATCH AR vs SPECULATIVE DECODING")
    print(f"{'=' * 94}")
    print("  Admission policy : sequential closed batches; no backlog and no refill")
    print("  EOS policy       : respected (ignore_eos=False)")
    print(
        f"  Workload         : {args.num_samples} samples = {num_batches} batches × "
        f"{args.batch_size} requests"
    )
    print(
        f"  Input length     : mean {statistics.mean(prompt_token_counts):,.1f} "
        f"tokens/request"
    )
    print(f"  Sampling         : temperature={args.temperature:g}")
    print("  GPU profiling    : verification + draft phase events (non-blocking)")

    headers = ("Metric", "AR baseline", args.algorithm, "Spec / AR")
    rows = [
        (
            "Batch latency mean ± SD",
            format_distribution(ar_latency_dist, "s"),
            format_distribution(spec_latency_dist, "s"),
            f"{spec_latency_dist['mean'] / ar_latency_dist['mean']:.3f}x",
        ),
        (
            "Total measured latency",
            f"{ar['elapsed_seconds']:.3f} s",
            f"{spec['elapsed_seconds']:.3f} s",
            f"{spec['elapsed_seconds'] / ar['elapsed_seconds']:.3f}x",
        ),
        (
            "E2E output throughput",
            f"{ar['e2e_output_throughput']:.2f} tok/s",
            f"{spec['e2e_output_throughput']:.2f} tok/s",
            f"{throughput_ratio:.3f}x",
        ),
        (
            "Total output tokens",
            f"{ar['output_tokens']:,}",
            f"{spec['output_tokens']:,}",
            f"{output_token_ratio:.3f}x",
        ),
        (
            "Output tokens/batch",
            format_distribution(ar_output_dist, "tokens"),
            format_distribution(spec_output_dist, "tokens"),
            "-",
        ),
        (
            "Mean TTFT",
            format_value(metric_mean_ms(ar, "ttft"), "ms"),
            format_value(metric_mean_ms(spec, "ttft"), "ms"),
            metric_ratio(spec, ar, "ttft"),
        ),
        (
            "Mean prefill",
            format_value(metric_mean_ms(ar, "prefill"), "ms"),
            format_value(metric_mean_ms(spec, "prefill"), "ms"),
            metric_ratio(spec, ar, "prefill"),
        ),
        (
            "Mean decode / request",
            format_value(metric_mean_ms(ar, "decode_latency"), "ms"),
            format_value(metric_mean_ms(spec, "decode_latency"), "ms"),
            metric_ratio(spec, ar, "decode_latency"),
        ),
        (
            "Mean TPOT",
            format_value(metric_mean_ms(ar, "tpot"), "ms/token"),
            format_value(metric_mean_ms(spec, "tpot"), "ms/token"),
            metric_ratio(spec, ar, "tpot"),
        ),
        (
            "Mean request E2E",
            format_value(metric_mean_ms(ar, "request_e2e"), "ms"),
            format_value(metric_mean_ms(spec, "request_e2e"), "ms"),
            metric_ratio(spec, ar, "request_e2e"),
        ),
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(4)
    ]
    print()
    print("  " + "  ".join(f"{value:<{widths[i]}}" for i, value in enumerate(headers)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(f"{value:<{widths[i]}}" for i, value in enumerate(row)))

    print(f"\n  E2E LATENCY SPEEDUP (AR / {args.algorithm}) : {speedup:.3f}x")
    print(
        "  Paired batch speedup mean ± SD             : "
        f"{paired_speedup_dist['mean']:.3f}x ± "
        f"{paired_speedup_dist['stdev']:.3f}x "
        f"(p50 {paired_speedup_dist['median']:.3f}x)"
    )
    print(
        "  Exact generated-token matches              : "
        f"{num_exact_output_matches}/{args.num_samples}"
    )
    if not exact_output_match:
        if args.temperature == 0:
            print(
                "  WARNING: Greedy outputs should match. Showing the first "
                "mismatches (sample indices are zero-based):"
            )
            for mismatch in mismatch_diagnostics[:5]:
                print(
                    "    sample {sample_index}: first difference at token "
                    "{first_difference}, AR={ar_token}, spec={spec_token}, "
                    "lengths={ar_length}/{spec_length}".format(**mismatch)
                )
            print(
                "  This indicates a target-path numerical difference or a "
                "correctness bug, not an expected effect of sparse drafting."
            )
        else:
            print(
                "  NOTE: Exact token equality is not required for stochastic "
                "sampling, even with identical distributional settings."
            )

    result: dict[str, Any] = {
        "algorithm": args.algorithm,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "num_batches": num_batches,
        "continuous_admission": False,
        "ignore_eos": False,
        "temperature": args.temperature,
        "ar_elapsed_seconds": ar["elapsed_seconds"],
        "spec_elapsed_seconds": spec["elapsed_seconds"],
        "e2e_latency_speedup": speedup,
        "ar_output_tokens": ar["output_tokens"],
        "spec_output_tokens": spec["output_tokens"],
        "exact_output_match": exact_output_match,
        "num_exact_output_matches": num_exact_output_matches,
        "mismatched_sample_indices": [
            diagnostic["sample_index"] for diagnostic in mismatch_diagnostics
        ],
        "output_mismatch_diagnostics": mismatch_diagnostics,
        "ar_batch_latency_mean_seconds": ar_latency_dist["mean"],
        "ar_batch_latency_stdev_seconds": ar_latency_dist["stdev"],
        "ar_batch_latency_p50_seconds": ar_latency_dist["median"],
        "spec_batch_latency_mean_seconds": spec_latency_dist["mean"],
        "spec_batch_latency_stdev_seconds": spec_latency_dist["stdev"],
        "spec_batch_latency_p50_seconds": spec_latency_dist["median"],
        "paired_batch_speedup_mean": paired_speedup_dist["mean"],
        "paired_batch_speedup_stdev": paired_speedup_dist["stdev"],
        "paired_batch_speedup_p50": paired_speedup_dist["median"],
        "ar_mean_ttft_ms": metric_mean_ms(ar, "ttft"),
        "spec_mean_ttft_ms": metric_mean_ms(spec, "ttft"),
        "ar_mean_prefill_ms": metric_mean_ms(ar, "prefill"),
        "spec_mean_prefill_ms": metric_mean_ms(spec, "prefill"),
        "ar_mean_decode_latency_ms": metric_mean_ms(ar, "decode_latency"),
        "spec_mean_decode_latency_ms": metric_mean_ms(spec, "decode_latency"),
        "ar_mean_tpot_ms": metric_mean_ms(ar, "tpot"),
        "spec_mean_tpot_ms": metric_mean_ms(spec, "tpot"),
        "ar_mean_request_e2e_ms": metric_mean_ms(ar, "request_e2e"),
        "spec_mean_request_e2e_ms": metric_mean_ms(spec, "request_e2e"),
    }
    if spec_stats is not None:
        num_drafts = int(spec_stats["num_drafts"])
        num_draft_tokens = int(spec_stats["num_draft_tokens"])
        num_accepted_tokens = int(spec_stats["num_accepted_tokens"])
        mean_acceptance_length = (
            1 + num_accepted_tokens / num_drafts if num_drafts else 1.0
        )
        acceptance_rate = (
            num_accepted_tokens / num_draft_tokens if num_draft_tokens else 0.0
        )
        print(f"\n  Mean accepted length (+ bonus) : {mean_acceptance_length:.3f}")
        print(f"  Draft token acceptance         : {100 * acceptance_rate:.2f}%")

        verification_ms = mean_gpu_latency_ms(spec_stats, "verification")
        draft_ms = mean_gpu_latency_ms(spec_stats, "draft")

        def show_gpu_ms(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f} ms / engine step"

        print(f"\n  {args.algorithm.upper()} GPU LATENCY BREAKDOWN (SPECULATIVE ONLY)")
        print("  ------------------------------------------------")
        print(f"  Verification phase            : {show_gpu_ms(verification_ms)}")
        print(f"  Draft phase                   : {show_gpu_ms(draft_ms)}")
        if draft_ms is not None:
            print(
                "  Draft / speculative token      : "
                f"{draft_ms / args.num_speculative_tokens:.3f} ms (amortized)"
            )
        profiled_step_ms = sum(
            value for value in (verification_ms, draft_ms) if value is not None
        )
        if profiled_step_ms:
            print(
                "  Profiled phase total           : "
                f"{profiled_step_ms:.3f} ms / engine step"
            )
            print(
                "  Profiled time / emitted token  : "
                f"{profiled_step_ms / mean_acceptance_length:.3f} ms "
                "(amortized)"
            )
        result.update(
            {
                "mean_acceptance_length": mean_acceptance_length,
                "draft_acceptance_rate": acceptance_rate,
                "mean_verification_gpu_ms": verification_ms,
                "mean_draft_gpu_ms": draft_ms,
            }
        )

    print(f"\n  RESULT_JSON {json.dumps(result, sort_keys=True)}")
    print(f"{'=' * 94}")
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)

    os.environ["VLLM_SPEC_DECODE_LATENCY_METRICS"] = "1"

    prompts, prompt_token_counts, warmup_prompts = prepare_prompts(args)
    num_batches = args.num_samples // args.batch_size
    print(
        f"Prepared {args.num_samples} {args.dataset} samples as {num_batches} "
        f"closed batches of {args.batch_size}; running AR first, then "
        f"{args.algorithm}"
    )

    print("\n[1/2] AR baseline")
    (
        ar,
        ar_outputs,
        _,
        ar_batch_elapsed,
        ar_batch_output_tokens,
    ) = run_closed_batches(args, "ar", prompts, warmup_prompts)
    print(f"\n[2/2] {args.algorithm}")
    (
        spec,
        spec_outputs,
        spec_stats,
        spec_batch_elapsed,
        spec_batch_output_tokens,
    ) = run_closed_batches(args, args.algorithm, prompts, warmup_prompts)

    ar_token_ids = output_token_ids(ar_outputs)
    spec_token_ids = output_token_ids(spec_outputs)
    num_exact_output_matches = sum(
        ar_ids == spec_ids
        for ar_ids, spec_ids in zip(ar_token_ids, spec_token_ids, strict=True)
    )
    mismatch_diagnostics = diagnose_output_mismatches(ar_token_ids, spec_token_ids)
    print_comparison(
        args,
        prompt_token_counts,
        ar,
        spec,
        num_exact_output_matches,
        spec_stats,
        ar_batch_elapsed,
        spec_batch_elapsed,
        ar_batch_output_tokens,
        spec_batch_output_tokens,
        mismatch_diagnostics,
    )


if __name__ == "__main__":
    main()
