import argparse
import json
import os
import statistics
import time
from collections.abc import Mapping
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

# Keep vLLM's periodic logger from splitting one benchmark run into multiple
# reporting windows. Exact measured-section statistics are computed from
# counter snapshots taken after warmup and after the benchmark.
os.environ["VLLM_LOG_STATS_INTERVAL"] = str(24 * 60 * 60)

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector


def read_spec_decode_stats(
    llm: LLM,
    num_spec_tokens: int,
) -> dict[str, int | list[int]]:
    """Read cumulative speculative-decoding Prometheus counters."""
    stats: dict[str, int | list[int]] = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
        "accepted_per_position": [0] * num_spec_tokens,
        "verification_latency_us": 0,
        "verification_steps": 0,
        "draft_latency_us": 0,
        "draft_steps": 0,
    }
    scalar_metrics = {
        "vllm:spec_decode_num_drafts": "num_drafts",
        "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
        "vllm:spec_decode_verification_latency_us": "verification_latency_us",
        "vllm:spec_decode_verification_steps": "verification_steps",
        "vllm:spec_decode_draft_latency_us": "draft_latency_us",
        "vllm:spec_decode_draft_steps": "draft_steps",
    }

    for metric in llm.get_metrics():
        if metric.name in scalar_metrics:
            assert isinstance(metric, Counter)
            key = scalar_metrics[metric.name]
            stats[key] = int(stats[key]) + metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            accepted_per_position = stats["accepted_per_position"]
            assert isinstance(accepted_per_position, list)
            for position, value in enumerate(metric.values[:num_spec_tokens]):
                accepted_per_position[position] += value

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Vegas or baseline offline benchmark on AIME'25 "
        "or token-length-controlled LongBench v2 prompts."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--dataset",
        choices=("aime25", "longbench-v2"),
        default="aime25",
        help="Use LongBench v2 to benchmark long-context decoding.",
    )
    parser.add_argument(
        "--longbench-length",
        choices=("short", "medium", "long", "any"),
        default="long",
        help="LongBench v2 length category to sample.",
    )
    parser.add_argument(
        "--target-input-tokens",
        type=int,
        default=None,
        help="Target chat-formatted LongBench input length. By default, use "
        "--max-model-len minus --max-tokens.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("vegas", "streamingllm", "none"),
        default="vegas",
        help="Use 'none' for the non-speculative vLLM baseline.",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Always generate --max-tokens tokens. This is useful for "
        "throughput tests but invalidates LongBench quality evaluation.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable model-specific thinking mode in the chat template when "
        "supported (enabled by default).",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=16,
        help="Generate this many untimed tokens to warm up runtime kernels; "
        "use 0 to disable.",
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=6)
    parser.add_argument("--sparse-attn-ratio", type=float, default=0.07)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def apply_chat_template(
    tokenizer: AutoTokenizer,
    prompt: str,
    enable_thinking: bool,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def format_longbench_v2(item: Mapping[str, Any]) -> str:
    """Format one LongBench v2 multiple-choice example."""
    return (
        "Read the following context and answer the multiple-choice question.\n\n"
        f"Context:\n{item['context']}\n\n"
        f"Question:\n{item['question']}\n\n"
        f"A. {item['choice_A']}\n"
        f"B. {item['choice_B']}\n"
        f"C. {item['choice_C']}\n"
        f"D. {item['choice_D']}\n\n"
        "Give the correct answer and explain your reasoning."
    )


def encode_long_text(
    tokenizer: AutoTokenizer,
    text: str,
    required_tokens: int,
) -> list[int]:
    """Tokenize enough of both ends of a potentially multi-megabyte text."""
    # Some LongBench v2 contexts contain millions of characters. Avoid
    # tokenizing the entire document when only a small head/tail slice can fit.
    chars_per_side = max(required_tokens * 4, 4096)
    while len(text) > 2 * chars_per_side:
        candidate = text[:chars_per_side] + text[-chars_per_side:]
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) >= required_tokens:
            return token_ids
        chars_per_side *= 2
    return tokenizer.encode(text, add_special_tokens=False)


def truncate_middle_to_tokens(
    tokenizer: AutoTokenizer,
    text: str,
    max_tokens: int,
) -> tuple[str, bool]:
    """Keep the beginning and end of text within a tokenizer budget."""
    token_ids = encode_long_text(tokenizer, text, max_tokens)
    if len(token_ids) <= max_tokens:
        return tokenizer.decode(token_ids, skip_special_tokens=True), False

    left = max_tokens // 2
    right = max_tokens - left
    kept_ids = token_ids[:left] + token_ids[-right:]
    return tokenizer.decode(kept_ids, skip_special_tokens=True), True


def build_longbench_prompt(
    tokenizer: AutoTokenizer,
    item: Mapping[str, Any],
    target_input_tokens: int,
    enable_thinking: bool,
) -> tuple[str, int]:
    """Build a chat-formatted prompt close to an exact input-token target."""
    item_without_context = dict(item)
    item_without_context["context"] = ""
    fixed_prompt = apply_chat_template(
        tokenizer,
        format_longbench_v2(item_without_context),
        enable_thinking,
    )
    fixed_tokens = len(tokenizer.encode(fixed_prompt, add_special_tokens=False))
    # Reserve a small margin for token merges at the context boundaries.
    context_budget = target_input_tokens - fixed_tokens - 16
    if context_budget <= 0:
        raise ValueError(
            "--target-input-tokens is too small for the LongBench question "
            "and chat template"
        )

    context, _ = truncate_middle_to_tokens(
        tokenizer, str(item["context"]), context_budget
    )
    prompt_item = dict(item)
    prompt_item["context"] = context

    # Retokenization can shift the length slightly at string boundaries. Trim
    # the context again until the final chat-formatted prompt fits.
    for _ in range(3):
        prompt = apply_chat_template(
            tokenizer,
            format_longbench_v2(prompt_item),
            enable_thinking,
        )
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_tokens <= target_input_tokens:
            return prompt, prompt_tokens
        context_budget -= prompt_tokens - target_input_tokens + 8
        if context_budget <= 0:
            break
        context, _ = truncate_middle_to_tokens(
            tokenizer, str(item["context"]), context_budget
        )
        prompt_item["context"] = context

    raise ValueError("Unable to fit a LongBench prompt in the requested budget")


def prepare_aime_prompts(
    tokenizer: AutoTokenizer,
    count: int,
    enable_thinking: bool,
) -> list[tuple[str, int]]:
    questions = load_dataset("math-ai/aime25", split="test")["problem"]
    prepared = []
    for index in range(count):
        prompt = apply_chat_template(
            tokenizer,
            questions[index % len(questions)],
            enable_thinking,
        )
        prepared.append(
            (
                prompt,
                len(tokenizer.encode(prompt, add_special_tokens=False)),
            )
        )
    return prepared


def prepare_longbench_prompts(
    tokenizer: AutoTokenizer,
    count: int,
    target_input_tokens: int,
    length_category: str,
    enable_thinking: bool,
) -> list[tuple[str, int]]:
    # Streaming avoids downloading and materializing the roughly 465 MB JSON
    # file. Individual contexts are still available in full before truncation.
    dataset = load_dataset(
        "THUDM/LongBench-v2",
        split="train",
        streaming=True,
    )
    prepared = []
    minimum_accepted_length = target_input_tokens - max(128, target_input_tokens // 100)
    for item in dataset:
        if length_category != "any" and item["length"] != length_category:
            continue
        prompt, prompt_tokens = build_longbench_prompt(
            tokenizer,
            item,
            target_input_tokens,
            enable_thinking,
        )
        # Skip examples whose source context is too short to populate the
        # requested bucket. This keeps Vegas/baseline length comparisons fair.
        if prompt_tokens < minimum_accepted_length:
            continue
        prepared.append((prompt, prompt_tokens))
        if len(prepared) == count:
            return prepared

    raise ValueError(
        f"LongBench v2 has only {len(prepared)} usable "
        f"{length_category!r} examples near {target_input_tokens} input tokens; "
        f"{count} were requested"
    )


def mean_median(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    return statistics.mean(values), statistics.median(values)


def summarize_common_metrics(outputs: list[Any], elapsed: float) -> dict[str, Any]:
    """Build metrics shared by autoregressive and speculative runs."""
    output_counts = []
    ttft_seconds = []
    prefill_seconds = []
    decode_seconds = []
    per_request_tpot = []
    request_e2e_seconds = []
    for request in outputs:
        output_tokens = sum(len(completion.token_ids) for completion in request.outputs)
        output_counts.append(output_tokens)
        metrics = request.metrics
        if metrics is None:
            continue
        if metrics.first_token_latency > 0:
            ttft_seconds.append(metrics.first_token_latency)
        if metrics.first_token_ts > 0 and metrics.scheduled_ts > 0:
            prefill_seconds.append(metrics.first_token_ts - metrics.scheduled_ts)
        if metrics.last_token_ts >= metrics.first_token_ts > 0:
            decode_time = metrics.last_token_ts - metrics.first_token_ts
            decode_seconds.append(decode_time)
            if metrics.first_token_latency > 0:
                request_e2e_seconds.append(metrics.first_token_latency + decode_time)
            if output_tokens > 1 and decode_time > 0:
                per_request_tpot.append(decode_time / (output_tokens - 1))

    output_tokens = sum(output_counts)
    return {
        "requests": len(outputs),
        "output_tokens": output_tokens,
        "output_tokens_min": min(output_counts) if output_counts else 0,
        "output_tokens_mean": statistics.mean(output_counts) if output_counts else 0,
        "output_tokens_max": max(output_counts) if output_counts else 0,
        "elapsed_seconds": elapsed,
        "request_throughput": len(outputs) / elapsed,
        "e2e_output_throughput": output_tokens / elapsed,
        "ttft": mean_median(ttft_seconds),
        "prefill": mean_median(prefill_seconds),
        "decode_latency": mean_median(decode_seconds),
        "tpot": mean_median(per_request_tpot),
        "request_e2e": mean_median(request_e2e_seconds),
    }


def format_mean_p50(summary: tuple[float, float] | None, scale: float = 1.0) -> str:
    if summary is None:
        return "n/a"
    mean, median = summary
    return f"mean {mean * scale:.3f} | p50 {median * scale:.3f}"


def print_section(title: str, rows: list[tuple[str, str]]) -> None:
    print(f"\n  {title}")
    print(f"  {'-' * len(title)}")
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{label_width}}  {value}")


def subtract_spec_stats(
    after: dict[str, int | list[int]],
    before: dict[str, int | list[int]],
) -> dict[str, int | list[int]]:
    result: dict[str, int | list[int]] = {}
    for key, after_value in after.items():
        before_value = before[key]
        if isinstance(after_value, list):
            assert isinstance(before_value, list)
            result[key] = [
                value - old
                for value, old in zip(after_value, before_value, strict=True)
            ]
        else:
            assert isinstance(before_value, int)
            result[key] = after_value - before_value
    return result


def mean_gpu_latency_ms(stats: dict[str, int | list[int]], phase: str) -> float | None:
    total_us = int(stats[f"{phase}_latency_us"])
    steps = int(stats[f"{phase}_steps"])
    return total_us / steps / 1000 if steps else None


def main() -> None:
    args = parse_args()
    if args.max_tokens >= args.max_model_len:
        raise ValueError(
            "--max-tokens must be smaller than --max-model-len so the prompt "
            "also fits in the context window"
        )
    if args.num_prompts < 1 or args.repeat < 1 or args.max_num_seqs < 1:
        raise ValueError("--num-prompts, --repeat, and --max-num-seqs must be positive")
    if args.warmup_tokens < 0:
        raise ValueError("--warmup-tokens cannot be negative")
    if args.target_input_tokens is not None and args.target_input_tokens < 1:
        raise ValueError("--target-input-tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    warmup_count = (
        min(args.num_prompts * args.repeat, args.max_num_seqs)
        if args.warmup_tokens
        else 0
    )
    required_prompt_count = args.num_prompts + warmup_count
    target_input_tokens = args.target_input_tokens
    if args.dataset == "longbench-v2":
        maximum_input_tokens = args.max_model_len - args.max_tokens
        if target_input_tokens is None:
            target_input_tokens = maximum_input_tokens
        if target_input_tokens > maximum_input_tokens:
            raise ValueError(
                "--target-input-tokens plus --max-tokens must not exceed "
                "--max-model-len"
            )
        prepared = prepare_longbench_prompts(
            tokenizer,
            required_prompt_count,
            target_input_tokens,
            args.longbench_length,
            args.enable_thinking,
        )
    else:
        prepared = prepare_aime_prompts(
            tokenizer,
            required_prompt_count,
            args.enable_thinking,
        )

    measured = prepared[: args.num_prompts]
    prompts = [prompt for prompt, _ in measured] * args.repeat
    prompt_token_counts = [count for _, count in measured] * args.repeat
    warmup_prompts = [
        prompt
        for prompt, _ in prepared[args.num_prompts : args.num_prompts + warmup_count]
    ]
    print(
        "Preparing benchmark: "
        f"{args.dataset}, {len(prompts)} requests, "
        f"{statistics.mean(prompt_token_counts):,.0f} mean input tokens"
    )

    speculative_config = None
    if args.algorithm != "none":
        speculative_config = {
            "method": "sparse_attn",
            "num_speculative_tokens": args.num_speculative_tokens,
            "sparse_attn_algorithm": args.algorithm,
            "sparse_attn_ratio": args.sparse_attn_ratio,
        }

    if speculative_config is not None:
        os.environ["VLLM_SPEC_DECODE_LATENCY_METRICS"] = "1"

    llm = LLM(
        model=args.model,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=args.seed,
        speculative_config=speculative_config,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Repeated prompts must not receive artificial prefill cache hits in
        # an end-to-end comparison between Vegas and the baseline.
        enable_prefix_caching=False,
        disable_log_stats=False,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        ignore_eos=args.ignore_eos,
    )

    if args.warmup_tokens:
        # Use held-out prompts with the same length profile. The warmup is
        # especially important because Vegas's sparse kernels compile lazily.
        warmup_params = SamplingParams(
            max_tokens=min(args.warmup_tokens, args.max_model_len - 1),
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            ignore_eos=True,
        )
        llm.generate(prompts=warmup_prompts, sampling_params=warmup_params)

    stats_before = None
    if speculative_config is not None:
        stats_before = read_spec_decode_stats(llm, args.num_speculative_tokens)

    started = time.perf_counter()
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    elapsed = time.perf_counter() - started

    spec_stats = None
    if stats_before is not None:
        stats_after = read_spec_decode_stats(llm, args.num_speculative_tokens)
        spec_stats = subtract_spec_stats(stats_after, stats_before)
        num_drafts = int(spec_stats["num_drafts"])
        num_draft_tokens = int(spec_stats["num_draft_tokens"])
        num_accepted_tokens = int(spec_stats["num_accepted_tokens"])
        accepted_per_position = spec_stats["accepted_per_position"]
        assert isinstance(accepted_per_position, list)
        mean_acceptance_length = (
            1 + num_accepted_tokens / num_drafts if num_drafts else 1.0
        )
        draft_acceptance_rate = (
            100 * num_accepted_tokens / num_draft_tokens if num_draft_tokens else 0.0
        )
        per_position_rates = "  ".join(
            f"{accepted / num_drafts:.3f}" if num_drafts else "0.000"
            for accepted in accepted_per_position
        )
    common = summarize_common_metrics(outputs, elapsed)

    print(f"\n{'=' * 88}")
    print("  VEGAS BENCHMARK RESULT")
    print(f"{'=' * 88}")
    print_section(
        "Run configuration",
        [
            ("Dataset", args.dataset),
            ("Algorithm", args.algorithm),
            ("Model", args.model),
            ("Requests", str(common["requests"])),
            (
                "Max model / output",
                f"{args.max_model_len:,} / {args.max_tokens:,} tokens",
            ),
            ("Ignore EOS", str(args.ignore_eos)),
        ],
    )
    print_section(
        "Workload & throughput (common)",
        [
            (
                "Input tokens / request",
                (
                    f"mean {statistics.mean(prompt_token_counts):,.1f} | "
                    f"range {min(prompt_token_counts):,}-{max(prompt_token_counts):,}"
                ),
            ),
            (
                "Output tokens / request",
                (
                    f"mean {common['output_tokens_mean']:,.1f} | range "
                    f"{common['output_tokens_min']:,}-{common['output_tokens_max']:,}"
                ),
            ),
            ("Measured wall time", f"{elapsed:.3f} s"),
            ("Request throughput", f"{common['request_throughput']:.3f} requests/s"),
            (
                "E2E output throughput",
                f"{common['e2e_output_throughput']:.2f} tokens/s",
            ),
        ],
    )
    print_section(
        "Request latency (common, wall clock)",
        [
            ("TTFT", f"{format_mean_p50(common['ttft'], 1000)} ms"),
            (
                "Prefill (scheduled -> first token)",
                f"{format_mean_p50(common['prefill'], 1000)} ms",
            ),
            (
                "Decode latency / request",
                f"{format_mean_p50(common['decode_latency'], 1000)} ms",
            ),
            ("TPOT", f"{format_mean_p50(common['tpot'], 1000)} ms/token"),
            (
                "Request E2E latency",
                f"{format_mean_p50(common['request_e2e'], 1000)} ms",
            ),
        ],
    )

    if spec_stats is not None:
        print_section(
            "Speculative decoding",
            [
                ("Draft length (gamma)", str(args.num_speculative_tokens)),
                ("Draft iterations (request-level)", f"{num_drafts:,}"),
                (
                    "Draft / accepted tokens",
                    f"{num_draft_tokens:,} / {num_accepted_tokens:,}",
                ),
                ("Mean accepted length (+ bonus)", f"{mean_acceptance_length:.3f}"),
                ("Draft token acceptance", f"{draft_acceptance_rate:.2f}%"),
                ("Acceptance by position", per_position_rates),
            ],
        )

        verification_ms = mean_gpu_latency_ms(spec_stats, "verification")
        draft_ms = mean_gpu_latency_ms(spec_stats, "draft")

        def show_ms(value: float | None) -> str:
            return f"{value:.3f} ms / engine step" if value is not None else "n/a"

        latency_rows = [
            ("Verification phase", show_ms(verification_ms)),
            ("Draft phase", show_ms(draft_ms)),
        ]
        if draft_ms is not None:
            latency_rows.append(
                (
                    "Draft / speculative token",
                    (f"{draft_ms / args.num_speculative_tokens:.3f} ms (amortized)"),
                )
            )
        print_section("GPU latency breakdown (speculative only)", latency_rows)

    result = {
        "dataset": args.dataset,
        "algorithm": args.algorithm,
        "requests": common["requests"],
        "input_tokens_mean": statistics.mean(prompt_token_counts),
        "output_tokens": common["output_tokens"],
        "elapsed_seconds": elapsed,
        "e2e_output_tokens_per_second": common["e2e_output_throughput"],
        "mean_ttft_ms": common["ttft"][0] * 1000 if common["ttft"] else None,
        "mean_prefill_ms": common["prefill"][0] * 1000 if common["prefill"] else None,
        "mean_tpot_ms": common["tpot"][0] * 1000 if common["tpot"] else None,
    }
    if spec_stats is not None:
        result.update(
            {
                "mean_acceptance_length": mean_acceptance_length,
                "draft_acceptance_rate": draft_acceptance_rate / 100,
                "mean_verification_gpu_ms": mean_gpu_latency_ms(
                    spec_stats, "verification"
                ),
                "mean_draft_gpu_ms": mean_gpu_latency_ms(spec_stats, "draft"),
            }
        )
    print(f"\n  RESULT_JSON {json.dumps(result, sort_keys=True)}")
    print(f"{'=' * 88}")


if __name__ == "__main__":
    main()
