"""Shared online-serving utilities for the Vegas throughput benchmarks."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
from tqdm import tqdm

METRIC_NAMES = {
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:spec_decode_num_drafts_total": "num_drafts",
    "vllm:spec_decode_num_draft_tokens_total": "num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "num_accepted_tokens",
    "vllm:spec_decode_verification_latency_us_total": "verification_latency_us",
    "vllm:spec_decode_verification_steps_total": "verification_steps",
    "vllm:spec_decode_draft_latency_us_total": "draft_latency_us",
    "vllm:spec_decode_draft_steps_total": "draft_steps",
}
POSITION_METRIC = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
METRIC_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+(?P<value>[-+0-9.eE]+)(?:\s+\d+)?$"
)
POSITION_RE = re.compile(r'(?:^|,)position="(?P<position>\d+)"(?:,|$)')


@dataclass
class RequestSample:
    index: int
    prompt_tokens: int
    output_tokens: int
    start_time: float
    first_token_time: float | None
    end_time: float
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.start_time

    @property
    def decode_latency(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.end_time - self.first_token_time

    @property
    def e2e_latency(self) -> float:
        return self.end_time - self.start_time

    @property
    def tpot(self) -> float | None:
        decode_latency = self.decode_latency
        if decode_latency is None or self.output_tokens <= 1:
            return None
        return decode_latency / (self.output_tokens - 1)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "stdev": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def _parse_prometheus(text: str) -> dict[str, float | list[float]]:
    values: dict[str, float | list[float]] = {
        name: 0.0 for name in METRIC_NAMES.values()
    }
    positions: dict[int, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if match is None:
            continue
        metric_name = match.group("name")
        value = float(match.group("value"))
        if metric_name in METRIC_NAMES:
            key = METRIC_NAMES[metric_name]
            assert isinstance(values[key], float)
            values[key] += value
        elif metric_name == POSITION_METRIC:
            labels = match.group("labels") or ""
            position_match = POSITION_RE.search(labels)
            if position_match is not None:
                position = int(position_match.group("position"))
                positions[position] = positions.get(position, 0.0) + value
    if positions:
        values["accepted_per_position"] = [
            positions.get(index, 0.0) for index in range(max(positions) + 1)
        ]
    else:
        values["accepted_per_position"] = []
    return values


async def fetch_metrics(
    session: aiohttp.ClientSession, metrics_url: str
) -> tuple[float, dict[str, float | list[float]]]:
    before = time.perf_counter()
    async with session.get(metrics_url) as response:
        response.raise_for_status()
        body = await response.text()
    after = time.perf_counter()
    return (before + after) / 2, _parse_prometheus(body)


def metric_delta(
    after: dict[str, float | list[float]],
    before: dict[str, float | list[float]],
) -> dict[str, float | list[float]]:
    result: dict[str, float | list[float]] = {}
    for key, after_value in after.items():
        before_value = before.get(key, [] if isinstance(after_value, list) else 0.0)
        if isinstance(after_value, list):
            assert isinstance(before_value, list)
            width = max(len(after_value), len(before_value))
            result[key] = [
                (after_value[index] if index < len(after_value) else 0.0)
                - (before_value[index] if index < len(before_value) else 0.0)
                for index in range(width)
            ]
        else:
            assert isinstance(before_value, float)
            result[key] = after_value - before_value
    return result


class StableWindowTracker:
    def __init__(
        self,
        *,
        total_requests: int,
        warmup_completions: int,
        tail_completions: int,
        metrics_url: str,
        session: aiohttp.ClientSession,
        poll_interval: float,
        description: str,
        show_progress: bool,
    ) -> None:
        self.total_requests = total_requests
        self.warmup_completions = warmup_completions
        self.measurement_end_rank = total_requests - tail_completions
        self.metrics_url = metrics_url
        self.session = session
        self.poll_interval = poll_interval
        self.samples: list[RequestSample] = []
        self.snapshots: list[tuple[float, dict[str, float | list[float]]]] = []
        self.lock = asyncio.Lock()
        self.started = asyncio.Event()
        self.ended = asyncio.Event()
        self.progress = tqdm(
            total=total_requests,
            desc=description,
            unit="request",
            disable=not show_progress,
        )

    async def record(self, sample: RequestSample) -> None:
        async with self.lock:
            self.samples.append(sample)
            completed = len(self.samples)
            self.progress.update()
            self.progress.set_postfix(
                phase=(
                    "warmup"
                    if completed <= self.warmup_completions
                    else "stable"
                    if completed <= self.measurement_end_rank
                    else "drain"
                ),
                output_tokens=sample.output_tokens,
            )
            if completed == self.warmup_completions:
                self.snapshots.append(
                    await fetch_metrics(self.session, self.metrics_url)
                )
                self.started.set()
            if completed == self.measurement_end_rank:
                self.snapshots.append(
                    await fetch_metrics(self.session, self.metrics_url)
                )
                self.ended.set()

    async def poll_metrics(self) -> None:
        await self.started.wait()
        while not self.ended.is_set():
            try:
                await asyncio.wait_for(self.ended.wait(), timeout=self.poll_interval)
            except TimeoutError:
                async with self.lock:
                    if not self.ended.is_set():
                        self.snapshots.append(
                            await fetch_metrics(self.session, self.metrics_url)
                        )

    def close(self) -> None:
        self.progress.close()


def _completion_text_present(event: dict[str, Any]) -> bool:
    for choice in event.get("choices", []):
        # Completion streaming can emit an empty string for a valid token
        # while the incremental decoder waits for enough bytes. Presence of
        # the choice/text field, rather than truthiness of decoded text, is the
        # correct first-token boundary.
        if "text" in choice:
            return True
        if "delta" in choice:
            return True
    return False


async def stream_completion(
    session: aiohttp.ClientSession,
    *,
    url: str,
    payload: dict[str, Any],
    index: int,
    prompt_tokens: int,
    headers: dict[str, str] | None = None,
) -> RequestSample:
    started = time.perf_counter()
    first_token: float | None = None
    output_tokens = 0
    error: str | None = None
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                if first_token is None and _completion_text_present(event):
                    first_token = time.perf_counter()
                usage = event.get("usage")
                if usage is not None:
                    output_tokens = int(usage.get("completion_tokens", output_tokens))
    except Exception as exc:  # Preserve all samples and fail after cleanup.
        error = f"{type(exc).__name__}: {exc}"
    return RequestSample(
        index=index,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        start_time=started,
        first_token_time=first_token,
        end_time=time.perf_counter(),
        error=error,
    )


RequestRunner = Callable[
    [aiohttp.ClientSession, int, str, int, dict[str, Any]],
    Awaitable[RequestSample],
]


async def run_online_workload(
    *,
    prompts: list[str],
    prompt_token_counts: list[int],
    payload_template: dict[str, Any],
    request_runner: RequestRunner,
    metrics_url: str,
    warmup_completions: int,
    tail_completions: int,
    poll_interval: float,
    description: str,
    show_progress: bool,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=120)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tracker = StableWindowTracker(
            total_requests=len(prompts),
            warmup_completions=warmup_completions,
            tail_completions=tail_completions,
            metrics_url=metrics_url,
            session=session,
            poll_interval=poll_interval,
            description=description,
            show_progress=show_progress,
        )
        poller = asyncio.create_task(tracker.poll_metrics())

        async def run_one(index: int, prompt: str, prompt_tokens: int) -> None:
            payload = dict(payload_template)
            payload["prompt"] = prompt
            payload["seed"] = int(payload_template["seed"]) + index
            sample = await request_runner(
                session, index, prompt, prompt_tokens, payload
            )
            await tracker.record(sample)

        tasks = [
            asyncio.create_task(run_one(index, prompt, prompt_token_counts[index]))
            for index, prompt in enumerate(prompts)
        ]
        await asyncio.gather(*tasks)
        await poller
        tracker.close()

    failures = [sample for sample in tracker.samples if sample.error]
    if failures:
        preview = "\n".join(
            f"request {sample.index}: {sample.error}" for sample in failures[:5]
        )
        raise RuntimeError(f"{len(failures)} requests failed:\n{preview}")
    if len(tracker.snapshots) < 2:
        raise RuntimeError("Stable measurement window did not produce two snapshots")

    tracker.snapshots.sort(key=lambda item: item[0])
    start_time, start_metrics = tracker.snapshots[0]
    end_time, end_metrics = tracker.snapshots[-1]
    duration = end_time - start_time
    counters = metric_delta(end_metrics, start_metrics)
    generation_tokens = float(counters["generation_tokens"])
    stable_samples = [
        sample
        for sample in tracker.samples
        if start_time <= sample.end_time <= end_time
    ]
    fallback_used = generation_tokens <= 0
    if fallback_used:
        generation_tokens = float(
            sum(sample.output_tokens for sample in stable_samples)
        )

    interval_rates = []
    for (left_ts, left), (right_ts, right) in zip(
        tracker.snapshots, tracker.snapshots[1:], strict=False
    ):
        interval = right_ts - left_ts
        if interval <= 0.05:
            continue
        delta = metric_delta(right, left)
        interval_rates.append(float(delta["generation_tokens"]) / interval)

    def values(attribute: str) -> list[float]:
        result = []
        for sample in stable_samples:
            value = getattr(sample, attribute)
            if value is not None:
                result.append(float(value))
        return result

    return {
        "stable_window": {
            "warmup_completions": warmup_completions,
            "measured_completion_target": len(prompts)
            - warmup_completions
            - tail_completions,
            "observed_completed_requests": len(stable_samples),
            "tail_completions": tail_completions,
            "duration_seconds": duration,
            "generation_tokens": generation_tokens,
            "generation_counter_fallback_used": fallback_used,
            "decode_throughput_tokens_per_second": generation_tokens / duration,
            "request_throughput_per_second": len(stable_samples) / duration,
            "throughput_interval_seconds": poll_interval,
            "throughput_samples_tokens_per_second": interval_rates,
            "throughput_distribution": distribution(interval_rates),
        },
        "latency_seconds": {
            "ttft": distribution(values("ttft")),
            "decode": distribution(values("decode_latency")),
            "tpot": distribution(values("tpot")),
            "e2e": distribution(values("e2e_latency")),
        },
        "output_tokens_per_request": distribution(
            [float(sample.output_tokens) for sample in stable_samples]
        ),
        "prompt_tokens_per_request": distribution(
            [float(sample.prompt_tokens) for sample in stable_samples]
        ),
        "speculative_counters": counters,
        "requests": [asdict(sample) for sample in tracker.samples],
    }


def speculative_summary(run: dict[str, Any], gamma: int) -> dict[str, Any] | None:
    counters = run["speculative_counters"]
    drafts = float(counters["num_drafts"])
    draft_tokens = float(counters["num_draft_tokens"])
    accepted_tokens = float(counters["num_accepted_tokens"])
    if drafts <= 0:
        return None
    verify_steps = float(counters["verification_steps"])
    draft_steps = float(counters["draft_steps"])
    positions = counters.get("accepted_per_position", [])
    assert isinstance(positions, list)
    return {
        "gamma": gamma,
        "draft_iterations": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "mean_accepted_length_including_bonus": 1 + accepted_tokens / drafts,
        "draft_token_acceptance_rate": (
            accepted_tokens / draft_tokens if draft_tokens else 0.0
        ),
        "acceptance_rate_by_position": [value / drafts for value in positions],
        "mean_verification_latency_ms": (
            float(counters["verification_latency_us"]) / verify_steps / 1000
            if verify_steps
            else None
        ),
        "mean_draft_latency_ms": (
            float(counters["draft_latency_us"]) / draft_steps / 1000
            if draft_steps
            else None
        ),
    }


def _format_number(value: float | int | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def print_distribution(
    label: str, stats: dict[str, Any], scale: float, unit: str
) -> None:
    def scaled(key: str) -> str:
        value = stats[key]
        return _format_number(value * scale if value is not None else None)

    print(
        f"  {label:<25} mean {scaled('mean')} | sd {scaled('stdev')} "
        f"| p50 {scaled('p50')} | p95 {scaled('p95')} "
        f"| p99 {scaled('p99')} {unit}"
    )


def print_run_summary(name: str, run: dict[str, Any], gamma: int) -> None:
    stable = run["stable_window"]
    print(f"\n{name}")
    print("-" * len(name))
    print(
        f"  Stable decode throughput : "
        f"{stable['decode_throughput_tokens_per_second']:,.2f} tokens/s"
    )
    print(
        f"  Stable request throughput: "
        f"{stable['request_throughput_per_second']:,.3f} requests/s"
    )
    print(
        f"  Measurement window       : {stable['duration_seconds']:,.3f} s, "
        f"{stable['generation_tokens']:,.0f} generated tokens"
    )
    throughput_dist = stable["throughput_distribution"]
    if throughput_dist["count"]:
        print_distribution("Throughput intervals", throughput_dist, 1, "tokens/s")
    print_distribution(
        "Output tokens/request", run["output_tokens_per_request"], 1, "tokens"
    )
    print_distribution("TTFT", run["latency_seconds"]["ttft"], 1000, "ms")
    print_distribution("Decode latency", run["latency_seconds"]["decode"], 1000, "ms")
    print_distribution("TPOT", run["latency_seconds"]["tpot"], 1000, "ms/token")
    print_distribution("E2E latency", run["latency_seconds"]["e2e"], 1000, "ms")
    spec = speculative_summary(run, gamma)
    if spec is not None:
        print(
            f"  Mean accepted (+bonus)   : "
            f"{spec['mean_accepted_length_including_bonus']:.3f} tokens/iteration"
        )
        print(
            f"  Draft acceptance         : "
            f"{100 * spec['draft_token_acceptance_rate']:.2f}%"
        )
        print(
            f"  Draft / verify GPU time  : "
            f"{_format_number(spec['mean_draft_latency_ms'])} / "
            f"{_format_number(spec['mean_verification_latency_ms'])} ms/step"
        )


def compare_runs(
    ar: dict[str, Any], spec: dict[str, Any], algorithm: str, gamma: int
) -> dict[str, Any]:
    ar_throughput = ar["stable_window"]["decode_throughput_tokens_per_second"]
    spec_throughput = spec["stable_window"]["decode_throughput_tokens_per_second"]
    ar_request_throughput = ar["stable_window"]["request_throughput_per_second"]
    spec_request_throughput = spec["stable_window"]["request_throughput_per_second"]
    ar_mean_output = ar["output_tokens_per_request"]["mean"]
    spec_mean_output = spec["output_tokens_per_request"]["mean"]

    latency_speedups: dict[str, float | None] = {}
    for metric in ("ttft", "decode", "tpot", "e2e"):
        ar_mean = ar["latency_seconds"][metric]["mean"]
        spec_mean = spec["latency_seconds"][metric]["mean"]
        latency_speedups[metric] = (
            ar_mean / spec_mean
            if ar_mean is not None and spec_mean not in (None, 0)
            else None
        )
    result = {
        "algorithm": algorithm,
        "ar_decode_throughput_tokens_per_second": ar_throughput,
        "spec_decode_throughput_tokens_per_second": spec_throughput,
        "decode_throughput_speedup": spec_throughput / ar_throughput,
        "request_throughput_speedup": (spec_request_throughput / ar_request_throughput),
        "mean_output_tokens_spec_over_ar": (
            spec_mean_output / ar_mean_output
            if ar_mean_output not in (None, 0) and spec_mean_output is not None
            else None
        ),
        "latency_speedup_ar_over_spec": latency_speedups,
        "speculative": speculative_summary(spec, gamma),
        "ar": ar,
        "spec": spec,
    }
    print_run_summary("AR baseline", ar, gamma)
    print_run_summary(algorithm, spec, gamma)
    print(
        f"\nSTABLE DECODE THROUGHPUT SPEEDUP ({algorithm} / AR): "
        f"{result['decode_throughput_speedup']:.3f}x"
    )
    print(
        "LATENCY SPEEDUP (AR / spec): "
        + ", ".join(
            f"{metric}={value:.3f}x" if value is not None else f"{metric}=n/a"
            for metric, value in latency_speedups.items()
        )
    )
    return result


def make_payload(
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    seed: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": "",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "seed": seed,
        "ignore_eos": ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def make_server_command(
    *,
    model: str,
    port: int,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    seed: int,
    speculative_config: dict[str, Any] | None,
    kv_transfer_config: dict[str, Any] | None = None,
    rope_scaling: dict[str, Any] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        str(seed),
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--disable-log-requests",
    ]
    if speculative_config is not None:
        command.extend(["--speculative-config", json.dumps(speculative_config)])
    if kv_transfer_config is not None:
        command.extend(["--kv-transfer-config", json.dumps(kv_transfer_config)])
    if rope_scaling is not None:
        # This vLLM revision exposes model-config overrides through
        # --hf-overrides rather than a standalone --rope-scaling flag.
        command.extend(
            ["--hf-overrides", json.dumps({"rope_scaling": rope_scaling})]
        )
    if extra_args:
        command.extend(extra_args)
    return command


def ensure_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # The AR and speculative servers run sequentially on the same port.
        # Allow rebinding after the AR server exits while its completed TCP
        # connections are still in TIME_WAIT. This does not permit binding
        # over an active listener (SO_REUSEPORT would be required for that).
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is already in use") from exc


class ManagedServer:
    def __init__(
        self,
        *,
        command: list[str],
        gpus: list[str],
        port: int,
        log_path: Path,
        startup_timeout: float,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.command = command
        self.gpus = gpus
        self.port = port
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.environment = environment or {}
        self.process: subprocess.Popen[str] | None = None
        self.log_file: Any = None

    def start(self) -> None:
        ensure_port_available(self.port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(self.environment)
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpus)
        environment["VLLM_SPEC_DECODE_LATENCY_METRICS"] = "1"
        environment["VLLM_LOG_STATS_INTERVAL"] = str(24 * 60 * 60)
        self.process = subprocess.Popen(
            self.command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout
        health_url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited with code {self.process.returncode}; "
                    f"see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        raise TimeoutError(
            f"vLLM server on port {self.port} did not become ready within "
            f"{self.startup_timeout:g}s; see {self.log_path}"
        )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self.log_file is not None:
            self.log_file.close()

    def __enter__(self) -> ManagedServer:
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


def speculative_config(
    algorithm: str,
    gamma: int,
    sparse_attn_ratio: float,
    sparse_attn_min_tokens: int = 256,
    draft_ffn_keep_ratio: float = 1.0,
) -> dict[str, Any]:
    return {
        "method": "sparse_attn",
        "num_speculative_tokens": gamma,
        "sparse_attn_algorithm": algorithm,
        "sparse_attn_ratio": sparse_attn_ratio,
        "sparse_attn_min_tokens": sparse_attn_min_tokens,
        "draft_ffn_keep_ratio": draft_ffn_keep_ratio,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
