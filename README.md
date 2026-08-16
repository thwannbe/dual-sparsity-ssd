# Joint Sparsity Self-Speculative Decoding

This repository is a research project for studying joint sparsity in
self-speculative decoding, with an emphasis on long-context inference,
reproducible FA2/FA3 evaluation, and lossless verification. It was forked from
the [Vegas codebase](https://github.com/platformxlab/vegas), which serves as the
initial sparse self-speculative decoding baseline.

## Installation

The project is implemented as a fork of vLLM and builds against a companion
[flash-attention fork](https://github.com/npz7yyk/vllm-flash-attn) (CUDA 12.x,
FlashAttention-2/3). This repository applies an FA2 score-collection patch at
build time and enables the companion codebase's native SM80/SM86 paged-KV
mainloop for one-token pages. This permits FA2-platform evaluation on Ampere
GPUs without a separate KV staging copy while preserving the default FA3 path
on Hopper. Build from source:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12 --managed-python
source .venv/bin/activate
uv pip install -v -e .   # compiles the CUDA/FA2 or FA3 kernels; takes a while
uv pip install datasets  # for benchmarks
```

## Usage

Enable the current sparse self-speculative baseline by passing a
`speculative_config` with `method="sparse_attn"`:

```python
from vllm import LLM, SamplingParams

speculative_config = {
    "method": "sparse_attn",
    "num_speculative_tokens": 6,
    "sparse_attn_algorithm": "vegas",   # Also supported: "streamingllm"
    "sparse_attn_ratio": 0.07,          # fraction of KV kept for drafting
    # "sparse_attn_min_tokens": 256,    # floor on the per-request KV budget
}

llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config=speculative_config,
)
print(llm.generate(..., SamplingParams(...)))
```

Key knobs (`speculative_config`):

| Field | Meaning | Default |
| --- | --- | --- |
| `sparse_attn_algorithm` | `"vegas"` or `"streamingllm"` | `"streamingllm"` |
| `sparse_attn_ratio` | Fraction of KV kept for drafting | `0.05` |
| `sparse_attn_min_tokens` | Floor on the per-request KV budget | `256` |
| `num_speculative_tokens` | Draft length per step | / |

The top-k ranking metric (`"logit"` raw scores vs `"weight"` rematerialized
softmax weights) is a class-level `SCORE_MODE` toggle on `VegasAttnOverrider`.

## Example

A complete, runnable end-to-end example (AIME'25, Qwen3-8B) lives in
[`benchmarks/benchmark_vegas.py`](benchmarks/benchmark_vegas.py):

```bash
# 24 GiB Ampere-friendly smoke benchmark (the script defaults)
python benchmarks/benchmark_vegas.py

# Compare against the same non-speculative vLLM baseline
python benchmarks/benchmark_vegas.py --algorithm none
```

The defaults use `max_model_len=8192`, `max_tokens=1024`, `max_num_seqs=8`,
and four AIME prompts so they fit on a single 24 GiB GPU. Increase these values
with the corresponding CLI flags for throughput experiments. The script also
runs a short, untimed warmup using the measured batch size because Vegas's
Triton kernels compile lazily. On the first Vegas launch, the engine also logs
`Loading sparse-attention CUDA extensions` while `nvcc` builds its JIT
extensions; this can take several minutes and is cached for later runs.
FA2/Ampere results validate the algorithm and provide a same-hardware baseline
comparison, but are not directly comparable to the paper's FA3/Hopper numbers.

For a long-context speed benchmark, the script can stream LongBench v2 and
truncate the middle of each context to form tokenizer-controlled input-length
buckets. The following pair of commands compares the baseline and Vegas with
two 32K-token inputs and exactly 1K output tokens per request:

```bash
python benchmarks/benchmark_vegas.py \
  --dataset longbench-v2 --longbench-length long \
  --algorithm none --max-model-len 40960 \
  --target-input-tokens 32768 --max-tokens 1024 --ignore-eos \
  --num-prompts 2 --max-num-seqs 2

python benchmarks/benchmark_vegas.py \
  --dataset longbench-v2 --longbench-length long \
  --algorithm vegas --max-model-len 40960 \
  --target-input-tokens 32768 --max-tokens 1024 --ignore-eos \
  --num-prompts 2 --max-num-seqs 2
```

Prefix caching is disabled so repeated inputs cannot make the measured prefill
artificially cheap. The final report separates metrics shared by every
algorithm (workload, throughput, TTFT, prefill latency, and TPOT) from the
speculative-only acceptance and GPU latency breakdown. Vegas additionally
reports verification- and draft-phase latency using two CUDA event pairs per
engine step, without an extra profiling synchronization. The trailing
`RESULT_JSON` line contains the same headline values for scripts.

Compute end-to-end speedup from the two `e2e_output_tokens_per_second` values,
and decode speedup as baseline `mean_tpot_ms` divided by Vegas
`mean_tpot_ms`. Use 8K, 16K, and 32K `--target-input-tokens` values to measure
scaling with context length.

For the paper-style short-input/long-output experiment, use the saturated
continuous-batching benchmark:

```bash
python benchmarks/benchmark_continuous_batching.py

# Use every CodeElo row instead of the 30-question AIME'25 replay.
python benchmarks/benchmark_continuous_batching.py --dataset codeelo
```

The default AIME workload duplicates all 30 rows 32 times and submits all 960
requests concurrently. It uses `max_num_seqs=96`, `max_model_len=37632`, a
32K output cap, and 92% GPU-memory utilization so both AR and Vegas fit on a
24 GiB GPU. Larger-memory GPUs can use `--max-num-seqs 128` together with
`--max-model-len 40960`. As in the paper, the first 400 completions are warmup;
the final 96 are excluded to avoid scheduler-drain bias. Decoder and speculative
Prometheus counters are snapshotted around that stable window. AR runs first,
then Vegas (or `--algorithm streamingllm`) runs with the same workload. The
report includes throughput interval statistics, TTFT, decode latency, TPOT,
E2E latency, acceptance, and draft/verification GPU time.

LongBench-v2 has a separate real prefill/decode-disaggregation benchmark:

```bash
# Requires NIXL and enough memory in both GPU groups. With two large-memory
# GPUs, GPU 0 runs prefill and GPU 1 runs decode.
python benchmarks/benchmark_disaggregated_longbench.py
```

It samples tokenizer-measured 96K-120K inputs, applies the Qwen3 YaRN factor-4
configuration, and transfers KV cache through `NixlConnector`. GPUs are split
evenly between one tensor-parallel prefill server and one tensor-parallel
decode server; an odd extra GPU goes to decode. Only the decode server's
counters and request timings enter the AR/speculative speedup. Qwen3-8B paper
defaults are used for maximum decode batch (4), sparsity (7%), and draft length
(Vegas 9, StreamingLLM 5).

The two-GPU minimum assumes GPUs large enough to hold Qwen3-8B plus a 131K KV
cache. On 24 GiB cards, use at least four GPUs (two-way tensor parallelism per
server), reduce the model/context, or use a deliberate KV-cache quantization
configuration; one 24 GiB GPU per server cannot hold the default BF16 workload.

`--ignore-eos` intentionally forces a fixed amount of decoding and is only for
performance measurement, not LongBench quality evaluation. The initial
LongBench stream may take tens of seconds to open.

## Code Layout

The sparse self-speculative implementation lives under
`vllm/v1/spec_decode/sparse_attn/`, plus a small set of vLLM integration edits.

```text
vllm/v1/spec_decode/sparse_attn/
├── proposer.py                     # SparseAttnProposer: drives the self-speculative draft loop
├── attn_overrider/
│   ├── __init__.py                 # BaseAttnOverrider + build_attention_overrider() dispatch
│   ├── vegas.py                    # VegasAttnOverrider: verification-guided KV selection
│   ├── streamingllm.py             # StreamingLLMAttnOverrider: sink + sliding-window baseline
│   └── utils/
│       ├── varlen_reduce.py        # CUDA kernel: reduce per-query scores/weights -> per-token metric
│       └── varlen_topk.py          # CUDA kernel: variable-length top-k KV selection
```

Integration points (edited vLLM files):

| File | What it does for Vegas |
| --- | --- |
| `vllm/config/speculative.py` | Adds the `sparse_attn_*` config fields (`method="sparse_attn"`) |
| `vllm/v1/worker/gpu_model_runner.py` | Constructs and drives `SparseAttnProposer`; CUDA-graph wiring |
| `vllm/v1/spec_decode/utils.py` | Shared spec-decode helpers used by the proposer |
| `vllm/v1/sample/rejection_sampler.py` | Accept/reject of drafted tokens |
| `vllm/v1/core/sched/scheduler.py` | Reserves lookahead slots so KV pages are allocated correctly for the draft tokens |
| `benchmarks/benchmark_vegas.py` | Small offline smoke benchmark |
| `benchmarks/benchmark_continuous_batching.py` | Saturated AIME'25/CodeElo serving benchmark |
| `benchmarks/benchmark_disaggregated_longbench.py` | Decode-only LongBench-v2 P/D benchmark |

The verification-guided selection relies on a modified attention kernel that
collects the per-token attention logits (raw pre-softmax QK scores, and
optionally the log-sum-exp for weight rematerialization) as a byproduct of the
verify pass. This is exposed through the `scores` parameter of
[`flash_attn_varlen_func`](vllm/vllm_flash_attn/flash_attn_interface.py) and
implemented in our companion
[flash-attention fork](https://github.com/npz7yyk/vllm-flash-attn). The companion
fork implements this path natively for FA3; this repository additionally
patches both the regular and paged/split-KV FA2 forward kernels to write the
same first/last-query score buffer without a second QK pass.

## Acknowledgements

This project is built on [vLLM](https://github.com/vllm-project/vllm) and was
forked from [Vegas](https://github.com/platformxlab/vegas) as its initial
baseline.
