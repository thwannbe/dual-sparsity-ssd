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

To compare AR and sparse speculative decoding in a single command without
continuous request admission, use the closed-batch benchmark:

```bash
python benchmarks/benchmark_vegas_fixed_batch.py \
  --algorithm vegas --batch-size 4 --num-samples 16 \
  --max-model-len 8192 --max-tokens 1024
```

It runs AR first and then Vegas (or `--algorithm streamingllm`) with the same
prompts and sampling parameters. `--num-samples` must be a multiple of
`--batch-size`; the example runs four independent closed batches. Each batch
finishes before the next is admitted, so there is no backlog from which the
scheduler can refill a completed slot. EOS is respected (`ignore_eos=False`),
so the active batch may naturally shrink as requests finish. The report shows
mean, standard deviation, and median batch latency as well as aggregate and
paired-batch speedups. It also compares AR and speculative wall-clock prefill,
decode/request, TPOT, and request E2E latency. GPU-event profiling is enabled by
default for the speculative run and adds lightweight verification- and
draft-phase timings. It records two event pairs per engine step and does not add
an explicit profiling synchronization. The default `--temperature 0` uses
greedy decoding and checks exact generated-token equality for every sample. A
nonzero temperature can be supplied explicitly for distributional sampling
comparisons.
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
| `benchmarks/benchmark_vegas.py` | End-to-end example / benchmark |

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
