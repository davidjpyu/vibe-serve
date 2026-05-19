# Roadmap

## Context

- **Model**: Llama-3.1-8B-Instruct (32 layers, hidden=4096, MLP=14336, GQA 32/8 heads).
- **Hardware**: single H200 (Hopper, ~140 GB HBM3e, ~4.8 TB/s memory BW).
- **Workload**: benchmark drives `/v1/completions` streaming SSE at Poisson rate
  (default rate=1, but can run higher), `max_tokens` per request (default 128),
  greedy (temperature=0). Server must also expose `/v1/chat/completions` for the
  OpenAI surface to be complete.
- **Headline metric**: aggregate output tok/s reported by `bench/benchmark.py`.
- **Implementation constraint**: own attention / MLP / norm / RoPE; use
  `transformers` only for config / tokenizer / weight loading.
- **Accuracy gate**: greedy outputs must match HF eager-attention reference
  exactly across 14 prompts (see `acc_checker/checker.py`).

## Strategy

The benchmark is **multi-request** (Poisson arrivals) so continuous batching
applies; H200 has FlashAttention 3 / FlashInfer support; decode-shape kernels
are bucketable so CUDA graphs apply. All three optimization-floor items are
in scope. After the floor is in place, speculative decoding (EAGLE3 / draft
model) is the obvious next win on a single-batch tail (low Poisson rate).

## Major

- **M1. Baseline FastAPI server with hand-written Llama** — `todo` — rounds: 0.
  *Why*: nothing exists yet. Need a correct, working baseline that passes the
  accuracy checker and serves both `/v1/completions` and `/v1/chat/completions`
  with streaming SSE. Without this every subsequent perf optimization has
  nowhere to land.

- **M2. Static KV cache, fp16/bf16, no per-token allocs** — `todo` — rounds: 0.
  *Why*: a naive decode loop reallocates KV each step (huge mem-BW waste); a
  preallocated KV cache + in-place writes is the precondition for both
  FlashAttention/FlashInfer decode and CUDA graphs. Usually rolled into the
  baseline if the baseline is written carefully — keep this as an explicit
  Major in case round-1 ships with `torch.cat`.

- **M3. Replace manual attention with FlashAttention / FlashInfer** —
  `todo` — rounds: 0. *Why*: hand-rolled SDPA in fp16 with separate softmax
  and matmul kernels wastes ~30-50% of decode time vs. fused attention.
  FlashInfer's batched-decode path is purpose-built for GQA + paged/static KV
  with low launch overhead.

- **M4. Continuous batching across in-flight requests** — `todo` — rounds: 0.
  *Why*: benchmark drives Poisson arrivals; with naive per-request decode the
  server is idle for most of the wall-clock. A continuous-batching scheduler
  (admit new requests into the decode batch each step) is the single biggest
  multi-request win.

- **M5. CUDA graphs on the decode path** — `todo` — rounds: 0. *Why*: at
  batch=N decode each step launches ~150+ kernels; on H200 launch overhead
  dominates per-token latency at small batch. Capturing the decode step into a
  CUDA graph (one per bucketed batch size) eliminates launch overhead and
  typically gives 20-40% on top of FlashAttention.

- **M6. Speculative decoding (EAGLE3 or draft model)** — `todo` — rounds: 0.
  *Why*: at low Poisson rate the workload is essentially single-batch, where
  arithmetic intensity is fundamentally memory-bound. Speculative decoding
  trades verifier compute for fewer decode steps — the canonical win in this
  regime. Defer until M1-M5 are profiler-verified.

## Minor

(none yet — track here as they appear)

## Done

(none yet)

## Parked

(none yet)

## Abandoned

(none yet)
