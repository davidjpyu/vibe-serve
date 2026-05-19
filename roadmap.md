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

## Round-1 baseline & profile recap

- **Baseline (round 1)**: 36.38 aggregate tok/s @ rate=1, max_tokens=128.
  TPOT median 27.3 ms, TTFT P99 22.35s (TTFT is huge because the asyncio.Lock
  fully serializes requests — request N waits for request N-1 to finish all
  128 tokens before its prefill even runs).
- **Profile verdict**: LAUNCH-BOUND. Decode step = 31.98 ms wall / 5.7 ms GPU
  busy = 18% GPU utilization. 1,468 kernels/step. 988k cudaLaunchKernel
  calls in the 25s window. GPU-busy across the whole benchmark = 16.7%.
  Zero CUDA graphs active. SDPA pinned to MATH backend for acc parity.
- **Bottleneck order from the profile**: (1) launch overhead → CUDA graphs;
  (2) cuBLAS GEMM dominance + softmax+matmul split in SDPA[MATH] →
  FlashAttention; (3) request serialization (9 idle gaps of 10-11 ms between
  requests) → continuous batching. The profile's suggested execution order
  was G → A → B → D → E → C → F → H (token-tape-on-device → CUDA graphs →
  FlashAttention → continuous batching → fused norms → cat-free RoPE …).

## Strategy update after round-1 profile

The optimization-floor priority *order* (continuous batching → attention
kernel → CUDA graphs) and the profile's suggested *order* (CUDA graphs →
FA → continuous batching) disagree, because they optimize different things:
the floor maximizes aggregate tok/s on a multi-request workload, while the
profile's single-stream view maximizes per-stream tok/s. The headline metric
is **aggregate tok/s**, and the benchmark drives **Poisson arrivals at
rate≥1**, so the floor's ordering is the right one for *this* objective.

Continuous batching first is also a structural change — it forces the KV
cache, the scheduler, and the per-request streaming queues into the shapes
that FlashAttention's varlen and CUDA-graph-bucketed decode will need
anyway. Doing CUDA graphs first means re-doing them with batch ≥ 1 a round
later. So we keep the floor ordering: CB → FA → CUDA graphs → workload-
specific wins.

## Major

- **M1. Baseline FastAPI server with hand-written Llama** — `done` —
  rounds: 1. Achieved 36.38 tok/s. Accuracy 14/14. Both endpoints stream.
  Static per-layer KV cache (batch=1). No `torch.cat` against the cache.

- **M2. Static KV cache, fp16/bf16, no per-token allocs** — `done` —
  rolled into M1. KV cache is preallocated `(1, num_kv_heads, 4096,
  head_dim)` per layer with in-place slice writes — confirmed by judge.
  One nit remaining: `_rotate_half` still uses `torch.cat` (37k calls/window
  per profile) — track as Minor m1.

- **M3. Continuous batching across in-flight requests** — `done` —
  rounds: 1 (round 2). Scheduler in `main.py` is the sole GPU consumer,
  decode batched over slots, asyncio.Lock removed. Benchmark went from
  36.38 → 243.4 tok/s (6.7×); accuracy still 14/14. TTFT P50 collapsed
  from ~11s to ~46ms.

- **M4. Replace manual attention with FlashAttention / FlashInfer** —
  `in_progress` (round 3) — rounds: 0. *Why*: serving path still calls
  `F.scaled_dot_product_attention` under `sdpa_kernel([SDPBackend.MATH])`
  which materializes the full attention matrix and runs separate
  softmax+matmul kernels (round-1 profile: ~3 ms/decode-step attention
  cost). FlashAttention's `flash_attn_with_kvcache` is purpose-built for
  the slot-pool design we already have: it takes `(B, max_seqlen, H_kv,
  D)` cache tensors, per-row `cache_seqlens`, optional `cache_batch_idx`
  for slot remapping, and writes new K/V into the cache in-place. Will
  also subsume the K/V cache write (today done via advanced-index
  scatter) and *can* fuse RoPE if we pass `rotary_cos/rotary_sin` —
  optionally closes Minor m1. Accuracy gate stays alive by keeping
  SDPA[MATH] reachable for `VibeServeModel.generate()` (acc_checker's
  entry point), with FA only on the scheduler's batched-decode path.

- **M5. CUDA graphs on the decode path** — `todo` — rounds: 0. *Why*:
  profile is unambiguous: 82% of decode wall is launch-gap. After M3+M4
  the bucketed decode-step is the natural unit to capture. Expected
  2-3× on top of M3+M4.

- **M6. Speculative decoding (EAGLE3 or draft model)** — `todo` —
  rounds: 0. *Why*: at low Poisson rate the workload is essentially
  single-batch; spec decode trades verifier compute for fewer decode
  steps. Defer until M3-M5 land.

## Minor

- **m1. Replace `_rotate_half`'s `torch.cat` with an alloc-free rotation**
  — `todo`. Profile shows 36,928 `CatArrayBatchedCopy` launches per 25s
  window. Cheap fix, do it inside whichever Major touches RoPE next.

## Done

- M1 (baseline server, round 1).
- M2 (static KV cache, round 1).
- M3 (continuous batching, round 2): 243.4 tok/s, acc 14/14.

## Parked

(none yet)

## Abandoned

(none yet)
