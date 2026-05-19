# Roadmap — Qwen3.5-9B inference server (H200, single-batch tok/s)

## Architecture snapshot

- Qwen3.5-9B hybrid causal LM. 32 layers, pattern is 3×linear-attention + 1×full-attention,
  repeated 8× (`full_attention_interval=4`).
- Linear-attention layers: Gated DeltaNet (GDN)-style. `linear_num_key_heads=16`,
  `linear_num_value_heads=32`, `linear_key_head_dim=128`, `linear_value_head_dim=128`,
  `linear_conv_kernel_dim=4` (causal conv-1d pre-projection cache).
- Full-attention layers: `num_attention_heads=16`, `num_key_value_heads=4` (GQA),
  `head_dim=256`, `attn_output_gate=true`, partial RoPE
  (`partial_rotary_factor=0.25` → only first 64 dims of head are rotated),
  `mrope_interleaved=true`, `rope_theta=1e7`.
- hidden=4096, intermediate=12288, SwiGLU MLP, RMSNorm eps=1e-6, vocab=248320,
  `tie_word_embeddings=false`.
- Optional MTP head: `mtp_num_hidden_layers=1`. Skip for round 1; can wire as
  speculative-decode draft later.
- Vision tower exists in checkpoint — text-only path is sufficient for this benchmark; weights
  must be skippable but config fields stay (tokenizer / chat template loads them).
- HF reference is loaded with `attn_implementation="eager"` at fp16 in `acc_checker/checker.py`.

## Approach summary

Path on Hopper/H200, framed against the **optimization floor**:

1. **Continuous batching** — benchmark is Poisson-arrival multi-request (`bench/benchmark.py`),
   so continuous batching is in scope (NOT a single-batch contract). Plan: in-scope after R1.
2. **Attention kernel** — full-attention layers are GQA with non-standard `attn_output_gate`
   and partial RoPE; FlashAttention-2 / FlashInfer batched-decode planned once eager path is
   correct. Linear-attention layers want `fla` (flash-linear-attention) Triton kernels — these
   ARE the attention kernel for those layers.
3. **CUDA graphs** — decode shapes are bucketable on (batch, max_seq_len). Plan: after the
   FlashInfer / `fla` kernels land so the captured graph is the fast path, not the eager one.

Only after these three are in place do we move to workload-specific optimizations
(MTP-speculative draft, quantization, prefix caching, …).

## Major

- **M1: Accurate baseline server** — `done` (R1).
  `main.py` exports `VibeServeModel` with hand-written full-attention (GQA 16:4, head_dim=256,
  partial RoPE 64/256, sigmoid output-gate) and GDN linear-attention (FLA kernels:
  `chunk_gated_delta_rule` prefill, `fused_recurrent_gated_delta_rule` decode). Caches are
  pre-allocated (`FullAttnCache` writes K/V in-place via `.copy_`; `GDNCache` holds fp32
  recurrent state + fp16 conv state). FastAPI `POST /v1/completions` returns SSE. 14/14
  EXACT token matches vs HF reference fp16 greedy. Bench sanity: 2/2 reqs, 0 errors.
  Carry-over caveats for R2: SSE is **emulated** (single text frame after full generation
  → broken TTFT/per-token streaming), and the server is **single-batch** behind an
  asyncio.Lock (~0.9 tok/s observed) — both addressed by M2.

- **M2: Continuous batching + per-token SSE streaming** — `in_progress`, R2.
  Decode many requests in lockstep on the GPU, one token per step, instead of one-request-at-a-time
  under a lock. Per-request KV slots for the 8 full-attn layers and per-request (recurrent_state,
  conv_state) slots for the 24 GDN layers. Single async step-engine task pulls newly arrived
  requests, prefills them (one at a time or in a batched prefill), assigns them to free slots,
  then runs a batched decode step that advances every active sequence by one token. Per-step
  outputs feed per-request asyncio.Queues so SSE writers emit one token immediately as it is
  produced. Headline metric: aggregate output tok/s on the Poisson workload.
  Why: this is the floor item that matters most here — the benchmark is multi-request Poisson,
  and round 1 caps at 1/(per-request generate time) because of the lock + emulated streaming.

- **M3: Attention kernel — FlashAttention/FlashInfer on full-attention layers** — `todo`, R3.
  Replace eager attention with FA2 / FlashInfer batched-decode (8 full-attn layers, GQA 16:4,
  head_dim=256, partial RoPE applied to first 64 dims). Folds RoPE+attention into one kernel
  group; skips materializing the full (B, H, T, T) attention matrix.
  Why: full-attention layers dominate per-token math once linear-attn kernels are tight.

- **M4: GDN Triton kernel (`fla` integration tightened)** — `todo`, R3-R4.
  Confirm `fla.ops.gated_delta_rule` (chunked or fused-recurrent path) is on the hot decode
  loop with no Python-side scalar ops between layers. Add the `causal_conv1d` CUDA path if
  the Triton conv-1d shows up in the profile.
  Why: 24 of 32 layers are linear-attn — kernel quality there dominates.

- **M5: CUDA graphs on decode** — `todo`, R4-R5.
  Capture per-batch-size buckets {1, 2, 4, 8, 16, 32} of the decode step. Replay covers both
  linear-attn (`fla` fused-recurrent) and full-attn (FlashInfer decode). Requires KV/GDN-state
  buffers to be persistent and indexed (no allocations per step).
  Why: at single-step decode on H200, host-side launch overhead is the dominant cost once
  kernels are tight.

## Minor

(none yet — minors will appear as profiler findings post-baseline)

## Done

- **M1** — Accurate baseline server. R1 commit `round-1-retry-1-judge`. 14/14 accuracy.

## Parked

(none yet)

## Abandoned

(none yet)
