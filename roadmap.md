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

- **M2: Continuous batching + per-token SSE streaming** — `done` (R2).
  `StepEngine` async task owns the GPU; alternates per-request prefill admit + batched decode
  over contiguous active prefix [0:B). Per-layer pooled caches: 8 × full-attn K/V pool
  `(MAX_BATCH=16, num_kv_heads=4, MAX_LEN=4096, head_dim=256)` fp16; 24 × GDN recurrent_state
  `(16,32,128,128)` fp32 + conv_state `(16, conv_dim=8192, 3)` fp16. SSE: one token per frame.
  Bench at `--rate 8 --num-requests 32 --max-tokens 64`: 264.9 tok/s aggregate
  (≈294× R1), 14/14 accuracy preserved.

- **M3: Attention kernel — FlashAttention on full-attention layers** — `done` (R3, via SDPA fallback).
  FA2/FA3 PyPI wheels do not target torch 2.12+cu13 and the sandbox is driver-only (no CUDA
  toolkit for from-source build), so the implementation landed on
  `F.scaled_dot_product_attention(..., enable_gqa=True)` — explicitly documented as the
  fallback in R3's task. The architectural wins from M3 are present: pool is NHD layout,
  GQA grouping is handled inside the kernel (no `repeat_interleave`), the explicit
  `matmul → mask → fp32 softmax → matmul` chain is gone. Decode uses
  `sdpa_kernel([SDPBackend.MATH])` to dodge cuDNN per-shape replanning on the growing K-cache.
  Headline tok/s: 244 tok/s mean (3 seeds) at `--rate 8 --num-requests 32 --max-tokens 64`
  vs R2's 264.9 single-sample (≈0.92×, within Poisson noise); 297 tok/s at `--rate 16` >
  R2's 265. Accuracy still 14/14.

- **M4: GDN Triton kernel (`fla` integration tightened)** — `todo`, R3-R4.
  Confirm `fla.ops.gated_delta_rule` (chunked or fused-recurrent path) is on the hot decode
  loop with no Python-side scalar ops between layers. Add the `causal_conv1d` CUDA path if
  the Triton conv-1d shows up in the profile.
  Why: 24 of 32 layers are linear-attn — kernel quality there dominates.

- **M5: CUDA graphs on decode** — `in_progress`, R4.
  Capture per-(B, kv_len_bucket) decode-step graphs. Buckets: B ∈ {1, 2, 4, 8, 16}
  (MAX_BATCH=16 from R2), kv_len ∈ {128, 256, 512, 1024, 2048, 4096}. Pools are already
  persistent and written in-place (R2/R3), so the only new persistent tensors needed are the
  per-step input/output staging buffers (`hidden_in[B,1,hidden]`, `lengths_after[B]`,
  `next_token[B]`). SDPA decode is pinned to `SDPBackend.MATH` (R3) or `EFFICIENT_ATTENTION`
  — both graph-capturable. fla's `fused_recurrent_gated_delta_rule` is Triton with fixed
  shapes per bucket, also graph-capturable. Per-row K-masking inside the graph: precompute
  a persistent `arange[max_kv_len]`, build `mask = arange < lengths_after.unsqueeze(...)`
  at replay time.
  Why: profile shows 1377 ms CPU vs 119 ms GPU for 16 decode steps (~2048 launches/token,
  ~78% GPU-idle per step). Eliminating launch overhead is the single largest remaining lever;
  predicted move from ~250 to ~600-900 tok/s aggregate.

## Minor

(none yet — minors will appear as profiler findings post-baseline)

## Done

- **M1** — Accurate baseline server. R1. 14/14 accuracy.
- **M2** — Continuous batching + per-token SSE. R2. 264.9 tok/s (≈294× R1), 14/14 accuracy.
- **M3** — SDPA `enable_gqa=True` on full-attn (FA fallback path). R3. 244-297 tok/s, 14/14.

## Parked

(none yet)

## Abandoned

(none yet)
