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

- **M1: Accurate baseline server (R1 active)** — `in_progress`, round 1.
  Build `main.py` exporting `VibeServeModel.from_pretrained(model_dir, device, dtype)` and
  `.generate(input_ids, max_new_tokens=N)`; wire FastAPI `POST /v1/completions` with
  streaming SSE per `serving-systems/references/tooling/openai-api.md`. Hand-implement the
  forward path: RMSNorm, partial+mrope RoPE, gated GQA full-attention, GDN linear-attention,
  SwiGLU MLP, LM head. Use `fla` (flash-linear-attention) for the GDN recurrence kernel if
  available, else chunked Python recurrence as a correctness fallback. Goal: accuracy checker
  passes token-for-token vs HF reference at fp16 greedy.
  Why: unlocks the loop — every later round depends on a correct forward path.

- **M2: Continuous batching for decode** — `todo`, planned R2-R3.
  Batch in-flight requests on the decode loop with per-request KV cache slots (full-attn) and
  per-request GDN state slots (`(state, conv_state)`). The benchmark sends Poisson arrivals,
  so wall-clock throughput is gated by how many sequences we can decode per step.
  Why: workload is multi-request; single-batch decode caps throughput at ~ 1/(decode time).

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

(none yet)

## Parked

(none yet)

## Abandoned

(none yet)
