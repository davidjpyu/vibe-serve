# Objective — Qwen3.5-9B inference server

Maximize **output token throughput (tok/s)** on a single H200 while keeping accuracy within the accuracy checker's tolerance. Build an OpenAI-compatible `/v1/chat/completions` and `/v1/completions` server.

## Notes

- **Hybrid architecture causal LM** (`Qwen/Qwen3.5-9B`, HF revision pinned in `reference/meta.json`).
  The model interleaves linear-attention (Gated DeltaNet-style) layers with full-attention
  layers in a 3:1 pattern (8 blocks × (3 linear + 1 full) = 32 layers). See `text_config.layer_types`
  in the model's `config.json` for the exact per-layer schedule, and `linear_*` / `mtp_*` /
  `mrope_*` fields for the conv-kernel, head-count, and rotary parameters.
- Hopper-class hardware assumed (H200 SM9.0, 144 GB HBM3e).
- Implement model layers explicitly (own linear-attention / full-attention / MLP / norm / RoPE);
  use `transformers` only as a utility for config / tokenizer / weight loading. The accuracy
  checker compares your output to a `transformers` reference, so you have a working spec to
  pattern-match against — but the forward path in your server must be your own code.
- BF16 is the natural baseline (matches the checkpoint dtype). FP16 is acceptable; quantization
  is an optimization, not a prerequisite.
- The model is **text-only** for this benchmark's purposes — the benchmark harness only sends
  text prompts. You may skip the vision encoder entirely if your scheduler/forward path doesn't
  need it; `text_config` is the relevant config sub-block. Do NOT remove the vision-related
  config fields from `config.json` (they're load-bearing for tokenizer / chat template).
- Multi-token-prediction (`mtp_num_hidden_layers=1`) is optional — you can ignore the MTP head
  for greedy decode and only use the main LM head. If you wire MTP up correctly it's a free
  speedup, but accuracy must still match HF reference token-for-token.
- Benchmark harness drives the server; it reports req/s and tok/s. Prefer tok/s as the primary
  metric.

## Warning

This model is materially harder than the Llama-3.1-8B example. The Gated DeltaNet kernel is
not in standard PyTorch — you'll need to implement it (e.g. as a Triton kernel or a chunked
recurrence over `linear_conv_kernel_dim=4`) or import a library that provides it
(`flash-linear-attention`, `fla`). Budget round 1 entirely for getting accuracy right; perf
optimization (continuous batching, FlashAttention on the full-attention layers, CUDA graphs)
should land in later rounds.
