"""FastAPI inference server for Llama-3.1-8B-Instruct on H200 with continuous batching.

- Hand-written model layers (RMSNorm, RoPE with Llama-3 scaling, GQA attention,
  SwiGLU MLP, decoder stack); transformers used only for tokenizer + config +
  weight loading.
- Per-layer KV cache: (N_SLOTS, num_kv_heads, max_cache_len, head_dim), written
  in place. No torch.cat in the decode path.
- Continuous batching: a background daemon thread is the sole GPU consumer.
  HTTP handlers submit a Job (via thread-safe queue.Queue) and drain tokens
  from a per-request asyncio.Queue. No asyncio.Lock around forward passes.
- Attention backend: SDPA with explicit MATH kernel for eager-equivalent
  semantics (accuracy gate).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from safetensors.torch import load_file as safetensors_load
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoTokenizer

# ---------------------------------------------------------------------------
# Model layers — explicit implementations
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return self.weight * x32.to(dtype)


def _compute_llama3_inv_freq(
    head_dim: int,
    rope_theta: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
    device: torch.device,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )
    smooth_factor = (
        original_max_position_embeddings / wavelen - low_freq_factor
    ) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


def _build_cos_sin_cache(
    max_seq_len: int,
    head_dim: int,
    rope_theta: float,
    rope_scaling: dict | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
        inv_freq = _compute_llama3_inv_freq(
            head_dim=head_dim,
            rope_theta=rope_theta,
            factor=float(rope_scaling["factor"]),
            low_freq_factor=float(rope_scaling["low_freq_factor"]),
            high_freq_factor=float(rope_scaling["high_freq_factor"]),
            original_max_position_embeddings=int(
                rope_scaling["original_max_position_embeddings"]
            ),
            device=device,
        )
    else:
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
    positions = torch.arange(0, max_seq_len, dtype=torch.float32, device=device)
    freqs = positions[:, None] * inv_freq[None, :]  # (max_seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding.

    q, k: (bsz, num_heads, seq_len, head_dim)
    cos, sin: (bsz, seq_len, head_dim) — per-row positions
    """
    cos = cos.unsqueeze(1)  # (bsz, 1, seq_len, head_dim)
    sin = sin.unsqueeze(1)
    q_emb = (q * cos) + (_rotate_half(q) * sin)
    k_emb = (k * cos) + (_rotate_half(k) * sin)
    return q_emb, k_emb


class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,    # (B, L, hidden)
        cos: torch.Tensor,              # (B, L, head_dim)
        sin: torch.Tensor,              # (B, L, head_dim)
        k_cache: torch.Tensor,          # (N_SLOTS, num_kv_heads, max_cache_len, head_dim)
        v_cache: torch.Tensor,          # same
        slot_ids: torch.Tensor,         # (B,) long
        cache_starts: torch.Tensor,     # (B,) long — positions to write into
        attn_mask: torch.Tensor | None, # additive mask, (B, 1, 1, max_kv) or None
        max_kv: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, num_heads, L, head_dim)
        k = (
            self.k_proj(hidden_states)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = apply_rope(q, k, cos, sin)

        # In-place write to preallocated KV cache. No torch.cat.
        if L == 1:
            # Decode: scatter via advanced indexing
            # slot_ids: (B,), cache_starts: (B,)
            # k: (B, num_kv_heads, 1, head_dim) -> squeeze L dim to (B, num_kv_heads, head_dim)
            k_cache[slot_ids, :, cache_starts, :] = k[:, :, 0, :]
            v_cache[slot_ids, :, cache_starts, :] = v[:, :, 0, :]
        else:
            # Prefill (B==1 in this round): slice assignment
            for b in range(B):
                s = int(slot_ids[b].item())
                cs = int(cache_starts[b].item())
                k_cache[s, :, cs : cs + L, :] = k[b]
                v_cache[s, :, cs : cs + L, :] = v[b]

        # Read the visible prefix [0:max_kv] of each row's slot.
        # Advanced indexing on dim 0 with slot_ids produces a contiguous (B, ...) copy.
        k_full = k_cache[slot_ids, :, :max_kv, :]  # (B, num_kv_heads, max_kv, head_dim)
        v_full = v_cache[slot_ids, :, :max_kv, :]

        with sdpa_kernel([SDPBackend.MATH]):
            attn_out = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=is_prefill,  # only the prefill case (q_len==kv_len)
                scale=self.scaling,
                enable_gqa=True,
            )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        return self.o_proj(attn_out)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_ids: torch.Tensor,
        cache_starts: torch.Tensor,
        attn_mask: torch.Tensor | None,
        max_kv: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(
            x, cos, sin, k_cache, v_cache,
            slot_ids, cache_starts, attn_mask, max_kv, is_prefill,
        )
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        return residual + x


class LlamaInner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


# ---------------------------------------------------------------------------
# VibeServeModel — owns weights, KV cache buffers, RoPE cache
# ---------------------------------------------------------------------------


class VibeServeModel(nn.Module):
    """Llama-3.1 forward + greedy generation, batched continuous-decode-ready.

    KV cache is shape (N_SLOTS, num_kv_heads, max_cache_len, head_dim), one
    pair (K, V) per decoder layer. Slots are owned by the scheduler in serving;
    `generate()` uses slot 0 in isolation for the accuracy checker.
    """

    def __init__(
        self,
        config,
        device: torch.device,
        dtype: torch.dtype,
        max_cache_len: int = 4096,
        num_slots: int = 16,
    ):
        super().__init__()
        self.config = config
        self.device_ = device
        self.dtype_ = dtype
        self.max_cache_len = max_cache_len
        self.num_slots = num_slots

        self.model = LlamaInner(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        for i in range(self.num_layers):
            self.register_buffer(
                f"_k_cache_{i}",
                torch.zeros(
                    num_slots, self.num_kv_heads, max_cache_len, self.head_dim,
                    device=device, dtype=dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                f"_v_cache_{i}",
                torch.zeros(
                    num_slots, self.num_kv_heads, max_cache_len, self.head_dim,
                    device=device, dtype=dtype,
                ),
                persistent=False,
            )

        # transformers 5.x stores rope_theta inside `rope_parameters`; older
        # versions expose it as a top-level attribute and use `rope_scaling`.
        rope_params = getattr(config, "rope_parameters", None) or {}
        rope_scaling = getattr(config, "rope_scaling", None) or rope_params or None
        rope_theta = (
            rope_params.get("rope_theta")
            or (rope_scaling and rope_scaling.get("rope_theta"))
            or getattr(config, "rope_theta", None)
            or 10000.0
        )
        rope_theta = float(rope_theta)
        cos, sin = _build_cos_sin_cache(
            max_seq_len=max_cache_len,
            head_dim=self.head_dim,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | os.PathLike,
        device: str | torch.device,
        dtype: torch.dtype,
        max_cache_len: int = 4096,
        num_slots: int = 16,
    ) -> "VibeServeModel":
        model_dir = str(model_dir)
        device = torch.device(device)
        config = AutoConfig.from_pretrained(model_dir)

        model = cls(
            config, device=device, dtype=dtype,
            max_cache_len=max_cache_len, num_slots=num_slots,
        )

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            files = sorted(set(index["weight_map"].values()))
        else:
            single = os.path.join(model_dir, "model.safetensors")
            assert os.path.exists(single), f"no weights found under {model_dir}"
            files = [single]

        state_dict: dict[str, torch.Tensor] = {}
        for fname in files:
            path = fname if os.path.isabs(fname) else os.path.join(model_dir, fname)
            shard = safetensors_load(path, device=str(device))
            for k, v in shard.items():
                state_dict[k] = v.to(dtype)

        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        real_missing = [
            k for k in missing
            if not (
                k.endswith("cos_cache")
                or k.endswith("sin_cache")
                or "_k_cache_" in k
                or "_v_cache_" in k
            )
        ]
        if real_missing:
            raise RuntimeError(f"missing weights: {real_missing[:5]} ...")
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Forward (batched-capable)
    # ------------------------------------------------------------------

    def _forward_inner(
        self,
        input_ids: torch.Tensor,       # (B, L)
        slot_ids: torch.Tensor,        # (B,)
        cache_starts: torch.Tensor,    # (B,)
        is_prefill: bool,
    ) -> torch.Tensor:
        """Run the full stack. Writes K/V into each row's slot at
        [cache_starts[b], cache_starts[b]+L) and returns last-token logits
        of shape (B, vocab).
        """
        B, L = input_ids.shape
        device = self.device_

        # Per-row positions: cache_starts[b] + [0..L-1]
        offsets = torch.arange(L, device=device, dtype=torch.long)
        position_ids = cache_starts.unsqueeze(1) + offsets.unsqueeze(0)  # (B, L)
        cos = self.cos_cache[position_ids]  # (B, L, head_dim)
        sin = self.sin_cache[position_ids]

        # kv_lens (visible after writing the new L tokens): cache_starts + L
        kv_lens = cache_starts + L  # (B,)
        max_kv = int(kv_lens.max().item())

        if is_prefill:
            attn_mask = None  # use is_causal=True
        else:
            # Decode (L==1): need additive mask covering padding positions.
            # mask[b, 0, 0, j] = -inf if j >= kv_lens[b], else 0
            col_idx = torch.arange(max_kv, device=device)  # (max_kv,)
            mask_bool = col_idx.unsqueeze(0) >= kv_lens.unsqueeze(1)  # (B, max_kv)
            attn_mask = torch.zeros(B, 1, 1, max_kv, device=device, dtype=self.dtype_)
            attn_mask.masked_fill_(
                mask_bool.unsqueeze(1).unsqueeze(1), float("-inf")
            )

        x = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            k_cache = self._buffers[f"_k_cache_{i}"]
            v_cache = self._buffers[f"_v_cache_{i}"]
            x = layer(
                x, cos, sin, k_cache, v_cache,
                slot_ids, cache_starts, attn_mask, max_kv, is_prefill,
            )
        x = self.model.norm(x)
        # logits for the last token of each row
        last_hidden = x[:, -1:, :]  # (B, 1, hidden)
        return self.lm_head(last_hidden).squeeze(1)  # (B, vocab)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        eos_token_ids: list[int] | None = None,
        slot_id: int = 0,
    ) -> torch.Tensor:
        """Greedy generation using slot `slot_id` (default 0). Returns
        (1, prompt_len + generated_len). Bit-identical to the round-1
        single-stream path for batch=1 because no masking/padding is applied.
        """
        if eos_token_ids is None:
            cfg_eos = getattr(self.config, "eos_token_id", None)
            if isinstance(cfg_eos, list):
                eos_token_ids = list(cfg_eos)
            elif cfg_eos is not None:
                eos_token_ids = [int(cfg_eos)]
            else:
                eos_token_ids = []

        input_ids = input_ids.to(self.device_, dtype=torch.long)
        prompt_len = input_ids.shape[1]
        if prompt_len + max_new_tokens > self.max_cache_len:
            raise ValueError(
                f"prompt_len + max_new_tokens ({prompt_len + max_new_tokens}) "
                f"exceeds max_cache_len ({self.max_cache_len})"
            )

        slot_ids = torch.tensor([slot_id], device=self.device_, dtype=torch.long)
        cache_starts = torch.tensor([0], device=self.device_, dtype=torch.long)

        # Prefill
        logits = self._forward_inner(
            input_ids, slot_ids=slot_ids, cache_starts=cache_starts, is_prefill=True,
        )
        next_token = int(logits.argmax(dim=-1).item())
        generated = [next_token]
        cache_count = prompt_len

        for _ in range(max_new_tokens - 1):
            if next_token in eos_token_ids:
                break
            nt = torch.tensor([[next_token]], device=self.device_, dtype=torch.long)
            cs = torch.tensor([cache_count], device=self.device_, dtype=torch.long)
            logits = self._forward_inner(
                nt, slot_ids=slot_ids, cache_starts=cs, is_prefill=False,
            )
            cache_count += 1
            next_token = int(logits.argmax(dim=-1).item())
            generated.append(next_token)

        gen_tensor = torch.tensor([generated], device=self.device_, dtype=torch.long)
        return torch.cat([input_ids, gen_tensor], dim=1)


# ---------------------------------------------------------------------------
# Continuous-batching scheduler
# ---------------------------------------------------------------------------

EOS_TOKEN_IDS: tuple[int, ...] = (128001, 128008, 128009)

# Sentinel marker pushed to a request's token queue when generation ends.
# Payload: ("done", finish_reason)


@dataclass
class Job:
    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    token_queue: asyncio.Queue
    event_loop: asyncio.AbstractEventLoop

    # Filled by scheduler
    slot_id: int | None = None
    cache_count: int = 0
    n_generated: int = 0
    last_token: int = -1
    finished: bool = False
    finish_reason: str = "length"


class Scheduler:
    """Background daemon thread; sole GPU consumer. Maintains a slot pool
    and runs prefill (one per tick) + batched decode (all active slots).
    """

    def __init__(self, model: VibeServeModel, num_slots: int):
        self.model = model
        self.num_slots = num_slots
        self.free_slots: list[int] = list(range(num_slots))
        self.active: list[Job] = []
        self.wait_queue: queue.Queue[Job] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, daemon=True, name="cb-sched")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def submit(self, job: Job) -> None:
        self.wait_queue.put(job)

    # --- internal -----------------------------------------------------

    def _push_token(self, job: Job, token: int) -> None:
        job.event_loop.call_soon_threadsafe(job.token_queue.put_nowait, token)

    def _finalize(self, job: Job) -> None:
        job.event_loop.call_soon_threadsafe(
            job.token_queue.put_nowait, ("done", job.finish_reason),
        )
        if job.slot_id is not None:
            self.free_slots.append(job.slot_id)
            job.slot_id = None

    def _loop(self) -> None:
        device = self.model.device_
        while not self.stop_event.is_set():
            # 1. Admit at most one new request per tick
            new_job: Job | None = None
            if self.free_slots:
                try:
                    new_job = self.wait_queue.get_nowait()
                except queue.Empty:
                    new_job = None

            if new_job is not None:
                new_job.slot_id = self.free_slots.pop(0)
                try:
                    self._prefill(new_job)
                except Exception as exc:
                    new_job.finished = True
                    new_job.finish_reason = "stop"
                    self._finalize(new_job)
                    print(f"[scheduler] prefill error: {exc}", flush=True)
                    continue
                if new_job.finished:
                    self._finalize(new_job)
                else:
                    self.active.append(new_job)

            # 2. Decode (batched) if anyone is active
            if not self.active:
                if new_job is None:
                    # idle — brief sleep so we don't spin the CPU
                    time.sleep(0.001)
                continue

            try:
                self._decode_step()
            except Exception as exc:
                print(f"[scheduler] decode error: {exc}", flush=True)
                # On a hard error, finish all in-flight jobs to avoid stalled
                # streams; surface a stop finish_reason.
                for j in self.active:
                    j.finished = True
                    j.finish_reason = "stop"

            # 3. Clean up finished jobs
            still: list[Job] = []
            for j in self.active:
                if j.finished:
                    self._finalize(j)
                else:
                    still.append(j)
            self.active = still

        # Stopped: drain in-flight jobs
        for j in self.active:
            j.finished = True
            j.finish_reason = "stop"
            self._finalize(j)
        self.active = []

    @torch.inference_mode()
    def _prefill(self, job: Job) -> None:
        device = self.model.device_
        prompt_len = len(job.prompt_ids)
        if prompt_len + job.max_tokens > self.model.max_cache_len:
            # Truncate the generation budget so we don't overflow the cache.
            job.max_tokens = max(1, self.model.max_cache_len - prompt_len)

        prompt = torch.tensor(
            [job.prompt_ids], device=device, dtype=torch.long
        )
        slot_ids = torch.tensor([job.slot_id], device=device, dtype=torch.long)
        cache_starts = torch.tensor([0], device=device, dtype=torch.long)

        logits = self.model._forward_inner(
            prompt, slot_ids=slot_ids, cache_starts=cache_starts, is_prefill=True,
        )  # (1, vocab)
        next_token = int(logits.argmax(dim=-1).item())
        job.cache_count = prompt_len
        job.n_generated = 1
        job.last_token = next_token

        if next_token in EOS_TOKEN_IDS:
            job.finished = True
            job.finish_reason = "stop"
            return

        self._push_token(job, next_token)
        if job.n_generated >= job.max_tokens:
            job.finished = True
            job.finish_reason = "length"

    @torch.inference_mode()
    def _decode_step(self) -> None:
        device = self.model.device_
        B = len(self.active)
        slot_ids = torch.tensor(
            [j.slot_id for j in self.active], device=device, dtype=torch.long
        )
        cache_starts = torch.tensor(
            [j.cache_count for j in self.active], device=device, dtype=torch.long
        )
        input_ids = torch.tensor(
            [[j.last_token] for j in self.active], device=device, dtype=torch.long
        )

        logits = self.model._forward_inner(
            input_ids, slot_ids=slot_ids, cache_starts=cache_starts,
            is_prefill=False,
        )  # (B, vocab)
        next_tokens = logits.argmax(dim=-1).tolist()

        for i, job in enumerate(self.active):
            if job.finished:
                continue
            nt = int(next_tokens[i])
            job.cache_count += 1
            job.n_generated += 1
            job.last_token = nt

            if nt in EOS_TOKEN_IDS:
                job.finished = True
                job.finish_reason = "stop"
                continue
            self._push_token(job, nt)
            if job.n_generated >= job.max_tokens:
                job.finished = True
                job.finish_reason = "length"


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = 0.0
    top_p: float = 1.0
    stop: str | list[str] | None = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = 0.0
    top_p: float = 1.0
    stop: str | list[str] | None = None
    stream: bool = False


STATE: dict[str, Any] = {}


def _model_dir() -> str:
    cand = os.environ.get("MODEL_DIR")
    if cand and Path(cand).exists():
        return cand
    for c in ["/model", "../model", "./model"]:
        if Path(c).exists():
            return str(Path(c).resolve())
    sym = Path("reference/model.symlink_target")
    if sym.exists():
        target = sym.read_text().strip()
        if Path(target).exists():
            return target
    raise RuntimeError("model directory not found; set MODEL_DIR env var")


N_SLOTS_DEFAULT = int(os.environ.get("N_SLOTS", "16"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = _model_dir()
    n_slots = N_SLOTS_DEFAULT
    print(f"[startup] loading model from {model_dir} (slots={n_slots})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = VibeServeModel.from_pretrained(
        model_dir, device="cuda:0", dtype=torch.float16,
        max_cache_len=4096, num_slots=n_slots,
    )
    scheduler = Scheduler(model, num_slots=n_slots)
    scheduler.start()

    STATE["model"] = model
    STATE["tokenizer"] = tokenizer
    name = getattr(model.config, "_name_or_path", None) or "llama-3.1-8b-instruct"
    STATE["model_name"] = name
    STATE["scheduler"] = scheduler
    print("[startup] model + scheduler ready", flush=True)
    try:
        yield
    finally:
        print("[shutdown] stopping scheduler", flush=True)
        scheduler.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": STATE.get("model_name", "llama-3.1-8b-instruct"),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# Per-request streaming helpers
# ---------------------------------------------------------------------------


def _utf8_safe_decode(
    tokenizer, all_ids: list[int], emitted_text_len: int
) -> tuple[str, int]:
    """Decode all_ids and return any suffix not yet emitted, holding back
    incomplete UTF-8 byte sequences (tokenizer renders them as U+FFFD)."""
    full = tokenizer.decode(all_ids, skip_special_tokens=True)
    if "�" in full[emitted_text_len:]:
        return "", emitted_text_len
    return full[emitted_text_len:], len(full)


def _completion_chunk(
    cmpl_id: str, model_name: str, text: str, finish_reason: str | None
) -> str:
    payload = {
        "id": cmpl_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {"text": text, "index": 0, "finish_reason": finish_reason}
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chat_chunk(
    cmpl_id: str, model_name: str, delta_content: str | None,
    finish_reason: str | None,
) -> str:
    delta: dict[str, Any] = {}
    if delta_content is not None:
        delta["content"] = delta_content
    payload = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _submit_and_drain(
    prompt_ids: list[int], max_tokens: int
) -> tuple[asyncio.Queue, Job]:
    """Build a Job and submit it to the scheduler. Returns the (queue, job)."""
    scheduler: Scheduler = STATE["scheduler"]
    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue = asyncio.Queue()
    job = Job(
        request_id=uuid.uuid4().hex,
        prompt_ids=prompt_ids,
        max_tokens=max_tokens,
        token_queue=token_queue,
        event_loop=loop,
    )
    scheduler.submit(job)
    return token_queue, job


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


async def _completion_stream(
    prompt_text: str, max_tokens: int, prompt_ids: list[int]
):
    tokenizer = STATE["tokenizer"]
    model_name = STATE["model_name"]
    cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    token_queue, _job = await _submit_and_drain(prompt_ids, max_tokens)

    emitted_ids: list[int] = []
    emitted_text_len = 0
    finish_reason: str = "length"

    while True:
        item = await token_queue.get()
        if isinstance(item, tuple) and item and item[0] == "done":
            finish_reason = item[1]
            break
        tok = int(item)
        emitted_ids.append(tok)
        new_text, emitted_text_len = _utf8_safe_decode(
            tokenizer, emitted_ids, emitted_text_len
        )
        if new_text:
            yield _completion_chunk(cmpl_id, model_name, new_text, None)

    # Flush any held-back UTF-8 (final decode)
    final_full = tokenizer.decode(emitted_ids, skip_special_tokens=True)
    leftover = final_full[emitted_text_len:]
    if leftover:
        yield _completion_chunk(cmpl_id, model_name, leftover, None)

    yield _completion_chunk(cmpl_id, model_name, "", finish_reason)
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    tokenizer = STATE["tokenizer"]
    if isinstance(req.prompt, list):
        prompt_text = req.prompt[0] if req.prompt else ""
    else:
        prompt_text = req.prompt

    prompt_ids: list[int] = tokenizer(
        prompt_text, add_special_tokens=True
    ).input_ids

    if req.stream:
        return StreamingResponse(
            _completion_stream(prompt_text, req.max_tokens, prompt_ids),
            media_type="text/event-stream",
        )

    # Non-streaming path: still goes through the scheduler.
    model_name = STATE["model_name"]
    cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    token_queue, _job = await _submit_and_drain(prompt_ids, req.max_tokens)

    out_ids: list[int] = []
    finish_reason = "length"
    while True:
        item = await token_queue.get()
        if isinstance(item, tuple) and item and item[0] == "done":
            finish_reason = item[1]
            break
        out_ids.append(int(item))

    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return JSONResponse(
        {
            "id": cmpl_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {"text": text, "index": 0, "finish_reason": finish_reason}
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(out_ids),
                "total_tokens": len(prompt_ids) + len(out_ids),
            },
        }
    )


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


async def _chat_stream(prompt_ids: list[int], max_tokens: int):
    tokenizer = STATE["tokenizer"]
    model_name = STATE["model_name"]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    token_queue, _job = await _submit_and_drain(prompt_ids, max_tokens)

    # First chunk: role assistant
    first = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

    emitted_ids: list[int] = []
    emitted_text_len = 0
    finish_reason: str = "length"
    while True:
        item = await token_queue.get()
        if isinstance(item, tuple) and item and item[0] == "done":
            finish_reason = item[1]
            break
        tok = int(item)
        emitted_ids.append(tok)
        new_text, emitted_text_len = _utf8_safe_decode(
            tokenizer, emitted_ids, emitted_text_len
        )
        if new_text:
            yield _chat_chunk(cmpl_id, model_name, new_text, None)

    final_full = tokenizer.decode(emitted_ids, skip_special_tokens=True)
    leftover = final_full[emitted_text_len:]
    if leftover:
        yield _chat_chunk(cmpl_id, model_name, leftover, None)

    yield _chat_chunk(cmpl_id, model_name, "", finish_reason)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    tokenizer = STATE["tokenizer"]
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids: list[int] = tokenizer(
        prompt_text, add_special_tokens=False
    ).input_ids

    if req.stream:
        return StreamingResponse(
            _chat_stream(prompt_ids, req.max_tokens),
            media_type="text/event-stream",
        )

    model_name = STATE["model_name"]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    token_queue, _job = await _submit_and_drain(prompt_ids, req.max_tokens)

    out_ids: list[int] = []
    finish_reason = "length"
    while True:
        item = await token_queue.get()
        if isinstance(item, tuple) and item and item[0] == "done":
            finish_reason = item[1]
            break
        out_ids.append(int(item))

    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return JSONResponse(
        {
            "id": cmpl_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(out_ids),
                "total_tokens": len(prompt_ids) + len(out_ids),
            },
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", workers=1)


if __name__ == "__main__":
    main()
