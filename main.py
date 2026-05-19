"""VibeServeModel — hand-rolled Qwen3.5-9B (hybrid GDN + GQA) inference path.

Accuracy-checker contract:
    model = VibeServeModel.from_pretrained(model_dir, device, dtype)
    output_ids = model.generate(input_ids, max_new_tokens=N)  # greedy, returns (1, T+N)

FastAPI server with continuous batching:
    POST /v1/completions  (OpenAI-compatible, per-token SSE on stream=true)

Engine design (R2):
- One async ``StepEngine`` task owns the GPU. Per-layer KV / GDN-state pools sized
  ``(MAX_BATCH, ...)`` are pre-allocated at startup; active sequences occupy the
  contiguous prefix ``[0, B)``. On finish, the last active slot is swap-popped into
  the freed position. Each decode step advances ALL active slots one token at a
  time, then yields to the event loop so SSE handlers can flush per-token frames.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoTokenizer

# SDPA backend preferences. Two separate orderings:
#   - Decode path (growing K_cache length each step → unique shape per call): use MATH.
#     cuDNN heuristic-searches a kernel per unique (B, H, T_q, T_k, D) tuple at ~125 ms
#     *per shape*, which dominates wall clock when T_k advances every step.
#   - Prefill / batch=1 paths (uniform self-attention): use FlashAttention/cuDNN first;
#     each request hits the same shape only once anyway and the kernel is much faster
#     than MATH for prefill's larger (B, H, T, T) workload.
SDPA_DECODE_BACKENDS = [SDPBackend.MATH]
SDPA_PREFILL_BACKENDS = [
    SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,
]

from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Qwen35RMSNorm(nn.Module):
    """RMSNorm with `(1 + weight)` scaling, matching HF Qwen3.5."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * (1.0 + self.weight.float())).to(in_dtype)


class Qwen35RotaryEmb(nn.Module):
    """Partial RoPE. For text-only inputs the three mrope axes share text positions,
    so the interleaving in HF collapses to standard RoPE on the first
    ``int(head_dim * partial_rotary_factor)`` dims of each head."""

    def __init__(self, head_dim: int, partial_rotary_factor: float, rope_theta: float):
        super().__init__()
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float64) / self.rotary_dim
        ))
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)

    def cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype):
        """position_ids: (B, T) long -> cos, sin each (B, T, rotary_dim)."""
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q,k: (B, T, H, D). cos/sin: (B, T, rotary_dim). Only the first rotary_dim of D rotates."""
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    rd = cos.shape[-1]
    q_rot, q_pass = q[..., :rd], q[..., rd:]
    k_rot, k_pass = k[..., :rd], k[..., rd:]
    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class Qwen35MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Full attention (GQA, head_dim=256, output-gated, partial RoPE).
# ---------------------------------------------------------------------------


class FullAttnCache:
    """Per-request KV cache used by the R1 ``.generate()`` path (batch=1). NHD layout."""

    def __init__(self, max_len: int, num_kv_heads: int, head_dim: int, device, dtype):
        self.k = torch.empty(1, max_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.empty(1, max_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        self.length = 0

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor):
        """new_k/new_v: (1, T, num_kv_heads, head_dim). Returns the populated prefix views."""
        L = new_k.shape[1]
        end = self.length + L
        self.k[:, self.length:end, :, :].copy_(new_k)
        self.v[:, self.length:end, :, :].copy_(new_v)
        self.length = end
        return self.k[:, :end, :, :], self.v[:, :end, :, :]


# Attention backend selection: try FA3, then FA2 (`flash_attn_with_kvcache`),
# then fall back to PyTorch SDPA with ``enable_gqa=True``. Set by ``select_attn_backend``.
ATTN_BACKEND: str = "uninitialized"
_FA_WITH_KVCACHE = None
_FA_FUNC = None


def select_attn_backend() -> str:
    """Pick the best installed attention backend; returns the name of the active backend.

    Order: ``flash_attn_3`` (Dao-AILab FA3 hopper) > ``flash_attn_2`` (`flash_attn` PyPI)
    > ``sdpa-gqa`` (always present). Logged once at startup.
    """
    global ATTN_BACKEND, _FA_WITH_KVCACHE, _FA_FUNC
    try:
        import flash_attn_interface  # type: ignore[import-not-found]
        _FA_WITH_KVCACHE = getattr(flash_attn_interface, "flash_attn_with_kvcache", None)
        _FA_FUNC = getattr(flash_attn_interface, "flash_attn_func", None)
        if _FA_WITH_KVCACHE is not None and _FA_FUNC is not None:
            ATTN_BACKEND = "flash_attn_3"
            return ATTN_BACKEND
    except ImportError:
        pass
    try:
        from flash_attn import flash_attn_func, flash_attn_with_kvcache  # type: ignore[import-not-found]
        _FA_WITH_KVCACHE = flash_attn_with_kvcache
        _FA_FUNC = flash_attn_func
        ATTN_BACKEND = "flash_attn_2"
        return ATTN_BACKEND
    except ImportError:
        pass
    ATTN_BACKEND = "sdpa-gqa"
    return ATTN_BACKEND


# Initialize the attention backend at import time so the accuracy-checker path
# (`VibeServeModel.from_pretrained(...).generate(...)`) sees a chosen backend even
# when the FastAPI lifespan isn't running.
select_attn_backend()


class Qwen35FullAttention(nn.Module):
    """GQA 16:4 / head_dim=256 / partial-RoPE / sigmoid-output-gated attention.

    The attention kernel is selected by ``select_attn_backend`` at server startup.
    All three call sites (R1 ``forward``, ``prefill_pool``, ``decode_pool``) route
    through the same backend so GQA grouping is handled inside the kernel and no
    explicit softmax / ``torch.repeat_interleave`` runs inside this class.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 rms_eps: float, rotary: Qwen35RotaryEmb):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim ** -0.5
        self.rotary = rotary

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps=rms_eps)

    # ---- helpers ----

    def _project_qkv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden -> (q, gate, k, v) in BTHD layout with q/k norms applied."""
        B, T, _ = hidden.shape
        qg = self.q_proj(hidden).view(B, T, self.num_heads, self.head_dim * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(B, T, self.num_heads * self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim)
        return q, gate, k, v

    def _attn_prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Self-attention over uniform-length Q/K (B, T, H, D), causal.

        Returns (B, T, num_q_heads, head_dim)."""
        if ATTN_BACKEND in ("flash_attn_2", "flash_attn_3") and _FA_FUNC is not None:
            return _FA_FUNC(q, k, v, causal=True, softmax_scale=self.scaling)
        # SDPA expects (B, H, T, D); permute for the call, then back.
        q_b = q.transpose(1, 2).contiguous()
        k_b = k.transpose(1, 2).contiguous()
        v_b = v.transpose(1, 2).contiguous()
        with sdpa_kernel(SDPA_PREFILL_BACKENDS):
            out = F.scaled_dot_product_attention(
                q_b, k_b, v_b, is_causal=True, enable_gqa=True, scale=self.scaling,
            )
        return out.transpose(1, 2).contiguous()  # (B, T, H_q, D)

    def _attn_decode(self, q_new: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
                     K_pool: torch.Tensor, V_pool: torch.Tensor,
                     lengths_after: torch.Tensor, max_kv_len: int, B: int) -> torch.Tensor:
        """Append `k_new`/`v_new` into the pool slot and run attention over the active prefix.

        K_pool / V_pool layout: ``(MAX_BATCH, MAX_LEN, num_kv_heads, head_dim)``.
        ``lengths_after``: int[B] K/V length *after* this step's append.
        Returns ``(B, 1, num_q_heads, head_dim)``.
        """
        # 1. Append new K/V in place into the pool at row i, position lengths_after[i]-1.
        batch_idx = torch.arange(B, device=K_pool.device)
        write_pos = lengths_after - 1
        K_pool[batch_idx, write_pos, :, :] = k_new[:, 0, :, :]
        V_pool[batch_idx, write_pos, :, :] = v_new[:, 0, :, :]

        if ATTN_BACKEND in ("flash_attn_2", "flash_attn_3") and _FA_WITH_KVCACHE is not None:
            # FA `with_kvcache` consumes (B, max_seqlen_k, H_kv, D) NHD caches and the new
            # K/V via `cache_seqlens` = lengths BEFORE the append. We pass the already-
            # appended caches with `k=None, v=None` and `cache_seqlens = lengths_after`.
            cache_seqlens = lengths_after.to(torch.int32)
            out = _FA_WITH_KVCACHE(
                q=q_new,
                k_cache=K_pool[:B],
                v_cache=V_pool[:B],
                k=None, v=None,
                cache_seqlens=cache_seqlens,
                causal=True,
                softmax_scale=self.scaling,
            )
            return out  # (B, 1, num_q_heads, head_dim)

        # SDPA fallback path.
        full_k_bthd = K_pool[:B, :max_kv_len, :, :]   # (B, max_kv_len, H_kv, D)
        full_v_bthd = V_pool[:B, :max_kv_len, :, :]
        # SDPA wants BHTD; do a transpose-view (last dim stays contiguous).
        q_b = q_new.transpose(1, 2).contiguous()                       # (B, H_q, 1, D)
        k_b = full_k_bthd.transpose(1, 2)                              # (B, H_kv, max_kv_len, D)
        v_b = full_v_bthd.transpose(1, 2)
        pos_j = torch.arange(max_kv_len, device=q_new.device)
        keep = pos_j[None, None, None, :] < lengths_after[:, None, None, None]
        attn_mask = torch.where(
            keep, q_new.new_zeros(()), q_new.new_full((), float("-inf")),
        )                                                              # (B, 1, 1, max_kv_len)
        with sdpa_kernel(SDPA_DECODE_BACKENDS):
            out_b = F.scaled_dot_product_attention(
                q_b, k_b, v_b, attn_mask=attn_mask, is_causal=False,
                enable_gqa=True, scale=self.scaling,
            )
        return out_b.transpose(1, 2).contiguous()                      # (B, 1, H_q, D)

    # ---- R1 single-batch path (used by .generate() / accuracy checker) ----

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor,
                cache: FullAttnCache | None) -> torch.Tensor:
        B, T, _ = hidden.shape
        q, gate, k, v = self._project_qkv(hidden)                       # all BTHD
        cos, sin = self.rotary.cos_sin(position_ids, q.dtype)
        q, k = apply_partial_rope(q, k, cos, sin)
        if cache is not None:
            full_k, full_v = cache.append(k, v)                         # NHD views
        else:
            full_k, full_v = k, v

        if T == full_k.shape[1]:
            # Pure self-attention (prefill, no prior context).
            out = self._attn_prefill(q, full_k, full_v)                 # (1, T, H_q, D)
        else:
            # T < cache_len: decode step (T == 1 in practice).
            if ATTN_BACKEND in ("flash_attn_2", "flash_attn_3") and _FA_FUNC is not None:
                # FA func handles uniform-length self-attn. For decode we need Q to
                # attend over the full cached K; use FA's `flash_attn_func` with a
                # K of shape (1, cache_len, H_kv, D) and Q of (1, T=1, H_q, D); FA
                # bottom-right-aligns the causal mask, so the single Q attends to all K.
                out = _FA_FUNC(q, full_k, full_v, causal=True, softmax_scale=self.scaling)
            else:
                # SDPA: bottom-right alignment for is_causal=True with Sq < Sk does
                # not exist; with Sq=1 a causal mask is trivially satisfied so we
                # just pass is_causal=False with no mask (Q attends to all K).
                q_b = q.transpose(1, 2).contiguous()
                k_b = full_k.transpose(1, 2)
                v_b = full_v.transpose(1, 2)
                with sdpa_kernel(SDPA_DECODE_BACKENDS):
                    out_b = F.scaled_dot_product_attention(
                        q_b, k_b, v_b, is_causal=False, enable_gqa=True, scale=self.scaling,
                    )
                out = out_b.transpose(1, 2).contiguous()

        out = out.reshape(B, T, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)

    # ---- R2 pooled paths (used by the StepEngine) ----

    def prefill_pool(self, hidden: torch.Tensor, position_ids: torch.Tensor,
                     K_pool: torch.Tensor, V_pool: torch.Tensor, slot: int) -> torch.Tensor:
        """Single-request prefill into slot ``slot`` of the per-layer K/V pool.
        K_pool/V_pool layout: (MAX_BATCH, MAX_LEN, num_kv_heads, head_dim) (NHD)."""
        B, T, _ = hidden.shape  # B == 1
        q, gate, k, v = self._project_qkv(hidden)
        cos, sin = self.rotary.cos_sin(position_ids, q.dtype)
        q, k = apply_partial_rope(q, k, cos, sin)

        # Write prompt K/V into the pool slot's prefix [0, T). NHD layout.
        K_pool[slot, :T, :, :].copy_(k[0])
        V_pool[slot, :T, :, :].copy_(v[0])

        out = self._attn_prefill(q, k, v)                               # (1, T, H_q, D)
        out = out.reshape(B, T, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)

    def decode_pool(self, hidden: torch.Tensor, K_pool: torch.Tensor, V_pool: torch.Tensor,
                    lengths_after: torch.Tensor, max_kv_len: int, B: int) -> torch.Tensor:
        """Batched single-token decode over the contiguous active prefix ``[0, B)``."""
        q, gate, k, v = self._project_qkv(hidden)                       # (B, 1, H_*, D)
        # Per-row position is lengths_after - 1 (0-indexed).
        positions = (lengths_after - 1).unsqueeze(1)                    # (B, 1)
        cos, sin = self.rotary.cos_sin(positions, q.dtype)
        q, k = apply_partial_rope(q, k, cos, sin)

        out = self._attn_decode(q, k, v, K_pool, V_pool, lengths_after, max_kv_len, B)
        out = out.reshape(B, 1, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Linear attention (Gated DeltaNet) — both per-request and pooled paths.
# ---------------------------------------------------------------------------


class GDNCache:
    """Per-request GDN state used by the R1 ``.generate()`` path."""

    def __init__(self, num_v_heads: int, head_k_dim: int, head_v_dim: int,
                 conv_dim: int, conv_kernel: int, device, dtype, state_dtype):
        self.recurrent_state = torch.zeros(1, num_v_heads, head_k_dim, head_v_dim,
                                           device=device, dtype=state_dtype)
        self.conv_state = torch.zeros(1, conv_dim, conv_kernel - 1, device=device, dtype=dtype)
        self.has_state = False


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, hidden_size: int, num_v_heads: int, num_k_heads: int,
                 head_k_dim: int, head_v_dim: int, conv_kernel: int, rms_eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel = conv_kernel
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim,
            kernel_size=conv_kernel, bias=False, groups=self.conv_dim,
            padding=conv_kernel - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        self.A_log = nn.Parameter(torch.empty(num_v_heads).uniform_(0, 16).log_())

        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)

        self.norm = FusedRMSNormGated(head_v_dim, eps=rms_eps, activation="swish")
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ---- helpers shared by both paths ----

    def _post_recurrent_proj(self, core_out: torch.Tensor, z: torch.Tensor,
                              B: int, T: int) -> torch.Tensor:
        core_flat = core_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_flat = self.norm(core_flat, z_flat)
        return self.out_proj(core_flat.reshape(B, T, self.value_dim))

    def _gate_decay(self, a: torch.Tensor) -> torch.Tensor:
        """g = -exp(A_log) * softplus(a + dt_bias), computed in fp32."""
        return -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

    # ---- R1 single-batch path ----

    def _conv_update_step(self, x: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        weight = self.conv1d.weight.squeeze(1)
        window = torch.cat([conv_state, x.to(conv_state.dtype)], dim=-1)
        conv_state.copy_(window[:, :, -(self.conv_kernel - 1):])
        out = (window * weight.unsqueeze(0)).sum(dim=-1, keepdim=True)
        return F.silu(out).to(x.dtype)

    def forward(self, hidden: torch.Tensor, cache: GDNCache | None) -> torch.Tensor:
        B, T, _ = hidden.shape
        use_cached = cache is not None and cache.has_state
        mixed_qkv = self.in_proj_qkv(hidden).transpose(1, 2)
        z = self.in_proj_z(hidden).reshape(B, T, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden)
        a = self.in_proj_a(hidden)
        if use_cached and T == 1:
            mixed_qkv = self._conv_update_step(mixed_qkv, cache.conv_state)
        else:
            if use_cached:
                mixed_qkv = torch.cat([cache.conv_state.to(mixed_qkv.dtype), mixed_qkv], dim=-1)
            if cache is not None:
                pad_left = self.conv_kernel - mixed_qkv.shape[-1]
                padded = F.pad(mixed_qkv, (max(pad_left, 0), 0))
                cache.conv_state.copy_(padded[:, :, -(self.conv_kernel - 1):])
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
            if use_cached:
                mixed_qkv = mixed_qkv[:, :, -T:]
        mixed_qkv = mixed_qkv.transpose(1, 2)
        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, T, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)
        beta = b.sigmoid()
        g = self._gate_decay(a)
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        init_state = cache.recurrent_state if use_cached else None
        out_state = cache is not None
        if use_cached and T == 1:
            core_out, last_state = fused_recurrent_gated_delta_rule(
                q, k, v, g=g, beta=beta, initial_state=init_state,
                output_final_state=out_state, use_qk_l2norm_in_kernel=True,
            )
        else:
            core_out, last_state = chunk_gated_delta_rule(
                q, k, v, g=g, beta=beta, initial_state=init_state,
                output_final_state=out_state, use_qk_l2norm_in_kernel=True,
            )
        if cache is not None and last_state is not None:
            cache.recurrent_state.copy_(last_state.to(cache.recurrent_state.dtype))
            cache.has_state = True
        return self._post_recurrent_proj(core_out, z, B, T)

    # ---- R2 pooled paths ----

    def prefill_pool(self, hidden: torch.Tensor, state_pool: torch.Tensor,
                     conv_pool: torch.Tensor, slot: int) -> torch.Tensor:
        """Single-request prefill that primes pool slot's GDN + conv state."""
        B, T, _ = hidden.shape  # B == 1
        mixed_qkv = self.in_proj_qkv(hidden).transpose(1, 2)  # (1, conv_dim, T)
        z = self.in_proj_z(hidden).reshape(B, T, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden)
        a = self.in_proj_a(hidden)

        # Save new conv state = last (kernel-1) of the (left-zero-padded if short) input.
        pad_left = self.conv_kernel - mixed_qkv.shape[-1]
        padded = F.pad(mixed_qkv, (max(pad_left, 0), 0))
        conv_pool[slot].copy_(padded[0, :, -(self.conv_kernel - 1):].to(conv_pool.dtype))

        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :T])
        mixed_qkv = mixed_qkv.transpose(1, 2)
        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, T, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)
        beta = b.sigmoid()
        g = self._gate_decay(a)
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        core_out, final_state = chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        state_pool[slot].copy_(final_state[0].to(state_pool.dtype))
        return self._post_recurrent_proj(core_out, z, B, T)

    def decode_pool(self, hidden: torch.Tensor, state_pool: torch.Tensor,
                    conv_pool: torch.Tensor, B: int) -> torch.Tensor:
        """Batched single-token decode over the contiguous active prefix ``[0, B)``."""
        mixed_qkv = self.in_proj_qkv(hidden).transpose(1, 2)  # (B, conv_dim, 1)
        z = self.in_proj_z(hidden).reshape(B, 1, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden)
        a = self.in_proj_a(hidden)

        # Batched causal-conv1d step on the active prefix, in-place state update.
        conv_state_slice = conv_pool[:B]
        window = torch.cat([conv_state_slice, mixed_qkv.to(conv_state_slice.dtype)], dim=-1)
        conv_state_slice.copy_(window[:, :, -(self.conv_kernel - 1):])
        weight = self.conv1d.weight.squeeze(1)  # (conv_dim, kernel)
        out_conv = (window * weight.unsqueeze(0)).sum(dim=-1, keepdim=True)
        mixed_qkv = F.silu(out_conv).to(mixed_qkv.dtype).transpose(1, 2)  # (B, 1, conv_dim)

        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, 1, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, 1, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, 1, self.num_v_heads, self.head_v_dim)
        beta = b.sigmoid()
        g = self._gate_decay(a)
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)

        init_state = state_pool[:B]
        core_out, final_state = fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=init_state,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        state_pool[:B].copy_(final_state)
        return self._post_recurrent_proj(core_out, z, B, 1)


# ---------------------------------------------------------------------------
# Decoder layer + full model
# ---------------------------------------------------------------------------


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, layer_type: str, hidden_size: int, intermediate_size: int,
                 rms_eps: float, full_attn: Qwen35FullAttention | None,
                 linear_attn: Qwen35GatedDeltaNet | None):
        super().__init__()
        self.layer_type = layer_type
        self.input_layernorm = Qwen35RMSNorm(hidden_size, eps=rms_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(hidden_size, eps=rms_eps)
        self.mlp = Qwen35MLP(hidden_size, intermediate_size)
        if layer_type == "full_attention":
            self.self_attn = full_attn
        else:
            self.linear_attn = linear_attn

    # ---- R1 per-request forward (used by .generate()) ----

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor,
                full_cache: FullAttnCache | None, gdn_cache: GDNCache | None) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        if self.layer_type == "full_attention":
            hidden = self.self_attn(hidden, position_ids, full_cache)
        else:
            hidden = self.linear_attn(hidden, gdn_cache)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden

    # ---- R2 pooled paths ----

    def prefill_pool(self, hidden: torch.Tensor, position_ids: torch.Tensor,
                     K_pool, V_pool, state_pool, conv_pool, slot: int) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        if self.layer_type == "full_attention":
            hidden = self.self_attn.prefill_pool(hidden, position_ids, K_pool, V_pool, slot)
        else:
            hidden = self.linear_attn.prefill_pool(hidden, state_pool, conv_pool, slot)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        return residual + self.mlp(hidden)

    def decode_pool(self, hidden: torch.Tensor, K_pool, V_pool, state_pool, conv_pool,
                    lengths_after: torch.Tensor, max_kv_len: int, B: int) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        if self.layer_type == "full_attention":
            hidden = self.self_attn.decode_pool(hidden, K_pool, V_pool, lengths_after, max_kv_len, B)
        else:
            hidden = self.linear_attn.decode_pool(hidden, state_pool, conv_pool, B)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        return residual + self.mlp(hidden)


class VibeServeModel(nn.Module):
    """Hand-rolled Qwen3.5-9B causal LM (text only)."""

    EOS_TOKEN_ID = 248044

    def __init__(self, config_text: Any, device, dtype):
        super().__init__()
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.dtype = dtype

        self.hidden_size = config_text.hidden_size
        self.vocab_size = config_text.vocab_size
        self.num_layers = config_text.num_hidden_layers
        self.head_dim = config_text.head_dim
        self.num_heads = config_text.num_attention_heads
        self.num_kv_heads = config_text.num_key_value_heads
        self.intermediate_size = config_text.intermediate_size
        self.rms_eps = config_text.rms_norm_eps
        self.layer_types = list(config_text.layer_types)

        self.linear_num_v_heads = config_text.linear_num_value_heads
        self.linear_num_k_heads = config_text.linear_num_key_heads
        self.linear_head_k_dim = config_text.linear_key_head_dim
        self.linear_head_v_dim = config_text.linear_value_head_dim
        self.linear_conv_kernel = config_text.linear_conv_kernel_dim
        self.linear_conv_dim = (
            self.linear_num_k_heads * self.linear_head_k_dim * 2
            + self.linear_num_v_heads * self.linear_head_v_dim
        )

        rope_params = config_text.rope_parameters
        self.rotary = Qwen35RotaryEmb(
            head_dim=self.head_dim,
            partial_rotary_factor=rope_params.get("partial_rotary_factor", 1.0),
            rope_theta=rope_params["rope_theta"],
        )

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        layers: list[Qwen35DecoderLayer] = []
        for lt in self.layer_types:
            if lt == "full_attention":
                full = Qwen35FullAttention(
                    self.hidden_size, self.num_heads, self.num_kv_heads, self.head_dim,
                    self.rms_eps, self.rotary,
                )
                linear = None
            else:
                full = None
                linear = Qwen35GatedDeltaNet(
                    self.hidden_size, self.linear_num_v_heads, self.linear_num_k_heads,
                    self.linear_head_k_dim, self.linear_head_v_dim,
                    self.linear_conv_kernel, self.rms_eps,
                )
            layers.append(Qwen35DecoderLayer(lt, self.hidden_size, self.intermediate_size,
                                             self.rms_eps, full, linear))
        self.layers = nn.ModuleList(layers)
        self.norm = Qwen35RMSNorm(self.hidden_size, eps=self.rms_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    # ---- Pool-aware single-step model passes (used by StepEngine) ----

    @torch.inference_mode()
    def prefill_to_slot(self, input_ids: torch.LongTensor, slot: int, pool: "Pool") -> int:
        """Run prefill into ``pool`` slot ``slot``. Returns first decoded token id (int)."""
        T = input_ids.shape[1]
        pos = torch.arange(T, device=self.device).unsqueeze(0)  # (1, T)
        hidden = self.embed_tokens(input_ids).to(self.dtype)
        for i, layer in enumerate(self.layers):
            hidden = layer.prefill_pool(
                hidden, pos,
                pool.full_K[i], pool.full_V[i],
                pool.gdn_state[i], pool.gdn_conv[i],
                slot,
            )
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden[:, -1, :])  # (1, V)
        return int(logits.argmax(dim=-1).item())

    @torch.inference_mode()
    def decode_batch(self, last_tokens: torch.LongTensor, pool: "Pool",
                     B: int, lengths_after: torch.Tensor, max_kv_len: int) -> torch.LongTensor:
        """Batched decode step. ``last_tokens``: (B, 1). Returns next-token ids (B,)."""
        hidden = self.embed_tokens(last_tokens).to(self.dtype)
        for i, layer in enumerate(self.layers):
            hidden = layer.decode_pool(
                hidden,
                pool.full_K[i], pool.full_V[i],
                pool.gdn_state[i], pool.gdn_conv[i],
                lengths_after, max_kv_len, B,
            )
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden[:, -1, :])  # (B, V)
        return logits.argmax(dim=-1)

    def decode_step_graph_safe(self, B_bucket: int, kv_len_bucket: int,
                                input_ids_buf: torch.Tensor,
                                lengths_after_buf: torch.Tensor,
                                next_token_buf: torch.Tensor,
                                pool: "Pool") -> None:
        """One full decode step that reads from / writes to fixed-address static
        buffers — the body that gets wrapped in a ``torch.cuda.CUDAGraph``.

        ``B_bucket`` and ``kv_len_bucket`` are Python ints baked into the captured
        kernel launches; tensor shapes are constant for a given bucket so the same
        captured graph can be replayed for any actual ``B <= B_bucket`` and any
        actual ``max_kv_len <= kv_len_bucket`` (the per-row attn mask handles the
        slack).
        """
        last_tokens = input_ids_buf[:B_bucket].unsqueeze(1)              # (B_bucket, 1)
        hidden = self.embed_tokens(last_tokens).to(self.dtype)
        lengths_after = lengths_after_buf[:B_bucket]
        for i, layer in enumerate(self.layers):
            hidden = layer.decode_pool(
                hidden,
                pool.full_K[i], pool.full_V[i],
                pool.gdn_state[i], pool.gdn_conv[i],
                lengths_after, kv_len_bucket, B_bucket,
            )
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden[:, -1, :])                          # (B_bucket, V)
        next_token_buf[:B_bucket].copy_(logits.argmax(dim=-1))

    # ---- R1 batch=1 .generate() retained for the accuracy checker ----

    def _build_caches_b1(self, max_len: int):
        full_caches: list[FullAttnCache | None] = []
        gdn_caches: list[GDNCache | None] = []
        for lt in self.layer_types:
            if lt == "full_attention":
                full_caches.append(FullAttnCache(
                    max_len, self.num_kv_heads, self.head_dim, self.device, self.dtype,
                ))
                gdn_caches.append(None)
            else:
                full_caches.append(None)
                gdn_caches.append(GDNCache(
                    num_v_heads=self.linear_num_v_heads,
                    head_k_dim=self.linear_head_k_dim,
                    head_v_dim=self.linear_head_v_dim,
                    conv_dim=self.linear_conv_dim,
                    conv_kernel=self.linear_conv_kernel,
                    device=self.device, dtype=self.dtype, state_dtype=torch.float32,
                ))
        return full_caches, gdn_caches

    def _forward_b1(self, hidden, position_ids, full_caches, gdn_caches) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, position_ids, full_caches[i], gdn_caches[i])
        return self.norm(hidden)

    @torch.inference_mode()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 16) -> torch.LongTensor:
        assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "Batch=1 only."
        input_ids = input_ids.to(self.device)
        prompt_len = input_ids.shape[1]
        max_len = prompt_len + max_new_tokens

        full_caches, gdn_caches = self._build_caches_b1(max_len)
        position_ids = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        hidden = self.embed_tokens(input_ids).to(self.dtype)
        hidden = self._forward_b1(hidden, position_ids, full_caches, gdn_caches)
        logits = self.lm_head(hidden[:, -1:, :])
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        output = torch.empty(1, max_len, dtype=input_ids.dtype, device=self.device)
        output[:, :prompt_len] = input_ids
        output[:, prompt_len:prompt_len + 1] = next_tok
        out_len = prompt_len + 1
        if next_tok.item() == self.EOS_TOKEN_ID:
            return output[:, :out_len]

        for step in range(1, max_new_tokens):
            cur_pos = torch.full((1, 1), prompt_len + step - 1, device=self.device, dtype=torch.long)
            hidden = self.embed_tokens(next_tok).to(self.dtype)
            hidden = self._forward_b1(hidden, cur_pos, full_caches, gdn_caches)
            logits = self.lm_head(hidden[:, -1:, :])
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output[:, prompt_len + step:prompt_len + step + 1] = next_tok
            out_len = prompt_len + step + 1
            if next_tok.item() == self.EOS_TOKEN_ID:
                break
        return output[:, :out_len]

    # ---- weight loading ----

    @classmethod
    def from_pretrained(cls, model_dir: str, device, dtype) -> "VibeServeModel":
        cfg = AutoConfig.from_pretrained(model_dir)
        text_cfg = cfg.text_config
        model = cls(text_cfg, device, dtype)
        model.to(device=device, dtype=dtype)
        model.eval()
        model._load_weights(model_dir)
        return model

    def _load_weights(self, model_dir: str) -> None:
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        shard_to_keys: dict[str, list[str] | None] = {}
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx = json.load(f)
            for hf_key, shard in idx["weight_map"].items():
                shard_to_keys.setdefault(shard, []).append(hf_key)
        else:
            for fn in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
                shard_to_keys[os.path.basename(fn)] = None

        own_state = dict(self.state_dict())
        assigned: set[str] = set()
        skipped_prefixes = ("model.visual.", "mtp.", "visual.", "vision_")

        for shard_name, keys in shard_to_keys.items():
            shard_path = os.path.join(model_dir, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                key_iter = keys if keys is not None else list(f.keys())
                for hf_key in key_iter:
                    if any(hf_key.startswith(p) for p in skipped_prefixes):
                        continue
                    own_key = self._map_hf_key(hf_key)
                    if own_key is None:
                        continue
                    if own_key not in own_state:
                        raise RuntimeError(f"Mapped key not in own state: {own_key} (from {hf_key})")
                    tensor = f.get_tensor(hf_key)
                    target = own_state[own_key]
                    if tensor.shape != target.shape:
                        raise RuntimeError(
                            f"Shape mismatch for {hf_key} -> {own_key}: "
                            f"{tensor.shape} vs {target.shape}"
                        )
                    target.copy_(tensor.to(target.dtype).to(target.device))
                    assigned.add(own_key)

        missing = {k for k in (set(own_state.keys()) - assigned) if not k.endswith(".inv_freq")}
        if missing:
            raise RuntimeError(f"Missing weights after load: {sorted(missing)[:20]}")

    def _map_hf_key(self, hf_key: str) -> str | None:
        if hf_key == "lm_head.weight":
            return "lm_head.weight"
        prefix = "model.language_model."
        if not hf_key.startswith(prefix):
            return None
        sub = hf_key[len(prefix):]
        if sub == "embed_tokens.weight":
            return "embed_tokens.weight"
        if sub == "norm.weight":
            return "norm.weight"
        if not sub.startswith("layers."):
            return None
        parts = sub.split(".", 2)
        idx = int(parts[1])
        rest = parts[2] if len(parts) > 2 else ""
        layer_type = self.layer_types[idx]
        if rest.startswith("self_attn.") and layer_type != "full_attention":
            return None
        if rest.startswith("linear_attn.") and layer_type != "linear_attention":
            return None
        return f"layers.{idx}.{rest}"


# ---------------------------------------------------------------------------
# Continuous-batching engine
# ---------------------------------------------------------------------------


@dataclass
class Pool:
    """Pre-allocated per-layer KV / GDN-state pools, sized for ``MAX_BATCH`` slots."""

    full_K: list[torch.Tensor | None]
    full_V: list[torch.Tensor | None]
    gdn_state: list[torch.Tensor | None]
    gdn_conv: list[torch.Tensor | None]
    lengths: torch.Tensor  # (MAX_BATCH,) on device

    @classmethod
    def allocate(cls, model: VibeServeModel, max_batch: int, max_len: int) -> "Pool":
        full_K: list[torch.Tensor | None] = []
        full_V: list[torch.Tensor | None] = []
        gdn_state: list[torch.Tensor | None] = []
        gdn_conv: list[torch.Tensor | None] = []
        for lt in model.layer_types:
            if lt == "full_attention":
                # NHD layout: (MAX_BATCH, MAX_LEN, num_kv_heads, head_dim).
                K = torch.empty(max_batch, max_len, model.num_kv_heads, model.head_dim,
                                device=model.device, dtype=model.dtype)
                V = torch.empty_like(K)
                full_K.append(K); full_V.append(V)
                gdn_state.append(None); gdn_conv.append(None)
            else:
                state = torch.zeros(max_batch, model.linear_num_v_heads,
                                    model.linear_head_k_dim, model.linear_head_v_dim,
                                    device=model.device, dtype=torch.float32)
                conv = torch.zeros(max_batch, model.linear_conv_dim,
                                   model.linear_conv_kernel - 1,
                                   device=model.device, dtype=model.dtype)
                gdn_state.append(state); gdn_conv.append(conv)
                full_K.append(None); full_V.append(None)
        lengths = torch.zeros(max_batch, dtype=torch.long, device=model.device)
        return cls(full_K=full_K, full_V=full_V, gdn_state=gdn_state, gdn_conv=gdn_conv,
                   lengths=lengths)


@dataclass
class EngineRequest:
    prompt_ids: torch.LongTensor  # (1, T) on device
    max_tokens: int
    out_queue: asyncio.Queue
    done: asyncio.Event
    last_token: int = 0
    tokens_emitted: int = 0
    finish_reason: str | None = None


class StepEngine:
    """Single asyncio.Task that owns the GPU. Admits new requests, runs a batched
    decode step over all active sequences, and pushes per-token results onto each
    request's queue."""

    def __init__(self, model: VibeServeModel, max_batch: int = 16, max_len: int = 4096,
                 step_log_interval: int = 0):
        self.model = model
        self.max_batch = max_batch
        self.max_len = max_len
        self.device = model.device
        self.pool = Pool.allocate(model, max_batch, max_len)

        self.waiting: deque[EngineRequest] = deque()
        self.active: list[EngineRequest] = []  # active[i] occupies pool slot i.
        self.B = 0

        self.wake = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.task: asyncio.Task | None = None
        self._step_log_interval = step_log_interval
        self._step_count = 0
        self._max_observed_B = 0

        # ----- CUDA graph state (set up by capture_graphs(); checked at decode time) -----
        self.disable_cuda_graphs = bool(int(os.environ.get("VIBE_DISABLE_CUDA_GRAPHS", "0")))
        self.graph_capture_done = False
        self.capture_stream: torch.cuda.Stream | None = None
        self.graph_pool = None  # torch.cuda.graph_pool_handle()
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._bucket_b = [1, 2, 4, 8, 16]
        self._bucket_kv = [128, 256, 512, 1024, 2048, 4096]
        self._graph_replay_counts: dict[tuple[int, int], int] = {}
        self._eager_fallback_count = 0
        # Persistent step-I/O staging buffers — addresses captured into each graph.
        self.input_ids_buf = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self.lengths_after_buf = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self.next_token_buf = torch.zeros(max_batch, dtype=torch.long, device=self.device)

    # ---- public API ----

    def submit(self, req: EngineRequest) -> None:
        self.waiting.append(req)
        self.wake.set()

    def warmup(self) -> None:
        """Pre-compile Triton (GDN) kernels and SDPA backend code paths at the shapes
        the engine will actually use. Without this, the FIRST benchmark request after a
        cold server start pays seconds of Triton/cuDNN compilation, badly skewing TTFT.
        Runs against an empty pool, then resets pool state so real requests start clean.

        Two pieces of warm-up:
          1. prefill across several T values so the GDN chunked-prefill Triton kernel
             primes its single-chunk / multi-chunk code paths and the full-attn SDPA
             prefill kernel tunes for the common shapes.
          2. batched decode at B=max_batch over a sweep of K-cache lengths so the
             single-step SDPA MATH path, the GDN fused-recurrent kernel, and the LM
             head GEMM are all warm before real traffic arrives.
        """
        device = self.device
        max_batch = self.max_batch

        # 1. Single-request prefill warmup at a range of T values.
        for T_w in (8, 32, 96):
            dummy_ids = torch.zeros(1, T_w, dtype=torch.long, device=device)
            self.model.prefill_to_slot(dummy_ids, 0, self.pool)

        # 2. Batched decode warmup at max_batch over a representative K-length sweep
        # (16 -> ~96, the same range a `max_tokens=64` benchmark hits).
        dummy_ids = torch.zeros(1, 16, dtype=torch.long, device=device)
        for slot in range(max_batch):
            self.model.prefill_to_slot(dummy_ids, slot, self.pool)
        self.pool.lengths[:max_batch] = 16
        last_tokens = torch.zeros(max_batch, 1, dtype=torch.long, device=device)
        for _ in range(80):
            self.pool.lengths[:max_batch] += 1
            lengths_after = self.pool.lengths[:max_batch]
            max_kv_len = int(lengths_after.max().item())
            self.model.decode_batch(last_tokens, self.pool, max_batch, lengths_after, max_kv_len)

        # 3. Reset pool state so real requests start with zero length.
        self._reset_pool_state()
        torch.cuda.synchronize()

    def _reset_pool_state(self) -> None:
        self.pool.lengths.zero_()
        for K in self.pool.full_K:
            if K is not None:
                K.zero_()
        for V in self.pool.full_V:
            if V is not None:
                V.zero_()
        for s in self.pool.gdn_state:
            if s is not None:
                s.zero_()
        for c in self.pool.gdn_conv:
            if c is not None:
                c.zero_()

    # ---- CUDA graph capture / replay ------------------------------------------------

    def capture_graphs(self) -> None:
        """Capture one ``torch.cuda.CUDAGraph`` per (B_bucket, kv_len_bucket) combination,
        all sharing a single ``graph_pool_handle`` to keep memory flat. Each captured
        forward reads from / writes to the persistent step-I/O buffers, so the engine's
        decode hot path just becomes ``update_buffers(); graph.replay()``.

        Must be called AFTER ``warmup()`` (so Triton kernel JIT compiles have already
        produced cached configs) and BEFORE ``start()`` (so the engine never runs the
        un-graphed path under traffic).
        """
        if self.disable_cuda_graphs:
            print("[graph] CUDA graphs DISABLED via VIBE_DISABLE_CUDA_GRAPHS=1; "
                  "engine will use the R3 eager decode path.", flush=True)
            return

        device = self.device
        self.graph_pool = torch.cuda.graph_pool_handle()
        # Side stream is required by the torch.cuda.graph API; it must NOT be the
        # default stream. Replays will record on whatever stream is current at
        # replay time (typically the default stream), which is fine.
        self.capture_stream = torch.cuda.Stream(device=device)

        # Each capture overwrites the pool's first B_bucket slots; reset after.
        for B_bucket in self._bucket_b:
            if B_bucket > self.max_batch:
                continue
            for kv_len_bucket in self._bucket_kv:
                if kv_len_bucket > self.max_len:
                    continue
                t0 = time.perf_counter()
                try:
                    self._capture_one_bucket(B_bucket, kv_len_bucket)
                except Exception as exc:
                    print(f"[graph] capture FAILED for B={B_bucket} kv_len={kv_len_bucket}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    continue
                dt_ms = (time.perf_counter() - t0) * 1000
                print(f"[graph] captured B={B_bucket} kv_len={kv_len_bucket} in {dt_ms:.0f}ms",
                      flush=True)
                self._graph_replay_counts[(B_bucket, kv_len_bucket)] = 0

        self.graph_capture_done = len(self.graphs) > 0
        # Wipe everything captured graphs may have written and start the engine clean.
        self._reset_pool_state()
        self.input_ids_buf.zero_()
        self.lengths_after_buf.zero_()
        self.next_token_buf.zero_()
        torch.cuda.synchronize()
        if self.graph_capture_done:
            print(f"[graph] {len(self.graphs)} graphs captured; "
                  f"buckets B={self._bucket_b} kv_len={self._bucket_kv}", flush=True)
        else:
            print("[graph] NO graphs captured; engine will use the R3 eager decode path.",
                  flush=True)

    def _capture_one_bucket(self, B_bucket: int, kv_len_bucket: int) -> None:
        # Seed dummy state. lengths_after = kv_len_bucket means the new K is appended
        # at position kv_len_bucket-1 — the worst case for this bucket.
        self.pool.lengths[:B_bucket].fill_(kv_len_bucket - 1)
        self.lengths_after_buf[:B_bucket].fill_(kv_len_bucket)
        if B_bucket < self.max_batch:
            # Padding rows write at position 0 and attend over only 1 K position;
            # output is discarded, but values must be in-range for the kernels.
            self.lengths_after_buf[B_bucket:].fill_(1)
        self.input_ids_buf.zero_()

        side_stream = self.capture_stream
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            # Triton autotune / cuBLAS algorithm-pick must finish BEFORE capture.
            with torch.inference_mode():
                for _ in range(3):
                    self.model.decode_step_graph_safe(
                        B_bucket, kv_len_bucket,
                        self.input_ids_buf, self.lengths_after_buf,
                        self.next_token_buf, self.pool,
                    )
            side_stream.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(g, pool=self.graph_pool):
                    self.model.decode_step_graph_safe(
                        B_bucket, kv_len_bucket,
                        self.input_ids_buf, self.lengths_after_buf,
                        self.next_token_buf, self.pool,
                    )
        torch.cuda.current_stream().wait_stream(side_stream)
        self.graphs[(B_bucket, kv_len_bucket)] = g

    def _pick_bucket(self, B: int, max_kv_len_actual: int) -> tuple[int, int] | None:
        """Smallest (B_bucket, kv_len_bucket) that fits, or ``None`` for eager fallback."""
        B_bucket = next((b for b in self._bucket_b if b >= B), None)
        kv_len_bucket = next((b for b in self._bucket_kv if b >= max_kv_len_actual), None)
        if B_bucket is None or kv_len_bucket is None:
            return None
        if (B_bucket, kv_len_bucket) not in self.graphs:
            return None
        return (B_bucket, kv_len_bucket)

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self.stop_event.set()
        self.wake.set()
        if self.task is not None:
            try:
                await asyncio.wait_for(self.task, timeout=5.0)
            except asyncio.TimeoutError:
                self.task.cancel()

    # ---- internal main loop ----

    async def _run(self) -> None:
        while not self.stop_event.is_set():
            if not self.active and not self.waiting:
                self.wake.clear()
                if self.stop_event.is_set():
                    return
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if self.stop_event.is_set():
                    return

            # Admit phase: prefill each new request alone, in turn.
            admitted_any = False
            while self.B < self.max_batch and self.waiting:
                req = self.waiting.popleft()
                self._admit(req)
                admitted_any = True

            # Decode phase: one batched step over all active.
            if self.active:
                self._decode_step()

            if self._step_log_interval > 0:
                self._step_count += 1
                self._max_observed_B = max(self._max_observed_B, self.B)
                if self._step_count % self._step_log_interval == 0:
                    replays = {f"B{b}/kv{k}": v
                               for (b, k), v in self._graph_replay_counts.items() if v > 0}
                    print(f"[engine] step={self._step_count} B={self.B} "
                          f"waiting={len(self.waiting)} max_observed_B={self._max_observed_B} "
                          f"replays={replays} eager_steps={self._eager_fallback_count}",
                          flush=True)

            # Yield to other tasks so SSE handlers can flush queued tokens.
            await asyncio.sleep(0)

    def _admit(self, req: EngineRequest) -> None:
        slot = self.B
        T = req.prompt_ids.shape[1]
        if T + req.max_tokens > self.max_len:
            # Clamp generation so we don't blow the slot's max context.
            req.max_tokens = max(0, self.max_len - T)
            if req.max_tokens == 0:
                req.finish_reason = "length"
                req.out_queue.put_nowait((None, "length"))
                req.done.set()
                return

        first_tok = self.model.prefill_to_slot(req.prompt_ids, slot, self.pool)
        self.pool.lengths[slot] = T
        self.active.append(req)
        self.B += 1
        if self.B > self._max_observed_B:
            self._max_observed_B = self.B

        req.tokens_emitted = 1
        req.last_token = first_tok
        if first_tok == VibeServeModel.EOS_TOKEN_ID:
            req.finish_reason = "stop"
            req.out_queue.put_nowait((None, "stop"))
            req.done.set()
            self._swap_pop_slot(self.B - 1)
        elif req.tokens_emitted >= req.max_tokens:
            req.out_queue.put_nowait((first_tok, None))
            req.finish_reason = "length"
            req.out_queue.put_nowait((None, "length"))
            req.done.set()
            self._swap_pop_slot(self.B - 1)
        else:
            req.out_queue.put_nowait((first_tok, None))

    def _decode_step(self) -> None:
        """Advance every active request by one token. Dispatches to a captured
        CUDAGraph for the (B, max_kv_len) bucket if one exists; falls back to the R3
        eager path (``model.decode_batch``) otherwise. Token bookkeeping (EOS check,
        max_tokens check, slot reclamation via ``_swap_pop_slot``) happens in eager
        Python *outside* the captured region — CUDAGraphs can't include conditional
        control flow."""
        B = self.B
        # Bump KV lengths in place; the new K/V is appended at position lengths-1
        # by the forward (eager or graph).
        self.pool.lengths[:B] += 1
        max_kv_len_actual = int(self.pool.lengths[:B].max().item())

        bucket = (
            None if (self.disable_cuda_graphs or not self.graph_capture_done)
            else self._pick_bucket(B, max_kv_len_actual)
        )

        if bucket is None:
            # Eager R3 path (also used until graphs are captured).
            self._eager_fallback_count += 1
            last_tokens = torch.tensor(
                [r.last_token for r in self.active], dtype=torch.long, device=self.device,
            ).unsqueeze(1)
            new_tokens = self.model.decode_batch(
                last_tokens, self.pool, B, self.pool.lengths[:B], max_kv_len_actual,
            )
            new_tokens_cpu = new_tokens.tolist()
        else:
            B_bucket, kv_len_bucket = bucket
            # Stage inputs into the captured graph's static buffers.
            last_tokens_cpu = [r.last_token for r in self.active]
            # Pad with zeros so the [B:B_bucket] slots are deterministic.
            if B_bucket > B:
                last_tokens_cpu = last_tokens_cpu + [0] * (B_bucket - B)
            self.input_ids_buf[:B_bucket].copy_(
                torch.tensor(last_tokens_cpu, dtype=torch.long, device=self.device),
            )
            self.lengths_after_buf[:B].copy_(self.pool.lengths[:B])
            if B_bucket > B:
                # Padding rows attend over 1 K position (in-range); their outputs are
                # written to next_token_buf[B:B_bucket] but never read out.
                self.lengths_after_buf[B:B_bucket].fill_(1)
            # Replay the captured graph. CUDAGraph.replay() records on whichever
            # stream is current here (the default stream — fine since capture's
            # side stream was joined back via wait_stream).
            self.graphs[bucket].replay()
            self._graph_replay_counts[bucket] += 1
            new_tokens_cpu = self.next_token_buf[:B].tolist()

        finished: list[int] = []
        for i, req in enumerate(self.active):
            tok = int(new_tokens_cpu[i])
            req.tokens_emitted += 1
            req.last_token = tok
            if tok == VibeServeModel.EOS_TOKEN_ID:
                req.finish_reason = "stop"
                req.out_queue.put_nowait((None, "stop"))
                req.done.set()
                finished.append(i)
            elif req.tokens_emitted >= req.max_tokens:
                req.out_queue.put_nowait((tok, None))
                req.finish_reason = "length"
                req.out_queue.put_nowait((None, "length"))
                req.done.set()
                finished.append(i)
            else:
                req.out_queue.put_nowait((tok, None))

        for i in sorted(finished, reverse=True):
            self._swap_pop_slot(i)

    def _swap_pop_slot(self, i: int) -> None:
        last = self.B - 1
        if i != last:
            for layer_idx in range(self.model.num_layers):
                K = self.pool.full_K[layer_idx]
                if K is not None:
                    K[i].copy_(K[last])
                    self.pool.full_V[layer_idx][i].copy_(self.pool.full_V[layer_idx][last])
                else:
                    self.pool.gdn_state[layer_idx][i].copy_(self.pool.gdn_state[layer_idx][last])
                    self.pool.gdn_conv[layer_idx][i].copy_(self.pool.gdn_conv[layer_idx][last])
            self.pool.lengths[i] = self.pool.lengths[last]
            self.active[i] = self.active[last]
        self.active.pop()
        self.B -= 1


# ---------------------------------------------------------------------------
# FastAPI server with per-token SSE streaming.
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

DEFAULT_MODEL_DIR = os.environ.get(
    "VIBE_MODEL_DIR",
    str(Path(__file__).resolve().parent / "reference"),
)


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop: str | list[str] | None = None
    stream: bool = False


_state: dict[str, Any] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    model_dir = DEFAULT_MODEL_DIR
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    max_batch = int(os.environ.get("VIBE_MAX_BATCH", "16"))
    max_len = int(os.environ.get("VIBE_MAX_LEN", "4096"))
    step_log_interval = int(os.environ.get("VIBE_STEP_LOG_INTERVAL", "0"))

    attn_backend = select_attn_backend()
    print(
        f"[attn] backend={attn_backend} head_dim=256 gqa=16:4 layout=NHD pool=(B,L,H,D)",
        flush=True,
    )

    print(f"[vibeserve] loading model from {model_dir} on {device} ({dtype})", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = VibeServeModel.from_pretrained(model_dir, device, dtype)
    print(f"[vibeserve] model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    engine = StepEngine(model, max_batch=max_batch, max_len=max_len,
                        step_log_interval=step_log_interval)
    t0 = time.perf_counter()
    engine.warmup()
    print(f"[vibeserve] warmup completed in {time.perf_counter() - t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    engine.capture_graphs()
    print(f"[vibeserve] graph capture completed in {time.perf_counter() - t0:.1f}s", flush=True)
    engine.start()
    print(f"[vibeserve] engine started (MAX_BATCH={max_batch}, MAX_LEN={max_len})", flush=True)

    _state["model"] = model
    _state["tokenizer"] = tok
    _state["device"] = device
    _state["engine"] = engine
    _state["model_id"] = os.path.basename(model_dir.rstrip("/")) or "qwen3.5-9b"
    try:
        yield
    finally:
        await engine.stop()


app = FastAPI(lifespan=_lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{
            "id": _state.get("model_id", "qwen3.5-9b"),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    if req.stream:
        return StreamingResponse(
            _stream_completion(prompt, req.max_tokens),
            media_type="text/event-stream",
        )
    return await _nonstream_completion(prompt, req.max_tokens)


def _submit_request(prompt: str, max_tokens: int) -> tuple[EngineRequest, int]:
    tok = _state["tokenizer"]
    engine: StepEngine = _state["engine"]
    device = _state["device"]
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    req = EngineRequest(
        prompt_ids=prompt_ids, max_tokens=max_tokens,
        out_queue=asyncio.Queue(), done=asyncio.Event(),
    )
    engine.submit(req)
    return req, prompt_len


async def _stream_completion(prompt: str, max_tokens: int):
    tok = _state["tokenizer"]
    model_id = _state.get("model_id", "qwen3.5-9b")
    cmpl_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    req, prompt_len = _submit_request(prompt, max_tokens)
    gen_token_ids: list[int] = []
    emitted_text = ""

    def chunk_frame(text_chunk: str, finish: str | None) -> str:
        payload = {
            "id": cmpl_id, "object": "text_completion", "created": created, "model": model_id,
            "choices": [{"text": text_chunk, "index": 0, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    while True:
        token_id, finish_reason = await req.out_queue.get()
        if token_id is not None:
            gen_token_ids.append(token_id)
            full = tok.decode(gen_token_ids, skip_special_tokens=True)
            delta = full[len(emitted_text):]
            emitted_text = full
            if delta:
                yield chunk_frame(delta, None)
            else:
                # Emit an empty per-token frame anyway so the SSE consumer sees
                # one data frame per generated token (matching the per-token contract).
                yield chunk_frame("", None)
        if finish_reason is not None:
            yield chunk_frame("", finish_reason)
            yield "data: [DONE]\n\n"
            return


async def _nonstream_completion(prompt: str, max_tokens: int):
    tok = _state["tokenizer"]
    model_id = _state.get("model_id", "qwen3.5-9b")
    cmpl_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    req, prompt_len = _submit_request(prompt, max_tokens)
    gen_token_ids: list[int] = []
    finish_reason = "stop"
    while True:
        token_id, fr = await req.out_queue.get()
        if token_id is not None:
            gen_token_ids.append(token_id)
        if fr is not None:
            finish_reason = fr
            break

    text = tok.decode(gen_token_ids, skip_special_tokens=True)
    return JSONResponse({
        "id": cmpl_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": len(gen_token_ids),
            "total_tokens": prompt_len + len(gen_token_ids),
        },
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
