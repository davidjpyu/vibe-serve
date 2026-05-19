"""VibeServeModel — hand-rolled Qwen3.5-9B (hybrid GDN + GQA) inference path.

Accuracy-checker contract:
    model = VibeServeModel.from_pretrained(model_dir, device, dtype)
    output_ids = model.generate(input_ids, max_new_tokens=N)  # greedy, returns (1, T+N)

FastAPI server:
    POST /v1/completions  (OpenAI-compatible, SSE on stream=true)
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

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
    """Partial RoPE. For text-only inputs the three mrope axes share the same
    position values, so the interleaving in HF collapses to standard RoPE on the
    first `int(head_dim * partial_rotary_factor)` dimensions of each head."""

    def __init__(self, head_dim: int, partial_rotary_factor: float, rope_theta: float,
                 mrope_section: list[int]):
        super().__init__()
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.head_dim = head_dim
        self.mrope_section = mrope_section
        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float64) / self.rotary_dim
        ))
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)

    def cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype):
        """position_ids: (B, T) long. Returns cos, sin each (B, T, rotary_dim)."""
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q,k: (B, H, T, D). cos/sin: (B, T, rotary_dim). Only the first rotary_dim of D rotates."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
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
# Full attention (GQA, head_dim=256, output-gated, partial RoPE)
# ---------------------------------------------------------------------------


class FullAttnCache:
    """Pre-allocated KV cache for one layer."""

    def __init__(self, max_len: int, num_kv_heads: int, head_dim: int, device, dtype):
        self.k = torch.empty(1, num_kv_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.empty(1, num_kv_heads, max_len, head_dim, device=device, dtype=dtype)
        self.length = 0

    def reset(self) -> None:
        self.length = 0

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor):
        L = new_k.shape[2]
        end = self.length + L
        self.k[:, :, self.length:end].copy_(new_k)
        self.v[:, :, self.length:end].copy_(new_v)
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]


class Qwen35FullAttention(nn.Module):
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

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor,
                cache: FullAttnCache | None) -> torch.Tensor:
        B, T, _ = hidden.shape
        qg = self.q_proj(hidden).view(B, T, self.num_heads, self.head_dim * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(B, T, self.num_heads * self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rotary.cos_sin(position_ids, q.dtype)
        q, k = apply_partial_rope(q, k, cos, sin)

        if cache is not None:
            full_k, full_v = cache.append(k, v)
        else:
            full_k, full_v = k, v

        full_k = full_k.repeat_interleave(self.num_kv_groups, dim=1)
        full_v = full_v.repeat_interleave(self.num_kv_groups, dim=1)

        S_q = q.shape[2]
        S_k = full_k.shape[2]
        scores = torch.matmul(q, full_k.transpose(-2, -1)) * self.scaling
        if S_q > 1:
            i = torch.arange(S_q, device=scores.device).unsqueeze(-1)
            j = torch.arange(S_k, device=scores.device).unsqueeze(0)
            mask = j > (S_k - S_q + i)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, full_v)
        out = out.transpose(1, 2).contiguous().reshape(B, T, self.num_heads * self.head_dim)

        out = out * torch.sigmoid(gate)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Linear attention (Gated DeltaNet)
# ---------------------------------------------------------------------------


class GDNCache:
    """Pre-allocated GDN recurrence + conv1d cache for one layer."""

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

    def _conv_update_step(self, x: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        """Single-token causal conv1d with silu, in-place state update.
        x: (B, conv_dim, 1). conv_state: (B, conv_dim, kernel-1)."""
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
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)

        init_state = cache.recurrent_state if use_cached else None
        out_state = cache is not None

        if use_cached and T == 1:
            core_out, last_state = fused_recurrent_gated_delta_rule(
                q, k, v, g=g, beta=beta,
                initial_state=init_state, output_final_state=out_state,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_out, last_state = chunk_gated_delta_rule(
                q, k, v, g=g, beta=beta,
                initial_state=init_state, output_final_state=out_state,
                use_qk_l2norm_in_kernel=True,
            )

        if cache is not None and last_state is not None:
            cache.recurrent_state.copy_(last_state.to(cache.recurrent_state.dtype))
            cache.has_state = True

        core_flat = core_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_flat = self.norm(core_flat, z_flat)
        core_out = core_flat.reshape(B, T, self.value_dim)
        return self.out_proj(core_out)


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

        rope_params = config_text.rope_parameters
        self.rotary = Qwen35RotaryEmb(
            head_dim=self.head_dim,
            partial_rotary_factor=rope_params.get("partial_rotary_factor", 1.0),
            rope_theta=rope_params["rope_theta"],
            mrope_section=rope_params.get("mrope_section", [11, 11, 10]),
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

    def _build_caches(self, max_len: int):
        full_caches: list[FullAttnCache | None] = []
        gdn_caches: list[GDNCache | None] = []
        conv_dim = (self.linear_num_k_heads * self.linear_head_k_dim) * 2 + (
            self.linear_num_v_heads * self.linear_head_v_dim
        )
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
                    conv_dim=conv_dim,
                    conv_kernel=self.linear_conv_kernel,
                    device=self.device,
                    dtype=self.dtype,
                    state_dtype=torch.float32,
                ))
        return full_caches, gdn_caches

    def _forward_layers(self, hidden: torch.Tensor, position_ids: torch.Tensor,
                        full_caches, gdn_caches) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, position_ids, full_caches[i], gdn_caches[i])
        return self.norm(hidden)

    @torch.inference_mode()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 16) -> torch.LongTensor:
        assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "Round 1 supports batch=1."
        input_ids = input_ids.to(self.device)
        prompt_len = input_ids.shape[1]
        max_len = prompt_len + max_new_tokens

        full_caches, gdn_caches = self._build_caches(max_len)

        # Prefill
        position_ids = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        hidden = self.embed_tokens(input_ids).to(self.dtype)
        hidden = self._forward_layers(hidden, position_ids, full_caches, gdn_caches)
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
            hidden = self._forward_layers(hidden, cur_pos, full_caches, gdn_caches)
            logits = self.lm_head(hidden[:, -1:, :])
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output[:, prompt_len + step:prompt_len + step + 1] = next_tok
            out_len = prompt_len + step + 1
            if next_tok.item() == self.EOS_TOKEN_ID:
                break

        return output[:, :out_len]

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
# FastAPI server (OpenAI-compatible /v1/completions, SSE on stream=true)
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


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    async def _startup() -> None:
        model_dir = DEFAULT_MODEL_DIR
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16
        print(f"[vibeserve] loading model from {model_dir} on {device} ({dtype})")
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(model_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = VibeServeModel.from_pretrained(model_dir, device, dtype)
        print(f"[vibeserve] loaded in {time.perf_counter() - t0:.1f}s")
        _state["model"] = model
        _state["tokenizer"] = tok
        _state["device"] = device
        _state["lock"] = asyncio.Lock()
        _state["model_id"] = os.path.basename(model_dir.rstrip("/")) or "qwen3.5-9b"

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
        stop_seqs: list[str] = []
        if req.stop is not None:
            stop_seqs = [req.stop] if isinstance(req.stop, str) else list(req.stop)

        if req.stream:
            return StreamingResponse(
                _stream_completion(prompt, req.max_tokens, stop_seqs),
                media_type="text/event-stream",
            )
        text, finish_reason, prompt_tokens, completion_tokens = await _run_completion(
            prompt, req.max_tokens, stop_seqs,
        )
        return JSONResponse({
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": _state.get("model_id", "qwen3.5-9b"),
            "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })

    return app


async def _run_completion(prompt: str, max_tokens: int, stop_seqs: list[str]):
    tok = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]
    lock: asyncio.Lock = _state["lock"]

    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    async with lock:
        output_ids = await asyncio.to_thread(
            model.generate, input_ids, max_tokens,
        )
    gen_ids = output_ids[0, prompt_len:].tolist()
    text, finish_reason, emitted = _decode_with_stop(gen_ids, tok, stop_seqs, max_tokens)
    return text, finish_reason, prompt_len, emitted


async def _stream_completion(prompt: str, max_tokens: int, stop_seqs: list[str]):
    cmpl_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_id = _state.get("model_id", "qwen3.5-9b")
    text, finish_reason, _pt, _ct = await _run_completion(prompt, max_tokens, stop_seqs)

    def frame(text_chunk: str, finish: str | None) -> str:
        payload = {
            "id": cmpl_id, "object": "text_completion", "created": created, "model": model_id,
            "choices": [{"text": text_chunk, "index": 0, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    if text:
        yield frame(text, None)
    yield frame("", finish_reason)
    yield "data: [DONE]\n\n"


def _decode_with_stop(token_ids: list[int], tokenizer, stop_seqs: list[str], max_tokens: int):
    if not token_ids:
        return "", "stop", 0
    finish_reason = "length" if len(token_ids) >= max_tokens else "stop"
    truncated = token_ids
    if truncated and truncated[-1] == VibeServeModel.EOS_TOKEN_ID:
        truncated = truncated[:-1]
        finish_reason = "stop"
    text = tokenizer.decode(truncated, skip_special_tokens=True)
    if stop_seqs:
        cut = len(text)
        hit_stop = False
        for s in stop_seqs:
            if not s:
                continue
            i = text.find(s)
            if i != -1 and i < cut:
                cut = i
                hit_stop = True
        if hit_stop:
            text = text[:cut]
            finish_reason = "stop"
    emitted = len(truncated) if not stop_seqs else len(
        tokenizer(text, add_special_tokens=False).input_ids
    )
    return text, finish_reason, emitted


app = _make_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
