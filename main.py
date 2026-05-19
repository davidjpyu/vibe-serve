"""Baseline FastAPI server for Llama-3.1-8B-Instruct on H200.

Hand-written model layers (RMSNorm, RoPE with Llama-3 scaling, GQA attention,
SwiGLU MLP, decoder stack); transformers is used only for the tokenizer and to
load weights from a local directory. SDPA is used for attention with
eager-equivalent (math) semantics. KV cache is preallocated per layer and
written in-place — no torch.cat in the decode path.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
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
    # q, k: (bsz, num_heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
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
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_start: int,
        cache_end: int,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (bsz, num_heads, seq_len, head_dim)
        k = (
            self.k_proj(hidden_states)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = apply_rope(q, k, cos, sin)

        # In-place write to preallocated KV cache (no torch.cat).
        k_cache[:, :, cache_start:cache_end, :] = k
        v_cache[:, :, cache_start:cache_end, :] = v

        k_full = k_cache[:, :, :cache_end, :]
        v_full = v_cache[:, :, :cache_end, :]

        # Causal mask only matters in prefill (seq_len > 1, kv_len == seq_len).
        # In decode (seq_len == 1) the single query attends to all of [0:cache_end].
        is_causal = seq_len > 1 and cache_start == 0
        with sdpa_kernel([SDPBackend.MATH]):
            attn_out = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scaling,
                enable_gqa=True,
            )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
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
        cache_start: int,
        cache_end: int,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, cos, sin, k_cache, v_cache, cache_start, cache_end)
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


class VibeServeModel(nn.Module):
    """Llama-3.1 forward / greedy generation wrapper.

    Per-layer KV cache buffers are preallocated to ``max_cache_len`` so the
    decode loop performs in-place writes (no per-token ``torch.cat``).
    """

    def __init__(
        self,
        config,
        device: torch.device,
        dtype: torch.dtype,
        max_cache_len: int = 4096,
    ):
        super().__init__()
        self.config = config
        self.device_ = device
        self.dtype_ = dtype
        self.max_cache_len = max_cache_len

        self.model = LlamaInner(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # KV cache buffers — non-persistent so they don't appear in state_dict.
        for i in range(self.num_layers):
            self.register_buffer(
                f"_k_cache_{i}",
                torch.zeros(
                    1, self.num_kv_heads, max_cache_len, self.head_dim,
                    device=device, dtype=dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                f"_v_cache_{i}",
                torch.zeros(
                    1, self.num_kv_heads, max_cache_len, self.head_dim,
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
    ) -> "VibeServeModel":
        model_dir = str(model_dir)
        device = torch.device(device)
        config = AutoConfig.from_pretrained(model_dir)

        model = cls(config, device=device, dtype=dtype, max_cache_len=max_cache_len)

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
            # tied embeddings fallback
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
        # Ignore extras such as inv_freq buffers.

        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_inner(
        self, input_ids: torch.Tensor, cache_start: int
    ) -> torch.Tensor:
        """Run the stack and return last-token logits.

        Writes K/V into the preallocated cache at positions
        [cache_start, cache_start + input_ids.shape[1]).
        """
        seq_len = input_ids.shape[1]
        cache_end = cache_start + seq_len

        positions = torch.arange(
            cache_start, cache_end, device=self.device_, dtype=torch.long
        )
        cos = self.cos_cache.index_select(0, positions)  # (seq_len, head_dim)
        sin = self.sin_cache.index_select(0, positions)

        x = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            k_cache = self._buffers[f"_k_cache_{i}"]
            v_cache = self._buffers[f"_v_cache_{i}"]
            x = layer(x, cos, sin, k_cache, v_cache, cache_start, cache_end)
        x = self.model.norm(x)
        return self.lm_head(x[:, -1:, :])  # (1, 1, vocab)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        eos_token_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """Greedy generation. Returns (1, prompt_len + generated_len) tensor."""
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

        # Prefill
        logits = self._forward_inner(input_ids, cache_start=0)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        generated_tokens = [next_token]
        # `cache_count` = number of tokens currently committed to the KV cache
        cache_count = prompt_len

        for _ in range(max_new_tokens - 1):
            if int(next_token.item()) in eos_token_ids:
                break
            # Write the just-sampled token at position `cache_count`
            logits = self._forward_inner(next_token, cache_start=cache_count)
            cache_count += 1
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)

        return torch.cat([input_ids] + generated_tokens, dim=1)


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------

EOS_TOKEN_IDS = (128001, 128008, 128009)


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
    # Fallback: read symlink target file from reference/
    sym = Path("reference/model.symlink_target")
    if sym.exists():
        target = sym.read_text().strip()
        if Path(target).exists():
            return target
    raise RuntimeError("model directory not found; set MODEL_DIR env var")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = _model_dir()
    print(f"[startup] loading model from {model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = VibeServeModel.from_pretrained(
        model_dir, device="cuda:0", dtype=torch.float16, max_cache_len=4096
    )
    STATE["model"] = model
    STATE["tokenizer"] = tokenizer
    name = getattr(model.config, "_name_or_path", None) or "llama-3.1-8b-instruct"
    STATE["model_name"] = name
    STATE["lock"] = asyncio.Lock()
    print("[startup] model ready", flush=True)
    yield


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
# Streaming generation core
# ---------------------------------------------------------------------------


def _utf8_safe_decode(
    tokenizer, all_ids: list[int], emitted_text_len: int
) -> tuple[str, int]:
    """Decode all_ids and return any suffix not yet emitted, holding back
    incomplete UTF-8 byte sequences (tokenizer renders them as U+FFFD)."""
    full = tokenizer.decode(all_ids, skip_special_tokens=True)
    if "�" in full[emitted_text_len:]:
        return "", emitted_text_len
    new_text = full[emitted_text_len:]
    return new_text, len(full)


def _step_generate(
    model: VibeServeModel,
    prompt_ids_list: list[int],
    max_new_tokens: int,
    eos_ids: tuple[int, ...] = EOS_TOKEN_IDS,
):
    """Generator yielding token ids for greedy decoding on the static KV cache."""
    device = model.device_
    prompt = torch.tensor([prompt_ids_list], device=device, dtype=torch.long)
    prompt_len = prompt.shape[1]

    if prompt_len + max_new_tokens > model.max_cache_len:
        max_new_tokens = max(1, model.max_cache_len - prompt_len)

    with torch.inference_mode():
        logits = model._forward_inner(prompt, cache_start=0)
        next_token = int(logits[:, -1, :].argmax(dim=-1).item())

    yield next_token
    if next_token in eos_ids:
        return
    cache_count = prompt_len

    for _ in range(max_new_tokens - 1):
        with torch.inference_mode():
            nt = torch.tensor([[next_token]], device=device, dtype=torch.long)
            logits = model._forward_inner(nt, cache_start=cache_count)
        cache_count += 1
        next_token = int(logits[:, -1, :].argmax(dim=-1).item())
        yield next_token
        if next_token in eos_ids:
            return


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


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


async def _completion_stream(prompt_text: str, max_tokens: int):
    model: VibeServeModel = STATE["model"]
    tokenizer = STATE["tokenizer"]
    lock: asyncio.Lock = STATE["lock"]
    model_name = STATE["model_name"]

    cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    prompt_ids: list[int] = tokenizer(prompt_text, add_special_tokens=True).input_ids

    async with lock:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def producer():
            try:
                for tok in _step_generate(model, prompt_ids, max_tokens):
                    asyncio.run_coroutine_threadsafe(q.put(tok), loop).result()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(q.put(("ERR", str(e))), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(q.put(SENTINEL), loop).result()

        fut = loop.run_in_executor(None, producer)

        emitted_ids: list[int] = []
        emitted_text_len = 0
        completion_tokens = 0
        finish_reason: str | None = None

        while True:
            item = await q.get()
            if item is SENTINEL:
                break
            if isinstance(item, tuple) and item[0] == "ERR":
                finish_reason = "stop"
                break
            tok = int(item)
            if tok in EOS_TOKEN_IDS:
                finish_reason = "stop"
                break
            emitted_ids.append(tok)
            completion_tokens += 1
            new_text, emitted_text_len = _utf8_safe_decode(
                tokenizer, emitted_ids, emitted_text_len
            )
            if new_text:
                yield _completion_chunk(cmpl_id, model_name, new_text, None)

        if finish_reason is None:
            finish_reason = "length" if completion_tokens >= max_tokens else "stop"

        yield _completion_chunk(cmpl_id, model_name, "", finish_reason)
        yield "data: [DONE]\n\n"
        await fut


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if isinstance(req.prompt, list):
        prompt_text = req.prompt[0] if req.prompt else ""
    else:
        prompt_text = req.prompt

    if req.stream:
        return StreamingResponse(
            _completion_stream(prompt_text, req.max_tokens),
            media_type="text/event-stream",
        )

    model: VibeServeModel = STATE["model"]
    tokenizer = STATE["tokenizer"]
    lock: asyncio.Lock = STATE["lock"]
    model_name = STATE["model_name"]

    prompt_ids: list[int] = tokenizer(prompt_text, add_special_tokens=True).input_ids
    cmpl_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    async with lock:
        def run():
            out: list[int] = []
            for tok in _step_generate(model, prompt_ids, req.max_tokens):
                if tok in EOS_TOKEN_IDS:
                    return out, "stop"
                out.append(tok)
            return out, ("length" if len(out) >= req.max_tokens else "stop")

        loop = asyncio.get_running_loop()
        out_ids, finish_reason = await loop.run_in_executor(None, run)

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


async def _chat_stream(prompt_ids: list[int], max_tokens: int):
    model: VibeServeModel = STATE["model"]
    tokenizer = STATE["tokenizer"]
    lock: asyncio.Lock = STATE["lock"]
    model_name = STATE["model_name"]

    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    async with lock:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def producer():
            try:
                for tok in _step_generate(model, prompt_ids, max_tokens):
                    asyncio.run_coroutine_threadsafe(q.put(tok), loop).result()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(q.put(("ERR", str(e))), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(q.put(SENTINEL), loop).result()

        fut = loop.run_in_executor(None, producer)

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
        completion_tokens = 0
        finish_reason: str | None = None

        while True:
            item = await q.get()
            if item is SENTINEL:
                break
            if isinstance(item, tuple) and item[0] == "ERR":
                finish_reason = "stop"
                break
            tok = int(item)
            if tok in EOS_TOKEN_IDS:
                finish_reason = "stop"
                break
            emitted_ids.append(tok)
            completion_tokens += 1
            new_text, emitted_text_len = _utf8_safe_decode(
                tokenizer, emitted_ids, emitted_text_len
            )
            if new_text:
                yield _chat_chunk(cmpl_id, model_name, new_text, None)

        if finish_reason is None:
            finish_reason = "length" if completion_tokens >= max_tokens else "stop"

        yield _chat_chunk(cmpl_id, model_name, "", finish_reason)
        yield "data: [DONE]\n\n"
        await fut


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    tokenizer = STATE["tokenizer"]
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids: list[int] = tokenizer(prompt_text, add_special_tokens=False).input_ids

    if req.stream:
        return StreamingResponse(
            _chat_stream(prompt_ids, req.max_tokens),
            media_type="text/event-stream",
        )

    model: VibeServeModel = STATE["model"]
    lock: asyncio.Lock = STATE["lock"]
    model_name = STATE["model_name"]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    async with lock:
        def run():
            out: list[int] = []
            for tok in _step_generate(model, prompt_ids, req.max_tokens):
                if tok in EOS_TOKEN_IDS:
                    return out, "stop"
                out.append(tok)
            return out, ("length" if len(out) >= req.max_tokens else "stop")

        loop = asyncio.get_running_loop()
        out_ids, finish_reason = await loop.run_in_executor(None, run)

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
