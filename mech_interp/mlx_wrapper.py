"""White-box activation steering for *quantized* models on Apple Silicon, via MLX.

Why this exists: Ollama/llama.cpp give us no activation access, and HF-transformers on
MPS caps at ~7B fp16 on a 32GB Mac. MLX runs a 4-bit 30B/35B MoE comfortably AND lets us
control the forward pass -- so we can insert steering on the residual stream of the *big,
capable* models, which is the one thing the rest of the stack can't do.

Key insight that makes this tractable: residual-stream steering needs NO MoE-routing
changes. Every decoder block (dense or MoE) ends with ``out = h + mlp(norm(h))`` and returns
the dense residual stream ``out`` in fp16. The MoE only lives *inside* ``mlp``. So we hook
the block's *output* -- identical for qwen2 (dense) and qwen3_moe (MoE) -- and the 4-bit
weights are irrelevant to the fp16 vector we add.

Hooking mechanism: mlx-lm models run ``for layer, c in zip(self.layers, cache): h =
layer(h, mask, c)``. We replace each ``model.model.layers[i]`` with a thin ``HookedLayer``
nn.Module that delegates to the original block and then (a) optionally records its output
(capture) and (b) optionally adds alpha*v or projects v out (steer / ablate). A shared
mutable ``SteerState`` toggles modes, so we can reuse mlx-lm's own ``generate_step``.

API mirrors mech_interp/model_wrapper.py (the HF/MPS wrapper): capture_resid,
continuation_logprob, steering(), ablate(), generate.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler


@dataclass
class SteerState:
    layers: set = field(default_factory=set)   # which layer indices the active op applies to
    capture: bool = False
    capture_full: bool = False                 # store all positions vs last token only
    store: dict = field(default_factory=dict)  # idx -> mx.array
    steer: bool = False                        # add alpha*vec
    ablate: bool = False                       # project vec out
    alpha: float = 0.0
    vec: mx.array | None = None                # (D,) steering direction (steer)
    unit: mx.array | None = None               # (D,) unit direction (ablate)


class HookedLayer(nn.Module):
    """Wraps one decoder block: run it, then capture / steer / ablate its residual output."""

    def __init__(self, inner: nn.Module, idx: int, state: SteerState):
        super().__init__()
        self.inner = inner       # the original block (a child module -> weights preserved)
        self._idx = idx          # leading underscore -> plain attr, not a param/child
        self._st = state

    def __call__(self, x, *args, **kwargs):
        out = self.inner(x, *args, **kwargs)
        st = self._st
        if self._idx in st.layers:
            if st.capture:
                st.store[self._idx] = out if st.capture_full else out[:, -1, :]
            if st.steer and st.vec is not None:
                out = out + (st.alpha * st.vec).astype(out.dtype)
            if st.ablate and st.unit is not None:
                u = st.unit.astype(out.dtype)
                proj = (out * u).sum(axis=-1, keepdims=True)   # (..., 1)
                out = out - proj * u
        return out


class MLXModel:
    def __init__(self, repo: str, tokenizer_config: dict | None = None):
        self.repo = repo
        self.model, self.tokenizer = load(repo, tokenizer_config=tokenizer_config or {})
        self.st = SteerState()
        self.n_layers = len(self.model.model.layers)
        self.d_model = int(self.model.args.hidden_size)
        # wrap every block in place; the Model.layers property reflects this same list.
        for i in range(self.n_layers):
            self.model.model.layers[i] = HookedLayer(self.model.model.layers[i], i, self.st)

    # ---- tokenization -------------------------------------------------------------
    def tok(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def render(self, messages: list[dict]) -> list[int]:
        return list(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))

    def _split_ids(self, prompt: str, continuation: str) -> tuple[list[int], list[int]]:
        p = self.tok(prompt)
        full = self.tok(prompt + continuation)
        # robust to BPE boundary merges: continuation = whatever follows the prompt prefix
        if full[: len(p)] == p:
            return p, full[len(p):]
        return p, self.tok(continuation)

    # ---- forward / capture --------------------------------------------------------
    def _logits(self, ids: list[int]) -> mx.array:
        # cast to f32: big models run bf16, which numpy can't consume and which is noisy for logsumexp
        return self.model(mx.array([ids]))[0].astype(mx.float32)    # (S, V)

    def capture_resid(self, ids: list[int], layers, full: bool = False) -> dict[int, np.ndarray]:
        """Residual-stream output of each requested block (last token by default)."""
        self.st.store = {}
        self.st.layers = set(layers)
        self.st.capture, self.st.capture_full = True, full
        self.st.steer = self.st.ablate = False
        _ = self.model(mx.array([ids]))
        mx.eval(list(self.st.store.values()))
        out = {}
        for i in layers:
            a = self.st.store[i].astype(mx.float32)   # bf16 -> f32 (numpy has no bfloat16)
            out[i] = np.array(a[0])                    # (D,) or (S, D)
        self.st.capture = False
        self.st.store = {}
        self.st.layers = set()
        return out

    def continuation_logprob(self, prompt: str, continuation: str) -> float:
        """Sum log p(continuation | prompt) under the (optionally steered) model."""
        p_ids, c_ids = self._split_ids(prompt, continuation)
        if not c_ids:
            return 0.0
        ids = p_ids + c_ids
        logits = self._logits(ids)                                   # (S, V)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        plen = len(p_ids)
        pred = logp[plen - 1: plen - 1 + len(c_ids)]                 # (len_c, V)
        tgt = mx.array(c_ids)
        score = mx.take_along_axis(pred, tgt[:, None], axis=-1).sum()
        return float(score)

    # ---- interventions (context managers) -----------------------------------------
    @contextlib.contextmanager
    def steering(self, layer: int, vec, alpha: float):
        v = mx.array(np.asarray(vec, dtype=np.float32))
        self.st.vec, self.st.alpha = v, float(alpha)
        self.st.layers, self.st.steer = {int(layer)}, True
        self.st.capture = self.st.ablate = False
        try:
            yield
        finally:
            self.st.steer, self.st.vec = False, None
            self.st.layers = set()

    @contextlib.contextmanager
    def ablate(self, layer: int, vec):
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        self.st.unit = mx.array(v)
        self.st.layers, self.st.ablate = {int(layer)}, True
        self.st.capture = self.st.steer = False
        try:
            yield
        finally:
            self.st.ablate, self.st.unit = False, None
            self.st.layers = set()

    # ---- generation ---------------------------------------------------------------
    def generate(self, ids: list[int], max_tokens: int = 48, temperature: float = 0.0,
                 seed: int | None = None, bad_ids=None, penalty: float = 0.0) -> str:
        """Greedy/sampled decode (steering applies automatically if a steering()/ablate()
        context is active). ``bad_ids`` + ``penalty`` apply a targeted logit penalty."""
        if seed is not None:
            mx.random.seed(seed)
        sampler = make_sampler(temp=float(temperature))
        lps = None
        if bad_ids and penalty:
            lps = make_logits_processors(logit_bias={int(t): -float(penalty) for t in set(bad_ids)})
        cache = make_prompt_cache(self.model)
        # stop on the UNION of EOS ids: TokenizerWrapper.eos_token_id (<|im_end|>) and
        # .eos_token_ids ({<|endoftext|>}) can differ; stopping on only one leaks the other
        # special token (and prompt echo) into the output as fake "commands".
        eos_ids = set(getattr(self.tokenizer, "eos_token_ids", None) or ())
        eos_ids.add(self.tokenizer.eos_token_id)
        toks: list[int] = []
        for tok, _ in generate_step(mx.array(ids), self.model, max_tokens=max_tokens,
                                    sampler=sampler, logits_processors=lps, prompt_cache=cache):
            t = int(tok)
            if t in eos_ids:
                break
            toks.append(t)
        return self.tokenizer.decode(toks)
