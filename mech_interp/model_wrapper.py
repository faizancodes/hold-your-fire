"""White-box wrapper around an HF causal LM (Qwen2.5-Coder) for the looping-circuit study.

Capabilities (Phase 1 of mech-inter-research-plan.md):
  - load a Qwen model on MPS/CPU
  - capture the residual stream at every decoder block, at a chosen token position
  - read next-token logits / score the log-prob of a candidate continuation
  - STEER: add alpha * v to a decoder block's output (residual stream) during any
    forward pass, including .generate() (raw PyTorch forward hooks; no nnsight needed)

Layer indexing convention used everywhere in this project:
  layer L  ==  the OUTPUT of decoder block L (0-indexed), i.e. resid-post of block L.
  capture_resid()[L] is hidden_states[L+1] from output_hidden_states (hidden_states[0]
  is the embedding output, captured as layer -1).  Steering at "layer L" hooks
  self.model.model.layers[L], whose output is exactly that residual stream.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


class ModelWrapper:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 dtype: torch.dtype = torch.float32):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation="eager",   # MPS-safe; avoids sliding-window sdpa issues
        ).to(device).eval()
        self.dtype = dtype
        self.layers = self.model.model.layers           # decoder blocks
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size
        self._handles: list = []

    # ---- prompting --------------------------------------------------------------
    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """messages: [{'role': 'system'|'user'|'assistant', 'content': str}, ...]"""
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _ids(self, text: str):
        return self.tok(text, return_tensors="pt").to(self.device)

    # ---- capture ----------------------------------------------------------------
    @torch.no_grad()
    def capture_resid(self, text: str, pos: int = -1) -> dict[int, np.ndarray]:
        """Residual stream at token `pos` for every block.  Returns {-1: embed, 0..n-1: blocks}."""
        ids = self._ids(text)
        out = self.model(**ids, output_hidden_states=True)
        hs = out.hidden_states                          # len = n_layers + 1
        p = pos if pos >= 0 else ids.input_ids.shape[1] + pos
        acts = {L - 1: hs[L][0, p, :].float().cpu().numpy() for L in range(len(hs))}
        return acts                                     # key -1 = embeddings, 0..n-1 = block outputs

    @torch.no_grad()
    def next_token_logits(self, text: str) -> torch.Tensor:
        out = self.model(**self._ids(text))
        return out.logits[0, -1, :].float().cpu()

    @torch.no_grad()
    def continuation_logprob(self, context: str, continuation: str) -> float:
        """Mean per-token log-prob the model assigns to `continuation` following `context`.
        Length-normalized so candidates of different lengths are comparable."""
        ctx_ids = self.tok(context, return_tensors="pt").input_ids
        full_ids = self.tok(context + continuation, return_tensors="pt").input_ids.to(self.device)
        n_ctx = ctx_ids.shape[1]
        n_cont = full_ids.shape[1] - n_ctx
        if n_cont <= 0:
            return float("nan")
        out = self.model(full_ids)
        logp = torch.log_softmax(out.logits[0].float(), dim=-1)
        # token at position i is predicted by logits at position i-1
        tot = 0.0
        for j in range(n_cont):
            pos = n_ctx + j
            tok_id = full_ids[0, pos]
            tot += logp[pos - 1, tok_id].item()
        return tot / n_cont

    # ---- steering ---------------------------------------------------------------
    def _make_hook(self, vec: torch.Tensor, alpha: float):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                return (out[0] + alpha * vec,) + tuple(out[1:])
            return out + alpha * vec
        return hook

    @contextlib.contextmanager
    def steering(self, layer: int, vec: np.ndarray | torch.Tensor, alpha: float):
        """Add alpha*unit(vec) to the output of decoder block `layer` for the duration.
        `vec` is L2-normalized internally so `alpha` is in residual-norm units."""
        v = torch.as_tensor(np.asarray(vec, dtype=np.float32))
        v = v / (v.norm() + 1e-8)
        v = v.to(self.device, self.dtype)
        h = self.layers[layer].register_forward_hook(self._make_hook(v, alpha))
        try:
            yield
        finally:
            h.remove()

    @contextlib.contextmanager
    def ablate(self, layer: int, vec: np.ndarray | torch.Tensor):
        """Project the output of decoder block `layer` onto the orthogonal complement of
        unit(vec) (remove the component along that direction) for the duration."""
        v = torch.as_tensor(np.asarray(vec, dtype=np.float32))
        v = v / (v.norm() + 1e-8)
        v = v.to(self.device, self.dtype)

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h @ v).unsqueeze(-1) * v
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hd = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            hd.remove()

    @contextlib.contextmanager
    def ablate_head_set(self, pairs):
        """Ablate a set of (layer, head) pairs simultaneously (multi-layer head ablation)."""
        from collections import defaultdict
        by_layer = defaultdict(list)
        for L, h in pairs:
            by_layer[L].append(h)
        n_heads = self.model.config.num_attention_heads
        handles = []
        for L, hs in by_layer.items():
            attn = self.layers[L].self_attn
            hd = attn.o_proj.in_features // n_heads

            def make(hs_, hd_):
                def pre(module, args):
                    x = args[0].clone()
                    for h in hs_:
                        x[..., h * hd_:(h + 1) * hd_] = 0
                    return (x,)
                return pre
            handles.append(attn.o_proj.register_forward_pre_hook(make(hs, hd)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def ablate_heads(self, layer: int, heads):
        """Zero the contribution of specific attention heads at decoder block `layer` by
        masking their slices of the o_proj input (mean-equivalent: heads add ~0)."""
        attn = self.layers[layer].self_attn
        n_heads = self.model.config.num_attention_heads
        hd = attn.o_proj.in_features // n_heads
        heads = list(heads)

        def pre(module, args):
            x = args[0].clone()
            for h in heads:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,)
        handle = attn.o_proj.register_forward_pre_hook(pre)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def generate_kv(self, text: str, max_new_tokens: int = 24,
                    temperature: float = 0.0, seed: int | None = None,
                    bad_ids=None, penalty: float = 0.0, logits_fn=None) -> str:
        """Manual KV-cache decode (greedy if temperature==0, else sampled). Bypasses
        model.generate(), which crashes on MPS (NDArray>2^32). Steering/ablate hooks apply
        at every step. `bad_ids`+`penalty`: targeted penalty on those token logits.
        `logits_fn(logits, seq_ids)->logits`: general logits processor (gets the full token
        sequence so far) for repetition_penalty / no_repeat_ngram baselines."""
        if seed is not None:
            torch.manual_seed(seed)
        bad = torch.tensor(list(bad_ids), device=self.device) if bad_ids else None
        ids = self._ids(text).input_ids
        seq = ids[0].tolist()

        def pick(raw):
            logits = raw
            if bad is not None and penalty:
                logits = logits.clone(); logits[bad] -= penalty
            if logits_fn is not None:
                logits = logits_fn(logits, seq)
            if temperature and temperature > 0:
                p = torch.softmax(logits.float() / temperature, dim=-1)
                return int(torch.multinomial(p, 1))
            return int(logits.argmax())

        out = self.model(ids, use_cache=True)
        past = out.past_key_values
        nxt = pick(out.logits[0, -1])
        gen: list[int] = []
        for _ in range(max_new_tokens):
            if nxt == self.tok.eos_token_id:
                break
            gen.append(nxt); seq.append(nxt)
            o = self.model(torch.tensor([[nxt]], device=self.device),
                           past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = pick(o.logits[0, -1])
        return self.tok.decode(gen, skip_special_tokens=True)

    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int = 80, temperature: float = 0.0) -> str:
        ids = self._ids(text)
        kw: dict = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        gen = self.model.generate(**ids, **kw)
        return self.tok.decode(gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def resid_norm_at(mw: ModelWrapper, text: str, layer: int) -> float:
    """Typical residual-stream norm at a layer (to set alpha in interpretable units)."""
    a = mw.capture_resid(text)[layer]
    return float(np.linalg.norm(a))
