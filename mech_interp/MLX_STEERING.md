# MLX activation steering for quantized (MoE) models on Apple Silicon

White-box residual-stream steering for **4-bit** models that the rest of the stack can't
reach: Ollama/llama.cpp expose no activations, and HF-transformers on MPS caps at ~7B fp16
on a 32 GB Mac. MLX runs a 4-bit 30B/35B MoE comfortably **and** lets us control the forward
pass, so we can steer the residual stream of the big, capable models — the one thing needed
to test a "recovery / do-the-right-thing" direction at a scale that can actually recover.

## The key insight (why this is tractable)

Residual-stream steering needs **no MoE-routing changes**. Every decoder block — dense or
MoE — ends with

```python
h   = x + self_attn(norm(x))      # attention residual
out = h + mlp(norm(h))            # mlp residual  ->  return out
```

and returns the **dense, fp16** residual stream `out`. The MoE only lives *inside* `mlp`.
Verified in the mlx-lm source for `qwen2.py` (dense), `qwen2_moe.py`, `qwen3_moe.py`, and
`olmoe.py` — all identical block structure, all driven by `for layer, c in zip(self.layers,
cache): h = layer(h, mask, c)`. So we hook the block's **output**, and the 4-bit weights are
irrelevant to the fp16 vector we add. One code path covers dense and every MoE family.

## How the hook works

mlx-lm models keep their decoder blocks in `model.model.layers` (a plain list, also exposed
via the `Model.layers` property). We replace each entry with a thin `HookedLayer` (an
`nn.Module`) that delegates to the original block, then optionally (a) records its output
(capture), (b) adds `alpha * v` (steer), or (c) projects `v` out (ablate). A shared mutable
`SteerState` toggles modes, so we reuse mlx-lm's own `generate_step` for decoding (steering
applies automatically inside any `model(...)` call). No fork of mlx-lm; no MoE internals.

## Files

| file | what |
|---|---|
| `mlx_wrapper.py` | `MLXModel`: `capture_resid`, `continuation_logprob`, `steering()`/`ablate()` context managers, `generate(..., bad_ids, penalty)` (the targeted loop-break in MLX). Mirrors the HF `model_wrapper.py` API. |
| `test_mlx_wrapper.py` | end-to-end mechanism validation (shapes, logprob sanity, hook-is-live, dose-response, loop-break decode). |
| `run_mlx_steer.py` | parameterized steering eval for any repo: builds a loop-vs-progress direction, **coherence-aware** layer pick, dose-response vs random/orthogonal controls, loop-break decode. Has a **memory guard** that refuses to load a model larger than free RAM (so it never swap-thrashes). Writes `results/mlx_steer_<repo>.json`. |

## Validation (done)

**Quantized dense — `Qwen2.5-Coder-1.5B-Instruct-4bit` (clean, the science check):**
- hook is live: `alpha=0` is an exact no-op; a random vector at `alpha=30` moves the metric.
- **direction-specific dose-response**: steering `-direction` at L=14 *monotonically reduces*
  repeat-preference (net **+1.06 drop**), while random (**-2.97**) and orthogonal controls
  move it the *other* way. Coherence preserved (novel-action logp stays ~-16, no collapse).
- targeted logit-penalty decode changes the emitted command (the loop-breaker, in MLX).

**Quantized MoE — `OLMoE-1B-7B-0125-Instruct-4bit` (mechanism on a real, non-Qwen MoE):**
- the wrapper **loads, captures, steers, and generates** on an actual 4-bit MoE end-to-end —
  proving the MoE path empirically, not just by source.
- honest negative on the *science*: at 1B active params the model is too fragile — steering at
  `alpha>=8` collapses coherence at **every** candidate layer (the coherence-aware picker
  reports "NONE"). Expected for a tiny MoE; mirrors the weak single-direction steering seen at
  1.5B.

**Quantized 30B MoE — `Qwen3-Coder-30B-A3B-Instruct-3bit` (the target, on a 32 GB M4):**
- **The artifact runs at the target scale.** The 13.4 GB 3-bit 30B MoE loads and the wrapper
  captures / steers / decodes across all 48 layers (d=2048) in ~15 GB of RAM. White-box steering
  of a quantized 30B MoE on Apple Silicon — the capability nothing else in the stack provides —
  works. (Key: we dropped to **3-bit**; steering is bit-width-agnostic, so this is the same
  experiment as 4-bit, just at ~13 GB instead of ~17 GB. The "need 21 GB" wall was a 4-bit
  artifact, not fundamental.)
- the coherence-aware layer pick does its job: it rejects the early layers (L8/L12 collapse
  novel-logp to ~-34) and selects from the coherence-preserving set {16, 24, 32}.
- **honest negative on the science:** the *crude* loop-vs-progress direction does **not** cleanly
  steer the 30B out of looping — at the probe alpha no layer actually reduces repeat-preference,
  and higher alpha (>=12) breaks coherence (novel-logp -24 -> -44). This is consistent with the
  whole study's recurring result that *single-direction loop steering is weak*, now confirmed at
  30B and through a 4-bit/3-bit MoE. It is exactly the signal that the next step is not a bigger
  model but a **better contrast** — a *recovery* direction built from `stuck -> recovered` vs
  `stuck -> stayed-stuck` decision points, not `loop vs progress`.

## Running the big model (30B)

Use the **3-bit** weights — `mlx-community/Qwen3-Coder-30B-A3B-Instruct-3bit` (~13.4 GB). Needs
**~15 GB free RAM**, not 21 GB; the GPU ceiling on a 32 GB M4 is ~22.9 GB so the wall was app
memory, not hardware. Steering is bit-width-agnostic, so 3-bit is the same experiment for ~4 GB
less RAM. (4-bit `...-4bit` ~16 GB needs ~18 GB free; `gpt-oss-20b-MXFP4-Q4` ~11 GB and
`Qwen1.5-MoE-A2.7B-Chat-4bit` ~8.5 GB are smaller real-MoE fallbacks.)

```bash
mech_interp/.venv/bin/python -m mech_interp.run_mlx_steer \
  mlx-community/Qwen3-Coder-30B-A3B-Instruct-3bit
```

Output: `results/mlx_steer_*.json` with the layer pick, dose-response vs controls, coherence,
and the loop-break decode.

## Status

Artifact **built, validated on a quantized model, and run on the literal 3-bit 30B MoE on a
32 GB M4** — the target capability (white-box steering of a quantized 30B MoE on Apple Silicon)
is delivered. The MoE-routing concern was a non-issue (residual steering is block-level); the
"need 21 GB" wall was a 4-bit artifact dissolved by 3-bit. Honest open item: the *crude*
loop-vs-progress direction steers the 30B only weakly (and breaks coherence at high alpha), so
the next experiment is a **recovery-contrast** direction (`stuck -> recovered` vs
`stuck -> stayed-stuck`), which this harness is now ready to run.
