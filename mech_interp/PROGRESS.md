# Looping-circuit study — running log

Model: **Qwen2.5-Coder-1.5B-Instruct** (white-box, HF transformers + MPS), isolated venv
at `mech_interp/.venv` (shadows `huggingface_hub<1.0` to satisfy transformers 4.49).
Steering/capture via raw PyTorch forward hooks on decoder blocks (no nnsight needed).

Layer convention: `layer L` = output (resid-post) of decoder block L; steering at L hooks
`model.model.layers[L]`. Capture also keeps `-1` = embedding output.

## Phase 1 — wrapper ✅
`model_wrapper.py`: load, `capture_resid` (all layers, any position), `next_token_logits`,
`continuation_logprob` (length-normalized), `steering(layer, v, alpha)` context manager
(adds `alpha * unit(v)` to a block's output during any forward, incl. generation).
Verified: capture works (resid-norm ~41 @L14), steering hook changes logits, log-probs sane.
Note: `.generate()` on MPS hits an NDArray>2³² assertion via Qwen sliding-window attn under
`sdpa` → switched to `attn_implementation="eager"` (fine; sliding window only matters >32k tok).
Forward-only methods (capture + logprob) are all the localization/calibration phases need.

## MVP — synthetic loop-vs-progress (de-risking) ⚠️ honest negative + a real behavioral effect
`synthetic.py` builds coding-agent transcripts; LOOP = same command ×K + same fail obs;
PROGRESS = K distinct productive commands (length-matched). `run_mvp.py`.

**What worked (real, large):** a LOOP context strongly induces a *repeat preference*.
On one scenario, logp(repeat) − logp(novel) = **+0.74 in LOOP vs −2.64 in PROGRESS**
(same scenario, matched length). Across 24 held-out scenarios the baseline repeat-pref is
**+0.24** (model favors repeating in loops). So there is a big looping tendency to steer.

**The catch (caught by rigor):** the naive loop-vs-progress probe is **perfectly separable
at EVERY layer (AUC 1.000, L0..L27)** — a textbook sign of a *trivial confound*
(loop and progress contexts just look very different), NOT a deep "loop-awareness"
representation. Steering with that diff-of-means direction does **not** break loops:
Δrepeat-pref @α=(2,4,6,8,12) = [-0.005,+0.008,+0.042,+0.081,+0.19] (it *increases* repeating
at high α), while random=[~0] and orthogonal=[~0]. And the dose-response is **flat**:
loop-projection ≈ −0.28 at every K=1..6 (a binary surface cue, not a graded repetition signal).

**Conclusion:** do NOT trust the L0 signal. Need the confound-controlled contrast
(`run_localize.py`): add a **VARIED-FAIL** condition (K *different* failing commands) so the
key contrast becomes **loop vs varied-fail** — matched on turn-count and on "failing
repeatedly", isolating *identical-command* repetition — plus an explicit per-layer
length-confound diagnostic (length-only AUC, |corr(projection, length)|). Find the layer
where a genuine, length-decorrelated repetition signal lives, and steer there.

## Phase 2/3 — confound-controlled localization ✅
**v1 (`run_localize.py`):** found the conditions were NOT length-matched (loop=231,
vfail=253, prog=195 tokens — `sed`/`git` commands are longer). Length *alone* gave
AUC 0.90–1.00. So the v1 "signal" was mostly a length artifact — caught, discarded.

**v2 (`run_localize2.py`):** EXACT length control — every command padded to 32 tokens,
every observation to 16, fixed thought → all conditions exactly **321.5 tokens**
(length-only AUC = **0.500**, len-corr = 0.01). Two contrasts:
- **REP** (loop-fail vs varied-fail; same vs different command, both failing): AUC **1.000**.
- **STUCK** (loop-fail vs loop-*progress*; SAME repeated command, failing vs improving obs —
  token-repetition held identical): AUC **1.000** at L2.

Honest interpretation: with length controlled, the model linearly encodes BOTH "my commands
repeat" and "my repetition is failing vs working" — at AUC 1.0 across layers. But decodability
is *cheap* (the inputs differ in surface tokens); it cannot distinguish a banal cue from a
causal mechanism. **The decisive test is causal steering/ablation (Phases 4–6).** Directions
are built from the length-controlled `acts2.npz`; the causal test runs on natural loop contexts.

## Phase 4/5 — causal test
`run_steer_eval.py`. Baseline natural-loop repeat-pref = **+0.348** (model favors repeating).

**Scan (Δrepeat-pref @α=8; negative = less repeating):** a real, *sign-consistent* causal axis:
- REP (−/+ sign): L4 −0.07/+0.06 … L16 −0.06/+0.12 — steering *away from* "same command"
  modestly reduces repeating; *toward* it increases it.
- STUCK (−/+): **L8 −0.17/+0.16**, L16 −0.12/+0.13 — steering *toward* "productive repetition"
  reduces repeating; toward "stuck" increases it.
- **Best = −STUCK @ L8 (Δpref −0.168)** — roughly halves the +0.35 repeat bias.

**Full causal result (−STUCK @ L8):**
- **steering** Δpref @α=(2,4,6,8,12) = [−0.05, −0.10, −0.14, −0.19, **−0.23**] — clean monotonic.
- **controls** random=[+0.01…+0.08], orthogonal=[+0.01…+0.08] — the effect is **specific**.
- **coherence** novel-action logp stable ~−1.8 across α — steering makes the *repeat* less
  attractive without breaking fluency.
- **ablation** (project the direction out): Δpref = **+0.019** [+0.004,+0.032] — removing it
  *slightly increases* repeating. So this is a **steering lever, not strictly "necessary"**.
- Mechanism: a layer-8 **"are my tests resolving?"** axis causally modulates the urge to re-run.

## Phase 6 — on-policy (greedy) ⚠️ honest limitation + a surprising asymmetry
`run_onpolicy.py` (α=20, manual KV-cache decode — `model.generate()` crashes on MPS).
- **Loop-escape = 0%** for no-steer / steer / random. The greedy loop is a strong attractor;
  the −0.23-nat nudge doesn't flip the argmax at coherence-preserving α.
- **Coherence = 1.0** (no gibberish).
- **Disruption on healthy runs: steer 100%, random 83%.** Steering changes healthy actions but
  not loop actions → **the loop is MORE robust to steering than healthy decisions are.** Naive
  single-direction steering disrupts healthy runs *before* it breaks loops — the opposite of the
  low-disruption goal. (An honest, important negative for the Intervention-Paradox angle.)

## Phase 6b — on-policy under sampling (realistic)
`run_sample_escape.py` (α=12, temp 0.8, 6 scenarios × 4 samples): **escape = 0% for
no-steer / steer / random alike.** Even with sampling, the model never abandons the looped
command. The −0.23-nat shift is real but the absolute repeat probability dominates, so it
doesn't change *which* command gets emitted.

## Bottom line (1.5B)
- **POSITIVE (representation + causal log-odds):** the agent encodes "unproductive repetition"
  (length-controlled); a **layer-8 "are my tests resolving?" direction causally and
  specifically** modulates the repeat log-odds (dose-dependent, control-beating,
  coherence-preserving).
- **NEGATIVE (behavior):** that single-direction lever is **too weak to break the loop
  on-policy** without disrupting healthy runs — the loop is a very strong attractor at 1.5B.
- **Implication:** naive diff-of-means steering is **not** a free low-disruption loop-breaker.
  Next: a larger model (7B), multi-layer/multi-direction steering, or steering *combined with*
  LocalGuard's monitor as the trigger. Honest negative > hype.

---

# Strengthening round (T1–T4)

## T4 — disentangle stuck-awareness vs test-reading ✅ (honest downgrade)
From cached acts2 (`run_disentangle.py`): on the **STUCK axis**, `vfail` (same failing obs,
different command) sits with `loopfail` (−0.74) while `loopprog` (improving obs) is far (−4.95)
→ **STUCK reads the test OUTPUT trend**, not abstract stuck-awareness. **REP** is a *distinct*
axis (cos(STUCK,REP)=0.37) reading command-repetition. So the "stuck representation" is the
**conjunction of two input-derived signals** (output-trend + command-repetition), not a unified
meta-cognitive concept. The lever we steered = "convince the model its tests are passing."

## T1 — real loop trajectories ✅ (refutes the "synthetic artifact" hypothesis)
`run_real_loops.py` on 24 of LocalGuard's audited tight loops (e.g. `find_file "SecretStr"` ×3 →
"No matches found"). Result is the *opposite* of expected:
- **Real loops are a STRONGER attractor: repeat-pref +2.33** (median +2.61, 100% positive) vs
  synthetic +0.35. Verbatim real repetition pulls the copy mechanism harder.
- **Synthetic STUCK direction does NOT transfer** (Δpref +0.19 @α=12 — slightly *increases*
  repeating; expected, since real loops are failed searches/typos, not test failures).
- **On-policy escape still 0%.**
So the 0% escape is **not** a synthetic artifact — synthetic was the *easy* case. This makes the
**circuit** (the copy mechanism) the thing to attack.

## T2 — the loop circuit ✅ (found induction heads; mechanism is distributed)
`run_circuit.py` on real loops (baseline repeat-pref +2.98), causal head ablation:
- **Layer-0 attention dominates** (ablating it: Δpref **−4.9**, flips the preference) but is
  **diffuse** — no single L0 head carries it; it's broad early processing the verbatim copy rides on.
- **Genuine induction heads found:** **L19H3** puts **0.97** of its decision-token attention on the
  *prior command tokens* (L19H5: 0.83) — and ablating them reduces repeating. So we found real
  copy-mechanism components, not just a direction.
- But per-head effects are small (~−0.13 vs +2.98), all in mid-late layers (L15/L19). **No single
  "loop head"; the re-run behavior is distributed.** → head ablation likely won't break loops;
  a targeted logit penalty should.

## T3 — smarter interventions ✅ (a targeted intervention DOES break loops)
`run_interventions.py` on the hard real loops (baseline on-policy escape 0%):
- **(a) ablate top induction heads:** escape **0%** AND **60% disruption** on healthy runs —
  ineffective *and* harmful (the mechanism is distributed, so you can't cleanly remove it).
- **(c) −STUCK steering @α=16:** escape **0%** (representation nudge still too weak).
- **(b) gated logit penalty on the repeated-command tokens (p=8): escape 100%** — and the
  escapes are **coherent** (`find. -name "*.metric"` → `git status`; `find_file "SecretStr"`
  → `open`), not gibberish. By construction it's **gated** (fires only when a loop is detected),
  so it's ~zero-disruption off-loop.

**Updated bottom line:** the loop is a strong, *distributed* copy behavior, so
representation-steering and head-ablation don't break it (and ablation even hurts healthy runs).
But a **monitor-gated, targeted logit penalty breaks 100% of real loops into coherent actions** —
the deployable, low-disruption answer, and exactly the LocalGuard pairing: *the monitor detects
the loop, a targeted edit breaks it.* The earlier "steering fails" negative stands for
representation-level nudges; the right tool is a targeted output edit triggered by the monitor.

## Efficacy–disruption frontier ✅ (answered the "is it just repetition penalty?" question — and corrected my hypothesis)
`run_frontier.py`. Efficacy = escape on real loops; disruption = action-change on *productive*
repetition (loopprog: same command re-run while fail-count drops). The monitor gate fires on
**10/10 real loops** and **0/12 productive-repetition** contexts.

| intervention | escape (efficacy) | disruption |
|---|---|---|
| none | 0.00 | 0.00 |
| **targeted penalty, gated (ours)** | **1.00** | **0.00** |
| targeted penalty, always-on | 1.00 | 0.00 |
| always-on repetition_penalty (r=2) | **0.00** | 0.92 |
| always-on no_repeat_ngram(3) | **0.00** | 0.50 |
| −STUCK steering (α=16) | 0.00 | 0.92 |

**Two honest conclusions, one of which overturns my own hypothesis:**
1. **We are NOT re-deriving repetition penalty.** The standard decoding tricks *fail* on these
   strong verbatim loops (escape 0 — too diffuse to flip the +2.98 preference; no-repeat lets a
   near-identical command through) AND disrupt healthy runs (0.5–0.92). So the result is non-trivial.
2. **The contribution is TARGETING, not gating.** Penalizing the *specific looped command's
   tokens* (identified from the trajectory/monitor) breaks loops at zero disruption — and the
   *gated* and *always-on* targeted variants are identical (both 1.00/0.00). Gating added no
   measurable value here, because consecutive exact-command repetition is *itself* the pathology
   (healthy runs don't do it), so the targeted penalty is inherently safe.
**So the monitor's real role is IDENTIFYING what's being looped (to target it), not timing.**
Caveat: generics tested at one strength each (r=2, n=3); stronger settings might break loops but
at even higher disruption. Gating would matter only where a healthy run wants to re-emit the exact
command (rare) — belt-and-suspenders.

## Tier-2 outcome experiment ✅ (the decisive question — honest negative, capability-gated)
`run_outcome.py` + `agent_env.py`: a REAL local agent loop (sandboxed shell, real execution,
portable sed, real `python test.py` success check) on 6 single-edit bug-fix tasks, all solvable
with one correct edit. Control vs treatment (break the loop on detection). The loop is the natural
weak-model failure: an A-B-A-B cycle (wrong no-op `sed` → `test` → repeat).

**Result — breaking the loop does NOT improve outcomes at 1.5B:**
| condition | task success | notes |
|---|---|---|
| control (no intervention) | **0 / 6** | all 6 loop (loop_rate 1.0) |
| treatment (targeted, test-exempt) | **0 / 6** | breaks loop live (escapes), then loops on `test` |
| treatment (aggressive, penalize all) | **0 / 6** | escapes every loop, still 0 success |

**Why (from the traces):** the model has a *broken mental model* — it repeatedly edits the
**test call** (`sed 's/add(-1,3)/add(2,3)/'`, a no-op since `sol.py` has no such string) and
**never `cat`s the source** to see `return a - b`. So un-sticking just redirects it to another
unproductive action; it lacks the capability to diagnose/fix the bug either way.

**Conclusion:** the intervention **works live** (reliably breaks loops in a real agent loop, not
just teacher-forced contexts — confirms T3), but **un-sticking is necessary, not sufficient.**
The outcome-flip is **capability-gated**: a model must be able to *recover* once un-stuck, and the
1.5B cannot. This empirically grounds the LocalGuard paper's own open gap — a working loop-breaker
alone does not improve outcomes with a weak model. **Decisive next step: re-run with a capable
model (7B-Instruct+) on real tasks — does breaking the loop then yield recovery?**
