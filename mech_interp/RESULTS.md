# Steering coding agents out of failure loops — results

A mechanistic-interpretability follow-up to LocalGuard. **Model: Qwen2.5-Coder-1.5B-Instruct**
(white-box, local MPS, raw forward hooks; no nnsight). All probes are scenario-grouped with
bootstrap CIs — LocalGuard's "validation lies" discipline, imported wholesale, and it earned its
keep here (it caught a confound that would otherwise have been the headline).

## The question
LocalGuard's most catchable failure is repetition (tight loops). Does the agent *internally
represent* "I'm repeating an unproductive command"? Is that representation **causally**
responsible for looping? And can steering it out break loops **less disruptively** than an
external reset (the lever the Intervention Paradox penalizes)?

## Method in one paragraph
Synthetic coding-agent transcripts in matched conditions: **loop** (same command ×K, same
failing output), **varied-fail** (K *different* failing commands), **progress** (K distinct
productive commands), and **loop-progress** (same command ×K but *improving* output). We read the
residual stream at the decision point, train diff-of-means probes (scenario-grouped CV), and —
crucially — judge the mechanism by **causal steering/ablation** (add/remove a direction during a
forward pass), not by probe accuracy.

## Findings

**1. There is a large behavioral loop tendency.** In a loop context the model strongly prefers to
repeat: log-prob(repeat) − log-prob(novel action) = **+0.35** averaged over held-out scenarios
(up to +0.7 individually), vs **−2.6** in a length-matched progress context.

**2. "Loop" is trivially decodable — a warning, not a result.** A naive loop-vs-progress probe
hits **AUC 1.000 at every layer**, but the conditions differ in length/surface tokens; **length
alone** separates them (AUC 0.90–1.00). Steering with that confounded direction does **not** break
loops (it slightly *increases* repeating). This is the mirage the grouped-bootstrap discipline
exists to catch.

**3. With length *exactly* controlled, the information is genuinely present.** Token-padding makes
all conditions identical (**321.5 tokens; length-only AUC 0.500**). Then both the
same-vs-different-command axis (**REP**) and the failing-vs-improving-outcome axis (**STUCK**,
with token-repetition held identical) are decodable at **AUC 1.000**. The model *has* the
information — but linear decodability is cheap and cannot tell a banal cue from a causal mechanism.

**4. The causal test is positive and specific.** Building directions from the length-controlled
activations and steering on natural loop contexts (baseline repeat-pref **+0.348**):

| | α=2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|
| **−STUCK @ L8** (steer) | −0.05 | −0.10 | −0.14 | −0.19 | **−0.23** |
| random (control) | +0.01 | +0.01 | +0.02 | +0.03 | +0.08 |
| orthogonal (control) | +0.01 | +0.01 | +0.02 | +0.03 | +0.08 |

A clean **monotonic dose-response**, **specific** to the direction (controls do nothing),
**coherence-preserving** (novel-action log-prob stays ~−1.8). Steering toward "my tests are
resolving" suppresses the urge to re-run; toward "still failing" *increases* it (sign-consistent).
**Ablation** (projecting the direction out) gives Δpref +0.019 — removing it slightly *increases*
repeating, so it's a **steering lever, not strictly necessary**. Mechanistically: a **layer-8
"are my tests resolving?" axis** causally modulates the decision to re-run.

**5. On-policy, the lever is too weak — and there's a surprising asymmetry (honest negative).**
Generating the next action (manual KV-cache decode; `model.generate()` crashes on MPS):
- **Loop-escape = 0%** greedy *and* under sampling (temp 0.8), for steer / random / no-steer
  alike. The −0.23-nat nudge doesn't change *which* command is emitted; the loop is a very strong
  attractor.
- **Coherence = 100%** (steering never produces gibberish).
- **Disruption on healthy runs: steer 100%, random 83%** at the α needed to dent loops. So
  **steering changes healthy decisions before it changes loop decisions** — the loop is *more*
  robust to perturbation than healthy reasoning is. Naive single-direction steering is therefore
  **the wrong tool for low-disruption loop-breaking**, at least at this scale.

## Bottom line
A genuine, rigorously-controlled **positive** at the representational/causal level — a specific
layer-8 direction encodes and causally modulates "unproductive repetition" — paired with an honest
**behavioral negative**: that lever is too weak to break the loop on-policy without disrupting
healthy runs at 1.5B. This *sharpens* LocalGuard's Intervention-Paradox thesis: a simple internal
nudge is not a free lunch; the loop attractor resists exactly the cheap intervention you'd hope to
use.

## Strengthening round (T1–T4) — and a real fix

Four follow-ups stress-tested and then *improved* the result:

**T4 — what is the "stuck" direction really?** On the STUCK axis, `vfail` (same failing output,
different command) sits with `loopfail` while `loopprog` (same command, *improving* output) is
far away. So **STUCK reads the test-output trend** — "are my tests still failing?" — not abstract
self-awareness. Command-repetition is a *separate* axis (REP; cos 0.37). The "stuck representation"
is the conjunction of two input-derived signals, not a unified meta-cognitive concept. (Honesty
downgrade on the interpretation.)

**T1 — real loops, and it refutes my own hypothesis.** On 24 of LocalGuard's audited real tight
loops (e.g. `find_file "SecretStr"` ×3 → "No matches found"), the model's repeat-preference is
**+2.33** (median +2.61, 100% positive) — *far stronger* than the synthetic +0.35. Real verbatim
repetition pulls the copy mechanism harder. The synthetic STUCK direction **does not transfer**
(it slightly *increases* repeating; expected, since real loops are failed searches/typos, not test
failures). On-policy escape: still 0%. **So the 0% escape was not a synthetic artifact — synthetic
was the easy case.**

**T2 — the circuit.** Causal head ablation on real loops: **layer-0 attention dominates** (ablating
it flips repeat-pref by −4.9) but is *diffuse*; and there are **genuine induction heads** (L19H3
puts 0.97 of its decision-token attention on the prior command tokens; L19H5: 0.83) whose ablation
reduces repeating. But per-head effects are small and spread across mid-late layers — **no single
"loop head"; the re-run behavior is distributed.**

**T3 — a smarter intervention that actually works.** On the hard real loops (0% baseline escape):
ablating the induction heads gets **0% escape and 60% disruption** on healthy runs (distributed →
useless *and* harmful); −STUCK steering at α=16 gets **0%**; but a **monitor-gated logit penalty on
the repeated-command tokens breaks 100% of loops** — into *coherent* alternatives (`git status`,
`open`), not gibberish — with ~zero disruption off-loop because it only fires when a loop is
detected.

## Revised bottom line (after the efficacy–disruption frontier)
Representation-level nudges (single-direction steering) and circuit ablation **don't** break these
loops — the copy behavior is a strong, distributed attractor, and the steerable "stuck" direction
is really test-output-reading. The frontier then sharpened *what* works and *why*:

- **The standard decoding tricks fail here.** Always-on `repetition_penalty` (r=2) and
  `no_repeat_ngram(3)` get **0% escape** on these strong verbatim loops *and* **0.5–0.9 disruption**
  of productive repetition. So this is **not** a re-derivation of repetition penalty.
- **The contribution is TARGETING, not gating.** Penalizing the *specific looped command's tokens*
  — identified from the trajectory/monitor — breaks **100%** of real loops into coherent actions at
  **zero** disruption. (Gated and always-on targeted are identical, because consecutive exact-command
  repetition *is* the pathology, so the targeted edit is inherently safe.)

So the deployable answer is **a targeted output edit on the monitor-identified looped command** —
*the monitor's job is to say what is being looped, and the edit suppresses exactly that.* That is
the LocalGuard pairing, and it turns the earlier honest negative into a non-trivial deployable
positive. (Honest caveats: 1.5B; N=10 real loops; generics tested at one strength; gating would
only matter for the rare healthy run that re-emits an exact command.)

## But does breaking the loop *help*? (Tier-2 outcome experiment — honest negative)
A real local agent loop (sandboxed shell, real `python test.py` success) on 6 single-edit,
solvable bug-fix tasks, control vs. loop-breaking treatment:

| condition | task success |
|---|---|
| control (let it loop) | **0 / 6** |
| treatment (targeted penalty) | **0 / 6** |
| treatment (penalize all repeats) | **0 / 6** |

The intervention **breaks the loop live** every time (confirming the mechanism in a real loop, not
just teacher-forced contexts) — but **task success doesn't move.** The traces show why: the 1.5B
model has a *broken mental model* (it edits the test call, never reads the source), so un-sticking
just sends it to a different dead end. **Un-sticking is necessary but not sufficient — the
outcome-flip is capability-gated.** A model must be able to *recover* once un-stuck, and the 1.5B
cannot. This is exactly the LocalGuard paper's own open gap, now empirically grounded: a working
loop-breaker alone does not improve outcomes with a weak model. The decisive next experiment is the
same one for the paper — a *capable* model on real tasks.

## Honest caveats & next steps
- 1.5B model, synthetic transcripts. The natural next steps: **(a) 7B** (a larger model may carry a
  stronger, more steerable direction), **(b) multi-layer / multi-direction** steering, **(c)**
  steering *triggered by* LocalGuard's monitor rather than always-on.
- Decodability AUC = 1.0 reflects that inputs differ in surface form; the **causal** numbers carry
  the mechanistic claim, by design.
- `model.generate()` is unusable on MPS (NDArray>2³²); all generation uses a manual KV-cache decode.

## Reproduce
`mech_interp/.venv/bin/python -m mech_interp.<run_localize2|run_steer_eval|run_onpolicy|run_sample_escape>`
then `... -m mech_interp.plots all`. Activations cached in `results/acts2.npz`; every number above
is in `results/*_results.json`. See `PROGRESS.md` for the full chronological log (including the
confound that was caught and discarded).
