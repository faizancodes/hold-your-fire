# LocalGuard-SWE — results narrative

All numbers below are from the **full** offline corpus (80,036 SWE-agent
trajectories → 127,092 prefix examples) with **instance-grouped** train/val/test
splits (test: 25,166 prefixes over 719 held-out instances). Failure is the
positive class. Thresholds are selected on validation and applied once to test.
Figures live in `results/figures/`; tables in `paper/results_offline_full.md`.

## Methodological contribution (read this first)

Our headline contribution is as much *how* we evaluate as *what* we find. The protocol
— **select on validation, then report a *paired instance-bootstrap* of the metric delta
on a held-out, instance-grouped test set**, with a random-abstention control — repeatedly
caught "wins" that were validation overfitting. Five separate times (advanced features,
val-selected features, a GRU, position-weighted labels, a survival GRU) the
validation-best model was *no better or significantly worse* on held-out test. Only two
changes survived (an ensemble, +0.003; the abstention reframing). This is why we trust the
**negative** results — the AUC ceiling and the irreducible early floor — as much as the
positive ones. Most agent-monitoring papers report one number on one split; the difference
between that and a real effect is exactly this protocol.

## Generalization (full across scale/shell; cross-scaffold needs a free alignment step)

The signal transfers fully when the action space is similar, and *partially — recoverable for
free —* across a structurally different scaffold:
- **Cross-model** (train swe-agent-llama-70b excluding the test tasks → test 8b+405b,
  unseen models *and* tasks): AUC 0.735 → **0.697** (gap +0.037); ECE 0.00→0.06,
  **recalibration restores 0.02**.
- **Cross-family, shell-style scaffold** (offline monitor on live qwen2.5-coder/mini-SWE
  traces): AUC **0.67–0.71** (≈ its 0.72 on Nebius), risk *levels* inflate (over-firing);
  **abstention cuts successful-run alarm crossings 6%→0%**.
- **Second independent source, structurally different scaffold — collapse, then free recovery.**
  Zero-shot on **500 OpenHands/CodeAct trajectories** (`CoderForge…`, SWE-bench Verified, same
  feature pipeline): naive AUC **collapses to 0.525**. But the failure *structure* is shared
  (top features point the same way as Nebius); only the **scale** differs (reads 11.8 vs 2.4),
  so the frozen tree's splits misfire. **Free, label-free quantile-alignment of features to the
  source marginal recovers AUC to 0.591** — matching the **in-domain ceiling (0.603, 5-fold CV)**
  with zero target labels; an **ensemble of transferable feature-family submodels** (mapped-volume
  ⊕ raw scale-invariant repetition ⊕ linear) reaches **0.604** (exceeding the in-domain single
  model), and **abstention lifts deployable AUC to 0.66** (44% coverage). **The biggest lever is
  *when* you predict**: the per-prefix AUC averages over noisy *early* prefixes, but at the agent's
  **final prefix** — a deployable **post-hoc patch-acceptance gate** — AUC reaches **0.722 zero-shot
  / 0.728 few-shot, i.e. *Nebius parity*, cross-scaffold and free.** So the cross-scaffold gap is
  specific to *early-warning* (a capable agent's early behavior is genuinely uninformative — its own
  tests pass while hidden SWE-bench tests fail), not to the monitor's signal overall.

So the *ranking* transfers at full strength across **model scale** and **shell-command agents**,
and across a structurally different scaffold once feature distributions are aligned (free): nearly
fully for a **post-hoc** gate (0.72), partially for **online early-warning** (0.60→0.66). Transfer
is governed by *action-space similarity + distribution alignment + when you predict*. Probabilities
are always target-specific (fixed by recalibration/abstention).

## Claim 1 — Failure is predictable from partial trajectories ✓ (with nuance)

Lightweight, CPU-only monitors predict eventual failure from prefix-visible
features well above chance and above length-only baselines:

| model | ROC AUC [95% CI] | AUPRC | ECE |
|-------|------------------|-------|-----|
| majority | 0.500 | 0.569 | 0.007 |
| step-count only | 0.623 [0.607, 0.636] | 0.707 | 0.014 |
| heuristic rules | 0.574 [0.538, 0.607] | 0.632 | 0.012 |
| logistic regression | 0.693 [0.673, 0.711] | 0.762 | 0.018 |
| random forest | 0.719 [0.699, 0.740] | 0.793 | 0.015 |
| **hist gradient boosting** | **0.722 [0.700, 0.743]** | 0.792 | **0.011** |

Risk separates the two populations and **grows over trajectory time** for failed
runs (≈0.52 → 0.78) while staying flat (~0.5) for successful runs — Figure 1. The
shuffled-label control gives AUC **0.522**, confirming the signal is not leakage.

Nuance: these are honest, moderate AUCs (~0.72), not the inflated numbers that
leakage produces. Labels are *terminal* outcomes attached to partial prefixes,
so some prefixes of eventually-failed runs are genuinely healthy — an irreducible
ceiling we state openly.

## Claim 2 — Cheap structured features are competitive ✓

- **Structured > step-count-only** (0.722 vs 0.623): the monitor is not merely
  learning "long runs fail."
- **Structured ≥ text**: adding TF-IDF text features *lowers* AUC (0.722 →
  0.672). Cheap tabular features beat bag-of-words here.
- Best single feature family is **`action_counts`** (0.708); testing-behavior
  alone is weakest (0.556) because most early prefixes have not run tests yet —
  Table 2 / Figure 2.
- Feature importance (Figure 5): high *search-to-edit ratio*, installs, and
  repeated test commands raise risk; actually *running tests* and *submitting*
  lower it — intuitive "drift vs. progress" signals. (The negative
  `max_command_repeat_count` coefficient is a known logistic collinearity
  artifact; tree models rank repetition as risk-increasing.)
- **The cheap classifier beats *two* frontier LLM judges *and* a local one** (same prompt +
  schema; only the model/provider changes; same 200 matched prefixes). AUC on that subset:
  classifier **0.77**; **GPT-5.5** (`gpt-5.5-2026-04-23`, OpenAI) **0.63**; **Claude Opus 4.8**
  (`anthropic/claude-opus-4.8` via OpenRouter) **0.62**; local `qwen2.5-coder:7b` **0.56**.
  Paired instance-bootstraps put the classifier **significantly ahead of both frontier
  models** — vs GPT-5.5 by +0.14 AUC (95% CI [0.02, 0.25]; 99% of resamples favor the
  classifier), vs Opus 4.8 by +0.15 (95% CI [0.04, 0.27]; 99.4%). **Two independent frontier
  models from two providers land at the same ~0.62** — so "cheap beats frontier" is not a quirk
  of one model. The frontier judges are more discerning than the 7B (intervene on 46–62% of
  prefixes vs 97%) but still far too trigger-happy for a disruption-aware monitor, run at
  2.6–5.6 s/prefix (vs ~9 ms), and cost real money ($0.44–$2.10 per 200, paid API). The result
  is **robust to reasoning effort/thinking**, which we swept on *both* models: GPT-5.5 at
  `reasoning_effort=high` gives 0.619 ≈ its low-effort 0.626 (paired Δ −0.006, n.s.); Claude
  Opus 4.8 with extended thinking gains a small, *borderline* bump (no-thinking 0.618 → low 0.639
  → high 0.633; low−none Δ +0.021, 95% CI [−0.001, +0.046], n.s.). Across **every** judge config
  tested — qwen-7B, GPT-5.5 (low/high), Opus 4.8 (none/low/high) — the cheap classifier wins
  significantly. More capability and more thinking buy little-to-no accuracy here: a frontier LLM
  is simply a *worse* early-failure predictor than 1 MB of gradient boosting.

**Deployment cost — latency *and* memory (measured, `cost.json`, M4 CPU, psutil RSS).**
The deployed model is **1.14 MB on disk** (vs the RandomForest's 206 MB) and serves from
a **~210 MB resident process** (Python + numpy/pandas/sklearn + model; the model itself
adds ~77 MB in RAM, including first-use lazy imports). Latency, measured on the *real*
`Monitor.assess()` path (feature extraction + predict + calibrate):
- **~8.6 ms per live prefix** (median; p90 ~15–23 ms) — dominated by sklearn's fixed
  per-`predict` call overhead, **not** compute (feature extraction is **0.14 ms**).
- **Batched: ~390k prefixes/s**; the full **25,166-prefix test set scores in <0.1 s**.

The LLM judge needs **4.7 GB** of weights and **11.6 s/prefix**. So the structured
monitor is **~20× lighter in memory** and **~1,300× faster per prefix** on the real path
(≥10⁴× batched). "Local, lightweight, CPU-only" is a measurement, not an adjective.

## Threshold tradeoff — why "accuracy" isn't the objective

| policy | threshold | success false-alarm | failed coverage | median lead |
|--------|-----------|---------------------|-----------------|-------------|
| T1 max-F1 | 0.376 | **0.986** | 0.999 | 18 |
| T2 FAR≤20% | 0.703 | 0.170 | 0.558 | 14 |
| T3 FAR≤10% | 0.778 | 0.090 | 0.437 | 13 |
| T4 FAR≤5% | 0.851 | 0.022 | 0.274 | 11 |

The F1-optimal threshold alerts **≈99% of all runs** — catastrophic for a
disruption-aware monitor. Useful operating points trade coverage for a small
success false-alarm budget, with **median 11–14 step** warning lead time.

## Claim 3 & 4 — Intervention is not automatically good (online)

We integrate the monitor into a local Ollama + mini-SWE-agent loop (Level-1 toy
bug-fix tasks) and run baseline / shadow / loop-guard. Shadow mode is **verified
behaviorally identical** to baseline at temperature 0 on all three tasks.

**loop-guard** fired 2 interventions/run (first alarm step 6–7) and produced
**0 disruptions** — it broke none of the runs, including the one that was
succeeding (Claim 3: the low-disruption design is safe). Outcomes were unchanged
vs. baseline (0 recoveries, 0 disruptions) while **avg steps fell 40 → 31**,
hinting at reduced wasted work (Claim 4) — see Figure 4.

Two honesty notes that are themselves findings:
- **evidence-gate never triggered** on these tasks (the agents read before
  editing, so the premature-edit condition never held). We exclude it rather than
  present a baseline re-run as an "intervention."
- We observed **temperature-0 nondeterminism** in local Ollama: a 0-intervention
  run flipped outcome vs. baseline. Small-N online outcome comparisons are
  therefore noisy — exactly why these numbers are **illustrative only** and no
  claim rests on them. This operationally echoes the Intervention Paradox: you
  must measure recovery vs. disruption on real runs, carefully, not trust AUC.

## Observability audit (which failures are catchable early) — human-validated

We did not trust the regex labeler on its own. One author re-labeled **50 prefixes
blind** (40 confidently-flagged failures + 10 missed failures), reading the actual
agent actions/observations — not the features — then joined to the heuristic
(`results/audits/human_validation.json`). The finding is a *correction* and a
*confirmation*:

- **Correction:** the heuristic calls **88%** of flagged failures "looping," but
  blind human reading finds only **40%** are *tight* loops (identical command/edit
  repeated). The regex's looping **precision is 0.37** (recall 0.94): most of what it
  labels "looping" is really **patch-churn** — varied edits whose tests never improve
  (e.g. permuting a date string while ignoring the real `NameError`; hard-coding
  `if name=="bla": return []` to pass a case). So "loops" overstates it.
- **Confirmation:** at the *family* level the picture holds — **70%** of
  confidently-flagged failures are **repetitive / non-progressing behavior**
  (40% literal loops + 30% churn), and coarse (family) agreement with the heuristic is
  **0.68**. The disagreement is almost entirely loop-vs-churn *within* this family, not
  wild misclassification.
- **Observability contrast survives:** of the *missed* failures, **70%** are
  genuinely **not-observable or insufficient-context** by human reading (the agent
  looks healthy, or the failure is task-difficulty bound) — exactly the prefixes the
  early floor and abstention concede. Only ~20% were observable-but-missed near-misses.

So the honest claim is: **confidently-caught agent failures are dominated by visible
repetition and churn** (motivating loop/no-progress interventions), while **missed
failures are genuinely subtle** — and "looping" should be read as the broader
"repetition/no-progress" family, not literal loops alone.

## Selective prediction — the deployable headline (abstention)

The terminal-on-prefix label is noise early (AUC 0.63) and informative late (0.81),
but "late" needs the unknown total length — not deployable. The deployable fix is a
monitor that **abstains** when a prefix-visible gate isn't met (`prefix_step ≥ S AND
|risk−0.5| ≥ C`) and reports a **risk–coverage curve** (Figure 6):

- **AUC 0.80 at 50% coverage, 0.83 at 30%** — vs 0.72 unconditional. A *random*
  abstention baseline stays flat at 0.72, so the lift is real, not mechanical.
- Abstaining on early prefixes cuts **success false-alarm rate 0.090 → 0.059 (−35%)**
  and improves Brier 0.208 → 0.167 — i.e. it reduces **disruption**, the metric that
  actually matters for a disruption-aware monitor.
- Retraining a separate model on the regime does not help (full model suffices); the
  gate is wired into `monitor.py` as an explicit abstain verdict.

Honest framing: *"AUC 0.80 @ 50% coverage / 0.83 @ 30% (vs 0.72 unconditional), with
35% fewer false alarms on successful runs"* — a higher, deployable headline than 0.722.

**Online validation (live agent, deterministic replay, N=25).** We captured 25 shadow
runs (qwen2.5-coder:7b; 14 succeeded, 11 failed) and replayed the ungated vs gated
monitor over the *same* trajectories (gate fixed from the offline study). The offline
monitor **over-fires badly under the cross-agent shift — it interrupts 79% of
*successful* runs** — and abstention cuts that ~10× to **7%**, at a coverage cost
(failures caught 73%→46%). Live confirmation that abstention reduces disruption.
**Deployment lesson: tune the gate to the target's trajectory-length distribution**
(a step floor right for long traces over-abstains on short runs). This is the
disruption-*risk* measurement on our live loop; a forced-intervention run (loop_guard)
changed no outcomes and trimmed wasted steps (40→31), and the full outcome-flip study
is tens of hours locally.

**Chasing the ceiling fails (a clean negative).** We tried to lift the *early* prefixes
with a genuinely cleaner label — weak-supervision relabeling (hurt) and a survival/
hazard GRU (early AUC *identical* at 0.633). Neither helped, because early failure is
genuinely undetermined or driven by task difficulty that doesn't transfer across the
grouped split. The early floor is irreducible — which is exactly why abstention (skip
the early prefixes) is the right answer, not relabeling.

## Headline takeaway

A local, lightweight, **CPU-only** monitor identifies actionable coding-agent
drift (AUC 0.72, well-calibrated, ~13-step lead) using cheap structured signals
that beat text and a local LLM judge — but accurate prediction is not the same as
safe intervention, which is why we measure recovery vs. disruption rather than
optimizing AUC.
