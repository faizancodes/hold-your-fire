# Hunting the recovery direction — a rigorous mechanistic study

**Question.** Is there a *causal* internal direction that moves a model from **persist** (repeat
the failing command, stay stuck) to **recover** (take the productive action) — a steerable
"do-the-right-thing" lever, not just a "stop-repeating" detector?

The earlier crude attempt (a `loop-vs-progress` diff-of-means from n=4 synthetic prompts, one
layer, no controls) failed at 30B and was rightly called out as under-rigorous. This is the
proper version.

## Method (the rigor)

- **The right contrast.** Not loop-vs-progress. We contrast **persist vs recover *inside the
  same stuck situation*** — the only difference is the decision, so a predictive axis cannot be
  a surface cue.
- **Two substrates.**
  1. *Synthetic battery* (`recovery_contexts.py`): 60 matched stuck contexts (a command
     repeated 3x, still failing) across 5 families (failed search / grep / test / cat / no-op
     edit) and 12 surface variants, **token-length padded** to kill the length confound.
  2. *Real on-policy trajectories* (`recovery_onpolicy.py`): the model runs the actual bug-fix
     agent loop (`agent_env.py`); at every step we capture the decision-token residuals at all
     layers and label the decision by the model's **own** behavior (persist = repeat a recent
     command while failing; recover = a new productive command out of trouble).
- **Discovery across all layers, multiple methods**: diff-of-means, a logistic probe scored with
  **scenario/task-grouped leave-one-group-out CV AUC**, and recovery-margin regression.
- **Confound controls**: length-only AUC (must be ~0.5); the decodability-vs-causality
  dissociation (a layer can be highly decodable yet causally inert); random + orthogonal
  steering controls of equal norm.
- **Causal validation** (the actual test): on **held-out** (family- or task-grouped) persist
  contexts, steer the candidate and measure the shift in recovery-margin (logP(productive) −
  logP(repeat)) with a **dose-response**, plus on-policy recover% and coherence. Decodability is
  cheap; only the causal numbers carry the claim.

All on quantized models via MLX (`mlx_wrapper.py`) so the same code runs 1.5B → 30B on a 32 GB Mac.

## Results

**1. Synthetic battery, 1.5B — a clean causal recovery lever.**
Length-only AUC 0.498 (confound controlled). The decodability≠causality dissociation showed
up immediately: L0 had the *highest* probe AUC (0.778) yet steering it did nothing causally
(recover 0%). The causal winner was **L9**: steering it drove held-out-family on-policy recovery
**0% → 100%** with a monotonic margin dose-response, while random/orthogonal controls moved the
metric the *wrong* way and coherence was preserved. This is the clearest result — a real,
causal, behavior-flipping recovery direction — in the regime where the model *deterministically*
persists.

**2. Synthetic battery, 30B — honest null, and it's diagnostic.**
The capable 30B does **not** fall into the synthetic loop: greedy labels were persist=2,
recover=0, other=58 (with command-only prompting it was unchanged). With no persist→recover
behavioral axis, the discovered direction is noise and the causal test correctly returns
negative (net dir −5 to −8). **Capable models see through constructed loops** — you need their
own stuck states.

**3. Real on-policy trajectories, 1.5B — the recovery axis on genuine behavior.**
Harvesting the 1.5B's own agent-loop decisions (after fixing two harness bugs — see below) gave
a **balanced** contrast: 326 decisions, persist=141 / recover=74. The axis is strongly decodable
(task-grouped CV AUC **0.94–0.95** at L20–L26), and **causally specific**: steering **L20** on
held-out-task (`dict_missing`) persist decisions raises the recovery-margin **−6.56 → −5.28
(net +1.28)** while random/orthogonal controls go negative. A real-data, held-out-task-validated
recovery direction. (Caveat: on-policy recover% is flat at 65% — at greedy temperature these
"persist" contexts already mostly emit productive commands, so there is no *deterministic* stuck
state to flip here; the behavioral flip lives in result #1.)

**Two harness bugs found and fixed along the way** (both were silently wrecking the capable-model
runs): (a) `generate` stopped only on `eos_token_id` (`<|im_end|>`) but the model emits
`<|endoftext|>` (a *different* id in `eos_token_ids`), leaking special tokens + prompt echo into
fake "commands"; (b) feeding a raw-text transcript instead of the **chat template** made the
instruct model behave erratically and never edit. With both fixed, models emit clean
`cat`/`sed`/`python` commands and the harvest balance went from 144/7 to 141/74.

**4. Interpretation — the direction is abstract, not lexical.**
A logit-lens projection of the L20 recovery direction through the unembedding returns **noise**
(no coherent "investigate" vs "repeat" tokens). The recovery axis is a *distributed disposition*
that a naive mid-layer logit lens can't decode; its real signature is behavioral (it shifts the
productive-vs-repeat preference), not a set of promoted tokens.

**5. Real on-policy trajectories, 30B — the capable model simply does not loop.**
With the fixed harness (clean commands, chat template), the 3-bit 30B was run on all 14 tasks.
Harvest: **persist=1, recover=7 → abort** — and the smoking gun is **reps=0 on 13 of 14 tasks**
(it even solved 1). The capable model *varies* its command every step instead of repeating; it
does not fall into the repetitive loop the recovery lever is defined against. This is not a
tooling artifact (the harness was fixed and the model emits clean commands and solves) — it is a
genuine property. Confirmed three independent ways now: synthetic-30B null, command-only-30B
null, on-policy-30B abort (reps≈0).

**6. Capable model, a second contrast — investigate-when-stuck vs act-when-stuck (also data-starved).**
Since a capable model flails (acts) rather than loops, the right recovery contrast for it is
"when stuck, INVESTIGATE the evidence vs ACT blindly" (`recovery_capable.py`). It too has no
data: the weak 1.5B almost never investigates (INVESTIGATE=0, all ACT/other), and the 7B goes
**straight to editing** — on `add_op` it issued 8 ACT commands, 0 investigate, and on `double`
it solved in a single ACT. Capable models don't *hesitate to investigate*; they confidently act,
and then succeed or fail on **competence** (the right vs wrong edit), which is task-difficulty-
and content-bound, not a steerable disposition. So this contrast starves from the *opposite*
side of the 1.5B's.

**The pattern across all capable-model probes (synthetic persist-vs-recover, on-policy
persist-vs-recover, investigate-vs-act):** a clean recovery *lever* requires a steerable
*disposition* (loop ↔ break-loop; act-blind ↔ investigate). Capable models don't exhibit those
dispositions on tractable tasks — they neither loop nor hesitate. Their failures are
**competence** failures (a confidently-wrong edit), and "make the correct edit" is content, not a
direction you can steer in. Three independent contrasts, same wall.

**7. Real hard tasks — LiveCodeBench (the decisive test of "give it harder tasks").**
The standing rebuttal was "capable models don't get stuck because the tasks are too easy." So we
built a local harness over **LiveCodeBench** (`lcb_data.py` + `lcb_env.py` + `recovery_lcb.py`):
175 contamination-free competitive-programming problems from 2025-01..04 (after both models'
cutoffs), run agentically (write -> run hidden tests -> see failure -> revise) with per-attempt
activation capture. The capable model **does** now genuinely struggle: the 7B solves **81%** of
easy but **0%** of medium, with 3/8 easy problems solving on one seed and stuck on another.

But the recovery contrast still has no data, for a newly precise reason:
- episode-level (struggle-then-solve vs stay-stuck): only **2** recover vs 57 stuck.
- progress-level (a revision that passes MORE hidden tests vs one that does not): only **5
  productive** vs **53 flailing**.

So on real hard tasks the capable model **does not recover gradually** — it one-shots or flails.
When it fails, its revisions almost never get *closer* to passing (5/58 ≈ 9% productive). There is
no gradual-recovery *behavior*, hence no gradual-recovery *direction* to steer. Harder tasks made
the model **stuck** but not **recoverable**: its hard-task failure is **competence** (it writes a
wrong solution and can't incrementally fix it), and competence is content, not a steerable axis.

**8. The deployable test — does a monitor-gated "reconsider" MESSAGE flip outcomes? It HURTS,
and the way it hurts is the punchline.** Steering can't inject competence, but a *message* can
inject a *strategy* — so we tested it (`lcb_intervene.py`): paired control (normal test feedback)
vs treatment (feedback + "stop tweaking, reconsider your whole approach") on LiveCodeBench flailed
episodes. Result on the 21 flailed episodes:

| difficulty | control solves | treatment solves | lift |
|---|---|---|---|
| easy (recoverable) | **4/5** | 2/5 | **-2** |
| medium (beyond competence) | 1/16 | 0/16 | -1 |
| ALL | 5/21 | 2/21 | **-3** |

Per-episode: on `abc387_b` and `abc389_b` the model **recovered on its own with normal feedback
but FAILED under the reconsider nudge** — the nudge told it to throw away a nearly-correct
solution and "start over", destroying its productive iteration. This is the **Intervention
Paradox** — LocalGuard's central thesis — demonstrated for the first time on a *capable* model and
*real hard tasks*: the eager intervention fires on runs that would have recovered (and disrupts
them, 4/5 -> 2/5), and can't help the runs that won't (competence, 1/16 -> 0/16). Net harmful.

So the paper's standing gap ("no demonstrated outcome-flip") is closed — in the honest direction:
the demonstrated outcome *effect* of an eager nudge is **negative**, which is the strongest
possible evidence for calibrated abstention over intervention. The deployable move on a flailing
capable agent is **not to nudge** but to stay silent (it often recovers on its own) or halt/escalate
(when it can't) — and the monitor's job is telling those two apart.

**9. The positive: you CAN make a capable model write correct code more often — via its own
verification, not steering.** Re-examining the "competence wall" claim revealed a reasoning error:
a capable model failing greedily on a hard task is usually an **elicitation gap**, not a competence
wall. Evidence (`lcb_passk.py`): on LiveCodeBench easy-frontier problems the 7B fails greedily but
solves in independent samples (`abc387_b` 2/12, `abc389_b` 3/12) -- the correct solution is
**reachable**, just low-probability. (Medium problems are 0/12 -> genuine competence walls.)

Two ways to exploit the reachable correct mode, one fails and one works:
- **Additive steering of a "correctness" direction FAILS** (`lcb_correctness.py`): the direction is
  decodable on teacher-forced codes (grouped-CV AUC 0.75) but a **mirage on fresh samples** (AUC
  0.45), and steering it only disrupts generation (never beats baseline). No usable *linear* lever.
- **Verification-guided SELECTION WORKS** (`lcb_verify_auc.py`, `lcb_verify_select.py`): the model's
  own explicit judgement ("will this pass all tests? YES/NO") ranks fresh correct samples above
  incorrect ones at **AUC 0.83**. Emitting the top-scored of K samples lifts **pass@1 0.57 -> 0.80
  (+0.23, ~half the gap to oracle pass@k)**, and +0.21 on held-out problems. The generation-
  verification gap is real and exploitable: the model recognizes the correct solution better than it
  generates it first-try, so let it pick among its own samples.

Boundary (honest): this only helps where the solution is **reachable** (pass@k>0). It cannot rescue
a genuine competence wall (you can't select a correct sample that was never generated).

**Solidified at larger N (`lcb_solidify.py`, 8 usable problems, the honest correction):** the
N=5 +0.23 was small-sample inflation. With proper N the verification-selection lift is only
**+0.08** (pass@1 0.55 -> 0.62; verify fresh-AUC 0.78 -- real signal but a weak top-1 selector).
And a cheaper, *non-internals* baseline dominates it: **self-consistency** (generate K, run them on
the public example inputs, take the majority output) reaches **0.88 (+0.33)**, beating verification
by 0.25. So the elicitation gap is real and *strongly* exploitable (0.55 -> 0.88), but the powerful
lever is **execution-based consensus**, not the model's internals; the internal correctness signal
gives only a minor assist. The honest takeaway: to make a capable model write correct code more
often on reachable problems, **sample many and let the public tests pick the consensus** -- a
selection/execution method, not steering and not (primarily) self-verification.

## Honest bottom line

- A **causal recovery direction exists and is reproducible**: it raises the model's preference
  for the productive action over repeating (margin shifts +0.66 to +2.56 across setups, controls
  negative, grouped-CV AUC ~0.94, generalizing across held-out families/tasks). The right
  contrast (persist-vs-recover-in-place) finds what loop-vs-progress could not.
- It is an **abstract disposition**, not a lexical "say cat" vector.
- The **clean behavioral flip** (0%→100%) is in the *deterministically-stuck* regime (weak model,
  constructed loop). On real sampled trajectories the persist behavior is noisier, so the lever
  shifts preference without a guaranteed flip.
- The **capable-model version is out of reach — and the reason is itself the finding.** The 30B
  does not loop (reps≈0 across 14 tasks, synthetic *and* real); it routes around repetition by
  varying its actions. So "recovery from looping" is intrinsically a **weak-model lever**: the
  failure it fixes (verbatim repetition) is rare in capable models, whose failures are different
  (a varied-but-wrong mental model). A recovery lever for capable models would therefore need a
  *different* contrast (right-fix vs wrong-fix), and a capable model genuinely stuck on hard tasks
  to supply it — the project's standing open gap, now sharpened from the mechanistic side: the
  thing to steer at 30B isn't "stop repeating," it's "reconsider the hypothesis."

## Files
`recovery_contexts.py` (battery), `recovery_onpolicy.py` (real-trajectory harvester + validator),
`run_recovery_direction.py` (synthetic discovery + causal validation), `interpret_recovery.py`
(logit lens), `mlx_wrapper.py` (quantized steering). Artifacts in `results/recovery_*.json` and
`results/recovery_*dir*_L*.npy`.
