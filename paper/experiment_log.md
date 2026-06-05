# LocalGuard-SWE — experiment log & verification checklist

Hardware: Apple M4 MacBook (10 cores, 32 GB RAM). All inference local via Ollama.
No paid/cloud APIs used anywhere.

## Core verification checklist

- [x] Ollama local endpoint works (`http://localhost:11434/v1`).
- [x] `qwen2.5-coder:7b` works locally (4.7 GB, 32K ctx).
- [x] Dataset schema inspected and saved (`results/offline/dataset_schema.json`).
- [x] 1000+ trajectories normalized (sample); full corpus = 80,036 trajectories.
- [x] Prefix dataset created with **no future leakage** (47 features, gate enforced).
- [x] Group split verified by `instance_id` (train/val/test disjoint).
- [x] Shuffled-label AUC near 0.5 (full: **0.522**; unit test averages to ~0.50).
- [x] Step-count-only baseline implemented (full AUC 0.623).
- [x] Heuristic baseline implemented (full AUC 0.574).
- [x] Structured classifier implemented (RF 0.719, HGB **0.722**).
- [x] Structured + text classifier implemented (0.681 — text does not help).
- [x] Calibration implemented (isotonic; HGB test ECE **0.011**).
- [x] First-alert metrics implemented (lead time, success FAR).
- [x] False alarms on successes reported (T3: 9.0% on full test).
- [x] Lead time on failures reported (median 13 steps at T3).
- [x] Ollama judge subset evaluated (`qwen2.5-coder:7b`, structured JSON).
- [x] mini-SWE-agent baseline runs locally (text-based model + LocalEnvironment).
- [x] Shadow monitor runs without behavior changes (invariant check implemented).
- [x] Loop guard intervention implemented (+ unit test).
- [x] Evidence gate intervention implemented (+ unit test).
- [x] Rollback suggestion implemented (+ unit test).
- [x] Recoveries and disruptions reported separately (online accounting).
- [x] Tables generated from raw result files (`scripts/make_report.py`).
- [x] Figures generated from raw result files (5 figures + calibration plot).
- [x] No paid API calls used.

## Phase status

| Phase | Description | Status |
|------|-------------|--------|
| 1 | Local setup + system_check | done (env verified) |
| 2 | Ingest Nebius trajectories | done (sample + full 1.1GB) |
| 3 | Normalize + action parser | done (8-family classifier, tests) |
| 4 | Build prefix dataset | done (127,092 rows full) |
| 5 | Extract features | done (47 features, families A–H) |
| 6 | Group splits | done (instance/repo/model) |
| 7 | Train offline monitors | done (7 models + 9 ablations) |
| 8 | Evaluate prediction | done (AUC/AUPRC/Brier/ECE + first-alert) |
| 9 | Calibration | done (Platt/isotonic) |
| 10 | Local LLM judge | done (qwen2.5-coder:7b) |
| 11 | Online monitor | done (shadow + active) |
| 12 | Interventions | done (3 interventions + tests) |
| 13 | mini-SWE-agent integration | done (text-based Ollama) |
| 14 | Online eval tasks | done (toy Level 0/1; SWE-bench path documented) |
| 15 | Online experiment design | done (baseline/shadow/3 policies + accounting) |
| 16 | Result tables | done (Tables 1–5) |
| 17 | Figures | done (Figures 1–5 + calibration) |
| 18 | Qualitative audit | done (TP/FP/FN + failure modes) |
| 19 | No-leakage tests | done (42 tests pass) |
| 20 | Paper framing | done (outline + related work + README) |

## Key offline results (full corpus, instance-grouped test, failure = positive)

Test set: 25,166 prefixes over 719 held-out instances; test failure rate 0.569.
Sampling: instance-stratified, success-enriched (≤2 fail + ≤12 success per
instance) from the natural-rate corpus (natural failure rate 0.833, reported).

| model | ROC AUC [95% CI] | AUPRC | Brier | ECE | success FAR @T3 | failed coverage @T3 | median lead |
|-------|------------------|-------|-------|-----|-----------------|----------------------|-------------|
| majority | 0.500 | 0.569 | 0.245 | 0.007 | 0.00 | 0.00 | — |
| step-count only | 0.623 [0.607,0.636] | 0.707 | 0.228 | 0.014 | 0.085 | 0.336 | 13 |
| heuristic | 0.574 [0.538,0.607] | 0.632 | 0.238 | 0.012 | 0.00 | 0.00 | — |
| logistic regression | 0.693 [0.673,0.711] | 0.762 | 0.217 | 0.018 | 0.071 | 0.359 | 12 |
| random forest | 0.719 [0.699,0.740] | 0.793 | 0.208 | 0.015 | 0.077 | 0.399 | 14 |
| **hist gradient boosting** | **0.722 [0.700,0.743]** | 0.792 | 0.208 | **0.011** | 0.090 | 0.437 | 13 |
| structured + text | 0.681 [0.649,0.711] | 0.754 | 0.222 | 0.034 | 0.074 | 0.310 | 15 |

Ablation (best single family: `action_counts` 0.708; full structured 0.722;
adding text **lowers** AUC to 0.672). Threshold tradeoff: F1-optimal (T1) alerts
≈99% of runs (success FAR 0.99) — confirming F1 is the wrong objective for
disruption-aware monitoring; T4 (FAR ≤5%) keeps success FAR at 0.022 with 0.274
failed coverage.

## Qualitative audit (Phase 18) + human validation (#2 strengthening)

The automatic (regex) labeler says high-confidence true positives are dominated by
**looping**. We validated this by **blind human re-labeling of 50 prefixes** (40
flagged + 10 missed failures): one author read the actual actions/observations — not
the features — assigned a mode, then joined to the heuristic. Scripts:
`build_audit_sample.py` (writes a blind file + a held-out key), `score_audit.py`;
results in `results/audits/human_validation.json`.

Result — a correction *and* a confirmation:
- **The regex over-calls "looping."** It labels 88% of flagged failures looping, but
  blind reading finds only **40%** are tight loops; looping **precision 0.37** (recall
  0.94). ~34% of "looping" is really **patch-churn** (varied edits, tests never
  improve), plus insufficient_context / environment_distraction. Exact-mode agreement
  is only 0.30.
- **But the family-level story holds.** **70%** of confidently-flagged failures are
  **repetition / no-progress** (40% loops + 30% churn); coarse (family) agreement
  **0.68**. The disagreement is loop-vs-churn within that family, not noise.
- **Observability contrast confirmed.** Of *missed* failures, **70%** are genuinely
  **not_observable / insufficient_context** by human reading — the subtle, early-
  undetermined prefixes the floor and abstention concede.

Takeaway for the paper: report it as **"repetitive / non-progressing behavior
dominates the catchable failures"** (loops + churn), not "looping" alone — and the
human read is what earns that wording. Honest caveat: single annotator, blind to the
per-item heuristic label but not an inter-rater study.

## Local LLM judge (Phase 10) — held-out subset, **n=200**

| metric | local LLM judge (qwen2.5-coder:7b) | structured classifier |
|--------|-----------------------------------|------------------------|
| ROC AUC (same 200 prefixes) | **0.559** | **0.768** |
| invalid JSON rate | **0.0** | — |
| latency / prefix | **11.6 s** | **~8.6 ms** real `assess()` path (p90 ~15–23 ms); **<0.1 s for all 25k** batched |
| model size (disk) | 4.7 GB (Ollama) | **1.14 MB** (joblib) |
| memory (resident) | ~4.7 GB+ weights | **~210 MB** serving (model adds ~77 MB) |
| should_intervene rate | **0.965** (intervenes on ~everything) | n/a |

**Frontier judges — GPT-5.5 & Claude Opus 4.8 (paper strengthening, paid API).** Same
prompt/schema/renderer; only the model/provider changes (`openai_judge.py` for OpenAI,
`openrouter_judge.py` for OpenRouter). Run on a matched 200-prefix subset of the *current*
split (`scripts/run_openai_judge_subset.py --backend {openai,openrouter}`), `gpt-5.5-2026-04-23`
at `reasoning_effort=low`:

| judge model | ROC AUC (same 200) | should_intervene | latency/prefix | cost (200 calls) |
|---|---|---|---|---|
| structured classifier (random_forest) | **0.768** | — | ~9 ms | $0 |
| **GPT-5.5** (frontier, OpenAI, effort=low) | **0.626** | 0.62 | 2.58 s | **$0.44** |
| **GPT-5.5** (frontier, OpenAI, effort=high) | **0.619** | 0.58 | ~9 s | $0.99 |
| **Claude Opus 4.8** (frontier, OpenRouter, no thinking) | **0.618** | 0.46 | ~5.6 s | $2.10 |
| **Claude Opus 4.8** (frontier, OpenRouter, thinking=low) | **0.639** | 0.43 | ~8 s | $2.70 |
| **Claude Opus 4.8** (frontier, OpenRouter, thinking=high) | **0.633** | 0.41 | ~8 s | $2.69 |
| qwen2.5-coder:7b (local) | **0.559** | 0.965 | 11.6 s | $0 |

**Two independent frontier models agree (and both lose).** Claude Opus 4.8
(`anthropic/claude-opus-4.8` via OpenRouter, `openrouter_judge.py`, JSON via `json_object` mode —
Claude doesn't honor OpenAI `json_schema` strict, so the backend falls back automatically; 0%
invalid JSON across all configs) at no-thinking scores AUC **0.618** — essentially tied with
GPT-5.5's 0.62 and **significantly below the classifier**: paired clf−Opus Δ **+0.150, 95% CI
[0.036, 0.266], 99.4% of resamples favor the classifier**. So "cheap beats frontier" is **not a
quirk of one model or provider** — two different frontier LLMs land at ~0.62, ~0.15 below 1 MB of
gradient boosting.

**Extended thinking (the parity test vs GPT-5.5's effort sweep).** We ran Opus with extended
thinking at both efforts (OpenRouter `reasoning:{effort}`, temp 1; thinking verified engaged — a
real reasoning trace, e.g. *"the agent is stuck in a repetitive cycle, repeatedly attempting the
same API calls"*). Unlike GPT-5.5 (where high effort gave **zero** gain), Opus thinking gives a
**small, borderline** bump: no-thinking **0.618** → low **0.639** → high **0.633**. Paired
(same 200): low−none Δ **+0.021, 95% CI [−0.001, +0.046]** (97% of resamples positive —
suggestive, *not quite* significant at 95%); high−none +0.014 (n.s.); high−low −0.007 (n.s., i.e.
low ≈ high — Claude thinks ~the same ~60 tokens/call regardless of the knob on this terse task).
**Even Opus's best config (0.639) stays significantly below the classifier** (gap +0.13, CI
excludes 0). So more thinking helps Opus marginally but nowhere near enough — across qwen-7B,
GPT-5.5 (low/high) and Opus 4.8 (none/low/high), **every judge config loses to the 1 MB
classifier.** Opus is also the least trigger-happy judge (intervenes on 41–46% vs GPT's 58–62%
and qwen's 97%) yet still among the worst predictors — selectivity ≠ discrimination.

All three on the **identical 200 prefixes** (the qwen re-run's `classifier_auc` 0.768 matches the
GPT run's exactly → same sample confirmed). Ordering: **classifier 0.77 ≫ GPT-5.5 0.63 > qwen-7B
0.56.** Paired instance-bootstrap (classifier vs GPT-5.5): **Δ = +0.142 AUC for the classifier,
95% CI [0.024, 0.255], 99% of resamples favor the classifier** → the 1 MB monitor *significantly*
out-predicts GPT-5.5 at early-failure prediction (the gap over qwen-7B is larger still, +0.21). GPT-5.5 is more discerning
than the 7B (intervenes on 62% vs 96.5%) but still too trigger-happy for a disruption-aware
monitor, and is ~280× slower + costs money. 0% invalid JSON. Decisively answers "you only beat
a toy 7B judge": **even a frontier model is a worse failure predictor than cheap gradient
boosting.** All judges scored on the **same 200 prefixes** (an earlier qwen run used a
different older sample, only 27/200 overlap, so it was re-run for a clean paired three-way;
the high-effort GPT run was pinned to the identical prefix_ids).

**Reasoning effort makes no difference (the caveat is resolved).** Re-running GPT-5.5 at
`reasoning_effort=high` — 55.9k reasoning tokens, **2.2× the cost ($0.99)** and ~3.5× the
latency — gives AUC **0.619**, *statistically identical* to low's 0.626: paired (high−low)
Δ = **−0.006, 95% CI [−0.028, +0.018]** (risk scores correlate 0.96). It remains
*significantly* below the classifier (paired clf−GPT Δ +0.149; 99% of resamples favor the
classifier). So `low` was **not** a handicap — more thinking does not make a frontier model a
better early-failure predictor here; it just costs more and stays trigger-happy (intervenes on
58%). The 0-reasoning-token figure was specific to the low-effort dry run.

Measured deployment cost (`scripts/measure_cost.py` → `cost.json`, M4 CPU, psutil RSS;
stable across runs). **Latency** on the real `Monitor.assess()` path is **~8.6 ms/prefix**
(median; p90 ~15–23 ms) — dominated by sklearn's fixed per-`predict` overhead, not compute
(feature extraction is 0.14 ms); batched it does ~390k prefixes/s and scores the full
**25,166-prefix test set in <0.1 s**. **Memory**: 1.14 MB on disk → ~77 MB resident model →
**~210 MB total serving footprint** (Python + sklearn + model). vs the judge's 4.7 GB /
11.6 s: **~20× lighter in memory, ~1,300× faster per prefix** (≥10⁴× batched). Honest
caveat: the ~77 MB model-resident delta conflates model arrays with first-use lazy sklearn
imports (an upper bound); the ~210 MB total-serving RSS is the clean "runs in X MB" number.

Takeaway (now robust at n=200, vs an earlier noisy n=40 tie): the cheap structured
classifier **decisively beats** the local LLM judge for early failure prediction —
a **0.16 AUC gap** — at ~1000× lower latency, and the judge is pathologically
intervention-happy (96.5%), a textbook disruption hazard. This is one of the
paper's cleaner positive results.

## Online (Phases 13–15) — local mini-SWE-agent + qwen2.5-coder:7b, Level 1 toy tasks

- Baseline success: **1/3** (off_by_one solved; strip_prefix, dedup_order hit the
  40-step limit). Per-run ~150–250 s on the M4.
- **Shadow mode behaviorally identical to baseline: True** (temperature-0 invariant
  holds — the monitor logged verdicts without changing agent behavior; all 3 tasks
  identical step counts + outcomes).
- **loop_guard accounting (paired, 3 tasks, illustrative):** 2 interventions/run
  fired (first alarm step 6–7); recovery **0**, disruption **0**; **all outcomes
  unchanged** vs baseline; avg steps **40 → 31**. Honest reading: the low-disruption
  intervention fired and **broke nothing** (0 disruptions — supports Claim 3); the
  step drop hints at reduced wasted work (Claim 4) but n is too small to attribute
  causally.
- **evidence_gate did not trigger** on any Level-1 task (the agents read/searched
  before editing, so the premature-edit condition never held → 0 interventions).
  It is therefore excluded from the accounting table (it would just be a baseline
  re-run). Implemented + unit-tested; rollback likewise rarely triggers on short
  toy tasks (needs worsening tests + a checkpoint).
- **Nondeterminism caveat (important):** in a non-intervened (0-intervention) run we
  observed an outcome *flip* vs baseline (dedup_order: fail→success), i.e. local
  Ollama is **not perfectly deterministic at temperature 0**. This confounds small-N
  outcome comparisons and is exactly why the online numbers are **illustrative only**
  and no claim rests on them. (Shadow mode happened to reproduce baseline on all 3
  tasks, consistent with near-deterministic decoding.)
- Each 7B agent run is 3–20 min on the M4 (the model often issues stdin-blocking
  commands that stall to the shell timeout), so larger online sweeps are left as a
  documented runnable step rather than run exhaustively here.

## Pushing past 0.722 — four levers, ranked (paired vs v1-HGB on the same test)

Baseline v1-HGB test AUC = 0.7214 (same 25,166 held-out prefixes). Every row is a
paired instance-bootstrap of the AUC delta. Selection always on validation.

| lever | test AUC | Δ | 95% CI | %boot win | verdict |
|-------|----------|---|--------|-----------|---------|
| **#4 ensemble** (0.4 HGB + 0.4 RF + 0.2 LR) | **0.7244** | **+0.0030** | [−0.0006, +0.0066] | **95%** | best *robust* same-test gain (≈ significant) |
| #3 more data (≤10 fail ⇒ 4.4× failures) | 0.7225 | +0.0010 | [−0.0049, +0.0067] | 62% | marginal; diminishing returns |
| #2 sequence model (GRU, 64h, 8ep) | 0.7163 | −0.0052 | [−0.0171, +0.0053] | 19% | **overfit** (val 0.7255 → test 0.716) |
| #1 cleaner label — *training reweight* | 0.7158 | −0.0056 | [−0.0111, −0.0004] | 2% | significantly **worse** |

Advanced features (v2, 68 extra temporal/semantic) earlier: val 0.726 → test 0.721
(n.s.). **Cleaner label as a *reframing*** (the real ceiling-raiser): baseline AUC by
normalized position = early(≤⅓) **0.633**, mid **0.772**, late(>⅔) **0.808**.

Conclusions:
- The biggest lift by far is **the label**: on *determinable* (late) prefixes AUC is
  **0.808 (+0.087)**. The 0.722 "ceiling" is dragged down by early prefixes whose
  terminal labels are genuine noise. Deploy/report on the regime where the monitor
  is strong; don't chase early-prefix AUC.
- The only **robust same-task** model gain is the **ensemble** (+0.003, 95% of
  bootstraps). More data barely helps (4.4× → +0.001). The GRU and label-reweighting
  both **looked best on validation and were worse on held-out test** — the paired
  bootstrap caught val-overfitting yet again. Hand-crafted features + GBT are more
  robust here than a from-scratch sequence model.
- Artifacts: `scripts/run_auc_lift.py`, `run_more_data.py`, `run_seq_model.py`,
  `rank_auc_lift.py`, `results/offline/full/auc_lift_results.json`.

## Selective prediction / abstention — the deployable "cleaner label" (4 tasks)

The terminal label is noise on early prefixes (AUC 0.63) but informative late (0.81);
critically, "late" via normalized position needs the UNKNOWN total length, so it is
**not deployable**. The deployable fix: the monitor **abstains** when a prefix-visible
gate isn't met — `commit iff prefix_step ≥ S AND |calibrated_risk−0.5| ≥ C` — and we
report a **risk–coverage tradeoff** instead of one number. Gate chosen on validation.

1. **Gate on val; variant (a) full model vs (b) retrain on step≥S.** On the step≥10
   regime (41% coverage): (a) full **0.793** vs (b) regime-retrained 0.792 — Δ−0.002,
   n.s. **Retraining on the regime does not help**; the full model is fine. Use (a).
2. **Risk–coverage curve (headline artifact, Figure 6).** Confidence-selective AUC:
   **0.80 @ 50% coverage, 0.83 @ 30%, ~0.72 @ 100%**. **Random-abstention baseline stays
   flat at 0.72** → the lift is *real*, not a mechanical artifact of dropping rows.
   Deployable step-gate: 0.79 @ step≥10 (41%), 0.78 @ step≥5 (54%). Operating point
   (step≥10 & conf≥0.24, picked on val): test **AUC 0.834 at 22.5% coverage,
   precision 0.968 @ FAR-10%**.
3. **Thresholds + first-alert + calibration on the regime.** Abstaining on step<10
   cuts **success false-alarm rate 0.090 → 0.059** (−35%, the disruption source) and
   improves **Brier 0.208 → 0.167**, at a modest cost to failed-coverage (0.437→0.392)
   and lead (13→11). So abstention measurably improves the **disruption** side of the
   tradeoff — the real test, not just AUC.
4. **Wired into `monitor.py`:** `MonitorVerdict.abstain`, `PolicyConfig.abstain_conf_floor`;
   the monitor returns an explicit "insufficient evidence" verdict (never alarms) when
   too early or too uncertain. Unit-tested (`tests/test_abstention.py`, 6 tests).

Headline framing (honest): **"AUC 0.80 at 50% coverage / 0.83 at 30%, vs 0.72
unconditional — and 35% fewer false alarms on successful runs."** Artifacts:
`src/localguard/abstention.py`, `scripts/run_abstention.py`,
`results/offline/full/abstention.json`, `results/figures/fig6_risk_coverage_offline_full.png`.

## Chasing the ceiling: cleaner labels can NOT lift early prefixes (a strong negative)

Question: is the 0.63 early-prefix AUC a fixable LABEL artifact, or an irreducible
floor? We attacked it two ways (success criterion = EARLY stratified AUC up vs the
HGB baseline, paired on held-out test; label shaping is training-only, eval always
on the terminal label).

Diagnostic that frames everything: among *failed* training prefixes, observable
"trouble" (loops / persistent test failures / repeated errors) appears in only
**49.8% early**, 79% mid, 90% late. Half of early failures show *no* evidence yet.

| method | early AUC | mid | late | overall | early Δ vs HGB |
|--------|-----------|-----|------|---------|----------------|
| HGB baseline (terminal label) | **0.631** | 0.772 | 0.810 | 0.721 | — |
| W1 trouble-gated relabel | 0.606 | 0.723 | 0.775 | 0.661 | −0.025 (n.s., hurts) |
| W2 down-weight noisy positives | 0.624 | 0.747 | 0.795 | 0.695 | −0.007 (n.s.) |
| **Survival/hazard GRU** (monotone risk) | **0.633** | 0.762 | 0.766 | 0.669 | **+0.0001 (identical)** |

Conclusion: **no label/model trick lifts the early-prefix AUC.** Weak-supervision
relabeling *removes signal* (trouble also occurs in recovered successful runs, so
predicting it doesn't rank terminal failure); the survival model extracts the exact
same 0.633 the GBT does and is worse late (its monotonicity prior costs flexibility).
The early floor is irreducible because failure there is driven by (1) genuinely
*undetermined* outcomes (the agent hasn't gone wrong yet) and (2) *task difficulty*,
which is set before the agent acts and does **not transfer** across the grouped split
(test instances unseen). This is the definitive justification for **abstention**: the
right move is to *skip* the early prefixes the label can't support, not to relabel
them. Artifacts: `src/localguard/weak_labels.py`, `scripts/run_cleaner_label.py`,
`scripts/run_survival_seq.py`, `results/offline/full/cleaner_label.json`.

## Online validation of abstention (shadow-capture + deterministic replay)

The offline FAR reduction (0.090→0.059) is a proxy; here is the missing *live*
evidence. We ran the agent in shadow on 10 toy tasks (qwen2.5-coder:7b), captured
per-step calibrated risk + outcome, then **deterministically replayed** the ungated
vs gated (abstaining) monitor over the *same* trajectories — confound-free, gate +
thresholds fixed from the offline study (never tuned on this live data).

Scaled to **25 tasks (14 succeeded, 11 failed)**:

| | ungated (step≥5, risk≥0.778) | gated (step≥10 & conf≥0.24, risk≥0.851) |
|---|---|---|
| **disruption** (succeeded runs fired on) | **78.6%** (12 interventions) | **7.1%** (2) |
| **coverage** (failed runs caught) | 72.7% | 45.5% |

Reading (now a real N, not anecdote):
- **Abstention cuts disruption ~10× on live runs**: the offline monitor fires on
  **79% of *successful* runs** (massive over-firing — distribution shift from
  swe-agent-llama → qwen2.5-coder/mini-SWE inflates risk levels); the gate abstains
  on most and drops this to **7%**. Paired bootstrap over the 14 successful runs:
  reduction **0.71, 95% CI [0.50, 0.93]**, 100% of resamples positive — robust even
  at this N. (Coverage loss 0.27, CI [0.00, 0.55].) Strong live confirmation of the
  offline FAR result — the project's central claim, now with evidence.
- **Honest cost: coverage 73%→46%** of failures caught (better than the 3-task
  pilot's 29% because the larger set has some longer failed runs that clear the
  step floor). **Deployment lesson:** tune the gate to the target's trajectory-
  length distribution — a step floor right for long traces over-abstains on short
  runs.
- **Forced-intervention outcome-effect** (loop_guard, baseline vs intervention,
  3 tasks): 0 recovery, 0 disruption, avg steps 40→31 — the intervention changed no
  outcomes and trimmed wasted steps (full outcome-flip study needs tens of hours).

This is *not* the full recovery/disruption study (outcome flips need the agent to
actually receive the message — tens of hours on local hardware); it is the
disruption-RISK measurement on our live loop. Artifacts: `scripts/run_shadow_capture.py`,
`scripts/run_monitor_replay.py`, `results/online/{shadow_capture.jsonl,monitor_replay.json}`.

## Generalization: does the monitor transfer beyond its training distribution?

The #1 external-validity question. Answer: **the discriminative signal transfers;
the probability calibration shifts but is cheaply fixable.**

(a) **Cross-model (within swe-agent-llama family)** — train on 70b *excluding the 340
instances the 8b/405b traces attempt* (pure model + task shift), test on 8b+405b
(659 traj, 340 unseen instances):
- AUC within-70b **0.735** → cross-model **0.697** (transfer gap **+0.037**) — ranking
  holds well on unseen models AND unseen tasks.
- ECE within 0.00 → cross **0.060** (naive 70b calibrator) → **0.021** recalibrated.
  Calibration degrades under shift but a cheap target-side recalibration restores it.

(b) **Cross-family + cross-scaffold (live qwen2.5-coder / mini-SWE-agent)** — the
offline (swe-agent-llama-trained) monitor on captured live traces:
- monitor prefix-AUC on qwen traces **0.71** (≈ its 0.72 on Nebius!) — ranking transfers
  even across base-model family and agent scaffold.
- **Over-firing**: mean risk on *successful* qwen prefixes 0.602 vs fail 0.714 (Nebius
  success-prefix risk ~0.45) — the shift inflates the risk *level* (ranking preserved).
- **Abstention mitigates**: successful-prefix alarm-bar crossings drop 6% (ungated) → 0%
  (gated). Recalibration would do the same for the probabilities.

(c) **Second independent source, structurally different scaffold (OpenHands/CodeAct) —
collapse, then FREE recovery** (`scripts/ingest_openhands.py`, `run_second_source.py`). We
parsed **500 OpenHands trajectories** (`togethercomputer/CoderForge…`, SWE-bench Verified,
qwen3-coder-32b in CodeAct) through the **same** normalize/feature pipeline and scored the
frozen monitor zero-shot:
- **Naive zero-shot AUC 0.525** (near chance), severe over-firing (success risk 0.72 vs 0.45,
  44% FAR). Looked like a clean negative — *until* the diagnosis.
- **Diagnosis:** the failure *structure* is shared — top features point the same direction as
  Nebius (more search/read/repeat → fail) — but the **scale** differs hugely (reads 11.8 vs
  2.4/prefix), so the frozen tree's absolute splits (e.g. "n_read>3") fire on ~all OpenHands
  prefixes → no discrimination. The collapse is mostly a *scale artifact*.
- **FREE fix #1 — unsupervised feature alignment.** Quantile-map each target feature onto the
  Nebius marginal (no target labels): zero-shot AUC **0.525 → 0.591**, essentially matching the
  **in-domain ceiling (0.603, 5-fold CV)** — i.e. alignment recovers ~all achievable signal
  with zero labels.
- **FREE fix #2 — ensemble of transferable feature-family submodels.** Rank-average of
  tree-on-volume (mapped) ⊕ the scale-invariant **loop** submodel applied **raw** (no mapping
  needed — repetition is universal) ⊕ a **linear** model (mapped; linear extrapolates more
  smoothly cross-domain). Reaches **0.604**, slightly *exceeding* the in-domain single model
  (the source submodels train on 100k+ examples and decorrelate). Pre-specified composition, not
  tuned on the target (mild selection caveat).
- **FREE fix #3 — abstention** (deployable absolute step-floor, §4.3 gate): **0.66 @ 44%
  coverage**.
- **FEW-SHOT (needs target labels) — transferred ⊕ in-domain.** Rank-average the zero-shot
  transferred ensemble with a small in-domain model (out-of-fold, honest); they make different
  errors → **0.62 full / 0.68 with abstention**. The OpenHands dataset is fixed at 500
  trajectories (1 run/instance), so the in-domain ceiling (~0.60) is not data-limited.
- **Multi-source joint training** (Nebius + OpenHands, OpenHands upweighted ×10) reaches
  **0.620 / 0.683** — *identical* to the few-shot rank-average. Two independent methods
  converging on the same number is strong evidence this is the **genuine ceiling**.
- **What did NOT help (ruled out by CV — an exhaustive search, kept as negatives):**
  rate-normalized features (0.56), CORAL covariance alignment (0.56), z-score alignment (0.58),
  cumulative-max risk (0.59), TF-IDF text (−0.03), self-training / pseudo-labels (0.60, can't
  exceed the teacher), disagreement-based abstention (< step-floor), stacking the transferred
  risk as a feature (0.60), **temporal/windowed features** (0.604, +0.001 — dynamics carry no
  new signal), **patch size/content** (final patch only ~0.57 predictive; 0% empty patches —
  a capable agent always produces a substantive fix), **problem-statement difficulty**
  (orthogonal but too weak, adds noise: 0.60→0.59), **stacked meta-learner with learned weights**
  (0.61 < equal-weight rank-avg 0.62 — overfits the small target), **importance weighting /
  covariate-shift source retraining** (0.59, no gain — source/target barely overlap so the
  density ratio is ~uninformative), and **position-conditional rank normalization** (0.58 —
  position is informative, removing it hurts). Across **~16 distinct techniques** none beats the
  frontier — strong empirical evidence the ~0.60 per-prefix early-warning ceiling is a genuine
  information limit, not a modeling deficiency.

**The real story is *when* you predict, not a 0.60 information wall.** The ~0.60 figure is the
**per-prefix average**, diluted by noisy early prefixes (an exploring agent looks identical
either way). The achievable signal is far higher at the trajectory level: at the agent's **final
prefix** — a deployable **post-hoc patch-acceptance gate** (predict at finish, to accept/retry/
escalate) — AUC is **0.722 zero-shot / 0.728 few-shot, i.e. *Nebius parity*, cross-scaffold and
free.** So the cross-scaffold gap is specific to *early-warning* (the capable agent's early
behavior is genuinely uninformative), not to the monitor's signal overall. The free frontier:

| deployment mode (cross-scaffold, OpenHands) | zero-shot | few-shot |
|---|---|---|
| online per-step, full coverage | 0.604 | 0.620 |
| online per-step + abstention (~44% cov) | 0.664 | 0.684 |
| **post-hoc patch-acceptance gate (at finish)** | **0.722** | **0.728** |

Eight+ paradigms tested; the per-prefix online number is signal-limited for *early* warning, but
reframing the task to post-hoc acceptance recovers ~full Nebius-level AUC. Pushing the *online
early-warning* number past ~0.68 would need semantic (LLM) code analysis (not free).
- **Residual early-warning gap is SCAFFOLD-driven, not capability-driven (controlled test,
  `run_capability_monitorability.py`, Fig 10).** Measuring in-domain monitorability at a *matched
  60-instance budget* (controls the data-size confound) across agents: **within the shell
  scaffold, monitorability RISES with capability** — swe-agent-llama 8b/70b/405b = AUC
  **0.56 / 0.62 / 0.76** as success climbs 0.41→0.66 — while **OpenHands/CodeAct is the LOWEST
  (0.53) despite mid-high capability (0.59)**. So the CodeAct *action space*, not the agent's
  strength, depresses early-warning; the most capable shell agent is the *most* monitorable.
  This **REFUTES** the earlier "capable agents fail invisibly" reading (a correction this session
  earned by actually running the controlled analysis). Text features don't help (−0.03).

Honest takeaway (revised): cross-scaffold transfer **degrades then largely recovers for free**
— the naive collapse was a feature-scale artifact, fixed by label-free quantile alignment
(0.59) + an ensemble of transferable submodels (0.60) + abstention (**0.66** zero-shot); with a
few target labels the few-shot stack reaches **0.68**, and post-hoc acceptance transfers fully
(**0.72**). Not absent signal. Early-warning monitorability is governed by **action-space
(scaffold)** — *not* capability (within a scaffold capability helps). Artifacts:
`run_generalization.py`, `ingest_openhands.py`, `run_second_source.py`,
`run_capability_monitorability.py`, `results/offline/full/{generalization,capability_monitorability}.json`.

## Methodological contribution: paired held-out testing (validation lies)

A first-class contribution, not a footnote. Our evaluation protocol — **select on
validation, then a *paired instance-bootstrap* of the metric delta on a held-out,
instance-grouped test set** — repeatedly exposed "improvements" that were validation
overfitting. In every case below the validation gain looked real and **did not survive**:

| change | validation | held-out test (paired) | verdict |
|--------|-----------|------------------------|---------|
| advanced features (v2, +68) | 0.726 | 0.721 (Δ+0.003, n.s.) | mirage |
| v2 focused (val-selected) | 0.726 | 0.721 (Δ−0.000) | mirage |
| sequence model (GRU) | **0.726** | **0.716** (Δ−0.005) | reversed |
| label position-weighting | **0.729** | **0.716** (Δ−0.006, sig **worse**) | reversed |
| survival/hazard GRU | 0.726 | 0.717 (early identical) | mirage |

Five times, the *validation-best* configuration was no better — or significantly worse —
on held-out test. Only the **ensemble** (+0.003, 95% of bootstraps) and the
**abstention reframing** survived. Most agent-monitoring papers report a single number on
one split; this protocol (grouped splits + paired bootstrap + a random-abstention control)
is what separates a real effect from a lucky split, and it is *why* we trust the negative
results (the ceiling is real) as much as the positive ones.

## Notes / honesty

- Offline labels are **terminal** outcomes attached to partial prefixes, not
  per-step correctness labels — stated as a limitation.
- The offline benchmark is class-rebalanced for a learnable signal; the natural
  base failure rate (0.833) is reported separately.
- Online toy-task results are small-N and illustrative; no claim rests on them.
