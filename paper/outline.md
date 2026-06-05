# LocalGuard-SWE: Disruption-Aware Failure Monitoring for Local Open-Source Coding Agents

## Abstract

LLM coding agents often fail after long trajectories of tool calls, edits, and
tests. Prior work shows that trajectory prefixes can predict failures, but
accurate prediction alone does not guarantee useful intervention. We study
failure monitoring for open-source coding agents under local-compute constraints
(an Apple-silicon laptop, Ollama, no paid APIs). Using 80,036 public SWE-agent
trajectories we train lightweight, CPU-only monitors from prefix-visible
structured features and compare them against heuristic and local-LLM judges. We
then integrate the monitor into a local Ollama + mini-SWE-agent loop and evaluate
interventions with recovery–disruption accounting. Our results show *when*
coding-agent failures are observable early, *which* signals matter, and *when*
intervention improves reliability or merely disrupts otherwise successful runs.

## 1. Introduction
- Problem: long coding-agent trajectories; final-outcome checks arrive too late.
- Gap: prediction ≠ safe prevention (Intervention Paradox).
- Constraints: local/open-source/consumer-hardware as a first-class requirement.
- Contributions:
  1. A reproducible, CPU-only, Mac-local monitoring pipeline over public
     SWE-agent trajectories.
  2. An honest offline study (no leakage; grouped splits; calibration;
     threshold/lead-time analysis).
  3. A disruption-aware online study with recovery/disruption accounting.
  4. A failure-observability audit: which failures are catchable early.

## 2. Background & related work
- Online prefix monitors (PrefixGuard, AgentForesight).
- The Intervention Paradox (motivates recovery/disruption accounting).
- Coding-agent trajectories & mini-SWE-agent; Ollama local inference.

## 3. Data & methodology
- Corpus: nebius/SWE-agent-trajectories (80,036; natural failure rate 0.833).
- Normalization → typed steps (8 command families, regex action parser).
- Prefix construction (terminal-outcome labels; schedule; **no future leakage**).
- Feature families A–H (length/pace, action counts, context-before-edit, file
  behavior, testing, loops, patch/churn [online], text).
- Group splits by instance/repo/model; success-enriched, class-rebalanced
  benchmark (natural rate reported).
- Leakage controls: forbidden-field guard, future-feature invariance, shuffled
  labels ≈ 0.5.

## 4. Offline failure prediction (Claim 1 & 2)
- Models: majority, step-count-only, heuristic, logistic, RF, HGB, structured+TF-IDF.
- Metrics: ROC AUC, AUPRC, Brier, ECE (instance-grouped bootstrap CIs).
- Result: HGB AUC **0.722** [0.700,0.743], ECE 0.011; structured > step-count; 
  **text does not help** (0.722 → 0.672).
- Ablation: `action_counts` strongest single family (0.708).
- Calibration: isotonic preserves/improves Brier; reliability plot.

## 5. Early warning & thresholds
- First-alert evaluation: success false-alarm rate vs failed coverage vs lead time.
- Threshold policies T1–T4; **F1-optimal alerts ≈99% of runs** (wrong objective).
- Risk-over-time: failed-run risk rises 0.52→0.78; successful stays ~0.5 (Fig 1).

## 6. Local LLM judge baseline (Claim 2)
- qwen2.5-coder:7b structured-JSON judge vs classifier on the same held-out set.
- Metrics: AUC/AUPRC, JSON validity, latency/prefix, intervention-happiness.
- Hypothesis: cheap structured features match/beat the local LLM judge for early
  prediction, at a fraction of the latency.

## 7. Online interventions (Claim 3 & 4)
- mini-SWE-agent + Ollama; shadow vs intervention; baseline/loop_guard/
  evidence_gate/rollback_suggest.
- Shadow invariant: identical behavior at temperature 0.
- Recovery/disruption accounting; resource savings (steps/tokens) without
  lowering success rate.
- Honest caveat: small-N online sample; illustrative, not conclusive.

## 8. Failure observability audit
- TP/FP/FN inspection; semi-automatic failure-mode taxonomy.
- Finding: **looping** failures are highly observable; **insufficient-context**
  and subtle failures are not → motivates loop_guard; bounds early-warning reach.

## 9. Limitations
- Terminal labels on partial prefixes (not per-step correctness).
- Rebalanced benchmark (natural rate reported separately).
- Single trajectory source / 3 base models; ARM Docker limits SWE-bench scale.
- Online sample size.

## 10. Conclusion
- Local, lightweight monitors identify actionable coding-agent drift; we show
  when intervention helps, when it harms, and which signals are reliable under
  open-source local-agent constraints.

## Tables & Figures
- T1 offline performance · T2 ablation · T3 threshold tradeoff · T4 online
  accounting · T5 local cost/resource.
- F1 risk-over-time · F2 PR curves (+judge) · F3 lead-time histogram · F4
  intervention accounting · F5 feature importance · calibration reliability.
