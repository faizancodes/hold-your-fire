# Related work

LocalGuard-SWE sits at the intersection of (1) online failure prediction for LLM
agents, (2) the safety/usefulness of *acting* on such predictions, and (3)
coding-agent evaluation. We do not claim to be the first to predict agent
failures; our contribution is a **local, open-source, disruption-aware** study
under consumer-hardware constraints, specialized to coding agents.

## Online failure-warning monitors

**PrefixGuard** (Zhang et al., *From LLM-Agent Traces to Online Failure-Warning
Monitors*, arXiv:2605.06455) turns raw agent traces into online prefix-risk
monitors: a one-time offline "StepView" induction generates deterministic
adapters for heterogeneous trace formats, then a differentiable event-abstraction
layer is trained jointly with a replaceable monitor backend from a prefix-warning
objective. Reported AUPRC: 0.900 (WebArena) / 0.710 (τ²-Bench) / 0.533
(SkillsBench) / 0.557 (TerminalBench). LocalGuard differs by (a) targeting
**coding** agents on SWE-agent trajectories, (b) using **CPU-only classical
models** instead of a differentiable abstraction layer, and (c) coupling
prediction to a disruption-aware intervention study.

**AgentForesight** (*Online Auditing for Early Failure Prediction in Multi-Agent
Systems*, arXiv:2605.08715) and **TRACES** (*Proactive Safety Auditing for
Multi-Turn LLM Agents via Trajectory-State Modeling*, arXiv:2605.27690) study
early failure prediction and proactive auditing, primarily in multi-agent /
safety settings. LocalGuard is single-agent and coding-specific, and emphasizes
*early-warning lead time* and *false alarms on successful runs* as first-class
metrics.

## Prediction ≠ safe prevention (our core motivation)

**The Intervention Paradox** (*Accurate Failure Prediction in Agents Does Not
Imply Effective Failure Prevention*, arXiv:2602.03338) is the key motivation for
our evaluation design. It shows a binary LLM critic with strong offline accuracy
(AUROC ≈ 0.94) can still cause severe degradation — a reported **26 percentage
point** collapse on one model while barely affecting another — because the harm
magnitude tracks the *agent's disruption-to-recovery ratio*, not the critic's
accuracy. It recommends a small (≈50-task) pre-deployment pilot to estimate
whether intervention helps. LocalGuard adopts this lens directly: we measure
**recovery vs. disruption** separately and never claim improvement from AUC or
success rate alone.

**Canonical Path Deviation** (*Capable but Unreliable*, arXiv:2602.19008) frames
long-horizon agent failure as deviation from a canonical solution path — a useful
mechanistic complement to our structured "drift" features (premature edits,
loops, churn).

## Coding-agent evaluation and trajectories

We use **nebius/SWE-agent-trajectories** (80,036 SWE-agent-style runs) as the
offline corpus, and **mini-SWE-agent** (Klieret et al.) for local online runs via
its text-based LiteLLM model against an **Ollama** OpenAI-compatible endpoint.
Online task success is measured on small local bug-fix repos (Level 0/1); the
**SWE-bench Verified Mini** (50-task) path is implemented but gated behind a
smoke-test because official SWE-bench evaluation is resource-intensive and
x86-oriented.

## Positioning

| Axis | PrefixGuard / AgentForesight | Intervention Paradox | **LocalGuard-SWE** |
|------|------------------------------|----------------------|--------------------|
| Domain | general / multi-agent | general | **coding agents** |
| Compute | research GPUs | research | **Mac-local, CPU, Ollama** |
| Model | learned abstractions / LLM critic | LLM critic | **classical + cheap LLM judge baseline** |
| Focus | prediction quality | when intervention harms | **both, under local constraints** |
| Cost | — | — | **$0 inference, open-source only** |
