0. Goal, hypothesis, and the branch that shapes everything
Goal. Find the internal mechanism by which a coding agent re-issues a command it has already tried (a "tight loop"), prove it's causally responsible, and build a steering vector that breaks loops during generation — then show this is less disruptive than LocalGuard's external hard reset.

Two competing hypotheses (your first real finding decides the rest):

H1 — ignorance: the model has no internal representation that "I've done this before / this isn't working," so it loops blindly. → Steering must inject novelty.
H2 — knows-but-doesn't-act: the model does represent "I'm repeating an unproductive action" but fails to route it into behavior. → Steering should amplify/route that existing signal.
Phase 3's probe decodability tells you which world you're in. Design for both.

Model & constraints. Use Qwen2.5-Coder-7B-Instruct (white-box, matches LocalGuard's live agent). Develop the whole pipeline on a small sibling (Qwen2.5-Coder-1.5B/3B) for speed, then confirm on 7B. Ollama can't expose activations — you must load via HF transformers + nnsight (steering/generation-time edits) and/or TransformerLens (head-level circuit analysis). On an M4/32GB, 7B in bf16 (~15GB) fits but generates slowly; the localization work runs on cached activations and is cheap.

Phase 1 — A white-box agent harness (the enabling infrastructure)
Swap the inference backend. LocalGuard runs mini-SWE-agent on Ollama. Write a drop-in backend that loads Qwen via HF/nnsight and exposes two extra capabilities the loop needs: (a) cache the residual stream at chosen layers/positions per generation, and (b) intervene (add a vector to the residual stream during .generate). Keep the agent loop, prompts, and parsing identical to LocalGuard's so trajectories stay comparable.
Reuse LocalGuard's labeling online. At every step, run action_parser + features.extract_features on the prefix so each step carries its real loop features (max_command_repeat_count, repeated_exact_command_last_3/5, same_test_command_repeated, edit_test_edit_test_loop_count, same_action_type_streak, read_same_file_repeatedly). These are your loop labels.
Define the readout position precisely. The "decision point" = the residual stream at the final context token (the one whose next token begins the action), per layer. You'll read here and steer across the generated action tokens.
Pick tasks. Start on a handful of synthetic loop-prone tasks and SWE-bench-Lite instances where 7B is known to flail. You want plenty of natural loops; you don't need task success.
Output: an instrumented agent that emits, per step: tokens, LocalGuard features, terminal label, and cached activations.

Phase 2 — A contrastive "loop vs not" dataset (the scientific crux)
The whole study lives or dies on a clean contrast. A loop state differs from a healthy state in many nuisance ways (longer context, repeated tokens, "frustration" language, harder task). Control them:

Primary contrast — within-command, first-vs-repeat. For a command C, take the decision-point activation at its first emission (label = novel) vs its k-th emission (label = loop). This holds the action content constant and varies only "have I been here before." This is your cleanest causal handle.
Secondary contrast — cross-trajectory. Loop steps from failing runs vs novel-progress steps from successful runs, bucket-matched on step index / context length to kill the length confound.
Synthetic dose-response set. Hand-build agentic contexts where the same failing command repeats K = 1…6 times, and read the signal as a function of K. A clean, controlled "does the seen-before signal grow with repetition?" curve.
Split discipline (reuse LocalGuard's rigor). Group every split by instance_id; never let two states from the same task straddle train/test. This is exactly the protocol that caught five mirages in LocalGuard — probing overclaims constantly, so inherit the discipline.
Output: a labeled activation dataset (loop/novel) with matched controls + a synthetic dose-response set.

Phase 3 — Localize the "repetition / unproductive-repeat" representation
Layer-wise linear probes. Train logistic probes on the residual stream at each layer to predict loop-vs-novel; plot probe-AUC by layer with instance-grouped CV + paired bootstrap CIs. The peak layer(s) locate the signal. This is the H1/H2 decision: high decodability ⇒ H2 (knows), chance ⇒ H1 (ignorant).
Distinguish "a repeat occurred" from "repetition is unproductive." Critical subtlety: an induction head trivially copying the previous command is banal. The interesting target is a signal that the repetition is failing. Test this by checking whether the direction also tracks the failure/no-progress label (e.g., does it correlate with tests_improving = 0, last_test_returncode != 0), not merely with "tokens repeated."
Attention/head analysis (TransformerLens). Score each head for attending from the decision token back to the prior identical command (induction-/previous-occurrence-style). Rank heads by this score on loop steps.
MLP/neuron analysis. Find neurons whose activation correlates with max_command_repeat_count (continuous) and with the loop label.
Extract candidate directions. At the peak layer, compute (a) the probe weight direction and (b) the diff-of-means direction v = mean(resid | novel) − mean(resid | loop). Also the first-vs-repeat direction from Phase 2.5. These feed steering.
Output: the layer(s) where loop-awareness lives, a ranked list of heads/neurons, and 2–3 candidate steering directions.

Phase 4 — Causal necessity: ablation & patching (off-policy, cheap)
Activation patching. Patch the loop-direction activation from a loop step into a novel step (and vice versa) at the peak layer; measure the change in the next-token logits for logit(repeat) − logit(novel). If patching the "loop" representation in makes a healthy model want to repeat, you've shown sufficiency of that representation for the intent.
Head/neuron ablation. Mean-ablate the candidate heads/neurons at loop steps; measure Δ p(repeat). Identifies which components are necessary.
Directional ablation. Project the residual stream off the loop direction; measure the same Δ. A clean test that the direction (not just specific heads) carries the behavior.
Metric. Δ p(repeat at the decision point) and Δlogit(repeat − novel), with paired bootstrap CIs. Score "did the top-1 action become novel?" as a cheap behavioral proxy.
Output: a validated minimal set of components/direction that are necessary for (or causally bias toward) repetition.

Phase 5 — Build & calibrate the steering vector
Construct v. Primary: diff-of-means (novel − loop) at the peak layer (a.k.a. contrastive activation addition). Adding +αv pushes a loop state toward the novel-state manifold. Keep the probe direction and first-vs-repeat direction as alternates; compare them.
Calibrate α (coefficient sweep). On held-out loop states, sweep α and watch two curves: loop-escape effect (rises) vs coherence/validity (falls past some α). Pick the α at the knee. Decide injection layer(s) and whether to apply at the decision token only or across all generated action tokens.
Decide the policy (depends on H1/H2): under H2, a modest +αv should suffice (route the existing signal); under H1, you may need a stronger/different "novelty" direction or to inject at more positions.
Output: a calibrated steering operator resid_L += α·v, plus its safe operating range.

Phase 6 — The payoff: on-policy steering test (must be live)
This is the headline experiment and it must intervene during real generation.

Trigger. Run live Qwen-7B agent trajectories; detect a loop in real time using LocalGuard's features (e.g., repeated_exact_command_last_3 = 1 or max_command_repeat_count ≥ 2). At that moment, switch on steering for subsequent generation.
Primary metric — loop-escape rate. Fraction of detected loops where the agent emits a novel next action, steering ON vs OFF.
Controls (this is where honesty is won or lost):
No-op (steering off) — baseline escape rate.
Random-direction vector, matched norm — rules out "any perturbation breaks loops."
Orthogonal-direction vector — shows the effect is specific to v.
Coherence/validity — fraction of steered actions that still parse as valid agent commands (guards against "it escaped by emitting gibberish").
Downstream effect. Steps-wasted saved, and outcome flips (does the run now resolve?) — scored with LocalGuard's recovery/disruption accounting and git-checkpoint replay.
The disruption test (the actual thesis). Apply the same steering to healthy, non-looping runs and measure how often it derails a run that would have succeeded. Compare this disruption rate head-to-head against LocalGuard's external hard reset / loop-guard (interventions.py) on the same healthy runs. The claim you're trying to earn: internal steering breaks loops while disrupting healthy runs far less than a hard reset.
Statistics. Matched-pair, instance-grouped design; report loop-escape and disruption deltas with paired instance-grouped bootstrap CIs (LocalGuard's protocol again).
Output: loop-escape rate, specificity vs controls, coherence, downstream recovery, and a steering-vs-reset disruption comparison.

Phase 7 — Triangulate, stress-test, write up
Mechanistic claim from three legs: probe (correlation) + ablation (necessity) + steering (sufficiency) → "component/direction X at layer L mediates unproductive repetition."
Robustness: does v transfer across unseen tasks (instance-grouped held-out), across a second model size/family, and does the dose-response (Phase 2.7) hold?
Honest negatives: report if steering trades coherence for escape, if the direction is entangled with "long context," or if it's just an induction head copying tokens (Phase 3.10 guards this). A null or partial result is still a real contribution.
Paper framing: "Steering coding agents out of failure loops: a low-disruption mechanistic intervention" — the disruption-vs-reset comparison is the bridge back to LocalGuard's Intervention-Paradox thesis.
Compute, de-risking, and risks
Feasibility / compute.

Localization (Phases 2–4) runs on cached activations — cheap, fully local even at 7B.
On-policy steering (Phase 6) is the slow part (7B generation on a Mac is a few tok/s). De-risk on 1.5B/3B first, get the full loop working, then confirm the headline numbers on 7B with a modest N (tens of looping episodes, instance-grouped).
No GPU training needed anywhere — probes are logistic regressions; steering is vector addition. Stays inside LocalGuard's constraints.
De-risking MVP (do this first, ~the first milestone): synthetic repeated-command prompts (Phase 2.7) → probe for the seen-before direction → diff-of-means vector → steer on the synthetic set and show the next-token distribution shifts from repeat→novel. If that works in isolation, the agentic version is "just" plumbing.

Key risks & mitigations:

Banal induction, not "stuck-awareness." → Phase 3.10: require the direction to track unproductiveness/failure, not just token repetition.
Steering escapes via incoherence. → random/orthogonal controls + validity rate (Phase 6.23).
Confounded direction (length/frustration). → matched contrasts (Phase 2.5–2.6).
Off-policy ≠ on-policy. → localize off-policy, but the causal steering test is strictly on-policy.
swe-agent-llama ≠ qwen. → generate fresh qwen trajectories so activations are on-distribution.
Reused LocalGuard assets: action_parser + features (loop labels), the mini-SWE-agent loop + git_checkpoints (live runs, replay), interventions.py (the reset baseline to beat), and the instance-grouped paired bootstrap (significance for every delta).