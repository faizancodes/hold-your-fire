Below is a detailed, research plan. The core idea is:

LocalGuard-SWE: a local, open-source, disruption-aware monitor that watches coding-agent trajectories, predicts when they are drifting toward failure, and only intervenes when the intervention is likely to help more than it hurts.

This is the publishable framing. Do not frame the project as “I trained a classifier on trajectories.” Recent work already does online failure-warning monitors and failure-step localization. PrefixGuard, for example, turns raw agent traces into online prefix-risk monitors, and AgentRx studies critical failure-step localization in agent trajectories.  ￼ The angle that makes your project stronger is: coding-specific, open-source-only, Mac-local, low-cost, and disruption-aware.

A very important paper to design around is The Intervention Paradox: it found that even a critic with strong offline AUROC can make agents worse, including a reported 26 percentage-point collapse in one setting, because interventions can disrupt trajectories that would have succeeded.  ￼ So your research should measure not only “did the monitor predict failure?” but also “did the intervention recover more runs than it disrupted?”

⸻

Project name

LocalGuard-SWE

Research goal

Build a local research system that studies this question:

Can a lightweight monitor, trained on coding-agent trajectory prefixes, detect when an open-source coding agent is going off-track and intervene safely enough to improve success or reduce wasted computation?

The system must run on an Apple M4 MacBook using local open-source models through Ollama. No paid APIs, no cloud GPUs, no remote inference.

Use public trajectory data first. The main offline dataset should be nebius/SWE-agent-trajectories, which contains 80,036 SWE-agent-style trajectories with fields including instance_id, model_name, target, trajectory, exit_status, generated_patch, and eval_logs.  ￼ Use local online experiments later with mini-SWE-agent, because mini-SWE-agent supports local models through LiteLLM by configuring model_kwargs such as custom_llm_provider and api_base.  ￼ Ollama has an OpenAI-compatible local endpoint at http://localhost:11434/v1, which makes it compatible with many OpenAI-style tools.  ￼

⸻

Hard constraints

1. Use only local inference through Ollama.
2. Do not call OpenAI, Anthropic, Gemini, Together, Fireworks, OpenRouter, or any paid/cloud inference provider.
3. Do not require GPU training.
4. Do not fine-tune a large LLM.
5. Start with CPU-friendly classical ML models.
6. All expensive experiments must have a small smoke-test mode.
7. All results must be reproducible from command-line scripts.
8. Do not use future information from a trajectory prefix.
9. Do not leak target, eval_logs, full final patch, or final exit status into features.
10. Split train/validation/test by instance_id, not by prefix row.

⸻

Recommended local models

Use these through Ollama:

ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b

qwen2.5-coder:7b is a good default because Ollama lists it as a 4.7GB model with a 32K context window; qwen2.5-coder:14b is listed as 9.0GB with a 32K context window.  ￼

⸻

Repository structure

Create this repo:

localguard-swe/
  README.md
  pyproject.toml
  Makefile
  .gitignore
  configs/
    offline_small.yaml
    offline_full.yaml
    online_shadow.yaml
    online_loop_guard.yaml
    online_evidence_gate.yaml
    online_rollback.yaml
    mini_swe_ollama_qwen25_7b.yaml
    mini_swe_ollama_qwen25_14b.yaml
  data/
    raw/                 # not committed
    interim/             # normalized trajectory chunks
    processed/           # prefix datasets
    samples/             # tiny sample fixtures for tests
  models/
    monitor/             # saved sklearn/joblib models
    calibrators/         # calibration models
  results/
    offline/
    online/
    figures/
    tables/
    audits/
  src/localguard/
    __init__.py
    schemas.py
    ingest_nebius.py
    normalize.py
    action_parser.py
    prefix_builder.py
    features.py
    split.py
    train.py
    evaluate.py
    calibrate.py
    thresholding.py
    monitor.py
    interventions.py
    mini_swe_wrapper.py
    ollama_judge.py
    git_checkpoints.py
    reporting.py
    utils.py
  scripts/
    system_check.py
    download_data.py
    inspect_dataset_schema.py
    build_prefix_dataset.py
    train_monitor.py
    evaluate_monitor.py
    run_ollama_judge_subset.py
    run_mini_swe_shadow.py
    run_mini_swe_intervention.py
    make_report.py
  tests/
    test_no_leakage.py
    test_group_split.py
    test_action_parser.py
    test_prefix_builder.py
    test_features.py
    test_thresholding.py
    test_interventions.py
  paper/
    outline.md
    related_work.md
    experiment_log.md

⸻

Main research claims to test

The project should test four claims.

Claim 1: failure is predictable from partial trajectories

At step t, use only the prefix up to step t and predict:

Will this trajectory eventually fail?

Measure AUC, AUPRC, calibration, false alarms on successful trajectories, and warning lead time.

Claim 2: coding-specific structured features are competitive

Compare:

simple heuristics
structured trajectory features
text features
structured + text features
local LLM judge through Ollama

The hypothesis is that cheap structured features may beat or match local LLM judging for early failure prediction.

Claim 3: intervention is not automatically good

Explicitly measure:

recovery = baseline failed, monitored run succeeded
disruption = baseline succeeded, monitored run failed

This is essential because prior work shows high offline prediction accuracy does not guarantee safe intervention.  ￼

Claim 4: local monitoring can reduce wasted work

Even if success-rate improvement is small, the monitor may still be useful if it reduces:

steps
tokens
repeated commands
patch churn
failed test loops

without lowering success rate.

⸻

Phase 1: local setup

Create a setup script.

brew install ollama
ollama serve
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b

Create a Python environment:

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install datasets pandas pyarrow numpy scikit-learn scipy matplotlib pydantic typer rich tqdm joblib pytest ollama openai
pip install mini-swe-agent

Add a system check:

python scripts/system_check.py

system_check.py should verify:

Python version
RAM
free disk
Ollama server available
qwen2.5-coder:7b responds
Docker available
mini-swe-agent imports
no paid API keys are required

Expected Ollama smoke test:

from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
resp = client.chat.completions.create(
    model="qwen2.5-coder:7b",
    messages=[{"role": "user", "content": "Reply with OK only."}],
    temperature=0,
)
print(resp.choices[0].message.content)

Verification gate:

PASS if the script prints:
- Ollama reachable
- qwen2.5-coder:7b reachable
- no paid API key needed
- mini-swe-agent import works

⸻

Phase 2: ingest public trajectories

Use the Nebius trajectory dataset as the primary offline dataset.

Implement:

python scripts/download_data.py --dataset nebius/SWE-agent-trajectories --mode sample --n 1000
python scripts/inspect_dataset_schema.py

The dataset can be loaded with:

from datasets import load_dataset
ds = load_dataset("nebius/SWE-agent-trajectories")

The Hugging Face dataset page shows this usage directly and reports a total file size around 1.11GB.  ￼

Create schemas

In src/localguard/schemas.py:

from pydantic import BaseModel
from typing import Any, Literal
class RawTrajectoryRow(BaseModel):
    instance_id: str
    model_name: str | None = None
    target: bool
    trajectory: str | list[dict[str, Any]]
    exit_status: str | None = None
    generated_patch: str | None = None
    eval_logs: str | None = None
class StepEvent(BaseModel):
    trajectory_id: str
    instance_id: str
    model_name: str | None
    step_index: int
    role: str | None = None
    raw_text: str = ""
    thought_text: str = ""
    action_text: str = ""
    observation_text: str = ""
    action_type: str = "unknown"
    command: str | None = None
    file_paths: list[str] = []
    returncode: int | None = None
    is_test_command: bool = False
    is_search_command: bool = False
    is_read_command: bool = False
    is_edit_command: bool = False
    is_git_command: bool = False
    is_install_command: bool = False
    is_submit_command: bool = False
    test_pass_count: int | None = None
    test_fail_count: int | None = None
    contains_traceback: bool = False
    contains_exception: bool = False
class NormalizedTrajectory(BaseModel):
    trajectory_id: str
    instance_id: str
    model_name: str | None
    target: bool
    steps: list[StepEvent]
    n_steps: int
class PrefixExample(BaseModel):
    prefix_id: str
    trajectory_id: str
    instance_id: str
    model_name: str | None
    prefix_step: int
    n_total_steps: int
    y_fail: int
    feature_dict: dict[str, float | int | str]

Verification gate:

PASS if 1000 raw rows can be converted into NormalizedTrajectory objects.
PASS if each normalized trajectory has:
- trajectory_id
- instance_id
- target label
- step list
- no eval_logs in step features

⸻

Phase 3: normalize trajectories

Implement normalize.py.

The input trajectory may be a JSON string or a list. Parse both. The output should be one normalized JSONL file:

data/interim/normalized_nebius_sample.jsonl
data/interim/normalized_nebius_full_part_000.parquet
...

For each raw trajectory step, extract:

role
model response
action
observation
command
file paths
action type
basic test result signals

SWE-agent-style trajectories usually contain thought/action/observation turns. SWE-agent documentation describes .traj files as JSON containing thought, action, observation, state, and query fields under the trajectory key.  ￼ mini-SWE-agent also provides .traj.json files that show the history of a run.  ￼

Action parser rules

In action_parser.py, implement deterministic regex-based classification.

Classify commands into:

read_file
search
edit
test
git
install
submit
environment
other

Examples:

READ_PATTERNS = [
    r"\bcat\b",
    r"\bsed\s+-n\b",
    r"\bnl\s+-ba\b",
    r"\bhead\b",
    r"\btail\b",
    r"\bless\b",
]
SEARCH_PATTERNS = [
    r"\brg\b",
    r"\bgrep\b",
    r"\bfind\b",
]
EDIT_PATTERNS = [
    r"\bapply_patch\b",
    r"cat\s+<<.*>\s*[\w./-]+",
    r"\bsed\s+-i\b",
    r"\bpython\b.*open\(.*['\"]w",
]
TEST_PATTERNS = [
    r"\bpytest\b",
    r"\bunittest\b",
    r"\btox\b",
    r"\bnox\b",
    r"\bpython\s+-m\s+pytest\b",
]
GIT_PATTERNS = [
    r"\bgit\s+diff\b",
    r"\bgit\s+status\b",
    r"\bgit\s+checkout\b",
    r"\bgit\s+reset\b",
]
SUBMIT_PATTERNS = [
    r"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    r"\bsubmit\b",
]

Extract file paths with a conservative regex:

FILE_RE = r"(?P<path>(?:[\w.-]+/)*[\w.-]+\.(?:py|pyi|js|ts|tsx|jsx|java|go|rs|cpp|c|h|hpp|rb|php|sh|yaml|yml|toml|json|md|rst|txt))"

Parse test outputs:

"1 failed, 4 passed"
"FAILED tests/test_x.py::test_y"
"ERROR"
"Traceback"
"AssertionError"

Verification gate:

PASS if action_parser unit tests cover at least:
- pytest command
- rg command
- cat/sed read command
- apply_patch edit
- git diff
- submit command
- traceback in observation

⸻

Phase 4: build prefix dataset

For each normalized trajectory, create many prefix examples.

Example:

trajectory has 40 steps
create prefixes at:
1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40

Do not create every possible prefix at full scale initially because 80,036 trajectories can create millions of rows. Use a configurable schedule.

Recommended prefix schedule:

def prefix_schedule(n_steps: int) -> list[int]:
    steps = set()
    for i in range(1, min(n_steps, 5) + 1):
        steps.add(i)
    for i in range(10, n_steps + 1, 5):
        steps.add(i)
    steps.add(n_steps)
    return sorted(steps)

Label:

y_fail = int(not trajectory.target)

Important: this is a terminal outcome label, not a perfect “this exact prefix is wrong” label. The paper must be honest about that.

Output:

data/processed/prefix_sample.parquet
data/processed/prefix_full_part_000.parquet

Verification gate:

PASS if no PrefixExample contains generated_patch, eval_logs, exit_status, or target inside feature_dict.
PASS if y_fail is present only as the label.
PASS if prefix_step <= n_total_steps for every row.

⸻

Phase 5: extract features

Implement features.py.

The feature extractor takes only steps 0..prefix_step.

Feature family A: trajectory length and pace

prefix_step
n_actions_seen
fraction_of_total_steps_seen    # only for offline analysis; do not use in deployment model
n_model_tokens_approx
n_observation_chars
avg_observation_chars

Important: fraction_of_total_steps_seen uses future knowledge because total length is unknown online. It may be used only in diagnostic plots, not in deployable models.

Feature family B: action counts

n_read
n_search
n_edit
n_test
n_git
n_install
n_submit
n_other
edit_to_read_ratio
test_to_edit_ratio
search_to_edit_ratio

Feature family C: context-before-edit

first_edit_step
n_reads_before_first_edit
n_searches_before_first_edit
edited_before_any_read
edited_before_any_search

Feature family D: file behavior

n_unique_files_seen
n_unique_files_read
n_unique_files_edited
n_unique_dirs_edited
n_test_files_touched
n_src_files_touched
edited_file_never_read_count
same_file_edit_count_max

Feature family E: testing behavior

n_test_runs
last_test_returncode
last_test_fail_count
last_test_pass_count
test_fail_count_delta
tests_improving
tests_worsening
same_test_command_repeated
n_tracebacks_seen
n_assertion_errors_seen

Feature family F: loop behavior

repeated_exact_command_last_3
repeated_exact_command_last_5
max_command_repeat_count
same_action_type_streak
edit_test_edit_test_loop_count
read_same_file_repeatedly

Feature family G: patch/churn behavior

For offline public trajectories, exact per-step patch may not always be available. For online mini-SWE-agent runs, collect this by running monitor-side commands after each step:

git diff --numstat
git diff --shortstat
git status --short

Features:

diff_files_changed
diff_lines_added
diff_lines_deleted
working_tree_dirty
patch_growth_since_last_step
patch_growth_without_test_improvement

Feature family H: text features

Use CPU-friendly text features:

TF-IDF over action_text + short observation excerpts
HashingVectorizer as memory-safe option

Do not embed everything with a large LLM. Keep this cheap.

Verification gate:

PASS if feature extraction works on 1000 trajectories in sample mode.
PASS if deployable feature set excludes total future length.
PASS if feature matrix has no string leakage from labels or eval logs.

⸻

Phase 6: split data correctly

Implement split.py.

Never randomly split prefix rows. That leaks because multiple prefixes from the same trajectory or task would appear in both train and test.

Use group-based splits:

primary split: group by instance_id
secondary split: group by repository name parsed from instance_id
stress split: hold out model_name

Example:

from sklearn.model_selection import GroupShuffleSplit
groups = df["instance_id"]
splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, df["y_fail"], groups))

Then split train into train/validation, again grouped by instance_id.

Verification gate:

PASS if set(train.instance_id) ∩ set(val.instance_id) = empty
PASS if set(train.instance_id) ∩ set(test.instance_id) = empty
PASS if set(val.instance_id) ∩ set(test.instance_id) = empty

⸻

Phase 7: train offline monitors

Implement train.py.

Start with these models:

baseline_majority
baseline_step_count_only
heuristic_rule_monitor
logistic_regression
random_forest
hist_gradient_boosting
structured_plus_tfidf_logistic

Do not start with deep learning. On an M4 MacBook, classical models are easier to train, easier to debug, and more publishable if they work surprisingly well.

Baselines

Majority baseline

Always predict the training failure rate.

Step-count-only baseline

Use only:

prefix_step
n_actions_seen

This checks whether your model is just learning that long trajectories fail.

Heuristic baseline

Create a simple score:

risk = 0
if edited_before_any_read: risk += 1
if repeated_exact_command_last_3: risk += 1
if patch_growth_without_test_improvement: risk += 1
if n_test_runs >= 2 and tests_worsening: risk += 1
if edited_file_never_read_count > 0: risk += 1

Logistic regression

Good for interpretability.

HistGradientBoostingClassifier

Good for tabular structured features without external dependencies.

Structured + TF-IDF

Use:

ColumnTransformer([
  ("num", StandardScaler(), numeric_cols),
  ("text", TfidfVectorizer(max_features=50000, ngram_range=(1,2)), text_col),
])

Verification gate:

PASS if shuffled-label AUC is near 0.5.
PASS if real-label AUC beats majority and step-count-only baselines.
PASS if model can save/load with joblib.
PASS if training completes on sample mode before full mode.

⸻

Phase 8: evaluate offline prediction

Implement evaluate.py.

Metrics:

ROC AUC
AUPRC
Brier score
Expected calibration error
precision at fixed false alarm rate
recall at fixed false alarm rate
median warning lead time
false alarm rate on successful trajectories

Use failure as the positive class:

y_fail = 1

First-alert evaluation

For each trajectory:

compute risk at each prefix
find first prefix where risk >= threshold

For failed trajectories:

lead_steps = n_total_steps - first_alarm_step

For successful trajectories:

false_alarm = any risk >= threshold

Report:

threshold
failed trajectories alerted
successful trajectories falsely alerted
median lead_steps
median lead_fraction

Select thresholds on validation set only.

Recommended threshold policies:

T1: maximize F1
T2: success false-alarm rate <= 20%
T3: success false-alarm rate <= 10%
T4: success false-alarm rate <= 5%

For this research, T3 and T4 matter most because unnecessary interventions are risky.

Verification gate:

PASS if threshold is chosen on validation only.
PASS if final test metrics are computed once after threshold selection.
PASS if confidence intervals are bootstrapped by instance_id, not prefix row.

⸻

Phase 9: calibration

Implement calibrate.py.

Use validation predictions to calibrate risk scores:

Platt scaling
isotonic regression

Metrics:

Brier score
ECE
calibration plot

Why this matters:

If the model says:

risk = 0.80

then approximately 80% of such prefixes should eventually fail. Calibration matters because the intervention policy needs a meaningful probability, not just a ranking.

Verification gate:

PASS if calibrated model improves or preserves Brier score on validation.
PASS if calibration is never fit on test data.

⸻

Phase 10: local LLM judge baseline through Ollama

Implement ollama_judge.py.

This is not the main model. It is a comparison baseline.

Use Ollama structured outputs so the local model returns valid JSON. Ollama supports structured JSON outputs via a format field and recommends Pydantic/Zod schemas plus low temperature for more deterministic results.  ￼

Create schema:

from pydantic import BaseModel, Field
class RiskJudgment(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
    likely_failure_modes: list[str]
    should_intervene: bool
    intervention_type: str
    evidence: list[str]

Prompt:

You are judging a coding-agent trajectory prefix.
You see only the prefix so far. You do not know the final answer.
Estimate whether the agent is drifting toward failure.
Return JSON matching the schema:
- risk_score: 0.0 means very healthy, 1.0 means very likely to fail
- likely_failure_modes: short list
- should_intervene: true/false
- intervention_type: one of none, loop_guard, evidence_gate, rollback_suggest
- evidence: concrete observations from the prefix

Run this only on a subset:

python scripts/run_ollama_judge_subset.py --n 1000 --model qwen2.5-coder:7b

Compare:

local LLM judge AUC/AUPRC
structured classifier AUC/AUPRC
runtime per prefix
JSON validity rate

Verification gate:

PASS if invalid JSON rate is reported.
PASS if local LLM judge is evaluated on the same held-out subset as the classifier.
PASS if runtime is reported.

⸻

Phase 11: implement online monitor

Implement monitor.py.

The monitor receives a trajectory prefix and returns:

class MonitorVerdict(BaseModel):
    risk_score: float
    calibrated_risk: float
    alarm: bool
    recommended_intervention: str
    evidence: dict

Basic policy:

def should_alarm(risk, step, last_alarm_step, n_interventions, config):
    if step < config.min_step:
        return False
    if n_interventions >= config.max_interventions:
        return False
    if last_alarm_step is not None and step - last_alarm_step < config.cooldown_steps:
        return False
    return risk >= config.threshold

Default safety settings:

min_step: 5
cooldown_steps: 5
max_interventions: 2
threshold_policy: "val_success_false_alarm_lte_10pct"

Do not intervene at step 0 or step 1. The Intervention Paradox paper reports that early interventions can be especially harmful because they disrupt already-correct trajectories.  ￼

Verification gate:

PASS if monitor can run in shadow mode without changing agent behavior.
PASS if every verdict is logged.
PASS if the policy refuses to intervene before min_step.

⸻

Phase 12: implement interventions

Implement interventions.py.

Start with three interventions.

Intervention A: loop guard

Trigger when:

risk high
AND repeated command/edit/test loop detected

Message to agent:

The monitor detected a possible loop: you appear to be repeating similar actions without new evidence.
Before making another edit:
1. Summarize the exact failing behavior.
2. State what evidence supports your current hypothesis.
3. Identify one new file, test, or error message to inspect.
4. Then run one targeted command.

This is low-disruption because it does not rollback or force a new plan.

Intervention B: evidence gate

Trigger when:

risk high
AND agent edited before reading/searching enough context

Message:

The monitor detected that edits may be happening before enough evidence has been gathered.
Pause editing. First inspect the relevant implementation and test files. Then state:
1. the likely root cause,
2. the smallest code region involved,
3. the test or reproduction command that will verify the fix.

Intervention C: rollback suggestion

Trigger when:

risk very high
AND recent edit increased failing tests or patch churn
AND git checkpoint exists

Message:

The monitor detected that the latest edit may have worsened the trajectory.
Consider rolling back to the previous checkpoint and trying a smaller patch. Before continuing:
1. inspect the latest test failure,
2. compare it to the previous failure,
3. choose whether to revert the last edit.

For the first version, do suggested rollback, not forced rollback. Forced rollback can be tested later.

Verification gate:

PASS if each intervention has:
- trigger condition
- injected message
- cooldown behavior
- logging
- unit test

⸻

Phase 13: integrate with mini-SWE-agent

Use mini-SWE-agent for online local experiments because it is simple, hackable, and supports local models through LiteLLM.  ￼

Create a mini-SWE-agent config for Ollama.

Example config file:

# configs/mini_swe_ollama_qwen25_7b.yaml
agent:
  step_limit: 40
  cost_limit: 0
  mode: yolo
model:
  model_name: "ollama_chat/qwen2.5-coder:7b"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://localhost:11434"
    temperature: 0
    num_ctx: 32768

If ollama_chat/... does not work in the installed LiteLLM version, try:

model:
  model_name: "qwen2.5-coder:7b"
  cost_tracking: "ignore_errors"
  model_kwargs:
    custom_llm_provider: "ollama"
    api_base: "http://localhost:11434"
    temperature: 0

LiteLLM’s Ollama documentation recommends ollama_chat for better responses and says to ensure the Ollama server is running.  ￼

Wrapper design

Do not deeply rewrite mini-SWE-agent at first. Create a wrapper that:

1. launches or calls mini-SWE-agent
2. captures .traj.json output
3. replays trajectory prefixes through the monitor
4. runs in shadow mode first

Then implement an instrumented fork only after shadow mode works.

Shadow mode:

agent runs normally
monitor logs what it would have done
agent never sees monitor messages

Intervention mode:

after each step:
  parse latest prefix
  compute risk
  if policy says alarm:
      inject intervention message into conversation
      continue agent loop

Verification gate:

PASS if baseline and shadow mode produce identical agent behavior when temperature=0.
PASS if intervention mode logs injected messages.
PASS if no paid API is called.

⸻

Phase 14: choose online evaluation tasks

Use a small local evaluation first.

Primary target:

SWE-bench Verified Mini

It contains 50 tasks and is described as a lightweight random subset of SWE-bench Verified.  ￼

Be careful: official SWE-bench evaluation is resource-intensive. The SWE-bench repo recommends an x86_64 machine with at least 120GB free storage, 16GB RAM, and 8 CPU cores.  ￼ Since this is an Apple Silicon Mac, begin with tiny smoke tests and selected tasks.

Optional evaluator:

swe-bench-fast

It is an unofficial Go evaluation harness that scores precomputed predictions, supports native ARM64 images for many instances, and says it does not generate predictions itself.  ￼ Use it only as an optional local speedup path, and clearly label it as unofficial if used in the paper.

Online experiment ladder:

Level 0: one trivial local toy repo
Level 1: 3 handpicked small Python bug-fix tasks
Level 2: 5 SWE-bench Verified Mini tasks
Level 3: 20 SWE-bench Verified Mini tasks
Level 4: all 50 SWE-bench Verified Mini tasks

Do not jump straight to all 50.

Verification gate:

PASS if one task can be run baseline, shadow, and intervention mode.
PASS if generated patches are saved.
PASS if final evaluation can classify success/failure for the task.

⸻

Phase 15: online experiment design

For each task, run:

baseline: mini-SWE-agent + qwen2.5-coder:7b
shadow: baseline + monitor logging only
loop_guard: monitor + loop intervention
evidence_gate: monitor + evidence intervention
rollback_suggest: monitor + rollback suggestion

Optional second model:

qwen2.5-coder:14b

Use temperature 0 first.

For each run, log:

task_id
model
policy
success
n_steps
n_interventions
first_alarm_step
total_tokens_approx
diff_files_changed
diff_lines_added
diff_lines_deleted
test_runs
final_patch
trajectory_path

Compute:

baseline_success_rate
policy_success_rate
average_steps
average_tokens
recovery_count
disruption_count
unchanged_success_count
unchanged_failure_count

Definitions:

recovery:
  baseline failed, intervention succeeded
disruption:
  baseline succeeded, intervention failed
safe saving:
  same success/failure result, fewer steps/tokens
harmful waste:
  same success/failure result, more steps/tokens

Verification gate:

PASS if every intervention run can be paired with a baseline run on the same task and model.
PASS if recoveries and disruptions are reported separately.
PASS if no claim is made based only on AUC.

⸻

Phase 16: expected result tables

Generate these tables automatically.

Table 1: offline monitor performance

model
features
split
ROC_AUC
AUPRC
Brier
ECE
false_alarm_success_10pct_threshold
failed_coverage
median_lead_steps

Table 2: feature ablation

feature_set
ROC_AUC
AUPRC
false_alarm_rate
median_lead_steps

Feature sets:

step_count_only
action_counts
context_before_edit
testing_behavior
loop_behavior
all_structured
all_structured_plus_text

Table 3: threshold tradeoff

threshold_policy
threshold
success_false_alarm_rate
failed_alert_rate
median_lead_steps

Table 4: online intervention results

policy
success_rate
avg_steps
avg_tokens
recoveries
disruptions
interventions_per_run

Table 5: local cost/resource table

model
ollama_tag
model_size
context_window
avg_tokens_per_sec
avg_run_time
success_rate

Do not overclaim from small online samples. Report confidence intervals.

Verification gate:

PASS if all tables are generated from saved JSONL/CSV result files.
PASS if each table has a script that reproduces it.

⸻

Phase 17: expected figures

Generate these figures:

Figure 1: risk over trajectory time

Plot average calibrated risk by normalized prefix position for:

successful trajectories
failed trajectories

Figure 2: precision-recall curve

Compare:

heuristic
logistic regression
gradient boosting
structured + text
Ollama judge subset

Figure 3: warning lead time

Histogram:

first alarm lead steps for failed trajectories

Figure 4: intervention accounting

Bar chart:

recoveries
disruptions
safe savings
harmful waste

Figure 5: feature importance

For logistic regression:

top positive risk features
top negative risk features

Verification gate:

PASS if figures can be regenerated by `python scripts/make_report.py`.

⸻

Phase 18: qualitative audit

Build a small terminal audit tool:

python scripts/audit_prefixes.py --kind false_positive --n 50
python scripts/audit_prefixes.py --kind true_positive --n 50
python scripts/audit_prefixes.py --kind false_negative --n 50

For each prefix, show:

instance_id
model_name
target
prefix_step
risk_score
last 5 actions
last 3 observations shortened
features that contributed most

Create manual labels:

insufficient_context
wrong_file
test_neglect
looping
patch_churn
environment_distraction
tool_format_issue
submission_too_early
not_observable

This helps answer:

Which failures are actually observable early?

That is publishable even if intervention results are mixed.

Verification gate:

PASS if at least 100 high-risk prefixes are manually or semi-manually audited.
PASS if the paper includes examples of true positives, false positives, and false negatives.

⸻

Phase 19: no-leakage tests

Implement strict tests.

test_no_leakage.py should fail if feature names or feature values include:

target
eval_logs
generated_patch
exit_status
resolved
passed_final
fail_to_pass final result

Also add a shuffle test:

Train on shuffled labels.
Expected AUC: around 0.5.
If much higher, there is leakage.

Add a future-feature test:

For a prefix at step t, changing steps after t must not change extracted features.

Verification gate:

PASS if future steps do not affect prefix features.
PASS if shuffled-label model is near random.

⸻

Phase 20: final paper framing

The paper should not claim:

We are the first to predict agent failures.

That is not defensible because PrefixGuard and AgentForesight-like work already study online failure prediction.  ￼

The paper should claim:

We study local, open-source, disruption-aware monitoring for coding agents under consumer-hardware constraints.

Better title:

LocalGuard-SWE: Disruption-Aware Failure Monitoring for Local Open-Source Coding Agents

Possible abstract:

LLM coding agents often fail after long trajectories of tool calls, edits, and tests. Prior work shows that trajectory prefixes can predict failures, but accurate prediction alone does not guarantee useful intervention. We study failure monitoring for open-source coding agents under local-compute constraints. Using public SWE-agent trajectories, we train lightweight monitors from prefix-visible structured features and compare them against heuristic and local-LLM judges. We then integrate the monitor into a local Ollama + mini-SWE-agent loop and evaluate interventions with recovery–disruption accounting. Our results show when coding-agent failures are observable early, which signals matter, and when intervention improves reliability or merely disrupts otherwise successful runs.

⸻

Main commands the coding agent should implement

# Setup
make setup
make pull-models
make system-check
# Data
make data-sample
make inspect-schema
make normalize-sample
make prefixes-sample
# Offline models
make train-small
make eval-small
make train-full
make eval-full
# Local LLM judge
make ollama-judge-small
# Online experiments
make mini-smoke
make online-shadow-small
make online-loop-guard-small
make online-evidence-gate-small
make online-rollback-small
# Reports
make tables
make figures
make report
make test

Example Makefile targets:

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install -e ".[dev]"
pull-models:
	ollama pull qwen2.5-coder:7b
	ollama pull qwen2.5-coder:14b
system-check:
	. .venv/bin/activate && python scripts/system_check.py
data-sample:
	. .venv/bin/activate && python scripts/download_data.py --mode sample --n 1000
normalize-sample:
	. .venv/bin/activate && python scripts/build_prefix_dataset.py --input sample --output data/processed/prefix_sample.parquet
train-small:
	. .venv/bin/activate && python scripts/train_monitor.py --config configs/offline_small.yaml
eval-small:
	. .venv/bin/activate && python scripts/evaluate_monitor.py --config configs/offline_small.yaml
test:
	. .venv/bin/activate && pytest -q

⸻

Minimum viable research result

The smallest complete version of the project is:

1. Parse Nebius trajectories.
2. Build prefix datasets.
3. Train structured monitor.
4. Evaluate early-warning quality.
5. Compare against heuristics and local Ollama judge on a subset.
6. Run mini-SWE-agent shadow mode on a few local tasks.
7. Run one low-disruption intervention.
8. Report recovery/disruption accounting.

This is enough to produce a serious workshop paper or preprint if done carefully.

⸻

Stronger publishable version

The stronger version adds:

1. Full offline evaluation on all 80,036 trajectories.
2. Grouped splits by instance, repo, and model.
3. Calibration and threshold analysis.
4. 50-task SWE-bench Verified Mini online evaluation.
5. Multiple local models: qwen2.5-coder:7b and qwen2.5-coder:14b.
6. Three interventions: loop guard, evidence gate, rollback suggestion.
7. Manual qualitative audit of failure observability.
8. Open-source reproducibility package.

That version has a much clearer publication story.

⸻

The core verification checklist

Keep this checklist in paper/experiment_log.md and update it after each phase.

[ ] Ollama local endpoint works.
[ ] qwen2.5-coder:7b works locally.
[ ] Dataset schema inspected and saved.
[ ] 1000 trajectories normalized.
[ ] Prefix dataset created with no future leakage.
[ ] Group split verified by instance_id.
[ ] Shuffled-label AUC near 0.5.
[ ] Step-count-only baseline implemented.
[ ] Heuristic baseline implemented.
[ ] Structured classifier implemented.
[ ] Structured + text classifier implemented.
[ ] Calibration implemented.
[ ] First-alert metrics implemented.
[ ] False alarms on successes reported.
[ ] Lead time on failures reported.
[ ] Ollama judge subset evaluated.
[ ] mini-SWE-agent baseline runs locally.
[ ] Shadow monitor runs without behavior changes.
[ ] Loop guard intervention implemented.
[ ] Evidence gate intervention implemented.
[ ] Rollback suggestion implemented.
[ ] Recoveries and disruptions reported separately.
[ ] Tables generated from raw result files.
[ ] Figures generated from raw result files.
[ ] No paid API calls used.

⸻

The key design decision

The most important thing is this:

Do not optimize only for AUC.
Optimize for safe intervention.

The actual publishable contribution is not:

A classifier predicts failure.

It is:

A local, lightweight monitor identifies actionable coding-agent drift and shows when intervention helps, when it harms, and which signals are reliable under open-source local-agent constraints.

That framing is much harder for reviewers to dismiss.