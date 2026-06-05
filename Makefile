# LocalGuard-SWE Makefile
# All targets are local-only. No paid API calls. No GPU training required.
#
# Two ways to run:
#   1) venv (reproducible, documented):  make setup && make system-check
#   2) global interpreter with src on path (fast iteration):
#        PYTHONPATH=src python3 scripts/<script>.py ...
#
# The variables below let you switch interpreters without editing the file.
PY ?= python3
PYTHONPATH_RUN ?= PYTHONPATH=src
RUN := $(PYTHONPATH_RUN) $(PY)

OLLAMA_MODEL ?= qwen2.5-coder:7b

.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup
.PHONY: setup
setup: ## Create .venv and install the package (editable) with dev extras
	$(PY) -m venv .venv
	. .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install -e ".[dev]"

.PHONY: setup-online
setup-online: ## Install optional online (mini-SWE-agent) dependencies
	. .venv/bin/activate && pip install -e ".[dev,online]"

.PHONY: pull-models
pull-models: ## Pull recommended local Ollama coding models
	ollama pull qwen2.5-coder:7b
	ollama pull qwen2.5-coder:14b

.PHONY: system-check
system-check: ## Verify Python, RAM, disk, Ollama, models, mini-SWE-agent
	$(RUN) scripts/system_check.py

# ---------------------------------------------------------------- data
.PHONY: data-sample
data-sample: ## Download a 1000-row sample of the Nebius trajectory dataset
	$(RUN) scripts/download_data.py --dataset nebius/SWE-agent-trajectories --mode sample --n 1000

.PHONY: data-full
data-full: ## Download the full Nebius trajectory dataset (~1.1GB)
	$(RUN) scripts/download_data.py --dataset nebius/SWE-agent-trajectories --mode full

.PHONY: inspect-schema
inspect-schema: ## Inspect and save the dataset schema
	$(RUN) scripts/inspect_dataset_schema.py

.PHONY: prefixes-sample
prefixes-sample: ## Normalize + build the prefix dataset from the sample
	$(RUN) scripts/build_prefix_dataset.py --input sample --output data/processed/prefix_sample.parquet

.PHONY: prefixes-full
prefixes-full: ## Normalize + build the prefix dataset from the full corpus
	$(RUN) scripts/build_prefix_dataset.py --input full --output data/processed/prefix_full.parquet

# ---------------------------------------------------------------- offline models
.PHONY: train-small
train-small: ## Train monitors on the sample prefix dataset
	$(RUN) scripts/train_monitor.py --config configs/offline_small.yaml

.PHONY: eval-small
eval-small: ## Evaluate monitors on the sample prefix dataset
	$(RUN) scripts/evaluate_monitor.py --config configs/offline_small.yaml

.PHONY: train-full
train-full: ## Train monitors on the full prefix dataset
	$(RUN) scripts/train_monitor.py --config configs/offline_full.yaml

.PHONY: eval-full
eval-full: ## Evaluate monitors on the full prefix dataset
	$(RUN) scripts/evaluate_monitor.py --config configs/offline_full.yaml

# ---------------------------------------------------------------- local LLM judge
.PHONY: ollama-judge-small
ollama-judge-small: ## Run the local Ollama judge baseline on a subset
	$(RUN) scripts/run_ollama_judge_subset.py --n 200 --model $(OLLAMA_MODEL)

# ---------------------------------------------------------------- online experiments
.PHONY: mini-smoke
mini-smoke: ## Smoke-test the mini-SWE-agent + Ollama wrapper on a toy repo
	$(RUN) scripts/run_mini_swe_shadow.py --level 0 --config configs/mini_swe_ollama_qwen25_7b.yaml

.PHONY: online-shadow-small
online-shadow-small:
	$(RUN) scripts/run_mini_swe_shadow.py --level 1 --config configs/online_shadow.yaml

.PHONY: online-loop-guard-small
online-loop-guard-small:
	$(RUN) scripts/run_mini_swe_intervention.py --level 1 --config configs/online_loop_guard.yaml

.PHONY: online-evidence-gate-small
online-evidence-gate-small:
	$(RUN) scripts/run_mini_swe_intervention.py --level 1 --config configs/online_evidence_gate.yaml

.PHONY: online-rollback-small
online-rollback-small:
	$(RUN) scripts/run_mini_swe_intervention.py --level 1 --config configs/online_rollback.yaml

.PHONY: swe-bench-smoke
swe-bench-smoke: ## Load SWE-bench Verified Mini tasks + check heavy-path prerequisites
	$(RUN) scripts/run_swe_bench_smoke.py --n 5

# ---------------------------------------------------------------- reports
.PHONY: tables
tables: ## Regenerate result tables from saved result files
	$(RUN) scripts/make_report.py --only tables

.PHONY: figures
figures: ## Regenerate figures from saved result files
	$(RUN) scripts/make_report.py --only figures

.PHONY: report
report: ## Regenerate all tables and figures
	$(RUN) scripts/make_report.py

.PHONY: audit
audit: ## Launch the qualitative prefix audit tool
	$(RUN) scripts/audit_prefixes.py --kind true_positive --n 50

.PHONY: test
test: ## Run the unit test suite
	$(RUN) -m pytest -q

.PHONY: smoke
smoke: ## End-to-end smoke on bundled fixtures + tests (no network/Ollama needed)
	$(RUN) scripts/build_prefix_dataset.py --input fixtures --output data/processed/prefix_fixtures.parquet
	$(RUN) -m pytest -q
