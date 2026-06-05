# mech_interp — localizing & steering the agent looping circuit

A mechanistic-interpretability study built on LocalGuard: *can we find, inside an
open-weight coding agent, the mechanism that makes it re-issue a command it has already
tried (a "tight loop"), and steer it out — less disruptively than an external reset?*

Plan: `../mech-inter-research-plan.md`. Running findings: `PROGRESS.md`.

## Setup
Isolated venv (reuses global torch/sklearn, shadows `huggingface_hub<1.0` for transformers 4.49):
```bash
python3 -m venv mech_interp/.venv --system-site-packages
mech_interp/.venv/bin/pip install 'huggingface_hub>=0.26,<1.0'
```
Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (white-box, HF + MPS). Run everything with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 mech_interp/.venv/bin/python -m mech_interp.<script>`.

## Components
| file | role |
|---|---|
| `model_wrapper.py` | load model; `capture_resid`, `continuation_logprob`, `steering()`/`ablate()` hooks |
| `synthetic.py` | coding-agent transcripts: loop / varied-fail / progress conditions |
| `run_mvp.py` | de-risking MVP (naive loop-vs-progress probe + steering) |
| `run_localize.py` | Phase 2/3 v1 (caught the length confound) |
| `run_localize2.py` | Phase 2/3 v2 — **exact length control** + REP & STUCK contrasts → `acts2.npz` |
| `run_steer_eval.py` | Phase 4/5 — causal: scan layers×sign, α-sweep + controls, ablation |
| `run_onpolicy.py` | Phase 6 — generate next action ±steering; loop-escape + disruption |
| `plots.py` | figures from the results JSONs |

## Key methodological point
Decodability (a probe separating loop from non-loop) is **cheap** and can reflect surface
cues (length, repeated tokens). We therefore (a) **exactly length-match** every condition by
token-padding, and (b) judge the mechanism by **causal** steering/ablation, not probe AUC.
All splits are scenario-grouped with bootstrap CIs (LocalGuard's "validation lies" discipline).
