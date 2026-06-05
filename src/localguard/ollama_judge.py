"""Local LLM judge baseline via Ollama structured outputs (Phase 10).

This is a COMPARISON baseline, not the main monitor. A local coding model
(default qwen2.5-coder:7b) reads a trajectory prefix and emits a structured risk
judgment. We compare its risk_score against the cheap structured classifier on the
same held-out prefixes, and we report JSON validity and per-prefix latency.

Everything runs against the local Ollama endpoint; no paid API is used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from pydantic import BaseModel, Field

from .schemas import StepEvent

DEFAULT_MODEL = "qwen2.5-coder:7b"
INTERVENTION_TYPES = ("none", "loop_guard", "evidence_gate", "rollback_suggest")


class RiskJudgment(BaseModel):
    """Structured judgment the local model is constrained to return."""

    risk_score: float = Field(ge=0.0, le=1.0)
    likely_failure_modes: list[str] = Field(default_factory=list)
    should_intervene: bool = False
    intervention_type: str = "none"
    evidence: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = (
    "You are a monitor judging a coding-agent trajectory prefix. "
    "You see only the prefix so far; you do NOT know the final answer or outcome. "
    "Estimate whether the agent is drifting toward failure."
)

USER_TEMPLATE = """Below is the prefix of a coding agent's run so far ({n_steps} steps shown).

{transcript}

Return JSON matching the schema (BE TERSE — at most 2 items per list, <= 10 words each):
- risk_score: 0.0 = very healthy / on-track, 1.0 = very likely to fail
- likely_failure_modes: <= 2 short failure modes you observe
- should_intervene: true only if an intervention would likely help more than hurt
- intervention_type: one of none, loop_guard, evidence_gate, rollback_suggest
- evidence: <= 2 concrete observations from the prefix
Do not assume failure just because the run is long. Judge only from the evidence."""


def render_prefix(steps: Sequence[StepEvent], max_steps: int = 12, obs_chars: int = 300) -> str:
    """Render the most recent ``max_steps`` steps as a compact transcript."""
    shown = steps[-max_steps:]
    start = len(steps) - len(shown)
    lines: list[str] = []
    for i, s in enumerate(shown):
        idx = start + i + 1
        action = (s.action_text or "").strip().replace("\n", " ")[:240]
        obs = (s.observation_text or "").strip()
        obs_excerpt = obs[:obs_chars].replace("\n", " ")
        lines.append(f"[step {idx}] ({s.action_type}) $ {action}")
        if obs_excerpt:
            lines.append(f"    -> {obs_excerpt}")
    return "\n".join(lines)


@dataclass
class JudgeOutput:
    judgment: RiskJudgment | None
    risk_score: float          # imputed to 0.5 when invalid, for paired comparison
    valid_json: bool
    latency_s: float
    raw: str = ""
    error: str = ""
    extra: dict = field(default_factory=dict)


def judge_prefix(
    steps: Sequence[StepEvent],
    model: str = DEFAULT_MODEL,
    num_ctx: int = 8192,
    max_steps: int = 12,
    timeout_impute: float = 0.5,
) -> JudgeOutput:
    """Run one structured risk judgment on a prefix via local Ollama."""
    import ollama

    transcript = render_prefix(steps, max_steps=max_steps)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(n_steps=len(steps), transcript=transcript)},
    ]
    t0 = time.time()
    try:
        resp = ollama.chat(
            model=model,
            messages=messages,
            format=RiskJudgment.model_json_schema(),
            options={"temperature": 0, "num_ctx": num_ctx, "num_predict": 220},
        )
        latency = time.time() - t0
        raw = resp["message"]["content"]
        judgment = RiskJudgment.model_validate_json(raw)
        risk = float(min(1.0, max(0.0, judgment.risk_score)))
        if judgment.intervention_type not in INTERVENTION_TYPES:
            judgment.intervention_type = "none"
        return JudgeOutput(judgment=judgment, risk_score=risk, valid_json=True,
                           latency_s=latency, raw=raw)
    except Exception as exc:
        return JudgeOutput(judgment=None, risk_score=timeout_impute, valid_json=False,
                           latency_s=time.time() - t0, error=str(exc)[:300])
