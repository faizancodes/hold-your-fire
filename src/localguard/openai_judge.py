"""Frontier LLM judge baseline via the OpenAI API (paper strengthening).

A STRONGER version of the local Ollama judge: same prompt, same structured schema,
same prefix rendering, same imputation/metrics — only the backend model changes, so
the comparison against the cheap structured classifier is apples-to-apples and the
ONLY difference is the judge model (GPT-5.5 vs the local qwen2.5-coder:7b).

Unlike the rest of the project this uses a PAID API; gated behind an explicit runner
with a dry-run cost check. The deployed monitor remains 100% local — this only buys a
stronger *baseline* to compare against.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from .ollama_judge import (  # reuse verbatim — keeps the comparison fair
    INTERVENTION_TYPES,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    JudgeOutput,
    RiskJudgment,
    render_prefix,
)
from .schemas import StepEvent

DEFAULT_OPENAI_MODEL = "gpt-5.5"


def judge_prefix_openai(
    steps: Sequence[StepEvent],
    client: Any,
    model: str = DEFAULT_OPENAI_MODEL,
    max_steps: int = 12,
    max_completion_tokens: int = 2000,
    reasoning_effort: str | None = "low",
    timeout_impute: float = 0.5,
    retries: int = 2,
) -> JudgeOutput:
    """One structured risk judgment on a prefix via the OpenAI Chat Completions API.

    Mirrors ``ollama_judge.judge_prefix`` exactly except for the backend. GPT-5-class
    (reasoning) models: no ``temperature`` (they reject != 1), ``max_completion_tokens``
    (not ``max_tokens``), and reasoning tokens count toward that budget — so it is set
    generously and reasoning effort kept low to bound cost/latency.
    """
    transcript = render_prefix(steps, max_steps=max_steps)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(n_steps=len(steps), transcript=transcript)},
    ]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": RiskJudgment,
        "max_completion_tokens": max_completion_tokens,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    last_err = ""
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            completion = client.chat.completions.parse(**kwargs)
            latency = time.time() - t0
            msg = completion.choices[0].message
            usage = completion.usage
            rdet = getattr(usage, "completion_tokens_details", None)
            extra = {
                "resolved_model": completion.model,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "reasoning_tokens": getattr(rdet, "reasoning_tokens", 0) or 0,
                "finish_reason": completion.choices[0].finish_reason,
                "reasoning_effort": reasoning_effort,
            }
            judgment = msg.parsed
            if judgment is None:  # refusal or truncated before producing JSON
                return JudgeOutput(judgment=None, risk_score=timeout_impute, valid_json=False,
                                   latency_s=latency, raw=(msg.refusal or "")[:300],
                                   error="no_parsed_output", extra=extra)
            risk = float(min(1.0, max(0.0, judgment.risk_score)))
            if judgment.intervention_type not in INTERVENTION_TYPES:
                judgment.intervention_type = "none"
            return JudgeOutput(judgment=judgment, risk_score=risk, valid_json=True,
                               latency_s=latency, raw=judgment.model_dump_json(), extra=extra)
        except TypeError as exc:
            # SDK/model rejected a kwarg (e.g. reasoning_effort unsupported) — drop it once
            if "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort", None)
                last_err = f"retry_without_reasoning_effort: {exc}"[:200]
                continue
            last_err = str(exc)[:300]
            break
        except Exception as exc:  # transient (rate limit / timeout / 5xx) — backoff + retry
            last_err = str(exc)[:300]
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    return JudgeOutput(judgment=None, risk_score=timeout_impute, valid_json=False,
                       latency_s=0.0, error=last_err)
