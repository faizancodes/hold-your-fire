"""Frontier LLM judge via OpenRouter (e.g. Anthropic Claude Opus 4.8) — cross-provider baseline.

Same prompt / schema / prefix renderer as the local Ollama and OpenAI judges; only the
model and provider change, so the comparison stays apples-to-apples. Uses the OpenAI-
compatible OpenRouter endpoint. Non-OpenAI providers honor structured outputs differently,
so we request JSON via response_format with a layered fallback (json_schema -> json_object
-> plain prompt) and parse the content tolerantly (the prompt already specifies the schema).

Paid API. Gated behind the same dry-run cost check as the OpenAI judge; the deployed
monitor stays 100% local — this only buys another baseline to compare against.
"""

from __future__ import annotations

import re
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-opus-4.8"

_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


def _parse_risk(content: str) -> RiskJudgment | None:
    """Tolerant parse: strip markdown fences, isolate the first {...}, validate."""
    s = _FENCE.sub("", content or "").replace("```", "").strip()
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        s = s[i : j + 1]
    try:
        return RiskJudgment.model_validate_json(s)
    except Exception:
        return None


def _reason_toks(usage: Any) -> int:
    d = getattr(usage, "completion_tokens_details", None)
    return (getattr(d, "reasoning_tokens", 0) or 0) if d else 0


def _or_cost(usage: Any) -> float | None:
    c = getattr(usage, "cost", None)
    if c is None and getattr(usage, "model_extra", None):
        c = usage.model_extra.get("cost")
    return c


def judge_prefix_openrouter(
    steps: Sequence[StepEvent],
    client: Any,
    model: str = DEFAULT_OPENROUTER_MODEL,
    max_steps: int = 12,
    max_tokens: int = 1500,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    timeout_impute: float = 0.5,
    retries: int = 2,
) -> JudgeOutput:
    """One structured risk judgment on a prefix via OpenRouter (OpenAI-compatible API).

    ``reasoning_effort`` (None | "low" | "high"): when set, enable Claude extended
    thinking via OpenRouter's ``reasoning`` param. Anthropic thinking requires
    temperature=1, so it overrides ``temperature``; set a generous ``max_tokens`` so the
    thinking trace + JSON answer fit.
    """
    transcript = render_prefix(steps, max_steps=max_steps)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(n_steps=len(steps), transcript=transcript)},
    ]
    schema_fmt = {"type": "json_schema", "json_schema": {
        "name": "risk_judgment", "strict": True, "schema": RiskJudgment.model_json_schema()}}
    formats: list[dict | None] = [schema_fmt, {"type": "json_object"}, None]
    use_temp = 1.0 if reasoning_effort else temperature  # Claude thinking requires temp=1

    last_err = ""
    for fmt in formats:
        fmt_name = fmt["type"] if fmt else "plain"
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                body_extra: dict[str, Any] = {"usage": {"include": True}}
                if reasoning_effort:
                    body_extra["reasoning"] = {"effort": reasoning_effort}
                kw: dict[str, Any] = dict(
                    model=model, messages=messages, max_tokens=max_tokens,
                    temperature=use_temp, extra_body=body_extra)
                if fmt is not None:
                    kw["response_format"] = fmt
                resp = client.chat.completions.create(**kw)
                latency = time.time() - t0
                content = resp.choices[0].message.content or ""
                usage = resp.usage
                extra = {
                    "resolved_model": getattr(resp, "model", model),
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "reasoning_tokens": _reason_toks(usage),
                    "or_cost_usd": _or_cost(usage),
                    "format_used": fmt_name,
                    "reasoning_effort": reasoning_effort,
                    "finish_reason": resp.choices[0].finish_reason,
                }
                judgment = _parse_risk(content)
                if judgment is None:  # honored but unparseable -> try the next format
                    last_err = f"unparseable via {fmt_name}: {content[:120]}"
                    break
                risk = float(min(1.0, max(0.0, judgment.risk_score)))
                if judgment.intervention_type not in INTERVENTION_TYPES:
                    judgment.intervention_type = "none"
                return JudgeOutput(judgment=judgment, risk_score=risk, valid_json=True,
                                   latency_s=latency, raw=judgment.model_dump_json(), extra=extra)
            except Exception as exc:
                last_err = str(exc)[:300]
                msg = str(exc).lower()
                if any(t in msg for t in ("response_format", "json_schema", "json_object", "not support")):
                    break  # this format is unsupported -> fall through to the next one
                if attempt < retries:  # transient (rate limit / timeout / 5xx)
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
    return JudgeOutput(judgment=None, risk_score=timeout_impute, valid_json=False,
                       latency_s=0.0, error=last_err)
