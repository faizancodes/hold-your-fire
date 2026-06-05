"""OpenRouter (Claude Opus 4.8) judge backend — logic tested with a MOCK client (no API spend)."""

from types import SimpleNamespace

import pytest

from localguard.openrouter_judge import _parse_risk, judge_prefix_openrouter
from localguard.schemas import StepEvent

_VALID = ('{"risk_score":0.8,"should_intervene":true,"intervention_type":"loop_guard",'
          '"likely_failure_modes":["loop"],"evidence":["repeat"]}')
STEPS = [StepEvent(trajectory_id="t", instance_id="i", step_index=k,
                   action_type="edit", action_text="x", observation_text="err") for k in range(6)]


def _resp(content, cost=0.012, model="anthropic/claude-4.8-opus-20260528"):
    usage = SimpleNamespace(prompt_tokens=110, completion_tokens=25, cost=cost,
                            completion_tokens_details=None)
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=usage, model=model)


class _Completions:
    def __init__(self, behavior, content=_VALID):
        self.behavior, self.content, self.calls = behavior, content, []

    def create(self, **kw):
        self.calls.append(kw)
        fmt = (kw.get("response_format") or {}).get("type", "plain")
        if self.behavior == "reject_schema" and fmt == "json_schema":
            raise Exception("response_format json_schema is not supported by this model")
        return _resp(self.content)


class _Client:
    def __init__(self, behavior="valid", content=_VALID):
        self.chat = SimpleNamespace(completions=_Completions(behavior, content))


# ---- tolerant parser ----
@pytest.mark.parametrize("raw", [
    _VALID,
    "```json\n" + _VALID + "\n```",
    "Here is my judgment:\n" + _VALID + "\nHope that helps.",
])
def test_parse_risk_tolerates_fences_and_prose(raw):
    j = _parse_risk(raw)
    assert j is not None and abs(j.risk_score - 0.8) < 1e-9


def test_parse_risk_returns_none_on_garbage():
    assert _parse_risk("I cannot produce JSON for this.") is None


# ---- backend ----
def test_valid_judgment_and_cost_extraction():
    out = judge_prefix_openrouter(STEPS, _Client("valid"), model="anthropic/claude-opus-4.8")
    assert out.valid_json and abs(out.risk_score - 0.8) < 1e-9
    assert out.extra["or_cost_usd"] == 0.012
    assert out.extra["resolved_model"].startswith("anthropic/claude-4.8-opus")


def test_falls_back_from_json_schema_to_json_object():
    client = _Client("reject_schema")
    out = judge_prefix_openrouter(STEPS, client)
    assert out.valid_json  # succeeded after falling back
    fmts = [(c.get("response_format") or {}).get("type", "plain") for c in client.chat.completions.calls]
    assert fmts[0] == "json_schema" and "json_object" in fmts  # tried schema first, then fell back


def test_unparseable_everywhere_is_invalid_and_imputed():
    out = judge_prefix_openrouter(STEPS, _Client("valid", content="no json here"),
                                  timeout_impute=0.5, retries=0)
    assert not out.valid_json and out.risk_score == 0.5


def test_uses_max_tokens_and_temperature_zero():
    client = _Client("valid")
    judge_prefix_openrouter(STEPS, client, max_tokens=900, temperature=0.0)
    kw = client.chat.completions.calls[0]
    assert kw["max_tokens"] == 900 and kw["temperature"] == 0.0
    assert "reasoning" not in kw["extra_body"]  # no thinking by default


def test_extended_thinking_sets_temp1_and_reasoning_param():
    client = _Client("valid")
    out = judge_prefix_openrouter(STEPS, client, reasoning_effort="high", temperature=0.0)
    kw = client.chat.completions.calls[0]
    assert kw["temperature"] == 1.0  # Claude extended thinking requires temp=1
    assert kw["extra_body"]["reasoning"] == {"effort": "high"}
    assert out.extra["reasoning_effort"] == "high"
