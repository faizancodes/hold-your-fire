"""OpenAI (GPT-5.5) judge backend — logic tested with a MOCK client (no API spend)."""

from types import SimpleNamespace

from localguard.ollama_judge import RiskJudgment
from localguard.openai_judge import judge_prefix_openai
from localguard.schemas import StepEvent


def _usage(p=120, c=20, r=5):
    return SimpleNamespace(prompt_tokens=p, completion_tokens=c,
                           completion_tokens_details=SimpleNamespace(reasoning_tokens=r))


def _completion(parsed, refusal=None, finish="stop", model="gpt-5.5-2026-04-23"):
    msg = SimpleNamespace(parsed=parsed, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)],
                           usage=_usage(), model=model)


class _Completions:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []

    def parse(self, **kw):
        self.calls.append(kw)
        if self.behavior == "reject_effort" and "reasoning_effort" in kw:
            raise TypeError("unexpected keyword argument 'reasoning_effort'")
        if self.behavior == "invalid":
            return _completion(None, refusal="cannot")
        j = RiskJudgment(risk_score=0.83, should_intervene=True, intervention_type="loop_guard")
        return _completion(j)


class _Client:
    def __init__(self, behavior="valid"):
        self.chat = SimpleNamespace(completions=_Completions(behavior))


STEPS = [StepEvent(trajectory_id="t1", instance_id="i1", step_index=k,
                   action_type="edit", action_text="x", observation_text="err") for k in range(6)]


def test_valid_judgment_parses_and_reports_tokens():
    out = judge_prefix_openai(STEPS, _Client("valid"), model="gpt-5.5")
    assert out.valid_json and out.judgment is not None
    assert abs(out.risk_score - 0.83) < 1e-9
    assert out.extra["prompt_tokens"] == 120 and out.extra["reasoning_tokens"] == 5
    assert out.extra["resolved_model"] == "gpt-5.5-2026-04-23"


def test_none_parsed_is_invalid_and_imputed():
    out = judge_prefix_openai(STEPS, _Client("invalid"), timeout_impute=0.5)
    assert not out.valid_json and out.judgment is None
    assert out.risk_score == 0.5  # imputed for paired comparison


def test_reasoning_effort_unsupported_falls_back_then_succeeds():
    client = _Client("reject_effort")
    out = judge_prefix_openai(STEPS, client, reasoning_effort="low")
    assert out.valid_json  # succeeded on the retry without reasoning_effort
    calls = client.chat.completions.calls
    assert "reasoning_effort" in calls[0] and "reasoning_effort" not in calls[-1]


def test_uses_completion_token_param_not_max_tokens():
    client = _Client("valid")
    judge_prefix_openai(STEPS, client, max_completion_tokens=1234)
    kw = client.chat.completions.calls[0]
    assert kw["max_completion_tokens"] == 1234 and "max_tokens" not in kw
    assert "temperature" not in kw  # reasoning models reject temperature != 1
