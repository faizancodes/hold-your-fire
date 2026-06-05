"""Online monitor: prefix in, verdict out (Phase 11).

The monitor wraps a trained :class:`MonitorModel` + a :class:`Calibrator` + a
safety policy. Given a trajectory prefix (and optional online patch features) it
returns a :class:`MonitorVerdict`. In shadow mode it logs what it *would* do
without ever raising an alarm, so baseline and shadow runs are behaviorally
identical at temperature 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field

from .abstention import is_committed
from .calibrate import Calibrator
from .features import extract_features
from .interventions import forced_intervention, select_intervention
from .schemas import StepEvent
from .train import MonitorModel


class MonitorVerdict(BaseModel):
    step: int
    risk_score: float
    calibrated_risk: float
    alarm: bool
    recommended_intervention: str = "none"
    reason: str = ""
    abstain: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)
    # Dynamic message for interventions whose text depends on the trajectory (e.g.
    # ``loop_break`` names the specific repeated command). The wrapper prefers this
    # over the static MESSAGES table when non-empty. ``target_command`` + ``penalty``
    # carry the white-box logit-penalty payload for backends that support it.
    intervention_message: str = ""
    target_command: str | None = None
    penalty: float = 0.0


@dataclass
class PolicyConfig:
    min_step: int = 5
    cooldown_steps: int = 5
    max_interventions: int = 2
    threshold: float = 0.5
    high_risk: float = 0.5
    very_high_risk: float = 0.8
    shadow: bool = False
    forced_kind: str | None = None   # if set, only this intervention family fires
    # Selective prediction: abstain (no judgement, never alarm) when the prefix
    # is too early (< min_step) OR the model is too uncertain (confidence below
    # this floor). Confidence = |calibrated_risk - 0.5|. 0.0 disables the
    # confidence gate (step floor via min_step still applies).
    abstain_conf_floor: float = 0.0


def should_alarm(
    risk: float, step: int, last_alarm_step: int | None,
    n_interventions: int, cfg: PolicyConfig,
) -> bool:
    """Safety policy gate. Never alarms before ``min_step`` (early interventions
    are the most disruptive per the Intervention Paradox)."""
    if step < cfg.min_step:
        return False
    if n_interventions >= cfg.max_interventions:
        return False
    if last_alarm_step is not None and (step - last_alarm_step) < cfg.cooldown_steps:
        return False
    return risk >= cfg.threshold


@dataclass
class Monitor:
    model: MonitorModel
    calibrator: Calibrator
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def assess(
        self,
        prefix_steps: Sequence[StepEvent],
        *,
        extra_features: dict[str, Any] | None = None,
        last_alarm_step: int | None = None,
        n_interventions: int = 0,
        has_checkpoint: bool = False,
    ) -> MonitorVerdict:
        import pandas as pd

        step = len(prefix_steps)
        feats = extract_features(prefix_steps, extra=extra_features)
        # the model expects f__-prefixed columns
        row = {f"f__{k}": v for k, v in feats.items()}
        df = pd.DataFrame([row])
        risk = float(self.model.predict_proba_fail(df)[0])
        crisk = float(self.calibrator.transform([risk])[0])

        # Selective prediction: abstain when the prefix is not judgeable yet
        # (too early) or the model is too uncertain. An abstaining monitor never
        # alarms and recommends nothing — it explicitly says "insufficient
        # evidence" rather than masquerading as a confident "low risk".
        if not is_committed(step, crisk, self.policy.min_step, self.policy.abstain_conf_floor):
            return MonitorVerdict(
                step=step, risk_score=risk, calibrated_risk=crisk, alarm=False,
                recommended_intervention="none", abstain=True,
                reason=(f"abstained: step<{self.policy.min_step}"
                        if step < self.policy.min_step
                        else f"abstained: confidence {abs(crisk-0.5):.3f} < {self.policy.abstain_conf_floor}"),
                evidence={"committed": False, "confidence": round(abs(crisk - 0.5), 4)},
            )

        alarm = (not self.policy.shadow) and should_alarm(
            crisk, step, last_alarm_step, n_interventions, self.policy
        )
        would_alarm = should_alarm(crisk, step, last_alarm_step, n_interventions, self.policy)

        # Recent commands let the loop-break intervention name the specific repeated
        # command (the targeted, low-disruption form validated in mech_interp/).
        recent_commands = [
            (s.command or s.action_text or "").strip() for s in prefix_steps
        ]

        if self.policy.forced_kind:
            decision = forced_intervention(
                self.policy.forced_kind, feats, crisk, has_checkpoint,
                recent_commands=recent_commands,
            )
        else:
            decision = select_intervention(
                feats, crisk, high_risk=self.policy.high_risk,
                very_high_risk=self.policy.very_high_risk, has_checkpoint=has_checkpoint,
                recent_commands=recent_commands,
            )

        return MonitorVerdict(
            step=step,
            risk_score=risk,
            calibrated_risk=crisk,
            alarm=bool(alarm and decision.kind != "none"),
            recommended_intervention=decision.kind,
            reason=decision.reason if would_alarm else "below policy threshold",
            intervention_message=decision.message,
            target_command=decision.target_command,
            penalty=decision.penalty,
            evidence={
                "would_alarm": would_alarm,
                "triggers": decision.triggers,
                "n_edit": feats.get("n_edit"),
                "n_test_runs": feats.get("n_test_runs"),
                "max_command_repeat_count": feats.get("max_command_repeat_count"),
                "edited_before_any_read": feats.get("edited_before_any_read"),
                "tests_worsening": feats.get("tests_worsening"),
                "repeated_command": decision.target_command,
            },
        )


def load_monitor(
    models_dir: Path,
    calibrators_dir: Path,
    model_name: str,
    policy: PolicyConfig | None = None,
) -> Monitor:
    """Load a trained model + its calibrator into a :class:`Monitor`."""
    safe = MonitorModel.safe_filename(model_name)
    model = MonitorModel.load(Path(models_dir) / f"{safe}.joblib")
    calib_path = Path(calibrators_dir) / f"{safe}.joblib"
    calibrator = Calibrator.load(calib_path) if calib_path.exists() else Calibrator(method="identity")
    return Monitor(model=model, calibrator=calibrator, policy=policy or PolicyConfig())
