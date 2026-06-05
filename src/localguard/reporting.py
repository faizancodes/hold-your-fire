"""Render tables (markdown) and figures (PNG) from saved result files (Phases 16-17).

Everything here reads ONLY the CSV/JSON artifacts written by evaluate_monitor.py
and the online runners, so figures and tables are fully reproducible from disk via
``python scripts/make_report.py`` without retraining anything.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Headless rendering.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .utils import read_json  # noqa: E402


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def csv_to_markdown(csv_path: Path, title: str | None = None) -> str:
    if not Path(csv_path).exists():
        return f"_(missing: {csv_path.name})_\n"
    df = pd.read_csv(csv_path)
    md = df.to_markdown(index=False)
    head = f"### {title}\n\n" if title else ""
    return f"{head}{md}\n"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_risk_by_position(figdir: Path, out_png: Path) -> bool:
    path = figdir / "risk_by_position.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    plt.figure(figsize=(6, 4))
    for outcome, label, color in [(0, "successful", "tab:green"), (1, "failed", "tab:red")]:
        g = df[df["y_fail"] == outcome].sort_values("pos_center")
        if len(g):
            plt.plot(g["pos_center"], g["mean_risk"], marker="o", label=label, color=color)
    plt.xlabel("normalized prefix position (step / total steps)")
    plt.ylabel("mean calibrated risk")
    plt.title("Figure 1: Risk over trajectory time")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    _save(out_png)
    return True


def fig_pr_curves(figdir: Path, out_png: Path, judge_csv: Path | None = None) -> bool:
    path = figdir / "pr_curves.json"
    if not path.exists():
        return False
    data = read_json(path)
    plt.figure(figsize=(6, 4))
    for name, d in data.items():
        plt.plot(d["recall"], d["precision"], label=f"{name} (AP={d.get('auprc')})")
    if judge_csv and Path(judge_csv).exists():
        from sklearn.metrics import precision_recall_curve

        jdf = pd.read_csv(judge_csv)
        if {"y_fail", "judge_risk"}.issubset(jdf.columns) and jdf["y_fail"].nunique() > 1:
            prec, rec, _ = precision_recall_curve(jdf["y_fail"], jdf["judge_risk"])
            plt.plot(rec, prec, "--", label="ollama_judge (subset)", color="black")
    plt.xlabel("recall (failed trajectories)")
    plt.ylabel("precision")
    plt.title("Figure 2: Precision-recall (failure = positive)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(out_png)
    return True


def fig_leadtime(figdir: Path, out_png: Path) -> bool:
    path = figdir / "leadtime.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    if df.empty or "lead_steps" not in df.columns:
        return False
    plt.figure(figsize=(6, 4))
    plt.hist(df["lead_steps"].dropna(), bins=20, color="tab:orange", edgecolor="black")
    plt.xlabel("warning lead time (steps before failure)")
    plt.ylabel("# failed trajectories")
    plt.title("Figure 3: Warning lead time at deploy threshold")
    plt.grid(alpha=0.3)
    _save(out_png)
    return True


def fig_intervention_accounting(online_dir: Path, out_png: Path) -> bool:
    files = sorted(Path(online_dir).glob("accounting_*.json"))
    if not files:
        return False
    rows = [read_json(f).get("accounting", {}) for f in files]
    df = pd.DataFrame(rows)
    if df.empty:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: outcome accounting (incl. unchanged, so a null result is still legible)
    metrics = [
        ("recovery_count", "recovery", "tab:green"),
        ("disruption_count", "disruption", "tab:red"),
        ("unchanged_success_count", "unchanged ✓", "tab:olive"),
        ("unchanged_failure_count", "unchanged ✗", "tab:gray"),
    ]
    present = [(m, lbl, c) for m, lbl, c in metrics if m in df.columns]
    x = range(len(df))
    width = 0.8 / max(1, len(present))
    for i, (m, lbl, c) in enumerate(present):
        ax1.bar([xi + i * width for xi in x], df[m], width=width, label=lbl, color=c)
    ax1.set_xticks([xi + width * (len(present) - 1) / 2 for xi in x])
    ax1.set_xticklabels(df["policy"], rotation=15, fontsize=8)
    ax1.set_ylabel("# tasks")
    ax1.set_title("Outcome accounting (recovery vs. disruption)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    # Right: wasted-work — avg steps baseline vs policy
    if {"avg_steps_baseline", "avg_steps_policy"}.issubset(df.columns):
        w = 0.35
        ax2.bar([xi - w / 2 for xi in x], df["avg_steps_baseline"], width=w, label="baseline", color="tab:blue")
        ax2.bar([xi + w / 2 for xi in x], df["avg_steps_policy"], width=w, label="policy", color="tab:orange")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(df["policy"], rotation=15, fontsize=8)
        ax2.set_ylabel("avg steps / run")
        ax2.set_title("Wasted work (avg steps): does it save effort?")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Figure 4: Intervention accounting")
    _save(out_png)
    return True


def fig_feature_importance(figdir: Path, out_png: Path, top_n: int = 15) -> bool:
    path = figdir / "feature_importance.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    df = df.reindex(df["coef"].abs().sort_values(ascending=False).index).head(top_n)
    df = df.sort_values("coef")
    plt.figure(figsize=(7, 5))
    colors = ["tab:red" if c > 0 else "tab:green" for c in df["coef"]]
    plt.barh(df["feature"], df["coef"], color=colors)
    plt.xlabel("logistic-regression coefficient (+ = higher failure risk)")
    plt.title("Figure 5: Top failure-risk features (logistic regression)")
    plt.grid(alpha=0.3, axis="x")
    _save(out_png)
    return True


def fig_risk_coverage(figdir: Path, out_png: Path) -> bool:
    """Risk-coverage curve: AUC vs coverage — the abstention headline artifact."""
    rc = figdir / "risk_coverage.csv"
    if not rc.exists():
        return False
    df = pd.read_csv(rc).sort_values("coverage")
    plt.figure(figsize=(6.5, 4.4))
    plt.plot(df["coverage"], df["auc_selective"], marker="o", color="tab:blue",
             label="confidence-selective")
    if "auc_random" in df.columns:
        plt.plot(df["coverage"], df["auc_random"], marker="x", linestyle="--",
                 color="tab:gray", label="random abstention (baseline)")
    sg = figdir / "risk_coverage_stepgate.csv"
    if sg.exists():
        s = pd.read_csv(sg).sort_values("coverage")
        plt.plot(s["coverage"], s["auc"], marker="s", color="tab:green",
                 label="step-gate (deployable)")
    plt.gca().invert_xaxis()  # high coverage on the left, abstain more to the right
    plt.xlabel("coverage (fraction of prefixes the monitor commits to judge)")
    plt.ylabel("ROC AUC on committed prefixes")
    plt.title("Risk-coverage: a monitor that abstains when it can't tell")
    plt.axhline(0.5, color="k", alpha=0.2)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save(out_png)
    return True


def fig_reliability(figdir: Path, out_png: Path) -> bool:
    path = figdir / "reliability.json"
    if not path.exists():
        return False
    data = read_json(path)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    for key, color in [("raw", "tab:blue"), ("calibrated", "tab:orange")]:
        pts = data.get(key, [])
        if pts:
            xs = [p["mean_predicted"] for p in pts]
            ys = [p["observed_freq"] for p in pts]
            plt.plot(xs, ys, marker="o", label=key, color=color)
    plt.xlabel("mean predicted risk")
    plt.ylabel("observed failure frequency")
    plt.title("Calibration (reliability) — " + str(data.get("model", "")))
    plt.legend()
    plt.grid(alpha=0.3)
    _save(out_png)
    return True


def fig_generalization(out_png: Path) -> bool:
    """Generalization (#1): AUC transfer + calibration before/after recalibration."""
    from .utils import RESULTS_OFFLINE
    p = RESULTS_OFFLINE / "full" / "generalization.json"
    if not p.exists():
        return False
    g = read_json(p)
    xm, xf, ss = g.get("cross_model", {}), g.get("cross_family_qwen", {}), g.get("second_source_openhands", {})
    ab = (ss.get("zero_shot_ensemble_abstain") or ss.get("zero_shot_aligned_abstain") or {}).get("step_ge_35", {})
    fs = (ss.get("fewshot_abstain") or {}).get("step_ge_35", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.2))
    labels = ["within\n70b", "cross-model\n8b+405b", "cross-family\nshell (qwen)",
              "CodeAct\nnaive 0-shot", "CodeAct\n+align+ens", "CodeAct\n+ens+abstain",
              "CodeAct\n+few-shot+abst", "CodeAct\npost-hoc gate"]
    aucs = [xm.get("auc_within_70b"), xm.get("auc_cross_model"), xf.get("monitor_auc_on_qwen"),
            ss.get("zero_shot_auc_naive"),
            ss.get("zero_shot_auc_aligned_ensemble") or ss.get("zero_shot_auc_quantile_aligned"),
            ab.get("auc"), fs.get("auc"), ss.get("posthoc_gate_auc_zero_shot")]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "gold", "tab:green", "tab:purple", "tab:blue"]
    ax1.bar(labels, aucs, color=colors)
    # in-domain ceiling marker across the CodeAct bars (free alignment ~matches it w/o labels)
    if ss.get("in_domain_auc_5fold_cv"):
        ax1.hlines(ss["in_domain_auc_5fold_cv"], 2.6, 5.4, color="k", ls=":", lw=1.5)
        ax1.text(5.4, ss["in_domain_auc_5fold_cv"] + 0.006, "in-domain\nceiling", ha="right", fontsize=7)
    ax1.axhline(0.5, color="k", alpha=0.4, ls="--")
    for i, a in enumerate(aucs):
        if a is not None:
            ax1.text(i, a + 0.008, f"{a:.3f}", ha="center", fontsize=8)
    ax1.set_ylim(0.5, 0.8); ax1.set_ylabel("ROC AUC")
    ax1.tick_params(axis="x", labelsize=7)
    ax1.set_title("Cross-scaffold collapse is mostly a scale artifact (free fixes recover it)", fontsize=9)
    ece = [xm.get("ece_within"), xm.get("ece_cross_naive"), xm.get("ece_cross_recalibrated")]
    ax2.bar(["within\n70b", "cross\n(naive)", "cross\n(recalibrated)"], ece,
            color=["tab:blue", "tab:red", "tab:green"])
    for i, e in enumerate(ece):
        if e is not None:
            ax2.text(i, e + 0.001, f"{e:.3f}", ha="center", fontsize=9)
    ax2.set_ylabel("Expected Calibration Error")
    ax2.set_title("Calibration shifts, but recalibration fixes it")
    fig.suptitle("Figure 7: Generalization — transfers across scale/shell; cross-scaffold collapse "
                 "is a scale artifact recovered free by alignment + abstention", fontsize=10)
    _save(out_png)
    return True


def fig_validation_lies(out_png: Path) -> bool:
    """Methodology (#4): validation gains that evaporate / reverse on held-out test."""
    from .utils import RESULTS_OFFLINE
    p = RESULTS_OFFLINE / "full" / "validation_vs_test.json"
    if not p.exists():
        return False
    d = read_json(p)
    cases = d["cases"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = range(len(cases)); w = 0.38
    ax.bar([i - w / 2 for i in x], [c["val"] for c in cases], width=w, label="validation", color="tab:purple")
    ax.bar([i + w / 2 for i in x], [c["test"] for c in cases], width=w, label="held-out test (paired)", color="tab:gray")
    ax.axhline(d.get("baseline_test", 0.7214), color="k", ls="--", alpha=0.6,
               label=f"baseline test ({d.get('baseline_test', 0.721):.3f})")
    ax.set_xticks(list(x))
    ax.set_xticklabels([c["label"].replace(" ", "\n", 1) for c in cases], fontsize=8)
    ax.set_ylim(0.70, 0.735); ax.set_ylabel("ROC AUC")
    ax.set_title("Figure 8: Validation lies — gains evaporate (or reverse) on held-out test")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    _save(out_png)
    return True


def fig_audit_validation(out_png: Path) -> bool:
    """Human-validated audit (#2): heuristic vs blind human failure-mode distribution."""
    from .utils import RESULTS_AUDITS
    p = RESULTS_AUDITS / "human_validation.json"
    if not p.exists():
        return False
    d = read_json(p)

    def counts(block):
        # values look like "16 (40%)" -> 16
        return {m: int(str(v).split()[0]) for m, v in d.get(block, {}).items()}

    heur, human = counts("heuristic_dist_flagged"), counts("human_dist_flagged")
    order = ["looping", "patch_churn", "insufficient_context", "environment_distraction",
             "test_neglect", "submission_too_early", "not_observable"]
    modes = [m for m in order if m in heur or m in human]
    hv = [heur.get(m, 0) for m in modes]
    uv = [human.get(m, 0) for m in modes]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = range(len(modes)); w = 0.38
    ax.bar([i - w / 2 for i in x], hv, width=w, label="heuristic (regex)", color="tab:red")
    ax.bar([i + w / 2 for i in x], uv, width=w, label="human (blind read)", color="tab:blue")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=8)
    ax.set_ylabel("# flagged failures (of 40)")
    prec = d.get("looping_precision_heuristic")
    ax.set_title(f"Figure 9: Human-validated audit — regex over-calls 'looping'\n"
                 f"(precision {prec}); blind read shows loops + churn", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    _save(out_png)
    return True


def _save(out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
