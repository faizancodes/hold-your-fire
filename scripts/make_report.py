#!/usr/bin/env python3
"""Regenerate all tables and figures from saved result files (Phases 16-17).

  python scripts/make_report.py --config configs/offline_full.yaml
  python scripts/make_report.py --only figures

Reads only the CSV/JSON artifacts on disk; never retrains. Writes PNGs to
results/figures/ and a consolidated markdown report to paper/results_<name>.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from localguard import reporting as R
from localguard.utils import (
    REPO_ROOT,
    RESULTS_FIGURES,
    RESULTS_ONLINE,
    RESULTS_TABLES,
    ensure_dirs,
    load_config,
    read_json,
)


def _maybe(fn, *a) -> str:
    name = getattr(fn, "__name__", "fig")
    ok = False
    try:
        ok = fn(*a)
    except Exception as exc:  # keep the report building even if one figure fails
        print(f"  [warn] {name}: {exc}")
    return "ok" if ok else "skip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/offline_full.yaml")
    ap.add_argument("--only", choices=["tables", "figures", "all"], default="all")
    args = ap.parse_args()

    cfg = load_config(args.config).data
    name = cfg.get("name", "offline")
    results_dir = REPO_ROOT / cfg["results_dir"]
    figdir = results_dir / "figdata"
    judge_csv = figdir / "ollama_judge_predictions.csv"
    ensure_dirs(RESULTS_FIGURES, RESULTS_TABLES)

    if args.only in ("figures", "all"):
        print(f"[report] figures for {name} -> {RESULTS_FIGURES}")
        print("  fig1 risk_by_position :", _maybe(R.fig_risk_by_position, figdir, RESULTS_FIGURES / f"fig1_risk_by_position_{name}.png"))
        print("  fig2 pr_curves        :", _maybe(R.fig_pr_curves, figdir, RESULTS_FIGURES / f"fig2_pr_curves_{name}.png", judge_csv))
        print("  fig3 leadtime         :", _maybe(R.fig_leadtime, figdir, RESULTS_FIGURES / f"fig3_leadtime_{name}.png"))
        print("  fig4 intervention acct:", _maybe(R.fig_intervention_accounting, RESULTS_ONLINE, RESULTS_FIGURES / "fig4_intervention_accounting.png"))
        print("  fig5 feature_importnce:", _maybe(R.fig_feature_importance, figdir, RESULTS_FIGURES / f"fig5_feature_importance_{name}.png"))
        print("  calibration reliability:", _maybe(R.fig_reliability, figdir, RESULTS_FIGURES / f"calibration_{name}.png"))
        print("  risk-coverage (abstain):", _maybe(R.fig_risk_coverage, figdir, RESULTS_FIGURES / f"fig6_risk_coverage_{name}.png"))
        print("  fig7 generalization    :", _maybe(R.fig_generalization, RESULTS_FIGURES / "fig7_generalization.png"))
        print("  fig8 validation-lies   :", _maybe(R.fig_validation_lies, RESULTS_FIGURES / "fig8_validation_lies.png"))
        print("  fig9 audit-validation  :", _maybe(R.fig_audit_validation, RESULTS_FIGURES / "fig9_audit_validation.png"))

    if args.only in ("tables", "all"):
        _build_markdown_report(name, results_dir)


def _build_markdown_report(name: str, results_dir: Path) -> None:
    lines: list[str] = [f"# LocalGuard-SWE results — `{name}`\n"]

    summary_path = results_dir / "split_summary.json"
    if summary_path.exists():
        s = read_json(summary_path)
        lines.append("## Dataset split\n")
        lines.append("```json\n" + __import__("json").dumps(s, indent=2) + "\n```\n")

    res_path = results_dir / "offline_results.json"
    if res_path.exists():
        res = read_json(res_path)
        lines.append(f"Best deployable model: **{res.get('best_model')}** "
                     f"(deploy policy: {res.get('deploy_policy')}).\n")

    lines.append("\n## Table 1 — Offline monitor performance\n")
    lines.append(R.csv_to_markdown(results_dir / "table1_offline.csv"))
    lines.append("\n## Table 2 — Feature ablation\n")
    lines.append(R.csv_to_markdown(results_dir / "table2_ablation.csv"))
    lines.append("\n## Table 3 — Threshold tradeoff (best model)\n")
    lines.append(R.csv_to_markdown(results_dir / "table3_thresholds.csv"))

    judge_path = results_dir / "ollama_judge.json"
    if judge_path.exists():
        j = read_json(judge_path)
        lines.append("\n## Local Ollama judge vs classifier (held-out subset)\n")
        lines.append("```json\n" + __import__("json").dumps(j, indent=2) + "\n```\n")

    cost_path = results_dir / "cost.json"
    if cost_path.exists():
        lines.append("\n## Deployment cost (CPU-only, measured)\n")
        lines.append("```json\n" + __import__("json").dumps(read_json(cost_path), indent=2) + "\n```\n")

    # online accounting (if any)
    acc_files = sorted(RESULTS_ONLINE.glob("accounting_*.json"))
    if acc_files:
        lines.append("\n## Table 4 — Online intervention accounting\n")
        rows = []
        for f in acc_files:
            rows.append(read_json(f).get("accounting", {}))
        import pandas as pd

        lines.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    # Table 5 — local cost / resource (from online run log)
    t5 = _cost_resource_table()
    if t5 is not None:
        lines.append("\n## Table 5 — Local cost / resource\n")
        lines.append(t5.to_markdown(index=False) + "\n")

    shadow_files = sorted(RESULTS_ONLINE.glob("shadow_check_level*.json"))
    for f in shadow_files:
        sc = read_json(f)
        lines.append(f"\nShadow behaviorally identical to baseline "
                     f"({f.stem}): **{sc.get('behaviorally_identical')}**\n")

    lines.append("\n## Figures\n")
    for fig in ["fig1_risk_by_position", "fig2_pr_curves", "fig3_leadtime",
                "fig4_intervention_accounting", "fig5_feature_importance", "calibration"]:
        lines.append(f"- `results/figures/{fig}_{name}.png`")
    lines.append("")

    out_paper = REPO_ROOT / "paper" / f"results_{name}.md"
    out_tables = RESULTS_TABLES / f"report_{name}.md"
    text = "\n".join(lines)
    out_paper.parent.mkdir(parents=True, exist_ok=True)
    out_paper.write_text(text)
    out_tables.parent.mkdir(parents=True, exist_ok=True)
    out_tables.write_text(text)
    print(f"[report] wrote {out_paper}")
    print(f"[report] wrote {out_tables}")


_MODEL_SPECS = {
    "ollama_chat/qwen2.5-coder:7b": {"ollama_tag": "qwen2.5-coder:7b", "model_size": "4.7GB", "context_window": "32K"},
    "ollama_chat/qwen2.5-coder:14b": {"ollama_tag": "qwen2.5-coder:14b", "model_size": "9.0GB", "context_window": "32K"},
}


def _cost_resource_table():
    """Aggregate Table 5 (cost/resource) from results/online/online_runs.jsonl."""
    import pandas as pd

    from localguard.utils import read_jsonl

    log = RESULTS_ONLINE / "online_runs.jsonl"
    if not log.exists():
        return None
    df = pd.DataFrame(list(read_jsonl(log)))
    if df.empty:
        return None
    rows = []
    for (model, policy), g in df.groupby(["model", "policy"]):
        runtime = g["runtime_s"].mean()
        tokens = g["total_tokens_approx"].mean()
        spec = _MODEL_SPECS.get(model, {"ollama_tag": model, "model_size": "?", "context_window": "?"})
        rows.append({
            "model": model, "policy": policy, **spec,
            "avg_tokens_per_sec": round(tokens / runtime, 1) if runtime else 0,
            "avg_run_time_s": round(runtime, 1),
            "avg_steps": round(g["n_steps"].mean(), 1),
            "success_rate": round(g["success"].mean(), 3),
            "n_runs": len(g),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
