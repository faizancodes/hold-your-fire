#!/usr/bin/env python3
"""Frontier LLM-judge baseline (GPT-5.5) on the SAME 200 test prefixes as the local
qwen2.5-coder:7b judge — a stronger baseline for the cheap structured classifier.

Paid API. Safe by construction:
  - replicates the local judge's sampling EXACTLY (same config/seed/n) -> same prefixes
  - CHECKPOINTS every judged prefix to JSONL -> resumable, a crash never re-spends
  - --limit for a cheap dry run; prints token usage + cost estimate before scaling
  - the deployed monitor stays 100% local; this only buys a stronger baseline

  # dry run (3 calls, measure cost):
  python scripts/run_openai_judge_subset.py --limit 3
  # full run (resumes from checkpoint):
  python scripts/run_openai_judge_subset.py --n 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from dotenv import load_dotenv

from localguard.evaluate import paired_bootstrap_auc_delta, prefix_metrics
from localguard.openai_judge import DEFAULT_OPENAI_MODEL, judge_prefix_openai
from localguard.schemas import NormalizedTrajectory
from localguard.train import MonitorModel
from localguard.utils import (
    INTERIM_DIR, REPO_ROOT, load_config, read_json, read_jsonl, write_json,
)


def _load_normalized(dataset_path: str) -> dict[str, NormalizedTrajectory]:
    path = INTERIM_DIR / f"normalized_{Path(dataset_path).stem}.jsonl"
    out: dict[str, NormalizedTrajectory] = {}
    for row in read_jsonl(path):
        t = NormalizedTrajectory(**row)
        out[t.trajectory_id] = t
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/offline_small.yaml")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-step", type=int, default=4)
    ap.add_argument("--max-step", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="cap calls this run (0 = all); for dry runs")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--max-completion-tokens", type=int, default=2000)
    ap.add_argument("--price-in", type=float, default=None, help="$ per 1M input tokens (for cost est.)")
    ap.add_argument("--price-out", type=float, default=None, help="$ per 1M output tokens (incl. reasoning)")
    ap.add_argument("--tag", default="", help="suffix for output/checkpoint files (e.g. 'high') to keep runs separate")
    ap.add_argument("--prefix-ids-from", default=None,
                    help="CSV with a prefix_id column; pin to EXACTLY these prefixes (overrides sampling)")
    ap.add_argument("--backend", default="openai", choices=["openai", "openrouter"])
    args = ap.parse_args()
    suf = f"_{args.tag}" if args.tag else ""

    load_dotenv(str(REPO_ROOT / ".env"))  # explicit path (frame-walking find_dotenv breaks under some shells)
    import os
    from openai import OpenAI
    or_effort = None if args.reasoning_effort in (None, "", "none", "off") else args.reasoning_effort
    if args.backend == "openrouter":
        from localguard.openrouter_judge import OPENROUTER_BASE_URL, judge_prefix_openrouter
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
        def judge_fn(steps):
            return judge_prefix_openrouter(steps, client, model=args.model,
                                           max_tokens=args.max_completion_tokens,
                                           reasoning_effort=or_effort)
    else:
        client = OpenAI()
        def judge_fn(steps):
            return judge_prefix_openai(steps, client, model=args.model,
                                       max_completion_tokens=args.max_completion_tokens,
                                       reasoning_effort=args.reasoning_effort)

    cfg = load_config(args.config)
    results_dir = REPO_ROOT / cfg["results_dir"]
    models_dir = REPO_ROOT / cfg["models_dir"]
    figdir = results_dir / "figdata"
    figdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = figdir / f"gpt_judge_checkpoint{suf}.jsonl"

    df = pd.read_parquet(REPO_ROOT / cfg["dataset"])
    folds = pd.read_parquet(results_dir / "split_assignment.parquet")
    test = df.merge(folds, on="prefix_id").query("fold == 'test'")
    test = test[(test["prefix_step"] >= args.min_step) & (test["prefix_step"] <= args.max_step)]

    if args.prefix_ids_from:  # pin to EXACTLY a prior run's prefixes (bulletproof apples-to-apples)
        ids = list(pd.read_csv(args.prefix_ids_from)["prefix_id"])
        sample = test[test["prefix_id"].isin(set(ids))].set_index("prefix_id").loc[ids].reset_index()
        print(f"[gpt-judge] pinned to {len(sample)} prefixes from {Path(args.prefix_ids_from).name}")
    else:
        # EXACT same stratified sample as the local judge (same seed/n -> same prefixes)
        per = args.n // 2
        pos, neg = test[test["y_fail"] == 1], test[test["y_fail"] == 0]
        take_pos = pos.sample(min(per, len(pos)), random_state=args.seed)
        take_neg = neg.sample(min(args.n - len(take_pos), len(neg)), random_state=args.seed)
        sample = pd.concat([take_pos, take_neg]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    done = {r["prefix_id"]: r for r in read_jsonl(ckpt_path)} if ckpt_path.exists() else {}
    todo = [r for _, r in sample.iterrows() if r["prefix_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[gpt-judge] {len(sample)} sampled ({int(sample['y_fail'].sum())} fail) | "
          f"{len(done)} already done | judging {len(todo)} now | model={args.model} effort={args.reasoning_effort}")

    normalized = _load_normalized(cfg["dataset"])
    toks_in = toks_out = toks_reason = 0
    or_cost_sum = 0.0  # OpenRouter native per-call cost when available
    n_valid_new = 0
    for i, r in enumerate(todo):
        traj = normalized.get(r["trajectory_id"])
        if traj is None:
            continue
        steps = traj.steps[: int(r["prefix_step"])]
        out = judge_fn(steps)
        ex = out.extra or {}
        toks_in += ex.get("prompt_tokens", 0)
        toks_out += ex.get("completion_tokens", 0)
        toks_reason += ex.get("reasoning_tokens", 0)
        or_cost_sum += ex.get("or_cost_usd") or 0.0
        n_valid_new += int(out.valid_json)
        rec = {
            "prefix_id": r["prefix_id"], "trajectory_id": r["trajectory_id"],
            "instance_id": r.get("instance_id", r["trajectory_id"]),
            "prefix_step": int(r["prefix_step"]), "y_fail": int(r["y_fail"]),
            "judge_risk": out.risk_score, "valid_json": out.valid_json,
            "latency_s": round(out.latency_s, 2),
            "should_intervene": bool(out.judgment.should_intervene) if out.judgment else False,
            "intervention_type": out.judgment.intervention_type if out.judgment else "none",
            "prompt_tokens": ex.get("prompt_tokens", 0), "completion_tokens": ex.get("completion_tokens", 0),
            "reasoning_tokens": ex.get("reasoning_tokens", 0), "resolved_model": ex.get("resolved_model", ""),
            "finish_reason": ex.get("finish_reason", ""), "or_cost_usd": ex.get("or_cost_usd"),
            "format_used": ex.get("format_used", ""), "error": out.error,
        }
        done[r["prefix_id"]] = rec
        with open(ckpt_path, "a") as fh:  # append-checkpoint immediately (resumable)
            fh.write(__import__("json").dumps(rec) + "\n")
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            print(f"  judged {i+1}/{len(todo)}  valid={n_valid_new}  "
                  f"tok_in={toks_in} tok_out={toks_out} (reason={toks_reason})")

    # ---- cost for THIS run's calls ----
    cost = None
    if or_cost_sum > 0:  # OpenRouter reports actual $ per call — exact, preferred
        cost = round(or_cost_sum, 4)
    elif args.price_in is not None and args.price_out is not None:
        cost = round(toks_in / 1e6 * args.price_in + toks_out / 1e6 * args.price_out, 4)
    print(f"[judge] tokens this run: in={toks_in} out={toks_out} (reasoning={toks_reason}) | "
          f"cost=${cost if cost is not None else '? (pass --price-in/--price-out)'}"
          f"{' [OpenRouter actual]' if or_cost_sum > 0 else ' [estimate]'}")

    if args.limit:  # dry run: stop before computing final metrics
        per_call = (toks_in + toks_out) / max(1, len(todo))
        proj = (cost / len(todo) * args.n) if cost else None
        print(f"[dry-run] ~{per_call:.0f} tokens/call, ${cost or 0:.4f} for {len(todo)} -> "
              f"projected n={args.n}: ~{per_call*args.n/1000:.0f}k tokens"
              f"{f', ~${proj:.2f}' if proj else ''}. Re-run without --limit for the full set.")
        return

    _finalize(args, df, results_dir, models_dir, done, toks_in, toks_out, toks_reason, cost)


def _finalize(args, df, results_dir, models_dir, done, toks_in, toks_out, toks_reason, cost):
    jdf = pd.DataFrame(list(done.values()))
    y = jdf["y_fail"].to_numpy(int)
    judge_m = prefix_metrics(y, jdf["judge_risk"].to_numpy(float))

    best_name = read_json(results_dir / "offline_results.json").get("best_model", "random_forest")
    clf = MonitorModel.load(models_dir / f"{MonitorModel.safe_filename(best_name)}.joblib")
    clf_df = df[df["prefix_id"].isin(jdf["prefix_id"])].set_index("prefix_id").loc[jdf["prefix_id"]].reset_index()
    clf_risk = clf.predict_proba_fail(clf_df)
    jdf["clf_risk"] = clf_risk
    clf_m = prefix_metrics(y, clf_risk)

    # paired bootstrap: classifier (base) vs GPT judge (new), grouped by instance
    paired = paired_bootstrap_auc_delta(
        jdf["instance_id"].to_numpy(), y, clf_risk, jdf["judge_risk"].to_numpy(float), n_boot=2000)

    ollama = read_json(results_dir / "ollama_judge.json")
    eff_effort = args.reasoning_effort
    if args.backend == "openrouter" and args.reasoning_effort in (None, "", "none", "off"):
        eff_effort = None  # OpenRouter w/o explicit effort = no extended thinking
    out_obj = {
        "judge_model": args.model,
        "backend": args.backend,
        "resolved_model": jdf["resolved_model"].mode().iat[0] if len(jdf) else args.model,
        "reasoning_effort": eff_effort,
        "n": int(len(jdf)), "n_pos": int(y.sum()),
        "invalid_json_rate": round(1 - jdf["valid_json"].mean(), 4),
        "avg_latency_s": round(float(jdf["latency_s"].mean()), 2),
        "gpt_judge_auc": judge_m.get("roc_auc"), "gpt_judge_auprc": judge_m.get("auprc"),
        "classifier": best_name, "classifier_auc": clf_m.get("roc_auc"),
        "local_judge_auc_qwen7b": ollama.get("judge_auc"),
        "should_intervene_rate": round(float(jdf["should_intervene"].mean()), 4),
        "paired_classifier_minus_gptjudge": {
            "auc_classifier": paired["auc_base"], "auc_gpt_judge": paired["auc_new"],
            "delta_gpt_minus_clf": paired["delta"], "delta_lo": paired["delta_lo"],
            "delta_hi": paired["delta_hi"], "frac_gpt_better": paired["frac_new_better"],
        },
        "tokens": {"input": toks_in, "output": toks_out, "reasoning": toks_reason},
        "est_cost_usd": cost,
    }
    suf = f"_{args.tag}" if args.tag else ""
    write_json(results_dir / f"gpt_judge{suf}.json", out_obj)
    jdf.to_csv(results_dir / "figdata" / f"gpt_judge_predictions{suf}.csv", index=False)
    print("\n=== GPT-5.5 judge (effort=%s) vs cheap classifier (same prefixes, n=%d) ===" % (args.reasoning_effort, len(jdf)))
    print(f"  classifier AUC : {clf_m.get('roc_auc'):.3f}")
    print(f"  GPT-5.5  AUC   : {judge_m.get('roc_auc'):.3f}   (local qwen7b judge: {ollama.get('judge_auc'):.3f})")
    print(f"  paired delta (GPT - clf): {paired['delta']:+.3f}  95% CI [{paired['delta_lo']}, {paired['delta_hi']}]"
          f"  frac GPT better: {paired['frac_new_better']}")
    print(f"  GPT should_intervene rate: {out_obj['should_intervene_rate']}  invalid JSON: {out_obj['invalid_json_rate']}")
    print(f"  avg latency: {out_obj['avg_latency_s']}s  | tokens in/out/reason: {toks_in}/{toks_out}/{toks_reason}"
          f"  | est_cost=${cost}")
    print(f"[gpt-judge] wrote {results_dir/'gpt_judge.json'}")


if __name__ == "__main__":
    main()
