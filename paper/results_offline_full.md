# LocalGuard-SWE results — `offline_full`

## Dataset split

```json
{
  "group_col": "instance_id",
  "n_train_rows": 81070,
  "n_val_rows": 20856,
  "n_test_rows": 25166,
  "n_train_groups": 2297,
  "n_val_groups": 575,
  "n_test_groups": 719,
  "fail_rate_train": 0.5733,
  "fail_rate_val": 0.5755,
  "fail_rate_test": 0.5687
}
```

Best deployable model: **hist_gradient_boosting** (deploy policy: T3_success_far_lte_10pct).


## Table 1 — Offline monitor performance

| model                          | features          | split    |   ROC_AUC |   ROC_AUC_lo |   ROC_AUC_hi |   AUPRC |   Brier |    ECE |   success_FAR@T3 |   failed_coverage@T3 |   median_lead_steps@T3 |
|:-------------------------------|:------------------|:---------|----------:|-------------:|-------------:|--------:|--------:|-------:|-----------------:|---------------------:|-----------------------:|
| baseline_majority              | none (constant)   | instance |    0.5    |       0.5    |       0.5    |  0.5687 |  0.2453 | 0.0068 |           0      |               0      |                      0 |
| baseline_step_count_only       | step count only   | instance |    0.6231 |       0.6068 |       0.6364 |  0.7067 |  0.2279 | 0.0141 |           0.0853 |               0.3359 |                     13 |
| heuristic_rule_monitor         | hand rules        | instance |    0.5744 |       0.5384 |       0.6073 |  0.6316 |  0.2377 | 0.0116 |           0      |               0      |                      0 |
| logistic_regression            | structured        | instance |    0.6931 |       0.6726 |       0.7106 |  0.7623 |  0.2168 | 0.0177 |           0.071  |               0.3593 |                     12 |
| random_forest                  | structured        | instance |    0.7191 |       0.6993 |       0.7397 |  0.7926 |  0.2085 | 0.0154 |           0.0771 |               0.399  |                     14 |
| hist_gradient_boosting         | structured        | instance |    0.7216 |       0.7002 |       0.7425 |  0.7916 |  0.2083 | 0.0111 |           0.0901 |               0.4373 |                     13 |
| structured_plus_tfidf_logistic | structured + text | instance |    0.681  |       0.6489 |       0.7115 |  0.7535 |  0.2219 | 0.0335 |           0.0737 |               0.3097 |                     15 |


## Table 2 — Feature ablation

| feature_set              |   ROC_AUC |   AUPRC |   success_FAR@T3 |   median_lead_steps@T3 |
|:-------------------------|----------:|--------:|-----------------:|-----------------------:|
| step_count_only          |    0.6401 |  0.709  |           0.0853 |                     13 |
| length_pace              |    0.6822 |  0.7555 |           0.0683 |                      6 |
| action_counts            |    0.7079 |  0.7747 |           0.0901 |                     14 |
| context_before_edit      |    0.6002 |  0.6531 |           0.0648 |                     13 |
| file_behavior            |    0.6191 |  0.6684 |           0.0478 |                     12 |
| testing_behavior         |    0.5562 |  0.6092 |           0.1488 |                     15 |
| loop_behavior            |    0.6734 |  0.7476 |           0.0765 |                     15 |
| all_structured           |    0.7216 |  0.7916 |           0.0901 |                     13 |
| all_structured_plus_text |    0.6718 |  0.7422 |           0.0737 |                     15 |


## Table 3 — Threshold tradeoff (best model)

| threshold_policy         |   threshold |   success_false_alarm_rate |   failed_alert_rate |   median_lead_steps |
|:-------------------------|------------:|---------------------------:|--------------------:|--------------------:|
| T1_max_f1                |      0.3755 |                     0.9863 |              0.9993 |                  18 |
| T2_success_far_lte_20pct |      0.7031 |                     0.17   |              0.5578 |                  14 |
| T3_success_far_lte_10pct |      0.7778 |                     0.0901 |              0.4373 |                  13 |
| T4_success_far_lte_5pct  |      0.8514 |                     0.0218 |              0.2743 |                  11 |


## Deployment cost (CPU-only, measured)

```json
{
  "hardware": "Apple M4 MacBook (10 cores, 32 GB RAM), CPU-only (sklearn HistGradientBoosting)",
  "model": "hist_gradient_boosting",
  "model_size_on_disk_mb": 1.14,
  "rss_baseline_libs_mb": 104.2,
  "model_resident_mb": 76.9,
  "rss_serving_mb": 205.9,
  "peak_rss_mb": 1009.3,
  "peak_rss_note": "peak includes loading the 25,166-row eval parquet for throughput; NOT a deployment cost",
  "n_prefixes_timed": 300,
  "assess_end_to_end_ms": {
    "median": 8.581,
    "p10": 7.381,
    "p90": 14.527
  },
  "feature_extraction_ms": {
    "median": 0.136,
    "p10": 0.074,
    "p90": 0.233
  },
  "raw_numpy_predict_ms": {
    "median": 6.634,
    "p10": 5.232,
    "p90": 13.243
  },
  "amortized_model_latency_ms": 0.0025,
  "batch_throughput_prefixes_per_s": 394795.0,
  "full_test_set_scoring_s_median": 0.06,
  "n_test_prefixes": 25166,
  "judge_latency_s_per_prefix": 11.6,
  "judge_model_weights_gb": 4.7,
  "speedup_vs_judge_realpath": 1352,
  "memory_ratio_vs_judge": 61
}
```


## Table 4 — Online intervention accounting

| policy            |   n_tasks |   baseline_success_rate |   policy_success_rate |   recovery_count |   disruption_count |   unchanged_success_count |   unchanged_failure_count |   safe_saving_count |   harmful_waste_count |   avg_steps_baseline |   avg_steps_policy |   interventions_per_run |
|:------------------|----------:|------------------------:|----------------------:|-----------------:|-------------------:|--------------------------:|--------------------------:|--------------------:|----------------------:|---------------------:|-------------------:|------------------------:|
| online_loop_guard |         3 |                  0.3333 |                0.3333 |                0 |                  0 |                         1 |                         2 |                   0 |                     0 |                   40 |                 31 |                       2 |


## Table 5 — Local cost / resource

| model                        | policy            | ollama_tag       | model_size   | context_window   |   avg_tokens_per_sec |   avg_run_time_s |   avg_steps |   success_rate |   n_runs |
|:-----------------------------|:------------------|:-----------------|:-------------|:-----------------|---------------------:|-----------------:|------------:|---------------:|---------:|
| ollama_chat/qwen2.5-coder:7b | baseline          | qwen2.5-coder:7b | 4.7GB        | 32K              |                 26   |            211.7 |          40 |          0.333 |        3 |
| ollama_chat/qwen2.5-coder:7b | online_loop_guard | qwen2.5-coder:7b | 4.7GB        | 32K              |                  4.5 |            935.1 |          31 |          0.333 |        3 |
| ollama_chat/qwen2.5-coder:7b | online_shadow     | qwen2.5-coder:7b | 4.7GB        | 32K              |                 25.6 |            274.7 |          40 |          0.333 |        3 |


Shadow behaviorally identical to baseline (shadow_check_level1): **True**


## Figures

- `results/figures/fig1_risk_by_position_offline_full.png`
- `results/figures/fig2_pr_curves_offline_full.png`
- `results/figures/fig3_leadtime_offline_full.png`
- `results/figures/fig4_intervention_accounting_offline_full.png`
- `results/figures/fig5_feature_importance_offline_full.png`
- `results/figures/calibration_offline_full.png`
