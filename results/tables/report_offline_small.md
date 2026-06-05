# LocalGuard-SWE results — `offline_small`

## Dataset split

```json
{
  "group_col": "instance_id",
  "n_train_rows": 10440,
  "n_val_rows": 2901,
  "n_test_rows": 3210,
  "n_train_groups": 320,
  "n_val_groups": 80,
  "n_test_groups": 100,
  "fail_rate_train": 0.6306,
  "fail_rate_val": 0.5722,
  "fail_rate_test": 0.667
}
```

Best deployable model: **random_forest** (deploy policy: T3_success_far_lte_10pct).


## Table 1 — Offline monitor performance

| model                          | features          | split    |   ROC_AUC |   ROC_AUC_lo |   ROC_AUC_hi |   AUPRC |   Brier |    ECE |   success_FAR@T3 |   failed_coverage@T3 |   median_lead_steps@T3 |
|:-------------------------------|:------------------|:---------|----------:|-------------:|-------------:|--------:|--------:|-------:|-----------------:|---------------------:|-----------------------:|
| baseline_majority              | none (constant)   | instance |    0.5    |       0.5    |       0.5    |  0.667  |  0.2311 | 0.0948 |           0      |               0      |                      0 |
| baseline_step_count_only       | step count only   | instance |    0.6421 |       0.5994 |       0.6848 |  0.8018 |  0.21   | 0.0843 |           0.0714 |               0.3737 |                     10 |
| heuristic_rule_monitor         | hand rules        | instance |    0.5396 |       0.4363 |       0.6434 |  0.702  |  0.2245 | 0.071  |           0.2429 |               0.3434 |                     18 |
| logistic_regression            | structured        | instance |    0.6911 |       0.6357 |       0.7493 |  0.833  |  0.2045 | 0.0709 |           0.1643 |               0.4798 |                     16 |
| random_forest                  | structured        | instance |    0.7155 |       0.6616 |       0.7676 |  0.8444 |  0.2022 | 0.0735 |           0.15   |               0.4798 |                     17 |
| hist_gradient_boosting         | structured        | instance |    0.6923 |       0.6396 |       0.7512 |  0.832  |  0.2053 | 0.0741 |           0.0857 |               0.4091 |                     14 |
| structured_plus_tfidf_logistic | structured + text | instance |    0.6396 |       0.5467 |       0.7459 |  0.795  |  0.2282 | 0.125  |           0.1929 |               0.3636 |                     14 |


## Table 2 — Feature ablation

| feature_set              |   ROC_AUC |   AUPRC |   success_FAR@T3 |   median_lead_steps@T3 |
|:-------------------------|----------:|--------:|-----------------:|-----------------------:|
| step_count_only          |    0.6619 |  0.8017 |           0.0714 |                   10   |
| length_pace              |    0.6818 |  0.8253 |           0.0786 |                   10   |
| action_counts            |    0.6699 |  0.8149 |           0.05   |                   13   |
| context_before_edit      |    0.5835 |  0.7311 |           0      |                   18   |
| file_behavior            |    0.6032 |  0.764  |           0      |                   13.5 |
| testing_behavior         |    0.5418 |  0.6939 |           0.0143 |                   16.5 |
| loop_behavior            |    0.6691 |  0.812  |           0.0857 |                   10.5 |
| all_structured           |    0.6923 |  0.832  |           0.0857 |                   14   |
| all_structured_plus_text |    0.6396 |  0.795  |           0.1929 |                   14   |


## Table 3 — Threshold tradeoff (best model)

| threshold_policy         |   threshold |   success_false_alarm_rate |   failed_alert_rate |   median_lead_steps |
|:-------------------------|------------:|---------------------------:|--------------------:|--------------------:|
| T1_max_f1                |      0.3735 |                     0.9786 |              0.9949 |                  23 |
| T2_success_far_lte_20pct |      0.6667 |                     0.2429 |              0.6061 |                  17 |
| T3_success_far_lte_10pct |      0.8462 |                     0.15   |              0.4798 |                  17 |
| T4_success_far_lte_5pct  |      0.9091 |                     0.0357 |              0.298  |                  13 |


## Local Ollama judge vs classifier (held-out subset)

```json
{
  "model": "qwen2.5-coder:7b",
  "n": 40,
  "n_pos": 20,
  "invalid_json_rate": 0.0,
  "avg_latency_s": 11.02,
  "judge_auc": 0.58,
  "judge_auprc": 0.5623015873015873,
  "judge_auc_valid_only": 0.58,
  "classifier": "random_forest",
  "classifier_auc": 0.5725,
  "classifier_auprc": 0.6021833029462322,
  "should_intervene_rate": 1.0
}
```


## Table 4 — Online intervention accounting

| policy               |   n_tasks |   baseline_success_rate |   policy_success_rate |   recovery_count |   disruption_count |   unchanged_success_count |   unchanged_failure_count |   safe_saving_count |   harmful_waste_count |   avg_steps_baseline |   avg_steps_policy |   interventions_per_run |
|:---------------------|----------:|------------------------:|----------------------:|-----------------:|-------------------:|--------------------------:|--------------------------:|--------------------:|----------------------:|---------------------:|-------------------:|------------------------:|
| online_evidence_gate |         3 |                  0.3333 |                0.6667 |                1 |                  0 |                         1 |                         1 |                   0 |                     0 |                   40 |                 40 |                       0 |
| online_loop_guard    |         3 |                  0.3333 |                0.3333 |                0 |                  0 |                         1 |                         2 |                   0 |                     0 |                   40 |                 31 |                       2 |


## Table 5 — Local cost / resource

| model                        | policy               | ollama_tag       | model_size   | context_window   |   avg_tokens_per_sec |   avg_run_time_s |   avg_steps |   success_rate |   n_runs |
|:-----------------------------|:---------------------|:-----------------|:-------------|:-----------------|---------------------:|-----------------:|------------:|---------------:|---------:|
| ollama_chat/qwen2.5-coder:7b | baseline             | qwen2.5-coder:7b | 4.7GB        | 32K              |                 11.1 |            328.3 |        28.2 |          0.4   |        5 |
| ollama_chat/qwen2.5-coder:7b | online_evidence_gate | qwen2.5-coder:7b | 4.7GB        | 32K              |                  4.3 |           1607.2 |        40   |          1     |        1 |
| ollama_chat/qwen2.5-coder:7b | online_loop_guard    | qwen2.5-coder:7b | 4.7GB        | 32K              |                  4.5 |            935.1 |        31   |          0.333 |        3 |
| ollama_chat/qwen2.5-coder:7b | online_shadow        | qwen2.5-coder:7b | 4.7GB        | 32K              |                 25.6 |            274.7 |        40   |          0.333 |        3 |


Shadow behaviorally identical to baseline (shadow_check_level1): **True**


## Figures

- `results/figures/fig1_risk_by_position_offline_small.png`
- `results/figures/fig2_pr_curves_offline_small.png`
- `results/figures/fig3_leadtime_offline_small.png`
- `results/figures/fig4_intervention_accounting_offline_small.png`
- `results/figures/fig5_feature_importance_offline_small.png`
- `results/figures/calibration_offline_small.png`
