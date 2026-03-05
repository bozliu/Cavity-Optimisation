# Cavity Optimisation Experiment Report

## Summary
- Selected design: `two_task`
- Model version: `v1`
- Radius accuracy (test): **0.8152**
- Radius ±1 class (test): **0.9795**
- Height R² (test): **0.9465**
- Height MAE (test): **1.2696 mm**

## Acceptance Gates
- Radius accuracy threshold: 0.863 -> False
- Height R² threshold: 0.857 -> True
- Overall: **False**

## Radius Models (Validation Ranking)
| model_name               | family    |   train_seconds |   val_radius_accuracy |   val_radius_within_1_class |   test_radius_accuracy |
|:-------------------------|:----------|----------------:|----------------------:|----------------------------:|-----------------------:|
| torch_radius_classifier  | legacy_nn |         15.9159 |                0.8386 |                      0.9715 |                 0.7804 |
| random_forest_classifier | legacy    |          0.4332 |                0.6772 |                      0.9684 |                 0.6524 |
| extra_trees_classifier   | modern    |          0.3442 |                0.6551 |                      0.962  |                 0.6603 |
| decision_tree_classifier | legacy    |          0.0084 |                0.6519 |                      0.9051 |                 0.6335 |

## Height Models (Validation Ranking)
| model_name              | family    |   train_seconds |   val_height_r2 |   val_height_mae |   test_height_r2 |
|:------------------------|:----------|----------------:|----------------:|-----------------:|-----------------:|
| torch_height_regressor  | legacy_nn |          9.9582 |          0.9699 |           1.0622 |           0.9395 |
| extra_trees_regressor   | modern    |          0.216  |          0.9653 |           1.0088 |           0.9241 |
| knn_regressor           | legacy    |          0.0026 |          0.9432 |           1.3635 |           0.9012 |
| decision_tree_regressor | legacy    |          0.0039 |          0.876  |           1.3576 |           0.8617 |
| linear_regression       | legacy    |          0.0033 |          0.4302 |           8.4758 |           0.3953 |

## Joint Models (Validation Ranking)
| model_name              | family   |   train_seconds |   val_radius_accuracy |   val_height_r2 |   test_radius_accuracy |   test_height_r2 |
|:------------------------|:---------|----------------:|----------------------:|----------------:|-----------------------:|-----------------:|
| multioutput_extra_trees | modern   |          0.4997 |                0.7627 |          0.9647 |                 0.7314 |           0.9241 |

## Artifacts
- Metrics CSV: `artifacts/metrics/metrics.csv`
- Metrics JSON: `artifacts/metrics/metrics.json`
- Best model metadata: `artifacts/models/best_model_metadata.json`
