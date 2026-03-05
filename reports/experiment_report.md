# Cavity Optimisation Experiment Report

## Summary
- Selected design: `multioutput`
- Model version: `v1`
- Radius accuracy (test): **0.9858**
- Radius ±1 class (test): **1.0000**
- Height R² (test): **0.9693**
- Height MAE (test): **0.6710 mm**

## Acceptance Gates
- Radius accuracy threshold: 0.863 -> True
- Height R² threshold: 0.857 -> True
- Overall: **True**

## Radius Models (Validation Ranking)
| model_name               | family    |   train_seconds |   val_radius_accuracy |   val_radius_within_1_class |   test_radius_accuracy |
|:-------------------------|:----------|----------------:|----------------------:|----------------------------:|-----------------------:|
| torch_radius_classifier  | legacy_nn |          8.2864 |                0.9778 |                           1 |                 0.9431 |
| extra_trees_classifier   | modern    |          0.2701 |                0.9684 |                           1 |                 0.9747 |
| random_forest_classifier | legacy    |          0.2437 |                0.9589 |                           1 |                 0.9605 |
| decision_tree_classifier | legacy    |          0.0044 |                0.9304 |                           1 |                 0.9463 |

## Height Models (Validation Ranking)
| model_name              | family    |   train_seconds |   val_height_r2 |   val_height_mae |   test_height_r2 |
|:------------------------|:----------|----------------:|----------------:|-----------------:|-----------------:|
| extra_trees_regressor   | modern    |          0.2276 |          0.9677 |           0.6893 |           0.972  |
| decision_tree_regressor | legacy    |          0.0039 |          0.9406 |           1.144  |           0.8917 |
| torch_height_regressor  | legacy_nn |         10.0961 |          0.9202 |           2.1937 |           0.8642 |
| knn_regressor           | legacy    |          0.0016 |          0.9053 |           1.537  |           0.9125 |
| linear_regression       | legacy    |          0.0098 |          0.4504 |           8.469  |           0.4185 |

## Joint Models (Validation Ranking)
| model_name              | family   |   train_seconds |   val_radius_accuracy |   val_height_r2 |   test_radius_accuracy |   test_height_r2 |
|:------------------------|:---------|----------------:|----------------------:|----------------:|-----------------------:|-----------------:|
| multioutput_extra_trees | modern   |          0.4994 |                0.9873 |          0.9673 |                 0.9889 |           0.9723 |

## Artifacts
- Metrics CSV: `artifacts/metrics/metrics.csv`
- Metrics JSON: `artifacts/metrics/metrics.json`
- Best model metadata: `artifacts/models/best_model_metadata.json`
