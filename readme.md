
```markdown
# BRIGHTEN Depression Prediction Pipeline

A machine learning pipeline for predicting depression severity (PHQ-9) from passive smartphone sensing and ecological momentary assessment (EMA) data, using the open-access BRIGHTEN V1 and V2 datasets.

---

## Overview

The BRIGHTEN study collected longitudinal EMA and passive phone sensor data from participants with depression across two study versions. This repository implements a complete ML pipeline: raw ingestion → cleaning → feature engineering → PCA → predictive modeling.

**Target variable:** PHQ-9 depression sum score (continuous regression)  
**Study versions:**
- **V1** — Calls, SMS, mobility + daily/weekly EMA
- **V2** — Extended passive sensors (GPS clusters, weather, communication) + daily/weekly EMA

**Four dataset variants tracked throughout:**

| Name | Description |
|---|---|
| `v1_day` | V1 daily granularity |
| `v2_day` | V2 daily granularity |
| `v1_week` | V1 weekly granularity |
| `v2_week` | V2 weekly granularity |

---

## Pipeline Steps

```
01_cleaning.ipynb                    → Raw data ingestion, date parsing, deduplication
02_outcome_codes.ipynb               → Outcome variable construction
02_var_clustering.ipynb              → Variable correlation clustering
02_processing_Pipeline_oct25.ipynb   → Transformation, scaling, train/val/test splits
03_subject_footprint.ipynb           → Per-subject data characterization
03_eda.ipynb                         → Exploratory data analysis
EDA.ipynb                            → Exploratory data analysis
03_feature_pca.ipynb                 → Feature-level PCA per cluster
04_pca_nbs.ipynb                     → Subject-level PCA & symptom correlation networks
04_predictive_models.ipynb           → Cross-validated modeling + SHAP interpretation
```

**Support modules (`scripts/`):** `preprocessing.py`, `feature_selection.py`, `visualization.py`, `clustering.py`, `modeling.py`

---

## Data

Data is available from [Synapse.org](https://synapse.org). Place raw CSVs in `BRIGHTEN_data/`.

---

## Models Benchmarked

Ridge Regression · Random Forest · XGBoost · HistGradientBoosting · GroupMean (subject-mean baseline)

All models use **group-aware cross-validation** (subjects never split across folds). Metrics: R², MAE, RMSE. SHAP used for interpretability.

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly scipy networkx
```
```

---

## README_03_EDA.md

```markdown
# 03_EDA.py — Exploratory Data Analysis

## Purpose

Conducts structured EDA on the four processed BRIGHTEN datasets (`v1_day`, `v2_day`, `v1_week`, `v2_week`) after the cleaning and transformation pipeline. The goal is to understand data distributions, missingness, skewness, and inter-variable relationships before modeling.

## Inputs

Reads the `*_trainval_transformed.csv` files produced by `02_processing_Pipeline_oct25.py` from `BRIGHTEN_data/`.

## Key Steps

1. **Distribution inspection** — Histograms and summary stats for numeric variables, split by dataset variant.
2. **Skewness & kurtosis audit** — Identifies features with skew > 1 or kurtosis > 2 across non-binary columns. These are flagged for further transformation.
3. **Missingness analysis** — Evaluates proportion of missing data per variable to inform imputation decisions.
4. **Correlation heatmaps** — Examines pairwise correlations across feature subsets (sensors, surveys, baseline).
5. **Target variable exploration** — Distribution of PHQ-9 scores across time, versions, and subject subgroups.

## Outputs

Primarily visual (matplotlib/seaborn plots). Flags skewed columns stored in `skewed_cols` dict for downstream use.

## Notes

- Weather features are partially excluded based on prior correlation analysis (only `humidity_mean/median` and `cloud_cover_mean/median` retained).
- Binary/indicator/missing-flag columns are excluded from skew/kurtosis calculations.
- Results inform which transformations (Yeo-Johnson, quantile) are applied in the processing pipeline.
```

---

## README_04_pca_nbs.md

```markdown
# 04_pca_nbs.py — Subject-Level PCA & Symptom Networks

## Purpose

Applies PCA **within each variable cluster** to reduce correlated features into interpretable principal components (PCs). Then constructs per-subject correlation networks across those PCs to capture individual symptom-sensor covariation structure.

## Inputs

- `*_trainval_transformed.csv` — Processed feature data (from `02_processing_Pipeline`)
- Cluster assignments from `02_var_clustering.py` (via `feature_selection.py` utilities)

## Key Steps

### 1. Symptom/Sensor Correlation Matrices
For each dataset variant, builds per-subject correlation matrices across sensor and EMA features separately, using `fs.make_symptom_matrices()`. Weather features are filtered to only the most informative subset.

### 2. PCA per Cluster
`fs.pca_on_clusters()` applies PCA (default 1 component per cluster) to each variable cluster, producing named PCs (e.g., `pc_mobility`, `pc_calls`, `pc_phq2`). Loadings heatmaps can be toggled. Results saved as `*_trainval_sensor_pca.csv`.

### 3. Per-Subject Network Visualization
For each subject, computes pairwise correlations across their PC scores and renders a weighted network graph (`fs.plot_network()`). Edges are color-coded: green = positive correlation, red = negative. Fixed node layouts are predefined for V1 daily and V1 weekly variants.

### 4. Heatmaps of PC Correlations
Group-level heatmaps of PC-to-PC correlations across all subjects for each dataset variant.

### 5. Train/Val/Test Split
`GroupShuffleSplit` is used to create subject-disjoint splits: 15% held-out test set, then 20% of remainder as validation. Splits are saved for downstream modeling.

## Outputs

- `*_trainval_sensor_pca.csv` — PCA-reduced feature files
- Network visualizations per subject (first 10 subjects per variant)
- Correlation heatmaps

## Key Functions Used

| Function | Module | Description |
|---|---|---|
| `make_symptom_matrices()` | `feature_selection` | Per-subject correlation matrix construction |
| `pca_on_clusters()` | `feature_selection` | PCA per cluster, returns scores + loadings |
| `merge_df_via_cluster_pca_dict()` | `feature_selection` | Merges PC scores onto original dataframe |
| `plot_network()` | `feature_selection` | Network graph of inter-PC correlations |
```

---

## README_04_predictive_models.md

```markdown
# 04_predictive_models.py — Predictive Modeling & SHAP

## Purpose

Trains and evaluates multiple regression models to predict PHQ-9 depression scores from processed features and PCA-derived components. Uses group-aware cross-validation to prevent subject leakage, then applies SHAP for feature attribution.

## Inputs

- PCA-reduced CSVs from `04_pca_nbs.py` (`*_trainval_sensor_pca.csv`)
- Train/val/test splits (subject-disjoint, from `GroupShuffleSplit`)

## Models

| Model | Notes |
|---|---|
| `Ridge` | Linear, regularized |
| `Random Forest` | Ensemble, `random_state=42` |
| `XGBoost` | `reg:squarederror` objective |
| `HistGradientBoosting` | Handles missing natively |
| `GroupMean` | Predicts subject's mean PHQ-9 — dummy baseline |

## Key Steps

### 1. Feature/Target Setup
For each dataset variant (`v1_week`, `v2_week`) and each time window (`8wks`, `both`), features (`X`) and target (`y = phq9_sum`) are constructed. PHQ-9/PHQ-2 columns are excluded from features to avoid leakage.

### 2. Cross-Validation
`GroupKFold` ensures subjects are not split across folds. Scoring: R², MAE, RMSE (negative). Results stored in nested `model_dict[name][y_col][time][model_name]`.

### 3. Validation Set Evaluation
In addition to CV, each model is evaluated on a held-out validation set. Predictions stored for downstream analysis.

### 4. SHAP Interpretation
After training, `shap.Explainer` is applied to the best model per fold. SHAP values are aggregated across folds for stable feature attribution. `shap.initjs()` enables interactive plots.

## Outputs

- `model_dict` — Nested dictionary of all CV scores, predictions, and fitted models
- SHAP summary plots per model/variant/time combination

## Metrics

| Metric | Key |
|---|---|
| R² | `r2_scores` |
| MAE | `mae_scores` (negated) |
| RMSE | `rmse_scores` (negated) |
| Pearson r | Custom `pearsonr_scorer` |

## Notes

- The `GroupMeanRegressor` requires manual `groups` passing and does not use `cross_validate` directly.
- A commented-out PHQ-9 baseline comparison block is present for benchmarking against baseline survey alone.
- Memory is managed with `gc.collect()` between SHAP runs due to large model sizes.
```

---

These are ready to copy into your repo. Want me to retry writing them to files once the container becomes available, or bundle them into a single document?