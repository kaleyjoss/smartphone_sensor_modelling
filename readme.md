
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


---

## README_03_EDA.md


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

![Sub responses](/Users/klj9278/Library/CloudStorage/Box-Box/Kaley_research_NYU/EMA_projects/smartphone_sensor_modelling_may26/EDA/charts/example_sub_responses.png)
![Corr across subs](/Users/klj9278/Library/CloudStorage/Box-Box/Kaley_research_NYU/EMA_projects/smartphone_sensor_modelling_may26/EDA/charts/strong_corr_heatmap.png)

## Outputs

Primarily visual (matplotlib/seaborn plots). Flags skewed columns stored in `skewed_cols` dict for downstream use.

## Notes

- Weather features are partially excluded based on prior correlation analysis (only `humidity_mean/median` and `cloud_cover_mean/median` retained).
- Binary/indicator/missing-flag columns are excluded from skew/kurtosis calculations.
- Results inform which transformations (Yeo-Johnson, quantile) are applied in the processing pipeline.


---

## README_04_pca_nbs.md


# 04_pca_nbs.py — Subject-Level PCA & Symptom Networks

## Purpose

Applies PCA **within each variable cluster** to reduce correlated features into interpretable principal components (PCs). Then constructs per-subject correlation networks across those PCs to capture individual symptom-sensor covariation structure.

## Inputs

- `*_trainval_transformed.csv` — Processed feature data (from `02_processing_Pipeline`)
- Cluster assignments from `02_var_clustering.py` (via `feature_selection.py` utilities)

## Key Steps

#### 1. Symptom/Sensor Correlation Matrices
For each dataset variant, builds per-subject correlation matrices across sensor and EMA features separately, using `fs.make_symptom_matrices()`. Weather features are filtered to only the most informative subset.

![Subject network](/Users/klj9278/Library/CloudStorage/Box-Box/Kaley_research_NYU/EMA_projects/smartphone_sensor_modelling_may26/results/feature_pca/ex_sub_sensor_correlation.png)

#### 2. PCA per Cluster
`fs.pca_on_clusters()` applies PCA (default 1 component per cluster) to each variable cluster, producing named PCs (e.g., `pc_mobility`, `pc_calls`, `pc_phq2`). Loadings heatmaps can be toggled. Results saved as `*_trainval_sensor_pca.csv`.

![HierAgg](/Users/klj9278/Library/CloudStorage/Box-Box/Kaley_research_NYU/EMA_projects/smartphone_sensor_modelling_may26/results/feature_pca/v1_hieragg_clustering.png)

#### 3. Per-Subject Network Visualization
For each subject, computes pairwise correlations across their PC scores and renders a weighted network graph (`fs.plot_network()`). Edges are color-coded: green = positive correlation, red = negative. Fixed node layouts are predefined for V1 daily and V1 weekly variants.
![PC sub](/Users/klj9278/Library/CloudStorage/Box-Box/Kaley_research_NYU/EMA_projects/smartphone_sensor_modelling_may26/results/subject_pc_networks/v1_day/94.0_pc_network.png)

#### 4. Heatmaps of PC Correlations
Group-level heatmaps of PC-to-PC correlations across all subjects for each dataset variant.
![PC Corrs](/Users/klj9278/Library/CloudStorage/Box-Box/Kaley_research_NYU/EMA_projects/smartphone_sensor_modelling_may26/results/feature_pca/v1_pc_correlation.png)

#### 5. Train/Val/Test Split
`GroupShuffleSplit` is used to create subject-disjoint splits: 15% held-out test set, then 20% of remainder as validation. Splits are saved for downstream modeling.

## Outputs
- `*_trainval_sensor_pca.csv` — PCA-reduced feature files
- Network visualizations per subject (first 10 subjects per variant)
- Correlation heatmaps

## Key Functions Used
| Function | Script | Description |
|---|---|---|
| `make_symptom_matrices()` | `feature_selection` | Per-subject correlation matrix construction |
| `pca_on_clusters()` | `feature_selection` | PCA per cluster, returns scores + loadings |
| `merge_df_via_cluster_pca_dict()` | `feature_selection` | Merges PC scores onto original dataframe |
| `plot_network()` | `feature_selection` | Network graph of inter-PC correlations |

---

## README_04_predictive_models.md
Trains and evaluates multiple regression models to predict PHQ-9 depression scores from processed features and PCA-derived components. Uses group-aware cross-validation to prevent subject leakage, then applies SHAP for feature attribution.

### Inputs

- PCA-reduced CSVs from `04_pca_nbs.py` (`*_trainval_sensor_pca.csv`)
- Train/val/test splits (subject-disjoint, from `GroupShuffleSplit`)

### Models

| Model | Notes |
|---|---|
| `HistGradientBoosting` | Handles missing natively |
| `GroupMean` | Predicts subject's mean PHQ-9 — dummy baseline |


#### 1. Feature/Target Setup
For each dataset variant (`v1_week`, `v2_week`) and each time window (`8wks`, `both`), features (`X`) and target (`y = phq9_sum`) are constructed. PHQ-9/PHQ-2 columns are excluded from features to avoid leakage.

#### 2. Cross-Validation
`GroupKFold` ensures subjects are not split across folds. Scoring: R², MAE, RMSE (negative). Results stored in nested `model_dict[name][y_col][time][model_name]`.

#### 3. Validation Set Evaluation
In addition to CV, each model is evaluated on a held-out validation set. Predictions stored for downstream analysis.

#### 4. SHAP Interpretation
After training, `shap.Explainer` is applied to the best model per fold. SHAP values are aggregated across folds for stable feature attribution. `shap.initjs()` enables interactive plots.

### Outputs

- `model_dict` — Nested dictionary of all CV scores, predictions, and fitted models
- SHAP summary plots per model/variant/time combination


| Pearson r | Custom `pearsonr_scorer` |

**Notes**

- The `GroupMeanRegressor` requires manual `groups` passing and does not use `cross_validate` directly.
- A commented-out PHQ-9 baseline comparison block is present for benchmarking against baseline survey alone.
- Memory is managed with `gc.collect()` between SHAP runs due to large model sizes.

---


# Results — mlVAR Group Comparison: End-of-Study Depression Status (`end_depressed`)

## 5.1 Analytic sample

| Variant | Group | N |
|---|---|---|
| V1 daily (4-node PCA) | Not depressed at 6 weeks | 103 |
| V1 daily (4-node PCA) | Depressed at 6 weeks | 62 |
| V2 daily (6-node PCA) | Not depressed at 6 weeks | 69 |
| V2 daily (6-node PCA) | Depressed at 6 weeks | 72 |

---

## 5.2 V1 daily — 4-node PCA model (communication/PHQ-2)

**Nodes:** `phq2`, `sms`, `calls`, `uncalls` (unreturned calls).

**Table 1. Mean lag-1 autoregressive (inertia) coefficients by group.**

| Node | Not depressed (N=103) | Depressed (N=62) | Δ (not-dep − dep) |
|---|---|---|---|
| phq2 | 0.40 | 0.48 | ≈ −0.08 |
| sms | 0.47 | 0.52 | ≈ −0.05 |
| calls | 0.32 | 0.40 | ≈ −0.08 |
| uncalls | 0.31 | 0.40 | ≈ −0.09 |

*Per-group autoregression values are read off each panel's self-loop labels. The "Differences"
panel reports its own independently-rendered Δ values (not simply the arithmetic difference of
the rounded per-group labels above); see Table 2 for the figure's own difference values.*

![v1 mlvar](/Users/klj9278/Library/CloudStorage/Box-Box/Kaley_research_NYU/EMA_projects/smartphone_sensor_modelling_may26/results/mlVAR/trainval/Figure_end_depressed2_pca_v1_day_165p_.pdf)

**Table 3. Permutation-significant differences (p < .05), V1 daily 4-node model.**

| Edge | Δ (not-dep − dep) | Direction |
|---|---|---|
| **uncalls → phq2** | **+0.04** | Stronger in the not-depressed group |

Only one edge survives the permutation test in this model: the cross-lagged path from
unreturned calls to next-day PHQ-2 is stronger in participants who were not depressed at 6 weeks.
None of the four autoregressive (inertia) edges reached significance in this model, despite all
four trending toward higher inertia in the depressed group (Table 2) — consistent in direction
with, but weaker in magnitude/significance than, the `stayed_depressed` group comparison reported
in `results_mlVAR.md` (§4.3–4.4), which used a stricter "stable status at both timepoints"
grouping.


---

## 5.3 V2 daily — 6-node PCA model (mobility/weather/PHQ-2)

**Nodes:** `commute`, `vehicle_location`, `active_mobility`, `sleep`, `phq2`, `temp`.

**Table 4. Mean lag-1 autoregressive (inertia) coefficients by group.**

| Node | Not depressed (N=69) | Depressed (N=72) |
|---|---|---|
| vehicle_location | 0.43 | 0.52 |
| sleep | 0.36 | 0.44 |
| active_mobility | 0.27 | 0.41 |
| commute / temp* | ≈ 0.27–0.29 | ≈ 0.36–0.38 |
| phq2 | 0.27 | 0.36 |

![v2 end depressed](/Users/klj9278/Library/CloudStorage/Box-Box/Kaley_research_NYU/EMA_projects/smartphone_sensor_modelling_may26/results/mlVAR/trainval/Figure_end_depressed_binary_pca_v2_day_141p_.pdf)

**Table 5. Permutation-significant differences (p < .05), V2 daily 6-node model.**

| Edge | Δ (not-dep − dep) | Direction |
|---|---|---|
| vehicle_location → commute | +0.07 | Stronger in the not-depressed group |
| commute → temp | +0.03 | Stronger in the not-depressed group |
| **phq2 (AR)** | **−0.12** | Stronger in the depressed group |

Three edges reach significance in the V2 replication. As in the V1 model, **PHQ-2 inertia is
significantly higher in the depressed group** (Δ = −0.12), reinforcing the core inertia finding
from the `stayed_depressed` analysis using an entirely different node set (weather/mobility
rather than communication) and a different grouping definition (6-week endpoint status rather
than stable baseline-to-endpoint status). The two significant cross-lags (`vehicle_location →
commute`, `commute → temp`) are both stronger in the not-depressed group and involve only
mobility/weather nodes, with no significant cross-lag involving `phq2`, `sleep`, or
`active_mobility`.

> **NOTE/LIMITATION:** This V2 node set is dominated by mobility and
> weather-adjacent variables (`commute`, `vehicle_location`, `active_mobility`, `temp` — 4 of 6
> nodes). The `commute → temp` and `vehicle_location → commute` effects may partly reflect
> geographic/seasonal patterns in travel and weather exposure rather than purely psychological
> dynamics. The phq2 autoregression difference is the only significant edge directly implicating
> mood and should be weighted most heavily as a replication of the inertia finding.

---

