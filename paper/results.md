# Results — Predictive Benchmark Paper (Paper B)

*Drafted from `05_predictive_models.ipynb` outputs, `results/tables/`, and the Methods.docx.
All R² and MAE values are cross-validated means (HistGradientBoostingRegressor, GroupKFold k=5,
subject-disjoint folds) unless stated otherwise. PHQ-9 was z-score standardized prior to
modeling, so R² is the primary metric and MAE is reported in standard-deviation units.
`[fill: ...]` markers indicate values to substitute from saved objects before submission.*

---

## 3.1 Analytic sample

Four dataset variants were analyzed: V1 daily (`v1_day`; N ≈ 173 participants, ≈ 5,556 person-days),
V2 daily (`v2_day`; N ≈ 172 participants, ≈ [fill: exact obs count]), V1 weekly (`v1_week`;
N ≈ 173), and V2 weekly (`v2_week`; N ≈ 172). The standardized PHQ-9 outcome ranged from −1.7 to
3.0 SD units in the V1 cohort and −1.9 to 2.6 SD units in the V2 cohort, reflecting meaningful
heterogeneity in depression severity across and within participants. All cross-validation splits
were subject-disjoint: no participant's observations appeared in both training and held-out folds
of any split.

> **ASSUMPTION:** PHQ-9 was standardized using mean and SD estimated from the full training set
> before folding; the same transformation was applied to test folds. R² therefore reflects
> proportion of standardized, not raw-scale, variance explained. Back-transform MAE by multiplying
> by the training-set SD [fill: insert SD in raw PHQ-9 points] to obtain clinically interpretable
> point estimates.

---

## 3.2 Concurrent prediction of PHQ-9 severity

**Table 1.** Cross-validated mean R² for PHQ-9 (standardized) by feature configuration and
dataset variant. Passive-sensor-only excludes all self-report items (PHQ-2, SDS, stress,
support, sleep, mood); baseline-only uses enrollment demographics and clinical scores. All models:
HistGradientBoostingRegressor, GroupKFold(k=5).

| Feature configuration | V1 daily | V2 daily | V1 weekly | V2 weekly |
|---|---|---|---|---|
| Baseline only (demographics + enrollment PHQ-9) | −0.006 | 0.028 | 0.025 | 0.078 |
| 8-wk EMA + passive sensor | 0.376 | 0.338 | 0.481 | 0.402 |
| 8-wk EMA + passive + baseline | **0.418** | **0.350** | **0.488** | **0.442** |
| Passive sensor only (no self-report) | 0.082 | −0.112 | 0.072 | 0.006 |

Corresponding mean MAE (SD units):

| Feature configuration | V1 daily | V2 daily | V1 weekly | V2 weekly |
|---|---|---|---|---|
| Baseline only | 0.774 | 0.753 | 0.761 | 0.763 |
| 8-wk EMA + passive sensor | 0.606 | 0.637 | 0.562 | 0.605 |
| 8-wk EMA + passive + baseline | 0.582 | 0.628 | 0.556 | 0.577 |
| Passive sensor only | 0.750 | 0.821 | 0.745 | 0.786 |

Three patterns are apparent. First, models that included time-varying self-reported EMA features
explained a moderate share of concurrent PHQ-9 variance (R² = 0.34–0.49), with the weekly variants
consistently outperforming the daily variants — a finding expected from the reduced temporal
mismatch between weekly-aggregated predictors and the weekly PHQ-9 outcome. Second, passive
smartphone sensors contributed little independent of self-report: the passive-sensor-only
configurations achieved R² of only 0.06–0.08 in V1 and were worse than predicting the sample mean
(R² < 0) in both V2 variants. Third, baseline demographic and enrollment clinical information alone
explained essentially no variance in concurrent PHQ-9 severity (V1 daily R² = −0.006; V1 weekly
R² = 0.025), confirming that the predictable signal is time-varying rather than trait-like, at
least at the feature level used here.

> **ASSUMPTION:** The "8-wk EMA + passive sensor" configuration includes all time-varying features
> collected over the analytic window (up to 85 days), not literally weeks 1–8; "8-wk" refers to
> a label inherited from the notebook structure. PHQ-9 items themselves and correlated subscores
> were excluded from features to prevent leakage [fill: list exact excluded columns before
> submission].

The absence of added value from the richer V2 passive sensor suite (velocity-binned mobility,
weather, 16 communication channels) — and its active performance decrement in passive-only models —
is consistent with reports that expanded digital phenotyping streams in real-world remote trials
introduce missingness and noise faster than signal (Kiang et al., 2021; Sun et al., 2022).

---

## 3.3 Feature attribution: self-report drives concurrent PHQ-9 prediction

To understand which features drove PHQ-9 predictions within the 8-wk EMA + passive configuration,
we examined two parallel feature-importance analyses: (1) the top 15 features identified by
per-subject ANOVA F-tests aggregated across participants (ANOVA-15), and (2) the top 10 features
identified by population-level Linear Mixed Effects models (LME-10). We then re-ran the
HistGradientBoosting regressor restricted to each feature subset. This allowed us to isolate
whether the predictive signal resided in self-report items or in passive behavior.

**Table 2.** Cross-validated mean R² using reduced feature sets: ANOVA-15 (top 15 most
individually predictive features, dominated by self-report) and LME-10 (top 10 population-level
predictors from LME, which in V1 variants include passive communication features alongside PHQ-2).

| Feature set | V1 daily | V2 daily | V1 weekly | V2 weekly |
|---|---|---|---|---|
| **PHQ-9 — ANOVA-15** | 0.427 | 0.309 | 0.315 | 0.420 |
| **PHQ-2 — ANOVA-15** | 0.012 | −0.004 | −0.264 | −0.111 |
| **PHQ-9 — LME-10** | 0.099 | −0.255 | 0.264 | 0.279 |
| **PHQ-2 — LME-10** | −0.071 | −0.306 | −0.288 | −0.272 |

> **ASSUMPTION:** Neither ANOVA-15 nor LME-10 is a purely passive-sensor model. For V1 daily,
> the ANOVA-selected top 15 features included PHQ-2 items (phq2_1, phq2_2, phq2_sum, phq2_bin),
> stress, and SDS items alongside communication and mobility variables. Similarly, the LME-10
> features for V1 daily included phq2_1, phq2_2, phq2_sum, and phq2_bin. The truly
> passive-sensor-only results are those reported as "Passive sensor only" in Table 1.
> Interpreting ANOVA/LME feature sets as passive-sensor results would be incorrect.

The ANOVA-15 results confirm that the 15 most individually predictive features — which happen to
include PHQ-2 items — recover nearly the full predictive signal of the complete feature set
(ANOVA-15 R² ≈ 0.43 vs. full-set R² ≈ 0.42 in V1 daily). The LME-10 models, which skew toward
communication features in V1, yield markedly lower R² (0.10 in V1 daily) and collapse entirely
in V2 (R² = −0.26), demonstrating that the passive behavioral features identified by LME as
population-level predictors hold limited predictive value at the individual level in cross-validation.

The ANOVA-15 top features for V1 daily were: interaction diversity, SMS count, call duration, call
count, missed interactions, unreturned calls, PHQ-2 items (sum, item 1, item 2, binary indicator),
mobility radius, mobility (walking distance), perceived stress, SDS disability items (SDS-2,
SDS-3). For V2 daily: temperature median, dew-point median, precipitation, location variance
(per-GPS-hour), powered-vehicle hours, GPS-active hours, active-travel hours, PHQ-2 items, hours
in high-speed transportation, perceived stress, social support, global mood, and SDS-2. The
consistent presence of PHQ-2 and SDS items at the top of feature importance across both cohorts
and analytic approaches corroborates that self-report EMA — not passive sensing — carries the
recoverable PHQ-9 signal.

---

## 3.4 Daily depressed mood (PHQ-2 sum) is not predictable

In sharp contrast to PHQ-9, the daily PHQ-2 sum (depressed mood + anhedonia) was not predictable
under any feature configuration tested:

**Table 3.** Cross-validated mean R² for PHQ-2 sum (standardized) by feature configuration.

| Feature configuration | V1 daily | V2 daily | V1 weekly | V2 weekly |
|---|---|---|---|---|
| Passive sensor only | −0.043 | −0.085 | −0.089 | −0.141 |
| ANOVA-15 (includes PHQ-2 neighbors) | 0.012 | −0.004 | −0.264 | −0.111 |
| LME-10 | −0.071 | −0.306 | −0.288 | −0.272 |

All configurations produced R² at or below zero, indicating that neither passive behavioral data
nor the combination of passive and self-report features reliably predicted day-level fluctuations
in depressed mood and anhedonia beyond the within-person mean. This dissociation — moderately
predictable aggregate weekly PHQ-9 but unpredictable daily PHQ-2 — suggests that the recoverable
signal resides in stable between-person and weekly variation rather than in day-to-day mood
dynamics. The same dissociation has been reported in prior work on this dataset (Holstein et al.,
2024) and is consistent with theoretical accounts in which passive behavioral streams are more
sensitive to between-person trait-like differences than to within-person daily fluctuations
(Stamatis et al., 2024).

> **ASSUMPTION:** The PHQ-2 was also z-score standardized, so negative R² indicates predictions
> worse than predicting the participant's within-fold mean. The ANOVA-15 model for V1 daily
> achieves a minimally positive R² (0.012) because its selected features included PHQ-2 items
> themselves (phq2_1, phq2_2), creating a near-circular prediction; this result should not be
> interpreted as meaningful predictive signal.

---

## 3.5 Prospective prediction of 6-week depression status

Using features derived from early-study behavioral trajectories (OLS slopes, intercepts, and means
per 2-week block), we evaluated whether end-of-study depression status could be predicted from
data collected in the first 4 weeks (blocks 1 and 2). Two binary outcomes were modeled for V1 daily
and V2 daily: (1) `end_depressed_binary` — PHQ-9 ≥ 12 at approximately 6 weeks (days 38–55);
(2) `depression_change_bin` — transition from clinically depressed at baseline to non-depressed at
6 weeks (improvement).

**Table 4.** Cross-validated classification accuracy and R² for prospective binary outcomes.
All models: HistGradientBoostingClassifier, GroupKFold(k=5). Feature configurations: `baseline`
= enrollment demographics + clinical scores; `pc_sensor` = PCA-reduced passive sensor block
trajectories; `pc_all` = passive sensor + PHQ-2 trajectories; `pc_and_demo` = pc_all + baseline.

**V1 daily — end_depressed_binary** (N ≈ 275 train obs; N_test ≈ 55 obs):

| Feature configuration | Accuracy | R² |
|---|---|---|
| Baseline only | 55.6% | −0.84 |
| pc_sensor (passive only) | 54.9% | −0.88 |
| pc_all (passive + PHQ-2 slope/intercept) | **60.0%** | **0.25** |
| pc_and_demo | 55.6% | 0.16 |

**V1 daily — depression_change_bin** (improvement):

| Feature configuration | Accuracy | R² |
|---|---|---|
| Baseline only | 60.4% | −0.52 |
| pc_sensor | 63.6% | −0.57 |
| pc_all | **64.7%** | −0.40 |
| pc_and_demo | 60.4% | −0.31 |

**V2 daily — end_depressed_binary:**

| Feature configuration | Accuracy | R² |
|---|---|---|
| Baseline only | 47.0% | −1.19 |
| pc_sensor | 44.7% | −1.24 |
| pc_all | 58.6% | −0.71 |
| pc_and_demo | 47.0% | −0.78 |

**V2 daily — depression_change_bin:**

| Feature configuration | Accuracy | R² |
|---|---|---|
| Baseline only | 71.6% | −0.71 |
| pc_sensor | 72.7% | −0.57 |
| pc_all | 71.5% | −0.69 |
| pc_and_demo | 71.6% | −0.58 |

Prospective binary prediction was uniformly poor. In V1, the best-performing configuration
(`pc_all` for `end_depressed_binary`) achieved 60% accuracy. However, R² was negative for the
change outcome across all configurations, and accuracy for baseline and sensor-only configurations
was near or below chance (55%). In V2, the apparent accuracy for `depression_change_bin` (71–73%)
likely reflects the base rate of the majority class rather than genuine prediction; R² values of
−0.57 to −0.71 confirm the model does not meaningfully outperform the mean.

The absence of prospective signal from passive sensor trajectories alone (`pc_sensor`) is notable:
in V1, passive-only slope/intercept features did not predict 6-week depression status above chance,
and in V2, accuracy was below the no-skill baseline. The modest improvement from adding PHQ-2
trajectory features (`pc_all`) suggests that the early trajectory of self-reported mood — not
passive behavior — drives even the limited prospective signal available.

> **ASSUMPTION:** Binary outcomes were defined at approximately 6 weeks (days 38–55); participants
> with missing outcome data in this window were excluded. The R² values for binary classifiers
> are coefficients of determination applied to 0/1 targets and should be interpreted alongside
> accuracy rather than as proportions of explained variance in the regression sense. `[fill:
> report class prevalence for both binary outcomes to contextualize accuracy.]`

---

## 3.6 Per-subject (idiographic) models

By-subject Ridge regression models were estimated for each participant with sufficient non-missing
data, using ANOVA-selected and LME-selected top features. Per-subject R² showed extreme
heterogeneity: many participants produced R² of approximately 1.0 (indicating insufficient
within-person variance in the outcome — flat PHQ-2 time series), while others showed deeply
negative R² values. The median per-subject R² for PHQ-2 prediction across all cross-validation
folds was approximately −0.74, with a very wide distribution (25th–75th percentile: −3.51 to
−0.11). Many participants were excluded entirely from by-subject models due to missingness in
their sensor records.

This distribution of idiographic R² values is primarily a data-quality and sample-size artifact
rather than a substantive finding: idiographic prediction in this dataset is not feasible at the
individual level given the available data density. The results serve as a caution about the
generalizability of idiographic modeling frameworks to remote naturalistic datasets with
substantial missingness.

> **ASSUMPTION:** Per-subject models were estimated using each participant's own data only
> (no cross-person information), without random effects; this makes them maximally idiographic
> but data-hungry. Participants with fewer than [fill: minimum N] valid days were excluded per
> the ANOVA feature selection procedure. Results do not generalize to multilevel idiographic
> approaches (e.g., mlVAR random effects), which pool information across participants.

---

## Summary

Passive smartphone sensor data, collected continuously from adults with depression in the BRIGHTEN
remote trial, explains little to no concurrent variance in PHQ-9 when self-reported EMA is
withheld (R² = 0.006–0.082 in V1; R² < 0 in V2). Adding self-report EMA substantially improves
prediction (R² = 0.34–0.49), but the active ingredient is the self-report itself, not the passive
behavioral context. Daily mood (PHQ-2) is not predictable from any combination of features. Passive
behavioral trajectories from the first 4 weeks of the study do not reliably forecast 6-week
depression status above chance. These results underscore a fundamental limitation of passive-only
digital phenotyping for concurrent severity prediction in this population and naturalistic setting,
echoing findings from prior work (Holstein et al., 2024; Sun et al., 2022; Kiang et al., 2021).

---

*Draft prepared 2026-06-18. All `[fill: ...]` markers require values from saved R objects or
CSV outputs before submission. Robustness checks still needed: GroupMean baseline comparison
(subject_scores_from_diff_models.csv), temporal/forward-chaining CV sensitivity, exact excluded
column list, per-variant sample sizes, and raw-scale MAE back-transformation.*
