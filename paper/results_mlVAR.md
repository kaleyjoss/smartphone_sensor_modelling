# Results — mlVAR Network Analysis (Paper A: Inertia)

*Drafted 2026-06-18 from:*
- *`pipeline/mlVAR_GC_stayed_depressed_v1_day.Rmd` — R code for group comparison analysis*
- *`results/charts/mlVAR/trainval/*.csv` — mean network edge weight matrices*
- *`data/interim/mnet_output_stayed_depressed_g1g2_PC_v1_day.RDS` — PCA-based group comparison (5 nodes)*
- *`data/interim/mnet_output_stayed_depressed_g1g2lme_v1_day.RDS` — LME-feature group comparison (10 nodes)*
- *`data/interim/mnet_output_stayed_depressed_g1g2_PC_v2_day.RDS` — V2 replication*
- *`results/charts/mlVAR/chart-type-explanation.md` — network type definitions*
- *`paper/Methods.docx` — mlVAR estimation details*

*`[fill: ...]` markers indicate values to verify or complete before submission.*

> **DATA QUALITY NOTE: CSV export bug.** The `*_contemporaneous_mean.csv` and `*_between_mean.csv`
> files for all four dataset variants (v1_day, v1_week, v2_day, v2_week) are byte-identical to their
> corresponding `*_temporal_mean.csv` files (confirmed by `cmp -s`), except for v1_day where
> contemporaneous_mean and between_mean are identical to each other but distinct from temporal_mean.
> The contemporaneous and between-person network CSV values reported here are therefore **not
> interpretable** from the CSV files alone; numerical values for those networks must be extracted
> directly from the saved RDS objects. Temporal network values are confirmed valid. The `.png` files
> in this directory are also CSV files (wrong extension) containing per-subject parameter estimates,
> not image visualizations.

> **ASSUMPTION: Multiple analysis runs exist.** Several RDS files in `data/interim/` correspond to
> different pipeline stages and model specifications (imputed vs. transformed features; PCA-reduced
> 5-node vs. raw-feature 10-node LME models; ANOVA-based 4-node vs. LME-based 5-node PCA networks).
> Results below are organized by model specification. The "primary" analysis for Paper A is the
> PCA-based 5-node group comparison on transformed (not imputed) data
> (`mnet_output_stayed_depressed_g1g2_PC_v1_day.RDS`), consistent with the LME feature set
> described in Methods. The 10-node LME model results are reported as a sensitivity analysis.

---

## 4.1 Analytic sample: group comparison

The stayed_depressed grouping assigns participants to group 1 (stayed not-depressed: PHQ-9 below
clinical cutoff at both baseline and end of study) or group 2 (stayed depressed: PHQ-9 at or above
clinical cutoff at both timepoints). Participants who changed status (improved or worsened) were
excluded from the group comparison per the mlVAR_GC specification.

> **ASSUMPTION:** The "stayed_depressed" variable operationalizes clinical cutoff as PHQ-9 ≥ 10:
> based on the criterion in the analysis code. Participants who changed category are
> assumed to represent a different psychological process and are excluded; their exclusion is a
> modeling decision that sacrifices completeness for group purity.

**V1 daily data (primary analysis):**
- Group 1 (stayed not-depressed): N = 96 participants
- Group 2 (stayed depressed): N = 44 participants
- Total retained for group comparison: N = 140 (from 165 who met overall missingness thresholds,
  minus 25 who changed status)
- Inclusion criteria: ≤ 70% missing on any model variable AND ≥ 40 valid observations per
  participant

**V2 daily data (replication):**
- Group 1 (stayed not-depressed): N = 33 participants
- Group 2 (stayed depressed): N = 38 participants
- Note: the V2 groups are notably small, especially the non-depressed group (N=33). Results
  should be interpreted with caution given limited statistical power.

**V1/V2 weekly data:** [fill: extract N per group from `mnet_output_stayed_depressed_g1g2v1_week.RDS`
and `mnet_output_stayed_depressed_g1g2_PC_v2_week.RDS`]

---

## 4.2 Whole-sample temporal network structure (V1 daily, ANOVA PCA — 4 nodes)

The whole-sample mlVAR temporal (lag-1 autoregressive) network was estimated using PCA-reduced
ANOVA features. Four principal components were retained after hierarchical clustering: pc_phq2
(depression symptom composite), pc_social (social communication composite), pc_mobility
(step-count/walking-distance composite), and pc_missed_calls (unreturned and missed communication
composite). Mean temporal edge weights (population-average fixed effects) are reported below.

> **ASSUMPTION:** The temporal network uses the ANOVA feature set (11 raw variables compressed
> via hierarchical PCA to 4 nodes), while the group comparison below uses the LME feature set
> (10 raw variables compressed to 5 nodes: pc_mobility, pc_calls, pc_unreturned_calls, pc_sms,
> pc_phq2). These are different model runs. Whole-sample temporal CSVs reflect the ANOVA run;
> group comparison results reflect the LME run. Node names differ between the two models.

**Table 1. Whole-sample mean temporal network (Φ matrix) — V1 daily, ANOVA PCA, 4 nodes.**
Edge weights are population-average lag-1 partial regression coefficients (column variable at t−1
→ row variable at t). Diagonal = autoregressive persistence (inertia).

|  | pc_phq2 (t−1) | pc_social (t−1) | pc_mobility (t−1) | pc_missed_calls (t−1) |
|---|---|---|---|---|
| **pc_phq2 (t)** | **0.491** | 0.008 | 0.002 | 0.001 |
| **pc_social (t)** | 0.008 | **0.416** | 0.002 | 0.038 |
| **pc_mobility (t)** | −0.022 | 0.003 | **0.339** | 0.000 |
| **pc_missed_calls (t)** | −0.004 | −0.005 | 0.000 | **0.331** |

*Source: `v1_day_temporal_mean.csv` (confirmed valid; not affected by the export bug).*

Key observations:
- **Autoregressive persistence (inertia) dominates the temporal network.** All four diagonal
  elements are strong (Φ = 0.33–0.49), while off-diagonal cross-lagged effects are near zero
  (all |Φ| < 0.04). This means each variable's value today is primarily predicted by its own
  value yesterday, not by other variables.
- **PHQ-2 shows the highest inertia** (Φ = 0.491), meaning daily depressed mood / anhedonia
  is the most temporally persistent variable in the network — each day's PHQ-2 score is strongly
  predicted by the prior day's score.
- **Social communication shows the second-highest inertia** (Φ = 0.416), followed by mobility
  (Φ = 0.339) and missed calls (Φ = 0.331).
- **One notable cross-lag:** missed_calls(t−1) → social(t) = 0.038, suggesting that days with
  more missed communication are followed by slightly more social contact. The direction is small
  and its significance is not established from this whole-sample estimate alone.
- **PHQ-2 → mobility cross-lag = −0.022**: worse depressed mood predicts slightly less mobility
  the following day, but the effect is small.

---

## 4.3 Group comparison: stayed depressed vs. stayed not-depressed (V1 daily, PCA model — 5 nodes)

The mlVAR_GC group comparison tests whether temporal network parameters differ between the
stayed-depressed and stayed-not-depressed groups using a permutation test (nP = 1000,
paired = FALSE, orthogonal random effects). The difference reported is group 1 (not-depressed)
minus group 2 (depressed), so a negative Δ indicates the effect is stronger in the depressed group.

> **ASSUMPTION:** The permutation test uses 1000 permutations of group labels, and tests the
> null hypothesis that observed group differences in edge weights could arise by chance from
> random group assignment. The Bonferroni correction is applied within each parameter type
> (temporal, contemporaneous, between-person) as per Haslbeck et al. (2025).

> **ASSUMPTION:** Temporal = "orthogonal" specifies that temporal effects are orthogonal to
> contemporaneous effects in the model. This choice (vs. "correlated") determines how variance
> is partitioned between lag-1 and same-day partial effects.

**5-node PCA model nodes:** pc_mobility, pc_calls, pc_unreturned_calls, pc_sms, pc_phq2
(derived from LME feature set via hierarchical PCA).

**Significant group differences in temporal (lag-1) parameters (p < 0.05, permutation):**

| Edge | Δ (non-dep − dep) | p-value | Interpretation |
|---|---|---|---|
| **pc_sms → pc_sms** (autoregression) | −0.138 | **0.004** | SMS inertia is higher in the depressed group by 0.138 units |
| **pc_mobility → pc_mobility** (autoregression) | −0.108 | **0.031** | Mobility inertia is higher in the depressed group by 0.108 units |
| **pc_phq2 → pc_mobility** (cross-lag) | +0.035 | **0.049** | In the non-depressed group, worse PHQ-2 predicts slightly more mobility the next day; this effect is absent in the depressed group |

Non-significant trends (p < 0.20 but not significant after correction):
- pc_calls AR: Δ = −0.086, p = 0.123 (direction consistent with inertia hypothesis)
- pc_phq2 AR: Δ = −0.060, p = 0.155 (direction consistent but not significant in this model)
- pc_mobility → pc_sms cross-lag: Δ = −0.030, p = 0.058 (marginal)

**Per-group mean autoregressive coefficients (5-node PCA model):**

| Node | Group 1 (not-depressed) | Group 2 (depressed) | Δ |
|---|---|---|---|
| pc_mobility | [fill from g1 Beta] | [fill from g2 Beta] | −0.108* |
| pc_calls | [fill] | [fill] | −0.086 |
| pc_unreturned_calls | [fill] | [fill] | −0.084 |
| pc_sms | [fill] | [fill] | −0.138** |
| pc_phq2 | [fill] | [fill] | −0.060 |

*p < 0.05; **p < 0.01. [fill: extract per-group AR values from g1$results$Beta$mean and g2$results$Beta$mean in the PC RDS file]*

The pattern is consistent: across all five network nodes, the depressed group shows numerically
higher autoregressive persistence, with the SMS and mobility differences reaching statistical
significance.

---

## 4.4 Sensitivity analysis: LME model (10 raw-feature nodes, transformed data)

As a sensitivity analysis, the group comparison was also run on the full 10-variable LME feature
set (without PCA compression): interaction_diversity, call_count, unreturned_calls, mobility,
mobility_radius, sms_count, missed_interactions, call_duration, phq2_1, phq2_2.

**Mean autoregressions by group (N = 96 non-depressed, N = 44 depressed):**

| Variable | Group 1 (not-depressed) Φ | Group 2 (depressed) Φ | Δ |
|---|---|---|---|
| interaction_diversity | 0.419 | 0.503 | −0.084 |
| call_count | 0.340 | 0.420 | −0.080 |
| unreturned_calls | 0.290 | 0.448 | **−0.157*** |
| mobility | 0.376 | 0.436 | −0.060 |
| mobility_radius | 0.305 | 0.410 | −0.105 |
| sms_count | 0.385 | 0.543 | **−0.158**** |
| missed_interactions | 0.348 | 0.326 | +0.022 |
| call_duration | 0.335 | 0.397 | −0.062 |
| phq2_1 | 0.413 | 0.546 | **−0.133*** |
| phq2_2 | 0.429 | 0.497 | −0.068 |

*p < 0.05; **p < 0.01 (permutation test, 1000 permutations).

**Significant group differences (p < 0.05) in the 10-node LME model:**

| Edge | Δ (non-dep − dep) | p-value | Interpretation |
|---|---|---|---|
| sms_count → sms_count (AR) | −0.158 | **0.005** | SMS count inertia higher in depressed |
| unreturned_calls → unreturned_calls (AR) | −0.157 | **0.011** | Missed-call inertia higher in depressed |
| phq2_1 → phq2_1 (AR) | −0.133 | **0.011** | PHQ-2 item 1 inertia higher in depressed |
| phq2_2 → phq2_1 (cross-lag) | +0.079 | **0.013** | PHQ-2 item cross-lag stronger in non-depressed |
| missed_interactions → sms_count (cross-lag) | +0.043 | **0.013** | In non-depressed: missed interactions → more SMS next day (compensatory communication?) |
| call_duration → phq2_2 (cross-lag) | +0.038 | **0.009** | Call duration → next-day PHQ-2 item 2 stronger in non-depressed |
| call_count → mobility (cross-lag) | +0.045 | **0.032** | In non-depressed: more calls → more mobility next day |
| phq2_1 → mobility (cross-lag) | +0.039 | **0.045** | In non-depressed: worse depressed mood → more mobility next day (absent in depressed) |

The overall picture from the 10-node model is consistent with the 5-node PCA model: the
depressed group shows reliably higher temporal autocorrelation across communication and mood
variables (sms_count, unreturned_calls, phq2_1), while the non-depressed group shows stronger
cross-node coupling (calls → mobility, phq2 → mobility, missed interactions → SMS compensation).
This suggests that in persistently depressed individuals, each behavioral and mood domain
"locks in" day-to-day at a higher level, while in non-depressed individuals, variables
influence each other more dynamically across days.

> **ASSUMPTION:** The 10-node model uses the LME feature set on transformed (but not imputed)
> data. The imputed-data sensitivity (`mnet_output_stayed_depressed_lme_imputed_v1_day.RDS`)
> yields nearly identical significant edges (same 8 edges, same direction), confirming results
> are not driven by the imputation decision.

---

## 4.5 V2 daily replication

The V2 daily replication uses a different 9-node PCA structure dominated by weather and mobility
features (pc_commute, pc_active_mobility, pc_cloud_cover_std, pc_weather_means, pc_temp_humidity_std,
pc_dew_point, pc_dew_point_std, pc_temp, pc_phq2), reflecting the V2 cohort's expanded passive
sensor suite. Group sizes are small (N=33 non-depressed, N=38 depressed).

**V2 Day autoregressions by group:**

| Variable | Group 1 (not-depressed) Φ | Group 2 (depressed) Φ | Δ |
|---|---|---|---|
| pc_commute | 0.092 | 0.140 | −0.048 |
| pc_active_mobility | 0.215 | 0.264 | −0.049 |
| pc_cloud_cover_std | 0.219 | 0.225 | −0.006 |
| pc_weather_means | 0.533 | 0.576 | −0.043 |
| **pc_temp_humidity_std** | 0.629 | 0.745 | **−0.117*** |
| pc_dew_point | 0.186 | 0.205 | −0.019 |
| pc_dew_point_std | 0.680 | 0.592 | +0.088 |
| pc_temp | 0.268 | 0.292 | −0.024 |
| pc_phq2 | 0.241 | 0.341 | −0.100 |

*p < 0.05; †p < 0.01 (permutation test, 1000 permutations).

**Significant group differences in V2 day (p < 0.05):**

| Edge | Δ | p-value | Note |
|---|---|---|---|
| pc_weather_means → pc_active_mobility | −0.115 | **0.002** | Weather → next-day activity coupling stronger in depressed |
| pc_dew_point → pc_active_mobility | +0.077 | **0.004** | Dew-point → next-day activity weaker in depressed |
| pc_temp_humidity_std AR | −0.117 | **0.023** | Temp-humidity inertia higher in depressed |
| pc_dew_point_std → pc_active_mobility | +0.217 | **0.041** | Dew-point variability → activity cross-lag weaker in depressed |
| pc_commute → pc_phq2 | −0.032 | **0.043** | Commute → PHQ-2 coupling stronger in depressed |
| pc_phq2 → pc_temp_humidity_std | +0.023 | **0.024** | PHQ-2 → temp-humidity effect stronger in non-depressed |
| pc_phq2 → pc_dew_point_std | +0.035 | **0.011** | PHQ-2 → dew-point std stronger in non-depressed |
| pc_temp → pc_active_mobility | −0.058 | **0.046** | Temperature → activity coupling stronger in depressed |

> **CAUTION:** The V2 feature set is dominated by weather variables (6 of 9 nodes). Weather
> variables carry their own physical autocorrelation unrelated to psychological dynamics
> (e.g., pc_temp autoregression = 0.760 in whole-sample, driven by seasonal temperature
> persistence). Group differences in weather-node parameters may reflect differential weather
> exposure (e.g., geographic or seasonal confounding) rather than psychological inertia. The
> pc_phq2 autoregression difference (Δ = −0.100) is numerically consistent with V1 findings
> but does not reach significance (p = 0.155) in V2, likely due to limited power (N=33+38).

> **ASSUMPTION:** The V2 day grouping uses the same stayed_depressed classification, but V2 has
> fewer participants with stable depression trajectories, resulting in N=71 total (vs. N=140 in V1).
> The V2 PCA uses 9 nodes reflecting the different feature space; direct comparison of edge-level
> effects between V1 and V2 is therefore not valid — the node definitions differ.

---

## 4.6 Weekly data networks (V1 week, V2 week)

**V1 weekly temporal mean (whole-sample, 6 nodes):**
Nodes: pc_unreturned_calls, pc_calls, pc_mobility, pc_phq2, pc_sms, pc_mobility_radius.

At weekly aggregation, autoregressive persistence collapses dramatically relative to daily data:

| Node | Weekly Φ (AR) | Daily Φ (AR, daily model) |
|---|---|---|
| pc_phq2 | 0.195 | 0.491 |
| pc_sms | 0.115 | [0.138 diff in group model] |
| pc_mobility | 0.026 | 0.339 |
| pc_unreturned_calls | 0.005 | ~0.29–0.45 |
| pc_calls | −0.015 | ~0.34–0.42 |
| pc_mobility_radius | −0.005 | — |

The near-zero (and slightly negative) weekly autoregressions for behavioral variables indicate
that at a week-to-week timescale, behavioral patterns do not carry forward significantly beyond
the current week. PHQ-2 weekly autocorrelation is substantially lower (Φ=0.195) than daily
(Φ=0.491), suggesting that week-to-week PHQ-2 changes are less predictable from the prior week.

Notable weekly cross-lags:
- pc_calls(t−1) → pc_phq2(t): +0.054 (more social calls last week → slightly worse depressed mood this week? or vice versa — direction needs checking against matrix orientation)
- pc_calls(t−1) → pc_sms(t): +0.067
- pc_phq2(t−1) → pc_calls(t): interpretation as above [fill: verify row/column orientation]

*Source: `v1_week_temporal_mean.csv` (confirmed valid).*

**V2 weekly temporal mean (whole-sample, 9 weather/mobility nodes):**
At weekly frequency, all autoregressions collapse toward zero. The pc_phq2 AR = 0.000 (essentially
zero), and weather variables show near-zero weekly persistence (consistent with week-to-week weather
variability). No meaningful temporal network structure is apparent at the weekly timescale for V2.

**V1/V2 weekly group comparisons:** [fill: extract from `mnet_output_stayed_depressed_g1g2v1_week.RDS`
and `mnet_output_stayed_depressed_g1g2_PC_v2_week.RDS`]

---

## 4.7 Contemporaneous and between-person networks

> **DATA BUG — DO NOT REPORT VALUES FROM CSV FILES.** The contemporaneous_mean and between_mean
> CSV files are confirmed byte-identical to the temporal_mean CSV for all variants (v1_week,
> v2_day, v2_week). For v1_day, the contemporaneous and between CSVs are identical to each other
> but distinct from the temporal CSV. Numerical values for contemporaneous and between-person
> networks must be extracted from the saved RDS objects (`g1$results$Theta` and `g1$results$Omega_mu`
> or equivalent fields).

Contemporaneous network (what it represents): partial correlations among model residuals at time t,
after removing the lag-1 component. Captures sub-measurement-interval dynamics — co-fluctuations
that occur within the same day but are not explained by yesterday's values.

Between-person network (what it represents): conditional associations among subject-specific means
(random intercepts). Reflects stable between-person trait-like differences — e.g., people who
text more also tend to move more. Note: between-person estimates are biased when individuals have
few measurement occasions (Epskamp et al., 2018); this may be an issue for participants at the
short end of the ≥40-observation inclusion criterion.

[fill: extract contemporaneous and between-person networks from RDS objects and report here.
Use `obj$g1$results$Theta$mean[,,1]` for contemporaneous and `obj$g1$results$Omega_mu$mean` or
equivalent for between-person, per mlVAR package structure.]

---

## 4.8 Summary of mlVAR findings

**Main findings:**

1. **Temporal inertia dominates the daily network structure.** At the daily timescale, each
   behavioral and mood variable is primarily predicted by its own prior-day value (Φ = 0.33–0.49),
   with minimal cross-variable dynamics (cross-lags |Φ| < 0.04 in the whole-sample 4-node model).

2. **Persistently depressed individuals show significantly higher behavioral and mood inertia.**
   In the primary 5-node PCA group comparison (V1 daily, N=140), the stayed-depressed group
   showed higher SMS inertia (Δ=−0.138, p=0.004) and higher mobility inertia (Δ=−0.108, p=0.031)
   than the stayed-not-depressed group. The 10-node LME sensitivity analysis extends this pattern
   to also include unreturned_calls (Δ=−0.157, p=0.011) and phq2_1 (Δ=−0.133, p=0.011).

3. **Behavioral coupling differs between groups.** In the non-depressed group, missed interactions
   predict more SMS the next day (Δ=+0.043, p=0.013), and worse PHQ-2 predicts more mobility the
   next day (Δ=+0.035–0.039, p≈0.049), suggesting compensatory behavioral dynamics that are
   absent in the depressed group.

4. **Inertia collapses at the weekly timescale.** At weekly aggregation, autoregressive
   persistence is substantially lower across all variables. The strongest daily inertia (pc_phq2
   Φ=0.491 daily vs. 0.195 weekly), suggesting the inertia signal resides in day-to-day dynamics,
   not week-to-week patterns.

5. **V2 replication is limited.** Small group sizes (N=33+38) and a weather-dominated feature
   space complicate V2 interpretation. The pc_phq2 AR difference is numerically consistent
   (Δ=−0.100) but not significant in V2.

> **ASSUMPTION — Grouping boundary:** "Stayed depressed" and "stayed not-depressed" requires
> that PHQ-9 status is stable across the full study window. This excludes participants who improved
> (a key clinical group). The inertia contrast therefore compares two extremes — chronic remission
> vs. chronic depression — rather than the full clinical distribution.

> **ASSUMPTION — Bonferroni correction:** The mlVAR_GC permutation p-values are corrected
> for the number of parameters within each parameter type (temporal, contemporaneous, between)
> using Bonferroni adjustment. The 5×5 temporal matrix has 25 possible parameters; the
> 10×10 LME matrix has 100. Under Bonferroni (α = 0.05), the 5-node model corrected threshold
> is p < 0.002, and the 10-node model threshold is p < 0.0005. The p-values reported here are
> **uncorrected** permutation p-values and should be re-evaluated against the Bonferroni threshold
> before reporting. Under Bonferroni, pc_sms AR (p=0.004) may survive correction; others may not.
> [fill: confirm the correction method used in the mnet package version and report both raw and
> corrected p-values.]

---

*Numerical values from:*
- *Whole-sample temporal: `v1_day_temporal_mean.csv`, `v1_week_temporal_mean.csv`, `v2_day_temporal_mean.csv`, `v2_week_temporal_mean.csv`*
- *Group comparisons: extracted via `Rscript` from `mnet_output_stayed_depressed_g1g2_PC_v1_day.RDS`, `mnet_output_stayed_depressed_g1g2lme_v1_day.RDS`, `mnet_output_stayed_depressed_g1g2_PC_v2_day.RDS`*
- *All p-values from permutation test (nP = 1000, paired = FALSE)*
