# Results — mlVAR Group Comparison: End-of-Study Depression Status (`end_depressed`)

*Drafted from `Figure_end_depressed2_pca_v1_day_165p_.pdf` (V1 daily, 4-node PCA model) and
`Figure_end_depressed_binary_pca_v2_day_141p_.pdf` (V2 daily, 6-node PCA model). Both figures
report group-level mlVAR temporal (lag-1) networks, raw between-group differences, and
permutation-significant differences (mlVAR_GC, `mnet` package). Values below are read directly
off the figure labels. `[fill: ...]` markers indicate values not present on the figures that
should be pulled from the underlying RDS objects before submission.*

> **Grouping note — distinct from Paper A's `stayed_depressed` analysis.** This analysis groups
> participants by **`end_depressed`**: PHQ-9 status at approximately 6 weeks only (depressed vs.
> not depressed at that single endpoint), rather than by stable status across both baseline *and*
> endpoint (`stayed_depressed`, used in `results_mlVAR.md`). The two grouping variables use
> different inclusion logic and produce different sample sizes; figures, edges, and N's from this
> document should not be merged with or compared directly to the `stayed_depressed` results
> without confirming both use the same underlying group definition.

---

## 5.1 Analytic sample

| Variant | Group | N |
|---|---|---|
| V1 daily (4-node PCA) | Not depressed at 6 weeks | 103 |
| V1 daily (4-node PCA) | Depressed at 6 weeks | 62 |
| V2 daily (6-node PCA) | Not depressed at 6 weeks | 69 |
| V2 daily (6-node PCA) | Depressed at 6 weeks | 72 |

> **ASSUMPTION:** Group sizes are read directly from the figure headers (`N =`). The V1 total
> (165) and V2 total (141) match the filenames (`..._165p_.pdf`, `..._141p_.pdf`), suggesting
> these are the full analytic samples for the `end_depressed` grouping at each variant, prior to
> any further exclusion. `[fill: confirm against the source RDS objects and report whether these
> totals reflect the full 8-week-feature-eligible sample or a further-restricted subset.]`

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

**Table 2. Differences panel — all edges, V1 daily 4-node model.**

| Edge | Δ (not-dep − dep) |
|---|---|
| phq2 (AR) | −0.06 |
| sms (AR) | −0.06 |
| calls (AR) | −0.06 |
| uncalls (AR) | −0.09 |
| uncalls → phq2 | +0.04 |
| (other cross-lags) | ≤ \|0.03\| |

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

> **ASSUMPTION:** The 4-node model (`phq2`, `sms`, `calls`, `uncalls`) is a reduced node set
> relative to the 5-node PCA model used for `stayed_depressed` (which also includes `pc_mobility`).
> Mobility does not appear in this figure; it is unclear whether it was dropped from the
> `end_depressed` PCA clustering or simply omitted from this particular plot. `[fill: confirm node
> set used for the V1 day end_depressed mlVAR_GC run.]`

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

*The `commute` and `temp` node labels visually overlap in both panels, so their individual
self-loop values cannot be reliably disambiguated from the figure alone. `[fill: extract
commute and temp AR values separately from the underlying RDS/CSV output.]*

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

> **CAUTION:** As noted in `results_mlVAR.md` §4.5, this V2 node set is dominated by mobility and
> weather-adjacent variables (`commute`, `vehicle_location`, `active_mobility`, `temp` — 4 of 6
> nodes). The `commute → temp` and `vehicle_location → commute` effects may partly reflect
> geographic/seasonal patterns in travel and weather exposure rather than purely psychological
> dynamics. The phq2 autoregression difference is the only significant edge directly implicating
> mood and should be weighted most heavily as a replication of the inertia finding.

---

## 5.4 Cross-variant comparison

| Finding | V1 daily (4-node) | V2 daily (6-node) |
|---|---|---|
| PHQ-2 inertia higher in depressed group | Yes, directionally (Δ≈−0.06 to −0.08) | **Yes, significant (Δ=−0.12, p<.05)** |
| Communication-node inertia higher in depressed group | Yes, directionally (sms, calls, uncalls all Δ≈−0.06 to −0.09) | N/A (no communication nodes in this model) |
| Significant edges | 1 (uncalls→phq2, stronger in not-depressed) | 3 (2 mobility/weather cross-lags + phq2 AR) |
| Direction of significant cross-lag effects | Stronger in not-depressed | Stronger in not-depressed (mobility/weather); phq2 AR stronger in depressed |

Across both independently-modeled variants, the only finding that replicates as **statistically
significant in at least one variant and directionally consistent in the other** is elevated PHQ-2
(depressed-mood/anhedonia) autoregression in the group depressed at 6 weeks. This is broadly
consistent with the `stayed_depressed` inertia results (`results_mlVAR.md` §4.3, §4.8), though
the `end_depressed` grouping is a different (single-timepoint) operationalization and uses
different node sets per variant, so the two analyses should be reported as separate (complementary)
results rather than as a single combined finding.

> **ASSUMPTION:** Because the V1 and V2 models use entirely different node sets (communication vs.
> mobility/weather), only `phq2` is directly comparable across variants. No claim is made here
> about whether mobility or communication inertia specifically replicates across V1 and V2, since
> the underlying variables differ.

---

## 5.5 Outstanding items before submission

- `[fill]` Confirm exact `commute` and `temp` autoregression values for V2 daily from source RDS
  (labels overlap on the figure).
- `[fill]` Confirm whether V1 daily's 4-node model omits `pc_mobility` by design or by plotting
  choice, and reconcile against the 5-node `stayed_depressed` model.
- `[fill]` Confirm the exact clinical/temporal definition of `end_depressed` used for this run
  (PHQ-9 ≥ 12 at days 38–55, per Methods §2.2 secondary outcomes) and verify it matches the N's
  reported in the figure headers (165 total V1, 141 total V2).
- `[fill]` Obtain p-values and uncorrected/Bonferroni-corrected thresholds for the 4 significant
  edges identified here (not visible on the figures themselves, which show only significance
  status via inclusion in the "Significant Differences" panel).
- `[fill]` Verify the V1/V2 weekly variants of this same `end_depressed` grouping have been run;
  not covered by the two figures provided.

---

*Figures referenced: `Figure_end_depressed2_pca_v1_day_165p_.pdf`,
`Figure_end_depressed_binary_pca_v2_day_141p_.pdf`. Values transcribed directly from on-figure
labels; no recomputation was performed.*
