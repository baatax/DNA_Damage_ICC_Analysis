# Statistical and QC Policy for DNA Damage ICC Analysis

This document defines required analysis and reporting policies to avoid pseudo-replication, normalization artifacts, and under-specified inference in DNA damage ICC workflows.

## 1) Experimental unit and model structure

### Primary experimental unit
- **Default unit of replication: well-level summaries.**
- Cells are nested within wells, and wells are nested within plate/batch/replicate groups.

### Required reporting
Every analysis report must explicitly state:
- Unit of replication used for each statistical test.
- Nesting/hierarchy in the experiment.
- Random effects/covariates included (e.g., plate, imaging session, replicate).

### Modeling guidance
- Prefer well-level models for primary inferential outputs.
- If cell-level modeling is used, use hierarchical/mixed models with random effects for well and plate/batch.
- Do not treat all cells as independent observations in genotype significance tests.

## 2) Normalization policy (goal-dependent)

Normalization must be chosen by the scientific question and declared in outputs.

### Dose-response within genotype
- Normalize relative to that genotype's own matched controls (typically DMSO), preferably per plate.

### Baseline genotype differences
- Preserve raw units when possible.
- Apply only imaging/plate correction if needed.
- Avoid genotype-wise z-scoring that removes baseline differences of interest.

### Cross-genotype drug sensitivity
- Use well-level effect sizes relative to matched controls:
  - Δ from DMSO
  - fold-change / log2 fold-change when appropriate
  - robust standardized effects
- Apply batch correction on well-level profiles if required.

## 3) QC decision framework

QC must include explicit thresholds and deterministic actions.

### Hard exclusion rules (default)
Exclude wells that satisfy any of:
- `n_cells < N_min` or `n_cells > N_max`
- `fraction_border_cells > border_max`
- `segmentation_coverage < coverage_min`
- `missing_feature_fraction > missing_max`

### Soft flags (reported, not excluded by default)
- Extreme crowding metrics.
- Intensity saturation indicators.
- Robust outlier well medians relative to plate controls.

### Required QC outputs
- `qc_well_table.csv`: pass/fail, metrics, reason list.
- `excluded_wells.csv`: well keys with reason codes.
- QC report (HTML/Markdown) with threshold overlays and summary counts.

## 4) Dose-response fitting policy

### Inputs
- Fit on **well-level** response summaries/effects.
- For dose variables: use `x = log10(dose_uM)` for `dose > 0`.
- Treat `dose = 0` controls as a separate anchor condition.

### Candidate models
- 4-parameter Hill.
- Constrained 3-parameter Hill (e.g., fixed bottom or top when justified).
- Optional monotonic fallback model when Hill is unstable.

### Robust fitting requirements
- Use parameter bounds.
- Use multiple initializations.
- Support weighting by `n_cells` or inverse well variance.
- Record convergence status and diagnostics.
- Handle/flag non-monotonic curves (toxicity/hormesis possibilities).

### Model selection/reporting
- Compare candidate models with AIC (or equivalent).
- Report chosen model type and alternatives considered.
- Prefer RMSE + residual diagnostics; do not rely only on R².

## 5) Genotype comparisons for EC50 and responses

### EC50 comparisons
- Avoid default Z-tests unless assumptions are explicitly validated.
- Preferred approach: cluster bootstrap at the well level, stratified by plate/replicate.
- Report `Δlog10(EC50)` with confidence intervals and p-values.

### Dose-specific response comparisons
- Compare well-level effects via stratified permutation tests or rank-based tests.
- Report effect sizes (e.g., Cliff's delta, Hedges' g) and confidence intervals.
- Apply multiple testing correction (FDR by default).

## 6) PCA and multivariate analysis policy

- Perform PCA on well-level profiles (raw, control-relative, and/or corrected as configured).
- Include diagnostics colored by plate/batch to assess confounding.
- Report variance explained and interpretable loadings.
- If single-cell embeddings are used, downsample or aggregate to avoid overwhelming replicate structure.

## 7) Crowding metric integration

Crowding metrics must be integrated via at least one explicit strategy:
- **QC:** exclude extreme-crowding wells.
- **Covariate adjustment:** regress out crowding effects.
- **Stratification:** report responses across crowding strata.

Reports must state which strategy was used and include supporting plots.

## 8) Reproducibility and provenance

Every run must persist:
- Resolved config snapshot (including derived parameters).
- Git commit hash (if repository present).
- Python and package versions.
- Random seed(s).
- Output manifest enumerating files and processing stages.

Caching/checkpoint outputs should include cache keys and invalidation logic tied to config/code changes.

## 9) Minimum plotting/reporting set

### QC (required)
- Cell count per well distributions with thresholds.
- Plate-layout QC pass/fail heatmaps.
- DAPI intensity sanity plots by plate.
- Segmentation proxy scatterplots.
- Crowding vs key DNA damage readout plots.
- PCA colored by plate/batch for confounding check.

### Core analysis (required)
- Dose-response plots with well points + fitted curves + CIs.
- EC50 forest plots across genotypes/drugs/features.
- `Δlog10(EC50)` comparison plots with bootstrap CIs.
- Dose-specific genotype effect plots with q-values.
- Well-level feature correlation heatmaps.

### Fit diagnostics (required)
- Residuals vs dose plots.
- Parameter plausibility distributions.
- Replicate consistency summaries (scatter and/or ICC).

## 10) Defaults checklist

Unless overridden in config:
- Unit of inference: well-level.
- Normalization: control-relative within plate × genotype × drug.
- EC50 inference: stratified cluster bootstrap.
- Multiple testing: FDR.
- Exclusion: hard QC failures removed before fitting.
- Provenance: mandatory manifest + versions + seeds.
