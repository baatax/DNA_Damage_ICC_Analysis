# DNA Damage ICC Analysis Pipeline

A production-ready bioinformatics pipeline for analyzing single-cell immunocytochemistry (ICC) data from DNA damage experiments. It processes parquet files containing single-cell features (DNA damage markers, morphology, intensity measurements) from multiple genotypes treated with DNA-damaging compounds, performs comprehensive statistical analysis, and generates publication-quality results.

> **Statistical policy update (important):** Use well-level summaries as the primary experimental unit. Cells are nested within wells, and wells are nested within plate/batch/replicate groups. Treating single cells as independent replicates can inflate significance. See [Statistical and QC Policy](ANALYSIS_STATISTICAL_POLICY.md) for required modeling and reporting practices.

## Overview

The pipeline is designed for studying how different genetic backgrounds (genotypes) respond to DNA-damaging compounds through dose-response relationships and cellular phenotyping. It supports multi-genotype, multi-drug experimental designs with automatic normalization, quality control, and statistical comparison.

## Project Structure

```
DNA_Damage_ICC_Analysis/
├── run_dna_damage_pipeline.py          # Main pipeline orchestrator
├── dna_damage_parquet_pipeline.py      # Core analysis components
├── dose_response_analysis.py           # Dose-response curve fitting & statistics
├── demo_dna_damage_pipeline.py         # Examples with synthetic data
├── example_dna_damage_config.json      # Template configuration
├── 2Cohort_TZ_config.json              # Example real-world config (Talazoparib)
└── submit_DDP_pipeline.slurm           # SLURM cluster job submission script
```

## Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- scikit-learn
- pyarrow

## Quick Start

### 1. Create a configuration file

Copy and edit the template configuration:

```bash
cp example_dna_damage_config.json my_experiment_config.json
```

The JSON config defines your experiment metadata, genotypes, drugs, data file paths, and expected EC50 values. See [Configuration](#configuration) for details.

### 2. Run the pipeline

```bash
python run_dna_damage_pipeline.py my_experiment_config.json \
    --output ./results \
    --workers 8 \
    --log-level INFO
```

### 3. Run on a SLURM cluster

Edit `submit_DDP_pipeline.slurm` to point to your config file and environment, then:

```bash
sbatch submit_DDP_pipeline.slurm
```

The pipeline supports checkpoint/resume so interrupted jobs can continue where they left off by adding the `--resume` flag.


### Create a small local test dataset from the 2-cohort Talazoparib config

Use this helper to sample each genotype/drug parquet into small representative test files
while preserving all columns and representation across available experimental groups
(`Dilut`, `dilut_string`, `group`, `treatment`, `is_control`).

```bash
python create_test_dataset_from_config.py \
  --source-config 2Cohort_TZ_config.json \
  --output-config 2Cohort_TZ_test_config.json \
  --output-data-dir test_data/2cohort_tz
```

Then run locally against the generated test config:

```bash
python run_dna_damage_pipeline.py 2Cohort_TZ_test_config.json --output ./test_output_2cohort_tz
```

Or submit test-data extraction on SLURM:

```bash
sbatch submit_generate_test_dataset.slurm
```

## Pipeline Steps

The pipeline executes 10 sequential analysis steps:

| Step | Description |
|------|-------------|
| 1 | Load parquet data from all genotypes and drugs |
| 2 | Validate schema and create canonical sample keys |
| 3 | Compute explicit cell-level/well-level QC and exclusion logs |
| 4 | Compile crowding metrics (cell density measurements) |
| 5 | Compile quality control (QC) metrics per well |
| 6 | Generate per-well aggregated CSV files |
| 7 | Compute well-level feature/effect profiles and PCA diagnostics |
| 8 | Fit robust dose-response candidate models |
| 9 | Statistical comparisons between genotypes with resampling |
| 10 | Save manifest with metadata |

## Configuration

The JSON configuration file has two main sections:

### `metadata`

| Field | Description |
|-------|-------------|
| `experiment_name` | Name for the experiment |
| `control_label` | Label for vehicle control (e.g., `"DMSO"`) |
| `dilut_column` | Column name containing dilution/concentration info |
| `output_dir` | Default output directory |

### `genotypes`

Each genotype contains a `description` and a `drugs` map. Each drug entry specifies:

| Field | Description |
|-------|-------------|
| `path` | Path to the parquet file with single-cell features |
| `ec50_um` | Expected EC50 in micromolar |
| `max_dose` | Maximum/top dose as a string (e.g., `"100uM"`, `"10mM"`). When set, the pipeline relabels non-control dose levels into a pseudo-serial dilution anchored at this value for downstream analyses. |
| `moa` | Mechanism of action |
| `dilution_factor` | Dilution series factor (e.g., `3.0` for 3x dilutions) |

### Pseudo dilution relabeling (for mislabeled raw doses)

If raw dose labels are inconsistent, set per-drug `max_dose` (and optionally `dilution_factor`, default `3.0`) in the config. During loading, the pipeline will:

1. Keep the original labels in `raw_dilut_string`.
2. Rank non-control dose levels by parsed concentration.
3. Reassign concentrations as `max_dose / dilution_factor^rank`.
4. Write corrected labels to `dilut_string` and corrected numeric values to `dilut_um`.

Controls keep the configured `control_label` and `dilut_um = 0`.

## Output Structure

```
output_dir/
├── crowding_analysis/             # Crowding tables + crowding plots
├── crowding_corrected_analysis/   # Full corrected analysis (tables/plots/models)
├── uncorrected_analysis/          # Full uncorrected analysis (tables/plots/models)
├── crowding/                      # Internal crowding computation artifacts
├── uncorrected/                   # Internal uncorrected artifacts
├── crowding_corrected/            # Internal corrected artifacts
├── cache/                         # Checkpoint data for resume
└── manifest.json                  # Pipeline metadata and file inventory
```

QC tables also include `tables/qc_exclusion_report.md`, a human-readable list of excluded wells/samples with fail reasons.

### PCA + dose-response visualization behavior

- Every PCA context now emits:
  - PC1/PC2 scatter plots,
  - per-PC loading bar plots,
  - **top-3 loading feature dose-response curves for each PC**.
- Dose-response panels are drawn against `dilut_um` and include `0` concentration (DMSO/vehicle) as the lowest point in the dilution series.
- EC50-focused and DMSO-focused PCA outputs both include these top-feature dose-response panels so DMSO controls are directly integrated with concentration series interpretation.
- For **EC50-focused PCA**, top-3 loading feature dose-response panels are now plotted from the full variant concentration series (all tested doses) while still deriving feature selection from EC50-focused PCA loadings.

## Experimental Unit and Statistical Design

- **Primary replication unit:** well-level summary values.
- **Nesting structure:** cells → well → plate/batch/imaging session.
- **Recommended random effects:** plate/batch and replicate in any inferential model.
- **Cell-level modeling:** optional and must be hierarchical/mixed-effects; do not use naive single-cell significance tests.

## Key Features Analyzed

- **DNA Damage Markers**: foci count, gamma-H2AX mean intensity
- **Proliferation**: Ki67 positivity and intensity
- **Morphology**: area, perimeter, eccentricity, solidity, circularity
- **Intensity**: DAPI, SYTO RNA mean intensity
- **Crowding**: local density, nearest-neighbor distance, neighbor counts
- **Quality Control**: cell count per well, segmentation coverage, border cells

## Analysis Components

### Dose-Response Modeling

Fits dose-response candidate models on **well-level effect sizes**:

- 4-parameter Hill
- constrained 3-parameter Hill
- optional monotonic fallback for non-monotonic/noisy curves

Core fitting policy:

- Use `log10(dose_uM)` for `dose > 0`; treat control (`dose = 0`) as a separate anchor.
- Apply explicit parameter bounds and multiple initializations.
- Use configurable weighting (e.g., by `n_cells` or inverse well variance).
- Record convergence status and diagnostics for every curve.

Canonical Hill equation:

```
y = bottom + (top - bottom) / (1 + (EC50 / x)^hill_slope)
```

Returns EC50, Hill slope, confidence intervals, RMSE, AIC, model choice, and convergence diagnostics for each drug-genotype combination.

Dose-response fit tables additionally include:
- `max_tested_dose_um`: highest non-control concentration observed per fit group.
- `ec50_pct_max`: EC50 represented as percentage of that max tested concentration.

All loaded datasets are validated against a 9-point dilution ladder anchored to each drug's configured `max_dose`
(default `100uM`) and `dilution_factor` (default `3.0`) plus DMSO control.
When raw labels are inconsistent, loader-side relabeling is applied using those same per-drug settings and then revalidated.

### Statistical Comparisons

- **Preferred:** cluster bootstrap EC50 comparisons at the well level (stratified by plate/replicate).
- **Dose-specific effects:** permutation or rank-based tests on well-level effects with stratification when needed.
- Effect size reporting (e.g., Cliff's delta or Hedges' g) with confidence intervals.
- Multiple testing correction (FDR recommended for discovery workflows).
- **EC50-neighborhood validation:** for each genotype/drug group, the pipeline now performs genotype comparisons at the nearest lower, nearest EC50-matched, and nearest higher tested concentrations; if EC50 is already at the highest tested concentration, the higher comparison uses the EC50 dose itself.
- **Combined 3-dose EC50-band analysis:** the same lower/EC50/higher concentration band is emitted as separate analysis groups for additional comparison outputs.

### Normalization

Normalization is goal-dependent and should be explicit in all reports:

- **Control-relative (recommended default):** compute Δ / fold-change relative to matched DMSO controls per plate × genotype × drug.
- **Plate correction:** remove plate-specific baseline shifts using control wells.
- **Across-plate correction (optional):** batch correction on well-level profiles only.

Avoid genotype-wise z-scoring when baseline genotype differences are part of the biological question.

## QC Requirements

- Define hard fail thresholds (cell count, border fraction, segmentation coverage, missing-feature fraction).
- Keep soft warning metrics (crowding extremes, intensity saturation, robust outlier scores).
- Emit decision tables and exclusion logs with reason codes.
- Exclude failed wells from downstream fitting by default.

Detailed threshold examples and reporting templates are provided in [Statistical and QC Policy](ANALYSIS_STATISTICAL_POLICY.md).

## Demo / Testing

Generate synthetic data and run example analyses:

```bash
# Run all examples
python demo_dna_damage_pipeline.py all

# Run a specific example
python demo_dna_damage_pipeline.py basic
python demo_dna_damage_pipeline.py dose-response
python demo_dna_damage_pipeline.py genotype
python demo_dna_damage_pipeline.py crowding
python demo_dna_damage_pipeline.py qc
```

For reproducible package setup and the canonical test command that uses repository test fixtures, see [TESTING_ENVIRONMENT.md](TESTING_ENVIRONMENT.md).

### Plot output notes

- PCA scatter axes include explained-variance percentages when variance metadata is available.
- PCA legends include per-group sample counts (`n=...`) for grouped PCA scatter plots.
- PCA loading plots are emitted for both **PC1** and **PC2**.
- Loading bars are channel-colored (Ki67 red, H2Ax orange, Sytogreen green, DAPI blue).
- Dose-response summaries/fold-change plots now show individual well-level dots alongside summary trends.
- Dose-response fit outputs include both EC50 (µM) and log10(EC50 [µM]) visualizations.
- Dose-response fit outputs also include EC50 as **% of max tested concentration** plus **log10(% max)** visualizations.
