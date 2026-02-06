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
| `max_dose` | Maximum dose as a string (e.g., `"100uM"`, `"10mM"`) |
| `moa` | Mechanism of action |
| `dilution_factor` | Dilution series factor (e.g., `3.0` for 3x dilutions) |

## Output Structure

```
output_dir/
├── per_genotype/      # Per-genotype normalized data
├── across_all/        # Across-all-genotypes normalized data
├── per_well/          # Well-level aggregated profiles
├── crowding/          # Cell crowding/density metrics
├── qc/               # Quality control reports
├── pca/              # PCA coordinates and loadings
├── dose_response/    # Fitted dose-response curves (Hill equation)
├── statistics/       # Genotype comparison results
├── cache/            # Checkpoint data for resume
└── manifest.json     # Pipeline metadata and file inventory
```

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

### Statistical Comparisons

- **Preferred:** cluster bootstrap EC50 comparisons at the well level (stratified by plate/replicate).
- **Dose-specific effects:** permutation or rank-based tests on well-level effects with stratification when needed.
- Effect size reporting (e.g., Cliff's delta or Hedges' g) with confidence intervals.
- Multiple testing correction (FDR recommended for discovery workflows).

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
