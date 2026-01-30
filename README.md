# DNA Damage ICC Analysis Pipeline

A production-ready bioinformatics pipeline for analyzing single-cell immunocytochemistry (ICC) data from DNA damage experiments. It processes parquet files containing single-cell features (DNA damage markers, morphology, intensity measurements) from multiple genotypes treated with DNA-damaging compounds, performs comprehensive statistical analysis, and generates publication-quality results.

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
| 2 | Preprocess with per-genotype normalization |
| 3 | Preprocess with across-all-genotypes normalization |
| 4 | Compile crowding metrics (cell density measurements) |
| 5 | Compile quality control (QC) metrics per well |
| 6 | Generate per-well aggregated CSV files |
| 7 | Compute PCA (Principal Component Analysis) |
| 8 | Fit dose-response curves |
| 9 | Statistical comparisons between genotypes |
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

## Key Features Analyzed

- **DNA Damage Markers**: foci count, gamma-H2AX mean intensity
- **Proliferation**: Ki67 positivity and intensity
- **Morphology**: area, perimeter, eccentricity, solidity, circularity
- **Intensity**: DAPI, SYTO RNA mean intensity
- **Crowding**: local density, nearest-neighbor distance, neighbor counts
- **Quality Control**: cell count per well, segmentation coverage, border cells

## Analysis Components

### Dose-Response Modeling

Fits a four-parameter Hill equation to dose-response data:

```
y = bottom + (top - bottom) / (1 + (EC50 / x)^hill_slope)
```

Returns EC50, Hill slope, R-squared, and confidence intervals for each drug-genotype combination.

### Statistical Comparisons

- Z-tests for EC50 comparisons between genotypes
- Mann-Whitney U tests for response value comparisons at specific doses
- Multiple testing correction (Bonferroni and FDR)

### Normalization

Two normalization modes are supported:
- **Per-genotype**: Robust z-score normalization within each genotype separately
- **Across-all**: Normalize across all genotypes together for direct comparison

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
